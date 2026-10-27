#!/usr/bin/env python3
"""Directly learnable velocity trained under exactly two L2 terms.

Implements the velocity-field action plan:

    L = L2(|RA|_rendered - |RA|_measured) + lambda * L2(MD_rendered - MD_measured)

- The velocity itself is the parameter — initialized from the camera-derived
  joint velocities, unbounded, no scales, no rotation, no squashing.
- Two parameterizations: ``joint`` (learnable (600,14,3) joint velocity
  field, skinned to points) and ``point-normal`` (camera-skinned baseline
  plus one learnable scalar per point per frame along the point's normal).
- Co-trained under the RA term: the shared human material and the per-point
  normals of the training frames. Antenna patterns, array geometry, point
  positions, skinning weights, background, and the fitted absolute receiver
  gains stay fixed.
- Nothing else enters the loss. Held-out frames and temporal discrimination
  are evaluated and reported, never trained on. ``two_term_loss`` is the
  entire objective and is unit-tested to be exactly the sum of its two
  parts; ``result.json`` re-verifies the identity on every logged step.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

from mm4DPS import Dataset, config_centre_m, current_dataset, dataset_from_argv, per_radar, radar_configs
from mm4DPS import DATASET_ROOT, MM3DPS_ROOT

import torch

from mm4DPS import peak_vram

from mm4DPS.data import (
    MeasuredReader,
    range_angle_slow_time_cube_frames_torch,
    range_profiles_all_channels_torch,
    range_bin_spacing_m,
    range_keep_mask,
)
from mm4DPS.signal_model import (
    TDM_LOOPS,
    assemble_static_profile,
    fft_window_kind,
    windowed_process_envelope,
    capture_summary,
    load_trajectories,
    official_motion_doppler_mask,
    person_range_angle_mask_torch,
    soft_joint_weights,
    soft_joint_weights_torch,
    surface_bistatic_path_rates,
    surface_bistatic_path_rates_per_point)
from mm4DPS import HUPR_NEAR_FIELD_M, array_axis
from mm4DPS.forward_render import doppler_planes_to_motion_magnitude, profiles_to_ra_magnitude_batched, profiles_to_ra_magnitude_fused, profiles_to_ra_spectrum_fused, range_profiles_to_ra_magnitude, slow_time_profiles_to_ra_complex, slow_time_ra_to_doppler_magnitude, profiles_to_rae_magnitude, profiles_to_rae_spectrum_fused, profiles_to_rae_spectrum_reference, rae_azimuth_rolled, elevation_floor_factor, slow_time_to_doppler
from mm4DPS.train_motion import (
    choose_motion_frames,
    microdoppler_marginal,
    microdoppler_marginal_batched)
from mm4DPS.train_material import SharedHumanMaterial
from mm4DPS.signal_model import (
    LearnedNoiseFloor,
    diffuse_floor_expected_cube,
    measure_diffuse_floor_expectation,
    scene_range_support,
    support_to_range_amplitudes)
from mm4DPS.scene import (
    person_solid_angle_weights, scene_solid_angle_weights, smpl_faces,
    assert_persistent_person_geometry,
    person_areas,
    surface_solid_angle_weights,
    surface_solid_angle_weights_torch)
from mm4DPS.forward_render import build_rasterizers

ROOT = Path(__file__).resolve().parent
# The dataset's radars, in order. Resolved from $MM4DPS_DATASET at import
# (the chain exports it to every stage) and re-resolved from --dataset when
# train() parses its arguments.
RADARS = current_dataset().names


# HUPR_FUSED_LOSS=0 routes the batched step's loss chain through the
# reference transforms (`profiles_to_ra_magnitude_batched` + per-step
# denominators) instead of the fused ones, so the two can be A/B'd in one
# build — the same escape hatch HUPR_DOPPLER_CHUNKED provides for the
# chunked K2b backward.
_FUSED_LOSS = os.environ.get("HUPR_FUSED_LOSS", "1") != "0"
# The scene's backgrounds for the batch from one fused kernel pair (the
# contributions synthesised inside; `assemble_static_profile_frames_cuda`)
# instead of the per-frame contribution chain. HUPR_FUSED_SCENE=0 restores
# the per-frame chain (the A/B; fp32 floor apart).
_FUSED_SCENE = os.environ.get("HUPR_FUSED_SCENE", "1") != "0"
# The person's position-only splat geometry cached per flat row for the
# captured step (see `build_row_geometry`). HUPR_ROW_GEOMETRY=0 re-renders
# it from the positions every step (the A/B; bitwise).
_ROW_GEOMETRY = os.environ.get("HUPR_ROW_GEOMETRY", "1") != "0"
# The sorted Doppler route for the captured step: the person's paths in
# (unit, channel, range bin) order (cached per frame in the row geometry),
# the contributions synthesised in the kernels, no binning per step
# (`forward_render._DopplerPlaneSorted`). Needs the row geometry, the
# closed form with taps > 0, the Hann window and `--deterministic off`
# (its shared-memory sums are order-free, as the default binning's are).
# HUPR_SORTED_DOPPLER=0 keeps the slot route (fp32 floor apart).
_SORTED_DOPPLER = os.environ.get("HUPR_SORTED_DOPPLER", "1") != "0"

# The RA plane term's post-FFT chain — |spectrum|, subtract, square, reduce —
# is ~10 separate passes over ~94 MB tensors (forward and backward) that
# torch.compile fuses into two kernels. Measured standalone at the step's
# real shapes: 3.07 -> 1.06 ms forward+backward, matching the eager chain at
# 2.1e-07 (values) / 1.5e-07 (gradients), and it captures into the training
# step's CUDA graph (compilation happens during the capture warm-up, before
# the capture itself). HUPR_COMPILED_PLANE_TERM=0 falls back to the eager
# chain. One compile worker is enough for this one small function, and it
# sidesteps a worker-pool hang observed once against a stale inductor cache.
_COMPILED_PLANE_TERM = (
    _FUSED_LOSS and os.environ.get("HUPR_COMPILED_PLANE_TERM", "1") != "0")
# The elevation term's route (`rae_terms_of`): "fused" -- both RAE views from one
# compiled gather and two FFTs on a contiguous buffer
# (`forward_render.profiles_to_rae_spectrum_fused`) instead of the index_add
# scatter, the weight multiply, the movedim, the two n=-padded FFTs and the two
# rolls; "reference" -- that chain (`profiles_to_rae_magnitude`), the A/B.
# MM4DPS_RAE_CHECK=1 compares the route's per-unit values and its gradients
# against "reference" cell by cell at start-up, which the 20-step loss dump
# cannot resolve below ~1e-9: a sum over millions of cells hides a one-ulp cell
# (docs/RUNTIME_RESULTS.md item 2).
# HUPR_RAE_ROUTE unset = chosen per dataset by measurement (`_rae_route_default`), since the
# two routes are bitwise each other and the faster one differs between the arrays.
_RAE_ROUTE_ENV = os.environ.get("HUPR_RAE_ROUTE")
# The plane term's pointwise chain in one kernel pass (docs/RUNTIME_STEP_20MS_PLAN.md A1):
# |.| -> square -> + floor * factor -> clamp_min -> sqrt -> target denormalisation -> subtract
# -> square, which eager runs as ~10 passes over 42.3 M cells a step. BITWISE -- the kernel uses
# round-to-nearest intrinsics so nvcc cannot contract a multiply-add into a single-rounding FMA,
# takes the magnitude with hypotf (PyTorch's complex abs is hypot, not sqrt(re^2+im^2)), and
# reproduces autograd's operation ORDER in the backward rather than its algebra. Measured on the
# real shape: 1.85 ms against 11.05 ms, every cell and every gradient identical.
# HUPR_RAE_TERM=eager restores the op-by-op chain; it is the A/B and what MM4DPS_RAE_CHECK
# compares against.
_RAE_TERM = os.environ.get("HUPR_RAE_TERM", "fused")
# The RAE loss's azimuth fftshift, dropped from BOTH sides (docs/RUNTIME_STEP_20MS_PLAN.md item B).
# Both elevation terms are a uniform mean over the flattened (range, angle, elevation) cells, and
# the shift is a cyclic permutation of those cells, so a permutation cannot change the mean -- nor
# the denominators, which are computed from the same targets, nor the floor, which is uniform in
# angle. The stored targets are un-shifted once at the cache build and the prediction's roll is
# skipped: 1.48 ms a step on the cascade, where the shift is a real roll over 646 MiB forward and
# the same again in the backward. Exact in real arithmetic; only the fp32 summation ORDER moves, so
# this is gated against a float64 reference rather than bitwise. HUPR_RAE_AZ_SHIFT=keep restores it.
_RAE_AZ_SHIFT_SKIP = os.environ.get("HUPR_RAE_AZ_SHIFT", "skip") == "skip"
_RAE_CHECK = os.environ.get("MM4DPS_RAE_CHECK") == "1"


def _rae_route_default(dataset_name: str) -> str:
    """Which RAE route is faster, measured. Both arrays now prefer the fused one.

    The routes are bitwise each other, so this is free to choose and the choice is measurement.
    It used to be dataset-dependent: HuPR preferred the fused route by 1.78 ms while the cascade
    preferred the reference chain by 1.55 ms, because the fused route's slot-table gather
    materialised a (units, planes, 602, 2, rows) intermediate -- 565 MiB a step, 78% of its slots
    empty -- which cost more on the cascade's 6.08 GiB store than the plumbing it replaced.

    Item C removed that intermediate (`forward_render._RaeSpectrumInput` builds the transform's
    input straight from the channels), and the cascade's preference flipped: RT-Pose 176 now
    measures 31.19 ms on the fused route against 35.26 on the reference chain, and HuPR 256 18.19
    against 19.83 (docs/RUNTIME_STEP_20MS_PLAN.md 11). So the default is the same for both, and
    this function stays only to record that it was a measurement rather than an assumption.
    """
    return "fused"


class _RaePlaneTermDifference(torch.autograd.Function):
    """The plane RAE term's per-cell DIFFERENCE (prediction - target), the pointwise chain in one pass.

    Forward returns ``pred - target`` per cell of the (units, planes, rows, angle, elevation)
    store. The caller squares it -- once for the global term, and once AFTER masking for the
    windowed one, which is the order main uses and the only order whose gradient chain matches it.
    Returning the square instead agrees in value but reassociates the backward at 1e-7, measured.
    The reduction is likewise left to PyTorch: a fused reduction would change the summation order
    and could not be bitwise.

    ``floor_row`` is the learned noise power per (unit, plane, row) BEFORE the factor, so that
    the factor multiply happens inside the kernel where eager does it; its gradient is returned
    as the per-cell ``grad_s`` reduced by PyTorch here, again to keep the order autograd uses.
    """

    @staticmethod
    def forward(ctx, spectrum, floor_row, target, rms, factor):
        from mm4DPS.cuda_ext import load_hupr_cuda
        ext = load_hupr_cuda()
        out = ext.hupr_rae_plane_term_forward(spectrum, floor_row, target, rms, factor, 1e-30)
        ctx.save_for_backward(spectrum, floor_row, target, rms)
        ctx.factor = factor
        return out

    @staticmethod
    def backward(ctx, grad_out):
        spectrum, floor_row, target, rms = ctx.saved_tensors
        from mm4DPS.cuda_ext import load_hupr_cuda
        ext = load_hupr_cuda()
        grad_spec, grad_s = ext.hupr_rae_plane_term_backward(
            grad_out.contiguous(), spectrum, floor_row, target, rms, ctx.factor, 1e-30)
        grad_floor = None
        if ctx.needs_input_grad[1]:
            # exactly autograd's own two steps: sum over the broadcast axes, then the factor
            grad_floor = grad_s.sum(dim=(-1, -2)).reshape(-1) * ctx.factor
        return (grad_spec if ctx.needs_input_grad[0] else None), grad_floor, None, None, None
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")



def _plane_term_sums_eager(spectrum: torch.Tensor,
                           target: torch.Tensor,
                           floor: torch.Tensor,
                           noise_power: torch.Tensor | None = None) -> torch.Tensor:
    """Per-unit sum of (sqrt(|spectrum|^2 + noise) - target)^2; `spectrum` is view_as_real.

    ``noise_power`` is the learned noise floor's expected power at every
    supervised cell, ``(units, planes, range, 1)`` broadcast over angle, or
    None. Power adds incoherently, so the floor enters under the square
    root exactly as it does for the micro-Doppler marginal -- the way the
    legacy single-frame reference path always had it. Without it the plane
    term compared a floor-less render to a floor-bearing target, and the
    learned floor took its gradient from the person's rows alone (the MD
    marginal), leaving its range shape unconstrained everywhere else:
    measured, its DCT range correction ran to the +/-3 clamp (x20) during
    the cosine descent, with a 24x floor in one run of two (seq 183).

    ``sqrt(re^2 + im^2)`` is the complex magnitude written in real ops so the
    whole expression stays inside one inductor graph; the reference chain's
    ``.abs()`` computes the same value (they differ only at the fp32
    rounding floor, gated by verify_fused_loss's probe A/B).

    The ``clamp_min`` is the zero-safe gradient, not a value change: a plane
    cell no path lands in is *exactly* zero, where ``sqrt``'s derivative is
    0/0 = NaN (measured: it NaN'd the whole run on step one). ``torch.abs``
    defines ``sgn(0) = 0`` there; clamping the squared magnitude at 1e-38
    (the fp32 normal floor, ~1e-19 in amplitude against radar magnitudes of
    1e3+) reproduces exactly that zero gradient, and the value ``sqrt(1e-38)``
    is below fp32 resolution of every target it is subtracted from.
    """
    power = spectrum[..., 0].square() + spectrum[..., 1].square()
    if noise_power is not None:
        power = power + noise_power
    # The guard AFTER the floor is added, so a pad cell (spectrum and floor
    # both exactly zero) takes the guard's value and the gradient into the
    # floor is masked there too; before it, sqrt's gradient at 0 reached the
    # floor as inf and inf * 0 = NaN. The guard is 1e-30, a normal float32
    # (1e-38 is a denormal, which the compiled kernels flush to zero).
    magnitude = torch.sqrt(torch.maximum(power, floor))
    return (magnitude - target).square().flatten(1).sum(dim=-1)


_plane_term_sums = (torch.compile(_plane_term_sums_eager, dynamic=False)
                    if _COMPILED_PLANE_TERM else _plane_term_sums_eager)

# The absolute receiver gain is a per-radar CONSTANT (a required CLI
# argument), yet the batched paths multiplied it into the full complex plane
# cube every step -- two ~189 MB passes forward and the same backward. Folded
# instead into the render's per-point gain (squared, because the `areas` slot
# enters the amplitude under a square root) and into the tiny background
# profiles before K2, the rendered profiles come out already scaled and those
# multiplies disappear. Same physics, same objective; the products associate
# differently, so values move at the fp32 floor -- gated by the same probe
# A/B as the fused chain. HUPR_FOLDED_ABSOLUTE_GAIN=0 restores the
# multiply-at-use form. The legacy per-frame path keeps that form always: it
# is the pinned reference.
_FOLDED_ABSOLUTE_GAIN = os.environ.get(
    "HUPR_FOLDED_ABSOLUTE_GAIN", "1") != "0"

# The epoch's short last batch (590 = 24 x 24 + 14 at frame-batch 24) falls
# back to the eager dynamic step. HUPR_STATIC_SHORT_STEP=1 routes it through
# the captured step instead, padded to frame_batch with repeats of the
# smallest frame and multiplied by a per-unit {0,1} mask before the loss sum
# (exact: a padded unit contributes 0.0 to the loss and every gradient; a
# real unit's values are multiplied by 1.0, bitwise). Measured at fb 24 it
# is a REGRESSION and stays off: the masked full-width step costs the full
# 42 ms while the 14-frame eager step costs ~25 -- fewer units beat fewer
# launches here.
_STATIC_SHORT_STEP = os.environ.get("HUPR_STATIC_SHORT_STEP", "0") == "1"

# The route gate's dump: the first N steps' per-unit terms and gradient norms
# of epoch 0 (MM4DPS_PRINT_STEPS=N; MM4DPS_PRINT_FIRST_STEP=1 is the old
# four-step form). A change the plan calls bitwise is accepted when every
# line equals the reference's (docs/RUNTIME_RESULTS.md).
_PRINT_STEPS = int(os.environ.get(
    "MM4DPS_PRINT_STEPS", "4" if os.environ.get("MM4DPS_PRINT_FIRST_STEP") == "1" else "0"))


def _md_marginal_prediction_eager(
    cube: torch.Tensor, ra_mask: torch.Tensor, range_rows: torch.Tensor,
    noise_floor: torch.Tensor,
) -> torch.Tensor:
    """``microdoppler_marginal_batched`` + floor + sqrt, in one graph.

    Exactly the expression every MD consumer runs today --
    ``(marginal.clamp_min(0) + floor).sqrt()`` over the masked power sums --
    written once so torch.compile fuses its ~7 passes over the Doppler cube
    into a reduction. Used by the training step and the evaluation alike.
    """
    power = cube.square() * ra_mask[:, None]
    marginal = (power.sum(dim=-1) * range_rows[:, None, :]).sum(dim=-1)
    return (marginal.clamp_min(0.0) + noise_floor).sqrt()


_md_marginal_prediction = (
    torch.compile(_md_marginal_prediction_eager, dynamic=False)
    if _COMPILED_PLANE_TERM else _md_marginal_prediction_eager)


def _normalized_l2_units_eager(
    prediction: torch.Tensor, target: torch.Tensor, mean_square: torch.Tensor,
) -> torch.Tensor:
    """``mean((pred - target)^2, per unit) / mean_square``, in one graph.

    The same expression the batched loss has always run, written once so
    torch.compile fuses it. Eagerly it is a subtract, a square and a reduction
    -- three passes over a (units, kept range, angle) tensor, ~750k floats at
    the step's shapes -- and the intermediates are pure bandwidth: the
    arithmetic per byte is negligible, so collapsing them into one reduction
    is worth roughly what the extra traffic costs.
    """
    return (prediction - target).square().flatten(1).mean(dim=-1) / mean_square


_normalized_l2_units = (
    torch.compile(_normalized_l2_units_eager, dynamic=False)
    if _COMPILED_PLANE_TERM else _normalized_l2_units_eager)


# Set from --ablate-loss-term at train() start; module-level so the pinned
# single-frame reference and the batched forms stay in agreement. A set, because
# a consolidation drops two terms at once (cons_both = loop0-ra + plane-ra).
_ABLATE_TERMS: set[str] = set()


def two_term_loss(
    ra_prediction: torch.Tensor,
    ra_target: torch.Tensor,
    md_prediction: torch.Tensor,
    md_target: torch.Tensor,
    md_weight: float,
    ra_loss: str = "l2",
    ra_plane_prediction: torch.Tensor | None = None,
    ra_plane_target: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The complete objective: L2 on |RA| plus weighted L2 on the MD column.

    Returns (total, ra_l2, md_l2) with total = ra_l2 + md_weight * md_l2 and
    nothing else. Both terms are means of squared differences of linear
    magnitudes, each divided by the *measured target's* mean square — a
    fixed, detached constant per frame and radar. The normalization keeps
    the term plain L2 (a constant factor cannot change the minimizer or add
    scale invariance: the prediction's own level is never in the
    denominator, so absolute amplitude stays supervised) while making the
    H/V radars and the two terms commensurate, so ``md_weight`` is a pure
    interplay knob rather than a unit fixer.

    ``ra_plane_*`` extends the RA term from loop 0 to **all 64 chirps**. The
    loop->Doppler DFT is unitary, so each of the 63 non-DC planes is a linear
    combination of every chirp in the frame and every measured chirp therefore
    contributes gradient. It is a second normalized L2 on linear magnitudes --
    the same form as the loop-0 term, not a new kind of objective -- and it
    carries its own divisor because a plane is a coherent sum over 64 loops and
    so sits at a different level from a single loop; sharing one divisor would
    let units alone decide which of the two dominates.
    """
    if ra_prediction.shape != ra_target.shape:
        raise ValueError("RA prediction and target must share a shape")
    if md_prediction.shape != md_target.shape:
        raise ValueError("MD prediction and target must share a shape")
    if "loop0-ra" in _ABLATE_TERMS:
        ra_l2 = torch.zeros((), device=ra_prediction.device)
    elif ra_loss == "pearson":
        # 1 - Pearson on linear magnitudes: correlation is invariant to each
        # map's affine normalization, so this is the "correlation RA loss on
        # normalized linear maps" — it optimizes the reported RA statistic
        # directly. Level is then supervised only through the MD term.
        a = ra_prediction.reshape(-1)
        b = ra_target.reshape(-1)
        a = a - a.mean()
        b = b - b.mean()
        denominator = torch.sqrt(
            a.square().sum() * b.square().sum()
        ).clamp_min(1e-20)
        ra_l2 = 1.0 - (a * b).sum() / denominator
    else:
        ra_l2 = (ra_prediction - ra_target).square().mean() / (
            ra_target.detach().square().mean().clamp_min(1e-30)
        )
    if ra_plane_prediction is not None and "plane-ra" not in _ABLATE_TERMS:
        ra_l2 = ra_l2 + (
            (ra_plane_prediction - ra_plane_target).square().mean()
            / ra_plane_target.detach().square().mean().clamp_min(1e-30)
        )
    md_l2 = (md_prediction - md_target).square().mean() / (
        md_target.detach().square().mean().clamp_min(1e-30)
    )
    return ra_l2 + md_weight * md_l2, ra_l2, md_l2


def two_term_loss_units(
    ra_prediction: torch.Tensor,
    ra_target: torch.Tensor,
    md_prediction: torch.Tensor,
    md_target: torch.Tensor,
    md_weight: float,
    ra_loss: str = "l2",
    ra_plane_prediction: torch.Tensor | None = None,
    ra_plane_target: torch.Tensor | None = None,
    ra_target_mean_square: torch.Tensor | None = None,
    ra_plane_target_mean_square: torch.Tensor | None = None,
    md_target_mean_square: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``two_term_loss`` with a leading unit axis, term for term.

    Not a new objective: the same two expressions, reduced over each
    unit's own elements instead of over one unit's. Pinned against
    ``two_term_loss`` per unit in tests/gates/test_runtime_refactor_equivalence.

    The ``*_mean_square`` arguments are the terms' denominators —
    ``target.detach().square().flatten(1).mean(-1).clamp_min(1e-30)`` — for
    callers that hold fixed targets and can compute each denominator once
    instead of re-reducing a constant every step. The plane target alone is
    ~47 MB per step, re-squared and re-reduced for a value that never
    changes. Omitted, the denominator is computed here exactly as before.
    """
    if ra_prediction.shape != ra_target.shape:
        raise ValueError("RA prediction and target must share a shape")
    if md_prediction.shape != md_target.shape:
        raise ValueError("MD prediction and target must share a shape")
    if "loop0-ra" in _ABLATE_TERMS:
        ra_l2 = torch.zeros(
            ra_prediction.shape[0], device=ra_prediction.device)
    elif ra_loss == "pearson":
        a = ra_prediction.flatten(1)
        b = ra_target.flatten(1)
        a = a - a.mean(dim=-1, keepdim=True)
        b = b - b.mean(dim=-1, keepdim=True)
        denominator = torch.sqrt(
            a.square().sum(dim=-1) * b.square().sum(dim=-1)
        ).clamp_min(1e-20)
        ra_l2 = 1.0 - (a * b).sum(dim=-1) / denominator
    else:
        if ra_target_mean_square is None:
            ra_target_mean_square = ra_target.detach().square().flatten(
                1).mean(dim=-1).clamp_min(1e-30)
        ra_l2 = _normalized_l2_units(
            ra_prediction, ra_target, ra_target_mean_square)
    if ra_plane_prediction is not None and "plane-ra" not in _ABLATE_TERMS:
        if ra_plane_target_mean_square is None:
            ra_plane_target_mean_square = ra_plane_target.detach().square(
            ).flatten(1).mean(dim=-1).clamp_min(1e-30)
        ra_l2 = ra_l2 + _normalized_l2_units(
            ra_plane_prediction, ra_plane_target, ra_plane_target_mean_square)
    if md_target_mean_square is None:
        md_target_mean_square = md_target.detach().square().mean(
            dim=-1).clamp_min(1e-30)
    md_l2 = _normalized_l2_units(
        md_prediction, md_target, md_target_mean_square)
    return ra_l2 + md_weight * md_l2, ra_l2, md_l2


def radiometric_person_gain(
    target_powers: list[float],
    predicted_powers: list[float],
    pedestal_powers: list[float],
) -> float:
    """Closed-form per-radar person amplitude calibration.

    The analogue of the fitted absolute receiver gain, for the person level:
    one scalar per radar such that rendered person marginal power matches
    the measured person-attributable power (measured minus its noise/clutter
    pedestal) over the training frames. Median across frames for
    robustness; fitted once after training, stored in the checkpoint,
    applied at render — no guarded refit stage.
    """
    if not (len(target_powers) == len(predicted_powers) == len(pedestal_powers)):
        raise ValueError("per-frame power lists must share a length")
    if not target_powers:
        raise ValueError("at least one frame is required")
    ratios = []
    for target, predicted, pedestal in zip(
        target_powers, predicted_powers, pedestal_powers,
    ):
        person = max(target - pedestal, 1e-30)
        ratios.append(person / max(predicted, 1e-30))
    return float(np.sqrt(np.median(ratios)))


class _FusedRowAdam:
    """Adam over the rows of one fused parameter that a step actually touched.

    E8/P1 replaces 590 per-frame ``raw_normals`` parameters with one flat
    tensor. ``torch.optim.Adam`` cannot be handed that tensor directly and
    still reproduce the training run: with 590 separate parameters it skips
    every one whose ``grad`` is ``None`` — 584 of them on any given step — so
    their moments neither decay nor take a step, and their bias-correction
    counters advance once per epoch rather than once per step. A fused
    parameter has a gradient on every row (zero on the untouched ones), and
    Adam with a zero gradient still moves a parameter whose moments are
    non-zero from earlier steps.

    This reproduces the skip exactly. The arithmetic is `_multi_tensor_adam`'s
    non-capturable branch, op for op — ``lerp_``, ``mul_``/``addcmul_``,
    ``sqrt``/``div_``/``add_``, ``addcdiv_`` — applied to the gathered rows,
    with the step counter carried per frame on the host (no device round trip:
    which frames a step touches is host knowledge).
    """

    def __init__(self, parameter: torch.nn.Parameter, lr: float,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8) -> None:
        self.parameter = parameter
        self.lr = float(lr)
        self.beta1, self.beta2 = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.exp_avg = torch.zeros_like(parameter)
        self.exp_avg_sq = torch.zeros_like(parameter)
        self.counts: dict[int, int] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        if set_to_none or self.parameter.grad is None:
            self.parameter.grad = None
        else:
            # Under CUDA-graph capture the gradient buffer must keep its
            # address across steps, so it is zeroed in place instead. Either
            # form starts the step at exactly 0.0.
            self.parameter.grad.zero_()

    def step(self, groups: list[tuple[int, torch.Tensor]]) -> None:
        """``groups`` pairs each touched key with the rows it owns.

        Keys are frames; every frame in a batch has been stepped the same
        number of times (each training frame appears in exactly one batch per
        epoch), so in practice this is one group per step — but the grouping is
        what makes the per-key counter correct rather than assumed.
        """
        gradient = self.parameter.grad
        if gradient is None:
            return
        by_count: dict[int, list[torch.Tensor]] = {}
        for key, rows in groups:
            by_count.setdefault(self.counts.get(int(key), 0), []).append(rows)
        for previous, row_list in by_count.items():
            rows = row_list[0] if len(row_list) == 1 else torch.cat(row_list)
            grad = gradient.index_select(0, rows)
            exp_avg = self.exp_avg.index_select(0, rows)
            exp_avg_sq = self.exp_avg_sq.index_select(0, rows)
            exp_avg.lerp_(grad, 1 - self.beta1)
            exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
            count = previous + 1
            bias_correction1 = 1 - self.beta1 ** count
            bias_correction2 = 1 - self.beta2 ** count
            step_size = (self.lr / bias_correction1) * -1
            denominator = exp_avg_sq.sqrt()
            denominator.div_(bias_correction2 ** 0.5).add_(self.eps)
            values = self.parameter.data.index_select(0, rows)
            values.addcdiv_(exp_avg, denominator, value=step_size)
            self.parameter.data.index_copy_(0, rows, values)
            self.exp_avg.index_copy_(0, rows, exp_avg)
            self.exp_avg_sq.index_copy_(0, rows, exp_avg_sq)
        for key, _ in groups:
            self.counts[int(key)] = self.counts.get(int(key), 0) + 1


class PerPointHumanMaterial(torch.nn.Module):
    """One 3DPS material vector per person point, as `PointPrimitives` has.

    `SharedHumanMaterial` broadcasts a single `(6,)` vector to every point of
    every frame. 3DPS trains `raw_materials (N, 6)` -- six per point. The
    difference is not a modelling preference: a per-point material needs the
    points to *persist*, and the camera point cloud rebuilds itself every frame
    (1,764 points in frame 0, 1,781 in frame 1, no correspondence), so a
    per-point parameter there would be seen by exactly one frame and could not
    be learned. This class is therefore only usable on a geometry with a fixed
    global point set -- GVHMR's 6,890 SMPL vertices with a per-frame visibility
    mask, or a constant-length cloud -- and the caller checks that.

    Initialized by broadcasting the shared vector the person stage fitted, so
    the run starts exactly where the shared-material run starts and the
    configuration space strictly contains it.
    """

    def __init__(self, initial_raw: torch.Tensor, points: int,
                 initial_area_scale: float = 0.02) -> None:
        super().__init__()
        if initial_raw.shape != (6,):
            raise ValueError(f"expected six raw material parameters, got {initial_raw.shape}")
        self.raw = torch.nn.Parameter(
            initial_raw.detach().clone().unsqueeze(0).expand(points, -1).contiguous()
        )
        self.log_area_scale = torch.nn.Parameter(
            torch.tensor(float(np.log(initial_area_scale)), dtype=initial_raw.dtype,
                         device=initial_raw.device)
        )
        self.radar_log_gain_delta = torch.nn.Parameter(
            torch.zeros((), dtype=initial_raw.dtype, device=initial_raw.device)
        )

    def for_indices(self, index: torch.Tensor) -> torch.Tensor:
        return self.raw.index_select(0, index)

    def areas_for_points(self, count: int) -> torch.Tensor:
        return self.log_area_scale.exp().expand(count)

    def radar_gain(self, radar_name: str) -> torch.Tensor:
        # The two-radar balance: the dataset's first radar exp(+delta), its
        # second exp(-delta); a single radar has no balance and gets 1.
        names = current_dataset().names
        if radar_name not in names:
            raise ValueError(f"unknown radar name: {radar_name}")
        if len(names) == 1:
            return torch.ones_like(self.radar_log_gain_delta)
        sign = 1.0 if names.index(radar_name) == 0 else -1.0
        return (sign * self.radar_log_gain_delta).exp()


class VelocityFieldModel(torch.nn.Module):
    """v is the parameter. No residuals, no bounds, no squashing.

    ``joint``: the full (frames,14,3) joint velocity field is a plain
    learnable tensor initialized from the camera velocities.
    ``point-normal``: the joint field stays frozen at the camera values and
    one learnable scalar per point per training frame moves that point along
    its (also learnable) unit normal.
    """

    def __init__(
        self,
        camera_joint_velocities: torch.Tensor,
        parameterization: str,
        frame_normals: dict[int, torch.Tensor],
        temporal_bands: int = 0,
        residual_bound: float = 0.0,
        residual_relative: float = 0.0,
        residual_per_height: float = 0.0,
        pelvis_height: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if parameterization not in ("joint", "point-normal", "temporal-basis"):
            raise ValueError(f"unknown parameterization: {parameterization}")
        self.parameterization = parameterization
        field = torch.as_tensor(camera_joint_velocities, dtype=torch.float32).clone()
        # --velocity-residual-bound b > 0 (temporal-basis only): the field is the mesh's kinematic velocity per joint, scaled
        # by a learned per-joint factor, plus a residual the DCT coefficients carry through b * tanh(. / b) -- bounded by b
        # in every component. The mesh's velocities are the carrier (a standing body stays standing: its kinematics are
        # 0.02-0.06 m/s and the free field put its hips at 0.7-0.9 m/s, docs/STATIC_SEQS_NEXT.md 1c), the scale absorbs the
        # mesh's under-estimation on moving joints (GVHMR + savgol smoothing: 1.4-1.6x on the walking sequences), and the
        # residual holds what the mesh cannot see (sway, breathing, tremor) within a physical bound.
        self.residual_bound = float(residual_bound)
        # --velocity-residual-relative beta: the bound grows with the joint's own kinematic speed, b + beta |v_kin| -- the
        # mesh's velocity error is a constant (what it cannot see) plus a share of the motion it smooths (savgol at 10 Hz
        # attenuates the gait peaks; the learned field ran 1.4-1.6x the kinematics on the walking sequences)
        self.register_buffer("kinematic_field", field.clone())
        self.residual_relative = float(residual_relative)
        # --velocity-residual-per-height gamma: the bound also grows with the pelvis's height above the support (m/s per m):
        # postural sway is an inverted pendulum about the feet, its velocity proportional to the height of the mass it
        # carries; a seated trunk (pelvis ~0.5 m) sways a fraction of a standing one (~1 m), a lying one not at all
        self.residual_per_height = float(residual_per_height)
        self.register_buffer("pelvis_height", (torch.zeros(field.shape[0]) if pelvis_height is None else torch.as_tensor(pelvis_height, dtype=torch.float32)).clamp_min(0.0))
        self.bounded = self.residual_bound > 0 or self.residual_relative > 0 or self.residual_per_height > 0
        self.joint_scale = torch.nn.Parameter(torch.ones(field.shape[1]), requires_grad=self.bounded)
        if field.ndim != 3 or field.shape[1:] != (14, 3):
            raise ValueError("camera joint velocities must have shape (frames,14,3)")
        if parameterization == "temporal-basis":
            # v_j(t) = sum_k c_{j,k} B_k(t): the whole trajectory lives in an
            # orthonormal DCT basis capped at `temporal_bands`, so every
            # frame's observation constrains shared coefficients (no
            # per-frame underdetermination) and anything above the cap
            # frequency — detection jitter — is unrepresentable by
            # construction. Initialized by projecting the camera velocities
            # onto the basis (a band-limited version of the camera init).
            frames_count = field.shape[0]
            if not 1 <= temporal_bands <= frames_count:
                raise ValueError("temporal bands must lie in [1, frames]")
            grid = torch.arange(frames_count, dtype=torch.float32)
            bands = torch.arange(temporal_bands, dtype=torch.float32)
            basis = torch.cos(
                math.pi * (grid[:, None] + 0.5) * bands[None, :] / frames_count
            ) * math.sqrt(2.0 / frames_count)
            basis[:, 0] = math.sqrt(1.0 / frames_count)
            self.register_buffer("temporal_basis", basis)
            coefficients = torch.einsum(
                "tk,tjc->kjc", basis, field,
            )
            if self.bounded:
                coefficients = torch.zeros_like(coefficients)          # the residual starts at zero: the field IS the kinematics
            self.temporal_coefficients = torch.nn.Parameter(coefficients)
            self.joint_velocity_field = torch.nn.Parameter(
                field, requires_grad=False,
            )
        else:
            self.joint_velocity_field = torch.nn.Parameter(
                field, requires_grad=(parameterization == "joint"),
            )
        # E8/P1: one flat (total points, 3) parameter instead of 590 per-frame
        # tensors. The rows are the per-frame normals concatenated in sorted
        # frame order, so `normals(frame, ...)` slices out bitwise what the
        # per-frame parameter held, and a whole batch's normals come out of one
        # `index_select` instead of six slices and a `cat`.
        #
        # Adam's semantics are NOT preserved by fusing alone and this is the
        # whole difficulty of the item: with 590 separate parameters the 584
        # that are untouched on a step have `grad is None` and Adam skips them
        # entirely, so their moments do not decay; with one fused parameter
        # every row carries a gradient (zero for the untouched rows) and Adam
        # with a zero gradient still moves the parameter. The fused tensor is
        # therefore stepped by `_FusedRowAdam` below, which applies torch's own
        # Adam arithmetic to exactly the rows that were touched and carries a
        # per-frame step counter — reproducing the skip rather than assuming it
        # away. It is deliberately not handed to the trainer's `torch.optim.Adam`.
        self.normal_frames = sorted(int(frame) for frame in frame_normals)
        self.normal_span: dict[int, tuple[int, int]] = {}
        rows = []
        offset = 0
        for frame in self.normal_frames:
            value = torch.as_tensor(frame_normals[frame], dtype=torch.float32).clone()
            self.normal_span[frame] = (offset, value.shape[0])
            offset += value.shape[0]
            rows.append(value)
        self.raw_normals_flat = torch.nn.Parameter(
            torch.cat(rows) if rows else torch.zeros(0, 3, dtype=torch.float32)
        )
        self._frame_index_cache: dict[int, torch.Tensor] = {}
        if parameterization == "point-normal":
            self.point_scalars = torch.nn.ParameterDict({
                str(frame): torch.nn.Parameter(torch.zeros(len(normals)))
                for frame, normals in frame_normals.items()
            })

    def _compose(self, basis_rows: torch.Tensor, kinematic_rows: torch.Tensor, height_rows: torch.Tensor | None = None) -> torch.Tensor:
        """temporal-basis: the field from the coefficients over `basis_rows` (..., K); bounded residual on the kinematics
        under --velocity-residual-bound / -relative / -per-height, the coefficients' field itself otherwise."""
        raw = torch.einsum("...k,kjc->...jc", basis_rows, self.temporal_coefficients)
        if not self.bounded:
            return raw
        b = self.residual_bound + self.residual_relative * torch.linalg.vector_norm(kinematic_rows, dim=-1, keepdim=True)
        if self.residual_per_height > 0 and height_rows is not None:
            b = b + self.residual_per_height * height_rows[..., None, None]
        b = b.clamp_min(1e-6)
        return self.joint_scale[:, None] * kinematic_rows + b * torch.tanh(raw / b)     # b -> 0: no residual (a still joint stays still)

    def joint_velocities_for(self, frame_ids: torch.Tensor) -> torch.Tensor:
        """The (units,14,3) joint velocities of a list of source frames.

        `point_velocities` builds one frame's (14,3) matrix at a time; this is
        the same expression with a leading unit axis, so a whole batch's
        matrices come out of one call. Used by the fixed-shape training step
        (P2), which cannot loop over frames on the host.
        """
        if self.parameterization == "temporal-basis":
            return self._compose(self.temporal_basis.index_select(0, frame_ids), self.kinematic_field.index_select(0, frame_ids), self.pelvis_height.index_select(0, frame_ids))
        return self.joint_velocity_field.index_select(0, frame_ids)

    def full_field(self) -> torch.Tensor:
        """The (frames,14,3) joint velocity field the renderer consumes."""
        if self.parameterization == "temporal-basis":
            return self._compose(self.temporal_basis, self.kinematic_field, self.pelvis_height)
        return self.joint_velocity_field

    def normals(self, frame: int, stored: torch.Tensor) -> torch.Tensor:
        """Unit normals for a frame: learnable if trained, stored otherwise."""
        span = self.normal_span.get(int(frame))
        if span is None:
            return stored
        start, count = span
        raw = self.raw_normals_flat[start:start + count]
        return raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True).clamp_min(1e-8)

    def frame_row_index(self, frame: int) -> torch.Tensor:
        """The rows of `raw_normals_flat` that belong to one training frame."""
        cached = self._frame_index_cache.get(int(frame))
        if cached is None:
            start, count = self.normal_span[int(frame)]
            cached = torch.arange(
                start, start + count, device=self.raw_normals_flat.device,
            )
            self._frame_index_cache[int(frame)] = cached
        return cached

    def has_span(self, frame: int) -> bool:
        """Whether this frame's normals are trained rather than stored."""
        return int(frame) in self.normal_span

    def normals_for_units(self, frames, stored) -> torch.Tensor:
        """Normals for a batch of frames, concatenated in order.

        E8 fused the per-unit slice-norm-clamp-divide into one `index_select`
        and, in doing so, silently dropped `normals()`'s fallback: a frame with
        no span used to fall back to its STORED normals, and the fused path
        subscripts `normal_span` and raises instead (f3b4720, 2026-08-17). That
        only ever fires in the held-out-split mode -- production passes
        `--train-all-frames`, which gives every frame a span -- so the mode has
        been broken since, silently, with a `KeyError` deep in the step.

        The fused path is kept for the case it was written for, where every
        frame is trained. Where it is not, this mirrors `normals()` frame by
        frame, which is what the fallback means.
        """
        if all(self.has_span(frame) for frame in frames):
            return self.normals_for_rows(torch.cat(
                [self.frame_row_index(frame) for frame in frames]))
        return torch.cat([self.normals(frame, rows)
                          for frame, rows in zip(frames, stored)])

    def normals_for_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """Unit normals for an arbitrary row selection of the flat parameter.

        The per-row arithmetic is `normals()`'s, so a batch of frames gathered
        in one `index_select` normalizes to bitwise what six separate slices
        normalized to — only the number of launches changes.
        """
        raw = self.raw_normals_flat.index_select(0, rows)
        return raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True).clamp_min(1e-8)

    def learned_normals(self) -> dict[str, torch.Tensor]:
        """Per-frame unit normals, keyed as the checkpoint has always held them."""
        return {
            str(frame): self.normals(frame, None).detach().cpu()
            for frame in self.normal_frames
        }

    def point_velocities(
        self,
        frame: int,
        skinning_weights: torch.Tensor,
        stored_normals: torch.Tensor,
        velocity_source_frame: int | None = None,
    ) -> torch.Tensor:
        """Per-point velocity in m/s for one frame.

        ``velocity_source_frame`` swaps the joint-level velocity for the
        temporal-discrimination evaluation; the per-point normal term (which
        belongs to this frame's surface) is kept.
        """
        source = frame if velocity_source_frame is None else velocity_source_frame
        if self.parameterization == "temporal-basis":
            joint_velocity = self._compose(self.temporal_basis[source], self.kinematic_field[source], self.pelvis_height[source])
        else:
            joint_velocity = self.joint_velocity_field[source]
        velocity = skinning_weights @ joint_velocity
        if self.parameterization == "point-normal":
            scalars = self.point_scalars.get(str(frame))
            if scalars is not None:
                velocity = velocity + scalars[:, None] * self.normals(
                    frame, stored_normals,
                )
        return velocity


def train(args: argparse.Namespace) -> dict:
    """`_train` under this trainer's array window (--array-window): set for the process while it trains and restored
    after, so the stages that follow in the same process (the render, the evaluation) keep the pipeline's window."""
    from mm4DPS.signal_model import ARRAY_WINDOW_ENV
    previous = os.environ.get(ARRAY_WINDOW_ENV)
    os.environ[ARRAY_WINDOW_ENV] = getattr(args, "array_window", "hann")
    try:
        return _train(args)
    finally:
        if previous is None:
            os.environ.pop(ARRAY_WINDOW_ENV, None)
        else:
            os.environ[ARRAY_WINDOW_ENV] = previous


def _train(args: argparse.Namespace) -> dict:
    global RADARS
    # THE TRAINING SUB-ARRAY. Every loss below goes through the beamformer,
    # which keeps the azimuth TX and drops the elevated ones, so this process
    # resolves every radar as its training subset (Radar.training_subset):
    # HuPR's boards render 2 of 3 TX, the cascade 9 of 12, with the kept TX
    # on their true loop slots. The measured side is decoded with the FULL
    # arrays (dataset.full()); the products are rendered by the renderer,
    # a separate process, with the full arrays. `--training-array full` is
    # the ablation: every TX rendered, the elevated ones discarded by the
    # beamformer as before (HuPR seq 256, 10 epochs: the same first-step loss
    # to five figures, 41.5 against 28.4 ms a step).
    # the loss's angle-FFT size, both sides (docs/FIDELITY_FIX_RESULTS.md 3.4): the radar's
    # training_n_angle unless --n-angle says otherwise; the product's size is interpolation
    from mm4DPS import N_ANGLE_OVERRIDE_ENV
    _raw = Dataset.of(args).full().radars[0]
    os.environ[N_ANGLE_OVERRIDE_ENV] = str(int(args.n_angle if args.n_angle else _raw.training_n_angle))
    training = Dataset.select(Dataset.of(args).name,
                              training_array=(args.training_array == "azimuth"))
    print(f"angle FFT: {training.radars[0].n_angle} bins in training (the product's "
          f"{training.full().radars[0].n_angle})", flush=True)
    RADARS = training.names
    if training.training_array:
        print("training sub-array: " + ", ".join(
            f"{r.name} {r.n_tx} of {training.full().radar(r.name).n_tx} TX (slots {list(r.tx_slots)})"
            for r in training.radars), flush=True)
    else:
        print("training array: the full recorded arrays (" + ", ".join(
            f"{r.name} {r.n_tx} TX" for r in training.radars) + ")", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("velocity-field training requires CUDA")
    if os.environ.get("MM4DPS_ANOMALY") == "1":
        torch.autograd.set_detect_anomaly(True)      # debugging aid: names the op that made a NaN
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from mm4DPS.rasterizer import reparameterize_torch
    from mm4DPS.rasterizer import render_factorized
    from mm4DPS.rasterizer import PointPrimitives, cull_gaussians, render_gaussians

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.deterministic == "on":
        # W4 (Phase 5): the last two nondeterministic reductions in the step
        # are torch's, not ours -- the `index_add_` backwards of
        # `raw_normals_flat.index_select(0, rows)` and
        # `joint.index_select(0, unit)`. Both become deterministic here. The
        # cuBLAS workspace setting is torch's documented requirement for
        # deterministic GEMMs and has to be in the environment before the
        # first cuBLAS handle is created, which is why it is set here rather
        # than asked of the caller.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    if args.eval_every < 1:
        raise SystemExit(f"--eval-every must be >= 1, got {args.eval_every}")
    if args.md_noise_floor_reference is not None:
        raise SystemExit(
            "--md-noise-floor-reference is removed. It estimated a frozen "
            "noise floor from a previous render pair before training; the "
            "floor is now part of the differentiable forward model. Drop the "
            "flag, or pass --noise-model off to train with no floor at all.")
    # The learned floor used to disable --cuda-graph here, on the reasoning
    # that "the floor changes every step and cannot be captured". That
    # confused a captured *constant* with a captured *computation*. The floor
    # is not read from a pre-computed tensor: `static_losses` calls
    # `noise_model.microdoppler_marginal_batched` inside the captured region,
    # so the graph records those kernels, and they read `log_scale` and
    # `range_coeff` at the addresses Adam updates in place outside the graph.
    # A replay therefore sees the current parameters, exactly as it does for
    # the velocity, material and background parameters it already captured.
    # Shapes are static (`(units, doppler, range)` from `stage_batch`, whose
    # length is fixed), there is no host sync in it, and both parameters sit
    # in `optimizer.param_groups`, so `graph_gradient_holders` already keeps
    # their gradient buffers stable. Measured 1.599 -> 1.331 s/epoch on the
    # steps phase with no change to the loss trace (docs/TRAINING_SPEED_ANALYSIS.md).
    global _ABLATE_TERMS
    _ABLATE_TERMS = {t for t in args.ablate_loss_term.split(",") if t and t != "none"}
    unknown = _ABLATE_TERMS - {"loop0-ra", "plane-ra", "md"}
    if unknown:
        raise SystemExit(f"--ablate-loss-term: unknown term(s) {sorted(unknown)}")
    if "md" in _ABLATE_TERMS:
        # The MD ablation is exactly weight zero: md_l2 stays computed and
        # logged, contributes nothing, and the two-term identity holds.
        args.microdoppler_weight = 0.0
    if _ABLATE_TERMS:
        print(f"ABLATION: loss term(s) {sorted(_ABLATE_TERMS)} removed from the "
              f"objective", flush=True)
    if args.static_anchor != "loop0":
        if args.step_path == "legacy":
            raise RuntimeError(
                "--static-anchor dc-mean is implemented for the batched "
                "step path only")
        print(f"static anchor: {args.static_anchor} (RA term reads the "
              f"slow-time mean plane)", flush=True)
    device = torch.device(args.device)
    joints, velocities = load_trajectories(
        args.sequence_dir, differentiator=args.velocity_differentiator,
    )
    joints[:, :, 1] += args.geometry_translation_y_m
    frames = choose_motion_frames(
        velocities, args.training_frames, edge_guard=args.velocity_edge_guard,
    )
    if args.train_all_frames:
        # Per-frame parameters need per-frame gradients: every chosen frame
        # trains, none are excluded — and nothing is held out, so the
        # per-epoch evaluation covers the SAME full frame set the steps
        # train on (it used to read a [::4] subset for speed; training and
        # evaluating on different frame sets was never the intent of this
        # mode). The guard that needs no held-out frames is the
        # temporal-discrimination delta.
        training_frames = list(frames)
        validation_frames = list(frames)
    else:
        # A genuine held-out split: the [::4] subset is excluded from
        # training and is what the evaluation reads.
        validation_frames = frames[::4]
        training_frames = [frame for frame in frames if frame not in set(validation_frames)]

    person_checkpoint = torch.load(
        args.person_checkpoint, map_location=device, weights_only=False,
    )
    # The joint scattering gains: one scalar per HuPR joint, spread over the
    # body by the skinning weights and multiplied into every point's area.
    # They used to be inherited from a checkpoint fitted on HuPR sequence 1
    # (0.1 on the trunk, 3.0 on the arms) and frozen; they are learned here,
    # from 1, with the materials -- the per-point material has the same
    # freedom point by point, the joint gains give it the body's coarse
    # structure to move as one.
    # Their geometric mean is pinned to 1: the body's overall level is the
    # area scale's and the receiver gain's, the joint gains carry only how
    # it is spread over the body (the three are one product otherwise).
    joint_log_gain = torch.nn.Parameter(torch.zeros(14, device=device),
                                        requires_grad=args.joint_gains == "learned")

    def joint_gains() -> torch.Tensor:
        return torch.exp(joint_log_gain - joint_log_gain.mean())

    configs, rasterizers = build_rasterizers(args, device)

    # GVHMR only: a person cloud that changes length frame to frame cannot
    # carry the per-point material or the per-point normals this trainer
    # fits. The failure is silent, so it is checked rather than documented.
    assert_persistent_person_geometry(args.sequence_dir)
    state = person_checkpoint["shared_background_state"]
    background = PointPrimitives(len(state["positions"]), device=str(device))
    background.load_state_dict(state)
    background.requires_grad_(False).eval()
    # The scene's materials and normals are what train. The `gains` arm --
    # a per-point scalar on a frozen splat -- was never used by any run
    # (measured: 89 result.json, all "material") and is deleted; `pre-item7`
    # has it if it is ever wanted back.
    # Materials and normals, not positions: the carrier phase is detached
    # (`detach_phase=True`), so a moved point would change the envelope
    # without the phase that justifies the move -- which is the failure
    # R003 measured when it optimized positions. `rotations` carries the
    # normal (3DPS derives it as the third column of the rotation matrix),
    # so making it trainable is what makes the normal trainable.
    background.raw_materials.requires_grad_(True)
    background.rotations.requires_grad_(True)

    # The scene renders differentiably through cached splats: each point's
    # unit-gain complex contribution is rasterized ONCE (positions, phases,
    # BSDF frozen), and the per-step background is the gain-weighted
    # coherent sum — one scatter, with gradient flowing to a per-point
    # gain shared by both radars. At gains == 1 this reproduces the frozen
    # background exactly (gated below), so the configuration space strictly
    # contains the old one.
    background_components = {}
    # Option B: the scene's physical solid-angle weights (fixed: the loaded normals, frozen positions) and the unit
    # every slot of both sets is expressed in; the person's per-frame weights below use the same unit.
    scene_slot: dict[str, torch.Tensor] = {}
    solid_angle_unit: dict[str, float] = {}
    if args.scene_solid_angle == "physical" or args.person_solid_angle_weight == "physical":
        with torch.no_grad():
            for radar in RADARS:
                weights, unit = scene_solid_angle_weights(background.positions.detach(), background.get_normals().detach(),
                    torch.as_tensor(config_centre_m(configs[radar]), dtype=torch.float32, device=device))
                scene_slot[radar], solid_angle_unit[radar] = weights.float(), unit
                print(f"solid angle ({radar}): scene unit {unit:.3e} sr per point (median), scene slot p10/p50/p90 "
                      f"{float(weights.quantile(0.1)):.3f}/{float(weights.median()):.3f}/{float(weights.quantile(0.9)):.3f}", flush=True)
    background_active = {}
    background_mask = {}
    background_static = {}
    with torch.no_grad():
        for radar, rasterizer in rasterizers.items():
            mask = cull_gaussians(
                background, rasterizer, near_field_m=HUPR_NEAR_FIELD_M,
            )
            background_active[radar] = mask.nonzero(as_tuple=True)[0]
            background_mask[radar] = mask
            # Static for the whole run, and precomputed so the per-step render
            # holds no host sync and no data-dependent shape: `int(mask.sum())`
            # is a device->host sync and `tensor[mask]` is a `nonzero`, and
            # either one makes the step uncapturable.
            background_static[radar] = {
                "index": background_active[radar],
                "positions": background.positions.index_select(
                    0, background_active[radar]).detach(),
                "areas": (scene_slot[radar].index_select(0, background_active[radar]) if args.scene_solid_angle == "physical"
                          else torch.ones(int(mask.sum()), device=device, dtype=torch.float32)),
            }
            components = render_factorized(
                background.positions[mask], background.get_normals()[mask],
                mask.float()[mask], background.raw_materials[mask],
                rasterizer, reparameterize_torch, detach_phase=True,
                shadow_mask=None, bsdf_mode="full", disabled_components=None,
                use_cuda_kernels=True, return_splat_components=True,
            )
            background_components[radar] = {
                key: (value.detach() if torch.is_tensor(value) else value)
                for key, value in components.items()
            }
    from mm4DPS.forward_render import (
        assemble_static_profile_cuda,
        assemble_static_profile_frames_cuda,
        _psf_table4, _SORTED_BINS, doppler_taps, sorted_doppler_order,
        assemble_surface_slow_time_profiles_cuda,
        assemble_surface_slow_time_profiles_batched_cuda,
        assemble_surface_slow_time_profiles_units_cuda,
        assemble_surface_doppler_planes_units_cuda,
    )
    import mm4DPS.forward_render as _cuda_render
    _cuda_render.set_deterministic(args.deterministic == "on")
    _cuda_render.set_doppler_amplitudes(args.doppler_amplitudes)
    # The truncated Doppler PSF needs the window (the unwindowed Dirichlet
    # tail is 1/delta) and the closed form.
    taps = args.doppler_taps if (args.doppler_amplitudes == "closed-form"
                                 and fft_window_kind() == "hann") else 0
    _cuda_render.set_doppler_taps(taps)
    print(f"Doppler PSF taps: {taps if taps else 'every plane (exact)'}", flush=True)
    print(f"Doppler-plane amplitudes: {args.doppler_amplitudes}"
          + (" (the Hann Doppler PSF at each path's fractional bin, no loop over the 64 loops)"
             if args.doppler_amplitudes == "closed-form" else " (the 64-loop sum against the plane weights)"),
          flush=True)
    static_assemble = assemble_static_profile_cuda
    slow_time_assemble = assemble_surface_slow_time_profiles_cuda
    slow_time_assemble_batched = assemble_surface_slow_time_profiles_batched_cuda
    slow_time_assemble_units = assemble_surface_slow_time_profiles_units_cuda
    doppler_planes_units = functools.partial(
        assemble_surface_doppler_planes_units_cuda,
        static_anchor=args.static_anchor)
    # The DC-plane term's render: the same assembler with the dc-mean anchor (the slow-time mean over the loops:
    # the static background in full, every person path at its Dirichlet-weighted mean) and one dummy Doppler bin,
    # 2 planes of work against the main call's 1 + support. Only the anchor is read.
    doppler_planes_units_dc = functools.partial(
        assemble_surface_doppler_planes_units_cuda, static_anchor="dc-mean")
    DC_DUMMY_BIN = TDM_LOOPS // 2 + 1

    # A: the scene's positions are frozen and the phase is detached, so the
    # splat geometry -- distances, antenna gains, carrier phasors, PSF
    # weights, destination indices -- is constant across steps. One fill call
    # per radar stores it; every later step recomputes only the BSDF
    # amplitude. The cached path replays the same multiplies on the stored
    # values, so the background re-render gate below still reads 0.000e+00.
    background_geometry_cache: dict[str, dict] = {radar: {} for radar in RADARS}

    # PER-FRAME SCENE VISIBILITY.
    #
    # The scene's splat components are cached because a per-point GAIN is
    # linear in them (see `current_background`). A visibility mask is exactly
    # such a gain -- 0 where the person occludes the point, 1 elsewhere -- so
    # switching it on costs one gather and one multiply, NOT a re-render. The
    # mask is (frames, N) bool, loaded once, and lives on the device for the
    # whole run; nothing per-step crosses the bus.
    #
    # What it does cost: `current_backgrounds()` used to be computed once per
    # parameter state and shared by every unit in the step (E5). With
    # visibility the background is frame-dependent, so a step assembles one
    # background per (frame, radar) instead of one per radar. That is the
    # honest price of the feature and it is measured, not assumed -- see
    # `--scene-visibility-benchmark`.
    scene_visibility = None
    if args.scene_visibility is not None:
        loaded = np.load(args.scene_visibility)
        # POINT COUNT CANNOT IDENTIFY A MASK. Every sequence's background model
        # is downsampled to the same 20000 points, so seq102's and seq256's
        # masks are both (600, 20000) and a shape check passes on the wrong
        # one -- while R083's per-sequence alignments differ by up to 1.6 m, so
        # the wrong mask shadows entirely the wrong points, silently. The
        # sidecar carries a sha256 of the exact float32 positions the mask was
        # cast against, which identifies the ALIGNMENT as well as the sequence.
        sidecar = Path(args.scene_visibility).with_suffix(".json")
        if not sidecar.exists():
            raise FileNotFoundError(
                f"{sidecar} is missing. A visibility mask is only meaningful "
                "against the geometry it was computed on, and its shape cannot "
                "establish that. Rebuild the mask with `pointset "
                "scene-shadow-mask`, which writes the sidecar.")
        manifest = json.loads(sidecar.read_text())
        digest = hashlib.sha256(
            background.positions.detach().cpu().numpy().astype(np.float32)
            .tobytes()).hexdigest()
        if manifest["point_checksum_sha256"] != digest:
            raise ValueError(
                "this visibility mask was computed against different scene "
                f"geometry: mask says {manifest['point_checksum_sha256'][:16]} "
                f"(sequence {manifest.get('sequence')}), this checkpoint is "
                f"{digest[:16]}. Per-sequence alignments differ by up to 1.6 m "
                "(R083), so the wrong mask shadows the wrong points and "
                "nothing downstream would notice.")
        # THE CHECKSUM CANNOT CATCH A STALE EXPERIMENT AGAINST THE RIGHT
        # GEOMETRY. It binds the mask to the scene POSITIONS, so it fires on
        # the wrong sequence or a refitted alignment -- but the mask also
        # depends on where the PERSON was put, and that is not in the
        # positions. Measured: seq38's mask cast at -1.15 instead of its own
        # -1.05 shadows 6.03% of the scene against the correct file's 5.69%,
        # a different mask with an identical checksum. So the sidecar carries
        # the translation it was cast at and this refuses a disagreement.
        sidecar_translation = manifest.get("translation_y_m")
        if sidecar_translation is None:
            raise ValueError(
                f"{sidecar} does not record `translation_y_m`. A mask is only "
                "meaningful for the person placement it was cast at, and the "
                "geometry checksum cannot establish that -- the person is not "
                "in the scene positions. Rebuild the mask with a current "
                "`pointset scene-shadow-mask`.")
        if float(sidecar_translation) != float(args.geometry_translation_y_m):
            raise ValueError(
                "this visibility mask was cast with the person at "
                f"translation_y_m={float(sidecar_translation)}, but this run "
                f"places the person at {float(args.geometry_translation_y_m)}. "
                "The scene geometry matches, so the checksum passes and the "
                "mask is still wrong: it shadows the points a differently "
                "placed person would have occluded.")
        if loaded.shape[1] != int(background.positions.shape[0]):
            raise ValueError(
                f"visibility mask is for {loaded.shape[1]} scene points but "
                f"the checkpoint has {int(background.positions.shape[0])}.")
        scene_visibility = torch.as_tensor(
            loaded, dtype=torch.float32, device=device)
        print(f"scene visibility: sequence {manifest.get('sequence')}, "
              f"geometry {digest[:12]}, {loaded.shape[0]} frames, "
              f"{loaded.shape[1]} points, "
              f"{100 * (1.0 - loaded.mean()):.1f}% shadowed on average",
              flush=True)

    # R094: the background's K1 permutation, held for the run. `assemble_static_
    # profile_cuda` rebuilds the sort whenever it misses, and the trainable
    # background hands it a fresh components dict every step, so without this
    # the 1693 us sort would be paid per step and swallow the map's win.
    background_map_cache: dict[str, dict] = {radar: {} for radar in RADARS}

    # A background assembled with `frame=None` uses the UNMASKED scene. That is
    # correct for the startup gates and for a run with no mask, and silently
    # wrong for anything that renders a frame. Three separate defects in this
    # codebase have had exactly that shape -- E8 dropped `normals()`'s
    # missing-span fallback, `rms_clip_grad` orphaned the captured graph's
    # buffer, and `render_losses` asked for a frameless background under a
    # loaded mask -- and all three ran clean and produced plausible numbers
    # while doing the wrong thing. So the frameless path is CLOSED by default
    # once a mask is loaded, and a caller that genuinely wants the unmasked
    # scene has to say so and say why.
    allow_unmasked: dict = {"depth": 0, "reason": None}

    @contextlib.contextmanager
    def unmasked_background(reason: str):
        allow_unmasked["depth"] += 1
        allow_unmasked["reason"] = reason
        try:
            yield
        finally:
            allow_unmasked["depth"] -= 1

    def k1_assemble(radar: str, components: dict,
                    gains: torch.Tensor) -> torch.Tensor:
        """Every background assemble goes through here, and through the map.

        R094: the fixed-order map rather than the atomic scatter. Same
        quantity; 6x faster forward, because 3.49M atomicAdds into 3072 cells
        is contention rather than bandwidth; and unlike the scatter it
        reproduces itself bitwise.
        """
        if background_map_cache is None:
            return static_assemble(components, gains)
        return assemble_static_profile_cuda(
            components, gains, map_cache=background_map_cache[radar])

    def visible_gains(radar: str, gains: torch.Tensor,
                      frame: int | None) -> torch.Tensor:
        """`gains` with the frame's occluded points switched off."""
        if scene_visibility is None:
            return gains
        if frame is None:
            if allow_unmasked["depth"] == 0:
                raise RuntimeError(
                    "a scene-visibility mask is loaded, but this background "
                    "assemble asked for no frame, so it would render the "
                    "UNMASKED scene and nothing downstream would notice. "
                    "Every render of a frame must pass that frame. If the "
                    "unmasked background is genuinely what is wanted (a "
                    "startup gate, an identity check), wrap the call in "
                    "`unmasked_background(reason)`.")
            return gains
        return gains * scene_visibility[frame].index_select(
            0, background_active[radar])

    # R093: the evaluation's (frame, radar) background memo.
    #
    # Measured, seq 256, one epoch: 4800 frame-keyed assemblies over just 1200
    # distinct (frame, radar) pairs, every pair computed exactly 4 times. The
    # repeats are NOT within a pass -- each of the four post-epoch passes (the
    # two `evaluate` calls, the radiometric pass, the final evaluation) walks
    # all 1200 once. So the memo has to span passes, and "for the life of one
    # pass" would buy nothing.
    #
    # What makes it safe is the KEY, not the scope: `parameter_version` is
    # bumped on every optimizer step, so a cached background can only be
    # returned at the parameter state that produced it. That is exactly E5's
    # argument -- no parameter changes inside `no_grad` -- but keyed rather
    # than assumed, so it cannot silently outlive its state the way a
    # pass-scoped cache would if a pass ever moved.
    #
    # Entries are only made with the gradient OFF. Training assembles with
    # grad on and never reads or writes this, so the captured step is
    # untouched. 1200 entries x 3x4x256 complex64 = 29 MB.
    parameter_version = {"n": 0}
    background_memo: dict = {}

    def background_splat(radar: str, carrier_only: bool = False) -> tuple[dict, torch.Tensor]:
        """The radar's background splat components and their base gains.

        ``carrier_only``: the fused frames path's inputs (the carrier, the
        fractional bins, the PSF table) instead of the (spread, paths)
        contributions; only on a filled geometry cache.

        FRAME-INDEPENDENT, and that is the whole point. Visibility enters as a
        per-point *gain* (`visible_gains`), never as an input to the render, so
        nothing here depends on the frame: same positions, same normals, same
        materials, same rasterizer. Splitting it out lets a caller that needs
        several frames' backgrounds at one parameter state render ONCE and
        assemble per frame, instead of re-rendering the identical splat per
        frame -- which is what the per-frame visibility path did (R091).
        """
        # The cached splat exists because a per-point *gain* is linear in the
        # cached contribution, so the BSDF could be evaluated once. A per-point
        # material and normal are not, so the scene's BSDF is re-evaluated
        # every step. Areas stay at one, exactly as the cache build used them,
        # so at the loaded material and normals this reproduces the cached
        # background bit for bit -- gated at startup below.
        fixed = background_static[radar]
        index = fixed["index"]
        components = render_factorized(
            fixed["positions"],
            background.get_normals().index_select(0, index),
            fixed["areas"],
            background.raw_materials.index_select(0, index),
            rasterizers[radar], reparameterize_torch, detach_phase=True,
            shadow_mask=None, bsdf_mode="full", disabled_components=None,
            use_cuda_kernels=True, return_splat_components=True,
            geometry_cache=background_geometry_cache[radar],
            carrier_only=carrier_only,
        )
        return components, fixed["areas"]

    def assemble_background(radar: str, splat: tuple[dict, torch.Tensor],
                            frame: int | None) -> torch.Tensor:
        """One frame's background from an already-rendered splat.

        Same two arguments `current_background` passed to `static_assemble`,
        in the same order, so a hoisted render assembles bitwise identically
        to the per-frame one.
        """
        components, gains = splat
        if frame is None or torch.is_grad_enabled():
            return k1_assemble(radar, components, visible_gains(radar, gains, frame))
        key = (radar, int(frame))
        if background_memo.get("version") != parameter_version["n"]:
            background_memo.clear()
            background_memo["version"] = parameter_version["n"]
        hit = background_memo.get(key)
        if hit is not None:
            return hit
        # `torch.stack` in both callers copies every row, so the cached tensor
        # is only ever read; nothing downstream can write through it.
        value = k1_assemble(radar, components, visible_gains(radar, gains, frame))
        background_memo[key] = value
        return value

    def current_background(radar: str, frame: int | None = None) -> torch.Tensor:
        return assemble_background(radar, background_splat(radar), frame)

    # The absolute receiver gain per radar: a number, or "auto" (the
    # default) -- one learned log-gain per radar, initialised from the
    # first frames' measured-to-rendered RA level and trained with the
    # rest, so no calibration precedes the trainer (HuPR's shipped gain
    # and RT-Pose's pre-training fit from the static scene, whose value
    # followed the background fit's convergence: docs/FIDELITY_PLAN.md
    # 2.2(a)). The trained value is the estimate the renderer applies.
    # `person`: the constant the person stage calibrated from the scene and fitted the person's level under
    # (docs/STATIC_SEQS_NEXT.md step 1), fixed here so that level survives the velocity stage
    _gain_flags = per_radar(args, "absolute-gain")
    if args.absolute_gain is not None:                                   # the radar-agnostic spelling: every radar the same
        _gain_flags = {radar: args.absolute_gain for radar in _gain_flags}
    absolute_gains = {radar: (None if str(value).lower() in ("auto", "init") else
                              float(person_checkpoint["absolute_gain"][radar]) if str(value).lower() == "person" else float(value))
                      for radar, value in _gain_flags.items()}
    gain_learned = any(value is None for value in absolute_gains.values())
    if gain_learned and not all(value is None for value in absolute_gains.values()):
        raise ValueError("--absolute-gain: either every radar learns its gain (auto / init) or none does")
    # init: the gain is set from the first frames' measured / rendered level exactly as auto, then FROZEN -- the receiver
    # gain is one calibration constant of the radar, and learned it is degenerate with the scene's per-point materials
    # (base2: 0.9 at the start, 33-88 at the end, the materials moving the other way; the person, with one material
    # and a fixed slot, cannot follow that trade)
    gain_frozen_after_init = any(str(value).lower() == "init" for value in _gain_flags.values())
    absolute_log_gain = torch.nn.Parameter(torch.zeros(len(RADARS), device=device))
    # the person's per-row splat geometry cache (`build_row_geometry`, below);
    # empty until built, and read by every render helper
    row_geometry: dict = {}
    # The boards' static antenna coupling, in product units, added to the
    # anchor (loop-0) plane after the absolute gain exactly as the renderer
    # adds it: the measured loop-0 RA carries it (49% of the cascade's
    # static energy in the near bins, 26-28% of HuPR's) and the render had
    # nothing there, a constant residual in the RA term. None when no
    # calibration is given (the previous behaviour).
    coupling_stack: torch.Tensor | None = None
    if args.antenna_coupling_calibration is not None:
        from mm4DPS.forward_render import coupling_block
        coupling_stack = torch.stack([
            coupling_block(training.radar(radar),
                           np.load(args.antenna_coupling_calibration / f"{radar}_coupling.npy"), device)
            for radar in RADARS])                                            # (radars, tx, rx, K)
        print("antenna coupling: added to the anchor plane from "
              f"{args.antenna_coupling_calibration.name} ({tuple(coupling_stack.shape)})", flush=True)
    keep_range = torch.from_numpy(range_keep_mask()).to(device)
    motion_doppler = torch.from_numpy(official_motion_doppler_mask()).to(device)
    # Every fftshifted Doppler bin except DC, which `fftshift` places at
    # `loops // 2`. DC is excluded because clutter removal subtracts the
    # slow-time mean on *both* sides, making that bin identically zero --
    # `doppler_plane_weights` rejects it for exactly this reason -- so these 63
    # bins are the complete non-trivial set, not a subset. The loop->Doppler
    # DFT is unitary, so every one of the frame's 64 chirps contributes to
    # every bin here: supervising these supervises all 64 measured chirps.
    ra_doppler_bins = [d for d in range(TDM_LOOPS) if d != TDM_LOOPS // 2]
    ra_doppler_index = torch.tensor(ra_doppler_bins, device=device)
    # Flat indices into a plane target's (63, range, angle) grid, or None for
    # the unmasked term. Assigned once the targets exist; bound here because
    # the loss helpers below close over it and run after that.
    plane_cell_index: torch.Tensor | None = None
    # Whether the micro-Doppler term can read its planes out of the plane
    # term's spectrum (the band is the planes' prefix and every person range
    # row lies in the kept range) instead of running its own chain over the
    # band -- decided once the targets exist (see `md_from_plane_spectrum`).
    md_from_plane_spectrum = False
    # Static buffer, refilled per epoch: the graph replays read whatever it
    # holds, so the plane subset can change without a re-capture as long as
    # its length does not.
    plane_sample_index: torch.Tensor | None = (
        None if args.plane_sample_count <= 0 else
        torch.arange(args.plane_sample_count, device=device))
    if plane_sample_index is not None and args.plane_sample_count > len(ra_doppler_bins):
        raise ValueError(
            f"--plane-sample-count {args.plane_sample_count} exceeds the "
            f"{len(ra_doppler_bins)} non-DC planes")
    if args.doppler_support_snr is not None and (
            args.plane_mask_db is not None
            or args.plane_mask_keep_energy is not None
            or args.plane_mask_snr is not None):
        # Both exclude cells that carry no signal, so composing them is
        # redundant -- and worse than redundant: the cell mask estimates its
        # noise floor as the median over (plane, angle) at each range, and once
        # the Doppler support has removed the noise-only planes that median is
        # contaminated by signal. Measured on seq 102, the mask then discards
        # 26.6% of the target L2 against 5% on the full cube, well past the
        # 16.5% that regressed seq 256 in R046.
        raise ValueError(
            "--doppler-support-snr and the --plane-mask-* thresholds both drop "
            "cells that carry no signal; pass one. The Doppler support is the "
            "stronger of the two: it also makes the render cheaper.")
    if plane_sample_index is not None and args.doppler_support_snr is not None:
        raise ValueError(
            "--plane-sample-count and --doppler-support-snr both choose which "
            "planes the term reads; pass one")
    if plane_sample_index is not None and args.plane_mask_db is not None:
        # The cell index is built against the full 63-plane grid, so gathering
        # a plane subset first would make it index the wrong cells. Composing
        # them needs the mask rebuilt per subset; not worth it until one of
        # the two earns its place.
        raise ValueError(
            "--plane-mask-db and --plane-sample-count cannot be combined")

    def plane_noise_power(radar_of_unit: torch.Tensor, plane_bins_rows) -> torch.Tensor | None:
        """The learned floor's expected power at the units' rendered planes, ``(units, planes, kept range, 1)``.

        ``plane_bins_rows`` is the units' fftshifted plane bins (pads -1 ->
        zero floor: a pad plane renders zero and its target is zero), or
        None for the flat non-DC set. Differentiable in the floor's
        parameters; sync-free (runs under graph capture).
        """
        if noise_model is None:
            return None
        power = noise_model.power().index_select(0, radar_of_unit)        # (units, 64, 256)
        if plane_bins_rows is None:
            gathered = power.index_select(1, ra_doppler_index)              # (units, planes, 256)
        else:
            bins = plane_bins_rows
            gathered = power.gather(1, bins.clamp_min(0)[:, :, None].expand(-1, -1, power.shape[2]))
            gathered = gathered * (bins >= 0)[:, :, None].to(gathered.dtype)
        gathered = gathered.index_select(2, keep_index)
        if plane_sample_index is not None:
            gathered = gathered.index_select(1, plane_sample_index)
        return gathered[..., None]

    def plane_cells(spectrum: torch.Tensor) -> torch.Tensor:
        """Gather the supervised cells, or pass the full grid through."""
        if plane_sample_index is not None:
            spectrum = spectrum.index_select(1, plane_sample_index)
        if plane_cell_index is None:
            return spectrum
        return spectrum.reshape(spectrum.shape[0], -1).index_select(
            1, plane_cell_index)

    def plane_cells_target(target: torch.Tensor) -> torch.Tensor:
        """The same gather on the (views, planes, range, angle) target."""
        if plane_sample_index is None:
            return target
        return target.index_select(1, plane_sample_index)
    # THE TARGETS ARE BUILT HERE, EVERY RUN, ON THE DEVICE. Blocks of frames
    # are decoded and beamformed at once (`range_angle_slow_time_cube_frames_torch`,
    # 5e-16 of the numpy arbiter relative to the peak); every per-frame
    # quantity -- the Doppler magnitude, the person's masks, the skinning
    # and solid-angle weights, the floor prior -- is computed from that
    # cube without leaving the GPU. Nothing is written to disk and nothing
    # read back: a run's targets are a function of its inputs, and a
    # 2.5 GB cache file that only a repeat of the same workspace could read
    # is not an output anyone wants.
    cache: dict[int, dict] = {}
    frame_normals: dict[int, torch.Tensor] = {}
    noise_prior_frames: dict[str, list] = {}

    def downstream_view():
        """The downstream consumer's own (range, angle) magnitude view of a (loops, tx, rx, K) cube, Doppler- and
        elevation-summed, plus its grid (range m per row, sin(azimuth) per column), or None when the consumer has no
        absolute level (HuPR: its input is min-max normalised per map). RT-Pose: the HRRadarPose adapter's untapered polar
        transform on the TI virtual-array table -- the trainer's own beamformer (channels_to_grid2d, the averaged mm4dps
        slots) reads the recording's person 1.2-1.5x differently from it on 153 / 164 (docs/STATIC_SEQS_NEXT.md 1e)."""
        if current_dataset().name != "rtpose":
            return None
        from downstream.rtpose_hrradarpose.adapter import RTPoseAdapter
        adapter = RTPoseAdapter(device=device)
        def view(cube: torch.Tensor) -> torch.Tensor:                  # (loops, tx, rx, K) complex -> (K, 128)
            return adapter.polar(cube[None].to(torch.complex64))[0].abs().sum(dim=(0, 3))
        n_range = current_dataset().radar(RADARS[0]).samples
        return view, np.arange(n_range) * range_bin_spacing_m(), (np.arange(128) - 64) / 64.0
    def consumer_loop0_view():
        """The consumer's own (range, azimuth, elevation) magnitude transform of a (..., tx, rx, K) per-channel loop-0
        profile -- the person-window term's view (--person-window-loss) -- with its range axis (m) and sin(azimuth) axis.
        RT-Pose: the HRRadarPose adapter's polar minus its Doppler FFT (the TI table, one channel per slot, 128 x 32,
        untapered). HuPR: the released generateHeatmap's angle stage (TX0 / TX2 on elevation row 0, TX1 on row 1 padded by
        2, the elevation FFT on channel columns 2..5, the azimuth FFT on every row, postProcessFFT3D's fftshift + flip;
        its clutter removal and Doppler selection are slow-time operations and do not apply to one loop). Differentiable,
        fixed-shape, no host sync: the step calls it on gathered range rows only."""
        name = current_dataset().name
        radar0 = current_dataset().radars[0]
        r_axis = np.arange(radar0.samples) * range_bin_spacing_m()
        if name == "rtpose":
            from downstream.rtpose_hrradarpose.adapter import RTPoseAdapter, N_AZ as _NAZ, N_EL as _NEL
            ad = RTPoseAdapter(device=device)
            n_slots = ad.n_az_slots * ad.n_el_rows
            def view(profile: torch.Tensor) -> torch.Tensor:                 # (..., tx, rx, K) complex -> (..., K, 128, 32)
                x = torch.movedim(profile, -1, -3)                            # (..., K, tx, rx)
                chan = x[..., ad.src_tx, ad.src_rx]                           # (..., K, kept)
                grid = x.new_zeros((*chan.shape[:-1], n_slots)).index_copy(-1, ad.dst_slot, chan)
                grid = grid.reshape(*chan.shape[:-1], ad.n_az_slots, ad.n_el_rows)
                y = torch.fft.fftshift(torch.fft.fft(grid, n=_NAZ, dim=-2), dim=-2)
                y = torch.fft.fftshift(torch.fft.fft(y, n=_NEL, dim=-1), dim=-1)
                return y.abs()
            return view, r_axis, (np.arange(_NAZ) - _NAZ // 2) / (_NAZ // 2), _NEL
        if name == "hupr":
            from downstream.hupr_mscsa.adapter import N_AZ as _NAZ, N_EL as _NEL
            def view(profile: torch.Tensor) -> torch.Tensor:                 # (..., 3, 4, K) complex -> (..., K, 64, 8)
                x = torch.movedim(profile, -1, -3)                            # (..., K, 3, 4)
                lead = x.shape[:-2]
                row0 = torch.cat([x[..., 0, :], x[..., 2, :]], dim=-1)        # TX0 RX0-3, TX2 RX0-3: azimuth channels 0..7
                pad = x.new_zeros((*lead, 2))
                row1 = torch.cat([pad, x[..., 1, :], pad], dim=-1)            # TX1 RX0-3 at columns 2..5
                grid = torch.stack([row0, row1] + [x.new_zeros((*lead, 8))] * (_NEL - 2), dim=-2)   # (..., K, 8 el rows, 8 az channels)
                cols = torch.fft.fft(grid[..., 2:6], dim=-2)                  # the elevation FFT on columns 2..5 only
                grid = torch.cat([grid[..., :2], cols, grid[..., 6:]], dim=-1)
                grid = torch.fft.fft(grid, n=_NAZ, dim=-1)                    # (..., K, el, az)
                y = torch.movedim(grid, -1, -2)                               # (..., K, az, el)
                return torch.fft.fftshift(y, dim=(-2, -1)).flip(dims=(-2, -1)).abs()
            return view, r_axis, (_NAZ // 2 - 1 - np.arange(_NAZ)) / (_NAZ // 2), _NEL
        return None
    pw_on = args.person_window_loss > 0
    pw_view = consumer_loop0_view() if pw_on else None
    if pw_on and pw_view is None:
        raise SystemExit(f"--person-window-loss: no consumer view for dataset {current_dataset().name!r}")
    if pw_on:
        _pw_view_fn, _pw_r_axis, _pw_s_axis, _pw_n_el = pw_view
        _pw_resolved = args.person_window_planes == "resolved"
        # the consumer's Doppler band among the trainer's 63 non-DC planes (fftshifted bins, DC = 32 absent), and whether the
        # consumer keeps the DC (RT-Pose's polar: DC kept, every bin summed; HuPR: clutter removed, chirps 24..39)
        if args.person_window_planes in ("consumer", "resolved"):
            _pw_bins = ([d for d in range(24, 40) if d != TDM_LOOPS // 2] if current_dataset().name == "hupr"
                        else [d for d in range(TDM_LOOPS) if d != TDM_LOOPS // 2])
            _pw_plane_index = torch.tensor([d if d < TDM_LOOPS // 2 else d - 1 for d in _pw_bins], device=device)
            _pw_keep_dc = current_dataset().name != "hupr"
        else:
            _pw_bins, _pw_plane_index, _pw_keep_dc = [], None, True
        # how many Doppler planes the stored window carries: one unless it is kept plane by plane
        _pw_bands = (len(_pw_bins) + (1 if _pw_keep_dc else 0)) if _pw_resolved else 1
        if _pw_resolved:
            print(f"person-window term: RESOLVED over {_pw_bands} Doppler planes "
                  f"({'DC + ' if _pw_keep_dc else ''}{len(_pw_bins)} band planes), one rms over the whole window",
                  flush=True)
        _PW_DC_SCALE, _PW_PLANE_SCALE = float(TDM_LOOPS), 2.0        # the unwindowed DC of a static signal = 64 x its mean; a Hann-windowed plane = 0.5 x the unwindowed
        pw_half_r = max(int(round(args.person_window_range / range_bin_spacing_m())), 1)
        pw_half_a = 10                                                     # +-10 sin bins hold +-0.6 m lateral beyond 3.8 m (RT-Pose) / 1.9 m (HuPR); the mask trims inside
        pw_offsets_r = torch.arange(-pw_half_r, pw_half_r + 1, device=device)
        pw_offsets_a = torch.arange(-pw_half_a, pw_half_a + 1, device=device)
        pw_side_energy: dict[str, list] = {radar: [0.0, 0.0] for radar in RADARS}
    noise_prior_power: torch.Tensor | None = None       # (radars, loops, range), the sequence's own floor
    dataset = Dataset.of(args).full()      # the RECORDED arrays: the measured side decodes with these
    readers = {radar: MeasuredReader(dataset, args.sequence, dataset.radar(radar),
                                     dataset_root=args.dataset_root) for radar in RADARS}
    target_block = 8
    measured_cubes: dict[str, torch.Tensor] = {}       # this block's (frames, loops, range, angle) per radar
    block_start = 0

    measured_rae: dict[str, torch.Tensor] = {}         # this block's (frames, loops, range, angle, elevation) per radar
    measured_channels: dict[str, torch.Tensor] = {}    # this block's (frames, loops, tx, rx, K) per-channel profiles (the product's layout)
    downstream_view_fn = downstream_view() if args.person_level_calibration_view == "downstream" else None
    measured_rae_unwindowed: dict[str, torch.Tensor] = {}
    elevation_on = args.elevation_term == "on"
    _win_rae_on = args.window_rae_terms == "on"
    if _win_rae_on and not elevation_on:
        raise SystemExit("--window-rae-terms on needs --elevation-term on: the windowed terms are the elevation terms at a second scope")
    # Whether the RAE targets are stored un-shifted in azimuth, which is only worth doing
    # where the shift is a real roll (the cascade); HuPR folds it into input signs for free.
    # Set here because the target cache below reads it.
    _rae_unshift_targets = bool(elevation_on and _RAE_AZ_SHIFT_SKIP and rae_azimuth_rolled())
    if elevation_on:
        print(f"elevation term: azimuth fftshift "
              f"{'skipped on both sides' if _rae_unshift_targets else 'applied'}"
              f" (a real roll on this array: {rae_azimuth_rolled()})", flush=True)
    static_on = args.static_term == "on"
    if static_on and args.step_path == "legacy":
        raise ValueError("--step-path legacy has no DC-plane term; run it with --static-term off")
    if elevation_on and args.training_array != "full":
        raise SystemExit("--elevation-term on needs --training-array full: the elevated TX must be rendered to be supervised")

    def measured_cube_of(radar: str, cache_index: int) -> torch.Tensor:
        """The frame's complex per-loop RA cube, from the block decoded on the device."""
        nonlocal block_start
        if cache_index >= block_start + target_block or not measured_cubes:
            block_start = cache_index - cache_index % target_block
            block = frames[block_start:block_start + target_block]
            contiguous = block == list(range(block[0], block[0] + len(block)))
            for name, reader in readers.items():
                adc = (reader.batch(block[0], len(block), device) if contiguous
                       else torch.stack([reader.batch(f, 1, device)[0] for f in block]))
                measured_cubes[name] = range_angle_slow_time_cube_frames_torch(
                    adc, n_angle=training.radar(name).n_angle, radar=dataset.radar(name))
                if elevation_on:
                    # the elevation term's target view: every TX on the 2D grid (data.elevation_grid_plan)
                    from mm4DPS.data import range_angle_elevation_slow_time_cube_frames_torch
                    # the TRAINING radar: the full array (the elevated TX rendered) at the loss's
                    # angle-FFT size (--n-angle), as the prediction side resolves it
                    measured_rae[name] = range_angle_elevation_slow_time_cube_frames_torch(
                        adc, radar=training.radar(name))
                    # the same view without the azimuth window, for the level calibration's target (a scalar per view
                    # is kept; the cube is dropped with the block)
                    if args.person_level_calibration == "peak-doppler":
                        measured_rae_unwindowed[name] = range_angle_elevation_slow_time_cube_frames_torch(
                            adc, radar=training.radar(name), window=False)
                if args.person_level_calibration_view == "downstream" or pw_on:
                    # the per-channel range profiles (the product's own layout), for the consumer's transform
                    measured_channels[name] = torch.stack([range_profiles_all_channels_torch(adc[i], radar=dataset.radar(name))
                                                           for i in range(adc.shape[0])])
        return measured_cubes[radar][cache_index - block_start]

    def measured_cube_channels_of(radar: str, cache_index: int) -> torch.Tensor:
        measured_cube_of(radar, cache_index)
        return measured_channels[radar][cache_index - block_start]

    def measured_rae_of(radar: str, cache_index: int) -> torch.Tensor:
        measured_cube_of(radar, cache_index)
        return measured_rae[radar][cache_index - block_start]

    def measured_rae_unwindowed_of(radar: str, cache_index: int) -> torch.Tensor:
        measured_cube_of(radar, cache_index)
        return measured_rae_unwindowed[radar][cache_index - block_start]

    centres = {radar: torch.as_tensor(config_centre_m(configs[radar]), dtype=torch.float32, device=device)
               for radar in RADARS}
    axes = {radar: torch.as_tensor(np.asarray(array_axis(configs[radar])), dtype=torch.float32, device=device)
            for radar in RADARS}
    person_faces = smpl_faces() if args.person_solid_angle_weight == "physical" else None
    with torch.no_grad():
        for cache_index, frame in enumerate(frames):
            loaded = np.load(args.sequence_dir / "person_frames" / f"frame_{frame:03d}.npz")
            pcl_np = loaded["pcl"].astype(np.float32)
            full_pcl_np = pcl_np.copy(); full_pcl_np[:, 1] += args.geometry_translation_y_m   # every vertex, for the mesh areas
            # Which global point each retained row is, so a per-point material
            # can be indexed by identity rather than by position in this
            # frame's cloud. Without a `visible` mask the cloud *is* the global
            # set and the mapping is the identity.
            global_index_np = (
                np.nonzero(loaded["visible"])[0].astype(np.int64)
                if "visible" in loaded
                else np.arange(len(pcl_np), dtype=np.int64)
            )
            if "visible" in loaded:
                # Per-frame ray-cast visibility over the fixed global point
                # set: only points the radar can see contribute this frame.
                pcl_np = pcl_np[loaded["visible"]]
            pcl_np[:, 1] += args.geometry_translation_y_m
            pcl = torch.from_numpy(pcl_np).to(device)                     # (N, 6): positions, normals
            entry = {
                "global_index": torch.from_numpy(global_index_np).to(device),
                # The size of the *global* set this frame draws from, before
                # the visibility mask. Constant across frames exactly when the
                # point set persists, which is what a per-point material needs.
                "global_point_total": torch.tensor(
                    len(loaded["pcl"]), device=device),
                "points": pcl[:, :3].contiguous(),
                "stored_normals": pcl[:, 3:6].contiguous(),
                "weights": soft_joint_weights_torch(
                    pcl[:, :3], torch.as_tensor(joints[frame], dtype=torch.float32, device=device),
                    args.skinning_temperature_m, args.skinning_topk),
                "views": {},
            }
            for radar, rasterizer in rasterizers.items():
                ra_mask = person_range_angle_mask_torch(
                    entry["points"], centres[radar], axes[radar],
                    range_bin_spacing_m(), include_mirrored_angle=True,
                )
                md_mask, md_rows = ra_mask, None
                if True:            # the box is always built: the probe reads it as pw_box_mask
                    # the person-window term's box, as a (range, angle) mask in the trainer's own grid.
                    # Built from a synthetic cloud filling the box rather than from the axis arithmetic,
                    # so the angle convention -- the beamformer's mirror -- comes from the same function
                    # the person's mask uses and cannot drift from it.
                    _j = torch.as_tensor(joints[frame], dtype=torch.float32, device=device)
                    _pelvis = _j[[0, 3]].mean(dim=0)
                    _u_r = (_pelvis - centres[radar]); _u_r = _u_r / _u_r.norm().clamp_min(1e-9)
                    _ax = axes[radar] / axes[radar].norm().clamp_min(1e-9)
                    _u_l = _ax - (_ax @ _u_r) * _u_r; _u_l = _u_l / _u_l.norm().clamp_min(1e-9)
                    _u_z = torch.tensor([0.0, 0.0, 1.0], device=device)
                    _dz_lo = float((_j[:, 2] - _pelvis[2]).min()) - 0.2
                    _dz_hi = float((_j[:, 2] - _pelvis[2]).max()) + 0.2
                    _g = torch.stack(torch.meshgrid(
                        torch.linspace(-args.person_window_range, args.person_window_range, 9, device=device),
                        torch.linspace(-args.person_window_lateral, args.person_window_lateral, 15, device=device),
                        torch.linspace(_dz_lo, _dz_hi, 13, device=device), indexing="ij"), dim=-1).reshape(-1, 3)
                    _box = (_pelvis[None] + _g[:, :1] * _u_r[None] + _g[:, 1:2] * _u_l[None] + _g[:, 2:3] * _u_z[None])
                    box_mask = person_range_angle_mask_torch(
                        _box, centres[radar], axes[radar], range_bin_spacing_m(),
                        range_padding_bins=0, angle_padding_bins=0, include_mirrored_angle=True,
                    )
                    del _j, _g, _box
                if args.microdoppler_support == "window":
                    md_mask = box_mask
                range_rows = ra_mask.any(dim=1)
                md_rows = md_mask.any(dim=1) if args.microdoppler_support == "window" else range_rows
                measured_cube = measured_cube_of(radar, cache_index)
                measured_doppler = slow_time_ra_to_doppler_magnitude(measured_cube)
                # the person mask's peak of the all-Doppler magnitude sum -- the plain 64-point FFT over the loops, no
                # window, no mean removal, DC kept, as a downstream tensor reads it (a magnitude sum over windowed
                # mean-removed planes rewards Doppler spread and drops the standing body's DC: measured 1.55x on a render
                # whose energy was 0.25x); the target of --person-level-calibration peak-doppler (a scalar per view)
                _dsum = torch.fft.fft(measured_cube, dim=0).abs().sum(dim=0)
                dsum_peak = float(torch.where(ra_mask, _dsum, torch.zeros_like(measured_doppler[0])).max())
                # the same map inside the mask, for the calibration that reads the person's own peak cell (--person-level-calibration-cell person)
                dsum_masked = torch.where(ra_mask, _dsum, torch.zeros_like(_dsum)).cpu()
                dsum_full = _dsum.cpu(); del _dsum
                if args.person_level_calibration_view == "downstream" and downstream_view_fn is not None:
                    # the recording in the consumer's transform, the person's window about the mesh pelvis (as contrast_node)
                    entry_dsum_ds = downstream_view_fn[0](measured_cube_channels_of(radar, cache_index)).cpu()
                # THE NOISE PRIOR, from this sequence: this frame's windowed
                # |RD|^2 per (Doppler, range), the mean over the angle cells
                # the person is not in (all of them at the ranges the person
                # is not at). The median over frames is the floor the learned
                # noise model starts from -- no calibration file, no other
                # sequence: what this recording's own empty cells show
                # through the same windowed transform the targets take.
                off_person = (~ra_mask).to(measured_doppler.dtype)           # (range, angle)
                floor_frame = ((measured_doppler ** 2) * off_person[None]).sum(dim=2) \
                    / off_person.sum(dim=1).clamp_min(1.0)[None]              # (loops, range)
                noise_prior_frames.setdefault(radar, []).append(floor_frame.cpu())
                if args.doppler_support_snr is not None:
                    # This frame's Doppler support, at this frame's person
                    # range. `range_rows` is the person's range mask, so the
                    # window follows the person rather than a fixed band.
                    power = (measured_doppler[:, range_rows] ** 2).sum(
                        dim=(1, 2))                       # (loops,)
                    if args.doppler_support_floor == "loops-median":
                        # HuPR's estimator: the median over the 64 bins of
                        # the person rows' own power. By construction at
                        # least half the bins clear it, still person or not.
                        floor = power.median().clamp_min(1e-30)
                    else:
                        # "off-person": the SAME range rows, the angle cells
                        # the person is NOT in. Per Doppler bin, the per-cell
                        # power of those cells -- the clutter skirt and the
                        # receiver floor at the person's own ranges, bin by
                        # bin -- scaled to the person's cell count. A bin is
                        # supported when the person's cells carry
                        # snr x more than their neighbours at that bin, so a
                        # still person supports the few bins it occupies.
                        # (Rows the person is not in are no reference: the
                        # skirt is range-dependent and made every bin pass.)
                        rows_power = measured_doppler[:, range_rows] ** 2      # (loops, rows, angle)
                        mask_rows = ra_mask[range_rows]                        # (rows, angle)
                        person_cells = int(mask_rows.sum())
                        off_cells = int((~mask_rows).sum())
                        power = (rows_power * mask_rows[None]).sum(dim=(1, 2))
                        per_cell_off = (rows_power * (~mask_rows)[None]).sum(
                            dim=(1, 2)) / max(off_cells, 1)
                        floor = (per_cell_off * person_cells).clamp_min(1e-30)
                    if args.doppler_support_floor == "energy":
                        # The fewest bins that carry --doppler-support-energy
                        # of the person cells' non-DC Doppler energy, in
                        # descending order of power. A still person's energy
                        # sits in a handful of bins around DC; a walking one
                        # spreads over ~20. The skirt a strong return leaves
                        # across every bin is real but carries almost no
                        # energy, and under an SNR test it made every view
                        # support 63 of 63 bins (SNR 1, 1.5 and 2 alike).
                        rows_power = measured_doppler[:, range_rows] ** 2
                        mask_rows = ra_mask[range_rows]
                        power = (rows_power * mask_rows[None]).sum(dim=(1, 2))
                        power[TDM_LOOPS // 2] = 0.0
                        order = torch.argsort(power, descending=True)
                        cumulative = torch.cumsum(power[order], dim=0) / power.sum().clamp_min(1e-30)
                        count = int((cumulative < args.doppler_support_energy).sum()) + 1
                        view_support = sorted(order[:count].tolist())
                    else:
                        view_support = torch.nonzero(
                            power >= args.doppler_support_snr * floor,
                            as_tuple=False).flatten().tolist()
                if args.static_anchor == "dc-mean":
                    # The measured DC plane: coherent mean over the 64 loops,
                    # then magnitude -- same static content as loop 0 at
                    # ~8x the SNR, matching the rendered anchor plane.
                    ra_target = measured_cube.mean(dim=0).abs().float()
                else:
                    # loop 0 of the cube just built: what the sequence's
                    # first-loop file holds, from the same transform
                    ra_target = measured_cube[0].abs().float()
                # The DC-plane term's target: the coherent mean over the 64 loops, then magnitude
                dc_target = measured_cube.mean(dim=0).abs().float() if args.static_term == "on" else None
                # The MC weight for this radar. `areas` is a solid angle, not
                # a free scalar; the person cloud is depth-derived and not
                # cosine-resampled, so a constant is not a valid weight (it
                # varies ~6600x across the cloud). Normalized to unit mean, so
                # only the per-point structure enters, never the global level.
                # The renderer applies the identical factor at its
                # render_factorized call -- keeping the two in step is the
                # whole point of computing it here rather than at render time.
                if args.person_solid_angle_weight == "physical":
                    solid_angle = person_solid_angle_weights(full_pcl_np, person_faces, global_index_np, centres[radar],
                                                             solid_angle_unit[radar], device).to(entry["weights"].dtype)
                elif args.person_solid_angle_weight == "on":
                    solid_angle = surface_solid_angle_weights_torch(
                        entry["points"], entry["stored_normals"], centres[radar]).to(entry["weights"].dtype)
                else:
                    solid_angle = torch.ones(len(entry["points"]), dtype=entry["weights"].dtype, device=device)
                entry["views"][radar] = {
                    "solid_angle": solid_angle,
                    "ra_mask": ra_mask,
                    # the micro-Doppler term's own support: the person's mask unless
                    # --microdoppler-support window puts it on the person-window term's box
                    "md_mask": md_mask, "md_rows": md_rows,
                    # the person-window box in the trainer's own (range, angle) grid, always built:
                    # the C1b diagnostic terms read it, and --microdoppler-support window uses it
                    "pw_box_mask": box_mask,
                    "range_rows": range_rows,
                    "ra_target": ra_target.contiguous(),
                    "dsum_peak": dsum_peak, "dsum_masked": dsum_masked, "dsum_full": dsum_full,
                    "dsum_downstream": (entry_dsum_ds if args.person_level_calibration_view == "downstream" and downstream_view_fn is not None else None),
                    **({"dc_target": dc_target.contiguous()} if dc_target is not None else {}),
                    "md_target": microdoppler_marginal(
                        measured_doppler, md_mask, md_rows, motion_doppler,
                    ).clamp_min(0.0).sqrt().detach(),
                    # float32, deliberately. These magnitudes run to ~5e6,
                    # and float16 tops out at 65504: storing them at half
                    # overflows to inf at *cache build*, before the loss can
                    # normalize anything, and every RA gradient comes back NaN.
                    # Measured that way round. 63 x 244 x 64 per frame per
                    # radar is 3.9 MB, i.e. ~4.6 GB on the device for a
                    # 590-frame sequence. If that ever needs halving, store
                    # `t / rms(t)` with the per-view rms beside it -- exactly
                    # equal, since the term already divides by mean(t^2) --
                    # rather than reaching for half precision again.
                    "ra_doppler_target": measured_doppler.index_select(
                        0, ra_doppler_index,
                    )[:, keep_range].to(torch.float32).detach(),
                    # This view's own Doppler support, decided from its own
                    # measurement. Narrowed into per-view bins after the loop,
                    # once the widest support in the run is known.
                    **({} if args.doppler_support_snr is None else
                       {"doppler_support": torch.tensor(
                           sorted(view_support), dtype=torch.long)}),
                }
                if elevation_on:
                    rae_cube = measured_rae_of(radar, cache_index)                     # (loops, range, angle, el)
                    # the person mask's peak of the Doppler-summed, elevation-summed |RAE| (plain FFT over the loops,
                    # DC kept): the 2D-array analogue of dsum_peak, the peak-doppler calibration's target
                    if args.person_level_calibration == "peak-doppler":
                        _dsum_rae = torch.fft.fft(measured_rae_unwindowed_of(radar, cache_index), dim=0).abs().sum(dim=(0, 3))   # (range, angle), no window
                        if _dsum_rae.shape != ra_mask.shape:
                            raise RuntimeError(f"the RAE view's (range, angle) grid {tuple(_dsum_rae.shape)} differs from the RA mask's {tuple(ra_mask.shape)}")
                        entry["views"][radar]["dsum_peak_rae"] = float(torch.where(ra_mask, _dsum_rae, torch.zeros_like(_dsum_rae)).max())
                        entry["views"][radar]["dsum_masked_rae"] = torch.where(ra_mask, _dsum_rae, torch.zeros_like(_dsum_rae)).cpu()
                        entry["views"][radar]["dsum_full_rae"] = _dsum_rae.cpu()
                        del _dsum_rae
                    rae_static = (rae_cube.mean(dim=0) if args.static_anchor == "dc-mean"
                                  else rae_cube[0]).abs().float()
                    rae_doppler = slow_time_to_doppler(rae_cube, dim=0).abs().index_select(
                        0, ra_doppler_index)[:, keep_range].float()                    # (planes, range kept, angle, el)
                    # stored at half with a per-view rms straight away: the cascade's cube is 8x the RA one
                    rae_rms = rae_doppler.square().mean().sqrt().clamp_min(1e-30)
                    if _rae_unshift_targets:
                        # both terms average uniformly over the (range, angle, elevation) cells, so
                        # undoing the angle shift on the target and skipping it on the prediction
                        # leaves every term and every denominator unchanged in real arithmetic
                        rae_static = torch.fft.ifftshift(rae_static, dim=-2)
                        rae_doppler = torch.fft.ifftshift(rae_doppler, dim=-2)
                    entry["views"][radar]["rae_target"] = rae_static.contiguous()
                    entry["views"][radar]["rae_doppler_target"] = (rae_doppler / rae_rms).to(torch.float16)
                    entry["views"][radar]["rae_doppler_rms"] = rae_rms
                    if args.static_term == "on":
                        entry["views"][radar]["dc_rae_target"] = rae_cube.mean(dim=0).abs().float().contiguous()
                    del rae_cube, rae_doppler
                if pw_on:
                    # the person-window term's target: the recording's loop-0 per-channel profile through the consumer's
                    # transform, cropped to a fixed box about the pelvis's cell, both azimuth sides (the beamformer's mirror:
                    # which side the consumer puts the person on is settled per sequence below from the recording's own
                    # window energy and printed); normalised by its rms over the window (a shape target)
                    with torch.no_grad():
                        hips = torch.as_tensor(joints[frame], dtype=torch.float32, device=device)[[0, 3]].mean(dim=0)
                        delta = hips - centres[radar]
                        g_r = float(delta.norm()); g_sine = float((delta @ (axes[radar] / axes[radar].norm().clamp_min(1e-12))) / max(g_r, 1e-6))
                        chans = measured_cube_channels_of(radar, cache_index).to(torch.complex64)        # (loops, tx, rx, K)
                        if args.person_window_planes == "resolved":
                            # the consumer's band kept plane by plane: (B, K, A, E). One normalisation over the
                            # whole cube below is what makes this term contain the micro-Doppler marginal.
                            Pm = slow_time_to_doppler(chans, dim=0)
                            parts = ([(_PW_DC_SCALE * _pw_view_fn(chans.mean(dim=0)))[None]] if _pw_keep_dc else [])
                            for _b0 in range(0, len(_pw_bins), 8):
                                parts.append(_PW_PLANE_SCALE * _pw_view_fn(Pm[_pw_bins[_b0:_b0 + 8]]))
                            full = torch.cat(parts, dim=0)
                            del Pm, parts
                        elif args.person_window_planes == "consumer":
                            full = (_PW_DC_SCALE * _pw_view_fn(chans.mean(dim=0)) if _pw_keep_dc else 0.0)   # the consumer's DC: the unwindowed mean
                            Pm = slow_time_to_doppler(chans, dim=0)                                     # (64, tx, rx, K): the trainer's planes' transform
                            for _b0 in range(0, len(_pw_bins), 8):
                                full = full + _PW_PLANE_SCALE * _pw_view_fn(Pm[_pw_bins[_b0:_b0 + 8]]).sum(dim=0)
                            del Pm
                            full = full[None]
                        else:
                            full = _pw_view_fn(chans[0])[None]                                          # (1, K, A, E): the loop-0 snapshot
                        del chans
                        K_, A_ = full.shape[1], full.shape[2]
                        r_ax = torch.as_tensor(_pw_r_axis, dtype=torch.float32, device=device); s_ax = torch.as_tensor(_pw_s_axis, dtype=torch.float32, device=device)
                        r0 = int(round(g_r / range_bin_spacing_m()))
                        ri = (r0 + pw_offsets_r).clamp(0, K_ - 1)
                        pw_entry = {"target": [], "mask": [], "range_index": ri, "angle_index": [], "energy": [], "weight": []}
                        rows_sq = full.index_select(1, ri).square(); rows_total = float(rows_sq.sum()); rows_cells = float(rows_sq.numel())   # the window's range rows, every azimuth
                        rows_median = float(rows_sq.flatten().median())
                        for side_sine in (g_sine, -g_sine):
                            a0 = int((s_ax - side_sine).abs().argmin()); ai = (a0 + pw_offsets_a).clamp(0, A_ - 1)
                            m = ((r_ax[ri] - g_r).abs()[:, None] <= args.person_window_range) & \
                                ((r_ax[ri][:, None] * (s_ax[ai][None, :] - side_sine)).abs() <= args.person_window_lateral)
                            crop = full.index_select(1, ri).index_select(2, ai)                          # (B, Rw, Aw, E)
                            mf = m.to(torch.float32)[None, ..., None]
                            energy = float((crop.square() * mf).sum())
                            # ONE rms over the whole (plane, range, azimuth, elevation) window, never per plane:
                            # per-plane normalisation would keep each plane's shape and discard the relative energy
                            # between planes, which is exactly the micro-Doppler content this term must carry.
                            cells = max(float(mf.sum()) * crop.shape[0] * crop.shape[-1], 1.0)
                            rms = (energy / cells) ** 0.5
                            pw_entry["target"].append((crop / max(rms, 1e-30)).to(torch.float16).contiguous())
                            pw_entry["mask"].append(m.to(torch.float32)); pw_entry["angle_index"].append(ai); pw_entry["energy"].append(energy)
                            n_win = max(float(mf.sum()) * crop.shape[-1], 1.0)          # mf carries the band axis, so this counts every plane
                            outside = (rows_median if args.person_window_snr_reference == "median"
                                       else max(rows_total - energy, 0.0) / max(rows_cells - n_win, 1.0))   # the rows' reference power OUTSIDE the window
                            snr = (energy / n_win) / max(outside, 1e-30)
                            pw_entry["weight"].append(1.0 if args.person_window_snr <= 1.0 else float(np.clip((snr - 1.0) / (args.person_window_snr - 1.0), 0.0, 1.0)))
                        entry["views"][radar]["pw"] = pw_entry
                        pw_side_energy[radar][0] += pw_entry["energy"][0]; pw_side_energy[radar][1] += pw_entry["energy"][1]
                        del full
            cache[frame] = entry
            if cache_index % 10 == 0 or cache_index == len(frames) - 1:
                print(
                    f"cached frame {cache_index + 1}/{len(frames)} ({frame})",
                    flush=True,
                )

    del readers, measured_cubes
    # The measured static scene in the calibration's quantity (the Doppler-summed unwindowed |RA| / |RAE|): a low quantile
    # per cell over the frames (reconstruction.clean_scene_maps' estimator: the person occupies any one cell in a minority
    # of frames) -- what the person-level calibration subtracts from both sides under --person-level-calibration-target person
    dsum_static: dict[str, dict[str, torch.Tensor]] = {}
    if args.person_level_calibration == "peak-doppler" and args.person_level_calibration_target == "person":
        for radar in RADARS:
            dsum_static[radar] = {}
            for key in ("dsum_full", "dsum_full_rae"):
                maps = [cache[frame]["views"][radar][key] for frame in frames if key in cache[frame]["views"][radar]]
                if maps:
                    stack = torch.stack(maps).float()                     # kthvalue: torch.quantile refuses > 2^24 elements
                    dsum_static[radar][key] = torch.kthvalue(stack, max(1, int(round(0.2 * len(maps)))), dim=0).values; del stack
    for frame in frames:
        for radar in RADARS:
            cache[frame]["views"][radar].pop("dsum_full", None); cache[frame]["views"][radar].pop("dsum_full_rae", None)
    # A per-point material is only meaningful if the point set persists. It
    # does when every frame's cloud is the same global set -- either a constant
    # cloud, or a fixed set with a per-frame `visible` mask, in which case the
    # global index runs to the same maximum every frame.
    # NOT the largest visible index: under a visibility mask the highest
    # visible vertex moves frame to frame, so that would reject a fixed set.
    # The size of the set each frame draws from is the invariant.
    _totals = {int(cache[frame]["global_point_total"]) for frame in frames}
    global_point_count = _totals.pop() if len(_totals) == 1 else None

    # `auto` is arm D of the change-3/4 comparison -- per-point material with
    # the area scale and the H/V delta off and 3DPS's mat_lr -- taken wherever
    # the geometry supports it, and the previously best shared configuration
    # where it does not. The bundle resolves together because its parts are not
    # independent: knobs off costs real accuracy against a shared 6-vector
    # (measured 2.108 -> 3.005 train_total at 3 epochs) and per-point loses
    # ~0.04 of centroid without the 1e-2 rate.
    if args.person_material == "auto":
        args.person_material = (
            "per-point" if global_point_count is not None else "shared")
        print(f"person material: auto -> {args.person_material}"
              + ("" if global_point_count is not None else
                 " (this geometry's point set does not persist across frames, "
                 "so a per-point material cannot be learned on it)"),
              flush=True)
    per_point_active = args.person_material == "per-point"
    if args.person_solid_angle_weight == "physical" and args.person_area_scale in ("auto", "learned"):
        args.person_area_scale = "unit"                     # the slot is the geometry; no learned scale on top of it
    if args.person_area_scale == "auto":
        args.person_area_scale = "frozen" if per_point_active else "learned"
    if args.person_radar_gain_delta == "auto":
        args.person_radar_gain_delta = "off" if per_point_active else "learned"
    if args.person_material_lr is None and per_point_active:
        args.person_material_lr = 1e-2

    if args.person_material == "per-point":
        if global_point_count is None:
            raise RuntimeError(
                "--person-material per-point needs a geometry whose point set "
                "persists across frames (a constant cloud, or a fixed global "
                "set with a per-frame `visible` mask). This sequence's cloud "
                "changes length frame to frame, so a per-point material would "
                "be seen by exactly one frame each and could not be learned.")
        material = PerPointHumanMaterial(
            person_checkpoint["shared_human_raw_material"].to(device),
            global_point_count,
            float(person_checkpoint["shared_human_area_scale"]),
        ).to(device)
    else:
        material = SharedHumanMaterial(
            person_checkpoint["shared_human_raw_material"].to(device),
            float(person_checkpoint["shared_human_area_scale"]),
        ).to(device)
        # One interface for both, so the render call sites do not branch.
        material.for_indices = lambda index: material.raw.unsqueeze(0).expand(
            index.shape[0], -1)
    if "person_radar_log_gain_delta" in person_checkpoint:
        with torch.no_grad():
            material.radar_log_gain_delta.copy_(
                person_checkpoint["person_radar_log_gain_delta"].to(device)
            )
    with torch.no_grad():
        if args.person_area_scale == "unit":
            material.log_area_scale.zero_()
        if args.person_radar_gain_delta == "off":
            material.radar_log_gain_delta.zero_()
    if args.person_area_scale != "learned":
        material.log_area_scale.requires_grad_(False)
    if args.person_radar_gain_delta in ("off", "frozen"):        # frozen: the person checkpoint's value (the person stage's
        material.radar_log_gain_delta.requires_grad_(False)       # calibrated H/V balance), not trained
    print(f"person area scale: {args.person_area_scale} "
          f"({float(material.log_area_scale.detach().exp()):.6g}); "
          f"radar gain delta: {args.person_radar_gain_delta} "
          f"({float(material.radar_log_gain_delta.detach()):+.6g})", flush=True)

    # Only training frames get learnable normals (and, for point-normal,
    # scalars); evaluation-only frames keep stored data.
    frame_normals = {
        frame: cache[frame]["stored_normals"].detach().cpu()
        for frame in training_frames
    }
    model = VelocityFieldModel(
        torch.from_numpy(velocities), args.velocity_parameterization, frame_normals,
        temporal_bands=args.temporal_bands,
        residual_bound=args.velocity_residual_bound if args.velocity_parameterization == "temporal-basis" else 0.0,
        residual_relative=args.velocity_residual_relative,
        residual_per_height=args.velocity_residual_per_height,
        # the pelvis (joint 0) above the lowest joint of the frame (the feet / the support)
        pelvis_height=torch.from_numpy(joints[:, 0, 2] - joints[:, :, 2].min(axis=1)).float(),
    ).to(device)
    initial_field = model.full_field().detach().clone()
    if model.bounded:
        print(f"velocity field: the mesh's kinematics x a learned per-joint scale + a residual bounded by {model.residual_bound} m/s "
              f"+ {model.residual_relative} x the joint's kinematic speed + {model.residual_per_height} x the pelvis height "
              f"(median {float(model.pelvis_height.median()):.2f} m), per component", flush=True)

    if args.velocity_parameterization == "joint":
        velocity_parameters = [model.joint_velocity_field]
    elif args.velocity_parameterization == "temporal-basis":
        velocity_parameters = [model.temporal_coefficients] + ([model.joint_scale] if model.bounded else [])
    else:
        velocity_parameters = list(model.point_scalars.values())
    # Sequenced training is a schedule, not a loss change: the loss stays
    # exactly the two L2 terms throughout. With --geometry-epochs N > 0, the
    # first N epochs update only material and normals (velocity frozen at
    # the camera init), the remaining epochs update only velocity on the
    # then-frozen geometry and level. This prevents the level-spread
    # coupling: a level that cannot move cannot be traded against Doppler
    # width. N = 0 is the original joint schedule, unchanged.
    if not 0 <= args.geometry_epochs <= args.epochs:
        raise ValueError("--geometry-epochs must lie within [0, --epochs]")
    # Background gains train with the geometry/material group: they are
    # material-class parameters of the scene, learning from all 600 frames'
    # RA gradients for the first time (the old background was a frozen,
    # detached constant with provably zero gradient).
    # E8/P1: the 590 per-frame normals are one fused parameter and are stepped
    # by `_FusedRowAdam`, not by this list — see the class docstring for why
    # handing the fused tensor to `torch.optim.Adam` would change the run.
    material_parameters = [
        parameter for parameter in material.parameters()
        if parameter.requires_grad
    ]
    person_material_groups = []
    if args.person_material_lr is not None:
        person_material_groups = [
            {"params": material_parameters, "lr": args.person_material_lr}
        ]
        material_parameters = []
    # The noise floor is now part of the differentiable forward model rather
    # than a constant estimated from a previous render. See
    # the noise-model forward plan, part 2, for what is and is not in the
    # autograd graph; hupr_noise_model.py for why a global scale cannot work.
    noise_model = None
    if args.noise_model in ("on", "measured"):
        if args.noise_envelope_calibration is None:
            # The default: the prior is this sequence's own off-person floor
            # (the median over frames of the windowed |RD|^2 outside the
            # person's cells, built while the targets were cached), so the
            # noise model starts at what this recording's empty cells show
            # and learns its scale and range shape from there. Nothing is
            # read from a calibration file or another sequence.
            if noise_prior_power is None and noise_prior_frames:
                noise_prior_power = torch.stack([
                    torch.stack(noise_prior_frames[radar]).median(dim=0).values for radar in RADARS])
            if noise_prior_power is None:
                raise RuntimeError("the sequence noise prior was not built with the target cache")
            prior = np.sqrt(noise_prior_power.numpy().astype(np.float64)).astype(np.float32)
            prior_source = "this sequence's off-person floor"
        else:
            # An explicit envelope file (the ablation): its Doppler-range
            # shape describes the noise PROCESS measured unwindowed, so what
            # is added to windowed rendered power is that power smoothed by
            # the window's 3-tap kernel along Doppler and range.
            calibration = json.loads(args.noise_envelope_calibration.read_text())
            prior = np.stack([
                np.asarray(
                    calibration["radars"][radar]["normalized_doppler_range_profile"],
                    dtype=np.float32,
                )
                for radar in RADARS
            ])
            if prior.shape[1:] != (TDM_LOOPS, 256):
                raise ValueError(
                    f"envelope prior must be (radars,{TDM_LOOPS},256), got {prior.shape}")
            prior = windowed_process_envelope(prior)
            prior_source = args.noise_envelope_calibration.name
        # Zero the guarded near field so the floor cannot be spent there.
        prior = prior * range_keep_mask()[None, None, :].astype(np.float32)
        noise_model = LearnedNoiseFloor(
            prior, n_coeff=args.noise_range_coefficients,
            initial_scale=None,
            device=device,
        ).to(device)
        print(f"noise model: learned floor, "
              f"{sum(p.numel() for p in noise_model.parameters())} parameters "
              f"({args.noise_range_coefficients} range coefficients per radar), "
              f"prior from {prior_source}", flush=True)

    # The diffuse floor, as TRAINING sees it. The renderer draws a
    # realization; a differentiable trainer cannot use one, but it can add the
    # process's expected power to its predicted power. Registering it on the
    # noise model means `plane_power`, `microdoppler_marginal` and
    # `microdoppler_marginal_batched` all pick it up, so the per-unit and the
    # three batched paths cannot disagree about it.
    #
    # Without this the trainer's only noise is the learned floor, so the
    # materials, normals and velocities are fitted against a forward model
    # missing a process the render then adds on top. The test ledger that kept
    # that gap shut went with `tests/` in the deployment strip.
    if args.diffuse_floor_calibration is not None:
        from mm4DPS.scene import PhaseCoherentDiffuseFloor

        if noise_model is None:
            raise RuntimeError(
                "--diffuse-floor-calibration needs --noise-model on: the "
                "diffuse floor's expected power is registered on the noise "
                "model so every consumer of it picks the floor up, and with "
                "the noise model off there is nothing to register it on")
        calibration = json.loads(args.diffuse_floor_calibration.read_text())
        scene_positions = state["positions"].detach().cpu().numpy()
        extra = []
        for radar_index, radar in enumerate(RADARS):
            centre = config_centre_m(configs[radar]).astype(np.float64)
            support = scene_range_support(
                scene_positions, centre, range_bin_spacing_m(),
                smoothing_bins=args.diffuse_floor_support_smoothing_bins)
            selected = calibration["selected_range_profile_amplitudes"][radar]
            level = 0.5 * (float(selected["near"]) + float(selected["far"]))
            # The renderer applies --diffuse-floor-scale on top of the
            # calibrated level (`range_amplitudes *= args.diffuse_floor_scale`),
            # so the trainer must too, or the two model floors of different
            # strength -- the very mismatch this change exists to remove.
            amplitudes = support_to_range_amplitudes(
                support, level) * args.diffuse_floor_scale
            expectation = measure_diffuse_floor_expectation(
                PhaseCoherentDiffuseFloor(
                    rasterizers[radar].tx_positions.detach().cpu().numpy(),
                    rasterizers[radar].rx_positions.detach().cpu().numpy(),
                    centre, array_axis(configs[radar]),
                    wavelength_m=299792458.0 / rasterizers[radar].center_freq,
                    range_bin_spacing_m=range_bin_spacing_m(),
                    active_range_bins=support > 0,
                    scatterers_per_bin=args.diffuse_floor_scatterers_per_bin,
                    temporal_correlation=float(
                        calibration["selected_temporal_correlation"]),
                    seed=args.diffuse_floor_seed + radar_index,
                ),
                amplitudes, draws=args.diffuse_floor_expectation_draws,
            )
            # Averaged over angle, to match the (radar, Doppler, range) shape
            # every consumer of `power()` expects. Measured on the real HuPR
            # array the angle marginal has CV 0.25 (max/mean 1.38x), which is
            # negligible against a rendered floor currently wrong by up to 40x
            # in the empty bins -- and it buys one registration point instead
            # of four call sites that could drift apart.
            cube = diffuse_floor_expected_cube(expectation)
            extra.append(torch.as_tensor(
                cube.mean(axis=2), dtype=torch.float32, device=device))
            print(f"diffuse floor [{radar}]: {int((support > 0).sum())}/256 "
                  f"supported range bins from the scene, level {level:.5f}, "
                  f"expectation from {expectation['draws']} draws, "
                  f"mean power {float(extra[-1].mean()):.4e}", flush=True)
        noise_model.set_extra_power(torch.stack(extra))

    def noise_md_marginal(radar: str, view: dict) -> torch.Tensor | float:
        if noise_model is None:
            return 0.0
        return noise_model.microdoppler_marginal(
            RADARS.index(radar), view["md_mask"], view["md_rows"],
            motion_doppler,
        )

    geometry_parameters = material_parameters
    # The scene's materials and normals get their own groups at 3DPS's rates
    # rather than riding the person's geometry lr.
    background_groups = [
        {"params": [background.raw_materials], "scene": True,
         "lr": args.background_material_lr, "clip": args.background_material_clip},
        {"params": [background.rotations], "scene": True,
         "lr": args.background_rotation_lr, "clip": args.background_rotation_clip},
    ]

    def build_optimizer(phase: str) -> torch.optim.Adam:
        """One optimizer per phase, so a phase starts with fresh Adam state
        (moments accumulated while a group idled at lr 0 misscale its first
        real steps otherwise)."""
        # The learned noise floor rides its own group: it is ~9 parameters
        # per radar against the scene's millions, and it needs a much larger
        # step than the geometry to move a log-scale at all.
        noise_groups = ([] if noise_model is None or args.noise_model == "measured" else [
            {"params": list(noise_model.parameters()), "lr": args.noise_lr},
        ]) + ([{"params": [absolute_log_gain], "lr": args.gain_lr}] if gain_learned and not gain_frozen_after_init else []
             ) + ([{"params": [joint_log_gain], "lr": args.gain_lr}] if args.joint_gains == "learned" else [])
        # One fused kernel over every parameter instead of a launch chain
        # per tensor (1.1 -> ~0.3 ms a step); the same update to float
        # rounding (`--adam fused|eager`).
        fused = {"fused": args.adam == "fused"}
        if phase == "joint-update":
            return torch.optim.Adam([
                {"params": velocity_parameters, "lr": args.velocity_lr},
                {"params": geometry_parameters, "lr": args.geometry_lr},
            ] + person_material_groups + background_groups + noise_groups, **fused)
        if phase == "geometry":
            return torch.optim.Adam(
                [{"params": geometry_parameters, "lr": args.geometry_lr}]
                + person_material_groups + background_groups + noise_groups, **fused)
        return torch.optim.Adam(velocity_parameters, lr=args.velocity_lr, **fused)

    def build_normals_optimizer(phase: str) -> _FusedRowAdam | None:
        """The fused normals' own Adam, rebuilt per phase like the others.

        `None` in the velocity phase: the reference leaves the normals out of
        that phase's optimizer entirely, so they are neither stepped nor
        zeroed and their gradients simply accumulate unused.
        """
        if phase == "velocity":
            return None
        return _FusedRowAdam(model.raw_normals_flat, args.geometry_lr)

    def phase_of(epoch: int) -> str:
        if args.geometry_epochs == 0:
            return "joint-update"
        return "geometry" if epoch < args.geometry_epochs else "velocity"

    optimizer = build_optimizer(phase_of(0))
    normals_optimizer = build_normals_optimizer(phase_of(0))

    # The optimizer is rebuilt per phase, so the base rates are re-read from
    # whichever optimizer is live rather than captured once up front.
    def base_learning_rates(
        opt: torch.optim.Adam, normals_opt: "_FusedRowAdam | None",
    ) -> tuple[list[float], float | None]:
        return ([group["lr"] for group in opt.param_groups],
                None if normals_opt is None else normals_opt.lr)

    def learning_rate_scale(epoch: int) -> float:
        """Constant 1.0, then cosine to `--lr-min-factor` (no warmup).

        `train_gaussian.get_lr_scale` with warmup_iters=0, inlined rather than
        imported so the trainer does not pull in that module's import cost.
        """
        if args.lr_schedule == "none":
            return 1.0
        decay_start = int(args.lr_decay_start_frac * args.epochs)
        if epoch < decay_start:
            return 1.0
        progress = min((epoch - decay_start) / max(args.epochs - decay_start, 1), 1.0)
        return args.lr_min_factor + 0.5 * (1.0 - args.lr_min_factor) * (
            1.0 + math.cos(math.pi * progress))

    def apply_learning_rate(
        opt: torch.optim.Adam, normals_opt: "_FusedRowAdam | None",
        bases: tuple[list[float], float | None], scale: float,
    ) -> None:
        # Adam is stepped eagerly, outside the captured graph (see P3 above),
        # so writing `group["lr"]` between epochs takes effect on the very
        # next step and needs no capturable-Adam arithmetic.
        group_bases, normals_base = bases
        for group, base in zip(opt.param_groups, group_bases):
            group["lr"] = base * scale
        if normals_opt is not None and normals_base is not None:
            normals_opt.lr = normals_base * scale

    base_lrs = base_learning_rates(optimizer, normals_optimizer)

    def frame_components(
        frame: int, radar: str, velocity_source_frame: int | None = None,
    ) -> tuple[dict, torch.Tensor]:
        """Splat components and bistatic path rates for one frame/radar."""
        entry = cache[frame]
        rasterizer = rasterizers[radar]
        count = len(entry["points"])
        normals = model.normals(frame, entry["stored_normals"])
        components = render_factorized(
            entry["points"], normals,
            person_areas(
                material.areas_for_points(count), entry["weights"] @ joint_gains(),
                entry["views"][radar]["solid_angle"],
                material.radar_gain(radar)),
            material.for_indices(entry["global_index"]), rasterizer,
            reparameterize_torch,
            detach_phase=True, shadow_mask=None, bsdf_mode="full",
            disabled_components=None, use_cuda_kernels=True,
            return_splat_components=True,
        )
        point_velocity = model.point_velocities(
            frame, entry["weights"], entry["stored_normals"],
            velocity_source_frame=velocity_source_frame,
        )
        rates = surface_bistatic_path_rates(
            entry["points"], point_velocity,
            rasterizer.tx_positions, rasterizer.rx_positions,
        )
        return components, rates

    # ------------------------------------------------------------------
    # Batched unit rendering (L3/L4/L6): one render_factorized call and one
    # K2 launch for every (frame, radar) in the batch.
    #
    # A "unit" is one (frame, radar, velocity-source) render. The two radars
    # differ in nothing but their TX/RX element positions — same antenna
    # patterns, boresights, chirp slope, sample rate and calibration
    # constants — which is asserted below rather than assumed, so the radar
    # axis folds into the point axis exactly like the frame axis does. A
    # batch-6 step becomes one BSDF call over ~24.5K points and one slow-time
    # splat, instead of twelve of each.
    # ------------------------------------------------------------------
    reference_rasterizer = rasterizers[RADARS[0]]
    for radar in RADARS[1:]:
        other = rasterizers[radar]
        for attribute in ("n_tx", "n_rx", "K", "center_freq", "slope",
                          "sample_rate", "radar_constant", "rx_dBFS_scale",
                          "adc_scale"):
            if getattr(reference_rasterizer, attribute) != getattr(other, attribute):
                raise RuntimeError(
                    f"radars disagree on {attribute}; the batched render path "
                    f"folds the radar axis into the point axis and requires "
                    f"identical chirp/array constants")
        for attribute in ("tx_boresights", "rx_boresights"):
            if not torch.equal(getattr(reference_rasterizer, attribute),
                               getattr(other, attribute)):
                raise RuntimeError(f"radars disagree on {attribute}")
        for antenna in ("tx_antenna", "rx_antenna"):
            first, second = (getattr(reference_rasterizer, antenna),
                             getattr(other, antenna))
            if not (torch.equal(first.E, second.E)
                    and torch.equal(first.H, second.H)
                    and first.C_scale == second.C_scale):
                raise RuntimeError(f"radars disagree on {antenna}")
    tx_array_stack = torch.stack([rasterizers[r].tx_positions for r in RADARS])
    rx_array_stack = torch.stack([rasterizers[r].rx_positions for r in RADARS])
    fixed_gain_stack = (None if gain_learned else torch.tensor(
        [absolute_gains[radar] for radar in RADARS], device=device))

    def gain_stack() -> torch.Tensor:
        """The per-radar absolute gain, ``(radars,)``: the learned one or the given constants."""
        return torch.exp(absolute_log_gain) if gain_learned else fixed_gain_stack

    def gain_values() -> dict[str, float]:
        return {radar: float(v) for radar, v in zip(RADARS, gain_stack().detach().cpu())}

    def unit_geometry(units: list[tuple[int, int, int | None]]) -> dict:
        """The per-point inputs of one render, for a list of units.

        Split out from the render so the evaluation can reuse one render's
        geometry against its own splat-geometry cache; positions, normals,
        materials and array geometry are what one render call consumes.
        """
        counts = [len(cache[frame]["points"]) for frame, _, _ in units]
        total = sum(counts)
        positions = torch.cat([cache[frame]["points"] for frame, _, _ in units])
        # E8/P1: one `index_select` over the fused normals parameter instead of
        # one slice, norm, clamp and divide per unit followed by a 12-way cat.
        # Per row the arithmetic is `model.normals()`'s, unchanged.
        normals = model.normals_for_units(
            [frame for frame, _, _ in units],
            [cache[frame]["stored_normals"] for frame, _, _ in units])
        point_gains = torch.cat([
            cache[frame]["weights"]
            * cache[frame]["views"][RADARS[radar_index]]["solid_angle"][:, None]
            for frame, radar_index, _ in units
        ]) @ joint_gains()
        radar_gains = torch.stack([material.radar_gain(radar) for radar in RADARS])
        if _FOLDED_ABSOLUTE_GAIN:
            # Squared: the areas slot enters the amplitude under a sqrt.
            radar_gains = radar_gains * gain_stack().square()
        per_point_radar_gain = torch.cat([
            radar_gains[radar_index].expand(count)
            for (_, radar_index, _), count in zip(units, counts)
        ])
        # Same three factors in the same order as the per-unit path, so the
        # areas are bitwise identical element by element.
        areas = (material.areas_for_points(total) * point_gains
                 * per_point_radar_gain)
        return {
            "units": list(units),
            "counts": counts,
            "positions": positions,
            "normals": normals,
            "areas": areas,
            "materials": material.for_indices(torch.cat([
                cache[frame]["global_index"] for frame, _, _ in units
            ])),
            "tx_positions": torch.cat([
                tx_array_stack[radar_index].expand(count, -1, -1)
                for (_, radar_index, _), count in zip(units, counts)
            ]),
            "rx_positions": torch.cat([
                rx_array_stack[radar_index].expand(count, -1, -1)
                for (_, radar_index, _), count in zip(units, counts)
            ]),
        }

    def unit_render(geometry: dict,
                    geometry_cache: dict | None = None,
                    cache_psf: bool = True) -> tuple[dict, torch.Tensor]:
        """One render_factorized call over every unit's points concatenated.

        ``geometry_cache``: the evaluation renders the *same* fixed chunks
        every epoch — same frames, same positions — so their splat geometry
        (distances, antenna gains, carrier phasors, PSF weights, indices) is
        constant and can be filled once and reused; only the BSDF amplitude
        (normals, materials, gains) moves. Same mechanism and gates as the
        background's per-step cache.
        """
        # With the row geometry built, the units' splat geometry comes out
        # of it (and the sorted Doppler route with it) instead of a per-chunk
        # cache: the same cached values, gathered by the units' rows.
        row_cache = (unit_row_cache(geometry) if row_geometry and "units" in geometry
                     else None)
        components = render_factorized(
            geometry["positions"], geometry["normals"], geometry["areas"],
            geometry["materials"], reference_rasterizer, reparameterize_torch,
            detach_phase=True, shadow_mask=None, bsdf_mode="full",
            disabled_components=None, use_cuda_kernels=True,
            return_splat_components=True,
            tx_positions=geometry["tx_positions"],
            rx_positions=geometry["rx_positions"],
            geometry_cache=row_cache if row_cache is not None else geometry_cache,
            cache_psf=cache_psf,
            carrier_only=row_cache is not None and sorted_doppler,
        )
        point_to_unit = torch.cat([
            torch.full((count,), index, dtype=torch.long, device=device)
            for index, count in enumerate(geometry["counts"])
        ])
        return components, point_to_unit[components["point_indices"]]

    def unit_rates(
        geometry: dict, units: list[tuple[int, int, int | None]],
    ) -> torch.Tensor:
        """Bistatic path rates for a list of units on an existing geometry."""
        velocities = torch.cat([
            model.point_velocities(
                frame, cache[frame]["weights"], cache[frame]["stored_normals"],
                velocity_source_frame=source,
            )
            for frame, _, source in units
        ])
        return surface_bistatic_path_rates_per_point(
            geometry["positions"], velocities,
            geometry["tx_positions"], geometry["rx_positions"],
        )

    def unit_components(units: list[tuple[int, int, int | None]]) -> tuple:
        """Splat components, path rates and per-path unit index for a batch."""
        geometry = unit_geometry(units)
        components, unit_of_path = unit_render(geometry)
        return (components, unit_rates(geometry, units), unit_of_path,
                geometry["counts"])

    def current_backgrounds() -> torch.Tensor:
        """Both radars' assembled backgrounds, stacked — one K1 pair.

        E5: `unit_scaled_profiles` calls `current_background(radar)` for both
        radars on every invocation, and the evaluation pass invokes it once per
        velocity source under `no_grad` with the gains unchanged — so 4 of every
        5 K1 launches recomputed an identical tensor, 100 per pass where 20
        suffice. Callers that make several invocations against one parameter
        state compute this once and pass it in.
        """
        return torch.stack([current_background(radar) for radar in RADARS])

    def dc_anchor_units(background: torch.Tensor, components: dict, rates: torch.Tensor,
                        unit_of_path: torch.Tensor, n_units: int) -> torch.Tensor | None:
        """The DC (slow-time mean) plane of the same render, ``(units, tx, rx, K)``, or None without the static term.

        The background enters as given (gain-scaled and coupled exactly like the anchor's), so the returned plane is
        on the same footing as ``ra_profile``. The dummy bin's plane is rendered and dropped."""
        if not static_on:
            return None
        dummy = (torch.full((n_units, 1), DC_DUMMY_BIN, dtype=torch.long, device=device)
                 if view_plane_bins is not None else [DC_DUMMY_BIN])
        anchor, _ = doppler_planes_units_dc(background, components, rates, unit_of_path, n_units, dummy)
        return anchor

    def unit_scaled_profiles(
        units: list[tuple[int, int, int | None]],
        components: dict,
        rates: torch.Tensor,
        unit_of_path: torch.Tensor,
        radar_backgrounds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gain-scaled RA profile and Doppler planes from ready components.

        E1: the renderer is asked for exactly the planes the objective reads —
        loop 0 (the RA term's raw slow-time sample) and the Doppler bins in
        `motion_bins` — instead of all 64 loop planes, of which 51 were built
        with atomics, angle-FFT'd, magnitude'd and discarded. The bin set is an
        *argument*, the way `range_keep_mask()` parameterizes the range region
        of interest: the caller says which bins it wants, the renderer produces
        those bins exactly and knows nothing about the loss.

        Returns `(ra_profile (units,3,4,K), doppler_planes (units,bins,3,4,K))`.
        """
        radar_index = torch.tensor(
            [radar_index for _, radar_index, _ in units], device=device,
        )
        if scene_visibility is not None:
            # One background PER UNIT: the mask makes the ASSEMBLY
            # frame-dependent, so E5's share-across-units shortcut no longer
            # applies to it. `radar_index` keeps meaning the radar, because the
            # absolute gain below is still per radar; only the background
            # selection changes.
            #
            # R091: the RENDER is still shared. Visibility is a per-point gain
            # and enters `static_assemble`, never `render_factorized`, so the
            # splat components are identical for every frame at one parameter
            # state. Calling `current_background` per unit re-rendered them
            # 24 times per radar per step -- a 20000-point BSDF re-evaluation
            # and a 28 MB (15 x 232608) contribution pair thrown away each
            # time. Rendering once per radar and assembling per frame is the
            # same arithmetic on the same tensors: `assemble_background` gets
            # the identical `(components, gains)` pair the per-frame path
            # built, so each row comes back bitwise unchanged.
            splats = {radar: background_splat(radar) for radar in RADARS}
            per_unit = torch.stack([
                assemble_background(RADARS[index], splats[RADARS[index]], frame)
                for frame, index, _ in units])
        elif radar_backgrounds is None:
            stacked = torch.stack(
                [current_background(radar) for radar in RADARS]
            )
            per_unit = None
        else:
            stacked, per_unit = radar_backgrounds, None
        if _FOLDED_ABSOLUTE_GAIN:
            # The dynamic contributions already carry the absolute gain
            # (folded into the per-point gain); only the tiny background
            # profiles still need it, before K2 adds them to loop 0.
            if per_unit is None:
                per_unit = (stacked * gain_stack()[:, None, None, None]
                            )[radar_index]
            else:
                per_unit = per_unit * gain_stack()[radar_index][
                    :, None, None, None]
            if coupling_stack is not None:
                per_unit = per_unit + coupling_stack[radar_index]
            ra_profile, planes = doppler_planes_units(
                per_unit, components, rates, unit_of_path,
                len(units), plane_bins_for(units),
            )
            return ra_profile, planes, dc_anchor_units(per_unit, components, rates, unit_of_path, len(units))
        background = stacked[radar_index] if per_unit is None else per_unit
        ra_profile, planes = doppler_planes_units(
            background, components, rates, unit_of_path, len(units),
            plane_bins_for(units),
        )
        gain = gain_stack()[radar_index]
        ra_profile = gain[:, None, None, None] * ra_profile
        dc_profile = dc_anchor_units(background, components, rates, unit_of_path, len(units))
        if dc_profile is not None:
            dc_profile = gain[:, None, None, None] * dc_profile
        if coupling_stack is not None:
            ra_profile = ra_profile + coupling_stack[radar_index]
            if dc_profile is not None:
                dc_profile = dc_profile + coupling_stack[radar_index]
        return ra_profile, gain[:, None, None, None, None] * planes, dc_profile

    def unit_profiles(
        units: list[tuple[int, int, int | None]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gain-scaled RA profile and Doppler planes for a batch of units."""
        components, rates, unit_of_path, _ = unit_components(units)
        return unit_scaled_profiles(units, components, rates, unit_of_path)

    def descend(objective: torch.Tensor) -> None:
        """The one and only backward.

        Both step paths funnel through here so the objective that receives a
        gradient is written once and cannot drift between them.

        `objective` is the summed two-term data loss and nothing else, and
        nothing is added to it here. A noise-floor anchor prior -- a prior on
        nine parameters per radar, gated on a weight that defaulted to zero --
        used to be added at this point. The sweep in
        docs/GVHMR_DIFFUSE_FLOOR_RESULTS.md measured weights 0, 1, 10 and 100
        to saturate the range-correction clamp identically, with 100 the
        worst held-out (0.9815 against 0.9770 at zero), so the term was
        removed rather than left wired up. The objective is two terms.
        """
        (0.5 * objective).backward()

    def losses_from_profiles(
        frame: int, radar: str, profiles: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Two-term loss for one frame given its gain-scaled (loops,3,4,K) profiles."""
        view = cache[frame]["views"][radar]
        ra_prediction = range_profiles_to_ra_magnitude(
            profiles[0].real, profiles[0].imag,
        )
        doppler_cube = slow_time_ra_to_doppler_magnitude(
            slow_time_profiles_to_ra_complex(profiles.real, profiles.imag),
        )
        # Forward-model completeness: the pipeline's scored renders carry
        # calibrated injected noise, so the training prediction carries the
        # same expected floor (zeros unless a reference is configured) —
        # without it, absolute L2 rewards dimming and spreading the person
        # to imitate the measured floor.
        md_prediction = (
            microdoppler_marginal(
                doppler_cube, view["md_mask"], view["md_rows"], motion_doppler,
            ).clamp_min(0.0) + noise_md_marginal(radar, view)
        ).sqrt()
        # `doppler_cube` already *is* the plane magnitudes this path needs:
        # |Doppler x range x angle|, fftshifted and clutter-removed. Selecting
        # the non-DC bins costs nothing here, so the single-frame reference
        # carries the same objective as the batched paths and stays a valid
        # arbiter for them.
        ra_plane_prediction = doppler_cube.index_select(
            0, ra_doppler_index,
        )[:, keep_range]
        if noise_model is not None:
            # Power adds incoherently; the plane term is a magnitude, so the
            # floor enters under the square root exactly as it does for MD.
            ra_plane_prediction = (
                ra_plane_prediction.square()
                + noise_model.plane_power(
                    RADARS.index(radar), ra_doppler_index, keep_range,
                )[..., None]
            ).sqrt()
        total, ra_l2, md_l2 = two_term_loss(
            ra_prediction[keep_range], view["ra_target"][keep_range],
            md_prediction, view["md_target"], args.microdoppler_weight,
            ra_loss=args.ra_loss,
            ra_plane_prediction=ra_plane_prediction,
            ra_plane_target=plane_targets_for(
                [(frame, RADARS.index(radar), None)]
            )[0],
        )
        return total, ra_l2, md_l2, md_prediction

    # ------------------------------------------------------------------
    # Batched loss chain (L5). The rank-fixed helpers above stay as the
    # pinned reference; these run the identical transforms over a leading
    # unit axis so all twelve units share one set of kernels.
    #
    # `keep_range` and `motion_doppler` are the same masks for every unit, so
    # they are resolved to index tensors once here and applied with
    # index_select — the boolean form re-ran nonzero() (and a host round trip)
    # on every use. `range_rows` genuinely varies per unit and becomes a
    # multiply-then-sum inside microdoppler_marginal_batched.
    # ------------------------------------------------------------------
    keep_index = keep_range.nonzero(as_tuple=True)[0]
    motion_index = motion_doppler.nonzero(as_tuple=True)[0]
    # The Doppler bins the renderer is asked to produce (E1). This is the
    # *only* place the loss tells the renderer what it reads, and it is the
    # same list the marginal indexes with, so nothing the caller later reads
    # can be missing from what was rendered.
    motion_bins = motion_index.tolist()
    if motion_bins != official_motion_doppler_mask().nonzero()[0].tolist():
        raise RuntimeError(
            "the rendered Doppler bin set must be exactly the bins the "
            "micro-Doppler mask reads")
    # The renderer is now asked for every non-DC plane, because the RA term
    # reads all of them. The MD term still reads only its own motion bins, so
    # it selects them by position inside the rendered set rather than by
    # Doppler index. Checked, not assumed: a mismatch here would silently
    # score the micro-Doppler on the wrong velocities.
    def plane_bins_for(units: list[tuple[int, int, int | None]]):
        """What the renderer should build for this batch.

        The flat non-DC list when every unit shares it, otherwise the batch's
        own (units, K) rows -- each frame renders the Doppler bins its own
        measurement puts above the noise floor.
        """
        if view_plane_bins is None:
            return ra_doppler_bins
        return view_plane_bins.index_select(0, view_rows_for(units))

    view_cache: dict[tuple[int, ...], dict] = {}
    # Splat-geometry caches for the evaluation's recurring render chunks,
    # keyed by the chunk's frame list. ~120 MB per 15-frame chunk with the
    # slim (cache_psf=False) contents.
    eval_geometry_caches: dict[tuple[int, ...], dict] = {}
    # The bound is the single largest term in the trainer's resident VRAM --
    # measured 8.2 GiB of a 13.3 GiB working set at 44 chunks -- so it is a
    # knob, not a constant. HUPR_EVAL_GEOMETRY_CACHE=0 disables the cache
    # entirely; the pass then recomputes the position-only constants, which is
    # exactly what it did before the cache existed. Cached and uncached both
    # run the same compiled graph on the same inputs.
    _EVAL_CACHE_BOUND = int(os.environ.get("HUPR_EVAL_GEOMETRY_CACHE", "44"))

    def unit_views(units: list[tuple[int, int, int | None]]) -> dict:
        """Per-unit targets and masks, stacked once per distinct unit list."""
        key = tuple((frame, radar_index) for frame, radar_index, _ in units)
        stacked = view_cache.get(key)
        if stacked is None:
            views = [
                cache[frame]["views"][RADARS[radar_index]]
                for frame, radar_index, _ in units
            ]
            radar_index = torch.tensor(
                [radar_index for _, radar_index, _ in units], device=device,
            )
            stacked = {
                "ra_mask": torch.stack([view["ra_mask"] for view in views]),
                "md_mask": torch.stack([view["md_mask"] for view in views]),
                "pw_box_mask": torch.stack([view["pw_box_mask"] for view in views]),
                "md_rows": torch.stack([view["md_rows"] for view in views]),
                "range_rows": torch.stack(
                    [view["range_rows"] for view in views]
                ).to(torch.float32),
                # the same two on the retained range bins, for the md term
                # read out of the plane spectrum
                "ra_mask_kept": torch.stack([view["ra_mask"] for view in views]).index_select(1, keep_index),
                "range_rows_kept": torch.stack(
                    [view["range_rows"] for view in views]
                ).to(torch.float32).index_select(1, keep_index),
                "ra_target": torch.stack(
                    [view["ra_target"] for view in views]
                ).index_select(1, keep_index),
                "md_target": torch.stack([view["md_target"] for view in views]),
                # The noise floor is *not* cached here: it is calibrated after
                # this closure is defined and rebound per radar, so it is
                # gathered fresh on every call.
                "radar_index": radar_index,
            }
            if _md_support_within_keep:
                stacked["ra_mask_keep"] = stacked["ra_mask"].index_select(
                    1, keep_index)
                stacked["range_rows_keep"] = stacked[
                    "range_rows"].index_select(1, keep_index)
            # Bounded. The evaluation passes reuse a fixed set of chunk unit
            # lists every epoch — that is what this is for. (The five velocity
            # sources this once served were removed 2026-08-25; the reuse that
            # remains is across epochs, not within a pass.) A training step's
            # list is freshly shuffled and never recurs, so once the bound is
            # reached those misses simply rebuild. ~1.3 MB per entry.
            if len(view_cache) < 64:
                view_cache[key] = stacked
        return stacked

    def md_prediction_units(
        units: list[tuple[int, int, int | None]], planes: torch.Tensor,
    ) -> torch.Tensor:
        """The (units, motion bins) micro-Doppler prediction, batched.

        `planes` are already the requested Doppler bins (E1): the loop axis,
        the clutter removal and the Doppler FFT were folded into the splat, so
        only the beamforming is left. Group E's Doppler-first ordering is what
        this generalizes — that ran the angle FFT on 20 of 64 Doppler bins;
        this never builds the other 51 planes at all.
        """
        views = unit_views(units)
        # The RA term needs every non-DC plane; the MD term reads its 13
        # motion bins only. The two counts differ by ~5x, so dispatching on
        # the shape is unambiguous -- and anything else is an error rather
        # than a silent reading of the wrong planes.
        if planes.shape[1] == plane_count:
            planes = motion_planes_of(planes, units)
        elif planes.shape[1] != len(motion_bins):
            raise RuntimeError(
                f"rendered {planes.shape[1]} Doppler planes; the loss reads "
                f"either {plane_count} (RA + MD) or "
                f"{len(motion_bins)} (MD only)")
        # The fused chain is the same transform (pinned in the suite); the
        # reference stays reachable through HUPR_FUSED_LOSS=0.
        doppler_cube = (
            profiles_to_ra_magnitude_fused(planes.real, planes.imag)
            if _FUSED_LOSS else
            doppler_planes_to_motion_magnitude(planes.real, planes.imag))
        noise_floor = (
            noise_model.microdoppler_marginal_batched(
                views["radar_index"], views["md_mask"], views["md_rows"],
                motion_index,
            ) if noise_model is not None
            else torch.zeros(
                len(views["radar_index"]), len(motion_bins), device=device)
        )
        # Forward-model completeness: the pipeline's scored renders carry
        # calibrated injected noise, so the training prediction carries the
        # same expected floor (zeros unless a reference is configured).
        if _COMPILED_PLANE_TERM:
            return _md_marginal_prediction(
                doppler_cube, views["md_mask"], views["md_rows"],
                noise_floor)
        return (
            microdoppler_marginal_batched(
                doppler_cube, views["md_mask"], views["md_rows"], None,
            ).clamp_min(0.0) + noise_floor
        ).sqrt()

    def losses_from_profiles_units(
        units: list[tuple[int, int, int | None]],
        ra_profile: torch.Tensor,
        planes: torch.Tensor,
        dc_profile: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-unit two-term loss from the gain-scaled RA profile and planes.

        The evaluation's loss chain, carrying the same exact rewrites the
        training step's does (HUPR_FUSED_LOSS / HUPR_COMPILED_PLANE_TERM,
        same gates): the fused RA-magnitude chain, the setup-time target
        denominators, and the compiled plane term. The evaluation runs
        identical expressions to before, just in the cheaper order.
        """
        views = unit_views(units)
        if _FUSED_LOSS:
            ra_prediction = profiles_to_ra_magnitude_fused(
                ra_profile.real, ra_profile.imag, keep_index)
            rows = view_rows_for(units)
            denominators = dict(
                ra_target_mean_square=flat_views[
                    "ra_mean_square"].index_select(0, rows),
                md_target_mean_square=flat_views[
                    "md_mean_square"].index_select(0, rows),
            )
            plane_denominator = flat_views[
                "ra_doppler_mean_square"].index_select(0, rows)
            if "plane-ra" in _ABLATE_TERMS:
                compiled_plane_l2 = None
                ra_plane_prediction = None
                md_prediction = md_prediction_units(units, planes)
            elif _COMPILED_PLANE_TERM:
                plane_spectrum = profiles_to_ra_spectrum_fused(
                    planes.real, planes.imag, keep_index)
                plane_target = plane_cells_target(plane_targets_for(units))
                unit_radars = torch.tensor([radar_index for _, radar_index, _ in units], device=device)
                compiled_plane_l2 = (
                    _plane_term_sums(
                        torch.view_as_real(plane_cells(plane_spectrum)),
                        plane_target, plane_magnitude_floor,
                        plane_noise_power(unit_radars, None if view_plane_bins is None
                                          else view_plane_bins.index_select(0, rows)))
                    / float(plane_target[0].numel()) / plane_denominator)
                ra_plane_prediction = None
                if _md_support_within_keep:
                    # Same spectrum reuse as the training step's (see
                    # static_losses): the motion bins are 13 of these 63.
                    md_prediction = _md_marginal_prediction(
                        motion_planes_of(plane_spectrum, units).abs(),
                        views["ra_mask_keep"], views["range_rows_keep"],
                        noise_model.microdoppler_marginal_batched(
                            views["radar_index"], views["ra_mask_keep"],
                            views["range_rows_keep"], motion_index,
                            range_index=keep_index,
                        ) if noise_model is not None
                        else torch.zeros(
                            len(views["radar_index"]), len(motion_bins),
                            device=device))
                else:
                    md_prediction = md_prediction_units(units, planes)
            else:
                compiled_plane_l2 = None
                ra_plane_prediction = profiles_to_ra_magnitude_fused(
                    planes.real, planes.imag, keep_index)
                denominators["ra_plane_target_mean_square"] = plane_denominator
                md_prediction = md_prediction_units(units, planes)
        else:
            compiled_plane_l2 = None
            denominators = {}
            ra_prediction = profiles_to_ra_magnitude_batched(
                ra_profile.real, ra_profile.imag,
            ).index_select(1, keep_index)
            # Range is dim 2 once the plane axis is present: (units, planes,
            # range, angle) against the loop-0 map's (units, range, angle).
            ra_plane_prediction = profiles_to_ra_magnitude_batched(
                planes.real, planes.imag,
            ).index_select(2, keep_index)
            md_prediction = md_prediction_units(units, planes)
        total, ra_l2, md_l2 = two_term_loss_units(
            ra_prediction, views["ra_target"],
            md_prediction, views["md_target"], args.microdoppler_weight,
            ra_loss=args.ra_loss,
            ra_plane_prediction=ra_plane_prediction,
            ra_plane_target=(None if ra_plane_prediction is None
                             else plane_targets_for(units)),
            **denominators,
        )
        if compiled_plane_l2 is not None:
            # Additive composition, exactly as the training step's.
            ra_l2 = ra_l2 + compiled_plane_l2
            total = total + compiled_plane_l2
        if elevation_on:
            rows = view_rows_for(units)
            unit_radars = torch.tensor([radar_index for _, radar_index, _ in units], device=device)
            rae = elevation_terms(rows, unit_radars,
                                  None if view_plane_bins is None else view_plane_bins.index_select(0, rows),
                                  ra_profile, planes)
            ra_l2 = ra_l2 + rae
            total = total + rae
        if pw_on:
            pw = person_window_terms(view_rows_for(units), ra_profile, planes)
            ra_l2 = ra_l2 + pw
            total = total + pw
        if static_on and dc_profile is not None:
            st = static_terms(view_rows_for(units), dc_profile)
            ra_l2 = ra_l2 + st
            total = total + st
        return total, ra_l2, md_l2, md_prediction

    # ------------------------------------------------------------------
    # P2 — a fixed path count per training step
    #
    # A step draws 6 frames whose point counts differ (1,498-2,687 here), so
    # `paths = points x n_tx x n_rx` was a different integer on every step:
    # measured over 24 consecutive steps, 24 distinct values spanning 263,424
    # to 316,080. W3/E3 made every shape *statically known* — no `nonzero`, no
    # device round trip — but statically known is not static, and a CUDA graph
    # replays a fixed set of kernels against fixed buffers. This is the item
    # that makes capture possible; on its own it costs a little and saves
    # nothing, which is the expected result.
    #
    # Two halves, one per radar. The two radars render the same six frames, so
    # each half is padded to the same `pad_points` and the radar axis of every
    # per-point quantity becomes "first half / second half" — the TX and RX
    # array stacks stop being per-step gathers and become constants, and the
    # per-radar gain keeps the `expand`-then-`cat` form whose backward is a
    # deterministic segment sum rather than a scatter-add.
    #
    # Padding rows repeat row 0 of the flat cache — a real point, so every
    # geometric quantity stays finite and no derivative sees a degenerate
    # input — and are switched off through `shadow_mask`, the renderer's
    # existing per-(point, tx) validity mask. `w_full` is multiplied by
    # `active.float()` unconditionally at rasterizer_factorized.py:513, so a
    # padded path's weight is exactly 0.0, its carrier is exactly 0.0, every
    # contribution it scatters is exactly 0.0, and — because the mask is a
    # constant — its gradient is exactly 0 as well, all the way back to the
    # material and the normals. That is W3's argument applied to points that
    # were added rather than to points that were already there. For a real
    # point `shadow_mask` is True and `active_i & True` is the same bool the
    # renderer computed before, so nothing about a real path changes.
    #
    # What the step still needs from the host is which six frames it draws.
    # That is one 6-element index; every other per-step quantity — the row
    # gather, the per-point unit index, the validity mask, the view rows, the
    # velocity source rows — is derived from it on the device, so the step
    # takes exactly one host-issued copy of per-step data.
    # ------------------------------------------------------------------
    flat_frames = list(frames)
    frame_position = {frame: index for index, frame in enumerate(flat_frames)}
    frame_counts = [int(cache[frame]["points"].shape[0]) for frame in flat_frames]
    frame_starts = np.cumsum([0] + frame_counts)
    flat_points = torch.cat([cache[frame]["points"] for frame in flat_frames])
    flat_weights = torch.cat([cache[frame]["weights"] for frame in flat_frames])
    # Parallel to `flat_points`, so the padded gather below indexes it with the
    # same rows. Padding rows land on global point 0 and are masked out by
    # `valid`, exactly as their positions are.
    flat_global_index = torch.cat(
        [cache[frame]["global_index"] for frame in flat_frames])
    # The per-frame entries become views of the flat tensors: the same values
    # at the same addresses the flat gather reads, so the evaluation and legacy
    # paths are bitwise untouched and nothing is stored twice.
    for index, frame in enumerate(flat_frames):
        start, stop = int(frame_starts[index]), int(frame_starts[index + 1])
        cache[frame]["points"] = flat_points[start:stop]
        cache[frame]["weights"] = flat_weights[start:stop]
    frame_offset_device = torch.as_tensor(
        frame_starts[:-1], dtype=torch.long, device=device)
    frame_count_device = torch.as_tensor(
        frame_counts, dtype=torch.long, device=device)
    frame_id_device = torch.as_tensor(
        flat_frames, dtype=torch.long, device=device)
    # Flat cloud row -> row of the fused normals parameter, so the batch's
    # normals come out of the same gather the positions do.
    normal_row_of_point = torch.zeros(
        int(frame_starts[-1]), dtype=torch.long, device=device)
    for index, frame in enumerate(flat_frames):
        span = model.normal_span.get(int(frame))
        if span is None:
            continue
        start = int(frame_starts[index])
        normal_row_of_point[start:start + span[1]] = torch.arange(
            span[0], span[0] + span[1], device=device)
    # Every (frame, radar) view, stacked once, in the order the step's unit
    # list indexes them: row = frame position * len(RADARS) + radar index.
    view_pairs = [(frame, radar_index)
                  for frame in flat_frames
                  for radar_index in range(len(RADARS))]

    def _cached_view(pair: tuple[int, int], key: str) -> torch.Tensor:
        frame, radar_index = pair
        return cache[frame]["views"][RADARS[radar_index]][key]

    # Per-view Doppler bins: (views, K), padded with -1. The 13 micro-Doppler
    # bins occupy positions 0..12 in EVERY view, so `md_plane_index` stays a
    # constant arange and the MD term needs no per-frame gather; each view's
    # own support follows. K is the widest support in the run, so the shape is
    # static -- one CUDA graph -- while the bins vary per frame.
    view_plane_bins = None
    md_list = sorted(torch.nonzero(motion_doppler, as_tuple=False).flatten().tolist())
    if args.doppler_support_snr is not None:
        md_set = set(md_list)
        rows_bins = []
        for pair in view_pairs:
            support = set(_cached_view(pair, "doppler_support").tolist()) - {TDM_LOOPS // 2}
            if args.render_band_always == "on":
                # HuPR's layout: the band first, at positions 0..len-1 of
                # every view, then the view's extra support.
                rows_bins.append(md_list + sorted(support - md_set))
            else:
                # Only what the view supports. A band bin the view does not
                # support is not rendered; the micro-Doppler marginal reads
                # the zero slot there (below: the floor alone).
                rows_bins.append(sorted(support))
        # One trailing -1 column on every view: a zero plane the micro-Doppler
        # gather can point absent band bins at. (With the band always
        # rendered it is simply padding.)
        width = max(len(b) for b in rows_bins) + (0 if args.render_band_always == "on" else 1)
        view_plane_bins = torch.full((len(view_pairs), width), -1,
                                     dtype=torch.long, device=device)
        for row, bins in enumerate(rows_bins):
            view_plane_bins[row, :len(bins)] = torch.tensor(
                bins, dtype=torch.long, device=device)
        from mm4DPS.forward_render import validate_per_unit_bins
        validate_per_unit_bins(view_plane_bins, TDM_LOOPS)
        widths = torch.tensor([len(b) for b in rows_bins])
        print(f"Doppler support: {width} planes of {len(ra_doppler_bins)} "
              f"(per view: median {int(widths.median())}, min {int(widths.min())}, "
              f"max {int(widths.max())}) at SNR >= {args.doppler_support_snr:g}x "
              f"the {args.doppler_support_floor} floor, band "
              f"{'always rendered' if args.render_band_always == 'on' else 'rendered where supported'}",
              flush=True)
        # Narrow each view's target to its own bins, zero-padding the tail.
        # A pad's W row is zero, so that plane renders exactly zero against a
        # zero target: it contributes nothing to either side of the term's
        # ratio, which is what makes the shared static width exact.
        position = {b: i for i, b in enumerate(ra_doppler_bins)}
        for row, pair in enumerate(view_pairs):
            frame, radar_index = pair
            view = cache[frame]["views"][RADARS[radar_index]]
            full = view["ra_doppler_target"]
            narrowed = full.new_zeros((width, *full.shape[1:]))
            bins = view_plane_bins[row]
            take = [position[int(b)] for b in bins.tolist() if b >= 0]
            narrowed[:len(take)] = full.index_select(
                0, torch.tensor(take, device=device))
            view["ra_doppler_target"] = narrowed
            if elevation_on:
                full_rae = view["rae_doppler_target"]
                narrowed_rae = full_rae.new_zeros((width, *full_rae.shape[1:]))
                narrowed_rae[:len(take)] = full_rae.index_select(0, torch.tensor(take, device=device))
                view["rae_doppler_target"] = narrowed_rae
            del view["doppler_support"]

    if noise_prior_power is None:
        noise_prior_power = torch.stack([
            torch.stack(noise_prior_frames[radar]).median(dim=0).values for radar in RADARS])
    if args.dump_targets is not None:
        # dev tooling: the built targets, for a numerical gate of this build against another
        def to_cpu(value):
            return value.detach().cpu() if isinstance(value, torch.Tensor) else {
                k: to_cpu(v) for k, v in value.items()}
        torch.save({"view_plane_bins": None if view_plane_bins is None else view_plane_bins.cpu(),
                    "noise_prior_power": noise_prior_power.cpu(),
                    "cache": {frame: to_cpu(entry) for frame, entry in cache.items()}}, args.dump_targets)
        print(f"dumped the targets to {args.dump_targets}", flush=True)

    # Planes the renderer produces and the loss reads. Under per-view bins
    # this is the padded width; otherwise the flat non-DC set.
    plane_count = (len(ra_doppler_bins) if view_plane_bins is None
                   else int(view_plane_bins.shape[1]))
    if view_plane_bins is None and not set(motion_bins).issubset(set(ra_doppler_bins)):
        raise RuntimeError(
            "the micro-Doppler bins must be a subset of the rendered planes")
    if view_plane_bins is None:
        md_plane_index = torch.tensor(
            [ra_doppler_bins.index(b) for b in motion_bins], device=device)
        md_index_by_view = None
    elif args.render_band_always == "on":
        # Per-view bins put the micro-Doppler set at positions 0..len-1 of
        # every view, precisely so this stays a constant.
        md_plane_index = torch.arange(len(motion_bins), device=device)
        md_index_by_view = None
        # THE MD TERM READS THE PLANE SPECTRUM. The band is the planes'
        # prefix, so its magnitude cube is the first len(motion_bins) planes
        # of the spectrum the plane term already computed -- on the cascade
        # 53 of 63 planes, i.e. the separate band chain was an 84% duplicate
        # of the plane chain (FFT, magnitude, and their backwards). The
        # spectrum keeps the retained range bins only, so the person's rows
        # must all be retained; they are (the guard removes the near field).
        rows_outside = any(
            bool((view["range_rows"] & ~keep_range).any())
            for entry in cache.values() for view in entry["views"].values())
        md_from_plane_spectrum = _FUSED_LOSS and _COMPILED_PLANE_TERM and not rows_outside \
            and len(motion_bins) * 2 >= plane_count
        if md_from_plane_spectrum:
            print(f"micro-Doppler term: read from the plane spectrum's first "
                  f"{len(motion_bins)} of {plane_count} planes (retained range)", flush=True)
    else:
        # Per view: where each band bin sits in the view's plane list, or the
        # trailing zero slot (position width-1, always -1) when the view does
        # not render it.
        md_plane_index = None
        rows = view_plane_bins.tolist()
        zero_slot = view_plane_bins.shape[1] - 1
        # -1 marks an absent band bin; the gather points it at the last
        # rendered plane of the batch, which is the trailing -1 zero plane
        # (no view fills the last slot).
        md_index_by_view = torch.tensor(
            [[row.index(b) if b in row else -1 for b in motion_bins] for row in rows],
            dtype=torch.long, device=device)
        if int((view_plane_bins[:, zero_slot] >= 0).sum()) != 0:
            raise RuntimeError("the trailing plane slot must be the -1 zero plane")

    # Every view renders K planes, K the widest support in the run, because
    # one captured graph needs one shape; a view's trailing -1 slots are
    # zero-weight pads. The Doppler-plane kernels skip a unit's pads (per-unit
    # active plane counts, derived from the bins inside the assembly), so a
    # view supported on 12 bins costs 12 bins' work in a 51-plane batch, and
    # the rendered planes are bitwise what walking every pad gave.

    def motion_planes_of(planes: torch.Tensor, units=None, view_rows=None) -> torch.Tensor:
        """The band's planes of a batch, ``(units, band, ...)``: a constant slice or a per-view gather.

        ``view_rows`` is the captured static step's device index into the
        per-view tensors; every other caller passes its unit list.
        """
        if md_index_by_view is None:
            return planes.index_select(1, md_plane_index)
        rows = view_rows if view_rows is not None else view_rows_for(units)
        index = md_index_by_view.index_select(0, rows)
        index = torch.where(index < 0, torch.full_like(index, planes.shape[1] - 1), index)
        unit_index = torch.arange(planes.shape[0], device=planes.device)[:, None]
        return planes[unit_index, index]

    if noise_model is not None and args.noise_model == "measured":
        # The floor is the recording's own, as measured (the prior at scale 1, no range correction), and stays so: a
        # receiver property the empty cells give directly. Learned, it absorbed model error into a range-shaped floor
        # (range correction up to e^+-1.7 on RT-Pose 183; the render's far floor 1.4-1.8x the recording's, which buried a
        # sitting person at 7.5 m -- docs/STATIC_SEQS_NEXT.md 1e).
        with torch.no_grad():
            noise_model.log_scale.zero_(); noise_model.range_coeff.zero_()
        for parameter in noise_model.parameters():
            parameter.requires_grad_(False)
        print("noise model: the measured floor, fixed (scale 1, no range correction)", flush=True)
    if noise_model is not None and args.noise_model == "on":
        # The scale has to start where the gradient is usable. Initialized
        # from the renderer's slow_time_nuisance_scale it was ~5 orders too
        # small -- that constant multiplies a background-magnitude quantile in
        # the renderer, not an absolute power -- and the resulting 4e-14
        # gradient sat far below Adam's epsilon, so the parameter never moved
        # at all. Anchoring it to a fraction of the measured micro-Doppler
        # power puts it in range and makes the starting point data-defined
        # rather than a unit accident.
        with torch.no_grad():
            probe_frame = training_frames[0]
            for radar_index, radar in enumerate(RADARS):
                view = cache[probe_frame]["views"][radar]
                target_power = view["md_target"].square().median()
                noise_model.log_scale.data[radar_index] = 0.0
                unit = noise_model.microdoppler_marginal(
                    radar_index, view["ra_mask"], view["range_rows"],
                    motion_doppler,
                ).median().clamp_min(1e-30)
                ratio = (args.noise_initial_fraction * target_power / unit)
                noise_model.log_scale.data[radar_index] = (
                    0.5 * torch.log(ratio.clamp_min(1e-30)))
            print("noise model initialized at "
                  f"{args.noise_initial_fraction:.0%} of the measured MD "
                  f"power: log_scale "
                  f"{[round(float(v), 3) for v in noise_model.log_scale]}",
                  flush=True)

    # Cell mask over the plane target's (63, range, angle) grid, built from
    # the targets themselves so it cannot drift from what is supervised.
    _threshold_flags = [name for name, value in (
        ("--plane-mask-db", args.plane_mask_db),
        ("--plane-mask-keep-energy", args.plane_mask_keep_energy),
        ("--plane-mask-snr", args.plane_mask_snr)) if value is not None]
    if len(_threshold_flags) > 1:
        raise ValueError(
            "these set the same threshold three different ways; pass one: "
            + ", ".join(_threshold_flags))
    if (args.plane_mask_keep_energy is not None
            and not 0.0 < args.plane_mask_keep_energy <= 1.0):
        raise ValueError("--plane-mask-keep-energy must lie in (0, 1]")
    _want_mask = bool(_threshold_flags)
    if _want_mask or args.plane_mask_dump is not None:
        if _want_mask and not _COMPILED_PLANE_TERM:
            raise ValueError(
                "--plane-mask-db needs the compiled plane term "
                "(HUPR_COMPILED_PLANE_TERM=1); the eager branch is unmasked")
        ever = None
        for pair in view_pairs:
            view = _cached_view(pair, "ra_doppler_target")
            ever = view.clone() if ever is None else torch.maximum(ever, view)
        if args.plane_mask_dump is not None:
            args.plane_mask_dump.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.plane_mask_dump,
                ever=ever.detach().cpu().numpy(),
                doppler_bins=np.asarray(ra_doppler_bins),
                keep_range=keep_range.detach().cpu().numpy(),
                plane_mask_db=np.asarray(
                    [np.nan if args.plane_mask_db is None else args.plane_mask_db]),
                plane_mask_dilate=np.asarray([args.plane_mask_dilate]))
            print(f"wrote plane-mask dump to {args.plane_mask_dump}", flush=True)
        if not _want_mask:
            del ever
        else:
            if args.plane_mask_snr is not None:
                # Per-range floor, broadcast over plane and angle. `threshold`
                # is a tensor here rather than a scalar; the comparison below
                # broadcasts either way.
                # Joint median over (plane, angle) at each range -- NOT a
                # median of medians, which is a different statistic.
                floor = ever.permute(1, 0, 2).reshape(
                    ever.shape[1], -1).median(dim=1).values
                threshold = args.plane_mask_snr * floor[None, :, None]
            elif args.plane_mask_db is not None:
                threshold = ever.amax() * (10.0 ** (args.plane_mask_db / 20.0))
            else:
                # Sort by magnitude and take the smallest prefix holding the
                # required L2. Dilation only ever adds cells, so the mask that
                # ships discards no more than the bound -- solving before it
                # is the conservative order.
                keep_fraction = args.plane_mask_keep_energy
                flat = ever.reshape(-1)
                ordered, _ = torch.sort(flat, descending=True)
                energy = torch.cumsum(ordered.double().square(), dim=0)
                need = energy[-1] * keep_fraction
                threshold = ordered[
                    int(torch.searchsorted(energy, need).clamp_max(
                        ordered.numel() - 1))]
            mask = ever >= threshold
            _discarded = float(torch.sqrt(
                1.0 - (ever.double().square() * mask).sum()
                / ever.double().square().sum()))
            if args.plane_mask_dilate > 0:
                width = 2 * args.plane_mask_dilate + 1
                mask = torch.nn.functional.max_pool2d(
                    mask.to(torch.float32).unsqueeze(0), width, stride=1,
                    padding=args.plane_mask_dilate).squeeze(0) > 0
            plane_cell_index = mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
            print(f"plane cell mask: {plane_cell_index.numel():,} of "
                  f"{mask.numel():,} cells "
                  f"({100.0 * plane_cell_index.numel() / mask.numel():.1f}%) "
                  + (f"at SNR >= {args.plane_mask_snr:g}x the per-range floor "
                     if args.plane_mask_snr is not None else
                     f"at {20.0 * math.log10(float(threshold / ever.amax())):+.1f} dB "
                     f"(= {float(threshold / ever.amax()):.2e} of peak), ") +
                  f"dilate {args.plane_mask_dilate}; discards "
                  f"{100.0 * _discarded:.1f}% of the target L2 before dilation",
                  flush=True)
            del ever, mask

    def _plane_target_view(pair: tuple[int, int]) -> torch.Tensor:
        view = _cached_view(pair, "ra_doppler_target")
        if plane_cell_index is None:
            return view
        return view.reshape(-1).index_select(0, plane_cell_index)

    # Filled view by view so the full-precision cube is never materialised;
    # under float16 that is the whole point of the option.
    _sample_target = _plane_target_view(view_pairs[0])
    _store_dtype = (torch.float16 if args.plane_target_dtype == "float16"
                    else torch.float32)
    _plane_store = torch.empty((len(view_pairs), *_sample_target.shape),
                               dtype=_store_dtype, device=device)
    _plane_rms = (None if _store_dtype is torch.float32 else
                  torch.empty(len(view_pairs), dtype=torch.float32, device=device))
    for _row, _pair in enumerate(view_pairs):
        _view = _plane_target_view(_pair).to(torch.float32)
        if _plane_rms is None:
            _plane_store[_row] = _view
        else:
            _rms = _view.square().mean().sqrt().clamp_min(1e-30)
            _plane_rms[_row] = _rms
            _plane_store[_row] = (_view / _rms).to(_store_dtype)
    del _sample_target

    flat_views = {
        "ra_mask": torch.stack([_cached_view(p, "ra_mask") for p in view_pairs]),
        "range_rows": torch.stack(
            [_cached_view(p, "range_rows") for p in view_pairs]
        ).to(torch.float32),
        "ra_target": torch.stack(
            [_cached_view(p, "ra_target") for p in view_pairs]
        ).index_select(1, keep_index),
        "md_target": torch.stack(
            [_cached_view(p, "md_target") for p in view_pairs]),
        # Stacked once here and dropped from the per-frame entries below: the
        # graph path needs a fixed address, and holding both copies would cost
        # 2.3 GB twice on a 590-frame sequence.
        "ra_doppler_target": _plane_store,
    }
    # the two masks on the retained range bins, for the md term read out of
    # the plane spectrum
    flat_views["ra_mask_kept"] = flat_views["ra_mask"].index_select(1, keep_index)
    flat_views["range_rows_kept"] = flat_views["range_rows"].index_select(1, keep_index)
    if _plane_rms is not None:
        flat_views["ra_doppler_rms"] = _plane_rms
    if static_on:
        # P1: the DC-plane targets on the retained range bins and the two region energies each view's term divides by
        flat_views["dc_target"] = torch.stack([_cached_view(p, "dc_target") for p in view_pairs]).index_select(1, keep_index)
        _m = flat_views["ra_mask_kept"].to(torch.float32)
        _e = flat_views["dc_target"].square()
        flat_views["dc_person_energy"] = (_e * _m).flatten(1).sum(dim=-1).clamp_min(1e-30)
        flat_views["dc_scene_energy"] = (_e * (1.0 - _m)).flatten(1).sum(dim=-1).clamp_min(1e-30)
        if elevation_on:
            flat_views["dc_rae_target"] = torch.stack([_cached_view(p, "dc_rae_target") for p in view_pairs]).index_select(1, keep_index)
            _e = flat_views["dc_rae_target"].square()
            flat_views["dc_rae_person_energy"] = (_e * _m[..., None]).flatten(1).sum(dim=-1).clamp_min(1e-30)
            flat_views["dc_rae_scene_energy"] = (_e * (1.0 - _m)[..., None]).flatten(1).sum(dim=-1).clamp_min(1e-30)
        _share = (flat_views["dc_person_energy"] / (flat_views["dc_person_energy"] + flat_views["dc_scene_energy"]))
        print(f"static term (P1): DC-plane targets {tuple(flat_views['dc_target'].shape[1:])}"
              + (f" and {tuple(flat_views['dc_rae_target'].shape[1:])}" if elevation_on else "")
              + f" per view; the person's cells hold {100 * float(_share.median()):.1f}% (median over views) of the DC target energy "
              f"and count as {100 * 0.5:.0f}% of the term; weight {args.static_weight:g}", flush=True)
        del _m, _e, _share
    if pw_on:
        # the person-window term's per-view store: the side the consumer puts the person on (per radar, the side with more
        # of the recording's window energy over the sequence), then that side's targets, masks and gather indices
        pw_side = {radar: int(np.argmax(pw_side_energy[radar])) for radar in RADARS}
        for radar in RADARS:
            e0, e1 = pw_side_energy[radar]
            print(f"person-window term [{radar}]: the consumer's azimuth side of the person = {'+' if pw_side[radar] == 0 else '-'}(delta . array axis) "
                  f"(recording's window energy {e0:.3e} vs its mirror {e1:.3e}, ratio {max(e0, e1) / max(min(e0, e1), 1e-30):.2f}); "
                  f"box {2 * pw_half_r + 1} range x {2 * pw_half_a + 1} azimuth x {_pw_n_el} elevation bins, window +-{args.person_window_range} m range, "
                  f"+-{args.person_window_lateral} m lateral; planes {args.person_window_planes}" + (f" ({len(_pw_bins)} Doppler bins{', DC kept' if _pw_keep_dc else ''})" if _pw_plane_index is not None else "") + f"; weight {args.person_window_loss:g}", flush=True)
        def _pw(pair, key):
            frame, radar_index = pair; e = cache[frame]["views"][RADARS[radar_index]]["pw"]; side = pw_side[RADARS[radar_index]]
            return e[key][side] if isinstance(e[key], list) else e[key]
        flat_views["pw_target"] = torch.stack([_pw(p, "target") for p in view_pairs])            # (views, B, Rw, Aw, E) float16, B = 1 unless resolved
        flat_views["pw_mask"] = torch.stack([_pw(p, "mask") for p in view_pairs])                # (views, Rw, Aw)
        flat_views["pw_range_index"] = torch.stack([_pw(p, "range_index") for p in view_pairs])  # (views, Rw)
        flat_views["pw_angle_index"] = torch.stack([_pw(p, "angle_index") for p in view_pairs])  # (views, Aw)
        flat_views["pw_count"] = (flat_views["pw_mask"].flatten(1).sum(dim=-1) * _pw_n_el * _pw_bands).clamp_min(1.0)
        flat_views["pw_weight"] = torch.tensor([_pw(p, "weight") for p in view_pairs], dtype=torch.float32, device=device)
        for radar_index, radar in enumerate(RADARS):
            wv = flat_views["pw_weight"][radar_index::len(RADARS)]
            print(f"person-window term [{radar}]: frame weights (window power over its rows' {args.person_window_snr_reference} outside it, 1x -> 0, {args.person_window_snr:g}x -> 1): "
                  f"median {float(wv.median()):.2f}, share at 1: {100 * float((wv >= 0.999).float().mean()):.0f}%, at 0: {100 * float((wv <= 0.001).float().mean()):.0f}%", flush=True)
        if _pw_plane_index is not None:
            # where each consumer bin sits among the rendered planes, per view: all 63 in fftshifted order, or the view's own
            # Doppler support (view_plane_bins, padded -1; a bin the view does not render gets position 0 with weight 0)
            if view_plane_bins is None:
                pos = _pw_plane_index[None, :].expand(len(view_pairs), -1).contiguous(); w = torch.ones_like(pos, dtype=torch.float32)
            else:
                bins_t = torch.tensor(_pw_bins, device=device)
                hit = view_plane_bins[:, :, None] == bins_t[None, None, :]                    # (views, K, B)
                w = hit.any(dim=1).to(torch.float32); pos = hit.to(torch.int64).argmax(dim=1) * w.to(torch.int64)
                print(f"person-window term: consumer Doppler bins rendered per view: median {int(w.sum(dim=1).median())} of {len(_pw_bins)}", flush=True)
            flat_views["pw_plane_pos"] = pos; flat_views["pw_plane_w"] = w
        for frame in frames:
            for radar in RADARS:
                cache[frame]["views"][radar].pop("pw", None)
    if elevation_on:
        if plane_cell_index is not None or plane_sample_index is not None:
            raise SystemExit("--elevation-term on does not combine with --plane-mask-* / --plane-sample-count")
        flat_views["pw_box_mask"] = torch.stack([_cached_view(p, "pw_box_mask") for p in view_pairs])
        flat_views["rae_target"] = torch.stack(
            [_cached_view(p, "rae_target") for p in view_pairs]).index_select(1, keep_index)
        if args.elevation_planes == "motion":
            rae_plane_index = (torch.arange(len(motion_bins), device=device) if md_plane_index is not None and view_plane_bins is not None
                               else md_plane_index)
            if rae_plane_index is None:
                raise SystemExit("--elevation-planes motion needs a constant micro-Doppler plane index (render_band_always on, or no per-view support)")
        else:
            rae_plane_index = None
        flat_views["rae_doppler_target"] = torch.stack([
            (_cached_view(p, "rae_doppler_target") if rae_plane_index is None
             else _cached_view(p, "rae_doppler_target").index_select(0, rae_plane_index)) for p in view_pairs])
        flat_views["rae_doppler_rms"] = torch.stack([_cached_view(p, "rae_doppler_rms") for p in view_pairs]).to(torch.float32)
        flat_views["rae_mean_square"] = flat_views["rae_target"].square().flatten(1).mean(dim=-1).clamp_min(1e-30)
        # mean(t^2) over the STORED planes (all of them: exactly rms^2; the motion band: its own share)
        _rae_ms = torch.empty(len(view_pairs), device=device)
        for _start in range(0, len(view_pairs), 32):
            _rae_ms[_start:_start + 32] = flat_views["rae_doppler_target"][_start:_start + 32].to(torch.float32).square().flatten(1).mean(dim=-1)
        flat_views["rae_doppler_mean_square"] = (_rae_ms * flat_views["rae_doppler_rms"].square()).clamp_min(1e-30)
        if _win_rae_on and args.window_el_bound == "on":
            # Which elevation bins does the person move in? Read it off the windowed plane target,
            # which has clutter removed, so the answer is the moving energy's own extent. Chunked
            # over views: the plane target is multi-GB and must never be materialised in float32.
            _boxk = flat_views["pw_box_mask"].index_select(1, keep_index).to(torch.float32)
            _e = torch.empty(flat_views["rae_doppler_target"].shape[0],
                             flat_views["rae_doppler_target"].shape[-1], device=device)
            for _s0 in range(0, _e.shape[0], 8):
                _t = flat_views["rae_doppler_target"][_s0:_s0 + 8].to(torch.float32)
                _e[_s0:_s0 + 8] = (_t.square() * _boxk[_s0:_s0 + 8, None, ..., None]).sum(dim=(1, 2, 3))
                del _t
            _keep = (_e >= args.window_el_threshold * _e.max(dim=1, keepdim=True).values.clamp_min(1e-30))
            flat_views["pw_el_mask"] = _keep.to(torch.float32)
            _share = (_e / _e.max(dim=1, keepdim=True).values.clamp_min(1e-30)).mean(dim=0)
            _prof = ", ".join(f"{v:.2f}" for v in sorted(_share.tolist(), reverse=True))
            print(f"windowed RAE terms: elevation bound on, {float(_keep.sum(dim=1).float().mean()):.1f} of "
                  f"{_e.shape[1]} bins kept per view (threshold {args.window_el_threshold:g} of the peak "
                  f"moving bin); mean per-bin share of the peak, sorted: [{_prof}]", flush=True)
            del _boxk, _e, _keep
        # NOT set to None when unused: `static_losses` gathers every flat view with
        # `index_select`, so a None value kills the captured step. Every reader uses
        # `flat_views.get("pw_el_mask")`, so an absent key behaves identically. (This is a bug on
        # main at bd6ce83, reproduced there with main's own default flags; it wants upstreaming.)
        _rae_bytes = flat_views["rae_doppler_target"].numel() * 2 + flat_views["rae_target"].numel() * 4
        print(f"elevation term: targets {tuple(flat_views['rae_target'].shape[1:])} loop-0 and "
              f"{tuple(flat_views['rae_doppler_target'].shape[1:])} planes per view, {_rae_bytes / 2**30:.2f} GiB on the device; "
              f"floor factor {elevation_floor_factor():.3f}", flush=True)
        for _frame in flat_frames:
            for _radar in RADARS:
                for _k in ("rae_target", "rae_doppler_target", "rae_doppler_rms"):
                    cache[_frame]["views"][_radar].pop(_k, None)
    for _frame in flat_frames:
        for _radar in RADARS:
            del cache[_frame]["views"][_radar]["ra_doppler_target"]
    # The loss denominators are per-view constants (the targets are fixed for
    # the whole run), so they are reduced once here instead of once per step —
    # the plane target alone is ~47 MB per step of squared-and-reduced
    # constant. Chunked so the transient `square()` never materializes the
    # full multi-GB plane-target cube a second time.
    if _plane_rms is not None:
        # mean(t^2) = rms^2 by construction, so this is exact rather than a
        # second reduction over the cube.
        flat_views["ra_doppler_mean_square"] = _plane_rms.square().clamp_min(1e-30)
    else:
        _plane_target = flat_views["ra_doppler_target"]
        _plane_denominator = torch.empty(_plane_target.shape[0], device=device)
        for _start in range(0, _plane_target.shape[0], 32):
            _plane_denominator[_start:_start + 32] = (
                _plane_target[_start:_start + 32].square().flatten(1).mean(dim=-1))
        flat_views["ra_doppler_mean_square"] = _plane_denominator.clamp_min(1e-30)
    flat_views["ra_mean_square"] = flat_views["ra_target"].square().flatten(
        1).mean(dim=-1).clamp_min(1e-30)
    flat_views["md_mean_square"] = flat_views["md_target"].square().mean(
        dim=-1).clamp_min(1e-30)
    # 0-d device tensor rather than a Python literal: inductor's codegen
    # promotes a tiny float literal to fp64 inside the reduction loop and
    # refuses the kernel; a float32 tensor input keeps the type uniform.
    plane_magnitude_floor = torch.full((), 1e-30, device=device)
    # The motion bins are a subset of the RA plane bins, so the MD cube's 13
    # spectra are 13 of the 63 the plane term computes -- reusable, IF the
    # micro-Doppler support lies entirely inside the kept range rows (the
    # plane spectra are range-sliced; the marginal's masked sums are then
    # exactly the full-range sums, every dropped row being a zero term).
    # Checked, not assumed, over every view; a geometry that violates it
    # falls back to the separate cube.
    _md_support_within_keep = bool(
        (flat_views["ra_mask"][:, ~keep_range].any() == False).item()
        and (flat_views["range_rows"][:, ~keep_range].abs().sum() == 0).item())
    if _md_support_within_keep:
        flat_views["ra_mask_keep"] = flat_views["ra_mask"].index_select(
            1, keep_index)
        flat_views["range_rows_keep"] = flat_views["range_rows"].index_select(
            1, keep_index)
    print(f"md support within kept range rows: {_md_support_within_keep} "
          f"(cube reuse {'on' if _md_support_within_keep else 'off'})",
          flush=True)

    view_rows_cache: dict[tuple, torch.Tensor] = {}

    def view_rows_for(units: list[tuple[int, int, int | None]]) -> torch.Tensor:
        """A unit list's rows into the flat per-view tensors.

        Cached per distinct unit list for the same reason `unit_views` is:
        the evaluation reuses a fixed set of chunk lists every epoch, and
        rebuilding the index costs a host upload per call.
        """
        key = tuple((frame, radar_index) for frame, radar_index, _ in units)
        rows = view_rows_cache.get(key)
        if rows is None:
            rows = torch.tensor(
                [frame_position[frame] * len(RADARS) + radar_index
                 for frame, radar_index, _ in units], device=device,
            )
            if len(view_rows_cache) < 64:
                view_rows_cache[key] = rows
        return rows

    def denormalise_plane_target(stored: torch.Tensor,
                                 rms: torch.Tensor | None) -> torch.Tensor:
        """Undo the per-view rms normalisation of the float16 store.

        A no-op under the float32 store, which holds the raw magnitudes, so
        that path stays bit-identical to before the option existed.
        """
        if rms is None:
            return stored
        return stored.to(torch.float32) * rms.reshape(
            -1, *([1] * (stored.dim() - 1)))

    def plane_targets_for(units: list[tuple[int, int, int | None]]) -> torch.Tensor:
        """The measured non-DC Doppler magnitudes for a unit list."""
        rows = view_rows_for(units)
        stored = flat_views["ra_doppler_target"].index_select(0, rows)
        rms = flat_views.get("ra_doppler_rms")
        return denormalise_plane_target(
            stored, None if rms is None else rms.index_select(0, rows))

    _rae_floor_factor = elevation_floor_factor() if elevation_on else 0.0
    if not elevation_on:
        rae_plane_index = None
    # resolved here, where the dataset is known; the loss functions below close over it
    _RAE_ROUTE = _RAE_ROUTE_ENV or _rae_route_default(dataset.name)
    if elevation_on:
        print(f"elevation term: RAE route {_RAE_ROUTE}"
              + ("" if _RAE_ROUTE_ENV else f" (the measured default for {dataset.name})"), flush=True)

    _RAE_VIEW_KEYS = (("rae_target", "rae_mean_square", "rae_doppler_target",
                       "rae_doppler_rms", "rae_doppler_mean_square")
                      # the windowed terms' masks, so the evaluation path gathers them too (the
                      # captured step gathers every flat view already)
                      + (("pw_box_mask",) if _win_rae_on else ())
                      # present as a key but None when the elevation bound is not in use, which
                      # is main's measured default -- so test the VALUE, not the key
                      + (("pw_el_mask",) if _win_rae_on
                         and flat_views.get("pw_el_mask") is not None else ()))

    def rae_terms_of(ra_profile: torch.Tensor, planes: torch.Tensor, views: dict,
                     floor_power: torch.Tensor | None, route: str,
                     term: str = _RAE_TERM) -> torch.Tensor:
        """`elevation_terms` from the (planes-narrowed) profiles' real and imaginary parts, by route.

        ``reference``: the op-by-op chain (`profiles_to_rae_magnitude`, then
        abs, floor, sqrt, difference, square, mean). ``fused``: both spectra
        from one compiled gather and two FFTs on a contiguous buffer
        (`profiles_to_rae_spectrum_fused`), the chain after `abs` unchanged --
        the same arithmetic, bitwise the reference (MM4DPS_RAE_CHECK=1).

        Under the four-term objective BOTH RAE views run on every kept range
        row (`keep_index`) and, at --elevation-planes all, on every rendered
        plane, so the gather this replaces carries the whole store rather than
        the person's ~21 rows item 2 was first measured on
        (docs/RUNTIME_RESULTS.md item 1 restricted the rows; it is not adopted
        here -- it is the item this objective puts back in question).
        """
        # `term` = "fused": the plane part's whole pointwise chain in one kernel pass, bitwise the
        # op-by-op form below (_RaePlaneTermSquaredError). It needs the plane spectrum COMPLEX, so
        # the fused-spectrum route hands it over directly and the reference route takes the
        # magnitude's own complex input -- the two spectrum routes stay independent of this one,
        # because which of them is faster differs by array while this is a gain on both.
        fused_term = (term == "fused" and floor_power is not None
                      and views["rae_doppler_target"].is_cuda)
        _shift = not _rae_unshift_targets
        if route == "reference":
            rae_prediction = profiles_to_rae_spectrum_reference(
                ra_profile.real, ra_profile.imag, keep_index, shift_azimuth=_shift).abs()
            plane_spectrum = (profiles_to_rae_spectrum_reference(
                planes.real, planes.imag, keep_index, shift_azimuth=_shift) if fused_term else None)
            rae_plane_prediction = (None if fused_term else profiles_to_rae_spectrum_reference(
                planes.real, planes.imag, keep_index, shift_azimuth=_shift).abs())
        else:
            # the COMPLEX profile straight through: .real / .imag are stride-2 views, and handing
            # them over costs two contiguous copies plus the gradient's recombination, 1.26 ms a
            # call (docs/RUNTIME_STEP_20MS_PLAN.md 12)
            rae_prediction = profiles_to_rae_spectrum_fused(
                ra_profile, keep_index, shift_azimuth=_shift).abs()
            plane_spectrum = profiles_to_rae_spectrum_fused(planes, keep_index, shift_azimuth=_shift)
            rae_plane_prediction = None if fused_term else plane_spectrum.abs()
        rae_l2 = ((rae_prediction - views["rae_target"]).square().flatten(1).mean(dim=-1)
                  / views["rae_mean_square"])
        if fused_term:
            diff = _RaePlaneTermDifference.apply(
                plane_spectrum if plane_spectrum.is_contiguous() else plane_spectrum.contiguous(),
                floor_power.reshape(-1),
                views["rae_doppler_target"], views["rae_doppler_rms"], _rae_floor_factor)
            squared = diff.square()
        else:
            if floor_power is not None:
                rae_plane_prediction = (rae_plane_prediction.square()
                                        + floor_power[..., None] * _rae_floor_factor).clamp_min(1e-30).sqrt()
            rae_plane_target = (views["rae_doppler_target"].to(torch.float32)
                                * views["rae_doppler_rms"].reshape(-1, 1, 1, 1, 1))
            squared = (rae_plane_prediction - rae_plane_target).square()
        rae_plane_l2 = squared.flatten(1).mean(dim=-1) / views["rae_doppler_mean_square"]
        if not _win_rae_on:
            return args.elevation_weight * (rae_l2 + rae_plane_l2)
        # The same two quantities restricted to the person's box, each normalised over the box:
        # term 3 (loop-0, clutter kept) and term 4 (per plane, DC excluded and clutter removed).
        # ONE transform, two scopes -- which is why every optimisation to PRODUCING the predictions
        # (items C, C2, P1, P2) serves all four terms rather than two.
        #
        # The box's angle axis is in the shifted convention. Item B un-shifts the targets and skips
        # the prediction's roll, so the mask must be un-shifted the SAME way or the window lands on
        # the wrong angles -- a wrong answer, not a crash.
        box = views["pw_box_mask"].index_select(1, keep_index).to(views["rae_target"].dtype)
        if _rae_unshift_targets:
            box = torch.fft.ifftshift(box, dim=-1)
        elm = views.get("pw_el_mask")
        b3 = box[..., None] if elm is None else box[..., None] * elm[:, None, None, :]
        c3 = b3.flatten(1).sum(dim=-1).clamp_min(1.0)
        den0 = ((views["rae_target"] * b3).square().flatten(1).sum(dim=-1) / c3).clamp_min(1e-30)
        win0 = ((rae_prediction - views["rae_target"]) * b3).square().flatten(1).sum(dim=-1) / c3 / den0
        b4 = (box[:, None, ..., None] if elm is None
              else box[:, None, ..., None] * elm[:, None, None, None, :])
        c4 = b4.expand(squared.shape).flatten(1).sum(dim=-1).clamp_min(1.0)
        # `denp` depends only on the target and the mask, both fixed per view, so it is a per-view
        # CONSTANT recomputed every step -- and computing it needs the fp32 plane target that A1's
        # kernel exists to avoid materialising. Precomputing it at the store build is the immediate
        # follow-up; it is left here for now so this merge is bitwise-verifiable against main.
        plane_target = (views["rae_doppler_target"].to(torch.float32)
                        * views["rae_doppler_rms"].reshape(-1, 1, 1, 1, 1))
        denp = ((plane_target * b4).square().flatten(1).sum(dim=-1) / c4).clamp_min(1e-30)
        # squared AFTER masking, as main does. Masking the already-squared value agrees in VALUE
        # (the mask is 0/1) but its gradient chain is (grad * b4) * 2 * d where main's is
        # grad * 2 * (d * b4) * b4 -- equal in exact arithmetic, 1e-7 apart in fp32.
        winpl = ((diff if fused_term else (rae_plane_prediction - plane_target)) * b4
                 ).square().flatten(1).sum(dim=-1) / c4 / denp
        return (args.elevation_weight * (rae_l2 + rae_plane_l2)
                + args.window_rae_weight * (win0 + winpl))

    rae_checked: dict = {"done": False}

    def rae_route_check(ra_profile: torch.Tensor, planes: torch.Tensor, views: dict,
                        floor_power: torch.Tensor | None) -> None:
        """MM4DPS_RAE_CHECK=1: the route's per-unit terms and its gradients against ``reference``, once.

        The route gate proper: the values (units,) and the gradients into the
        four profile tensors -- and into the floor, which carries gradient
        under --rae-floor-gradient on -- compared cell by cell (bitwise, or
        the max relative deviation), which the 20-step loss dump cannot
        resolve below ~1e-9.
        """
        rae_checked["done"] = True
        with torch.enable_grad():
            leaves = [t.detach().clone().requires_grad_(True) for t in (ra_profile, planes)]
            floor = (None if floor_power is None
                     else floor_power.detach().clone().requires_grad_(True))
            wrt = leaves + ([] if floor is None else [floor])
            outputs = {}
            pairs = {"reference/eager": ("reference", "eager"), "active": (_RAE_ROUTE, _RAE_TERM)}
            for name, (route, term) in pairs.items():
                value = rae_terms_of(*leaves, views, floor, route, term)
                grads = torch.autograd.grad(value.sum(), wrt)
                outputs[name] = (value.detach(), [g.detach() for g in grads])
        ref_value, ref_grads = outputs["reference/eager"]
        names = ["ra", "planes"] + ([] if floor is None else ["floor"])
        for route in ("active",):
            value, grads = outputs[route]
            def compare(a, b):
                if torch.equal(a, b):
                    return "bitwise"
                return f"max rel {float((a - b).abs().max() / b.abs().max().clamp_min(1e-30)):.2e}"
            print(f"rae route check: {route} (route={_RAE_ROUTE}, term={_RAE_TERM}) vs reference/eager: "
                  f"values {compare(value, ref_value)}; gradients "
                  + ", ".join(f"{n} {compare(g, r)}" for n, g, r in zip(names, grads, ref_grads)), flush=True)

    def elevation_terms(rows: torch.Tensor, radar_of_unit: torch.Tensor, plane_bins_rows,
                        ra_profile: torch.Tensor, planes: torch.Tensor,
                        views: dict | None = None) -> torch.Tensor:
        """The two elevation terms per unit (loop-0 |RAE| and the plane |RAE|, each a normalised L2), weighted.

        `rows` index the flat per-view store; `views` is the store already
        gathered at those rows (the captured step gathers every flat view
        once, `static_losses`; the eval path passes None and gathers here).
        The plane target is the half-precision store denormalised by its
        rms; the learned floor enters the plane prediction as in the RA
        plane term, scaled to an RAE cell by `elevation_floor_factor`.
        Sync-free and fixed-shape: runs under capture.
        """
        eval_path = views is None
        if eval_path:
            views = {key: flat_views[key].index_select(0, rows) for key in _RAE_VIEW_KEYS}
        if rae_plane_index is not None:
            planes = planes.index_select(1, rae_plane_index)
        floor_power = plane_noise_power(radar_of_unit, plane_bins_rows)
        if floor_power is not None and rae_plane_index is not None:
            floor_power = floor_power.index_select(1, rae_plane_index)
        if floor_power is not None:
            # The floor was LEARNED by the RA terms alone (as before the elevation term existed);
            # the RAE view read it, scaled to its cells, without a gradient into it -- with a
            # gradient it dragged the shared floor 2.5-3x under the measured off-person power on
            # the cascade (docs/FIDELITY_FIX_RESULTS.md 3.8). That diagnosis was made while the
            # scale factor was wrong (1.02 against the exact count-and-window factor), and the fix
            # bundled the exact factor with the detach. --rae-floor-gradient on unbundles them, so
            # the RAE plane term can fit the floor the way the RA plane term does; it is what an
            # objective that drops the RA plane term needs, since that term is the floor's only
            # substantial gradient path (probe: |g| noise_model 0.168 RT-Pose / 0.248 HuPR).
            floor_power = floor_power if args.rae_floor_gradient == "on" else floor_power.detach()
        if _RAE_CHECK and eval_path and not rae_checked["done"]:   # the start-up evaluation, off the captured step
            rae_route_check(ra_profile, planes, views, floor_power)
        return rae_terms_of(ra_profile, planes, views, floor_power, _RAE_ROUTE)

    def static_terms(rows: torch.Tensor, dc_profile: torch.Tensor) -> torch.Tensor:
        """P1, per unit: the DC plane's L2 in the person's RA-mask cells over their own target energy, plus the rest over
        theirs -- in the RA view and, with the elevation term, the RAE view -- times --static-weight. Sync-free."""
        mask = flat_views["ra_mask_kept"].index_select(0, rows).to(torch.float32)
        pred = (profiles_to_ra_magnitude_fused(dc_profile.real, dc_profile.imag, keep_index) if _FUSED_LOSS
                else profiles_to_ra_magnitude_batched(dc_profile.real, dc_profile.imag).index_select(1, keep_index))
        sq = (pred - flat_views["dc_target"].index_select(0, rows)).square()
        term = ((sq * mask).flatten(1).sum(dim=-1) / flat_views["dc_person_energy"].index_select(0, rows)
                + (sq * (1.0 - mask)).flatten(1).sum(dim=-1) / flat_views["dc_scene_energy"].index_select(0, rows))
        if elevation_on:
            pred_rae = profiles_to_rae_magnitude(dc_profile.real, dc_profile.imag, keep_index)
            sq = (pred_rae - flat_views["dc_rae_target"].index_select(0, rows)).square()
            term = term + ((sq * mask[..., None]).flatten(1).sum(dim=-1) / flat_views["dc_rae_person_energy"].index_select(0, rows)
                           + (sq * (1.0 - mask)[..., None]).flatten(1).sum(dim=-1) / flat_views["dc_rae_scene_energy"].index_select(0, rows))
        return args.static_weight * term

    def person_window_terms(rows: torch.Tensor, ra_profile: torch.Tensor, planes: torch.Tensor | None = None) -> torch.Tensor:
        """The person-window term per unit (--person-window-loss): the loop-0 |RAE| of the person's window in the consumer's
        transform, the prediction and the target each normalised by their own rms over the window, the masked L2 mean,
        times the weight. Fixed-shape gathers on cached indices, no host sync: runs under capture."""
        ri = flat_views["pw_range_index"].index_select(0, rows)                                    # (units, Rw)
        ai = flat_views["pw_angle_index"].index_select(0, rows)                                    # (units, Aw)
        target = flat_views["pw_target"].index_select(0, rows).to(torch.float32)                   # (units, B, Rw, Aw, E)
        m = flat_views["pw_mask"].index_select(0, rows)[:, None, ..., None]                        # (units, 1, Rw, Aw, 1)
        n = flat_views["pw_count"].index_select(0, rows)                                           # (units,)
        rows_profile = torch.gather(ra_profile, -1, ri[:, None, None, :].expand(-1, ra_profile.shape[1], ra_profile.shape[2], -1))
        if _pw_plane_index is not None and planes is not None:
            # the consumer's Doppler sum from the rendered planes (clutter-removed Hann planes, all 63 or the view's own support)
            # on the window's range rows: the DC from the loop-0 anchor where the consumer keeps it, the band's planes summed
            # in magnitude, a bin the view does not render weighted 0
            pos = flat_views["pw_plane_pos"].index_select(0, rows); w = flat_views["pw_plane_w"].index_select(0, rows)      # (units, B)
            rows_planes = torch.gather(planes, -1, ri[:, None, None, None, :].expand(-1, planes.shape[1], planes.shape[2], planes.shape[3], -1))
            rows_planes = torch.gather(rows_planes, 1, pos[:, :, None, None, None].expand(-1, -1, planes.shape[2], planes.shape[3], rows_planes.shape[-1]))   # (units, B, tx, rx, Rw)
            band = _PW_PLANE_SCALE * (_pw_view_fn(rows_planes) * w[:, :, None, None, None])                             # (units, B, Rw, A, E)
            if _pw_resolved:
                # keep the planes: the term is the window's (plane, range, azimuth, elevation) shape, and one
                # normalisation over the whole cube is what carries the relative energy between planes
                rae = (torch.cat([(_PW_DC_SCALE * _pw_view_fn(rows_profile))[:, None], band], dim=1)
                       if _pw_keep_dc else band)
            else:
                rae = band.sum(dim=1)[:, None]                                                     # (units, 1, Rw, A, E)
                if _pw_keep_dc:
                    rae = rae + (_PW_DC_SCALE * _pw_view_fn(rows_profile))[:, None]
        else:
            rae = _pw_view_fn(rows_profile)[:, None]                                               # (units, 1, Rw, A, E)
        crop = torch.gather(rae, 3, ai[:, None, None, :, None].expand(-1, rae.shape[1], rae.shape[2], -1, rae.shape[4]))   # (units, B, Rw, Aw, E)
        rms = ((crop.square() * m).flatten(1).sum(dim=-1) / n).clamp_min(1e-30).sqrt()
        l2 = ((crop / rms[:, None, None, None, None] - target).square() * m).flatten(1).sum(dim=-1) / n
        return args.person_window_loss * flat_views["pw_weight"].index_select(0, rows) * l2

    static_units = len(RADARS) * args.frame_batch
    static_radar_index = torch.as_tensor(
        [radar_index for radar_index in range(len(RADARS))
         for _ in range(args.frame_batch)], dtype=torch.long, device=device)
    def static_gain_of_units() -> torch.Tensor:
        return gain_stack().index_select(0, static_radar_index)
    # The step's only per-step host inputs: the frames' positions in the
    # flat cache, and the per-unit validity mask (all ones except on the
    # epoch's padded short last batch). Everything else is derived on the
    # device.
    stage_batch = torch.zeros(args.frame_batch, dtype=torch.long, device=device)
    stage_mask = torch.ones(len(RADARS) * args.frame_batch, device=device)
    static_state: dict = {"pad_points": 0, "noise_floor": None}

    def static_resize(pad_points: int) -> None:
        """(Re)build the constants that depend on the padded half length."""
        static_state.update({
            "pad_points": int(pad_points),
            "position": torch.arange(int(pad_points), device=device),
            "zeros": torch.zeros(int(pad_points), dtype=torch.long, device=device),
            "zero_one": torch.zeros(1, dtype=torch.long, device=device),
            "tx_positions": torch.cat([
                tx_array_stack[radar_index].expand(int(pad_points), -1, -1)
                for radar_index in range(len(RADARS))]),
            "rx_positions": torch.cat([
                rx_array_stack[radar_index].expand(int(pad_points), -1, -1)
                for radar_index in range(len(RADARS))]),
        })

    def static_geometry() -> dict:
        """The batch's padded per-point inputs, from `stage_batch` alone."""
        pad_points = static_state["pad_points"]
        position = static_state["position"]
        counts = frame_count_device.index_select(0, stage_batch)
        ends = torch.cumsum(counts, 0)
        begins = torch.cat([static_state["zero_one"], ends[:-1]])
        unit = torch.searchsorted(ends, position, right=True).clamp_(
            max=args.frame_batch - 1)
        valid = position < ends[-1]
        rows_half = torch.where(
            valid,
            position - begins.index_select(0, unit)
            + frame_offset_device.index_select(0, stage_batch).index_select(0, unit),
            static_state["zeros"],
        )
        # One copy of the batch's rows per radar (radar-major, like the units).
        rows = rows_half.repeat(len(RADARS))
        radar_gains = torch.stack([material.radar_gain(radar) for radar in RADARS])
        if _FOLDED_ABSOLUTE_GAIN:
            # Squared: the areas slot enters the amplitude under a sqrt.
            radar_gains = radar_gains * gain_stack().square()
        per_point_radar_gain = torch.cat([
            radar_gains[radar_index].expand(pad_points)
            for radar_index in range(len(RADARS))])
        point_gains = flat_weights.index_select(0, rows) @ joint_gains()
        # Same three factors in the same order as every other path, so the
        # areas are bitwise identical element by element.
        areas = (material.areas_for_points(len(RADARS) * pad_points) * point_gains
                 * per_point_radar_gain)
        return {
            "positions": flat_points.index_select(0, rows),
            "normals": model.normals_for_rows(
                normal_row_of_point.index_select(0, rows)),
            "areas": areas,
            "materials": material.for_indices(
                flat_global_index.index_select(0, rows)),
            "tx_positions": static_state["tx_positions"],
            "rx_positions": static_state["rx_positions"],
            "valid": valid.repeat(len(RADARS)),
            "rows_half": rows_half,
            "unit": unit,
            "unit_of_point": torch.cat([unit + radar_index * args.frame_batch
                                        for radar_index in range(len(RADARS))]),
        }

    # ------------------------------------------------------------------
    # The person's position-only splat geometry, cached per flat row.
    #
    # The captured step gathers its batch's points out of `flat_points` and
    # re-rendered everything from the positions every step: distances,
    # antenna gains, carrier phasors, range bins, PSF weights, splat indices
    # -- ~3 ms of a 27 ms step for values that never change (the positions
    # are the GVHMR sequence; only the BSDF amplitude and the rates move).
    # `render_factorized`'s geometry cache stores exactly those values for
    # one fixed point set; here they are filled ONCE for every flat row
    # (chunked fill passes) and the step's batch cache is a row gather of
    # them, so the cached path runs on the same stored values it would have
    # computed -- bitwise the uncached render (the cache's own gate).
    # ~2.4 KB a row per radar (RT-Pose seq 183: 1.0 M rows, 2.4 GB).
    # HUPR_ROW_GEOMETRY=0 restores the per-step render.
    # ------------------------------------------------------------------
    sorted_doppler = (_SORTED_DOPPLER and _ROW_GEOMETRY and doppler_taps() > 0
                      and args.doppler_amplitudes == "closed-form"
                      and fft_window_kind() == "hann" and args.deterministic == "off")

    def build_row_geometry() -> None:
        total = int(frame_starts[-1])
        n_tx, n_rx = reference_rasterizer.n_tx, reference_rasterizer.n_rx
        chunk = 16384
        started = time.perf_counter()
        for radar_index in range(len(RADARS)):
            store = None
            for start in range(0, total, chunk):
                rows = torch.arange(start, min(start + chunk, total), device=device)
                count = int(rows.numel())
                fill: dict = {}
                with torch.no_grad():
                    render_factorized(
                        flat_points.index_select(0, rows),
                        model.normals_for_rows(normal_row_of_point.index_select(0, rows)),
                        material.areas_for_points(count),
                        material.for_indices(flat_global_index.index_select(0, rows)),
                        reference_rasterizer, reparameterize_torch, detach_phase=True,
                        shadow_mask=None, bsdf_mode="full", disabled_components=None,
                        use_cuda_kernels=True, return_splat_components=True,
                        tx_positions=tx_array_stack[radar_index].expand(count, -1, -1),
                        rx_positions=rx_array_stack[radar_index].expand(count, -1, -1),
                        geometry_cache=fill, cache_psf=False)
                spread = int(fill["flat_indices"].shape[0])
                samples = int(fill["output_shape"][2])
                if store is None:
                    store = {
                        "wi": torch.empty(total, n_tx, 3, device=device),
                        "alpha_tx": torch.empty(total, n_tx, device=device),
                        "wo": torch.empty(total, n_rx, 3, device=device),
                        "G_rx": torch.empty(total, n_rx, device=device),
                        "cos_phi": torch.empty(total, n_tx * n_rx, device=device),
                        "sin_phi": torch.empty(total, n_tx * n_rx, device=device),
                        "n_frac": torch.empty(total, n_tx * n_rx, device=device),
                        "n_floor": torch.empty(total, n_tx * n_rx, dtype=torch.int16, device=device),
                        "psf_table_real": fill["psf_table_real"],
                        "psf_table_imag": fill["psf_table_imag"],
                        "n_grid": fill["n_grid"], "spread": spread, "samples": samples,
                    }
                for key in ("wi", "alpha_tx", "wo", "G_rx"):
                    store[key][rows] = fill[key]
                for key in ("cos_phi", "sin_phi", "n_frac"):
                    store[key][rows] = fill[key].reshape(count, n_tx * n_rx)
                # the integer range bin, from the centre tap's flat index
                store["n_floor"][rows] = (fill["flat_indices"][spread // 2] % samples
                                          ).reshape(count, n_tx * n_rx).to(torch.int16)
            if sorted_doppler:
                # The sorted Doppler route's order, per frame: each (row,
                # channel)'s rank in the frame's (channel, range bin) sort,
                # and the frame's per-block runs (relative to its own entries).
                channels = n_tx * n_rx
                store["rank"] = torch.empty(total, channels, dtype=torch.int32, device=device)
                per_frame = channels * (samples // _SORTED_BINS)
                store["frame_runs"] = torch.empty(len(flat_frames), per_frame, 4,
                                                  dtype=torch.int32, device=device)
                channel_of = torch.arange(channels, device=device)
                for index in range(len(flat_frames)):
                    start, stop = int(frame_starts[index]), int(frame_starts[index + 1])
                    n_floor = store["n_floor"][start:stop].long().reshape(-1)
                    channel = channel_of[None, :].expand(stop - start, -1).reshape(-1)
                    order, runs = sorted_doppler_order(
                        n_floor, channel, torch.zeros_like(n_floor), samples, 1)
                    rank = torch.empty_like(order)
                    rank[order.long()] = torch.arange(order.numel(), device=device, dtype=torch.int32)
                    store["rank"][start:stop] = rank.reshape(stop - start, channels)
                    store["frame_runs"][index] = runs
            row_geometry[radar_index] = store
        row_geometry["shapes"] = {}
        torch.cuda.synchronize()
        gib = sum(v.numel() * v.element_size() for store in row_geometry.values()
                  if isinstance(store, dict) for v in store.values() if torch.is_tensor(v)) / 2**30
        print(f"row geometry: {total} rows x {len(RADARS)} radar(s) cached, "
              f"{gib:.2f} GiB, {time.perf_counter() - started:.1f} s"
              + ("; sorted Doppler route on" if sorted_doppler else ""), flush=True)

    def row_geometry_shapes(pad_points: int) -> dict:
        """The shape-only splat constants for a padded batch (per pad length)."""
        hit = row_geometry["shapes"].get(pad_points)
        if hit is not None:
            return hit
        n_tx, n_rx = reference_rasterizer.n_tx, reference_rasterizer.n_rx
        store = row_geometry[0]
        spread, samples = store["spread"], store["samples"]
        M = len(RADARS) * pad_points
        t_idx = torch.arange(n_tx, device=device)[None, :, None].expand(M, -1, n_rx).reshape(-1)
        r_idx = torch.arange(n_rx, device=device)[None, None, :].expand(M, n_tx, -1).reshape(-1)
        m_idx = torch.arange(M, device=device)[:, None, None].expand(-1, n_tx, n_rx).reshape(-1)
        hit = {
            "tx_indices": t_idx, "rx_indices": r_idx, "point_indices": m_idx,
            # the three indices above ARE the row-major enumeration of (M, n_tx, n_rx):
            # `forward_render._path_rates_of` reads the rates as a view
            "paths_are_rows": True,
            "base_idx": (t_idx * (n_rx * samples) + r_idx * samples).to(torch.int32),
            "dn": torch.arange(-(spread // 2), spread // 2 + 1, device=device, dtype=torch.int32),
            "output_shape": (n_tx, n_rx, samples),
            "channel_of_path": (torch.arange(n_tx * n_rx, device=device)).long(),
            "path_index": torch.arange(M * n_tx * n_rx, device=device, dtype=torch.int32),
            "table4": _psf_table4(store["psf_table_real"], store["psf_table_imag"]),
            # the sorted route reads no splat indices; the cached path's
            # consumers only ever `.int()` this
            "flat_placeholder": torch.zeros(1, 1, dtype=torch.int32, device=device),
        }
        row_geometry["shapes"][pad_points] = hit
        return hit

    def static_sorted_order(geometry: dict, n_floor: torch.Tensor,
                            n_frac: torch.Tensor, shapes: dict) -> dict:
        """The batch's sorted Doppler order from the per-frame ranks and runs (device only).

        Entries are laid out unit-major (radar-major units, as `units` is
        built), each unit's region holding its frame's entries in the
        frame's cached order, the padded points' entries after every real
        one where no run reaches them.
        """
        pad_points = int(geometry["rows_half"].numel())
        channels = shapes["output_shape"][0] * shapes["output_shape"][1]
        samples = shapes["output_shape"][2]
        counts = frame_count_device.index_select(0, stage_batch)            # (frame_batch,)
        unit_counts = counts.repeat(len(RADARS))                              # radar-major units
        unit_start = torch.cat([static_state["zero_one"], torch.cumsum(unit_counts, 0)[:-1]]) * channels
        real_entries = unit_counts.sum() * channels
        unit = geometry["unit_of_point"]                                      # (radars * pad_points,)
        valid = geometry["valid"]
        position = static_state["position"].repeat(len(RADARS))
        # the padded points' entries, radar-major after every real entry
        pad_count = pad_points - counts.sum()
        radar_of_point = torch.arange(len(RADARS), device=device).repeat_interleave(pad_points)
        pad_ordinal = (position - counts.sum()).clamp_min(0) + radar_of_point * pad_count
        rank = torch.cat([store["rank"].index_select(0, geometry["rows_half"])
                          for store in (row_geometry[r] for r in range(len(RADARS)))])
        entry = torch.where(
            valid[:, None], unit_start.index_select(0, unit)[:, None] + rank,
            real_entries + pad_ordinal[:, None] * channels + shapes["channel_of_path"][None, :])
        path_of_entry = torch.zeros(entry.numel(), dtype=torch.int32, device=device)
        path_of_entry.scatter_(0, entry.reshape(-1).long(), shapes["path_index"])
        frame_runs = torch.cat([row_geometry[r]["frame_runs"].index_select(0, stage_batch)
                                for r in range(len(RADARS))])               # (units, blocks, 4)
        # each unit's runs shifted to its region; an empty run stays (0, 0)
        shifted = frame_runs + unit_start.to(torch.int32)[:, None, None]
        zero = torch.zeros_like(frame_runs[..., :2])
        runs = torch.cat([
            torch.where(frame_runs[..., 1:2] > frame_runs[..., 0:1], shifted[..., :2], zero),
            torch.where(frame_runs[..., 3:4] > frame_runs[..., 2:3], shifted[..., 2:], zero)], dim=-1)
        return {
            "path_of_entry": path_of_entry,
            "runs": runs.reshape(-1, 4).contiguous(),
            "record_static": torch.stack([n_frac, n_floor.view(torch.float32)], dim=-1).contiguous(),
            "table4": shapes["table4"],
        }

    def unit_row_cache(geometry: dict) -> dict:
        """The units' splat geometry out of the row cache, for the eager paths (no padding)."""
        units = geometry["units"]
        n_tx, n_rx = reference_rasterizer.n_tx, reference_rasterizer.n_rx
        channels = n_tx * n_rx
        rows = [torch.arange(int(frame_starts[frame_position[frame]]),
                             int(frame_starts[frame_position[frame] + 1]), device=device)
                for frame, _, _ in units]
        radars = [radar_index for _, radar_index, _ in units]
        def gathered(key: str) -> torch.Tensor:
            return torch.cat([row_geometry[r][key].index_select(0, rr) for rr, r in zip(rows, radars)])
        store = row_geometry[0]
        spread, samples = store["spread"], store["samples"]
        M = sum(int(rr.numel()) for rr in rows)
        t_idx = torch.arange(n_tx, device=device)[None, :, None].expand(M, -1, n_rx).reshape(-1)
        r_idx = torch.arange(n_rx, device=device)[None, None, :].expand(M, n_tx, -1).reshape(-1)
        m_idx = torch.arange(M, device=device)[:, None, None].expand(-1, n_tx, n_rx).reshape(-1)
        n_floor = gathered("n_floor").reshape(-1).to(torch.int32)
        n_frac = gathered("n_frac").reshape(-1)
        out = {
            "wi": gathered("wi"), "alpha_tx": gathered("alpha_tx"),
            "wo": gathered("wo"), "G_rx": gathered("G_rx"),
            "cos_phi": gathered("cos_phi").reshape(-1),
            "sin_phi": gathered("sin_phi").reshape(-1),
            "n_frac": n_frac,
            "psf_table_real": store["psf_table_real"], "psf_table_imag": store["psf_table_imag"],
            "n_grid": store["n_grid"],
            "point_indices": m_idx, "tx_indices": t_idx, "rx_indices": r_idx,
            "output_shape": (n_tx, n_rx, samples),
        }
        if sorted_doppler:
            counts = torch.tensor([int(rr.numel()) for rr in rows], device=device)
            unit_start = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)[:-1]]) * channels
            rank = gathered("rank")                                          # (M, channels)
            unit_of_row = torch.repeat_interleave(torch.arange(len(units), device=device), counts)
            entry = unit_start.index_select(0, unit_of_row)[:, None] + rank
            path_of_entry = torch.empty(entry.numel(), dtype=torch.int32, device=device)
            path_of_entry.scatter_(0, entry.reshape(-1).long(),
                                   torch.arange(entry.numel(), device=device, dtype=torch.int32))
            frame_runs = torch.stack([row_geometry[r]["frame_runs"][frame_position[frame]]
                                      for (frame, r, _) in units])            # (units, blocks, 4)
            shifted = frame_runs + unit_start.to(torch.int32)[:, None, None]
            zero = torch.zeros_like(frame_runs[..., :2])
            runs = torch.cat([
                torch.where(frame_runs[..., 1:2] > frame_runs[..., 0:1], shifted[..., :2], zero),
                torch.where(frame_runs[..., 3:4] > frame_runs[..., 2:3], shifted[..., 2:], zero)], dim=-1)
            out["flat_indices"] = torch.zeros(1, 1, dtype=torch.int32, device=device)
            out["sorted"] = {
                "path_of_entry": path_of_entry, "runs": runs.reshape(-1, 4).contiguous(),
                "record_static": torch.stack([n_frac, n_floor.view(torch.float32)], dim=-1).contiguous(),
                "table4": _psf_table4(store["psf_table_real"], store["psf_table_imag"]),
            }
        else:
            base_idx = (t_idx * (n_rx * samples) + r_idx * samples).to(torch.int32)
            dn = torch.arange(-(spread // 2), spread // 2 + 1, device=device, dtype=torch.int32)
            out["flat_indices"] = base_idx[None, :] + torch.remainder(n_floor[None, :] + dn[:, None], samples)
        return out

    def static_row_cache(geometry: dict) -> dict:
        """The batch's splat geometry gathered out of the row cache (device only)."""
        rows_half = geometry["rows_half"]
        shapes = row_geometry_shapes(int(rows_half.numel()))
        stores = [row_geometry[radar_index] for radar_index in range(len(RADARS))]
        def gathered(key: str) -> torch.Tensor:
            return torch.cat([store[key].index_select(0, rows_half) for store in stores])
        n_floor = gathered("n_floor").reshape(-1).to(torch.int32)
        samples = shapes["output_shape"][2]
        n_frac = gathered("n_frac").reshape(-1)
        if sorted_doppler:
            return {
                "wi": gathered("wi"), "alpha_tx": gathered("alpha_tx"),
                "wo": gathered("wo"), "G_rx": gathered("G_rx"),
                "cos_phi": gathered("cos_phi").reshape(-1),
                "sin_phi": gathered("sin_phi").reshape(-1),
                "n_frac": n_frac,
                "psf_table_real": stores[0]["psf_table_real"],
                "psf_table_imag": stores[0]["psf_table_imag"],
                "n_grid": stores[0]["n_grid"],
                "flat_indices": shapes["flat_placeholder"],
                "point_indices": shapes["point_indices"],
                "tx_indices": shapes["tx_indices"], "rx_indices": shapes["rx_indices"],
                "paths_are_rows": True,
                "output_shape": shapes["output_shape"],
                "sorted": static_sorted_order(geometry, n_floor, n_frac, shapes),
            }
        # `flat_idx_all` as the uncached path forms it: base + (n_floor + dn) mod K
        flat = shapes["base_idx"][None, :] + torch.remainder(
            n_floor[None, :] + shapes["dn"][:, None], samples)
        return {
            "wi": gathered("wi"), "alpha_tx": gathered("alpha_tx"),
            "wo": gathered("wo"), "G_rx": gathered("G_rx"),
            "cos_phi": gathered("cos_phi").reshape(-1),
            "sin_phi": gathered("sin_phi").reshape(-1),
            "n_frac": n_frac,
            "psf_table_real": stores[0]["psf_table_real"],
            "psf_table_imag": stores[0]["psf_table_imag"],
            "n_grid": stores[0]["n_grid"],
            "flat_indices": flat,
            "point_indices": shapes["point_indices"],
            "paths_are_rows": True,
            "tx_indices": shapes["tx_indices"], "rx_indices": shapes["rx_indices"],
            "output_shape": shapes["output_shape"],
        }

    def static_render(geometry: dict) -> tuple[dict, torch.Tensor]:
        components = render_factorized(
            geometry["positions"], geometry["normals"], geometry["areas"],
            geometry["materials"], reference_rasterizer, reparameterize_torch,
            detach_phase=True,
            shadow_mask=geometry["valid"][:, None].expand(
                -1, reference_rasterizer.n_tx),
            bsdf_mode="full", disabled_components=None, use_cuda_kernels=True,
            return_splat_components=True,
            tx_positions=geometry["tx_positions"],
            rx_positions=geometry["rx_positions"],
            geometry_cache=static_row_cache(geometry) if row_geometry else None,
            carrier_only=sorted_doppler,
        )
        return components, geometry["unit_of_point"][components["point_indices"]]

    def static_rates(geometry: dict) -> torch.Tensor:
        """Per-point bistatic path rates, with no per-frame host loop.

        The reference builds one `(points,14) @ (14,3)` matmul per unit and
        concatenates twelve of them. Here the six frames' `(14,3)` matrices are
        gathered per point and contracted with a multiply-and-sum, which is the
        same 14-term dot product per point in a shape that does not change from
        step to step. The two radars render the same six frames, so the second
        half's velocities are the first half's and are not recomputed.
        """
        joint = model.joint_velocities_for(
            frame_id_device.index_select(0, stage_batch))
        weights = flat_weights.index_select(0, geometry["rows_half"])
        velocity = (weights.unsqueeze(-1)
                    * joint.index_select(0, geometry["unit"])).sum(dim=1)
        return surface_bistatic_path_rates_per_point(
            geometry["positions"], velocity.repeat(len(RADARS), 1),
            geometry["tx_positions"], geometry["rx_positions"])

    def static_backgrounds() -> torch.Tensor:
        """One assembled background per (frame, radar) unit of the static step.

        Rows follow `static_radar_index`: radar-major, `frame_batch` frames per
        radar -- the order `units` is built in, so the row for unit i is the
        background unit i renders against.

        WITHOUT a mask the background does not depend on the frame, so this is
        E5's two assembles broadcast over the units: the exact expression
        `static_profiles` used before R092, so a mask-off run is byte for byte
        the old one.

        WITH a mask it is frame-dependent and every unit gets its own. R092 is
        the whole point of this function: until it, the captured training step
        called `current_backgrounds()` -- `frame=None`, which makes
        `visible_gains` a no-op -- so the mask reached the evaluation pass and
        never reached the training loss. R087 wired `unit_scaled_profiles` and
        left its sibling behind, and production takes the sibling.

        The render is NOT repeated (R091): `background_splat` runs once per
        radar, and the frames differ only in the gain vector handed to
        `static_assemble`.

        CAPTURE SAFETY. The frame ids are derived on the device from
        `stage_batch` through `frame_id_device`, exactly as `static_rates`
        derives its velocities, so the captured graph reads whatever the
        staging buffer holds and a new batch of frames needs no re-capture.
        No host sync, no data-dependent shape; the per-slot loop is unrolled
        into a fixed `frame_batch` launches per radar at capture time.
        """
        if scene_visibility is None:
            return current_backgrounds().index_select(0, static_radar_index)
        frame_ids = frame_id_device.index_select(0, stage_batch)
        rows = []
        for radar in RADARS:
            fused = (_FUSED_SCENE and background_map_cache is not None
                     and "n_frac" in background_geometry_cache[radar])
            # `--scene-update-every k`: the off-step version of the captured
            # step renders the scene without a gradient (same forward)
            with torch.set_grad_enabled(torch.is_grad_enabled() and scene_grad["on"]):
                components, gains = background_splat(radar, carrier_only=fused)
            # (frame_batch, active): the batch's rows of the mask restricted to
            # this radar's culled points, in one gather. `gains * visibility`
            # is `visible_gains`'s product in `visible_gains`'s order, so an
            # all-ones row reproduces the unmasked gains bitwise.
            visibility = scene_visibility.index_select(
                0, frame_ids).index_select(1, background_active[radar])
            frame_gains = gains.unsqueeze(0) * visibility
            if fused:
                # The batch's backgrounds in one kernel each way, the
                # contributions synthesised inside (fp32 floor against the
                # per-frame chain; `qa scene-check`).
                rows.append(assemble_static_profile_frames_cuda(
                    components, frame_gains.contiguous(),
                    background_map_cache[radar]))
                continue
            rows.append(torch.stack([k1_assemble(radar, components, frame_gains[slot])
                                     for slot in range(args.frame_batch)]))
        return torch.cat(rows)

    def static_profiles(
        components: dict, rates: torch.Tensor, unit_of_path: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        per_unit = static_backgrounds()
        # Derived on the device from `stage_batch`, exactly as the loss derives
        # its `view_rows`, so the captured graph reads whatever the buffer
        # holds and the bins can change per step without a re-capture.
        static_bins = ra_doppler_bins if view_plane_bins is None else \
            view_plane_bins.index_select(0, torch.cat([
                stage_batch * len(RADARS) + radar_index
                for radar_index in range(len(RADARS))]))
        if _FOLDED_ABSOLUTE_GAIN:
            # Per unit rather than per radar, which is the same multiply on the
            # same operands: `static_gain` IS the gain stack gathered by
            # `static_radar_index`, so scaling after the gather is bitwise what
            # scaling before it was.
            static_gain = static_gain_of_units()
            per_unit = per_unit * static_gain[:, None, None, None]
            if coupling_stack is not None:
                per_unit = per_unit + coupling_stack.index_select(0, static_radar_index)
            ra_profile, planes = doppler_planes_units(
                per_unit, components, rates, unit_of_path,
                static_units, static_bins,
            )
            return ra_profile, planes, dc_anchor_units(per_unit, components, rates, unit_of_path, static_units)
        ra_profile, planes = doppler_planes_units(
            per_unit, components, rates, unit_of_path,
            static_units, static_bins,
        )
        static_gain = static_gain_of_units()
        ra_profile = static_gain[:, None, None, None] * ra_profile
        dc_profile = dc_anchor_units(per_unit, components, rates, unit_of_path, static_units)
        if dc_profile is not None:
            dc_profile = static_gain[:, None, None, None] * dc_profile
        if coupling_stack is not None:
            ra_profile = ra_profile + coupling_stack.index_select(0, static_radar_index)
            if dc_profile is not None:
                dc_profile = dc_profile + coupling_stack.index_select(0, static_radar_index)
        return ra_profile, static_gain[:, None, None, None, None] * planes, dc_profile

    def static_losses(
        ra_profile: torch.Tensor, planes: torch.Tensor, dc_profile: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        view_rows = torch.cat([
            stage_batch * len(RADARS) + radar_index
            for radar_index in range(len(RADARS))])
        views = {key: value.index_select(0, view_rows)
                 for key, value in flat_views.items()}
        # The fused chain slices range before the angle FFT, folds fftshift
        # into channel signs and hoists the transpose ahead of the FFT — the
        # same arithmetic as the reference chain on far fewer bytes, pinned
        # against it in the suite. Output orientation is unchanged
        # (range, angle), so every target reads exactly as before.
        # HUPR_FUSED_LOSS=0 (precedent: HUPR_DOPPLER_CHUNKED) runs the
        # reference chain instead, so the two can be A/B'd in one build.
        md_mask, md_rows = views["ra_mask"], views["range_rows"]
        # The learned floor is a function of the parameters, so it is
        # recomputed here rather than read from the captured static tensor.
        # `view_rows` above is all radar-0 units then all radar-1 units, so
        # the per-unit radar index is that same concatenation.
        static_floor = static_state["noise_floor"]
        radar_of_unit = torch.cat([
            torch.full((len(stage_batch),), radar_index,
                       dtype=torch.long, device=device)
            for radar_index in range(len(RADARS))])
        if noise_model is not None:
            static_floor = noise_model.microdoppler_marginal_batched(
                radar_of_unit, md_mask, md_rows, motion_index)
        if _FUSED_LOSS:
            ra_prediction = profiles_to_ra_magnitude_fused(
                ra_profile.real, ra_profile.imag, keep_index)
            if "plane-ra" in _ABLATE_TERMS:
                # Plane term ablated: neither the compiled nor the eager
                # form is computed; the planes are still rendered because
                # the MD term reads its 13 bins out of them.
                compiled_plane_l2 = None
                ra_plane_prediction = None
            elif _COMPILED_PLANE_TERM:
                plane_spectrum = profiles_to_ra_spectrum_fused(
                    planes.real, planes.imag, keep_index)
                plane_target = plane_cells_target(denormalise_plane_target(
                    views["ra_doppler_target"], views.get("ra_doppler_rms")))
                # The compiled term IS the reference plane term — mean as
                # sum / count, divided by the precomputed denominator — so
                # it composes with the two-term loss by plain addition
                # below. `ra_plane_prediction=None` keeps the eager plane
                # branch out of the graph entirely.
                compiled_plane_l2 = (
                    _plane_term_sums(
                        torch.view_as_real(plane_cells(plane_spectrum)),
                        plane_target, plane_magnitude_floor,
                        plane_noise_power(radar_of_unit, None if view_plane_bins is None
                                          else view_plane_bins.index_select(0, view_rows)))
                    / float(plane_target[0].numel())
                    / views["ra_doppler_mean_square"])
                ra_plane_prediction = None
            else:
                compiled_plane_l2 = None
                ra_plane_prediction = profiles_to_ra_magnitude_fused(
                    planes.real, planes.imag, keep_index)
                floor_power = plane_noise_power(radar_of_unit, None if view_plane_bins is None
                                                else view_plane_bins.index_select(0, view_rows))
                if floor_power is not None:
                    # clamped: a pad plane is exactly zero with a zero floor,
                    # and sqrt's gradient at 0 is inf (inf * 0 = NaN)
                    ra_plane_prediction = (ra_plane_prediction.square() + floor_power).clamp_min(1e-30).sqrt()
            if md_from_plane_spectrum and compiled_plane_l2 is not None:
                # The band's magnitude cube out of the plane spectrum
                # (its prefix, retained range bins); the masks likewise.
                doppler_cube = plane_spectrum[:, :len(motion_bins)].abs()
                md_mask, md_rows = views["ra_mask_kept"], views["range_rows_kept"]
            else:
                # HuPR's 13 of 63 planes: the separate band chain is cheaper
                # than a second consumer of the spectrum (an index-select
                # backward plus a full-size gradient add, measured +1.4 ms).
                motion_planes = motion_planes_of(planes, view_rows=view_rows)
                doppler_cube = profiles_to_ra_magnitude_fused(
                    motion_planes.real, motion_planes.imag)
        else:
            compiled_plane_l2 = None
            ra_prediction = profiles_to_ra_magnitude_batched(
                ra_profile.real, ra_profile.imag,
            ).index_select(1, keep_index)
            ra_plane_prediction = profiles_to_ra_magnitude_batched(
                planes.real, planes.imag,
            ).index_select(2, keep_index)
            motion_planes = motion_planes_of(planes, view_rows=view_rows)
            doppler_cube = doppler_planes_to_motion_magnitude(
                motion_planes.real, motion_planes.imag)
        if _COMPILED_PLANE_TERM:
            md_prediction = _md_marginal_prediction(
                doppler_cube, md_mask, md_rows, static_floor)
        else:
            md_prediction = (
                microdoppler_marginal_batched(
                    doppler_cube, md_mask, md_rows, None,
                ).clamp_min(0.0) + static_floor
            ).sqrt()
        denominators = dict(
            ra_target_mean_square=views["ra_mean_square"],
            ra_plane_target_mean_square=views["ra_doppler_mean_square"],
            md_target_mean_square=views["md_mean_square"],
        ) if _FUSED_LOSS else {}
        total, ra_l2, md_l2 = two_term_loss_units(
            ra_prediction, views["ra_target"],
            md_prediction, views["md_target"], args.microdoppler_weight,
            ra_loss=args.ra_loss,
            ra_plane_prediction=ra_plane_prediction,
            ra_plane_target=(None if ra_plane_prediction is None
                             else denormalise_plane_target(
                                 views["ra_doppler_target"],
                                 views.get("ra_doppler_rms"))),
            **denominators,
        )
        if compiled_plane_l2 is not None:
            # The plane term enters both ra_l2 and the total additively, so
            # composing it here is the reference expression re-associated —
            # gated at the fp32 floor by verify_fused_loss's probe A/B.
            ra_l2 = ra_l2 + compiled_plane_l2
            total = total + compiled_plane_l2
        if elevation_on:
            # the RAE store read from the step's one gather of the flat views (`views` above),
            # not gathered a second time (docs/RUNTIME_RESULTS.md item 6)
            rae = elevation_terms(view_rows, radar_of_unit,
                                  None if view_plane_bins is None else view_plane_bins.index_select(0, view_rows),
                                  ra_profile, planes, views=views)
            ra_l2 = ra_l2 + rae
            total = total + rae
        if pw_on:
            pw = person_window_terms(view_rows, ra_profile, planes)
            ra_l2 = ra_l2 + pw
            total = total + pw
        if static_on:
            st = static_terms(view_rows, dc_profile)
            ra_l2 = ra_l2 + st
            total = total + st
        # The short-batch mask: 1.0 on real units (x * 1.0 is x, bitwise),
        # 0.0 on the padded units of the epoch's last batch, zeroing their
        # loss and every gradient they would send.
        return total * stage_mask, ra_l2 * stage_mask, md_l2 * stage_mask

    def static_step() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One training step's per-unit losses, at a shape that never changes."""
        geometry = static_geometry()
        components, unit_of_path = static_render(geometry)
        ra_profile, planes, dc_profile = static_profiles(
            components, static_rates(geometry), unit_of_path)
        return static_losses(ra_profile, planes, dc_profile)

    # The static step covers the batched path for the parameterizations whose
    # per-point velocity is a pure skinned joint field. `point-normal` adds a
    # per-frame scalar times the learned normal and keeps the dynamic path.
    static_path = (args.step_path == "batched"
                   and args.velocity_parameterization != "point-normal")
    capture_step_path = static_path and args.cuda_graph == "on"
    if static_path and _ROW_GEOMETRY:
        build_row_geometry()

    if gain_learned:
        # The gain starts at the level ratio of the measured to the rendered
        # loop-0 |RA| over the first frames (render at gain 1, in product
        # units after the coupling): the trainer then moves it with the
        # materials. One number per radar, never per frame.
        with torch.no_grad():
            absolute_log_gain.zero_()
            probe = training_frames[:min(4, len(training_frames))]
            for radar_index, radar in enumerate(RADARS):
                units = [(frame, radar_index, None) for frame in probe]
                ra_profile, _, _ = unit_profiles(units)
                predicted = profiles_to_ra_magnitude_fused(ra_profile.real, ra_profile.imag, keep_index)
                target = torch.stack([cache[frame]["views"][radar]["ra_target"] for frame in probe])[:, keep_range]
                ratio = torch.sqrt(target.square().sum() / predicted.square().sum().clamp_min(1e-30))
                absolute_log_gain.data[radar_index] = torch.log(ratio.clamp_min(1e-30))
            print("absolute gain initialised from the measured/rendered RA level of the first "
                  f"{len(probe)} frames: {gain_values()}; " + ("FROZEN from here (init)" if gain_frozen_after_init else "learned from here"), flush=True)

    def load_trained_state(vf_path: str) -> None:
        """A trained velocity_field.pt's joint gains, absolute gain, normals and field into this trainer's parameters."""
        vf = torch.load(vf_path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            if args.joint_gains == "learned" and vf.get("inherited_joint_scattering_gains") is not None:
                joint_log_gain.copy_(torch.log(torch.as_tensor(vf["inherited_joint_scattering_gains"], dtype=torch.float32, device=device)))
            if gain_learned and vf.get("absolute_gains"):
                absolute_log_gain.copy_(torch.log(torch.tensor([float(vf["absolute_gains"][r]) for r in RADARS], device=device)))
            loaded = 0
            for frame, (offset, count) in model.normal_span.items():
                value = vf.get("learned_normals", {}).get(str(frame), vf.get("learned_normals", {}).get(frame))
                if value is not None and tuple(value.shape) == (count, 3):
                    model.raw_normals_flat[offset:offset + count].copy_(torch.as_tensor(value, dtype=torch.float32, device=device)); loaded += 1
            field = vf.get("joint_velocity_field_mps")
            if field is not None and args.velocity_parameterization == "temporal-basis" and tuple(field.shape) == tuple(model.joint_velocity_field.shape):
                target = torch.as_tensor(field, dtype=torch.float32, device=device)
                if model.bounded:
                    scale = vf.get("joint_velocity_scale_learned")
                    if scale is not None:
                        model.joint_scale.copy_(torch.as_tensor(scale, dtype=torch.float32, device=device))
                    b = (model.residual_bound + model.residual_relative * torch.linalg.vector_norm(model.kinematic_field, dim=-1, keepdim=True)
                         + model.residual_per_height * model.pelvis_height[:, None, None]).clamp_min(1e-6)
                    target = b * torch.atanh(((target - model.joint_scale[:, None] * model.kinematic_field) / b).clamp(-0.999, 0.999))
                model.temporal_coefficients.copy_(torch.einsum("tk,tjc->kjc", model.temporal_basis, target))
            elif field is not None and args.velocity_parameterization == "joint":
                model.joint_velocity_field.copy_(torch.as_tensor(field, dtype=torch.float32, device=device))
        print(f"trained state loaded from {vf_path} (normals for {loaded} frames)", flush=True)

    if os.environ.get("MM4DPS_CALIB_LOAD_VF"):                 # diagnostic: calibrate at a trained state
        load_trained_state(os.environ["MM4DPS_CALIB_LOAD_VF"])

    person_level_calibration: dict = {}

    def calibrate_person_level() -> None:
        """--person-level-calibration: the person's area scale set so its rendered level matches the measurement (see the
        flag); at --person-level-calibration-when init (before training) or end (after it, on the state that ships --
        the trained velocity field and normals lower the beamformed peak ~1.8x on a standing body: TDM MIMO motion
        between the TX slots decorrelates the phases across TX, so the level at init is not the level rendered)."""
        nonlocal person_level_calibration
        if args.person_level_calibration == "energy":
            with torch.no_grad():
                base_log_area = material.log_area_scale.detach().clone()
                calib_frames = training_frames[::2]
                B_maps, P_maps, masks_c, t_energy = [], [], [], []
                for frame in calib_frames:
                    for radar_index, radar in enumerate(RADARS):
                        components, rates = frame_components(frame, radar)
                        zero_bg = torch.zeros_like(current_background(radar, frame))
                        person0 = (gain_stack()[radar_index] * slow_time_assemble(zero_bg, components, rates, loop_chunk_size=args.loop_chunk_size))[0]   # loop 0 (tx, rx, K)
                        scene = gain_stack()[radar_index] * current_background(radar, frame)
                        if coupling_stack is not None:
                            scene = scene + coupling_stack[radar_index]
                        P = profiles_to_ra_spectrum_fused(person0.real[None].contiguous(), person0.imag[None].contiguous(), keep_index)[0]
                        B = profiles_to_ra_spectrum_fused(scene.real[None].contiguous(), scene.imag[None].contiguous(), keep_index)[0]
                        view = cache[frame]["views"][radar]; m = view["ra_mask"].index_select(0, keep_index)
                        B_maps.append(B); P_maps.append(P); masks_c.append(m)
                        t_energy.append(float((view["ra_target"].index_select(0, keep_index).square() * m).sum()))
                t_energy = torch.tensor(t_energy, device=device); valid = t_energy > 0
                def ratio(k: float) -> float:
                    e = torch.stack([((B + k * P).abs().square() * m).sum() for B, P, m in zip(B_maps, P_maps, masks_c)])
                    return float((e / t_energy.clamp_min(1e-30))[valid].median())
                ratio0 = ratio(1.0); lo, hi = float(np.log(1 / 64)), float(np.log(64))
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    if ratio(float(np.exp(mid))) < 1.0: lo = mid
                    else: hi = mid
                k = float(np.exp(0.5 * (lo + hi))); material.log_area_scale.copy_(base_log_area + 2.0 * float(np.log(k)))
                person_level_calibration = {"mode": "energy", "k_amplitude": k, "energy_ratio_before": ratio0, "energy_ratio_after": ratio(k),
                                            "area_scale": float(material.log_area_scale.detach().exp()), "frames": int(valid.sum())}
                print(f"person level calibration (energy, this stage's render): person mask loop-0 energy render/measured {ratio0:.3f} -> x{k:.3f} in "
                      f"amplitude -> {ratio(k):.3f} over {int(valid.sum())} frames; area scale {float(base_log_area.exp()):.4g} -> "
                      f"{person_level_calibration['area_scale']:.4g}", flush=True)
                del B_maps, P_maps
        if args.person_level_calibration == "peak-doppler" and args.person_level_calibration_view == "downstream" and downstream_view_fn is not None:
            # THE CONSUMER'S OWN VIEW: the recording (cached at cache time) and the render -- the scene's loop and the person's
            # 64 loops, per channel, through the downstream adapter's transform -- Doppler- and elevation-summed |RA| on the
            # adapter's grid; the person's window is the mesh pelvis's cell +-0.5 m in range, +-8 deg in azimuth (the
            # convention contrast_node.py verified against the recording); k by bisection on the median peak ratio.
            with torch.no_grad():
                view_fn, r_axis, s_axis = downstream_view_fn
                base_log_area = material.log_area_scale.detach().clone()
                calib_frames = training_frames[::4]
                R_g, S_g = np.meshgrid(r_axis, s_axis, indexing="ij")
                B_ch, P_ch, wins, t_peaks = [], [], [], []
                for frame in calib_frames:
                    for radar_index, radar in enumerate(RADARS):
                        components, rates = frame_components(frame, radar)
                        zero_bg = torch.zeros_like(current_background(radar, frame))
                        person_loops = gain_stack()[radar_index] * slow_time_assemble(zero_bg, components, rates, loop_chunk_size=args.loop_chunk_size)   # (loops, tx, rx, K)
                        scene = gain_stack()[radar_index] * current_background(radar, frame)
                        if coupling_stack is not None:
                            scene = scene + coupling_stack[radar_index]
                        view = cache[frame]["views"][radar]
                        pelvis = cache[frame]["joints"][0] if "joints" in cache[frame] else cache[frame]["points"].mean(dim=0)
                        px, py = float(pelvis[1]), float(pelvis[0])                     # label frame: x fwd = world Y, y lateral = -X
                        g_r, g_s = float(np.hypot(px, -py)), float(np.sin(-np.arctan2(-py, px)))
                        win = (np.abs(R_g - g_r) <= 0.5) & (np.abs(S_g - g_s) <= np.sin(np.radians(8)))
                        wins.append(torch.as_tensor(win, device=device))
                        measured = view["dsum_downstream"].to(device)
                        t_peaks.append(float(torch.where(wins[-1], measured, torch.zeros_like(measured)).max()))
                        B_ch.append(scene.to(torch.complex64)); P_ch.append(person_loops.to(torch.complex64))
                t_peak = torch.tensor(t_peaks, device=device); valid = t_peak > 0
                def ratio(k: float) -> float:
                    peaks = []
                    for B, P, w in zip(B_ch, P_ch, wins):
                        total = view_fn(B[None] + k * P)
                        peaks.append(torch.where(w, total, torch.zeros_like(total)).max())
                    return float((torch.stack(peaks) / t_peak.clamp_min(1e-30))[valid].median())
                ratio0 = ratio(1.0); lo, hi = float(np.log(1 / 64)), float(np.log(64))
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    if ratio(float(np.exp(mid))) < 1.0: lo = mid
                    else: hi = mid
                k = float(np.exp(0.5 * (lo + hi))); material.log_area_scale.copy_(base_log_area + 2.0 * float(np.log(k)))
                person_level_calibration = {"mode": "peak-doppler", "view": "downstream", "k_amplitude": k, "peak_ratio_before": ratio0, "peak_ratio_after": ratio(k),
                                            "area_scale": float(material.log_area_scale.detach().exp()), "frames": int(valid.sum())}
                print(f"person level calibration (peak-doppler, this stage's render, the DOWNSTREAM view -- the adapter's polar, the pelvis window): "
                      f"person peak render/measured {ratio0:.3f} -> x{k:.3f} in amplitude -> {ratio(k):.3f} over {int(valid.sum())} frames; area scale "
                      f"{float(base_log_area.exp()):.4g} -> {person_level_calibration['area_scale']:.4g}", flush=True)
                del B_ch, P_ch
        elif args.person_level_calibration == "peak-doppler":
            with torch.no_grad():
                base_log_area = material.log_area_scale.detach().clone()
                calib_frames = training_frames[::4]
                # per frame: the scene's RA (static: every loop the same) and the person's 64-loop RA maps, complex, so that
                # sum_d |FFT_d(B + k P)| over the mask is a function of k without a render; the same plain FFT as the target
                B_maps, P_maps, masks_c, t_peaks, statics = [], [], [], [], []
                cell_person = args.person_level_calibration_cell == "person"
                cell_side = args.person_level_calibration_cell == "side"
                target_person = args.person_level_calibration_target == "person"
                def side_of(m: torch.Tensor, own: torch.Tensor, half_width: int = 6) -> torch.Tensor:
                    """The mask restricted to +-half_width angle bins about the rendered person's own peak angle: the cached mask
                    holds the person's angle AND its mirror (person_range_angle_mask_torch include_mirrored_angle), and the
                    mirror side holds scene (RT-Pose 164: the corner at the mirror of the person at 7.5 m read as the peak)."""
                    j = int(torch.where(m, own, torch.zeros_like(own)).sum(dim=0).argmax())
                    cols = torch.zeros(m.shape[1], dtype=torch.bool, device=m.device); cols[max(j - half_width, 0):j + half_width + 1] = True
                    return m & cols[None, :]
                def neighbourhood(m: torch.Tensor, cell: tuple[int, int]) -> torch.Tensor:
                    """+-1 cell about `cell` (the person's own peak): the registration's tolerance in range and azimuth."""
                    n = torch.zeros_like(m); r, a = cell
                    n[max(r - 1, 0):r + 2, max(a - 1, 0):a + 2] = True
                    return n
                for frame in calib_frames:
                    for radar_index, radar in enumerate(RADARS):
                        components, rates = frame_components(frame, radar)
                        zero_bg = torch.zeros_like(current_background(radar, frame))
                        person_loops = gain_stack()[radar_index] * slow_time_assemble(zero_bg, components, rates, loop_chunk_size=args.loop_chunk_size)   # (loops, tx, rx, K)
                        scene = gain_stack()[radar_index] * current_background(radar, frame)
                        if coupling_stack is not None:
                            scene = scene + coupling_stack[radar_index]
                        view = cache[frame]["views"][radar]
                        m = view["ra_mask"].index_select(0, keep_index)
                        if elevation_on:
                            # the full array's (azimuth, elevation) view, the loop FFT taken first (linear operations commute):
                            # the scene is DC only (64 x its profile at bin 0), the person's bins are its loop FFT
                            P_fft = torch.fft.fft(person_loops, dim=0)                                                     # (loops, tx, rx, K)
                            S = profiles_to_rae_magnitude(P_fft[1:].real.contiguous(), P_fft[1:].imag.contiguous(), keep_index, window=False).sum(dim=(0, 3))   # (R, A): sum_{d != 0} sum_el, no window (the downstream's convention)
                            B_maps.append((64.0 * scene, P_fft[0])); P_maps.append(S)
                            if cell_person or cell_side:
                                # the person alone, all Doppler bins: its own peak cell in the mask, and the recording's peak within +-1 of it
                                own = profiles_to_rae_magnitude(P_fft[0].real[None].contiguous(), P_fft[0].imag[None].contiguous(), keep_index, window=False)[0].sum(dim=-1) + S
                                if cell_side:
                                    m = side_of(m, own)
                                else:
                                    cell = divmod(int(torch.where(m, own, torch.zeros_like(own)).argmax()), own.shape[1])
                                    m = neighbourhood(m, cell) & m
                            if target_person:
                                # the person's own contribution: the measured map less the measured static scene (its low quantile
                                # over frames), the render less its scene-only render, peak over the mask on both sides
                                static_m = dsum_static[radar]["dsum_full_rae"].to(device).index_select(0, keep_index)
                                measured = view["dsum_masked_rae"].to(device).index_select(0, keep_index)
                                t_peaks.append(float(torch.where(m, (measured - static_m).clamp_min(0), torch.zeros_like(measured)).max()))
                                statics.append(profiles_to_rae_magnitude(B_maps[-1][0].real[None].contiguous(), B_maps[-1][0].imag[None].contiguous(), keep_index, window=False)[0].sum(dim=-1))
                            elif cell_person or cell_side:
                                t_peaks.append(float(torch.where(m, view["dsum_masked_rae"].to(device).index_select(0, keep_index), torch.zeros_like(own)).max()))
                            else:
                                t_peaks.append(float(view["dsum_peak_rae"]))
                        else:
                            P = profiles_to_ra_spectrum_fused(person_loops.real.contiguous(), person_loops.imag.contiguous(), keep_index)   # (loops, R, A) complex
                            B = profiles_to_ra_spectrum_fused(scene.real[None].contiguous(), scene.imag[None].contiguous(), keep_index)[0]
                            B_maps.append(B); P_maps.append(P)
                            if cell_person or cell_side:
                                own = torch.fft.fft(P, dim=0).abs().sum(dim=0)
                                if cell_side:
                                    m = side_of(m, own)
                                else:
                                    cell = divmod(int(torch.where(m, own, torch.zeros_like(own)).argmax()), own.shape[1])
                                    m = neighbourhood(m, cell) & m
                            if target_person:
                                static_m = dsum_static[radar]["dsum_full"].to(device).index_select(0, keep_index)
                                measured = view["dsum_masked"].to(device).index_select(0, keep_index)
                                t_peaks.append(float(torch.where(m, (measured - static_m).clamp_min(0), torch.zeros_like(measured)).max()))
                                statics.append(64.0 * B.abs())
                            elif cell_person or cell_side:
                                t_peaks.append(float(torch.where(m, view["dsum_masked"].to(device).index_select(0, keep_index), torch.zeros_like(own)).max()))
                            else:
                                t_peaks.append(float(view["dsum_peak"]))
                        masks_c.append(m)
                t_peak = torch.tensor(t_peaks, device=device); valid = t_peak > 0
                def ratio(k: float) -> float:
                    peaks = []
                    for i, (B, P, m) in enumerate(zip(B_maps, P_maps, masks_c)):
                        if elevation_on:
                            dc = B[0] + k * B[1]
                            total = profiles_to_rae_magnitude(dc.real[None].contiguous(), dc.imag[None].contiguous(), keep_index, window=False)[0].sum(dim=-1) + k * P
                        else:
                            total = torch.fft.fft(B[None] + k * P, dim=0).abs().sum(dim=0)
                        if target_person:
                            total = (total - statics[i]).clamp_min(0)          # the person's own contribution, the scene-only render removed
                        peaks.append(torch.where(m, total, torch.zeros_like(total)).max())
                    return float((torch.stack(peaks) / t_peak.clamp_min(1e-30))[valid].median())
                ratio0 = ratio(1.0); lo, hi = float(np.log(1 / 64)), float(np.log(64))
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    if ratio(float(np.exp(mid))) < 1.0: lo = mid
                    else: hi = mid
                k = float(np.exp(0.5 * (lo + hi))); material.log_area_scale.copy_(base_log_area + 2.0 * float(np.log(k)))
                person_level_calibration = {"mode": "peak-doppler", "cell": args.person_level_calibration_cell, "target": args.person_level_calibration_target, "k_amplitude": k, "peak_ratio_before": ratio0, "peak_ratio_after": ratio(k),
                                            "area_scale": float(material.log_area_scale.detach().exp()), "frames": int(valid.sum())}
                print(f"person level calibration (peak-doppler, this stage's render, cell = {args.person_level_calibration_cell}, target = {args.person_level_calibration_target}): person mask peak of the Doppler-summed |RA| render/measured "
                      f"{ratio0:.3f} -> x{k:.3f} in amplitude -> {ratio(k):.3f} over {int(valid.sum())} frames; area scale {float(base_log_area.exp()):.4g} -> "
                      f"{person_level_calibration['area_scale']:.4g}", flush=True)
                del B_maps, P_maps
        if args.person_level_calibration == "peak":
            # docs/STATIC_SEQS_NEXT.md step 1, at the render that ships: the person's amplitude factor k that makes the person
            # mask's peak |RA| (loop 0, the retained range bins, the trainer's own beamformer) match the measured one, median
            # over the training frames, by bisection on log k -- through log_area_scale (amplitude ~ sqrt(area)), with
            # everything else at its initial state (the gain as just set). The person stage's calibration of the same
            # quantity is done on a render of every vertex; this one sees the visible vertices, the coupling and the
            # learned normals' start, i.e. exactly what the velocity stage renders. Frozen afterwards by --person-area-scale.
            with torch.no_grad():
                base_log_area = material.log_area_scale.detach().clone()
                def render_peaks() -> tuple[np.ndarray, np.ndarray]:
                    t_list, p_list = [], []
                    for start in range(0, len(training_frames), args.eval_frame_batch):
                        chunk = training_frames[start:start + args.eval_frame_batch]
                        for radar_index, radar in enumerate(RADARS):
                            units = [(frame, radar_index, None) for frame in chunk]
                            ra_profile, _, _ = unit_profiles(units)
                            pred = profiles_to_ra_magnitude_fused(ra_profile.real, ra_profile.imag, keep_index)
                            views = unit_views(units); mask = views["ra_mask_kept"]
                            p_list.append(torch.where(mask, pred, torch.zeros_like(pred)).flatten(1).max(dim=1).values)
                            t_list.append(torch.where(mask, views["ra_target"], torch.zeros_like(pred)).flatten(1).max(dim=1).values)
                    return torch.cat(t_list).cpu().numpy(), torch.cat(p_list).cpu().numpy()
                t_peak, p_peak = render_peaks(); valid = t_peak > 0
                ratio0 = float(np.median(p_peak[valid] / t_peak[valid]))
                lo, hi = float(np.log(1 / 64)), float(np.log(64))
                for _ in range(24):
                    mid = 0.5 * (lo + hi); material.log_area_scale.copy_(base_log_area + 2.0 * mid)
                    _, p_peak = render_peaks()
                    if float(np.median(p_peak[valid] / t_peak[valid])) < 1.0: lo = mid
                    else: hi = mid
                k = float(np.exp(0.5 * (lo + hi))); material.log_area_scale.copy_(base_log_area + 2.0 * float(np.log(k)))
                _, p_peak = render_peaks(); ratio1 = float(np.median(p_peak[valid] / t_peak[valid]))
                person_level_calibration = {"k_amplitude": k, "peak_ratio_before": ratio0, "peak_ratio_after": ratio1,
                                            "area_scale": float(material.log_area_scale.detach().exp()), "frames": int(valid.sum())}
                print(f"person level calibration (peak, this stage's render): person mask peak render/measured {ratio0:.3f} -> x{k:.3f} in amplitude "
                      f"-> {ratio1:.3f} over {int(valid.sum())} frames; area scale {float(base_log_area.exp()):.4g} -> {person_level_calibration['area_scale']:.4g}"
                      + ("" if args.person_area_scale != "learned" else "  (WARNING: the area scale is learned; the level will move)"), flush=True)


    if args.person_level_calibration != "off" and args.person_level_calibration_when == "init":
        calibrate_person_level()

    # ------------------------------------------------------------------
    # P3 = E4a — capture the training step
    #
    # The step is host-bound: ~1,277 device launches issued over ~23 ms of
    # wall clock against ~8.9 ms of GPU. Capturing forward and backward once
    # and replaying collapses that issue cost to a single graph launch.
    #
    # **Forward and backward are captured; the optimizer is not**, and that is
    # a correctness decision rather than an oversight. `torch.optim.Adam` has
    # two arithmetic paths. `capturable=False` computes the bias corrections as
    # Python floats and finishes with `addcdiv_(m, denom, value=-lr/bc1)`;
    # `capturable=True` has to keep the step counter on the device, so it folds
    # the step size into the denominator with an extra division —
    # `denom /= step_size`, then `addcdiv_(m, denom)`. The two agree in exact
    # arithmetic and disagree in the last bit, on every parameter, on every
    # step. Under capture the bias corrections cannot be Python floats, so
    # capturing the optimizer means adopting the capturable arithmetic and
    # changing the run. After P1 the optimizer is 5 tensors plus one fused-row
    # update — a few dozen launches — so it is left outside, eager and exact.
    #
    # What the capture requires, and what provides it:
    #   * static shapes             — P2;
    #   * static input buffers      — `stage_batch`, the six frames' positions;
    #     every other per-step quantity is derived from it inside the captured
    #     region;
    #   * no CPU<->GPU sync inside  — the static step has none (measured 0/step
    #     after P2): the two `torch.tensor([...], device=cuda)` uploads the
    #     dynamic path rebuilt per step are constants here;
    #   * stable gradient buffers   — `zero_grad(set_to_none=False)`, so the
    #     graph accumulates into the same addresses on every replay. Starting a
    #     step from a zeroed buffer instead of from `None` is exact: the
    #     accumulation adds the step's gradient to 0.0.
    #
    # The graph is captured once per padded size, and `pad_points` is held
    # monotone, so a re-capture happens only when an epoch draws a larger batch
    # than every epoch before it. Captures are counted and their cost is inside
    # the epoch's measured steps phase, not hidden beside it.
    # ------------------------------------------------------------------
    # Whether the step being run (or captured) carries the scene's gradient
    # (`--scene-update-every`); read at capture time by `static_backgrounds`.
    scene_grad: dict = {"on": True}
    global_step: dict = {"n": 0}
    graph_state: dict = {
        "graph": None, "outputs": None, "index_cache": None,
        "pad_points": 0, "captures": 0, "capture_s": 0.0,
    }

    def release_graph_pool() -> None:
        """Drop the captured step and its private CUDA-graph pool, once the epochs are done.

        The pool is 2.9-3.7 GiB on the cascade and is dead weight after the last epoch, but the
        post-training work allocates on top of it: `calibrate_person_level` asks `adapter.polar`
        for a 512 MiB buffer, and on a 24 GB card that is the difference between finishing and
        an OOM. The four-term fit peaks at 22.9 GiB reserved of 23.5
        (docs/RUNTIME_4TERM_RESULTS.md 1), so the margin is ~640 MiB; three baseline runs and
        then the A1 run died exactly here, AFTER completing all 400 epochs, which is the most
        expensive possible place to fail. Nothing replays the graph once the loop is over, so
        this only gives memory back -- no arithmetic changes.
        """
        if graph_state["graph"] is None and graph_state.get("graph_off") is None:
            return
        graph_state["graph"] = None
        graph_state["outputs"] = None
        graph_state["graph_off"] = None
        graph_state["outputs_off"] = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print("released the captured step's graph pool: "
              f"{torch.cuda.memory_reserved() / 2 ** 30:.2f} GiB reserved, "
              f"{torch.cuda.max_memory_reserved() / 2 ** 30:.2f} GiB was the peak", flush=True)

    def graph_gradient_holders() -> list[torch.Tensor]:
        holders = [parameter for group in optimizer.param_groups
                   for parameter in group["params"]]
        if normals_optimizer is not None:
            holders.append(model.raw_normals_flat)
        return holders

    def zero_graph_gradients() -> None:
        for parameter in graph_gradient_holders():
            if parameter.grad is not None:
                parameter.grad.zero_()

    def capture_static_step() -> None:
        """Warm up and capture one training step's forward and backward."""
        started = time.perf_counter()
        # Release the previous graph and its private memory pool *before*
        # warming up the next one. Each capture gets its own pool: sharing one
        # across graphs is only safe if their lifetimes are managed by hand,
        # and here they are not — measured, a shared pool ran three captures
        # and then corrupted an index buffer on the first epoch that replayed
        # without re-capturing.
        graph_state["graph"] = None
        graph_state["outputs"] = None
        graph_state["graph_off"] = None
        graph_state["outputs_off"] = None
        torch.cuda.synchronize()

        def one_iteration() -> tuple:
            totals, ra_terms, md_terms = static_step()
            batch_loss = totals.sum()
            descend(batch_loss)
            return totals, ra_terms, md_terms

        # Two eager iterations first, on the default stream: they create the
        # gradient buffers at stable addresses and build the cuFFT plans, the
        # PSF and index caches and the autograd graph the capture must find
        # ready. Three more on a side stream, as capture requires.
        for _ in range(2):
            one_iteration()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                zero_graph_gradients()
                one_iteration()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        zero_graph_gradients()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = one_iteration()
        torch.cuda.synchronize()
        # the off-step version: the scene rendered without a gradient
        graph_off, outputs_off = None, None
        if args.scene_update_every > 1:
            scene_grad["on"] = False
            with torch.cuda.stream(stream):
                for _ in range(2):
                    zero_graph_gradients()
                    one_iteration()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            zero_graph_gradients()
            graph_off = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph_off):
                outputs_off = one_iteration()
            torch.cuda.synchronize()
            scene_grad["on"] = True
        graph_state.update({
            "graph": graph, "outputs": outputs,
            "graph_off": graph_off, "outputs_off": outputs_off,
            # `render_factorized` caches its three `(M x n_tx x n_rx,)` channel
            # index tensors on the function, keyed by `(M, n_tx, n_rx, device)`,
            # and *replaces the whole dict* on a miss. The capture bakes in
            # pointers to the training-M entry; the evaluation pass renders a
            # different M, misses, and drops the last reference to it — after
            # which the allocator is free to hand that memory to something
            # else and the next replay reads garbage indices. Holding the dict
            # here keeps those tensors alive at the addresses the graph
            # recorded, for exactly as long as the graph that recorded them.
            # (Measured: without this the run survives every epoch that
            # re-captures and dies on the first one that does not — a
            # device-side "index out of bounds" from a replayed index kernel.)
            "index_cache": getattr(render_factorized, "_idx_cache", None),
            "pad_points": static_state["pad_points"],
            "captures": graph_state["captures"] + 1,
            "capture_s": graph_state["capture_s"] + time.perf_counter() - started,
        })
        print(f"captured training step: {static_state['pad_points']} padded "
              f"points per radar, {plane_count} planes, "
              f"capture {graph_state['captures']}, "
              f"{graph_state['capture_s']:.2f} s cumulative", flush=True)

    def render_losses(
        frame: int, radar: str, velocity_source_frame: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render one frame for one radar and return the two-term loss."""
        components, rates = frame_components(frame, radar, velocity_source_frame)
        # THE FRAME, not a frameless background: this path asked for the
        # unmasked scene under a loaded mask, so `--step-path legacy` -- the
        # pinned arbiter -- compared a mask-on run against an unmasked
        # background and called them equivalent. The guard in `visible_gains`
        # now makes that combination raise rather than mislead.
        profiles = gain_stack()[RADARS.index(radar)] * slow_time_assemble(
            current_background(radar, frame), components, rates,
            loop_chunk_size=args.loop_chunk_size,
        )
        return losses_from_profiles(frame, radar, profiles)

    EVAL_METRICS = ("total", "ra_l2", "md_l2")

    def evaluate(subset: list[int]) -> dict:
        # The evaluation is exactly the training objective, re-measured under
        # no_grad at the epoch's parameters: the two-term loss (loop-0 RA L2
        # + 63-plane RA L2 + MD L2) per frame and radar, averaged. The
        # negative velocity sources and their md_temporal_delta were removed
        # 2026-08-25 at the owner's request — the evaluation renders exactly
        # one velocity per frame, the trained one, same as a training step.
        #
        # S4 (runtime): the per-frame scalars stay on the device, are reduced
        # there, and cross the bus once for the whole pass.
        rows = {radar: [] for radar in RADARS}
        with torch.no_grad():
            if args.step_path == "legacy":
                for frame in subset:
                    for radar in RADARS:
                        total, ra_l2, md_l2, _ = render_losses(frame, radar)
                        rows[radar].append(torch.stack((total, ra_l2, md_l2)))
            else:
                # E5, extended over the whole pass: no parameter changes
                # inside `no_grad`, so the scene's BSDF re-render and both
                # K1 assemblies are computed once per pass instead of once
                # per chunk -- ten identical recomputations before.
                # Frame-independent only when nothing is shadowed; with a
                # mask `unit_scaled_profiles` assembles per unit and this
                # would be both unused and, under the guard, a refusal.
                pass_backgrounds = (
                    None if scene_visibility is not None
                    else current_backgrounds())
                for start in range(0, len(subset), args.eval_frame_batch):
                    chunk = subset[start:start + args.eval_frame_batch]
                    units = [
                        (frame, radar_index, None)
                        for radar_index in range(len(RADARS))
                        for frame in chunk
                    ]
                    geometry = unit_geometry(units)
                    # A per-chunk splat-geometry cache: the chunk lists are
                    # fixed for the whole run, so their position-only render
                    # constants are filled on the first pass and reused.
                    # Bounded to the recurring chunk lists (the per-epoch
                    # evaluation of all 590 training frames is 40 chunks);
                    # one-off passes past the bound miss and run uncached.
                    # cache_psf=False keeps each chunk ~120 MB instead of
                    # ~200: the PSF pair is re-lerped per pass from the
                    # cached fractional bins through the same compiled graph
                    # the fill ran (bitwise), so 40 caches cost ~4.7 GB.
                    chunk_key = tuple(chunk)
                    chunk_cache = eval_geometry_caches.get(chunk_key)
                    if (chunk_cache is None
                            and len(eval_geometry_caches) < _EVAL_CACHE_BOUND):
                        chunk_cache = eval_geometry_caches.setdefault(
                            chunk_key, {})
                    components, unit_of_path = unit_render(
                        geometry, chunk_cache, cache_psf=False)
                    ra_profile, planes, dc_profile = unit_scaled_profiles(
                        units, components, unit_rates(geometry, units),
                        unit_of_path, pass_backgrounds,
                    )
                    totals, ra_terms, md_terms, _ = losses_from_profiles_units(
                        units, ra_profile, planes, dc_profile,
                    )
                    block = torch.stack(
                        (totals, ra_terms, md_terms), dim=-1,
                    ).reshape(len(RADARS), len(chunk), 3)
                    for radar_index, radar in enumerate(RADARS):
                        rows[radar].extend(block[radar_index])
            # One transfer for the whole pass: (radars, frames, 3). The mean
            # stays in numpy float64 over the same float32 values in the same
            # order as the per-frame float() path produced.
            block = torch.stack([
                torch.stack(rows[radar]) for radar in RADARS
            ]).cpu().numpy().astype(np.float64)
        columns = {
            "total": block[..., 0], "ra_l2": block[..., 1],
            "md_l2": block[..., 2],
        }
        return {
            radar: {
                metric: float(columns[metric][radar_index].mean())
                for metric in EVAL_METRICS
            }
            for radar_index, radar in enumerate(RADARS)
        }

    def snapshot() -> dict:
        return {
            "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "material": {k: v.detach().clone() for k, v in material.state_dict().items()},
            "background_raw_materials": background.raw_materials.detach().clone(),
            "background_rotations": background.rotations.detach().clone(),
            "absolute_log_gain": absolute_log_gain.detach().clone(),
            "joint_log_gain": joint_log_gain.detach().clone(),
        }

    # Calibrate the training-domain noise floor from our own renders: the
    # per-Doppler-bin (noisy - clean) rendered spectrogram power is the
    # pipeline's full injected-noise stack in the deliverable domain; a
    # render-derived constant kappa converts it into this trainer's marginal
    # domain (ratio of the same clean person power expressed both ways).
    # No measured data is involved anywhere.
    def two_set_evaluation() -> dict:
        """Both reported frame sets; one pass when they are the same set."""
        validation = evaluate(validation_frames)
        return {
            "validation_frames": validation,
            "training_frames": (validation
                                if validation_frames == training_frames
                                else evaluate(training_frames)),
        }

    def gradient_probe() -> None:
        """MM4DPS_GRAD_PROBE=<json>: the objective's terms one at a time at the current parameters, diagnostic only.

        For each probe frame (MM4DPS_GRAD_PROBE_FRAMES="f,f,..." or 12 evenly spaced training frames) and each term --
        loop-0 |RA|, the 63-plane |RA| (with the learned floor), the MD marginal, the two elevation terms, and the
        loop-0 / plane terms split into the person's RA-mask cells and the rest -- the value, its gradient norm per
        parameter group, and the directional derivatives d term / d log(person amplitude) (exact: the person's
        amplitude is proportional to sqrt(area_scale)), d term / d log(total amplitude) (the rendered signal scaled,
        the floors and the coupling fixed) and their difference, the scene's. MM4DPS_GRAD_PROBE_VF=<velocity_field.pt>
        first loads a trained field's joint gains, absolute gain, normals and velocity so the probe runs at the
        converged state (with --person-checkpoint pointing at that run's best_shared_human_material.pt). Nothing
        here is read by training: it runs before the first step and the trainer returns when
        MM4DPS_GRAD_PROBE_ONLY is set (the default).
        """
        vf_path = os.environ.get("MM4DPS_GRAD_PROBE_VF")
        if vf_path:
            load_trained_state(vf_path)

        frames_env = os.environ.get("MM4DPS_GRAD_PROBE_FRAMES")
        probe_frames = ([int(f) for f in frames_env.split(",")] if frames_env
                        else [training_frames[i] for i in np.linspace(0, len(training_frames) - 1, 12).astype(int)])
        material.log_area_scale.requires_grad_(True)
        groups = {"velocity": velocity_parameters, "person_material_raw": [material.raw],
                  "person_log_area_scale": [material.log_area_scale], "person_radar_log_gain_delta": [material.radar_log_gain_delta],
                  "joint_log_gain": [joint_log_gain], "absolute_log_gain": [absolute_log_gain],
                  "person_normals": [model.raw_normals_flat], "background_materials": [background.raw_materials],
                  "background_rotations": [background.rotations],
                  "noise_model": [] if noise_model is None else list(noise_model.parameters())}
        for name in ("background_materials", "background_rotations"):
            for parameter in groups[name]:
                parameter.requires_grad_(True)
        params = [parameter for name in groups for parameter in groups[name] if parameter.requires_grad]
        offsets = {}
        cursor = 0
        for name in groups:
            offsets[name] = [i for i, parameter in enumerate(params) if any(parameter is q for q in groups[name])]

        def terms_of(units, ra_profile, planes, dc_profile=None):
            """{term: (units,) tensor} -- the objective's terms, the reference expressions, separately."""
            views = unit_views(units); rows = view_rows_for(units)
            unit_radars = torch.tensor([radar_index for _, radar_index, _ in units], device=device)
            plane_bins_rows = None if view_plane_bins is None else view_plane_bins.index_select(0, rows)
            out = {}
            ra_pred = profiles_to_ra_magnitude_batched(ra_profile.real, ra_profile.imag).index_select(1, keep_index)
            ra_target = views["ra_target"]; den = flat_views["ra_mean_square"].index_select(0, rows)
            sq = (ra_pred - ra_target).square(); n = float(sq[0].numel()); mask = views["ra_mask_kept"].to(sq.dtype)
            out["loop0_ra"] = sq.flatten(1).mean(-1) / den
            out["loop0_ra_person_cells"] = (sq * mask).flatten(1).sum(-1) / n / den
            out["loop0_ra_scene_cells"] = (sq * (1 - mask)).flatten(1).sum(-1) / n / den
            out["_loop0_target_energy_share_person"] = (ra_target.square() * mask).flatten(1).sum(-1) / ra_target.square().flatten(1).sum(-1)
            out["_loop0_prediction_energy_share_person"] = (ra_pred.square() * mask).flatten(1).sum(-1) / ra_pred.square().flatten(1).sum(-1)
            out["_loop0_person_cells_prediction_over_target"] = (ra_pred * mask).flatten(1).sum(-1) / (ra_target * mask).flatten(1).sum(-1).clamp_min(1e-30)
            out["_loop0_scene_cells_prediction_over_target"] = (ra_pred * (1 - mask)).flatten(1).sum(-1) / (ra_target * (1 - mask)).flatten(1).sum(-1).clamp_min(1e-30)
            plane_pred = profiles_to_ra_magnitude_batched(planes.real, planes.imag).index_select(2, keep_index)
            floor = plane_noise_power(unit_radars, plane_bins_rows)
            if floor is not None:
                plane_pred = (plane_pred.square() + floor).clamp_min(1e-30).sqrt()            # floor is (units, planes, range, 1)
            plane_target = plane_targets_for(units); pden = flat_views["ra_doppler_mean_square"].index_select(0, rows)
            psq = (plane_pred - plane_target).square(); pn = float(psq[0].numel()); pmask = mask[:, None]
            out["plane_ra"] = psq.flatten(1).mean(-1) / pden
            out["plane_ra_person_cells"] = (psq * pmask).flatten(1).sum(-1) / pn / pden
            out["plane_ra_scene_cells"] = (psq * (1 - pmask)).flatten(1).sum(-1) / pn / pden
            out["_plane_target_energy_share_person"] = (plane_target.square() * pmask).flatten(1).sum(-1) / plane_target.square().flatten(1).sum(-1)
            out["_plane_person_cells_prediction_over_target"] = (plane_pred * pmask).flatten(1).sum(-1) / (plane_target * pmask).flatten(1).sum(-1).clamp_min(1e-30)
            md_pred = md_prediction_units(units, planes); md_target = views["md_target"]
            out["md"] = args.microdoppler_weight * (md_pred - md_target).square().mean(-1) / flat_views["md_mean_square"].index_select(0, rows)
            out["_md_prediction_over_target"] = md_pred.sum(-1) / md_target.sum(-1).clamp_min(1e-30)
            if elevation_on:
                rae_pred = profiles_to_rae_magnitude(ra_profile.real, ra_profile.imag, keep_index)
                rae_target = flat_views["rae_target"].index_select(0, rows)
                out["elevation_loop0_rae"] = args.elevation_weight * (rae_pred - rae_target).square().flatten(1).mean(-1) / flat_views["rae_mean_square"].index_select(0, rows)
                pl = planes if rae_plane_index is None else planes.index_select(1, rae_plane_index)
                rae_plane_pred = profiles_to_rae_magnitude(pl.real, pl.imag, keep_index)
                fp = floor
                if fp is not None and rae_plane_index is not None:
                    fp = fp.index_select(1, rae_plane_index)
                if fp is not None:
                    fp = fp if args.rae_floor_gradient == "on" else fp.detach()
                    rae_plane_pred = (rae_plane_pred.square() + fp[..., None] * _rae_floor_factor).clamp_min(1e-30).sqrt()
                rae_plane_target = (flat_views["rae_doppler_target"].index_select(0, rows).to(torch.float32)
                                    * flat_views["rae_doppler_rms"].index_select(0, rows).reshape(-1, 1, 1, 1, 1))
                out["elevation_plane_rae"] = args.elevation_weight * (rae_plane_pred - rae_plane_target).square().flatten(1).mean(-1) / flat_views["rae_doppler_mean_square"].index_select(0, rows)
                # ---------------------------------------------------------------- C1b, diagnostic only
                # The proposal: replace the person window and micro-Doppler with the SAME two RAE
                # terms restricted to the person's box -- term 3 loop-0 (clutter kept, as term 1),
                # term 4 per plane (DC excluded and clutter removed, as term 2). Evaluated here at
                # the baseline's converged state, which is the only fair place to compare a
                # candidate term with the terms it would replace: a term sitting at its own optimum
                # has a small gradient and one away from any optimum has a large one.
                box = views["pw_box_mask"].index_select(1, keep_index).to(torch.float32)                              # (units, range kept, angle)
                _b3 = box[..., None]                                                                                  # over elevation
                _den0 = ((rae_target * _b3).square().flatten(1).sum(-1) / _b3.flatten(1).sum(-1).clamp_min(1.0)).clamp_min(1e-30)
                out["c1b_win0_rae"] = (((rae_pred - rae_target) * _b3).square().flatten(1).sum(-1)
                                       / _b3.flatten(1).sum(-1).clamp_min(1.0)) / _den0
                _b4 = box[:, None, ..., None]                                                                         # over planes and elevation
                _denp = ((rae_plane_target * _b4).square().flatten(1).sum(-1) / _b4.expand_as(rae_plane_target).flatten(1).sum(-1).clamp_min(1.0)).clamp_min(1e-30)
                out["c1b_winpl_rae"] = (((rae_plane_pred - rae_plane_target) * _b4).square().flatten(1).sum(-1)
                                        / _b4.expand_as(rae_plane_target).flatten(1).sum(-1).clamp_min(1.0)) / _denp
            # The control for micro-Doppler's spatial sum: the SAME cells micro-Doppler uses, compared
            # cell by cell instead of after summing power over range and angle. If this already
            # over-drives, the sum is the cause and the window is innocent.
            _mdm = views["md_mask"].index_select(1, keep_index).to(torch.float32)[:, None]
            _mp = plane_pred.index_select(1, motion_index) if plane_pred.shape[1] != len(motion_bins) else plane_pred
            _mt = plane_target.index_select(1, motion_index) if plane_target.shape[1] != len(motion_bins) else plane_target
            _dm = ((_mt * _mdm).square().flatten(1).sum(-1) / _mdm.expand_as(_mt).flatten(1).sum(-1).clamp_min(1.0)).clamp_min(1e-30)
            out["md_cellwise"] = (((_mp - _mt) * _mdm).square().flatten(1).sum(-1)
                                  / _mdm.expand_as(_mt).flatten(1).sum(-1).clamp_min(1.0)) / _dm
            if static_on and dc_profile is not None:
                # P1's pieces: the person cells and the scene cells, RA view and RAE view
                m = views["ra_mask_kept"].to(torch.float32)
                dc_pred = profiles_to_ra_magnitude_batched(dc_profile.real, dc_profile.imag).index_select(1, keep_index)
                dsq = (dc_pred - flat_views["dc_target"].index_select(0, rows)).square()
                out["static_dc_ra_person"] = args.static_weight * (dsq * m).flatten(1).sum(-1) / flat_views["dc_person_energy"].index_select(0, rows)
                out["static_dc_ra_scene"] = args.static_weight * (dsq * (1 - m)).flatten(1).sum(-1) / flat_views["dc_scene_energy"].index_select(0, rows)
                out["_dc_person_cells_prediction_over_target"] = (dc_pred * m).flatten(1).sum(-1) / (flat_views["dc_target"].index_select(0, rows) * m).flatten(1).sum(-1).clamp_min(1e-30)
                out["_dc_target_energy_share_person"] = flat_views["dc_person_energy"].index_select(0, rows) / (flat_views["dc_person_energy"] + flat_views["dc_scene_energy"]).index_select(0, rows)
                if elevation_on:
                    dc_rae = profiles_to_rae_magnitude(dc_profile.real, dc_profile.imag, keep_index)
                    dsq = (dc_rae - flat_views["dc_rae_target"].index_select(0, rows)).square()
                    out["static_dc_rae_person"] = args.static_weight * (dsq * m[..., None]).flatten(1).sum(-1) / flat_views["dc_rae_person_energy"].index_select(0, rows)
                    out["static_dc_rae_scene"] = args.static_weight * (dsq * (1 - m)[..., None]).flatten(1).sum(-1) / flat_views["dc_rae_scene_energy"].index_select(0, rows)
                out["static_dc"] = out["static_dc_ra_person"] + out["static_dc_ra_scene"] + out.get("static_dc_rae_person", 0) + out.get("static_dc_rae_scene", 0)
            if pw_on:
                out["person_window"] = person_window_terms(rows, ra_profile, planes)
            out["total"] = (out["loop0_ra"] + out["plane_ra"] + out["md"] + out.get("elevation_loop0_rae", 0) + out.get("elevation_plane_rae", 0)
                            + out.get("static_dc", 0) + out.get("person_window", 0))
            return out

        records = []
        if os.environ.get("MM4DPS_GRAD_PROBE_DUMP"):
            # the trainer's own rendered loop-0 profile of the first probe frame, for comparison with the product
            with torch.no_grad():
                units = [(probe_frames[0], 0, None)]; ra_profile, _, _ = unit_profiles(units)
                np.save(os.environ["MM4DPS_GRAD_PROBE_DUMP"], ra_profile[0].detach().cpu().numpy())
                # and the trainer's own loop-0 |RA| view of every probe frame against its target (the windowed beamformer
                # over the retained rows), with the person mask: where the static scene is off in the objective's own eyes
                maps = []
                for frame in probe_frames:
                    units = [(frame, radar_index, None) for radar_index in range(len(RADARS))]
                    ra_profile, _, _ = unit_profiles(units); v = unit_views(units)
                    pred = profiles_to_ra_magnitude_fused(ra_profile.real, ra_profile.imag, keep_index)
                    entry = {"frame": frame, "pred": pred.detach().cpu().numpy(), "target": v["ra_target"].detach().cpu().numpy(),
                             "mask": v["ra_mask_kept"].detach().cpu().numpy()}
                    if elevation_on:
                        # the 2D array's loop-0 |RAE| (range, azimuth, elevation) and its target, the elevation term's own view
                        entry["rae_pred"] = profiles_to_rae_magnitude(ra_profile.real, ra_profile.imag, keep_index).detach().cpu().numpy()
                        entry["rae_target"] = flat_views["rae_target"].index_select(0, view_rows_for(units)).float().detach().cpu().numpy()
                    maps.append(entry)
                np.save(os.environ["MM4DPS_GRAD_PROBE_DUMP"].replace(".npy", "_maps.npy"), np.array(maps, dtype=object), allow_pickle=True)
        import torch._functorch.config as _functorch_config
        _functorch_config.donated_buffer = False        # the compiled kernels forbid retain_graph otherwise; probe only

        def forward(units):
            """The terms at the current parameters, the rendered signal under a unit scale `s` (the floors and the
            antenna coupling fixed): one fresh graph per backward, as the compiled kernels require."""
            ra_profile, planes, dc_profile = unit_profiles(units)
            s = torch.ones((), device=device, requires_grad=True)
            radar_index = torch.tensor([radar_index for _, radar_index, _ in units], device=device)
            if coupling_stack is not None:
                coupling = coupling_stack[radar_index]
                ra_s = s * (ra_profile - coupling) + coupling
                dc_s = None if dc_profile is None else s * (dc_profile - coupling) + coupling
            else:
                ra_s = s * ra_profile
                dc_s = None if dc_profile is None else s * dc_profile
            return s, terms_of(units, ra_s, s * planes, dc_s)

        for frame in probe_frames:
            units = [(frame, radar_index, None) for radar_index in range(len(RADARS))]
            for parameter in params:
                parameter.grad = None
            s, terms = forward(units)
            names = list(terms)
            for name in names:
                if name.startswith("_"):
                    records.append({"frame": frame, "term": name, "value": float(terms[name].mean())}); continue
                s, terms = forward(units); value = terms[name]
                grads = torch.autograd.grad(value.sum(), [s] + params, allow_unused=True)
                d_total = float(grads[0]) if grads[0] is not None else 0.0
                per_group = {}
                for group in groups:
                    total_sq = sum(float(grads[1 + i].double().square().sum()) for i in offsets[group] if grads[1 + i] is not None)
                    per_group[group] = total_sq ** 0.5
                g_area = next((grads[1 + i] for i in offsets["person_log_area_scale"]), None)
                d_person = 2.0 * float(g_area) if g_area is not None else 0.0
                records.append({"frame": frame, "term": name, "value": float(value.sum()), "d_log_total_amplitude": d_total,
                                "d_log_person_amplitude": d_person, "d_log_scene_amplitude": d_total - d_person, "grad_norm": per_group})
            del terms
        for parameter in params:
            parameter.grad = None
        out_path = Path(os.environ["MM4DPS_GRAD_PROBE"]); out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {}
        for name in sorted({r["term"] for r in records}):
            rows = [r for r in records if r["term"] == name]
            entry = {"value_mean": float(np.mean([r["value"] for r in rows]))}
            if "d_log_person_amplitude" in rows[0]:
                for key in ("d_log_total_amplitude", "d_log_person_amplitude", "d_log_scene_amplitude"):
                    values = np.array([r[key] for r in rows]); entry[key + "_mean"] = float(values.mean()); entry[key + "_sign_consistency"] = float(np.mean(np.sign(values) == np.sign(values.mean())))
                entry["grad_norm_mean"] = {g: float(np.mean([r["grad_norm"][g] for r in rows])) for g in groups}
            summary[name] = entry
        json.dump({"sequence": args.sequence, "state": "trained" if vf_path else "initial", "frames": probe_frames,
                   "person_area_scale": float(material.log_area_scale.detach().exp()), "absolute_gain": gain_values(),
                   "joint_gains": [float(g) for g in joint_gains().detach().cpu()], "records": records, "summary": summary},
                  open(out_path, "w"), indent=1)
        print(f"gradient probe -> {out_path}", flush=True)
        for name, entry in summary.items():
            if "d_log_person_amplitude_mean" in entry:
                print(f"  {name:<28} value {entry['value_mean']:9.4f}  d/dlog person {entry['d_log_person_amplitude_mean']:+10.4f} (sign consistency {entry['d_log_person_amplitude_sign_consistency']:.2f})"
                      f"  d/dlog scene {entry['d_log_scene_amplitude_mean']:+10.4f}  |g| person_area {entry['grad_norm_mean']['person_log_area_scale']:.3g} joint_gains {entry['grad_norm_mean']['joint_log_gain']:.3g} "
                      f"material {entry['grad_norm_mean']['person_material_raw']:.3g} normals {entry['grad_norm_mean']['person_normals']:.3g} scene_mat {entry['grad_norm_mean']['background_materials']:.3g} scene_rot {entry['grad_norm_mean']['background_rotations']:.3g} velocity {entry['grad_norm_mean']['velocity']:.3g} abs_gain {entry['grad_norm_mean']['absolute_log_gain']:.3g}", flush=True)
            else:
                print(f"  {name:<28} {entry['value_mean']:.4f}", flush=True)

    if os.environ.get("MM4DPS_GRAD_PROBE"):
        gradient_probe()
        if os.environ.get("MM4DPS_GRAD_PROBE_ONLY", "1") == "1":
            return {"status": "gradient-probe-only", "probe": os.environ["MM4DPS_GRAD_PROBE"]}

    initial_validation = two_set_evaluation()
    best = snapshot()
    best_loss = float(np.mean([
        initial_validation["validation_frames"][radar]["total"] for radar in RADARS
    ]))
    history = []
    evaluations = 0
    phase = phase_of(0)
    for epoch in range(args.epochs):
        if phase_of(epoch) != phase:
            phase = phase_of(epoch)
            optimizer = build_optimizer(phase)
            normals_optimizer = build_normals_optimizer(phase)
            base_lrs = base_learning_rates(optimizer, normals_optimizer)
        if args.lr_schedule != "none":
            apply_learning_rate(
                optimizer, normals_optimizer, base_lrs, learning_rate_scale(epoch))
        if plane_sample_index is not None:
            plane_sample_index.copy_(torch.randperm(
                len(ra_doppler_bins), device=device,
                generator=torch.Generator(device=device).manual_seed(
                    args.seed + epoch))[:args.plane_sample_count])
        order = training_frames.copy()
        random.Random(args.seed + epoch).shuffle(order)
        # S2/S3 (runtime): the three per-frame-per-radar float() calls that
        # filled these records were 36 CPU<->GPU round trips per step, each
        # one draining the pipeline in the middle of the batch's forward
        # pass. The detached scalars stay on the device in `pending_scalars`
        # and cross the bus once per epoch; `pending_records` holds the
        # dicts they fill in, in step order, so `history` is unchanged.
        torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        # P2: the epoch's shuffle is host knowledge, so the padded half length
        # is chosen here — the largest batch this epoch actually draws, not a
        # bound valid for any shuffle: ~15% over the mean rather than ~37%.
        # Inside the timed region, so the plan and any re-capture are counted
        # against the epoch that pays for them.
        if static_path:
            all_batches = [
                [frame_position[frame] for frame in order[start:start + args.frame_batch]]
                for start in range(0, len(order), args.frame_batch)
            ]
            if _STATIC_SHORT_STEP:
                # The short last batch is padded to frame_batch with repeats
                # of the smallest frame in the flat cache — masked to zero
                # in the loss, and small so the padded batch's point count
                # stays under the epoch's largest full batch and never
                # drives `pad_points` (and with it every step's padded
                # work) upward.
                real_counts = [len(batch) for batch in all_batches]
                smallest_position = int(np.argmin(frame_counts))
                planned_batches = [
                    batch + [smallest_position] * (args.frame_batch - len(batch))
                    for batch in all_batches
                ]
                mask_plan = np.zeros(
                    (len(planned_batches), len(RADARS), args.frame_batch),
                    dtype=np.float32)
                for row, count in enumerate(real_counts):
                    mask_plan[row, :, :count] = 1.0
                epoch_step_mask = torch.as_tensor(
                    mask_plan.reshape(len(planned_batches), -1), device=device)
            else:
                planned_batches = [batch for batch in all_batches
                                   if len(batch) == args.frame_batch]
                epoch_step_mask = None
            epoch_batch_index = torch.as_tensor(
                planned_batches, dtype=torch.long, device=device)
            # Monotone: an epoch whose largest batch is smaller keeps the
            # padding it already has rather than forcing a re-capture for a
            # fraction of a percent of paths.
            pad_points = max(
                max(sum(frame_counts[position] for position in batch)
                    for batch in planned_batches),
                static_state["pad_points"],
            )
            if pad_points != static_state["pad_points"]:
                static_resize(pad_points)
                if row_geometry:
                    # built before the capture: the shape-only constants
                    # must not be allocated inside the captured graph
                    row_geometry_shapes(int(pad_points))
            if static_state["noise_floor"] is None:
                # The no-floor path's zero. With the noise model on,
                # `static_losses` recomputes the floor from the parameters
                # inside the step (captured with it) and never reads this.
                static_state["noise_floor"] = torch.zeros(
                    len(static_radar_index), len(motion_bins), device=device)
            if (capture_step_path
                    and static_state["pad_points"] != graph_state["pad_points"]):
                # Captured before the step loop, so the warm-up iterations
                # belong to no step: the loop's first `zero_grad` clears
                # whatever they left behind.
                stage_batch.copy_(epoch_batch_index[0])
                capture_static_step()
        epoch_totals = []
        pending_records: list[dict] = []
        pending_scalars: list[torch.Tensor] = []
        # Frames are independent, so a batch of K frames fuses into one
        # slow-time scatter per radar (the dominant kernel); each frame's
        # per-frame velocity parameters still receive exactly one gradient
        # contribution per epoch — only shared parameters see K-fold fewer,
        # K-fold larger Adam steps.
        for step_index, start in enumerate(range(0, len(order), args.frame_batch)):
            if args.profile_steps and epoch == 0 and step_index == 2:
                # MM4DPS_PROFILE_STACK=N groups the table by the N innermost Python frames, which
                # names the LINE that launched each kernel. Shapes alone leave the attribution to
                # inference, and inference is what over-estimated A1 by 60%
                # (docs/RUNTIME_STEP_20MS_PLAN.md 9). Costs nothing but profiler overhead, and the
                # run exits after the captured steps anyway.
                _stack = int(os.environ.get("MM4DPS_PROFILE_STACK", "0"))
                profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=os.environ.get("MM4DPS_PROFILE_SHAPES") == "1",
                    with_stack=_stack > 0)
                profiler.__enter__()
            if args.profile_steps and epoch == 0 and step_index == 2 + args.profile_steps:
                profiler.__exit__(None, None, None)
                _stack = int(os.environ.get("MM4DPS_PROFILE_STACK", "0"))
                _shapes = os.environ.get("MM4DPS_PROFILE_SHAPES") == "1"
                # The name column is wide unconditionally: the 55-character default truncates
                # every templated kernel to an indistinguishable prefix, which is how a profile of
                # the fused route was once attributed to the reference route's kernels.
                print(profiler.key_averages(
                    group_by_input_shape=_shapes, group_by_stack_n=_stack).table(
                    sort_by="self_cuda_time_total", row_limit=160 if _shapes else 110,
                    max_name_column_width=200), flush=True)
                if _stack:
                    profiler.export_chrome_trace(str(args.output_dir / "trace.json.gz"))
                    print(f"trace: {args.output_dir / 'trace.json.gz'}", flush=True)
                return {"status": "profiled"}
            batch_frames = order[start:start + args.frame_batch]
            # `--scene-update-every k`: the scene's parameters move on every
            # k-th step of the run (a global count, so an epoch's length does
            # not alias the schedule); the other steps render it frozen.
            scene_step = (args.scene_update_every <= 1
                          or global_step["n"] % args.scene_update_every == 0)
            global_step["n"] += 1
            # A captured step accumulates into gradient buffers whose addresses
            # the graph baked in, so those are zeroed rather than dropped.
            keep_gradient_buffers = graph_state["graph"] is not None
            optimizer.zero_grad(set_to_none=not keep_gradient_buffers)
            if normals_optimizer is not None:
                normals_optimizer.zero_grad(set_to_none=not keep_gradient_buffers)
            records = {
                frame: {"epoch": epoch + 1, "phase": phase, "frame": frame}
                for frame in batch_frames
            }
            step_scalars: dict[int, dict[str, torch.Tensor]] = {
                frame: {} for frame in batch_frames
            }
            if args.step_path == "batched":
                # One render_factorized call and one K2 launch for all
                # (frame, radar) units, then one backward over their summed
                # loss. The legacy path below computes the identical
                # quantities with twelve of each; it stays as the arbiter.
                units = [
                    (frame, radar_index, None)
                    for radar_index in range(len(RADARS))
                    for frame in batch_frames
                ]
                replayed = False
                scalar_rows = None
                if static_path and (_STATIC_SHORT_STEP
                                    or len(batch_frames) == args.frame_batch):
                    # P2: the fixed-shape step. Its per-step host inputs are
                    # the frames' positions in the flat cache and the unit
                    # mask; every other per-step quantity is derived from
                    # them on the device. The epoch's short last batch runs
                    # padded and masked through the same captured step
                    # (HUPR_STATIC_SHORT_STEP=0 restores the old dynamic
                    # fallback for it).
                    stage_batch.copy_(epoch_batch_index[step_index])
                    if epoch_step_mask is not None:
                        stage_mask.copy_(epoch_step_mask[step_index])
                    # The step's outputs carry frame_batch slots per radar;
                    # a short batch's real units sit in the first
                    # len(batch_frames) slots of each radar half.
                    scalar_rows = [
                        radar_index * args.frame_batch + slot
                        for radar_index in range(len(RADARS))
                        for slot in range(len(batch_frames))
                    ]
                    if graph_state["graph"] is not None:
                        # P3: forward and backward are one graph launch. The
                        # outputs are the captured tensors, refreshed in place
                        # by the replay — and read out below by a `torch.stack`
                        # that allocates outside the graph, so the per-step
                        # scalars do not alias what the next replay overwrites.
                        if scene_step or graph_state["graph_off"] is None:
                            graph_state["graph"].replay()
                            totals, ra_terms, md_terms = graph_state["outputs"]
                        else:
                            graph_state["graph_off"].replay()
                            totals, ra_terms, md_terms = graph_state["outputs_off"]
                        replayed = True
                    else:
                        scene_grad["on"] = scene_step
                        totals, ra_terms, md_terms = static_step()
                        scene_grad["on"] = True
                else:
                    ra_profile, planes, dc_profile = unit_profiles(units)
                    totals, ra_terms, md_terms, _ = losses_from_profiles_units(
                        units, ra_profile, planes, dc_profile,
                    )
                unit_scalars = torch.stack(
                    (totals.detach(), ra_terms.detach(), md_terms.detach()), dim=-1,
                )
                if epoch == 0 and step_index < _PRINT_STEPS:
                    # the route gate: the first steps' per-unit terms, to the digit (nine
                    # significant digits round-trip a float32, so equal lines are bitwise)
                    print(f"step {step_index} per-unit (total, ra, md): "
                          + " ".join(f"{float(t):.9g}/{float(r):.9g}/{float(m):.9g}"
                                     for t, r, m in unit_scalars.tolist()), flush=True)
                for index, (frame, radar_index, _) in enumerate(units):
                    row = index if scalar_rows is None else scalar_rows[index]
                    step_scalars[frame][RADARS[radar_index]] = unit_scalars[row]
                if not replayed:
                    batch_loss = totals.sum()
                    descend(batch_loss)
            else:
                for radar in RADARS:
                    pairs = [frame_components(frame, radar) for frame in batch_frames]
                    profiles_batch = gain_stack()[RADARS.index(radar)] * (
                        slow_time_assemble_batched(
                            current_background(radar),
                            [components for components, _ in pairs],
                            [rates for _, rates in pairs],
                        )
                    )
                    batch_loss = None
                    for offset, frame in enumerate(batch_frames):
                        total, ra_l2, md_l2, _ = losses_from_profiles(
                            frame, radar, profiles_batch[offset],
                        )
                        step_scalars[frame][radar] = torch.stack(
                            (total.detach(), ra_l2.detach(), md_l2.detach())
                        )
                        batch_loss = total if batch_loss is None else batch_loss + total
                    descend(batch_loss)
            if epoch == 0 and step_index < _PRINT_STEPS:
                # the route gate, backward half: every group's gradient norm
                norms = []
                for gi, group in enumerate(optimizer.param_groups):
                    total_sq = sum(float(p.grad.double().square().sum()) for p in group["params"]
                                   if p.grad is not None)
                    norms.append(f"g{gi}={total_sq ** 0.5:.9g}")
                if normals_optimizer is not None and model.raw_normals_flat.grad is not None:
                    norms.append(f"normals={float(model.raw_normals_flat.grad.double().norm()):.9g}")
                print(f"step {step_index} gradient norms: " + " ".join(norms), flush=True)
            # `rms_clip_grad` from 3DPS (train_gaussian.py:753-767),
            # inlined: scale the group's gradient so its RMS does not
            # exceed the group's clip.
            for group in optimizer.param_groups:
                clip = group.get("clip")
                if clip is None:
                    continue
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    grad = parameter.grad.data
                    rms = torch.sqrt(torch.mean(grad * grad)).clamp_min(1e-30)
                    grad.mul_(torch.clamp(clip / rms, max=1.0))
            if scene_step:
                optimizer.step()
            else:
                # the scene's groups sit out: their parameters are taken out
                # of the groups for this step, so Adam neither moves them nor
                # touches their moments
                held = [(group, group["params"]) for group in optimizer.param_groups
                        if group.get("scene")]
                for group, _ in held:
                    group["params"] = []
                optimizer.step()
                for group, params in held:
                    group["params"] = params
            if normals_optimizer is not None:
                normals_optimizer.step([
                    (frame, model.frame_row_index(frame)) for frame in batch_frames
                ])
            # The background's materials and normals just moved, so every
            # memoized background belongs to a parameter state that no longer
            # exists. Bumping the version retires them all in O(1).
            parameter_version["n"] += 1
            for frame in batch_frames:
                pending_records.append(records[frame])
                pending_scalars.append(torch.stack(
                    [step_scalars[frame][radar] for radar in RADARS]
                ))
        # One transfer for the whole epoch. The record values and the
        # per-frame mean are then computed in numpy float64 over the same
        # float32 scalars in the same order as the per-step float() path.
        epoch_block = torch.stack(pending_scalars).cpu().numpy().astype(np.float64)
        torch.cuda.synchronize()
        steps_seconds = time.perf_counter() - epoch_start
        for step_record, values in zip(pending_records, epoch_block):
            for radar_index, radar in enumerate(RADARS):
                step_record[radar] = {
                    "total": float(values[radar_index, 0]),
                    "ra_l2": float(values[radar_index, 1]),
                    "md_l2": float(values[radar_index, 2]),
                }
            step_record["mean_total"] = float(np.mean([
                step_record[radar]["total"] for radar in RADARS
            ]))
            epoch_totals.append(step_record["mean_total"])
            history.append(step_record)
        # `--eval-every N` runs the evaluation on every Nth epoch, plus the
        # last one and the geometry-phase boundary, both of which selection
        # reads. On a skipped epoch `held_out` is +inf, so the snapshot branch
        # below simply does not fire -- N coarsens best-checkpoint selection
        # by the same factor, and that is the whole trade. Restores the flag
        # removed in 95fb78a, when the pass was 0.169 s of a 1.15 s epoch;
        # it is 0.472 s of 1.747 s now (docs/CHAIN_TIMING_ANALYSIS.md S9).
        evaluated = ((epoch + 1) % args.eval_every == 0
                     or epoch + 1 == args.epochs
                     or (args.geometry_epochs > 0
                         and epoch + 1 == args.geometry_epochs))
        if evaluated:
            validation = evaluate(validation_frames)
            evaluations += 1
            torch.cuda.synchronize()
            held_out = float(np.mean(
                [validation[radar]["total"] for radar in RADARS]))
            held_out_text = f"{held_out:.6g}"
            md_text = " ".join(f"{radar[0].upper()}_md={validation[radar]['md_l2']:.6g}"
                               for radar in RADARS)
        else:
            validation = None
            held_out = float("inf")
            held_out_text = "skipped"
            md_text = " ".join(f"{radar[0].upper()}_md=skipped" for radar in RADARS)
        eval_seconds = time.perf_counter() - epoch_start - steps_seconds
        print(
            f"epoch={epoch + 1}/{args.epochs} phase={phase} "
            f"train_total={np.mean(epoch_totals):.6g} "
            f"held_out_total={held_out_text} "
            f"{md_text} "
            f"steps_s={steps_seconds:.3f} eval_s={eval_seconds:.3f} "
            f"step_ms={1000.0 * steps_seconds / max(1, step_index + 1):.2f}"
            # Only when a schedule is active, so the default log line is
            # byte-identical to every run recorded before this flag existed.
            + (f" lr_scale={learning_rate_scale(epoch):.4f}"
               if args.lr_schedule != "none" else ""),
            flush=True,
        )
        if noise_model is not None and (epoch + 1) % max(1, args.eval_every) == 0:
            # The floor's own trajectory: its log scale and the range
            # correction's extremes per radar (the DCT correction is clamped
            # to +/-3, i.e. x20; a coefficient at the clamp is a fit that
            # wanted more than the model allows and is worth seeing).
            with torch.no_grad():
                correction = (noise_model.range_coeff @ noise_model.basis)
                print("noise floor: " + " ".join(
                    f"{radar}: log_scale {float(noise_model.log_scale[i]):+.3f} "
                    f"range correction [{float(correction[i].min()):+.2f}, {float(correction[i].max()):+.2f}]"
                    for i, radar in enumerate(RADARS)), flush=True)
            if gain_learned:
                print("absolute gain: " + " ".join(f"{radar}: {value:.3f}" for radar, value in gain_values().items()),
                      flush=True)
            print("joint gains: " + " ".join(f"{float(v):.2f}" for v in joint_gains().detach().cpu()), flush=True)
        if args.geometry_epochs > 0 and epoch + 1 == args.geometry_epochs:
            # End of the geometry phase: what it produced is the frozen
            # starting point of the velocity phase, and best-state selection
            # restarts from here so a good-RA/camera-velocity epoch cannot
            # shadow the velocity training.
            best_loss = float("inf")
            best = snapshot()
        elif phase == "velocity":
            # Per-frame velocity parameters cannot generalize to held-out
            # frames by construction, so the held-out total is blind to this
            # phase; select by training loss and keep the held-out
            # temporal-discrimination delta as the reported guard.
            train_total = float(np.mean(epoch_totals))
            if train_total < best_loss:
                best_loss = train_total
                best = snapshot()
        elif held_out < best_loss:
            best_loss = held_out
            best = snapshot()

    # The epochs are over: give the captured step's pool back before the post-training work,
    # which is where a 24 GB card runs out (see `release_graph_pool`).
    release_graph_pool()
    model.load_state_dict(best["model"])
    material.load_state_dict(best["material"])
    with torch.no_grad():
        background.raw_materials.copy_(best["background_raw_materials"])
        background.rotations.copy_(best["background_rotations"])
        absolute_log_gain.copy_(best["absolute_log_gain"])
        joint_log_gain.copy_(best["joint_log_gain"])
    if args.person_level_calibration != "off" and args.person_level_calibration_when == "end":
        calibrate_person_level()
    final_validation = two_set_evaluation()

    # Radiometric person calibration: one closed-form scalar per radar,
    # fitted once at the trained state, stored in the checkpoint. Replaces
    # the guarded render-time amplitude-refit stage for velocity-field runs.
    radiometric_gains = {}
    with torch.no_grad():
        for radar_index, radar in enumerate(RADARS):
            if args.step_path == "legacy":
                target_powers, predicted_powers, pedestal_powers = [], [], []
                for frame in training_frames:
                    view = cache[frame]["views"][radar]
                    prediction = render_losses(frame, radar)[3]
                    target_power = view["md_target"].square()
                    target_powers.append(float(target_power.sum()))
                    pedestal_powers.append(
                        float(target_power.min()) * target_power.numel()
                    )
                    predicted_powers.append(float(prediction.square().sum()))
            else:
                # L9: the same 590 per-frame renders, batched like evaluation.
                # The three per-frame powers stay on the device and cross the
                # bus once per chunk instead of three times per frame (3,540
                # round trips over the two radars).
                rows = []
                pass_backgrounds = (
                    None if scene_visibility is not None
                    else current_backgrounds())
                for start in range(0, len(training_frames), args.eval_frame_batch):
                    chunk = training_frames[start:start + args.eval_frame_batch]
                    units = [(frame, radar_index, None) for frame in chunk]
                    geometry = unit_geometry(units)
                    components, unit_of_path = unit_render(geometry)
                    prediction = md_prediction_units(units, unit_scaled_profiles(
                        units, components, unit_rates(geometry, units), unit_of_path,
                        pass_backgrounds,
                    )[1])
                    target_power = unit_views(units)["md_target"].square()
                    rows.append(torch.stack((
                        target_power.sum(dim=-1),
                        target_power.min(dim=-1).values * target_power.shape[-1],
                        prediction.square().sum(dim=-1),
                    ), dim=-1))
                block = torch.cat(rows).cpu().numpy().astype(np.float64)
                target_powers = list(block[:, 0])
                pedestal_powers = list(block[:, 1])
                predicted_powers = list(block[:, 2])
            radiometric_gains[radar] = radiometric_person_gain(
                target_powers, predicted_powers, pedestal_powers,
            )
    print(
        "radiometric person gains: "
        + " ".join(f"{radar[0].upper()} {radiometric_gains[radar]:.4f}" for radar in RADARS), flush=True,
    )

    # Runtime proof that nothing beyond the two terms was optimized: the
    # logged totals must equal ra + lambda * md on every step, up to
    # float32 rounding of the tensor sum (the logged parts are float32
    # values re-added here in float64).
    two_term_identity = all(
        abs(
            row[radar]["total"]
            - (row[radar]["ra_l2"] + args.microdoppler_weight * row[radar]["md_l2"])
        ) <= 1e-6 * max(1.0, abs(row[radar]["total"]))
        for row in history for radar in RADARS
    )

    with torch.no_grad():
        field = model.full_field().detach().cpu()
        field_delta = (field - initial_field.cpu())
    checkpoint_result = {
        "motion_representation": "velocity_field",
        "velocity_parameterization": args.velocity_parameterization,
        "joint_velocity_field_mps": field,
        "joint_velocity_scale_learned": (model.joint_scale.detach().cpu() if model.bounded else None),
        "velocity_residual_bound_mps": model.residual_bound,
        "velocity_residual_relative": model.residual_relative,
        "velocity_residual_per_height": model.residual_per_height,
        "person_window_loss": {"weight": args.person_window_loss, "planes": args.person_window_planes, "snr": args.person_window_snr, "range_m": args.person_window_range, "lateral_m": args.person_window_lateral,
                               "side": ({r: ("+" if pw_side[r] == 0 else "-") for r in RADARS} if pw_on else None)},
        "learned_normals": model.learned_normals(),
        "inherited_joint_scattering_gains": joint_gains().detach().cpu(),
        "joint_scattering_gains_learned": args.joint_gains == "learned",
        "person_checkpoint": str(args.person_checkpoint.resolve()),
        "geometry_translation_y_m": args.geometry_translation_y_m,
        "velocity_differentiator": args.velocity_differentiator,
        "absolute_gains": gain_values(),
        "absolute_gain_learned": gain_learned and not gain_frozen_after_init,
        "radiometric_person_amplitude_gain": radiometric_gains,
        "person_level_calibration": person_level_calibration,
        "array_window": args.array_window,
        # Option B: the scene's slot (median 1) the renderer multiplies in as its vertex_areas weight, and the unit
        # the person's per-frame physical weights are divided by (None: the slot is 1.0 / the flag is off)
        "background_point_gains": (scene_slot[RADARS[0]].detach().cpu() if args.scene_solid_angle == "physical" else None),
        "solid_angle_unit": dict(solid_angle_unit) if solid_angle_unit else None,
        "person_solid_angle_weight": args.person_solid_angle_weight,
        "skinning": {
            "mode": "soft local-surface velocity",
            "temperature_m": args.skinning_temperature_m,
            "topk": args.skinning_topk,
        },
        "capture": capture_summary(),
    }
    if noise_model is not None:
        # THE TRAINED FLOOR: the expected windowed |RD|^2 per (radar, Doppler,
        # range), angle-mean, that the trainer added to its rendered power.
        # The renderer draws its receiver floor from this -- the noise model
        # is learned here and carried in the checkpoint, not read from a
        # calibration file anywhere.
        with torch.no_grad():
            checkpoint_result["noise_floor_power"] = noise_model.power().detach().cpu().float()
            checkpoint_result["noise_floor_radars"] = list(RADARS)
    if args.velocity_parameterization == "point-normal":
        checkpoint_result["point_normal_scalars"] = {
            key: value.detach().cpu() for key, value in model.point_scalars.items()
        }
    torch.save(checkpoint_result, args.output_dir / "velocity_field.pt")

    material_checkpoint = dict(person_checkpoint)
    material_checkpoint["shared_human_raw_material"] = material.raw.detach().cpu()
    material_checkpoint["shared_human_area_scale"] = float(
        material.log_area_scale.detach().exp()
    )
    material_checkpoint["person_radar_log_gain_delta"] = (
        material.radar_log_gain_delta.detach().cpu()
    )
    # The gains above carry downstream by multiplying `vertex_areas`,
    # because a gain is linear in the contribution. Materials and normals
    # are NOT linear in it, so they carry only by replacing the scene state
    # the renderer loads. Without this the trainer optimizes the scene,
    # selects a best snapshot, restores it -- and then every downstream
    # render reads `shared_background_state` straight from the person
    # checkpoint and gets the *untrained* scene, while the person material
    # it was co-trained against ships in full. Measured before this was
    # written: the material arm scored worse than the gains arm on RA
    # (Pearson H 0.450 vs 0.496, V MSE 2.84e+08 vs 1.45e+08) purely
    # because of this gap, and two material arms 10x apart in learning
    # rate produced identical renders.
    material_checkpoint["shared_background_state"] = {
        key: value.detach().cpu()
        for key, value in background.state_dict().items()
    }
    torch.save(
        material_checkpoint, args.output_dir / "best_shared_human_material.pt",
    )

    result = {
        "status": "completed",
        "absolute_receiver_gain": gain_values(),
        "absolute_gain_learned": gain_learned and not gain_frozen_after_init,
        "loss_protocol": (
            "L2(first-loop |RA|) + L2(Doppler-plane |RA|) + "
            f"{args.microdoppler_weight:g} * L2(micro-Doppler magnitude marginal)"
            + (f" + {args.elevation_weight:g} * (L2(loop-0 |RAE|) + L2(Doppler-plane |RAE|))" if elevation_on else "")
            + (f" + {args.person_window_loss:g} * L2(rms-normalised {'consumer-band Doppler-summed' if args.person_window_planes == 'consumer' else 'loop-0'} |RAE| of the person's window in the consumer's transform)" if pw_on else "")
            + (f" + {args.static_weight:g} * (DC-plane L2 in the person's cells / their target energy + outside / theirs"
               + (", RA and RAE views)" if elevation_on else ", RA view)") if static_on else "")
            + "; each L2 normalized by its measured target's energy (fixed constant, no prediction in the denominator); no other terms"
        ),
        "two_term_identity_holds_on_every_logged_step": two_term_identity,
        "velocity_parameterization": args.velocity_parameterization,
        "velocity_is_direct_parameter": True,
        "velocity_bounds_or_squashing": None,
        "trained_parameters": {
            # THREE parameterizations, so this cannot be a two-way ternary.
            # It was one, and `temporal-basis` -- the production default behind
            # every shipped result -- fell into the `else` and was recorded as
            # "per-point scalars along normals". It has no such parameter:
            # `point_scalars` is built only under `point-normal` (see
            # SkinnedVelocityField.__init__) and only added to the velocity
            # under that same test. Same shape as R099: a recorded field nobody
            # read back against the code.
            "velocity": {
                "joint":
                    "joint_velocity_field (frames,14,3), camera-initialized",
                "temporal-basis":
                    "temporal_coefficients (bands,14,3) in an orthonormal DCT "
                    "basis, initialized by projecting the camera velocities; "
                    "per-JOINT and skinned to points, no per-point term "
                    "(joint_velocity_field frozen)",
                "point-normal":
                    "per-point scalars along normals, zero-initialized "
                    "(camera-skinned joint field frozen)",
            }[args.velocity_parameterization],
            "material": "shared human material (raw, log_area_scale, radar_log_gain_delta)",
            "normals": "per-point unit normals of the training frames",
            "background": (
                "per-point 3DPS materials (raw_materials, 6 each) and normals "
                "(rotations, 4 each); positions frozen"
            ),
            "person_material": args.person_material,
            "person_area_scale": args.person_area_scale,
            "person_radar_gain_delta": args.person_radar_gain_delta,
            "person_material_lr": args.person_material_lr,
            "fixed": [
                "antenna patterns", "TX/RX geometry", "point positions",
                "skinning weights", "background geometry and materials",
                "inherited joint scattering gains", "absolute receiver gains",
            ],
        },
        "microdoppler_weight": args.microdoppler_weight,
        "ra_loss": args.ra_loss,
        "temporal_bands": (
            args.temporal_bands
            if args.velocity_parameterization == "temporal-basis" else None
        ),
        "noise_model": (
            {"enabled": False} if noise_model is None else {
                "enabled": True,
                "form": "learned expected power: exp(2 log_scale) * frozen "
                        "Doppler-range prior^2 * exp(DCT range correction)",
                "envelope_prior": (str(args.noise_envelope_calibration)
                                   if args.noise_envelope_calibration is not None
                                   else "this sequence's off-person floor"),
                "range_coefficients": args.noise_range_coefficients,
                "in_autograd_graph": True,
                "not_learned_here": (
                    "AR(1) frame correlation and angular correlation do not "
                    "change the expected power and are set per sequence from "
                    "the measured data outside the graph"
                ),
                **noise_model.state_dict_summary(),
            }
        ),
        "md_noise_floor_legacy": {
            radar: 0.0
            for radar in RADARS
        },
        "radiometric_person_amplitude_gain": radiometric_gains,
        "training_schedule": (
            "joint updates" if args.geometry_epochs == 0 else
            f"sequenced: {args.geometry_epochs} geometry epochs then "
            f"{args.epochs - args.geometry_epochs} velocity epochs"
        ),
        "geometry_epochs": args.geometry_epochs,
        "train_all_frames": args.train_all_frames,
        "validation_frames_held_out": not args.train_all_frames,
        "training_frames": training_frames,
        "validation_frames": validation_frames,
        "epochs": args.epochs,
        # Selection granularity, not a runtime detail: the best checkpoint can
        # only be taken on an evaluated epoch.
        "eval_every": args.eval_every,
        "lr_schedule": args.lr_schedule,
        "plane_mask_db": args.plane_mask_db,
        "plane_mask_keep_energy": args.plane_mask_keep_energy,
        "plane_mask_snr": args.plane_mask_snr,
        "plane_target_dtype": args.plane_target_dtype,
        "plane_sample_count": args.plane_sample_count,
        "plane_mask_dilate": args.plane_mask_dilate,
        "lr_decay_start_frac": args.lr_decay_start_frac,
        "lr_min_factor": args.lr_min_factor,
        "evaluations": evaluations,
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "best_held_out_total": best_loss,
        "joint_field_delta_stats_mps": {
            "rms": float(field_delta.square().mean().sqrt()),
            "max_abs": float(field_delta.abs().max()),
            "trained_frame_rms": float(
                field_delta[training_frames].square().mean().sqrt()
            ),
        },
        "geometry_translation_y_m": args.geometry_translation_y_m,
        "velocity_differentiator": args.velocity_differentiator,
        "rae_route": _RAE_ROUTE if elevation_on else None,
        "capture": capture_summary(),
    }
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    # Attach BEFORE the file is written, or the number reaches stdout and
    # never the artifact -- which is what happened on the first repeats.
    result["peak_vram"] = peak_vram()
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def make_parser() -> argparse.ArgumentParser:
    dataset = dataset_from_argv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--person-checkpoint", type=Path, required=True)
    radar_configs(parser, dataset)
    parser.add_argument("--mm3dps-root", type=Path, default=Path(MM3DPS_ROOT))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--velocity-parameterization",
        choices=("joint", "point-normal", "temporal-basis"),
        default="joint",
    )
    parser.add_argument(
        "--ra-loss", choices=("l2", "pearson"), default="l2",
        help="RA term form: normalized L2 (default) or 1 - Pearson on "
             "linear magnitudes (scale-invariant correlation)",
    )
    parser.add_argument(
        "--md-noise-floor-reference", type=Path, default=None,
        help="REMOVED. This pointed at a previous render pair from which a "
             "frozen 13-element noise floor was estimated before training. "
             "The floor is now part of the differentiable forward model "
             "(--noise-model); passing this is an error so no run silently "
             "keeps the old pre-calibration.",
    )
    parser.add_argument(
        "--noise-model", choices=("on", "off", "measured"), default="on",
        help="learn the expected noise floor as part of the forward model: "
             "exp(2*log_scale) * frozen Doppler-range prior^2 * exp(DCT range "
             "correction), ~9 parameters per radar, added to the predicted "
             "power under the micro-Doppler and 63-plane terms. 'off' renders "
             "no floor at all and is the arbiter.")
    parser.add_argument(
        "--noise-envelope-calibration", type=Path, default=None,
        help="an ablation only: an envelope file whose "
             "normalized_doppler_range_profile replaces the sequence's own "
             "off-person floor as the learned noise model's prior shape")
    parser.add_argument(
        "--noise-range-coefficients", type=int, default=8,
        help="DCT coefficients in the learned range correction, per radar. A "
             "global scale provably cannot fit the level: the amplitude "
             "deficit is range-dependent and inverts between the radars "
             "(docs/NOISE_MODEL_RESULTS.md, change 2).")
    parser.add_argument("--noise-lr", type=float, default=3e-2)
    parser.add_argument("--gain-lr", type=float, default=1e-2,
                        help="Adam rate of the learned log receiver gain (--absolute-gain auto)")
    parser.add_argument(
        "--joint-gains", choices=("learned", "fixed"), default="learned",
        help="the 14 joint scattering gains: learned with the materials (geometric mean 1), "
             "or fixed at 1 (the per-point material alone spreads the body's level)")
    parser.add_argument(
        "--noise-initial-fraction", type=float, default=0.1,
        help="the learned floor starts at this fraction of the measured "
             "micro-Doppler power. The absolute value is not important -- it "
             "is learned -- but the starting point must put the gradient "
             "above Adam's epsilon, which the renderer's nuisance scale does "
             "not (it is a multiplier on a background quantile there).")
    parser.add_argument(
        "--diffuse-floor-calibration", type=Path, default=None,
        help="diffuse_floor_calibration.json. When given, the diffuse floor "
             "joins the training forward model as expected power, so the "
             "trainer and the renderer model the same processes. Support and "
             "range shape come from the scene the camera reconstructed, the "
             "same way the renderer builds them.")
    parser.add_argument(
        "--diffuse-floor-scale", type=float, default=1.0,
        help="multiplies the calibrated level, exactly as the renderer's flag "
             "of the same name does. Must match the render's value.")
    parser.add_argument(
        "--diffuse-floor-support-smoothing-bins", type=int, default=9)
    parser.add_argument("--diffuse-floor-scatterers-per-bin", type=int, default=4)
    parser.add_argument("--diffuse-floor-seed", type=int, default=100)
    parser.add_argument(
        "--diffuse-floor-expectation-draws", type=int, default=200,
        help="Monte-Carlo draws for the floor's expected power. Its marginals "
             "are reductions over the other two axes, so they converge fast: "
             "1.1%% split-half on the Doppler marginal at 150 draws.")
    parser.add_argument(
        "--person-solid-angle-weight", choices=("on", "off", "physical"), default="off",
        help="physical (docs/STATIC_SEQS_PLAN.md Option B, 2026-09-19): the slot IS the solid angle, A_i |cos|/d^2 "
             "from the posed SMPL mesh's vertex areas and mesh normals, in the scene's unit (median scene dOmega), "
             "computed, not learned -- the person's area scale is then unit and untrained. "
             "on: weight the person's `areas` by cos_o/d^2 at unit mean -- the Monte-Carlo "
             "weight of the depth camera's cloud, whose sampling density is the view's. "
             "The SMPL mesh's is not, and on it the weight is one more cosine on every "
             "oblique surface (the limbs' sides): off is the production setting since "
             "docs/FIDELITY_FIX_RESULTS.md 3.12, on the 2026-09-16 form. Must match the "
             "renderer's flag of the same name, or the materials are trained under a "
             "different forward model than they are evaluated in.")
    for radar in dataset.radars:
        parser.add_argument(radar.flag("noise-initial-scale"), type=float,
                            default={"horizontal": 0.0029, "vertical": 0.0028}.get(radar.name, 0.0))
    parser.add_argument(
        "--temporal-bands", type=int, default=420,
        help="temporal-basis mode: DCT bands kept; 420 of 600 frames at "
             "10 Hz caps the representable velocity spectrum at 3.5 Hz — "
             "gait harmonics stay, detection jitter cannot be expressed",
    )
    parser.add_argument("--training-frames", type=int, default=60)
    parser.add_argument(
        "--train-all-frames", action="store_true",
        help="train on every chosen frame instead of excluding the [::4] "
             "evaluation subset; with --training-frames 600 this trains all "
             "590 differentiator-valid frames of the sequence",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument(
        "--eval-every", type=int, default=1,
        help="run the evaluation every N epochs; the last epoch and the "
             "geometry-phase boundary always evaluate. Snapshots are taken "
             "only on evaluated epochs, so N coarsens best-checkpoint "
             "selection granularity by exactly the same factor -- that is the "
             "whole trade, and it changes the trained output. Default 1 keeps "
             "the historical per-epoch cadence. At 500 epochs the pass is "
             "0.472 s of a 1.747 s epoch (27%), so N=5 takes training from "
             "914 s to 725 s; measured cost on one 500-epoch trace was "
             "+0.34%% on the selected held_out against a 0.88%% run-to-run "
             "spread (docs/CHAIN_TIMING_ANALYSIS.md section 9).")
    parser.add_argument(
        "--static-term", choices=("off", "on"), default="off",
        help="docs/STATIC_SEQS_PLAN.md P1 (an ablation; default off: as a sixth term on the five-term objective it did not "
             "move the person's level knobs -- area_scale 1.3e-3 -> 1.1e-3 on 176, 6.8e-4 -> 7.8e-5 on 108 -- and cost "
             "the moving band: 183 MD spectrogram r 0.948 -> 0.838, RD 0.72 -> 0.62; step 22-24 -> 27-31 ms): the DC plane -- the slow-time mean over the "
             "64 chirps, the static return at 8x the one-chirp anchor's SNR -- as its own normalised L2, the person's "
             "RA-mask cells over their own target energy PLUS the rest over theirs, in the RA view and (elevation on) "
             "the RAE view. The anchor and plane terms are untouched. off = the objective before it (ablation)")
    parser.add_argument("--static-weight", type=float, default=1.0, help="weight of the DC-plane term in the total")
    parser.add_argument(
        "--scene-solid-angle", choices=("off", "physical"), default="off",
        help="physical: the scene's `areas` slot is A_i |cos|/d^2 (local area from the cloud's spacing, the loaded "
             "normals, fixed) in its own median unit instead of 1.0; saved as background_point_gains for the renderer")
    parser.add_argument(
        "--array-window", choices=("hann", "none"), default="none",
        help="the azimuth / elevation taper of THIS trainer's beamformer, targets and predictions alike (range and "
             "Doppler keep the pipeline's Hann). none (default since nowin, docs/STATIC_SEQS_NEXT.md 1d) = the downstream "
             "consumers' view (HuPR's preprocessing and the HRRadarPose adapter beamform untapered): a rendered wall's "
             "coherent response is 10x the recording's there while the Hann view reads 1x; gate 176 37 -> 28, 168 27 -> 13, "
             "108 28 -> 17 cm. hann = the pre-nowin trainer. Set for this process only; the products and the evaluation "
             "keep the pipeline's window and the suite scores both views")
    parser.add_argument(
        "--person-level-calibration-target", choices=("total", "person"), default="total",
        help="peak-doppler: what the mask's peak is read on. total (default) = the whole map, both sides (a scene scatterer "
             "inside the mask can be the peak: RT-Pose 164 / 153); person = the person's own contribution -- the measured map "
             "less the measured static scene (its 20th percentile per cell over the frames), the render less its scene-only "
             "render. Rejected as the default (pcal, docs/STATIC_SEQS_NEXT.md 1c): where the person stands still the "
             "percentile at the person's cell IS the person, so the target collapses (176 x1.27 instead of x2.2, gate 28 -> 35). "
             "side = the mask restricted to +-6 angle bins about the rendered person's own peak angle: the cached mask holds "
             "the person's angle and its MIRROR, and scene on the mirror side set the level on 164 / 153 (person at 0.67x)")
    parser.add_argument(
        "--person-level-calibration-view", choices=("trainer", "downstream"), default="downstream",
        help="peak-doppler: the transform the level is read in. trainer = this trainer's untapered full-array RAE beamformer "
             "(channels_to_grid2d, averaged slots) and its person mask; downstream = the consumer's own transform (RT-Pose: "
             "the HRRadarPose adapter's polar on the TI virtual-array table, one channel per slot) on the pelvis's window -- "
             "the two read the recording's person 1.2-1.5x apart on 153 / 164 (docs/STATIC_SEQS_NEXT.md 1e); HuPR's input "
             "is normalised per map and keeps the trainer view")
    parser.add_argument(
        "--person-window-loss", type=float, default=1.0,
        help="weight of the person-window term: the loop-0 |RAE| SHAPE of the registered person's window (the mesh pelvis "
             "+-range / +-lateral, every elevation bin) in the CONSUMER's own transform (RT-Pose: the HRRadarPose adapter's "
             "polar on the TI table, one channel per slot, 128 azimuth x 32 elevation, untapered; HuPR: the released "
             "generateHeatmap grid, 64 azimuth x 8 elevation with its fftshift + flip), render against recording, each "
             "normalised by its own rms over the window so the term cannot move the level. The real-trained RT-Pose model "
             "reads a rendered seated person +40 cm too high from that pattern -- the range x azimuth x elevation "
             "magnitudes of the person's cells, the elevation structure first -- and nothing in the objective looked at it "
             "(docs/SEQ164_ANALYSIS.md 3b, SEQ164_PLAN.md step 1). Default 1 = the recipe since pwl (STATIC_SEQS_NEXT 1f): "
             "164 41.5 -> 29.2, HuPR 38 RAc +0.08 / +0.11, HuPR 40 horizontal MDs -0.07 recorded as its cost. 0 = off")
    parser.add_argument(
        "--window-rae-terms", choices=("off", "on"), default="off",
        help="add the two WINDOWED RAE terms: the loop-0 |RAE| and the plane |RAE| of the elevation "
             "terms restricted to the person's box, each with its own normalisation over the box. "
             "They are the same functional as the global elevation terms at a second scope, so the "
             "objective becomes one term over {global, windowed} x {loop-0, per plane}. Intended to "
             "replace the person-window term (which the loop-0 one generalises, in the trainer's own "
             "view rather than the consumer's) and micro-Doppler (which the plane one replaces at a "
             "measured 0.80x / 1.00x of its velocity gradient). Use with --person-window-loss 0 and "
             "--ablate-loss-term ...,md (docs/OBJECTIVE_ABLATION.md).")
    parser.add_argument("--window-rae-weight", type=float, default=1.0,
                        help="weight of the two windowed RAE terms in the total")
    parser.add_argument(
        "--window-el-bound", choices=("off", "on"), default="off",
        help="bound the two windowed RAE terms in elevation to the bins the person actually moves "
             "in. The bound is read off the windowed PLANE target, which has clutter removed, so "
             "the moving energy identifies its own elevation extent per view and no axis arithmetic "
             "is involved -- the elevation convention is never reimplemented. Without it the "
             "windowed terms span every elevation bin, which on the cascade is a full-height column "
             "through the person's range and azimuth, mostly scene. Meaningless on HuPR, whose RAE "
             "view has two elevation bins.")
    parser.add_argument("--window-el-threshold", type=float, default=0.1,
                        help="an elevation bin is kept when its share of the windowed plane target's "
                             "moving energy is at least this fraction of the peak bin's")
    parser.add_argument(
        "--microdoppler-support", choices=("person", "window"), default="person",
        help="which cells the micro-Doppler marginal sums over. person (default): the person's own "
             "(range, angle) support, from its point cloud. window: the person-window term's box "
             "instead -- +-person-window-range in range and +-person-window-lateral across the line "
             "of sight about the pelvis -- so micro-Doppler and the window term share one region. "
             "The box is built by passing a synthetic cloud that fills it through the same mask "
             "function the person uses, so the angle convention (the beamformer's mirror) is shared "
             "rather than reimplemented.")
    parser.add_argument(
        "--person-window-planes", choices=("consumer", "loop0", "resolved"), default="loop0",
        help="which Doppler content the person-window term compares. consumer (default): the Doppler-summed |RAE| the consumer "
             "reads, from the trainer's rendered planes -- HuPR: the clutter-removed chirps 24..39 of the released heatmap (its "
             "model never sees the static return); RT-Pose: the DC (the unwindowed slow-time mean, the loop-0 anchor x 64) plus "
             "every non-DC plane (x2, the Hann window's coherent gain), as the adapter's polar sums them. loop0: the loop-0 "
             "snapshot alone -- the default (pwl, docs/STATIC_SEQS_NEXT.md 1f: 164 29.2 against the consumer band's 31.4; both cost "
             "HuPR 40's horizontal micro-Doppler ~0.07 of MDs, the consumer band gains its vertical)")
    parser.add_argument(
        "--person-window-snr", type=float, default=0.0,
        help="the frame's term is weighted by the window's information content in the consumer's view: its mean power over the "
             "reference power of its range rows outside it (--person-window-snr-reference), mapped linearly from 0 at 1x to 1 at this ratio; "
             "<= 1 = no weighting. A window the "
             "consumer's band does not resolve above its surroundings (a still person in HuPR's clutter-removed band: the "
             "floor) is not a shape target and inflated the moving band when it was one (HuPR 40, STATIC_SEQS_NEXT 1f) -- but the "
             "weighting let RT-Pose 164's seated reading back in with either reference (pws 45.4, pwm 50.2), so it is OFF by default (0)")
    parser.add_argument("--person-window-snr-reference", choices=("median", "mean"), default="median",
                        help="what the window's power is compared with for the frame weight: the median cell power of its range rows "
                             "outside it (the floor; default) or their mean (a wall at the person's range then switches the term "
                             "off where it matters: RT-Pose 164's seated frames beside the far wall, pws)")
    parser.add_argument("--person-window-range", type=float, default=0.6, help="the person window's half-width in range (m)")
    parser.add_argument("--person-window-lateral", type=float, default=1.2,
                        help="the person window's half-width across the line of sight (m; 1.2 covers the rendered person's azimuth streak)")
    parser.add_argument(
        "--person-level-calibration-cell", choices=("person", "mask", "side"), default="mask",
        help="peak-doppler: which cell the level is read at. mask (default) = the mask's peak, both sides; person = the "
             "rendered person's own peak cell in the mask against the recording's peak within +-1 cell of it. The mask's "
             "peak can be a scene scatterer (RT-Pose 153 read 1.07x and lowered the person), but the person cell was worse "
             "on the 11 (cal2, docs/STATIC_SEQS_NEXT.md 1c: the walking sequences read 1.23-1.27x and were dimmed x0.8, "
             "153 33 -> 53 cm; 176 37 -> 107 cm): the render's own peak is not where the recording's person peak is")
    parser.add_argument(
        "--person-level-calibration-when", choices=("init", "end"), default="end",
        help="when the level is set: end (default) = after training, on the trained velocity field and normals that ship; "
             "init = before training (the trained field defocuses a standing body's beamformed peak ~1.8x, TDM MIMO)")
    parser.add_argument(
        "--person-level-calibration", choices=("off", "peak", "peak-doppler", "energy"), default="peak-doppler",
        help="at the start, after the gain init, set the person's area scale so the person mask's peak of THIS stage's "
             "render matches the measured one (median over the training frames), then keep it (--person-area-scale "
             "frozen). peak: the loop-0 |RA|. peak-doppler: the magnitude summed over all 64 Doppler bins (the 63 "
             "windowed mean-removed planes and the DC plane), what a downstream tensor reads at the person -- the "
             "static scene is DC-only, so |B + k P_dc| + k sum_d |P_d| is solved from one render per frame. energy: the "
             "person mask's loop-0 ENERGY, sum over the mask of |B + k P|^2 against the measured sum of |T|^2 -- a peak "
             "depends on the beamformer (the trainer's azimuth row against a downstream 2D array read 1.2x and 0.44x on "
             "one render), an energy does not (docs/STATIC_SEQS_NEXT.md step 1)")
    parser.add_argument(
        "--static-anchor", choices=("loop0", "dc-mean"), default="loop0",
        help="the RA term's static-anchor plane. loop0 (default, historical): "
             "the raw first slow-time sample. dc-mean: the slow-time mean "
             "(the DC Doppler plane) — the same static information at "
             "~sqrt(64)=8x the SNR, on both the rendered and the measured "
             "side. Changing this changes the objective; batched step path "
             "only.",
    )
    parser.add_argument(
        "--velocity-edge-guard", type=int, default=0,
        help="frames dropped from each end of the sequence before choosing "
             "training frames. This is what made a 600-frame sequence train "
             "on 590; it is a differentiator-edge convention, not a held-out "
             "calibration split -- nothing was ever calibrated on those ten "
             "frames. The default is now 0 (train on all 600); 5 restores the "
             "historical behaviour and is the arbiter.")
    parser.add_argument(
        "--ablate-loss-term", default="none",
        help="ablation: drop loss components from the objective -- a comma-separated "
             "subset of loop0-ra, plane-ra, md, or 'none'. The dropped term is still "
             "computed and logged where cheap, but contributes neither to the total nor "
             "a gradient; 'md' is implemented as microdoppler-weight 0. A list is what a "
             "consolidation needs: 'loop0-ra,plane-ra' leaves the objective supervised "
             "through the RAE views alone.",
    )
    parser.add_argument(
        "--geometry-epochs", type=int, default=0,
        help="sequenced training: first N epochs update only material and "
             "normals (velocity frozen at camera init), the rest update only "
             "velocity on the frozen geometry; 0 = joint updates throughout",
    )
    parser.add_argument(
        "--microdoppler-weight", type=float, default=1.0,
        help="lambda between the RA and MD L2 terms — the only loss hyperparameter",
    )
    parser.add_argument(
        "--eval-frame-batch", type=int, default=15,
        help="frames per batched evaluation chunk (2x this many render units); "
             "affects only how the identical no-grad renders are grouped",
    )
    parser.add_argument(
        "--doppler-amplitudes", choices=("closed-form", "dft"), default="closed-form",
        help="how a path's Doppler-plane amplitudes are computed: the window's response "
             "at the path's fractional Doppler bin in closed form (default; 7e-7 of the "
             "float64 reference, `qa doppler-check`) or the 64-loop sum against the plane "
             "weights (the historical kernels, 5e-6 of the same reference; the A/B)")
    parser.add_argument(
        "--doppler-taps", type=int, choices=(0, 3, 5, 7, 9, 11, 13, 15), default=7,
        help="the Doppler PSF's taps: a path deposits into the planes within (taps-1)/2 bins of "
             "its own Doppler bin, as the range PSF is truncated to 15 bins; 0 renders every "
             "supported plane exactly. The truncation against the exact planes is 7e-4 at 15, "
             "7.7e-3 at 7, 2.3e-2 at 5 (`qa doppler-check --taps`); 400 epochs on RT-Pose 183, "
             "7 taps and exact are indistinguishable (spectrogram +0.941 / +0.933, RD occupied "
             "+0.399 / +0.394, the same learned floor) at 41 against 47 ms a step, and 5 taps "
             "saves 0.6 ms more at three times the truncation -- the default is 7")
    parser.add_argument(
        "--deterministic", choices=("on", "off"), default="off",
        help="make every reduction in the training step deterministic, so a "
             "step reproduces itself bit for bit: the ordered binning for "
             "K2b's gather plus torch's deterministic `index_add_`. Measured "
             "cost 0.73 -> 1.12 s per epoch, i.e. 6.1 -> 9.3 min per 500 "
             "(docs/RUNTIME_OPTIMIZATION_REPORT.md, Phase 5 W4), which is why "
             "it is off by default",
    )
    parser.add_argument(
        "--cuda-graph", choices=("on", "off"), default="on",
        help="capture the fixed-shape training step's forward and backward as "
             "a CUDA graph and replay it. Exact: the optimizer stays outside "
             "the graph and eager, because capturable Adam's arithmetic "
             "differs from the eager form's in the last bit")
    parser.add_argument(
        "--step-path", choices=("batched", "legacy"), default="batched",
        help="'batched' renders every (frame, radar) unit of a step in one "
             "render_factorized call and one slow-time splat; 'legacy' keeps "
             "the per-frame, per-radar loop as the pinned arbiter. The two "
             "compute the same quantities",
    )
    parser.add_argument("--profile-steps", type=int, default=0,
                        help="profile N steps then exit (dev tooling)")
    parser.add_argument("--adam", choices=("fused", "eager"), default="fused",
                        help="Adam's kernel: one fused launch over every parameter, or torch's per-tensor chain")
    parser.add_argument("--dump-targets", type=Path, default=None,
                        help="write the built targets to this file (dev tooling: the numerical gate)")
    parser.add_argument("--background-material-lr", type=float, default=1e-2)
    parser.add_argument("--background-rotation-lr", type=float, default=5e-3)
    parser.add_argument("--background-material-clip", type=float, default=1.0)
    parser.add_argument("--background-rotation-clip", type=float, default=0.5)
    parser.add_argument(
        "--person-material-lr", type=float, default=None,
        help="learning rate for the person material. Defaults to "
             "--geometry-lr, which is what a shared 6-vector has always used. "
             "3DPS trains its per-point materials at mat_lr=0.01 "
             "(train_gaussian.py:800), 10x that, and the background arm showed "
             "the recipe matters as much as the parameterization.",
    )
    parser.add_argument(
        "--person-area-scale", choices=("auto", "learned", "frozen", "unit"),
        default="auto",
        help="what the person's `areas` slot carries. That slot is the "
             "hemisphere solid-angle MC weight dOmega, entering as "
             "sqrt(G_rx * dOmega) -- NOT a surface area; 3DPS passes 1.0 for "
             "every point and so does this project's background, so the "
             "person's `exp(log_area_scale)` is a global amplitude-squared "
             "knob sitting in a geometric slot. `frozen` keeps the fitted "
             "value but stops training it and `unit` is 3DPS parity -- but "
             "both are for use WITH `--person-material per-point`, which has "
             "the capacity to absorb level. Measured with the default shared "
             "6-vector they cost real accuracy, so the default stays "
             "`learned`.",
    )
    parser.add_argument(
        "--person-radar-gain-delta", choices=("auto", "learned", "off", "frozen"),
        default="auto",
        help="the reciprocal H/V person gain, exp(+d) and exp(-d). 3DPS has no "
             "such term; it exists here to absorb H/V calibration mismatch in "
             "one degree of freedom. `off` sets it to 0 -- what the physics "
             "says if the absolute gains are right -- and is safe alongside a "
             "per-point material, but on the default geometry the fitted "
             "delta is -0.345, a 2x H/V imbalance a shared 6-vector cannot "
             "re-absorb, so the default stays `learned`.",
    )
    parser.add_argument(
        "--person-material", choices=("auto", "shared", "per-point"),
        default="shared",
        help="`shared` (the production setting since docs/FIDELITY_FIX_RESULTS.md 3.12: "
             "one material for the body -- per-point materials let the fit switch the "
             "limbs off): one 6-vector broadcast to every person point, as the person "
             "stage fitted it. `per-point`: one 6-vector per point of a fixed global "
             "point set, as 3DPS trains `raw_materials (N,6)`; requires a persistent "
             "point set. `auto`: per-point where the geometry allows it (the 2026-09-16 "
             "production form, an ablation).",
    )
    parser.add_argument("--velocity-lr", type=float, default=1e-2)
    parser.add_argument(
        "--velocity-residual-bound", type=float, default=0.0,
        help="temporal-basis: > 0 makes the field the mesh's kinematic joint velocities x a learned per-joint scale + a "
             "residual bounded by this many m/s per component (b tanh(. / b)); 0 = the free field. The free field puts a "
             "standing body's hips at 0.7-0.9 m/s against the mesh's 0.02-0.06 and its Doppler skirt 2-4x the real sway "
             "(docs/STATIC_SEQS_NEXT.md 1c, P2); postural sway / breathing / tremor are a few cm/s")
    parser.add_argument(
        "--velocity-residual-per-height", type=float, default=0.0,
        help="with the residual bound: + this x the pelvis's height above the frame's lowest joint (m/s per m) -- postural "
             "sway as an inverted pendulum: a standing body (pelvis ~1 m) sways, a seated one (~0.5 m) half as much, a lying "
             "one not at all (RT-Pose 164's seated frames are DC in the recording; a constant bound of 0.015 m/s spread them)")
    parser.add_argument(
        "--velocity-residual-relative", type=float, default=0.75,
        help="with --velocity-residual-bound b: the bound is b + this x the joint's kinematic speed (the mesh's velocity "
             "error grows with the motion it smooths); 0 = the constant bound")
    parser.add_argument("--geometry-lr", type=float, default=1e-3)
    # Constant-lr Adam runs the whole schedule at full step size, so late
    # epochs bounce in a noise ball instead of settling: R042 measured seq 256
    # still gaining +0.012-0.017 correlation from epoch 400 to 500. "cosine"
    # holds the base lr until `--lr-decay-start-frac` of the run, then decays
    # to `--lr-min-factor` of it, matching `train_gaussian.get_lr_scale` --
    # the same schedule `training/dual.py` already drives its Adam with.
    # Default "none" leaves the run bit-identical.
    parser.add_argument("--lr-schedule", choices=("none", "cosine"), default="none")
    # The 63-plane RA term's target cube is (views, 63, 244, 64) fp32 --
    # 4.5 GiB on a 590-frame sequence, the second-largest training
    # allocation. Most of those cells never carry signal: measured on seq
    # 102, only 8.1% (H) / 12.0% (V) come within 20 dB of the peak over the
    # whole video. `--plane-mask-db X` keeps cells whose across-view maximum
    # is within X dB of the cube's peak, dilated by `--plane-mask-dilate`
    # bins in range and angle so a cell that only ever sits just outside the
    # support still gets supervised. Off by default.
    #
    # NOTE this cannot save the term's dominant cost. Azimuth exists only
    # after the angle FFT, so a mask cannot make the render or the FFT
    # cheaper -- it shrinks the stored target and the elementwise chain.
    parser.add_argument("--plane-mask-db", type=float, default=None)
    # The threshold that predicts damage is not "X dB below peak" but how much
    # of the target this throws away -- and a fixed dB does not transfer
    # between sequences: -30 dB discards 12.3% of seq 102's L2 but 16.5% of
    # seq 256's, and only the latter regressed (R046/R047). This keeps the
    # smallest set of cells holding the requested fraction of the cube's
    # energy, solved per sequence from the cube itself. A linear fraction
    # rather than dB, per the project's linear-amplitude convention.
    #
    # 0.999 is the shipped value. The memory curve is flat across this
    # range -- 98.5% to 99.9% energy moves peak by ~6 points, ~260 MiB --
    # while the risk curve is steep, with a measured failure between 9% and
    # 16.5% of L2 discarded. 99.9% discards 3.2%, a 3-5x margin under the
    # nearest known-good point, and that margin costs almost nothing.
    parser.add_argument("--plane-mask-keep-energy", type=float, default=None)
    # Keep cells that ever rose K x above the noise floor at their own range.
    # The floor is estimated from the cube as the median over (plane, angle)
    # at each range bin -- most of those 63x64 samples are noise at any range,
    # so the median is robust, and it needs no model.
    #
    # This is the criterion that TRANSFERS. A per-range floor spans ~30x
    # across range on seq 102, so any single global threshold is mismatched at
    # most ranges by construction. Measured offline (R047): at K=5 seqs 102
    # and 256 keep 16.0% and 15.3% of cells and discard 8.7% and 9.1% of the
    # target L2 -- the same behaviour from the same K -- where -30 dB differs
    # 2x in cells between them and the energy rule differs by 6 points.
    parser.add_argument("--plane-mask-snr", type=float, default=None)
    # Render only the Doppler planes that carry signal. The support is
    # decided PER FRAME from that frame's measured Doppler at the person's
    # range -- bins above K x the local noise floor, unioned with the 13
    # micro-Doppler bins the MD term needs unconditionally -- and the rendered
    # set is the union of those per-frame supports over the run.
    #
    # Nothing here is fitted to a sequence: the width is whatever that
    # sequence's motion demands, so a fast-moving sequence simply renders more
    # planes. Worst case it selects all 63 and we are where we started; it can
    # never clip signal that is above the noise floor.
    #
    # Plane count is a linear multiplier on the K2 splat, which profile-step
    # measures at 52% of the training step (its backward alone is 30.6%), so
    # this is the largest lever left. Measured on seqs 38/102/256 at K=2:
    # 63 -> 30/35/30 planes.
    parser.add_argument("--doppler-support-snr", type=float, default=None)
    parser.add_argument("--elevation-term", choices=("off", "on"), default="on",
                        help="docs/FIDELITY_FIX_PLAN.md 1.4 F1 (the production setting, docs/FIDELITY_FIX_RESULTS.md): "
                             "also supervise the (range, azimuth, elevation) view of EVERY TX (loop-0 and the "
                             "Doppler planes, the same normalised L2), so the elevated TX rows the products ship "
                             "are fitted, not extrapolated; needs --training-array full. off = the pre-2026-09-16 "
                             "trainer, for an ablation only")
    parser.add_argument("--elevation-planes", choices=("all", "motion"), default="motion",
                        help="which Doppler planes the elevation plane term reads: every rendered plane, or the "
                             "micro-Doppler band only (the person's planes; the static scene is in loop-0) -- "
                             "the cascade's 8 elevation bins x 63 planes are the term's cost")
    parser.add_argument("--n-angle", type=int, default=None,
                        help="the azimuth angle-FFT size the loss uses on both sides; default: the radar's "
                             "training_n_angle (HuPR 16, the lossless 2N-1 of an 8-slot row; cascade 86, the "
                             "slot count -- docs/FIDELITY_FIX_RESULTS.md 3.4); the product's size (64 / 128) "
                             "is an ablation")
    parser.add_argument("--elevation-weight", type=float, default=1.0,
                        help="weight of the two elevation terms (loop-0 RAE + plane RAE) in the total")
    parser.add_argument("--rae-floor-gradient", choices=("off", "on"), default="off",
                        help="let the plane RAE term fit the learned noise floor, as the plane RA "
                             "term does, instead of reading it detached. Needed by any objective "
                             "that drops the plane RA term, which is otherwise the floor's only "
                             "substantial gradient path (docs/FIDELITY_FIX_RESULTS.md 3.8 detached "
                             "it while the scale factor was still wrong)")
    parser.add_argument("--training-array", choices=("azimuth", "full"), default="full",
                        help="which TX the trainer renders: every recorded TX (default since 2026-09-16: the "
                             "elevation term supervises them) or the azimuth sub-array the beamformer keeps "
                             "(the pre-2026-09-16 trainer; with --elevation-term off only)")
    parser.add_argument("--doppler-support-floor", choices=("loops-median", "off-person", "energy"),
                        default="loops-median",
                        help="which bins a view renders: above HuPR's median-over-loops floor "
                             "(loops-median); above the person rows' off-person cells at that bin "
                             "(off-person); or the fewest bins holding --doppler-support-energy "
                             "of the person cells' Doppler energy (energy)")
    parser.add_argument("--doppler-support-energy", type=float, default=0.99,
                        help="with --doppler-support-floor energy: the energy fraction the "
                             "rendered bins must carry")
    parser.add_argument("--render-band-always", choices=("on", "off"), default="on",
                        help="on (HuPR): every view renders the whole motion band plus its support; "
                             "off: only the bins the view supports, the micro-Doppler marginal "
                             "reading zero (the floor) at unsupported band bins")
    # Supervise a random K of the 63 planes per epoch instead of all of them.
    # The term is a MEAN over cells and the denominator is the full cube's
    # precomputed mean square, so a uniform subset is an unbiased estimator of
    # the full term -- no rescale. Over a 400-epoch run every plane is drawn
    # ~400*K/63 times. 0 = all 63 (default).
    parser.add_argument("--plane-sample-count", type=int, default=0)
    # Writes the across-view maximum of the plane target cube (63, range,
    # angle) plus the mask actually built, so the threshold can be chosen and
    # inspected offline against the real cube rather than guessed.
    parser.add_argument("--plane-mask-dump", type=Path, default=None)
    # velocity_field.py's own note beside the target cache: half precision
    # overflows on the raw magnitudes (~5e6 against fp16's 65504), but storing
    # `t / rms(t)` with the per-view rms beside it does not, and the term
    # already divides by mean(t^2) so the normalization is exact. Halves the
    # 4.4 GiB cube with no threshold and no sequence dependence. The residual
    # is fp16's ~1e-3 relative precision on the target, which is why this is
    # a flag under test rather than the default.
    parser.add_argument("--plane-target-dtype",
                        choices=("float32", "float16"), default="float32")
    parser.add_argument("--plane-mask-dilate", type=int, default=2)
    parser.add_argument("--lr-decay-start-frac", type=float, default=0.5)
    parser.add_argument("--lr-min-factor", type=float, default=0.01)
    parser.add_argument(
        "--scene-update-every", type=int, default=1,
        help="update the scene's materials and normals every k-th step only: on the other "
             "steps the scene renders without a gradient (its forward is unchanged, so the "
             "loss is), and its Adam groups are not stepped. The captured step exists in "
             "two versions. A different optimiser schedule for the scene, not a model "
             "change (docs/TRAINING_SPEED_PLAN_2.md 3.2); default 1 (every step)")
    parser.add_argument(
        "--antenna-coupling-calibration", type=Path, default=None,
        help="directory of <radar>_coupling.npy (HuPR's seed or `calibration sequence-coupling`): "
             "the boards' static near-field coupling, added to the anchor plane after the absolute "
             "gain, as the renderer adds it; default none")
    parser.add_argument("--scene-visibility", type=Path, default=None,
                        help="(frames, N) bool .npy from `pointset "
                             "scene-shadow-mask`: False where the person "
                             "occludes that scene point on that frame. It is "
                             "applied as a per-point GAIN on the cached scene "
                             "splat, so it costs a gather and a multiply "
                             "rather than a re-render. Must have been computed "
                             "on THIS checkpoint's geometry -- the point count "
                             "is checked, the alignment is not, and a mask "
                             "from a different alignment is silently wrong.")
    parser.add_argument("--absolute-gain", default=None,
                        help="the same --absolute-gain-<radar> value for every radar (auto | init | person | a number); "
                             "the radar-agnostic spelling a candidate's flags use on both datasets")
    for radar in dataset.radars:
        parser.add_argument(radar.flag("absolute-gain"), default="auto",
                            help="the radar's absolute receiver gain: a number, person (the person stage's calibrated "
                                 "constant, frozen), init (set from the first frames' "
                                 "measured/rendered level, then frozen: one calibration constant), or auto (default) -- "
                                 "learned, one log-gain per radar, initialised from the first frames' "
                                 "measured/rendered RA level; the trained value is reported and rendered with")
    parser.add_argument(
        "--velocity-differentiator",
        choices=("savgol", "central", "fourth-order", "band-limited"),
        default="savgol",
    )
    parser.add_argument("--skinning-temperature-m", type=float, default=0.25)
    parser.add_argument("--skinning-topk", type=int, default=3)
    parser.add_argument("--loop-chunk-size", type=int, default=64)
    parser.add_argument(
        "--frame-batch", type=int, default=24,
        help="frames fused into one slow-time scatter per optimizer step. "
             "24 is the measured sweet spot on a 24 GB Ada card: 500 epochs "
             "in 9.8 min at 1.17 s/epoch, and on seq 102 it also trained "
             "BETTER than the historical 6 over 500 epochs (best held-out "
             "0.904 vs 1.132; temporal deltas H +1.94/V +2.13 vs "
             "+1.04/+1.11). Above 24 the K2b backward's scattered gradient "
             "working set (planes x base x 8 B) leaves L2 and the step "
             "regresses (fb 30: 42 -> 76 ms). Gate any change with "
             "cross-batch identity; a "
             "candidate must be a multiple of the reference batch)",
    )
    parser.add_argument("--geometry-translation-y-m", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    print(json.dumps(train(make_parser().parse_args()), indent=2))
