"""The differentiable forward render: the fused CUDA path and the training views.

Two modules merged here by `docs/CONSOLIDATION_AND_OO_PLAN.md` C6:
`cuda/cuda_render.py` and `forward/renderer_training.py`. No name collisions.

    cuda_render         the fused kernels, the only assembly path
    renderer_training   profiles -> RA/Doppler views, and the losses on them

`_ext()` now delegates to `mm4DPS.cuda_ext.load_hupr_cuda`, so both extensions
are located the same way and neither CUDA directory needs to be a package.
"""

from __future__ import annotations

import os

import json
import math
import numpy as np
import torch

from mm4DPS import Dataset, Radar, current_dataset
from mm4DPS.signal_model import (
    CHIRP_DURATION_S, SLOW_TIME_PHASE_SIGN, TDM_LOOPS, WAVELENGTH_M, apply_fft_window,
    doppler_dft_rows, doppler_plane_weights, fft_window, fft_window_kind, tdm_sample_times, window_cache_key)


# --------------------------------------------------------------------
# the fused CUDA render path
# --------------------------------------------------------------------



def _ext():
    """The hupr_cuda extension, loaded once, or a RuntimeError naming the build.

    The loading, the cache and the error message moved to `mm4DPS.cuda_ext` in
    C6, so that both extensions are found the same way -- by file path out of
    their own source directory, which is what let mm4DPS/cuda stop being a
    package. Behaviour is unchanged: still raised at CALL time, still names the
    command.
    """
    from mm4DPS.cuda_ext import load_hupr_cuda

    return load_hupr_cuda()


def is_available() -> bool:
    """Whether the HuPR kernels are loadable. Never swallows the reason."""
    try:
        _ext()
    except RuntimeError:
        return False
    return True


_TIMES_CACHE: dict = {}


def _timing(loops, chirp_duration_s):
    """Fill the chirp-timing defaults from the current dataset's radar."""
    radar = current_dataset().radars[0]
    return (radar.loops if loops is None else loops,
            radar.chirp_duration_s if chirp_duration_s is None else chirp_duration_s)


def _tdm_times(loops: int, chirp_duration_s: float, device) -> torch.Tensor:
    """TDM sample times on the device, built once per (loops, duration).

    Runtime: this is a fixed capture constant, but rebuilding it with
    ``torch.as_tensor(numpy_array, device='cuda')`` inside K2 was a blocking
    host-to-device copy on every call — one CPU<->GPU round trip per frame per
    radar, 12 per training step and 1,500 per evaluation pass.
    """
    key = (int(loops), float(chirp_duration_s), str(device))
    cached = _TIMES_CACHE.get(key)
    if cached is None:
        cached = torch.as_tensor(
            tdm_sample_times(loops, chirp_duration_s),
            dtype=torch.float32, device=device,
        )
        _TIMES_CACHE[key] = cached
    return cached


_TX_TIMES64_CACHE: dict = {}


def _tx_times64(loops: int, chirp_duration_s: float, device) -> torch.Tensor:
    """``(n_tx, 2)`` float64: each TX's first sample time and its loop period, for the closed form.

    The closed form's fractional Doppler bin is ``rate * T * L / lambda``;
    with ``T`` from the float32 time table the bin carried ~2e-6 of
    rounding, which the amplitude's slope turns into ~6e-6 of the peak. A
    float64 table removes that; the float32 rate is the model's own.
    """
    key = (int(loops), float(chirp_duration_s), str(device))
    cached = _TX_TIMES64_CACHE.get(key)
    if cached is None:
        times = np.asarray(tdm_sample_times(loops, chirp_duration_s), dtype=np.float64)
        cached = torch.from_numpy(np.ascontiguousarray(
            np.stack([times[0], times[1] - times[0]], axis=1))).to(device)
        _TX_TIMES64_CACHE[key] = cached
    return cached


def _static_gain_map(splat_components: dict, base: int, n_points: int) -> dict:
    """E9b (Phase 5 W2): the scene's static scatter, sorted once.

    ``contrib``, ``flat_indices`` and ``point_indices`` describe the *static*
    background: the trainer builds them once at startup, detaches them and
    never rebuilds them, and only ``gains`` changes. So K1 is a fixed linear
    map ``out = S . gains`` and ``S`` can be sorted and prefix-summed once:

    * ``cell_offsets`` / ``slot_contrib`` / ``slot_point`` — the contributions
      permuted into destination-cell order, for a segmented reduction forward;
    * ``point_offsets`` / ``point_paths`` — each point's paths, contiguous, for
      a per-point reduction backward.

    Both replace ``atomicAdd``s with fixed-order sums, which is what makes
    ``background_gains`` deterministic.

    **The cache lives in the caller's own components dict**, not in a
    module-level table keyed by tensor identity. Phase 4 lost three epochs to
    exactly that distinction: a cache the CUDA graph had baked pointers into
    was dropped on a miss and the replay read freed memory
    (``train_hupr_velocity_field.py``'s ``index_cache`` note). Here the map's
    lifetime is the components dict's, and the trainer holds that for the whole
    run.
    """
    cached = splat_components.get("_static_gain_map")
    if cached is not None and cached["key"] == (base, n_points):
        return cached

    contrib_real = splat_components["contrib_real"].detach()
    contrib_imag = splat_components["contrib_imag"].detach()
    flat = splat_components["flat_indices"].long().reshape(-1)
    point = splat_components["point_indices"].long()
    spread, paths = contrib_real.shape
    device = contrib_real.device

    # Forward: pairs in destination-cell order. `stable=True` makes the slot
    # order a function of the data alone, so the sum is reproducible.
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=base)
    cell_offsets = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)
    ]).to(torch.int32).contiguous()
    slot_contrib = torch.stack([
        contrib_real.reshape(-1).index_select(0, order),
        contrib_imag.reshape(-1).index_select(0, order),
    ], dim=-1).contiguous()
    slot_point = point.index_select(0, order % paths).to(torch.int32).contiguous()

    # Backward: each point's paths, contiguous.
    path_order = torch.argsort(point, stable=True)
    point_counts = torch.bincount(point, minlength=n_points)
    point_offsets = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device), point_counts.cumsum(0)
    ]).to(torch.int32).contiguous()

    cached = {
        "key": (base, n_points),
        "cell_offsets": cell_offsets,
        "slot_contrib": slot_contrib,
        "slot_point": slot_point,
        "point_offsets": point_offsets,
        "point_paths": path_order.to(torch.int32).contiguous(),
        "flat_indices": splat_components["flat_indices"].int().contiguous(),
        "contrib_real": contrib_real.contiguous(),
        "contrib_imag": contrib_imag.contiguous(),
    }
    splat_components["_static_gain_map"] = cached
    return cached


class _StaticGainMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gains, mapping, base):
        ext = _ext()
        out_real, out_imag = ext.hupr_static_gain_map_forward(
            mapping["cell_offsets"], mapping["slot_contrib"],
            mapping["slot_point"], gains.contiguous(), int(base))
        ctx.mapping = mapping
        ctx.save_for_backward(gains)
        return out_real, out_imag

    @staticmethod
    def backward(ctx, grad_out_real, grad_out_imag):
        (gains,) = ctx.saved_tensors
        mapping = ctx.mapping
        ext = _ext()
        grad_gains, _, _ = ext.hupr_static_gain_map_backward(
            grad_out_real.contiguous(), grad_out_imag.contiguous(),
            mapping["contrib_real"], mapping["contrib_imag"],
            mapping["flat_indices"].int(), mapping["point_offsets"],
            mapping["point_paths"], gains.contiguous(), False)
        return grad_gains, None, None


def _static_gain_geometry(splat_components: dict, base: int, n_points: int,
                          cache: dict) -> dict:
    """R094: K1's sort structure, cached across STEPS rather than per call.

    `_static_gain_map` caches in the caller's components dict, which is right
    for a frozen scene: the dict lives for the whole run. The trainable
    background rebuilds its components every step, so that cache misses every
    step and the sort is rebuilt -- measured 1693 us, which is most of what
    routing the trainable path through the map was meant to save.

    But the sort depends on `flat_indices` and `point_indices` ALONE. Those are
    splat GEOMETRY, and the background's geometry is frozen and cached
    (`background_geometry_cache`), so the permutation is constant for the whole
    run while only the contributions move. This holds it in a cache the caller
    owns, keyed on the identity of the very `flat_indices` tensor it was built
    from -- and it keeps a reference to that tensor, so the identity cannot be
    a recycled address. A miss rebuilds; it degrades to the old cost, never to
    a wrong answer.
    """
    flat_source = splat_components["flat_indices"]
    cached = cache.get("geometry")
    if (cached is not None and cached["key"] == (base, n_points)
            and cached["flat_source"] is flat_source):
        return cached

    flat = flat_source.long().reshape(-1)
    point = splat_components["point_indices"].long()
    paths = flat_source.shape[1]
    device = flat.device

    # Same construction as `_static_gain_map`, minus the contributions:
    # `stable=True` makes the slot order a function of the data alone, so
    # every cell's sum is the same sum in the same order on every step.
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=base)
    cell_offsets = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)
    ]).to(torch.int32).contiguous()
    slot_point = point.index_select(0, order % paths).to(torch.int32).contiguous()
    path_order = torch.argsort(point, stable=True)
    point_counts = torch.bincount(point, minlength=n_points)
    point_offsets = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device), point_counts.cumsum(0)
    ]).to(torch.int32).contiguous()

    cached = {
        "key": (base, n_points),
        "flat_source": flat_source,
        "order": order,
        "cell_offsets": cell_offsets,
        "slot_point": slot_point,
        "point_offsets": point_offsets,
        "point_paths": path_order.to(torch.int32).contiguous(),
        "flat_indices": flat_source.int().contiguous(),
        "point_indices": point.contiguous(),
        "flat_long": flat.contiguous(),
        "shape": tuple(flat_source.shape),
    }
    cache["geometry"] = cached
    return cached


def _slot_contrib_for(splat_components: dict, geometry: dict) -> torch.Tensor:
    """The step's contributions permuted into cell order, gathered ONCE.

    Hoisted onto the components dict because a step assembles one background
    per frame off ONE render: 24 frames share these contributions, so gathering
    28 MB per frame would cost more than the map formulation saves. Detached on
    purpose -- the gradient into `contrib` is produced by the backward kernel
    in the original layout, so letting autograd also trace this gather would
    count it twice.
    """
    hit = splat_components.get("_slot_contrib")
    if hit is not None and hit["order"] is geometry["order"]:
        return hit["value"]
    order = geometry["order"]
    with torch.no_grad():
        value = torch.stack([
            splat_components["contrib_real"].detach().reshape(-1).index_select(0, order),
            splat_components["contrib_imag"].detach().reshape(-1).index_select(0, order),
        ], dim=-1).contiguous()
    splat_components["_slot_contrib"] = {"order": order, "value": value}
    return value


def _scene_frames_geometry(splat_components: dict, n_frac: torch.Tensor,
                           samples: int, cache: dict) -> dict:
    """The fused frames path's order: the scene's paths sorted by (channel, range bin).

    A cell (channel, bin) then owns one or two contiguous runs of that order
    (two where the PSF wraps at the range axis), and the cells of a channel
    walk overlapping runs. Built once per run from the splat geometry
    (`flat_indices`' centre tap gives each path's channel and integer bin),
    keyed like `_static_gain_geometry` on the identity of `flat_indices`.

    ``entry`` (paths, 4) int32: path, point, (channel << 16) | n_floor, the
    fractional bin's float bits. ``runs`` (base, 4) int32: begin1, end1,
    begin2, end2 into ``entry``.
    """
    flat_source = splat_components["flat_indices"]
    cached = cache.get("frames_geometry")
    if cached is not None and cached["flat_source"] is flat_source:
        return cached
    spread, paths = flat_source.shape
    half = spread // 2
    device = flat_source.device
    centre = flat_source[half].long()
    channel = centre // samples
    n_floor = centre % samples
    point = splat_components["point_indices"].long()
    key = channel * samples + n_floor
    order = torch.argsort(key, stable=True)
    sorted_key = key.index_select(0, order)
    entry = torch.stack([
        order, point.index_select(0, order),
        (channel.index_select(0, order) << 16) | n_floor.index_select(0, order),
        n_frac.index_select(0, order).view(torch.int32).long(),
    ], dim=-1).to(torch.int32).contiguous()
    n_channels = int(channel.max().item()) + 1 if paths else 0
    base = n_channels * samples
    cells = torch.arange(base, device=device)
    c, b = cells // samples, cells % samples
    lo1 = torch.searchsorted(sorted_key, c * samples + (b - half).clamp_min(0))
    hi1 = torch.searchsorted(sorted_key, c * samples + (b + half).clamp_max(samples - 1), right=True)
    wrap_low = b - half < 0
    wrap_high = b + half >= samples
    lo2 = torch.where(wrap_low, torch.searchsorted(sorted_key, c * samples + samples + b - half),
                      torch.where(wrap_high, torch.searchsorted(sorted_key, c * samples), lo1))
    hi2 = torch.where(wrap_low, torch.searchsorted(sorted_key, c * samples + samples - 1, right=True),
                      torch.where(wrap_high, torch.searchsorted(sorted_key, c * samples + b + half - samples, right=True), lo1))
    runs = torch.stack([lo1, hi1, lo2, hi2], dim=-1).to(torch.int32).contiguous()
    cached = {"flat_source": flat_source, "entry": entry, "runs": runs,
              "order32": order.to(torch.int32),
              "point_sorted32": point.index_select(0, order).to(torch.int32),
              "base": base, "samples": samples, "spread": spread}
    cache["frames_geometry"] = cached
    return cached


_PSF_TABLE4_CACHE: dict = {}


def _psf_table4(table_real: torch.Tensor, table_imag: torch.Tensor) -> torch.Tensor:
    """The Hann PSF table packed for the kernels' one-load lerp.

    ``(spread, n_grid, 4)`` float32 of (re[lo], im[lo], re[lo + 1], im[lo + 1])
    with the last column repeating itself (the lerp's clamp), so a lerp is
    one 16-byte load instead of four; the arithmetic on the values is the
    same, so the result is bitwise the two-table lerp's. Cached on the
    tables' identity (they live for the run).
    """
    key = (id(table_real), id(table_imag))
    hit = _PSF_TABLE4_CACHE.get(key)
    if hit is not None and hit[0] is table_real:
        return hit[2]
    hi = torch.cat([table_real[:, 1:], table_real[:, -1:]], dim=1)
    hi_i = torch.cat([table_imag[:, 1:], table_imag[:, -1:]], dim=1)
    packed = torch.stack([table_real, table_imag, hi, hi_i], dim=-1).contiguous()
    _PSF_TABLE4_CACHE[key] = (table_real, table_imag, packed)
    return packed


def _gather_rows(table: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``table.index_select(0, index)`` for a (rows, 4) float32 table, 26x faster.

    torch's 2-D index_select takes its slow small-inner-dimension path for
    four-float rows (1.46 ms for 2.9 M rows); the same rows viewed as one
    complex128 each go through the vectorised 1-D gather (0.055 ms).
    """
    if table.shape[1] != 4 or not table.is_contiguous():
        return table.index_select(0, index)
    return table.view(torch.complex128).squeeze(1).index_select(0, index)[:, None].view(torch.float32)


def sorted_doppler_order(n_floor: torch.Tensor, channel: torch.Tensor,
                         group: torch.Tensor, samples: int, groups: int
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """The sorted Doppler route's order for one set of paths.

    Paths sorted by (group, channel, range bin) -- a group is a unit, or a
    frame when the order is cached per frame -- so a block of
    `_SORTED_BINS` consecutive bins of one (group, channel) owns one or two
    contiguous runs of entries (two where the range PSF wraps): the paths
    whose 15-tap window touches any of its bins. Returns ``path_of_entry``
    (paths,) int32 and ``runs`` (groups * channels * samples / 32, 4) int32:
    begin1, end1, begin2, end2 into the entries.
    """
    channels = int(channel.max().item()) + 1 if channel.numel() else 0
    half = _RANGE_SPREAD // 2
    key = (group.long() * channels + channel.long()) * samples + n_floor.long()
    order = torch.argsort(key, stable=True)
    sorted_key = key.index_select(0, order)
    per_group = samples // _SORTED_BINS
    blocks = torch.arange(groups * channels * per_group, device=key.device)
    c = blocks // per_group
    b0 = (blocks % per_group) * _SORTED_BINS
    b1 = b0 + _SORTED_BINS - 1
    lo1 = torch.searchsorted(sorted_key, c * samples + (b0 - half).clamp_min(0))
    hi1 = torch.searchsorted(sorted_key, c * samples + (b1 + half).clamp_max(samples - 1), right=True)
    wrap_low = b0 - half < 0
    wrap_high = b1 + half >= samples
    lo2 = torch.where(wrap_low, torch.searchsorted(sorted_key, c * samples + samples + b0 - half),
                      torch.where(wrap_high, torch.searchsorted(sorted_key, c * samples), lo1))
    hi2 = torch.where(wrap_low, torch.searchsorted(sorted_key, c * samples + samples - 1, right=True),
                      torch.where(wrap_high, torch.searchsorted(sorted_key, c * samples + b1 + half - samples, right=True), lo1))
    runs = torch.stack([lo1, hi1, lo2, hi2], dim=-1).to(torch.int32).contiguous()
    return order.to(torch.int32).contiguous(), runs


_RANGE_SPREAD = 15   # the range PSF's taps (fixed; rasterizer SPREAD)
_SORTED_BINS = 32    # range bins per block of the sorted route's forward (kernels.cu SORTED_BINS)
_SCENE_MAX_FRAMES = 8  # frames per scene-frames kernel launch (kernels.cu SCENE_MAX_FRAMES)


class _DopplerPlaneSorted(torch.autograd.Function):
    """The sorted Doppler route: planes from the carrier and the path rates.

    Forward: the closed-form window amplitudes in entry order, then one
    kernel over the cells' runs synthesising each contribution (carrier x
    range PSF lerp) and applying the path's window to the planes. Backward:
    one thread per entry, the carrier and rate gradients. Fixed orders
    throughout; against the slot route it sits at the fp32 floor
    (`qa sorted-check`).
    """

    @staticmethod
    def forward(ctx, carrier, path_rates, record_static, path_of_entry, runs,
                tx_indices, unit_of_path, active_planes, bin_to_plane,
                windowed, anchor, tx_times, taps, table4,
                planes, unit_base, samples, spread, loops, phase_const,
                channels_per_point):
        ext = _ext()
        # the per-entry record: the step's carrier beside the cached
        # fractional and integer range bins, gathered into entry order
        record = _gather_rows(torch.cat([carrier, record_static], dim=1), path_of_entry)
        out_real, out_imag, amps_e = ext.hupr_doppler_plane_sorted_forward(
            runs, record, path_of_entry, tx_indices, path_rates, float(phase_const),
            unit_of_path, active_planes, bin_to_plane, int(windowed), int(anchor),
            tx_times, int(taps), table4, int(planes), int(unit_base),
            int(samples), int(spread), int(loops))
        ctx.save_for_backward(record, amps_e, path_of_entry, tx_indices, path_rates,
                              unit_of_path, active_planes, bin_to_plane, tx_times, table4)
        ctx.scalars = (int(planes), int(unit_base), int(samples), int(spread), int(loops),
                       int(taps), int(windowed), int(anchor), float(phase_const),
                       int(channels_per_point))
        return out_real, out_imag

    @staticmethod
    def backward(ctx, grad_out_real, grad_out_imag):
        (record, amps_e, path_of_entry, tx_indices, path_rates, unit_of_path,
         active_planes, bin_to_plane, tx_times, table4) = ctx.saved_tensors
        (planes, unit_base, samples, spread, loops, taps, windowed, anchor, phase_const,
         channels_per_point) = ctx.scalars
        grad_carrier, grad_rates = _ext().hupr_doppler_plane_sorted_backward(
            grad_out_real, grad_out_imag, path_of_entry, record, amps_e, tx_indices,
            path_rates, unit_of_path, active_planes, bin_to_plane, table4,
            phase_const, planes, unit_base, channels_per_point, samples, spread, loops,
            taps, windowed, anchor, tx_times)
        return (grad_carrier, grad_rates) + (None,) * 19


class _StaticSceneFrames(torch.autograd.Function):
    """K1 for a batch of frames from the carrier, contributions never built.

    `hupr_static_scene_frames_forward` walks each cell's paths in the
    (channel, range bin) order of `_scene_frames_geometry`, synthesising
    every contribution (carrier x the PSF lerp, the same expression
    `_psf_lerp_eager` / `_splat_contribs_eager` fuse) and applying every
    frame's gain to it at once, in a fixed order; the backward writes the
    carrier gradient with one thread per path. The gains are the frozen
    areas times the frame's visibility and carry no gradient here.
    """

    @staticmethod
    def forward(ctx, carrier, gains_t, table4, geometry):
        # the carrier and each path's point gains gathered into entry order,
        # so the forward's walk of its runs is coalesced end to end
        out = _ext().hupr_static_scene_frames_forward(
            geometry["runs"], geometry["entry"],
            carrier.index_select(0, geometry["order32"]),
            table4, _gather_rows(gains_t, geometry["point_sorted32"]),
            int(geometry["base"]), int(geometry["samples"]))
        ctx.save_for_backward(gains_t, table4)
        ctx.geometry = geometry
        return out

    @staticmethod
    def backward(ctx, grad_out):
        gains_t, table4 = ctx.saved_tensors
        # (frames, base, 2) -> (base, frames, 2): a cell's frames in one sector
        grad_out_t = grad_out.transpose(0, 1).contiguous()
        grad_carrier = _ext().hupr_static_scene_frames_backward(
            grad_out_t, ctx.geometry["entry"], table4, gains_t,
            int(ctx.geometry["samples"]))
        return grad_carrier, None, None, None


def assemble_static_profile_frames_cuda(splat_components: dict,
                                        frame_gains: torch.Tensor,
                                        map_cache: dict) -> torch.Tensor:
    """K1 for a batch of frames at once, ``(frames, tx, rx, samples)`` complex.

    Takes the carrier-only components (`render_factorized(carrier_only=True)`
    on a filled geometry cache) and the ``(frames, points)`` gains; one
    forward kernel and one backward kernel for the whole batch, against the
    per-frame `assemble_static_profile_cuda` chain (contributions assembled,
    permuted, summed and differentiated per frame). Fixed-order sums, so it
    reproduces itself bitwise; against the per-frame chain it sits at the
    fp32 floor (the same products, fused differently; `qa scene-check`).
    """
    tx, rx, samples = splat_components["output_shape"]
    base = tx * rx * samples
    if frame_gains.requires_grad:
        raise ValueError("assemble_static_profile_frames_cuda: the gains carry no gradient")
    geometry = _scene_frames_geometry(
        splat_components, splat_components["n_frac"], samples, map_cache)
    if geometry["base"] != base:
        raise ValueError(f"scene frames geometry covers {geometry['base']} cells, the profile has {base}")
    table4 = _psf_table4(splat_components["psf_table_real"], splat_components["psf_table_imag"])
    # the kernels hold at most `_SCENE_MAX_FRAMES` frames' accumulators in
    # registers; a larger batch (HuPR's 24) goes through in slices
    outs = [
        _StaticSceneFrames.apply(
            splat_components["carrier"], frame_gains[start:start + _SCENE_MAX_FRAMES].t().contiguous(),
            table4, geometry)
        for start in range(0, frame_gains.shape[0], _SCENE_MAX_FRAMES)]
    out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
    return torch.view_as_complex(out).reshape(frame_gains.shape[0], tx, rx, samples)


class _StaticGainMapTrainable(torch.autograd.Function):
    """K1 through the fixed sparse map, WITH gradients into the contributions.

    `_StaticGainMap` refuses this case and falls back to the atomic scatter,
    which is 6x slower (measured 246.9 us vs 41.4 us at the trainer's shapes:
    3.49M atomicAdds into 3072 cells is contention, not bandwidth). Nothing
    about the map forbids a contribution gradient -- the backward kernel has
    written it all along behind `need_contrib`, which no caller had ever set,
    because until the scene became trainable nobody needed it.
    """

    @staticmethod
    def forward(ctx, contrib_real, contrib_imag, gains, slot_contrib, geometry,
                base):
        ext = _ext()
        out_real, out_imag = ext.hupr_static_gain_map_forward(
            geometry["cell_offsets"], slot_contrib, geometry["slot_point"],
            gains.contiguous(), int(base))
        ctx.save_for_backward(contrib_real, contrib_imag, gains)
        ctx.geometry = geometry
        return out_real, out_imag

    @staticmethod
    def backward(ctx, grad_out_real, grad_out_imag):
        """Fixed-order sums where the order MATTERS, fast kernels where it does not.

        The two gradients are not alike. `grad_gains` is a REDUCTION over each
        point's paths, so its summation order decides whether it reproduces
        itself -- that is the whole reason E9b built the map's per-point
        fixed-order sum. `grad_contrib` is a plain elementwise product,
        `gains[point] * grad_out[cell]`: no reduction, no order, and both
        kernels return it bitwise equal to the torch reference (measured
        0.000e+00 for real and imaginary parts alike).

        So the gains gradient, when anything asks for it, comes from the map's
        ordered reduction. When nothing asks for it -- which is every call the
        trainable background makes, because there the gains are the frozen
        areas rather than a parameter -- the splat backward computes the
        contribution gradient with one thread per contribution instead of one
        warp per point: 28.3 us against the map backward's 307.5, measured at
        the trainer's shapes. Nothing nondeterministic survives, because the
        only order-dependent output is the one this branch discards.
        """
        contrib_real, contrib_imag, gains = ctx.saved_tensors
        geometry = ctx.geometry
        need_contrib = bool(ctx.needs_input_grad[0] or ctx.needs_input_grad[1])
        need_gains = bool(ctx.needs_input_grad[2])
        ext = _ext()
        if need_gains:
            grad_gains, grad_c_real, grad_c_imag = ext.hupr_static_gain_map_backward(
                grad_out_real.contiguous(), grad_out_imag.contiguous(),
                contrib_real.detach().contiguous(),
                contrib_imag.detach().contiguous(),
                geometry["flat_indices"], geometry["point_offsets"],
                geometry["point_paths"], gains.contiguous(), need_contrib)
        elif need_contrib:
            # The splat backward, with the gains gradient switched OFF. That
            # gradient is the only reason the kernel reads `contrib` (28 MB)
            # and the only reason it ends in an atomic, and nothing here wants
            # it: under a trainable background the gains are the frozen areas,
            # not a parameter. `grad_contrib` is written by the same statements
            # either way, so the flag cannot change it by a bit -- which is
            # what makes this a free saving rather than a trade.
            _, grad_c_real, grad_c_imag = ext.hupr_static_gain_splat_backward(
                grad_out_real.contiguous(), grad_out_imag.contiguous(),
                contrib_real.detach().contiguous(),
                contrib_imag.detach().contiguous(),
                geometry["flat_indices"], geometry["point_indices"],
                gains.contiguous(), False)
            grad_gains = None
        else:
            grad_gains = grad_c_real = grad_c_imag = None
        if not need_contrib:
            grad_c_real = grad_c_imag = None
        return grad_c_real, grad_c_imag, grad_gains, None, None, None


class _StaticGainSplat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, contrib_real, contrib_imag, flat_indices, point_indices,
                gains, base):
        ext = _ext()
        out_real, out_imag = ext.hupr_static_gain_splat_forward(
            contrib_real.contiguous(), contrib_imag.contiguous(),
            flat_indices.contiguous().int(), point_indices.contiguous(),
            gains.contiguous(), int(base))
        ctx.save_for_backward(contrib_real, contrib_imag, flat_indices,
                              point_indices, gains)
        return out_real, out_imag

    @staticmethod
    def backward(ctx, grad_out_real, grad_out_imag):
        contrib_real, contrib_imag, flat_indices, point_indices, gains = (
            ctx.saved_tensors)
        ext = _ext()
        grad_gains, grad_c_real, grad_c_imag = ext.hupr_static_gain_splat_backward(
            grad_out_real, grad_out_imag,
            contrib_real.contiguous(), contrib_imag.contiguous(),
            flat_indices.contiguous().int(), point_indices.contiguous(),
            gains.contiguous(), True)
        return grad_c_real, grad_c_imag, None, None, grad_gains, None


def assemble_static_profile_cuda(splat_components, point_gains,
                                 map_cache=None) -> torch.Tensor:
    """K1: drop-in for hupr_chirp_motion.assemble_static_profile.

    E9b routes this through the fixed sparse map whenever the contributions are
    constants — which is every call the trainer makes, because the scene is
    static. A caller that wants gradients *into* the contributions still gets
    the scatter, which is the general case the kernel tests pin.
    """
    tx, rx, samples = splat_components["output_shape"]
    base = tx * rx * samples
    if map_cache is not None:
        # R094: the map formulation for a caller whose splat geometry is fixed
        # across steps but whose contributions are not -- i.e. the trainable
        # background. Atomic-free and, unlike the scatter, bitwise reproducible
        # against itself.
        geometry = _static_gain_geometry(
            splat_components, base, point_gains.shape[0], map_cache)
        out_real, out_imag = _StaticGainMapTrainable.apply(
            splat_components["contrib_real"], splat_components["contrib_imag"],
            point_gains, _slot_contrib_for(splat_components, geometry),
            geometry, base)
        return torch.complex(out_real, out_imag).reshape(tx, rx, samples)
    if (not splat_components["contrib_real"].requires_grad
            and not splat_components["contrib_imag"].requires_grad):
        mapping = _static_gain_map(splat_components, base, point_gains.shape[0])
        out_real, out_imag = _StaticGainMap.apply(point_gains, mapping, base)
        return torch.complex(out_real, out_imag).reshape(tx, rx, samples)
    out_real, out_imag = _StaticGainSplat.apply(
        splat_components["contrib_real"], splat_components["contrib_imag"],
        splat_components["flat_indices"].int(),
        splat_components["point_indices"].long(),
        point_gains, base)
    return torch.complex(out_real, out_imag).reshape(tx, rx, samples)


class _SlowTimeSplat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, contrib_real, contrib_imag, flat_indices, tx_indices,
                path_rates, times, phase_const, base):
        ext = _ext()
        out_real, out_imag = ext.hupr_slow_time_splat_forward(
            contrib_real.contiguous(), contrib_imag.contiguous(),
            flat_indices.contiguous().int(), tx_indices.contiguous(),
            path_rates.contiguous(), times.contiguous(),
            float(phase_const), int(base))
        ctx.save_for_backward(contrib_real, contrib_imag, flat_indices,
                              tx_indices, path_rates, times)
        ctx.phase_const = float(phase_const)
        ctx.base = int(base)
        return out_real, out_imag

    @staticmethod
    def backward(ctx, grad_out_real, grad_out_imag):
        (contrib_real, contrib_imag, flat_indices, tx_indices, path_rates,
         times) = ctx.saved_tensors
        ext = _ext()
        grad_c_real, grad_c_imag, grad_rates = ext.hupr_slow_time_splat_backward(
            grad_out_real, grad_out_imag,
            contrib_real.contiguous(), contrib_imag.contiguous(),
            flat_indices.contiguous().int(), tx_indices.contiguous(),
            path_rates.contiguous(), times.contiguous(),
            ctx.phase_const, ctx.base)
        return grad_c_real, grad_c_imag, None, None, grad_rates, None, None, None


def assemble_surface_slow_time_profiles_cuda(
    background_profile: torch.Tensor,
    splat_components,
    point_path_rate_mps: torch.Tensor,
    point_scattering_gain: torch.Tensor | None = None,
    loops: int | None = None,
    chirp_duration_s: float | None = None,
    wavelength_m: float = WAVELENGTH_M,
    loop_chunk_size: int = 64,  # accepted for signature parity; unused
) -> torch.Tensor:
    """K2: drop-in for hupr_chirp_motion.assemble_surface_slow_time_profiles."""
    loops, chirp_duration_s = _timing(loops, chirp_duration_s)
    output_shape = tuple(splat_components["output_shape"])
    if output_shape != tuple(background_profile.shape):
        raise ValueError("splat and background output shapes must match")
    real = splat_components["contrib_real"]
    imag = splat_components["contrib_imag"]
    paths = real.shape[1]
    if paths == 0:
        return background_profile[None].expand(loops, -1, -1, -1).clone()
    point_indices = splat_components["point_indices"].long()
    tx_indices = splat_components["tx_indices"].long()
    rx_indices = splat_components["rx_indices"].long()
    if point_scattering_gain is not None:
        scale = point_scattering_gain[point_indices][None, :]
        real = real * scale
        imag = imag * scale
    path_rates = point_path_rate_mps[point_indices, tx_indices, rx_indices]
    times = _tdm_times(loops, chirp_duration_s, real.device)
    base = int(np.prod(output_shape))
    phase_const = SLOW_TIME_PHASE_SIGN * 2.0 * math.pi / wavelength_m
    out_real, out_imag = _SlowTimeSplat.apply(
        real, imag, splat_components["flat_indices"].int(), tx_indices,
        path_rates.float(), times, phase_const, base)
    dynamic = torch.complex(out_real, out_imag).reshape(loops, *output_shape)
    return background_profile[None] + dynamic


def assemble_surface_slow_time_profiles_batched_cuda(
    background_profile: torch.Tensor,
    splat_components_list,
    point_path_rates_list,
    loops: int | None = None,
    chirp_duration_s: float | None = None,
    wavelength_m: float = WAVELENGTH_M,
) -> torch.Tensor:
    """Batched K2: per-frame kernel launches (launch cost is negligible)."""
    loops, chirp_duration_s = _timing(loops, chirp_duration_s)
    frames = [
        assemble_surface_slow_time_profiles_cuda(
            background_profile, components, rates,
            loops=loops, chirp_duration_s=chirp_duration_s,
            wavelength_m=wavelength_m,
        )
        for components, rates in zip(splat_components_list, point_path_rates_list)
    ]
    return torch.stack(frames, dim=0)


def assemble_surface_slow_time_profiles_units_cuda(
    background_profiles: torch.Tensor,
    splat_components,
    point_path_rate_mps: torch.Tensor,
    unit_of_path: torch.Tensor,
    units: int,
    loops: int | None = None,
    chirp_duration_s: float | None = None,
    wavelength_m: float = WAVELENGTH_M,
) -> torch.Tensor:
    """K2 over a whole batch of (frame, radar) units in ONE kernel launch.

    ``splat_components`` comes from a single ``render_factorized`` call over
    every unit's points concatenated, so ``point_indices`` index the stacked
    cloud and ``unit_of_path`` says which unit each path belongs to.

    The kernel is unchanged. K2 already writes ``out[l * base + flat]``; giving
    it ``base' = units * base`` and pre-offsetting each path's flat index by
    ``unit * base`` makes the same launch write a ``(loops, units, tx, rx, K)``
    cube instead of ``(loops, tx, rx, K)``. One launch replaces twelve, the
    arithmetic per path is untouched, and the backward indexes the enlarged
    output with the same expression it always did.

    Returns ``(units, loops, n_tx, n_rx, K)``.
    """
    loops, chirp_duration_s = _timing(loops, chirp_duration_s)
    output_shape = tuple(splat_components["output_shape"])
    if background_profiles.shape != (units, *output_shape):
        raise ValueError(
            f"background profiles must be (units, {output_shape}), got "
            f"{tuple(background_profiles.shape)}")
    real = splat_components["contrib_real"]
    imag = splat_components["contrib_imag"]
    if real.shape[1] == 0:
        return background_profiles[:, None].expand(-1, loops, -1, -1, -1).clone()
    point_indices = splat_components["point_indices"].long()
    tx_indices = splat_components["tx_indices"].long()
    rx_indices = splat_components["rx_indices"].long()
    path_rates = point_path_rate_mps[point_indices, tx_indices, rx_indices]
    times = _tdm_times(loops, chirp_duration_s, real.device)
    base = int(np.prod(output_shape))
    phase_const = SLOW_TIME_PHASE_SIGN * 2.0 * math.pi / wavelength_m
    # Follow-on C: int32. The destination is at most units*base -- 92,160 for
    # an evaluation chunk -- and this tensor is 4.8-11.0 M entries that the
    # binning reads twice and the backward once, so the second four bytes were
    # 115 MB of pure bandwidth per training step.
    flat_indices = (splat_components["flat_indices"].int()
                    + unit_of_path.int()[None, :] * int(base))
    out_real, out_imag = _SlowTimeSplat.apply(
        real, imag, flat_indices, tx_indices,
        path_rates.float(), times, phase_const, units * base)
    dynamic = torch.complex(out_real, out_imag).reshape(
        loops, units, *output_shape).transpose(0, 1)
    return background_profiles[:, None] + dynamic


# W4 (Phase 5): when on, K2b's binning uses the ordered placement instead of
# the atomicAdd one, so the gather sums each cell's contributions in pair-index
# order and the whole step reproduces itself bit for bit. Measured cost:
# +0.44 ms per training step and +1.75 ms per evaluation chunk
# (docs/RUNTIME_OPTIMIZATION_REPORT.md, Phase 5 W4). Off by default.
_DETERMINISTIC = False


def set_deterministic(flag: bool) -> None:
    """Select the ordered (reproducible) binning for K2b's forward."""
    global _DETERMINISTIC
    _DETERMINISTIC = bool(flag)


_WEIGHT_CACHE: dict = {}


def _plane_weights(bins: tuple, loops: int, device,
                   static_anchor: str = "loop0") -> tuple:
    """(planes, loops) real/imag weight matrices, built once per bin set.

    Cached for the same reason ``_tdm_times`` is: after E1 the training step is
    dispatch-bound (32.6 ms wall against 9.2 ms of GPU), so rebuilding a
    constant per call is pure added latency on the critical path.
    """
    key = (bins, int(loops), str(device), static_anchor, fft_window_kind())
    cached = _WEIGHT_CACHE.get(key)
    if cached is None:
        weights = doppler_plane_weights(
            bins, loops=loops, static_anchor=static_anchor).to(device)
        cached = (weights.real.contiguous(), weights.imag.contiguous())
        _WEIGHT_CACHE[key] = cached
    return cached


_DOPPLER_ROW_CACHE: dict = {}


def _doppler_row_table(loops: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """``(loops, loops)`` float32 real/imag Doppler rows for every unshifted frequency, on ``device``.

    Row ``f`` is ``signal_model.doppler_dft_rows([f])`` -- the same numpy
    values `doppler_plane_weights` casts, so a row gathered from here is
    bitwise that function's row. Built once per (loops, device, window).
    """
    key = (int(loops), str(device), fft_window_kind())
    if key not in _DOPPLER_ROW_CACHE:
        rows = doppler_dft_rows(np.arange(loops), loops).astype(np.complex64)
        _DOPPLER_ROW_CACHE[key] = (
            torch.from_numpy(np.ascontiguousarray(rows.real)).to(device),
            torch.from_numpy(np.ascontiguousarray(rows.imag)).to(device))
    return _DOPPLER_ROW_CACHE[key]


def _per_unit_plane_weights(bins: torch.Tensor, loops: int, device,
                            static_anchor: str = "loop0") -> tuple:
    """``(units, 1 + planes, loops)`` real/imag weights, one row set per unit.

    The map ``doppler_plane_weights`` builds, vectorised over units, with a
    pad marker: a bin of ``-1`` yields an all-zero W row, so that plane renders
    exactly zero and contributes nothing to either side of the loss -- which is
    what lets frames with different support widths share one static shape, and
    so one CUDA graph.

    Deliberately free of host synchronisation: no ``.cpu()``, no ``bool()``, no
    ``.tolist()``. This runs inside the captured region, where any sync is a
    capture failure. The bins are validated once where they are built.

    Row 0 is the static anchor, identical for every unit. Rows 1.. are
    gathered from ``_doppler_row_table`` -- ``doppler_dft_rows`` for every
    frequency, the numpy values ``doppler_plane_weights`` casts -- so a unit's
    row is bitwise that function's row for the same bin; a pad row is zero.
    """
    dc = loops // 2
    units, planes = bins.shape
    keep = (bins >= 0)[:, :, None]
    unshifted = ((bins.clamp_min(0) + dc) % loops).reshape(-1)
    table_real, table_imag = _doppler_row_table(loops, bins.device)
    zero = torch.zeros((), dtype=table_real.dtype, device=table_real.device)
    real = torch.where(keep, table_real.index_select(0, unshifted).reshape(units, planes, loops), zero)
    imag = torch.where(keep, table_imag.index_select(0, unshifted).reshape(units, planes, loops), zero)
    head_real = torch.zeros(units, 1, loops, dtype=real.dtype, device=real.device)
    head_imag = torch.zeros_like(head_real)
    if static_anchor == "dc-mean":
        head_real = head_real + 1.0 / float(loops)
    elif static_anchor == "loop0":
        head_real[:, :, 0] = 1.0
    else:
        raise ValueError(f"unknown static anchor: {static_anchor}")
    return (torch.cat([head_real, real], dim=1).to(torch.float32).contiguous(),
            torch.cat([head_imag, imag], dim=1).to(torch.float32).contiguous())


def validate_per_unit_bins(bins: torch.Tensor, loops: int) -> None:
    """Check a per-unit bin table once, where a host sync is free.

    ``_per_unit_plane_weights`` cannot do this: it runs under graph capture.
    """
    if bins.dim() != 2:
        raise ValueError(f"per-unit bins must be 2-D, got {tuple(bins.shape)}")
    valid = bins[bins >= 0]
    if valid.numel() == 0:
        raise ValueError("per-unit bins select no Doppler planes at all")
    if int(valid.max()) >= loops:
        raise ValueError(f"Doppler bins must lie in [0, {loops})")
    if bool((valid == loops // 2).any()):
        raise ValueError(
            f"Doppler bin {loops // 2} is DC; the rendered planes are the non-DC bins, so it "
            f"cannot be rendered consistently with the reference")
    for row in range(bins.shape[0]):
        row_bins = bins[row][bins[row] >= 0]
        if len(torch.unique(row_bins)) != row_bins.numel():
            raise ValueError(f"unit {row} repeats a Doppler bin")


# How the per-path Doppler-plane amplitudes are computed: "dft" sums the
# path's phasor over the 64 loops against the plane weights (the historical
# kernels); "closed-form" evaluates the Hann Doppler PSF at the path's
# fractional bin (doppler_closed_form.cuh) -- the same quantity to fp32
# rounding, without the loop. `set_doppler_amplitudes` selects it; the
# trainer's --doppler-amplitudes flag is the caller.
_DOPPLER_AMPLITUDES = "dft"


def set_doppler_amplitudes(mode: str) -> None:
    global _DOPPLER_AMPLITUDES
    if mode not in ("dft", "closed-form"):
        raise ValueError(f"doppler amplitudes must be 'dft' or 'closed-form', got {mode!r}")
    _DOPPLER_AMPLITUDES = mode


def doppler_amplitudes_mode() -> str:
    return _DOPPLER_AMPLITUDES


# The Doppler PSF's taps: with the Hann window a path's response is 1e-3 of
# its peak at 7 bins, so (as the range PSF is cut at 15 bins) a path
# deposits into the 15 bins around its own Doppler bin only; 0 = every
# supported plane (exact). Closed form only.
_DOPPLER_TAPS = 15


def set_doppler_taps(taps: int) -> None:
    global _DOPPLER_TAPS
    if taps not in (0, 3, 5, 7, 9, 11, 13, 15):
        raise ValueError(f"doppler taps must be 0 (every plane) or an odd count up to 15, got {taps}")
    _DOPPLER_TAPS = int(taps)


def doppler_taps() -> int:
    return _DOPPLER_TAPS


class _DopplerPlaneSplat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, contrib_real, contrib_imag, flat_indices, tx_indices,
                path_rates, times, weight_real, weight_imag, phase_const, base,
                unit_of_path=None, active_planes=None, plane_bins=None,
                windowed=0, anchor=0, tx_times=None, bin_to_plane=None, taps=0):
        ext = _ext()
        # E9a (Phase 5 W1): the gather. Same arithmetic, one block per output
        # cell instead of 2*spread*planes atomicAdds per path -- 134 M atomics
        # a training step become none at all. Measured 3.13x on the step's
        # shapes and 2.42x on an evaluation chunk's.
        out_real, out_imag, amps, _n_of_path = ext.hupr_doppler_plane_gather_forward(
            contrib_real.contiguous(), contrib_imag.contiguous(),
            flat_indices.contiguous().int(), tx_indices.contiguous(),
            path_rates.contiguous(), times.contiguous(),
            weight_real.contiguous(), weight_imag.contiguous(),
            float(phase_const), int(base), int(_DETERMINISTIC),
            unit_of_path, active_planes, plane_bins, int(windowed), int(anchor), tx_times,
            bin_to_plane, int(taps))
        # `amps` is the forward's (paths, planes) plane-amplitude buffer,
        # saved so the backward reads it instead of rebuilding it chunk by
        # chunk -- the same values bitwise (same accumulation order over the
        # loop axis), minus half the backward's sincos and FMA work.
        ctx.save_for_backward(contrib_real, contrib_imag, flat_indices,
                              tx_indices, path_rates, times,
                              weight_real, weight_imag, amps)
        ctx.unit_of_path = unit_of_path
        ctx.active_planes = active_planes
        ctx.plane_bins = plane_bins
        ctx.windowed = int(windowed)
        ctx.anchor = int(anchor)
        ctx.tx_times = tx_times
        ctx.bin_to_plane = bin_to_plane
        ctx.taps = int(taps)
        ctx.phase_const = float(phase_const)
        ctx.base = int(base)
        return out_real, out_imag

    @staticmethod
    def backward(ctx, grad_out_real, grad_out_imag):
        (contrib_real, contrib_imag, flat_indices, tx_indices, path_rates,
         times, weight_real, weight_imag, amps) = ctx.saved_tensors
        ext = _ext()
        grad_c_real, grad_c_imag, grad_rates = (
            ext.hupr_doppler_plane_splat_backward(
                grad_out_real, grad_out_imag,
                contrib_real.contiguous(), contrib_imag.contiguous(),
                flat_indices.contiguous().int(), tx_indices.contiguous(),
                path_rates.contiguous(), times.contiguous(),
                weight_real.contiguous(), weight_imag.contiguous(),
                ctx.phase_const, ctx.base, forward_amps=amps,
                unit_of_path=ctx.unit_of_path,
                active_planes=ctx.active_planes, plane_bins=ctx.plane_bins,
                windowed=ctx.windowed, anchor=ctx.anchor, tx_times=ctx.tx_times,
                bin_to_plane=ctx.bin_to_plane, taps=ctx.taps))
        return (grad_c_real, grad_c_imag, None, None, grad_rates,
                None, None, None, None, None, None, None, None, None, None, None, None, None)


def _path_rates_of(point_path_rate_mps: torch.Tensor, splat_components: dict, point_indices: torch.Tensor,
                   tx_indices: torch.Tensor, rx_indices: torch.Tensor) -> torch.Tensor:
    """The per-path rates ``rates[point, tx, rx]`` -- a view when the paths ARE the rows.

    The trainer's cached geometry enumerates every (point, tx, rx) of the
    padded batch in row-major order (`row_geometry_shapes`, which says so
    with ``paths_are_rows``), so the gather is the flattened tensor itself.
    The advanced index costs nothing forward but its backward is torch's
    sort-based deterministic ``index_put_``: 0.82 ms a step on RT-Pose 183
    (2.5 M paths), 5.8% of the step, to place each path's gradient once
    into zeros. The view's backward is the same placement (0 + g = g) with
    no kernel at all (docs/RUNTIME_RESULTS.md item 7). Any other index
    (the uncached path's culled, active subsets) takes the gather.
    """
    if splat_components.get("paths_are_rows") and point_path_rate_mps.numel() == point_indices.numel():
        return point_path_rate_mps.reshape(-1)
    return point_path_rate_mps[point_indices, tx_indices, rx_indices]


def assemble_surface_doppler_planes_units_cuda(
    background_profiles: torch.Tensor,
    splat_components,
    point_path_rate_mps: torch.Tensor,
    unit_of_path: torch.Tensor,
    units: int,
    doppler_bins,
    loops: int | None = None,
    chirp_duration_s: float | None = None,
    wavelength_m: float = WAVELENGTH_M,
    static_anchor: str = "loop0",
) -> tuple[torch.Tensor, torch.Tensor]:
    """E1: render the static-anchor plane and the requested Doppler bins.

    ``static_anchor="loop0"`` (default) keeps the historical raw loop-0
    sample as plane 0; ``"dc-mean"`` makes plane 0 the slow-time mean (the
    DC Doppler plane) — same static content at ~8x the SNR. The background
    is added to plane 0 either way, exactly: it is constant over the loop
    axis, so its loop-0 sample and its slow-time mean are the same value.

    Assembles the per-unit Doppler planes. Its torch reference was removed
    in the deployment strip along with the backend gates.
    Of the 64 loop planes the old kernel built, the objective reads loop 0's
    magnitude map and 13 Doppler bins, so the linear loop->Doppler map is
    folded into the splat and only ``1 + len(doppler_bins)`` planes are ever
    written: 4.6x fewer atomicAdds, nothing truncated, and the
    ``(units, 64, 3, 4, K)`` cube and the Python-side Doppler FFT both go away.

    The units batching is unchanged: ``base' = units * base`` with each path's
    flat index pre-offset by ``unit * base``, so one launch still covers the
    whole batch.

    Returns ``(ra_profile, doppler_planes)``:

    * ``ra_profile``     ``(units, 3, 4, K)``            — loop 0 plus the background;
    * ``doppler_planes`` ``(units, len(bins), 3, 4, K)`` — the requested bins.

    The background is added to the RA plane only. It is constant over the loop
    axis, and every plane's weights carry the consumer's mean removal
    (``doppler_dft_rows``), so it would render as exactly zero in every
    Doppler plane.
    """
    loops, chirp_duration_s = _timing(loops, chirp_duration_s)
    output_shape = tuple(splat_components["output_shape"])
    if background_profiles.shape != (units, *output_shape):
        raise ValueError(
            f"background profiles must be (units, {output_shape}), got "
            f"{tuple(background_profiles.shape)}")
    # A 2-D `doppler_bins` is (units, planes): every unit renders its own
    # Doppler rows, so a frame supervises only the bins its own measurement
    # puts above the noise floor. A zero row is a pad -- its W row is zero, so
    # that plane renders exactly zero and contributes nothing to the loss.
    per_unit_bins = (torch.is_tensor(doppler_bins) and doppler_bins.dim() == 2)
    active_planes = None
    plane_bins = None
    bin_to_plane = None
    if per_unit_bins:
        if doppler_bins.shape[0] != units:
            raise ValueError(
                f"per-unit doppler bins must be (units, planes), got "
                f"{tuple(doppler_bins.shape)} for {units} units")
        bins = tuple(range(doppler_bins.shape[1]))       # only its length is read
        # Each unit's active plane count: the anchor plane plus its bins up
        # to the last real one. The kernels skip the trailing pads (zero W
        # rows, so exact zeros either way) rather than accumulating them, so
        # a frame supported on 12 bins costs 12 bins' work inside a batch
        # whose static shape is the widest frame's. Counted to the LAST real
        # bin, not the number of real bins, so an interior pad -- which the
        # trainer never builds, but which would otherwise drop a real plane
        # -- is still rendered. Sync-free: this runs under graph capture.
        slot = torch.arange(doppler_bins.shape[1], device=doppler_bins.device)
        last = torch.where(doppler_bins >= 0, slot[None, :],
                           slot.new_full((), -1)).amax(dim=1)
        active_planes = (last + 2).to(torch.int32).contiguous()
        if _DOPPLER_AMPLITUDES == "closed-form":
            # (units, 1 + planes): the anchor's column first (ignored), then
            # every plane's fftshifted bin; the kernels evaluate the Hann
            # Doppler PSF at each path's fractional bin instead of summing
            # the 64 loops against W (W is still passed: its shape says
            # units and planes, and the DFT route stays selectable).
            plane_bins = torch.cat([doppler_bins.new_full((units, 1), -1), doppler_bins],
                                   dim=1).to(torch.int32).contiguous()
            if _DOPPLER_TAPS > 0:
                # (units, loops + 1): each unshifted bin's plane index (1..K)
                # in the unit's row, -1 where the unit does not render it;
                # column `loops` takes the pads. Sync-free (graph capture).
                dc = loops // 2
                unshifted = torch.where(doppler_bins >= 0, (doppler_bins + dc) % loops,
                                        doppler_bins.new_full((), loops))
                bin_to_plane = doppler_bins.new_full((units, loops + 1), -1)
                bin_to_plane.scatter_(1, unshifted, torch.arange(
                    1, doppler_bins.shape[1] + 1, device=doppler_bins.device)[None, :].expand(units, -1))
                bin_to_plane[:, loops] = -1
                bin_to_plane = bin_to_plane.to(torch.int32).contiguous()
    else:
        bins = (tuple(doppler_bins) if isinstance(doppler_bins, (list, tuple))
                else tuple(torch.as_tensor(doppler_bins).reshape(-1).tolist()))
    sorted_route = splat_components.get("sorted")
    if sorted_route is not None:
        # The sorted route (`_DopplerPlaneSorted`): the carrier and the
        # cached range bins in place of the contributions.
        if not (per_unit_bins and bin_to_plane is not None):
            raise ValueError("the sorted Doppler route needs per-unit bins, the closed form and taps > 0")
        real = splat_components["carrier"][:, 0]
    else:
        real = splat_components["contrib_real"]
        imag = splat_components["contrib_imag"]
    if per_unit_bins:
        weight_real, weight_imag = _per_unit_plane_weights(
            doppler_bins, loops, real.device, static_anchor)
    else:
        weight_real, weight_imag = _plane_weights(
            bins, loops, real.device, static_anchor)
    if real.shape[-1] == 0:
        zeros = background_profiles.new_zeros((units, len(bins), *output_shape))
        return background_profiles.clone(), zeros
    point_indices = splat_components["point_indices"].long()
    tx_indices = splat_components["tx_indices"].long()
    rx_indices = splat_components["rx_indices"].long()
    path_rates = _path_rates_of(point_path_rate_mps, splat_components, point_indices, tx_indices, rx_indices)
    times = _tdm_times(loops, chirp_duration_s, real.device)
    base = int(np.prod(output_shape))
    phase_const = SLOW_TIME_PHASE_SIGN * 2.0 * math.pi / wavelength_m
    if sorted_route is not None:
        out_real, out_imag = _DopplerPlaneSorted.apply(
            splat_components["carrier"], path_rates.float().contiguous(),
            sorted_route["record_static"], sorted_route["path_of_entry"], sorted_route["runs"],
            tx_indices.contiguous(), unit_of_path.long().contiguous(), active_planes, bin_to_plane,
            int(fft_window_kind() == "hann"), int(static_anchor == "dc-mean"),
            _tx_times64(loops, chirp_duration_s, real.device), _DOPPLER_TAPS,
            sorted_route["table4"], weight_real.shape[1], base, output_shape[2],
            _RANGE_SPREAD, loops, phase_const, output_shape[0] * output_shape[1])
        planes = torch.complex(out_real, out_imag).reshape(
            1 + len(bins), units, *output_shape)
        anchor, doppler = _SplitAnchor.apply(planes)
        return background_profiles + anchor, doppler
    # Follow-on C: int32. The destination is at most units*base -- 92,160 for
    # an evaluation chunk -- and this tensor is 4.8-11.0 M entries that the
    # binning reads twice and the backward once, so the second four bytes were
    # 115 MB of pure bandwidth per training step.
    flat_indices = (splat_components["flat_indices"].int()
                    + unit_of_path.int()[None, :] * int(base))
    out_real, out_imag = _DopplerPlaneSplat.apply(
        real, imag, flat_indices, tx_indices, path_rates.float(), times,
        weight_real, weight_imag, phase_const, units * base,
        unit_of_path.long().contiguous() if per_unit_bins else None,
        active_planes, plane_bins, int(fft_window_kind() == "hann"),
        int(static_anchor == "dc-mean"),
        _tx_times64(loops, chirp_duration_s, real.device) if plane_bins is not None else None,
        bin_to_plane, _DOPPLER_TAPS if bin_to_plane is not None else 0)
    planes = torch.complex(out_real, out_imag).reshape(
        1 + len(bins), units, *output_shape)
    # Split plane 0 from the rest with one autograd node whose backward is a
    # single cat, instead of two selects whose backwards each zero-fill the
    # whole (units, 1 + planes, tx, rx, K) cube and add (464 MB a pass on the
    # cascade, six passes a step).
    anchor, doppler = _SplitAnchor.apply(planes)
    return background_profiles + anchor, doppler


class _SplitAnchor(torch.autograd.Function):
    """``(1 + planes, units, ...)`` -> ``(units, ...)`` and ``(units, planes, ...)``; backward is one cat."""

    @staticmethod
    def forward(ctx, planes):
        ctx.units = planes.shape[1]
        return planes[0], planes[1:].transpose(0, 1)

    @staticmethod
    def backward(ctx, grad_anchor, grad_doppler):
        if grad_anchor is None:
            grad_anchor = torch.zeros_like(grad_doppler[:, 0])
        if grad_doppler is None:
            grad_doppler = torch.zeros(ctx.units, 0, *grad_anchor.shape[1:], dtype=grad_anchor.dtype,
                                       device=grad_anchor.device)
        return torch.cat([grad_anchor[None], grad_doppler.transpose(0, 1)], dim=0)

# --------------------------------------------------------------------
# training views and the losses on them
# --------------------------------------------------------------------

def coupling_block(radar: Radar, coupling: np.ndarray, device) -> torch.Tensor:
    """A radar's antenna-coupling calibration as a ``(n_tx, n_rx, samples)`` complex64 block.

    ``coupling`` is ``(azimuth TX x RX, near_bins)`` in the order
    `virtual_array_slow_time` builds (`calibration sequence-coupling`, HuPR's
    seed): virtual channel c is (azimuth TX c // n_rx, RX c % n_rx). The
    block is zero beyond the near bins and on the elevated TX; it is in
    product units, so it adds after the absolute gain. Works for the full
    radar and for its training sub-array alike (`Radar.training_subset`
    re-indexes `azimuth_tx` in the same order).
    """
    padded = np.zeros((radar.virtual_channels, radar.samples), dtype=np.complex64)
    padded[:, :coupling.shape[1]] = coupling
    block = np.zeros((radar.n_tx, radar.n_rx, radar.samples), dtype=np.complex64)
    for i, tx in enumerate(radar.azimuth_tx):
        block[tx] = padded[i * radar.n_rx:(i + 1) * radar.n_rx]
    return torch.from_numpy(block).to(device)


def build_rasterizers(args, device, checkpoint=None) -> tuple[dict, dict]:
    """The two radar configs and their mm3DPS rasterizers.

    Five call sites carried a byte-identical copy of this -- the three
    trainers, the renderer and the frame-1 range calibration -- so the mesh
    path, the antenna pattern names and the config wiring had five places to
    disagree. `checkpoint` exists because the calibration keeps its checkpoint
    on `args.checkpoint` while everything else uses `args.person_checkpoint`;
    the mesh is always `<checkpoint>.parent/input_scene/scene/mesh.ply`.

    The imports stay inside the call: `splat.rasterizer` pulls in the
    built CUDA extension, and importing this module must not require it.
    """
    from mm4DPS import rasterizer as bsdf_torch
    from mm4DPS.rasterizer import Rasterizer

    if checkpoint is None:
        checkpoint = args.person_checkpoint
    dataset = Dataset.of(args)
    configs = dataset.configs(args)
    pattern_root = args.mm3dps_root / "assets/antenna_pattern/MMWCAS"
    common = dict(
        mesh_file=str(checkpoint.parent / "input_scene/scene/mesh.ply"),
        tx_pattern_file=str(pattern_root / "tx1_76.npy"),
        rx_pattern_file=str(pattern_root / "rx1_76.npy"), device=str(device),
    )
    rasterizers = {
        name: Rasterizer(config_file=str(path), **common)
        for name, path in dataset.config_flags(args).items()
    }
    bsdf_torch.set_frequency(rasterizers[dataset.names[0]].center_freq)
    return configs, rasterizers



def range_profiles_to_ra_complex(
    range_profile_real: torch.Tensor,
    range_profile_imag: torch.Tensor,
    n_angle: int | None = None,
    radar: Radar | None = None,
) -> torch.Tensor:
    """Convert one renderer profile to complex RA with shape (range,angle).

    The assembly is the renderer-side counterpart of ``hupr_adc.range_angle_map``:
    TX0/RX0..3 followed by TX2/RX0..3 forms the uniform eight-element ULA.
    The azimuth window is applied on the grid by ``_to_grid`` (the measured
    chain applies the same one in ``data.virtual_to_grid``). The range-major layout matches the saved HuPR targets while retaining the
    phase needed for circularity, complex ACF, and phase-PSD diagnostics.
    """
    profile = torch.complex(range_profile_real, range_profile_imag)
    if profile.ndim != 3:
        raise ValueError(f"expected renderer range profile (tx,rx,K), got {tuple(profile.shape)}")
    radar = _layout(radar)
    virtual = _virtual_channels(profile, radar)
    angle = torch.fft.fftshift(torch.fft.fft(virtual, n=_angles(n_angle, radar), dim=0), dim=0)
    return angle.transpose(0, 1).contiguous()


def range_profiles_to_ra_magnitude(
    range_profile_real: torch.Tensor,
    range_profile_imag: torch.Tensor,
    n_angle: int | None = None,
    radar: Radar | None = None,
) -> torch.Tensor:
    """Convert renderer output (TX=3,RX=4,range) to |RA| (range,angle)."""
    return range_profiles_to_ra_complex(
        range_profile_real, range_profile_imag, n_angle=n_angle, radar=radar,
    ).abs()


def slow_time_profiles_to_ra_complex(
    range_profile_real: torch.Tensor,
    range_profile_imag: torch.Tensor,
    n_angle: int | None = None,
    virtual_channel_gain: torch.Tensor | None = None,
    radar: Radar | None = None,
) -> torch.Tensor:
    """Convert (loop,TX=3,RX=4,range) profiles to complex (loop,range,angle)."""
    profile = torch.complex(range_profile_real, range_profile_imag)
    if profile.ndim != 4:
        raise ValueError(
            f"expected slow-time renderer profiles (loop,tx,rx,K), got {tuple(profile.shape)}"
        )
    radar = _layout(radar)
    virtual = _virtual_channels(profile, radar)
    if virtual_channel_gain is not None:
        if virtual_channel_gain.shape != (virtual.shape[1],) or not virtual_channel_gain.is_complex():
            raise ValueError(f"virtual channel gain must be a complex vector of length {virtual.shape[1]}")
        virtual = virtual * virtual_channel_gain[None, :, None]
    angle = torch.fft.fftshift(torch.fft.fft(virtual, n=_angles(n_angle, radar), dim=1), dim=1)
    return angle.permute(0, 2, 1).contiguous()


def slow_time_ra_to_doppler_complex(
    ra_complex: torch.Tensor,
    remove_static_clutter: bool = True,
) -> torch.Tensor:
    """Return fftshifted complex Doppler×range×angle from per-loop RA.

    THE Doppler transform, both sides: mean removal (when asked), the Doppler
    window, the FFT over loops, fftshift. `doppler_plane_weights` is its
    linear map, row for row.
    """
    if ra_complex.ndim != 3:
        raise ValueError("expected complex slow-time RA with shape (loop,range,angle)")
    return slow_time_to_doppler(ra_complex, dim=0, remove_static_clutter=remove_static_clutter)


def slow_time_to_doppler(values: torch.Tensor, dim: int,
                         remove_static_clutter: bool = True) -> torch.Tensor:
    """Mean removal, Doppler window, FFT and fftshift along ``dim`` of any tensor."""
    if remove_static_clutter:
        values = values - values.mean(dim=dim, keepdim=True)
    values = apply_fft_window(values, "doppler", dim)
    return torch.fft.fftshift(torch.fft.fft(values, dim=dim), dim=dim)


def slow_time_ra_to_doppler_magnitude(
    ra_complex: torch.Tensor,
    remove_static_clutter: bool = True,
) -> torch.Tensor:
    """Return fftshifted |Doppler×range×angle| from complex per-loop RA."""
    return slow_time_ra_to_doppler_complex(
        ra_complex,
        remove_static_clutter=remove_static_clutter,
    ).abs()


# ----------------------------------------------------------------------
# Batch-agnostic forms of the three transforms above.
#
# The originals are rank-fixed (they assert ndim == 3 / ndim == 4), which is
# what forced the trainer to run the loss chain one frame at a time: a handful
# of kernels on a (64, 3, 4, 256) tensor each, twelve times over, entirely
# launch-latency bound. These take a leading batch of any rank and index with
# ellipsis and negative dims, so all twelve units go through one set of
# kernels. Same transforms, same axes, same order; the rank-fixed versions
# stay as the pinned references and are checked against these in the suite.
# ----------------------------------------------------------------------

def _layout(radar: Radar | None) -> Radar:
    """The board whose array the beamformer assembles: the dataset's first.

    HuPR's two boards are the same IWR1843 layout and RT-Pose has one board,
    so one description answers; a dataset with two different boards would
    pass the radar explicitly.
    """
    return radar if radar is not None else current_dataset().radars[0]


def _angles(n_angle: int | None, radar: Radar) -> int:
    return radar.n_angle if n_angle is None else n_angle


_GRID_CACHE: dict = {}


def _virtual_channels(profile: torch.Tensor, radar: Radar | None = None) -> torch.Tensor:
    """The azimuth virtual array, on its half-wavelength grid: ``(..., G, K)``.

    HuPR: TX0/RX0..3 followed by TX2/RX0..3, the uniform eight-element ULA,
    and the grid is the identity, so this is exactly the concatenation. The
    cascade: nine azimuth TX x sixteen RX = 144 channels on 86 slots, the
    channels that share a slot averaged (``Radar.azimuth_grid``).
    """
    radar = _layout(radar)
    if profile.shape[-3:-1] != (radar.n_tx, radar.n_rx):
        raise ValueError(
            f"expected renderer profiles (...,{radar.n_tx},{radar.n_rx},K), got {tuple(profile.shape)}")
    if tuple(radar.azimuth_tx) == tuple(range(radar.n_tx)):
        # Every TX is an azimuth TX (the training sub-array): the
        # concatenation of the per-TX blocks IS the (tx, rx) axes flattened,
        # a free view -- the same bytes in the same order as the cat below,
        # without n_tx selects and their n_tx zero-filling backwards.
        virtual = profile.reshape(*profile.shape[:-3], radar.n_tx * radar.n_rx, profile.shape[-1])
    else:
        virtual = torch.cat([profile[..., tx, :, :] for tx in radar.azimuth_tx], dim=-2)
    return _to_grid(virtual, radar)


def _to_grid(virtual: torch.Tensor, radar: Radar) -> torch.Tensor:
    """Place ``(..., V, K)`` virtual channels on the ``(..., G, K)`` azimuth grid, windowed for the angle FFT.

    The azimuth window (`signal_model.fft_window("azimuth", G)`) is applied
    here, per grid slot, so every angle FFT the renderer takes -- and it
    takes them all through `_virtual_channels` -- sees the same window the
    measured chain applies in `data.virtual_to_grid`.
    """
    slots, n = radar.azimuth_grid()
    real_dtype = virtual.real.dtype if virtual.is_complex() else virtual.dtype
    key = (radar.name, str(virtual.device), virtual.dtype, window_cache_key())
    plan = _GRID_CACHE.get(key)
    if plan is None:
        window = fft_window("azimuth", n)
        taper = (torch.ones(n, device=virtual.device, dtype=real_dtype) if window is None
                 else torch.from_numpy(window).to(device=virtual.device, dtype=real_dtype))
        if n == virtual.shape[-2] and bool((slots == np.arange(n)).all()):
            plan = _GRID_CACHE[key] = (None, taper[:, None])
        else:
            index = torch.as_tensor(slots, device=virtual.device)
            counts = torch.zeros(n, device=virtual.device, dtype=real_dtype)
            counts.index_add_(0, index, torch.ones(len(slots), device=virtual.device, dtype=counts.dtype))
            plan = _GRID_CACHE[key] = (index, (taper / counts)[:, None])
    index, weights = plan
    if index is None:
        return virtual * weights
    grid = virtual.new_zeros((*virtual.shape[:-2], n, virtual.shape[-1]))
    grid.index_add_(-2, index, virtual)
    return grid * weights


def profiles_to_ra_complex_batched(
    range_profile_real: torch.Tensor,
    range_profile_imag: torch.Tensor,
    n_angle: int | None = None,
    radar: Radar | None = None,
) -> torch.Tensor:
    """(...,tx,rx,K) renderer profiles -> complex (...,range,angle)."""
    radar = _layout(radar)
    virtual = _virtual_channels(
        torch.complex(range_profile_real, range_profile_imag), radar)
    angle = torch.fft.fftshift(
        torch.fft.fft(virtual, n=_angles(n_angle, radar), dim=-2), dim=-2)
    return angle.transpose(-1, -2).contiguous()


def profiles_to_rae_magnitude(
    range_profile_real: torch.Tensor,
    range_profile_imag: torch.Tensor,
    range_index: torch.Tensor | None = None,
    window: bool = True,
    radar: Radar | None = None,
) -> torch.Tensor:
    """``(..., tx, rx, K)`` renderer profiles of the FULL array -> ``|RAE|`` ``(..., range kept, n_angle, n_el)``.

    The elevation term's view (docs/FIDELITY_FIX_PLAN.md 1.4 F1): every
    TX-RX channel on the 2D (azimuth, elevation) grid of
    `data.elevation_grid_plan`, both windows, the azimuth FFT and the
    elevation FFT, fftshifted -- the transform the measured targets take in
    `data.range_angle_elevation_slow_time_cube_frames_torch`. The range slice
    runs first (the FFTs are over the channel axes, so it commutes).
    """
    return profiles_to_rae_spectrum_reference(
        range_profile_real, range_profile_imag, range_index, window, radar).abs()


def profiles_to_rae_spectrum_reference(
    range_profile_real: torch.Tensor,
    range_profile_imag: torch.Tensor,
    range_index: torch.Tensor | None = None,
    window: bool = True,
    radar: Radar | None = None,
    shift_azimuth: bool = True,
) -> torch.Tensor:
    """``profiles_to_rae_magnitude`` without the final ``abs``: the reference route's COMPLEX RAE view.

    Split out so a caller that needs the complex spectrum -- the fused plane term, which takes
    the magnitude itself -- can have it from this route as well as from
    `profiles_to_rae_spectrum_fused`. `profiles_to_rae_magnitude` is this followed by ``abs``,
    so it is unchanged op for op.
    """
    from mm4DPS.data import channels_to_grid2d, elevation_grid_plan
    radar = _layout(radar)
    plan = elevation_grid_plan(radar)
    profile = torch.complex(range_profile_real, range_profile_imag)
    if range_index is not None:
        profile = profile.index_select(-1, range_index)
    lead = profile.shape[:-3]
    channels = profile.reshape(*lead, radar.n_tx * radar.n_rx, profile.shape[-1])
    grid = channels_to_grid2d(channels, radar, axis=-2, window=window)         # (..., G_az, G_el, K)
    moved = torch.movedim(grid, -1, -3)                                        # (..., K, G_az, G_el)
    out = torch.fft.fft(moved, n=plan["n_angle"], dim=-2)
    if shift_azimuth:
        out = torch.fft.fftshift(out, dim=-2)
    return torch.fft.fftshift(torch.fft.fft(out, n=plan["n_el"], dim=-1), dim=-1)


def elevation_floor_factor(radar: Radar | None = None) -> float:
    """The learned RA floor's per-cell power, carried to an RAE cell.

    White receiver noise of power p per channel. The RA view averages the
    azimuth channels sharing a slot (``1/count`` weights) and windows the
    slots, so an RA cell carries ``p * sum_s w_az(s)^2 / count_s``; the RAE
    view places every TX x RX channel on its own (azimuth, elevation) slot
    (shared slots averaged likewise), windows both axes and transforms
    both, so an RAE cell carries ``p * sum_s w_az(s)^2 w_el(s)^2 / count_s``
    over the 2D slots. The ratio is exact for white noise and a constant
    per radar (HuPR 1.89, the cascade 1.67, with the untapered elevation axis). The 2026-09-16
    value (the elevation window's energy alone, 1.02 on the cascade) under-
    floored the RAE view 3.6x and, through the term's gradient, pulled the
    shared learned floor 2.5-3x below the measured off-person power on the
    cascade (docs/FIDELITY_FIX_RESULTS.md 3.8).
    """
    from mm4DPS.data import elevation_grid_plan
    radar = _layout(radar)
    plan = elevation_grid_plan(radar)
    n_el = plan["n_el_slots"]
    w_az = fft_window("azimuth", plan["n_az"])
    w_az = np.ones(plan["n_az"]) if w_az is None else np.asarray(w_az)
    w_el = np.ones(n_el)                                   # untapered, as `channels_to_grid2d`
    flat = plan["az_slots"] * n_el + plan["el_slots"]
    counts2d = np.bincount(flat, minlength=plan["n_az"] * n_el)
    # each occupied 2D slot contributes (w_az w_el)^2 / count once
    occupied = sorted(set(int(v) for v in flat))
    rae = sum((w_az[v // n_el] * w_el[v % n_el]) ** 2 / counts2d[v] for v in occupied)
    slots, n = radar.azimuth_grid()
    w1 = fft_window("azimuth", n); w1 = np.ones(n) if w1 is None else np.asarray(w1)
    counts1d = np.bincount(slots, minlength=n)
    ra = sum(w1[g] ** 2 / counts1d[g] for g in sorted(set(int(v) for v in slots)))
    return float(rae / ra)


def profiles_to_ra_magnitude_batched(
    range_profile_real: torch.Tensor,
    range_profile_imag: torch.Tensor,
    n_angle: int | None = None,
    radar: Radar | None = None,
) -> torch.Tensor:
    """(...,3,4,K) renderer profiles -> |RA| (...,range,angle)."""
    return profiles_to_ra_complex_batched(
        range_profile_real, range_profile_imag, n_angle=n_angle,
    ).abs()








_ANGLE_SHIFT_SIGNS: dict = {}


def _angle_shift_signs(channels: int, n_angle: int, device) -> torch.Tensor:
    """``(-1)^n`` per virtual channel — ``fftshift`` folded into the input.

    For even ``n_angle``, ``fftshift`` after the angle FFT indexes bin
    ``k + n_angle/2 (mod n_angle)``, and
    ``X[k + N/2] = sum_n x[n] e^{-i pi n} e^{-2 pi i nk/N}`` — so multiplying
    channel ``n`` by ``e^{-i pi n} = (-1)^n`` *before* the FFT produces the
    fftshifted spectrum with no shift pass at all. The factors are exactly
    ``+1``/``-1``, so nothing is perturbed; the FFT just sees sign-flipped
    inputs. The signs touch only the ``channels`` real inputs (the zero
    padding to ``n_angle`` is unaffected).
    """
    if n_angle % 2:
        raise ValueError("the sign fold requires an even angle FFT length")
    key = (channels, str(device))
    signs = _ANGLE_SHIFT_SIGNS.get(key)
    if signs is None:
        signs = torch.where(
            torch.arange(channels, device=device) % 2 == 0, 1.0, -1.0,
        )[:, None]
        _ANGLE_SHIFT_SIGNS[key] = signs
    return signs


def _spectrum_input_eager(real: torch.Tensor, imag: torch.Tensor, channel_of_slot: torch.Tensor,
                          slot_valid: torch.Tensor, slot_weight: torch.Tensor,
                          range_index: torch.Tensor, n_angle: int) -> torch.Tensor:
    """The angle FFT's input, ``(..., range kept, n_angle)`` complex, in one pass.

    ``real``/``imag`` are the renderer's ``(..., channels, K)`` profiles.
    Slot ``g`` of the azimuth grid gathers its channels
    (``channel_of_slot[g, :]``, ``slot_valid`` masking the padding), scaled
    by the slot's window / multiplicity and the ``(-1)^g`` fftshift sign
    (``slot_weight``); the kept range rows are gathered, the layout
    transposed to range-major and the zero padding to ``n_angle`` written --
    the same arithmetic as complex -> grid index_add -> index_select ->
    signs -> transpose -> pad, as one compiled kernel over one output.
    """
    lead = real.shape[:-2]
    kept = real.index_select(-1, range_index)                            # (..., C, R)
    kept_i = imag.index_select(-1, range_index)
    gathered_r = kept[..., channel_of_slot, :]                           # (..., G, M, R)
    gathered_i = kept_i[..., channel_of_slot, :]
    mask = slot_valid.to(real.dtype)[:, :, None]
    slot_r = (gathered_r * mask).sum(-2) * slot_weight[:, None]          # (..., G, R)
    slot_i = (gathered_i * mask).sum(-2) * slot_weight[:, None]
    out = torch.zeros((*lead, range_index.numel(), n_angle, 2), dtype=real.dtype, device=real.device)
    out[..., :slot_weight.numel(), 0] = slot_r.transpose(-1, -2)
    out[..., :slot_weight.numel(), 1] = slot_i.transpose(-1, -2)
    return torch.view_as_complex(out)


_spectrum_input = torch.compile(_spectrum_input_eager, dynamic=False)
_SPECTRUM_PLAN: dict = {}


def _spectrum_plan(radar: Radar, channels: int, n_angle: int, device, dtype):
    """The slot -> channel table, validity, and per-slot weight x sign, cached per radar and device."""
    key = (radar.name, channels, n_angle, str(device), dtype, window_cache_key())
    plan = _SPECTRUM_PLAN.get(key)
    if plan is None:
        slots, n = radar.azimuth_grid()
        if channels != radar.n_tx * radar.n_rx:
            raise ValueError(f"expected {radar.n_tx * radar.n_rx} channels (every TX x RX), got {channels}")
        # the azimuth virtual channel v = (i, r) is flattened channel azimuth_tx[i] * n_rx + r of the
        # (tx, rx) profile -- the full array (every TX rendered) gathers its azimuth rows the same way
        channel_ids = [tx * radar.n_rx + r for tx in radar.azimuth_tx for r in range(radar.n_rx)]
        members = [[] for _ in range(n)]
        for v, slot in enumerate(slots):
            members[int(slot)].append(channel_ids[v])
        width = max(len(m) for m in members)
        table = torch.zeros((n, width), dtype=torch.long)
        valid = torch.zeros((n, width), dtype=torch.bool)
        for g, m in enumerate(members):
            table[g, :len(m)] = torch.tensor(m)
            valid[g, :len(m)] = True
        window = fft_window("azimuth", n)
        taper = torch.ones(n, dtype=torch.float64) if window is None else torch.from_numpy(np.asarray(window)).double()
        counts = torch.tensor([max(len(m), 1) for m in members], dtype=torch.float64)
        signs = _angle_shift_signs(n, n_angle, "cpu").double().reshape(-1)
        weight = (taper / counts * signs).to(dtype)
        plan = _SPECTRUM_PLAN[key] = (table.to(device), valid.to(device), weight.to(device))
    return plan


def profiles_to_ra_spectrum_fused(
    range_profile_real: torch.Tensor,
    range_profile_imag: torch.Tensor,
    range_index: torch.Tensor | None = None,
    n_angle: int | None = None,
    radar: Radar | None = None,
) -> torch.Tensor:
    """The fused chain's complex RA spectrum, ``(..., range kept, angle)``.

    ``profiles_to_ra_magnitude_fused`` without the final ``abs`` -- for
    consumers that fuse the magnitude into their own kernel. When every TX
    is an azimuth TX (the training sub-array, HuPR's pair) the FFT's input
    is built by one compiled gather (`_spectrum_input`); otherwise by the
    op-by-op chain below (the same values to float rounding).
    """
    radar = _layout(radar)
    n_angle = _angles(n_angle, radar)
    if _FUSED_SPECTRUM_INPUT and range_profile_real.is_cuda:
        channels = radar.n_tx * radar.n_rx
        real = range_profile_real.reshape(*range_profile_real.shape[:-3], channels, range_profile_real.shape[-1])
        imag = range_profile_imag.reshape(*range_profile_imag.shape[:-3], channels, range_profile_imag.shape[-1])
        if range_index is None:
            range_index = torch.arange(real.shape[-1], device=real.device)
        table, valid, weight = _spectrum_plan(radar, channels, n_angle, real.device, real.dtype)
        return torch.fft.fft(_spectrum_input(real, imag, table, valid, weight, range_index, n_angle), dim=-1)
    virtual = _virtual_channels(
        torch.complex(range_profile_real, range_profile_imag), radar)
    if range_index is not None:
        virtual = virtual.index_select(-1, range_index)
    signs = _angle_shift_signs(virtual.shape[-2], n_angle, virtual.device)
    return torch.fft.fft(
        (virtual * signs).transpose(-1, -2).contiguous(), n=n_angle, dim=-1)


_FUSED_SPECTRUM_INPUT = os.environ.get("HUPR_FUSED_SPECTRUM_INPUT", "1") == "1"


def _rae_spectrum_input_eager(real: torch.Tensor, imag: torch.Tensor, channel_of_slot: torch.Tensor,
                              slot_valid: torch.Tensor, slot_weight: torch.Tensor,
                              range_index: torch.Tensor, n_az: int, n_el_slots: int,
                              n_angle: int, n_el: int) -> torch.Tensor:
    """The 2D angle-elevation FFT's input, ``(units, ..., R, n_angle, n_el)`` complex, in one pass.

    ``real``/``imag`` are the renderer's ``(units, ..., channels, K)`` full-
    array profiles; ``range_index`` ``(units, R)`` is each unit's rows (a
    shared row set is the same row expanded to every unit). Slot ``s`` of
    the flattened ``(n_az, n_el_slots)`` grid gathers its channels
    (``channel_of_slot[s, :]``, ``slot_valid`` masking the padding), scaled
    by the slot's window / multiplicity and the ``(-1)^(g_az + g_el)``
    fftshift signs (``slot_weight``); the rows are gathered, the layout
    transposed to range-major and the zero padding to ``(n_angle, n_el)``
    written -- the arithmetic of `data.channels_to_grid2d` (index_add,
    weights) + movedim + the two ``n=``-padded FFT inputs, as one compiled
    kernel over one output (`_spectrum_input` for the azimuth path).
    """
    units = real.shape[0]
    lead = real.shape[:-2]
    rows = range_index.reshape(units, *([1] * (real.dim() - 2)), -1).expand(*real.shape[:-1], -1)
    kept = real.gather(-1, rows)                                          # (units, ..., C, R)
    kept_i = imag.gather(-1, rows)
    gathered_r = kept[..., channel_of_slot, :]                            # (units, ..., S, M, R)
    gathered_i = kept_i[..., channel_of_slot, :]
    mask = slot_valid.to(real.dtype)[:, :, None]
    slot_r = (gathered_r * mask).sum(-2) * slot_weight[:, None]           # (units, ..., S, R)
    slot_i = (gathered_i * mask).sum(-2) * slot_weight[:, None]
    n_rows = range_index.shape[-1]
    out = torch.zeros((*lead, n_rows, n_angle, n_el, 2), dtype=real.dtype, device=real.device)
    out[..., :n_az, :n_el_slots, 0] = torch.movedim(slot_r.reshape(*lead, n_az, n_el_slots, n_rows), -1, -3)
    out[..., :n_az, :n_el_slots, 1] = torch.movedim(slot_i.reshape(*lead, n_az, n_el_slots, n_rows), -1, -3)
    return torch.view_as_complex(out)


_rae_spectrum_input = torch.compile(_rae_spectrum_input_eager, dynamic=False)
_RAE_SPECTRUM_PLAN: dict = {}


def _rae_spectrum_plan(radar: Radar, device, dtype):
    """The 2D slot -> channel table, validity, per-slot weight x signs, and the grid sizes, cached per radar and device."""
    from mm4DPS.data import elevation_grid_plan
    key = (radar.name, radar.n_tx, radar.n_rx, str(device), dtype, fft_window_kind())
    plan = _RAE_SPECTRUM_PLAN.get(key)
    if plan is None:
        p = elevation_grid_plan(radar)
        n_az, n_el_slots, n_el, n_angle = p["n_az"], p["n_el_slots"], p["n_el"], p["n_angle"]
        channels = radar.n_tx * radar.n_rx
        flat = p["az_slots"] * n_el_slots + p["el_slots"]                 # channel c = tx * n_rx + rx -> slot
        members = [[] for _ in range(n_az * n_el_slots)]
        for c, slot in enumerate(flat):
            members[int(slot)].append(c)
        width = max(len(m) for m in members)
        table = torch.zeros((len(members), width), dtype=torch.long)
        valid = torch.zeros((len(members), width), dtype=torch.bool)
        for g, m in enumerate(members):
            table[g, :len(m)] = torch.tensor(m)
            valid[g, :len(m)] = True
        # the weights EXACTLY as `channels_to_grid2d` computes them (float32 window product over
        # float32 counts), so the fused input is bitwise the scattered grid's
        w_az = fft_window("azimuth", n_az)
        w_az = np.ones(n_az) if w_az is None else w_az
        w_el = np.ones(n_el_slots)
        counts = torch.tensor([float(len(m)) for m in members], dtype=dtype)
        weight = torch.as_tensor(np.outer(w_az, w_el).reshape(-1), dtype=dtype) / counts.clamp_min(1.0)
        # The fftshifts folded into the input as (-1)^g_az (-1)^g_el where that is bitwise
        # the roll: a power-of-two length (measured: 16, 128 and 8 exact; 86 differs at
        # 2.9e-7 of the peak, so the cascade's azimuth axis keeps the roll after the FFT).
        # An odd elevation length is left in DFT order on both sides (`elevation_shift`).
        g_az = torch.arange(len(members)) // n_el_slots
        g_el = torch.arange(len(members)) % n_el_slots
        fold_az = n_angle & (n_angle - 1) == 0
        signs = (_angle_shift_signs(n_angle, n_angle, "cpu").reshape(-1)[g_az].to(dtype) if fold_az
                 else torch.ones(len(members), dtype=dtype))
        if elevation_shift(radar):
            signs = signs * _angle_shift_signs(n_el, n_el, "cpu").reshape(-1)[g_el].to(dtype)
        # The direct form the fused kernel reads (docs/RUNTIME_STEP_20MS_PLAN.md item C). The
        # channel -> slot map is a PARTITION -- on the cascade 76 slots hold one channel and 58 hold
        # two, 192 in all -- so every output cell of the (n_angle, n_el) grid needs at most two
        # channels and one weight, and no cell is written twice. Laid out over the OUTPUT grid
        # rather than over the 602-slot table, which is 78% empty: the table form materialises a
        # (units, planes, 602, 2, rows) intermediate, 565 MiB a step, 3.5x the buffer the transform
        # consumes.
        w_signed = (weight * signs).to(torch.float32)
        ch_a = torch.full((n_angle * n_el,), -1, dtype=torch.int32)
        ch_b = torch.full((n_angle * n_el,), -1, dtype=torch.int32)
        w_out = torch.zeros((n_angle * n_el,), dtype=torch.float32)
        for g, m in enumerate(members):
            a, e = g // n_el_slots, g % n_el_slots      # the 602-slot grid is (n_az, n_el_slots)
            cell = a * n_el + e                        # the output grid is (n_angle, n_el)
            w_out[cell] = w_signed[g]
            if len(m) > 0: ch_a[cell] = m[0]
            if len(m) > 1: ch_b[cell] = m[1]
            if len(m) > 2:
                raise RuntimeError(f"slot {g} holds {len(m)} channels; the fused input assumes at most two")
        # The inverse map, for the backward: each channel belongs to exactly one output cell (the
        # map is a partition), so the gradient is a permutation times the slot weight -- no
        # accumulation, no atomics, and deterministic without asking for it.
        cell_of_channel = torch.full((channels,), -1, dtype=torch.int32)
        for g, m in enumerate(members):
            a, e = g // n_el_slots, g % n_el_slots
            for c in m:
                cell_of_channel[c] = a * n_el + e
        if int((cell_of_channel < 0).sum()) and len(members):
            raise RuntimeError("a channel was placed in no slot; the fused input's backward assumes a partition")
        # Which elevation columns of the (n_angle, n_el) grid hold anything at all. Derived, never
        # assumed: the cascade occupies 4 of 8 (rows 1, 3, 4 carry no antenna and 7 is the 7 -> 8
        # pad), HuPR occupies both of 2, and a different board gets whatever it gets. An all-zero
        # column's azimuth transform is exactly zero, so it need not be computed
        # (docs/RUNTIME_STEP_20MS_PLAN.md item C2). The same tables are rebuilt over the compact
        # grid, so the kernel writes (n_angle, n_occupied) directly.
        occupied_el = sorted({int(g % n_el_slots) for g, m in enumerate(members) if len(m) > 0})
        n_occ = len(occupied_el)
        el_to_compact = {e: i for i, e in enumerate(occupied_el)}
        ch_a_c = torch.full((n_angle * n_occ,), -1, dtype=torch.int32)
        ch_b_c = torch.full((n_angle * n_occ,), -1, dtype=torch.int32)
        w_c = torch.zeros((n_angle * n_occ,), dtype=torch.float32)
        cell_of_channel_c = torch.full((channels,), -1, dtype=torch.int32)
        for g, m in enumerate(members):
            a, e = g // n_el_slots, g % n_el_slots
            if e not in el_to_compact:
                if len(m):
                    raise RuntimeError(f"slot {g} holds channels in a column marked empty")
                continue
            cell = a * n_occ + el_to_compact[e]
            w_c[cell] = w_signed[g]
            if len(m) > 0: ch_a_c[cell] = m[0]
            if len(m) > 1: ch_b_c[cell] = m[1]
            for c in m:
                cell_of_channel_c[c] = cell
        plan = _RAE_SPECTRUM_PLAN[key] = (
            table.to(device), valid.to(device), (weight * signs).to(device), n_az, n_el_slots, n_angle, n_el, fold_az,
            ch_a.to(device), ch_b.to(device), w_out.to(device), cell_of_channel.to(device),
            tuple(occupied_el), ch_a_c.to(device), ch_b_c.to(device), w_c.to(device),
            cell_of_channel_c.to(device),
            # on the device already: building it from a Python list per call is a host-to-device
            # copy, which the training step's CUDA-graph capture forbids
            torch.tensor(occupied_el, dtype=torch.long, device=device))
    return plan


def rae_azimuth_rolled(radar: Radar | None = None) -> bool:
    """Whether the RAE view's azimuth fftshift is an actual roll rather than folded into input signs.

    A power-of-two angle length folds the shift into the per-slot signs and costs nothing (HuPR,
    16); any other length keeps the roll after the FFT (the cascade, 86 = 2 x 43), and that roll
    moves 646 MiB a step forward and the same again in the backward -- 1.48 ms measured. Only where
    this is true is there anything for the shift-skipping of item B to save.
    """
    from mm4DPS.data import elevation_grid_plan
    n_angle = elevation_grid_plan(_layout(radar))["n_angle"]
    return (n_angle & (n_angle - 1)) != 0


def elevation_shift(radar: Radar | None = None) -> bool:
    """Whether the elevation axis of the RAE view is fftshifted: yes for an even length (the sign fold is exact), no for an odd one (both sides in DFT order; the term sums over cells)."""
    from mm4DPS.data import elevation_grid_plan
    return elevation_grid_plan(_layout(radar))["n_el"] % 2 == 0


class _RaeSpectrumInput(torch.autograd.Function):
    """The RAE view's 2D-FFT input, built straight from the channels.

    Bitwise the table form (`_rae_spectrum_input_eager`), which it replaces to avoid that form's
    (units, planes, n_az * n_el_slots, width, rows) intermediate -- 565 MiB a step on the cascade,
    78% of its slots empty, 3.5x the buffer the transform consumes. The backward is a permutation
    times the slot weight, because the channel -> slot map is a partition: no accumulation, no
    atomics, deterministic without asking (docs/RUNTIME_STEP_20MS_PLAN.md item C).
    """

    @staticmethod
    def forward(ctx, spec, range_index, ch_a, ch_b, w_slot, cell_of_channel,
                n_angle, n_el, lead_per_unit):
        from mm4DPS.cuda_ext import load_hupr_cuda
        out = load_hupr_cuda().hupr_rae_spectrum_input(
            spec, range_index, ch_a, ch_b, w_slot, n_angle, n_el, lead_per_unit)
        ctx.save_for_backward(range_index, cell_of_channel, w_slot)
        ctx.shape = (spec.shape[-2], spec.shape[-1], lead_per_unit)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        from mm4DPS.cuda_ext import load_hupr_cuda
        range_index, cell_of_channel, w_slot = ctx.saved_tensors
        channels, bins, lead_per_unit = ctx.shape
        grad_spec = load_hupr_cuda().hupr_rae_spectrum_input_backward(
            grad_out.contiguous(), range_index, cell_of_channel, w_slot,
            channels, bins, lead_per_unit)
        return (grad_spec,) + (None,) * 8


#: HUPR_RAE_INPUT=table restores the slot-table gather, the A/B MM4DPS_RAE_CHECK compares against.
_RAE_INPUT = os.environ.get("HUPR_RAE_INPUT", "direct")
#: HUPR_RAE_C2=off transforms every elevation column, including the structurally empty ones.
_RAE_C2 = os.environ.get("HUPR_RAE_C2", "on") != "off"


def profiles_to_rae_spectrum_fused(
    range_profile: torch.Tensor,
    range_index: torch.Tensor,
    radar: Radar | None = None,
    shift_azimuth: bool = True,
) -> torch.Tensor:
    """``(units, ..., tx, rx, K)`` full-array profiles -> complex ``(units, ..., R, n_angle, n_el)``: the fftshifted 2D spectrum of the gathered rows.

    ``profiles_to_rae_magnitude`` without the final ``abs`` and with the
    grid scatter, the layout copies and the fftshift rolls replaced by one
    compiled gather (`_rae_spectrum_input`) and two FFTs on the pre-padded
    buffer (the azimuth roll stays where its length is not a power of two)
    -- the same arithmetic on one contiguous buffer, bitwise the reference
    on the accepted configuration (docs/RUNTIME_RESULTS.md item 2).
    ``range_index`` is ``(R,)`` (every unit the same rows) or ``(units, R)``
    (the person's rows per unit).
    """
    radar = _layout(radar)
    channels = radar.n_tx * radar.n_rx
    # the COMPLEX profile: taking .real / .imag here would hand the kernel two stride-2 views and
    # cost two contiguous copies forward plus the gradient's recombination, measured at 1.26 ms a
    # call (docs/RUNTIME_STEP_20MS_PLAN.md 12)
    real = range_profile.reshape(*range_profile.shape[:-3], channels, range_profile.shape[-1])
    if range_index.dim() == 1:
        range_index = range_index[None].expand(real.shape[0], -1)
    (table, valid, weight, n_az, n_el_slots, n_angle, n_el, fold_az,
     ch_a, ch_b, w_slot, cell_of_channel, occupied_el, ch_a_c, ch_b_c, w_c,
     cell_of_channel_c, occ_index) = _rae_spectrum_plan(
         radar, real.device, real.real.dtype)   # the plan's weights are real, the profile complex
    compact = _RAE_C2 and len(occupied_el) < n_el and real.is_cuda and _RAE_INPUT == "direct"
    if _RAE_INPUT == "direct" and real.is_cuda:
        lead_per_unit = 1
        for d in range(1, real.dim() - 2):
            lead_per_unit *= real.shape[d]
        grid = _RaeSpectrumInput.apply(
            real.contiguous(), range_index.contiguous(),
            ch_a_c if compact else ch_a, ch_b_c if compact else ch_b,
            w_c if compact else w_slot, cell_of_channel_c if compact else cell_of_channel,
            n_angle, len(occupied_el) if compact else n_el, lead_per_unit)
    else:
        kernel = _rae_spectrum_input if (_COMPILED_RAE_INPUT and real.is_cuda) else _rae_spectrum_input_eager
        grid = kernel(real.real.contiguous(), real.imag.contiguous(), table, valid, weight,
                      range_index, n_az, n_el_slots, n_angle, n_el)
    if compact:
        # the azimuth transform only over the columns that hold something; the rest are exactly
        # zero, so the elevation stage is fed the same values it would have been
        angle = torch.fft.fft(grid, dim=-2)
        full = torch.zeros((*angle.shape[:-1], n_el), dtype=angle.dtype, device=angle.device)
        full.index_copy_(-1, occ_index, angle)      # capture-safe: the index lives on the device
        spectrum = torch.fft.fft(full, dim=-1)
    else:
        spectrum = torch.fft.fft(torch.fft.fft(grid, dim=-2), dim=-1)
    if fold_az or not shift_azimuth:
        return spectrum
    return torch.fft.fftshift(spectrum, dim=-2)


_COMPILED_RAE_INPUT = os.environ.get("HUPR_COMPILED_RAE_INPUT", "1") == "1"   # the gather eager (same ops) when 0


def profiles_to_ra_magnitude_fused(
    range_profile_real: torch.Tensor,
    range_profile_imag: torch.Tensor,
    range_index: torch.Tensor | None = None,
    n_angle: int | None = None,
    radar: Radar | None = None,
) -> torch.Tensor:
    """``profiles_to_ra_magnitude_batched`` with the range slice pulled first.

    Exactly ``profiles_to_ra_magnitude_batched(real, imag)`` followed by
    ``index_select`` of ``range_index`` on the range axis, computed in an
    order that does the same arithmetic on far fewer bytes:

    * the range slice runs *before* the angle FFT — the FFT is over the
      virtual-channel axis, so it is independent per range bin and the two
      operations commute exactly;
    * ``fftshift`` is folded into ``(-1)^n`` channel signs
      (``_angle_shift_signs``), removing the roll's full-tensor copy;
    * the range-major transpose is hoisted *before* the FFT, where the tensor
      has 8 channels instead of ``n_angle`` angle bins — 8x fewer bytes to
      reorder — and the FFT then writes the (range, angle) layout directly.

    Same output shape and orientation as the reference chain:
    ``(..., range kept, angle)``. The equivalence gate that pinned this against
    the reference went with ``tests/`` in the deployment strip.
    """
    return profiles_to_ra_spectrum_fused(
        range_profile_real, range_profile_imag, range_index, n_angle, radar,
    ).abs()


def doppler_planes_to_motion_magnitude(
    plane_real: torch.Tensor,
    plane_imag: torch.Tensor,
    n_angle: int | None = None,
    radar: Radar | None = None,
) -> torch.Tensor:
    """|Doppler x range x angle| from planes that are *already* Doppler bins.

    The tail of ``motion_doppler_magnitude_batched`` for the E1 renderer, which
    returns the requested Doppler bins directly: the loop axis, the clutter
    removal and the Doppler FFT have all been folded into the splat, so only
    the beamforming is left.

    Clutter removal and the background do not reappear here and do not need
    to: the plane weights carry the mean removal (``doppler_dft_rows``), so
    anything constant over the loop axis -- the background -- renders as
    exactly zero in every plane, and DC itself is not among the rendered
    bins.

    ``(..., bins, 3, 4, K)`` -> ``(..., bins, range, angle)``.
    """
    radar = _layout(radar)
    virtual = _virtual_channels(torch.complex(plane_real, plane_imag), radar)
    angle = torch.fft.fftshift(
        torch.fft.fft(virtual, n=_angles(n_angle, radar), dim=-2), dim=-2)
    return angle.transpose(-1, -2).abs()


def subtract_spatial_noise_floor(
    doppler_magnitude: torch.Tensor,
    foreground_ra_mask: torch.Tensor,
    quantile: float = 0.5,
) -> torch.Tensor:
    """Subtract a per-Doppler-bin sensor floor estimated outside the person ROI.

    The floor is detached before subtraction so optimization cannot improve the
    score by manipulating the background estimate. Raw measured/rendered maps
    remain available for visualization; this function only defines a cleaner
    motion-comparison target.
    """
    if doppler_magnitude.ndim != 3:
        raise ValueError("expected Doppler magnitude with shape (D,range,angle)")
    if foreground_ra_mask.shape != doppler_magnitude.shape[1:]:
        raise ValueError("foreground mask must match the range-angle dimensions")
    if foreground_ra_mask.dtype != torch.bool:
        raise ValueError("foreground mask must be boolean")
    if not 0.0 <= quantile < 1.0:
        raise ValueError("noise-floor quantile must lie in [0,1)")
    background = doppler_magnitude[:, ~foreground_ra_mask]
    if background.shape[1] == 0:
        raise ValueError("noise-floor estimation needs non-foreground bins")
    floor = torch.quantile(background.detach(), quantile, dim=1)
    return (doppler_magnitude - floor[:, None, None]).clamp_min(0.0)


def linear_pearson_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - Pearson correlation of linear-magnitude maps.

    The linear-domain counterpart of `log_pearson_loss`, and the domain every
    metric in this project is supposed to live in (docs/METRICS_ALL_PRODUCTS.md).
    dB compresses dynamic range, so a dB correlation stays high over a floor
    that is badly wrong; the linear one does not. Same statistic as
    the background-refinement Pearson loss, hoisted here so the renderer can
    report it without importing a training module.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"prediction {tuple(prediction.shape)} != target {tuple(target.shape)}")
    pred = prediction.reshape(-1)
    truth = target.reshape(-1)
    pred = pred - pred.mean()
    truth = truth - truth.mean()
    denom = torch.sqrt(pred.square().sum() * truth.square().sum()).clamp_min(1e-20)
    return 1.0 - (pred * truth).sum() / denom


def log_pearson_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Scale-robust loss for measured and rendered magnitude maps."""
    if prediction.shape != target.shape:
        raise ValueError(f"prediction {tuple(prediction.shape)} != target {tuple(target.shape)}")
    def to_db(values: torch.Tensor) -> torch.Tensor:
        peak = values.detach().amax().clamp_min(1e-20)
        floor = peak * (10.0 ** (-42.0 / 20.0))
        return 20.0 * torch.log10(values.clamp_min(floor) / peak)

    pred = to_db(prediction).reshape(-1)
    truth = to_db(target).reshape(-1)
    pred = pred - pred.mean()
    truth = truth - truth.mean()
    denom = torch.sqrt(pred.square().sum() * truth.square().sum()).clamp_min(1e-20)
    return 1.0 - (pred * truth).sum() / denom


def range_profiles_to_ra_magnitude_frames(
    range_profile_real: torch.Tensor,
    range_profile_imag: torch.Tensor,
    n_angle: int | None = None,
    radar: Radar | None = None,
) -> torch.Tensor:
    """``(frames,3,4,K)`` -> ``(frames,K,n_angle)`` magnitudes.

    :func:`range_profiles_to_ra_complex` with a leading frame axis: the same
    TX0/RX0..3 + TX2/RX0..3 ULA assembly and the same transform, so a frame of
    this equals the per-frame call on that frame's profile.
    """
    profile = torch.complex(range_profile_real, range_profile_imag)
    if profile.ndim != 4:
        raise ValueError(
            f"expected (frames,tx,rx,K), got {tuple(profile.shape)}")
    radar = _layout(radar)
    virtual = _virtual_channels(profile, radar)
    angle = torch.fft.fftshift(
        torch.fft.fft(virtual, n=_angles(n_angle, radar), dim=1), dim=1)
    return angle.transpose(1, 2).contiguous().abs()
