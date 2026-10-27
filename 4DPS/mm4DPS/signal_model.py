"""The radar signal model: chirp motion, and the receiver's noise.

Two modules merged here by `docs/CONSOLIDATION_AND_OO_PLAN.md` C8, from
`forward/chirp_motion.py` and `forward/noise.py`. No name collisions.

    chirp motion   TDM sample times, joint radial velocities, the slow-time
                   phase, and the Doppler masks built from them
    noise          the learned floor, the receiver nuisances, the diffuse floor

Named signal_model rather than signal so it cannot shadow the standard library.
"""

from __future__ import annotations

import json
import math
import numpy as np
import torch

from dataclasses import dataclass
from pathlib import Path

from mm4DPS import HUPR, Radar, current_dataset


# --------------------------------------------------------------------
# chirp motion: sample times, joint velocities, slow-time phase
# --------------------------------------------------------------------

# HuPR's chirp, under the names the HuPR-only paths use. Anything shared
# between datasets takes a Radar or resolves the current dataset's.
_IWR1843 = HUPR.radars[0]
CHIRP_DURATION_S = _IWR1843.chirp_duration_s
NUM_TX = _IWR1843.n_tx
TDM_LOOPS = _IWR1843.loops
LOOP_PERIOD_S = _IWR1843.loop_period_s
CARRIER_HZ = _IWR1843.carrier_hz
WAVELENGTH_M = 299_792_458.0 / CARRIER_HZ

# Sign of the slow-time phase ramp, phase = SIGN * 2 pi * (dL/dt) * t / lambda,
# where dL/dt is the bistatic path rate and is positive when a point recedes.
#
# This is fixed empirically against measured data, not derived.  With the
# previous value of -1 the rendered Doppler axis ran opposite to the measured
# one: on sequence 102, receding mapped to negative Doppler bins in the render
# and to positive bins in the measurement, and the rendered and measured Doppler
# centroids correlated -0.987.  Negating the ramp raised the full micro-Doppler
# spectrogram correlation from +0.540 to +0.851.
#
# Caveat worth keeping in view: the slow-time phase is no longer the exact time
# derivative of the static path phase, which still carries a leading minus sign.
# That inconsistency is real and says the static/angle phase convention deserves
# the same scrutiny.  It cannot be settled from the angle axis as things stand,
# because the measured angle axis is mapped onto camera bins by an empirically
# fitted slope (-0.958 horizontal, +0.578 vertical) that absorbs both a sign and
# a scale, so it neither confirms nor refutes a global conjugation.
SLOW_TIME_PHASE_SIGN = 1.0

# --------------------------------------------------------------------
# the FFT window: one definition for every transform, both sides
# --------------------------------------------------------------------
#
# Every FFT the pipeline takes -- range over the ADC samples, azimuth over
# the virtual array, Doppler over the TDM loops -- is Hann-windowed, and the
# measured side (targets, products, metrics) and the rendered side use the
# SAME window from here, so the two are consistent to rounding. HuPR's
# released preprocessing is unwindowed; `MM4DPS_FFT_WINDOW=none` restores
# that for an A/B and is not a production setting.
#
# Two Hann forms, chosen by what the axis is:
#
# * the sampled axes (range, Doppler) take the PERIODIC Hann,
#   w[n] = 0.5 - 0.5 cos(2 pi n / N), n = 0..N-1. Its DFT is exactly three
#   bins (0.5 at 0, -0.25 at +/-1), so a tone's windowed spectrum is
#   0.5 D(delta) - 0.25 D(delta-1) - 0.25 D(delta+1) with D the N-point
#   Dirichlet kernel -- the renderer's range PSF (`rasterizer.hann_psf`)
#   verbatim, and the closed form the Doppler planes will use. The
#   symmetric `np.hanning(N)` is NOT this window (cos(2 pi n/(N-1))) and
#   would leave a 1/N mismatch between the two sides.
# * the array axis (azimuth; elevation when formed) takes the symmetric Hann
#   without its zero end points, `np.hanning(N + 2)[1:-1]`: an aperture of
#   8 (HuPR) or 86 (cascade) elements cannot afford to discard an element,
#   and no DFT identity is needed there -- the window is a plain per-element
#   multiply applied identically on both sides.
FFT_WINDOW_ENV = "MM4DPS_FFT_WINDOW"


def fft_window_kind() -> str:
    """``"hann"`` (the pipeline's window) or ``"none"`` (the A/B override)."""
    import os
    kind = os.environ.get(FFT_WINDOW_ENV, "hann")
    if kind not in ("hann", "none"):
        raise ValueError(f"{FFT_WINDOW_ENV} must be 'hann' or 'none', got {kind!r}")
    return kind


def hann_periodic(n: int) -> np.ndarray:
    """The periodic Hann window of length ``n``, float64."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / float(n))


def hann_array(n: int) -> np.ndarray:
    """The symmetric Hann over ``n`` array elements, no zero end points, float64."""
    return np.hanning(n + 2)[1:-1]


ARRAY_WINDOW_ENV = "MM4DPS_ARRAY_WINDOW"


def array_window_kind() -> str:
    """The ARRAY axes' window (azimuth, elevation): ``"hann"`` (default) or ``"none"``.

    The downstream consumers (HuPR's released preprocessing, RT-Pose's HRRadarPose adapter) beamform without a taper,
    and a coherent extended reflector -- a rendered wall -- has an unwindowed response the Hann view does not show
    (RT-Pose 176: the far wall's unwindowed peak 9.8x the recording's where the windowed view reads 0.97x). The velocity
    trainer sets this for its own process (`--array-window`, docs/STATIC_SEQS_NEXT.md 1d); the range and Doppler
    windows are the pipeline's regardless.
    """
    import os
    kind = os.environ.get(ARRAY_WINDOW_ENV, "hann")
    if kind not in ("hann", "none"):
        raise ValueError(f"{ARRAY_WINDOW_ENV} must be 'hann' or 'none', got {kind!r}")
    return kind


class array_window:
    """``with array_window("none"):`` -- the array taper for the block, restored after (the transform caches are keyed on it)."""
    def __init__(self, kind: str) -> None:
        self.kind, self.previous = kind, None
    def __enter__(self):
        import os
        self.previous = os.environ.get(ARRAY_WINDOW_ENV); os.environ[ARRAY_WINDOW_ENV] = self.kind
        return self
    def __exit__(self, *exc):
        import os
        if self.previous is None:
            os.environ.pop(ARRAY_WINDOW_ENV, None)
        else:
            os.environ[ARRAY_WINDOW_ENV] = self.previous
        return False


def window_cache_key() -> str:
    """What the transform caches key their windows on: both kinds."""
    return f"{fft_window_kind()}/{array_window_kind()}"


def fft_window(axis: str, n: int) -> np.ndarray | None:
    """The window for one transform, or None when windows are off.

    ``axis`` is ``"range"``, ``"doppler"`` (periodic Hann) or ``"azimuth"`` /
    ``"elevation"`` (array Hann, off under `array_window_kind() == "none"`).
    """
    if fft_window_kind() == "none":
        return None
    if axis in ("range", "doppler"):
        return hann_periodic(n)
    if axis in ("azimuth", "elevation"):
        return None if array_window_kind() == "none" else hann_array(n)
    raise ValueError(f"unknown FFT axis {axis!r}")


def apply_fft_window(values, axis: str, dim: int):
    """``values`` windowed along ``dim`` for the FFT over ``axis`` (numpy or torch, any dtype)."""
    window = fft_window(axis, values.shape[dim])
    if window is None:
        return values
    shape = [1] * values.ndim
    shape[dim] = -1
    if isinstance(values, np.ndarray):
        return values * window.reshape(shape).astype(
            np.float64 if values.dtype in (np.complex128, np.float64) else np.float32)
    dtype = torch.float64 if values.dtype in (torch.complex128, torch.float64) else torch.float32
    return values * torch.from_numpy(window).to(device=values.device, dtype=dtype).reshape(shape)


def window_power_kernel(axis: str) -> np.ndarray:
    """The 3-tap kernel the window applies to a stationary process's power spectrum.

    A wide-sense-stationary signal with power spectrum ``P`` has, after the
    periodic Hann, the expected windowed power ``P * |H|^2 / sum(|H|^2)`` with
    ``|H|^2 = (0.0625, 0.25, 0.0625)`` at bins (-1, 0, +1) -- the calibrated
    noise envelopes describe the process and are smoothed by this before they
    are added to windowed rendered power. The identity when windows are off.
    """
    if fft_window(axis, 4) is None:
        return np.array([0.0, 1.0, 0.0])
    return np.array([1.0, 4.0, 1.0]) / 6.0


def range_window_profiles(profiles, dim: int = -1):
    """Range profiles of an unwindowed process -> the same process seen through the range window.

    A range profile is the FFT of ADC samples; windowing the samples with
    the periodic Hann is, in the profile domain, the circular 3-tap
    ``0.5 X[k] - 0.25 X[k-1] - 0.25 X[k+1]`` (the window's three-bin DFT,
    convolved). Exact for any draw, so a process synthesised per range bin
    -- the diffuse floor, the receiver nuisances -- gets the same window
    the scene's PSF and the measured ADC carry. The identity when windows
    are off. numpy or torch, along ``dim``.
    """
    if fft_window("range", 4) is None:
        return profiles
    if isinstance(profiles, np.ndarray):
        return (0.5 * profiles - 0.25 * np.roll(profiles, 1, axis=dim)
                - 0.25 * np.roll(profiles, -1, axis=dim))
    return (0.5 * profiles - 0.25 * torch.roll(profiles, 1, dims=dim)
            - 0.25 * torch.roll(profiles, -1, dims=dim))


def unwindowed_range_reference(profiles, quantile: float) -> "torch.Tensor":
    """A scene level the calibrated nuisance scales were fitted against: the magnitude quantile of the UNWINDOWED profile.

    The nuisance and floor seeds define their level relative to the 0.999
    quantile of ``|background profile|`` as the renderer produced it before
    windows. The rendered profile is now the window's response, whose peak
    magnitude for a resolved target is the window's coherent gain,
    ``mean(w) = 0.5`` for the periodic Hann, times the unwindowed peak -- so
    the reference is the windowed quantile divided by that gain. (Dividing
    the window out exactly in the ADC domain was measured and rejected: the
    window is 1.5e-4 at sample 1, and amplifying the renderer's fp32 atomic
    rounding by 6,600x there moved the rendered products by 1e-5 run to
    run, against a 7e-7 floor.) The plain quantile when windows are off.
    """
    magnitude = profiles.abs()
    window = fft_window("range", profiles.shape[-1])
    gain = 1.0 if window is None else float(window.mean())
    return torch.quantile(magnitude, quantile) / gain


def windowed_process_envelope(amplitude: np.ndarray) -> np.ndarray:
    """A ``(radars, doppler, range)`` amplitude envelope of a process -> its expected windowed amplitude.

    Power is smoothed circularly by `window_power_kernel` along the Doppler
    axis (the fftshifted layout is circular) and along range, then the square
    root is taken. The identity when windows are off.
    """
    power = np.asarray(amplitude, dtype=np.float64) ** 2
    for axis, name in ((1, "doppler"), (2, "range")):
        kernel = window_power_kernel(name)
        power = (kernel[1] * power
                 + kernel[0] * np.roll(power, 1, axis=axis)
                 + kernel[2] * np.roll(power, -1, axis=axis))
    return np.sqrt(power).astype(np.float32)


JOINT_TO_BODY_REGION = np.array(
    [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4], dtype=np.int64,
)
BODY_REGION_NAMES = ("leg_A", "leg_B", "torso_head", "arm_A", "arm_B")
OFFICIAL_CENTRAL_DOPPLER_SLICE = slice(24, 40)






def offset_frame_index(frame: int, offset: int, frame_count: int = 600) -> int:
    """Apply an integer temporal offset with explicit sequence-boundary clamping."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if not 0 <= frame < frame_count:
        raise ValueError("frame index is outside the sequence")
    return int(np.clip(frame + offset, 0, frame_count - 1))


def interpolate_frame_values(values: np.ndarray, frame: int, offset_frames: float) -> np.ndarray:
    """Linearly sample a frame-major trajectory at a fractional frame offset."""
    if values.ndim < 1 or len(values) == 0:
        raise ValueError("values must have a non-empty frame dimension")
    if not 0 <= frame < len(values):
        raise ValueError("frame index is outside the sequence")
    position = float(np.clip(frame + offset_frames, 0.0, len(values) - 1.0))
    lower = int(np.floor(position))
    upper = min(lower + 1, len(values) - 1)
    alpha = position - lower
    return ((1.0 - alpha) * values[lower] + alpha * values[upper]).astype(
        values.dtype, copy=False,
    )




def smooth_joint_velocities(
    joints_xyz: np.ndarray,
    fps: float = 10.0,
    window: int = 11,
    polynomial_order: int = 2,
    method: str = "savgol",
    cutoff_hz: float = 3.5,
    rolloff_hz: float = 1.0,
) -> np.ndarray:
    """Differentiate metric 3-D joints.

    ``savgol`` is the historical default: a 1.1 s Savitzky-Golay window, which
    is heavy smoothing for a signal whose velocity energy peaks at 2-5 Hz. It
    retains about 5% at 2 Hz, roughly 10 dB less radial velocity than a plain
    central difference. ``central`` and ``fourth-order`` widen the band; note
    that above about 3.5 Hz the joint data is largely detection jitter rather
    than motion, so a wider band buys noise as well as signal.
    """
    from scipy.signal import savgol_filter

    if joints_xyz.ndim != 3 or joints_xyz.shape[1:] != (14, 3):
        raise ValueError(f"expected (frames,14,3) joints, got {joints_xyz.shape}")
    if not np.isfinite(joints_xyz).all():
        raise ValueError("joint trajectory contains non-finite coordinates")
    if window % 2 != 1 or window <= polynomial_order or window > len(joints_xyz):
        raise ValueError("Savitzky-Golay window must be odd, > order, and <= frame count")
    if method == "savgol":
        return savgol_filter(
            joints_xyz, window, polynomial_order, deriv=1, delta=1.0 / fps,
            axis=0, mode="interp",
        ).astype(np.float32)
    if method == "central":
        return np.gradient(
            joints_xyz.astype(np.float64), 1.0 / fps, axis=0,
        ).astype(np.float32)
    if method == "band-limited":
        return band_limited_joint_velocities(
            joints_xyz, fps=fps, cutoff_hz=cutoff_hz, rolloff_hz=rolloff_hz,
        )
    if method != "fourth-order":
        raise ValueError(
            "method must be savgol, central, fourth-order or band-limited"
        )
    values = joints_xyz.astype(np.float64)
    velocity = np.gradient(values, 1.0 / fps, axis=0)
    if len(values) >= 5:
        # Fourth-order central difference in the interior; the two-point
        # estimate already computed stands in at the edges.
        velocity[2:-2] = (
            values[:-4] - 8.0 * values[1:-3] + 8.0 * values[3:-1] - values[4:]
        ) / (12.0 / fps)
    return velocity.astype(np.float32)


def band_limited_joint_velocities(
    joints_xyz: np.ndarray,
    fps: float = 10.0,
    cutoff_hz: float = 3.5,
    rolloff_hz: float = 1.0,
) -> np.ndarray:
    """Differentiate in frequency, passing real motion and rejecting jitter.

    The two existing choices both miss, in opposite directions. Savitzky-Golay
    over 1.1 s keeps under 1% of a limb-swinging sequence's velocity energy,
    because that energy sits at 2-3.5 Hz. A fourth-order stencil passes 2-3.5 Hz
    almost intact but also passes everything above it, and above about 3.5 Hz the
    10 Hz joint annotations are dominated by detection jitter rather than motion;
    measured, that trade cost sequences 102 and 256 half their Doppler centroid
    correlation.

    So differentiate exactly (multiply by j*omega) and apply a raised-cosine
    stop above ``cutoff_hz``, which keeps the band where the motion lives and
    drops the band where the jitter lives.
    """
    values = np.asarray(joints_xyz, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (14, 3):
        raise ValueError(f"expected (frames,14,3) joints, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("joint trajectory contains non-finite coordinates")
    if not 0.0 < cutoff_hz <= fps / 2.0:
        raise ValueError("cutoff must be positive and at or below Nyquist")
    if rolloff_hz < 0.0:
        raise ValueError("rolloff must be nonnegative")

    frames = len(values)
    # Reflect the ends so the transform does not treat the walk as periodic.
    padded = np.concatenate((values[::-1], values, values[::-1]), axis=0)
    frequencies = np.fft.rfftfreq(len(padded), d=1.0 / fps)
    response = np.zeros_like(frequencies)
    passband = frequencies <= cutoff_hz
    response[passband] = 1.0
    if rolloff_hz > 0.0:
        skirt = (frequencies > cutoff_hz) & (frequencies <= cutoff_hz + rolloff_hz)
        response[skirt] = 0.5 * (
            1.0 + np.cos(np.pi * (frequencies[skirt] - cutoff_hz) / rolloff_hz)
        )
    kernel = (2j * np.pi * frequencies * response)[:, None, None]
    velocity = np.fft.irfft(
        np.fft.rfft(padded, axis=0) * kernel, n=len(padded), axis=0,
    )
    return velocity[frames : 2 * frames].astype(np.float32)


def nearest_joint_labels(points_xyz: np.ndarray, joints_xyz: np.ndarray) -> np.ndarray:
    """Assign each surface point to its nearest HuPR skeleton joint."""
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    if joints_xyz.shape != (14, 3):
        raise ValueError("joints must have shape (14,3)")
    squared = ((points_xyz[:, None, :] - joints_xyz[None, :, :]) ** 2).sum(axis=2)
    return squared.argmin(axis=1).astype(np.int64)


def soft_joint_weights(
    points_xyz: np.ndarray,
    joints_xyz: np.ndarray,
    temperature_m: float = 0.25,
    topk: int = 3,
) -> np.ndarray:
    """Return sparse, normalized Gaussian skinning weights for 14 joints.

    Only the nearest ``topk`` joints receive nonzero weight. This avoids the
    discontinuous velocity boundaries produced by hard nearest-joint labels
    while keeping the factorized-rendering cost proportional to ``topk``.
    Every row sums to one, so soft skinning preserves the total point area.
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    if joints_xyz.shape != (14, 3):
        raise ValueError("joints must have shape (14,3)")
    if not np.isfinite(points_xyz).all() or not np.isfinite(joints_xyz).all():
        raise ValueError("points and joints must be finite")
    if temperature_m <= 0:
        raise ValueError("temperature must be positive")
    if not 1 <= topk <= 14:
        raise ValueError("topk must lie in [1,14]")

    squared = ((points_xyz[:, None, :] - joints_xyz[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argpartition(squared, kth=topk - 1, axis=1)[:, :topk]
    selected_squared = np.take_along_axis(squared, nearest, axis=1)
    logits = -selected_squared / (2.0 * temperature_m ** 2)
    logits -= logits.max(axis=1, keepdims=True)
    selected_weights = np.exp(logits)
    selected_weights /= selected_weights.sum(axis=1, keepdims=True)
    weights = np.zeros_like(squared, dtype=np.float32)
    np.put_along_axis(weights, nearest, selected_weights.astype(np.float32), axis=1)
    return weights








def soft_joint_weights_torch(points_xyz, joints_xyz, temperature_m: float = 0.25, topk: int = 3):
    """:func:`soft_joint_weights` on the device: ``(N, 3)``, ``(14, 3)`` -> ``(N, 14)`` float32.

    The same arithmetic in float32 (the numpy form computed the logits in
    the points' dtype and stored float32); the nearest `topk` by a full
    sort rather than a partition, which picks the same set.
    """
    import torch
    if joints_xyz.shape != (14, 3):
        raise ValueError("joints must have shape (14,3)")
    squared = ((points_xyz[:, None, :] - joints_xyz[None, :, :]) ** 2).sum(dim=2)
    nearest = squared.topk(topk, dim=1, largest=False).indices
    logits = -squared.gather(1, nearest) / (2.0 * temperature_m ** 2)
    logits = logits - logits.amax(dim=1, keepdim=True)
    selected = torch.exp(logits)
    selected = selected / selected.sum(dim=1, keepdim=True)
    weights = torch.zeros_like(squared, dtype=torch.float32)
    return weights.scatter(1, nearest, selected.to(torch.float32))


def person_range_angle_mask_torch(
    points_xyz,
    radar_center_xyz,
    array_axis_xyz,
    range_bin_spacing_m: float,
    n_range: int | None = None,
    n_angle: int | None = None,
    range_padding_bins: int = 4,
    angle_padding_bins: int = 3,
    include_mirrored_angle: bool = True,
):
    """:func:`person_range_angle_mask` on the device, in one scatter.

    The numpy form runs a Python loop of 2 x 9 x 7 = 126 boolean scatters
    over the whole cloud, which made it the largest remaining CPU stage in
    the render once the nuisance moved to the GPU. This broadcasts the
    padding offsets into the index arithmetic and writes the mask once.

    The arithmetic is integer after ``round``, so the two agree exactly
    unless a point's spatial sine lands within an ULP of a half-bin
    boundary. The test that pinned them against each other on random clouds went
    with ``tests/`` in the deployment strip. The mask feeds the run-summary correlations
    only -- no saved product is derived from it.
    """
    import torch

    beam = current_dataset().radars[0]
    n_range = beam.samples if n_range is None else n_range
    n_angle = beam.n_angle if n_angle is None else n_angle
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    device = points_xyz.device
    center = torch.as_tensor(
        radar_center_xyz, dtype=points_xyz.dtype, device=device,
    )
    axis = torch.as_tensor(
        array_axis_xyz, dtype=points_xyz.dtype, device=device,
    )
    axis = axis / axis.norm().clamp_min(1e-12)
    delta = points_xyz - center[None, :]
    ranges = delta.norm(dim=1)
    spatial_sine = ((delta @ axis) / ranges.clamp_min(1e-8)).clamp(-1.0, 1.0)
    range_index = torch.round(ranges / range_bin_spacing_m).long()
    angle_indices = [torch.round(n_angle / 2 + n_angle / 2 * spatial_sine).long()]
    if include_mirrored_angle:
        angle_indices.append(
            torch.round(n_angle / 2 - n_angle / 2 * spatial_sine).long()
        )
    range_offsets = torch.arange(
        -range_padding_bins, range_padding_bins + 1, device=device,
    )
    angle_offsets = torch.arange(
        -angle_padding_bins, angle_padding_bins + 1, device=device,
    )
    mask = torch.zeros(n_range * n_angle, dtype=torch.bool, device=device)
    for angle_index in angle_indices:
        rr = range_index[:, None, None] + range_offsets[None, :, None]
        aa = angle_index[:, None, None] + angle_offsets[None, None, :]
        good = (rr >= 0) & (rr < n_range) & (aa >= 0) & (aa < n_angle)
        flat = (rr * n_angle + aa)[good]
        mask.index_fill_(0, flat.reshape(-1), True)
    mask = mask.reshape(n_range, n_angle)
    if not bool(mask.any()):
        raise ValueError("person geometry produced an empty range-angle mask")
    return mask


def person_range_angle_masks_torch(
    points_xyz,
    radar_center_xyz,
    array_axis_xyz,
    range_bin_spacing_m: float,
    n_range: int | None = None,
    n_angle: int | None = None,
    range_padding_bins: int = 4,
    angle_padding_bins: int = 3,
    include_mirrored_angle: bool = True,
):
    """:func:`person_range_angle_mask_torch` for every frame at once: ``(F, N, 3)`` -> ``(F, range, angle)``.

    The same index arithmetic with the frame folded into the flat index,
    so each frame's mask is bitwise the per-frame call's.
    """
    import torch

    beam = current_dataset().radars[0]
    n_range = beam.samples if n_range is None else n_range
    n_angle = beam.n_angle if n_angle is None else n_angle
    if points_xyz.ndim != 3 or points_xyz.shape[2] != 3:
        raise ValueError("points must have shape (F,N,3)")
    device = points_xyz.device
    center = torch.as_tensor(radar_center_xyz, dtype=points_xyz.dtype, device=device)
    axis = torch.as_tensor(array_axis_xyz, dtype=points_xyz.dtype, device=device)
    axis = axis / axis.norm().clamp_min(1e-12)
    delta = points_xyz - center[None, None, :]
    ranges = delta.norm(dim=2)
    spatial_sine = ((delta @ axis) / ranges.clamp_min(1e-8)).clamp(-1.0, 1.0)
    range_index = torch.round(ranges / range_bin_spacing_m).long()
    angle_indices = [torch.round(n_angle / 2 + n_angle / 2 * spatial_sine).long()]
    if include_mirrored_angle:
        angle_indices.append(torch.round(n_angle / 2 - n_angle / 2 * spatial_sine).long())
    range_offsets = torch.arange(-range_padding_bins, range_padding_bins + 1, device=device)
    angle_offsets = torch.arange(-angle_padding_bins, angle_padding_bins + 1, device=device)
    frames = len(points_xyz)
    frame_base = (torch.arange(frames, device=device) * (n_range * n_angle))[:, None, None, None]
    mask = torch.zeros(frames * n_range * n_angle, dtype=torch.bool, device=device)
    for angle_index in angle_indices:
        rr = range_index[:, :, None, None] + range_offsets[None, None, :, None]
        aa = angle_index[:, :, None, None] + angle_offsets[None, None, None, :]
        good = (rr >= 0) & (rr < n_range) & (aa >= 0) & (aa < n_angle)
        flat = (frame_base + rr * n_angle + aa)[good]
        mask.index_fill_(0, flat.reshape(-1), True)
    mask = mask.reshape(frames, n_range, n_angle)
    if not bool(mask.flatten(1).any(dim=1).all()):
        raise ValueError("person geometry produced an empty range-angle mask")
    return mask


def doppler_bin_mps(radar: Radar | None = None) -> float:
    """Radial velocity per Doppler bin: lambda / (2 * loops * loop period)."""
    radar = radar or current_dataset().radars[0]
    return radar.wavelength_m / (2.0 * radar.loops * radar.loop_period_s)


#: HuPR's central 16 bins in VELOCITY: bins 24..39 at 0.1408 m/s a bin, i.e.
#: -1.13 .. +0.99 m/s. That is what the band means physically -- the
#: velocities a walking person's body occupies -- and it is what another
#: radar keeps. The cascade samples Doppler 3.6x finer (0.039 m/s a bin, a
#: 780 us loop against HuPR's 216 us), so the same 16 bins there would be
#: +/-0.31 m/s: the walking person sits OUTSIDE them, the micro-Doppler term
#: sees an empty band, and the trainer empties the band by smearing the
#: person over Doppler (RT-Pose seq 183, 400 epochs: rendered spread 15 bins
#: against 3.9 measured, centroid r +0.15). Measured on seq 183: the person
#: at +0.57 m/s sits at bin 45, at -0.55 m/s at bin 18.
MOTION_BAND_MPS = ((24 - 32) * 0.140828, (39 - 32) * 0.140828)


def official_motion_doppler_mask(
    exclude_zero_width: int = 1,
    loops: int | None = None,
    radar: Radar | None = None,
) -> np.ndarray:
    """The motion Doppler bins: HuPR's released central 16, or the same velocity band on another radar.

    Zero and its neighbours are excluded either way.
    """
    radar = radar or current_dataset().radars[0]
    loops = radar.loops if loops is None else loops
    if loops != 64:
        raise ValueError("released HuPR preprocessing is defined for 64 loops")
    mask = np.zeros(loops, dtype=bool)
    center = loops // 2
    if radar.name in HUPR.names:
        mask[OFFICIAL_CENTRAL_DOPPLER_SLICE] = True
    else:
        per_bin = doppler_bin_mps(radar)
        lo = max(0, center + int(np.floor(MOTION_BAND_MPS[0] / per_bin)))
        hi = min(loops, center + int(np.ceil(MOTION_BAND_MPS[1] / per_bin)) + 1)
        mask[lo:hi] = True
    mask[center - exclude_zero_width:center + exclude_zero_width + 1] = False
    return mask


def tdm_sample_times(
    loops: int | None = None,
    chirp_duration_s: float | None = None,
    radar: Radar | None = None,
) -> np.ndarray:
    """Return acquisition times (loop, physical TX) including each TX's intra-loop offset.

    TX ``t`` fires in slot ``radar.tx_order.index(t)`` of every loop. HuPR's
    slot order is the identity; the cascade's is 12 down to 1.
    """
    radar = radar or current_dataset().radars[0]
    loops = radar.loops if loops is None else loops
    chirp_duration_s = radar.chirp_duration_s if chirp_duration_s is None else chirp_duration_s
    slot_of_tx = np.asarray(radar.slot_of_tx(), dtype=np.float64)
    loop_start = (np.arange(loops, dtype=np.float64)[:, None]
                  * (radar.loop_slots or radar.n_tx) * chirp_duration_s)
    return loop_start + slot_of_tx[None, :] * chirp_duration_s


def joint_radial_velocities(
    joints_xyz: np.ndarray,
    joint_velocities_xyz: np.ndarray,
    radar_center_xyz: np.ndarray,
) -> np.ndarray:
    """Project 3-D joint velocity onto radar line of sight (away is positive)."""
    delta = joints_xyz - np.asarray(radar_center_xyz, dtype=np.float32)[None, :]
    direction = delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-8)
    return (joint_velocities_xyz * direction).sum(axis=1).astype(np.float32)


def joint_bistatic_path_rates(
    joints_xyz: np.ndarray,
    joint_velocities_xyz: np.ndarray,
    tx_positions_xyz: np.ndarray,
    rx_positions_xyz: np.ndarray,
) -> np.ndarray:
    """Return d/dt(|joint-TX|+|joint-RX|), shape (14, n_tx, n_rx)."""
    if joints_xyz.shape != (14, 3) or joint_velocities_xyz.shape != (14, 3):
        raise ValueError("joints and velocities must have shape (14,3)")
    if tx_positions_xyz.ndim != 2 or tx_positions_xyz.shape[1] != 3 \
            or rx_positions_xyz.ndim != 2 or rx_positions_xyz.shape[1] != 3:
        raise ValueError("TX/RX positions must have shape (n_tx,3)/(n_rx,3)")
    tx_delta = joints_xyz[:, None, :] - tx_positions_xyz[None, :, :]
    rx_delta = joints_xyz[:, None, :] - rx_positions_xyz[None, :, :]
    tx_direction = tx_delta / np.maximum(np.linalg.norm(tx_delta, axis=2, keepdims=True), 1e-8)
    rx_direction = rx_delta / np.maximum(np.linalg.norm(rx_delta, axis=2, keepdims=True), 1e-8)
    tx_rate = (joint_velocities_xyz[:, None, :] * tx_direction).sum(axis=2)
    rx_rate = (joint_velocities_xyz[:, None, :] * rx_direction).sum(axis=2)
    return (tx_rate[:, :, None] + rx_rate[:, None, :]).astype(np.float32)


def soft_surface_velocities(
    skinning_weights: torch.Tensor,
    joint_velocities_xyz: torch.Tensor,
    joint_scales: torch.Tensor | None = None,
    global_velocity_transform: torch.Tensor | None = None,
    joint_velocity_residual_xyz: torch.Tensor | None = None,
) -> torch.Tensor:
    """Blend scaled joint velocities into one local velocity per surface point.

    The optional residual is a learned per-joint velocity correction in m/s
    added to the camera-derived velocity *before* the scales, so the scales
    keep their meaning as multipliers of a physical velocity:
    ``v = scales * (v_camera + residual)``.
    """
    if skinning_weights.ndim != 2 or skinning_weights.shape[1] != 14:
        raise ValueError("skinning weights must have shape (points,14)")
    if joint_velocities_xyz.shape != (14, 3):
        raise ValueError("joint velocities must have shape (14,3)")
    if joint_velocity_residual_xyz is not None:
        if joint_velocity_residual_xyz.shape != (14, 3):
            raise ValueError("joint velocity residual must have shape (14,3)")
        joint_velocities_xyz = joint_velocities_xyz + joint_velocity_residual_xyz
    if joint_scales is None:
        joint_scales = torch.ones(
            14, dtype=joint_velocities_xyz.dtype, device=joint_velocities_xyz.device,
        )
    if skinning_weights.device != joint_velocities_xyz.device:
        raise ValueError("weights and velocities must be on the same device")
    if joint_scales.shape == (14,):
        scaled = joint_velocities_xyz * joint_scales[:, None]
    elif joint_scales.shape == (14, 3):
        scaled = joint_velocities_xyz * joint_scales
    else:
        raise ValueError("joint scales must have shape (14,) or (14,3)")
    if global_velocity_transform is not None:
        if global_velocity_transform.shape != (3, 3):
            raise ValueError("global velocity transform must have shape (3,3)")
        scaled = scaled @ global_velocity_transform.T
    return skinning_weights @ scaled


def surface_bistatic_path_rates(
    points_xyz: torch.Tensor,
    point_velocities_xyz: torch.Tensor,
    tx_positions_xyz: torch.Tensor,
    rx_positions_xyz: torch.Tensor,
) -> torch.Tensor:
    """Differentiable per-point d/dt(|point-TX|+|point-RX|)."""
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    if point_velocities_xyz.shape != points_xyz.shape:
        raise ValueError("point velocities must match points")
    if tx_positions_xyz.ndim != 2 or tx_positions_xyz.shape[1] != 3 \
            or rx_positions_xyz.ndim != 2 or rx_positions_xyz.shape[1] != 3:
        raise ValueError("TX/RX positions must have shape (n_tx,3)/(n_rx,3)")
    tx_delta = points_xyz[:, None, :] - tx_positions_xyz[None, :, :]
    rx_delta = points_xyz[:, None, :] - rx_positions_xyz[None, :, :]
    tx_direction = tx_delta / torch.linalg.vector_norm(
        tx_delta, dim=2, keepdim=True,
    ).clamp_min(1e-8)
    rx_direction = rx_delta / torch.linalg.vector_norm(
        rx_delta, dim=2, keepdim=True,
    ).clamp_min(1e-8)
    tx_rate = (point_velocities_xyz[:, None, :] * tx_direction).sum(dim=2)
    rx_rate = (point_velocities_xyz[:, None, :] * rx_direction).sum(dim=2)
    return tx_rate[:, :, None] + rx_rate[:, None, :]


def surface_bistatic_path_rates_per_point(
    points_xyz: torch.Tensor,
    point_velocities_xyz: torch.Tensor,
    tx_positions_xyz: torch.Tensor,
    rx_positions_xyz: torch.Tensor,
) -> torch.Tensor:
    """``surface_bistatic_path_rates`` with a per-point array geometry.

    Same arithmetic in the same order; the only difference is that the TX/RX
    positions carry a leading point axis, so one call covers points belonging
    to different radars (and different frames) at once. A ``(1, n_tx, 3)``
    argument reproduces the shared-array function bitwise. Kept beside it
    rather than folded into it, because that one is a pinned reference.
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    if point_velocities_xyz.shape != points_xyz.shape:
        raise ValueError("point velocities must match points")
    if tx_positions_xyz.ndim != 3 or tx_positions_xyz.shape[-1] != 3:
        raise ValueError("per-point TX positions must have shape (N,n_tx,3)")
    if rx_positions_xyz.ndim != 3 or rx_positions_xyz.shape[-1] != 3:
        raise ValueError("per-point RX positions must have shape (N,n_rx,3)")
    tx_delta = points_xyz[:, None, :] - tx_positions_xyz
    rx_delta = points_xyz[:, None, :] - rx_positions_xyz
    tx_direction = tx_delta / torch.linalg.vector_norm(
        tx_delta, dim=2, keepdim=True,
    ).clamp_min(1e-8)
    rx_direction = rx_delta / torch.linalg.vector_norm(
        rx_delta, dim=2, keepdim=True,
    ).clamp_min(1e-8)
    tx_rate = (point_velocities_xyz[:, None, :] * tx_direction).sum(dim=2)
    rx_rate = (point_velocities_xyz[:, None, :] * rx_direction).sum(dim=2)
    return tx_rate[:, :, None] + rx_rate[:, None, :]


def assemble_static_profile(
    splat_components: dict[str, torch.Tensor | tuple[int, int, int]],
    point_gains: torch.Tensor,
) -> torch.Tensor:
    """Coherent static profile from splat components with per-point gains.

    The factorized rasterizer's per-point splats are linear in a per-point
    scalar, so a static scene renders as one gain-weighted scatter — the
    cached-splat form of the background. Gradients flow to ``point_gains``;
    positions and phases stay frozen inside the cached components.
    """
    real = splat_components["contrib_real"]
    imag = splat_components["contrib_imag"]
    flat_indices = splat_components["flat_indices"].long().reshape(-1)
    weights = point_gains[splat_components["point_indices"].long()][None, :]
    tx, rx, samples = splat_components["output_shape"]
    out_real = real.new_zeros(tx * rx * samples)
    out_imag = imag.new_zeros(tx * rx * samples)
    out_real.index_add_(0, flat_indices, (real * weights).reshape(-1))
    out_imag.index_add_(0, flat_indices, (imag * weights).reshape(-1))
    return torch.complex(out_real, out_imag).reshape(tx, rx, samples)




def doppler_plane_weights(
    doppler_bins,
    loops: int = TDM_LOOPS,
    device=None,
    static_anchor: str = "loop0",
) -> torch.Tensor:
    """The (1 + len(bins), loops) linear map from loop planes to what is read.

    Row 0 is the static-anchor plane the RA term reads. With the default
    ``static_anchor="loop0"`` it is the one-hot ``e_0``: plane 0 is loop 0
    verbatim, the raw slow-time sample (no Doppler transform). With
    ``static_anchor="dc-mean"`` it is the uniform ``1/loops`` row: plane 0 is
    the slow-time mean — the DC Doppler plane proper, carrying the same
    static information as any single loop at ~sqrt(loops) (~8x at 64) the
    SNR, since the static return repeats in every loop while noise averages
    down. Rows 1.. are the same either way: the
    fftshifted DFT rows for the requested Doppler bins, so that

        out[d] = sum_l W[d, l] * profile[l]

    reproduces ``fftshift(fft(profile, dim=loop))[bin]`` exactly.

    ``fftshift`` maps output index ``d`` to unshifted frequency
    ``(d + loops // 2) % loops``, and the angle is reduced with integer
    arithmetic before it reaches a trig function: ``d' * l`` reaches ~3,500 for
    64 loops, and evaluating ``cos`` of the un-reduced ~346 rad angle in
    float32 loses 2e-5 — measured, that is 1.98e-06 on the micro-Doppler
    marginal, ten times the nondeterminism floor the gates are written
    against. Reducing ``(d' * l) % loops`` first is exact and free.

    The DC bin is rejected: the consumer removes the slow-time mean before the
    Doppler transform, so DC carries no static return (exactly zero without a
    window; with the window, only the movers' leakage), and the rendered
    plane set is the non-DC bins. Rows 1.. are ``doppler_dft_rows``: the DFT
    row times the Doppler window, minus its mean over loops -- the mean
    removal folded in, so the rows reproduce ``fftshift(fft(window * (x -
    mean(x))))[bin]``.
    """
    bins = torch.as_tensor(doppler_bins, dtype=torch.long).reshape(-1)
    if bins.numel() == 0:
        raise ValueError("at least one Doppler bin must be requested")
    if int(bins.min()) < 0 or int(bins.max()) >= loops:
        raise ValueError(f"Doppler bins must lie in [0, {loops})")
    if len(torch.unique(bins)) != bins.numel():
        raise ValueError("Doppler bins must be distinct")
    dc = loops // 2
    if bool((bins == dc).any()):
        raise ValueError(
            f"Doppler bin {dc} is DC; the rendered planes are the non-DC bins "
            f"(mean removal makes DC the static return's residual, not a Doppler plane)")
    if static_anchor not in ("loop0", "dc-mean"):
        raise ValueError(f"unknown static anchor: {static_anchor}")
    weights = torch.zeros(1 + bins.numel(), loops, dtype=torch.complex64)
    if static_anchor == "dc-mean":
        weights[0, :] = 1.0 / float(loops)
    else:
        weights[0, 0] = 1.0
    unshifted = ((bins + dc) % loops).numpy()
    weights[1:] = torch.from_numpy(doppler_dft_rows(unshifted, loops).astype(np.complex64))
    return weights.to(device) if device is not None else weights


def dirichlet_reference(delta: torch.Tensor, loops: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``D(delta) = sum_l exp(-2 pi j delta l / L)`` and ``dD/ddelta``, float64, for any real ``delta``.

    ``D = exp(j psi) R`` with ``psi = -pi delta (L-1)/L`` and
    ``R = sin(pi delta) / sin(pi delta / L)``; near ``delta = 0 (mod L)`` R
    and R' come from the series so the 0/0 is never formed. The float64
    reference the kernels are gated against.
    """
    delta = delta.to(torch.float64)
    L = float(loops)
    # reduce to the nearest multiple of L for the series test only
    wrapped = delta - loops * torch.round(delta / loops)
    near = wrapped.abs() < 1e-3
    x = torch.where(near, torch.ones_like(delta), delta)
    sin_num, sin_den = torch.sin(math.pi * x), torch.sin(math.pi * x / L)
    R = sin_num / sin_den
    Rp = (math.pi * torch.cos(math.pi * x) * sin_den
          - (math.pi / L) * sin_num * torch.cos(math.pi * x / L)) / sin_den ** 2
    # series in the wrapped offset e (the multiple of L only flips a sign of magnitude 1)
    e = wrapped
    a = (math.pi * e) ** 2
    b = (math.pi * e / L) ** 2
    num = 1 - a / 6 + a * a / 120
    den = 1 - b / 6 + b * b / 120
    R_series = L * num / den
    dnum = math.pi * math.pi * e * (-1.0 / 3 + a / 30)
    dden = (math.pi / L) ** 2 * e * (-1.0 / 3 + b / 30)
    Rp_series = L * (dnum * den - num * dden) / den ** 2
    # sign of the (delta - e) multiple: sin(pi delta) = (-1)^(mL) sin(pi e) with L even -> +; sin(pi delta / L) = (-1)^m sin(pi e / L)
    m = torch.round((delta - e) / loops)
    sign = torch.where(m.remainder(2) == 0, 1.0, -1.0).to(torch.float64)
    R = torch.where(near, sign * R_series, R)
    Rp = torch.where(near, sign * Rp_series, Rp)
    psi = -math.pi * delta * (L - 1) / L
    phasor = torch.polar(torch.ones_like(psi), psi)
    D = phasor * R.to(torch.complex128)
    dD = phasor * (1j * (-math.pi * (L - 1) / L) * R + Rp).to(torch.complex128)
    return D, dD


def doppler_plane_amplitudes_reference(
    rates: torch.Tensor, tx: torch.Tensor, times: torch.Tensor, phase_const: float,
    bins: torch.Tensor, static_anchor: str = "loop0", closed_form: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-path Doppler-plane amplitudes and their rate gradient, float64, two ways.

    ``rates`` (paths,), ``tx`` (paths,) long, ``times`` (loops, n_tx) TDM
    sample times, ``bins`` (paths, planes) fftshifted Doppler bins with -1
    pads; returns ``A`` (paths, 1 + planes) complex128 with the static
    anchor first, and ``dA/drate``.

    ``closed_form=False`` is the sum over loops with ``doppler_dft_rows``
    (what the DFT kernels compute); ``closed_form=True`` is the Hann
    Doppler PSF: with ``nu`` the path's fractional unshifted Doppler bin,
    ``A_d = exp(j phi0) [0.5 D(d'-nu) - 0.25 D(d'-1-nu) - 0.25 D(d'+1-nu)
    - m_d D(-nu)]``, ``m_d`` the folded mean removal (0.5 at DC, -0.25 at
    +/-1, else 0), and without a window ``A_d = exp(j phi0) D(d'-nu)``.
    The two agree to float64 rounding; the gate checks that.
    """
    rates = rates.to(torch.float64); times = times.to(torch.float64)
    loops, n_tx = times.shape
    paths, planes = bins.shape
    dc = loops // 2
    keep = bins >= 0
    unshifted = ((bins.clamp_min(0) + dc) % loops).to(torch.float64)
    t0 = times[0][tx]                                  # (paths,)
    T = (times[1] - times[0])[tx]
    phi0 = phase_const * rates * t0
    omega = phase_const * rates * T
    nu = omega * loops / (2.0 * math.pi)
    e0 = torch.polar(torch.ones_like(phi0), phi0)
    windowed = fft_window("doppler", loops) is not None
    if closed_form:
        delta = unshifted - nu[:, None]                # (paths, planes) = d' - nu
        D0, dD0 = dirichlet_reference(delta, loops)
        if windowed:
            Dm, dDm = dirichlet_reference(delta - 1.0, loops)
            Dp, dDp = dirichlet_reference(delta + 1.0, loops)
            Dz, dDz = dirichlet_reference(-nu, loops)
            m = torch.zeros_like(unshifted)
            m = torch.where(unshifted == 0, torch.full_like(m, 0.5), m)
            m = torch.where((unshifted == 1) | (unshifted == loops - 1), torch.full_like(m, -0.25), m)
            body = 0.5 * D0 - 0.25 * Dm - 0.25 * Dp - m * Dz[:, None]
            dbody = 0.5 * dD0 - 0.25 * dDm - 0.25 * dDp - m * dDz[:, None]
        else:
            body, dbody = D0, dD0
        dnu = phase_const * T * loops / (2.0 * math.pi)
        A = e0[:, None] * body
        dA = e0[:, None] * (1j * (phase_const * t0)[:, None] * body - dbody * dnu[:, None])
        if static_anchor == "loop0":
            A0, dA0 = e0, 1j * phase_const * t0 * e0
        else:
            Dz, dDz = dirichlet_reference(-nu, loops)
            A0 = e0 * Dz / loops
            dA0 = e0 * (1j * phase_const * t0 * Dz - dDz * dnu) / loops
    else:
        rows = torch.from_numpy(doppler_dft_rows(np.arange(loops), loops))      # (loops, loops) complex128
        phase = phase_const * rates[:, None] * times[:, tx].T                    # (paths, loops)
        phasor = torch.polar(torch.ones_like(phase), phase)
        W = rows[unshifted.long()]                                                # (paths, planes, loops)
        A = torch.einsum("pdl,pl->pd", W, phasor)
        dA = torch.einsum("pdl,pl->pd", W, 1j * phase_const * times[:, tx].T * phasor)
        if static_anchor == "loop0":
            A0, dA0 = phasor[:, 0], 1j * phase_const * times[0][tx] * phasor[:, 0]
        else:
            A0 = phasor.mean(dim=1)
            dA0 = (1j * phase_const * times[:, tx].T * phasor).mean(dim=1)
    A = torch.where(keep, A, torch.zeros_like(A))
    dA = torch.where(keep, dA, torch.zeros_like(dA))
    return (torch.cat([A0[:, None], A], dim=1), torch.cat([dA0[:, None], dA], dim=1))


def doppler_dft_rows(unshifted_bins: np.ndarray, loops: int) -> np.ndarray:
    """The (bins, loops) complex128 map from loop samples to Doppler bins, as the consumer takes them.

    Row ``d`` reproduces ``fft(window * (x - mean(x)))[d]``: the DFT row for
    unshifted frequency ``d``, times the Doppler window, minus the row's
    mean over loops -- which is the slow-time mean removal folded in. Without
    a window the fold is exactly zero for every non-DC row (the DFT row sums
    to zero), so this is the plain DFT row the reference always used; with
    the Hann it is what makes the rendered planes agree with a measured chain
    that removes the mean BEFORE it windows. Angles come from an exactly
    reduced integer product (see ``doppler_plane_weights``).
    """
    reduced = (np.outer(np.asarray(unshifted_bins), np.arange(loops)) % loops).astype(np.float64)
    rows = np.exp(1j * (-2.0 * np.pi * reduced / float(loops)))
    window = fft_window("doppler", loops)
    if window is not None:
        rows = rows * window[None, :]
        rows = rows - rows.mean(axis=1, keepdims=True)
    return rows


def phase_ramps(
    radial_velocity_mps: torch.Tensor,
    velocity_scale: torch.Tensor | float = 1.0,
    loops: int | None = None,
    chirp_duration_s: float | None = None,
    wavelength_m: float = WAVELENGTH_M,
) -> torch.Tensor:
    """Monostatic carrier phase exp(j·SLOW_TIME_PHASE_SIGN·4π v_r t / λ).

    Shape (joint,loop,TX). See SLOW_TIME_PHASE_SIGN for why the sign is what it
    is; it is fixed against measured data rather than derived.
    """
    if radial_velocity_mps.ndim != 1:
        raise ValueError("radial velocities must be a vector")
    radar = current_dataset().radars[0]
    loops = radar.loops if loops is None else loops
    chirp_duration_s = radar.chirp_duration_s if chirp_duration_s is None else chirp_duration_s
    times = torch.as_tensor(
        tdm_sample_times(loops, chirp_duration_s),
        dtype=radial_velocity_mps.dtype, device=radial_velocity_mps.device,
    )
    phase = (
        SLOW_TIME_PHASE_SIGN * 4.0 * math.pi * radial_velocity_mps[:, None, None]
        * velocity_scale * times / wavelength_m
    )
    return torch.polar(torch.ones_like(phase), phase)


def assemble_slow_time_profiles(
    background_profile: torch.Tensor,
    group_profiles: torch.Tensor,
    group_radial_velocity_mps: torch.Tensor,
    velocity_scale: torch.Tensor | float = 1.0,
    loops: int | None = None,
) -> torch.Tensor:
    """Sum static background and skeleton-group profiles across slow time.

    ``background_profile`` is complex (TX,RX,K); ``group_profiles`` is complex
    (group,TX,RX,K). Range migration and BSDF amplitude change over the 13.8 ms
    capture are neglected, while the carrier phase uses every physical TX time.
    """
    if background_profile.ndim != 3:
        raise ValueError("background profile must have shape (tx,rx,K)")
    if group_profiles.ndim != 4 or group_profiles.shape[1:3] != background_profile.shape[:2]:
        raise ValueError("group profiles must have shape (group,tx,rx,K)")
    if len(group_profiles) != len(group_radial_velocity_mps):
        raise ValueError("one radial velocity is required per group")
    ramps = phase_ramps(group_radial_velocity_mps, velocity_scale, loops=loops)
    dynamic = torch.einsum("glt,gtrk->ltrk", ramps, group_profiles)
    return background_profile[None, :, :, :] + dynamic


def assemble_bistatic_slow_time_profiles(
    background_profile: torch.Tensor,
    group_profiles: torch.Tensor,
    group_path_rate_mps: torch.Tensor,
    velocity_scale: torch.Tensor | float = 1.0,
    loops: int | None = None,
    chirp_duration_s: float | None = None,
    wavelength_m: float = WAVELENGTH_M,
) -> torch.Tensor:
    """Assemble slow time using exact per-group, per-TX/RX path derivatives."""
    radar = current_dataset().radars[0]
    loops = radar.loops if loops is None else loops
    chirp_duration_s = radar.chirp_duration_s if chirp_duration_s is None else chirp_duration_s
    if background_profile.ndim != 3:
        raise ValueError("background profile must have shape (tx,rx,K)")
    if group_profiles.ndim != 4 or group_profiles.shape[1:3] != background_profile.shape[:2]:
        raise ValueError("group profiles must have shape (group,tx,rx,K)")
    if group_path_rate_mps.shape != (len(group_profiles), *background_profile.shape[:2]):
        raise ValueError("path rates must have shape (group,tx,rx)")
    times = torch.as_tensor(
        tdm_sample_times(loops, chirp_duration_s),
        dtype=group_path_rate_mps.dtype, device=group_path_rate_mps.device,
    )
    phase = (
        SLOW_TIME_PHASE_SIGN * 2.0 * math.pi * group_path_rate_mps[:, None, :, :]
        * velocity_scale * times[None, :, :, None] / wavelength_m
    )
    ramps = torch.polar(torch.ones_like(phase), phase)
    dynamic = torch.einsum("gltr,gtrk->ltrk", ramps, group_profiles)
    return background_profile[None, :, :, :] + dynamic


def capture_summary(radar: Radar | None = None) -> dict:
    radar = radar or current_dataset().radars[0]
    return {
        "physical_chirp_duration_us": radar.chirp_duration_s * 1e6,
        "tdm_transmitters": radar.n_tx,
        "slow_time_loops": radar.loops,
        "same_tx_loop_period_us": radar.loop_period_s * 1e6,
        "last_loop_start_ms": (radar.loops - 1) * radar.loop_period_s * 1e3,
        "full_192_chirp_capture_ms": radar.loops * radar.loop_period_s * 1e3,
        "carrier_hz_assumption": radar.carrier_hz,
        "wavelength_m": radar.wavelength_m,
        "timing_provenance": (
            "HuPR Table 1 reports 72 us chirp duration; released code proves 3-TX TDM "
            "and 64 slow-time loops. Additional inter-chirp idle time is unpublished."
        ),
    }


def load_trajectories(
    sequence_dir: Path, differentiator: str = "savgol",
) -> tuple[np.ndarray, np.ndarray]:
    """Joints and their velocities.

    The differentiator is a real modelling choice, not a detail. The historical
    ``savgol`` default spans 1.1 s and keeps under 1% of the joint-velocity
    energy of a limb-swinging sequence, because that energy sits above 2 Hz. It
    is exposed here so training and rendering can be held to the same choice;
    they must match, or the fitted scales compensate for a mismatch.
    """
    frames = sequence_frame_count(sequence_dir)
    joints = np.stack([
        np.load(sequence_dir / "person_frames" / f"frame_{frame:03d}.npz")["joints_xyz"]
        for frame in range(frames)
    ]).astype(np.float32)
    return joints, smooth_joint_velocities(joints, method=differentiator)


def sequence_frame_count(sequence_dir: Path) -> int:
    """How many frames a person sequence holds: its metadata, else its files."""
    metadata = sequence_dir / "metadata.json"
    if metadata.is_file():
        return int(json.loads(metadata.read_text())["frames"])
    return len(list((sequence_dir / "person_frames").glob("frame_*.npz")))

# --------------------------------------------------------------------
# noise: the learned floor and the receiver nuisances
# --------------------------------------------------------------------

def range_dct_basis(n_range: int, n_coeff: int) -> torch.Tensor:
    """``(n_coeff, n_range)`` DCT-II basis, excluding the constant term.

    The constant term is excluded because ``log_scale`` already carries the
    level; keeping both would make the pair unidentifiable.
    """
    if n_coeff < 1:
        raise ValueError("n_coeff must be positive")
    index = np.arange(n_range, dtype=np.float64)
    rows = [
        np.cos(np.pi * (order + 1) * (index + 0.5) / n_range)
        for order in range(n_coeff)
    ]
    return torch.from_numpy(np.stack(rows).astype(np.float32))



class LearnedNoiseFloor(torch.nn.Module):
    """Expected noise power over (radar, Doppler, range), differentiably."""

    def __init__(
        self,
        envelope_prior: np.ndarray,
        n_coeff: int = 8,
        initial_scale: tuple[float, ...] | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        prior = np.asarray(envelope_prior, dtype=np.float32)
        if prior.ndim != 3:
            raise ValueError("envelope_prior must be (radars, doppler, range)")
        if not np.all(np.isfinite(prior)) or np.any(prior < 0):
            raise ValueError("envelope_prior must be finite and nonnegative")
        self.n_radars, self.n_doppler, self.n_range = prior.shape
        self.register_buffer("prior_power",
                             torch.from_numpy(prior).to(device) ** 2)
        self.register_buffer("basis",
                             range_dct_basis(self.n_range, n_coeff).to(device))
        start = (
            torch.zeros(self.n_radars, device=device) if initial_scale is None
            else torch.log(torch.tensor(
                initial_scale, dtype=torch.float32, device=device)).clamp_min(-20.0)
        )
        self.log_scale = torch.nn.Parameter(start)
        self.range_coeff = torch.nn.Parameter(
            torch.zeros(self.n_radars, n_coeff, device=device))
        # Expected power of the *other* nuisance processes, frozen. See
        # `set_extra_power`.
        self.register_buffer(
            "extra_power",
            torch.zeros(self.n_radars, self.n_doppler, self.n_range,
                        device=device))

    def set_extra_power(self, power: torch.Tensor) -> None:
        """Add another process's expected power to this forward model.

        Not a parameter and not learned: a frozen ``(radars, doppler, range)``
        contribution from a process the renderer also draws, so that training
        and rendering model the same set of processes. The diffuse floor is
        the first tenant.

        It lives here rather than beside the loss because `power()` is what
        `plane_power`, `microdoppler_marginal` and
        `microdoppler_marginal_batched` all read. Adding it at those call
        sites instead would mean patching the per-unit path and three batched
        paths separately, and the two would drift the first time one was
        touched -- which is the exact failure this whole line of work exists
        to close.

        The anchor loss that used to read `power()` here was removed with the
        rest of the anchor term (cleanup phase 5b); `fit_to_measured` below is
        now the only consumer that compares this model against a measured
        floor, and it compares the *total* modelled floor rather than one
        component against the whole, which was the flaw that made the earlier
        absorption audit unsound.
        """
        if power.shape != self.extra_power.shape:
            raise ValueError(
                f"extra power must be {tuple(self.extra_power.shape)}, "
                f"got {tuple(power.shape)}")
        if not torch.isfinite(power).all() or (power < 0).any():
            raise ValueError("extra power must be finite and nonnegative")
        self.extra_power = power.to(self.extra_power.dtype).to(
            self.extra_power.device)

    def learned_power(self) -> torch.Tensor:
        """``(radars, doppler, range)`` power of the *learned* floor alone.

        Reporting and the saturation check read this, so a frozen
        contribution cannot mask what the trained parameters are doing.
        """
        # Bounded so an early step cannot drive the floor to absurdity; +-3 in
        # the log is a factor of 20 either way on the range shape, far wider
        # than the 2.7x the measurement asks for.
        correction = torch.exp(
            torch.clamp(self.range_coeff @ self.basis, -3.0, 3.0)
        )
        scale = torch.exp(torch.clamp(2.0 * self.log_scale, -30.0, 10.0))
        return scale[:, None, None] * self.prior_power * correction[:, None, :]

    def power(self) -> torch.Tensor:
        """``(radars, doppler, range)`` total expected noise power.

        The learned floor plus every frozen process registered through
        `set_extra_power`. This is what the forward model adds.
        """
        return self.learned_power() + self.extra_power


    def microdoppler_marginal(
        self,
        radar_index: int,
        ra_mask: torch.Tensor,
        range_rows: torch.Tensor,
        motion_doppler: torch.Tensor,
    ) -> torch.Tensor:
        """The floor's contribution to ``microdoppler_marginal``'s output.

        That reduction is ``(power * ra_mask).sum(angle)[range_rows].sum(range)``
        per motion Doppler bin. The noise power has no angle dependence, so the
        angle sum is a per-range count of unmasked angles.
        """
        power = self.power()[radar_index]                      # (doppler, range)
        angles = ra_mask.sum(dim=1).to(power.dtype)            # (range,)
        weighted = power * angles[None, :]
        return weighted[:, range_rows].sum(dim=1)[motion_doppler]

    def microdoppler_marginal_batched(
        self,
        radar_index: torch.Tensor,
        ra_mask: torch.Tensor,
        range_rows: torch.Tensor,
        motion_index: torch.Tensor,
        range_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``microdoppler_marginal`` over a leading batch of units.

        ``range_index`` restricts the range axis, for the callers that work on
        the retained bins (244) rather than all 256.

        Mirrors ``_md_marginal_prediction_eager``'s reduction: ``ra_mask`` is
        ``(units, range, angle)`` and ``range_rows`` ``(units, range)``, and
        the rows ``range_rows`` drops are already zero because ``ra_mask``
        zeroed them.
        """
        power = self.power()
        if range_index is not None:
            power = power.index_select(2, range_index)
        power = power.index_select(0, radar_index)          # (units, D, range)
        angles = ra_mask.sum(dim=-1).to(power.dtype)        # (units, range)
        weighted = power * angles[:, None, :]
        marginal = (weighted * range_rows[:, None, :].to(power.dtype)).sum(dim=-1)
        return marginal.index_select(1, motion_index)

    def plane_power(self, radar_index: int, doppler_index: torch.Tensor,
                    keep_range: torch.Tensor) -> torch.Tensor:
        """``(planes, range)`` floor for the 63-plane RA term, angle-broadcast."""
        return self.power()[radar_index].index_select(
            0, doppler_index)[:, keep_range]

    def state_dict_summary(self) -> dict:
        with torch.no_grad():
            # `learned_power`, not `power`: a frozen contribution registered
            # through `set_extra_power` must not mask what the trained
            # parameters did.
            power = self.learned_power()
            return {
                "log_scale": self.log_scale.detach().cpu().tolist(),
                "range_coeff": self.range_coeff.detach().cpu().tolist(),
                "mean_power_per_radar": power.mean(dim=(1, 2)).cpu().tolist(),
                "mean_extra_power_per_radar":
                    self.extra_power.mean(dim=(1, 2)).cpu().tolist(),
                "range_correction_min_max": [
                    float(torch.exp(torch.clamp(
                        self.range_coeff @ self.basis, -3.0, 3.0)).min()),
                    float(torch.exp(torch.clamp(
                        self.range_coeff @ self.basis, -3.0, 3.0)).max()),
                ],
            }

# ----------------------------------------------------------------------
# receiver_noise: Phase-coherent receiver/oscillator noise in the radar slow-time domain.
# ----------------------------------------------------------------------


@dataclass
class CorrelatedRangeGain:
    """Frame-AR(1) real range gain shared by every loop and TX/RX channel."""

    range_bins: int
    sigma_fraction: float
    frame_correlation: float
    smoothing_passes: int = 3
    seed: int = 0

    def __post_init__(self) -> None:
        if self.range_bins <= 0:
            raise ValueError("range_bins must be positive")
        if not np.isfinite(self.sigma_fraction) or self.sigma_fraction < 0:
            raise ValueError("sigma_fraction must be finite and nonnegative")
        if not 0.0 <= self.frame_correlation < 1.0:
            raise ValueError("frame_correlation must lie in [0,1)")
        if self.smoothing_passes < 0:
            raise ValueError("smoothing_passes must be nonnegative")
        self._rng = np.random.default_rng(self.seed)
        self._state = self._innovation()
        self.frame_index = -1

    def _innovation(self) -> np.ndarray:
        values = self._rng.normal(size=self.range_bins)
        for _ in range(self.smoothing_passes):
            values = (
                np.roll(values, 1) + 2.0 * values + np.roll(values, -1)
            ) / 4.0
        values -= values.mean()
        values /= max(float(values.std()), 1e-12)
        return values

    def step(self) -> np.ndarray:
        rho = self.frame_correlation
        self._state = (
            rho * self._state
            + np.sqrt(max(1.0 - rho * rho, 0.0)) * self._innovation()
        )
        log_gain = self.sigma_fraction * self._state
        gain = np.exp(log_gain - 0.5 * self.sigma_fraction ** 2)
        self.frame_index += 1
        return gain.astype(np.float32)

    def metadata(self) -> dict:
        return {
            "model": (
                "real range-correlated frame-AR(1) receiver gain shared "
                "across all slow-time loops and TX/RX channels"
            ),
            "range_bins": self.range_bins,
            "sigma_fraction": self.sigma_fraction,
            "frame_correlation": self.frame_correlation,
            "range_smoothing_passes": self.smoothing_passes,
            "seed": self.seed,
            "phase_coherence_contract": (
                "one real gain per range bin is shared by every TX/RX and "
                "slow-time loop, so all virtual-antenna relative phases remain exact"
            ),
        }



@dataclass
class CorrelatedReceiverNoise:
    """Radar-specific complex gain applied before the Doppler/angle FFTs.

    One complex scalar is shared by every TX/RX channel and range bin in a
    slow-time loop.  It therefore changes common receiver/oscillator phase
    without corrupting virtual-array relative phase.  The frame-level phase and
    log-amplitude states are stationary AR(1) processes, with optional linear
    drift and a held-out periodic common-phase term.  A separate within-burst
    AR(1) phase residual creates a physically generated zero-Doppler ridge
    after the existing Doppler FFT.
    """

    phase_sigma_rad: float
    phase_drift_rad_per_frame: float = 0.0
    periodic_phase_amplitude_rad: float = 0.0
    periodic_phase_frequency_cycles_per_frame: float = 0.0
    periodic_phase_offset_rad: float | None = None
    amplitude_sigma_fraction: float = 0.0
    frame_correlation: float = 0.92
    chirp_phase_sigma_rad: float = 0.0
    chirp_correlation: float = 0.95
    chirp_frame_correlation: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("phase_sigma_rad", self.phase_sigma_rad),
            ("periodic_phase_amplitude_rad", self.periodic_phase_amplitude_rad),
            ("amplitude_sigma_fraction", self.amplitude_sigma_fraction),
            ("chirp_phase_sigma_rad", self.chirp_phase_sigma_rad),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.phase_drift_rad_per_frame):
            raise ValueError(
                "phase_drift_rad_per_frame must be finite"
            )
        if (
            not np.isfinite(self.periodic_phase_frequency_cycles_per_frame)
            or not 0.0
            <= self.periodic_phase_frequency_cycles_per_frame
            <= 0.5
        ):
            raise ValueError(
                "periodic_phase_frequency_cycles_per_frame must lie in [0,0.5]"
            )
        if (
            self.periodic_phase_amplitude_rad > 0
            and self.periodic_phase_frequency_cycles_per_frame <= 0
        ):
            raise ValueError(
                "positive periodic phase amplitude requires positive frequency"
            )
        if (
            self.periodic_phase_offset_rad is not None
            and not np.isfinite(self.periodic_phase_offset_rad)
        ):
            raise ValueError("periodic_phase_offset_rad must be finite")
        for name, value in (
            ("frame_correlation", self.frame_correlation),
            ("chirp_correlation", self.chirp_correlation),
            ("chirp_frame_correlation", self.chirp_frame_correlation),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0,1)")
        self._rng = np.random.default_rng(self.seed)
        self._phase_state = float(
            self._rng.normal(0.0, self.phase_sigma_rad)
        )
        self._log_amplitude_state = float(
            self._rng.normal(0.0, self.amplitude_sigma_fraction)
        )
        self._chirp_state: np.ndarray | None = None
        self._phase_drift_state = 0.0
        if self.periodic_phase_amplitude_rad > 0:
            if self.periodic_phase_offset_rad is None:
                self._periodic_phase_offset_rad = float(
                    self._rng.uniform(-np.pi, np.pi)
                )
            else:
                self._periodic_phase_offset_rad = float(
                    self.periodic_phase_offset_rad
                )
        else:
            self._periodic_phase_offset_rad = 0.0
        self.frame_index = -1

    def _advance_frame_state(self) -> None:
        rho = self.frame_correlation
        innovation = np.sqrt(max(1.0 - rho * rho, 0.0))
        self._phase_state = float(
            rho * self._phase_state
            + innovation * self._rng.normal(0.0, self.phase_sigma_rad)
        )
        self._log_amplitude_state = float(
            rho * self._log_amplitude_state
            + innovation
            * self._rng.normal(0.0, self.amplitude_sigma_fraction)
        )

    def step(self, loops: int = 1) -> np.ndarray:
        """Advance one 10 Hz frame and return ``loops`` complex gains."""
        if loops <= 0:
            raise ValueError("loops must be positive")
        self._advance_frame_state()
        self._phase_drift_state += self.phase_drift_rad_per_frame
        within = np.zeros(loops, dtype=np.float64)
        if loops > 1 and self.chirp_phase_sigma_rad > 0:
            innovation = np.empty(loops, dtype=np.float64)
            innovation[0] = self._rng.normal(
                0.0, self.chirp_phase_sigma_rad,
            )
            rho_chirp = self.chirp_correlation
            innovation_scale = np.sqrt(
                max(1.0 - rho_chirp * rho_chirp, 0.0)
            ) * self.chirp_phase_sigma_rad
            for loop in range(1, loops):
                innovation[loop] = (
                    rho_chirp * innovation[loop - 1]
                    + self._rng.normal(0.0, innovation_scale)
                )
            if self._chirp_state is None or len(self._chirp_state) != loops:
                self._chirp_state = innovation
            else:
                rho_frame = self.chirp_frame_correlation
                self._chirp_state = (
                    rho_frame * self._chirp_state
                    + np.sqrt(max(1.0 - rho_frame * rho_frame, 0.0))
                    * innovation
                )
            # Loop zero already contains the frame-level phase state.  Express
            # within-burst oscillator variation relative to that first loop.
            within = self._chirp_state - self._chirp_state[0]
        # Subtracting half the stationary log-variance keeps E[gain] near one.
        amplitude = np.exp(
            self._log_amplitude_state
            - 0.5 * self.amplitude_sigma_fraction ** 2
        )
        next_frame_index = self.frame_index + 1
        periodic_phase = self.periodic_phase_amplitude_rad * np.sin(
            2.0
            * np.pi
            * self.periodic_phase_frequency_cycles_per_frame
            * next_frame_index
            + self._periodic_phase_offset_rad
        )
        phase = (
            self._phase_state
            + self._phase_drift_state
            + periodic_phase
            + within
        )
        self.frame_index += 1
        return (amplitude * np.exp(1j * phase)).astype(np.complex64)

    def metadata(self) -> dict:
        return {
            "model": (
                "radar-specific frame-AR(1), drift, and optional periodic "
                "common-phase gain with within-burst AR(1) oscillator residual"
            ),
            "phase_sigma_rad": self.phase_sigma_rad,
            "phase_variance_rad2": self.phase_sigma_rad ** 2,
            "phase_drift_rad_per_frame": self.phase_drift_rad_per_frame,
            "periodic_phase_amplitude_rad": (
                self.periodic_phase_amplitude_rad
            ),
            "periodic_phase_frequency_cycles_per_frame": (
                self.periodic_phase_frequency_cycles_per_frame
            ),
            "periodic_phase_offset_rad": self._periodic_phase_offset_rad,
            "amplitude_sigma_fraction": self.amplitude_sigma_fraction,
            "frame_correlation": self.frame_correlation,
            "chirp_phase_sigma_rad": self.chirp_phase_sigma_rad,
            "chirp_correlation": self.chirp_correlation,
            "chirp_frame_correlation": self.chirp_frame_correlation,
            "seed": self.seed,
            "phase_coherence_contract": (
                "one complex gain per slow-time loop is shared by every TX/RX "
                "channel and range bin before the angle/Doppler FFTs"
            ),
        }

# ----------------------------------------------------------------------
# nuisance_model: The nuisance forward model, shared by training and rendering.
# ----------------------------------------------------------------------


def scene_range_support(
    positions: np.ndarray,
    radar_centre: np.ndarray,
    range_bin_spacing_m: float,
    range_bins: int = 256,
    smoothing_bins: int = 9,
) -> np.ndarray:
    """Diffuse scattering support per range bin, from the scene's own points.

    Returns a non-negative profile normalized to unit maximum. It is a
    *shape*: the absolute level stays a separate calibrated scalar per radar,
    which is a real physical unknown, where the near/far pair it replaces was
    two hand-set numbers.

    ``smoothing_bins`` is a boxcar over range. The reconstruction samples
    surfaces at whatever density the depth model produced, so the raw
    histogram carries that sampling as much as it carries scattering area;
    smoothing over a few bins removes the sampling grain without touching the
    metre-scale structure that the split-half test showed is real.
    """
    if positions.ndim != 2 or positions.shape[1] < 3:
        raise ValueError(f"positions must be (N, >=3), got {positions.shape}")
    if range_bin_spacing_m <= 0:
        raise ValueError("range bin spacing must be positive")
    if smoothing_bins < 1:
        raise ValueError("smoothing_bins must be at least 1")

    distance = np.linalg.norm(
        positions[:, :3].astype(np.float64)
        - np.asarray(radar_centre, dtype=np.float64).reshape(1, 3),
        axis=1,
    )
    index = np.rint(distance / range_bin_spacing_m).astype(np.int64)
    inside = (index >= 0) & (index < range_bins)
    support = np.bincount(
        index[inside], minlength=range_bins,
    ).astype(np.float64)[:range_bins]

    if smoothing_bins > 1:
        kernel = np.ones(smoothing_bins, dtype=np.float64) / smoothing_bins
        support = np.convolve(support, kernel, mode="same")

    peak = support.max()
    if peak <= 0:
        raise ValueError(
            "the scene has no points within the radar's range window, so it "
            "cannot supply diffuse floor support"
        )
    return support / peak



def support_to_range_amplitudes(
    support: np.ndarray, level: float,
) -> np.ndarray:
    """Per-bin field amplitude for a diffuse floor of the given ``level``.

    ``PhaseCoherentDiffuseFloor`` sums ``scatterers_per_bin`` scatterers of
    this amplitude in each bin, so the power it produces goes as the amplitude
    squared. Diffuse power should track the scattering area the scene presents
    at that range, so the amplitude tracks its square root.

    ``level`` is scaled to the support's *mean*, not its peak, so the total
    power a given level produces does not depend on how peaked one room's
    reconstruction happens to be.
    """
    amplitudes = np.sqrt(np.clip(support, 0.0, None))
    mean = amplitudes.mean()
    if mean <= 0:
        raise ValueError("support is empty")
    return amplitudes * (float(level) / mean)






def measure_diffuse_floor_expectation(
    floor, range_amplitudes: np.ndarray, draws: int = 200, loops: int = 64,
) -> dict:
    """The diffuse floor's expected power, measured from the model itself.

    Training needs the floor's expected power over ``(Doppler, range, angle)``.
    Deriving that in closed form means propagating an AR(1) latent through the
    angle transform, the slow-time FFT and clutter removal, and carrying the
    transforms' normalizations by hand. An error anywhere in that chain is
    exactly the silent kind this module exists to prevent, so it is measured
    instead: Monte Carlo over the sampler, using no measured data and fitting
    nothing. It is the model's own expectation, computed from the model, with
    the real array geometry the caller passes in.

    Returned factorized, as ``range_power[k] * doppler_shape[d] *
    angle_shape[a]``, because that is affordable and because the plane it
    produces was measured separable to 3.0% relative L1. Each marginal is a
    reduction over the other two axes, so it converges far faster than the
    cube would: 1.1% split-half at 150 draws for the Doppler marginal, against
    24.5% for the raw cube at 60.

    ``range_power`` is the *measured* per-range total, not ``amplitude^2``. It
    tracks it to r = 0.99994 -- which is the check that ties this back to
    `diffuse_floor_expected_power`, verified directly against the sampler --
    but it also carries the transforms' normalization, which is why it is
    measured rather than assumed.

    The Doppler DC bin comes back zero: clutter removal subtracts the
    slow-time mean, annihilating it identically. The Doppler axis is
    fftshifted, so that bin is ``loops // 2``, not 0. Some angle bins come back
    zero too, for the same kind of reason -- the array's aperture does not
    illuminate them.
    """
    import torch

    from mm4DPS.forward_render import (slow_time_profiles_to_ra_complex,
                                        slow_time_ra_to_doppler_magnitude)

    if draws < 1:
        raise ValueError("draws must be positive")
    if loops < 1:
        raise ValueError("loops must be positive")

    doppler = angle = range_power = None
    for _ in range(draws):
        cube = range_window_profiles(torch.from_numpy(
            floor.slow_time_cube(range_amplitudes, loops=loops)
        ))
        power = (slow_time_ra_to_doppler_magnitude(
            slow_time_profiles_to_ra_complex(cube.real, cube.imag),
        ).double() ** 2).numpy()                     # (doppler, range, angle)
        marginals = (power.sum(axis=(1, 2)), power.sum(axis=(0, 1)),
                     power.sum(axis=(0, 2)))
        if doppler is None:
            doppler, angle, range_power = marginals
        else:
            doppler = doppler + marginals[0]
            angle = angle + marginals[1]
            range_power = range_power + marginals[2]

    total = float(range_power.sum())
    if total <= 0:
        raise ValueError("the diffuse floor produced no power")
    return {
        "range_power": range_power / draws,
        "doppler_shape": doppler / doppler.sum(),
        "angle_shape": angle / angle.sum(),
        "draws": int(draws),
    }



def diffuse_floor_expected_cube(expectation: dict) -> np.ndarray:
    """Assemble ``(Doppler, range, angle)`` expected power from its marginals.

    The separable form validated against the sampler. Training adds this to
    its predicted power exactly as it already adds `noise_model.plane_power`.
    """
    return (expectation["doppler_shape"][:, None, None]
            * expectation["range_power"][None, :, None]
            * expectation["angle_shape"][None, None, :])
