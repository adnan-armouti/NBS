"""The renderer's video output, and the products it writes alongside.

Three modules merged here by `docs/CONSOLIDATION_AND_OO_PLAN.md` C10, in
dependency order -- `render/product_writer.py`, `render/video.py` and
`render/all_loops_video.py`.

    ProductWriter   the render products and their pending scalars
    the canvas      panels, heatmaps with axes, fonts, the colours
    run()           one sequence rendered frame by frame, and the video of it

Both `ROOT = Path(__file__).resolve().parent` constants went with the merge:
video.py's was never referenced, and all_loops_video's seven uses all read
`ROOT / WORKSPACE / "..."`, where WORKSPACE is absolute -- so pathlib discarded
the left operand and the prefix was already a no-op.

`run()` is 2,044 lines. Phase 2 of the plan extracts collaborator classes from
it, in this file, adding no new module.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
import contextlib
import io
import json
import matplotlib
import numpy as np
import os
import subprocess
import time
import torch
import zipfile

from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import Future, ThreadPoolExecutor
from matplotlib import colormaps
from numpy.lib.format import open_memmap
from pathlib import Path

from mm4DPS import DATASET_ROOT, HUPR_NEAR_FIELD_M, MM3DPS_ROOT, REPO_ROOT, WORKSPACE, array_axis, peak_vram
from mm4DPS import Dataset, config_centre_m, current_dataset, dataset_from_argv, per_radar, radar_configs, sequence_dir
from mm4DPS.data import range_profiles_all_channels_torch, slow_time_product
from mm4DPS.data import RADPROPOSER_SINGLE_RADAR_SHAPE, CascadeRecording, adc_frame_to_radproposer_tensor, apply_cascade_calibration, cascade_calibration, decode_dca1000_frame_torch, decode_tda2_frame_torch, find_archive, range_angle_slow_time_cube_torch, range_bin_spacing_m, range_keep_mask, range_profiles_to_radproposer_tensor
from mm4DPS.forward_render import build_rasterizers, linear_pearson_loss, log_pearson_loss, slow_time_profiles_to_ra_complex, slow_time_ra_to_doppler_complex, slow_time_ra_to_doppler_magnitude, subtract_spatial_noise_floor
from mm4DPS.scene import PhaseCoherentDiffuseFloor, PhaseCoherentStaticJitter, person_areas, person_solid_angle_weights, smpl_faces, surface_solid_angle_weights_torch, transform_person_joints, transform_person_pcl
from mm4DPS.signal_model import BODY_REGION_NAMES, CorrelatedRangeGain, array_window, CorrelatedReceiverNoise, assemble_bistatic_slow_time_profiles, assemble_slow_time_profiles, capture_summary, interpolate_frame_values, joint_bistatic_path_rates, joint_radial_velocities, nearest_joint_labels, official_motion_doppler_mask, offset_frame_index, person_range_angle_mask_torch, range_window_profiles, scene_range_support, smooth_joint_velocities, soft_joint_weights_torch, soft_surface_velocities, support_to_range_amplitudes, surface_bistatic_path_rates, sequence_frame_count, unwindowed_range_reference
from mm4DPS.train_material import SharedHumanMaterial


# --------------------------------------------------------------------
# ProductWriter: the render products and their pending scalars
# --------------------------------------------------------------------

DEFAULT_WRITER_THREADS = 8


class ProductWriter:
    """Append-only writer for a render's ``.npy`` products.

    Register each product with :meth:`add`, hand it device tensors with
    :meth:`append`, and call :meth:`flush` on a chunk boundary. ``flush``
    returns the host views of the rows it just staged, so the caller can read
    back what it wrote -- the run-summary metrics and the diagnostic video
    both need the frame's RD maps on the host -- without a second transfer.
    """

    def __init__(self, directory: Path, chunk_frames: int,
                 device: torch.device,
                 threads: int = DEFAULT_WRITER_THREADS) -> None:
        if chunk_frames < 1:
            raise ValueError("chunk_frames must be positive")
        self.directory = Path(directory)
        self.chunk_frames = int(chunk_frames)
        self.device = device
        self._pool = ThreadPoolExecutor(
            max_workers=threads, thread_name_prefix="product-write",
        )
        self._fd: dict[str, int] = {}
        self._offset: dict[str, int] = {}        # .npy header length
        self._row_bytes: dict[str, int] = {}
        self._rows_total: dict[str, int] = {}    # declared row count
        self._written: dict[str, int] = {}       # rows already handed to pwrite
        self._staged: dict[str, int] = {}        # rows in the current buffer
        # Two buffers per product, alternating: the pool may still be reading
        # the one it was given while the next chunk fills the other.
        self._buffers: dict[str, list[torch.Tensor]] = {}
        self._pending: dict[str, list[Future | None]] = {}
        self._slot: dict[str, int] = {}
        self.pinned_bytes = 0

    # -- registration ---------------------------------------------------

    def add(self, key: str, shape: tuple[int, ...], dtype,
            row_shape: tuple[int, ...] | None = None,
            rows_per_frame: int = 1) -> None:
        """Create ``<key>.npy`` and reserve its staging buffers.

        ``shape`` is the product's declared shape, exactly as it appears in
        the .npy header. ``row_shape`` is the shape of one :meth:`append`,
        which is ``shape[1:]`` for every product except the RadProPoser pair:
        those declare ``(frames, 2, ...)`` and are appended one *radar* at a
        time, so their row is ``shape[2:]`` and ``rows_per_frame`` is 2.
        A file is a flat byte stream, so appending its rows in order fills it
        whatever the declared shape groups them into.
        """
        if key in self._fd:
            raise ValueError(f"product {key} registered twice")
        path = self.directory / f"{key}.npy"
        # open_memmap writes the header and sizes the file exactly as before,
        # so the bytes this class then pwrites land in a file identical to the
        # one the memmap path produced. The mapping itself is dropped
        # immediately -- it is the mapping, not the file, that was expensive.
        mapping = open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        header = int(mapping.offset)
        total_bytes = int(mapping.nbytes)
        del mapping
        row_shape = tuple(shape[1:]) if row_shape is None else tuple(row_shape)
        row_bytes = int(np.dtype(dtype).itemsize
                        * int(np.prod(row_shape, dtype=np.int64)))
        if row_bytes <= 0 or total_bytes % row_bytes:
            raise ValueError(
                f"product {key}: row shape {row_shape} does not tile "
                f"{shape}")
        self._fd[key] = os.open(path, os.O_WRONLY)
        self._offset[key] = header
        self._row_bytes[key] = row_bytes
        self._rows_total[key] = total_bytes // row_bytes
        self._written[key] = 0
        self._staged[key] = 0
        self._slot[key] = 0
        capacity = self.chunk_frames * int(rows_per_frame)
        self._buffers[key] = [
            _host_buffer((capacity, *row_shape), _torch_dtype(dtype))
            for _ in range(2)
        ]
        self._pending[key] = [None, None]
        self.pinned_bytes += 2 * capacity * row_bytes

    # -- the frame loop -------------------------------------------------

    def append(self, key: str, row) -> None:
        """Stage one row. ``row`` is a device tensor of the row's shape."""
        staged = self._staged[key]
        buffer = self._buffers[key][self._slot[key]]
        if staged >= buffer.shape[0]:
            raise RuntimeError(
                f"product {key} staged {staged + 1} rows in one chunk but its "
                f"buffer holds {buffer.shape[0]}; rows_per_frame is wrong")
        if self._written[key] + staged >= self._rows_total[key]:
            raise RuntimeError(
                f"product {key} would write row "
                f"{self._written[key] + staged} of {self._rows_total[key]}")
        buffer[staged].copy_(row, non_blocking=True)
        self._staged[key] = staged + 1

    def flush(self) -> dict[str, np.ndarray]:
        """Complete the chunk's transfers and hand its rows to the pool.

        Returns ``{key: host view of the rows just staged}``. The views alias
        the pinned buffers the writer threads are reading, which is safe --
        nothing mutates them -- but they are only valid until the *next*
        flush of that product.
        """
        # One synchronize for the whole chunk. Every `.cpu()` this replaces
        # was its own. It happens before the empty check, not after, because
        # the caller's own deferred work -- the correlation scalars staged in
        # PendingScalars -- rides on the same stream and is resolved against
        # this synchronize, whether or not a product had rows this chunk.
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        active = [key for key, staged in self._staged.items() if staged]
        if not active:
            return {}
        # Submit every product first and only then wait, so the pool has all
        # 22 jobs at once. Waiting inside the submit loop would serialize the
        # chunk behind one product's disk write at a time.
        views: dict[str, np.ndarray] = {}
        for key in active:
            slot = self._slot[key]
            count = self._staged[key]
            view = self._buffers[key][slot][:count].numpy()
            views[key] = view
            offset = (self._offset[key]
                      + self._written[key] * self._row_bytes[key])
            self._written[key] += count
            self._pending[key][slot] = self._pool.submit(
                _pwrite_all, self._fd[key], view, offset,
            )
            self._staged[key] = 0
            self._slot[key] = 1 - slot
        # The buffer the next chunk will fill is the one the chunk before
        # last was written from; it has had a full chunk of compute to
        # finish, so this normally waits on nothing.
        for key in active:
            outstanding = self._pending[key][self._slot[key]]
            if outstanding is not None:
                outstanding.result()
                self._pending[key][self._slot[key]] = None
        return views

    def close(self) -> None:
        """Flush, wait for every write, and check each product is complete.

        The completeness check is the reason this is not just
        :meth:`shutdown`: a product that is one row short is a file full of
        the zeros ``open_memmap`` left, and silently shipping that is exactly
        the failure this writer must not introduce.
        """
        try:
            self.flush()
            for futures in self._pending.values():
                for index, future in enumerate(futures):
                    if future is not None:
                        future.result()
                        futures[index] = None
        finally:
            self.shutdown()
        short = {
            key: (self._written[key], self._rows_total[key])
            for key in self._rows_total
            if self._written[key] != self._rows_total[key]
        }
        if short:
            raise RuntimeError(f"products left partly written: {short}")

    def shutdown(self) -> None:
        """Stop the pool and close the files. Idempotent, and never raises.

        Called from the renderer's ``finally`` so an interrupted render
        releases its threads and descriptors without masking the exception
        that interrupted it.
        """
        self._pool.shutdown(wait=True)
        for fd in self._fd.values():
            os.close(fd)
        self._fd.clear()


def _host_buffer(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """A staging buffer, pinned when there is a CUDA runtime to pin it for.

    Pinning is what makes the device->host copy asynchronous, so production
    always gets it. The unpinned fallback exists so the writer -- and the
    byte-identity test that covers it -- runs on a machine without a GPU.
    """
    buffer = torch.empty(shape, dtype=dtype)
    return buffer.pin_memory() if torch.cuda.is_available() else buffer


def _pwrite_all(fd: int, array: np.ndarray, offset: int) -> None:
    """``os.pwrite`` the whole array, tolerating a short write."""
    data = memoryview(np.ascontiguousarray(array)).cast("B")
    written = 0
    while written < len(data):
        written += os.pwrite(fd, data[written:], offset + written)


def _torch_dtype(dtype) -> torch.dtype:
    mapping = {
        np.dtype(np.float16): torch.float16,
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float64): torch.float64,
        np.dtype(np.complex64): torch.complex64,
        np.dtype(np.complex128): torch.complex128,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
    }
    key = np.dtype(dtype)
    if key not in mapping:
        raise ValueError(f"no torch dtype for product dtype {key}")
    return mapping[key]


class PendingScalars:
    """The frame loop's scalar metrics, resolved one chunk at a time.

    Each ``float(log_pearson_loss(...))`` in the loop was a device
    synchronization -- ten per frame, all of them for values nothing reads
    until the run summary is written. :meth:`add` copies the 0-dim tensor into
    a pinned float64 buffer instead and returns the slot to read it from after
    the chunk's single synchronize.

    float64 rather than float32 so that ``float(values()[slot])`` is the same
    double the old ``float(tensor)`` produced: widening a float32 is exact.
    """

    def __init__(self, capacity: int) -> None:
        self._buffer = _host_buffer((capacity,), torch.float64)
        self._count = 0

    def add(self, value: torch.Tensor) -> int:
        if self._count >= self._buffer.shape[0]:
            raise RuntimeError("pending-scalar buffer is full")
        slot = self._count
        self._buffer[slot].copy_(value, non_blocking=True)
        self._count = slot + 1
        return slot

    def values(self) -> np.ndarray:
        """The staged values. Only valid after the chunk's synchronize."""
        return self._buffer[:self._count].numpy()

    def reset(self) -> None:
        self._count = 0

# --------------------------------------------------------------------
# the canvas: panels, heatmaps, fonts and colours
# --------------------------------------------------------------------

matplotlib.use("Agg")





# HuPR ships 14 joints, not the 17 of COCO. Read off the data and confirmed by
# segment length: 0/3 hips (0.18 m apart), 1/4 knees, 2/5 ankles, 6 neck, 7 head,
# 8/11 shoulders (0.36 m apart), 9/12 elbows, 10/13 wrists. Thighs measure
# 0.38-0.40 m, shins 0.34-0.35 m, upper arms 0.28-0.30 m, forearms 0.27 m.
HIPS = (0, 3)

NECK = 6

SKELETON = (
    (0, 1), (1, 2),          # leg A: hip, knee, ankle
    (3, 4), (4, 5),          # leg B
    (0, 3),                  # pelvis
    (6, 7),                  # neck to head
    (6, 8), (6, 11),         # neck to each shoulder
    (8, 11),                 # shoulder span
    (8, 9), (9, 10),         # arm A: shoulder, elbow, wrist
    (11, 12), (12, 13),      # arm B
)



























# ----------------------------------------------------------------------
# video_plot: Plain, reusable plotting helpers for HuPR deliverable videos.
# ----------------------------------------------------------------------


BLACK = (24, 24, 24)

GRAY = (92, 92, 92)

GRID = (218, 218, 218)

RESAMPLING = getattr(Image, "Resampling", Image)



def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()



def robust_unit(
    values: np.ndarray, reference: float | None = None,
) -> np.ndarray:
    """Linear magnitude scaled to [0, 1] by a robust peak.

    Linear, not dB: every metric and every visualization in this project is on
    linear magnitude. This is the same convention
    build_velocity_comparison_media.py already uses for the side-by-side
    videos, so the annotated diagnostic video and the comparison videos now
    show the same thing.
    """
    peak = max(
        float(np.percentile(values, 99.9)) if reference is None else float(reference),
        1e-12,
    )
    return np.clip(np.asarray(values, dtype=np.float64) / peak, 0.0, 1.0)



def _tick_label(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.1f}"



def heatmap_with_axes(
    values: np.ndarray,
    size: tuple[int, int],
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xticks: list[float],
    yticks: list[float],
    xlabel: str,
    ylabel: str,
    reference: float | None = None,
) -> Image.Image:
    """Render a magnitude heatmap with explicit axes and tick labels.

    Array row zero is displayed at the bottom (Matplotlib ``origin='lower'``
    convention), matching both HuPR range-angle and range-Doppler products.
    """
    width, height = size
    left, right, top, bottom = 78, 18, 12, 62
    plot_w = width - left - right
    plot_h = height - top - bottom
    normalized = robust_unit(values, reference=reference)
    rgb = (colormaps["magma"](np.flipud(normalized))[:, :, :3] * 255).astype(np.uint8)
    heatmap = Image.fromarray(rgb).resize((plot_w, plot_h), RESAMPLING.BICUBIC)

    image = Image.new("RGB", size, "white")
    image.paste(heatmap, (left, top))
    draw = ImageDraw.Draw(image)
    draw.rectangle((left, top, left + plot_w, top + plot_h), outline=BLACK, width=2)

    x0, x1 = xlim
    for value in xticks:
        ratio = (value - x0) / (x1 - x0)
        x = left + ratio * plot_w
        draw.line((x, top + plot_h, x, top + plot_h + 6), fill=BLACK, width=2)
        draw.text((x, top + plot_h + 10), _tick_label(value), fill=BLACK,
                  font=font(14), anchor="ma")

    y0, y1 = ylim
    for value in yticks:
        ratio = (value - y0) / (y1 - y0)
        y = top + plot_h - ratio * plot_h
        draw.line((left - 6, y, left, y), fill=BLACK, width=2)
        draw.text((left - 10, y), _tick_label(value), fill=BLACK,
                  font=font(14), anchor="rm")

    draw.text((left + plot_w / 2, height - 17), xlabel, fill=BLACK,
              font=font(17), anchor="mm")
    label_box = Image.new("RGBA", (plot_h, 28), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_box)
    label_draw.text((plot_h / 2, 14), ylabel, fill=BLACK, font=font(17), anchor="mm")
    label_box = label_box.rotate(90, expand=True)
    image.paste(label_box, (4, top + (plot_h - label_box.height) // 2), label_box)
    return image



def contain_rgb(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Fit an RGB frame without changing its aspect ratio."""
    result = Image.new("RGB", size, "white")
    fitted = image.copy()
    fitted.thumbnail(size, RESAMPLING.LANCZOS)
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    result.paste(fitted, (x, y))
    ImageDraw.Draw(result).rectangle((0, 0, size[0] - 1, size[1] - 1), outline=GRID, width=2)
    return result

# --------------------------------------------------------------------
# run(): one sequence, frame by frame, and the video of it
# --------------------------------------------------------------------

# Beside the code's outputs rather than inside a workspace: the cache is a
# pure function of the sequence's ADC archive, so every workspace rendering
# that sequence shares one (plan section 6).
#
WIDTH, HEIGHT, FPS = 1920, 1080, 10
BG, FG, MUTED, CYAN = (255, 255, 255), BLACK, GRAY, (45, 162, 170)
RANGE_KEEP = range_keep_mask()

# The diffuse floor's two historical range windows, 0.5-2.5 m and 9.2-11.0 m.
# They are no longer where the floor is placed -- support comes from the
# camera's scene reconstruction, because these windows correlate -0.27..-0.60
# with the floor measured from the ADC while scene support correlates
# +0.36..+0.56. They survive only to reproduce the old calibration's *level*,
# which was a bin-count-weighted mean of its near and far amplitudes. Constants
# rather than flags: nothing may tune a window that no longer places anything.
DIFFUSE_FLOOR_LEGACY_NEAR_M = (0.5, 2.5)
DIFFUSE_FLOOR_LEGACY_FAR_M = (9.2, 11.0)
FIRST_RANGE_BIN = int(np.flatnonzero(RANGE_KEEP)[0])
FIRST_RANGE_M = FIRST_RANGE_BIN * range_bin_spacing_m()



def background_profile_shape_error(
    rendered_rd: np.ndarray,
    measured_rd: np.ndarray,
    doppler_rows: slice = slice(24, 40),
) -> float:
    """Shape error of the displayed near-DC range profile, on linear magnitude.

    This is deliberately *not* a correlation.  The panel shows clutter-removed
    RD, so on a pose-static frame its content is nuisance whose rendered and
    measured realizations are independent by construction -- a framewise
    correlation there would read near zero no matter how good the model is.
    Comparing range profiles instead measures what the eye actually judges,
    and is defined on every frame including static ones.

    The overall level is divided out before averaging, so this reports *shape*
    disagreement only.  A uniform level difference is invisible in a
    peak-normalized panel, whereas a band that is too bright over part of the
    range is exactly what shows up; mixing the two hides the band behind a
    constant.

    Linear, not dB. It was a mean absolute dB difference until 2026-09-03; the
    equivalent on linear magnitude is the mean absolute fractional deviation
    of the rendered/measured profile ratio about its own median. 0 is a
    perfect shape match and 0.5 means the ratio departs from its own level by
    half, on average.
    """
    keep = RANGE_KEEP

    def profile(values: np.ndarray) -> np.ndarray:
        rows = np.asarray(values, dtype=np.float64)[doppler_rows].mean(axis=0)
        return np.maximum(rows[keep], 1e-20)

    ratio = profile(rendered_rd) / profile(measured_rd)
    level = float(np.median(ratio))
    if level <= 0:
        return float("inf")
    return float(np.abs(ratio / level - 1.0).mean())


def virtual_channel_gain_tensor(
    channel_magnitude: list[float],
    channel_phase_rad: list[float],
    tx_count: int,
    rx_count: int,
    azimuth_tx: tuple[int, ...] = (0, 2),
) -> np.ndarray:
    """Expand an 8-element virtual-array calibration onto a (TX, RX) grid.

    ``virtual_array_slow_time`` builds the ULA as TX0.RX0..3 then TX2.RX0..3,
    so virtual element ``v`` maps to ``azimuth_tx[v // rx_count]`` and
    ``v % rx_count``. Transmitters outside the azimuth pair carry no virtual
    element and are left at unit gain rather than dropped, so the tensor can
    multiply the full profile stack without reshaping it.
    """
    magnitude = np.asarray(channel_magnitude, dtype=np.float64)
    phase = np.asarray(channel_phase_rad, dtype=np.float64)
    if magnitude.shape != phase.shape or magnitude.ndim != 1:
        raise ValueError("magnitude and phase must be matching 1-D arrays")
    if magnitude.size != len(azimuth_tx) * rx_count:
        raise ValueError(
            f"expected {len(azimuth_tx) * rx_count} virtual elements, "
            f"got {magnitude.size}"
        )
    if not np.all(np.isfinite(magnitude)) or np.any(magnitude < 0):
        raise ValueError("channel magnitudes must be finite and nonnegative")
    if not np.all(np.isfinite(phase)):
        raise ValueError("channel phases must be finite")
    gains = np.ones((tx_count, rx_count), dtype=np.complex128)
    for index, (magnitude_value, phase_value) in enumerate(zip(magnitude, phase)):
        transmitter = azimuth_tx[index // rx_count]
        receiver = index % rx_count
        gains[transmitter, receiver] = magnitude_value * np.exp(1j * phase_value)
    return gains


def density_normalized_scene_profile(
    profile: np.ndarray,
    point_counts: np.ndarray,
) -> np.ndarray:
    """Divide a rendered range profile by the sampling density that produced it.

    The background is a monocular unprojection, so the number of reconstructed
    points per range bin follows how many *pixels* landed on a surface, not the
    surface's physical extent.  A back wall filling the image contributes far
    more points than a nearby floor patch of the same area.  Summing randomly
    phased scatterers gives a profile magnitude proportional to ``sqrt(N)``, so
    dividing by ``sqrt(N)`` recovers the per-scatterer envelope and removes the
    reconstruction artefact.

    Without this, the sequence-1 cloud puts 59% of its 20,000 points in
    5.5-7.5 m, which alone lifts that band about 5.8 dB above 4-5.5 m and shows
    up as a spurious bright band.
    """
    profile = np.asarray(profile, dtype=np.float64)
    counts = np.asarray(point_counts, dtype=np.float64)
    if profile.shape != counts.shape:
        raise ValueError("profile and point counts must have the same shape")
    if not np.all(np.isfinite(profile)) or np.any(profile < 0):
        raise ValueError("profile must be finite and nonnegative")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("point counts must be finite and nonnegative")
    # Bins with no reconstructed geometry carry no scene skirt at all; the
    # calibrated part of the mix still covers them.
    return np.divide(
        profile, np.sqrt(counts),
        out=np.zeros_like(profile), where=counts > 0,
    )


def smooth_active_range_profile(
    profile: np.ndarray,
    active: np.ndarray,
    smoothing_bins: int,
) -> np.ndarray:
    """Masked moving average of a range profile over its active bins.

    A phase-noise skirt is the local scatterer convolved with the range
    response, so it follows the broad clutter envelope rather than every bin's
    fine structure.  Averaging only over active bins keeps the guarded
    near/far bands from pulling the envelope toward zero at the edges.
    """
    if smoothing_bins <= 1:
        return np.where(active, profile, 0.0)
    kernel = np.ones(int(smoothing_bins), dtype=np.float64)
    masked = np.where(active, profile, 0.0)
    weight = np.convolve(active.astype(np.float64), kernel, mode="same")
    total = np.convolve(masked, kernel, mode="same")
    smoothed = np.divide(
        total, weight, out=np.zeros_like(masked), where=weight > 0,
    )
    # A "same" convolution bleeds past the mask; keep the guard bands empty.
    return np.where(active, smoothed, 0.0)


def remix_slow_time_range_shape(
    calibrated_envelope: np.ndarray,
    scene_range_profile: np.ndarray,
    scene_proportional_fraction: float,
    scene_range_smoothing_bins: int = 1,
) -> np.ndarray:
    """Blend the held-out Doppler-range envelope's range marginal toward the
    rendered scene's own range profile.

    A receiver phase-noise skirt scales with the *local* scatterer amplitude,
    whereas a thermal floor does not.  The held-out calibration measures their
    sum on the real scene, so replaying its range shape verbatim on a rendered
    scene with a different range profile misplaces the skirt part.  This
    reallocates ``scene_proportional_fraction`` of the range marginal onto the
    rendered background's own per-range magnitude while preserving the
    calibrated Doppler shape exactly -- that shape is what reproduces the
    measured DC-decaying ridge.  The active-bin mean amplitude is preserved so
    the existing scale parameter keeps its meaning and the two knobs stay
    separable: this one moves floor between range bins, the scale sets its
    overall level.

    ``scene_range_smoothing_bins`` must be wide enough that only the broad
    clutter envelope is transferred.  Replaying the rendered scene's bin-level
    range structure onto the floor makes the RD texture trivially separable
    from measured data (C2ST AUC above 0.9), because the point-cloud scene's
    range profile is far spikier than any real clutter envelope.
    """
    if not 0.0 <= scene_proportional_fraction <= 1.0:
        raise ValueError("scene_proportional_fraction must lie in [0,1]")
    if scene_range_smoothing_bins < 1:
        raise ValueError("scene_range_smoothing_bins must be at least one")
    envelope = np.asarray(calibrated_envelope, dtype=np.float64)
    profile = np.asarray(scene_range_profile, dtype=np.float64)
    if envelope.ndim != 2:
        raise ValueError("calibrated envelope must be two-dimensional")
    if profile.shape != (envelope.shape[1],):
        raise ValueError("scene range profile must match the envelope range axis")
    if not np.all(np.isfinite(envelope)) or np.any(envelope < 0):
        raise ValueError("calibrated envelope must be finite and nonnegative")
    if not np.all(np.isfinite(profile)) or np.any(profile < 0):
        raise ValueError("scene range profile must be finite and nonnegative")
    if scene_proportional_fraction == 0.0:
        return envelope.copy()
    marginal = envelope.mean(axis=0)
    active = marginal > 0
    if not np.any(active) or not np.any(profile[active] > 0):
        return envelope.copy()
    profile = smooth_active_range_profile(
        profile, active, scene_range_smoothing_bins,
    )
    if not np.any(profile[active] > 0):
        return envelope.copy()
    # Doppler shape per range bin, unchanged by the remix.
    shape = np.zeros_like(envelope)
    shape[:, active] = envelope[:, active] / marginal[active]
    # Put the scene profile on the calibrated marginal's scale before mixing so
    # the blend is a redistribution rather than a level change.
    scaled = np.zeros_like(profile)
    scaled[active] = (
        profile[active]
        * marginal[active].mean()
        / profile[active].mean()
    )
    blended = (
        (1.0 - scene_proportional_fraction) * marginal
        + scene_proportional_fraction * scaled
    )
    return shape * blended[None, :]


def blend_envelope_range_edges(
    envelope: np.ndarray,
    active: np.ndarray,
    blend_bins: int,
) -> np.ndarray:
    """Taper hard range edges out of a calibrated Doppler-range envelope.

    The cross-sequence calibration overwrites the envelope with the constant
    1.0 inside a hard 9.4-10.8 m rectangle
    (``profile_hupr_receiver_noise_multisequence.py``, "flatten the held-out
    empty region"), against neighbouring bins at ~1.40. The renderer
    multiplies its nuisance by that envelope, so the discontinuity is
    reproduced in every frame at every Doppler bin -- which is what a vertical
    line in the RD video is. Measured on a seq-38 render: a **40.5%**
    single-bin step at 10.78 m, where the measurement itself steps by 12.6%.

    A raised-cosine moving average along the range axis removes it. This
    cannot destroy measured structure: the same calibration already
    median-smooths its range profile over 31 bins, so anything narrower than
    ``blend_bins`` in the result was put there by the flattening, not by the
    data. The Doppler shape at each range is untouched.
    """
    if blend_bins <= 1:
        return envelope
    values = np.asarray(envelope, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("envelope must be (doppler, range)")
    if active.shape != (values.shape[1],):
        raise ValueError("active mask must match the envelope's range axis")
    kernel = np.hanning(int(blend_bins) + 2)[1:-1]
    kernel = kernel / kernel.sum()
    weight = np.convolve(active.astype(np.float64), kernel, mode="same")
    masked = values * active[None, :]
    smoothed = np.stack([
        np.convolve(row, kernel, mode="same") for row in masked
    ])
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = np.where(weight[None, :] > 0, smoothed / weight[None, :], 0.0)
    # Outside the retained bins the envelope is zeroed downstream anyway;
    # keeping it explicit here stops the taper leaking across the guard.
    return np.where(active[None, :], smoothed, 0.0)


def phase_noise_spectral_envelope(
    loops: int,
    knee_bins: float,
    white_power_fraction: float,
) -> np.ndarray:
    """Return an fftshifted analytic leakage/phase-noise amplitude envelope."""
    if loops <= 1:
        raise ValueError("loops must be greater than one")
    if not np.isfinite(knee_bins) or knee_bins <= 0:
        raise ValueError("knee_bins must be finite and positive")
    if not 0.0 <= white_power_fraction <= 1.0:
        raise ValueError("white_power_fraction must lie in [0,1]")
    doppler = np.arange(loops, dtype=np.float64) - loops // 2
    lorentzian_power = 1.0 / (1.0 + (np.abs(doppler) / knee_bins) ** 2)
    power = (
        white_power_fraction
        + (1.0 - white_power_fraction) * lorentzian_power
    )
    amplitude = np.sqrt(power)
    far = np.abs(doppler) >= max(loops // 3, 1)
    amplitude /= max(float(np.median(amplitude[far])), 1e-12)
    return amplitude


def panel(canvas: Image.Image, xy: tuple[int, int], image: Image.Image,
          title: str, subtitle: str = "") -> None:
    x, y = xy
    draw = ImageDraw.Draw(canvas)
    draw.text((x, y), title, fill=FG, font=font(22))
    if subtitle:
        draw.text((x, y + 27), subtitle, fill=MUTED, font=font(15))
    canvas.paste(image, (x, y + 50))


def annotate_rgb(rgb: Image.Image, cached: np.lib.npyio.NpzFile) -> Image.Image:
    image = rgb.copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle(cached["bbox_xyxy"].tolist(), outline=(255, 91, 76), width=4)
    joints = cached["joints_xy"]
    edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (0, 6), (3, 6),
             (6, 7), (6, 8), (8, 9), (9, 10), (6, 11), (11, 12), (12, 13)]
    for a, b in edges:
        draw.line((*joints[a], *joints[b]), fill=CYAN, width=4)
    for x, y in joints:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 221, 87))
    return image


def rd_plot(values: np.ndarray) -> Image.Image:
    return heatmap_with_axes(
        values[24:40, RANGE_KEEP], (650, 390),
        xlim=(FIRST_RANGE_M, 10.99), ylim=(-8.0, 7.0),
        xticks=[0.5, 2, 4, 6, 8, 10], yticks=[-8, -4, 0, 4, 7],
        xlabel="Range (m)", ylabel="Doppler bin",
    )


def compose(
    rgb: Image.Image, cached: np.lib.npyio.NpzFile,
    measured_h: np.ndarray, rendered_h: np.ndarray,
    measured_v: np.ndarray, rendered_v: np.ndarray,
    frame: int, corr_h: float | None, corr_v: float | None, model_label: str,
    skinning_label: str, bg_error_h: float, bg_error_v: float,
    total_frames: int,
) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 18),
              f"HuPR — {total_frames} frames, all chirps ({model_label})",
              fill=FG, font=font(30))
    draw.text((1580, 24),
              f"Frame {frame + 1}/{total_frames}   {frame / FPS:05.1f} s",
              fill=FG, font=font(19))
    left_rgb = contain_rgb(annotate_rgb(rgb, cached), (480, 360))
    panel(canvas, (30, 82), left_rgb, "RGB and HuPR 14-joint skeleton")
    def correlation_label(value: float | None) -> str:
        return (
            f"{value:.3f}" if value is not None
            else "no person Doppler this frame"
        )

    notes = [
        "64 slow-time loops per frame",
        "192 physical TDM chirps per frame",
        skinning_label,
        "14 articulated joint motion groups",
        "Display: central 16 Doppler bins",
        "Person metric: fixed ROI, motion frames",
        "",
        "Person micro-Doppler correlation",
        f"  Horizontal: {correlation_label(corr_h)}",
        f"  Vertical:   {correlation_label(corr_v)}",
        "",
        "Background range-profile error (lower better)",
        f"  Horizontal: {bg_error_h:.2f} dB",
        f"  Vertical:   {bg_error_v:.2f} dB",
        "",
        "Color scale: normalized -50 to 0 dB",
    ]
    y = 535
    for line in notes:
        draw.text((38, y), line, fill=FG if line else MUTED, font=font(18))
        y += 34

    panel(canvas, (540, 82), rd_plot(measured_h),
          "Horizontal radar — measured", "Angle-mean range-Doppler; central 16 bins")
    panel(canvas, (1230, 82), rd_plot(rendered_h),
          "Horizontal radar — rendered",
          f"Person correlation: {correlation_label(corr_h)}"
          f"   Background: {bg_error_h:.2f} dB")
    panel(canvas, (540, 570), rd_plot(measured_v),
          "Vertical radar — measured", "Angle-mean range-Doppler; board +90 degrees")
    panel(canvas, (1230, 570), rd_plot(rendered_v),
          "Vertical radar — rendered",
          f"Person correlation: {correlation_label(corr_v)}"
          f"   Background: {bg_error_v:.2f} dB")
    return canvas


def on_device(values, device):
    """A device tensor, whether the nuisance model returned one or a array.

    The nuisance models hand back a torch tensor already on ``device`` when
    their draws live there, and a numpy array otherwise. Accepting both is
    what lets the GPU-draw path avoid a host round trip per frame per radar
    without a second copy of every call site.
    """
    if isinstance(values, torch.Tensor):
        return values if str(values.device) == str(device) else values.to(device)
    return torch.from_numpy(values).to(device)


def surface_slow_time_assembler():
    """The surface slow-time assembly, on the fused CUDA kernel.

    ``assemble_surface_slow_time_profiles_cuda`` computes each contribution in
    registers and scatters it immediately, where the torch reference
    materialized ``(loops, spread, paths)`` complex tensors per loop chunk
    before one ``scatter_add``. This renderer used to run the torch reference
    for every one of its 1,200 (frame, radar) assemblies, which is why the
    render stage did not benefit from any of the kernel work in
    ``docs/RUNTIME_OPTIMIZATION_REPORT.md``.

    The startup gate that recomputed the first assembly of each radar with the
    reference and compared was removed in the deployment strip, together with
    the ``--slow-time-backend`` switch it belonged to. `gate_label` is retained
    in the signature because call sites pass it; it is now ignored.
    """
    from mm4DPS.forward_render import assemble_surface_slow_time_profiles_cuda

    def assemble(background_profile, components, path_rate, *,
                 gate_label=None, **kwargs):
        return assemble_surface_slow_time_profiles_cuda(
            background_profile, components, path_rate, **kwargs)
    return assemble

    return assemble



@dataclass
class DiffuseFloor:
    """The diffuse floor: one model per background, or none.

    P1b-1 of docs/CONSOLIDATION_AND_OO_PLAN.md. `models` is empty when the
    floor is disabled, and every consumer tests exactly that, as run() did.
    `bin_ranges` is None in that case -- it was an unbound local before.
    """

    models: dict
    range_amplitudes: dict
    bin_ranges: "np.ndarray | None"
    calibration: "dict | None"

    @property
    def enabled(self) -> bool:
        return bool(self.models)


def _configure_diffuse_floor(args, configs, device, rasterizers, state) -> DiffuseFloor:
    """run()'s diffuse-floor setup block, verbatim, as a builder."""
    diffuse_floor_models = {}
    diffuse_floor_range_amplitudes = {}
    bin_ranges = None
    diffuse_floor_calibration = (
        json.loads(args.diffuse_floor_calibration.read_text())
        if args.diffuse_floor_calibration is not None else None
    )
    if diffuse_floor_calibration is not None or args.diffuse_floor_relative_amplitude > 0:
        bin_ranges = np.arange(256, dtype=np.float64) * range_bin_spacing_m()
        legacy_near = ((bin_ranges >= DIFFUSE_FLOOR_LEGACY_NEAR_M[0])
                       & (bin_ranges < DIFFUSE_FLOOR_LEGACY_NEAR_M[1]))
        legacy_far = ((bin_ranges >= DIFFUSE_FLOOR_LEGACY_FAR_M[0])
                      & (bin_ranges <= DIFFUSE_FLOOR_LEGACY_FAR_M[1]))
        rectangle_bins = legacy_near | legacy_far
        scene_positions = state["positions"].detach().cpu().numpy()
        for radar_index, (name, rasterizer) in enumerate(rasterizers.items()):
            # Support: where the camera says there are surfaces to scatter
            # from, the person's own range band included. The two hand-drawn
            # windows this replaces correlate -0.27..-0.60 with the floor
            # measured from the ADC -- they put diffuse power where the real
            # floor is weakest -- against +0.36..+0.56 for the scene.
            support = (
                scene_range_support(
                    scene_positions,
                    config_centre_m(configs[name]).astype(np.float64),
                    range_bin_spacing_m(),
                    smoothing_bins=args.diffuse_floor_support_smoothing_bins,
                )
                if args.diffuse_floor_support == "geometry" else None
            )
            active_floor_bins = (
                support > 0 if support is not None else rectangle_bins
            )
            range_amplitudes = np.zeros(256, dtype=np.float64)
            if diffuse_floor_calibration is None:
                if support is not None:
                    range_amplitudes = support_to_range_amplitudes(
                        support, args.diffuse_floor_relative_amplitude)
                else:
                    range_amplitudes[rectangle_bins] = (
                        args.diffuse_floor_relative_amplitude)
                temporal_correlation = args.diffuse_floor_temporal_correlation
                spatial_sines = None
            else:
                selected = diffuse_floor_calibration["selected_range_profile_amplitudes"][name]
                if support is not None:
                    # One calibrated level per radar instead of the near/far
                    # pair: the level is a real physical unknown, the two
                    # windows were not. The mean of the old profile is
                    # preserved so this changes shape and support, not level.
                    near_bins, far_bins = legacy_near, legacy_far
                    level = float(
                        (float(selected["near"]) * near_bins.sum()
                         + float(selected["far"]) * far_bins.sum())
                        / max(int(rectangle_bins.sum()), 1)
                    )
                    range_amplitudes = support_to_range_amplitudes(support, level)
                else:
                    range_amplitudes[legacy_near] = float(selected["near"])
                    range_amplitudes[legacy_far] = float(selected["far"])
                temporal_correlation = float(
                    diffuse_floor_calibration["selected_temporal_correlation"]
                )
                spatial_sines = np.asarray(
                    diffuse_floor_calibration["selected_spatial_sine_by_range"][name],
                    dtype=np.float64,
                )
            range_amplitudes *= args.diffuse_floor_scale
            diffuse_floor_range_amplitudes[name] = range_amplitudes
            diffuse_floor_models[name] = PhaseCoherentDiffuseFloor(
                rasterizer.tx_positions.detach().cpu().numpy(),
                rasterizer.rx_positions.detach().cpu().numpy(),
                config_centre_m(configs[name]),
                array_axis(configs[name]),
                wavelength_m=299792458.0 / rasterizer.center_freq,
                range_bin_spacing_m=range_bin_spacing_m(),
                active_range_bins=active_floor_bins,
                scatterers_per_bin=args.diffuse_floor_scatterers_per_bin,
                temporal_correlation=temporal_correlation,
                seed=args.diffuse_floor_seed + radar_index,
                spatial_sine_by_range=spatial_sines,
                backend=args.nuisance_backend,
                device=str(device),
                draws=args.nuisance_draws,
            )
    return DiffuseFloor(
        models=diffuse_floor_models,
        range_amplitudes=diffuse_floor_range_amplitudes,
        bin_ranges=bin_ranges,
        calibration=diffuse_floor_calibration,
    )



@dataclass
class ReceiverNoise:
    """The receiver's own noise: the phase-coherent static jitter, and one
    noise model per radar or none. P1b-2 of docs/CONSOLIDATION_AND_OO_PLAN.md."""

    models: object
    jitter: object


def _configure_receiver_noise(args, background) -> ReceiverNoise:
    """run()'s setup span, verbatim, as a builder."""
    jitter = None
    if args.static_jitter_sigma_um > 0:
        jitter = PhaseCoherentStaticJitter(
            background.positions.detach().cpu().numpy(),
            background.get_normals().detach().cpu().numpy(),
            sigma_m=args.static_jitter_sigma_um * 1e-6,
            temporal_correlation=args.static_jitter_temporal_correlation,
            common_variance_fraction=args.static_jitter_common_variance_fraction,
            seed=args.static_jitter_seed,
        )
    receiver_noise_models = {}
    receiver_phase_sigmas = per_radar(args, "receiver-phase", "sigma-rad")
    receiver_amplitude_sigmas = per_radar(args, "receiver-amplitude", "sigma")
    receiver_phase_drifts = per_radar(args, "receiver-phase", "drift-rad-per-frame")
    receiver_periodic_phase_amplitudes = per_radar(args, "receiver-phase", "periodic-amplitude-rad")
    receiver_periodic_phase_frequencies = per_radar(args, "receiver-phase", "periodic-frequency-cycles-per-frame")
    receiver_chirp_sigmas = per_radar(args, "receiver-chirp-phase", "sigma-rad")
    if any(
        receiver_phase_sigmas[name] > 0
        or receiver_phase_drifts[name] != 0
        or receiver_periodic_phase_amplitudes[name] > 0
        or receiver_amplitude_sigmas[name] > 0
        or receiver_chirp_sigmas[name] > 0
        for name in receiver_phase_sigmas
    ):
        for radar_index, name in enumerate(current_dataset().names):
            frame_correlation = per_radar(args, "receiver-noise", "frame-correlation")[name]
            if frame_correlation is None:
                frame_correlation = args.receiver_noise_frame_correlation
            chirp_frame_correlation = per_radar(args, "receiver-noise-chirp-frame", "correlation")[name]
            if chirp_frame_correlation is None:
                chirp_frame_correlation = (
                    args.receiver_noise_chirp_frame_correlation
                )
            receiver_noise_models[name] = CorrelatedReceiverNoise(
                phase_sigma_rad=receiver_phase_sigmas[name],
                phase_drift_rad_per_frame=receiver_phase_drifts[name],
                periodic_phase_amplitude_rad=(
                    receiver_periodic_phase_amplitudes[name]
                ),
                periodic_phase_frequency_cycles_per_frame=(
                    receiver_periodic_phase_frequencies[name]
                ),
                amplitude_sigma_fraction=receiver_amplitude_sigmas[name],
                frame_correlation=frame_correlation,
                chirp_phase_sigma_rad=receiver_chirp_sigmas[name],
                chirp_correlation=args.receiver_noise_chirp_correlation,
                chirp_frame_correlation=chirp_frame_correlation,
                seed=args.receiver_noise_seed + radar_index,
            )
    return ReceiverNoise(
        models=receiver_noise_models,
        jitter=jitter,
    )



@dataclass
class FrameReceiverNuisance:
    """The per-frame receiver nuisance: one model per radar when its scale is
    positive. P1b-2 of docs/CONSOLIDATION_AND_OO_PLAN.md."""

    models: object
    scales: object


def _configure_frame_receiver_nuisance(args, configs, device, rasterizers) -> FrameReceiverNuisance:
    """run()'s setup span, verbatim, as a builder."""
    frame_receiver_nuisance_models = {}
    frame_receiver_nuisance_scales = per_radar(args, "frame-receiver-nuisance", "scale")
    if any(scale > 0 for scale in frame_receiver_nuisance_scales.values()):
        for radar_index, (name, rasterizer) in enumerate(rasterizers.items()):
            correlation = per_radar(args, "frame-receiver-nuisance", "correlation")[name]
            if args.frame_receiver_nuisance_mode == "common-range-gain":
                frame_receiver_nuisance_models[name] = CorrelatedRangeGain(
                    range_bins=256,
                    sigma_fraction=frame_receiver_nuisance_scales[name],
                    frame_correlation=correlation,
                    smoothing_passes=args.frame_receiver_nuisance_range_smoothing,
                    seed=args.frame_receiver_nuisance_seed + radar_index,
                )
            else:
                frame_receiver_nuisance_models[name] = PhaseCoherentDiffuseFloor(
                    rasterizer.tx_positions.detach().cpu().numpy(),
                    rasterizer.rx_positions.detach().cpu().numpy(),
                    config_centre_m(configs[name]),
                    array_axis(configs[name]),
                    wavelength_m=299792458.0 / rasterizer.center_freq,
                    range_bin_spacing_m=range_bin_spacing_m(),
                    active_range_bins=RANGE_KEEP,
                    scatterers_per_bin=(
                        args.frame_receiver_nuisance_scatterers_per_bin
                    ),
                    temporal_correlation=correlation,
                    seed=args.frame_receiver_nuisance_seed + radar_index,
                    backend=args.nuisance_backend,
                    device=str(device),
                    draws=args.nuisance_draws,
                )
    return FrameReceiverNuisance(
        models=frame_receiver_nuisance_models,
        scales=frame_receiver_nuisance_scales,
    )



@dataclass
class LocalAdditiveNuisance:
    """The local additive nuisance: one model per radar when its scale is
    positive. P1b-2 of docs/CONSOLIDATION_AND_OO_PLAN.md."""

    models: object
    scales: object


def _configure_local_additive_nuisance(args, configs, device, rasterizers) -> LocalAdditiveNuisance:
    """run()'s setup span, verbatim, as a builder."""
    local_additive_nuisance_models = {}
    local_additive_scales = per_radar(args, "local-additive-nuisance", "scale")
    if any(scale > 0 for scale in local_additive_scales.values()):
        for radar_index, (name, rasterizer) in enumerate(rasterizers.items()):
            correlation = per_radar(args, "local-additive-nuisance", "correlation")[name]
            local_additive_nuisance_models[name] = (
                PhaseCoherentDiffuseFloor(
                    rasterizer.tx_positions.detach().cpu().numpy(),
                    rasterizer.rx_positions.detach().cpu().numpy(),
                    config_centre_m(configs[name]),
                    array_axis(configs[name]),
                    wavelength_m=299792458.0 / rasterizer.center_freq,
                    range_bin_spacing_m=range_bin_spacing_m(),
                    active_range_bins=RANGE_KEEP,
                    scatterers_per_bin=(
                        args.local_additive_nuisance_scatterers_per_bin
                    ),
                    temporal_correlation=correlation,
                    seed=args.local_additive_nuisance_seed + radar_index,
                    backend=args.nuisance_backend,
                    device=str(device),
                    draws=args.nuisance_draws,
                )
            )
    return LocalAdditiveNuisance(
        models=local_additive_nuisance_models,
        scales=local_additive_scales,
    )



class ReceiverFloor:
    """The receiver floor the trainer learned, drawn per frame in product units.

    The motion checkpoint carries ``noise_floor_power`` (radars, Doppler,
    range): the expected windowed |RD|^2 per cell, angle-mean, that the
    trainer's learned noise model added to its rendered power. The renderer
    reproduces it as white per-channel complex Gaussian noise in the range
    profiles, shaped over (Doppler, range) so that the consumer's transform
    -- mean removal, the Doppler window, the FFT, the beamformer -- shows
    that power in expectation. The shaping amplitude is calibrated against
    the transform itself (a few unit draws through the same functions, a
    fixed-point correction for the window's bin mixing), so it holds to the
    draw count's statistics, not to a model of the chain. Added AFTER the
    absolute gain and the coupling: the trainer's floor is in product units.
    Nothing is read from a calibration file.
    """

    def __init__(self, power_by_radar: dict, seed: int, device, draws: int = 8, iterations: int = 3,
                 array_window_kind: str = "hann"):
        from mm4DPS.signal_model import range_window_profiles
        self.amplitude: dict[str, torch.Tensor] = {}
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(int(seed))
        self.device = device
        self._range_window = range_window_profiles
        # The prior is the angle-mean windowed |RD|^2 the TRAINER measured in ITS beamformer -- under its array window
        # (--array-window; the checkpoint's `array_window`). White channel noise reads 2.67x more power untapered than
        # Hann-tapered, so the draws are calibrated under the same taper, or the floor lands 2.4x (1.55x in amplitude)
        # above the recording's (RT-Pose 164, docs/STATIC_SEQS_NEXT.md 1e).
        with array_window(array_window_kind):
            for radar_name, power in power_by_radar.items():
                radar = current_dataset().radar(radar_name)
                target = torch.as_tensor(power, dtype=torch.float32, device=device)   # (loops, range)
                if tuple(target.shape) != (radar.loops, radar.samples):
                    raise ValueError(f"noise_floor_power for {radar_name} must be (loops, range), got {tuple(target.shape)}")
                amplitude = target.clamp_min(0.0).sqrt()
                for _ in range(iterations):
                    achieved = torch.zeros_like(target)
                    for _ in range(draws):
                        cube = self._draw(radar, amplitude)
                        ra = slow_time_profiles_to_ra_complex(cube.real, cube.imag, radar=radar)
                        achieved += (slow_time_ra_to_doppler_magnitude(ra) ** 2).mean(dim=2) / draws
                    ratio = torch.where(achieved > 0, target / achieved.clamp_min(1e-30), torch.zeros_like(target))
                    amplitude = amplitude * ratio.sqrt()
                self.amplitude[radar_name] = amplitude

    def _draw(self, radar, amplitude: torch.Tensor) -> torch.Tensor:
        shape = (radar.loops, radar.n_tx, radar.n_rx, radar.samples)
        z = torch.complex(
            torch.randn(shape, generator=self.generator, device=self.device),
            torch.randn(shape, generator=self.generator, device=self.device)) / math.sqrt(2.0)
        spectrum = torch.fft.fftshift(torch.fft.fft(z, dim=0), dim=0) * amplitude[:, None, None, :]
        cube = torch.fft.ifft(torch.fft.ifftshift(spectrum, dim=0), dim=0)
        return self._range_window(cube)

    def draw(self, radar_name: str) -> torch.Tensor:
        return self._draw(current_dataset().radar(radar_name), self.amplitude[radar_name])


def _configure_receiver_floor(args, motion, device):
    """The checkpoint floor when it is there and asked for; None otherwise."""
    if args.receiver_floor_from_checkpoint != "on":
        return None
    power = motion.checkpoint.get("noise_floor_power")
    if power is None:
        print("receiver floor: the motion checkpoint carries no noise_floor_power; none drawn", flush=True)
        return None
    radars = motion.checkpoint.get("noise_floor_radars") or current_dataset().names
    floor = ReceiverFloor({name: power[i] for i, name in enumerate(radars)},
                          seed=args.receiver_floor_seed, device=device,
                          array_window_kind=motion.checkpoint.get("array_window", "hann"))
    print(f"receiver floor: drawn from the trained noise model in the checkpoint (prior measured under the "
          f"{motion.checkpoint.get('array_window', 'hann')} array window; "
          + ", ".join(f"{n} mean power {float((a ** 2).mean()):.4e}" for n, a in floor.amplitude.items())
          + ")", flush=True)
    return floor


@dataclass
class SlowTimeNuisance:
    """The within-frame slow-time nuisance, from its cross-sequence calibration.
    P1b-2 of docs/CONSOLIDATION_AND_OO_PLAN.md."""

    models: object
    calibration: object


def _configure_slow_time_nuisance(args, configs, device, diffuse_floor, rasterizers) -> SlowTimeNuisance:
    """run()'s setup span, verbatim, as a builder."""
    slow_time_nuisance_models = {}
    slow_time_nuisance_calibration = (
        json.loads(args.slow_time_nuisance_calibration.read_text())
        if args.slow_time_nuisance_calibration is not None else None
    )
    if (
        args.slow_time_nuisance_scale > 0
        or any((v or 0) > 0 for v in per_radar(args, "slow-time-nuisance", "scale").values())
    ):
        for radar_index, (name, rasterizer) in enumerate(rasterizers.items()):
            nuisance_frame_correlation = per_radar(args, "slow-time-nuisance", "correlation")[name]
            if nuisance_frame_correlation is None:
                nuisance_frame_correlation = (
                    args.slow_time_nuisance_correlation
                )
            spatial_sines = None
            if args.slow_time_nuisance_angle_source == "calibrated":
                slow_time_radar_calibration = (
                    slow_time_nuisance_calibration.get("radars", {}).get(name, {})
                    if slow_time_nuisance_calibration is not None else {}
                )
                if "spatial_sine_by_range" in slow_time_radar_calibration:
                    spatial_sines = np.asarray(
                        slow_time_radar_calibration["spatial_sine_by_range"],
                        dtype=np.float64,
                    )
                elif diffuse_floor.calibration is not None:
                    spatial_sines = np.asarray(
                        diffuse_floor.calibration[
                            "selected_spatial_sine_by_range"
                        ][name],
                        dtype=np.float64,
                    )
                else:
                    raise ValueError(
                        "calibrated slow-time nuisance angles require "
                        "spatial_sine_by_range in the slow-time calibration"
                    )
                if spatial_sines.shape[1] != args.slow_time_nuisance_scatterers_per_bin:
                    raise ValueError(
                        "calibrated slow-time nuisance angles require the same "
                        "scatterer count as the calibration; use --slow-time-nuisance-"
                        "angle-source uniform for a denser field"
                    )
            slow_time_nuisance_models[name] = PhaseCoherentDiffuseFloor(
                rasterizer.tx_positions.detach().cpu().numpy(),
                rasterizer.rx_positions.detach().cpu().numpy(),
                config_centre_m(configs[name]),
                array_axis(configs[name]),
                wavelength_m=299792458.0 / rasterizer.center_freq,
                range_bin_spacing_m=range_bin_spacing_m(),
                active_range_bins=RANGE_KEEP,
                scatterers_per_bin=args.slow_time_nuisance_scatterers_per_bin,
                temporal_correlation=nuisance_frame_correlation,
                seed=args.slow_time_nuisance_seed + radar_index,
                spatial_sine_by_range=spatial_sines,
                backend=args.nuisance_backend,
                device=str(device),
                draws=args.nuisance_draws,
            )
    return SlowTimeNuisance(
        models=slow_time_nuisance_models,
        calibration=slow_time_nuisance_calibration,
    )



@dataclass
class MotionCheckpoint:
    """What the motion checkpoint provides: the velocity-field mode, the joint
    gains and scales, the skinning, and the frame offset. Loaded early because
    the geometry and the trajectories both depend on it. P1b-3."""

    checkpoint: object
    velocity_field_mode: object
    surface_motion: object
    joint_gains: object
    joint_scales: object
    skinning: object
    skinning_mode: object
    skinning_topk: object
    skinning_temperature_m: object
    slow_time_assemble: object
    velocity_rotation: object
    velocity_frame_offset: object
    noise_floor_quantile: object


def _configure_motion_checkpoint(args, device) -> MotionCheckpoint:
    """run()'s setup span, verbatim, as a builder."""
    joint_gains = joint_scales = None
    motion_checkpoint = torch.load(
        args.motion_checkpoint, map_location=device, weights_only=False,
    )
    velocity_field_mode = (
        motion_checkpoint.get("motion_representation") == "velocity_field"
    )
    surface_motion = velocity_field_mode or (
        motion_checkpoint.get("motion_representation") == "surface_points"
    )
    slow_time_assemble = surface_slow_time_assembler()
    if velocity_field_mode:
        # The velocity itself was learned (train_hupr_velocity_field.py):
        # no scales, no rotation. The field replaces the camera velocities
        # after they are computed below.
        joint_scales = torch.ones(14, device=device)
        joint_gains = motion_checkpoint["inherited_joint_scattering_gains"].to(device)
        velocity_rotation = None
    elif surface_motion:
        joint_scales = motion_checkpoint["surface_joint_scales"].to(device)
        joint_gains = motion_checkpoint["inherited_joint_scattering_gains"].to(device)
        if joint_scales.shape not in ((14,), (14, 3)) or joint_gains.shape != (14,):
            raise ValueError("surface checkpoint must contain 14 joint scales and gains")
        velocity_rotation = motion_checkpoint.get("global_velocity_rotation_matrix")
        if velocity_rotation is not None:
            velocity_rotation = velocity_rotation.to(device)
    else:
        if motion_checkpoint.get("motion_representation") == "joint_gains_only":
            # `mm4DPS.train_motion inherit-joint-gains` writes this: the
            # joint scattering gains a velocity-field run needs, and nothing
            # else. It is not a motion model and must not be rendered as one.
            raise ValueError(
                f"{args.motion_checkpoint} carries only the pass-through joint "
                f"scattering gains (motion_representation=joint_gains_only). It "
                f"has no joint scales and no velocity rotation, so it cannot "
                f"drive a render. Render a velocity field instead: the "
                f"surface-motion trainer was removed in the deployment strip "
                f"(git show pre-deploy-strip:mm4DPS/training/motion.py).")
        joint_scales = motion_checkpoint["articulated_joint_scales"].to(device)
        joint_gains = motion_checkpoint["articulated_joint_gains"].to(device)
        if joint_scales.shape != (14,) or joint_gains.shape != (14,):
            raise ValueError("motion checkpoint must contain 14 joint scales and gains")
        velocity_rotation = None
    if args.velocity_scale_mode == "identity":
        # The checkpoint's scales were fit against RD correlation and are
        # anti-physical: they suppress the distal joints that carry fast
        # micro-Doppler (wrist 0.49) and amplify slow proximal ones (hip 4.89).
        # Identity uses the camera-measured joint velocities as they are, which
        # recovers roughly a factor of five of the missing fast-Doppler energy.
        joint_scales = torch.ones_like(joint_scales)
    if args.velocity_scale_floor > 0.0:
        # The checkpoint's scales were fit against RD correlation, which the
        # torso's bulk return dominates, and they ended up suppressing exactly
        # the distal joints that carry fast micro-Doppler (wrist 0.49, elbow
        # 0.47) while amplifying slow proximal ones (hip 4.89). Scaling a
        # joint *below* its camera-measured velocity has no physical basis --
        # the camera velocity is a measurement -- so the magnitude is floored
        # while its direction is kept.
        magnitude = joint_scales.abs() if joint_scales.ndim == 1 else (
            joint_scales.norm(dim=1, keepdim=True)
        )
        boost = torch.clamp(
            args.velocity_scale_floor / magnitude.clamp_min(1e-6), min=1.0,
        )
        joint_scales = joint_scales * boost
    if args.velocity_gain != 1.0:
        # The learned per-joint scales were fit against RD correlation, a
        # metric dominated by the background rather than by the person's
        # micro-Doppler. Measured against the spectrogram they suppress radial
        # velocity roughly threefold, which removes the fast-limb Doppler the
        # pose task depends on. This multiplier restores it without discarding
        # the per-joint structure the checkpoint learned.
        joint_scales = joint_scales * args.velocity_gain
    skinning = motion_checkpoint.get("skinning", {"mode": "hard", "topk": 1})
    skinning_mode = skinning.get("mode", "hard")
    if skinning_mode not in ("hard", "soft", "soft local-surface velocity"):
        raise ValueError(f"unsupported skinning mode: {skinning_mode}")
    skinning_temperature_m = float(skinning.get("temperature_m") or 0.25)
    skinning_topk = int(skinning.get("topk", 1))
    velocity_frame_offset = float(
        motion_checkpoint.get("selected_velocity_frame_offset", 0)
        if args.velocity_frame_offset is None else args.velocity_frame_offset
    )
    noise_floor_quantile = motion_checkpoint.get("noise_floor_quantile")
    return MotionCheckpoint(
        checkpoint=motion_checkpoint,
        velocity_field_mode=velocity_field_mode,
        surface_motion=surface_motion,
        joint_gains=joint_gains,
        joint_scales=joint_scales,
        skinning=skinning,
        skinning_mode=skinning_mode,
        skinning_topk=skinning_topk,
        skinning_temperature_m=skinning_temperature_m,
        slow_time_assemble=slow_time_assemble,
        velocity_rotation=velocity_rotation,
        velocity_frame_offset=velocity_frame_offset,
        noise_floor_quantile=noise_floor_quantile,
    )



@dataclass
class VelocityState:
    """The per-frame velocity products: smoothed joint velocities, and under
    the velocity field the learned normals, point scalars and residual.
    Computed late, from the loaded trajectories. P1b-3."""

    velocities: object
    learned_frame_normals: object
    point_normal_scalars: object
    residual: object
    residual_np: object


def _configure_velocity_state(args, joints, motion) -> VelocityState:
    """run()'s setup span, verbatim, as a builder."""
    residual_np = None
    velocities = smooth_joint_velocities(
        joints, method=args.velocity_differentiator,
    )
    learned_frame_normals: dict[int, np.ndarray] = {}
    point_normal_scalars: dict[int, np.ndarray] = {}
    if motion.velocity_field_mode:
        field = np.asarray(
            motion.checkpoint["joint_velocity_field_mps"].detach().cpu().numpy(),
            dtype=np.float32,
        )
        if field.shape != velocities.shape:
            raise ValueError(
                f"velocity field shape {field.shape} does not match "
                f"camera velocities {velocities.shape}"
            )
        velocities = field
        learned_frame_normals = {
            int(key): np.asarray(value.detach().cpu().numpy(), dtype=np.float32)
            for key, value in motion.checkpoint.get("learned_normals", {}).items()
        }
        point_normal_scalars = {
            int(key): np.asarray(value.detach().cpu().numpy(), dtype=np.float32)
            for key, value in motion.checkpoint.get("point_normal_scalars", {}).items()
        }
        print(
            f"velocity field active ({motion.checkpoint.get('velocity_parameterization')}): "
            f"{len(learned_frame_normals)} frames with learned normals, "
            f"{len(point_normal_scalars)} with point scalars", flush=True,
        )
    velocity_residual = motion.checkpoint.get("surface_joint_velocity_residual_mps")
    if velocity_residual is not None:
        # Learned per-frame per-joint velocity correction (plan Stage A),
        # applied before the scales: v = scales * (v_camera + residual).
        # Adding it here, before the fractional-frame interpolation, is exact
        # because the interpolation is linear.
        if not motion.surface_motion:
            raise ValueError(
                "velocity residual requires a surface-motion checkpoint"
            )
        residual_np = np.asarray(
            velocity_residual.detach().cpu().numpy(), dtype=np.float32,
        )
        if residual_np.shape != velocities.shape:
            raise ValueError(
                "velocity residual shape "
                f"{residual_np.shape} does not match velocities {velocities.shape}"
            )
        velocities = velocities + residual_np
        print(
            "velocity residual active: rms "
            f"{float(np.sqrt(np.mean(residual_np ** 2))):.4f} m/s, max "
            f"{float(np.max(np.abs(residual_np))):.4f} m/s", flush=True,
        )
    return VelocityState(
        velocities=velocities,
        learned_frame_normals=learned_frame_normals,
        point_normal_scalars=point_normal_scalars,
        residual=velocity_residual,
        residual_np=residual_np,
    )



@dataclass
class LoopMetrics:
    """The three accumulators the frame loop fills and the payload reports --
    the only state that crosses from the loop to the teardown. P1b-3."""

    correlations: object
    fixed_track: object
    background_profile_errors: object


def _configure_loop_metrics() -> LoopMetrics:
    """run()'s setup span, verbatim, as a builder."""
    correlations = {
        name: {
            "combined": [], "strict_dra": [], "angle_integrated_rd": [],
            # Linear-domain counterparts. The three above go through
            # `log_pearson_loss` (dB, -42 dB floor); these are the ones
            # that match the project's linear-amplitude convention.
            "strict_dra_linear": [], "angle_integrated_rd_linear": [],
        }
        for name in current_dataset().names
    }
    fixed_track_correlations: dict[str, list[float | None]] = {
        name: [] for name in current_dataset().names
    }
    background_profile_errors: dict[str, list[float]] = {
        name: [] for name in current_dataset().names
    }
    return LoopMetrics(
        correlations=correlations,
        fixed_track=fixed_track_correlations,
        background_profile_errors=background_profile_errors,
    )



@dataclass
class SceneGeometry:
    """The rig's placement: phase model, boresight translation, depth scale. P1b-4."""

    phase_model: object
    translation_y_m: object
    depth_scale: object


def _configure_scene_geometry(args, checkpoint, motion) -> SceneGeometry:
    """run()'s setup span, verbatim, as a builder."""
    phase_model = (
        "per-point bistatic" if motion.surface_motion
        else motion.checkpoint.get("phase_model", "monostatic")
    )
    geometry_translation_y_m = float(
        checkpoint.get(
            "geometry_translation_y_m",
            motion.checkpoint.get("geometry_translation_y_m", 0.0),
        )
        if args.geometry_translation_y_m is None else args.geometry_translation_y_m
    )
    geometry_depth_scale = float(
        checkpoint.get("geometry_depth_scale", 1.0)
        if args.geometry_depth_scale is None else args.geometry_depth_scale
    )
    return SceneGeometry(
        phase_model=phase_model,
        translation_y_m=geometry_translation_y_m,
        depth_scale=geometry_depth_scale,
    )



@dataclass
class MotionFrames:
    """Which frames the metrics may use: the motion detector's reliable set and
    the person track records it came from. P1b-4."""

    detection_protocol: object
    reliable: object
    fixed_range_key: object
    track_records: object


def _configure_motion_frames(args) -> MotionFrames:
    """run()'s setup span, verbatim, as a builder."""
    detection = json.loads(args.motion_detection_result.read_text())
    reliable_frames = set(detection["accepted_frames"])
    track_records = {
        (int(row["frame"]), radar): row["radars"][radar]
        for row in detection["frame_records"] for radar in current_dataset().names
    }
    fixed_range_key = "observed_range_m"
    if args.pose_audit_result is not None:
        pose_audit = json.loads(args.pose_audit_result.read_text())
        reliable_frames = set(pose_audit["pose_motion_frames"])
        fixed_range_key = "corrected_cloud_range_m"
        detection_protocol = (
            "independent camera-pose motion + calibrated camera-person range, "
            f"{len(reliable_frames)}/600 frames"
        )
    else:
        detection_protocol = (
            f"full-sequence predeclared H/V detector, {len(reliable_frames)}/600 frames"
        )
    if not track_records:
        raise ValueError("calibration result must contain track_records")
    return MotionFrames(
        detection_protocol=detection_protocol,
        reliable=reliable_frames,
        fixed_range_key=fixed_range_key,
        track_records=track_records,
    )



@dataclass
class Background:
    """The static scene: its rasterizers, the trained background primitives, and
    the per-frame visibility mask or None. P1b-4."""

    rasterizers: object
    primitives: object
    state: object
    area_weights: object
    range_point_counts: object
    visibility: object


def _configure_background(args, checkpoint, device) -> Background:
    """run()'s setup span, verbatim, as a builder."""
    from mm4DPS.rasterizer import PointPrimitives, cull_gaussians, render_gaussians
    _, rasterizers = build_rasterizers(args, device)
    state = checkpoint["shared_background_state"]
    background = PointPrimitives(len(state["positions"]), device=str(device))
    background.load_state_dict(state)
    background.requires_grad_(False).eval()
    # Scene gains learned by the velocity trainer (per-point, linear in the
    # renderer's vertex_areas weight); ones when the checkpoint predates them.
    gains = checkpoint.get("background_point_gains")
    background_area_weights = (torch.ones(len(state["positions"])) if gains is None else torch.as_tensor(gains)).to(device)
    # Reconstruction sampling density per range bin, used to strip the
    # monocular point-count artefact out of the scene-proportional floor.
    background_range_point_counts = np.bincount(
        np.clip(
            (
                np.linalg.norm(
                    background.positions.detach().cpu().numpy(), axis=1,
                ) / range_bin_spacing_m()
            ).astype(np.int64),
            0, 255,
        ),
        minlength=256,
    ).astype(np.float64)
    # Per-frame scene occlusion by the person. Precomputed offline by
    # `mm4DPS.preprocessing.pointset scene-shadow-mask` against THIS
    # checkpoint's background points -- the mask is data, nothing ray-casts
    # here. Resident on device as one (frames, N) bool; the loop indexes it.
    scene_visibility = None
    if args.scene_visibility is not None:
        loaded = np.load(args.scene_visibility)
        if loaded.shape[1] != len(background.positions):
            raise ValueError(
                f"scene visibility has {loaded.shape[1]} points, the "
                f"checkpoint background has {len(background.positions)}; the "
                "mask must be built against the same checkpoint")
        if loaded.shape[0] < args.frame_count:
            raise ValueError(
                f"scene visibility covers {loaded.shape[0]} frames, "
                f"render needs {args.frame_count}")
        scene_visibility = torch.from_numpy(loaded).to(device)
        print(f"per-frame scene visibility: {args.scene_visibility} "
              f"{tuple(loaded.shape)}, "
              f"{float((~loaded).sum(1).mean()):.0f} points shadowed per frame",
              flush=True)
    return Background(
        rasterizers=rasterizers,
        primitives=background,
        state=state,
        area_weights=background_area_weights,
        range_point_counts=background_range_point_counts,
        visibility=scene_visibility,
    )



@dataclass
class BackgroundProfiles:
    """The background's culled masks and its range profiles per radar, rendered
    once. P1b-4."""

    masks: object
    profiles: object


def _configure_background_profiles(receiver_noise, scene) -> BackgroundProfiles:
    """run()'s setup span, verbatim, as a builder."""
    from mm4DPS.rasterizer import PointPrimitives, cull_gaussians, render_gaussians
    background_masks = {}
    background_profiles = {}
    with torch.no_grad():
        for name, rasterizer in scene.rasterizers.items():
            mask = cull_gaussians(
                scene.primitives, rasterizer, near_field_m=HUPR_NEAR_FIELD_M,
            )
            background_masks[name] = mask
            if receiver_noise.jitter is None and scene.visibility is None:
                # use_cuda_kernels=False: the fused step5 kernel is built
                # with --use_fast_math, whose sincos on ~1e4 rad carrier
                # phases deviates from proper range reduction; coherent
                # cancellation amplifies that to ~24% in the wall's speckle
                # bins. The velocity trainer's background uses the scatter
                # path, so the deliverable render must match it.
                real, imag = render_gaussians(
                    scene.primitives, rasterizer,
                    scene.area_weights * mask.float(), active_mask=mask,
                    shadow_mask=None, detach_phase=True, bsdf_mode="full",
                    disabled_components=None, use_cuda_kernels=False,
                )
                background_profiles[name] = torch.complex(real, imag)
    return BackgroundProfiles(
        masks=background_masks,
        profiles=background_profiles,
    )



@dataclass
class Person:
    """The person: the trained shared (or per-point) material, the materials
    closure the loop calls per frame, and the joints. P1b-5."""

    material: object
    current_index: object
    materials_for: object
    joints: object


def _configure_person(args, checkpoint, device, rig) -> Person:
    """run()'s setup span, verbatim, as a builder."""
    person_raw = checkpoint["shared_human_raw_material"].to(device)
    person_per_point = person_raw.ndim == 2
    human = SharedHumanMaterial(
        person_raw[0] if person_per_point else person_raw,
        float(checkpoint["shared_human_area_scale"]),
    ).to(device)
    if person_per_point:
        human.per_point_raw = person_raw
        print(f"person material: per-point, {tuple(person_raw.shape)}", flush=True)
    current_person_index: dict = {"index": None}

    def person_materials(count: int) -> torch.Tensor:
        """The person's material rows for the frame currently being rendered."""
        if not person_per_point:
            return human.for_points(count)
        index = current_person_index["index"]
        if index is None or index.shape[0] != count:
            raise RuntimeError(
                "per-point person material needs this frame's global point "
                f"index; have {None if index is None else index.shape[0]} "
                f"rows for {count} points")
        return human.per_point_raw.index_select(0, index)
    if "person_radar_log_gain_delta" in checkpoint:
        human.radar_log_gain_delta.data.copy_(
            checkpoint["person_radar_log_gain_delta"].to(device)
        )
    human.requires_grad_(False).eval()
    joints = np.stack([
        np.load(args.sequence_dir / "person_frames" / f"frame_{frame:03d}.npz")["joints_xyz"]
        for frame in range(sequence_frame_count(args.sequence_dir))
    ]).astype(np.float32)
    joints = transform_person_joints(
        joints, rig.depth_scale, rig.translation_y_m,
    )
    return Person(
        material=human,
        current_index=current_person_index,
        materials_for=person_materials,
        joints=joints,
    )



@dataclass
class Products:
    """What the render writes: the product writer and its declared products,
    the DRA frame index, and the ffmpeg pipe when a video is wanted. P1b-5."""

    write_video: object
    chunk_frames: object
    writer: object
    dra_frame_to_index: object
    ffmpeg: object
    stage_cube: object          # (key, frame, complex frame) -> staged in the product's storage form
    scales: object              # {key: (frames,) float32 device tensor} for the half-precision form


def _configure_products(args, device, frames) -> Products:
    """run()'s setup span, verbatim, as a builder."""
    write_video = args.write_video == "on"
    # The diagnostic video annotates each frame with that frame's own
    # correlations and RD maps, so with it on the loop cannot run ahead of the
    # host: the chunk is one frame and the writer behaves as the memmap did.
    # The media chain and every gate run with it off.
    product_chunk_frames = 1 if write_video else max(
        1, int(args.product_chunk_frames))
    writer = ProductWriter(args.product_dir, product_chunk_frames, device)
    product_shape = (args.frame_count, 64, 256)
    dataset = current_dataset()
    for name in dataset.names:
        for kind in ("measured", "rendered"):
            writer.add(f"{name}_{kind}_rd", product_shape, np.float32)
    # THE PRODUCT. Complex range-Doppler, every channel, no bin dropped:
    # (frames, loops, tx, rx, samples) complex64 -- HuPR (64, 3, 4, 256),
    # 1.57 MB a frame; the cascade (64, 12, 16, 256), 12.6 MB a frame.
    # Everything else this writer holds is a view of it.
    # The measured cube is the recording alone -- the archive decoded and
    # range-windowed -- so it is not stored: the one consumer that read it
    # (the spectrogram stage, `media.MeasuredCube`) decodes it from the
    # archive frame by frame, 2 s a radar. `--save-measured-cube on` writes
    # it beside the rendered one (a 7.5 GB cascade file) for a consumer
    # outside this repository.
    # The cube's storage: float16 (real, imag) pairs with one float32 scale
    # per frame, the GT ADC's four bytes a complex sample (`data.ProductCube`
    # reads it back as complex64; gated at 4.9e-4 of every cell, docs/).
    for radar in dataset.radars:
        for kind in (("measured", "rendered") if args.save_measured_cube == "on" else ("rendered",)):
            writer.add(f"{radar.name}_{kind}", (args.frame_count, *radar.product_shape, 2), np.float16)
    product_scales = {f"{radar.name}_{kind}": torch.zeros(args.frame_count, dtype=torch.float32, device=device)
                      for radar in dataset.radars
                      for kind in (("measured", "rendered") if args.save_measured_cube == "on" else ("rendered",))}

    def stage_cube(key: str, frame: int, value: torch.Tensor) -> None:
        """Hand one frame of a cube to the writer as float16 pairs at the frame's own scale."""
        # the frame's peak at 2^15: every cell keeps float16's eleven
        # significant bits of its own magnitude down to 2^-14 of the
        # peak, and the scale is a device scalar -- no host sync
        scale = value.abs().amax().clamp_min(1e-30) / 32768.0
        product_scales[key][frame] = scale
        writer.append(key, torch.view_as_real(value / scale).to(torch.float16))
    if args.save_raw_rd_products:
        for name in current_dataset().names:
            for kind in ("measured", "rendered"):
                writer.add(f"{name}_{kind}_rd_raw", product_shape, np.float32)
    if args.save_first_loop_ra_products:
        ra_shape = (args.frame_count, dataset.radars[0].samples, dataset.radars[0].n_angle)
        for name in current_dataset().names:
            for kind in ("measured", "rendered"):
                writer.add(f"{name}_{kind}_first_loop_ra_complex",
                           ra_shape, np.complex64)
                # the same loop in the downstream consumers' view: no azimuth taper (HuPR's released preprocessing, the
                # HRRadarPose adapter); the fidelity suite scores both views (docs/STATIC_SEQS_NEXT.md 1d)
                writer.add(f"{name}_{kind}_first_loop_ra_untapered_complex",
                           ra_shape, np.complex64)
                # the same loop before any angle transform: every channel's complex range profile (tx, rx, range),
                # what the suite's complex-range-profile, ADC and range-azimuth-elevation scores read
                r = dataset.radar(name)
                writer.add(f"{name}_{kind}_first_loop_rp_complex",
                           (args.frame_count, r.n_tx, r.n_rx, r.samples), np.complex64)
    if args.save_radproposer_products:
        if dataset.name != "hupr":
            raise ValueError("RadProposer products are HuPR's published point proposals; "
                             "--save-radproposer-products is HuPR-only")
        radproposer_shape = (
            args.frame_count,
            2,
            *RADPROPOSER_SINGLE_RADAR_SHAPE,
        )
        for kind in ("measured", "rendered"):
            # Two rows per frame: the radar axis is filled inside the radar
            # loop, horizontal then vertical, which is the order the rows
            # occupy in the file.
            writer.add(f"{kind}_radproposer_complex", radproposer_shape,
                       np.complex64,
                       row_shape=RADPROPOSER_SINGLE_RADAR_SHAPE,
                       rows_per_frame=2)
    dra_frame_to_index: dict[int, int] = {}
    if args.save_pose_motion_dra:
        dra_frames = sorted(frame for frame in frames.reliable if frame < args.frame_count)
        dra_frame_to_index = {frame: index for index, frame in enumerate(dra_frames)}
        np.save(args.product_dir / "pose_motion_dra_frames.npy", np.asarray(dra_frames, dtype=np.int32))
        dra_shape = (len(dra_frames), int(official_motion_doppler_mask().sum()),
                     dataset.radars[0].samples, dataset.radars[0].n_angle)
        for name in current_dataset().names:
            for kind in ("measured", "rendered"):
                writer.add(f"{name}_{kind}_motion_dra", dra_shape, np.float32)
                if args.save_pose_motion_dra_complex:
                    writer.add(f"{name}_{kind}_motion_dra_complex",
                               dra_shape, np.complex64)
    ffmpeg = None
    if write_video:
        ffmpeg = subprocess.Popen([
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(args.output),
        ], stdin=subprocess.PIPE)
    print(f"product writer: {product_chunk_frames}-frame chunks, "
          f"{writer.pinned_bytes / 1e6:.0f} MB staged", flush=True)
    return Products(
        write_video=write_video,
        chunk_frames=product_chunk_frames,
        writer=writer,
        dra_frame_to_index=dra_frame_to_index,
        ffmpeg=ffmpeg,
        stage_cube=stage_cube,
        scales=product_scales,
    )



@dataclass
class Calibrations:
    """The rig's fitted constants applied at render time: antenna coupling, the
    absolute receiver gains, virtual-channel gains, the motion Doppler mask
    and the range keep mask. P1b-5."""

    antenna_coupling: object
    absolute_gains: object
    virtual_channel_gains: object
    motion_doppler: object
    keep_range: object


def _configure_calibrations(args, device) -> Calibrations:
    """run()'s setup span, verbatim, as a builder."""
    dataset = current_dataset()
    antenna_coupling = {name: None for name in dataset.names}
    if args.antenna_coupling_calibration is not None:
        for radar in dataset.radars:
            channels = np.load(
                args.antenna_coupling_calibration / f"{radar.name}_coupling.npy")
            padded = np.zeros((radar.virtual_channels, radar.samples), dtype=np.complex64)
            padded[:, :channels.shape[1]] = channels
            # Virtual channel c is (azimuth TX c // n_rx, RX c % n_rx), the
            # order virtual_array_slow_time builds.
            block = np.zeros((1, *radar.product_shape[1:]), dtype=np.complex64)
            for i, tx in enumerate(radar.azimuth_tx):
                block[0, tx] = padded[i * radar.n_rx:(i + 1) * radar.n_rx]
            antenna_coupling[radar.name] = torch.from_numpy(
                block * args.antenna_coupling_scale).to(device)
    absolute_gains = per_radar(args, "absolute-gain")
    virtual_channel_gains = {}
    if args.virtual_channel_calibration is not None:
        channel_profile = json.loads(
            args.virtual_channel_calibration.read_text()
        )
        for radar in dataset.radars:
            entry = channel_profile["radars"][radar.name]
            virtual_channel_gains[radar.name] = torch.from_numpy(
                virtual_channel_gain_tensor(
                    entry["channel_magnitude"], entry["channel_phase_rad"],
                    radar.n_tx, radar.n_rx,
                ).astype(np.complex64)
            ).to(device)
    motion_doppler = torch.from_numpy(official_motion_doppler_mask()).to(device)
    keep_range = torch.from_numpy(RANGE_KEEP).to(device)
    return Calibrations(
        antenna_coupling=antenna_coupling,
        absolute_gains=absolute_gains,
        virtual_channel_gains=virtual_channel_gains,
        motion_doppler=motion_doppler,
        keep_range=keep_range,
    )



@dataclass
class Measured:
    """The measured side: where each radar's recording is, and how a frame is read.

    The measured branch is decoded from the recording every render: 2 s per
    radar. A 900 MB on-disk cache used to save that, and served nothing else.
    HuPR reads its DCA1000 archive members; the cascade reads the four TDA2
    device files and applies TI's per-channel calibration on the device.
    """

    dataset: Dataset
    archive: object                      # HuPR: the zip; the cascade: the sequence directory
    calibration: dict = field(default_factory=dict)   # cascade: (rx, slot, sample) per radar

    def open(self, stack, sequence: int) -> dict[str, object]:
        """One sequential reader per radar, entered on ``stack``."""
        sources = {}
        archive = None
        for radar in self.dataset.radars:
            if radar.decoder == "dca1000":
                archive = archive or stack.enter_context(zipfile.ZipFile(self.archive))
                sources[radar.name] = stack.enter_context(
                    archive.open(f"single_{sequence}/{radar.member}/adc_data.bin"))
            elif radar.decoder == "tda2":
                sources[radar.name] = CascadeRecording(Path(self.archive), radar)
            else:
                raise NotImplementedError(f"decoder {radar.decoder!r}")
        return sources

    @staticmethod
    def frame_words(source, radar) -> np.ndarray:
        """The next frame's raw int16 words, as recorded."""
        if radar.decoder == "tda2":
            return source.frame_words()
        return np.frombuffer(source.read(radar.bytes_per_frame), dtype=np.int16)

    def decode(self, words: np.ndarray, name: str, device) -> torch.Tensor:
        """Raw words -> ``(rx, chirp, sample)`` complex64 on the device.

        np.array(copy=True): the words are read-only -- a frombuffer view
        over the zip payload -- and torch.from_numpy on a read-only buffer
        yields a tensor it considers unsafe to write.
        """
        radar = self.dataset.radar(name)
        tensor = torch.from_numpy(np.array(words, dtype=np.int16, copy=True)).to(device)
        if radar.decoder == "tda2":
            frame = decode_tda2_frame_torch(tensor, radar)
            return apply_cascade_calibration(frame, self.calibration[name], radar)
        return decode_dca1000_frame_torch(tensor, radar)


def _configure_measured(args) -> Measured:
    """run()'s setup span, verbatim, as a builder."""
    dataset = Dataset.of(args)
    if dataset.radars[0].decoder == "tda2":
        config_dir = getattr(args, dataset.radars[0].attr("", "config")).parent
        calibration = {
            radar.name: cascade_calibration(radar, config_dir / radar.calibration)
            for radar in dataset.radars}
        return Measured(dataset=dataset, archive=dataset.sequence_dir(args.sequence),
                        calibration=calibration)
    return Measured(dataset=dataset, archive=find_archive(args.dataset_root, args.sequence))


def _result_payload(
    args,
    *,
    metrics,
    frames,
    diffuse_floor,
    frame_receiver,
    velocity,
    local_additive,
    measured,
    motion,
    rig,
    scene,
    receiver_noise,
    slow_time,
    products,
) -> dict:
    """render_result.json, from what the setup and the loop produced.

    Pure data: every value here is read from an input or derived from one
    in place; nothing is computed that the frame loop did not. The 36
    keyword parameters are the complete list of what the result depends on,
    made explicit by P1a of docs/CONSOLIDATION_AND_OO_PLAN.md so that the
    component classes of P1b can each own their slice of it.
    """
    first_radar = current_dataset().names[0]
    return {
        "status": "completed", "frames": args.frame_count, "fps": FPS,
        "measured_cube": args.save_measured_cube,
        "product_dtype": "float16 (real, imag) pairs, a float32 scale per frame",
        "dataset": current_dataset().name,
        "dataset_sequence": args.sequence,
        "frame_range_inclusive": [0, args.frame_count - 1],
        "dataset_root": str(args.dataset_root.resolve()),
        "model_label": args.model_label,
        "duration_seconds": args.frame_count / FPS,
        "slow_time_loops_per_frame": 64, "physical_tx_chirps_per_frame": 192,
        "motion_representation": (
            "soft-skinned local surface velocity with per-point bistatic phase"
            if motion.surface_motion else "joint-group carrier phase"
        ),
        "velocity_scale": (
            "14x3 learned joint-axis values" if motion.joint_scales.ndim == 2
            else "14 learned joint-specific values"
        ),
        "velocity_scale_choice": (
            "initialized from validation-selected global 0.95, then jointly optimized "
            "with 14 shared scattering gains using independent validation early selection"
        ),
        "velocity_gain": args.velocity_gain,
        "velocity_scale_floor": args.velocity_scale_floor,
        "velocity_scale_mode": args.velocity_scale_mode,
        "velocity_differentiator": args.velocity_differentiator,
        "joint_velocity_scales": motion.joint_scales.cpu().tolist(),
        "joint_scattering_gains": motion.joint_gains.cpu().tolist(),
"person_amplitude_contract": (
            "the person's amplitude is trained end to end -- "
            "shared_human_area_scale and person_radar_log_gain_delta in the "
            "person checkpoint -- and no constant is applied on top of it. "
            "The scale multipliers that used to sit here were a second, "
            "untrained amplitude model competing with the first"
        ),
        "person_checkpoint": str(args.person_checkpoint.resolve()),
        "motion_checkpoint": str(args.motion_checkpoint.resolve()),
        "skinning": motion.skinning,
        "velocity_field": (
            {"enabled": False} if not motion.velocity_field_mode else {
                "enabled": True,
                "parameterization": motion.checkpoint.get(
                    "velocity_parameterization"
                ),
                "frames_with_learned_normals": len(velocity.learned_frame_normals),
                "frames_with_point_scalars": len(velocity.point_normal_scalars),
            }
        ),
        "joint_velocity_residual": (
            {"enabled": False} if velocity.residual is None else {
                "enabled": True,
                "bound_mps": motion.checkpoint.get(
                    "joint_velocity_residual_bound_mps"
                ),
                "rms_mps": float(np.sqrt(np.mean(velocity.residual_np ** 2))),
                "max_abs_mps": float(np.max(np.abs(velocity.residual_np))),
            }
        ),
        "selected_velocity_frame_offset": motion.velocity_frame_offset,
        "noise_floor_quantile": motion.noise_floor_quantile,
        "phase_model": rig.phase_model,
        "geometry_translation_y_m": rig.translation_y_m,
        "geometry_depth_scale": rig.depth_scale,
        "near_field_guard_m": HUPR_NEAR_FIELD_M,
        "first_retained_range_bin": FIRST_RANGE_BIN,
        "near_field_applied_symmetrically_to_measured_and_rendered": True,
        "per_frame_scene_visibility": (
            {"enabled": False,
             "note": "the scene contributes identically in every frame"}
            if scene.visibility is None else {
                "enabled": True,
                "mask": str(args.scene_visibility),
                "background_points": int(scene.visibility.shape[1]),
                "shadowed_per_frame_mean": round(
                    float((~scene.visibility).sum(1).double().mean().item()), 1),
                "model": "binary ray-cast occlusion of the checkpoint's "
                         "background points by the posed SMPL mesh, "
                         "precomputed offline",
                "applied_as": "shadow_mask on the background render, "
                              "re-rendered once per 10 Hz frame and held "
                              "fixed across the 64 loops within each frame",
            }
        ),
        "static_scene_jitter": (
            {"enabled": False}
            if receiver_noise.jitter is None else {
                "enabled": True,
                **receiver_noise.jitter.metadata(
                    wavelength_m=299792458.0 / next(iter(scene.rasterizers.values())).center_freq,
                ),
                "background_rerendered_once_per_10hz_frame": True,
                "background_held_fixed_across_64_loops_within_each_frame": True,
                "same_perturbed_points_reused_for_horizontal_and_vertical": True,
                "motion_scale": args.static_jitter_motion_scale,
                "motion_frames": len(frames.reliable),
            }
        ),
        "receiver_noise": (
            {"enabled": False}
            if not receiver_noise.models else {
                "enabled": True,
                "applied_domain": (
                    "complex slow-time TX/RX range profiles before additive "
                    "thermal floor and before angle/Doppler FFTs"
                ),
                "background_position_jitter_enabled": receiver_noise.jitter is not None,
                "models": {
                    name: model.metadata()
                    for name, model in receiver_noise.models.items()
                },
            }
        ),
        "unsupported_range_diffuse_floor": (
            {"enabled": False}
            if not diffuse_floor.models else {
                "enabled": True,
                "calibration": (
                    str(args.diffuse_floor_calibration.resolve())
                    if args.diffuse_floor_calibration is not None else None
                ),
                "range_profile_amplitudes": {
                    name: {
                        "near": float(values[
                            (diffuse_floor.bin_ranges >= DIFFUSE_FLOOR_LEGACY_NEAR_M[0])
                            & (diffuse_floor.bin_ranges < DIFFUSE_FLOOR_LEGACY_NEAR_M[1])
                        ][0]),
                        "far": float(values[
                            (diffuse_floor.bin_ranges >= DIFFUSE_FLOOR_LEGACY_FAR_M[0])
                            & (diffuse_floor.bin_ranges <= DIFFUSE_FLOOR_LEGACY_FAR_M[1])
                        ][0]),
                    }
                    for name, values in diffuse_floor.range_amplitudes.items()
                },
                "reference_quantile": args.diffuse_floor_reference_quantile,
                "amplitude_scale": args.diffuse_floor_scale,
                "near_range_m": [
                    *DIFFUSE_FLOOR_LEGACY_NEAR_M,
                ],
                "far_range_m": [
                    *DIFFUSE_FLOOR_LEGACY_FAR_M,
                ],
                "radars_are_independent_nuisance_fields": True,
                "held_fixed_across_64_loops_within_each_frame": True,
                "models": {
                    name: model.metadata() for name, model in diffuse_floor.models.items()
                },
            }
        ),
        "within_frame_slow_time_nuisance": (
            {"enabled": False}
            if not slow_time.models else {
                "enabled": True,
                "model": (
                    "loop-varying latent diffuse scatterers; shared complex coefficient "
                    "across TX/RX with exact bistatic steering phase"
                ),
                "scale_relative_to_background_range_envelope": (
                    args.slow_time_nuisance_scale
                ),
                "radar_specific_scales": {
                    name: args.slow_time_nuisance_scale if value is None else value
                    for name, value in per_radar(args, "slow-time-nuisance", "scale").items()
                },
                "range_envelope_source": (
                    "flat retained-range thermal floor"
                    if args.slow_time_nuisance_mode == "thermal" else
                    "analytic Lorentzian phase-noise ridge plus weak white floor"
                    if args.slow_time_nuisance_mode == "phase-noise" else
                    "rendered static-background magnitude (legacy ablation)"
                    if slow_time.calibration is None else
                    (
                        "measured 2-D Doppler-range expectation with person range removed"
                        if "normalized_doppler_range_profile" in
                        slow_time.calibration["radars"][first_radar] else
                        "measured nonzero-Doppler range calibration with person range removed"
                    )
                ),
                "spectral_synthesis": (
                    args.slow_time_nuisance_mode == "phase-noise"
                    or (
                        args.slow_time_nuisance_mode == "spectral"
                        and slow_time.calibration is not None
                        and "normalized_doppler_range_profile" in
                        slow_time.calibration["radars"][first_radar]
                    )
                ),
                "frame_fixed_fraction": per_radar(args, "slow-time-nuisance", "fixed-fraction"),
                "scene_proportional_range_fraction": {
                    **per_radar(args, "slow-time-nuisance", "scene-fraction"),
                    "range_smoothing_bins": (
                        args.slow_time_nuisance_scene_smoothing_bins
                    ),
                    "model": (
                        "fraction of the calibrated range marginal reallocated "
                        "onto the rendered background's own per-range magnitude; "
                        "the calibrated Doppler shape is preserved exactly and "
                        "the scene profile is smoothed so only its broad "
                        "clutter envelope is transferred"
                    ),
                },
                "analytic_phase_noise": (
                    None if args.slow_time_nuisance_mode != "phase-noise" else {
                        **{f"{name}_knee_bins": value for name, value in
                           per_radar(args, "slow-time-nuisance", "knee-bins").items()},
                        **{f"{name}_white_power_fraction": value for name, value in
                           per_radar(args, "slow-time-nuisance", "white-power-fraction").items()},
                    }
                ),
                "calibration": (
                    None if args.slow_time_nuisance_calibration is None else
                    str(args.slow_time_nuisance_calibration.resolve())
                ),
                "minimum_range_amplitude_before_scale": (
                    args.slow_time_nuisance_min_amplitude
                ),
                "frame_spectral_innovation_correlations": {
                    name: model.temporal_correlation
                    for name, model in slow_time.models.items()
                },
                "empty_far_spectral_innovation_correlations": per_radar(args, "slow-time-nuisance", "empty-correlation"),
                "mechanism": args.slow_time_nuisance_mode,
                "angle_source": args.slow_time_nuisance_angle_source,
                "seed": args.slow_time_nuisance_seed,
                "scope_warning": (
                    "weak coherent hardware/diffuse nuisance ablation; not claimed "
                    "static-scene physical velocity"
                ),
                "models": {
                    name: model.metadata()
                    for name, model in slow_time.models.items()
                },
            }
        ),
        "frame_receiver_nuisance": (
            {"enabled": False}
            if not frame_receiver.models else {
                "enabled": True,
                "applied_domain": (
                    "multiplicative positive real range gain after rendering "
                    "and before angle/Doppler FFTs"
                    if args.frame_receiver_nuisance_mode == "common-range-gain"
                    else
                    "additive complex TX/RX range profiles after rendering "
                    "and before angle/Doppler FFTs"
                ),
                "held_constant_within_64_loop_burst": True,
                "does_not_create_new_nonzero_doppler": True,
                "mode": args.frame_receiver_nuisance_mode,
                "phase_contract": (
                    "common-range-gain mode shares one positive real gain "
                    "across all TX/RX and loops at each range bin"
                    if args.frame_receiver_nuisance_mode == "common-range-gain"
                    else
                    "amplitude-aligned mode projects the perturbation onto "
                    "the current complex profile, preserving TX/RX phase"
                    if args.frame_receiver_nuisance_mode == "amplitude-aligned"
                    else "complex coherent additive perturbation"
                ),
                "radar_specific_scales": frame_receiver.scales,
                "models": {
                    name: model.metadata()
                    for name, model in frame_receiver.models.items()
                },
            }
        ),
        "local_additive_receiver_nuisance": (
            {"enabled": False}
            if not local_additive.models else {
                "enabled": True,
                "applied_domain": (
                    "complex TX/RX range profiles after rendering and before "
                    "angle/Doppler FFTs"
                ),
                "held_constant_within_64_loop_burst": True,
                "does_not_create_new_nonzero_doppler": True,
                "radar_specific_scales": local_additive.scales,
                "phase_contract": (
                    "each latent scatterer uses one complex coefficient shared "
                    "across all TX/RX paths with exact bistatic steering phase"
                ),
                "models": {
                    name: model.metadata()
                    for name, model in local_additive.models.items()
                },
            }
        ),
        "rendered_motion_groups": "14 HuPR skeleton joints",
        "display_body_regions": list(BODY_REGION_NAMES),
        "dynamic_doppler_correlation": {
            name: {
                metric: {"mean": float(np.mean(values)), "std": float(np.std(values))}
                for metric, values in metrics.items()
            }
            for name, metrics in metrics.correlations.items()
        },
        "per_frame_dynamic_doppler_correlation": metrics.correlations,
        "dynamic_doppler_frames": sorted(
            frame for frame in frames.reliable if frame < args.frame_count
        ),
        "virtual_channel_calibration": (
            None if args.virtual_channel_calibration is None
            else str(args.virtual_channel_calibration.resolve())
        ),
        "absolute_receiver_gain": {
            **per_radar(args, "absolute-gain"),
            "model": (
                "one frozen constant per radar applied to the complex range "
                "profiles after all nuisance and before every derived product, "
                "representing ADC/amplifier scaling. Scale-invariant metrics "
                "are unaffected by construction; only the absolute-scale gate "
                "responds. Must be fit from held-out sequences"
            ),
        },
        "background_range_profile_shape_error": {
            "model": (
                "mean absolute error of the peak-normalized near-DC range "
                "profile shown in the video panels; defined on every frame, "
                "unlike the person micro-Doppler correlation. Not a "
                "correlation: the displayed map is clutter-removed, so its "
                "static-frame content is nuisance whose rendered and measured "
                "realizations are independent by construction"
            ),
            **{
                name: {
                    "mean": float(np.mean(values)) if values else None,
                    "max": float(np.max(values)) if values else None,
                    "per_frame": [float(value) for value in values],
                }
                for name, values in metrics.background_profile_errors.items()
            },
        },
        "fixed_person_range_rd_correlation": {
            name: {
                "mean": float(np.mean([value for value in values if value is not None])),
                "frames": sum(value is not None for value in values),
                "coverage_fraction": (
                    sum(value is not None for value in values) / args.frame_count
                ),
                "per_frame": values,
            }
            for name, values in metrics.fixed_track.items()
        },
        "evaluation_protocol": (
            "HuPR central 16 Doppler bins minus zero±1; metric person range-angle "
            "ROI on independently labelled camera-pose-motion frames only; "
            "combined=0.25 strict DRA + 0.75 angle-integrated RD"
        ),
        "video_metric_protocol": (
            "angle-mean RD correlation in a fixed +/-4 range-bin person ROI; "
            + frames.detection_protocol
        ),
        "capture": capture_summary(),
        "measured_branch_cache": (
            {"enabled": False, "note": "removed: the ADC decodes in 2 s per radar"}
        ),
        "video": str(args.output.resolve()) if products.write_video else None,
        "video_written": products.write_video,
        "compact_products": str(args.product_dir.resolve()),
        "full_cube_storage_policy": (
            "official (2,64,4,4,256) complex RadProPoser tensors are retained"
            if args.save_radproposer_products else
            "full 64x256x64 complex cubes are processed per frame but not retained; "
            "angle-mean 64x256 range-Doppler products are saved to avoid ~10 GB output"
        ),
        "radproposer_input_storage": {
            "saved": args.save_radproposer_products,
            "products": (
                "measured_radproposer_complex.npy and "
                "rendered_radproposer_complex.npy"
                if args.save_radproposer_products else None
            ),
            "per_frame_shape": [2, *RADPROPOSER_SINGLE_RADAR_SHAPE],
            "dtype": "complex64",
            "axis_contract": (
                "radar, Doppler, padded-TX, RX, shifted-range"
            ),
            "preprocessing": (
                "slow-time clutter mean removal followed by forward-normalized "
                "and fftshifted range/RX/padded-TX/Doppler transforms; exactly "
                "matches the official RadProPoser HuPR process_full_radar.py"
            ),
        },
        "raw_rd_storage": {
            "saved": args.save_raw_rd_products,
            "products": (
                "*_rd_raw.npy; angle-mean Doppler magnitude before slow-time "
                "mean subtraction"
                if args.save_raw_rd_products else None
            ),
            "purpose": (
                "preserve the physical zero-Doppler ridge for leakage/noise "
                "mechanism diagnostics"
                if args.save_raw_rd_products else None
            ),
        },
        "first_loop_ra_storage": {
            "saved": args.save_first_loop_ra_products,
            "products": (
"*_first_loop_ra_complex.npy; complex64 range×local-angle; "
                "*_first_loop_rp_complex.npy; complex64 tx×rx×range"
                if args.save_first_loop_ra_products else None
            ),
            "purpose": (
                "same-domain cross-sequence receiver temporal diagnostics"
                if args.save_first_loop_ra_products else None
            ),
        },
        "pose_motion_dra_storage": {
            "saved": args.save_pose_motion_dra,
            "frames": len(products.dra_frame_to_index),
            "doppler_bins": int(official_motion_doppler_mask().sum())
            if args.save_pose_motion_dra else 0,
            "products": "*_motion_dra.npy plus pose_motion_dra_frames.npy"
            if args.save_pose_motion_dra else None,
            "complex_products": (
                "*_motion_dra_complex.npy; complex64 Doppler×range×angle"
                if args.save_pose_motion_dra_complex else None
            ),
        },
    }


def _render_frames(
    args,
    calibration,
    configs,
    device,
    diffuse_floor,
    frame_receiver,
    frames,
    local_additive,
    measured,
    metrics,
    model_label,
    motion,
    person,
    products,
    receiver_noise,
    rendered_background,
    rig,
    scene,
    skinning_label,
    slow_time,
    velocity,
    receiver_floor=None,
) -> int:
    """Every frame, rendered and written; the ffmpeg return code.

    run()'s frame loop, verbatim, as P1c of docs/CONSOLIDATION_AND_OO_PLAN.md.
    The 21 parameters are its measured interface: sixteen components and five
    plain values. It hands back the one thing the teardown reads.
    """
    _smpl_faces = smpl_faces() if args.person_solid_angle_weight == "physical" else None   # Option B: the posed mesh's faces
    from mm4DPS.rasterizer import render_factorized
    from mm4DPS.rasterizer import PointPrimitives, cull_gaussians, render_gaussians
    from mm4DPS.rasterizer import reparameterize_torch
    return_code = 0
    try:
        with contextlib.ExitStack() as stack:
            frames_archive = stack.enter_context(
                zipfile.ZipFile(args.dataset_root / "frames.zip")
            ) if products.write_video else None
            sources = measured.open(stack, args.sequence)
            stack.enter_context(torch.no_grad())
            # One chunk's deferred host work: five correlations per radar per
            # frame -- strict, angle-integrated, their linear counterparts and
            # the fixed-track one -- and the records that attribute them back
            # to a frame when the chunk is resolved.
            pending = PendingScalars(products.chunk_frames * 2 * 5)
            chunk: list[dict] = []
            _t_physics = _t_compose = 0.0
            _t_mark = time.perf_counter()
            for frame in range(args.frame_count):
                if receiver_noise.jitter is not None or scene.visibility is not None:
                    if receiver_noise.jitter is not None:
                        scale = (
                            args.static_jitter_motion_scale
                            if frame in frames.reliable else 1.0
                        )
                        scene.primitives.positions.copy_(torch.from_numpy(
                            receiver_noise.jitter.step(displacement_scale=scale)
                        ).to(device))
                    frame_background_profiles = {}
                    for background_name, background_rasterizer in scene.rasterizers.items():
                        mask = rendered_background.masks[background_name]
                        # Per-frame occlusion of the scene by the person: a
                        # gate on the background's own points, indexed out of
                        # a resident (frames, N) bool. Contents vary per
                        # frame, shape does not, and nothing crosses to the
                        # host inside the loop.
                        shadow = None if scene.visibility is None else (
                            scene.visibility[frame, :, None].expand(
                                -1, background_rasterizer.tx_positions.shape[0])
                        )
                        background_real, background_imag = render_gaussians(
                            scene.primitives, background_rasterizer,
                            scene.area_weights * mask.float(),
                            active_mask=mask, shadow_mask=shadow,
                            detach_phase=True,
                            bsdf_mode="full", disabled_components=None,
                            use_cuda_kernels=False,
                        )
                        frame_background_profiles[background_name] = torch.complex(
                            background_real, background_imag,
                        )
                else:
                    frame_background_profiles = rendered_background.profiles
                if diffuse_floor.enabled:
                    with_floor = {}
                    for background_name, background_profile in frame_background_profiles.items():
                        reference = unwindowed_range_reference(
                            background_profile, args.diffuse_floor_reference_quantile,
                        )
                        # A per-range-bin process: drawn unwindowed, then
                        # seen through the range window (the scene's PSF and
                        # the measured ADC carry it already).
                        floor = range_window_profiles(on_device(
                            diffuse_floor.models[background_name].step(
                                diffuse_floor.range_amplitudes[background_name],
                            ), device,
                        ))
                        with_floor[background_name] = (
                            background_profile
                            + reference * floor
                        )
                    frame_background_profiles = with_floor
                cached = np.load(args.sequence_dir / "person_frames" / f"frame_{frame:03d}.npz")
                person_pcl = cached["pcl"]
                person_global_index = (
                    np.nonzero(cached["visible"])[0].astype(np.int64)
                    if "visible" in cached
                    else np.arange(len(person_pcl), dtype=np.int64)
                )
                if "visible" in cached:
                    # Per-frame ray-cast visibility over the fixed global
                    # point set: render only what the radar can see.
                    person_pcl = person_pcl[cached["visible"]]
                person.current_index["index"] = torch.from_numpy(
                    person_global_index).to(device)
                pcl_np = transform_person_pcl(
                    person_pcl, rig.depth_scale, rig.translation_y_m,
                )
                if frame in velocity.learned_frame_normals:
                    if len(velocity.learned_frame_normals[frame]) != len(pcl_np):
                        raise ValueError(
                            f"learned normals for frame {frame} have "
                            f"{len(velocity.learned_frame_normals[frame])} points, "
                            f"cloud has {len(pcl_np)}"
                        )
                    pcl_np[:, 3:6] = velocity.learned_frame_normals[frame]
                nearest = None if motion.surface_motion else nearest_joint_labels(
                    pcl_np[:, :3], person.joints[frame]
                )
                geometry = torch.from_numpy(pcl_np).to(device)
                weights = None
                if motion.skinning_mode in ("soft", "soft local-surface velocity"):
                    # the skinning weights on the device, as the trainer computes them
                    weights = soft_joint_weights_torch(
                        geometry[:, :3],
                        torch.as_tensor(person.joints[frame], dtype=geometry.dtype, device=device),
                        motion.skinning_temperature_m, motion.skinning_topk)
                # The measured branch crosses to the device once per frame per
                # radar, as 0.79 MB of raw int16 words; the DCA1000 decode and
                # the beamforming both happen there. Uploading the decoded
                # frame would be 2x the bytes and the RA cube 11x.
                raw_adc = {
                    radar.name: measured.frame_words(sources[radar.name], radar)
                    for radar in measured.dataset.radars
                }
                measured_adc = {
                    name: measured.decode(words, name, device)
                    for name, words in raw_adc.items()
                }
                frame_slots: dict[str, dict[str, int]] = {}
                for name, rasterizer in scene.rasterizers.items():
                    if motion.surface_motion:
                        if weights is None:
                            raise RuntimeError("surface motion requires soft skinning weights")
                        weights_tensor = weights
                        point_gains = weights_tensor @ motion.joint_gains
                        # The Monte-Carlo weight for this radar. Computed the
                        # same way as the trainer's, and combined with the
                        # other factors by the same shared function, so the
                        # materials are evaluated in the forward model they
                        # were fitted in. See hupr_person_forward_model.
                        if args.person_solid_angle_weight == "physical":
                            # Option B: the posed FULL mesh's vertex areas at the rendered vertices, |cos|/d^2 from the
                            # mesh normals, in the trainer's scene unit (the checkpoint's)
                            solid_angle = person_solid_angle_weights(
                                transform_person_pcl(cached["pcl"], rig.depth_scale, rig.translation_y_m), _smpl_faces,
                                person_global_index, config_centre_m(configs[name]).astype(np.float64),
                                float(motion.checkpoint["solid_angle_unit"][name]), device).to(point_gains.dtype)
                        elif args.person_solid_angle_weight == "on":
                            solid_angle = surface_solid_angle_weights_torch(
                                geometry[:, :3], geometry[:, 3:6],
                                config_centre_m(configs[name]).astype(np.float64),
                            ).to(point_gains.dtype)
                        else:
                            solid_angle = torch.ones_like(point_gains)
                        components = render_factorized(
                            geometry[:, :3], geometry[:, 3:6],
                            person_areas(
                                person.material.areas_for_points(len(geometry)),
                                point_gains, solid_angle,
                                person.material.radar_gain(name)),
                            person.materials_for(len(geometry)), rasterizer,
                            reparameterize_torch, detach_phase=True,
                            shadow_mask=None, bsdf_mode="full", disabled_components=None,
                            use_cuda_kernels=True, return_splat_components=True,
                        )
                        source_velocity = torch.from_numpy(interpolate_frame_values(
                            velocity.velocities, frame, motion.velocity_frame_offset,
                        )).to(device)
                        point_velocity = soft_surface_velocities(
                            weights_tensor, source_velocity, motion.joint_scales,
                            global_velocity_transform=motion.velocity_rotation,
                        )
                        if frame in velocity.point_normal_scalars:
                            scalars = torch.from_numpy(
                                velocity.point_normal_scalars[frame]
                            ).to(device)
                            if len(scalars) != len(geometry):
                                raise ValueError(
                                    f"point scalars for frame {frame} do not "
                                    "match the cloud"
                                )
                            point_velocity = point_velocity + (
                                scalars[:, None] * geometry[:, 3:6]
                            )
                        path_rate = surface_bistatic_path_rates(
                            geometry[:, :3], point_velocity,
                            rasterizer.tx_positions, rasterizer.rx_positions,
                        )
                        profiles = motion.slow_time_assemble(
                            frame_background_profiles[name], components, path_rate,
                            loop_chunk_size=args.loop_chunk_size,
                            gate_label=name,
                        )
                    else:
                        group_profiles = []
                        for joint in range(14):
                            if weights is None:
                                selected = torch.from_numpy(nearest == joint).to(device)
                                point_weights = None
                            else:
                                selected = weights[:, joint] > 0
                                point_weights = weights[selected, joint]
                            count = int(selected.sum())
                            if count == 0:
                                group_profiles.append(torch.zeros_like(frame_background_profiles[name]))
                                continue
                            real, imag = render_factorized(
                                geometry[selected, :3], geometry[selected, 3:6],
                                (person.material.areas_for_points(count) if point_weights is None else
                                 person.material.areas_for_points(count) * point_weights)
                                * person.material.radar_gain(name)
                                * person_area_weight_scale,
                                person.materials_for(count),
                                rasterizer, reparameterize_torch, detach_phase=True,
                                shadow_mask=None, bsdf_mode="full", disabled_components=None,
                                use_cuda_kernels=True,
                            )
                            group_profiles.append(torch.complex(real, imag))
                        center = config_centre_m(configs[name]).astype(np.float32)
                        velocity_frame = offset_frame_index(
                            frame, int(motion.velocity_frame_offset), len(velocity.velocities),
                        )
                        joint_radial = joint_radial_velocities(
                            person.joints[frame], velocity.velocities[velocity_frame], center,
                        )
                        radial = joint_radial * motion.joint_scales.cpu().numpy()
                        path_rate = joint_bistatic_path_rates(
                            person.joints[frame], velocity.velocities[velocity_frame],
                            rasterizer.tx_positions.detach().cpu().numpy(),
                            rasterizer.rx_positions.detach().cpu().numpy(),
                        ) * motion.joint_scales.cpu().numpy()[:, None, None]
                        stacked_groups = (
                            torch.stack(group_profiles) * motion.joint_gains[:, None, None, None]
                        )
                        if rig.phase_model == "bistatic":
                            profiles = assemble_bistatic_slow_time_profiles(
                                frame_background_profiles[name], stacked_groups,
                                torch.from_numpy(path_rate).to(device), 1.0,
                            )
                        else:
                            profiles = assemble_slow_time_profiles(
                                frame_background_profiles[name], stacked_groups,
                                torch.from_numpy(radial).to(device), 1.0,
                            )
                    if name in receiver_noise.models:
                        gains = torch.from_numpy(
                            receiver_noise.models[name].step(profiles.shape[0])
                        ).to(device)
                        profiles = profiles * gains[:, None, None, None]
                    if name in frame_receiver.models:
                        if (
                            args.frame_receiver_nuisance_mode
                            == "common-range-gain"
                        ):
                            range_gain = (
                                frame_receiver.models[name].step()
                            )
                            range_gain[~RANGE_KEEP] = 1.0
                            profiles = profiles * torch.from_numpy(
                                range_gain
                            ).to(device)[None, None, None, :]
                        else:
                            reference = unwindowed_range_reference(
                                frame_background_profiles[name],
                                args.diffuse_floor_reference_quantile,
                            ).clamp_min(1e-20)
                            frame_nuisance = range_window_profiles(on_device(
                                frame_receiver.models[name].step(
                                    frame_receiver.scales[name]
                                    * RANGE_KEEP.astype(np.float64)
                                ), device,
                            ))
                            if (
                                args.frame_receiver_nuisance_mode
                                == "amplitude-aligned"
                            ):
                                mean_profile = profiles.mean(dim=0)
                                phase_unit = (
                                    mean_profile
                                    / mean_profile.abs().clamp_min(1e-20)
                                )
                                signed_amplitude = (
                                    frame_nuisance * phase_unit.conj()
                                ).real
                                frame_nuisance = signed_amplitude * phase_unit
                            profiles = (
                                profiles + reference * frame_nuisance[None]
                            )
                    if name in local_additive.models:
                        reference = unwindowed_range_reference(
                            frame_background_profiles[name],
                            args.diffuse_floor_reference_quantile,
                        ).clamp_min(1e-20)
                        local_nuisance = range_window_profiles(on_device(
                            local_additive.models[name].step(
                                local_additive.scales[name]
                                * RANGE_KEEP.astype(np.float64)
                            ), device,
                        ))
                        profiles = (
                            profiles + reference * local_nuisance[None]
                        )
                    if slow_time.models:
                        nuisance_scale = per_radar(args, "slow-time-nuisance", "scale")[name]
                        if nuisance_scale is None:
                            nuisance_scale = args.slow_time_nuisance_scale
                        background_magnitude = frame_background_profiles[name].abs()
                        reference = unwindowed_range_reference(
                            frame_background_profiles[name],
                            args.diffuse_floor_reference_quantile,
                        ).clamp_min(1e-20)
                        spectral_amplitudes = None
                        radar_calibration = (
                            None if slow_time.calibration is None else
                            slow_time.calibration["radars"][name]
                        )
                        if args.slow_time_nuisance_mode == "thermal":
                            range_amplitudes = (
                                nuisance_scale
                                * RANGE_KEEP.astype(np.float64)
                            )
                        elif args.slow_time_nuisance_mode == "phase-noise":
                            knee_bins = per_radar(args, "slow-time-nuisance", "knee-bins")[name]
                            white_fraction = per_radar(args, "slow-time-nuisance", "white-power-fraction")[name]
                            doppler_envelope = phase_noise_spectral_envelope(
                                profiles.shape[0], knee_bins, white_fraction,
                            )
                            spectral_amplitudes = (
                                nuisance_scale
                                * doppler_envelope[:, None]
                                * RANGE_KEEP.astype(np.float64)[None, :]
                            )
                            range_amplitudes = None
                        elif (
                            radar_calibration is not None
                            and "normalized_doppler_range_profile" in radar_calibration
                        ):
                            spectral_envelope = np.asarray(
                                radar_calibration[
                                    "normalized_doppler_range_profile"
                                ],
                                dtype=np.float64,
                            )
                            if spectral_envelope.shape != (64, 256):
                                raise ValueError(
                                    "slow-time nuisance Doppler-range calibration "
                                    "profile must have shape (64,256)"
                                )
                            scene_fraction = per_radar(args, "slow-time-nuisance", "scene-fraction")[name]
                            if scene_fraction > 0:
                                scene_range_profile = torch.quantile(
                                    background_magnitude.reshape(
                                        -1, background_magnitude.shape[-1]
                                    ),
                                    0.75, dim=0,
                                ).cpu().numpy()
                                if args.slow_time_nuisance_scene_density_normalize:
                                    scene_range_profile = (
                                        density_normalized_scene_profile(
                                            scene_range_profile,
                                            scene.range_point_counts,
                                        )
                                    )
                                spectral_envelope = remix_slow_time_range_shape(
                                    spectral_envelope,
                                    scene_range_profile,
                                    scene_fraction,
                                    args.slow_time_nuisance_scene_smoothing_bins,
                                )
                            if args.slow_time_nuisance_edge_blend_bins > 1:
                                spectral_envelope = blend_envelope_range_edges(
                                    spectral_envelope, RANGE_KEEP,
                                    args.slow_time_nuisance_edge_blend_bins,
                                )
                            spectral_amplitudes = (
                                nuisance_scale * spectral_envelope
                            )
                            spectral_amplitudes[:, ~RANGE_KEEP] = 0.0
                            range_amplitudes = None
                        elif slow_time.calibration is None:
                            range_envelope = torch.quantile(
                                background_magnitude.reshape(
                                    -1, background_magnitude.shape[-1]
                                ),
                                0.75, dim=0,
                            ) / reference
                            range_amplitudes = nuisance_scale * np.maximum(
                                range_envelope.cpu().numpy(),
                                args.slow_time_nuisance_min_amplitude,
                            )
                        else:
                            range_envelope = np.asarray(
                                radar_calibration[
                                    "normalized_range_profile"
                                ],
                                dtype=np.float64,
                            )
                            if range_envelope.shape != (256,):
                                raise ValueError(
                                    "slow-time nuisance calibration profile must have "
                                    "256 range bins"
                                )
                            range_amplitudes = (
                                nuisance_scale * range_envelope
                            )
                        if spectral_amplitudes is None:
                            assert range_amplitudes is not None
                            range_amplitudes[~RANGE_KEEP] = 0.0
                            nuisance_array = (
                                slow_time.models[name].slow_time_cube(
                                    range_amplitudes, loops=profiles.shape[0],
                                )
                            )
                        else:
                            fixed_fraction = per_radar(args, "slow-time-nuisance", "fixed-fraction")[name]
                            empty_correlation = per_radar(args, "slow-time-nuisance", "empty-correlation")[name]
                            correlation_map = None
                            if empty_correlation is not None:
                                correlation_map = np.full(
                                    spectral_amplitudes.shape,
                                    slow_time.models[
                                        name
                                    ].temporal_correlation,
                                    dtype=np.float64,
                                )
                                range_m = (
                                    np.arange(
                                        spectral_amplitudes.shape[1],
                                        dtype=np.float64,
                                    )
                                    * range_bin_spacing_m()
                                )
                                empty_range = (
                                    (range_m >= 9.4) & (range_m <= 10.8)
                                )
                                doppler = (
                                    np.arange(
                                        spectral_amplitudes.shape[0]
                                    )
                                    - spectral_amplitudes.shape[0] // 2
                                )
                                far_doppler = np.abs(doppler) >= 20
                                correlation_map[np.ix_(
                                    np.flatnonzero(far_doppler),
                                    np.flatnonzero(empty_range),
                                )] = empty_correlation
                            nuisance_array = slow_time.models[
                                name
                            ].slow_time_cube_from_spectrum(
                                spectral_amplitudes,
                                frame_fixed_fraction=fixed_fraction,
                                temporal_correlation_by_doppler_range=(
                                    correlation_map
                                ),
                            )
                        nuisance = range_window_profiles(on_device(nuisance_array, device))
                        profiles = profiles + reference * nuisance
                    # Frozen hardware receiver gain. The renderer reproduces the
                    # scene and nuisance up to one unknown constant per radar
                    # (ADC/amplifier scaling), which no normalized metric can
                    # observe. Applying it here, after every nuisance term and
                    # before any product is derived, keeps RA, RD and the
                    # RadProPoser tensor on one consistent absolute scale.
                    if name in calibration.virtual_channel_gains:
                        # Frozen per-virtual-channel calibration residual. Real
                        # hardware has one; it produces the fixed angular
                        # pattern that repeats at every range bin, which the
                        # measured data shows and an uncalibrated render lacks.
                        profiles = profiles * calibration.virtual_channel_gains[name][
                            None, :, :, None
                        ]
                    if calibration.absolute_gains[name] != 1.0:
                        profiles = profiles * calibration.absolute_gains[name]
                    if calibration.antenna_coupling[name] is not None:
                        # The board's static near-field coupling. Added after
                        # the absolute gain because it is measured in the same
                        # units as the products, not in render units.
                        #
                        # It is static, so clutter removal -- the slow-time
                        # mean subtraction -- annihilates it exactly, and the
                        # RD products are unchanged by construction. It shows
                        # up only in the range-angle products, which is where
                        # the measurement carries it: 26-28% of all measured
                        # RA range energy sits in these bins and the render
                        # had nothing there at all.
                        profiles = profiles + calibration.antenna_coupling[name]
                    if receiver_floor is not None and name in receiver_floor.amplitude:
                        # The trained receiver floor, in product units.
                        profiles = profiles + receiver_floor.draw(name)
                    if args.save_radproposer_products:
                        products.writer.append(
                            "rendered_radproposer_complex",
                            range_profiles_to_radproposer_tensor(profiles),
                        )
                        products.writer.append(
                            "measured_radproposer_complex",
                            adc_frame_to_radproposer_tensor(measured_adc[name]),
                        )
                    center = config_centre_m(configs[name]).astype(np.float32)
                    rendered_ra = slow_time_profiles_to_ra_complex(profiles.real, profiles.imag)
                    rendered_doppler_complex = slow_time_ra_to_doppler_complex(
                        rendered_ra
                    )
                    rendered_doppler = rendered_doppler_complex.abs()
                    measured_ra = range_angle_slow_time_cube_torch(
                        measured_adc[name]
                    )
                    measured_doppler_complex = slow_time_ra_to_doppler_complex(
                        measured_ra
                    )
                    measured_doppler = measured_doppler_complex.abs()
                    # THE PRODUCT, both sides: the per-channel, per-loop
                    # range profiles -- range-windowed and range-FFT'd, no
                    # slow-time transform, no magnitude, no angle FFT, no
                    # channel dropped. `profiles` is exactly what rendered_ra
                    # above was built from; the consumer takes the Doppler
                    # transform (`data.product_to_doppler`).
                    products.stage_cube(f"{name}_rendered", frame, slow_time_product(profiles))
                    if args.save_measured_cube == "on":
                        products.stage_cube(f"{name}_measured", frame,
                                   slow_time_product(range_profiles_all_channels_torch(
                                       measured_adc[name], device=device)))
                    if args.save_first_loop_ra_products:
                        products.writer.append(
                            f"{name}_rendered_first_loop_ra_complex",
                            rendered_ra[0],
                        )
                        products.writer.append(
                            f"{name}_measured_first_loop_ra_complex",
                            measured_ra[0],
                        )
                        with array_window("none"):
                            products.writer.append(
                                f"{name}_rendered_first_loop_ra_untapered_complex",
                                slow_time_profiles_to_ra_complex(profiles[:1].real, profiles[:1].imag)[0],
                            )
                            products.writer.append(
                                f"{name}_measured_first_loop_ra_untapered_complex",
                                range_angle_slow_time_cube_torch(measured_adc[name])[0],
                            )
                        products.writer.append(f"{name}_rendered_first_loop_rp_complex", profiles[0])
                        products.writer.append(
                            f"{name}_measured_first_loop_rp_complex",
                            range_profiles_all_channels_torch(measured_adc[name], device=device)[0])
                    if args.save_raw_rd_products:
                        rendered_doppler_raw = slow_time_ra_to_doppler_magnitude(
                            rendered_ra, remove_static_clutter=False,
                        )
                        measured_doppler_raw = slow_time_ra_to_doppler_magnitude(
                            measured_ra, remove_static_clutter=False,
                        )
                        rendered_doppler_raw[:, ~calibration.keep_range, :] = 0.0
                        measured_doppler_raw[:, ~calibration.keep_range, :] = 0.0
                        products.writer.append(f"{name}_rendered_rd_raw",
                                      rendered_doppler_raw.mean(dim=2))
                        products.writer.append(f"{name}_measured_rd_raw",
                                      measured_doppler_raw.mean(dim=2))
                    rendered_doppler[:, ~calibration.keep_range, :] = 0.0
                    measured_doppler[:, ~calibration.keep_range, :] = 0.0
                    ra_mask = person_range_angle_mask_torch(
                        geometry[:, :3], center, array_axis(configs[name]),
                        range_bin_spacing_m(), include_mirrored_angle=True,
                    )
                    range_rows = ra_mask.any(dim=1)
                    angle_count = ra_mask.sum(dim=1).clamp_min(1)
                    metric_rendered = rendered_doppler
                    metric_measured = measured_doppler
                    if motion.noise_floor_quantile is not None:
                        metric_rendered = subtract_spatial_noise_floor(
                            rendered_doppler, ra_mask, float(motion.noise_floor_quantile),
                        )
                        metric_measured = subtract_spatial_noise_floor(
                            measured_doppler, ra_mask, float(motion.noise_floor_quantile),
                        )
                    # Each of these was a `float(...)` on a 0-dim device
                    # tensor, i.e. a synchronization, for a number nothing
                    # reads until the chunk ends. Staged now, resolved in the
                    # flush below with the identical Python arithmetic.
                    slots = {"strict": pending.add(log_pearson_loss(
                        metric_rendered[calibration.motion_doppler][:, ra_mask],
                        metric_measured[calibration.motion_doppler][:, ra_mask],
                    ))}
                    rendered_roi_rd = (
                        (metric_rendered * ra_mask[None]).sum(dim=2) / angle_count[None]
                    )[calibration.motion_doppler][:, range_rows]
                    measured_roi_rd = (
                        (metric_measured * ra_mask[None]).sum(dim=2) / angle_count[None]
                    )[calibration.motion_doppler][:, range_rows]
                    slots["rd"] = pending.add(
                        log_pearson_loss(rendered_roi_rd, measured_roi_rd)
                    )
                    slots["strict_linear"] = pending.add(linear_pearson_loss(
                        metric_rendered[calibration.motion_doppler][:, ra_mask],
                        metric_measured[calibration.motion_doppler][:, ra_mask],
                    ))
                    slots["rd_linear"] = pending.add(
                        linear_pearson_loss(rendered_roi_rd, measured_roi_rd)
                    )
                    frame_slots[name] = slots
                    if frame in products.dra_frame_to_index:
                        # The writer appends, so the row a frame lands on is
                        # the count of DRA frames before it -- which is what
                        # dra_frame_to_index holds, the map being built from
                        # the same sorted list.
                        products.writer.append(f"{name}_rendered_motion_dra",
                                      rendered_doppler[calibration.motion_doppler])
                        products.writer.append(f"{name}_measured_motion_dra",
                                      measured_doppler[calibration.motion_doppler])
                        if args.save_pose_motion_dra_complex:
                            products.writer.append(
                                f"{name}_rendered_motion_dra_complex",
                                rendered_doppler_complex[calibration.motion_doppler])
                            products.writer.append(
                                f"{name}_measured_motion_dra_complex",
                                measured_doppler_complex[calibration.motion_doppler])
                    track = frames.track_records.get((frame, name))
                    if frame in frames.reliable and track is not None:
                        center_bin = int(round(track[frames.fixed_range_key] / range_bin_spacing_m()))
                        lo = max(0, center_bin - 4)
                        hi = min(256, center_bin + 5)
                        slots["fixed"] = pending.add(log_pearson_loss(
                            rendered_doppler.mean(dim=2)[calibration.motion_doppler, lo:hi],
                            measured_doppler.mean(dim=2)[calibration.motion_doppler, lo:hi],
                        ))
                    products.writer.append(f"{name}_rendered_rd",
                                  rendered_doppler.mean(dim=2))
                    products.writer.append(f"{name}_measured_rd",
                                  measured_doppler.mean(dim=2))
                chunk.append({
                    "frame": frame, "slots": frame_slots,
                    # Only the video reads the person npz again; without it,
                    # holding a chunk's worth of open handles buys nothing.
                    "cached": cached if products.write_video else None,
                })
                # Where the wall clock actually goes. The GPU sits at 11% during
                # a 600-frame render, so the cost is not the physics; splitting
                # the loop here says how much of it is the chunk's transfers,
                # product writes and panel composition.
                _t_physics += time.perf_counter() - _t_mark
                _t_mark = time.perf_counter()
                if (len(chunk) == products.chunk_frames
                        or frame == args.frame_count - 1):
                    # One synchronize for the chunk, then the rows go to disk
                    # on the writer's threads. `views` is what just crossed,
                    # so the run-summary metrics and the video read the
                    # products rather than transferring them a second time.
                    views = products.writer.flush()
                    resolved = pending.values()
                    for position, record in enumerate(chunk):
                        chunk_frame = record["frame"]
                        display = {}
                        for name in current_dataset().names:
                            slots = record["slots"][name]
                            strict_corr = 1.0 - float(resolved[slots["strict"]])
                            rd_corr = 1.0 - float(resolved[slots["rd"]])
                            corr = 0.25 * strict_corr + 0.75 * rd_corr
                            strict_corr_linear = 1.0 - float(
                                resolved[slots["strict_linear"]]
                            )
                            rd_corr_linear = 1.0 - float(
                                resolved[slots["rd_linear"]]
                            )
                            if chunk_frame in frames.reliable:
                                metrics.correlations[name]["combined"].append(corr)
                                metrics.correlations[name]["strict_dra"].append(strict_corr)
                                metrics.correlations[name]["angle_integrated_rd"].append(rd_corr)
                                metrics.correlations[name]["strict_dra_linear"].append(strict_corr_linear)
                                metrics.correlations[name]["angle_integrated_rd_linear"].append(rd_corr_linear)
                            fixed_corr = (
                                1.0 - float(resolved[slots["fixed"]])
                                if "fixed" in slots else None
                            )
                            metrics.fixed_track[name].append(fixed_corr)
                            display[name] = (
                                views[f"{name}_measured_rd"][position],
                                views[f"{name}_rendered_rd"][position],
                                fixed_corr,
                            )
                        # The background shape error is a run-summary metric,
                        # not a video annotation; it is computed whether or
                        # not the diagnostic video is written.
                        background_errors = {
                            name: background_profile_shape_error(
                                display[name][1], display[name][0],
                            )
                            for name in current_dataset().names
                        }
                        for name in current_dataset().names:
                            metrics.background_profile_errors[name].append(
                                background_errors[name]
                            )
                        if products.write_video:
                            if set(display) != {"horizontal", "vertical"}:
                                raise ValueError(
                                    "the diagnostic video composes a horizontal and a "
                                    "vertical panel; pass --write-video off for "
                                    f"{sorted(display)}")
                            payload = frames_archive.read(
                                f"frames/single_{args.sequence}/processed/images/{chunk_frame:09d}.jpg"
                            )
                            rgb = Image.open(io.BytesIO(payload)).convert("RGB")
                            canvas = compose(
                                rgb, record["cached"],
                                *display["horizontal"][:2], *display["vertical"][:2],
                                chunk_frame, display["horizontal"][2], display["vertical"][2],
                                model_label, skinning_label,
                                background_errors["horizontal"], background_errors["vertical"],
                                args.frame_count,
                            )
                            if chunk_frame == args.poster_frame:
                                canvas.save(args.output.with_name(args.output.stem + "_poster.png"))
                            assert products.ffmpeg is not None and products.ffmpeg.stdin is not None
                            products.ffmpeg.stdin.write(np.asarray(canvas, dtype=np.uint8).tobytes())
                    pending.reset()
                    chunk.clear()
                _t_compose += time.perf_counter() - _t_mark
                _t_mark = time.perf_counter()
                if frame % 25 == 0 or frame == args.frame_count - 1:
                    total = _t_physics + _t_compose
                    print(
                        f"all-loop rendered+encoded {frame + 1}/{args.frame_count}"
                        f"  [physics {_t_physics:.0f}s {_t_physics / max(total, 1e-9):.0%}"
                        f"  compose+encode {_t_compose:.0f}s "
                        f"{_t_compose / max(total, 1e-9):.0%}]",
                        flush=True,
                    )
            # Inside the loop's `with`, so a render that raises never reaches
            # it: close() is what verifies every product is complete, and a
            # short product must be an error, not a file of zeros.
            products.writer.close()
            for key, scales in products.scales.items():
                np.save(args.product_dir / f"{key}_scale.npy", scales.cpu().numpy())
    finally:
        # Idempotent, and never raises: an interrupted render must release its
        # writer threads and descriptors without masking what interrupted it.
        products.writer.shutdown()
        if products.ffmpeg is not None:
            if products.ffmpeg.stdin is not None:
                products.ffmpeg.stdin.close()
            return_code = products.ffmpeg.wait()
    return return_code


def run(args: argparse.Namespace) -> dict:
    Dataset.of(args)  # selects it for this process before anything reads it
    if not torch.cuda.is_available():
        raise RuntimeError("all-loop rendering requires CUDA")
    if not 1 <= args.frame_count <= 600:
        raise ValueError("frame_count must lie in [1,600]")
    if args.save_pose_motion_dra_complex and not args.save_pose_motion_dra:
        raise ValueError(
            "--save-pose-motion-dra-complex requires --save-pose-motion-dra"
        )
    for value in per_radar(args, "absolute-gain").values():
        if not np.isfinite(value) or value <= 0:
            raise ValueError("absolute receiver gains must be finite and positive")
    if args.slow_time_nuisance_scale < 0 or args.slow_time_nuisance_min_amplitude < 0:
        raise ValueError("slow-time nuisance amplitudes must be nonnegative")
    if args.slow_time_nuisance_scatterers_per_bin <= 0:
        raise ValueError("slow-time nuisance scatterer count must be positive")
    for value in per_radar(args, "slow-time-nuisance", "scale").values():
        if value is not None and value < 0:
            raise ValueError("radar-specific slow-time nuisance scales must be nonnegative")
    for value in (*per_radar(args, "slow-time-nuisance", "fixed-fraction").values(),
                  *per_radar(args, "slow-time-nuisance", "white-power-fraction").values(),
                  *per_radar(args, "slow-time-nuisance", "scene-fraction").values()):
        if not 0.0 <= value <= 1.0:
            raise ValueError("slow-time nuisance fractions must lie in [0,1]")
    for value in per_radar(args, "slow-time-nuisance", "empty-correlation").values():
        if value is not None and not 0.0 <= value < 1.0:
            raise ValueError(
                "empty-floor temporal correlations must lie in [0,1)"
            )
    for value in per_radar(args, "slow-time-nuisance", "knee-bins").values():
        if not np.isfinite(value) or value <= 0:
            raise ValueError("phase-noise knee bins must be finite and positive")
    for value in per_radar(args, "local-additive-nuisance", "scale").values():
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                "background/local nuisance scales must be finite and nonnegative"
            )
    for value in per_radar(args, "local-additive-nuisance", "correlation").values():
        if not 0.0 <= value < 1.0:
            raise ValueError(
                "local additive nuisance correlations must lie in [0,1)"
            )
    if args.local_additive_nuisance_scatterers_per_bin <= 0:
        raise ValueError(
            "local additive nuisance scatterer count must be positive"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.product_dir.mkdir(parents=True, exist_ok=True)
    from mm4DPS.rasterizer import reparameterize_torch
    from mm4DPS.rasterizer import render_factorized
    from mm4DPS.rasterizer import PointPrimitives, cull_gaussians, render_gaussians

    device = torch.device(args.device)
    checkpoint = torch.load(args.person_checkpoint, map_location=device, weights_only=False)
    motion = _configure_motion_checkpoint(args, device)
    rig = _configure_scene_geometry(args, checkpoint, motion)
    if args.person_solid_angle_weight == "physical" and not (motion.checkpoint.get("solid_angle_unit")):
        raise ValueError("--person-solid-angle-weight physical needs a velocity checkpoint trained with it (solid_angle_unit)")
    if args.model_label is not None:
        model_label = args.model_label
    elif rig.translation_y_m != 0.0:
        model_label = "v8 range calibrated"
    elif rig.phase_model == "bistatic":
        model_label = "v7 bistatic"
    elif motion.skinning_mode == "soft":
        model_label = "v4 soft skinning"
    else:
        model_label = "v3"
    if motion.surface_motion:
        skinning_label = (
            f"Soft top-{motion.skinning_topk} surface motion, bistatic phase"
        )
    elif motion.skinning_mode == "soft":
        skinning_label = f"Soft top-{motion.skinning_topk} skinning; {rig.phase_model} phase"
    else:
        skinning_label = "Hard nearest-joint assignment"
    configs = Dataset.of(args).configs(args)
    frames = _configure_motion_frames(args)
    scene = _configure_background(args, checkpoint, device)
    receiver_noise = _configure_receiver_noise(args, scene.primitives)
    rendered_background = _configure_background_profiles(receiver_noise, scene)
    frame_receiver = _configure_frame_receiver_nuisance(args, configs, device, scene.rasterizers)
    local_additive = _configure_local_additive_nuisance(args, configs, device, scene.rasterizers)
    diffuse_floor = _configure_diffuse_floor(args, configs, device, scene.rasterizers, scene.state)
    slow_time = _configure_slow_time_nuisance(args, configs, device, diffuse_floor, scene.rasterizers)
    receiver_floor = _configure_receiver_floor(args, motion, device)
    # A per-point trainer run writes an (N, 6) `shared_human_raw_material`:
    # one 3DPS material per point of the global set, not one shared vector.
    # `SharedHumanMaterial` asserts (6,), so without this the render of a
    # per-point run would fail outright -- and a silent fallback to the shared
    # vector would be worse, because it would render a scene the trainer never
    # optimized. Same failure mode the background hit before its state was
    # persisted.
    person = _configure_person(args, checkpoint, device, rig)
    velocity = _configure_velocity_state(args, person.joints, motion)

    # The annotated side-by-side video is a diagnostic, not a product: the
    # deliverable media are built from the saved .npy products by
    # build_velocity_comparison_media.py. Composing and encoding it is ~21%
    # of a render, so the media chain turns it off (WRITE_VIDEO=off); every
    # saved product and every run-summary metric is computed either way.
    products = _configure_products(args, device, frames)
    metrics = _configure_loop_metrics()
    # Static antenna coupling, calibrated from held-out sequences. The
    # coefficients are per virtual channel -- TX0's four receivers then TX2's,
    # the layout slow_time_profiles_to_ra_complex expects -- so they drop into
    # the profile rows those channels come from.
    calibration = _configure_calibrations(args, device)
    measured = _configure_measured(args)
    return_code = _render_frames(
        args=args,
        calibration=calibration,
        configs=configs,
        device=device,
        diffuse_floor=diffuse_floor,
        frame_receiver=frame_receiver,
        frames=frames,
        local_additive=local_additive,
        measured=measured,
        metrics=metrics,
        model_label=model_label,
        motion=motion,
        person=person,
        products=products,
        receiver_noise=receiver_noise,
        rendered_background=rendered_background,
        rig=rig,
        scene=scene,
        skinning_label=skinning_label,
        slow_time=slow_time,
        velocity=velocity,
     receiver_floor=receiver_floor)
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
    result = _result_payload(
        args,
        metrics=metrics,
        frames=frames,
        diffuse_floor=diffuse_floor,
        frame_receiver=frame_receiver,
        velocity=velocity,
        local_additive=local_additive,
        measured=measured,
        motion=motion,
        rig=rig,
        scene=scene,
        receiver_noise=receiver_noise,
        slow_time=slow_time,
        products=products,
    )
    # Whether the static near-field coupling was applied, and from where.
    #
    # Nothing recorded this until 2026-09-04, which is exactly why it could go
    # missing untraced: the driver applies the calibration only when the
    # directory happens to exist, so a run without it rendered a near field 46x
    # too weak (first-15-bin mean 405 against a measured 18,563) and said
    # nothing. A render that cannot tell you whether a forward-model term was
    # active cannot be audited against another render.
    result["antenna_coupling"] = {
        "applied": args.antenna_coupling_calibration is not None,
        "calibration": (str(args.antenna_coupling_calibration)
                        if args.antenna_coupling_calibration is not None else None),
        "scale": float(args.antenna_coupling_scale),
    }
    # Attach BEFORE the file is written; see velocity_field.py.
    result["peak_vram"] = peak_vram()
    (args.product_dir / "render_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def make_parser() -> argparse.ArgumentParser:
    dataset = dataset_from_argv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--sequence", type=int, default=1)
    sequence_dir(parser)
    parser.add_argument("--person-checkpoint", type=Path, default=WORKSPACE / "training/person_sequence_1_first_loop_shared_person_preserved/best_shared_human_material.pt")
    parser.add_argument("--motion-checkpoint", type=Path, default=WORKSPACE / "training/person_sequence_1_all_loops_motion_v3/chirp_motion.pt")
    radar_configs(parser, dataset)
    parser.add_argument("--mm3dps-root", type=Path,
                        default=Path(MM3DPS_ROOT))
    parser.add_argument("--product-dir", type=Path, default=WORKSPACE / "rendered/person_sequence_1_all_loops_v3")
    parser.add_argument("--save-measured-cube", choices=("on", "off"), default="off",
                        help="also write <radar>_measured.npy, the recording's own slow-time cube "
                             "(7.5 GB on the cascade, identical run to run); off: consumers decode "
                             "it from the archive (media.MeasuredCube)")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "deliverables/hupr_sequence_1_person_all_chirps_v3_simple.mp4")
    parser.add_argument("--poster-frame", type=int, default=120)
    parser.add_argument(
        "--write-video", choices=("on", "off"), default="on",
        help="whether to compose and encode the annotated diagnostic video "
             "(~21%% of a render). The saved .npy products and every "
             "run-summary metric are identical either way; the deliverable "
             "media come from build_velocity_comparison_media.py, not from "
             "this video.")
    parser.add_argument(
        "--product-chunk-frames", type=int, default=8,
        help="how many frames of products are staged before they are written "
             "(default 8). The rows cross to pinned host memory as they are "
             "produced and go to disk on a thread pool, so one chunk costs "
             "one device synchronization instead of ~20 per frame. Products "
             "are byte for byte identical at any value. "
             "PRODUCTION RUNS AT 1, and it is worth knowing why before "
             "tuning this: --write-video on forces 1, because the annotated "
             "video needs each frame's own numbers on the host, and the "
             "chain defaults VF_DIAG_VIDEO to on. So the 13.3 s the render "
             "half gained (ledger R061) came entirely from chunk 1 -- the "
             "pinned asynchronous copies, the single synchronization per "
             "frame and the threaded write -- and none of it from chunking. "
             "Raising this above 1 does anything only where the media chain "
             "turns the video off; R061 measured that case too. A sweep of "
             "chunk sizes is not where the remaining time is. Measured over "
             "the sequence-102 product set: 23.3 ms/frame of memmap row "
             "writes against 3.8 ms/frame here.")
    parser.add_argument("--frame-count", type=int, default=600)
    parser.add_argument("--model-label", help="short display label for an ablation model")
    parser.add_argument("--geometry-translation-y-m", type=float)
    parser.add_argument("--geometry-depth-scale", type=float)
    parser.add_argument(
        "--person-solid-angle-weight", choices=("on", "off", "physical"), default="off",
        help="weight each person point by its Monte-Carlo solid angle "
             "cos_o/d^2 (unit mean, so the global level is unchanged). The "
             "renderer's `areas` slot is documented as that weight; 3DPS may "
             "pass 1.0 because it cosine-resamples its cloud, this project's "
             "person cloud is not resampled, and the correct weight varies by "
             "6600x across it, so a constant is not a valid weight. 'off' "
             "restores the pre-fix behaviour and is the arbiter.")
    parser.add_argument("--scene-visibility", type=Path, default=None,
                        help="(frames, N) bool mask from "
                             "`pointset scene-shadow-mask`: per-frame "
                             "occlusion of the scene by the person. "
                             "Must be built against this checkpoint.")
    parser.add_argument("--static-jitter-sigma-um", type=float, default=0.0)
    parser.add_argument("--static-jitter-temporal-correlation", type=float, default=0.6)
    parser.add_argument("--static-jitter-common-variance-fraction", type=float, default=0.2)
    parser.add_argument("--static-jitter-seed", type=int, default=0)
    parser.add_argument("--static-jitter-motion-scale", type=float, default=1.0)
    for radar in dataset.radars:
        parser.add_argument(radar.flag("receiver-phase", "sigma-rad"), type=float, default=0.0)
    for radar in dataset.radars:
        parser.add_argument(radar.flag("receiver-phase", "drift-rad-per-frame"), type=float, default=0.0)
    for radar in dataset.radars:
        parser.add_argument(radar.flag("receiver-phase", "periodic-amplitude-rad"), type=float, default=0.0)
    for radar in dataset.radars:
        parser.add_argument(radar.flag("receiver-phase", "periodic-frequency-cycles-per-frame"), type=float, default=0.0)
    for radar in dataset.radars:
        parser.add_argument(radar.flag("receiver-amplitude", "sigma"), type=float, default=0.0)
    parser.add_argument(
        "--receiver-noise-frame-correlation", type=float, default=0.92,
    )
    for radar in dataset.radars:
        parser.add_argument(radar.flag("receiver-noise", "frame-correlation"), type=float)
    for radar in dataset.radars:
        parser.add_argument(radar.flag("receiver-chirp-phase", "sigma-rad"), type=float, default=0.0)
    parser.add_argument(
        "--receiver-noise-chirp-correlation", type=float, default=0.95,
    )
    parser.add_argument(
        "--receiver-noise-chirp-frame-correlation", type=float, default=0.0,
    )
    for radar in dataset.radars:
        parser.add_argument(radar.flag("receiver-noise-chirp-frame", "correlation"), type=float)
    parser.add_argument("--receiver-noise-seed", type=int, default=3100)
    parser.add_argument("--diffuse-floor-relative-amplitude", type=float, default=0.0)
    parser.add_argument("--diffuse-floor-calibration", type=Path)
    parser.add_argument("--receiver-floor-from-checkpoint", choices=("on", "off"), default="on",
                        help="draw the receiver floor the trainer learned (the motion checkpoint's "
                             "noise_floor_power) into the products; off renders that floor absent")
    parser.add_argument("--receiver-floor-seed", type=int, default=5100)
    parser.add_argument("--diffuse-floor-reference-quantile", type=float, default=0.999)
    parser.add_argument("--diffuse-floor-temporal-correlation", type=float, default=0.3)
    parser.add_argument("--diffuse-floor-scatterers-per-bin", type=int, default=4)
    parser.add_argument(
        "--antenna-coupling-calibration", type=Path, default=None,
        help="mm4DPS.calibration.geometry output. The board's static "
             "near-field return, which the render otherwise lacks entirely: "
             "cull_gaussians drops every scene point inside the 0.5 m guard, "
             "so rendered/measured over the first 15 range bins is 0.0007 "
             "against a measurement that carries 26-28%% of its range energy "
             "there. Calibrated from held-out sequences; cross-sequence "
             "coherence 0.9999, which is what says it is hardware.")
    parser.add_argument(
        "--antenna-coupling-scale", type=float, default=1.0,
        help="0 disables the coupling without removing the calibration")
    parser.add_argument(
        "--diffuse-floor-support", choices=("geometry", "rectangles"),
        default="geometry",
        help="where the diffuse floor exists. `geometry` (default): every "
             "range bin the camera's scene reconstruction puts surfaces in, "
             "the person's band included. `rectangles`: the two hand-drawn "
             "windows below, kept as the arbiter -- they correlate -0.27 to "
             "-0.60 with the floor measured from the ADC.")
    parser.add_argument(
        "--diffuse-floor-support-smoothing-bins", type=int, default=9,
        help="boxcar over range applied to the scene's range histogram, to "
             "remove the depth model's sampling grain without touching the "
             "metre-scale structure a split-half test shows is real")
    parser.add_argument("--diffuse-floor-seed", type=int, default=100)
    parser.add_argument("--diffuse-floor-scale", type=float, default=1.0)
    for radar in dataset.radars:
        parser.add_argument(radar.flag("frame-receiver-nuisance", "scale"), type=float, default=0.0)
    for radar in dataset.radars:
        parser.add_argument(radar.flag("frame-receiver-nuisance", "correlation"), type=float, default={"horizontal": 0.9, "vertical": 0.88}.get(radar.name, 0.9))
    parser.add_argument(
        "--frame-receiver-nuisance-scatterers-per-bin",
        type=int, default=48,
    )
    parser.add_argument(
        "--frame-receiver-nuisance-mode",
        choices=("complex", "amplitude-aligned", "common-range-gain"),
        default="complex",
    )
    parser.add_argument(
        "--frame-receiver-nuisance-range-smoothing", type=int, default=3,
    )
    parser.add_argument(
        "--frame-receiver-nuisance-seed", type=int, default=2100,
    )
    for radar in dataset.radars:
        parser.add_argument(radar.flag("local-additive-nuisance", "scale"), type=float, default=0.0)
    for radar in dataset.radars:
        parser.add_argument(radar.flag("local-additive-nuisance", "correlation"), type=float, default=0.0)
    parser.add_argument(
        "--local-additive-nuisance-scatterers-per-bin",
        type=int,
        default=48,
    )
    parser.add_argument(
        "--local-additive-nuisance-seed", type=int, default=4100,
    )
    parser.add_argument("--slow-time-nuisance-scale", type=float, default=0.0,
                        help="coherent within-frame diffuse amplitude relative to scene")
    for radar in dataset.radars:
        parser.add_argument(radar.flag("slow-time-nuisance", "scale"), type=float)
    parser.add_argument("--slow-time-nuisance-calibration", type=Path,
                        help="measured nonzero-Doppler range-envelope calibration JSON")
    parser.add_argument("--slow-time-nuisance-min-amplitude", type=float, default=0.01,
                        help="minimum normalized range envelope before nuisance scaling")
    parser.add_argument(
        "--slow-time-nuisance-mode",
        choices=("spectral", "thermal", "phase-noise"),
        default="spectral",
        help=(
            "spectral reproduces a calibrated RD envelope; thermal injects a "
            "flat coherent floor; phase-noise uses an analytic DC-decaying "
            "ridge plus weak white floor"
        ),
    )
    parser.add_argument("--slow-time-nuisance-correlation", type=float, default=0.0,
                        help="fallback AR(1) correlation across 10 Hz frames")
    for radar in dataset.radars:
        parser.add_argument(radar.flag("slow-time-nuisance", "correlation"), type=float, help=f"{radar.name} spectral-innovation AR(1) correlation across frames")
    for radar in dataset.radars:
        parser.add_argument(radar.flag("slow-time-nuisance", "empty-correlation"), type=float, help=f"latent AR(1) correlation in the {radar.name} 9.4-10.8 m, |Doppler|>=20 pure-noise region")
    for radar in dataset.radars:
        parser.add_argument(radar.flag("slow-time-nuisance", "fixed-fraction"), type=float, default=0.0)
    for radar in dataset.radars:
        parser.add_argument(radar.flag("slow-time-nuisance", "scene-fraction"), type=float, default=0.0, help="fraction of the calibrated slow-time range marginal reallocated onto the rendered background's own range profile, modelling the clutter-proportional phase-noise skirt")
    parser.add_argument(
        "--virtual-channel-calibration", type=Path, default=None,
        help=(
            "frozen held-out per-virtual-channel complex calibration residual "
            "from mm4DPS.calibration.geometry"
        ),
    )
    for radar in dataset.radars:
        parser.add_argument(radar.flag("absolute-gain"), type=float, default=1.0, help='frozen hardware receiver gain applied to the complex range profiles; fit from held-out sequences only, never from the application sequence')
    parser.add_argument(
        "--slow-time-nuisance-scene-density-normalize",
        action="store_true",
        help=(
            "divide the rendered background range profile by sqrt(points per "
            "range bin) before mixing, removing the monocular reconstruction's "
            "sampling-density artefact"
        ),
    )
    parser.add_argument(
        "--slow-time-nuisance-scene-smoothing-bins",
        type=int, default=33,
        help=(
            "range-bin width of the moving average applied to the rendered "
            "background profile before it is mixed into the slow-time floor; "
            "must stay broad enough that scene texture is not imprinted"
        ),
    )
    for radar in dataset.radars:
        parser.add_argument(radar.flag("slow-time-nuisance", "knee-bins"), type=float, default=3.0)
    for radar in dataset.radars:
        parser.add_argument(radar.flag("slow-time-nuisance", "white-power-fraction"), type=float, default={"horizontal": 0.35, "vertical": 0.75}.get(radar.name, 0.35))
    parser.add_argument("--slow-time-nuisance-scatterers-per-bin", type=int, default=4,
                        help="coherent angular scatterers in each active range bin")
    parser.add_argument(
        "--slow-time-nuisance-angle-source", choices=("calibrated", "uniform"),
        default="calibrated",
        help="use four calibrated directions or a seeded dense diffuse field",
    )
    parser.add_argument(
        "--slow-time-nuisance-edge-blend-bins", type=int, default=9,
        help="raised-cosine taper width, in range bins, applied to the "
             "calibrated Doppler-range envelope. The cross-sequence "
             "calibration flattens a hard 9.4-10.8 m rectangle into it, which "
             "the renderer reproduces as a vertical step in every RD frame "
             "(measured 40.5%% at 10.78 m against 12.6%% in the measurement). "
             "0 or 1 disables the taper and restores the pre-fix behaviour, "
             "which is the arbiter.")
    parser.add_argument("--slow-time-nuisance-seed", type=int, default=1100)
    parser.add_argument(
        "--nuisance-backend", choices=("torch", "numpy"), default="torch",
        help="where the slow-time nuisance cube's inverse FFT and steering "
             "projection run. Both keep every random draw and the whole AR(1) "
             "state on the CPU, so the seeded nuisance realization is "
             "identical either way; only the arithmetic moves, in complex128. "
             "'numpy' is the arbiter and was the only path before this flag "
             "existed.")
    parser.add_argument(
        "--nuisance-draws", choices=("cpu", "gpu"), default="gpu",
        help="where the nuisance models' random draws and AR(1) latent state "
             "live. 'cpu' is the seeded numpy stream every earlier render "
             "used. 'gpu' draws with torch on the device: a DIFFERENT "
             "realization of the same process, not a rounding difference, so "
             "it changes every rendered product and is gated statistically "
             "rather than against a tolerance ("
             "on sequence 102 over 600 frames, all 14 scored metrics move by "
             "less than re-seeding the CPU path does). 'cpu' is the arbiter "
             "and reproduces every render made before this flag existed. "
             "Requires --nuisance-backend torch.")
    parser.add_argument(
        "--velocity-differentiator",
        choices=("savgol", "central", "fourth-order", "band-limited"), default="savgol",
        help=(
            "how joint velocities are differentiated from the 10 Hz track; "
            "savgol is the historical 1.1 s window and discards about 10 dB "
            "of radial velocity relative to a central difference"
        ),
    )
    parser.add_argument(
        "--velocity-scale-mode", choices=("checkpoint", "identity"),
        default="checkpoint",
        help=(
            "identity discards the checkpoint's per-joint velocity scales and "
            "uses the camera-measured joint velocities directly"
        ),
    )
    parser.add_argument(
        "--velocity-scale-floor", type=float, default=0.0,
        help=(
            "minimum magnitude for any joint's velocity scale; 0 disables. "
            "Prevents the checkpoint from rendering a joint slower than the "
            "camera measured it"
        ),
    )
    parser.add_argument(
        "--velocity-gain", type=float, default=1.0,
        help=(
            "uniform multiplier on the checkpoint's per-joint velocity scales; "
            "raises micro-Doppler modulation depth without changing which "
            "joints move relative to each other"
        ),
    )
    parser.add_argument(
        "--velocity-frame-offset", type=float,
        help="override checkpoint velocity offset; fractional values are supported by surface models",
    )
    parser.add_argument("--loop-chunk-size", type=int, default=4)
    parser.add_argument("--motion-detection-result", type=Path, required=True,
                        help="this sequence's `motion detect-person-motion` result")
    parser.add_argument("--pose-audit-result", type=Path, default=None,
                        help="this sequence's pose audit; restricts the metrics to its moving frames")
    parser.add_argument("--save-pose-motion-dra", action="store_true")
    parser.add_argument(
        "--save-pose-motion-dra-complex",
        action="store_true",
        help=(
            "also save compact complex DRA tensors on pose-motion frames; "
            "requires --save-pose-motion-dra"
        ),
    )
    parser.add_argument(
        "--save-raw-rd-products", action="store_true",
        help="save angle-mean RD before slow-time mean subtraction",
    )
    parser.add_argument(
        "--save-radproposer-products",
        action="store_true",
        help=(
            "save measured/rendered official complex RadProPoser input tensors; "
            "about 5 GB for 600 frames"
        ),
    )
    parser.add_argument(
        "--save-first-loop-ra-products",
        action="store_true",
        help="save measured/rendered complex first-loop RA tensors",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    print(json.dumps(run(make_parser().parse_args()), indent=2))
