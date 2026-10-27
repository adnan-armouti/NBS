"""Calibrations from the recording: the antenna coupling, the motion tracks.

    sequence-coupling   each radar's static near-field coupling, from this
                        sequence's own recording (the chain's preprocessing)
    noise-envelope      the floor prior's shape from held-out sequences, a
                        diagnostic (the trainer learns the floor from the
                        sequence itself)
    motion tracks       the batched energy maps and tracks `motion
                        detect-person-motion` fits to

Everything that used to fit a seed file -- the frame-0 person range sweep,
the angle axes, the person-track extrinsic -- is gone with the seeds
(docs/SEEDS.md): the person is registered to the radar frame by frame as
its sequence is built (`reconstruction person-sequence`).
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import torch

from pathlib import Path

from mm4DPS import Dataset, current_dataset, dispatch
from mm4DPS.data import MeasuredReader, range_angle_slow_time_cube_torch, range_profiles_all_channels_torch
from mm4DPS.signal_model import official_motion_doppler_mask


# ----------------------------------------------------------------------
# noise-envelope: the trainer's floor prior for a dataset that ships none
# ----------------------------------------------------------------------

def run_noise_envelope(args: argparse.Namespace) -> dict:
    """The learned noise floor's prior shape, from HELD-OUT sequences of a dataset.

    HuPR's `normalized_doppler_range_profile` came from three held-out
    sequences; this is the same object for another dataset, from the same
    kind of data: per frame the angle-mean of the clutter-removed |RD|
    (Doppler x range), the person's range band (the annotated torso range
    +/- `--person-margin-m`) dropped, the median over every frame of every
    held-out sequence, then divided by its median over the empty region
    (|Doppler| >= 20 bins, 9.4-10.8 m) so the white receiver floor sits at 1.
    Only the shape matters to the trainer: LearnedNoiseFloor learns the
    scale and a range correction on top of it.

    Without this prior a noise-free trainer has no floor to explain the
    measured spectrogram's floor with, and the velocity field is driven to
    smear the person over Doppler to fill it (RT-Pose seq 183, 400 epochs:
    rendered spread 15.2 bins against 3.9 measured, centroid r +0.19).
    """
    import torch
    from mm4DPS import Dataset
    from mm4DPS.data import MeasuredReader, RTPoseSequence, range_angle_slow_time_cube_frames_torch
    dataset = Dataset.of(args)
    device = torch.device(args.device)
    radar = dataset.radars[0]
    spacing = range_bin_spacing_m(radar)
    profiles = []
    for sequence in args.sequences:
        seq = RTPoseSequence(dataset, sequence)
        annotation = seq.annotation()
        reader = MeasuredReader(dataset, sequence, radar)
        for start in range(0, reader.frames, args.chunk):
            count = min(args.chunk, reader.frames - start)
            adc = reader.batch(start, count, device)
            ra = range_angle_slow_time_cube_frames_torch(adc, radar=radar)          # (n, loops, range, angle)
            # The envelope describes the noise PROCESS: measured with the
            # unwindowed Doppler transform (mean removed), and the pipeline's
            # window is applied where the envelope is used -- the trainer
            # smooths the prior by the window's power kernel
            # (`signal_model.window_power_kernel`), the renderer draws slow-time
            # noise from it and the product's consumer windows the draw.
            ra = ra - ra.mean(dim=1, keepdim=True)
            rd = torch.fft.fftshift(torch.fft.fft(ra, dim=1), dim=1).abs().mean(dim=3)  # (n, loops, range)
            rd = rd.cpu().numpy()
            for i in range(count):
                entry = annotation.get(start + i)
                if entry is not None:
                    torso = entry["joints"][[0, 7, 9, 12]].mean(axis=0) + RTPoseSequence.ANNOTATION_TO_LIDAR
                    centre = float(np.hypot(torso[0], torso[1]))
                    lo = max(0, int((centre - args.person_margin_m) / spacing))
                    hi = min(radar.samples, int((centre + args.person_margin_m) / spacing) + 1)
                    rd[i, :, lo:hi] = np.nan
                profiles.append(rd[i])
        reader.close()
    stack = np.stack(profiles)
    median = np.nanmedian(stack, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    loops = radar.loops
    doppler = np.abs(np.arange(loops) - loops // 2)
    empty = np.zeros_like(median, dtype=bool)
    empty[doppler >= 20, int(9.4 / spacing):int(10.8 / spacing) + 1] = True
    level = float(np.median(median[empty]))
    normalized = median / max(level, 1e-30)
    result = {
        "status": "completed", "dataset": dataset.name,
        "calibration_sequences": [int(s) for s in args.sequences], "frames": int(len(profiles)),
        "fit_policy": "median over every frame of the held-out sequences, the annotated person's range band "
                      f"(+/- {args.person_margin_m} m) dropped per frame; the scalar level is the median over "
                      "|Doppler| >= 20 bins and 9.4-10.8 m (the unit white floor)",
        "radars": {radar.name: {
            "normalized_doppler_range_profile": normalized.tolist(),
            "normalized_range_profile": normalized.mean(axis=0).tolist(),
            "empty_region_level": level, "person_margin_m": args.person_margin_m}},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "radars"}, indent=2))
    return result


def make_parser_noise_envelope() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=run_noise_envelope.__doc__)
    parser.add_argument("--dataset", default="rtpose", choices=("rtpose",))
    parser.add_argument("--sequences", type=int, nargs="+", required=True, help="HELD-OUT sequences")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--person-margin-m", type=float, default=0.75)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _entry_noise_envelope():
    run_noise_envelope(make_parser_noise_envelope().parse_args())


# ----------------------------------------------------------------------
# sequence-coupling: the board's static near-field coupling from the sequence's own recording
# ----------------------------------------------------------------------

def run_sequence_coupling(args: argparse.Namespace) -> dict:
    """The static antenna coupling of each radar, from this sequence's recording.

    HuPR's coupling is a seed fitted on held-out sequences (`docs/SEEDS.md`);
    a dataset without one gets the same quantity from the recording it is
    about to render: the coherent mean over frames and loops of the
    range-windowed profiles at the near range bins, per azimuth virtual
    channel -- ``(azimuth TX x RX, near_bins)`` complex64, the layout
    `video._configure_calibrations` reads. Nothing else lives within the
    near bins (0.5 m on the cascade): the room is metres away and the person
    never stands on the rig, so the coherent mean over the sequence is the
    coupling alone (the moving person averages out; a static object would
    not, and would belong there anyway). On the cascade the coupling column
    is 49% of the static map's energy and the render had nothing there.
    The mean over every `--stride`-th frame; the fraction of the near-bin
    energy the mean explains is recorded as the coherence.
    """
    from mm4DPS.data import MeasuredReader, range_profiles_all_channels_torch
    dataset = Dataset.select(args.dataset)
    device = args.device
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"status": "completed", "dataset": dataset.name, "sequence": int(args.sequence),
                    "near_bins": int(args.near_bins), "stride": int(args.stride), "radars": {},
                    "model": "coherent mean over frames and loops of the range-windowed profiles at the near "
                             "range bins, per azimuth virtual channel; this sequence's own recording"}
    for radar in dataset.radars:
        reader = MeasuredReader(dataset, args.sequence, radar)
        frames = list(range(0, min(reader.frames, args.frame_count), args.stride))
        total = torch.zeros(len(radar.azimuth_tx), radar.n_rx, args.near_bins, dtype=torch.complex64, device=device)
        power = 0.0

        def accumulate(adc) -> None:
            nonlocal power
            profiles = range_profiles_all_channels_torch(adc, radar=radar)          # (loops, tx, rx, range)
            near = profiles[:, list(radar.azimuth_tx), :, :args.near_bins]
            total.add_(near.mean(dim=0))
            power += float(near.abs().square().mean())

        if radar.decoder == "dca1000":
            # A HuPR archive member is deflated: a strided read that skips
            # frames reopens and re-inflates from the start each time (347 s
            # for a sequence) -- so one sequential pass, keeping every
            # stride-th frame.
            last = frames[-1] + 1
            for start in range(0, last, args.batch):
                block = reader.batch(start, min(args.batch, last - start), device)
                for offset in range(len(block)):
                    if (start + offset) % args.stride == 0:
                        accumulate(block[offset])
        else:
            for frame in frames:                       # the cascade: a file per frame
                accumulate(reader.batch(frame, 1, device)[0])
        coupling = (total / len(frames)).reshape(-1, args.near_bins).cpu().numpy()   # (azimuth TX x RX, near)
        coherence = float(np.abs(coupling).__pow__(2).mean() / max(power / len(frames), 1e-30))
        np.save(args.output_dir / f"{radar.name}_coupling.npy", coupling.astype(np.complex64))
        result["radars"][radar.name] = {"shape": list(coupling.shape), "frames": len(frames),
                                        "mean_magnitude": float(np.abs(coupling).mean()),
                                        "coherent_fraction_of_near_power": coherence}
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def make_parser_sequence_coupling() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=run_sequence_coupling.__doc__)
    parser.add_argument("--dataset", default="rtpose")
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--near-bins", type=int, default=15, help="HuPR's seed uses 15 (0.5 m on the cascade)")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--frame-count", type=int, default=600)
    parser.add_argument("--batch", type=int, default=16, help="frames per read of a deflated archive")
    parser.add_argument("--device", default="cuda:0")
    return parser


def _entry_sequence_coupling():
    run_sequence_coupling(make_parser_sequence_coupling().parse_args())


COMMANDS = {
    "noise-envelope": _entry_noise_envelope,
    "sequence-coupling": _entry_sequence_coupling,
}


def main() -> int:
    return dispatch(COMMANDS, __name__)


# ----------------------------------------------------------------------
# Batched GPU forms of `motion_energy_map` and `extract_motion_track`.
#
# The numpy pair above stays the arbiter, and `--backend numpy` still runs it;
# the test that pinned these against it went with `tests/` in the deployment
# strip. They exist because `detect-person-motion` ran a
# `for frame in range(600)` with a nested `for radar` at **0.0% GPU**, two
# thirds of it numpy FFTs. Everything below takes a leading frame axis and
# keeps the whole sequence on the device.
# ----------------------------------------------------------------------


def _gaussian_blur_reflect_torch(x, sigmas, truncate: float = 4.0):
    """``scipy.ndimage.gaussian_filter`` on the last two axes, in torch.

    Reproduces scipy's construction rather than approximating it: radius
    ``int(truncate*sigma + 0.5)``, weights ``exp(-x^2/2 sigma^2)`` normalised
    to sum one, applied separably. scipy's default ``mode='reflect'`` repeats
    the edge sample -- ``d c b a | a b c d | d c b a`` -- which is numpy's
    'symmetric' and is NOT torch's ``'reflect'`` padding, so it is built here
    by flipping rather than by ``F.pad``. Getting that wrong is silent: it
    only moves the two edge columns of the gate.
    """
    import torch
    import torch.nn.functional as F

    for dim, sigma in zip((-2, -1), sigmas):
        radius = int(truncate * float(sigma) + 0.5)
        if radius < 1:
            continue
        offsets = torch.arange(-radius, radius + 1,
                               device=x.device, dtype=x.dtype)
        weights = torch.exp(-0.5 * (offsets / float(sigma)) ** 2)
        weights = weights / weights.sum()
        length = x.shape[dim]
        flipped = x.flip(dim)
        left = flipped.narrow(dim, length - radius, radius)
        right = flipped.narrow(dim, 0, radius)
        padded = torch.cat([left, x, right], dim=dim)
        moved = padded.movedim(dim, -1)
        shape = moved.shape
        # The kernel is symmetric, so conv1d's cross-correlation is scipy's
        # correlate1d on the same weights.
        blurred = F.conv1d(moved.reshape(-1, 1, shape[-1]),
                           weights.view(1, 1, -1))
        x = blurred.reshape(*shape[:-1], -1).movedim(-1, dim)
    return x


def motion_energy_maps_torch(frames_adc):
    """``(frames, RX, chirps, ADC)`` -> ``(frames, range, angle)`` float32.

    The batched form of :func:`motion_energy_map`: same operations, same
    order, one leading frame axis. Differs from the numpy arbiter at the
    ~1e-7 level -- cuFFT's reduction order and torch's complex64 slow-time
    mean, both characterised in `mm4DPS/adc.py`'s beamforming docstring.
    """
    import torch
    from mm4DPS.data import range_angle_slow_time_cube_frames_torch
    from mm4DPS.signal_model import official_motion_doppler_mask

    from mm4DPS.forward_render import slow_time_to_doppler
    ra = range_angle_slow_time_cube_frames_torch(frames_adc)
    doppler = slow_time_to_doppler(ra.to(torch.complex128), dim=1).abs()
    mask = torch.from_numpy(official_motion_doppler_mask()).to(ra.device)
    return doppler[:, mask].mean(dim=1).to(torch.float32)


def extract_motion_tracks_torch(
    energy_ra, person_points_xyz, point_counts, radar_center_xyz,
    radar_array_axis_xyz, *, range_padding_m: float = 0.18,
    angle_padding_sine: float = 0.12,
):
    """Batched :func:`extract_motion_track`.

    ``person_points_xyz`` is ``(frames, max_points, 3)`` padded with NaN and
    ``point_counts`` says how many rows of each are real, so a sequence whose
    cloud changes length frame to frame still reduces in one call --
    `nanquantile` ignores the padding.

    Returns ``(observed_range, observed_sine, confidence)``, each ``(frames,)``.
    """
    import torch
    from mm4DPS.data import range_bin_spacing_m

    device = energy_ra.device
    frames = energy_ra.shape[0]
    if energy_ra.ndim != 3:
        raise ValueError("energy maps must have shape (frames, range, angle)")
    n_range, n_angle = energy_ra.shape[1], energy_ra.shape[2]
    points = person_points_xyz.to(torch.float64)
    centre = radar_center_xyz.to(device=device, dtype=torch.float64).reshape(1, 1, 3)
    axis = radar_array_axis_xyz.to(device=device, dtype=torch.float64).reshape(3)
    axis = axis / torch.clamp(torch.linalg.vector_norm(axis), min=1e-12)

    delta = points - centre
    ranges = torch.linalg.vector_norm(delta, dim=-1)
    person_sine = (delta @ axis) / torch.clamp(ranges, min=1e-8)
    valid = (torch.arange(points.shape[1], device=device)[None, :]
             < point_counts.to(device)[:, None])
    nan = torch.tensor(float("nan"), device=device, dtype=torch.float64)
    ranges_v = torch.where(valid, ranges, nan)
    sine_v = torch.where(valid, person_sine, nan)

    range_lo = torch.clamp(
        torch.nanquantile(ranges_v, 0.02, dim=1) - range_padding_m, min=0.0)
    range_hi = torch.nanquantile(ranges_v, 0.98, dim=1) + range_padding_m
    sine_lo = torch.nanquantile(sine_v, 0.02, dim=1) - angle_padding_sine
    sine_hi = torch.nanquantile(sine_v, 0.98, dim=1) + angle_padding_sine

    range_values = (torch.arange(n_range, device=device, dtype=torch.float64)
                    * range_bin_spacing_m())
    sine_values = 2.0 * (torch.arange(n_angle, device=device, dtype=torch.float64)
                         - n_angle / 2) / n_angle
    range_gate = ((range_values[None, :] >= range_lo[:, None])
                  & (range_values[None, :] <= range_hi[:, None]))
    direct = ((sine_values[None, :] >= sine_lo[:, None])
              & (sine_values[None, :] <= sine_hi[:, None]))
    mirrored = ((sine_values[None, :] >= -sine_hi[:, None])
                & (sine_values[None, :] <= -sine_lo[:, None]))
    gate = range_gate[:, :, None] & (direct | mirrored)[:, None, :]
    if int(gate.flatten(1).sum(dim=1).min()) < 20:
        raise ValueError("person gate is too small")

    energy = energy_ra.to(torch.float64)
    peak = torch.clamp(energy.flatten(1).max(dim=1).values, min=1e-12)
    db = torch.clamp(
        20.0 * torch.log10(energy / peak[:, None, None] + 1e-12), -55.0, 0.0)
    smooth = _gaussian_blur_reflect_torch(db, (1.2, 0.9))

    neg_inf = torch.tensor(float("-inf"), device=device, dtype=torch.float64)
    gated = torch.where(gate, smooth, neg_inf)
    flat_peak = gated.flatten(1).argmax(dim=1)
    peak_r = torch.div(flat_peak, n_angle, rounding_mode="floor")
    peak_a = flat_peak - peak_r * n_angle

    rr = torch.arange(n_range, device=device)[None, :, None]
    aa = torch.arange(n_angle, device=device)[None, None, :]
    local = (gate & ((rr - peak_r[:, None, None]).abs() <= 3)
             & ((aa - peak_a[:, None, None]).abs() <= 4))

    masked = torch.where(gate, smooth, nan)
    floor = torch.nanquantile(masked.flatten(1), 0.70, dim=1)
    median = torch.nanquantile(masked.flatten(1), 0.50, dim=1)

    weights = torch.where(
        local, torch.clamp(smooth - floor[:, None, None], min=0.0),
        torch.zeros((), device=device, dtype=torch.float64))
    total = weights.flatten(1).sum(dim=1)
    # The numpy form's fallback, vectorised: a frame whose gated window is
    # entirely at or below the 70th percentile weights the window uniformly.
    degenerate = total <= 1e-8
    weights = torch.where(degenerate[:, None, None], local.to(torch.float64),
                          weights)
    total = torch.where(degenerate, local.flatten(1).sum(dim=1).to(torch.float64),
                        total)

    observed_range = ((weights * range_values[None, :, None]).flatten(1).sum(dim=1)
                      / total)
    observed_sine = ((weights * sine_values[None, None, :]).flatten(1).sum(dim=1)
                     / total)
    confidence = smooth.flatten(1).gather(1, flat_peak[:, None]).squeeze(1) - median
    return observed_range, observed_sine, confidence


if __name__ == "__main__":
    raise SystemExit(main())
