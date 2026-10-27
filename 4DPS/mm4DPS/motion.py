"""Per-frame motion detection and the pose-derived motion frames every
correlation metric is gated on.
"""

from __future__ import annotations

import json
import matplotlib.pyplot as plt
import yaml
import numpy as np
import zipfile
import torch


from mm4DPS.calibration import extract_motion_tracks_torch, motion_energy_maps_torch


from mm4DPS.signal_model import load_trajectories

from mm4DPS import Dataset, config_centre_m, current_dataset, dispatch, radar_configs, sequence_dir

from mm4DPS import DATASET_ROOT, HUPR_OFFICIAL_ROOT, WORKSPACE

import argparse

import matplotlib
from mm4DPS.data import HUPR, MeasuredReader, find_archive

HUPR_FRAME_WORDS = HUPR.radars[0].raw_words_per_frame
from mm4DPS import array_axis
from pathlib import Path


# ----------------------------------------------------------------------
# detect_person_motion: Detect H/V-consistent human motion tracks throughout one HuPR sequence.
# ----------------------------------------------------------------------
matplotlib.use("Agg")




# Colours for the timeline; any radar not named here takes the default cycle.
RADAR_COLORS = {"horizontal": "#4c78a8", "vertical": "#e45756"}






def render_timeline(result: dict, output: Path) -> None:
    rows = result["frame_records"]
    frames = np.asarray([row["frame"] for row in rows])
    accepted = np.asarray([row["accepted"] for row in rows], dtype=bool)
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    colors = RADAR_COLORS
    for radar in current_dataset().names:
        confidence = np.asarray([row["radars"][radar]["confidence_db"] for row in rows])
        residual = np.asarray([row["radars"][radar]["range_residual_m"] for row in rows])
        observed = np.asarray([row["radars"][radar]["observed_range_m"] for row in rows])
        axes[0].plot(frames, confidence, color=colors.get(radar), linewidth=0.8, label=radar)
        axes[1].plot(frames, residual, color=colors.get(radar), linewidth=0.8, label=radar)
        axes[2].plot(frames, observed, color=colors.get(radar), linewidth=0.8, label=radar)
    axes[0].axhline(result["thresholds"]["minimum_confidence_db"], color="black", linestyle="--")
    residual_limit = result["thresholds"]["maximum_corrected_range_residual_m"]
    axes[1].axhline(residual_limit, color="black", linestyle="--")
    axes[1].axhline(-residual_limit, color="black", linestyle="--")
    for axis in axes:
        for frame in frames[accepted]:
            axis.axvline(frame, color="#54a24b", alpha=0.08, linewidth=1.0)
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")
    axes[0].set_ylabel("Peak confidence (dB)")
    axes[1].set_ylabel("Observed - corrected\ncloud range (m)")
    axes[2].set_ylabel("Observed range (m)")
    axes[2].set_xlabel("Frame index")
    fig.suptitle(
        "HuPR sequence 1: full-sequence dual-radar human-motion detection\n"
        f"green = accepted ({result['accepted_frames_count']}/600 frames)",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)



def run_detect_person_motion(args: argparse.Namespace) -> dict:
    translation_y = float(args.translation_y_m)
    calibration_source = "explicit --translation-y-m"
    dataset = Dataset.of(args)
    configs = dataset.configs(args)
    RADARS = dataset.names
    # No prior detector to measure recall against: a sequence has only
    # its own detection.
    calibration = None
    calibration_reliable = None

    # The whole sequence, on the GPU, with no Python loop over frames.
    #
    # This stage used to be a `for frame in range(600)` with a nested
    # `for radar`, in numpy, measured at **0.0% GPU** with 99% of samples at
    # zero -- two thirds of it the range/angle/Doppler FFTs through
    # single-threaded pocketfft. Every step now runs batched on the device:
    # the raw int16 words go up, and one array of per-frame scalars comes
    # back. The only host work left is reading bytes off disk and writing
    # JSON.
    #
    # It is not bitwise against the numpy arbiter and cannot be: cuFFT's
    # reduction order differs, and torch reduces the complex64 slow-time mean
    # differently. Measured on 24 real frames of sequence 102, both radars:
    # observed_range 1.2e-07, observed_spatial_sine 1.7e-06, confidence_db
    # 1.7e-06, all relative. `--backend numpy` runs the original path and is
    # the arbiter. The test that pinned the pair went with `tests/`.
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.frames is None:
        args.frames = int(json.loads((args.sequence_dir / "metadata.json").read_text())["frames"])
    frames = list(range(args.frames))

    # The person clouds, stacked once. Padded with NaN and carried with their
    # true lengths so a sequence whose cloud changes length frame to frame
    # still reduces in one call.
    clouds = [
        np.load(args.sequence_dir / "person_frames" / f"frame_{frame:03d}.npz"
                )["pcl"][:, :3].astype(np.float32)
        for frame in frames
    ]
    counts = np.array([len(cloud) for cloud in clouds], dtype=np.int64)
    padded = np.full((len(frames), int(counts.max()), 3), np.nan, np.float64)
    for index, cloud in enumerate(clouds):
        padded[index, :len(cloud)] = cloud
    points_t = torch.from_numpy(padded).to(device)
    counts_t = torch.from_numpy(counts).to(device)

    # The representative point is the median of the *translated* cloud, so it
    # is taken over the real rows only.
    corrected = points_t.clone()
    corrected[:, :, 1] += translation_y
    valid = (torch.arange(padded.shape[1], device=device)[None, :]
             < counts_t[:, None])
    # `nanquantile(0.5)`, NOT `nanmedian`: for an even number of points numpy
    # averages the two middle values and torch's median returns the lower of
    # them. On a 6,890-point cloud that is a real difference, not rounding --
    # measured 2.29e-04 m on `corrected_cloud_range_m`, which feeds the
    # `corrected_range_residual` gate. `quantile` with linear interpolation is
    # numpy's definition.
    representative = torch.nanquantile(
        torch.where(valid[:, :, None], corrected,
                    torch.tensor(float("nan"), device=device,
                                 dtype=corrected.dtype)),
        0.5, dim=1)

    tracks: dict[str, dict] = {}
    if True:
        for radar_desc in dataset.radars:
            radar = radar_desc.name
            config = configs[radar]
            centre = config_centre_m(config).astype(np.float32)
            energies = []
            reader = MeasuredReader(dataset, args.sequence, radar_desc, dataset_root=args.dataset_root)
            # Sequential reads in chunks: one host->device transfer of raw
            # int16 per chunk, and the decode itself on the device. The chunk
            # bounds the complex128 Doppler cube, which is what the peak
            # memory is (the cascade's frame is 16x HuPR's, so its chunk is
            # cut to match).
            chunk = max(1, args.frame_chunk * (HUPR_FRAME_WORDS // radar_desc.raw_words_per_frame or 1))
            for start in range(0, len(frames), chunk):
                count = min(chunk, len(frames) - start)
                energies.append(motion_energy_maps_torch(reader.batch(start, count, device)))
            reader.close()
            energy = torch.cat(energies)
            observed_range, observed_sine, confidence = extract_motion_tracks_torch(
                energy, points_t, counts_t,
                torch.from_numpy(centre),
                torch.from_numpy(np.asarray(array_axis(config))),
            )
            predicted = torch.linalg.vector_norm(
                representative - torch.from_numpy(centre).to(
                    device=device, dtype=representative.dtype)[None, :],
                dim=-1)
            tracks[radar] = {
                "observed_range_m": observed_range,
                "observed_spatial_sine": observed_sine,
                "confidence_db": confidence,
                "corrected_cloud_range_m": predicted,
                "range_residual_m": observed_range - predicted,
            }
            print(f"detected human motion, {radar}: {args.frames} frames on "
                  f"{device}", flush=True)

    # The gates, vectorised over frames -- `classify_detection`'s three
    # predicates, in the same order, so `rejection_reasons` reads the same.
    low_confidence = torch.stack(
        [tracks[radar]["confidence_db"] < args.minimum_confidence_db
         for radar in RADARS]).any(dim=0)
    # Cross-radar range agreement: the spread of the observed ranges across
    # radars (for two, |H - V|; for one, zero, so the gate never fires).
    observed = torch.stack([tracks[radar]["observed_range_m"] for radar in RADARS])
    disagreement = observed.max(dim=0).values - observed.min(dim=0).values
    hv_disagree = disagreement > args.maximum_hv_range_disagreement_m
    residual = torch.stack(
        [tracks[radar]["range_residual_m"].abs()
         > args.maximum_corrected_range_residual_m
         for radar in RADARS]).any(dim=0)

    # One transfer for the whole sequence: three flags and five scalars per
    # radar per frame. Everything above stayed on the device.
    flags = torch.stack([low_confidence, hv_disagree, residual]).cpu().numpy()
    scalars = {
        radar: {key: value.cpu().numpy()
                for key, value in tracks[radar].items()}
        for radar in RADARS
    }
    frame_records = []
    for frame in frames:
        rejection_reasons = [
            name for name, column in zip(
                ("confidence", "h_v_range_disagreement",
                 "corrected_range_residual"), flags)
            if bool(column[frame])
        ]
        frame_records.append({
            "frame": frame,
            "accepted": not rejection_reasons,
            "rejection_reasons": rejection_reasons,
            "radars": {
                radar: {key: float(values[frame])
                        for key, values in scalars[radar].items()}
                for radar in RADARS
            },
        })

    return _finish_detect_person_motion(
        args, frame_records, translation_y, calibration_source,
        calibration_reliable, calibration)


def _finish_detect_person_motion(args, frame_records, translation_y,
                                 calibration_source, calibration_reliable,
                                 calibration) -> dict:
    """The summary both backends share, so neither can drift from the other.

    `calibration` is None whenever `--translation-y-m` was given, which is the
    only case the chain uses; the recall fields it feeds are then reported as
    absent rather than invented, exactly as before.
    """
    accepted_frames = [row["frame"] for row in frame_records if row["accepted"]]
    accepted_set = set(accepted_frames)
    reason_counts = {
        reason: sum(reason in row["rejection_reasons"] for row in frame_records)
        for reason in ("confidence", "h_v_range_disagreement", "corrected_range_residual")
    }
    result = {
        "status": "completed",
        "sequence": args.sequence,
        "frames": args.frames,
        "geometry_translation_y_m": translation_y,
        "geometry_translation_source": calibration_source,
        "thresholds": {
            "minimum_confidence_db": args.minimum_confidence_db,
            "maximum_hv_range_disagreement_m": args.maximum_hv_range_disagreement_m,
            "maximum_corrected_range_residual_m": args.maximum_corrected_range_residual_m,
            "provenance": (
                "thresholds were fixed before the full-sequence run and do not use "
                "v7/v8 rendered correlation"
            ),
        },
        "accepted_frames": accepted_frames,
        "accepted_frames_count": len(accepted_frames),
        "coverage_fraction": len(accepted_frames) / args.frames,
        "calibration_reliable_frames_count": (
            None if calibration_reliable is None else len(calibration_reliable)
        ),
        "calibration_reliable_recall": (
            None if calibration_reliable is None
            else len(accepted_set & calibration_reliable) / len(calibration_reliable)
        ),
        "new_frames_outside_calibration_candidates": (
            None if calibration_reliable is None else sorted(
                accepted_set - set(calibration["motion_frames"])
                - set(calibration["audit_frames"])
            )
        ),
        "rejection_reason_counts_nonexclusive": reason_counts,
        "frame_records": frame_records,
        "limits": [
            "the detector is motion-sensitive and does not determine visual person presence",
            "the camera person cloud supplies only a broad search gate and corrected range prior",
            "precision still requires an external empty-scene/person-motion ground truth audit",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    render_timeline(result, args.output_dir / "detection_timeline.png")
    return result





def make_parser_detect_person_motion() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--sequence", type=int, default=1)
    parser.add_argument("--frames", type=int, default=None,
                        help="frames to detect over (default: the sequence's)")
    sequence_dir(parser)
    radar_configs(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--translation-y-m", type=float, required=True,
        help="the person's range translation in this sequence (0 for a sequence "
             "registered to the radar per frame)",
    )
    parser.add_argument("--device", default="cuda:0",
                        help="device for the batched motion detection; falls "
                             "back to CPU when CUDA is unavailable")
    parser.add_argument("--frame-chunk", type=int, default=8,
                        help="frames per beamforming batch. Bounds the peak: "
                             "the complex128 Doppler cube is ~0.27 GB a frame, "
                             "so 8 is ~2.2 GB and the stage fits beside a "
                             "training run.")
    parser.add_argument("--minimum-confidence-db", type=float, default=3.0)
    parser.add_argument("--maximum-hv-range-disagreement-m", type=float, default=0.15)
    parser.add_argument("--maximum-corrected-range-residual-m", type=float, default=0.12)
    return parser


def _entry_detect_person_motion():
        print(json.dumps(run_detect_person_motion(make_parser_detect_person_motion().parse_args()), indent=2))

# ----------------------------------------------------------------------
# pose_motion_frames: Per-sequence pose-motion frame labels, camera derived.
# ----------------------------------------------------------------------
matplotlib.use("Agg")




THRESHOLD_SENSITIVITY = (0.10, 0.15, 0.20, 0.25, 0.30)



def pose_motion_score(velocities: np.ndarray) -> np.ndarray:
    """90th percentile across the fourteen smoothed joint speeds, per frame."""
    return np.percentile(np.linalg.norm(velocities, axis=2), 90, axis=1)



def render_figure(score: np.ndarray, threshold: float, sequence: int, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11.0, 3.4), dpi=120)
    axis.plot(score, linewidth=0.9, color="#1f77b4")
    axis.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.0,
                 label=f"{threshold:g} m/s")
    axis.fill_between(np.arange(len(score)), 0, score.max() * 1.05,
                      where=score >= threshold, color="#d62728", alpha=0.12,
                      label="motion frames")
    axis.set_xlim(0, len(score) - 1)
    axis.set_ylim(0, score.max() * 1.05)
    axis.set_xlabel("frame")
    axis.set_ylabel("pose speed (m/s)")
    axis.set_title(f"HuPR sequence {sequence} pose motion score")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)



def run_pose_motion_frames(args: argparse.Namespace) -> dict:
    _, velocities = load_trajectories(args.sequence_dir)
    score = pose_motion_score(velocities)
    if args.frame_count is not None:
        score = score[: args.frame_count]
    motion = np.flatnonzero(score >= args.pose_motion_threshold_m_s).astype(int).tolist()
    trained_on = set(json.loads(args.motion_training_result.read_text())["training_frames"])
    development = sorted(trained_on)
    independent = sorted(set(motion) - trained_on)

    result = {
        "status": "completed",
        "sequence": args.sequence,
        "pose_motion_definition": (
            "90th percentile across 14 smoothed 3-D joint speeds; camera-derived "
            "and independent of radar candidates and renderer correlation"
        ),
        "primary_pose_threshold_m_s": args.pose_motion_threshold_m_s,
        "threshold_rationale": (
            "0.15 m/s equals 1.5 cm per 10 Hz frame; the same threshold is used "
            "for every sequence so the figures stay comparable"
        ),
        "split_protocol": (
            "development frames are the surface-motion training frames; every "
            "other motion frame is independent and was never tuned against"
        ),
        "motion_training_result": str(args.motion_training_result),
        "frames": int(len(score)),
        "pose_motion_score_m_s": score.tolist(),
        "pose_motion_frames": motion,
        "model_development_frames": development,
        "independent_pose_motion_frames": independent,
        "median_pose_speed_m_s": float(np.median(score)),
        "threshold_sensitivity": [
            {
                "threshold_m_s": threshold,
                "pose_motion_frames": int(np.sum(score >= threshold)),
            }
            for threshold in THRESHOLD_SENSITIVITY
        ],
        "limits": [
            "camera pose does not label breathing or other sub-pose micromotion",
            "pose velocity is estimated at 10 Hz and can miss motion between frames",
            "no radar-candidate cross-check is included here; this file carries "
            "the camera labels only",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pose_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    # Downstream tools take a --motion-detection path as well; the labels are
    # the camera ones, so the same payload serves both and stays self-describing.
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    render_figure(np.asarray(score), args.pose_motion_threshold_m_s, args.sequence,
                  args.output_dir / "pose_audit.png")
    return result



def make_parser_pose_motion_frames() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=38)
    parser.add_argument("--sequence-dir", type=Path,
                        default=WORKSPACE / "person_sequence_38_hfov40")
    parser.add_argument("--pose-motion-threshold-m-s", type=float, default=0.15)
    parser.add_argument(
        "--motion-training-result", type=Path,
        default=WORKSPACE / "training/seq38_motion_hfov40/result.json",
        help="supplies the development frames, matching sequence 1's rule",
    )
    parser.add_argument("--frame-count", type=int)
    parser.add_argument("--output-dir", type=Path,
                        default=WORKSPACE / "calibration/seq38_pose_motion")
    return parser


def _entry_pose_motion_frames():
        summary = run_pose_motion_frames(make_parser_pose_motion_frames().parse_args())
        print(json.dumps({
            key: value for key, value in summary.items()
            if key != "pose_motion_score_m_s"
        }, indent=2)[:1200])




# ----------------------------------------------------------------------
# rank_sequence_motion: how sequences 38, 102 and 256 were chosen (docs/SEEDS.md)
# ----------------------------------------------------------------------



TORSO_JOINTS = (2, 5, 6, 7)          # hips, pelvis, head

LIMB_END_JOINTS = (0, 3, 10, 13)     # ankles, wrists



def sequence_scores(joints: np.ndarray, fps: float = 10.0) -> dict:
    """Motion statistics for one sequence's (frames, 14, 2) joint track."""
    if joints.ndim != 3 or joints.shape[1:] != (14, 2):
        raise ValueError(f"expected (frames,14,2) joints, got {joints.shape}")
    # Normalize by body scale so pixel units do not favour subjects who
    # happen to stand closer to the camera.
    scale = np.median(
        np.linalg.norm(joints[:, 7] - joints[:, 6], axis=1)
    )
    scale = float(scale) if scale > 1e-6 else 1.0
    velocity = np.gradient(joints, 1.0 / fps, axis=0) / scale
    speed = np.linalg.norm(velocity, axis=2)
    per_frame = speed.mean(axis=1)
    torso = speed[:, list(TORSO_JOINTS)].mean()
    limbs = speed[:, list(LIMB_END_JOINTS)].mean()
    moving = per_frame > np.percentile(per_frame, 50) * 2.0
    return {
        "mean_speed_body_lengths_per_s": float(per_frame.mean()),
        "p90_speed": float(np.percentile(per_frame, 90)),
        "moving_frame_fraction": float(moving.mean()),
        "limb_over_torso_speed": float(limbs / max(torso, 1e-9)),
        "body_scale_px": scale,
    }



def load_split(path: Path, names: list[int]) -> dict[int, np.ndarray]:
    entries = json.loads(path.read_text())
    if len(entries) != len(names):
        raise ValueError(
            f"{path.name} has {len(entries)} sequences but the config lists "
            f"{len(names)}"
        )
    out = {}
    for sequence, frames in zip(names, entries):
        out[int(sequence)] = np.array(
            [frame["joints"] for frame in frames], dtype=np.float64,
        )
    return out



def run(args: argparse.Namespace) -> dict:
    config = yaml.safe_load(args.official_config.read_text())["DATASET"]
    tracks: dict[int, np.ndarray] = {}
    for phase, filename in (
        ("train", "hrnet_annot_train.json"),
        ("val", "hrnet_annot_val.json"),
        ("test", "hrnet_annot_test.json"),
    ):
        names = [int(value) for value in config[f"{phase}Name"]]
        tracks.update(load_split(args.annotations / filename, names))

    scored = {
        sequence: sequence_scores(joints)
        for sequence, joints in sorted(tracks.items())
    }
    order = sorted(
        scored, key=lambda s: scored[s]["mean_speed_body_lengths_per_s"],
        reverse=True,
    )
    result = {
        "status": "completed",
        "sequences_scored": len(scored),
        "reference_sequence": args.reference,
        "reference_rank": order.index(args.reference) + 1
        if args.reference in scored else None,
        "ranking_by_mean_speed": order,
        "scores": scored,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result



def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations", type=Path, default=DATASET_ROOT / "annotations",
    )
    parser.add_argument(
        "--official-config", type=Path,
        default=HUPR_OFFICIAL_ROOT / "config/mscsa_prgcn.yaml",
    )
    parser.add_argument("--reference", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path, default=WORKSPACE / "calibration/sequence_motion_ranking",
    )
    return parser



COMMANDS = {
    "detect-person-motion": _entry_detect_person_motion,
    "pose-motion-frames": _entry_pose_motion_frames,
}


def main() -> int:
    return dispatch(COMMANDS, __name__)


if __name__ == "__main__":
    raise SystemExit(main())
