"""Comparison media, metric reports, and the permutation control.

`permutation_and_sweep` is here rather than in `scoring` because it imports
`mm4DPS.evaluate`, which imports `scoring` -- together they would form an
import cycle that neither had on its own.
"""

from __future__ import annotations

import itertools
import json
import shutil
import subprocess


from mm4DPS.data import ADC_SAMPLES, TDM_LOOPS, range_bin_spacing_m

from mm4DPS import config_centre_m, current_dataset, dispatch, pose_audit, sequence_dir


import argparse

import numpy as np

from mm4DPS.qa import profile_provenance_lines, require_same_profile
from mm4DPS import qa as physics_fingerprint
from PIL import Image, ImageDraw
from matplotlib import cm
from mm4DPS.evaluation import _load, _spectrogram, SPEC_VERSION, git_sha
from mm4DPS.scene import transform_person_pcl
from mm4DPS import DATASET_ROOT, WORKSPACE
from pathlib import Path
from skimage.metrics import structural_similarity


# ----------------------------------------------------------------------
# microdoppler_spectrograms: Micro-Doppler spectrograms at the person's range, measured vs rendered.
# ----------------------------------------------------------------------


ROOT = Path(__file__).resolve().parent

# The RadProposer tensor stacks the radars in dataset order; the spectrogram
# reads its radar's row by that index.



def bell_weights(half_width: int, shape: str = "hann") -> np.ndarray:
    """Symmetric weights across the range window, centre weighted most."""
    if half_width < 0:
        raise ValueError("half_width must be nonnegative")
    size = 2 * half_width + 1
    if size == 1:
        return np.ones(1, dtype=np.float64)
    if shape == "hann":
        weights = np.hanning(size + 2)[1:-1]
    elif shape == "gaussian":
        offsets = np.arange(size, dtype=np.float64) - half_width
        weights = np.exp(-0.5 * (offsets / (half_width / 2.0 + 1e-9)) ** 2)
    elif shape == "rect":
        weights = np.ones(size, dtype=np.float64)
    else:
        raise ValueError("shape must be hann, gaussian or rect")
    total = weights.sum()
    if total <= 0:
        raise ValueError("weights must have positive sum")
    return weights / total



def person_range_bins(
    sequence_dir: Path, frame_count: int, percentile: float,
    depth_scale: float, translation_y_m: float,
) -> np.ndarray:
    """Per-frame person range bin, taken from the person point cloud itself.

    The obvious source, ``corrected_cloud_range_m`` in the motion-detection
    record, is wrong for this purpose: it reads about 3.09 m while the person's
    radar return -- isolated by differencing motion against still frames --
    peaks at 2.49 m in *both* the measurement and the render. Centring a
    seven-bin window on the track therefore samples 0.6 m away from the person,
    which reads as a 20 dB deficit against a measurement whose noise fills that
    gap while a noise-free render leaves it empty.

    A low percentile of the cloud's own range is used instead: the body surface
    facing the radar dominates the return, and p25 lands within about 0.05 m of
    the measured peak on every frame checked.
    """
    spacing = range_bin_spacing_m()
    ranges = np.empty(frame_count, dtype=np.float64)
    for frame in range(frame_count):
        cloud = np.load(
            sequence_dir / "person_frames" / f"frame_{frame:03d}.npz"
        )["pcl"]
        points = transform_person_pcl(cloud, depth_scale, translation_y_m)[:, :3]
        ranges[frame] = float(np.percentile(
            np.linalg.norm(points, axis=1), percentile,
        ))
    return np.clip(np.round(ranges / spacing).astype(np.int64), 0, ADC_SAMPLES - 1)



def spectrogram(
    tensor: np.ndarray,
    radar: str,
    centre_bins: np.ndarray,
    half_width: int,
    weight_shape: str,
    combination: str = "power",
) -> np.ndarray:
    """Return (Doppler, frame) linear power.

    Normalization is deferred to the plotting stage so that the same stored
    array can be shown against its own peak or, more usefully, against the
    background level measured while the person is at rest.
    """
    weights = bell_weights(half_width, weight_shape)
    offsets = np.arange(-half_width, half_width + 1)
    radar_index = current_dataset().names.index(radar)
    frames = len(centre_bins)
    out = np.empty((TDM_LOOPS, frames), dtype=np.float64)
    for frame in range(frames):
        # The official tensor stores range fftshifted.
        bins = (centre_bins[frame] + offsets + ADC_SAMPLES // 2) % ADC_SAMPLES
        window = np.asarray(tensor[frame, radar_index][..., bins])
        if combination == "coherent":
            cell = (window * weights).sum(axis=-1)      # (Doppler, TX, RX)
            out[:, frame] = (np.abs(cell) ** 2).sum(axis=(1, 2))
        else:
            # Power combination. The person spans several range bins whose
            # phases differ, so summing them coherently cancels a
            # target-dependent fraction of the return -- with this geometry it
            # suppresses the rendered person about 24 dB more than the measured
            # one and invents a deficit that is not in the data.
            out[:, frame] = (
                (np.abs(window) ** 2) * weights
            ).sum(axis=-1).sum(axis=(1, 2))
    return out



class MeasuredCube:
    """The measured cube the way `CubeTensor` serves it, decoded from the archive.

    The renderer no longer stores ``<radar>_measured.npy`` (the recording
    alone, 7.5 GB on the cascade, identical run to run): a frame is decoded
    from the archive and range-windowed as the renderer's measured branch
    does (`data.range_profiles_all_channels_torch`), then takes the same
    Doppler transform, range shift and constant. Frames are read in order
    (the spectrogram walks them), so the archive is one sequential pass.
    """

    def __init__(self, sequence: int, frame_count: int | None = None):
        from mm4DPS.data import MeasuredReader
        dataset = current_dataset()
        self.readers = [MeasuredReader(dataset, sequence, radar) for radar in dataset.radars]
        frames = min(reader.frames for reader in self.readers)
        if frame_count is not None:
            frames = min(frames, int(frame_count))
        self.shape = (int(frames), len(self.readers))

    def __getitem__(self, key):
        from mm4DPS.data import product_to_doppler, range_profiles_all_channels_torch
        frame, radar_index = key
        reader = self.readers[radar_index]
        adc = reader.batch(int(frame), 1, "cpu")[0]                # (rx, chirp, sample)
        profiles = range_profiles_all_channels_torch(adc, radar=reader.radar)  # (loop, tx, rx, range)
        cube = product_to_doppler(profiles.numpy())
        cube = np.fft.fftshift(cube, axes=-1)
        return cube / float(np.prod(cube.shape))


class CubeTensor:
    """The shipped slow-time cubes read the way the RadProposer tensor is.

    HuPR's spectrograms read HuPR's published RadProposer tensor: slow-time
    mean removed, forward-normalised shifted FFTs over Doppler and range,
    ``(frame, radar, doppler, tx, rx, range)``. The shipped product is the
    per-loop range profiles, so ``tensor[frame, radar_index]`` takes the
    pipeline's Doppler transform (`data.product_to_doppler`: mean removal,
    the Doppler window, FFT, fftshift), shifts the range axis and applies
    the same constant, and the spectrogram code above is unchanged. A
    dataset with no RadProposer tensor (the cascade) goes through this.
    """

    def __init__(self, product_dir: Path, side: str):
        from mm4DPS.data import ProductCube
        names = current_dataset().names
        self.cubes = [ProductCube(product_dir / f"{name}_{side}.npy") for name in names]
        self.shape = (int(self.cubes[0].shape[0]), len(names))

    def __getitem__(self, key):
        from mm4DPS.data import product_to_doppler
        frame, radar_index = key
        cube = self.cubes[radar_index][frame]                      # (loop, tx, rx, range) complex64
        cube = product_to_doppler(cube)                            # (doppler, tx, rx, range), mean removed
        cube = np.fft.fftshift(cube, axes=-1)                      # the official range shift
        return cube / float(np.prod(cube.shape))                   # norm="forward" over every axis


def motion_runs(pose_audit: dict, frame_count: int) -> list[tuple[int, int]]:
    frames = np.array(sorted(int(f) for f in pose_audit["pose_motion_frames"]))
    frames = frames[frames < frame_count]
    if frames.size == 0:
        return []
    splits = np.split(frames, np.where(np.diff(frames) > 1)[0] + 1)
    return [(int(run[0]), int(run[-1])) for run in splits]



def run(args: argparse.Namespace) -> dict:
    motion = json.loads(args.motion_detection.read_text())
    pose = json.loads(args.pose_audit.read_text())
    if (args.product_dir / "measured_radproposer_complex.npy").is_file():
        measured = np.load(
            args.product_dir / "measured_radproposer_complex.npy", mmap_mode="r",
        )
        rendered = np.load(
            args.product_dir / "rendered_radproposer_complex.npy", mmap_mode="r",
        )
        tensor_source = "RadProposer tensor"
    else:
        rendered = CubeTensor(args.product_dir, "rendered")
        if (args.product_dir / f"{current_dataset().names[0]}_measured.npy").is_file():
            measured = CubeTensor(args.product_dir, "measured")
            tensor_source = "the shipped range-Doppler cubes, DC zeroed, range shifted, forward-normalised"
        else:
            render_result = json.loads((args.product_dir / "render_result.json").read_text())
            measured = MeasuredCube(int(render_result["dataset_sequence"]), rendered.shape[0])
            tensor_source = ("the rendered range-Doppler cube and the recording decoded from the "
                             "archive, DC zeroed, range shifted, forward-normalised")
    frame_count = int(measured.shape[0])
    if args.frame_count is not None:
        frame_count = min(frame_count, args.frame_count)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "status": "completed",
        "product_dir": str(args.product_dir.resolve()),
        "label": args.label,
        "frames": frame_count,
        "method": {
            "range_window_bins": 2 * args.half_width + 1,
            "person_range_source": (
                f"p{args.person_range_percentile:g} of the person cloud's own "
                "range, not the motion-detection track"
            ),
            "range_weighting": args.weight_shape,
            "range_combination": args.range_combination,
            "channel_combination": "power sum over padded-TX and RX",
            "doppler": "official forward-normalized shifted FFT over 64 loops",
            "clutter_removal": "slow-time mean, from the official preprocessing",
            "tensor_source": tensor_source,
            "commutation_note": (
                "range weighting is applied after the Doppler transform; the "
                "transform is linear so this equals weighting the loops first"
            ),
        },
        "motion_runs": motion_runs(pose, frame_count),
        "radars": {},
    }
    for radar in current_dataset().names:
        bins = person_range_bins(
            args.sequence_dir, frame_count, args.person_range_percentile,
            args.depth_scale, args.translation_y_m,
        )
        for source, tensor in (("measured", measured), ("rendered", rendered)):
            image = spectrogram(
                tensor, radar, bins, args.half_width, args.weight_shape,
                args.range_combination,
            )
            np.save(
                args.output_dir / f"{radar}_{source}_microdoppler_power.npy",
                image,
            )
        np.save(args.output_dir / f"{radar}_person_range_bin.npy", bins)
        result["radars"][radar] = {
            "person_range_bin_median": int(np.median(bins)),
            "person_range_m_median": float(
                np.median(bins) * range_bin_spacing_m()
            ),
        }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result



def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="rendered")
    parser.add_argument("--half-width", type=int, default=5)
    sequence_dir(parser)
    parser.add_argument("--person-range-percentile", type=float, default=25.0)
    parser.add_argument(
        "--range-combination", choices=("power", "coherent"), default="power",
    )
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--translation-y-m", type=float, default=-0.7282782486)
    parser.add_argument(
        "--weight-shape", choices=("hann", "gaussian", "rect"), default="hann",
    )
    parser.add_argument("--motion-detection", type=Path, required=True,
                        help="this sequence's `motion detect-person-motion` result")
    pose_audit(parser)
    parser.add_argument("--frame-count", type=int, default=None)
    return parser


def _entry_microdoppler_spectrograms():
        print(json.dumps(run(make_parser().parse_args()), indent=2)[:800])

# ----------------------------------------------------------------------
# media_metrics_report: Quantitative measured-vs-rendered metrics for the comparison media.
# ----------------------------------------------------------------------


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator > 0 else float("nan")












def fmt(metrics: dict, keys: tuple) -> str:
    parts = []
    for key in keys:
        value = metrics[key]
        parts.append(f"{value:.4f}" if abs(value) < 1000 else f"{value:.3e}")
    return " | ".join(parts)






# ----------------------------------------------------------------------
# velocity_comparison_media: Side-by-side comparison media for the learnable-velocity experiments.
# ----------------------------------------------------------------------




def to_unit(values: np.ndarray, reference: float) -> np.ndarray:
    """Linear magnitude to [0,1] display units: value / measured reference."""
    return np.clip(values / reference, 0.0, 1.0)



def colorize(unit: np.ndarray) -> np.ndarray:
    return (cm.viridis(unit)[..., :3] * 255).astype(np.uint8)



def panel_video(
    output: Path,
    columns: list[tuple[str, np.ndarray]],
    fps: float,
    scale: int,
    title: str,
) -> None:
    """Encode (frames, rows, cols) magnitude stacks side by side into mp4."""
    frames = min(len(stack) for _, stack in columns)
    reference = float(np.percentile(np.asarray(columns[0][1][: min(frames, 100)]), 99.9))
    header = 22
    gap = 4
    rows = columns[0][1].shape[1]
    cols = columns[0][1].shape[2]
    width = (cols * scale + gap) * len(columns) - gap
    height = rows * scale + header
    encoder = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(output),
    ], stdin=subprocess.PIPE)
    assert encoder.stdin is not None
    for frame in range(frames):
        canvas = Image.new("RGB", (width, height), (12, 12, 12))
        draw = ImageDraw.Draw(canvas)
        for index, (label, stack) in enumerate(columns):
            unit = to_unit(np.flipud(np.asarray(stack[frame])), reference)
            tile = Image.fromarray(colorize(unit)).resize(
                (cols * scale, rows * scale), Image.NEAREST,
            )
            x = index * (cols * scale + gap)
            canvas.paste(tile, (x, header))
            draw.text((x + 2, 4), label, fill=(235, 235, 235))
        draw.text((width - 150, 4), f"{title}  f{frame:03d}", fill=(160, 160, 160))
        encoder.stdin.write(np.asarray(canvas, dtype=np.uint8).tobytes())
    encoder.stdin.close()
    encoder.wait()
    if encoder.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {output}")
    print(f"wrote {output} ({frames} frames, {len(columns)} columns)")



def spectrogram_figure(
    output: Path,
    rows: list[tuple[str, np.ndarray]],
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = float(np.percentile(rows[0][1], 99.9))
    figure, axes = plt.subplots(
        len(rows), 1, figsize=(12.0, 1.9 * len(rows)), dpi=150, sharex=True,
    )
    for axis, (label, power) in zip(np.atleast_1d(axes), rows):
        magnitude = np.sqrt(np.maximum(power, 0.0))
        unit = np.clip(magnitude / np.sqrt(reference), 0.0, 1.0)
        # The clutter-removed DC row is identically zero; drop it so the
        # figure has no black bar, matching the house convention.
        keep = ~np.all(power == 0.0, axis=1)
        axis.imshow(
            unit[keep], aspect="auto", origin="lower", cmap="viridis",
            vmin=0.0, vmax=1.0,
        )
        axis.set_ylabel(label, fontsize=8)
        axis.tick_params(labelsize=7)
    np.atleast_1d(axes)[-1].set_xlabel("frame", fontsize=8)
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)
    print(f"wrote {output}")



def main_velocity_comparison_media() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--sequence", type=int, default=38)
    parser.add_argument(
        "--runs", nargs="+", required=True,
        help="label=workspace pairs, e.g. baseline=baseline_38",
    )
    parser.add_argument("--render-name", default=None,
                        help="render directory name; default seq<S>_v3_600")
    parser.add_argument("--spectrogram-variant", default="hw10_v3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--scale", type=int, default=2)
    physics_fingerprint.add_varying_argument(parser)
    args = parser.parse_args()
    render_name = args.render_name or f"seq{args.sequence}_v3_600"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = [pair.split("=", 1) for pair in args.runs]
    # R097: products of two chain profiles are indistinguishable from the
    # outside and are not interchangeable in an A/B -- the profiles run
    # different amounts of stochastic training. Refuse before rendering
    # anything, so the failure costs seconds rather than a comparison.
    chain_profile = require_same_profile(
        [args.outputs_root / workspace for _, workspace in runs],
        "velocity-comparison-media")
    # R099: and the FORWARD MODEL, not just the chain profile. The products of
    # a mask A/B whose flag never reached the render are perfectly consistent
    # with each other -- both `enabled: false` -- so a guard that only catches
    # unintended differences passes them. `--varying` makes the comparison
    # declare its variable, and the declared key must actually differ.
    physics_fingerprint.require_comparable(
        [args.outputs_root / workspace / "rendered" / render_name
         for _, workspace in runs],
        "velocity-comparison-media", getattr(args, "varying", ()))
    first_workspace = args.outputs_root / runs[0][1]
    for radar in current_dataset().names:
        render_dir = first_workspace / "rendered" / render_name
        measured_rd = np.load(render_dir / f"{radar}_measured_rd.npy", mmap_mode="r")
        measured_ra = np.abs(np.load(
            render_dir / f"{radar}_measured_first_loop_ra_complex.npy", mmap_mode="r",
        ))
        rd_columns = [("measured", measured_rd)]
        ra_columns = [("measured", measured_ra)]
        for label, workspace in runs:
            run_render = args.outputs_root / workspace / "rendered" / render_name
            rd_columns.append((label, np.load(
                run_render / f"{radar}_rendered_rd.npy", mmap_mode="r",
            )))
            ra_columns.append((label, np.abs(np.load(
                run_render / f"{radar}_rendered_first_loop_ra_complex.npy",
                mmap_mode="r",
            ))))
        panel_video(
            args.output_dir / f"seq{args.sequence}_{radar}_rd.mp4",
            rd_columns, args.fps, args.scale,
            f"{radar} RD",
        )
        panel_video(
            args.output_dir / f"seq{args.sequence}_{radar}_ra.mp4",
            [
                (label, np.transpose(stack, (0, 2, 1)))
                for label, stack in ra_columns
            ],
            args.fps, args.scale,
            f"{radar} RA",
        )
        md_rows = [(
            "measured",
            np.load(
                first_workspace / "deliverables" / f"microdoppler_seq{args.sequence}"
                / args.spectrogram_variant / f"{radar}_measured_microdoppler_power.npy"
            ),
        )]
        for label, workspace in runs:
            md_rows.append((label, np.load(
                args.outputs_root / workspace / "deliverables"
                / f"microdoppler_seq{args.sequence}" / args.spectrogram_variant
                / f"{radar}_rendered_microdoppler_power.npy"
            )))
        spectrogram_figure(
            args.output_dir / f"seq{args.sequence}_{radar}_microdoppler.png",
            md_rows,
            f"sequence {args.sequence} {radar} micro-Doppler ({args.spectrogram_variant})",
        )


def _entry_velocity_comparison_media():
        main_velocity_comparison_media()

# ----------------------------------------------------------------------
# permutation_and_sweep: Validation checks 3 and 4 of the evaluation specification, on real products.
# ----------------------------------------------------------------------


SEQUENCES = ("38", "102", "256")

P_FA_SWEEP = (1e-2, 1e-3, 1e-4)

CORRELATIONS = ("centroid_r", "spread_r", "spectrogram_r", "energy_r")



def product_dir(root: Path, seq: str) -> Path:
    return root / f"vf_gvhmr_{seq}" / "rendered" / f"seq{seq}_final_600"



def spectrogram_dir(root: Path, seq: str) -> Path:
    """The PERSON-RANGE-GATED spectrogram, which is the reduction the pipeline
    and the paper use. The first version of this module passed no spectrogram
    directory and so silently scored the ungated motion_dra sum instead --
    a different physical quantity, disagreeing by up to 0.26 in centroid
    correlation. `mm4DPS.evaluate` now refuses that fallback, which is how the
    inconsistency surfaced."""
    return (root / f"vf_gvhmr_{seq}" / "deliverables" /
            f"microdoppler_seq{seq}" / "hw10_v2")



















# ----------------------------------------------------------------------
# occlusion_check_media: Show the scored wall region and the occlusion trace, over the source RA video.
# ----------------------------------------------------------------------
# A communication artefact, not an analysis. It COMPOSES the existing
# `artifacts_doppler_support` RA videos -- frame i in, frame i out, no
# resampling -- and draws on them the exact region whose power is being
# tracked, so "are we measuring the right part of the RA map?" can be answered
# by looking.
#
# THE REGION IS SHAPED LIKE THE WALL, which an earlier version of this tool got
# backwards. Measured on all six products, the wall ridge is ONE range bin wide
# at half power (4.3 cm) and SEVEN OR EIGHT azimuth bins wide -- it is a
# near-vertical line in an (x=range, y=azimuth) RA map. A region 26 range bins
# long and one azimuth bin tall lies across that line rather than along it: it
# oversamples range, undersamples azimuth, and captured only 3.8-8.5% of the
# wall's energy. The region below is derived from each product's own measured
# mean map and captures 13-47% with a similar cell count.

# occlusion-check-media is HuPR's wall-band check: the captions and the
# azimuth/elevation reading are HuPR's two boards.

# The source videos' geometry, asserted against ffprobe before anything is
# drawn. `panel_video` built them as two columns of a (frames, 64, 256)
# azimuth-by-range stack at scale 2 with a 22 px header and a 4 px gap, and it
# calls `np.flipud`, so angle bin 0 is at the BOTTOM of each panel.
SOURCE_WIDTH, SOURCE_HEIGHT, SOURCE_FRAMES, SOURCE_FPS = 1028, 150, 600, 10.0
SOURCE_HEADER, SOURCE_GAP, SOURCE_SCALE = 22, 4, 2
ANGLE_BINS, RANGE_BINS = 64, 256
UPSCALE = 2                      # 1028x150 -> 2056x300, nearest neighbour
PLOT_HEIGHT, INFO_HEIGHT = 250, 70

SEARCH_BAND = (120, 190)         # 5.2-8.2 m: where the back wall ridge lives
RANGE_HALF_WIDTH = 1             # ridge peak +/- this; the ridge is 1 bin at
                                 # half power, +/-1 absorbs frame-to-frame wobble
ANGLE_CONTOUR = 0.10             # angular window = the ridge's 10%-of-peak
                                 # extent. Two reasons, and they agree: it
                                 # captures 83-88% of the ridge's energy where
                                 # the half-power window captures only 69-71%,
                                 # and at 12-13 bins it is at least as wide as
                                 # the person's own silhouette (9-14 bins,
                                 # median). A window narrower than the shadow
                                 # saturates -- at half power the occluded
                                 # share sat at 1.0 for 17% of seq 256 and 47%
                                 # of seq 102, which throws away exactly the
                                 # gradation the trace exists to show.
DESPECKLE = 3                    # centred median, kills single-frame nulls
CLEAR_QUANTILE = 0.75            # the wall's "clear" level, from its own trace
OCCLUDE_ENTER, OCCLUDE_LEAVE = 0.50, 0.70   # fractions of that level
MIN_RUN_FRAMES = 5               # 0.5 s; below this it is speckle, not an event
SNAP_FRAMES = 12                 # how far back to look for the true crossing
POSE_NULL_SHIFTS = range(30, 600, 10)   # shifted-pose null for the check
FONT_PATH = (Path(cm.__file__).resolve().parent
             / "mpl-data/fonts/ttf/DejaVuSans.ttf")


def probe_source(path: Path) -> dict:
    """ffprobe the source and return its stream properties."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames", "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    stream = json.loads(out.stdout)["streams"][0]
    numerator, denominator = stream["r_frame_rate"].split("/")
    return {"width": int(stream["width"]), "height": int(stream["height"]),
            "frames": int(stream["nb_frames"]),
            "fps": float(numerator) / float(denominator)}


def assert_source(path: Path) -> dict:
    found = probe_source(path)
    expected = {"width": SOURCE_WIDTH, "height": SOURCE_HEIGHT,
                "frames": SOURCE_FRAMES, "fps": SOURCE_FPS}
    wrong = {k: (found[k], v) for k, v in expected.items() if found[k] != v}
    if wrong:
        raise ValueError(
            f"{path.name} is not the expected source video: "
            + ", ".join(f"{k} is {a}, expected {b}" for k, (a, b) in wrong.items())
            + ". The overlay geometry is derived from these properties, so a "
              "differing file must not be composed against it.")
    return found


def read_frames(path: Path):
    """Yield the source's frames as (H, W, 3) uint8, in order, no resampling."""
    stride = SOURCE_WIDTH * SOURCE_HEIGHT * 3
    process = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    assert process.stdout is not None
    try:
        while True:
            raw = process.stdout.read(stride)
            if len(raw) < stride:
                break
            yield np.frombuffer(raw, np.uint8).reshape(
                SOURCE_HEIGHT, SOURCE_WIDTH, 3)
    finally:
        process.stdout.close()
        process.wait()


def wall_region(mean_power: np.ndarray) -> dict:
    """Locate the wall ridge in this product's own measured mean RA map.

    Range: the peak of the range profile summed over azimuth, +/- one bin.
    Angle: the extent of the angle cut at that range down to ANGLE_CONTOUR of
    its peak -- wide enough to hold the person's whole shadow, so the occluded
    share is a fraction rather than a saturated 1.
    """
    low, high = SEARCH_BAND
    band = mean_power[low:high]
    peak = low + int(band.sum(1).argmax())
    cut = mean_power[peak]
    inside = np.nonzero(cut >= ANGLE_CONTOUR * cut.max())[0]
    return {"range": (peak - RANGE_HALF_WIDTH, peak + RANGE_HALF_WIDTH),
            "peak_range_bin": peak,
            "azimuth": (int(inside.min()), int(inside.max()))}


def region_boxes(region: dict) -> list[tuple]:
    """Final-canvas boxes for the region, one per panel, as (x0,y0,x1,y1)."""
    panel = (RANGE_BINS * SOURCE_SCALE + SOURCE_GAP) * UPSCALE
    cell = SOURCE_SCALE * UPSCALE
    first, last = region["range"]
    low, high = region["azimuth"]
    # flipud: angle bin 0 is at the bottom, so the HIGH bin gives the top edge.
    top = (SOURCE_HEADER * UPSCALE) + (ANGLE_BINS - 1 - high) * cell
    bottom = (SOURCE_HEADER * UPSCALE) + (ANGLE_BINS - low) * cell
    return [(column * panel + first * cell, top,
             column * panel + (last + 1) * cell, bottom)
            for column in range(2)]


def occluded_fraction(covered: np.ndarray, mean_power: np.ndarray,
                      region: dict) -> np.ndarray:
    """Per frame, the share of the region's measured power behind the person.

    `covered` is `mm4DPS.qa.occupancy` -- the same function the
    Blocker C analysis uses -- giving, per frame and angle bin, whether the
    person's own silhouette spans that bin.

    The weight is the region's measured power per azimuth bin. That is
    deliberate: the measured RA map ALREADY contains the array's angular
    response, so weighting by it uses the measurement's own angular
    distribution. Convolving a second time with a modelled beam pattern
    double-counts it -- tried, and it both shrinks the signal to ~1% and
    weakens the anti-correlation on seq 256 horizontal from -0.44 to -0.05,
    because the Dirichlet sidelobes span the whole field and dilute the share.
    """
    first, last = region["range"]
    low, high = region["azimuth"]
    weight = mean_power[first:last + 1, low:high + 1].sum(0)
    weight = weight / weight.sum()
    return (covered[:, low:high + 1] * weight[None, :]).sum(1)


def occlusion_state(magnitude: np.ndarray) -> tuple[np.ndarray, float]:
    """Binary occluded/clear, from the MEASURED trace alone. 1 = occluded.

    Both traces come from ground truth and nothing else: the same function
    applied to a RENDERED trace gives the rendered red, and the two can then
    be compared. The person's pose is deliberately NOT an input -- it is kept
    as an independent check, reported in the PROVENANCE.

    The wall's "clear" level is an upper quantile of its own trace, and
    occlusion is a substantial fractional drop below it. A fixed global
    threshold does not work: seq 38 sits at 0.05 when occluded and at two
    different clear levels, 0.45 and 0.85, so an Otsu split lands between the
    two clear levels and calls the lower one occluded. A fraction of the clear
    level separates them correctly.

    Two details matter for the events being usable:

    * Short events are kept. seq 102 has occlusions lasting 1.0-1.6 s in its
      first 12 s; anything that suppresses events closer together than a
      couple of seconds loses them outright.
    * Transitions carry no lag. The hysteresis fires when the signal reaches
      the FAR threshold, which is part-way through the change, so each
      transition is snapped back to where the trace crossed the midpoint --
      on seq 38 that puts the first reveal at frame 118, the first elevated
      sample, rather than several frames after it.
    """
    pad = DESPECKLE // 2
    smoothed = np.median(np.lib.stride_tricks.sliding_window_view(
        np.pad(magnitude, (pad, pad), mode="edge"), DESPECKLE), axis=-1)
    reference = float(np.quantile(smoothed, CLEAR_QUANTILE))
    low, high = OCCLUDE_ENTER * reference, OCCLUDE_LEAVE * reference
    middle = 0.5 * (OCCLUDE_ENTER + OCCLUDE_LEAVE) * reference

    state = bool(smoothed[0] < low)
    out = np.empty(len(smoothed), dtype=bool)
    out[0] = state
    for index in range(1, len(smoothed)):
        if state and smoothed[index] > high:
            state = False
        elif not state and smoothed[index] < low:
            state = True
        out[index] = state

    def absorb(series: np.ndarray) -> np.ndarray:
        start = 0
        while start < len(series):
            stop = start
            while stop + 1 < len(series) and series[stop + 1] == series[start]:
                stop += 1
            if stop - start + 1 < MIN_RUN_FRAMES and start > 0:
                series[start:stop + 1] = series[start - 1]
            start = stop + 1
        return series

    out = absorb(out)
    for edge in np.nonzero(np.diff(out.astype(int)))[0] + 1:
        going_low = out[edge]
        back = edge
        while back - 1 >= max(0, edge - SNAP_FRAMES):
            previous = smoothed[back - 1]
            if (going_low and previous > middle) or (
                    not going_low and previous < middle):
                break
            back -= 1
        if back < edge:
            out[back:edge] = out[edge]
    # snapping can recreate short runs, so absorb once more
    return absorb(out), reference


def occlusion_plot(fraction: np.ndarray, state: np.ndarray, trace: np.ndarray,
                   maximum: float, label: str, width: int,
                   banner: str = "") -> tuple:
    """The traces, rendered once, with the pixel span of the DATA area.

    Returns `(image, x_first, x_last)`. The caller needs those two x values:
    matplotlib draws the traces inside the axes, which is inset by its margins
    (here x = 60.3 .. 2047.7 of 2056), so a playhead swept across the full
    image width does not line up with the curve underneath it. That offset was
    reading as a 1.4-1.7 s lag in the red signal -- seq 38's reveal is
    detected at frame 118 but the playhead only reached the drawn step at
    frame 132, and seq 102's 3.3 s event appeared to fire at 4.9 s.
    """
    import matplotlib.pyplot as plt

    frames = len(fraction)
    figure, axis = plt.subplots(
        figsize=(width / 100.0, PLOT_HEIGHT / 100.0), dpi=100)
    figure.patch.set_facecolor("#0c0c0c")
    axis.set_facecolor("#0c0c0c")
    index = np.arange(frames)
    edges = np.nonzero(np.diff(state.astype(int)))[0] + 1
    first = True
    for start, stop in zip(np.r_[0, edges], np.r_[edges, frames]):
        if not state[start]:
            # shade CLEAR, not occluded: on every product the person covers the
            # window most of the time, so the clear intervals are the rare ones
            # and they are where the wall should brighten.
            axis.axvspan(start, stop - 1, color="#4fc3f7", alpha=0.16, lw=0,
                         label="CLEAR (debounced state)" if first else None)
            first = False
    axis.step(index, state.astype(float), where="post", color="#ff5c5c",
              linewidth=2.6,
              label=f"OCCLUSION (binary): 1 = {label} occluded by the person, "
                    f"0 = clear")
    axis.plot(index, trace, color="#4fc3f7", linewidth=1.3,
              label=f"measured region mean |RA|, linear magnitude, "
                    f"normalised by its own max {maximum:.4e}")
    for edge in edges:
        axis.axvline(edge, color="#ffd479", lw=0.9, ls="--", alpha=0.8)
    axis.set_xlim(0, frames - 1)
    axis.set_ylim(-0.05, 1.42)
    axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.set_xlabel("frame index", color="#cccccc", fontsize=9)
    axis.set_ylabel("normalised", color="#cccccc", fontsize=9)
    seconds = axis.secondary_xaxis(
        "top", functions=(lambda v: v / SOURCE_FPS, lambda v: v * SOURCE_FPS))
    seconds.set_xlabel("seconds", color="#cccccc", fontsize=9)
    for spine in list(axis.spines.values()) + list(seconds.spines.values()):
        spine.set_color("#555555")
    axis.tick_params(colors="#cccccc", labelsize=8)
    seconds.tick_params(colors="#cccccc", labelsize=8)
    axis.grid(alpha=0.18, color="#888888")
    axis.legend(loc="upper left", fontsize=8.5, framealpha=0.9, ncol=3,
                facecolor="#1a1a1a", edgecolor="#555555", labelcolor="#e8e8e8")
    if banner:
        axis.text(0.998, 1.30, banner, transform=axis.transAxes, ha="right",
                  va="top", fontsize=11, color="#ffd479", weight="bold")
    figure.tight_layout(pad=0.6)
    figure.canvas.draw()
    image = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
    plt.close(figure)
    x_first = float(axis.transData.transform((0, 0))[0])
    x_last = float(axis.transData.transform((frames - 1, 0))[0])
    if image.shape[0] != PLOT_HEIGHT or image.shape[1] != width:
        scale = width / image.shape[1]
        x_first, x_last = x_first * scale, x_last * scale
        image = np.asarray(Image.fromarray(image).resize(
            (width, PLOT_HEIGHT), Image.BILINEAR))
    return image, x_first, x_last


def run_occlusion_check_media(args: argparse.Namespace) -> dict:
    from mm4DPS.qa import occupancy
    from mm4DPS import array_axis
    from PIL import ImageFont

    args.output_dir.mkdir(parents=True, exist_ok=True)
    radars = current_dataset().names
    configs = {name: json.loads(getattr(args, f"{name}_config").read_text())
               for name in radars}
    font = ImageFont.truetype(str(FONT_PATH), 17)
    small = ImageFont.truetype(str(FONT_PATH), 14)
    spacing = range_bin_spacing_m()
    width = SOURCE_WIDTH * UPSCALE
    top_height = SOURCE_HEIGHT * UPSCALE
    rows = 1 + int(args.with_rendered) + int(args.mask_on_dir is not None)
    height = INFO_HEIGHT + top_height + PLOT_HEIGHT * rows
    report: dict = {"videos": [], "sources": {}}

    for sequence in args.sequences:
        workspace = args.outputs_root / f"{args.workspace_prefix}{sequence}"
        _ = require_same_profile([workspace], "occlusion-check-media")
        render = workspace / "rendered" / f"seq{sequence}_{args.render_suffix}"
        candidates = [d for d in workspace.glob(f"person_sequence_{sequence}_*")
                      if (d / "person_frames").is_dir()]
        if len(candidates) != 1:
            raise ValueError(f"expected one person sequence in {workspace}, "
                             f"found {[d.name for d in candidates]}")
        translation = float(json.loads(
            (render / "render_result.json").read_text())["geometry_translation_y_m"])

        for radar in radars:
            source = args.source_dir / f"seq{sequence}_{radar}_ra.mp4"
            report["sources"][source.name] = assert_source(source)

            cube = np.load(
                render / f"{radar}_measured_first_loop_ra_complex.npy")
            mean_power = (np.abs(cube).astype(np.float64) ** 2).mean(0)
            region = wall_region(mean_power)
            first, last = region["range"]
            low, high = region["azimuth"]

            # Linear magnitude, matching the RA panel above it, which is
            # np.abs of the same cube. Never dB.
            magnitude = np.abs(cube)[:, first:last + 1, low:high + 1].mean((1, 2))
            maximum = float(magnitude.max())
            trace = magnitude / maximum

            covered = occupancy(
                candidates[0], SOURCE_FRAMES, translation,
                config_centre_m(configs[radar]),
                np.asarray(array_axis(configs[radar]), float), radar)
            fraction = occluded_fraction(covered, mean_power, region)
            state, clear_level = occlusion_state(magnitude)
            edges = np.nonzero(np.diff(state.astype(int)))[0] + 1
            # The pose is now an INDEPENDENT check, not an input. Raw
            # agreement is not interpretable on its own -- where the binarised
            # geometry is constant it has no null spread at all -- so it is
            # scored against shifts of the pose against the measurement.
            agree = lambda g: float((state == (g >= 0.5)).mean())
            geometry_agreement = agree(fraction)
            geometry_null = np.array(
                [agree(np.roll(fraction, k)) for k in POSE_NULL_SHIFTS])
            spread = float(geometry_null.std(ddof=1))
            geometry_z = (None if spread < 1e-9 else
                          (geometry_agreement - float(geometry_null.mean())) / spread)

            axis_name = "azimuth" if radar == "horizontal" else "elevation"
            clear = int((~state).sum())
            caption = ""
            if radar == "vertical":
                caption = ("this array's virtual ULA lies on -Z, so its 64-bin "
                           "axis is ELEVATION, not azimuth. The window is the "
                           "wall's elevation extent and the trace is occupancy "
                           "in that coordinate.")
                if not len(edges):
                    caption += (" It is constant: a standing person spans far "
                                "more elevation than the wall window, so this "
                                "radar has no occlusion transition to show.")
            elif not len(edges):
                caption = ("the person's silhouette covers this window in all "
                           f"{SOURCE_FRAMES} frames, so there is no transition "
                           "here. That is the finding, not a bug.")

            plot, plot_x_first, plot_x_last = occlusion_plot(
                fraction, state, trace, maximum,
                f"{axis_name} bins {low}-{high}", width,
                banner="GROUND TRUTH (measured)" if args.with_rendered else "")

            extra_rows = []
            rendered_row = None
            if args.with_rendered:
                # Same region, same extraction, same detector -- the ONLY
                # difference is which cube the numbers come from, so any
                # disagreement is the data and not the code.
                rendered_cube = np.load(
                    render / f"{radar}_rendered_first_loop_ra_complex.npy")
                rendered_magnitude = np.abs(rendered_cube)[
                    :, first:last + 1, low:high + 1].mean((1, 2))
                rendered_maximum = float(rendered_magnitude.max())
                rendered_state, _ = occlusion_state(rendered_magnitude)
                rendered_plot, _, _ = occlusion_plot(
                    fraction, rendered_state,
                    rendered_magnitude / rendered_maximum, rendered_maximum,
                    f"{axis_name} bins {low}-{high}", width,
                    banner="RENDERED (from_scratch)")
                rendered_row = {
                    "plot": rendered_plot, "state": rendered_state,
                    "maximum": rendered_maximum,
                    "transitions": int(np.diff(
                        rendered_state.astype(int)).astype(bool).sum()),
                    "frames_occluded": int(rendered_state.sum()),
                    "state_agreement": float((rendered_state == state).mean()),
                    "trace_correlation": (
                        None if rendered_magnitude.std() < 1e-12 else
                        float(np.corrcoef(rendered_magnitude, magnitude)[0, 1])),
                    "trace_relative_spread": float(
                        rendered_magnitude.std() / rendered_magnitude.mean()),
                    "measured_relative_spread": float(
                        magnitude.std() / magnitude.mean()),
                }
                extra_rows.append(rendered_plot)
            if args.mask_on_dir is not None:
                # Third arm, same region / extraction / detector / plot.
                mask_cube = np.load(
                    args.mask_on_dir / f"{radar}_rendered_first_loop_ra_complex.npy")
                mask_magnitude = np.abs(mask_cube)[
                    :, first:last + 1, low:high + 1].mean((1, 2))
                mask_max = float(mask_magnitude.max())
                mask_state, _ = occlusion_state(mask_magnitude)
                mask_plot, _, _ = occlusion_plot(
                    fraction, mask_state, mask_magnitude / mask_max, mask_max,
                    f"{axis_name} bins {low}-{high}", width,
                    banner="RENDERED, per-frame visibility MASK ON")
                extra_rows.append(mask_plot)
                report.setdefault("mask_on", []).append({
                    "product": f"seq{sequence}_{radar}",
                    "transitions": int(np.diff(
                        mask_state.astype(int)).astype(bool).sum()),
                    "frames_occluded": int(mask_state.sum()),
                    "relative_spread": round(float(
                        mask_magnitude.std() / mask_magnitude.mean()), 4),
                })
            boxes = region_boxes(region)
            output = args.output_dir / f"seq{sequence}_{radar}_occlusion_check.mp4"
            encoder = subprocess.Popen([
                "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                "-r", f"{SOURCE_FPS}", "-i", "-", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-crf", "18", str(output),
            ], stdin=subprocess.PIPE)
            assert encoder.stdin is not None

            written = 0
            for index, frame in enumerate(read_frames(source)):
                canvas = np.full((height, width, 3), 12, np.uint8)
                canvas[INFO_HEIGHT:INFO_HEIGHT + top_height] = np.repeat(
                    np.repeat(frame, UPSCALE, 0), UPSCALE, 1)
                playhead = plot.copy()
                # into the AXES' data span, not the image's full width
                column = int(round(
                    plot_x_first + index / (SOURCE_FRAMES - 1)
                    * (plot_x_last - plot_x_first)))
                column = min(max(column, 0), width - 1)
                playhead[:, max(0, column - 1):column + 2] = (255, 214, 121)
                canvas[INFO_HEIGHT + top_height:
                       INFO_HEIGHT + top_height + PLOT_HEIGHT] = playhead
                for row, panel in enumerate(extra_rows, start=1):
                    strip = panel.copy()
                    strip[:, max(0, column - 1):column + 2] = (255, 214, 121)
                    top = INFO_HEIGHT + top_height + PLOT_HEIGHT * row
                    canvas[top:top + PLOT_HEIGHT] = strip
                image = Image.fromarray(canvas)
                draw = ImageDraw.Draw(image)
                for x0, y0, x1, y1 in boxes:
                    draw.rectangle(
                        [x0 - 2, INFO_HEIGHT + y0 - 2, x1 + 1, INFO_HEIGHT + y1 + 1],
                        outline=(255, 92, 92), width=2)
                label = "OCCLUDED" if bool(state[index]) else "CLEAR"
                draw.text((10, 6),
                          f"seq {sequence}  {radar}  wall region: range bins "
                          f"{first}-{last} ({first * spacing:.2f}-"
                          f"{last * spacing:.2f} m) x {axis_name} bins "
                          f"{low}-{high}  =  {(last - first + 1) * (high - low + 1)} "
                          f"cells   frame {index:03d}/{SOURCE_FRAMES - 1}  "
                          f"t={index / SOURCE_FPS:5.1f}s   {label}  "
                          f"(occluded share {fraction[index]:.2f})",
                          fill=(235, 235, 235), font=font)
                draw.text((10, 26),
                          "red box = the exact region the trace measures, on "
                          "BOTH panels (left: measured, right: "
                          f"{args.right_panel_label}). "
                          "It is narrow in range and wide in "
                          f"{axis_name} because the wall ridge is: 1 range bin "
                          "wide at half power, 12-13 angle bins to the 10% "
                          "contour.",
                          fill=(255, 92, 92), font=small)
                if caption:
                    draw.text((10, 48), caption, fill=(255, 212, 121),
                              font=small)
                encoder.stdin.write(np.asarray(image, np.uint8).tobytes())
                written += 1
            encoder.stdin.close()
            encoder.wait()
            if encoder.returncode != 0 or written != SOURCE_FRAMES:
                raise RuntimeError(
                    f"encode failed for {output} ({written} frames written)")

            correlation = (None if state.std() < 1e-9
                           else float(np.corrcoef(
                               state.astype(float), magnitude)[0, 1]))
            graded_correlation = (None if fraction.std() < 1e-9
                                  else float(np.corrcoef(
                                      fraction, magnitude)[0, 1]))
            report["videos"].append({
                "path": str(output), "sequence": sequence, "radar": radar,
                "angular_axis": axis_name,
                "range_bins": [int(first), int(last)],
                "peak_range_bin": int(region["peak_range_bin"]),
                "range_m": [round(first * spacing, 3), round(last * spacing, 3)],
                "angle_bins": [int(low), int(high)],
                "cells": int((last - first + 1) * (high - low + 1)),
                "normalisation_max_linear_magnitude": maximum,
                "frames_occluded": int(state.sum()), "frames_clear": clear,
                "transitions": int(len(edges)),
                "transition_frames": [int(e) for e in edges],
                "correlation_binary_vs_magnitude": correlation,
                "clear_level_fraction_of_max": round(clear_level, 4),
                "geometry_agreement": round(geometry_agreement, 3),
                **({} if rendered_row is None else {
                    "rendered": {k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in rendered_row.items()
                                 if k not in ("plot", "state")}}),
                "geometry_agreement_null": round(float(geometry_null.mean()), 3),
                "geometry_agreement_z": (None if geometry_z is None
                                         else round(geometry_z, 2)),
                "correlation_graded_share_vs_magnitude": graded_correlation,
                "occluded_share_min_max": [round(float(fraction.min()), 3),
                                           round(float(fraction.max()), 3)],
            })
            print(f"wrote {output}  {(last - first + 1) * (high - low + 1)} cells  "
                  f"clear={clear}/600  transitions={len(edges)}", flush=True)
    write_occlusion_provenance(args, report)
    return report


def write_occlusion_provenance(args: argparse.Namespace, report: dict) -> None:
    """The folder's own record, in the shape the other artifact folders use."""
    import hashlib

    lines = [
        "artifacts_occlusion_check -- do we agree on WHERE the wall power is",
        "measured and WHEN the person is occluding it?",
        "",
        "A communication artefact. Every frame is composed from an existing RA",
        "video: frame i in is frame i out, no resampling, no interpolation, and",
        "no render was run. Source pixels are unmodified under the overlay.",
        "",
        "Built by: python -m mm4DPS.evaluation.media occlusion-check-media",
        f"Commit:   {git_sha()}",
        "",
        "SOURCES (composed from, never written to):",
    ]
    for name, properties in sorted(report["sources"].items()):
        digest = hashlib.sha256((args.source_dir / name).read_bytes()).hexdigest()
        lines += [f"  {name}",
                  f"    {properties['width']}x{properties['height']}, "
                  f"{properties['fps']:g} fps, {properties['frames']} frames",
                  f"    sha256 {digest}"]
    lines += [
        "",
        "Measured cubes for the region, the trace and the weights, per seq S:",
        f"  {args.outputs_root}/{args.workspace_prefix}<S>/rendered/"
        f"seq<S>_{args.render_suffix}/<radar>_measured_first_loop_ra_complex.npy",
        "  -- the same workspaces the source videos were built from.",
        "",
        "REGION -- shaped like the wall, derived per product from its own",
        "measured mean RA map, not fixed by hand:",
        f"  range   the peak of the range profile summed over angle within "
        f"{SEARCH_BAND[0]}-{SEARCH_BAND[1]} (5.2-8.2 m), +/- {RANGE_HALF_WIDTH} bin.",
        f"  angle   the angle cut at that range, out to {ANGLE_CONTOUR:.0%} of "
        f"its peak.",
        "  The angle window is set to the 10% contour rather than half power",
        "  for two reasons that agree: it captures 83-88% of the ridge's energy",
        "  against 69-71%, and at 12-13 bins it is at least as wide as the",
        "  person's own silhouette (9-14 bins, median), so the occluded share",
        "  is a fraction rather than a saturated 1. At half power it saturated",
        "  for 17% of seq 256 and 47% of seq 102 frames.",
        "  Measured on all six, the ridge is ONE range bin wide at half power",
        "  (4.3 cm) and SEVEN OR EIGHT angle bins wide. An earlier version of",
        "  this tool used 26 range bins x 1 angle bin, which lies ACROSS the",
        "  ridge instead of along it and captured only 3.8-8.5% of the wall's",
        "  energy; the region below captures 13-47% with a similar cell count.",
        "",
        "OCCUPANCY: mm4DPS.qa.occupancy, the function the",
        "Blocker C analysis (ledger R066) uses -- not a second implementation.",
        "  Per frame the person's SMPL points go into the renderer frame",
        "  (y += geometry_translation_y_m) and project to angle-FFT bins as",
        "  32 -/+ 32 sin(theta) against that radar's own array axis; the",
        "  silhouette is [min bin, max bin] over the person's own points, no",
        "  beam pad. The horizontal array lies on +X (AZIMUTH); the vertical",
        "  array lies on -Z (ELEVATION), and is labelled as such.",
        "",
        "RED TRACE -- BINARY, 1 = occluded and 0 = clear, drawn as a step.",
        "  It is the debounced state of an underlying graded quantity: the",
        "  share of the region's MEASURED power that sits behind the person,",
        "  = sum over angle bins in the silhouette of that bin's share of the",
        "  region's mean power. The graded share is reported per video below",
        "  and is burned into each frame's header, but the plotted trace and",
        "  the events are binary.",
        "  Weighting by the measured power is deliberate: the measured RA map",
        "  already contains the array's angular response, so this uses the",
        "  measurement's own angular distribution. Convolving again with a",
        "  modelled beam pattern double-counts it -- tried, and it shrinks the",
        "  signal to ~1% and weakens the seq 256 horizontal anti-correlation",
        "  from -0.44 to -0.05, because Dirichlet sidelobes span the whole",
        "  field and dilute the share.",
        "",
        "TRANSITION EVENTS -- dashed lines, and the binary they define.",
        "  BOTH TRACES COME FROM GROUND TRUTH AND NOTHING ELSE. Red is a",
        "  detector applied to the blue trace: the same function applied to a",
        "  RENDERED trace gives the rendered red, and the two can be compared.",
        "  The person's pose is NOT an input; it is kept as an independent",
        "  check and reported per video below as `geometry agreement`.",
        "",
        f"  The wall's CLEAR level is the {CLEAR_QUANTILE:.0%} quantile of its own",
        f"  despeckled trace ({DESPECKLE}-frame centred median). Occluded below",
        f"  {OCCLUDE_ENTER:.0%} of that level, clear again above "
        f"{OCCLUDE_LEAVE:.0%}, runs shorter than {MIN_RUN_FRAMES} frames "
        f"({MIN_RUN_FRAMES / SOURCE_FPS:g} s) absorbed.",
        "  A fraction of the clear level rather than a fixed threshold: seq 38",
        "  sits at 0.05 occluded and at TWO clear levels, 0.45 and 0.85, so a",
        "  global split lands between the clear levels and calls the lower one",
        "  occluded.",
        "",
        f"  Transitions are snapped back up to {SNAP_FRAMES} frames to where the",
        "  trace crossed the midpoint between the two thresholds. Hysteresis",
        "  fires at the FAR threshold, part-way through the change, which read",
        "  as red lagging the RA map. Snapped, seq 38's first reveal sits at",
        "  frame 118 -- the first elevated sample.",
        "",
        "  Short events are kept deliberately: seq 102's first 12 s contain two",
        "  occlusions of 1.0 and 1.6 s. An earlier detector enforced 2 s between",
        "  events and lost both.",
        "BLUE TRACE: mean |RA| over exactly the region's cells, LINEAR",
        "  MAGNITUDE, never dB, matching the RA panel above it. Normalised by",
        "  its own per-video maximum, printed below.",
        "",
        "PER VIDEO:",
    ]
    for row in report["videos"]:


        lines += [
            f"  {Path(row['path']).name}",
            f"    seq {row['sequence']} {row['radar']}: range bins "
            f"{row['range_bins'][0]}-{row['range_bins'][1]} "
            f"({row['range_m'][0]:.2f}-{row['range_m'][1]:.2f} m, peak "
            f"{row['peak_range_bin']}) x {row['angular_axis']} bins "
            f"{row['angle_bins'][0]}-{row['angle_bins'][1]} = {row['cells']} cells",
            f"    normalisation max |RA| = "
            f"{row['normalisation_max_linear_magnitude']:.6e}",
            f"    frames occluded {row['frames_occluded']}, "
            f"clear {row['frames_clear']}, transitions {row['transitions']}",
            f"    transition frames {row['transition_frames']}",
            f"    pose agreement {row['geometry_agreement']:.3f} vs "
            f"shifted-pose null {row['geometry_agreement_null']:.3f}"
            + ("  (null degenerate: binarised pose is constant)"
               if row["geometry_agreement_z"] is None
               else f"  z = {row['geometry_agreement_z']:+.2f}"),

        ]
    lines += [
        "",
        "POSE AGREEMENT is the independent check, since the pose is not an",
        "input to the detector. It is only interpretable against its null:",
        "where the binarised pose is constant the null has no spread and the",
        "comparison says nothing. Only seq 256 horizontal clears it --",
        "0.598 against a shifted-pose null of 0.426, z +2.66. See",
        "docs/SCENE_CONDITIONING.md.",
    ]
    lines.extend([""] + profile_provenance_lines(
        [args.outputs_root / f"{args.workspace_prefix}{sequence}"
         for sequence in args.sequences]))
    (args.output_dir / "PROVENANCE").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output_dir / 'PROVENANCE'}", flush=True)




# ----------------------------------------------------------------------
# gt-convention-video: camera | RA | RE, to check up/down and left/right.







# ----------------------------------------------------------------------
# ra-panel-video: the measured|rendered RA panel, from any product dir.







def make_parser_occlusion_check_media() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    from mm4DPS import radar_configs
    radar_configs(parser)
    parser.add_argument("--source-dir", type=Path,
                        default=Path("outputs/artifacts_doppler_support"))
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--sequences", type=int, nargs="+",
                        default=[38, 102, 256])
    parser.add_argument("--workspace-prefix", default="vf_ds10_")
    parser.add_argument("--render-suffix", default="v2_600")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/artifacts_occlusion_check"))
    parser.add_argument("--mask-on-dir", type=Path, default=None,
                        help="product dir of a per-frame-visibility render; "
                             "adds a third row scored identically")
    parser.add_argument("--right-panel-label", default="from_scratch",
                        help="what the source video's RIGHT panel actually "
                             "shows; only names it in the burnt-in caption")
    parser.add_argument("--with-rendered", action="store_true",
                        help="add a second plot row for the RENDERED cube, "
                             "built by the same region, extraction and "
                             "detector as the measured one")
    return parser


def _entry_occlusion_check_media():
        print(json.dumps(run_occlusion_check_media(make_parser_occlusion_check_media().parse_args()), indent=2))



# ----------------------------------------------------------------------
# what the preprocessing produced: the mesh on the video, the points with and
# without the visibility mask
# ----------------------------------------------------------------------

def _write_video(frames_dir: Path, output: Path, fps: int = 10) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", str(frames_dir / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(output)], check=True)


def run_mesh_on_rgb(args: argparse.Namespace) -> dict:
    """The GVHMR SMPL mesh drawn on the left camera's frames (RT-Pose).

    Every vertex of the person sequence is taken back from the radar world
    to the LiDAR frame (the extrinsic seed inverted) and projected with the
    shipped LiDAR->camera calibration -- the same path the geometry came in
    by, so a mesh off the person here is a placement error, not a drawing
    one. The SMPL faces are drawn back to front as a translucent surface,
    radar-visible faces warm and hidden ones cool, with the wireframe on
    top. The annotated joints (shifted by ANNOTATION_TO_LIDAR) are drawn as
    crosses for reference.
    """
    from mm4DPS import RTPOSE, Dataset
    from mm4DPS.data import DEFAULT_MODEL, RTPoseSequence, SmplBody
    from mm4DPS.reconstruction import rtpose_extrinsic_seed
    Dataset.select("rtpose")
    seq = RTPoseSequence(RTPOSE, args.sequence)
    seed = rtpose_extrinsic_seed(args.repo)
    faces = SmplBody(args.model_path).faces.numpy()
    metadata = json.loads((args.sequence_dir / "metadata.json").read_text())
    frames = int(metadata["frames"])
    scratch = args.output.with_suffix("").with_name(args.output.stem + "_frames")
    scratch.mkdir(parents=True, exist_ok=True)
    annotation = seq.annotation()
    for frame in range(frames):
        stem = seq.stem_for_frame(frame)
        image = Image.open(seq.image_path(stem)).convert("RGB")
        payload = np.load(args.sequence_dir / "person_frames" / f"frame_{frame:03d}.npz")
        pixels = seq.project(seq.to_lidar(payload["pcl"][:, :3].astype(np.float64), seed), image_size=image.size)
        visible = payload["visible"] if "visible" in payload else np.ones(len(pixels), dtype=bool)
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        tri = pixels[faces]                                            # (F, 3, 3)
        in_front = (tri[:, :, 2] > 0).all(axis=1)
        order = np.argsort(-tri[:, :, 2].mean(axis=1))                 # far first
        face_visible = visible[faces].sum(axis=1) >= 2
        for f in order:
            if not in_front[f]:
                continue
            xy = [(float(x), float(y)) for x, y in tri[f, :, :2]]
            colour = (255, 200, 60, 110) if face_visible[f] else (110, 80, 200, 110)
            draw.polygon(xy, fill=colour)
        if args.wireframe_stride > 0:
            for f in order[::args.wireframe_stride]:
                if in_front[f]:
                    xy = [(float(x), float(y)) for x, y in tri[f, :, :2]]
                    draw.polygon(xy, outline=(40, 20, 20, 90))
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(image)
        if frame in annotation:
            joints = seq.project(annotation[frame]["joints"] + seq.ANNOTATION_TO_LIDAR, image_size=image.size)
            for x, y, depth in joints:
                if depth > 0:
                    draw.line([x - 6, y, x + 6, y], fill=(255, 40, 40), width=2)
                    draw.line([x, y - 6, x, y + 6], fill=(255, 40, 40), width=2)
        draw.text((12, 10), f"seq {args.sequence} radar frame {frame} image {stem}  "
                            f"warm: radar-visible SMPL faces  cool: hidden  red: annotation",
                  fill=(255, 255, 255))
        image.save(scratch / f"{frame:05d}.png")
    _write_video(scratch, args.output)
    if not args.keep_frames:
        shutil.rmtree(scratch)
    return {"output": str(args.output), "frames": frames}


def _points_panel(scene: np.ndarray, person: np.ndarray, size: int, title: str,
                  scene_kept: np.ndarray | None = None, person_kept: np.ndarray | None = None) -> Image.Image:
    """Top view (x right, range up) of scene and person points on a fixed metre grid."""
    span_x, span_y = 5.0, 8.0
    image = Image.new("RGB", (size, size), (18, 18, 24))
    draw = ImageDraw.Draw(image)
    def px(points):
        return np.stack([(points[:, 0] / span_x + 0.5) * size, (1.0 - points[:, 1] / span_y) * size], axis=1)
    for r in range(1, 8):
        y = (1.0 - r / span_y) * size
        draw.line([0, y, size, y], fill=(40, 40, 50))
        draw.text((4, y - 12), f"{r} m", fill=(120, 120, 140))
    for points, kept, colour, dim in ((scene, scene_kept, (170, 170, 170), (70, 40, 40)),
                                      (person, person_kept, (255, 200, 60), (110, 70, 160))):
        p = px(points)
        for i in range(len(p)):
            x, y = p[i]
            if not (0 <= x < size and 0 <= y < size):
                continue
            c = colour if kept is None or kept[i] else dim
            draw.point((x, y), fill=c) if kept is None or kept[i] else draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=c)
    draw.polygon([(size / 2, size - 4), (size / 2 - 6, size - 16), (size / 2 + 6, size - 16)], fill=(255, 45, 85))
    draw.text((10, 8), title, fill=(255, 255, 255))
    return image


def run_scene_and_person(args: argparse.Namespace) -> dict:
    """Scene + person points, LEFT without and RIGHT with the per-frame visibility mask.

    Both panels come from the same files: the sequence's global_scene_points
    (the aligned scene), its person_frames (the placed person), and, on the
    right only, scene_visibility.npy and each frame's `visible`. Everything
    up to placement is identical on both sides; the mask is the only
    difference, and on the right the masked points are drawn dim (scene:
    dark red; person: purple) rather than dropped, so what it removed stays
    visible. Dataset-agnostic: HuPR sequences draw the same way.
    """
    metadata = json.loads((args.sequence_dir / "metadata.json").read_text())
    frames = int(metadata["frames"])
    if args.checkpoint is not None:
        # The run's own scene and mask: the background the chain trained and
        # the per-frame shadow it cast against it (scene-shadow-mask).
        import torch
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["shared_background_state"]
        scene = state["positions"].numpy().astype(np.float64)
        scene_visibility = np.load(args.scene_mask)
        if scene_visibility.shape[1] != len(scene):
            raise ValueError(f"scene mask is over {scene_visibility.shape[1]} points, the checkpoint holds {len(scene)}")
    else:
        scene = np.load(args.sequence_dir / "global_scene_points.npy")[:, :3].astype(np.float64)
        scene_visibility = np.load(args.sequence_dir / "scene_visibility.npy")
        if len(scene) > args.max_scene_points:
            # The RGB-fused scaffold is ~250k points; drawing needs far fewer.
            keep = np.linspace(0, len(scene) - 1, args.max_scene_points).astype(int)
            scene, scene_visibility = scene[keep], scene_visibility[:, keep]
    scratch = args.output.with_suffix("").with_name(args.output.stem + "_frames")
    scratch.mkdir(parents=True, exist_ok=True)
    for frame in range(frames):
        payload = np.load(args.sequence_dir / "person_frames" / f"frame_{frame:03d}.npz")
        person = payload["pcl"][:, :3].astype(np.float64).copy()
        person[:, 1] += args.translation_y_m
        visible = payload["visible"] if "visible" in payload else np.ones(len(person), dtype=bool)
        left = _points_panel(scene, person, args.size, f"frame {frame}: scene + person, no visibility")
        right = _points_panel(scene, person, args.size,
                              f"with visibility: {int((~scene_visibility[frame]).sum())} scene pts shadowed, "
                              f"{int((~visible).sum())} person pts hidden",
                              scene_kept=scene_visibility[frame], person_kept=visible)
        canvas = Image.new("RGB", (2 * args.size + 8, args.size), (0, 0, 0))
        canvas.paste(left, (0, 0)); canvas.paste(right, (args.size + 8, 0))
        canvas.save(scratch / f"{frame:05d}.png")
    _write_video(scratch, args.output)
    if not args.keep_frames:
        shutil.rmtree(scratch)
    return {"output": str(args.output), "frames": frames}


def make_parser_mesh_on_rgb() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=run_mesh_on_rgb.__doc__)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wireframe-stride", type=int, default=4,
                        help="draw every n-th face's edges over the surface; 0 for none")
    parser.add_argument("--keep-frames", action="store_true")
    from mm4DPS.data import DEFAULT_MODEL
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL, help="the SMPL npz (its faces)")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    return parser


def make_parser_scene_and_person() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=run_scene_and_person.__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--translation-y-m", type=float, default=0.0,
                        help="the calibrated person range shift, if the sequence has one")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="person checkpoint whose shared_background_state is the scene to draw "
                             "(default: the sequence's global_scene_points)")
    parser.add_argument("--scene-mask", type=Path, default=None,
                        help="(frames, points) bool mask over the checkpoint's scene, e.g. the chain's "
                             "visibility/seq<S>_s2_shadow.npy; required with --checkpoint")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--max-scene-points", type=int, default=40000)
    parser.add_argument("--keep-frames", action="store_true")
    return parser


def _entry_mesh_on_rgb():
    print(json.dumps(run_mesh_on_rgb(make_parser_mesh_on_rgb().parse_args()), indent=2))


def _entry_scene_and_person():
    print(json.dumps(run_scene_and_person(make_parser_scene_and_person().parse_args()), indent=2))


COMMANDS = {
    "mesh-on-rgb": _entry_mesh_on_rgb,
    "scene-and-person": _entry_scene_and_person,
    "microdoppler-spectrograms": _entry_microdoppler_spectrograms,
    "velocity-comparison-media": _entry_velocity_comparison_media,
    "occlusion-check-media": _entry_occlusion_check_media,
}


def main() -> int:
    return dispatch(COMMANDS, __name__)


if __name__ == "__main__":
    raise SystemExit(main())
