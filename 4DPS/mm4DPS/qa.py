"""Provenance guards, the orphan audit, and the deployment path's diagnostics.

Four modules merged here by `docs/CONSOLIDATION_AND_OO_PLAN.md` C1:
`qa/chain_profile.py`, `qa/physics_fingerprint.py`, `qa/orphans.py` and
`diagnostics/support.py`. Bodies are verbatim; three edits were needed and are
the whole of the behaviour change in this batch:

1. `provenance_lines` was defined in BOTH guards. They are now
   `profile_provenance_lines` (which chain profile built a workspace) and
   -- see 2 -- nothing else.
2. `physics_fingerprint.provenance_lines` was never called. It survived every
   AST orphan sweep because `chain_profile.provenance_lines` shares the name and
   the sweep matches names, not bindings: one live import protected two
   definitions. Removed. This is a real limit of `find()` below, now recorded in
   its docstring.
3. `orphans.REPO` counted three `__file__` parents from `mm4DPS/qa/orphans.py`.
   This file is one level up, so it counts two. Re-derived rather than copied,
   which is the rule for every `__file__` value in this consolidation.

`diagnostics/support.py`'s three helpers are grouped here by role rather than by
call graph -- the weakest merge in the plan, and named as such in it. They are
what the deployment path kept of the old audit suites; recover those with
`git show pre-deploy-strip:mm4DPS/diagnostics/scene.py`.

Run a tool with `python -m mm4DPS.qa <command>`; `orphans --methods` includes
class methods. `check-environment` is the check the Makefile lost in the strip.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np

from mm4DPS import DATASETS, Dataset, Install, apply
from mm4DPS.cuda_ext import stamp_status



# --------------------------------------------------------------------
# chain profile: which CHAIN_PROFILE built a workspace (R097)
# --------------------------------------------------------------------


MARKER = "chain_profile.txt"
VALID = ("full", "velocity")


def profile_of_workspace(workspace: Path) -> str | None:
    """The CHAIN_PROFILE that built ``workspace``, or None if unrecorded."""
    marker = Path(workspace) / "deliverables" / MARKER
    if not marker.exists():
        return None
    value = marker.read_text().strip()
    return value if value in VALID else None


def require_same_profile(workspaces, tool: str) -> str:
    """Refuse to score products built by different chain profiles.

    ``workspaces`` is an iterable of workspace directories. Returns the shared
    profile so a caller can record it in its own provenance.
    """
    seen: dict[str, list[str]] = {}
    for workspace in workspaces:
        name = Path(workspace).name
        profile = profile_of_workspace(workspace)
        seen.setdefault(profile or "UNRECORDED", []).append(name)

    if len(seen) == 1:
        (only,) = seen
        if only != "UNRECORDED":
            return only
        raise SystemExit(
            f"{tool}: none of these workspaces records which CHAIN_PROFILE "
            f"built it ({', '.join(sorted(seen['UNRECORDED']))}). They were "
            "built before the marker existed, so whether they share a profile "
            "cannot be established -- and two profiles produce products that "
            "look identical while their person stacks differ. Rebuild them, "
            "or pass --allow-unrecorded-profile if you know their provenance "
            "by other means and are willing to own it.")

    detail = "; ".join(f"{profile}: {', '.join(sorted(names))}"
                       for profile, names in sorted(seen.items()))
    raise SystemExit(
        f"{tool}: these workspaces were built by DIFFERENT chain profiles "
        f"({detail}). The products are comparable in shape and not in "
        "provenance: the profiles run different amounts of stochastic "
        "training, so an arm comparison across them varies more than the "
        "thing under test. Rebuild the odd one out under the same profile.")


def profile_provenance_lines(workspaces) -> list[str]:
    """Lines naming each workspace's profile, for an artifact's PROVENANCE."""
    lines = ["CHAIN_PROFILE of each source workspace:"]
    for workspace in workspaces:
        profile = profile_of_workspace(workspace) or "UNRECORDED"
        lines.append(f"  {Path(workspace).name}: {profile}")
    lines.append(
        "  (products of different profiles look identical and are not "
        "interchangeable in an A/B -- ledger R097)")
    return lines


# --------------------------------------------------------------------
# physics fingerprint: what forward model built a product (R099)
# --------------------------------------------------------------------


RESULT = "render_result.json"

# Scalars compared directly.
SCALAR_KEYS = (
    "geometry_translation_y_m",
    "geometry_depth_scale",
    "near_field_guard_m",
    "first_retained_range_bin",
    "phase_model",
    "motion_representation",
    "velocity_differentiator",
    "velocity_gain",
    "velocity_scale_mode",
    "velocity_scale_floor",
    "selected_velocity_frame_offset",
    "slow_time_loops_per_frame",
    "physical_tx_chirps_per_frame",
    "fps",
    "frames",
)

# Nested switches compared on one field -- usually whether they are on at all.
NESTED_KEYS = {
    "per_frame_scene_visibility": "enabled",
    "antenna_coupling": "applied",
    "frame_receiver_nuisance": "enabled",
    "local_additive_receiver_nuisance": "enabled",
    "within_frame_slow_time_nuisance": "enabled",
    "receiver_noise": "enabled",
    "unsupported_range_diffuse_floor": "enabled",
    "static_scene_jitter": "enabled",
    "joint_velocity_residual": "enabled",
    "velocity_field": "parameterization",
}


def fingerprint(product_dir: Path) -> dict:
    """The physics switches a product was rendered with."""
    product_dir = Path(product_dir)
    payload = json.loads((product_dir / RESULT).read_text())
    out: dict = {}
    for key in SCALAR_KEYS:
        if key in payload:
            out[key] = payload[key]
    for key, field in NESTED_KEYS.items():
        value = payload.get(key)
        if isinstance(value, dict) and field in value:
            out[f"{key}.{field}"] = value[field]
    gain = payload.get("absolute_receiver_gain")
    if isinstance(gain, dict):
        for radar in sorted(gain):
            out[f"absolute_receiver_gain.{radar}"] = gain[radar]
    return out


def label(product_dir) -> str:
    """A name that distinguishes two ARMS, not just two product dirs.

    Both arms of an A/B render into the same product name (`seq256_v2_600`), so
    labelling by basename alone collapses them into one entry and the refusal
    message names one product where it means two. The workspace is the part
    that differs.
    """
    product_dir = Path(product_dir).resolve()
    workspace = product_dir.parent.parent.name
    return f"{workspace}/{product_dir.name}"


def differences(products) -> dict[str, dict[str, object]]:
    """Every fingerprint key on which the products do not all agree."""
    prints = {label(p): fingerprint(p) for p in products}
    keys = set().union(*(set(f) for f in prints.values())) if prints else set()
    out = {}
    for key in sorted(keys):
        values = {name: f.get(key, "<absent>") for name, f in prints.items()}
        if len({json.dumps(v, sort_keys=True) for v in values.values()}) > 1:
            out[key] = values
    return out


def require_comparable(products, tool: str, varying=()) -> dict:
    """Refuse a comparison whose products vary the wrong things, or nothing.

    ``varying`` names the fingerprint keys this comparison EXISTS to vary --
    e.g. ``("per_frame_scene_visibility.enabled",)`` for a mask A/B. Those keys
    must differ; every other key must agree.
    """
    products = list(products)
    if len(products) < 2:
        return {}
    varying = tuple(varying)
    found = differences(products)

    unintended = {k: v for k, v in found.items() if k not in varying}
    if unintended:
        detail = "; ".join(
            f"{key} " + ", ".join(f"{name}={value!r}"
                                  for name, value in sorted(values.items()))
            for key, values in unintended.items())
        raise SystemExit(
            f"{tool}: these products were rendered with DIFFERENT physics, in "
            f"switches this comparison did not declare: {detail}. They are "
            "comparable in shape and not in forward model, so a difference "
            "between them is not attributable to the thing under test. "
            "Re-render the odd one out, or declare the key in `varying` if the "
            "difference is deliberate.")

    missing = [key for key in varying if key not in found]
    if missing:
        values = {f"{label(p)}:{key}": fingerprint(p).get(key, "<absent>")
                  for p in products for key in missing}
        raise SystemExit(
            f"{tool}: this comparison says it varies {', '.join(missing)}, and "
            f"the products AGREE on it ({values}). The arms are identical in "
            "the one respect being tested, so any table built from them would "
            "attribute a difference to a switch that was the same on both "
            "sides. This is R099: a visibility A/B whose two arms both "
            "recorded per_frame_scene_visibility.enabled = false, scored and "
            "written up before anyone read it. Check the flag actually reached "
            "the render.")
    return found




VARYING_HELP = (
    "physics-fingerprint keys this comparison EXISTS to vary, e.g. "
    "per_frame_scene_visibility.enabled. Products must differ in these and "
    "agree on everything else. Declaring it is what makes a null comparison "
    "-- two arms identical in the thing under test -- a refusal rather than a "
    "table (ledger R099).")


def add_varying_argument(parser) -> None:
    """Declare `--varying` in ONE place.

    Two copies of a flag drift, and one did. A consistency test caught this one
    the same day it was written; that test went with `tests/` in the deployment
    strip, so declaring `--varying` in one place is the whole defence now.
    """
    parser.add_argument("--varying", nargs="*", default=(), metavar="KEY",
                        help=VARYING_HELP)


# --------------------------------------------------------------------
# the three helpers the deployment path kept from diagnostics/
# --------------------------------------------------------------------


def person_angle_bins(points_xyz: np.ndarray, centre: np.ndarray,
                      axis: np.ndarray, radar: str) -> np.ndarray:
    """Angle-FFT bin of each point, ``32 -/+ 32 sin(theta)``.

    ``person_range_angle_mask`` carries both signs because HuPR does not
    publish the array phase convention. The horizontal sign is settled here
    from the data: over 600 frames the measured angle centroid of the person's
    own return correlates +0.77..+0.96 with ``32 - 32 sin`` and equally
    negatively with the other, on all three sequences. The vertical array
    lies on -Z and measures ELEVATION, where the person's excursion is too
    small to settle anything; its sign does not affect any conclusion here.
    """
    delta = points_xyz - centre
    distance = np.linalg.norm(delta, axis=-1)
    sine = (delta @ axis) / np.maximum(distance, 1e-8)
    # The vertical board's array lies on -Z; every other array (HuPR's
    # horizontal, the cascade) lies on +X and takes the settled sign.
    return 32.0 + 32.0 * sine if radar == "vertical" else 32.0 - 32.0 * sine


def occupancy(sequence_dir_path: Path, frames: int, translation_y_m: float,
              centre: np.ndarray, axis: np.ndarray, radar: str) -> np.ndarray:
    """(frames, 64) bool: azimuth bins spanned by the person's own points.

    No beam pad. The pad used for event attribution widens the person by half
    a main lobe, which makes the bins around boresight covered in >99% of
    frames and therefore untestable -- it blinds exactly the bins the effect
    is strongest in.
    """
    covered = np.zeros((frames, 64), dtype=bool)
    bins = np.arange(64)
    for frame in range(frames):
        payload = np.load(sequence_dir_path / "person_frames" / f"frame_{frame:03d}.npz")
        points = payload["pcl"][:, :3].astype(np.float64).copy()
        points[:, 1] += translation_y_m
        angle = person_angle_bins(points, centre, axis, radar)
        covered[frame] = (bins >= angle.min()) & (bins <= angle.max())
    return covered


def require_completed_render(directory: Path, product_names: list[str]) -> Path:
    """Reject arrays that may still be written by an active renderer."""
    candidates = (directory / "render_result.json", directory.parent / "render_result.json")
    marker = next((path for path in candidates if path.is_file()), None)
    if marker is None:
        raise RuntimeError(
            f"no render_result.json completion marker for {directory}; "
            "use --allow-unverified-render only for externally verified legacy products"
        )
    metadata = json.loads(marker.read_text())
    if metadata.get("status") != "completed":
        raise RuntimeError(f"render is not completed according to {marker}")
    products = [directory / name for name in product_names]
    missing = [str(path) for path in products if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"render products are missing: {missing}")
    newest_product_time = max(path.stat().st_mtime_ns for path in products)
    if marker.stat().st_mtime_ns < newest_product_time:
        raise RuntimeError(
            f"completion marker {marker} predates an input array; render may be incomplete"
        )
    return marker


# --------------------------------------------------------------------
# the orphan audit: report definitions no code references
# --------------------------------------------------------------------


REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "mm4DPS"
#: Reached by the runtime rather than by a reference, so never reported.
DUNDER_OR_PROTOCOL = {"forward", "backward", "main", "__init__", "__post_init__",
                      "__call__", "__enter__", "__exit__", "__repr__"}


def _sources() -> dict[Path, ast.Module]:
    trees = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in str(path) or path.name == "setup.py":
            continue
        try:
            trees[path] = ast.parse(path.read_text())
        except SyntaxError:
            continue
    return trees


def _named(tree: ast.Module, include_methods: bool):
    """(name, first_line, last_line) for each definition worth reporting."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            yield node.name, start, node.end_lineno
            if include_methods and isinstance(node, ast.ClassDef):
                for member in node.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        yield (member.name,
                               min([member.lineno]
                                   + [d.lineno for d in member.decorator_list]),
                               member.end_lineno)


def references(name: str, trees: dict[Path, ast.Module],
               own: tuple[Path, int, int]) -> int:
    """AST references to `name`, ignoring the definition's own span."""
    own_path, low, high = own
    total = 0
    for path, tree in trees.items():
        for node in ast.walk(tree):
            line = getattr(node, "lineno", None)
            if path == own_path and line is not None and low <= line <= high:
                continue
            if isinstance(node, ast.Name) and node.id == name:
                total += 1
            elif isinstance(node, ast.Attribute) and node.attr == name:
                total += 1
            elif isinstance(node, ast.ImportFrom) and any(
                    alias.name == name for alias in node.names):
                total += 1
    return total


def find(include_methods: bool = False) -> list[tuple[int, str]]:
    trees = _sources()
    found = []
    for path, tree in trees.items():
        for name, low, high in _named(tree, include_methods):
            if name in DUNDER_OR_PROTOCOL or name.startswith("__"):
                continue
            if references(name, trees, (path, low, high)) == 0:
                found.append((high - low + 1,
                              f"{path.relative_to(REPO)}::{name}"))
    found.sort(reverse=True)
    return found

# ----------------------------------------------------------------------
# The tool entry points. `paths`, `environment` and `radar-config` had a
# __main__ each and NOTHING invoked any of them -- no Makefile target, no chain
# call. `make check-environment` had gone with the strip, which is why
# cuda/setup.py still referred to a `check_environment` that did not exist.
# They are subcommands here, so the capability is reachable again and the three
# colliding `main`s become one.
# ----------------------------------------------------------------------
def _entry_radar_config(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(Install.from_env().config_dir))
    parser.add_argument("--baseline-m", type=float, default=0.03)
    parser.add_argument("--vertical-roll-deg", type=float, default=90.0)
    args = parser.parse_args(argv)
    paths = write_pair(args.output_dir, args.baseline_m, args.vertical_roll_deg)
    print("\n".join(str(path) for path in paths))
    return 0


def _entry_check_environment(argv) -> int:
    """Where this machine's locations resolved to, and what is missing."""
    parser = argparse.ArgumentParser(prog="check-environment", description=__doc__)
    parser.parse_args(argv)
    # apply() FIRST: it is what the chain does at its entry point, so without it
    # this reports a dataset root the pipeline will never use.
    apply()
    install = Install.from_env()
    for name, status in install.describe().items():
        print(f"{name:18s} {status}")
    # Content, not mtime. A stale .so presents as a pybind signature error that
    # names a function and says nothing about compilation; the build stamp was
    # written for this purpose and had no reader until C6.
    print()
    stale = False
    for name, status in stamp_status().items():
        print(f"{name:14s} {status}")
        stale = stale or "STALE" in status or "NOT BUILT" in status
    if stale:
        print("  BOTH extensions are required: a tree with only hupr_cuda passes")
        print("  every check here and then dies four stages into a chain.")
    missing = install.missing()
    if missing:
        print(f"\nMISSING: {', '.join(missing)}")
    return 1 if (missing or stale) else 0


def _entry_environment(argv) -> int:
    """The variables `apply()` exports so child processes inherit them."""
    parser = argparse.ArgumentParser(prog="environment", description=__doc__)
    parser.parse_args(argv)
    for key, value in apply().items():
        print(f"{key}={value}")
    return 0


def _entry_orphans(argv) -> int:
    parser = argparse.ArgumentParser(prog="orphans", description=find.__doc__)
    parser.add_argument("--methods", action="store_true",
                        help="include class methods (they were missed for a long "
                             "time; forward/backward are still excluded because "
                             "the runtime calls them, not a reference)")
    args = parser.parse_args(argv)
    found = find(args.methods)
    if not found:
        print("no unreferenced definitions")
        return 0
    for lines, where in found:
        print(f"{lines:5d}  {where}")
    print(f"\n{len(found)} unreferenced definitions, {sum(f[0] for f in found)} lines.")
    print("INVESTIGATE, do not delete: check for a live invocation first, and read "
          "the limits in this module's docstring.")
    return 0


# --------------------------------------------------------------------
# the cascade: its config, and the board orientation checked from data
# --------------------------------------------------------------------

def _entry_cascade_config(argv) -> int:
    """Write configs/mmwcas_cascade.json from the RT-Pose radar description."""
    from mm4DPS import write_cascade
    parser = argparse.ArgumentParser(prog="cascade-config", description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(Install.from_env().config_dir))
    args = parser.parse_args(argv)
    print(write_cascade(args.output_dir))
    return 0


def person_track(sequence: int, device: str = "cuda:0", top_fraction: float = 0.01) -> np.ndarray:
    """Per annotated frame: measured (range m, sin theta_X) of the person, annotated torso (x fwd, y left).

    The measured person is the power-weighted centroid of the top ``top_fraction``
    cells of the clutter-removed, loop-averaged range-azimuth map (bins 10..200 in
    range), with ``sin theta_X = (N/2 - bin) / (N/2)`` -- the convention the HuPR
    boards settled from data (:func:`person_angle_bins`). The annotated torso is the
    mean of pelvis, thorax and the two shoulders, in the LiDAR frame.
    """
    import torch
    from mm4DPS import RTPOSE, Dataset
    from mm4DPS.data import (RTPoseSequence, apply_cascade_calibration, cascade_calibration,
                             decode_tda2_frame_torch, range_angle_slow_time_cube_torch)
    Dataset.select("rtpose")
    seq = RTPoseSequence(RTPOSE, sequence)
    radar = seq.radar
    calibration = cascade_calibration(
        radar, Install.from_env().config_dir / radar.calibration).to(device)
    recording = seq.recording()
    rows = []
    for frame, entry in seq.annotation().items():
        if frame >= recording.frames:
            continue
        words = torch.from_numpy(recording.frame_words(frame)).to(device)
        adc = apply_cascade_calibration(decode_tda2_frame_torch(words, radar), calibration, radar)
        ra = range_angle_slow_time_cube_torch(adc, remove_static_clutter=True, radar=radar).abs().mean(0)[10:200]
        flat = ra.flatten()
        top = torch.topk(flat, max(1, int(flat.numel() * top_fraction)))
        weight = top.values
        rng = float((weight * (top.indices // ra.shape[1]).float()).sum() / weight.sum()) + 10
        bin_ = float((weight * (top.indices % ra.shape[1]).float()).sum() / weight.sum())
        torso = entry["joints"][[0, 7, 9, 12]].mean(axis=0)
        rows.append((rng * radar.range_bin_m, (radar.n_angle / 2 - bin_) / (radar.n_angle / 2),
                     torso[0], torso[1]))
    return np.asarray(rows, dtype=np.float64)


def fit_lateral(track: np.ndarray) -> dict:
    """The radar's (forward, left) offset from the LiDAR that best explains the track.

    Least squares (soft-L1) over range and sin theta, with the radar's boresight
    along the LiDAR's x. Estimator named in the result: rms of the range and sin
    residuals over the frames used.
    """
    from scipy.optimize import least_squares

    def residuals(p):
        forward, left = track[:, 2] - p[0], track[:, 3] - p[1]
        rng = np.hypot(forward, left)
        return np.concatenate([(rng - track[:, 0]) / 0.05, (-left / rng - track[:, 1]) / 0.03])
    fit = least_squares(residuals, [0.0, 0.0], loss="soft_l1")
    n = len(track)
    res = residuals(fit.x)
    return {"radar_in_lidar_forward_m": float(fit.x[0]), "radar_in_lidar_left_m": float(fit.x[1]),
            "rms_range_m": float(np.sqrt(np.mean((res[:n] * 0.05) ** 2))),
            "rms_sin": float(np.sqrt(np.mean((res[n:] * 0.03) ** 2))), "frames": int(n)}


def _entry_orientation_check(argv) -> int:
    """Does the measured person land on the annotated person, or on its mirror?

    Over a sequence's annotated frames the measured sin theta_X of the person
    (from the beamformed, clutter-removed range-azimuth map) is correlated with
    the annotation's -y_left / range: +1 means the person's return lands on the
    person under the config's board orientation, -1 means on its mirror, i.e.
    the board is rotated 180 degrees from what configs/mmwcas_cascade.json says.
    Also reports the lateral-offset fit and its residual. Walking sequences only
    make sense here: a person on boresight cannot tell the two apart.
    """
    parser = argparse.ArgumentParser(prog="orientation-check", description=__doc__)
    parser.add_argument("--sequences", type=int, nargs="+", default=[183, 186, 153])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = {}
    for sequence in args.sequences:
        track = person_track(sequence, args.device)
        annotated = -track[:, 3] / np.hypot(track[:, 2], track[:, 3])
        r_sin = float(np.corrcoef(track[:, 1], annotated)[0, 1])
        r_range = float(np.corrcoef(track[:, 0], np.hypot(track[:, 2], track[:, 3]))[0, 1])
        report[sequence] = {"frames": int(len(track)), "pearson_sin": r_sin, "pearson_range": r_range,
                            "verdict": "person" if r_sin > 0.5 else "MIRROR" if r_sin < -0.5 else "undecided",
                            "lateral_fit": fit_lateral(track)}
        fit = report[sequence]["lateral_fit"]
        print(f"seq {sequence}: {len(track)} frames, r(sin theta_X, -y_left/r) = {r_sin:+.3f}, "
              f"r(range) = {r_range:+.3f} -> {report[sequence]['verdict']}; radar at "
              f"({fit['radar_in_lidar_forward_m']:+.2f} fwd, {fit['radar_in_lidar_left_m']:+.2f} left) m, "
              f"rms range {fit['rms_range_m']:.3f} m, rms sin {fit['rms_sin']:.3f}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if all(v["verdict"] == "person" for v in report.values()) else 1


def _entry_window_check(argv: list[str]) -> int:
    """The FFT-window consistency gate: one point target through both chains.

    A synthetic ADC frame of one scatterer (fractional range bin, azimuth,
    radial velocity) is taken through the MEASURED chain -- `data`'s range
    window + range FFT, the gridded azimuth window + angle FFT,
    `slow_time_to_doppler` -- and the same scatterer is synthesised the
    RENDERED way -- `rasterizer.hann_psf` at the fractional bin, the carrier
    phase, the slow-time phasor -- and taken through `forward_render`'s
    beamformer and `doppler_plane_weights`. With one window definition on
    both sides the two agree to floating-point rounding, and the gate says
    by how much. Also checked: the per-unit plane weights against
    `doppler_plane_weights` (bitwise), the RadProposer tensor from ADC
    against the one from range profiles, and the 15-bin PSF truncation's
    error against the untruncated response (the one approximation, 3DPS's).
    """
    import torch
    from mm4DPS.data import (adc_frame_to_radproposer_tensor, range_angle_slow_time_cube_torch,
                             range_profiles_all_channels_torch, range_profiles_to_radproposer_tensor)
    from mm4DPS.forward_render import (_per_unit_plane_weights, slow_time_profiles_to_ra_complex,
                                       slow_time_to_doppler)
    from mm4DPS.rasterizer import hann_psf
    from mm4DPS.signal_model import (SLOW_TIME_PHASE_SIGN, apply_fft_window, doppler_plane_weights,
                                     fft_window_kind, range_window_profiles, tdm_sample_times,
                                     unwindowed_range_reference)
    parser = argparse.ArgumentParser(prog="mm4DPS.qa window-check", description=_entry_window_check.__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="hupr")
    parser.add_argument("--tolerance", type=float, default=2e-6,
                        help="max |difference| / max |reference| the two chains must meet")
    parser.add_argument("--range-bin", type=float, default=71.37)
    parser.add_argument("--azimuth-deg", type=float, default=17.0)
    parser.add_argument("--velocity-mps", type=float, default=0.83)
    args = parser.parse_args(argv)
    dataset = Dataset.select(args.dataset)
    torch.manual_seed(0)
    worst = 0.0
    report: dict = {"dataset": dataset.name, "fft_window": fft_window_kind(), "radars": {}}
    for radar in dataset.radars:
        K, L = radar.samples, radar.loops
        rx_pos, tx_pos = radar.rotated_positions()          # half-wavelength units, (az, el)
        wavelength = 299_792_458.0 / radar.carrier_hz
        sin_az = np.sin(np.deg2rad(args.azimuth_deg))
        # per-channel carrier phase: pi * (az_tx + az_rx) * sin(theta) on the half-wavelength grid
        phase = np.pi * (tx_pos[:, None, 0] + rx_pos[None, :, 0]) * sin_az       # (tx, rx)
        times = tdm_sample_times(radar=radar)                                    # (loop, tx)
        rate = 2.0 * args.velocity_mps                                           # bistatic path rate
        slow = SLOW_TIME_PHASE_SIGN * 2.0 * np.pi / wavelength * rate * times    # (loop, tx)
        n = np.arange(K)
        tone = np.exp(1j * 2.0 * np.pi * args.range_bin * n / K)                 # (K,)
        # the ADC frame (rx, chirp, sample): chirp = loop * slots + slot, TX tx_order[slot]
        adc = np.zeros(radar.adc_frame_shape, dtype=np.complex128)
        slots = radar.loop_slots or radar.n_tx
        for l in range(L):
            for slot in range(slots):
                tx = radar.tx_order[slot] if slot < len(radar.tx_order) else None
                if tx is None:
                    continue
                for r in range(radar.n_rx):
                    adc[r, l * slots + slot] = np.exp(1j * (phase[tx, r] + slow[l, tx])) * tone
        adc_t = torch.from_numpy(adc.astype(np.complex64))
        # MEASURED chain
        measured_profiles = range_profiles_all_channels_torch(adc_t, radar=radar)          # (loop, tx, rx, K)
        measured_ra = range_angle_slow_time_cube_torch(adc_t, radar=radar)                  # (loop, range, angle)
        measured_doppler = slow_time_to_doppler(measured_ra.to(torch.complex128), dim=0)    # (doppler, range, angle)
        # RENDERED synthesis: the window's PSF at the fractional bin (untruncated), same phases
        delta = torch.arange(K, dtype=torch.float64) - args.range_bin
        psf = hann_psf(delta, K).numpy()                                                    # (K,) complex
        rendered = (np.exp(1j * (phase[None] + slow[:, :, None]))[:, :, :, None]
                    * psf[None, None, None, :])                                             # (loop, tx, rx, K)
        rendered_t = torch.from_numpy(rendered.astype(np.complex64))
        rendered_ra = slow_time_profiles_to_ra_complex(rendered_t.real, rendered_t.imag, radar=radar)
        rendered_doppler = slow_time_to_doppler(rendered_ra.to(torch.complex128), dim=0)
        bins = [d for d in range(L) if d != L // 2]
        weights = doppler_plane_weights(bins, loops=L).to(torch.complex128)                 # (1 + 63, loops)
        planes = torch.einsum("dl,ltrk->dtrk", weights[1:], rendered_t.to(torch.complex128))
        planes_ra = slow_time_profiles_to_ra_complex(planes.real.float(), planes.imag.float(), radar=radar)
        per_unit = _per_unit_plane_weights(torch.tensor([bins]), L, "cpu")
        def rel(a, b):
            a = torch.as_tensor(a).to(torch.complex128); b = torch.as_tensor(b).to(torch.complex128)
            return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))
        checks = {
            "range profiles: measured (window + FFT) vs rendered (PSF)": rel(measured_profiles, rendered_t),
            "loop RA: measured beamformer vs rendered beamformer": rel(rendered_ra, measured_ra),
            "Doppler cube: rendered vs measured": rel(rendered_doppler, measured_doppler),
            "Doppler planes (plane weights) vs measured Doppler bins": rel(
                planes_ra, measured_doppler[bins]),
            "per-unit plane weights vs doppler_plane_weights (bitwise)": float(
                torch.equal(torch.complex(per_unit[0][0], per_unit[1][0]).to(torch.complex64),
                            doppler_plane_weights(bins, loops=L)) is False),
        }
        if dataset.name == "hupr":
            from_adc = adc_frame_to_radproposer_tensor(adc_t)
            from_profiles = range_profiles_to_radproposer_tensor(measured_profiles)
            checks["RadProposer tensor: from ADC vs from range profiles"] = rel(from_profiles, from_adc)
        # the profile-domain window (the nuisance draws) against windowing the ADC samples
        random_adc = torch.randn(L, radar.n_tx, radar.n_rx, K, dtype=torch.complex128)
        via_adc = torch.fft.fft(apply_fft_window(random_adc, "range", -1), dim=-1)
        via_profiles = range_window_profiles(torch.fft.fft(random_adc, dim=-1))
        checks["range window in the profile domain (3-tap) vs on the ADC samples"] = rel(via_profiles, via_adc)
        # the scene reference: a resolved target's windowed peak over the window's coherent gain is its unwindowed peak
        recovered = unwindowed_range_reference(rendered_t, 1.0)
        unwindowed_peak = float(torch.fft.fft(torch.from_numpy(tone), dim=-1).abs().max())
        checks["unwindowed reference from a windowed point target vs its unwindowed peak"] = float(
            abs(recovered - unwindowed_peak) / unwindowed_peak)
        # the one approximation: the 15-bin PSF truncation, against the untruncated response
        truncated = psf.copy()
        peak = int(round(args.range_bin))
        keep = np.abs(np.arange(K) - peak) <= 7
        truncated[~keep] = 0.0
        checks["15-bin PSF truncation vs untruncated (the 3DPS approximation)"] = float(
            np.abs(truncated - psf).max() / np.abs(psf).max())
        report["radars"][radar.name] = checks
        for name, value in checks.items():
            loose = "truncation" in name or "unwindowed reference" in name    # sample 0 is unrecoverable: ~1/256
            flag = "" if (value <= args.tolerance or loose) else "   <-- FAIL"
            print(f"{radar.name:11s} {name:66s} {value:.3e}{flag}")
            if not loose:
                worst = max(worst, value)
    verdict = "PASS" if worst <= args.tolerance else "FAIL"
    print(f"window-check [{report['fft_window']}]: worst chain difference {worst:.3e} "
          f"(tolerance {args.tolerance:.0e}) -> {verdict}")
    print(json.dumps(report, indent=1))
    return 0 if verdict == "PASS" else 1


def _entry_doppler_check(argv: list[str]) -> int:
    """The Doppler-plane kernel gate: DFT and closed-form routes against the float64 reference.

    Synthetic paths (rates spanning every Doppler bin, exact bin centres and
    the aliasing edge included), the trainer's per-unit plane layout with
    pads: the forward amplitudes and the rate gradient from
    `hupr_doppler_plane_gather_forward` / `..._splat_backward`, on the DFT
    route (the 64-loop sum) and on the closed-form route
    (doppler_closed_form.cuh), are each compared with
    `signal_model.doppler_plane_amplitudes_reference` in float64. The
    closed form must be at least as close to the reference as the DFT
    route is; both are reported. Also: the two routes' rendered planes and
    contribution gradients against each other, and the timing of each.
    """
    import math, torch
    from mm4DPS.forward_render import _ext, _per_unit_plane_weights
    from mm4DPS.signal_model import (SLOW_TIME_PHASE_SIGN, doppler_plane_amplitudes_reference,
                                     fft_window_kind, tdm_sample_times)
    parser = argparse.ArgumentParser(prog="mm4DPS.qa doppler-check", description=_entry_doppler_check.__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="rtpose")
    parser.add_argument("--paths-per-unit", type=int, default=6000)
    parser.add_argument("--units", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=2e-6,
                        help="max |kernel - reference| / max |reference| the closed form must meet")
    parser.add_argument("--taps", type=int, default=15, help="the truncated route's taps to report")
    args = parser.parse_args(argv)
    dataset = Dataset.select(args.dataset, training_array=True)
    radar = dataset.radars[0]
    ext = _ext()
    dev = "cuda"
    torch.manual_seed(0)
    L = radar.loops
    lam = 299_792_458.0 / radar.carrier_hz
    k = SLOW_TIME_PHASE_SIGN * 2.0 * math.pi / lam
    times_np = tdm_sample_times(radar=radar)                          # (loops, n_tx)
    times = torch.from_numpy(times_np).float().to(dev)
    T = float(times_np[1, 0] - times_np[0, 0])
    v_bin = 2.0 * math.pi / (k * T * L)
    units, ppu = args.units, args.paths_per_unit
    paths = units * ppu
    # rates: uniform over +/- L/2 bins, plus exact bins, half bins, the edge and near-bin offsets
    special = torch.cat([torch.arange(-L // 2, L // 2).double(), torch.arange(-L // 2, L // 2).double() + 0.5,
                         torch.tensor([L / 2 - 1e-3, -L / 2 + 1e-3, L / 2, -L / 2]),
                         torch.arange(-4, 4).double() + 1e-4, torch.arange(-4, 4).double() + 1e-2,
                         torch.arange(-4, 4).double() - 3e-3])
    nus = torch.cat([torch.rand(paths - special.numel(), dtype=torch.float64) * L - L / 2, special])
    rates = (nus * v_bin).float().to(dev)
    tx = torch.randint(0, radar.n_tx, (paths,), device=dev)
    unit_of_path = torch.arange(units, device=dev).repeat_interleave(ppu)
    K = L - 1                                                          # the widest support: every non-DC bin
    bins = torch.full((units, K + 1), -1, dtype=torch.long, device=dev)
    non_dc = [d for d in range(L) if d != L // 2]
    widths = [K, K // 2, 12, 27][:units] + [K] * max(0, units - 4)
    for u, w in enumerate(widths):
        chosen = torch.tensor(non_dc, device=dev)[torch.randperm(K, device=dev)[:w]]
        bins[u, :w] = chosen
    spread = 15
    cells = 64 * 32
    base = units * cells
    c_re = torch.randn(spread, paths, device=dev); c_im = torch.randn(spread, paths, device=dev)
    flat = (torch.randint(0, cells, (spread, paths), device=dev) + unit_of_path[None, :] * cells).int()
    g_re = torch.randn((K + 2) * base, device=dev); g_im = torch.randn((K + 2) * base, device=dev)
    report = {"dataset": dataset.name, "fft_window": fft_window_kind(), "radar": radar.name}
    verdict = True
    for anchor in ("loop0", "dc-mean"):
        w_re, w_im = _per_unit_plane_weights(bins, L, dev, anchor)
        planes = w_re.shape[1]
        slot = torch.arange(bins.shape[1], device=dev)
        last = torch.where(bins >= 0, slot[None, :], slot.new_full((), -1)).amax(dim=1)
        active = (last + 2).to(torch.int32).contiguous()
        plane_bins = torch.cat([bins.new_full((units, 1), -1), bins], dim=1).to(torch.int32).contiguous()
        g_re_a, g_im_a = g_re[:planes * base].contiguous(), g_im[:planes * base].contiguous()
        A_ref, dA_ref = doppler_plane_amplitudes_reference(
            rates.double().cpu(), tx.cpu(), torch.from_numpy(times_np), k,
            bins[unit_of_path].cpu(), anchor, closed_form=True)
        results = {}
        # the window route's bin -> plane table, as forward_render builds it
        unsh = torch.where(bins >= 0, (bins + L // 2) % L, bins.new_full((), L))
        table = bins.new_full((units, L + 1), -1)
        table.scatter_(1, unsh, torch.arange(1, bins.shape[1] + 1, device=dev)[None, :].expand(units, -1))
        table[:, L] = -1
        table = table.to(torch.int32).contiguous()
        for route in ("dft", "closed-form", "closed-form-15tap"):
            kw = dict(unit_of_path=unit_of_path, active_planes=active)
            if route.startswith("closed-form"):
                kw.update(plane_bins=plane_bins, windowed=int(fft_window_kind() == "hann"),
                          anchor=int(anchor == "dc-mean"),
                          tx_times=torch.from_numpy(np.ascontiguousarray(np.stack(
                              [times_np[0], times_np[1] - times_np[0]], axis=1))).to(dev))
            if route == "closed-form-15tap":
                kw.update(bin_to_plane=table, taps=args.taps)
            def fwd():
                return ext.hupr_doppler_plane_gather_forward(
                    c_re, c_im, flat, tx, rates, times, w_re, w_im, k, base, 1, **kw)
            out_re, out_im, amps, _n = fwd()
            def bwd():
                return ext.hupr_doppler_plane_splat_backward(
                    g_re_a, g_im_a, c_re, c_im, flat, tx, rates, times, w_re, w_im, k, base, 1,
                    forward_amps=amps, **kw)
            gc_re, gc_im, g_rate = bwd()
            # timing
            for _ in range(3): fwd(); bwd()
            torch.cuda.synchronize()
            s0 = torch.cuda.Event(enable_timing=True); s1 = torch.cuda.Event(enable_timing=True)
            s0.record()
            for _ in range(20): o = fwd(); bwd()
            s1.record(); torch.cuda.synchronize()
            ms = s0.elapsed_time(s1) / 20
            if route == "closed-form-15tap":
                # the compact (paths, 3 + taps) row scattered back onto the
                # planes: entry 0 the anchor, 1..taps the window bins, the
                # two fold-only entries after; a zero entry deposits nothing
                W = 3 + args.taps
                row = torch.view_as_complex(amps.view(paths, W, 2)).cpu()
                A = torch.zeros(paths, planes, dtype=torch.complex128)
                n = _n.cpu().long()
                A[:, 0] = row[:, 0]
                tab = table.cpu().long()[unit_of_path.cpu()]           # (paths, L + 1)
                for w in range(1, W):
                    if w <= args.taps:
                        entry_bin = (n - args.taps // 2 + (w - 1)) % L
                    else:
                        entry_bin = torch.full_like(n, 1 if w == args.taps + 1 else L - 1)
                    plane = tab.gather(1, entry_bin[:, None])[:, 0]
                    ok = (plane >= 0) & (row[:, w] != 0)
                    A[ok.nonzero()[:, 0], plane[ok]] = row[ok, w].to(torch.complex128)
            else:
                A = torch.view_as_complex(amps.view(paths, planes, 2)).to(torch.complex128).cpu()
            # the reference rate gradient: sum_d Re(T_d dA_d), T_d from the same grad_out and contribs
            T = torch.zeros(paths, planes, dtype=torch.complex128)
            go = torch.complex(g_re_a, g_im_a).to(torch.complex128).cpu().view(planes, base)
            cc = torch.complex(c_re, c_im).to(torch.complex128).cpu()
            fl = flat.long().cpu()
            # the kernel's T_d = sum_s conj(grad_out[d, flat_s]) * contrib_s
            for sidx in range(spread):
                gsel = go[:, fl[sidx]].T                                 # (paths, planes)
                T += gsel.conj() * cc[sidx][:, None]
            g_rate_ref = (T * dA_ref).real.sum(dim=1)
            results[route] = {
                "amps_vs_reference": float((A - A_ref).abs().max() / A_ref.abs().max()),
                "rate_gradient_vs_reference": float((g_rate.double().cpu() - g_rate_ref).abs().max()
                                                    / g_rate_ref.abs().max()),
                "ms_forward_backward": ms,
                "_out": (out_re.clone(), out_im.clone(), gc_re.clone(), gc_im.clone(), g_rate.clone()),
            }
        a, b, w = results["dft"]["_out"], results["closed-form"]["_out"], results["closed-form-15tap"]["_out"]
        cross = {
            "planes closed-form vs dft": float((a[0] - b[0]).abs().max() / a[0].abs().max()),
            "grad_contrib closed-form vs dft": float((a[2] - b[2]).abs().max() / a[2].abs().max()),
            "rate_gradient closed-form vs dft": float((a[4] - b[4]).abs().max() / a[4].abs().max()),
            "planes 15-tap vs closed-form (the truncation)": float((w[0] - b[0]).abs().max() / b[0].abs().max()),
            "grad_contrib 15-tap vs closed-form": float((w[2] - b[2]).abs().max() / b[2].abs().max()),
            "rate_gradient 15-tap vs closed-form": float((w[4] - b[4]).abs().max() / b[4].abs().max()),
        }
        for route in results: del results[route]["_out"]
        report[anchor] = {"dft": results["dft"], "closed-form": results["closed-form"],
                          "closed-form-15tap": results["closed-form-15tap"], "cross": cross,
                          "planes": planes, "widths": widths, "paths": paths}
        cf, df = results["closed-form"], results["dft"]
        ok = (cf["amps_vs_reference"] <= max(args.tolerance, df["amps_vs_reference"])
              and cf["rate_gradient_vs_reference"] <= max(args.tolerance, df["rate_gradient_vs_reference"]))
        verdict = verdict and ok
        print(f"[{anchor}] planes {planes}, widths {widths}, paths {paths}")
        for route in ("dft", "closed-form", "closed-form-15tap"):
            r = results[route]
            print(f"  {route:18s} amps vs float64 {r['amps_vs_reference']:.3e}   rate gradient vs float64 "
                  f"{r['rate_gradient_vs_reference']:.3e}   forward+backward {r['ms_forward_backward']:.2f} ms")
        for name, value in cross.items():
            print(f"  {name:36s} {value:.3e}")
        print(f"  -> {'PASS' if ok else 'FAIL'}")
    print(f"doppler-check [{report['fft_window']}]: {'PASS' if verdict else 'FAIL'}")
    print(json.dumps(report, indent=1))
    return 0 if verdict else 1


def _entry_scene_check(argv: list[str]) -> int:
    """The fused scene-frames kernels against the per-frame contribution chain.

    Random scene paths (points x channels), a Hann PSF table, random carrier
    and fractional bins, a batch of frame gains: the fused forward
    (`hupr_static_scene_frames_forward`) against the reference built from
    `_psf_lerp_eager` + `_splat_contribs_eager` + a per-frame index_add, and
    the fused carrier gradient against autograd through that reference.
    """
    import argparse
    import json
    import time
    import torch
    from mm4DPS.forward_render import _ext, _gather_rows, _psf_table4, _scene_frames_geometry
    from mm4DPS.rasterizer import HannPSFTable, _psf_lerp_eager, _splat_contribs_eager
    parser = argparse.ArgumentParser(prog="mm4DPS.qa scene-check", description=_entry_scene_check.__doc__)
    parser.add_argument("--points", type=int, default=20000)
    parser.add_argument("--channels", type=int, default=144)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=6e-6,
                        help="max |fused - reference| / max |reference| (the fp32 floor)")
    args = parser.parse_args(argv)
    ext = _ext()
    dev = "cuda"
    torch.manual_seed(0)
    K, spread = 256, 15
    table = HannPSFTable(K, spread, n_grid=1024, device=dev)
    points, channels, frames = args.points, args.channels, args.frames
    paths = points * channels
    base = channels * K
    w = torch.rand(paths, device=dev) * 1e-3
    w[torch.rand(paths, device=dev) < 0.2] = 0.0          # inactive paths
    phi = torch.rand(paths, device=dev) * 6.2831853
    cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
    n_peak = torch.rand(paths, device=dev) * (K - 20) + 10
    n_floor = n_peak.floor().long()
    n_frac = (n_peak - n_floor.float()).contiguous()
    channel = torch.arange(paths, device=dev) % channels
    dn = torch.arange(-(spread // 2), spread // 2 + 1, device=dev)
    flat = (channel[None, :] * K + (n_floor[None, :] + dn[:, None]) % K).int().contiguous()
    point = torch.arange(paths, device=dev) // channels
    gains = torch.rand(frames, points, device=dev) * (torch.rand(frames, points, device=dev) > 0.3)

    # the reference chain, per frame
    w_ref = w.clone().requires_grad_(True)
    psf_r, psf_i = _psf_lerp_eager(table.psf_real, table.psf_imag, n_frac, table.n_grid)
    c_re, c_im = _splat_contribs_eager(w_ref, cos_phi, sin_phi, psf_r, psf_i)
    ref = []
    for f in range(frames):
        g = gains[f][point]
        out_re = torch.zeros(base, device=dev).index_add_(0, flat.long().reshape(-1), (c_re * g[None, :]).reshape(-1))
        out_im = torch.zeros(base, device=dev).index_add_(0, flat.long().reshape(-1), (c_im * g[None, :]).reshape(-1))
        ref.append(torch.stack([out_re, out_im], dim=-1))
    ref = torch.stack(ref)                                                  # (frames, base, 2)
    probe = torch.randn_like(ref)
    (ref * probe).sum().backward()
    grad_w_ref = w_ref.grad.clone()

    # the fused kernels
    components = {"flat_indices": flat, "point_indices": point, "output_shape": (1, channels, K)}
    geometry = _scene_frames_geometry(components, n_frac, K, {})
    assert geometry["base"] == base
    w_act = w * (w > 1e-20).float()
    carrier = torch.stack([w_act * cos_phi, w_act * sin_phi], dim=-1).contiguous()
    gains_t = gains.t().contiguous()
    probe_t = probe.transpose(0, 1).contiguous()
    def fwd():
        carrier_s = carrier.index_select(0, geometry["order32"])
        gains_s = _gather_rows(gains_t, geometry["point_sorted32"])
        return ext.hupr_static_scene_frames_forward(
            geometry["runs"], geometry["entry"], carrier_s, table4, gains_s, base, K)
    table4 = _psf_table4(table.psf_real, table.psf_imag)
    bwd = lambda: ext.hupr_static_scene_frames_backward(
        probe_t, geometry["entry"], table4, gains_t, K)
    out = fwd()
    grad_carrier = bwd()
    grad_w = (grad_carrier[:, 0] * cos_phi + grad_carrier[:, 1] * sin_phi) * (w > 1e-20).float()
    def clock(fn):
        fn(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / 10 * 1e3
    ms_forward, ms_backward = clock(fwd), clock(bwd)
    ms = ms_forward + ms_backward
    rel = lambda a, b: float((a - b).abs().max() / b.abs().max())
    report = {
        "points": points, "channels": channels, "frames": frames, "paths": paths,
        "forward_vs_reference": rel(out, ref),
        "carrier_gradient_vs_autograd": rel(grad_w, grad_w_ref),
        "ms_forward": ms_forward, "ms_backward": ms_backward,
        "repeat_bitwise": bool(torch.equal(out, fwd())),
    }
    ok = (report["forward_vs_reference"] <= args.tolerance
          and report["carrier_gradient_vs_autograd"] <= args.tolerance and report["repeat_bitwise"])
    print(f"scene-check: forward {report['forward_vs_reference']:.3e}  gradient "
          f"{report['carrier_gradient_vs_autograd']:.3e}  {ms_forward:.2f} + {ms_backward:.2f} ms  "
          f"repeat bitwise {report['repeat_bitwise']}  -> {'PASS' if ok else 'FAIL'}")
    print(json.dumps(report, indent=1))
    return 0 if ok else 1


def _entry_sorted_check(argv: list[str]) -> int:
    """The sorted Doppler route against the slot route on synthetic paths.

    Random paths (points x channels x units) with a carrier, fractional and
    integer range bins and Doppler rates; the slot route is fed the
    materialised contributions (carrier x range PSF lerp) and the window
    amplitudes, the sorted route the carrier and bins. Planes, the carrier
    gradient (the slot route's contribution gradient folded through the PSF)
    and the rate gradient are compared at the fp32 floor.
    """
    import argparse
    import json
    import math
    import time
    import torch
    from mm4DPS.forward_render import (_ext, _gather_rows, _per_unit_plane_weights, _psf_table4,
                                       sorted_doppler_order)
    from mm4DPS.rasterizer import HannPSFTable, _psf_lerp_eager, _splat_contribs_eager
    from mm4DPS.signal_model import SLOW_TIME_PHASE_SIGN, fft_window_kind, tdm_sample_times
    parser = argparse.ArgumentParser(prog="mm4DPS.qa sorted-check", description=_entry_sorted_check.__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="rtpose")
    parser.add_argument("--points", type=int, default=3000)
    parser.add_argument("--units", type=int, default=4)
    parser.add_argument("--taps", type=int, default=7)
    parser.add_argument("--tolerance", type=float, default=6e-6)
    args = parser.parse_args(argv)
    dataset = Dataset.select(args.dataset, training_array=True)
    radar = dataset.radars[0]
    ext = _ext()
    dev = "cuda"
    torch.manual_seed(0)
    L = radar.loops
    n_tx, n_rx, K = radar.n_tx, radar.n_rx, 256
    channels = n_tx * n_rx
    lam = 299_792_458.0 / radar.carrier_hz
    k = SLOW_TIME_PHASE_SIGN * 2.0 * math.pi / lam
    times_np = tdm_sample_times(radar=radar)
    T = float(times_np[1, 0] - times_np[0, 0])
    v_bin = 2.0 * math.pi / (k * T * L)
    tx_times = torch.from_numpy(np.ascontiguousarray(np.stack(
        [times_np[0], times_np[1] - times_np[0]], axis=1))).to(dev)
    units, points = args.units, args.points
    M = units * points
    paths = M * channels
    spread = 15
    table = HannPSFTable(K, spread, n_grid=1024, device=dev)
    # per-path geometry
    point = torch.arange(paths, device=dev) // channels
    channel = torch.arange(paths, device=dev) % channels
    tx = channel // n_rx
    unit = point // points
    n_peak = torch.rand(M, device=dev) * 60 + 20                     # a person's range bins
    n_peak = (n_peak[:, None] + torch.rand(M, channels, device=dev) * 2).reshape(-1)
    n_floor = n_peak.floor().long()
    n_frac = (n_peak - n_floor.float()).contiguous()
    w = torch.rand(paths, device=dev) * 1e-3
    w[torch.rand(paths, device=dev) < 0.1] = 0.0
    phi = torch.rand(paths, device=dev) * 6.2831853
    cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
    nus = torch.randn(M, device=dev) * 6
    rates = (nus[:, None].expand(-1, channels).reshape(-1) * v_bin).float().contiguous()
    base = channels * K
    dn = torch.arange(-(spread // 2), spread // 2 + 1, device=dev)
    flat = (channel[None, :] * K + (n_floor[None, :] + dn[:, None]) % K + unit[None, :] * base).int().contiguous()
    # the supported bins per unit: every non-DC bin (the widest support)
    non_dc = torch.tensor([d for d in range(L) if d != L // 2], device=dev)
    bins = non_dc[None, :].expand(units, -1).contiguous()
    anchor = "loop0"
    w_re, w_im = _per_unit_plane_weights(bins, L, dev, anchor)
    planes = w_re.shape[1]
    active = torch.full((units,), planes, dtype=torch.int32, device=dev)
    plane_bins = torch.cat([bins.new_full((units, 1), -1), bins], dim=1).to(torch.int32).contiguous()
    unsh = (bins + L // 2) % L
    b2p = bins.new_full((units, L + 1), -1)
    b2p.scatter_(1, unsh, torch.arange(1, bins.shape[1] + 1, device=dev)[None, :].expand(units, -1))
    b2p[:, L] = -1
    b2p = b2p.to(torch.int32).contiguous()
    windowed = int(fft_window_kind() == "hann")
    times = torch.from_numpy(times_np).float().to(dev)

    # the slot route on the materialised contributions
    w_ref = w.clone().requires_grad_(True)
    rates_ref = rates.clone().requires_grad_(True)
    psf_r, psf_i = _psf_lerp_eager(table.psf_real, table.psf_imag, n_frac, table.n_grid)
    c_re, c_im = _splat_contribs_eager(w_ref, cos_phi, sin_phi, psf_r, psf_i)
    from mm4DPS.forward_render import _DopplerPlaneSplat
    out_re, out_im = _DopplerPlaneSplat.apply(
        c_re, c_im, flat, tx, rates_ref, times, w_re, w_im, k, units * base,
        unit, active, plane_bins, windowed, 0, tx_times, b2p, args.taps)
    ref = torch.stack([out_re, out_im], dim=-1)
    probe = torch.randn_like(ref)
    (ref * probe).sum().backward()
    grad_w_ref, grad_rates_ref = w_ref.grad.clone(), rates_ref.grad.clone()

    # the sorted route
    from mm4DPS.forward_render import _DopplerPlaneSorted
    path_of_entry, runs = sorted_doppler_order(n_floor, channel, unit, K, units)
    w_s = w.clone().requires_grad_(True)
    rates_s = rates.clone().requires_grad_(True)
    w_act = w_s * (w_s > 1e-20).to(w_s.dtype)
    carrier = torch.stack([w_act * cos_phi, w_act * sin_phi], dim=-1)
    record_static = torch.stack([n_frac, n_floor.to(torch.int32).view(torch.float32)], dim=-1)
    def fwd():
        return _DopplerPlaneSorted.apply(
            carrier, rates_s, record_static, path_of_entry, runs, tx, unit, active, b2p,
            windowed, 0, tx_times, args.taps, _psf_table4(table.psf_real, table.psf_imag),
            planes, base, K, spread, L, k, channels)
    o_re, o_im = fwd()
    out = torch.stack([o_re, o_im], dim=-1)
    (out * probe).sum().backward()
    rel = lambda a, b: float((a - b).abs().max() / b.abs().max())
    report = {
        "paths": paths, "units": units, "planes": planes, "taps": args.taps,
        "planes_vs_slot_route": rel(out, ref),
        "carrier_gradient_vs_slot_route": rel(w_s.grad, grad_w_ref),
        "rate_gradient_vs_slot_route": rel(rates_s.grad, grad_rates_ref),
        # shared-memory atomics: the warps' order is not fixed, so a repeat
        # agrees to the fp32 floor, not bitwise (as the default slot route)
        "repeat_vs_repeat": rel(out, torch.stack(list(fwd()), dim=-1)),
    }
    def clock(fn):
        fn(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / 10 * 1e3
    with torch.no_grad():
        report["ms_sorted_forward"] = clock(fwd)
        report["ms_slot_forward"] = clock(lambda: _DopplerPlaneSplat.apply(
            c_re.detach(), c_im.detach(), flat, tx, rates, times, w_re, w_im, k, units * base,
            unit, active, plane_bins, windowed, 0, tx_times, b2p, args.taps))
    ok = (report["planes_vs_slot_route"] <= args.tolerance
          and report["carrier_gradient_vs_slot_route"] <= args.tolerance
          and report["rate_gradient_vs_slot_route"] <= args.tolerance
          and report["repeat_vs_repeat"] <= args.tolerance)
    print(f"sorted-check [{args.taps} taps]: planes {report['planes_vs_slot_route']:.3e}  carrier grad "
          f"{report['carrier_gradient_vs_slot_route']:.3e}  rate grad {report['rate_gradient_vs_slot_route']:.3e}  "
          f"forward {report['ms_sorted_forward']:.2f} ms (slot route {report['ms_slot_forward']:.2f})  "
          f"repeat {report['repeat_vs_repeat']:.1e}  -> {'PASS' if ok else 'FAIL'}")
    print(json.dumps(report, indent=1))
    return 0 if ok else 1


COMMANDS = {
    "sorted-check": _entry_sorted_check,
    "scene-check": _entry_scene_check,
    "doppler-check": _entry_doppler_check,
    "window-check": _entry_window_check,
    "orphans": _entry_orphans,
    "check-environment": _entry_check_environment,
    "environment": _entry_environment,
    "radar-config": _entry_radar_config,
    "cascade-config": _entry_cascade_config,
    "orientation-check": _entry_orientation_check,
}


def main() -> int:
    parser = argparse.ArgumentParser(prog="mm4DPS.qa", description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS))
    args, rest = parser.parse_known_args()
    return COMMANDS[args.command](rest)


if __name__ == "__main__":
    raise SystemExit(main())