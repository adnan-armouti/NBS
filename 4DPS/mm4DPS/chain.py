"""The chain, in Python. Phase A of docs/LEAN_ACTION_PLAN.md.

`scripts/` was 1,318 lines of bash orchestrating ~15 `python -m` entry points.
Every one of those spawns an interpreter and goes through an argparse layer that
exists only for CLI dispatch, and that layer is a large share of the statements
the acceptance run never executes. This module replaces the bash stage by stage;
each stage lands behind the acceptance gate, and the shell script it replaces
becomes a thin delegator until the last one can be deleted.

Ported so far:
  setup_workspace   scripts/setup_workspace.sh
  TrainerFlags      scripts/vf_flags.sh
  seeds             scripts/build_hupr_seeds.sh
  reference_sequence  scripts/build_reference_sequence.sh
  person_stack      scripts/make_gvhmr_person_workspace.sh
  treatment         scripts/make_treatment_workspace.sh
  RenderRequest     scripts/run_hupr_v23_angle_deconvolution_ablation_132.sh

WHY THE HEAVY STAGES STILL SPAWN A PROCESS. Calling every stage in-process would
remove more argparse layer, and it would also put the trainer's ~13 GiB peak, the
background fit and two concurrent renders in ONE interpreter. The chain relies on
a stage's GPU memory being returned when its process exits. So `_stage` runs CUDA
work as a subprocess and light work in-process, and the reclaim of the CLI layer
is limited to the modules that are genuinely called in-process. Trading an OOM
for a few hundred statements would be a bad deal.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mm4DPS import Dataset, SequenceDir, Workspace, current_dataset
from mm4DPS.signal_model import fft_window_kind

REPO = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------
# setup_workspace: outputs/<run>/, empty but for the tracked configs.
# ----------------------------------------------------------------------
def setup_workspace(run: str, repo: Path | None = None) -> Path:
    """Create `outputs/<run>/` with the layout and the radar configs.

    Nothing is installed from `seeds/` any more: every calibration a run
    uses -- the scene's alignment and materials, the antenna coupling, the
    person's placement, the receiver gain and floor, the joint gains -- is
    computed from the recording inside the chain or learned by the trainer
    (docs/SEEDS.md). The one provided input, the GVHMR vertex dump, is read
    by the person stage from `gvhmr_vertices`.
    """
    root = repo or REPO
    space = Workspace.for_run(run, root / "outputs")
    workspace = space.root
    for name in Workspace.LAYOUT:
        (workspace / name).mkdir(parents=True, exist_ok=True)

    configs = workspace / "configs"
    if configs.is_symlink() or configs.exists():
        configs.unlink() if configs.is_symlink() else shutil.rmtree(configs)
    configs.symlink_to(root / "configs")
    return workspace


GVHMR_JOBS: dict = {}          # output path -> the running GVHMR process (`start_gvhmr`)


def gvhmr_output(person_workspace: Path, sequence: int) -> Path:
    return person_workspace / "gvhmr" / f"seq{sequence}.npz"


def start_gvhmr(sequence: int, person_workspace: Path) -> None:
    """Launch GVHMR for the sequence now, in the background, on this run's GPU.

    The chain calls this before the reference stage, so GVHMR's ~9 s of
    model loading and most of its ~30 s of work overlap the scaffold, the
    background fit and the coupling; the person stage collects the result.
    """
    from mm4DPS import gvhmr
    output = gvhmr_output(person_workspace, sequence)
    if output.is_file() or output in GVHMR_JOBS:
        return
    GVHMR_JOBS[output] = gvhmr.start(current_dataset(), sequence, output, output.with_suffix(".log"))


def gvhmr_vertices(sequence: int, workspace: Path, log: Path | None = None) -> Path:
    """The person's SMPL vertices per camera frame, from GVHMR, computed for this run.

    Collects the process `start_gvhmr` launched (or runs GVHMR now when the
    person stage is called on its own). `tools/fast_dump.py` in the GVHMR
    environment: the frames in memory, the two ViTs in half precision, ~30 s
    a sequence; the mesh is a deterministic function of the frames.
    Written to `<workspace>/gvhmr/seq<S>.npz` and kept with the run's
    products (`gvhmr_vertices.npz`). Nothing is read from `seeds/`.
    """
    from mm4DPS import gvhmr
    output = gvhmr_output(workspace, sequence)
    if output.is_file():
        return output
    started = time.time()
    if output not in GVHMR_JOBS:
        start_gvhmr(sequence, workspace)
    gvhmr.finish(GVHMR_JOBS.pop(output), output)
    note(f"GVHMR: waited {time.time() - started:.1f} s for the mesh", log)
    return output


def _replace_tree(source: Path, target: Path) -> None:
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


# ----------------------------------------------------------------------
# TrainerFlags: scripts/vf_flags.sh
# ----------------------------------------------------------------------
PRODUCTION_TRAINER_FLAGS = (
    "--epochs", "400",
    "--training-frames", "600",
    "--train-all-frames",
    "--eval-every", "10",
    "--lr-schedule", "cosine",
    "--plane-target-dtype", "float16",
    "--doppler-support-snr", "1.0",
)


@dataclass
class TrainerFlags:
    """The trainer's flags, composed the way scripts/vf_flags.sh composed them.

    `replace` is a FOOTGUN and it has fired: setting it to add one flag silently
    drops the other seven, and the run then trains a different configuration
    while looking like production. Nothing downstream can tell -- the trainer did
    exactly what it was asked. So a replacement announces what it dropped.

    `append` adds and keeps the defaults. Use it unless you genuinely mean to
    discard the production configuration; a replacement is still allowed, because
    an ablation legitimately wants one.
    """

    replace: str | None = None
    append: str | None = None
    notes: list[str] = field(default_factory=list)

    def compose(self) -> list[str]:
        if self.replace:
            composed = self.replace.split()
            stated = set(composed)
            dropped = [flag for flag in PRODUCTION_TRAINER_FLAGS
                       if flag.startswith("--") and flag not in stated]
            if dropped:
                self.notes.append(
                    "VF_EXTRA REPLACES the production flags. Dropped unless you "
                    "re-stated them:")
                self.notes.extend(f"    {flag}" for flag in dropped)
                self.notes.append(
                    "  (append adds flags instead of replacing them)")
        else:
            composed = list(PRODUCTION_TRAINER_FLAGS)
        return composed + (self.append.split() if self.append else [])

    def as_string(self) -> str:
        return " ".join(self.compose())


# ----------------------------------------------------------------------
# Stage plumbing
# ----------------------------------------------------------------------
PYTHON = os.environ.get("PY", "/home/adnan/.conda/envs/hupr/bin/python")

# hupr_env.sh's job, done from this checkout rather than from a hardcoded path.
from mm4DPS import apply as _apply
_apply()


def note(message: str, log: Path | None = None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as handle:
            handle.write(line + "\n")


# The two material fits of the preprocessing are initialisations -- the
# velocity trainer fits the scene's and the person's materials on for 400
# epochs -- so their lengths are the chain's to set: 75 iterations (the
# fit is 30 ms an iteration on two boards) and one epoch. 150/1 was gated
# at 400 epochs against 500/3 with the learned gain (183 RA 0.879 ->
# 0.884, RD 0.650 -> 0.654, HuPR 256 equal or better); 75 rides the same
# gate (docs/FIDELITY_PLAN.md 6-8).
BACKGROUND_ITERS = int(os.environ.get("MM4DPS_BACKGROUND_ITERS", "75"))
PERSON_EPOCHS = int(os.environ.get("MM4DPS_PERSON_EPOCHS", "1"))


def run_module(module: str, *argv: str, repo: Path | None = None,
               log: Path | None = None, capture: bool = False,
               workspace: Path | None = None) -> str:
    """One stage, as its own process.

    CUDA work goes through here rather than being called in-process: the chain
    relies on a stage's GPU memory being released when the process exits.

    `workspace` sets HUPR_WORKSPACE for the child. It is NOT optional in
    practice: several stages resolve their own defaults through
    `mm4DPS.WORKSPACE`, and `inherit-joint-gains` reads its seed
    checkpoint from `<workspace>/training/...`. The shell scripts exported it
    and dropping it sends those stages to `outputs/workspace`.
    """
    root = repo or REPO
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{root}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(root)
    if workspace is not None:
        env["HUPR_WORKSPACE"] = str(workspace)
    # torch.compile's parallel compile-worker pool hangs for 300 s at process
    # exit when the process's stdout is a file (which every stage's is, here):
    # the person material trainer took 305 s under this launcher against 5 s
    # by hand (measured 2026-09-12, `timeout -s ABRT` dump: training done,
    # main thread in the atexit shutdown). One compile thread: no pool, the
    # few small graphs a stage compiles cost nothing extra.
    env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    command = [PYTHON, "-m", module, *[str(a) for a in argv]]
    if capture:
        done = subprocess.run(command, cwd=root, env=env, check=True,
                              capture_output=True, text=True)
        return done.stdout
    stream = None
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open("a")
    try:
        subprocess.run(command, cwd=root, env=env, check=True,
                       stdout=stream, stderr=subprocess.STDOUT if stream else None)
    finally:
        if stream is not None:
            stream.close()
    return ""


# ----------------------------------------------------------------------
# reference_scene: the scene from the recording
# ----------------------------------------------------------------------
def reference_scene(sequences, workspace: Path, repo: Path | None = None,
                    frames: str = "0,150,300,450", hfov_deg: float | None = None,
                    log: Path | None = None) -> None:
    """Scaffold, background material model and antenna coupling per sequence.

    Steps are skipped when their outputs exist, so a run resumes -- that is
    load-bearing for restarts and is preserved exactly.
    """
    root = repo or REPO
    from mm4DPS import DEPTH_ANYTHING_PATH

    dataset = current_dataset()
    (workspace / "configs").mkdir(parents=True, exist_ok=True)
    for config in (radar.config for radar in dataset.radars):
        target = workspace / "configs" / config
        if not target.is_file():
            source = root / "configs" / config
            if not source.is_file():
                raise FileNotFoundError(f"missing tracked config {source}")
            shutil.copy(source, target)
            note(f"seeded {target}", log)

    depth = Path(os.environ.get("DEPTH_ANYTHING_PATH", str(DEPTH_ANYTHING_PATH)))
    if dataset.scene_source == "depth" and not depth.is_dir():
        raise FileNotFoundError(f"depth model not found: {depth}")

    for sequence in sequences:
        scaffold_dir = workspace / "scaffolds" / f"seq{sequence}"
        if (scaffold_dir / "pcl.npy").is_file():
            note(f"[seq {sequence}] scaffold already present, keeping it", log)
        elif dataset.scene_source == "lidar":
            note(f"[seq {sequence}] scaffold from the LiDAR", log)
            # MM4DPS_SCAFFOLD_PERSON_SEQUENCES=<dir with {s}>: a registered person sequence per sequence for the per-frame
            # mesh cut (reconstruction lidar-scaffold --person-sequence-dir); without it the scaffold-cut sweep follows later
            person_seq = os.environ.get("MM4DPS_SCAFFOLD_PERSON_SEQUENCES")
            person_args = (["--person-sequence-dir", person_seq.format(s=sequence)]
                           if person_seq and Path(person_seq.format(s=sequence)).is_dir() else [])
            run_step("mm4DPS.reconstruction", "lidar-scaffold",
                     "--sequence", sequence, "--output-dir", scaffold_dir,
                     "--repo", root, *person_args, repo=root, log=log, workspace=workspace)
        else:
            note(f"[seq {sequence}] scaffold from frames {frames}", log)
            samples = ",".join(f"{sequence}:{f}" for f in frames.split(","))
            run_step("mm4DPS.reconstruction", "scaffold",
                       "--samples", samples, "--hfov-deg", (dataset.camera_hfov_deg if hfov_deg is None else hfov_deg),   # the dataset's camera (HuPR 52.8, radar-estimated); the workspace names keep their historical 'hfov40' suffix
                       "--dataset", dataset.name, "--model-path", depth,
                       "--output-dir", scaffold_dir, repo=root, log=log,
                       workspace=workspace)

        background = workspace / "training" / f"seq{sequence}_background_hfov40"
        if (background / "best_model.pt").is_file():
            note(f"[seq {sequence}] background model already present, keeping it", log)
        else:
            note(f"[seq {sequence}] background material fit", log)
            fit_background(sequence, workspace, repo=root, log=log)
        result = json.loads((background / "result.json").read_text())
        note(f"[seq {sequence}] background loss {result['best_loss']:.4f} "
             f"({result['iterations']} iters, {result['elapsed_seconds']:.0f} s, "
             f"{result[dataset.names[0] + '_active_points']} active points)", log)

        # The antenna coupling from this sequence's own recording: the
        # coherent mean of the near range bins per virtual channel, added to
        # the rendered profiles at render time. (HuPR's used to be a seed
        # fitted on held-out sequences; every dataset now fits its own.)
        coupling_dir = workspace / "calibration" / "antenna_coupling"
        if not coupling_files(coupling_dir, dataset):
            note(f"[seq {sequence}] antenna coupling from the recording", log)
            run_step("mm4DPS.calibration", "sequence-coupling", "--dataset", dataset.name,
                       "--sequence", sequence, "--output-dir", coupling_dir,
                       repo=root, log=log, workspace=workspace)
            coupling = json.loads((coupling_dir / "result.json").read_text())["radars"]
            note(f"[seq {sequence}] antenna coupling: " + ", ".join(
                f"{r}: coherent {v['coherent_fraction_of_near_power']:.4f}" for r, v in coupling.items()), log)

        # (no receiver-gain fit here: the velocity trainer learns the gain
        # with the materials, from the measured/rendered RA level of its
        # first frames -- `train_motion absolute-gain` remains a diagnostic)


def reference_sequence(run: str, sequence: int, repo: Path | None = None) -> None:
    """Preprocessing: raw ADC -> the scene the person stack inherits."""
    root = repo or REPO
    workspace = root / "outputs" / run
    if not workspace.is_dir():
        raise FileNotFoundError(
            f"workspace {workspace} missing; run setup_workspace first")
    log = workspace / "deliverables" / f"chain_seq{sequence}.log"
    note(f"chain start: run={run} seq={sequence}", log)
    note("reference scene", log)
    reference_scene([sequence], workspace, repo=root, log=log)
    note(f"reference sequence DONE: run={run} seq={sequence}", log)


# ----------------------------------------------------------------------
# person_stack: scripts/make_gvhmr_person_workspace.sh
# ----------------------------------------------------------------------
PERSON_INHERITS = (
    "training/seq{s}_background_hfov40",
    "scaffolds/seq{s}",
)


def _persistent_geometry(sequence_dir: Path, probe: int = 40) -> bool:
    """Does this cloud carry point identity across frames?

    The monocular depth cloud changes length frame to frame, so it cannot hold a
    per-point material or trained per-point normals and the trainer rejects it.
    The GVHMR SMPL mesh is a fixed 6890 points. Checked by length, on the first
    `probe` frames, exactly as the shell did.
    """
    import numpy as np
    frames = SequenceDir(sequence_dir).frames()[:probe]
    if not frames:
        return False
    return len({int(np.load(f)["pcl"].shape[0]) for f in frames}) == 1


def scaffold_cut_enabled() -> bool:
    """SCAFFOLD_CUT=off keeps the annotation-cut scaffold (the pre-2b chain), for A/B runs."""
    return os.environ.get("SCAFFOLD_CUT", "on").lower() not in ("off", "0", "false")


def fit_background(sequence: int, workspace: Path, repo: Path | None = None,
                   log: Path | None = None) -> None:
    """The background material model fitted on the workspace's scaffold (its static |RA| targets).

    An initialisation: the velocity trainer keeps fitting the scene's
    materials and normals for its 400 epochs (`background_groups`), so the
    fit's length is set by the chain (BACKGROUND_ITERS), not by the fitter's
    default. Run by reference_scene on the scaffold and by person_stack
    again after the scaffold is cut with the registered person.
    """
    root = repo or REPO
    dataset = current_dataset()
    scaffold_dir = workspace / "scaffolds" / f"seq{sequence}"
    background = workspace / "training" / f"seq{sequence}_background_hfov40"
    if background.is_dir():
        shutil.rmtree(background)
    run_step("mm4DPS.train_motion", "dual-radar", "--dataset", dataset.name,
             *[part for r in dataset.radars for part in (
                 r.flag("", "config"), workspace / "configs" / r.config,
                 r.flag("", "target"), scaffold_dir / f"{r.name}_static_ra.npy")],
             "--pcl", scaffold_dir / "pcl.npy", "--iters", BACKGROUND_ITERS,
             "--output-dir", background, repo=root, log=log, workspace=workspace)
    result = json.loads((background / "result.json").read_text())
    note(f"[seq {sequence}] background loss {result['best_loss']:.4f} "
         f"({result['iterations']} iters, {result['elapsed_seconds']:.0f} s, "
         f"{result[dataset.names[0] + '_active_points']} active points)", log)


def person_stack(sequence: int, base: str, run: str,
                 repo: Path | None = None, person_extra: str = "") -> Path:
    """Build the person sequence from the GVHMR dump and fit the person material on it.

    The sequence is registered to the radar frame by frame as it is built
    (`reconstruction person-sequence`: the radar's moving-energy range over
    GVHMR's centroid range, a scale about the camera origin), so it gets no
    translation and no range sweep; then one material pass over all frames
    and the motion detection.
    """
    root = repo or REPO
    scene = root / "outputs" / base
    if not scene.is_dir():
        raise FileNotFoundError(
            f"no baseline workspace {scene} to inherit the scene from")

    workspace = setup_workspace(run, repo=root)
    log = workspace / "deliverables" / f"gvhmr_person_seq{sequence}.log"
    note(f"GVHMR person stack: seq={sequence} run={run} scene from {base}", log)
    vertices = gvhmr_vertices(sequence, workspace, log=log)

    for template in PERSON_INHERITS:
        name = template.format(s=sequence)
        target, source = workspace / name, scene / name
        if not target.exists() and source.exists():
            shutil.copytree(source, target)

    person = workspace / f"person_sequence_{sequence}_v2clean"
    if not (person / "metadata.json").is_file():
        note(f"person sequence from the GVHMR dump, registered to the radar per frame", log)
        # the registration in the camera's angular sector (docs/FIDELITY_FIX_RESULTS.md 3.5) and the
        # azimuth registration to the radar (3.11) are the stage's defaults; REGISTRATION_SECTOR=off is
        # the pre-2026-09-16 search, REGISTRATION_AZIMUTH=off the camera's azimuth -- ablations
        sector = ["--registration-sector", os.environ.get("REGISTRATION_SECTOR", "on"),
                  "--registration-azimuth", os.environ.get("REGISTRATION_AZIMUTH", "on"),
                  "--depth-registration", os.environ.get("DEPTH_REGISTRATION", "translate")]
        run_step("mm4DPS.reconstruction", "person-sequence", "--dataset", current_dataset().name,
                 "--sequence", sequence, "--vertices", vertices,
                 "--scaffold-dir", workspace / f"scaffolds/seq{sequence}",
                 "--output-dir", person, "--repo", root, *sector, repo=root, log=log, workspace=workspace)
        scale = json.loads((person / "metadata.json").read_text())["gvhmr_conversion"]["depth_scale"]
        note(f"person range: depth scale median {scale['depth_scale']:.3f} "
             f"(IQR {scale['ratio_iqr'][0]:.3f}-{scale['ratio_iqr'][1]:.3f}), "
             f"{scale['confident_frames']}/{scale['frames']} frames detected, "
             f"{len(scale['flagged_frames'])} flagged", log)

    background = workspace / f"training/seq{sequence}_background_hfov40/best_model.pt"
    static_pcl = workspace / f"scaffolds/seq{sequence}/pcl.npy"
    extra = person_extra.split() if person_extra else []

    # docs/STATIC_SEQS_NEXT.md 2b: the scaffold's ghost of the person. The scaffold cut the person out about the dataset
    # annotation's column / bbox; where the annotation is off the person (RT-Pose 108 / 168 / 164: 0.45-0.6 m laterally)
    # the person's own LiDAR / depth points stayed in the scene and render as a scatterer the recording lacks. The
    # registered person is where the person is: its swept surface is cut out of this workspace's scaffold (once; the
    # uncut cloud is kept beside it), and the background is fitted again on the cut scaffold before the person fit.
    scaffold_meta = workspace / f"scaffolds/seq{sequence}/scaffold_metadata.json"
    cut_per_frame = scaffold_meta.is_file() and bool(json.loads(scaffold_meta.read_text()).get("person_cut"))
    if cut_per_frame:
        note(f"[seq {sequence}] scaffold already cut per frame by the registered mesh; the sweep is skipped", log)
    if scaffold_cut_enabled() and not cut_per_frame and not (workspace / f"scaffolds/seq{sequence}/scaffold_cut.json").is_file():
        cut = json.loads(run_step("mm4DPS.reconstruction", "scaffold-cut",
                                  "--scaffold-dir", workspace / f"scaffolds/seq{sequence}", "--sequence-dir", person,
                                  repo=root, log=log, capture=True, workspace=workspace))
        note(f"[seq {sequence}] scaffold cut with the registered person: {cut['removed']} of {cut['scaffold_points']} "
             f"points removed (radius {cut['radius_m']} m, {cut['frames_sampled']} frames)", log)
        fit_background(sequence, workspace, repo=root, log=log)

    # The person's placement is the registration's: no translation. The
    # file remains because the velocity stage and the shadow mask read the
    # placement from it (one source of truth for the number).
    translation = 0.0
    calibration_dir = workspace / f"calibration/seq{sequence}_person_range_v2"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    (calibration_dir / "result.json").write_text(json.dumps({
        "status": "completed",
        "protocol": "per-frame radar registration in the person sequence (radar_depth_scales); no sweep",
        "selected_geometry_translation_y_m": 0.0}, indent=2) + "\n")

    note(f"person material ({PERSON_EPOCHS} epochs, all frames)", log)
    run_step("mm4DPS.train_material", "person-sequence",
               "--sequence-dir", person, "--background-model", background,
               "--static-pcl", static_pcl,
               "--output-dir", workspace / f"training/seq{sequence}_person_v2",
               "--epochs", PERSON_EPOCHS, "--geometry-translation-y-m", translation,
               # Option B (docs/STATIC_SEQS_PLAN.md): the same solid-angle slot in the person stage, the trainer and the render
               "--person-solid-angle-weight", os.environ.get("PERSON_SOLID_ANGLE", "off"),
               "--scene-solid-angle", os.environ.get("SCENE_SOLID_ANGLE", "off"),
               *extra, repo=root, log=log, workspace=workspace)

    note("motion detection (pose-motion-frames deferred to the velocity stage)", log)
    run_step("mm4DPS.motion", "detect-person-motion",
             "--sequence", sequence, "--sequence-dir", person,
             "--translation-y-m", translation,
             "--output-dir", workspace / f"calibration/seq{sequence}_detect_v2",
             repo=root, log=log, workspace=workspace)

    import torch
    from mm4DPS.data import MeasuredReader
    MeasuredReader.release()          # the recording's frames kept in memory across the stages
    fitted = torch.load(
        workspace / f"training/seq{sequence}_person_v2/best_shared_human_material.pt",
        map_location="cpu", weights_only=False)
    note(f"DONE: person material "
         f"{tuple(fitted['shared_human_raw_material'].shape)} area scale "
         f"{float(fitted['shared_human_area_scale'])}", log)
    return workspace


def _run_in_process(module: str, command: str, argv: list[str],
                    log: Path | None = None, workspace: Path | None = None) -> str:
    """Call a stage's own entry point in this interpreter; returns its stdout.

    For the light steps (the fits of a few seconds, the calibrations, the
    masks): a process of their own paid 1.3-2.9 s of imports and CUDA
    initialisation each, ~25 s of the chain's first minute. The module's
    own parser builds the Namespace, so validation and defaults stay
    exactly where they were; `workspace` is exported as HUPR_WORKSPACE for
    the call (the module-level `mm4DPS.WORKSPACE` default is the chain's
    own, so every path a step needs is passed explicitly). The step's GPU
    memory is released after it (`torch.cuda.empty_cache`).
    """
    import importlib
    import contextlib
    import io
    target = importlib.import_module(module)
    entry = target.COMMANDS[command]
    buffer = io.StringIO()
    argv_held, env_held = sys.argv, os.environ.get("HUPR_WORKSPACE")
    try:
        if workspace is not None:
            os.environ["HUPR_WORKSPACE"] = str(workspace)
        sys.argv = [f"{module} {command}", *[str(a) for a in argv]]
        with contextlib.redirect_stdout(buffer):
            entry()
    finally:
        sys.argv = argv_held
        if env_held is None:
            os.environ.pop("HUPR_WORKSPACE", None)
        else:
            os.environ["HUPR_WORKSPACE"] = env_held
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    if log is not None and buffer.getvalue():
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as handle:
            handle.write(buffer.getvalue())
    return buffer.getvalue()


# The light steps run in the chain's interpreter (`_run_in_process`);
# MM4DPS_STAGE_PROCESS=1 sends them through a process each (the A/B).
_STAGE_PROCESS = os.environ.get("MM4DPS_STAGE_PROCESS", "0") == "1"


def run_step(module: str, command: str, *argv, repo: Path | None = None,
             log: Path | None = None, workspace: Path | None = None,
             capture: bool = False) -> str:
    """A light stage: in-process by default, its own process on request."""
    if _STAGE_PROCESS:
        return run_module(module, command, *argv, repo=repo, log=log, workspace=workspace,
                          capture=capture)
    return _run_in_process(module, command, list(argv), log=None if capture else log,
                           workspace=workspace)


# ----------------------------------------------------------------------
# treatment: scripts/make_treatment_workspace.sh
# ----------------------------------------------------------------------
TREATMENT_CALIBRATIONS = ("seq{s}_person_range_v2", "seq{s}_detect_v2")


def treatment(base: str, treat: str, sequence: int, repo: Path | None = None,
              person_source_run: str | None = None) -> Path:
    """The person sequence plus the person checkpoint fitted ON it.

    Every input is named explicitly. Discovering a person checkpoint by
    convention quietly mixes a scene from one workspace with a person fitted
    against another, and the symptom is not an error but a KeyError deep in
    training, on a frame whose normals span the checkpoint does not have.
    """
    root = repo or REPO
    scene = root / "outputs" / base
    target = root / "outputs" / treat
    if not scene.is_dir():
        raise FileNotFoundError(f"baseline workspace {scene} missing")
    if target.is_dir():
        raise FileExistsError(f"treatment workspace {target} already exists")

    source = None
    for candidate in [c for c in (person_source_run,
                                 f"gvhmr_person_{sequence}") if c]:
        path = root / "outputs" / candidate
        if (path / f"training/seq{sequence}_person_v2"
                / "best_shared_human_material.pt").is_file() and                 _persistent_geometry(path / f"person_sequence_{sequence}_v2clean"):
            source = path
            print(f"person checkpoint: {candidate} (fitted on the GVHMR mesh)")
            break
    if source is None:
        raise FileNotFoundError(
            f"sequence {sequence} has GVHMR geometry but no person checkpoint "
            "fitted on it. Run the person stage first.")

    setup_workspace(treat, repo=root)
    # the person sequence built (and registered) in the person workspace
    shutil.copytree(source / f"person_sequence_{sequence}_v2clean",
                    target / f"person_sequence_{sequence}_v2clean")
    _replace_tree(source / f"training/seq{sequence}_person_v2",
                  target / f"training/seq{sequence}_person_v2")
    for template in TREATMENT_CALIBRATIONS:
        name = template.format(s=sequence)
        if (source / "calibration" / name).is_dir():
            _replace_tree(source / "calibration" / name,
                          target / "calibration" / name)
    # the antenna coupling fitted from the recording travels with the scene
    coupling = scene / "calibration" / "antenna_coupling"
    if coupling.is_dir():
        _replace_tree(coupling, target / "calibration" / "antenna_coupling")
    (target / "person_source.txt").write_text(f"{source}\n")
    print(f"treatment workspace {target} ready (GVHMR person geometry, "
          f"checkpoint from {source.name}, seq {sequence})")
    return target


# ----------------------------------------------------------------------
# RenderRequest: scripts/run_hupr_v23_angle_deconvolution_ablation_132.sh
# ----------------------------------------------------------------------
# That script was ~40 environment variables translated into one very long argv.
# The env-var indirection WAS the renderer's interface, and it hid a defect for
# a long time: `SCENE_VISIBILITY` was simply absent from the list the velocity
# stage exported, so every product of a mask A/B recorded
# `per_frame_scene_visibility: {"enabled": false}` on BOTH arms (R099). A
# dataclass makes the same contract explicit and checkable.
#
# The renderer's nuisance processes -- phase noise, chirp phase, the frame
# and local additive nuisances, the seeded slow-time envelope, the diffuse
# floor -- were HuPR's fitted receiver model, constants fitted outside this
# repo on unwindowed data. They are OFF: the noise model is learned by the
# trainer (its floor starts from the sequence's own off-person cells) and
# the renderer draws its receiver floor from the checkpoint
# (`--receiver-floor-from-checkpoint`). Every dataset renders the same way.
_RECEIVER_MODEL_FIELDS = ("absolute_gain", "slow_time", "slow_time_rho", "scene_fraction",
                          "receiver_phase", "chirp_phase", "frame_nuisance", "local_nuisance",
                          "phase_drift", "frame_correlation", "local_correlation",
                          "empty_correlation", "fixed_fraction")
_NO_RECEIVER_MODEL = {key: 0.0 for key in _RECEIVER_MODEL_FIELDS} | {"absolute_gain": 1.0}
#: The per-radar request fields, and the receiver-model key each one overrides.
_PER_RADAR_FIELDS = ("absolute_gain", "slow_time", "slow_time_rho", "scene_fraction",
                     "receiver_phase", "chirp_phase", "frame_nuisance", "local_nuisance")


def receiver_model(dataset: Dataset) -> dict[str, dict]:
    """Zeros for every radar: no fitted receiver model; the floor is the trainer's."""
    return {radar.name: dict(_NO_RECEIVER_MODEL) for radar in dataset.radars}


@dataclass
class RenderRequest:
    """One render of one variant, as data instead of forty exported variables."""

    sequence: int
    sequence_dir: Path
    person_checkpoint: Path
    motion_checkpoint: Path
    product_dir: Path
    output_video: Path
    model_label: str
    translation_y: float
    pose_audit: Path
    motion_detection: Path
    workspace: Path
    frame_count: int = 132
    write_video: str = "on"
    scene_visibility: Path | None = None
    person_solid_angle: str = "off"
    dataset: str = "hupr"
    #: Per-radar overrides of the receiver model, ``{field: {radar: value}}``
    #: for the fields in ``_PER_RADAR_FIELDS``; anything not given is the
    #: dataset's fitted model (HuPR's numbers, or zeros).
    per_radar: dict = field(default_factory=dict)
    scene_smoothing_bins: int = 33
    diffuse_floor: float = 0.0             # HuPR's fitted diffuse floor is gone; the learned floor absorbs it
    diffuse_floor_calibration: Path | None = None
    velocity_gain: float = 1.0
    velocity_floor: float = 0.0
    velocity_mode: str = "checkpoint"
    velocity_diff: str = "savgol"
    slow_time_edge_blend: int = 9
    nuisance_seed_offset: int = 0
    nuisance_backend: str = "torch"
    nuisance_draws: str = "gpu"
    antenna_coupling: str = "on"
    antenna_coupling_dir: Path | None = None
    channel_calibration: Path | None = None
    product_chunk_frames: int | None = None
    scene_density_normalize: bool = False
    save_noise_diagnostic_products: bool = False
    radproposer_products: bool = True      # HuPR's published point proposals; HuPR only
    receiver_floor: str = "on"             # the trained floor from the checkpoint (off: the clean variant)
    device: str = "cuda:0"

    def radar_value(self, radar: str, key: str) -> float:
        """One radar's value of one receiver-model field, override first."""
        override = self.per_radar.get(key, {})
        if radar in override:
            return float(override[radar])
        return float(receiver_model(Dataset.named(self.dataset))[radar][key])

    def argv(self) -> list[str]:
        dataset = Dataset.named(self.dataset)
        g = self.workspace
        coupling_dir = self.antenna_coupling_dir or (g / "calibration/antenna_coupling")
        offset = self.nuisance_seed_offset
        v = self.radar_value
        argv = [
            "--dataset", dataset.name,
            "--sequence", str(self.sequence),
            "--sequence-dir", str(self.sequence_dir),
            "--person-checkpoint", str(self.person_checkpoint),
            "--motion-checkpoint", str(self.motion_checkpoint),
            "--product-dir", str(self.product_dir),
            "--output", str(self.output_video),
            "--frame-count", str(self.frame_count),
            "--write-video", str(self.write_video),
        ]
        if self.product_chunk_frames:
            argv += ["--product-chunk-frames", str(self.product_chunk_frames)]
        argv += ["--receiver-floor-from-checkpoint", self.receiver_floor,
                 "--receiver-floor-seed", str(5100 + offset)]
        argv += [
            "--model-label", str(self.model_label),
            "--geometry-translation-y-m", str(self.translation_y),
            "--pose-audit-result", str(self.pose_audit),
            "--motion-detection-result", str(self.motion_detection),
            "--geometry-depth-scale", "1.0",
            "--velocity-frame-offset", "0.0",
            "--velocity-gain", str(self.velocity_gain),
            "--velocity-scale-floor", str(self.velocity_floor),
            "--velocity-scale-mode", str(self.velocity_mode),
            "--velocity-differentiator", str(self.velocity_diff),
            "--static-jitter-sigma-um", "0",
        ]
        if self.scene_visibility:
            argv += ["--scene-visibility", str(self.scene_visibility)]
        argv += ["--person-solid-angle-weight", str(self.person_solid_angle)]
        for r in dataset.radars:
            argv += [r.flag("receiver-phase", "sigma-rad"), str(v(r.name, "receiver_phase")),
                     r.flag("receiver-phase", "drift-rad-per-frame"), str(v(r.name, "phase_drift")),
                     r.flag("receiver-noise", "frame-correlation"), str(v(r.name, "frame_correlation")),
                     r.flag("receiver-chirp-phase", "sigma-rad"), str(v(r.name, "chirp_phase"))]
        argv += ["--receiver-noise-chirp-correlation", "0.3"]
        # The calibration file switches the floor ON in the renderer whatever
        # the scale, so a floor of 0 (a dataset without HuPR's fitted floor)
        # passes no file.
        if self.diffuse_floor > 0:
            if self.diffuse_floor_calibration is None:
                raise ValueError("a diffuse floor needs its calibration file (diffuse_floor_calibration)")
            argv += ["--diffuse-floor-calibration", str(self.diffuse_floor_calibration)]
        if self.antenna_coupling != "off" and coupling_files(coupling_dir, dataset):
            argv += ["--antenna-coupling-calibration", str(coupling_dir)]
        argv += ["--diffuse-floor-scale", str(self.diffuse_floor)]
        for r in dataset.radars:
            argv += [r.flag("frame-receiver-nuisance", "scale"), str(v(r.name, "frame_nuisance")),
                     r.flag("frame-receiver-nuisance", "correlation"), str(v(r.name, "frame_correlation"))]
        argv += [
            "--frame-receiver-nuisance-mode", "common-range-gain",
            "--frame-receiver-nuisance-range-smoothing", "0",
        ]
        for r in dataset.radars:
            argv += [r.flag("local-additive-nuisance", "scale"), str(v(r.name, "local_nuisance")),
                     r.flag("local-additive-nuisance", "correlation"), str(v(r.name, "local_correlation"))]
        argv += ["--local-additive-nuisance-scatterers-per-bin", "48"]
        for r in dataset.radars:
            argv += [r.flag("slow-time-nuisance", "scale"), str(v(r.name, "slow_time"))]
        argv += ["--slow-time-nuisance-mode", "thermal"]
        for r in dataset.radars:
            argv += [r.flag("slow-time-nuisance", "correlation"), str(v(r.name, "slow_time_rho")),
                     r.flag("slow-time-nuisance", "empty-correlation"), str(v(r.name, "empty_correlation")),
                     r.flag("slow-time-nuisance", "fixed-fraction"), str(v(r.name, "fixed_fraction"))]
        argv += [
            "--slow-time-nuisance-scatterers-per-bin", "48",
            "--slow-time-nuisance-angle-source", "calibrated",
            "--slow-time-nuisance-edge-blend-bins", str(self.slow_time_edge_blend),
            "--slow-time-nuisance-seed", str(1100 + offset),
            "--diffuse-floor-seed", str(100 + offset),
            "--local-additive-nuisance-seed", str(4100 + offset),
            "--frame-receiver-nuisance-seed", str(2100 + offset),
            "--nuisance-backend", str(self.nuisance_backend),
            "--nuisance-draws", str(self.nuisance_draws),
        ]
        for r in dataset.radars:
            argv += [r.flag("absolute-gain"), str(v(r.name, "absolute_gain"))]
        if self.channel_calibration:
            argv += ["--virtual-channel-calibration", str(self.channel_calibration)]
        for r in dataset.radars:
            argv += [r.flag("slow-time-nuisance", "scene-fraction"), str(v(r.name, "scene_fraction"))]
        argv += ["--slow-time-nuisance-scene-smoothing-bins", str(self.scene_smoothing_bins)]
        if self.scene_density_normalize:
            argv += ["--slow-time-nuisance-scene-density-normalize"]
        if self.save_noise_diagnostic_products:
            argv += ["--save-raw-rd-products", "--save-first-loop-ra-products"]
            if self.radproposer_products:
                argv += ["--save-radproposer-products"]
        argv += ["--save-pose-motion-dra", "--save-pose-motion-dra-complex",
                 "--device", str(self.device)]
        return argv

    @classmethod
    def from_env(cls) -> "RenderRequest":
        """Build from the variables the shell wrapper still exports.

        Transitional: it exists so callers that have not been ported yet keep
        working, and it goes when the velocity stage constructs this directly.
        """
        e = os.environ.get
        workspace = Path(e("HUPR_WORKSPACE", str(REPO / "outputs/workspace")))
        def _flag(name, default="0"):
            return e(name, default) == "1"
        # HuPR's two boards were exported as *_H / *_V; the override dict takes
        # whichever of them the environment set.
        env_keys = {"absolute_gain": "ABSOLUTE_GAIN", "slow_time": "SLOW_TIME",
                    "slow_time_rho": "SLOW_TIME_RHO", "scene_fraction": "SCENE_FRACTION",
                    "receiver_phase": "RECEIVER_PHASE", "chirp_phase": "CHIRP_PHASE",
                    "frame_nuisance": "FRAME_NUISANCE", "local_nuisance": "LOCAL_NUISANCE"}
        per_radar = {}
        for key, env_name in env_keys.items():
            for radar, suffix in (("horizontal", "H"), ("vertical", "V")):
                if e(f"{env_name}_{suffix}") is not None:
                    per_radar.setdefault(key, {})[radar] = float(e(f"{env_name}_{suffix}"))
        return cls(
            sequence=int(e("SEQUENCE", "1")),
            sequence_dir=Path(e("SEQUENCE_DIR", str(workspace / "person_sequence_1_first_loop"))),
            person_checkpoint=Path(e("PERSON_CHECKPOINT", "")),
            motion_checkpoint=Path(e("MOTION_CHECKPOINT", "")),
            product_dir=Path(e("PRODUCT_DIR", str(workspace / "rendered/adnan_v23_angle_deconvolution_132"))),
            output_video=Path(e("OUTPUT_VIDEO", str(workspace / "deliverables/hupr_adnan_v23_angle_deconvolution_132.mp4"))),
            model_label=e("MODEL_LABEL", "V23 held-out angle-response deconvolution ablation"),
            translation_y=float(e("TRANSLATION_Y", "-0.7282782486")),
            pose_audit=Path(e("POSE_AUDIT", "")),
            motion_detection=Path(e("MOTION_DETECTION", "")),
            workspace=workspace,
            frame_count=int(e("FRAME_COUNT", "132")),
            write_video=e("WRITE_VIDEO", "on"),
            scene_visibility=Path(e("SCENE_VISIBILITY")) if e("SCENE_VISIBILITY") else None,
            person_solid_angle=e("PERSON_SOLID_ANGLE", "off"),
            per_radar=per_radar,
            scene_smoothing_bins=int(e("SCENE_SMOOTHING_BINS", "33")),
            diffuse_floor=float(e("DIFFUSE_FLOOR", "0.0")),
            diffuse_floor_calibration=Path(e("DIFFUSE_FLOOR_CALIBRATION")) if e("DIFFUSE_FLOOR_CALIBRATION") else None,
            velocity_gain=float(e("VELOCITY_GAIN", "1.0")),
            velocity_floor=float(e("VELOCITY_FLOOR", "0.0")),
            velocity_mode=e("VELOCITY_MODE", "checkpoint"),
            velocity_diff=e("VELOCITY_DIFF", "savgol"),
            slow_time_edge_blend=int(e("SLOW_TIME_EDGE_BLEND", "9")),
            nuisance_seed_offset=int(e("NUISANCE_SEED_OFFSET", "0")),
            nuisance_backend=e("NUISANCE_BACKEND", "torch"),
            nuisance_draws=e("NUISANCE_DRAWS", "gpu"),
            antenna_coupling=e("ANTENNA_COUPLING", "on"),
            antenna_coupling_dir=Path(e("ANTENNA_COUPLING_DIR")) if e("ANTENNA_COUPLING_DIR") else None,
            channel_calibration=Path(e("CHANNEL_CAL")) if e("CHANNEL_CAL") else None,
            product_chunk_frames=int(e("PRODUCT_CHUNK_FRAMES")) if e("PRODUCT_CHUNK_FRAMES") else None,
            scene_density_normalize=_flag("SCENE_DENSITY_NORMALIZE"),
            save_noise_diagnostic_products=_flag("SAVE_NOISE_DIAGNOSTIC_PRODUCTS"),
        )

    def run(self, repo: Path | None = None, log: Path | None = None) -> None:
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is unavailable; the renderer requires the GPU")
        argv = self.argv()
        self.product_dir.mkdir(parents=True, exist_ok=True)   # the command, verbatim, for the checkpoint
        (self.product_dir / "render_argv.json").write_text(json.dumps(argv) + "\n")
        run_module("mm4DPS.video", *argv, repo=repo, log=log, workspace=self.workspace)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    workspace = sub.add_parser("setup-workspace")
    workspace.add_argument("run", nargs="?", default="workspace")
    flags = sub.add_parser("trainer-flags")
    flags.add_argument("--replace", default=None)
    flags.add_argument("--append", default=None)
    reference = sub.add_parser("reference-sequence")
    reference.add_argument("run")
    reference.add_argument("sequence", type=int)
    person = sub.add_parser("person-stack")
    person.add_argument("sequence", type=int)
    person.add_argument("--base", required=True)
    person.add_argument("--run", required=True)
    person.add_argument("--person-extra", default="")
    treat = sub.add_parser("treatment")
    treat.add_argument("base")
    treat.add_argument("treat")
    treat.add_argument("sequence", type=int)
    treat.add_argument("--person-source-run", default=None)
    sub.add_parser("render-from-env")
    onward = sub.add_parser("velocity-onward")
    onward.add_argument("run"); onward.add_argument("sequence", type=int)
    chain = sub.add_parser("run")
    chain.add_argument("run"); chain.add_argument("sequence", type=int)
    chain.add_argument("--dataset", default=None, choices=("hupr", "rtpose"),
                       help="which dataset the sequence belongs to "
                            "(default: $MM4DPS_DATASET, else hupr)")
    chain.add_argument("--mask-off", action="store_true",
                       help="the control arm: no per-frame scene visibility")
    chain.add_argument("--intermediates", action="store_true",
                       help="keep .work/ (every working product, the no-noise render, "
                            "videos, per-step history, stage logs)")
    chain.add_argument("--artifacts", type=Path, default=None, metavar="DIR",
                       help="also build the two media folders into DIR")
    again = sub.add_parser("rerender", help="forward render a sequence again from its checkpoint.pt")
    again.add_argument("checkpoint", type=Path)
    again.add_argument("--output-dir", type=Path, required=True)
    again.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.command == "rerender":
        print(f"product: {rerender(args.checkpoint, args.output_dir, device=args.device)}")
        return 0
    if args.command == "velocity-onward":
        velocity_onward(
            args.run, args.sequence,
            vf_param=os.environ.get("VF_PARAM", "temporal-basis"),
            vf_lambda=float(os.environ.get("VF_LAMBDA", "1.0")),
            trainer_flags=(os.environ.get("VF_EXTRA") or "").split() or None,
            render_tag=os.environ.get("RENDER_TAG"),
            scene_visibility=Path(os.environ["SCENE_VISIBILITY"])
            if os.environ.get("SCENE_VISIBILITY") else None,
            noise_free_render=os.environ.get("VF_NOISE_FREE_RENDER", "on") == "on",
            parallel_renders=os.environ.get("VF_PARALLEL_RENDERS", "on") == "on",
            diagnostic_video=os.environ.get("VF_DIAG_VIDEO", "on") == "on",
            keep_eval_products=os.environ.get("KEEP_EVAL_PRODUCTS", "1") == "1")
        return 0
    if args.command == "run":
        seqdir = run_chain(
            args.run, args.sequence,
            mask=not args.mask_off, intermediates=args.intermediates,
            artifacts=args.artifacts,
            trainer_replace=os.environ.get("VF_EXTRA") or None,
            trainer_append=os.environ.get("VF_EXTRA_APPEND") or None,
            vf_param=os.environ.get("VF_PARAM", "temporal-basis"),
            dataset=args.dataset)
        print(f"outputs: {seqdir}")
        return 0

    if args.command == "render-from-env":
        RenderRequest.from_env().run(repo=REPO)
        return 0

    if args.command == "person-stack":
        person_stack(args.sequence, args.base, args.run,
                     person_extra=args.person_extra)
        return 0
    if args.command == "treatment":
        treatment(args.base, args.treat, args.sequence,
                  person_source_run=args.person_source_run)
        return 0

    if args.command == "reference-sequence":
        reference_sequence(args.run, args.sequence)
        return 0

    if args.command == "setup-workspace":
        created = setup_workspace(args.run)
        print(f"workspace ready: {created}")
        from mm4DPS import describe
        for name, status in describe().items():
            print(f"  {name:18s} {status}")
        return 0
    composed = TrainerFlags(replace=args.replace, append=args.append)
    text = composed.as_string()
    for note in composed.notes:
        print(note, file=sys.stderr)
    print(text)
    return 0




# ----------------------------------------------------------------------
# velocity_onward: scripts/run_velocity_field_onward.sh
# ----------------------------------------------------------------------
# The clean variant zeroes every nuisance. In the old serial shell loop this
# relied on `export` ordering inside a shared subshell; as fields it cannot leak
# between variants.
NOISE_FREE_FIELDS = ("slow_time", "receiver_phase", "chirp_phase", "frame_nuisance", "local_nuisance")


def noise_free_overrides(dataset: Dataset, absolute_gain: dict[str, float]) -> dict:
    """The clean render: no trained floor drawn (every fitted nuisance is already zero)."""
    return dict(
        per_radar={"absolute_gain": dict(absolute_gain),
                   **{key: {name: 0.0 for name in dataset.names} for key in NOISE_FREE_FIELDS}},
        diffuse_floor=0.0, receiver_floor="off")


def velocity_onward(run: str, sequence: int, repo: Path | None = None,
                    vf_param: str = "temporal-basis", vf_lambda: float = 1.0,
                    trainer_flags: list[str] | None = None,
                    render_tag: str | None = None,
                    scene_visibility: Path | None = None,
                    noise_free_render: bool = True,
                    parallel_renders: bool = True,
                    diagnostic_video: bool = True,
                    keep_eval_products: bool = True) -> None:
    """Train the velocity field, render, score, clean up."""
    root = repo or REPO
    workspace = root / "outputs" / run
    person = workspace / f"person_sequence_{sequence}_v2clean"
    log = workspace / "deliverables" / f"vf_onward_seq{sequence}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    for needed in (person / "metadata.json",
                   workspace / f"training/seq{sequence}_person_v2/best_shared_human_material.pt",
                   workspace / f"calibration/seq{sequence}_person_range_v2/result.json"):
        if not needed.exists():
            raise FileNotFoundError(f"missing prerequisite {needed}")

    translation = json.loads(
        (workspace / f"calibration/seq{sequence}_person_range_v2/result.json").read_text()
    )["selected_geometry_translation_y_m"]
    dataset = current_dataset()
    frame_count = int(json.loads((person / "metadata.json").read_text())["frames"])
    note(f"vf onward: run={run} seq={sequence} param={vf_param} "
         f"lambda={vf_lambda} TY={translation} gain=learned", log)

    vf_dir = workspace / f"training/seq{sequence}_vf"
    if (vf_dir / "result.json").is_file():
        note("velocity-field training already present, skipping", log)
    else:
        # THE NOISE MODEL IS LEARNED. The trainer's floor starts from this
        # sequence's own off-person windowed |RD|^2 and learns its scale and
        # range shape; the trained floor rides in the checkpoint and the
        # renderer draws its receiver floor from it. No calibration file, no
        # fitted receiver-model constants, no diffuse-floor seed: the
        # calibrations under seeds/calibration/ are not read by this stage
        # (docs/SEEDS.md lists what still is, elsewhere).
        noise_model = os.environ.get("NOISE_MODEL", "on")
        flags = ["--noise-model", noise_model,
                 "--person-solid-angle-weight", os.environ.get("PERSON_SOLID_ANGLE", "off"),
                 "--scene-solid-angle", os.environ.get("SCENE_SOLID_ANGLE", "off")]
        # The objective. Since the 2026-09-22 consolidation sweep the default is FOUR terms --
        # loop-0 |RAE|, the plane |RAE| over every rendered plane, micro-Doppler and the person
        # window -- because each RA term is its RAE term computed on a subset of the array's
        # channels, and dropping the two costs nothing downstream on either dataset while the
        # plane consolidation improves micro-Doppler (docs/OBJECTIVE_ABLATION.md). The plane
        # consolidation needs both of its companions: --elevation-planes all, or it is a Doppler
        # band restriction rather than a consolidation, and --rae-floor-gradient on, or the
        # learned noise floor loses its only substantial gradient path.
        # OBJECTIVE=legacy6 restores the six-term objective the September results were made under.
        objective = os.environ.get("OBJECTIVE", "2x2")
        if objective == "2x2":
            # The default since 2026-09-23: ONE functional over {global, windowed} x {loop-0, per
            # plane}. The windowed pair replaces the person window (which the windowed loop-0 term
            # generalises, in the trainer's own view rather than the consumer's) and micro-Doppler
            # (which the windowed plane term replaces at 0.80x / 1.00x of its velocity gradient).
            # Reconstruction improves markedly -- cascade clutter-removed RA 0.470 -> 0.525, above
            # even the six-term objective's 0.495, range-Doppler up on every array, HuPR
            # micro-Doppler unchanged -- at a cost of 0.010 cascade micro-Doppler and a downstream
            # gap of +0.005 AP on HuPR (10 sequences worse, 6 better). The regression is real and
            # is adopted deliberately (docs/OBJECTIVE_ABLATION.md).
            flags += ["--ablate-loss-term", "loop0-ra,plane-ra,md",
                      "--person-window-loss", "0", "--window-rae-terms", "on",
                      "--elevation-planes", "all", "--rae-floor-gradient", "on"]
        elif objective == "4term":
            flags += ["--ablate-loss-term", "loop0-ra,plane-ra",
                      "--elevation-planes", "all", "--rae-floor-gradient", "on"]
        elif objective != "legacy6":
            raise SystemExit(f"OBJECTIVE={objective!r}: expected '2x2', '4term' or 'legacy6'")
        # The temporal basis keeps 420 of 600 DCT bands (a 3.5 Hz cap at
        # 10 Hz). A sequence of another length keeps the same cap: the same
        # fraction of its own bands.
        if frame_count != 600:
            flags += ["--temporal-bands", str(int(round(420 / 600 * frame_count)))]
        # The step's memory scales with the channels it renders per batch.
        # HuPR keeps its measured sweet spot (24 frames a step); another
        # dataset gets the frame count that keeps the step at the 576
        # channel-frames a 24 GB card holds (measured: the cascade at 3 x 192),
        # counted on the TRAINING sub-array the trainer renders (the cascade:
        # 9 x 16 = 144 channels, 4 frames a step and at eval -- the eval has
        # no backward, and its result is the same to 2e-9 at 2 or 4).
        channels = sum(r.n_tx * r.n_rx for r in dataset.training_subset().radars)
        if dataset.name != "hupr":
            flags += ["--frame-batch", str(max(1, int(576 / channels))),
                      "--eval-frame-batch", str(max(1, int(576 / channels)))]
            # The Doppler-plane support stays HuPR's: the median-over-loops
            # floor and the band in every view (62 of 63 planes on the
            # cascade). Rendering only the bins that carry the person's
            # Doppler energy (`--doppler-support-floor energy`, band off)
            # was measured on seq 183 at 400 epochs: the bins with no
            # measured energy are then unsupervised, nothing penalises
            # energy rendered into them, and the person smears over Doppler
            # (rendered spread 13.3 bins against 4.5, centroid r +0.62
            # against +0.98). The zero targets are what keep it compact.
        note("velocity-field training", log)
        trainer_argv = [
            "--sequence", sequence,
            "--sequence-dir", person,
            "--person-checkpoint", workspace / f"training/seq{sequence}_person_v2/best_shared_human_material.pt",
            "--output-dir", vf_dir, "--geometry-translation-y-m", translation,
            "--velocity-parameterization", vf_param,
            "--microdoppler-weight", vf_lambda,
            *(["--antenna-coupling-calibration", workspace / "calibration" / "antenna_coupling"]
              if coupling_files(workspace / "calibration" / "antenna_coupling", dataset) else []),
            *flags, *(trainer_flags or [])]
        vf_dir.mkdir(parents=True, exist_ok=True)      # the command, verbatim, for the checkpoint
        (vf_dir / "trainer_argv.json").write_text(json.dumps([str(a) for a in trainer_argv]) + "\n")
        run_module("mm4DPS.velocity_field", *trainer_argv, repo=root, log=log, workspace=workspace)

    # Install at the paths the standard drivers expect: the renderer dispatches
    # on motion_representation, so the untouched render path drives it.
    motion_dir = workspace / f"training/seq{sequence}_motion_v2"
    motion_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(vf_dir / "velocity_field.pt", motion_dir / "surface_chirp_motion.pt")
    shutil.copy(vf_dir / "result.json", motion_dir / "result.json")
    # The receiver gain the trainer learned is the gain the renders apply:
    # the estimate is the model's own, made after training, not a
    # calibration before it.
    gains = {name: float(value) for name, value in
             json.loads((vf_dir / "result.json").read_text())["absolute_receiver_gain"].items()}
    note(f"absolute receiver gain (learned): {gains}", log)

    # DESTRUCTIVE IN-PLACE WRITE, guarded (R088). The line below copies this
    # stage's OUTPUT over its own INPUT, so the workspace stops being rerunnable
    # after one pass -- a second run starts from a checkpoint carrying per-point
    # person materials where the trainer expects the shared six-vector, and fails
    # with "expected six raw material parameters". The pristine copy is kept on
    # the first pass only.
    person_input = workspace / f"training/seq{sequence}_person_v2/best_shared_human_material.pt"
    pristine = person_input.with_suffix(person_input.suffix + ".pristine")
    if os.environ.get("VF_KEEP_INPUT", "1") == "1" and person_input.is_file() \
            and not pristine.is_file():
        shutil.copy(person_input, pristine)
        note(f"preserved the pre-training person checkpoint at {pristine}", log)
    shutil.copy(vf_dir / "best_shared_human_material.pt", person_input)

    run_step("mm4DPS.motion", "pose-motion-frames",
             "--sequence", sequence, "--sequence-dir", person,
             "--motion-training-result", motion_dir / "result.json",
             "--output-dir", workspace / f"calibration/seq{sequence}_pose_v2",
             repo=root, log=log, workspace=workspace)
    detect_dir = workspace / f"calibration/seq{sequence}_detect_v2"
    if not (detect_dir / "result.json").is_file():
        run_module("mm4DPS.motion", "detect-person-motion",
                   "--sequence", sequence, "--sequence-dir", person,
                   "--translation-y-m", translation, "--output-dir", detect_dir,
                   repo=root, log=log, workspace=workspace)

    tag = render_tag or os.environ.get("RENDER_TAG", "v2")

    def render_variant(variant: str, variant_log: Path) -> None:
        noisy = variant == "noisy"
        suffix = f"seq{sequence}_{tag}_600" if noisy else f"seq{sequence}_{tag}_no_noise_600"
        media = f"hw10_{tag}" if noisy else f"hw10_{tag}_no_noise"
        product = workspace / "rendered" / suffix
        request = RenderRequest(
            sequence=sequence, sequence_dir=person,
            person_checkpoint=person_input,
            motion_checkpoint=motion_dir / "surface_chirp_motion.pt",
            product_dir=product,
            output_video=workspace / "deliverables" / f"hupr_seq{sequence}_{media}_600.mp4",
            model_label=f"seq {sequence}: velocity field {vf_param} lambda {vf_lambda}",
            translation_y=translation,
            pose_audit=workspace / f"calibration/seq{sequence}_pose_v2/pose_audit.json",
            motion_detection=detect_dir / "result.json",
            workspace=workspace, frame_count=frame_count,
            write_video="on" if diagnostic_video and dataset.name == "hupr" else "off",
            scene_visibility=scene_visibility,
            person_solid_angle=os.environ.get("PERSON_SOLID_ANGLE", "off"),
            dataset=dataset.name,
            save_noise_diagnostic_products=True, velocity_diff="savgol",
            radproposer_products=dataset.name == "hupr",
            antenna_coupling="on",
            **({"per_radar": {"absolute_gain": gains,
                              "scene_fraction": {n: 0.0 for n in dataset.names}},
                "diffuse_floor": 0.0, "receiver_floor": os.environ.get("MM4DPS_RECEIVER_FLOOR", "on")}   # off: the noisy variant without the drawn floor (diagnostic, static_viz)
               if noisy else noise_free_overrides(dataset, gains)),
        )
        note(f"render v2 {variant}", variant_log)
        request.run(repo=root, log=variant_log)
        note(f"spectrograms v2 {variant}", variant_log)
        run_module("mm4DPS.media", "microdoppler-spectrograms",
                   "--product-dir", product, "--sequence-dir", person,
                   "--translation-y-m", translation, "--half-width", 10,
                   "--label", "v2", "--motion-detection", detect_dir / "result.json",
                   "--pose-audit", workspace / f"calibration/seq{sequence}_pose_v2/pose_audit.json",
                   "--output-dir", workspace / f"deliverables/microdoppler_seq{sequence}/{media}",
                   repo=root, log=variant_log, workspace=workspace)

    noisy_log = workspace / "deliverables" / f"render_noisy_seq{sequence}.log"
    clean_log = workspace / "deliverables" / f"render_clean_seq{sequence}.log"
    for handle in (noisy_log, clean_log):
        handle.parent.mkdir(parents=True, exist_ok=True)
        handle.write_text("")
    variants_note = "+clean" if noise_free_render else " only (noise_free_render=False)"
    if parallel_renders and noise_free_render:
        # Each render holds ~244 MiB and ~11% of the card, so serialising them
        # left it idle: 2 x 63.4 s serial against 71.2 s for the concurrent pair.
        # There is no measured cache any more (it saved 4 s and cost 900 MB), so
        # the manifest race that used to force a cold pair serial is gone too.
        note(f"renders v2 noisy{variants_note}, concurrent on one GPU", log)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(render_variant, "noisy", noisy_log),
                       pool.submit(render_variant, "clean", clean_log)]
            for future in futures:
                future.result()
    else:
        note(f"renders v2 noisy{variants_note}, serial", log)
        render_variant("noisy", noisy_log)
        if noise_free_render:
            render_variant("clean", clean_log)
    with log.open("a") as handle:
        for source in (noisy_log, clean_log):
            handle.write(source.read_text())

    (workspace / "deliverables" / "final_render_variant.txt").write_text("v2\n")

    note("scoring", log)
    for media in ("hw10_v2", "hw10_v2_no_noise", "hw10_v3", "hw10_v3_no_noise",
                  f"hw10_{tag}", f"hw10_{tag}_no_noise"):
        directory = workspace / f"deliverables/microdoppler_seq{sequence}/{media}"
        if not directory.is_dir():
            continue
        # --sequence-dir and --frame-selection are BOTH explicit. The default
        # selection needs the person cloud, and passing only --pose-audit made
        # this abort while `| tee` swallowed the exit status, so the stage
        # reported success and wrote no score.json.
        stdout = run_module(
            "mm4DPS.evaluation", "score-microdoppler",
            "--spectrogram-dir", directory, "--sequence-dir", person,
            "--frame-selection", "pose-speed",
            "--pose-audit", workspace / f"calibration/seq{sequence}_pose_v2/pose_audit.json",
            "--output", directory / "score.json",
            repo=root, capture=True, workspace=workspace)
        (directory / "score.stdout").write_text(stdout)
        print(stdout, end="")
        with log.open("a") as handle:
            handle.write(stdout)

    # KEEP_EVAL_PRODUCTS preserves what `mm4DPS.evaluate` reads:
    # first_loop_ra_complex, which cleanup never touched, and motion_dra, which
    # it did. The suite deliberately does NOT re-render to recover these.
    patterns = ["*_radproposer_complex.npy", "*_motion_dra_complex.npy"]
    if keep_eval_products:
        note("cleanup of large diagnostic products (keeping motion_dra for the "
             "eval suite)", log)
    else:
        note("cleanup of large diagnostic products (keep_eval_products=False: "
             "removing motion_dra too)", log)
        patterns.append("*_motion_dra.npy")
    for pattern in patterns:
        for stale in (workspace / "rendered").glob(f"*/{pattern}"):
            stale.unlink()
    note(f"vf onward DONE: run={run} seq={sequence}", log)


# ----------------------------------------------------------------------
# run_chain: scripts/run_velocity_chain.sh
# ----------------------------------------------------------------------
def _stage(name: str, function, seconds: Path, log: Path):
    """Run one stage, time it, and record the seconds where the total is."""
    note(f"STAGE {name}: start", log)
    started = time.time()
    try:
        result = function()
    except Exception:
        note(f"STAGE {name}: FAILED", log)
        raise
    elapsed = int(time.time() - started)
    note(f"STAGE {name}: done in {elapsed}s", log)
    with seconds.open("a") as handle:
        handle.write(f"{name} {elapsed}\n")
    return result


def run_chain(run: str, sequence: int, repo: Path | None = None,
              mask: bool = True, intermediates: bool = False,
              artifacts: Path | None = None,
              trainer_replace: str | None = None,
              trainer_append: str | None = None,
              vf_param: str = "temporal-basis",
              dataset: str | None = None) -> Path:
    """One sequence end to end: raw ADC + the provided GVHMR geometry.

    ONE PROFILE. `CHAIN_PROFILE=full` was removed in the deployment strip, and
    chain_profile.txt is still written with the constant "velocity" because
    mm4DPS/qa.py refuses to score an artifact across two profiles
    or an UNRECORDED one -- "unknown" and "the same" are different claims (R097).

    PERSON_RUN is shared between the arms of an A/B on purpose. The person fit is
    stochastic training: a chain run twice fits two different
    best_shared_human_material.pt files (measured, seq38 fd5cebc419f4577d against
    3df42cef1542958f), so an arm comparison that lets that vary is comparing two
    things at once.
    """
    root = repo or REPO
    # The dataset is selected ONCE here and exported ($MM4DPS_DATASET) so every
    # stage subprocess reads the same radars, shapes and config names.
    dataset = Dataset.select(dataset or os.environ.get("MM4DPS_DATASET", "hupr"))
    # THE LAYOUT. One directory per sequence holds the seven files a run is
    # for; the chain's three working workspaces (reference, person, treatment)
    # and every intermediate live under .work/ inside it, and .work/ is removed
    # at the end unless --intermediates. The internal names the stages use
    # (vf_, _v2, hw10_v2vis, ...) therefore never appear in an output.
    seqdir = root / "outputs" / run / f"seq{sequence}"
    work = f"{run}/seq{sequence}/.work"
    # treat<S>: the media resolves a workspace as <prefix><sequence>.
    ref, person, treat = f"{work}/ref", f"{work}/person", f"{work}/treat{sequence}"
    setup_workspace(ref, repo=root)
    setup_workspace(person, repo=root)
    workspace = root / "outputs" / ref
    # The run's own log and stage timings are deliverables, so they live in
    # seqdir from the start; anything under .work/ is gone at the end.
    seqdir.mkdir(parents=True, exist_ok=True)
    log = seqdir / "run.log"
    log.write_text("")

    seconds = seqdir / ".stage_seconds"
    seconds.write_text("")
    note(f"chain start: run={run} seq={sequence} dataset={dataset.name} "
         f"mask={'on' if mask else 'off'} intermediates={intermediates}", log)
    # The mask is the default. Off is the control arm of an A/B, and the two
    # arms differ in nothing else: same seeds, same person fit, same flags.
    render_tag = "v2vis" if mask else "v2"
    scene_visibility = (root / "outputs" / treat / "visibility"
                        / f"seq{sequence}_s2_shadow.npy") if mask else None
    if mask:
        trainer_append = " ".join(filter(None, [
            trainer_append, f"--scene-visibility {scene_visibility}"]))

    # GVHMR runs on this GPU from the first second, beside the reference
    # stage; the person stage waits for whatever is left of it.
    note("GVHMR started (the person's mesh from the RGB frames)", log)
    start_gvhmr(sequence, root / "outputs" / person)
    _stage("preprocessing", lambda: reference_sequence(ref, sequence, repo=root),
           seconds, log)

    fitted = (root / "outputs" / person
              / f"training/seq{sequence}_person_v2/best_shared_human_material.pt")
    if fitted.is_file():
        note(f"person_stack: reusing {person} (shared across arms)", log)
        with seconds.open("a") as handle:
            handle.write("person_stack 0\n")
    else:
        _stage("person_stack",
               lambda: person_stack(sequence, ref, person, repo=root),
               seconds, log)

    _stage("treatment",
           lambda: treatment(ref, treat, sequence, repo=root,
                             person_source_run=person),
           seconds, log)

    _stage("scene_visibility",
           lambda: scene_shadow_mask(treat, sequence, repo=root),
           seconds, log)

    flags = TrainerFlags(replace=trainer_replace, append=trainer_append)
    composed = flags.compose()
    for line in flags.notes:
        note(line, log)
    note(f"trainer flags: {' '.join(composed)}", log)

    _stage("velocity_onward",
           lambda: velocity_onward(
               treat, sequence, repo=root, vf_param=vf_param,
               trainer_flags=composed, render_tag=render_tag,
               scene_visibility=scene_visibility,
               noise_free_render=intermediates, diagnostic_video=intermediates,
               keep_eval_products=intermediates),
           seconds, log)

    # THE METRICS TABLE, in the chain rather than transcribed by hand. This is
    # the only stage whose output goes into the paper directly, and until now
    # nothing rendered it: the suite wrote JSON and a human copied numbers.
    _stage("evaluate",
           lambda: evaluate_products(treat, sequence, repo=root, render_tag=render_tag),
           seconds, log)

    for target in (root / "outputs" / treat, workspace):
        (target / "deliverables").mkdir(parents=True, exist_ok=True)
        (target / "deliverables" / "chain_profile.txt").write_text("velocity\n")

    note(f"chain DONE: run={run} seq={sequence}", log)
    note(f"per-stage seconds: {seconds.read_text().replace(chr(10), ' ').strip()}", log)
    collect(seqdir, root / "outputs" / work, sequence, render_tag,
            mask=mask, intermediates=intermediates, artifacts=artifacts,
            repo=root, log=log, seconds=seconds)
    return seqdir


def coupling_files(directory: Path, dataset) -> bool:
    """Whether `directory` holds an antenna-coupling file for every radar of `dataset`."""
    return all((Path(directory) / f"{radar.name}_coupling.npy").is_file() for radar in dataset.radars)


def collect(seqdir: Path, work: Path, sequence: int, render_tag: str, *,
            mask: bool, intermediates: bool, artifacts: Path | None,
            repo: Path, log: Path, seconds: Path) -> None:
    """The seven files, from the working workspaces; then the workspaces go.

    checkpoint.pt   the velocity field, the person material, and `render_inputs`: the scene and
                    person geometry exactly as the renderer receives it (positions, normals,
                    materials, visibility, calibrations) -- see `render_inputs`
    history.json    per-epoch means of the trainer's per-step log
    metrics.json    the evaluation suite's JSON + the score + render_config
    run.log         every stage's log, concatenated
    <radar>_rendered.npy   the rendered slow-time cube (the measured one is not stored: media.MeasuredCube decodes it)
    gvhmr_vertices.npz     the person's SMPL vertices per camera frame, from GVHMR on this run

    --artifacts builds the two media folders here, before .work/ is removed,
    because the media reads the working products.
    """
    import shutil
    import torch
    from mm4DPS.qa import fingerprint

    dataset = current_dataset()
    treat = work / f"treat{sequence}"
    train = treat / f"training/seq{sequence}_vf"
    product = treat / f"rendered/seq{sequence}_{render_tag}_600"
    seqdir.mkdir(parents=True, exist_ok=True)
    from mm4DPS.data import ProductCube
    cube = ProductCube(product / f"{dataset.names[0]}_rendered.npy")
    frames = int(cube.shape[0])
    render_result = json.loads((product / "render_result.json").read_text())
    measured_stored = render_result.get("measured_cube") == "on"

    velocity_field = torch.load(train / "velocity_field.pt", map_location="cpu", weights_only=False)
    person_material = torch.load(train / "best_shared_human_material.pt", map_location="cpu", weights_only=False)
    torch.save({
        "velocity_field": velocity_field,
        "person_material": person_material,
        "render_inputs": render_inputs(work, treat, sequence, frames, render_tag, velocity_field, person_material, repo),
    }, seqdir / "checkpoint.pt")

    steps = json.loads((train / "history.json").read_text())
    by_epoch: dict[int, list] = {}
    for entry in steps:
        by_epoch.setdefault(int(entry["epoch"]), []).append(entry)
    epochs = []
    for epoch in sorted(by_epoch):
        rows = by_epoch[epoch]
        mean = lambda radar, key: sum(r[radar][key] for r in rows) / len(rows)
        epochs.append({"epoch": epoch, "steps": len(rows),
                       "mean_total": sum(r["mean_total"] for r in rows) / len(rows),
                       **{radar: {k: mean(radar, k) for k in ("total", "ra_l2", "md_l2")}
                          for radar in dataset.names}})
    result = json.loads((train / "result.json").read_text())
    (seqdir / "history.json").write_text(json.dumps({
        "epochs": epochs,
        "final": result.get("final_validation"),
        "best_validation_total": result.get("best_held_out_total"),
        "validation_frames_held_out": result.get("validation_frames_held_out"),
        "loss_protocol": result.get("loss_protocol"),
    }, indent=1))

    metrics = json.loads((treat / f"deliverables/evaluation/{dataset.name}_seq{sequence}.json").read_text())
    score = treat / f"deliverables/microdoppler_seq{sequence}/hw10_{render_tag}/score.json"
    if score.is_file():
        metrics["microdoppler_score"] = json.loads(score.read_text())
    metrics["render_config"] = {
        **fingerprint(product),          # the R099 key set, exactly
        "mask": "on" if mask else "off",
        "chain_profile": "velocity",
        "render_result_keys": sorted(render_result),
        "stage_seconds": dict(zip(*[iter(seconds.read_text().split())] * 2)),
    }
    seconds.unlink()
    radar0 = dataset.radars[0]
    metrics["render_config"]["product_layout"] = {
        "files": [f"{r}_{k}.npy" for r in dataset.names
                  for k in (("rendered", "measured") if measured_stored else ("rendered",))],
        "measured": ("stored beside the rendered cube" if measured_stored else
                     "not stored: the recording alone, decoded from the archive on demand "
                     "(media.MeasuredCube; --save-measured-cube on to write it)"),
        "axes": ["frame", "loop", "tx", "rx", "range"],
        "shape": [frames, *radar0.product_shape],
        "dtype": ("float16 (real, imag) pairs, a trailing axis of 2, each frame divided by its own "
                  "float32 scale in <radar>_<kind>_scale.npy (data.ProductCube reads it back as complex64)"),
        "loop": f"the {radar0.loops} TDM loops in slow time; no Doppler transform, no mean removed",
        "range": f"{fft_window_kind()}-windowed FFT of the {radar0.samples} ADC samples; bin 0 is DC "
                 f"(the rendered side's range PSF is the same window's response)",
        "doppler": "taken by the consumer: mean removal, the Doppler window, FFT, fftshift "
                   "(data.product_to_doppler / forward_render.slow_time_to_doppler)",
        "fft_window": fft_window_kind(),
        "windows": "range: periodic Hann; azimuth: symmetric Hann without end points; "
                   "doppler: periodic Hann after mean removal (signal_model.fft_window)",
    }
    (seqdir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    # the person's GVHMR mesh, as computed for this run (22-45 MB)
    vertices = work / "person" / "gvhmr" / f"seq{sequence}.npz"
    if vertices.is_file():
        vertices.rename(seqdir / "gvhmr_vertices.npz")
    for radar in dataset.names:
        for kind in (("rendered", "measured") if measured_stored else ("rendered",)):
            (product / f"{radar}_{kind}.npy").rename(seqdir / f"{radar}_{kind}.npy")
            scale = product / f"{radar}_{kind}_scale.npy"
            if scale.is_file():
                scale.rename(seqdir / scale.name)

    with (seqdir / "run.log").open("a") as out:
        for path in sorted(work.rglob("deliverables/*.log")):
            out.write(f"\n===== {path.relative_to(work)} =====\n{path.read_text()}")

    if artifacts is not None:
        _artifacts(artifacts, work, sequence, render_tag, mask, repo, log)

    if intermediates:
        note(f"collect: kept {work}", log)
    else:
        shutil.rmtree(work)
        note("collect: removed .work/", log)


MATERIAL_COLUMNS = ["eps_real", "eps_imag", "sigma_h_m", "correlation_length_m", "coherent_fraction", "thickness_m"]


def render_inputs(work: Path, treat: Path, sequence: int, frames: int, render_tag: str,
                  velocity_field: dict, person_material: dict, root: Path) -> dict:
    """The geometry exactly as the renderer receives it, for the checkpoint.

    Everything after preprocessing, registration, alignment and training:
    the scene's points in the radar frame with the normals the renderer
    derives from the trained quaternions and the physical materials, the
    person's points per frame as registered to the radar, its learned
    normals and materials, both visibility masks, and the calibrations the
    render applies (the antenna coupling, the gains, the floor). Units are
    metres in the radar world: +X right, +Y boresight (range), +Z up; frame
    f is radar frame f.
    """
    import numpy as np
    import torch
    from mm4DPS.rasterizer import PointPrimitives, reparameterize_torch
    from mm4DPS.data import range_bin_spacing_m
    dataset = current_dataset()
    person_dir = treat / f"person_sequence_{sequence}_v2clean"
    metadata = json.loads((person_dir / "metadata.json").read_text())

    state = person_material["shared_background_state"]
    primitives = PointPrimitives(len(state["positions"]), device="cpu")
    primitives.load_state_dict(state)
    with torch.no_grad():
        scene_normals = primitives.get_normals().float()
        scene_physical = reparameterize_torch(state["raw_materials"]).float()
        person_raw = person_material["shared_human_raw_material"]
        person_physical = reparameterize_torch(person_raw).float()
    shadow = treat / "visibility" / f"seq{sequence}_s2_shadow.npy"
    scene_visible = torch.from_numpy(np.load(shadow)) if shadow.is_file() else None

    positions, mesh_normals, visible, joints = [], [], [], []
    for frame in range(frames):
        loaded = np.load(person_dir / "person_frames" / f"frame_{frame:03d}.npz")
        positions.append(loaded["pcl"][:, :3])
        mesh_normals.append(loaded["pcl"][:, 3:6])
        visible.append(loaded["visible"])
        joints.append(loaded["joints_xyz"])
    # The trainer keeps a normal per VISIBLE vertex per frame, in vertex
    # order; the render substitutes them for the mesh normals of exactly
    # those vertices (video.py). Scatter them the same way.
    person_normals = torch.from_numpy(np.stack(mesh_normals)).float()
    for frame, key in ((f, str(f)) for f in range(frames)):
        rows = torch.from_numpy(np.nonzero(visible[frame])[0])
        learned = velocity_field.get("learned_normals", {}).get(key)
        if learned is not None:
            if len(learned) != len(rows):
                raise ValueError(f"frame {frame}: {len(learned)} learned normals, {len(rows)} visible vertices")
            person_normals[frame, rows] = learned.float()

    scaffold_meta = work / "ref" / "scaffolds" / f"seq{sequence}" / "scaffold_metadata.json"
    coupling_dir = treat / "calibration" / "antenna_coupling"
    coupling = {radar.name: torch.from_numpy(np.load(coupling_dir / f"{radar.name}_coupling.npy"))
                for radar in dataset.radars if (coupling_dir / f"{radar.name}_coupling.npy").is_file()}
    def read_json(path: Path):
        return json.loads(path.read_text()) if path.is_file() else None

    train = treat / f"training/seq{sequence}_vf"
    product = treat / f"rendered/seq{sequence}_{render_tag}_600"
    commands = {
        # The exact invocations, paths as they were in the run's workspace:
        # a re-render maps them onto this dict (person_frames <- person,
        # the two checkpoints <- velocity_field / person_material, the shadow
        # mask <- scene.visible, the coupling <- radar.antenna_coupling, the
        # two frame-selection results <- motion_frames) and the tracked
        # configs at `git_sha`. The raw recording is the dataset's.
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                  text=True).stdout.strip(),
        "trainer_argv": read_json(train / "trainer_argv.json"),
        "trainer_result": read_json(train / "result.json"),
        "render_argv": read_json(product / "render_argv.json"),
        "render_result": read_json(product / "render_result.json"),
        "coupling_result": read_json(coupling_dir / "result.json"),
    }
    return {
        "frame": "radar world, metres: +X right, +Y boresight (range), +Z up; index f = radar frame f",
        "range_bin_m": float(range_bin_spacing_m()),
        "person_sequence_metadata": metadata,                          # reconstruction person-sequence's record
        "motion_frames": {                                             # the render's two required frame selections
            "detection": read_json(treat / f"calibration/seq{sequence}_detect_v2/result.json"),
            "pose_audit": read_json(treat / f"calibration/seq{sequence}_pose_v2/pose_audit.json"),
        },
        "commands": commands,
        "scene": {
            "positions_m": state["positions"].float(),                    # (N, 3), aligned to the radar
            "normals": scene_normals,                                     # (N, 3) unit, from the trained quaternions
            "rotations": state["rotations"].float(),                      # (N, 4) the quaternions themselves [w, x, y, z]
            "raw_materials": state["raw_materials"].float(),              # (N, 6) the trained parameters
            "physical_materials": scene_physical,                         # (N, 6) MATERIAL_COLUMNS
            "material_columns": MATERIAL_COLUMNS,
            "point_gains": velocity_field.get("background_point_gains"),
            "visible": scene_visible,                                     # (frames, N) bool: not shadowed by the person
            "alignment": (json.loads(scaffold_meta.read_text()).get("scene_alignment")
                          if scaffold_meta.is_file() else None),
        },
        "person": {
            "positions_m": torch.from_numpy(np.stack(positions)).float(), # (frames, V, 3) registered to the radar
            "visible": torch.from_numpy(np.stack(visible)),               # (frames, V) bool: in the field of view, not self-occluded
            "joints_m": torch.from_numpy(np.stack(joints)).float(),       # (frames, 14, 3) HuPR joint order
            "normals": person_normals,                                    # (frames, V, 3) what the render uses: learned where visible, else the mesh's
            "mesh_normals": torch.from_numpy(np.stack(mesh_normals)).float(),  # (frames, V, 3) the SMPL mesh normals (the initialisation)
            "raw_materials": person_raw.float(),                          # (V, 6)
            "physical_materials": person_physical,                        # (V, 6) MATERIAL_COLUMNS
            "material_columns": MATERIAL_COLUMNS,
            "area_scale": float(person_material["shared_human_area_scale"]),
            "joint_velocity_field_mps": velocity_field.get("joint_velocity_field_mps"),   # (frames, 14, 3)
            "joint_scattering_gains": velocity_field.get("inherited_joint_scattering_gains"),
            "skinning": velocity_field.get("skinning"),
            "registration": metadata.get("gvhmr_conversion", {}).get("depth_scale"),
            "placement": {k: metadata.get(k) for k in ("geometry_method", "camera", "extrinsic_seed", "camera_pairing")},
        },
        "radar": {
            "absolute_receiver_gain": velocity_field.get("absolute_gains"),
            "antenna_coupling": coupling,                                 # {radar: (azimuth TX x RX, near bins) complex64}
            "noise_floor_power": velocity_field.get("noise_floor_power"),
            "noise_floor_radars": velocity_field.get("noise_floor_radars"),
        },
    }


def rerender(checkpoint: Path, output_dir: Path, device: str = "cuda:0") -> Path:
    """Forward render a sequence again from its checkpoint.pt alone.

    Rebuilds the render's inputs from `render_inputs` -- the person frames,
    the two checkpoints, the shadow mask, the antenna coupling, the two
    frame-selection results -- and runs the recorded render command against
    them, with only its paths rewritten. Needs the dataset's recording (the
    measured cube the render compares against) and the tracked configs at
    `commands.git_sha`; nothing else. Returns the product directory.
    """
    import numpy as np
    import torch
    checkpoint = Path(checkpoint)
    stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    inputs = stored["render_inputs"]
    scene, person, radar = inputs["scene"], inputs["person"], inputs["radar"]
    argv = list(inputs["commands"]["render_argv"])
    dataset = Dataset.select(argv[argv.index("--dataset") + 1])
    root = REPO
    ws = Path(output_dir) / "workspace"
    sequence_dir = ws / "person_sequence"
    (sequence_dir / "person_frames").mkdir(parents=True, exist_ok=True)
    frames = int(person["positions_m"].shape[0])
    positions, normals = person["positions_m"].numpy(), person["mesh_normals"].numpy()
    visible, joints = person["visible"].numpy(), person["joints_m"].numpy()
    for frame in range(frames):
        np.savez(sequence_dir / "person_frames" / f"frame_{frame:03d}.npz",
                 pcl=np.concatenate([positions[frame], normals[frame]], axis=-1).astype(np.float32),
                 joints_xyz=joints[frame].astype(np.float32), visible=visible[frame])
    (sequence_dir / "metadata.json").write_text(json.dumps(inputs["person_sequence_metadata"], indent=2) + "\n")
    torch.save(stored["velocity_field"], ws / "velocity_field.pt")
    torch.save(stored["person_material"], ws / "person_material.pt")
    np.save(ws / "scene_visibility.npy", scene["visible"].numpy())
    coupling_dir = ws / "antenna_coupling"
    coupling_dir.mkdir(exist_ok=True)
    for name, block in radar["antenna_coupling"].items():
        np.save(coupling_dir / f"{name}_coupling.npy", block.numpy())
    (ws / "motion_detection.json").write_text(json.dumps(inputs["motion_frames"]["detection"]) + "\n")
    (ws / "pose_audit.json").write_text(json.dumps(inputs["motion_frames"]["pose_audit"]) + "\n")
    product = Path(output_dir) / "rendered"
    rewrite = {
        "--sequence-dir": sequence_dir, "--person-checkpoint": ws / "person_material.pt",
        "--motion-checkpoint": ws / "velocity_field.pt", "--product-dir": product,
        "--output": Path(output_dir) / "render.mp4", "--scene-visibility": ws / "scene_visibility.npy",
        "--antenna-coupling-calibration": coupling_dir, "--motion-detection-result": ws / "motion_detection.json",
        "--pose-audit-result": ws / "pose_audit.json", "--device": device,
    }
    for i, flag in enumerate(argv[:-1]):
        if flag in rewrite:
            argv[i + 1] = str(rewrite[flag])
    (Path(output_dir) / "render_argv.json").write_text(json.dumps(argv) + "\n")
    run_module("mm4DPS.video", *argv, repo=root, log=Path(output_dir) / "rerender.log", workspace=ws)
    return product


def _artifacts(out: Path, work: Path, sequence: int, render_tag: str,
               mask: bool, repo: Path, log: Path) -> None:
    """The media folders for one sequence, from the working products.

    HuPR: the velocity comparison and the wall-band occlusion check. RT-Pose:
    the velocity comparison, the GVHMR mesh on the left camera's video, and
    the scene + person points without and with the visibility mask.
    """
    import shutil
    dataset = current_dataset()
    arm = "maskon" if mask else "maskoff"
    vdir = out / f"seq{sequence}_{arm}"
    odir = out / f"seq{sequence}_{arm}_occlusion"
    for d in (vdir, odir):
        if d.exists():
            shutil.rmtree(d)
    run_module("mm4DPS.media", "velocity-comparison-media",
               "--outputs-root", work, "--sequence", sequence,
               "--runs", f"{arm}=treat{sequence}",
               "--render-name", f"seq{sequence}_{render_tag}_600",
               "--spectrogram-variant", f"hw10_{render_tag}",
               "--output-dir", vdir, repo=repo, log=log)
    if dataset.name != "hupr":
        treat = work / f"treat{sequence}"
        person = treat / f"person_sequence_{sequence}_v2clean"
        translation = json.loads(
            (treat / f"calibration/seq{sequence}_person_range_v2/result.json").read_text()
        )["selected_geometry_translation_y_m"]
        run_module("mm4DPS.media", "mesh-on-rgb", "--sequence", sequence,
                   "--sequence-dir", person, "--output", vdir / f"seq{sequence}_mesh_on_rgb.mp4",
                   "--repo", repo, repo=repo, log=log)
        run_module("mm4DPS.media", "scene-and-person", "--sequence-dir", person,
                   "--translation-y-m", translation,
                   "--checkpoint", treat / f"training/seq{sequence}_person_v2/best_shared_human_material.pt",
                   "--scene-mask", treat / "visibility" / f"seq{sequence}_s2_shadow.npy",
                   "--output", vdir / f"seq{sequence}_scene_and_person.mp4", repo=repo, log=log)
        return
    run_module("mm4DPS.media", "occlusion-check-media",
               "--source-dir", vdir, "--outputs-root", work,
               "--sequences", sequence, "--workspace-prefix", "treat",
               "--render-suffix", f"{render_tag}_600", "--with-rendered",
               "--right-panel-label", f"mask-{'on' if mask else 'off'}",
               "--output-dir", odir, repo=repo, log=log)


def evaluate_products(treat: str, sequence: int, repo: Path | None = None,
                      emit: str = "both", render_tag: str = "v2") -> None:
    """Run the eval suite on the finished products and render the paper tables.

    Group A of docs/LEAN_ACTION_PLAN.md: evaluation/measures.py is the engine
    behind every column of `tab:fidelity`, and before this stage existed the
    acceptance run never drove it -- it read 37 of 278 statements executed and 84
    of 86 branches never taken, which made it look dead when it was simply not
    exercised.
    """
    root = repo or REPO
    workspace = root / "outputs" / treat
    out = workspace / "deliverables" / "evaluation"
    run_module("mm4DPS.evaluation", "evaluate",
               "--workspace", workspace, "--dataset", current_dataset().name, "--out", out,
               # EXACT, not seq<S>_*_600. That wildcard matches
               # rendered/seq<S>_v3_600, the SEED directory setup_workspace
               # creates to carry the fitted absolute gain: it holds a
               # render_result.json and no products, so the suite died on a
               # missing .npy. Only the tagged noisy render is scored.
               "--product-glob", f"rendered/seq{sequence}_{render_tag}_600",
               "--sequence-dir-template",
               str(workspace / f"person_sequence_{sequence}_v2clean"),
               "--spectrogram-dir-template",
               str(workspace / f"deliverables/microdoppler_seq{sequence}/hw10_v2vis"),
               "--frame-selection", "pose-speed", "--emit", emit,
               repo=root, log=workspace / "deliverables" / f"evaluate_seq{sequence}.log",
               workspace=workspace)


def scene_shadow_mask(treat: str, sequence: int,
                      repo: Path | None = None) -> None:
    """Cast the per-frame shadow mask of the background points against the person.

    The mask and the trainer read the same checkpoint and the same placement
    file, so they cannot disagree (the trainer refuses a mask cast at another
    translation against its geometry). The scaffold is built in the radar
    frame -- the LiDAR one through the extrinsic seed, the monocular one
    through the fitted scene alignment -- and the background model is fitted
    on it in-chain, so nothing is moved here.
    """
    root = repo or REPO
    target = root / "outputs" / treat
    checkpoint = target / f"training/seq{sequence}_person_v2/best_shared_human_material.pt"
    translation = json.loads(
        (target / f"calibration/seq{sequence}_person_range_v2/result.json").read_text()
    )["selected_geometry_translation_y_m"]
    (target / "visibility").mkdir(parents=True, exist_ok=True)
    mask = run_step("mm4DPS.pointset", "scene-shadow-mask",
                    "--checkpoint", checkpoint,
                    "--sequence-dir", target / f"person_sequence_{sequence}_v2clean",
                    "--sequence", sequence, "--translation-y-m", translation,
                    "--output", target / "visibility" / f"seq{sequence}_s2_shadow.npy",
                    repo=root, capture=True, workspace=target)
    (target / "visibility" / f"mask_seq{sequence}.json").write_text(mask)
    # Was mm4DPS/preprocessing/visibility_report.py, a module that existed only
    # so this formatting lived in Python rather than in a shell script's quoting.
    # The shell scripts became this file, and nothing called it.
    report = json.loads(mask)
    seconds, shadowed = report["seconds"], report["shadowed_per_frame"]
    note(f"[visibility] {report['frames']} frames, "
         f"{report['background_points']} points, "
         f"{seconds['visibility']:.1f} s "
         f"({seconds['per_frame_ms']:.1f} ms/frame), "
         f"shadowed/frame mean {shadowed['mean']} "
         f"(min {shadowed['min']}, max {shadowed['max']})")


if __name__ == "__main__":
    raise SystemExit(main())
