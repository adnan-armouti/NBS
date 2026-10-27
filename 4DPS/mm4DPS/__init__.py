"""HuPR dual-radar differentiable rendering.

One importable package. `docs/CONSOLIDATION_AND_OO_PLAN.md` is flattening the
twelve subpackages into ~20 modules with a 400-line floor; this file is the
package's own vocabulary and holds three things:

    Install       where THIS machine keeps the dataset, models and toolkit
    Workspace     one run's output layout, and SequenceDir under it
    the radar     the IWR1843BOOST hardware description and config builder

THE RULE FOR THIS FILE: **define, never resolve.** Class and table definitions
cost nothing at import. Environment-derived *values* must not be computed here at
import time, because `mm4DPS/__init__.py` runs before anything else can call
`apply()` -- and `paths.py` and `environment.py` disagreed about
`HUPR_DATASET_ROOT`, so which one won depended on the placement of three import
statements (`docs/CONSOLIDATION_AND_OO_PLAN.md` and `docs/OO_REFACTOR_PLAN.md`
0.1). The back-compat names below are therefore resolved through module
`__getattr__` on first ACCESS, not at import. That is what makes the merge safe;
turning any of them into a plain module-level assignment reinstates the defect.

World convention
----------------
The initial rig frame is right handed: +Y is radar boresight, +X is image
right/horizontal azimuth, and +Z is up.  The horizontal radar's long virtual
array lies on +X.  HuPR Figure 3 shows the vertical board rotated clockwise;
with the TI-documented RX1-to-RX4 ordering this is +90 degrees about +Y and
places the same array on -Z.  The roll and approximate 3 cm baseline remain CLI
options because HuPR does not publish metric extrinsics.

The HuPR paper's Table 1 swaps the physical TX/RX labels (4 TX, 3 RX).  The
IWR1843 device and the released ``process_iwr1843.py`` unambiguously use 3 TX
and 4 RX.  Renderer arrays therefore follow the hardware/code (3 TX, 4 RX),
while the table values are retained under ``hupr_provenance`` for auditing.
"""

from __future__ import annotations

import json
import math
import os
import sys
from argparse import ArgumentParser
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

#: The repository root. Anchor paths on this, never on ``__file__`` depth:
#: counting parents has silently broken this tree three times (the unit/gates
#: test split, the qa move, and hupr_cuda_render losing third_party/ when it
#: moved into mm4DPS/cuda/ -- which took the CUDA extension out of the build and
#: turned 22 gate tests into skips without failing anything).
REPO_ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------
# Install: where this machine keeps things
# ----------------------------------------------------------------------
DEPTH_ANYTHING_REVISION = "d2fc6a93601aabb1139a3bf0ebfcb4e89c67817f"
DEPTH_ANYTHING_REPO = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"

#: Not derivable from the checkout. Override from the environment on another box.
DEFAULT_CUDA_HOME = "/usr/local/cuda-12.8"
DEFAULT_DATASET_ROOT = "/home/adnan/Documents/Data/HuPR"
DEFAULT_RTPOSE_ROOT = "/data/datasets/RT-Pose"
DEFAULT_OCTONET_ROOT = "/data/mmsd/datasets/OctoNet/OctoNet-upload"   # the node; the workstation copy was deleted


def _env_path(variable: str, fallback: str) -> Path:
    return Path(os.environ.get(variable, fallback)).expanduser()


def _dataset_fallback() -> Path:
    """Three places, in order, so a self-contained copy needs no configuration.

    HuPR is not ours to redistribute and the repacked radar archives are 0.88 GB
    each against GitHub's 100 MB per-file limit, so the dataset is never in the
    repository.

      1. $HUPR_DATASET_ROOT      set it and nothing else matters
      2. <repo>/data/HuPR        drop or symlink a copy beside the code
      3. this machine's install
    """
    candidates = [REPO_ROOT / "data" / "HuPR",
                  Path("/mnt/ssd2tb/data/HuPR_subset"),
                  Path("/mnt/ssd2tb/data/HuPR")]
    return next((c for c in candidates if (c / "radar").is_dir()), candidates[0])


@dataclass(frozen=True)
class Install:
    """Where this machine keeps the dataset, the models and the toolkit.

    Built by `from_env()` at an ENTRY POINT and passed down, never resolved at
    import. Frozen: a stage that is handed one cannot repoint it halfway through
    a run.
    """

    repo: Path
    dataset_root: Path
    mm3dps_root: Path
    smpl_dir: Path
    depth_anything: Path
    hupr_official: Path
    config_dir: Path
    cuda_home: Path
    workspace: Path

    @classmethod
    def from_env(cls, **overrides) -> "Install":
        """Resolve from the environment, with this checkout as the fallback.

        `apply()` should have run first if the caller wants this process's
        defaults exported to its children; `from_env()` itself mutates nothing.
        """
        hf_home = Path(os.environ.get("HF_HOME", str(REPO_ROOT / "models" / "hf")))
        packaged_smpl = REPO_ROOT / "models" / "smpl"
        vendored_official = REPO_ROOT / "third_party" / "hupr_official"
        values = dict(
            repo=REPO_ROOT,
            dataset_root=_env_path("HUPR_DATASET_ROOT", str(_dataset_fallback())),
            mm3dps_root=_env_path("MM3DPS_ROOT", str(Path(__file__).resolve().parent / "splat")),
            smpl_dir=_env_path("SMPL_MODEL_DIR", str(
                packaged_smpl if packaged_smpl.is_dir() else "/mnt/ssd2tb/models/smpl")),
            depth_anything=_env_path("DEPTH_ANYTHING_PATH", str(
                hf_home / "hub"
                / f"models--{DEPTH_ANYTHING_REPO.replace('/', '--')}"
                / "snapshots" / DEPTH_ANYTHING_REVISION)),
            hupr_official=_env_path("HUPR_OFFICIAL_ROOT", str(
                vendored_official if vendored_official.is_dir()
                else "/mnt/ssd2tb/code/multidataset_experiments/external/hupr_official")),
            config_dir=_env_path("HUPR_CONFIG_DIR", str(REPO_ROOT / "configs")),
            cuda_home=_env_path("CUDA_HOME", DEFAULT_CUDA_HOME),
            workspace=_env_path("HUPR_WORKSPACE", str(REPO_ROOT / "outputs" / "workspace")),
        )
        values.update({k: Path(v) for k, v in overrides.items()})
        return cls(**values)

    def describe(self) -> dict[str, str]:
        """What each location resolved to, and whether it exists."""
        return {name: f"{value}  [{'ok' if value.exists() else 'MISSING'}]"
                for name, value in vars(self).items()}

    def missing(self) -> list[str]:
        return [name for name, value in vars(self).items() if not value.exists()]


def apply() -> dict[str, str]:
    """Export this machine's defaults so child processes inherit them.

    Replaces `hupr_env.sh`, which was tracked in git and hardcoded absolute
    paths into whichever checkout committed it: every chain script sourced it, so
    a fresh clone put the SOURCE tree on PYTHONPATH and silently ran that tree's
    code -- finding 1 of `docs/REPRODUCTION_AND_COVERAGE.md`. Deriving from
    `REPO_ROOT` makes that impossible.

    `setdefault`, so a caller's own value always wins.
    """
    hf_home = REPO_ROOT / "models" / "hf"
    resolved = {
        "HF_HOME": str(hf_home),
        "DEPTH_ANYTHING_PATH": str(
            hf_home / "hub"
            / f"models--{DEPTH_ANYTHING_REPO.replace('/', '--')}"
            / "snapshots" / DEPTH_ANYTHING_REVISION),
        "CUDA_HOME": DEFAULT_CUDA_HOME,
        "HUPR_DATASET_ROOT": DEFAULT_DATASET_ROOT,
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
    }
    for key, value in resolved.items():
        os.environ.setdefault(key, value)
    return {key: os.environ[key] for key in resolved}


# ----------------------------------------------------------------------
# Workspace: one run's output layout, owned in one place
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SequenceDir:
    """One person sequence on disk.

    `person_frames/frame_%03d.npz` was spelled out in EIGHT modules before this
    existed. One spelling, here.
    """

    root: Path

    @property
    def person_frames(self) -> Path:
        return self.root / "person_frames"

    def frame(self, index: int) -> Path:
        return self.person_frames / f"frame_{index:03d}.npz"

    def frames(self) -> list[Path]:
        return sorted(self.person_frames.glob("frame_*.npz"))

    def metadata(self) -> dict:
        return json.loads((self.root / "metadata.json").read_text())

    def frame_count(self) -> int:
        return int(self.metadata()["frames"])

    def has_person_stack(self) -> bool:
        return self.person_frames.is_dir()


@dataclass(frozen=True)
class Workspace:
    """One run's outputs.

    Frozen, and with no method that removes `rendered/`: it holds the seeded
    absolute gain and the chain aborts without it.
    """

    root: Path

    @classmethod
    def for_run(cls, run: str, outputs_root: Path | None = None) -> "Workspace":
        base = Path(outputs_root) if outputs_root is not None else REPO_ROOT / "outputs"
        return cls(base / run)

    #: Exactly what `chain.setup_workspace` creates. Kept beside the accessors
    #: so the two cannot drift; the first draft of this tuple guessed and was
    #: wrong in two entries.
    LAYOUT = ("calibration", "training", "scaffolds", "rendered",
              "deliverables", "person_sequences_raw")

    def sequence(self, name: str) -> SequenceDir:
        return SequenceDir(self.root / name)

    def render(self, sequence: int, suffix: str) -> Path:
        return self.root / "rendered" / f"seq{sequence}_{suffix}"

    def rendered(self) -> Path:
        return self.root / "rendered"

    def training(self) -> Path:
        return self.root / "training"

    def calibration(self) -> Path:
        return self.root / "calibration"


# ----------------------------------------------------------------------
# Back-compat module attributes, resolved on ACCESS (PEP 562).
#
# `from mm4DPS import DATASET_ROOT` still works and still reads the environment
# at the moment it is asked, exactly as the old lazy `from mm4DPS.paths import
# ...` inside a function body did. A plain module-level assignment here would
# resolve during `import mm4DPS` -- before any entry point can call `apply()` --
# and silently change the dataset root. Consolidation batches replace these with
# an explicit `Install`; the last batch deletes them.
# ----------------------------------------------------------------------
_LAZY = {
    "DATASET_ROOT": "dataset_root", "MM3DPS_ROOT": "mm3dps_root",
    "SMPL_MODEL_DIR": "smpl_dir", "DEPTH_ANYTHING_PATH": "depth_anything",
    "HUPR_OFFICIAL_ROOT": "hupr_official", "CONFIG_DIR": "config_dir",
    "WORKSPACE": "workspace", "CUDA_HOME": "cuda_home",
}


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(Install.from_env(), _LAZY[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY))


# ----------------------------------------------------------------------
# Flag groups shared by the command-line entry points, and their front door.
#
# Was cli.py. Sixty-five modules once built a parser and respelled the same
# flags -- the radar config pair thirteen times, --sequence-dir twelve,
# --pose-audit eight. Every default below resolves when the parser is BUILT,
# never at import: cli.py computed two of them at module level, which is fine
# in a leaf module and is the C2 defect when done here.
# ----------------------------------------------------------------------
def radar_configs(parser: ArgumentParser, dataset: "Dataset | None" = None) -> ArgumentParser:
    """``--dataset`` and one ``--<radar>-config`` per radar it recorded.

    The configs describe the hardware, not a run. Which radars exist is read
    from ``--dataset`` before this parser is built (:func:`dataset_from_argv`),
    so HuPR gets ``--horizontal-config``/``--vertical-config`` exactly as before
    and RT-Pose gets ``--cascade-config``.
    """
    dataset = dataset or dataset_from_argv()
    config_dir = Install.from_env().config_dir
    if "--dataset" not in parser._option_string_actions:
        parser.add_argument("--dataset", default=dataset.name, choices=DATASETS,
                            help="which dataset's radars, scene source and layout "
                                 "apply (default: $MM4DPS_DATASET, else hupr)")
    for radar in dataset.radars:
        parser.add_argument(radar.flag("", "config"), type=Path,
                            default=config_dir / radar.config)
    return parser


def sequence_dir(parser: ArgumentParser) -> ArgumentParser:
    """``--sequence-dir``: the person sequence a run reads."""
    parser.add_argument("--sequence-dir", type=Path,
                        default=Install.from_env().workspace / "person_sequence_1_first_loop")
    return parser


def pose_audit(parser: ArgumentParser) -> ArgumentParser:
    """``--pose-audit``: what restricts every metric to moving frames (this sequence's own)."""
    parser.add_argument("--pose-audit", type=Path, required=True)
    return parser


def dispatch(commands: dict, module: str) -> int:
    """Run the subcommand named by the first argument.

    The merged modules each hold several former executables, and this is the
    front door: it pops the subcommand and leaves the rest of argv to that
    command's own parser, so every flag is the one it always had. Eighteen
    copies of this function existed until they were folded back here.
    """
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("usage: python -m %s <command> [options]\n\ncommands:\n  %s"
              % (module, "\n  ".join(commands)), file=sys.stderr)
        return 2
    command = sys.argv.pop(1)
    return commands[command]() or 0


def describe() -> dict[str, str]:
    """What the locations resolved to. Kept for the environment check."""
    return Install.from_env().describe()


def peak_vram() -> dict:
    """`torch.cuda.max_memory_allocated`, in MiB, for this process.

    Reported beside the `nvidia-smi` figure rather than instead of it. This one
    is what the model held; nvidia-smi is what the process reserved from the
    driver, which adds the CUDA context, the caching allocator's free blocks
    and fragmentation. At 23-24 GiB of a 24.0 GiB card the gap between them is
    the entire remaining margin, so a budget needs both.

    torch is imported inside the call so that importing the package stays cheap.
    """
    import torch

    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "torch_max_memory_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
        "torch_max_memory_reserved_mib": torch.cuda.max_memory_reserved() / 2 ** 20,
    }


# --------------------------------------------------------------------
# The radar: IWR1843BOOST hardware, and the config pair it builds
# --------------------------------------------------------------------

C_LIGHT_M_S = 299_792_458.0

# HuPR Table 1 / released-code constants.
CARRIER_FREQUENCY_HZ = 77.0e9  # exact start/centre frequency is not published
FREQUENCY_SLOPE_HZ_S = 60.012e12  # Table: 60.012 MHz/us (printed as MHz)
SAMPLE_RATE_HZ = 4.4e6
CHIRP_DURATION_S = 72e-6
ADC_SAMPLES = 256
TDM_LOOPS_PER_FRAME = 64
PHYSICAL_CHIRPS_PER_FRAME = TDM_LOOPS_PER_FRAME * 3
FRAMES_PER_SECOND = 10
NUM_TX = 3
NUM_RX = 4
# HuPR uses one compact IWR1843BOOST rather than the earlier cascaded array.
# Keep the cascaded renderer default at 1.5 m and opt HuPR into this guard.
HUPR_NEAR_FIELD_M = 0.5


# ----------------------------------------------------------------------
# Radar and Dataset: the hardware description every stage reads
#
# 241 sites across 16 modules once spelled HuPR's two-single-chip layout out
# by hand -- ("horizontal", "vertical"), NUM_TX = 3, (64, 3, 4, 256). RT-Pose
# has one 12x16 cascade. A stage that needs to know which radars exist, how
# many channels each has or where its config lives asks the Dataset; nothing
# below this comment is HuPR-specific except the two instances at the bottom.
# ----------------------------------------------------------------------
DATASETS = ("hupr", "rtpose", "octonet")
DATASET_ENV = "MM4DPS_DATASET"
N_ANGLE_OVERRIDE_ENV = "MM4DPS_N_ANGLE_OVERRIDE"
TRAINING_ARRAY_ENV = "MM4DPS_TRAINING_ARRAY"


@dataclass(frozen=True)
class Radar:
    """One radar: its chirp, its TDM schedule, its array and its recording.

    ``name`` is the product prefix (``<name>_rendered.npy``) and the infix of
    every per-radar flag (``--<name>-config``, ``--absolute-gain-<name>``).
    Positions live in the config file, not here: the renderer reads
    ``tx_array``/``rx_array`` from it, and ``board_rotation_deg`` is applied
    when that file is BUILT, so a config on disk is always what the renderer
    sees.
    """

    name: str
    config: str            # file name under Install.config_dir
    member: str            # the recording's tag in the dataset (HuPR: hori/vert)
    n_tx: int
    n_rx: int
    tx_order: tuple[int, ...]    # TDM slot k carries physical TX tx_order[k]
    azimuth_tx: tuple[int, ...]  # the TX whose RX rows form the azimuth array
    loops: int                   # TDM loops per frame (slow time)
    samples: int                 # ADC samples per chirp
    sample_rate_hz: float
    slope_hz_s: float
    carrier_hz: float
    chirp_duration_s: float
    frame_rate_hz: float
    decoder: str                 # "dca1000" (HuPR) | "tda2" (cascade)
    fov_zero_db_deg: float       # two-way element pattern's 0 dB crossing
    #: Antenna phase centres in half-wavelengths of ``design_hz``, on the
    #: board BEFORE ``board_rotation_deg``: (azimuth, elevation) per physical
    #: TX index and per recorded RX channel. The config builder places them;
    #: the beamformer's virtual-array grid is derived from them.
    tx_az: tuple[int, ...] = ()
    tx_el: tuple[int, ...] = ()
    rx_az: tuple[int, ...] = ()
    rx_el: tuple[int, ...] = ()
    design_hz: float = 77.0e9
    n_angle: int = 64            # beam-space bins of the azimuth angle FFT (the product's / the evaluation's)
    #: the velocity trainer's azimuth angle-FFT size (docs/FIDELITY_FIX_RESULTS.md 3.4): the
    #: lossless 2N-1 of an N-slot row for a magnitude L2 (HuPR 8 slots -> 16), the slot count
    #: for the cascade (86); the product size is interpolation the loss does not need
    training_n_angle: int = 16
    board_rotation_deg: float = 0.0
    calibration: str | None = None  # cascade TX-phase calibration file
    #: A TRAINING SUB-ARRAY (see training_subset): the slot each kept TX
    #: fires in, and how many slots the recorded loop has. None on a full
    #: radar, where tx_order carries the same information.
    tx_slots: tuple[int, ...] | None = None
    loop_slots: int | None = None
    #: OctoNet (2026-10-27): the waveform. "fmcw" = everything above (chirp slope, ADC samples, TDM loops);
    #: "sfcw" = a stepped-frequency sweep of n_freq tones from f_start_hz to f_stop_hz, one complex sample per
    #: tone and Tx-Rx pair, no slow time (one sweep per frame): ``samples`` then counts the tones, ``loops`` is 1,
    #: and the range axis is the IFFT over the tones (bin c / 2B). The FMCW-only members are 0 and unused.
    modulation: str = "fmcw"
    f_start_hz: float = 0.0
    f_stop_hz: float = 0.0
    n_freq: int = 0

    @property
    def chirps(self) -> int:
        """Physical chirps per frame: every slot of every loop (the recording's, for a sub-array too)."""
        return (self.loop_slots or self.n_tx) * self.loops

    @property
    def adc_frame_shape(self) -> tuple[int, int, int]:
        """The decoded frame: ``(rx, chirp, sample)``."""
        return (self.n_rx, self.chirps, self.samples)

    @property
    def raw_words_per_frame(self) -> int:
        return self.n_rx * self.chirps * self.samples * 2

    @property
    def bytes_per_frame(self) -> int:
        return self.raw_words_per_frame * 2

    @property
    def product_shape(self) -> tuple[int, int, int, int]:
        """One frame of the shipped product: ``(loop, tx, rx, sample)``."""
        return (self.loops, self.n_tx, self.n_rx, self.samples)

    @property
    def virtual_channels(self) -> int:
        """Elements of the azimuth virtual array the beamformer uses."""
        return len(self.azimuth_tx) * self.n_rx

    @property
    def wavelength_m(self) -> float:
        return C_LIGHT_M_S / self.carrier_hz

    @property
    def bandwidth_hz(self) -> float:
        if self.modulation == "sfcw":
            return self.f_stop_hz - self.f_start_hz
        return self.slope_hz_s * self.samples / self.sample_rate_hz

    @property
    def range_bin_m(self) -> float:
        if self.modulation == "sfcw":
            return C_LIGHT_M_S / (2.0 * self.bandwidth_hz)                 # c / 2B: 3.75 cm for the Vayyar
        return C_LIGHT_M_S * self.sample_rate_hz / (2.0 * self.slope_hz_s * self.samples)

    @property
    def unambiguous_range_m(self) -> float:
        """SFCW: c / (2 df); FMCW: the full range axis."""
        if self.modulation == "sfcw":
            return C_LIGHT_M_S / (2.0 * self.bandwidth_hz / (self.n_freq - 1))
        return self.range_bin_m * self.samples

    @property
    def loop_period_s(self) -> float:
        return (self.loop_slots or self.n_tx) * self.chirp_duration_s

    def slot_of_tx(self) -> tuple[int, ...]:
        """The loop slot each TX (by index) fires in."""
        if self.tx_slots is not None:
            return tuple(self.tx_slots)
        slots = [0] * self.n_tx
        for slot, tx in enumerate(self.tx_order):
            slots[tx] = slot
        return tuple(slots)

    @property
    def is_training_subset(self) -> bool:
        return self.tx_slots is not None

    def training_subset(self) -> "Radar":
        """The radar with only the TX that form the azimuth array.

        Every loss the velocity-field trainer computes -- the loop-0 RA, the
        Doppler-plane RA and the micro-Doppler marginal -- goes through the
        beamformer, which keeps `azimuth_tx` and drops the rest before the
        angle FFT. So the elevated TX (HuPR's TX1; the cascade's TX 0-2) are
        rendered every step and discarded: 1/3 of HuPR's paths, 1/4 of the
        cascade's, with no gradient through them. The trainer renders this
        sub-array instead. The kept TX keep their loop slots (tx_slots), so
        the slow-time phase is exactly the full array's; the products are
        still rendered with the full array by the renderer.
        """
        if self.is_training_subset:
            return self
        keep = tuple(self.azimuth_tx)
        slots = self.slot_of_tx()
        # The angle FFT keeps the radar's n_angle. An FFT of exactly the
        # virtual array's grid length (the cascade: 86 slots for 128) was
        # measured: 100 ms a step against 98 -- the step's cost is in the
        # Doppler-plane splat over paths x planes, not in the angle bins.
        return dataclasses.replace(
            self, n_tx=len(keep), tx_order=tuple(range(len(keep))), azimuth_tx=tuple(range(len(keep))),
            tx_az=tuple(self.tx_az[t] for t in keep) if self.tx_az else (),
            tx_el=tuple(self.tx_el[t] for t in keep) if self.tx_el else (),
            tx_slots=tuple(slots[t] for t in keep), loop_slots=self.loop_slots or self.n_tx)

    def config_subset(self, config: dict) -> dict:
        """A full radar's config with only this sub-array's TX entries."""
        if not self.is_training_subset:
            return config
        # The RECORDED radar, asked for explicitly: in the trainer's process
        # `Dataset.named` resolves to the sub-array by default, whose
        # `azimuth_tx` is already re-indexed (0..n-1) -- read from there this
        # kept the config's first n entries, an elevated TX among them.
        full = Dataset.named(current_dataset_name(), training_array=False).radar(self.name)
        keep = list(full.azimuth_tx)
        reduced = dict(config)
        reduced["tx_array"] = [config["tx_array"][t] for t in keep]
        reduced["training_subset"] = {"kept_tx": keep, "tx_slots": list(self.tx_slots),
                                      "loop_slots": self.loop_slots,
                                      "kept_names": [t.get("name", str(i)) for i, t in
                                                     zip(keep, reduced["tx_array"])]}
        return reduced

    def rotated_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """``(rx, tx)`` phase centres as (azimuth, elevation) half-wavelengths, after the board rotation.

        The only rotations a board on a rig takes are 0 and 180 degrees about
        boresight; 180 negates both coordinates (3DPS's ``flip_upside_down``).
        """
        rx = np.array(list(zip(self.rx_az, self.rx_el)), dtype=float)
        tx = np.array(list(zip(self.tx_az, self.tx_el)), dtype=float)
        if self.board_rotation_deg % 360 == 180:
            rx, tx = -rx, -tx
        elif self.board_rotation_deg % 360 != 0:
            raise ValueError(f"board_rotation_deg must be 0 or 180, got {self.board_rotation_deg}")
        return rx, tx

    def azimuth_grid(self) -> tuple[np.ndarray, int]:
        """Where each azimuth virtual channel sits on the half-wavelength grid.

        Virtual channel ``v = i * n_rx + r`` (azimuth TX ``azimuth_tx[i]``,
        RX ``r``, the order :func:`virtual_array_slow_time` builds) has
        azimuth phase centre ``tx_az + rx_az``; the grid starts at the
        smallest one. Returns ``(slot per virtual channel, grid length)``.
        HuPR's eight channels fill slots 0..7 exactly; the cascade's 144 fill
        86 slots, some of them twice.
        """
        rx, tx = self.rotated_positions()
        positions = np.array([tx[t, 0] + rx[r, 0] for t in self.azimuth_tx for r in range(self.n_rx)])
        slots = np.round(positions - positions.min()).astype(int)
        return slots, int(slots.max()) + 1

    def flag(self, prefix: str, suffix: str = "") -> str:
        """``--<prefix>-<name>-<suffix>``: the per-radar flag spelling."""
        parts = [part for part in (prefix, self.name, suffix) if part]
        return "--" + "-".join(parts)

    def attr(self, prefix: str, suffix: str = "") -> str:
        """The ``args`` attribute :meth:`flag` parses into."""
        return self.flag(prefix, suffix)[2:].replace("-", "_")


@dataclass(frozen=True)
class Dataset:
    """Which radars a dataset recorded, and where its scene and person come from."""

    name: str
    radars: tuple[Radar, ...]
    scene_source: str            # "depth" (monocular, HuPR) | "lidar" (RT-Pose)
    frame_count: int | None      # frames per sequence when fixed; None: per sequence
    root_env: str                # the environment variable naming its root
    training_array: bool = False  # the radars are training sub-arrays (Radar.training_subset)
    #: the RGB camera's horizontal field of view when the dataset publishes no intrinsics (the monocular scaffold's
    #: back-projection and the person stage's audit camera read it); None = the dataset's calibrated intrinsics
    camera_hfov_deg: float | None = None
    #: the dataset publishes 2D keypoints in the RGB frame (HuPR's hrnet_annot_*.json): the person stage registers the mesh's
    #: ray on them (--image-registration gt); a dataset without them keeps GVHMR's ray
    image_keypoints: bool = False
    #: the azimuth registration of the person against the radar's return: "sequence" = the per-sequence affine yaw fit
    #: (RT-Pose: 86 azimuth channels resolve it, ~0.5 deg residuals); "constant" = one rig yaw for the dataset
    #: (camera_radar_yaw_deg; HuPR: 8 azimuth channels, per-sequence fits scatter -2.6..+6.8 deg around it)
    azimuth_registration: str = "sequence"
    camera_radar_yaw_deg: float = 0.0
    #: how the radar's depth scale places the body along its ray: "frame" = radar_depth_scales' per-frame ratio (RT-Pose: the
    #: person walks, the 86-channel cascade's range track is reliable frame by frame); "sequence" = one number for the
    #: sequence, the median ratio over confident frames (HuPR: GVHMR's depth error is the camera's focal, a constant, and the
    #: 8-channel board's per-frame range of a slow / seated person is not -- seq 38's registered range jumped 3.1 -> 5.6 m on
    #: 216 of 600 frames under "frame", the person projecting at half size); the 2D-keypoint ray carries the per-frame position
    depth_scale_mode: str = "frame"

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(radar.name for radar in self.radars)

    def root(self) -> Path:
        """Where this dataset is on this machine: ``$<root_env>``, else the install's."""
        if self.name == "hupr":
            return Install.from_env().dataset_root
        if self.name == "octonet":
            return _env_path(self.root_env, DEFAULT_OCTONET_ROOT)
        return _env_path(self.root_env, DEFAULT_RTPOSE_ROOT)

    def sequence_dir(self, sequence: int) -> Path:
        """RT-Pose: ``<root>/sequences/<n>``; HuPR keeps its archives (find_archive)."""
        if self.name == "hupr":
            raise ValueError("HuPR sequences live in archives; use mm4DPS.data.find_archive")
        return self.root() / "sequences" / str(int(sequence))

    def radar(self, name: str) -> Radar:
        for radar in self.radars:
            if radar.name == name:
                return radar
        raise KeyError(f"{self.name} has no radar {name!r}; it has {self.names}")

    def config_path(self, radar: Radar | str, config_dir: Path | None = None) -> Path:
        radar = radar if isinstance(radar, Radar) else self.radar(radar)
        return (config_dir or Install.from_env().config_dir) / radar.config

    def configs(self, args) -> dict[str, dict]:
        """The parsed ``--<name>-config`` files, by radar name (a training sub-array's TX only)."""
        return {radar.name: radar.config_subset(json.loads(getattr(args, radar.attr("", "config")).read_text()))
                for radar in self.radars}

    def config_flags(self, args) -> dict[str, Path]:
        """The config file per radar; for a training sub-array, a reduced copy written beside the run's output."""
        paths = {radar.name: getattr(args, radar.attr("", "config")) for radar in self.radars}
        if not self.training_array:
            return paths
        out = Path(getattr(args, "output_dir", Install.from_env().workspace)) / "training_configs"
        out.mkdir(parents=True, exist_ok=True)
        reduced = {}
        for radar in self.radars:
            target = out / radar.config
            subset = radar.config_subset(json.loads(paths[radar.name].read_text()))
            target.write_text(json.dumps(subset, indent=2) + "\n")
            reduced[radar.name] = target
            # Which TX the rasterizer will place: named, so a wrong pick (an
            # elevated TX kept, an azimuth one dropped) is visible in the log.
            print(f"training sub-array config: {radar.name} keeps TX "
                  f"{subset['training_subset']['kept_names']} of {len(json.loads(paths[radar.name].read_text())['tx_array'])}",
                  flush=True)
        return reduced

    def training_subset(self) -> "Dataset":
        """This dataset with every radar reduced to its training sub-array."""
        if self.training_array:
            return self
        return dataclasses.replace(
            self, radars=tuple(r.training_subset() for r in self.radars), training_array=True)

    def full(self) -> "Dataset":
        """The recorded arrays, whatever this process trains on."""
        return _DATASETS[self.name]

    @staticmethod
    def named(name: str, training_array: bool | None = None) -> "Dataset":
        """The dataset by name; the training sub-array variant when ``$MM4DPS_TRAINING_ARRAY`` is 1."""
        if name not in _DATASETS:
            raise KeyError(f"unknown dataset {name!r}; known: {DATASETS}")
        dataset = _DATASETS[name]
        if training_array is None:
            training_array = os.environ.get(TRAINING_ARRAY_ENV) == "1"
        dataset = dataset.training_subset() if training_array else dataset
        # The trainer's angle-FFT size (docs/FIDELITY_FIX_PLAN.md, 2026-09-16): the velocity trainer
        # exports $MM4DPS_N_ANGLE_OVERRIDE from --n-angle so every radar it resolves -- and so both
        # sides of its loss -- beamform on that many bins instead of the radar's product size.
        # Nothing else sets it; the chain's other stages never inherit it.
        override = os.environ.get(N_ANGLE_OVERRIDE_ENV)
        if override:
            dataset = dataclasses.replace(
                dataset, radars=tuple(dataclasses.replace(r, n_angle=int(override)) for r in dataset.radars))
        return dataset

    @staticmethod
    def select(name: str, training_array: bool = False) -> "Dataset":
        """Make ``name`` THIS process's dataset and return it.

        Exports ``$MM4DPS_DATASET`` so that :func:`current_dataset` -- what a
        helper deep in a module reads when it was handed no radar -- agrees
        with the flag the entry point parsed, and so that child processes
        inherit it. ``training_array`` additionally exports
        ``$MM4DPS_TRAINING_ARRAY=1``: every radar this process resolves is
        then its training sub-array (the velocity-field trainer does this;
        nothing else may, and the chain's stage processes never inherit it
        because the trainer spawns none).
        """
        os.environ[DATASET_ENV] = name if name in _DATASETS else Dataset.named(name).name
        os.environ[TRAINING_ARRAY_ENV] = "1" if training_array else "0"
        return Dataset.named(name, training_array)

    @staticmethod
    def of(args) -> "Dataset":
        """The dataset an ``args`` namespace names (``--dataset``, default HuPR), selected.

        Keeps the process's training-sub-array state as it is.
        """
        return Dataset.select(getattr(args, "dataset", None) or os.environ.get(DATASET_ENV, "hupr"),
                              training_array=os.environ.get(TRAINING_ARRAY_ENV) == "1")


def per_radar(args, prefix: str, suffix: str = "") -> dict:
    """``{radar name: args.<prefix>_<name>_<suffix>}`` for the current dataset's radars.

    The reading side of the per-radar flags :func:`radar_configs` and the
    renderer's parser spell with :meth:`Radar.flag`.
    """
    return {radar.name: getattr(args, radar.attr(prefix, suffix))
            for radar in current_dataset().radars}


def current_dataset_name() -> str:
    return os.environ.get(DATASET_ENV, "hupr")


def current_dataset() -> Dataset:
    """The dataset this PROCESS is working on: ``$MM4DPS_DATASET``, else HuPR.

    The chain exports it to every stage it launches, so a helper deep in a
    module that was handed no radar still reads the right dataset's numbers.
    Resolved on each call, never at import. In the trainer's process this is
    the training sub-array variant (``$MM4DPS_TRAINING_ARRAY``).
    """
    return Dataset.named(current_dataset_name())


def dataset_from_argv(argv=None) -> Dataset:
    """Read ``--dataset`` BEFORE the full parser is built.

    The per-radar flags a parser registers depend on which radars exist, so
    the dataset has to be known first. A two-stage parse: this one looks only
    for ``--dataset`` and ignores everything else.
    """
    pre = ArgumentParser(add_help=False)
    pre.add_argument("--dataset", default=os.environ.get(DATASET_ENV, "hupr"), choices=DATASETS)
    known, _ = pre.parse_known_args(sys.argv[1:] if argv is None else argv)
    return Dataset.named(known.dataset)


def _hupr_radar(name: str, member: str, fov_zero_db_deg: float) -> Radar:
    return Radar(
        name=name, config=f"iwr1843_{name}.json", member=member,
        n_tx=NUM_TX, n_rx=NUM_RX, tx_order=(0, 1, 2), azimuth_tx=(0, 2),
        loops=TDM_LOOPS_PER_FRAME, samples=ADC_SAMPLES,
        sample_rate_hz=SAMPLE_RATE_HZ, slope_hz_s=FREQUENCY_SLOPE_HZ_S,
        carrier_hz=CARRIER_FREQUENCY_HZ, chirp_duration_s=CHIRP_DURATION_S,
        frame_rate_hz=FRAMES_PER_SECOND, decoder="dca1000",
        fov_zero_db_deg=fov_zero_db_deg,
        tx_az=(5, 7, 9), tx_el=(0, 1, 0), rx_az=(0, 1, 2, 3), rx_el=(0, 0, 0, 0),
        design_hz=CARRIER_FREQUENCY_HZ, n_angle=64, training_n_angle=16)


#: HuPR: two IWR1843BOOST boards, the second rolled 90 degrees about boresight.
#: The FOV numbers are the two-way element pattern's 0 dB crossings either side
#: of boresight, from assets/antenna_pattern/MMWCAS/{tx1_76,rx1_76}.npy: the
#: horizontal board's azimuth is the pattern's H-plane, the vertical board's
#: is its E-plane.
#: HuPR publishes no camera intrinsics. GVHMR runs at its default focal sqrt(w^2 + h^2) = 724 px on the 512 px frames, and
#: the radar finds the person at 0.70-0.73x GVHMR's depth on every test sequence (median 0.712 over the 21, 2026-09-21,
#: docs/DOWNSTREAM_RESULTS.md 7): the true focal is 724 x 0.712 = 515 px, a horizontal field of view of
#: 2 atan(256 / 515) = 52.8 degrees; the 2D-keypoint registration (--image-registration gt) then measured the focal the image
#: implies at the radar's range at 513-540 px over the 21 test sequences, median 525 px = 52.0 degrees, the value kept. The 40
#: degrees assumed before compressed the monocular scaffold 1.4x toward the
#: optical axis (the floor 0.3 m too high, the walls too narrow) and projected the registered mesh 1.3x too large in the
#: audit; the person's own placement is invariant to it (the ray (u - c) / f and the radar's depth scale cancel).
HUPR_CAMERA_HFOV_DEG = 52.0
#: the camera -> radar yaw of the HuPR rig, in the azimuth registration's convention (positive increases the person's
#: spatial sine): the median of the per-sequence affine fits over the 16 test sequences that fitted (2026-09-21: +2.0..+4.9 on
#: 14, +6.8 on 245, -2.6 on 251, the mirror on 15); the per-sequence values are the 8-channel board's noise around it
HUPR_CAMERA_RADAR_YAW_DEG = 3.05
HUPR = Dataset(
    name="hupr",
    radars=(_hupr_radar("horizontal", "hori", 68.2), _hupr_radar("vertical", "vert", 20.6)),
    scene_source="depth", frame_count=600, root_env="HUPR_DATASET_ROOT", camera_hfov_deg=HUPR_CAMERA_HFOV_DEG, image_keypoints=True,
    azimuth_registration="constant", camera_radar_yaw_deg=HUPR_CAMERA_RADAR_YAW_DEG, depth_scale_mode="sequence")

#: RT-Pose: one TI MMWCAS-RF-EVM cascade. Chirp and TDM schedule from the
#: RT-POSE repository's hardware_param.m; the board is the 3DPS/ColoRadar
#: board rotated 180 degrees about boresight (paper Fig. 1), which the config
#: builder applies to every antenna position and a QA command verifies from
#: data. The radar is described here; its decoder and config arrive with it.
RTPOSE = Dataset(
    name="rtpose",
    radars=(Radar(
        name="cascade", config="mmwcas_cascade.json", member="cascade",
        # hardware_param.m: TxToEnable = [12 .. 1], so slot k carries TX 12-k;
        # TX 4..12 (0-based 3..11) sit at elevation 0 and form the azimuth array.
        # chirp_duration_s is the chirp INTERVAL, ramp 60 us + idle 5 us.
        n_tx=12, n_rx=16, tx_order=tuple(range(11, -1, -1)), azimuth_tx=tuple(range(3, 12)),
        loops=64, samples=256, sample_rate_hz=5.0e6, slope_hz_s=64.985e12,
        carrier_hz=77.0e9, chirp_duration_s=65.0e-6, frame_rate_hz=10,
        decoder="tda2", fov_zero_db_deg=68.2, board_rotation_deg=180.0,
        calibration="mmwcas_calibration.npz",   # TI calibrateResults_high.mat, beside the config
        # TI_Cascade_TX_position_{azi,ele} by physical TX; RX by RECORDED
        # channel (master 0-3, slave1 4-7, slave2 8-11, slave3 12-15), i.e.
        # TI_Cascade_RX_position_azi permuted by TI_Cascade_RX_ID. The same
        # geometry as ColoRadar's antenna_cfg.txt, in half-wavelengths of
        # F_design = 76.8 GHz.
        tx_az=(11, 10, 9, 32, 28, 24, 20, 16, 12, 8, 4, 0),
        tx_el=(6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        rx_az=(50, 51, 52, 53, 0, 1, 2, 3, 46, 47, 48, 49, 11, 12, 13, 14),
        rx_el=(0,) * 16, design_hz=76.8e9, n_angle=128, training_n_angle=86),),
    scene_source="lidar", frame_count=None, root_env="RTPOSE_DATASET_ROOT")

#: OctoNet (NeurIPS 2025 D&B, Scene 1): the Vayyar IMAGEVK-74 SFCW radar on node_1 (63-67 GHz in 100 tones, 20 Tx x
#: 20 Rx = 400 pairs, one sweep per frame at ~3.1 Hz, no slow time), with 3 RealSense D455 depth cameras (nodes 1/3/5)
#: and 125 Hz OptiTrack MoCap as the scene/person sources. UWB (Novelda X4M200, node_1) and Wi-Fi CSI (4 receivers)
#: are described in mm4DPS.octonet.SENSORS: they are not FMCW radars and get their own heads (octonet_heads).
#: The Tx element layout is UNVERIFIED (the coupling shows two Tx sub-groups): tx_az/rx_az are placeholders and no
#: azimuth beamforming is used; only pair-wise range profiles (mm4DPS.octonet). Decoder "octonet" = the NBS loaders
#: (nbs/preprocessing/octonet_io) through mm4DPS.octonet.OctoNetSequence, not an ADC stream.
OCTONET = Dataset(
    name="octonet",
    radars=(Radar(
        name="vayyar", config="vayyar_imagevk74.json", member="vayyar",
        n_tx=20, n_rx=20, tx_order=tuple(range(20)), azimuth_tx=tuple(range(20)),
        loops=1, samples=100, sample_rate_hz=0.0, slope_hz_s=0.0, carrier_hz=65.0e9, chirp_duration_s=0.0,
        frame_rate_hz=3.1, decoder="octonet", fov_zero_db_deg=60.0,
        tx_az=tuple(range(20)), tx_el=(0,) * 20, rx_az=(0,) * 20, rx_el=tuple(range(20)),
        design_hz=65.0e9, n_angle=64, training_n_angle=20,
        modulation="sfcw", f_start_hz=63.0e9, f_stop_hz=67.0e9, n_freq=100),),
    scene_source="rgbd", frame_count=None, root_env="OCTONET_DATASET_ROOT",
    camera_hfov_deg=87.0,          # nominal D455 depth HFOV; OctoNet ships no intrinsics (and no RGB video)
    image_keypoints=False, azimuth_registration="constant", camera_radar_yaw_deg=0.0, depth_scale_mode="sequence")

_DATASETS = {"hupr": HUPR, "rtpose": RTPOSE, "octonet": OCTONET}


def rotation_about_y(degrees: float) -> np.ndarray:
    """Right-handed rotation around the common boresight (+Y)."""
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def iwr1843boost_layout_lambda() -> tuple[np.ndarray, np.ndarray]:
    """Return RX and TX phase centres in board (azimuth, boresight, elevation).

    The four receivers use lambda/2 spacing.  TX0 and TX2 are the two
    non-elevated transmitters used by HuPR for the eight-element azimuth ULA;
    their 2-lambda separation makes the virtual phase centres uniform at
    lambda/2.  TX1 supplies elevation information and is offset by lambda/2.

    Only relative phase-centre positions affect the renderer, so the complete
    layout is centred at its bounding-box centre.
    """
    rx = np.array([(0.5 * i, 0.0, 0.0) for i in range(NUM_RX)], dtype=float)
    tx = np.array(
        [
            (2.5, 0.0, 0.0),
            (3.5, 0.0, 0.5),
            (4.5, 0.0, 0.0),
        ],
        dtype=float,
    )
    all_positions = np.vstack((rx, tx))
    centre = (all_positions.min(axis=0) + all_positions.max(axis=0)) / 2.0
    rel = all_positions - centre
    return rel[:NUM_RX], rel[NUM_RX:]


def _entries(
    prefix: str,
    positions_m: np.ndarray,
    boresight: np.ndarray,
) -> list[dict]:
    rows = []
    for index, position in enumerate(positions_m, start=1):
        row = {
            "name": f"{prefix}{index}",
            "pos_mm": np.round(position * 1000.0, 6).tolist(),
            "boresight": np.round(boresight, 8).tolist(),
            "polarization": "V",
        }
        if prefix == "TX":
            row["radius"] = 1.0
        else:
            row["power_dBm"] = 12.0
        rows.append(row)
    return rows


def build_config(
    name: str,
    centre_m: Iterable[float],
    roll_deg: float,
) -> dict:
    """Build one mm3DPS-compatible config plus auditable HuPR metadata."""
    centre = np.asarray(tuple(centre_m), dtype=float)
    if centre.shape != (3,):
        raise ValueError(f"centre_m must have shape (3,), got {centre.shape}")
    rotation = rotation_about_y(roll_deg)
    rx_lambda, tx_lambda = iwr1843boost_layout_lambda()
    wavelength_m = C_LIGHT_M_S / CARRIER_FREQUENCY_HZ
    rx_world = centre + (rotation @ (rx_lambda * wavelength_m).T).T
    tx_world = centre + (rotation @ (tx_lambda * wavelength_m).T).T
    boresight = rotation @ np.array([0.0, 1.0, 0.0])

    range_resolution_m = (
        C_LIGHT_M_S * SAMPLE_RATE_HZ
        / (2.0 * FREQUENCY_SLOPE_HZ_S * ADC_SAMPLES)
    )
    max_unambiguous_range_m = (
        C_LIGHT_M_S * SAMPLE_RATE_HZ / (2.0 * FREQUENCY_SLOPE_HZ_S)
    )
    return {
        "modulation_scheme": "FMCW",
        "carrierFrequency": CARRIER_FREQUENCY_HZ,
        "sampleRate": SAMPLE_RATE_HZ,
        "freqSlope": FREQUENCY_SLOPE_HZ_S,
        "numAdcSamples": ADC_SAMPLES,
        "adcStartTime": 0.0,
        "rampEndTime": CHIRP_DURATION_S,
        "tx_array": _entries("TX", tx_world, boresight),
        "rx_array": _entries("RX", rx_world, boresight),
        "hupr_provenance": {
            "sensor": "TI IWR1843BOOST",
            "radar_name": name,
            "coordinate_frame": "+Y boresight, +X right, +Z up",
            "centre_m": centre.tolist(),
            "board_roll_about_boresight_deg": float(roll_deg),
            "table_1": {
                "physical_transmitters_printed": 4,
                "physical_receivers_printed": 3,
                "frequency_slope_mhz_per_us": 60.012,
                "sample_rate_ksps": 4400,
                "chirp_duration_us": 72,
                "chirps_in_frame": 64,
                "adc_samples": 256,
                "frames_per_second": 10,
                "maximum_range_m_approx": 11.0,
                "range_resolution_m_printed": 0.048,
            },
            "released_code": {
                "physical_transmitters": NUM_TX,
                "physical_receivers": NUM_RX,
                "tdm_loops_per_frame": TDM_LOOPS_PER_FRAME,
                "physical_chirps_per_frame": PHYSICAL_CHIRPS_PER_FRAME,
                "lvds_lanes": 2,
                "angle_fft_bins": 64,
                "elevation_fft_bins": 8,
            },
            "derived": {
                "wavelength_m": wavelength_m,
                "range_bin_spacing_m": range_resolution_m,
                "max_unambiguous_range_m": max_unambiguous_range_m,
                "adc_capture_duration_us": ADC_SAMPLES / SAMPLE_RATE_HZ * 1e6,
            },
            "assumptions_requiring_alignment": [
                "77 GHz carrier because exact HuPR start/centre frequency is unpublished",
                "3 cm radar-centre baseline; paper only says the radars are kept as near as 3 cm",
                "vertical-board roll sign and translation must be confirmed from scene/radar alignment",
            ],
        },
    }


def config_centre_m(config: dict) -> np.ndarray:
    """The board centre a config was built around, in the world frame (metres).

    HuPR's configs record it under ``hupr_provenance``; the cascade's under
    ``provenance``. Either way it is the point the antenna positions were
    placed around, so a config without one answers with the array's mean.
    """
    for key in ("hupr_provenance", "provenance"):
        block = config.get(key)
        if isinstance(block, dict) and "centre_m" in block:
            return np.asarray(block["centre_m"], dtype=float)
    positions = [row["pos_mm"] for row in config["tx_array"] + config["rx_array"]]
    return np.asarray(positions, dtype=float).mean(axis=0) / 1000.0


def array_axis(config: dict) -> np.ndarray:
    """Unit direction of the four-element physical receiver ULA."""
    positions = np.asarray([row["pos_mm"] for row in config["rx_array"]])
    direction = positions[-1] - positions[0]
    return direction / np.linalg.norm(direction)


def validate_pair(horizontal: dict, vertical: dict) -> dict:
    """Validate Table-1 constants, hardware layout, and relative rig geometry."""
    errors: list[str] = []
    warnings: list[str] = []

    expected_scalars = {
        "sampleRate": SAMPLE_RATE_HZ,
        "freqSlope": FREQUENCY_SLOPE_HZ_S,
        "numAdcSamples": ADC_SAMPLES,
        "rampEndTime": CHIRP_DURATION_S,
    }
    for name, config in (("horizontal", horizontal), ("vertical", vertical)):
        for key, expected in expected_scalars.items():
            if not math.isclose(float(config.get(key, math.nan)), float(expected), rel_tol=1e-9):
                errors.append(f"{name}.{key}: expected {expected}, got {config.get(key)}")
        if len(config.get("tx_array", [])) != NUM_TX:
            errors.append(f"{name}: renderer requires {NUM_TX} TX entries")
        if len(config.get("rx_array", [])) != NUM_RX:
            errors.append(f"{name}: renderer requires {NUM_RX} RX entries")

    h_axis, v_axis = array_axis(horizontal), array_axis(vertical)
    axis_dot = float(np.dot(h_axis, v_axis))
    if abs(axis_dot) > 1e-6:
        errors.append(f"receiver array axes are not orthogonal (dot={axis_dot:.6g})")

    h_c = np.asarray(horizontal["hupr_provenance"]["centre_m"], dtype=float)
    v_c = np.asarray(vertical["hupr_provenance"]["centre_m"], dtype=float)
    baseline_m = float(np.linalg.norm(v_c - h_c))
    if not math.isclose(baseline_m, 0.03, abs_tol=1e-6):
        warnings.append(f"radar centre baseline is {baseline_m:.4f} m, not initial 0.0300 m")

    warnings.append(
        "HuPR Table 1 prints 4 TX / 3 RX; IWR1843 hardware and released code use 3 TX / 4 RX"
    )
    range_resolution = (
        C_LIGHT_M_S * SAMPLE_RATE_HZ
        / (2.0 * FREQUENCY_SLOPE_HZ_S * ADC_SAMPLES)
    )
    if not math.isclose(range_resolution, 0.048, abs_tol=0.001):
        warnings.append(
            f"Table parameters imply {range_resolution:.5f} m range-bin spacing, "
            "not the paper's approximate 0.048 m"
        )

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "geometry": {
            "horizontal_array_axis": h_axis.tolist(),
            "vertical_array_axis": v_axis.tolist(),
            "axis_dot": axis_dot,
            "centre_baseline_m": baseline_m,
        },
        "signal": {
            "range_bin_spacing_m": range_resolution,
            "max_unambiguous_range_m": (
                C_LIGHT_M_S * SAMPLE_RATE_HZ / (2.0 * FREQUENCY_SLOPE_HZ_S)
            ),
            "tdm_loops_per_frame": TDM_LOOPS_PER_FRAME,
            "physical_chirps_per_frame": PHYSICAL_CHIRPS_PER_FRAME,
        },
    }


def build_radar_config(radar: Radar, centre_m: Iterable[float] = (0.0, 0.0, 0.0),
                       adc_start_s: float = 0.0, ramp_end_s: float | None = None,
                       provenance: dict | None = None) -> dict:
    """One mm3DPS-compatible config from a :class:`Radar`'s own antenna table.

    Positions are the table's half-wavelengths of ``radar.design_hz`` after
    the board rotation, centred at the layout's bounding-box centre and laid
    on the world frame: azimuth on +X, elevation on +Z, boresight +Y. This
    is the cascade's builder; HuPR's pair keeps :func:`build_config`, whose
    files are committed and must not move.
    """
    centre = np.asarray(tuple(centre_m), dtype=float)
    rx, tx = radar.rotated_positions()
    half_wavelength_m = C_LIGHT_M_S / radar.design_hz / 2.0
    allp = np.vstack((rx, tx))
    mid = (allp.min(axis=0) + allp.max(axis=0)) / 2.0
    def world(points):
        rel = (points - mid) * half_wavelength_m
        return centre + np.stack([rel[:, 0], np.zeros(len(rel)), rel[:, 1]], axis=1)
    boresight = np.array([0.0, 1.0, 0.0])
    return {
        "modulation_scheme": "FMCW",
        "carrierFrequency": radar.carrier_hz,
        "sampleRate": radar.sample_rate_hz,
        "freqSlope": radar.slope_hz_s,
        "numAdcSamples": radar.samples,
        "adcStartTime": adc_start_s,
        "rampEndTime": radar.chirp_duration_s if ramp_end_s is None else ramp_end_s,
        "tx_array": _entries("TX", world(tx), boresight),
        "rx_array": _entries("RX", world(rx), boresight),
        "provenance": {
            "radar_name": radar.name,
            "coordinate_frame": "+Y boresight, +X azimuth (image right), +Z elevation (up)",
            "centre_m": centre.tolist(),
            "board_rotation_deg": radar.board_rotation_deg,
            "design_frequency_hz": radar.design_hz,
            "half_wavelength_m": half_wavelength_m,
            "tx_order": list(radar.tx_order),
            "azimuth_tx": list(radar.azimuth_tx),
            "azimuth_grid_slots": int(radar.azimuth_grid()[1]),
            "angle_fft_bins": radar.n_angle,
            **(provenance or {}),
        },
    }


CASCADE_PROVENANCE = {
    "sensor": "TI MMWCAS-RF-EVM, 12 TX x 16 RX, TDA2 capture",
    "dataset": "RT-Pose (arXiv 2407.13930)",
    "chirp_source": "RT-POSE repository data_processing/mmWave-Matlab/main/cascade/input/hardware_param.m",
    "antenna_source": "the same file's TI_Cascade_{TX,RX}_position_{azi,ele} and TI_Cascade_RX_ID; "
                      "identical to ColoRadar's antenna_cfg.txt (mmIR/3DPS)",
    "rx_channel_order": "as recorded: master 0-3, slave1 4-7, slave2 8-11, slave3 12-15",
    "tdm_order": "TxToEnable = [12 .. 1]: slot k of a loop carries TX 12-k",
    "board_rotation": "180 degrees about boresight relative to ColoRadar's rig (paper Fig. 1); "
                      "negates both antenna coordinates, as 3DPS's flip_upside_down does; "
                      "verified from data by mm4DPS.qa orientation-check",
    "chirp": {"start_frequency_hz": 77.0e9, "adc_start_s": 5e-6, "ramp_end_s": 60e-6,
              "idle_s": 5e-6, "interval_s": 65e-6, "sample_rate_hz": 5e6,
              "slope_hz_s": 64.985e12, "samples": 256, "loops": 64, "frame_period_s": 0.1},
    "calibration": "mmwcas_calibration.npz (TI calibrateResults_high.mat): per-(TX, RX) range-offset and phase "
                   "correction applied to the measured ADC before any product "
                   "(TI calibrationCascade, phaseCalibOnly=1)",
    "assumptions_requiring_alignment": [
        "carrierFrequency is the 77 GHz start frequency, the 3DPS/ColoRadar convention",
        "the radar extrinsic is not shipped with RT-Pose; it is estimated per sequence",
        "the 180 degree board rotation is verified from data on the first sequence",
    ],
}


def write_cascade(output_dir: Path) -> Path:
    """``configs/mmwcas_cascade.json`` from the RT-Pose radar description."""
    radar = RTPOSE.radars[0]
    config = build_radar_config(radar, adc_start_s=5e-6, ramp_end_s=60e-6,
                                provenance=CASCADE_PROVENANCE)
    report = validate_configs(RTPOSE, {radar.name: config})
    if not report["valid"]:
        raise ValueError("generated invalid cascade config: " + "; ".join(report["errors"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / radar.config
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


def validate_configs(dataset: "Dataset", configs: dict[str, dict]) -> dict:
    """Validate a dataset's configs: HuPR's pair as before, any other per radar.

    For a dataset that is not HuPR the check is that each config's chirp
    scalars and array sizes agree with its :class:`Radar`; the pair-geometry
    checks (orthogonal arrays, the 3 cm baseline) are HuPR's.
    """
    if dataset.name == "hupr":
        return validate_pair(configs["horizontal"], configs["vertical"])
    errors: list[str] = []
    for radar in dataset.radars:
        config = configs[radar.name]
        expected = {"sampleRate": radar.sample_rate_hz, "freqSlope": radar.slope_hz_s,
                    "numAdcSamples": radar.samples}
        for key, value in expected.items():
            if not math.isclose(float(config.get(key, math.nan)), float(value), rel_tol=1e-9):
                errors.append(f"{radar.name}.{key}: expected {value}, got {config.get(key)}")
        if len(config.get("tx_array", [])) != radar.n_tx:
            errors.append(f"{radar.name}: {radar.n_tx} TX entries expected")
        if len(config.get("rx_array", [])) != radar.n_rx:
            errors.append(f"{radar.name}: {radar.n_rx} RX entries expected")
    return {"valid": not errors, "errors": errors, "warnings": [],
            "geometry": {f"{r.name}_array_axis": array_axis(configs[r.name]).tolist()
                         for r in dataset.radars},
            "signal": {"range_bin_spacing_m": dataset.radars[0].range_bin_m,
                       "tdm_loops_per_frame": dataset.radars[0].loops,
                       "physical_chirps_per_frame": dataset.radars[0].chirps}}


def write_pair(
    output_dir: Path,
    baseline_m: float = 0.03,
    vertical_roll_deg: float = 90.0,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    horizontal = build_config("horizontal", (-baseline_m / 2.0, 0.0, 0.0), 0.0)
    vertical = build_config("vertical", (baseline_m / 2.0, 0.0, 0.0), vertical_roll_deg)
    h_path = output_dir / "iwr1843_horizontal.json"
    v_path = output_dir / "iwr1843_vertical.json"
    report_path = output_dir / "validation_report.json"
    h_path.write_text(json.dumps(horizontal, indent=2) + "\n")
    v_path.write_text(json.dumps(vertical, indent=2) + "\n")
    report = validate_pair(horizontal, vertical)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if not report["valid"]:
        raise ValueError("generated invalid configs: " + "; ".join(report["errors"]))
    return h_path, v_path, report_path
