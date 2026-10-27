"""GVHMR inside the chain: the RGB frames -> the SMPL vertices per frame, in the camera frame.

The person's mesh comes from GVHMR (a monocular mesh-recovery model in its
own checkout and conda environment, `GVHMR_ROOT` / `GVHMR_PYTHON`). The
chain used to read its output from `seeds/gvhmr_vertices/`, made by hand;
now the person stage produces it for any sequence that has none:

`mm4DPS/gvhmr_fast_dump.py` (this repo's; run with cwd = GVHMR_ROOT) runs in the GVHMR environment
on the sequence's GPU, from the frames in memory -- no video is written or
decoded: YOLO tracking, ViTPose and the ViT features (the two ViTs in half
precision, crops at the demo's half resolution), GVHMR in static-camera
mode, the SMPL vertices (`vertices_camera (frames, 6890, 3)` float32 npz).
~30 s a sequence, ~9 s of it the models loading, which the chain hides by
starting GVHMR together with the reference stage (`start` / `finish`).

    python -m mm4DPS.gvhmr vertices --dataset rtpose --sequence 183 --output seq183.npz
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from mm4DPS import Dataset, dispatch

GVHMR_ROOT = Path(os.environ.get("GVHMR_ROOT", "/home/adnan/Desktop/GVHMR"))
GVHMR_PYTHON = Path(os.environ.get("GVHMR_PYTHON", "/home/adnan/.conda/envs/gvhmr/bin/python"))
# RT-Pose's left camera: 1067.29 px on the 1440 x 1080 centre crop -> a
# 25.65 mm full-frame-equivalent focal length, which GVHMR's static-camera
# mode takes as `f_mm` (docs/RTPOSE.md). HuPR publishes no intrinsics and
# GVHMR estimates its own from the image size.
RTPOSE_F_MM = 25.6545


def gvhmr_command(dataset, sequence: int, output: Path) -> list[str]:
    """The GVHMR invocation for a sequence: `tools/fast_dump.py` in the GVHMR environment."""
    tool = Path(__file__).resolve().with_name("gvhmr_fast_dump.py")   # ours; runs with cwd = GVHMR_ROOT
    for needed in (tool, GVHMR_PYTHON):
        if not needed.exists():
            raise FileNotFoundError(f"GVHMR is not where the chain expects it: {needed} "
                                    "(GVHMR_ROOT / GVHMR_PYTHON)")
    command = [str(GVHMR_PYTHON), str(tool), "--out", str(output)]
    if dataset.name == "hupr":
        from mm4DPS import Install
        command += ["--hupr-zip", str(Install.from_env().dataset_root / "frames.zip"), "--sequence", str(sequence)]
    elif dataset.name == "rtpose":
        from mm4DPS.data import RTPoseSequence
        seq = RTPoseSequence(dataset, sequence)
        command += ["--frames", str(seq.dir / "camera" / "left"), "--f_mm", str(RTPOSE_F_MM)]
    else:
        raise NotImplementedError(dataset.name)
    return command


def start(dataset, sequence: int, output: Path, log: Path) -> subprocess.Popen:
    """Launch GVHMR for a sequence in the background (the chain starts it with the reference stage)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)                 # the GVHMR tree's own imports, not this checkout's
    handle = log.open("w")
    process = subprocess.Popen(gvhmr_command(dataset, sequence, output), cwd=GVHMR_ROOT, env=env,
                               stdout=handle, stderr=subprocess.STDOUT)
    process._mm4dps_log = handle               # closed by `finish`
    return process


def finish(process: subprocess.Popen, output: Path, frames: int | None = None) -> np.ndarray:
    """Wait for a `start`ed GVHMR and validate its output."""
    process.wait()
    process._mm4dps_log.close()
    if process.returncode != 0 or not output.is_file():
        raise RuntimeError(f"GVHMR failed writing {output} (exit {process.returncode}); see its log beside it")
    vertices = np.load(output)["vertices_camera"]
    if vertices.ndim != 3 or vertices.shape[1:] != (6890, 3) or (frames is not None and vertices.shape[0] != frames):
        raise RuntimeError(f"GVHMR wrote {vertices.shape}, expected ({frames or '*'}, 6890, 3)")
    return vertices


def run_vertices(args: argparse.Namespace) -> dict:
    dataset = Dataset.select(args.dataset)
    started = time.time()
    log = args.output.with_suffix(".log")
    vertices = finish(start(dataset, args.sequence, args.output, log), args.output)
    result = {"status": "completed", "dataset": dataset.name, "sequence": int(args.sequence),
              "frames": int(vertices.shape[0]), "output": str(args.output),
              "seconds": round(time.time() - started, 1), "log": str(log)}
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def make_parser_vertices() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=run_vertices.__doc__)
    parser.add_argument("--dataset", required=True, choices=("hupr", "rtpose"))
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="the vertices npz to write")
    return parser


def _entry_vertices():
    run_vertices(make_parser_vertices().parse_args())


COMMANDS = {"vertices": _entry_vertices}


def main() -> int:
    return dispatch(COMMANDS, __name__)


if __name__ == "__main__":
    raise SystemExit(main())
