# 3DPS: 3D Point Splatting for mmWave Radar Novel View Synthesis

> Code release for the NeurIPS 2026 paper. The PDF and supplementary
> material accompany this archive.

## Abstract

3DPS introduces a differentiable, point-splatting renderer for the
77 GHz cascaded-radar Range–Azimuth (RA) imaging pipeline. Each scene
is represented as a per-point set of physically-grounded ITU material
parameters splatted onto a hemisphere; closed-form path integration
through a per-frame chirp-window FFT cascade yields range-profile
phasors at ~3 minutes per scene on a single RTX 4090. We evaluate on a
six-scene ColoRadar benchmark with a strict middle-frame-bracketed
8-train / 1-test held-out NVS protocol, and compare against three
recent radar-NVS baselines (DART, RadarFields, RadarSplat) re-trained
under an identical protocol. 3DPS attains the highest mean test-frame
RA Cartesian correlation (`cart_corr`) on the benchmark while training
1–2 orders of magnitude faster.

---

## Requirements

- Python 3.12 (tested), CUDA 11.8 (tested, RTX 4090).
- PyTorch 2.7.1 (cu118), Mitsuba 3.7.1, DrJit 1.2.0, NumPy 1.26, SciPy
  1.16, scikit-image 0.25, matplotlib 3.10, trimesh 4.7, Open3D 0.19,
  plyfile 1.1, h5py 3.15, tqdm 4.67. See [requirements.txt](requirements.txt) /
  [environment.yml](environment.yml) for pinned versions.

```bash
conda env create -f environment.yml -n mm3DPS
conda activate mm3DPS
# Build the optional CUDA extension (used by the factorized rasterizer):
cd mm3DPS/cuda && python setup.py build_ext --inplace && cd ../..
```

The CUDA extension is **optional** — the PyTorch rasterizer fallback
runs everywhere; building the extension just speeds the BSDF kernel up.

The three baselines (DART, RadarFields, RadarSplat) each require their
own conda env with the upstream package installed. The path to that
upstream commit is pinned in `baselines/<name>/UPSTREAM_COMMIT.txt`,
and any local modifications are recorded as `.patch` files under
`baselines/<name>/patches/`.

---

## Data

The release ships **no raw data**. See [data/README.md](data/README.md)
for the full preprocessing pipeline. In short:

1. Download ColoRadar (sequences `2_28_2021_outdoors_run0`, `run1`,
   `run2`) and slot it under `data/`.
2. Run `python -m mm3DPS.preprocessing.alignment.run_alignment --scene <scene>`
   followed by `python -m mm3DPS.preprocessing.alignment.pass2.run_pass2 --scene <scene>`.
3. The result is
   `data/alignment_data/<scene>/cascade/cascaded_frame_<F>_aligned_pass2.json`,
   the canonical pass-2 alignment used by every training and evaluation
   entry point in this release.

> All training and evaluation use
> `cascaded_frame_<F>_aligned_pass2.json` by default. The opt-out flag
> `--no_pass2` falls back to pass-1 (`cascaded_frame_<F>_aligned.json`).

### The six paper scenes

```
seq_0_frame_135   seq_1_frame_185   seq_1_frame_438
seq_2_frame_105   seq_2_frame_160   seq_2_frame_300
```

Each ships with a 9-frame cascaded sequence under
`data/<scene>/radar/cascaded_frame_*.npy` plus matching pass-2 aligned
configs under `data/alignment_data/<scene>/cascade/`. The held-out test
frame is the middle frame of each 9-frame sequence (e.g. 135 for
`seq_0_frame_135`); the eight training frames are
`{F-4, F-3, F-2, F-1, F+1, F+2, F+3, F+4}`.

---

## Training

Single scene (~3 min on RTX 4090, 32 GB system RAM, ~3 GB GPU memory):

```bash
python -m mm3DPS.train_frame_nvs \
       --scene seq_0_frame_135 \
       --target_n 20000 \
       --iters 500
```

(The thin wrapper `python train.py --scene seq_0_frame_135 ...` is
equivalent.)

All six scenes:

```bash
for s in seq_0_frame_135 seq_1_frame_185 seq_1_frame_438 \
         seq_2_frame_105 seq_2_frame_160 seq_2_frame_300; do
  python -m mm3DPS.train_frame_nvs --scene "$s" --target_n 20000 --iters 500
done
```

Each run writes a per-scene output dir (default
`output/training/<scene>/`) containing `best_model.pt`, `results.json`,
and intermediate artefacts. The CLI exposes the full set of
hyperparameters (LR sweeps, ablation flags, etc.); pass `--help` for
the complete list.

---

## Evaluation

### RA novel-view metrics

Per scene:

```bash
python -m mm3DPS.frame_nvs_report --output-dir output/training/
```

This walks every run directory, parses each `results.json`, and emits
a markdown table whose `cart_corr` column is the paper's primary
test-frame RA correlation.

### CRP/ADC product-agnostic evaluation

`mm3DPS.evaluation.eval_crp_adc` (called by
`mm3DPS.rerender_test_rp`) computes per-scene phase-corrected complex
range profiles and ADC tensors. The CRP/ADC eval is round-trip
bit-exact at machine precision (per the paper's CRP/ADC ablation):

```bash
python -m mm3DPS.rerender_test_rp \
       --scene seq_0_frame_135 \
       --checkpoint checkpoints/seq_0_frame_135/best_model.pt
```

This writes phase-corrected `.npy` artefacts under
`output/crp_adc_eval/<scene>/`, which the figure scripts in
`figures/generate_fig_crp_adc.py` read back.

---

## Pre-trained checkpoints

The release ships pre-trained models for **3 of the 4 evaluated
methods** at six paper scenes each, totalling 18 checkpoint sets. See
[checkpoints/README.md](checkpoints/README.md) for the full layout, the
inference recipe per baseline, and the schema of every `metadata.json`.

| Method        | Per-scene file                         | × 6     | Shipped |
|---------------|----------------------------------------|---------|---------|
| 3DPS (ours)   | `checkpoints/<scene>/best_model.pt`    | ~6 MB   | ✅      |
| DART          | `checkpoints/baselines/dart/<scene>/model.{npz,json}` | ~37 MB  | ✅      |
| RadarSplat    | `checkpoints/baselines/radarsplat/<scene>/{ckpt.pt,cfg.yml}` | ~25 MB  | ✅      |
| RadarFields   | n/a                                    | ~720 MB | ❌ (re-train; ~5 min/scene on RTX 4090, see below) |

RadarFields checkpoints are not shipped because each `.pth` is ~120 MB
(of which only ~42 MB are model weights; the rest is optimizer +
loss-history state for resume). Re-train with:

```bash
python -m baselines.radarfields.runner.run_phase4 --gpus 0 1
```

The 3DPS checkpoints document training hyperparameters + the exact
reproduction command:

```json
{
  "scene": "seq_0_frame_135",
  "test_frame": 135,
  "train_frames": [131, 132, 133, 134, 136, 137, 138, 139],
  "alignment_source": "pass-2",
  "target_n": 20000,
  "num_iters": 500,
  "test_cart_corr": 0.6442,
  "reproduction_command": "python -m mm3DPS.train_frame_nvs --scene seq_0_frame_135 --target_n 20000 --iters 500"
}
```

Load + render a checkpoint:

```python
from mm3DPS.load_pretrained import load_trained_config
import torch

config = load_trained_config("seq_0_frame_135")
state = torch.load("checkpoints/seq_0_frame_135/best_model.pt")
# state contains: positions (N,3), rotations (N,4 quaternion),
# raw_materials (N,6 raw pre-reparameterization).
```

The state dict is consumed by `mm3DPS.train_gaussian.train_gaussians`
when `--init_checkpoint` points at the file.

---

## Results

Reported in the paper (Tables 1 and 2). The numbers below are produced
end-to-end by `figures/generate_tables.py` from per-scene `metrics.json`
files written by each baseline's `runner/run_phase4.py`.

> Test-frame RA Cartesian correlation (`cart_corr`), held-out middle
> frame, six paper scenes. Higher is better.

| Method        | Mean | Wall time |
|---------------|------|-----------|
| DART          | (Table 1) | ~2 min/scene  |
| RadarFields   | (Table 1) | ~5 min/scene  |
| RadarSplat    | (Table 1) | ~12 min/scene |
| **3DPS**      | (Table 1) | **~3 min/scene** |

Refer to the paper for the full per-scene breakdown plus the CRP/ADC
product-agnostic evaluation (Table 2).

---

## Baselines

See [baselines/README.md](baselines/README.md) for the per-baseline
operational guide. Briefly:

- **DART** — `python -m baselines.dart.runner.run_phase4 --gpus 0 1`
- **RadarFields** — `python -m baselines.radarfields.runner.run_phase4 --gpus 0 1`
- **RadarSplat** — `python -m baselines.radarsplat.runner.run_phase4 --gpus 0 1`

Each Phase-4 driver shell-spawns the actual training under each
baseline's own conda env, then runs `finalize_metrics.py` (under the
mm3DPS env) to apply the canonical 3DPS metric harness so all four
methods produce comparable numbers.

---

## Figures

See [figures/README.md](figures/README.md) for the script-by-script
figure / table mapping. Inputs:

- The six checkpoints under `checkpoints/`.
- Per-baseline `metrics.json` produced by each baseline's `run_phase4.py`.
- The CRP/ADC eval artefacts under `output/crp_adc_eval/<scene>/`.

---

## Repository structure

```
README.md
LICENSE                            # PolyForm Noncommercial 1.0.0
requirements.txt
environment.yml
train.py                           # thin wrapper around mm3DPS.train_frame_nvs

mm3DPS/                            # canonical 3DPS package
  __init__.py
  config.py                        # TrainingConfigSionna + load_config_from_json
  load_pretrained.py               # per-scene config + (factory) pattern loader
  rasterizer.py                    # core rasterizer
  rasterizer_factorized.py         # MIMO-factored rasterizer (paper default)
  rasterizer_nonfactored.py        # PyTorch fallback
  bsdf_torch.py                    # ITU-material BSDF + Fresnel + reparam
  psf.py                           # Hann PSF table
  train_frame_nvs.py               # canonical paper training entry point
  train_gaussian.py                # core training loop / Gaussian field utilities
  _pose_utils.py                   # per-loop pose interpolation helpers
  rerender_test_rp.py              # CRP/ADC complex re-render evaluator
  frame_nvs_report.py              # NVS metrics aggregation
  material_diagnostics.py          # diagnostics + reparam helpers
  cuda/                            # optional CUDA extension (sources + setup.py)
  data/                            # ADC↔RA, range_res helpers (vendored)
  evaluation/utils/                # metrics + viz helpers (vendored)
  losses/                          # loss helpers (vendored)
  preprocessing/                   # ColoRadar -> data/ pipeline + alignment
    preproc.py                     #   top-level driver (adc / configs / lidar / mesh)
    ColoRadar_tools/               #   raw ColoRadar parsers
    calib/                         #   sensor calibration files (cascade + transforms)
    alignment/                     #   cascade alignment (pass-1 + pass-2)

baselines/
  common/                          # shared scene list, NVS split, eval harness
  dart/                            # WiseLabCMU/dart adapter + runner + patches
  radarfields/                     # princeton-computational-imaging/RadarFields
  radarsplat/                      # umautobots/radarsplat
  README.md

configs/
  <scene>/config.json              # per-scene paper-settings JSON

assets/
  antenna_pattern/MMWCAS/          # tx1_76.npy + rx1_76.npy

data/
  README.md                        # ColoRadar fetch + preprocessing pipeline

checkpoints/
  README.md                        # checkpoint layout + per-baseline inference recipes
  <scene>/                         # 3DPS (ours)
    best_model.pt                  #   ~1 MB; positions, rotations, raw_materials
    metadata.json
  baselines/
    dart/<scene>/                  # DART
      model.{npz,json}
      metadata.json
    radarsplat/<scene>/            # RadarSplat
      ckpt.pt
      cfg.yml
      metadata.json
  # RadarFields checkpoints not shipped; re-train via run_phase4.py

figures/                           # paper-figure & table generators
  README.md
```

---

## License

Code in this release is distributed under [PolyForm Noncommercial
1.0.0](LICENSE). It is licensed for **non-commercial research use
only**. Attribution (cite the paper) is required. All rights reserved
by the authors. Third-party LICENSE / AUTHORS files (PoissonRecon, etc.)
are preserved unmodified where they appear under `third_party/`.

## Citation

```bibtex
@inproceedings{anonymous2026threedps,
  title     = {{3DPS}: {3D} Point Splatting for {mmWave} Radar Novel View Synthesis},
  author    = {Anonymous Authors},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  note      = {Anonymous submission}
}
```
