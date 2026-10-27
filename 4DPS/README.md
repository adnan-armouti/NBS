# 4DPS — differentiable radar rendering of a moving person

**4DPS** renders synthetic 77 GHz radar from a camera reconstruction of a moving
person and compares it against the measured recording on range-angle,
range-Doppler and micro-Doppler. It fits a per-joint velocity field, skinned to
the person's points, so that the rendered micro-Doppler matches the measured
one. The package is `mm4DPS/`, 19 modules.

**What the ablation supports.** A monocular camera reconstruction alone already
gets the *bulk* radial motion right — unfitted, micro-Doppler centroid
correlation is +0.95 to +0.97. What it does not get is the *limb* signature:
unfitted spread and energy correlations sit at or below zero. Fitting the
velocity field against the measured radar raises those to +0.71 to +0.85 and
+0.86 to +0.94. The camera supplies the bulk Doppler; the velocity field
supplies the micro-Doppler structure. Numbers and definitions:
[docs/METRICS.md](docs/METRICS.md).

**Built on 3DPS.** The differentiable renderer — the Gaussian splat primitives,
the Jones BSDF, the factorized rasterizer and its CUDA kernels — is the 3DPS
authors' code, adapted, in `mm4DPS/rasterizer.py` and `mm4DPS/splat/`. The
attribution, what was kept and what was removed are in
[mm4DPS/splat/PROVENANCE.md](mm4DPS/splat/PROVENANCE.md). 4DPS adds the person,
the motion, the second radar and the training against measured data.

## 1. Setup

Python 3.12 with PyTorch 2.7 + CUDA. From the repository root:

```bash
export HUPR_DATASET_ROOT=/path/to/HuPR          # see Dataset
make build-cuda                                 # both CUDA extensions, stamped
python -m mm4DPS.qa check-environment           # every path resolved, both stamps current
```

`check-environment` is not optional: a stale CUDA extension presents as a
pybind signature error naming a function, not as a build error, and the build
stamp is what catches it — it compares the sources' content, not their mtime.

**Two CUDA extensions, both required.** `hupr_cuda` (`mm4DPS/cuda/`) holds
4DPS's kernels; `mm3DPS_cuda` (`mm4DPS/splat/cuda/`) holds 3DPS's BSDF and
scatter splat. `make build-cuda` builds both; `TORCH_CUDA_ARCH_LIST` in the
Makefile is the target GPU (8.9 for an RTX 4090). A tree with only the first
passes every import and dies four stages into a chain.

### Dataset

HuPR is not ours to redistribute; its repacked radar archives are 0.88 GB
each. The code looks for it at `$HUPR_DATASET_ROOT`, then `<repo>/data/HuPR`:

```
<dataset root>/
  frames.zip                     frames/single_<seq>/processed/images/<frame:09d>.jpg
  radar/radar_seq_<lo>-<hi>.zip  single_<seq>/{hori,vert}/adc_data.bin
  annotations/hrnet_annot_{train,val,test}.json
```

`find_archive` matches the `seq_<lo>-<hi>` range in the filename, so a
one-sequence-per-archive repack resolves identically.

RT-Pose is read from `$RTPOSE_DATASET_ROOT` (default `/data/datasets/RT-Pose`):
`sequences/<n>/{radar/bin,lidar,camera/left}`, `Train.json`, `calib/camera`.
Its radar extrinsic seed lives under `seeds/rtpose/`.

### Models and seeds — provided, not committed

| what | size | where |
|---|---|---|
| Depth-Anything-V2-Metric-Indoor-Large, revision `d2fc6a9` | 1.3 GB | `$HF_HOME` — `snapshot_download("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf", revision="d2fc6a93601aabb1139a3bf0ebfcb4e89c67817f")` |
| SMPL neutral model | 16 MB | `models/smpl/SMPL_NEUTRAL.npz` |
| GVHMR checkout + environment (`/home/adnan/Desktop/GVHMR`, conda env `gvhmr`) | — | the person mesh model the chain runs per sequence |

`seeds/` holds the one input a run reads that is not the dataset: the
RT-Pose radar extrinsic (`rtpose/radar_extrinsic.json`, the identity). The
person's mesh comes from GVHMR run inside the chain (`mm4DPS.gvhmr`, in
GVHMR's own checkout and conda environment, `GVHMR_ROOT` / `GVHMR_PYTHON`:
about a minute a sequence on the run's GPU). **Nothing else is seeded**:
the scene scaffold and its material model, the antenna coupling, the
person's placement (registered to the radar frame by frame as its sequence
is built), the joint scattering gains, the receiver gain and the noise
floor are all computed from the recording in the chain or learned by the
trainer; the monocular scene's range alignment is fitted from the
recording too. `docs/SEEDS.md` is the inventory.

The SMPL weights came from a third-party re-upload, not the official
registered download; settle that provenance before publishing anything derived
from them.

## 2. Running one sequence end to end

```bash
python -m mm4DPS.chain run <run> <seq>         # e.g. run mine 102; ~11 min on one RTX 4090
```

writes `outputs/<run>/seq<seq>/`:

```
checkpoint.pt            the trained velocity field, the person material, and `render_inputs`: the scene and
                         person geometry as the renderer receives it (positions, normals, materials,
                         visibility, calibrations, the frame selections and the exact commands --
                         enough to forward render again: `chain rerender checkpoint.pt --output-dir DIR`)
history.json             per-epoch training losses, both radars, and the final validation pass
metrics.json             every metric in the paper, the micro-Doppler score, and render_config
horizontal_rendered.npy  the slow-time cube, (600, 64, 3, 4, 256, 2) float16, 470 MB
horizontal_rendered_scale.npy  one float32 per frame: the factor the frame was divided by
vertical_rendered.npy
vertical_rendered_scale.npy
run.log
```

The products are stored as float16 (real, imag) pairs at each frame's own
scale -- four bytes a complex sample, the GT ADC's int16 I/Q footprint;
`mm4DPS.data.ProductCube(path)[frame]` returns the frame as complex64,
every cell within 4.9e-4 of the renderer's value. No bin and no channel is
dropped: axes are
`(frame, loop, tx, rx, range)` — the per-loop range profiles in slow time,
range being the Hann-windowed FFT of the 256 ADC samples (the rendered
side's range PSF is that window's response). Every view the pipeline
computes (range-angle, range-Doppler, the spectrograms) is a windowed
forward FFT and a magnitude away, from the one definition:
`signal_model.fft_window` (periodic Hann over range and Doppler, symmetric
Hann without end points over the azimuth grid), applied identically to the
measured and the rendered side — `data.product_to_doppler` takes the
Doppler transform (mean removal, window, FFT, fftshift). HuPR's released
preprocessing is unwindowed; `MM4DPS_FFT_WINDOW=none` restores it for an
A/B and is not a production setting.

```bash
python -m mm4DPS.chain run <run> <seq> --intermediates   # also keep .work/: every working product, the
                                                          #   no-noise render, videos, per-step history, stage logs
python -m mm4DPS.chain run <run> <seq> --artifacts DIR   # also build the two comparison-media folders into DIR
python -m mm4DPS.chain run <run> <seq> --mask-off        # the control arm: no per-frame scene visibility
```

```bash
python -m mm4DPS.chain run <run> 183 --dataset rtpose            # RT-Pose: one cascade radar, docs/RTPOSE.md
```

`--dataset {hupr,rtpose}` (default `hupr`, or `$MM4DPS_DATASET`) selects
the radars, the scene source and the layout; every stage reads the same
`Dataset` description (`mm4DPS/__init__.py`). RT-Pose's outputs are the
same files with one radar, `cascade_{rendered,measured}.npy` at
`(299, 64, 12, 16, 256)`, 3.8 GB each.

The per-frame scene-visibility mask is on by default. `--artifacts` and
`--intermediates` are independent. The chain's three working workspaces and
every intermediate live under `outputs/<run>/seq<seq>/.work/` during the run
and are removed at the end unless `--intermediates`; there is no cache
anywhere — the measured ADC decodes from the archive in 2 s per radar.
`outputs/` is entirely regenerable from `seeds/` and the dataset.

What a run keeps: the rendered cube (`<radar>_rendered.npy`, `(frames, 64
loops, tx, rx, 256, 2)` float16 with its per-frame scale: 3.8 GB for a
cascade sequence, 0.47 GB per HuPR board -- the GT's own size) and three
small files. The measured cube is not stored -- it is
the recording alone, identical run to run -- the one consumer that read
it (the spectrogram stage) decodes it from the archive frame by frame
(`media.MeasuredCube`; `--save-measured-cube on` writes it for a consumer
outside this repository). `--intermediates` keeps `.work/` as well: the
clean render's cube, the two renders' micro-Doppler cubes
(`*_motion_dra.npy`, 2.1 GB each on the cascade) and the trainer's target
cache -- about 19 GB on a cascade sequence against 7.5 without.

The trainer's production flags (`--epochs 400 --training-frames 600
--train-all-frames --eval-every 5 --lr-schedule cosine --plane-target-dtype
float16 --doppler-support-snr 1.0`) are the chain's defaults; `VF_EXTRA_APPEND`
adds to them and `VF_EXTRA` replaces them, for ablations. 400 epochs with a
cosine schedule is uniform across sequences on purpose — a per-sequence epoch
count is not a publishable configuration.

The trainer renders each radar's **training sub-array**: the TX that form
the azimuth virtual array (HuPR's TX1 and TX3; the cascade's TX4-12), on
their true TDM slots. Every loss goes through the beamformer, which drops
the elevated TX before the angle FFT, so those TX were rendered and
discarded every step; without them the losses are the same (HuPR seq 256:
the first step's loss agrees to five figures) at 28.4 against 41.5 ms a
step. `--training-array full` is the ablation. The renderer and every
product keep the full arrays. The Doppler-plane kernels skip a view's
padding planes (per-unit active plane counts), compute each path's plane
amplitudes in closed form (the Hann Doppler PSF at the path's fractional
bin, `--doppler-amplitudes closed-form`, 7e-7 of the float64 reference
against the loop sum's 5e-6; `qa doppler-check`) truncated to the 7 planes
around the path's own Doppler bin (`--doppler-taps 7`, as the range PSF is
cut at 15 bins; 7.7e-3 of the exact planes, indistinguishable from them at
400 epochs), and read the plane gradient transposed. On the captured step
the person's position-only splat geometry is cached per flat row
(`build_row_geometry`), the scene's backgrounds and the person's Doppler
planes come from kernels that synthesise every contribution from the BSDF
carrier and the cached range bins (`qa scene-check`, `qa sorted-check`;
the person's paths sorted by (unit, channel, range bin) once per frame, so
nothing is binned per step), and the BSDF's own steps run compiled
(`HUPR_COMPILED_BSDF`, `HUPR_FUSED_SCENE`, `HUPR_ROW_GEOMETRY`,
`HUPR_SORTED_DOPPLER` are the A/B switches, `--deterministic on` keeps the
fixed-order slot route). A cascade step is 15 ms against 95 at the start
of docs/TRAINING_SPEED_PLAN.md, a 400-epoch sequence ~13 minutes of chain;
`--scene-update-every k` exists but was measured worse at k=4 and stays
off. The support
itself stays HuPR's
median-over-loops floor with the band in every view --
`--doppler-support-floor energy` (only the bins carrying the person's
energy) is available but was measured to smear the person over Doppler at
400 epochs, since unrendered bins go unsupervised (docs/RTPOSE.md).

## 3. The pipeline

`python -m mm4DPS.chain run` is five stages, each a subprocess so its GPU memory
is released when it exits:

| stage | module | what |
|---|---|---|
| preprocessing | `reconstruction`, `train_motion`, `calibration` | the scene scaffold, its material model, the antenna coupling |
| person stack | `reconstruction`, `train_material`, `motion` | the person sequence from the GVHMR dump, registered to the radar per frame; the person material fitted on it (one pass); motion detection |
| scene visibility | `pointset` | the per-frame visibility mask of the background points behind the person (`scene-shadow-mask`) |
| velocity onward | `velocity_field`, `video`, `media` | the velocity field trained against the measured RA and micro-Doppler; the render; the spectrograms and score |
| evaluate | `evaluation` | every metric, to `metrics.json` |

The forward model: `scene` (the static room and the person's point cloud),
`signal_model` (chirp motion, the receiver's noise), `forward_render` (the
fused CUDA render path and the training views), `rasterizer` (3DPS's splat
renderer), `data` (the ADC archives and the SMPL model), `cuda_ext` (both
extensions, their build and their stamps). `qa` holds the two provenance
guards — a comparison across chain profiles or across products rendered with
different physics is refused, not warned about — and the orphan audit.

## 4. Conventions that decide correctness

- **Linear magnitude, never dB.** Every metric is on linear amplitude; log
  compression inflates the agreement of empty bins (the same renders score
  0.96 on a dB spectrogram and 0.79 on linear power). The one exception is
  PSNR, a log of a scalar ratio computed after the error. Details in
  [docs/METRICS.md](docs/METRICS.md).
- **Name the estimator beside any number that decides something.** The
  metrics document does; `metrics.json` records the frame selection and the
  spectrogram reduction it used.
- **World frame.** +Y is radar boresight, +X image-right, +Z up. The
  horizontal radar's array lies on +X; the vertical board is rotated +90°
  about +Y and its array lies on −Z. HuPR's Table 1 swaps the TX/RX labels;
  the hardware is 3 TX, 4 RX and the code follows the hardware.
- **The velocity field is per joint**, 14 joints × 3 in a 420-band DCT basis,
  skinned to the points — not per point. `point-normal` adds one learnable
  scalar per point per frame; `joint` is the unbounded baseline.
- **The trainer is not reproducible run to run; the renderer is** — to
  float32 epsilon on rendered products (`atomicAdd` order) and bitwise on
  measured ones. A single-seed trained number is indicative, not a result.
- **Two products whose `render_config` differ in anything but the declared
  variable are not comparable**, and `mm4DPS.media` refuses to build a
  comparison from them.

## 5. Layout

```
mm4DPS/          19 modules; `mm4DPS/__init__.py` holds Install, Workspace and the radar description
mm4DPS/cuda/     4DPS's CUDA sources and their built extension
mm4DPS/splat/    3DPS's CUDA sources, its antenna patterns, and PROVENANCE.md
configs/         the two IWR1843 configs; the MMWCAS cascade config and its TI calibration
seeds/           provided inputs (see Setup)
models/          SMPL, and the fetched Depth-Anything under models/hf/
docs/METRICS.md  every metric's definition and the conventions behind them
docs/RTPOSE.md   the RT-Pose path: the cascade, its calibration, the extrinsic, the LiDAR scene, GVHMR
Makefile         build-cuda, check-environment, audit
```

The research record — the ledger, the reproduction study, the coverage audit,
the plans behind the consolidation — lives in the source repository, not here.
