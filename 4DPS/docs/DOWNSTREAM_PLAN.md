# Plan: the downstream pose models on real and rendered radar (Tables 4 and 5)

Status: 2026-09-14, a plan only -- nothing here is implemented. The
products it consumes exist: 235 HuPR and 101 RT-Pose sequences rendered by
4DPS (`/home/adnan/Documents/Data/4DPS_products`, docs/SWEEP_RESULTS.md).

## 0. The one principle

Every downstream number in the paper comes from **one training procedure
that does not know which domain it is fed**. Real and rendered radar enter
through the same adapter from the same representation (the complex
slow-time range-profile cube, `(frames, 64 loops, TX, RX, 256 range)` --
what 4DPS renders and what the recording decodes to), take the same
split, the same seeds, the same optimizer, schedule, batch, augmentation
and model-selection rule, and are scored by the same evaluator. The only
thing that differs between two cells of a table is the bytes in the cube.
Any difference in a score is then the sim-to-real gap and nothing else --
the discipline of `/home/adnan/Desktop/mm3DGS/baselines` (one protocol,
one metrics module, per-method adapters, canonical result paths, a table
generator that no number bypasses).

## 1. Layout: `/home/adnan/Desktop/4DPS/downstream/`

```
downstream/
  README.md                     the protocol, the commands, where results land
  common/
    domain.py        Domain = {real, rendered, real+rendered}; Source: for a (dataset, sequence, domain) the
                     cube -- real: mm4DPS.data.MeasuredReader + the Hann range FFT (+ the cascade calibration);
                     rendered: mm4DPS.data.ProductCube. Same dtype, same axes, same window. One class, two backends.
    split.py         Split(dataset): train / val / test sequence lists, frozen in split.json with the rule that made them
    protocol.py      Protocol: seeds (5), epochs / steps, optimizer, lr schedule, batch, augmentation, the model-selection
                     rule; hashed; the hash is written into every result.json and checked by the table generator
    adapter.py       Adapter (abstract): cube -> the model's input tensor, per frame, on the GPU; an identity gate method
    trainer.py       Trainer (abstract): fit(protocol, split, source) -> checkpoint; evaluate(checkpoint, source, split) -> metrics
    runner.py        the matrix: for every (train domain, test domain, seed) cell -> results/<dataset>/<train>__<test>/seed<k>/
    results.py       the result schema; aggregate mean (std) over seeds; emit the LaTeX macros (\rApReal ... \rMPJPESelf)
    metrics.py       what both datasets share: retention ratio, gap decomposition, per-frame error distributions
  hupr_mscsa/                   MSCSA-PRGCN, HuPR's released baseline (third_party/hupr_official)
    adapter.py       cube -> the official per-frame heatmap npy (process_iwr1843.generateHeatmap from the range profiles)
    dataset.py       the official dataset class pointed at an adapter output directory per domain
    trainer.py       wraps third_party/hupr_official/main.py (config, --seed, --dir) under the Protocol
    evaluator.py     the official COCO-style keypoint AP (misc/cocoeval), plus the extras in section 5
  rtpose_hrradarpose/           HRRadarPose, RT-Pose's released baseline (third_party/RT-POSE)
    adapter.py       cube -> the DZYX tensor on the paper's ROI grid (64 x 16 x 64 x 160), on the GPU, per frame
    dataset.py       det3d dataset returning the tensor from the adapter on the fly (section 3.2: the tensors do not fit on disk)
    trainer.py       wraps tools/train.py with configs/cruw_pose/hr3d.py under the Protocol
    evaluator.py     MPJPE / MRPE / Abs-MPJPE (det3d.datasets.utils.eval), plus the extras
  results/
    hupr/<train>__<test>/seed<k>/{metrics.json, checkpoint.pt, log}
    rtpose/<train>__<test>/seed<k>/...
    tables/{table4.tex, table5.tex, macros.tex}   generated, never hand-edited
```

One folder per downstream task and dataset, as asked; `common/` is the
part that must be identical between them and between domains.

## 2. Step 1: reproduce each paper's real-data result (the setup gate)

Nothing rendered is scored before the real-only training reproduces the
published number to within a stated tolerance. That is what validates the
adapter, the split and the protocol.

### 2.1 HuPR -- MSCSA-PRGCN

* **Split**: the official one in `config/mscsa_prgcn.yaml`: 193 train, 21
  val, 21 test sequences (test: 15, 16, 17, 38-42, 244-256). All 235 are
  delivered, so the split is used as published.
* **Input**: the official preprocessing (`preprocessing/process_iwr1843.py`)
  builds per-frame range-azimuth-elevation heatmaps from the raw ADC: range
  FFT, the two boards' virtual arrays, angle FFTs, chirp grouping, clutter
  removal. The adapter runs **the same `generateHeatmap` code from the
  range-profile cube** (the official range FFT replaced by the cube's,
  which for the real domain is our decoder's Hann-windowed FFT). Both
  domains go through it identically. The one documented departure from
  the official pipeline is the range window (official: none; ours: Hann
  on both sides, docs/TRAINING_SPEED_PLAN.md 2.1).
* **Protocol**: the official config (200 epochs, batch 20, Adam 1e-4,
  lr decay 0.999 / 2000 iters), `--seed {0,1,2}`, `-sr 1`. Model selection
  on the val split of the training domain; the test split scored once.
* **Gate**: real -> real AP within +/- 2 AP of the figure the HuPR paper
  reports for MSCSA-PRGCN on the official test split (read from the
  paper's table when the gate is written, not from memory) and of the
  released weights, which the evaluator also scores (`\rApMeas`), and the
  Hann-window departure measured (train the real arm once with the
  unwindowed official npy as a control).
* **Cost**: 193 x 600 frames, 8-frame groups; the official README reports
  training in hours on one GPU; measure one epoch first and fix the epoch
  count in the Protocol before any rendered arm runs.

### 2.2 RT-Pose -- HRRadarPose

* **Split**: not public (docs/TWO_PERSON_AND_SEQ21.md 1.3) and we hold 106
  of 239 folders, 101 usable, 66 one-person. The paper's 9.91 cm MPJPE /
  9.91 cm MRPE is over its own test set of all 240 sequences; it cannot
  be reproduced literally. Our split: the **66 one-person** sequences
  stratified by (environment, activity, subject) into 46 / 6 / 14 train /
  val / test, frozen in `split.json` with the seed that drew it; the 35
  two-person sequences form an extra **real-only** test set reported
  separately (the annotated subject is the target, as the benchmark
  scores it; section 1.2 of TWO_PERSON_AND_SEQ21.md). The tolerance for
  "reproduced" is stated up front: MPJPE within 1.5x of the paper's on a
  quarter of its data, and the MRPE-vs-MPJPE relation of Table 2 of the
  paper preserved.
* **Input**: the paper's 4D tensor `Doppler(64) x Z x Y x X` on the ROI
  grid of `hr3d.py` (`roi1`: x 0.77-8.02 m, y +/-5.02, z -1.09-4.71 at
  0.045 / 0.157 / 0.363 m -> 160 x 64 x 16) in float16, normalised as the
  config states. The authors build it with MATLAB (calibration, Doppler
  FFT, 128-azimuth x 32-elevation FFTs) then `4Dradar2xyz.py`
  (polar -> Cartesian trilinear). The adapter re-implements those two
  steps on the GPU from the cube: Doppler FFT over the 64 loops, the
  86-slot virtual array to the 128 x 32 angle grid, the same trilinear
  polar-to-Cartesian resample on the same axes (`arrAzimuth 0-120 deg /
  128`, `arrElevation 0-60 / 32`, `arrRange 0-11.6 / 256`). Both domains
  identically; the cascade calibration is applied to the real cube where
  the MATLAB step applies it.
* **Storage**: the full tensor is 134 MB a frame (40 GB a sequence; the
  authors quote 41 TB for 240); the ROI tensor 21 MB a frame, 6.3 GB a
  sequence, 1.3 TB for 101 sequences x 2 domains -- more than the 864 GB
  free. So the dataset class computes the ROI tensor **on the fly on the
  GPU** from the cube (a Doppler FFT, two angle FFTs and a gather per
  frame: ~2 ms), and caches to disk only when `--cache-dir` is given and
  the split fits. The cube itself (3.76 GB a sequence per domain) is what
  is stored; it already is.
* **Protocol**: `hr3d.py` as released (50 epochs, batch 16, its
  optimizer and grad clip), `nms_pre/post_max_size = 1` (one person a
  frame, as released), seeds {0, 1, 2, 3, 4}.
* **Gate**: real -> real MPJPE / MRPE / Abs-MPJPE on our test split
  within the stated tolerance; the released weights, if they load,
  scored on our test split as a second reference.

## 3. Step 2: the same procedure on rendered data

For each dataset the runner produces the training arms below with
**nothing changed but the domain flag**. Rendered labels are the
annotations that drove the render (the same GT), so the rendered-test
column measures information content, not unbiased accuracy -- the paper
already says so in Table 4's caption.

| arm | train on | selects on | tests on |
|---|---|---|---|
| R -> R | real train | real val | real test (the reference) |
| R -> S | the R -> R checkpoint | -- | rendered test |
| S -> R | rendered train | rendered val | real test (TSTR) |
| S -> S | the S -> R checkpoint | -- | rendered test |
| R+S -> R | real + rendered train, **same optimizer steps as R** (the batch drawn 1:1) | real val | real test (Table 5's augmentation arm) |
| R+S -> S | that checkpoint | -- | rendered test |
| low-shot | real subsets N in {1, 3, 10, 30, all} sequences, with and without the rendered train set | real val | real test (Fig. lowshot; the real-only curve gives Table 5's real-data equivalent) |

Table 5 as written wants *edited* scenes ("Real + 4DPS edits"); no edits
exist yet, so the first fill is real + the same-scene renders and the
caption is adjusted to say so; the editing arm slots in as a fourth
domain (`rendered-edited`) when those products exist, through the same
runner.

Every cell runs five seeds (the author's decision of 2026-09-15; the first
draft said three); the result is mean (std). Seeds are the
Protocol's, so the R and S arms of one seed share their initialisation
and data order (only the bytes differ).

## 4. Step 3: inference across every permutation

`runner.matrix()` evaluates every trained checkpoint on every test
domain, so the results directory holds the full `train x test` grid
(3 x 2 per dataset x 5 seeds = 30 evaluations per dataset), and
`results.tables()` fills Tables 4 and 5 and `main.tex`'s macros
(`\rApReal`, `\rApSub`, `\rApTSTR`, `\rApSelf`, `\rMPJPEReal`,
`\rMPJPESub`, `\rMPJPETSTR`, `\rMPJPESelf`, `\rApRealOnly`, `\rApAug`,
`\rMPJPEAug`, `\rSeqEquiv`, `\rNlow`, `\rApLowShot`, `\rRetention`,
`\rApMeas`) from `results/` alone. No number is typed into the LaTeX.

## 5. Metrics to report (the tables' and the extras)

**Already in Tables 4-5**: HuPR AP, AP50, AP75 (COCO OKS on the 14 HuPR
keypoints, the official evaluator); RT-Pose MPJPE (cm).

**Add** -- each is one column or one supplement table, all from the same
evaluator run:

| metric | why |
|---|---|
| RT-Pose MRPE and Abs-MPJPE alongside MPJPE | the paper's own three; MRPE is localisation, which is where a range mis-registration in the render would show |
| per-joint error (RT-Pose) / per-keypoint AP (HuPR) | limbs vs trunk: the micro-Doppler fit is about limbs, and the sim-to-real gap should be largest on wrists and ankles if the renders under-model them |
| PCK at 10 cm and 15 cm (RT-Pose), PCK at 0.2 of the head size (HuPR) | threshold metrics are less dominated by a few large failures than means |
| per-activity / per-environment breakdown (RT-Pose has both in `filemeta.json`; HuPR by its action set) | which scenes transfer and which do not |
| **retention** = (S -> R) / (R -> R) for AP, and (R -> R) / (S -> R) for MPJPE | the one-number sim-to-real summary the prose quotes (`\rRetention`) |
| **the gap decomposition**: (S -> S) - (S -> R) against (R -> R) - (R -> S) | symmetric or not: whether the renders lack information (low S -> S) or realism (high S -> S, low S -> R) -- the question Sec. 4.5's text already poses |
| mixed-domain test: the R -> R model on a 50/50 real/rendered test set | a cheap check that the model does not behave differently on renders it was never shown |
| the low-shot curve's crossing point: the smallest N at which real-only matches real + rendered | Table 5's real-data equivalent, read off the curve rather than eyeballed |
| the training curves (val metric vs step) for R and S of one seed, overlaid | whether the rendered domain trains as fast and as stably; a lag is a gap the final number hides |
| a feature-space distance between real and rendered test sets under the R -> R model (mean cosine distance of the penultimate features, matched frame by frame) | a label-free sim-to-real distance, per frame, to correlate with the fidelity metrics of Table 3 per sequence -- the link between rendering fidelity and downstream utility |
| the two-person real test set (RT-Pose) under every model | context for the limitation; not in the main tables |

Seeds: three per cell, std reported; a single-seed number is not a
result (docs/METRICS.md). Frame selection: every frame the official
evaluator scores (no motion subset here; the downstream task is
per-frame pose).

## 6. Gates before any table is filled

1. **Adapter identity** (`Adapter.gate()`): the real-domain adapter's
   output for one sequence equals the official preprocessing's output
   for that sequence up to the documented window difference (HuPR: the
   official npy from `process_iwr1843.py` on the raw ADC, compared with
   and without the Hann window; RT-Pose: our polar-to-Cartesian resample
   against `4Dradar2xyz.py` run on a tensor we build from the same cube,
   bitwise on the resample step since it is the same code).
2. **Domain-blindness** (`runner.gate()`): the rendered and real
   `Source`s for one sequence produce tensors of identical shape, dtype,
   axes and normalisation; the training loop's data order and
   initialisation for seed k are identical across domains (hash of the
   first batch's indices and of the initial weights).
3. **Reproduction** (section 2): real -> real within tolerance before a
   rendered arm is launched.
4. **Protocol hash** in every `metrics.json`; the table generator refuses
   to mix results from two hashes.

### 6.3 The VRAM profile is not part of the protocol

`runner.py --vram 24 | 80` (default `$DOWNSTREAM_VRAM`, else 24) says how
many trainings share one card: one on a 24 GB card (RTX 3090 / 4090), the
measured count on an 80 GB card (A100 / H100; `VRAM_PROFILES` in
runner.py, the measurement in docs/DOWNSTREAM_RESULTS.md 1.5). The batch
(HuPR 20, RT-Pose 16), the step count and the learning-rate schedule are
the released recipes' and are the same under every profile: a larger
batch would change the optimisation (Adam at 1e-4 with a decay every 2000
iterations; a one-cycle schedule over a fixed step count), so the
reproduction gate against the papers' numbers would no longer test the
papers' recipe, and a rendered-vs-real gap measured at one batch size
would not be comparable with one measured at another. The 80 GB cards buy
concurrency (more cells at once on one card), not a different experiment.

## 7. Compute and order

| item | GPU-hours (estimate; measure one epoch first) |
|---|---|
| HuPR adapter, both domains, 235 sequences | ~2 (GPU) |
| MSCSA-PRGCN, one arm, one seed, 200 epochs at `-sr 1` | to be measured; the official README recommends `-sr 10` for speed -- if adopted, adopted for every arm |
| HRRadarPose, one arm, one seed, 50 epochs on 46 sequences with on-the-fly tensors | to be measured |
| the matrix: 3 arms x 5 seeds x 2 datasets (the low-shot curve is not run) | ~ (arm cost) x 15 per dataset |

Order: `common/` and the two adapters with their identity gates; the
HuPR real -> real reproduction; the RT-Pose real -> real reproduction on
the one-person split; then the rendered and mixed arms through the
runner; low-shot last; tables generated, macros pasted by the generator.
