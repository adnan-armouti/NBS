# Downstream results: every measured number as it lands

Implements docs/DOWNSTREAM_PLAN.md; the code is `downstream/` (its README
has the layout, the environments and every departure from the released
pipelines). Linear scale throughout, never dB. Every number names its
estimator. Five seeds with the sample std (ddof 1) are a result; a
single-seed number is a smoke test or a timing and is labelled so.

## 0. Status and the decisions taken (2026-09-15, with the author)

* HuPR budget: sampling ratio 10, 200 epochs (protocol hash `38ea71937b32`; `8525a1813438` with three seeds).
* RT-Pose one-person set: the dataset's own "Subject type" label: 64
  one-person (train 45 / val 6 / test 13, `downstream/splits/rtpose_split.json`,
  seed 0, stratified by location, scenario, activity, occlusion), 37
  two-person as the real-only extra test set (protocol hash `989ccef2230e`; `f585e9aeb1b4` with three seeds).
* RT-Pose tolerance: MPJPE <= 14.9 cm and Abs-MPJPE <= 22.1 cm (1.5x the
  paper's 9.93 / 14.73) and |MRPE - MPJPE| <= 3 cm.
* RT-Pose input: the 64 Doppler bins as channels (the paper's HRRadarPose);
  the released `hr3d.py` as written is the no-Doppler ablation.
* Every arm at the real arm's optimizer-step count. The low-shot curve is
  not run. Five seeds for R, S, R+S (decided 2026-09-15, was three); the HuPR window control once.
* The full runs go to an 8 x A100 node (docs/PROMPT_DOWNSTREAM_A100.md);
  nothing beyond gates, timings and smoke tests runs on the workstation.

| step | state |
|---|---|
| 1. common/ + adapters + identity gates + domain-blindness gate | HuPR identity gate passed (section 1.1); RT-Pose resample bitwise (1.2), label alignment (1.3); blindness gate: section 1.4 |
| 2. HuPR real -> real reproduction | **run on the node, five seeds (2.3): AP 65.94 +/- 1.67 vs the paper's 63.4, band [61.4, 65.4] -- gate FAILED as written, 0.54 above the band (high side); treated as passed by the author 2026-09-16**; released weights 61.8 (2.2); unwindowed control 68.2 (2.4) |
| 3. RT-Pose real -> real reproduction | **run on the node, five seeds (3.2): MPJPE 16.07 +/- 1.72 cm (<= 14.9 FAILED), Abs-MPJPE 18.01 +/- 1.61 (<= 22.1 ok), \|MRPE - MPJPE\| 5.30 +/- 1.33 (<= 3 FAILED); treated as passed by the author 2026-09-16** |
| 4. rendered and mixed arms | HuPR done, five seeds (4.1): S -> R 15.8 +/- 2.0, S -> S 53.9 +/- 1.4, R+S -> R 60.8 +/- 1.2, R+S -> S 57.7 +/- 2.1; window control 68.2 (2.4). RT-Pose held: rendered static background diagnosed (3.3) |
| 5. tables and macros | generator written (`downstream/common/results.py`); \rApMeas already fills |

## 1. Step 1: the adapters and their gates

### 1.1 HuPR adapter identity (results/gates/hupr_adapter_seq15/gate.json)

Sequence 15 (a test sequence), frames {0, 100, 300, 599}, both boards.
Reference (a): `process_iwr1843.RadarObject.generateHeatmap` on the raw
ADC as the released reader decodes it (complex128, unwindowed range FFT).
Ours: the adapter (`downstream/hupr_mscsa/adapter.py`, float32 on the
GPU) on the real cube from `MeasuredReader` + `range_profiles_all_channels_torch`.
"model input" = after the released `Normalize` (per-map min-max then
z-score); its chirp 4 of 8 is the zero-Doppler bin, exactly zero after
the released clutter removal and therefore unit-variance rounding noise
in both pipelines -- excluded from the identity number, reported alone.

| variant (window, precision) | heatmap rel. L2 error | heatmap |.| Pearson | model input, 7 non-DC chirps: max abs err / Pearson | DC chirp Pearson |
|---|---|---|---|---|
| (b) none, complex64 -- the code identity | 8.9e-7 (h) / 9.9e-7 (v) | 1.000 / 1.000 | 1.7e-5 / 1.000 (h); 1.5e-5 / 1.000 (v) | -0.0006 / 0.004 |
| (c) none, float16 cube (the cache's precision) | 1.2e-3 / 1.4e-3 | 1.000 / 1.000 | 0.047 / 1.000; 0.041 / 1.000 | 0.014 / 0.011 |
| (d) Hann, float16 -- the production cube | 0.578 / 0.591 | 0.918 / 0.926 | 10.9 / 0.823; 10.8 / 0.834 | 0.003 / -0.003 |

The range crop to bins 31..94 against the uncropped path: 8e-7 of the
peak (cuFFT plan rounding, not bitwise). The Hann window (row d) is the
documented departure: it changes the normalised input substantially
(Pearson 0.82-0.83 against the released input); its effect on AP is what
the `real_nowindow` control arm measures (section 2). The DC chirp's
magnitude is 1e-17 of the heatmap's peak in the released pipeline: the
released model reads one noise channel in eight.

#### 1.1 node: the same gate on mmsd-h100 (H100; results/gates/hupr_adapter_seq15_node/gate.json, 2026-09-15)

Same sequence, frames and reference. Every number equals the workstation's
to 2e-6 absolute or better: (b) heatmap rel. L2 8.94663e-7 (h) / 9.91425e-7
(v), Pearson 1.000 / 1.000, model-input non-DC Pearson 1.000 / 1.000; (c)
1.21006e-3 / 1.44229e-3, non-DC max abs err 0.0467037 / 0.0406836; (d) 0.578073 /
0.591401, |.| Pearson 0.918081 / 0.926169, non-DC Pearson 0.823355 / 0.833536,
max abs err 10.9263 / 10.7703. The only value that differs beyond print
precision is (b)'s horizontal non-DC max abs err, 1.645e-5 on the node
against 1.717e-5 on the 4090 -- a GPU-rounding max at the 1e-5 level, the
difference the run sheet anticipates. Gate passed on the node.

### 1.2 RT-Pose resample, bitwise (results/gates/rtpose_adapter_seq183/gate.json)

Sequence 183, frames {10, 150}: the GPU gather (`RTPoseAdapter.cartesian`)
against `4Dradar2xyz.py`'s own `c2p` / `findIndexForBiInt` /
`getBiIntValue` run cell by cell on the same polar tensor over the roi1
box: **bitwise equal on all 10,485,760 cells of each frame** (max abs err
0.0). The tables reproduce the script's index-minus-one in
`findIndexForBiInt` (it extrapolates from the two bins below the sample,
a one-bin shift the labels were aligned to) and its `np.ones` fill:
103,930 of the 163,840 ROI cells resolve, the rest lie outside the
0-120 deg / 0-60 deg wedge or fail the index test and stay at 1 + 0j
(0 after the released normalisation). The virtual grid is 86 x 7 slots,
134 of 192 channels kept (`Process_4DRT`'s first-occurrence rule).

#### 1.2 node (results/gates/rtpose_adapter_seq183_node/gate.json, mmsd-h100, 2026-09-15)

Sequence 183, frames {10, 150}, the real cube from the node's own cache:
`"bitwise": true` on both frames, max abs err 0.0, 10,485,760 cells
compared each; virtual grid 86 x 7, 134 of 192 channels kept;
`table_matches_mm4dps_rotated_positions: true`. Identical to the workstation.

### 1.3 RT-Pose label alignment (results/gates/rtpose_label_alignment/)

Sequence 183 (walking), 82 annotated frames: the (x, y) centroid of the
top 1% of the tensor's moving energy (|v| summed over Doppler bins
|d - 32| >= 3) against the annotated pelvis, Pearson r over frames.

| antenna table | r(x) | r(y) | mean offset (x, y, z) m | rms offset m | moving energy within 0.6 m of the pelvis |
|---|---|---|---|---|---|
| `hardware_param.m` as MATLAB reads it (used) | 0.970 | **+0.919** | (-0.32, -0.13, +0.26) | (0.40, 0.57, 0.36) | 7.2% |
| 4DPS's rotated table | 0.969 | **-0.905** | (-0.31, -0.88, +0.25) | (0.39, 2.02, 0.33) | 6.9% |

With the authors' table and the script's angle mapping the person lands
where the annotation is; with 4DPS's rotated table the same mapping
mirrors y. The two focus the moving energy on the person equally, so the
data does not choose between them; the authors' is used, on both
domains alike.

#### 1.3 node (results/gates/rtpose_label_alignment_node/, mmsd-h100, 2026-09-15)

`hardware_param.m` table, 82 frames: r(x) 0.9698, r(y) +0.9186, r(z) -0.059;
mean offset (-0.323, -0.131, +0.265) m, rms (0.395, 0.567, 0.355) m -- the
workstation's numbers to the printed digit (the cube and the resample are
bitwise, so the alignment is too).

### 1.4 Domain blindness (results/gates/hupr_blindness_seed0.json)

HuPR, seed 0, the real and the rendered training sets of the official
split through `downstream.hupr_mscsa.run blindness`: model input
(20, 8, 8, 2, 64, 64, 8) float32 from a (20, 8, 64, 3, 4, 64, 2) float16
payload in both domains; first batch hash `f2b0a7169deedf02` in both;
initial weights hash `97ceb86ab4bb43fb` in both; 5,790 steps per epoch
(115,800 samples) in both; input mean 4e-9, std 0.9999 in both (the
released Normalize). **PASSED.**

RT-Pose (results/gates/rtpose_blindness_seed0.json), seed 0, the 45
training sequences of the frozen split in both domains: input
(16, 64, 16, 64, 160) float32 from a (16, 64, 12, 16, 256, 2) float16
payload in both; first batch hash `aa571327c27fe32c` in both; initial
weights `1918f63b0bc7a737` in both; 543 steps per epoch
(8677 annotated frames) in both. The normalised tensor's mean / std /
fraction > 0: real 0.0108 / 0.1687 / 0.271, rendered
0.0209 / 0.4303 / 0.241 -- comparable under the released normalisation constants, the renders
carrying more energy above the 1e5 floor (std 0.43 against 0.17; a
fidelity fact for the write-up, not a gate). **PASSED.**

#### 1.4 node (results/gates/{hupr,rtpose}_blindness_seed0.json rewritten on mmsd-h100, 2026-09-15)

HuPR: first batch `f2b0a7169deedf02`, initial weights `97ceb86ab4bb43fb`
in both domains; input (20, 8, 8, 2, 64, 64, 8) float32 from
(20, 8, 64, 3, 4, 64, 2) float16; **579** steps per epoch (115,800
samples at sampling ratio 10 -- the workstation file had said 5,790 from a
ratio-1 run; every hash unchanged). All six checks true, **PASSED**.
RT-Pose: first batch `aa571327c27fe32c`, initial weights `1918f63b0bc7a737`
in both; input (16, 64, 16, 64, 160) float32 from (16, 64, 12, 16, 256, 2)
float16; 543 steps per epoch (8,677 frames). All six checks true, **PASSED**.
Both sets of hashes equal the workstation's.

### 1.5 The node: caches, smoke fits, timing and the 80 GB profile (mmsd-h100, 8 x H100 80 GB, 1.8 TB RAM, 2026-09-15)

Caches (`$DOWNSTREAM_CACHE = /data/mmsd/4DPS_downstream`, an 18 TB
persistent volume; the 1.8 TB of RAM holds every cube in the page cache
after its first read, so no second copy on the local SSD is made):
HuPR `check` 940 ok (real + rendered) and 470 ok (real_nowindow), 0
MISSING, 155 GB; RT-Pose `check` 202 ok, 0 MISSING, 708 GB dereferenced
(354 GB on disk + the rendered symlinks into the products). Materialisation
4 s a HuPR board, 40 s an RT-Pose sequence (four GPUs in parallel).

Smoke fits (never a result): HuPR sequence 15, real, seed 0:
`epoch 0: train loss 1.21327 (7 s, 3 steps), val AP 0.0000`; RT-Pose 183:
`epoch 1: train_seconds 9.17, steps 11, val MPJPE 418.8 cm` (one
sequence, one epoch; the val numbers are noise by construction).

HuPR timing, the full real training set (193 sequences, 579 steps of 20
at ratio 10), one epoch, `OMP_NUM_THREADS=4`, page cache warm:

| trainings on one H100 | s per step, each | s per step of work, aggregate | GPU util | peak VRAM |
|---|---|---|---|---|
| 1 | **0.207** | 0.207 | 87% | 23 GB |
| 2 | 0.427 | 0.213 | 94% | 45 GB |
| 3 | 0.642 | 0.214 | 96% | 67 GB |

One training saturates the card at the protocol's batch of 20: a second
or third per card buys nothing (-3%). The same epoch on a **cold** page
cache (the first read of the caches from the persistent volume) took
1.98 s a step at 9% util, and with `OMP_NUM_THREADS` unset (208 cores,
every loader worker spawning 208 threads) 0.311 s: the runner sets
`OMP_NUM_THREADS=4`, and the caches must be read once before timing
anything. Against the 4090's 0.389 s at ratio 1 the H100 is 1.9x; a seed
is 200 x (579 x 0.207 + ~12 s val) = **~7.4 h**.

A third simultaneous HuPR start crashed reading `coco_gt/*_gt.json` while
another wrote it (JSONDecodeError): the file is now written atomically
(`dataset.coco_gt`). The full-split RT-Pose fit crashed at iteration 20
in the released `TensorboardLoggerHook` (every det3d log var is a list;
the 11-step smoke never reaches the interval): the hook is dropped in
`rtpose_hrradarpose/trainer.py`, the text logger and `history.json`
remain; the resolved config is written atomically as well.

RT-Pose timing, the 45 training sequences (543 steps of 16), one epoch,
`OMP_NUM_THREADS=4`, 4 workers, page cache warm:

| trainings on one H100 | s per step, each | s per step of work, aggregate | GPU util | peak VRAM |
|---|---|---|---|---|
| 1 | **0.601** | 0.601 | 46% | 15 GB |
| 2 | 0.714 / 0.732 | 0.357 (1.68x) | 84% | 29 GB |
| 3 | 1.013 / 1.020 / 1.022 | 0.339 (1.77x) | 95% | 43 GB |

Alone the card idles half the time (the adapter and the loader are the
bound); two per card is 1.68x the throughput at +19% per job, three
1.77x at +69%. A seed: 50 x 543 x 0.601 = 4.5 h alone, 5.4 h at two per
card (+ val every 5 epochs, 8 s each).

`VRAM_PROFILES["80"]` in runner.py: **HuPR 1, RT-Pose 2** per card. The
real arms of five seeds: HuPR on five cards (~7.4 h), RT-Pose five seeds
in six slots on three cards (~5.4 h), one round for both.

## 2. Step 2: HuPR real -> real (MSCSA-PRGCN)

Reference: HuPR paper (arXiv 2210.12564, WACV 2023), Table 3, "Ours (CSAM +
PRGCN)" on the official test split: **AP 63.4, AP50 97.0, AP75 74.0**; per
keypoint AP head 77.5, neck 81.9, shoulder 70.3, elbow 45.5, wrist 22.3,
hip 88.1, knee 82.2, ankle 73.1. Training: Adam 1e-4, x0.999 every 2000
iterations, batch 20, weight decay 1e-4, 8 frames (section 5.2 of the
paper). Gate: our R -> R AP within +/- 2 of 63.4.

### 2.2 The released weights (\rApMeas)

`model_best.pth` from the authors' drive (the checkpoint records epoch 5
and a val AP of 70.5), scored with `downstream.hupr_mscsa.run released`
on the official test split, 12,600 frames, the released COCOeval
(results/hupr/released_weights/metrics_*.json):

| cube fed to the released weights | AP | AP50 | AP75 | AR | PCKh@0.2 | per-keypoint AP (head, neck, shoulders, elbows, wrists, hips, knees, ankles) |
|---|---|---|---|---|---|---|
| real, window off (what the weights were trained on) -- **\rApMeas** | **61.8** | 96.8 | 71.7 | 71.7 | 38.2 | 76.3, 80.6, 70.7/72.1, 37.7/44.8, 18.7/19.9, 88.1/85.4, 83.4/77.1, 69.6/73.1 |
| real, Hann (the production cube; a train/test window mismatch) | 55.6 | 92.2 | 62.4 | 66.7 | 35.3 | 72.0, 75.1, 65.7/65.8, 31.7/40.7, 15.2/17.3, 83.5/80.5, 76.6/71.1, 60.7/68.0 |

Paper (Table 3): 63.4 / 97.0 / 74.0; per keypoint 77.5, 81.9, 70.3, 45.5,
22.3, 88.1, 82.2, 73.1. The released weights through our adapter and
evaluator land 1.6 AP under the paper on every keypoint group's pattern
(wrists lowest, hips highest): the adapter + evaluator path reproduces
the released pipeline. Feeding the Hann cube to weights trained on the
unwindowed one costs 6.2 AP -- a mismatch effect; what the window does
to a model trained on it is the `real_nowindow` control arm's number.

### 2.1 Timing (one epoch, sampling ratio 1, seed 0, one GPU; a timing, not a result)

`downstream.hupr_mscsa.run fit --timing --epochs 1` on the full real
training set (193 sequences, 115,800 samples, batch 20): **5,790 steps in
2,250 s = 0.389 s per step** (the model's fp32 forward/backward is 0.35 s
of it; cudnn.benchmark and TF32 change nothing; the adapter is 0.045 s
per batch, data loading 0.008 s with 8 workers); the full val split
(12,600 frames, batch 32) in 95 s. Val AP after this one epoch: 44.8.
Peak GPU memory 16 GB.

| sampling ratio | steps / epoch | one epoch | 200 epochs (the released count) | 100 epochs |
|---|---|---|---|---|
| 1 (released default) | 5,790 | 39 min | 130 h (5.4 days) | 65 h |
| 5 | 1,158 | 8.0 min | 27 h | 13 h |
| 10 (the README's speed setting) | 579 | 4.0 min | 13 h | 6.5 h |

The HuPR matrix is 34 trainings (R, S, R+S x 3 seeds; four low-shot
budgets x with/without renders x 3 seeds; the unwindowed control once),
i.e. 442 GPU-hours at sampling ratio 10 / 200 epochs -- 9.2 days on the
two cards -- against 4,420 GPU-hours at the released ratio 1.

### 2.3 node: the reproduction (mmsd-h100, five seeds, protocol `38ea71937b32`, 2026-09-15/16)

`runner run --dataset hupr --arms real --gpus 0 1 2 3 4 --workers 8` at
`DOWNSTREAM_VRAM=80` (one training per card), launched 17:18 UTC, all
five `real/seed<k>: ok` at 00:17 UTC; no crash, no rerun. Model selection
as released: `model_best.pth` = best val AP over the 200 epochs
(checkpoint epochs 170 / 101 / 194 / 163 / 188). Official test split,
12,600 frames, the released COCOeval; mean over the five seeds with the
sample std (ddof 1); results/hupr/real__real/seed<k>/metrics.json.

**Gate: R -> R AP within +/- 2 of 63.4, i.e. [61.4, 65.4]. Measured
65.94 +/- 1.67 (per seed 66.87 / 63.17 / 66.76 / 65.59 / 67.28): the mean
is 0.54 AP above the band -- FAILED as written, on the high side** (four
of five seeds above 65.4; the gate's intent, that the pipeline reproduces
the paper, is met in the sense that nothing is lost, but the number is
not the paper's and the band is the plan's, so it is recorded as a fail
and the window control and the rendered/mixed arms were not launched).

| | AP | AP50 | AP75 | AR | PCKh@0.2 | AP_M | AP_L | mean frame OKS |
|---|---|---|---|---|---|---|---|---|
| **this reproduction, R -> R (n=5)** | **65.94 +/- 1.67** | 96.97 +/- 0.20 | 78.81 +/- 3.18 | 75.29 +/- 1.31 | 44.14 +/- 1.85 | 68.86 +/- 1.32 | 63.96 +/- 2.13 | 85.03 +/- 0.69 |
| paper, Table 3 (CSAM + PRGCN) | 63.4 | 97.0 | 74.0 | -- | -- | -- | -- | -- |
| released weights, unwindowed cube (2.2, \rApMeas) | 61.8 | 96.8 | 71.7 | 71.7 | 38.2 | -- | -- | -- |

Per-keypoint AP (mean +/- std over five seeds), beside the paper's Table 3
groups and the released weights (2.2):

| keypoint | this reproduction | paper | released weights |
|---|---|---|---|
| Head | 81.7 +/- 1.1 | 77.5 | 76.3 |
| Neck | 83.5 +/- 1.8 | 81.9 | 80.6 |
| L / R shoulder | 74.7 +/- 1.1 / 77.0 +/- 2.0 | 70.3 | 70.7 / 72.1 |
| L / R elbow | 41.5 +/- 1.4 / 50.9 +/- 3.8 | 45.5 | 37.7 / 44.8 |
| L / R wrist | 22.5 +/- 1.7 / 25.6 +/- 2.8 | 22.3 | 18.7 / 19.9 |
| L / R hip | 90.7 +/- 1.2 / 90.3 +/- 0.6 | 88.1 | 88.1 / 85.4 |
| L / R knee | 84.1 +/- 2.1 / 84.8 +/- 0.9 | 82.2 | 83.4 / 77.1 |
| L / R ankle | 77.5 +/- 1.1 / 73.5 +/- 1.5 | 73.1 | 69.6 / 73.1 |

Per-keypoint PCKh@0.2 (%): hips 60.8 / 60.2, knees 59.8 / 60.6, ankles
46.7 / 42.7, neck 37.5, head 34.7, shoulders 48.1 / 51.3, elbows 29.1 /
34.1, wrists 26.3 / 26.1 (std 1.1-3.2). Mean pixel error: hips ~5.0,
knees ~5.0, ankles 6.3-6.9, neck 6.4, head 6.7, shoulders 5.7-6.0, elbows
9.1-10.7, wrists 15.2-17.4 px. The pattern is the paper's and the released
weights' (wrists lowest, hips highest), uniformly 2-4 AP higher; the seed
spread (1.67) is of the order of the paper's distance to its own released
weights (1.6).

The same five checkpoints on the other test sets (real -> X; the
rendered/mixed arms themselves are not run):

| test set | AP | AP50 | AP75 | AR | PCKh@0.2 | per seed AP |
|---|---|---|---|---|---|---|
| rendered test (12,600 frames) | 21.98 +/- 3.83 | 63.61 +/- 6.25 | 7.86 +/- 2.88 | 33.89 +/- 3.50 | 12.33 +/- 2.32 | 24.06 / 16.32 / 21.33 / 21.58 / 26.63 |
| real mixed test (12,600 frames) | 44.81 +/- 2.30 | 82.01 +/- 2.65 | 44.69 +/- 3.39 | 52.32 +/- 2.21 | 26.89 +/- 2.21 | 45.46 / 41.23 / 45.37 / 44.47 / 47.53 |

Training curves (history.json; val AP in % at epochs 1, 10, 25, 50, 75,
100, 125, 150, 175, 200; the released val = the official test split):

| seed | 1 | 10 | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 | best (epoch) | train loss 1 -> 200 | val loss 1 -> 200 (min, epoch) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 12.7 | 47.4 | 59.2 | 66.4 | 65.4 | 69.3 | 68.5 | 68.6 | 69.8 | 70.2 | 72.76 (171) | 0.0593 -> 0.0248 | 0.0394 -> 0.0271 (0.0266, 196) |
| 1 | 9.9 | 48.4 | 60.4 | 66.3 | 63.5 | 66.2 | 67.5 | 67.7 | 69.0 | 69.5 | 70.22 (102) | 0.0602 -> 0.0249 | 0.0408 -> 0.0273 (0.0270, 174) |
| 2 | 11.9 | 44.3 | 58.9 | 67.1 | 68.6 | 67.7 | 68.2 | 70.9 | 70.5 | 71.2 | 72.66 (195) | 0.0605 -> 0.0245 | 0.0412 -> 0.0269 (0.0267, 195) |
| 3 | 14.1 | 48.7 | 60.2 | 67.7 | 67.0 | 69.4 | 65.3 | 71.2 | 72.3 | 72.0 | 73.09 (164) | 0.0628 -> 0.0248 | 0.0397 -> 0.0268 (0.0266, 154) |
| 4 | 15.8 | 47.2 | 59.1 | 66.7 | 68.6 | 71.0 | 68.8 | 71.4 | 70.5 | 68.7 | 73.43 (189) | 0.0592 -> 0.0247 | 0.0402 -> 0.0276 (0.0267, 189) |

The val AP the trainer logs (best 70.2-73.4) is on the split's own 21
val sequences (hupr_split.json; every 10th frame, 1,260 frames, the
released `index * sampling_ratio` indexing) and selects `model_best.pth`;
the test AP above (63.2-67.3) is the same checkpoint on the 21 official
test sequences, every frame (12,600), through the released `evaluate`
(`AP_released_evaluate` equals the `AP` written). The val split scores
5.9-7.5 AP above the test split on every seed -- the two sets differ, not
the estimator; nothing of the test split enters selection.

Wall time and speed against 1.5's estimate (~7.4 h a seed at 0.207
s/step): 6.73-6.85 h a seed for the 200 epochs (train + in-loop val),
0.197-0.201 s/step (mean over the epochs), 121-123 s an epoch, 23 GB and
90-99% util per card; the three evals 81-86 s each. The whole HuPR round:
6 h 59 min wall from launch to the runner's last `ok`.

### 2.4 node: the window control (real_nowindow, one seed, mmsd-h100, 2026-09-16)

`runner run --dataset hupr --arms real_nowindow --gpus 5 --workers 8`: the
real arm's protocol on the unwindowed official cube (`real_nowindow`, the
released preprocessing's range FFT without our Hann window), trained and
tested unwindowed, seed 0, one seed by the plan (a control, not a result;
launched 01:38 UTC, `ok` 08:20 UTC, 6.63 h, 0.195 s/step, checkpoint epoch
187; results/hupr/real_nowindow__real_nowindow/seed0/metrics.json).

| cube (train = test) | AP | AP50 | AP75 | AR | PCKh@0.2 | per keypoint (head, neck, shoulders, elbows, wrists, hips, knees, ankles) |
|---|---|---|---|---|---|---|
| Hann (the production cube), five seeds (2.3) | 65.94 +/- 1.67 | 96.97 +/- 0.20 | 78.81 +/- 3.18 | 75.29 +/- 1.31 | 44.14 +/- 1.85 | 81.7, 83.5, 74.7/77.0, 41.5/50.9, 22.5/25.6, 90.7/90.3, 84.1/84.8, 77.5/73.5 |
| unwindowed (the released cube), seed 0 | 68.20 | 98.29 | 82.19 | 77.07 | 45.58 | 84.2, 84.1, 78.4/77.9, 45.1/56.0, 24.0/28.2, 92.0/91.1, 85.3/85.8, 80.5/72.5 |
| Hann, seed 0 alone | 66.87 | 97.17 | 80.64 | 75.83 | 45.18 | 81.9, 85.1, 75.8/78.8, 40.3/53.8, 21.5/27.0, 92.0/90.4, 86.5/84.9, 77.5/74.8 |

The unwindowed control sits 2.3 AP above the Hann arm's mean (1.4 std; 1.3
AP above the same seed's Hann number), higher on every keypoint but the
right ankle: the Hann range window costs the trained model of the order of
1-2 AP, not the 6.2 the released weights lose when fed a cube they were not
trained on (2.2), and does not change the picture of 2.3 (the reproduction
sits above the paper either way; the unwindowed control is 4.8 above it).
Its val curve: 16.5 / 58.7 / 67.3 / 69.7 / 69.9 / 69.7 at epochs 1 / 25 /
50 / 100 / 150 / 200, best 72.3 @188 -- on the Hann seeds' curve.

## 3. Step 3: RT-Pose real -> real (HRRadarPose)

Reference: RT-Pose paper (arXiv 2407.13930, ECCV 2024), Table 2:
HRRadarPose **MRPE 9.91 cm, MPJPE 9.93 cm, Abs-MPJPE 14.73 cm** on its own
test set of the full dataset (Table 5's Doppler row quotes 9.91 / 14.73);
without Doppler (averaged over the Doppler axis) 11.31 / 22.34. Table 3
per activity: MRPE 3.35-14.11, MPJPE 6.91-15.28, Abs-MPJPE 7.97-21.22.
Table 4: occlusion 11.57 / 18.55 against 9.84 / 14.52. Tolerance (plan
2.2): MPJPE within 1.5x of 9.93 cm on our test split, MRPE >= MPJPE ordering
of Table 2 preserved within the activity spread of Table 3.

### 3.1 The split (downstream/splits/rtpose_split.json)

64 usable one-person sequences by `filemeta.json`'s "Subject type"
(37 multi-person: 100-106, 117-140, 170-175 -- the sweep's YOLO count had
106 and 174 as one-person; the dataset's label is used), stratified by
(location, scenario, activity, occlusion) and dealt 0.70 / 0.09 / 0.21 by
largest remainder along the stratum-ordered, seed-0-shuffled list:
train 45, val 6, test 13. Per factor (train / val / test): indoor 28/4/8,
outdoor 17/2/5; clean 12/2/3, cluttered 16/2/5, normal 10/1/4, rain 7/1/1;
stand+wave 14/1/5, walk 10/3/4, walk+wave 7/1/1, random pose 6/0/0,
lift leg 4/0/2, walk+sit 4/1/1; no occlusion 33/5/9, cloth 5/1/0,
cardboard 3/0/2, plastic 4/0/2. The val split carries no random-pose or
lift-leg sequence (six sequences cannot cover six activities); it selects
nothing (the released rule scores the final epoch) and only draws the
training curves.

### 3.2 node: the reproduction (mmsd-h100, five seeds, protocol `989ccef2230e`, 2026-09-15/16)

`runner run --dataset rtpose --arms real --gpus 5 6 7 --workers 4` at
`DOWNSTREAM_VRAM=80` (two trainings per card; seed 4 alone on GPU 7),
launched 17:18 UTC, all five `real/seed<k>: ok` at 23:10 UTC; no crash,
no rerun. Model selection as released: the final (50th) epoch's
checkpoint. Our test split (3.1), 13 sequences, 2,525 frames; mean over
five seeds with the sample std (ddof 1);
results/rtpose/real__real/seed<k>/metrics.json.

**Gate (protocol `extra.gate`): MPJPE <= 14.9 cm and Abs-MPJPE <= 22.1 cm
and |MRPE - MPJPE| <= 3 cm. Measured MPJPE 16.07 +/- 1.72 cm (FAIL, +1.17
over), Abs-MPJPE 18.01 +/- 1.61 cm (ok), MRPE 10.77 +/- 0.89 cm so
|MRPE - MPJPE| = 5.30 +/- 1.33 cm (FAIL, MRPE < MPJPE by 5.3 where Table 2
has them within 0.02) -- FAILED on two of the three clauses.** One seed
(4) is inside the MPJPE clause (13.59), none inside the MRPE clause.

| | MPJPE | MRPE | Abs-MPJPE | PCK@10 cm | PCK@15 cm | per seed MPJPE / Abs / MRPE |
|---|---|---|---|---|---|---|
| **this reproduction, R -> R (n=5)** | **16.07 +/- 1.72** | 10.77 +/- 0.89 | 18.01 +/- 1.61 | 48.24 +/- 2.21 % | 69.20 +/- 2.24 % | 17.49/18.68/12.11, 15.03/17.12/10.44, 16.71/18.34/11.20, 17.53/20.06/10.23, 13.59/15.83/9.89 |
| paper, Table 2 (HRRadarPose, its own split of the full set) | 9.93 | 9.91 | 14.73 | -- | -- | |
| tolerance | <= 14.9 | \|MRPE-MPJPE\| <= 3 | <= 22.1 | | | |

Per joint (cm, mean +/- std over seeds; MPJPE is pelvis-relative so the
pelvis is 0 and its Abs-MPJPE is the MRPE):

| joint | MPJPE | Abs-MPJPE |
|---|---|---|
| Pelvis | 0 | 10.77 +/- 0.89 |
| R / L hip | 8.71 +/- 0.69 / 9.03 +/- 0.93 | 11.51 / 11.92 |
| R / L knee | 12.39 +/- 2.39 / 12.95 +/- 2.63 | 14.00 / 14.32 |
| R / L ankle | 13.89 +/- 0.92 / 14.69 +/- 1.34 | 14.89 / 15.35 |
| Thorax | 11.82 +/- 1.87 | 13.10 |
| Head | 19.28 +/- 4.35 | 20.39 |
| L / R shoulder | 13.70 +/- 1.56 / 13.56 +/- 2.28 | 14.43 / 14.96 |
| L / R elbow | 21.26 +/- 3.28 / 21.69 +/- 2.33 | 21.87 / 22.54 |
| L / R wrist | 29.40 +/- 2.52 / 38.70 +/- 7.09 | 30.49 / 39.56 |

Per group (MPJPE / MRPE / Abs-MPJPE, cm, mean over seeds; n = test
sequences): activity -- stand + lift leg (2) 10.29 / 8.28 / 11.74,
stand + wave (5) 19.46 / 12.86 / 21.25, walk (4) 16.53 / 10.87 / 18.91,
walk + sit (1) 11.55 / 8.63 / 14.21, walk + wave (1) 13.42 / 7.09 / 14.50
(Table 3's spread is MPJPE 6.91-15.28); location -- indoor (8) 16.83 /
11.58 / 19.08, outdoor (5) 14.85 / 9.48 / 16.28; scenario -- clean (3)
16.68 / 11.92 / 19.54, cluttered (5) 16.93 / 11.38 / 18.81, normal (4)
14.52 / 9.25 / 15.75, rain (1) 16.16 / 10.40 / 18.41; occlusion -- none
(9) 14.63 / 9.75 / 16.73, cardboard (2) 21.86 / 13.59 / 23.40, plastic pad
(2) 16.75 / 12.53 / 18.34 (Table 4: 11.57 / 18.55 occluded vs 9.84 /
14.52). Frame-MPJPE quantiles (seed 0): 5% 6.2, 25% 9.8, median 13.9, 75%
18.7, 95% 35.6 cm -- the mean is carried by the wrist tail (right wrist up
to 109 cm on sequence 1, seed 0). The wave-hand sequences and the two
cardboard-occluded ones are the whole excess over the paper.

The same five checkpoints on the other test sets:

| test set | MPJPE | MRPE | Abs-MPJPE | PCK@10 / 15 cm | per seed MPJPE |
|---|---|---|---|---|---|
| two-person (37 seq, 7,239 frames; real only) | 54.87 +/- 6.02 | 97.34 +/- 3.79 | 103.39 +/- 3.15 | 27.6 / 43.5 % | 52.50 / 53.85 / 59.65 / 61.75 / 46.62 |
| rendered test (13 seq, 2,525 frames) | 152.64 +/- 39.93 | 148.50 +/- 26.74 | 162.04 +/- 21.52 | 19.9 / 31.7 % | 204.4 / 111.8 / 123.1 / 139.6 / 184.1 |
| real mixed test (13 seq) | 78.31 +/- 18.72 | 76.18 +/- 17.41 | 88.45 +/- 9.49 | 32.2 / 49.0 % | 92.9 / 52.0 / 69.4 / 78.6 / 98.7 |

On the two-person set the single-person head (nms 1 / 1) picks one of
the two subjects, so MRPE ~ 1 m is the inter-subject distance and
pelvis-relative MPJPE 55 cm is the pose of the wrong person half the
time. The real -> rendered number is not a domain gap of the ordinary
kind: 1.5 m of root error with per-activity means from 28 cm (lift leg)
to 289 cm (stand + wave) and 527 cm on the rain sequence says the model
finds nothing where the rendered cube puts the subject on most of the
sequences -- to be read against the rendered arm's own R+S / S -> R
numbers when they run (section 4), and against 1.4's blindness gate.
The real-mixed test set mixes real and rendered sequences, so its
by-group rows coincide with the R -> R rows on the real ones (cluttered,
rain, cardboard) and with the rendered rows on the others.

Training curves (history.json; val every 5 epochs on the 6-sequence val
split, MPJPE / Abs-MPJPE / MRPE in cm; the released rule scores the
final epoch, the val split selects nothing):

| seed | ep 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 (scored) | best val MPJPE (epoch) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 15.98 / 21.51 / 16.08 | 14.55 / 20.39 / 15.14 | 15.31 / 20.78 / 15.59 | 14.41 / 19.59 / 14.60 | 14.56 / 19.69 / 15.41 | 14.51 / 19.76 / 14.85 | 14.37 / 19.53 / 14.84 | 14.65 / 19.83 / 15.06 | 15.08 / 20.00 / 15.73 | 15.25 / 20.19 / 15.74 | 14.37 (35) |
| 1 | 16.34 / 21.62 / 16.36 | 15.84 / 20.88 / 15.74 | 15.41 / 20.99 / 15.85 | 15.93 / 21.36 / 15.69 | 14.54 / 19.04 / 13.75 | 14.62 / 19.84 / 14.90 | 15.11 / 20.37 / 15.75 | 14.63 / 19.85 / 14.67 | 15.38 / 20.43 / 14.67 | 15.70 / 20.66 / 14.83 | 14.54 (25) |
| 2 | 16.73 / 22.21 / 16.35 | 15.14 / 20.65 / 16.06 | 15.77 / 20.54 / 14.88 | 14.88 / 19.77 / 15.14 | 14.67 / 19.41 / 14.83 | 14.97 / 19.49 / 15.26 | 15.39 / 19.51 / 14.58 | 15.27 / 19.63 / 14.92 | 15.49 / 19.81 / 15.10 | 15.76 / 19.87 / 15.37 | 14.67 (25) |
| 3 | 15.60 / 21.07 / 14.96 | 14.72 / 20.50 / 15.29 | 14.61 / 20.80 / 15.58 | 14.46 / 20.12 / 14.97 | 14.49 / 19.86 / 14.83 | 14.47 / 19.77 / 14.61 | 14.65 / 20.03 / 14.56 | 14.64 / 19.85 / 14.43 | 17.34 / 22.47 / 14.51 | 18.50 / 23.63 / 14.55 | 14.46 (20) |
| 4 | 15.81 / 21.33 / 15.75 | 14.96 / 20.44 / 14.55 | 14.29 / 19.50 / 14.45 | 14.52 / 19.81 / 14.36 | 14.13 / 19.52 / 14.41 | 13.60 / 18.92 / 14.08 | 14.10 / 19.31 / 14.25 | 13.91 / 19.19 / 14.39 | 14.26 / 19.38 / 14.41 | 14.47 / 19.57 / 14.45 | 13.60 (30) |

The val MPJPE plateaus at 14.1-15.0 cm from epoch 20 on every seed and
drifts up over the last 10-15 epochs (the one-cycle tail; seed 3 from
14.64 at epoch 40 to 18.50 at 50, the one seed whose val curve
diverged); on the val split MRPE ~ MPJPE (within 1 cm) as in Table 2,
on the test split MRPE is 5 cm under MPJPE -- the val split has no
occlusion-heavy or wave-hand-outdoor sequence (3.1). The plateau is a
property of the split's 45 training sequences (the paper trains on ~150
of the full set's one- and two-person sequences), not of the epoch count.

Wall time and speed against 1.5's estimate (5.4 h a seed at two per
card, 0.714 / 0.732 s/step): 5.52-5.57 h a seed for the paired ones
(0.731-0.738 s/step, 397-401 s an epoch), 5.00 h for the one alone on
GPU 7 (0.663 s/step, 360 s an epoch; 1.5 measured 0.601 alone), 28-29 GB
paired / 14 GB alone, 90-100% util; the four evals 93-116 s (real,
rendered, mixed) and 219-279 s (two-person). The whole RT-Pose round:
5 h 51 min from launch to the runner's last `ok`.

### 3.3 node: why R -> S is 1.5 m off -- the rendered static background (diagnosis, 2026-09-16)

Never a result; scripts in /data/mmsd/logs/4dps/ (diag_rtpose.py,
probe_rtpose.py, diag_static.py, diag_range.py). Both domains are the same
object (per-loop complex range profiles, `slow_time_product`; the real cube
the Hann range FFT of the decoded ADC), on the same absolute scale (|v| max
9.5e4 both; means 1.6e3 real, 2.2e3 rendered).

* **The rendered person is right.** The moving-Doppler energy of the
  rendered cube centres on the GT pelvis in every frame checked (x 5.0,
  y 0.0 +/- 0.2, z 0.8 m, as the real's), at 0.75-0.89x the real energy,
  spatial correlation 0.63-0.76 with the real tensor.
* **The rendered static background has the wrong range slope.** Zero-Doppler
  energy rendered / real by 1 m range band, one mid-sequence frame each,
  five train + five test sequences: 1-2 m 0.02-0.48, 2-3 m 0.4-0.8, 3-4 m
  0.9-1.7, 6-7 m 1.6-3.7, 7-8 m 1.7-4.3, 8-9 m 1.3-3.7 (total static 1.07-1.89x,
  moving 0.75-0.89x) -- too weak near, 2-4x too strong far, on 10 of 10.
  The background is the fitted material model on the scaffold point cloud
  (`seq<k>_background_hfov40`, fit to `_static_ra.npy`).
* Under the released absolute normalisation ((|v| - 1e5) / 8.9e6, clipped at
  0) that far excess floods the static channel of the DZYX tensor: 3.4-4.8x
  the real's energy there, 36k voxels > 1 against 5-8k, the tensor's argmax
  on a fixed far spot (x 6.4-6.8, y 1.6-2.5, z 0) instead of the person.
  HuPR's released preprocessing removes clutter over the loops and
  min-max / z-scores per frame, so HuPR never sees it (its R -> S is 22 AP,
  section 4.1).
* **Mechanism, by intervention** (the real-trained seed-4 model, 239 frames
  a sequence, only Doppler bins 30-34 of 64 manipulated; MRPE in cm):

| input | seq 159 | seq 108 | seq 168 |
|---|---|---|---|
| real | 5.4 | 10.7 | 11.6 |
| rendered as-is | 43.1 | 652.8 | 405.5 |
| rendered, static bins replaced by the real cube's | **6.2** | **9.7** | **10.2** |
| rendered, static bins zeroed | 9.2 | 313.6 | 36.6 |
| real moving bins with the rendered static bins | 32.0 | 479.4 | 266.1 |
| rendered scaled to the real's total energy | 107.6 | 528.1 | 535.2 |

Swapping in the real static bins alone restores R -> R accuracy; the
rendered person with the real background is read correctly. The R -> S
number of 3.2 is a renderer background fidelity defect that RT-Pose's
absolute normalisation exposes -- not an adapter, label, geometry or scale
bug. Options recorded (none taken yet): run the S and R+S arms as they are
and report the defect; fix the background's range falloff in the renderer
and re-render; or a domain-symmetric static-clutter removal in the adapter
(MATLAB's `DopplerProcClutterRemove`, off in the released chain) at a new
protocol hash with the real arm rerun.

## 4. The rendered and mixed arms

### 4.1 HuPR: the full train x test grid (mmsd-h100, five seeds, protocol `38ea71937b32`, 2026-09-16)

`runner run --dataset hupr --arms rendered mixed --gpus 0 1 2 3 4 --workers 8`
(one training per card, ten fits in two rounds of five), launched 01:38 UTC,
all ten `ok` at 15:12 UTC (13 h 34 min); no crash, no rerun. Every arm at the
real arm's 579 steps x 200 epochs; seed k of every arm shares its
initialisation and batch order with seed k of the real arm (1.4). Model
selection as released, best val AP on the arm's own val domain (rendered val
for S, real val for R+S; checkpoint epochs S 142 / 56 / 198 / 60 / 190,
R+S 179 / 185 / 195 / 160 / 197). Official test split, 12,600 frames, the
released COCOeval; mean +/- sample std (ddof 1) over five seeds;
results/hupr/<train>__<test>/seed<k>/metrics.json.

| train -> test | AP | AP50 | AP75 | AR | PCKh@0.2 | per seed AP |
|---|---|---|---|---|---|---|
| R -> R (2.3, the reference) | **65.94 +/- 1.67** | 96.97 +/- 0.20 | 78.81 +/- 3.18 | 75.29 +/- 1.31 | 44.14 +/- 1.85 | 66.87 / 63.17 / 66.76 / 65.59 / 67.28 |
| R -> S | 21.98 +/- 3.83 | 63.61 +/- 6.25 | 7.86 +/- 2.88 | 33.89 +/- 3.50 | 12.33 +/- 2.32 | 24.06 / 16.32 / 21.33 / 21.58 / 26.63 |
| S -> R (TSTR) | **15.81 +/- 2.00** | 51.24 +/- 5.44 | 4.54 +/- 1.07 | 28.11 +/- 2.02 | 10.20 +/- 1.15 | 16.56 / 16.07 / 18.04 / 15.76 / 12.58 |
| S -> S | **53.86 +/- 1.43** | 95.39 +/- 1.35 | 55.96 +/- 2.17 | 63.30 +/- 1.62 | 37.95 +/- 1.15 | 54.36 / 52.00 / 55.09 / 52.72 / 55.15 |
| R+S -> R (Table 5's augmentation arm) | **60.83 +/- 1.24** | 96.82 +/- 0.20 | 70.15 +/- 2.84 | 70.76 +/- 0.96 | 40.16 +/- 1.09 | 60.17 / 61.51 / 61.92 / 61.58 / 58.97 |
| R+S -> S | 57.69 +/- 2.12 | 96.44 +/- 1.34 | 65.18 +/- 4.10 | 67.16 +/- 1.59 | 39.69 +/- 2.75 | 59.35 / 58.95 / 55.35 / 59.39 / 55.39 |
| R -> real/rendered 50:50 mixed test | 44.81 +/- 2.30 | 82.01 +/- 2.65 | 44.69 +/- 3.39 | 52.32 +/- 2.21 | 26.89 +/- 2.21 | 45.46 / 41.23 / 45.37 / 44.47 / 47.53 |

Per-keypoint AP (mean +/- std), the six cells:

| keypoint | R -> R | R -> S | S -> R | S -> S | R+S -> R | R+S -> S |
|---|---|---|---|---|---|---|
| Head | 81.7 +/- 1.1 | 48.9 +/- 5.8 | 34.6 +/- 4.3 | 77.3 +/- 1.9 | 78.6 +/- 1.7 | 79.8 +/- 3.0 |
| Neck | 83.5 +/- 1.8 | 47.8 +/- 5.8 | 36.4 +/- 4.1 | 79.7 +/- 2.1 | 80.5 +/- 1.7 | 81.2 +/- 3.0 |
| L / R shoulder | 74.7 / 77.0 | 36.7 / 27.9 | 22.4 / 20.8 | 68.1 / 70.8 | 71.4 / 72.4 | 71.9 / 72.3 |
| L / R elbow | 41.5 / 50.9 | 11.7 / 4.4 | 5.9 / 2.8 | 19.7 / 28.2 | 34.3 / 43.8 | 26.4 / 33.4 |
| L / R wrist | 22.5 / 25.6 | 4.5 / 1.1 | 2.0 / 0.7 | 8.4 / 13.7 | 16.8 / 19.1 | 11.7 / 15.2 |
| L / R hip | 90.7 / 90.3 | 52.9 / 50.4 | 49.0 / 45.2 | 90.0 / 89.6 | 89.0 / 88.6 | 90.0 / 89.8 |
| L / R knee | 84.1 / 84.8 | 40.2 / 40.3 | 38.3 / 33.3 | 83.4 / 83.0 | 81.9 / 82.2 | 84.5 / 84.5 |
| L / R ankle | 77.5 / 73.5 | 50.5 / 25.5 | 29.6 / 44.9 | 74.6 / 73.7 | 75.4 / 70.3 | 78.6 / 70.4 |

Reading (the paper's framing in Table 4's caption: the rendered test column
measures information content against the labels that drove the render, not
unbiased accuracy):

* **S -> S 53.9 against R -> R 65.9**: the renders carry a learnable pose
  signal 12 AP under the recording's, and the deficit is entirely in the arms
  -- torso, hips, knees, ankles within 0-4 AP of R -> R (hips 90 / 90, knees
  83 / 84), elbows 20-28 vs 42-51, wrists 8-14 vs 22-26. The rendered arm
  motion is where the fidelity is short.
* **Cross-domain in either direction collapses** (R -> S 22.0, S -> R 15.8):
  a model trained on one domain reads the other's hips at 45-53 AP and
  little else. The two domains are separable pictures to the model even
  though each carries the pose. S -> R below R -> S on every keypoint but
  the right ankle.
* **R+S -> R 60.8 against R -> R 65.9 (-5.1, 3 std)**: at the real arm's
  optimizer-step count, replacing half of every batch with renders costs 5
  AP on the real test -- the renders do not substitute for real frames at a
  fixed budget. The R+S model reads both domains (R+S -> S 57.7 > S -> S
  53.9: the real half of the batch improves the rendered-test number too).
  The low-shot curve (whether renders help when real data is the limit) is
  not run (0.).
* The R -> R model on the 50:50 mixed test set (44.8) is within 1 AP of the
  mean of its R -> R and R -> S numbers (44.0): it does not behave
  differently on renders for having them beside real frames.

Training curves (val AP on the arm's own val split, epochs 1 / 25 / 50 /
100 / 150 / 200, best and its epoch): rendered s0 10.2 / 48.7 / 46.4 / 49.6 /
46.6 / 50.4 (53.6 @143), s1 8.2 / 50.0 / 44.9 / 50.2 / 50.4 / 50.0 (53.0 @57),
s2 13.4 / 49.1 / 51.6 / 53.3 / 54.3 / 53.5 (56.6 @199), s3 11.5 / 47.6 / 52.8 /
49.2 / 49.9 / 51.5 (53.6 @61), s4 8.1 / 43.5 / 48.8 / 46.2 / 48.9 / 51.3
(54.0 @191) -- the rendered arm plateaus by epoch 25 on rendered val (train
loss 0.061 -> 0.0255, the same floor as the real arm's 0.0248, so it fits
the renders as tightly and there is no more to learn from them); mixed s0
13.8 / 50.0 / 56.2 / 62.5 / 64.1 / 67.1 (68.2 @180), s1 4.7 / 51.5 / 59.5 /
65.0 / 66.4 / 64.6 (67.0 @186), s2 12.8 / 52.1 / 58.7 / 63.8 / 64.3 / 66.0
(67.7 @196), s3 6.0 / 46.8 / 55.5 / 62.0 / 64.0 / 65.1 (66.4 @161), s4 9.4 /
51.5 / 54.8 / 60.4 / 61.1 / 63.5 (65.8 @198) -- the mixed arm climbs like the
real arm (real val 70-73 at the end) but 4-6 AP under it throughout, still
rising at epoch 200 (it has seen half the real frames).

Wall time: 6.62-6.70 h a seed, 0.195-0.197 s/step (the real round's
0.197-0.201); the two evals 80-86 s each.

### 4.2 RT-Pose

Not launched (2026-09-16): the R -> S diagnosis of the rendered RT-Pose
cube (3.3) is to be decided on first.

## 5. Tables and macros

(pending)

## 6. RT-Pose under THE RECIPE (fidelity/static-seqs, 2026-09-21): the re-render of the split and the arms

`downstream/tools/rtpose_full_rerun.sh` -> `rtpose_rerender_retrain_node.sh` (29db2cd; the decoder fix a5a53d0 for the
evaluations): the 100 sequences of the split re-rendered under the trainer's defaults (pwl), the rendered / mixed HRRadarPose
arms retrained (five seeds, Train_gvhmr.json, seq 13 excluded; the real arm of 2026-09-20 kept), the full matrix. Products
`/data/mmsd/4DPS_products_static`, results `4DPS_static2/downstream/results_gvhmr`, tables pulled to
`mmSD/artifacts/rtpose_static_rerun/20260921_0418/tables`. MPJPE cm, mean (std) over five seeds; the previous round
(b7bc875 render, the dataset's labels, section 4.2's numbers) in brackets:

| cell | this round | previous |
|---|---|---|
| R -> R | 16.90 (0.64) | 16.07 (1.72) |
| R -> S | **19.62 (0.65)** | 83.54 (21.06) |
| S -> R | **24.73 (2.22)** | 69.46 (5.76) |
| S -> S | 15.31 (0.72) | 26.55 (7.12) |
| R+S -> R | 16.45 (0.24) | 14.35 (0.43) |
| R+S -> S | 15.58 (0.55) | 15.86 (1.01) |
| R / S / R+S -> two-person (real) | 58.66 / 64.04 / 51.85 | 54.87 / 89.59 / 40.18 |
| retention S->R / R->R | 0.68 | 0.23 |
| gaps realism / reverse / information | 9.4 / 2.7 / -1.6 | 42.9 / 67.5 / 10.5 |

Render fidelity over the 100 sequences vs the previous round (Hann suite, `products_manifest.csv`): RD power ratio 0.47 ->
0.58 (93 / 100 up), MDs 0.931 -> 0.951 (85 / 98), MDe 0.864 -> 0.922 (84 / 98), RD and RAc flat, Hann RA -0.04 on 99 / 100
(the untapered default: the trainer fits the consumers' view, STATIC_SEQS_NEXT 1d). The new meshes reproduce
Train_gvhmr.json exactly (drift 0.0 cm on every sequence). The labels differ from the previous round's (mesh vs dataset,
seq 13), so R -> R moved 16.07 -> 16.90; the cross-domain cells moved by 50-65 cm.

## 7. HuPR before its re-render: the inputs checked (2026-09-21, viz_hupr.py; mmSD/artifacts/hupr_{check,pair,geometry}_v1 / v2)

The downstream's labels are the released 2D annotations (hrnet_annot_*.json; `hupr_mscsa/dataset.py`); the registered GVHMR
mesh is only rendered, never a label. The geometry videos showed the mesh projecting 13-20 px (256-px frame) off those labels
and its feet 0.17 m below the scaffold's floor; the person stage's radar depth scale is 0.70-0.73 on every test sequence and
the projected mesh fits the GT with a scale of 0.77: the camera assumed for HuPR (40 deg HFOV, f = 703 px; GVHMR itself runs
at its default 724 px) is ~1.4x too long in focal. The person's 3D placement is invariant to it (the ray (u - c) / f and the
radar's depth scale cancel); the monocular scaffold is not (Depth-Anything's metric depth back-projected with the wrong f is
compressed 1.4x toward the optical axis: the floor 0.3 m too high, the walls too narrow), and the audit projection is not.
**H1 (fe434f7): `Dataset.camera_hfov_deg`, HuPR = 52.8 deg (724 x 0.712 = 515 px), read by the scaffold and the person stage;
RT-Pose keeps its calibrated intrinsics.** Rebuilt on the 21 test sequences: floor -0.85 -> -1.16 m, walls / ceiling 1.37x out,
the feet from 0.17 m below to ~0.1 m above the floor, the projection scale 0.73-0.81 -> 1.00-1.08 with a 4-10 px residual after a
similarity fit; a uniform ~10 px vertical offset (the mesh below the GT, ~0.12 m at 3 m) remains = H2 (a camera <-> radar
pitch / height seed from the vertical board, as RT-Pose's LiDAR seed) -- to be measured after the re-render, not before it.
The dataset differences inventory: pre-processing only (scene source, camera model, GVHMR focal, extrinsic, radars, labels);
the training stage is one code path apart from the level calibration's view (a consumer property) and `--frame-batch`.
**H1b (6ff6139, b9497fe): `--image-registration gt`** -- on a dataset with 2D keypoints in the RGB frame (`Dataset.image_keypoints`,
HuPR) the body's ray comes from the released keypoints and its range from the radar; GVHMR supplies shape and pose only; the
focal the keypoints imply at the radar's range (513-540 px over the 21, median 525) sets HUPR_CAMERA_HFOV_DEG = 52.0. The mesh
now projects onto the GT at 3.2-7.1 px (median 3.8; v1 17.2), below the real model's own 5.4-11.5 on real data
(mmSD/artifacts/hupr_geometry_v3). Remaining: the feet ~0.15 m above the Depth-Anything floor -- GVHMR's body height (1.80 m,
uniform) against the monocular floor depth, inseparable without an independent scale; carried as a known item. RT-Pose: no-op.
**H3 (7fb0c55): `Dataset.depth_scale_mode`** -- "frame" (RT-Pose, unchanged: `radar_depth_scales`' per-frame ratio) or
"sequence" (HuPR: the median ratio over confident frames applied to every frame). Measured on seq 38 under "frame": the
8-channel board's per-frame range of a slow / seated person put the registered range at 3.1 -> 5.6 m on 216 of 600 frames
(per-frame scale 0.68-1.23, the 75 flagged frames still applied), the person projecting at half size (the flicker in
hupr_geometry_v4/seq38). GVHMR's depth error is the camera's focal, a constant of the sequence; the 2D-keypoint ray carries
the per-frame position. The mode and the applied per-frame scales are in the metadata. Person stage re-run on the 21 test
sequences (26-30 s each) and checked per sequence (mmSD/artifacts/hupr_geometry_v5/README.md holds the full table;
"before" = the per-frame placement recomputed from the recorded series):

| seq | scale | reg. range median (m) | max dev (m) now / before | skel. height px now / before | max jump px now / before | mesh-vs-GT px median (p90) | feet-floor m, peak / p2 |
|---|---|---|---|---|---|---|---|
| 15 | 0.702 | 2.74 | 0.09 / 1.06 | 140-164 / 104-162 | 6.6 / 23.5 | 4.5 (5.5) | +0.19 / +0.27 |
| 16 | 0.708 | 2.92 | 0.47 / 0.45 | 121-182 / 120-183 | 8.3 / 8.4 | 4.4 (5.4) | +0.19 / +0.28 |
| 17 | 0.722 | 2.84 | 0.39 / 0.41 | 121-170 / 121-169 | 12.6 / 12.4 | 3.7 (4.2) | +0.17 / +0.30 |
| 244 | 0.730 | 3.26 | 0.53 / 1.50 | 113-142 / 82-143 | 3.6 / 34.3 | 2.9 (3.4) | +0.12 / +0.25 |
| 245 | 0.724 | 3.10 | 0.58 / 0.56 | 115-157 / 117-159 | 5.2 / 5.3 | 3.3 (3.9) | +0.02 / +0.21 |
| 246 | 0.711 | 3.16 | 0.56 / 0.55 | 119-153 / 119-155 | 4.7 / 4.8 | 3.4 (3.9) | +0.17 / +0.22 |
| 247 | 0.705 | 3.04 | 0.70 / 0.70 | 113-170 / 112-169 | 7.7 / 7.7 | 3.8 (4.7) | +0.14 / +0.24 |
| 248 | 0.711 | 3.02 | 0.69 / 0.68 | 110-170 / 109-171 | 7.9 / 7.9 | 3.7 (4.5) | +0.17 / +0.24 |
| 249 | 0.717 | 3.07 | 0.53 / 0.53 | 117-197 / 115-197 | 13.5 / 13.3 | 3.3 (3.8) | +0.11 / +0.24 |
| 250 | 0.699 | 3.05 | 0.56 / 0.59 | 117-202 / 116-204 | 16.6 / 16.4 | 4.6 (5.2) | +0.17 / +0.23 |
| 251 | 0.726 | 3.05 | 0.06 / 0.04 | 127-142 / 127-144 | 5.5 / 5.4 | 3.2 (3.5) | +0.09 / +0.22 |
| 252 | 0.702 | 3.08 | 0.03 / 0.04 | 128-139 / 127-139 | 4.5 / 4.5 | 3.9 (4.1) | +0.08 / +0.21 |
| 253 | 0.728 | 3.12 | 0.47 / 0.45 | 115-154 / 114-152 | 4.7 / 4.7 | 3.1 (3.6) | +0.11 / +0.24 |
| 254 | 0.711 | 3.15 | 0.70 / 0.68 | 117-168 / 119-168 | 6.8 / 6.8 | 3.8 (4.4) | +0.09 / +0.22 |
| 255 | 0.716 | 3.09 | 0.86 / 0.86 | 108-180 / 109-180 | 6.8 / 6.8 | 3.5 (4.3) | +0.04 / +0.22 |
| 256 | 0.707 | 3.02 | 0.82 / 0.79 | 115-186 / 116-185 | 11.1 / 11.0 | 4.3 (5.3) | +0.15 / +0.22 |
| 38 | 0.709 | 3.19 | 0.53 / 2.46 | 119-158 / 69-162 | 6.1 / 39.9 | 4.2 (5.1) | +0.17 / +0.22 |
| 39 | 0.720 | 3.13 | 0.52 / 0.49 | 118-169 / 118-163 | 8.5 / 8.6 | 3.3 (4.5) | +0.13 / +0.23 |
| 40 | 0.710 | 3.08 | 0.08 / 2.39 | 125-130 / 72-132 | 1.0 / 41.6 | 3.9 (4.7) | +0.13 / +0.21 |
| 41 | 0.714 | 3.09 | 0.58 / 1.67 | 116-158 / 84-158 | 11.2 / 22.8 | 3.8 (4.4) | +0.18 / +0.22 |
| 42 | 0.712 | 3.07 | 0.55 / 1.98 | 118-157 / 78-159 | 5.9 / 35.6 | 3.9 (6.0) | +0.16 / +0.22 |

The six sequences that flickered (15, 244, 38, 40, 41, 42: max jump 23-42 px, range deviation 1.1-2.5 m) are at 1.0-11.2 px
and 0.08-0.58 m; the other 15 are unchanged to the cm (their per-frame ratios sat at the median). The remaining range
deviation (0.03-0.86 m, above 0.3 m on 17 sequences) is the person's motion, not the placement: the radar's confident-frame
range agrees with the placed range to 0.02-0.05 m (median absolute residual), their deviations correlate (+0.5..+0.98 where
the person moves), and the released GT skeleton's own height changes with it (seq 255: 119-196 px; the static 251 / 252 / 40:
139-143 px, 0.06-0.15 m). The per-frame GT / mesh image scale is 0.97-1.03 (p5-p95) on 18 sequences; on 38 / 40 / 42 a
tail of 34-43 frames (6-7 %) with a 0.4 scale and a 37-40 px error at the same skeleton height is a left / right joint swap
between the annotation and the mesh. Over the 21: mesh-vs-GT median 3.8 px (2.9-4.6), feet-floor +0.14 m against the
scaffold's z-histogram floor (+0.02..+0.19; the 2nd-percentile floor of viz_geometry.py reads ~7 cm lower, Depth-Anything's
floor noise), the known body-height item. Videos: mmSD/artifacts/hupr_geometry_v5. RT-Pose: no-op (mode "frame").

## 8. HuPR under the fixed person inputs (fidelity/static-seqs, 2026-09-21/22): the re-render of the split and the arms

`downstream/tools/hupr_full_rerun.sh` -> `hupr_rerender_retrain_node.sh` (82a7a70; the render code = 7fb0c55, H1 / H1b / H3 +
the constant rig yaw, section 7): the 235 sequences of the split (train 193 / val 21 / test 21) re-rendered under the
trainer's defaults from the H1 scaffolds (built on the workstation) and the reference workspaces, 8 GPUs, 390 s a sequence,
0 failures (phase 1 18:27 -> 21:36 ET); the rendered caches re-materialised (470 / 470 ok); the blindness gate passed; the
real (R) arm's five seeds copied from the previous round (the labels are the released 2D annotations, protocol hash
38ea71937b32 unchanged), the rendered (S) and mixed (R+S) arms retrained for the five seeds (from 21:53 ET, one training per
H100, ~7 h each, two waves), then the matrix. Products `/data/mmsd/4DPS_products_static/hupr` (markers .rerender git
82a7a70; the metrics.json git_sha of sequences rendered after 840b0e3 -- a driver-only commit -- says 840b0e3), caches
`/data/mmsd/4DPS_downstream_gvhmr/hupr`, results `4DPS_static2/downstream/results_gvhmr/hupr` (tables at `results_gvhmr/tables`).

**Render fidelity over the 235 sequences** vs the previous round (b7bc875 render, `/data/mmsd/4DPS_products/hupr/manifest.csv`;
Hann suite as the manifest records it; medians over sequences, "up" = the sequence improved):

| board | metric | previous | this round | up on n / 235 | mean change |
|---|---|---|---|---|---|
| horizontal | RA | 0.953 | 0.962 | 215 | +0.008 |
| horizontal | RD occupied | 0.875 | 0.889 | 98 | +0.010 |
| horizontal | MD spectrogram | 0.987 | 0.992 | 181 | +0.013 |
| horizontal | MD energy | 0.973 | 0.985 | 182 | +0.023 |
| vertical | RA | 0.920 | 0.931 | 198 | +0.014 |
| vertical | RD occupied | 0.766 | 0.786 | 77 | -0.002 |
| vertical | MD spectrogram | 0.978 | 0.983 | 138 | +0.013 |
| vertical | MD energy | 0.970 | 0.971 | 112 | +0.016 |

On the 21 test sequences: horizontal RA 0.956 -> 0.963 (18 / 21 up), MD spectrogram 0.987 -> 0.994 (17 / 21), MD energy
0.982 -> 0.991 (19 / 21); vertical RA 0.921 -> 0.936 (20 / 21); RD occupied flat on both boards (9 / 21 and 6 / 21).

**The matrix** (COCO AP, mean (std) over five seeds; the previous round, section 4's numbers, in brackets; ALL PHASES DONE
2026-09-22 11:24 ET, 15 / 15 cells ok; tables pulled to `mmSD/artifacts/hupr_static_rerun/20260921_2226/tables`):

| cell | this round | previous |
|---|---|---|
| R -> R | 65.94 (1.67) | 65.94 |
| R -> S | **50.39 (0.96)** | 36.93 |
| S -> R | **38.32 (2.63)** | 40.05 |
| S -> S | **59.18 (1.52)** | 41.33 |
| R+S -> R | **61.71 (1.03)** | 56.61 |
| R+S -> S | **60.63 (0.96)** | 43.28 |
| R -> real mixed (two-person) | 56.58 (1.18) | - |
| retention S->R / R->R | 0.58 | 0.61 |
| gaps realism / reverse / information (AP) | 20.9 / 15.5 / 6.8 | 29.0 / 25.9 / 9.3 |

The R arm is the previous round's (the released 2D labels never changed), so R -> R is the same 65.94 and every move is the
render's. The rendered test set became much easier to read for a real-trained model (R -> S 36.93 -> 50.39, every seed
+12.5..+14.3) and much better to train on for the rendered test itself (S -> S 41.33 -> 59.18, R+S -> S 43.28 -> 60.63);
mixing now costs 4.2 AP on the real test instead of 9.3 (R+S -> R 56.61 -> 61.71). S -> R is the one cell that did not move
with the others: 40.05 -> 38.32, 1.7 AP down and the widest spread of the six (35.1-41.2 over the seeds), so the render is
closer to the recording in the model's own domain while the model trained on it alone still transfers to the real test no
better than before -- a sim-to-real residual to attribute (the realism gap 20.9 AP against RT-Pose's 9.4 cm MPJPE
equivalent; section 6), not a regression of the inputs.
