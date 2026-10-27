# The full sweep: 4DPS fitted to every usable HuPR and RT-Pose sequence

Status: 2026-09-14 19:20, complete: **336 of 336 usable sequences delivered**
(101 RT-Pose, 235 HuPR). The sweep of docs/FULL_SWEEP_PLAN.md ran 2026-09-13
22:38 to 2026-09-14 16:29 on two RTX 4090s, one sequence each, `main` at
`e5f30d4`; the reruns of the ten failures at `3e5f910`, 17:22-17:51. Products in
`/home/adnan/Documents/Data/4DPS_products/<dataset>/seq<S>/`, one line per
attempt in each `manifest.csv`. Every number below is read from the
sequences' `metrics.json` (the evaluation suite of docs/METRICS.md: linear
amplitude, never dB; the motion subset = frames with pose speed >= 0.15 m/s,
pre-registered). "Median (IQR)" over sequences, a HuPR sequence counting
once per board.

## 1. Coverage and cost

| dataset | usable | delivered | wall a sequence | per-sequence products |
|---|---|---|---|---|
| RT-Pose | 101 (of 106 folders: `0` empty, `107`/`193` no radar, `155` no annotation, `169` nobody in the camera's view) | **101** after the reruns (98 in the sweep, 123/157/174 on rerun) | 552 s median (520-594; 299 frames) | 3.76 GB cube + 92 MB checkpoint + 22 MB mesh |
| HuPR | 235 annotated | **235** after the reruns (229 in the sweep, 52/76/154/168/188/228 on rerun) | 313 s median (293-327; 600 frames, two boards) | 2 x 0.47 GB cubes + 185 MB checkpoint + 45 MB mesh |

18 h of wall for the sweep, 30 min for the reruns; 626 GB delivered (RT-Pose
366, HuPR 260); 864 GB free after. Throughput 22-24 sequences an hour.

The ten failures and what they were (all deterministic, all fixed in
`3e5f910`, none a fit that went wrong):

| sequences | cause | fix |
|---|---|---|
| HuPR 52, 154 | the person fills a sampled scaffold frame (the padded annotation bbox masks 100% of the pixels): an empty scene cloud | `reconstruction.usable_samples`: that frame swapped for the nearest annotated one showing the scene (no other sequence's frames change; the next most masked is 0.953) |
| HuPR 76, 168, 188, 228; RT-Pose 157, 174 | a still person: 1-5 frames above the pose-speed threshold, fewer than the three the motion-subset correlations and SSIM need | the motion-subset metrics recorded as undefined (`NaN`, `frames_scored`, a note) instead of failing the sequence; the products are delivered |
| RT-Pose 123 | the LiDAR scaffold's neighbour search took the whole card (a 485k-point merged cloud, 23.6 GB) and starved GVHMR's model load | the search bounded at 32M cells a chunk (0.57 GB, 2.2 s; normals identical except on degenerate neighbourhoods, 25 of 484,555); GVHMR's loads retry on OOM |
| RT-Pose 169 | nobody in the camera's view for the whole recording | unusable; excluded |

## 2. Fitting performance, HuPR (229 sequences x 2 boards in the sweep; all 235 in 2.1)

| metric | median | IQR | min | max | >= threshold |
|---|---|---|---|---|---|
| range-azimuth Pearson | **0.926** | 0.909-0.948 | 0.818 | 0.975 | 100% >= 0.8 |
| range-azimuth, clutter-removed (the attributable part) | 0.571 | 0.354-0.658 | -0.003 | 0.824 | |
| range-azimuth SSIM / NRMSE / PSNR | 0.733 / 0.372 / 31.8 dB* | | | | |
| range-Doppler occupied Pearson (CFAR Mask A, frames with >= 8 cells) | **0.762** | 0.643-0.851 | 0.026 | 0.927 | 88% >= 0.5 |
| micro-Doppler spectrogram Pearson | **0.980** | 0.969-0.985 | 0.238 | 0.989 | 98% >= 0.8 |
| micro-Doppler energy Pearson | **0.957** | 0.937-0.974 | 0.388 | 0.997 | 98% >= 0.8 |
| micro-Doppler centroid / spread Pearson | 0.968 / 0.889 | | | | |
| micro-Doppler clDice (topology) | 0.822 | 0.766-0.877 | 0.202 | 1.000 | |
| micro-Doppler SSIM | 0.977 | 0.965-0.983 | | | |

\*PSNR is the one logarithm the suite allows: a log of a scalar ratio after
the error is computed on linear amplitude.

By board: horizontal RA 0.948, RD 0.844, MD spectrogram 0.984, MD energy
0.947; vertical RA 0.909, RD 0.719, MD spectrogram 0.978, MD energy 0.969.
The vertical board is consistently the harder one on RA and RD (the
elevation view of a monocular scene), not on micro-Doppler.

Motion subset: median 269 of 600 frames scored (7-594). RD occupied Pearson
correlates with the number of motion frames (Pearson +0.53 across
products): sequences where the person barely moves have thin Doppler
occupancy and a noisier statistic, not a worse fit.

Registration: GVHMR's depth scale median 0.719 (IQR 0.706-0.742: HuPR's
GVHMR runs ~30% long on every sequence), per-sequence IQR width median
0.025, max 0.048 -- tight on all 229; the per-frame radar registration
behaves the same on every HuPR sequence.

Lowest products: RA -- 27v 0.818, 225v 0.822, 8v 0.851; MD spectrogram --
21 (0.24 v / 0.37 h, a 99-frame motion subset), 60 (0.72 / 0.74), 58v 0.73.
Sequence 21 is the one HuPR outlier worth a look (both boards low on
micro-Doppler with an ordinary motion count).

### 2.1 With the six reruns (235 sequences, 470 products)

RA 0.926 (0.909-0.948), RA clutter-removed 0.568, RD occupied 0.762
(0.643-0.847), MD spectrogram 0.980 (0.969-0.985), MD energy 0.957
(0.937-0.975), MD centroid 0.969, clDice 0.821 -- the medians move by at
most 0.004. The six reruns: 52 RA 0.922 h / 0.888 v, 154 0.943 h with 524 motion
frames; 76/168/188/228 are the still recordings (3-6 motion frames), their
RA 0.92-0.96 and their motion-subset figures on those few frames.

## 3. Fitting performance, RT-Pose (98 sequences in the sweep, one cascade radar; all 101 in 3.1)

| metric | median | IQR | min | max | >= threshold |
|---|---|---|---|---|---|
| range-azimuth Pearson | **0.852** | 0.809-0.884 | 0.669 | 0.930 | 77% >= 0.8 |
| range-azimuth, clutter-removed | 0.279 | 0.164-0.430 | -0.001 | 0.586 | |
| range-azimuth SSIM / NRMSE / PSNR | 0.829 / 0.477 / 32.5 dB | | | | |
| range-Doppler occupied Pearson | **0.600** | 0.478-0.667 | 0.064 | 0.818 | 71% >= 0.5 |
| micro-Doppler spectrogram Pearson | **0.931** | 0.903-0.949 | 0.667 | 0.969 | 99% >= 0.8 |
| micro-Doppler energy Pearson | **0.876** | 0.815-0.923 | 0.535 | 0.971 | 79% >= 0.8 |
| micro-Doppler centroid / spread Pearson | 0.919 / 0.675 | | | | |
| micro-Doppler clDice | 0.735 | 0.685-0.773 | 0.509 | 0.919 | |
| micro-Doppler SSIM | 0.906 | 0.859-0.927 | | | |

Motion subset: median 230 of 299 frames (12-299). Registration: depth scale
median 1.106 (IQR 1.080-1.171: GVHMR ~10% short at RT-Pose's focal
length), confident frames median 99%.

**35 of the 102 RT-Pose folders are two-person recordings** (the rooftop
set: 100-105, 117-140, 170-175; YOLO on three frames a sequence, the
annotation carries one pose a frame and does not say). The chain is
single-person -- GVHMR keeps the longest/largest track and the radar's
per-frame range registration follows whichever person moves more -- so on
those the person is registered against the wrong mover part of the time
(registration IQR width 0.2-1.3 against 0.06 on the one-person
sequences; radar-vs-GVHMR range correlation ~0 against 0.99). Split:

| RT-Pose | n | RA | RA clutter-removed | RD occupied | MD spectrogram | MD energy | MD centroid | clDice |
|---|---|---|---|---|---|---|---|---|
| one person | 64 | **0.864** (0.814-0.901) | 0.380 | **0.636** | **0.940** | **0.904** | 0.970 | 0.749 |
| two people | 34 | 0.830 (0.800-0.857) | 0.196 | 0.473 | 0.904 | 0.820 | 0.833 | 0.693 |

The two-person numbers are what a single-person model gets when the
second mover is unmodelled -- reportable as such, not as the method's
fidelity. The paper's RT-Pose table should be the one-person 66 (section 3.1),
with the two-person set as a stated limitation.

Lowest products (all two-person unless noted): RA -- 175 0.669, 125/126
0.712, 15 0.733 (one person), 18 0.740; MD spectrogram -- 108 0.667, 101
0.814, 120/116 0.817.

### 3.1 With the three reruns (101 sequences)

| RT-Pose | n | RA | RA clutter-removed | RD occupied | MD spectrogram | MD energy | MD centroid | clDice |
|---|---|---|---|---|---|---|---|---|
| all | 101 | 0.853 (0.808-0.885) | 0.279 | 0.592 | 0.930 | 0.876 | 0.919 | 0.735 |
| one person | 66 | **0.866** (0.816-0.902) | 0.378 | **0.632** | **0.940** | **0.904** | 0.971 | 0.751 |
| two people | 35 | 0.830 (0.793-0.856) | 0.201 | 0.469 | 0.903 | 0.819 | 0.840 | 0.696 |

123 RA 0.776, 174 RA 0.883 (5 motion frames), 157 RA on a one-frame motion
subset (its motion metrics undefined, the one such RT-Pose sequence).

## 4. HuPR against RT-Pose

HuPR fits better on every metric (RA 0.926 vs 0.852, RD 0.76 vs 0.60, MD
spectrogram 0.98 vs 0.93). The differences that plausibly explain it, in
the order I would test them: the cascade's 144-channel training sub-array
against 8 per board (a far sharper azimuth response for the same point
model to match); the outdoor rooftop scenes with two people (section 3);
the LiDAR scene's sparsity (Livox, 0.5 m normal radius) against the dense
monocular depth. The clutter-removed RA -- the part of the range-azimuth
agreement the model has to earn -- is 0.57 on HuPR and 0.38 on one-person
RT-Pose.

## 5. What to check next

1. HuPR 21 (both boards' micro-Doppler at 0.24/0.37): a registration or
   tracking fault, or a genuinely odd recording.
2. The two-person RT-Pose set: either a second GVHMR track and a second
   person in the render, or report them as out of scope.
3. A shuffle control on the sweep's distributions (the permutation null
   of docs/METRICS.md was run on three sequences; the write-up should
   quote it against these medians).
