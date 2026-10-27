# Fidelity plan: the person's range, the scene's brightness, the room's echoes

Status: 2.1 and 2.2(c) done and gated (section 4), 2026-09-13, branch `hann-windowing`. Written after the six
end-to-end runs in `outputs/e2e_*` (three RT-Pose, three HuPR; 400 epochs,
`--intermediates`). Every number below is from those runs unless it says
otherwise; the estimator is named beside it.

## 1. What is wrong, measured

### 1.1 Seq 153's blank measured micro-Doppler is a person-placement error

The spectrogram stage reads the measured cube in a +/-10-bin range window
centred on the **GVHMR person cloud's own p25 range** per frame
(`media.person_range_bins`, written as `cascade_person_range_bin.npy`).
Against the radar's moving-energy range -- the range bin of maximum power
in the measured `motion_dra` cube (Doppler band x angle summed), the
estimator used throughout this section:

| sequence | median \|GVHMR bin - radar bin\| | frames off by > 10 bins | the bad stretches |
|---|---:|---:|---|
| 153 | 9 | **41 %** | frames 95-150: GVHMR 146 -> 64 against radar 157 -> 80 (+11..+16 bins); frames 5-25: GVHMR 89 -> 161 against radar 88 -> 120 (up to -41: a GVHMR track failure) |
| 183 | 4 | 5 % | within +/-5 throughout |
| 186 | 4 | 14 % | one stretch at -12 (frames ~180-220) |

The person is placed by one sequence-wide depth scale from
`reconstruction.radar_depth_scale` (153: 1.143, IQR 1.11-1.17) and one
frame-0 range translation (`calibration frame1-person-range`, 153: -0.55 m).
A per-frame error of 25-45 cm (a 3.5 cm bin) cannot be absorbed by either,
so on 153 the measured window misses the person while the rendered person
-- placed at the same wrong range -- fills it. It is the cause of 153's
scores (spectrogram +0.54, energy +0.46, RD occupied +0.21, against
+0.93 / +0.88-0.93 / +0.34-0.39 on 183 and 186), and it puts the rendered
person at the wrong range in its RA and RD videos. Nothing in the trainer
or renderer is involved.

### 1.2 The RA maps: the static scene, not the person, and not the speed-ups

The objective is unchanged and reads RA twice:
`total = L2(loop-0 |RA|) + L2(the 63 range-azimuth Doppler planes) + 1.0 x L2(micro-Doppler marginal)`,
each normalised by its target's mean square (the history's `ra_l2` is the
first two together). The speed-ups changed no term, no physics and no
supervision: each was gated at the fp32 floor step by step (the first step
to seven figures; on HuPR the sorted route matches the old one on four
consecutive steps' losses and gradient norms to six or seven figures), and
the metrics agree -- seq 183's RA Pearson is 0.424 / 0.423 before the batch
(exact planes / 7 taps) and 0.423-0.425 on the four runs since; HuPR's RA
Pearson is 0.81-0.87 against the 0.73-0.87 (mean 0.80) recorded in
`docs/METRICS.md` before any of it.

What the loop-0 |RA| actually shows (seq 183 frame 120, same colour scale,
`first_loop_ra_complex` products):

| region | measured energy | rendered (clean) energy | note |
|---|---:|---:|---|
| the wall band, range bins 126-138 | 7.8e12 (27 % of the map) | **8.6e13 (80 %)** | measured: a stripe across every angle (the specular wall's sidelobe smear, `docs/RTPOSE.md`); rendered: an angle-concentrated point 11x brighter |
| the room, bins 40-125 | 6.6e12 | 1.4e13 | the LiDAR + RGB scaffold renders as a fan at 2x |
| the coupling column, bins 0-8 | 1.2e13 (43 %) | 0 | the cascade has no antenna-coupling calibration (HuPR's is a seed) |
| the person | at the right range and angle on 183 and 186 | | |

Pearson over the map 0.45; over the wall band alone 0.74. The floor
injection is not it: rendered-with-floor 1.081e14 against clean 1.075e14.

HuPR (seq 256 frame 120): the person matches; the far wall (bins 145-160)
renders 2-3x too bright and broader in angle; per-frame Pearson 0.85.
The absolute gain is a *median* over the brightest decile
(`train_motion absolute-gain`), so a scene whose brightest structure is
one specular wall keeps that wall too bright and the rest about right.

### 1.3 The RD maps: multipath

The measured RD carries, behind the person at the same Doppler, a trail
of echoes at greater range (the person's return bouncing off the walls)
and other movers; the renderer is single-bounce and produces the person
alone. RD occupied Pearson: cascade 0.34-0.39, HuPR 0.78-0.90 (both as
before the session).

## 2. The plan

Three independent items, in order of leverage. Each has its own gate;
none touches the training loop's arithmetic, so the runtime work stands.

### 2.1 Per-frame range registration of the person (fixes 1.1; 153 outright)

**Where.** `reconstruction.run_lidar_scaffold`'s GVHMR conversion writes
`person_frames/frame_NNN.npz` (vertices scaled by one `k` about the camera
origin); `calibration frame1-person-range` then picks one translation on
frame 0. The radar measures the person's range directly in every frame it
is in; use it in every frame.

**What.**
1. In `radar_depth_scale`, keep the per-frame ratio
   `r_f = radar top-1 % RA centroid range / GVHMR centroid range` (already
   computed for the median) and return the series, not only its median.
   Extend it to every frame the person is detected in (the motion
   detection's frames), not the annotated frames only.
2. Smooth `r_f` over time (a Savitzky-Golay or a 1 s median: the true
   ratio drifts slowly with the person's distance; the frame-to-frame
   jitter is the centroid's) and clamp it to the sequence's IQR widened by
   a factor, so a bad frame cannot move the body by metres. Frames with no
   radar detection take the interpolated value.
3. Scale each frame's vertices by its own `r_f` about the camera origin
   (the same construction as now: pixels preserved, body larger and
   farther), record `depth_scale_per_frame` in `metadata.json` beside the
   median, and keep the frame-0 translation calibration as the residual
   it was designed for (it should come out near zero once the scale is
   per frame).
4. For the GVHMR track failure at 153's start (frames 5-25, the cloud 40
   bins beyond the radar): `r_f` handles it if the radar detection is
   trusted over the cloud; add a guard that flags frames where the
   clamped `r_f` saturates, and excludes them from the spectrogram score
   and the micro-Doppler supervision (`pose_audit` already carries a
   reliable-frame list -- extend it).

**Gate.** The table of 1.1 recomputed: median |GVHMR - radar| under 3
bins and no frame beyond 10 on all three sequences; then the 400-epoch
chain on 153, 183 and 186 (one per GPU, the third after): 153's
spectrogram and energy correlations should land in 183/186's band
(+0.93 / +0.88-0.93), and 183/186 must not move outside their own n=4
spread (spectrogram 0.927-0.937). HuPR is untouched (its person range
comes from its own calibration and is within 4 bins).

**Cost.** One to two days. Runs: three chains, ~16 min each.

### 2.2 The static scene's brightness (1.2)

Two separate causes, two separate fixes.

**(a) The specular wall (both datasets).** The wall is a planar specular
reflector: the measured return is a coherent glint whose sidelobes smear
across angle, and the scaffold renders it as a dense field of small
facets with the fitted material, which adds up to a point 11x too bright
on the cascade (2-3x on HuPR). Options, to be measured in this order:

1. *Per-structure gain in the material fit.* The absolute gain is one
   median over the brightest decile; let the background material trainer
   (`train_motion dual-radar`, the scene's `raw_materials` and normals
   under the velocity trainer's `background_groups`) see the loop-0 RA
   term on **mean-removed** magnitude as well as raw magnitude, so the
   wall's excess costs it something the median does not. Cheapest; may be
   enough on HuPR.
2. *A specular (mirror) lobe with a facet-size term.* The BSDF has KA and
   SPM lobes; a planar wall's coherent sum over facets is what makes the
   render a point. Add the facet coherence length as a per-point material
   parameter (the `l_c` slot exists in `reparameterize_torch`'s physics
   vector) and let the wall's facets decohere -- the sidelobe smear the
   measurement shows is a coherence-length effect, not a gain.
3. *If neither closes it:* a per-sequence, per-radar wall-band gain as an
   explicit calibration (fitted like `absolute-gain`, on the wall band
   only), recorded in `metrics.json`; honest and cheap, but a second
   knob.

**(b) The scaffold fan (cascade).** The LiDAR + RGB scaffold's floor and
ceiling points render as a 2x fan over the room. Two things to try:
weight the scaffold's diffuse points by their measured evidence (the
scene-visibility mask machinery, `scene-shadow-mask`, already carries a
per-point visibility; extend it with a per-point *measured-support*
weight from the static map), and drop the RGB-depth cloud where the
LiDAR has coverage (`--no-rgb-fusion` exists; measure the RA Pearson
with it).

**(c) The cascade's coupling column (bins 0-8, 43 % of the measured
loop-0 energy).** HuPR ships a seed (`seeds/calibration/antenna_coupling`,
additive and static per virtual channel; `docs/SEEDS.md` recommends
fitting it in-chain). Fit it for the cascade from the sequence's own
static frames: the mean over the sequence's off-person frames of the
measured range profiles at bins 0-8 per virtual channel, added to the
rendered profiles in product units the way HuPR's is
(`video.py --antenna-coupling-calibration`). One `calibration` command,
one chain stage before the render; it also removes the largest single
term from the RA residual.

**Gate.** For each step: seq 183's loop-0 RA Pearson over the map (0.45
now) and over the wall band (0.74), the wall-band energy ratio (11x),
and the fan's ratio (2x); then the 400-epoch chain on 183 and HuPR 256
-- the micro-Doppler metrics must not move (the person is not touched)
and RA Pearson must rise. HuPR's RA Pearson against `docs/METRICS.md`'s
matched/permuted table: the permuted maximum is 0.77, so a matched value
that rises while the permuted does not is the evidence.

**Cost.** (c) one day; (a1) one day; (b) one to two days; (a2) a week.

### 2.3 Multipath in RD (1.3)

The echo trail is the person's return reflected by a wall: a second
scatterer at the person's Doppler and at range `|radar -> wall -> person|`
with the wall's reflectivity. A first-order model is cheap once the wall
is known: for each wall plane (from the scaffold), mirror the person's
points through it and render the mirrored cloud with the wall's material
weight and the same velocities (the mirrored path's Doppler is the same
to first order). This is a second render call of the person per frame
per wall (one or two walls), inside the existing path budget (the sorted
route's cost is per path). Measure on seq 183 frame 120 first (the trail
is visible there); gate on RD occupied Pearson (0.39). This is the
speculative item and the last.

**Cost.** Two days to measure on one frame; more if it helps.

## 3. Order and what to expect

1. 2.1 first (a data fix; 153 is unusable until it is done, and it
   changes nothing for the other two sequences beyond a few bins).
2. 2.2(c) the coupling, then 2.2(a1), then 2.2(b): each a day, each
   measured on the loop-0 RA table above before a chain run.
3. 2.2(a2) only if the wall is still the residual after (a1).
4. 2.3 last, on one frame, before deciding.

Every 400-epoch run is ~16 min on the cascade and ~7.5 min on HuPR
(`docs/TRAINING_SPEED_PLAN_2.md`), so the gates are cheap: one sequence
per GPU, n=1 for a step's direction, n>=3 for a claim.

## 4. Done: 2.1 and 2.2(c), gated at 400 epochs

**2.1** `reconstruction.radar_depth_scales` (the only path; the sequence-wide
median is gone): per frame, the radar's moving-power range-profile peak
over the GVHMR centroid range, confidence-gated, median-filtered, polished
within unflagged runs (Savitzky-Golay, 51 frames, quadratic -- a median
filter's steps gave the cloud 0.2-0.4 m/s of spurious radial motion),
GVHMR track failures flagged and followed. The three RT-Pose person seeds
were rebuilt (the old ones kept as `*_v2clean_median`; the estimator is
3.6 s a sequence, the seed build 176 s). Person range against the radar's
moving-energy peak: median 2 / 1 / 1 bins on 153 / 183 / 186 (from 9 / 4 /
4), 1 / 4 / 2 % of frames beyond 10 bins (from 41 / 5 / 14 %). The frame-0
range sweep stays and now measures the residual (153: +0.15 m, the torso's
front against the cloud's centroid; before, -0.55 m to a wrong frame).

**2.2(c)** `calibration sequence-coupling` fits each radar's near-field
coupling from the sequence's own recording (cascade: 99.95 % coherent over
frames; HuPR's seed 99.99 %); the chain runs it in preprocessing for a
dataset without a seed and carries it to the treatment; the renderer adds
it as HuPR's, and the trainer adds it to the anchor plane after the
absolute gain (`--antenna-coupling-calibration`, both datasets); the gain
fit leaves the coupling bins out (`--exclude-near-bins 15`).

The gate (400 epochs, `--intermediates`, one run each; `e2e_*` before,
`fid_*` after):

| sequence | spectrogram | energy | centroid | RA Pearson | RA SSIM | RD occupied | train / val | wall |
|---|---|---|---|---|---|---|---|---|
| 153 | +0.537 -> **+0.930** | +0.458 -> **+0.926** | +0.835 -> +0.988 | 0.463 -> **0.915** | 0.656 -> 0.894 | 0.213 -> **0.512** | 1.324 / 1.368 -> 0.939 / 1.018 | 966 -> 976 s |
| 183 | +0.934 -> +0.938 | +0.875 -> +0.851 | +0.991 -> +0.995 | 0.425 -> **0.857** | 0.640 -> 0.804 | 0.388 -> 0.423 | 1.326 / 1.391 -> 1.310 / 1.395 | 938 -> 944 s |
| 186 | +0.937 -> +0.934 | +0.925 -> +0.877 | +0.989 -> +0.994 | 0.403 -> **0.846** | 0.637 -> 0.802 | 0.344 -> **0.631** | 1.445 / 1.527 -> 1.269 / 1.357 | 971 -> 952 s |
| HuPR 256 (H / V) | +0.972 / +0.980 -> +0.978 / +0.980 | +0.897 / +0.967 -> +0.921 / +0.970 | +0.957 / +0.926 -> +0.987 / +0.938 | 0.858 / 0.833 -> 0.858 / 0.833 | 0.619 / 0.615 -> 0.621 / 0.613 | 0.825 / 0.779 -> 0.829 / 0.781 | 0.815 / 0.954 -> 0.806 / 0.942 | 865 -> 455 s (the artifacts stage) |

Runtime: the chain is unchanged (the coupling fit is 3 s of preprocessing;
the anchor-plane add is one tensor add a step; the per-frame registration
runs in the seed build, not in the chain).

Open from this round: the fitted absolute gain differs 3x between
sequences of one rig (153: 23.3, 183: 7.3, 186: see its log) -- the
scene's brightness, item 2.2(a), next.

## 5. The frame-0 sweep removed; the preprocessing in-process (18-26 s)

The frame-0 range sweep added +0.05..+0.25 m to the registered cloud (an
offset from one frame's RA correlation; the cloud's median range already
sits 2-3 cm beyond the radar's moving-energy peak on every sequence). Off
for radar-registered sequences, 400 epochs: 183 RA Pearson 0.857 -> 0.874,
RD occupied 0.42 -> 0.65, val 1.39 -> 1.20; 186 unchanged. Pass A (a first
material fit on 132 frames) existed only to give the sweep a checkpoint
and goes with it. HuPR keeps both: the sweep is its only range
calibration.

The light stages (scaffold, background fit, coupling, gain, the person
passes, joint gains, motion detection, the visibility mask) run in the
chain's interpreter (`run_step`; a process each cost 1.3-2.9 s of imports
and CUDA initialisation, ~25 s of the first minute; `MM4DPS_STAGE_PROCESS=1`
for the A/B); the scaffold's static map is beamformed on the GPU and its
preview image is opt-in (11 -> 6 s). Seq 183: preprocessing 30 -> 17 s,
person stack 28 -> 10 s, alignment 5 -> 2 s.

Shortening the two material fits (background 150 iterations, person pass
B 1 epoch: 18 s in all) was measured at 400 epochs and NOT adopted: 153
unchanged, 183 worse (spectrogram +0.935 -> +0.877, energy +0.896 ->
+0.664) -- the absolute-gain fit follows the background fit's convergence
(the median per-cell ratio: 8.4 after 500 iterations, 36.9 after 150;
the energy ratio, tried instead, gave 183 a gain of 1.2 the trainer's
materials could not raise). The gain must become one number per rig,
fitted across recordings (2.2(a)); the fits keep their lengths until it
is (`MM4DPS_BACKGROUND_ITERS`, `MM4DPS_PERSON_EPOCHS` for the A/B).

## 6. The absolute gain is learned (2.2(a) closed); the short fits adopted

The gain was the last calibration fitted *before* training: HuPR's from a
shipped seed (`seeds/seq<S>/absolute_gain_render_result.json`, boards at
3.0 / 8.5), RT-Pose's from the median measured/rendered static |RA| ratio
over the brightest decile after the background fit -- bimodal (wall cells
~0.6, room cells >100), so it followed the fit's convergence (183: 8.4 at
500 iterations, 36.9 at 150) and differed 3x between sequences of one rig.

Now one `absolute_log_gain` parameter per radar in the velocity trainer
(`--absolute-gain auto`, the default; `--gain-lr 1e-2`), initialised from
the measured/rendered loop-0 |RA| level of the first 4 frames at gain 1
and learned with the materials; the value at the end of training is the
estimate, saved in the checkpoint and `result.json`
(`absolute_receiver_gain`) and handed to every render. The losses are
normalised by the target's level only (`mean((pred - target)^2) /
mean(target^2)`), so the rendered level has to match *during* training --
a gain fitted only afterwards would leave the bounded reflectivities to
absorb it -- which is why it is a trainer parameter and not a post-hoc
fit. It is identified only jointly with the materials' scale: the same
sequence lands at 13.5 with a 500-iteration background fit and 28 with a
150-iteration one, at the same metrics.

400 epochs, learned against the pre-fitted gain (`metrics.json`: RA
Pearson mean over frames / RD occupied Pearson / micro-Doppler
spectrogram, energy correlation):

| sequence | gain | RA | RD occ. | MD spec. | MD energy |
|---|---|---|---|---|---|
| 183 fitted -> learned | 8.4 -> 13.5 | 0.874 -> 0.879 | 0.652 -> 0.650 | +0.935 -> +0.941 | +0.896 -> +0.926 |
| 153 fitted -> learned | 23.3 -> 28.6 | 0.915 -> 0.920 | 0.512 -> 0.592 | +0.930 -> +0.922 | +0.926 -> +0.956 |
| 256 H/V seed -> learned | 3.0/8.5 -> 6.0/5.0 | 0.858/0.833 -> 0.856/0.831 | 0.829/0.781 -> 0.868/0.779 | +0.978/+0.980 -> +0.982/+0.980 | +0.921/+0.970 -> +0.925/+0.969 |

The short material fits (background 150 iterations, person pass B 1
epoch), which failed in section 5 through the gain fit, pass with the
learned gain: 183 RA 0.879 -> 0.884, RD 0.650 -> 0.654, MD +0.941 /
+0.925, at 18 s of preprocessing (10 + 6 + 2) against 28. They are the
chain's defaults (`BACKGROUND_ITERS`, `PERSON_EPOCHS`).
Gated on the other two as well: 153 RA 0.920 -> 0.919, RD 0.592 -> 0.602,
MD +0.922 -> +0.928 / +0.956 -> +0.957 (preprocessing 11 + 6 + 2 s); HuPR
256 H/V RA 0.856/0.831 -> 0.855/0.833, RD 0.868/0.779 -> 0.875/0.787, MD
energy +0.925/+0.969 -> +0.939/+0.972 (HuPR's background model is a seed,
so only its person pass shortened). Wall clock is unchanged where it
matters: RT-Pose 183 velocity stage 747 s, of which training 690 s (400 x
1.15 s of steps at 15.3 ms, 80 evaluations x 1.65 s, ~90 s of start-up:
target cache, compile, capture).

## 7. No seeds: the person built and registered in-chain on every dataset

Every per-sequence and rig seed is out of the chain (docs/SEEDS.md): the
person sequence is built from the GVHMR dump and registered to the radar
frame by frame on both datasets (HuPR's camera at the rig, co-boresighted;
its GVHMR runs 30% long, scale 0.707 on 256), the background material and
the antenna coupling are fitted in-chain on both, the joint scattering
gains are learned (geometric mean 1), and the frame-0 sweep, pass A, the
bake and the seed-pointing defaults are gone. The visibility ray cast is
grid-culled (bitwise the brute force, 258 -> 5 ms a frame).

400 epochs, seedless against section 6's runs:

| sequence | RA | RD occ. | MD spec. | MD energy | held-out |
|---|---|---|---|---|---|
| 256 H/V | 0.855/0.833 -> **0.951/0.914** | 0.875/0.787 -> 0.917/0.819 | +0.985/+0.983 -> +0.987/+0.988 | +0.939/+0.972 -> +0.936/+0.965 | 0.858 -> 0.681 |
| 183 | 0.884 -> 0.881 | 0.654 -> 0.641 | +0.941 -> +0.929 | +0.925 -> +0.889 | 1.176 -> 1.195 |
| 153 | 0.919 -> 0.918 | 0.602 -> 0.585 | +0.928 -> +0.926 | +0.957 -> +0.950 | -- |

HuPR's gain is the background model: the seed was fitted on an unaligned
scaffold and moved into place by the bake, the in-chain fit sees the
aligned scene. RT-Pose's only substantive change is the joint gains (its
sequence rebuilds bitwise, its coupling and background were in-chain
already): 183's MD energy correlation is +0.889 learned, +0.902 with the
gains fixed at 1 (`--joint-gains fixed`), +0.925 with sequence 1's fitted
values -- one run each, against a run-to-run spread of +0.896..+0.926 on
this sequence with an identical configuration. Learned stays the default:
the model's own estimate, and no number from another recording.

## 8. The scene alignment fitted in-chain; the preprocessing on the GPU

HuPR's scene alignment constants are gone: `reconstruction.fit_range_alignment`
fits the monocular scene's range translation from the recording on the GPU
(0.3 s) -- the occupancy-vs-person-free-map correlation of the old step 1,
range only, since the map does not identify the lateral or the height
(docs/SEEDS.md). The rest of the preprocessing moved off the CPU and out of
per-frame host round trips: the person's range and targets per block of
frames, the visibility computed in memory before the frames are written,
the voxelisation and PCA normals of the scaffolds in torch (positions
bitwise Open3D's, 99.2-99.7% of normals within 0.1 degree), the person
material's masks in one scatter and its target median on the device
(bitwise), a HuPR archive member inflated once per process. Background fit
75 iterations. Preprocessing: RT-Pose 183 **16 s** (scaffold 6, fit 1,
sequence 3, material 4, detection 2), HuPR 256 **18 s** (scaffold 8, fit 2,
sequence 3, material 3, detection 1).

400 epochs against section 7:

| sequence | RA | RD occ. | MD spec. | MD energy | held-out |
|---|---|---|---|---|---|
| 256 H/V | 0.951/0.914 -> **0.957/0.925** | 0.917/0.819 -> 0.919/0.822 | +0.987/+0.988 -> +0.988/+0.989 | +0.936/+0.965 -> +0.946/+0.974 | 0.681 -> 0.644 |
| 183 | 0.881 -> 0.879 | 0.641 -> 0.629 | +0.929 -> +0.927 | +0.889 -> +0.907 | 1.195 -> 1.204 |
| 153 | 0.918 -> 0.912 | 0.585 -> 0.595 | +0.926 -> +0.930 | +0.950 -> +0.963 | -- |
