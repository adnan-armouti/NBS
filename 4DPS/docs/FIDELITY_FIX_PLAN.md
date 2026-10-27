# Fidelity fix plan: the HuPR arms and the RT-Pose static background

Written 2026-09-16 from the downstream results (docs/DOWNSTREAM_RESULTS.md
2.3, 3.3, 4.1) and the renderer as it stands (docs/METHODS_INVENTORY.md B-C,
`mm4DPS/`). Two defects, one rule: **every fix is to the renderer, its
objective or its shared preprocessing, applied to every sequence of every
dataset; nothing targets one dataset.** Development on six sequences,
HuPR 38 / 102 / 256 and RT-Pose 183 / 153 / 186; a fix ships only when it
removes the defect on the sequences that show it and moves no metric on
the others by more than its seed spread (section 4).

(Split membership, for the downstream proxies of section 4: HuPR 38 and
256 are official test, 102 is train; RT-Pose 183 and 186 are train, 153
val. A real-trained model's number on the *render* of a training sequence
is still an unseen-input number; it is noted where it matters.)

## 0. What the two downstream models consume -- the fidelity that matters

The rendered cube is bit-compatible with the recording (per-loop complex
range profiles, every TX-RX pair, 64 loops). What each model then reads
from it decides which rendered quantities have to be right.

| | MSCSA-PRGCN (HuPR) | HRRadarPose (RT-Pose) |
|---|---|---|
| static scene | **removed** (mean over the 64 loops subtracted before the Doppler FFT) | **kept** (no clutter removal; Doppler bin 32 of 64 is the static scene) |
| Doppler | the central 8 of 64 bins after the shift (28..35): the slow-radial part of the motion only | all 64 bins as input channels |
| angle | azimuth FFT 64 over the 8-channel row **and an elevation FFT 8 over the two rows (TX0/TX2 row and the elevated TX1 row)**, per board; the vertical board's azimuth is world elevation | azimuth 128 x elevation 32 over the 86 x 7 virtual grid: **the 3 elevated TX of the cascade form the elevation axis** |
| level | per-frame min-max then z-score: **scale-free** | `(|v| - 1e5) / 8.9e6` clipped at 0: **absolute**; a cell below 1e5 is 0, a cell at 1e7 is 1.1 |
| output / metric | 2D heatmaps of 14 keypoints in the image; COCO AP over OKS 0.5..0.95 | 3D heatmaps of 15 joints; MPJPE (pelvis-relative), MRPE (pelvis), Abs-MPJPE, cm |

Two things the renderer does today collide with this table:

* **The training sub-array.** The trainer renders and supervises only the
  TX that form the azimuth virtual array (HuPR 2 of 3 TX per board, the
  cascade 9 of 12; `velocity_field.train`, "THE TRAINING SUB-ARRAY",
  `--training-array azimuth`), because every loss term goes through the
  azimuth beamformer. The products render **all** TX with the fitted
  parameters. So the elevated TX rows of every product -- HuPR's second
  row, the cascade's 3 elevated TX -- are an *extrapolation of the fit*,
  never compared to the recording. Both downstream models read exactly
  those rows for their elevation axis.
* **The loss weighting over range.** Every term is `mean over cells of
  (|pred| - |meas|)^2 / mean over cells of |meas|^2` per frame and radar
  (`velocity_field.two_term_loss`, METHODS_INVENTORY C). Linear magnitude,
  one divisor for the whole map: a cell's weight in the loss is its own
  power, i.e. ~1/R^4 for a static scatterer. The far scene is almost
  unweighted; its rendered level is whatever the free per-point gains
  settle at. MSCSA-PRGCN never sees the static scene, HRRadarPose reads it
  at absolute scale out to 8 m.

The three cross-cutting diagnostics of section 3 (per-TX-row audit,
per-range-band audit, arm-versus-torso audit) are built once and read on
all six sequences.

## 1. HuPR: the arms (docs/DOWNSTREAM_RESULTS.md 4.1)

### 1.1 The evidence

S -> S 53.9 +/- 1.4 against R -> R 65.9 +/- 1.7. Per keypoint, S -> S is
within 0-4 AP of R -> R on hips (90.0 / 89.6 vs 90.7 / 90.3), knees
(83.4 / 83.0 vs 84.1 / 84.8), ankles (74.6 / 73.7 vs 77.5 / 73.5), head and
neck (77.3 / 79.7 vs 81.7 / 83.5), and short by half on the arms: elbows
19.7 / 28.2 vs 41.5 / 50.9, wrists 8.4 / 13.7 vs 22.5 / 25.6. Cross-domain
collapses both ways (R -> S 22.0, S -> R 15.8), with the same shape: hips
~50, arms 1-12. So the rendered torso and legs carry the pose the real ones
do; the rendered arms carry much less of it, and whatever they carry is not
what the real arms look like to the model.

What "the arm" is to the model: the elbow's and wrist's energy in the
central 8 Doppler bins (|radial velocity| small -- the arm at the turning
points of its swing, or a still arm), at its range, azimuth and elevation
cell, on both boards. Not the micro-Doppler of the swing.

### 1.2 Hypotheses, ranked by how directly the renderer's design predicts them

**H1. The elevated TX row is unsupervised (section 0).** The arm is the
body part whose elevation varies most (hanging, swinging, raised), and the
elevation row is what separates a raised forearm from the torso in the
horizontal board's heatmap. If the rendered TX1 row is wrong in level or
phase relative to the azimuth row, the elevation FFT of the render puts the
arm's energy at a different elevation than the recording's -- and it does
so on every sequence, which is what the cross-domain collapse looks like.

**H2. The arm geometry is wrong before rendering.** GVHMR's monocular
arm pose is the weakest part of a monocular fit (depth ambiguity of a
forearm pointing toward the camera; occlusion by the torso), and the
per-frame radar registration is *one scale per frame about the camera
origin* (`reconstruction.radar_depth_scales`): it fixes the torso's range
and leaves the arm's range error as GVHMR left it. A forearm rendered
10-20 cm off in range is 2-4 range bins off (4.7 cm a bin), i.e. in a
different heatmap cell.

**H3. The Doppler support leaves the arm under-constrained.**
`--doppler-support-snr 1.0` with the `loops-median` floor (production):
per frame, a Doppler bin is rendered in the plane term only if the
measured power at the person's range rows exceeds the median over the 64
bins; the rest are supervised as zero, which is a constant -- no gradient.
The swinging arm is the person's energy in the outer bins; those bins are
below the median exactly when the arm is the only thing in them, so the
arm's velocity and gain get gradient only through the 16 central bins of
the MD term and through the planes the torso lights. The plane term never
tells the fit "the wrist is here at 1.5 m/s". This is the k-SNR mask the
question names. It does not *suppress* the arm (an unrendered plane has no
gradient); it leaves the arm's velocity field and the elbow/wrist
scattering gains less determined than the torso's, so they sit near their
initialisation -- and the render's low-Doppler arm energy (what the model
reads) is then whatever the initialisation gives.

**H4. The arm's scattering level.** One shared six-parameter material for
the whole body, 14 per-joint gains with the geometric mean pinned to 1,
soft-skinned to the SMPL points. If the fit lowers the elbow/wrist gains
(they are the cheapest way to reduce loss on outer Doppler planes that
*are* rendered but land in the wrong cell), the arm is rendered dim in
every Doppler bin, including the 8 the model reads.

**H5. The temporal cap.** The joint velocity field lives in a DCT basis of
420 of 600 bands: 3.5 Hz at 10 fps. Walking arm swing is ~1 Hz with
harmonics to ~3 Hz; a hand wave 2-4 Hz. Marginal; measured, not assumed.

### 1.3 Diagnostics (one afternoon on the GPU, no training)

Each is a script under `mm4DPS/qa/` or `downstream/tools/`, run on the six
sequences, its numbers into docs/FIDELITY_FIX_RESULTS.md.

* **D1 -- per-TX-row audit (H1; also section 2).** For each board and
  sequence, every 10th frame: the measured and rendered complex range
  profiles of the *elevated* TX row against the *azimuth* rows -- the
  ratio of static (loop-mean) magnitude per range bin, elevated / azimuth,
  measured vs rendered; the phase difference between the elevated row and
  the azimuth row at the person's range (what the elevation FFT reads);
  and the elevation-FFT heatmap (the adapter's `heatmap_from_cube`, the 8
  elevation bins at the person's cells) measured vs rendered, Pearson and
  the argmax elevation bin. *Verdict*: if the elevated row's level or phase
  is off by more than the azimuth rows are, H1 is confirmed.
* **D2 -- arm-versus-torso audit (H2-H4).** From the SMPL skinning weights,
  a per-point body-part label (torso, upper arm, forearm/hand, thigh,
  shank/foot, head). Per frame: project each part's visible points into
  the (range, azimuth) cells of each board (`person_range_angle_masks_torch`
  per part) and read the measured and rendered energy of those cells in
  the clutter-removed Doppler cube, (a) summed over all Doppler bins,
  (b) over the model's central 8. Report, per sequence and part, rendered
  / measured energy and the range and azimuth centroid offset of the
  rendered part's energy from the measured part's. *Verdict*: a forearm
  ratio far below the torso's is H4; a centroid offset in range with a
  good ratio is H2; both good in (a) but bad in (b) is the Doppler
  distribution (H3/H5).
* **D3 -- GVHMR arm geometry against the labels (H2).** HuPR ships 2D
  keypoints per frame. Project GVHMR's 14 joints (after the radar
  registration) into the image and report the per-joint pixel error
  against the labels on the three sequences, and the arm joints' range
  spread (the forearm's rendered range minus the torso's) per frame.
  *Verdict*: arm joints with 2-3x the torso's pixel error, or range
  excursions the radar's moving-power profile does not show, is H2.
* **D4 -- the fitted joint gains and the Doppler support (H3, H4).** Read
  `joint_log_gain` from the three checkpoints; read the per-frame rendered
  Doppler support (the union at K = 1) and mark, per frame, which bins the
  wrist's bistatic path rate falls in (from the fitted velocity field) and
  whether they were rendered. *Verdict*: wrist/elbow gains well under 1
  is H4; wrist bins outside the support in most motion frames is H3.
* **D5 -- the downstream proxy (all).** The five real-trained MSCSA-PRGCN
  seeds on the render of each of the three sequences: per-keypoint AP and
  per-keypoint pixel error per sequence (the evaluator's per-sequence
  block extended per keypoint). This is the number the fix has to move; it
  runs in ~1 min a sequence and a seed. Baseline it first.
* **D6 -- the velocity spectrum (H5).** The camera joint velocities'
  DCT energy above 3.5 Hz per joint, before the cap.

### 1.4 Fixes, first principles

Ordered by what the diagnostics are expected to show; each is general.

* **F1. Supervise every rendered channel.** Train with the full array
  (`--training-array full` exists as an ablation, measured at +46% step
  time on HuPR, 41.5 vs 28.4 ms) *and* add the elevated rows to the loss:
  either a second beamformed view (the elevation FFT over the TX rows, the
  same three terms on the (range, elevation) map) or, simpler and exact,
  the plane and loop-0 terms on the per-channel range profiles of the
  elevated row (no beamformer needed: the loss is on |S_{t,r}[k]| per
  channel, the same normalised L2). Principle: nothing the product ships
  may be an extrapolation of the fit. Applies to both datasets (cascade:
  3 elevated TX, 48 channels).
* **F2. Doppler support from the model, not from the measured SNR.** The
  set of planes to render per frame is the union of the measured support
  (as now) and the planes the *rendered* person can populate -- the range
  of the fitted bistatic path rates over the visible points, +/- the
  7-tap PSF. Principle: the support may not depend on whether the
  recording is bright where the thing being fitted is; a bin the render
  puts energy in must be compared. Cost: more planes on fast sequences
  (the worst case 63 is already tolerated); the CUDA-graph step needs a
  fixed plane count, so the support is computed once per sequence as the
  union over frames, as now.
* **F3. Registration that follows the limb, not only the torso.** Replace
  the single per-frame scale by a per-part range correction: the moving-
  power range profile at the person's azimuth, split by Doppler sign or
  by the part masks of D2, gives a range for the swinging limb separate
  from the torso; the arm joints are translated in range to it (bounded
  by the SMPL bone length). Principle: the radar measures range; the
  monocular mesh only proposes it. Only if D3 shows the arm range wrong.
* **F4. A per-part scattering amplitude with a prior.** Keep the shared
  material; let the per-joint gains be regularised to 1 (an L2 on
  `joint_log_gain`) instead of pinned in geometric mean, so a limb is
  dimmed only when the data supports it -- and once F1/F2 give the arm
  gradient, it will be. Only if D4 shows the arm gains collapsed.
* **F5. Raise the temporal cap** to the Nyquist band count if D6 shows
  energy above 3.5 Hz on the arm joints; the cap was a regulariser for
  the torso's smooth trajectories, not a physical limit.

Not a fix: changing the HuPR adapter (its preprocessing is the released
one and is applied to both domains; section 0).

## 2. RT-Pose: the static background (docs/DOWNSTREAM_RESULTS.md 3.3)

### 2.1 The evidence

Zero-Doppler energy rendered / real per range band on 10 of 10 sequences:
0.02-0.48 at 1-2 m, 0.4-0.8 at 2-3 m, ~1 at 3-4 m, 1.6-3.7 at 6-7 m,
1.7-4.3 at 7-8 m; moving energy 0.75-0.89 everywhere. Under the released
absolute normalisation the far excess floods the DZYX static channel
(3.4-4.8x the real's energy, 5-7x the voxels above 1). Swapping only the
five static Doppler bins of the rendered tensor for the real cube's puts
the real-trained model back at R -> R accuracy (MRPE 6.2 / 9.7 / 10.2 cm on
159 / 108 / 168 against 43 / 653 / 406 as rendered): the rendered person
is read correctly, the rendered background is not.

### 2.2 Hypotheses

**H1. The elevated TX rows (section 0).** The cascade's 3 elevated TX
(48 of 192 channels) are never supervised. The DZYX tensor's elevation
axis is a 32-point FFT over the 7 virtual rows; the static scene is
resolved in elevation mostly by those rows. A rendered elevated row at the
wrong level does not change the azimuth RA map the loss sees, and puts
energy at every elevation the resample then spreads over z.

**H2. The range weighting of the static term (section 0).** The loop-0 RA
term with one divisor per map: cells at 7 m carry (2/7)^4 ~ 0.7% of the
weight of cells at 2 m. The far scene's per-point gains
(`background_point_gains`, free per point) and materials are effectively
unconstrained. Their initialisation is the 75-iteration `train_motion
dual-radar` fit, whose loss is `forward_render.log_pearson_loss`: a
Pearson correlation of the *dB* maps (peak-referenced, 42 dB floor) --
scale-free by construction, so it fits the background's *shape* with
equal weight across the dynamic range and sets no level at all. The
absolute level of every range band is therefore decided only in the
velocity stage's linear term, which cannot see the far bands. The far
scene ends up at a level nothing ever fitted. The near deficit (0.02-0.5x at 1-3 m) is the other face
of the same thing: the bright near cells dominate the loss, and a fit that
cannot match their *shape* (sparse LiDAR points, one normal per point) is
pushed to under-render them rather than mismatch them.

**H3. The scaffold's coverage over range.** The LiDAR scaffold is 20,000
points, PCA normals at 0.5 m radius, the person's column cleared (0.6 m).
Livox density falls with range and the near floor and the rig's own
surroundings may be outside its field of view; the far wall is dense and
planar (a specular slab facing the radar). Under H2 an over-dense far
plane with unconstrained gains renders bright.

**H4. The absolute level, per radar.** One learned receiver gain per
radar (`g_radar`) sets the absolute scale; it is fitted where the loss
weight is (near range, the person's Doppler planes). If the near scene is
under-rendered (H2) the gain compensates upward, and the far scene rides
up with it.

### 2.3 Diagnostics

* **D1 -- per-TX-row audit** (the same script as 1.3 D1, on the cascade):
  the static range profile per range band, rendered / measured, separately
  for the 9 azimuth TX and the 3 elevated TX. *Verdict*: excess only in
  the elevated rows is H1; excess in both is H2-H4.
* **D2 -- loss weight by range band.** On the fitted checkpoint, the
  loop-0 term's per-cell squared error summed per 1 m range band, as a
  fraction of the total, measured against the band's share of the static
  *energy* -- the number that says how little the far scene counts.
  Cheap: one forward pass.
* **D3 -- scaffold density and the fitted gains by range.** Points per
  range band; `background_point_gains` distribution per band; the
  material's coherent fraction per band. *Verdict*: far gains above their
  initial values, or a far band with 5x the point density, is H2/H3.
* **D4 -- the downstream proxy.** The five real-trained HRRadarPose seeds
  on the render of 183 / 153 / 186: MRPE and MPJPE, plus the static-band
  swap (3.3's probe) as the ceiling. The number the fix must move.

### 2.4 Fixes, first principles

* **F1. Supervise every rendered channel** (1.4 F1; the same change).
* **F2. Equalise the static term over range.** Keep the linear magnitude
  (the house rule: no dB); normalise per range row instead of per map:
  `sum over rows of mean over angle (|pred| - |meas|)^2 / mean over angle
  |meas|^2`, rows weighted equally (or by the row's share of the *count*
  of cells above the floor, so empty rows do not count). Principle: the
  objective must constrain the rendered level at every range the
  downstream reads, and the 1/R^4 of the radar equation must not decide
  which ranges are fitted. The same divisor structure is already in place
  for the plane term ("its own divisor because a plane sits at a
  different level"); this is the range analogue. The MD term is unchanged.
* **F3. Scaffold coverage.** If D3 shows the near floor absent or the far
  wall over-dense: voxel size by range (constant points per steradian, not
  per cubic metre), and the near floor from the LiDAR frames of the
  *whole* sequence rather than the annotated samples. General to both
  scaffolds (the monocular one has the same choice in `run_scaffold`).
* **F4. Fit the receiver gain on the static scene alone first** (the
  loop-0 term, all ranges equalised by F2), then freeze it for the
  velocity stage, so the person's planes cannot pull the absolute level.
  Only if D2/D3 show H4.

**On static clutter removal in the RT-Pose adapter.** The released
HRRadarPose chain keeps the static scene (its `DopplerProcClutterRemove`
runs with the removal off) and the paper's numbers are with it. Removing
it in our adapter would (a) change the downstream model we claim to
reproduce -- the R -> R number moves and a new protocol hash is needed --
and (b) hide the background defect rather than fix it, on this dataset
only. It is therefore **not the fix**. It is a legitimate *sensitivity
experiment* to run once, on the real arm, to report how much of
HRRadarPose's accuracy comes from the static channel (the current models
lose 2-15 cm MRPE when it is zeroed at test time); that number belongs in
the paper's discussion, not in the pipeline.

## 3. The shared instruments (build once)

1. `per_tx_row_audit(sequence)`: measured vs rendered static and moving
   energy per range band per TX row; the elevation-FFT view per board /
   per elevation bin. Both datasets.
2. `part_energy_audit(sequence)`: per body part, per Doppler-bin group,
   rendered / measured energy and centroid offsets, from the SMPL part
   masks. Both datasets (RT-Pose's joints map to the same parts).
3. `downstream_proxy(sequence, domain)`: the five real-trained downstream
   models on one sequence's cube in one domain -- HuPR per-keypoint AP and
   pixel error, RT-Pose MRPE / MPJPE / per joint, plus the static-band
   swap for RT-Pose. Runs on the node in a minute a sequence.

## 4. The development loop and the regression gate

* **Sequences**: HuPR 38 (still person -- the arm at rest, elevation and
  placement only), 102 (walking -- the swing), 256 (the gate/timing
  workhorse); RT-Pose 183 (gate/timing), 153 (the blank micro-Doppler
  case), 186. Every diagnostic and every candidate fix runs on all six,
  every time.
* **Baseline** (before any change): the fidelity suite as it stands
  (docs/METRICS.md: the RA / RD / MD Pearson, SSIM, NRMSE, the person-
  masked numbers), the three instruments above, and the downstream
  proxies. Written to docs/FIDELITY_FIX_RESULTS.md with the chain commit.
* **Per candidate fix** (one flag or one code change at a time): re-fit
  the six sequences (~7.5 min each on one card; six in parallel), re-run
  the same set. A fix is *accepted* when (i) the defect's own number moves
  the way the diagnosis predicts on the sequences that show it (HuPR:
  forearm / torso rendered-over-measured energy ratio and the arm
  keypoints' R -> S proxy; RT-Pose: the static energy ratio per band within
  0.7-1.4 at every band, and the R -> S MRPE proxy within 2x the real), and
  (ii) no fidelity-suite metric on any of the six sequences worsens by
  more than its run-to-run spread -- measured once by fitting the baseline
  chain twice with different seeds on the six sequences (the sweep reports
  no seed spread) -- and no downstream proxy on any sequence worsens. A fix that helps one dataset and costs the other is not a fix.
* **Order**: D1 and D2 of both sections first (they are the same two
  scripts) -- they decide between H1 (train every channel) and H2 (range
  equalisation), which are the two changes most likely to be needed and
  are independent; then the arm-specific D3-D6. Implement F1 and 2.4 F2
  first (each is a contained change to `velocity_field.train` and the loss),
  gate them, then F2 (Doppler support) if D4 asks for it.
* **After acceptance**: re-render every sequence of both datasets with the
  chain at the accepted commit (the products are versioned by chain commit
  and `render_config`), re-materialise the rendered caches
  (`domain materialise --domain rendered`, 4 s a board / 40 s a sequence),
  re-run the domain-blindness gate, and rerun the rendered and mixed arms
  of both datasets (HuPR 13.5 h, RT-Pose ~11 h on the node). The real arms
  do not move.

## 5. What is *not* in this plan

* Fixing the arm through the downstream adapter or the split.
* Static clutter removal as a pipeline change (2.4).
* Per-dataset flags in the chain.
