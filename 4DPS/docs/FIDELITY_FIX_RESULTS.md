# Fidelity fix results: the six/eight-sequence dev loop (docs/FIDELITY_FIX_PLAN.md)

All runs on mmsd-h100 (8 x H100), one sequence per GPU, from the workstation's preprocessed
workspaces (`outputs/fixdev/seq<k>/.work`, chain commit 96da32a for preprocessing); every
candidate retrains the velocity field, renders, scores (the chain's own suite), then the audit
(`downstream/tools/fidelity_audit.py`) and the downstream proxy (`downstream/tools/proxy.py`, the
five real-trained downstream models on the render). Dev sequences: HuPR 38 / 102 / 256 / 40 (40:
official test, the hardest R -> R), RT-Pose 183 / 153 / 186 / 108 (108: official test, the
527 cm R -> S case). Linear scale, never dB; every number names its estimator.

## 1. The run-to-run spread (base_node vs base_node2, identical flags)

| metric | spread over the eight sequences |
|---|---|
| RA Pearson, RAc Pearson | <= 0.005, <= 0.008 |
| RD occupied Pearson | <= 0.02 |
| MD centroid / spread / energy r | <= 0.02 / <= 0.06 / <= 0.01 |
| elevated-row static ratio (total) | up to 0.8 on HuPR 256 (5.75 vs 4.95) |
| HuPR proxy AP (per sequence, 5 seeds) | +/- 0.2 to 3.0 (40: 14.6 vs 17.6) |
| RT-Pose proxy MPJPE | 183/186 +/- 1; 153 81 vs 71; 108 365 vs 233 (the failing fits are noisy) |

The gate (plan 4): a change is accepted when its own defect number moves as predicted and no
suite metric worsens by more than this spread on any sequence, and no proxy worsens beyond its
seed std.

## 2. Baseline diagnostics (base_node), what the eight sequences showed

* **H1 confirmed on every sequence.** Elevated-TX-row static energy, rendered / measured:
  HuPR vertical 3.5-15.5x (far bands 15-49x), horizontal 0.6-5.8x; RT-Pose 0.8-4.5x overall,
  3.6-26x at 6-9 m. Elevated-over-azimuth phase at the person on RT-Pose: +27 .. +150 deg
  (circular std 83-118): the DZYX tensor's z axis unconstrained. Elevation view Pearson RT-Pose
  0.16-0.64, argmax agreement 7-27%.
* **H2 (range slope) present but secondary**: azimuth-row static by band 0.0-0.6x at 1-3 m,
  1.3-4x at 6-7 m, both datasets.
* **Arms: no amplitude or placement deficit** on any HuPR sequence (part energy ratios uniform
  1.0-2.1x across torso / arms / legs, centroid offsets <= 0.5 bins). GVHMR arm joints vs the
  HuPR labels: 6 px on 102 / 256 (torso 7-8), 17-18 px on 38 / 40 (torso 10), in the 512-px frame.
* **Joint scattering gains collapsed everywhere**: torso 0.09-0.5, legs 1.2-47.
* **Temporal cap (H5) ruled out**: 0-3% of the camera joint-velocity energy above 3.5 Hz, the
  same for arms, torso and legs; none in the fitted field.
* **RT-Pose 108's fit fails on its own**: the per-frame registration locks the person at 4.10 m
  (the moving-power peak; rain / another mover) against 5.59 m in the labels and GVHMR's own
  range x 1.1; 177/299 confident frames. RD r 0.05, MD centroid ~0 in both baselines. 183 / 153 /
  186 are placed within 0.3 m of the labels. -> F6.

## 3. Candidates

Flags are the velocity trainer's (`mm4DPS/velocity_field.py`); every default is the baseline.

| candidate | flags | verdict |
|---|---|---|
| f1_elev | `--training-array full --elevation-term on` | defect fixed; accepted in the motion form below |
| f1_motion | + `--elevation-planes motion` | **accepted** (same accuracy, cheaper) |
| f1m_f2 | + `--ra-loss l2-range` | **rejected** (RA / RAc down everywhere, no proxy gain) |
| f1m_f4 | f1_motion + `--joint-gain-prior 1.0` | not adopted (pins every gain to ~1; MD spread down on 256 / 40; nothing else moves) |
| f1m_nfft | f1_motion + `--n-angle 16` (HuPR) / `86` (cascade) | **accepted** (within spread or better; -10 / -13% runtime) |
| f6_reg, f1n_f6 | + `REGISTRATION_SECTOR=on` (person stage rerun) | **accepted** with one caveat (3.5) |
| f1m_dsup | f1_motion + `--doppler-support-model on` | neutral, not adopted (3.6) |
| f1n_rebuild | accepted flags, person stage rebuilt (the control for F6) | -- |

### 3.1 F1, the elevation term (f1_elev, f1_motion)

Elevated-row static per band, rendered / measured: HuPR 256 vertical 15-26x -> 0.6-1.0x; RT-Pose
108 20-26x -> 1.0-1.4x, 153 17-20x -> 1.05-1.4x; the elevated rows now track the azimuth rows
band by band on all eight. Elevated / azimuth phase at the person, RT-Pose: 183 +150 -> -2 deg,
153 +93 -> -9, 186 +27 -> -16 (circular std 51-80 from 93-118); 108 unchanged (its fit fails,
section 2). HuPR horizontal +8 .. +35 deg (as before), vertical mixed. Elevation view Pearson:
RT-Pose 153 0.64 -> 0.72, 183 0.61 -> 0.63, 186 0.17 -> 0.21; HuPR unchanged (0.75-0.94).

Downstream proxies (mean over 5 seeds, +/- seed std): HuPR AP 38 16.1 -> 19.8 (+/- 4.6), 102
14.0 -> 17.6 (2.0), 256 23.0 -> 21.9 (4.0), 40 14.6 -> 15.2 (3.2) -- mean +2.2; RT-Pose MPJPE
153 **81.3 -> 17.1 cm** (MRPE 84 -> 14), 183 23.4 -> 28.9 (std 16), 186 29.4 -> 30.7, 108 365 -> 344
(the fit that fails). Suite (azimuth-only): RA r -0.003 .. -0.013, RD +0.01 .. +0.05, MD within
spread.

f1_motion (the elevation plane term on the micro-Doppler band only, loop-0 unchanged) is
indistinguishable from f1_elev on every metric (e.g. 153 MPJPE 15.6, 38 AP 15.0 +/- 4.7) and
cheaper: HuPR vf 346-391 s vs 415-428, RT-Pose 920-1015 s vs 979-1069.

Runtime (step ms, 400 epochs; base 16.4-17.2 HuPR, 10.7-12.0 RT-Pose): f1_elev 30.0 / 28.1;
f1_motion 23.0 / 26.4; VRAM 9 -> 30 GB (RT-Pose, all planes) / 27 GB (motion). The full array
alone (12 vs 9 cascade TX, 3 vs 2 HuPR) is measured at +46% on HuPR; the rest is the 2D angle
FFT + L2 of the term, and on the cascade the azimuth term's fall-back off its fused gather --
fixed afterwards (the gather generalised to the full array, bitwise the reference; in the nfft
numbers below).

### 3.2 F2, the range-equalised loop-0 term (f1m_f2) -- rejected

RA Pearson down on every sequence (HuPR 102 0.958 -> 0.944, 38 0.934 -> 0.921; RT-Pose 186
0.855 -> 0.782, 183 0.866 -> 0.825), RAc down (102 0.809 -> 0.738), the near static levels drop
further (total static 0.15-0.29); the over-bright far bands come toward 1 (HuPR 102 4.7 -> 2.3x)
but the under-bright ones get worse (38 2-4 m: 0.13 -> 0.06); proxies do not improve (RT-Pose
183 / 186 MRPE 25 -> 29, HuPR 256 AP 22.5 -> 18.9). Equal weight per row over-corrects: the
per-band static shape is a scaffold / material limit, not a weighting one. The flag stays,
documented as measured.

### 3.3 F4, the joint-gain prior (f1m_f4) -- not adopted

At W = 1 every gain sits at 0.95-1.02 (the prior dominates the 14 log-gains). No metric moves
beyond the spread except MD spread r on 256 (0.75 -> 0.56) and 40 (0.89 -> 0.91); proxies flat.
The gains' collapse in the baseline therefore costs nothing the instruments see, and the arms'
deficit is not an amplitude one -- consistent with section 2. A milder W is untested.

### 3.4 The loss's angle-FFT sizes (f1m_nfft) -- accepted

`--n-angle 16` (HuPR, the lossless 2N-1 of an 8-slot row) and `--n-angle 86` (cascade, the
slot count) against f1_motion's 64 / 128: RA / RAc / RD within the spread or better (HuPR 38
RAc 0.432 -> 0.477, RD 0.666 -> 0.696; 40 RD 0.324 -> 0.349), MD within spread, proxies
unchanged within their std. Runtime: HuPR vf 346-391 -> 312-317 s (steps 23.0 -> 18.9-19.7
ms), RT-Pose 920-1015 -> 806-852 s (steps 26.4 -> 21.5-23.0 ms).

### 3.5 F6, registration in the camera's sector (f6_reg, f1n_f6) -- accepted with one caveat

The person stage is rerun for it, and the chain documents the person material fit as stochastic,
so the comparison is `f1n_f6` (accepted flags + `REGISTRATION_SECTOR=on`, person stage rebuilt)
against `f1n_rebuild` (the same flags, person stage rebuilt without the sector). The rebuild
alone moves nothing on RT-Pose (183 RD 0.678 vs 0.678) and up to the seed std on the HuPR
proxies (40: 15.9 vs 20.8 AP).

Placement: 183 / 153 / 186 unchanged to the centimetre (the argmax was already in the sector);
**108 from 4.10 m to 5.49 m against 5.59 m in the labels**, 299 / 299 confident frames (177
before). Metrics, sector vs rebuild control: HuPR 38 RD 0.693 -> 0.762 (AP 17.0 -> 18.5), 40 RD
0.345 -> 0.501 (RAc 0.228 -> 0.242), 102 RD 0.890 -> 0.899 (MD centroid 0.93 -> 1.00; AP 17.7
-> 16.1, std 3.1), 256 unchanged; RT-Pose 108 proxy MRPE 330 -> 216 cm, MPJPE 331 -> 250; 153 /
186 unchanged; **183 RD 0.678 -> 0.644 and RAc 0.458 -> 0.442** with the same placement (the
per-frame radar ranges moved +4 cm, one bin), the proxy unchanged (30.9 vs 31.6, std 21). That
one dip exceeds the spread; it is the sector picking the person's own moving cells over the
strongest mover in the row, one bin apart, and is recorded as the caveat.

On 108 the suite then scores 2 motion frames instead of 215: at the labelled range the person's
measured moving energy is faint under the rain (rendered / measured 0.19), so the radar-side
motion detector accepts almost no frame -- the fit cannot lock on 108 at any placement, and the
sector at least puts the render where the labels are.

### 3.6 F2 (Doppler), the support from the rendered person (f1m_dsup) -- neutral, not adopted

Every metric within the spread of f1_motion on all eight sequences (HuPR 38 MD centroid 0.65 ->
0.87 is the one visible move; its proxy 15.0 vs 15.3); no runtime change. The measured k-SNR
support already covers the planes the person populates; H3 is not what limits the arms.

### 3.7 The arms, what the loop can and cannot say

Of the plan's five arm hypotheses, H1 is fixed (F1), H2 / H4 / H5 are ruled out by the
instruments (section 2, 3.3) and H3 by 3.6. The R -> S per-keypoint proxy moved little on the
arms (elbows 4-10, wrists 1.5-10 AP per sequence before and after). The number that decides the
arms is S -> S per keypoint, which needs the rendered arm re-trained on the re-rendered
sequences (docs/DOWNSTREAM_PLAN.md; 13.5 h on the node for HuPR) -- outside this loop. What the
loop establishes is that the rendered arms are placed, weighted and now elevated as the
recording's, so a remaining S -> S deficit would sit below the instruments' resolution (the
limb's micro-structure within a range-angle cell), not in a supervised quantity.

### 3.8 The elevation term's floor (floorfix) -- accepted

With the elevation term the learned noise floor sat 2.5-3.1x UNDER the measured off-person power at
the person's rows on the cascade (183 2.50, 186 2.88, 153 3.13; the baseline 0.69; HuPR 1.05-1.06):
the RAE term carried the floor to its cells by the elevation window's energy alone (1.02) where the
exact count-and-window factor is 1.67 (cascade) / 1.89 (HuPR), and its gradient dragged the shared
floor. Also, a Hann over the cascade's 7 elevation slots weighted its end rows (the 9 azimuth TX and
TX0) at 0.146. Fixed (431b4b7): the exact factor, the floor detached in the RAE term, the elevation
axis untapered on both sides (bitwise the same transform on both). Floor restored (183 0.69, 186
0.81); metrics within spread; RT-Pose RA r +0.008..+0.013.

### 3.9 The RT-Pose R -> S residual, localised (probes on the accepted renders)

Swap probes with the real-trained seed-4 model (MRPE, cm): 183 real 4.3, rendered 29.8; real
person on the rendered static 7.8; **real background with the rendered person (a 2 m box about the
GT pelvis) 26.6; rendered background with the real person 10.1** (186: 4.9 / 24.6 / 7.8 / 26.6 /
8.7). The residual is the rendered person's own content; the static scene now costs ~3 cm. The
baseline renders show the same (183: 28.2 / 9.2). Floor-corrected, the person's energy per part is
right (the earlier 0.3-0.45x was the floor in the measured cells); its height (elevation centroid
offset -0.07..+0.13 of 32 bins) is right; its moving-power peak coincides with the measured one
(median 0.000 m). Rejected causes: the soft skinning (0.10 / 0.15 m widened the cascade's spread,
6.0 / 4.9 vs 3.9 bins), the floor (3.8: fixed, no change to the residual), the registration
(`centroid`: pelvis on the labels within 0.1 m but RD Pearson 0.68 -> 0.47 / 0.71 -> 0.42 and the
proxies unchanged -- not adopted, 8c7627f + revert of the default). What remains: in the recording
the strongest moving return of a walking person is the leading limb 0.2-0.3 m ahead of the pelvis
(label - measured peak +0.29 m on 183, +0.18 on 186, +0.08 on 153); the render's is body-centred.
The rendered person's part-wise share of the moving return is the open question; the next
instrument is that share per part in the RD domain, floor-corrected, real vs rendered.

`--joint-gains fixed` (gains_fixed): every metric within the spread of f1m_nfft; not adopted, not
needed.

### 3.10 The RT-Pose residual, closed out for now (2026-09-17, probes on the accepted renders)

Figures: /home/adnan/Desktop/mmSD/artifacts/rtpose_residual/ (range-Doppler maps, range profiles, peak vs
label per frame, the swap probes).

The residual is a bias plus scatter of the real-trained model's root on the rendered person: 183 mean
(-10.9, +10.1, -8.8) cm in (range, lateral, up) with per-axis std 11 / 17 / 16 against (0, 0, 0) and
1 / 3 / 3 on the real cube; 186 (-11.4, +11.7, -3.4). Tested in the model's own input (the normalised
DZYX tensor, a 2 m box about the GT pelvis): the rendered person's static band agrees with the real
one (centroids within 0.05 m, same extent, same voxel count); its MOVING bands sit 0.10 m nearer and
0.05 m lateral of the real ones with the same extent, at 0.55-0.7 of the energy. Probes:
* scaling the rendered person patch x1.3-2.5: no change (183 29.8 -> 28.3);
* a synthetic multipath tail behind the rendered person (0.3x at +0.32 m): the range bias goes
  (-11 -> 0) but MRPE only 29.8 -> 26.7 (lateral / vertical bias and the scatter stay); the REAL
  tail pasted in: no change -- the model does not read the tail;
* the person's own part-wise share of the moving return cannot be resolved in range-Doppler at this
  resolution (the parts' masks overlap); what the maps show (fig1) is the real person as several
  sharp limb sub-returns at distinct range x Doppler and the render as one smooth blob;
* `--plane-rows person` (the plane terms on the person's rows, normalised on the person; c187d80):
  RD Pearson 183 0.68 -> 0.66, 186 0.71 -> 0.63, proxies unchanged or worse (186 MRPE 22.7 -> 30.9),
  bias unchanged -- rejected (the flag stays, off).

Conclusion: with the static scene, placement, level, floor, spread, skinning, registration and the
loss's focus all ruled out, what is left is the person MODEL's limb-resolved structure -- a shared
material over 6,890 points with joint-skinned velocities renders the limbs as one smooth return
where the recording has sharp per-limb specular flashes at distinct range and Doppler. HRRadarPose
reads that structure (RD Pearson 0.65-0.72 on the cascade against 0.9 on HuPR's coarse Doppler).
Closing this needs a person-model change (per-part materials / normals, or limb-level specular
scatterers), a day or more with an uncertain payoff, and is left open. The RT-Pose S -> R column
from the current renders will read ~2-3x the real MPJPE on walking sequences; the R -> S and
R + S -> R columns are the ones the accepted fixes improved (153: 81 -> 15 cm).

### 3.11 The person model, taken apart (2026-09-17, floorfix renders; `downstream/tools/person_structure.py`)

The instrument renders the person alone from the product's checkpoint with the trainer's own
forward model (`render_factorized` + the K2 splat: the registered SMPL points, the learned
normals and per-point materials, the joint gains, the fitted field), so it can be split by
body part, by BSDF lobe and into its per-point contributions; the measured side is the
recording's range profiles, floor-corrected per Doppler bin. Cascade frame: 64 loops x 780 us =
49.9 ms, one Doppler bin = 0.039 m/s; a walking torso at 0.65 m/s sits at bin 16-17. The probes
named below live in /data/mmsd/logs/4dps/ on the node (`probe5.py` core / halo swaps,
`probe6.py` the halo split, `az_offset.py` / `az_cen.py` / `az_check.py` the azimuth,
`box_peak.py` / `box_profiles.py` / `halo_doppler.py` the model's input, `ghost_range.py` /
`ghost_probe.py` the multipath, `vel_calib.py` / `vel_fit.py` the field, `cand_report.sh`).

**(c) normals, (d) fit -- settled.** The learned per-frame normals stay on the mesh normals
(angle to the mesh normal, median 2.4 deg, 95th pct 9 deg, none beyond 30 deg, 183/186/153/256);
they are not the lever. The loss is not converged at 400 epochs: on 183 the total is 3.13 at
epoch 10, 2.73 at 200 (the constant-lr half), then 2.65 / 2.41 / 2.19 / 2.12 at 250 / 300 / 350
/ 399 -- the MD term 0.33 -> 0.058 in the cosine tail alone -- so the base lr is too large for
part of the parameters and the fit only settles as it decays (same shape on 186, 153, 256).

**The sparsity is not the difference.** On the person's patch (range rows +-2, |Doppler| >= 3),
angle-summed power, medians over 10 frames of 183: peak/mean measured 128 vs rendered 128, top-1%
share 0.59 vs 0.55, local maxima above 0.2 x peak 2 vs 2. Fig1's "smooth blob" is not a
sparsity deficit at this level; it is WHERE the energy is.

**(a) the velocity field: the fit made it worse than its initialisation.**
Against the measured moving-Doppler centroid of the person's rows (m/s, per frame), the CAMERA's
hips' radial velocity fits with slope 1.00 / 0.95 and r = 0.97 / 0.98 (183 / 186, 20 / 24
frames); the FITTED field's hips fit with slope 0.66 / 0.57 and r = 0.83 / 0.79. Per joint the
fitted radial rms is 1.5-2.8x the camera's (183: hips 0.34 -> 0.62, ankles 0.38 -> 1.06 m/s)
with correlations 0.4-0.7. The torso's internal radial-velocity spread (max - min over hips,
neck, shoulders, median over frames) is 0.06 / 0.09 m/s in the camera field and **0.84 / 0.92
m/s in the fitted one**: the loss tears the rigid torso apart. In the render this is exactly
the smear of fig1: on frame 118 the torso+head ALONE spans Doppler -18..-8 (two peaks, -16 and
-12) where the measured torso is a 2-bin peak at -16/-15 and the slower groups (-13/-12 and
-10/-9, limbs in the slow phase of their swing, 220 + 220 of 1000) are as strong as the torso
(315). The within-limb gradient of the soft skinning is compressed too: the least-squares slope
of the skinned radial velocity along a bone over the rigid segment's span is 0.5 on the legs
and 0.1-0.4 on the arms (temperature 0.25 m on 0.25-0.45 m bones; the analytic factor for a
0.25 m bone is 0.24).

**(b) the limbs are dark, by two mechanisms.** Rendered share of the person's moving energy
(medians, 183): torso+head 0.70, upper arms 0.14, forearms/hands 0.08, thighs 0.02, shanks/feet
0.04 -- with the fitted joint gains already at 2.2 / 3.9 (R knee / ankle) and 0.13-0.8 on the
arms. (i) The per-point Monte-Carlo weight is cos/R^2 per VERTEX with no surface measure, and
SMPL's topology is far from uniform: head + hands + forearms hold 45% of the vertices on 20% of
the area, the thighs 6% on 17% (per-vertex area 1.2 cm^2 on the head and hands, 7.9 on the
thighs). (ii) Per unit (area x gain x weight), even the radar-facing leg points (cos > 0.9)
return 6-12% of the torso's facing points on frame 118 (71% on frame 61): the per-point
materials have gone bimodal -- 28-32% of the points specular (l_c 10-25 mm, sigma_h 1-10 um, a
KA lobe of alpha 0.2-0.4 = 10-20 deg; the legs, bright only inside their cone), the rest
diffuse (l_c ~2 mm, v_KA = sigmoid((k l_c - 6) 2) ~ 0.004, an SPM lobe of kappa ~0.6, near
isotropic; the torso, dim per steradian but bright everywhere). The point sum's coherence gain
|sum c_p|^2 / sum |c_p|^2 per range bin is 0.45 (random phase; ~137 paths a bin). The
measurement's limb/torso ratio is what continuous-surface physics predicts (a limb's specular
strip sqrt(lambda a / 2) ~ 1.2 cm x 40 cm ~ 50 cm^2 against the torso's Fresnel zone ~ 200
cm^2, times four limbs), and what a 3 cm vertex spacing with narrow per-point lobes cannot
sample.

**The velocity field is not what the model misreads (vcam, chain, cvs).** `vcam`, the field
frozen at the camera's (`--velocity-lr 0`, zero code): the probe is unchanged -- 183 MRPE 30.0
cm, bias (-11.4, +12.0, -0.5); 186 23.3, (-8.9, +13.8, +1.0) -- while the RD Pearson drops
(183 0.68 -> 0.50, 186 0.71 -> 0.55; the MD spread 0.50 -> 0.02: with the soft skinning's
compressed limb gradients the camera field renders too narrow a Doppler, which is what the free
field was smearing the torso to cover). `chain` (`--joint-kinematics chain`: the joint
velocities of rigid bodies on the kinematic tree, 36 dofs a frame, no bone stretches; the camera
field projects on it to 0.011 / 0.026 / 0.011 m/s rms on 183 / 186 / 256) keeps the field
physical -- torso spread 0.25 / 0.31 m/s (free: 0.84 / 0.92), joint correlations with the
camera 0.7-0.9 (free: 0.4-0.7) -- and leaves the probe at 28.3 / 25.3 (bias -10.9, +12.6, -5.0 /
-13.2, +9.9, -4.4); `cvs` (chain + `--person-vertex-area on` + `--person-material shared`)
28.0 / 25.6 (153: 14.0 / 11.7 against floorfix's 12.3). The bias is not in the Doppler
structure.

The person-model candidates against the gate (section 1's spread; the probe = the seed-4
real-trained model's MRPE on 183 / 186 / 153, floorfix 29.8 / 25.2 / 12.3):

| candidate | what | fidelity suite vs floorfix | proxies | probe | verdict |
|---|---|---|---|---|---|
| vcam | field frozen at the camera's | RD 183 0.68 -> 0.50, 186 0.71 -> 0.55, 153 0.63 -> 0.40; MD spread 0.50 -> 0.02 | HuPR 256 AP 22.8 -> 15.4 | 30.0 / 23.3 / 12.3 | diagnostic only |
| chain | rigid-body kinematic tree, 36 dofs | RA within 0.005; RAc -0.010 / -0.012 / -0.011 on 183 / 40 / 186 (spread 0.008); RD within 0.01; MD spread 256 0.74 -> 0.95, 186 0.42 -> 0.64, 183 0.50 -> 0.42 | HuPR 40 AP 21.2 -> 18.3 (std 2.8); RT-Pose MRPE 25.0 / 15.8 / 22.7 -> 23.3 / 15.6 / 23.0 | 28.3 / 25.3 / 14.0 | not adopted, code removed (its defect moves: torso spread 0.84 -> 0.25 m/s, joint corr 0.4-0.7 -> 0.7-0.9; nothing downstream) |
| varea | barycentric vertex area as the measure | HuPR RD -0.03..-0.06, RAc 256 -0.10; RT-Pose RD -0.02..-0.05 | HuPR 256 AP 22.8 -> 9.9 | 29.4 / 25.2 / 19.6 | rejected, code removed |
| shared | one material for the body | RT-Pose RAc +0.07..+0.12, RD +0.01..+0.03; HuPR within spread | HuPR 102 / 256 AP 16.2 / 22.8 -> 42.3 / 40.1 (!), 38 19.2; RT-Pose 153 MPJPE 14.9 -> 19.2, 108 438 -> 467 | 29.6 / 24.7 / 18.8 | not adopted (an existing flag); the HuPR AP jump is a lead for the HuPR arm, outside this loop |
| cvs | chain + varea + shared | HuPR RD -0.04..-0.06; RT-Pose RAc +0.05..+0.11 | HuPR 40 AP 27.7, 38 17.5, 102 13.3; RT-Pose 153 13.4 | 28.0 / 25.6 / 11.7 | rejected |
| lbs | rigid-body LBS velocity skinning (SMPL weights, the bodies' angular velocities from their joints) | RT-Pose RD -0.07..-0.09; HuPR within spread | RT-Pose 183 MRPE 25.0 -> 28.1, 153 15.8 -> 22.2 | 30.3 / 24.0 / 13.8 | rejected, code removed (its defect moves: limb slope / rigid 0.5 -> 1.0-1.2) |

**Where it is: the person's azimuth.** Core / halo swap in the model's input (2 m box about the
GT pelvis; core = |dx| <= 0.35 m, |dy| <= 0.30 m; `probe5.py`): rendered core + real halo **10.4
/ 7.4 cm, bias gone** (-0.3, +0.5, +1.5 / -1.2, -0.5, +2.4); real core + rendered halo 20.8 /
14.9 with the lateral bias (+11.9 / +8.2); the real with its moving halo zeroed 23.9 / 64.6. The
model reads what sits OUTSIDE +-0.3 m lateral of the label -- and that is where the real person's
return is. In the raw range-azimuth domain (`az_offset.py`, the renderer's own beamformer on
both sides, the moving-power azimuth peak at the person's rows), the real return is 3-5 degrees
from the rendered person on 183 / 186 -- a near-constant yaw per sequence, the same sign at
positive and negative azimuths (183: -4.7 +- 3.6 deg, -0.30 m at the person's range; 186: -4.0
+- 4.6, -0.21 m; 153: +1.1 +- 1.1, +0.12 m), uncorrelated with the lateral velocity (r = -0.05:
not the camera clock) and with the range (r = -0.05: not a translation). In the model's input
(`box_peak.py`, the moving return's peak cluster relative to the label): the real return at y =
-0.14 +- 0.17 m, x = -0.27 (183; 186: -0.14 / -0.30), the rendered at y = +0.02 +- 0.13, x =
-0.34 (+0.04 / -0.38) -- the render sits 0.16-0.17 m lateral and 0.07 m nearer of where the
radar sees the person, and the model, which learned the label's offset from the real return,
adds that offset to the render: the (-11, +12) cm bias. The static scene is placed right (the
DC amplitude at the wall behind the person 1133 real vs 1206 rendered); the person's placement
is the camera's -> LiDAR -> radar extrinsic's, registered per frame in RANGE against the
radar's moving-power peak (A.2.3) and never in azimuth. Against the LABELS (`az_label.py`,
the same centroid estimator) the measured return sits at +1.2 / +1.6 / +3.0 deg (183 / 186 /
153, std 1.3-3.0) and the floorfix render at +5.0 / +5.1 / +2.6 deg: the labels track the
radar to within the return's own offset from the pelvis, and the yaw is in the placed MESH
(the GVHMR mesh through the shipped stereo calibration, the image crop and
seeds/rtpose/radar_extrinsic.json), which the loop's label-based audit
(`fidelity_audit._geometry_vs_labels`) compares to the labels' 3D joints in metres, not in the
radar's angle, and so did not flag. (The commit message of a8e32b0 says the labels share the
extrinsic; they do not -- this paragraph is the correction.) Multipath is real but secondary:
the recording has ghosts of the moving person at the person's own Doppler behind it (+0.55..
+1.05 m and +1.9..+2.3 m when the person is at R >= 4 m, 34-106% of the peak in the moving
range profile; nothing in the render); image-antenna ghosts via the scaffold's floor and
ceiling (`ghost_probe.py`, gamma 0.5 / 1.0 in amplitude) remove the range bias (-10.9 -> -2.9
/ +2.8) and nothing else (MRPE 28.3 / 28.2), as 3.10's synthetic tail did.

**A latent bug on the way.** The beamformer's angle axis runs opposite to
`person_range_angle_mask_torch`'s `n/2 + n/2 sin` convention (the mesh at sine -0.405 lands in
mask bin 38 of 128, the real return in bin 80, the rendered person's own product in 90-94), so
F6's registration sector (`include_mirrored_angle=False`) sat on the mirrored side of boresight
and never contained the person: the range registration survived on the person's azimuth
sidelobes (a mover's sidelobes are still the strongest moving thing at the wrong angle). Fixed
(the sector on both mirrors, the flag's documented purpose); gated as `sector2` below.

**Candidate `azreg`: the azimuth registered to the radar as the range is**
(`reconstruction.radar_azimuth_yaw`, `--registration-azimuth on`, chain env
`REGISTRATION_AZIMUTH`): per frame asin(radar moving-power azimuth at the registered range) -
asin(scaled mesh centroid azimuth) in the beamformer's spatial sine (the sign checked on the
data: median |radar - mesh| 0.08 with it, 0.4-0.6 mirrored), the depth registration's schedule
(confidence gate, median +-7, IQR flag, Savitzky-Golay 51), applied as a rotation about the
radar's vertical through its centre; the pixels stay the camera's. Three estimators of the
radar's azimuth, RT-Pose only, seed-4 probe (MRPE cm, bias fwd / lat / up):
* inside the one-sided sector (the sector clips it): yaw -1.5 deg on 183, nothing changes;
* the parabolic peak (`azreg2`): yaws 183 -5.5, 186 -4.7, 153 +1.2, 108 +4.4 deg; the real -
  rendered peak offset goes to +0.9 / +0.3 / +0.3 deg -- but the probe: 183 35.7 (-4.4, -18.5,
  -4.7; lateral std 50), 186 **17.4** (-5.6, -5.3, -1.6), 153 16.5 (-2.6, +10.6, +2.3): the peak
  over-registers. Measured with centroids of increasing width (`az_cen.py`, +-2 / 5 / 12 / 24
  bins of 128), the recording's return has a lateral pedestal the peak ignores: floorfix's
  offset is -4.7 deg by the peak and -4.0 by the wide centroid on 183 (186: -4.2 / -3.4; 153:
  +1.2 / +0.1), and the model's zero-bias point follows the wide centroid (183: bias +12 cm at a
  -0.25 m centroid offset, -18.5 at +0.10);
* the floor-removed energy centroid over +-0.19 in sine about the peak (`azreg3`, the
  estimator kept): yaws 183 -4.1 deg (IQR -6.1..-2.1), 186 -2.7, 153 +0.4, 108 +3.3; residual
  after +0.02 +- 1.3 deg; the probe 183 27.4 (-5.5, -5.3, -7.4), 186 **19.2 (-5.3, +0.4, -0.8)**,
  153 15.7 (-3.5, +6.5, +2.1): the lateral bias is gone on both, the range bias halved;
* the same yaw applied as ONE rotation per sequence, the median over confident frames (`azreg4`,
  the form kept: the offset is the extrinsic's, and the smoothed per-frame series still wanders
  +-1.3 deg = 8 cm at 3.5 m): 183 28.7 (-9.7, -7.9, -6.7; std 13 / 16 / 17), 186 **15.7 (-9.1,
  +0.8, -1.0; std 8 / 10 / 10)**, 153 15.1 (-3.6, +6.0, +1.9).

**The gate (`azreg8`: the eight sequences, the person stage rebuilt with
`REGISTRATION_AZIMUTH=on`, five-seed proxies).** Yaws found (median over confident frames): HuPR
38 +1.8 deg, 102 +4.5, 256 +2.5, 40 +1.3 (358 of 600 frames detected: the still person);
RT-Pose 183 -4.1, 153 +0.4, 186 -2.7, 108 +3.3 -- HuPR's co-boresighted camera has the same
kind of yaw. Against floorfix: RA within 0.007 everywhere; RAc 38 0.460 -> 0.558, 183 0.473 ->
0.540, 186 0.527 -> 0.562, 108 0.000 -> 0.190, and 102 0.810 -> 0.798, 256 0.743 -> 0.729 (the
one item beyond the 0.008 spread); RD 38 0.698 -> 0.745, 40 0.332 -> 0.487, 108 0.067 -> 0.125,
the rest within 0.013; MD centroid / spread / energy within spread except 40's centroid 0.13 ->
-0.21 (a near-zero centroid's sign) and 40's part energy ratio 3.0 -> 8.3. Proxies: **HuPR AP 38
15.0 -> 29.7, 102 16.2 -> 43.8, 256 22.8 -> 43.3, 40 21.2 -> 33.2** (seed std 2.1-5.5; elbows 4.4
/ 7.9 / 7.4 / 11.6 -> 7.7 / 27.1 / 23.6 / 15.2); RT-Pose five-seed MRPE 183 25.0 -> 27.2 (std 15),
153 15.8 -> 17.8 (std 3.8), **186 22.7 -> 16.8**, 108 439 -> 289; the seed-4 probe 183 30.3
(-8.3, -8.4, -7.2), 186 16.5 (-9.6, +0.4, -1.8), 153 13.1 (-3.0, +2.9, +1.8). Accepted: the
defect it was built for is gone (the lateral bias on 183 / 186, the yaw on every sequence),
nothing worsens beyond the spread except RAc on 102 / 256 by 0.004-0.006 over it, and the
HuPR arm gains 2x in AP. `REGISTRATION_AZIMUTH=on` is the person stage's default from this
commit; `off` reproduces the camera's azimuth for an ablation.

**What is left, and where it is.** On the registered render the core / halo split is clean:
rendered core + real halo **8.5 / 7.4 cm** (std 4 / 8 / 6 and 2.5 / 7 / 4) -- the person's own
return, once its azimuth is the radar's, is at the 2x-real target; real core + rendered halo
13.5 / 10.3; both rendered 27.4 / 19.2 (`probe5.py` on azreg3). The halo's value is in its MOVING
bins (`probe6.py` on azreg4: rendered + the real halo's moving bins 20.7 / 11.7, its static bins
28.7 / 15.7 = nothing) and it sits BESIDE and IN FRONT of the person, not behind it (real
beside only, |dy| > 0.3 m at the person's rows: 23.5 / 13.0; real in front only, dx < -0.35 m:
26.2 / 12.4; real behind only, dx > 0.35 m: 28.0 / 15.6 = nothing). Moving energy at the person's
own range rows beyond +-0.3 m lateral and 0.35 m ahead of the pelvis is the swinging arms and the
leading leg: the limbs, which the render carries at 2-14% of the torso's brightness per facing
point (above). The multipath ghosts behind the person are real (the range-profile tails above,
the +9..+15-bin hump at the wall) but the model does not read them: image-antenna ghosts via the
scaffold's floor and ceiling on the registered render (`az3_ghost05`, gamma 0.5) put the render's
"beside" amplitude at 60-107 against the real 70-149 and "behind" at 32-58 against 8-101, remove
the range bias (-5.5 -> +1.3, -5.3 -> +1.8) and move the MRPE 27.4 -> 26.1, 19.2 -> 17.7 only.

Figures: /home/adnan/Desktop/mmSD/artifacts/rtpose_now/ (`viz_now.py`, `viz_pred.py` on the node):
fig1 range-Doppler measured / before / now, fig2 the same in range-azimuth with the labelled
pelvis, fig3 the azimuth track per frame (real - rendered: 183 before -3.7 deg / -0.23 m, now
-0.2 deg / 0.00 m), fig4 the seed-4 model's predicted root per axis per frame on the real cube
and both renders, fig5 the MRPE bars.

**Conclusion.** Two things, in order: (1) the person is placed 3-5 degrees from where the
radar sees it on 183 / 186 (the mesh placement's yaw; the labels are within 1-3 deg of the return) -- the azimuth registration (one yaw per
sequence, the radar's) takes 186 from 25.2 to 15.7 cm with the bias vector gone and 183's bias
from (-11, +12) to (-10, -8) cm; (2) what remains on 183 (scatter 13-17 cm per axis) is the
rendered limbs' brightness -- the moving energy beside and ahead of the pelvis that the real
cube has and the render lacks -- and not the Doppler smear, not multipath, not the level. The
limb-brightness item is the person model's (the per-point lobes on a 3 cm vertex spacing, the
per-point material bimodality); `shared` moved the RT-Pose RAc the right way (+0.07..+0.12) and
HuPR 102 / 256's AP from 16 / 23 to 42 / 40 but fails elsewhere, so the physical form (a facet
lobe set by the vertex's own size, one material) is the next candidate, not a flag.

### 3.12 The limb geometry: the depth registration resized the body (2026-09-17, tr8) -- accepted

Asked whether the rendered limbs are the right size (`limb_geom.py`: the registered mesh's 14
joints against the RT-Pose labels' 15, bone by bone, per labelled frame): on azreg8 the RT-Pose
bones were 1.0-1.1x the labels' on 183 / 186 (k = 1.05 / 1.07) and 1.2-1.3x on 153 / 108 (k =
1.10 / 1.11) -- legs 1.16-1.31, forearms 1.11-1.25 -- and **the HuPR person was 1.07 m tall**
(head-to-ankle; thigh 0.28 m, shoulder width 0.28 m) where GVHMR's raw body is 1.8 m: the depth
registration (A.2.3) applied the radar / GVHMR range ratio as a SCALE about the camera origin
("pixels preserved, a body k times larger and farther"), and HuPR's ratio is 0.71. GVHMR's error
is its depth (the focal length it assumes), not the body's size, so the body now keeps its size
and moves along its centroid's camera ray to k times its range (`--depth-registration translate`,
chain env `DEPTH_REGISTRATION`; `scale` is the old form, an ablation; only the diagnostic video's
overlay read the projected joints). After: HuPR head-to-ankle 1.46-1.52 m, thigh 0.39-0.41 m,
shoulders 0.37-0.40 m; RT-Pose legs 0.9-1.15x and forearms 0.9-1.1x the labels' (the body
0.83-0.92 of the labels' head-to-ankle: GVHMR's RT-Pose bodies are ~10% short, and the label's
head is not SMPL's head joint), hip width 0.6-0.85 and neck-head 0.33 by joint convention.

Gate (tr8 vs azreg8, eight sequences, the person stage rebuilt, five-seed proxies): HuPR RAc
0.558 / 0.798 / 0.729 / 0.211 -> 0.591 / 0.811 / 0.746 / 0.181, RD 0.745 / 0.904 / 0.922 / 0.487
-> 0.761 / 0.915 / 0.927 / 0.494, MD within spread, 40's part energy ratio 8.3 -> 1.6 (the shrunken
body was that outlier); **AP 29.7 / 43.8 / 43.3 / 33.2 -> 34.3 / 46.9 / 45.0 / 40.9** (std
1.0-5.7; floorfix 15.0 / 16.2 / 22.8 / 21.2). RT-Pose RA / RAc / RD within the spread (183 0.876 /
0.538 / 0.685, 186 0.870 / 0.560 / 0.705, 153 0.913 / 0.514 / 0.644); MRPE 183 27.2 -> 26.3, 153
17.8 -> 15.3, 186 16.8 -> 16.7, 108 289 -> 157 (108's suite NaNs: the pose-speed selection keeps 2
of its frames); the seed-4 probe 183 29.5 (-9.3, -7.5, -7.4), 186 17.8 (-9.6, -0.7, -2.6), 153
13.5 (-2.9, +3.5, +2.1). Accepted; the one item outside the spread is HuPR 40's RAc (-0.03 on a
still person's 0.2).

The limb brightness is not the geometry: with the right limb sizes the rendered legs are still
0.07-0.09 and the arms 0.07-0.26 of the person's moving energy (183 / 186), and the real halo's
moving bins are still worth 29.5 -> 21.3 / 17.8 -> 12.0 cm to the model (`probe6.py` on tr8).
**The limbs' brightness, three candidates on the corrected geometry** (the seed-4 probe 183 /
186 / 153 and the five-seed RT-Pose MRPE; tr8 is the base: probe 29.5 / 17.8 / 13.5, MRPE
26.3 / 15.3 / 16.7):

| candidate | limb share of the moving energy (183: torso / thighs / shanks / upper arms / forearms) | RT-Pose RAc 183 / 153 / 186 | probe | five-seed MRPE 183 / 153 / 186 | HuPR AP 38 / 102 / 256 / 40 |
|---|---|---|---|---|---|
| tr8 | 0.55 / 0.04 / 0.05 / 0.13 / 0.13 | 0.538 / 0.514 / 0.560 | 29.5 / 17.8 / 13.5 | 26.3 / 15.3 / 16.7 | 34.3 / 46.9 / 45.0 / 40.9 |
| sh8, one material (`--person-material shared`) | 0.30 / 0.13 / 0.08 / 0.24 / -- | 0.633 / 0.579 / 0.665 | 26.1 / 15.8 / 21.1 | 23.7 / 16.4 / 15.8 | 34.0 / 43.7 / 48.3 / 45.0 |
| sg8, one material + gains fixed at 1 | 0.67 / 0.04 / 0.02 / 0.22 / -- | 0.634 / 0.579 / 0.666 | 28.7 / 17.5 / 12.2 | 26.4 / 13.5 / 16.9 | 32.7 / 50.3 / 47.1 / 43.9 |
| **sao8, one material + `--person-solid-angle-weight off`** | **0.27 / 0.24 / 0.07 / 0.25 / 0.23** | **0.645 / 0.590 / 0.678** | **25.5 / 15.4 / 12.5** | **23.3 / 13.9 / 15.7** | 33.5 / 45.8 / 46.0 / 43.3 |

One material (sh8): the body goes fully diffuse (KA 0, SPM 1.0), the limbs come up on 183, the
RT-Pose RAc +0.1 on every sequence, HuPR within the spread; accepted (b52eb05). With the gains
fixed too (sg8) the limbs go dark again (the fit turns them off through the material's level
instead) and the proxies drop back, so the gains stay learned. The cos/R^2 Monte-Carlo weight
(`surface_solid_angle_weights_torch`) is the depth camera cloud's -- its sampling density
follows the view -- and on the SMPL mesh it is one more cosine on every oblique surface, the
limbs' sides; off (sao8), the 183 limbs carry 0.73 of the moving energy (the recording's
split), RA / RAc / RD rise on both datasets (HuPR 102 RD 0.911 -> 0.927, 256 0.926 -> 0.937,
38 RAc 0.665 -> 0.668; RT-Pose RD +0.01..+0.02), the proxies within std or better; accepted
(4f9df3f). On 186 the learned gains still keep the torso at 0.88 of the energy; its probe is
15.4 regardless. The seed-4 bias vectors now: 183 (-7.6, -6.4, -6.0), 186 (-8.2, +0.7, -1.8),
153 (-2.5, +4.1, +1.5); the range component (-8 cm) is what remains on both walking sequences,
with the multipath tail (3.11) as its identified source.

### 3.13 The defaults, end to end, and the real reference (2026-09-17, def8)

`def8`: `chain run`'s defaults with no flag and no env on the eight sequences, the person
stage rebuilt -- the registration sector on both mirrors, the azimuth registration (one yaw
per sequence), the translation depth registration, one material for the person, the person's
cos/R^2 weight off (the trainer and the render argv both carry `--person-solid-angle-weight
off`; the trainer's material is the shared six-vector). Against sao8 everything is within the
run-to-run spread: RA 0.953 / 0.960 / 0.959 / 0.952 (HuPR), 0.894 / 0.909 / 0.892 / 0.842
(RT-Pose); RAc 0.669 / 0.831 / 0.764 / 0.208 and 0.651 / 0.585 / 0.681; RD 0.769 / 0.925 / 0.938
/ 0.467 and 0.706 / 0.670 / 0.742 / 0.271; no NaN except 108's micro-Doppler and clutter-removed
statistics (2 motion frames); the RD signal power ratio rendered / measured 0.44-0.76 (HuPR),
0.38-0.44 (RT-Pose), the floor ratio 0.94-1.02. Proxies (five seeds): HuPR AP 32.7 / 46.4 / 47.3
/ 45.0; RT-Pose MRPE 24.9 / 17.2 / 16.4 / 251 (MPJPE 23.0 / 17.5 / 26.8 / 270); the seed-4 probe
183 27.4 (-8.3, -7.3, -5.5), 186 16.1 (-8.8, -0.4, -2.1), 153 23.6 (a lateral std of 58 cm:
outlier frames; sao8's 12.5).

The reference the proxies are measured against, the same five real-trained models on the REAL
cubes of the same sequences (`proxy --domain real`): HuPR AP 38 55.6 +- 3.2, 102 79.9 +- 1.6,
256 67.2 +- 1.8, 40 52.9 +- 4.5; RT-Pose MRPE 183 4.6 +- 0.5, 153 8.0 +- 0.3, 186 4.5 +- 0.3,
108 10.4 +- 0.9 (MPJPE 7.7 / 11.3 / 7.1 / 16.2). So the R -> S column now stands at 0.6-0.85 of
the R -> R AP on HuPR (from 0.2-0.35 with the floorfix renders) and at 2-5x the R -> R MRPE on
RT-Pose (183 5x, 186 3.6x, 153 2x; from 5-6x). Not on par: what remains on RT-Pose is the
frame-to-frame scatter of the real-trained model's root on the render (per-axis std 8-19 cm on
183 / 186 against 1-4 cm on the real cube), whose carrier is the person's moving return beside
and ahead of the pelvis (3.11); the mean bias is now within 9 cm on every axis.

### 3.14 The geometry, taken to the end (2026-09-18): the lateral scale, the elevation, the pose

**The labels' frame is scaled laterally per recording session.** Against the labels, in the
azimuth-only beamformer on the real cubes of every walking sequence (r >= 0.95; `az_all.py`),
sin(real return) = a + b sin(label): Indoor-Clean (178-192) b = 0.92-0.99, a = +0.01..+0.04;
Indoor-Cluttered (10-19, 123-152) b = 1.02-1.19, a = +0.05..+0.10; Outdoor (111-113, 153-172) b
= 1.01-1.14, a = +0.05..+0.07. The mesh inherits the annotation frame through the camera ->
LiDAR -> radar chain (rendered - label: no scale, slope -0.02 / +0.02 on 183 / 186), the real
return carries the session's scale, and a yaw alone leaves it: in the model's input real_y =
0.909 / 0.916 / 0.914 x rendered_y on 183 / 186 / 153 (`az_scale.py`, def8). The registration
is now the affine map sin_radar = a + b sin_mesh per sequence (`radar_azimuth_yaw`, b only
where the mesh spans azimuth: sine std >= 0.10 and r >= 0.90, else the yaw), applied per frame
as a placement: 183 -0.050 + 0.906 sin (1.6 deg rms), 186 -0.051 + 0.934 (3.0), 153 +0.013 +
1.037 (1.0); HuPR and 108 yaw-only (a sine std of 0.002-0.08 -- the scale is unidentifiable
and, unguarded, ran to the clip). Gate (aff8 vs def8): suite within spread; HuPR AP 34.3 / 44.7
/ 46.0 / 42.1; RT-Pose MRPE 24.7 / 18.2 / 16.2; probe 183 27.4 -> 25.8 (-5.1, -4.9, -6.3), 186
16.1 -> **13.9** (-5.5, -2.6, -0.9); real_y / rendered_y 0.973 / 0.975 on 183 / 186 (153 went
0.914 -> 0.895: its beamformer and model-input scales disagree by 7%, the elevation coupling of
the adapter's 2D angle processing). Accepted (d8d4e7c). The range: the mesh pelvis sits 29 / 27
cm in front of the label on 183 / 186 (-2 cm on 153) because the registration aligns it to the
measured moving-power peak, and the labelled pelvis is 0.29 / 0.24 / 0.07 m behind that peak
per session -- the mesh's own predicted return peak (its visible moving vertices' brightness-
weighted range histogram) is only 5-13 cm ahead of its pelvis and does not track the label's
offset per frame (r <= 0.3), so this too is the labels' frame, which the real-trained model
learns per environment; the mesh follows the radar, as it must.

**Where the mesh stands against the labels (aff8).** Pelvis: bias (-29, +1, -5) / (-27, +3, -4)
/ (-2, +29, -6) cm on 183 / 186 / 153 = the sessions' label offsets; the per-frame deviation is
SLOW (a 9-frame mean carries 7-14 cm, the fast residual 1.5-4 cm, frame-to-frame 1.5-2 cm): no
placement jitter. Pose, relative to the pelvis (mesh - label std): hips / knees / neck 3-6 cm,
ankles 10-14, elbows 5-21, wrists 6-31 (186's left arm). Timing: the real and rendered
per-frame return deviations correlate at lag 0 (r 0.6-0.8, flat over +-4 frames): the camera-
radar pairing is right.

**Elevation.** In the adapter's polar tensor the person's moving-energy elevation profile: on 153
real and rendered agree (peak bin 16.2 vs 16.3, r 0.87); on 183 / 186 the REAL profile is flat
(uniform 3% per bin, peak wandering 11 +- 6.5 bins) and the render focused (bin 18-19 +- 1-2).
On the real cubes alone (`elev_real.py`): every Indoor-Clean sequence (178, 183-188) is
unfocused for the person AND for the strongest static reflector (peakiness 1.3-1.5, static peak
+-3 bins) while every other session is focused (static peakiness 1.9-2.4 at bin 17 +- 0); the
inter-TX phases of the strongest static cell wander 10-45 deg frame to frame on that session
(178: 27-94) against < 2 deg elsewhere (`elev_phase.py`). A receiver/TDM phase instability of
that recording session, azimuth TX included -- a property of the hardware run the real-trained
model has seen in every cube of the environment. Probes on aff8: the rendered tensor with the
real cube's z profile per column 25.8 -> 20.3 / 13.9 -> 12.1; the real cube with the rendered
z profile 4.3 -> 9.3 / 4.9 -> 6.5 (`probe7.py`): the elevation is worth ~5 / ~2 cm. A per-frame
per-TX phase jitter on the rendered channels with sigma from the recording's static scene
(15-33 deg; `txjit_probe.py`) changes nothing (26.6 / 13.6 / 25.0): the static-scene estimate
is far below what a flat elevation needs, and I did not inflate it.

**The pose (an experiment, not a production path; removed after).** The mesh's limbs warped
onto the labelled 3D joints per frame (skinned offsets relative to the pelvis, the placement the
radar's; median |offset| knees 4-11 cm, ankles 10-14, wrists 7-22): probe 183 28.0, 186 16.8,
153 29.5 (aff8 25.8 / 13.9 / 20.8); five-seed MRPE 25.4 / 16.0 / 19.6 (24.7 / 16.2 / 18.2). No
gain: with placement AND pose at the labels' accuracy the real-trained model scatters the same.

**Conclusion of the geometry track.** Placement (range, yaw, lateral scale, body size) and pose
are now at the accuracy the labels can certify, the mean biases are <= 6 cm on every axis, and
the frame-to-frame scatter of the real-trained model's root on the render (std 12 / 16 / 18 cm on
183, 6 / 11 / 10 on 186) does not move with any further geometric correction, including the
labels themselves. What the model reads per frame beyond position is the person's return
content -- its elevation structure on the degraded session (~5 cm) and the rest in the range /
azimuth structure of the moving return (3.11: the real halo's moving bins are worth 29.5 ->
21.3 / 17.8 -> 12.0) -- which the single-bounce point model with one material and GVHMR's
surface does not reproduce frame by frame. That is not a geometry-initialisation item.

## 4. What ships, and what it costs

Since commit 7aeccda the accepted configuration is the DEFAULT of `chain run` (no flag needed):
the trainer's `--training-array full`, `--elevation-term on`, `--elevation-planes motion`, the
radars' `training_n_angle` (HuPR 16, cascade 86; `Radar.training_n_angle`), the person stage's
`--registration-sector on` (chain env `REGISTRATION_SECTOR`, default on; on both mirrors of the
beamformer's angle axis since 3.11), its `--registration-azimuth on` (chain env
`REGISTRATION_AZIMUTH`, default on since 3.11: one yaw per sequence to the radar's moving
return), its `--depth-registration translate` (chain env `DEPTH_REGISTRATION`, default since
3.12: the body keeps GVHMR's size), the trainer's `--person-material shared` and
`--person-solid-angle-weight off` (chain env `PERSON_SOLID_ANGLE`, default off since 3.12, the
render argv carries it), and the fused full-array gather. Verified end to end with no flag and
no env on the eight sequences (3.13, `def8`: the person stage rebuilt, every metric within the
spread of sao8). The ablations: `REGISTRATION_SECTOR=off`, `REGISTRATION_AZIMUTH=off`,
`DEPTH_REGISTRATION=scale`, `PERSON_SOLID_ANGLE=on`, `--person-material auto`. (The earlier
`defaults` check of 2026-09-16 on 256 / 183: RA 0.956 / RD 0.921; 183: RA 0.866 / RD 0.645.) The rejected F2 (`--ra-loss l2-range`),
F4 (`--joint-gain-prior`) and the model Doppler support are removed from the code (7aeccda);
`--elevation-term off --training-array azimuth --n-angle 64|128` and `REGISTRATION_SECTOR=off`
reproduce the pre-2026-09-16 trainer for an ablation.

Runtime on the node (velocity stage: 400 epochs + render + score, one sequence per H100):

| | HuPR, per sequence | RT-Pose, per sequence |
|---|---|---|
| baseline | 272-290 s (16.4-17.2 ms / step) | 477-519 s (10.7-12.0 ms / step) |
| + F1 all planes | 415-428 s (30.0 ms) | 979-1069 s (28.1 ms), 30 GB |
| + F1 motion planes | 346-391 s (23.0 ms) | 920-1015 s (26.4 ms), 27 GB |
| + fused gather, n-angle 16 / 86 | 312-317 s (18.9-19.7 ms) | 806-852 s (21.5-23.0 ms) |
| **final = the defaults** (+ the sector; the person stage +11-17 s) | **298-312 s** (+8%) | **802-842 s** (+63%) |

Where the RT-Pose step goes (seq 183, steady state, 20-epoch fits of the same code): azimuth
array at 128 bins 11.3 ms; at 86 bins 11.2 (the FFT size is free on the cascade); the full
12-TX array without the elevation term **13.4 ms (+2.1, the third more paths)**; with the
elevation term **22.0 ms (+8.6)**. The term's plane part runs on 53 of the 63 planes on the
cascade -- its micro-Doppler band is 53 bins wide (HuPR's is 16), so `--elevation-planes motion`
narrows almost nothing there -- over 244 range rows x 86 x 8 cells per unit, four units a step,
as eager op-by-op tensor passes (scatter to the 2D grid, layout copies, two FFTs each with an
fftshift roll, abs, the L2), forward and backward. Section 6 has the plan to remove that cost.

## 5. Next

1. Re-render both datasets with the accepted defaults and retrain the downstream arms:
   `downstream/tools/rerender_and_retrain.sh` on the node (section 5.1). Every improvement of
   sections 3.11-3.14 is the default path (`--registration-sector on`, `--registration-azimuth on`
   with the affine map, `--depth-registration translate`, `--person-material shared`,
   `--person-solid-angle-weight off`, also the video config's default), so the script passes no
   flag beyond `chain.PRODUCTION_TRAINER_FLAGS` and rebuilds the person stage from the GVHMR dump.
   The S -> S per-keypoint AP and the RT-Pose S -> R / R+S -> R numbers are the acceptance test
   the loop cannot run.

### 5.1 The re-render procedure (2026-09-17)

`chain run` needs Depth-Anything and GVHMR, which the node does not have, so the reference
workspaces are built on the workstation and everything else runs on the node
(`downstream/tools/rerender.py`, both halves):

    # workstation (4DPS-fidelity checkout; ~15 s a sequence, 3 workers a 4090)
    export DOWNSTREAM_PRODUCTS=/home/adnan/Documents/Data/4DPS_products   # the previous sweep's GVHMR dumps
    python -m downstream.tools.rerender ref --dataset rtpose --workers 3 --gpu 1
    python -m downstream.tools.rerender ref --dataset hupr   --workers 3 --gpu 0
    rsync -a outputs/rerender_hupr outputs/rerender_rtpose nvidia@mmsd-h100:/data/mmsd/4DPS/outputs/
    # node
    nohup downstream/tools/rerender_and_retrain.sh > /data/mmsd/logs/4dps/rerender_and_retrain.log 2>&1 &

The script: (1) every sequence of both manifests (RT-Pose first), one per GPU on all eight,
`rerender prep` (the workspaces' absolute links re-pointed at the node checkout) ->
`devloop run --rebuild-person` -> `rerender install` (products into `$DOWNSTREAM_PRODUCTS`,
the previous sweep's folder to `/data/mmsd/derived/4DPS_products_floorfix`, the manifest row
rewritten, `.rerender` marks the commit; re-runs skip installed sequences); (2) the rendered
caches re-materialised for the split; (3) the previous rendered / mixed arms and the real arm's
rendered-domain evaluations archived to `downstream/results_floorfix`, the blindness gate,
`runner run --arms real rendered mixed` (HuPR GPUs 0-4, RT-Pose 5-7 two per card), `runner
matrix`, `results`.

Smoke test on the node (RT-Pose 183, GPU 4, commit dab5b95 + the tools): person stage 12 s
(affine `sin_radar = -0.0497 + 0.9055 sin_mesh`, residual 1.64 deg rms -- the same fit as
aff8's), velocity field 798 s (400 epochs, 21.9 ms a step, 20.3 GB peak), evaluation:
RA r 0.900 / RAc 0.660 / RD-occupied r 0.712 / MD spectrogram r 0.946 against aff8's
0.894 / 0.652 / 0.716 / 0.951 (the same pipeline; the trainer's seed noise). Manifest row
written, products installed, the previous folder archived. Two-person sequences (RT-Pose 100, 101, 105, 106, 118 in the first hour; the dataset's 100-105,
117-140, 170-175 record two people and GVHMR tracks one): the azimuth registration's sign check
found the mirrored convention fitting better (100: median |sine| residual 0.543 with the array's
sign, 0.247 mirrored) and aborted the person stage. The check now skips the registration for the
sequence instead of aborting -- the registration corrects the extrinsic's yaw and needs a return
attributable to the mesh's person; the mesh keeps the camera's azimuth and the metadata records
why (`azimuth_registration: "off: ..."`). One-person sequences are unaffected (every one of them
passed the check). The five were re-queued into the running script, whose install marker still
reads b7bc875; their products are the only ones made with the later commit.

Estimated wall for phase 1 at
these speeds: 101 x 800 s + 235 x ~350 s over 8 GPUs = ~5.7 h; phase 3 ~14 h (two HuPR
rounds of ~7 h on five cards, RT-Pose two rounds of 5.4 h on six slots).

1b. The rendered limbs' brightness (3.11): the moving energy beside and ahead of the pelvis
   that the recording has and the render carries at 2-14% of the torso's per facing point; the
   real halo's moving bins are worth 28.7 -> 20.7 cm on 183 and 15.7 -> 11.7 on 186 to the
   real-trained model on the registered render. A person-model change: the per-point lobe set
   by the vertex's own size (the facet's physical-optics width, lambda / spacing, in place of
   the sigma_h-driven GGX alpha on a 3 cm spacing) with one material for the body; `shared`
   alone moved RAc +0.07..+0.12 and HuPR 102 / 256 AP to 42 / 40 but fails elsewhere.
2. The 183 registration dip (3.5): a hysteresis on the sector peak across frames, if strictness
   is wanted.
3. 108 (rain): a sequence the fit cannot lock on; report it as such or drop it from the render
   set on the detector's evidence (2 motion frames).

### 5.2 The re-render's fidelity against the first sweep (2026-09-18, phase 1 complete)

Every product's `metrics.json` (the suite of docs/METRICS.md), the re-render (commit b7bc875 /
16cf551) against the first sweep's products (docs/SWEEP_RESULTS.md, `main` e5f30d4, archived
at /data/mmsd/derived/4DPS_products_floorfix; its medians reproduce that document's tables:
HuPR RA 0.926 / RAc 0.568 / RD 0.762 / MD 0.980, RT-Pose 0.853 / 0.279 / 0.592 / 0.930).
Median (IQR) over products (a HuPR sequence once per board); the paired delta's median and
the share of products that improved. Script: /data/mmsd/logs/4dps/fid_compare.py.

| HuPR, 470 products | re-render | first sweep | paired delta | improved |
|---|---|---|---|---|
| range-azimuth Pearson | **0.934** (0.919-0.953) | 0.926 (0.909-0.948) | +0.006 | 92% |
| RA clutter-removed (the attributable part) | **0.628** (0.439-0.709) | 0.568 (0.352-0.657) | +0.061 | 90% |
| RA SSIM / NRMSE | 0.739 / 0.353 | 0.733 / 0.372 | +0.004 / -0.015 | 79% / 91% |
| range-Doppler occupied Pearson | **0.820** (0.716-0.878) | 0.762 (0.643-0.847) | +0.047 | 95% |
| micro-Doppler spectrogram / energy Pearson | 0.981 / 0.971 | 0.980 / 0.957 | +0.003 / +0.007 | 68% / 63% |
| micro-Doppler centroid / spread Pearson | 0.942 / 0.880 | 0.969 / 0.891 | -0.015 / -0.007 | 30% / 43% |
| micro-Doppler clDice / SSIM | 0.815 / 0.976 | 0.821 / 0.977 | -0.002 / 0.000 | 44% / 51% |

By board: horizontal RA 0.953 (was 0.948), RAc 0.669 (0.611), RD 0.875 (0.843); vertical RA
0.920 (0.909), RAc 0.578 (0.510), RD 0.766 (0.718). Thresholds: RA >= 0.8 100% (100%), RD >= 0.5
91% (88%), MD spectrogram >= 0.8 98% (98%), MD energy >= 0.8 97% (98%). Largest single drops:
74 (both boards) MD spectrogram -0.10/-0.19 and RD -0.15/-0.17, 6 vertical MD -0.12, 36
horizontal -0.10; largest RA drop 185 vertical -0.039. The one metric that moved down as a
group is the micro-Doppler centroid (median -0.015, 30% improved): the shared material and the
registration change the limbs' Doppler weight against the torso's.

| RT-Pose, 101 products | re-render | first sweep | paired delta | improved |
|---|---|---|---|---|
| range-azimuth Pearson | 0.841 (0.796-0.888) | 0.853 (0.808-0.885) | -0.007 | 36% |
| RA clutter-removed | **0.361** (0.249-0.536) | 0.279 (0.162-0.428) | +0.075 | 93% |
| RA SSIM / NRMSE | 0.816 / 0.490 | 0.829 / 0.477 | -0.019 / +0.010 | 14% / 37% |
| range-Doppler occupied Pearson | **0.628** (0.545-0.728) | 0.592 (0.468-0.666) | +0.046 | 87% |
| micro-Doppler spectrogram / energy Pearson | 0.938 / 0.886 | 0.930 / 0.876 | +0.008 / +0.007 | 77% / 55% |
| micro-Doppler centroid / spread Pearson | 0.940 / 0.657 | 0.919 / 0.675 | +0.003 / -0.005 | 59% / 48% |
| micro-Doppler clDice / SSIM | 0.748 / 0.881 | 0.735 / 0.904 | +0.011 / -0.012 | 64% / 29% |

One-person sequences (65): RA 0.877 (was 0.865), RAc 0.449 (0.380), RD 0.683 (0.635), MD
spectrogram 0.945 (0.940), MD energy 0.904 (0.904), clDice 0.762 (0.749); two-person (36):
RA 0.823 (0.830), RAc 0.270 (0.196), RD 0.527 (0.468), MD spectrogram 0.914 (0.900).
Thresholds: RA >= 0.8 72% (was 76%), RD >= 0.5 80% (69%), MD spectrogram >= 0.8 98% (98%),
MD energy >= 0.8 82% (77%). Largest drops: RA 1 -0.129 (0.796 -> 0.667), 150 -0.077, 174
-0.056; MD spectrogram 167 -0.102; RD 157 -0.287 (a still person: 0.181 -> -0.106 on a handful
of frames). The full-image RA Pearson and SSIM slip by 0.007 / 0.019 while the clutter-removed
RA (the person's own return) gains 0.075: the whole-image statistic is dominated by the static
scene, which the person's material no longer trades against (one shared 6-vector), and the
translated (unscaled) body is the right size but sits with the registered azimuth, so the
static-scene part of the image loses a little of the compensation it had.

Reading: no regression on the attributable metrics (clutter-removed RA and RD occupancy up on
87-95% of the products of both datasets, micro-Doppler spectrogram up on 68-77%); the
whole-image RA on RT-Pose and the HuPR micro-Doppler centroid are the two statistics that
moved down, both by less than their IQR.

### 5.3 Downstream, round 1 of phase 3 (interim, 2026-09-18 18:50 UTC)

The first round of cells (HuPR rendered seeds 0-2, mixed 0-1, 200 epochs; RT-Pose rendered 0-2,
mixed 0-2, 50 epochs), every real-arm checkpoint re-evaluated on the new rendered test set. Mean
+- std over the seeds run; "first" = the first sweep's arms (downstream/results_floorfix; RT-Pose
had no S / R+S arms then, its rendered products having been held). Scripts:
/data/mmsd/logs/4dps/{ds_compare,rt_seq}.py.

HuPR (AP x100; R -> R 65.9 +- 1.5, AP50 97.0, AP75 78.8, AR 75.3, PCKh 44.1):

| arm | AP now | AP first | delta to R -> R now (first) | AP50 | AP75 | AR | PCKh |
|---|---|---|---|---|---|---|---|
| S -> R (3 seeds) | **39.6 +- 1.7** | 15.8 +- 1.8 | -26.3 (-50.1) | 85.3 (51.2) | 28.1 (4.5) | 51.0 (28.1) | 26.1 (10.2) |
| R -> S (5) | **36.9 +- 0.9** | 22.0 +- 3.4 | -29.0 (-43.9) | 81.8 (63.6) | 24.7 (7.9) | 48.1 (33.9) | 21.9 (12.3) |
| S -> S (3) | 41.1 +- 0.8 | 53.9 +- 1.3 | -24.8 (-12.1) | 84.1 (95.4) | 35.2 (56.0) | 52.4 (63.3) | 26.5 (38.0) |
| R+S -> R (2) | 57.0 +- 0.4 | 60.8 +- 1.1 | -8.9 (-5.1) | 94.8 (96.8) | 62.8 (70.2) | 67.7 (70.8) | 36.9 (40.2) |
| R+S -> S (2) | 44.2 +- 1.1 | 57.7 +- 1.9 | -21.8 (-8.3) | 85.6 (96.4) | 41.8 (65.2) | 55.9 (67.2) | 26.3 (39.7) |

Per keypoint, S -> R (first sweep in brackets): elbows 15.0 / 13.7 (5.9 / 2.8), wrists 7.6 / 4.6
(2.0 / 0.7), shoulders 53.9 / 55.3 (22.4 / 20.8), head / neck 63.4 / 66.9 (34.6 / 36.4), hips
76 (49 / 45), knees 66 / 68 (38 / 33), ankles 61 / 59 (30 / 45); R -> R: elbows 42 / 51, wrists
23 / 26, shoulders 75 / 77, hips 91. The two transfer arms (S -> R, R -> S) gain 15-24 AP; the
rendered-domain self-consistency (S -> S, R+S -> S) drops 13 AP -- the new renders are no longer
an easier version of themselves -- and R+S -> R loses 3.8 AP on two seeds.

RT-Pose (cm; R -> R MPJPE 16.1 +- 1.5, Abs-MPJPE 18.0, MRPE 10.8, PCK10 48.2%, PCK15 69.2%):

| arm | MPJPE now | MPJPE first | Abs-MPJPE | MRPE | PCK10 % | PCK15 % |
|---|---|---|---|---|---|---|
| S -> R (3) | 65.4 +- 1.5 | -- | 74.9 +- 10.2 | 61.7 +- 12.1 | 30.4 | 46.3 |
| R -> S (5) | 83.5 +- 18.8 | 152.6 +- 35.7 | 91.9 (162.0) | 83.8 (148.5) | 29.7 (19.9) | 46.1 (31.7) |
| S -> S (3) | 22.9 +- 4.0 | -- | 28.9 | 19.4 +- 9.0 | 41.9 | 61.4 |
| R+S -> R (3) | **14.4 +- 0.4** | -- | 17.8 +- 0.7 | 11.7 +- 0.6 | 49.0 | 69.6 |
| R+S -> S (3) | 16.2 +- 1.1 | -- | 20.2 | 14.7 | 46.0 | 66.4 |

R+S -> R beats R -> R (14.4 vs 16.1 MPJPE, PCK10 49.0 vs 48.2; MRPE 11.7 vs 10.8). The S -> R
and R -> S means are set by five of the 13 test sequences, per sequence (MPJPE / MRPE, cm):

| seq | activity | R -> R | S -> R | R+S -> R | S -> S | R -> S now | R -> S first |
|---|---|---|---|---|---|---|---|
| 141 | Walk, cluttered, pad | 9.1 / 9.2 | 16.8 / 16.4 | 8.7 / 8.7 | 9.3 / 10.1 | 12.7 / 17.4 | 32.3 / 35.1 |
| 159 | Stand and Lift Leg, outdoor | 8.7 / 5.4 | 14.6 / 11.1 | 8.7 / 5.1 | 9.5 / 6.0 | 15.5 / 11.1 | 37.7 / 45.5 |
| 166 | Walk and Wave Hand, outdoor | 13.4 / 7.1 | 18.5 / 14.3 | 13.5 / 8.7 | 15.4 / 8.8 | 17.4 / 12.7 | 76.5 / 66.1 |
| 191 | Stand and Lift Leg, clean | 11.9 / 11.2 | 17.9 / 17.1 | 12.0 / 11.7 | 14.2 / 11.3 | 16.8 / 13.2 | 17.6 / 15.7 |
| 185 | Walk, clean | 20.6 / 11.5 | 29.8 / 20.5 | 19.8 / 11.1 | 19.1 / 10.7 | 26.8 / 19.4 | 25.7 / 27.1 |
| 1 | Stand and Wave Hand, cluttered | 17.0 / 10.4 | 26.5 / 17.3 | 14.9 / 9.2 | 21.9 / 29.9 | 69.5 / 82.6 | 214.8 / 127.2 |
| 148 | Walk, cluttered, cardboard | 21.6 / 12.6 | 40.6 / 28.0 | 18.7 / 11.4 | 25.2 / 10.8 | 21.8 / 14.8 | 74.0 / 56.9 |
| 18 | Walk, cluttered | 14.8 / 10.2 | 66.1 / 50.0 | 11.3 / 9.8 | 16.2 / 10.8 | 42.8 / 44.6 | 71.4 / 91.1 |
| 176 | Stand and Wave Hand, clean | 17.5 / 13.1 | 75.0 / 66.3 | 14.9 / 16.4 | 22.5 / 16.4 | 236.9 / 242.1 | 317.3 / 228.7 |
| 168 | Stand and Wave Hand, outdoor, pad | 24.4 / 15.9 | 105.6 / 94.9 | 24.1 / 24.0 | 39.1 / 50.8 | 165.8 / 132.4 | 236.4 / 294.3 |
| 164 | Walk and Sit, outdoor | 11.5 / 8.6 | 109.5 / 152.5 | 10.8 / 11.1 | 32.9 / 49.3 | 123.2 / 197.9 | 206.0 / 245.7 |
| 147 | Stand and Wave Hand, cluttered, cardboard | 22.1 / 14.6 | 160.4 / 157.5 | 16.2 / 10.6 | 29.6 / 18.5 | 56.7 / 32.0 | 147.4 / 178.6 |
| 108 | Stand and Wave Hand, rain | 16.2 / 10.4 | 168.7 / 156.0 | 14.0 / 14.5 | 42.5 / 18.3 | 280.3 / 269.4 | 527.1 / 518.5 |

On the eight walking / leg-lifting sequences S -> R is 1.1-2.0x R -> R (141, 159, 166, 191,
185, 1) or 2-4x (148, 18); on the five near-static ones (Stand and Wave Hand 108, 147, 168,
176; Walk and Sit 164) the render puts the person a metre or more from the labels (R -> S MRPE
1.3-2.7 m, S -> S MRPE 0.5 m on 164 / 168): the person's placement comes from the radar's
MOVING return (the range and azimuth registrations, the motion detector), and a standing
person waving a hand gives it a few frames of arm return to lock on (5.1 item 3 recorded 108
as "2 motion frames"). Those sequences are ~40% of the training set by activity, so the
S -> R model is trained on labels that disagree with the renders there, which is what drags
the walking sequences (18: 66 cm) as well. The next fix is the registration on static
sequences (a static-return range lock: the person's zero-Doppler return against the empty
scene's, not the moving return), not the person model.

### 5.5 Downstream, the full five-seed matrix on the re-render (2026-09-18 22:29 UTC, node)

`rerender_and_retrain.sh` phase 3 complete: 15 / 15 cells ok on each dataset, blindness gates
passed, matrix and tables regenerated (downstream/results/tables). Mean +- std over five seeds;
"first" = the first sweep's arms (results_floorfix).

| HuPR (AP x100) | now | first | delta |
|---|---|---|---|
| R -> R | 65.9 +- 1.5 | 65.9 | |
| S -> R | **40.1 +- 1.6** | 15.8 +- 1.8 | +24.2 |
| R -> S | **36.9 +- 0.9** | 22.0 +- 3.4 | +14.9 |
| S -> S | 41.3 +- 0.7 | 53.9 +- 1.3 | -12.5 |
| R+S -> R | 56.6 +- 1.6 | 60.8 +- 1.1 | -4.2 |
| R+S -> S | 43.3 +- 2.0 | 57.7 +- 1.9 | -14.4 |
| window control R -> R (unwindowed) | 68.2 (n = 1) | 68.2 | |

S -> R per keypoint (first): elbows 14.6 / 14.7 (5.9 / 2.8), wrists 7.3 / 4.9 (2.0 / 0.7), shoulders
54.6 / 55.8 (22.4 / 20.8), head / neck 63.3 / 66.6, hips 76.9 / 76.2 (49.0 / 45.2), knees 67.0 / 68.0,
ankles 62.1 / 58.8; R -> R elbows 41.5 / 50.9, wrists 22.5 / 25.6, shoulders 74.7 / 77.0, hips 90.7 / 90.3.

| RT-Pose (cm) | MPJPE | Abs-MPJPE | MRPE | PCK10 % | PCK15 % |
|---|---|---|---|---|---|
| R -> R | 16.1 +- 1.5 | 18.0 +- 1.4 | 10.8 +- 0.8 | 48.2 | 69.2 |
| S -> R | 69.5 +- 5.2 | 75.4 +- 8.5 | 67.1 +- 15.1 | 31.8 | 48.1 |
| R -> S (first 152.6 +- 35.7) | 83.5 +- 18.8 | 91.9 | 83.8 | 29.7 | 46.1 |
| S -> S | 26.6 +- 6.4 | 30.7 | 19.9 +- 7.5 | 41.6 | 61.0 |
| **R+S -> R** | **14.4 +- 0.4** | **18.1 +- 0.7** | 12.1 +- 0.7 | **49.8** | **69.9** |
| R+S -> S | 15.9 +- 0.9 | 20.2 | 14.4 | 46.9 | 66.9 |
| two-person part, R / S / R+S -> R | 54.9 / 89.6 / 40.2 | 103 / 134 / 96 | 97 / 125 / 92 | | |

R+S -> R is at or above the real arm on every metric (MPJPE 14.4 vs 16.1, PCK10 49.8 vs 48.2, the
seed spread 0.4 vs 1.5; MRPE 12.1 vs 10.8). S -> R and R -> S, per test sequence (MPJPE / MRPE, five
seeds): the eight walking / leg-lifting sequences 141 17.4 / 16.0, 159 14.6 / 11.2, 166 18.7 / 14.1,
191 20.1 / 17.0, 185 31.3 / 19.7, 1 25.7 / 16.2, 148 35.7 / 26.3, 18 60.7 / 49.3 against R -> R
9.1 / 8.7 / 13.4 / 11.9 / 20.6 / 17.0 / 21.6 / 14.8 (1.1-2.0x on six, 1.7x and 4.1x on 148 / 18);
the five near-static ones 176 69.5 / 60.2, 168 113.6 / 101.9, 164 149.0 / 194.0, 147 158.5 / 161.5,
108 188.1 / 184.9 (R -> R 17.5 / 24.4 / 11.5 / 22.1 / 16.2): the zero-Doppler image, section 5.4.

## 6. The RT-Pose runtime plan (the +63%)

The +2.1 ms of the full array is the physics (every TX rendered); the +8.6 ms of the elevation
plane term is implementation, and most of it is avoidable without touching what is supervised:

1. **The plane part of the term on the person's range rows only.** The Doppler planes carry the
   mover; the static scene lives in loop-0, which stays full-range. Restricting the plane RAE
   term to `range_rows` (the person's rows, 10-25 of 244, already a per-view mask the MD term
   uses) cuts its cells 10-20x and supervises exactly the cells whose elevation the plane term
   can inform. Expected: 8.6 -> ~1.5 ms.
2. **One contiguous gather + one 2D FFT.** Replace `channels_to_grid2d` (index_add scatter,
   weight multiply, reshape) + `movedim` + two `fft`/`fftshift` pairs by the azimuth path's
   pattern: a precomputed (slot -> channel) table gathered by one compiled kernel into a
   contiguous `(..., K, G_az, G_el)` buffer with the two windows and the fftshift signs folded
   into per-slot weights, then a single `torch.fft.fftn` over the last two axes. Removes ~6 of
   the ~15 full-tensor passes (and their backward), all bitwise the reference (the azimuth
   path's gate shows how).
3. **The L2 as one compiled reduction.** The azimuth plane term already has `_plane_term_sums`
   (a compiled kernel: floor, sqrt, difference, square, sum in one pass). The RAE term does
   abs -> square -> sub -> square -> mean eagerly; feed it the same kernel (it takes a complex
   spectrum and a target). Halves the remaining passes.
4. **No elevation zero-padding**: `n_el = 7` (the cascade's slot span) instead of 8 -- an exact
   DFT, one bin fewer (-12% on the term); HuPR unchanged (2).
5. **Render the elevated TX only where the term reads them.** The splat renders the 3 elevated
   TX (48 of 192 channels) on every plane and every range bin, while the term reads their planes
   on the person's rows (after 1) and loop-0 everywhere. A per-TX-group plane set in the sorted
   Doppler route (the elevated TX on loop-0 + the person's planes) recovers most of the +2.1 ms
   of the full array. Deeper (the CUDA kernels' path lists), so last.
6. **Housekeeping**: the per-step `index_select` of every flat view includes the RAE store
   (71 MB a step on the cascade) and `elevation_terms` gathers it again -- read the gathered
   view once; keep the store in half precision as now.

Expected after 1-4: ~15 ms a step (+33% over the 11.3 baseline, from +95%); after 5: ~13.5
(+20%). Each step is gated like every other change: the eight sequences, the spread, the
proxies -- 1 and 4 change what is supervised (the rows; the padding) and need the full gate, 2,
3 and 6 must be bitwise and are gated numerically.
