# The near-static RT-Pose sequences: diagnosis and plan (branch `fidelity/static-seqs`, 2026-09-19)

Numbers below are from this branch's own runs: `base2` (the branch's baseline: the 11-sequence dev set,
`launch_cand.sh base2 ""`, 15 min wall on the 8 GPUs, RT-Pose 820-882 s / sequence), the gradient probe
(`grad_probe_node.sh`), `viz_pair.py`, `viz_geometry.py`. Artifacts: `mmSD/artifacts/static_pair`,
`static_probe`, `static_geometry`. Every estimator is named where a number is given.

## 1. What fails, in the downstream model's own domain

The real-trained HRRadarPose seeds lose the person on the render when the person stops being the brightest
thing in the model's ROI. In the adapter's polar |RA| (no window, no clutter removal, summed over Doppler and
elevation), person = max within +-0.4 m / +-8 deg of the label, competitor = max elsewhere in the ROI
(x 0.77-8.02 m, |y| <= 5 m). Pooled over the 1463 labelled frames of 148 / 18 / 176 / 168 / 164 / 147 / 108
(seed-mean R->S MRPE per frame):

| render person / competitor | frames | R->S MRPE median | frames > 1 m |
|---|---|---|---|
| < 0.5 | 178 | 390 cm | 99 % |
| 0.5-1 | 627 | 159 cm | 79 % |
| 1-1.5 | 123 | 22 cm | 18 % |
| 1.5-2 | 150 | 17 cm | 2 % |
| >= 2 | 385 | 16 cm | 1 % |

The recording's own contrast is a much weaker predictor (contrast 1.5-2 in the recording still fails 53 % of
the frames on the render), and so is the person's absolute render/recording ratio alone. Both sides
contribute: the render's person peak is 0.2-0.4x the recording's in every diverge window (its zero-Doppler
part 0.1-0.35x), and on 176 the render's background peaks are at 0.95-1.04x the recording's. 164 shows the
switch inside one sequence: R->S tracks the label while the person walks (frames 100-190) and fails in the two
sitting phases.

## 2. Why the objective does not prevent it (the gradient probe)

The velocity-field objective has five terms, all normalised L2 on linear magnitudes (each divided by its
target's mean square): loop-0 |RA| (one chirp), the 63-plane |RA| (mean-removed Doppler planes, with the
learned floor), the MD marginal, and -- added on top since `--elevation-term on` became the default -- the
loop-0 |RAE| and the plane |RAE| (`static_losses`: `total = total + rae`). Only the two loop-0 terms see the
zero-Doppler content.

*Tried and reverted (2026-09-19, candidate `rae3`, commit 2ae0317, reverted in f58ef38):* making the RAE terms
replace the RA terms (three terms). Gate under the dataset labels within the run-to-run noise (176 110.7 +- 70.6 vs
base2 117.8 +- 69.8; 168 142.5 / 146.0; 164 172.0 / 158.0; 108 228 +- 216 / 298.7; 183 24.6 / 25.0; 186 17.9 / 15.9;
153 24.7 / 19.4 cm), but the fidelity suite regressed on every RT-Pose sequence (RA -0.01..-0.03; clutter-removed RA
176 0.11 -> 0.06, 168 0.30 -> 0.18, 153 0.59 -> 0.43) and HuPR 40's micro-Doppler collapsed (spectrogram r 0.885 ->
0.551); step 23.9 -> 23.5 ms. The five-term objective stays; P1 below is its sixth term.

`MM4DPS_GRAD_PROBE` evaluates each term alone on 12 frames and backpropagates it alone. d(term)/d log(person
amplitude), negative = the term pulls the person brighter (exact: the person's amplitude is sqrt(area_scale)):

| state | seq | loop-0 RA | plane RA | MD | elev. loop-0 | elev. plane | person share of loop-0 target energy | of plane target |
|---|---|---|---|---|---|---|---|---|
| trained | 176 | -0.0012 | +0.029 | -0.150 | -0.002 | +0.053 | 5.1 % | 37 % |
| trained | 168 | -0.0003 | +0.018 | -0.131 | -0.001 | +0.042 | 7.0 % | 36 % |
| trained | 164 | +0.014 | +0.062 | -0.120 | +0.014 | +0.044 | 6.5 % | 45 % |
| trained | 108 | -0.011 | +0.012 | -0.039 | -0.007 | +0.031 | 6.6 % | 17 % |
| trained | 183 (good) | +0.003 | +0.099 | -0.181 | -0.009 | +0.036 | 17 % | 56 % |
| trained | 186 (good) | +0.019 | +0.092 | -0.145 | -0.017 | +0.009 | 20 % | 58 % |
| initial | 176 | -0.005 | +0.145 | +0.97 | -0.006 | +0.24 | | |
| initial | 108 | +0.017 | +1.96 | +7.96 | +0.009 | +0.62 | | |
| initial | 183 | -0.015 | -0.011 | -0.115 | -0.015 | -0.017 | | |

Three facts follow.

1. **The static body is nearly unsupervised.** Its only supervision is loop-0 (and loop-0 RAE), where the
   person's RA-mask cells are 5-7 % of the target energy on the failing sequences (17-20 % on 183 / 186). The
   loop-0 gradient on the person's amplitude is 10-100x smaller than the moving-band terms', and its gradient
   norm goes to the scene (176: 25 into scene rotations, 7 into scene materials, 0.0017 into the person area
   scale); the loop-0 residual is still 0.25-0.44 of the target energy at the end.
2. **The moving-band terms penalise spread more than dimness.** At the initial state on the failing sequences
   they push the person DIMMER (+0.6..+8 MD, +0.15..+2 plane) although the rendered person carries only
   35-75 % of the target's moving energy; per-cell L2 on a broader-than-real return is reduced by dimming. The
   trained state is an equilibrium of plane "dimmer" (+0.01..+0.14, sign-consistent over frames) against MD
   "brighter"; the static body's level is a by-product of it.
3. **The parameters let the trainer dim the torso selectively.** The brightness gradient lands on the
   person's material (norm 5-20) and the joint gains, not on the area scale (0.02-0.06). Trained gains, 176:
   hips 0.09 / 0.09, shoulders 0.48 / 0.51, elbows 5.0 / 4.7, ankles 3.7 / 6.0; 108: one elbow 10.0; coherent
   fraction 0.056 against the scene's 0.50. The limbs (what the plane / MD terms see) are brightened, the
   torso (what only loop-0 sees) is killed. This is also why the fourth branch's "dc-mean anchor" alone could
   not help: a better anchor keeps the person at 5 % of its term.

Geometry is not the cause (`viz_geometry`): feet on the floor (0.04-0.15 m), 47-56 % of the mesh visible,
learned-vs-mesh person normals ~2 deg; the mesh-vs-label pelvis offsets (0.25 / 0.44 / 0.34 / 0.54 m on 176 /
168 / 164 / 108) are the labels' per-session frame (docs/FIDELITY_FIX_RESULTS.md 3.14) -- see section 4.

## 3. The plan: one change per commit, both datasets, no per-dataset flag, no label-derived term

Acceptance for every step: the probe's d/dlog(person) per term, `viz_pair` contrast on 176 / 168 / 164 / 108,
the five-seed gate on the 7 RT-Pose dev sequences (`gate_node.py`), `fid_node.py` within base2's spread
(RT-Pose cascade RA 0.78-0.91, RAc 0.11-0.69, RD 0.11-0.74; HuPR RA 0.91-0.96 / 0.93-0.94), runtime per step
and wall per stage. Run-to-run trainer variance is large on these sequences (176: base2 117.8 +- 69.8 cm vs
242 on the archived re-render of the same code), so a candidate is judged on the whole set, and a borderline
one on a second trainer run.

**P1 -- a region-normalised zero-Doppler term (the sixth term, in both views).** Add the DC plane (the 64-chirp mean: the static plane, with
sqrt(64) the SNR of the one-chirp anchor) as its own normalised L2, normalised *within the person's RA mask by
the mask's own target energy* and outside it by the rest. The mask is the geometry-derived one the MD
marginal already uses (`person_range_angle_mask_torch`), on both datasets, not a label. Target signature:
d(static term)/d log(person) of order 0.1, commensurate with the moving-band terms; contrast >= 1.5 on the
failing sequences; HuPR unchanged.

**P2 -- stop rewarding dimming-and-spreading.** Only if P1 leaves the plane term pushing "dimmer" below 1x
energy: compare local energy (a PSF-scale box filter of |.|^2) instead of per-cell linear magnitude in the
plane / MD terms -- the measured magnitude is one speckle realisation, the render should match its local
expectation.

**P3 -- remove the non-physical freedom.** Bound the joint gains (|log g| <= log 2) and hold the coherent
fraction near the person stage's value. Fixed gains alone were rejected on the fourth branch; here they are
the complement of P1 / P2, not a substitute.

## 4. The labels (decided 2026-09-19): every radar frame, from the registered mesh

The dataset's `Train.json` annotates radar frames 2-240 of each 299-frame recording, in a per-session camera
frame that is laterally scaled and range-offset against the radar (3.14: Indoor-Clean mesh 0.2-0.37 m nearer
than the label; other sessions 0.2-0.4 m lateral). The person stage registers the GVHMR mesh to the radar's
own return, so its joints are the radar-consistent pose for all 299 frames. `downstream/tools/relabel_rtpose.py`
writes them as `Train_gvhmr.json` next to the dataset's file, in the label schema and joint order:

- axes verified on every walking sequence: label x forward = radar-world Y (r >= 0.98), label y lateral =
  -X (r = +1.00, the beamformer mirror), z up;
- joints 9-11 are the LEFT arm and 12-14 the RIGHT (against the mesh's anatomical sides; the HRRadarPose task
  list has it right, the dataset's `Keypoints_meta.txt` has the arms swapped);
- pelvis = midpoint of the SMPL hips (the dataset's pelvis is the hip midpoint to 1 cm), thorax = SMPL neck
  (6-16 cm from the dataset's), head = the mesh's head top (the dataset's head is 0.2-0.6 m from either
  definition -- the noisiest joint of the camera estimator), the rest the SMPL regressor joints;
- the two-person extra set, the excluded sequences and every sequence outside the split keep the dataset's
  labels (one mesh cannot label two people).

**Sequence 13 is a dataset defect and is excluded from the split** (`rtpose_split.json`, `excluded`): the
dataset ships sequence 12's raw radar under 13 (all eight `radar/bin` files identical by md5; the only
duplicate among the 104 raw sequences), while 13's camera, mesh and labels are a standing person at 5.2 m and
the radar walks 2.8-5.7 m. Every real-trained checkpoint so far has trained on it. The train part is 44
sequences now.

Selection: `downstream.common.domain.rtpose_labels()` -- `Train_gvhmr.json` when present, `RTPOSE_LABELS=
Train.json` for the dataset's own; `prepare_root` re-points the cache's `Train.json` link. On the node this
branch uses its own cache root `/data/mmsd/4DPS_downstream_gvhmr` (cubes linked, labels linked) so the
shared `/data/mmsd/4DPS_downstream` and the other session's runs are untouched.

Consequences to keep in mind: numbers under the new labels are not comparable with the RT-Pose paper's or
with the archived matrix; R->R will change (more frames, a radar-consistent frame, no seq 13); every
real-trained checkpoint the gate uses must be retrained before any candidate is gated under the new labels.

## 5. Next (in order)

1. Retrain the real arm under the new labels: 5 seeds on 5 GPUs (`real__real`; the archived run took 5 h 41
   min per seed at 50 epochs on ~239 frames / sequence, so ~7 h wall with 299), and re-evaluate `real__rendered`
   on the base2 renders; then the base2 gate under the new labels is the reference point.
2. P1 on the 3 remaining GPUs (11-sequence dev set, ~40 min at 3 GPUs), probe + viz_pair + fid; gate once
   the seeds of step 1 exist.
3. P2 / P3 as needed; then the full re-render + retrain of both datasets.
