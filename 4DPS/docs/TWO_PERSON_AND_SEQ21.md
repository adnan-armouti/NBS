# Two findings from the sweep: the two-person RT-Pose recordings, and HuPR 21

Written 2026-09-14, before any change. Everything here is measured on the
delivered products (`/home/adnan/Documents/Data/4DPS_products`), the
dataset's own metadata (`filemeta.json`, `Train.json`), the RT-Pose paper
(arXiv 2407.13930, ECCV 2024) and the authors' released code
(github.com/ipl-uw/RT-POSE at `main`, 2026-09-14).

## 1. The two-person RT-Pose recordings

### 1.1 How many, and what the dataset says about them

`filemeta.json` labels every sequence's `Subject type`: **66 of 242 are
"Multi-Person"** (27%); of the 106 folders on this machine, **38** are
(100-107, 117-140, 170-175; the rooftop scenes "Stand and Wave Hand" x 16,
"Stand with Random Pose" x 14, "Walk" x 8). My YOLO count found 35 of the
38 (106 and 174 have the second person out of frame at the sampled times;
107 has no radar). The paper (section 3.2): "Each nonocclusive and
occlusive condition is recorded in single- and multi-person settings."

**The annotation is one subject per frame, in every sequence.** `Train.json`
carries exactly one pose row per annotated frame for all 242 sequences,
the two-person ones included (100: 237 frames, all with one row; 101:
221; 102: 233). The paper's workflow labels "the human subject" with one
LiDAR bounding box (section 3.4). The second person in a two-person
recording is therefore an **unannotated distractor**: present in the
radar, the LiDAR and the camera, absent from the labels.

### 1.2 How the RT-Pose authors' own method handles it

* The paper describes HRRadarPose's decoder as multi-person: "picking
  top-k human centers ... In scenarios with multiple people, we apply range
  non-maximum suppression" (section 4).
* The **released** configuration decodes **one** person a frame:
  `configs/cruw_pose/hr3d.py` sets `nms_pre_max_size=1,
  nms_post_max_size=1`; the dataset loader keeps one pose a sample
  (`cruw_pose.py`: `sample['poses'] = [obj['pose']]`) and the evaluation
  scores the detection against `gt[seq][frame][0]['pose']`, the single
  annotated subject. MPJPE / MRPE / Abs-MPJPE are all single-subject
  numbers. No multi-person result is reported in the paper (Table 4 is
  the occlusion split, Fig. 2 the activity / environment / occlusion
  distributions).
* So in the benchmark as released, a two-person frame is a **single-
  subject problem with a distractor**: the network is trained with one
  labelled centre while the tensor holds two people, and is scored on
  the one annotated person.

### 1.3 Train or test?

The released config reads `Train.json` for training and `Test.json` for
testing. Hugging Face ships `Train.json` with labels for all 242
sequences and 239 sequence folders; `Test.json` is not public, and the
paper does not state which sequences are held out. The train/test
membership of the multi-person sequences is therefore **not
recoverable** from what is published. What is certain: every sequence we
have, single- and multi-person alike, carries "Train" labels in the
public release.

### 1.4 Why our chain gets them wrong -- it is both, and the objective as well

The chain assumes one person in three places; none of them is the point
renderer itself, which is a sum over points and does not care how many
bodies the points belong to.

**Preprocessing (person stack)**

| step | assumption | what happens with two people | evidence |
|---|---|---|---|
| GVHMR mesh (`gvhmr_fast_dump.track_one_person`) | one track: the longest, then the largest | YOLO tracks both; the chain keeps one of them, not necessarily the annotated subject, and the tracker can switch between them | the GVHMR track's range against the annotated subject's: Pearson -0.40 to +0.59, median distance 0.4-2.7 m on the ten two-person sequences checked (single-person 183/10: +0.99/+1.00, 0.45 m = GVHMR's size prior) |
| per-frame radar registration (`radar_depth_scales`, `person_range_series`) | the moving-power range-profile peak is the person | the peak is whichever person moves more that frame, or the other person's range | radar peak against the annotated subject: -0.35 to +0.47; against the GVHMR track: -0.17 to +0.53; the fitted depth-scale IQR width 0.2-1.3 against 0.04-0.07 on single-person sequences; on 175 the peak sits at 0.47 m (near-field) half the time |
| the LiDAR scaffold (`run_lidar_scaffold`) | clears one person's column (the annotated one, 0.6 m radius) | the second person's LiDAR points become "static scene" | not quantified; it puts a standing person's return into the background material fit |
| motion detection and the pose-speed frame selection | one mover | mixed | -- |

**Model, renderer and objective (velocity_field, video)**

| component | assumption |
|---|---|
| the person is one point set of 6890 SMPL vertices with one 14-joint velocity field (frames, 14, 3) skinned to it, one shared material, one area scale | one body |
| the micro-Doppler target mask is one person's (range, angle) support projected from that cloud | one body |
| the real RA / 63-plane / MD targets contain **both** people's returns; the objective asks one modelled body to explain them | the loss pulls the one body's velocity field and materials toward the sum of two movers -- this is what the sweep measured: RA 0.830 vs 0.866, RD 0.47 vs 0.63, MD energy 0.82 vs 0.90, MD centroid 0.84 vs 0.97 (two-person vs one-person medians) |

So: **both**. The preprocessing picks and registers the wrong or a mixed
person; and even with the right person, the single-body model is fitted
against radar that contains two, so the objective is mis-specified. The
forward model (B in docs/METHODS_INVENTORY.md) itself needs no change to
render two bodies -- it is a sum over points -- but every structure
around it (mesh source, registration, joints, skinning, MD mask, motion
frames) is single-person.

### 1.5 What each way forward would take

| option | what it is | cost | what it buys |
|---|---|---|---|
| A. **Do what the benchmark does: fit the annotated subject** | choose the GVHMR track by the annotation (the track whose projected box / 3D pelvis matches the annotated pose per frame, 221-237 annotated frames a sequence); register the range against the annotated pelvis (LiDAR-grade, within 2 cm per the paper) or the RA-map peak nearest the annotated (range, angle) instead of the range-profile peak; clear only that subject from the scaffold | 1-2 days, preprocessing only; a gate on 183/10 (annotated == current) | the correct subject is fitted and reported on, exactly as HRRadarPose is scored; the second person stays an unmodelled distractor in the real radar, so RA/RD on those 35 sequences remain bounded below the single-person ones -- reportable as the benchmark's own single-subject protocol |
| B. **Model both people** | GVHMR for the two tracks; two point sets, two 14-joint fields (or one 28-joint field), per-person registration by the RA-map peak at each person's angle, per-person MD masks (the MD target as the union), the shadow masks over both meshes | ~1 week; new gates; the second person's mesh is only as good as GVHMR's track of an unlabelled person | the objective becomes well-specified on the 35 sequences; the numbers become comparable to the single-person ones |
| C. **Exclude the 35** | -- | none | a smaller RT-Pose set (66); the dataset's own baseline does not exclude them |

Recommendation: A now (it is the benchmark's definition of the task and
its annotation, and it fixes the actual failure -- the wrong subject
being registered), B if the paper wants the two-person set on equal
footing. Not C.

## 2. HuPR sequence 21

**What the 0.24 / 0.37 are.** The micro-Doppler **spectrogram Pearson**
(`microdoppler.spectrogram_r` in `metrics.json`): the Pearson correlation
between the real and the rendered amplitude spectrograms -- Doppler bin
by frame, linear amplitude, the person-masked marginal over range and
angle -- computed over the **motion subset** (frames with pose speed
>= 0.15 m/s, pre-registered; here 99 of 600, frames 119-521). 0.238 on
the vertical board, 0.374 on the horizontal. Over all 600 frames the same
correlation is 0.736 / 0.843; the energy-per-frame correlation 0.668 /
0.831.

**What the recording is.** A person standing still for the whole minute,
raising and lowering one arm sideways (the RGB frames: arm horizontal at
150, down at 300, horizontal again at 330-560). The body does not move
(centroid range 2.90 +/- 0.03 m for 550 frames); the arm moves mostly
across the beam, so its radial velocity -- what micro-Doppler sees -- is
small. The radar's own motion detector accepts 166 of 600 frames; the
registration is fine (depth scale 0.757, IQR 0.742-0.771, RA 0.967 /
0.931, the best RA products of the sweep); the person is where it should
be.

**Why the correlation is low.** Both spectrograms are nearly empty: the
real energy sits in five Doppler bins around DC (bins 30-35 of 64 carry
94% of it) and is concentrated in a few single-frame spikes (frames ~140,
~520, ~595: the arm's snap at the top or bottom of the raise). The render
puts 55% of its energy in the same bins but (i) underestimates those
spikes 5-7x (the fitted velocity field is band-limited to 3.5 Hz -- a
one-frame event at 10 Hz is above it by construction) and (ii) has a
second lobe at bins 21-23 (-0.4 m/s, 10-16% of its energy) that the real
signal does not have -- a spurious radial component of the lateral arm
motion, from the camera-derived initialisation the 99 supervised frames
cannot correct. On a 99-frame subset dominated by two spikes, a Pearson
of 0.24-0.37 follows; the same render scores 0.74-0.84 over the full
sequence because the near-silent frames agree.

There is also a rendered energy burst at frames 0-20 (5x the real):
GVHMR's first frames before its track settles give the camera
initialisation a spurious motion the trainer keeps. Outside the motion
subset, so not the cause here, but a systematic thing worth a fix
(ignore or damp the first ~20 frames of the camera velocities).

**Verdict.** Not a pipeline failure: a still person with a lateral,
low-SNR, spike-like micro-Doppler, on which the pre-registered
motion-subset metric is dominated by two frames. The number is right
about what it measures. Two things it does point at: the 3.5 Hz temporal
band cap cannot represent single-frame events, and the initialisation's
spurious start-of-sequence motion survives training.
