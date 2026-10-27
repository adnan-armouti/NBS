# Seeds: what a run takes from outside the recording

Status: 2026-09-13, branch `hann-windowing`. `seeds/` is git-ignored. This
is the inventory of every input a run reads that is not the dataset itself,
what would break without it, and the values that shape a run without being
files.

## The rule

A number that describes the *recording* -- a floor, a gain, a coupling, a
placement, a material -- is estimated from that recording inside the chain
or learned by the trainer, and lands in the run's own outputs. The only
file under `seeds/` a run reads is the RT-Pose rig extrinsic (the identity)
that no in-chain estimator can identify; the person's mesh is produced by
GVHMR inside the chain. Nothing else is
installed into a workspace (`chain.setup_workspace` creates the layout and
links the tracked configs, and that is all).

## Read

| seed | read by | what it is | if it went away |
|---|---|---|---|
| `seeds/rtpose/radar_extrinsic.json` | `reconstruction.rtpose_extrinsic_seed` (the LiDAR scaffold and the person sequence) | the RT-Pose radar-at-the-LiDAR extrinsic: rotation identity, translation 0, from the rig and two independent person measurements (recorded in the file) | the identity is the fallback in code; the measured static map cannot identify a lateral shift (correlation 0.19 -> 0.24 over a +/-0.5 m search that walks the height to its edge), which is why it is not refitted |
| `seeds/rtpose/orientation_check.json` | `qa orientation-check` only | the board orientation verified from the data | a QA record; the chain does not read it |
| `configs/*.json`, `configs/mmwcas_calibration.npz` | everything | the radar descriptions and TI's cascade phase/amplitude calibration | tracked, not seeds |

## Computed in the chain instead (was a seed)

| was | now |
|---|---|
| `seeds/gvhmr_vertices/<dataset>/seq<S>.npz` -- GVHMR's SMPL vertices per camera frame, dumped by hand for six sequences | `mm4DPS.gvhmr vertices` at the start of the person stage: the RGB frames encoded losslessly (HuPR's JPEGs from `frames.zip`, RT-Pose's left camera) and run through GVHMR in its own environment (`GVHMR_ROOT`, `GVHMR_PYTHON`) on the run's GPU -- 4 s encode + 55-60 s inference a sequence, deterministic given the frames (the same video twice is bitwise; a lossy re-encode moves the mesh 3-5 mm median, a tenth of a range bin). Kept with the products as `gvhmr_vertices.npz`. |
| `seeds/gvhmr_person_geometry/person_sequence_<S>_v2clean/` and `seeds/rtpose/gvhmr_person_geometry/...` -- the person sequences prebuilt outside the chain (HuPR's through a range anchor of an old depth-cloud pass, the frame-0 sweep and a cleanup; RT-Pose's by a 176 s seed build) | `reconstruction person-sequence`, in the person stage (RT-Pose 5 s, HuPR 10 s): the vertices placed in the radar world (RT-Pose: camera -> LiDAR by the shipped calibration -> radar by the extrinsic; HuPR: the camera at the rig, co-boresighted, (right, down, forward) -> (x, -z, y)), GVHMR's depth rescaled by the radar frame by frame (`radar_depth_scales`: the moving-power range-profile peak over GVHMR's centroid range, a scale about the camera origin; HuPR's GVHMR runs 30% long -- median scale 0.707, IQR 0.699-0.717 on 256 -- RT-Pose's 7% short), the 14 joints from the SMPL regressor, the loop-0 \|RA\| targets, the per-frame visibility. RT-Pose 183 rebuilt in-chain is bitwise the seed (positions, normals, joints, visibility, scales); the targets 8e-8 relative (GPU FFT against numpy). |
| `seeds/seq<S>/seq<S>_person_range_hfov40/` -- the frame-0 person-range sweep of an earlier pass | gone with the sweep: the per-frame registration places the person (translation 0 on every dataset); `calibration frame1-person-range` and the person's pass A were removed |
| `seeds/background/seq<S>_background_hfov40/best_model.pt` -- the scene's material model from an earlier pass (fitted on an UNALIGNED scaffold, which is what `bake-scene-alignment` then moved) | `train_motion dual-radar` in the reference stage on every dataset, 75 iterations on the aligned scaffold (2-3 s); the bake is gone |
| `seeds/motion_init/.../chirp_motion.pt` -- 14 joint scattering gains fitted on HuPR sequence 1 (0.1 on the trunk, 3.0 on the arms), passed through by `train_motion inherit-joint-gains` and frozen; both datasets read it | learned by the velocity trainer: `joint_log_gain (14,)`, geometric mean pinned to 1 (the body's level is the area scale's and the receiver gain's), `--gain-lr`; saved as `inherited_joint_scattering_gains` in the velocity checkpoint, which the renderer reads as before. `inherit-joint-gains` and `--initial-motion-checkpoint` were removed. |
| `seeds/calibration/antenna_coupling/` -- HuPR's coupling fitted on held-out sequences, unwindowed | `calibration sequence-coupling` on every dataset (HuPR 4 s, the cascade 1 s): the coherent mean of the near bins per virtual channel from this recording (99.98% coherent on 256; 0.87-0.89 complex correlation with the old seed at 0.75x its level -- the seed's window was not this one's) |
| `seeds/seq<S>/absolute_gain_render_result.json` and the RT-Pose pre-training gain fit | the trainer's learned per-radar gain (docs/FIDELITY_PLAN.md 6) |
| `seeds/calibration/person_track_extrinsics/result.json` -- sequence 1's camera-person track fit; `motion detect-person-motion` and the renderer fell back to it when not given a translation or a detection | never read: `--translation-y-m`, `--motion-detection-result` and `--pose-audit` are required arguments now, the chain passes this sequence's own |
| `seeds/calibration/person_motion_detection_all_frames/` -- HuPR sequence 1's detection and pose audit, the renderer's and media's defaults | the same: the chain's per-sequence `seq<S>_detect_v2` and `seq<S>_pose_v2`, required arguments |
| `seeds/calibration/receiver_noise_*`, `diffuse_floor_calibration`, `seeds/rtpose/calibration/noise_envelope.json`, `sequence_56_angle_axes` | unread since the learned floor (`calibration calibrate-angle-axes` was removed; `noise-envelope` is a diagnostic) |

Everything under `seeds/` except `rtpose/*.json` is therefore unread and
can be deleted; it is left in place until you say so.

## Values that are not files

| value | where | what |
|---|---|---|
| ~~`HUPR_SCENE_ALIGNMENT`~~ | the HuPR scaffold | gone: `reconstruction.fit_range_alignment` fits the monocular scene's range translation into the radar frame from the recording, on the GPU (0.3 s): the min-max Pearson correlation of the camera-posed cloud's (range, angle) occupancy against the person-free measured map (the per-cell 25th percentile of loop-mean \|RA\| over every 4th frame), mean over both boards, inside the cloud's own angle bins, beyond 0.5 m. Range only (lateral and height 0: the co-located rig): the map does not identify them -- competing optima 0.6-1.4 m apart within 0.03 of the best score on every sequence -- while the range optimum is unique: 256 -0.34 m (r 0.25 -> 0.47), 38 -0.82 (0.16 -> 0.48), 102 -0.73 (0.16 -> 0.50). The old constants scored 0.34 / 0.16 / 0.44 on the same objective; 256's was a hand-picked candidate (`step3 chose extra0`), not a fit. |
| `BACKGROUND_ITERS` 75, `PERSON_EPOCHS` 1 (`chain.py`; `MM4DPS_BACKGROUND_ITERS`, `MM4DPS_PERSON_EPOCHS`) | preprocessing | the length of the two material initialisations the trainer refits for 400 epochs (docs/FIDELITY_PLAN.md 6) |
| the scaffold's sampled frames `0,150,300,450`, `hfov_deg` 40, the assumed HuPR intrinsics (40 degrees at 512 px, for the joints' image projection) | HuPR scaffold and person sequence | HuPR publishes no camera calibration; the camera is placed at the rig, co-boresighted, and the radar registers the person's range |
| `radar_depth_scales(min_confidence=0.35, smooth_frames=7, polish_frames=51, scale_bounds=(0.5, 2.0))`, `radar_person_range(min_bin=10, max_bin=200, window=5)` | person sequence | the per-frame radar registration of the GVHMR person |
| `--near-bins 15`, coupling `--stride 3` | coupling | the range bins treated as coupling, not scene; every third frame in the mean |
| `--doppler-taps 7`, PSF spread 15 | trainer | the Hann Doppler PSF support and the range PSF spread (fixed; the 3DPS paper) |
| 400 epochs, cosine schedule, `--gain-lr 1e-2` (the receiver gain and the joint gains) | trainer | uniform across sequences, never per sequence |
| the annotation-frame offset (0.2 m left of the LiDAR), the 180-degree board rotation | `data.py`, `reconstruction.py` | RT-Pose rig geometry, verified from the data (docs/RTPOSE_4DPS.md) |
| the camera/LiDAR file pairing: file stem = `a` x radar frame + `b`, fitted on the annotation's pairs (`RTPoseSequence.clock`; 3 f - 4 on 85 sequences, 2 f - 2 on 17, exact on every annotated frame) | `data.py` | which GVHMR row and LiDAR frame belong to a radar frame (docs/RTPOSE.md) |
