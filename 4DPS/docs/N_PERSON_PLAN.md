# Plan: 4DPS for N-person sequences

Status: 2026-09-14, a plan only -- nothing here is implemented. Motivation
and the measurements behind it: docs/TWO_PERSON_AND_SEQ21.md (38 of the
106 RT-Pose folders on disk are two-person recordings; the dataset's
annotation and the authors' released baseline treat them as one annotated
subject plus an unannotated second person).

## 0. The rule that governs every step: N = 1 is bitwise the shipped chain

Every change below is a generalisation from one person to N, and the
**first accuracy gate of every step is that N = 1 reproduces the delivered
single-person products exactly** -- the same person frames, the same
targets, the same loss at the first step, the same rendered cube -- so the
336 delivered sequences are never re-trained. Concretely, each step ships
with a gate that runs the new code on RT-Pose 183 and HuPR 256 and compares
against the delivered checkpoint (`render_inputs`, which holds the person
frames, the scene, the masks and the commands) and product:

| what | gate | tolerance |
|---|---|---|
| person frames (positions, normals, visibility, joints) | equal to `render_inputs.person.*` | bitwise |
| the LiDAR / monocular scaffold | equal to `render_inputs.scene.positions_m` | bitwise |
| the trainer's targets and first-step loss | `--dump-targets`, first step | 1e-6 relative (the existing trainer gate) |
| the rendered cube from `chain rerender` | against the delivered cube | the float16 product's quantisation (2.4e-4 of the frame peak), as gated on 2026-09-13 |

The N-person code paths are taken only when the sequence has N > 1 (the
dataset's `Subject type`, or the tracker finding more than one persistent
track); with N = 1 every new structure collapses to today's -- one track,
one point set, one field, one mask -- through the same functions, not a
parallel implementation. That is what makes the gate meaningful and what
lets the single-person results stand without another GPU-hour.

## A. GVHMR + YOLO tracks for N people (`gvhmr_fast_dump.py`, `gvhmr.py`)

Today: `track_one_person` keeps the longest, then largest, YOLO track;
ViTPose / HMR2 / GVHMR run on that one track's crops; the npz holds
`vertices_camera (frames, 6890, 3)`.

Plan:
1. `track_people(tracker, frames, n)`: the same YOLO tracking with
   `persist=True`; rank tracks by length then area exactly as now; keep the
   top `n` (n = 1 gives today's choice, so the N = 1 gate is the current
   selection by construction). Interpolate gaps and smooth per track as
   now. Tracks shorter than a minimum (say 30% of the frames) are dropped;
   if fewer than `n` remain, N is reduced and recorded.
2. Run ViTPose + features + GVHMR **per track** (GVHMR is static-camera
   and single-subject; running it once per track is the demo's own way).
   Output `vertices_camera (N, frames, 6890, 3)` with a `track_valid
   (N, frames)` mask (a person out of frame has no mesh that frame), plus
   each track's bbox series for step B.
3. Identity across frames is the tracker's; an ID switch shows up as a
   discontinuity in the pelvis trajectory and is the one failure mode to
   check on the two-person set (RT-Pose 100-105, 117-140, 170-175). A
   swap detector (pelvis jump > 1 m between consecutive frames) flags the
   sequence in metadata rather than fixing it silently.
4. `--n-people` flag on the tool; the chain passes the dataset's `Subject
   type` (RT-Pose `filemeta.json`: Single-/Multi-Person -> 1 / 2) or a
   `--people N` override. HuPR is single-person throughout.
5. Cost: GVHMR's inference scales with N (~30 s -> ~55 s for two); still
   overlapped with the reference stage.

Gate: N = 1 on 183 / 256 -> the npz bitwise equal to the delivered
`gvhmr_vertices.npz`.

## B. Per-frame radar registration for N people (`reconstruction.radar_depth_scales`, `person_range_series`, `run_person_sequence`)

Today: one range per frame -- the moving-power **range profile's** peak
(the power summed over angle), divided by the tracked mesh's centroid
range, gives one depth scale per frame. With two movers the profile has
two peaks and the argmax follows the stronger.

Plan: register each person against the moving power **at that person's
angle**, not the angle-integrated profile.
1. Project each track's mesh centroid into the radar's (range, azimuth)
   for the frame (the camera placement gives the direction exactly; only
   the range is uncertain, since that is what GVHMR gets wrong). The
   clutter-removed RA cube (already computed in `person_range_series`,
   `range_angle_slow_time_cube_frames_torch(remove_static_clutter=True)`)
   is reduced to a range profile **within a +/- 8 deg azimuth window
   around each person's direction** instead of over all angles; its peak,
   parabola-refined, is that person's radar range; the confidence is the
   window's energy fraction as now. For N = 1 the window is widened to the
   full angle axis, which is today's profile exactly (the gate).
2. Two people at the same azimuth and different ranges are separated by
   range (two peaks in the same window: take the one nearest the mesh's
   prior range scaled by the running median scale); two people at the
   same range and different azimuths are separated by the window. Two
   people at the same range and azimuth are one radar target and cannot
   be separated -- flag, do not fit.
3. Per-person scale series with the same median filter, Savitzky-Golay
   polish and bounds as now; `depth_scale_per_frame (N, frames)` in the
   metadata; person frames written per person (`person_frames/p<k>/
   frame_XXX.npz`), or one file with a leading person axis -- the loader
   in D decides; keep one layout.
4. Where the annotation exists (RT-Pose, 221-237 of 299 frames), the
   annotated subject's pelvis range is a **check**, not an input: the
   metadata records which track it matches and the per-frame residual, so
   the fit stays annotation-free as now while the report can say which of
   the N is the benchmark's subject.

Gate: N = 1 -> `depth_scale_per_frame` bitwise the delivered series (the
full-angle window is the same tensor reduction); the two-person set ->
radar-vs-mesh range correlation per person > 0.9 (single-person sequences
sit at 0.97-0.99), the registration IQR width < 0.1.

## C. The LiDAR scaffold for N people (`reconstruction.run_lidar_scaffold`)

Today: on each sampled annotated frame, LiDAR points within
`PERSON_CLEAR_RADIUS_M` = 0.6 m of the annotated subject's column are
removed; a second person's points stay in the "static" scene and get a
material in the background fit.

Plan:
1. Clear a column per person. The columns come from the N registered
   meshes of step B at the sampled frames (the mesh, placed by B, is a
   better person footprint than a 0.6 m cylinder; its convex hull in the
   ground plane dilated by 0.3 m); for the annotated subject the
   annotation's column is also cleared, as now. N = 1 keeps the
   annotation-column clearing exactly (the mesh-based clearing is added
   only for N > 1 -- or, if it is adopted for N = 1 too, that is a
   scaffold change and is gated at 400 epochs, not bitwise).
2. The sampled frames themselves: prefer frames where the N people are
   all inside the LiDAR's field of view and apart (> 1 m), so every
   column is a clean cut; the sampler already takes annotated frames
   spread over the sequence.
3. The static |RA| map the background fit and the range alignment read
   (`clean_scene_maps`, the per-cell 25th percentile over frames) is
   person-free by construction (a mover never sits in the 25th
   percentile) and needs no change.
4. HuPR: monocular scaffold, single-person dataset -- unchanged.

Gate: N = 1 -> `pcl.npy` and `scaffold_metadata.json` bitwise the
delivered scene (`render_inputs.scene.positions_m`).

## D. The model and the renderer for N people (`velocity_field.py`, `video.py`, `signal_model.py`, `scene.py`)

Today: one point set (6890 SMPL vertices, per-frame positions and
visibility), one joint velocity field `(frames, 14, 3)` in the DCT basis,
one skinning (`soft_joint_weights_torch` over 14 joints), one shared
6-parameter material and area scale, one learned per-frame normal set,
one micro-Doppler mask (`person_range_angle_masks_torch`, the person's
(range, angle) support), the scene shadow mask from one mesh.

Plan -- a person axis, everywhere the person is indexed:
1. **Point sets.** The per-frame person cloud becomes the concatenation
   of the N meshes' visible vertices with a `person_id (points,)` label
   (`assert_persistent_person_geometry` extends to N fixed sets). The
   renderer's per-frame gather (`video._render_frames`: `cached["pcl"]`,
   `cached["visible"]`, `person_global_index`) reads the concatenation;
   the splat kernels are a sum over points and see no difference.
2. **Velocity field.** `VelocityFieldModel` holds `(N, frames, 14, 3)`
   (temporal coefficients `(N, bands, 14, 3)`), initialised from each
   person's camera joint velocities. `point_velocities` skins each
   person's points from **its own** 14 joints: the skinning weights are
   computed per person against that person's joints (`soft_joint_weights_torch`
   called per person and block-diagonally assembled -- a point never
   skins to the other person's joints). Joint scattering gains
   `(N, 14)` with the geometric-mean pin per person, or shared `(14,)`
   across people (a decision to gate: shared is the safer prior with
   fewer parameters).
3. **Materials and normals.** `SharedHumanMaterial` per person `(N, 6)`
   with per-person area scales (two bodies differ in size and clothing);
   learned normals per person's visible points, as now, indexed through
   the concatenation.
4. **Micro-Doppler mask.** `person_range_angle_masks_torch` per person;
   the trainer's MD marginal and the media's spectrograms read the
   **union** mask (the real radar does not know who is who); a
   per-person marginal is kept as a diagnostic (the rendered side can
   split by `person_id`; the real side cannot).
5. **Visibility.** The self-occlusion / FOV ray cast runs against the
   union of the N meshes' triangles (person A occludes person B); the
   scene shadow mask likewise (`sequence_visibility`, `scene-shadow-mask`)
   -- both take a triangle set and are indifferent to how many bodies it
   came from.
6. **Motion frames.** The pose-speed selection and the radar motion
   detector use the max over people (a frame counts as motion if anyone
   moves).
7. **Renderer flags.** `--people N` recorded in `render_result.json` and
   the checkpoint's `render_inputs` (`person.positions_m (N, frames, V, 3)`
   etc.; N = 1 keeps today's shapes by squeezing the axis, so
   `chain rerender` on a delivered checkpoint is unchanged).

Gate: N = 1 -> the first training step's loss to 1e-6 of today's and the
rendered cube within the float16 quantisation (the same code path with a
length-1 person axis).

## E. The targets and the loss for N people (`velocity_field.two_term_loss`, the target build)

The real RA, 63-plane and MD targets already contain everyone -- they are
the recording. What changes is only what the rendered side puts against
them and which cells the MD term reads:

1. **Loop-0 RA and 63-plane RA terms**: unchanged in form; the rendered
   profile is the sum over the N point sets, so the terms are
   well-specified once D renders both. Nothing to change in
   `two_term_loss`.
2. **MD term**: the rendered marginal and the real marginal both under
   the **union** person mask (D.4); the 13-bin band unchanged. With one
   mask per person and separate rendered marginals, a per-person MD
   term is possible on the rendered side but has no real counterpart --
   so the term stays a single union marginal. The Doppler support
   (`--doppler-support-snr`) is per frame from the real Doppler at the
   union's range rows: automatically the union of both people's support.
3. **Divisors**: per frame and radar from the real target, as now.
4. **Evaluation** (`evaluation.py`): RA / RD metrics are whole-map and
   need no change; micro-Doppler metrics on the union mask; the motion
   subset as D.6. Report per sequence, and additionally the annotated
   subject's per-person rendered marginal against ... nothing real --
   so no per-person real metric is claimed.

Gate: N = 1 -> targets bitwise, loss identical (the union of one mask is
the mask).

## F. Order, cost, and what "done" looks like

| step | depends on | effort | gate |
|---|---|---|---|
| A tracks | -- | 1 day | N = 1 bitwise; two-person set: two persistent tracks on 38/38 |
| B registration | A | 1-2 days | N = 1 bitwise; two-person set: per-person range correlation > 0.9 |
| C scaffold | B | 0.5 day | N = 1 bitwise |
| D model/render | A, B | 3-4 days | N = 1 first-step loss 1e-6, cube at float16 quantisation |
| E loss/eval | D | 1 day | N = 1 bitwise targets |
| sweep of the 38 two-person sequences | all | ~7 h on two GPUs (two bodies: ~1.4x the step) | RA / RD / MD on the 38 within the one-person distributions (RA 0.816-0.902 IQR) |

The single-person products are not re-trained at any point; the paper's
Table 3 keeps its 336 rows and gains the 38 re-fitted two-person rows in
place of their single-body fits.
