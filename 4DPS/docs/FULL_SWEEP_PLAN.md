# Running 4DPS over both datasets: storage, prerequisites, the driver, the timeline

Status: 2026-09-13, `main` at the float16 product. Numbers from today's
twelve end-to-end runs (docs/TRAINING_SPEED_PLAN_3.md 4) and a coverage
check of every sequence on disk.

## 1. What is runnable

| dataset | on disk | runnable | why not the rest |
|---|---|---|---|
| HuPR | 276 sequences in the archives (1-276), 600 frames each | **235** | 41 sequences (82, 111-118, 125-132, 136, 139, 140, ...) are in no split of the official config, so they have no `hrnet_annot_*` annotation -- the scaffold's person mask reads it (`reconstruction.frame_annotation`) and refuses. These are exactly the sequences the HuPR release does not annotate. |
| RT-Pose | 106 folders | **101** | `0` is empty; `107` and `193` have no radar `bin/` recording; `155` has no annotation (the LiDAR scaffold samples annotated frames). |

31,629 RT-Pose frames over the 101 (94 x 299, the rest 300-509);
141,000 HuPR frames over the 235.

## 2. Storage, at the float16 product (4 bytes a complex sample)

| dataset | per sequence | total |
|---|---|---|
| RT-Pose | 12.58 MB a frame x frames: 3.76 GB at 299, up to 6.4 GB at 509; + 12 MB of checkpoint, metrics, history, log | **398 GB** |
| HuPR | 2 x 0.472 GB + 29 MB | **229 GB** |
| both | | **~630 GB** |

Plus the person's GVHMR mesh kept with each run (`gvhmr_vertices.npz`,
45 MB a HuPR sequence, 22 MB an RT-Pose one): **~13 GB** more.

`/home/adnan/Documents/Data` is on the root NVMe, which has **1.5 TB
free** (57% used) after clearing the scratch. The whole sweep with every
sequence's product fits with ~850 GB to spare; nothing needs deleting
first. `--intermediates` must stay OFF (it would add ~25 GB a sequence).

## 3. GVHMR is in the chain (2026-09-13)

The person stage runs GVHMR itself (`mm4DPS.gvhmr vertices`): 4-6 s to
encode the RGB frames losslessly, 55-60 s of GVHMR inference on the run's
GPU, deterministic given the frames. Measured end to end with one sequence
per GPU, at the fast tool (frames in memory, half-precision ViTs, started
with the reference stage): RT-Pose 183 **514 -> 544 s** (+30 s), HuPR 256
**292 -> 323 s** (+31 s); the metrics unchanged within their spread. No batch step before
the sweep; nothing under `seeds/` but the RT-Pose extrinsic.

## 4. The sweep: a shell loop, as today's runs were

Two queues, one per GPU (`final_queue.sh`'s loop with three additions):
for each runnable sequence -- the list from section 1, longest first --
skip it if `<destination>/<dataset>/seq<S>/metrics.json` exists, run
`python -m mm4DPS.chain run <dataset>_seq<S> <S> --dataset <dataset>` (no
artifacts, no intermediates), `mv outputs/<run>/seq<S>` to the destination
(same filesystem, instant), remove the run shell, append one line to
`<destination>/<dataset>/manifest.csv` (sequence, exit code, wall seconds,
the per-stage seconds line, the four headline metrics). A failed sequence
leaves its log as `seq<S>/FAILED.log` and the loop continues; rerunning the
script picks up where it stopped.

Layout at the destination:

```
/home/adnan/Documents/Data/4DPS_products/
  hupr/    manifest.csv
           seq1/  horizontal_rendered.npy  horizontal_rendered_scale.npy
                  vertical_rendered.npy    vertical_rendered_scale.npy
                  gvhmr_vertices.npz  checkpoint.pt  metrics.json  history.json  run.log
  rtpose/  manifest.csv
           seq1/  cascade_rendered.npy  cascade_rendered_scale.npy
                  gvhmr_vertices.npz  checkpoint.pt  metrics.json  history.json  run.log
```

`checkpoint.pt` is one dict of three sections:

| key | what |
|---|---|
| `velocity_field` | the trainer's checkpoint: `joint_velocity_field_mps (frames, 14, 3)`, `learned_normals {frame: (V, 3)}`, `inherited_joint_scattering_gains (14,)`, `absolute_gains`, `noise_floor_power`, `skinning`, `background_point_gains` |
| `person_material` | the material checkpoint the render reads: `shared_background_state` (the scene's `positions (N, 3)`, `rotations (N, 4)`, `raw_materials (N, 6)`), `shared_human_raw_material (V, 6)`, `shared_human_area_scale`, `person_radar_log_gain_delta` |
| `render_inputs` | the geometry exactly as the renderer receives it (`chain.render_inputs`): `scene.positions_m (N, 3)` aligned to the radar, `scene.normals (N, 3)` from the trained quaternions, `scene.raw_materials` / `physical_materials (N, 6)` (`material_columns`: eps_real, eps_imag, sigma_h_m, correlation_length_m, coherent_fraction, thickness_m), `scene.visible (frames, N)` bool (the person's shadow), `scene.alignment` (the fitted range translation); `person.positions_m (frames, V, 3)` registered to the radar frame by frame, `person.visible (frames, V)`, `person.joints_m (frames, 14, 3)`, `person.normals` (learned) and `mesh_normals` (SMPL), `person.raw_materials` / `physical_materials (V, 6)`, `area_scale`, `joint_velocity_field_mps`, `joint_scattering_gains`, `registration` (the per-frame depth scales), `placement` (camera, extrinsic, the camera-frame pairing per radar frame); `radar.absolute_receiver_gain`, `radar.antenna_coupling {radar: complex}`, `radar.noise_floor_power`; `frame`: the coordinate convention (metres, +X right, +Y boresight, +Z up; index f = radar frame f) |

Also in `render_inputs`: `motion_frames` (the motion-detection result and
the pose audit, the render's two required frame selections),
`person_sequence_metadata` (the person stage's record: placement, the
camera-frame pairing, the per-frame registration), and `commands` (the
git SHA, the trainer and render command lines verbatim with their
results, the coupling fit). That is everything a forward render reads
except the dataset's recording and the tracked `configs/`:

    python -m mm4DPS.chain rerender <seq>/checkpoint.pt --output-dir <dir>

rebuilds the render's inputs from the checkpoint and runs the recorded
command against them. Gated 2026-09-13 on RT-Pose 10 and HuPR 256 (10-epoch
runs, `.work/` already removed): the re-rendered cubes match the runs' own
products to the float16 product's quantisation -- per-frame scales
identical, max |difference| 2.44e-4 of the frame peak (half a float16 ulp
at the top of the range), 0.13-0.15% of cells one rounding step apart
(float32 accumulation-order noise, below float16 resolution).

`gvhmr_vertices.npz` (the SMPL vertices in the camera frame, before
registration) is kept automatically -- it is a product, not an artifact, and
does not depend on `--artifacts`.

## 5. Timeline (two 4090s, one sequence each)

| step | GPU-hours | wall |
|---|---|---|
| HuPR, 235 x ~323 s | 21.1 | ~10.5 h |
| RT-Pose, 102 x ~580 s (frame-scaled) | 16.4 | ~8.2 h |
| total | ~37.5 | **~19 h**, one unattended day |

Per-sequence wall today: RT-Pose 500-537 s, HuPR 292-306 s (no artifacts,
no intermediates). The sweep's own overhead is the move and the check,
under a second a sequence.

## 6. Order of work

1. The two queue scripts (an hour), dry-run on one sequence per dataset
   writing to `4DPS_products/`, to check the move, the skip and the
   manifest line.
2. HuPR sweep, then RT-Pose (the RT-Pose queue starts on a GPU as soon as
   the HuPR queue on it drains).
3. When both manifests are complete: a summary table across all 336
   sequences (the distributions of the four headline metrics, wall time
   per frame, failures with their reasons) to `docs/`.

Decided: destination `4DPS_products`; the 41 unannotated HuPR and 3 broken
RT-Pose sequences are skipped (listed in the manifest as such).
