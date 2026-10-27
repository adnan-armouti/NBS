# OctoNet visualizations (Scene 1, recording 20240518134358 "sit" / 20240518135416 "walk", subject 2)

Produced on `mmsd-h100` by `python -m viz.<group>` from `nbs/` (scripts in `nbs/viz/`), 2026-10-27.
Frame everywhere: MoCap (OptiTrack) world, metres, Z up, floor at 0, origin = the subject's standing spot.

**Global caveats.** Node positions are measured (OptiTrack rigid bodies `Node1`..`Node5`). Camera
orientations are **calibrated** by `preprocessing/octonet_calibrate.py` (pitch/roll from a RANSAC floor
plane, yaw from the MoCap-synchronised person; position = the node rigid body, camera-to-marker offset
ignored); the Vayyar/UWB on node_1 share node_1's camera yaw by assumption. WiFi Tx/Rx and the obstacle
are schematic from paper Fig. 2(a) (±0.5 m). Depth = 8-bit PNG × 10 m/255 with nominal D455 intrinsics.
Vayyar SFCW shows the static scene only: the person is not recoverable (WRITE_UP #19).

| node | yaw (deg) | assumed yaw | yaw spread | pitch | roll | floor height vs marker z | range check cam/world |
|---|---|---|---|---|---|---|---|
| 1 | −179.1 | −179.8 | ±0.2 | −1.7 | +1.6 | 0.94 / 0.90 m | 0.989 |
| 3 | −136.3 | −144.8 | ±0.4 | −1.8 | +0.3 | 0.92 / 0.93 m | 1.035 |
| 5 | −25.0 | −29.7 | ±0.9 | +3.1 | +0.7 | 0.96 / 0.89 m | 1.101 |

Cross-camera median nearest-neighbour distance, assumed → calibrated: 1–5 0.33 → 0.06 m, 3–5 0.31 → 0.11 m,
1–3 0.14 → 0.17 m. ICP of nodes 3/5 onto node 1 asks for 13° / 0.3 m and 7° / 2.8 m (walls are ambiguous)
and is reported in `extrinsics.json` but NOT applied. Node 5 reads ranges 10 % long: a camera-to-marker
offset or a per-camera depth scale; open item.

## nodes/ — node layout + sensor availability
- `nodes_topdown_2d.png` — top-down plan: measured nodes 1–5 (with heights), subject spot, 4.1 m sensing square (assumed centred), schematic WiFi Tx/Rx-1..4 + obstacle, D455 87° FoV wedges (nodes 1/3/5), Vayyar R_max 3.71 m arc.
- `nodes_3d.png` — the same in 3D with mounting heights, camera boresights and a MoCap skeleton at the origin for scale.
- `node_sensor_matrix.png` — node × sensor: green = directory exists on disk; number = recordings (of 976) with a path in `cut_manual.csv`. Vayyar + UWB only on node_1; RGB-D on 1/3/5; WiFi on 1–4; ToF on 4; TI mmWave + IRA on all.
- `captions.txt`

## geometry_init/ — initialized scene + person geometry, and how
- `scene_scaffold_per_camera.png` — per RGB-D node: temporal-median depth (30 frames of the walk clip) with the person column outlined, and its single-camera cloud. Extrinsics-free; the trustworthy panel.
- `scene_scaffold_topdown.png`, `scene_scaffold_3d.png` — the three clouds fused under the CALIBRATED poses, person column cut, coloured by camera. Node 1 and node 3 agree on the far wall at X ≈ −3.75 m; node 5 sees the opposite wall at X ≈ +3.7 m.
- `person_skeleton.png` — 20-joint MoCap skeleton with bones for standing / sitting / lying (`sleep`) frames; joint order = OptiTrack CSV bone order (verified to 0.0 mm): Hip, Chest, Neck, Head, L/R Shoulder-UArm-FArm-Hand, L/R Thigh-Shin-Foot-Toe.
- `init_pipeline.png` — block diagram of the init pipeline; dashed = pending (SMPL fit).
- `calibration.png` — `octonet_calibrate` diagnostics per camera: floor inliers + person centroids on the median depth, yaw solve vs the MoCap hip, fused top-down assumed vs calibrated with overlap metrics.
- Saved scaffold: node `/data/mmsd/NBS/data/OctoNet/scaffold_S1_calibrated/pcl.npy` (≈54k × 6, xyz + PCA normals, 5 cm voxels) + `scaffold_metadata.json`; extrinsics at `calib_S1/extrinsics.json` (copy in `nbs/assets/calib_S1_extrinsics.json`).
- `captions.txt`

## render_vs_gt/ — measured vs rendered
- `sfcw_gt_vs_rendered.png` — GT: clutter-removed static range profile (bins 0–5 and the last 6 guarded), residual time–range map, static RA. Rendered: `sfcw_forward.render_sfcw_pairs` on the scaffold (Vayyar array at node_1, uniform ITU concrete). GT is coupling/sidelobe dominated (peak 0.22 m); rendered peaks at the nearest floor (0.75 m). Nothing at 2.92 m in either, as expected. Pair order tx-major + uniform Tx pitch are assumptions (azimuth only).
- `uwb_gt_vs_rendered.png` — GT CIR vs range (bin 0.64 cm; the per-recording pickle holds ONE 1535-bin frame). Rendered = PROTOTYPE incoherent pulse-sum, not a 4DPS head (pending).
- `wifi_gt_vs_rendered.png` — per receiver: GT |CSI| (frame + temporal mean, both antennas) and CIR; rendered `wifi_forward.render_csi` with schematic Tx/Rx. GT notches are HT40 null/pilot subcarriers; 40 MHz gives 7.5 m taps, so only the first 1–2 are informative. Rx-4 (NLOS) GT is not visibly suppressed.
- `captions.txt`

## Open items these figures expose
1. `octonet_calibrate` v2: solve the camera-to-marker offset (node 5's 10 % range ratio) and the radar/UWB mount orientation on node_1 (currently = camera yaw by assumption); consider a joint wall-alignment refinement with a sane prior instead of free ICP.
2. SFCW: the static return beyond the coupling sits near the per-frame noise floor (residual −1.2 dB vs mean); the SFCW objective must be the temporal-mean channel, clutter-guarded.
3. UWB head (pulse PSF) and a real UWB frame stream (only one frame per recording was found — check other recordings / the raw HF files).
4. WiFi: HT40 null/pilot mask before any loss; Tx/Rx positions need measuring.
