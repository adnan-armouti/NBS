# Geometry videos v2 (OctoNet Scene 1)

Static scene points and the dynamic person, in the style of the 4DPS scene-and-person videos (dark background,
grey scene, warm person, skeleton on top, metre grid, radar marker and range ring). Made by
`nbs/viz/geometry_video.py` on `mmsd-h100` and copied here.

| file | content |
|---|---|
| `20240518135416_walk.mp4` | walking in place, 25 s at 10 fps |
| `20240518134358_sit.mp4` | sitting, 25 s |
| `20240525164329_pushup.mp4` | push-ups, 25 s |
| `20240521213724_jumpingjack.mp4` | jumping jacks, 25 s |
| `scene_overview.mp4` | 15 s: the static scaffold orbiting; then coloured by source camera (red node 1, green node 3, blue node 5); then the person added |

Each recording video: **left** = top-down X–Y in the MoCap world frame (metres, Z up, origin = the subject's standing
spot, star); **right** = 3-D perspective, slowly orbiting, floor plane at Z = 0. Title: recording stamp and activity;
subtitle: time from the first MoCap-covered frame, frame counter, person source and rate.

## What is drawn, and where it comes from
- **Static scaffold (grey, 40k of 74.6k points)**: `python -m mm4DPS.reconstruction scaffold --dataset octonet`
  output (`4dps_scaffold_S1/pcl.npy`): temporal-median depth of the three RealSense D455 streams (nodes 1/3/5; 8-bit
  0–10 m PNGs × the per-camera depth scale 1.027/1.075/1.167), back-projected with nominal intrinsics, placed with the
  **calibrated extrinsics** (`octonet_calibrate`: node rigid-body position + floor height, floor-plane pitch/roll,
  MoCap-synchronised person yaw), fused, 5 cm voxels. The **person column (r < 0.7 m) is cut** from the scaffold, so
  the floor under the subject is a hole by construction.
- **Person (gold points = 2 000 of the 6 890 SMPL vertices; orange bones = the 24 SMPL joints)**: SMPL fitted to the
  20-joint OptiTrack skeleton (`mm4DPS/octonet_person.py`; mean joint RMSE 2.3 cm walking, 3.2–3.4 cm seated /
  push-up), re-fitted at **10 Hz** for these videos (`$NBS_DATA_OUT/person10hz_S1/<rec>/`); frames outside MoCap
  coverage are skipped at the start.
- **Sensors**: the five node rigid bodies (coloured squares, labels), the Vayyar/UWB at node 1 with its boresight
  (red arrow; = the camera yaw by assumption) and the 3.71 m SFCW unambiguous-range ring (red dotted); the 4.1 m
  sensing square (green dashed); WiFi Tx (red triangle) and Rx 1–4 (blue triangles) are **schematic** from the paper's
  Fig. 2(a) (±0.5 m), marked with `~`.

## Known caveats visible in the videos
- The scaffold does not contain the node's own body/cover near node 1 (the cameras cannot see it) nor anything
  inside the cut person column.
- Node 3's camera has large invalid depth regions, so its contribution is sparse (overview, green).
- Lying/low poses (push-up) bring SMPL vertices to within a few cm of the floor; occasional vertices below Z = 0 are
  fit error (joint RMSE ~3 cm), not geometry.
- The 3-D panel subsamples the scaffold for speed; distances are exact, densities are not.

## Commands
```
# on the node, after: source tools/env_node.sh
cd nbs
python -m viz.geometry_video --rec 20240518135416 --rec 20240518134358 --rec 20240525164329 --rec 20240521213724 --overview --seconds 25 --fps 10 --workers 20
# 10 Hz person refit (65 s per recording, GPU):
python -c "from mm4DPS.octonet_person import run_octonet_person; ..."   # see geometry_video.py docstring
```
