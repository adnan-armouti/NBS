# OctoNet visualizations (Scene 1, recording 20240518134358 "sit" / 20240518135416 "walk", subject 2)

Produced on `mmsd-h100` by `python -m viz.<group>` from `nbs/` (scripts in `nbs/viz/`), 2026-10-27.
Frame everywhere: MoCap (OptiTrack) world, metres, Z up, floor at 0, origin = the subject's standing spot.

**Global caveats.** Node positions are measured (OptiTrack rigid bodies `Node1`..`Node5`); sensor
*orientations* are assumed (horizontal, aimed at the origin) until `octonet_calibrate` exists. WiFi Tx/Rx
and the obstacle are schematic from paper Fig. 2(a) (±0.5 m). Depth = 8-bit PNG × 10 m/255 with nominal
D455 intrinsics. Vayyar SFCW shows the static scene only: the person is not recoverable (WRITE_UP #19).

## nodes/ — node layout + sensor availability
- `nodes_topdown_2d.png` — top-down plan: measured nodes 1–5 (with heights), subject spot, 4.1 m sensing square (assumed centred), schematic WiFi Tx/Rx-1..4 + obstacle, D455 87° FoV wedges (nodes 1/3/5), Vayyar R_max 3.71 m arc.
- `nodes_3d.png` — the same in 3D with mounting heights, camera boresights and a MoCap skeleton at the origin for scale.
- `node_sensor_matrix.png` — node × sensor: green = directory exists on disk; number = recordings (of 976) with a path in `cut_manual.csv`. Vayyar + UWB only on node_1; RGB-D on 1/3/5; WiFi on 1–4; ToF on 4; TI mmWave + IRA on all.
- `captions.txt`

## geometry_init/ — initialized scene + person geometry, and how
- `scene_scaffold_per_camera.png` — per RGB-D node: temporal-median depth (30 frames of the walk clip) with the person column outlined, and its single-camera cloud. Extrinsics-free; the trustworthy panel.
- `scene_scaffold_topdown.png`, `scene_scaffold_3d.png` — the three clouds fused under the ASSUMED poses, person column cut, coloured by camera. The clouds do not close into one room (yaw is wrong): this is the concrete motivation for `octonet_calibrate`.
- `person_skeleton.png` — 20-joint MoCap skeleton with bones for standing / sitting / lying (`sleep`) frames; joint order = OptiTrack CSV bone order (verified to 0.0 mm): Hip, Chest, Neck, Head, L/R Shoulder-UArm-FArm-Hand, L/R Thigh-Shin-Foot-Toe.
- `init_pipeline.png` — block diagram of the init pipeline; dashed = pending (`octonet_calibrate`, SMPL fit).
- Saved scaffold: node `/data/mmsd/NBS/data/OctoNet/scaffold_S1_assumed/pcl.npy` (50,032 × 6, xyz + PCA normals, 5 cm voxels) + `scaffold_metadata.json`.
- `captions.txt`

## render_vs_gt/ — measured vs rendered
- `sfcw_gt_vs_rendered.png` — GT: clutter-removed static range profile (bins 0–5 and the last 6 guarded), residual time–range map, static RA. Rendered: `sfcw_forward.render_sfcw_pairs` on the scaffold (Vayyar array at node_1, uniform ITU concrete). GT is coupling/sidelobe dominated (peak 0.22 m); rendered peaks at the nearest floor (0.75 m). Nothing at 2.92 m in either, as expected. Pair order tx-major + uniform Tx pitch are assumptions (azimuth only).
- `uwb_gt_vs_rendered.png` — GT CIR vs range (bin 0.64 cm; the per-recording pickle holds ONE 1535-bin frame). Rendered = PROTOTYPE incoherent pulse-sum, not a 4DPS head (pending).
- `wifi_gt_vs_rendered.png` — per receiver: GT |CSI| (frame + temporal mean, both antennas) and CIR; rendered `wifi_forward.render_csi` with schematic Tx/Rx. GT notches are HT40 null/pilot subcarriers; 40 MHz gives 7.5 m taps, so only the first 1–2 are informative. Rx-4 (NLOS) GT is not visibly suppressed.
- `captions.txt`

## Open items these figures expose
1. `octonet_calibrate`: per-camera yaw/pitch (and the radar/UWB mount) — the fused scaffold is wrong without it.
2. SFCW: the static return beyond the coupling sits near the per-frame noise floor (residual −1.2 dB vs mean); the SFCW objective must be the temporal-mean channel, clutter-guarded.
3. UWB head (pulse PSF) and a real UWB frame stream (only one frame per recording was found — check other recordings / the raw HF files).
4. WiFi: HT40 null/pilot mask before any loss; Tx/Rx positions need measuring.
