# OctoNet visualizations (Scene 1, recording 20240518134358 "sit" / 20240518135416 "walk", subject 2)

Produced on `mmsd-h100` by `python -m viz.<group>` from `nbs/` (scripts in `nbs/viz/`), 2026-10-27.
Frame everywhere: MoCap (OptiTrack) world, metres, Z up, floor at 0, origin = the subject's standing spot.

**Global caveats.** Node positions are measured (OptiTrack rigid bodies `Node1`..`Node5`). Camera
orientations are **calibrated** by `preprocessing/octonet_calibrate.py` (pitch/roll from a RANSAC floor
plane, yaw from the MoCap-synchronised person; position = the node rigid body, camera-to-marker offset
ignored); the Vayyar/UWB on node_1 share node_1's camera yaw by assumption. WiFi Tx/Rx and the obstacle
are schematic from paper Fig. 2(a) (±0.5 m). Depth = 8-bit PNG × 10 m/255 with nominal D455 intrinsics.
Vayyar SFCW shows the static scene only: the person is not recoverable (WRITE_UP #19).

| node | yaw (deg) | assumed yaw | yaw spread | pitch | roll | depth scale | floor height vs marker z |
|---|---|---|---|---|---|---|---|
| 1 | −179.1 | −179.8 | ±0.2 | −1.6 | +1.7 | 1.027 | 0.92 / 0.90 m |
| 3 | −136.6 | −144.8 | ±1.2 | −2.3 | +0.3 | 1.075 | 0.87 / 0.93 m |
| 5 | −25.0 | −29.7 | ±0.9 | +2.6 | +0.7 | 1.167 | 0.84 / 0.89 m |

**Depth scale (resolves the node-5 "10 % long" item).** The person blob is the camera-facing body surface,
~0.12 m in front of the MoCap hip; with that correction the person gives an absolute per-camera depth scale
(measured = scale × true). The walls two cameras share confirm it is a scale, not a camera-to-marker offset:
node 5's boresight is along +X, so an offset cannot move the −Y wall, yet that wall sat 0.65 m further for
node 5 than for node 3 before the scale and 1 cm apart after it (−3.46 vs −3.47 m); nodes 1 and 3 put the −X
wall at −3.52 / −3.41 m. Likely cause: per-node 8-bit depth encoding with different max distances. Cross-camera
median nearest-neighbour distance now: 1–5 0.08 m, 3–5 0.10 m, 1–3 0.20 m. A landmark approach (the other
nodes as range targets) was tried and rejected: the detections were background clutter
(`calibration_v2.png`, kept as the record). Horizontal camera-to-marker offsets remain unobservable (set 0).

**SFCW static target (`nbs/method/sfcw_target.py`, figure `render_vs_gt/sfcw_static_target.png`).** The
objective is the clutter-guarded temporal-mean channel: bins 0–5 and the last 6 masked, log|range profile|
per pair minus its guarded-band median, L1 against the render, weighted per bin by the cross-recording
correlation of |rp| (median 0.87, min 0.74 at 1.2–2.5 m over 5 Scene-1 activities, so the static target is
recording-invariant and is pooled: `sfcw_target_S1/target_pool5.npz`). Mean-channel SNR 22–38 dB (median 28),
weakest at 1.3–2.3 m. The residual is slow drift, not frame noise: lag-1 autocorrelation 0.7, linear decay over
~28 s, effective independent frames ≈ 4.4 of 283, so split-half tests on interleaved frames are uninformative
(0.996) and only the cross-recording reliability is a usable weight. Use `kind='ra'` only once the pair order /
Tx pitch assumptions are verified (mount-yaw check).

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
- `sfcw_gt_vs_rendered.png` — GT: clutter-removed static range profile (bins 0–5 and the last 6 guarded), residual time–range map, static RA. Rendered: `sfcw_forward.render_sfcw_pairs` on the calibrated scaffold (Vayyar array at node_1, uniform ITU concrete). GT is coupling/sidelobe dominated (peak 0.22 m). Nothing at 2.92 m in either, as expected. Pair order tx-major + uniform Tx pitch are assumptions (azimuth only).
- `sfcw_static_target.png` — the SFCW target definition: split halves, per-bin reliability / SNR, pooled vs single recording, residual autocorrelation.
- `mount_yaw_sweep.png` — node_1 Vayyar / UWB boresight sweeps vs the measured static returns, and the Tx×Rx direct-coupling heat maps (pair-order check).
- `uwb_gt_vs_rendered.png` — GT CIR vs range (bin 0.64 cm; the per-recording pickle holds ONE 1535-bin frame). Rendered = PROTOTYPE incoherent pulse-sum, not a 4DPS head (pending).
- `wifi_gt_vs_rendered.png` — per receiver: GT |CSI| (frame + temporal mean, both antennas) and CIR; rendered `wifi_forward.render_csi` with schematic Tx/Rx. GT notches are HT40 null/pilot subcarriers; 40 MHz gives 7.5 m taps, so only the first 1–2 are informative. Rx-4 (NLOS) GT is not visibly suppressed.
- `captions.txt`

## Task B (4DPS copy adapted to OctoNet): `4DPS/docs/OCTONET.md`
- `geometry_init/person_smpl_fit.png` — SMPL fitted to the OptiTrack skeleton (walk 2.34 cm, sit 3.41 cm mean joint RMSE), the person stage.
- `render_vs_gt/sfcw_fit_static.png` — first static SFCW fit on the calibrated scaffold: loss 1.157 → 0.627 (held recording 1.140 → 0.653).

## Open items these figures expose
1. **Vayyar element layout.** The direct-coupling energy over the 20×20 Tx×Rx grid is smooth (so the 400 pairs are a 20×20 grid) but does not follow a uniform-pitch L-array: the Tx elements sit in two sub-groups (`mount_yaw_sweep.png`, `octonet_mount_yaw.py`). The azimuth/RA axis in `vayyar_radar_config` is therefore unverified; obtain the IMAGEVK-74 element coordinates (vSDK / datasheet in the old NBS `pdfs/IMAGEVK-74.pdf`) before any RA-based loss. Until then, fit SFCW on the pair-wise range profiles only (`kind='logmag'`).
1b. Radar/UWB mount yaw on node_1: **unmeasurable** from the static returns. The SFCW loss vs yaw is a flat basin from about −100° to +120° around the camera yaw (it only rises when the radar faces the wall behind it); the UWB prototype curve has no peak above z = 3. Both keep the camera yaw, level, recorded as an assumption in `extrinsics.json`. Also exposed: `render_sfcw_pairs` has no antenna element pattern (a cos^p pattern was assumed for the sweep).
2. SFCW drift: the per-recording temporal mean is biased by up to a few dB at 1.2–2.5 m by thermal drift; a per-pair gain/phase ramp model would raise the usable SNR there.
3. UWB head (pulse PSF) and a real UWB frame stream (only one frame per recording was found — check other recordings / the raw HF files).
4. WiFi: HT40 null/pilot mask before any loss; Tx/Rx positions need measuring.
