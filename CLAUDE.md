# NBS — working notes for Claude Code

## What this repo is
RF inverse rendering on **OctoNet** (NeurIPS 2025 D&B, `hku-aiot/OctoNet`): Vayyar IMAGEVK-74 **SFCW**
radar (63–67 GHz, 100 freq pts, 20 Tx × 20 Rx), **UWB** CIR, **Wi-Fi CSI**, with RGB-D and 125 Hz MoCap
ground truth. Renderer = 3DPS range splat (`nbs/mm3DPS`, SFCW mode added) + the NBS forward models
(`nbs/method/sfcw_forward.py`, `wifi_forward.py`). `4DPS/` is a snapshot of the 4DPS repo adopted for its
**person + scene preprocessing** only: none of SFCW/UWB/WiFi record Doppler, so 4DPS's Doppler path,
velocity field and micro-Doppler loss are NOT used (see `docs/prompts/port_4dps_prompt.md` §B1).

## Where things run
- **Node `mmsd-h100`** (NVIDIA Brev, 8×H100, compute cap 9.0): repo at `/data/mmsd/NBS/code` (git,
  `receive.denyCurrentBranch=updateInstead`), data `/data/mmsd/datasets/OctoNet/OctoNet-upload` (921 GB,
  974 Vayyar / 976 MoCap clips), products `/data/mmsd/NBS/data`, scratch `/ephemeral/scratch`.
  **Always `source tools/env_node.sh` first** (PYTHONPATH override, gcc-12, ARCH 9.0, data roots).
  Python = `/data/mmsd/envs/nbs/bin/python` (venv overlay on `mm4dps`: torch 2.9.1 cu128, numpy 1.26).
  The node cannot reach GitHub: push from the workstation with `tools/push_node.sh` (it first discards the
  figures the node regenerated under `viz/`, which otherwise block the updateInstead push); pull node commits
  back with `git pull node main`. The subject walks IN PLACE, so motion-based foregrounds are empty. If `ssh mmsd-h100` says publickey denied: `brev login && brev refresh`.
- **Workstation**: `/home/adnan/Desktop/mmSD/NBS` (this checkout), GitHub `origin`
  (`adnan-armouti/NBS`). Figures are produced on the node (`python -m viz.<group>` from `nbs/`, scripts in `nbs/viz/`) and rsync'd back into `viz/` here.
- Sensors per OctoNet node (from the data): node_1 Vayyar+UWB+RGB-D+WiFi+IRA+acoustic+thermal+polar;
  node_2 WiFi; node_3 RGB-D+WiFi+thermal; node_4 WiFi+ToF+acoustic; node_5 RGB-D. (All have IRA + TI mmWave.)

## Hard rules
- **Never write under `/home/adnan/Desktop/4DPS`** (the 4DPS development checkout). `4DPS/` here is the
  copy to edit. Never touch `/data/mmsd/4DPS*` or `/data/mmsd/datasets` on the node.
- Never `pkill -f <pattern>` from a Bash call whose own command line contains the pattern.
- No multibounce / image-method rendering proposals.
- Keep the SFCW math single-sourced in `nbs/method/sfcw_forward.py`; the 4DPS copy imports it.

## Measured truths (trust these over the paper / repo card)
- Vayyar pickle = dict `{timestamps, data}`, `data (400,100,T)` complex128, **3.1 Hz** frame rate; range
  compression = IFFT over the 100-freq axis (bin 3.75 cm, R_max 3.71 m). Raw profile is dominated by
  Tx→Rx coupling at bin 0: always subtract the per-pair temporal mean and guard the near/wrap bins.
- **The moving person is NOT observable in the Vayyar SFCW** (99.8% static, dynamic −24 dB,
  phase-decorrelated at 3.1 Hz; `docs/WRITE_UP.md` #19). SFCW targets static scene + material only; never
  imply a human return. UWB/WiFi may see the person — test detectability before building a person term.
- MoCap `mocap_pose/*_skeleton.npy` = dict `{timestamps, positions}`, `(F,20,3)` **mm**, Z-up,
  floor-referenced, 125 Hz, timestamps ISO strings. The npy equals the OptiTrack CSV skeleton (verified 0.0 mm):
  the subject simply stands at the MoCap origin (earlier notes said "XY recentered"; same effect: pose + height).
  Joint order = CSV bone order: Hip, Chest, Neck, Head, L/R Shoulder-UArm-FArm-Hand, L/R Thigh-Shin-Foot-Toe.
  Pickled by numpy 2 → `octonet_io.numpy2_compat()` under numpy 1.x.
- Viz results and the open items they expose: `viz/README.md`. Assumed-pose scaffold on the node:
  `/data/mmsd/NBS/data/OctoNet/scaffold_S1_assumed/pcl.npy`.
- `cut_manual.csv` at the data root is the master join table (976 rows; 974 have Vayyar + MoCap).
- **Node positions ARE in the data** (correcting the older notes): the OptiTrack CSV exports
  (`mocap_csv_final/mocap_S1/**/*.csv`) carry rigid bodies `Node1`..`Node5`, constant across Scene-1 takes;
  see `nbs/preprocessing/octonet_scene.py` (`NODE_POS_S1`, `load_rigid_bodies`). Scenes 2/3 CSVs have no
  node bodies. WiFi Tx/Rx and the obstacle are NOT tracked: schematic from paper Fig. 2(a) only (±0.5 m).
  Camera orientations are calibrated by `nbs/preprocessing/octonet_calibrate.py` (floor plane → pitch/roll,
  MoCap-synchronised person → yaw; yaw spread < 1°; results in `viz/README.md`). `octonet_scene.camera_pose`
  returns the calibrated pose when `$NBS_DATA_OUT/calib_S1/extrinsics.json` exists (copy in `nbs/assets/`),
  else the look-at-origin assumption; figure titles must print `pose_source()`. Each camera has a **depth scale**
  (measured = scale × true: 1.027 / 1.075 / 1.167 for nodes 1/3/5, from the person range with the body-surface
  correction, confirmed by shared walls): always decode depth with `octonet_scene.load_depth_m_calibrated(png, node)`.
  Horizontal camera-to-marker offsets are unobservable (0). The subject walks IN PLACE: motion foregrounds are empty.
- SFCW objective = `nbs/method/sfcw_target.py` (clutter-guarded temporal-mean channel, pooled over recordings,
  per-bin cross-recording reliability; residual is drift with n_eff ≈ 4 frames). Details in `viz/README.md`.
- RGB-D: depth PNGs are **8-bit, linear 0–10 m** (3.9 cm/level; OctoNet `png_to_depth(0,10)`), 640×480,
  ~30 Hz, nodes 1/3/5. The RGB .mp4 is NOT shipped (only cut-window json/txt). No intrinsics shipped
  (nominal D455 87°×58°). Per-node clip stamps differ by ~1 s: resolve paths via `cut_manual.csv`
  (`octonet_scene.modality_path`).
- UWB (Novelda X4M200, node_1): pickle dict `{frame: 1535 floats, timestamp}`, 2.5 GHz BW, 9.9 m max.
  WiFi: per receiver k (stored under `node_k/wifi`) dict `{timestamp: list(F), data: (F,2,114) complex}`,
  ch 36 / 5.18 GHz / 40 MHz, Rx-4 NLOS behind a 3 cm board.
- **Vayyar array layout is unverified**: the direct-coupling structure shows the 20 Tx in two sub-groups, not the
  uniform-pitch line assumed in `vayyar_radar_config`. No RA/azimuth-based loss until the real IMAGEVK-74 element
  coordinates are in; fit on pair-wise range profiles (`sfcw_target`, `kind='logmag'`). Radar/UWB mount yaw on
  node_1 = camera yaw by assumption (sweep was flat, `octonet_mount_yaw.py`).
- OctoNet's own loader code: GitHub `aiot-lab/OctoNet` (`dataset_loader.py`).

## Key docs
`docs/node_migration_plan.md` (how the node is set up), `docs/prompts/viz_octonet_prompt.md`,
`docs/prompts/port_4dps_prompt.md` (the adaptation plan), `docs/WRITE_UP.md` (obstacle log).
