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
- SFCW objective = `nbs/method/sfcw_target.py` (temporal-mean channel pooled over recordings, two-tone on-board
  coupling removed, per-bin cross-recording reliability; residual is drift with n_eff ≈ 4 frames). **Range bins < 24
  (0.9 m) are the node's own near-field return (cover/housing/table), absent from the scaffold: the static fit masks
  them (`octonet_fit` GUARD_LO=24)**; they are not window leakage (T4). Details in `viz/README.md`, tracker T4.
- **Person detectability (T9, 14 recordings): no band passes** (SFCW z −0.1, UWB 2.3, WiFi 0.0 against permutation
  nulls); no person term is built. UWB shows a weak direct-return effect only on large-motion activities.
- UWB static target: the coherent RF temporal mean is valid (frame coherence 0.985); WiFi: the complex mean is destroyed
  by per-packet carrier phase (coherence 0.01) → magnitude-only target, mask subcarriers [5,33,47,66,80,108]
  (`nbs/method/rf_static_targets.py`). UWB pulse template from the coupling peak: fc 7.0 GHz, −10 dB BW 1.46 GHz,
  FWHM 12.9 cm, on-board delay 4.5 cm (`nbs/method/uwb_template.py`); the paper's "2.5 GHz" is not what the data shows.
- Multi-band material fit (`4DPS/mm4DPS/octonet_fit_multiband.py`, T11/T12): voxel-grid ITU `abcd` field, SFCW+UWB+WiFi
  joint; the held-out-band gate gives only 2–3 % gains (SFCW 2σ, UWB marginal, WiFi fail) → **no band-synthesis claim
  yet**. Known limits: WiFi Tx/Rx schematic (measure them), UWB pose assumed, ITU bounds saturate in the tails
  (tighten d≤1.0, c≤0.1 S/m or use a class prior). Bounded per-point ρ (0.590) is not comparable to the ITU-bounded
  grid (0.72–0.80): the former has ±26 dB of freedom, the latter ~12 dB.
- **Project goal (user, 2026-10-28): train on ONE modality (two jointly only as a lower-preference alternative) and
  evaluate ZERO-SHOT transfer to the other bands.** Single-modality training check: `viz/render_vs_gt_v2/README.md`
  (T19). Geometry videos: `viz/geometry_v2/` (mp4s git-ignored, on workstation + node).
- Progress on all open items: `docs/plans/OPEN_ITEMS_TRACKER.md`.
- RGB-D: depth PNGs are **8-bit, linear 0–10 m** (3.9 cm/level; OctoNet `png_to_depth(0,10)`), 640×480,
  ~30 Hz, nodes 1/3/5. The RGB .mp4 is NOT shipped (only cut-window json/txt). No intrinsics shipped
  (nominal D455 87°×58°). Per-node clip stamps differ by ~1 s: resolve paths via `cut_manual.csv`
  (`octonet_scene.modality_path`).
- UWB (Novelda X4M200, node_1): the pickle is a **stream** of per-frame dicts `{frame: 1535 floats, timestamp}` —
  use `octonet_io.load_uwb` (plain `pickle.load` returns only the first frame): ~1576 frames at 17.06 Hz per recording.
  The frame is RF-sampled (23.328 GS/s, carrier ~7.2 GHz), bin 6.43 mm, 9.9 m; static-dominated (residual −31 dB).
  WiFi: per receiver k (stored under `node_k/wifi`) dict `{timestamp: list(F), data: (F,2,114) complex}`,
  ch 36 / 5.18 GHz / 40 MHz, Rx-4 NLOS behind a 3 cm board.
- **Vayyar array**: datasheet L-array (Tx1..20 on one edge, Rx21..40 on the adjacent edge, corner Tx20/Rx21 = tx 19 /
  rx 0 in the tx-major reshape, D = 0.04 m per edge), consistent with the strongest coupling pairs. But the raw phasors
  carry uncalibrated per-element phases (range-independent +90..105°/element ramp along the first pair axis, random
  along the second), so **no RA/azimuth processing or complex loss until the per-channel phase calibration of
  `docs/plans/2026-10-28_open_items.md` §4 is done**; fit on pair-wise range profiles (`sfcw_target`, `kind='logmag'`).
  Radar/UWB mount yaw on node_1 = camera yaw by assumption (range-only sweep was flat; redo with RA after calibration).
- OctoNet's own loader code: GitHub `aiot-lab/OctoNet` (`dataset_loader.py`).

## OctoNet in the 4DPS copy (Task B, done 2026-10-27)
`--dataset octonet` is registered (`mm4DPS/__init__.py`: SFCW `Radar`, `OCTONET` dataset). Stages through the standard
entries: `python -m mm4DPS.reconstruction scaffold --dataset octonet --rec <rec> --output-dir <dir>` and
`… person-sequence --dataset octonet --sequence <rec> --output-dir <dir>`; sequence access `mm4DPS.octonet.OctoNetSequence`;
heads `mm4DPS/octonet_heads.py` (`selftest`); static fit `python -m mm4DPS.octonet_fit fit-static`. Contracts, results and
open items: `4DPS/docs/OCTONET.md`. The 4DPS `chain.py` is not used for OctoNet.

## Plans
`docs/plans/2026-10-28_open_items.md`: material identifiability + what to take from Hofmann, coupling removal, person
detectability test, Vayyar assumptions V1–V10 (awaiting line-item approval), mount yaw, UWB template.

## Key docs
`docs/node_migration_plan.md` (how the node is set up), `docs/prompts/viz_octonet_prompt.md`,
`docs/prompts/port_4dps_prompt.md` (the adaptation plan), `docs/WRITE_UP.md` (obstacle log).
