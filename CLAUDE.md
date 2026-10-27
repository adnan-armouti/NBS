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
  The node cannot reach GitHub: push from the workstation with `git push node main`; pull node commits
  back with `git pull node main`. If `ssh mmsd-h100` says publickey denied: `brev login && brev refresh`.
- **Workstation**: `/home/adnan/Desktop/mmSD/NBS` (this checkout), GitHub `origin`
  (`adnan-armouti/NBS`). Figures are produced on the node and rsync'd back into `viz/` here.
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
  floor-referenced, 125 Hz, timestamps ISO strings; **XY is recentered on the subject** (pose + height only,
  not room position). Pickled by numpy 2 → `octonet_io.numpy2_compat()` under numpy 1.x.
- `cut_manual.csv` at the data root is the master join table (976 rows; 974 have Vayyar + MoCap).
- Sensor extrinsics are NOT in the data: node positions come from the paper in `pdf/`;
  `octonet_calibrate.py` (RGB-D ↔ room geometry) does not exist yet.

## Key docs
`docs/node_migration_plan.md` (how the node is set up), `docs/prompts/viz_octonet_prompt.md`,
`docs/prompts/port_4dps_prompt.md` (the adaptation plan), `docs/WRITE_UP.md` (obstacle log).
