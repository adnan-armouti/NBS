# Prompt — OctoNet visualizations (scene geometry, nodes, rendered vs GT)

> **Status 2026-10-27 (node migration):** the BLOCKER below is obsolete — OctoNet is intact on `mmsd-h100` at
> `/data/mmsd/datasets/OctoNet/OctoNet-upload` and this repo runs there (`tools/env_node.sh`, `python -m viz.<group>`
> from `nbs/`). Node positions come from the OptiTrack CSV rigid bodies (`nbs/preprocessing/octonet_scene.py`), not only
> the paper; depth PNGs are 8-bit 0–10 m; RGB video is not shipped. Paths: `reference_code/NBS/...` → `nbs/...`,
> `reference_code/4DPS/` → `4DPS/`, `viz/` → this repo's `viz/`. See `CLAUDE.md` and `docs/node_migration_plan.md`.

> Self-contained task prompt to paste into a fresh Claude Code session. Do not rely on prior
> conversation context. Project: `/home/adnan/Desktop/NBS` — RF inverse-rendering (3DPS→SFCW/OctoNet
> migration). Read `CLAUDE.md` and `WRITE_UP.md` first (env, paths, obstacle log).

## ⚠️ BLOCKER — read before starting

The extracted OctoNet dataset (**921 GB** at `/data/datasets/OctoNet/OctoNet-upload/`) was **deleted**
after the download (the `/data` SSD filled to 100% with unrelated 4DPS experiment outputs). It must
be **re-acquired before any data-dependent viz can run**, and there is currently **no free disk space**
(root `/` ≈ 63 GB, `/data` ≈ 22 GB free). First step is therefore a disk/data decision with the user:
free ~1 TB on `/data` (do NOT delete `/data/experiments/4DPS_*` or `/data/datasets/*` without explicit
permission), or restore OctoNet from a backup, then re-run the downloader (see
`scratchpad/octonet_download_par.sh` pattern in the obstacle log, or the HF `download_octonet.sh`).
Our NBS loader/renderer code is intact (`reference_code/NBS/preprocessing/octonet_*.py`,
`method/sfcw_forward.py`); only the raw data is gone.

The **node-position viz (#1) is partly data-independent** (positions come from the paper, not the
data) and can be produced now; #2 and #3 need the restored dataset.

## Goal

Save a set of figures to `/home/adnan/Desktop/NBS/viz/` (create it; it is on root `/`, which has room).
One subfolder per group: `viz/nodes/`, `viz/geometry_init/`, `viz/render_vs_gt/`. Each figure is a
`.png` at ≥150 dpi with a title, axis labels/units (metres), and a short caption `.txt` or embedded
text. Also write `viz/README.md` indexing every figure with one line each.

## What to visualize (minimum)

### 1. OctoNet node layout + per-node sensor availability  → `viz/nodes/`
- **Source of node positions: the paper**, OpenReview `https://openreview.net/pdf?id=z3TftXOizf`
  (and its supplement). The PDF is **not** in `pdfs/` and OpenReview blocks `curl`/WebFetch — fetch it
  via the browser skill (built-in or Chrome) or ask the user to drop it in `pdfs/`. Extract: the room
  dimensions, the 5 node positions + the 3 RGB-D cameras + the Vayyar radar + OptiTrack, and node
  mounting heights/orientations. **These extrinsics are NOT in the data or the repo** — the paper is
  the only source (confirmed). Record the numbers you extract into
  `reference_code/NBS/preprocessing/octonet_config.py` (there are `SensorPose` placeholders + a
  `MOCAP_SUBJECT_SPOT_XY` anchor already).
- **2D top-down room plan**: room rectangle, 5 nodes (labelled node_1..5), 3 RGB-D cams, Vayyar,
  MoCap origin / subject standing spot `MOCAP_SUBJECT_SPOT_XY ≈ (-0.22,-0.21) m`; annotate each
  sensor's rough FoV/boresight.
- **3D view**: same, with mounting heights (Z-up, floor=0), subject skeleton at the spot for scale.
- **Node × sensor matrix** (heatmap/table): rows = node_1..5, cols = {vayyar, mmWave(TI), wifi, uwb,
  IRA, ToF, acoustic, seekThermal, depthCamera, polar}, cell = available?. Derive availability from
  **non-null columns of `cut_manual.csv`** (one representative recording) AND the `node_*/` dir
  listing. Known facts to sanity-check against: **Vayyar lives only on node_1**; ToF only on node_4;
  depthCamera on a subset. Use `octonet_io.load_index()`.

### 2. Our initialized scene geometry (scene + person) + how  → `viz/geometry_init/`
- **Scene**: back-project the RGB-D depth (`depthCamera`: PNG depth + MP4 RGB, loader
  `octonet_io`/OctoNet's `load_depthcamera_data`) of the 3 camera nodes into a fused point cloud in
  the MoCap world frame. Show the fused scaffold top-down + 3D, coloured by source camera. (Needs the
  per-camera extrinsics from #1 / `octonet_calibrate` — see "Dependencies".)
- **Person**: overlay the 20-joint MoCap skeleton (`octonet_io.load_mocap`; mm→m, Z-up, floor-ref;
  note **XY is recentered on the subject — pose + height only, not room position**, see
  `WRITE_UP.md` #18). Draw bones; show one standing + one `sleep` frame (head Z ≈ 0.2 m lying vs
  ≈1.5 m standing) to demonstrate the init captures posture.
- **"How" panel**: a labelled diagram of the init pipeline (RGB-D → depth backproject → fuse → static
  scaffold with human masked; MoCap → person skeleton), mirroring the XRF55 scaffold design
  (`WRITE_UP.md` #3) but re-grounded to RGB-D+MoCap. If `octonet_calibrate.py` does not yet exist,
  produce the person viz (self-contained from MoCap) and the single-camera scene viz, and clearly
  label the fused-scene panel as pending calibration.

### 3. Rendered vs GT, per modality  → `viz/render_vs_gt/`
For each of **SFCW radar, UWB, WiFi CSI**, a side-by-side **GT (measured) vs Rendered (forward model)**:
- **SFCW radar** — GT: load a Vayyar clip (`octonet_io.load_vayyar`, (T,400,100) complex),
  `mm3DPS.data.ra_utils.sfcw_range_compress` (IFFT over freq) → range profile / 2D RA
  (`vayyar_to_ra_complex`). **Apply clutter removal (subtract per-pair temporal mean) + guard the
  near-coupling and IFFT-wrap bins**, else the direct Tx→Rx coupling at ~0 m dominates. **Caveat
  (WRITE_UP #19):** the moving human is NOT recoverable from this radar (−24 dB dynamic,
  phase-decorrelated at 3.1 Hz); show the *static-scene* return and label it as such — do not imply a
  human return. Rendered: `method/sfcw_forward` / the renderer's SFCW mode on the initialized scaffold.
- **UWB** — GT: CIR frames (loader entries `{timestamp, frame}`; confirm shape from data). Rendered:
  requires a UWB CIR head (see the 4DPS port prompt `md/port_4dps_prompt.md`); if absent, show GT only
  and mark Rendered "pending UWB head".
- **WiFi CSI** — GT: `(4 nodes, F, 2, 114)` complex CSI (loader entries `{timestamp, data}`; the WiFi
  forward model `method/wifi_forward.py` already exists as a bistatic SFCW-style renderer). Rendered:
  `wifi_forward.render_csi` on the scaffold with the WiFi node geometry. Show |CSI| per subcarrier and
  the CIR (`csi_to_cir`).

Use a consistent colormap, dB scale with a fixed floor for RF magnitudes, and label range axes in
metres (SFCW bin = `c/(2·BW)` = 3.75 cm; R_max ≈ 3.71 m).

## Files to read first

- `reference_code/NBS/preprocessing/octonet_io.py` — loaders (`load_vayyar`, `load_mocap`,
  `load_index`, `resolve`, `pairs_to_tx_rx`); has the numpy2 shim for MoCap `.npy`.
- `reference_code/NBS/preprocessing/octonet_config.py` — all measured constants + `SensorPose`
  placeholders; data root, Vayyar/MoCap facts.
- `reference_code/NBS/preprocessing/octonet_sync.py` — timestamp alignment (MoCap↔Vayyar).
- `reference_code/NBS/mm3DPS/data/ra_utils.py` — `sfcw_range_compress`, `vayyar_to_ra_complex`.
- `reference_code/NBS/method/sfcw_forward.py`, `method/wifi_forward.py` — forward models.
- `WRITE_UP.md` obstacles #15–#19b (OctoNet migration, measured truths, the static-dominance finding).

## Dependencies / ordering

- #1 node matrix + #2 person-from-MoCap + #3 SFCW-GT and WiFi-GT are runnable once the **data is
  restored** (no new code).
- #1 node *positions*, #2 *fused* scene, and #3 *Rendered* panels depend on **sensor extrinsics** →
  need `octonet_calibrate.py` (not yet built; RGB-D↔room-geometry calibration, NOT subject
  translation — MoCap XY is recentered). Where a panel is blocked on this, still emit the runnable
  half and label the blocked half clearly rather than skipping it.

## Environment

- Conda env `mmir` (`~/.conda/envs/mmir/bin/python`, numpy 1.26). MoCap `.npy` need the numpy2→1
  `_core` shim — call `octonet_io.numpy2_compat()` (already handled inside `load_mocap`).
- Run modules from `reference_code/NBS/` (`python -m preprocessing....`). Matplotlib `Agg` backend.

## Self-check before producing figures

- Did I confirm the data is restored (or clearly scope to the data-independent figures)?
- Did I apply clutter removal + bin guarding to every SFCW figure, and NOT imply a human return?
- Did I label MoCap as pose+height only (XY recentered), and every blocked panel as blocked?
- Did I write `viz/README.md` indexing each figure, and put figures under the three subfolders?
