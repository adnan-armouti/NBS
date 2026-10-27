# Prompt — Port 4DPS into NBS and adapt it to OctoNet (SFCW / UWB / WiFi)

> **Status 2026-10-27 (node migration):** the BLOCKER below is obsolete — OctoNet is intact on `mmsd-h100` at
> `/data/mmsd/datasets/OctoNet/OctoNet-upload` and this repo runs there (`tools/env_node.sh`, `python -m viz.<group>`
> from `nbs/`). Node positions come from the OptiTrack CSV rigid bodies (`nbs/preprocessing/octonet_scene.py`), not only
> the paper; depth PNGs are 8-bit 0–10 m; RGB video is not shipped. Paths: `reference_code/NBS/...` → `nbs/...`,
> `reference_code/4DPS/` → `4DPS/`, `viz/` → this repo's `viz/`. See `CLAUDE.md` and `docs/node_migration_plan.md`.
> **Task A and Task B were executed 2026-10-27**: see `4DPS/docs/OCTONET.md` for what was built and what is open.

> Self-contained task prompt for a fresh Claude Code session. Project: `/home/adnan/Desktop/NBS`.
> Read `CLAUDE.md` and `WRITE_UP.md` (obstacles #15–#19b) first. This prompt contains BOTH the copy
> task and the adaptation plan; get the copy + Part B (plan) reviewed by the user before writing
> adaptation code.

## 0. Hard guardrail (do not violate)

**NEVER edit, create, move, or delete anything under `/home/adnan/Desktop/4DPS`.** That is the
protected development checkout. You only ever READ from it and COPY OUT of it. All adaptation happens
in the destination copy inside NBS. If any tool call would write under `/home/adnan/Desktop/4DPS`,
stop.

## 1. Context you need

- 4DPS (`/home/adnan/Desktop/4DPS`, package `mm4DPS/`, Python 3.12 / PyTorch 2.7 / CUDA, conda env
  `hupr`) is a differentiable **77 GHz FMCW** radar renderer of a **moving person**. It is built on
  **3DPS** (the Gaussian-splat range renderer + Jones BSDF + factorized rasterizer + CUDA kernels,
  vendored at `mm4DPS/splat/`) and adds the "4th D" = **Doppler/velocity**: a per-joint velocity
  field skinned to the person, trained so the rendered **micro-Doppler** matches the measurement.
- NBS already contains: the 3DPS renderer with an **SFCW mode I added** (`reference_code/NBS/mm3DPS/`
  — `modulation_scheme: FMCW|SFCW` × `geometry_mode: monostatic|bistatic`), the SFCW forward core
  `reference_code/NBS/method/sfcw_forward.py` (`render_sfcw_mono/pairs`, `sfcw_to_range`), a WiFi
  bistatic renderer `method/wifi_forward.py`, and OctoNet loaders/config/sync
  (`preprocessing/octonet_{io,config,sync}.py`). The SFCW self-test `method/sfcw_migration_selftest.py`
  is 21/21. Do NOT throw this away — the port should REUSE it (see Part B).

### ⚠️ Two realities that shape the whole plan

1. **None of the three target modalities records Doppler.** SFCW Vayyar is a 3.1 Hz stepped-frequency
   sweep (phase decorrelates between frames; no slow-time Doppler); UWB is per-frame CIR; WiFi CSI is
   per-frame subcarrier channel. OctoNet's **TI mmWave** modality *does* carry velocity, but it is a
   **pre-processed point cloud** (`x,y,z,velocity`), not a render target here, and not in scope.
   → **Range splatting (the 3DPS front-end inside 4DPS) is the correct renderer for all three; 4DPS's
   Doppler-plane path + velocity field are NOT used for them.** So "move to 4DPS" is justified by its
   **preprocessing**, not its Doppler renderer. (4DPS ⊃ 3DPS, so adopting 4DPS still gives the range
   splat, keeps the Doppler option open for a future TI-mmWave head, and — the point — brings the
   human-in-scene preprocessing.)
2. **For SFCW specifically, the moving person is NOT observable** (OctoNet Vayyar is 99.8% static,
   dynamic component −24 dB, phase-decorrelated at 3.1 Hz — see `WRITE_UP.md` #19/#19b and memory
   `nbs-vayyar-static-dominated`). So fitting a *person* into the SFCW render will not match GT; SFCW
   should target **static scene + material** inverse rendering. The person preprocessing is still
   valuable for (a) masking the human out of the SFCW scaffold, and (b) UWB/WiFi, which may see the
   person. State this explicitly; do not build a person→SFCW Doppler objective.

### ⚠️ Data/disk prerequisite (currently blocking any run)

The extracted OctoNet dataset (921 GB) was **deleted** and `/data` is **100% full** (your own 4DPS
experiments in `/data/experiments/4DPS_*` ≈ 1.4 TB + RT-Pose 488 GB). Re-acquiring OctoNet needs
~768 GB download + ~1.5 TB extraction headroom; root `/` ≈ 63 GB, `/data` ≈ 22 GB free. **Code-level
porting/adaptation can proceed without the data**, but any end-to-end run (and `check-environment`,
GVHMR, Depth-Anything) needs the user to free ~1 TB or restore from backup first. Do NOT delete the
user's 4DPS experiment data to make room.

## 2. Task A — copy 4DPS into NBS (read-only w.r.t. source)

Destination: `reference_code/4DPS/` (sibling of `reference_code/NBS/`, `reference_code/3DPS_submission/`).
- First **measure** sizes: `du -sh /home/adnan/Desktop/4DPS/*`. Root `/` has only ~63 GB free, so do
  NOT blindly `cp -r` the whole tree — exclude heavy non-code dirs and point env vars at the originals.
- Copy the **code + small assets**, exclude bulky/regenerable/data dirs. Suggested:
  ```
  rsync -a --info=progress2 \
    --exclude 'outputs/' --exclude '.git/' --exclude '**/__pycache__/' \
    --exclude 'models/hf/' \            # Depth-Anything download — point $HF_HOME at the original
    --exclude 'third_party/RT-POSE/' --exclude 'third_party/hupr_official/' \  # keep if small; else env-point
    /home/adnan/Desktop/4DPS/  /home/adnan/Desktop/NBS/reference_code/4DPS/
  ```
  Keep: `mm4DPS/` (incl. `cuda/`, `splat/`), `configs/`, `seeds/`, `models/smpl/` (~40 MB), `Makefile`,
  `README.md`, `docs/`. Verify the two CUDA source trees copied (`mm4DPS/cuda/`, `mm4DPS/splat/cuda/`).
- Record provenance: write `reference_code/4DPS/PORTED_FROM.md` noting the source path, the copy date,
  the rsync excludes, and "original is read-only; all edits here". Do NOT copy `.git` (keeps it from
  looking like the upstream repo).
- Confirm the original is untouched: `git -C /home/adnan/Desktop/4DPS status` (if it's a repo) should
  be clean; nothing under it should have a new mtime.

## 3. Task B — plan to get 4DPS working on OctoNet (get this reviewed before coding)

4DPS has **no OO base classes** to subclass. A dataset is a frozen `mm4DPS.Dataset` dataclass +
`mm4DPS.Radar` dataclass registered in `_DATASETS` (`mm4DPS/__init__.py`), plus `if dataset.name == …`
branches in the entry functions. A "rendering head" is a free function in `forward_render.py` + a
matching measured-side transform in `data.py`. So "override HuPR/RT-Pose → OctoNet" = **add an
OctoNet `Dataset`/`Radar`, an OctoNet branch in the preprocessing entry points, and OctoNet decode in
`data.py`** — the HuPR/RT-Pose paths stay as-is and are simply not selected.

### B1. Doppler decision (3DPS vs 4DPS) — resolved

Use the **range-splat (3DPS) front-end** for SFCW/UWB/WiFi; **do not** engage the Doppler-plane path
(`signal_model.doppler_plane_*`, `forward_render.assemble_surface_doppler_planes_units_cuda`,
`velocity_field`, the micro-Doppler loss term). Adopt the 4DPS *codebase* for its preprocessing. If a
TI-mmWave-point-cloud head is added later (it has velocity), revisit Doppler then.

### B2. Dataset + preprocessing (the main reason to move to 4DPS)

Add to the 4DPS copy (`reference_code/4DPS/mm4DPS/`):
- **`Radar` instance(s)** for the OctoNet sensors. Critically, `Radar` currently assumes **FMCW**
  (`slope_hz_s`, `samples`, `range_bin_m = c·fs/(2·slope·samples)`, carrier 77 GHz). Extend it (new
  fields / a `modulation` discriminator) for **SFCW Vayyar**: `f_start=63e9, f_stop=67e9, n_freq=100,
  df≈40.4 MHz, BW=4 GHz, range_res=c/(2·BW)=3.75 cm, R_max≈3.71 m, 20 Tx × 20 Rx L-array (400 pairs),
  65 GHz centre`. **Reuse the already-correct values and antenna layout from
  `reference_code/NBS/preprocessing/vayyar_radar_config.py` and the constants in `octonet_config.py`.**
  For UWB and WiFi, add minimal `Radar`-like descriptors (center freq / bandwidth / antenna geometry);
  they are not FMCW either.
- **`Dataset(name="octonet", …)`** registered in `_DATASETS`. Set `scene_source="depth"` (OctoNet has
  RGB-D via `depthCamera`: PNG depth + MP4 RGB), `root_env="OCTONET_DATASET_ROOT"`, `frame_count`,
  `camera_hfov_deg` (from the RealSense D455 intrinsics — read from data, not assumed),
  `image_keypoints`, `azimuth_registration`, `depth_scale_mode`.
- **Scene scaffold**: OctoNet ships **metric RGB-D** (3× RealSense D455), so prefer the depth path but
  **do not reuse `build_scaffold` verbatim** — it is hard-coded `if dataset != "hupr": raise
  NotImplementedError` and runs *monocular* Depth-Anything. OctoNet has *real* metric depth (no
  mono-depth needed) from **3 cameras** → add an `run_octonet_scaffold(args)` beside `build_scaffold`
  / `run_lidar_scaffold` in `reconstruction.py` that: backprojects each camera's metric depth, fuses
  the 3 clouds in the MoCap world frame, cuts the person column, estimates PCA normals, voxel-downs,
  and emits the SAME contract (`pcl.npy` `(N,6)`, `<sensor>_static_ra.npy`, `scaffold_metadata.json`)
  so downstream stages stay dataset-agnostic. The per-camera + radar extrinsics come from
  `octonet_calibrate.py` (NOT YET BUILT — RGB-D↔room-geometry calibration anchored on the fixed
  subject spot; MoCap XY is recentered so it can't be triangulated from subject translation — see #18).
- **Person path**: OctoNet has an **RGB camera + 20-joint MoCap**. Two options — (a) reuse 4DPS's
  **GVHMR→SMPL per-frame** path (needs the RGB video + `$GVHMR_ROOT` + `models/smpl`) by adding an
  `octonet` branch in `reconstruction.run_person_sequence` (placement/axes/depth-scale, like the
  `rtpose`/`hupr` blocks), OR (b) **use the MoCap skeleton directly** (`octonet_io.load_mocap`, 20
  joints, mm→m, Z-up, floor-referenced) and fit SMPL to it, which is more accurate than monocular HMR
  since OctoNet gives true 3-D joints. Recommend (b) for the person geometry, keeping GVHMR as a
  fallback. Remember the person is only useful for UWB/WiFi + masking; **not** for SFCW matching.
- **data.py decode**: add OctoNet sequence access modeled on `RTPoseSequence`/`MeasuredReader` that
  yields the measured target for each modality from our existing loaders (`octonet_io.load_vayyar`
  etc.), in the canonical complex layout 4DPS expects; wire `Dataset`/`Radar.decoder`.

### B3. Rendering heads, per modality

Add each as: signal synthesis in `signal_model.py` (if the forward model differs from FMCW) + a
forward view fn in `forward_render.py` + the matching measured transform in `data.py` + a loss term
beside `velocity_field.two_term_loss`. **Reuse the NBS SFCW/WiFi forward code rather than rewriting.**

- **SFCW radar (Vayyar)** — the forward model already exists and is validated in NBS
  (`method/sfcw_forward.render_sfcw_pairs` = Hofmann Eq.10 discrete `f_c`, monostatic for the compact
  board; `sfcw_to_range` = IFFT over the 100 freq steps; `ra_utils.vayyar_to_ra_complex`). Port these
  into the 4DPS copy as the SFCW head (range view = IFFT-over-frequency, NOT the FMCW range-FFT; RA =
  azimuth FFT over the 20-Tx co-array, elevation marginalized). Objective: **static-scene |RA| /
  phasor L1 (L1-IF per Hofmann)**, person masked out. No Doppler term.
- **WiFi CSI** — forward model exists: `method/wifi_forward.render_csi` (bistatic stepped-subcarrier
  `exp(-j2πf·(Rt+Rr)/c)`) + `csi_to_cir`. Port as the WiFi head. Measured target: `(4 nodes, F, 2,
  114)` complex CSI. WiFi nodes are genuinely non-co-located → use `geometry_mode="bistatic"` and
  (later) real LOS/occlusion (deferred in NBS). The person MAY be visible here (75 Hz) → a person term
  is plausible, unlike SFCW.
- **UWB CIR** — no NBS forward model yet. Add one: UWB is wideband impulse → the render is a bistatic
  path-sum convolved with the UWB pulse → CIR (`Σ A_m · p(t − (R_tx+R_rx)/c)`), i.e. a range-splat
  with the UWB pulse PSF instead of the Hann range PSF. Reuse the bistatic path machinery; define the
  UWB antenna geometry per node. Measured target: the CIR frames (`{timestamp, frame}` entries;
  confirm bin count/shape from data).

For all three: the 3DPS splat produces per-point complex contributions; each head differs only in the
**range kernel** (SFCW IFFT-of-stepped-freq, UWB pulse, WiFi subcarrier phase) and the **array
combination** — exactly the separable-head structure 4DPS already uses for RA/RD/RAE.

### B4. Data format per modality (confirmed from OctoNet's loader + our inspection)

| Modality | Stored format | Shape / dtype | Doppler? | Render view |
|---|---|---|---|---|
| **SFCW (Vayyar)** | `vayyar_pickle/*.pickle` dict `{timestamps, data}` | `data (400,100,T)` complex128; ts = datetime | **No** (3.1 Hz sweep) | IFFT-over-freq range → 2D RA |
| **WiFi CSI** | `node_*/wifi/exp-*/*.pickle` list `{timestamp, data}` | `(4 nodes, F, 2, 114)` complex | No | stepped-subcarrier → \|CSI\| + CIR |
| **UWB** | `node_*/uwb/data/*.pickle` list `{timestamp, frame}` | per-frame CIR (confirm bins) | No | pulse range-splat → CIR |
| TI mmWave *(not in scope)* | `node_*/mmWave/data/*.pickle` | point cloud `x,y,z,velocity` | **Yes** (already processed) | — |
| MoCap | `mocap_pose/*_skeleton.npy` dict `{timestamps, positions}` | `(F,20,3)` mm, Z-up, **XY recentered**, 125 Hz; numpy2 shim | — | person geometry |
| RGB-D | `node_*/depthCamera/...` | PNG depth + MP4 RGB | — | scene scaffold |

**Verdict on splatting:** range splatting is valid for SFCW/UWB/WiFi (just different range kernels);
**Doppler splatting is not applicable** to any of them (no slow-time Doppler is recorded). So the
3DPS range path inside 4DPS is sufficient; 4DPS is adopted for its person+scene preprocessing.

### B5. Reconcile with existing NBS work (don't duplicate)

- The SFCW physics already lives in `reference_code/NBS/method/sfcw_forward.py` + the `mm3DPS` SFCW
  mode. Decide: (i) have the 4DPS copy **import** these from NBS, or (ii) port them into
  `reference_code/4DPS/mm4DPS/signal_model.py`+`forward_render.py`. Prefer a thin reuse to keep one
  source of truth for the SFCW math.
- 4DPS vendors its own 3DPS at `mm4DPS/splat/` (FMCW). The SFCW branch logic I added to NBS's
  `mm3DPS/rasterizer*.py`, `psf.py`, `data/io_utils.py`, `data/ra_utils.py` must be mirrored into the
  4DPS copy's `rasterizer.py` if you render SFCW through 4DPS's rasterizer rather than through
  `sfcw_forward`. Simplest first milestone: render SFCW via the NBS `sfcw_forward` path as a 4DPS
  "head", bypassing 4DPS's FMCW rasterizer entirely.

## 4. Files to read first

- `/home/adnan/Desktop/4DPS/mm4DPS/__init__.py` — `Radar` + `Dataset` dataclasses, `_DATASETS`,
  `_hupr_radar`, `build_radar_config`.
- `/home/adnan/Desktop/4DPS/mm4DPS/chain.py` — the 5-stage orchestration (`run_chain`, `dispatch`).
- `/home/adnan/Desktop/4DPS/mm4DPS/reconstruction.py` — `build_scaffold` (depth, HuPR-only),
  `run_lidar_scaffold` (RT-Pose), `run_person_sequence` (GVHMR+SMPL per-frame).
- `/home/adnan/Desktop/4DPS/mm4DPS/data.py` — decode + shapes + `SmplBody` + `MeasuredReader`.
- `/home/adnan/Desktop/4DPS/mm4DPS/rasterizer.py` — 3DPS range splat + BSDF + `HannPSFTable`.
- `/home/adnan/Desktop/4DPS/mm4DPS/signal_model.py` + `forward_render.py` — slow-time/Doppler + view
  heads (read to see the head pattern; the Doppler parts are the ones you will NOT use).
- `/home/adnan/Desktop/4DPS/mm4DPS/splat/PROVENANCE.md` — the 3DPS/4DPS boundary.
- NBS side: `reference_code/NBS/method/sfcw_forward.py`, `method/wifi_forward.py`,
  `preprocessing/octonet_{config,io,sync}.py`, `preprocessing/vayyar_radar_config.py`,
  `mm3DPS/data/ra_utils.py` (SFCW helpers), `WRITE_UP.md` #15–#19b.

## 5. Environment / build

- 4DPS env is conda `hupr` (Python 3.12, PyTorch 2.7 + CUDA; NBS uses `mmir`/numpy 1.26 — they differ,
  so the ported 4DPS likely runs under `hupr`, our OctoNet loaders under `mmir`; keep that in mind, or
  make `octonet_io` import-clean under `hupr`). Two CUDA extensions (`hupr_cuda`, `mm3DPS_cuda`) must
  be rebuilt in the copy: `make build-cuda` / `python -m mm4DPS.cuda_ext build`, `ARCH=8.9` (RTX 4090).
  `.build_stamp.json` (SHA of sources) detects staleness; `python -m mm4DPS.qa check-environment`.
- External: Depth-Anything (HF, `$HF_HOME`), SMPL (`models/smpl/SMPL_NEUTRAL.npz`), GVHMR
  (`$GVHMR_ROOT` + its own env) — only needed for the person path / monocular scaffold; the OctoNet
  scaffold uses real RGB-D and may not need Depth-Anything at all.

## 6. Risks / open questions

- `octonet_calibrate.py` doesn't exist yet → no sensor extrinsics → scene fusion + any Rendered-vs-GT
  is blocked until built (RGB-D↔room geometry, anchored on the subject spot).
- SFCW-can't-see-the-person (#19): a 4DPS-style person→radar objective is wrong for SFCW; scope SFCW to
  static scene/material. Confirm whether UWB/WiFi actually resolve the person before building person
  objectives for them (repeat the controlled detectability test from #19 on UWB/WiFi).
- Env split (`hupr` vs `mmir`), numpy 1.x vs 2.x pickle shim for MoCap.
- Disk/data gone (§1) — nothing runs end-to-end until ~1 TB is freed or OctoNet is restored.
- Keep the SFCW math single-sourced (B5) — don't fork `sfcw_forward` into 4DPS and let them drift.

## 7. Test plan

- Copy: `du -sh reference_code/4DPS`; `python -c "import mm4DPS"` (from the copy, env `hupr`); original
  `/home/adnan/Desktop/4DPS` mtimes unchanged.
- Dataset wiring: `mm4DPS.dataset_from_argv(["--dataset","octonet"])` returns the OctoNet `Dataset`;
  `check-environment` passes for the OctoNet radar config.
- SFCW head (data-independent): render a synthetic point target through the ported SFCW head → range
  peak at the right bin (mirror `method/sfcw_migration_selftest.py`).
- End-to-end (needs data restored): `python -m mm4DPS.chain run octonet <seq>` produces a static-scene
  SFCW |RA| that matches the measured clutter-removed |RA|; WiFi |CSI|/CIR rendered vs GT.

## 8. Rollback

The copy is additive and self-contained: `rm -rf reference_code/4DPS` undoes the port. No NBS files
are modified by Task A; Part B edits live only inside `reference_code/4DPS/`. The original 4DPS
checkout is never touched.
