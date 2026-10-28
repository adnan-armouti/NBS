# OctoNet in the 4DPS copy (Task B of `docs/prompts/port_4dps_prompt.md`), 2026-10-27

4DPS is adopted for its preprocessing contracts, not its Doppler renderer: none of SFCW / UWB / Wi-Fi record
slow-time Doppler, so the velocity field and the micro-Doppler loss are not used. The HuPR / RT-Pose paths are
untouched; `--dataset octonet` selects the new branches. Run everything on `mmsd-h100` after
`source tools/env_node.sh` (both `nbs/` and `4DPS/` on PYTHONPATH; the OctoNet code stays single-sourced in `nbs/`).

## Registration (`mm4DPS/__init__.py`)
- `Radar` gained `modulation` ("fmcw" | "sfcw"), `f_start_hz`, `f_stop_hz`, `n_freq`; `bandwidth_hz`, `range_bin_m`
  (c/2B = 3.75 cm) and `unambiguous_range_m` (3.71 m) branch on it. FMCW radars are unchanged (test).
- `OCTONET = Dataset(name="octonet", radars=(vayyar SFCW,), scene_source="rgbd", root_env="OCTONET_DATASET_ROOT")`,
  in `_DATASETS` / `DATASETS`; root falls back to the node path. The Tx layout is a placeholder (unverified, two Tx
  sub-groups): no azimuth beamforming anywhere, pair-wise range profiles only.
- UWB and Wi-Fi descriptors: `mm4DPS.octonet.SENSORS`. `qa check-environment` skips non-FMCW radars.
- Tests: `tests/test_octonet_registration.py` (3 pass).

## Stages and contracts
| stage | entry | output |
|---|---|---|
| scaffold | `python -m mm4DPS.reconstruction scaffold --dataset octonet --rec 20240518135416 --output-dir <dir>` → `mm4DPS/octonet.py::run_octonet_scaffold` | `pcl.npy` (N,6, MoCap world), `source_node.npy`, `vayyar_static_rp.npy` (400,100 pooled target), `vayyar_static_target.npz`, `scaffold_metadata.json` |
| person | `python -m mm4DPS.reconstruction person-sequence --dataset octonet --sequence 20240518134358 --output-dir <dir>` → `mm4DPS/octonet_person.py::run_octonet_person` | `person_sequence.npz` (`vertices_world (T,6890,3)`, `normals_world`, `joints_world (T,24,3)`, `t`, `valid`, `betas`, `pose`, `trans`, `faces`, RMSEs), `person_metadata.json` |
| sequence access | `mm4DPS.octonet.OctoNetSequence(rec)`: `vayyar()`, `uwb()`, `wifi(k)`, `mocap()`, `extrinsics()`, `sensor_pose()`, `scaffold()`, `static_target()`, `summary()`; `render_inputs(scaffold_dir, extrinsics)` → the heads' `(cloud, material, sensor)` | — |
| heads | `mm4DPS/octonet_heads.py`: `sfcw_head → rp (400,100)`, `wifi_head → (4,2,114)`, `uwb_head → (cir_rf, cir_env)` on 1535 bins; `python -m mm4DPS.octonet_heads selftest` (10/10) | — |
| static fit | `python -m mm4DPS.octonet_fit fit-static [--scaffold-dir <scaffold stage dir>]` | `$NBS_DATA_OUT/fit_S1_sfcw/{material.npz, fit_log.json}`, `viz/render_vs_gt/sfcw_fit_static.png` |

Scene source: three D455 depth streams (8-bit 0–10 m × per-camera depth scale), temporal median, calibrated poses
(`nbs/preprocessing/octonet_calibrate.py`), person column cut; voxel + PCA normals with 4DPS's own GPU routines
(`reconstruction.voxel_downsample_torch`, `pca_normals_torch`). Person source: SMPL fitted to the 20-joint OptiTrack
skeleton (one shape per recording, per-frame pose, per-joint local offsets because OptiTrack's hip/knee centres sit up
to 9 cm outside SMPL's; slerped to the Vayyar frame times, any time array accepted): 2.34 cm mean joint RMSE walking,
3.41 cm seated, ~65 s per recording on one H100.

## Heads (`mm4DPS/octonet_heads.py`), shared input convention
`cloud = {xyz (N,3), normals (N,3)}`, `material = {abcd (N,4) ITU P.2040, rho (N,) reflectivity scale}`,
`sensor = {pos, boresight}` (SFCW/UWB) or `{tx, rx{k}, los{k}}` (Wi-Fi); torch, GPU, chunked, differentiable in
rho / abcd / xyz. The SFCW head imports the NBS reflectivity / incidence / geometry (`method/sfcw_forward.py`) and
only vectorises the 400-pair loop (equal to the NBS function to 5e-13). Each head differs only in its range kernel:
Hann IFFT over the 100 tones (SFCW), subcarrier phase + LOS term (Wi-Fi), Gaussian pulse of 2.5 GHz bandwidth with
1/(Rt·Rr) (UWB; the measured frame is a real baseband amplitude, so the envelope output is the comparable one).

## First end-to-end result: static SFCW fit, Scene 1
Loss = `nbs/method/sfcw_target.sfcw_static_loss(kind='logmag', weight='xcorr')` on the pooled 5-recording target,
per-point rho ∈ [1/20, 20] (ITU concrete dispersion fixed) + L2, Adam, 30k points: 1.157 → 0.627 (238 steps, 37 s);
the held single recording 1.140 → 0.653 (same gain, so the fit explains the shared static scene). The uniform-concrete
render is 20–30 dB too bright at 0.2–1 m and too dark beyond 2.5 m; the fit flattens the pair-mean profile onto the GT
within ~2 dB over 0.5–3.3 m, individual pairs follow the slow shape but not the pair-specific ripple. An unbounded rho
reaches 0.588 by sending a few points to ~1000×: the per-point field is under-determined by one profile per pair.

## Open items (plans: `docs/plans/2026-10-28_open_items.md`)
1. Vayyar element layout (two Tx sub-groups): obtain the IMAGEVK-74 coordinates before any RA / azimuth term.
2. rho < 1 near the radar may be compensating the coupling's Hann sidelobes leaking into bins 6–25: model the
   coupling in the target or widen the guard.
3. `abcd` (dispersion) is not identifiable from a 4 GHz static profile; a material-class prior (`matmodels`) is the
   way to make it physical. A smoothness prior on rho is the cheap alternative to the bound.
4. Person term for UWB / Wi-Fi: not built. Test detectability first (WRITE_UP #19 repeated on UWB/WiFi); the SMPL
   sequence and `person_sequence.npz` are ready for it.
5. Radar/UWB mount yaw = camera yaw by assumption; UWB band constants from the datasheet (the pulse template can be read from the data's coupling peak). The UWB pickle is a 17 Hz stream (fixed 2026-10-28: `octonet_io.load_uwb`); the heads' `cir_rf` output is the comparable one (RF-sampled frames).
6. `octonet_fit` reads paths under `$NBS_DATA_OUT`; a `Workspace`-style layout can replace that when the chain is
   needed. The 4DPS `chain.py` (HuPR/RT-Pose stage orchestration, GVHMR, Depth-Anything) is NOT used for OctoNet.
