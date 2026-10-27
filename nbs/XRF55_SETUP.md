# Running 3DPS on XRF55 (mmWave RA) — setup status

Goal: train the 3DPS Gaussian renderer on XRF55's single-chip mmWave RA. XRF55 differs from stock
ColoRadar (which 3DPS targets): single-chip IWR6843ISK (3Tx/4Rx → 64-az RA), processed magnitude
heatmaps `(1,17,256,128)`=`[doppler 0:64 | azimuth 64:128]` (no ADC), and a **fixed** radar (only the
human moves). Flip `DATASET = 'xrf55'` at the top of `mm3DPS/train_frame_nvs.py` to use the XRF55 path.

## DONE + verified

1. **Single-chip config generator** — `preprocessing/xrf55_radar_config.py`; configs written to
   `data/XRF55/Scene{N}/configs/single_chip.json`. Verified: ΔR from cfg = **0.0675 m** (data-driven),
   RX 0.5λ, **8-vx azimuth ULA uniform 0.5λ**, **middle TX elevated (+Z, toward ceiling)**, boresight +Y,
   loads via `_pose_utils.load_pose_from_config`. (Reuses mmIR `antenna_layout_single_chip`; carrier = 62 GHz
   band centre, adjustable.)
2. **Trainer GT/loss edits** (`mm3DPS/train_frame_nvs.py`, marked `# XRF55:`):
   - GT loading uses the **RA half only** `[...,64:128]` (Doppler half dropped); no ADC.
   - **Magnitude-RA loss** (`_compute_ra_loss`/`_mag_ra_loss`, default Pearson).
   - **Single-chip RA assembly** (`range_profile_to_ra_single_chip`, 64-az) at every loss/metric/export
     site; polar→cart grid set to 64 az.
3. **Fixed-pose dataset builder** — `build_xrf55_dataset(scene)`: one sample, pose from `single_chip.json`,
   GT = **static RA** (median across clips of the RA-half → suppresses the moving human). Verified on
   Scene1: pose tx(3,3)/rx(4,3) with middle-TX elevated, GT (64 az, 256 range). The trainer's dataset
   call and `seed_cfg` are branched to XRF55.
4. Trainer compiles; `DATASET='coloradar'` is behaviour-equivalent to stock.

5. **Model init** — `init_xrf55_model(scene)` (`mm3DPS/train_frame_nvs.py`): builds the `Rasterizer`
   from `single_chip.json` with **factory antenna patterns**, points `mesh_file` at `<scene>/scene/mesh.ply`
   only to resolve the scaffold `pcl.npy`, runs `init_visible_weighted(enable_occlusion=False)` (no mesh
   needed), and sets the **BSDF carrier to 62 GHz**. Verified on Scene1: 124,645-pt scaffold → 20k model.
6. **CUDA extension built** (`cd mm3DPS/cuda && python setup.py build_ext --inplace`, sm_89).

## CONFIRMED WORKING (render smoke test, Scene1)

Set `DATASET = 'xrf55'` and run the trainer (the XRF55 path ignores `--train_frames/--test_frame`).
End-to-end render → single-chip RA → magnitude-RA loss → backprop verified:
- rendered rp `(3 TX, 4 RX, 256)` → single-chip RA `(64 az, 256 range)` = GT shape;
- **loss 0.97 → 0.81, RA-vs-GT corr 0.03 → 0.19 over 25 steps** (materials+rotations only, 20k pts) —
  the fit improves, so the gradients are correct. (Caught + fixed a real `(TX,RX)→(RX,TX)` permute bug
  in `range_profile_to_ra_single_chip` along the way.)

Scale up with more iters / points / `learn_positions_lr` / densification for real numbers.

## Refinements (not blockers)
- **Verify the single-chip RA azimuth orientation** (fftshift/sign of `range_profile_to_ra_single_chip`)
  against a real clip's RA — the corr climbs, so it's at least roughly aligned, but confirm before
  trusting absolute numbers.
- Carrier is the 62 GHz band centre; tune if the true chirp config is obtained.

The confirmed `mm3DPS/train_frame_nvs.py` + `preprocessing/xrf55_radar_config.py` are copied into `misc/`.
