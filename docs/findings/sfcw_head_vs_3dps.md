# Findings: the SFCW head vs 3DPS rendering (living checklist)

Started 2026-10-28 from `docs/plans/2026-10-28_sfcw_head_audit.md` (the full audit with numbers) and the diagnosis
`docs/plans/2026-10-28_sfcw_diagnosis.md`. Rule: every invalid difference gets struck through here when it is reverted,
with the commit and the number that proves it. Nothing is struck without a measurement.

## The lesson
The OctoNet SFCW head was a standalone point-sum forward model (`nbs/method/sfcw_forward.py::render_sfcw_pairs`) that
bypassed the 3DPS rasterizer; the port plan called it the "simplest first milestone" and it was never replaced. The
3DPS rasterizer already had the SFCW mode with the only two changes the method needs (FMCW → SFCW range compression;
material evaluated at the band's frequency). Every other difference was a regression of 3DPS physics.
Measured consequence (uniform concrete, no training, calibrated Scene-1 scaffold): iteration-0 pair-mean profile
correlation −0.33 (head) vs +0.64 (3DPS `render_factorized`), RMSE 6.5 vs 2.45 dB, position/material gradient ratio
2607 vs 6.8.

## Invalid differences (to revert) — strike through when done
- [x] ~~**D3 scattering model**: Fresnel amplitude only instead of the six-parameter Jones BSDF (KA/SPM/directive/broad lobes, slab, polarisation, energy gating).~~ Reverted (T23, `sfcw_head` → `render_factorized`).
- [x] ~~**D4 radar equation**: Γ/(d_tx·d_rx) (power 1/R⁴, no area, no cos_o) instead of √(f·cos)·√G_tx/d_tx·√(G_rx·dΩ) with the receiver hemisphere weight dΩ.~~ Reverted (T23; areas = 1 per active point).
- [x] ~~**D5 antenna element gain**: none, instead of E/H-plane patterns per Tx and Rx.~~ Reverted (T23) with the datasheet stand-in `nbs/assets/antenna_pattern/VAYYAR/{tx,rx}_65.npy` (E 180°, H 82°, 5.8 dBi, cos^p); a measured pattern would replace the file.
- [x] ~~**D6 incidence**: array-centre incidence and no back-face culling, instead of per-element directions with cos ≤ 0 paths zeroed.~~ Reverted (T23; scaffold normals are flipped toward the radar first, since PCA normals of nodes 3/5 face their own cameras).
- [ ] **D7 occlusion**: no shadow mask (3DPS: init-time ray-cast `shadow_mask`). Secondary.
- [x] ~~**D8 phase in the gradient**: carrier phase and bin position attached to autograd (speckle gradients) instead of `detach_phase=True`.~~ Reverted (T23 default `detach_phase=True`).
- [x] **D13 per-channel gains**: implemented in the head (`sensor['gain_tx'/'gain_rx']`) and fitted in `octonet_fit fit-static` (20 + 20 log-amplitude gains, L2 0.01; fitted std 0.30 / 0.32 ≈ 2.6 / 2.8 dB, matching the GT's by-Tx 2.8 / by-Rx 2.7 dB). NOT yet in the grid fits (`octonet_fit_multiband`, `octonet_eval_panels`), whose pair-image correlation therefore sits at the pair-blind ceiling (0.41 of 0.45).
- [ ] **D16 normals**: PCA on the 5 cm cloud (wall normals 33° off) instead of planar fits on the full-resolution depth. Secondary.

## Valid differences (keep)
- D1 SFCW range compression = IFFT over the tones with the Hann PSF (the rasterizer's SFCW branch, bin = c/(2·K·df)).
- D2 material at the band frequency: ε′, ε″ from the ITU dispersion (a,b,c,d) at the band centre (per-tone evaluation
  is unnecessary within 4 GHz, < 0.1 dB); σ_h enters the BSDF as 4πσ_h/λ, so roughness is frequency-aware already.
- D9 per-pair bistatic path and carrier phase; D10 point primitive with areas = 1; D12 loss normalised by the masked
  median (no absolute calibration in the raw data); D14 per-pair log-magnitude range-profile loss (no RA term until the
  array is calibrated, T5–T7); D15 CUDA kernels (speed only).

## Guard rails (so the wrong version cannot run again)
- `octonet_heads.sfcw_head` IS the 3DPS-routed head; the old point sum is `sfcw_head_pointsum_deprecated` and raises
  unless `OCTONET_ALLOW_DEPRECATED_SFCW=1` is set explicitly.
- `nbs/method/sfcw_forward.render_sfcw_pairs` remains only as the reference of `sfcw_migration_selftest` (FMCW/SFCW
  parity on point targets); no head, fit, or figure script calls it.
- The heads self-test asserts the head equals a direct `render_factorized` call and that the deprecated path raises.

## Status log
- 2026-10-28: audit written, proof render done (`nbs/method/sfcw_3dps_render_check.py`).
- 2026-10-28 (night): **T23 done** — `octonet_heads.sfcw_head` is `render_factorized`; self-test 7/7 for SFCW (head == direct call to 1e-7, deprecated path raises, point targets, gradients to abcd and to the six raw parameters, gains). Single-modality SFCW panel (`viz/render_vs_gt_v2`, grid fit, no gains): pair-mean profile corr **0.53–0.67 → 0.964**, profile RMSE **2.5 → 0.81 dB**, image corr 0.27 → 0.41 (ceiling 0.45 without gains), per-pair median corr 0.36. `octonet_fit fit-static` (six parameters + gains): pool loss → 0.491, held recording 0.542 (old head 0.590 / 0.619 with an unphysical ρ).
- Open: D7 occlusion mask, D16 normals; gains in the grid fits; `preprocessing/octonet_mount_yaw.py` still sweeps with the old point sum and must be redone with the head before T15.
