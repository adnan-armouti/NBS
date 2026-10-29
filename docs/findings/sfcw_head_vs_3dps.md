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
- [~] **D7 occlusion**: implemented (`mm4DPS/octonet_geometry.py::visibility_mask`, depth-buffer visibility from the radar on a 0.25° grid + back hemisphere; `shadow_mask.npy` written by the scaffold stage, the head consumes `cloud['shadow_mask']`) but NOT adopted: it removes 17 % of points and does not improve iteration 0 (+0.636 → +0.614; the near node body is already masked by the bin < 0.9 m guard). Off by default (`OCTONET_SHADOW_MASK=1` to enable); revisit with a fit rather than at iteration 0.
- [x] ~~**D8 phase in the gradient**: carrier phase and bin position attached to autograd (speckle gradients) instead of `detach_phase=True`.~~ Reverted (T23 default `detach_phase=True`).
- [x] ~~**D13 per-channel gains** (new, Vayyar-specific): the raw phasors carry a fixed per-Tx + per-Rx gain (4 dB spread, 92 % additive); 20 + 19 gains must be modelled.~~ Done: in the head (`sensor['gain_tx'/'gain_rx']`), in `octonet_fit fit-static`, and in the grid fits (`octonet_fit_multiband`, `octonet_eval_panels`, `--no-gains` to disable). Fitted gains correlate with the GT's per-Rx pattern at 0.98–0.99 and per-Tx at 0.95–0.97 with the right 2.8–3.0 dB spread; SFCW pair-image corr 0.41 → 0.58 (v1 scaffold, 400 steps), profile corr 0.971–0.978, RMSE 0.63–0.72 dB.
- [~] **D16 normals**: planar-fit normals on the full-resolution depth implemented (`octonet_geometry.planar_normals_depth`, r = 12 px; scaffold v2 `4dps_scaffold_S1_v2`, stage default `--normals planar`). Per camera they are better where measurable (node_1 wall 14.7° → 13.2°, floors 2–3° → 2°), iteration-0 profile corr +0.619 → +0.636; the fused-cloud 'wall' metric (34° → 37°) is dominated by node_3 points in that region which fail every method (60–71°): clutter or a misregistered node_3 far field, not a normals problem. **Not adopted as the fit default** (2026-10-29 A/B, 400 steps): without gains v2 trains worse than v1 (loss 0.735 vs 0.665, profile corr 0.877 vs 0.964); with gains the two are within seed noise (0.625–0.649 vs 0.618–0.644). Available via `OCTONET_SCAFFOLD_DIR=.../4dps_scaffold_S1_v2`.

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
- 2026-10-29: gains in the grid fits (D13 done; with the full 400 steps and the fixed convergence rule the SFCW grid fit gives loss 0.62–0.65, pair-image corr 0.51–0.54, profile corr 0.90–0.95, RMSE 1.2–1.4 dB over seeds: the gains buy per-pair fit at a small cost in pair-mean fidelity vs the no-gains 0.964 — an earlier agent report of 0.578 / 0.584 could not be reproduced with the committed code); D7 implemented, not adopted; D16 planar normals implemented (scaffold v2). Mount-yaw sweep redone through the head (`octonet_mount_yaw_v2.py`): yaw-sensitive now, but a broad monotone trend (+40…+60°, 4.5 % of the loss), the three metrics disagree and a 60-step material refit erases it; verdict: keep the camera yaw and level pitch (T15 stays open until an RA criterion exists).
- 2026-10-29: **UWB head routed through the same rasterizer** (`uwb_head` → `render_factorized` with a `UWB` modulation branch: K = 1535, bin c/2fs, the measured X4 RF pulse as a `TemplatePSFTable`, on-board delay 7 bins, zero per-path carrier phase because the template carries its own carrier; X4 datasheet pattern stand-in 65°; FMCW/SFCW bitwise unchanged, 21/21 + heads self-test). A point target reproduces the measured coupling pulse (corr 0.992). Iteration 0 on the scaffold: loss 1.18 → 0.93 and the strongest rendered return moves to 6.61 m (the far wall, where the GT's cluster is). The old UWB point sum is `uwb_head_pointsum_deprecated` (guarded).
