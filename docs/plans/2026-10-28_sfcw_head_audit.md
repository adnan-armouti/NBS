# Audit: the SFCW rendering head vs 3DPS FMCW rendering (2026-10-28)

**Finding first.** The OctoNet SFCW head (`4DPS/mm4DPS/octonet_heads.py::sfcw_head` → `nbs/method/sfcw_forward.py::
render_sfcw_pairs`) does not sit on the 3DPS rasterizer. It is the standalone point-sum forward model written in July
for the NBS SFCW migration, adopted by the port plan (§B5, "render SFCW via the NBS sfcw_forward path, bypassing the
rasterizer") as the "simplest first milestone", and never replaced. The 3DPS rasterizer already has an SFCW mode
(`nbs/mm3DPS/rasterizer.py`, `rasterizer_factorized.py`; `method/sfcw_migration_selftest.py` 21/21) that makes exactly
the two changes you specified: FMCW → SFCW range compression, and the material evaluated at the band's frequency.
Everything else in the head is a departure from 3DPS, and the departures are what broke it.

**Proof** (`nbs/method/sfcw_3dps_render_check.py`, node, calibrated Scene-1 scaffold, uniform ITU concrete, no training):

| | current head | 3DPS `render_factorized`, SFCW config |
|---|---|---|
| iteration-0 pair-mean profile corr vs GT | −0.33 | **+0.64** (σ_h 0.03 mm, l_c 3 cm; +0.57…+0.65 over σ_h ≤ 0.3 mm) |
| profile RMSE (dB, median-normalised) | 6.5 | **2.45** |
| energy from points < 1.5 m / −X wall / beyond 3.71 m | 71 % / 0.1 % / 1 % | 2.8 % / 25.8 % / 41.5 % |
| median |∂L/∂xyz| ÷ |∂L/∂material| | 2607 | **6.8** with the phase detached (26 381 attached) |
| per-pair image corr at iteration 0 | −0.09 | 0.14 (ceiling 0.45 without the 39 channel gains) |

## Difference-by-difference

| # | term | 3DPS FMCW (`render_factorized`) | SFCW head (`render_sfcw_pairs`) | verdict | change required |
|---|---|---|---|---|---|
| 1 | FMCW → SFCW range compression | Hann PSF splat on the K-bin range axis, bin = c·fs/(2·S·K) | IFFT over the 100 tones with a Hann window | **valid** (the intended change #1); the rasterizer's SFCW branch already does it (bin = c/(2·K·df), same PSF) | none: use the rasterizer's branch |
| 2 | material frequency dependence | one carrier (77 GHz), `bsdf.set_frequency` | ITU dispersion (a,b,c,d) → Fresnel per tone | **valid** (the intended change #2, Hofmann/Wirth-style per-band material), but per-tone evaluation is unnecessary within 4 GHz (< 0.1 dB) | evaluate ε(f) from (a,b,c,d) at the band centre and feed the rasterizer's six-parameter material: ε′, ε″ from the dispersion law, σ_h, l_c, τ, thickness as in 3DPS (σ_h enters as 4πσ_h/λ, so roughness is frequency-aware by construction) |
| 3 | scattering model | six-parameter Jones BSDF: slab Fresnel (s and p, polarisation basis), KA + SPM coherent lobes, directive + broad incoherent lobes, CBS, energy gating, f·cos | s-pol Fresnel amplitude Γ(θ_i) only; isotropic; no roughness, no lobes, no slab, no polarisation | **invalid** — this is the root cause of the shape error (grazing floor not suppressed, specular wall not formed) | revert: `render_factorized` with `reparameterize_torch` |
| 4 | geometry / radar equation | amplitude = C_radar · √(f·cos) · √G_tx/d_tx · √(G_rx·dΩ), with dΩ the hemisphere weight (= 1 per active point in the MC density-ratio form, or A·cos_o/d_rx² in the surface-integral form) → power ∝ 1/R² for the MC form | Γ / (d_tx · d_rx) → power ∝ 1/R⁴, no area, no cos_o, no antenna | **invalid** — the receiver solid-angle integral absorbs one distance; the head double-counts range (30 dB against the far wall over 6.6 m vs 1.2 m) | revert to the rasterizer's form with areas = 1 for active points (3DPS release convention) |
| 5 | antenna element gain | measured E/H plane patterns (`AntennaPatternTorch`, 361 samples, Gram-Schmidt local frame) for every Tx and Rx | none | **invalid** | revert; Vayyar has no measured pattern file, so synthesise `assets/antenna_pattern/VAYYAR/{tx,rx}_65.npy` from the datasheet 3 dB beamwidths (E 180°, H 82°, 5.8 dBi) as cos^p planes — flagged as a datasheet stand-in, replaceable by a measured pattern |
| 6 | incidence / back-facing | cos_i and cos_o per Tx/Rx path from exact per-element directions; back-facing paths (cos ≤ 0) zeroed | cos_i from the array centre only (used inside Fresnel); no back-face culling | **invalid** (minor: the 4 cm aperture makes the centre approximation harmless, but the culling matters) | revert |
| 7 | occlusion | `shadow_mask` (M, n_tx) from init-time ray tracing (Mitsuba) | none | **invalid** (secondary) | revert; for the point scaffold use a depth-buffer visibility from the radar in place of the mesh ray cast (no mesh exists), or run without a mask first and add it as the next step |
| 8 | phase handling in the gradient | `detach_phase=True`: carrier phase and bin position are constants for autograd; gradients flow through the amplitude | phase attached: position gradients 2600× material gradients (speckle) | **invalid** | revert: `detach_phase=True` |
| 9 | per-pair bistatic path | d_tx + d_rx per (point, Tx, Rx); carrier phase 2πf·τ | same | valid | none |
| 10 | primitive | points with an active mask (release: no opacity), normals from the model | bare points | valid in substance | pass the scaffold points, normals and `areas = 1` |
| 11 | material storage / bounds | raw (M,6) with `reparameterize_torch` bounds, per-point | ρ per point in [1/20, 20] or a voxel-grid (a,b,c,d) | mixed: per-point is 3DPS; the grid was added for the cross-band transfer | keep the 3DPS raw (M,6) parameterisation with ε from (a,b,c,d) at the band centre; grid storage stays an option on top |
| 12 | absolute gain | fixed `radar_constant` (C_radar, GT-match ×100) | loss normalised by the masked median | valid either way | keep the median normalisation in the loss (the raw Vayyar phasors carry no absolute calibration) |
| 13 | per-channel gains | not needed (TI cascade data arrive calibrated) | not modelled; the raw Vayyar phasors have a fixed per-Tx + per-Rx gain pattern (4 dB spread, 92 % additive, frame-stable 0.92) | **new, needed** | add 20 + 19 gain parameters applied to the rendered pair image (closed form from the medians or learned) |
| 14 | loss | RA-map correlation / MSE on magnitude images (azimuth FFT over the virtual array) | per-pair log-magnitude L1 on range profiles, xcorr-weighted, bins < 0.9 m masked | valid: no RA term is possible until the Vayyar array is calibrated (T5–T7); the per-pair profile loss is the right stand-in, with 13 and 8 | keep; add the gains |
| 15 | CUDA kernels | fused Step-4/5 kernels (`mm3DPS_cuda`, built on the node) | pure torch loop over 400 pairs | valid (speed only) | `use_cuda_kernels=True` once the torch path is validated (the check above used the torch path) |
| 16 | normals | from the model (trained or PCA) | PCA on the 5 cm cloud (wall normals 33° off from 8-bit depth) | valid but low quality | planar-fit normals from the full-resolution depth (secondary) |

Items 1, 2, 9, 10, 12, 14, 15 stand. Items 3–8 must be reverted by routing the head through `render_factorized`; 13 is
the one genuinely new term the Vayyar data needs; 16 and 7 are secondary improvements.

## Changes to make it happen (T23)
1. `octonet_heads.sfcw_head` → build the Vayyar SFCW config at the calibrated pose (`vayyar_radar_config.build_config`,
   already done in the check), instantiate `mm3DPS.rasterizer.Rasterizer` with the synthesised Vayyar pattern, map the
   material state to the rasterizer's raw (M,6): ε′, ε″ = ITU dispersion (a,b,c,d) at 65 GHz; σ_h, l_c, τ, thickness
   per point (defaults 0.1 mm / 1 cm / 0.5 / 0.1 m, learnable); call `render_factorized(..., detach_phase=True)`;
   return the (400,100) complex profiles in the target's tx-major pair order; multiply by the 39 channel gains.
2. Keep `render_sfcw_pairs` only as the reference in `sfcw_migration_selftest` (the FMCW/SFCW parity tests), not as a head.
3. `octonet_fit` / `octonet_fit_multiband`: optimise the six raw parameters (and the gains) instead of ρ; same loss.
4. Acceptance: iteration-0 profile corr ≥ +0.6 (now reproducible), trained profile corr > 0.8 and RMSE < 1.5 dB on the
   pooled target with the same gain on the held recording; pair-image corr → its ceiling (~0.6 with gains).
5. Then UWB (same audit: its head is the same point-sum with a pulse kernel; the fix is the same routing with the UWB
   pulse as the PSF) and the cross-band protocol.
