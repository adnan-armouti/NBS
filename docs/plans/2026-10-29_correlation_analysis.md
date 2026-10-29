# Why SFCW / UWB correlations trailed the 3DPS FMCW numbers, and what changed (2026-10-29, corrected)

Question (user): the trained SFCW pair-image correlation (0.53) and UWB envelope correlation (0.46) in
`viz/render_vs_gt_v3` were far below what FMCW training gave in 3DPS. What limits them, what must change?

The first version of this document proposed three "levers". The user's review turned each into a correction of the
pipeline, which is the real finding: **the OctoNet fits had departed from 3DPS/4DPS in three places, and the gap was
those departures, not the data.**

| what the first version proposed | the correction | what was done |
|---|---|---|
| "per-point six-parameter material instead of the 24 cm voxel grid" | 3DPS optimises per-point materials; this should never have been anything else | `PerPointMaterial` is the default of `octonet_fit_multiband` and of the panels; the voxel grid is `--material grid` for transfer experiments only |
| "a learned additive noise floor on the rendered magnitude" (one data-driven scalar) | the floor must be physics-informed: port the 4DPS noise model first | 4DPS `LearnedNoiseFloor` ported (`octonet_noise.py`): measured envelope prior from held-out recordings × learned scale × DCT range correction, added in POWER under the magnitude |
| "UWB near guard at 1.0 m" | gating is acceptable only at the near/far boundary; better, port 4DPS, which models the near field instead of gating | 4DPS coupling block ported (`octonet_noise.py`): coherent near-bin mean of held-out recordings added after a learned absolute gain; adopted from bin 0 in both bands, so no gate remains |

Code: `4DPS/mm4DPS/octonet_noise.py` (new), `octonet_fit_multiband.py` (`PerPointMaterial`, `BandNuisance`, switches),
`octonet_eval_panels.py` (switches, far-region metrics), `octonet_fit.py` (`fit-static --noise-model --coupling`),
`nbs/method/sfcw_target.py` (`normalize=`). Measurements: node `$NBS_DATA_OUT/noise_S1/`. Panels: `viz/render_vs_gt_v4/`.

## 1. Result

v4 panels (per-point material + channel gains + learned noise floor + coupling block, tight ITU bounds, 3 seeds, 400
steps) against v3 (voxel grid, no nuisances, near gates) and the data ceilings (one recording against the pool):

| band, metric (region = the pre-coupling-model supervised region, so v3 and v4 are comparable) | v3 | **v4** | ceiling |
|---|---|---|---|
| SFCW pair image ≥ 0.9 m, fine 3.75 cm bins | 0.528 | **0.788 ± 0.002** | 0.89 |
| SFCW pair image ≥ 0.9 m, 19 cm smoothing | 0.730 | **0.904** | 0.95 |
| SFCW pair-mean range profile corr / RMSE | 0.90–0.95 / 1.2–1.4 dB | **0.998 / 0.62 dB** | 0.99 |
| SFCW pair image, all supervised bins (v4: from bin 0) | — | 0.90 | — |
| SFCW per-pair corr, median | 0.32 | **0.88** | — |
| UWB envelope ≥ 1 m, fine 6.4 mm bins | 0.459 | **0.929 ± 0.001** | 0.95 |
| UWB envelope 1–9 m, fine / 39 cm | — | 0.937 / 0.997 | 0.95 / 0.98 |
| UWB RMSE ≥ 1 m | 8.0 dB | **2.7 dB** | — |
| WiFi |CSI| corr all / RMSE | 0.940 / 0.96 dB | 0.881 ± 0.056 / 1.33 dB | — |

SFCW is now within 0.10 of its fine-scale ceiling and 0.05 of the 19 cm ceiling; UWB is within 0.02 of its ceiling.
WiFi is unchanged within seed spread (one seed of three is worse under the per-point prior; WiFi uses no nuisance
model and no roughness, its head reads only the permittivity).

## 2. What each correction is worth (seed 0, 400 steps; SFCW pair image ≥ 0.9 m fine / 19 cm; UWB envelope ≥ 1 m fine / 39 cm)

| model | SFCW | UWB |
|---|---|---|
| voxel grid, no nuisances (the v3 model) | 0.524 / 0.726 | 0.454 / 0.820 |
| voxel grid + noise floor + coupling block | 0.728 / 0.864 | 0.880 / 0.992 |
| per-point, no nuisances | 0.698 / 0.849 | 0.636 / 0.870 |
| per-point + noise floor | 0.796 / 0.907 | 0.907 / 0.991 |
| per-point + coupling block | 0.751 / 0.883 | 0.905 / 0.979 |
| **per-point + noise floor + coupling block (default)** | **0.786 / 0.904** | **0.927 / 0.997** |

Each of the three corrections is worth more than any single lever the first version proposed. On SFCW the noise
floor alone equals the full model in the ≥ 0.9 m region (0.796 vs 0.786, within the 3-seed spread); the coupling block
adds the near bins (per-pair median corr 0.69 → 0.88) at no cost beyond that. On UWB the two nuisances are each worth
+0.27 and together +0.29.

## 3. The ported models, exactly

**Per-point material** (3DPS): one six-parameter Jones material per scaffold point, frequency-aware: ε′, ε″ from a
per-point ITU-R P.2040 dispersion ε′ = a f^b, σ = c f^d at the band centre with (a, c) per point and the exponents
(b, d) frozen at the init class (a single band cannot identify them; `--free-exponents` for joint fits), and σ_h, l_c,
τ, thickness per point in the 3DPS raw space (`reparameterize_torch` bounds). L2 prior 0.05 to the init in raw space.

**Learned noise floor** (4DPS `signal_model.LearnedNoiseFloor` + `calibration noise-envelope`): expected noise power
per channel and range bin P(c, k) = s² · prior(c, k) · exp(clip(Σ_j w_j DCT_j(k), ±3)), with the prior measured from
five HELD-OUT Scene-1 recordings (6–10 of the activity-distinct list; the pooled target is 1–5) as the power of the
frame-to-frame residual about each recording's coherent mean, median over recordings, normalised to unit level
beyond the near region; for UWB the person's range band (2.17–3.67 m, ±0.75 m about the subject) is dropped and
interpolated as in 4DPS; for SFCW it is not (the person is not observable in the SFCW, a measured fact). The learned
scale s starts at the target's 5th-percentile power and the 8 DCT coefficients at zero. The floor enters UNDER the
square root of the rendered power, |model| = √(|render|² + P), behind a learned per-band absolute gain (init = the
iteration-0 median log ratio over the far bins), and the loss is then the un-normalised log-magnitude L1
(`sfcw_static_loss(normalize=False)`). Nothing is drawn: the target is a temporal mean, so the expectation is what is
compared. Fitted SFCW floor at 30 k points: range correction 0.06–3.7×, i.e. the receiver floor is not white in range.

**Coupling block** (4DPS `calibration sequence-coupling` + `forward_render.coupling_block`): per channel the coherent
mean over frames and the same five held-out recordings of the complex range profiles at the near bins (SFCW: 400 × 24
on the two-tone-cleaned profiles the target is built from; UWB: the RF frame, 155 bins = 1.0 m), coherent fraction of
the near power 0.991 (SFCW) / 0.999 (UWB). Added to the render after the absolute gain. Adoption rule, fixed before the
measurement: a region is modelled (not gated) if the held-out block predicts the POOLED target's near bins with a
per-channel log-magnitude correlation ≥ 0.9, the GT's own far-bin reproducibility. Measured: SFCW 0.999 (bins 0–5,
direct coupling residue) and 0.984 (bins 6–23, housing/table); UWB 0.994 (direct pulse, bins 0–44) and 0.979
(housing/table, 45–154), RF correlation 0.9999 / 0.996 → adopted from bin 0 in both bands. No near gate remains.

On the near/far boundary the user asked about: the antenna far-field distance 2D²/λ is 0.7 m for the Vayyar's 4 cm
aperture at 4.6 mm and about 2 cm for the X4 at 4.3 cm, so the old gates (0.9 m SFCW, 0.29 m then 1.0 m UWB) were
never far-field bounds; they were the extent of the node structure the scaffold does not contain. That extent is
exactly what the coupling block carries. The far-field numbers are recorded in `octonet_noise.py` as the gating rule
only for a band whose block ever fails the adoption rule.

## 4. What is left, with its cause

- **SFCW fine-scale pair image, 0.79 against 0.89.** Per-pair speckle at λ = 4.6 mm: jittering the same cloud by 5 mm
  drops render-vs-render correlation to 0.45, and 10° normal noise to 0.84; the GT itself reproduces at 0.89 between
  recordings. 3DPS scored beamformed range–azimuth images, where the coherent sum over channels averages this out; we
  cannot beamform until the array is calibrated (T5–T7). The pair-mean profile, the closest analogue, is at 0.998.
  Bounded per-point range refinement with the phase attached (±1 / ±2 cm) was tried and made it worse: rejected.
- **UWB 9–10 m band.** The GT envelope there is −56 dB, as strong as the far wall at 7 m, while the scaffold has 0.5 %
  of its points beyond 9 m from the X4 (p99 range 8.5 m). The 1–9 m correlation is 0.937 against 0.929 for ≥ 1 m, so
  it costs little now, but its source is unverified (geometry outside the depth cameras' coverage or the end of the
  X4's 1535-bin frame; multibounce is not a model we use).
- **WiFi seed spread** under the per-point prior (0.80–0.93); WiFi Tx/Rx positions are still schematic.

## 5. Rejected in this analysis (measured; do not retry without new evidence)

Per-pair delay compensation (SFCW), a static person term (SFCW), bounded per-point range refinement with attached
phase, wider UWB roughness bounds (σ_h up to 2 cm: fitted roughness stays at 0.1 mm), the occlusion mask and planar
normals as fit defaults, and the data-driven magnitude-additive floor (withdrawn in favour of the 4DPS model).

## 6. Next

1. T20, the project's goal: train on one band with the per-point model and evaluate zero-shot on the others. The four
   frequency-independent parameters transfer per point as they are; ε′, ε″ need the dispersion, which one band cannot
   identify per point, so fit per-point (a, c) with the exponents from an ITU class prior per scaffold segment, and
   redo T12's held-out-band gate (done with the grid) per point.
2. Report the metrics 3DPS reported: pair-mean profile and smoothed images now, the RA image once T5–T7 unblock; keep
   the fine pair image as a diagnostic with its ceiling printed beside it.
3. Identify the UWB 9–10 m return before it is modelled or excluded.
