# Open items after v4: analysis and the approach forward (2026-10-29)

Scope: the four items left open after the per-point + 4DPS-nuisance port (`2026-10-29_correlation_analysis.md`):
the SFCW fine-scale gap, the UWB 9–10 m return, the WiFi seed, and T20 (zero-shot transfer). Everything below was
measured on the node today; scripts: `4DPS/mm4DPS/octonet_zero_shot.py` (T20 protocol, three arms), the ablation
switches of `octonet_fit_multiband` (`--material segment`, `--smooth-knn`, `--smooth-loss`), node
`$NBS_DATA_OUT/{zero_shot_S1_*, open_items_diag.json, t20_diag2.json}`.

## 1. The finding that reorders everything: nothing a single band learns transfers

T20 pilot, protocol = train on band X alone (the v4 per-point fit), freeze the material, render band Y with only Y's
receiver nuisances fitted on Y (absolute gain, floor scale + range correction, and for SFCW the 20 + 20 channel gains;
the coupling block is measured, not fitted). Three arms: **transfer** (X's material), **baseline** (uniform ITU
concrete, the init), **upper** (Y's own fit). Metric = the v4 correlation on Y's pre-coupling region.

| X → Y | baseline | transfer | upper bound |
|---|---|---|---|
| SFCW → UWB (envelope ≥ 1 m, 3 seeds) | 0.658 ± 0.004 | **0.657 ± 0.003** | 0.929 ± 0.001 |
| UWB → SFCW (pair image ≥ 0.9 m, 3 seeds) | 0.630 ± 0.004 | **0.622 ± 0.002** | 0.758 ± 0.001 |

Zero gain, both directions. It stays zero for every material form and regulariser tried (seed 0, same protocol):

| trained material | SFCW → UWB transfer (baseline 0.652) | UWB → SFCW transfer (baseline 0.634) |
|---|---|---|
| per-point (v4) | 0.654 | 0.625 |
| per-point + kNN Laplacian 1.0 / 10 | 0.655 / 0.652 | 0.626 / 0.628 |
| per-point, range-smoothed loss (5 / 15 bins) | 0.653 | 0.626 |
| 16 segments / 64 segments (k-means on position + normal) | 0.651 / 0.649 | 0.633 / 0.629 |
| voxel grid V = 64 | 0.655 | 0.634 |
| per-point on the planar-normal scaffold (v2) | 0.631 (baseline 0.630) | 0.622 (baseline 0.628) |

Why, measured:

- **The two bands' fitted materials are unrelated.** Per-point log-correlation between the SFCW-fitted and the
  UWB-fitted maps: ε′ 0.02, ε″ 0.05, σ_h 0.03, l_c 0.04, thickness 0.01 (τ 0.27, and τ barely moves). Over 16
  shared segments the correlations are −0.26 (ε′), −0.35 (thickness), +0.33 (σ_h). Per surface the medians sit at
  the init (ε′ floor / ceiling / walls: 5.15 / 5.17 / 5.06 from SFCW, 4.84 / 4.84 / 4.95 from UWB): neither band
  finds "the floor differs from the walls".
- **The in-band gain of the per-point material is memorisation.** Receiver nuisances alone (uniform concrete) reach
  SFCW 0.424 / UWB 0.418 loss; the per-point material takes that to 0.291 / 0.140; a 16-segment material takes it
  nowhere (0.410 / 0.404 at 400 steps, i.e. the nuisance level). The trained SFCW render's own fine structure survives
  a 2.5 mm jitter of its points at only 0.50 (0.73 at 19 cm) while it matches the GT at 0.79: the match beyond ~0.5 is
  the 120 k per-point parameters reproducing the GT's speckle realisation, not geometry-driven rendering. A joint
  two-band per-point fit satisfies both bands (0.343 / 0.157) for the same reason, so the "two bands jointly"
  alternative is not a way out either; the joint 16-segment fit is again nuisance-level (0.411 / 0.409).
- **What is left after the nuisances is speckle, not a structured model error.** 96 % of the SFCW residual variance
  is the per-pair-per-bin remainder (per-pair means 3 %, per-bin means 0.5 %), with lag-1 range autocorrelation 0.26
  (the Hann main lobe; the GT's own recording-to-recording residual has 0.19) and no pair-to-pair correlation.
  Incoherently integrated images already match: per-Tx (mean over Rx) 0.97, per-Rx 0.94, pair-mean profile 0.998.

Conclusion. With one static view per band, a 3.9 cm-quantised depth scaffold, assumed poses and datasheet stand-in
patterns, the receiver nuisances plus geometry explain all the coarse structure the two bands share; what remains
per band is fine-scale residual that a free material memorises and a constrained material cannot explain. **The
material is not identifiable from a single band in the current setup, so there is nothing to transfer.** The zero-shot
goal needs the information about material to be made identifiable first, not more fitting.

## 2. What does carry material information in one static view, and what hides it

A single static view constrains material through (i) the level of each surface relative to the others at the
incidence it is seen under, (ii) the Fresnel curve along a surface (a wall is seen from ~20° to ~70° along range),
(iii) slab interference (thickness) and (iv) the roughness-to-wavelength ratio. The array gives no incidence diversity
(4 cm aperture), polarisation is unknown (V-assumptions), and (iv) is band-specific by construction. So (i)–(iii) on a
handful of surfaces is the whole signal. Today it is hidden by: normals (A2: 10° of normal noise alone caps the
render-vs-render image at 0.84; wall normals were 33° off in T22; per-camera planar normals did not help), the
per-band pose and pattern assumptions (UWB yaw/pitch, X4 65° stand-in, Vayyar stand-in), and a loss that asks the
material to explain speckle. A per-point material absorbs each of these per band, which is exactly what the
uncorrelated maps show.

## 3. Proposed approach, in order, each with a gate

1. **Synthetic identifiability test first (T28, ~1 day).** Render a known scene with the same scaffold, the same
   heads, the measured floor and coupling block, and a known material assignment (3–5 ITU classes on the room's
   planes). Fit from one band exactly as on real data and evaluate zero-shot on the other, with the same three arms.
   If transfer fails on synthetic data, the failure is identifiability / head sensitivity (what each band can see of
   the six parameters) and the goal needs more views or different information; if it succeeds, the real-data failure
   is calibration and geometry, and items 2–4 are the fix. This separates the two explanations with the renderer as
   the data generator and costs nothing new. Gate: synthetic transfer reaches ≥ 50 % of (upper − baseline).
2. **Material-free pose and pattern calibration (T30, ~0.5 day).** The nuisance-only arm is a criterion that no
   material can absorb: sweep the UWB yaw/pitch and the Vayyar yaw and fit only the receiver nuisances (uniform
   concrete) at each pose; the loss minimum is the pose. The earlier yaw sweeps were unmeasurable because a material
   refit erased them; this arm removes that escape. Same for the X4 beamwidth (65° stand-in) and the Vayyar pattern.
   Gate: a single minimum > 3 σ of the loss noise; otherwise keep the camera yaw and say so.
3. **Room planes as the material unit, with an ITU-class prior (T29, 2–3 days).** RANSAC planes on the scaffold
   (floor, ceiling, walls, table); normals snapped to the planes (the strongest normal prior available, and the
   physical room); one material per plane, drawn from an ITU-class mixture (concrete, brick, plasterboard, wood,
   glass, metal) with the class exponents, so (a, c, b, d) are a class choice plus a small deviation rather than
   free per point. This is the only form in which (i)–(iii) of §2 is the signal and speckle is not. Gate: the
   SFCW-fitted and UWB-fitted per-plane ε′ correlate > 0.5 (today −0.26 over k-means segments), and the T20 transfer
   arm beats the baseline by > 3 × seed std.
4. **Loss on what is band-shareable.** With planes in place, fit the per-plane level and the along-plane Fresnel
   curve (the range-smoothed log magnitude, `--smooth-loss`, already implemented) and keep the fine pair image as a
   diagnostic with its ceiling printed beside it. The 3DPS analogues we can compute now are the per-Tx / per-Rx
   integrated images (0.97 / 0.94) and the pair-mean profile; the RA image waits for T5–T7.
5. **T20 restated** with the implemented protocol (`octonet_zero_shot.py`) and a pass condition: transfer − baseline
   > 3 × seed std and ≥ 25 % of (upper − baseline), both directions, reported with the nuisance-only baseline. The
   current number is 0 % and is the honest state of band synthesis on this data.

If item 1 says the information is not there even synthetically, the options are more views of the same room with
the same sensors (the dataset has none: node_1 is the only Vayyar/UWB node, fixed) or the two-band alternative with
constrained materials, which §1 shows reaches only the nuisance level today. That decision is the user's.

## 4. The three smaller items, closed

- **UWB 9–10 m return: a device artefact, masked.** The envelope rises 47 dB over the last 60 bins to a peak at the
  very last bin in all 10 Scene-1 recordings (shape correlation 0.975) and in Scenes 2 and 3 (different rooms; peak
  at the last bin, +16 / +22 dB), so it is the X4 frame end, not geometry; its held-out coherent mean predicts the pool
  at only 0.63, below the block adoption rule, so it is masked instead (last 80 bins = 0.51 m, `load_targets`).
  v4 UWB after the mask: loss 0.140 ± 0.002, RMSE 2.65 dB, correlation unchanged at 0.929.
- **WiFi seed: optimisation, fixed.** Seed 2's loss reached its minimum at step 318 and climbed afterwards (lr 0.03);
  at lr 0.01 / 600 steps the three seeds give 0.071 / 0.085 / 0.087. WiFi now defaults to lr 0.01, 600 steps
  (`BAND_DEFAULTS`): v4 WiFi 0.081 ± 0.007, correlation 0.935 ± 0.013.
- **SFCW fine-scale gap: speckle, and part of the 0.79 is memorised.** See §1; the honest SFCW numbers to report
  are the 19 cm image (0.90), the per-Tx / per-Rx integrated images (0.97 / 0.94) and the pair-mean profile (0.998),
  with the fine pair image (0.79, ceiling 0.89, jitter floor 0.50) as a diagnostic.

## 5. Rejected today (measured)

kNN Laplacian smoothing (1, 10), k-means segment materials (16, 64), range-smoothed loss (5 / 15 bins), the voxel
grid, and the planar-normal scaffold as routes to transfer (all equal to baseline); modelling the UWB frame end as a
coherent block (reproducibility 0.63 < 0.9).
