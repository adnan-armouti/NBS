# Why SFCW / UWB correlations are below the 3DPS FMCW numbers, and what has to change (2026-10-29)

Question (user): the trained SFCW pair-image correlation (0.53) and UWB envelope correlation (0.46) in
`viz/render_vs_gt_v3` are far below what FMCW training gave in 3DPS. What limits them, and what must change?

Method: measure the **ceiling** of each metric from the data itself (one recording against the pool of five), measure
how fast a render loses correlation under known geometry noise, then test each candidate lever one at a time on the
node (`nbs/method/correlation_analysis.py`, `mm4DPS.octonet_fit fit-static`, `mm4DPS.octonet_fit_uwb`). JSON of the
analysis: node `$NBS_DATA_OUT/correlation_analysis.json`; fit outputs `fit_S1_sfcw/`, `fit_S1_uwb/`.

## 1. Summary

| band | v3 panel (grid V=64 + gains) | per-point six-parameter + gains | + learned noise floor | data ceiling (one recording vs pool) |
|---|---|---|---|---|
| SFCW pair image, fine (3.75 cm bins) | 0.528 | 0.677 (held recording 0.635) | **0.734 (held 0.692)** | 0.89 |
| SFCW pair image, 5-bin (19 cm) smoothing | 0.730 | 0.836 | **0.869 (held 0.850)** | 0.95 |
| SFCW pair-mean range profile | 0.90–0.95 | 0.987 | — | 0.99 |
| SFCW loss pool / held | 0.629 / — | 0.488 / 0.542 | **0.426 / 0.485** | — |
| UWB envelope, fine (6.4 mm bins), guard 0.29 m | 0.459 | 0.520 | 0.625 | 0.954 |
| UWB envelope, fine, guard 1.0 m | — | 0.622 | **0.694** | 0.947 |
| UWB envelope, 61-bin (39 cm), guard 1.0 m | 0.678 (guard 0.29) | 0.870 | **0.886** | 0.979 |
| UWB, 1.0–9.0 m only: fine / 15-bin / 61-bin | — | 0.859 / 0.947 / 0.962 | **0.854 / 0.945 / 0.973** | 0.947 / 0.962 / 0.979 |

Three changes account for most of the gap, in order of effect:

1. **Representation: per-point six-parameter material instead of the 24 cm voxel grid of ITU `abcd`.** SFCW fine
   image 0.533 → 0.674, 19 cm scale 0.730 → 0.836, profile 0.937 → 0.987; UWB 61-bin 0.678 → 0.870. The grid cannot
   express the per-surface contrast the pair images contain; the bounded six-parameter per-point material (3DPS's own
   parametrisation) can, and it still generalises to a held-out recording (SFCW held 0.635–0.692 vs pool 0.677–0.734).
2. **A learned additive noise floor on the rendered magnitude.** The raw profiles never fall below the receiver noise;
   the render does, so the log-magnitude loss spends material on reproducing floor-level bins. One learned scalar
   (0.30 × the render median for SFCW, 3e-4 relative for UWB) gives SFCW +0.06 fine / +0.03 smoothed / loss
   0.488 → 0.426 and UWB +0.07 fine / +0.02 at 39 cm. (`--floor` in both fit scripts.)
3. **UWB near guard 1.0 m (not 0.29 m), like SFCW's 0.9 m.** On a common 1.0 m evaluation mask the 1.0 m-guard fit is
   0.622 vs 0.525 for the 0.29 m-guard fit at fine scale. Bins below 1 m are the node's own body and table, which the
   scaffold does not contain; fitting them poisons the material of the first metres.

With all three, **UWB is at its data ceiling over 1–9 m** (0.85 / 0.95 / 0.97 vs 0.95 / 0.96 / 0.98) and **SFCW is at
0.73 / 0.87 against ceilings of 0.89 / 0.95**. The remaining gaps have identified causes (§3, §4).

## 2. Why the 3DPS FMCW numbers were higher: the metric, not only the model

- 3DPS compared **range–azimuth images**, i.e. the coherent sum over all virtual channels. That integration averages
  the per-channel speckle. Here the array geometry is uncalibrated (T5–T7), so we score the **400 individual Tx–Rx
  range profiles at 3.75 cm**: a far harsher, speckle-dominated metric. The matching quantity we can compute, the
  pair-mean profile, is already at 0.987 (ceiling 0.99).
- The pair-level fine image is intrinsically speckle: at λ = 4.6 mm a 5 mm position jitter of the same point cloud
  drops render-vs-render correlation to 0.45 (0.70 at 5-bin smoothing), and 10° normal noise to 0.84; the GT itself
  is reproducible only at 0.89 between recordings (0.66–0.97 pairwise). No material model can beat that; sub-wavelength
  geometry would be needed, and a bounded per-point range refinement with the phase attached (±1 / ±2 cm) was tried and
  made it worse (pool 0.671 → 0.648 / 0.661; held 0.629 → 0.620 / 0.626): 30 000 free offsets against 400 × 76
  observations are not identifiable. Rejected.
- UWB is different: its fine structure IS reproducible (0.954) and insensitive to cm geometry noise (0.95 at 10 cm
  smoothing even with 4 cm jitter), so its gap was model/representation, which §1 closes.

## 3. SFCW: what is left (0.73 → 0.89 at fine scale)

| cause | evidence | status |
|---|---|---|
| speckle from cm-scale geometry at λ 4.6 mm | §2; render-vs-render cap 0.45–0.48 under 0.5–4 cm jitter | inherent to the pair-level fine metric; report smoothed / profile / (after T5–T7) RA metrics as in 3DPS |
| residual range slope of the render | per-band residual after the fit: +0.2…+0.9 dB between 1 and 3 m, pair-std 5.1–5.7 dB in every band | small; the floor removed part of it; the rest sits with the 1/R² hemisphere weight vs the Vayyar's unknown range gain (needs a point target, T7) |
| per-pair delays | peak-bin spread 0.66 bins, 50 % additive; applying them: grid 0.522, per-point 0.640 (no gain) | rejected |
| static person in the loss | profile 0.987 → 0.912, image 0.674 → 0.623 | rejected (person is below the detectability gate, T9) |
| occlusion mask / planar normals | D7 / D16 in the findings file | implemented, within seed noise, not adopted |
| array calibration | T5–T7 blocked on an external reference | the only route to an RA metric and a coherent loss |

## 4. UWB: what is left

- **The 9–10 m band.** The GT envelope there is −56 dB, as strong as the far wall at 7 m (−57.5 dB), while the scaffold
  has 0.5 % of its points beyond 9 m from the X4 (p99 range 8.5 m). After the fit this band carries a −0.8 dB·ln
  residual and a negative fine correlation; it alone pulls the 1–10 m correlations from 0.85 / 0.95 / 0.97 down to
  0.69 / 0.82 / 0.89. Its origin is unverified (geometry outside the depth cameras' coverage or the end of the X4's
  1535-bin frame at 9.87 m; multibounce is not a model we use). Until identified, **cap the UWB loss and metrics at
  9 m** (`--range-max` to add; the 8–9 m band is also degraded, +0.29 residual, corr 0.79 at 39 cm).
- Roughness bounds at λ 4.3 cm: raising the σ_h bound from 0.9 mm to 2 cm changes nothing (0.621 vs 0.622; fitted
  σ_h stays ~0.1 mm). Not a lever.
- Pose (yaw) and the 65° pattern stand-in remain assumptions; the by-range residual after the fit is flat within
  ±0.2 dB·ln between 1 and 8 m, so there is no evidence of a pose error large enough to see in a single envelope.

## 5. What to change in the pipeline (recommended, in order)

1. **Single-band fits use the per-point six-parameter material + channel gains + learned floor** (`fit-static --floor`,
   `octonet_fit_uwb --guard-m 1.0 --floor`); the voxel-grid `abcd` field stays for the cross-band transfer experiments
   only. For zero-shot transfer (T20) the frequency-independent four (σ_h, l_c, τ, thickness) transfer per point as
   they are; ε′, ε″ need the dispersion, which a single band cannot identify per point → fit per-point (a, c) with the
   exponents (b, d) tied to an ITU class prior (one class per scaffold segment) rather than free per voxel.
2. **Noise floor as a standard term** of every band's head output (one scalar per band; it is a receiver property).
3. **UWB guard 1.0 m and range cap 9 m** until the 9–10 m return is explained.
4. **Report the metrics 3DPS reported**: pair-mean profile corr / RMSE and the smoothed image for SFCW now; RA image
   once T5–T7 unblock. Keep the fine pair image as a diagnostic with its ceiling (0.89) printed beside it.
5. Re-run `viz/render_vs_gt` panels (v4) with 1–4 so the tables reflect the adopted model.

Rejected in this analysis (do not retry without new evidence): per-pair delay compensation, static person term,
bounded range refinement with attached phase, UWB roughness bounds, occlusion mask, planar normals (as fit defaults).
