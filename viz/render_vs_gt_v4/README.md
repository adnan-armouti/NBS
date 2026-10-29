# Single-modality training check (render_vs_gt_v4)

Each band is fitted ALONE (its own target only) with `4DPS/mm4DPS/octonet_fit_multiband.py`: the 3DPS per-point six-parameter material (eps from a per-point ITU (a, c) at the band centre with the exponents frozen, sigma_h / l_c / tau / thickness per point in the 3DPS raw space; 20k scaffold points, Adam lr 0.03, L2 0.05 to the init, 400 steps), the ported 4DPS receiver noise floor (measured held-out envelope prior x learned scale x 8-coefficient DCT range correction, added in power) and the ported 4DPS coupling block (coherent near-bin mean of held-out recordings, added after a learned absolute gain where it reproduces the target; the guard stays elsewhere), under TIGHT ITU bounds (a ∈ [1,15], b ∈ ±0.1, c ≤ 0.1 S/m, d ≤ 1.0) for 3 seeds, plus one LOOSE run (b ±0.2, c ≤ 2, d ≤ 1.5). 'iter 0' = uniform ITU concrete with the iteration-0 gain and floor. SFCW/UWB losses are log-magnitude L1 in absolute units (SFCW: pair-wise range profiles, xcorr-weighted, 20+20 channel gains; UWB: log-envelope, reliability-weighted); WiFi keeps the median-normalised magnitude-only loss (6 HT40 notches masked). Metrics are computed on the masked, median-normalised dB quantities (so they are comparable with v2/v3). Figures: `<band>_gt_iter0_trained.png`, `<band>_material.png`. Produced by `mm4DPS/octonet_eval_panels.py`.

## Summary (tight bounds, mean ± std over 3 seeds)

| band | loss iter 0 | loss trained (tight, 3 seeds) | loss trained (loose, 1 seed) | dB RMSE iter 0 → trained (SFCW ≥ 0.9 m, UWB ≥ 1 m) | corr iter 0 → trained (SFCW pair image ≥ 0.9 m, ceiling 0.89; UWB envelope ≥ 1 m, ceiling 0.95; WiFi all) | steps / runtime |
|---|---|---|---|---|---|---|
| SFCW | 0.717 ± 0.008 | **0.291 ± 0.003** | 0.287 | 8.87 ± 0.07 → **4.85 ± 0.02** | 0.175 ± 0.012 → **0.788 ± 0.002** | 400 ± 0 / 19 ± 0 s |
| UWB | 0.627 ± 0.007 | **0.147 ± 0.002** | 0.152 | 7.72 ± 0.05 → **2.73 ± 0.04** | 0.456 ± 0.019 → **0.929 ± 0.001** | 400 ± 0 / 5 ± 0 s |
| WIFI | 0.327 ± 0.010 | **0.100 ± 0.014** | 0.092 | 3.89 ± 0.33 → **1.33 ± 0.30** | -0.070 ± 0.044 → **0.881 ± 0.056** | 400 ± 0 / 4 ± 0 s |

## SFCW

Best tight seed: `sfcw_tight_s1` (loss 0.2881). Figures: `sfcw_gt_iter0_trained.png`, `sfcw_material.png`.

| run | gains | loss iter0 → trained | dB RMSE (pairs × bins) | image corr (all supervised bins) | image corr ≥ 0.9 m fine / 19 cm (ceiling 0.89 / 0.95) | per-pair corr (median) | pair-mean profile corr / RMSE dB | gain std tx/rx dB (GT) | gain corr tx / rx / pairs |
|---|---|---|---|---|---|---|---|---|---|
| sfcw_loose_s0 | yes | 0.7057 → 0.2866 | 8.11 → 4.20 | 0.599 → 0.901 | 0.190 → **0.792** / 0.907 | 0.632 → 0.878 | 0.921 → 0.998 / 2.49 → 0.62 | 2.81/3.25 (2.87/3.08) | 0.98 / 0.96 / 0.93 |
| sfcw_tight_s0 | yes | 0.7085 → 0.2911 | 8.11 → 4.26 | 0.599 → 0.898 | 0.190 → **0.786** / 0.904 | 0.632 → 0.875 | 0.921 → 0.998 / 2.49 → 0.67 | 2.78/3.37 (2.87/3.08) | 0.98 / 0.96 / 0.93 |
| sfcw_tight_s1 | yes | 0.7153 → 0.2881 | 8.23 → 4.22 | 0.586 → 0.900 | 0.160 → **0.791** / 0.906 | 0.623 → 0.879 | 0.909 → 0.999 / 2.66 → 0.61 | 2.80/3.36 (2.87/3.08) | 0.98 / 0.95 / 0.93 |
| sfcw_tight_s2 | yes | 0.7267 → 0.2943 | 8.26 → 4.26 | 0.593 → 0.898 | 0.175 → **0.786** / 0.903 | 0.621 → 0.872 | 0.928 → 0.998 / 2.54 → 0.62 | 2.91/3.27 (2.87/3.08) | 0.98 / 0.96 / 0.93 |

SFCW alone (T23 head = the 3DPS rasterizer; D13 per-channel gains fitted): the trained pair-mean profile follows the GT bin for bin over 0.9-3.5 m including the floor-ring dips and the rise beyond 2.6 m (the folded -X wall), profile corr 0.999, RMSE 0.61 dB. The per-pair image correlation is 0.900 (per-pair median 0.879); without gains a pair-blind render is capped at ~0.45 by the GT's fixed per-Tx/per-Rx gain banding (4 dB spread, 92 % additive). The fitted gains have std 2.80/3.36 dB (GT-derived 2.87/3.08) and correlate 0.98/0.95 with the GT's per-Tx/per-Rx pattern (0.93 on the 400-pair pattern). What remains is the per-pair ripple within each profile, which is below the scaffold's fidelity (8-bit depth, 3.9 cm quantisation at lambda 4.6 mm) and the uncalibrated per-element phases, so it is not a target. Loose bounds: see the table.


Fitted field (best tight seed) at 65.0 GHz: ε′ median 5.07 [p5 3.50, p95 7.66], σ median 1.189 [p5 0.499, p95 1.397] S/m; nearest ITU class: {'concrete': 18661, 'brick': 36, 'glass': 446, 'wood': 3, 'plasterboard': 358, 'marble': 496}.

## UWB

Best tight seed: `uwb_tight_s1` (loss 0.1438). Figures: `uwb_gt_iter0_trained.png`, `uwb_material.png`.

| run | loss iter0 → trained | dB RMSE (masked) | corr all supervised | corr ≥ 1 m fine / 39 cm (ceiling 0.95 / 0.98) | corr 1–9 m fine / 39 cm | corr 0.3–4 m | corr 4–9.9 m | RMSE dB 0.3–4 / 4–9.9 m |
|---|---|---|---|---|---|---|---|---|
| uwb_loose_s0 | 0.6202 → 0.1516 | 7.35 → 2.64 | 0.697 → 0.960 | 0.473 → **0.927** / 0.997 | 0.934 / 0.996 | 0.524 → 0.952 | 0.531 → 0.926 | 7.28 → 2.30 / 7.55 → 2.89 |
| uwb_tight_s0 | 0.6202 → 0.1496 | 7.35 → 2.63 | 0.697 → 0.961 | 0.473 → **0.927** / 0.996 | 0.935 / 0.996 | 0.524 → 0.953 | 0.531 → 0.926 | 7.28 → 2.27 / 7.55 → 2.90 |
| uwb_tight_s1 | 0.6371 → 0.1438 | 7.41 → 2.55 | 0.682 → 0.962 | 0.430 → **0.930** / 0.997 | 0.939 / 0.997 | 0.450 → 0.956 | 0.524 → 0.927 | 7.91 → 2.19 / 7.27 → 2.82 |
| uwb_tight_s2 | 0.6249 → 0.1465 | 7.30 → 2.61 | 0.696 → 0.961 | 0.464 → **0.928** / 0.997 | 0.936 / 0.998 | 0.495 → 0.953 | 0.536 → 0.927 | 7.52 → 2.29 / 7.33 → 2.86 |

UWB alone: the trained envelope tracks the GT mean level over the whole 0.3-9.9 m span (iter 0 was +15 dB too bright below 1.5 m and -20 dB too dark at 6-7 m), and the 6.3-6.8 m far-wall cluster appears at the right range. The GT's fine structure (20-30 dB interference fades every few cm) is not reproduced, so the correlation saturates near 0.5 and the RMSE at 6.7 dB: that structure needs the scaffold geometry to sub-cm accuracy and the UWB mount pose, both of which are assumed (camera yaw, node marker). The near range (0.3-1.5 m) keeps a +5-10 dB excess: node body returns absent from the scaffold.


Fitted field (best tight seed) at 7.0 GHz: ε′ median 4.85 [p5 2.79, p95 7.06], σ median 0.209 [p5 0.125, p95 0.233] S/m; nearest ITU class: {'concrete': 19034, 'brick': 59, 'glass': 169, 'wood': 49, 'plasterboard': 359, 'marble': 330}.

## WIFI

Best tight seed: `wifi_tight_s0` (loss 0.0884). Figures: `wifi_gt_iter0_trained.png`, `wifi_material.png`.

| run | loss iter0 → trained | dB RMSE all | corr all | Rx-1 RMSE / corr | Rx-2 | Rx-3 | Rx-4 (NLOS) |
|---|---|---|---|---|---|---|---|
| wifi_loose_s0 | 0.3132 → 0.0916 | 3.49 → 1.15 | -0.132 → 0.913 | 4.79→1.54 / -0.16→0.86 | 3.63→1.31 / -0.14→0.93 | 2.07→0.97 / -0.23→0.85 | 2.91→0.48 / -0.31→0.98 |
| wifi_tight_s0 | 0.3132 → 0.0884 | 3.49 → 1.08 | -0.132 → 0.925 | 4.79→1.21 / -0.16→0.91 | 3.63→1.48 / -0.14→0.93 | 2.07→0.85 / -0.23→0.85 | 2.91→0.53 / -0.31→0.98 |
| wifi_tight_s1 | 0.3338 → 0.0908 | 3.88 → 1.15 | -0.032 → 0.914 | 3.92→1.09 / -0.30→0.93 | 3.73→1.19 / 0.26→0.95 | 2.49→0.88 / -0.31→0.84 | 4.98→1.38 / -0.06→0.88 |
| wifi_tight_s2 | 0.3354 → 0.1202 | 4.31 → 1.75 | -0.046 → 0.802 | 6.74→1.39 / -0.08→0.89 | 3.97→2.97 / -0.13→0.69 | 2.81→1.15 / -0.67→0.68 | 2.29→0.53 / 0.50→0.98 |

WiFi alone: the trained |CSI| reproduces the broad spectral shape of every receiver and antenna (corr 0.87, RMSE 1.3 dB), including Rx-2's deep fade at subcarrier ~35 on antenna 0 and the NLOS Rx-4 slope; residuals are within +-3 dB except Rx-1 antenna 1 above subcarrier 90. This is achieved with SCHEMATIC Tx/Rx positions (+-0.5 m), so the material field is absorbing geometry error: the fitted eps' (median 5.2) differs from the SFCW-alone (3.2) and UWB-alone (1.6) fields of the same scene, which is the cross-band caveat.


Fitted field (best tight seed) at 5.18 GHz: ε′ median 5.24 [p5 2.49, p95 8.52], σ median 0.166 [p5 0.109, p95 0.211] S/m; nearest ITU class: {'concrete': 19325, 'brick': 10, 'glass': 13, 'wood': 188, 'plasterboard': 464, 'marble': 0}.

## Model state of these runs

Material / nuisances of the best tight runs: sfcw: material per-point, noise model on, coupling on, first supervised bin 0, nuisance {"log_gain": -2.3734349633499874, "floor": {"log_scale": -1.1760426205766918, "range_coeff": [-1.622338214481594, -0.85720919385727, -1.0284130257749067, 0.08846346117283382, 0.2984772056094385, 1.1602724953008448, 0.5848660715383778, 1.2602682696064926], "range_correction_min_max": [0.049787068367863944, 20.085536923187668], "median_power": 0.13742399719746273}, "coupling": {"adopt_from_bin": 0, "near_bins": 24}}; uwb: material per-point, noise model on, coupling on, first supervised bin 0, nuisance {"log_gain": 11.349482039642488, "floor": {"log_scale": -8.918991271247599, "range_coeff": [-0.664398085277167, 2.4324664762879715, 0.4884646370812384, -0.44067436742233757, -0.6690864433121884, 1.363238372956777, 0.600110123171647, -1.0726261300699724], "range_correction_min_max": [0.049787068367863944, 20.085536923187668], "median_power": 1.6261220846020117e-08}, "coupling": {"adopt_from_bin": 0, "near_bins": 156}}

## Reproduce

```
source tools/env_node.sh && cd 4DPS
python -m mm4DPS.octonet_noise measure        # held-out envelope + coupling (once)
OCTONET_PANELS_TAG=v4 CUDA_VISIBLE_DEVICES=3 python -m mm4DPS.octonet_eval_panels fit --band sfcw --seeds 0,1,2 --steps 400 --loose
OCTONET_PANELS_TAG=v4 CUDA_VISIBLE_DEVICES=4 python -m mm4DPS.octonet_eval_panels fit --band uwb  --seeds 0,1,2 --steps 400 --loose
OCTONET_PANELS_TAG=v4 CUDA_VISIBLE_DEVICES=5 python -m mm4DPS.octonet_eval_panels fit --band wifi --seeds 0,1,2 --steps 400 --loose
OCTONET_PANELS_TAG=v4 python -m mm4DPS.octonet_eval_panels figures
```

