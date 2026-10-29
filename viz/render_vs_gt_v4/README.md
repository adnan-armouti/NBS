# Single-modality training check (render_vs_gt_v4)

Each band is fitted ALONE (its own target only) with `4DPS/mm4DPS/octonet_fit_multiband.py`: the 3DPS per-point six-parameter material (eps from a per-point ITU (a, c) at the band centre with the exponents frozen, sigma_h / l_c / tau / thickness per point in the 3DPS raw space; 20k scaffold points, Adam lr 0.03, L2 0.05 to the init, 400 steps), the ported 4DPS receiver noise floor (measured held-out envelope prior x learned scale x 8-coefficient DCT range correction, added in power) and the ported 4DPS coupling block (coherent near-bin mean of held-out recordings, added after a learned absolute gain where it reproduces the target; the guard stays elsewhere), under TIGHT ITU bounds (a ∈ [1,15], b ∈ ±0.1, c ≤ 0.1 S/m, d ≤ 1.0) for 3 seeds, plus one LOOSE run (b ±0.2, c ≤ 2, d ≤ 1.5). 'iter 0' = uniform ITU concrete with the iteration-0 gain and floor. SFCW/UWB losses are log-magnitude L1 in absolute units (SFCW: pair-wise range profiles, xcorr-weighted, 20+20 channel gains; UWB: log-envelope, reliability-weighted); WiFi keeps the median-normalised magnitude-only loss (6 HT40 notches masked). Metrics are computed on the masked, median-normalised dB quantities (so they are comparable with v2/v3). Figures: `<band>_gt_iter0_trained.png`, `<band>_material.png`. Produced by `mm4DPS/octonet_eval_panels.py`.

## Summary (tight bounds, mean ± std over 3 seeds)

| band | loss iter 0 | loss trained (tight, 3 seeds) | loss trained (loose, 1 seed) | dB RMSE iter 0 → trained (SFCW ≥ 0.9 m, UWB ≥ 1 m) | corr iter 0 → trained (SFCW pair image ≥ 0.9 m, ceiling 0.89; UWB envelope ≥ 1 m, ceiling 0.95; WiFi all) | steps / runtime |
|---|---|---|---|---|---|---|
| SFCW | 0.717 ± 0.008 | **0.292 ± 0.002** | 0.291 | 8.87 ± 0.07 → **4.85 ± 0.02** | 0.175 ± 0.012 → **0.788 ± 0.002** | 400 ± 0 / 18 ± 0 s |
| UWB | 0.641 ± 0.008 | **0.140 ± 0.002** | 0.144 | 7.84 ± 0.05 → **2.65 ± 0.03** | 0.409 ± 0.023 → **0.929 ± 0.001** | 400 ± 0 / 6 ± 0 s |
| WIFI | 0.327 ± 0.010 | **0.081 ± 0.007** | 0.070 | 3.89 ± 0.33 → **1.00 ± 0.10** | -0.070 ± 0.044 → **0.935 ± 0.013** | 600 ± 0 / 7 ± 0 s |

## SFCW

Best tight seed: `sfcw_tight_s0` (loss 0.2897). Figures: `sfcw_gt_iter0_trained.png`, `sfcw_material.png`.

| run | gains | loss iter0 → trained | dB RMSE (pairs × bins) | image corr (all supervised bins) | image corr ≥ 0.9 m fine / 19 cm (ceiling 0.89 / 0.95) | per-pair corr (median) | pair-mean profile corr / RMSE dB | gain std tx/rx dB (GT) | gain corr tx / rx / pairs |
|---|---|---|---|---|---|---|---|---|---|
| sfcw_loose_s0 | yes | 0.7057 → 0.2915 | 8.11 → 4.23 | 0.599 → 0.899 | 0.190 → **0.789** / 0.905 | 0.632 → 0.876 | 0.921 → 0.998 / 2.49 → 0.62 | 2.82/3.28 (2.87/3.08) | 0.98 / 0.96 / 0.93 |
| sfcw_tight_s0 | yes | 0.7085 → 0.2897 | 8.11 → 4.23 | 0.599 → 0.899 | 0.190 → **0.789** / 0.905 | 0.632 → 0.874 | 0.921 → 0.998 / 2.49 → 0.66 | 2.79/3.37 (2.87/3.08) | 0.98 / 0.96 / 0.93 |
| sfcw_tight_s1 | yes | 0.7153 → 0.2907 | 8.23 → 4.23 | 0.586 → 0.899 | 0.160 → **0.789** / 0.906 | 0.623 → 0.876 | 0.909 → 0.999 / 2.66 → 0.61 | 2.80/3.37 (2.87/3.08) | 0.99 / 0.95 / 0.93 |
| sfcw_tight_s2 | yes | 0.7267 → 0.2950 | 8.26 → 4.27 | 0.593 → 0.898 | 0.175 → **0.786** / 0.904 | 0.621 → 0.871 | 0.928 → 0.998 / 2.54 → 0.66 | 2.89/3.25 (2.87/3.08) | 0.98 / 0.96 / 0.93 |

SFCW alone (T23 head = the 3DPS rasterizer; D13 per-channel gains fitted): the trained pair-mean profile follows the GT bin for bin over 0.9-3.5 m including the floor-ring dips and the rise beyond 2.6 m (the folded -X wall), profile corr 0.998, RMSE 0.66 dB. The per-pair image correlation is 0.899 (per-pair median 0.874); without gains a pair-blind render is capped at ~0.45 by the GT's fixed per-Tx/per-Rx gain banding (4 dB spread, 92 % additive). The fitted gains have std 2.79/3.37 dB (GT-derived 2.87/3.08) and correlate 0.98/0.96 with the GT's per-Tx/per-Rx pattern (0.93 on the 400-pair pattern). What remains is the per-pair ripple within each profile, which is below the scaffold's fidelity (8-bit depth, 3.9 cm quantisation at lambda 4.6 mm) and the uncalibrated per-element phases, so it is not a target. Loose bounds: see the table.


Fitted field (best tight seed) at 65.0 GHz: ε′ median 5.08 [p5 3.53, p95 7.67], σ median 1.190 [p5 0.477, p95 1.402] S/m; nearest ITU class: {'concrete': 18584, 'brick': 39, 'glass': 486, 'wood': 1, 'plasterboard': 367, 'marble': 523}.

## UWB

Best tight seed: `uwb_tight_s1` (loss 0.1384). Figures: `uwb_gt_iter0_trained.png`, `uwb_material.png`.

| run | loss iter0 → trained | dB RMSE (masked) | corr all supervised | corr ≥ 1 m fine / 39 cm (ceiling 0.95 / 0.98) | corr 1–9 m fine / 39 cm | corr 0.3–4 m | corr 4–9.9 m | RMSE dB 0.3–4 / 4–9.9 m |
|---|---|---|---|---|---|---|---|---|
| uwb_loose_s0 | 0.6314 → 0.1440 | 7.41 → 2.58 | 0.694 → 0.962 | 0.434 → **0.926** / 0.995 | 0.933 / 0.995 | 0.532 → 0.953 | 0.485 → 0.924 | 7.26 → 2.29 / 7.70 → 2.84 |
| uwb_tight_s0 | 0.6314 → 0.1421 | 7.41 → 2.54 | 0.694 → 0.963 | 0.434 → **0.928** / 0.995 | 0.935 / 0.995 | 0.532 → 0.954 | 0.485 → 0.926 | 7.26 → 2.25 / 7.70 → 2.79 |
| uwb_tight_s1 | 0.6519 → 0.1384 | 7.51 → 2.47 | 0.674 → 0.964 | 0.379 → **0.930** / 0.996 | 0.937 / 0.996 | 0.456 → 0.956 | 0.466 → 0.928 | 7.89 → 2.19 / 7.44 → 2.71 |
| uwb_tight_s2 | 0.6403 → 0.1399 | 7.41 → 2.51 | 0.688 → 0.964 | 0.416 → **0.930** / 0.997 | 0.937 / 0.998 | 0.500 → 0.954 | 0.482 → 0.930 | 7.50 → 2.28 / 7.53 → 2.72 |

UWB alone: the trained envelope tracks the GT mean level over the whole 0.3-9.9 m span (iter 0 was +15 dB too bright below 1.5 m and -20 dB too dark at 6-7 m), and the 6.3-6.8 m far-wall cluster appears at the right range. The GT's fine structure (20-30 dB interference fades every few cm) is not reproduced, so the correlation saturates near 0.5 and the RMSE at 6.7 dB: that structure needs the scaffold geometry to sub-cm accuracy and the UWB mount pose, both of which are assumed (camera yaw, node marker). The near range (0.3-1.5 m) keeps a +5-10 dB excess: node body returns absent from the scaffold.


Fitted field (best tight seed) at 7.0 GHz: ε′ median 4.88 [p5 2.81, p95 7.11], σ median 0.209 [p5 0.124, p95 0.233] S/m; nearest ITU class: {'concrete': 19040, 'brick': 61, 'glass': 153, 'wood': 38, 'plasterboard': 353, 'marble': 355}.

## WIFI

Best tight seed: `wifi_tight_s0` (loss 0.0711). Figures: `wifi_gt_iter0_trained.png`, `wifi_material.png`.

| run | loss iter0 → trained | dB RMSE all | corr all | Rx-1 RMSE / corr | Rx-2 | Rx-3 | Rx-4 (NLOS) |
|---|---|---|---|---|---|---|---|
| wifi_loose_s0 | 0.3132 → 0.0704 | 3.49 → 0.87 | -0.132 → 0.951 | 4.79→0.96 / -0.16→0.95 | 3.63→0.97 / -0.14→0.97 | 2.07→0.93 / -0.23→0.84 | 2.91→0.53 / -0.31→0.98 |
| wifi_tight_s0 | 0.3132 → 0.0711 | 3.49 → 0.85 | -0.132 → 0.953 | 4.79→0.92 / -0.16→0.95 | 3.63→0.95 / -0.14→0.97 | 2.07→0.95 / -0.23→0.83 | 2.91→0.52 / -0.31→0.98 |
| wifi_tight_s1 | 0.3338 → 0.0849 | 3.88 → 1.09 | -0.032 → 0.923 | 3.92→0.93 / -0.30→0.95 | 3.73→0.98 / 0.26→0.97 | 2.49→0.92 / -0.31→0.83 | 4.98→1.46 / -0.06→0.86 |
| wifi_tight_s2 | 0.3354 → 0.0867 | 4.31 → 1.05 | -0.046 → 0.928 | 6.74→1.35 / -0.08→0.89 | 3.97→1.10 / -0.13→0.95 | 2.81→1.00 / -0.67→0.81 | 2.29→0.56 / 0.50→0.98 |

WiFi alone: the trained |CSI| reproduces the broad spectral shape of every receiver and antenna (corr 0.87, RMSE 1.3 dB), including Rx-2's deep fade at subcarrier ~35 on antenna 0 and the NLOS Rx-4 slope; residuals are within +-3 dB except Rx-1 antenna 1 above subcarrier 90. This is achieved with SCHEMATIC Tx/Rx positions (+-0.5 m), so the material field is absorbing geometry error: the fitted eps' (median 5.2) differs from the SFCW-alone (3.2) and UWB-alone (1.6) fields of the same scene, which is the cross-band caveat.


Fitted field (best tight seed) at 5.18 GHz: ε′ median 5.17 [p5 3.08, p95 7.72], σ median 0.166 [p5 0.109, p95 0.214] S/m; nearest ITU class: {'concrete': 19519, 'brick': 3, 'glass': 38, 'wood': 98, 'plasterboard': 340, 'marble': 2}.

## Model state of these runs

Material / nuisances of the best tight runs: sfcw: material per-point, noise model on, coupling on, first supervised bin 0, nuisance {"log_gain": -2.3965211650308884, "floor": {"log_scale": -1.197979254992213, "range_coeff": [-1.4298900724583832, -0.9913813872630622, -0.9184539774585155, 0.015393281454984628, 0.4076573635700514, 0.9834042100834389, 0.5029086855518109, 1.127696399622918], "range_correction_min_max": [0.049787068367863944, 12.977154555199391], "median_power": 0.13564627569759782}, "coupling": {"adopt_from_bin": 0, "near_bins": 24}}; uwb: material per-point, noise model on, coupling on, first supervised bin 0, nuisance {"log_gain": 11.328744070445154, "floor": {"log_scale": -8.9148191677537, "range_coeff": [-0.7175223670448492, 2.5137891322541486, 0.492474993813227, -0.6104575100426194, -0.6706622093324548, 1.4038347616745976, 0.5695965150571469, -1.012463952371838], "range_correction_min_max": [0.049787068367863944, 20.085536923187668], "median_power": 1.662668107851161e-08}, "coupling": {"adopt_from_bin": 0, "near_bins": 156}}

## Reproduce

```
source tools/env_node.sh && cd 4DPS
python -m mm4DPS.octonet_noise measure        # held-out envelope + coupling (once)
OCTONET_PANELS_TAG=v4 CUDA_VISIBLE_DEVICES=3 python -m mm4DPS.octonet_eval_panels fit --band sfcw --seeds 0,1,2 --loose     # lr 0.03, 400 steps
OCTONET_PANELS_TAG=v4 CUDA_VISIBLE_DEVICES=4 python -m mm4DPS.octonet_eval_panels fit --band uwb  --seeds 0,1,2 --loose
OCTONET_PANELS_TAG=v4 CUDA_VISIBLE_DEVICES=5 python -m mm4DPS.octonet_eval_panels fit --band wifi --seeds 0,1,2 --loose     # lr 0.01, 600 steps
OCTONET_PANELS_TAG=v4 python -m mm4DPS.octonet_eval_panels figures
```

