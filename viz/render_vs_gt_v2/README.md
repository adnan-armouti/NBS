# Single-modality training check (render_vs_gt_v2)

Each band is fitted ALONE (its own target only) with the voxel-grid ITU material field of `4DPS/mm4DPS/octonet_fit_multiband.py` (V = 64 voxels per axis over the calibrated scaffold, 20k scaffold points, Adam lr 0.03, TV 1e-3, up to 400 steps), under TIGHT bounds (a ∈ [1,15], b ∈ ±0.1, c ≤ 0.1 S/m, d ≤ 1.0) for 3 seeds, plus one LOOSE run (b ±0.2, c ≤ 2, d ≤ 1.5). 'iter 0' = the grid's initialisation = uniform ITU concrete. Losses are the bands' median-normalised log-magnitude L1 (SFCW: pair-wise range profiles, bins < 0.9 m masked as node near-field, xcorr-weighted; UWB: log-envelope, guard 0.29 m, reliability-weighted; WiFi: log|CSI| per receiver/antenna/subcarrier, 6 HT40 notches masked, magnitude only). Metrics are computed on the same masked, median-normalised dB quantities. Figures: `<band>_gt_iter0_trained.png`, `<band>_material.png`. Produced by `mm4DPS/octonet_eval_panels.py`.

## Summary (tight bounds, mean ± std over 3 seeds)

| band | loss iter 0 | loss trained (tight, 3 seeds) | loss trained (loose, 1 seed) | dB RMSE iter 0 → trained | corr iter 0 → trained | steps / runtime |
|---|---|---|---|---|---|---|
| SFCW | 1.024 ± 0.004 | **0.718 ± 0.017** | 0.693 | 11.16 ± 0.04 → **8.15 ± 0.16** | -0.089 ± 0.001 → **0.271 ± 0.030** | 329 ± 52 / 35 ± 5 s |
| UWB | 1.184 ± 0.018 | **0.516 ± 0.039** | 0.573 | 12.97 ± 0.04 → **6.71 ± 0.38** | -0.089 ± 0.014 → **0.525 ± 0.058** | 400 ± 0 / 5 ± 0 s |
| WIFI | 0.327 ± 0.010 | **0.098 ± 0.017** | 0.159 | 3.89 ± 0.33 → **1.32 ± 0.38** | -0.070 ± 0.044 → **0.866 ± 0.085** | 251 ± 106 / 6 ± 3 s |

## SFCW

Best tight seed: `sfcw_tight_s1` (loss 0.6956). Figures: `sfcw_gt_iter0_trained.png`, `sfcw_material.png`.

| run | loss iter0 → trained | dB RMSE (pairs × bins) | image corr | per-pair corr (median) | pair-mean profile corr / RMSE dB |
|---|---|---|---|---|---|
| sfcw_loose_s0 | 1.0214 → 0.6930 | 11.16 → 7.91 | -0.088 → 0.326 | -0.098 → 0.251 | -0.331 → 0.610 / 6.45 → 2.46 |
| sfcw_tight_s0 | 1.0214 → 0.7211 | 11.16 → 8.21 | -0.088 → 0.261 | -0.098 → 0.207 | -0.331 → 0.531 / 6.45 → 2.64 |
| sfcw_tight_s1 | 1.0202 → 0.6956 | 11.11 → 7.93 | -0.091 → 0.313 | -0.094 → 0.240 | -0.371 → 0.669 / 6.45 → 2.28 |
| sfcw_tight_s2 | 1.0302 → 0.7359 | 11.20 → 8.31 | -0.089 → 0.240 | -0.095 → 0.184 | -0.337 → 0.464 / 6.54 → 2.79 |

SFCW alone: the fit flattens the uniform-concrete render (which was +10 dB too bright at 0.9 m and -7 dB too dark at 3.5 m) onto the GT pair-mean profile within about +-2 dB over 0.9-2.6 m, reproducing the floor-ring dips at 1.5 / 2.0 / 2.2 m. It still under-predicts the rise of the GT beyond 2.6 m (GT +6 dB by 3.5 m, render flat: the far wall sits beyond the 3.71 m unambiguous range and wraps, and the subject spot at 2.9 m is a static chair/person return the scaffold cuts), so the residual reaches -5 dB at 3.5 m. The per-pair image correlation stays low (0.27): the GT image has a strong pair-wise banding (per-element gain differences across the 400 pairs) that the render cannot produce without a per-channel gain calibration, and the per-pair ripple is below the scaffold's fidelity. Loose bounds reach 0.69 vs 0.72 by pushing sigma to 5 S/m on a few voxels.


Fitted field (best tight seed) at 65.0 GHz: ε′ median 3.19 [p5 0.85, p95 18.93], σ median 0.745 [p5 0.010, p95 4.626] S/m; nearest ITU class: {'concrete': 10429, 'brick': 4004, 'glass': 303, 'wood': 1957, 'plasterboard': 2147, 'marble': 1160}.

## UWB

Best tight seed: `uwb_tight_s1` (loss 0.4618). Figures: `uwb_gt_iter0_trained.png`, `uwb_material.png`.

| run | loss iter0 → trained | dB RMSE (masked) | corr all | corr 0.3–4 m | corr 4–9.9 m | RMSE dB 0.3–4 / 4–9.9 m |
|---|---|---|---|---|---|---|
| uwb_loose_s0 | 1.1792 → 0.5733 | 12.97 → 7.20 | -0.091 → 0.455 | 0.204 → 0.547 | -0.233 → 0.511 | 12.29 → 7.38 / 13.38 → 7.10 |
| uwb_tight_s0 | 1.1792 → 0.5532 | 12.97 → 7.10 | -0.091 → 0.464 | 0.204 → 0.551 | -0.233 → 0.526 | 12.29 → 7.27 / 13.38 → 6.99 |
| uwb_tight_s1 | 1.1645 → 0.4618 | 12.92 → 6.19 | -0.104 → 0.603 | 0.189 → 0.692 | -0.238 → 0.620 | 11.98 → 6.19 / 13.50 → 6.19 |
| uwb_tight_s2 | 1.2080 → 0.5321 | 13.02 → 6.86 | -0.071 → 0.507 | 0.225 → 0.623 | -0.200 → 0.533 | 12.13 → 6.80 / 13.57 → 6.90 |

UWB alone: the trained envelope tracks the GT mean level over the whole 0.3-9.9 m span (iter 0 was +15 dB too bright below 1.5 m and -20 dB too dark at 6-7 m), and the 6.3-6.8 m far-wall cluster appears at the right range. The GT's fine structure (20-30 dB interference fades every few cm) is not reproduced, so the correlation saturates near 0.5 and the RMSE at 6.7 dB: that structure needs the scaffold geometry to sub-cm accuracy and the UWB mount pose, both of which are assumed (camera yaw, node marker). The near range (0.3-1.5 m) keeps a +5-10 dB excess: node body returns absent from the scaffold.


Fitted field (best tight seed) at 7.0 GHz: ε′ median 1.62 [p5 0.97, p95 16.90], σ median 0.007 [p5 0.001, p95 0.311] S/m; nearest ITU class: {'concrete': 3156, 'brick': 13527, 'glass': 250, 'wood': 799, 'plasterboard': 1013, 'marble': 1255}.

## WIFI

Best tight seed: `wifi_tight_s2` (loss 0.0842). Figures: `wifi_gt_iter0_trained.png`, `wifi_material.png`.

| run | loss iter0 → trained | dB RMSE all | corr all | Rx-1 RMSE / corr | Rx-2 | Rx-3 | Rx-4 (NLOS) |
|---|---|---|---|---|---|---|---|
| wifi_loose_s0 | 0.3132 → 0.1587 | 3.49 → 2.73 | -0.132 → 0.622 | 4.79→1.27 / -0.16→0.91 | 3.63→5.21 / -0.14→0.41 | 2.07→0.90 / -0.23→0.84 | 2.91→0.52 / -0.31→0.98 |
| wifi_tight_s0 | 0.3132 → 0.1220 | 3.49 → 1.86 | -0.132 → 0.746 | 4.79→1.31 / -0.16→0.90 | 3.63→3.32 / -0.14→0.38 | 2.07→0.92 / -0.23→0.84 | 2.91→0.56 / -0.31→0.98 |
| wifi_tight_s1 | 0.3338 → 0.0880 | 3.88 → 1.03 | -0.032 → 0.929 | 3.92→1.29 / -0.30→0.90 | 3.73→1.13 / 0.26→0.95 | 2.49→0.96 / -0.31→0.83 | 4.98→0.65 / -0.06→0.97 |
| wifi_tight_s2 | 0.3354 → 0.0842 | 4.31 → 1.07 | -0.046 → 0.924 | 6.74→1.38 / -0.08→0.89 | 3.97→1.15 / -0.13→0.95 | 2.81→1.05 / -0.67→0.75 | 2.29→0.53 / 0.50→0.98 |

WiFi alone: the trained |CSI| reproduces the broad spectral shape of every receiver and antenna (corr 0.87, RMSE 1.3 dB), including Rx-2's deep fade at subcarrier ~35 on antenna 0 and the NLOS Rx-4 slope; residuals are within +-3 dB except Rx-1 antenna 1 above subcarrier 90. This is achieved with SCHEMATIC Tx/Rx positions (+-0.5 m), so the material field is absorbing geometry error: the fitted eps' (median 5.2) differs from the SFCW-alone (3.2) and UWB-alone (1.6) fields of the same scene, which is the cross-band caveat.


Fitted field (best tight seed) at 5.18 GHz: ε′ median 5.22 [p5 2.38, p95 8.91], σ median 0.160 [p5 0.047, p95 0.306] S/m; nearest ITU class: {'concrete': 16688, 'brick': 519, 'glass': 465, 'wood': 137, 'plasterboard': 1457, 'marble': 734}.

## Reproduce

```
source tools/env_node.sh && cd 4DPS
CUDA_VISIBLE_DEVICES=3 python -m mm4DPS.octonet_eval_panels fit --band sfcw --seeds 0,1,2 --steps 400 --voxels 64 --loose
CUDA_VISIBLE_DEVICES=4 python -m mm4DPS.octonet_eval_panels fit --band uwb  --seeds 0,1,2 --steps 400 --voxels 64 --loose
CUDA_VISIBLE_DEVICES=5 python -m mm4DPS.octonet_eval_panels fit --band wifi --seeds 0,1,2 --steps 400 --voxels 64 --loose
python -m mm4DPS.octonet_eval_panels figures
```

