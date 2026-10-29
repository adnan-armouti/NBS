# Single-modality training check (render_vs_gt_v3)

Each band is fitted ALONE (its own target only) with the voxel-grid ITU material field of `4DPS/mm4DPS/octonet_fit_multiband.py` (V = 64 voxels per axis over the calibrated scaffold, 20k scaffold points, Adam lr 0.03, TV 1e-3, up to 400 steps), under TIGHT bounds (a ∈ [1,15], b ∈ ±0.1, c ≤ 0.1 S/m, d ≤ 1.0) for 3 seeds, plus one LOOSE run (b ±0.2, c ≤ 2, d ≤ 1.5). 'iter 0' = the grid's initialisation = uniform ITU concrete. Losses are the bands' median-normalised log-magnitude L1 (SFCW: pair-wise range profiles, bins < 0.9 m masked as node near-field, xcorr-weighted; UWB: log-envelope, guard 0.29 m, reliability-weighted; WiFi: log|CSI| per receiver/antenna/subcarrier, 6 HT40 notches masked, magnitude only). Metrics are computed on the same masked, median-normalised dB quantities. Figures: `<band>_gt_iter0_trained.png`, `<band>_material.png`. Produced by `mm4DPS/octonet_eval_panels.py`.

## Summary (tight bounds, mean ± std over 3 seeds)

| band | loss iter 0 | loss trained (tight, 3 seeds) | loss trained (loose, 1 seed) | dB RMSE iter 0 → trained | corr iter 0 → trained | steps / runtime |
|---|---|---|---|---|---|---|
| SFCW | 0.881 ± 0.009 | **0.629 ± 0.004** | 0.628 | 9.72 ± 0.10 → **7.22 ± 0.06** | 0.120 ± 0.013 → **0.528 ± 0.005** | 400 ± 0 / 19 ± 0 s |
| UWB | 0.960 ± 0.013 | **0.574 ± 0.036** | 0.604 | 11.21 ± 0.45 → **8.00 ± 0.86** | 0.148 ± 0.087 → **0.459 ± 0.109** | 400 ± 0 / 5 ± 0 s |
| WIFI | 0.327 ± 0.010 | **0.080 ± 0.005** | 0.159 | 3.89 ± 0.33 → **0.96 ± 0.08** | -0.070 ± 0.044 → **0.940 ± 0.012** | 400 ± 0 / 5 ± 0 s |

## SFCW

Best tight seed: `sfcw_tight_s0` (loss 0.6249). Figures: `sfcw_gt_iter0_trained.png`, `sfcw_material.png`.

| run | gains | loss iter0 → trained | dB RMSE (pairs × bins) | image corr | per-pair corr (median) | pair-mean profile corr / RMSE dB | gain std tx/rx dB (GT) | gain corr tx / rx / pairs |
|---|---|---|---|---|---|---|---|---|
| sfcw_loose_s0 | yes | 0.8765 → 0.6277 | 9.66 → 7.22 | 0.137 → 0.528 | 0.173 → 0.315 | 0.659 → 0.930 / 2.90 → 1.11 | 2.96/2.82 (2.79/2.69) | 0.96 / 0.99 / 0.93 |
| sfcw_tight_s0 | yes | 0.8736 → 0.6249 | 9.66 → 7.17 | 0.137 → 0.533 | 0.173 → 0.338 | 0.659 → 0.937 / 2.90 → 1.06 | 2.90/2.95 (2.79/2.69) | 0.95 / 0.98 / 0.92 |
| sfcw_tight_s1 | yes | 0.8941 → 0.6347 | 9.86 → 7.31 | 0.105 → 0.522 | 0.148 → 0.308 | 0.595 → 0.927 / 3.13 → 1.17 | 3.18/2.98 (2.79/2.69) | 0.94 / 0.98 / 0.92 |
| sfcw_tight_s2 | yes | 0.8746 → 0.6262 | 9.65 → 7.18 | 0.119 → 0.530 | 0.149 → 0.304 | 0.653 → 0.914 / 2.77 → 1.24 | 2.99/2.83 (2.79/2.69) | 0.96 / 0.98 / 0.93 |

SFCW alone (T23 head = the 3DPS rasterizer; D13 per-channel gains fitted): the trained pair-mean profile follows the GT bin for bin over 0.9-3.5 m including the floor-ring dips and the rise beyond 2.6 m (the folded -X wall), profile corr 0.937, RMSE 1.06 dB. The per-pair image correlation is 0.533 (per-pair median 0.338); without gains a pair-blind render is capped at ~0.45 by the GT's fixed per-Tx/per-Rx gain banding (4 dB spread, 92 % additive). The fitted gains have std 2.90/2.95 dB (GT-derived 2.79/2.69) and correlate 0.95/0.98 with the GT's per-Tx/per-Rx pattern (0.92 on the 400-pair pattern). What remains is the per-pair ripple within each profile, which is below the scaffold's fidelity (8-bit depth, 3.9 cm quantisation at lambda 4.6 mm) and the uncalibrated per-element phases, so it is not a target. Loose bounds: see the table.


Fitted field (best tight seed) at 65.0 GHz: ε′ median 5.08 [p5 3.00, p95 8.47], σ median 0.209 [p5 0.025, p95 1.053] S/m; nearest ITU class: {'concrete': 5521, 'brick': 6611, 'glass': 675, 'wood': 15, 'plasterboard': 3041, 'marble': 4137}.

## UWB

Best tight seed: `uwb_tight_s2` (loss 0.5236). Figures: `uwb_gt_iter0_trained.png`, `uwb_material.png`.

| run | loss iter0 → trained | dB RMSE (masked) | corr all | corr 0.3–4 m | corr 4–9.9 m | RMSE dB 0.3–4 / 4–9.9 m |
|---|---|---|---|---|---|---|
| uwb_loose_s0 | 0.9611 → 0.6043 | 11.41 → 8.58 | 0.121 → 0.381 | -0.014 → 0.316 | 0.172 → 0.408 | 11.75 → 8.58 / 11.16 → 8.54 |
| uwb_tight_s0 | 0.9611 → 0.6015 | 11.41 → 8.80 | 0.121 → 0.364 | -0.014 → 0.299 | 0.172 → 0.388 | 11.75 → 8.72 / 11.16 → 8.82 |
| uwb_tight_s1 | 0.9752 → 0.5967 | 11.63 → 8.39 | 0.057 → 0.401 | -0.066 → 0.380 | 0.120 → 0.411 | 12.19 → 8.25 / 11.19 → 8.39 |
| uwb_tight_s2 | 0.9443 → 0.5236 | 10.58 → 6.81 | 0.265 → 0.611 | 0.094 → 0.541 | 0.361 → 0.647 | 11.76 → 6.96 / 9.72 → 6.67 |

UWB alone: the trained envelope tracks the GT mean level over the whole 0.3-9.9 m span (iter 0 was +15 dB too bright below 1.5 m and -20 dB too dark at 6-7 m), and the 6.3-6.8 m far-wall cluster appears at the right range. The GT's fine structure (20-30 dB interference fades every few cm) is not reproduced, so the correlation saturates near 0.5 and the RMSE at 6.7 dB: that structure needs the scaffold geometry to sub-cm accuracy and the UWB mount pose, both of which are assumed (camera yaw, node marker). The near range (0.3-1.5 m) keeps a +5-10 dB excess: node body returns absent from the scaffold.


Fitted field (best tight seed) at 7.0 GHz: ε′ median 4.73 [p5 2.08, p95 8.32], σ median 0.034 [p5 0.003, p95 0.171] S/m; nearest ITU class: {'concrete': 2426, 'brick': 10490, 'glass': 411, 'wood': 455, 'plasterboard': 3343, 'marble': 2875}.

## WIFI

Best tight seed: `wifi_tight_s1` (loss 0.0731). Figures: `wifi_gt_iter0_trained.png`, `wifi_material.png`.

| run | loss iter0 → trained | dB RMSE all | corr all | Rx-1 RMSE / corr | Rx-2 | Rx-3 | Rx-4 (NLOS) |
|---|---|---|---|---|---|---|---|
| wifi_loose_s0 | 0.3132 → 0.1587 | 3.49 → 2.73 | -0.132 → 0.622 | 4.79→1.27 / -0.16→0.91 | 3.63→5.21 / -0.14→0.41 | 2.07→0.90 / -0.23→0.84 | 2.91→0.52 / -0.31→0.98 |
| wifi_tight_s0 | 0.3132 → 0.0822 | 3.49 → 0.92 | -0.132 → 0.946 | 4.79→1.09 / -0.16→0.93 | 3.63→1.11 / -0.14→0.96 | 2.07→0.85 / -0.23→0.85 | 2.91→0.48 / -0.31→0.98 |
| wifi_tight_s1 | 0.3338 → 0.0731 | 3.88 → 0.88 | -0.032 → 0.950 | 3.92→1.04 / -0.30→0.94 | 3.73→0.93 / 0.26→0.97 | 2.49→0.92 / -0.31→0.84 | 4.98→0.53 / -0.06→0.98 |
| wifi_tight_s2 | 0.3354 → 0.0840 | 4.31 → 1.07 | -0.046 → 0.924 | 6.74→1.37 / -0.08→0.89 | 3.97→1.15 / -0.13→0.95 | 2.81→1.06 / -0.67→0.74 | 2.29→0.53 / 0.50→0.98 |

WiFi alone: the trained |CSI| reproduces the broad spectral shape of every receiver and antenna (corr 0.87, RMSE 1.3 dB), including Rx-2's deep fade at subcarrier ~35 on antenna 0 and the NLOS Rx-4 slope; residuals are within +-3 dB except Rx-1 antenna 1 above subcarrier 90. This is achieved with SCHEMATIC Tx/Rx positions (+-0.5 m), so the material field is absorbing geometry error: the fitted eps' (median 5.2) differs from the SFCW-alone (3.2) and UWB-alone (1.6) fields of the same scene, which is the cross-band caveat.


Fitted field (best tight seed) at 5.18 GHz: ε′ median 5.11 [p5 2.52, p95 8.91], σ median 0.165 [p5 0.056, p95 0.314] S/m; nearest ITU class: {'concrete': 16930, 'brick': 310, 'glass': 510, 'wood': 115, 'plasterboard': 1453, 'marble': 682}.

## Reproduce

```
source tools/env_node.sh && cd 4DPS
CUDA_VISIBLE_DEVICES=3 python -m mm4DPS.octonet_eval_panels fit --band sfcw --seeds 0,1,2 --steps 400 --voxels 64 --loose
CUDA_VISIBLE_DEVICES=4 python -m mm4DPS.octonet_eval_panels fit --band uwb  --seeds 0,1,2 --steps 400 --voxels 64 --loose
CUDA_VISIBLE_DEVICES=5 python -m mm4DPS.octonet_eval_panels fit --band wifi --seeds 0,1,2 --steps 400 --voxels 64 --loose
python -m mm4DPS.octonet_eval_panels figures
```

