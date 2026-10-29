# A_all3: all three bands train one per-point material

Model: per-point six-parameter 3DPS material (tight ITU bounds, L2 0.05 to the init), learned noise floor + coupling block (SFCW, UWB), 20 + 20 SFCW channel gains; joint loss = Σ_b L_b / L_b(iteration 0); Adam lr 0.02, 600 steps; 3 seeds. 'iter 0' = uniform ITU concrete with the iteration-0 gain/floor. Reference = the single-band v4 fits (`viz/render_vs_gt_v4`). Metrics on the masked, median-normalised dB quantities (comparable with v4).

## Training bands (mean ± std over seeds)

| band | loss iter 0 | loss trained, joint | loss trained, single-band v4 | corr trained, joint | corr trained, single-band v4 | metric |
|---|---|---|---|---|---|---|
| SFCW | 0.717 ± 0.008 | **0.348 ± 0.002** | 0.292 ± 0.002 | **0.731 ± 0.002** | 0.788 ± 0.002 | pair image ≥ 0.9 m (ceiling 0.89) |
| UWB | 0.641 ± 0.008 | **0.155 ± 0.003** | 0.140 ± 0.002 | **0.919 ± 0.002** | 0.929 ± 0.001 | envelope ≥ 1 m (ceiling 0.95) |
| WIFI | 0.327 ± 0.010 | **0.082 ± 0.004** | 0.081 ± 0.007 | **0.934 ± 0.010** | 0.935 ± 0.013 | |CSI| all receivers |

## SFCW (training band)

| run | loss iter0 → trained | dB RMSE (pairs × bins) | image corr (all supervised bins) | image corr ≥ 0.9 m fine / 19 cm (ceiling 0.89 / 0.95) | per-pair corr (median) | pair-mean profile corr / RMSE dB | gain std tx/rx dB (GT) | gain corr tx / rx / pairs |
|---|---|---|---|---|---|---|---|---|
| sfcw_s0 | 0.7085 → 0.3465 | 8.11 → 4.76 | 0.599 → 0.874 | 0.190 → **0.733** / 0.869 | 0.632 → 0.846 | 0.921 → 0.995 / 2.49 → 1.09 | 2.95/4.46 (2.87/3.08) | 0.97 / 0.94 / 0.90 |
| sfcw_s1 | 0.7153 → 0.3469 | 8.23 → 4.75 | 0.586 → 0.874 | 0.160 → **0.732** / 0.869 | 0.623 → 0.847 | 0.909 → 0.996 / 2.66 → 0.99 | 3.01/4.54 (2.87/3.08) | 0.97 / 0.95 / 0.91 |
| sfcw_s2 | 0.7267 → 0.3511 | 8.26 → 4.78 | 0.593 → 0.872 | 0.175 → **0.729** / 0.865 | 0.621 → 0.844 | 0.928 → 0.995 / 2.54 → 1.06 | 3.15/4.78 (2.87/3.08) | 0.97 / 0.94 / 0.90 |

Shared material at 65.0 GHz (best seed `sfcw_s0`): ε′ median 5.21 [p5 2.61, p95 8.24], σ median 1.059 S/m; nearest ITU class {'concrete': 17124, 'brick': 117, 'glass': 602, 'wood': 192, 'plasterboard': 1045, 'marble': 920}.

## UWB (training band)

| run | loss iter0 → trained | dB RMSE (masked) | corr all supervised | corr ≥ 1 m fine / 39 cm (ceiling 0.95 / 0.98) | corr 1–9 m fine / 39 cm | corr 0.3–4 m | corr 4–9.9 m |
|---|---|---|---|---|---|---|---|
| uwb_s0 | 0.6314 → 0.1583 | 7.41 → 2.74 | 0.694 → 0.957 | 0.434 → **0.917** / 0.994 | 0.923 / 0.994 | 0.532 → 0.939 | 0.485 → 0.920 |
| uwb_s1 | 0.6519 → 0.1546 | 7.51 → 2.68 | 0.674 → 0.958 | 0.379 → **0.918** / 0.994 | 0.925 / 0.995 | 0.456 → 0.939 | 0.466 → 0.921 |
| uwb_s2 | 0.6403 → 0.1521 | 7.41 → 2.66 | 0.688 → 0.959 | 0.416 → **0.921** / 0.996 | 0.928 / 0.996 | 0.500 → 0.940 | 0.482 → 0.924 |

Shared material at 7.0 GHz (best seed `uwb_s2`): ε′ median 5.25 [p5 2.89, p95 8.03], σ median 0.192 S/m; nearest ITU class {'concrete': 17864, 'brick': 109, 'glass': 464, 'wood': 79, 'plasterboard': 696, 'marble': 788}.

## WIFI (training band)

| run | loss iter0 → trained | dB RMSE all | corr all | Rx-1 RMSE / corr | Rx-2 | Rx-3 | Rx-4 (NLOS) |
|---|---|---|---|---|---|---|---|
| wifi_s0 | 0.3132 → 0.0766 | 3.49 → 0.91 | -0.132 → **0.947** | 4.79→1.14 / -0.16→0.93 | 3.63→0.93 / -0.14→0.97 | 2.07→0.93 / -0.23→0.84 | 2.91→0.51 / -0.31→0.98 |
| wifi_s1 | 0.3338 → 0.0837 | 3.88 → 1.06 | -0.032 → **0.928** | 3.92→1.02 / -0.30→0.95 | 3.73→0.82 / 0.26→0.97 | 2.49→0.90 / -0.31→0.84 | 4.98→1.40 / -0.06→0.87 |
| wifi_s2 | 0.3354 → 0.0853 | 4.31 → 1.06 | -0.046 → **0.926** | 6.74→1.43 / -0.08→0.88 | 3.97→1.09 / -0.13→0.95 | 2.81→1.00 / -0.67→0.82 | 2.29→0.50 / 0.50→0.98 |

Shared material at 5.18 GHz (best seed `wifi_s0`): ε′ median 5.21 [p5 2.61, p95 8.24], σ median 0.146 S/m; nearest ITU class {'concrete': 17124, 'brick': 117, 'glass': 602, 'wood': 192, 'plasterboard': 1045, 'marble': 920}.

