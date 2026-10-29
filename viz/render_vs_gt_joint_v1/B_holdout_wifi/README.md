# B_holdout_wifi: SFCW + UWB train one per-point material; WIFI held out (zero-shot)

Model: per-point six-parameter 3DPS material (tight ITU bounds, L2 0.05 to the init), learned noise floor + coupling block (SFCW, UWB), 20 + 20 SFCW channel gains; joint loss = Σ_b L_b / L_b(iteration 0); Adam lr 0.02, 600 steps; 3 seeds. 'iter 0' = uniform ITU concrete with the iteration-0 gain/floor. Reference = the single-band v4 fits (`viz/render_vs_gt_v4`). Metrics on the masked, median-normalised dB quantities (comparable with v4).

## Training bands (mean ± std over seeds)

| band | loss iter 0 | loss trained, joint | loss trained, single-band v4 | corr trained, joint | corr trained, single-band v4 | metric |
|---|---|---|---|---|---|---|
| SFCW | 0.717 ± 0.008 | **0.342 ± 0.002** | 0.292 ± 0.002 | **0.738 ± 0.001** | 0.788 ± 0.002 | pair image ≥ 0.9 m (ceiling 0.89) |
| UWB | 0.641 ± 0.008 | **0.152 ± 0.002** | 0.140 ± 0.002 | **0.921 ± 0.001** | 0.929 ± 0.001 | envelope ≥ 1 m (ceiling 0.95) |

## Held-out band WIFI: zero-shot (T20 protocol: only WIFI's receiver nuisances fitted, material frozen)

| arm | material | loss | corr (|CSI| all receivers) |
|---|---|---|---|
| baseline | uniform ITU concrete | 0.3275 ± 0.0101 | **-0.070 ± 0.044** |
| transfer | trained on SFCW + UWB | 0.3331 ± 0.0119 | **-0.136 ± 0.067** |
| upper | WIFI's own single-band fit (v4) | 0.0809 ± 0.0070 | **0.935 ± 0.013** |

Transfer − baseline = -0.065 ± 0.079 = -7 % of (upper − baseline); gate (> 3 σ and ≥ 25 %): **FAIL**.

## SFCW (training band)

| run | loss iter0 → trained | dB RMSE (pairs × bins) | image corr (all supervised bins) | image corr ≥ 0.9 m fine / 19 cm (ceiling 0.89 / 0.95) | per-pair corr (median) | pair-mean profile corr / RMSE dB | gain std tx/rx dB (GT) | gain corr tx / rx / pairs |
|---|---|---|---|---|---|---|---|---|
| sfcw_s0 | 0.7085 → 0.3403 | 8.11 → 4.70 | 0.599 → 0.877 | 0.190 → **0.739** / 0.874 | 0.632 → 0.850 | 0.921 → 0.996 / 2.49 → 1.04 | 2.92/4.24 (2.87/3.08) | 0.97 / 0.94 / 0.91 |
| sfcw_s1 | 0.7153 → 0.3399 | 8.23 → 4.69 | 0.586 → 0.877 | 0.160 → **0.739** / 0.874 | 0.623 → 0.849 | 0.909 → 0.997 / 2.66 → 0.96 | 3.02/4.38 (2.87/3.08) | 0.97 / 0.95 / 0.91 |
| sfcw_s2 | 0.7267 → 0.3446 | 8.26 → 4.72 | 0.593 → 0.875 | 0.175 → **0.736** / 0.870 | 0.621 → 0.850 | 0.928 → 0.995 / 2.54 → 1.03 | 3.12/4.47 (2.87/3.08) | 0.98 / 0.94 / 0.91 |

Shared material at 65.0 GHz (best seed `sfcw_s1`): ε′ median 4.89 [p5 2.98, p95 7.57], σ median 1.112 S/m; nearest ITU class {'concrete': 17311, 'brick': 167, 'glass': 658, 'wood': 27, 'plasterboard': 704, 'marble': 1133}.

## UWB (training band)

| run | loss iter0 → trained | dB RMSE (masked) | corr all supervised | corr ≥ 1 m fine / 39 cm (ceiling 0.95 / 0.98) | corr 1–9 m fine / 39 cm | corr 0.3–4 m | corr 4–9.9 m |
|---|---|---|---|---|---|---|---|
| uwb_s0 | 0.6314 → 0.1546 | 7.41 → 2.71 | 0.694 → 0.958 | 0.434 → **0.919** / 0.994 | 0.925 / 0.994 | 0.532 → 0.941 | 0.485 → 0.922 |
| uwb_s1 | 0.6519 → 0.1501 | 7.51 → 2.65 | 0.674 → 0.959 | 0.379 → **0.921** / 0.995 | 0.927 / 0.995 | 0.456 → 0.943 | 0.466 → 0.922 |
| uwb_s2 | 0.6403 → 0.1501 | 7.41 → 2.64 | 0.688 → 0.959 | 0.416 → **0.922** / 0.996 | 0.929 / 0.996 | 0.500 → 0.941 | 0.482 → 0.925 |

Shared material at 7.0 GHz (best seed `uwb_s1`): ε′ median 4.89 [p5 2.98, p95 7.57], σ median 0.195 S/m; nearest ITU class {'concrete': 17311, 'brick': 167, 'glass': 658, 'wood': 27, 'plasterboard': 704, 'marble': 1133}.

## WIFI (held out)

Rows: iter0 = the baseline arm (uniform concrete + fitted nuisances), trained = the transfer arm (two-band material + fitted nuisances).

| run | loss iter0 → trained | dB RMSE all | corr all | Rx-1 RMSE / corr | Rx-2 | Rx-3 | Rx-4 (NLOS) |
|---|---|---|---|---|---|---|---|
| wifi_s0 | 0.3132 → 0.3173 | 3.49 → 3.54 | -0.132 → **-0.127** | 4.79→4.87 / -0.16→-0.14 | 3.63→3.62 / -0.14→-0.16 | 2.07→2.15 / -0.23→-0.24 | 2.91→2.92 / -0.31→-0.32 |
| wifi_s1 | 0.3338 → 0.3359 | 3.88 → 3.89 | -0.032 → **-0.058** | 3.92→3.91 / -0.30→-0.34 | 3.73→3.63 / 0.26→0.28 | 2.49→2.68 / -0.31→-0.28 | 4.98→4.98 / -0.06→-0.11 |
| wifi_s2 | 0.3354 → 0.3459 | 4.31 → 4.27 | -0.046 → **-0.222** | 6.74→6.77 / -0.08→-0.49 | 3.97→3.95 / -0.13→-0.16 | 2.81→2.71 / -0.67→-0.64 | 2.29→2.05 / 0.50→0.63 |

Shared material at 5.18 GHz (best seed `wifi_s0`): ε′ median 4.96 [p5 3.04, p95 7.62], σ median 0.154 S/m; nearest ITU class {'concrete': 17392, 'brick': 161, 'glass': 622, 'wood': 18, 'plasterboard': 712, 'marble': 1095}.

