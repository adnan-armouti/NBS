# B_holdout_uwb: SFCW + WIFI train one per-point material; UWB held out (zero-shot)

Model: per-point six-parameter 3DPS material (tight ITU bounds, L2 0.05 to the init), learned noise floor + coupling block (SFCW, UWB), 20 + 20 SFCW channel gains; joint loss = Σ_b L_b / L_b(iteration 0); Adam lr 0.02, 600 steps; 3 seeds. 'iter 0' = uniform ITU concrete with the iteration-0 gain/floor. Reference = the single-band v4 fits (`viz/render_vs_gt_v4`). Metrics on the masked, median-normalised dB quantities (comparable with v4).

## Training bands (mean ± std over seeds)

| band | loss iter 0 | loss trained, joint | loss trained, single-band v4 | corr trained, joint | corr trained, single-band v4 | metric |
|---|---|---|---|---|---|---|
| SFCW | 0.717 ± 0.008 | **0.295 ± 0.002** | 0.292 ± 0.002 | **0.782 ± 0.000** | 0.788 ± 0.002 | pair image ≥ 0.9 m (ceiling 0.89) |
| WIFI | 0.327 ± 0.010 | **0.091 ± 0.015** | 0.081 ± 0.007 | **0.875 ± 0.078** | 0.935 ± 0.013 | |CSI| all receivers |

## Held-out band UWB: zero-shot (T20 protocol: only UWB's receiver nuisances fitted, material frozen)

| arm | material | loss | corr (envelope ≥ 1 m (ceiling 0.95)) |
|---|---|---|---|
| baseline | uniform ITU concrete | 0.4222 ± 0.0013 | **0.627 ± 0.002** |
| transfer | trained on SFCW + WIFI | 0.4225 ± 0.0010 | **0.624 ± 0.003** |
| upper | UWB's own single-band fit (v4) | 0.1408 ± 0.0017 | **0.930 ± 0.001** |

Transfer − baseline = -0.002 ± 0.001 = -1 % of (upper − baseline); gate (> 3 σ and ≥ 25 %): **FAIL**.

## SFCW (training band)

| run | loss iter0 → trained | dB RMSE (pairs × bins) | image corr (all supervised bins) | image corr ≥ 0.9 m fine / 19 cm (ceiling 0.89 / 0.95) | per-pair corr (median) | pair-mean profile corr / RMSE dB | gain std tx/rx dB (GT) | gain corr tx / rx / pairs |
|---|---|---|---|---|---|---|---|---|
| sfcw_s0 | 0.7085 → 0.2923 | 8.11 → 4.29 | 0.599 → 0.896 | 0.190 → **0.782** / 0.901 | 0.632 → 0.874 | 0.921 → 0.998 / 2.49 → 0.67 | 2.74/3.39 (2.87/3.08) | 0.98 / 0.96 / 0.93 |
| sfcw_s1 | 0.7153 → 0.2957 | 8.23 → 4.29 | 0.586 → 0.896 | 0.160 → **0.782** / 0.902 | 0.623 → 0.869 | 0.909 → 0.998 / 2.66 → 0.62 | 2.85/3.42 (2.87/3.08) | 0.98 / 0.95 / 0.93 |
| sfcw_s2 | 0.7267 → 0.2958 | 8.26 → 4.30 | 0.593 → 0.896 | 0.175 → **0.782** / 0.901 | 0.621 → 0.870 | 0.928 → 0.998 / 2.54 → 0.69 | 2.86/3.28 (2.87/3.08) | 0.98 / 0.96 / 0.93 |

Shared material at 65.0 GHz (best seed `sfcw_s0`): ε′ median 5.27 [p5 2.77, p95 8.15], σ median 1.103 S/m; nearest ITU class {'concrete': 18237, 'brick': 49, 'glass': 483, 'wood': 169, 'plasterboard': 629, 'marble': 433}.

## WIFI (training band)

| run | loss iter0 → trained | dB RMSE all | corr all | Rx-1 RMSE / corr | Rx-2 | Rx-3 | Rx-4 (NLOS) |
|---|---|---|---|---|---|---|---|
| wifi_s0 | 0.3132 → 0.0737 | 3.49 → 0.90 | -0.132 → **0.947** | 4.79→1.00 / -0.16→0.95 | 3.63→1.06 / -0.14→0.96 | 2.07→0.91 / -0.23→0.84 | 2.91→0.53 / -0.31→0.98 |
| wifi_s1 | 0.3338 → 0.0892 | 3.88 → 1.17 | -0.032 → **0.911** | 3.92→1.13 / -0.30→0.93 | 3.73→1.19 / 0.26→0.95 | 2.49→0.90 / -0.31→0.84 | 4.98→1.40 / -0.06→0.88 |
| wifi_s2 | 0.3354 → 0.1095 | 4.31 → 1.79 | -0.046 → **0.767** | 6.74→1.44 / -0.08→0.88 | 3.97→3.10 / -0.13→0.49 | 2.81→0.95 / -0.67→0.81 | 2.29→0.49 / 0.50→0.98 |

Shared material at 5.18 GHz (best seed `wifi_s0`): ε′ median 5.27 [p5 2.77, p95 8.15], σ median 0.152 S/m; nearest ITU class {'concrete': 18237, 'brick': 49, 'glass': 483, 'wood': 169, 'plasterboard': 629, 'marble': 433}.

## UWB (held out)

Rows: iter0 = the baseline arm (uniform concrete + fitted nuisances), trained = the transfer arm (two-band material + fitted nuisances).

| run | loss iter0 → trained | dB RMSE (masked) | corr all supervised | corr ≥ 1 m fine / 39 cm (ceiling 0.95 / 0.98) | corr 1–9 m fine / 39 cm | corr 0.3–4 m | corr 4–9.9 m |
|---|---|---|---|---|---|---|---|
| uwb_s0 | 0.4204 → 0.4212 | 5.22 → 5.24 | 0.826 → 0.824 | 0.624 → **0.620** / 0.892 | 0.619 / 0.891 | 0.741 → 0.731 | 0.643 → 0.646 |
| uwb_s1 | 0.4227 → 0.4230 | 5.20 → 5.20 | 0.827 → 0.827 | 0.628 → **0.628** / 0.895 | 0.628 / 0.895 | 0.744 → 0.746 | 0.647 → 0.645 |
| uwb_s2 | 0.4235 → 0.4234 | 5.22 → 5.22 | 0.826 → 0.826 | 0.628 → **0.625** / 0.893 | 0.625 / 0.892 | 0.744 → 0.740 | 0.646 → 0.646 |

Shared material at 7.0 GHz (best seed `uwb_s0`): ε′ median 5.27 [p5 2.77, p95 8.15], σ median 0.193 S/m; nearest ITU class {'concrete': 18237, 'brick': 49, 'glass': 483, 'wood': 169, 'plasterboard': 629, 'marble': 433}.

