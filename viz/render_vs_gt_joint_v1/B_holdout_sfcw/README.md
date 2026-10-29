# B_holdout_sfcw: UWB + WIFI train one per-point material; SFCW held out (zero-shot)

Model: per-point six-parameter 3DPS material (tight ITU bounds, L2 0.05 to the init), learned noise floor + coupling block (SFCW, UWB), 20 + 20 SFCW channel gains; joint loss = Σ_b L_b / L_b(iteration 0); Adam lr 0.02, 600 steps; 3 seeds. 'iter 0' = uniform ITU concrete with the iteration-0 gain/floor. Reference = the single-band v4 fits (`viz/render_vs_gt_v4`). Metrics on the masked, median-normalised dB quantities (comparable with v4).

## Training bands (mean ± std over seeds)

| band | loss iter 0 | loss trained, joint | loss trained, single-band v4 | corr trained, joint | corr trained, single-band v4 | metric |
|---|---|---|---|---|---|---|
| UWB | 0.641 ± 0.008 | **0.145 ± 0.003** | 0.140 ± 0.002 | **0.925 ± 0.001** | 0.929 ± 0.001 | envelope ≥ 1 m (ceiling 0.95) |
| WIFI | 0.327 ± 0.010 | **0.081 ± 0.003** | 0.081 ± 0.007 | **0.934 ± 0.008** | 0.935 ± 0.013 | |CSI| all receivers |

## Held-out band SFCW: zero-shot (T20 protocol: only SFCW's receiver nuisances fitted, material frozen)

| arm | material | loss | corr (pair image ≥ 0.9 m (ceiling 0.89)) |
|---|---|---|---|
| baseline | uniform ITU concrete | 0.4199 ± 0.0019 | **0.631 ± 0.004** |
| transfer | trained on UWB + WIFI | 0.4223 ± 0.0009 | **0.622 ± 0.002** |
| upper | SFCW's own single-band fit (v4) | 0.2917 ± 0.0023 | **0.788 ± 0.002** |

Transfer − baseline = -0.009 ± 0.002 = -6 % of (upper − baseline); gate (> 3 σ and ≥ 25 %): **FAIL**.

## UWB (training band)

| run | loss iter0 → trained | dB RMSE (masked) | corr all supervised | corr ≥ 1 m fine / 39 cm (ceiling 0.95 / 0.98) | corr 1–9 m fine / 39 cm | corr 0.3–4 m | corr 4–9.9 m |
|---|---|---|---|---|---|---|---|
| uwb_s0 | 0.6314 → 0.1477 | 7.41 → 2.62 | 0.694 → 0.961 | 0.434 → **0.924** / 0.995 | 0.931 / 0.995 | 0.532 → 0.948 | 0.485 → 0.924 |
| uwb_s1 | 0.6519 → 0.1414 | 7.51 → 2.56 | 0.674 → 0.962 | 0.379 → **0.926** / 0.995 | 0.933 / 0.996 | 0.456 → 0.951 | 0.466 → 0.925 |
| uwb_s2 | 0.6403 → 0.1473 | 7.41 → 2.59 | 0.688 → 0.961 | 0.416 → **0.925** / 0.997 | 0.932 / 0.997 | 0.500 → 0.947 | 0.482 → 0.927 |

Shared material at 7.0 GHz (best seed `uwb_s1`): ε′ median 5.22 [p5 3.18, p95 7.68], σ median 0.202 S/m; nearest ITU class {'concrete': 18898, 'brick': 59, 'glass': 257, 'wood': 60, 'plasterboard': 319, 'marble': 407}.

## WIFI (training band)

| run | loss iter0 → trained | dB RMSE all | corr all | Rx-1 RMSE / corr | Rx-2 | Rx-3 | Rx-4 (NLOS) |
|---|---|---|---|---|---|---|---|
| wifi_s0 | 0.3132 → 0.0775 | 3.49 → 0.93 | -0.132 → **0.944** | 4.79→1.18 / -0.16→0.92 | 3.63→0.96 / -0.14→0.97 | 2.07→0.94 / -0.23→0.84 | 2.91→0.53 / -0.31→0.98 |
| wifi_s1 | 0.3338 → 0.0850 | 3.88 → 1.08 | -0.032 → **0.924** | 3.92→1.02 / -0.30→0.95 | 3.73→0.93 / 0.26→0.97 | 2.49→0.93 / -0.31→0.84 | 4.98→1.37 / -0.06→0.88 |
| wifi_s2 | 0.3354 → 0.0792 | 4.31 → 1.01 | -0.046 → **0.933** | 6.74→1.37 / -0.08→0.89 | 3.97→1.09 / -0.13→0.96 | 2.81→0.91 / -0.67→0.85 | 2.29→0.50 / 0.50→0.98 |

Shared material at 5.18 GHz (best seed `wifi_s0`): ε′ median 5.18 [p5 2.62, p95 8.11], σ median 0.158 S/m; nearest ITU class {'concrete': 18181, 'brick': 57, 'glass': 317, 'wood': 190, 'plasterboard': 736, 'marble': 519}.

## SFCW (held out)

Rows: iter0 = the baseline arm (uniform concrete + fitted nuisances), trained = the transfer arm (two-band material + fitted nuisances).

| run | loss iter0 → trained | dB RMSE (pairs × bins) | image corr (all supervised bins) | image corr ≥ 0.9 m fine / 19 cm (ceiling 0.89 / 0.95) | per-pair corr (median) | pair-mean profile corr / RMSE dB | gain std tx/rx dB (GT) | gain corr tx / rx / pairs |
|---|---|---|---|---|---|---|---|---|
| sfcw_s0 | 0.4181 → 0.4210 | 5.38 → 5.43 | 0.833 → 0.830 | 0.635 → **0.625** / 0.763 | 0.813 → 0.813 | 0.981 → 0.980 / 1.40 → 1.43 | 2.07/3.84 (2.87/3.08) | 0.91 / 0.92 / 0.85 |
| sfcw_s1 | 0.4190 → 0.4232 | 5.39 → 5.45 | 0.832 → 0.828 | 0.632 → **0.621** / 0.759 | 0.813 → 0.812 | 0.981 → 0.980 / 1.40 → 1.44 | 1.63/3.25 (2.87/3.08) | 0.93 / 0.92 / 0.85 |
| sfcw_s2 | 0.4225 → 0.4227 | 5.43 → 5.45 | 0.830 → 0.828 | 0.625 → **0.620** / 0.758 | 0.813 → 0.813 | 0.980 → 0.980 / 1.43 → 1.45 | 1.74/2.89 (2.87/3.08) | 0.95 / 0.94 / 0.89 |

Shared material at 65.0 GHz (best seed `sfcw_s0`): ε′ median 5.18 [p5 2.62, p95 8.11], σ median 1.139 S/m; nearest ITU class {'concrete': 18181, 'brick': 57, 'glass': 317, 'wood': 190, 'plasterboard': 736, 'marble': 519}.

