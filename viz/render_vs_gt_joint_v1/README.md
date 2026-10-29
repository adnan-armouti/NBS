# Joint training (render_vs_gt_joint_v1)

A: all three bands optimise one per-point scene material. B: two bands train, the third is held out and evaluated zero-shot (only its receiver nuisances fitted; arms baseline / transfer / upper). Model and protocol: `4DPS/mm4DPS/octonet_joint_panels.py`, `octonet_zero_shot.py`. Reference single-band numbers: `viz/render_vs_gt_v4/`.

| set | subfolder | training bands: loss trained (joint vs single v4) | held-out band: baseline → transfer → upper (corr) | gate |
|---|---|---|---|---|
| A_all3 | [A_all3/](A_all3/README.md) | SFCW 0.348 ± 0.002 (v4 0.292 ± 0.002); UWB 0.155 ± 0.003 (v4 0.140 ± 0.002); WIFI 0.082 ± 0.004 (v4 0.081 ± 0.007) |  |  |
| B_holdout_sfcw | [B_holdout_sfcw/](B_holdout_sfcw/README.md) | UWB 0.145 ± 0.003 (v4 0.140 ± 0.002); WIFI 0.081 ± 0.003 (v4 0.081 ± 0.007) | SFCW: 0.631 → **0.622** → 0.788 | FAIL (-6 % of the gap) |
| B_holdout_uwb | [B_holdout_uwb/](B_holdout_uwb/README.md) | SFCW 0.295 ± 0.002 (v4 0.292 ± 0.002); WIFI 0.091 ± 0.015 (v4 0.081 ± 0.007) | UWB: 0.627 → **0.624** → 0.930 | FAIL (-1 % of the gap) |
| B_holdout_wifi | [B_holdout_wifi/](B_holdout_wifi/README.md) | SFCW 0.342 ± 0.002 (v4 0.292 ± 0.002); UWB 0.152 ± 0.002 (v4 0.140 ± 0.002) | WIFI: -0.070 → **-0.136** → 0.935 | FAIL (-7 % of the gap) |
