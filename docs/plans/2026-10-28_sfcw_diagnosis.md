# SFCW diagnosis (2026-10-28): why the pair-image correlation is 0.27

Probes: `nbs/method/sfcw_diagnose{,2,3,4}.py` (JSON under `$NBS_DATA_OUT/sfcw_diagnose*.json`), figure
`viz/render_vs_gt_v2/sfcw_diagnosis.png`. Everything below is measured on the node on the calibrated Scene-1 scaffold
(40k of 74.6k points) and the pooled 5-recording target with the two-tone coupling removed and bins < 0.9 m masked.

## Verdict
The SFCW head (`4DPS/mm4DPS/octonet_heads.py::sfcw_head` → `nbs/method/sfcw_forward.py::render_sfcw_pairs`) is a
**coherent sum of isotropic Fresnel point scatterers** (amplitude Γ(θ_i)/(R_t·R_r), phase from the path, no area, no
scattering lobe, no antenna pattern), evaluated on a 5 cm point cloud at λ = 4.6 mm. That model is wrong for surfaces
at this wavelength, and three further defects stack on top. In order of size:

### 1. Surface physics / sampling (≈ 40 dB, the shape error) — ROOT CAUSE
Physical-optics coherent integration over 1 m² planar patches of concrete at 65 GHz (`sfcw_diagnose4`):

| patch | λ/4 sampling (correct) | 5 cm sampling (ours) |
|---|---|---|
| −X wall at 6.6 m, normal incidence | −36.4 dB | −18.2 dB |
| same wall, 5° off-normal | −49.3 dB | — |
| floor at 1.2 m, 65° incidence | −55.7 dB | +0.7 dB |
| floor at 2.5 m, 80° incidence | −74.8 dB | — |
| **wall / floor ratio** | **+19.3 dB** | **−18.9 dB** |

A grazing floor patch cancels itself through Fresnel-zone interference (−56 dB); a wall at normal incidence does not
(−36 dB, and 13 dB weaker already at 5° off-normal). The 5 cm point sum cannot cancel (points are 11 λ apart), so the
grazing floor comes out **56 dB too bright** and the specular wall 18 dB too weak. In the rendered profile this is
visible as: 71 % of the masked-band energy from points within 1.5 m of the node, 0.1 % from the −X wall, 1 % from the
60 % of scaffold points that lie beyond the 3.71 m unambiguous range. The GT has the opposite shape: it rises +6 dB
from 2.6 to 3.5 m in every one of the five recordings, exactly where the −X wall (6.4–6.9 m, perpendicular to node_1's
boresight) folds to (6.6 − 3.71 = 2.9 m). The iteration-0 pair-mean profile correlation is −0.33 and no choice of
element pattern (cos^p, −0.33…−0.70) or of the 3DPS Jones BSDF as currently callable (−0.47…−0.57, κ clamped at 10)
fixes it: the 1/R⁴ point sum with isotropic or κ ≤ 10 lobes cannot make a far specular wall beat a near grazing floor.
The material fit then "explains" the GT by dimming the near floor and brightening the far points (ρ 0.2 near, 1.6
far), which is why the per-point fit reached 0.59 and the ITU-bounded grid could not.

### 2. Objective: speckle-dominated gradients (the material cannot learn)
Gradient norms of the loss at iteration 0 (`sfcw_diagnose`, per-point medians):

| parameter | median |grad| | note |
|---|---|---|
| xyz (position) | 3.5e-2 | **2600× rho** |
| rho (reflectivity scale) | 1.3e-5 | |
| normals | 1.2e-5 | only through cos θ_i in Fresnel |
| abcd: a / b / c / d | 5e-7 / 1e-5 / 4e-6 / 7e-7 | a and d effectively dead: dispersion is not identifiable within 63–67 GHz |

The coherent log-magnitude of a 5 cm point sum is a speckle realisation: moving a point by 1 mm (80° of phase)
changes the bin's magnitude by dB, changing its material by 10 % does not. Under an **incoherent expected-power
objective** (Σ_i P_i |PSF(r−R_i)|², `sfcw_diagnose2`) the ratio xyz/rho drops from 2607 to **1.0** and the loss is
the same (0.63 vs 0.65): the material gradient becomes meaningful at no cost. The GT itself is a fixed speckle
realisation of the real scene which no 8-bit-depth scaffold can reproduce coherently (3.9 cm quantisation = 8 λ), so
the coherent per-pair image should not be the target at all.

### 3. GT per-channel gains (the pair-image ceiling)
The GT (pair × range) image has a per-pair level spread of 4.05 dB that is **fixed over frames** (pattern correlation
0.92) and **92 % explained by an additive per-Tx + per-Rx gain** (residual 1.17 dB; by-Tx 2.8 dB, by-Rx 2.7 dB). A
pair-blind render cannot correlate with that image above **0.45** (GT vs its own pair-mean); with the 39 gains known
the ceiling is 0.64. The raw phasors are uncalibrated in gain as well as in phase (the calibration lives in the board's
NVM, `pdf/VTRIG-74.pdf`). Fitting 20 + 19 gains jointly is a closed-form step.

### 4. Inputs
- Normals: floor fine (median 3.7° from vertical, 75 % within 20°); the −X wall poor (median 33° off, 37 % within 20°)
  because the 8-bit depth staircases at 6.6 m. Incidence seen by the radar: median cos θ_i 0.39, 35 % below 0.3.
- 60 % of the 40k points are beyond 3.71 m and fold into the window (true physics for a stepped-frequency sweep, but
  only with a correct specular model do they contribute correctly).
- Bins < 0.9 m are the node's own body (masked, T4); the antenna element pattern (82° H-plane) is absent.
- Not a cause: the coupling (removed, T4), the per-element phase ramp (magnitude only), the pair order (range only).

## Fix plan (SFCW head v2), in the order that removes the most error first
1. **Surface-aware scattering per point, summed incoherently** (fixes 1 and 2 together). Each scaffold point is a
   planar patch of area A (voxel²) with normal n. Its monostatic backscatter power is the physical-optics Fresnel-zone
   term, not an isotropic Γ: P_i = |Γ(θ_i)|² · G_spec(θ_i, A) · cos²θ_i · A / (λ² R⁴) with G_spec the PO pattern of a
   flat patch (sinc²(2 k √A sin θ_i)-type: 0 dB at normal incidence, −56 dB at 65° for 1 m², −30 dB for a 5 cm
   facet) blended with a rough-surface lobe (3DPS SPM/KA, σ_h, l_c) for the diffuse part. This is the 3DPS Jones BSDF
   with its specular lobe normalised to PO instead of the κ ≤ 10 clamp, which is what a 65 GHz room needs. Summed as
   expected power on the range grid (the incoherent objective of §2); a coherent output stays available for later.
   Acceptance: iteration-0 pair-mean profile correlation > 0 (currently −0.33) and the far rise reproduced within ~3 dB.
2. **Per-channel gains**: 20 Tx + 19 Rx gain parameters in the head (or closed-form from the GT medians each step),
   applied to the rendered pair image. Acceptance: pair-image correlation ceiling 0.45 → 0.64 realised.
3. **Element pattern**: datasheet 82° (H) × 180° (E) beamwidths as a cos^p (or measured-free) weight; boresight = the
   calibrated camera yaw until T5–T7 are unblocked.
4. **Normals**: planar-fit normals on the full-resolution depth (RANSAC per voxel neighbourhood) instead of PCA on the
   voxelised cloud, so wall normals land within 10°.
5. Keep: bins < 0.9 m masked (node body), coupling removal, pooled target, xcorr weights; ρ bounded or grid-stored.
6. Re-run T19 for SFCW only; the metric that must move is the pair-mean profile correlation (0.53 → > 0.8) and the
   per-pair image correlation toward its ceiling (0.27 → ~0.6 with gains).

What this does NOT fix: the ITU dispersion exponents remain unidentifiable from one band (gradients 1e-7), which is
the reason for the cross-band protocol; and sub-bin speckle in the GT, which is not a target.
