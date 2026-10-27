# NBS — consolidated results for paper writing (honest status)

> Single reference for paper writing. Each result is tagged **[VALIDATED] / [PARTIAL] /
> [RETRACTED] / [BLOCKED]**. Updated Jun 2026 after the method-experiment audit
> ([md/method_audit.md](md/method_audit.md)). Do not cite anything tagged RETRACTED.

---

## 0. One-line thesis & positioning
**Thesis.** Treat the RF-frequency axis as synthesizable for *sensing*: train on bands, render
a held-out band's **raw, phase-coherent radar signal** (complex ADC/RA, supervised at the
magnitude `|RA|` level) by inverse-rendering band-conditioned scene materials through a
differentiable physics radar forward model, and validate on a downstream sensing task.

**Competitors (all 3DGS/NeRF RF radiance fields):** Wideband3DGS (arXiv 2505.20714, sim, PAS),
XFreq-GS (arXiv 2605.11432, sim, PAS), **XBandGS** (MobiCom'26 submission — physics-guided MoE,
**real USRP data**, PAS). All output **power / PAS**, scored by image similarity, framed as
channel modeling. None emit a raw phase-coherent signal; none have a downstream sensing task.

**Headline differentiator (the one to lead with):** **raw-signal output + a downstream-sensing
payoff, vs the PAS/power field of all prior cross-band work.** Real-data and "physics-grounded"
are NOT differentiators (XBandGS has both). Full memo: [md/positioning.md](md/positioning.md).

**Crisp novelty sentence.** *Unlike Wideband3DGS/XFreq-GS/XBandGS, which fit a power-level RF
radiance field (PAS) and interpolate it across frequency, NBS synthesizes the held-out band's
raw radar signal by inverse-rendering band-conditioned scene materials through a differentiable
physics radar forward model, and validates it on a downstream sensing task.*

---

## 1. Honest status table

| Item | Status | One-line |
|---|---|---|
| Positioning vs 3 competitors | **[VALIDATED]** | read the papers; raw-signal + downstream is the white space |
| Stage 1 — geometry init + rig calibration | **[VALIDATED]** | per-scene scaffolds, rigs match the rig photo |
| Stage 2 — HAR upper bounds | **[VALIDATED]** | mmWave 87.5% (within-dist), the payoff ceiling |
| G-ITU mechanism (differentiable dispersion recovery) | **[VALIDATED]** | the inverse loop converges through the real BSDF |
| Stage 5 — material recovery + held-band synthesis on real geometry | **[PARTIAL]** | beats a wrong-material floor by 17 dB; corr & interp-comparison invalid |
| "Physics beats interpolation" at the **roughness regime change** | **[VALIDATED]** (re-run, proper controls) | NBS 100% material-ID, −30.8 dB (25 dB below floor); validated interp fails (+0.3 dB). `method/eval_v2.py` |
| "Physics beats interpolation" in general (similar materials) | **[RETRACTED]** | only holds where materials matter (regime change); not universal |
| mmWave↔WiFi band translation | **[RETRACTED]** | did not beat the wrong-material floor |
| All RA/CSI **correlation** numbers | **[RETRACTED]** | geometry-dominated (wrong materials score 0.99+) |
| Two-metric eval harness (floor + material-ID) | **[VALIDATED]** | `method/eval_v2.py`; the correct way to score cross-band |
| RGB-derived material GT (eval-only) | **[VALIDATED]** | `preprocessing/material_labels.py`: seg → ITU material → scaffold |
| Raw vs normalized RA for material recovery | **[VALIDATED]** | equivalent (rel. amplitudes carry material; XRF55 RA is raw-scale) |
| Forward model reproduces REAL XRF55 RA | **[VALIDATED]** | **corr 0.42→0.76 after chirp-config calibration** (range scale/offset; XRF55 config unpublished); in prior-radar-renderer range. Residual = monocular geometry + single-bounce multipath. `method/real_inverse.py` |
| Cross-modality mmWave↔WiFi (sim, proper metrics) | **[PARTIAL]** | mmW→WiFi: matID 0.67, −20 dB (beats −8.1 floor). WiFi→mmW: matID 1.0 but **roughness unobservable from WiFi** → mmWave prediction fails. **Observability asymmetry.** `method/translate_v2.py` |
| WiFi geometry (XRF55) | **[VALIDATED]** | 1 Tx + 3 Rx×3ant, 5.64 GHz, 3.1m square; reconstructed Tx is 0.16 m from our calibrated camera (cross-validates both). `xrf55_config.wifi_layout()` |
| Multibounce ray-tracer (image method), differentiable + verified | **[VALIDATED]** | `method/multibounce{,_torch}.py`: renders WiFi multipath; **matches mmIR `image_method.py` to 8.9e−15 m**, autograd vs FD 6e−9. ~tens of paths/link, fast. The fast specular core of mmIR's MC tracer; ready for CUDA port. |
| Camera-pose leveling (floor → vertical) | **[VALIDATED]** | Camera had 7–11° unestimated pitch/roll (RA objective is horizontal). Estimate from floor, fold into rig (`level_R`); tilts →<2.5° all scenes. |
| Real WiFi rendering from the scaffold | **[VALIDATED]** | multibounce + leveling + **room-box geometry calibration**: corr ≈0 → **per-link 0.973** (full 0.52). `method/wifi_geom_calib.py` |
| Real WiFi material recovery | **[NOT SENSITIVE]** | Even at per-link 0.97, materials don't change the fit (wrong-material floor 0.513 ≈ recovered 0.505): WiFi is geometry/LOS-dominated, material-insensitive (the observability asymmetry). |
| Real mmWave↔WiFi translation | **[IN PROGRESS]** | Reframed: the **shared geometry** is the band-transferable content (renders both modalities); recover it from one, render the other. WiFi now renderable + geometry-calibrated. |
| Real-data material recovery vs RGB labels | **[BLOCKED]** | geometry fit 0.76 not yet enough for reliable single-band material recovery; needs geometry/multipath closure |
| Downstream payoff on REAL multi-band raw data | **[BLOCKED]** | needs R1 capture; no dataset closes it |

---

## 2. Datasets & Stage 1 (geometry, calibration) — [VALIDATED]
- **Competitive data audit:** XRF55 mmWave is processed magnitude `(17 frames, 256 range, 128 =
  [Doppler 0:64 | azimuth 64:128])` → **RA = `A[frame,:,64:128]`** (256×64). WiFi CSI `(270,1000)`
  amplitude-only (= 3 tx × 3 rx × 30 subcarrier). Videos RGB-only. MM-Fi not on disk; PanoRadar +
  ColoRadarPlus are on disk (raw complex). See `data/XRF55/AUDIT.md`.
- **Calibration (cross-sensor, non-co-located camera/radar):** monocular metric depth
  (Depth-Anything-V2-Metric) → pseudo-RA → align to the measured radar RA. Two-stage: per-scene
  **rig** (recovered yaw 62–74°, x −1.6 to −1.75 m, y 0.75–0.80 m, z −0.36 m across all 4 scenes,
  matching the rig photo and consistent across scenes = independent validation) + per-clip depth
  scale. Recovered radar range-res ≈ 0.076 m/bin. `data/XRF55/CALIBRATION.md`.
- **Moving human handled:** geometry is a per-SCENE static scaffold (human masked out via 2D pose);
  the human is per-clip dynamic content. Scaffolds: Scene1 124k pts, Scene2 264k, Scene3 197k,
  Scene4 196k. Manifests index all clips with a frozen subject-disjoint split. `data/XRF55/STAGE1_OUTPUTS.md`.

## 3. Stage 2 — downstream HAR upper bounds (the payoff ceiling) — [VALIDATED]
55-class HAR, chance 1.8%. Top-1: **mmWave 0.859 across all 4 rooms** (within-distribution, 34k
train / 8.5k test) and **0.875 on room 1**; 0.441 cross-subject; WiFi 0.038 cross-subject (≈chance)
/ 0.446 within-dist; RFID 0.061 / 0.207.
**Key finding:** mmWave generalizes cross-subject; WiFi/RFID are classifiable only
within-distribution (amplitude CSI/RFID is subject-specific). → the cross-band payoff should use
the within-distribution split; **mmWave is the strongest payoff vehicle.** Harness
`eval/downstream/` supports the real↔synth input swap. `data/XRF55/STAGE2_HAR_BASELINES.md`.

## 4. Method — forward/inverse model — [VALIDATED mechanism]
- Differentiable, frequency-conditioned: `bsdf_torch.set_frequency` makes the carrier a per-eval
  input to the REAL ITU-R P.2040 slab Fresnel; ITU dispersion `eps_r=a·f^b`, `sigma=c·f^d` made
  grad-enabled (`method/freq_material.py`) — the in-repo `itu_materials.py` was a non-diff lookup.
- **Single-chip (XRF55) RA pipeline:** `ra_utils.adc_to_ra_complex_single_chip` — azimuth ULA =
  Tx{0,2}×4Rx = 8 antennas at 0.5λ (Tx1 elevated, excluded), 64 azimuth bins (matches XRF55).
- The inverse loop (recover materials by back-prop through the forward model) **converges** on the
  real BSDF with multi-antenna `|RA|` supervision + noise. `method/ra_forward.py`, `method/gitu_gate*.py`.

## 5. Stage 5 — held-band synthesis on real room geometry — [PARTIAL]
On the real Scene1 scaffold (2000 pts, 4 material classes), recover per-class materials from train
{24,40,60} GHz, synthesize held-out **90 GHz** RA. **NBS −37.5 dB |RA| error, ~17 dB below the
wrong-material floor (−20 dB)** → genuine material recovery + synthesis on real geometry.
**Caveats:** corr 1.0 was geometry-dominated (invalid); the interp baseline (−5.9 dB) was degenerate
(unfair comparison). Track A (real geometry, controlled materials/signal). `md/stage5_result.md`.

## 5b. Corrected re-run — physics vs interpolation, scoped — [VALIDATED]
`method/eval_v2.py` (post-audit, two metrics + validated baseline). Roughness regime change
(train smooth {18,28,40} GHz, held 120 GHz, `4π·σ_h/λ` up to 4.0), RF-only library-assignment
recovery (reparam bounds as the only material prior; materials NOT given to training):
- **Material contribution:** wrong permittivity floor −21.9 dB; **wrong roughness floor −5.9 dB**
  → at the held band the dominant material effect is the frequency-dependent ROUGHNESS rolloff.
- **NBS: −30.8 dB, material-ID accuracy 1.00** (beats floor by 25 dB). **Interp (validated
  non-degenerate): +0.3 dB — fails** (cannot extrapolate the nonlinear rolloff).
- **Raw vs normalized: equivalent** (−30.8 vs −30.2, both ID 1.00). Material info is in the
  *relative* amplitudes (survive normalization); the global scale is unknowable in real data.
- **Real-data material GT (`preprocessing/material_labels.py`, eval-only):** ADE20K semantic seg
  on the XRF55 RGB frame → ITU material → projected to radar-frame points via the Stage-1
  calibration (Scene1 clip 01_01_01: 54k labelled pts, wood floor / concrete walls / glass windows).
- **Honest claim to make in the paper:** physics-material recovery beats field interpolation
  *where the cross-band response is nonlinear* (roughness/penetration regime changes), with
  material-ID validated against vision; it is competitive-not-superior where materials barely
  matter. Secondary to the raw-signal + downstream-sensing headline.

## 6. The audit — what we learned about evaluation — [VALIDATED methodology lesson]
- **RA/CSI correlation is geometry-dominated** in the cross-band setting (geometry fixed across
  bands → peak locations fixed → wrong materials score corr 0.99+). Unlike NVS, where the viewpoint
  changes and correlation is meaningful. Single-chip's broad azimuth makes it worse but is not the
  root cause. **Never report RA correlation as material/cross-band fidelity.**
- **Use two metrics:** (1) dB error **vs a wrong-material floor**; (2) **material-recovery accuracy**
  vs RGB-derived material labels (off-the-shelf segmentation → ITU material → projected to scaffold
  via the Stage-1 calibration; **eval-only**, training stays RF-only with reparam + library priors).
- **Validate baselines are non-degenerate** before any comparison.
- **Quantify the material contribution** to the cross-band difference — if small, the problem is
  mostly trivial geometry re-rendering (a real risk for single-chip + similar materials).
Full: [md/method_audit.md](md/method_audit.md), memory `nbs-eval-methodology`.

## 7. Open risks / blockers
- **R1 (the data gap, [BLOCKED]):** the headline downstream payoff needs REAL multi-band raw data
  + a task. XRF55 is single-band + processed; mmWave↔WiFi is cross-modality. Needs a single-radar
  multi-carrier capture (hardware), or PanoRadar/ColoRadarPlus for within-radar.
- **Is the problem hard?** Must quantify material vs geometry contribution to the cross-band signal.
- **Single-chip vs cascade:** the material/physics claim may be better shown on cascade radar
  (ColoRadar/PanoRadar, sharp azimuth) than single-chip XRF55.
- **Real-data fidelity:** single-bounce model vs real multipath/hardware/human.
- **Award language stays conditional** on R1 (per the area-chair committee).

## 7a. Key physics finding — WiFi CSI is material-INSENSITIVE
On real XRF55 data, even after the WiFi render reaches per-link corr **0.973** (multibounce +
camera leveling + room-box geometry calibration), **recovering materials does not change the fit**:
the wrong-material floor (all-concrete) corr **0.513** ≈ default **0.518** ≈ recovered **0.505**.
WiFi at 5.64 GHz indoors is **geometry/line-of-sight-dominated**; the material (permittivity)
contributes only ~8% of the signal, below the residual multipath error. This is the same
**observability asymmetry** seen in simulation: the material-sensitive modality is **mmWave**
(roughness rolloff at 60 GHz), not WiFi. **Consequence for the cross-band claim:** the
band-transferable content is the **shared room GEOMETRY** (it renders both modalities); material
is a small, band-specific correction. Recover the geometry(+materials) from the material-sensitive
band, transfer the geometry, add the small material correction. (`method/real_wifi_material.py`,
obstacle #13.)

## 7a-2. Shared-geometry cross-band synthesis (real, 4 scenes) — `method/cross_modality.py`
| scene | zero-shot (non-WiFi geom → WiFi, per-link) | oracle (WiFi-calibrated geom) | Δcorr right-vs-wrong materials |
|---|---|---|---|
| Scene1 | 0.445 | 0.973 | 0.007 |
| Scene2 | 0.115 | 0.940 | 0.008 |
| Scene3 | −0.180 | 0.905 | 0.025 |
| Scene4 | −0.275 | 0.804 | 0.001 |
| **mean** | **0.026** | **0.906** | **0.010** |

**Findings:** (1) **WiFi material-insensitivity holds across all scenes** (Δ ≤ 0.025): the forward
model's material term does not move the WiFi fit. (2) **The WiFi forward model is validated** —
oracle per-link **0.906** once the room scale is set. (3) **Zero-shot geometry transfer is limited
by source-band reconstruction completeness** (mean 0.026): the monocular+mmWave scaffold captures
the near-field scene but not the full room envelope (extents disagree 0.3–2.2 m), and WiFi multipath
is acutely room-scale-sensitive. **This is the observability asymmetry at the geometry level:
cross-band transfer is bounded by what the source band reconstructs** (scaffold sees the scene; WiFi
needs the walls). The calibration that closes it (room scale) is a handful of global params — so the
honest unlock is a full-room geometry source (Azure Kinect depth, multi-view, or room-envelope priors),
not a better forward model. [VALIDATED forward model + material-insensitivity; geometry transfer BLOCKED on full-room reconstruction]

## 7a-3. Radar range resolution — determined from data (not a free fit)
dR = c/(2B) is set by the chirp bandwidth B (unpublished; band 60-64 GHz so B≤4 GHz ⇒ dR≥0.0375 m).
**Data-driven determination** (`preprocessing/determine_range_res_direct.py`): fix pose to the
calibrated rig and trust the **metric** monocular depth as absolute scale (a=1); the depth-vs-radar
range-profile match peaks cleanly at **dR = 0.0675 m/bin** across 4 clips (per-clip 0.065–0.080,
corr 0.73). ⇒ **B ≈ 2.2 GHz**, max range 17.3 m, **margin ±6.8 cm/bin**. The physical floor 0.0375 m
(B=4 GHz) is firmly excluded (corr 0.25 there). This **replaces the old fit 0.076** (≈11% high — not
the 2× I had worried about; the metric depth was trustworthy). Scaffolds built with 0.076 are ~11%
too deep → rebuild when convenient. Exact B awaits the authors' chirp cfg.

## 7a-4. Real mmWave<->WiFi transfer, BOTH directions (run 2026-06-09)
| direction | transfer | result |
|---|---|---|
| mmWave → WiFi | scaffold geometry → render WiFi | zero-shot per-link **0.026** (oracle 0.91 if geom recalibrated to WiFi) |
| WiFi → mmWave | WiFi room box → render mmWave RA | dB-corr **0.08–0.19** |

**Both directions are weak, and the cause is structural, not a tooling gap:** the two bands are
dominated by **disjoint scene content**. mmWave RA images *near-field content* (furniture, subject)
that single-view depth reconstructs but WiFi never observes; WiFi CSI is *room-envelope multipath*
off walls that the camera frustum (mono OR Kinect — both single-view) never fully captures. Neither
band alone reconstructs the FULL scene, so neither predicts the other. Root cause: **XRF55 has no
two carriers of the SAME modality** — NBS's interpolation premise (K bands of one modality → held-out
band) has no real instance here; mmWave and WiFi are different *modalities*, not nearby *bands*.
**Consequence for the paper:** real data VALIDATES the forward models (mmWave 0.76, WiFi oracle 0.91)
and CHARACTERIZES the observability limit; the cross-band *synthesis* claim is carried by SIM
(multi-carrier: G-ITU gate, roughness regime, material-ID — [[nbs-gitu-gate-passed]]).

## 7b. Obstacles faced (and how we cleared them)
Each obstacle, its reframe, and the solution. Obstacles cleared are part of the contribution.

| # | Obstacle | Reframe + solution | Status |
|---|---|---|---|
| 1 | Concurrent competitors (XBandGS/XFreq-GS/Wideband3DGS) own cross-band PAS | Differentiate on **raw-signal output + downstream sensing**, not real-data/physics (they have those) | Cleared (positioning) |
| 2 | XRF55 mmWave is processed magnitude, single-band | Use it as the **downstream-task** vehicle; `\|RA\|` still encodes relative inter-antenna phase | Cleared |
| 3 | Moving human corrupts the static geometry | **Per-scene static scaffold** with the human masked via 2D pose; human = dynamic content | Cleared |
| 4 | RA correlation is geometry-dominated (wrong materials score 0.99) | Add a **material-ID metric** + **dB vs wrong-material floor**; RGB-seg labels for material GT (eval-only) | Cleared |
| 5 | Interp/copy baselines silently degenerated | **Assert baselines non-degenerate** before any comparison | Cleared |
| 6 | Physics-beats-interpolation not universal | **Scope** it to where the response is nonlinear (roughness regime change); demonstrated 100% material-ID, 25 dB below floor | Cleared |
| 7 | Sim-to-real RA gap (corr 0.47) | It was **calibration**: XRF55 chirp config unpublished → range axis mis-scaled. Recover a 4-scalar calibration → **0.76** | Cleared (mmWave) |
| 8 | WiFi Tx/Rx geometry unknown | **Reconstruct** from the paper + rig; cross-validates (Tx 0.16 m from calibrated camera) | Cleared |
| 9 | **WiFi CSI is multipath-dominated; 3DPS single-bounce renders corr ≈ 0 (selectivity ~0)** | **Added a lightweight image-method multibounce ray-tracer** (`method/multibounce.py`): fit room planes → LOS + 1st/2nd-order specular images → sum paths. ~11 paths/link, fast, differentiable in plane materials. mmIR has a Monte-Carlo version; this is the fast specular core. **Result: reproduces the multipath signature (selectivity 0.28 = real).** | **CLEARED** (multipath rendered) |
| 11 | Exact WiFi channel still corr ≈ 0 with tilted geometry | **The room appeared tilted because the CAMERA POSE was tilted** (~7–11° unestimated pitch/roll the horizontal RA objective can't see). **Estimate it from the floor plane and fold into the rig** (`ra_geometry.estimate_level`, stored as `level_R`; integrated into `prep_scene`). Floor tilt 10.7°→2.5° (all 4 scenes 7–10°→<2°). **Result: WiFi render corr ≈0 → per-link 0.45 (full 0.21), selectivity 0.29≈0.28.** | **CLEARED** — WiFi now renderable |
| 11b | Residual WiFi fit (per-link 0.45) | **The room box was mis-sized** (monocular scaffold doesn't reach the true walls → wrong path lengths). **Calibrate the box to the WiFi CSI** (`method/wifi_geom_calib.py`, derivative-free, the WiFi analog of the mmWave chirp calibration). **Result: per-link 0.45 → 0.973** (full 0.21→0.52). | **CLEARED** |
| 11c | Differentiable multibounce + correctness | Built torch version (`method/multibounce_torch.py`); **verified against mmIR `image_method.py`: specular point matches to 8.9e−15 m over 2000 scenes; gradient autograd-vs-FD 6e−9.** | **VERIFIED** |
| 11d | Keep 3DPS's speed advantage over mmIR's MC tracer | **Vectorized GPU render + native CUDA kernel** (`method/cuda_multibounce.py`, same cpp_extension/sm_89 toolchain as 3DPS). Both match the reference to **1.8e−9**; **~500× faster** than the Python-loop tracer (0.33 ms vs 164 ms on 64 links×12 planes). | **CLEARED** |
| 11e | Fully-fused differentiable CUDA training | Added the **CUDA backward kernel** (analytic Fresnel derivative) wrapped in `autograd.Function`. **Gradient matches torch-autograd to 1.3e−13, finite-diff to 2.5e−9**; fused fwd+bwd **2.5× faster** per step (1.0 vs 2.5 ms). End-to-end material recovery can run entirely on the fused kernel. | **CLEARED** |
| 13 | Real WiFi material recovery: even at per-link 0.97, materials don't change the fit (wrong-material floor 0.513 ≈ recovered 0.505) | **WiFi CSI is material-INSENSITIVE** (geometry/LOS-dominated; materials ~8% of signal, below residual multipath). Same **observability asymmetry**: the material-sensitive modality is mmWave (roughness at 60 GHz), not WiFi. → For transfer, the **shared GEOMETRY is the band-transferable content** (it renders both); material is a small band-specific correction. To expose WiFi material sensitivity: finer geometry (Kinect depth + furniture). | Honest physics; informs the claim |
| 12 | XRF55 mmWave single-band (no 2nd carrier) | mmWave↔WiFi via a **shared scene model**: recover geometry(+materials) from one modality, render the other. Geometry transfers (renders both); WiFi multipath now renderable (multibounce) and geometry-calibratable (per-link 0.97). | **Unblocked** (geometry-led) |
| 14 | Zero-shot geometry transfer fails (mean per-link 0.026): monocular+mmWave scaffold misses the room envelope; WiFi is acutely room-scale-sensitive | **Forward model is validated** (oracle 0.906); the gap is **source-band reconstruction completeness**, a few global room-scale params. Constructive unlock: full-room geometry — **Azure Kinect depth** (exists in XRF55), multi-view fusion, or a room-envelope prior that extends the scaffold to the walls. Cross-band transfer is bounded by what the source band reconstructs (observability asymmetry at the geometry level). | Forward model CLEARED; transfer BLOCKED on full-room geometry |

| 15 | **XRF55 retired**: mmWave RA corrupted by metal near the radar, no ADC/processing code, authors unresponsive | Migrated to **OctoNet** (NeurIPS 2025) + its **Vayyar IMAGEVK-74 SFCW** radar, which ships raw complex Tx/Rx-pair phasors — exactly the inverse-rendering input. 921 GB downloaded/extracted to the datasets SSD. | **CLEARED** |
| 16 | 3DPS renderer is FMCW/TI-specific | **Additive SFCW mode** gated by two orthogonal config axes: `modulation_scheme: FMCW\|SFCW` × `geometry_mode: monostatic\|bistatic` (monostatic = the fast co-located approximation that makes 3DPS ≫ mmIR; bistatic reserved for WiFi). Hofmann Eq.10 with discrete `f_c`; range = IFFT over freq steps; Hann PSF unchanged. `method/sfcw_migration_selftest.py` **21/21 PASS**, FMCW path bit-for-bit preserved; G-ITU dispersion gate also passes under SFCW (corr 1.000). | **CLEARED** |
| 17 | Published/assumed OctoNet parameters are **wrong** | Measured from the real files: Vayyar **3.08–3.12 Hz** (HF card says ~10 fps), MoCap **125 Hz** (not 120), **millimetres** (not m), **Z-up floor-referenced** (not Y-up). MoCap `.npy` need a numpy 2→1 `_core` shim. `cut_manual.csv` is a master join table (974 clips with both vayyar+mocap). All folded into `preprocessing/octonet_config.py`. | **CLEARED** |
| 18 | MoCap XY is **recentered on the subject**, not room position (18 subjects all centroid ≈(−0.22,−0.21) m; "walk" translates 0.11 m in 100 s — activities are done in place) | MoCap gives **pose + absolute height only**. Sensor extrinsics cannot be triangulated from subject translation → calibration must use the **RGB-D cameras as the bridge**, anchored on the fixed standing spot, with the radar's range/azimuth to the subject as extra constraints. | Strategy revised |
| 19 | **The moving human is NOT recoverable from the OctoNet Vayyar returns.** Six controlled tests all null: magnitude range profile, detrended envelope, anti-aliased envelope, per-bin spectra, complex slow-time Doppler (SNR ≈1.5 = noise floor), and a properly-powered posture classification (**leave-one-out 35.7%, BELOW the 48.9% permutation null, p=0.85**, n=28 clips / 6+ subjects). An earlier "80% posture accuracy" was in-sample overfitting with n=1 and is **retracted**; so is the earlier "stable 0.74 m human return", which proved to be **0.035 Hz thermal drift** (the radar peaked at the same frequency regardless of the true repetition rate). | **Diagnosed, not a bug.** Loading matches OctoNet's own loader exactly; the SFCW operation is verified correct (100-axis is genuinely raw frequency: flat magnitude, linear phase wind, IFFT raises sparsity 1.63→29.48); MoCap's own spectrum recovers the repetition rate exactly. Hardware is healthy (**0/400 dead pairs**, 23.9 dB inter-pair range). The cause is physical: the capture is **99.8 % a fixed instrument/room response** (frame-to-frame complex corr **0.999**; corr between *different recordings* **0.998**), with the entire dynamic component at **−24.2 dB** and drift-dominated — and at λ=4.6 mm sampled at 3.1 Hz a human's phase **decorrelates between frames**, so coherent/Doppler integration cannot dig it out. Vayyar EIRP is only −5 dBm. OctoNet's own benchmark never needs a physical return (it feeds raw complex frames to a CNN). | **Diagnosed** |
| 19b | Consequence + constructive path | The Vayyar stream is **unsuitable as a dynamic-human sensor**, but its dominance by a *stable, highly reproducible* static response is exactly what **static-scene material/geometry inverse rendering wants** — and that is the NBS target (the pipeline already builds a per-scene static scaffold with the human masked out, obstacle #3). Repeatability across recordings (corr 0.998) makes it a strong, low-variance fitting target. → **Re-scope OctoNet/Vayyar to static-scene + material inverse rendering**; use RGB-D room geometry (not the subject) to calibrate the radar pose, which also sidesteps obstacle #18. Human-dynamics claims should not rest on this radar. | Re-scoped |

## 8. Reproducibility — key files
- Positioning: `md/positioning.md`. Plan: `EXECUTION_PLAN.md` / `PLAN_v2.md`.
- Stage 1: `reference_code/NBS/preprocessing/` (calibrate_rig, prep_scene, build_scaffold, build_manifest).
- Stage 2: `reference_code/NBS/eval/downstream/`.
- Method: `reference_code/NBS/method/` (freq_material, ra_forward, wifi_forward, gitu_gate*, stage5_crossband, band_translate) + `mm3DPS/bsdf_torch.py::set_frequency`, `mm3DPS/data/ra_utils.py::adc_to_ra_complex_single_chip`.
- Audits: `method/diagnose_translate.py`, `method/diagnose_stage5.py`.
- Datasets in `~/Documents/Data`; outputs/scaffolds/manifests in `data/XRF55/`.
