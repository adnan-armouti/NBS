# NBS Execution Plan — from framing to results

> Builds on [PLAN_v2.md](PLAN_v2.md) (strategy, gates) and [md/positioning.md](md/positioning.md) (competitive read). This file is the **operational** plan: the concrete stages, files, and gates to produce results, starting with XRF55, then MM-Fi and OPERAnet.
>
> **Thesis (unchanged):** synthesize a held-out band's raw, phase-coherent radar signal by inverse-rendering band-transferable physical materials through a differentiable physics radar forward model, and prove it on a downstream sensing task. Power-field cross-band methods (XBandGS, XFreq-GS, Wideband3DGS) cannot do either.

---

## 0. Plan-critical reality (from the on-disk data audit, Jun 2026)

The audit changed two assumptions in PLAN.md's dataset table. Read this before building anything.

| Finding | Evidence | Consequence for the plan |
|---|---|---|
| **XRF55 mmWave is a processed range-Doppler/RA tensor** `(1,17,256,128)`, float32 (no raw ADC). | `~/Documents/Data/XRF55/RF/Scene*/mmWave/*.npy` | A coherently-formed RA/RD product still encodes **relative inter-antenna phase** (the azimuth FFT across virtual antennas), so it supports the relative-phase signal claim exactly as `|RA|` supervision does in 3DPS/mmIR. We scope out **absolute** phase (noise / sub-mm registration). The real limit is that XRF55 mmWave is **single-band**: it validates signal fidelity *at its band* and carries downstream, but the clean *held-out-carrier* cross-band synthesis needs a multi-carrier substrate (sim / PanoRadar). |
| **XRF55 WiFi CSI is `(270,1000)` float64, real-valued.** Phase likely already discarded. | `~/Documents/Data/XRF55/RF/Scene*/wifi/*.npy` | Complex CIR may not be recoverable. WiFi CIR rendering (Stage 3) is **gated on G-CSI** below. |
| **XRF55 RGB-D/IR unconfirmed.** Videos are `.mkv`; depth/IR streams not verified. | `~/Documents/Data/XRF55/videos/open/scene*/` | Stage 1 must `ffprobe` one file first. If RGB-only, monocular depth still works; if depth present, use it directly. |
| **XRF55 cross-band = mmWave (60 GHz) ↔ WiFi (2.4/5 GHz).** | dataset modalities | This is cross-**modality** + cross-**band** at once, the hardest, possibly ill-posed at signal level. Needs the within-radar multi-carrier fallback (PanoRadar/ColoRadarPlus/sim) for the clean claim. |
| **MM-Fi is not on disk.** | `~/Documents/Data/` has XRF55, OPERAnet, PanoRadar, ColoRadarPlus | MM-Fi must be acquired (Stage 7). |
| **PanoRadar (572 GB) and ColoRadarPlus are on disk and are raw-complex.** | `~/Documents/Data/PanoRadar/*/radar_data.npy` complex64; ColoRadarPlus | These, plus sim, are the **raw-signal-fidelity track** and the home of the G-ITU and phase-fidelity experiments. |

**The raw-phase claim (relative inter-antenna phase) is supportable on *any* dataset that gives a coherently-formed product** (`|RA|`/RD or complex CIR) to evaluate per-antenna magnitude + relative phase against, because our forward model emits complex CIR/ADC. So the two-track split is **not** "can/can't support phase"; it is **single-band vs multi-carrier**:
- **Track A — Clean cross-band (held-out carrier) synthesis:** needs ≥2 carriers. Run in sim (mmIR, driven at many carriers) + PanoRadar / ColoRadarPlus. Owns the phase-fidelity-vs-band-gap curve, extrapolation across `4π·sigma_h/λ≈1`, and the G-ITU gate (E1/E2/E3/E5).
- **Track B — Single-band reconstruction fidelity + downstream payoff:** XRF55 first, then MM-Fi, then OPERAnet. Validates relative-phase signal fidelity at the native band (via `|RA|`-style supervision) and carries the train-on-real-vs-train-on-synthesized downstream result (E4). The XRF55 cross-band axis is mmWave↔WiFi (cross-modality), handled honestly with negative controls.

XRF55 is started first per the directive (Track B). Track A runs in parallel because the clean *cross-band* phase-vs-band-gap result lives there.

---

## Repo and data conventions

- **Method code:** create `/home/adnan/Desktop/NBS/reference_code/NBS/` as a copy of `3DPS_submission` (keep `mmIR_submission` as the physics-rich reference to port dispersion + CIR from).
- **Preprocessed data out:** `/home/adnan/Desktop/NBS/data/<dataset>/<scene_or_clip>/...` (per the directive; the large source datasets stay in `~/Documents/Data`).
- **Configs:** `reference_code/NBS/configs/<dataset>/...` mirroring the existing `configs/seq_*/config.json` schema.
- **Per-run outputs / metrics:** `reference_code/NBS/output/<run>/...` with `aggregate_metrics.json` + per-scene `best_metrics.json` (so `/check-prose` can trace paper numbers).

---

## Stage 0 — Setup + data-reality audit  (gate: **G0**)

**Goal:** stand up the NBS repo and turn the audit's open questions into verified facts before building.

1. **Create the NBS code copy.**
   ```bash
   cp -r /home/adnan/Desktop/NBS/reference_code/3DPS_submission \
         /home/adnan/Desktop/NBS/reference_code/NBS
   mkdir -p /home/adnan/Desktop/NBS/reference_code/NBS/preprocessing
   mkdir -p /home/adnan/Desktop/NBS/data
   ```
   Recreate the conda env from `NBS/environment.yml`; verify `python -m mm3DPS.train_frame_nvs --help` runs (CUDA kernels build).
2. **XRF55 video probe (resolves RGB-D/IR).** `ffprobe` one `.mkv` (e.g. `videos/open/scene1/01/01_01.mkv`); record stream count/types. Decide: depth available → use it; RGB-only → monocular depth (Stage 1).
3. **XRF55 mmWave axis characterization (G-XAXIS).** Load one `mmWave/*.npy`; determine what `(1,17,256,128)` axes are (antenna vs chirp vs range vs Doppler). Decide whether **azimuth/RA is recoverable** (needed to reuse the RA-based metric-scale step) or whether only **range** is usable for scale.
4. **XRF55 CSI complex check (G-CSI).** Inspect `wifi/*.npy`: is it amplitude-only, or amplitude+phase interleaved / recoverable to complex `H(f)`? This decides whether Stage 3 can render *phase-coherent* CIR or only magnitude.
5. **Label + split manifest.** XRF55 has no official split file. Build `data/XRF55/manifest.csv` from the `subject_action_trial` filename convention (55 actions); define a subject-disjoint train/test split and freeze it.
6. **Sensor metadata.** XRF55 ships no calibration. Record the radar config (IWR6843 defaults: center freq, bandwidth, chirp slope, ADC rate) and camera intrinsics assumptions; flag every assumed value.

**G0 exit:** a one-page `data/XRF55/AUDIT.md` answering depth?, mmWave axes?, CSI complex?, split frozen?, sensor params?. **If CSI is amplitude-only → G-CSI fails → Stage 3 is magnitude-level and the XRF55 WiFi direction drops to a magnitude claim** (note it, do not pretend otherwise).

---

## Stage 1 — Preprocessing: geometry initialization for XRF55  (Track B)

**Goal:** produce the per-clip 3D point initialization 3DPS needs, by porting the phase56 depth extractor and adapting it to XRF55. Output to `/home/adnan/Desktop/NBS/data`.

**Source to port (from `/home/adnan/Desktop/3DGPS/mm3DPS/phase56`):**
- `gen_mono_depth.py` — MiDaS DPT-Large monocular depth (`torch.hub intel-isl/MiDaS`). Reuse directly.
- `radar_depth.py::build_radar_world_points` / `build_radar_lifted_points` — metric-scale correction from the radar map. **These assume a range-azimuth (RA) map**; XRF55 mmWave is range-Doppler (per G-XAXIS). Adapt the rescale to use the **radar range axis** for metric scale (least-squares scale+shift of mono-depth against the dominant per-column range return) when azimuth is unavailable.
- `shared_geometry.py::SharedGeometry.from_pcl`, `build_rgb_views.py` — geometry container + view metadata.

**New files in `reference_code/NBS/preprocessing/`:**
- `xrf55_loader.py` — read `.mkv` frames (RGB, and depth/IR if G0 found them), mmWave `.npy`, CSI `.npy`, pose `.json`; expose a clip iterator keyed by `subject_action_trial`.
- `gen_mono_depth.py` — ported MiDaS; RGB frame in → disparity out.
- `radar_metric_scale.py` — the adapted metric-scale step (range-based fallback + RA-based path if azimuth recoverable). Quote the rescale math from `radar_depth.py:25-50` and `:74-97`; replace the RA peak-finder with the XRF55 mmWave equivalent.
- `build_init_points.py` — fuse metric depth (+ camera intrinsics) into an oriented point cloud `pcl.npy` (cols 0-2 xyz, 3-5 unit normals), matching what `train_gaussian.py` reads (`pcl.npy` at `train_gaussian.py:540`).

**Output layout (REVISED for the moving human — static scaffold is per SCENE, not per clip):**
```
data/XRF55/<scene>/scene/pcl.npy   # SHARED static scaffold (human masked out)
data/XRF55/manifest_<scene>.csv    # all clips -> scaffold + radar + pose + frozen split
data/XRF55/<clip>/views/depth_*.npy, alignment.json, ra_overlay.png
```
The static room geometry is a per-scene constant (one room, fixed rig); the moving
human is per-clip dynamic content (in the 17 radar frames + 2D pose), masked out of the
scaffold via `ra_geometry.human_mask`. See [data/XRF55/STAGE1_OUTPUTS.md](data/XRF55/STAGE1_OUTPUTS.md).

**Validation:** overlay initialized points on RGB (reuse `phase56/radar_bridge.py::sanity_overlay`); check metric scale against the dominant radar range. **Done when** ≥1 XRF55 clip initializes and renders a plausible RD/RA under the existing renderer.

**STATUS (Jun 2026): IMPLEMENTED + TUNED + METRIC.** `reference_code/NBS/preprocessing/` runs end-to-end in the `mmir` env. Default depth backend = **Depth-Anything-V2-Metric-Indoor** (true-metric meters; `--backend midas` for the relative fallback). Two-stage calibration: per-scene RIG (`<scene>_rig.json`, median joint-pose solve, validated stable and matching the rig photo: yaw 62.5°, x −1.70 m, y 0.80 m, z −0.36 m) + per-clip depth scale. Tuned via `tune_align.py` (dB correlation, az_sin 0.4, blur (1,8), max reduction). Metric depth removed the affine ambiguity, recovered the radar range-resolution (~0.076 m/bin), and produced wall-faithful geometry (room walls at constant range, matching radar RA bands). Outputs `data/XRF55/<clip>/scene/pcl.npy` + `alignment.json` + overlays. Full recipe + honest residuals: [data/XRF55/CALIBRATION.md](data/XRF55/CALIBRATION.md). Next: batch all clips; optionally fix a=1 + solve scene-level range_res once authors confirm the radar config.

---

## Stage 2 — Downstream-task baselines + upper bounds  (Track B)

**Goal:** the train-on-real numbers that the synthesized-band result is measured against. Without these the payoff experiment (Stage 6) has no yardstick.

1. **Tasks on XRF55:** 55-class action recognition (primary), 2D pose (from `pose/*.json`).
2. **Per-modality native models** (off-the-shelf / XRF55-paper backbones): one classifier each on **mmWave RD**, **WiFi CSI**, **RFID**. Report top-1 on the frozen split. These are the **upper bounds** (model on real band X tested on real band X).
3. **Lower-bound / sanity baselines:** nearest-band copy; per-band-independent model (no synthesis); chance.
4. **Wire the harness** at `reference_code/NBS/eval/downstream/` so a model can be pointed at *real* or *synthesized* inputs interchangeably (the Stage 6 swap).

**Done when** native-band top-1 is reproduced for mmWave and WiFi on the frozen split, logged to `output/baselines/aggregate_metrics.json`.

**STATUS (Jun 2026): HARNESS BUILT + FIRST UPPER BOUNDS (Scene1, floors — 10 trials/10 epochs).** Top-1 (chance 1.8%): mmWave 0.441 cross-subject / 0.592 within-dist; WiFi 0.038 / 0.446; RFID 0.061 / 0.207. **Key finding:** mmWave generalizes cross-subject; WiFi/RFID are classifiable only within-distribution (amplitude CSI/RFID is subject-specific). So the cross-band payoff (Stage 6) should use the within-distribution split, and **mmWave is the strongest payoff vehicle**. Harness `eval/downstream/` supports the real↔synthesized swap via `XRFHar(input_dir=...)`. Full results: [data/XRF55/STAGE2_HAR_BASELINES.md](data/XRF55/STAGE2_HAR_BASELINES.md).

---

## Stage 3 — WiFi CIR forward model  (gated on **G-CSI**; needed for the WiFi direction)

**Goal:** a renderer that maps the shared scene + materials to a **WiFi channel** (CIR / CSI), so the same scene can emit both the radar and the WiFi observables.

- mmIR/3DPS have **no CIR/OFDM/WiFi path** (confirmed: zero matches for `cir/ofdm/wifi/csi`). This is new code.
- Host it parallel to the FMCW ADC accumulator: new `reference_code/NBS/mm3DPS/cir_rasterizer.py` (analog of `mmIR_submission/sensor/adc_accumulator.py`, whose phase law is `phi = 2*pi*(f0*tau + 1/2*S*tau^2) + ...`).
- **WiFi is bistatic** (Tx → scene → Rx), not monostatic radar. Path accumulation: per multipath `i`, complex gain `a_i` and delay `tau_i`; `H(f) = Σ_i a_i exp(-j 2π f tau_i)`; CIR = IFFT over the OFDM subcarrier grid. Reuse the existing ray/path + E-field + BSDF machinery; change the sensor model and the Tx/Rx geometry.
- **Carrier set by the band parameter** from Stage 4 (2.4/5 GHz for WiFi vs 60/77 GHz for radar), so the same materials are evaluated at the WiFi wavelength.

**Reframe (Jun 2026):** CSI being amplitude-only is fine — render the **complex** CIR/CSI (phase physically determined by scene+materials) and supervise the **magnitude** `|CSI|`, exactly as complex-ADC is supervised by `|RA|`. Because each modality is ~single-band, the cross-band bridge is **material identification** (`method/band_translate.py`), not dispersion-law fitting.

**STATUS (Jun 2026): RETRACTED — result was invalid.** The reported mmWave↔WiFi translation (corr 0.95–0.99, copy baseline 0.000) does NOT hold: the metric is **geometry-dominated** (wrong materials score corr 0.998 / dB −24.5, beating NBS's −23.0), and the copy baseline was **degenerate** (recovered amplitude `[0,0,0]`). NBS did not beat a wrong-material floor. `method/wifi_forward.py` (complex bistatic CSI/CIR) is sound code, but no cross-band translation is demonstrated. See [md/method_audit.md](md/method_audit.md) and [md/band_translate_result.md](md/band_translate_result.md). The render-complex/supervise-magnitude *premise* may still be valid; proving it needs a geometry-independent metric + validated baseline.

**Done when** the CIR renderer reproduces a held-out WiFi CSI clip on a static scene to a stated correlation.

---

## Stage 4 — Frequency-differentiable material model  (gate: **G-ITU**; the mechanism)

**Goal:** make the 6-parameter material respond to a **differentiable frequency input**, so one scene re-renders at any carrier and gradients flow to band-invariant material coefficients. This is PLAN_v2's G-ITU gate.

**Current state (verified):**
- `NBS/mm3DPS/bsdf_torch.py:28` hardcodes `WAVELENGTH = 3.9e-3` (77 GHz); `K_WAVE` at `:29`. Frequency enters at GGX roughness `4π·sigma_h/λ` (`:209`), SPM vMF (`:244-246`), CBS (`:352`), SPM validity (`:176`), ITU slab phase `q = 2π·thickness·a/λ` (`:134`). The 6 params `[eps_real, eps_imag, sigma_h, l_c, tau, thickness]` are reparameterized at `reparameterize_torch` (`:31-71`).
- mmIR has the **ITU dispersion table** `eps_r = a·f^b, sigma = c·f^d` (`renderer/materials/itu_materials.py:266`, coefficient table `:41-138`) but it returns **plain floats** (non-differentiable, name-keyed lookup). This is exactly the unbuilt gradient path.

**Work:**
1. **Thread frequency through the BSDF.** Replace the `WAVELENGTH` constant in `bsdf_torch.py` with a per-evaluation `wavelength = c / f` argument; propagate `f` into the rasterizer (`rasterizer_factorized.py::render_factorized`, the CUDA kernels `cuda/bsdf_forward.cu` / `bsdf_backward.cu` take `lambda_i, lambda_o` already, so expose `f` upstream).
2. **Make dispersion differentiable.** Add two band-coefficient parameters per material so `eps_real(f) = a·f^b` (and optionally `sigma(f) = c·f^d`) become a grad-enabled function, initialized from the ITU table values in `itu_materials.py`. Recover `(a,b)` by backprop, not a lookup. The other params (`sigma_h, l_c, tau, thickness`) stay band-invariant; their **wavelength dependence is already in the physics** (roughness, slab phase).
3. **Keep a learned residual only for the regime change.** Near `4π·sigma_h/λ ≈ 1` (smooth→rough), add a small bounded correction conditioned on `(sigma_h, λ)`; ablate it (PLAN_v2 E7b).

**G-ITU exit (toy scene):** make `(a,b)` grad-enabled, run an A→B material-transfer optimization on a sim/toy scene, re-render the raw signal at a held-out carrier, and report quantified phase/signal error. **Pass → "analytic dispersion continuation" stays in the headline; fail → fall back to per-band physics materials re-rendered through the wavelength-dependent forward model** (still novel, no "analytic dispersion" claim).

**STATUS (Jun 2026): G-ITU MECHANISM PASSED (toy + hardened); E3 advantage is MODEST in the realistic regime.**
- `method/freq_material.py` + `method/gitu_gate.py` (idealized toy): recovering `(a,b,c,d)` and continuing to a held-out carrier gives −75 to −96 dB vs interpolation's −34 to −70 dB. *But that 40 dB gap was inflated by noiselessness + unphysical interpolation.*
- **Hardened** (`method/ra_forward.py` + `method/gitu_gate_ra.py`): REAL 3DPS ITU slab Fresnel (`bsdf_torch.set_frequency` added), the actual single-chip/XRF55 array (Tx{0,2}×4Rx 0.5λ ULA), RA via new `ra_utils.adc_to_ra_complex_single_chip` (64 az bins = XRF55), **20 dB noise**, **|RA| supervision**. The differentiable dispersion loop converges through the real BSDF (corr ≈0.999) → **mechanism PASS**. The advantage over *physically-constrained* interpolation is **robustness, not magnitude**: NBS error/corr flat across 30–160 GHz; interpolation is better *within* the train span but degrades/erratic in extrapolation (crossover ~100 GHz). Realistic gap ≤ ~1 dB (noise/geometry-limited), not 40 dB. Figure: `output/gitu_gate_ra/ra_band_gap_sweep.png`.
- **Regime-change test — E3 RE-VALIDATED (corrected harness `method/eval_v2.py`).** The old `gitu_roughness.py` corr numbers were geometry-dominated and its interp baseline degenerate (retracted). Corrected re-run (two metrics + validated baseline; train smooth {18,28,40} → held 120 GHz rough): material-contribution floor = **−5.9 dB** (wrong roughness); **NBS −30.8 dB, material-ID accuracy 1.00** (25 dB below floor); **validated interp +0.3 dB → fails**. "Physics beats interpolation" **holds, scoped to where materials matter** (nonlinear roughness regime change). Raw vs normalized RA: equivalent. Real-data material GT (eval-only) via RGB seg: `preprocessing/material_labels.py`. See [md/method_audit.md](md/method_audit.md) RE-RUN section.
- **Positioning consequence (honest, scoped):** physics-vs-interpolation advantage is NOT universal — it is concentrated where the cross-band response is *nonlinear/structured* (roughness regime change shown; penetration/multi-bounce by extension); in smooth densely-sampled regimes interpolation is competitive. So "analytic dispersion + physical material recovery" is a **defensible secondary** differentiator (robustness + regime-change fidelity); the **headline stays raw-signal + downstream-sensing vs PAS**. Full honest write-up: [md/gitu_gate_result.md](md/gitu_gate_result.md). TODO: CUDA rasterizer `f`-threading; mmIR sim (Track A); penetration/multi-bounce tests.

---

## Stage 5 — Cross-band inverse rendering, both directions  (the core experiment)

**Goal:** train on one band, synthesize the held-out band, in both directions.

**Track A (clean, within-radar multi-carrier — the defensible version):** in sim (mmIR) and on PanoRadar/ColoRadarPlus, fit materials at training carriers, synthesize the raw signal at a held-out carrier. Report complex-CIR/ADC correlation, range error, and phase error vs band gap (E1/E2/E3). This is where the phase headline is earned.

**Track B (XRF55, the ambitious cross-modality demonstration):**
1. **mmWave → WiFi:** fit scene materials to XRF55 mmWave (60 GHz, via the RD forward model), re-render the **WiFi** observable at 2.4/5 GHz (Stage 3), compare to the real WiFi clip.
2. **WiFi → mmWave:** fit to WiFi CSI, re-render the mmWave RD, compare to the real mmWave clip.
3. **Honesty:** this is cross-modality + cross-band simultaneously and may be ill-posed at the signal level (different geometry, non-co-located, processed inputs). Report it with a negative control (predict-the-mean) and label it a stretch; the clean cross-band claim is Track A. If signal-level transfer is ill-posed, fall back to **feature/observable-level** transfer that still feeds Stage 6.

**Protocol:** hold-one-band-out; per-direction config in `configs/XRF55/`. Reuse the optimization loop in `train_frame_nvs.py`; loss on the available observable (complex if possible, else magnitude). Baselines: nearest-band copy, per-band-independent, and a **ported XBandGS-style power-field head** (to show it cannot emit the signal/phase that Stage 6 needs).

**Done when** both directions produce a synthesized held-out-band observable with a reported fidelity number and the baselines are run identically.

**STATUS (Jun 2026): partial — core recovery survives audit, the interp comparison does NOT.** `method/stage5_crossband.py` renders the real Scene1 scaffold at train {24,40,60} GHz, recovers per-class dispersion+roughness, synthesizes the held-out **90 GHz** RA. After audit ([md/method_audit.md](md/method_audit.md)): the reported **corr 1.000 is geometry-dominated → meaningless**; the reported **interp baseline −5.9 dB was degenerate** (worse than a −20 dB wrong-material floor) → "beats interp" retracted. **What survives:** NBS **−37.5 dB is ~17 dB below the wrong-material floor (−20 dB)** → on this dense over-determined real-geometry scene the model genuinely recovers materials and synthesizes the held band. Still Track A (real geometry, controlled materials/signal); real multi-band raw validation blocked on R1. Write-up: [md/stage5_result.md](md/stage5_result.md).

---

## Stage 6 — Downstream payoff: native-band model on the synthesized band  (the headline for Track B)

**Goal:** the train-once-deploy-cross-band result.

1. Take the Stage 2 native-band model for band X (trained on **real** X).
2. Feed it band X **synthesized from** band Y (Stage 5).
3. Report top-1 as **% of the train-on-real upper bound** (target ≥ Y%, from PLAN_v2 P1-b), against nearest-band copy and per-band-independent.
4. **The load-bearing ablation (E5):** the contrast is **not** "magnitude vs complex" (an `|RA|` map already carries relative inter-antenna phase). It is the **coherently-formed product our model emits (RA/RD/CIR, invertible to the complex signal, supporting range/Doppler/micro-Doppler) vs a power-collapsed product (PAS, `|·|²` over angle) that cannot be inverted to the complex signal and loses the fast-time/range structure.** Show a phase-dependent sensing task that succeeds on the synthesized coherent product and fails on the PAS-collapsed version. Run the clean version on Track A where carriers are controlled.

**Done when** `output/xrf55_crossband/aggregate_metrics.json` reports accuracy-recovery for both directions with all baselines.

---

## Stage 7 — Scale out: MM-Fi, then OPERAnet

- **MM-Fi (must acquire; not on disk).** WiFi CSI + mmWave + LiDAR + RGB-D. LiDAR gives a *real* geometry anchor (no monocular-depth dependence) and pose/HAR tasks. Repeat Stages 1/5/6. mmWave there is processed point-cloud (flag for raw-level limits, per PLAN.md).
- **OPERAnet (on disk).** WiFi CSI (`.mat` v7.3 / HDF5), Kinect RGB-D skeleton, **UWB with complex CIR (35 samples)**, power. The UWB CIR is the one **raw-ish complex** real signal in the decks. **But OPERAnet is comm-style CIR, not monostatic radar (PLAN_v2 E6: not natively renderable by mmIR/3DPS)** — use it for downstream HAR and as the UWB-band context, not for the monostatic material claim. Loader must handle HDF5 `.mat`.
- Keep the geometry-anchor differentiator: MM-Fi/ColoRadarPlus give LiDAR; XRF55/OPERAnet give RGB-D.

---

## Cross-cutting

**Baselines (every dataset):** nearest-band copy; per-band-independent; **ported XBandGS-style power/PAS head** (anchor showing power cannot emit phase/signal); a **phase-capable but no-dispersion** complex baseline (PLAN_v2 E8, so the win is dispersion/regime, not "we output complex"). Do **not** chase XBandGS on PAS-SSIM.

**Metrics:** complex CIR/ADC correlation (mag + phase); sub-bin range error (cm); phase-derived Doppler/displacement; phase-error-vs-band-gap curve (Track A); downstream top-1 as % of upper bound (Track B); RA/RD correlation (`cart_corr`, reuse `evaluation/utils/metrics.py`). Always split interpolation vs extrapolation.

**Gates (go/no-go):**

| Gate | Question | Status (Stage 0 audit) | Fallback if it fails |
|---|---|---|---|
| **G0** | XRF55 depth?/mmWave axes?/CSI complex?/split? | **DONE** — see [data/XRF55/AUDIT.md](data/XRF55/AUDIT.md). RGB-only; mmWave + CSI magnitude-only; split TBD. | range-based scale; magnitude-level WiFi |
| **G-CSI** | Is XRF55 CSI complex-recoverable? | **Moot / reframed** — CSI is amplitude-only `(270,1000)`, but that does NOT block synthesis: render complex CIR/CSI, supervise `|CSI|` (like complex-ADC → `|RA|`). mmWave↔WiFi translation via **material identification** (`band_translate.py`). | n/a — render-complex/supervise-magnitude |
| **G-XAXIS** | Does XRF55 mmWave have a beamformed **angle** axis (→ relative inter-antenna phase claim)? | **RESOLVED: YES.** mmWave `(17 frames, 256 range, 128=[Doppler 0:64 \| azimuth 64:128])`. RA = `A[frame,:,64:128]`. Azimuth half preserves relative inter-antenna phase → supports the `|RA|` claim (plus range + micro-Doppler). | n/a — XRF55 mmWave supports the relative-phase claim AND downstream |
| **G-ITU** | Differentiable `eps_r(f)=a·f^b` transfer works on a toy scene? | **PASS** (`method/gitu_gate.py`): held-out band complex-corr 1.0; analytic dispersion recovers the raw signal to −75 to −96 dB across 30–100 GHz vs interpolation's −34 to −70 dB (gap widens with band distance). "analytic dispersion" stays in the headline. | per-band physics materials, drop "analytic dispersion" from headline |
| **G-TrackB** | Is mmWave↔WiFi signal-level transfer better than the mean? | not started | feature/observable-level transfer feeding Stage 6; Track A carries the signal claim |

**Risks:** XRF55 processed-only data caps the signal claim (mitigated by Track A); cross-modality ill-posedness (mitigated by within-radar Track A + negative controls); MM-Fi acquisition latency; CUDA-kernel changes for the frequency input (test against the PyTorch reference path in `cuda/reference.py`).

**Milestones (toward Sep 4):**
- Now: Stage 0 (G0) + start Stage 1 + start Track A sim harness + start Stage 4 (G-ITU).
- Jun: Stage 1 done; Stage 2 upper bounds; G-ITU decided on toy scene.
- Jul: Stage 5 Track A (phase-vs-band-gap, E1/E2/E3); Stage 3 if G-CSI; Stage 5 Track B both directions.
- Aug: Stage 6 payoff (both tracks); MM-Fi (Stage 7) if acquired; ablations; freeze mid-Aug.
- Aug 28 / Sep 4: abstract / paper.

---

## Target directory structure (new under `reference_code/NBS/`)

```
reference_code/NBS/
  preprocessing/
    xrf55_loader.py          # .mkv/.npy/.json clip reader
    gen_mono_depth.py        # ported MiDaS depth
    radar_metric_scale.py    # range/RA metric-scale correction
    build_init_points.py     # -> data/<ds>/<clip>/scene/pcl.npy
  mm3DPS/
    bsdf_torch.py            # MODIFIED: frequency-as-input + dispersion (a,b) (Stage 4)
    rasterizer_factorized.py # MODIFIED: thread f through render
    cir_rasterizer.py        # NEW: WiFi/OFDM CIR forward model (Stage 3)
    cuda/                     # MODIFIED: expose f / lambda upstream
  eval/downstream/           # HAR/pose harness, real-or-synth swap (Stage 2,6)
  configs/{XRF55,MMFi,OPERAnet}/...
  output/<run>/{aggregate_metrics.json, <scene>/best_metrics.json}
```

---

## Things not in the original 5 steps that this plan adds

1. **A data-reality gate (Stage 0)** — XRF55 is processed-only and missing calibration/split; this must be resolved before building, and it reclassifies XRF55 as the downstream track.
2. **A separate multi-carrier track (Track A)** on sim/PanoRadar/ColoRadarPlus — not because XRF55 is phase-blind (its `|RA|`/RD carries relative inter-antenna phase), but because the clean *held-out-carrier* cross-band synthesis needs ≥2 carriers and XRF55 mmWave is single-band.
3. **The frequency input must reach the CUDA kernels**, not just the Python BSDF (the kernels already accept per-direction `lambda`; the plumbing is upstream).
4. **A phase-capable, no-dispersion baseline** (E8) so the win is dispersion/regime, not "we output complex."
5. **The load-bearing ablation (E5)** run where phase is well-defined (Track A), since XRF55 may not support it.
6. **MM-Fi acquisition** as an explicit task (it is not on disk).
7. **OPERAnet's comm-CIR caveat** — usable for HAR + UWB context, not the monostatic material claim.
8. **Honest framing of mmWave↔WiFi** as cross-modality+cross-band with negative controls, not a clean cross-band result.
