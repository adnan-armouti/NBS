# PLAN_v2 — NBS: Novel Band Synthesis of the RAW, Phase-Coherent Radar Signal

> **One-line thesis:** Cross-band RF synthesis should recover *band-invariant physical scene materials* and re-render the **raw, phase-coherent radar signal** — and we measure this by a *quantitative phase-fidelity-at-an-unseen-carrier* result that drives a phase-dependent sensing task, an axis the Power Angular Spectrum (PAS) of every prior cross-band radiance field cannot occupy.
>
> **Venue target:** ACM MobiCom (abstract ~Aug 28, paper ~Sep 4, 2026).
> **Award framing is *conditional*, not asserted:** best-paper language is permitted in the abstract **only if** the early-July data gate (R1) yields a real, calibrated, distinct-carrier capture. Absent that, the paper ships as an honest, sim-anchored strong-accept on **a capability axis orthogonal to XBandGS** — not a best-paper bid against a same-venue real-data campaign.
> **Project dir:** `/home/adnan/Desktop/NBS`.
> **In-repo levers (verified against source, Jun 2026):**
> - `reference_code/mmIR_submission` — differentiable FMCW radar renderer (multi-bounce, SMS, diffraction, Jones-polarization BSDF) emitting **raw complex ADC**; FMCW phase law in `sensor/adc_accumulator.py`.
> - **Differentiable per-frequency physics materials exist** (`renderer/bsdf/reparameterization.py::reparameterize_physics_params_drjit`): grad-enabled `[eps_real, eps_imag, sigma_h, l_c, tau, thickness]`, sigmoid/log-bounded, wired into Fresnel/slab via `renderer/bsdf/mmwave_scalar.py::permittivity_to_ior`. **This is a real, in-repo differentiable inverse-rendering lever — at a FIXED carrier.**
> - `reference_code/3DPS_submission` — differentiable point-splatting RA renderer (complex range profiles), fast (~minutes/scene).
> - **What does NOT exist in-repo (do not claim it does):** the *band-coupling*. `renderer/materials/itu_materials.py::get_itu_properties` is a non-differentiable, **name-keyed Python lookup** that returns plain floats from hardcoded `(a,b,c,d)` via `eps_r=a·f^b`, `sigma=c·f^d`. The dispersion coefficients `(a,b,c,d)` carry **no gradient** and the in-repo grad path optimizes a **per-frequency, band-LOCAL** material, not a band-invariant `eps_r(f)` law. Building a differentiable `eps_r(f)/sigma(f)` parameterization that ties the two together is **unbuilt engineering on a 4-week clock** and is gated below (G-ITU).
> **STATUS: thesis re-centered + red-team-hardened (Jun 2026).** Supersedes PLAN.md. Two hard gates decide the paper's class: **G-ITU** (differentiable dispersion inverse loop demonstrated on a toy scene) and **R1** (real distinct-carrier capture in hand), both by the early-July gate.

---

## Crisp novelty claim (the one sentence, de-overclaimed)

> **NBS synthesizes a held-out RF band's raw, phase-coherent radar signal by inverse-rendering band-invariant scene materials through a differentiable physics radar forward model, and quantifies — for the first time — the *phase fidelity* of that synthesized signal at an unseen carrier well enough to drive a phase-dependent sensing task; this is a capability axis the power-angular-spectrum of every prior cross-band radiance field (XBandGS / XFreq-GS / Wideband3DGS) structurally cannot occupy.**

**Two deletions from the prior first-claim, per the red-team:**
1. "**ITU-dispersive materials**" is **removed from the headline** until G-ITU passes. If G-ITU passes, the headline upgrades to "…band-invariant, *analytically-continued ITU-dispersive* materials…"; if it fails, Pillar 2 is re-scoped to the GGX/IOR/roughness path the repo already differentiates (see G-ITU fallback). The headline never asserts a mechanism the code has not demonstrated.
2. "**first to synthesize raw phase-coherent signal**" is qualified to the *measured phase-fidelity* result, not the mere act of emitting complex numbers.

The paper is **one** load-bearing argument with **two** pillars:

- **Pillar 1 — Capability, stated as a QUANTITATIVE target (not a tautology).** The headline is **not** "phase is undefined on PAS" (true by construction, cut as a discovery — see Overclaims). The headline is: *synthesized-band phase is correct enough to do the task.* Concretely we commit to targets:
  - **(P1-a) Phase fidelity:** synthesized-band micro-Doppler / vital-sign error **within X% of train-on-real-at-that-band** (target X ≤ 20% at the primary band gap; the value is set from the E2 sim sweep at the early-July gate and frozen).
  - **(P1-b) Downstream recovery:** phase-dependent task accuracy **≥ Y% of the train-on-real upper bound** (target Y ≥ 80%), and **strictly above** a *coherent-complex-but-no-dispersion* baseline.
  - **The real foil is a phase-CAPABLE baseline, not just PAS.** We add an **ADC-magnitude / non-coherent-complex** baseline so the win reads "*correct* phase," not "*has* phase." PAS is retained only as the trivial lower anchor.
- **Pillar 2 — Mechanism (depth / why it generalizes), scoped to what is built or gated.** Material inverse-rendering through the differentiable forward model. The repo *already* differentiates per-frequency `eps_r, sigma, sigma_h, l_c, thickness`. The **research delta vs XBandGS** is **conditional on G-ITU**: if the differentiable `eps_r(f)=a·f^b` inverse loop is demonstrated, the delta is "*analytic, non-learned* continuation across band + a *bounded learned residual only for the smooth-to-rough regime change* `4π·σ_h/λ≈1`." Until G-ITU passes, the positioning table marks this row **GATED** and the fallback delta (below) is used.

---

## What changed vs PLAN.md — and why

| # | PLAN.md (old) | PLAN_v2 (new) | Why (decision brief + red-team) |
|---|---|---|---|
| 1 | Four differentiators. | **Two surviving: (1) raw complex-signal synthesis, (2) a phase-dependent sensing capability — both stated as *quantitative* targets.** Real-data, geometry-anchoring, physics-vs-blackbox retired as headlines. | XBandGS (same venue) has real USRP data, a shared scaffold, and a physics-MoE. Re-center is necessary but not sufficient. |
| 2 | Output level loosely specified (ADC *or* RA fallback). | **Raw, phase-coherent complex ADC/CIR is the non-negotiable output; the win is *measured phase correctness*, not phase presence.** | PAS via Bartlett is non-invertible; but "we output complex, they output magnitude" is a strawman win. We commit to a phase-error number. |
| 3 | Downstream = generic HAR/pose/occupancy. | **Narrowed to a phase-dependent capability with committed accuracy targets (P1-a/P1-b) and a phase-capable foil.** Promoted to load-bearing E1. | Red-team: tautology risk. The hard, novel question is "phase *correct enough* at an unseen carrier?", not "magnitude has no phase." |
| 4 | Materials a co-equal pitch element. | **Mechanism pillar, *conditional on G-ITU*.** Headline drops "ITU-dispersive" until the differentiable dispersion loop is demonstrated. | itu_materials.py is a non-differentiable lookup; the delta-vs-XBandGS is aspirational until built. |
| 5 | Plenoptic headline floated. | **Cut from headline** (≤1 corollary figure). | Unanimously rejected; view×band = already-done × already-done; RF does not factorize. |
| 6 | Real human decks as headline. | **Clean substrate = mmIR sim + a single-radar multi-carrier in-house capture; decks demoted to labeled stretch.** Best-paper framing *conditioned on R1*. | Boldest claim is least evaluable on cross-hardware decks; feasibility is the decision driver. |
| 7 | Beat them on PAS-SSIM; one ported baseline. | **Do not chase PAS-SSIM. Add a *fair phase-capable* baseline (frequency-conditioned complex field / complex 3DGS) plus the ported XBandGS head.** | Porting only a power head and showing "it can't emit phase" is a strawman a hostile reviewer rejects. |
| 8 | Data treated as decided. | **Top risk = no clean distinct-carrier substrate (R1) + unbuilt dispersion loop (G-ITU); both hard-gated early July; explicit award-language condition; E6 run as a gating result IN-plan.** | Feasibility was the lowest reviewer score; the dataset table was aspirational. |

**Net:** one fully-nailed, *quantified* capability claim (Pillar 1) backed by a mechanism (Pillar 2) whose ambition is **gated to demonstrated code**, on a substrate whose existence is **gated to a real capture** — ambition preserved, overclaim removed.

---

## Positioning vs the competition

All three competitors share the paradigm: explicit 3DGS scene, shared geometry across bands, frequency-conditioned per-primitive RF attributes, **PAS / power-radiance output**, scored by SSIM/PSNR, framed as comm/channel/digital-twin.

| Axis | Wideband3DGS | XFreq-GS | **XBandGS** (MobiCom'26) | **NBS (PLAN_v2)** |
|---|---|---|---|---|
| Output signal level | PAS / power | PAS / power | PAS (beamformed power) | **Raw complex ADC / CIR (phase-coherent)** |
| Phase / fast-time preserved? | No | No | No (Bartlett discards complex CIR) | **Yes — and we *quantify its correctness*** |
| Cross-band mechanism | Learned freq embedding | Learned freq-adaptive attrs | Learned per-Gaussian residual + physics-MoE | **GATED (G-ITU): analytic ITU `eps_r=a·f^b` + bounded regime residual. Until G-ITU: differentiable per-band physics materials (GGX/IOR/roughness/slab) — already in-repo** |
| Handles smooth→rough / multi-bounce regime change? | No | No | **Disclaims extrapolation** (7.7% unseen-freq drop) | **Target: graceful through `4π·σ_h/λ≈1` — falsifiable, gated on G-ITU** |
| Downstream sensing task? | None | None | None | **Yes — phase-dependent, with committed accuracy targets** |
| Problem framing | Comm field | Comm field | Comm Tx→Rx | **Monostatic radar imaging of a scene** |
| What is recovered | Per-Gaussian attrs | Per-Gaussian attrs | Per-Gaussian power coeffs (+ latent `ε̂_r`) | **Per-point physics materials (`eps_r, sigma, sigma_h, l_c, thickness`); band-invariant continuation GATED on G-ITU** |
| Real data | Sim only | Sim (likely) | **Sim + real USRP** | Sim (mmIR) + **single-radar multi-carrier in-house — GATED on R1**; if R1 slips, sim-anchored and *framed orthogonal, not superior, on the real-data axis* |
| Headline metric | SSIM | SSIM | SSIM/PSNR/MSE | **Phase error vs band gap; complex-CIR corr; cm-range; downstream accuracy-recovery; PAS-SSIM shown alongside** |

**Why competitors cannot follow (kept narrow):** (i) PAS→power is non-invertible, so a PAS pipeline cannot emit phase; (ii) comm Tx→Rx framing has no scene to image and no native sensing task. We do **not** claim "they cannot do analytic dispersion" until G-ITU proves *we* can.

**Sim-vs-real asymmetry (confronted head-on).** If NBS is primarily simulated, it does **not** out-rank XBandGS on the real-data axis — and we will not pretend it does. Two honest stances: **(A)** *with R1:* one real distinct-carrier phase-dependent result is something XBandGS's power campaign **structurally cannot report**, so we win on a *new axis* rather than out-real-ing them on theirs. **(B)** *without R1:* we frame the contribution explicitly as "a capability axis orthogonal to XBandGS," remove award language, and let the sim study stand on the phase-fidelity-vs-band-gap curve plus the phase-capable-baseline comparison.

---

## Method

### Scene representation (shared, band-invariant geometry + materials)
- **Geometry** `{μ, Σ}`: static 3D scene from real optical geometry (LiDAR/RGB-D) or sim mesh; frequency-invariant (same assumption XBandGS uses — not claimed as a differentiator).
- **Materials (recovered quantity):** per-surface physics parameters `eps_r, sigma (eps_imag), sigma_h, l_c, thickness`. **These are differentiably recovered in-repo at a fixed carrier today** (`reparameterize_physics_params_drjit` → `permittivity_to_ior` → Fresnel/slab). The **band-invariant** parameterization (recover `a,b` such that `eps_r=a·f^b`, then continue to a new band) is **the part to build (G-ITU)**.

### Band + view conditioning
- **Band conditioning — two clearly-separated cases:**
  - **G-ITU PASS:** band dependence is the analytic ITU law `eps_r(f)=a·f^b, sigma(f)=c·f^d` (coefficients made grad-enabled and wired into the autograd graph feeding `permittivity_to_ior`), plus wavelength entering the roughness factor and FMCW phase law. *This is the structural opposite of a learned per-band table — but only claimable post-G-ITU.*
  - **G-ITU FALLBACK:** if the differentiable dispersion loop is not demonstrated by early July, band conditioning falls back to recovering per-band physics materials and the **wavelength-driven** terms the forward model already exposes (roughness `4π·σ_h/λ`, slab phase, FMCW phase law). The contribution is then "physically-grounded re-render through the wavelength-dependent forward model + the regime-change residual," **not** analytic dispersion continuation.
- **Regime-change residual (the only learned cross-band term, gated):** near `4π·σ_h/λ≈1`, scattering goes smooth→rough and multi-bounce coherence degrades (ITU does not model this). A *small, bounded* learned correction conditioned on `(σ_h, λ, local geometry)` activates near the crossing. Falsifiable: below the crossing, analytic/physics terms alone suffice; through it, NBS should degrade gracefully where learned field-interpolation collapses. **This claim has teeth only on scenes that actually span the crossing (see E7).**
- **View conditioning:** standard radar sensor pose drives ray tracing; spatial NVS is a free byproduct (3DPS) reported at most as one corollary figure.

### Physics forward model
- **mmIR:** differentiable FMCW renderer emitting **raw complex ADC** via the FMCW phase law `φ_k(τ)=2π[f_0·τ+½S·τ²]+2π[S·τ+f_d]·t_k` (`sensor/adc_accumulator.py`). This is the engine that produces phase.
- **3DPS:** differentiable point-splatting RA renderer (complex range profiles); E6 decides which engine applies per dataset.

### Inverse → synthesis loop
1. **Forward:** materials + geometry + pose + carrier `f_q` → raw complex ADC (mmIR) / complex range profile (3DPS).
2. **Inverse (train):** observe raw ADC at training bands; backprop complex-signal loss through the differentiable forward model to recover materials. *In-repo today for per-band materials; G-ITU extends the same graph to the dispersion coefficients.*
3. **Synthesis (test):** **G-ITU PASS** → push recovered `(a,b,c,d)` through the analytic law to held-out `f_q` and forward-render (physically-grounded continuation). **G-ITU FALLBACK** → re-render through the wavelength-dependent forward model at `f_q` from per-band materials transferred via the regime residual; report the honest gap.

---

## Experiment suite (mapped to defending the thesis)

**Load-bearing: E1. Build on the simulator first.** New: **E6 and a G-ITU demonstration are gating results reported IN the paper, not to-dos. E7 adds multi-scene external validity and the analytic-vs-residual sensitivity split. E8 is the fair phase-capable baseline.**

| # | Experiment | Defends against… | Clean substrate (must work) | Stretch (labeled) |
|---|---|---|---|---|
| **E1 — LOAD-BEARING (quantified)** | Phase-dependent task on synthesized held-out-band ADC: report **(P1-a)** phase-error-within-X%-of-train-on-real and **(P1-b)** downstream accuracy ≥ Y% of upper bound. Foils: **ADC-magnitude / non-coherent-complex** (primary), PAS-magnitude (trivial anchor). Win = "*correct* phase," not "has phase." | All three + the phase-capable foil | mmIR sim (clean phase GT) + single-radar multi-carrier (R1) | Cross-hardware decks only if raw access lands |
| **E2 — Phase fidelity vs band gap** | Synthesized vs GT complex-CIR **phase error as a function of band gap Δf**, with **downstream accuracy plotted against that phase error**. Sets/freezes X,Y. PAS-SSIM reported alongside. | All three (a curve they cannot produce) | Single-radar multi-carrier + sim | — (no phase fidelity on cross-hardware) |
| **E3 — Extrapolation: physics vs interpolation** | Band-gap sweep vs ported XBandGS head **and vs E8 phase-capable baseline**; win = NBS graceful, learned interp collapses, **especially across `4π·σ_h/λ≈1`**. | XBandGS + phase-capable competitor | Static multi-material scene (sim / ColoRadar+) | Larger gaps spanning the crossing |
| **E4 — Downstream accuracy-recovery** | Train on real band A; test on A synthesized from B; gap to (i) train-on-real upper bound, (ii) nearest-band, (iii) per-band-independent, (iv) E8. | All three (none has a downstream task) | Single-radar multi-band / sim | Cross-hardware transfer (labeled "may be ill-posed" + negative control) |
| **E5 — Material-identity invariance** | Optimize materials at A → continue to B vs materials independently optimized at B. Agreement = transfer; disagreement = honest residual. | XBandGS | Static multi-material (sim GT + ColoRadar+) | — |
| **E6 — Engine-to-data audit (REPORTED, gating)** | State per dataset whether mmIR/3DPS *can* render it: **monostatic FMCW (renderable)** vs **comm-CIR (not, e.g. OPERAnet)**. Output is a table in the paper; the dataset table inherits its verdicts. | Blocking gate (gates R1) | n/a | n/a |
| **E7 — External validity + analytic-vs-residual split** | (a) Material transfer on **2–3 distinct static multi-material scenes** (not one) to show recovery generalizes, not overfits. (b) **Sensitivity ablation isolating closed-form ITU vs learned regime residual across the crossing**, on scenes that actually span it. | XBandGS + generalization reviewers | ≥3 sim scenes + ColoRadar+ rooms | — |
| **E8 — Fair phase-capable baseline** | Frequency-conditioned **complex** field / complex-3DGS that **emits phase** but lacks analytic dispersion. The win must be the dispersion/regime handling, not "we output complex." | The "strawman" objection directly | Same substrate as E3 | — |
| **G-ITU — Differentiable dispersion demo (GATE)** | End-to-end: make `(a,b,c,d)` grad-enabled, wire into mmIR autograd, run a **working A→B material-transfer optimization on a toy scene**, re-render raw ADC at held-out carrier with **quantified phase error**. PASS by early July or Pillar-2 headline drops to fallback. | The core unbuilt-mechanism objection | mmIR toy scene | — |

**Cut from this cycle:** plenoptic view×band (≤1 corollary fig); PAS-SSIM as a target; cross-hardware decks as any headline.

---

## Datasets (clean-vs-stretch tiers; E6 verdicts inlined)

### Tier 0 — Clean / headline (must exist by early July)
- **mmIR simulator.** GT materials, GT phase, arbitrary carriers. Substrate for E1 (first), E2, E3, E5, E7, G-ITU. **E6: renderable (it is the engine).**
- **Single-radar multi-carrier in-house capture (R1 primary fix).** Reconfigure **TI IWR6843 + DCA1000** across center-frequency / bandwidth sub-bands on a *static rig* with LiDAR/RGB-D. **Cross-band phase coherence is NOT automatic** across distinct center frequencies on one IWR6843 — it must be engineered and measured (see calibration below). Present-tense "phase-coherent" is used **only after** the calibration validation passes. **E6: renderable (monostatic FMCW raw ADC).**
- **ColoRadar+** (LiDAR + RGB-D + mmWave, static multi-material rooms). For E3/E5/E7 material claims. Its "two bands" are two **~77 GHz configs**, NOT distinct carriers — usable for static multi-material claims, **not** as the distinct-carrier substrate. **E6: renderable (monostatic).**

**Cross-band phase-coherence calibration procedure (R1, concrete).** (1) **Shared clock/LO discipline:** drive sub-band chirps from the same reference oscillator; log per-sub-band synthesizer settle. (2) **Per-band delay/phase calibration:** corner-reflector at known range per sub-band; estimate and remove per-band constant phase + group delay. (3) **Coherence validation (a reported metric, not an assumption):** measure residual cross-band phase error on a static reference target; **E1/E2 real-data phase metrics are only reported if this residual is below the X% budget.** If coherence fails, real data downgrades to magnitude-level (E1's magnitude foil still valid) and the phase headline reverts to sim.

### Tier 1 — Stretch (explicitly labeled, optional)
- **XRF55** (WiFi CSI + mmWave IWR6843/DCA1000 + RFID + RGB-D/IR). Raw ADC **pending author contact** — chase this week; downstream-utility stretch only. **E6: mmWave path renderable IF raw ADC obtained.**
- **OPERAnet** (WiFi CFR + UWB CIR + passive-WiFi-radar + Kinect). **Comm-style CIR — NOT monostatic radar; E6 verdict: NOT renderable by mmIR/3DPS.** Excluded from method claims; at most context.
- **MM-Fi** (WiFi CSI + mmWave point-cloud + LiDAR + RGB-D). mmWave is **processed point-cloud, not raw — E6: not raw-renderable.** Stretch context only.

**Hard rules.** Material/phase claims (E3, E5, E7) run **only** on static multi-material scenes — **never** on human bodies. Cross-hardware decks **never** headline and ship with a negative-control "may be ill-posed" characterization.

---

## Baselines

- **E8 — fair phase-capable baseline (NEW, mandatory):** frequency-conditioned **complex** field / complex-3DGS that emits phase but has no analytic dispersion. The extrapolation/regime win must beat *this*, proving the delta is dispersion/regime handling, not complex output.
- **XBandGS physics-MoE head ported into the raw setting:** the power-head limitation (cannot emit phase) is shown but framed honestly as the *power-paradigm* anchor — **not** the primary win (red-team: power head can't emit phase is trivial).
- **Nearest-band copy** (lower bound).
- **Per-band-independent model** (ceiling on "do we need synthesis").
- **Analytic-ITU-only ablation** (NBS without the learned residual) — isolates closed-form vs residual (E7b).
- **ADC-magnitude / non-coherent-complex** (E1 primary foil).
- **PAS-magnitude head of NBS** (trivial anchor).

---

## Evaluation metrics

**Phase-fidelity (the core, quantified):**
- **Phase error vs band gap** (E2): complex-CIR phase RMSE as a function of Δf, with downstream accuracy overlaid.
- **(P1-a)** synthesized-band micro-Doppler/vital-sign error as **% of train-on-real** (target ≤ X%, frozen at gate).
- **(P1-b)** downstream accuracy as **% of upper bound** (target ≥ Y%), **above E8**.

**Metrics PAS competitors cannot report:** complex CIR/ADC correlation (mag + phase); sub-bin range error (cm); phase-only Doppler (Hz) / displacement (mm).

**Cross-band phase-coherence residual (rig validity gate):** measured residual cross-band phase error; real-data phase metrics void if above budget.

**"We don't lose their game":** PAS-SSIM/PSNR alongside, in the interpolation regime (cf. Wideband3DGS 2.8%, XBandGS 7.7% unseen-freq drop). Always split **interpolation vs extrapolation** and name the failure boundary ourselves.

---

## Risk register (honest)

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| **R1** | **No clean real distinct-carrier substrate.** Engines single-band; decks cross-hardware; ColoRadar+ "two bands" are ~77 GHz configs; XRF55 raw pending. | **Critical** | **HARD artifact gate, early July:** a captured, **phase-coherence-calibrated, LiDAR-aligned** in-house multi-carrier set must exist, with the coherence residual measured below budget. If not: paper **explicitly demoted to sim-anchored strong-accept (Tier 2), award language removed from the abstract.** Start the capture in June. |
| **G-ITU** | **Differentiable ITU-dispersion inverse loop does not exist** (`itu_materials.py` is a non-diff lookup; grad path is per-band `eps_r/sigma`, not `a,b,c,d`). | **Critical** | **Demonstrate by early July** a grad-enabled `(a,b,c,d)` wired into mmIR autograd with a working A→B toy-scene transfer + quantified held-out phase error. **If it slips, DROP "analytic ITU continuation" from the headline** and re-scope Pillar 2 to the GGX/IOR/roughness/slab path the repo already differentiates (still a real, novel raw-signal re-render — just not "analytic dispersion"). |
| **R2** | **E1 muddy** → "different decoder head." | **Critical** | Build E1 first on sim; pick a categorically-phase task; commit X,Y at gate; require the *phase-capable* foil. If not clean by mid-July, headline → E3 material-extrapolation. |
| **R3** | **Tautology objection** (phase undefined on PAS by construction). | High | Headline is the **quantitative** P1-a/P1-b result vs a **phase-capable** baseline, never "PAS has no phase." |
| **R4** | **Sim-only loses to same-venue real USRP.** | High | Two explicit stances (with/without R1); never claim real-axis superiority sim-only; frame orthogonal capability. |
| **R5** | **Strawman single baseline.** | High | E8 fair phase-capable baseline is mandatory; XBandGS head is the anchor, not the win. |
| **R6** | **Timeline overload.** | High | One task, one clean substrate, the two mandatory baselines (E8 + ported head), one downstream number; materials/view are supporting. |
| **R7** | **Cross-hardware decks under-determined.** | Med-High | Demote to labeled "may be ill-posed" stretch + negative control; never headline. |
| **R8** | **Material claim overfits one scene.** | Medium | E7a: ≥3 distinct static multi-material scenes. |
| **R9** | **Over-claiming extrapolation** ("why is error > XBandGS 2.8–7.7%?"). | Medium | Honest interp-vs-extrap split; be in their regime for interp, reserve the win for the principled crossing; E7b isolates closed-form vs residual. |
| **R10** | **Engine ≠ dataset** (OPERAnet comm-CIR). | Med-High | E6 reported as a table; OPERAnet excluded from method claims. |

---

## Timeline (toward Sep 4, 2026)

- **Jun, week 1 (NOW):** **E6 audit** (renderability table). Start **in-house multi-carrier capture** + **build the cross-band phase-coherence calibration** (corner reflector, per-band delay). Begin **G-ITU** implementation (grad-enable `(a,b,c,d)`, wire to autograd). Email XRF55 authors. Stand up mmIR with GT materials.
- **Jun:** Build **E1 on sim first**; pick the phase task; implement **E8 fair phase-capable baseline** + port the XBandGS head.
- **Early July — GATE 1 (R1 + G-ITU):** Is a calibrated real distinct-carrier capture in hand (coherence residual measured)? Is G-ITU demonstrated on a toy scene? **R1 no → Tier-2 sim-anchored, award language removed. G-ITU no → drop analytic-ITU from headline, use fallback Pillar 2.** Freeze X, Y from the E2 sim sweep.
- **Mid-July — GATE 2 (R2):** Does E1 hold cleanly (P1-a/P1-b met vs the phase-capable foil)? **No → headline switches to E3.**
- **Jul:** E2 (phase-vs-band-gap), E3 (extrapolation vs E8 + ported head), E5 (invariance), E7 (multi-scene + analytic-vs-residual split) on clean substrates.
- **Aug:** E4 downstream recovery; labeled cross-hardware stretch; ablations. **Freeze headline by mid-Aug.**
- **Aug 28 / Sep 4:** abstract / paper. One corollary view×band figure if time; else future work.

---

## Rollback (descope ladder)

1. **Best-paper target (conditional):** E1 clean on **real** in-house multi-carrier (coherence validated) + G-ITU-backed E3 regime-change win + E4 recovery. **Requires R1 AND G-ITU by early July.** Only here is award language used.
2. **R1 slips (real data):** prove E1/E2/E3/E5/E7 on **mmIR sim + ColoRadar+**. Honest **sim-anchored strong-accept**, framed as a capability axis **orthogonal to** XBandGS — not superior on real data. Award language removed.
3. **G-ITU slips (mechanism):** keep raw-signal capability; Pillar 2 = differentiable **per-band physics-material** re-render through the wavelength-dependent forward model + regime residual (no "analytic dispersion" claim). Still strong-accept.
4. **R2 fails (E1):** drop the capability headline; lead with **E3 material-extrapolation vs E8/ported head through the regime change** (the em-physics thesis). Defensible strong-accept on sim + ColoRadar+.
5. **Engineering rollback:** modules independent; flags disable the learned residual (→ analytic/physics-only) or the phase head (→ magnitude foil); per-run `config.json`; mmIR/3DPS swappable per E6.

**Bottom-line feasibility verdict.** The thesis is ambitious and the *capability axis* is structurally un-rebuttable; the risks are concentrated in two demonstrable artifacts — **R1 (a calibrated real distinct-carrier capture)** and **G-ITU (a differentiable dispersion inverse loop)** — both gated to early July. Award language is **earned, not asserted**: used only if both pass. Otherwise the paper ships as an honest sim-anchored strong-accept on an axis XBandGS cannot enter, with every mechanism claim scoped to demonstrated code.

---

## Changelog vs PLAN.md
- Replaced the four-differentiator pitch with two quantified survivors (raw phase-coherent synthesis + a phase-dependent capability with committed targets P1-a/P1-b); retired real-data, geometry-anchoring, physics-vs-blackbox as headlines (XBandGS preempts them).
- Restated Pillar 1 from "PAS cannot represent phase" (cut as tautology) to a measured **phase-fidelity-at-an-unseen-carrier** result against a **phase-capable** foil (ADC-magnitude / non-coherent-complex), not just PAS.
- Scoped Pillar 2 to demonstrated code: removed "ITU-dispersive materials" from the headline; added **G-ITU** as a hard gate (the `(a,b,c,d)` dispersion loop is unbuilt; only per-band `eps_r/sigma` are differentiable in-repo), with a GGX/IOR/roughness fallback.
- Added E6 (renderability audit) and E7/E8 (multi-scene external validity, analytic-vs-residual split, fair phase-capable baseline) as in-paper gating/experiments; conditioned best-paper/award language on the R1 real-data gate; specified a concrete cross-band phase-coherence calibration + validation procedure.

## How the area-chair committee shaped this
- **Held the line on ambition while excising overclaim:** the committee kept the categorical raw-signal capability as the thesis but forced every mechanism claim to be scoped to in-repo, source-verified code (differentiable per-band physics materials exist; the dispersion coefficient gradient path does not) — hence the G-ITU gate rather than an asserted lever.
- **Converted feasibility risk into dated, falsifiable artifacts:** R1 (calibrated real distinct-carrier capture) and G-ITU (toy-scene dispersion inverse loop) become early-July go/no-go gates with explicit demotions, replacing "data is decided" optimism.
- **Made the delta vs XBandGS reviewer-proof:** added a fair phase-capable baseline (E8) so the win is "correct phase / analytic dispersion," not "we emit complex"; and confronted the sim-vs-real asymmetry directly (orthogonal-axis framing when real data is absent).
- **Set award language as conditional, not aspirational:** best-paper framing is permitted only if both hard gates pass; otherwise the plan self-demotes to an honest sim-anchored strong-accept.
