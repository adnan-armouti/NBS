# NBS Positioning Memo — pressure-testing the novelty against three cross-band RF papers

> **Status: memo for review. PLAN.md is NOT edited yet** (proposed edits are quoted in §6). Target venue: MobiCom (NBS = top-priority submission).
>
> **Headline finding (read this first):** "Scanned Document 7" is **XBandGS — "Cross-Band RF Radiance Field Modeling via 3D Gaussian Splatting and Propagation Mixture-of-Experts," a MobiCom'26 submission (Oct 2026, Austin).** It is **not** a black-box field interpolator. It is **physics-guided** (Fresnel reflection, UTD diffraction, diffuse scattering, free-space path loss), it uses a **shared 3D geometric scaffold across bands**, and — critically — **it has real measured data (USRP, 0.7–5.8 GHz, 2 indoor + 2 outdoor scenes).** This single paper independently invalidates three of the four differentiators PLAN.md currently leans on (real data, geometry-anchoring, "physics-grounded vs black-box"). The two that survive are **raw-signal-level synthesis (ADC/CIR)** and **a downstream sensing task**. Everything below argues NBS must re-center on those two, plus the radar/inverse-rendering framing that our in-repo 3DPS/mmIR engines uniquely enable.

---

## 1. One-paragraph summary of each paper

**(P1) XBandGS — Cross-Band RF Radiance Field Modeling via 3DGS + Propagation MoE (MobiCom'26 submission; the scanned PDF).**
Given a 3D scene with a fixed multi-antenna Rx and a moving Tx, XBandGS learns a single wideband model that predicts the **Power Angular Spectrum (PAS)** at the Rx for arbitrary Tx pose *and arbitrary unseen frequency*. The scene is a set of 3D Gaussians whose **geometry {μ, Σ} is shared (frequency-invariant) across all bands**; band-to-band variation is pushed into a **physics-guided Mixture-of-Experts** with four EM experts — LoS (free-space path loss, Eq. 18), reflection (Fresnel Γ(θ,f;ε_r,μ_r) with *learned effective material params ε̂_r, σ̂*, Eqs. 19–20), scattering (Eq. 21), and diffraction (UTD/knife-edge D(ν,λ), Eqs. 23–24) — gated by a frequency- and Tx-geometry-aware network with physics-prior scores (Eqs. 12–16). Frequency is hybrid-encoded (Fourier + log f + 1/f + learnable per-band embedding, Eq. 6). Experts are pre-trained on Sionna mechanism-isolated data (LoS-only, reflection-only, …) then jointly fine-tuned. Rendering is tile-based differentiable splatting onto an angular hemisphere with a frequency-conditioned **residual** modulation of per-Gaussian attenuation/intensity (Eqs. 28–30); loss is SSIM on the PAS image. **Data: Sionna sim (1–94 GHz, 5 indoor scenes) + real USRP measurements (USRP-N210 Tx on a robot, USRP-2974+B210 2×2 Rx; 0.7–5.8 GHz; 10 frequencies; 2 indoor + 2 outdoor scenes).** Results: SSIM **0.880 sim / 0.711 real**; at *unseen* frequencies only **~7.7% SSIM / 6.3% PSNR** degradation vs a model trained on the full band; beats NeRF2, WRFGS+, GSRF. **Cannot do:** raw signal/phase (PAS is a power-magnitude image; Bartlett beamforming discards complex CIR), no downstream sensing task, geometry is RF-derived (no optical/LiDAR anchor — they explicitly *reject* optical anchoring à la RF-3DGS on privacy grounds), single Tx→Rx communication-link framing (not monostatic radar imaging of the scene).

**(P2) Wideband3DGS — Wideband RF Radiance Field via Frequency-embedded 3DGS (arXiv 2505.20714).**
Frequency-embedded 3D Gaussian Splatting that models a **wideband RF radiance field / PAS** as a function of position and frequency, so one model renders the field at untrained frequencies. **Data: Sionna-simulated, 1–100 GHz, 6 indoor scenes, point-cloud-initialized Gaussians.** Reports only a **~2.8% drop** predicting at untrained frequencies. **Pure field/channel reconstruction — simulated only, no real measurement, no raw signal, no downstream sensing task.** It is the "fit-the-field-and-interpolate-frequency" baseline; less physics machinery than XBandGS.

**(P3) XFreq-GS — Cross-Frequency Wireless Radiation Field Reconstruction with 3DGS (arXiv 2605.11432; Sheng Wang, Hengtao He, et al. — a *different* group from XBandGS).**
3D Gaussian primitives with **shared geometry + frequency-adaptive RF attributes** that reconstruct the **PAS** across multiple carrier frequencies, trained on sparse frequencies and predicting unseen ones, framed for wireless-channel applications (resource allocation, cross-link prediction). **Appears simulated** (3GPP/ray-traced; no USRP campaign found — *flag: I could not fully extract the experiments PDF, low confidence on real-vs-sim*). **No downstream sensing task; PAS-level, not raw signal.** Conceptually the closest sibling of XBandGS but without the explicit 4-mechanism physics-MoE.

**Relationship between the three.** All three are the *same paradigm*: an explicit 3DGS (or radiance-field) scene representation, **shared geometry across bands**, **frequency-conditioned per-primitive RF attributes**, output = **PAS / power radiance field**, evaluated by **image-similarity (SSIM/PSNR/MSE) on the field**, framed as **communication / channel / digital-twin**. XBandGS is the strongest (physics-MoE + real data). They differ from NBS-as-planned far less than PLAN.md assumes.

---

## 2. Comparison table

| Axis | **Wideband3DGS** (P2) | **XFreq-GS** (P3) | **XBandGS** (P1, MobiCom'26) | **NBS (proposed)** |
|---|---|---|---|---|
| Real vs simulated data | Sim only (Sionna) | Sim (likely; low conf.) | **Sim + REAL (USRP, 0.7–5.8 GHz, 4 scenes)** | Real (OPERAnet/MM-Fi/XRF55) + sim |
| Signal level: raw ADC/CIR vs power/field | PAS / power field | PAS / power field | PAS (beamformed power image) | **Raw complex ADC / CIR / complex range profiles** |
| Geometry-anchored to a *real optical* scene? | Point-cloud-init (sim) | Shared geom (RF-derived) | Shared 3DGS scaffold, **RF-derived (rejects optical anchor)** | **Real LiDAR / RGB-D optical anchor** |
| Physics forward-model vs learned-field | Learned field + freq embed | Learned field, freq-adaptive | **Physics-guided MoE (Fresnel/UTD/scatter/PL) over learned field** | **Differentiable physics radar forward model + inverse rendering of materials** |
| Interpolation vs extrapolation across bands | Interp (2.8% drop) | Interp | Interp (7.7% SSIM drop unseen) + claims extrap | Interp first; extrap = labeled stretch |
| Downstream sensing task? | **None** | **None** | **None** (comm/digital-twin) | **Yes — HAR/pose/occupancy/3D-recon on synthesized band** |
| Problem framing | Comm channel/field | Comm channel/field | Comm channel/field (Tx→Rx link) | **Sensing / scene & material reconstruction (radar imaging)** |
| What is recovered | Per-Gaussian field attrs | Per-Gaussian field attrs | Per-Gaussian power coeffs (+ latent ε̂_r) | **Transferable per-point ITU materials (ε, roughness, slab)** |
| Evaluation metric | SSIM/PSNR on PAS | SSIM on PAS | SSIM/PSNR/MSE on PAS | Complex-signal fidelity (range/Doppler, phase) **+ downstream task accuracy** |

---

## 3. The white space NBS can own — and an honest risk read

### What is genuinely still open (argued from the table)
Reading down the table, the columns where **all three competitors are identical and NBS is alone** are exactly two rows, plus a framing row:

1. **Raw complex signal synthesis (ADC / CIR / complex range profiles), not power/PAS.** Every competitor outputs a *power-magnitude angular image* and scores it with SSIM. Bartlett beamforming → PAS **throws away the complex fast-time/phase structure** that ranging, micro-Doppler, and most radar sensing need. NBS, via mmIR/3DPS, renders the **complex ADC** (Eqs. in mmIR: φ = 2π(S·τ·t + f₀·τ), per-sample cosine modulation). Synthesizing the raw signal at a new band is **strictly harder and strictly more useful**, and is something a PAS pipeline structurally cannot be asked to do.

2. **A downstream sensing payoff.** None of the three runs a sensing task. NBS's "train a band-A sensing model, feed it band-A *synthesized from band-B*, compare to the train-on-real upper bound" is a payoff axis the entire competing line lacks. It also reframes NBS from a *channel-modeling* paper into a *sensing* paper — a different reviewer pool and a different bar.

3. **Sensing/inverse-rendering framing (recover the scene, not the field).** Competitors fit a *field* (the channel between one Tx and one Rx) and interpolate it in frequency. NBS **inverts a physics radar forward model to recover band-conditioned scene materials** (permittivity, roughness, slab thickness — real physical quantities, anchored to *real optical geometry*), then *forward-renders the raw signal at a new band*. This is inverse graphics / material recovery, not field fitting. It generalizes to non-communication, non-human scenes (a wall, a room) where "PAS at an Rx array" is not even the quantity of interest.

**The ownable claim = the intersection of (1)+(2)+(3):** raw-signal novel-*band* synthesis for *sensing*, via differentiable physics-based inverse+forward rendering on real optically-anchored geometry. That intersection is empty in the literature today.

### Honest overlap / risk read (where we still collide)
- **"Real measured data" is no longer a differentiator.** XBandGS has real USRP data across 10 bands and 4 scenes, including outdoor. Delete this from the headline. (PLAN.md §"Positioning update" bullet 1 and the whole framing in lines 21–26 are now wrong on this point.)
- **"Physics-grounded vs black-box interpolation" is dead as stated.** XBandGS is *more* physics-instrumented than what NBS-at-the-field-level proposed (4 analytic EM experts + physics-prior gating + path-loss decoupling, with an ablation showing each matters). The candidate angle in the brief — "physically-grounded band synthesis through a shared 3D scene + EM forward model, band-conditioned material/scattering" — is **almost verbatim what XBandGS does.** We cannot claim physics-grounding as novel; we can only claim a *different kind* of physics (a **radar imaging forward model that produces the raw signal**, vs a **PAS field regressor**).
- **"Geometry-anchored" is half-dead.** XBandGS shares a 3D geometric scaffold across bands. NBS's surviving sliver is *optical* anchoring (LiDAR/RGB-D) to **true scene geometry + physical materials** — but XBandGS will argue optical anchoring is a *liability* (privacy, needs a camera), so this is a contested, not a clean, win. Don't lead with it.
- **The raw-signal claim has a rebuttal we must pre-empt.** A reviewer will say "PAS vs ADC is just an output choice; beamform a complex field and you have the signal." Rebuttal: the competitors never produce a phase-coherent complex field per fast-time sample — they regress a power image and supervise with SSIM on magnitude; recovering complex CIR at a held-out band with correct phase/delay is the hard part and is what enables ranging/Doppler. We must *show* this, not assert it.
- **The downstream-task differentiator rests on the weakest data.** The deck datasets (MM-Fi/OPERAnet/XRF55) are human-activity sets with **non-co-located, different-hardware, often-processed** bands. Cross-modality raw synthesis (WiFi CIR ↔ mmWave ADC) may be ill-posed at the sample level (PLAN.md already flags this). So our *strongest* novelty leans on our *riskiest* experiment. Have the controlled-simulator and single-radar-multi-band (ColoRadar+ two configs) fallbacks ready as the "clean" version of the claim.
- **We can't trivially "beat XBandGS on their turf."** Their turf is PAS reconstruction with a moving Tx / fixed Rx array — a comm link, not monostatic radar. Re-implementing that to win SSIM is a distraction and plays to their strength. Better: **port their physics-MoE idea into the raw-signal/radar setting as *our* baseline** and show it loses the phase/sensing information that NBS keeps.
- **Naming/landscape risk:** XBandGS (P1) and XFreq-GS (P3) are *two different groups* converging on the same idea in the same cycle. Cross-band RF radiance fields are now a crowded, hot subfield. The bar for "first/novel" is high; lean on the sensing + raw-signal axis where the crowd is absent.

---

## 4. The single crisp novelty sentence

> **Unlike Wideband3DGS, XFreq-GS, and XBandGS — which fit a power-level RF radiance field (PAS) of a communication channel and interpolate it across frequency — NBS is the first to synthesize the *raw complex radar signal* (ADC/CIR) at an unseen band by inverse-rendering band-conditioned scene *materials* through a differentiable physics-based *radar* forward model anchored to real optical geometry, and the first to validate cross-band synthesis on a downstream *sensing* task.**

Compressed alternative for the abstract:

> *Prior cross-band RF radiance fields render a power angular spectrum and interpolate it across frequency; NBS renders the raw radar signal and transfers it across frequency — by recovering physical scene materials, not by fitting a field — and proves it on a downstream sensing task.*

---

## 5. Minimal experiments/claims to defend the sentence against each competitor

**Defends raw-signal claim (vs all three).**
- E1. **Raw complex synthesis with phase fidelity.** Hold-one-band-out; report complex CIR/ADC correlation, range-profile error, and a phase/delay-sensitive metric (range or Doppler error). Show competitors *cannot* report these because PAS discards them — and that NBS's own power-projection still matches them on SSIM (so we don't lose the field metric while gaining the signal).

**Defends downstream-sensing claim (vs all three).**
- E2. **Train-on-real vs train-on-synthesized.** Train a sensing model (HAR / pose / occupancy / 3D-recon) on real band A; test on band A **synthesized from band B**; compare to (i) train-on-real upper bound, (ii) nearest-band copy, (iii) per-band independent model. The gap to the upper bound is the headline number. Run the *clean* version on the controlled simulator and on single-radar multi-band (ColoRadar+) first; treat cross-hardware deck datasets as the stretch.

**Defends inverse-rendering / material claim specifically vs XBandGS (P1).**
- E3. **Material transfer beats field interpolation.** Re-implement XBandGS's physics-MoE *field-interpolation* head as a baseline in our radar/raw setting; show that recovering transferable ITU materials (mmIR) and re-rendering at the held-out band beats interpolating per-Gaussian attributes — especially at larger band gaps / extrapolation, where field interpolation should degrade and a physical forward model should hold.
- E4. **Optical-anchor generalization.** Show NBS works on a **non-human, non-comm scene** (ColoRadar+ room/mapping) where "PAS at an Rx array" is not the target — demonstrating the sensing/scene-recon framing is genuinely outside the competitors' problem definition.

**Defends against the "PAS is enough" rebuttal.**
- E5. **Ablate the output representation.** Same model, two heads (PAS-magnitude vs complex-ADC); show the downstream sensing task that *succeeds on ADC fails on PAS-magnitude* (e.g., anything needing range/Doppler/phase). This is the load-bearing experiment for the whole positioning.

**Scope honesty (state explicitly in the paper).** Interpolation is the claim; extrapolation is a labeled stretch (matches XBandGS's own framing and Wideband3DGS's 2.8% / XBandGS's 7.7% interpolation-regime numbers — be in the same regime, don't over-claim).

---

## 6. Concrete proposed edits to PLAN.md (quote current → new)

> **Not applied yet — for your review.** These target the `## ⚠️ Positioning update` section (PLAN.md lines 15–27) and the one-line goal (line 13).

**Edit A — fix the factual error about the competitors (real data + physics).**

*Current (lines 21–26):*
> Both are **communication/propagation** papers (predict channel / power-angular-spectrum), **simulated**, with **no real geometric modality** and **no downstream sensing task**. That is exactly where NBS must differentiate to win best paper:
> 1. **Real measured data**, not Sionna ray-tracing.
> 2. **Raw-signal-level** synthesis (ADC / CIR), not just power maps.
> 3. **Anchored to a real geometric scene** (LiDAR / RGB-D) — a *sensing/scene-reconstruction* framing, not a comm-channel framing.
> 4. **A downstream-task payoff** …

*Proposed:*
> **Correction (Jun 2026):** "Scanned Document 7" is **XBandGS**, a **MobiCom'26 submission** (our venue) that does cross-band PAS via a **physics-guided MoE (Fresnel/UTD/scatter/path-loss)** on a **shared 3D geometric scaffold**, with **real USRP data (0.7–5.8 GHz, 10 freqs, 2 indoor + 2 outdoor)** — SSIM 0.880 sim / 0.711 real, ~7.7% drop at unseen freqs. This **kills three of our four planned differentiators**: real data, geometry-anchoring, and "physics-grounded vs black-box." Only two survive, and they are now the whole pitch:
> 1. **Raw complex-signal synthesis (ADC/CIR/complex range profiles), not PAS/power.** All three competitors output a power-magnitude angular image (SSIM-scored); beamforming discards the phase/fast-time structure sensing needs.
> 2. **A downstream sensing-task payoff** (HAR/pose/occupancy/3D-recon on the synthesized band) — none of the three has one; all are comm/channel/digital-twin.
> 3. *(supporting, not headline)* **Inverse-rendering of transferable physical materials via a differentiable radar forward model** (3DPS/mmIR), anchored to **real optical geometry** — a sensing/scene-reconstruction framing, vs their field-fitting/channel framing. Note XBandGS deliberately avoids optical anchoring (privacy), so this is a contested, not clean, win — don't lead with it.

**Edit B — sharpen the one-line goal to the surviving axes.**

*Current (line 13):*
> Treat the **RF-frequency axis $f$** as a synthesizable dimension: train on $K$ wireless bands, **render the scene's signal at a held-out band** at the raw-signal level, via band-specific encoders → a shared 3D wireless scene representation (geometry-anchored) → a physics-informed decoder that renders at query frequency $f_q$. Primary claim = **interpolation**; extrapolation = stretch.

*Proposed:*
> Treat the **RF-frequency axis $f$** as a synthesizable dimension for **sensing**: train on $K$ bands and **render the held-out band's *raw complex radar signal* (ADC/CIR)** — not a power/PAS field — by **inverse-rendering band-conditioned scene materials** through a differentiable physics-based radar forward model (3DPS/mmIR) anchored to **real optical geometry**, then validating the synthesized band on a **downstream sensing task**. Primary claim = **interpolation**; extrapolation = stretch. *Differentiator vs Wideband3DGS / XFreq-GS / XBandGS: raw signal + sensing payoff + material recovery, not PAS-field interpolation.*

**Edit C — add a baseline.**

*Current (line 79):*
> - **Wideband3DGS** and **XFreq-GS** (cross-frequency WRF) — the direct competitors; reproduce on sim, then show our real-data / raw-level / geometry-anchored advantage.

*Proposed:*
> - **XBandGS** (MobiCom'26, physics-MoE + real USRP), **Wideband3DGS**, **XFreq-GS** — the direct competitors. **Do not chase them on PAS-SSIM (their turf).** Instead, **port XBandGS's physics-MoE as a field-interpolation baseline in our raw-signal/radar setting** and show it loses phase/sensing information; lead with raw-signal fidelity (E1), downstream-task gap (E2), and material-transfer-beats-interpolation (E3).

**Edit D — re-rank the top risk.**

*Current (line 92):*
> - **Novelty erosion** from Wideband3DGS / XFreq-GS — *the* top risk now. Mitigation: real + raw + geometry-anchored + downstream task. **This must be airtight in the intro.**

*Proposed:*
> - **Novelty erosion** — *the* top risk, now led by **XBandGS (same venue, physics-MoE, real data)**, then XFreq-GS / Wideband3DGS. Mitigation is **only** raw-signal + downstream-sensing + material-inverse-rendering (real data and "physics-grounding" are no longer ours to claim). The load-bearing experiment is **E5** (a sensing task that succeeds on synthesized ADC but fails on PAS-magnitude). This must be airtight in the intro.

---

## Appendix — key extracted facts (for citation in the paper)

- **XBandGS:** MobiCom'26, Austin. 3DGS shared geometry + freq-conditioned MoE (LoS Eq.18, Reflection Eqs.19–20 w/ learned ε̂_r,σ̂, Scattering Eq.21, Diffraction Eqs.23–24); physics-prior gating Eqs.12–16; hybrid freq encoding Eq.6 (Fourier+logf+1/f+embed); tile rasterizer to angular hemisphere Eqs.28–30; loss = 1−SSIM (Eq.31) + MoE load-balance/entropy (Eq.32). Data: Sionna 1–94 GHz × 5 indoor; **real USRP-N210 Tx (robot) + USRP-2974+B210 2×2 Rx, 0.7–5.8 GHz, 10 freqs, 2 indoor + 2 outdoor.** Baselines NeRF2/WRFGS+/GSRF. SSIM 0.880 sim / 0.711 real; unseen-freq drop ~7.7% SSIM, 6.3% PSNR, 16.9% MSE. Ablation: single-expert and PAS-level-fusion both hurt → MoE + feature-level fusion matter. Train 7.5 h/scene, infer 21.9 ms. **No raw signal, no downstream task, RF-only geometry.**
- **Wideband3DGS (2505.20714):** freq-embedded 3DGS, PAS/radiance field, Sionna 1–100 GHz × 6 indoor, point-cloud init, ~2.8% unseen-freq drop. Sim only, no raw signal, no downstream task.
- **XFreq-GS (2605.11432):** 3DGS shared geometry + freq-adaptive attrs, PAS, cross-frequency, comm framing. Likely sim (low confidence — experiments PDF not fully extracted). No MoE confirmed, no raw signal, no downstream task. Different author group from XBandGS.
- **In-repo engines (our lever):** **3DPS** = differentiable point-splatting *forward* renderer for 77 GHz cascaded radar RA imaging (NVS, single band, ColoRadar; complex range profiles → RA; cart_corr metric; ITU 6-param materials; ~3 min/scene). **mmIR** = physics-based differentiable FMCW radar renderer (multi-bounce, SMS, free-space diffraction, Jones polarization, frequency-dependent BSDF, **raw ADC output** via φ=2π(S·τ·t+f₀·τ); hardware-accurate scaling). Together they give NBS what the competitors lack: a **raw-signal, material-recovering, radar forward model** to invert per band.
