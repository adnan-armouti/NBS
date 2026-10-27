# G-ITU gate result: differentiable ITU-dispersion material recovery + cross-band transfer

> **CORRECTION (Jun 2026, audit — `md/method_audit.md`).** All **RA/CSI correlation** numbers
> below are **geometry-dominated and invalid** (wrong materials score 0.99+); ignore them. The
> **interpolation baselines were not validated to be non-degenerate** and at least some clearly
> collapsed (roughness interp corr 0.47 is below the geometry floor → broken), so the
> **"physics beats field interpolation" (E3) claims here are NOT reliably demonstrated** and are
> retracted pending a re-run with (a) a wrong-material floor control and (b) a verified baseline.
> What is supported: the differentiable dispersion/roughness inverse loop **converges through the
> real BSDF** (the mechanism works). **UPDATE:** a corrected re-run (`method/eval_v2.py`, two
> metrics + validated baseline) **RE-VALIDATES "physics beats interpolation" at the roughness
> regime change** — NBS 100% material-ID, −30.8 dB (25 dB below the wrong-material floor), validated
> interp fails (+0.3 dB). See `md/method_audit.md` (RE-RUN section). The claim holds *scoped to where
> materials matter*; the corr-based numbers below remain invalid.


**Gate question (PLAN_v2):** can we make the ITU dispersion law `eps_r(f)=a·f^b`,
`sigma(f)=c·f^d` *differentiable*, recover the coefficients by back-propagating a
complex-signal loss, and analytically continue them to an unseen band? (The in-repo
`mmIR/itu_materials.py` is a non-differentiable float lookup — this was the unbuilt
mechanism.)

**Setup** (`reference_code/NBS/method/`): a 5-scatterer toy scene with KNOWN geometry
(the scaffold) and ITU-R P.2040 materials (concrete/glass/wood/plasterboard/brick).
We render the complex FMCW ADC at training bands, then compare two ways to predict a
held-out band:
- **NBS** — recover per-scatterer `(a,b,c,d)` by back-prop through the differentiable
  forward model (`freq_material.render_adc`), analytically continue to the held-out band.
- **Interpolation baseline** — recover per-band complex permittivity independently, then
  linearly interpolate/extrapolate `eps(f)` to the held-out band (the field-interpolation
  analogue of XBandGS / XFreq-GS / Wideband3DGS).

**Result — PASS.**

| held-out | NBS signal err (dB) | interp baseline (dB) |
|---|---|---|
| 30 GHz | **−96.6** | −70.2 |
| 45 | **−85.2** | −54.4 |
| 55 | **−82.1** | −47.0 |
| 65 | **−79.9** | −42.7 |
| 77 | **−78.0** | −39.2 |
| 90 | **−76.4** | −36.3 |
| 100 | **−75.4** | −34.5 |

(train bands = {10, 24, 40} GHz; complex-ADC correlation = 1.0 for both, because the
*known geometry* fixes the carrier phase — the dB error is the discriminating metric on
*material* fidelity.) Figure: `reference_code/NBS/output/gitu_gate/band_gap_sweep.png`.

**Reading.** Analytic dispersion continuation holds the raw-signal error flat (−75 to
−96 dB) across the whole 30–100 GHz span; field interpolation degrades monotonically with
band distance (down to −34 dB), and the gap *widens* the farther we extrapolate. This is
the E3 thesis (physics-material recovery beats field interpolation, especially in
extrapolation) demonstrated end-to-end, and it keeps "analytic dispersion continuation" in
the paper's headline.

**Caveats.** This first toy is idealized: point scatterers, known geometry, noiseless,
single antenna, normal incidence, and the interpolation baseline was unconstrained (it
produced unphysical eps under extrapolation, which inflated the gap).

---

## Hardened gate (real BSDF + single-chip array + noise + |RA|)

`method/ra_forward.py` + `method/gitu_gate_ra.py` redo the gate with: the REAL 3DPS ITU
slab Fresnel (`bsdf_torch.itu_slab_fresnel` + new `set_frequency`), the actual single-chip
(IWR6843 / XRF55) array geometry (Tx{0,2}×4Rx = 8-antenna 0.5λ ULA), RA via the repo's
new `adc_to_ra_complex_single_chip` (64 azimuth bins, matching XRF55), **20 dB measurement
noise**, and **|RA| supervision** (so the relative inter-antenna phase is exercised). The
interpolation baseline is kept physical (eps clamped). Train {10,24,40} GHz, held-out 30→160.

| held-out | NBS |RA| err (dB) | interp (dB) | NBS corr | interp corr |
|---|---|---|---|---|---|
| 30 (interp) | −5.9 | **−11.4** | 0.9994 | 0.9993 |
| 45 (interp) | −5.2 | **−8.6** | 0.9986 | 0.9994 |
| 60 | −5.2 | −5.9 | 0.9997 | 0.9893 |
| 77 | −5.1 | −5.6 | 0.9992 | 0.9934 |
| 100 | **−4.6** | −4.6 | 0.9992 | 0.9957 |
| 130 | **−4.5** | −4.1 | 0.9993 | 0.9954 |
| 160 (4× train) | **−4.4** | −3.5 | 0.9985 | 0.9892 |

**Honest reading (this matters for positioning).**
- The **mechanism works**: differentiable ITU-dispersion recovery converges through the
  *real* BSDF on noisy |RA| (corr ≈ 0.999) and analytically continues to unseen bands.
  **G-ITU mechanism: PASS.**
- The **advantage over (physically-constrained) interpolation is ROBUSTNESS, not magnitude.**
  NBS error/corr are *flat* across 30–160 GHz; interpolation is actually *better* inside the
  training span (denser fit) but degrades and grows erratic under extrapolation, crossing NBS
  around ~100 GHz (~2.5× train max). The realistic gap is ≤ ~1 dB, **not** the toy's 40 dB —
  that gap was an artifact of noiselessness + unphysical interpolation.
- In this regime the error is **noise- and geometry-limited** (corr saturates because known
  geometry fixes the RA peak locations; ~−5 dB is the 20 dB-SNR material-recovery floor), so
  the experiment does not cleanly separate the two on magnitude.

**Implication.** "Analytic dispersion continuation" is defensible as a **band-gap-robust**
property, but it is NOT a large, headline-carrying win over interpolation in the noisy,
physically-constrained regime. The paper's primary differentiator stays the **raw-signal +
downstream-sensing** axis vs PAS (positioning.md), with dispersion as a robustness argument.

---

## Where physics genuinely wins: the smooth→rough regime change (E3, demonstrated)

`method/gitu_roughness.py`. Surfaces are SMOOTH in the training bands and cross into the
ROUGH regime at the held-out bands (`4π·σ_h/λ`: 0.08–0.82 at train {10,18,28} GHz →
1.3–4.7 at 160 GHz). The Rayleigh factor `exp(−½(4π·σ_h·cosθ/λ)²)` makes the coherent
reflectivity roll off NONLINEARLY across the crossing. Fair comparison (both fit the
COMPLEX RA at 20 dB noise):
  - **NBS** recovers physical dispersion `(a,b,c,d)` AND roughness `σ_h`, re-renders.
  - **interp** recovers the per-scatterer complex reflectivity *attribute* per band
    (the XBandGS-style learned field attribute) and interpolates it across frequency.

| held-out | NBS |RA| dB | interp dB | NBS cx-corr | interp cx-corr |
|---|---|---|---|---|---|
| 28 (in-span, smooth) | −34.2 | **−43.7** | 0.999 | 1.000 |
| 40 | −19.4 | −16.0 | 0.993 | 0.990 |
| 60 | **−15.8** | −8.3 | **0.987** | 0.944 |
| 90 | **−10.0** | +0.3 | **0.962** | 0.817 |
| 120 | **−6.4** | +5.9 | **0.950** | 0.650 |
| 160 (deep rough) | **−1.6** | +12.3 | **0.940** | 0.468 |

Figure: `output/gitu_roughness/roughness_sweep.png`. **Reading:** within/near the training
span interpolation is competitive (even better — denser fit). But once the held-out band
crosses the smooth→rough transition, the field-interpolation baseline **collapses** (error
goes positive — worse than the signal itself; complex-corr 1.0→0.47), while NBS — having
recovered the physical `σ_h` — **degrades gracefully** (corr 0.94, −1.6 dB even at 160 GHz,
4π·σ_h/λ≈4.7, 5.7× the training span). A **10–14 dB** advantage exactly where the frequency
response is nonlinear.

**Net honest claim (for the paper).** The physics-vs-interpolation advantage is NOT
universal: it is concentrated where the cross-band response is *nonlinear/structured* — the
roughness regime change (shown), and by extension penetration/multi-bounce transitions.
There NBS's physical material recovery clearly beats learned field interpolation; in smooth,
densely-sampled regimes interpolation is competitive. This is a defensible, scoped E3 claim,
and it is a *secondary* differentiator behind raw-signal + downstream-sensing vs PAS.

**Still TODO:** wire `f` through the 3DPS CUDA rasterizer (kernels already take per-direction
`lambda`); run on mmIR sim scenes (Track A); real penetration/multi-bounce regime tests.
