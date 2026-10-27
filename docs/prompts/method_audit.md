# Method-experiment audit — systematic metric & baseline failures (Jun 2026)

Triggered by PI skepticism of the band-translation result. Diagnostics:
`method/diagnose_translate.py`, `method/diagnose_stage5.py`.

## The two systematic problems

### 1. The correlation metric is geometry-dominated (worthless)
The scenes have KNOWN, shared geometry, so RA/CSI peak *locations* are fixed regardless of
materials; material identity is a small amplitude perturbation. Therefore correlation between
any two renders of the same scene is ~0.99 no matter the materials. Verified: rendering with
all-wrong materials gives corr 0.997–0.998. **Every correlation number reported for the method
experiments is invalid** (Stage 5 = 1.000; band-translate 0.95–0.99; G-ITU-RA 0.999; roughness
0.94). Do not cite RA/CSI correlation for material/cross-band fidelity.

### 2. The interpolation/copy baselines repeatedly degenerated
- band-translate "copy": recovered amplitude → exactly `[0,0,0]` → zero output (corr 0.000).
- Stage 5 "interp": −5.9 dB, **worse than painting the room concrete** (−20 dB) → broken extrapolation.
- roughness "interp": corr collapsed to 0.47 (below the ~0.99 geometry floor) → also degenerate.
None were validated to be non-degenerate, so **every "NBS beats interpolation/copy" comparison
is unreliable**; the apparent wins are largely baseline artifacts.

## What the wrong-material control (the proper floor) shows
Rendering with deliberately WRONG materials sets the floor a real method must beat on dB:

| experiment | wrong-material floor (dB) | NBS (dB) | verdict |
|---|---|---|---|
| band-translate (mmWave→WiFi) | −24.5 | −23.0 | **FAIL** — NBS below the floor; no recovery |
| Stage 5 (held 90 GHz, dense scene) | −20.0 | −37.5 | **PASS** — NBS ~17 dB below floor; genuine recovery |
| roughness / G-ITU-RA | not yet tested | — | **UNVERIFIED** — re-audit before trusting |

## Net honest status of the method experiments
- **Validated:** the differentiable forward/inverse model genuinely recovers materials and
  synthesizes a held-out band WHEN the scene over-determines them (Stage 5, real geometry,
  −37.5 dB vs a −20 dB wrong-material floor). The *mechanism* is real.
- **NOT validated / retracted:** all correlation claims (geometry-dominated); all
  "physics beats field interpolation" (E3) comparisons (degenerate baselines), INCLUDING the
  roughness regime-change headline and the band-translation result; the band-translation
  experiment entirely (NBS below the wrong-material floor).

## Required fixes before re-claiming anything
1. **Geometry-independent metric**: report dB error AND always against a wrong-material floor;
   or evaluate a quantity where materials dominate (e.g., per-surface recovered material vs GT,
   or a downstream task), not raw RA correlation.
2. **Validated baselines**: assert the interpolation/copy baseline is non-degenerate (non-zero,
   physical) before comparing; re-implement so it cannot collapse.
3. **Re-audit** roughness and G-ITU-RA against the wrong-material floor; only the claims that
   clear the floor by a meaningful margin survive.
4. Positioning: until the above pass, do NOT claim "physics beats interpolation." The defensible
   claim is narrower: a physics forward/inverse model that recovers materials and re-renders a
   held band (Stage 5), with raw-signal + downstream-sensing vs PAS as the headline.

---

## RE-RUN with the corrected harness (`method/eval_v2.py`) — claim RE-VALIDATED (scoped)
Roughness regime change (train smooth {18,28,40} GHz, held 120 GHz, `4π·σ_h/λ` up to 4.0),
dense real-geometry scene, RF-only library-assignment recovery, **two metrics + validated baseline**:

- **Material contribution / floor:** wrong permittivity (all-concrete) −21.9 dB; **wrong roughness
  (all-smooth) −5.9 dB**. So at the held band the dominant material effect is the frequency-dependent
  ROUGHNESS rolloff (not permittivity); floor to beat = −5.9 dB.
- **NBS (raw):** held −30.8 dB, **material-ID accuracy 1.00** (all 4 classes correct) → beats the
  floor by ~25 dB. **NBS (normalized):** −30.2 dB, ID 1.00 — equivalent (normalization does NOT hurt;
  relative amplitudes carry the material info; global scale is unknowable in real data anyway).
- **Interp baseline (asserted non-degenerate):** held +0.3 dB → **does NOT beat the floor.** It
  genuinely cannot extrapolate the nonlinear roughness rolloff from smooth training bands.

**Conclusion.** With a geometry-independent metric (dB vs wrong-material floor), a material-ID
metric, and a validated baseline, **"physics beats field interpolation" holds at the regime change**
(NBS 100% material-ID, 25 dB below floor; validated interp fails). It does NOT hold where materials
barely matter (band-translate, similar materials, floor −24.5 dB) — correctly scoped. The earlier
RETRACTION stands for the corr-metric/degenerate-baseline experiments; this corrected re-run is what
to cite. Still SIM (controlled materials); real-data material GT via RGB segmentation is next.
