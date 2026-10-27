# mmWave ↔ WiFi band translation — RETRACTED (invalid result)

> **RETRACTED (Jun 2026).** The result previously reported here (NBS corr 0.95–0.99, copy
> baseline corr 0.000) is **invalid**. It does not demonstrate cross-band translation.
> Diagnostics: `method/diagnose_translate.py`.

## Why it is invalid
1. **The correlation/dB metric is geometry-dominated.** The scene geometry is known and shared,
   so RA/CSI peak *locations* are fixed and material identity is a small perturbation. Rendering
   the target band with WRONG materials scores as well as or better than NBS:
   - all-concrete (every surface wrong): corr 0.998, dB −24.5
   - shuffled class→material: corr 0.997, dB −22.2
   - NBS (reported): corr 0.989, dB −23.0  ← **worse than all-concrete**
   So neither corr nor dB here measures whether translation/material recovery happened.
2. **The "copy" baseline was degenerate, not beaten.** Its recovered per-class amplitude
   converged to `[0,0,0]` (zero-initialized, never moved) → all-zero output → corr 0.000.
   NBS was compared against a baseline that produced nothing.

## Status
No cross-band translation is demonstrated. NBS does not beat a wrong-material control here.
The premise (render complex CIR/CSI, supervise |CSI|; identify materials to bridge bands)
may still be sound, but proving it requires (a) a geometry-independent metric or a scene where
materials actually dominate the signal, (b) a validated non-degenerate baseline, and (c)
showing NBS beats a wrong-material floor. None of those held here.

See the systematic write-up in `md/method_audit.md`.
