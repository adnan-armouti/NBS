# Stage 5: cross-band inverse-rendering core on the real XRF55 room geometry

> **CORRECTION (Jun 2026, after audit — `md/method_audit.md`).** The reported **corr 1.000 is
> geometry-dominated and meaningless** (wrong materials also score 0.99+); ignore it. The
> reported **interp baseline (−5.9 dB) was degenerate** (worse than a wrong-material render,
> −20 dB), so "NBS beats interp" is an unfair comparison — retract that framing. **What
> survives:** NBS's **−37.5 dB is ~17 dB below the proper wrong-material floor (−20 dB)**, so on
> this dense, over-determined real-geometry scene the model *does* genuinely recover materials
> and synthesize the held-out band. That core claim holds; the interp comparison and the corr do not.


**What.** Scale the validated single-chip RA forward model + dispersion/roughness recovery
from point-scatterer toys to a REAL scene: the Scene1 static scaffold (the reconstructed
room, ~10^5 points). Render the room at training carriers, recover per-material-class
physics, synthesize a held-out band, and visualise it. `method/stage5_crossband.py`.

**Setup.** 2000 scaffold points; 4 material classes by height (floor=wood, low=plasterboard,
mid=concrete, high=glass) with roughness straddling the smooth→rough crossing; single-chip
(XRF55) array; RA = 256 range × 64 azimuth; 20 dB noise; train {24, 40, 60} GHz; held-out
**90 GHz** (rough regime).

**Result.**

| held-out 90 GHz | NBS (recover physics) | interp (XBandGS-style attribute) |
|---|---|---|
| |RA| correlation | **1.000** | 0.848 |
| |RA| error (dB)  | **−37.5** | −5.9 |

Figure `output/stage5_crossband/held_band_ra.png`: the NBS-synthesized held-band RA is
visually identical to ground truth; the interpolation synthesis is washed out.

**Why so clean here (vs the −10 dB point-scatterer test).** A dense scene with few material
classes (500 points each) heavily OVER-determines the per-class physics, so the recovery is
well-conditioned and the held-band synthesis is near-exact. This is the realistic regime: a
real room has many measurements but few material types — exactly where physical material
recovery shines and field interpolation (which has no shared cross-band structure) does not.

**Honest scope.** This is Track A on REAL geometry: the geometry is the real reconstructed
room, but the materials and the multi-band signal are CONTROLLED (simulated through the
forward model). XRF55 mmWave is single-band + processed magnitude, so a real multi-band
*raw* validation is impossible on it — that is blocked on the R1 single-radar multi-carrier
capture (or PanoRadar/ColoRadarPlus). What this DOES establish: the NBS core loop (render →
recover physical materials → synthesize a held-out band) works end-to-end at realistic scene
scale on real geometry, and decisively beats interpolation across the regime change.

**Next.** Thread `f` through the 3DPS CUDA rasterizer for full-resolution/full-scaffold
rendering; run on mmIR sim scenes with GT materials; and, for the headline downstream payoff,
the R1 real multi-carrier capture (the data gap that no existing dataset closes).
