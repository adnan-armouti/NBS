# PLAN v3 — Cross-band synthesis of the DYNAMIC (human) signal over sequences

Date: 2026-06-12. Supersedes the single-snapshot, static-scene framing of the real-data
mmWave↔WiFi experiments (PLAN_v2 / WRITE_UP §7a-4).

## 0. Why v3 exists (the correction)

The real mmWave↔WiFi transfer looked weak (per-link 0.026; dB-corr 0.08–0.19, WRITE_UP §7a-4),
and I attributed it to "disjoint scene content." **That diagnosis was wrong in a specific, important
way:** it was measured on the **static** scene (human masked out to build the scaffold; WiFi reduced
by `np.median` over time → the static component). The static multipath *is* environment-specific and
modality-specific — but it is precisely the part **both** modalities discard for sensing.

The **dynamic** signal — the moving human — is **shared** by both bands and is the informative one:
- WiFi sensing removes the static (LoS + environment) and keeps the human reflection via CSI-ratio /
  conjugate multiplication (FarSense; "Boosting WiFi Sensing via CSI Ratio"). Widar3.0's BVP is a
  *domain-independent* human-velocity representation extracted from WiFi — environment stripped by
  construction.
- mmWave radar removes static clutter via MTI / zero-Doppler suppression and reads the human from
  micro-Doppler; static humans are hard *because* they lack Doppler.
- Cross-modal WiFi↔radar via a shared human representation is established: RF-CM (HAR transfer across
  WiFi/mmWave regardless of setup); RF-Genesis (human **mesh** → mmWave via physical-optics, then a
  diffusion refiner); latent-diffusion generators emitting WiFi/FMCW/RFID from a shared body+activity
  latent.

**Therefore:** novel band synthesis between mmWave and WiFi *is* valid — provided we target the
**dynamic, human-induced** signal over **temporal sequences**, with the human's motion as the shared,
band-independent latent and each band's scattering as the band-specific part. This is the NBS thesis
(shared geometry + band-conditioned materials) re-instantiated with the **moving human** as the scene
content that both bands actually observe.

## 0.5 Premise check — DONE (real XRF55, 2026-06-12)

Before committing to the plan, verified that the two bands' **dynamic** signals track the same human:
across 80 matched Scene1 clips, **WiFi dynamic energy (temporal std of CSI) vs mmWave Range-Doppler
energy correlate positively: Spearman +0.68, Pearson +0.58** (off-DC absolute Doppler energy +0.39;
Doppler *spread* anti-correlates −0.67 but that measures velocity *dispersion*, not motion magnitude —
confounded). So both bands' dynamic content rises and falls together with the human's motion intensity,
exactly as the shared-human hypothesis predicts. (Earlier single-snapshot transfer used the STATIC
component and saw ~0 — consistent: the signal lives in the dynamics, not the static map.)

## 1. Hypothesis

Let `H_t` be the human's 3D state (pose / mesh / position) at time t — band-independent, the only
moving thing in an XRF55 clip. Then:
- mmWave Range-Doppler sequence `RD_t = F_mmw(H_t, H_{t-1}; scene)` (Doppler from inter-frame motion).
- WiFi dynamic CSI sequence `dCSI_t = F_wifi(H_t; scene)` (human-perturbed multipath).
Both are driven by the **same** `H_t`. So we can recover `H_t` (a motion trajectory) from band A's
sequence and render band B's sequence — supervising on the **dynamic** signal, not the static map.

XRF55 fit: only-the-human-moves; synchronized mmWave (RD + RA), WiFi (1000-sample CSI), RFID, and
Azure Kinect RGB(+D, captured) — a clean shared-human testbed.

## 2. What changes vs v1/v2 — target the DYNAMIC signal

| | v1/v2 (what we did) | v3 (what to do) |
|---|---|---|
| WiFi reduction | `np.median` over 1000 samples → **static** | dynamic component: CSI-ratio across antennas / conjugate / temporal variance |
| mmWave map | Range-Azimuth only (`[...,64:128]`) | **Range-Doppler** (`[...,0:64]`) for motion + RA for geometry |
| Scene model | static scaffold (human masked OUT) | static scaffold **+ a moving human** scatterer per frame (human is IN) |
| Unit of work | single snapshot | **temporal sequence** (17 mmWave frames / WiFi sub-windows) |
| Shared latent | per-plane materials | **human motion trajectory `H_t`** (+ band-specific scattering) |
| Supervision | absolute |RA|/|CSI| corr (geometry-dominated) | dynamic-signal corr (human-dominated) + a downstream HAR check |

## 3. Shared representation — the human `H_t`

Options, cheapest→best:
1. **Monocular human mesh from the released RGB** (HMR/SMPL per frame; e.g. a 4D-Humans-style model).
   Available now (videos are RGB). Noisy depth/scale → anchor scale to the mmWave RA range (we already
   calibrate range, §RANGE_RES) and to the floor (leveling, `level_R`).
2. **Kinect depth** (captured, not released) → metric mesh. Author ask (already on the list).
3. **Recover `H_t` directly from band A's dynamic signal** (RD → radial-velocity/position track),
   the pure RF-only setting — the real NBS claim (no optical at inference).

Start with (1) for a forward-model that renders both bands from the same mesh; build toward (3).

## 4. Forward models — human as a dynamic scatterer

- **mmWave RD**: extend the RA forward model to Range-Doppler. Doppler bin ∝ radial velocity =
  range-rate of the human mesh vertices between frames `H_t→H_{t+1}`. Render `RD_t` from the moving
  mesh (RF-Genesis-style physical-optics on the body surface is the reference; our lightweight
  splat+multibounce is the fast version). XRF55 ships real RD → direct supervision.
- **WiFi dynamic CSI**: the multibounce renderer (already built, CUDA, verified) with the **human mesh
  added as a set of moving reflectors** on top of the static room box. The dynamic component =
  `CSI_t − CSI_static` (or the antenna CSI-ratio), which cancels the static room and exposes the
  human-path contribution — exactly what the renderer can predict and what sensing uses.
- **Band-specific scattering**: the human's reflectivity at 5.64 GHz vs 60 GHz (skin/clothing
  permittivity + roughness) is the *band-conditioned material* term — the NBS material-recovery hook,
  now on the human rather than the walls (and, unlike the walls, the human signal is NOT static-dominated,
  so materials should actually move the metric — re-test the material-sensitivity finding here).

## 5. Concrete code changes (this repo)

1. **Data loaders** (`preprocessing/xrf55_io.py`): add `wifi_csi_sequence(clip)` → (9, 30, T) complex/
   amplitude over time (stop collapsing to median); add `mmwave_rd(clip)` → (17, 256, 64) Range-Doppler
   from `[...,0:64]`.
2. **Dynamic signal ops** (new `method/dynamic.py`): CSI-ratio / conjugate / temporal-variance for WiFi;
   MTI (high-pass over frames) for mmWave RA; RD extraction. These define the supervision targets.
3. **Human model** (new `preprocessing/human_mesh.py`): per-frame SMPL/mesh from RGB (mmir env), placed
   in the radar frame via the existing rig (`level_R`, range scale). Cache per clip like depth.
4. **Dynamic forward models**:
   - `method/ra_forward.py` → add `render_rd(mesh_t, mesh_tp1, ...)` (Doppler).
   - `method/multibounce_torch.py` / `cuda_multibounce.py` → accept extra moving point-reflectors
     (the human vertices) and return per-frame CSI; add a `dynamic_csi()` wrapper (CSI − static).
5. **Sequence eval** (`method/eval_v2.py` → add `eval_dynamic`): correlate predicted vs real dynamic
   signals over the sequence; report against a static-only baseline (must beat it) and a wrong-motion
   floor; plus a downstream HAR sanity check (does the synthesized band support the action label?).
6. **Material re-test**: rerun the material-sensitivity ablation (WRITE_UP §7a) on the **dynamic** WiFi
   signal — the static-dominance that made materials invisible should be gone.

## 6. Experiments

- **E1 — Shared-human consistency.** Recover `H_t` from mmWave RD; render WiFi dynamic CSI; correlate
  with real over the sequence. Reverse direction (WiFi dynamics → human track → mmWave RD). Beat the
  static baseline and a wrong-/shuffled-motion floor.
- **E2 — Band synthesis with band-conditioned human scattering.** Hold out one band; render its dynamic
  signal from the shared `H_t` + band-specific human reflectivity. Report dynamic-corr + dB-vs-floor.
- **E3 — Downstream sensing (the load-bearing test).** Train the XRF55 HAR model on the *synthesized*
  held-out band; report accuracy vs training on the real band (and vs PAS-magnitude, the E5 from
  positioning.md). This is what a top venue wants: the synthesized band is *useful*.
- **E4 — RF-only inference.** Drop the optical mesh at test time; recover `H_t` from band A alone.

## 7. Risks & honest caveats

- **Human geometry source.** Without Kinect depth, the mesh is monocular (noisy scale/pose). Mitigate
  with the existing range + floor anchors; quantify sensitivity. (Author ask for depth still helps.)
- **WiFi dynamic SNR.** The human-path CSI is small vs the static; the CSI-ratio/conjugate ops are the
  literature-standard fix, but commodity-CSI noise is real — expect modest absolute corr.
- **Physics vs learned drift.** If the human forward model is too weak, the pipeline could collapse into
  a learned cross-modal translator (the competitors' regime). Keep the physics forward model load-bearing
  (render, don't regress) and use learning only for refinement — preserves the NBS differentiator.
- **Still cross-MODALITY, not multi-carrier-same-modality.** This is the honest framing: mmWave and WiFi
  differ in hardware, not just carrier. The shared-human-dynamics result is a legitimate cross-band
  synthesis instance, but the cleanest NBS claim (K carriers of one modality → held-out carrier) still
  wants a multi-carrier dataset (sim already validates it).

## 8. Milestones

1. Loaders + dynamic-signal ops + verify the real dynamic WiFi/mmWave signals are human-correlated
   (sanity: dynamic energy tracks the action) — **1 step, no new physics**.
2. Monocular human mesh per frame, cached, placed in the radar frame.
3. Dynamic forward models (RD + human-augmented multibounce); E1 consistency.
4. E2 band synthesis + material re-test on dynamic signal.
5. E3 downstream HAR on synthesized band (the headline) + E4 RF-only.

## 9. One-paragraph pitch (if it works)

> On XRF55, the only moving scatterer is the human, and both the 60 GHz radar and the 5.64 GHz WiFi
> link observe its motion — radar as Range-Doppler, WiFi as a perturbation of the dynamic multipath.
> NBS recovers a single band-independent human-motion trajectory from one band and renders the held-out
> band's *dynamic* signal through a differentiable physics forward model with band-conditioned human
> scattering, then shows the synthesized band supports downstream action recognition. Unlike PAS-field
> interpolators, NBS synthesizes the raw, phase-coherent dynamic signal a sensing task actually needs —
> and unlike static-scene rendering, it targets the human content both bands share.
