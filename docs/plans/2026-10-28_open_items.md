# Open items after Task B: findings and plans (2026-10-28)

Sources read for this: Hofmann et al. code `nihofm/inverse-radar-rendering` (slang SFCW kernels, MAROON data layout),
Vayyar IMAGEVK-74 datasheet + API guide (`pdfs` of the old NBS checkout; the Mini-Circuits VTRIG-74 download is bot-gated
and could not be fetched from this machine), the OctoNet paper tables and the `aiot-lab/OctoNet` loader, and probes run
on the node today (UWB pickle stream, Vayyar coupling / phase coherence).

Two corrections to yesterday's statements come first, because they change the plans:

* **UWB is a 17 Hz stream, not one frame.** The UWB pickle is a *stream* of appended per-frame dicts; `pickle.load`
  returns only the first. `octonet_io.load_uwb` now reads all of them: 1576 frames × 1535 bins, 17.06 Hz, 92 s for
  the sit recording. The frame is RF-sampled (spectral peak at 7.2 GHz for a 23.328 GS/s sampler; 98.7 % of the energy
  above 0.1 cycles/bin), so the head's RF output is the comparable one, not the envelope. Residual/static energy
  −31 dB; frame-to-frame correlation 0.9985: UWB is as static-dominated as SFCW. (WiFi and Vayyar files are single objects.)
* **The Vayyar array is the datasheet's L-array, but the raw phasors are not beamformable as-is.** Datasheet Fig. 4:
  Tx1..Tx20 along one edge, Rx21..Rx40 along the adjacent edge, corner at Tx20/Rx21, array length 0.04 m, PCB 80 mm.
  The strongest direct-coupling pairs in the data are (tx 18–19, rx 0–7) → the corner is at tx index 19 / rx index 0
  under the tx-major reshape, consistent with the figure. But a phase-coherence probe on the static return shows a
  clean, range-independent +90..105°/element phase ramp along the first pair axis and random phases along the second,
  at every range bin including the coupling. Range-independent means it is not geometry: the channels carry
  per-element phase terms (and possibly sweep-to-sweep phase between Tx) that must be calibrated before any
  azimuth/elevation processing. The per-pair phase pattern is not a fixed channel constant either (bin-to-bin
  similarity 0.1, frame-to-frame at the coupling 0.28), so a one-shot coupling-based calibration does not work; see §4.

---

## 1. Material model: what is identifiable, and what to take from Hofmann

### Why "one scalar per point" here when 3DPS/4DPS fit six parameters
Our 3DPS/4DPS material is six raw parameters per point (`mm4DPS/rasterizer.py::reparameterize_torch`): ε′, ε″,
roughness σ_h, correlation length l_c, the specular/diffuse mixing τ, and a layer thickness. They became identifiable
because the measurements vary the **viewing geometry**: ColoRadar's radar moves through the scene (hundreds of poses),
and HuPR/RT-Pose see a moving person, so every point is observed over a range of incidence angles and the
angle-dependence of Fresnel (ε′, ε″), of the rough-surface lobe (σ_h, l_c, τ) and of the layer interference
(thickness) separates them. The NBS Vayyar observation of Scene 1 is **one static view**: the 400 pairs span a 4 cm
aperture (the bistatic angle to a point at 3 m changes by < 1°), there is no motion, and the 63–67 GHz sweep is a 6 %
fractional bandwidth. Per point we therefore observe one complex number per pair, and the pairs are nearly redundant
in angle. That leaves one effective degree of freedom per point for reflectivity (plus the phase, which is geometry).
It is not that the six-parameter model is wrong; it is that this measurement does not excite it.

### Frequency dependence is the point, and the way to get it is across bands, not within one
The ITU-R P.2040 model (`nbs/method/freq_material.py`) is dispersive by construction: ε′ = a·f^b, σ = c·f^d. Within
63–67 GHz the induced change in Fresnel reflectivity is below 0.1 dB for every ITU class, hence "not identifiable
within the band". Across the bands OctoNet records at the same scene it is large: WiFi 5.18 GHz, UWB ~7.2 GHz,
Vayyar 65 GHz. A shared (a, b, c, d) field fitted jointly to all three is identifiable in exactly the directions that
matter for synthesis at an unseen band (the exponents b, d), and it is the NBS thesis. The static Scene-1 fit should
therefore become a **joint multi-band fit of one material field**, with each band's head consuming the same
`abcd` (the heads already accept it) and per-band gains.

### What Hofmann's code does, and what to adopt
`sfcw_signal_material.slang` evaluates per hit: five material models selectable at compile time — baseline albedo,
Blinn-Phong mix, layered GGX, **Fresnel smooth** (complex IOR n = 1 + 99·p, k = 100·p, s/p polarisation mix) and
**Fresnel rough** (the same with GGX D·G). Parameters are sigmoids of free values clamped to ±10. The material is
**not dispersive** (n, k constant over 72–82 GHz), confirming that within one band dispersion is not the lever.
Storage (`diffradar.py`): global, voxel grid, hash grid (InstantNGP MLP), or per vertex. Loss (`main.py`): **L1 on the
complex raw signal** per pair and frequency (`l1_complex`, the default), magnitude L1/L2, or L2 on a holographic
reconstruction. Also: a learned **global sub-wavelength registration offset**; **empty-room filtering** (a capture of
the room without the object, subtracted: MAROON ships these); an optional measured **antenna pattern** (APC); zeroing
simulated pairs where the real ones are zeroed. The forward model is a near-field path sum (1/d_tx²·d_rx², cos·cos)
over 65k ray hits per step with OptiX, i.e. the same structure as our monostatic/bistatic `render_sfcw_pairs`.

Adopt, in this order:
1. **Spatially regularised material storage** instead of free per-point values: a voxel or hash grid queried at the
   scaffold points (Hofmann's `VoxelGridMaterial` / `HashGridMaterial` are ~20 lines each). This is the structural
   answer to the runaway points; the [1/20, 20] bound becomes unnecessary.
2. **Complex (phase-coherent) L1 loss** as a second term once §4 gives trustworthy inter-channel phases, with a learned
   sub-wavelength registration offset (λ = 4.6 mm, so a 1 mm error is 80° of phase). Until then: log-magnitude only.
3. **Fresnel + GGX roughness as the per-frequency reflectivity**, driven by our ITU ε(f) rather than their constant
   (n, k): i.e. keep `abcd` as the state and add σ_h (GGX α) as a fifth field, which the UWB/WiFi incidence
   diversity (receivers around the room) can excite even though the Vayyar cannot.
4. Their **s/p polarisation mix** parameter: the Vayyar antennas are linearly polarised (datasheet), so the mix is
   known per pair geometry, not a free parameter; implement as a fixed term.
5. Empty-room filtering: not available (no empty capture); the coupling plan in §2 is our substitute.

Deliverables: `octonet_fit` v2 = grid-stored `abcd` (+ σ_h) shared across bands, heads for SFCW/UWB/WiFi in one loss,
log-magnitude first; a held-out band test (fit on two bands, predict the third) as the synthesis gate.

## 2. Coupling leakage into the first guarded bins

Facts: the direct Tx→Rx coupling is the strongest term in every pair, at bin 5–6 (0.2 m, the on-board path), and its
Hann sidelobes fall to ~−15 dB only by 1 m (bins 6–25). The current guard masks bins 0–5. The coupling's complex
pattern is **not** stable frame to frame (similarity 0.28 across frames), so it is a drifting term, not a constant.

Plan (ordered, each step measurable by the fitted ρ near the radar returning toward ~1 and the loss not degrading):
1. **Per-frame coupling fit and removal in the frequency domain.** For each frame and pair, fit a complex tone
   A·exp(−j2πfτ) with τ constrained near the on-board delay (bins 4–7), plus its first sidelobes implicitly, by least
   squares on the un-windowed sweep; subtract; then window and range-compress. This removes the leakage with its
   sidelobes rather than masking them. Validate on the pooled target: energy in bins 6–25 should drop and the
   cross-recording correlation there should rise.
2. **Window choice.** Range-compress with a lower-sidelobe window (Blackman-Harris, −92 dB) for the target and the same
   window in the head (the PSF is the window's transform); costs 1.7× in resolution (6.4 cm) but kills the leakage
   beyond ~0.4 m. Report both.
3. **Coupling term in the forward model.** Add the fitted coupling (per pair, from step 1) back into the render so the
   loss compares like with like in the bins where removal is imperfect; the guard then shrinks to bins 0–3.
4. Keep the guard at 0–5 plus a down-weight of bins 6–12 as the fallback if 1–3 do not help.

## 3. Person term for UWB / WiFi, and the SFCW question

### What the OctoNet numbers actually say
SFCW HAR: **in-domain 98.3 % / 93.8 %** (10 / 62 classes, ResNet), **cross-user 12.3 % / 1.6 %**, cross-scene
11.3 % (10-class chance = 10 %); SFCW HPE 142 mm in-domain → 315 mm cross-user. UWB: 98.3 % in-domain, **62.6 %
cross-user**, 27 % cross-scene. The in-domain SFCW number is not evidence that the person's motion is in the signal:
train and test share subjects and sessions, the frames within a recording correlate at 0.999, and a network can
classify from the session's static fingerprint (where the chair stands, where the subject stands). The collapse to
chance across users says that signature does not transfer, which is what a static fingerprint does. UWB holding
62 % across users says UWB does carry person information that generalises. So the answer to "is the person in the
SFCW data?" is: at the −24 dB, phase-decorrelated level we measured; not identifiable per frame by physics, and not
shown to generalise by the authors either. We should not model a person return in SFCW. We should keep the person
**as an occluder** of the static scene (shadowing of the floor/wall behind them), which is first-order physics and
needs no Doppler; whether that occlusion is measurable is part of the test below.

### Test before building (decides the modelling, one day on the node)
For each of SFCW, UWB (now 17 Hz), WiFi (75 Hz), per recording:
1. Static removal: subtract the per-recording temporal-mean channel.
2. **Physics-predicted residual**: render the person (SMPL from `person_sequence.npz`, now available per frame) as
   (a) a direct return and (b) an occluder of the scaffold, through the band's head, at the measured frame times;
   subtract the rendered static scene to get the predicted residual.
3. Score: correlation over time between the measured residual energy per range bin and the predicted one, against a
   permutation null (shuffle frames within the recording, 1000 draws), across ≥ 10 recordings and activities. Also
   the simpler model-free check used in July: the repetition rate from the MoCap must appear in the residual's
   spectrum above the null.
4. Decision: a band passes if the median z-score across recordings > 3 and the effect survives cross-recording.
   Passing bands get a person term (direct + occlusion); failing bands keep the person as an occluder only, or not
   at all if even occlusion fails.
Expectation from today's numbers: UWB (−31 dB residual, generalising HAR) is the likeliest to pass; WiFi plausible
via LOS blockage; SFCW likely occlusion-only or fail.

## 4. Vayyar geometry and calibration (line items for approval)

Assumptions, each with its source and how reasonable it is. Items marked **(verify)** have a test in the plan below.

| # | Assumption | Source | Assessment |
|---|---|---|---|
| V1 | 20 Tx on one straight edge, 20 Rx on the adjacent straight edge (L-array) | datasheet Fig. 4, "40 embedded Tx/Rx antennas", API: 20 Tx × 20 Rx pairs | solid |
| V2 | Array length D = 0.04 m per edge, hence uniform pitch 2.1 mm (0.455 λ at 65 GHz) | datasheet angular-resolution formula uses D = 0.04 m; uniformity is NOT stated | plausible; the pitch value is a guess within ±20 %. **(verify)** |
| V3 | Corner (closest Tx–Rx pair) = Tx20 / Rx21, i.e. tx index 19, rx index 0 (0-based) | datasheet labels; strongest coupling pairs (tx 18–19, rx 0–7) | supported by data |
| V4 | Pair order in the 400-vector is tx-major (`p = tx·20 + rx`), matching OctoNet's `(400,100)` layout | OctoNet loader; API `GetPairId(i,j)` (order not documented) | unverified; coupling structure is smooth under this reshape (0.97), which rules out interleaving but not the transpose. **(verify)** |
| V5 | All 20 Tx were used (LOW_RATE profile), 100 frequency points 63–67 GHz, RBW default | 400 pairs in the data; OctoNet text (4 GHz BW); API `ApplySettings` | solid for 20 Tx and 100 points; the exact start/stop (63.0/67.0) is OctoNet's statement, not read from the data |
| V6 | Per-element phase: raw phasors are **uncalibrated**; a range-independent +90..105°/element ramp exists along the first pair axis; the second axis carries range-varying phases | today's probe | observed; mechanism unknown (LO phase step per Tx switch, or per-channel line lengths) |
| V7 | Element pattern: 3 dB beamwidth 180° (E-plane) × 82° (H-plane), gain 5.8 dBi, linear polarisation | datasheet Table 4 | solid; the plane orientation relative to the L edges must be inferred (figure 5 axes) |
| V8 | Tx power −10 dBm, Rx noise figure unknown | datasheet | only matters for absolute gain, which we fit |
| V9 | The board's mount on node_1: at the marker, yaw = camera yaw, level | our assumption (sweep flat) | weak; §4b |
| V10 | Processing: range = IFFT over the 100 tones with a Hann window; no range calibration offset | matches OctoNet's own `vayyar_to_ra`-style use and Hofmann | the on-board delay (coupling at bin 5–6 = 0.2 m) shows a ~0.2 m path offset that should be subtracted from all ranges **(verify)** |

Plan to retire the "(verify)" items (two to three days, node):
1. **Per-channel phase calibration from a known point.** We lack a corner reflector, but we have the subject's
   MoCap-known standing position and, better, a controlled frame: pick the recording and frame where the person is
   closest to the radar and most compact (seated 'sit' recording, 2.9 m); the person is a weak return, so instead use
   the **floor/ceiling specular points** predicted by the calibrated scaffold: for each pair the strongest scaffold
   return at a chosen range bin has a known geometry phase. Solve per-Tx and per-Rx phase offsets (38 unknowns) by
   least squares over many bins and pairs under V1–V4; the fit residual tells whether a rank-1 (Tx + Rx) calibration
   explains the data. If it does not, test the hypothesis that Tx sweeps are mutually incoherent (per-Tx, per-frame
   random phase): then only the Rx axis is usable for beamforming and the azimuth/elevation assignment of V1 decides
   which angle we get.
2. **Resolve V4 and the pitch (V2).** With the calibration applied, form the static RA image under (tx-major, pitch p)
   and (rx-major, pitch p) for p in 1.6–2.6 mm and score each against the rendered occupancy of the calibrated scaffold
   (the same normalised correlation `fit_lidar_alignment` uses). The correct combination shows the −X wall and the
   floor where the scaffold has them; the wrong transpose puts walls in elevation.
3. **Range offset (V10):** fit the one-way on-board delay from the coupling bin and subtract it; check the floor's
   range in the calibrated RA against the scaffold.
4. Record the result as `vayyar_radar_config` v2 (positions, order, phase table, range offset) and re-enable the RA
   term and the complex loss (§1.2).

### 4b. Radar mount yaw, as a coarse-to-fine alignment
Same recipe as 3DPS/4DPS once the array is trustworthy: (i) coarse: yaw ∈ [−60°, 60°] step 5°, pitch ∈ {−10, 0, 10},
score = normalised correlation between the measured static |RA| (coupling removed) and the rendered occupancy |RA|
of the calibrated scaffold at that pose; (ii) fine: ±5° step 0.5° around the best, plus a lateral offset ±10 cm;
(iii) refine jointly with the material fit using the complex loss and Hofmann's sub-wavelength registration offset.
Acceptance: the coarse curve must have a single peak > 3σ above its flat level (the range-only sweep yesterday was
flat because range profiles carry no angle; RA does). Camera yaw stays the prior and the fallback.

### 4c. UWB band constants, data-driven
The X4M200 pulse is in the data: the direct coupling at bins 3–8 is an isolated copy of the transmitted pulse. Plan:
(i) estimate the pulse template per recording as the temporal-mean frame over bins 0–40 (RF, carrier included);
confirm the carrier 7.2 GHz and the sampler 23.328 GS/s from its spectrum (done today: 7.07–7.23 GHz); (ii) use this
template as the UWB head's range kernel in place of the Gaussian-at-datasheet-band; (iii) the range axis is
c/(2·fs) = 6.43 mm per bin, which matches 9.9 m / 1535; (iv) the UWB's own mount is treated like the radar's in 4b
with the 17 Hz stream now available. No datasheet dependence remains except the sampler rate, which the bin spacing
confirms.

## Order of work proposed
1. §4.1–4.3 (array calibration and order) — unblocks the RA term, the complex loss and the yaw alignment.
2. §2 coupling removal — improves every SFCW fit immediately, independent of §4.
3. §3 detectability test — one day, decides whether a person term is built at all.
4. §1 material v2 (grid storage, multi-band joint fit, held-out band gate) — the paper's result.
5. §4b yaw alignment and §4c UWB template as they become unblocked by 1.
