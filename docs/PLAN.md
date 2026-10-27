# Plan: NBS — Novel Band Synthesis (Novel View Synthesis for the frequency axis)

> **Venue target:** MobiCom Summer 2026 (abstract ~Aug 28, paper ~Sep 4) — **aiming for a best-paper award.**
> **Thesis home:** Ch. 7 (Frequency / Multispectral Axis), §"Novel Band Synthesis".
> **Project dir:** `/home/adnan/Desktop/NBS` (copy of `template`).
> **Maps to deck project:** "★ Novel Band Synthesis (NBS) — top priority" (stays #1 for MobiCom; sharpen vs competitors).
> **STATUS: dataset strategy DECIDED — Sim (Wideband3DGS/Sionna) + real deck datasets (MM-Fi, OPERAnet, XRF55) with downstream tasks. See §Datasets.**

---

## One-line goal

Treat the **RF-frequency axis $f$** as a synthesizable dimension: train on $K$ wireless bands, **render the scene's signal at a held-out band** at the raw-signal level, via band-specific encoders → a shared 3D wireless scene representation (geometry-anchored) → a physics-informed decoder that renders at query frequency $f_q$. Primary claim = **interpolation**; extrapolation = stretch.

## ⚠️ Positioning update — the novelty bar moved (read first)

Cross-frequency wireless-field reconstruction is **no longer empty space**. Two very recent papers already do "render the field at an unseen frequency":
- **Wideband3DGS** — "Wideband RF Radiance Field Modeling Using Frequency-embedded 3D Gaussian Splatting" (arXiv 2505.20714). Sionna-simulated, 1–100 GHz, 6 indoor scenes, point-cloud-initialized Gaussians, reports only **2.8% drop** predicting at untrained frequencies. Code/data: https://github.com/sim-2-real/Wideband3DGS
- **XFreq-GS** — "Cross-Frequency Wireless Radiation Field Reconstruction with 3DGS" (arXiv 2605.11432). Simulated PAS maps, 10 carriers 1–94 GHz.

Both are **communication/propagation** papers (predict channel / power-angular-spectrum), **simulated**, with **no real geometric modality** and **no downstream sensing task**. That is exactly where NBS must differentiate to win best paper:
1. **Real measured data**, not Sionna ray-tracing.
2. **Raw-signal-level** synthesis (ADC / CIR), not just power maps.
3. **Anchored to a real geometric scene** (LiDAR / RGB-D) — a *sensing/scene-reconstruction* framing, not a comm-channel framing.
4. **A downstream-task payoff**: a model trained on band A still works after we synthesize band A from band B (the "train-once, deploy-cross-band" story), and/or scene-reconstruction quality on the synthesized band.

Related forward-model synthesis prior work to cite/contrast: **RF-Genesis** (SenSys'23), **mmGen** ("One Snapshot is All You Need", 2025), **NeRF2**, **GWRF**. Within-radar cross-band: **Rahman & Gurbuz '21**. Fusion-not-synthesis: **X-Fi (ICLR'25)**.

## Datasets — the open question

The advisor's ask: a dataset with **≥1 geometric modality (RGB-D / LiDAR)** + **≥2 wireless modalities** (mmWave / UWB / WiFi / acoustic …), oriented to **scene modeling/reconstruction**, ideally raw signal. **Honest finding: no single REAL dataset cleanly satisfies all four constraints today.** The candidates split into camps:

| Dataset | Geometric | Wireless modalities / bands | Real? | Scene-recon? | Raw? | Fit |
|---|---|---|---|---|---|---|
| **Multimodal-Wireless** (le-liang.github.io/mmw) | LiDAR + RGB + Depth | radar + mmWave CSI (~28 GHz / 4.9 GHz) | **Simulated** (CARLA+RT) | Automotive/V2X, dynamic | proc. | **Partial** — 2 RF + geometry, but sim & not static-scene |
| **ColoRadar+** (arpg.github.io/coloradarplus) | LiDAR + **RGB-D** | 2× mmWave radar (single-chip + cascade) | **Real** | **Yes** (mapping) | **Raw + heatmap** | **Partial** — geometry-rich & real, but one band *family* (two radar configs, not 2 distinct bands) |
| **PanoRadar** (penn-waves-lab) | LiDAR | 1× rotating single-chip mmWave (raw I/Q) | Real | Yes (indoor) | **Raw I/Q** | Weak for NBS (single band) — but it's the 3DPS+ dataset |
| **MAOUD** (Nature Sci Data 2026) | laser point cloud + RGB | acoustic (multibeam sonar) | Real | Yes (underwater) | — | Weak — 1 acoustic modality |
| **DICHASUS** | LiDAR + camera (antenna pose) | distributed massive-MIMO CSI | Real | localization | raw CSI | Weak — single band, comm |
| WRF datasets (Wideband3DGS, XFreq-GS, NeRF2) | none / implicit | multi-band RF | Simulated | propagation | — | These are *competitors*, not our data |

**Also worth a closer look** (surfaced late, not yet verified): "Semantically Annotated Multimodal Dataset for RF Interpretation and Prediction" (arXiv 2604.01433) — described as synchronized **RF heatmaps + LiDAR point clouds + panoramic camera**. Could be the closest real multimodal-scene fit; **TODO: verify modalities/bands.**

### ✅ DECIDED (advisor, Jun 2026): Sim (C) + real multi-band deck datasets with downstream tasks

The chosen strategy combines a controlled simulator for method development/competitor comparison with the deck's **real multi-band datasets**, using their **downstream tasks** as the evaluation payoff that the (simulated, geometry-free, task-free) competitors lack. Note these deck datasets *do* satisfy "≥1 geometric + ≥2 wireless" — they're human-centered rather than scene-recon, which we accept because the downstream-task evaluation is the differentiator.

- **(C) Simulation — method dev + fair competitor comparison.** **Wideband3DGS / Sionna** (https://github.com/sim-2-real/Wideband3DGS): controlled multi-band (1–100 GHz), point-cloud-initialized, ground-truth field. Use to develop the frequency-conditioned decoder, ablate physics priors, and beat Wideband3DGS/XFreq-GS *on their own turf* before claiming real-data results.
- **(Real) Multi-band datasets with downstream tasks** — for the headline "synthesize a held-out band, run the native-band model on it" result:
  - **MM-Fi** — WiFi CSI + mmWave + **LiDAR + RGB-D**; tasks: HAR, pose estimation. Geometric anchor + 2 wireless. (mmWave is point-cloud/processed — flag for raw-level work.)
  - **OPERAnet** — WiFi CSI (raw CFR) + UWB CIR (raw) + passive-WiFi-radar (SDR raw) + **Kinect (RGB-D)**; task: HAR. **All-raw, multi-band, with a geometric anchor** — strongest real fit for raw-signal synthesis.
  - **XRF55** — WiFi CSI + mmWave (TI IWR6843 via DCA1000) + RFID + RGB-D/IR; task: 55-action recognition. **Raw ADC pending author contact** (the deck's open risk — chase this week).
- **(Optional real sanity check)** **ColoRadar+** (LiDAR + RGB-D + two mmWave configs) for a *scene-reconstruction*, non-human flavor if we want a second real domain. De-emphasized for now.

**Downstream-task payoff (the core experiment):** train a band-specific sensing model on real band A → test it on **synthesized** band A produced from band B → compare to the train-on-real upper bound, on HAR / vital-signs / pose. This is what makes NBS a *sensing* paper, not a channel-modeling paper, and is the axis on which we beat the simulated competitors.

**Geometric anchor across all:** MM-Fi/ColoRadar+ give LiDAR; OPERAnet/XRF55 give RGB-D (Kinect) — keep the shared geometry-anchored scene representation as the differentiator vs the geometry-free WRF baselines.

## Approach (minimum decisions to start)

1. **Scene representation:** geometry-anchored (LiDAR/RGB-D → points/Gaussians) shared across bands, à la mm3DGS — this is our differentiator vs the geometry-free WRF baselines.
2. **Band-specific encoders → shared latent → physics-informed decoder at $f_q$:** EM priors (frequency-dependent permittivity/roughness, ITU/KA-SPM) condition the decoder on $f_q$.
3. **Claim scoping:** interpolation between observed bands first; extrapolation only as an explicitly-labeled stretch (matches the risk register).
4. **Raw-level target:** synthesize ADC/CIR (or beamformed RA as fallback), not just power maps — this is a key novelty vs Wideband3DGS/XFreq-GS.
5. **Downstream eval:** a band-specific model trained on real band A, tested on **synthesized** band A (from band B), vs trained-on-real upper bound.

## Files / structure to create

- `~/Documents/Data/{OPERAnet,XRF55,MM-Fi,Wideband3DGS}/` — datasets, central store outside the repo (`data/download.sh` writes here); `~/Documents/Data/NBS_own_collection/` if Track B.
- `src/encoders/` — per-band encoders.
- `src/scene_repr.py` — shared geometry-anchored representation.
- `src/decoder_physics.py` — frequency-conditioned, physics-informed decoder.
- `src/train_nbs.py` — hold-one-band-out training/eval.
- `eval/downstream/` — cross-band downstream-task harness.
- `md/` — dataset-decision log, physics-prior notes.

## Baselines / comparisons

- **Wideband3DGS** and **XFreq-GS** (cross-frequency WRF) — the direct competitors; reproduce on sim, then show our real-data / raw-level / geometry-anchored advantage.
- **Rahman & Gurbuz '21** (within-radar cross-band adaptation).
- Naïve: nearest-band copy; per-band independent model (no synthesis).

## Evaluation

1. **Reconstruction on training bands** (sanity).
2. **Hold-one-band-out synthesis** — signal-level error (ADC/CIR), and field/RA fidelity.
3. **Downstream HAR / sensing / scene-recon on synthesized vs real** band — the payoff metric.
4. **Interpolation vs extrapolation** breakdown (be explicit about the regime).

## Risks / open questions

- **Novelty erosion** from Wideband3DGS / XFreq-GS — *the* top risk now. Mitigation: real + raw + geometry-anchored + downstream task. **This must be airtight in the intro.**
- **Dataset (decided, residual risks):** (i) **XRF55 raw ADC** is still pending author contact — chase this week; OPERAnet is the all-raw fallback. (ii) MM-Fi mmWave is processed point-cloud, not raw — limits raw-level synthesis there. (iii) human-centered data means the "scene" is mostly one material (body) — keep ColoRadar+ as the multi-material/scene sanity check if reviewers push on generality.
- **EM rendering is harder than optical** (multipath, phase). Mitigation: interpolation-first, physics priors incremental.
- **Raw-signal synthesis** across genuinely different bands (WiFi CIR ↔ mmWave ADC) may be ill-posed at the sample level — fall back to a common intermediate representation if needed (and say so).
- **MobiCom timeline is tight** — if own-collection slips, fall back to Track A (ColoRadar+ two-config) + Track C (sim) for a still-publishable, if narrower, result.

## Timeline (toward Sep 4, 2026)

- **Now (Jun, week 1):** email XRF55 authors for raw ADC; pull OPERAnet + MM-Fi; stand up Wideband3DGS/Sionna sim.
- **Jun:** method on the sim set; reproduce Wideband3DGS/XFreq-GS comparisons (beat them on their turf).
- **Jul:** real-data hold-one-band-out synthesis on OPERAnet/MM-Fi/XRF55 + first downstream-task numbers.
- **Aug:** downstream-task experiments; ablations; freeze results by mid-Aug.
- **Aug 28 / Sep 4:** abstract / paper.

## Rollback

Track structure *is* the rollback: own-collection (B) failing falls back to ColoRadar+ (A) + sim (C). Method modules (encoders/decoder/scene) are independent; ablation flags disable the physics prior or geometry anchor. Per-run `config.json`; `git revert`/`git stash`.
