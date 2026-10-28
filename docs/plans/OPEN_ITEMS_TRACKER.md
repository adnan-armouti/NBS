# Open items tracker (NBS / OctoNet)

One line per item; status ∈ {todo, running, blocked, done, rejected}. Evidence = the file to look at. Details and
rationale: `docs/plans/2026-10-28_open_items.md` (§ numbers below refer to it). Update the status line, not the history.

| id | item | status | acceptance | evidence | next action | owner |
|---|---|---|---|---|---|---|
| T1 | Node-5 depth scale / camera offsets | done 10-27 | shared walls agree < 0.1 m | `viz/geometry_init/calibration.png`, `nbs/assets/calib_S1_extrinsics.json` | — | — |
| T2 | SFCW static target with reliability | done 10-27 | pooled target, cross-rec corr 0.87 | `nbs/method/sfcw_target.py`, `viz/render_vs_gt/sfcw_static_target.png` | revisit weights after T4 | — |
| T3 | UWB stream loader (pickle = per-frame dicts) | done 10-28 | 1576 frames @ 17 Hz read | `nbs/preprocessing/octonet_io.py::load_uwb` | — | — |
| T4 | Coupling leakage into bins 6–25 (§2) | running 10-28 | fitted ρ < 1 m returns toward 1; bins 6–25 cross-rec corr up; loss not worse | `nbs/method/sfcw_coupling.py`, `viz/render_vs_gt/sfcw_coupling.png` | per-frame tone fit → window test → coupling term in render | agent B |
| T5 | Vayyar per-channel phase calibration (§4.1, V6) | running 10-28 | rank-1 (Tx+Rx) phases explain the static return (coherence along both axes > 0.5) or the Tx-incoherence hypothesis is established | `nbs/preprocessing/vayyar_calibrate.py`, `calib_S1/vayyar_channel_calibration.json`, `viz/render_vs_gt/vayyar_calibration.png` | — | agent A |
| T6 | Vayyar pair order + pitch (§4.2, V2/V4) | running 10-28 | RA of the calibrated static scene puts the −X wall / floor where the scaffold has them; tx-major vs rx-major decided by RA correlation margin > 3σ | same as T5 | needs T5 | agent A |
| T7 | Vayyar range offset (§4.3, V10) | running 10-28 | floor range in RA matches scaffold within 1 bin | same as T5 | needs T5 | agent A |
| T8 | Vayyar assumptions V1–V10 approval | blocked | user approves each line | `docs/plans/2026-10-28_open_items.md` §4 table | user; VTRIG-74 PDF → `pdf/` | user |
| T9 | Person detectability test, SFCW / UWB / WiFi (§3) | running 10-28 | per band: median z > 3 over ≥ 10 recordings, cross-recording | `nbs/method/person_detectability.py`, `viz/render_vs_gt/person_detectability.png` | person stage on 10 recordings → residual-correlation + permutation null | agent C |
| T10 | Person term (direct + occlusion) for passing bands | todo | loss on held frames improves vs static-only | — | after T9 | — |
| T11 | Material v2: grid/hash storage (Hofmann) (§1 item 1) | todo | no runaway points without the bound; loss ≤ bounded per-point | `4DPS/mm4DPS/octonet_fit.py` | after T4 | — |
| T12 | Multi-band joint fit of one ITU `abcd` field (§1) | todo | held-out band predicted better than per-band fit; synthesis gate | — | after T11, T4; UWB/WiFi targets defined like T2 | — |
| T13 | Complex (phase-coherent) L1 loss + sub-wavelength registration (§1 item 2) | blocked | needs T5–T7 | — | after T5 | — |
| T14 | GGX roughness field on top of ITU Fresnel (§1 item 3) | todo | identifiable from UWB/WiFi incidence diversity | — | after T12 | — |
| T15 | Radar mount yaw, coarse-to-fine on RA (§4b) | blocked | single coarse peak > 3σ; fine ±0.5° | — | needs T5–T7 | — |
| T16 | UWB pulse template from the coupling peak (§4c) | todo | head's CIR matches the measured coupling shape | `4DPS/mm4DPS/octonet_heads.py` | one day; independent | — |
| T17 | UWB / WiFi static targets (reliability-weighted, like T2) | todo | cross-rec corr per bin/subcarrier | — | prerequisite of T12 | — |
| T18 | OctoNet in the 4DPS chain / Workspace layout | deferred | — | `4DPS/docs/OCTONET.md` item 6 | only if the chain is needed | — |

## History
- 10-27: T1, T2 done; Task B done (`4DPS/docs/OCTONET.md`).
- 10-28: T3 fixed (UWB had been read as one frame); plan written; T4, T5–T7, T9 launched in parallel on the node.
