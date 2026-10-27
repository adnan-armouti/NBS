# Objective-term ablation: Step 0, the gradient checks

Run 2026-09-22 on the node, before any fit. Plan: `misc/LOSS_ABLATION_PROMPT.md`.

`MM4DPS_GRAD_PROBE` at the **converged** baseline state (`MM4DPS_GRAD_PROBE_VF` =
`base2`'s trained `velocity_field.pt`), 12 evenly spaced frames, each term evaluated
and backpropagated alone. Sequences: RT-Pose 176 and HuPR 38, both in their dataset's
test split. Each was run twice, at `--elevation-planes motion` (shipped) and `all`.
Raw output: `probe/{rtpose_seq176,hupr_seq38}_{motion,all}/probe.json`.

## Q1 — does loop-0 RAE reach the scene the way loop-0 RA does?

Gradient norm, term evaluated alone:

| | scene materials | scene normals | absolute gain |
|---|---|---|---|
| **RT-Pose 176** | | | |
| `loop0_ra` | 45.70 | 95.12 | 4.188 |
| `elevation_loop0_rae` | 19.82 | 46.16 | 2.106 |
| ratio | 0.43 | 0.49 | 0.50 |
| **HuPR 38** | | | |
| `loop0_ra` | 6.734 | 5.767 | 2.235 |
| `elevation_loop0_rae` | 5.850 | 4.224 | 1.681 |
| ratio | 0.87 | 0.73 | 0.75 |

**PASS on both datasets.** Loop-0 RAE delivers gradient of the same order to the
scene's materials, its normals and the absolute receiver level. The factor of about
two on the cascade and about 0.8 on HuPR tracks the elevation bin count (8 against
2): the same constraint spread over more bins. `cons_rae0` does not orphan the scene.

## Q2 — does the plane RAE term reach velocity the way the plane RA term does?

Gradient norm on `velocity`:

| | `plane_ra` | `elevation_plane_rae` | ratio |
|---|---|---|---|
| RT-Pose 176, `motion` (53/63) | 0.5277 | 0.5592 | 1.06 |
| RT-Pose 176, `all` (63/63) | 0.5277 | 0.3743 | 0.71 |
| HuPR 38, `motion` (13/63) | 1.7130 | 1.6130 | 0.94 |
| HuPR 38, `all` (63/63) | 1.7130 | 1.8240 | 1.06 |

**PASS at either band setting.** Opening the band does not help and on the cascade
slightly lowers the norm, because the term is a mean over cells and the added
out-of-band planes are mostly empty: more cells, same signal, smaller mean gradient.

**But the probe cannot test what the out-of-band planes are for.** Their zero targets
penalise energy rendered where the recording has none, and the probe runs at a state
the RA plane term already made compact. At that state the constraint is satisfied and
contributes little gradient — which is exactly what a satisfied constraint looks like,
not evidence that it is unnecessary. `cons_raepl` still has to be judged by the fit.

## What the probe found that was not being looked for

1. **Only `plane_ra` carries real gradient to the noise model** — 0.168 on RT-Pose
   and 0.248 on HuPR, against 0 for `elevation_plane_rae` (detached there by design)
   and 0.060 / 0.321 for `md`. The learned floor is fitted mainly by the RA plane
   term. `cons_raepl` therefore removes the dominant fitting path for the nine
   noise-floor parameters and leaves them to micro-Doppler alone. This was not in the
   design and is the strongest argument yet for expecting `cons_raepl` to cost
   something. It also means `cons_raepl` and `abl_md` interact: together they would
   leave the floor almost unsupervised.
2. **Neither plane term nor micro-Doppler touches the scene at all** (scene materials
   and normals exactly 0). The two loop-0 terms are the only scene path, as
   `docs/STATIC_SEQS_PLAN.md` §2 states. This is what makes Q1 the load-bearing check.
3. **Micro-Doppler dominates the velocity path** — 2.10 on RT-Pose and 7.92 on HuPR,
   three to five times either plane term. Expect `abl_md` to be load-bearing.
4. **The person-window term puts more gradient on scene materials than loop-0 RA
   does on RT-Pose** (110.3 against 45.70; on HuPR 19.59 against 6.734), while its
   gradient to the absolute gain is ~1e-7, by construction. A term named for the
   person is the largest single source of scene-material gradient on the cascade.
   Worth understanding before the paper describes it as a person term.

## Verdict

Both consolidations clear the gate. `cons_rae0` is clean. `cons_raepl` clears it on
the velocity path but now carries a named, specific risk (the noise floor), so it is
run with `--elevation-planes all` as planned and read against that risk.

Proceed to the sweep.

---

# Step 0b — the two objections, addressed

Raised after the first probe: if the RA view is a subset of the RAE view, the
gradient norms should not differ; and the RAE plane term must fit the noise floor the
way the RA plane term does before it can replace it.

## 1. Why the loop-0 norms differ, when the views should differ only by an FFT

**The implementation is symmetric.** Audited against the requirement that the only
distinction be which FFTs are applied:

| | RA view | RAE view |
|---|---|---|
| angle FFT size | `radar.n_angle` | `plan["n_angle"] = radar.n_angle` — same |
| azimuth taper | `fft_window("azimuth", n)` | `fft_window("azimuth", n_az)` — same call |
| in the trainer | `None` (`array_window_kind() == "none"`) | `None` — both untapered |
| elevation taper | n/a | none, by design (3.8) |
| slots sharing channels | averaged | averaged |
| range bins | `keep_index` | `keep_index` |
| channels | `azimuth_tx` only: 2 of 3 (HuPR), 9 of 12 (cascade) | every TX--RX channel |

So there is no taper, size or binning asymmetry. The azimuth array occupies **one
elevation row** of the 2D grid, and the grid's azimuth slots share the azimuth
array's origin.

**At the complex level the premise is exactly right.** The elevation DFT is
invertible, so the set of complex RAE bins determines the per-row azimuth spectra,
one of which *is* the RA view. Complex RA is recoverable from complex RAE.

**At the magnitude level it is not, and that is the whole difference.** The loss
compares $|\cdot|$, and the elevation DFT mixes the rows *before* the modulus, so
$\{|\mathrm{RAE}[\cdot,\cdot,e]|\}_e$ does not determine $|\mathrm{RA}|$. Nothing is
lost from the data; what is lost is the ability to read one row's magnitude out of
the mixed magnitudes.

**And that accounts for the numbers.** Normalising each term's gradient norm by its
own value:

| | term value | scene materials | scene normals | absolute gain |
|---|---|---|---|---|
| **RT-Pose 176** raw ratio RAE/RA | 0.564 | 0.434 | 0.485 | 0.503 |
| per unit of loss | — | **0.769** | **0.860** | **0.891** |
| **HuPR 38** raw ratio | 0.826 | 0.869 | 0.732 | 0.752 |
| per unit of loss | — | **1.052** | **0.887** | **0.911** |

Between 85 and 95 per cent of the apparent gap is that the RAE term simply *has a
smaller residual at the converged state*, not that its path is weaker: per unit of
loss the two deliver the same gradient to the scene, to within 11--23 per cent on the
cascade and 5--11 per cent on HuPR. The RAE residual is smaller for the same reason
the magnitudes are not invertible — row errors that are out of phase partially cancel
in the modulus.

The remaining 10--20 per cent on the cascade is the honest cost of the consolidation:
the part of the azimuth row's error that the mixed magnitudes cannot see. It is
small, it is larger where there are more elevation bins (8 on the cascade, 2 on
HuPR), and it is what the fit-level sweep is there to price.

## 2. The RAE plane term now fits the floor

`--rae-floor-gradient on` (commit e7a57d6) removes the `detach()` on the floor inside
the RAE plane term, default off so the shipped recipe is unchanged. Measured against
the reference:

| | `plane_ra` (reference) | `elevation_plane_rae`, detached | `elevation_plane_rae`, gradient on |
|---|---|---|---|
| RT-Pose 176, \|g\| noise_model | 0.1684 | 0 | **0.1708** |
| HuPR 38, \|g\| noise_model | 0.2478 | 0 | **0.2507** |

Within 1.4 per cent of the reference on both datasets, and nothing else moved: the
velocity norm (0.3743 / 1.824) and the term value (0.7469 / 1.2590) are identical to
the detached run. The change is surgical.

**One acceptance check carries forward.** `docs/FIDELITY_FIX_RESULTS.md` 3.8 detached
the floor *at the same time* as correcting its scale factor from 1.02 to the exact
count-and-window factor, and the failure it was reacting to — the learned floor
sitting 2.5--3.1x under the measured off-person power on the cascade (183 2.50, 186
2.88, 153 3.13, against a healthy 0.69) — was attributed to that wrong factor. The
detach was bundled with the correction, never measured against it. With the exact
factor in place the gradient should be safe, but the first fit run with
`--rae-floor-gradient on` must re-measure that ratio and compare it to 0.69 before
any result from it is believed.

## Consequences for the sweep

- `cons_raepl` and `cons_both` run with `--rae-floor-gradient on`, since they drop
  the floor's other gradient path.
- Their acceptance includes the floor-against-measured ratio, not only fidelity and
  the gate.

---

# Step 0c — full audit of RA against RAE

Everything that differs between the two loop-0 terms and the two plane terms, beyond
the elevation FFT. Checked in the code, not inferred.

## Confirmed identical

| | RA | RAE |
|---|---|---|
| angle FFT size | `radar.n_angle` | `plan["n_angle"] = radar.n_angle` |
| azimuth grid | 8 slots (HuPR), 86 (cascade) | 8, 86 — **equal, measured** |
| azimuth taper | `fft_window("azimuth", ·)` = `None` in the trainer | same call, same `None` |
| channels sharing a slot | averaged | averaged |
| range bins | `keep_index` | `keep_index` |
| loop-0 source | `measured_cube[0].abs()` | `rae_cube[0].abs()` |
| `--static-anchor dc-mean` | `measured_cube.mean(0)` | `rae_cube.mean(0)` — **honoured by both** |
| plane index | `ra_doppler_index` | the same `ra_doppler_index` |
| plane target storage | float16 + per-view rms (`--plane-target-dtype float16`, shipped) | float16 + per-view rms |
| divisor | mean of the target's square, detached, per view | the same, with the band's own share |
| the learned floor | added to the *prediction* | added to the *prediction* |

## Real deviations

1. **Channel set** — intended, and the point of the consolidation. RA is the azimuth
   row only: 2 of 3 TX on HuPR (8 of 12 channels), 9 of 12 on the cascade (144 of
   192). RAE is every channel on the 2D grid.
2. **The RAE grid is sparse, and the cascade's is very sparse.** Occupied cells:
   HuPR 12 of 16 (75%), cascade **134 of 602 (22%)**. The RA grid is fully occupied
   (8 of 8, 86 of 86). So the RAE view transforms a mostly-empty aperture, which is
   part of why its per-cell residual, and therefore its raw gradient norm, is lower.
3. **The elevation DFT zero-pads on the cascade: length 8 over 7 occupied slots.**
   The azimuth axis does not (86 slots, 86-point FFT in the trainer). So within the
   RAE view one axis is exact and the other is interpolated, which makes the 8
   elevation bins non-independent and dilutes the per-cell error further.
   **The fix exists and is not merged:** commit 5807da1 on `runtime/elevation-term`,
   *"the exact elevation DFT, no zero padding (n_el 8 -> 7 on the cascade; HuPR
   unchanged at 2)"*, touching `data.py` and `forward_render.py` only. If the
   consolidation is going to stand on the RAE view, this belongs on the branch first:
   it is the one remaining difference that is an artefact rather than a design choice.
4. **Weighting** — the RAE pair is multiplied by `--elevation-weight` (default 1.0)
   and shares that single knob; the RA terms carry no weight. Identical at the
   default, but the pair cannot be weighted apart.
5. **The floor's extra scale** — the RAE plane term multiplies the floor by
   `elevation_floor_factor` (1.842 cascade, 1.500 HuPR) to carry it to an RAE cell.
   Necessary, not a bug, but it is a second place the floor can be got wrong, and it
   was got wrong once (3.8, factor 1.02).
6. **`--ra-loss pearson`** applies only to the RA loop-0 term. Irrelevant at the
   default `l2`, but an objective built on RAE loses that option.
7. **Ablation hooks** — `_ABLATE_TERM` gates only the RA terms; the RAE pair is
   ablatable only through its shared weight until the two enum values are added.
8. **Plane band** — `--elevation-planes motion` by default. Addressed: the
   consolidations run with `all`.
9. **Floor gradient** — addressed by `--rae-floor-gradient on`.

## What this changes

Items 8 and 9 are already handled. Item 3 is the one that should be fixed before the
sweep if the consolidation is meant to be a fair test: cherry-pick 5807da1 onto
`ablation/objective-terms`, re-run the two loop-0 probes, and see whether the
per-unit-of-loss gap on the cascade (0.77--0.89) closes toward HuPR's (0.89--1.05).
That is a cheap, decisive check, and it is the last mechanical difference between the
two views.

Items 1, 2, 4, 5, 6 and 7 are design, not defects, and belong in the write-up rather
than in a fix.

---

# Step 2 — the HuPR gate, written and validated

`downstream/tools/static_viz/gate_hupr_node.py`, the HuPR twin of `gate_node.py`:
the five real-trained MSCSA-PRGCN models on a candidate's renders and on the real
cubes, per sequence, AP with the seed mean and spread.

It drives the released evaluation path (`HuPRTrainer.evaluate`) rather than
reimplementing AP, because that path is what produced every HuPR number in the paper.
A subclass overrides `_test_entries` to hand it one sequence at a time; the results
root is a scratch tree whose `train/real/seed*` is a symlink, so nothing is written
into the reported results; the cache root is a scratch tree whose real cubes are links
to `4DPS_downstream_gvhmr` and whose rendered cubes are materialised from the
candidate's products through `Source.materialise`.

**Validation, the whole test split under seed 0:**

```
VALIDATE: gate AP 0.6687   committed real__real seed0 AP 0.6687   delta 0.0000
VALIDATE OK
```

Exact. The gate reproduces the committed metrics bit for bit, so its real column can
be trusted as the reference the ablation is read against. `--validate` exits non-zero
on any disagreement.

Environment: `/data/mmsd/envs/micromamba/envs/hupr_mscsa/bin/python`, one GPU.

Both gates now exist. The sweep can be read on both datasets.

---

# Steps 3–5 — the consolidation sweep: results

Run 2026-09-22, 17:00–21:20 ET on the node's 8 H100s. **102 fits, zero failures.**

Baseline: the shipped renders (`/data/mmsd/4DPS_products_static`), which needed no
re-fit — their render commits are ancestors of this branch (HuPR 82a7a70 → HEAD is
one `mm4DPS` commit, the default-off floor flag; RT-Pose 29db2cd → HEAD is six, all
HuPR-only or no-ops for RT-Pose) and the shipped run logged seed 0. Sequences: the
released test splits, 13 RT-Pose and 21 HuPR. One renderer seed, so only aggregates
are interpretable.

## Reconstruction, median paired delta against the shipped render

Counts are sequences moved by more than the ±0.01 run-to-run spread, worse/better.

| | cascade RA | cascade RA_c | cascade RD_occ | cascade MD energy | cascade MD spec |
|---|---|---|---|---|---|
| shipped | 0.860 | 0.495 | 0.679 | 0.959 | 0.968 |
| `cons_rae0` | −0.008 (5/0) | **−0.023 (10/0)** | +0.001 (1/3) | −0.001 (3/2) | +0.001 (0/0) |
| `cons_raepl` | −0.001 (1/1) | **+0.010 (0/6)** | −0.005 (5/1) | **+0.013 (1/7)** | +0.006 (0/2) |
| `cons_both` | −0.007 (4/0) | **−0.016 (10/0)** | −0.005 (2/1) | **+0.013 (0/7)** | +0.005 (0/1) |

HuPR agrees and localises the effect. The horizontal board (2 elevation bins) barely
moves under any candidate; the vertical board (which resolves elevation) tracks the
cascade: `cons_rae0` vertical RA −0.008 (7/0) and RA_c −0.009 (10/0), `cons_raepl`
vertical MD energy +0.005 (0/7), `cons_both` vertical RA −0.008 (3/0) and RA_c
−0.004 (6/0). Every candidate costs a little RD occupied on HuPR (−0.003 to −0.007).

## Downstream, the gap to real

`gap = R→S − R→R` per sequence (AP: `R→R − R→S`); `delta` is the candidate's gap
minus the baseline's. Positive = the ablation widened the gap.

| | RT-Pose Δ MRPE | split | HuPR Δ AP | split |
|---|---|---|---|---|
| `cons_rae0` | 0.0 cm | 8 worse / 5 better | +0.003 | 10 / 7 |
| `cons_raepl` | +0.1 cm | 8 / 5 | **−0.004** | **4 / 9** |
| `cons_both` | −0.2 cm | 5 / 8 | +0.004 | 9 / 7 |

Reference values: RT-Pose R→R 13.3 cm median, R→S baseline 11.5; HuPR R→R 0.678 AP,
R→S baseline 0.536. R→R reproduced to the last digit across independent gate runs.

**No candidate has a measurable downstream cost on either dataset.** Every delta is
inside the noise: ±0.2 cm against per-sequence swings of 100–300 cm between identical
fits, and ±0.004 AP against a 1.7-point seed spread.

## Verdicts

- **`cons_raepl` (5 terms) — take it.** Better fidelity on both datasets, downstream
  neutral on RT-Pose and *better* on HuPR. It is strictly better than the shipped
  objective on everything measured, for one fewer term. It carries two flags with it:
  `--elevation-planes all` (63 planes, device memory 5.14 → ~6.1 GiB on the cascade)
  and `--rae-floor-gradient on`. Without the first it is a band restriction rather
  than a consolidation; without the second the noise floor loses its only substantial
  gradient path.
- **`cons_rae0` (5 terms) — a trade, and strictly worse than `cons_raepl`.**
  Downstream neutral, but clutter-removed RA falls 0.495 → 0.457 on the cascade with
  10 sequences worse and none better. $\rho_c$ is a column the paper reports.
- **`cons_both` (4 terms) — downstream neutral, and the two effects partly cancel.**
  It keeps `cons_raepl`'s micro-Doppler gain and carries most of `cons_rae0`'s
  RA_c cost (−0.023 + 0.010 ≈ −0.013 predicted, −0.016 observed). The choice between
  five terms and four is the choice between keeping Table 2's $\rho_c$ and dropping
  one more term.

The gradient probe had it backwards: it endorsed the loop-0 consolidation and flagged
the plane one over the noise-floor path, and the fits say the loop-0 one costs RA_c
while the plane one gains. The floor worry was real but fixable — the §3.8 detach was
a consequence of the wrong scale factor, and with `--rae-floor-gradient on` the
learned floor reproduces the shipped one to 0.9% (seq 176 cascade log_scale −0.099
against −0.108, range correction [−0.16, +0.09] against [−0.18, +0.12]). That is why
the probe was written as a gate against broken objectives, not as the verdict.

## What this does not yet settle

1. **Test split only.** These renders are ones a consumer *reads*; the rendered→real
   arm, the renders a model *trains on*, is untouched. The corpus-wide confirmation
   is still required before the paper claims a smaller objective.
2. **One renderer seed.** No single-sequence difference is interpretable; the
   aggregates carry the result, and they are trustworthy here mainly because the
   worse/better counts are so one-sided.
3. **The two leave-one-outs and the three mirrors have not been run** (~35 GPU-h).

---

# Step 0 (round 2) — two questions the probe raised, and their answers

## 1. The RA/RAE port is correct. My inference from the gradient norms was not.

I wrote that at the four-term converged state the dropped `loop0_ra` would supply
7.96 of scene-material gradient against `elevation_loop0_rae`'s 1.70, and read that
as the RA_c cost showing in the gradients. **That inference is invalid.** The state
being probed is the one the four-term objective converged to: it minimises the RAE
term and does not minimise the dropped RA term, so the dropped term has a large
gradient by construction. Comparing two terms at a point optimised for one of them
is biased toward the other, and no conclusion about the port follows from it.

The fair comparison is the first probe, taken at the six-term converged state with
both terms in the objective: per unit of loss, RAE delivered 0.77--0.89 of RA's
scene gradient on the cascade and 0.89--1.05 on HuPR. Those numbers stand.

**And the port is now proved, not argued.** `porttest.py`: for random complex
profiles, invert the elevation DFT of the complex RAE view and take the elevation row
the azimuth transmitters occupy. It reproduces the complex RA view to

| array | azimuth TX row | other TX rows | channels clashing in an azimuth-row cell | max relative error |
|---|---|---|---|---|
| cascade | 6 | 0, 2, 5 | 0 | 6.4e-08 |
| HuPR horizontal | 0 | 1 | 0 | 4.2e-08 |
| HuPR vertical | 0 | 1 | 0 | 4.2e-08 |

No channel of a non-azimuth transmitter lands in an azimuth-row cell on any array, so
nothing is averaged across rows. The RAE view contains the RA view exactly.

What the consolidation loses is therefore **only** the modulus: the loss compares
$|\cdot|$, and the elevation DFT mixes the rows before the magnitude is taken, so the
set $\{|\mathrm{RAE}[\cdot,\cdot,e]|\}_e$ does not determine $|\mathrm{RA}|$ even
though the complex data does. That is the whole of the measured RA_c cost, and it is
a property of the objective's form, not of the implementation.

## 2. The person-window term is a local *scene* constraint, because its window is
unbounded in elevation.

The window is bounded in range (`pw_half_r` from `--person-window-range`, ±0.6 m) and
in azimuth (`pw_half_a = 10` sin bins, masked to ±1.2 m lateral). **It is not bounded
in elevation at all**: `_pw_n_el` is the consumer's elevation bin count and every bin
is taken. On the cascade the shipped run prints the box as
`27 range x 21 azimuth x 32 elevation bins` — a full-height column through the
person's range and azimuth.

At loop-0 with clutter kept, that column is mostly scene: the person holds 5.1--7.0%
of the loop-0 target energy on the hard cascade sequences and 17--20% on the good ones
(`docs/STATIC_SEQS_PLAN.md` §2). Matching the column's *shape* therefore mostly means
matching the scene's vertical structure around the person.

Two things keep this from being simply a defect:

- The term still sends more gradient to the person's material (21.34) than to the
  scene's (10.11) on RT-Pose 176. It is not misdirected; it supervises both.
- The failure it was built for was the consumer reading a rendered seated person
  40 cm too high from the elevation pattern. If the rendered column's vertical
  structure is wrong, the consumer misreads the person's height, so constraining the
  column — scene included — is a plausible fix for exactly that failure.

What is a problem is the accounting. In the four-term objective the other three terms
send **zero** gradient to the scene's materials (plane RAE 0, micro-Doppler 0), so the
person window supplies **10.11 of 11.81, or 86%, of all scene-material gradient on the
cascade**. This predates the consolidation — in the six-term probe it was already the
largest single source (110.3 against `loop0_ra`'s 45.7 and `loop0_rae`'s 19.8) — but
dropping `loop0_ra` removed the main alternative.

**Consequence for the sweep in flight:** `abl4_win` is not the small ablation its
justification suggests. On the cascade it removes 86% of the scene's material
supervision. If it comes out removable, that result needs scrutiny rather than
acceptance, and the honest reading would be that the scene's materials are barely
supervised in the four-term objective at all.

**Consequence for the paper:** the term is described as a person term and is, in
effect, a local (range, azimuth, elevation) shape constraint on the person's column in
the consumer's transform. Either bound it in elevation and re-measure, or describe it
as what it does.

---

# Round 2 — micro-Doppler and the person window, from the four-term objective

Baseline: `cons_both`, already fitted and gated on the two test splits, so no re-fit.
102 fits, zero failures. Same protocol: paired per-sequence deltas, one renderer seed,
RT-Pose 164 and 176 reported but excluded from medians.

## Downstream, the gap to the baseline

| candidate | terms | RT-Pose Δ MRPE | split | HuPR Δ AP | split |
|---|---|---|---|---|---|
| `abl4_win` | 3 | +0.1 cm (mean +2.4) | 9 / 4 | **+0.011** | **14 / 3** |
| `abl4_md` | 3 | +0.1 cm (mean +1.3) | 8 / 5 | **+0.015** | **13 / 6** |
| `abl4_both` | 2 | +0.8 cm (mean +3.2) | 9 / 4 | **+0.022** | **15 / 3** |

Round 1's consolidations all sat at or below 0.004 AP with near-even splits. Round 2's
removals are three to six times larger and one-sided. **Both terms are load-bearing,
they cost independently, and dropping both compounds** (+0.011 + 0.015 ≈ +0.022).

RT-Pose medians are flat because its per-sequence noise is large, but the sequences
that move are the ones each term was built for: `abl4_win` costs **+29.3 cm on seq
164**, the seated sequence the window term was added for, and `abl4_md` costs +7.2
there and +2.8 / +2.5 on 168 / 147.

## Reconstruction

Every round-2 removal improves the range--azimuth image and range--Doppler while
costing micro-Doppler:

| | cascade RA_c | cascade RD_occ | cascade MD spec | HuPR vert RD_occ |
|---|---|---|---|---|
| `cons_both` | 0.470 | 0.682 | 0.974 | 0.682 |
| `abl4_win` | −0.001 (3/3) | +0.006 (1/4) | +0.001 (0/0) | +0.011 (1/11) |
| `abl4_md` | **+0.015 (0/10)** | +0.008 (1/4) | **−0.012 (8/0)** | +0.009 (1/9) |
| `abl4_both` | **+0.023 (0/6)** | **+0.021 (0/11)** | −0.009 (4/0) | **+0.019 (0/18)** |

So the four-term objective is **not** the best on every fidelity metric: dropping both
would give cascade RA_c 0.513 and RD occupied 0.699, both above the original six-term
baseline's 0.495 and 0.679. It is the objective that serves the consumer best, and the
paper should say so rather than leave a reviewer to find it in the table.

One finding that did not survive contact: the gradient probe put 86% of the cascade's
scene-material gradient in the person-window term, which suggested removing it would
strip the scene's supervision. It did not — cascade RA moved −0.002 and RA_c −0.001
with balanced counts. Gradient norm is not fit quality. The probe has now had its
ranking overturned twice and its presence/absence signals confirmed twice; it is worth
keeping as a gate and not as a predictor.

## The objective settles at four terms

$$\mathcal{L} = \mathcal{L}_{\mathrm{RAE},0} + \mathcal{L}_{\mathrm{RAE},\mathrm{pl}}
+ w_{\mathrm{MD}}\mathcal{L}_{\mathrm{MD}} + w_{\mathrm{win}}\mathcal{L}_{\mathrm{win}}$$

Two terms removed because they were provably redundant (the complex identity in
`porttest.py`), two kept because removing them costs measurable downstream accuracy.

---

# Round 3 — can micro-Doppler and the person window be merged? No.

Three three-term candidates, each replacing {micro-Doppler, window} with a single
term, against the four-term baseline on the two test splits. 102 fits, zero failures.

| candidate | the single term | HuPR Δ AP | split | RT-Pose Δ MRPE | split |
|---|---|---|---|---|---|
| **C1** `cons3_rd` | the window kept **per Doppler plane**, one rms over the whole (plane, range, azimuth, elevation) cube | **+0.021** | 15 / 3 | +1.1 cm | 7 / 6 |
| **C2** `cons3_consumer` | the window on the consumer's Doppler-**summed** band | **+0.051** | 17 / 2 | −0.3 cm | 6 / 7 |
| **C3** `cons3_mdwin` | micro-Doppler over the **window's box** | **+0.014** | 16 / 2 | +0.3 cm (mean +3.0) | 11 / 2 |
| *reference* | `abl4_both`: just delete both terms | *+0.022* | *15 / 3* | *+0.8 cm* | *9 / 4* |

**C1 costs exactly what deleting both terms costs.** C2 costs more than twice that.
C3 is the least bad and still nowhere near the four-term objective.

## Why C1 failed, which is not what it looks like

C1 was the design that should have worked: all four axes kept, one normalisation over
the whole cube so the relative energy between planes survives. The probe at each
candidate's own converged state, velocity gradient per term evaluated alone:

| | micro-Doppler (4-term) | loop-0 window (4-term) | C1's resolved window |
|---|---|---|---|
| RT-Pose 176 | **3.01** | 0.0025 | **0.519** |
| HuPR 38 | **0.999** | 0.0008 | **5.52** |

**The two datasets fail for opposite reasons, and both trace to the normalisation.**

*RT-Pose is under-supervised.* `_pw_keep_dc` is true there and `_PW_DC_SCALE` is 64
against `_PW_PLANE_SCALE` 2, so one enormous static plane dominates a single global
rms and the 63 moving planes barely enter the loss. C1 degenerates toward the loop-0
window, which is why it lands indistinguishable from deleting both terms.

*HuPR is over-driven.* Clutter is removed there, so there is no DC plane and all 15
planes are moving. C1's window then sends 5.5x micro-Doppler's velocity gradient and
4.9x its person-material gradient (180 against 37). A cell-wise $L_2$ over a
Doppler-resolved cube is a far harsher demand than matching the marginal:
micro-Doppler sums power over space first, so weak limb returns spread over many
cells still count, while a per-cell $L_2$ makes them negligible and demands exact
agreement everywhere. The fit chases it and gives ground elsewhere — HuPR vertical
micro-Doppler energy −0.009 (8 worse / 1 better) and vertical RA_c −0.010 (11 / 1).

So the global rms preserves the relative plane energy, as intended, and at the same
time lets the DC swamp one dataset while making the other's objective too aggressive.
**C1 is mis-normalised, not refuted.** A corrected variant would exclude the DC plane
from the resolved window on RT-Pose (or normalise it apart from the band) and on HuPR
either weight the term down or compare the Doppler marginal *within* the window rather
than cell by cell. That is a post-submission experiment with a specific hypothesis,
not a guess.

## One signal that survives every candidate

Every round-2 and round-3 removal **improves range--Doppler agreement**, and C1 most
of all: cascade RD occupied +0.017 (1 worse / 10 better), HuPR vertical +0.026
(**1 worse / 20 better**), the most one-sided effect measured in the whole sweep. The
Doppler-resolved window genuinely helps the Doppler product; it just does not do
micro-Doppler's job. The loop-0 window's cost to range--Doppler is real and worth a
sentence in the paper.

## Conclusion

**Four terms is the floor.** Micro-Doppler and the person window are orthogonal by
construction — one keeps Doppler and discards space, the other keeps space and
discards Doppler — and all three attempts to build a single term carrying both land at
or worse than deleting them. The ablation the paper reports is therefore: six terms to
four, two removed as provably redundant, two kept because removing them costs
measurable downstream accuracy, and three merge attempts that fail for a stated
reason.

---

# Round 4 — C1b, the 2x2 objective, adopted as the default

$$\mathcal{L} = \underbrace{\mathcal{L}_{\mathrm{RAE},0} + \mathcal{L}_{\mathrm{RAE},\mathrm{pl}}}_{\text{global}}
+ \underbrace{\mathcal{L}^{\mathrm{win}}_{\mathrm{RAE},0} + \mathcal{L}^{\mathrm{win}}_{\mathrm{RAE},\mathrm{pl}}}_{\text{windowed on the person}}$$

One functional -- a normalised $L_2$ on $|\mathrm{RAE}|$ -- over
{global, windowed} x {loop-0, per plane}. The windowed loop-0 term keeps clutter, as
the global loop-0 term does; the windowed plane term excludes DC and removes clutter,
as the global plane term does. Both use the **trainer's** RAE view, not the consumer's
beamformer: the objective should not be designed around the downstream model.

## Why this worked where C1 did not

The probe at the four-term baseline's converged state, which is the only fair place to
compare a candidate term with the term it would replace:

| | velocity grad | ratio to micro-Doppler | person material | ratio |
|---|---|---|---|---|
| RT-Pose, micro-Doppler | 3.01 | — | 270.6 | — |
| RT-Pose, windowed plane term | **2.401** | **0.80x** | 246.9 | 0.91x |
| *C1's single resolved window* | *0.519* | *0.17x* | *21.96* | *0.08x* |
| HuPR, micro-Doppler | 0.999 | — | 37.07 | — |
| HuPR, windowed plane term | **0.995** | **1.00x** | 39.76 | 1.07x |
| *C1's* | *5.52* | *5.5x* | *180.1* | *4.9x* |

Splitting loop-0 from the planes gives each its own normalisation, and the plane term
excludes DC exactly as the global plane term does -- so the x64 static plane can no
longer swamp the moving planes, which was C1's RT-Pose failure, and the term is no
longer sharing a divisor that made it over-drive on HuPR.

The control that separates the two mechanisms: micro-Doppler's own cells compared
**cell by cell** instead of after summing power over space gave 1.14x on RT-Pose and
**2.13x** on HuPR. So dropping the spatial sum does roughly double the gradient on
HuPR -- but the windowed plane term lands at 1.00x, so windowing and normalising
properly cancels that inflation.

## Results against the four-term baseline

**Reconstruction, the best of any candidate:**

| | baseline | C1b | delta | worse/better |
|---|---|---|---|---|
| cascade RA_c | 0.470 | **0.525** | **+0.049** | 1/11 |
| cascade RD occupied | 0.682 | 0.699 | +0.016 | 0/9 |
| cascade RA | 0.846 | 0.860 | +0.007 | 0/4 |
| cascade MD spectrogram | 0.974 | 0.966 | −0.010 | 7/0 |
| HuPR vert RD occupied | 0.682 | 0.698 | +0.015 | 1/16 |
| HuPR MD spectrogram, both boards | 0.995 / 0.987 | 0.995 / 0.987 | **−0.000** | — |

Cascade clutter-removed RA at 0.525 is **above the original six-term objective's
0.495**, and HuPR micro-Doppler is preserved exactly, which is the velocity
supervision working as the probe predicted.

**Downstream:**

| | Δ | split |
|---|---|---|
| RT-Pose, all 13 | +0.2 cm (mean +1.9) | 8 / 5 |
| RT-Pose, excluding 164 / 176 | **+0.1 cm (mean −0.0)** | 6 / 5 |
| HuPR | **+0.005 AP (mean +0.010)** | **10 / 6** |

Four times closer than anything before it (C3 +0.014, C1 +0.021, deleting both
+0.022, C2 +0.051) and the first near-even split. The bar was ≤0.004 with a near-even
split, so this is borderline rather than a pass.

**Seq 164 costs +21.6 cm** (28.5 → 50.1), the seated sequence the person-window term
was built for. Two candidate causes, separable: the view change (Win constrained the
shape in the *consumer's* transform, which is what misread the seated person by
40 cm), or the elevation column (the windowed terms span all 8 elevation bins while a
seated person occupies few). `c1b_el` tests the second.

## Adopted

`OBJECTIVE=2x2` is the default in `chain.py`; `4term` and `legacy6` remain reachable.
The +0.005 AP downstream regression is real and is adopted deliberately, for the
reconstruction gain and the uniform structure. The paper must state the trade rather
than let a reviewer find it.

---

# c1b_el — measured and not run: the elevation axis has nothing to bound

The windowed RAE terms span every elevation bin, which on the cascade is a full-height
column through the person's range and azimuth, mostly scene. `--window-el-bound on`
was implemented to bound them to the bins the person actually moves in, read off the
windowed *plane* target (clutter removed, so the moving energy identifies its own
extent, with no axis arithmetic and so no chance of reimplementing the elevation
convention wrongly).

Measured at the `c1b` fits' converged state, the mean per-bin share of the peak
moving bin, sorted:

| | profile |
|---|---|
| RT-Pose 176, 8 bins | **0.94, 0.85, 0.82, 0.72, 0.67, 0.66, 0.65, 0.54** |
| HuPR 38, 2 bins | **1.00, 0.59** |

**The moving energy is flat across elevation.** The weakest cascade bin still holds
54% of the peak. No threshold isolates the person: 0.5 keeps all eight bins, 0.6 keeps
seven, 0.7 keeps four while discarding bins at two-thirds of the peak. The cause is
the view itself -- eight bins over seven occupied slots with zero padding on a sparse
aperture, so a compact source smears across the whole axis.

**So the bound cannot work, and the sweep was not run.** It would have cost 5.2
GPU-hours to reproduce `c1b`.

## What this settles about seq 164

C1b costs +21.6 cm on RT-Pose seq 164, the seated sequence the person-window term was
built for. Two hypotheses:

1. the elevation column -- the windowed terms span every elevation bin;
2. the view change -- the person window constrained the shape in the **consumer's**
   beamformer, and C1b's windowed terms use the trainer's RAE view.

The profile rules out (1) as actionable: the elevation axis has no structure to
exploit, so the column is not separable and cannot be the fixable cause. **(2) is the
remaining explanation**, and it is the one place where keeping the objective
independent of the downstream model has a measured cost. The test would be a variant
whose windowed loop-0 term uses the consumer's view while the plane term stays in the
trainer's -- a post-submission experiment.
