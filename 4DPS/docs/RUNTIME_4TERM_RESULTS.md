# Runtime under the four-term objective, on the workstation's two RTX 4090s

Branch `runtime/4term`, from `ablation/objective-terms` at 8761e41. Plan:
`misc/RUNTIME_4TERM_PROMPT.md`. Every number below was measured on this machine, at
`--frame-batch 4` on RT-Pose and 24 on HuPR (the chain's own values: 576 channel-frames
over the training sub-array's channels), 400 epochs, the four-term objective as
`chain.py`'s default.

## 0. Two departures from the plan, both forced

**The branch is cut from `ablation/objective-terms`, not from `main`.** The plan says
`runtime/4term` is made from `main` after the four-term default has landed there. It has
not landed: `main` at c122492 has neither the default (`OBJECTIVE=4term`, 0d31d30) nor the
tooling the plan asks for -- `prepare_ref.sh`, `gate_hupr_node.py`, `porttest.py`,
`docs/OBJECTIVE_ABLATION.md` are all only on `ablation/objective-terms`. That branch does
contain `main` (`git merge-base --is-ancestor main ablation/objective-terms` holds), so
branching from it is the plan's intent -- the four-term objective on top of current `main`
-- and nothing of `main` is lost.

`runtime/elevation-term`, the branch the ports come from, is NOT a descendant of `main`:
it forked at 431b4b7 and `main` has moved on. So its commits are ports, not merges, and
one of them could not be cherry-picked at all (section 3).

## 1. The memory constraint fired: the fit does not fit

Measured before anything else, as the plan asks. **Under the four-term objective the
RT-Pose fit does not fit on a 24 GB card at the shipped `--frame-batch 4`.**

| | peak allocated | peak reserved | of a 23.53 GiB card |
|---|---|---|---|
| RT-Pose 176 | 19.09 GiB | 22.91 GiB | 637 MiB spare |
| RT-Pose 108 | 18.65 GiB | 22.78 GiB | 780 MiB spare |
| HuPR 38 | 5.40 GiB | 10.65 GiB | |
| HuPR 256 | 5.14 GiB | 10.24 GiB | |

Three findings, in the order they were hit:

1. **With the default allocator it OOMs outright.** The first RT-Pose run died with 21.60
   GiB allocated and 23.43 GiB in use, trying for 512 MiB more. `PYTORCH_ALLOC_CONF=`
   `expandable_segments:True` -- which changes no arithmetic, only how the caching
   allocator segments its pool -- is what makes it fit. Every number here is measured with
   it set, on baseline and candidates alike.
2. **It OOMs *after* training, not during it.** All 400 epochs run; the failure is in
   `calibrate_person_level`, a post-training step whose `adapter.polar` wants a 512 MiB
   buffer while 2.9-3.7 GiB is still held in the training step's CUDA-graph private pool.
   Releasing that pool before the calibration would give the fit several GiB of headroom
   and is the obvious fix; it is a memory change, not a runtime one, so it is reported
   rather than taken.
3. **Any co-resident process pushes it over, and the harness is one.** `devloop run
   --rebuild-person` does the person stage in-process, so the parent holds a ~686 MiB CUDA
   context on the same card for the whole fit. With it, RT-Pose OOMs at the calibration;
   without it, the same sequence completes with 637 MiB spare. So the baseline and every
   candidate are run from a base that already holds the person stage
   (`devloop prepare --base pbase`, no `--rebuild-person`), which is also what makes a
   candidate differ from the baseline by the trainer alone.

`--frame-batch` was therefore NOT lowered: at 4 the fit completes, so baseline and
candidates are comparable with each other and with the plan's H100 reference points. The
margin is 637 MiB, and a card doing anything else at the same time does not have it.

## 2. The baseline, measured here

`b4`, the four-term default, five sequences, 400 epochs. The velocity stage onward
(`velocity_onward` + `evaluate`); `step ms` is the 400-epoch median, `train s` the sum of
the epoch times.

| sequence | step ms | epoch s | train s | wall s | peak alloc GiB |
|---|---|---|---|---|---|
| RT-Pose 176 (near-static, the hard one) | 43.22 | 3.241 | 1308 | 1552 | 19.09 |
| RT-Pose 108 (near-static) | 41.83 | 3.137 | 1265 | 1441 | 18.65 |
| RT-Pose 191 (well-behaved) | 40.61 | 3.046 | 1227 | 1409 | 18.17 |
| HuPR 38 | 24.28 | 0.607 | 251 | 331 | 5.40 |
| HuPR 256 | 23.83 | 0.596 | 246 | 343 | 5.14 |

75 steps an epoch on RT-Pose (600 frames at `--frame-batch 4`, two of every eight held out
of the training set by the protocol), 25 on HuPR (at 24).

**The four-term objective is SLOWER per step than the six-term one it replaced**, on the
same card: 40.6-43.2 ms against the plan's 37.8-40.2 ms reference for the six-term fit on a
4090. Dropping the two range-azimuth terms does not pay for what the consolidation costs,
because the plane RAE term now reads every Doppler plane. The elevation term's target store
is **6.08 GiB** on the cascade -- `(63, 244, 86, 8)` planes plus `(244, 86, 8)` loop-0, the
plan's ~6.1 GiB confirmed -- and 0.73 GiB on HuPR, whose plane count is 40 rather than 63
because `--doppler-support-snr 1.0` narrows the band to the widest support in the run.

The fidelity baseline these are gated against:

| sequence | radar | RA | RAc | RD | MD centroid | MD spread | MD energy |
|---|---|---|---|---|---|---|---|
| RT-Pose 176 | cascade | 0.725 | 0.089 | 0.622 | 0.977 | 0.892 | 0.945 |
| RT-Pose 108 | cascade | 0.825 | -- | -- | -- | -- | -- |
| RT-Pose 191 | cascade | 0.839 | 0.425 | 0.679 | 0.934 | 0.801 | 0.953 |
| HuPR 38 | horizontal | 0.935 | 0.583 | 0.912 | 1.000 | 0.991 | 0.994 |
| HuPR 38 | vertical | 0.914 | 0.602 | 0.823 | 0.997 | 0.981 | 0.987 |
| HuPR 256 | horizontal | 0.962 | 0.730 | 0.934 | 1.000 | 0.984 | 0.994 |
| HuPR 256 | vertical | 0.927 | 0.695 | 0.858 | 0.997 | 0.941 | 0.984 |

RT-Pose 108 is near-static enough that every metric built on the moving return is
undefined there (`nan` for the clutter-removed and micro-Doppler columns, and its occupied
range-Doppler correlation is -0.256 on a handful of frames). It is a timing and a
correctness sequence, not a fidelity one; its RA column is the only figure of merit it
carries.

## 3. The ports, and why one of them is a rewrite

The plan says to take `d5c2ef5`, `59176a5` and `d4daab3` as gated-bitwise, and `3ab773d`
as a tool. Two of the four cherry-pick cleanly; `d5c2ef5` does not, and the reason is worth
stating because it changes what the plan's table means.

| commit | what happened |
|---|---|
| `3ab773d` route gate | cherry-picks clean |
| `d4daab3` path rates as a view | cherry-picks clean; touches neither the elevation term nor the objective |
| `59176a5` RAE store from the step's gather | applied by hand, 16 insertions / 8 deletions -- exactly the original's stat |
| `d5c2ef5` fused RAE views | **conflicts; rewritten** |

**`d5c2ef5` is not independent of `c13b99d`.** On `runtime/elevation-term` the stack is
`c13b99d` -> `d5c2ef5` -> ... , and the fused commit is written against the person-rows
restriction: its `elevation_terms` reads `rae_rows_index` / `rae_rows_weight` /
`rae_doppler_energy`, store keys that `c13b99d` creates. Cherry-picked alone it leaves
conflict markers in the middle of the term. So "take these three, they were gated bitwise"
cannot be done as stated: one of them presupposes the item the plan asks to re-evaluate.

It is still portable, because `forward_render.profiles_to_rae_spectrum_fused` -- the part
that does the work -- takes `range_index` as either `(R,)`, every unit the same rows, or
`(units, R)`, the person's rows per unit. The `forward_render.py` and `data.py` halves of
the commit are purely additive and apply unchanged; only the `velocity_field.py` half was
rewritten, to call the fused function with `keep_index` (every kept row) and to keep the
ablation branch's floor-gradient switch, which did not exist when `d5c2ef5` was written.

**And this is why its measured gain does not carry over.** `d5c2ef5` was measured at
15.06 -> 15.02 ms a step: nothing, because `c13b99d` had already cut the term's tensors to
~4 M cells and the chain it replaced was launch-bound. Under the four-term objective the
same substitution runs on every kept range row and, at `--elevation-planes all`, every
rendered plane -- on the cascade 63 planes x 244 rows x 128 angle x 8 elevation, about
63 M cells a step against ~4 M. The plan's 0.04 ms is a measurement of a different
workload, and the port had to be re-measured from scratch (section 4).

Both halves were checked before any fit: an eager A/B on CPU of `profiles_to_rae_magnitude`
against `profiles_to_rae_spectrum_fused(...).abs()`, on both shapes the trainer uses
(loop-0 `(units, tx, rx, K)` and planes `(units, planes, tx, rx, K)`), is bitwise in the
values and in both gradients. That check ran at a power-of-two angle length, so it exercised
only the sign-fold branch; the fit's own lengths are `radar.n_angle` = **86** on the cascade
and **16** on HuPR, which take opposite branches -- 16 folds the azimuth fftshift into the
input signs, 86 is not a power of two and keeps the roll after the FFT. Both are covered by
the GPU gate of section 4, one per dataset.

## 4. `c1`: the three ports, gated bitwise

`c1` = the route gate (`3ab773d`) + the fused RAE views on every kept row (the `d5c2ef5`
rewrite) + the RAE store read from the step's one gather (`59176a5`) + the path rates as a
view (`d4daab3`). Prepared from the same `pbase` workspaces as `b4`, so it differs from the
baseline by the trainer's code alone.

### The gate

Two instruments, both of them the branch's own, and `c1` passes both.

1. **The 20-step deterministic dump** (`MM4DPS_PRINT_STEPS=20`, `--deterministic on`, a
   one-epoch fit whose trainer argv is copied from `b4`'s own `trainer_argv.json`, so the
   two codes are compared at identical flags): the first 20 steps' per-unit terms and every
   parameter group's gradient norm at nine significant digits. **All 40 lines identical**
   on RT-Pose 176 and on HuPR 256.
2. **The cell-by-cell route check** (`MM4DPS_RAE_CHECK=1`, `rae_route_check`), which is the
   decisive one, because a sum over millions of cells hides a one-ulp cell below ~1e-9: the
   per-unit terms and the gradients into the four profile tensors and into the floor,
   fused against reference, on the same inputs. **Bitwise in the values and in all five
   gradients** on both datasets -- and this exercises the *compiled* gather on the GPU,
   which the CPU A/B of section 3 could not.

Between them the two fold-branches are both covered: HuPR's angle length 16 folds the
azimuth fftshift into the input signs, the cascade's 86 keeps the roll.

A third, weaker confirmation falls out of the run itself -- but it is weaker than it looks,
and not equality. The bitwise gate holds under `--deterministic on`; the runs that produce
these products use the shipped `--deterministic off`, where the sorted-Doppler route makes
two runs of the SAME code differ at 4e-7 in the gradients from step 0 and at 3e-3 in the
losses by step 19. So `c1`'s fidelity is expected to sit within the run-to-run spread of
`b4`'s, not to equal it, and the table below is read against that spread rather than as an
identity. The identity claim rests on the deterministic dump and the route check above.

### What `c1` costs and saves

Same queue order and the same pair of concurrent jobs as `b4` on both cards, so the two are
comparable step for step.

| sequence | b4 step ms | c1 step ms | delta | b4 train s | c1 train s |
|---|---|---|---|---|---|
| RT-Pose 176 | 43.22 | 43.97 | +1.7% | 1308 | 1333 |
| RT-Pose 108 | 41.83 | 42.87 | +2.5% | 1265 | 1297 |
| RT-Pose 191 | 40.61 | 40.01 | -1.5% | 1227 | 1218 |
| HuPR 38 | 24.28 | **21.30** | **-12.3%** | 251 | 223 |
| HuPR 256 | 23.83 | **20.99** | **-11.9%** | 246 | 219 |

**HuPR gains 12%; RT-Pose does not move.** The RT-Pose deltas disagree in sign across three
sequences (+1.7, +2.5, -1.5) and average +0.9%, which is a null, not a regression -- but it
is emphatically not the win the plan's table implies either.

The mechanism is the azimuth length, and it is the crossover `d5c2ef5`'s own commit message
names. HuPR transforms 16 angles: a power of two, so the fused route folds the azimuth
fftshift into the input signs and the roll disappears, over a 0.73 GiB store where the
chain it replaces is launch-bound. The cascade transforms **86 = 2 x 43**, 43 prime: not a
power of two, so `fold_az` is false, the roll still runs after the FFT, and the store is
6.08 GiB -- 42.3 M cells a step against HuPR's. At that size the term is bandwidth-bound,
so removing kernel launches buys nothing and the extra contiguous-buffer write costs
something. `d5c2ef5` was measured at ~4 M cells and reported 0.04 ms; at 42.3 M cells the
same substitution is a wash.

`d4daab3` (the path rates as a view, 1.16 ms on the branch) should be a gain on both
datasets and is presumably what keeps RT-Pose from regressing outright. Separating it from
the fused route needs a probe series, not a full fit: `HUPR_RAE_ROUTE=reference` switches
the fused substitution off inside `c1`'s own build, so an 8-epoch trainer-only fit on one
idle card isolates it exactly. That is section 7.

### The bitwise change doubles as the run-to-run null

`c1` is bitwise `b4` under `--deterministic on`, and these products were made with the
shipped `--deterministic off`. So every `c1` - `b4` fidelity difference is **pure run-to-run
noise of this pipeline**, measured rather than quoted -- an empirical null, and the right
yardstick for `c2`.

| sequence | radar | RA | RAc | RD | MDc | MDs | MDe |
|---|---|---|---|---|---|---|---|
| RT-Pose 176 | cascade | -0.001 | +0.010 | -0.008 | -0.001 | +0.001 | -0.005 |
| RT-Pose 108 | cascade | -0.006 | -- | -0.033 | -- | -- | -- |
| RT-Pose 191 | cascade | +0.005 | -0.002 | -0.004 | +0.008 | **-0.148** | -0.017 |
| HuPR 38 | horizontal | +0.003 | +0.011 | +0.003 | -0.000 | +0.003 | -0.002 |
| HuPR 38 | vertical | -0.001 | -0.002 | +0.002 | +0.000 | -0.004 | +0.000 |
| HuPR 256 | horizontal | -0.000 | +0.000 | -0.001 | +0.000 | +0.001 | -0.000 |
| HuPR 256 | vertical | -0.000 | -0.003 | -0.001 | +0.001 | +0.005 | +0.002 |

**The spreads docs/FIDELITY_FIX_RESULTS.md section 1 quotes understate this configuration.**
Against RA 0.005 / RAc 0.008 / RD 0.02 / MD centroid 0.02 / spread 0.06 / energy 0.01, a
change that cannot possibly have altered the result exceeds the quoted spread in five
cells, and RT-Pose 191's micro-Doppler spread swings **0.801 -> 0.653**, two and a half
times the quoted tolerance, between two bitwise-identical trainers.

HuPR is tight: nothing off 256 moves past 0.005, and only one cell on 38 (RAc +0.011)
passes its quoted spread. The variability is an RT-Pose property, and it is worst on the
micro-Doppler metrics of the moving sequence.

**This bounds what any single-run results gate on RT-Pose can conclude**, which matters
directly for `c13b99d`: a one-run MD-spread difference smaller than ~0.15 on RT-Pose is not
evidence of anything. `c2` is therefore read against this null, with that limitation stated
rather than papered over.

### The null measured properly: two runs of identical code (2026-09-23)

The single-sample null above was later replaced by the real thing. Two 400-epoch fits of
**bitwise-identical code** (the A1 build, `a1b` and `a1c`), same sequence, same flags:

| RT-Pose 176, cascade | RA | RAc | RD | MD centroid | MD spread | MD energy |
|---|---|---|---|---|---|---|
| `a1c` - `a1b`, identical code | 0.004 | 0.005 | 0.006 | 0.015 | 0.014 | **0.027** |
| the spread FIDELITY_FIX_RESULTS.md 1 quotes | 0.005 | 0.008 | 0.02 | 0.02 | 0.06 | **0.01** |

Five of the six sit inside their quoted tolerance. **Micro-Doppler energy does not: its
run-to-run spread on this sequence is at least 0.027, nearly three times the 0.01 it is gated
at.** Anything smaller than ~0.03 in MD energy on RT-Pose is noise, and a gate that treats
0.01 as significant there will raise false alarms -- as it did on this branch, where the A1
run's `-0.027` looked like a regression from a change that is bitwise by construction, and
the repeat came back `+0.000` from the same code.

### The full null, from every bitwise-equivalent run on disk

`b4`, `c1`, `a1`, `a1b` and `a1c` all compute bitwise-identical arithmetic -- `c1`'s three ports
were gated bitwise, A1 was gated bitwise, and `release_graph_pool` is memory-only -- so every
pairwise difference among their products is run-to-run noise. That is four independent runs on
RT-Pose 176 and two or three on the rest: **15-16 pairwise samples per metric**, without running
anything new. (`c2` is excluded: it changes what is supervised.)

Range (max - min) over each sequence's runs:

| sequence / board | runs | RA | RAc | RD | MDc | MDs | MDe |
|---|---|---|---|---|---|---|---|
| RT-Pose 176 cascade | 4 | 0.004 | 0.015 | 0.008 | 0.015 | 0.014 | **0.027** |
| RT-Pose 108 cascade | 2 | 0.006 | -- | **0.033** | -- | -- | -- |
| RT-Pose 191 cascade | 2 | 0.005 | 0.002 | 0.004 | 0.008 | **0.148** | 0.017 |
| HuPR 38 horizontal | 2 | 0.003 | 0.011 | 0.003 | 0.000 | 0.003 | 0.002 |
| HuPR 38 vertical | 2 | 0.001 | 0.002 | 0.002 | 0.000 | 0.004 | 0.000 |
| HuPR 256 horizontal | 3 | 0.000 | 0.001 | 0.001 | 0.000 | 0.003 | 0.001 |
| HuPR 256 vertical | 3 | 0.001 | 0.003 | 0.002 | 0.001 | 0.005 | 0.002 |

**Three conclusions, and the third is a correction to this document.**

1. **The noise is a property of the sequence, not of the metric.** HuPR is tight everywhere
   (nothing above 0.011, most below 0.005); the cascade is one to two orders of magnitude looser.
   A single tolerance table covering both datasets cannot be right, and the spreads
   docs/FIDELITY_FIX_RESULTS.md section 1 quotes -- measured under the SIX-term objective on a
   different sequence set -- are exceeded on the cascade by five of six metrics: RA 0.006 > 0.005,
   RAc 0.015 > 0.008, RD 0.033 > 0.020, MD spread 0.148 > 0.060, MD energy 0.027 > 0.010. Only
   MD centroid is inside. On HuPR all six are comfortable.
2. **RT-Pose 108 must not be gated on at all, and must not be pooled.** Its clutter-removed and
   micro-Doppler metrics are undefined (section 2) and its occupied range-Doppler correlation is
   -0.256 with a run-to-run range of 0.033 -- a metric whose noise is an eighth of its own
   magnitude carries no information. Pooling it is what would push an RD tolerance to 0.035 and
   hide real effects on 176 and 191, whose RD noise is 0.008 and 0.004.
3. **An earlier claim in this section was over-precise and is withdrawn.** It read "the occupied
   range-Doppler correlation's identical-code spread is 0.006": that was one pair on one sequence.
   Per sequence the RD spread is 0.008 (176), 0.004 (191) and 0.033 (108, degenerate).

**The `c13b99d` verdict stands, read per sequence.** `c2` moved RD by -0.042 on 176 against a
0.008 null (5x) and -0.030 on 191 against a 0.004 null (7x), and on HuPR by -0.009 to -0.018
against nulls of 0.001-0.003 (3-9x), the same direction on all seven pairs. Its micro-Doppler
spread move on 191 (-0.179 against a 0.148 null) is NOT significant, as was already stated. And
the verdict's primary evidence was never the fidelity suite: it is the downstream gate, +11.9 cm
MPJPE on 176 at a seed spread of 3.7-4.2 cm.

### Recommended tolerances for this configuration

Per dataset, the maximum observed range rounded up, with RT-Pose 108 excluded:

| | RA | RAc | RD | MDc | MDs | MDe |
|---|---|---|---|---|---|---|
| **RT-Pose (176, 191)** | 0.011 | 0.015 | 0.010 | 0.015 | **0.150** | **0.030** |
| **HuPR (38, 256)** | 0.005 | 0.015 | 0.005 | 0.005 | **0.010** | 0.005 |

**A max over n runs is a LOWER BOUND that grows with n, not a tolerance** -- and this table caught
itself doing that. Item C's 400-epoch run is another bitwise-equivalent sample, and folding it in
moved RT-Pose 176's RA range from 0.004 (4 runs) to **0.011** (5 runs) and HuPR 256 vertical's MD
spread from 0.005 (3 runs) to **0.010** (4 runs). Both of those had been flagged as "outside the
null" on a change gated bitwise by two instruments on real data, which is the strongest available
evidence that they were noise -- and they were. The numbers above are the n = 2-5 values; they will
keep growing as bitwise changes accumulate.

So the practical rule, which does not depend on getting the tolerance exactly right: **require a
single-run difference to be a MULTIPLE of the observed range, not merely to exceed it.** The
`c13b99d` verdict satisfies that (RD -0.042 against a 0.008 range on 176, 5x; -0.030 against 0.004
on 191, 7x). A difference at 1-2x the range is not evidence, and the two cells this table flagged
were at 1.6x and 2x.

Every bitwise change is a free null sample. Regenerating this table from all of them
(`mmSD/artifacts/.../null2.py`) costs nothing and should be done whenever one lands.

These supersede section 1's spreads **for the four-term objective on these sequences only**; they
are not proposed as a replacement for that table, which describes a different experiment. Two
practical consequences: a single-run RT-Pose gate cannot resolve a micro-Doppler spread change
below 0.15 or an energy change below 0.03, so effects that size need repeats or a different
metric; and HuPR's tightness means it is the sensitive dataset for fidelity, even though RT-Pose
is the sensitive one downstream.

## 5. `c13b99d` under the four-term objective: what is actually at risk

`c2` = `c1` + `c13b99d`, so the `c1` -> `c2` delta isolates the item. Before the numbers,
what the item does and does not drop, because the plan's statement of the risk names the
wrong axis.

**`c13b99d` restricts ROWS, not planes.** It computes the plane part of the elevation term
on the rows of the person's RA mask instead of all 244 kept rows. The Doppler-plane axis is
untouched: at `--elevation-planes all` every one of the 63 rendered planes is still
supervised, including the out-of-band ones, on the person's rows. So the plan's concern --
"including the out-of-band planes' zero targets, which are what keep the rendered person
compact in Doppler" -- is a concern about `--elevation-planes motion`, which this item is
not; the zero targets that keep the person compact in Doppler sit at the person's own
ranges, and those rows are exactly the ones `c13b99d` keeps.

**What it drops is the off-person range rows, across every plane** -- the rendered floor
against the measured floor at ranges the person is not at. Under the six-term objective
that cost nearly nothing, because the RA plane term supervised every range row and the
divisor was left at the reference's, so the surviving cells kept their weight. Under the
four-term objective the RA plane term is gone and nothing else supervises those cells.

**And that lands on the noise floor specifically.** `docs/OBJECTIVE_ABLATION.md`'s probe
found that only `plane_ra` carried real gradient to the nine noise-floor parameters
(0.168 RT-Pose, 0.248 HuPR, against 0 for the RAE plane term, which was detached there),
which is why the four-term default turns `--rae-floor-gradient on`: the RAE plane term is
now the floor's only substantial gradient path. The floor is an OFF-person quantity --
it is estimated from off-person power and its whole job is to describe the cells away from
the mover. `c13b99d` removes most of those cells from the only term that still fits it.
That is the specific, named reason the old verdict cannot be assumed, and it is sharper
than "it removes supervision": the item and `--rae-floor-gradient on` interact, and the
four-term objective needs both.

The port therefore keeps the floor-gradient switch rather than `c13b99d`'s unconditional
detach, gathers the floor to the person's rows, and is gated on results, not bitwise.

### What `c2` costs and saves

| sequence | b4 step ms | c2 step ms | delta | b4 train s | c2 train s | b4 peak alloc | c2 peak alloc |
|---|---|---|---|---|---|---|---|
| RT-Pose 176 | 43.22 | **17.75** | **-59%** | 1308 | 543 | 19.09 | **13.67** |
| RT-Pose 108 | 41.83 | **17.09** | **-59%** | 1265 | 524 | 18.65 | **13.35** |
| RT-Pose 191 | 40.61 | **16.84** | **-59%** | 1227 | 516 | 18.17 | **12.73** |
| HuPR 38 | 24.28 | **14.77** | **-39%** | 251 | 157 | 5.40 | 4.23 |
| HuPR 256 | 23.83 | **15.00** | **-37%** | 246 | 159 | 5.14 | 4.07 |

`c13b99d` is by far the largest effect measured here: it takes the RT-Pose step from 43 ms to
**17 ms** and more than halves the fit. It also **dissolves the memory problem of section 1**:
peak reserved falls 22.91 -> 16.22 GiB on RT-Pose 176, so the fit stops living inside a
637 MiB margin and no longer needs `expandable_segments` or an otherwise-empty card.

### And what it costs, read against the measured null

The comparison that matters is not `c2` against the quoted spreads but `c2` against `c1`,
because `c1` is the same pipeline's noise. Both columns are differences from `b4`.

| sequence | radar | metric | null (c1) | c2 | verdict |
|---|---|---|---|---|---|
| RT-Pose 176 | cascade | RD | -0.008 | **-0.042** | outside |
| RT-Pose 191 | cascade | RD | -0.004 | **-0.030** | outside |
| RT-Pose 108 | cascade | RD | -0.033 | -0.111 | degenerate metric |
| HuPR 38 | horizontal | RD | +0.003 | **-0.010** | outside |
| HuPR 38 | vertical | RD | +0.002 | **-0.018** | outside |
| HuPR 256 | horizontal | RD | -0.001 | **-0.017** | outside |
| HuPR 256 | vertical | RD | -0.001 | **-0.009** | outside |
| RT-Pose 191 | cascade | MD spread | -0.148 | -0.179 | inside the null |
| HuPR 38 | horizontal | RAc | +0.011 | +0.023 | marginal |
| HuPR 38 | vertical | RAc | -0.002 | -0.015 | marginal |

**The occupied range-Doppler correlation falls on every sequence and every board**, by
-0.009 to -0.042, where the bitwise null moves by at most 0.008 and changes sign. Seven of
seven usable radar-sequence pairs move the same way. Nothing else is separable from the
null: RT-Pose's micro-Doppler spread moves less than its own noise, and the RAc changes on
HuPR 38 are marginal in both directions.

That is the predicted failure, in the predicted place. The term that lost its cells is the
one supervising the Doppler planes away from the person, and range-Doppler is the view that
reads exactly those cells -- the rendered floor against the measured floor. It is not a
micro-Doppler regression and not a range-azimuth regression; it is specifically the
range-Doppler image, which is what the argument in section 5 says it should be.

## 6. The downstream gate on `c13b99d`

The five real-trained models per dataset, on the candidate's renders and on the real cubes,
per sequence. RT-Pose: `gate_node.py`, HRRadarPose, the mesh-label seeds
(`4DPS_results_gvhmr`, `Train_gvhmr.json` -- the labels those checkpoints were trained
under). HuPR: `gate_hupr_node.py`, MSCSA-PRGCN, AP by the released evaluator. `+-` is the
spread over the five seeds; the `real` column is the same for both candidates and is the
reference the rendered column is chasing.

**RT-Pose (cm, lower is better):**

| sequence | real MRPE | b4 MRPE | c2 MRPE | real MPJPE | b4 MPJPE | c2 MPJPE | frames > 1 m |
|---|---|---|---|---|---|---|---|
| 176 | 16.5 | 20.0 +- 4.2 | **27.8 +- 3.7** | 17.7 | 24.4 | **36.3** | 1% -> **4%** |
| 108 | 13.3 | 8.4 +- 0.4 | 9.4 +- 1.1 | 11.0 | 12.2 | **17.3** | 0% -> 0% |
| 191 | 8.0 | 9.2 +- 0.5 | 9.6 +- 0.7 | 11.4 | 15.5 | 16.5 | 0% -> 0% |

**MPJPE worsens on all three sequences -- +11.9, +5.1 and +1.0 cm** -- and on 176 the root
error worsens by 7.8 cm, about two seed sigmas, with the share of failed frames going 1% to
4%. This is not noise and it is not marginal: on the hard near-static sequence the rendered
pose error grows by half again.

**HuPR (AP, higher is better):**

| sequence | real AP | b4 AP | c2 AP | b4 gap | c2 gap |
|---|---|---|---|---|---|
| 38 | 0.556 | 0.481 +- 0.041 | 0.508 +- 0.029 | 0.076 | 0.048 |
| 256 | 0.672 | 0.488 +- 0.031 | 0.482 +- 0.033 | 0.184 | 0.190 |

**On HuPR it is neutral.** +0.027 on 38 and -0.006 on 256, both well inside the seed spread.

So the item's cost is **dataset-specific, and it tracks the fidelity evidence exactly**: the
range-Doppler correlation fell by -0.030 to -0.042 on RT-Pose and by only -0.009 to -0.018
on HuPR, and the downstream reading follows -- a large pose regression on the cascade, none
on HuPR. The mechanism of section 5 predicted the metric; the size of the fidelity move
predicted which dataset would pay.

## 7. Attribution: which port does what, on an idle card

The full fits above time a candidate, not an item. The per-item estimator is the one
docs/RUNTIME_RESULTS.md attributes with: the trainer alone, 8 epochs, the median `step_ms`
of epochs 5-8 (after the CUDA-graph capture), one sequence, **the other card left idle**.
`HUPR_RAE_ROUTE=reference` switches the fused substitution off inside `c1`'s own build, so
the fused route is isolated without a second checkout.

| config | RT-Pose 176 | HuPR 256 |
|---|---|---|
| `b4` | 45.63 | 23.73 |
| `+ d4daab3 + 59176a5` (reference route) | **43.82** | 22.61 |
| `+ the fused route` (= `c1`) | 45.37 | **20.83** |
| `c2` | 18.74 | 14.87 |

| item | RT-Pose | HuPR |
|---|---|---|
| `d4daab3` + `59176a5` | **-1.81 ms** | **-1.12 ms** |
| `d5c2ef5` fused route | **+1.55 ms** | **-1.78 ms** |
| `c13b99d` person rows | **-25.1 ms** | **-5.96 ms** |

**The fused route should not be a global default.** It is bitwise either way, so the choice
is free: `HUPR_RAE_ROUTE=fused` on HuPR and `reference` on the cascade takes RT-Pose to
-1.81 ms (-4.0%) instead of the -0.25 ms full `c1` manages, and keeps HuPR's -2.90 ms
(-12.2%).

The same A/B at the RESTRICTED rows of `c2`, in case the fused kernel earns its place once it
gathers the rows inside the compiled kernel instead of materialising a gathered profile:

| | fused | reference |
|---|---|---|
| `c2` RT-Pose 176 | 18.74 | **18.67** |
| `c2` HuPR 256 | **14.87** | 15.25 |

Same verdict at a tenth of the cells, so the split is a property of the array, not of the row
count. `_rae_route_default` therefore picks by dataset, `HUPR_RAE_ROUTE` still overrides, and
`result.json` records which route made the product. Verified end to end: RT-Pose 176 prints
`RAE route reference` and steps at **43.80 ms** (against 45.63 baseline and 45.37 under a
global fused default), HuPR 256 prints `fused` and steps at **20.84 ms** (against 23.73).

## 8. A lever that looked good and is not: the length-86 FFT

The cascade transforms **86 = 2 x 43** angles, 43 prime, which cuFFT must handle by
Bluestein rather than a radix decomposition. Padding the RAE view to 128 looked like the
one lever that buys speed without touching supervision. **Measured, it is a net loss.**

Two stacked FFTs over `(4, 63, 244, n_angle, 8)` complex64, forward only, on one 4090:

| n_angle | cells | 2D FFT ms | ms per M cells |
|---|---|---|---|
| **86** (the fit's own) | 42.3 M | **3.67** | 0.087 |
| 88 | 43.3 M | 3.02 | 0.070 |
| 96 | 47.2 M | 3.29 | 0.070 |
| **128** (next power of two) | 63.0 M | **4.38** | 0.070 |

The prime factor costs a **24% per-cell** penalty, not the several-fold one Bluestein
suggests -- and padding to 128 adds 49% more cells, so it is 4.38 ms against 3.67 ms:
**19% slower in absolute terms.** 88 would be marginally better than 86 (3.02 vs 3.67) but
changes the angular grid for 0.65 ms.

**The more useful consequence is what this says about where the time is.** Both 2D FFTs cost
3.67 ms forward, so with the adjoint roughly 7.3 ms of the ~27.9 ms the plane RAE term costs
at full rows -- about a quarter. The other ~20 ms is the gather, the `abs`, the floor add,
the `sqrt`, the difference, the square and the weighted reduction, plus their backwards:
**an elementwise chain over 42.3 M cells, i.e. memory traffic.** So the remaining lever, for
anyone who cannot restrict the rows, is fusing that chain -- which is the branch's item 3,
already known to be non-bitwise (gradients at 1e-7) and therefore needing a float64
tolerance rather than a bitwise gate. Fixing the FFT length is not the lever; it never was.

## 9. The verdict, and what to adopt

Raw products: `mmSD/artifacts/runtime_4term/20260923_0009` (every `timing.json` and
`metrics.json`, the four gate JSONs, the route-gate dumps, the probe logs and the table
tools). `main` advanced to d5467da while this ran -- the other session's round-2 ablation
result, docs only, no code overlap with this branch -- so `runtime/4term` merges cleanly but
has not been merged or pushed.


### 3. `c13b99d` under the four-term objective: **rejected as a default**

The item whose old result does not carry over. It is the largest speedup measured here by a
wide margin -- **-25.1 ms a step on RT-Pose, -59% of the fit, and it dissolves the memory
ceiling of section 1** (peak reserved 22.91 -> 16.22 GiB). It is also the only change here
that costs accuracy, and on RT-Pose the cost is large: **+11.9 cm MPJPE on 176**, +5.1 on
108, +1.0 on 191, with failed frames 1% -> 4%. Twenty-five milliseconds a step is not worth
eleven centimetres of pose error.

The old verdict does not survive because the reason it was nearly free is gone. Under the
six-term objective the RA plane term supervised every range row; under this one the plane
RAE term is the only plane supervision AND -- through `--rae-floor-gradient on` -- the noise
floor's only substantial gradient path. The floor is an off-person quantity, the item drops
the off-person rows, and range-Doppler is the view that reads them. Predicted, then measured.

**On HuPR it is neutral downstream** (AP within the seed spread on both sequences) and saves
-5.96 ms a step, -37% of the fit. Adopting it there alone is defensible on these numbers,
but it would make the supervised row set **dataset-dependent**, i.e. two different
objectives behind one name, which is a change to the objective and not a runtime decision.
That is the author's call, not this branch's; it is reported, not taken.

### What this branch proposes

| change | gate | RT-Pose | HuPR | adopt |
|---|---|---|---|---|
| `3ab773d` route gate | instrument, no arithmetic | -- | -- | yes |
| `d4daab3` + `59176a5` | **bitwise** | **-1.81 ms** | **-1.12 ms** | yes |
| `d5c2ef5` fused route | **bitwise** | +1.55 ms | **-1.78 ms** | **per dataset** |
| `c13b99d` person rows | results | -25.1 ms, **+11.9 cm MPJPE** | -5.96 ms, neutral | **no** |

The fused route is bitwise whichever way it goes, so making `HUPR_RAE_ROUTE` default to
`fused` on HuPR and `reference` on the cascade is free and strictly better than one global
default: RT-Pose keeps the full -1.81 ms (-4.0%) instead of -0.25 ms, HuPR keeps -2.90 ms
(-12.2%).

### 4. The corpus figure

235 HuPR + 101 RT-Pose sequences, at the per-sequence means measured here, on one RTX 4090.
"Velocity stage" is the 400 epochs; "stage onward" adds the render and the score, which
none of these changes touch.

| candidate | velocity stage | stage onward | saving vs b4 |
|---|---|---|---|
| `b4` | 51.7 GPU-h | 63.2 GPU-h | -- |
| `c1` | 50.4 GPU-h | 61.2 GPU-h | **1.3 / 2.0 GPU-h (2.6% / 3.1%)** |
| `c2` | 25.1 GPU-h | 35.0 GPU-h | **26.6 / 28.1 GPU-h (51.5% / 44.5%)** |

`c1` saves 2 GPU-hours a corpus, all of it on HuPR. `c2` saves 28, and halves the velocity
stage. The plan's H100 reference points (~14.8 min RT-Pose, ~5.7 min HuPR a sequence) put
the same corpus at roughly 33 GPU-h stage-onward on an H100, so the ratios carry even though
the hours do not.


Under the recommendation -- the two bitwise ports everywhere, the fused route on HuPR only,
`c13b99d` nowhere -- the corpus costs **44.5 GPU-h** in the velocity stage against the
baseline's 51.7: a saving of **7.3 GPU-h (14%)**, and 7.4 GPU-h (12%) stage-onward. Taking
`c13b99d` as well would make it 25.1 / 35.0 GPU-h, a 26.6 GPU-h (51%) saving -- which is the
prize the pose regression is being weighed against, and on RT-Pose it does not pay.

### What was not done, and what it would take

* **Repeats.** Section 4's null shows this pipeline's RT-Pose micro-Doppler spread wanders by
  0.148 between bitwise-identical trainers. Every results gate here is a single run, so only
  effects larger than that null are claimed -- which the range-Doppler move and the pose
  regression both are. Tightening the rest needs repeats the plan did not budget.
* **The elementwise chain.** Section 8 locates the remaining RT-Pose cost: roughly 20 of the
  27.9 ms the plane RAE term spends is memory traffic over 42.3 M cells, not FFT. Fusing that
  chain is the branch's item 3, already known non-bitwise (gradients at 1e-7), so it needs a
  stated float64 tolerance rather than a bitwise gate. It is the only lever left that does
  not change what is supervised.
* **Releasing the CUDA-graph pool** before `calibrate_person_level` (section 1.2), which would
  end the 637 MiB margin independently of `c13b99d`.
* **`5807da1`** (the exact elevation DFT) stays unported, as the plan directs.
