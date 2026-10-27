# Getting an RT-Pose sequence to 10 minutes end to end

A plan, not a result. Every number in sections 1-3 is measured on this workstation (one RTX
4090, `--frame-batch 4`, 400 epochs, the four-term objective) and is reproducible from
`mmSD/artifacts/runtime_4term/20260923_0009`; everything in sections 4-6 is an estimate with
its basis named. Companion to `docs/RUNTIME_4TERM_RESULTS.md`, which is where the baseline,
the ports and the `c13b99d` verdict come from.

**The short answer.** The target needs **~12 ms a step** if nothing else changes, against
43.2 ms today. Safe work -- changes that leave what is supervised alone -- gets to about
**22-24 ms**, which is 14-15 minutes, not 10. Ten minutes needs one change to the objective,
plus two cheap non-stepping cuts. The one change worth trying is **not** `c13b99d`, which is
already measured and rejected; it is a subsampling of the same rows `c13b99d` throws away,
which is what keeps the noise floor's gradient path alive.

## 1. Where the 26 minutes actually goes

RT-Pose 176, `b4`, 1552 s wall, from the chain's own timestamps and the trainer's per-epoch
`steps_s` / `eval_s`:

| component | seconds | share |
|---|---|---|
| **stepping** (30,000 steps at 43.2 ms) | **1308** | **84.3%** |
| evaluation (40 evals, 2.25 s each, `--eval-every 10`) | 90 | 5.8% |
| trainer setup + teardown | 102 | 6.6% |
| render + spectrograms | 47 | 3.0% |
| score, cleanup, collect, evaluate | 8 | 0.5% |

Two things worth knowing before optimising anything. **The render is not the problem** -- 47 s
of 1552, and no proposal here touches it. And **the non-stepping cost is 247 s**, which is 41%
of a 600 s budget all by itself: even at zero milliseconds a step, the current pipeline cannot
do 10 minutes without touching it.

75 steps an epoch, because RT-Pose sequences are 299 frames at `--frame-batch 4`. 400 epochs
is the protocol.

## 2. Where the 43 ms a step goes

`--profile-steps 20`, self CUDA time over 20 captured steps, bucketed by kernel role
(`mmSD/artifacts/.../prof/`). The profiler's 41.11 ms/step agrees with the 400-epoch median of
43.22 ms, so the table can be trusted as a decomposition.

| bucket | ms/step | share |
|---|---|---|
| **elementwise (not the render)** | **17.84** | **42.1%** |
| render: Doppler-plane splat (fwd+bwd) | 5.56 | 13.1% |
| RAE/RA 2D FFT | 4.29 | 10.1% |
| fftshift roll | 2.76 | 6.5% |
| gather / scatter / index | 2.59 | 6.1% |
| optimizer (fused Adam) | 2.22 | 5.3% |
| render: static scene / gain | 1.82 | 4.3% |
| `abs` and its backward (`sgn`) | 1.60 | 3.8% |
| memcpy / fill | 0.97 | 2.3% |
| render: BSDF / scatter splat | 0.63 | 1.5% |
| reductions | 0.59 | 1.4% |
| unbucketed | 1.47 | 3.5% |
| **total** | **41.11** | |

The renderer -- the part that does the physics -- is **8.01 ms, under a fifth of the step.**
Everything above it in the table is loss arithmetic.

## 3. What the elevation term costs, and what it is made of

Profiling `c2` (the `c13b99d` port, which computes the plane RAE term on ~21 rows instead of
244) isolates the term by difference:

| bucket | `b4` full rows | `c2` person rows | the term at full rows |
|---|---|---|---|
| elementwise | 17.84 | 3.81 | **14.03** |
| RAE/RA 2D FFT | 4.29 | 0.46 | 3.83 |
| fftshift roll | 2.76 | 0.07 | 2.69 |
| `abs` / `sgn` | 1.60 | 0.19 | 1.41 |
| gather / scatter / index | 2.59 | 1.35 | 1.24 |
| everything else | 12.03 | 11.84 | ~0 |
| **step** | **41.11** | **17.72** | **23.39** |

**The elevation term is 23.4 ms of the 41.1 ms step, and 60% of that is elementwise
arithmetic, not FFT.** The 2D FFT is 3.8 ms and the roll 2.7 ms. This is the single most
important fact in this document, because it says the lever is memory traffic over the term's
42.3 M cells a step -- not the transform, and not the renderer.

`docs/RUNTIME_4TERM_RESULTS.md` section 8 already ruled out the transform: the cascade's
length-86 azimuth FFT (86 = 2 x 43) carries only a 24% per-cell penalty, and padding to 128
is *slower* in absolute terms because it adds 49% more cells.

## 4. The budget, and what has to be true

600 s, minus the non-stepping cost, over 30,000 steps:

| non-stepping cost | stepping budget | required ms/step |
|---|---|---|
| 247 s (today) | 353 s | **11.8** |
| 190 s (`--eval-every 20`) | 410 s | 13.7 |
| 160 s (also a trimmed teardown) | 440 s | **14.7** |

So the two cheap non-stepping cuts are worth ~3 ms a step of slack, which is a large fraction
of what the hard work would have to find. Take them first.

## 5. The levers, with the gate each one needs

Ordered by measured size. "Safe" means it does not change what is supervised, so it is gated
bitwise or against a float64 reference -- never against results.

| # | lever | est. saving | basis | gate | safe |
|---|---|---|---|---|---|
| 1 | **Fuse the elevation term's elementwise chain** | **-11 to -14 ms** | 14.03 ms measured; a fused kernel's traffic floor is ~1.3 ms (423 MB fwd + 761 MB bwd at ~900 GB/s) | float64 tolerance -- known non-bitwise at 1e-7 (the branch's item 3) | yes |
| 2 | **Drop the azimuth fftshift from both sides** | **-2.7 ms** | 2.69 ms measured | float64 -- exact in real arithmetic, the fp32 reduction order changes | yes |
| 3 | `--eval-every 20` | -45 s wall (~1.5 ms equiv.) | 90 s measured over 40 evals | selection granularity is a protocol choice, not arithmetic | yes |
| 4 | Trim the teardown (30,000-record `history.json`) | -30 to -50 s wall | 102 s setup+teardown measured | none -- it is bookkeeping | yes |
| 5 | Fused-Adam / param-group count | -1 ms | 2.22 ms over 23 calls a step | bitwise if only grouping changes | yes |
| 6 | **Subsample the off-person rows** | **-6 to -9 ms** | scales with cells; see section 6 | **results** (fidelity + both downstream gates) | **no** |
| 7 | bf16 / complex32 for the RAE view | -3 to -5 ms | halves the term's traffic | results | no |
| 8 | `c13b99d` person rows only | -23.4 ms | measured | **already gated and REJECTED**: +11.9 cm MPJPE on 176 | no |

**Lever 2 is worth explaining**, because it looks too good. The elevation term is
`mean((pred - target)^2)` over the flattened (range, angle, elevation) cells, with uniform
weights. `fftshift` on the angle axis is a cyclic permutation of those cells. A permutation
does not change a uniform mean -- so if the stored target is un-shifted once at cache build
and the prediction's roll is simply not applied, the term's value is unchanged in exact
arithmetic. Only the fp32 summation order moves, which is why it is float64-gated rather than
bitwise. 2.7 ms for a reduction-order change is the best ratio on this list.

Safe levers 1-5 together: **41.1 - 14 - 2.7 - 1 = ~23.4 ms a step**, and ~160 s non-stepping.
That is 30,000 x 23.4 ms + 160 s = **862 s, about 14.5 minutes.** Real, worth having, and
short of the target.

## 6. The one objective change worth trying, and why it is not `c13b99d`

`c13b99d` is rejected on measurement, not on principle: it buys 23.4 ms and costs +11.9 cm
MPJPE on RT-Pose 176. The reason it fails is specific and is the key to a better version.

It drops the **off-person range rows** across every Doppler plane. Under the four-term
objective the plane RAE term is the only plane supervision AND, through
`--rae-floor-gradient on`, the noise floor's only substantial gradient path
(`docs/OBJECTIVE_ABLATION.md` step 0, finding 1). The floor is an off-person quantity. So
`c13b99d` removes almost every cell that constrains the floor, and the measured consequence is
a range-Doppler correlation drop on all seven radar-sequence pairs.

**The proposal: keep the person's rows at full weight, and keep a rotating stride-sampled
subset of the off-person rows, reweighted so the term is an unbiased estimator of the
full-row term.** Concretely: the person's ~21 rows every step, plus 1 in 8 of the remaining
~223 rows with the stride offset advancing each step, weighted x8. Over any 8 consecutive
steps every off-person row is visited once.

Why this should behave where `c13b99d` did not:

* The floor keeps a gradient path **every step**, from ~28 off-person rows rather than none.
* The estimator is unbiased for the same loss, so the term's scale and its weight against
  micro-Doppler and the person window are unchanged -- which is the failure mode `rt1b` hit
  upstream and `c13b99d` avoided by keeping the reference divisor.
* Stride rotation rather than random draw keeps the added gradient variance low and the run
  reproducible.

Cost: ~49 of 244 rows, so the term's cells fall ~5x. After levers 1 and 2 the term is ~6 ms;
at 5x fewer cells it is ~1.5 ms. **Step ~15.4 ms**, stepping 462 s, plus ~160 s non-stepping
= **622 s, about 10.4 minutes.** Pushing to 1 in 12 would clear 600 s.

It changes what is supervised, so per the plan's own rule it is a result to report and a
decision for the author -- and it must clear the fidelity suite plus `gate_node.py` and
`gate_hupr_node.py`, read against the empirical run-to-run null of
`docs/RUNTIME_4TERM_RESULTS.md` section 4, not the quoted spreads.

## 7. Recommended next steps, in order

1. **Levers 3 and 4 first** (half a day, no gate beyond reproducing a fit). They are free,
   they are worth ~3 ms of step-time slack, and they make every later measurement shorter.
2. **Lever 2, the fftshift** (half a day). Smallest change on the list with the best ratio.
   Un-shift the stored target at cache build, drop the roll, gate against a float64 reference
   with a stated tolerance. Expect -2.7 ms.
3. **Lever 1, the fused elementwise chain** (the real work, 2-3 days). One kernel for the
   term's forward from the complex spectrum, the fp16 target and the floor; one for its
   backward. Keep the existing chain as the reference route -- `HUPR_RAE_ROUTE` already has
   the shape for this -- and gate cell by cell against it in float64. Expect -11 to -14 ms.
   Re-profile afterwards: at ~24 ms a step the renderer becomes a third of the step and the
   next bottleneck is elsewhere.
4. **Then decide whether 10 minutes is worth an objective change.** If yes, implement lever 6
   behind a flag (`--elevation-plane-row-stride N`, default off = today's behaviour), and gate
   it on the five sequences with both downstream gates. If the range-Doppler correlation holds
   where `c13b99d`'s fell, it is the difference between 14.5 and 10.4 minutes.
5. **Do not** re-open `c13b99d`, the length-86 FFT padding, or `5807da1`. The first is
   measured and rejected, the second is measured and a net loss, the third is out of scope by
   the author's decision.

## 8. What would make these estimates firmer

* The fused kernel's achievable fraction of its traffic floor is the biggest unknown in
  lever 1. A one-afternoon microbenchmark -- the term's forward on 42.3 M cells, hand-written
  against the current chain -- would size it before anyone commits to the backward.
* Section 4's budget assumes the non-stepping cost is compressible to 160 s. The 102 s of
  setup+teardown has not been decomposed; the target-store build and the 30,000-record
  history are the suspects, and `MM4DPS_PROFILE_SHAPES=1` plus a few timers would settle it.
* Every fidelity claim in the companion document is a single run, and that document's section 4
  shows this pipeline's RT-Pose micro-Doppler spread wanders by 0.148 between bitwise-identical
  trainers. Lever 6 will need repeats to be gated honestly.
