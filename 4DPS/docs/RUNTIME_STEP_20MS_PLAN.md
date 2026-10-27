# Getting the RT-Pose step from 43.2 ms to <= 20 ms (stepping 1308 s -> <= 600 s)

An action plan for four pieces of work, with the gate each needs and a go/no-go test before the
expensive one. Measured numbers are from this workstation (one RTX 4090, `--frame-batch 4`,
400 epochs) and reproduce from `mmSD/artifacts/runtime_4term/20260923_0009`.

**Recommendation, up front.** Do **A1** (fuse the elevation term's pointwise chain, ~-13.5 ms,
**bitwise**) and **C** (stop transforming the four empty elevation rows, ~-1.55 ms, **bitwise**)
first: together they are ~15 ms and cost nothing in guarantees. Then **B** (drop the azimuth
fftshift, -2.69 ms, float64 tolerance). Treat **A2** (fold the elevation DFT and the reduction
into the kernel, -2.2 ms, float64 tolerance) as **optional** -- it is the only item that spends
the project's bitwise guarantee on the loss path, and if A1 + B + C clears 600 s it should not be
paid for. Section 5 has the arithmetic; the short version is that A1 + C alone reach ~26 ms,
adding B reaches ~23 ms, and 600 s (20.0 ms) needs A2 or one of the two trims in section 5.

| item | what | saving | gate |
|---|---|---|---|
| **A1** | fuse the pointwise chain (abs, square, floor, sqrt, target denorm, subtract, square) | **-8.4 ms measured** (estimated -13 to -14; see section 9) | **bitwise, verified** |
| **C** | azimuth FFT on the 4 occupied elevation rows, not 8 | -1.55 ms | **bitwise** |
| **B** | drop the azimuth fftshift from both sides of the RAE loss | -2.69 ms | float64 tolerance |
| A2 | additionally fold the 4-tap elevation DFT and the reduction in | -2.0 to -2.5 ms | float64 tolerance |

**The two targets are the same statement.** RT-Pose sequences are 299 frames, so at
`--frame-batch 4` an epoch is 75 steps and 400 epochs is **30,000 steps**. 600 s / 30,000 =
**exactly 20.0 ms a step.** Nothing in this document touches the evaluation cadence, the epoch
count, the render or the score; every saving below is step time.

## 0. Which objective these numbers are under -- please confirm

Everything measured here is under the **four-term objective**, `chain.py`'s default since
2026-09-22 (`OBJECTIVE=4term`), which the trainer reaches with
`--ablate-loss-term loop0-ra,plane-ra --elevation-planes all --rae-floor-gradient on`. The four
terms, by the names the code uses (`_ABLATE_TERMS` knows six; two are dropped):

| # | term | code name | what it compares |
|---|---|---|---|
| 1 | $\mathcal{L}_{\mathrm{RAE},0}$ | `elevation_loop0_rae` | rendered vs measured **\|RAE\|** of the loop-0 (static) profile, over all 244 kept range rows |
| 2 | $\mathcal{L}_{\mathrm{RAE},\mathrm{pl}}$ | `elevation_plane_rae` | rendered vs measured **\|RAE\|** of **every** rendered Doppler plane (63 on the cascade at `--elevation-planes all`), all 244 kept rows, the learned floor added in power with a gradient into it |
| 3 | $w_{\mathrm{MD}}\mathcal{L}_{\mathrm{MD}}$ | `md` | the micro-Doppler marginal |
| 4 | $w_{\mathrm{win}}\mathcal{L}_{\mathrm{win}}$ | `person_window` | the person-window term |

**Dropped:** `loop0-ra` and `plane-ra` -- the two range-azimuth terms, each of which was its
RAE term computed on a subset of the array's channels.

"The elevation term is 23.4 of the 41.1 ms step" means **terms 1 and 2 together**, and term 2
is nearly all of it: term 1 is one plane, term 2 is 63, so term 1 is ~1.6% of their shared
cost. All three work items below are about term 2.

## 1. What the 23.4 ms is made of

`--profile-steps 20`, self CUDA time, bucketed; the elevation term isolated by differencing
against a build that computes term 2 on ~21 rows instead of 244.

| bucket | full rows | 21 rows | **term 1+2** |
|---|---|---|---|
| elementwise | 17.84 | 3.81 | **14.03** |
| RAE 2D FFT | 4.29 | 0.46 | 3.83 |
| fftshift roll (azimuth) | 2.76 | 0.07 | 2.69 |
| `abs` and its backward (`sgn`) | 1.60 | 0.19 | 1.41 |
| gather / scatter / index | 2.59 | 1.35 | 1.24 |
| everything else (render 8.0, Adam 2.2, ...) | 12.03 | 11.84 | ~0 |
| **step** | **41.11** | **17.72** | **23.39** |

Inside the FFT bucket, by kernel: the azimuth transform (86 = 2 x 43) is
`prime_fft_factor<43>` 1.43 ms + `regular_fft_factor<2>` 1.36 ms = **2.79 ms**; the elevation
transform is `vector_fft<8>` **1.49 ms**.

The term's tensors are `(4 units, 63 planes, 244 rows, 86 angle, 8 elevation)` =
**42.3 M cells a step**, and the loss is a uniform mean over them. That number is what every
lever below is really about.

## 2. Q1: the virtual array's sparsity -- no, we are not using it

Measured from `data.elevation_grid_plan` on the cascade. 12 TX x 16 RX = 192 channels placed on
an 86 x 7 (azimuth x elevation) grid, **134 slots occupied of 602 (22.3%)**; padded to the
transform's 86 x 8, **134 nonzero of 688 (19.5%)**. Per elevation row:

| elevation row | occupied azimuth slots | structure |
|---|---|---|
| 0 | 16 of 86 | three runs: 21-28, 60-63, 71-74 |
| **1** | **0** | **empty** |
| 2 | 16 of 86 | the same, shifted +1 |
| **3** | **0** | **empty** |
| **4** | **0** | **empty** |
| 5 | 16 of 86 | the same, shifted +2 |
| 6 | **86 of 86** | the full azimuth ULA, contiguous |
| 7 (pad) | 0 | the 7 -> 8 zero pad |

**Four of the eight elevation rows fed to the azimuth FFT are identically zero**, and three of
the remaining four carry 16 nonzero samples out of 86.

Today none of this is exploited. `forward_render._rae_spectrum_input` allocates
`(units, planes, rows, 86, 8)` complex, **zero-fills all 42.3 M cells (338 MB a step)**, writes
the 19.5% that are real, and hands the whole thing to two dense FFTs. So we pay a full
338 MB zero-fill, a full-width azimuth FFT over 8 columns of which 4 are zero, and a full
8-point elevation FFT whose input is half zeros -- every step, forward and backward.

HuPR is different and needs no such work: its grid is 8 x 2 with both elevation rows occupied.
The sparsity lever is cascade-only.

## 3. Q2: how to handle the azimuth and elevation transforms

The two axes want opposite treatments, because their sparsity is of different kinds.

**Azimuth (86): keep the FFT, run it on 4 columns instead of 8.** Row 6 is a dense contiguous
86-element ULA, so its transform is exactly what cuFFT is good at. The fix is not a different
algorithm, it is not transforming zeros: build the input compactly as
`(units, planes, rows, 86, 4)` over the occupied elevation rows {0, 2, 5, 6} and transform
that. The FFT of an all-zero column is exactly zero, so supplying zeros to the next stage
reproduces today's values **bitwise**. Halves the azimuth FFT (2.79 -> ~1.4 ms) and halves the
input buffer (338 -> 169 MB).

**Elevation (8): do not FFT it at all -- fold it into the loss kernel as a 4-tap contraction.**
With only rows {0, 2, 5, 6} nonzero, the 8-point DFT is
$X[k] = \sum_{n \in \{0,2,5,6\}} x[n]\,W_8^{nk}$ -- a fixed 4 x 8 complex twiddle matrix, 4
complex multiply-accumulates per output. An 8-point FFT costs about the same arithmetic, so the
win is **not** the MACs: it is that the 8-column complex intermediate never has to exist. The
fused kernel reads the azimuth-FFT output (86 x 4 per row), contracts to 8 elevation bins in
registers, takes the magnitude, adds the floor, subtracts the target, squares and accumulates.
That removes the elevation FFT kernel (1.49 ms), the `abs`/`sgn` pair (1.41 ms) and the whole
elementwise chain (14.03 ms) in one stroke, because they all stop being separate passes over
42.3 M cells.

This is why the elevation axis is the right place to cut and the azimuth axis is not: the
elevation DFT is short enough to live inside a fused kernel's registers, and it is the last
stage before the pointwise arithmetic, so folding it collapses the entire tail of the term.

**Optional, later: the structured sparsity inside rows 0, 2 and 5.** Their 16 samples sit in
three contiguous runs (8, 4, 4), so their 86-point DFT is a sum of three short DFTs times three
phase ramps -- roughly 260 complex MACs against the FFT's ~2200-equivalent. It is the largest
remaining arithmetic saving on the transform side and also the most intricate; it should not be
attempted until items A and B below have landed and been re-profiled.

## 4. The three work items

### A. Fuse the term's tail -- split in two, because most of it can be bitwise and the rest cannot

The first version of this plan proposed one fused kernel covering the elevation DFT, the
magnitude, the floor, the difference and the reduction, and called the whole thing float64-gated.
That gives away more than it needs to. Three of those five stages are **pointwise**, and
pointwise fusion reassociates nothing: every output cell is computed from the same inputs by the
same sequence of operations, so it can be bitwise. The two that cannot be bitwise are the two
that change the ORDER of a sum -- the elevation DFT and the reduction.

So the item splits, and the split matters because A1 carries almost all of the saving.

#### A1 (bitwise): fuse the pointwise chain only

One forward kernel and one backward kernel covering `abs` -> `square` -> `+ floor * factor` ->
`clamp_min` -> `sqrt` -> target denormalisation (`fp16 * rms`) -> subtract -> square. **Keep
cuFFT for both transforms. Keep PyTorch's `mean` for the reduction.** The kernel's output is the
per-cell squared difference, which the existing reduction then consumes exactly as today.

*Traffic:* forward reads the spectrum (338 MB complex64, or 169 MB after item C) and the fp16
target (81 MB) and writes the squared difference (169 MB) ≈ 588 MB; the reduction then reads
169 MB; the backward reads spectrum + target and writes the gradient into the spectrum ≈ 757 MB.
**~1.7 ms in total**, against the **15.44 ms** those stages cost now (14.03 elementwise + 1.41
`abs`/`sgn`).

*Expected:* **-13 to -14 ms** -- about 85% of the whole item's value. *Gate:* **bitwise**, by
`MM4DPS_RAE_CHECK=1` on values and all five gradients plus a 20-step deterministic dump.

Two implementation requirements, without which it will not be bitwise:

* **Compile it with `-fmad=false`, or write the arithmetic with `__fmul_rn` / `__fadd_rn`
  intrinsics.** nvcc contracts `a*b + c` into a single-rounding FMA by default. PyTorch's eager
  elementwise kernels each perform one operation, so the rounding boundary is the kernel
  boundary and every intermediate is rounded. A fused kernel with FMA contraction on rounds
  once where eager rounds twice, and that alone is enough to lose bitwise agreement -- it is
  almost certainly what put the previous branch's compiled item 3 at 1e-7. **This costs
  nothing here**: the kernel moves 400-750 MB and is bandwidth-bound, so the extra ALU
  instructions hide entirely. (`--use_fast_math` is already deliberately off in both
  extensions for a related reason, documented in `cuda_ext.py`.)
* **The backward must multiply its per-cell factors in the same order autograd does.** The
  chain's backward is a product of per-cell derivatives; the product is associative
  mathematically but not in fp32. Read the order off the current graph and match it. If it
  cannot be matched, the *forward* is still bitwise and only the backward falls to a float64
  tolerance -- which is worth knowing before the work starts, not after.

#### A2 (float64 tolerance): additionally fold the elevation DFT and the reduction in

Then the 4-tap elevation contraction of section 3 happens in registers, the 8-column complex
intermediate never exists, and the kernel emits the per-unit scalar directly.

*Expected:* a **further -2 to -2.5 ms** (the 1.49 ms elevation FFT, plus ~0.9 ms of the
materialised squared-difference round trip). *Gate:* float64 reference with a stated tolerance.

*Why it cannot be bitwise, precisely:* an 8-point radix-2 FFT and an explicit 4-term twiddle sum
compute the same value by different associations and different twiddle factors; and a fused
reduction uses its own block decomposition rather than PyTorch's tree. Both are reassociation,
not approximation.

*Two costs specific to A2, and the second is a genuine regression risk:*

* **Determinism.** If the fused reduction uses atomics it is nondeterministic, which breaks
  `--deterministic on` -- the instrument every bitwise gate in this project depends on. The
  reduction must be a fixed deterministic tree.
* **Accuracy can get WORSE, not just different.** Each unit reduces ~10.6 M cells. PyTorch's
  `mean` is a tree reduction; a naive sequential fp32 accumulator over 10.6 M terms loses
  accuracy badly. A2 must use pairwise/tree accumulation or fp64 accumulators, or it will be
  measurably less accurate than what it replaces -- and a float64 gate would catch that, which
  is exactly why the gate is against float64 and not against A1.

**Recommendation: do A1, measure, and treat A2 as optional.** A1 is bitwise and worth ~13.5 ms;
A2 is worth ~2.2 ms and spends the project's bitwise guarantee on the loss path. If the budget
in section 5 closes without A2, do not do A2.

### B. Drop the azimuth fftshift from both sides of the RAE loss

Both elevation terms are a **uniform mean over the flattened (range, angle, elevation) cells**.
`fftshift` on the angle axis is a cyclic permutation of those cells, and a permutation does not
change a uniform mean. So: store the RAE target un-shifted (once, at cache build) and stop
applying the roll to the prediction. The term's value is unchanged in exact arithmetic; only the
fp32 summation order moves.

*Expected:* **-2.69 ms.** *Gate:* float64 tolerance (reduction order). *Scope, and this is the
trap:* the change must touch **only** the two elevation terms and their target store. The RA
path's `profiles_to_ra_magnitude_fused` feeds a masked term (`ra_mask_kept`) whose mask is
indexed in shifted angle coordinates, and the `window=False` calls around line 4247 serve the
downstream's own convention. Leave both alone.

### C. Stop transforming the four empty elevation rows

Have `_rae_spectrum_plan` record the occupied elevation columns (it already computes
`el_slots`), build the compact `(..., 86, 4)` buffer, transform that, and scatter back to 8
columns of which 4 are exactly zero -- or, once A has landed, hand the 4 columns straight to the
fused kernel and skip the scatter entirely.

*Expected:* **-1.25 ms** on the azimuth FFT, plus **-0.3 to -0.5 ms** of zero-fill and buffer
traffic. *Gate:* **bitwise**, on its own -- zeros in, exact zeros out, and the same values reach
the elevation stage. It becomes float64-gated only when combined with A, which changes the
summation order anyway. Because it is bitwise standalone, **do C before A**: it is the cheapest
of the three and it shrinks the tensor A has to be benchmarked against.

## 5. Does this reach 20 ms?

| | gate | conservative | optimistic |
|---|---|---|---|
| baseline | | 41.11 | 41.11 |
| **A1: fused pointwise chain** | **bitwise** | **-13.0** | **-14.0** |
| B: no azimuth fftshift | float64 | -2.69 | -2.69 |
| C: 4 elevation columns | bitwise | -1.55 | -1.75 |
| **subtotal, nothing reassociated but B** | | **23.9** | **22.7** |
| A2: fold the elevation DFT + reduction | float64 | -2.0 | -2.5 |
| **step** | | **21.9** | **20.2** |
| **stepping (30,000 steps)** | | **657 s** | **606 s** |

The bitwise-only line -- A1 and C, leaving even the fftshift alone -- is **26.6 to 25.8 ms**,
i.e. 798 to 774 s. So a guarantee-preserving path exists and gets a third of the way; reaching
600 s needs B and probably A2, both of which are reassociation at the fp32 floor.

**The three items land at 20-22.5 ms, i.e. 606-672 s.** The target sits at the optimistic edge,
and the swing is almost entirely item A: it carries three quarters of the saving, and whether it
lands at 0.75 ms or 3 ms decides whether the target is met.

Two safe trims buy the remaining margin, and both need investigation rather than design:

* **The optimizer: 2.22 ms a step over 23 calls.** Fused Adam is already on, so this is
  param-group count, not kernel choice. Consolidating groups that share a learning rate should
  be bitwise or near it. Plausibly **-0.5 to -0.8 ms**.
* **`unbucketed` 1.47 ms and `memcpy / fill` 0.97 ms** have not been attributed to a call site.
  `MM4DPS_PROFILE_SHAPES=1` would name them.

With those, 19-21 ms and **570-630 s**. That is the honest range: the target is reachable
without touching what is supervised, but not with comfortable margin, and not if A
underdelivers.

**What is deliberately not proposed.** Restricting term 2's rows (`c13b99d`) is measured and
rejected -- 23.4 ms for +11.9 cm MPJPE on 176 (`docs/RUNTIME_4TERM_RESULTS.md` 6). Padding the
azimuth FFT to 128 is measured and a net loss (section 8 there). Reduced precision for the RAE
view would change results and is not needed if A lands. The renderer (8.0 ms) is untouched.

## 6. Order of work, with a go/no-go before the expensive item

1. **Item C** (half a day). Bitwise, self-contained, and it shrinks the tensor everything else
   is measured against. Confirm with `MM4DPS_RAE_CHECK=1` plus a 20-step deterministic dump;
   expect ~1.5 ms.
2. **Item B** (half a day). Un-shift the target at cache build, drop the roll, gate against
   float64 with a stated tolerance. Expect ~2.7 ms. After 1 and 2 the step should read
   **~37 ms**; if it does not, stop and re-profile before starting A.
3. **GO/NO-GO for A1 (one afternoon).** Write the *forward* pointwise kernel only, standalone,
   on the real shape, compiled `-fmad=false`, and check two things: that it is bitwise the
   current chain's forward cell by cell, and that it runs near its ~0.65 ms traffic floor. If it
   is bitwise but slow, the arithmetic in section 5 does not close. If it is fast but not
   bitwise, find the rounding difference before building the backward -- do not accept 1e-7 here,
   because A1's whole value is that it does not need a results gate.
4. **Item A1** (2-3 days, on a pass at step 3). Forward and backward, the reference route kept
   permanently, gated by `MM4DPS_RAE_CHECK=1` on values and all five gradients. Re-profile
   afterwards: at ~24 ms the renderer is a third of the step and the loss is no longer the
   bottleneck.
5. **Re-read the budget.** If A1 + B + C clears 600 s, stop -- A2 spends the project's bitwise
   guarantee on the loss path for ~2.2 ms and should not be paid for unless it is needed.
6. **Then the two trims** (optimizer groups, the unattributed 2.4 ms) to buy margin under 600 s.
7. **Only if the target still misses**: A2, then the structured sparsity inside rows 0/2/5
   (section 3). Both are arithmetic-only and change nothing about what is supervised.

## 7. One correctness note that applies to all of them

None of A1, A2, B or C changes which cells are supervised, so none may be gated on results.
A1 and C are gated bitwise; A2 and B change floating-point association and are gated against a
float64 reference with a stated tolerance. If any of them moves a fidelity metric, that is a bug
in the implementation, not a trade to be accepted.

Where a float64 tolerance is the gate (A2, B), the scale to judge it against is already measured:
under the shipped `--deterministic off` the sorted-Doppler route makes **two runs of identical
code** differ at 4e-7 in the gradients from step 0 and 3e-3 in the losses by step 19
(`docs/RUNTIME_4TERM_RESULTS.md` 4). A per-step deviation at 1e-7 is therefore below the
pipeline's own nondeterminism. That is the argument for accepting it -- but it is not an argument
for skipping the fidelity suite, because the trained checkpoint does differ and the products must
be shown to sit inside the empirical run-to-run null rather than assumed identical.

## 9. A1 as built and measured (2026-09-23)

**Done, bitwise, and it delivers less than this plan estimated.** `hupr_rae_plane_term_forward` /
`_backward` (`mm4DPS/cuda/kernels.cu`, commit 84c1db4) fuse the plane term's pointwise tail.

| | RT-Pose 176 | HuPR 256 |
|---|---|---|
| A1 off (`HUPR_RAE_TERM=eager`) | 43.76 ms | 20.86 ms |
| **A1 on** | **35.33 ms** | **18.12 ms** |
| **A1's saving** | **-8.43 ms (-19.3%)** | **-2.75 ms (-13.2%)** |
| cumulative from `b4` | 45.63 -> 35.33 = **-22.6%** | 23.73 -> 18.12 = **-23.6%** |

*The estimate in section 5 was -13 to -14 ms; the measurement is -8.4.* The over-estimate came
from attributing all 14.03 ms of the profile's "elementwise" bucket to this chain. Isolated on the
real shape the eager chain is 11.05 ms forward+backward and the kernel is 1.85 ms, so ~9 ms was
always the ceiling, and the step sees -8.4. The standalone bench predicted the outcome; the
bucket attribution did not. **Trust the isolated bench over a bucket share** for the items still
outstanding.

The kernel costs **1.88 ms a step** and removes **10.04 ms** (elementwise 17.84 -> 9.95,
`abs`/`sgn` 1.60 -> 0.12, memcpy/fill 0.97 -> 0.30).

*Gate, both datasets:* `MM4DPS_RAE_CHECK=1` bitwise in values and all five gradients, and the
20-step deterministic dump identical with the term off and on. Three details were load-bearing and
are worth carrying into A2 and any other fused kernel here:

1. **Round-to-nearest intrinsics, not FMA.** `torch.compile` on the identical chain deviates at
   1.3e-7..2.6e-7 on 20-40% of cells -- measured, and almost certainly what put the earlier
   branch's item 3 at 1e-7. Contraction costs nothing to give up in a bandwidth-bound kernel.
2. **`hypotf`, not `sqrtf(re*re + im*im)`.** PyTorch's complex `abs` is hypot; the naive form
   differs at 8.7e-8 on 11% of cells.
3. **`sgn(0) = 0` for a complex input.** An out-of-band plane's cell is exactly zero on real data,
   so the unguarded `re/mag` gave 0/0 = NaN. Random-input testing never produced an exact zero and
   passed; the first real fit's gradients came back NaN. **Gate fused kernels on real data.**

The per-dataset RAE route default is unaffected: re-measured with A1 on, the cascade still prefers
the reference spectrum by 1.73 ms and HuPR the fused one by 1.70 ms.

### Where the step now stands, and the honest revision

Re-profiled with A1 on: **32.92 ms/step** (was 41.11).

| bucket | before A1 | after A1 |
|---|---|---|
| other elementwise | 17.84 | 9.95 |
| render (splat + scene + BSDF) | 8.01 | 8.01 |
| RAE 2D FFT | 4.29 | 4.29 |
| fftshift roll | 2.76 | 2.72 |
| gather / scatter / index | 2.59 | 2.58 |
| optimizer | 2.22 | 2.23 |
| the fused kernel | -- | 1.88 |
| `abs` / `sgn` | 1.60 | 0.12 |
| **step** | **41.11** | **32.92** |

**The revised budget does not reach 20 ms on safe work.** At the 400-epoch median scale A1 puts
RT-Pose at roughly 33.5 ms, i.e. ~1005 s of stepping against 1308. The remaining safe levers are
B (-2.7), C (-1.55), A2 (-2.2) and the optimizer (-0.8): together ~7.25 ms, landing at **~26 ms
and ~790 s**. Reaching 600 s needs about 6 ms more than the safe set contains.

So the decision the next round faces is explicit: **~790 s is available without touching what is
supervised; 600 s is not.** Closing the last 190 s means the renderer (8.01 ms, a third of the
step now, and the only bucket nothing here has touched), or reduced precision, or the
row-subsampling estimator of `docs/RUNTIME_10MIN_PLAN.md` section 6. Those are results-gated
choices, and this document does not pre-empt them.

The remaining 9.95 ms of elementwise is now the largest single bucket after the renderer, and it
has NOT been attributed to call sites. On the evidence of A1's over-estimate, that attribution --
`MM4DPS_PROFILE_SHAPES=1`, plus isolated benches of the candidates -- should come before any more
kernel work is committed to.

### A1 confirmed at 400 epochs, and a blocker it exposed

| | `b4` | **A1** | |
|---|---|---|---|
| RT-Pose 176 step | 43.22 ms | **33.31 ms** | **-22.9%** |
| RT-Pose 176 stepping | 1308 s | **1008 s** | **-22.9%** |
| RT-Pose 176 wall | 1552 s | **1179 s** | -24.0% |
| RT-Pose 176 evaluation | 90 s | 79 s | -12% |
| HuPR 256 step | 23.83 ms | **18.26 ms** | **-23.4%** |
| HuPR 256 stepping | 246 s | **190 s** | -22.7% |
| HuPR 256 evaluation | 22.9 s | 22.8 s | **unchanged** |

The 8-epoch probe predicted both medians to under 1% (35.33 -> 33.31 after the concurrency
difference, 18.12 -> 18.26), so it stands as the estimator for the remaining items.

**The evaluation is render-bound on HuPR and only partly term-bound on the cascade** -- 22.9 ->
22.8 s against a 23% faster step on HuPR, 90 -> 79 s on RT-Pose. No loss-side lever will buy much
evaluation time, which matters because `--eval-every` is not available as a lever.

**The first attempt at this run FAILED, and it promoted section 1.2 from an observation to a
blocker.** All 400 epochs completed and then `calibrate_person_level` died for 512 MiB, with
2.91 GiB idle in the captured step's private pool -- the same failure that killed three baseline
runs, now reached sooner because A1's two extra store-sized tensors consumed the ~640 MiB margin.
`release_graph_pool` (0fee824) drops the graphs after the last epoch: **22.97 -> 16.55 GiB
reserved, 6.42 GiB handed back**, and the run completes. Memory only; the training peak is
unchanged at 19.09 GiB allocated. Anyone adding a kernel to this step should expect to hit that
margin and should take this fix first.

## 10. Step 1: the step attributed, and what it says about C (2026-09-23)

`MM4DPS_PROFILE_STACK=N` widens the profiler's name column and groups by Python frames. The frames
do not attach to CUDA kernels (they are linked through launches, not stacks), but the full kernel
signatures are specific enough to attribute by role. **This supersedes the bucket table of
section 9**, which lumped 9.95 ms into "other elementwise".

RT-Pose 176, A1 on, the cascade's default **reference** spectrum route, 32.89 ms/step:

| role | ms/step | calls/step |
|---|---|---|
| render: Doppler-plane splat | 5.55 | 4 |
| **reference RAE route: `movedim` / contiguous copies** | **5.38** | **52** |
| azimuth FFT (86 = 2 x 43) | 2.79 | 14 |
| fftshift roll | 2.72 | 12 |
| optimizer (fused Adam) | 2.21 | 15 |
| **A1's fused term kernel** | **1.88** | 2 |
| render: static scene | 1.82 | 2 |
| elevation FFT (8) | 1.49 | 4 |
| gathers (`index_select` etc.) | 1.43 | 44 |
| reference RAE route: grid scatter (`index_add`) | 1.06 | 9 |
| reference RAE route: complex add | 0.93 | 7 |
| reference RAE route: complex zero-fill | 0.85 | 12 |
| render: BSDF / scatter splat | 0.63 | 4 |
| compiled (inductor) | 0.97 | 17 |
| reductions, float zero-fill, `torch.complex`, cat, memcpy | 2.17 | 155 |

**The elevation term is 17.5 ms of the 32.9 ms step (53%), and only 1.88 ms of that is the loss
arithmetic.** 8.66 ms is data plumbing to get the profiles into FFT shape (copies, the grid
scatter, zero-fills, `torch.complex`), 4.28 ms is the two transforms and 2.72 ms is the fftshift.
The renderer is 8.00 ms. Nothing else exceeds 2.3 ms.

### Why the fused route loses, measured

The fused route exists to replace that plumbing with one compiled gather, yet it is 2.5 ms
*slower*. Profiling both routes says exactly where it goes:

| role | reference | fused | delta |
|---|---|---|---|
| compiled (`_rae_spectrum_input`) | 0.97 | **8.79** | **+7.82** |
| `movedim` / contiguous copies | 5.38 | 4.20 | -1.18 |
| fftshift roll | 2.72 | 1.43 | -1.29 |
| grid scatter | 1.06 | 0.34 | -0.72 |
| complex zero-fill | 0.85 | 0.10 | -0.75 |
| `torch.complex` | 0.36 | 0.18 | -0.18 |
| azimuth FFT | 2.79 | 3.19 | +0.40 |
| **step** | **32.89** | **35.35** | **+2.46** |

So the fused route removes about 5 ms of plumbing and spends 8.79 ms doing it. **And the reason is
the slot table's shape, which is where the array's sparsity finally bites.**
`_rae_spectrum_input_eager` gathers through a `(S, width)` table with `S = n_az * n_el_slots =
86 * 7 = 602` and `width = 2`:

| intermediate | shape | size |
|---|---|---|
| `kept` | (4, 63, 192, 244) | 90 MiB (x2 for re/im) |
| **`gathered`** | **(4, 63, 602, 2, 244)** | **565 MiB** |
| slot sums | (4, 63, 602, 244) | 282 MiB |
| output buffer | (4, 63, 244, 86, 8) | 323 MiB |
| **what the FFT actually needs** | **(4, 63, 244, 86, 4)** | **161 MiB** |

**The `gathered` intermediate is 565 MiB and only 22% of its slots hold a channel** (134 of 602),
and it is 3.5x the size of the buffer the transform consumes. At ~900 GB/s the whole kernel should
move ~400 MB and cost under a millisecond; it costs 8.79 ms, so it is running at roughly a tenth
of achievable bandwidth, and the blow-up is why.

### What this does to the plan

**Item C is no longer a 1.55 ms saving; it is the fix for an 8.79 ms kernel.** The four empty
elevation rows and the 78% empty slot table are the same fact seen twice: gathering straight into
the compact `(86, 4)` layout, without a 602-slot intermediate, removes the largest tensor in the
step. If `_rae_spectrum_input` lands anywhere near its traffic floor the fused route becomes
~28 ms and beats the reference route by ~4-5 ms rather than losing to it by 2.5.

**C is therefore promoted to the next item of work**, ahead of B. Two things follow from A1's
lesson and must be done in this order:

1. **Bench the rewritten gather in isolation first**, on the real shape, against the current
   compiled version. A1 was over-estimated 60% by reasoning from a bucket; the isolated bench was
   right. Until that bench exists the "~7 ms" above is an inference from a bandwidth argument, not
   a measurement.
2. **Then re-run the route A/B and re-decide `_rae_route_default`.** The per-dataset default was
   measured twice under the old gather; if C flips the cascade to the fused route, B's saving
   changes too (the fused route already rolls only one axis: 1.43 ms against 2.72).

The rest of the ordering is unchanged, with sizes now measured rather than assumed: B up to 2.72 ms
(reference route) or 1.43 ms (fused), the optimizer 2.21 ms, A2 ~2.2 ms, the renderer 8.00 ms.

## 11. Item C built and confirmed (2026-09-23)

`_RaeSpectrumInput` (`mm4DPS/cuda/kernels.cu`, commit 5204f9b) builds the RAE transform's input
straight from the channels, without the slot table's (units, planes, 602, 2, rows) intermediate.

| | `b4` | after A1 | **after C** | total |
|---|---|---|---|---|
| RT-Pose 176 step | 43.22 ms | 33.31 | **29.44 ms** | **-31.9%** |
| RT-Pose 176 stepping | 1308 s | 1008 | **892 s** | -31.8% |
| RT-Pose 176 wall | 1552 s | 1179 | **1058 s** | -31.8% |
| RT-Pose 176 peak reserved | 22.91 GiB | 23.52 | **22.96 GiB** | -- |
| HuPR 256 step | 23.83 ms | 18.26 | **18.56 ms** | -22.1% |
| HuPR 256 stepping | 246 s | 190 | **194 s** | -21.1% |

**The 8-epoch probes over-read the 400-epoch median by about 6%, consistently** -- A1 probed 35.33
and landed 33.31, C probed 31.19 and landed 29.44. Worth knowing when sizing what is left: subtract
~6% from a probe before believing it.

Three results beyond the headline:

1. **The cascade's route preference flipped**, as section 10 predicted: the fused route measures
   31.19 ms against the reference chain's 35.26 once the gather is fixed, where it lost by 1.73 ms
   before. `_rae_route_default` returns "fused" for both arrays now, and the old dataset-dependence
   is gone.
2. **C gave back the memory A1 spent.** Peak reserved 23.52 -> 22.96 GiB, essentially the
   baseline's 22.91, because the 565 MiB intermediate is no longer allocated. The margin that
   `release_graph_pool` protects is back to where it was.
3. **HuPR gained nothing from C** (18.26 -> 18.56 ms, inside its own run-to-run range). Expected:
   its grid is 8 x 2, so there was no blow-up to remove. C is a cascade fix.

Fidelity sits inside the run-to-run null once that null is measured honestly -- and C's own run is
what exposed the null's flaw, that a max over n runs grows with n
(docs/RUNTIME_4TERM_RESULTS.md 4).

### What is left

At 29.44 ms, the remaining safe items are C2 (the four empty elevation columns in the azimuth FFT),
B (the fftshift, now 1.43 ms on the fused route rather than 2.72 on the reference one), the
optimizer's 2.21 ms and A2's ~2.2 ms. **Every one of those sizes predates the current profile and
should be re-measured before it is believed** -- Step 1 moved C from "1.55 ms" to "5.80 ms" and
section 9 moved A1 from "-13 to -14" to "-8.4". A fresh profile at 29.44 ms is the first thing the
next round should do.

## 12. P1, P2 and P3 (2026-09-23)

### P3 first, because it is a correction: the optimizer was never 2.21 ms

`Optimizer.step#Adam.step` (a profiler scope), `aten::_fused_adam_` (the op) and
`multi_tensor_apply_kernel` (the kernel) all report the SAME GPU time, and section 10's bucket
summed all three. Adam's real cost is **0.73 ms a step**, not 2.21. It is already fused, its seven
calls are seven parameter groups with genuinely different learning rates, and merging the two that
share `--gain-lr` would save launch overhead on a bandwidth-bound update. **P3 is closed: there is
nothing here worth taking.**

The attribution tool now counts CUDA kernels only. Every bucket in section 10 was inflated wherever
a scope or an aten op overlapped its kernel; the tables below are the corrected form.

### P1: the copies were my own wiring

Section 10 left 4.48 ms of generic copies unexplained. Shapes do not attach to CUDA kernels, so the
hypothesis was tested directly: `rae_terms_of` passed `planes.real` and `planes.imag` to the fused
route, and those are **stride-2 views** of a 96 MiB complex tensor, so the C1 wiring copied both.
Measured at **1.26 ms a call** once the gradient's recombination is counted. The kernel now reads the
complex profile as `float2` -- one 8-byte load a channel rather than two 4-byte loads from separate
arrays -- and returns a complex gradient. **-1.36 ms, bitwise.**

### P2 (item B): the azimuth fftshift, dropped from both sides

Sized first: the roll moves 323 MiB in and out forward and again in the backward, **1.48 ms**
standalone. Delivered **-1.59 ms** in situ. Scoped by `rae_azimuth_rolled` to arrays where the shift
is a real roll -- the cascade (86 = 2 x 43); HuPR folds it into input signs for free and is
untouched, its dumps bitwise either way.

**Not bitwise, and gated as such:** step 0's per-unit terms agree to **1.93e-07** and its gradient
norms to **2.39e-07**, which is fp32 reassociation and below this pipeline's own 4e-7 gradient
nondeterminism under the shipped `--deterministic off`. The 20-step dump diverges to 2e-2, which is
the chaotic amplification already documented for a 4e-7 start -- step 0 is the comparison that
carries information, and a 20-step trajectory is not a tolerance test.

### Where the step stands, corrected

| | after C2 | **after P1 + P2** |
|---|---|---|
| step (profiler) | 28.99 | **25.36 ms** |

| role | ms/step | calls/step |
|---|---|---|
| render: Doppler-plane splat | 5.55 | 4 |
| **unattributed** | **4.00** | -- |
| copies (`movedim` / `contiguous`), sites unknown | 2.14 | 48 |
| A1's fused term kernel | 1.87 | 2 |
| render: static scene | 1.76 | 2 |
| azimuth FFT | 1.70 | 14 |
| elevation FFT | 1.47 | 6 |
| compiled (inductor) | 1.26 | 42 |
| gathers | 1.04 | 42 |
| optimizer | 0.73 | 7 |
| render: BSDF / scatter splat | 0.64 | 4 |
| reductions, zero-fills, cat, memcpy, complex ops | 2.28 | ~150 |

**The renderer is 7.95 ms -- 31% of the step and the largest item.** After it come 4.00 ms
unattributed and 2.14 ms of copies whose call sites are still unknown: 6.1 ms, a quarter of the step,
that nobody has looked at. On this round's evidence -- C went from "1.55 ms" to 5.80, A1 from
"-13 to -14" to -8.4, the optimizer from 2.21 to 0.73 -- **that is where the next round should look,
before any remaining item is costed.**

## 13. The step fully attributed at 25.36 ms, and a costed path to 20 ms (2026-09-23)

### Why attribution here is hard, which is worth recording before the numbers

**The captured step is a CUDA-graph replay, so its kernels have no CPU parent and no Python stack.**
They are launched by `cudaGraphLaunch`, not by an aten op, so `with_stack` and the Chrome trace's
flow events cannot place them: parsing the trace attributes 24.53 of 25.36 ms to one annotation, and
only `aten::_fused_adam_` -- which runs outside the graph -- resolves to a call site. Three separate
attempts in this round failed on that before the cause was clear. Attribution must therefore come
from kernel signatures plus **bisection against the env switches**, which is how the table below was
built. Anyone profiling this step should start there and not repeat the trace parsing.

### The step, by role

| block | ms/step | calls | what it is |
|---|---|---|---|
| render: Doppler-plane splat | 5.55 | 4 | the physics |
| **C2's reassembly: `index_copy_` + its backward `index_select`** | **2.39** | 6 | 0.66 + 1.73 |
| copies (`movedim` / `contiguous`) | 2.14 | 48 | **sites unknown** |
| A1's fused term kernel | 1.87 | 2 | at its traffic floor |
| render: static scene | 1.76 | 2 | the physics |
| azimuth FFT (4 of 8 columns) | 1.70 | 14 | |
| elevation FFT (8-point) | 1.47 | 6 | |
| compiled (inductor) | 1.26 | 42 | `_plane_term_sums`, the MD term |
| gathers of the stores | 1.04 | 42 | |
| optimizer (fused Adam) | 0.73 | 7 | nothing to win (section 12) |
| render: BSDF / scatter splat | 0.64 | 4 | the physics |
| reductions, zero-fills, cat, memcpy, complex ops | 2.28 | ~150 | |
| **total** | **25.0** (bucketed) | | 25.36 measured |

`indexSelectSmallIndex<complex>` at 1.73 ms was the largest single unknown, and it is **C2's
backward**: the adjoint of `full.index_copy_(-1, occ_index, angle)` is
`grad.index_select(-1, occ_index)`. Confirmed by bisection -- with `HUPR_RAE_C2=off` both index
kernels vanish and the azimuth FFT doubles (1.70 -> 3.16). C2 is still a net win, re-measured after
P1 and P2: **27.41 ms against 28.29** (-0.88).

### The path to 20 ms, costed

**Step 1. A2 -- fold the elevation DFT into the loss kernel. -3.9 ms. NOT bitwise.**
This is now much larger than the ~2.2 ms guessed in section 5, and C2 is what revealed why. The
elevation stage needs its 8-column buffer reassembled, so the compact azimuth output is scattered
back (`index_copy_`, 0.66), its gradient is gathered back (`index_select`, 1.73), and then an
8-point FFT runs (1.47). **Folding the 4-tap elevation contraction into A1's kernel deletes all
three at once**: the kernel would consume the compact `(86, 4)` azimuth output and emit the per-cell
squared error, with no 8-column buffer existing at any point. 25.36 -> **~21.5 ms**.

*Gate:* float64 reference with a stated tolerance, as P2 was. Two hard requirements, both from
section 5: the reduction must be a **deterministic tree** (atomics would break `--deterministic on`,
which every bitwise gate here depends on), and it must accumulate **pairwise or in fp64** (a naive
fp32 sum over 10.6 M cells a unit would be measurably LESS accurate than the tree it replaces).

**Step 2. Attribute the 2.14 ms of copies. Probably bitwise.** 48 calls a step, sites unknown.
Method is bisection, not tracing: profile with `HUPR_FUSED_LOSS`, `HUPR_COMPILED_PLANE_TERM`, the
person-window and static terms switched, and diff. **A specific hypothesis worth testing first:**
the RA path's `profiles_to_ra_magnitude_fused`, which the static and person-window terms use, still
takes `.real` / `.imag` -- the same stride-2 view bug P1 fixed for the RAE path, for the same
reason. If so the fix is the same and it is bitwise. Half a day.

**Step 3. Only if still short: the renderer, 7.95 ms and 31% of the step.** Its knobs (point count,
Doppler taps) change what is rendered, so it is results-gated, not float64-gated.

| after | step | bitwise? |
|---|---|---|
| today | 25.36 ms | -- |
| + A2 | ~21.5 ms | **no** (float64 tolerance) |
| + the copies, if they are P1's bug again | **~20 ms** | yes |

**So 20 ms is reachable, and it costs exactly one non-bitwise step.** Everything else on the path is
bitwise or better. If A2's tolerance is unacceptable, the remaining route to 20 ms is the renderer,
which is results-gated -- a strictly worse trade.

## 14. P1 + P2 confirmed, and Step 2's bisection (2026-09-23)

| | `b4` | **after P1 + P2** | total |
|---|---|---|---|
| RT-Pose 176 step | 43.22 ms | **26.05 ms** | **-39.7%** |
| RT-Pose 176 stepping | 1308 s | **790 s** | **-39.6%** |
| RT-Pose 176 wall | 1552 s | **952 s** | -38.7% |
| RT-Pose 176 peak reserved | 22.91 GiB | 22.79 GiB | -- |
| HuPR 256 step | 23.83 ms | **17.92 ms** | -24.8% |
| HuPR 256 stepping | 246 s | **187 s** | -24.0% |

### Step 2: where the 2.14 ms of copies live, by bisection

| configuration | copies | calls |
|---|---|---|
| baseline | 2.14 ms | 48 |
| **`--elevation-term off`** | **0.91 ms** | 40 |
| `--person-window-loss 0` | 2.19 ms | 39 |
| `--ablate-loss-term ...,md` | 2.21 ms | 48 |

**1.23 ms of the copies are the elevation term's; the other 0.91 ms is the render / MD path.** The
hypothesis in section 13 -- that the RA path's `.real` / `.imag` views were the cause, as they were
for the RAE path in P1 -- is **wrong**: under the four-term objective `plane-ra` is ablated, so that
path never runs in the captured step.

The actual source is the reduction boundary. `_RaePlaneTermSquaredError` returns the per-cell squared
error and `.flatten(1).mean(dim=-1)` reduces it, so `mean`'s backward hands back an **expanded,
stride-0 view** that `grad_out.contiguous()` materialises -- 161 MiB written and read for a tensor
whose value is one scalar per unit.

**This should not be fixed on its own, because A2 deletes it.** Folding the reduction into the kernel
removes the `squared` tensor, the expanded gradient, C2's `index_copy_` / `index_select` reassembly
(2.39 ms) and the separate elevation FFT (1.47 ms) in one change. So Step 2 folds into Step 1, and
**A2 is worth roughly 4.5-5 ms rather than the 3.9 estimated in section 13** -- which would put the
step at about 21 ms, with the remaining ~1 ms to 20 available from the 0.91 ms of render-path copies
or from the renderer itself.

A2 is not started. It remains the one non-bitwise step on the path, it needs a deterministic tree
reduction and pairwise or fp64 accumulation (section 5), and its gate is a float64 reference with a
stated tolerance.
