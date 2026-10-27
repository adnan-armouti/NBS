# Runtime: the next two steps, and the item the parity audit added

Written against `6486a02` (`runtime/2x2`), sized from the H100 profile in
`RUNTIME_2X2_RESULTS.md` and the audit in `RENDERER_DATASET_PARITY_AUDIT.md`.

## Where the step actually goes

Measured, 20 captured steps, all optimisations on, route `fused`:

| bucket | RT-Pose 176 | HuPR 256 |
|---|---|---|
| renderer | 9.12 ms (44.7%) | 10.49 ms (55.5%) |
| everything else | 11.26 | 8.42 |
| **step** | **20.38** | **18.91** |

Stepping is 594–645 s a sequence against a target of <= 600 s, so both steps below are
optimisation past the point of need. That is the frame to read them in: the reason to do
step 1 is that it is cheap and bitwise, not that the target demands it.

Every item carried over from `RUNTIME_STEP_20MS_PLAN.md` §13/§14 — A2, the step-2
copies, the structured azimuth sparsity, the `den0`/`denp` precompute — was re-measured
on this hardware and dropped. Their targets now total 1.5–1.7 ms, and A2 was the only
one that would have needed a float64 gate rather than a bitwise one. The reasoning is in
`RUNTIME_2X2_RESULTS.md`.

---

## Step 1 — hoist the PSF factor out of `static_scene_frames_forward_kernel`

**Target.** 4.60 ms/step on HuPR 256 (24% of the step, 6 launches) and 0.97 ms/step on
RT-Pose 176. Its own backward is 0.17 ms, so the forward costs about 27x the backward
per launch — that asymmetry is what makes it worth a look.

**Why there is something to hoist.** `_StaticSceneFrames.backward` returns a gradient
for `carrier` and `None` for `gains_t`, `table4` and `geometry`, and the class docstring
states that the gains "are the frozen areas times the frame's visibility and carry no
gradient here" — `assemble_static_profile_frames_cuda` raises if they do. So inside the
kernel only `carrier` varies between steps. The tap offset and the PSF lookup,

```cuda
int dn = bin - (e.z & 0xffff);
if (dn > half) dn -= K;
if (dn < -half) dn += K;
psf_lerp(table4, dn + half, __int_as_float(e.w), n_grid, &pr, &pi);
```

are functions of `runs`, `entry` and `table4` alone — frozen geometry — and are
recomputed on every one of the 30,000 steps.

**The change.** Precompute `(pr, pi)` per (cell, entry) once, at the point where
`_scene_frames_geometry` is built, and have the kernel read one `float2` in place of the
`entry[k]` load, the `dn` wrap and the `psf_lerp`.

**Gate: bitwise, by construction.** The precomputed values are the *same* values the
kernel computes today, from the same `psf_lerp` on the same inputs, and the subsequent
multiply-add chain and its lane butterfly are untouched — so the accumulation order is
unchanged. Verify it as every other item in this round was verified: `--deterministic on`
with `MM4DPS_PRINT_STEPS=20`, and the 20-step dumps must be byte-identical. Only step 0
is informative for a reassociating change, but this one should hold at every step, so
all 20 lines matching is the bar.

**Do not skip the real-data check.** A1's kernel passed on `randn` and produced NaNs on
real data, because `randn` never produces an exact zero and `re/mag` became 0/0. Any new
kernel here is gated on a real sequence, not on synthetic input.

**Why it might buy nothing.** Two reasons, and the bench exists to settle them:

1. `table4` is small and almost certainly L2-resident, so the lerp may already be
   near-free, while the precomputed table is a DRAM read of 8 bytes per (cell, entry).
   This trade can go either way and is the whole question.
2. The inner loop reads `gains_s + k * frames` — every cell whose runs include path `k`
   re-reads that whole gains row. If that traffic is what binds, removing the lerp moves
   nothing. Measure it before building anything: compare the kernel's achieved bandwidth
   against the sum of its distinct inputs.

**First measurement, before writing any CUDA.** Size the precomputed table and check it
is affordable: `geometry["entry"].numel()` against the number of (cell, entry) pairs,
which is `sum(run lengths)` over `geometry["runs"]`. Two floats per pair. If that does
not fit comfortably beside a ~30 GiB working set, stop here.

**Cost if it lands.** Bounded above by the lerp's share of 4.60 ms on HuPR and 0.97 on
RT-Pose. A realistic expectation is a fraction of that, which on the corpus is worth
roughly 0.5–1.5 GPU-hours of the 30.6 — state the measured number, not this range.

---

## Step 2 — `doppler_plane_sorted`, and only if step 1 lands

**Target.** `doppler_plane_sorted_forward` 3.14 ms + `_backward` 2.88 ms = **6.02 ms on
RT-Pose 176**, 30% of the step; 4.12 ms on HuPR 256. The largest single item left.

**Why it is second, not first.** There is no equivalent free lunch here. The kernel's
inputs are the carrier and the path rates, both of which carry gradient, so there is no
frozen quantity to hoist the way step 1 has. Anything that helps is either a reassociation
of the accumulation — which breaks the fixed-order sum the kernel exists to guarantee,
and so is not bitwise — or a change to what is splatted, which is results-gated.

**What the gate would be.** Not bitwise. A change to the summation order needs a float64
reference with a stated tolerance; a change to what is splatted needs the fidelity suite
on all five sequences plus `gate_node.py` and `gate_hupr_node.py`. That is the full
protocol, for an item whose ceiling is 30% of a step time that already meets its target.

**Recommendation.** Do not open this unless step 1 lands *and* something downstream
raises the bar above 600 s a sequence. If it is opened, survey first and cost each option
against its gate before writing code — the pattern in this round has been that
pre-measurement estimates were wrong by factors of 1.5 to 3 in both directions.

---

## Step 3 — from the parity audit: re-measure C2, and possibly remove it

Not in the original plan; it came out of the dataset parity audit
(`RENDERER_DATASET_PARITY_AUDIT.md` §4c) and it is the cheapest of the three.

The `compact` branch in `profiles_to_rae_spectrum_fused` runs the azimuth FFT over the
occupied elevation columns only, then un-compacts with one `index_copy_` before the
elevation FFT. It is taken on the cascade (sparse elevation grid) and not on HuPR.

Two reasons to revisit it:

1. **Runtime.** Its `index_copy_` costs **0.63 ms/step** on the H100 — the third largest
   entry in RT-Pose's unattributed bucket. It was worth −0.85 ms on a 4090, where the
   traffic it avoided was expensive; that saving has not been re-measured here, and on a
   machine with 3x the bandwidth it may well be net negative.
2. **Uniformity.** It is the only conditional in the fit whose taken branch depends on
   the array. Removing it would leave the executed path identical on both datasets apart
   from the ADC decoder and the gain calibration.

**How to measure it.** `HUPR_RAE_C2=off` against the default, same sequence, same flags,
step time only — no new code. The two routes are bitwise each other by construction (the
skipped columns are exactly zero), and `MM4DPS_RAE_CHECK=1` confirms it cell by cell on
values and all gradients, so this is one A/B run per dataset and a dump comparison.

**Decide on the measurement.** If `off` is faster or within the run-to-run null on
RT-Pose, remove the branch. If it is still clearly positive, keep it and record the
number next to §4c so the asymmetry is at least justified by a current measurement
rather than a 4090 one.

---

## Ordering

| | item | gate | measured target | effort |
|---|---|---|---|---|
| 1st | **Step 3** — re-measure C2 | bitwise, already proven | 0.63 ms + an asymmetry | two A/B runs, no code |
| 2nd | **Step 1** — PSF hoist | bitwise by construction | <= 4.60 ms HuPR, <= 0.97 RT-Pose | a bench, then one kernel |
| 3rd | **Step 2** — `doppler_plane_sorted` | float64 or full results | 6.02 ms RT-Pose | a survey first |

Step 3 goes first because it needs no new code, settles a question the audit raised, and
either wins time or justifies the one remaining asymmetry with a current number.

---

# Resolved: Steps 3 and 1, measured 2026-09-23

## Step 3 — C2: KEEP IT. Do not remove the branch.

Measured on both machines, RT-Pose 176, two replicates per arm, 12 epochs, paired on two
GPUs so any host contention hits both arms equally.

| machine | frame batch | C2 on | C2 off | C2's effect |
|---|---|---|---|---|
| 2x RTX 4090 | 1 | 26.30 / 26.30 ms | 26.88 / 26.89 ms | **saves 0.58 ms/step** |
| H100 | 1 | 8.76 / 8.77 ms | 8.72 / 8.73 ms | costs 0.04 ms/step |
| H100 | 4 (production) | 20.77 / 20.73 ms | 20.52 / 20.53 ms | costs 0.22 ms/step |
| HuPR 256, H100 | default | 19.30 ms | 19.29 ms | 0.01 ms — a no-op, as predicted |

The within-arm replicate spread is 0.00-0.04 ms on every row, so every effect above is
many times its own null. HuPR is a no-op because `n_el_slots = 2` and `n_el = 2`, so
`len(occupied_el) < n_el` cannot hold there — which is why that row also serves as an
independent null.

**Gate:** all 40 dump lines byte-identical between on and off under `--deterministic on`,
and `MM4DPS_RAE_CHECK` reports "values bitwise; gradients ra bitwise, planes bitwise,
floor bitwise" in both arms. The two routes are bitwise each other, so the choice is
free and is purely a measurement.

**Verdict.** The earlier plan said to remove the branch if `off` was faster on the H100.
`off` *is* faster there — by 0.22 ms at the production frame batch — but that was the
wrong question to decide on alone. On a 4090 C2 is worth **0.58 ms/step in the other
direction**, an order of magnitude more than it costs on the H100. Removing it would
trade 0.58 ms on the machine that is short of headroom for 0.22 ms on the machine that
has plenty. So C2 stays on by default, on both arrays.

This leaves the array-dependent conditional in place, and that is the right call: it is a
pure sparsity optimisation whose output is bitwise identical either way, it costs the
H100 0.22 ms a step (0.19 GPU-hours over a 101-sequence RT-Pose corpus, against 30.6),
and it is load-bearing on 24 GB cards. The asymmetry is now justified by a current
measurement on both machines rather than by a single 4090 number, which was the point of
revisiting it.

**One observation recorded but not established.** At `--frame-batch 2` on a 4090, the
`off` arm ran out of memory after 3 epochs while `on` completed at 23.0 GiB reserved of
23.5. That is consistent with the compact path lowering the peak — the non-compact
azimuth FFT runs over all `n_el` columns instead of the occupied ones — but it is a
single observation and `torch_max_memory_allocated` as logged does not discriminate
between the arms, so it is **not** evidence, only a note. It did not affect the decision.

**Aside, worth knowing:** RT-Pose under the 2x2 objective does **not** fit a 24 GB 4090 at
`--frame-batch 4`; it needs 1 or 2. And on torch 2.9 the allocator variable was renamed,
so `PYTORCH_CUDA_ALLOC_CONF` is ignored with a deprecation warning — it must be
`PYTORCH_ALLOC_CONF=expandable_segments:True`.

## Step 1 — PSF hoist: CLOSED, do not build it

The two pre-checks the plan required were run, and the mechanism does not survive them.

**Table size — passes.** 51,338,880 (cell, entry) pairs on RT-Pose 176 = **391.7 MiB** as
two float32; 3,492,720 pairs = 26.6 MiB on HuPR 256. Affordable.

**Is the kernel bound by what the hoist removes — fails, in two different ways.**

- `table4`, the PSF source the lerp reads, is **120 KiB**, comfortably L2-resident on an
  H100 (50 MB L2). So the lerp is already close to free. The precomputed table, by
  contrast, has **no reuse**: 51.3M pairs drawn from 3.4M path entries, so `entry[k]`
  today gets about 15x reuse and largely hits cache, while a per-pair PSF table must
  stream from DRAM. The hoist cuts bytes read per pair from 40 to 32 and yet *raises*
  DRAM traffic. That is risk 1 from the plan, confirmed.
- On HuPR — the larger target at 4.60 ms/step — the kernel is not memory-bound at all.
  `SCENE_LANES = 8`, so the launch is `3072 cells x 8 = 24,576 threads = 768 warps` over
  132 SMs, about **9% occupancy**, each thread walking ~142 run iterations serially (max
  ~1263). Achieved bandwidth is ~255 GiB/s, **7.6% of peak**. It is latency- and
  occupancy-starved. Removing the lerp cannot help it.

RT-Pose's launch is the opposite — 49,152 cells x 8 = 393,216 threads, and ~1978 GiB/s,
roughly 60% of peak — so it is the memory-system-bound one, and it is also the one where
the reuse argument says the hoist makes DRAM traffic worse.

**The lever that would actually help HuPR** is splitting each cell's run walk across more
threads to fill the GPU. That changes the summation order, so it carries the same
non-bitwise gate as Step 2 and is not a cheap win.

Closed at the cost of two probe runs and no kernel work, which is what the pre-check was
for.

## Where that leaves the ordering

Steps 3 and 1 are both resolved and neither produces a code change. Step 2
(`doppler_plane_sorted`, 6.02 ms on RT-Pose) is the only item left, it is not
bitwise-gateable, and stepping already meets its 600 s target. **The bitwise path is
exhausted.** Anything further is a float64-or-results-gated decision about what the model
is supervised on, which is not a runtime call to make unilaterally.
