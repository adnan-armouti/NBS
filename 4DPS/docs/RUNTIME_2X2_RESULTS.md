# The runtime optimisations under the 2x2 objective, on 8x H100 80GB

Branch `runtime/2x2`, `runtime/4term` merged with `main` at bd6ce83. Measured on the node, five
sequences, 400 epochs, one GPU each, two waves of five so both configurations saw the same machine
load. Raw products under `/data/mmsd/4DPS_2x2/outputs/{off2x2,on2x2}`.

## 0. What the objective is now, and why the optimisations still apply

`main` moved from four terms to **`OBJECTIVE=2x2`**: one functional over {global, windowed} x
{loop-0, per plane}. Micro-Doppler and the person window are ablated; two WINDOWED RAE terms take
their place (`--ablate-loss-term loop0-ra,plane-ra,md --person-window-loss 0 --window-rae-terms on
--elevation-planes all --rae-floor-gradient on`).

The two global RAE terms are unchanged, and the windowed pair is computed from the SAME prediction
tensors -- one transform, two scopes. So every optimisation here, which all attack PRODUCING those
tensors, now serves four terms rather than two.

## 1. The gate: bitwise against main's own code

Run against main at bd6ce83 in its own worktree, plus the arithmetic-free print instrument and the
`pw_el_mask` fix of section 4. 20-step deterministic dumps, RT-Pose 176 and HuPR 256:

| configuration | vs main |
|---|---|
| every optimisation OFF | **BITWISE**, both sequences -- the transcription is exact |
| A1 + C + C2 + P1 ON | **BITWISE**, both sequences |
| + item B (the azimuth fftshift) | HuPR **BITWISE**; RT-Pose step 0 to **1.59e-07** in the terms and **8.01e-08** in the gradient norms |

All four bitwise items hold unchanged under the new objective. Item B remains the only reassociating
change, and its deviation is below this pipeline's own 4e-7 gradient nondeterminism under the shipped
`--deterministic off`. HuPR is bitwise even with item B because its 16 angles fold the shift into
input signs rather than rolling.

## 2. What it is worth on H100

| sequence | off ms | on ms | delta | stepping off | stepping on |
|---|---|---|---|---|---|
| RT-Pose 176 | 25.43 | **20.97** | **-17.5%** | 779 s | **645 s** |
| RT-Pose 108 | 24.20 | **19.73** | -18.5% | 740 s | **605 s** |
| RT-Pose 191 | 23.68 | **19.32** | -18.4% | 724 s | **594 s** |
| HuPR 38 | 19.96 | **18.88** | -5.4% | 212 s | 201 s |
| HuPR 256 | 20.46 | **19.43** | -5.0% | 217 s | 207 s |

Corpus, 101 RT-Pose + 235 HuPR sequences:

| | off | on | saving |
|---|---|---|---|
| stepping | 35.0 GPU-h | 30.6 GPU-h | **4.42 (12.6%)** |
| stage onward | 47.4 GPU-h | 43.2 GPU-h | 4.13 (8.7%) |

## 3. The two findings that matter more than the percentages

**These gains are roughly half the 4090's, and HuPR's are a fifth.** On the workstation the same
items were worth -40% on RT-Pose and -25% on HuPR; here they are -18% and -5%. The reason is that
every one of them removes MEMORY TRAFFIC, and an H100 has ~3.35 TB/s against the 4090's ~1 TB/s, so
there is proportionally less traffic cost to remove. The absolute saving is nearly constant across
sequences -- -4.46, -4.47, -4.36 ms on RT-Pose and -1.08, -1.03 on HuPR -- which is what a fixed
amount of removed traffic on a fixed tensor size looks like. **HuPR is launch-bound on this
hardware, not bandwidth-bound**, which is why it barely moves.

**The 20 ms / 600 s target was a 4090 framing and is essentially met here without A2.** Stepping is
594-645 s across the three RT-Pose sequences, and the step is 19.3-21.0 ms. A2 -- the one
non-bitwise item left, and the one that needed a deterministic tree reduction and pairwise or fp64
accumulation -- is not needed to reach the target on this hardware. If node runs are what matter, it
should not be paid for.

**Memory is a non-issue here.** Peak reserved is ~30 GiB of 80 on RT-Pose. Everything in
docs/RUNTIME_4TERM_RESULTS.md section 1 -- the 637 MiB margin, the three OOM'd baseline runs,
`release_graph_pool` as a blocker -- is a 24 GB workstation problem only. `release_graph_pool` is
still correct and still worth keeping; it is simply not load-bearing on the node.

## 4. A bug on main, found by the gate and wanting upstreaming

`static_losses` gathers every flat view with `value.index_select(0, view_rows)`, and with the
windowed RAE terms on and the elevation bound off -- main's own measured choice -- `pw_el_mask` is
set to None, so the captured step dies on its first call. **Reproduced against main at bd6ce83 with
main's own default flags: `AttributeError: 'NoneType' object has no attribute 'index_select'`,
rc=1.** Every reader already uses `flat_views.get(...)`, so the key is simply not inserted when
unused. This branch carries the one-line fix; main still needs it.

## 5. What is not done

* Fidelity and the downstream gates have NOT been run under 2x2. They are not needed for the four
  bitwise items -- bitwise is a stronger statement than any results gate -- but item B reassociates
  at 1e-7 on RT-Pose, so its products should be shown to sit inside the run-to-run null before the
  branch is relied on. That null must also be re-measured for this objective.
* `denp` and `den0` are per-view constants recomputed every step, and computing `denp` materialises
  the fp32 plane target that A1's kernel exists to avoid. Precomputing them at the store build is
  the obvious next commit; it was left out so this merge stayed bitwise-verifiable against main.
* A2 is not started, and on this hardware it is not needed (section 3).

## Re-costing the planned next steps, measured on the H100 under 2x2

The next steps carried over from `runtime/4term` (`docs/RUNTIME_STEP_20MS_PLAN.md`
§13/§14) were all sized on a 4090 under the four-term objective. Both the hardware and
the objective have changed, so they were re-measured before being believed — the same
discipline that moved C from "1.55 ms" to 5.80 ms and A1 from "-13 to -14" to -8.4.

`MM4DPS_PROFILE_STACK=1 --profile-steps 20`, all optimisations on, route `fused`:

| bucket | RT-Pose 176 | HuPR 256 |
|---|---|---|
| **renderer (total)** | **9.12 (44.7%)** | **10.49 (55.5%)** |
| — `doppler_plane_sorted` fwd + bwd | 6.02 | 4.12 |
| — `static_scene_frames` fwd + bwd | 1.25 | 4.77 |
| — `bsdf_step4` fwd + bwd | 0.95 | 1.01 |
| unattributed | 4.40 | 2.00 |
| — of which `index_copy_` (C2 scatter) | 0.63 | — |
| — of which `rae_spectrum_input` fwd + bwd (C) | 0.40 | 0.58 |
| copies (`direct_copy`) | 1.07 | 0.69 |
| azimuth FFT | 0.91 | 0 |
| compiled (inductor) | 0.75 | 0.83 |
| A1 fused term kernel | 0.68 | 0.57 |
| gathers | 0.55 | 1.98 |
| reductions | 0.52 | 0.47 |
| elevation FFT | 0.48 | 0.49 |
| step | 20.38 | 18.91 |

Two label corrections to the bucketing script: the rows it prints as "reference RAE
route" are generic complex kernels matched by a signature pattern written when that
route was live (`indexFuncLargeIndex...ReduceAdd`, `CUDAFunctor_add<c10::complex>`).
The run log says `RAE route fused`, so those 0.64 / 0.24 ms are other callers, not the
reference route. And C2's `index_copy_` falls through to unattributed.

### Verdicts

**A2 (fold the elevation DFT and its reduction into A1's kernel) — drop.** It was
sized at 4.5-5 ms on a 4090. Everything it could absorb now totals 1.68 ms on RT-Pose
(elevation FFT 0.48 + reductions 0.52 + A1 0.68) and 1.53 on HuPR, and it would absorb
only part of each. It is also the one item in the plan that is *not* bitwise: it needs a
float64 gate, a deterministic tree reduction and pairwise accumulation. A ceiling of
3-8% does not justify that gate. A2 was a bandwidth play, and the H100's ~3.35 TB/s
against the 4090's ~1 TB/s is exactly what removed it.

**Step 2's copies — drop.** 1.07 / 0.69 ms. This was already subsumed by A2 and is now
below the threshold where a restructuring pays.

**Structured azimuth sparsity (rows 0/2/5 as three contiguous runs, short DFTs plus
phase ramps) — drop.** Its entire target is the azimuth FFT: 0.91 ms on RT-Pose and
**zero on HuPR**, where `fold_az` already removes it (no `prime_fft_factor` or
`regular_fft_factor` kernel appears in the HuPR profile at all). So it is an RT-Pose-only
item with a ceiling of 4.5%, and short DFTs plus ramps are not bitwise against the full
FFT.

**Precompute `den0` / `denp` at the store build — drop as a runtime item.** The
candidate compiled kernels total about 0.04 ms/step; the whole inductor bucket is
0.75 / 0.83. The argument for doing it anyway is memory, not speed: computing `denp`
materialises the fp32 plane target that A1 exists to avoid. Peak reserved is ~30 GiB of
80 here, so that argument has no force on this hardware either.

**The renderer — the only item of size left, and it grew.** 7.95 ms and 31% of the step
on the 4090; 9.12 ms and 44.7% on RT-Pose here, 10.49 and 55.5% on HuPR. It moved the
opposite way from everything else because it is compute-bound, so the H100's bandwidth
does not touch it. Every other bucket shrank around it.

It is two different kernels depending on the dataset:

- **RT-Pose: `doppler_plane_sorted_forward` 3.14 + `_backward` 2.88 = 6.02 ms**, 30% of
  the step, one call each.
- **HuPR: `static_scene_frames_forward` 4.60 ms alone**, 24% of the step over 6 calls,
  against a backward of 0.17 ms — the forward costs 27x its own backward per call.

That asymmetry is the one lead worth following. `_StaticSceneFrames.backward` returns a
gradient for `carrier` and `None` for everything else, and the class docstring states
that the gains "carry no gradient here". So inside
`static_scene_frames_forward_kernel`, only `carrier` varies between steps: the tap
offset `dn`, the `psf_lerp(table4, dn + half, frac)` it feeds, and the `gains_s` row are
all functions of frozen geometry (`runs`, `entry`, `table4`) that the kernel recomputes
30,000 times. Hoisting the per-(cell, entry) complex PSF factor into a precomputed table
trades a `table4` gather plus a lerp for one 8-byte load, and is **bitwise by
construction** — the same values in the same order, read instead of recomputed. Whether
it pays is a question for an isolated kernel bench, not an argument; the gains row is
re-read per cell that shares the path and may already be the binding constraint.

### What this means for the target

Stepping is 594-645 s per sequence here against a target of <= 600 s, so this is
optimisation past the point of need. The remaining levers are one bitwise candidate
inside the renderer worth at most a few percent unless the PSF hoist lands well, and a
set of items whose gates now cost more than their savings.
