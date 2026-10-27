# From 30 minutes to 10: the next steps for the cascade trainer

Status: plan, 2026-09-12, branch `hann-windowing`. Follows
`TRAINING_SPEED_PLAN.md`, whose 3.1 (closed-form Doppler amplitudes) and
3.2 (transposed plane-gradient reads) are done and measured.

## 1. Where the step is now (measured)

RT-Pose seq 183, windowed pipeline, full support (62 planes), sub-array
(144 channels), 4 frames a step, closed form + transposed reads:
**54.2 ms a step, 4.07 s an epoch, ~1,820 s for 400 epochs with 80
evaluations** (about 1,630 s of steps and 190 s of evaluation).

The 20-step eager profile (55 ms a step; the captured step is the same
kernels without launch gaps):

| where | ms a step | share |
|---|---|---|
| Doppler-plane backward (`..._chunked_amps`, transposed) | 8.3 | 15% |
| Doppler-plane forward: gather 4.5, closed-form amplitudes 3.2, binning ~2 | ~10 | 18% |
| static scene gain map, forward + backward (20k points x 144 channels x 15) | 5.6 | 10% |
| BSDF step 4, forward + backward | 2.6 | 5% |
| **the torch tail**: complex()/reshape/transpose of the (4 x 63 x 144 x 256) plane cube (464 MB a pass), the fused RA spectrum, plane-cell gathers, `select_backward` zero-fills, gradient `add_`s, copies | ~25 | ~45% |
| the rest (FFTs, loss reductions, optimiser) | ~4 | 7% |

The Doppler kernels are a third of the step; the tail is the biggest
single item and is nothing but data movement.

## 2. The target

~600 s for a sequence. With `--eval-every 10` (95 s of evaluation) that
is ~505 s of steps: **17 ms a step** at 75 steps an epoch, a 3.2x step
reduction. Without touching eval: 13.7 ms. Each lever below has its
saving estimated against the profile; the sum of the safe ones reaches
about 24 ms, the sum of everything about 15 ms. Getting to 10 minutes
therefore needs at least one of the levers that changes what is
supervised (section 4), and that is a decision, not an optimisation.

## 3. Levers that change nothing about the model (do these first)

### 3.1 Kill the torch tail (-12 to -15 ms; a few days)

The plane cube leaves the gather kernel as two real `(planes, units*base)`
arrays and then goes through `torch.complex` -> reshape -> transpose ->
(range `index_select`) -> sign fold -> transpose+contiguous -> FFT, and
back through all of it in autograd, with `planes[:, 0]` / `planes[:, 1:]`
selects whose backward zero-fills the full cube twice and `add_`s them.
Every one of these is a full pass over 464 MB.

* have the gather kernel write the layout the angle FFT reads: complex
  interleaved, range kept (only the `keep_range` bins, ~half), sign-folded,
  channel axis last -- the FFT then runs on the kernel's output directly.
  One custom output format, no torch passes between kernel and FFT.
* return the RA plane and the Doppler planes as two outputs of the
  autograd function (the backward is one `cat`), not two selects of one.
* fuse the plane-cell gather + magnitude + the loss reduction into the
  compiled plane term for the whole chain (it is compiled for the loss
  part already; extend it upstream to the FFT output).
* the same for the loop-0 RA term and the micro-Doppler marginal.

Estimate from the profile: the tail's 25 ms halves or better. Gate: the
loss is bitwise the current chain's (a layout change) or at the fp32
floor where the reduction order moves.

### 3.2 The static scene gain map (-3 ms)

20k scene points x 144 channels x 15 spread, forward and backward, every
step, for the scene's materials/normals gradient. Two options: (a) update
the scene's parameters every k-th step only (their gradient is a sum over
frames anyway; k = 4 is a different optimiser schedule for the scene, not
a model change) -- saves 3/4 of it; (b) render the scene at one channel
per grid slot (86 of 144): the scene is static and its beamformed RA at
the averaged doubled slots is what the loss reads; approximation at the
1e-2 level on the scene term only.

### 3.3 `--eval-every 10` (-95 s a run)

Half the evaluation passes. Deferred by decision; the final step.

### 3.4 Smaller items (-2 to -3 ms together)

* the Doppler forward's deterministic binning (~2 ms) runs every step
  though the flat indices only change with the batch: cache the binning
  per batch composition (the same frames recur every epoch in a different
  order -- key on the sorted frame set).
* the transpose pass of 3.2 (~1 ms): write the plane gradient transposed
  from the loss's backward directly.
* the two `pad_points` re-captures per run: pad once with headroom.

After 3.1-3.4: **~30-35 ms a step, ~1,000-1,150 s** (with eval-every 10,
~900-1,050 s). Not 10 minutes.

### 3.5 Truncate the Hann Doppler PSF to 15 taps, as 3DPS truncates the range PSF (-10 to -13 ms)

Today every path writes its amplitude into every supported plane -- the
closed form makes each value O(1), but there are still 62 of them per
path, and the gather and the backward walk all 62. With the window the
Doppler response is compact: measured on the reference, the Hann Doppler
PSF is 1.02e-3 of its peak at 7 bins from the path's fractional bin and
6.7e-4 at 8 (-60 / -64 dB) -- the same level as the range PSF's
truncation at +/-7 bins (7.3e-4, `qa window-check`). So a path can
deposit into the 15 planes around its own Doppler bin only, exactly the
3DPS range trick's second half now legitimately available in Doppler
(it was not without the window: the Dirichlet tail is 1/delta). Per path
the amplitudes, the gather's slot walk and the backward's tile loop
shrink from 62 planes to at most 15; on the cascade that is most of the
Doppler term's remaining 18 ms. The approximation is quantified by the
same gate as range (a point target through both chains) and is the one
the model already accepts in range. The kernels need a per-path plane
window (start bin, 15 entries) instead of the per-unit plane list, and the
supported-plane set stays as the OUTPUT (a path outside every supported
plane deposits nothing).

### 3.6 What section 3 has delivered so far (measured, seq 183, 10 epochs)

| change | ms a step | note |
|---|---|---|
| start of plan 2 | 54.2 | |
| 3.5, the Doppler PSF taps | exact 47.1, 7: **41.2**, 5: 40.6 (with the learned floor in the plane term, below, and the tail cuts) | `--doppler-taps {15,9,7,5}`, **default 7**; truncation of the planes 7.3e-4 / 3.5e-3 / 7.7e-3 / 2.3e-2 (the Hann response falls 18 dB an octave from -31 dB at 2.5 bins, in bin units whatever N is -- 64 Doppler bins are no more compact than 256 range bins). 400 epochs on seq 183, 7 taps against exact: spectrogram +0.941 / +0.933, energy +0.884 / +0.861, RD occupied +0.399 / +0.394, the same learned floor (level 1.04, W1 6.7) -- indistinguishable at n=1. Two gather variants were measured and dropped: a sparse (paths, 18) layout with a shared-atomic gather (2.4x slower: a cell's paths cluster in Doppler and serialise), and a per-slot reach test in the dense gather (its presence in the loop cost 2.8 ms a step even when disabled, 4.5 -> 7.3 ms; enabled, 4.3). |
| the scene's backgrounds from one fused kernel pair (3.1/3.2): the contributions never built | **30.8** | `assemble_static_profile_frames_cuda`, `HUPR_FUSED_SCENE=0` for the per-frame chain. The step's scene chain was 12 ms, not the 5.6 of section 1: build the (15, 2.9 M) contributions 0.6, permute them into cell order 2.4, stack 1.3, four per-frame segmented sums 2.2, four per-frame backwards 2.4, the gradient adds and the spread reduction ~3. Now: the paths sorted once by (channel, range bin) so a cell walks one or two contiguous runs and a channel's cells stream through L1; carrier x the PSF lerp synthesised per slot, every frame's gain applied to it, fixed-order sums; the backward one thread per path in the same order. `qa scene-check`: 1.1e-6 forward / 2.3e-7 gradient against the torch chain, 0.72 + 0.24 ms for 2.9 M paths x 4 frames. First step of the trainer the same to seven figures. Two layouts were measured on the way: cell-order permutation with random 16-byte records, 3.1 + 1.9 ms; the sorted order with random carrier reads, 2.8 + 0.24; gathered into the sorted order, 0.72 + 0.24. |
| the window route's amplitudes compact, (paths, 3 + taps), and a tap-lane gather | **27.5** | the dense (paths, planes) amplitudes were 972 MB a step at 7 taps -- a 1 ms memset and a 4.3 ms gather of 90% zeros. Now every entry is written (zero where the path deposits nothing) and the gather runs one warp per slot stream with its lanes over the 3 + taps entries into per-warp shared accumulators (distinct planes per slot, no atomics; the warps summed in a fixed order, so the cell's sum is a function of its slot order alone, as before). `qa doppler-check --taps 7` PASS, the same truncation; the trainer's first step the same to seven figures. The scene path's two gathers: the (points, 4) gains gather took torch's small-inner-dimension path (1.46 ms); as complex128 rows it is 0.055 ms. |
| 3.4, the person's position-only splat geometry cached per flat row (the binning cache, generalised) | **25.7** | the captured step re-rendered distances, antenna gains, carrier phasors, range bins, PSF weights and splat indices from the batch's positions every step. `build_row_geometry` fills them once for every flat row (chunked fill passes of `render_factorized`'s geometry cache, 0.2 s), and the step's cache is a row gather of them; the cached path runs on the stored values, so the first step is bitwise the uncached one. 1.9 GiB for seq 183's 842 k rows (peak 10.7 -> 12.8 GiB). `HUPR_ROW_GEOMETRY=0` for the A/B. |
| the sorted Doppler route for the person (3.1 + 3.4 + the contribution fusion, in one) | **17.1** | `_DopplerPlaneSorted`, `HUPR_SORTED_DOPPLER=0` for the slot route. The person's paths in (unit, channel, range bin) order -- each frame's order and per-block runs cached in the row geometry, the batch's assembled by one scatter per step -- so the binning (count, scan, place: 1.9 ms), the (15, paths) contributions and PSF lerp (1.5 ms) and the flat indices go; the forward is one block per (unit, channel, 32 range bins), one lane per cell, eight warps over the block's entries, each lane synthesising its own tap's contribution from the entry's record and adding the path's window row (taken by shuffle from the lanes that hold it) into its accumulator row (shared atomics; the order is not fixed, as the default binning's is not, so `--deterministic on` keeps the slot route); the backward is one thread per entry (neighbours share cache lines): 1.3 ms against 3.4. `qa sorted-check`: planes 1.1e-6, carrier gradient 2.5e-7, rate gradient 1.4e-7 against the slot route; forward 2.3 ms against the slot route's 6.1 at the trainer's shapes; the trainer's first step the same to seven figures. Two layouts were measured on the way: a cell per half-warp (every entry visited once per cell, 15x the visits: L2-bound, 2.5 ms) and one fixed-order warp per block (16.6 KB of accumulators a warp: six warps an SM, 6 ms). |
| the BSDF's steps 1-3 and the amplitude compiled on the cached path; the evaluation's renders out of the row geometry | **15.0**; eval 2.3 -> 1.6 s; peak memory 13.3 -> 7.5 GiB | `_prelude` / `_amplitude` (`HUPR_COMPILED_BSDF=0` for the inline chain): ~150 elementwise launches a render become a few fused kernels; the slab Fresnel is carried as real pairs (cancellation-free complex sqrt, Smith's division with the untaken branch's divisor guarded -- an inf there poisoned the gradient) so the whole function fuses; against a float64 reference the real form and the complex64 reference are equally accurate (max 3.3e-5, 99% 1.2e-6). `unit_render` takes the units' geometry out of the row cache too (`unit_row_cache`), so the evaluation's 40 per-chunk caches (4.7 GiB) go and its chunks run the sorted route. First step 6.653583 against 6.653587 (fp32 floor). One-off: the dynamic compile, ~35 s at the first step. |
| 3.2(a), `--scene-update-every k` (a flag, default 1) | k=4: 14.8 (from 17.2, before the BSDF compile) | the scene's materials and normals move on every k-th step of the run; the other steps replay a second captured graph in which the scene renders without a gradient (the same forward, so the same loss), and the scene's Adam groups sit out (their parameters lifted out of the groups for that step, so the moments are untouched). Not the default: a different optimiser schedule for the scene. 400 epochs on seq 183 (`rt400s_k4` against `rt400s_k1`): train 1.360 / val 1.430 against 1.322 / 1.393, energy correlation +0.798 against +0.845, RD occupied +0.384 against +0.402 -- the scene needs its updates; stays off. |
| the learned floor in the compiled plane term (a fix, +3 ms) | | `plane_noise_power`, (units, planes, kept range, 1) gathered from the floor every step: the floor's range shape had been unconstrained off the person's rows (a 24x floor in one run of two). The gather is the next tail item: 2.9 ms for a (units, 63, 110) table that changes only with the floor's 9 parameters. |
| `_virtual_channels` as a free view when every TX is an azimuth TX (the sub-array) | | n_tx selects and their zero-filling backwards gone, bitwise |
| the RA plane / Doppler planes split as one autograd node (one cat backward) | 48.6 | two full-cube zero-fills + adds gone, bitwise |
| the md term read from the plane spectrum's prefix when the band is most of the planes (the cascade: 53 of 63) | **46.6** | the separate band chain (84% of the plane chain) gone; first-step loss the same to six figures |

Where the rest of the tail is, from the shape-grouped profile: the
(15, paths) contribution assembly for the 2.76 M scene paths and 1.9 M
person paths every step (`_splat_contribs` and the PSF re-lerp: ~5 ms, the
next target -- synthesise the contribution inside the Doppler and static
kernels from (w, phase, psf) instead of materialising it), the plane cube's
remaining passes (complex(), transpose, the range index_select and its
backward: ~4 ms), the compiled loss reductions (~3 ms), copies (~3 ms).

## 4. Levers that change what is supervised (decisions)

### 4.1 Stochastic supervision of the unsupported planes (-8 to -10 ms)

The per-view energy support (render only the ~26 bins that carry the
person's energy) was rejected because the other ~36 bins went
unsupervised and the person smeared into them. Keep the supervision but
sample it: every step render the supported bins plus k random
unsupported ones (k = 8), with the unsupported term weighted by
(36/k) -- an unbiased estimator of the full plane loss, at 34 planes
instead of 62. The trainer already has the machinery (`--plane-sample-count`
draws a uniform plane subset per epoch; `view_plane_bins` is per view).
Risk: gradient noise on the zero targets; the 400-epoch micro-Doppler
metrics decide (centroid, spread, spectrogram, energy against the
full-support run).

### 4.2 One channel per grid slot (-20% to -30% of the step)

86 of 144 paths per point: the beamformer averages the 58 doubled slots,
so rendering one of each keeps the aperture, the angle FFT, the loop slots
and the Doppler; it changes the doubled slots' amplitudes by the
far-field split between the two channels (the bistatic angle across 16.8
cm). Every per-path cost (Doppler term, static map, BSDF, splat) drops
40%. A flag (`--training-array slots`), gated by the 400-epoch metrics.

### 4.3 Frames per epoch

299 frames a step-epoch. The velocity field is a temporal DCT basis (209
bands, a 3.5 Hz cap at 10 Hz); a random half of the frames per epoch,
alternating, halves the epoch. It is equivalent to fewer full epochs in
compute, and the "uniform 400 epochs" rule exists so that no per-sequence
schedule is tuned; a uniform "half the frames per epoch" is a schedule
too. Listed because it is the only lever that reaches 10 minutes on its
own; not recommended without a 400-epoch A/B on all three sequences.

### 4.4 What is NOT proposed

* PSF spread (fixed at 15), the angle FFT length (measured: nothing), a
  sparser aperture (changes the beam the loss sees), fewer person points,
  per-point amplitude/rate approximations, the azimuth/elevation-PSF
  splat (rejected: keeps the model tied to the actual TX and RX), fewer
  epochs.

## 5. The arithmetic

| state | ms a step | 400 epochs + evals |
|---|---|---|
| now | 54 | ~1,820 s |
| + 3.1 tail | ~40 | ~1,400 s |
| + 3.5 15-tap Doppler PSF | ~28 | ~1,030 s |
| + 3.2, 3.4 | ~22 | ~850 s |
| + 3.3 eval-every 10 | 22 | ~755 s (12.5 min) |
| + 4.1 stochastic unsupported planes | ~19 | ~660 s (11 min) |
| + 4.2 one channel per slot | ~13 | ~490 s (8 min) |
| + 4.3 half the frames | ~13 | ~300 s |

With 3.5 in the safe set, section 3 alone reaches ~12.5 minutes; 4.1 or
4.2 takes it under 10.

HuPR follows the same path at its own scale (26.7 ms now; the tail is the
same code): ~15 ms after section 3, ~250 s a sequence.

### 3.7 The 400-epoch gate of the batch (seq 183, `compare_runs.py`, n=1 each so far)

| run | centroid | spread | spectrogram | energy | RA Pearson | RD occupied | floor level | train | val | velocity stage |
|---|---|---|---|---|---|---|---|---|---|---|
| `rt400f_tap7` (before this batch: taps 7, learned floor) | +0.993 | +0.622 | +0.941 | +0.884 | +0.423 | +0.399 | log_scale +0.096 | 1.3247 | 1.3977 | 1,782 s |
| `rt400s_k1` (the batch through the sorted route) | +0.993 | +0.529 | +0.930 | +0.845 | +0.423 | +0.402 | +0.097 | 1.3217 | 1.3930 | 850 s |
| `rt400s_k4` (`--scene-update-every 4`) | +0.992 | +0.498 | +0.928 | +0.798 | +0.422 | +0.384 | +0.098 | 1.3596 | 1.4303 | 782 s |

Two more runs of the final tree (with the compiled BSDF): `rt400s_b` +0.995 / +0.678 / +0.927 / +0.790 / +0.423 / +0.386, train 1.3461 / val 1.4162, 768 s; `rt400s_c` +0.993 / +0.574 / +0.928 / +0.814 / +0.424 / +0.391, 1.3287 / 1.3987, 779 s. So at n=3 the final tree's energy correlation (0.790-0.845) sits below the two pre-batch runs (0.861, 0.884; exact and 7 taps), spectrogram 0.927-0.930 against 0.933-0.941, with RA, RD occupied, the floor and the losses the same (run b is the worst on every count, including its loss). The per-step arithmetic is not the cause: on HuPR seq 256 the sorted route and the slot route agree on the first four steps' per-unit terms AND every parameter group's gradient norm to six or seven figures (`MM4DPS_PRINT_FIRST_STEP=1`, `hupr_first.sh`), batch after batch -- the two routes compute the same step; what differs run to run is the order-free sum (both routes have one). The HuPR chain gate (`gate_hupr_final`, 10 epochs, two boards) passes at 17.1 ms a step against 26.4; its 10-epoch losses (2.63, 2.43 sorted; 2.32, 2.31 slot) go through a transient at epoch 3 (the loss rises to 8-9 in every run) and are not a gate. The renderer's numerical gate: 1.96e-6, PASS (unchanged renderer). Whether the energy correlation's n=3-against-n=2 gap is chaos needs more old-tree runs; the trees compute the same gradients.

## 5.1 The chain around the trainer (found while gating)

The person stack took 153-458 s per chain run against upstream's 28 s.
Measured: the person material trainer (pass A) runs in 5 s by hand and
305 s under the chain's launcher -- torch.compile's parallel
compile-worker pool hangs for 300 s at process exit when stdout is a
file (`timeout -s ABRT` dump: training finished, the main thread in the
atexit shutdown). Fixed in `chain.run_module`:
`TORCHINDUCTOR_COMPILE_THREADS=1` for every stage. The motion detection
(138 s) is measured next under the same fix. Every gate today carried
that overhead; the chain gate should be ~3 min end to end.

## 6. Order and gates

0. 3.5 (the 15-tap Doppler PSF) first: it is the largest single saving,
   the smallest change (the kernels' plane loop bounds), and its gate is
   the point-target check already in place.
1. 3.1 (the tail), in three commits: the kernel output layout + FFT
   (bitwise gate against the current chain on a step), the two-output
   autograd function (bitwise), the compiled upstream chain (fp32 floor).
   Timing after each.
2. 3.4 and 3.2(a). Timing.
3. 4.1 behind a flag, 400-epoch A/B on seq 183 (both GPUs: full support
   vs sampled), metrics side by side.
4. 4.2 behind a flag, the same A/B.
5. 3.3 last, with the same A/B as evidence that the held-out curve is
   still sampled densely enough.
6. 4.3 only if asked.
