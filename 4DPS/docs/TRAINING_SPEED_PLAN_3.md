# RT-Pose training under 10 minutes: where the 747 s go, and the plan

Status: 2026-09-13, branch `hann-windowing`, after the seedless chain
(docs/FIDELITY_PLAN.md 7-8). Everything measured on seq 183 (299 frames,
the cascade's 9 x 16 training sub-array, 4 frames a step, 75 steps an
epoch, 400 epochs, evaluation every 5), GPU 0 idle otherwise.

## 0. The frame count: 299 is the recording

Every RT-Pose sequence we use has **299 frames recorded**: the TDA2 index
header of each of the four device files declares 299, the data files are
exactly 299 frames long (940,572,672 bytes = 299 x 4 RX x 192 chirps x 256
samples x 2 x 2 bytes), and the camera and LiDAR folders hold 299 files each
(stems 000002 ... 000598, every second stem). Nothing is dropped anywhere
in the chain: the person sequence, the targets and the training frame list
all carry 299. The capture's first radar frame simply never made it into the
recording (the sensors are stamped from 000002). No fix needed.

## 1. Where the velocity stage's 747 s go (measured)

| phase | s | how measured |
|---|---|---|
| trainer start-up | **90** -> 26 (3.1 done: 66 s -> 1.5 s) | wall stamps on a cold run (below) |
|   of which the target cache, frame by frame | 66 | `cached frame k/299`, 220 ms a frame |
|   its 2.5 GB `vf_targets_seq183.pt` written to disk | 2.5 | -- and read back by nothing in a fresh run |
|   gain initialisation, row geometry, imports | 6 | |
|   graph capture (8 distinct step shapes) + the first epoch at 8 s | 15 | `captured training step ... 6.8 s cumulative` |
| training steps | **458** | 400 x 75 x 15.3 ms |
| evaluation | **132** | 80 x 1.65 s (every 5 epochs: 75 held-out frames in chunks of 2, forward only, then the micro-Doppler correlation) |
| render (noisy + clean, concurrent) + spectrograms + scoring | 57 | run log |
| `collect` with `--artifacts` (videos, PNGs, mesh-on-RGB) | 125 | outside the stage; the cube itself is `rename`d, not copied |

Training = 690 s. To 600 s the arithmetic is: start-up -70, evaluation
-66, and the step from 15.3 to ~10 ms (-160). Any two of the three do it;
all three put the stage near 6.5 minutes.

### The step, kernel by kernel (20 eager steps, `--profile-steps`, self CUDA time)

| ms a step | what | note |
|---|---|---|
| 2.55 | **BSDF step-4 backward** (scene 1.3 + person 1.3) | one thread per (point, TX, RX) path, its 27 gradient terms **atomically added** into per-(point, TX), per-(point, RX) and per-point slots -- 16 threads collide on every (point, TX) slot, 144 on every point. The forward is 0.12 ms. |
| 3.13 | Doppler planes, sorted route: forward 1.44, amplitudes 0.47, backward 1.22 | 1.9 M person paths x 15 range taps x 7 Doppler taps deposited into (unit, channel, 63, range) accumulators in shared memory |
| 1.32 | static scene frames: forward 1.01, backward 0.31 | 2.9 M scene paths x 15 range taps |
| 1.10 | Adam | unfused (per-parameter kernels) |
| ~4.0 | **the torch tail**: `copy_` 1.16 (58 launches), `mul` 0.59 (106), `fill_` 0.46 (104), `index_add_` 0.58 (14), `gather` 0.55 (39), `cat` 0.35 (33), `add` 0.32 (25), `index_put` 0.61, `indexing_backward` 0.48, `complex` 0.27 | glue: views, zero-fills, gradient adds, the plane-cell gathers |
| ~2.0 | the compiled loss graphs (5 `CompiledFxGraph`s, the Triton reductions) | |
| 0.5 | FFTs (4) | |
| ~1 | the rest (elementwise, scatter/gather of the row geometry) | |

Total ~15.5 ms, matching the captured step's 15.3.

## 2. The question of GEMM: what the splat is, and what would replace it

What the renderer computes, per channel c and frame: a range-Doppler plane
Y_c[r, d] = sum over paths p of a_pc * h_r(r - r_pc) * h_d(d - d_pc), where
a_pc is the complex amplitude (BSDF x carrier phase), r_pc the path's
fractional range bin, d_pc its fractional Doppler bin, and h_r, h_d the
Hann-window point spread functions (15 and 7 taps kept). In words: a
**non-uniform to uniform transform** -- points at arbitrary (range,
Doppler) positions deposited onto a regular grid through a compact kernel.
That is a type-1 NUFFT, and its two implementations are:

* **sparse spreading** (what we do): each path writes taps_r x taps_d = 105
  values. 1.9 M x 105 = 200 M multiply-adds a step for the person, 44 M for
  the scene. Arithmetically tiny; the cost is memory traffic and the
  ordering of the deposits (atomics, or the sorted route's shared
  accumulators). This is where the 4.4 ms of splat kernels go.
* **dense GEMM**: Y_c = A_c E, with A_c the (channels x points) amplitude
  matrix and E the (points x cells) matrix of complex exponentials. It is
  exact (no PSF truncation) and it is a matmul -- but E has no reuse across
  channels unless the path's bin is the same for every channel, and it is
  not: across the cascade's 17 cm virtual aperture a point's range differs
  by up to 2 bins between channels (the range migration that the per-path
  taps place exactly). Dropping that makes E channel-independent and
  (points x 64 loops x 256 samples) per frame: 6,890 x 16,384 complex, a
  (144 x 6,890) @ (6,890 x 16,384) product per frame = **130 GFLOP a
  frame, 520 a step** -- 100x the spreading's arithmetic. At the card's
  ~30 TFLOP/s complex-fp32 that is 15-25 ms, against 3.1 ms now. The
  scene (no Doppler) would be 0.7 GFLOP x 4 frames, comparable to the 1.3
  ms it costs today, with the range migration approximated away. So no:
  GEMM is not the better primitive here, and the "scatter" is not a
  backbone we chose over it -- it is the cheap algorithm for this
  transform.

The NUFFT literature's other idea -- a narrow spreading kernel on an
oversampled grid, then FFTs -- was proposed here and **withdrawn after
measurement** (3.4 below): a kernel narrow enough to save deposits is
less accurate than the 7-tap truncation, one accurate enough is wider,
and the FFT passes cost more than the deposits saved. The truncation is
the cheap approximation for this transform; its accuracy knob is the tap
count (`--doppler-taps 15`: 8.5e-4 against 4.8e-3, +~1.5 ms a step).

The biggest single kernel is not a splat at all. The BSDF backward's 2.5
ms is contention: 2.9 M + 1.9 M threads each doing 27 atomic adds into
slots that 16, 9 or 144 of them share. One warp per (point, TX) walking
the 16 RX with shuffle reductions, and the per-point terms reduced in
shared memory, writes each slot once: **2.5 -> ~0.6 ms**, with the same
sums in a fixed order (a bitwise gate against the current kernel is not
possible -- the atomics have no order -- but the reference torch backward
is, at 1e-6).

## 3. The levers, in order

Each has a numerical gate that does not need a 400-epoch run; the
end-to-end run comes once, at the end (section 5).

### 3.1 The target cache on the GPU (-60 s a run; numerical gate 1e-7)

`velocity_field.py`'s cache loop calls the **numpy** `range_angle_slow_time_cube`
per frame (255 ms a frame on the cascade, measured in docs/FIDELITY_PLAN.md
7's profiling) and numpy `surface_solid_angle_weights` and
`soft_joint_weights`. The GPU cube exists (`range_angle_slow_time_cube_frames_torch`,
5e-16 of numpy relative to the peak) and the person sequence builder
already runs the same decode in blocks. Do the loop in blocks of frames
on the device: decode -> cube -> RA target, plane targets, Doppler support,
the floor prior, the masks, the weights -- one transfer per block. 66 s ->
~5 s. And stop writing `vf_targets_seq<S>.pt`: it is 2.5 GB per run that
only a repeat run of the same workspace reads, which the chain never does
(the no-cache rule). Gate: every cached tensor within 1e-6 relative of the
numpy build on 8 frames; the first step's loss to seven figures.

### 3.2 Evaluation every 10 epochs, chunks of 4 (-70 s; exact)

The evaluation costs 1.65 s and selects the best snapshot; with the
cosine schedule the best is in the last epochs in every run so far. Every
10 epochs halves it (-66 s); `--eval-frame-batch 4` (the training batch;
the evaluation has no backward, so memory is not the constraint it was)
takes the remaining 40 x 1.65 to ~40 x 1.1. Gate: the metrics at the
epochs that are still evaluated are identical.

### 3.3 The BSDF backward without atomics (-1.9 ms; 1e-6 against the torch reference)

Section 2's last paragraph. `bsdf_backward.cu`: one warp per (point, TX),
lanes over RX (16 of 32 lanes, or two (point, TX) pairs a warp), shuffle
reductions for the per-(point, TX) and per-point terms, the per-(point, RX)
terms through shared memory with one add per slot per block. Also the
forward's early-out (`gf == 0`) shows most person paths are culled per
step -- compact the path list before the backward rather than launching
every path and returning.

### 3.4 The spreading kernel -- WITHDRAWN (measured 2026-09-13)

Section 2's NUFFT form was costed without its FFT passes and its kernel
width was guessed. Measured on 20,000 synthetic paths against the exact
Hann-windowed 64-loop synthesis (float64, `max|err|/peak`):

| route | error | deposits a path (Doppler axis) |
|---|---|---|
| truncated PSF, 7 taps (today) | 4.8e-3 | 7 |
| truncated PSF, 15 taps (`--doppler-taps 15`) | 8.5e-4 | 15 |
| truncated PSF, 31 taps | 1.2e-4 | 31 |
| Kaiser-Bessel spreading on a 2x grid, width 6 | 9.3e-3 | 6 + FFT passes |
| width 8 | 3.2e-3 | 8 + FFT passes |
| width 12 | 8.1e-4 | 12 + FFT passes |

A spreading kernel narrow enough to save deposits is *less* accurate than
the 7-tap truncation it would replace, and one accurate enough (12 wide,
2-D: 144 deposits) costs more than today's 105 -- before the two FFT
passes per direction over 576 (128 x 512) grids (~2 ms a step) that the
formulation adds. It is not a speed lever, and as an accuracy lever it is
dominated by `--doppler-taps 15` (+~1.5 ms a step for 5x less truncation
error). The range axis stays at 3DPS's 15-tap spread in any case. Dropped.
### 3.5 The torch tail (-1.5 to -2 ms; bitwise)

About 400 small launches a step (`copy_`, `mul`, `fill_`, `cat`, `gather`,
`add`) are glue: buffers re-zeroed every step, views materialised, the
per-unit results concatenated, gradients added into pre-existing buffers.
Under graph capture they cost their kernel time, not their launch, and
their kernel time is 4 ms. Pre-allocate once, write in place, and fold the
elementwise chains into the compiled loss graphs (`torch.compile` already
owns the loss; extend its boundary to the plane-cube assembly and the
gradient adds). Bitwise: the same arithmetic, fewer kernels.

### 3.6 Fused Adam (-0.7 ms; bitwise up to the fused kernel's summation)

`torch.optim.Adam(..., fused=True)` on the parameter groups that are plain
tensors (the row-fused normals keep `_FusedRowAdam`). One kernel instead of
per-parameter ones.

### 3.7 The scene's BSDF at the point, not the path (-1.2 ms; a model change, gated at 400 epochs)

The scene's BSDF is evaluated per (point, TX, RX): 2.9 M evaluations a
step, forward and backward, for 20,000 points. Across the 17 cm aperture
at 3-6 m the incidence angle changes by under 3 degrees, and the
rough-surface lobe by well under a percent. Evaluating it per (point,
TX) -- 180 k evaluations -- and applying the per-path carrier phase on top
removes a step's worth of BSDF work for the scene. The person keeps the
per-path BSDF (its normals turn, its lobe is what the Doppler term reads).
This changes the rendered scene at the 1e-3 level, so it is gated
end-to-end, not numerically.

## 4. The arithmetic -- and what was done (2026-09-13)

| item | gate | result |
|---|---|---|
| 3.1 targets on the device, no cache file | targets within 2.4e-7 of the numpy build, first step to seven figures | start-up 90 -> 26 s |
| 3.2 evaluation every 10, chunks of 4 | the evaluation identical to 2e-9 at 2 or 4 | 1.65 -> 0.98 s, 80 -> 40 of them |
| 3.3 BSDF backward, a warp per point | every gradient within 3.5e-7 of the atomic route; first step to seven figures | 2 x 1.27 -> 0.49 + 0.31 ms |
| 3.5 the spectrum input as one compiled gather | bitwise (0.0) on the spectrum and its gradients | 2.92 -> 2.02 ms fwd + bwd |
| 3.6 fused Adam | 4.9e-8 of the eager update after five steps | 1.10 -> 0.89 ms |
| 3.4 NUFFT spreading | measured worse than the truncation (section 3.4) | withdrawn |

Step 15.3 -> **12.65 ms** (steady state, seq 183). What remains of the
tail (aten self time 3.7 ms a step): fused Adam's 7 launches 0.9, the
row-geometry scatter `index_put` 0.6, 33 small `cat`s 0.35, the
person + scene plane add and the complex() assembly ~0.9, the MD term's
copies 0.4.

| state | ms a step | steps | eval | start-up | training |
|---|---|---|---|---|---|
| before | 15.3 | 458 | 132 | 90 | 690 |
| now (projected) | 12.65 | 380 | 39 | 26 | ~445 |
| **measured, 400 epochs** | 12.8 | | | | **445** (velocity stage 503, the chain 527) |

Measured end to end (400 epochs, seq 183): training 690 -> **445 s**, the
velocity stage 747 -> 503 s, the whole chain 527 s; RA 0.879 -> 0.879, RD
occupied 0.629 -> 0.637, micro-Doppler +0.927/+0.907 -> +0.929/+0.926,
held-out 1.204 -> 1.199. HuPR 256: training 253 -> 231 s, the metrics
within their spread (RA 0.957/0.925 -> 0.951/0.925, RD 0.919/0.822 ->
0.919/0.823, held-out 0.644 -> 0.647).

## 5. Order and gates

1. 3.1 and 3.2 (a day): numerical gates as stated; then one 400-epoch run
   on 183 to confirm the identical trajectory (3.1 changes the targets at
   1e-7, 3.2 changes when the best snapshot can be taken).
2. 3.6, 3.5 (two days): bitwise / first-step gates only.
3. 3.3 (two days): the reference-torch gate at 1e-6; first step to seven figures.
4. 3.7 last, as the one deliberate approximation, with its own 400-epoch A/B.
