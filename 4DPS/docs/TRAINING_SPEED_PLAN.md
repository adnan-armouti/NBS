# Making the velocity-field trainer tractable on the cascade

Status: plan, 2026-09-12. Nothing below is implemented; the numbers in
section 1 are measured, the numbers in sections 3-4 are estimates and are
marked as such.

## 1. Where we are (measured)

Estimator throughout: `step_ms` from the trainer's epoch line, median of
steady-state epochs (no re-capture), one job per GPU, RTX 4090.

| | HuPR seq 256 | RT-Pose seq 183 |
|---|---|---|
| person points per frame | ~3,600 | ~3,400 |
| scene points the trainer renders | 20,000 | 20,000 |
| frames | 600 | 299 |
| training sub-array | 2 TX x 4 RX = **8** channels per board, 2 boards | 9 TX x 16 RX = **144** channels |
| Doppler planes rendered per view | 33 | 62 (median 55) |
| frames a step | 24 | 4 |
| step (ms) | 28.4 | 94.6 |
| epoch (s) | 0.71 | 7.10 |
| 400 epochs + 80 evaluations (s) | ~330 | ~3,050 |

(The per-view energy support that brought seq 183 to 69 ms a step was
rejected on quality -- at 400 epochs the unrendered bins go unsupervised
and the person smears over Doppler; `docs/RTPOSE.md` -- so the state to
improve is the full support above.)

The point counts are the same. The gap is the array: 18x the channels per
point, half offset by half the frames, and a further ~2x from smaller
steps (75 launches an epoch instead of 25) and the 20k-point scene being
re-splatted at 144 channels every step.

Where a cascade step goes (20-step profile, eager, before the plane
skipping): `doppler_plane_splat_backward_chunked_amps` 53%,
`doppler_plane_splat_forward`/gather 15%, `doppler_plane_amplitudes` 5%;
the rest is the BSDF, the static (scene) splat, the range/angle FFTs and
the loss chain.

### 1.1 What is already factorised, and what is not

`render_factorized` (`mm4DPS/rasterizer.py`) factorises the **inputs**:
Steps 2-3 compute per (point, TX) and per (point, RX) the distances,
directions, Fresnel terms, antenna gains and carrier phases, and the pair
phase is the outer sum `phi_tx + phi_rx` (line ~1211). The relative phase
between virtual pairs -- the azimuth information -- is exact by
construction and costs `M x (n_tx + n_rx)` trig calls, not `M x n_tx x
n_rx`.

What is **not** factorised is the accumulation. Every (point, TX, RX) is a
*path* with its own 15-bin Hann PSF splat into the `(n_tx, n_rx, K)`
profile (Step 5, `scatter_splat.cu`), and in 4DPS its own Doppler-plane
amplitudes: for each of the ~62 planes, a 64-loop sum
`A_d = sum_l W[d,l] exp(j k rdot t_{l,tx})` (`doppler_plane_amplitudes_kernel`),
re-walked in the backward for the rate gradient. Per step on seq 183:

| term | ops per step | note |
|---|---|---|
| person paths | 4 x 3,400 x 144 = 1.96 M | |
| Doppler amplitudes, forward | 1.96 M x 62 planes x 64 loops = 7.8 G complex MAC + 1.96 M x 64 sincos | `doppler_plane_amplitudes_kernel` |
| Doppler amplitudes, backward (rate gradient) | the same 7.8 G again, plus 64 sincos per path | the 64-loop inner loop of `..._backward_chunked_amps_kernel` |
| PSF splat / gather, forward | 1.96 M x 62 x 15 = 1.8 G | scattered by cell (`doppler_plane_gather_kernel`, deterministic binning) |
| PSF gather, backward | 1.96 M x 62 x 15 scattered reads of `grad_out` | the other half of the backward kernel |
| scene, static plane only | 20,000 x 144 x 15 = 43 M scatter | `static_gain_splat`, forward + backward |
| BSDF, Step 4 | (13,600 + 20,000) x 144 = 4.8 M pair evaluations | `bsdf_step4_forward` |

**Roughly 80% of the Doppler-term arithmetic is the 64-loop slow-time
sum, done per path per plane, twice.** That sum has a closed form
(section 3.1). The PSF splat is the *cheap* synthesis -- 15 bins per path
instead of the 256 fast-time samples -- and section 3.3 explains why the
"dense fast-time GEMM" idea does not beat it.

## 2. Decisions and constraints

### 2.1 Decided 2026-09-12, done on branch `hann-windowing`: Hann on every FFT, both sides

Before the branch only the rendered range response was Hann-shaped
(`hann_psf`); the measured range, and azimuth and Doppler on both sides,
were unwindowed (HuPR's released preprocessing uses direct FFTs) -- an
inconsistency in range of 73% on a point target (`qa window-check` with
`MM4DPS_FFT_WINDOW=none`). Now every FFT takes its window from one
definition, `signal_model.fft_window`:

| axis | window | measured side | rendered side |
|---|---|---|---|
| range (256 ADC samples) | periodic Hann `0.5 - 0.5 cos(2 pi n/N)` | `data.*`: on the ADC samples before the range FFT | `rasterizer.hann_psf`, that window's exact response (unchanged) |
| azimuth (8 / 86 grid slots) | symmetric Hann without end points, `hanning(N+2)[1:-1]` | `data.virtual_to_grid`, per slot | `forward_render._to_grid`, per slot (every angle FFT goes through it) |
| Doppler (64 loops) | periodic Hann, after mean removal | `forward_render.slow_time_to_doppler` (the products' consumers, the targets, the metrics, motion detection) | `doppler_dft_rows`: DFT row x window, minus its mean over loops (the mean removal folded in), used by `doppler_plane_weights` and the per-unit weights (gathered from one table, bitwise) |
| RadProposer's four axes | as above; the padded-TX axis over its 3 real entries | `adc_frame_to_radproposer_tensor` | `range_profiles_to_radproposer_tensor` |

The shipped product moved to slow time -- `(frame, loop, tx, rx, range)`,
per-loop range profiles -- because a windowed Doppler transform cannot be
undone by a consumer that must remove the mean first; `data.product_to_doppler`
takes the transform. The calibrated noise envelopes describe the process
(estimated unwindowed); the trainer smooths the prior by the window's
3-tap power kernel (`windowed_process_envelope`), the renderer draws
slow-time noise from the envelope and the consumer windows the draw.
The processes the renderer synthesises per range bin -- the diffuse
floor, the receiver nuisances -- never went through a PSF, so their draws
are put through the range window in the profile domain (the exact 3-tap
`range_window_profiles`), and the scene level their calibrated scales
are relative to is the UNWINDOWED profile's magnitude quantile
(`unwindowed_range_reference`: the window divided out in the ADC domain,
sample 0 aside). Without both, the rendered floor came out 1.5-1.8x too
high relative to the measured one (`floor_level_ratio` 0.56/0.66 ->
0.86/1.21 on the first branch gate).

Gate: `python -m mm4DPS.qa window-check --dataset {hupr,rtpose}` -- one
point target through both chains agrees to 1.5e-7 (float32 rounding) on
range profiles, loop RA, the Doppler cube, the Doppler planes and the
RadProposer tensor; the per-unit weights are bitwise
`doppler_plane_weights`; the 15-bin PSF truncation is the one
approximation left, 7.3e-4 of the peak, and it is the same on both
datasets. Every HuPR number moves with this; the branch re-baselines them.

3.1's closed form is then the Hann Doppler PSF: `0.5 D(delta) - 0.25
D(delta - 1) - 0.25 D(delta + 1)`, three Dirichlet terms sharing one
numerator per path, exactly as `hann_psf` in range, plus the folded mean
removal (a per-path constant times the row's mean). A Hann-windowed
Doppler response is compact (-31 dB first sidelobe, decaying 18 dB an
octave), which also makes a truncated Doppler splat legitimate later, the
way the 15-bin range truncation is.

### 2.2 Constraints

- The forward model stays tied to the recorded array: every rendered
  channel is a real (TX, RX) pair at its real position and loop slot, the
  beamformer is the measurement's, and nothing is deposited in the
  angle domain. (The azimuth/elevation-PSF splat is rejected below, 5.5.)
- HuPR's published numbers are not moved by default. The renderer (the
  products, the numerical gate at 2e-6) is untouched by everything here;
  the trainer changes that alter fp32 rounding are validated on HuPR with
  an n >= 5 end-of-run arm before HuPR is switched to them (the trainer is
  not reproducible step to step, so a first-step gate is necessary but not
  sufficient).
- PSF spread stays 15. 400 epochs + cosine stays uniform across sequences.
  Every loss is on linear magnitude.
- `--eval-every 10` remains a final, separately-approved step.

## 3. The proposal, in order

### 3.1 Closed-form slow-time DFT in the Doppler-plane kernels -- DONE (branch `hann-windowing`)

Implemented as `mm4DPS/cuda/doppler_closed_form.cuh` (the Hann Doppler
PSF, three Dirichlet terms sharing the path's fractional bin, the folded
mean removal, the analytic rate gradient), selected by the trainer's
`--doppler-amplitudes {closed-form,dft}` (default closed-form). Gate:
`qa doppler-check` -- against the float64 reference
(`signal_model.doppler_plane_amplitudes_reference`, itself 4e-14 from the
loop sum), the closed-form kernels are at **7.3e-7** (amplitudes) and
**5.4e-7** (rate gradient); the DFT kernels are at 5.1e-6 / 4.0e-6. The
closed form is the more accurate route by 7x, at fp32 rounding. Measured
(seq 183, 10 epochs): alone it gave nothing -- 91.5 -> 94.4 ms a step --
because the backward was bound by its scattered `grad_out` reads, not by
the loop arithmetic (section 3.2); with 3.2 it is the difference between
64.1 and **54.2 ms**.

The original analysis, kept for the record:

**Mechanism.** The plane weights are plain fftshifted DFT rows
(`signal_model.doppler_plane_weights`, "reproduces
`fftshift(fft(profile, dim=loop))[bin]` exactly"), and a path's slow-time
phasor is a pure tone: with TDM sample times
`t_{l,tx} = t0_tx + l T_loop`,

    exp(j k rdot t_{l,tx}) = exp(j k rdot t0_tx) * exp(j w l),   w = k rdot T_loop.

So for plane `d` (unshifted frequency `d'`):

    A_d = exp(j k rdot t0_tx) * sum_{l=0}^{63} exp(j (w - 2 pi d'/64) l)
        = exp(j k rdot t0_tx) * (1 - exp(j 64 w)) / (1 - q_d),   q_d = exp(j w) exp(-j 2 pi d'/64),

because `exp(-j 2 pi d') = 1`. The numerator is the **same for every
plane**; only the denominator differs, by a constant per plane. Per
path: two sincos (`w`, `64 w`) and one more for `t0_tx`; per plane: one
complex multiply and one complex division. Against today's 64 MACs and
one sincos per (path, loop) walked once per plane tile, that is a
~30-60x reduction of the amplitude arithmetic. Plane 0 (`loop0` anchor,
`W = e_0`) is `exp(j k rdot t_{0,tx})`; the `dc-mean` anchor is the
`d' = 0` case of the same formula. Padded (`-1`) planes stay zero.

**Gradient.** The rate gradient the backward computes is
`dA_d/d(rdot) = dA_d/dw * k T_loop` (plus the `t0_tx` term), with

    dA_d/dw = [ -j 64 exp(j 64 w) (1 - q_d) + (1 - exp(j 64 w)) j q_d ] / (1 - q_d)^2,

again one numerator shared over planes. The backward's 64-loop inner
loop disappears; what remains of that kernel is the 15-bin `grad_out`
gather.

**Numerics.** (a) Near a bin centre, `1 - q_d -> 0` and `1 - exp(j64w)
-> 0` together (`A_d -> 64`); evaluate with a guarded series for
`|1 - q_d| < eps` (the Dirichlet kernel's standard treatment: `A = 64 *
exp(j 63 (w - 2 pi d'/64)/2) * sinc-ratio`, or a 3-term Taylor
expansion), and choose `eps` from a float64 reference so that the fp32
error stays below 1e-6 relative across the whole `w` range. (b) The
result is not bitwise today's 64-term sum; it is the same quantity to
fp32 rounding (~1e-6), which is at the trainer's own nondeterminism
floor. The renderer does not use these kernels, so products and the
2e-6 gate are unaffected. (c) The mod-64 integer reduction the current
weights rely on becomes irrelevant: the angle `w` is reduced once per
path, in float, exactly as `sincosf_reduced` already does.

**What changes for the model.** Nothing: the same slow-time model, the
same bins, the same per-pair rates and TDM slot times. Doppler stays
exact per pair (no per-point rate approximation).

**Expected gain (estimate, to be measured at step 1 before anything is
integrated).** Amplitudes kernel 5% -> ~0; the backward kernel's loop
part (~80% of its arithmetic) -> ~0, leaving its scattered `grad_out`
gather; the forward gather and the static splat unchanged. Step 95 ->
roughly 45-55 ms on seq 183 (~1.8x); HuPR 28 -> roughly 18-22 ms (its
33 planes x 64 loops are the same structure). 400 epochs + evals: RT-Pose
~3,050 -> ~1,600-1,900 s; HuPR ~330 -> ~230-260 s.

**Risk.** Low on the model, medium on the implementation (a new
backward; the guard region). Fully gated: synthetic bitwise-vs-float64
reference, then the trainer's first-step and 10-epoch A/B, then the HuPR
n >= 5 end-of-run arm.

### 3.2 Coalesce the backward's `grad_out` gather -- DONE

The per-unit backward reads the `(base, planes)` transpose (one transpose
pass; `HUPR_DOPPLER_AMPS_T=0` restores the planes-major reads): bitwise
the same result, and on the cascade -- whose planes-major working set,
62 planes x 147k cells x 8 B = 73 MB, is the whole L2 -- the step went
from 91.5 to 64.1 ms on the DFT route and 94.4 to **54.2 ms** on the
closed-form route (seq 183, 10 epochs, steady state). HuPR seq 256
(windowed pipeline, 41 planes): 32.3 -> 26.7 ms.

| seq 183, ms a step | planes-major reads | transposed reads |
|---|---|---|
| DFT amplitudes | 91.5 | 64.1 |
| closed-form amplitudes | 94.4 | **54.2** (4.07 s an epoch) |

After both, the 20-step profile is: Doppler backward 15%, Doppler forward
18% (gather 8%, amplitudes 6%), the static (scene) gain map 10%, BSDF 5%,
and the rest a long tail of torch ops in the loss chain (select/index
backward, elementwise, copies: ~45%). The Doppler term is no longer the
bottleneck; the next lever is that tail.

The original analysis, kept for the record:

After 3.1 the backward is a scattered read of `grad_out[(d, cell)]` for
15 cells x 62 planes per path. The `..._t` kernel variants already read a
`(base, planes)` transpose so a path's plane column is one contiguous
segment, but the **per-unit W route** (the one every per-view-support run
takes) is wired to the non-transposed `chunked_amps` kernel only. Extend
the transposed layout to the per-unit route (the transpose is one cheap
kernel; the per-unit W is only read in the loop part that 3.1 removes,
so the remaining kernel no longer needs `unit_of_path` at all).
Estimate: the remaining backward halves; step ~45-55 -> ~35-45 ms. Risk:
low; the kernel exists for the other route and is bitwise-gated there.

### 3.3 Why NOT the dense fast-time synthesis (evaluated, not recommended)

The "factorised synthesis" idea -- accumulate `R[tx, rx, n] = sum_p a_p
U_p[tx, n] V_p[rx, n]` over the 256 fast-time samples and FFT once -- keeps
the pair phase exact (it is the same outer sum the code already forms)
and removes the PSF truncation, but it costs `M x n_tx x n_rx x 256 x
planes` MACs = 76 G complex MACs a step on seq 183 (62 planes), against the
15-bin splat's 1.8 G. Density does not reduce the count: an outer product over
(TX, RX) still visits every pair for every sample. It only wins if a GEMM
runs ~40x faster than the scattered gather, which needs TF32/FP16 tensor
cores (a precision the losses have not been validated at) and a
per-point-chunk materialisation of `U V` (600 MB per 2k points). And to
stay exact it must keep the per-pair amplitude and rate as scalars inside
the sum, at which point it is the same work as today's kernels with a
256-wide inner loop instead of 15. Verdict: after 3.1 the sparse splat is
the cheaper synthesis by an order of magnitude in arithmetic; this is
kept as an option only for a future FP16 study, not for this plan.

### 3.4 Smaller items, in the order they pay

| item | saving on seq 183 (estimate) | what it changes |
|---|---|---|
| `--eval-every 10` | ~95 s of the 190 s eval per run (HuPR ~23 s) | half the held-out curve's points; deferred by decision, final step |
| scene contribution cached across the 5-epoch eval | eval time only | none on training |
| fewer re-captures (pad points grow per epoch: 7 captures per run at ~0.8 s) | ~5 s a run | pad with headroom once; none on results |
| the static (scene) splat at 144 channels every step | ~3-5 ms a step (to be profiled after 3.1) | none if the scene's splat components stay cached as now; the cost is the per-step gain multiply + scatter, which 3.2's layout also helps |

## 4. What the plan is expected to deliver

Estimates, RT-Pose seq 183, to be replaced by measurements as each step
lands:

| state | ms/step | s/epoch | 400 epochs + 80 evals |
|---|---|---|---|
| R7 (unwindowed, full support, 62 planes) | 95 | 7.1 | ~3,050 s (51 min) |
| branch, windowed, DFT route, planes-major (the pre-3.1 state) | 91.5 | 6.86 | ~2,940 s |
| after 3.1 + 3.2 (measured) | **54.2** | **4.07** | **~1,820 s (30 min)** |
| + `--eval-every 10` (if approved) | -- | -- | ~1,720 s |

HuPR seq 256: 32.3 -> 26.7 ms a step on the windowed pipeline (41 planes;
the R7 unwindowed state was 28.4 with 33): ~315 s for 400 epochs.

HuPR seq 256: ~330 s today -> ~250 s after 3.1-3.2 (its steps are
smaller in the same proportion). The remaining floor is the BSDF, the
range/angle FFTs of 4 x 27 x 144 x 256 per step and the loss chain; going
below it means rendering fewer paths (section 5.1) or fewer frames.

## 5. Alternatives considered

### 5.1 One channel per grid slot (86 of 144) -- available, approximate

The beamformer averages the 58 doubled grid slots (`Radar.azimuth_grid`:
28 slots hold one channel, 58 hold two). Rendering only the first channel
of each doubled slot keeps the same 86-slot aperture, angle FFT and loop
slots and removes 40% of the paths (linear in everything above). It is an
approximation: the two channels of a slot share a phase centre but not
their bistatic amplitude or their exact range (the far-field split
between them), and the measured side still averages both. Cheap (a day,
mostly gating), reversible (`--training-array slots`), and orthogonal to
3.1-3.2. Recommended only if 3.1-3.2 fall short of the target, and then
with a 400-epoch A/B on the micro-Doppler metrics, since it changes the
rendered RA at the ~1e-2 level, not the ~1e-6 level.

### 5.2 A sparser training aperture (every other RX, or fewer TX) -- not recommended

Changes the beam the RA loss sees; the measured cube would have to be
sub-apertured identically for the loss to be consistent, and the
products would then be rendered from a model fitted on a coarser beam
than they show. Larger saving than 5.1, larger and less defensible change.

### 5.3 Fewer person points per frame -- not recommended

Linear saving, but the person is what every loss is about, and 3,400
points is already HuPR's density.

### 5.4 Per-point (instead of per-pair) amplitude or Doppler rate -- not recommended

The bistatic angle across the 16.8 cm aperture at 2-5 m is <= 5 degrees,
so a per-point amplitude is a few-percent approximation on diffuse terms
and larger on a specular lobe. The rate spread across the array is
`v_tangential x aperture / range ~ 1 m/s x 0.168 / 3 m = 0.06 m/s`, ~1.5 of
the cascade's 0.039 m/s Doppler bins at the array edges for a person
walking across at 3 m -- not negligible against a measured spread of ~4
bins. Neither is needed by 3.1-3.2, which keep both exact.

### 5.5 Splat directly into range-azimuth(-elevation) with an angular PSF -- rejected

Depositing each point at its (range, azimuth, elevation) with a PSF for
the angle FFTs makes the cost independent of the array (a 2-D/3-D
footprint per point instead of 144 paths), but it severs the model from
the recorded array: no per-pair amplitude, no doubled slots, no TDM
slot-time coupling between Doppler and TX, and the beam pattern becomes a
modelled kernel rather than the consequence of the antenna positions.
Rejected by decision: the method stays tied to the actual TX and RX.

### 5.6 Reduced precision in the kernels (TF32/FP16 accumulation) -- deferred

Could apply to 3.3's GEMM form or to the existing splat's contributions.
Unvalidated against the losses' 1e-6-level gates; deferred until 3.1-3.2
have landed and the residual is known.

### 5.7 One sequence over two GPUs -- rejected

Intra-sequence parallelism is not used in this project (one job per GPU;
different sequences run concurrently). Three sequences on two GPUs is
already how production runs.

## 6. Steps

Each step ends with a measurement in the format of section 1's table and
a gate; a failing gate reverts the step, nothing is committed until its
gate passes (one commit per step).

1. **Reference and benchmark for 3.1, standalone** (half a day).
   A float64 torch implementation of the closed form (`A_d`, `dA_d/dw`)
   on synthetic paths spanning `w` across all 64 bins including exact bin
   centres; compare to the existing 64-term sum in float64 (must agree to
   1e-12) and to the fp32 kernels' outputs (the current kernels vs the
   float64 reference: this gives the *existing* fp32 error, the bar the
   new kernels must not exceed). Choose the guard `eps`.
2. **Kernels for 3.1** (1-2 days). New forward amplitudes and backward
   rate-gradient paths in `kernels.cu` behind the same bindings
   (`hupr_doppler_plane_gather_forward`, `..._splat_backward`), selected by
   a trainer flag `--doppler-amplitudes {dft,closed-form}` (default
   `closed-form` for non-HuPR, `dft` for HuPR until step 5). Gate: synthetic
   A/B against step 1's float64 reference; the per-unit active-plane
   skipping preserved (pads exactly zero).
3. **Measure 3.1** (an hour). `vf_ablate.sh` on the seq-183 workspace, 10
   epochs, both flags, one job per GPU; the first-step loss must agree to
   the trainer's floor (5 significant figures on HuPR seq 256 was the
   sub-array's bar); record `step_ms`. Go/no-go: >= 1.4x on seq 183 or
   stop and re-profile.
4. **3.2, the transposed gather for the per-unit route** (a day). Gate:
   bitwise against the non-transposed kernel on synthetic inputs (the
   transposition is a layout change only), then step 3's timing again.
5. **HuPR validation of the closed form** (GPU time, little work). n >= 5
   400-epoch runs of seq 256 per arm (`dft` vs `closed-form`), the
   evaluation suite's micro-Doppler and RA metrics compared as
   distributions; switch HuPR's default only if the arms are
   indistinguishable at that n. Until then HuPR keeps `dft` and its
   numbers do not move.
6. **RT-Pose 400-epoch check** (GPU time). Seq 183 with the final
   trainer against the current `rtpose_after` run: micro-Doppler
   centroid/spread/spectrogram/energy correlations, measured/rendered
   spread, RA Pearson, the micro-Doppler figure side by side. Then the
   production runs (183, 186, 153) and `METRICS_RTPOSE_MASKON.md`.
7. **Only then, and only if wanted:** 5.1 (`--training-array slots`) as a
   measured, flagged option; `--eval-every 10` as the final step.

Effort: steps 1-4 about a week of implementation and gating; steps 5-6
are mostly GPU time (HuPR: 5 x 2 x ~6 min; RT-Pose: 3 x ~25 min after
3.1-3.2).

## 7. What gets recorded

For every step: the table of section 1 (both datasets, `step_ms`,
s/epoch, 400-epoch projection, estimator named), the first-step loss
agreement, the gate result, and -- for anything that touches HuPR -- the
n >= 5 end-of-run comparison. The RT-Pose before/after quality comparison
goes to `outputs/rtpose_maskon/METRICS_RTPOSE_MASKON.md` and the
timing table to `docs/RTPOSE.md` ("Training time").
