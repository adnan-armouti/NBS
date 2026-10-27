# Metrics: definitions, conventions, and where each number comes from

Every metric in the paper is computed by `mm4DPS/evaluation.py` and written,
per sequence, to `outputs/<run>/seq<S>/metrics.json` by
`python -m mm4DPS.chain run`. This file defines them. Nothing here is
computed by hand.

## Conventions

**Domain.** Every metric is computed on **linear amplitude**, defined as
`sqrt(max(power, 0))` where `power` is the stored product. Never on decibels,
never on power. Log compression inflates the agreement of empty bins: the same
renders score 0.959-0.975 on a dB spectrogram and 0.790-0.942 on the identical
data in linear power. Amplitude, not power, because the house rule is "linear
magnitude" and magnitude is amplitude; RA and RD are already magnitude maps, so
amplitude keeps micro-Doppler in the same domain as the other two products.

**The one exception: PSNR.** `PSNR = 20*log10(peak / RMSE)`, where `peak` is
the maximum of the *real* amplitude map and RMSE is computed on linear
amplitude. The logarithm is applied to a scalar ratio *after* the error is
computed on linear data, so it does not compress anything that is being
measured. This is what makes the number comparable to 3DPS, mmIR, and the
image-reconstruction literature. Unlogged `peak/RMSE` is comparable to nothing
and must not be reported as PSNR.

**Micro-Doppler DC row.** Drop rows that are identically zero across all frames
(the clutter-removed DC row) before computing anything.

**Frame selection.** State explicitly whether a metric is over all frames or
the motion subset (person-cloud centroid speed >= 0.15 m/s, pre-registered).
`metrics.json` records `frames_scored` and `frame_selection_rule`; do not
describe a number as "over motion frames" unless `frames_scored < total`.

**Seeds.** Every trained-model number is the mean of >= 3 seeds with the
standard deviation reported. A single-seed number is not a result.


## Segmentation masks

Two masks, built differently on purpose. Neither uses a learned segmenter: a
model trained on natural images would sit inside a physics-fidelity metric and
apply its biases unequally to the rendered and real signatures.

**Mask A - range-Doppler occupancy.** Cell-averaging CFAR on the **real**
product, applied unchanged to both real and rendered. Purpose: ask whether the
render puts energy where the real capture has it. Report the false-alarm rate
`P_fa` used; sweep it in the supplement.

**Mask B - micro-Doppler topology.** Computed **independently on each
signature**. Seed with CFAR at a strict `P_fa`, then grow by hysteresis into
4-connected neighbours above a looser threshold derived from the same noise
estimate. Independent masks because the question is whether the rendered
structure has the same connectivity as the real one; a shared mask would assume
that rather than measure it. Hysteresis because plain thresholding fragments
the weak limb arcs and destroys the connectivity the topology metrics exist to
score.


## Range-azimuth

Per frame, then mean and standard deviation across frames.

**The headline range-azimuth metric is the clutter-removed one.** The plain
metrics below are retained as sanity figures and **must not be reported as
evidence that a render is correct**, because they do not discriminate.

| Metric | Definition |
|---|---|
| **Clutter-removed Pearson** | Pearson per frame on `x - mean_t(x)`, **each side using its own temporal mean**, then averaged |
| **Occupied clutter-removed Pearson** | the same, restricted to Mask A computed on the clutter-removed **real** product |
| Occupancy fraction | mean fraction of cells in Mask A, reported so the restriction is auditable |
| Pearson *(sanity only)* | `corrcoef(real.flatten(), rendered.flatten())[0,1]` |
| SSIM *(sanity only)* | `structural_similarity(real, rendered, data_range=real.max())` |
| NRMSE *(sanity only)* | `RMSE / sqrt(mean(real**2))` |
| PSNR *(sanity only)* | `20*log10(real.max() / RMSE)` |

**Why, measured.** Scored against a *different* sequence's real product over
nine ordered sequence pairs per radar:

| variant | matched min | matched mean | permuted max | permuted mean | permuted pairs above the weakest matched |
|---|---:|---:|---:|---:|---:|
| Pearson | +0.7315 | +0.8010 | +0.7735 | +0.6745 | 2 / 12 |
| SSIM | +0.5661 | +0.7409 | **+0.7974** | +0.6888 | **11 / 12** |
| occupied Pearson | +0.7407 | +0.7988 | +0.7936 | +0.6976 | 2 / 12 |
| **clutter-removed** | +0.0398 | +0.2653 | +0.0183 | +0.0008 | **0 / 12** |
| **occupied clutter-removed** | +0.0469 | +0.2994 | +0.0217 | +0.0013 | **0 / 12** |

The plain Pearson and SSIM are dominated by the static room, the range gate and
the antenna pattern, which every HuPR recording shares. SSIM is the worst: its
permuted maximum exceeds its matched *mean*, and four renders score higher
against another sequence than against their own. A metric that cannot separate
two sequences cannot separate two methods.

**Occupancy restriction alone does not fix it** — the static scene is bright
and sits inside the occupied cells. Slow-time mean removal is what works.

**Each side removes its own mean.** A shared mean would put measured data into
the rendered path, which is the replay this project forbids.

The absolute value falls from ~0.80 to ~0.30. That is the correction, not a
regression: roughly 0.67 of the old figure was scene the model never had to
earn, and what remains is attributable because the permuted control sits at
+0.0013.

**Report the weak number.** The discrimination is carried by sequences 102 and
256, matched +0.34 to +0.54. **Sequence 38 sits near the floor** — +0.0469
horizontal and +0.1033 vertical against a permuted ceiling of +0.0217, so it
separates by roughly 2x and no more. Sequence 38 is mostly still, and after its
temporal mean is removed there is little left in its range-azimuth maps. That
belongs in the paper next to the other two.


## Range-Doppler

Split each frame by Mask A. `M` is the occupied set, `U` the unoccupied set.
All ratios are rendered over real, so 1.00 is exact.

| Metric | Definition |
|---|---|
| Occupied Pearson | Pearson over `M` only, per frame, then averaged |
| Signal power ratio | `sum(rendered[M]**2) / sum(real[M]**2)` |
| Floor level ratio | `mean(rendered[U]) / mean(real[U])`, per frame, then averaged |
| Floor sigma ratio | `std(rendered[U]) / std(real[U])`, per frame, then averaged |
| SCNR ratio | `SCNR_rendered / SCNR_real`, where `SCNR = mean(x[M]**2) / mean(x[U]**2)` |

Supplement diagnostics:

| Metric | Definition |
|---|---|
| Floor Wasserstein-1 | `scipy.stats.wasserstein_distance` between rendered and real amplitude histograms, pooled over every unoccupied bin and every frame |
| Floor lag-1 autocorrelation | Per unoccupied bin, Pearson of `x[t]` against `x[t+1]` across frames; report the mean for real and rendered separately, not a ratio |

Report the level ratio and the sigma ratio separately. A render can match the
floor's mean exactly with no speckle at all, and level alone cannot see it.


## Micro-Doppler

`S[d, t]` is the amplitude spectrogram, Doppler bin by frame. Power weights use
`S**2`.

| Metric | Definition |
|---|---|
| centroid | Per frame, `sum(d * S**2) / sum(S**2)` with `d` measured signed from DC. Report Pearson of the real against rendered series. |
| spread | Per frame, `sqrt(sum((d - centroid)**2 * S**2) / sum(S**2))`. Pearson of the series. |
| spectrogram | Pearson over the whole flattened 2D amplitude array. |
| energy | Per frame, `sum(S**2)`. Pearson of the series. |
| env+ | Per frame, the largest `d > 0` with `S[d,t]` above the Mask B threshold. Pearson of the series. |
| env- | Per frame, the most negative such `d`. Pearson of the series. |
| BW ratio | `mean(env+ - env-)` rendered over real. One scalar for total Doppler extent. |
| clDice | Centreline Dice between the Mask B segmentations of real and rendered, using a skeletonisation of each. |

Also report **median spread in bins** for real and rendered as a pair, not only
their correlation. Correlation can be high while the rendered signature is
narrower or wider, and a ratio names that where a correlation hides it.

Supplement diagnostics:

| Metric | Definition |
|---|---|
| Cadence error | Peak of the mean cadence spectrum, the FFT of `S` along each Doppler bin averaged over bins, which recovers the stride rate. Report absolute error in Hz. |
| Spectrogram SSIM | As RA, on the 2D spectrogram |

## What `metrics.json` holds

| key | what |
|---|---|
| `radars.<radar>.range_azimuth` | the clutter-removed and sanity Pearson/SSIM/NRMSE/PSNR of the range-azimuth product, mean and std over frames |
| `radars.<radar>.range_azimuth_suppressed` | the same on the clutter-removed product |
| `radars.<radar>.range_doppler` | the Mask-A occupied metrics and the noise-floor figures |
| `radars.<radar>.microdoppler` | centroid, spread, spectrogram, energy, env+/env-, BW ratio, clDice, cadence, Betti deltas, spectrogram SSIM, the median spreads in bins |
| `radars.<radar>.frames_scored`, `frame_selection_rule` | the motion subset actually used |
| `microdoppler_score` | the spectrogram stage's own score, for cross-checking |
| `render_config` | the physics the render used: visibility on/off and its mask, gains, nuisance seeds, the velocity parameterisation, product layout, stage timings. Two products whose `render_config` differ in anything but the declared variable must not be compared. |
| `spectrogram_reduction` | `person-range-gated` -- the 21-bin Hann window at the person's range. The ungated alternative disagrees by up to 0.26 in centroid correlation and is refused unless asked for. |

The paper's table is rendered from these JSONs; `hupr_fidelity.{md,tex}` are
written under `--intermediates`.

### What the velocity field actually contributes

The ablation is the unfitted render — the velocity field trained for **zero
epochs**, leaving the camera-derived initial field untouched, rendered and
scored identically. It differs from the fitted run in exactly one thing.

| seq | radar | metric | unfitted | fitted |
|---|---|---|---:|---:|
| 102 | H | centroid | **+0.970** | +0.997 |
| 102 | H | spread | **+0.211** | +0.734 |
| 102 | H | energy | **+0.017** | +0.864 |
| 102 | V | centroid | **+0.968** | +0.996 |
| 102 | V | spread | **+0.278** | +0.755 |
| 102 | V | energy | **−0.037** | +0.944 |
| 38 | H | centroid | **+0.953** | +0.979 |
| 38 | H | spread | **−0.103** | +0.713 |
| 38 | H | energy | **+0.023** | +0.899 |
| 38 | V | centroid | **+0.958** | +0.983 |
| 38 | V | spread | **−0.163** | +0.847 |
| 38 | V | energy | **−0.114** | +0.940 |

**The camera already supplies most of the bulk radial motion.** Unfitted
centroid correlation is +0.953 to +0.970, and fitting moves it to +0.974 to
+0.997 — a gain of +0.016 to +0.040. That is resolvable (the true run-to-run
floor on centroid is 0.0036, so these are 4.6× to 11× it) but it is an order of
magnitude smaller than what the fit does to the limb signature. Anyone claiming
the velocity field is what makes the centroid work would be wrong; the camera
has most of it already.

**The velocity field supplies the limb signature.** Unfitted spread is
**−0.163 to +0.278** and unfitted energy is **−0.114 to +0.023** — at or below
zero, meaning the unfitted render carries no information about the width of the
micro-Doppler signature or its frame-to-frame energy. Fitted, those reach
+0.713 to +0.847 and +0.864 to +0.940.

That contrast is the contribution, and it is a narrower and stronger claim than
"the render matches the measurement". Across three sequences and two radars,
**all twenty-four comparisons gain beyond the measured run-to-run floor and
none is worse** — but the *sizes* are what matter: centroid gains ~0.03 where
spread gains ~0.61 and energy ~0.91 (`docs/EVAL_VALIDATION_REPORT.md`,
check 5).

Power ratios, rendered/measured, where 1.000 is exact:

| band | mean | spread (max/min) |
|---|---|---|
| RD anchor (scatterer-free bins, clutter-removed) | 0.458 | 18.0x |
| RA anchor (same bins, first loop, clutter kept) | 0.765 | 4.3x |
| first 15 range bins (RA) | 0.952 | 1.1x |
| person band (RD) | 0.814 | 2.2x |
| all range (RD) | 0.780 | 2.2x |

**RD anchor and RA anchor are the same range bins and are not
interchangeable.** Clutter removal subtracts whatever is constant across the 64
slow-time loops, and the measurement at those bins is 99.5% loop-coherent, so
RD sees about 1% of what RA sees. A level fitted on one can be badly wrong for
the other -- that is exactly what happened when the diffuse floor level was
fitted on RD alone: RD went 0.105 -> 0.933 while RA went 0.817 -> 19.534 and
the range-angle maps were visibly wrecked. The fit is reverted and
The diffuse-floor fitter (removed in the deployment strip) reported a fit unsafe when the two domains
disagree. `docs/FLOOR_LEVEL_AND_NEAR_FIELD_RESULTS.md` records it.

Rendered spread sits below measured on every radar. That is the
under-dispersion: rendered magnitude CV is 0.52 against 0.78-0.81 measured, so
the spectrogram is narrower than it should be. It is the open item most likely
to move the `spread` column.

Earlier stages of this work are in `docs/HIGH_MOTION_SEQUENCES_REPORT.md`,
`docs/NOISE_MODEL_RESULTS.md`, `docs/GVHMR_DIFFUSE_FLOOR_RESULTS.md` and
`docs/FORWARD_MODEL_CONSTANT_AUDIT.md`. Their numbers predate the linear
spectrogram metric and the GVHMR switch, so they are history, not the current
state.

---


## Reporting hazards

1. **Three spectrogram correlations can be computed** for one sequence and
   radar: dB power, linear amplitude, linear power. They differ by up to 0.17.
   Only the linear-amplitude number is the specified one, and it is the one in
   `metrics.json`.
2. **A dB loss is live in the material trainer.** `mm4DPS/train_material.py`
   fits the person material through `log_pearson_loss`, a decibel-domain
   objective. It is a training objective, not a metric, but it contradicts the
   linear rule, and changing it invalidates every person checkpoint. Flagged,
   not fixed.
3. **n=1 is not a result.** The trainer is not reproducible run to run
   (`docs/` in the source repository measures the floor); every trained-model
   number should be the mean of >= 3 seeds with the spread reported. The
   renderer alone is reproducible to float32 epsilon (`atomicAdd` order,
   max relative 7.1e-7 on rendered products) and bitwise on measured ones.
