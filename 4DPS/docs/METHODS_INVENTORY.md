# 4DPS methods inventory: everything implemented, from both repositories' histories

Written 2026-09-14 for the paper's methods section. Sources: the `vf-clean`
branch of `/home/adnan/Desktop/hupr_radar_rendering_velocity` (568 commits,
2026-08-07 to 2026-09-10, forked from `hupr_radar_rendering` at `113a05f`) and
`main` of `/home/adnan/Desktop/4DPS` (54 commits, 2026-09-11 to 2026-09-13,
the deployed tree). Every item names the module that implements it today
(`mm4DPS/` in 4DPS) so equations can be transcribed from the code, not from
memory. Numbers are the measured ones recorded in the commit messages and
`docs/`; each says which sequence it was measured on.

Where the paper puts things, per the owner's instruction:

* **Section 4 (evaluation / datasets)**: a high-level overview of the
  preprocessing chain (part A below); the details go to the supplement.
* **Methods**: the forward model as equations (part B: bistatic radar
  equation -> single-bounce point-scatterer rendering of received power ->
  what 4DPS synthesises, the complex slow-time / range-Doppler cube per
  physical TX-RX pair), the runtime work that makes fitting a whole dataset
  tractable (part D), and the objective with its three terms, especially
  the k-SNR Doppler support of the 63-plane term (part C).
* **Supplement**: the derivations between the radar equation and the
  rendering equation, the preprocessing details, the numerical gates.

---

## A. Preprocessing: the inputs a sequence's fit needs, all computed from the recording

The chain (`mm4DPS/chain.py`, `python -m mm4DPS.chain run`) has five stages.
Nothing is seeded from outside the recording except the RT-Pose rig extrinsic
(the identity) and the two pretrained vision models (`docs/SEEDS.md`).
Preprocessing is 10-18 s a sequence; GVHMR ~30 s, overlapped.

### A.1 The static scene ("scaffold")

| dataset | method | module |
|---|---|---|
| HuPR (monocular RGB, no calibration published) | Depth-Anything-V2 metric depth on 4 sampled annotated frames (0/150/300/450), the annotated person cut out, back-projected with an assumed 40 deg HFOV, voxel-downsampled and PCA normals on the GPU (`voxel_downsample_torch`, `pca_normals_torch`), 20,000 points | `reconstruction.run_scaffold` |
| RT-Pose (Livox LiDAR at the radar) | the LiDAR frames of the annotated samples, the person's column cleared (0.6 m radius), LiDAR -> radar world by the rig extrinsic, PCA normals (0.5 m radius, Livox's sparsity), 20,000 points | `reconstruction.run_lidar_scaffold` |

**Scene-to-radar range alignment, fitted in-chain (HuPR)**: the monocular
cloud's range translation is the argmax of the min-max-normalised Pearson
correlation between the cloud's (range, angle) occupancy and the
person-free measured map (per-cell 25th percentile of loop-mean |RA| over
every 4th frame), mean over both boards, beyond 0.5 m; range only, because
the map does not identify lateral/height for a co-located rig (competing
optima 0.6-1.4 m apart). GPU, 0.3 s. Seq 256: -0.34 m (r 0.25 -> 0.47);
38: -0.82; 102: -0.73. `reconstruction.fit_range_alignment`,
`clean_scene_maps`. (History: the earlier `HUPR_SCENE_ALIGNMENT` constants
and the `bake-scene-alignment` stage, vf-clean 2026-09-08, R078-R084, were
replaced by this fit on 2026-09-13, commit `5117416`.)

**Scene material initialisation**: 3DPS per-point materials and normals
(as quaternions) fitted to the person-free static |RA| for 75 iterations
(`train_motion dual-radar`, `BACKGROUND_ITERS`), then refined by the
velocity trainer for the 400 epochs. Started as a shipped seed
(`seeds/background`), moved in-chain 2026-09-13 (`a47250b`, `80cf95b`).

**Antenna coupling from the recording**: the coherent mean of the near
range bins (15) per virtual channel over every 3rd frame -- the board's own
static near-field return, added to the rendered profiles in the trainer and
the renderer (99.98% coherent on seq 256). `calibration.run_sequence_coupling`
(`1c8f5e3`; earlier a seed fitted on held-out sequences, vf-clean `cfa0de5`,
`7ae3800`).

### A.2 The person

1. **Mesh recovery**: GVHMR (video human mesh recovery, static-camera mode)
   on the RGB frames, run inside the chain in its own environment, started
   with the reference stage so its ~30 s overlap: frames held in memory,
   YOLO tracking of the largest persistent track, ViTPose and the HMR2
   features in half precision, half-resolution crops (the demo's), SMPL
   vertices (6890) per camera frame. `mm4DPS/gvhmr.py`,
   `gvhmr_fast_dump.py` (`ebb6394`, `0772574`). RT-Pose's true focal length
   (25.65 mm FF-equivalent) is passed; HuPR's is estimated from the image.
2. **Camera -> radar placement**: RT-Pose: camera -> LiDAR by the shipped
   stereo calibration, LiDAR -> radar by the extrinsic; the camera frame
   paired to each radar frame by the rig's clock fitted on the annotation
   (`RTPoseSequence.clock`, stem = 3f-4 or 2f-2; `aa31bec`). HuPR: the
   camera at the rig, co-boresighted, (right, down, forward) -> (x, -z, y).
3. **Per-frame radar registration of depth**: GVHMR's depth is a size
   prior (HuPR 30% long, RT-Pose 7% short). Per frame, the ratio of the
   radar's moving-power range-profile peak (parabolic interpolation) to
   GVHMR's centroid range, frames below a detection confidence
   interpolated, median-filtered over +/-7 frames, Savitzky-Golay polished
   over 51, bounded [0.5, 2]; applied as a scale about the camera origin
   (pixels preserved). `reconstruction.radar_depth_scales`,
   `person_range_series` (`1c8f5e3`; replaced the frame-0 range sweep and
   the rigid dual-radar anchor of vf-clean `a94c82d`/`c0888ad`).
4. **The 14 joints** from the SMPL regressor in HuPR's joint order; joint
   velocities by Savitzky-Golay differentiation (`signal_model.load_trajectories`).
5. **Person material initialisation**: one shared six-parameter 3DPS
   material for the body (`train_material.SharedHumanMaterial`), one epoch
   (`PERSON_EPOCHS`), refined in the velocity trainer; area scale learned.
6. **Motion frames**: the pose-speed selection (centroid speed >= 0.15 m/s,
   pre-registered) and the radar-side person-motion detector on the GPU
   (`motion.run_pose_motion_frames`, `run_detect_person_motion`).

### A.3 Visibility (fixed global point sets + per-frame binary masks)

The scene and the person are fixed point sets; what changes per frame is a
boolean visibility (`b88a938`, vf-clean 2026-08-17). Both masks are
ray-cast on the GPU with a grid-culled Moeller-Trumbore intersection
against the SMPL triangles (`pointset.TriangleGrid`, `ray_cast_visible`,
`sequence_visibility`):

* **person self-occlusion + field of view** per frame (`visible (frames, 6890)`);
* **scene shadow**: background points whose ray to the radar passes
  through the person's mesh that frame (`scene-shadow-mask`,
  `visible (frames, N)`), wired through the rasterizer's `shadow_mask`
  hook into training and rendering (vf-clean `f52d9e3`, R087-R099: with
  the mask the wall dims on all six products as measured).

### A.4 Radar signal processing shared by both sides

One window definition for every FFT on the measured and rendered side
(`signal_model.fft_window`; `753263c`): periodic Hann over range (256 ADC
samples) and Doppler (64 loops, after slow-time mean removal), symmetric
Hann without end points over the azimuth grid. Range PSF on the rendered
side is that window's exact response (`rasterizer.hann_psf`), truncated to
15 bins (fixed, the 3DPS paper). Cascade (RT-Pose): the TI MMWCAS
phase/amplitude calibration applied, 12 TX x 16 RX, the 9 azimuth TX on
the 86-slot virtual grid; HuPR: two IWR1843 boards (horizontal, vertical),
3 TX x 4 RX, TX1/TX3 the azimuth array. `data.py` (`9087022`, `9a030b2`).

---

## B. The forward model: what is rendered, and the equation to write

### B.1 Rendering primitive (3DPS, vendored: `rasterizer.py`, `splat/`)

Per point p with position x_p, unit normal n_p (parameterised as a
quaternion for the scene, `PointPrimitives.get_normals`), area weight a_p
(vertex area x opacity; the person's from a Monte-Carlo solid-angle weight,
`scene.surface_solid_angle_weights_torch`, x a learned area scale), and six
physical material parameters m_p = (eps_real, eps_imag, sigma_h,
correlation length, coherent fraction tau, thickness) through
`reparameterize_torch`: the Jones BSDF (coherent specular slab + SPM/KA
rough-surface diffuse blend, all six ITU parameters) gives a complex
scattering amplitude f(m_p, n_p, d_tx, d_rx) for the incident and exit
directions to TX t and RX r. `render_factorized` step 4
(`bsdf_step4_forward`).

### B.2 The single-bounce range profile per physical TX-RX pair

For TX t at x_t, RX r at x_r, range bin k (the Hann range PSF h at the
path's fractional bin), the rendered complex range profile is the coherent
sum over the visible points

    S_{t,r}[k] = g_radar * sum_p  G_t(p) G_r(p) * a_p * f_p(t, r)
                 * exp(-j k_wave (|x_p - x_t| + |x_p - x_r|)) / (|x_p - x_t| |x_p - x_r|)
                 * h[k - (|x_p - x_t| + |x_p - x_r|) / (2 dR)]     (15 taps)
                 + C_{t,r}[k]                                        (near-bin antenna coupling)

with G the antenna patterns (the MMWCAS patterns in `splat/assets`),
g_radar the learned absolute receiver gain per radar (B.4), C the coupling
of A.1. This is the bistatic radar equation's amplitude form, one bounce,
squared magnitude = received power; the supplement carries the steps from
the radar equation (mmIR / 3DPS style) to it. The scene's points enter with
a learned per-point gain (`background_point_gains`) and the person's with
a learned per-joint scattering gain (14, geometric mean pinned to 1,
skinned to the points; `joint_log_gain`, `80cf95b`).

### B.3 Slow time: the chirp-motion phase, the product 4DPS synthesises

The frame is 64 TDM loops of the TX; loop l of TX t samples at time
t_{l,t} (`signal_model.tdm_sample_times`). Each person point moves with
velocity v_p (B.4); its bistatic path rate to (t, r) is

    dL_p/dt (t, r) = v_p . u_{p,t} + v_p . u_{p,r},   u the unit vectors from TX/RX to the point
    (`surface_bistatic_path_rates`)

and its contribution to loop l carries the phase ramp
exp(+j 2 pi (dL_p/dt) t_{l,t} / lambda) (`SLOW_TIME_PHASE_SIGN`,
`assemble_bistatic_slow_time_profiles`; the static scene contributes to
every loop unchanged, i.e. only to DC after mean removal). The rendered
product is therefore the complex slow-time cube

    S[l, t, r, k],   (64 loops, TX, RX, 256 range bins) per frame,

exactly the shape of the recording's demultiplexed ADC after the range
FFT, for every physical TX-RX pair (no bin or channel dropped; the training
renders the azimuth sub-array, the product every TX). The range-Doppler map
per antenna pair is the windowed 64-point DFT over l after subtracting the
loop mean (`data.product_to_doppler`); range-azimuth is the windowed angle
FFT over the virtual array; the micro-Doppler spectrogram is the
person-masked power marginal over (range, angle) per Doppler bin. Stored
as float16 (re, im) pairs with one float32 scale per frame -- the GT
ADC's int16 I/Q footprint, every cell within 4.9e-4 of the float32 render
(`72b89b8`, `ed3c54a`).

Range migration and BSDF change within the 13.8 ms capture are neglected;
the carrier phase uses every physical TX time.

### B.4 What is learned (the parameters), what is fixed

| learned | shape | note |
|---|---|---|
| joint velocity field v_j(f) | (frames, 14, 3) in an orthonormal DCT temporal basis of 420/600 bands (3.5 Hz cap), initialised from the camera's joint velocities projected on the basis | `VelocityFieldModel`, `temporal-basis` (vf-clean `3f7ea44`) |
| point velocities | skinned from the joints by soft joint weights (temperature, top-k) | `soft_joint_weights_torch` |
| per-point person normals of the training frames | (visible points, 3) per frame, one fused parameter | `raw_normals_flat` (`f3b4720`) |
| person material (shared 6-vector), area scale | | `SharedHumanMaterial` |
| scene materials (N, 6), normals (quaternions), per-point gains | | `PointPrimitives` |
| absolute receiver gain per radar, joint scattering gains (14) | log-parameterised, lr 1e-2 | `80cf95b` |
| noise floor: expected power over (radar, Doppler, range), a log scale + 8 range-DCT coefficients on a measured envelope prior | inside the differentiable forward model, added to the MD prediction before the sqrt | `LearnedNoiseFloor` (vf-clean `358a0c1`, `74e86ee`) |
| fixed | positions (registered per frame), skinning weights, antenna patterns, array geometry, window, PSF width, visibility masks, coupling | |

The render adds the receiver nuisance model (frame- and chirp-correlated
receiver phase, AR(1) frame correlation fitted per sequence, slow-time
thermal nuisance, local additive nuisance) for the "noisy" product;
training uses the expectation (`signal_model` noise; vf-clean `68c5f53`,
`657fd73`, `5873b31`).

---

## C. The objective: three normalised L2 terms on linear magnitude

`velocity_field.two_term_loss` (the "third" term is the plane extension of
the RA term; `docs/LOSS_TERMS_AND_TARGETS.md` in vf-clean, `68e3e19`):

    L = L_RA,loop0 + L_RA,planes + lambda * L_MD,      lambda = 1

Each term is mean squared difference of linear magnitudes divided by the
measured target's mean square (detached, per frame and radar): plain L2
with a unit fixer, never scale-invariant (the prediction's level is never
in the denominator). No dB anywhere (vf-clean `af4e488`, the house rule).
All three targets come from one front end per frame and radar: decode,
TDM demux, Hann range FFT, azimuth FFT on the windowed virtual grid ->
the complex slow-time RA cube (64 loops x 256 range x angle).

| term | target | what it uniquely supervises |
|---|---|---|
| **loop-0 RA** | \|RA\| of loop 0, static clutter kept | the only path through which the static scene (materials, normals, gains, absolute level) gets gradient -- a constant lives in DC only |
| **63-plane RA** | slow-time mean removed, windowed 64-point DFT over loops, magnitude, the DC plane dropped (identically zero on both sides), (63, 244 range, angle) per frame and radar; its own divisor (a plane is a coherent sum over 64 loops) | the spatially resolved motion field -- "which cell at which velocity"; every one of the 64 chirps contributes (the DFT is unitary); restored the RA:MD gradient balance on shared parameters (55.8 -> 1.23 material, 832 -> 1.48 velocity) |
| **micro-Doppler (MD)** | from the same clutter-removed Doppler cube: power, masked to the person's (range, angle) support projected from the point cloud, summed over angle and the masked rows, the 13 motion bins, sqrt; the learned floor added on the prediction side | the velocity distribution, insensitive by construction to range/angle misregistration |

Trained on all frames of the sequence (`--train-all-frames`), 400 epochs
with a cosine schedule, uniform across sequences (never per sequence;
vf-clean `14b5307`, R040-R043). Held-out evaluation every 10 epochs.

### C.1 The k-SNR Doppler support of the plane term (the speed-up that changed what is rendered)

`--doppler-support-snr K` (vf-clean `7d4a8bb`, `93a5292`; production K =
1.0): the plane term does not render all 63 Doppler planes. **Per frame**,
from that frame's *measured* Doppler at the person's range rows, the
support is the set of Doppler bins whose power exceeds K times the local
noise floor (HuPR's median-over-loops floor at that range), unioned with
the 13 micro-Doppler bins the MD term needs unconditionally; the rendered
set per view is the union of the per-frame supports. Nothing is fitted to
a sequence: a fast-moving sequence renders more planes, the worst case is
all 63, and a bin above the noise floor can never be clipped. Plane count
is a linear multiplier on the Doppler splat (52% of the step at the time),
so at K = 2 seqs 38/102/256 went 63 -> 30/35/30 planes. The unrendered
planes are still supervised as zero (rendered floor) -- the alternative
(`--doppler-support-floor energy`, only bins carrying the person's energy)
was measured to smear the person over Doppler at 400 epochs because
unrendered bins go unsupervised, and stays off.

Companions on the same term: the plane targets in float16
(`--plane-target-dtype float16`, adopted after an A/B, `d6e5763`); a
per-sequence cell mask by k-SNR against a per-range floor was built and
measured (`--plane-mask-snr`, R046/R047: the criterion that transfers
across sequences where a dB threshold does not) but is not in the
production flags; random plane subsampling was rejected (`7516b3c`).

---

## D. Runtime: from 95 ms to 12.7 ms a cascade step, 400 epochs in ~7.5 min of chain

All numbers on RT-Pose seq 183 (299 frames, 144 training channels, 4
frames a step) unless noted; HuPR (8 channels a board, 24 frames a step) is
~5 min. Each change was gated numerically against the route it replaced
(the gate is stated); the 400-epoch metrics were re-checked at the end of
each batch. `docs/TRAINING_SPEED_PLAN{,_2,_3}.md`.

### D.1 The step's structure (vf-clean, 2026-08-17/18, "runtime groups A-F", "phases 3-5")

* **Fused CUDA kernels K1/K2**: K1 the gain-weighted static scatter of the
  scene, K2 the slow-time velocity splat of the person (`08df3a5`); K2b's
  forward as a gather, K1 as two fixed sparse maps (`968cb6b`, `7dbe6ec`).
* **Sync removal and batching**: 226 syncs a step removed, one render and
  one K2 launch per step, the loss chain batched (86 -> 3 syncs), Doppler-
  first FFT ordering (`c3cf98c`, `5dc06da`, `d9954d4`); the frame-batched
  step (`aab7783`).
* **Fixed shapes and CUDA-graph capture of the whole training step**: one
  fused normals parameter instead of 590, a fixed path count per step, no
  data-dependent compaction; captured 22.6 -> 10.8 min per run then
  (`f3b4720`, `c730f25`, `97e1417`); captured with the noise model
  (`9093a28`).
* **Cached splat geometry**: with positions fixed, everything that does not
  read normals or materials (distances, antenna gains, carrier phases, PSF
  weights, splat indices) is computed once per frame and the BSDF alone is
  re-evaluated each step (`c007b0f`; `build_row_geometry` per flat row for
  the captured step, `11287fd`).
* **Doppler support per frame** (C.1), per-unit Doppler weights, the fused
  velocity-source axis for evaluation (`4660c70`, `6d3a79d`).
* **Training sub-array**: only the TX forming the azimuth virtual array are
  rendered in training (HuPR TX1/TX3; cascade TX4-12), since the beamformer
  drops the elevated TX before the angle FFT: identical losses (five
  figures), 41.5 -> 28.4 ms on HuPR 256 (`56307f7` R7). Products keep all TX.

### D.2 The Doppler-plane route (4DPS, 2026-09-12/13)

* **Closed-form Doppler-plane amplitudes**: the Hann Doppler PSF evaluated
  at the path's fractional Doppler bin instead of a 64-loop DFT per path
  (7e-7 of the float64 reference against the loop sum's 5e-6), and the
  plane gradient read transposed: 94.4 -> 54.2 ms (`cb4ffe0`).
* **7-tap Doppler PSF**: each path deposits into the 7 planes around its
  own bin, as the range PSF is cut at 15 (7.7e-3 of the exact planes,
  indistinguishable at 400 epochs) (`99e17fb`, plan 2 s3.5). A NUFFT/
  Kaiser-Bessel spreading alternative was measured worse per deposit and
  withdrawn (plan 3 s3.4).
* **The sorted Doppler route**: the person's paths sorted once per frame by
  (unit, channel, range bin); the kernels synthesise every contribution
  from the BSDF carrier and the cached range bins, no binning and no
  contribution arrays per step (`6acbeff`); the scene's backgrounds for the
  batch from one fused kernel pair (`8143a0a`); the learned floor inside
  the compiled plane term (`1c23487`).
* **Compiled BSDF steps 1-3 and the amplitude on the cached path**; the
  angle FFT's input from the plane cube in one compiled gather (bitwise)
  (`3404289`, `1e231c4`).
* **BSDF step-4 backward without global atomics**: one warp per point
  (3.5e-7 of the atomic route): 2 x 1.27 -> 0.49 + 0.31 ms (`5f2ee4d`).
* **Fused Adam** (4.9e-8 of the eager update; 1.10 -> 0.89 ms) and
  evaluation every 10 epochs in chunks of 4 (identical to 2e-9) (`60dd75e`).
* **Targets built on the device in blocks**, no cache file (within 2.4e-7
  of the numpy build; start-up 90 -> 26 s) (`81626e8`).
* `--scene-update-every k` was built and measured worse at k = 4; off.

Result (plan 3 s4): 15.3 -> 12.8 ms a step measured; the velocity stage
747 -> 503 s; the chain 527 s at 400 epochs; the metrics unchanged within
their spread. Chain start to end today: RT-Pose ~9.5 min, HuPR ~5.2 min
(sweep manifests, 2026-09-14).

### D.3 Outside the trainer

* The render's slow-time assembly and nuisance model on the CUDA path
  (2.9x), products written off the critical path (`ecc4b90`, `4178ea3`).
* Preprocessing on the GPU: grid-culled ray casting, voxel downsample,
  PCA normals, batched person ranges and masks, the in-memory ADC reader,
  the light stages in-process; 16-18 s a sequence (`a47250b`, `5117416`).
* The person material trainer GPU-bound; motion detection batched on the
  GPU (52 -> 6.6 s) (`f72b256`, `efdf1b3`).
* GVHMR in-chain, half precision, from frames in memory, overlapped:
  marginal +30 s (`0772574`).
* Float16 product with per-frame scale, the GT's size (`72b89b8`).

---

## E. Evaluation (for section 4; definitions in `docs/METRICS.md`)

Linear amplitude everywhere, never dB (the same renders score 0.96 on a dB
spectrogram and 0.79 on linear power). Range-azimuth Pearson with static
clutter suppressed (passes a permutation null), SSIM, NRMSE, PSNR;
range-Doppler occupied Pearson under a CFAR mask from the real product
(frames with >= 8 occupied cells); micro-Doppler spectrogram Pearson and
energy/centroid/spread correlations on the motion subset (pose speed >=
0.15 m/s, pre-registered); micro-Doppler topology by clDice under
hysteresis masks built independently per signature. A shuffle/permutation
control precedes any claim (`7a4af40`, `99de797`, `cca4e37`).

## F. Not in the paper, but in the history (so they are not re-proposed)

Rejected or withdrawn after measurement: Pearson RA loss (correlation is
affine-invariant, level lost); the depth cloud as person geometry; the SMPL
fitter (replaced by GVHMR, `b3c962d`); post-hoc person amplitude refit;
the diffuse floor level fit; moment-matched nuisance scale; random plane
subsampling; one channel per grid slot; a sparser aperture; fewer person
points; per-point Doppler rates; splatting directly into RA with an angular
PSF; the NUFFT spreading kernel; `--scene-update-every`; the energy Doppler
support; the camera-FOV scale and azimuth rotation in the scene alignment;
per-sequence epoch counts.
