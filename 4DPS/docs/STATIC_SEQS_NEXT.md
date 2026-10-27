# The near-static RT-Pose sequences: what is established, and the next steps (2026-09-20)

Companion to docs/STATIC_SEQS_PLAN.md (the diagnosis of 2026-09-19). Every number below is the five real-trained
HRRadarPose seeds under the dataset's labels (`gate_node.py`), the chain's fidelity suite (`fid_node.py`), or the
person/competitor contrast in the model's polar RA (`contrast_node.py`: person = max |RA| within +-0.4 m / +-8 deg of the
label, competitor = max elsewhere in the model ROI; below 1 the model picks the other scatterer). Artifacts:
`mmSD/artifacts/static_level/` (the three-column videos recording | before | after, the gate and contrast JSONs).

## 1. What is established

**The failure is the rendered person's level in the model's domain, and nothing else.** Render-only test (`lvl.sh`): base2's
trained state, no retraining, the person's `areas` slot scaled so the person window's PEAK |RA| matches the recording's
(x2.4 / x3.3 / x1.1 in amplitude on 176 / 108 / 183):

| seq | base2 (before) | after: level = peak | contrast before -> after | fidelity |
|---|---|---|---|---|
| 176 | 117.8 +- 69.8 cm, 32 % of frames > 1 m | **36.9 +- 14.5 cm, 6 %** | 0.52 -> 1.22 | RA 0.78 -> 0.77, RD power ratio 0.53 -> 0.89 |
| 108 | 298.7 +- 102.6, 56 % | **30.4 +- 4.3, 1 %** | 0.44 -> 1.10 | RA 0.84 -> 0.84, RD power ratio 0.39 -> 0.93 |
| 183 | 25.0 +- 2.9, 0 % | 24.5 +- 3.0, 0 % | 1.66 -> 1.81 | unchanged |

Matching the window SUM instead (x1.82 / 1.59 / 1.37) is not enough (176 60.5, 108 172.2): the render's person is broader
than the real one (108: 36 cells above half-peak against 10), so the peak, which is what the model's competitor test sees,
is the quantity to match. The real reference is 13.1 / 10.4 / 4.6 cm.

**How the level is set today, and why the trainer always lowers it.** The person stage (1 epoch, 38 optimiser steps) sets
the level through its area scalar (0.0425 on 176; the material stays at its ITU-concrete initialisation). The velocity
stage then lowers it, on the near-static sequences only, through whichever level parameter is free -- measured one by one:

| candidate | held | the trainer used instead | person level (176 / 108 / 183, x recording peak) |
|---|---|---|---|
| base2 | -- | area scale x joint gains (176: hips 0.09, elbows 5-11) | 0.38 / 0.27 / 0.91 |
| frz | area scale, joint gains | coherent fraction (0.055 vs 0.47 on 183), receiver gain (34 / 8.9 vs 88 / 63) | 0.39 / 0.35 / 0.81 |
| physp | + slot = physical geometry | receiver gain (8x down), coherent fraction | 0.20 / 0.19 / 0.39 |
| physp_fm | + material frozen | receiver gain alone (4.7 -> 0.9, 10 -> 0.6) | 0.13 / 0.14 / 0.36 |
| physp_fm1 | + gain fixed at 1 | nothing left -- but the person stage never reached the level without its area knob | 0.13 / 0.16 / 0.18 |
| physp_fmg | + gain frozen at the first-frames value | 108's level came out at 0.92 by accident -> **gate 18.8 +- 1.9 cm** | 0.28 / 0.92 / 0.29 |
| physp_e20 | + person stage 20 epochs | its material moves (coherent 0.73) but cannot reach the level at a physical slot | 0.17 / 0.17 / 0.25 |

The velocity objective's preference for a dim, broad body (the gradient probe of the plan, section 2) is real, but it only
matters because the level is free; with the level right, the same objective's render is fine for the downstream (108 at
0.92 -> 18.8 cm; the render-only test above).

**Rejected on this branch, with numbers:** P1, the DC-plane term (`--static-term`, default off: gate worse, moving-band
fidelity down, the level knobs untouched); the three-term objective (rae3: suite regressed, reverted); the scene's physical
solid angle (phys: the near-field points at 1-2 m get 50-300x the median weight and the scene fit collapses, RA 0.78 -> 0.51).
The person's physical slot (`--person-solid-angle-weight physical`) is correct geometry but useless while the level is not
anchored to the measurement; it stays available.

## 1b. Step 1 implemented and measured (2026-09-20, commits af435bd .. b84c89b)

The level fit, in the end, has to be done **after training, on the state that ships, on the downstream's quantity**:
- the person stage's own calibration (its render of every vertex, a dB ratio) does not transfer: lvlfit 0.42-0.54x;
- a loop-0 peak at init (lvlfit2) ships at 0.44-0.57x: the trained velocity field and normals defocus a standing body's
  beamformed peak ~1.8x (TDM MIMO: motion between the TX slots decorrelates the phases across TX);
- a magnitude sum over the windowed mean-removed planes reads 1.55x on a render whose energy is 0.25x (it rewards Doppler
  spread and drops DC); the downstream's quantity is the plain 64-point FFT, DC kept, |RAE| of every TX summed over
  Doppler and elevation, unwindowed, peak over the person mask -- `--person-level-calibration peak-doppler
  --person-level-calibration-when end` (velocity_field.py), one 64-loop render of the person per calibration frame.

With that calibration and everything level-related frozen (`lvlfit4/5`: area frozen, joint gains 1, material frozen,
receiver gain = the person stage's scene calibration, H/V delta from the person stage kept), five real-trained seeds,
dataset labels, base2 in brackets:

| seq | gate MRPE (cm) | frames > 1 m | contrast (base2 -> now) |
|---|---|---|---|
| 176 | **35.9 +- 16.7** (117.8) | 4 % | 0.52 -> 1.17 |
| 168 | **26.3 +- 6.2** (146.0) | 2 % | 0.68 -> 2.09 |
| 108 | **23.0 +- 6.4** (298.7) | 1 % | 0.44 -> 1.56 |
| 164 | 144.0 +- 73.7 (158.0) | 27 % | 0.53 -> 0.70: the person is at 0.72x but the scaffold's ghost of the person at its start position is a scatterer at 1.27x the recording's competitor -- a scaffold item, not a level item |
| 183 | 31.7 +- 7.1 (25.0) | 1 % | 1.66 -> 1.99 |
| 186 | 19.5 +- 4.6 (15.9) | 0 % | 1.55 -> 2.23 |
| 153 | 20.6 +- 4.1 (19.4) | 1 % | 1.96 -> 1.42 |

Three of the four near-static failures are fixed at the render-only test's level. The cost of the *frozen* part of that
recipe is the moving band: MD energy r down 0.07-0.16 on the walking RT-Pose sequences, HuPR 256 -0.01..-0.04 across
the suite, HuPR 102's vertical radar -0.02..-0.09, and HuPR 40's micro-Doppler collapsed (spectrogram r 0.89 -> 0.42,
energy r 0.93 -> 0.15): with the material and the gain frozen the velocity stage has no way to set the moving return's
level. That fails the section-1 rule, so the accepted form is the calibration ALONE on the unchanged trainer
(`lvlcal` = base2 + `--person-level-calibration peak-doppler`, running on all 11 at the time of writing): the moving
band fits as before, and the static level is corrected at the end. Its numbers decide the commit.

Found on the way and recorded: the person stage's material barely moves in one epoch (the area scalar sets its level);
the per-radar `--absolute-gain-<radar>` flags are dataset-specific (`--absolute-gain` is the radar-agnostic spelling);
`--person-radar-gain-delta off` discards the calibrated H/V balance (`frozen` keeps it); a scene-dominated person mask
(164's ghost) makes the person stage's calibration collapse (guarded, keeps the fit's level).

## 1c. lvlcal, the scaffold cut, and where the competitor really is (2026-09-20, 12018fe .. 25ecda3)

**The gate from here on is the five mesh-label seeds** (`results_gvhmr`, the retrain done 2026-09-20; `GATE_CKPT`, labels
Train_gvhmr.json). base2 re-gated on them: 176 161.7, 168 79.9, 108 229.6, 164 178.7, 183 12.7, 186 11.3, 153 22.9 cm
(real 16.5 / 13.3 / 13.3 / 10.5 / 3.4 / 3.8 / 10.4). The dataset-label seeds are kept in the logs (`_ds`) for comparison.

**lvlcal = base2's trainer unchanged + `--person-level-calibration peak-doppler` at the end.** Calibration factors: the
near-static sequences x2.0-2.6 in amplitude (their trained person at 0.44-0.54x the recording's peak), the walking ones
x1.03-1.11; HuPR 38 x2.08, HuPR 40 x2.19, 102 / 256 x1.03. Gate (mesh seeds): **176 36.6 +- 14, 168 26.9 +- 9, 108 27.9 +-
16**, 183 10.6, 186 10.0, 164 194 (see below), 153 32.8 +- 27 (worse: the calibration read the mask's peak as 1.07x and
LOWERED the person x0.92 -- the mask's peak there is a scene scatterer, not the person; the same on 164, x1.24 while the
person is at 0.69x: the calibration must read the person's own peak cell, not the mask's, next commit). Fidelity vs base2:
every RT-Pose sequence and HuPR 102 / 256 within +-0.01-0.02 on every metric (168 MDe +0.14); **the two near-static HuPR
sequences regress on the moving band: 38 RAc 0.64 -> 0.49, MDe 0.946 -> 0.916, RA 0.950 -> 0.928; 40 RAc 0.24 -> 0.11**,
while their RD power ratio goes 0.58 -> 1.09 / 0.48 -> 1.01. The x2.1 brightens the person's moving band, and on a
standing body that band is the trainer's fiction (next paragraph), so the level fix exposes it. Recorded as a regression
on those metrics; not hidden in the spread.

**The trained velocity field on a standing body is fiction, but replacing it does not help.** Learned joint speeds
(median over frames, hips): 176 0.82, 168 0.67, 108 0.94 m/s against the mesh's 0.06 / 0.02 / 0.02 m/s (walking 183 / 186 /
153: 0.75 / 0.81 / 1.06 learned vs 0.46 / 0.56 / 0.77). The render-only probe `vel.sh` (learned field -> the mesh's
kinematic velocities, level as calibrated): gate WORSE (dataset seeds 176 58 -> 71, 108 43 -> 78, 168 92 -> 140), micro-
Doppler collapsed (176 MDs 0.95 -> 0.71, 168 MDe 0.85 -> 0.01, RD 0.61 -> 0.40), the person's DC share unchanged (dc 0.43 ->
0.49). The field reproduces the recording's micro-Doppler statistics on these sequences with velocities the body does not
have; the DC deficit is not its doing. Left as is; it is P2's subject (the moving-band terms reward spread).

**The scaffold ghost (2b) is real, is the annotation's fault, is fixed, and was not the competitor.** At the scaffold's 8
sampled LiDAR frames the dataset annotation's root sits 0.45-0.6 m lateral of the registered mesh (108: 0.65 vs 0.06 m;
168: -0.73 vs -1.18; 164 frames 194-239: -0.68 vs -1.09; the mesh-derived labels agree with the mesh to ~7 cm), so the
0.6 m cut-out column left the person's own LiDAR / depth points in the scene (108: ~300 scaffold points inside the torso at
every sampled frame; 164: ~1500 at the end position). Commit c0f9241: `reconstruction scaffold-cut` (points within 0.25 m of
the registered person's surface in any frame, the floor under the feet kept) in `person_stack` once the registered person
exists, then the background fitted again (`chain.fit_background`, 1 s); both datasets, `SCAFFOLD_CUT=off` for A/B. Removed:
164 8437, 183 5763, 186 5701, 153 1847, 168 979, 108 518, 176 46 points; HuPR 38 352, 102 1439, 40 91. Figures
`artifacts/static_cut/cut_{164,108,168,183}.png`. Candidate `cut` = lvlcal + the cut, all 11: gate (mesh seeds) 176 40.4,
168 26.0, 108 41.6 +- 50 (one seed), 164 211, 183 10.7, 186 10.6, 153 33.4 -- **no change within the run-to-run noise**;
fidelity within noise (164 RA 0.899 -> 0.879, HuPR 38 as lvlcal). Kept because it removes a verified artefact, not because
it moved a number.

**Where the competitor is (contrast_node.py now says): the far wall, and it is the background fit's.** In the render the
strongest static cell outside the person is at x 6.4-6.8 m (164 / 108 / 153: y -2.2..-5 m, the corner; 176: y 0.2 m,
straight ahead) in 95-100 % of frames, 1.1-3.5x the recording at that cell (windowed view: 1.3 / 6.3x); the recording's own
strongest static cell is at (1.4, 2.1) m, 56 deg off boresight -- outside the LiDAR / camera field, absent from the
scaffold (0.05x). The static scene's render/recording ratio by range (person window excluded): 0.67-0.9 at 1-4 m,
1.1-2.0 at 6-8 m, on 164 / 108 / 176 / 153; on the walking 183 it is 0.56-0.92 everywhere. **At the velocity stage's initial
state (`ep0`, 0 epochs) the far cell is already 3.5x (164) / 4.7x (176)**: the slope is set by the background fit and the
propagation model, and 400 epochs of per-point materials leave it at 1.1-3.5x. Render-only probe `far.sh` (scene points
beyond 5.5 m -- 17-18k of the 20k active points -- at x0.5 / x0.33 amplitude): gate 164 211 -> 138 -> 98, 176 40 -> 27 -> 27,
but RA Pearson 164 0.879 -> 0.858 -> 0.823, 176 0.787 -> 0.756 -> 0.726: the whole far band is not what is wrong, the
wall's specular flash cells are. That is the next item (section 2, step 1c).

**1c measured and rejected (cal2 = cut + the calibration at the person's own peak cell, all 11).** The walking sequences
read 1.23-1.27x and were dimmed x0.78-0.81 (183 / 186 person 0.69 / 0.73x, 153 33 -> 53 cm); 176 read 0.50x, x2.16, and
still gated 107 +- 122 (its person at 0.53x after: this run's trained person was dimmer, and the recording's peak within
+-1 cell of the render's own peak is not the recording's person peak -- the render's person is broader and sits off by
more than a cell). Gate (mesh seeds): 176 107, 168 36, 108 54, 164 192, 183 11.3, 186 11.0, 153 53. The mask's peak
stays the default (`--person-level-calibration-cell mask`); 153's scene-in-mask case is left for the shape work (P2).

**1d resolved: the far wall's flash is a VIEW mismatch -- the trainer's Hann-tapered beamformer against the downstream's
untapered one.** The gradient-probe hook had been dead since `load_trained_state` was introduced (its body sat in an
unreachable inner function; rejoined in ab8c143) -- the probe now also dumps the trainer's own loop-0 |RA| and |RAE| views.
In the trainer's own view (windowed, 86 x 8), cut/176's far wall at 6-7 m reads 0.98-1.12x the target, elevation
profiles alike: the objective sees nothing to fix. The product cubes put through the trainer's transform
(`channels_to_grid2d`, loop 0, elevation-summed) with the azimuth Hann: far bands 0.94x, the wall's peak 0.97x; the same
channels WITHOUT the taper: far bands 1.7x, **the render's peak 9.8x the recording at that cell** (2-3 m: 0.76x). A
rendered wall is a coherent extended reflector (dense points, per-point phase): its untapered array response is a
narrow flash with the unwindowed array's sidelobe / grating structure; the real wall's return is spread. HuPR's released
preprocessing and the HRRadarPose adapter beamform untapered, so that is the view the downstream consumes and the view
the objective never sees. Commit de86794: `--array-window {hann,none}` on the velocity trainer (its beamformer, targets
and predictions alike; range and Doppler keep the pipeline's Hann; the render, the products and the evaluation keep the
pipeline's window, so the fidelity numbers stay comparable). Candidate `nowin` = cal2 + `--array-window none`, all 11.

**nowin measured (lvlcal + `--array-window none`, mask cell; all 11, 2026-09-20).** Gate on the mesh-label seeds, base2 ->
lvlcal -> nowin (real): **176 162 -> 37 -> 27.7 +- 16 (16.5); 168 80 -> 27 -> 12.7 +- 1.4 (13.3); 108 230 -> 28 -> 16.7 +- 8.5
(13.3)**; 164 179 -> 194 -> 121 +- 22 (10.5); 183 12.7 -> 10.6 -> 9.6 (3.4); 186 11.3 -> 10.0 -> 10.1 (3.8); 153 22.9 -> 32.8 ->
13.8 +- 2.0 (10.4). Contrast: 168 / 176 / 108 frames < 1 at 0 / 8 / 25 % (base2 98-100 %). Fidelity, the suite's own (Hann)
view vs base2: RA -0.02..-0.06 on every RT-Pose sequence (176 0.78 -> 0.73, 183 0.90 -> 0.84), RAc -0.05..-0.14 (183 0.66 ->
0.53), RD -0.01..-0.05, MDs / MDe +0.01..+0.04; HuPR 256 RA -0.002 / -0.015, 102 -0.003 / -0.015, 38 as lvlcal (RAc 0.64 ->
0.42, MDe 0.95 -> 0.93), 40 vert MDs 0.84 -> 0.79. **The same renders scored by the same suite in the consumers' untapered
view (`reeval.sh`, MM4DPS_ARRAY_WINDOW=none on both sides): lvlcal RA 0.57 / 0.70 / 0.81 / 0.76 on 176 / 168 / 183 / 153 ->
nowin 0.78 / 0.88 / 0.86 / 0.89; RAc 0.08 / 0.21 / 0.52 / 0.49 -> 0.10 / 0.23 / 0.59 / 0.54; RD +0.02..+0.06.** The two views
disagree by construction: the trainer can fit one; the Hann suite measures the other. The downstream reads the untapered
one, and the gate agrees with it (168 at the real model's own error). Decision for the user: which view section 1's
"no regression" is measured in. Recommendation: `--array-window none` becomes the trainer's default and the fidelity
suite reports the consumers' view beside the Hann one (both sides, one env), so the rule is checked where the downstream
looks.

**Accepted (user, 2026-09-20): `--array-window none` is the trainer's default (24cfce1) and the fidelity suite scores both
views** -- the renderer writes `<radar>_<side>_first_loop_ra_untapered_complex` beside the tapered product, evaluation.py adds
`range_azimuth_untapered` / `range_azimuth_suppressed_untapered`, fid_node prints them ("untapered RA / RAc"). On nowin/176:
tapered RA 0.733, untapered 0.778 (lvlcal 0.773 / 0.574).

**The Doppler-domain picture of a standing body (skirt.py, floor.py, nofloor.sh on nowin 176 / 108).** Plain 64-point FFT,
|X_d| / |X_0| at the person's cell, recording: +-1 0.15 / 0.06, +-2 0.07 / 0.03, +-8 0.02 / 0.01 (sum over |d| >= 3: 0.77 /
0.34); render: +-1 0.46-0.55, +-2 0.25-0.31, +-8 0.045-0.08 (sum 2.0 / 3.6) -- the same shape 2-4x stronger: the learned
velocity field's micro-motion of the standing body is 2-4x the real sway, so after the level calibration (which matches
the Doppler SUM) the DC sits at 0.2-0.5x and the moving band at 1.0-1.2x (contrast_node's dc / moving split). That is P2's
subject and the remaining lever on 176 / 108 (28 / 17 cm vs 16.5 / 13.3). The drawn receiver floor is not it: without it
(`MM4DPS_RECEIVER_FLOOR=off`) the person's skirt stays (+-1 0.29 / 0.33); the floor's level is 1.4x the recording's (1.4e6 vs
1.0e6 per cell-bin, 0.0013 of the wall's DC) and a static cell of the render is pure DC without it. The recording's static
cells carry a smooth within-frame drift (the wall: |x_l| 1.18 -> 1.00 over the frame, phase 0.07 rad) the render has not;
it is 0.5 % at +-1 and not the person's skirt.

**pcal measured and rejected (nowin + `--person-level-calibration-target person`, all 11).** The person-free static estimate
(the 20th percentile per cell over the frames) is the person itself where the person stands still, so the target collapses
on exactly the sequences that matter: 176 x1.27 (x2.2 with the total), 168 x1.20, 108 x1.33; person 0.38-0.50x after; gate
176 35.1, 168 19.0, 108 31.9, 164 140, 183 9.8, 186 10.0, 153 13.3 (nowin 27.7 / 12.7 / 16.7 / 121 / 9.6 / 10.1 / 13.8).
The total stays the default. The walking sequences' scene-in-mask reading is a smaller error than any estimator tried.

**vres measured (nowin + `--velocity-residual-bound 0.1`: the field = the mesh's kinematics x a learned per-joint scale + a
residual bounded by 0.1 m/s per component; ef38169), all 11.** Gate (mesh seeds): **176 20.4 +- 4.4 (real 16.5), 168 10.9
(13.3), 108 9.5 (13.3), 164 68.5 (from 121), 183 10.2, 186 10.0, 153 14.6**; the standing sequences' calibration factors
fell to x1.04-1.14 (the level was right before the calibration once the body stopped spreading). Fidelity vs nowin: RA
+0.01..+0.02 both views, MDe +0.03 on 176 / 186, HuPR 38 recovers most of the level fix's cost (RAc 0.42 -> 0.53, MDe 0.93
-> 0.965, vert RAc 0.37 -> 0.48); but the walking sequences' moving band pinches: RD 153 0.647 -> 0.556, 183 / 186 -0.02 /
-0.03, MDs 153 -0.06, HuPR 102 RD -0.01, MDs -0.004 / -0.005; 168 MDc 0.885 -> 0.311 (a standing body's Doppler centroid is
noise). The constant bound is right for a standing body and too tight for gait, where the mesh's smoothing removes the
swing peaks and a per-joint scale cannot restore a waveform. Hence `--velocity-residual-relative beta` (d7ebbf3): the bound
b + beta |v_kin| per joint and frame -- the mesh's velocity error as a constant plus a share of the motion it smooths.
Candidate vrel (b 0.1, beta 0.75) running.

**vrel measured (b 0.1 + 0.75 |v_kin|; d7ebbf3), all 11.** Gate (mesh seeds): 176 20.0, 168 10.9, 108 10.7, 164 84.4 +- 29,
183 10.2, 186 9.7, 153 13.6 -- vres's gains kept; the gait pinch gone: RD 153 0.674 (vres 0.556, nowin 0.647), 183 0.695
(0.649 / 0.673), 186 0.719 (0.658 / 0.691), MDs 153 0.941 (0.890), 176 MDs 0.974 MDe 0.930; HuPR 256 neutral, HuPR 40 hori MDs
0.864 (nowin 0.898, vres 0.890) the one metric down. **vrel is the recipe from here** (the flags become defaults once 164 is
settled, one commit).

**164's residual, located (blocks164.py, cell.py, peaks.py).** Per 15-frame block under vres the error is 9-25 cm (the real
model's own) everywhere except frames 30-90: 54 / 214 / 352 / 97 cm, when the person walks to and stands at the far corner
(range 7.3-7.7 m, lateral +2.0; the ROI ends at 8.02 m). There the render's strongest static cell, at (6.5, -2.3) in every
frame, is NOT over-bright: at each side's own peak the render is 0.86x the recording (DC 0.85, |d| >= 3 1.20); it is DISPLACED:
the render's corner peaks at (6.51, -2.26), the recording's at (6.43, -2.48), 2 deg / 0.23 m apart -- and on 153, the same
room, the render's corner sits exactly on the recording's (6.43, -2.48). A yaw of the scaffold is not it (the occupancy-map
correlation is flat in yaw, the matched static peaks show no common rotation: median offsets -0.5 / +0.5 / +0.4 deg on 164 /
176 / 183). 164's scaffold differs from 153's at that corner because two of its eight sampled LiDAR frames (47, 70) have
the person standing there, and the annotation's 0.6 m column -- 0.4 m off the person -- cut the corner's own points and
left the person's. Commit ae9c29b: the column cut is off (the registered mesh cuts the person, c0f9241); the node's env
has no Depth-Anything, so the scaffolds are built on the workstation (`build_scaffolds_ws.sh`) and taken by the runner
(`SCAFFOLD_REBUILD=1 SCAFFOLD_SRC=...`). Candidate `nocol` = vrel + the column-free scaffold (164: 18991 person points
removed by the mesh cut against 8437 before) running on the 7 RT-Pose sequences.

**nocol measured (vrel + the column-free scaffold; ae9c29b), the 7 RT-Pose sequences.** Gate: **176 17.8 +- 2.4 (real 16.5),
168 10.3, 108 12.1, 183 9.9, 186 9.5, 153 13.2** -- the best yet on every sequence but 164 (78.6; frames 45-75 still 292 /
403 cm). The corner was never beside the person: the map's lateral axis is -y_label, so the corner at map (6.5, -2.3) is
4.3 m from the person at map lateral +2.0 (an earlier mis-mapping in 1c); the column-free scaffold is kept on its own
merits (the annotation column removed scene and left the person; the numbers above).

**What 164's frames 45-75 are (crop.py, the person's cell at its real map position).** The person SITS still at 7.5 m
(pelvis height -0.69 m, four sit / stand transitions at frames 20-30, 90-100, 175-185, 275-285; the geometry is smooth:
mesh / joints / labels step <= 14 cm per frame -- the "0.5-1 m jitter" reported on seq164_geometry.mp4 is not in any
source I can measure, see the reply of 2026-09-20 09:10). In those frames all five seeds put the root at random far cells:
the render shows them no person. Frame 60, recording vs render: Doppler-summed peak 3.4e8 vs 2.9e8 (0.86x) but **DC 2.7e8 vs
1.5e8 (0.54x) and the moving band 0.7e8 vs 1.4e8 (2.1x)** -- a sitting body's return is DC in the recording and half-moving
in the render: the residual bound of 0.1 m/s (vres / vrel) still lets a sitting body's energy into |d| >= 3. Candidate
`vb03` (bound 0.03 m/s, the postural-sway scale; relative 0.75 kept for gait) running on all 11.

**A renderer bug found on the way, fixed (2835559):** since `--array-window none` the trainer measured its floor prior in
the untapered beamformer while the renderer calibrated the drawn floor in the Hann one; white channel noise reads 2.67x
more power untapered, so every render since nowin drew its floor 2.4x too strong in power (1.55x amplitude; chnoise.py:
render / recording per channel 2.42 -> 0.83 after). Re-rendered nocol (`nocolf`): gate unchanged within noise (176 18.4,
168 10.0, 108 12.0, 164 75.9, 183 10.6, 186 10.0, 153 12.2), RAc +0.02-0.03 on every sequence. `--noise-model measured`
(a714b80, the floor fixed at the recording's own) was tried as nmeas before the bug was found: no gain (164 96.7, others
within noise); the learned floor stays the default.

**164 narrowed to the sitting frames' level, and the level to the calibration's VIEW (2026-09-20 10:00-11:00).**
- vb03 (residual bound 0.03 m/s): 164 71, frames 45-75 278 / 305; others as nocol. Not the DC / moving split alone.
- The person sits on a chair: the recording's cell at (7.25, +1.63) carries 1.2e8 when the person is away and 2.7e8 sitting;
  the scaffold-cut SWEEP removed the chair (0 points at 0.2-1.2 m after it). Commit 4baf8b3: the LiDAR scaffold cuts the person
  out of each sampled frame by the registered mesh of THAT frame (`--person-sequence-dir`, `MM4DPS_SCAFFOLD_PERSON_SEQUENCES`;
  the chain skips the sweep); the chair survives (528 points). pfcut: chair cell 0.67x the recording's; 164 86 cm, the
  sitting frames unchanged (267 / 303) -- the chair was not it either; the rest 176 18.9 / 168 11.1 / 108 16.4 / 183 10.6 /
  186 10.0 / 153 12.4.
- swap_box_node.py (the person's box of the input tensor swapped, seeds 0 / 4, frames 45-75): real 14 / 36 cm; render 199;
  render scene + REAL person box 68 / 82; real scene + render person 50 / 228; **render, person box x2: 70 / 115**. The rendered
  sitting person is about half the level the model needs, and the rendered scene around it holds the rest.
- The person's window peak ratio render / recording (the adapter's view, +-0.6 m of the label) is flat in range within each
  sequence but sits at 0.64-0.72 on 153 / 164 against 0.84-0.98 on 183 / 186 -- while the calibration reads 1.0. Two causes
  found: (a) the trainer's person mask includes the MIRROR azimuth (`person_range_angle_mask_torch include_mirrored_angle`),
  and on 164 / 153 scene on the mirror side (the corner) is the mask's peak -> `--person-level-calibration-cell side`
  (ae73d39): on 164 the calibration then reads 1.00 on the person's own side -- and the adapter's view still says 0.73;
  (b) **the two transforms disagree**: on the same cubes and the same window the trainer's beamformer (channels_to_grid2d,
  the mm4dps slots averaged) reads the render / recording person peak at 0.97-1.12 where the adapter's (the TI table, one
  channel per slot) reads 0.63-0.93 -- 164 0.75 vs 1.12, 153 0.63 vs 0.97, 183 0.93 vs 1.03; azimuth / elevation FFT
  padding does not explain it (twoviews.py). Commit d1463c3: `--person-level-calibration-view downstream` reads both sides
  in the consumer's own transform (the adapter's polar, the pelvis window). Candidate `dsv` running on all 11.
- No elevation / pitch offset between render and recording (pitch.py: the static map's elevation centroids agree within
  0.5 bin at every range); the sitting person 2 bins lower in the render is the geometry at the array's resolution.

**dsv measured (the calibration in the consumer's transform; d1463c3), all 11.** The person reads 0.96-1.05x in the
adapter's view on every sequence (calibration factors 164 x1.34, 153 x1.37, 176 x1.69, 108 x1.82, 168 x1.35, 183 x0.97,
186 x1.07). Gate (mesh seeds): **176 17.8, 168 10.4, 108 9.1, 183 10.9, 186 9.6, 153 12.5, 164 56.7** (from 76; the sitting
frames 199 / 243, frames 120-299 at 40). Fidelity: RAc up on every sequence (176 0.13, 168 0.35, 164 0.54, 153 0.57 -- the
best of the branch), RA / RD / MD within noise. **dsv is the recipe.** In the sitting frames the render's person is still
DC 0.56-0.72x / moving 1.4-4x the recording's (the recording's sitting person is DC only; its moving band is the floor),
and the chair reads 0.42x. Candidate `vb0` (`--velocity-residual-bound 0`: the residual purely proportional to the
joint's kinematic speed, a still joint stays still) running on all 11.

**vb0 measured (dsv + `--velocity-residual-bound 0`: the residual purely proportional to the joint's kinematic speed;
664b1c0), all 11.** The person's DC reads 0.86-1.13x on every sequence. Gate: **164 41.5 +- 7.5** (the sitting frames 120 /
176, frames 120-299 at 27), 176 18.5, 168 11.5, 108 9.6, 183 10.7, 186 9.7, 153 12.6. Cost: the standing sequences' sway is
gone from the render and their MD energy correlation drops (176 MDe 0.945 -> 0.907, 168 0.902 -> 0.820, 153 MDs -0.01);
everything else within noise (183 RAc +0.01). The recording's standing person does sway (its +-1 Doppler bin holds 0.15 of
the DC), so a zero constant bound is physically short for standing; postural sway is 1-3 cm/s, breathing ~1 cm/s --
candidate `vb015` (0.015 m/s) running to sit between vb0 and dsv on both counts. The chair the person sits on renders at
0.42x the recording (a strong dihedral in reality; a dim point cloud in the render, and its cell weighs 1 % of the wall's in
the loop-0 term) -- the remaining scene-side item of the sitting frames (the swap probe: render scene + real person 68 / 82).

**vb015 measured (bound 0.015 m/s), all 11.** 164 52.4 (the sitting frames back at 222 / 243: the seated body spreads at
0.015 as at 0.03), 176 18.3, 168 10.8, 108 9.4, 183 10.4, 186 9.4, 153 12.7; MDe 176 0.916, 168 0.955 (the sway kept).
HuPR under dsv / vb0 / vb015: seq 38 back at base2 or above (RAc 0.60 / 0.54 / 0.59 vs base2 0.64; RD 0.78 vs 0.77; MDs
0.98 vs 0.97; MDe 0.97 vs 0.95; vert RAc 0.52 / 0.44 / 0.51 vs 0.43), seq 102 unchanged -- the level fix's HuPR cost of
section 1c is gone with the consumer-view calibration. The three bounds: 0.03 -> 164 56.7, 0.015 -> 52.4, 0 -> 41.5; the
standing MDe 0.945 / 0.916 / 0.907 (176), 0.902 / 0.955 / 0.820 (168) -- inside the run-to-run band (168's MDe ran
0.82-0.955 across identical recipes). A seated trunk is supported and does not sway; a standing one does: candidate `vbh`
(5f5c4a6: `--velocity-residual-per-height 0.015`, the bound proportional to the pelvis's height above the support, an
inverted pendulum -- standing ~0.014 m/s, seated ~0.008, constant 0) running on all 11.

**vbh measured (the height-proportional sway bound: standing 0.013 m/s, seated 0.007), all 11:** 164 84.7 (the seated
frames 230 / 290: a seated body spreads at 0.007 m/s as at 0.015 and 0.03 -- only 0 keeps it DC as the recording is), the
rest as before (176 18.4, 168 10.6, 108 9.3, 183 10.6, 186 9.7, 153 12.6; MDe 176 0.930, 168 0.927). Rejected.

## THE RECIPE (2026-09-20 13:10, the trainer's defaults since this commit): `--array-window none`,
`--person-level-calibration peak-doppler --person-level-calibration-view downstream` (cell mask, target total, when end),
`--velocity-residual-bound 0 --velocity-residual-relative 0.75` (the field = the mesh's kinematics x a learned per-joint
scale + a residual bounded by 0.75 x the joint's own kinematic speed: a still joint stays still), the column-free scaffold
cut per frame by the registered mesh (`MM4DPS_SCAFFOLD_PERSON_SEQUENCES`; the sweep otherwise), the drawn floor calibrated
under the trainer's array window. = candidate `vb0`. Gate on the five mesh-label seeds (real model in brackets):

| 176 | 168 | 108 | 164 | 183 | 186 | 153 |
|---|---|---|---|---|---|---|
| 18.5 (16.5) | 11.5 (13.3) | 9.6 (13.3) | 41.5 (10.5) | 10.7 (3.4) | 9.7 (3.8) | 12.6 (10.4) |

against base2's 162 / 80 / 230 / 179 / 12.7 / 11.3 / 22.9 this morning. Fidelity (the suite's Hann view) vs base2: RT-Pose
RA -0.02..-0.06, RAc +0.02..+0.06 (176 0.11 -> 0.13, 168 0.30 -> 0.35, 164 0.54 -> 0.53, 183 0.66 -> 0.59, 186 0.69 ->
0.60, 153 0.59 -> 0.57), RD -0.01..-0.02, MDs +0.01..+0.02, MDe -0.04 (176) / +0.11 (168) / +0.03 (164, 153) / +0.07 (183) /
+0.06 (186); in the consumers' untapered view RA +0.01..+0.2 on every sequence. HuPR 38: RAc 0.64 -> 0.54, RD 0.77 ->
0.77, MDs 0.97 -> 0.98, MDe 0.95 -> 0.97, vertical RAc 0.43 -> 0.44; 102 / 256 / 40 within noise.

**164's remaining 41.5 cm:** the seated frames 45-75 at 120 / 176 cm (the rest of the sequence 13-54). The rendered
seated person now has the recording's level and DC share; what differs is the scene beside them -- the chair renders at
0.42x the recording's (a dihedral in reality, a dim point cloud in the render; its cell weighs 1 % of the wall's in the
loop-0 term, so the per-point material never fits it) -- and the swap probe put the scene's share of the failure at
about half (render scene + real person: 68 / 82 cm in those frames). A loss that weights a cell by its own magnitude
(the plan's P2 "compare local energy") is the lever; it is objective work, not a flag.

## 1f. Seq 164 analysed frame by frame; the person-window term (2026-09-20 evening, 00abbb7 .. this commit)

**The analysis (docs/SEQ164_ANALYSIS.md).** The frames that fail are the SEATED frames of both periods: the five mesh seeds
read the rendered seated pelvis +40 cm too high (the standing height) on every seated block (real seeds 0 +- 2), the
first period adds a 15-31 cm range bias toward the radar and off-target seeds at the mirror azimuth (12.6 of the 41.5).
Polar-domain swaps (`polar_swap_node.py`): the model reads the person's (R, A, E) magnitude pattern wrong, the ELEVATION
structure first (the recording's (R, A, E) in the +-0.6 m window -> 19 / 17 cm on frames 30-90 / 195-270; the recording's
Doppler structure -> worse). The scene's share is the person's own range band beside it (`scene_swap_node.py`: 100 -> 51,
36 -> 20), not the near / far field or the floor. Measured out: level, the mesh slide (<= 12 cm on the labels), the joint
gains (`g1` 44.7 / `gtorso` 41.9), the antenna elevation sign (`elm` 88.4), a fixed per-row phase term.

**Step 1 = `--person-window-loss` (9240e50): the loop-0 |RAE| SHAPE of the registered person's window in the consumer's own
transform** (RT-Pose: the adapter's TI-table polar minus its Doppler FFT; HuPR: the released generateHeatmap angle stage),
render vs recording, each rms-normalised over the window (it cannot move the level), the consumer's azimuth side of the
person settled per sequence from the recording's window energy (RT-Pose ratios 3-30; HuPR 1.0-1.5: its persons stand near
boresight and the two sides coincide). Step time +10 % (21.7 -> 24 ms). Three candidates, all 11 (mesh seeds; vb0 in brackets):

| cand | 176 | 168 | 164 | 108 | 183 | 186 | 153 |
|---|---|---|---|---|---|---|---|
| vb0 | 18.5 | 11.5 | 41.5 +- 7.5 | 9.6 | 10.7 | 9.7 | 12.6 |
| pw1 (w 1, +-0.6 m) | 17.9 | 11.1 | 39.3 +- 10.4 | 8.7 | 10.5 | 9.8 | 11.8 |
| pw3 (w 3, +-0.6 m) | 18.6 | 11.4 | 40.5 +- 11.7 | 8.7 | 10.5 | 9.7 | 12.1 |
| **pwl (w 1, +-1.2 m lateral)** | 17.7 | 11.3 | **29.2 +- 4.1** | 10.1 | 10.5 | 9.4 | 11.8 |

What moved inside 164 (`seeds_blocks.py`, `pwres.py`): the term's own residual 0.46 -> 0.13 (pw1) / 0.14 (pwl) / 0.11
(pw3), the window's elevation-profile distance on the seated blocks 0.12-0.16 -> 0.07-0.10 -- **the recording's seated
signature IS reachable by the single-bounce point set** (the question step 1 was built to answer), and the render's
seated DC-peak elevation lobe sits on the recording's (bin 17 / 17 at frames 60 / 70; vb0 15). Under pw1 the vertical
bias vanished in 2/3 of the seated blocks (+36..+43 -> +3..+6) but the seeds split (seeds 0 / 3 at 15 / 21 cm medians,
1 / 2 / 4 still in the +40 mode) and new off-target seeds appeared, so the gate stayed. Under pwl (the window widened to
the person's azimuth-streak band, plan step 2) the off-target seed-frames fell 49 -> 9, frames 45-75 went 121 / 176 ->
31 / 39 with the vertical bias gone (+5 / +3), and the sequence gate 41.5 -> 29.2 with the seed std 7.5 -> 4.1. Still at
+40: the sit-down / stand-up blocks 30-45, 75-90, 180-195 and two seated blocks (210-225, 255-270); the first period's
radial bias (-25..-31 cm, the shins) is untouched by any of the three. The weight is not the lever (pw3 = pw1); the window is.

**Fidelity (Hann view) pwl vs vb0.** RT-Pose RA within +-0.013 on every sequence; RAc 164 +0.025, 186 +0.023, 176 / 183 /
153 +0.004..+0.006, **168 -0.06 (0.350 -> 0.289; pw1 0.300, pw3 0.261 -- consistent, a real cost in the Hann view while
its untapered RAc rises 0.407 -> 0.433)**; RD +-0.01; MDe 176 +0.013, 153 +0.020, 168 -0.012. Untapered RAc +0.017..+0.027
on 168 / 164 / 186 / 153. **HuPR 38 (near-static) gains: RAc 0.538 -> 0.620 horizontal, 0.435 -> 0.541 vertical, RA +0.01,
MDe +0.004 / +0.029.** HuPR 102 within +-0.01; HuPR 256 within 0.003 (vertical MDe 0.983 -> 0.969). **HuPR 40 (the other
near-static HuPR sequence) loses its micro-Doppler under every variant: horizontal MDs 0.844 -> 0.772 (pw1 0.774, pw3 0.786),
MDe 0.939 -> 0.874 (0.901 / 0.925); vertical MDs 0.752 -> 0.703, MDe 0.881 -> 0.799 -- against a re-baseline band of
+-0.005 horizontal / +-0.02 vertical (`vb0b`: 0.842 / 0.936 / 0.769 / 0.868).** That is a section-1 regression on HuPR 40.

**Reading.** The term reads the consumer's view for RT-Pose (DC kept, every Doppler bin) but not for HuPR: the HuPR consumer
removes the clutter over the 64 loops and reads Doppler chirps 24..39 only, so a loop-0 (static) shape term constrains the
HuPR person's materials on a quantity the HuPR model never sees, and on the near-static HuPR 40 it pays in the moving band.
The consistent definition -- the person window's |RAE| of the Doppler planes the consumer reads (HuPR: the clutter-removed
band; RT-Pose: DC and all bins), built from the trainer's rendered planes -- is the next commit (plan step 1b); pwl's
+-1.2 m window stays. Not adopted as a default yet: 164 41.5 -> 29.2 and HuPR 38's RAc are the gains, HuPR 40's MDs / MDe
(-0.05..-0.08) and 168's Hann RAc (-0.06) the costs, recorded here.

**Step 1b measured: `pwc` = the term on the Doppler planes the consumer reads (294e095 / 021800a, `--person-window-planes
consumer`, the default; +-1.2 m window, weight 1), all 11.** Gate: 176 17.8 / 168 10.9 / **164 31.4 +- 8.2** / 108 9.0 / 183
10.3 / 186 9.0 / 153 11.9 (pwl 29.2; vb0 41.5) -- 164 as pwl, the rest in band. Fidelity vs vb0: RT-Pose RA within +-0.015,
RAc 168 -0.05 again (0.297), 164 / 186 +0.01, RD +-0.01, MDe 168 +0.06; HuPR 256 clean (vertical MDe 0.987, pwl's -0.013 gone).
**HuPR 40: the vertical radar now gains (MDs 0.752 -> 0.799, MDe 0.881 -> 0.943) but the horizontal still loses MDs 0.844 ->
0.770, MDe 0.939 -> 0.891 and now RAc 0.136 -> 0.067, with the render's RD power ratio 0.67 -> 1.11**: on a still person the
clutter-removed band's window is the floor, and a shape term on a floor-dominated window inflates the moving band to match
noise. The workstation's HuPR 38 / 102 runs of pwc died on a full root disk (this branch's rejected workstation candidates'
workspaces were deleted to free it, metrics kept). **Step 1c (70bd0f7, f58990c): each frame's term weighted by the window's
information content in the consumer's view** -- its mean power over its range rows' mean outside it, 0 at 1x, 1 at
`--person-window-snr 3`x (RT-Pose 164: median 1.0, 56 % of frames at 1; HuPR 38: 1.0 / 0.93; HuPR 40: 0.59 / 0.25) -- candidate
`pws`, all 11. **pws measured: 164 45.4 +- 12.6 (worse than vb0)** -- with the rows' MEAN as the reference, the seated frames
beside the far wall (45-75: the wall shares their range rows) got weight ~0 and the term switched off exactly where it
matters (the window residual there back to 0.47-0.53, the +42 cm reading and 12 / 17 off-target seed-frames back); the rest
in band (176 17.6 / 168 11.3 / 108 8.6 / 183 10.3 / 186 9.0 / 153 11.4). HuPR 40 closer to its band (horizontal MDs 0.808,
MDe 0.935, RAc 0.120, RDp 0.82; vertical MDs 0.780, MDe 0.913 -- both above vb0), HuPR 256 clean. So the weight's reference must
be the floor, not the rows' mean: `--person-window-snr-reference median` (190b939; 164: 78 % of frames at weight 1, 0 at 0)
= candidate `pwm`; beside it `pwh` = the consumer-band term at weight 0.5 without the weighting (`--person-window-snr 0`).

**pwm / pwh measured (2026-09-20 23:35 ET) and the person-window term's seven candidates in one table** (gate on the mesh seeds;
HuPR 40 against vb0's 0.844 / 0.939 horizontal MDs / MDe and 0.752 / 0.881 vertical, band +-0.005 / +-0.02; HuPR 38 RAc against
0.538 / 0.435; 168's Hann RAc against 0.350):

| cand | form | 164 | other 6 RT-Pose | HuPR 40 hori MDs / MDe | HuPR 40 vert MDs / MDe | HuPR 38 RAc h / v | 168 RAc | 256 |
|---|---|---|---|---|---|---|---|---|
| vb0 | -- | 41.5 +- 7.5 | 18.5 / 11.5 / 9.6 / 10.7 / 9.7 / 12.6 | 0.844 / 0.939 | 0.752 / 0.881 | 0.538 / 0.435 | 0.350 | -- |
| pw1 | loop-0, +-0.6 m, w 1 | 39.3 +- 10.4 | in band | 0.774 / 0.901 | 0.757 / 0.886 | 0.585 / 0.527 | 0.300 | clean |
| pw3 | loop-0, +-0.6 m, w 3 | 40.5 +- 11.7 | in band | 0.786 / 0.925 | 0.667 / 0.726 | 0.607 / 0.498 | 0.261 | clean |
| **pwl** | **loop-0, +-1.2 m, w 1** | **29.2 +- 4.1** | in band | 0.772 / 0.874 | 0.703 / 0.799 | **0.620 / 0.541** | 0.289 | vert MDe -0.013 |
| pwc | consumer band, +-1.2 m, w 1 | 31.4 +- 8.2 | in band | 0.770 / 0.891 (RAc 0.067) | **0.799 / 0.943** | (disk) | 0.297 | clean |
| pws | band + weight (rows' mean) | 45.4 +- 12.6 | in band | 0.808 / 0.935 | 0.780 / 0.913 | (disk) | 0.259 | clean |
| pwm | band + weight (rows' median) | 50.2 +- 32.2 | in band | 0.785 / 0.893 | 0.791 / 0.939 | 0.579 / 0.511 | 0.275 | vert RA 0.877 (-0.04) |
| pwh | consumer band, w 0.5, no weight | 36.5 +- 5.7 | in band | 0.791 / 0.911 | 0.808 / 0.934 | 0.596 / 0.518 | 0.302 | clean |

What the table says: (1) the +-1.2 m window is the lever on 164 (29-31 against 39-41 with +-0.6 m; the weight 3 / 0.5 do not
help); (2) the frame weighting (pws, pwm) is harmful on 164 -- the term must act on the seated frames beside the wall, and
any weighting that lowers them there lets the +40 cm reading and the off-target seeds back in -- it is dropped (the flags
stay, default `--person-window-snr 0` = off from the next commit); (3) **HuPR 40's horizontal micro-Doppler loses 0.04-0.07
of MDs under every form of the term** (the loop-0 form also costs its vertical, the consumer-band form gains it); HuPR 38
gains 0.04-0.08 / 0.06-0.11 of RAc under every form; 168's Hann RAc loses 0.05-0.09 under every form while its untapered
RAc rises; 256 / 102 are clean except pwl's / pwm's one vertical metric. The term as a default is therefore a trade the
user decides: pwl (164 -12.3 cm, HuPR 38 +0.08 / +0.11 RAc; HuPR 40 -0.07 / -0.065 hori and -0.05 / -0.08 vert MDs / MDe,
168 Hann RAc -0.06) or pwc (164 -10.1, HuPR 40 hori -0.07 / -0.05 / RAc -0.07 but vert +0.05 / +0.06, 256 clean). Not
flipped to a default here.

## THE RECIPE, amended (2026-09-21 00:05 ET, the trainer's defaults since this commit): vb0 + `--person-window-loss 1
--person-window-lateral 1.2 --person-window-planes loop0` (= candidate pwl, section 1f; the user's call).
Gate on the five mesh-label seeds (real model in brackets):

| 176 | 168 | 164 | 108 | 183 | 186 | 153 |
|---|---|---|---|---|---|---|
| 17.7 (16.5) | 11.3 (13.3) | **29.2 +- 4.1** (10.5) | 10.1 (13.3) | 10.5 (3.4) | 9.4 (3.8) | 11.8 (10.4) |

against vb0's 18.5 / 11.5 / 41.5 / 9.6 / 10.7 / 9.7 / 12.6. Fidelity vs vb0: RT-Pose RA within +-0.013, RAc 164 +0.025 / 186
+0.023 / 168 -0.06 (Hann; untapered +0.026), MDe 176 +0.013 / 153 +0.020; HuPR 38 RAc 0.538 -> 0.620 / 0.435 -> 0.541; 102 within
+-0.01; 256 vertical MDe -0.013; **HuPR 40 horizontal MDs 0.844 -> 0.772, MDe 0.939 -> 0.874, vertical MDs 0.752 -> 0.703, MDe
0.881 -> 0.799: the recorded cost of this default.** The frame weighting stays off (`--person-window-snr 0`).

## 2. Next steps, in order

1. **[done: `lvlcal`, section 1c; accepted pending the user's call on the HuPR 38 / 40 moving-band cost]** The level
   calibration at the end of training on the downstream's quantity, the trainer otherwise unchanged (the frozen recipe
   lvlfit4/5 costs the moving band on every sequence; lvlcal costs it only where the calibration is x2 on a standing body).
1c. **[measured, rejected -- section 1c]** The calibration at the person's own peak cell (`--person-level-calibration-cell
   person`) dims the walking sequences and does not help the standing ones; the mask's peak stays.
1d. **[measured, section 1c: nowin]** `--array-window none`: gate 176 27.7 / 168 12.7 / 108 16.7 / 153 13.8 (real 16.5 / 13.3 /
   13.3 / 10.4); Hann-view fidelity -0.02..-0.06 RA, untapered-view fidelity +0.05..+0.21 RA. Awaiting the user's call on
   the view; then the default flips and the suite reports both views.
1e. **[vrel = the recipe (section 1c); 164's corner located: the annotation column's damage to the scaffold, nocol running]** 164 (121 cm) and the near-static residual (176 28, 108 17 vs 16.5 / 13.3 real):** 164's competitor is still the corner
   at (6.5, -2.5) at 0.89x with the person at 0.77x -- the person's DC share (dc 0.49, moving 1.03: the velocity fiction of
   P2) and the corner's remaining excess. Next probe: the corner cell in the untapered trainer view after nowin.
2. **Both datasets.** HuPR's person level goes through the same stage; verify its AP does not regress (the dev set's four
   HuPR sequences, `fid_node.py`, the proxy). HuPR 38 / 40 are the near-static HuPR sequences and show the same
   moving-band cost as the RT-Pose standing ones once the level is right (1c).
2b. **[done, c0f9241, section 1c]** The scaffold's ghost of the person: the annotation's column was 0.45-0.6 m off; the
   registered person's swept surface is cut out of the scaffold in person_stack and the background fitted again. Gate
   unchanged (the ghost was not the competitor -- the far wall is); kept as a correctness fix.
3. **The residual (30-37 vs 13 / 10 cm real):** the render's person is broader than the real one (peak/sum ratio, cells
   above half-peak). That is the shape question (plan P2: compare local energy rather than per-cell magnitude; or the
   coherent structure of the standing body) and the scene's near-field, and it is second order after the level.
4. **Option A (sixth branch `fidelity/solid-angle-sampling`, /home/adnan/Desktop/4DPS-solid, 9 commits on 9336102, not
   merged): closed, same verdict.** Both point sets as first hits of one 0.25 deg direction grid, constant dOmega, the level
   knobs removed: construction sound (183 held, steps 15-20 % faster), but on the near-static sequences the level moved into
   the shared material's coherent fraction (0.52 -> 0.05), the person's peak stayed at 0.37x and contrast 0.53-0.55 on 176;
   fidelity regressed beyond spread (176 RAc 0.11 -> 0.05, RD 0.61 -> 0.54; HuPR 38 RA 0.95 -> 0.92 from the first hit taking
   the nearest layer of the monocular wall; the grazing floor 2-3x too bright at physical weight). Its write-up:
   4DPS-solid/docs/SOLID_ANGLE_SAMPLING.md; artifacts in mmSD/artifacts/solid_angle. It is the right formulation for the
   *slot* and nothing else: neither the slot nor the scene weighting anchors the level -- only step 1 does. Not resumed
   until step 1 is in; if it is resumed later it goes behind the level fit, with the near-field and the layered walls handled.

## 3. Running / resources

- Node GPUs 0-7 are all this session's since 2026-09-20 04:00 (the retrain is done: `results_gvhmr`, the gate's seeds). One
  candidate on the 9 node sequences is one round (~15 min) with `GPUS="0 1 2 3 4 5 6 7"`; the gates run one sequence per GPU
  (`GATE_TAG` per process; `GATE_CKPT` selects the seed set).
- **The workstation's two 24 GB cards are free again** (the sixth-branch session is closed): they take one RT-Pose sequence
  each (~22 min per sequence there against ~15 on an H100; HuPR 38 ~5 min), with the workstation tools of this branch
  (`run_cand.sh`, `eval_cand.sh`: seeds 0 and 4 only, dataset labels). Use them to widen step 1's fast set (e.g. 168 and 164
  on the workstation while 176 / 108 / 183 run on the node) and for the HuPR dev sequences in step 2, so a candidate's
  full 11-sequence answer arrives in one round instead of four. The five-seed gate stays on the node (it needs the five
  real-trained checkpoints there, and ~3 min on one GPU).
- Tools on this branch: `contrast_node.py <cand> <seqs>`, `shape.py` (window sum vs peak), `lvl.sh` (render-only level
  test), `gate_node.py` (now dumps per-frame seed predictions), `viz_triple.py <before> <after> <seqs>`.
- Running the tools: always one sequence per GPU in parallel on 5-7 (`CUDA_VISIBLE_DEVICES=<g> ... &`, three at a time),
  never sequentially. The video tools (`viz_pair.py`, `viz_triple.py`) are bound by matplotlib's frame encoding
  (~140 s per 299-frame video, GPU idle): render 3 videos in parallel, and for a quick look use `VIZ_FRAMES=60`
  (12 s of video) or lower the dpi; `contrast_node.py` / `shape.py` are ~30 s per sequence and are the fast checks.
