# Seq 164: the next steps, in order (2026-09-20, 20:45 ET)

From docs/SEQ164_ANALYSIS.md. The recipe is vb0 (17310e3, the trainer's defaults); its gate on the five mesh-label seeds
is 176 18.5 / 168 11.5 / 108 9.6 / **164 41.5** / 183 10.7 / 186 9.7 / 153 12.6 (real 16.5 / 13.3 / 13.3 / 10.5 / 3.4 / 3.8 /
10.4). What 164's 41.5 is made of (the analysis, section 1): the seated frames of BOTH periods read +40 cm too high (the
standing pelvis height), the first period adds a 15-31 cm range bias toward the radar and off-target seeds (12.6 cm of
the 41.5); what the model reads wrong is the (range x azimuth x elevation) magnitude pattern of the person's cells,
the elevation structure first; the scene's share is the person's own range band beside it. The level, the Doppler /
DC split (the mesh slide), the joint gains, the antenna elevation sign, the near / far scene and the floor are measured
out. The mechanism behind the recording's seated elevation signature is NOT established; multibounce rendering is off
the table (user rule). So the steps below (1) make the trainer fit the person's return where the consumer reads it
and (2) measure, on the way, whether the recording's seated signature is reachable by a single-bounce point set at
all -- that answer decides how far this branch can take 164.

**Rules for every step** (the user's): first-principles, both datasets, no per-dataset flags, no label-derived fixes
(the person's window is the registered mesh's, as the level calibration's pelvis window already is), one change per
commit with its gate + fidelity numbers, no regression on the section-1 fidelity metrics, and the candidate runs as one
round of all 11 sequences: the 7 RT-Pose + HuPR 40 / 256 on the node (`GPUS="0 1 2 3 4 5 6 7" SEQS=... launch_cand.sh
<cand> "<VF_APPEND>"`, RT-Pose with `SCAFFOLD_REBUILD=1 SCAFFOLD_SRC=/data/mmsd/scaffolds_pf`, HuPR NEVER with
`SCAFFOLD_REBUILD=1` -- the node has no Depth-Anything), HuPR 38 / 102 on the workstation (`run_cand.sh <cand> hupr
<38|102> <gpu>`, metrics scp'd to `outputs/<cand>/seq<k>_ws_metrics.json`); then `gate_node.py` (GATE_CKPT =
results_gvhmr, one sequence per GPU), `contrast_node.py`, `fid_node.py` (both views), and the per-block seed
decomposition `seeds164.py` on 164 (and on 176 / 153 as controls). No candidate is run before the user has read this.

**HuPR 40's known state** (STATIC_SEQS_NEXT, the reverted docs commit 6796460, re-run of vb0 without the rebuild):
HuPR 256 within noise of base2; **HuPR 40 under vb0: RAc 0.136 / 0.097 (base2 0.235 / 0.121), RD 0.429 / 0.232 (0.514
/ 0.240), MDs 0.844 / 0.752 (0.885 / 0.837), MDe 0.939 / 0.881 (0.925 / 0.935)** -- its RAc ~0.1 below base2 since the
untapered view (nowin 0.111), RD 0.43 and the vertical MDs 0.75 vb0-specific lows not yet placed against the run-to-run
band. The first round below therefore includes **a second vb0 run of HuPR 40 and 256** (no code change) so that the
band is known before any step is judged on HuPR 40; a step is not called a HuPR 40 regression inside that band.

## Step 1. The person's return fitted in the consumer's view: a person-window (R, A, E) term (`--person-window-loss`)

**Status (2026-09-20 21:30 ET): implemented (9240e50) and measured on all 11 as pw1 / pw3 / pwl (STATIC_SEQS_NEXT 1f).** The
signature is reachable (residual 0.46 -> 0.13, the seated elevation lobe on the recording's); the widened window (step 2,
`pwl`) takes 164 41.5 -> 29.2 +- 4.1 with no RT-Pose gate regression and HuPR 38 RAc +0.08 / +0.11, at the cost of HuPR 40's
micro-Doppler (MDs / MDe -0.05..-0.08 on both radars, outside its band) and 168's Hann RAc (-0.06). Not a default yet.
**Step 1b (next commit):** the same term on the Doppler planes the consumer reads -- HuPR: the clutter-removed band (chirps
24..39 of the released heatmap; the loop-0 shape is a quantity the HuPR model never sees, which is where HuPR 40's cost comes
from); RT-Pose: DC and every bin -- from the trainer's rendered planes and the recording's own planes, the +-1.2 m window kept.
Judged as (e) below plus HuPR 40 against `vb0b`.
**Step 1b / 1c measured (STATIC_SEQS_NEXT 1f, the seven-candidate table): the consumer-band form (pwc) gates 164 at 31.4 and
gains HuPR 40's vertical but keeps its horizontal MDs cost; the frame weighting (pws / pwm) is dropped (it lets the seated
reading back in). Best on 164: pwl 29.2 +- 4.1. The default stays off pending the user's call on HuPR 40's horizontal MDs
(-0.04..-0.07 under every form).**

**(a) What we do now, and how the change differs.** Today the velocity trainer's loop-0 objective is the RA term
(untapered since nowin), the plane / MD terms and an elevation term, all in the trainer's own beamformer (the mm4dps
table, slots averaged) over the whole ROI; the person's cells weigh ~1 % of a wall's in the RA term, and the elevation
term is scene-dominated. The consumer (the HRRadarPose adapter) reads the person through the TI table, one channel per
slot, a 32-bin elevation FFT and a Cartesian resample -- the view in which the model's seated reading fails and in
which nothing in the objective looks at the person's elevation. The level calibration already went to that view
(d1463c3, `--person-level-calibration-view downstream`: the trainer's and the adapter's transforms read the same person
1.2-1.5x apart). The change: one more loop-0 term, the |RAE| magnitudes of the person's window (the registered mesh's
pelvis +-0.6 m in x / y, all range-azimuth cells of the window, all 32 elevation bins; DC-only and Doppler-summed, each
normalised by the recording's window sum so the term is a shape term and cannot move the level) in the CONSUMER's
transform, render against recording, weighted like the elevation term. For HuPR the same term reads each radar through
HuPR's released preprocessing (`downstream/hupr_mscsa/adapter.py`, `HuPRAdapter`, beside `RTPoseAdapter`; HuPR's consumers read 2-D range-azimuth per
radar, so there the term is the person window's RA shape in that view -- the same flag, the dataset's own consumer
view, as the calibration flag does today). The window is wider laterally than the calibration's (+-1.2 m) when
`--person-window-lateral` says so, to cover the azimuth streak (step 2). One commit; default off until measured, on
once accepted (as every recipe flag before it).

**(b) The issue it solves.** The trainer never sees the quantity the model reads wrong (analysis 3b): the person's
(R, A, E) pattern in the adapter's view. With it in the objective, the person's per-point materials, normals and
the joint gains -- the only person parameters the trainer has -- are pushed to reproduce the recording's seated
elevation / range / azimuth pattern instead of the trainer-view RA sum. It is also the measurement that decides the
rest: if the term's residual on the seated frames does not fall (the lobe pattern stays the standing one), the
recording's signature is not reachable by a single-bounce point set above the floor, and steps 3-5 are what is left.

**(c) Expected effect.** 164: the seated blocks are the target -- the swap's ceiling is 19 / 17 cm for frames 30-90 /
195-270 (the recording's (R, A, E) in the window) against vb0's 100 / 36; reaching half of that puts the gate near
20-25 (real 10.5). If only the vertical reading moves, the first period keeps its range bias (the on-target mean stays
~40 there) and the gate lands ~30. The other ten: the standing / walking persons already read right in this view, so
the term's gradient there is small; expect +-2 cm on 176 / 168 / 108 / 183 / 186 / 153 (the run-to-run band), a
possible MDe move where the materials shift (168's band is 0.82-0.955 across identical recipes).

**(d) Risk.** HuPR: the person term in HuPR's consumer view changes the person's materials on 38 / 40 (near-static)
and 102 / 256; the level is fixed by construction (normalised term), the moving band can move (MDs / MDe, HuPR 40's
known lows -- judged against the second vb0 run). RT-Pose fidelity: RA / RAc in the Hann view may move by the
0.01-0.03 the branch has seen on every person-side change; the untapered view is the one that should improve. Compute:
one adapter transform of the person's window per calibration frame per step (the calibration already does this once at
the end; here it is per step on the subset of frames the loop-0 term uses) -- measure the step time; if > 20 % slower,
sample the frames.

**(e) How it is judged.** (1) The direct target: the person-window elevation profile distance render vs recording on
the seated frames (`sit.py` lobes 8..24 at the DC peak; `elcal.py` row phase medians) before / after -- this says
whether the fit reached the signature at all; (2) `seeds164.py` on 164: the vertical bias per seated block (+40 -> ?),
the off-target seed-frames (49 -> ?), the radial bias of 15-45 / 75-90; (3) the gate on all 7, the HuPR proxies, the
fidelity table both views, `contrast_node.py`; (4) no section-1 regression outside the band. Accept if 164 improves by
more than the band with (1) moving in the right direction and (4) holding; reject and record if (1) does not move
(then the signature is outside the model: see the note after step 5).

## Step 2. The rendered person's azimuth streak

**(a)** Now: the rendered person is a coherent point set with per-point geometric phase; untapered, its azimuth
response is a narrow main lobe with the full array's sidelobe streak across all azimuths at the person's range (every
RA frame of the render; the recording's person is a compact main lobe, its sidelobes decorrelated). The scene swap
puts this band at half of the seated failure's scene share (analysis 3c: the same-range band 100 -> 51, 36 -> 20). The
change is not a new term: the step-1 window widened laterally (`--person-window-lateral 1.2`, then the person's whole
range band) makes the streak cost in the consumer's view, and the person's per-point coherence parameters (the
material's coherent fraction and roughness columns, per point, already trainable) are what the trainer can move to
decorrelate it. One commit: the wider window as the default if step 1's narrow one is accepted first.

**(b)** The scene-side half of the seated failure (the person's own sidelobes read as scene at the person's range).

**(c)** 164: the same-range-band swap's 50.7 / 20.1 are the ceiling of this step alone; combined with step 1, the
seated blocks approach the "real scene + render person" 34 / 13. 176 / 168 / 108 (standing): the streak exists there
too and is harmless to the gate today; RAc in the untapered view should rise (the streak is what RAc's suppressed view
counts). HuPR: the same window on HuPR's RA views; the HuPR persons' streaks are smaller (nearer, wider bodies).

**(d)** Moving the coherence columns changes the moving band's texture (MDs / MDe); judged against the band. No
compute cost beyond step 1.

**(e)** A streak metric added to `contrast_node.py`: the render / recording energy ratio in the person's range band
outside +-0.6 m lateral, per frame; the scene swap's "same range band" line re-run; the gate and fidelity as in step 1.

## Step 3. The mirror azimuth in the person mask

**(a)** Now: the trainer's person mask includes the mirror azimuth (`person_range_angle_mask_torch
include_mirrored_angle`; STATIC_SEQS_NEXT 1c lesson 2), so rendered energy at the person's mirror is unconstrained by
the person terms; the calibration already reads the person's own side (`--person-level-calibration-cell side` exists,
mask is the default). Change: the person mask excludes the mirror (one flag, default flipped once measured); for HuPR
(no beamformer mirror) it is a no-op by construction -- not a per-dataset flag.

**(b)** The off-target seeds of frames 45-75 go to the mirror azimuth (-17 deg at 7.7-8.3 m) in most cases (analysis
5). First measurement, before the commit: the render / recording ratio at the mirror cells of the seated frames
(`blocks164.py` with the competitor window set to the mirror); if the render is not brighter there, this step is
dropped -- the off-target seeds vanish with the person's pattern fixed (the (R, A, E) swap: 0 / 305), so it may not
be needed.

**(c)** 164: the 12.6 cm of off-target seed-frames, if the mirror is the cause; nothing on the standing sequences
(no off-target seeds there). **(d)** None expected on fidelity (the mask only gates the person terms). **(e)** the
off-target count per block (`seeds164.py`), the mirror-cell ratio, the gate.

## Step 4. The first period's range pattern (the shins)

**(a)** Now: in frames 15-45 and 75-90 the render's peak sits at the knees / shins, 0.25-0.3 m nearer than the
pelvis, and the seeds follow it (radial -31 cm); the recording's peak is also 0.15-0.25 m nearer than the pelvis (the
chair's cell) and the model reads it right. The gain probes (uniform, torso x4) did not move the bias, so it is not the
joint gains; the candidates inside the model are the seated torso's learned normals (a reclined torso's specular lobe
misses the radar; the recording's torso is a rough, broad scatterer) and the per-point roughness on the torso. No
change is proposed until step 1 has run: the (R, A, E) term contains the range pattern and should move it if the
torso's material / normals can; the measurement after step 1 is the radial bias of blocks 15-45 / 75-90 and the
render's range profile in the window (`zprof.py`) against the recording's.

**(b)** The -15..-31 cm radial bias of the first period (the on-target mean of 43-50 in 45-90). **(c)** 164 frames
15-45 / 75-90 from 38-54 toward the second period's 27-30; nothing elsewhere. **(d)** None beyond step 1's. **(e)**
`seeds164.py` radial bias, `zprof.py` range profiles; if step 1 leaves it, the next commit is a normal prior on the
person (the mesh normal as the learned normal's anchor, one weight, both datasets), measured the same way.

## Step 5. The scene at the person's range (the chair, the wall cells)

**(a)** Now: the chair renders at 0.42x (STATIC_SEQS_NEXT), the corner competitor at 0.71-0.76x; per-cell loss weighting
(ec0d60a) and geometry-aware sampling (56c3c0c) were measured and reverted. After steps 1-3 the scene swap's
"same range band" line is re-measured; if the band still costs more than the streak (step 2's metric says how much is
the person's own), the remaining scene item is the elevation content of the scene's near-person cells in the consumer
view -- the same window term, applied to a scene band (the person's range band across azimuth) as a shape term. Not
before the measurement; not a new weighting scheme (both rejected forms stay rejected).

**(b)-(e)** as step 2's, on the scene band; judged by the scene swap lines and the gate.

## If step 1's residual does not fall (the note)

The analysis leaves open whether the recording's seated signature -- the elevation lobe pattern of a seated body on a
chair over the ground, in a 4-row array -- can be produced by any arrangement of single-bounce point scatterers on the
mesh. Step 1 answers it: the trainer, told to match the pattern in the consumer's view with every person parameter it
has, either reaches it or not. If not, 164's seated frames are a limit of the single-bounce model on this rig (a
seated person 7.5 m out beside a wall, read by a model that decodes posture from an elevation signature the render
cannot make), to be reported as such with the numbers of analysis section 4, and the branch's remaining gate work goes
to the second period's smaller residual (steps 2-3, the streak and the mirror) and the ten other sequences. That is a
finding, not a flag; the user decides then whether the geometry (the chair in the person's point set rather than the
scaffold's) is the next avenue.

## The first round (when the user says go)

1. Commit: `--person-window-loss` (step 1, narrow window), the analysis tools under `downstream/tools/static_viz/`
   (`seeds164.py`, `zprof.py`, `polar_swap_node.py`, `scene_swap_node.py`, `elcal.py`, `gains.sh`, `elmirror.sh` with
   the symlink guard), docs.
2. Node: `launch_cand.sh pw1 "--person-window-loss <w>"` on the 7 RT-Pose + HuPR 40 / 256 (`SCAFFOLD_REBUILD=1
   SCAFFOLD_SRC=/data/mmsd/scaffolds_pf` for RT-Pose only), plus `vb0b` = vb0 again on HuPR 40 / 256 (the band); workstation:
   `run_cand.sh pw1 hupr 38 0`, `run_cand.sh pw1 hupr 102 1`.
3. Gate (mesh seeds, one sequence per GPU), contrast, fid (both views), `seeds164.py` on 164 / 176 / 153, `sit.py` /
   `elcal.py` on 164's seated frames; the step's (e) decides; record in STATIC_SEQS_NEXT and here.
