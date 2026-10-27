# RT-Pose seq 164 under THE RECIPE (vb0): why some frames match the recording and others do not (2026-09-20, 20:30 ET)

Companion to docs/STATIC_SEQS_NEXT.md (sections 1-1e, THE RECIPE). Everything here is the five mesh-label real-trained
HRRadarPose seeds (`results_gvhmr`, labels Train_gvhmr.json) on vb0's render of seq 164 (commit 17310e3's tree), the
recording's cube `/data/mmsd/4DPS_downstream/rtpose/seq164/cascade_real.npy`, and the adapter's own transform (the TI
table, untapered, 64-point Doppler FFT, 32 elevation bins, the DZYX Cartesian resample with a 0.36 m z voxel). The
gate's per-frame dump (`gate_vb0.json`: every seed's pelvis per frame) is the source of every seed number. Tools written
for this analysis (scratch copies in `/tmp/*.py` on the node, `<scratchpad>/` on the workstation; listed at the end):
`seeds164.py` (per-block seed decomposition), `zprof.py` (model-tensor z / range profiles), `polar_swap_node.py`
(axis swaps inside the person's polar window), `scene_swap_node.py` (partial scene swaps), `gains.sh` / `elmirror.sh`
(render-only probes), `elmono.py` / `elcal.py` / `elsign.py` / `rowphase.py` (elevation row phases). The frames of
`artifacts/static_check_v4/seq164_check.mp4` were extracted with ffmpeg (299 frames, one per radar frame) and read
frame by frame; `seq176_check.mp4` and the `static_pair_v4` / `static_geometry_v4` videos of 176 / 168 / 108 / 183 /
186 / 153 were the controls.

## 0. The answer in one paragraph

The frames that match are the ones where the person stands or walks; the frames that do not are the ones where the
person **sits** -- both seated periods, not only the first. In the seated frames the rendered person has the recording's
level (window peak 0.95-1.47x, DC 0.75-1.08x, contrast render >= recording in every block), sits at the label's range
and azimuth within a cell, and the model still reads it wrong in one specific way: **the five seeds put the pelvis
+40 cm too high** (median vertical bias +40..+43 cm over the on-target seed-frames of every seated block, both
periods; the real seeds: 0 +- 2 cm), i.e. at the height of a standing pelvis. In the first seated period (7.5-7.8 m,
beside the far wall) that comes with a 15-30 cm range bias toward the radar and, in frames 45-75, one to two of the
five seeds per frame leaving the person for a far cell or the person's mirror azimuth -- those outliers are 12.6 of
the 41.5 cm gate. The second seated period (5.0 m, open floor) has the same +40 cm and nothing else (27-45 cm per
block against the real model's 8-9): it is not "fine", it is the same failure at a smaller amplitude. The polar-domain
swaps locate what the model reads wrong: **the (range x azimuth x elevation) magnitude pattern of the person's cells,
above all its elevation structure** -- giving the render's person window the recording's (R, A, E) magnitudes and
keeping the render's own Doppler distribution gates the seated frames at 17-19 cm (real 8-13); the recording's
elevation structure alone closes the second period entirely (35.7 -> 17.5) and most of the first; the recording's
Doppler structure alone does nothing (130 / 41). The level, the Doppler / DC split (the mesh-slide hypothesis), the
per-joint gains, the near and far scene, the floor voxels and the antenna-elevation sign were each tested and are not
it (section 6). What produces the recording's seated elevation signature is *not* established (section 5): the render's
elevation content at the person is the geometric one and the recording's is not, and the difference is site- and
posture-dependent, not a fixed calibration term.

## 1. The sequence, block by block

Label range R (radial), posture (pelvis z from the mesh labels), the recording's and the render's gate (seed-mean MRPE,
cm; `gate_vb0.json`), the render's seed **median** (the middle seed's mean over the block), the mean over on-target seed-
frames (< 1 m), the number of off-target seed-frames (of 75 = 5 seeds x 15 frames), and the on-target bias (median
over seed-frames of prediction - label along the label's line of sight (+ = farther), lateral, vertical; `seeds164.py`);
the person's window peak render / recording (Doppler-summed, +-0.4 m / +-8 deg), its DC-only ratio and the
person / competitor contrast (`blocks164.py`, medians per block).

| block | R, posture | real gate | vb0 gate | vb0 seed-median | on-target mean | off-target | bias rad / lat / vert (cm) | level tot / DC | contrast ren / rec |
|---|---|---|---|---|---|---|---|---|---|
| 0-15 | 6.8 m, standing (z -0.30) | 9.6 | 19.9 | 21.1 | 19.9 | 0 | -14.5 / -11.6 / +1.9 | 1.24 / 0.98 | 2.41 / 1.29 |
| 15-30 | 7.2, sitting down (z -0.51) | 13.9 | 38.3 | 36.3 | 38.3 | 0 | **-30.8** / -7.1 / +20.5 | 0.95 / 0.76 | 0.95 / 0.66 |
| 30-45 | 7.6, seated (z -0.68) | 5.9 | 54.5 | 54.8 | 54.5 | 0 | **-31.3** / -8.2 / **+42.7** | 1.21 / 1.08 | 1.26 / 0.71 |
| 45-60 | 7.7, seated | 10.1 | 120.6 | 106.0 | 43.1 | **16** | -14.5 / +4.7 / **+42.9** | 0.95 / 0.95 | 0.75 / 0.54 |
| 60-75 | 7.8, seated | 25.1 (2 outliers) | 176.4 | 153.8 | 50.0 | **28** | -22.5 / +1.8 / **+43.2** | 0.98 / 0.75 | 0.41 / 0.29 |
| 75-90 | 7.6, seated -> rising | 10.7 | 53.9 | 51.5 | 50.0 | 1 | **-28.5** / -5.8 / **+39.9** | 1.47 / 0.90 | 1.27 / 0.62 |
| 90-105 | 7.1, standing up / walking | 12.1 | 22.0 | 21.1 | 22.0 | 0 | -14.0 / +1.6 / +5.2 | 1.23 / 0.83 | 1.22 / 0.66 |
| 105-120 | 6.4, walking | 12.3 | 13.2 | 13.9 | 13.2 | 0 | +2.9 / +10.9 / +0.3 | 1.09 / 1.71 | 2.95 / 2.08 |
| 120-135 | 5.4, walking | 8.6 | 14.4 | 14.3 | 14.4 | 0 | +0.9 / +13.1 / -0.1 | 1.09 / 1.95 | 3.90 / 2.36 |
| 135-150 | 4.6, walking | 6.5 | 14.4 | 14.1 | 14.4 | 0 | -0.8 / +13.3 / +0.9 | 0.88 / 0.80 | 4.19 / 3.43 |
| 150-165 | 4.2, walking | 6.9 | 12.3 | 12.7 | 12.3 | 0 | -3.3 / +10.8 / -0.5 | 0.76 / 0.85 | 3.46 / 3.19 |
| 165-180 | 4.5, turning / sitting down | 15.7 | 17.4 | 18.0 | 17.4 | 0 | -9.3 / +8.0 / +3.1 | 1.18 / 0.89 | 4.19 / 2.93 |
| 180-195 | 4.9, sitting down (z -0.62) | 10.6 | 48.7 | 49.8 | 48.7 | 0 | **-25.2** / +5.4 / **+41.8** | 1.10 / 1.03 | 2.04 / 1.20 |
| 195-210 | 5.0, seated (z -0.65) | 8.0 | 29.2 | 25.8 | 29.2 | 0 | -0.5 / +11.7 / **+36.1** | 0.91 / 1.02 | 0.93 / 0.74 |
| 210-225 | 5.0, seated | 7.6 | 31.0 | 25.0 | 27.2 | 1 | -0.4 / +11.3 / +4.4 (bimodal) | 0.97 / 1.03 | 0.80 / 0.57 |
| 225-240 | 5.1, seated | 7.9 | 30.3 | 31.4 | 30.3 | 0 | -1.8 / +10.0 / **+38.5** | 0.97 / 0.99 | 0.65 / 0.49 |
| 240-255 | 5.1, seated | 8.5 | 42.6 | 32.4 | 32.3 | 3 | -3.8 / +11.7 / **+41.3** | 1.08 / 1.21 | 0.96 / 0.68 |
| 255-270 | 5.1, seated | 8.7 | 44.6 | 47.9 | 44.6 | 0 | -17.3 / +11.0 / **+43.4** | 0.76 / 0.76 | 0.55 / 0.51 |
| 270-285 | 4.8, standing up | 9.5 | 32.6 | 34.7 | 32.6 | 0 | -17.5 / +7.7 / +24.1 | 0.94 / 0.81 | 2.36 / 1.81 |
| 285-300 | 4.3, standing / walking | 11.1 | 12.8 | 13.1 | 12.8 | 0 | -3.8 / +11.9 / -1.2 | 1.19 / 1.08 | 4.99 / 3.54 |
| **all** | | **10.5** | **41.5** | | 29.9 | 49 / 1495 (3.3 %) | -8.8 / +7.5 / +5.8 | | |

Reading the table:
- **Standing and walking frames (0-15, 90-180, 285-299) match**: 12-22 cm against the real 7-15, no vertical bias,
  the lateral +11..+13 cm of frames 105-165 is a constant offset of the render's person along the walk (the seeds put
  it a cell to one side; the same sign on 176 / 153 / 168 at 8-13 cm, the branch's known residual, not this sequence's).
- **Every seated block carries a +36..+43 cm vertical bias** on the on-target seeds, in both periods, and the real
  seeds carry none (|vert| <= 2 cm in every block). A seated pelvis is at z -0.62..-0.70; +0.40 puts it at -0.25: the
  standing pelvis height. The model reads the rendered seated person as a standing one.
- **The first period adds a radial bias of -15..-31 cm** (the seeds nearer than the label) from the sit-down (15-30)
  through the rise (75-90), and **off-target seed-frames in 45-75** (16 and 28 of 75). The off-target seed-frames are
  12.6 cm of the 41.5 cm gate; the on-target biases are most of the rest (the on-target mean is 29.9).
- **The second period has the vertical bias alone** (radial -0.5..-4 cm except the sit-down / stand-up frames): 27-45
  per block against the real 8-9. This is the "steady 27 cm of frames 120-299 against the real 9" of the task statement:
  it is the seated frames 180-270 at 29-49 cm; the walking frames 105-180 and 285-299 are at 12-17.
- Level is not the separator: the window peak is at 0.95-1.47x the recording in every seated block and the DC share
  0.75-1.08x; the render's contrast exceeds the recording's in every block (the model finds the recording's person at
  contrast 0.29-0.74 and loses the render's at 0.41-1.27).

## 2. Frame by frame at the transitions (the video, then the numbers)

Frame numbers are radar frames (video frame n+1). RA = the video's range-azimuth |RA| (recording left, render right,
one linear colour scale per row); the label pelvis is the cyan +, the seeds' pelvises the magenta x.

- **Frames 0-15, standing at 6.8-7.0 m, +17 deg.** Recording: a compact blob at the label with a Doppler tail (the
  person shifts weight). Render: the same range, a narrower blob elongated along AZIMUTH -- the untapered array's
  sidelobe streak of a coherent point-like return (section 1d of STATIC_SEQS_NEXT for the wall; the person shows the
  same). Both seed sets on the label (8 / 15 cm at frame 10).
- **Frames 15-30, sitting down.** The pelvis moves back onto the chair (label radial 7.05 -> 7.4 m); the recording's
  blob follows the label (frame 25: the real seeds at 7.1-7.2 m, 17 cm; frame 30: on the label, 4 cm). The render's
  blob **stays at 6.9-7.0 m**, where the feet and shins are (the side view: feet at Y 6.9-7.0, knees 7.0-7.1, pelvis
  7.3-7.4 for radial 7.6), and the seeds follow it: 38 / 55 / 54 cm at frames 20 / 25 / 30, radial bias -31 cm.
- **Frames 30-45, seated.** Frame 40: recording blob exactly on the label (7.6 m), real seeds 6 cm; the render's blob
  at 7.3-7.4 m, all five seeds on it, 55 cm (radial -31, vertical +43). No outliers yet.
- **Frames 45-75, seated, the worst block.** Recording: the person on the label in every frame (real seeds 7-16 cm;
  the real model's own two outliers at frames ~55 and ~70 give its 25 cm block). Render: a FAINT blob on the label
  (level 0.95-0.98x by the window peak -- faint because the colour scale is the recording's peak), four seeds on it and
  one or two per frame elsewhere: frame 45 one seed at (2.5 m, 0 deg); frames 55, 60, 70 one seed at (7.7-8.3 m,
  **-17 deg**) = the person's MIRROR azimuth (the person at +17 deg); frame 60: no seed on the person (352 cm). Frame 50
  with all five on the person: 31 cm. The seed-mean per frame (video panel 3) is therefore a spiky trace of 100-350 cm
  made by one seed at a time, on top of the ~45 cm of the on-target seeds.
- **Frames 75-100, standing up.** The label moves toward the radar (7.5 -> 7.1 m); the recording's blob follows it
  (real seeds 5-19 cm); the render's blob is 0.2-0.4 m nearer than the label (7.1 vs 7.5 at frame 80, 7.0 vs 7.4 at 85,
  6.9 vs 7.3 at 90) and the seeds follow the blob (44-49 cm). By frame 100 the person walks and both match.
- **Frames 105-165, walking** (6.4 -> 4.2 m): both match (8-14 cm); the render's person is again a range-narrow,
  azimuth-elongated streak against the recording's compact blob, but the model does not mind while the person walks.
- **Frames 175-195, sitting down at 5.0 m, -13 deg.** Frame 180: recording on the label (16 cm); render 0.15 m nearer,
  34 cm; vertical bias +42 as soon as the pelvis is down.
- **Frames 195-270, seated at 5.0 m.** Both blobs bright and on the label in range-azimuth (frame 200: 8 vs 19 cm;
  220: 7 vs 20); the render's excess is entirely the +36..+43 cm vertical reading (the RA panel cannot show it; the
  gate dump does). Off-target seeds are rare here (4 of 380 seed-frames in 195-270).
- **Frames 270-299, standing up and walking**: the bias fades with the posture (+24 at 270-285, -1 at 285-299).

What the mesh does meanwhile (`kinblock.py`, the trainer's savgol kinematics, medians per block): pelvis speed 0.045
m/s in 45-60 and 0.076 in 60-75 (raw frame-to-frame 0.044 / 0.084), 0.14 in 30-45, 0.30 in 75-90; in the second seated
period 0.005-0.025 m/s. Pelvis height -0.68..-0.70 (first) / -0.65..-0.66 (second). The label's radial range in the first
period goes 7.56 -> 7.73 -> 7.81 -> 7.57 (block means): 0.25 m over 30 frames, of which the real seeds follow all but
5-12 cm (their radial bias -5.3 / -11.9 / -6.0 in 45-60 / 60-75 / 75-90): **the mesh drifts away from the real return
by at most ~12 cm** in those frames. That is the size of the mesh-slide effect on the labels; it is not the size of the
failure.

## 3. What the model reads wrong: the swaps

### 3a. Box swap in the model's tensor (`swap_box_node.py`, the person's box +-0.8 m x / y, +-1.2 m z; MRPE over the
frames, 5 seeds this time via `scene_swap_node.py`)

| input, frames 30-90 / 195-270 | MRPE (cm) |
|---|---|
| real | 13.0 / 8.1 |
| vb0 | 100.4 / 35.7 |
| vb0 scene + REAL person box | 35.9 / 15.1 |
| real scene + vb0 person box | 34.4 / 13.4 |
| vb0, person box x1.5 (seeds 0/4) | 55 / 60 -- 32 / 28 |
| vb0, person box x2 (seeds 0/4) | 54 / 66 -- 35 / 33 |

The level (x1.5, x2) does not close it. Each half -- the rendered person in the real scene, the real person in the
rendered scene -- gates near the real model in the second period (13 / 15 vs 8) and at a third of vb0 in the first; only
the two rendered halves together fail. Both halves are faults; the person's is the larger one in the first period.

### 3b. Which axis of the person's cells (`polar_swap_node.py`: the person's window |x - x_label| <= 0.6, |y| <= 0.6 in
the adapter's polar (64 Doppler x 256 range x 128 azimuth x 32 elevation, complex); hybrids resampled to the model's
tensor; 5 seeds; MRPE | off-target seed-frames | on-target bias radial / lateral / vertical)

| window variant | frames 30-90 | frames 195-270 |
|---|---|---|
| real | 13.0 \| 2/305 \| -5 / +2 / **0** | 8.1 \| 0/380 \| +1 / +6 / **-1** |
| vb0 | 100.4 \| 45/305 \| -26 / -6 / **+42** | 35.7 \| 4/380 \| -4 / +11 / **+40** |
| render + the recording's window (all D, R, A, E) | 46.0 \| 21/305 \| -8 / +5 / **+2** | 21.0 \| 2/380 \| 0 / +2 / **+1** |
| render window with the recording's **(R, A, E) magnitudes**, the render's own Doppler distribution per cell | **19.0 \| 0/305 \| -10 / +5 / +3** | **17.0 \| 0/380 \| -1 / -2 / +2** |
| render window with the recording's **elevation structure** (per (D, R, A) cell: the render's energy, the recording's complex elevation profile) | 51.3 \| 13/305 \| -16 / -2 / **+36** | **17.5 \| 0/380 \| 0 / +9 / +1** |
| render window with the recording's **(R, A) energy** (Doppler- and elevation-summed), the render's own D x E structure | 57.9 \| 10/305 \| -26 / +6 / **+42** | 40.9 \| 6/380 \| -6 / 0 / **+42** |
| render window with the recording's **Doppler structure** (per (R, A, E) cell) | 130.3 \| 72/305 \| -23 / -7 / **+42** | 41.3 \| 8/380 \| -5 / +12 / **+42** |

- The +40 cm and the off-target seeds go away when the render's person cells carry the recording's (R, A, E)
  magnitudes, with the render's own Doppler distribution kept (19 / 17 cm, no off-target seed-frame, vertical +2..+3).
  **The model reads the rendered seated person's height and range from the range x azimuth x elevation magnitude pattern;
  its Doppler distribution is not what it reads wrong.**
- In the second period the elevation structure alone is the whole fault (17.5, vertical +1). In the first period the
  recording's elevation structure on the render's per-(D, R, A) energies leaves +36 -- there the render has 1.5-4.7x the
  recording's energy in the moving band |d| >= 3 at the person's peak (`sit.py`, frames 60-75; the mesh slide rendered
  as Doppler), so in those cells the "recording's elevation profile" is the recording's noise floor scaled up: this
  variant is not clean in the first period, and the (R, A, E) variant is the one to read. The (R, A) energy alone
  (the range pattern without the elevation) fixes nothing: the elevation is the necessary part; in the first period the
  range pattern (the shins' cells 0.3 m nearer) comes with it.
- The recording's Doppler structure on the render makes it worse (130): the render's own Doppler distribution is at
  least compatible with a correct reading. The mesh-slide-as-Doppler hypothesis of the task statement is real as a
  signal difference (moving band 1.5-4.7x in 60-75) and is **not what the model reads wrong**.

### 3c. Which part of the scene (`scene_swap_node.py`: the render's person box kept, the scene outside it taken from the
recording in one region; 5 seeds; MRPE over the frames)

| scene region replaced by the recording's | frames 30-90 | frames 195-270 |
|---|---|---|
| none (vb0) | 100.4 | 35.7 |
| everything outside the box | 34.4 | 13.4 |
| near field, x < label - 0.8 m | 104.8 | 34.1 |
| far field, x > label + 0.8 m | 100.4 | 24.4 |
| **the same range band (|x - label| <= 0.8) at other lateral positions** | **50.7** | **20.1** |
| floor voxels z <= -0.9 m outside the box | 100.3 | 35.1 |
| bottom two z rows everywhere incl. the box | 76.2 | 40.0 |

The scene's share is the person's own range band beside the person -- where the rendered person's azimuth sidelobe
streak lies (the narrow-range column across azimuth in every RA frame of the render; the recording's person has a
compact main lobe and no streak) and, in the first period, the corner / far wall cells at the same range (the render's
competitor at (6.4, -2.5) m in every block, 0.71-0.76x the recording there: dimmer, not brighter). The near field, the
far field beyond the person and the floor are not the scene's fault.

### 3d. The model's tensor at the person (`zprof.py`, the person's box +-0.6 m; energy by z and by range rel. label)

Range profiles of recording and render inside the box are alike (frame 60, 0.1 m bands from -0.6: rec 212 153 154 175
165 197 105 80; ren 211 165 144 176 184 223 114 102; the range centroid -0.15 vs -0.13 m rel. label on both -- both
peaks are 0.15-0.25 m nearer than the pelvis label, at the knees / the chair). The z profiles are broad on both (the
sparse elevation array replicates energy over z; centroid +1.0..+1.1 m on both at 7.5 m, +0.4..+0.5 at 5 m); the render's
seated profile peaks one voxel (0.36 m) LOWER (z -0.73..-0.36) than the recording's (0.00..+0.36) at frames 40-70 and
210-250 -- the one systematic tensor-level difference, in the direction opposite to the model's reading, which says
the model's height reading is not the z centroid but the pattern (section 3b).

## 4. What the seated person's elevation signature looks like (`sit.py`, `elcal.py`, `elsign.py`)

- Elevation FFT profile at the DC peak of the person's window (`sit.py`, bins 8..24 of 32, three lobes ~5.5 bins apart,
  0.7-1.0 relative -- the sparse 4-row array's sidelobe structure): the recording's standing person peaks at bin 12
  (frames 10, 80-90: the standing cell) and its seated person at bin 17-18 (frames 40-75, 200-260); the render's person
  peaks at bin 15-16 in both postures. A 5-bin move is 0.31 in sin(el) = 2.4 m of height at 7.7 m: the recording's
  seated / standing difference is not the body's 0.4 m; it is a change of which lobe dominates, and the model, trained
  on recordings, decodes posture from it. The render's seated person shows the render's standing lobe.
- Row phase (`elcal.py`: rows 1 / 4 / 6 vs row 0 of the TI table at the person's DC peak, row 0 restricted to the 58
  azimuth slots overlapping the other rows' 16 so the beams match; sin(el) = phase / pi per half-wavelength row; medians
  over all frames): **164 seated: recording +0.297 (IQR +0.21..+0.38, n 156), render -0.094 (-0.24..+0.01)**; 164
  standing: +0.206 (broad, -0.70..+0.41) vs -0.085; **153 walking: +0.175 vs -0.141 (n 299 each)**; **176 standing
  (indoor): -0.188 (-0.30..+0.03) vs +0.304 (+0.23..+0.32)**. On all three the sign is opposite; the difference is +0.34
  on the outdoor site (164 / 153) and -0.43 indoors (176). The render's 164 value is the geometric one (torso at -0.45 m
  relative to the radar at 7.7 m: -0.06); the render's 176 value (+0.30 = 1.2 m ABOVE the radar for a standing body at
  4.1 m) is not, so "the render is physical, the recording is not" is not the statement either.
- On the strong static corner cell (6.4, -2.5) of 164 / 153 the two AGREE (+0.281 vs +0.262; 153: +0.278 vs +0.285),
  so a fixed per-row phase term (a calibration offset) is ruled out as the explanation; at the mid-range boresight
  cells of 164 (3.8-5 m, the floor / the low structure) they disagree in sign (rec +0.3..+0.5, ren -0.1..-0.4;
  `elsign.py`, frame 130). The difference is site- and target-dependent.
- The render-only probe `elm` (the workspace's cascade config with every antenna's elevation coordinate negated, azimuth
  untouched; vb0's trained state; `elmirror.sh`) moved the render's person row phase by -0.09 -> +0.11 on 164 and
  +0.30 -> +0.29 on 176 -- a change of +-0.1 or none where a pure geometry mirror would flip the sign -- and the gate
  went 164 41.5 -> **88.4** (15 % frames > 1 m, seated vertical bias unchanged at +39..+40), 176 18.5 -> 20.7. The
  antenna elevation sign is not it, and the render's row phase at the person is dominated by a term that is not the
  antenna geometry (not identified; candidates in section 7).

## 5. The off-target seeds of frames 45-75

Off-target seed-frames: 16 / 28 of 75 in 45-60 / 60-75 (2 / 75 for the real model in 60-75). Where they go (the video
frames 45-75 and the gate dump): the person's mirror azimuth (-17 deg, at 7.7-8.3 m -- the far wall cells there) in
most cases, a near cell (2.5 m, 0 deg) in a few. The trainer's person mask includes the mirror azimuth
(`person_range_angle_mask_torch include_mirrored_angle`, STATIC_SEQS_NEXT 1c), so rendered energy at the mirror is
unconstrained by the person terms; whether the render is brighter than the recording at the mirror cells was not
measured here (it is the first measurement of plan step 4). With the recording's (R, A, E) magnitudes in the person's
window the off-target seed-frames go to 0 / 305 (section 3b), so the off-target seeds are a consequence of the
person's pattern being unreadable, not of an independent competitor: the competitor at (6.4, -2.5) is 0.71-0.76x the
recording's in every block (`blocks164.py`).

## 6. Ruled out, with the measurement

| hypothesis | measurement | verdict |
|---|---|---|
| the rendered seated person's level | window peak 0.95-1.47x, DC 0.75-1.08x per seated block (`blocks164.py`); person box x1.5 / x2 in the tensor: 100 -> 55 / 54 (30-90), 36 -> 32 / 35 (195-270) (`swap_box_node.py`, seeds 0/4) | not the level; the x2 gain is the outlier seeds returning, the biases stay |
| the mesh slide rendered as Doppler (the task's hypothesis) | the render's moving band is 1.5-4.7x the recording's at frames 60-75 (`sit.py`) -- real; the recording's Doppler structure on the render's window: 130 / 41 (worse); the render's own Doppler with the recording's (R, A, E) magnitudes: 19 / 17 (`polar_swap_node.py`) | the Doppler distribution is not what the model reads wrong; the slide's effect on the labels is <= 12 cm (the real seeds' radial bias) |
| the per-joint scattering gains (legs 1.2-2.7 over shoulders 0.36-1.16 on 164) putting the return in the shins | render-only re-renders with uniform gains (`g1`: 44.7) and torso x4 in power over the legs (`gtorso`: 41.9), block pattern, vertical +42 and radial -26..-31 identical to vb0 (`gains.sh`, 5 seeds) | not the gains; the model's height reading does not move with the body's energy distribution |
| the antenna elevation sign (the 180 deg board rotation applied to the render, verified from data for azimuth only) | `elm` render: 164 88.4, 176 20.7; the person's row phase moves +-0.1 or not at all (section 4) | not the sign; the render's row phase is not geometry-dominated |
| the near field, the far wall beyond the person, the floor | scene swaps by region: 104.8 / 100.4 / 100.3 (30-90), 34.1 / 24.4 / 35.1 (195-270) against vb0 100.4 / 35.7 (`scene_swap_node.py`) | not those; the same-range band is (50.7 / 20.1) |
| range alone (7.5-7.8 m near the wall) | 153 at 7.9 m walking: 8 / 9 cm real / render, contrast ~1 on both (`seq153_pair.mp4` frame 250; `blocks164.py` on 153: level 1.00-1.12, contrast ren >= rec) | not the range; the posture |
| the label (mesh) being off the real person | the real seeds' bias per seated block: radial -5..-12, lateral +2..+7, vertical 0 +- 1 cm | the labels sit on the real return to ~10 cm |
| a fixed per-row elevation phase (calibration) term in the recording | the corner cell agrees in both (+0.28 / +0.26); 176's difference has the opposite sign of 164 / 153's | not a fixed term |
| the chair alone | the recording's seated DC peak is the chair cell (7.25, +1.63) in frames 50-70 (`sit.py`), the render's peak is 0.3 m lateral of it at the same range; the (R, A) energy swap (which brings the chair cell's energy) does not fix (57.9 / 40.9) | the chair's level (0.42x, STATIC_SEQS_NEXT) is not sufficient; its elevation structure would be part of the (R, A, E) pattern |

## 7. Established vs inferred

**Established (measured above):**
1. The mismatch is the seated posture, both periods; standing and walking frames match (section 1).
2. The model reads the rendered seated pelvis +40 cm too high (the standing height) on every seated block, plus a
   15-31 cm range bias toward the radar and off-target seeds in the first period only (section 1, 2).
3. What it reads wrong is the (R, A, E) magnitude pattern of the person's cells, the elevation structure first; the
   Doppler distribution, the level, the gains, the antenna elevation sign, the near / far scene and the floor are not it
   (sections 3b, 6).
4. The scene's share is the same-range band beside the person (the render's azimuth streak and the wall cells at that
   range) (section 3c).
5. The recording's seated person carries an elevation signature (lobe 17-18, row phase +0.30) different from its
   standing one (lobe 12) and from the render's (lobe 15-16, row phase -0.09 in both postures); the difference is not a
   fixed calibration offset (section 4).

**Inferred (consistent with the measurements, not proven):**
- The real-trained model decodes seated vs standing from the elevation signature (the lobe pattern of the 4-row array),
  and falls back to the standing height when the signature is the standing one: +40 cm = the seated -> standing pelvis
  difference.
- The mechanism behind the recording's seated signature is a propagation / scattering term the single-bounce render
  does not produce at the person -- the body's interaction with the seat and the ground beneath it (the recording's
  seated DC peak is the chair's cell; the mid-range low cells disagree in sign too), or the sparse elevation array's
  response to a real extended, partly incoherent body against the render's coherent point set. Which one is not
  established here; multibounce rendering is off the table by the user's rule, so the plan (SEQ164_PLAN.md) goes through
  what the single-bounce model can move -- the elevation content of the person's return as the trainer fits it, in the
  view the consumer reads -- and a measurement that decides whether the recording's signature is reachable there at all.
- The first period's range bias (the render's peak at the shins / knees, 0.25-0.3 m nearer than the pelvis; the
  recording's peak also 0.15-0.25 m nearer but read correctly by the model) is part of the same pattern question, not a
  separate level or gain item (the gain probes did not move it).

## 8. Tools (scratch; to be committed under downstream/tools/static_viz/ with the plan's first step)

`seeds164.py <dir> <seq> <cand> <gate.json>` -- per-block seed decomposition of a gate dump; `zprof.py <cand> <seq>
<frames>` -- model-tensor z / range profiles in the person's box; `polar_swap_node.py <cand> <seq> <lo> <hi>` -- the axis
swaps of 3b (rtpose_hr3d env, `GATE_CKPT`, `GATE_TAG`); `scene_swap_node.py <cand> <seq> <lo> <hi>` -- the region swaps
of 3c (5 seeds); `gains.sh <src> <dst> <seq> <gpu> "<14 gains>"` and `elmirror.sh <src> <dst> <seq> <gpu>` -- render-only
probes (NB: `.work/person/configs` is a symlink to the repo's `configs/`; `elmirror.sh` was corrected to never write
through it after its first run did -- the node's tracked `configs/mmwcas_cascade.json` was restored to HEAD (md5
e206e016) before any other run); `elcal.py <cand> <seq>` / `elsign.py <cand> <seq> <frame>` / `rowphase.py` / `elmono.py`
-- row-phase elevation probes. Probe products on the node: `outputs/{g1,gtorso,elm}/seq164`, `outputs/elm/seq176`,
gates `/data/mmsd/logs/4dps/static2/gate_{g1,gtorso}_gvhmr.p*.json`, `gate_elm_gvhmr.p3/p4.json`, logs under
`/data/mmsd/logs/4dps/static2/probes/`, probe outputs `/tmp/a69_*.txt`.
