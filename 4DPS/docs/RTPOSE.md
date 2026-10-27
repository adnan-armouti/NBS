# RT-Pose through 4DPS

RT-Pose (arXiv 2407.13930) records one TI MMWCAS-RF-EVM cascade radar
(12 TX x 16 RX, TDA2 capture), a Livox Horizon LiDAR and a stereo camera
pair on one plate, at 10 Hz, 299 radar frames a sequence, with 15
Human3.6M joints annotated on the LiDAR. 106 of its 242 sequences have raw
data on this machine (`/data/datasets/RT-Pose`). This document is what the
`--dataset rtpose` path does differently from HuPR, and the facts it rests
on. Every fact below was checked against the data; where a number decides
something the estimator is named.

```bash
export RTPOSE_DATASET_ROOT=/data/datasets/RT-Pose            # default: this path
python -m mm4DPS.chain run <run> 183 --dataset rtpose --artifacts outputs/<run>/artifacts
```

writes the same seven files as a HuPR run, with one radar:

```
outputs/<run>/seq183/{checkpoint.pt, history.json, metrics.json, run.log}
outputs/<run>/seq183/cascade_rendered.npy (+ _scale.npy)  (299, 64, 12, 16, 256, 2) float16 at a per-frame scale, 3.8 GB
artifacts/seq183_maskon/  the micro-Doppler figure, the RA and RD videos,
                          seq183_mesh_on_rgb.mp4, seq183_scene_and_person.mp4
```

## The radar description

`mm4DPS/__init__.py` holds `RTPOSE`, a `Dataset` with one `Radar`
("cascade"): chirp from the RT-POSE repository's `hardware_param.m`
(256 samples at 5 MHz, slope 64.985 THz/s, 77 GHz start, 60 us ramp + 5 us
idle, 64 loops x 12 TX, 10 Hz); `tx_order = (11, ..., 0)` because
`TxToEnable = [12 .. 1]`; the nine TX at elevation 0 form the azimuth
array; TI's antenna table in *recorded* channel order (master 0-3,
slave1 4-7, slave2 8-11, slave3 12-15 -- `TI_Cascade_RX_position_azi`
permuted by `TI_Cascade_RX_ID`), identical to ColoRadar's
`antenna_cfg.txt`, in half-wavelengths of 76.8 GHz; `board_rotation_deg
= 180`; `n_angle = 128`.

`configs/mmwcas_cascade.json` is built from it (`python -m mm4DPS.qa
cascade-config`); `configs/mmwcas_calibration.npz` holds the `RangeMat` and
`PeakValMat` of TI's `calibrateResults_high.mat`.

**The board orientation is checked from data, not the photograph.**
`python -m mm4DPS.qa orientation-check` beamforms every annotated frame
of a walking sequence and correlates the person's measured `sin theta_X`
(top-1% centroid of the clutter-removed range-azimuth map, `sin = (N/2 -
bin)/(N/2)` -- the sign the HuPR boards settled from data) with the
annotation's `-y_left / range`. With the config as written: seq 183
r = +0.985 (n = 163), seq 186 +0.952 (n = 135), seq 153 +0.996 (n = 101).
The mirror would read -0.95. The return lands on the person.

## Decoding and calibration

`data.CascadeRecording` reads the four `*_data.bin` device files a frame
at a time (frame count from the `*_idx.bin` header). `decode_tda2_frame_torch`
undoes TI's `readBinFile` layout -- I/Q interleaved int16, (rx, sample, slot,
loop) column-major -- into the same `(rx, chirp, sample)` frame the DCA1000
decode produces, so `tdm_split` and every beamformer after it are shared.
`cascade_calibration` is TI's `calibrationCascade` (range-offset phase ramp
and phase correction per TX/RX, `phaseCalibOnly = 1`) as one diagonal
`(rx, slot, sample)` multiplier: the calibrated ADC still inverts exactly
from the shipped cube.

The beamformer places the 9 x 16 azimuth virtual channels on the
half-wavelength grid (86 slots, channels sharing a slot averaged) and
takes a 128-point angle FFT (`Radar.azimuth_grid`, `forward_render.
_virtual_channels`, `data.virtual_to_grid`). For HuPR the grid is the
identity and the transform is the tensor it always was.

## The extrinsic

RT-Pose ships no radar extrinsic. `seeds/rtpose/radar_extrinsic.json`
seeds it with the axis mapping (LiDAR x forward, y left, z up -> radar
+Y, -X, +Z), identity rotation and zero translation, and records why the
translation is zero to +/-0.03 m laterally: on the clean outdoor track
(seq 153) the LiDAR's own person points lie 0.20 +/- 0.02 m LEFT of the
annotated centroid (n = 34 frames; mean of LiDAR points within 0.7 m of
the annotated column) while the radar's person track puts the radar
0.22 m RIGHT of the annotation origin (n = 101; least squares over range
and sin of the top-1% RA centroid against the annotated torso, rms range
0.057 m, rms sin 0.017). The two cancel: the radar is laterally at the
LiDAR, and the annotation frame is a third origin 0.2 m to the left
(`RTPoseSequence.ANNOTATION_TO_LIDAR`), presumably the radar frame as
RT-POSE's own unreleased calibration had it. The indoor tracks are too
noisy to add to this (rms sin 0.09). The rig photo's "radar left of the
LiDAR" is the box, not the array.

A translation search of the LiDAR occupancy against the measured static
map exists (`lidar-scaffold --fit-alignment`) and is off: the measured
map's azimuth structure is the sidelobe smear of a few specular walls,
and the search moved the correlation from 0.189 to 0.235 while walking
the height to the edge of its range. HuPR found the same.

## The scene: LiDAR + RGB

`reconstruction lidar-scaffold`: eight annotated frames spread over the
sequence; per frame the Livox points AND a dense cloud from the left
image's Depth-Anything metric depth, both with the annotated person's
column (0.6 m) cut out, merged at 3 cm, placed by the seed.

The LiDAR alone is sparse and its pattern repeats (8 frames: 15k points
at 2 cm; 40 frames: 18.8k), so the floor, walls and ceiling stay as scan
lines. Depth-Anything is dense but not metric-consistent here: on seq 183
its back wall runs 6.2 -> 7.2 m across the image where the LiDAR reads a
flat 6.1 m, and its near floor is 34% too far. So the depth is **pinned
to the LiDAR as a field**: the LiDAR points are projected into the image
(6.8k of them land in it), the ratio LiDAR / predicted depth is taken at
each, its medians on a 12 x 9 image grid (empty cells from the nearest
filled one) are interpolated bilinearly to every pixel, and the depth map
is multiplied by that. After it the depth cloud's wall sits within a few
cm of the LiDAR's across the room and its floor at the LiDAR's -0.75 m;
the residual at the LiDAR pixels is +/-0.5% (IQR). One scalar left the
wall tilted 8.7 degrees against the LiDAR's 2.4; a plane in (u, v) did not
fix it either -- the error is a field. Result on seq 183: 133k points,
98% facing the radar (|cos| > 0.05; wall 0.82, ceiling 0.74, floor 0.35
grazing), background fit loss 0.39 with 19.4k active points (HuPR seq
256: 0.42). `--no-rgb-fusion` gives the LiDAR-only scaffold, which needs
`--normal-radius-m 0.5`: the scan lines run radially about 0.4 m apart on
the far wall, and any PCA neighbourhood inside one line gives a normal
perpendicular to the line of sight that `cull_gaussians` then drops
(median |cos| 0.007 at HuPR's 0.18 m; 1,565 of 15,081 points survived,
loss 0.86).

The measured static map is the sampled frames' calibrated |RA| median.

## The person

GVHMR (`~/Desktop/GVHMR`, env `gvhmr`, `tools/rtpose_dump.py`) on the left
camera's frames as a 10 Hz video, with the true focal length
(`--f_mm 25.6545`: 1067.29 px on the 1440 x 1080 centre crop of the
calibrated 1468 x 1128 sensor). `reconstruction person-sequence` (now in the chain, for both datasets)
then builds what the chain consumes: for radar frame `f` the GVHMR row of
the camera image the rig's clock puts at `f` (`RTPoseSequence.camera_index`:
the file stems are master-clock ticks, `3 f - 4` on 85 of the 102 runnable
sequences -- three camera and LiDAR frames a radar frame, 457-509 files
for 299 radar frames -- and `2 f - 2` on the 17 others such as 183, 186,
153; fitted on the annotation's pairs and exact on every one of them. The
constant "index + 2" the three development sequences suggested held only
on the 2-tick sequences: on the others it fed the person stage a camera
frame up to 20 s off, which the per-frame registration then "fixed" with
depth scales of 0.6-2.0 and a 3.5 m tall person -- caught on the first
sweep sequences, 2026-09-13),
camera -> LiDAR by the shipped calibration, LiDAR -> radar by the seed;
GVHMR's depth rescaled by the radar -- the median over annotated frames of
(radar top-1% RA centroid range) / (GVHMR centroid range): 1.09, 1.12,
1.14 on 183 / 186 / 153, GVHMR's size prior placing the body about 14%
short (0.855-0.869 of the annotated range). The scaling is about the
camera origin, so every vertex keeps its pixel and the body comes out k
times larger as well as farther, which is what the error is; a
translation along the centroid's ray moved off-centre vertices ~12 px.
Then HuPR's 14 joints, the joints in the image, the single-loop measured
targets `cascade_first_loop_ra.npy` (299, 256, 128), the scene, and
`compute-frame-visibility` in place. Nothing reads the annotation's joints
for geometry; they pick frames and cut the person out of the scene.
`mesh-on-rgb` draws the SMPL faces back through the same calibration onto
the left camera's frames as the check.

## What the chain does differently

| stage | HuPR | RT-Pose |
|---|---|---|
| scaffold | Depth-Anything on four frames, the fitted alignment table | `lidar-scaffold` |
| absolute gain | shipped seed, fitted on held-out sequences | `train_motion absolute-gain`: median measured / rendered static \|RA\| over the map's brightest decile (16.9 on 183) |
| seeds root | `seeds/` | `seeds/rtpose/` (the person sequences, the extrinsic) |
| scene alignment | bake the table into the checkpoint | none: the scaffold is built in the radar frame |
| noise model | the same for both: the trainer's learned floor, its prior this sequence's own off-person windowed \|RD\|², carried in the checkpoint and drawn by the renderer (`docs/SEEDS.md`) | the same |
| RadProposer products, two-panel video | yes | no (HuPR's published tensor; HuPR's two boards) |
| trainer | 24 frames a step, 420 of 600 DCT bands | 4 frames a step (4 x 144 sub-array channels), 209 of 299 bands (the same 3.5 Hz cap) |
| Doppler-plane support | HuPR's median-over-loops floor, the band in every view (33 of 63 planes) | the same rule (62 of 63 planes: the cascade's finer Doppler puts the person's rows above the median almost everywhere) |
| motion Doppler band | the released bins 24-39 (+/-1.1 m/s at 0.141 m/s a bin) | the same velocities: bins 3-58 at 0.039 m/s a bin (`signal_model.MOTION_BAND_MPS`) |
| range calibration search | -1.40..+0.20 m (monocular depth places the person far) | -1.40..+1.40 m |
| spectrograms | the RadProposer tensor | the shipped cubes, DC zeroed, range shifted, forward-normalised |
| artifacts | velocity comparison + wall-band occlusion check | velocity comparison + `mesh-on-rgb` + `scene-and-person` |

`scene-and-person` draws, per frame, LEFT the scene and person points
with no visibility and RIGHT the same points with the run's own shadow
mask (`visibility/seq<S>_s2_shadow.npy` over the trained background),
masked points dim rather than dropped. Everything up to placement is
shared between the panels; the mask is the only difference.

## The motion Doppler band is a velocity band

HuPR's released "central 16 Doppler bins" (24-39) are +/-1.1 m/s on its
216 us loop. The cascade's loop is 780 us, 3.6x finer in Doppler (0.039 m/s
a bin, +/-1.25 m/s unambiguous), so the same 16 bins there are +/-0.31 m/s
and the walking person sits outside them: measured on seq 183, walking
away at +0.57 m/s the person is at bin 45, coming back at -0.55 m/s at bin
18, with a spread of 3-6 bins. With the band applied in bins the
micro-Doppler term saw an empty band and, over 400 epochs, emptied it by
smearing the person over Doppler (rendered spread 15 bins against 3.9,
centroid r +0.15; joint velocities to 10 m/s). The band is now defined in
velocity (`MOTION_BAND_MPS`) and converted per radar: HuPR's bins are the
same 13 as before, the cascade's are 53.

## Training time

The velocity stage on seq 183 took 3,350 s at 400 epochs against HuPR's
450-500 s: the cascade renders 8x HuPR's channels (192 against 24) and,
under HuPR's median-over-loops support, 62 of 63 Doppler planes in every
view against HuPR's 33. The step's cost is in the Doppler-plane splat
kernels -- 53% of the CUDA time in `doppler_plane_splat_backward`, 11% in
the forward, 5% in the amplitudes -- and scales with paths x planes, so
the changes act on those two factors and on nothing the losses read:

| change | what | seq 183, 10 epochs, steady-state ms a step (s an epoch) |
|---|---|---|
| training sub-array | 9 of 12 TX rendered (the beamformer keeps only those); HuPR 2 of 3 | 12 TX, 3 a step: 79.7 (8.0) -> 9 TX, 4 a step: 94.6 (7.10) |
| per-view energy support (`--doppler-support-floor energy 0.99`, band off) | each view renders only the bins holding 99% of its person Doppler energy (median 27 planes, widest 51) | 94.6 (7.10) -> 88.3 (6.62) -- **rejected**: at 400 epochs the unrendered bins are unsupervised and the person smears over Doppler (rendered spread 13.3 bins against 4.5, centroid r +0.62 against +0.98) |
| kernel-side plane skipping | one captured graph at the widest view; the Doppler-plane kernels stop at each view's own plane count (bitwise the same outputs: the skipped planes were zero-weight pads) | with the energy support 88.3 (6.62) -> 69.1 (5.18); with the retained loops-median support the views are 54-62 planes wide, so the saving is small |
| angle FFT at the grid length (86 for 128) | measured 100 against 98 ms: no gain, not adopted; the trainer keeps the radar's `n_angle` | -- |

HuPR seq 256 (the sub-array alone; its support and band are unchanged):
41.5 -> 28.4 ms a step, 1.04 -> 0.71 s an epoch, the first step's loss
the same to five figures. Estimator throughout: `step_ms` from the
trainer's epoch line, mean of epochs 6-10 (no re-capture), one run per
GPU. The retained state is therefore the sub-array with the full
support: ~95 ms a step, ~7.1 s an epoch, ~3,050 s for 400 epochs with
80 evaluations on seq 183. What remains is the Doppler-plane term's
arithmetic itself; `docs/TRAINING_SPEED_PLAN.md` is the plan for it.

## Sequences

The first pass uses three single-person sequences: 183 (indoor, clean,
walk), 186 (indoor, clean, walk and wave hand) and 153 (outdoor, walk).
