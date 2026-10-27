# Results Section 3 on RT-Pose: action plan for the local workstation

Written 2026-09-21. The design this implements is Sec. 6.5 of
`/home/adnan/Desktop/4DPS/latex/writing_review/02_writing_analysis_and_rewrites.md` (the paper text is Sec. 4.6 of
`sec/4_experiments.tex` and the protocol is A.11 of `sec/A_supplement.tex` in the Overleaf clone
`/home/adnan/Desktop/4DPS/latex/6a7f4f567b7cc4b0e392ef61`). Read those three first; this file says how to run it.

Hardware: the workstation's two GPUs are RTX 4090s (24 GB each; `nvidia-smi` on 2026-09-21 showed both idle), not 3090s.
The node (mmsd-h100, 8 GPUs) is busy with the HuPR re-render and is not used here except as the source of files.

## 0. What runs here and what does not

Runs on the existing 15 RT-Pose checkpoints (five seeds x real / rendered / mixed), no downstream training:
error consistency, debiased CKA and Procrustes at every tap in the three available cells of the 2x2 design, head transfer
by linear probe, paired-input localisation, the random-init floor, and (last, optional) the crosscoder.

Needs downstream training, so it is a node job and out of scope for this plan: the same-init real--real control (one
extra real arm per seed, different data order), the positive-control arms (models trained on the unfit render and on 3DPS
renders), full head retraining, and stitching. This plan leaves the code paths for them in place (the sampler's
order-seed argument, the cell type) but does not run them. HuPR waits for its re-render (ETA 2026-09-22 ~13:00 ET).

## 1. Worktree and branch

Two sessions share the 4DPS checkouts, so this work gets its own worktree (memory: parallel-sessions-worktrees).

```bash
cd /home/adnan/Desktop/4DPS-static
git fetch origin
git worktree add /home/adnan/Desktop/4DPS-repr -b analysis/representation fidelity/static-seqs
cd /home/adnan/Desktop/4DPS-repr
ln -s /home/adnan/Desktop/4DPS-static/third_party third_party   # git-ignored; det3d (RT-POSE) lives here
ln -s /home/adnan/Desktop/4DPS-static/models models              # git-ignored; SMPL regressor etc.
```

Code goes under `downstream/repr/` (a new package: `__init__.py`, `taps.py`, `activations.py`, `similarity.py`,
`consistency.py`, `probe.py`, `paired.py`, `controls.py`, `report.py`) plus a driver `downstream/tools/repr_rtpose.py`
and this plan copied to `docs/REPR_ANALYSIS_PLAN.md`. Nothing outside `downstream/` changes except a one-line sampler
argument (Sec. 6). Env: `rtpose_hr3d` (has det3d, torch); check `downstream/envs/build_envs.sh` for how it was built.
Before any push: `git log fidelity/static-seqs..origin/fidelity/static-seqs` to see foreign commits; never push to
`fidelity/static-seqs` from this worktree, only to `analysis/representation`. Merge later by the user.

## 2. Data to pull from the node, into a new folder under /home/adnan/Documents/Data

Everything lands in `/home/adnan/Documents/Data/4DPS_repr/` (1.3 TB free on that disk). The tunnel is ~6 MB/s per
stream; use ~20 parallel rsyncs (memory: brev-transfer-and-dataset-paths). Sizes are from the node on 2026-09-21.

| what | node path | local path | size |
|---|---|---|---|
| 15 checkpoints | `/data/mmsd/4DPS_static2/downstream/results_gvhmr/rtpose/train/{real,rendered,mixed}/seed{0..4}/epoch_50.pth` (+ the `.log`, `.log.json`) | `4DPS_repr/checkpoints/rtpose/<arm>/seed<k>/` | 15 x 20 MB |
| predictions of every evaluation | `/data/mmsd/4DPS_static2/downstream/results_gvhmr/rtpose/*__*/seed*/predictions.json` | `4DPS_repr/predictions/rtpose/` | small |
| blindness gate | `.../results_gvhmr/gates/rtpose_blindness_seed0.json` (also in `mmSD/artifacts/rtpose_static_rerun/20260921_0418/`) | `4DPS_repr/gates/` | tiny |
| test-split cubes, real | `/data/mmsd/4DPS_downstream/rtpose/seq<k>/cascade_real.npy, cascade_real_scale.npy` for k in 1 18 108 141 147 148 159 164 166 168 176 185 191 | `4DPS_repr/cache/rtpose/seq<k>/` | 13 x ~3.5 GB |
| test-split cubes, rendered | `/data/mmsd/4DPS_products_static/rtpose/seq<k>/cascade_rendered.npy, cascade_rendered_scale.npy` (the gvhmr cache symlinks there) | same folder | 13 x ~3.5 GB |
| train-split cubes, real (for the linear probe only) | `/data/mmsd/4DPS_downstream/rtpose/seq<k>/cascade_real*.npy` for the 44 train sequences of `downstream/splits/rtpose_split.json` | same layout | 44 x ~3.5 GB, pull last |
| labels | `/data/mmsd/datasets/RT-Pose/Train_gvhmr.json` (the cache's `Train.json` symlink points here) | `4DPS_repr/labels/Train_gvhmr.json`, linked as `cache/rtpose/Train.json` | small |
| cache metadata | `/data/mmsd/4DPS_downstream_gvhmr/rtpose/{file_meta.txt,hr3d_dzyx_resolved.py}` | `4DPS_repr/cache/rtpose/` | tiny |

Then `export DOWNSTREAM_CACHE=/home/adnan/Documents/Data/4DPS_repr/cache` and
`export DOWNSTREAM_RESULTS=/home/adnan/Documents/Data/4DPS_repr/results` so the existing `downstream.common` loaders
resolve to the local copies (see `downstream/common/domain.py`: `CACHE_ROOT`, `rtpose_labels()`).

Gate before anything else: load `real/seed0` through the existing trainer's evaluate path on the local test cubes and
reproduce R->R MPJPE 16.65 for seed 0 (five-seed mean 16.90 (0.64), `REAL_REAL_gvhmr_summary.json`). If it does not
reproduce, the cache or label copy is wrong; stop and fix before computing anything.

## 3. Taps (HRRadarPose, `RadarPoseNet` in third_party/RT-POSE)

Register forward hooks by module path; store each tap's output per frame as float16:

| tap | module | note |
|---|---|---|
| T0 input features | `reader` (RadarFeatureNet) output | the 64-Doppler-bin cube after the stem |
| T1 | `backbone.backbone.layer1` output | HRNet layer 1 |
| T2a..b | `backbone.backbone.stage2` outputs (one per branch) | one activation matrix per parallel branch |
| T3a..c | `backbone.backbone.stage3` outputs | idem |
| T4a..d | `backbone.backbone.stage4` outputs, if the config has a stage 4 | idem |
| T5 fused | `backbone` output (after `final_conv`) | the tensor the head reads |
| T6 head | `pose_head` pre-activation heatmap and regression maps | last tap |

Activation matrix convention (A.11): spatial positions are samples, a frame contributes D*H*W rows of C channels;
the flattened alternative (one row of C*D*H*W per frame) is computed too and reported in the supplement. Subsample
positions with a fixed seed if a tap's rows exceed ~2e6 per model; record the seed.

The taps for MSCSA-PRGCN (HuPR, later): chirp-network output (RA and RE), `RAradarEncoder.layer{1,2,3}` after the
temporal merge (and RE), the decoder's pre-sigmoid heatmap.

## 4. The 2x2 cells available now

| | same initialisation | different initialisation |
|---|---|---|
| real vs real | not available (needs the extra real arm; node) | real_i vs real_j, 10 pairs |
| real vs rendered | real_k vs rendered_k, 5 pairs (same init, same data order, different data) | real_i vs rendered_j, i != j, 20 pairs |

Report every measurement within each available cell; never pool the 5 same-init pairs with the 20. The claim rests on
the different-init column. The mixed arm is reported as a third model family against real (mixed_i vs real_j) but is
not part of the 2x2.

Pre-registered margin (write it into `results/PREREGISTERED.md` before the first number is computed): the
real--rendered distance is within 1.2x the real--real mean of the same cell, at every tap. Bootstrap: 1,000 resamples
over probe frames and over seed pairs; 95% intervals.

## 5. Steps, in order, with GPU assignment

S0. Setup and gate (Sec. 1, 2). CPU + GPU0.
S1. Error consistency (CPU, minutes): from `predictions.json`, per-frame MPJPE per model on the real test split; for
    every pair of models, the Pearson correlation of per-frame MPJPE and the per-joint agreement (fraction of frames on
    which both models' per-joint error is on the same side of the pair's median). Cells of Sec. 4, plus the mixed arm.
    Output `results/consistency_rtpose.json` and `tables/repr_consistency.tex`.
S2. Activation extraction (GPU0 and GPU1, one model each, ~15 models x 13 test sequences x 299 frames): real probes
    first, rendered probes second, all taps of Sec. 3, float16 to `results/acts/<arm>_seed<k>/<probe>/<tap>.npy`.
    Budget the disk: estimate tap sizes on one sequence before the full run.
S3. Debiased CKA and Procrustes (GPU, hours): unbiased-HSIC linear CKA (Nguyen, Raghu, Kornblith 2021) in minibatches
    and orthogonal Procrustes distance after whitening (Williams et al. 2021), both matrix conventions, every pair in
    every available cell, both probe sets; bootstrap; margin test. Sanity: a model against itself gives CKA 1 and
    distance 0; the random-init floor (S6) sits far below the band. Output per-tap curves with intervals,
    `results/similarity_rtpose.json`, `tables/repr_similarity.tex`, panels (b) of the figure.
S4. Head transfer by linear probe (GPU, needs the 44 real train cubes): freeze each rendered backbone at T5, fit a
    linear map to the head's targets (or a linear head on the pose targets) on the real training split, evaluate on the
    real test split; against the same probe on each real backbone. Output `results/probe_rtpose.json`.
S5. Paired-input localisation (GPU): for each real test frame and its rendered twin (same sequence, same frame index),
    the per-tap feature distance within one model (each real model, then each rendered model), normalised by the
    distance between neighbouring real frames; attributed to input cells by the gradient of the distance w.r.t. the
    input tensor, summed over channels, averaged over frames; shown as maps over range x azimuth and range x Doppler.
    Output `results/paired_rtpose/` and panel (d).
S6. Random-init floor (GPU, minutes): a `RadarPoseNet` built with `set_random_seed(99)` and no training, run through
    S2 and S3 against each real model.
S7. Report: one script writes `tables/repr_*.tex` and `figs/repr_rtpose.pdf` (panels a, b, d now; c, stitching, is a
    placeholder until the node runs it), in the same style as `downstream.common.results` (generated header line,
    protocol hash), and a `results/REPR_SUMMARY.md` with every number the paper's Sec. 4.6 `\todo{}` slots need.
S8. Optional, last: crosscoder at T3 (or T5), BatchTopK with latent scaling, three dictionary seeds, real--real control
    dictionary, decoder-norm-ratio classification, top-activating frames and cells per exclusive feature; ablation
    validation. Only if S1--S7 are done and reviewed.

## 6. The one code change outside `downstream/repr/`

`downstream/common/sampling.py::DomainBatchSampler` takes the cell seed for both the weights (via `set_random_seed`
in the trainer) and the data order. Add an optional `order_seed` (default: the cell seed, so every existing run is
unchanged) and thread it through `Cell` / the runner as an arm variant `real_reorder`, so the node can later train
the same-init real--real control with `set_random_seed(k)` for the weights and `order_seed = k + 1000` for the
batches. Gate: with `order_seed` unset, `first_batch_hash` and `initial_weights_hash` are unchanged for every seed.
Do not run the arm here.

## 7. Rules

One change per commit, in the worktree, on `analysis/representation`. Every number written to a table comes from a
JSON in `results/` with the protocol hash and the git commit of the code that made it. Do not modify the checkpoints,
the caches or anything under `/data/mmsd` on the node. Do not touch the other worktrees (`4DPS`, `4DPS-fidelity`,
`4DPS-solid*`, `4DPS-static`) or their outputs. Use both GPUs but leave a job's VRAM under 22 GB; no job on the node.
Report the S0 gate result before starting S2, and the S3 result before starting S4.
