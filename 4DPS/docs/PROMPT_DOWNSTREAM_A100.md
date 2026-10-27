# Prompt: run the downstream experiments (Tables 4-5) on the 8 x A100 server

You are running `downstream/` of the 4DPS repository (docs/DOWNSTREAM_PLAN.md
as decided on 2026-09-15; the code, its gates and the numbers so far are in
downstream/README.md and docs/DOWNSTREAM_RESULTS.md). Nothing in this prompt
is open: every setting is in `downstream/common/protocol.py` and every split
in `downstream/splits/`. Do not change them; if something forces a change,
stop and report it with the reason, because the protocol hash is written
into every result and the table generator refuses to mix hashes.

Rules: linear scale, never dB; name the estimator beside every number;
five seeds with std -- a single-seed number is not a result; report
failures with their output, never "it passed" without the gate's numbers;
the trainings per card come from the runner's `--vram` profile (24: one per card; 80: the measured count, docs/DOWNSTREAM_RESULTS.md 1.5) -- never change the batch; append every measured number
to docs/DOWNSTREAM_RESULTS.md as it lands and commit to main of the 4DPS
checkout after each gated step with a message that states the gate and its
result.

## 1. What has to be on the server

| what | from the workstation | size | where the code looks (env var) |
|---|---|---|---|
| the repository | `/home/adnan/Desktop/4DPS` at main (`git clone` or rsync; `third_party/` included) | 1 GB | -- |
| the cube caches, both datasets, every domain | `rsync -aL /home/adnan/Documents/Data/4DPS_downstream/` (`-L` dereferences the RT-Pose rendered symlinks into real copies) | 155 GB HuPR (real, rendered, real_nowindow) + 708 GB RT-Pose (real + rendered) | `DOWNSTREAM_CACHE` (default `/home/adnan/Documents/Data/4DPS_downstream`) |
| HuPR annotations | `/home/adnan/Documents/Data/HuPR/annotations/` (three JSONs) | 400 MB | `HUPR_DATASET_ROOT` -- a directory that contains `annotations/` |
| RT-Pose labels | `/data/datasets/RT-Pose/{Train.json,filemeta.json}` | 100 MB | `RTPOSE_DATASET_ROOT` -- a directory that contains both files |
| the released MSCSA-PRGCN weights | `downstream/results/hupr/released_weights/{model_best.pth,metrics*.json}` (already scored: \rApMeas 61.8) | 427 MB | in the repo tree (gitignored .pth) |

Alternatively, with the raw recordings, the products and the 4DPS project
environment (mm4DPS + torch) on the node, rebuild the caches there in
about an hour instead of copying them -- every file in them is derived:

```bash
export HUPR_DATASET_ROOT=<HuPR copy> RTPOSE_DATASET_ROOT=<RT-Pose copy> \
       DOWNSTREAM_PRODUCTS=<4DPS_products copy> DOWNSTREAM_CACHE=<where to put it>
<project python> -m downstream.common.domain materialise --dataset hupr                        # ~30 min, 104 GB
<project python> -m downstream.common.domain materialise --dataset hupr --domain real_nowindow  # ~20 min, 52 GB (the control)
<project python> -m downstream.common.domain materialise --dataset rtpose                      # ~30 min, 380 GB + symlinks to the products
```

Either way the trainers import nothing from mm4DPS: the caches are the
cubes in their final form (float16 pairs + per-frame scale); only
`materialise` and the adapter gates need the project environment. Put the caches on
local NVMe; the HuPR arms read 104 GB at random (the page cache should
hold it: 256 GB of RAM or more on the node), the RT-Pose arms 12.6 MB a
frame.

## 2. Environments

```bash
cd <4DPS>
bash downstream/envs/build_envs.sh conda        # hupr_mscsa (torch 2.5.1+cu124) and rtpose_hr3d (torch 2.1.2+cu121)
export DOWNSTREAM_HUPR_PY=<conda envs>/hupr_mscsa/bin/python
export DOWNSTREAM_RTPOSE_PY=<conda envs>/rtpose_hr3d/bin/python
export DOWNSTREAM_CACHE=<the copied 4DPS_downstream>
export HUPR_DATASET_ROOT=<dir with annotations/>  RTPOSE_DATASET_ROOT=<dir with Train.json, filemeta.json>
export PYTHONPATH=<4DPS>
alias python=$DOWNSTREAM_HUPR_PY        # the runner and the table generator run in the hupr_mscsa env
```

`build_envs.sh` records the exact versions; on A100s set
`TORCH_CUDA_ARCH_LIST="8.0"` for the iou3d_nms build. The script also
copies the released `misc/coco.py` / `misc/cocoeval.py` over pycocotools'
(the HuPR README's instruction; the 14-keypoint OKS sigmas live there).
Any version that differs from downstream/README.md must be written into
that table.

## 3. Gates before any training (each takes minutes)

```bash
$DOWNSTREAM_HUPR_PY   -m downstream.common.domain check --dataset hupr     # 0 MISSING (real, rendered; real_nowindow for the control)
$DOWNSTREAM_RTPOSE_PY -m downstream.common.domain check --dataset rtpose   # 0 MISSING
python -m downstream.common.runner gate --dataset hupr   --gpus 0          # domain blindness: must print "passed": true
python -m downstream.common.runner gate --dataset rtpose --gpus 1
DOWNSTREAM_SMOKE_SEQS=15  CUDA_VISIBLE_DEVICES=0 $DOWNSTREAM_HUPR_PY   -m downstream.hupr_mscsa.run        fit --train-domain real+rendered --seed 0 --timing --epochs 1
DOWNSTREAM_SMOKE_SEQS=183 CUDA_VISIBLE_DEVICES=1 $DOWNSTREAM_RTPOSE_PY -m downstream.rtpose_hrradarpose.run fit --train-domain real+rendered --seed 0 --timing --epochs 1
```

Expected from the gates (the workstation's numbers; the hashes must be
identical between the two domains, and the HuPR ones should equal these):
HuPR first batch `f2b0a7169deedf02`, initial weights `97ceb86ab4bb43fb`,
input (20, 8, 8, 2, 64, 64, 8) float32, 579 steps/epoch (ratio 10). RT-Pose: first batch
`aa571327c27fe32c`, initial weights `1918f63b0bc7a737`, input (16, 64, 16, 64, 160)
float32 from (16, 64, 12, 16, 256, 2) float16, 543 steps/epoch. The
smoke fits must end with an `epoch ...` line; their outputs land in
`results/*_timing/` and are never tabulated. Record the two gate JSONs
(`downstream/results/gates/*_blindness_seed0.json`) in
docs/DOWNSTREAM_RESULTS.md 1.4 and commit ("gate: domain blindness on the
A100 node -- passed / failed with ...").

## 4. Step 2 and 3: the real -> real reproductions (five seeds each)

Run both datasets at once on disjoint GPUs; each `runner run` keeps one
training per card and, after each training, scores its checkpoint on every
test domain (real, rendered, the 50/50 mixed test for the real arm, the
RT-Pose two-person set):

```bash
export DOWNSTREAM_VRAM=80      # 80 GB cards: HuPR 1 per card, RT-Pose 2 per card (measured, docs/DOWNSTREAM_RESULTS.md 1.5)
python -m downstream.common.runner run --dataset hupr   --arms real --gpus 0 1 2 3 4 --workers 8 > results/hupr_real.log   2>&1 &
python -m downstream.common.runner run --dataset rtpose --arms real --gpus 5 6 7     --workers 4 > results/rtpose_real.log 2>&1 &
```

(Five seeds: the five HuPR trainings take one card each; the five RT-Pose
trainings share three cards two at a time. Read the real caches once
before launching -- `cat $DOWNSTREAM_CACHE/*/*/*_real.npy > /dev/null` --
so the page cache is warm; a cold first epoch is 10x slower.)

Cost (H100 node, docs/DOWNSTREAM_RESULTS.md 1.5): HuPR 579 steps x 0.207 s +
val 12 s = 2.2 min an epoch, 200 epochs = ~7.4 h a seed; RT-Pose 0.714 s a
step at two per card, 543 steps an epoch (8,677 annotated training frames
/ 16), 50 epochs = ~5.4 h a seed. Both finish in one round in parallel.
(Workstation, RTX 4090: 0.39 s and ~0.75 s a step, 13 h and ~6 h a seed.) Progress: `results/<dataset>/train/real/seed<k>/history.json`
(per epoch: train loss, val metric, seconds) and `.../log`.

**HuPR gate** (protocol.py `extra.gate`): mean R -> R AP over the five
seeds within +/- 2 of the paper's 63.4 (HuPR paper Table 3, "Ours (CSAM +
PRGCN)": AP 63.4, AP50 97.0, AP75 74.0). Report AP, AP50, AP75, PCKh@0.2,
the per-keypoint APs, and the std over seeds, beside the paper's numbers
and beside the released weights' 61.8 (docs/DOWNSTREAM_RESULTS.md 2.2).
Then run the window control once (its cubes are in the cache):

```bash
python -m downstream.common.runner run --dataset hupr --arms real_nowindow --gpus 0 --workers 8
```

and report its AP against the Hann arm's mean: that difference is the
range-window departure's cost, quoted in the paper's methods.

**RT-Pose gate** (protocol.py `extra.gate`): on our test split (13
one-person sequences, `downstream/splits/rtpose_split.json`), mean over
seeds: MPJPE <= 14.9 cm and Abs-MPJPE <= 22.1 cm (1.5x the paper's Table 2:
9.93 / 14.73 on its own full test set) and |MRPE - MPJPE| <= 3 cm (Table 2
has MRPE 9.91 ~ MPJPE 9.93). Report MPJPE, MRPE, Abs-MPJPE, PCK@10/15 cm,
per joint, per activity / location / scenario / occlusion (the evaluator
writes all of them), the two-person set's numbers, and the std over seeds.

If a gate fails: do not start a rendered arm. Write the numbers and the
training curves (history.json) into the results doc, commit, and report.
If it passes: write the numbers, commit ("reproduction: HuPR R->R AP x.x
(std y.y) vs 63.4 -- passed"), and continue.

## 5. Step 4: the rendered and mixed arms

```bash
python -m downstream.common.runner run --dataset hupr   --gpus 0 1 2 3 --workers 8 > results/hupr_matrix.log   2>&1 &
python -m downstream.common.runner run --dataset rtpose --gpus 4 5 6 7 --workers 4 > results/rtpose_matrix.log 2>&1 &
```

The runner skips the finished real arms and trains S (rendered) and R+S
(mixed: the real arm's step count, every batch half real / half rendered)
for seeds 0-4, then scores every checkpoint on every test domain;
`python -m downstream.common.runner matrix --dataset <d>` afterwards
catches any evaluation that is still missing. 6 HuPR trainings on 4 cards
= 2 rounds x 13 h; 6 RT-Pose on 4 cards = 2 rounds x 6 h. The low-shot
curve is not run (decided 2026-09-15); `--low-shot` exists but is not part
of the protocol.

## 6. Step 5: the tables

```bash
python -m downstream.common.results        # results/tables/{table4,table5,supplement,macros}.tex, summary.{md,json}
```

Everything comes from `results/**/metrics.json` alone; the generator
refuses to mix protocol hashes. Paste `summary.md` into
docs/DOWNSTREAM_RESULTS.md sections 4-5, commit, and hand over
`results/tables/` -- `macros.tex` is a list of `\renewcommand`s for the
paper's main.tex macros (`\rApReal`, `\rApSub`, `\rApTSTR`, `\rApSelf`,
`\rMPJPEReal`, `\rMPJPESub`, `\rMPJPETSTR`, `\rMPJPESelf`, `\rApRealOnly`,
`\rApAug`, `\rMPJPEAug`, `\rRetention`, `\rApMeas`; `\rSeqEquiv`, `\rNlow`,
`\rApLowShot` print `--` because the low-shot curve is not run), to be
`\input` after main.tex's macro block; `table4.tex` / `table5.tex` are the
complete tabulars (AP, AP50, AP75; MPJPE, MRPE, Abs-MPJPE) for
sec/4_experiments.tex; `supplement.tex` holds retention, the gap
decomposition, per-keypoint / per-joint tables, PCK, the two-person set,
the mixed test and the per-group RT-Pose breakdown.

## 7. What to report back

For each dataset: the gate's numbers with the tolerance and pass/fail;
the four Table 4 cells and the R+S cell as mean (std, n = 5); retention
and the gap decomposition; the control's AP; anything that deviated from
downstream/README.md (a version, a crash and its fix, a rerun). Attach
`results/tables/summary.md`. Every number in linear units, with its
estimator named (COCO AP over OKS thresholds 0.5:0.95 on the 14 HuPR
keypoints with the released sigmas; RT-Pose MPJPE = pelvis-relative joint
error averaged per sequence then over sequences, in cm, as the released
evaluator averages it).
