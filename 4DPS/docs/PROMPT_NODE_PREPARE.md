# Prompt 2: prepare the node for the downstream experiments (no training)

Set up the server to run the downstream experiments of
<REMOTE_ROOT>/4DPS/docs/DOWNSTREAM_PLAN.md as decided (docs/PROMPT_DOWNSTREAM_A100.md
is the run sheet; downstream/README.md documents the code; the repository
is installed per docs/PROMPT_NODE_INSTALL.md and its environment variables
are exported). Already on the server: the rendered products at
<REMOTE_ROOT>/4DPS_products/{hupr,rtpose} (the rtpose half may still be
transferring: ~4 h), HuPR at $HUPR_DATASET_ROOT (radar/*.zip, annotations/,
frames.zip) and RT-Pose at $RTPOSE_DATASET_ROOT (sequences/<S>/radar/bin,
Train.json, filemeta.json, calib/).

Rules: report every count and every gate with its numbers, never "it
passed"; do not start any training run; for any transfer use
`rsync --archive --partial --info=progress2` under nohup with a log, never
`--delete`, and give me the resume command; commit to main of the 4DPS
checkout after each gate with a message that states the gate and its
result (show me `git status` first).

## 1. Verify what is there before doing anything else

```bash
ls $DOWNSTREAM_PRODUCTS/hupr/seq*/metrics.json | wc -l                     # 235
ls $DOWNSTREAM_PRODUCTS/rtpose/seq*/metrics.json | wc -l                   # 101 (seq169 is a FAILED folder without one; manifest.csv lists the 101 ok)
for f in $DOWNSTREAM_PRODUCTS/*/seq*/*_rendered.npy; do [ -f "${f%.npy}_scale.npy" ] || echo "no scale: $f"; done   # nothing
ls $HUPR_DATASET_ROOT/annotations/hrnet_annot_{train,val,test}.json $HUPR_DATASET_ROOT/radar/*.zip | wc -l      # 3 + 28
ls $RTPOSE_DATASET_ROOT/{Train.json,filemeta.json} $RTPOSE_DATASET_ROOT/calib
$DOWNSTREAM_HUPR_PY -c "
from downstream.common.split import Split; import os
s = Split.load('rtpose'); root = os.environ['RTPOSE_DATASET_ROOT']
missing = [q for q in s.all_sequences() if not os.path.isfile(f'{root}/sequences/{q}/radar/bin/master_0000_data.bin')]
print(len(s.all_sequences()), 'sequences in the frozen split (45/6/13 + 37 two-person); missing radar/bin:', missing)"
```

Report the counts. HuPR's frames.zip is not used by anything downstream.

## 2. Materialise the caches (the real domain does not wait for the products)

Every training reads cubes from $DOWNSTREAM_CACHE in the product form
(float16 pairs + per-frame scale). Real cubes come from the recordings,
rendered ones from the products (HuPR: a slice of the product's bytes;
RT-Pose: a symlink). Run in the `hupr_mscsa` env (it has torch + numpy;
mm4DPS.data needs nothing else), one dataset per GPU, under nohup with logs:

```bash
CUDA_VISIBLE_DEVICES=0 $DOWNSTREAM_HUPR_PY -m downstream.common.domain materialise --dataset hupr   --domain real            # 235 x 2 boards, ~30 min, 52 GB
CUDA_VISIBLE_DEVICES=0 $DOWNSTREAM_HUPR_PY -m downstream.common.domain materialise --dataset hupr   --domain real_nowindow   # the window control, ~20 min, 52 GB
CUDA_VISIBLE_DEVICES=1 $DOWNSTREAM_HUPR_PY -m downstream.common.domain materialise --dataset rtpose --domain real            # 101, ~30 min, 380 GB
# after the rtpose products have fully arrived (their manifest.csv and every *_rendered_scale.npy present):
$DOWNSTREAM_HUPR_PY -m downstream.common.domain materialise --dataset hupr   --domain rendered      # 52 GB
$DOWNSTREAM_HUPR_PY -m downstream.common.domain materialise --dataset rtpose --domain rendered      # symlinks only
$DOWNSTREAM_HUPR_PY -m downstream.common.domain check --dataset hupr;  $DOWNSTREAM_HUPR_PY -m downstream.common.domain check --dataset hupr --domain real_nowindow
$DOWNSTREAM_HUPR_PY -m downstream.common.domain check --dataset rtpose        # 0 MISSING each
```

A run that stops can be restarted with the same command: existing files
are skipped. Report the sizes (`du -shL $DOWNSTREAM_CACHE/*`) and the
`check` counts.

## 3. The gates that need no training (plan section 6.1, 6.2)

Adapter identity, one sequence per dataset, in the `hupr_mscsa` env
(HuPR reads the raw zip of sequence 15 -- 21 s of `generateHeatmap` per
frame in the released numpy; RT-Pose reads the real cube of 183 from the
cache):

```bash
export DOWNSTREAM_SCRATCH=<REMOTE_ROOT>/scratch_gate
CUDA_VISIBLE_DEVICES=0 $DOWNSTREAM_HUPR_PY -c "
from pathlib import Path; from downstream.hupr_mscsa.adapter import HuPRAdapter
r = HuPRAdapter().gate(15, [0, 100, 300, 599], Path('downstream/results/gates/hupr_adapter_seq15_node'))
for b, v in r['boards'].items():
    for k, m in v.items(): print(b, k, {kk: f'{vv:.3g}' for kk, vv in m.items()})"
CUDA_VISIBLE_DEVICES=1 $DOWNSTREAM_HUPR_PY -c "
from pathlib import Path; from downstream.rtpose_hrradarpose.adapter import RTPoseAdapter
import json; print(json.dumps(RTPoseAdapter().gate(183, [10, 150], Path('downstream/results/gates/rtpose_adapter_seq183_node')), indent=1))"
CUDA_VISIBLE_DEVICES=1 $DOWNSTREAM_HUPR_PY -m downstream.rtpose_hrradarpose.gates --sequence 183 --table ti --report-dir downstream/results/gates/rtpose_label_alignment_node
```

Expected (the workstation's, in `downstream/results/gates/` and
docs/DOWNSTREAM_RESULTS.md 1.1-1.3): HuPR `b_window_none` heatmap rel. L2
~1e-6 and `model_input_pearson_nonDC` 1.000 on both boards,
`c_window_none_float16` rel. L2 ~1e-3, `d_hann_float16_production`
magnitude Pearson 0.92/0.93; RT-Pose `"bitwise": true` on both frames,
103,930 valid cells; label alignment r(x) ~0.97, r(y) ~+0.92. Rounding
differs between an A100 and a 4090 at the 1e-7 level, nothing more.

Domain blindness needs both domains of the training split, so it runs
after the rendered caches exist:

```bash
$DOWNSTREAM_HUPR_PY -m downstream.common.runner gate --dataset hupr   --gpus 0   # "passed": true; first batch f2b0a7169deedf02, initial weights 97ceb86ab4bb43fb, 579 steps/epoch
$DOWNSTREAM_HUPR_PY -m downstream.common.runner gate --dataset rtpose --gpus 1   # "passed": true; first batch aa571327c27fe32c, initial weights 1918f63b0bc7a737, 543 steps/epoch
```

The hashes must match the workstation's exactly (they are functions of
the seed, the split and the model definition, not of the data bytes); a
mismatch means the environment changed the model init or the sampler and
is a stop.

Then the two smoke fits of the run sheet's section 3 (one sequence, one
epoch, never tabulated), with `--train-domain real` before the rendered
caches exist and `real+rendered` after:

```bash
DOWNSTREAM_SMOKE_SEQS=15  CUDA_VISIBLE_DEVICES=0 $DOWNSTREAM_HUPR_PY   -m downstream.hupr_mscsa.run        fit --train-domain real --seed 0 --timing --epochs 1 --workers 8
DOWNSTREAM_SMOKE_SEQS=183 CUDA_VISIBLE_DEVICES=1 $DOWNSTREAM_RTPOSE_PY -m downstream.rtpose_hrradarpose.run fit --train-domain real --seed 0 --timing --epochs 1 --workers 4
```

Each must end with an `epoch ...` line; report the seconds per step they
print against the workstation's (HuPR 0.39 s a step of 20, RT-Pose ~0.75 s
a step of 16) -- that is the A100 timing the run sheet's estimates scale by.

## 4. Report

The section 1 counts; cache sizes and `check` results; the three adapter
gates' numbers beside the workstation's; the two blindness JSONs' hashes
and `passed`; the smoke fits' step times. Append all of it to
docs/DOWNSTREAM_RESULTS.md as a "node" subsection under 1.1-1.4 and
commit. Nothing trains until I say so.
