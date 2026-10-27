# Prompt 1: install 4DPS on the A100 node

Install the 4DPS repository on this server for the downstream experiments
of docs/DOWNSTREAM_PLAN.md (docs/PROMPT_DOWNSTREAM_A100.md is the run
sheet; downstream/README.md the code's own documentation). Nothing here
trains anything. Report every command's outcome with its output; never
"done" without the check that shows it.

## 1. The code

```bash
cd <REMOTE_ROOT>
git clone git@github.com:adnan-armouti/4DPS.git 4DPS        # main at a537b0e or later
```

`third_party/` is git-ignored in this repository (`.gitignore` line 46),
so it does not arrive with the clone. Copy it from the workstation:

```bash
nohup rsync --archive --partial --info=progress2 \
  --exclude 'build/' --exclude '*.so' --exclude '__pycache__/' \
  adnan@<WORKSTATION>:/home/adnan/Desktop/4DPS/third_party/ <REMOTE_ROOT>/4DPS/third_party/ > rsync_third_party.log 2>&1 &
```

(95 MB: `hupr_official`, `RT-POSE`, `3DPS_submission`; the excluded `.so`
and `build/` are the workstation's compiled extension, rebuilt below.
The same command resumes an interrupted copy.) Then check:

```bash
ls third_party/hupr_official/{main.py,config/mscsa_prgcn.yaml,preprocessing/process_iwr1843.py,misc/cocoeval.py}
ls third_party/RT-POSE/{tools/train.py,configs/cruw_pose/hr3d.py,data_processing/4Dradar2xyz.py,det3d/ops/iou3d_nms/setup.py}
ls configs/{iwr1843_horizontal.json,iwr1843_vertical.json,mmwcas_cascade.json,mmwcas_calibration.npz}   # tracked; must be there
ls downstream/splits/rtpose_split.json downstream/results/gates/                                          # the frozen split and the workstation's gate numbers
```

The released MSCSA-PRGCN weights are git-ignored too: copy
`downstream/results/hupr/released_weights/` (427 MB, `model_best.pth` +
its three `metrics*.json`) with the same rsync form; `metrics.json` there
is what fills `\rApMeas`.

## 2. The environments

```bash
cd <REMOTE_ROOT>/4DPS
CUDA_HOME=<a CUDA 12.x toolkit> TORCH_CUDA_ARCH_LIST="8.0" bash downstream/envs/build_envs.sh conda
```

builds `hupr_mscsa` (python 3.10, torch 2.5.1+cu124, numpy 1.23.5,
pycocotools 2.0.8 with the released `misc/coco.py` / `misc/cocoeval.py`
copied over it) and `rtpose_hr3d` (python 3.9, torch 2.1.2+cu121,
spconv-cu120, and the `det3d/ops/iou3d_nms` extension built in place --
the released import chain needs it; `det3d/ops/dcn` does not build against
torch 2.x and is unused). If any version resolves differently, write the
version that did into downstream/README.md's table and say so.

There is no separate project environment to build: `mm4DPS.data`
(ProductCube, MeasuredReader, the calibration helpers, the Hann range FFT)
imports only numpy and torch and needs none of mm4DPS's CUDA extensions --
verified on the workstation in the `hupr_mscsa` env. Use `hupr_mscsa` for
`downstream.common.domain materialise` and for the adapter gates as well.
If `from mm4DPS.data import ProductCube, MeasuredReader` tries to build or
load a CUDA extension here, stop and report it rather than working around it.

Put these in the shell profile the jobs will use, then verify each:

```bash
export HUPR_DATASET_ROOT=<REMOTE_HUPR_ROOT>           # contains radar/*.zip and annotations/
export RTPOSE_DATASET_ROOT=<REMOTE_RTPOSE_ROOT>       # contains sequences/, Train.json, filemeta.json, calib/
export DOWNSTREAM_PRODUCTS=<REMOTE_ROOT>/4DPS_products
export DOWNSTREAM_CACHE=<REMOTE_ROOT>/4DPS_downstream  # ~863 GB when complete; local NVMe
export DOWNSTREAM_HUPR_PY=<conda envs>/hupr_mscsa/bin/python
export DOWNSTREAM_RTPOSE_PY=<conda envs>/rtpose_hr3d/bin/python
export PYTHONPATH=<REMOTE_ROOT>/4DPS
export TQDM_DISABLE=1
```

## 3. Import checks (each must print without a traceback)

```bash
$DOWNSTREAM_HUPR_PY -c "from mm4DPS.data import ProductCube, MeasuredReader, range_profiles_all_channels_torch; print('mm4DPS.data ok')"
$DOWNSTREAM_HUPR_PY -c "
import sys; sys.path.insert(0, 'third_party/hupr_official')
from models import HuPRNet; from tools import Runner; from pycocotools.cocoeval import COCOeval
import numpy as np; assert hasattr(np, 'float'); print('hupr_official ok')"
$DOWNSTREAM_RTPOSE_PY -c "
import sys; sys.path.insert(0, 'third_party/RT-POSE')
from det3d.torchie import Config; from det3d.models import build_detector; from det3d.datasets import build_dataset
cfg = Config.fromfile('third_party/RT-POSE/configs/cruw_pose/hr3d.py'); m = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
print('det3d ok, params', sum(p.numel() for p in m.parameters()))"      # 2,002,194
$DOWNSTREAM_HUPR_PY -c "from downstream.common.protocol import HUPR, RTPOSE; print(HUPR.hash, RTPOSE.hash)"   # 38ea71937b32 989ccef2230e (five seeds; 8525a1813438 f585e9aeb1b4 were the three-seed hashes)
$DOWNSTREAM_RTPOSE_PY -c "from downstream.rtpose_hrradarpose.adapter import RTPoseAdapter; a = RTPoseAdapter(); print(a.input_shape, int(a.valid.sum()))"   # (64, 16, 64, 160) 103930
```

The two protocol hashes must equal the workstation's; if they differ the
Protocol text changed in transit and nothing may run until it is found.

## 4. Report

The clone's commit, the rsync log's last line, the versions actually
installed (`pip list` of both envs for torch, numpy, pycocotools, spconv),
the five import checks' outputs, and the exported paths.
