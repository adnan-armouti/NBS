# Plan — continue NBS on the NVIDIA Brev node (`mmsd-h100`)

> **Done 2026-10-27:** Phases 1–2 executed with the repo relocated to `/home/adnan/Desktop/mmSD/NBS` (GitHub `adnan-armouti/NBS`),
> node copy `/data/mmsd/NBS/code`, venv `/data/mmsd/envs/nbs` (numpy 1.26.4 / open3d 0.19 pinned), 4DPS snapshot in `4DPS/`
> with both CUDA extensions built (ARCH 9.0). Layout names changed: `nbs/` (was `reference_code/NBS`), `4DPS/` (was
> `reference_code/4DPS`). Phase 3 (viz) runs via `nbs/viz/*.py`. Decisions D1–D4 resolved by the user: un-ignore (new repo),
> venv overlay, viz first, PDF in `pdf/`. Phase 3 (viz) and Phase 4 (Task B) done the same day: `viz/README.md`, `4DPS/docs/OCTONET.md`.

> Written 2026-10-27 for the user. Supersedes the "BLOCKER" sections of `md/viz_octonet_prompt.md`
> and `md/port_4dps_prompt.md`: the OctoNet data is not lost, it was moved. Everything below was
> checked live on the node today. Execute Phases 1–2 before running either prompt.

## 0. What changed since the two prompts were written (today, 12:47–12:52)

| Fact in the prompts | Reality on the node (checked 2026-10-27) |
|---|---|
| OctoNet "deleted", must be re-downloaded | **Intact** at `/data/mmsd/datasets/OctoNet/OctoNet-upload/` — 921 GB, 9.4 M files, `.download_complete` marker, `download.log` ends "EXTRACTION COMPLETE" (2026-07-22). 974 Vayyar clips, 976 MoCap clips, `cut_manual.csv`, `mocap_csv_final/mocap_S{1,2,3}`. |
| No disk space anywhere | `/data` 18 TB: **2.5 TB free**; `/ephemeral` 5.9 TB RAID scratch: **4.6 TB free**; root `/` 94 GB free (never put data there). |
| Sensor availability "to be derived" | Read straight from the node dirs: **node_1** IRA acoustic depthCamera mmWave polar seekThermal **uwb** wifi (+ Vayyar, under `vayyar_pickle/`); **node_2** IRA mmWave wifi; **node_3** IRA depthCamera mmWave seekThermal wifi; **node_4** IRA ToF acoustic mmWave wifi; **node_5** IRA depthCamera mmWave. ⇒ UWB only on node_1; WiFi on nodes 1–4 (matches the `(4 nodes, F, 2, 114)` CSI shape); RGB-D on nodes 1, 3, 5 (145 / 150 / 159 GB). |
| Compute = RTX 4090, `ARCH=8.9` | 8 × H100 80 GB, **compute capability 9.0** (`ARCH=9.0`), 208 cores, 1.8 TB RAM page cache. |
| Env `mmir` (numpy 1.26) / `hupr` | Node env `mm4dps`: Python 3.12.14, torch 2.9.1+cu128, **numpy 1.26.4**, scipy 1.16.1, matplotlib, cv2, imageio, trimesh, PIL, yaml, ninja. Missing: open3d, pandas, mitsuba/drjit. numpy 1.x ⇒ the MoCap numpy2 shim in `octonet_io` is still needed and still tested. |
| 4DPS source = workstation `/home/adnan/Desktop/4DPS` | Workstation main = **b51b91f** (4 untracked items, nothing dirty). Node `/data/mmsd/4DPS` = 71da773, an **ancestor** of b51b91f, so the workstation is the newer one. Port from the workstation as the prompt says. |

Still true: the OpenReview PDF (`z3TftXOizf`) is not in `pdfs/`; `octonet_calibrate.py` does not exist;
SFCW cannot see the moving person (WRITE_UP #19 / memory `nbs-vayyar-static-dominated`); none of
SFCW/UWB/WiFi record Doppler, so the range splat suffices and 4DPS is adopted for its preprocessing.

## 1. Phase 1 — move NBS to the node (~30 min, mostly waiting on the tunnel)

Mirror the matprior layout (`/data/mmsd/matprior/{code,data,build,logs}`), which already works on this node.

**Layout**
```
/data/mmsd/NBS/
  code/        git checkout of this repo (receive.denyCurrentBranch=updateInstead), remote `node` on the workstation
  code/reference_code/NBS/     the NBS method code (gitignored today → rsync, see decision D1)
  code/reference_code/4DPS/    the 4DPS port (Phase 2)
  code/viz/    figures (rsync'd back to the workstation viz/ after each run)
  data/        OctoNet-derived products (scaffolds, caches); never on /
  build/       CUDA JIT / ext build cache
  logs/
```

**Commands (workstation → node)**
```bash
# 1a. git home on the node
ssh mmsd-h100 'mkdir -p /data/mmsd/NBS/{data,build,logs} && git init -q /data/mmsd/NBS/code && git -C /data/mmsd/NBS/code config receive.denyCurrentBranch updateInstead'
cd /home/adnan/Desktop/NBS
git remote add node nvidia@mmsd-h100:/data/mmsd/NBS/code
git push node "$(git branch --show-current)"
ssh mmsd-h100 'git -C /data/mmsd/NBS/code checkout -q '"$(git branch --show-current)"' && git -C /data/mmsd/NBS/code log --oneline -1'

# 1b. the gitignored / untracked parts that are actually needed (≈100 MB total; skip the 16 GB 3DPS baselines,
#     2.2 GB 3DPS data, latex, data/XRF55)
rsync -a --info=progress2 -e "ssh -o ControlMaster=no -o ControlPath=none" \
  --exclude '__pycache__/' --exclude 'output/' --exclude 'logs/' \
  reference_code/NBS/  mmsd-h100:/data/mmsd/NBS/code/reference_code/NBS/
rsync -a -e "ssh -o ControlMaster=no -o ControlPath=none" md/ pdfs/ misc/ WRITE_UP.md EXECUTION_PLAN.md PLAN_v2.md \
  mmsd-h100:/data/mmsd/NBS/code/   # (rsync each; or commit them first — see D1)
```
If `reference_code/NBS/output/` holds results the viz prompt should compare against, include it (check `du -sh` first).

**Env on the node** — do not pip into `mm4dps` (it is the 4DPS runtime). Overlay a venv on it:
```bash
ssh mmsd-h100 '/data/mmsd/envs/micromamba/envs/mm4dps/bin/python -m venv --system-site-packages /data/mmsd/envs/nbs \
  && /data/mmsd/envs/nbs/bin/pip install -q open3d pandas \
  && /data/mmsd/envs/nbs/bin/python -c "import torch,numpy,open3d,pandas;print(torch.__version__,numpy.__version__)"'
```
Add mitsuba/drjit only if the 3DPS preprocessing path (`mm3DPS/preprocessing`) turns out to be used; the OctoNet
loaders, `sfcw_forward`, `wifi_forward`, `ra_utils` import only torch/numpy/scipy.

**`tools/env_node.sh`** (new file in the NBS repo, modeled on `/data/mmsd/matprior/code/tools/env_node.sh`). This is
the one non-obvious hazard on this node: `~/.4dps_env.sh` exports `PYTHONPATH=/data/mmsd/4DPS` into **every** shell,
so without an override `import mm4DPS` silently loads the main 4DPS checkout instead of the port.
```bash
# source /data/mmsd/NBS/code/tools/env_node.sh
export NBS=/data/mmsd/NBS
export PYTHONPATH=$NBS/code/reference_code/NBS:$NBS/code/reference_code/4DPS     # overrides ~/.4dps_env.sh
export PATH=/data/mmsd/envs/nbs/bin:/data/mmsd/envs/micromamba/envs/mm4dps/bin:$PATH   # python, ninja
export CUDA_HOME=/data/mmsd/envs/micromamba/envs/cuda121
export CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12          # nvcc rejects the system gcc 13
export TORCH_CUDA_ARCH_LIST=9.0                          # H100 (workstation 4090 was 8.9)
export OCTONET_DATASET_ROOT=/data/mmsd/datasets/OctoNet/OctoNet-upload
export NBS_DATA_OUT=$NBS/data/OctoNet
export MPLBACKEND=Agg OMP_NUM_THREADS=4 TQDM_DISABLE=1
```

**Path rewiring (one small code edit)** — `reference_code/NBS/preprocessing/octonet_config.py:23-25` hard-codes
`OCTONET_ROOT=/data/datasets/OctoNet/OctoNet-upload` and `NBS_DATA_OUT=/home/adnan/Desktop/NBS/data/OctoNet`.
Make both `os.environ.get(..., <old default>)` so the same file works on both machines (and matches 4DPS's
`root_env` convention, which the port will need as `OCTONET_DATASET_ROOT` anyway). Nothing else in the
OctoNet/SFCW code hard-codes a path.

**Smoke test** (proves data + env + shim on the node before any new work):
```bash
ssh mmsd-h100 'source /data/mmsd/NBS/code/tools/env_node.sh && cd $NBS/code/reference_code/NBS && python -c "
from preprocessing import octonet_io as IO, octonet_config as C
idx=IO.load_index(); print(\"rows\",len(idx))
m=IO.load_mocap(sorted(__import__(\"glob\").glob(C.OCTONET_ROOT+\"/mocap_pose/*_skeleton.npy\"))[0]); print(\"mocap\",m[\"positions\"].shape)
v=IO.load_vayyar(sorted(__import__(\"glob\").glob(C.OCTONET_ROOT+\"/vayyar_pickle/*.pickle\"))[0]); print(\"vayyar\",v.shape,v.dtype)"
&& python -m method.sfcw_migration_selftest'
```
Expected: 976 rows, `(F,20,3)`, `(T,400,100)` complex128, self-test 21/21.

## 2. Phase 2 — port 4DPS onto the node (Task A of `port_4dps_prompt.md`, ~1 h incl. CUDA build)

Same guardrail: **never write under `/home/adnan/Desktop/4DPS`**. Copy workstation → node directly (no
intermediate copy on the workstation root disk, which the prompt worried about).
```bash
rsync -a --info=progress2 -e "ssh -o ControlMaster=no -o ControlPath=none" \
  --exclude '.git/' --exclude '__pycache__/' --exclude 'outputs/' --exclude 'downstream/' \
  --exclude 'models/hf/' --exclude 'webpage/' --exclude 'latex/' --exclude 'arxiv_latex/' --exclude 'misc/' \
  --exclude '*.so' --exclude '.build_stamp.json' \
  /home/adnan/Desktop/4DPS/  mmsd-h100:/data/mmsd/NBS/code/reference_code/4DPS/
```
Keeps `mm4DPS/` (22 MB, incl. `cuda/` + `splat/cuda/`), `configs/`, `seeds/` (1.5 GB), `models/smpl/`,
`third_party/` (101 MB), `Makefile`, `README.md`, `docs/`. ≈1.7 GB ⇒ ~5 min on one tunnel stream.
Then on the node:
```bash
source /data/mmsd/NBS/code/tools/env_node.sh
cd $NBS/code/reference_code/4DPS
printf 'Ported 2026-10-27 from /home/adnan/Desktop/4DPS @ main b51b91f (read-only source). rsync excludes: .git outputs downstream models/hf webpage latex arxiv_latex misc *.so. All edits live here.\n' > PORTED_FROM.md
make build-cuda ARCH=9.0 PYTHON=$(which python)      # or: python -m mm4DPS.cuda_ext build
python -c "import mm4DPS, sys; print(mm4DPS.__file__)"   # must print the /data/mmsd/NBS/... path, not /data/mmsd/4DPS
python -m mm4DPS.qa check-environment
```
Verify afterwards on the workstation: `git -C /home/adnan/Desktop/4DPS status --short` still shows only the 4
pre-existing untracked items. Do **not** reuse `/data/mmsd/matprior/code/methods/4DPS` as the source even though it
is the same commit: it carries the matprior `material_model=` modifications.

## 3. Phase 3 — run the viz prompt on the node (first real work; ~half a day)

Run `md/viz_octonet_prompt.md` from `/data/mmsd/NBS/code` with these amendments:
- Delete its BLOCKER section (data is present). Figures go to `$NBS/code/viz/…`; after each run
  `rsync -a mmsd-h100:/data/mmsd/NBS/code/viz/ /home/adnan/Desktop/NBS/viz/` so they land in the repo.
- Node × sensor matrix: cross-check against the dir listing in §0 (UWB only node_1, RGB-D nodes 1/3/5, WiFi 1–4).
- Node **positions** still need the paper: drop the OpenReview PDF into `pdfs/` on the workstation (OpenReview
  blocks curl; the node has internet for HF but no browser). Everything else in #1 and all of #2-person and
  #3-GT panels run now; #2-fused-scene and #3-Rendered stay labelled "pending `octonet_calibrate`" / "pending
  UWB head" exactly as the prompt already says.
- Use the `nbs` env (open3d for the RGB-D back-projection). Heavy: the depthCamera clips are MP4 RGB + PNG depth
  (145–159 GB per camera node); read one clip per camera, not the folder.

## 4. Phase 4 — Task B of the port prompt (multi-day), node-specific notes only

The plan in `port_4dps_prompt.md` §3 stands unchanged (OctoNet `Dataset`/`Radar`, `run_octonet_scaffold`,
MoCap-driven person, SFCW/WiFi heads reusing `sfcw_forward`/`wifi_forward`, new UWB CIR head, no Doppler path).
On the node:
- `Dataset.root_env="OCTONET_DATASET_ROOT"` (already exported by `env_node.sh`); products under
  `/data/mmsd/NBS/data`, scratch under `/ephemeral/scratch` if a stage needs >1 TB of temporaries.
- The 8 H100s let the per-modality self-tests and the first end-to-end SFCW static-scene fit run in parallel;
  set `CUDA_VISIBLE_DEVICES` per job, leave `OMP_NUM_THREADS=4`.
- Keep the SFCW math single-sourced (prompt §B5): the 4DPS copy imports `method.sfcw_forward` from
  `reference_code/NBS` via the PYTHONPATH above rather than forking it.
- Before building any person objective for UWB/WiFi, repeat the detectability test of WRITE_UP #19 on UWB
  (node_1) and WiFi (nodes 1–4) — prompt §6 risk, now runnable.

## 5. Decisions for the user (defaults in bold)

- **D1** `reference_code/` is gitignored, so node-side edits to `reference_code/NBS` and the 4DPS port would not be
  versioned. **Un-ignore `reference_code/NBS/` and `reference_code/4DPS/`** (they are now the development code, not
  reference copies; ~90 MB + ~1.7 GB — exclude `seeds/`, `models/`, `third_party/` from git) so node commits flow
  back with `git pull node`. Alternative: keep rsync-only mirroring.
- **D2** Env: **venv overlay `/data/mmsd/envs/nbs` on `mm4dps`** (torch 2.9.1/cu128, numpy 1.26) vs. recreating
  `environment.yml` (torch 2.7.1/cu118 + mitsuba). Overlay unless the 3DPS mitsuba preprocessing is needed.
- **D3** Order: **viz prompt first** (fast, validates loaders/data/env on the node, grounds the geometry), then the
  port. Both are independent of each other.
- **D4** Drop the OctoNet paper PDF into `pdfs/` (only source of node extrinsics).

## 6. Rollback
`rm -rf /data/mmsd/NBS /data/mmsd/envs/nbs` on the node and `git remote remove node` on the workstation undo
everything; nothing under `/home/adnan/Desktop/4DPS`, `/data/mmsd/4DPS*`, or `/data/mmsd/datasets` is touched.
