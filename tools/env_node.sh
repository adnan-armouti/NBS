# Source on mmsd-h100 before running anything in NBS:
#   source /data/mmsd/NBS/code/tools/env_node.sh
#
# ~/.4dps_env.sh exports PYTHONPATH=/data/mmsd/4DPS into every shell on the node; without the
# override below `import mm4DPS` would load the main 4DPS checkout instead of this repo's 4DPS/.
export NBS=/data/mmsd/NBS
export PYTHONPATH=$NBS/code/nbs:$NBS/code/4DPS
export PATH=/data/mmsd/envs/nbs/bin:/data/mmsd/envs/micromamba/envs/mm4dps/bin:$PATH   # python (venv overlay), ninja
export CUDA_HOME=/data/mmsd/envs/micromamba/envs/cuda121
export CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12          # nvcc rejects the system gcc 13
export TORCH_CUDA_ARCH_LIST=9.0                          # H100
export OCTONET_DATASET_ROOT=/data/mmsd/datasets/OctoNet/OctoNet-upload
export NBS_DATA_OUT=$NBS/data/OctoNet
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
export TQDM_DISABLE=1
