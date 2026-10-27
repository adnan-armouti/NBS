# Entry points that are hard to get wrong.
#
# Everything here exists because the long-hand version has cost this project
# time. `docs/SIMPLIFICATION_PLAN.md` argues the general case; the short version
# is that a stale CUDA extension, a PYTHONPATH pointing at another checkout,
# and a missing seed all present as something OTHER than themselves.

PYTHON ?= /home/adnan/.conda/envs/hupr/bin/python
REPO   := $(shell cd $(dir $(lastword $(MAKEFILE_LIST))) && pwd)
ARCH   ?= 8.9

.PHONY: help build-cuda audit check-environment clean-pyc

help:
	@echo "build-cuda         build BOTH CUDA extensions and stamp what they came from"
	@echo ""
	@echo "There is no test target: tests/ was removed in the deployment strip."
	@echo "chain              python -m mm4DPS.chain run <run> <sequence>"
	@echo "audit              report definitions no code references (investigate, do not delete)"
	@echo "check-environment  where the dataset, models and CUDA extensions resolved to"
	@echo ""
	@echo "There is no test target and no scripts/: the chain is Python now."
	@echo "The acceptance run is two chain arms at --epochs 10 plus the four"
	@echo "media folders. See docs/DEPLOY_LEAN.md."
	@echo ""
	@echo "PYTHON=$(PYTHON)"
	@echo "REPO=$(REPO)"

# BOTH extensions. hupr_cuda alone is not enough: mm3DPS_cuda is required by
# splat_scatter, a chain dies four stages in without it, and the environment
# check that used to catch this went with qa/.
build-cuda:
	@PYTHONPATH=$(REPO) TORCH_CUDA_ARCH_LIST=$(ARCH) $(PYTHON) -m mm4DPS.cuda_ext build

audit:
	@PYTHONPATH=$(REPO) $(PYTHON) -m mm4DPS.qa orphans --methods

# Back after the strip removed it. A stale CUDA .so presents as a pybind
# signature error rather than a build error, and a PYTHONPATH pointing at
# another checkout presents as a missing file: both are visible here.
check-environment:
	@PYTHONPATH=$(REPO) $(PYTHON) -m mm4DPS.qa check-environment

clean-pyc:
	@find $(REPO)/mm4DPS -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@echo "removed __pycache__ (the built extension and its stamp are left alone)"
