#!/usr/bin/env bash
# Push main to the node checkout. The node regenerates tracked figure files under viz/ when the viz
# scripts run there, and receive.denyCurrentBranch=updateInstead refuses a push into a dirty tree, so
# discard those regenerated outputs first (the workstation copies are canonical: rsync them back
# before committing).
set -e
ssh -o ControlMaster=no -o ControlPath=none mmsd-h100 'cd /data/mmsd/NBS/code && git checkout -q -- . && git clean -q -fd'   # untracked non-ignored only: ignored dirs (4DPS/seeds, models, __pycache__) survive
git push node "${1:-main}"
