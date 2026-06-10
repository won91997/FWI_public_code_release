#!/usr/bin/env bash
# Docker entry point for the paper-facing ten-model FAN-10000m benchmark.
#
# Expected container layout:
#   /workspace/repo  -> repository root (this project)
#   /workspace/data  -> dataset root with seismic_full/ and vmodel_full/
#
# Usage inside the container:
#   export DATA_ROOT=/workspace/data
#   export WORK_ROOT=/workspace/repo
#   bash run_benchmark_docker.sh [resolution-constrained|target-resolution|both]
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/workspace/repo}"
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
MODE="${1:-both}"

if [[ ! -d "${DATA_ROOT}/seismic_full" || ! -d "${DATA_ROOT}/vmodel_full" ]]; then
  echo "[ERROR] DATA_ROOT must contain seismic_full/ and vmodel_full/." >&2
  echo "        Current DATA_ROOT=${DATA_ROOT}" >&2
  exit 2
fi

export WORK_ROOT DATA_ROOT
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export SEEDS="${SEEDS:-42}"
export BATCH_PER_GPU="${BATCH_PER_GPU:-2}"
export EPOCHS="${EPOCHS:-100}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${WORK_ROOT}"

echo "[$(date '+%F %T')] Preparing split and train statistics..."
bash 02_prepare/run_prepare.sh

case "${MODE}" in
  resolution-constrained|fair)
    echo "[$(date '+%F %T')] Running Resolution-Constrained benchmark (seed=${SEEDS})..."
    bash run_resolution_constrained.sh
    ;;
  target-resolution|nonfair)
    echo "[$(date '+%F %T')] Running Target-Resolution benchmark (seed=${SEEDS})..."
    bash run_full_resolution_nonfair.sh
    ;;
  both)
    echo "[$(date '+%F %T')] Running Resolution-Constrained benchmark (seed=${SEEDS})..."
    bash run_resolution_constrained.sh
    echo "[$(date '+%F %T')] Running Target-Resolution benchmark (seed=${SEEDS})..."
    bash run_full_resolution_nonfair.sh
    ;;
  *)
    echo "[ERROR] Unknown mode: ${MODE}" >&2
    echo "        Use: resolution-constrained | target-resolution | both" >&2
    exit 2
    ;;
esac

echo "[$(date '+%F %T')] Docker benchmark workflow finished."
