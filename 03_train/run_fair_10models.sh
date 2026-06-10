#!/usr/bin/env bash
# 阶段 3A：Resolution-Constrained 设置训练 + 测试 10 模型（legacy code flag: Fair）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${PIPELINE_ROOT}/config.env"

export WORK_ROOT="${PIPELINE_ROOT}"
export DATA_ROOT
export OUT_ROOT="${OUT_ROOT_FAIR}"
export GPU_IDS SEEDS BATCH_PER_GPU EPOCHS PYTHON_BIN HF_ENDPOINT

bash "${PIPELINE_ROOT}/02_prepare/run_prepare.sh"
bash "${PIPELINE_ROOT}/run_resolution_constrained.sh" "$@"

echo "[$(date '+%F %T')] Resolution-Constrained 十模型训练+测试完成，日志: ${OUT_ROOT_FAIR}"
