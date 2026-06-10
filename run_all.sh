#!/usr/bin/env bash
# 全流程顺序执行（可按需跳过已有步骤）
set -euo pipefail

PIPELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${PIPELINE_ROOT}/config.env"

MODE="${1:-fair}"  # fair | nonfair | both

echo "========== FWI 十模型全流程 =========="
echo "Pipeline: ${PIPELINE_ROOT}"
echo "数据:     ${DATA_ROOT}"
echo "模式:     ${MODE}"
echo "======================================"

# 若数据集已存在可注释掉下一行
# bash "${PIPELINE_ROOT}/01_dataset/run_make_dataset.sh"

bash "${PIPELINE_ROOT}/02_prepare/run_prepare.sh"

case "${MODE}" in
  fair)
    bash "${PIPELINE_ROOT}/03_train/run_fair_10models.sh"
    ;;
  nonfair)
    bash "${PIPELINE_ROOT}/03_train/run_nonfair_10models.sh"
    ;;
  both)
    bash "${PIPELINE_ROOT}/03_train/run_fair_10models.sh"
    bash "${PIPELINE_ROOT}/03_train/run_nonfair_10models.sh"
    ;;
  *)
    echo "用法: bash run_all.sh [fair|nonfair|both]"
    exit 1
    ;;
esac

bash "${PIPELINE_ROOT}/04_eval/run_refresh_summaries.sh"
echo "全流程结束。"
