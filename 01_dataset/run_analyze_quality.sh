#!/usr/bin/env bash
# 阶段 1（可选）：扫描数据集质量统计（全库扫描，约 11044 样本）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${PIPELINE_ROOT}/config.env"

LOG="${1:-${DATA_ROOT}/dataset_quality_report.log}"

echo "[$(date '+%F %T')] 分析数据集: ${DATA_ROOT}"
echo "（脚本内 BASE 默认为 DATA_ROOT，若路径不同请直接编辑 analyze_dataset_quality.py）"
"${PYTHON_BIN}" "${PIPELINE_ROOT}/01_dataset/analyze_dataset_quality.py" 2>&1 | tee "${LOG}"

echo "报告: ${LOG}"
