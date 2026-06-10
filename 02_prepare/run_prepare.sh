#!/usr/bin/env bash
# 阶段 2：生成 train/val/test 划分 + 训练集归一化统计
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${PIPELINE_ROOT}/config.env"

mkdir -p "${SPLIT_DIR}"

if [[ ! -d "${DATA_ROOT}/seismic_full" ]]; then
  echo "[ERROR] 数据集不存在: ${DATA_ROOT}/seismic_full"
  echo "        请先运行: bash 01_dataset/run_make_dataset.sh"
  exit 1
fi

if [[ ! -f "${GLOBAL_MAP}" ]]; then
  echo "[$(date '+%F %T')] 生成 split (global_map.csv)..."
  "${PYTHON_BIN}" "${PIPELINE_ROOT}/benchmark/make_split_from_inline.py" \
    --data-root "${DATA_ROOT}" \
    --out-dir "${SPLIT_DIR}"
fi

if [[ ! -f "${STATS_JSON}" ]]; then
  echo "[$(date '+%F %T')] 计算 train stats..."
  "${PYTHON_BIN}" "${PIPELINE_ROOT}/benchmark/compute_train_stats.py" \
    --global-map-csv "${GLOBAL_MAP}" \
    --out-json "${STATS_JSON}"
fi

echo "[$(date '+%F %T')] 准备完成:"
echo "  ${GLOBAL_MAP}"
echo "  ${STATS_JSON}"
