#!/usr/bin/env bash
# 阶段 4：对已有 checkpoint 单独跑 test 集评测
# 用法:
#   bash 04_eval/run_test_single.sh ConvNeXtKaggle /path/to/checkpoint.pth /path/to/eval_out [fair|nonfair]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${PIPELINE_ROOT}/config.env"

MODEL="${1:?模型名，如 ConvNeXtKaggle}"
CHECKPOINT="${2:?checkpoint.pth 路径}"
EXPORT_DIR="${3:?eval 输出目录}"
FAIR_FLAG="${4:-nonfair}"

mkdir -p "${EXPORT_DIR}"

if [[ "${FAIR_FLAG}" == "fair" ]]; then
  export BENCHMARK_FAIR_MODE=1
else
  export BENCHMARK_FAIR_MODE=0
fi

"${PYTHON_BIN}" "${PIPELINE_ROOT}/benchmark/unified_benchmark_test.py" \
  --model "${MODEL}" \
  --checkpoint "${CHECKPOINT}" \
  --data-root "${DATA_ROOT}" \
  --global-map-csv "${GLOBAL_MAP}" \
  --stats-json "${STATS_JSON}" \
  --align-multiple 32 \
  --align-mode crop \
  --batch-size "${BATCH_PER_GPU}" \
  --export-dir "${EXPORT_DIR}" \
  --benchmark-eval "${PIPELINE_ROOT}/benchmark/benchmark_eval.py" \
  --benchmark-eval-json "${EXPORT_DIR}/metrics.json" \
  --benchmark-eval-lpips-mode real

echo "metrics: ${EXPORT_DIR}/metrics.json"
