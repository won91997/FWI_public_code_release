#!/usr/bin/env bash
# 阶段 4：刷新十模型汇总表（从 metrics.json 读取，不重训）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${PIPELINE_ROOT}/config.env"

"${PYTHON_BIN}" "${PIPELINE_ROOT}/04_eval/refresh_all_summaries.py"
echo "汇总表已更新，见 results/ 目录"
