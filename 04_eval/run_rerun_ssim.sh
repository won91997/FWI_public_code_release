#!/usr/bin/env bash
# 阶段 4：仅重算 SSIM（已有 pred.npy / gt.npy，不重跑推理）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${PIPELINE_ROOT}/config.env"

"${PYTHON_BIN}" "${PIPELINE_ROOT}/04_eval/rerun_ssim_ten_models.py" "$@"
