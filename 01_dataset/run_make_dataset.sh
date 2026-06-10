#!/usr/bin/env bash
# 阶段 1：从 SEG-Y 制作 OpenFWI 五炮数据集
# 输出: seismic_full/*.npy (N,5,3000,256) + vmodel_full/*.npy (N,256,256)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${PIPELINE_ROOT}/config.env"

OUT_DIR="${1:-${DATA_ROOT}}"
mkdir -p "$(dirname "${OUT_DIR}")"

if [[ ! -f "${VELOCITY_SEGY}" ]]; then
  echo "[ERROR] 请设置速度 SEG-Y 路径: export VELOCITY_SEGY=/path/to/velocity.SEGY"
  echo "        或在 config.env 中修改 VELOCITY_SEGY"
  exit 1
fi

echo "[$(date '+%F %T')] 制作五炮数据集 -> ${OUT_DIR}"
echo "  地震目录: ${SEISMIC_DIR}"
echo "  速度文件: ${VELOCITY_SEGY}"

cd "${PIPELINE_ROOT}/01_dataset"
"${PYTHON_BIN}" make_shot_gather_dataset.py \
  --seismic-dir "${SEISMIC_DIR}" \
  --velocity "${VELOCITY_SEGY}" \
  --out "${OUT_DIR}" \
  --five-channels \
  --allow-partial \
  --depth-range 10000 \
  --receiver-by gx_gy \
  --flip-flop \
  2>&1 | tee "${OUT_DIR%/}/make_dataset.log"

echo "[$(date '+%F %T')] 完成。样本 shape: seismic (N,5,3000,256), velocity (N,256,256)"
