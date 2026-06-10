#!/usr/bin/env bash
# ABA_FWI、FCNVMB 用前 8 张卡 (0-7) DDP 训练，输出到单独文件夹
# 用法: bash benchmark/run_aba_fcnvmb_gpu0_7.sh
# Docker: docker run --rm --gpus '"device=0,1,2,3,4,5,6,7"' --shm-size=32g -v ... fwi-benchmark:cu118 bash /workspace/repo/benchmark/run_aba_fcnvmb_gpu0_7.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${WORK_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${WORK_ROOT}"

export WORK_ROOT
export DATA_ROOT="${DATA_ROOT:-}"
export BATCH_PER_GPU=2
export SEEDS="${SEEDS:-42}"
export EPOCHS="${EPOCHS:-100}"
# 前 8 张卡
export GPU_IDS="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
# 输出到单独文件夹（与后 8 卡 benchmark_logs 区分）
export BENCHMARK_OUTPUT_ROOT="${WORK_ROOT}/benchmark_logs_gpu0_7"
# 仅跑 ABA_FWI、FCNVMB
export TASK_FILTER="ABA_FWI,FCNVMB"

echo "========== ABA_FWI + FCNVMB 前 8 卡 DDP =========="
echo "GPU_IDS=${GPU_IDS}"
echo "BENCHMARK_OUTPUT_ROOT=${BENCHMARK_OUTPUT_ROOT}"
echo "TASK_FILTER=${TASK_FILTER}"
echo "BATCH_PER_GPU=${BATCH_PER_GPU} SEEDS=${SEEDS} EPOCHS=${EPOCHS}"
echo "=========================================="

bash run_benchmark_suite.sh
