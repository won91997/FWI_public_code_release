#!/usr/bin/env bash
# 单独重跑 DDNet70 和 TU_Net（64×64 公平配置）
# 用法: bash benchmark/run_ddnet_tunet_fair.sh
# Docker: docker run --rm --gpus '"device=0,1,2,3,4,5,6,7"' --shm-size=32g \
#   -v <WORK_ROOT>:/workspace/repo \
#   -v <DATA_ROOT>:<DATA_ROOT> \
#   -w /workspace/repo -e WORK_ROOT=/workspace/repo -e DATA_ROOT=<DATA_ROOT> \
#   fwi-benchmark:cu118 bash benchmark/run_ddnet_tunet_fair.sh

set -e
cd "${WORK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export BENCHMARK_FAIR_MODE=1
export TASK_FILTER=DDNet70,TU_Net
export BENCHMARK_OUTPUT_ROOT="${BENCHMARK_OUTPUT_ROOT:-${WORK_ROOT:-.}/benchmark_logs_fair_seed1997}"
export LOG_ROOT="${BENCHMARK_OUTPUT_ROOT}"
export SEEDS="${SEEDS:-1997}"
export EPOCHS="${EPOCHS:-100}"
export BATCH_PER_GPU="${BATCH_PER_GPU:-2}"

echo "=== 重跑 DDNet70 + TU_Net (64×64 公平配置) ==="
echo "BENCHMARK_FAIR_MODE=${BENCHMARK_FAIR_MODE}"
echo "TASK_FILTER=${TASK_FILTER}"
echo "BENCHMARK_OUTPUT_ROOT=${BENCHMARK_OUTPUT_ROOT}"
echo "SEEDS=${SEEDS}"
echo "================================"

bash run_benchmark_suite.sh
