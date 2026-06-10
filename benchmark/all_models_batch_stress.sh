#!/usr/bin/env bash
# Main-eight batch 2 stress test: run 1 epoch per shared-suite model
# 用法: 在 Docker 内执行: bash benchmark/all_models_batch_stress.sh
# 或: docker run --rm --gpus all --shm-size=32g -v ... fwi-benchmark:cu118 bash /workspace/repo/benchmark/all_models_batch_stress.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${WORK_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${WORK_ROOT}"
export WORK_ROOT
export DATA_ROOT="${DATA_ROOT:-}"
export BATCH_PER_GPU=2
export GPU_IDS="${GPU_IDS:-8,9,10,11,12,13,14,15}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export EPOCHS=1
export SEEDS=42

LOG_DIR="${WORK_ROOT}/benchmark_logs"
STRESS_LOG="${LOG_DIR}/batch_stress_all.log"
mkdir -p "${LOG_DIR}"

echo "========== Main-eight Batch 2 Stress Test ==========" | tee "${STRESS_LOG}"
echo "BATCH_PER_GPU=2, EPOCHS=1, SEEDS=42" | tee -a "${STRESS_LOG}"
echo "GPU: $(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader | head -1)" | tee -a "${STRESS_LOG}"
echo "" | tee -a "${STRESS_LOG}"

MODELS=(InversionNet VelocityGAN FuteFWI DCNet DDNet70 TU_Net ABA_FWI FCNVMB)
PASSED=()
FAILED=()

for MODEL in "${MODELS[@]}"; do
  echo "---------- ${MODEL} (batch=2, 1 epoch) ----------" | tee -a "${STRESS_LOG}"
  START=$(date +%s)
  if TASK_FILTER="${MODEL}" bash run_benchmark_suite.sh >> "${STRESS_LOG}" 2>&1; then
    END=$(date +%s)
    echo "[OK] ${MODEL} 通过, 耗时 $((END - START))s" | tee -a "${STRESS_LOG}"
    PASSED+=("${MODEL}")
  else
    echo "[FAIL] ${MODEL} OOM 或报错" | tee -a "${STRESS_LOG}"
    FAILED+=("${MODEL}")
  fi
  echo "" | tee -a "${STRESS_LOG}"
done

echo "========== 压力测试结束 ==========" | tee -a "${STRESS_LOG}"
echo "通过: ${PASSED[*]:-无}" | tee -a "${STRESS_LOG}"
echo "失败: ${FAILED[*]:-无}" | tee -a "${STRESS_LOG}"
echo "详细日志: ${STRESS_LOG}" | tee -a "${STRESS_LOG}"
