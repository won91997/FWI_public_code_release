#!/usr/bin/env bash
# 重启 fair benchmark（与之前 ABA 用的相同配置），但不训练 ABA，其余全部重跑
# 用法: bash benchmark/run_fair_restart_no_aba.sh
#
# 双容器并行：
#   - GPU 0-7:  ConvNeXtFWI, FuteFWI, InversionNet, VelocityGAN, DCNet
#   - GPU 8-15:  DDNet70, TU_Net, FCNVMB

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${WORK_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${WORK_ROOT}"

LOG_ROOT="${WORK_ROOT}/benchmark_logs_fair_seed1997"
DATA_ROOT="${DATA_ROOT:-}"

echo "=== 1. 停止并移除当前 fair 容器 ==="
docker stop fair_seed1997_gpu0_7 fair_seed1997_gpu8_15 2>/dev/null || true
docker rm fair_seed1997_gpu0_7 fair_seed1997_gpu8_15 2>/dev/null || true

echo ""
echo "=== 2. 清理需重跑的 checkpoint（保留 ABA）==="
rm -rf "${LOG_ROOT}/ddnet70_seed1997" "${LOG_ROOT}/tu_net_seed1997" "${LOG_ROOT}/fcnvmb_seed1997"
rm -rf "${LOG_ROOT}/convnext_fwi_seed1997" "${LOG_ROOT}/futefwi_seed1997" "${LOG_ROOT}/inversionnet_seed1997"
rm -rf "${LOG_ROOT}/velocitygan_seed1997" "${LOG_ROOT}/dcnet_seed1997"
echo "已清理 DDNet70, TU_Net, FCNVMB, ConvNeXtFWI, FuteFWI, InversionNet, VelocityGAN, DCNet"

echo ""
echo "=== 3. 启动双容器（公平模式，排除 ABA）==="

docker run -d --name fair_seed1997_gpu0_7 --gpus '"device=0,1,2,3,4,5,6,7"' --shm-size=32g \
  -v "${WORK_ROOT}:/workspace/repo" -v "${DATA_ROOT}:${DATA_ROOT}" -w /workspace/repo \
  -e WORK_ROOT=/workspace/repo -e DATA_ROOT="${DATA_ROOT}" -e BENCHMARK_FAIR_MODE=1 \
  -e TASK_FILTER=ConvNeXtFWI,FuteFWI,InversionNet,VelocityGAN,DCNet \
  -e BENCHMARK_OUTPUT_ROOT=/workspace/repo/benchmark_logs_fair_seed1997 \
  -e SEEDS=1997 -e EPOCHS=100 -e BATCH_PER_GPU=2 \
  -e GPU_IDS=0,1,2,3,4,5,6,7 -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  fwi-benchmark:cu118 bash run_benchmark_suite.sh

docker run -d --name fair_seed1997_gpu8_15 --gpus '"device=8,9,10,11,12,13,14,15"' --shm-size=32g \
  -v "${WORK_ROOT}:/workspace/repo" -v "${DATA_ROOT}:${DATA_ROOT}" -w /workspace/repo \
  -e WORK_ROOT=/workspace/repo -e DATA_ROOT="${DATA_ROOT}" -e BENCHMARK_FAIR_MODE=1 \
  -e TASK_FILTER=DDNet70,TU_Net,FCNVMB \
  -e BENCHMARK_OUTPUT_ROOT=/workspace/repo/benchmark_logs_fair_seed1997 \
  -e SEEDS=1997 -e EPOCHS=100 -e BATCH_PER_GPU=2 \
  -e GPU_IDS=8,9,10,11,12,13,14,15 -e CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
  -e MASTER_PORT=29501 \
  fwi-benchmark:cu118 bash run_benchmark_suite.sh

echo ""
echo "已启动："
echo "  - fair_seed1997_gpu0_7  (GPU 0-7):  ConvNeXtFWI, FuteFWI, InversionNet, VelocityGAN, DCNet"
echo "  - fair_seed1997_gpu8_15 (GPU 8-15): DDNet70, TU_Net, FCNVMB"
echo ""
echo "ABA_FWI 已保留，未重跑。"
echo "查看日志: tail -f ${LOG_ROOT}/run.log"
