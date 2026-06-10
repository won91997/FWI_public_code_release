#!/usr/bin/env bash
# 全量 Benchmark 发射前 Checklist（在 Docker 内执行）
# 用法: docker run --gpus all --shm-size=32g ... fwi-benchmark:cu118 bash /workspace/repo/benchmark/pre_launch_checklist.sh
# 注意：--shm-size=32g 必须，否则 NCCL 会报 "Error while creating shared memory segment"
# 镜像构建: docker build -t fwi-benchmark:cu118 -f benchmark/docker/Dockerfile . （详见 benchmark/docker/BUILD_AND_RUN.md）
# 国内拉取: docker.1ms.run/ 前缀
#
# 重要：所有测试统一 8 卡 DDP 真实环境，不使用单卡。
# VelocityGAN 不使用 --sync-bn：Wasserstein_GP 的 gradient penalty 需要 create_graph=True（二阶梯度），
# 与 SyncBatchNorm 的 backward 不兼容，会触发 "derivative for batch_norm_backward_elemt is not implemented"。

set -e
cd /workspace/repo

# 8 卡 DDP 配置（与 run_benchmark_suite.sh 一致）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WORK_ROOT="${WORK_ROOT:-/workspace/repo}"
export DATA_ROOT="${DATA_ROOT:-}"
NUM_GPUS=8

echo "========== 1. 清理僵尸进程 =========="
pkill -f "run_benchmark|train_velocitygan|train\.py|unified_benchmark|benchmark_eval" 2>/dev/null || true
sleep 2
echo "GPU 状态 (${NUM_GPUS} 卡):"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv 2>/dev/null || nvidia-smi

echo ""
echo "========== 2. 备份 InversionNet 数据（若存在）=========="
LOG_ROOT="${LOG_ROOT:-${WORK_ROOT}/benchmark_logs}"
if [[ -d "${LOG_ROOT}/inversionnet_seed42" ]]; then
  BACKUP="${LOG_ROOT}/backup_before_full_$(date +%Y%m%d_%H%M)"
  mkdir -p "${BACKUP}"
  cp -r "${LOG_ROOT}/inversionnet_seed42" "${BACKUP}/" 2>/dev/null || true
  cp "${LOG_ROOT}/benchmark_metrics.csv" "${BACKUP}/" 2>/dev/null || true
  echo "已备份到 ${BACKUP}"
else
  echo "无 inversionnet_seed42，跳过备份"
fi

echo ""
echo "========== 3. VelocityGAN 快速验证（8 卡 DDP，1 epoch）=========="
cd FuTE-FWI
torchrun --nproc_per_node=${NUM_GPUS} train_velocitygan.py -d FlatVel -v A \
  --use-unified-loader \
  --data-root "${DATA_ROOT}" \
  --global-map-csv "${WORK_ROOT}/benchmark/generated_split/global_map.csv" \
  --stats-json "${WORK_ROOT}/benchmark/generated_split/train_stats.json" \
  --align-multiple 32 --align-mode crop \
  --seed 42 --batch-size 8 --epochs 1 \
  --save-interval 5 --vis-interval 5 \
  --output /tmp/velocitygan_smoke --name VelocityGAN

echo ""
echo "========== VelocityGAN 8 卡 DDP 验证通过！可启动全量 Benchmark =========="
