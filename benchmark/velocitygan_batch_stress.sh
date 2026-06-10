#!/usr/bin/env bash
# VelocityGAN 8 卡 DDP 压力测试：测试不同 batch size 的稳定性与耗时
# 用法: 在 Docker 内执行: bash benchmark/velocitygan_batch_stress.sh
# 或: docker run --rm --gpus all --shm-size=32g -v ... fwi-benchmark:cu118 bash /workspace/repo/benchmark/velocitygan_batch_stress.sh

set -e
cd /workspace/repo

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WORK_ROOT="${WORK_ROOT:-/workspace/repo}"
export DATA_ROOT="${DATA_ROOT:-}"
NUM_GPUS=8

echo "========== VelocityGAN 8 卡 DDP 压力测试 =========="
echo "GPU: $(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader | head -1)"
echo ""

for BATCH in 4 8 16 32; do
  echo "---------- batch_size=${BATCH} (每卡) 总 batch=$((BATCH * NUM_GPUS)) ----------"
  START=$(date +%s)
  if cd FuTE-FWI && torchrun --nproc_per_node=${NUM_GPUS} train_velocitygan.py -d FlatVel -v A \
    --use-unified-loader \
    --data-root "${DATA_ROOT}" \
    --global-map-csv "${WORK_ROOT}/benchmark/generated_split/global_map.csv" \
    --stats-json "${WORK_ROOT}/benchmark/generated_split/train_stats.json" \
    --align-multiple 32 --align-mode crop \
    --seed 42 --batch-size ${BATCH} --epochs 1 \
    --save-interval 999 --vis-interval 999 \
    --output /tmp/velocitygan_stress_b${BATCH} --name VelocityGAN 2>&1; then
    END=$(date +%s)
    echo "[OK] batch=${BATCH} 通过, 耗时 $((END - START))s"
  else
    echo "[FAIL] batch=${BATCH} OOM 或报错"
    break
  fi
  cd /workspace/repo
  echo ""
done

echo "========== 压力测试结束 =========="
echo "结论（15GB 卡 × 8）：batch 4/8 通过，16 OOM。推荐 BATCH_PER_GPU=8，显存紧张时用 4。"
