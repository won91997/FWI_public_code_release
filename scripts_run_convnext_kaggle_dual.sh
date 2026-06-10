#!/bin/bash
# ConvNeXtKaggle 双模式重训：Fair + NonFair，batch=2，seed=42
# 使用修正后的 vmin/vmax 反归一化

set -e
WORK_ROOT="${WORK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATA_ROOT="${DATA_ROOT:-}"
OUT_ROOT="${WORK_ROOT}/benchmark_logs_convnext_kaggle_seed42_b2"
GPU_SET="${GPU_SET:-0,1,2,3,4,5,6,7}"

# 清理旧容器
docker rm -f convnextkaggle_seed42_b2_dualmode 2>/dev/null || true

mkdir -p "$OUT_ROOT"

if [[ -z "${DATA_ROOT}" ]]; then
  echo "[ERROR] DATA_ROOT is not set. Set it to the dataset root containing seismic_full/ and vmodel_full/." >&2
  exit 2
fi

# 可选：加载 Kaggle 预训练权重（若存在）
INIT_CKPT="${WORK_ROOT}/ConvNeXt-Kaggle/weights/bartley_unet2d_convnext_seed1_epochbest_FT.pth"
INIT_OPT=""
[[ -f "$INIT_CKPT" ]] && INIT_OPT="--init-ckpt $INIT_CKPT"

echo "[$(date +%F\ %T)] ConvNeXtKaggle dual-mode: Fair + NonFair, batch=2, seed=42" | tee "$OUT_ROOT/summary.log"

# 容器内 OUT_ROOT 路径
CONTAINER_OUT="/workspace/repo/benchmark_logs_convnext_kaggle_seed42_b2"

# 挂载整个 repository 目录，确保 benchmark 与 ConvNeXt-Kaggle 均可用
docker run -d --name convnextkaggle_seed42_b2_dualmode \
  --gpus "\"device=${GPU_SET}\"" \
  --shm-size=32g \
  -v "${WORK_ROOT}:/workspace/repo" \
  -v "${DATA_ROOT}:/workspace/data" \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub \
  -e KAGGLE_CONVNEXT_PRETRAINED=1 \
  -e KAGGLE_CONVNEXT_BACKBONE=convnext_small.fb_in22k_ft_in1k \
  fwi-benchmark:cu118 bash -c "
set -e
cd /workspace/repo
mkdir -p $CONTAINER_OUT
OUT=$CONTAINER_OUT
INIT_OPT='$INIT_OPT'
python -m pip -q install monai==1.3.2

# === FAIR 模式 ===
echo \"[\$(date +%F\ %T)] profile FAIR\" | tee -a \$OUT/summary.log
BENCHMARK_FAIR_MODE=1 python /workspace/repo/profile_model_metrics.py --repo convnext-kaggle --model ConvNeXtKaggle --shape 1,5,2976,256 --device cuda 2>&1 | tee \$OUT/ConvNeXtKaggle_fair_profile.log || true

echo \"[\$(date +%F\ %T)] train FAIR (batch=2, seed=42)\" | tee -a \$OUT/summary.log
BENCHMARK_FAIR_MODE=1 torchrun --nproc_per_node=8 --master_port=29521 /workspace/repo/benchmark/unified_benchmark_train.py \
  --model ConvNeXtKaggle \
  --data-root /workspace/data \
  --global-map-csv /workspace/repo/benchmark/generated_split/global_map.csv \
  --stats-json /workspace/repo/benchmark/generated_split/train_stats.json \
  --align-multiple 32 --align-mode crop \
  --seed 42 --epochs 100 --save-interval 5 --vis-interval 5 --batch-size 2 \
  --output-path \$OUT/convnext_kaggle_fair_seed42 --save-name checkpoint --sync-bn \$INIT_OPT \
  2>&1 | tee \$OUT/ConvNeXtKaggle_fair_seed42.log

echo \"[\$(date +%F\ %T)] eval FAIR\" | tee -a \$OUT/summary.log
BENCHMARK_FAIR_MODE=1 python /workspace/repo/benchmark/unified_benchmark_test.py \
  --model ConvNeXtKaggle \
  --checkpoint \$OUT/convnext_kaggle_fair_seed42/checkpoint.pth \
  --data-root /workspace/data \
  --global-map-csv /workspace/repo/benchmark/generated_split/global_map.csv \
  --stats-json /workspace/repo/benchmark/generated_split/train_stats.json \
  --align-multiple 32 --align-mode crop --batch-size 2 \
  --export-dir \$OUT/convnext_kaggle_fair_seed42/eval \
  --benchmark-eval /workspace/repo/benchmark/benchmark_eval.py \
  --benchmark-eval-json \$OUT/convnext_kaggle_fair_seed42/eval/metrics.json \
  --benchmark-eval-lpips-mode real \
  2>&1 | tee -a \$OUT/ConvNeXtKaggle_fair_seed42.log

# === NONFAIR 模式 ===
echo \"[\$(date +%F\ %T)] profile NONFAIR\" | tee -a \$OUT/summary.log
BENCHMARK_FAIR_MODE=0 python /workspace/repo/profile_model_metrics.py --repo convnext-kaggle --model ConvNeXtKaggle --shape 1,5,2976,256 --device cuda 2>&1 | tee \$OUT/ConvNeXtKaggle_nonfair_profile.log || true

echo \"[\$(date +%F\ %T)] train NONFAIR (batch=2, seed=42)\" | tee -a \$OUT/summary.log
BENCHMARK_FAIR_MODE=0 torchrun --nproc_per_node=8 --master_port=29522 /workspace/repo/benchmark/unified_benchmark_train.py \
  --model ConvNeXtKaggle \
  --data-root /workspace/data \
  --global-map-csv /workspace/repo/benchmark/generated_split/global_map.csv \
  --stats-json /workspace/repo/benchmark/generated_split/train_stats.json \
  --align-multiple 32 --align-mode crop \
  --seed 42 --epochs 100 --save-interval 5 --vis-interval 5 --batch-size 2 \
  --output-path \$OUT/convnext_kaggle_nonfair_seed42 --save-name checkpoint --sync-bn \$INIT_OPT \
  2>&1 | tee \$OUT/ConvNeXtKaggle_nonfair_seed42.log

echo \"[\$(date +%F\ %T)] eval NONFAIR\" | tee -a \$OUT/summary.log
BENCHMARK_FAIR_MODE=0 python /workspace/repo/benchmark/unified_benchmark_test.py \
  --model ConvNeXtKaggle \
  --checkpoint \$OUT/convnext_kaggle_nonfair_seed42/checkpoint.pth \
  --data-root /workspace/data \
  --global-map-csv /workspace/repo/benchmark/generated_split/global_map.csv \
  --stats-json /workspace/repo/benchmark/generated_split/train_stats.json \
  --align-multiple 32 --align-mode crop --batch-size 2 \
  --export-dir \$OUT/convnext_kaggle_nonfair_seed42/eval \
  --benchmark-eval /workspace/repo/benchmark/benchmark_eval.py \
  --benchmark-eval-json \$OUT/convnext_kaggle_nonfair_seed42/eval/metrics.json \
  --benchmark-eval-lpips-mode real \
  2>&1 | tee -a \$OUT/ConvNeXtKaggle_nonfair_seed42.log

echo \"[\$(date +%F\ %T)] done\" | tee -a \$OUT/summary.log
"

echo ""
echo "容器已启动: convnextkaggle_seed42_b2_dualmode"
echo "配置: Fair + NonFair, batch=2, seed=42"
echo "输出: $OUT_ROOT"
echo "查看日志: docker logs -f convnextkaggle_seed42_b2_dualmode"
