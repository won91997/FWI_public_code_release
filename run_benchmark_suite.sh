#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# Strict benchmark runner (profile/train/eval + single-seed)
# - Linux/Docker friendly（推荐容器 fwi-benchmark:cu118，见 benchmark/docker/BUILD_AND_RUN.md）
# - Sequential execution for fair timing
# - Unified CSV for architecture + quality metrics
#
# ========== 重要：所有测试统一 8 卡 DDP 真实环境 ==========
# - 训练：所有 9 个模型均使用 torchrun --nproc_per_node=8（8 卡 DDP）
# - 不使用单卡，确保与真实生产/论文环境一致
# - profile/eval 阶段使用可见 GPU 中的首卡
# ==========================================================

# -----------------------------
# User configurable section
# -----------------------------
# 基准测试公平性：默认前 8 张卡 (0-7)，可通过 GPU_IDS 覆盖（如后 8 张：8,9,10,11,12,13,14,15）
BENCHMARK_GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${BENCHMARK_GPU_IDS}}"
GPU_IDS="${BENCHMARK_GPU_IDS}"

# ==========================================
# 统一公平性设置：单卡 Batch Size（总 batch = BATCH_PER_GPU × 8）
# 全员单精度 fp32（禁用 AMP）
# ==========================================
# 公平基准：全员 batch 2/卡（总 16），统一压力测试
BATCH_PER_GPU="${BATCH_PER_GPU:-2}"
BATCH_UNIFIED="${BATCH_PER_GPU}"
BATCH_INVNET="${BATCH_PER_GPU}"
BATCH_VELGAN="${BATCH_PER_GPU}"
BATCH_FUTEFWI="${BATCH_FUTEFWI:-2}"
BATCH_ABA_FWI="${BATCH_ABA_FWI:-2}"
BATCH_CONVNEXT="${BATCH_PER_GPU}"

# 自动检测：若未设置则使用脚本所在目录（支持宿主机/Docker 双模式）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${WORK_ROOT:-${SCRIPT_DIR}}"
# 公平模式：BENCHMARK_FAIR_MODE=1 时，ABA_FWI 使用 FairWrapper（入口下采样到 70x70）
# 详见 benchmark/FAIR_BENCHMARK.md；必须从头重训，结果存 benchmark_logs_fair
export BENCHMARK_FAIR_MODE="${BENCHMARK_FAIR_MODE:-0}"
# 输出根目录，可覆盖以单独存放（如前8卡 → benchmark_logs_gpu0_7）
# 公平模式时自动使用 benchmark_logs_fair，与普通结果隔离
if [[ "${BENCHMARK_FAIR_MODE}" == "1" ]]; then
  BENCHMARK_OUTPUT_ROOT="${BENCHMARK_OUTPUT_ROOT:-${WORK_ROOT:-.}/benchmark_logs_fair}"
else
  BENCHMARK_OUTPUT_ROOT="${BENCHMARK_OUTPUT_ROOT:-${WORK_ROOT:-.}/benchmark_logs}"
fi
LOG_ROOT="${LOG_ROOT:-${BENCHMARK_OUTPUT_ROOT}}"
USE_DDP="${USE_DDP:-1}"  # 1=所有模型 8 卡 DDP（推荐，真实环境）；0=单卡（不推荐）
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
# 仅跑指定模型，如 TASK_FILTER=FuteFWI 或 TASK_FILTER=ABA_FWI,FCNVMB（逗号分隔多模型）
TASK_FILTER="${TASK_FILTER:-}"
# 若系统 python 为 2.7，请设置 PYTHON_BIN=python3
PYTHON_BIN="${PYTHON_BIN:-python}"
# torchrun 未安装时回退到 python -m torch.distributed.run
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
if [[ "${TORCHRUN_BIN}" == "torchrun" ]] && ! command -v torchrun &>/dev/null; then
  TORCHRUN_BIN="${PYTHON_BIN} -m torch.distributed.run"
fi
DATA_ROOT="${DATA_ROOT:-}"
SEEDS="${SEEDS:-42}"
BUDGET_MODE="${BUDGET_MODE:-epoch}" # epoch|wallclock
WALLCLOCK_HOURS="${WALLCLOCK_HOURS:-6}"
# 统一脚本训练轮数（每5轮保存+可视化）
EPOCHS="${EPOCHS:-100}"
# InversionNet 使用 -eb -nb（epoch_block * num_block = epochs），需与 EPOCHS 对齐
INV_EB=5
[[ ${EPOCHS} -lt 5 ]] && INV_EB=${EPOCHS}
INV_NB=$(( (EPOCHS + INV_EB - 1) / INV_EB ))

# 从 GPU_IDS 解析 GPU 数量（用于 DDP nproc_per_node）
IFS=',' read -r -a _gpu_arr <<< "${GPU_IDS}"
NUM_GPUS="${#_gpu_arr[@]}"

if [[ -z "${DATA_ROOT}" ]]; then
  echo "[ERROR] DATA_ROOT is not set. Set it to the dataset root containing seismic_full/ and vmodel_full/." >&2
  echo "Example: DATA_ROOT=/path/to/DATA_ROOT bash $0" >&2
  exit 2
fi

# 容器场景：若 GPU_IDS=8,9,10,11,12,13,14,15 但容器内仅 8 卡 (0-7)，则 CUDA 设备应使用 0-7
NUM_VISIBLE_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
FIRST_GPU="${_gpu_arr[0]:-0}"
if [[ "${FIRST_GPU}" -ge 8 ]] && [[ "${NUM_VISIBLE_GPUS}" -eq 8 ]]; then
  EFFECTIVE_GPU_IDS="0,1,2,3,4,5,6,7"
else
  EFFECTIVE_GPU_IDS="${GPU_IDS}"
fi


# ----------------------------------------------------------
# TASK format (8 fields, pipe-delimited):
# model_name|model_category|repo_dir|gpu_id|profile_cmd|split_cmd|train_cmd|eval_cmd
#
# IMPORTANT: Use __SEED__ as placeholder for BENCHMARK_SEED.
#            It will be replaced at runtime for each seed iteration.
#
# profile_cmd should print:
# METRICS params_m=<float> flops_g=<float> infer_ms=<float>
#
# eval_cmd should ideally print:
# MSE: <float>
# MAE: <float>
# PSNR: <float>
# SSIM: <float>
# L1-Grad: <float>
# LPIPS: <float>
# ----------------------------------------------------------

# Step 0: Generate split and train stats if missing
SPLIT_DIR="${WORK_ROOT}/benchmark/generated_split"
GLOBAL_MAP="${SPLIT_DIR}/global_map.csv"
STATS_JSON="${SPLIT_DIR}/train_stats.json"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking split and stats..."
if [[ ! -f "${GLOBAL_MAP}" ]]; then
  echo "Generating split..."
  ${PYTHON_BIN} "${WORK_ROOT}/benchmark/make_split_from_inline.py" \
    --data-root "${DATA_ROOT}" --out-dir "${SPLIT_DIR}"
fi
if [[ ! -f "${STATS_JSON}" ]]; then
  echo "Computing full train-only stats (robust_max from p99 of |seismic|)..."
  ${PYTHON_BIN} "${WORK_ROOT}/benchmark/compute_train_stats.py" \
    --global-map-csv "${GLOBAL_MAP}" --out-json "${STATS_JSON}"
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Split and stats ready."

UNIFIED_TRAIN="${PYTHON_BIN} ${WORK_ROOT}/benchmark/unified_benchmark_train.py"
UNIFIED_TEST="${PYTHON_BIN} ${WORK_ROOT}/benchmark/unified_benchmark_test.py"
EVAL_PY="${WORK_ROOT}/benchmark/benchmark_eval.py"

TASKS=(
  # ConvNeXt-FWI is run by the top-level wrappers through the legacy ConvNeXtKaggle code path.
  # This lower-level suite intentionally keeps only the eight shared-task models.

  # 1. FuteFWI (FuTE-FWI) - 原生 train，Transformer 训练最慢之二
  "FuteFWI|Transformer|${WORK_ROOT}/FuTE-FWI|0|${PYTHON_BIN} \"${WORK_ROOT}/profile_model_metrics.py\" --repo futefwi --model FuteFWI --shape 1,5,2976,256 --device cuda||${PYTHON_BIN} train_futefwi.py -d FlatVel -v A --use-unified-loader --data-root \"${DATA_ROOT}\" --global-map-csv \"${GLOBAL_MAP}\" --stats-json \"${STATS_JSON}\" --align-multiple 32 --align-mode crop --seed __SEED__ --batch-size ${BATCH_FUTEFWI} --epochs ${EPOCHS} --save-interval 5 --vis-interval 5 --sync-bn --output \"${BENCHMARK_OUTPUT_ROOT}/futefwi_seed__SEED__\" --name FuteFWI|${UNIFIED_TEST} --model FuteFWI --checkpoint \"${BENCHMARK_OUTPUT_ROOT}/futefwi_seed__SEED__/FuteFWI_FlatVel_A_D.pt\" --data-root \"${DATA_ROOT}\" --global-map-csv \"${GLOBAL_MAP}\" --stats-json \"${STATS_JSON}\" --align-multiple 32 --align-mode crop --batch-size ${BATCH_FUTEFWI} --export-dir \"${BENCHMARK_OUTPUT_ROOT}/futefwi_seed__SEED__/eval_results\" --benchmark-eval \"${EVAL_PY}\" --benchmark-eval-json \"${BENCHMARK_OUTPUT_ROOT}/futefwi_seed__SEED__/eval_results/metrics.json\" --benchmark-eval-lpips-mode real"

  # 2. InversionNet (OpenFWI) - 原生 train，unified_test 评测（OpenFWI 无 test.py）
  "InversionNet|CNN|${WORK_ROOT}/OpenFWI|0|${PYTHON_BIN} \"${WORK_ROOT}/profile_model_metrics.py\" --repo openfwi --model InversionNet --shape 1,5,2976,256 --device cuda||${PYTHON_BIN} train.py --use-unified-loader --data-root \"${DATA_ROOT}\" --global-map-csv \"${GLOBAL_MAP}\" --stats-json \"${STATS_JSON}\" --align-multiple 32 --align-mode crop --seed __SEED__ -m InversionNet -b ${BATCH_INVNET} --lr 1e-4 -eb ${INV_EB} -nb ${INV_NB} -j 4 --vis-interval 5 --sync-bn -o \"${BENCHMARK_OUTPUT_ROOT}\" -n inversionnet_seed__SEED__ -s \"\"|${UNIFIED_TEST} --model InversionNet --checkpoint \"${BENCHMARK_OUTPUT_ROOT}/inversionnet_seed__SEED__/checkpoint.pth\" --data-root \"${DATA_ROOT}\" --global-map-csv \"${GLOBAL_MAP}\" --stats-json \"${STATS_JSON}\" --align-multiple 32 --align-mode crop --batch-size ${BATCH_INVNET} --export-dir \"${BENCHMARK_OUTPUT_ROOT}/inversionnet_seed__SEED__/eval\" --benchmark-eval \"${EVAL_PY}\" --benchmark-eval-json \"${BENCHMARK_OUTPUT_ROOT}/inversionnet_seed__SEED__/eval/metrics.json\" --benchmark-eval-lpips-mode real"

  # 3. VelocityGAN (FuTE-FWI) - 原生 train，unified_test 评测（FuTE-FWI test.py 无 benchmark 参数）
  "VelocityGAN|GAN|${WORK_ROOT}/FuTE-FWI|0|${PYTHON_BIN} \"${WORK_ROOT}/profile_model_metrics.py\" --repo futefwi --model VelocityGAN --shape 1,5,2976,256 --device cuda||${PYTHON_BIN} train_velocitygan.py -d FlatVel -v A --use-unified-loader --data-root \"${DATA_ROOT}\" --global-map-csv \"${GLOBAL_MAP}\" --stats-json \"${STATS_JSON}\" --align-multiple 32 --align-mode crop --seed __SEED__ --batch-size ${BATCH_VELGAN} --epochs ${EPOCHS} --save-interval 5 --vis-interval 5 --output \"${BENCHMARK_OUTPUT_ROOT}/velocitygan_seed__SEED__\" --name VelocityGAN|${UNIFIED_TEST} --model VelocityGAN --checkpoint \"${BENCHMARK_OUTPUT_ROOT}/velocitygan_seed__SEED__/VelocityGAN_FlatVel_A_D.pt\" --data-root \"${DATA_ROOT}\" --global-map-csv \"${GLOBAL_MAP}\" --stats-json \"${STATS_JSON}\" --align-multiple 32 --align-mode crop --batch-size ${BATCH_VELGAN} --export-dir \"${BENCHMARK_OUTPUT_ROOT}/velocitygan_seed__SEED__/eval_results\" --benchmark-eval \"${EVAL_PY}\" --benchmark-eval-json \"${BENCHMARK_OUTPUT_ROOT}/velocitygan_seed__SEED__/eval_results/metrics.json\" --benchmark-eval-lpips-mode real"

  # 4. DCNet - 统一脚本（每5轮保存+可视化）
  "DCNet|CNN|${WORK_ROOT}/DCNet|0|${PYTHON_BIN} \"${WORK_ROOT}/profile_model_metrics.py\" --repo dcnet --model DCNet --shape 1,5,2976,256 --device cuda||${UNIFIED_TRAIN} --model DCNet --data-root \"${DATA_ROOT}\" --global-map-csv \"${GLOBAL_MAP}\" --stats-json \"${STATS_JSON}\" --align-multiple 32 --align-mode crop --seed __SEED__ --epochs ${EPOCHS} --save-interval 5 --vis-interval 5 --batch-size ${BATCH_UNIFIED} --output-path \"${BENCHMARK_OUTPUT_ROOT}/dcnet_seed__SEED__\" --save-name checkpoint --sync-bn|${UNIFIED_TEST} --model DCNet --checkpoint \"${BENCHMARK_OUTPUT_ROOT}/dcnet_seed__SEED__/checkpoint.pth\" --data-root \"${DATA_ROOT}\" --global-map-csv \"${GLOBAL_MAP}\" --stats-json \"${STATS_JSON}\" --align-multiple 32 --align-mode crop --export-dir \"${BENCHMARK_OUTPUT_ROOT}/dcnet_seed__SEED__/eval\" --benchmark-eval \"${EVAL_PY}\" --benchmark-eval-json \"${BENCHMARK_OUTPUT_ROOT}/dcnet_seed__SEED__/eval/metrics.json\" --benchmark-eval-lpips-mode real"

  # 5. DDNet70 - 统一脚本（每5轮保存+可视化）
  "DDNet70|UNet|${WORK_ROOT}/ddnet|0|${PYTHON_BIN} \"${WORK_ROOT}/profile_model_metrics.py\" --repo ddnet --model DDNet70 --shape 1,5,2976,256 --device cuda||${UNIFIED_TRAIN} --model DDNet70 --data-root \"${DATA_ROOT}\" --global-map-csv \"${GLOBAL_MAP}\" --stats-json \"${STATS_JSON}\" --align-multiple 32 --align-mode crop --seed __SEED__ --epochs ${EPOCHS} --save-interval 5 --vis-interval 5 --batch-size ${BATCH_UNIFIED} --output-path \"${BENCHMARK_OUTPUT_ROOT}/ddnet70_seed__SEED__\" --save-name checkpoint --sync-bn|${UNIFIED_TEST} --model DDNet70 --checkpoint \"${BENCHMARK_OUTPUT_ROOT}/ddnet70_seed__SEED__/checkpoint.pth\" --data-root \"${DATA_ROOT}\" --global-map-csv \"${GLOBAL_MAP}\" --stats-json \"${STATS_JSON}\" --align-multiple 32 --align-mode crop --export-dir \"${BENCHMARK_OUTPUT_ROOT}/ddnet70_seed__SEED__/eval\" --benchmark-eval \"${EVAL_PY}\" --benchmark-eval-json \"${BENCHMARK_OUTPUT_ROOT}/ddnet70_seed__SEED__/eval/metrics.json\" --benchmark-eval-lpips-mode real"

  # 6. TU_Net - 统一脚本（每5轮保存+可视化）
  "TU_Net|UNet|${WORK_ROOT}/TU-Net|0|${PYTHON_BIN} \"${WORK_ROOT}/profile_model_metrics.py\" --repo tu-net --model TU_Net --shape 1,5,2976,256 --device cuda||${UNIFIED_TRAIN} --model TU_Net --data-root \"${DATA_ROOT}\" --global-map-csv \"${GLOBAL_MAP}\" --stats-json \"${STATS_JSON}\" --align-multiple 32 --align-mode crop --seed __SEED__ --epochs ${EPOCHS} --save-interval 5 --vis-interval 5 --batch-size ${BATCH_UNIFIED} --output-path \"${BENCHMARK_OUTPUT_ROOT}/tu_net_seed__SEED__\" --save-name checkpoint --sync-bn|${UNIFIED_TEST} --model TU_Net --checkpoint \"${BENCHMARK_OUTPUT_ROOT}/tu_net_seed__SEED__/checkpoint.pth\" --data-root \"${DATA_ROOT}\" --global-map-csv \"${GLOBAL_MAP}\" --stats-json \"${STATS_JSON}\" --align-multiple 32 --align-mode crop --export-dir \"${BENCHMARK_OUTPUT_ROOT}/tu_net_seed__SEED__/eval\" --benchmark-eval \"${EVAL_PY}\" --benchmark-eval-json \"${BENCHMARK_OUTPUT_ROOT}/tu_net_seed__SEED__/eval/metrics.json\" --benchmark-eval-lpips-mode real"

  # 7. ABA_FWI - 统一脚本（每5轮保存+可视化）
  "ABA_FWI|ABA|${WORK_ROOT}/ABA-FWI|0|${PYTHON_BIN} \"${WORK_ROOT}/profile_model_metrics.py\" --repo aba-fwi --model ABA_FWI --shape 1,5,2976,256 --device cuda||${UNIFIED_TRAIN} --model ABA_FWI --data-root \"${DATA_ROOT}\" --global-map-csv \"${GLOBAL_MAP}\" --stats-json \"${STATS_JSON}\" --align-multiple 32 --align-mode crop --seed __SEED__ --epochs ${EPOCHS} --save-interval 5 --vis-interval 5 --batch-size ${BATCH_ABA_FWI} --output-path \"${BENCHMARK_OUTPUT_ROOT}/aba_fwi_seed__SEED__\" --save-name checkpoint --sync-bn|${UNIFIED_TEST} --model ABA_FWI --checkpoint \"${BENCHMARK_OUTPUT_ROOT}/aba_fwi_seed__SEED__/checkpoint.pth\" --data-root \"${DATA_ROOT}\" --global-map-csv \"${GLOBAL_MAP}\" --stats-json \"${STATS_JSON}\" --align-multiple 32 --align-mode crop --export-dir \"${BENCHMARK_OUTPUT_ROOT}/aba_fwi_seed__SEED__/eval\" --benchmark-eval \"${EVAL_PY}\" --benchmark-eval-json \"${BENCHMARK_OUTPUT_ROOT}/aba_fwi_seed__SEED__/eval/metrics.json\" --benchmark-eval-lpips-mode real"

  # 8. FCNVMB - 统一脚本（每5轮保存+可视化）
  "FCNVMB|FCN|${WORK_ROOT}/ABA-FWI|0|${PYTHON_BIN} \"${WORK_ROOT}/profile_model_metrics.py\" --repo aba-fwi --model FCNVMB_FWI --shape 1,5,2976,256 --device cuda||${UNIFIED_TRAIN} --model FCNVMB --data-root \"${DATA_ROOT}\" --global-map-csv \"${GLOBAL_MAP}\" --stats-json \"${STATS_JSON}\" --align-multiple 32 --align-mode crop --seed __SEED__ --epochs ${EPOCHS} --save-interval 5 --vis-interval 5 --batch-size ${BATCH_UNIFIED} --output-path \"${BENCHMARK_OUTPUT_ROOT}/fcnvmb_seed__SEED__\" --save-name checkpoint --sync-bn|${UNIFIED_TEST} --model FCNVMB --checkpoint \"${BENCHMARK_OUTPUT_ROOT}/fcnvmb_seed__SEED__/checkpoint.pth\" --data-root \"${DATA_ROOT}\" --global-map-csv \"${GLOBAL_MAP}\" --stats-json \"${STATS_JSON}\" --align-multiple 32 --align-mode crop --export-dir \"${BENCHMARK_OUTPUT_ROOT}/fcnvmb_seed__SEED__/eval\" --benchmark-eval \"${EVAL_PY}\" --benchmark-eval-json \"${BENCHMARK_OUTPUT_ROOT}/fcnvmb_seed__SEED__/eval/metrics.json\" --benchmark-eval-lpips-mode real"
)


# -----------------------------
# Helpers
# -----------------------------
ts() { date "+%Y-%m-%d %H:%M:%S"; }

contains_gpu() {
  local target="$1"
  IFS=',' read -r -a all_gpus <<< "${GPU_IDS}"
  for g in "${all_gpus[@]}"; do
    if [[ "${g}" == "${target}" ]]; then
      return 0
    fi
  done
  return 1
}

run_stage() {
  local name="$1"
  local repo_dir="$2"
  local gpu_id="$3"
  local cmd="$4"
  local log_file="$5"
  local stage="$6"

  # 当 gpu_id 不在 GPU_IDS 时（如后 8 卡 GPU_IDS=8-15 但 TASK 写 gpu_id=0），用 GPU_IDS 首卡代替，不跳过
  if ! contains_gpu "${gpu_id}"; then
    local first_gpu
    first_gpu=$(echo "${GPU_IDS}" | cut -d',' -f1)
    gpu_id="${first_gpu}"
    echo "[$(ts)] [INFO] ${name}: gpu_id not in GPU_IDS, using first GPU=${gpu_id}" | tee -a "${log_file}"
  fi

  if [[ ! -d "${repo_dir}" ]]; then
    echo "[$(ts)] [FAIL] ${name}: repo_dir not found: ${repo_dir}" | tee -a "${log_file}"
    return 2
  fi

  if [[ -z "${cmd}" ]]; then
    echo "[$(ts)] [INFO] ${name} ${stage}: empty command, skipped." | tee -a "${log_file}"
    return 0
  fi

  echo "[$(ts)] [INFO] Task=${name} Stage=${stage} Repo=${repo_dir} GPU=${gpu_id}" | tee -a "${log_file}"
  echo "[$(ts)] [INFO] CMD: ${cmd}" | tee -a "${log_file}"
  echo "[$(ts)] [INFO] DATA_ROOT=${DATA_ROOT}" | tee -a "${log_file}"

  # 后 8 卡 (8-15) 与前 8 卡 (0-7) 并行时需不同 MASTER_PORT，避免端口冲突（直接写入 torchrun 命令更可靠）
  local master_port="${MASTER_PORT:-}"
  if [[ -z "${master_port}" ]] && [[ "${GPU_IDS}" == *"8"* ]]; then
    master_port=29501
  fi
  local torchrun_port_args=""
  [[ -n "${master_port}" ]] && torchrun_port_args=" --master_port=${master_port}"

  # DDP 模式：训练时使用全部 GPU，所有模型 8 卡 DDP（容器内用 EFFECTIVE_GPU_IDS）
  local cuda_devices="${gpu_id}"
  if [[ "${USE_DDP}" == "1" ]] && [[ "${stage}" == "train" ]]; then
    cuda_devices="${EFFECTIVE_GPU_IDS}"
    # 必须替换为 torchrun，否则会退化为单进程/DataParallel
    if [[ "${repo_dir}" == *"OpenFWI"* ]] && [[ "${cmd}" == *"train.py"* ]]; then
      cmd=$(echo "${cmd}" | sed "s|[^ ]* train\.py|${TORCHRUN_BIN} --nproc_per_node=${NUM_GPUS}${torchrun_port_args} train.py|")
    elif [[ "${repo_dir}" == *"FuTE-FWI"* ]] && [[ "${cmd}" == *"train_velocitygan.py"* ]]; then
      cmd=$(echo "${cmd}" | sed "s|[^ ]* train_velocitygan\.py|${TORCHRUN_BIN} --nproc_per_node=${NUM_GPUS}${torchrun_port_args} train_velocitygan.py|")
    elif [[ "${repo_dir}" == *"FuTE-FWI"* ]] && [[ "${cmd}" == *"train_futefwi.py"* ]]; then
      cmd=$(echo "${cmd}" | sed "s|[^ ]* train_futefwi\.py|${TORCHRUN_BIN} --nproc_per_node=${NUM_GPUS}${torchrun_port_args} train_futefwi.py|")
    elif [[ "${cmd}" == *"unified_benchmark_train.py"* ]]; then
      cmd=$(echo "${cmd}" | sed "s|${PYTHON_BIN} [^ ]*unified_benchmark_train\.py|${TORCHRUN_BIN} --nproc_per_node=${NUM_GPUS}${torchrun_port_args} ${WORK_ROOT}/benchmark/unified_benchmark_train.py|")
    fi
  elif [[ "${EFFECTIVE_GPU_IDS}" != "${GPU_IDS}" ]]; then
    cuda_devices=$(echo "${EFFECTIVE_GPU_IDS}" | cut -d',' -f1)
  fi

  (
    cd "${repo_dir}"
    export CUDA_VISIBLE_DEVICES="${cuda_devices}"
    export DATA_ROOT
    export PYTHONUNBUFFERED=1
    export BENCHMARK_BUDGET_MODE="${BUDGET_MODE}"
    export BENCHMARK_WALLCLOCK_HOURS="${WALLCLOCK_HOURS}"
    export BENCHMARK_SEED="${BENCHMARK_SEED:-42}"
    # FuteFWI 专用：缓解 CUDNN_STATUS_INTERNAL_ERROR（部分 T4/驱动组合），仅影响本任务
    if [[ "${name}" == "FuteFWI" ]] && [[ "${stage}" == "train" ]]; then
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    fi
    bash -lc "${cmd}"
  ) >> "${log_file}" 2>&1
}

parse_profile_metric() {
  local key="$1"
  local log_file="$2"
  awk -v k="${key}" '
    /METRICS/ {
      for (i = 1; i <= NF; i++) {
        if ($i ~ ("^" k "=")) {
          split($i, a, "=");
          val = a[2];
        }
      }
    }
    END {
      if (val == "") print "NA";
      else print val;
    }
  ' "${log_file}"
}

parse_eval_metric() {
  local key="$1"
  local log_file="$2"
  local key_lower
  key_lower=$(echo "${key}" | tr '[:upper:]' '[:lower:]')
  awk -v target="${key_lower}" '
    function lower(s){ return tolower(s) }
    {
      line = lower($0)
      if (index(line, target) > 0) {
        for (i = NF; i >= 1; i--) {
          token = $i
          gsub(",", "", token)
          gsub(";", "", token)
          gsub("\\)", "", token)
          gsub("\\(", "", token)
          if (token ~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/) {
            val = token
            break
          }
        }
      }
    }
    END {
      if (val == "") print "NA";
      else print val;
    }
  ' "${log_file}"
}


# -----------------------------
# Main
# -----------------------------
mkdir -p "${LOG_ROOT}"
METRICS_CSV="${LOG_ROOT}/benchmark_metrics.csv"
echo "Model,Model Category,Params(M),FLOPs(G),Train(h),Infer(ms),MSE,MAE,PSNR,SSIM,L1-Grad,LPIPS,Status" > "${METRICS_CSV}"

echo "==================================================" | tee "${LOG_ROOT}/summary.log"
echo "Benchmark start: $(ts)" | tee -a "${LOG_ROOT}/summary.log"
echo "WORK_ROOT=${WORK_ROOT}" | tee -a "${LOG_ROOT}/summary.log"
echo "LOG_ROOT=${LOG_ROOT}" | tee -a "${LOG_ROOT}/summary.log"
echo "GPU_IDS=${GPU_IDS} (fixed for fair benchmark)" | tee -a "${LOG_ROOT}/summary.log"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG_ROOT}/summary.log"
echo "USE_DDP=${USE_DDP} (NUM_GPUS=${NUM_GPUS})" | tee -a "${LOG_ROOT}/summary.log"
echo "STOP_ON_FAIL=${STOP_ON_FAIL}" | tee -a "${LOG_ROOT}/summary.log"
echo "DATA_ROOT=${DATA_ROOT}" | tee -a "${LOG_ROOT}/summary.log"
echo "SEEDS=${SEEDS}" | tee -a "${LOG_ROOT}/summary.log"
echo "USE_DDP=${USE_DDP} (all models: ${NUM_GPUS} GPUs DDP)" | tee -a "${LOG_ROOT}/summary.log"
echo "BATCH_PER_GPU=${BATCH_PER_GPU} (Total=${BATCH_PER_GPU}×${NUM_GPUS}=$((BATCH_PER_GPU * NUM_GPUS)) 公平统一)" | tee -a "${LOG_ROOT}/summary.log"
echo "BUDGET_MODE=${BUDGET_MODE}" | tee -a "${LOG_ROOT}/summary.log"
echo "WALLCLOCK_HOURS=${WALLCLOCK_HOURS}" | tee -a "${LOG_ROOT}/summary.log"
echo "EPOCHS=${EPOCHS} (统一脚本每5轮保存+可视化)" | tee -a "${LOG_ROOT}/summary.log"
echo "==================================================" | tee -a "${LOG_ROOT}/summary.log"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[$(ts)] [WARN] nvidia-smi not found. Continue anyway." | tee -a "${LOG_ROOT}/summary.log"
else
  echo "==== GPU 状态 (nvidia-smi) ====" | tee -a "${LOG_ROOT}/summary.log"
  nvidia-smi 2>&1 | tee -a "${LOG_ROOT}/summary.log" || true
fi

failed_tasks=()

IFS=',' read -r -a seed_list <<< "${SEEDS}"

for item in "${TASKS[@]}"; do
  IFS='|' read -r name category repo_dir gpu_id profile_cmd split_cmd train_cmd eval_cmd <<< "${item}"
  if [[ -n "${TASK_FILTER}" ]]; then
    _match=0
    IFS=',' read -r -a _filter_arr <<< "${TASK_FILTER}"
    for _f in "${_filter_arr[@]}"; do
      _f=$(echo "${_f}" | xargs)
      [[ "${name}" == "${_f}" ]] && _match=1 && break
    done
    if [[ "${_match}" -eq 0 ]]; then
      echo "[$(ts)] [SKIP] ${name} (TASK_FILTER=${TASK_FILTER})" | tee -a "${LOG_ROOT}/summary.log"
      continue
    fi
  fi
  if [[ -z "${eval_cmd:-}" ]]; then
    eval_cmd="${train_cmd:-}"
    train_cmd="${split_cmd:-}"
    split_cmd=""
  fi
  # profile once per model
  base_log="${LOG_ROOT}/${name}.log"
  : > "${base_log}"
  echo "[$(ts)] [RUN ] ${name} profile" | tee -a "${LOG_ROOT}/summary.log"
  if ! run_stage "${name}" "${repo_dir}" "${gpu_id}" "${profile_cmd}" "${base_log}" "profile"; then
    failed_tasks+=("${name}")
    if [[ "${STOP_ON_FAIL}" == "1" ]]; then
      break
    fi
  fi
  params_m=$(parse_profile_metric "params_m" "${base_log}")
  flops_g=$(parse_profile_metric "flops_g" "${base_log}")
  infer_ms=$(parse_profile_metric "infer_ms" "${base_log}")

  # split generation once per model
  if [[ -n "${split_cmd}" ]]; then
    echo "[$(ts)] [RUN ] ${name} split" | tee -a "${LOG_ROOT}/summary.log"
    if ! run_stage "${name}" "${repo_dir}" "${gpu_id}" "${split_cmd}" "${base_log}" "split"; then
      failed_tasks+=("${name}")
      if [[ "${STOP_ON_FAIL}" == "1" ]]; then
        break
      fi
    fi
  fi

  for seed in "${seed_list[@]}"; do
    seed=$(echo "${seed}" | xargs)
    log_file="${LOG_ROOT}/${name}_seed${seed}.log"
    : > "${log_file}"
    task_status="OK"
    export BENCHMARK_SEED="${seed}"
    echo "[$(ts)] [RUN ] ${name} seed=${seed}" | tee -a "${LOG_ROOT}/summary.log"

    # Replace __SEED__ placeholder with actual seed value
    seed_train_cmd="${train_cmd//__SEED__/${seed}}"
    seed_eval_cmd="${eval_cmd//__SEED__/${seed}}"

    train_start=$(date +%s)
    if ! run_stage "${name}" "${repo_dir}" "${gpu_id}" "${seed_train_cmd}" "${log_file}" "train"; then
      task_status="FAIL(train)"
    fi
    train_end=$(date +%s)
    if [[ -n "${train_cmd}" ]]; then
      train_elapsed=$((train_end - train_start))
      # 至少 4 位有效数字
      train_h=$(awk -v s="${train_elapsed}" 'BEGIN {printf "%.6g", s/3600.0}')
    else
      train_h="NA"
    fi

    if [[ "${task_status}" == "OK" ]]; then
      if ! run_stage "${name}" "${repo_dir}" "${gpu_id}" "${seed_eval_cmd}" "${log_file}" "eval"; then
        task_status="FAIL(eval)"
      fi
    fi

    mse=$(parse_eval_metric "mse" "${log_file}")
    mae=$(parse_eval_metric "mae" "${log_file}")
    psnr=$(parse_eval_metric "psnr" "${log_file}")
    ssim=$(parse_eval_metric "ssim" "${log_file}")
    l1grad=$(parse_eval_metric "l1-grad" "${log_file}")
    lpips=$(parse_eval_metric "lpips" "${log_file}")

    echo "${name}#seed${seed},${category},${params_m},${flops_g},${train_h},${infer_ms},${mse},${mae},${psnr},${ssim},${l1grad},${lpips},${task_status}" >> "${METRICS_CSV}"

    if [[ "${task_status}" == "OK" ]]; then
      echo "[$(ts)] [OK  ] ${name} seed=${seed}" | tee -a "${LOG_ROOT}/summary.log"
    else
      echo "[$(ts)] [FAIL] ${name} seed=${seed} (${task_status})" | tee -a "${LOG_ROOT}/summary.log"
      failed_tasks+=("${name}#seed${seed}")
      if [[ "${STOP_ON_FAIL}" == "1" ]]; then
        echo "[$(ts)] stop on first failure." | tee -a "${LOG_ROOT}/summary.log"
        break 2
      fi
    fi
  done
done

echo "==================================================" | tee -a "${LOG_ROOT}/summary.log"
echo "Benchmark end: $(ts)" | tee -a "${LOG_ROOT}/summary.log"
echo "Metrics CSV: ${METRICS_CSV}" | tee -a "${LOG_ROOT}/summary.log"
AGG_CSV="${LOG_ROOT}/benchmark_metrics_agg.csv"
${PYTHON_BIN} "${WORK_ROOT}/benchmark/aggregate_seeds.py" --in-csv "${METRICS_CSV}" --out-csv "${AGG_CSV}" >> "${LOG_ROOT}/summary.log" 2>&1 || true
echo "Aggregated CSV: ${AGG_CSV}" | tee -a "${LOG_ROOT}/summary.log"
# 生成 BENCHMARK_RESULTS_SUMMARY.md（所有指标至少 4 位有效数字）
_LOGS_PARENT="$(dirname "${LOG_ROOT}")"
${PYTHON_BIN} "${WORK_ROOT}/benchmark/generate_benchmark_summary.py" \
  --logs "${LOG_ROOT}" --logs-gpu07 "${_LOGS_PARENT}/benchmark_logs_gpu0_7" \
  --out "${LOG_ROOT}/BENCHMARK_RESULTS_SUMMARY.md" >> "${LOG_ROOT}/summary.log" 2>&1 || true
if [[ "${#failed_tasks[@]}" -eq 0 ]]; then
  echo "[RESULT] all tasks finished successfully." | tee -a "${LOG_ROOT}/summary.log"
  exit 0
else
  echo "[RESULT] failed tasks: ${failed_tasks[*]}" | tee -a "${LOG_ROOT}/summary.log"
  exit 1
fi
