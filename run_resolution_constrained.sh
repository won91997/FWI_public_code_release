#!/usr/bin/env bash
set -euo pipefail

# Paper-facing 10-model Resolution-Constrained benchmark (legacy code flag: Fair).
# Runnable models: ConvNeXtKaggle, FuteFWI, InversionNet, VelocityGAN, DCNet,
# DDNet70, TU_Net, ABA_FWI, FCNVMB, VIFNet.

WORK_ROOT="${WORK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATA_ROOT="${DATA_ROOT:-}"
OUT_ROOT="${OUT_ROOT:-${WORK_ROOT}/logs_fair}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
SEEDS="${SEEDS:-42}"
BATCH_PER_GPU="${BATCH_PER_GPU:-2}"
EPOCHS="${EPOCHS:-100}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
if [[ "${TORCHRUN_BIN}" == "torchrun" ]] && ! command -v torchrun &>/dev/null; then
  TORCHRUN_BIN="${PYTHON_BIN} -m torch.distributed.run"
fi

IFS=',' read -r -a GPU_ARR <<< "${GPU_IDS}"
NUM_GPUS="${#GPU_ARR[@]}"

SPLIT_DIR="${WORK_ROOT}/benchmark/generated_split"
GLOBAL_MAP="${SPLIT_DIR}/global_map.csv"
STATS_JSON="${SPLIT_DIR}/train_stats.json"
SUMMARY_LOG="${OUT_ROOT}/resolution_constrained_fair_summary.log"

mkdir -p "${OUT_ROOT}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${SUMMARY_LOG}"
}

if [[ -z "${DATA_ROOT}" ]]; then
  echo "[ERROR] DATA_ROOT is not set. Set it to the dataset root containing seismic_full/ and vmodel_full/." >&2
  echo "Example: DATA_ROOT=/path/to/DATA_ROOT bash $0" >&2
  exit 2
fi

ensure_runtime_deps() {
  local pip_index="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
  if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import monai
PY
  then
    log "install missing runtime dependency: monai==1.3.2"
    local pip_trust=""
    if [[ -n "${PIP_TRUSTED_HOST:-}" ]]; then
      for h in ${PIP_TRUSTED_HOST}; do pip_trust=" ${pip_trust} --trusted-host ${h}"; done
    elif [[ "${pip_index}" == *"tuna"* ]]; then
      pip_trust=" --trusted-host pypi.tuna.tsinghua.edu.cn --trusted-host files.pythonhosted.org"
    fi
    # shellcheck disable=SC2086
    "${PYTHON_BIN}" -m pip install -q -i "${pip_index}" ${pip_trust} monai==1.3.2
  fi
}

ensure_split_stats() {
  mkdir -p "${SPLIT_DIR}"
  if [[ ! -f "${GLOBAL_MAP}" ]]; then
    log "generate split"
    "${PYTHON_BIN}" "${WORK_ROOT}/benchmark/make_split_from_inline.py" \
      --data-root "${DATA_ROOT}" \
      --out-dir "${SPLIT_DIR}"
  fi
  if [[ ! -f "${STATS_JSON}" ]]; then
    log "compute train stats"
    "${PYTHON_BIN}" "${WORK_ROOT}/benchmark/compute_train_stats.py" \
      --global-map-csv "${GLOBAL_MAP}" \
      --out-json "${STATS_JSON}"
  fi
}

run_main_eight() {
  log "run main 8 models via run_benchmark_suite.sh (Resolution-Constrained)"
  BENCHMARK_FAIR_MODE=1 \
  WORK_ROOT="${WORK_ROOT}" \
  DATA_ROOT="${DATA_ROOT}" \
  GPU_IDS="${GPU_IDS}" \
  BATCH_PER_GPU="${BATCH_PER_GPU}" \
  BATCH_FUTEFWI="${BATCH_PER_GPU}" \
  BATCH_ABA_FWI="${BATCH_PER_GPU}" \
  EPOCHS="${EPOCHS}" \
  SEEDS="${SEEDS}" \
  BENCHMARK_OUTPUT_ROOT="${OUT_ROOT}/main_10models_fair" \
  TASK_FILTER="FuteFWI,InversionNet,VelocityGAN,DCNet,DDNet70,TU_Net,ABA_FWI,FCNVMB" \
  bash "${WORK_ROOT}/run_benchmark_suite.sh"
}

run_convnext_kaggle() {
  local seed="$1"
  local seed_out="${OUT_ROOT}/convnext_kaggle_fair_seed${seed}"
  local init_ckpt="${WORK_ROOT}/ConvNeXt-Kaggle/weights/bartley_unet2d_convnext_seed1_epochbest_FT.pth"
  local port_base=29621
  local init_opt=()
  mkdir -p "${seed_out}"
  [[ -f "${init_ckpt}" ]] && init_opt=(--init-ckpt "${init_ckpt}")

  log "ConvNeXt-FWI Resolution-Constrained seed=${seed}"
  BENCHMARK_FAIR_MODE=1 \
  "${PYTHON_BIN}" "${WORK_ROOT}/profile_model_metrics.py" \
    --repo convnext-kaggle \
    --model ConvNeXtKaggle \
    --shape 1,5,2976,256 \
    --device cuda 2>&1 | tee "${seed_out}/ConvNeXtKaggle_profile.log"

  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  KAGGLE_CONVNEXT_PRETRAINED="${KAGGLE_CONVNEXT_PRETRAINED:-1}" \
  KAGGLE_CONVNEXT_BACKBONE="${KAGGLE_CONVNEXT_BACKBONE:-convnext_small.fb_in22k_ft_in1k}" \
  BENCHMARK_FAIR_MODE=1 \
  ${TORCHRUN_BIN} --nproc_per_node="${NUM_GPUS}" --master_port="${port_base}" \
    "${WORK_ROOT}/benchmark/unified_benchmark_train.py" \
    --model ConvNeXtKaggle \
    --data-root "${DATA_ROOT}" \
    --global-map-csv "${GLOBAL_MAP}" \
    --stats-json "${STATS_JSON}" \
    --align-multiple 32 \
    --align-mode crop \
    --seed "${seed}" \
    --epochs "${EPOCHS}" \
    --save-interval 5 \
    --vis-interval 5 \
    --batch-size "${BATCH_PER_GPU}" \
    --output-path "${seed_out}" \
    --save-name checkpoint \
    --sync-bn \
    "${init_opt[@]}" 2>&1 | tee "${seed_out}/ConvNeXtKaggle_seed${seed}.log"

  BENCHMARK_FAIR_MODE=1 \
  "${PYTHON_BIN}" "${WORK_ROOT}/benchmark/unified_benchmark_test.py" \
    --model ConvNeXtKaggle \
    --checkpoint "${seed_out}/checkpoint.pth" \
    --data-root "${DATA_ROOT}" \
    --global-map-csv "${GLOBAL_MAP}" \
    --stats-json "${STATS_JSON}" \
    --align-multiple 32 \
    --align-mode crop \
    --batch-size "${BATCH_PER_GPU}" \
    --export-dir "${seed_out}/eval" \
    --benchmark-eval "${WORK_ROOT}/benchmark/benchmark_eval.py" \
    --benchmark-eval-json "${seed_out}/eval/metrics.json" \
    --benchmark-eval-lpips-mode real 2>&1 | tee -a "${seed_out}/ConvNeXtKaggle_seed${seed}.log"
}

run_vifnet() {
  local seed="$1"
  local seed_out="${OUT_ROOT}/vifnet_fair_seed${seed}"
  local port_base=29641
  mkdir -p "${seed_out}"

  log "VIFNet Resolution-Constrained seed=${seed}"
  BENCHMARK_FAIR_MODE=1 \
  "${PYTHON_BIN}" "${WORK_ROOT}/profile_model_metrics.py" \
    --repo vif-net \
    --model VIFNet \
    --shape 1,5,2976,256 \
    --device cuda 2>&1 | tee "${seed_out}/VIFNet_profile.log"

  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  BENCHMARK_FAIR_MODE=1 \
  ${TORCHRUN_BIN} --nproc_per_node="${NUM_GPUS}" --master_port="${port_base}" \
    "${WORK_ROOT}/benchmark/unified_benchmark_train.py" \
    --model VIFNet \
    --data-root "${DATA_ROOT}" \
    --global-map-csv "${GLOBAL_MAP}" \
    --stats-json "${STATS_JSON}" \
    --align-multiple 32 \
    --align-mode crop \
    --seed "${seed}" \
    --epochs "${EPOCHS}" \
    --save-interval 5 \
    --vis-interval 5 \
    --batch-size "${BATCH_PER_GPU}" \
    --output-path "${seed_out}" \
    --save-name checkpoint \
    --sync-bn 2>&1 | tee "${seed_out}/VIFNet_seed${seed}.log"

  BENCHMARK_FAIR_MODE=1 \
  "${PYTHON_BIN}" "${WORK_ROOT}/benchmark/unified_benchmark_test.py" \
    --model VIFNet \
    --checkpoint "${seed_out}/checkpoint.pth" \
    --data-root "${DATA_ROOT}" \
    --global-map-csv "${GLOBAL_MAP}" \
    --stats-json "${STATS_JSON}" \
    --align-multiple 32 \
    --align-mode crop \
    --batch-size "${BATCH_PER_GPU}" \
    --export-dir "${seed_out}/eval" \
    --benchmark-eval "${WORK_ROOT}/benchmark/benchmark_eval.py" \
    --benchmark-eval-json "${seed_out}/eval/metrics.json" \
    --benchmark-eval-lpips-mode real 2>&1 | tee -a "${seed_out}/VIFNet_seed${seed}.log"
}

main() {
  ensure_runtime_deps
  ensure_split_stats
  run_main_eight
  IFS=',' read -r -a seed_arr <<< "${SEEDS}"
  for seed in "${seed_arr[@]}"; do
    run_convnext_kaggle "${seed}"
    run_vifnet "${seed}"
  done
  log "all Resolution-Constrained tasks completed"
}

main "$@"
