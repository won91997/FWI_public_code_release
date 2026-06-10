#!/usr/bin/env bash
set -e

WORK_ROOT="${WORK_ROOT:-/workspace/repo}"
DATA_ROOT="${DATA_ROOT:-}"

echo "==== Preflight: system ===="
python --version
python - <<'PY'
import torch, torchvision, numpy, numba, scipy, sklearn, skimage, lpips, cv2, matplotlib, timm, einops
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("numpy:", numpy.__version__)
print("numba:", numba.__version__, "(>=0.58.1 for VelocityGAN)")
print("scipy:", scipy.__version__)
print("sklearn:", sklearn.__version__)
print("skimage:", skimage.__version__)
print("lpips:", lpips.__version__ if hasattr(lpips, "__version__") else "ok")
print("cv2:", cv2.__version__)
print("matplotlib:", matplotlib.__version__)
print("timm:", timm.__version__ if hasattr(timm, "__version__") else "ok")
print("einops:", einops.__version__, "(FuteFWI)")
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu_count:", torch.cuda.device_count())
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

echo "==== Preflight: paths ===="
test -d "${WORK_ROOT}/OpenFWI"
test -d "${WORK_ROOT}/FuTE-FWI"
test -d "${WORK_ROOT}/DCNet"
test -d "${WORK_ROOT}/ddnet"
test -d "${WORK_ROOT}/TU-Net"
test -d "${WORK_ROOT}/ABA-FWI"
test -d "${WORK_ROOT}/ConvNeXt-Kaggle"
test -d "${WORK_ROOT}/VIF-Net"
echo "OK: repository folders exist (8 baseline directories)"

test -d "${DATA_ROOT}"
test -d "${DATA_ROOT}/seismic_full"
test -d "${DATA_ROOT}/vmodel_full"
test -f "${DATA_ROOT}/split_by_inline.npy"
test -f "${DATA_ROOT}/sample_inline.npy"
echo "OK: dataset folders/files exist"

echo "==== Preflight: split generation dry run ===="
python "${WORK_ROOT}/benchmark/make_split_from_inline.py" \
  --data-root "${DATA_ROOT}" \
  --out-dir "${WORK_ROOT}/benchmark/generated_split"
echo "OK: split files generated"

echo "==== Preflight: train-only stats ===="
python "${WORK_ROOT}/benchmark/compute_train_stats.py" \
  --global-map-csv "${WORK_ROOT}/benchmark/generated_split/global_map.csv" \
  --out-json "${WORK_ROOT}/benchmark/generated_split/train_stats.json"
echo "OK: train-only stats generated (data_robust_max from p99 of |seismic|)"

echo "==== Preflight: evaluator smoke test ===="
python - <<'PY'
import numpy as np, tempfile, os, subprocess, sys
pred = np.random.rand(8, 64, 64).astype("float32")
gt = np.random.rand(8, 64, 64).astype("float32")
work_root = os.environ.get("WORK_ROOT", "/workspace/repo")
with tempfile.TemporaryDirectory() as d:
    pp = os.path.join(d, "pred.npy")
    gp = os.path.join(d, "gt.npy")
    np.save(pp, pred)
    np.save(gp, gt)
    cmd = [sys.executable, os.path.join(work_root, "benchmark", "benchmark_eval.py"), "--pred", pp, "--gt", gp, "--lpips-mode", "proxy"]
    subprocess.run(cmd, check=True)
print("OK: evaluator smoke test passed")
PY

echo "==== Preflight done ===="
