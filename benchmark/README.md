# Benchmark Framework

## Public Release Note

The paper-facing entry points are the root-level scripts:

- `run_resolution_constrained.sh` for the Resolution-Constrained setting.
- `run_full_resolution_nonfair.sh` for the Target-Resolution setting.

Some files in this directory are historical implementation notes from the
comparison workspace. They may mention legacy terms such as `Fair`, `NonFair`,
`ConvNeXtFWI`, or deleted local log folders. For reproducing the paper protocol,
use `README.md` and `MODEL_CODE_MAPPING.md` at the repository root as the source
of truth.

# Unified Benchmark Workflow

## 0) Docker 容器（推荐 fwi-benchmark:cu118）

**推荐使用项目自建镜像** `fwi-benchmark:cu118`（PyTorch 2.3.1+cu118，依赖已固定）：

```bash
cd /path/to/repo
docker build -t fwi-benchmark:cu118 -f benchmark/docker/Dockerfile .
docker run --gpus all -it --rm -v $(pwd):/workspace/repo -v <DATA_ROOT>:/workspace/data --shm-size=32g fwi-benchmark:cu118 /bin/bash
```

详见 `benchmark/docker/BUILD_AND_RUN.md`。

**国内加速**：拉取镜像时加前缀 `docker.1ms.run/`；Dockerfile 的 `FROM` 已使用该前缀。

## 1) Generate one shared split (train/val/test)

Use your fixed split metadata:

```bash
python benchmark/make_split_from_inline.py \
  --data-root <DATA_ROOT> \
  --out-dir benchmark/generated_split
```

Outputs:
- `benchmark/generated_split/train_idx.txt`
- `benchmark/generated_split/val_idx.txt`
- `benchmark/generated_split/test_idx.txt`
- `benchmark/generated_split/global_map.csv`

All models must use these same indices.

## 2) Unified evaluation

Each model should export:
- prediction: `pred.npy`
- ground truth: `gt.npy`

Then run:

```bash
python benchmark/benchmark_eval.py --pred pred.npy --gt gt.npy --out-json eval.json
```

Metrics:
- MSE, MAE, PSNR, SSIM, L1-Grad, LPIPS (proxy)

## 3) Strict training budget

Set in env before running:

```bash
export BUDGET_MODE=epoch      # or wallclock
export WALLCLOCK_HOURS=6
```

`run_benchmark_suite.sh` exports them as:
- `BENCHMARK_BUDGET_MODE`
- `BENCHMARK_WALLCLOCK_HOURS`

Your train scripts should read these and enforce budget.

## 4) Multi-seed runs and aggregate

```bash
export SEEDS=42
bash run_benchmark_suite.sh
```

Outputs:
- raw per-run csv: `benchmark_logs/benchmark_metrics.csv`
- optional multi-seed aggregation: `benchmark_logs/benchmark_metrics_agg.csv` (paper uses seed 42)

## 5) How to integrate existing repos

For each repo's eval command:
1. save predictions and gts to `.npy`
2. call `benchmark/benchmark_eval.py`
3. print metric lines:
   - `MSE: ...`
   - `MAE: ...`
   - `PSNR: ...`
   - `SSIM: ...`
   - `L1-Grad: ...`
   - `LPIPS: ...`

Then `run_benchmark_suite.sh` can parse and write unified tables.

