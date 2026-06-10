# Benchmark 容器构建与运行（一步步）

本文档为 **fwi-benchmark:cu118** 容器的权威说明，被 `benchmark/README.md`、`IMPLEMENTATION_GUIDE_ZH.md`、`run_benchmark_docker.sh` 等引用。

**准备就绪清单**：① 镜像已构建 ② 数据已挂载 ③ preflight 通过 ④ VelocityGAN 点火通过 → 可执行步骤 5 全量测试。

## 依赖清单（已写入 requirements.txt + Dockerfile）

| 类别 | 包 | 版本 | 用途 |
|------|-----|------|------|
| 基础 | numpy | 1.26.4 | 全模型 |
| 基础 | numba | >=0.58.1 | VelocityGAN（必须≥0.58.1 兼容 numpy 1.26） |
| 基础 | scipy, sklearn, skimage | 固定 | 数据处理 |
| 基础 | einops | 0.8.0 | FuteFWI Transformer |
| 评测 | lpips | 0.1.4 | LPIPS 指标 |
| 评测 | pytorch-msssim | 1.0.0 | SSIM 指标 |
| 模型 | timm | >=0.9.0 | ConvNeXt-FWI |
| 其他 | thop, opencv, h5py, PyWavelets | 固定 | profile/数据 |

PyTorch 2.3.1+cu118 在 Dockerfile 中单独安装。

---

## 步骤 1：构建新容器镜像

在 **repository 根目录** 执行（注意 `-f` 和 `.`）：

```bash
cd <WORK_ROOT>
docker build -t fwi-benchmark:cu118 -f benchmark/docker/Dockerfile .
```

- 国内已配置：`FROM docker.1ms.run/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`
- pip 使用清华源（`PIP_INDEX`）
- 构建时间约 5–15 分钟

---

## 步骤 2：启动容器并挂载

```bash
docker run --gpus all -it --rm \
  -v <WORK_ROOT>:/workspace/repo \
  -v <DATA_ROOT>:<DATA_ROOT> \
  --shm-size=32g \
  -w /workspace/repo \
  fwi-benchmark:cu118 /bin/bash
```

- `--gpus all`：使用全部 GPU
- `-v`：代码和数据挂载
- `--shm-size=32g`：避免 DataLoader 多进程共享内存不足

---

## 步骤 3：容器内预检查

```bash
export WORK_ROOT=/workspace/repo
export DATA_ROOT=<DATA_ROOT>
bash benchmark/preflight_check.sh
```

通过则继续，否则先修复报错。

---

## 步骤 4：VelocityGAN 点火测试（1 epoch）

```bash
export WORK_ROOT=/workspace/repo
export DATA_ROOT=<DATA_ROOT>
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash benchmark/pre_launch_checklist.sh
```

或仅跑 VelocityGAN 单模型：

```bash
TASK_FILTER=VelocityGAN SEEDS=42 EPOCHS=2 bash run_benchmark_suite.sh
```

**统一 batch**：默认 `BATCH_PER_GPU=8`（总 64），全员公平。

---

## 步骤 5：全量 Benchmark（准备就绪后执行）

**前置确认**：步骤 1–4 均通过（镜像构建、挂载、preflight、VelocityGAN 点火）。

```bash
# 宿主机一键启动（推荐）
docker run --rm --gpus all --shm-size=32g \
  -v <WORK_ROOT>:/workspace/repo \
  -v <DATA_ROOT>:<DATA_ROOT> \
  -e SEEDS=42 \
  fwi-benchmark:cu118 bash /workspace/repo/run_benchmark_docker.sh
```

或容器内执行：

```bash
export WORK_ROOT=/workspace/repo
export DATA_ROOT=<DATA_ROOT>
export SEEDS=42
bash run_benchmark_docker.sh
```

（`run_benchmark_docker.sh` 会安装额外依赖，本镜像已包含主要依赖，可跳过）

---

## 模型特定配置（T4 15GB × 8 卡）

| 模型 | Batch/卡 | 说明 |
|------|----------|------|
| 多数模型 | 4 | `BATCH_PER_GPU=4`，总 batch=32 |
| FuteFWI | 2 | Transformer 显存大，`BATCH_FUTEFWI=2`；benchmark 对 FuteFWI 训练单独设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解 CuDNN 错误 |
| ABA_FWI | 2 | 显存大，`BATCH_ABA_FWI=2` |

### ABA_FWI 输出尺寸修复（2026-02-28）

ABA_FWI 原始 UNet 解码器可能输出 255×255，与 benchmark 统一 label (256,256) 不匹配，导致 `RuntimeError: The size of tensor a (255) must match the size of tensor b (256)`。

**修复**：`ABA-FWI/ABA-FWI_2.0/net/ABA_FWI.py` 的 `forward` 末尾增加 `F.interpolate` 对齐，与 FuteFWI/InversionNet 一致。详见 `benchmark/CHANGELOG_BENCHMARK.md` 第十四节。

---

## 常见问题

| 问题 | 处理 |
|------|------|
| `COPY benchmark/docker/requirements.txt` 失败 | 必须在 repository 根目录执行 `docker build` |
| 数据路径不存在 | 调整 `-v` 中宿主机路径；确认 `<DATA_ROOT>` 存在 |
| preflight 报 7 个 repo 不存在 | 确认挂载 `-v <WORK_ROOT>:/workspace/repo` |
| LPIPS 下载失败 | benchmark 默认 `--lpips-mode real`（论文协议）；网络受限时可临时用 `--lpips-mode proxy`，但结果与论文 LPIPS 不可比 |
| VelocityGAN inplace 错误 | 已修复：Discriminator 改用 InstanceNorm（norm="in"），与 WGAN-GP 的 create_graph=True 兼容 |
| ABA_FWI 255 vs 256 报错 | 已修复：forward 末尾 `F.interpolate` 对齐 output_size |
| FuteFWI CUDNN_STATUS_INTERNAL_ERROR | 已缓解：`run_benchmark_suite.sh` 对 FuteFWI 训练单独设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；若仍失败可尝试升级 CUDA/cuDNN 或更新驱动 |
| FuteFWI NCCL Error 1 | 已修复：`train_futefwi.py` 改用 DDP（DistributedDataParallel）替代 DataParallel，与其他模型一致 |
| DDNet70/TU_Net profile NA | 已修复：`profile_model_metrics.py` 将 `infer_ms` 提前到 `try_flops_g` 之前，避免 thop hooks 导致 `ReLU total_ops` 报错 |
| ABA_FWI/FCNVMB profile NA | 已修复：`profile_model_metrics.py` 移除 `ABA_Loss` 导入，aba-fwi 分支仅导入 `ABA_FWI`、`FCNVMB_FWI` |
