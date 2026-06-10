# 统一基准测试实施说明（详细版）

本文档对应当前工程中已经创建好的脚本与目录，目标是实现：

1. 所有模型使用同一份 `train/val/test`（来自 `split_by_inline.npy`）  
2. 所有模型用同一个 evaluator 计算统一指标  
3. 统一训练预算（`epoch` 或 `wall-clock` 二选一）  
4. 每个模型使用固定 seed 42 运行，并生成结果汇总表  

---

## 0. 当前已提供的脚本

位于 `benchmark/`：

- `make_split_from_inline.py`  
  基于 `split_by_inline.npy` 和 `sample_inline.npy` 生成统一切分文件
- `compute_train_stats.py`  
  扫描训练集，计算 data/label 的 min/max 统计，输出 `train_stats.json`。  
  **必须对全量训练集运行（不加 `--max-samples`）**，否则归一化参数不精确
- `unified_loader.py`  
  统一 DataLoader，所有模型共用。支持 `time_downsample`、`channel_mode`、`align_multiple`、`align_mode` 等参数
- `benchmark_eval.py`  
  统一评测脚本，输入预测和真值 `.npy`，输出 MSE/MAE/PSNR/SSIM/L1-Grad/LPIPS 六项指标
- `aggregate_seeds.py`  
  对固定 seed 结果生成 CSV/Markdown 汇总
- `README.md`  
  简明流程说明
- `docker/Dockerfile`  
  推荐统一容器镜像 `fwi-benchmark:cu118`（CUDA11.8 + PyTorch2.3.1），详见 `docker/BUILD_AND_RUN.md`
- `docker/requirements.txt`  
  Python 依赖固定版本列表
- `preflight_check.sh`  
  运行前环境/依赖/数据/脚本一致性检查（会自动生成 split 和全量 train_stats）
- `pre_launch_checklist.sh`  
  VelocityGAN 8 卡 DDP 点火测试（1 epoch），确认无 sync-bn 兼容问题
- `velocitygan_batch_stress.sh`  
  VelocityGAN batch 压力测试（4/8/16/32），用于确定 BATCH_PER_GPU
- `futefwi_batch_stress.sh`  
  FuteFWI batch 压力测试（2/4/8），用于确定 BATCH_FUTEFWI
- `FAIRNESS_CHECK.md`  
  公平性检查报告、已修复阻塞项、启动前确认
- `OUTPUT_PARAMS.md`  
  输出文件说明、CSV 列、profile 格式
- `CHANGELOG_BENCHMARK.md`  
  完整修改记录与公平性说明

根目录：

- `run_benchmark_suite.sh`  
  严格版总控脚本（`stats生成 -> profile -> train -> eval -> 聚合`，已内置自动检测 split 和 train_stats）
- `profile_model_metrics.py`  
  模型 profiling 脚本（Params / FLOPs / 推理速度），支持 `--repo openfwi|futefwi|dcnet|ddnet|tu-net|aba-fwi|convnext-fwi|convnext-kaggle --model <名称> --shape B,C,H,W`
- `unified_benchmark_train.py`  
  统一训练脚本，支持 10 个唯一架构（InversionNet/VelocityGAN/FuteFWI/DCNet/DDNet70/TU_Net/ABA_FWI/ConvNeXtFWI/ConvNeXtKaggle/FCNVMB），使用 UnifiedFWIDataset + 各模型原生 Loss
- `run_convnext_kaggle_dual.sh`  
  ConvNeXtKaggle 双模式训练（Fair + NonFair），batch=2、seed=42、100 epochs，详见 `FAIR_BENCHMARK.md` 中「ConvNeXtKaggle 双模式训练」
- `unified_benchmark_test.py`  
  统一测试脚本，加载 checkpoint、推理、导出 pred.npy/gt.npy、调用 benchmark_eval。**InversionNet/VelocityGAN/FuteFWI 的 benchmark 评测均使用此脚本**（OpenFWI 无 test.py，FuTE-FWI test.py 无 benchmark 参数）

---

## 0.1 Docker 环境准备（可直接执行）

### A. 宿主机要求（必须）

1. Linux 服务器（推荐 Ubuntu 20.04/22.04）
2. NVIDIA Driver 正常（`nvidia-smi` 可用）
3. Docker 已安装
4. NVIDIA Container Toolkit 已安装（容器内可见 GPU）

检查命令：

```bash
nvidia-smi
docker --version
docker run --rm --gpus all docker.1ms.run/nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

> 第 3 条命令如果失败，先修复 `nvidia-container-toolkit`，否则容器内无法使用 GPU。

> **国内用户**：所有 Docker 拉取请加前缀 `docker.1ms.run/`，例如：
> - `docker pull nvidia/cuda:xxx` → `docker pull docker.1ms.run/nvidia/cuda:xxx`
> - `docker pull nvcr.io/nvidia/xxx` → `docker pull docker.1ms.run/nvcr.io/nvidia/xxx`

### B. 构建镜像（在项目根目录）

`benchmark/docker/Dockerfile` 的 `FROM` 已使用 `docker.1ms.run/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`，国内构建时自动走镜像加速。

```bash
cd /path/to/repo
docker build -t fwi-benchmark:cu118 -f benchmark/docker/Dockerfile .
```

### C. 启动容器（8 卡）

```bash
docker run --gpus all --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/to/repo:/workspace/repo \
  -v <DATA_ROOT>:<DATA_ROOT> \
  -w /workspace/repo \
  --name fwi-benchmark \
  -it fwi-benchmark:cu118 /bin/bash
```

说明：

- `--shm-size=32g`：**必须**，避免 NCCL/DataLoader 多进程共享内存不足（64g 也可，32g 已足够）
- `-v` 两个挂载：
  - 代码目录挂载到 `/workspace/repo`
  - 数据目录按你实际路径挂载到容器内同路径

### D. 容器内预检查（强烈建议每次都跑）

```bash
export WORK_ROOT=/workspace/repo
export DATA_ROOT=<DATA_ROOT>
bash benchmark/preflight_check.sh
```

该脚本会检查：

- Python + 关键依赖版本
- CUDA/GPU 是否可见
- 八个仓库目录是否存在（OpenFWI、FuTE-FWI、DCNet、ddnet、TU-Net、ABA-FWI、ConvNeXt-FWI、ConvNeXt-Kaggle）
- 数据目录与关键 `.npy` 是否存在
- 统一 split 生成是否成功
- 统一 evaluator 是否可运行

如果这一步失败，不要直接开始 benchmark，先修复环境。

### E. 8 卡 DDP 配置（必须，真实环境）

**所有 benchmark 测试统一使用 8 卡 DDP，禁止单卡。**

| 阶段 | GPU 配置 | 说明 |
|------|----------|------|
| **训练** | 8 卡 DDP | 所有 9 个模型均用 `torchrun --nproc_per_node=8` 启动 |
| profile | 首卡 | 测参数量、FLOPs、单卡推理时间 |
| eval | 首卡 | 推理 + 算指标 |

- 启动容器必须使用 `docker run --gpus all` 或 `--gpus '"device=0,1,2,3,4,5,6,7"'` 暴露 8 张卡
- `run_benchmark_docker.sh`、`run_benchmark_suite.sh`、`pre_launch_checklist.sh` 均已配置 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- 总有效 batch = `BATCH_PER_GPU × 8`（如 BATCH_PER_GPU=4 则总 batch=32）

---

## 0.2 依赖版本固定建议（避免报错）

已在 `benchmark/docker/requirements.txt` 固定版本。  
核心组合如下（与 `Dockerfile` 一致）：

- CUDA: 11.8
- PyTorch: `2.3.1+cu118`
- torchvision: `0.18.1+cu118`
- torchaudio: `2.3.1+cu118`
- numpy: `1.26.4`
- numba: `>=0.58.1`（必须 ≥0.58.1 才能与 numpy 1.26 兼容，否则 VelocityGAN 等会崩溃）
- scipy: `1.11.4`
- scikit-image: `0.22.0`
- scikit-learn: `1.4.2`
- lpips: `0.1.4`
- pytorch-msssim: `1.0.0`
- thop: `0.1.1.post2209072238`
- einops: `0.8.0`（FuteFWI Transformer 模型必须）
- matplotlib: `3.8.4`（可视化与画图）
- tqdm: `4.66.4`（进度条）
- tensorboard: `2.16.2`（训练日志可视化，可选）

**全仓库依赖汇总**（一个 Docker 容器跑所有模型所需的全部 Python 包）：

以下版本与 `benchmark/docker/requirements.txt` 完全一致。构建 Docker 镜像时会自动安装。

| 包名 | 固定版本 | 使用模型/仓库 | 必须/可选 |
|------|---------|-------------|---------|
| `torch` | `2.3.1+cu118` | 全部 | 必须 |
| `torchvision` | `0.18.1+cu118` | 全部 | 必须 |
| `torchaudio` | `2.3.1+cu118` | — | 可选 |
| `numpy` | `1.26.4` | 全部 | 必须 |
| `numba` | `>=0.58.1` | VelocityGAN 等（scipy/skimage 间接依赖） | 必须 |
| `scipy` | `1.11.4` | 全部 | 必须 |
| `scikit-image` | `0.22.0` | benchmark_eval (SSIM/PSNR) | 必须 |
| `scikit-learn` | `1.4.2` | 部分仓库的 metrics | 必须 |
| `lpips` | `0.1.4` | benchmark_eval (LPIPS 指标) | 必须 |
| `pytorch-msssim` | `1.0.0` | benchmark_eval (SSIM 指标) | 必须 |
| `thop` | `0.1.1.post2209072238` | profile_model_metrics (FLOPs/Params) | 必须 |
| `einops` | `0.8.0` | FuTE-FWI/FuteFWI/Ablation_1/Ablation_2 (Transformer `rearrange`) | 必须 |
| `timm` | `>=0.9.0` | ConvNeXt-FWI（ConvNeXt backbone） | 必须 |
| `matplotlib` | `3.8.4` | 训练可视化 | 必须 |
| `pandas` | `2.2.2` | aggregate_seeds, CSV 处理 | 必须 |
| `h5py` | `3.11.0` | 部分仓库的数据读取 | 必须 |
| `PyWavelets` | `>=1.4.0` | ABA-FWI (wtconv 小波卷积) | 必须 |
| `opencv-python` | `4.9.0.80` | 部分仓库的图像处理 | 必须 |
| `tqdm` | `4.66.4` | 训练进度条 | 必须 |
| `tensorboard` | `2.16.2` | 训练日志可视化 | 可选 |

> **如何手动安装（不用 Docker 时）**：
> ```bash
> # 1. 安装 PyTorch（CUDA 11.8）
> pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 \
>     --index-url https://download.pytorch.org/whl/cu118
> # 2. 安装其他依赖
> pip install -r benchmark/docker/requirements.txt
> ```

注意：

- **不要随意升级 torch / torchvision**，否则部分仓库或 `lpips` 可能出现兼容问题
- 若你必须升级，务必先重新运行 `preflight_check.sh`
- DCNet 仓库内部有自定义的 `DCNet_config.py`、`loss.py` 等，无额外 pip 依赖

---

## 0.3 一键执行顺序（给实施同学）

1. 构建镜像 `fwi-benchmark:cu118`（`docker build -t fwi-benchmark:cu118 -f benchmark/docker/Dockerfile .`）  
2. 启动容器并挂载代码 + 数据（`docker run --gpus all ... fwi-benchmark:cu118 ...`）  
3. `bash benchmark/preflight_check.sh`  
4. 生成统一 split（`make_split_from_inline.py`）  
5. 计算全量 train-only 统计（`compute_train_stats.py`，不加 `--max-samples`）  
6. 跑单模型单 seed 验证链路  
7. 跑全模型单 seed（`bash run_benchmark_suite.sh`，步骤 4-5 已内置自动检查）  
8. 查看 `benchmark_metrics.csv` 与 `benchmark_metrics_agg.csv`

参考命令：

```bash
export WORK_ROOT=/workspace/repo
export DATA_ROOT=<DATA_ROOT>
export GPU_IDS=0,1,2,3,4,5,6,7
export SEEDS=42
export BUDGET_MODE=wallclock
export WALLCLOCK_HOURS=6

python benchmark/make_split_from_inline.py --data-root "${DATA_ROOT}" --out-dir benchmark/generated_split
bash run_benchmark_suite.sh
```

---

## 0.4 常见环境报错与修复

1. **`RuntimeError: CUDA error` / 容器看不到 GPU**  
   - 检查 `docker run --gpus all ...`
   - 检查宿主机驱动与 toolkit

2. **`libGL.so` / `cv2` 导入失败**  
   - 镜像里已安装 `libgl1 libglib2.0-0`，确认用的是本文 Dockerfile 构建镜像

3. **`lpips` 首次运行下载模型失败**  
   - 镜像构建阶段已做 warmup；若网络受限，可提前离线缓存

4. **DataLoader worker 挂掉 / shared memory 不足**  
   - 增大 `--shm-size`（推荐 32g，不足时试 64g）
   - 临时把 `num_workers` 调小验证

5. **不同仓库路径硬编码导致找不到数据**  
   - 优先通过新增 CLI 参数覆盖
   - 必要时在仓库 `path_config.py` 中统一到容器路径

6. **`numba 0.57.1 requires numpy<1.25`（VelocityGAN 等崩溃）**  
   - fwi-benchmark:cu118 等镜像可能自带 numba 0.57.1，与 numpy 1.26 不兼容
   - 推荐使用项目自建镜像 `fwi-benchmark:cu118`（已固定 numba>=0.58.1）；或 `pip install "numba>=0.58.1"`

---

## 1. 统一数据入口：按 inline 切分

### 1.1 目标

你已经固定了数据集和切分元信息，因此所有模型必须共用同一份样本索引：

- `split_by_inline.npy`：0/1/2 分别表示 train/val/test
- `sample_inline.npy`：每个样本对应 inline 编号

### 1.2 执行命令

```bash
python benchmark/make_split_from_inline.py \
  --data-root <DATA_ROOT> \
  --out-dir benchmark/generated_split
```

### 1.3 产物

- `benchmark/generated_split/train_idx.txt`
- `benchmark/generated_split/val_idx.txt`
- `benchmark/generated_split/test_idx.txt`
- `benchmark/generated_split/global_map.csv`

其中 `global_map.csv` 记录了：

- 全局样本 id
- 来自哪个 `seismic*.npy/vmodel*.npy`
- 文件内索引
- inline id
- split 标记

### 1.4 接入原则

每个仓库的数据读取器最终都要支持“按全局样本 id 取样”。  
最简单做法：在适配层里读取 `global_map.csv`，只用指定 split 的样本。

---

## 1.5 统一适配协议（Unified Adaptation Protocol，强制执行）

你的数据是高分辨率多炮：

- 输入：`(5, 3000, 256)`
- 输出：`(1, 256, 256)`（评测时统一为这个 shape）

对所有模型，执行 **“数据不动，模型动”** 协议：

1. **统一 DataLoader（必须）**  
   不允许直接使用各仓库默认 dataset（很多仓库硬编码了 70x70 或 1-shot）。
   已提供：`benchmark/unified_loader.py`
2. **统一 train-only 统计（必须）**  
   归一化参数只能来自训练集。
   已提供：`benchmark/compute_train_stats.py`
3. **统一 Clip + 归一化（必须）**  
   所有模型必须通过 `UnifiedFWIDataset` 加载数据，其内部已实现：
   - **Seismic**：先 `clip(x, -data_robust_max, data_robust_max)`，再 `/ data_robust_max` → [-1, 1]
   - **Vmodel**：min-max 到 [0,1]，clip 后 `(x-0.5)*2` → [-1, 1]
   禁止各仓库使用自定义 dataset 或不同归一化策略，否则公平性失效。
4. **统一形状约束（必须）**  
   - 模型输入必须接受 `(N, 5, 3000, 256)`（或统一降采样后的固定形状，如 `align_multiple=32` 裁剪至 `(N, 5, 2976, 256)`）
   - 模型输出必须是 `(N, 1, 256, 256)`
   - 所有模型已改为**分辨率无关（Resolution-Agnostic）**架构：编码器末端使用 `AdaptiveAvgPool2d((1,1))`，解码器末端使用 `F.interpolate` 到 `output_size`，通过构造函数 `output_size=(256,256)` 参数控制
5. **统一 OOM 降维策略（必须）**  
   如果显存不足，所有模型必须使用同一降维策略，不能每个模型单独选。

---

## 1.6 Unified Loader 使用方法

### A. 先生成 split 与 global_map

```bash
python benchmark/make_split_from_inline.py \
  --data-root <DATA_ROOT> \
  --out-dir benchmark/generated_split
```

### B. 计算 train-only 统计

```bash
python benchmark/compute_train_stats.py \
  --global-map-csv benchmark/generated_split/global_map.csv \
  --out-json benchmark/generated_split/train_stats.json
```

> **重要**：必须对全量训练集运行，**不要加 `--max-samples`**。  
> 早期版本的 `preflight_check.sh` 曾使用 `--max-samples 2000`（仅采样 2000 个样本计算统计），
> 这会导致归一化参数不精确，影响所有模型的训练效果和评测公平性。已修正。  
> 全量计算可能需要 10-30 分钟（取决于数据量），但只需运行一次，产物 `train_stats.json` 会被缓存。  
> `run_benchmark_suite.sh` 启动时会自动检测 `train_stats.json` 是否存在，不存在则自动全量生成。

### C. 在训练/测试脚本中引用

示例（伪代码）：

```python
from benchmark.unified_loader import UnifiedFWIDataset
from torch.utils.data import DataLoader

train_set = UnifiedFWIDataset(
    data_root="<DATA_ROOT>",
    split="train",
    global_map_csv="benchmark/generated_split/global_map.csv",
    stats_json="benchmark/generated_split/train_stats.json",
    time_downsample=1,
    channel_mode="all",
)
train_loader = DataLoader(train_set, batch_size=..., shuffle=True, num_workers=...)
```

---

## 1.7 OOM 情况下的统一降维策略（只允许选一种并固定）

推荐优先顺序：

1. **时间下采样（优先）**
   - `time_downsample=3`，输入变为 `(5, 1000, 256)`
2. **通道降维（次选）**
   - `channel_mode="middle"`，输入变为 `(1, 3000, 256)`

重要：一旦选择某策略，所有模型必须一致执行，并写入实验设置。

此外，针对 CNN 下采样链路的“尺寸整除陷阱”，统一 loader 已支持自动对齐：

- `--align-multiple 32`
- `--align-mode crop|pad`
- `--target-time` / `--target-width`（可选显式指定）

推荐：

- 对 `3000x256` 输入，benchmark 固定使用 `--align-mode crop --align-multiple 32`。  
  时间维**裁剪**到 32 的倍数（3000 → 2976，丢弃末尾 24 个时间步），**不是填充**。实现见 `unified_loader._align_shape()`：`crop` 时 `return arr[:, :t, :w].copy()`。可避免 skip-connection 尺寸报错。

---

## 1.8 分辨率无关（Resolution-Agnostic）架构改造详解

### 问题背景

原始 OpenFWI / FuTE-FWI 的 `InversionNet`、`Generator（VelocityGAN）` 模型在编码器末端和解码器末端
均有**硬编码的空间尺寸假设**（针对原始 70x70 输出设计），导致无法接受 256 宽度的高分辨率数据。具体表现：

1. **编码器末端**：`convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, ceil(70/8)), padding=0)`  
   该卷积核大小 `(8, 9)` 是为把编码特征精确压缩到 `(1, 1)` 而硬算的。输入宽度不是 70 时，输出不再是 `(1,1)`，导致解码器崩溃。

2. **解码器末端**：`F.pad(x, [-5, -5, -5, -5])` 或 `F.pad(x, [-5, -6, -10, -10])`  
   用负数 padding 裁剪解码输出到精确的 `(70, 70)` 或 `(140, 70)`。新数据下裁剪值错误，输出尺寸不对。

### 解决方案（方案 A：改模型适配数据）

核心原则：**保持模型层数和通道结构不变**，仅替换两处与空间尺寸耦合的组件。

#### 改动 1：编码器末端 → AdaptiveAvgPool2d

将 `self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(...), padding=0)` 替换为：

```python
self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
self.bottleneck = ConvBlock(dim4, dim5, kernel_size=1, padding=0)
```

效果：无论编码器输出的空间尺寸是多少（如 `(256, 24, 32)` 或 `(256, 8, 9)`），`AdaptiveAvgPool2d` 总是将其压缩为 `(dim4, 1, 1)`，随后 `bottleneck` 升维到 `(dim5, 1, 1)`。

> 注意：这等效于 Global Average Pooling，参数量比原始 convblock8（大卷积核）更少。

#### 改动 2：解码器末端 → F.interpolate

将 `F.pad(x, [...])` 行**直接删除**，在最后一层卷积（`deconv6` / `deconv7`）之后追加：

```python
if x.shape[-2:] != tuple(self.output_size):
    x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
```

效果：解码器正常通过转置卷积上采样（如输出 `(1, 80, 80)` 或 `(1, 160, 81)`），最终通过双线性插值统一缩放到目标尺寸 `(256, 256)`。

#### 改动 3：构造函数新增 `output_size` 参数

所有被改动的模型类新增参数 `output_size=(256, 256)`（tuple），在 `__init__` 中存储：

```python
class InversionNet(nn.Module):
    def __init__(self, dim1=32, ..., output_size=(256, 256), **kwargs):
        super().__init__()
        self.output_size = output_size
        ...
```

训练/测试脚本通过 CLI 参数 `--output-height 256 --output-width 256` 传入，然后 `output_size=(args.output_height, args.output_width)` 传给模型构造函数。

### 已改造的模型完整清单（全部 16 个已通过验证）

所有模型已通过 forward pass 验证：输入 `(1, 5, 2976, 256)` → 输出 `(1, 1, 256, 256)` 或 `(1, 2, 256, 256)`。

#### A 类：InversionNet 族（AdaptiveAvgPool + bottleneck + F.interpolate）

这类模型共享同一种改造模式：编码器末端 `convblock8`（硬编码 kernel）→ `AdaptiveAvgPool2d((1,1))` + `bottleneck(Conv 1x1)`；解码器末端 `F.pad([-5,-5,-5,-5])` → `F.interpolate(output_size)`。

| # | 模型 | 文件 | 新增参数 |
|---|------|------|---------|
| 1 | OpenFWI/**InversionNet** | `OpenFWI/network.py` | `output_size=(256,256)` |
| 2 | OpenFWI/**UPFWI** (FCN4_Deep_Resize_2) | `OpenFWI/network.py` | `output_size=(256,256)` |
| 3 | FuTE-FWI/**Generator** (VelocityGAN) | `FuTE-FWI/models/VelocityGAN.py` | `output_size=(256,256)` |
| 4 | FuTE-FWI/**InversionNet** | `FuTE-FWI/models/InversionNet.py` | `output_size=(256,256)` |
| 5 | DCNet/**InversionNet** | `DCNet/func/comparison_net.py` | `output_size=(256,256)` |
| 6 | ddnet/**InversionNet** | `ddnet/net/InversionNet.py` | `output_size=(256,256)` |
| 7 | TU-Net/**InversionNet** | `TU-Net/model/InversionNet.py` | `output_size=(256,256)` |

> 注意：#5、#6、#7 是 OpenFWI InversionNet 在 DCNet/ddnet/TU-Net 仓库中的副本，改造方式完全一致。

#### B 类：DCNet（AdaptiveAvgPool + bottleneck + F.interpolate）

| # | 模型 | 文件 | 改造方式 |
|---|------|------|---------|
| 8 | DCNet/**DCNet** | `DCNet/DCNet.py` | `lastBlock`（硬编码 `kernel_size=(8, ceil(70*s/8))`）→ `global_pool(AdaptiveAvgPool2d) + bottleneck(Conv 1x1)`；`F.pad([-5,-5,-5,-5])` → `F.interpolate(output_size)`；`DCModel` 新增 `output_size` 参数透传 |

#### C 类：DDNet70 族（decoder output_lim 动态化）

这类模型使用 UNet 架构，解码器中的 `unetUp`/`netUp` 层通过 `output_lim` 参数控制输出尺寸。改造方式：将硬编码的 `[9,9],[18,18],[35,35],[70,70]` 改为基于 `output_size` 动态计算 `[oh//8, ow//8], [oh//4, ow//4], [oh//2, ow//2], [oh, ow]`。

| # | 模型 | 文件 | 新增参数 |
|---|------|------|---------|
| 9 | DCNet/**DDNet70Model** | `DCNet/func/ddnet.py` | `output_size=(256,256)`；`F.pad([-1,-1,-29,-29])` → `F.interpolate(output_size)` |
| 10 | ddnet/**DDNet70Model** | `ddnet/net/DDNet70.py` | `output_size=(256,256)`；动态 `output_lim` |
| 11 | ddnet/**SDNet70Model** | `ddnet/net/DDNet70.py` | `output_size=(256,256)`；动态 `output_lim` |
| 12 | TU-Net/**DDNet70Model** | `TU-Net/model/DDNet70.py` | `output_size=(256,256)`；动态 `output_lim` |
| 13 | TU-Net/**SDNet70Model** | `TU-Net/model/DDNet70.py` | `output_size=(256,256)`；动态 `output_lim` |

> 额外修复：ddnet DDNet70Model 中引用了 `unetUp1/unetUp2/netUp1/netUp2`（4 个不存在的类名，原始代码 bug），已修复为使用 `unetUp/netUp`。

#### D 类：TU_Net（SeismicRecordDownSampling 动态化 + decoder output_lim 动态化）

| # | 模型 | 文件 | 改造方式 |
|---|------|------|---------|
| 14 | TU-Net/**TU_Net** | `TU-Net/model/TU_Net.py` + `TU-Net/model/details.py` | `SeismicRecordDownSampling.forward` 中 `F.interpolate(x, size=[560, 70])` → `F.interpolate(x, size=[width*8, width])`（动态计算）；`SeismicRecordDownSampling2.forward` 中 `F.interpolate(dim_reduce7, size=(70, 70))` → `F.interpolate(dim_reduce7, size=(w, w))`（动态计算）；decoder `UNetUp2` 的 `output_lim` 从 `[18,18],[35,35],[70,70]` → `[oh//4, ow//4],[oh//2, ow//2],[oh, ow]`；末尾追加 `F.interpolate` 保证精确输出 |

#### E 类：FuteFWI（Transformer，AdaptiveAvgPool + 动态 pos_embedding + F.interpolate）

这是最复杂的改造，涉及 ResNet encoder 的 `ConstantPad2d` 去除、Transformer 的位置编码动态化、decoder 的 `ConstantPad2d` 替换。

| # | 模型 | 文件 | 改造方式 |
|---|------|------|---------|
| 15 | FuTE-FWI/**FuteFWI** | `FuTE-FWI/models/FuteFWI.py` | ResNet：移除 `self.pad = ConstantPad2d((0,0,-2,-3),0)` 和 forward 中的 `self.pad(x)` 调用；新增 `self.adaptive_pool = AdaptiveAvgPool2d((grid_h, grid_w))` 将 ResNet 输出统一到固定 grid（默认 20x12）；`self.pos_embedding` 改为 `nn.Parameter(torch.randn(1, grid_h*grid_w, hidden_size))`（动态长度）；`rearrange` 中的 `h=20, w=12` 改为 `h=h, w=w`（运行时从特征图获取）；decoder 移除 `self.pad = ConstantPad2d((-13,-13,-10,-10),0)` 和 `self.pad(x)` 调用，替换为 `F.interpolate(output_size)` |
| 16a | FuTE-FWI/**Ablation_1** | `FuTE-FWI/models/FuteFWI.py` | `patch_embedding` 后新增 `AdaptiveAvgPool2d((grid_h, grid_w))`；`pos_embedding` 和 `rearrange` 同上动态化；decoder 同上 |
| 16b | FuTE-FWI/**Ablation_2** | `FuTE-FWI/models/FuteFWI.py` | ResNet 后新增 `AdaptiveAvgPool2d((grid_h, grid_w))` 替代原来的 `rearrange(x1=3, x2=2)` 硬编码 reshape；decoder 同上 |

#### 无需改造的模型

| 模型 | 文件 | 原因 |
|------|------|------|
| OpenFWI/FuTE-FWI **Discriminator** | `OpenFWI/network.py`、`FuTE-FWI/models/VelocityGAN.py`、`TU-Net/model/InversionNet.py` | `forward` 末尾 `x.view(x.shape[0], -1)` 天然适配任意尺寸 |
| ddnet **DDNetModel** | `ddnet/net/DDNet.py` | 使用 `label_dsp_dim` 参数控制输出，天然灵活 |
| ddnet/DCNet **FCNVMB** | `ddnet/net/FCNVMB.py`、`DCNet/func/comparison_net.py` | 使用 `label_dsp_dim` 参数控制输出 |
| FCNVMB **UnetModel** | `FCNVMB/func/UnetModel.py` | 使用 `label_dsp_dim` 参数控制输出 |

#### 跳过的模型（非 5-shot FWI benchmark 适用）

| 模型 | 文件 | 原因 |
|------|------|------|
| TU-Net **DDNetModel/SDNetModel** | `TU-Net/model/DDNet.py` | 29 通道输入、201×301 输出（SEG 数据集专用） |
| TU-Net **TU_Net_SEG** | `TU-Net/model/TU_Net_SEG.py` | 29 通道输入、201×301 输出（SEG 数据集专用） |

### 各类模型数据流验证

以下使用输入 `(B, 5, 2976, 256)`（经 `align_multiple=32, align_mode=crop` 后）进行验证。

#### A 类：InversionNet（以 OpenFWI InversionNet 为例）

```
编码器：
  convblock1  (stride 2,1) → (B, 32,  1488, 256)
  convblock2  (stride 2,1) → (B, 64,  744,  256)
  convblock3  (stride 2,1) → (B, 64,  372,  256)
  convblock4  (stride 2,1) → (B, 128, 186,  256)
  convblock5  (stride 2)   → (B, 128, 93,   128)
  convblock6  (stride 2)   → (B, 256, 47,   64)
  convblock7  (stride 2)   → (B, 256, 24,   32)
  global_pool              → (B, 256, 1,    1)    ← AdaptiveAvgPool2d((1,1))
  bottleneck               → (B, 512, 1,    1)    ← Conv2d(256, 512, kernel_size=1)

解码器：
  deconv1 (kernel=5)       → (B, 512, 5,    5)
  deconv2 (stride=2)       → (B, 256, 10,   10)
  deconv3 (stride=2)       → (B, 128, 20,   20)
  deconv4 (stride=2)       → (B, 64,  40,   40)
  deconv5 (stride=2)       → (B, 32,  80,   80)
  deconv6 (Tanh)           → (B, 1,   80,   80)
  F.interpolate            → (B, 1,   256,  256)  ← bilinear, align_corners=False
```

#### B 类：DCNet

```
编码器（与 InversionNet 类似的下采样链）：
  convblock1~7             → (B, 256, ?, ?)        ← 输出尺寸取决于输入
  global_pool              → (B, 256, 1, 1)         ← AdaptiveAvgPool2d((1,1))
  bottleneck               → (B, 512, 1, 1)         ← Conv2d 1x1

解码器：
  deconvblock1~6           → (B, 1, ~80, ~80)
  F.interpolate            → (B, 1, 256, 256)       ← 替代原来的 F.pad([-5,-5,-5,-5])
```

#### C 类：DDNet70Model（以 ddnet/net/DDNet70.py 为例）

```
编码器（4层 Maxpool 下采样）：
  enc1     (64 ch)         → (B, 64,  2976, 256)
  pool1                    → (B, 64,  1488, 128)
  enc2     (128 ch)        → (B, 128, 1488, 128)
  pool2                    → (B, 128, 744,  64)
  enc3     (256 ch)        → (B, 256, 744,  64)
  pool3                    → (B, 256, 372,  32)
  enc4     (512 ch)        → (B, 512, 372,  32)
  pool4                    → (B, 512, 186,  16)
  center   (1024 ch)       → (B, 1024, 186, 16)

解码器（使用动态 output_lim 控制精确尺寸）：
  unetUp4 (lim=[128,128]) → (B, 512, 128, 128)    ← oh//2=128, ow//2=128
  unetUp3 (lim=[64,64])   → (B, 256, 64,  64)     ← oh//4=64,  ow//4=64
  unetUp2 (lim=[32,32])   → (B, 128, 32,  32)     ← oh//8=32,  ow//8=32
  ... 最终 F.interpolate   → (B, 1,  256, 256)
```

> `output_lim` 的动态计算方式：
> ```python
> oh, ow = output_size  # (256, 256)
> lim8 = [oh // 8, ow // 8]   # [32, 32]
> lim4 = [oh // 4, ow // 4]   # [64, 64]
> lim2 = [oh // 2, ow // 2]   # [128, 128]
> lim1 = [oh, ow]              # [256, 256]
> ```

#### D 类：TU_Net

```
编码器：
  SeismicRecordDownSampling → F.interpolate(size=[width*8, width])  ← 动态计算
  SeismicRecordDownSampling2 → F.interpolate(size=(w, w))           ← 动态计算
  UNet encoder (pool1~4)    → 逐层 1/2 下采样

解码器：
  UNetUp2 (lim=[64,64])   → (B, C, oh//4, ow//4)
  UNetUp2 (lim=[128,128]) → (B, C, oh//2, ow//2)
  UNetUp2 (lim=[256,256]) → (B, C, oh, ow)
  final_conv               → (B, 1, ~256, ~256)
  F.interpolate            → (B, 1, 256, 256)       ← 精确保证输出尺寸
```

#### E 类：FuteFWI（Transformer）

```
编码器：
  ResNet50 backbone        → (B, 2048, ?, ?)        ← 输出尺寸取决于输入
  adaptive_pool            → (B, 2048, 20, 12)       ← AdaptiveAvgPool2d((grid_h, grid_w))
  1x1 conv (patch_embed)   → (B, hidden_size, 20, 12)
  rearrange + pos_embedding→ (B, 240, hidden_size)   ← 240 = 20*12

Transformer：
  TransformerEncoder ×N    → (B, 240, hidden_size)

解码器：
  rearrange                → (B, hidden_size, 20, 12) ← h,w 从特征图动态获取
  upconv1~4 (stride=2)     → (B, C, 320, 192)        ← 逐层2x上采样
  F.interpolate            → (B, 1, 256, 256)         ← 替代原来的 ConstantPad2d 裁剪
```

> **FuteFWI 新增构造参数**：
> - `output_size=(256, 256)` — 控制最终输出尺寸
> - `grid_h=20, grid_w=12` — 控制 Transformer 输入 grid 大小（影响 token 数量）

---

## 2. 统一评测：所有仓库共用 evaluator

### 2.1 统一评测输入格式

每个模型测试结束后，导出两个文件：

- `pred.npy`：预测速度模型，形状 `(N,H,W)` 或 `(N,1,H,W)`
- `gt.npy`：真值速度模型，形状同上

### 2.2 执行统一评测

```bash
python benchmark/benchmark_eval.py \
  --pred /path/to/pred.npy \
  --gt /path/to/gt.npy \
  --out-json /path/to/eval.json
```

### 2.3 统一指标

输出并解析以下指标：

- `MSE`（越小越好）
- `MAE`（越小越好）
- `PSNR`（越大越好）
- `SSIM`（越大越好）
- `L1-Grad`（越小越好）
- `LPIPS`（越小越好）

已支持两种 LPIPS 模式：

- **real**（默认，论文协议）：使用 `lpips` 库 + AlexNet 特征，需下载 torchvision 预训练权重
- **proxy**：不依赖预训练权重的代理指标，仅作网络受限环境下的备用，结果与论文 LPIPS 不可比

**当前 benchmark 默认使用 `--lpips-mode real`（AlexNet），与论文报告的 LPIPS 一致**。real 模式示例：

```bash
python benchmark/benchmark_eval.py \
  --pred /path/to/pred.npy \
  --gt /path/to/gt.npy \
  --lpips-mode real \
  --lpips-backbone alex \
  --device cuda
```

为了公平，所有模型必须使用相同的 LPIPS 配置（mode/backbone/device）。

---

## 3. 统一训练预算

### 3.1 两种预算模式（任选其一）

- `epoch`：每个模型固定 epoch（更接近传统论文设置）
- `wallclock`：每个模型固定训练时长（更公平地比较效率）

### 3.2 运行时环境变量

```bash
export BUDGET_MODE=wallclock
export WALLCLOCK_HOURS=6
```

总控脚本会把它们下发给各任务：

- `BENCHMARK_BUDGET_MODE`
- `BENCHMARK_WALLCLOCK_HOURS`

### 3.3 仓库训练脚本如何配合

各仓库训练入口建议读取以上环境变量并执行：

- `epoch` 模式：按预设 epoch 正常结束
- `wallclock` 模式：到达时长就保存 checkpoint 并结束

这样 `Train(h)` 才能跨模型严格可比。

---

## 3.4 训练轮数要不要统一（结论）

**要统一，但需要和预算口径一致。**

- 如果你选择 `epoch` 预算：  
  所有模型必须用同样的 `max_epoch`，并且早停策略一致（建议都关闭早停，或都使用同一 patience）
- 如果你选择 `wallclock` 预算：  
  不再强制相同 epoch，而是强制相同训练时长（例如每模型 6 小时）

建议你不要同时混用两套口径。  
论文主表建议采用一套主协议，另一套作为补充实验（附录）。

---

## 3.5 公平性总清单（必须统一 / 必须报告）

> **详细说明**：参见 `benchmark/BENCHMARK_FAIRNESS.md`

### A. 必须统一（Hard Constraints）

1. **数据切分**：同一份 `train/val/test` 索引（`split_by_inline.npy`）
2. **输入输出尺寸**：同一尺寸策略（固定裁剪/补零/重采样）
3. **评测脚本**：统一走 `benchmark_eval.py`
4. **预算口径**：统一为 `epoch` 或统一为 `wallclock`
5. **随机种子集合**：所有模型使用同一组 seeds
6. **硬件与精度**：同型号 GPU、**相同 GPU 数量**、相同 AMP 策略、相同 batch 定义方式

**GPU 数量公平**：InversionNet 与 VelocityGAN 均支持 8 卡 DDP，使用 `torchrun --nproc_per_node=8` 启动。VelocityGAN **不使用** `--sync-bn`（WGAN-GP 与 SyncBatchNorm 不兼容）。详见 `benchmark/FAIRNESS_CHECK.md`、`benchmark/CHANGELOG_BENCHMARK.md`。

### B. 必须报告（Reporting Constraints）

1. 模型超参（lr、优化器、weight decay、scheduler）
2. 预算细节（epoch 数或 wall-clock 小时）
3. 运行环境（CUDA、PyTorch、驱动、GPU 型号）
4. 失败样本和重跑策略（是否重启、是否剔除异常 run）
5. 论文主结果使用固定 seed 42；`aggregate_seeds.py` 仅用于可选的多 seed 汇总

---

## 3.6 我建议的“最公平且可落地”协议

### 主协议（推荐）

- 预算：`wallclock`（例如 6h / model / seed）
- 评测：统一 evaluator，同一测试集，一次性评估
- 论文主结果：固定 seed 42

优点：对不同收敛速度的模型更公平，特别适合比较 CNN / GAN / Transformer。

### 补充协议（可选）

- 预算：`epoch`（例如 120 epoch）
- 其余设置不变

作用：和既有论文常见设置对齐，便于横向对照。

---

## 3.7 还需要额外注意的公平性细节（容易忽略）

1. **归一化口径统一**：  
   指标必须在同一域计算（建议都在物理域）
2. **数据增强一致性**：  
   若某模型开启增强，其它模型也应开启同等强度增强，或全部关闭
3. **checkpoint 选择规则一致**：  
   都按“验证集最优”或都按“最后 epoch”，不能混用
4. **推理速度测试协议一致**：  
   相同输入尺寸、相同 batch size、固定 warmup + 固定迭代次数
5. **FLOPs 统计口径一致**：  
   相同输入形状、相同工具（同版本 `thop`）
6. **异常 run 处理规则预先定义**：  
   如 NaN/崩溃是否重跑，最多重跑次数必须提前写清

---

## 3.8 可复现性：随机种子机制

为保证单 seed 实验的可复现性，所有训练脚本均已内置统一的 seed 设置函数 `setup_benchmark_seed()`：

```python
def setup_benchmark_seed(args):
    import random
    seed = getattr(args, 'seed', None)
    if seed is None:
        seed = int(os.environ.get('BENCHMARK_SEED', '42'))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'[Benchmark] Seed set to {seed}')
```

#### seed 优先级

1. **命令行参数 `--seed <值>`**（最高优先级）
2. **环境变量 `BENCHMARK_SEED`**（次优先级，`run_benchmark_suite.sh` 自动设置）
3. **默认值 `42`**（兜底）

#### 涉及的训练脚本

| 脚本 | 已添加 `setup_benchmark_seed()` | 已添加 `--seed` CLI 参数 |
|------|:---:|:---:|
| `OpenFWI/train.py` | Y | Y |
| `FuTE-FWI/train_velocitygan.py` | Y | Y |
| `FuTE-FWI/train_inversionnet.py` | Y | Y |
| `FuTE-FWI/train_futefwi.py` | Y | Y |

#### `run_benchmark_suite.sh` 的 seed 传递机制

总控脚本使用 `__SEED__` 字符串占位符（而非 Shell 变量 `${BENCHMARK_SEED}`），
在 seed 循环内通过 bash 字符串替换在运行时展开：

```bash
for seed in "${seed_list[@]}"; do
    export BENCHMARK_SEED="${seed}"
    seed_train_cmd="${train_cmd//__SEED__/${seed}}"
    seed_eval_cmd="${eval_cmd//__SEED__/${seed}}"
    # 执行 seed_train_cmd 和 seed_eval_cmd
done
```

> **为什么不用 `${BENCHMARK_SEED}`？**  
> Shell 数组在定义时就会展开所有变量。如果在 `TASKS=()` 定义中写 `seed${BENCHMARK_SEED}`，
> 该变量在数组创建时就被展开（此时 seed 循环尚未开始，变量为空），导致所有 seed 共用同一个（空的）路径。  
> 使用字面字符串 `__SEED__` 作为占位符，推迟到循环内才替换，避免了这个问题。

#### 验证 seed 是否生效

训练开始时会打印 `[Benchmark] Seed set to <值>`。检查训练日志确认：

```
[Benchmark] Seed set to 42
```

---

## 3.9 训练过程保存与可视化（已统一实现）

为减少重复实验和提升问题定位效率，统一要求：

1. **每 5 轮保存一次 checkpoint**
2. **每 5 轮导出一次预测速度图与 GT 对比图**

**当前实现**：

- **OpenFWI**：`--vis-interval 5`，`-eb 5`（每 5 轮保存）
- **FuTE-FWI**：`--save-interval 5 --vis-interval 5`
- **统一脚本**（DCNet/DDNet70/TU_Net/ABA_FWI/ConvNeXtFWI/ConvNeXtKaggle/FCNVMB）：`--save-interval 5 --vis-interval 5`，输出：
  - `checkpoint_5.pth`、`checkpoint_10.pth` 等（每 5 轮）
  - `checkpoint.pth`（最后一份，供 eval 加载）
  - `visualizations/epoch_0005.png`、`epoch_0010.png` 等（pred vs GT 对比图）

可视化文件默认输出在各自输出目录下的 `visualizations/`。

---

## 4. 单 seed 运行与统计

### 4.1 设置 seed 列表

```bash
export SEEDS=42
```

推荐使用 seed 42（与默认配置一致）。所有模型必须使用完全相同的 seed 列表。

### 4.2 seed 执行流程

`run_benchmark_suite.sh` 的执行顺序：

```
对每个 TASK:
  1. [一次] profile (模型参数量/FLOPs/推理速度)
  2. [一次] split (可选)
  3. 对每个 seed:
     3a. 替换 train_cmd / eval_cmd 中的 __SEED__ 占位符为实际 seed 值
     3b. export BENCHMARK_SEED=<seed>
     3c. 执行训练 (计时 → Train(h))
     3d. 执行评测 (解析 6 项指标)
     3e. 写入 CSV 一行
```

### 4.3 seed 如何传入训练脚本

有两个互补渠道，任一生效即可：

1. **命令行参数**：`--seed __SEED__`  
   在 TASK 定义中写 `--seed __SEED__`，运行时 `__SEED__` 会被替换为实际 seed 值。  
   训练脚本通过 `args.seed` 读取。

2. **环境变量**：`BENCHMARK_SEED`  
   由 `run_benchmark_suite.sh` 在 seed 循环内 `export BENCHMARK_SEED=<seed>`。  
   训练脚本中 `setup_benchmark_seed()` 会读取此环境变量作为兜底。

### 4.4 总控输出

`run_benchmark_suite.sh` 产出：

- `benchmark_logs/<model_name>_seed<seed>.log`（每个 model+seed 的完整训练/评测日志）
- `benchmark_logs/benchmark_metrics.csv`（逐 seed 原始行，格式见下）
- `benchmark_logs/benchmark_metrics_agg.csv`（可选多 seed 聚合；论文主结果使用 seed 42）
- `benchmark_logs/summary.log`（执行摘要和时间戳）

CSV 格式示例：

```csv
Model,Model Category,Params(M),FLOPs(G),Train(h),Infer(ms),MSE,MAE,PSNR,SSIM,L1-Grad,LPIPS,Status
InversionNet-smoke#seed42,CNN,12.5,45.3,0.0123,15.2,0.0015,0.023,35.2,0.95,0.012,0.05,OK
InversionNet-smoke#seed42,CNN,12.5,45.3,0.0130,15.2,0.0018,0.025,34.8,0.94,0.013,0.06,OK
```

聚合由 `benchmark/aggregate_seeds.py` 自动完成。

---

## 5. 总控脚本字段说明

### 5.1 TASK 格式

`run_benchmark_suite.sh` 的 `TASKS` 数组中，每个任务是一个 pipe (`|`) 分隔的字符串，共 **8 个字段**：

```
model_name|model_category|repo_dir|gpu_id|profile_cmd|split_cmd|train_cmd|eval_cmd
```

| 字段 | 说明 | 示例 |
|------|------|------|
| `model_name` | 唯一标识，用于日志和 CSV | `InversionNet-smoke` |
| `model_category` | 模型类别（CNN/GAN/Transformer） | `CNN` |
| `repo_dir` | 仓库绝对路径（`cd` 到此目录后执行命令） | `${WORK_ROOT}/OpenFWI` |
| `gpu_id` | 使用的 GPU ID | `0` |
| `profile_cmd` | Profiling 命令，输出 `METRICS params_m=... flops_g=... infer_ms=...` | `python profile_model_metrics.py ...` |
| `split_cmd` | 可选，数据切分命令（留空则跳过） | 通常为空 |
| `train_cmd` | 训练命令（计入 `Train(h)` 耗时） | `python train.py ...` |
| `eval_cmd` | 评测命令（解析 6 项统一指标） | InversionNet/VelocityGAN/FuteFWI 用 `unified_benchmark_test.py`；其余用 `unified_benchmark_test.py` |

### 5.2 `__SEED__` 占位符

在 `train_cmd` 和 `eval_cmd` 中，使用字面字符串 `__SEED__` 表示 seed 值占位。
运行时会被替换为实际的 seed 数字。

常见用法：

```bash
# 训练命令中传入 seed 并区分输出目录
"... --seed __SEED__ --save_dir ${LOG_DIR}/InversionNet_seed__SEED__"

# 评测命令中指向对应 seed 的 checkpoint
"... --checkpoint_path ${LOG_DIR}/InversionNet_seed__SEED__/checkpoint.pth"
```

> **不要**在 TASKS 定义中使用 `${BENCHMARK_SEED}`，因为它会在数组创建时就被 Shell 展开为空字符串。

---

## 6. 各仓库接入状态与改动说明

> **状态总结**：所有 5 个仓库、16 个模型已全部完成分辨率无关架构改造并通过验证。

### 6.1 OpenFWI（已完成 ✓）

**改造模型**：`InversionNet`、`FCN4_Deep_Resize_2`（UPFWI）

| 文件 | 改动 |
|------|------|
| `network.py` | `InversionNet` / `FCN4_Deep_Resize_2`：`convblock8` → `global_pool(AdaptiveAvgPool2d) + bottleneck(Conv 1x1)`；移除 `F.pad`，添加 `F.interpolate(output_size)`；新增 `output_size` 构造参数；移除 `from math import ceil` |
| `train.py` | 添加 `setup_benchmark_seed()`；模型创建传入 `output_size`；新增 `--seed`/`--output-height`/`--output-width` CLI 参数 |
| 评测 | **OpenFWI 无 test.py**，benchmark 使用 `unified_benchmark_test.py`；模型创建传入 `output_size`；LPIPS 固定 `proxy` 模式 |

**训练入口完整参数：**

```bash
python train.py \
  --use-unified-loader \
  --data-root <DATA_ROOT> \
  --global-map-csv <GLOBAL_MAP> \
  --stats-json <STATS_JSON> \
  --align-multiple 32 --align-mode crop \
  --seed <SEED> \
  --output-height 256 --output-width 256 \
  -m InversionNet -b <batch_size> -eb <epoch_block> -nb <num_block> \
  --vis-interval 5
```

### 6.2 FuTE-FWI（已完成 ✓）

**改造模型**：`Generator (VelocityGAN)`、`InversionNet`、`FuteFWI`、`Ablation_1`、`Ablation_2`

| 文件 | 改动 |
|------|------|
| `models/VelocityGAN.py` | `Generator`：`convblock8` → `global_pool + bottleneck`；移除 `F.pad`，添加 `F.interpolate(output_size)`；新增 `output_size` 参数。**Discriminator**：默认 `norm="in"`（InstanceNorm），与 WGAN-GP 的 `create_graph=True` 兼容；ConvBlock/DeconvBlock 使用 `inplace=False` |
| `models/InversionNet.py` | `InversionNet`：同上 |
| `models/FuteFWI.py` | **ResNet**：移除 `self.pad = ConstantPad2d((0,0,-2,-3),0)` 及 forward 中的 `self.pad(x)` |
| | **FuteFWI**：新增 `output_size`, `grid_h`, `grid_w` 参数；添加 `self.adaptive_pool = AdaptiveAvgPool2d((grid_h, grid_w))` 在 ResNet 输出后；`pos_embedding` 改为 `nn.Parameter(torch.randn(1, grid_h*grid_w, hidden_size))`（动态长度）；`rearrange` 中 `h=20, w=12` 改为运行时动态获取 `h, w = x.shape[2], x.shape[3]`；decoder 移除 `ConstantPad2d` 裁剪，替换为 `F.interpolate(output_size)` |
| | **Ablation_1**：新增 `output_size`, `grid_h`, `grid_w`；`patch_embedding` 后添加 `AdaptiveAvgPool2d((grid_h, grid_w))`；`pos_embedding` 和 `rearrange` 同上动态化；decoder 移除 `ConstantPad2d`，替换为 `F.interpolate(output_size)` |
| | **Ablation_2**：新增 `output_size`, `grid_h`, `grid_w`；ResNet 后添加 `AdaptiveAvgPool2d((grid_h, grid_w))`；decoder 同上 |
| `train_velocitygan.py` | 添加 `setup_benchmark_seed()`；模型创建传入 `output_size` |
| `train_inversionnet.py` | 添加 `setup_benchmark_seed()`；模型创建传入 `output_size` |
| `train_futefwi.py` | 添加 `setup_benchmark_seed()`；模型创建传入 `output_size`, `grid_h`, `grid_w`；**DDP 支持**：torchrun 启动时用 DistributedDataParallel + DistributedSampler（见 CHANGELOG 第十八节） |
| 评测 | **FuTE-FWI test.py 无 benchmark 参数**，benchmark 使用 `unified_benchmark_test.py`；LPIPS 固定 `proxy` 模式 |
| `utils/argparser.py` | 所有 parser 新增 `--seed`/`--output-height`/`--output-width` 参数 |
| `utils/loss.py` | `Wasserstein_GP` 新增 `model_for_gp` 参数，DDP 下用 `model.module` 计算 gradient penalty |
| `utils/utils.py` | `train_gan` 传入 `model_for_gp=getattr(model_d, "module", model_d)` |

> **VelocityGAN 8 卡 DDP 说明**：不使用 `--sync-bn`（WGAN-GP 与 SyncBatchNorm 不兼容）。Discriminator 已改用 InstanceNorm。Docker 需加 `--shm-size=32g`。

> **FuteFWI 特别说明**：
> - 需要安装 `einops`（`pip install einops`），否则 import 失败
> - **8 卡 DDP**：使用 `torchrun --nproc_per_node=8 train_futefwi.py ...`，与 VelocityGAN 等一致；已从 DataParallel 改为 DistributedDataParallel
> - 构造函数新增 `grid_h=20, grid_w=12` 参数控制 Transformer 的 token grid 大小
> - 默认 `grid_h=20, grid_w=12` 对应 240 个 token，与原始论文设置一致
> - 若需要调整 grid 分辨率，可传入不同 `grid_h/grid_w`（token 数 = `grid_h × grid_w`，会影响计算量和显存）

### 6.3 DCNet（已完成 ✓）

**改造模型**：`DCNet (DCModel)`、`DDNet70Model`、`InversionNet`

| 文件 | 改动 |
|------|------|
| `DCNet.py` | `DCNet`：`lastBlock`（含硬编码 `kernel_size=(8, ceil(70*sample_spatial/8))`）→ `global_pool(AdaptiveAvgPool2d) + bottleneck(Conv 1x1)`；`F.pad([-5,-5,-5,-5])` → `F.interpolate(output_size)`；`DCModel` 新增 `output_size` 参数并透传给 `DCNet`；移除 `from math import ceil` |
| `func/ddnet.py` | `DDNet70Model`：新增 `output_size` 参数；两处 `F.pad(dc1_up2, [-1,-1,-29,-29])` → `F.interpolate(dc1_up2, size=self.output_size)`（分别对应 dc1 和 dc2 两个解码分支） |
| `func/comparison_net.py` | `InversionNet`：同 OpenFWI InversionNet 的改造方式（`convblock8` → `global_pool + bottleneck`；`F.pad` → `F.interpolate`；新增 `output_size` 参数）；移除 `from math import ceil` |

> **DCModel.forward 特殊说明**：原始代码中 `DCModel.forward(self, x, label_dsp_dim)` 接受一个 `label_dsp_dim` 参数但内部并未使用（由 `DCNet` 内部硬编码处理）。改造后 `output_size` 通过构造函数传入，`label_dsp_dim` 仍需保留以保持 API 兼容，调用时传 `None` 即可：`model(x, label_dsp_dim=None)`。

### 6.4 ddnet（已完成 ✓）

**改造模型**：`InversionNet`、`DDNet70Model`、`SDNet70Model`

| 文件 | 改动 |
|------|------|
| `net/InversionNet.py` | 同 OpenFWI InversionNet 改造；新增 `import torch.nn.functional as F`（原文件缺少）；移除 `from math import ceil`；新增 `output_size` 参数 |
| `net/DDNet70.py` | **DDNet70Model**：新增 `output_size` 参数；基于 `output_size` 动态计算 `lim8/lim4/lim2/lim1` 替代原来的硬编码 `[9,9],[18,18],[35,35],[70,70]`；**修复原始代码 bug**：将不存在的 `unetUp1/unetUp2/netUp1/netUp2` 替换为实际存在的 `unetUp/netUp` 类 |
| `net/DDNet70.py` | **SDNet70Model**：同上改造方式；同样修复了 `unetUp1/unetUp2` bug |

> **原始 bug 修复说明**：`ddnet/net/DDNet70.py` 中 `DDNet70Model.__init__` 引用了 `unetUp1`、`unetUp2`、`netUp1`、`netUp2` 四个类，但这些类在文件中从未定义（只定义了 `unetUp` 和 `netUp`）。这是原始仓库的代码 bug，会导致 `NameError`。已修正为使用 `unetUp`/`netUp`。

### 6.5 TU-Net（已完成 ✓）

**改造模型**：`InversionNet`、`DDNet70Model`、`SDNet70Model`、`TU_Net`

| 文件 | 改动 |
|------|------|
| `model/InversionNet.py` | 同 OpenFWI InversionNet 改造；新增 `output_size` 参数；移除 `from math import ceil` |
| `model/DDNet70.py` | **DDNet70Model** / **SDNet70Model**：同 `ddnet/net/DDNet70.py` 改造方式（动态 `output_lim`）；新增 `output_size` 参数 |
| `model/details.py` | **SeismicRecordDownSampling**：`F.interpolate(x, size=[560, 70])` → `F.interpolate(x, size=[width*8, width])`（`width=x.shape[3]`）；**SeismicRecordDownSampling2**：`F.interpolate(dim_reduce7, size=(70, 70))` → `F.interpolate(dim_reduce7, size=(w, w))`（`w=x.shape[3]`） |
| `model/TU_Net.py` | **TU_Net**：新增 `output_size` 参数；decoder 中 `UNetUp2` 的硬编码 `output_lim` → 基于 `output_size` 动态计算 `[oh//4, ow//4]`, `[oh//2, ow//2]`, `[oh, ow]`；`forward` 末尾添加 `F.interpolate(output_size)` 保证精确尺寸；新增 `import torch.nn.functional as F` |

> **跳过的 SEG 模型**：`TU-Net/model/DDNet.py`（DDNetModel/SDNetModel）和 `TU-Net/model/TU_Net_SEG.py` 使用 29 通道输入、201×301 输出，属于 SEG 数据集专用模型，不适用于当前 5-shot FWI benchmark，未进行改造。

---

## 7. 结果表头对齐（你论文表格可直接对应）

`benchmark_metrics.csv` 与 `benchmark_metrics_agg.csv` 会输出以下列（正常完成时均有值）：

| 列名 | 来源 | 说明 |
|------|------|------|
| `Model` | 脚本 | 模型名称 |
| `Model Category` | 脚本 | 类别（CNN/GAN/Transformer 等） |
| `Params(M)` | profile 输出 `METRICS params_m=...` | 参数量（百万） |
| `FLOPs(G)` | profile 输出 `METRICS flops_g=...` | 浮点运算量（十亿） |
| `Train(h)` | 训练耗时 | 秒转小时 |
| `Infer(ms)` | profile 输出 `METRICS infer_ms=...` | 单样本推理耗时（毫秒） |
| `MSE` | `benchmark_eval.py` 打印 | 均方误差 |
| `MAE` | `benchmark_eval.py` 打印 | 平均绝对误差 |
| `PSNR` | `benchmark_eval.py` 打印 | 峰值信噪比 |
| `SSIM` | `benchmark_eval.py` 打印 | 结构相似度 |
| `L1-Grad` | `benchmark_eval.py` 打印 | 梯度 L1 差异 |
| `LPIPS` | `benchmark_eval.py` 打印 | 感知损失 |
| `Status` | 脚本 | OK / FAIL(train) / FAIL(eval) |

**前提**：所有 eval 必须走统一 `benchmark_eval.py`（通过 `--benchmark-eval` 传入），否则 MSE/MAE/PSNR/SSIM/L1-Grad/LPIPS 会解析为 NA。

---

## 8. 推荐执行顺序（实操）

1. 先生成统一 split  
2. 先接通 1 个模型（建议 OpenFWI InversionNet）  
3. 验证 CSV 字段完整  
4. 再并行补齐其他模型  
5. 论文主实验使用固定 seed 42；如需稳定性分析，可额外运行其他 seed 并用 `aggregate_seeds.py` 生成聚合表

---

## 9. 常见问题与排查

- `Train(h)=NA`：`train_cmd` 为空，或训练阶段未执行
- `PSNR/L1-Grad=NA`：`eval_cmd` 未打印对应字段，或没有走统一 evaluator
- `FLOPs=NA`：环境未安装 `thop`（可选，不影响其余流程）；DDNet70/TU_Net 若 params/infer 也为 NA，检查 `profile_model_metrics.py` 是否已按「先 infer_ms 再 try_flops_g」顺序执行（见 CHANGELOG 第十七节）；ABA_FWI/FCNVMB 若 params/infer 为 NA，检查是否已移除 `ABA_Loss` 导入（见 CHANGELOG 第十九节）
- 指标对不上：检查是否在同一尺度域（归一化域 vs 物理域）计算

---

## 10. 完整命令参考（可直接复制运行）

`OpenFWI` 与 `FuTE-FWI` 的训练/测试入口已全面支持 `--use-unified-loader`、`--seed`、`--output-height`/`--output-width`。

### 10.1 OpenFWI InversionNet 训练

```bash
cd OpenFWI

python train.py \
  --use-unified-loader \
  --data-root <DATA_ROOT> \
  --global-map-csv ../benchmark/generated_split/global_map.csv \
  --stats-json ../benchmark/generated_split/train_stats.json \
  --time-downsample 1 \
  --channel-mode all \
  --align-multiple 32 \
  --align-mode crop \
  --seed 42 \
  --output-height 256 --output-width 256 \
  -m InversionNet \
  -b 8 \
  --lr 1e-4 \
  -eb 5 -nb 20 \
  --vis-interval 5 \
  -o benchmark_out -n inversionnet -s seed42
```

参数说明：
- `--seed 42`：设置随机种子，保证可复现
- `--output-height 256 --output-width 256`：模型输出分辨率（默认 256，与数据集 GT 一致）
- `--align-multiple 32 --align-mode crop`：输入时间维裁剪到 32 的倍数（3000 → 2976）
- `-eb 5 -nb 20`：每 5 epoch 保存一次 checkpoint，共跑 100 epoch

### 10.2 OpenFWI InversionNet 测试 + 评测

**说明**：OpenFWI 仓库无 `test.py`，benchmark 使用 `unified_benchmark_test.py` 评测。手动评测示例：

```bash
cd /path/to/repo

python benchmark/unified_benchmark_test.py \
  --model InversionNet \
  --checkpoint benchmark_logs/inversionnet_seed42/checkpoint.pth \
  --data-root <DATA_ROOT> \
  --global-map-csv benchmark/generated_split/global_map.csv \
  --stats-json benchmark/generated_split/train_stats.json \
  --align-multiple 32 --align-mode crop \
  --batch-size 8 \
  --export-dir benchmark_logs/inversionnet_seed42/eval \
  --benchmark-eval benchmark/benchmark_eval.py \
  --benchmark-eval-json benchmark_logs/inversionnet_seed42/eval/metrics.json \
  --benchmark-eval-lpips-mode real
```

参数说明：
- `--checkpoint`：OpenFWI 训练输出的 checkpoint.pth（含 `model` 键）
- `--export-dir`：导出 `pred.npy` 和 `gt.npy` 的目录
- `--benchmark-eval-lpips-mode real`：使用 proxy 模式（不依赖预训练权重）

### 10.3 FuTE-FWI VelocityGAN 训练

```bash
cd FuTE-FWI

python train_velocitygan.py \
  --use-unified-loader \
  --data-root <DATA_ROOT> \
  --global-map-csv ../benchmark/generated_split/global_map.csv \
  --stats-json ../benchmark/generated_split/train_stats.json \
  --align-multiple 32 \
  --align-mode crop \
  --seed 42 \
  --output-height 256 --output-width 256 \
  --batch-size 8 \
  --epochs 100 \
  --save-interval 5 --vis-interval 5 \
  --output ../benchmark_logs/velocitygan_seed42 \
  --name VelocityGAN
```

**8 卡 DDP**：使用 `torchrun --nproc_per_node=8 train_velocitygan.py ...`。Discriminator 已改用 InstanceNorm（`norm="in"`），与 WGAN-GP 的 gradient penalty（`create_graph=True`）兼容，避免 BatchNorm 的 inplace 错误。Docker 需加 `--shm-size=32g`。

### 10.4 FuTE-FWI VelocityGAN 测试 + 评测

**说明**：FuTE-FWI 的 `test.py` 无 `--use-unified-loader`、`--export-dir`、`--benchmark-eval` 等 benchmark 参数，benchmark 使用 `unified_benchmark_test.py`。手动评测示例：

```bash
cd /path/to/repo

python benchmark/unified_benchmark_test.py \
  --model VelocityGAN \
  --checkpoint benchmark_logs/velocitygan_seed42/VelocityGAN_FlatVel_A_D.pt \
  --data-root <DATA_ROOT> \
  --global-map-csv benchmark/generated_split/global_map.csv \
  --stats-json benchmark/generated_split/train_stats.json \
  --align-multiple 32 --align-mode crop \
  --batch-size 8 \
  --export-dir benchmark_logs/velocitygan_seed42/eval_results \
  --benchmark-eval benchmark/benchmark_eval.py \
  --benchmark-eval-json benchmark_logs/velocitygan_seed42/eval_results/metrics.json \
  --benchmark-eval-lpips-mode real
```

> VelocityGAN 训练保存为 `{name}_{dataset}_D.pt`（如 `VelocityGAN_FlatVel_A_D.pt`），为纯 state_dict 格式。

### 10.5 FuTE-FWI InversionNet 训练

```bash
cd FuTE-FWI

python train_inversionnet.py \
  --use-unified-loader \
  --data-root <DATA_ROOT> \
  --global-map-csv ../benchmark/generated_split/global_map.csv \
  --stats-json ../benchmark/generated_split/train_stats.json \
  --align-multiple 32 \
  --align-mode crop \
  --seed 42 \
  --output-height 256 --output-width 256 \
  --batch-size 8 \
  --epochs 100 \
  --save-interval 5 --vis-interval 5 \
  --output ../benchmark_logs/fute_invnet_seed42 \
  --name InversionNet
```

### 10.6 一键全量运行（推荐）

**推荐**（所有模型 8 卡 DDP，Docker 内运行）：

```bash
# 推荐：使用项目自建镜像 fwi-benchmark:cu118（docker build -t fwi-benchmark:cu118 -f benchmark/docker/Dockerfile .）
docker run --gpus all --rm --ipc=host \
  -v /path/to/repo:/workspace/repo \
  -v <DATA_ROOT>:<DATA_ROOT> \
  --shm-size=32g -e SEEDS=42 \
  fwi-benchmark:cu118 bash /workspace/repo/run_benchmark_docker.sh
```

使用项目推荐镜像 `fwi-benchmark:cu118`：

```bash
docker run ... fwi-benchmark:cu118 bash /workspace/repo/run_benchmark_docker.sh
```

或容器内直接执行：

```bash
export WORK_ROOT=/workspace/repo
export DATA_ROOT=<DATA_ROOT>
export SEEDS=42
bash run_benchmark_suite.sh
```

脚本会自动：
1. 检测并生成 split + train_stats（如果不存在）
2. 对每个模型做 profiling
3. 对每个 seed 执行训练 + 评测
4. 汇总到 `benchmark_logs/benchmark_metrics.csv`
5. 聚合为 `benchmark_logs/benchmark_metrics_agg.csv`

### 10.7 九大唯一架构与统一脚本

`run_benchmark_suite.sh` 已接入 **10 个唯一架构**：

| 模型 | 类别 | 仓库 | 训练/测试方式 |
|------|------|------|---------------|
| InversionNet | CNN | OpenFWI | 原生 train.py；评测用 unified_benchmark_test（OpenFWI 无 test.py） |
| VelocityGAN | GAN | FuTE-FWI | 原生 train_velocitygan.py；评测用 unified_benchmark_test（无 --sync-bn） |
| FuteFWI | Transformer | FuTE-FWI | 原生 train_futefwi.py（8 卡 DDP，`--sync-bn`）；评测用 unified_benchmark_test |
| DCNet | CNN | DCNet | 统一 unified_benchmark_train/test |
| DDNet70 | UNet | ddnet | 统一 unified_benchmark_train/test |
| TU_Net | UNet | TU-Net | 统一 unified_benchmark_train/test |
| ABA_FWI | ABA | ABA-FWI | 统一 unified_benchmark_train/test |
| ConvNeXtFWI | CNN | ConvNeXt-FWI | 统一 unified_benchmark_train/test |
| ConvNeXtKaggle | CNN | ConvNeXt-Kaggle | 统一 unified_benchmark_train/test；另见 `run_convnext_kaggle_dual.sh` 双模式（Fair+NonFair） |
| FCNVMB | FCN | ABA-FWI (FCNVMB_FWI) | 统一 unified_benchmark_train/test |

**统一脚本**（`benchmark/unified_benchmark_train.py`、`unified_benchmark_test.py`）：
- 使用 `UnifiedFWIDataset` + MSE 损失，公平对比
- 支持 DDP（`torchrun --nproc_per_node=N`）
- 自动映射模型到对应仓库并加载
- **每 5 轮保存 checkpoint**（`--save-interval 5`），输出 `checkpoint_5.pth`、`checkpoint_10.pth` 等，同时覆盖 `checkpoint.pth` 供 eval 使用
- **每 5 轮导出 pred vs GT 对比图**（`--vis-interval 5`），输出到 `visualizations/epoch_0005.png` 等
- 默认 `EPOCHS=100`，可通过 `export EPOCHS=1` 做冒烟测试

**Profile 支持**（`profile_model_metrics.py`）：
- `--repo` 现支持：`openfwi|futefwi|dcnet|ddnet|tu-net|aba-fwi|convnext-fwi|convnext-kaggle`
- `_ensure_repo_path(repo)` 自动将各仓库路径加入 `sys.path`，避免 `import network`/`from models import` 失败
- `run_stage` 会 `cd` 到 `repo_dir` 后执行 profile 命令
- **执行顺序**：先 `infer_ms` 再 `try_flops_g`，避免 thop hooks 污染模型导致 DDNet70/TU_Net 报 `ReLU total_ops` 错误

**注意**：若系统 `python` 指向 2.7，请设置 `PYTHON_BIN=python3` 再运行 `run_benchmark_suite.sh`。

---

## 11. 上线前核对清单（Checklist）

请在正式跑全量实验前逐项确认：

**环境与数据**

- [ ] `docker run --gpus all ... fwi-benchmark:cu118 ...` 或 fwi-benchmark:cu118，容器内 `nvidia-smi` 可见 8 张 T4
- [ ] **8 卡 DDP**：所有训练统一 8 卡，禁止单卡；`run_benchmark_docker.sh`、`pre_launch_checklist.sh` 已配置 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- [ ] `bash benchmark/preflight_check.sh` 全部通过
- [ ] `benchmark/generated_split/*.txt` 已生成，样本数与预期一致
- [ ] `benchmark/generated_split/train_stats.json` 已生成（**全量训练集**，不是 `--max-samples 2000`）

**模型架构（9 种架构，共 16 个配置变体已通过验证）**

- [ ] 确认不存在旧 checkpoint（旧架构含 `convblock8`/`lastBlock`/硬编码 `output_lim`，新架构含 `global_pool + bottleneck`/动态 `output_lim`，不兼容）
- [ ] 冒烟测试通过：训练日志中出现 `[Benchmark] Seed set to <seed>`
- [ ] 冒烟测试通过：所有模型输出形状为 `(B, 1, 256, 256)` 或 `(B, 2, 256, 256)`（无 size mismatch 报错）
- [ ] FuteFWI 模型需确认 `einops` 已安装（`python -c "import einops; print(einops.__version__)"` 输出 `0.8.0`）
- [ ] ConvNeXtFWI 模型需确认 `timm` 已安装（`python -c "import timm"` 无报错）
- [ ] DCModel / FCNVMB 调用时传入 `label_dsp_dim=[256, 256]`（统一脚本与 profile 已处理）

**评测与统计**

- [ ] 所有 `eval_cmd` 都会打印 6 项指标（MSE/MAE/PSNR/SSIM/L1-Grad/LPIPS）
- [ ] 评测统一使用 `--benchmark-eval-lpips-mode real`（当前默认，避免 LPIPS real 下载失败）
- [x] `SEEDS` 默认设置为 `42`
- [ ] `BUDGET_MODE` 只选一种口径（epoch 或 wallclock），不混用

**运行**

- [ ] 所有实验在**同一 Docker 容器**内完成（推荐 `fwi-benchmark:cu118`），禁止混用宿主机
- [ ] `run_benchmark_suite.sh` 里的 `DATA_ROOT`、`WORK_ROOT` 已替换为真实容器内路径
- [ ] **统一 Batch Size**：`BATCH_PER_GPU` 默认 8（总 batch 64），全员统一
- [ ] TASK 中的 `__SEED__` 占位符正确（不是 `${BENCHMARK_SEED}`）
- [ ] 先完成单模型单 seed 冒烟测试并检查 CSV 字段完整（参见 Section 12）
- [ ] 再启动全模型全 seed 的正式运行

---

## 12. 冒烟测试与故障定位（必须先做）

先跑最小训练步数，快速验证"能否前向 + 反向 + 输出正确尺寸"。

### 12.1 OpenFWI InversionNet（示例）

```bash
cd OpenFWI

python train.py \
  --use-unified-loader \
  --data-root <DATA_ROOT> \
  --global-map-csv ../benchmark/generated_split/global_map.csv \
  --stats-json ../benchmark/generated_split/train_stats.json \
  --align-multiple 32 \
  --align-mode crop \
  --seed 42 \
  --output-height 256 --output-width 256 \
  -m InversionNet -b 2 -eb 1 -nb 1 -j 2 \
  --vis-interval 1 \
  -o smoke_test -n invnet -s test1
```

**验证要点**（检查训练日志）：

1. `[Benchmark] Seed set to 42` — seed 设置成功
2. 无 `RuntimeError: size mismatch` — 模型架构与数据尺寸兼容
3. loss 数值正常下降（不是 NaN 或无穷大）
4. `visualizations/epoch_0001.png` 生成 — 可视化输出正常
5. `checkpoint.pth` 生成 — checkpoint 保存正常

### 12.2 FuTE-FWI VelocityGAN（示例）

```bash
cd FuTE-FWI

python train_velocitygan.py \
  --use-unified-loader \
  --data-root <DATA_ROOT> \
  --global-map-csv ../benchmark/generated_split/global_map.csv \
  --stats-json ../benchmark/generated_split/train_stats.json \
  --align-multiple 32 \
  --align-mode crop \
  --seed 42 \
  --output-height 256 --output-width 256 \
  --batch-size 2 --epochs 1 \
  --save-interval 1 --vis-interval 1 \
  --output ../smoke_velocitygan --name VelocityGAN
```

### 12.3 冒烟测试完整链路（训练→评测）

```bash
cd OpenFWI

# 训练 1 epoch
python train.py \
  --use-unified-loader \
  --data-root <DATA_ROOT> \
  --global-map-csv ../benchmark/generated_split/global_map.csv \
  --stats-json ../benchmark/generated_split/train_stats.json \
  --align-multiple 32 --align-mode crop \
  --seed 42 --output-height 256 --output-width 256 \
  -m InversionNet -b 2 -eb 1 -nb 1 -j 2 \
  -o smoke_test -n invnet -s smoke

# 测试 + 评测
python test.py \
  --use-unified-loader \
  --data-root <DATA_ROOT> \
  --global-map-csv ../benchmark/generated_split/global_map.csv \
  --stats-json ../benchmark/generated_split/train_stats.json \
  --align-multiple 32 --align-mode crop \
  --output-height 256 --output-width 256 \
  -m InversionNet -b 2 -j 2 \
  -o smoke_test -n invnet -s smoke \
  -r checkpoint.pth \
  --export-dir ../smoke_eval \
  --benchmark-eval ../benchmark/benchmark_eval.py \
  --benchmark-eval-json ../smoke_eval/metrics.json
```

**评测成功的标志**：日志中出现 6 项指标（即使数值很差也说明链路通了）：

```
MSE: 0.xxxxx
MAE: 0.xxxxx
PSNR: xx.xx
SSIM: 0.xxxx
L1-Grad: 0.xxxxx
LPIPS: 0.xxxxx
```

### 12.4 常见报错与处理

| 报错 | 原因 | 修复 |
|------|------|------|
| `numba 0.57.1 requires numpy<1.25,>=1.21, but you have numpy 1.26.4` | numba 0.57.1 不支持 numpy 1.26，VelocityGAN 等依赖 Numba 的模型会崩溃 | `pip install "numba>=0.58.1"`（0.58.1+ 支持 numpy 1.26） |
| `RuntimeError: invalid hash value` (LPIPS real 模式) | lpips 库下载 AlexNet 权重时哈希校验失败 | 使用 `--benchmark-eval-lpips-mode real`（proxy 不依赖预训练权重） |
| `RuntimeError: size mismatch` | 加载了旧 checkpoint（旧架构有 `convblock8`/`lastBlock`，新架构有 `global_pool + bottleneck`） | 删除旧 checkpoint，重新训练 |
| `RuntimeError: Expected 4-dimensional input` | 数据维度不对 | 检查 `output_channel_dim=True` 是否设置 |
| `FileNotFoundError: train_stats.json` | 未生成 train-only 统计 | 运行 `compute_train_stats.py` |
| `CUDA out of memory` | batch size 过大 | 减小 `-b`；或使用 `--time-downsample` |
| 评测显示 `LPIPS: NA` | `lpips` 库未安装 | `pip install lpips==0.1.4` |
| `ModuleNotFoundError: No module named 'einops'` | FuteFWI/Ablation 模型需要 einops | `pip install einops==0.8.0` |
| `NameError: name 'unetUp1' is not defined` | ddnet/DDNet70.py 原始 bug 未修复 | 确认使用了修改后的 `ddnet/net/DDNet70.py`（`unetUp1` → `unetUp`） |
| `TypeError: forward() missing ... 'label_dsp_dim'` | DCModel 需要 `label_dsp_dim` 参数 | 调用时传 `model(x, label_dsp_dim=None)` |
| `RuntimeError: The size of tensor a (X) must match the size of tensor b (Y)` | UNet skip-connection 尺寸不匹配 | 检查输入是否经过 `align_multiple=32` 裁剪（`3000→2976`） |
| `unrecognized arguments: --use-unified-loader`（FuteFWI） | train_futefwi 缺少 benchmark 参数 | 确认 `FuTE-FWI/utils/argparser.py` 中 `parse_train_futefwi_args()` 已添加 `--use-unified-loader` 等参数 |
| `ModuleNotFoundError: No module named 'tkinter'`（DCNet） | matplotlib 使用 TkAgg 需 GUI | 将 `DCNet/lib_config.py` 中 `matplotlib.use('TkAgg')` 改为 `matplotlib.use('Agg')` |
| `AttributeError: 'list' object has no attribute 'size'`（DDNet70） | DataLoader 返回的 label 为 list | 确认 `unified_benchmark_train.py` 中已添加 `if isinstance(label, list): label = torch.stack(label)` |
| `ModuleNotFoundError: No module named 'net.ABA_FWI'` | ABA-FWI 缺少 ABA_FWI 模块 | 确认存在 `ABA-FWI/ABA-FWI_2.0/net/ABA_FWI.py`（封装 ABA_FWI_SEG） |
| `ImportError: cannot import name 'ABA_Loss' from 'net.ABA_FWI'`（profile） | ABA_FWI.py 无 ABA_Loss | `profile_model_metrics.py` aba-fwi 分支仅导入 `ABA_FWI`、`FCNVMB_FWI`，移除 `ABA_Loss`（见 CHANGELOG 第十九节） |
| `ImportError: cannot import name 'ConvNeXtFWI'` | ConvNeXt-FWI 源码缺失 | 确认存在 `ConvNeXt-FWI/models/ConvNeXtFWI.py` 和 `__init__.py`；需安装 `timm` |

---

## 13. 修改记录与已修改文件完整清单

本节记录了为实现"高分辨率公平基准测试"所做的**全部代码修改**，共涉及 **7 个仓库、25+ 个文件、16 个模型**。包含 2026-02-26 六模型训练失败修复（FuteFWI、DCNet、DDNet70、ABA_FWI、ConvNeXtFWI、FCNVMB）。便于复查、回溯和 Code Review。

### 13.1 致命问题修复

#### A. 模型架构硬编码 → 分辨率无关（Resolution-Agnostic）— 第一批（4 个模型）

**问题**：`InversionNet` / `Generator` / `FCN4_Deep_Resize_2` 的编码器末端 `convblock8` 使用硬编码卷积核 `(8, ceil(70/8))`，仅适用于宽度 70 的输入；解码器末端 `F.pad` 裁剪到固定尺寸 `(70,70)` / `(140,70)`。无法处理 256 宽度的高分辨率数据。

**修复**：

| 修改文件 | 修改内容 |
|---------|---------|
| `OpenFWI/network.py` | `InversionNet`：`convblock8` → `global_pool(AdaptiveAvgPool2d) + bottleneck(Conv 1x1)`；移除 `F.pad`；添加 `F.interpolate(output_size)`；新增 `output_size` 构造参数 |
| `OpenFWI/network.py` | `FCN4_Deep_Resize_2`（UPFWI）：同上 |
| `FuTE-FWI/models/VelocityGAN.py` | `Generator`：同上；移除 `from math import ceil` |
| `FuTE-FWI/models/InversionNet.py` | `InversionNet`：同上；移除 `import math` |

#### A2. 模型架构硬编码 → 分辨率无关 — 第二批（12 个模型）

**问题**：DCNet、ddnet、TU-Net、FuTE-FWI(FuteFWI) 仓库中的多个模型存在类似的空间尺寸硬编码问题。

**修复**：

**DCNet 仓库（3 个模型）：**

| 修改文件 | 模型 | 修改内容 |
|---------|------|---------|
| `DCNet/DCNet.py` | `DCNet` / `DCModel` | `lastBlock`（`kernel_size=(8, ceil(70*s/8))`）→ `global_pool(AdaptiveAvgPool2d) + bottleneck(Conv 1x1)`；`F.pad([-5,-5,-5,-5])` → `F.interpolate(output_size)`；`DCModel` 新增 `output_size` 参数透传；移除 `from math import ceil` |
| `DCNet/func/ddnet.py` | `DDNet70Model` | 新增 `output_size` 参数；两处 `F.pad(dc1_up2, [-1,-1,-29,-29])` → `F.interpolate(dc1_up2, size=self.output_size)`（dc1、dc2 两个解码分支） |
| `DCNet/func/comparison_net.py` | `InversionNet` | 同 OpenFWI InversionNet 改造（`convblock8` → `global_pool + bottleneck`；`F.pad` → `F.interpolate`）；移除 `from math import ceil` |

**ddnet 仓库（3 个模型）：**

| 修改文件 | 模型 | 修改内容 |
|---------|------|---------|
| `ddnet/net/InversionNet.py` | `InversionNet` | 同 OpenFWI InversionNet 改造；**新增** `import torch.nn.functional as F`（原文件缺少）；移除 `from math import ceil` |
| `ddnet/net/DDNet70.py` | `DDNet70Model` | 新增 `output_size` 参数；硬编码 `output_lim`（`[9,9],[18,18],[35,35],[70,70]`）→ 基于 `output_size` 动态计算（`[oh//8,ow//8],...,[oh,ow]`）；**修复原始 bug**：`unetUp1/unetUp2/netUp1/netUp2`（不存在的类）→ `unetUp/netUp` |
| `ddnet/net/DDNet70.py` | `SDNet70Model` | 同上改造方式和 bug 修复 |

**TU-Net 仓库（4 个模型）：**

| 修改文件 | 模型 | 修改内容 |
|---------|------|---------|
| `TU-Net/model/InversionNet.py` | `InversionNet` | 同 OpenFWI InversionNet 改造；移除 `from math import ceil` |
| `TU-Net/model/DDNet70.py` | `DDNet70Model` / `SDNet70Model` | 同 `ddnet/net/DDNet70.py` 改造方式；动态 `output_lim` |
| `TU-Net/model/details.py` | `SeismicRecordDownSampling` | `F.interpolate(x, size=[560, 70])` → `F.interpolate(x, size=[width*8, width])`，`width=x.shape[3]` |
| `TU-Net/model/details.py` | `SeismicRecordDownSampling2` | `F.interpolate(dim_reduce7, size=(70, 70))` → `F.interpolate(dim_reduce7, size=(w, w))`，`w=x.shape[3]` |
| `TU-Net/model/TU_Net.py` | `TU_Net` | 新增 `output_size` 参数；decoder `UNetUp2` 的硬编码 `output_lim` → 动态计算；`forward` 末尾添加 `F.interpolate(output_size)`；**新增** `import torch.nn.functional as F` |

**FuTE-FWI 仓库 — FuteFWI Transformer 系列（3 个模型）：**

| 修改文件 | 模型 | 修改内容 |
|---------|------|---------|
| `FuTE-FWI/models/FuteFWI.py` | `ResNet`（共用组件） | 移除 `self.pad = nn.ConstantPad2d((0,0,-2,-3),0)` 及 forward 中的 `self.pad(x)` 调用 |
| `FuTE-FWI/models/FuteFWI.py` | `FuteFWI` | 新增 `output_size`, `grid_h`, `grid_w` 参数；添加 `self.adaptive_pool = AdaptiveAvgPool2d((grid_h, grid_w))` 在 ResNet 输出后；`pos_embedding` 改为 `nn.Parameter(torch.randn(1, grid_h*grid_w, hidden_size))`；`rearrange` 中 `h=20, w=12` → 运行时 `h, w = x.shape[2], x.shape[3]`；decoder 移除 `ConstantPad2d((-13,-13,-10,-10),0)` 和 `self.pad(x)`，替换为 `F.interpolate(output_size)`；添加 `import torch.nn.functional as F` |
| `FuTE-FWI/models/FuteFWI.py` | `Ablation_1` | 新增 `output_size`, `grid_h`, `grid_w`；`patch_embedding` 后添加 `AdaptiveAvgPool2d`；`pos_embedding` 和 `rearrange` 动态化；decoder 同上 |
| `FuTE-FWI/models/FuteFWI.py` | `Ablation_2` | 新增 `output_size`, `grid_h`, `grid_w`；ResNet 后添加 `AdaptiveAvgPool2d`；decoder 同上 |

#### B. Shell 变量提前展开 → `__SEED__` 占位符

**问题**：`run_benchmark_suite.sh` 中 `TASKS` 数组在定义时使用 `${BENCHMARK_SEED}`，该变量在数组创建时就被 Shell 展开为空字符串（因为 seed 循环尚未开始），导致所有 seed 共用同一个输出路径，结果互相覆盖。

**修复**：完全重写 `run_benchmark_suite.sh`。使用字面字符串 `__SEED__` 作为占位符，在 seed 循环内通过 `${train_cmd//__SEED__/${seed}}` 进行运行时替换。

#### C. 训练脚本缺少 seed 设置 → `setup_benchmark_seed()`

**问题**：`OpenFWI/train.py` 和 `FuTE-FWI/train_*.py` 均未设置 `torch.manual_seed` 等随机种子，单 seed 运行不可复现。

**修复**：

| 修改文件 | 修改内容 |
|---------|---------|
| `OpenFWI/train.py` | 添加 `setup_benchmark_seed()` 函数；新增 `--seed` CLI 参数 |
| `FuTE-FWI/train_velocitygan.py` | 同上 |
| `FuTE-FWI/train_inversionnet.py` | 同上 |
| `FuTE-FWI/train_futefwi.py` | 同上 |

### 13.2 重要问题修复

#### D. 缺少 compute_train_stats 全量生成

**问题**：`preflight_check.sh` 使用 `--max-samples 2000` 只采样部分训练集计算统计；`run_benchmark_suite.sh` 未包含 stats 生成步骤。

**修复**：

| 修改文件 | 修改内容 |
|---------|---------|
| `benchmark/preflight_check.sh` | 移除 `--max-samples 2000`，改为全量计算 |
| `run_benchmark_suite.sh` | Step 0 自动检测 `train_stats.json`，不存在则全量生成 |

#### E. 模型实例化未传入 output_size

**问题**：模型构造函数新增了 `output_size` 参数，但训练/测试脚本仍使用旧的创建方式。

**修复**：

| 修改文件 | 修改内容 |
|---------|---------|
| `OpenFWI/train.py` | `model = network.model_dict[args.model](..., output_size=output_size)` |
| `FuTE-FWI/train_velocitygan.py` | `model = Generator(output_size=output_size)` |
| `FuTE-FWI/train_inversionnet.py` | `model = InversionNet(output_size=output_size)` |
| `FuTE-FWI/train_futefwi.py` | `model = FuteFWI(output_size=output_size, grid_h=20, grid_w=12)` |
| `unified_benchmark_test.py` | 创建模型时传入 `output_size`（及 FuteFWI 的 `grid_h`, `grid_w`）；checkpoint 加载支持 `model_state_dict`、`model`（OpenFWI）、纯 state_dict（FuTE-FWI .pt） |
| `profile_model_metrics.py` | `get_model()` 传入 `output_size`；`_ensure_repo_path(repo)` 为各仓库添加路径到 `sys.path`，解决 profile 阶段 import 失败 |

#### F. LPIPS 评测参数

**说明**：benchmark 统一使用 `--lpips-mode real`（AlexNet backbone，论文协议）。`unified_benchmark_test.py` 调用 `benchmark_eval.py` 时显式传入 `--lpips-mode real`；网络受限时可临时改用 `proxy`，但其结果与论文 LPIPS 不可比。

#### G. CLI 参数缺失

| 修改文件 | 新增参数 |
|---------|---------|
| `OpenFWI/train.py` | `--seed`, `--output-height`, `--output-width` |
| `unified_benchmark_test.py` | `--output-height`, `--output-width`（评测入口） |
| `FuTE-FWI/utils/argparser.py` | 所有 4 个 parser 均新增 `--seed`, `--output-height`, `--output-width` |

#### H. 原始代码 bug 修复

| 修改文件 | Bug 描述 | 修复方式 |
|---------|---------|---------|
| `ddnet/net/DDNet70.py` | `DDNet70Model` / `SDNet70Model` 引用不存在的类 `unetUp1`, `unetUp2`, `netUp1`, `netUp2`（NameError） | 替换为实际存在的 `unetUp`, `netUp` 类 |
| `TU-Net/model/DDNet70.py` | 同上 | 同上 |
| `DCNet/DCNet.py` | `DCModel.forward()` 签名中 `label_dsp_dim` 为必选参数但内部未使用 | 保留 API 不变，调用时传 `None` |

#### I. VelocityGAN 8 卡 DDP 兼容

**问题**：VelocityGAN 使用 WGAN-GP，`create_graph=True` 与 SyncBatchNorm 不兼容，8 卡 DDP 下 gradient penalty 计算异常。

**修复**：

| 修改文件 | 修改内容 |
|---------|---------|
| `FuTE-FWI/models/VelocityGAN.py` | Discriminator 默认 `norm="in"`（InstanceNorm）；ConvBlock/DeconvBlock `inplace=False` |
| `FuTE-FWI/utils/loss.py` | `Wasserstein_GP` 新增 `model_for_gp` 参数，DDP 下用 `model.module` 计算 GP |
| `FuTE-FWI/utils/utils.py` | `train_gan` 传入 `model_for_gp=getattr(model_d, "module", model_d)` |
| `run_benchmark_suite.sh` | VelocityGAN 训练无 `--sync-bn` |

#### J. 六模型训练失败修复（2026-02-26）

**问题**：FuteFWI、DCNet、DDNet70、ABA_FWI、ConvNeXtFWI、FCNVMB 在 benchmark 训练阶段报错，无法完成训练。

**修复**：

| 模型 | 错误 | 修改 |
|------|------|------|
| **FuteFWI** | `unrecognized arguments: --use-unified-loader ...` | `FuTE-FWI/utils/argparser.py`：`parse_train_futefwi_args()` 新增 `--use-unified-loader`、`--data-root`、`--global-map-csv`、`--stats-json`、`--align-multiple`、`--align-mode`、`--seed`、`--save-interval`、`--vis-interval`、`--output-height`、`--output-width`、`--sync-bn` |
| **DCNet** | `ModuleNotFoundError: No module named 'tkinter'` | `DCNet/lib_config.py`：`matplotlib.use('TkAgg')` → `matplotlib.use('Agg')` |
| **DDNet70** | `AttributeError: 'list' object has no attribute 'size'` | `benchmark/unified_benchmark_train.py`：训练/验证循环中 `if isinstance(label, list): label = torch.stack(label)` |
| **ABA_FWI** | `ModuleNotFoundError: No module named 'net.ABA_FWI'` | 新增 `ABA-FWI/ABA-FWI_2.0/net/ABA_FWI.py`：封装 `ABA_FWI_SEG`，提供 `ABA_FWI(output_size=...)` 和 `forward(x)` |
| **ABA_FWI/FCNVMB** | profile 阶段 `ImportError: cannot import name 'ABA_Loss'` | `profile_model_metrics.py` aba-fwi 分支移除 `ABA_Loss` 导入（见 CHANGELOG 第十九节） |
| **ConvNeXtFWI** | `ImportError: cannot import name 'ConvNeXtFWI'`（源码缺失） | 新增 `ConvNeXt-FWI/models/ConvNeXtFWI.py`、`__init__.py`：基于 timm ConvNeXt + 简单 decoder 的新实现 |
| **FCNVMB** | 同 ABA_FWI（使用 aba-fwi 仓库） | 新增 `ABA-FWI/ABA-FWI_2.0/net/FCNVMB.py`：`FCNVMB_FWI(model_dim=..., in_channels=5)` 和 `forward(x)` |

**性能影响**：除 ConvNeXtFWI 外，其余修改不改变模型结构或训练逻辑。ConvNeXtFWI 因原始源码缺失，当前为替代实现，与原版可能存在差异。

### 13.3 文档与依赖更新

| 修改文件 | 修改内容 |
|---------|---------|
| `benchmark/IMPLEMENTATION_GUIDE_ZH.md` | 全面更新：涵盖所有 16 个模型改造详情、依赖清单、数据流验证、各仓库接入说明、六模型训练失败修复 |
| `benchmark/docker/requirements.txt` | 已包含 `einops==0.8.0`（FuteFWI）、`timm>=0.9.0`（ConvNeXtFWI）、`PyWavelets>=1.4.0`（ABA-FWI）；新增注释标明各模型依赖 |
| `benchmark/CHANGELOG_BENCHMARK.md` | 新增第 12 节：六模型训练失败修复 |

### 13.4 已修改文件完整列表（按仓库分组）

#### OpenFWI（2 个文件）

| 文件 | 修改内容 |
|------|---------|
| `OpenFWI/network.py` | `InversionNet` + `FCN4_Deep_Resize_2` 分辨率无关改造 |
| `OpenFWI/train.py` | `setup_benchmark_seed()`；`output_size` 传入；CLI 参数 |

> **说明**：OpenFWI 无 `test.py`，benchmark 评测使用 `unified_benchmark_test.py`。

#### FuTE-FWI（9 个文件）

| 文件 | 修改内容 |
|------|---------|
| `FuTE-FWI/models/VelocityGAN.py` | `Generator` 分辨率无关改造；`Discriminator` 默认 `norm="in"`，ConvBlock/DeconvBlock `inplace=False`（8 卡 DDP + WGAN-GP 兼容） |
| `FuTE-FWI/models/InversionNet.py` | `InversionNet` 分辨率无关改造 |
| `FuTE-FWI/models/FuteFWI.py` | `ResNet` 移除 pad；`FuteFWI`/`Ablation_1`/`Ablation_2` 分辨率无关改造（AdaptivePool + 动态 pos_embedding + F.interpolate） |
| `FuTE-FWI/train_velocitygan.py` | `setup_benchmark_seed()`；`output_size` |
| `FuTE-FWI/train_inversionnet.py` | `setup_benchmark_seed()`；`output_size` |
| `FuTE-FWI/train_futefwi.py` | `setup_benchmark_seed()`；`output_size`/`grid_h`/`grid_w` |
| `FuTE-FWI/utils/argparser.py` | 所有 parser 新增 `--seed`/`--output-height`/`--output-width`；`parse_train_futefwi_args()` 新增 benchmark 参数（`--use-unified-loader`、`--data-root` 等） |
| `FuTE-FWI/utils/loss.py` | `Wasserstein_GP` 新增 `model_for_gp` 参数，DDP 下用 `model.module` 计算 gradient penalty |
| `FuTE-FWI/utils/utils.py` | `train_gan` 传入 `model_for_gp=getattr(model_d, "module", model_d)` |

#### DCNet（4 个文件）

| 文件 | 修改内容 |
|------|---------|
| `DCNet/DCNet.py` | `DCNet`/`DCModel` 分辨率无关改造 |
| `DCNet/func/ddnet.py` | `DDNet70Model` 分辨率无关改造 |
| `DCNet/func/comparison_net.py` | `InversionNet` 分辨率无关改造 |
| `DCNet/lib_config.py` | `matplotlib.use('TkAgg')` → `matplotlib.use('Agg')`（无头环境兼容） |

#### ddnet（2 个文件）

| 文件 | 修改内容 |
|------|---------|
| `ddnet/net/InversionNet.py` | `InversionNet` 分辨率无关改造；新增 `import F` |
| `ddnet/net/DDNet70.py` | `DDNet70Model`/`SDNet70Model` 分辨率无关改造 + 原始 bug 修复 |

#### TU-Net（4 个文件）

| 文件 | 修改内容 |
|------|---------|
| `TU-Net/model/InversionNet.py` | `InversionNet` 分辨率无关改造 |
| `TU-Net/model/DDNet70.py` | `DDNet70Model`/`SDNet70Model` 分辨率无关改造 |
| `TU-Net/model/details.py` | `SeismicRecordDownSampling`/`SeismicRecordDownSampling2` 硬编码尺寸动态化 |
| `TU-Net/model/TU_Net.py` | `TU_Net` 分辨率无关改造；新增 `import F` |

#### ABA-FWI（2 个文件，新增）

| 文件 | 修改内容 |
|------|---------|
| `ABA-FWI/ABA-FWI_2.0/net/ABA_FWI.py` | 新增：封装 `ABA_FWI_SEG`，提供 `ABA_FWI(output_size=...)` 和 `forward(x)`；`forward` 末尾若输出尺寸与 `output_size` 不符则 `F.interpolate` 对齐（与 FuteFWI/InversionNet 一致） |
| `ABA-FWI/ABA-FWI_2.0/net/FCNVMB.py` | 新增：`FCNVMB_FWI(model_dim=..., in_channels=5)` 和 `forward(x)` |

#### ConvNeXt-FWI（2 个文件，新增）

| 文件 | 修改内容 |
|------|---------|
| `ConvNeXt-FWI/models/ConvNeXtFWI.py` | 新增：基于 timm ConvNeXt + decoder 的 FWI 模型（原源码缺失） |
| `ConvNeXt-FWI/models/__init__.py` | 新增：导出 `ConvNeXtFWI` |

#### 根目录 / benchmark（4 个文件）

| 文件 | 修改内容 |
|------|---------|
| `run_benchmark_suite.sh` | 完全重写：`__SEED__` 占位符；auto-detect split+stats；InversionNet/VelocityGAN/FuteFWI 评测用 `unified_benchmark_test.py` |
| `benchmark/unified_benchmark_train.py` | 训练/验证循环中 `if isinstance(label, list): label = torch.stack(label)`（DDNet70 修复） |
| `benchmark/unified_benchmark_test.py` | 统一评测入口；checkpoint 加载支持 `model_state_dict`/`model`/纯 state_dict；LPIPS 固定 `proxy` 模式 |
| `profile_model_metrics.py` | `get_model()` 传入 `output_size`；`_ensure_repo_path(repo)` 解决 profile 阶段 import 失败 |

### 13.5 未修改但需注意的文件

| 文件 | 说明 |
|------|------|
| `OpenFWI/network.py` — `Discriminator` | 使用 `view(B, -1)` 展平，天然适配任意输入尺寸，无需修改 |
| `FuTE-FWI/models/VelocityGAN.py` — `Discriminator` | 同上 |
| `ddnet/net/DDNet.py` — `DDNetModel` | 使用 `label_dsp_dim` 参数控制输出，天然灵活，无需修改 |
| `ddnet/net/FCNVMB.py` — `FCNVMB` | 使用 `label_dsp_dim` 参数控制输出，无需修改 |
| `DCNet/func/comparison_net.py` — `FCNVMB` | 同上 |
| `FCNVMB/func/UnetModel.py` — `UnetModel` | 使用 `label_dsp_dim` 参数控制输出，无需修改 |
| `benchmark/unified_loader.py` | 未修改，已满足需求 |
| `benchmark/benchmark_eval.py` | 未修改，支持 `--lpips-mode real`（benchmark 默认，论文协议）和 `proxy`（备用） |
| `benchmark/compute_train_stats.py` | 未修改，`--max-samples` 改在调用端控制 |

### 13.6 跳过的模型（不适用于当前 benchmark）

| 文件 | 模型 | 跳过原因 |
|------|------|---------|
| `TU-Net/model/DDNet.py` | `DDNetModel` / `SDNetModel` | 29 通道输入、201×301 输出，SEG 数据集专用 |
| `TU-Net/model/TU_Net_SEG.py` | `TU_Net_SEG` | 29 通道输入、201×301 输出，SEG 数据集专用 |

