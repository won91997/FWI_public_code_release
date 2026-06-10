# Benchmark 公平模式说明

## 问题背景

原设计为 70×70 的模型（InversionNet、ABA_FWI、FCNVMB、FuteFWI、VelocityGAN）被强行喂入 256×256 数据，导致：

- **感受野/处理密度不匹配**：3×3 卷积在 70×70 上覆盖“半个波长”，在 256×256 上只能看到“波的一小段”
- **ABA_FWI**：FLOPs 飙至 641G，Transformer 陷入局部像素泥潭，条纹伪影
- **简单 F.interpolate**：只是“伪适配”，强行拉伸输入，模型内部无法消化多出的信息量

## 方案一：内部降维适配法（已实现）

**做法**：在模型 forward 入口下采样到 70×70，核心处理后再上采样回 256×256。

**启用方式**：

```bash
export BENCHMARK_FAIR_MODE=1
# 然后正常运行 benchmark
./run_benchmark_suite.sh
```

**适用模型**：ABA_FWI、FCNVMB、FuteFWI、InversionNet、VelocityGAN（通过 `unified_benchmark_train` / `unified_benchmark_test` 训练的）

**插值选择**：

- 下采样 (256→70)：`mode='area'`，更好保留波场能量平均值，减少混叠
- 上采样 (70→256)：`mode='bicubic'`，输出更平滑，减少锯齿

**效果**：

- 测试模型架构本身的特征提取能力，而非显存承受力
- ABA_FWI FLOPs 可降至约 40–60G（与 DCNet 20G 同量级）
- 条纹伪影有望消失，Attention 能捕捉完整地质结构

**实现位置**：

- `benchmark/fair_wrapper.py`：`FairWrapper` 类
- `benchmark/unified_benchmark_train.py`：`get_model_for_benchmark()`
- `benchmark/unified_benchmark_test.py`：使用 `get_model_for_benchmark`
- `profile_model_metrics.py`：使用 `get_model_for_benchmark`

**重要**：使用 `BENCHMARK_FAIR_MODE=1` 时，**必须从头重训**。切勿用之前在 256×256 上训练的 checkpoint 加上 FairWrapper 做测试——权重已适应高分辨率特征分布，效果会变差。请清空旧 checkpoint，从 epoch 0 开始训练。

### 模型覆盖（公平闭环）

| 类型 | 模型 | 处理方式 |
|------|------|----------|
| **统一脚本** | ABA_FWI、FCNVMB、TU_Net、DDNet70、DCNet、ConvNeXtFWI、ConvNeXtKaggle | `fair_wrapper.py` 配置，自动套 FairWrapper |
| **独立脚本** | InversionNet | `OpenFWI/train.py` 手动集成 FairWrapper |
| **独立脚本** | FuteFWI | `FuTE-FWI/train_futefwi.py` 手动集成 FairWrapper |
| **特殊** | VelocityGAN | **不加 FairWrapper**，仅 `SyncBatchNorm` 保证物理量级；GAN 需 256×256 生成纹理 |

所有模型（除 VelocityGAN）均执行 256 (Input) → 70 (Model) → 256 (Output)，FLOPs 预期在 10G～80G 区间，不再出现 600G+。

## 方案二：架构缩放法（待实现）

针对 CNN 类模型（DCNet、DDNet70、TU_Net、ConvNeXtFWI）：

- 在 Encoder 最前端增加 `Conv2d(kernel=7, stride=2, padding=3)`，将 256→128
- 或调整 Stride，使 bottleneck 特征图保持在 8×8 或 16×16

针对 Transformer（ABA_FWI、FuteFWI）：

- 将 Patch Size 增大 3–4 倍（如 16×16 或 32×32），避免序列长度爆炸

## 方案三：VelocityGAN 特殊处理（已实现）

- **SyncBatchNorm**：`BENCHMARK_FAIR_MODE=1` 时自动启用，与其他 CNN 保持物理量级一致
- **不加 FairWrapper**：GAN 需 256×256 高分辨率生成纹理细节

## 配置汇总

| 环境变量 | 说明 | 默认 |
|----------|------|------|
| `BENCHMARK_FAIR_MODE` | 1=启用 FairWrapper（方案一） | 0 |
| `BENCHMARK_OUTPUT_ROOT` | 输出根目录，可覆盖 | 见下 |

**输出目录**：公平模式时自动使用 `benchmark_logs_fair`，与普通 `benchmark_logs` 隔离，避免覆盖原结果。

## 运行示例

```bash
# 清理旧权重（必须从头训练）
rm -rf benchmark_logs_fair/*/checkpoint*.pth benchmark_logs_fair/*/epoch_*.png

# 公平模式：全量或指定模型
export BENCHMARK_FAIR_MODE=1
./run_benchmark_suite.sh

# 仅跑部分模型
BENCHMARK_FAIR_MODE=1 TASK_FILTER=ABA_FWI,FCNVMB,InversionNet,FuteFWI ./run_benchmark_suite.sh

# 指定种子
BENCHMARK_FAIR_MODE=1 TASK_FILTER=ABA_FWI SEEDS=42 ./run_benchmark_suite.sh
```

## 验证 FLOPs

运行 Profile 后，除 VelocityGAN 外，所有模型 FLOPs 应在 10G～80G 之间，不再出现 600G+。

## 赛道与考核点（论文 Methodology 参考）

| 赛道 | 模型 | 核心逻辑 | 考核点 |
|------|------|----------|--------|
| **A 回归组** | ABA_FWI, FuteFWI, InversionNet, DCNet, TU_Net, DDNet70, FCNVMB, ConvNeXtFWI, ConvNeXtKaggle | 256→[Down 70]→Model→[Up 256]→Loss | 用最少算力从有限低频信息恢复最准确地质结构 |
| **B 生成组** | VelocityGAN | 256→Model→256→Loss | 生成最逼真的高频纹理（LPIPS 优异） |

## 单独输出目录 + 指定种子

```bash
# 输出到 benchmark_logs_fair_seed1997，避免与 benchmark_logs_fair 冲突
BENCHMARK_FAIR_MODE=1 BENCHMARK_OUTPUT_ROOT=/path/to/repo/benchmark_logs_fair_seed1997 SEEDS=1997 ./run_benchmark_suite.sh
```

## ConvNeXtKaggle 双模式训练（Fair + NonFair）

ConvNeXtKaggle 源自 Kaggle 地震速度建模 baseline，支持 Fair（70×70 内部）与 NonFair（全尺寸 2976×256）两种模式。使用独立脚本 `run_convnext_kaggle_dual.sh` 可一次完成双模式训练与评测。

### 脚本与输出

| 项目 | 说明 |
|------|------|
| **脚本** | `run_convnext_kaggle_dual.sh`（根目录） |
| **输出** | `benchmark_logs_convnext_kaggle_seed42_b2`（Fair + NonFair 共用） |
| **GPU** | 默认 `0-7`，可通过 `GPU_SET=8,9,10,11,12,13,14,15` 指定后八卡 |

### 运行方式

```bash
# 默认前八卡
bash run_convnext_kaggle_dual.sh

# 指定后八卡
GPU_SET=8,9,10,11,12,13,14,15 bash run_convnext_kaggle_dual.sh
```

### 流程与耗时（100 epochs，batch=2，8 卡 DDP）

1. **profile FAIR** → 输出 `params_m`、`flops_g`、`infer_ms`（70×70 路径）
2. **train FAIR** → 约 1.5–2 小时
3. **eval FAIR** → 导出 metrics.json
4. **profile NONFAIR** → 全尺寸路径效率指标
5. **train NONFAIR** → 约 20 小时（全尺寸 2976×256）
6. **eval NONFAIR** → 导出 metrics.json

### 技术要点

- **Fair 模式**：`FAIR_MODELS_CONFIG["ConvNeXtKaggle"]=70`，与 ABA_FWI、ConvNeXtFWI 等基准一致
- **输出对齐**：模型内部 `_align_output` 将预测插值到 256×256，避免 loss 尺寸不匹配
- **反归一化**：使用 `train_stats.json` 的 `vmin`/`vmax`，替代原硬编码 `*1500+3000`
- **预训练**：`KAGGLE_CONVNEXT_PRETRAINED=1` 时从 HuggingFace 加载 `convnext_small.fb_in22k_ft_in1k`；国内环境已配置 `HF_ENDPOINT=https://hf-mirror.com`
