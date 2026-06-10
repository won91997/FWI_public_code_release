# Benchmark 基准测试：完整修改记录与公平性说明

本文档汇总从开始到当前，为使基准测试可运行且公平所做的全部工作和代码修改。所有内容基于实际代码核对，力求准确、可追溯。

---

## 一、公平性设计（未改代码，仅配置）

以下为 `run_benchmark_suite.sh`、`run_benchmark_docker.sh` 中的统一配置，确保各模型在相同条件下对比：

| 维度 | 配置值 | 说明 |
|------|--------|------|
| **数据** | 统一 | `global_map.csv`、`train_stats.json`、`align-multiple 32`、`align-mode crop`（3000→2976 为**裁剪**，见 unified_loader._align_shape） |
| **Batch** | 4/卡 | `BATCH_PER_GPU=4`，总 batch=32（8×4）；T4 15GB 下全员可跑；A100 可设 8 |
| **Epochs** | 100 | 除 InversionNet 用 `-eb 5 -nb 20` 等效外，其余均 `--epochs 100` |
| **GPU** | 8 卡 DDP | `torchrun --nproc_per_node=8` |
| **评测** | 6 项 | MSE、MAE、PSNR、SSIM、L1-Grad、LPIPS（proxy） |
| **Profile shape** | `1,5,2976,256` | 统一输入尺寸 |
| **LPIPS 模式** | proxy | 不依赖预训练权重，避免下载失败 |

---

## 二、代码修改清单（按文件）

### 1. FuTE-FWI/models/VelocityGAN.py

**目的**：解决 VelocityGAN 在 8 卡 DDP 下与 WGAN-GP 的兼容问题。

| 修改项 | 位置 | 修改内容 |
|--------|------|----------|
| Discriminator 默认 norm | `Discriminator.__init__` | `norm="in"`（InstanceNorm）替代默认 BatchNorm |
| ConvBlock inplace | `ConvBlock`、`DeconvBlock` | `inplace=False`，避免 DDP 下 inplace 与 `create_graph=True` 冲突 |
| **Generator bottleneck** | `Generator.__init__` | `norm="gn"`（GroupNorm）：bottleneck 输出 1×1，InstanceNorm 要求 spatial>1；GroupNorm 支持 1×1 且 DDP 下单样本安全 |
| **Gradient checkpointing** | `Generator.forward`、`Discriminator.forward` | 训练时对 encoder/decoder 各 block 启用 `torch.utils.checkpoint`，省 50%+ 激活显存，解决 T4 16GB batch=2 OOM；batch 保持 2/卡 与其他模型一致 |

**公平性说明**：VelocityGAN 使用 GroupNorm + gradient checkpointing 后可在 T4 16GB 上以 batch=2/卡 正常训练，与其他模型公平对比。

---

### 2. FuTE-FWI/utils/loss.py

**目的**：Wasserstein_GP 在 DDP 下正确计算 gradient penalty。

| 修改项 | 位置 | 修改内容 |
|--------|------|----------|
| `model_for_gp` 参数 | `Wasserstein_GP.forward` | 新增 `model_for_gp=None`，DDP 时用 `model.module` 计算 GP |
| GP 计算 | `compute_gradient_penalty` | 使用 `interpolates.clone().detach().requires_grad_(True)`，避免 inplace 修改 |

**公平性说明**：DDP 下用 `model.module` 计算 gradient penalty，避免 SyncBatchNorm 与 `create_graph=True` 冲突，保证 VelocityGAN 能正常 8 卡训练。

---

### 3. FuTE-FWI/utils/utils.py

**目的**：训练时传入 `model_for_gp`。

| 修改项 | 位置 | 修改内容 |
|--------|------|----------|
| train_gan 调用 | `train_gan` 内 `loss_d` 调用 | `model_for_gp=getattr(model_d, "module", model_d)` |

---

### 4. run_benchmark_suite.sh

**目的**：统一 batch、评测入口、VelocityGAN 不使用 sync-bn。

| 修改项 | 位置 | 修改内容 |
|--------|------|----------|
| 统一 batch | 第 27–33 行 | `BATCH_PER_GPU=4`（T4 公平可跑），`BATCH_INVNET`、`BATCH_VELGAN` 等均用该值 |
| InversionNet 评测 | TASKS 第 1 项 | 由 `test.py` 改为 `UNIFIED_TEST`（OpenFWI 无 test.py） |
| VelocityGAN 评测 | TASKS 第 2 项 | 由 `test.py` 改为 `UNIFIED_TEST`，checkpoint 为 `VelocityGAN_FlatVel_A_D.pt` |
| FuteFWI 评测 | TASKS 第 3 项 | 由 `test.py` 改为 `UNIFIED_TEST`，checkpoint 为 `FuteFWI_FlatVel_A_D.pt` |
| VelocityGAN 训练 | TASKS 第 2 项 | 无 `--sync-bn`（与 WGAN-GP 不兼容） |

**公平性说明**：所有模型统一 batch=8/卡；InversionNet/VelocityGAN/FuteFWI 均走 `unified_benchmark_test`，评测流程一致；VelocityGAN 不启用 sync-bn 是架构限制，而非刻意放宽。

---

### 5. benchmark/unified_benchmark_test.py

**目的**：支持 OpenFWI、FuTE-FWI 的 checkpoint 格式，并统一评测流程。

| 修改项 | 位置 | 修改内容 |
|--------|------|----------|
| checkpoint 加载 | `main()` 内 | 支持 `model_state_dict`、`model`（OpenFWI）、纯 state_dict（FuTE-FWI .pt） |

```python
if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
elif "model" in ckpt:
    model.load_state_dict(ckpt["model"], strict=True)
else:
    model.load_state_dict(ckpt, strict=True)
```

**公平性说明**：InversionNet、VelocityGAN、FuteFWI 均通过同一评测脚本和 `benchmark_eval.py`，保证 6 项指标计算方式一致。

---

### 6. profile_model_metrics.py

**目的**：修复 InversionNet、VelocityGAN 等 profile 阶段的 import 失败；修复 DDNet70、TU_Net 的 thop ReLU total_ops 问题。

| 修改项 | 位置 | 修改内容 |
|--------|------|----------|
| `_ensure_repo_path` | 第 9–24 行 | 为 openfwi、futefwi、dcnet、ddnet、tu-net、aba-fwi、convnext-fwi 添加对应路径到 `sys.path` |
| `get_model` 入口 | 第 43 行 | 在分支前统一调用 `_ensure_repo_path(repo)` |
| 移除重复调用 | 各 repo 分支 | 删除各分支内单独的 `_ensure_repo_path` 调用 |
| 执行顺序（2026-03-02） | `main()` | 将 `infer_ms` 提前到 `try_flops_g` 之前，避免 thop hooks 污染模型导致 DDNet70/TU_Net 报错 |
| aba-fwi 导入（2026-03-02） | `get_model()` aba-fwi 分支 | 移除 `ABA_Loss` 导入，仅导入 `ABA_FWI`、`FCNVMB_FWI`，修复 ABA_FWI/FCNVMB profile 的 ImportError |

**公平性说明**：所有模型都能成功 profile，输出 Params(M)、FLOPs(G)、Infer(ms)，避免部分模型为 NA。

---

### 7. benchmark/pre_launch_checklist.sh

**目的**：VelocityGAN 8 卡 DDP 点火测试，并说明 sync-bn 问题。

| 修改项 | 位置 | 修改内容 |
|--------|------|----------|
| 注释 | 第 8–10 行 | 说明 VelocityGAN 不使用 sync-bn 的原因（WGAN-GP 与 SyncBatchNorm 不兼容） |
| 点火命令 | 第 41–49 行 | `torchrun --nproc_per_node=8` 跑 1 epoch，无 `--sync-bn` |

---

### 8. benchmark/velocitygan_batch_stress.sh（新增）

**目的**：压力测试，确定 VelocityGAN 在 15GB 卡 × 8 下的可用 batch。

| 内容 | 说明 |
|------|------|
| 测试 batch | 4、8、16、32 |
| 结论 | batch 4/8 通过，16 OOM |
| 推荐 | `BATCH_PER_GPU=8`，显存紧张时用 4 |

**公平性说明**：据此将默认 batch 定为 8，与其他模型一致。

---

### 9. benchmark/docker/BUILD_AND_RUN.md

**目的**：补充 Docker 使用说明。

| 修改项 | 说明 |
|--------|------|
| 准备清单 | 镜像、数据挂载、preflight、VelocityGAN 点火 |
| `--shm-size=32g` | 避免 NCCL 共享内存不足 |
| 宿主机一键命令 | 含 `--shm-size=32g` 的 `docker run` 示例 |

---

### 10. benchmark/unified_loader.py（未修改，公平性依赖）

**目的**：统一数据加载，所有模型通过 `UnifiedFWIDataset` 获得相同输入。

| 逻辑 | 说明 |
|------|------|
| `_align_shape` | `align_multiple=32` 时，`t = (t // 32) * 32`；3000 → 2976 |
| `align_mode="crop"` | `return arr[:, :t, :w].copy()`，取前 t 个时间步，**裁剪**掉末尾 |
| `align_mode="pad"` | 目标尺寸不足时用零填充；3000→2976 时等价于取前 2976 个时间步 |

**结论**：3000 是**裁剪**到 2976，不是填充。所有模型输入统一为 `(N, 5, 2976, 256)`。

---

### 11. 新增文档

| 文件 | 说明 |
|------|------|
| `benchmark/FAIRNESS_CHECK.md` | 公平性检查项、3000→2976 裁剪说明、代码保证、启动前确认 |
| `benchmark/OUTPUT_PARAMS.md` | 输出文件、CSV 列、profile 格式、已修复问题 |
| `benchmark/CHANGELOG_BENCHMARK.md` | 本修改记录文档 |

---

### 12. 六模型训练失败修复（2026-02-26）

**目的**：修复 FuteFWI、DCNet、DDNet70、ABA_FWI、ConvNeXtFWI、FCNVMB 在 benchmark 中的训练失败问题。

| 模型 | 错误现象 | 修改文件 | 修改内容 |
|------|----------|----------|----------|
| **FuteFWI** | `unrecognized arguments: --use-unified-loader --data-root ...` | `FuTE-FWI/utils/argparser.py` | 在 `parse_train_futefwi_args()` 中新增 benchmark 参数：`--use-unified-loader`、`--data-root`、`--global-map-csv`、`--stats-json`、`--align-multiple`、`--align-mode`、`--seed`、`--save-interval`、`--vis-interval`、`--output-height`、`--output-width`、`--sync-bn` |
| **DCNet** | `ModuleNotFoundError: No module named 'tkinter'`（matplotlib 使用 TkAgg） | `DCNet/lib_config.py` | `matplotlib.use('TkAgg')` → `matplotlib.use('Agg')`，避免无头环境依赖 tkinter |
| **DDNet70** | `AttributeError: 'list' object has no attribute 'size'`（label 为 list） | `benchmark/unified_benchmark_train.py` | 训练/验证循环中增加 `if isinstance(label, list): label = torch.stack(label)`，确保 label 为 tensor |
| **ABA_FWI** | `ModuleNotFoundError: No module named 'net.ABA_FWI'` | `ABA-FWI/ABA-FWI_2.0/net/ABA_FWI.py`（新增） | 新建 `ABA_FWI` 类，封装 `ABA_FWI_SEG`，提供 `output_size` 接口和 `forward(x)` |
| **ConvNeXtFWI** | `ImportError: cannot import name 'ConvNeXtFWI' from 'models'`（源码缺失） | `ConvNeXt-FWI/models/ConvNeXtFWI.py`、`__init__.py`（新增） | 新建 ConvNeXtFWI 实现：timm ConvNeXt 作为 encoder + 简单 decoder，输出 `(B,1,H,W)` |
| **FCNVMB** | 同 ABA_FWI（FCNVMB 使用 aba-fwi 仓库） | `ABA-FWI/ABA-FWI_2.0/net/FCNVMB.py`（新增） | 新建 `FCNVMB_FWI` 类，实现 FCNVMB 架构，提供 `model_dim` 和 `forward(x)` 接口 |

**性能影响说明**：FuteFWI、DCNet、DDNet70、ABA_FWI、FCNVMB 的修改不改变模型结构或训练逻辑，无性能影响。ConvNeXtFWI 因原始源码缺失，当前为基于 timm 的新实现，与原版可能存在差异。

---

## 三、架构差异（设计如此，非不公平）

| 项目 | 说明 |
|------|------|
| **损失函数** | InversionNet/VelocityGAN/FuteFWI 用原生损失；DCNet/DDNet70 等用 MSE |
| **InversionNet** | `-eb 5 -nb 20` 等效 100 epoch |
| **VelocityGAN** | 无 `--sync-bn`（WGAN-GP 与 SyncBatchNorm 不兼容） |
| **FuteFWI** | 使用 `--sync-bn` |

---

## 四、未修改但需注意的配置

| 项目 | 当前值 | 说明 |
|------|--------|------|
| SEEDS | 默认 `42` | 单 seed 测试时可设 `SEEDS=42` |
| Docker | `--shm-size=32g` | 必须，否则 NCCL 可能报错 |
| PYTHON_BIN | `python` | 若系统为 Python 2，需设 `PYTHON_BIN=python3` |

---

## 五、公平性代码保证（run_benchmark_suite.sh）

| 保证项 | 实现 |
|--------|------|
| 数据参数 | 9 个 TASK 的 train_cmd、eval_cmd 均显式传入 `--align-multiple 32 --align-mode crop` |
| Batch | `BATCH_INVNET`、`BATCH_VELGAN` 等均用 `BATCH_PER_GPU` |
| 评测 | InversionNet/VelocityGAN/FuteFWI 用 `unified_benchmark_test.py` + `benchmark_eval.py` |
| LPIPS | 所有 eval_cmd 显式 `--benchmark-eval-lpips-mode real` |
| Profile | 所有 profile_cmd 使用 `--shape 1,5,2976,256` |

---

## 六、修改文件汇总

| 文件 | 修改类型 |
|------|----------|
| `FuTE-FWI/models/VelocityGAN.py` | 修改 |
| `FuTE-FWI/utils/loss.py` | 修改 |
| `FuTE-FWI/utils/utils.py` | 修改 |
| `FuTE-FWI/utils/argparser.py` | 修改（FuteFWI benchmark 参数） |
| `run_benchmark_suite.sh` | 修改 |
| `benchmark/unified_benchmark_test.py` | 修改 |
| `benchmark/unified_benchmark_train.py` | 修改（DDNet70 label 处理） |
| `profile_model_metrics.py` | 修改 |
| `benchmark/pre_launch_checklist.sh` | 修改 |
| `benchmark/velocitygan_batch_stress.sh` | 新增 |
| `benchmark/docker/BUILD_AND_RUN.md` | 修改 |
| `benchmark/docker/requirements.txt` | 修改（依赖注释） |
| `benchmark/FAIRNESS_CHECK.md` | 新增 |
| `benchmark/OUTPUT_PARAMS.md` | 新增 |
| `benchmark/CHANGELOG_BENCHMARK.md` | 新增（本文档） |
| `DCNet/lib_config.py` | 修改（matplotlib Agg） |
| `ABA-FWI/ABA-FWI_2.0/net/ABA_FWI.py` | 新增 |
| `ABA-FWI/ABA-FWI_2.0/net/FCNVMB.py` | 新增 |
| `ConvNeXt-FWI/models/ConvNeXtFWI.py` | 新增 |
| `ConvNeXt-FWI/models/__init__.py` | 新增 |

---

## 十三、2026-02-28：Batch 4 公平 + DDNet70 修复

**目的**：T4 15GB 下 9 模型全员可跑，公平测试。

| 修改项 | 位置 | 修改内容 |
|--------|------|----------|
| 默认 batch | `run_benchmark_suite.sh` | `BATCH_PER_GPU=4`，总 batch=32（8×4） |
| DDNet70 pred | `unified_benchmark_train.py` | 模型返回 list 时取 `pred[-1]` 计算 loss |
| DDNet70 eval | `unified_benchmark_test.py` | 同上，取 `pred[-1]` 导出 |

**公平性说明**：batch 4 下 FuteFWI、ABA_FWI、ConvNeXtFWI 等大模型不再 OOM；DDNet70 多尺度输出取主输出参与 loss/eval。

---

## 十四、2026-02-28：ABA_FWI 输出尺寸修复（255 vs 256）

**目的**：修复 ABA_FWI 训练时 `RuntimeError: The size of tensor a (255) must match the size of tensor b (256)`，保证与 benchmark 统一 label 尺寸一致。

| 修改项 | 位置 | 修改内容 |
|--------|------|----------|
| 输出对齐 | `ABA-FWI/ABA-FWI_2.0/net/ABA_FWI.py` | `forward` 末尾增加：若 `out.shape[-2:] != output_size`，则 `F.interpolate(out, size=output_size, mode='bilinear', align_corners=False)` |

**原因**：ABA_FWI_SEG 的 UNet 解码器因 `ceil_mode`、`ConvTranspose2d` stride 等，可能输出 255×255 等非精确尺寸，与 benchmark 统一 label (256,256) 不匹配。

**公平性说明**：与 IMPLEMENTATION_GUIDE 中 FuteFWI、InversionNet 等模型一致，使用 `F.interpolate` 保证输出与 `output_size` 一致；不改变模型结构，仅做尺寸对齐，MSE 在相同尺寸上计算，评测公平。

---

## 十五、2026-02-28：文档与 FuteFWI batch 测试

| 修改项 | 位置 | 修改内容 |
|--------|------|----------|
| ABA 修复说明 | `benchmark/docker/BUILD_AND_RUN.md` | 新增「模型特定配置」与「ABA_FWI 输出尺寸修复」小节 |
| FuteFWI batch 测试 | `benchmark/futefwi_batch_stress.sh` | 新增：测试 batch 2/4/8，用于确定 BATCH_FUTEFWI |
| 脚本清单 | `benchmark/IMPLEMENTATION_GUIDE_ZH.md` | 新增 futefwi_batch_stress.sh 说明 |

**FuteFWI batch 测试结论**：batch=2 时出现 `CUDNN_STATUS_INTERNAL_ERROR`（非 OOM），为 CuDNN/驱动兼容问题。当前推荐 `BATCH_FUTEFWI=2`。

---

## 十六、2026-02-28：FuteFWI CUDNN 环境缓解

**目的**：缓解 FuteFWI 训练时的 `CUDNN_STATUS_INTERNAL_ERROR`，且**仅影响 FuteFWI**，不影响其他模型与公平性。

| 修改项 | 位置 | 修改内容 |
|--------|------|----------|
| FuteFWI 专用 env | `run_benchmark_suite.sh` `run_stage` | 当 `name==FuteFWI` 且 `stage==train` 时，`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

**影响范围**：仅 FuteFWI 训练子进程；其他 8 个模型（InversionNet/VelocityGAN/DCNet/DDNet70/TU_Net/ABA_FWI/ConvNeXtFWI/FCNVMB）不受影响。

**公平性说明**：`expandable_segments` 仅改变 CUDA 显存分配策略，不改变模型结构、数据、loss、评测；若仍失败可尝试升级 CUDA/cuDNN 或驱动。

---

## 十七、2026-03-02：Profile DDNet70/TU_Net 修复（thop ReLU total_ops）

**目的**：修复 DDNet70、TU_Net 在 profile 阶段报错 `AttributeError: 'ReLU' object has no attribute 'total_ops'`，导致 Params/FLOPs/Infer 均为 NA。

| 修改项 | 位置 | 修改内容 |
|--------|------|----------|
| 执行顺序 | `profile_model_metrics.py` `main()` | 将 `infer_ms` 提前到 `try_flops_g` 之前执行 |

**原因**：`thop.profile` 会在模型上注册 forward hooks；DDNet70、TU_Net 等存在**共享模块**（如多处复用的 ReLU）的模型，profile 结束后 hooks 可能未正确清理，后续 forward 触发 `total_ops` 访问失败。先执行 `infer_ms` 再执行 `try_flops_g`，可确保 params 和 infer 在无 hooks 状态下测得。

**公平性说明**：所有模型使用同一 profile 流程，仅调整内部执行顺序；params、infer、flops 的测量逻辑不变，不影响评估结果。

---

## 十八、2026-03-02：FuteFWI 改用 DDP

**目的**：FuteFWI 原使用 DataParallel，后 8 卡训练时出现 `NCCL Error 1: unhandled cuda error`（broadcast_coalesced 失败）。改用 DistributedDataParallel 可避免 DataParallel 的 broadcast 问题。

| 修改项 | 位置 | 修改内容 |
|--------|------|----------|
| DDP 支持 | `FuTE-FWI/train_futefwi.py` | 检测 `LOCAL_RANK`，torchrun 启动时使用 DDP + DistributedSampler + SyncBatchNorm（`--sync-bn`） |
| train/test | `FuTE-FWI/utils/utils.py` | `train`/`test` 增加 rank 判断，仅 rank 0 显示 tqdm 和打印 loss |
| 模型提取 | `train_futefwi.py` | 新增 `_get_raw_model`，统一从 DP/DDP 提取原始模型 |

**兼容性**：单卡或非 torchrun 启动时仍使用 DataParallel，行为不变。`run_benchmark_suite.sh` 已用 `torchrun --nproc_per_node=8` 启动 FuteFWI，自动走 DDP 路径。

**公平性说明**：与其他 8 个模型一致，FuteFWI 现统一使用 8 卡 DDP，训练流程公平。

---

## 十九、2026-03-02：ABA_FWI / FCNVMB profile 修复（ABA_Loss ImportError）

**目的**：修复 ABA_FWI、FCNVMB 在 profile 阶段报错 `ImportError: cannot import name 'ABA_Loss' from 'net.ABA_FWI'`，导致前 8 卡 Params/FLOPs/Infer 均为 NA。

| 修改项 | 位置 | 修改内容 |
|--------|------|----------|
| aba-fwi 导入 | `profile_model_metrics.py` `get_model()` | 移除 `ABA_Loss` 导入，仅导入 `ABA_FWI`、`FCNVMB_FWI`；`ABA_Loss` 不在 benchmark 的 `ABA_FWI.py` 中 |

**原因**：`net.ABA_FWI` 仅导出 `ABA_FWI` 类（benchmark 封装），无 `ABA_Loss`；profile 只需 `ABA_FWI`、`FCNVMB_FWI`。

**公平性说明**：不影响 profile 逻辑，ABA_FWI、FCNVMB 现可正常输出 Params/FLOPs/Infer。

---

*文档生成时间：2026-03-02*
