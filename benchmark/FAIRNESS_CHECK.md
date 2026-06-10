# Benchmark 公平性检查报告

## ✅ 已确认公平项

| 维度 | 配置 | 说明 |
|------|------|------|
| **数据** | 统一 | 所有模型使用同一 `global_map.csv`、`train_stats.json`、`align-multiple 32`、`align-mode crop` |
| **Batch** | 8/卡 | `BATCH_PER_GPU=8`，总 batch=64，全员统一 |
| **Epochs** | 100 | 除 InversionNet 用 `-eb -nb` 等效 100 轮外，其余均 `--epochs 100` |
| **GPU** | 8 卡 DDP | 所有训练统一 `torchrun --nproc_per_node=8` |
| **评测** | 6 项指标 | MSE、MAE、PSNR、SSIM、L1-Grad、LPIPS（proxy 模式） |
| **Seeds** | 1 个 | 42 |
| **Profile** | 统一 shape | `1,5,2976,256` |

## 📐 数据维度：3000 → 2976 是裁剪（crop）

**实现位置**：`benchmark/unified_loader.py` 的 `_align_shape()`

- **align-multiple 32**：时间维 `t = (t // 32) * 32`，3000 → 2976
- **align-mode crop**：`return arr[:, :t, :w].copy()`，取前 2976 个时间步
- **结论**：**裁剪**（丢弃末尾 24 个时间步），不是填充。所有模型输入统一为 `(N, 5, 2976, 256)`。

## 🔒 公平性代码保证（run_benchmark_suite.sh）

| 保证项 | 实现方式 |
|--------|----------|
| 数据参数 | 9 个 TASK 的 train_cmd、eval_cmd 均显式传入 `--align-multiple 32 --align-mode crop`、`--global-map-csv`、`--stats-json` |
| Batch 统一 | `BATCH_INVNET`、`BATCH_VELGAN`、`BATCH_FUTEFWI` 等均引用 `BATCH_PER_GPU` |
| 评测入口 | InversionNet/VelocityGAN/FuteFWI 均用 `UNIFIED_TEST`（unified_benchmark_test.py），同一 `benchmark_eval.py` |
| LPIPS | 所有 eval_cmd 显式 `--benchmark-eval-lpips-mode real` |
| Profile shape | 所有 profile_cmd 使用 `--shape 1,5,2976,256` |

## ⚠️ 架构差异（设计如此）

| 项目 | 说明 |
|------|------|
| **损失函数** | InversionNet/VelocityGAN/FuteFWI 用原生损失；DCNet/DDNet70 等用 MSE |
| **InversionNet** | 用 `-eb 5 -nb 20` 等效 100 epoch，与 `EPOCHS=100` 对齐 |
| **VelocityGAN** | 无 `--sync-bn`（与 WGAN-GP 不兼容） |

## 🔧 已修复阻塞项

### 2026-02-26

| 问题 | 处理 |
|------|------|
| OpenFWI 无 `test.py` | InversionNet 评测改用 `unified_benchmark_test.py` |
| FuTE-FWI `test.py` 无 benchmark 参数 | VelocityGAN/FuteFWI 评测改用 `unified_benchmark_test.py` |
| checkpoint 格式差异 | `unified_benchmark_test` 支持 OpenFWI `model` 键、FuTE-FWI 纯 state_dict |

### 2026-03-02

| 问题 | 处理 |
|------|------|
| FuteFWI NCCL Error 1（DataParallel broadcast 失败） | `train_futefwi.py` 改用 DDP，与其他 8 个模型一致 |
| DDNet70/TU_Net profile Params/FLOPs/Infer 为 NA | `profile_model_metrics.py` 将 `infer_ms` 提前到 `try_flops_g` 之前，避免 thop hooks 报错 |
| ABA_FWI/FCNVMB profile Params/FLOPs/Infer 为 NA | `profile_model_metrics.py` 移除 `ABA_Loss` 导入，aba-fwi 分支仅导入 `ABA_FWI`、`FCNVMB_FWI` |

## 📋 启动前最终确认

- [ ] `preflight_check.sh` 通过
- [ ] `pre_launch_checklist.sh`（VelocityGAN 点火）通过
- [ ] 数据路径 `<DATA_ROOT>` 可访问
- [ ] Docker 使用 `--shm-size=32g`

---

**结论**：公平性配置已就绪，阻塞项已修复，可启动全量测试。
