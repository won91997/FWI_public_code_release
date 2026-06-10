# Benchmark 输出参数说明

## 输出文件

| 文件 | 说明 |
|------|------|
| `benchmark_logs/benchmark_metrics.csv` | 每个模型×每个 seed 的原始结果 |
| `benchmark_logs/benchmark_metrics_agg.csv` | 按模型聚合（均值±标准差） |
| `benchmark_logs/summary.log` | 运行摘要 |

## CSV 列说明

| 列 | 来源 | 说明 |
|----|------|------|
| **Model** | 任务名 | 如 InversionNet#seed42 |
| **Model Category** | 任务定义 | CNN/GAN/Transformer/UNet 等 |
| **Params(M)** | profile | 参数量（百万），来自 `profile_model_metrics.py` |
| **FLOPs(G)** | profile | 浮点运算量（十亿），来自 thop.profile |
| **Train(h)** | 脚本计时 | 训练耗时（小时） |
| **Infer(ms)** | profile | 单次推理耗时（毫秒） |
| **MSE** | benchmark_eval | 均方误差 |
| **MAE** | benchmark_eval | 平均绝对误差 |
| **PSNR** | benchmark_eval | 峰值信噪比 |
| **SSIM** | benchmark_eval | 结构相似度 |
| **L1-Grad** | benchmark_eval | 梯度 L1 差 |
| **LPIPS** | benchmark_eval | 感知损失（proxy 模式） |
| **Status** | 脚本 | OK / FAIL(train) / FAIL(eval) |

## Profile 输出格式

`profile_model_metrics.py` 必须打印一行：
```
METRICS params_m=<float> flops_g=<float> infer_ms=<float>
```

脚本通过 `parse_profile_metric` 从 `{Model}.log` 解析上述值；若 profile 失败则为 `NA`。

## 已修复问题

### 2026-02-27

- **InversionNet / VelocityGAN profile 失败**：`profile_model_metrics.py` 未将 OpenFWI、FuTE-FWI 加入 `sys.path`，导致 `import network` / `from models import` 失败。
- **修复**：`_ensure_repo_path(repo)` 统一为所有 repo 添加对应路径，profile 可正常输出 Params/FLOPs/Infer。

### 2026-03-02

- **DDNet70 / TU_Net profile 失败**：`thop.profile` 注册的 hooks 在共享模块（如 ReLU）上导致 `AttributeError: 'ReLU' object has no attribute 'total_ops'`，Params/FLOPs/Infer 均为 NA。
- **修复**：`profile_model_metrics.py` 中将 `infer_ms` 提前到 `try_flops_g` 之前执行，确保 params 和 infer 在无 thop hooks 状态下测得；flops 若 thop 失败仍为 NA，但不影响 params/infer。
- **ABA_FWI / FCNVMB profile 失败**：`from net.ABA_FWI import ABA_FWI, ABA_Loss` 报错，`ABA_FWI.py` 无 `ABA_Loss`，前 8 卡 Params/FLOPs/Infer 均为 NA。
- **修复**：`profile_model_metrics.py` 中 aba-fwi 分支仅导入 `ABA_FWI`、`FCNVMB_FWI`，移除 `ABA_Loss`。

## 聚合脚本

`aggregate_seeds.py` 可将同一模型的多个 seed 聚合为 `mean±std`（论文主实验使用固定 seed 42，此脚本为可选工具），输出到 `benchmark_metrics_agg.csv`。

## 指标格式：至少 4 位有效数字

**要求**：所有数值指标（Params、FLOPs、Train、Infer、MSE、MAE、PSNR、SSIM、L1-Grad、LPIPS）至少保留 4 位有效数字，便于区分模型差异。

**实现**：
- `benchmark/benchmark_utils.py`：`fmt_sig(x, n=4)` 统一格式化
- `benchmark_eval.py`：打印与 JSON 输出均使用 `fmt_sig`
- `aggregate_seeds.py`：聚合输出使用 `fmt_sig`
- `profile_model_metrics.py`：使用 `%.6g`（≥4 位有效数字）
- `run_benchmark_suite.sh`：`train_h` 使用 `%.6g`
- `generate_benchmark_summary.py`：从 CSV 生成 `BENCHMARK_RESULTS_SUMMARY.md` 时使用 `fmt_sig`
