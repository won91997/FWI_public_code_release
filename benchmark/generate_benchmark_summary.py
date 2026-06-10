#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 benchmark_metrics_agg.csv 生成 BENCHMARK_RESULTS_SUMMARY.md，所有指标至少 4 位有效数字。
用法：
  python benchmark/generate_benchmark_summary.py \
    --logs benchmark_logs \
    --logs-gpu07 benchmark_logs_gpu0_7 \
    --out benchmark_logs/BENCHMARK_RESULTS_SUMMARY.md
"""
import argparse
import csv
import sys
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parent.parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))
from benchmark.benchmark_utils import fmt_sig


def parse_args():
    p = argparse.ArgumentParser(description="Generate BENCHMARK_RESULTS_SUMMARY.md from CSV")
    p.add_argument("--logs", default="benchmark_logs", help="后 8 卡日志目录")
    p.add_argument("--logs-gpu07", default="benchmark_logs_gpu0_7", help="前 8 卡日志目录")
    p.add_argument("--out", default=None, help="输出路径，默认 {logs}/BENCHMARK_RESULTS_SUMMARY.md")
    return p.parse_args()


def load_csv(path: Path):
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def fmt_cell(val: str) -> str:
    if not val or str(val).strip().upper() == "NA":
        return "NA"
    try:
        return fmt_sig(float(val))
    except (TypeError, ValueError):
        return str(val)


def main():
    args = parse_args()
    root = _BENCH_ROOT
    logs = root / args.logs
    logs_gpu07 = root / args.logs_gpu07
    out_path = Path(args.out) if args.out else logs / "BENCHMARK_RESULTS_SUMMARY.md"

    agg_path = logs / "benchmark_metrics_agg.csv"
    agg_gpu07_path = logs_gpu07 / "benchmark_metrics_agg.csv"

    rows_back = load_csv(agg_path)
    rows_front = load_csv(agg_gpu07_path)

    cols = ["Params(M)", "FLOPs(G)", "Train(h)", "Infer(ms)", "MSE", "MAE", "PSNR", "SSIM", "L1-Grad", "LPIPS"]

    def row_to_cells(r):
        return [fmt_cell(r.get(c, "")) for c in cols]

    def build_table(rows, filter_ok=True):
        lines = []
        header = "| 模型 | 类别 | " + " | ".join(cols) + " | 状态 |"
        sep = "|------|------|" + "|".join(["----------"] * len(cols)) + "|------|"
        lines.append(header)
        lines.append(sep)
        for r in rows:
            status = "OK" if filter_ok else "—"
            cells = row_to_cells(r)
            model = r.get("Model", "")
            cat = r.get("Model Category", "")
            line = f"| {model} | {cat} | " + " | ".join(cells) + f" | {status} |"
            lines.append(line)
        return "\n".join(lines)

    # 后 8 卡：只显示有 eval 结果的（非 NA 的 MSE）
    back_ok = [r for r in rows_back if r.get("MSE", "NA") not in ("", "NA")]
    back_fail = [r for r in rows_back if r.get("MSE", "NA") in ("", "NA")]

    # 前 8 卡
    front_ok = [r for r in rows_front if r.get("MSE", "NA") not in ("", "NA")]

    # 失败模型：MSE=NA 的（有 eval 结果的算成功）
    fail_reasons = {
        "FuteFWI": "FAIL(train) - 训练阶段报错",
        "VelocityGAN": "FAIL(train) - 训练阶段报错",
        "FCNVMB": "255 vs 256 shape 不匹配 → **已修复** (F.interpolate)",
    }
    failed = []
    for r in rows_back + rows_front:
        m = r.get("Model", "")
        if m and r.get("MSE", "NA") in ("", "NA"):
            failed.append(m)

    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")

    md = f"""# Benchmark 训练结果总结

**统计时间**: {today}

---

## 一、后 8 张卡 (benchmark_logs/) 已完成模型

{build_table(back_ok)}

**质量排名 (PSNR 从高到低)**:
"""
    # 按 PSNR 排序
    sorted_ok = sorted(back_ok, key=lambda r: (float(r.get("PSNR") or -1)), reverse=True)
    for i, r in enumerate(sorted_ok, 1):
        psnr = fmt_cell(r.get("PSNR", ""))
        ssim = fmt_cell(r.get("SSIM", ""))
        mse = fmt_cell(r.get("MSE", ""))
        md += f"{i}. {r.get('Model', '')}: PSNR {psnr}, SSIM {ssim}, MSE {mse}\n"

    md += f"""
---

## 二、前 8 张卡 (benchmark_logs_gpu0_7/) 已完成模型

{build_table(front_ok) if front_ok else "（暂无）"}

---

## 三、训练失败模型

| 模型 | 失败原因 |
|------|----------|
"""
    for m in sorted(set(failed)):
        md += f"| {m} | {fail_reasons.get(m, 'FAIL')} |\n"

    md += """
---

## 四、进行中

（可从日志手动更新）

---

## 五、配置

- Batch: 2/卡 × 8 卡 = 16
- Epochs: 100
- Seed: 42
- 数据: OpenFWI 256, ~8919 训练样本
- **指标格式**: 至少 4 位有效数字，便于区分模型差异
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
