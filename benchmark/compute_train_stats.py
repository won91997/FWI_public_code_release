#!/usr/bin/env python3
"""
根据 global_map.csv 中 train 集样本计算归一化统计量，生成 train_stats.json。

UnifiedFWIDataset 使用该文件进行 seismic 和 vmodel 的归一化。
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Compute train-only stats for UnifiedFWIDataset")
    p.add_argument("--global-map-csv", required=True, help="global_map.csv from make_split_from_inline.py")
    p.add_argument("--out-json", default=None, help="output path (default: same dir as csv, train_stats.json)")
    p.add_argument("--sample-limit", type=int, default=0, help="max samples to use (0=all, 文档要求全量；>0 仅用于快速测试)")
    p.add_argument("--percentile", type=float, default=99, help="percentile for robust_max (default 99)")
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.global_map_csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"global_map.csv not found: {csv_path}")

    out_path = args.out_json
    if out_path is None:
        out_path = csv_path.parent / "train_stats.json"
    else:
        out_path = Path(out_path)

    # 收集 train 集行
    train_rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row.get("split", -1)) == 0:
                train_rows.append(row)

    if not train_rows:
        raise RuntimeError("No train samples in global_map.csv")

    # 文档要求：必须对全量训练集运行。sample_limit=0 表示全量；>0 仅用于快速测试
    sample_limit = len(train_rows) if args.sample_limit <= 0 else min(args.sample_limit, len(train_rows))
    if len(train_rows) > sample_limit:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(train_rows), size=sample_limit, replace=False)
        train_rows = [train_rows[i] for i in np.sort(idx)]
    print(f"Using {len(train_rows)} samples for stats")

    # 逐样本计算 percentile，避免 OOM
    pct = args.percentile
    robust_max_per_sample = []
    vmodel_min = []
    vmodel_max = []

    for i, row in enumerate(train_rows):
        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(train_rows)} samples...")
        seismic_path = row["seismic_path"]
        vmodel_path = row["vmodel_path"]
        in_file_idx = int(row["in_file_idx"])
        seismic = np.load(seismic_path, mmap_mode="r")[in_file_idx]
        vmodel = np.load(vmodel_path, mmap_mode="r")[in_file_idx]
        robust_max_per_sample.append(float(np.percentile(np.abs(seismic), pct)))
        vmodel_min.append(float(vmodel.min()))
        vmodel_max.append(float(vmodel.max()))

    # 取各样本 p99 的 95 分位作为 robust_max，避免极端值
    robust_max = float(np.percentile(robust_max_per_sample, 95))
    if robust_max <= 0:
        robust_max = float(np.percentile(robust_max_per_sample, 99.9)) or 1.0

    vmin = min(vmodel_min)
    vmax = max(vmodel_max)
    # 使用略微扩展的范围，避免边界
    margin = (vmax - vmin) * 0.02 or 1.0
    vmin = max(0, vmin - margin)
    vmax = vmax + margin

    stats = {
        "seismic": {"robust_max": robust_max},
        "vmodel": {"vmin": float(vmin), "vmax": float(vmax)},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"train_stats.json written to {out_path}")
    print(f"  seismic.robust_max = {robust_max:.4f}")
    print(f"  vmodel.vmin = {vmin:.4f}, vmodel.vmax = {vmax:.4f}")


if __name__ == "__main__":
    main()
