#!/usr/bin/env python3
"""
统一基准测试脚本：加载 checkpoint，推理，导出 pred.npy/gt.npy，调用 benchmark_eval。
"""
from __future__ import print_function

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

# 确保 benchmark 和 repository 在 path
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BENCH_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

from benchmark.unified_loader import UnifiedFWIDataset
from benchmark.unified_benchmark_train import get_model_for_benchmark, _forward, MODEL_MAP


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--global-map-csv", required=True)
    p.add_argument("--stats-json", required=True)
    p.add_argument("--align-multiple", type=int, default=32)
    p.add_argument("--align-mode", default="crop")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--export-dir", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"], help="Which split to run inference on.")
    p.add_argument("--max-batches", type=int, default=None, help="Limit number of batches (e.g. 1 for one-sample export).")
    p.add_argument("--benchmark-eval", default=None)
    p.add_argument("--benchmark-eval-json", default=None)
    p.add_argument("--benchmark-eval-lpips-mode", default="real", choices=["real", "proxy"])
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    repo, model_name = MODEL_MAP[args.model]
    output_size = (256, 256)
    model = get_model_for_benchmark(args.model, output_size)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    elif "model" in ckpt:
        # OpenFWI 格式
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        # FuTE-FWI .pt 等纯 state_dict
        model.load_state_dict(ckpt, strict=True)
    model = model.to(device)
    model.eval()

    dataset = UnifiedFWIDataset(
        data_root=args.data_root,
        split=args.split,
        global_map_csv=args.global_map_csv,
        stats_json=args.stats_json,
        align_multiple=args.align_multiple,
        align_mode=args.align_mode,
        output_channel_dim=True,
        need_edge=False,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    with open(args.stats_json, "r", encoding="utf-8") as f:
        stats = json.load(f)
    # train_stats.json 使用 vmodel.vmin, vmodel.vmax（与 unified_loader 一致）
    vmodel_stats = stats.get("vmodel", {})
    label_min = float(vmodel_stats.get("vmin", 2000))
    label_max = float(vmodel_stats.get("vmax", 6000))

    pred_list, gt_list = [], []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if args.max_batches is not None and bi >= args.max_batches:
                break
            if len(batch) == 3:
                data, label, _ = batch
            else:
                data, label = batch
            data = data.to(device, non_blocking=True)
            pred = _forward(model, data, repo, model_name, output_size)
            if isinstance(pred, list):
                pred = pred[0]
            pred_np = pred.cpu().numpy()
            label_np = label.numpy()
            # 反归一化到物理域 (benchmark_eval 期望物理域)
            pred_np = (pred_np + 1) * 0.5 * (label_max - label_min) + label_min
            label_np = (label_np + 1) * 0.5 * (label_max - label_min) + label_min
            pred_list.append(pred_np[:, 0])
            gt_list.append(label_np[:, 0])

    pred_all = np.concatenate(pred_list, axis=0)
    gt_all = np.concatenate(gt_list, axis=0)

    # When split is train/val, export to export_dir/train/ or export_dir/val/ so test stays at export_dir/
    if args.split in ("train", "val"):
        out_dir = os.path.join(args.export_dir, args.split)
    else:
        out_dir = args.export_dir
    os.makedirs(out_dir, exist_ok=True)
    pred_path = os.path.join(out_dir, "pred.npy")
    gt_path = os.path.join(out_dir, "gt.npy")
    np.save(pred_path, pred_all)
    np.save(gt_path, gt_all)
    print("Exported: {} {} (split={})".format(pred_path, gt_path, args.split))

    if args.split == "test" and args.benchmark_eval and os.path.isfile(args.benchmark_eval):
        cmd = [
            sys.executable,
            args.benchmark_eval,
            "--pred", pred_path,
            "--gt", gt_path,
            "--lpips-mode", args.benchmark_eval_lpips_mode,
        ]
        if args.benchmark_eval_json:
            cmd.extend(["--out-json", args.benchmark_eval_json])
        subprocess.check_call(cmd)
        if args.benchmark_eval_json:
            print("Metrics: {}".format(args.benchmark_eval_json))


if __name__ == "__main__":
    main()
