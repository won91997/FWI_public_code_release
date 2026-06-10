#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7 selected models predict on the same representative test sample, generate comparison figure.
"""
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BENCH_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

from benchmark.unified_loader import UnifiedFWIDataset
from benchmark.unified_benchmark_train import _get_model, _forward, MODEL_MAP

# 7 completed models (PSNR rank order) + checkpoint paths
MODELS_CONFIG = [
    ("DCNet", "benchmark_logs/dcnet_seed42/checkpoint.pth", "model_state_dict"),
    ("TU_Net", "benchmark_logs/tu_net_seed42/checkpoint.pth", "model_state_dict"),
    ("VelocityGAN", "benchmark_logs/velocitygan_seed42/VelocityGAN_FlatVel_A_D.pt", "state_dict"),
    ("InversionNet", "benchmark_logs/inversionnet_seed42/checkpoint.pth", "model"),
    ("DDNet70", "benchmark_logs/ddnet70_seed42/checkpoint.pth", "model_state_dict"),
    ("ConvNeXtKaggle", "logs_fair/convnext_kaggle_fair_seed42/checkpoint.pth", "model_state_dict"),
    ("ABA_FWI", "benchmark_logs/aba_fwi_seed42/checkpoint.pth", "model_state_dict"),
]


def load_model_and_predict(model_name, ckpt_path, ckpt_format, data, device, stats):
    repo, mname = MODEL_MAP[model_name]
    output_size = (256, 256)
    model = _get_model(repo, mname, output_size)
    full_path = os.path.join(ROOT, ckpt_path)
    ckpt = torch.load(full_path, map_location="cpu")
    if ckpt_format == "model_state_dict":
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    elif ckpt_format == "model":
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model = model.to(device).eval()
    with torch.no_grad():
        x = data.to(device, non_blocking=True)
        pred = _forward(model, x, repo, mname, output_size)
        if isinstance(pred, list):
            pred = pred[-1]
        pred = pred.cpu().numpy()
    return pred


def main():
    data_root = os.environ.get("DATA_ROOT")
    global_map = os.path.join(ROOT, "benchmark/generated_split/global_map.csv")
    stats_json = os.path.join(ROOT, "benchmark/generated_split/train_stats.json")
    sample_idx = int(os.environ.get("SAMPLE_IDX", "0"))
    out_path = os.environ.get("OUT_PATH", os.path.join(ROOT, "benchmark_logs/compare_7models_sample.png"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(stats_json, "r", encoding="utf-8") as f:
        stats = json.load(f)
    vmin = float(stats.get("vmodel", {}).get("vmin", 2000))
    vmax = float(stats.get("vmodel", {}).get("vmax", 6000))

    test_set = UnifiedFWIDataset(
        data_root=data_root,
        split="test",
        global_map_csv=global_map,
        stats_json=stats_json,
        align_multiple=32,
        align_mode="crop",
        output_channel_dim=True,
        need_edge=False,
    )
    if sample_idx >= len(test_set):
        sample_idx = len(test_set) // 2
    data, label = test_set[sample_idx]
    data = data.unsqueeze(0)
    label = label.unsqueeze(0)
    gt_np = label.numpy()
    gt_phys = (gt_np + 1) * 0.5 * (vmax - vmin) + vmin

    preds = {}
    for model_name, ckpt_path, ckpt_format in MODELS_CONFIG:
        full_path = os.path.join(ROOT, ckpt_path)
        if not os.path.isfile(full_path):
            print("Skip {} (checkpoint not found: {})".format(model_name, full_path))
            continue
        pred = load_model_and_predict(model_name, ckpt_path, ckpt_format, data, device, stats)
        pred_phys = (pred + 1) * 0.5 * (vmax - vmin) + vmin
        preds[model_name] = np.squeeze(pred_phys[0, 0])

    gt_2d = np.squeeze(gt_phys[0, 0])
    all_imgs = [gt_2d] + [preds[m] for m, _, _ in MODELS_CONFIG if m in preds]
    all_titles = ["Ground Truth"] + [m for m, _, _ in MODELS_CONFIG if m in preds]
    vmin_plot = min(float(img.min()) for img in all_imgs)
    vmax_plot = max(float(img.max()) for img in all_imgs)

    n = len(all_imgs)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axs = np.atleast_2d(axs)
    for i, (img, title) in enumerate(zip(all_imgs, all_titles)):
        r, c = i // ncols, i % ncols
        axs[r, c].imshow(img, cmap="jet", vmin=vmin_plot, vmax=vmax_plot)
        axs[r, c].set_title(title, fontsize=12)
        axs[r, c].axis("off")
    for j in range(i + 1, nrows * ncols):
        r, c = j // ncols, j % ncols
        axs[r, c].axis("off")
    plt.suptitle("7 Models on Same Test Sample (sample_idx={})".format(sample_idx), fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved: {}".format(out_path))


if __name__ == "__main__":
    main()
