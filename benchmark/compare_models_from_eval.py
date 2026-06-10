#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load pred.npy from 7 selected models' eval results, pick same sample, generate comparison figure.
No torch needed - uses existing eval outputs.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 7 selected models in PSNR rank order, (name, pred_path, gt_path)
MODELS_EVAL = [
    ("DCNet", "benchmark_logs/dcnet_seed42/eval/pred.npy", "benchmark_logs/dcnet_seed42/eval/gt.npy"),
    ("TU_Net", "benchmark_logs/tu_net_seed42/eval/pred.npy", "benchmark_logs/tu_net_seed42/eval/gt.npy"),
    ("VelocityGAN", "benchmark_logs/velocitygan_seed42/eval_results/pred.npy", "benchmark_logs/velocitygan_seed42/eval_results/gt.npy"),
    ("InversionNet", "benchmark_logs/inversionnet_seed42/eval/pred.npy", "benchmark_logs/inversionnet_seed42/eval/gt.npy"),
    ("DDNet70", "benchmark_logs/ddnet70_seed42/eval/pred.npy", "benchmark_logs/ddnet70_seed42/eval/gt.npy"),
    ("ConvNeXtKaggle", "logs_fair/convnext_kaggle_fair_seed42/eval/pred.npy", "logs_fair/convnext_kaggle_fair_seed42/eval/gt.npy"),
    ("ABA_FWI", "benchmark_logs/aba_fwi_seed42/eval/pred.npy", "benchmark_logs/aba_fwi_seed42/eval/gt.npy"),
]


def main():
    sample_idx = int(os.environ.get("SAMPLE_IDX", "0"))
    out_path = os.environ.get("OUT_PATH", os.path.join(ROOT, "benchmark_logs/compare_7models_sample.png"))

    gt_np = None
    all_imgs = []
    all_titles = []

    for name, pred_path, gt_path in MODELS_EVAL:
        pred_full = os.path.join(ROOT, pred_path)
        gt_full = os.path.join(ROOT, gt_path)
        if not os.path.isfile(pred_full):
            print("Skip {} (not found: {})".format(name, pred_full))
            continue
        pred = np.load(pred_full)
        gt = np.load(gt_full)
        if gt_np is None:
            gt_np = gt
        n = pred.shape[0]
        if sample_idx >= n:
            sample_idx = n // 2
        p = np.squeeze(pred[sample_idx])
        if p.ndim == 3:
            p = p[0]
        all_imgs.append(p)
        all_titles.append(name)

    if gt_np is None:
        print("No eval data found")
        return
    g = np.squeeze(gt_np[sample_idx])
    if g.ndim == 3:
        g = g[0]
    all_imgs = [g] + all_imgs
    all_titles = ["Ground Truth"] + all_titles

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
    plt.suptitle("7 Models on Same Representative Sample (idx={})".format(sample_idx), fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved: {}".format(out_path))


if __name__ == "__main__":
    main()
