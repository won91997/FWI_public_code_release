#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate pred vs GT visualization from eval pred.npy/gt.npy (same format as unified_benchmark_train)."""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True, help="path to pred.npy")
    p.add_argument("--gt", required=True, help="path to gt.npy")
    p.add_argument("--out", required=True, help="output path, e.g. epoch_0100.png")
    p.add_argument("--sample", type=int, default=0, help="sample index (default 0)")
    args = p.parse_args()

    pred = np.load(args.pred)
    gt = np.load(args.gt)
    p = np.squeeze(pred[args.sample, 0]) if pred.ndim >= 4 else np.squeeze(pred[args.sample])
    g = np.squeeze(gt[args.sample, 0]) if gt.ndim >= 4 else np.squeeze(gt[args.sample])
    vmin = min(float(p.min()), float(g.min()))
    vmax = max(float(p.max()), float(g.max()))
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(p, cmap="jet", vmin=vmin, vmax=vmax)
    axs[0].set_title("Prediction")
    axs[0].axis("off")
    axs[1].imshow(g, cmap="jet", vmin=vmin, vmax=vmax)
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
