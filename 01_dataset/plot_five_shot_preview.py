#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""五炮预览：横排 5 子图；横轴=道，纵轴=时间向下为正（炮集惯例）。"""
import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 坐标轴与刻度：Times New Roman（无该字体时按列表回退）
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman",
            "Nimbus Roman No9 L",
            "DejaVu Serif",
        ],
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)


def main():
    data_root = os.environ.get("DATA_ROOT", "")
    p = argparse.ArgumentParser()
    p.add_argument(
        "--npy",
        default=os.path.join(data_root, "seismic_full", "seismic2.npy") if data_root else "",
        help="seismic_full/*.npy 路径",
    )
    p.add_argument(
        "--index",
        type=int,
        default=45,
        help="样本下标（默认 45：全库扫描五炮最少仍有 255 列有道，中间炮 256 列满）",
    )
    p.add_argument(
        "--out",
        default="five_shot_preview.png",
        help="输出 PNG",
    )
    p.add_argument(
        "--clip-percentile",
        type=float,
        default=95.0,
        metavar="P",
        help="对称色标用 percentile(|振幅|, P) 作为 vmax；P 越小对比越强(更艳)、越容易饱和。默认 95。",
    )
    args = p.parse_args()

    if not os.path.isfile(args.npy):
        print("missing:", args.npy or "<empty --npy; set DATA_ROOT or pass --npy>", file=sys.stderr)
        sys.exit(1)

    d = np.load(args.npy, mmap_mode="r")
    s = np.asarray(d[args.index], dtype=np.float32)
    nch, nt, nx = s.shape
    if nch != 5:
        print("warn: expected 5 channels, got", nch, file=sys.stderr)
    dt = 0.002
    tmax = nt * dt

    pclip = max(50.0, min(99.9, float(args.clip_percentile)))
    v = float(np.percentile(np.abs(s), pclip)) or 1.0

    # 横向 1×5；横纵坐标轴标签 + Times New Roman（经 rcParams）
    fig, axes = plt.subplots(
        1,
        5,
        figsize=(16.0, 5.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for i, ax in enumerate(axes):
        ax.imshow(
            s[i],
            aspect="auto",
            cmap="seismic",
            vmin=-v,
            vmax=v,
            extent=(0.5, float(nx) + 0.5, tmax, 0.0),
            origin="upper",
            interpolation="bilinear",
        )
        if i > 0:
            ax.tick_params(axis="y", labelleft=False)
        if nx > 1:
            ax.set_xticks(np.unique(np.linspace(1, nx, num=min(5, nx), dtype=int)))
        ax.set_yticks([0.0, 2.0, 4.0, tmax])
    axes[0].set_ylabel("Time (s)")
    axes[2].set_xlabel("Trace number")
    fig.savefig(args.out, dpi=140, bbox_inches="tight", pad_inches=0.02)
    print("saved", args.out)


if __name__ == "__main__":
    main()
