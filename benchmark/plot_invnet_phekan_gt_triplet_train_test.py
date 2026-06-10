#!/usr/bin/env python3
"""
训练集、测试集各取 1 个样本：真值 | InversionNet | PHE-KAN 三列对比图（2 行）。
物理速度 m/s；PHE-KAN 使用 per_sample_phys 反归一化（与 unified_phekan_eval 一致）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "benchmark")
PHEKAN = os.path.join(ROOT, "Resolution-Constrained", "PHE-KAN")
for p in (ROOT, BENCH, PHEKAN):
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmark.unified_benchmark_train import MODEL_MAP, _forward, get_model_for_benchmark
from benchmark.unified_loader import UnifiedFWIDataset, _align_shape
from net.KAN_initial6 import KANModel


def _rows_for_split(csv_path: str, split_name: str) -> list[dict]:
    m = {"train": 0, "val": 1, "test": 2}
    v = m[split_name]
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["split"]) == v:
                rows.append(row)
    return rows


def _aligned_vm_phys(row: dict, ds: UnifiedFWIDataset) -> np.ndarray:
    vm = np.asarray(np.load(row["vmodel_path"])[int(row["in_file_idx"])], dtype=np.float32)
    return _align_shape(
        vm,
        ds.align_multiple,
        ds.align_mode,
        0,
        ds.target_width,
        is_seismic=False,
    )


def _load_invnet(ckpt_path: str, device: torch.device):
    model = get_model_for_benchmark("InversionNet", (256, 256))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt
    if any(str(k).startswith("module.") for k in sd.keys()):
        sd = {str(k).replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def _load_phekan(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    if any(str(k).startswith("module.") for k in sd.keys()):
        sd = {str(k).replace("module.", "", 1): v for k, v in sd.items()}
    net = KANModel()
    net.load_state_dict(sd, strict=True)
    return net.to(device).eval()


def _load_initial_tensor(row: dict, cache_dir: str, device: torch.device) -> torch.Tensor:
    fname = f"f{int(row['file_idx']) + 1}_s{int(row['in_file_idx'])}.npy"
    path = os.path.join(cache_dir, fname)
    ini = np.load(path).astype(np.float32)
    return torch.from_numpy(ini).view(1, 1, 256, 256).float().to(device)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inversionnet-ckpt", required=True)
    p.add_argument("--phekan-ckpt", required=True)
    p.add_argument("--invnet-cache-dir", required=True)
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
    p.add_argument("--global-map-csv", required=True)
    p.add_argument("--stats-json", required=True)
    p.add_argument("--train-index", type=int, default=0)
    p.add_argument("--test-index", type=int, default=0)
    p.add_argument(
        "--extent-x",
        type=float,
        default=3200.0,
        help="横轴 Position 右端 (m)",
    )
    p.add_argument(
        "--extent-y",
        type=float,
        default=10000.0,
        help="纵轴 Depth 下端 (m)，地表在上方 y=0",
    )
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _predict_one(
    split: str,
    idx: int,
    rows: list[dict],
    device: torch.device,
    label_min: float,
    label_max: float,
    inv_model,
    phe_model,
    args,
):
    ds = UnifiedFWIDataset(
        data_root=args.data_root,
        split=split,
        global_map_csv=args.global_map_csv,
        stats_json=args.stats_json,
        align_multiple=32,
        align_mode="crop",
        output_channel_dim=True,
        need_edge=False,
    )
    if idx < 0 or idx >= len(ds):
        raise IndexError(f"{split} index {idx} out of range [0,{len(ds)})")
    row = rows[idx]
    seismic, _ = ds[idx]
    gt = _aligned_vm_phys(row, ds)

    with torch.no_grad():
        x = seismic.unsqueeze(0).to(device)
        repo, name = MODEL_MAP["InversionNet"]
        inv_out = _forward(inv_model, x, repo, name, (256, 256))
        if isinstance(inv_out, (list, tuple)):
            inv_out = inv_out[0]
        inv_np = inv_out.cpu().numpy()[0, 0]
        inv_phy = (inv_np + 1.0) * 0.5 * (label_max - label_min) + label_min

        init_t = _load_initial_tensor(row, args.invnet_cache_dir, device)
        phe_out = phe_model(x, init_t)
        if isinstance(phe_out, (list, tuple)):
            phe_out = phe_out[0]
        phe_np = phe_out.cpu().numpy()[0, 0]
        phe_np = np.clip(phe_np, 0.0, 1.0)
        vmn, vmx = float(gt.min()), float(gt.max())
        phe_phy = phe_np * (vmx - vmn) + vmn

    return gt, inv_phy, phe_phy


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(args.stats_json, "r", encoding="utf-8") as f:
        stats = json.load(f)
    vm = stats.get("vmodel", {})
    label_min = float(vm.get("vmin", 2000))
    label_max = float(vm.get("vmax", 6000))

    rows_train = _rows_for_split(args.global_map_csv, "train")
    rows_test = _rows_for_split(args.global_map_csv, "test")
    if not rows_train or not rows_test:
        raise SystemExit("train or test split empty in csv")

    inv_model = _load_invnet(args.inversionnet_ckpt, device)
    phe_model = _load_phekan(args.phekan_ckpt, device)

    pairs = [
        ("train", args.train_index, rows_train, f"Train sample #{args.train_index}"),
        ("test", args.test_index, rows_test, f"Test sample #{args.test_index}"),
    ]

    # 每个子图「面板」为正方形，256×256 格点在屏幕上等距为正方格；坐标轴仍为横 0–extent_x、纵 0–extent_y
    fig, axes = plt.subplots(2, 3, figsize=(14, 10), constrained_layout=True)
    # imshow extent: (left, right, bottom, top)；origin=upper 时 top=0 表示地表在上方，深度向下增大
    ex, ey = float(args.extent_x), float(args.extent_y)
    extent = [0.0, ex, ey, 0.0]

    for ri, (split, idx, rows, title_prefix) in enumerate(pairs):
        gt, inv_p, phe_p = _predict_one(
            split, idx, rows, device, label_min, label_max, inv_model, phe_model, args
        )
        vmin = float(min(gt.min(), inv_p.min(), phe_p.min()))
        vmax = float(max(gt.max(), inv_p.max(), phe_p.max()))
        for ci, (data, name) in enumerate(
            [(gt, "Ground truth"), (inv_p, "InversionNet"), (phe_p, "PHE-KAN")]
        ):
            ax = axes[ri, ci]
            im = ax.imshow(
                data,
                cmap="viridis",
                aspect="auto",
                extent=extent,
                vmin=vmin,
                vmax=vmax,
                origin="upper",
                interpolation="nearest",
            )
            ax.set_title(f"{title_prefix}\n{name}", fontsize=11)
            ax.set_xlabel("Position (m)")
            ax.set_ylabel("Depth (m)")
            ax.set_xlim(0.0, ex)
            ax.set_ylim(ey, 0.0)
            # 正方形画板：每个网格像在屏幕上为正方形（物理上横向 3200m、垂向 10000m 仅体现在刻度）
            ax.set_box_aspect(1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="m/s")

    fig.suptitle(
        "OpenFWI 256 | GT vs InversionNet vs PHE-KAN (PHE-KAN uses InvNet initial; velocity m/s)",
        fontsize=12,
    )
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close()
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
