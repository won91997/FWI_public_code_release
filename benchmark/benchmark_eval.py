#!/usr/bin/env python3
import argparse
import json
import math
import pathlib
import sys
from typing import Dict

# 确保可从任意 cwd 导入 benchmark 包（run_stage 会 cd 到 repo_dir）
_BENCH_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

import numpy as np

from benchmark.benchmark_utils import fmt_sig
import torch


def parse_args():
    p = argparse.ArgumentParser("Unified evaluator for benchmark")
    p.add_argument("--pred", required=True, help="predictions npy, shape (N,H,W) or (N,1,H,W)")
    p.add_argument("--gt", required=True, help="ground truth npy, shape (N,H,W) or (N,1,H,W)")
    p.add_argument("--out-json", default=None, help="optional output json path")
    p.add_argument("--lpips-mode", choices=["proxy", "real"], default="real", help="LPIPS mode")
    p.add_argument("--lpips-backbone", choices=["alex", "vgg"], default="alex")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return p.parse_args()


def squeeze_hw(x: np.ndarray) -> np.ndarray:
    if x.ndim == 4 and x.shape[1] == 1:
        return x[:, 0]
    if x.ndim == 3:
        return x
    raise ValueError(f"Unsupported shape: {x.shape}")


def mse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean((pred - gt) ** 2))


def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt)))


def gt_data_range_global(gt: np.ndarray) -> float:
    """PSNR: data_range over the entire test set (all samples, all pixels)."""
    dr = float(np.max(gt) - np.min(gt))
    return dr if dr > 0 else 1.0


def gt_data_range_per_sample(gt: np.ndarray) -> np.ndarray:
    """SSIM: per-sample data_range = max(GT) - min(GT), shape (N,)."""
    dr = np.max(gt, axis=(1, 2)) - np.min(gt, axis=(1, 2))
    return np.where(dr > 0, dr, 1.0)


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    m = mse(pred, gt)
    if m <= 0:
        return 99.0
    dr = gt_data_range_global(gt)
    return float(20.0 * math.log10(dr) - 10.0 * math.log10(m))


def ssim_global(pred: np.ndarray, gt: np.ndarray) -> float:
    # Global (non-windowed) SSIM; C1/C2 scaled by per-sample GT dynamic range,
    # matching skimage data_range = max(GT) - min(GT) per image (Wang et al.).
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    dr = gt_data_range_per_sample(gt)
    c1 = (0.01 * dr) ** 2
    c2 = (0.03 * dr) ** 2
    mu_x = pred.mean(axis=(1, 2))
    mu_y = gt.mean(axis=(1, 2))
    var_x = pred.var(axis=(1, 2))
    var_y = gt.var(axis=(1, 2))
    cov = ((pred - mu_x[:, None, None]) * (gt - mu_y[:, None, None])).mean(axis=(1, 2))
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    s = np.where(den == 0, 0.0, num / den)
    return float(np.mean(s))


def l1_grad(pred: np.ndarray, gt: np.ndarray) -> float:
    # Mean absolute difference of gradient magnitude (Sobel-like finite diff)
    dx_p = np.diff(pred, axis=2, append=pred[:, :, -1:])
    dy_p = np.diff(pred, axis=1, append=pred[:, -1:, :])
    dx_g = np.diff(gt, axis=2, append=gt[:, :, -1:])
    dy_g = np.diff(gt, axis=1, append=gt[:, -1:, :])
    gm_p = np.sqrt(dx_p**2 + dy_p**2)
    gm_g = np.sqrt(dx_g**2 + dy_g**2)
    return float(np.mean(np.abs(gm_p - gm_g)))


def lpips_proxy(pred: np.ndarray, gt: np.ndarray) -> float:
    # True LPIPS requires deep features; for unified no-extra-dependency
    # benchmark script, we provide a stable proxy.
    # You can replace with real LPIPS later by loading torch+lpips.
    x = pred.reshape(pred.shape[0], -1)
    y = gt.reshape(gt.shape[0], -1)
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    num = np.sum(x * y, axis=1)
    den = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-8
    cos = num / den
    return float(np.mean(1.0 - cos))


def lpips_real(pred: np.ndarray, gt: np.ndarray, backbone: str = "alex", device: str = "cuda") -> float:
    import lpips  # lazy import

    use_cuda = device == "cuda" and torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")
    metric = lpips.LPIPS(net=backbone, verbose=False).to(dev)

    pred_t = torch.from_numpy(pred).float().to(dev)
    gt_t = torch.from_numpy(gt).float().to(dev)
    # ensure shape (N, 1, H, W)
    pred_t = pred_t[:, None, :, :]
    gt_t = gt_t[:, None, :, :]
    # repeat to 3 channels as LPIPS expects RGB-like input
    pred_t = pred_t.repeat(1, 3, 1, 1)
    gt_t = gt_t.repeat(1, 3, 1, 1)

    # normalize each sample to [-1, 1] safely
    def _norm01(x):
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min + 1e-8)

    pred_t = _norm01(pred_t) * 2 - 1
    gt_t = _norm01(gt_t) * 2 - 1

    with torch.no_grad():
        val = metric(pred_t, gt_t).mean().item()
    return float(val)


def evaluate(pred: np.ndarray, gt: np.ndarray, lpips_mode: str, lpips_backbone: str, device: str) -> Dict[str, float]:
    if lpips_mode == "real":
        lp = lpips_real(pred, gt, backbone=lpips_backbone, device=device)
    else:
        lp = lpips_proxy(pred, gt)
    return {
        "MSE": mse(pred, gt),
        "MAE": mae(pred, gt),
        "PSNR": psnr(pred, gt),
        "SSIM": ssim_global(pred, gt),
        "L1-Grad": l1_grad(pred, gt),
        "LPIPS": lp,
    }


def main():
    args = parse_args()
    pred = squeeze_hw(np.load(args.pred))
    gt = squeeze_hw(np.load(args.gt))
    if pred.shape != gt.shape:
        raise RuntimeError(f"Shape mismatch pred={pred.shape}, gt={gt.shape}")

    metrics = evaluate(
        pred,
        gt,
        lpips_mode=args.lpips_mode,
        lpips_backbone=args.lpips_backbone,
        device=args.device,
    )
    # 至少 4 位有效数字，便于区分模型差异
    for k, v in metrics.items():
        print(f"{k}: {fmt_sig(v)}")

    if args.out_json:
        out = pathlib.Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        # JSON 中数值保留至少 4 位有效数字（与 print 一致）
        metrics_fmt = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and v is not None and math.isfinite(v):
                metrics_fmt[k] = float(fmt_sig(v))
            else:
                metrics_fmt[k] = v
        out.write_text(json.dumps(metrics_fmt, indent=2), encoding="utf-8")
        print(f"Saved json: {out}")


if __name__ == "__main__":
    main()

