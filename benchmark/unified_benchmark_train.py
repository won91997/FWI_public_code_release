#!/usr/bin/env python3
"""
统一基准训练脚本：支持 10 个唯一架构，每个模型使用其原版 Loss 函数。
公平基准：相同数据、相同预算、相同优化器，Loss 保留各模型原始设计。
"""
from __future__ import print_function

import argparse
import json
import os
import shutil
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import cv2
except ImportError:
    cv2 = None

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BENCH_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

from benchmark.unified_loader import UnifiedFWIDataset
from benchmark.fair_wrapper import FairWrapper, FAIR_MODELS_CONFIG


# ---------------------------------------------------------------------------
#  Edge extraction (from velocity model, using Canny — same as TU-Net/ABA-FWI)
# ---------------------------------------------------------------------------
def _extract_contours_np(vmodel_2d):
    """Canny edge detection on a single 2D velocity map -> binary {0,1}."""
    if cv2 is None:
        return np.zeros_like(vmodel_2d, dtype=np.float32)
    norm = cv2.normalize(vmodel_2d.astype(np.float32), None, 0, 1,
                         cv2.NORM_MINMAX, cv2.CV_32F)
    u8 = (norm * 255).astype(np.uint8)
    canny = cv2.Canny(u8, 10, 15)
    return np.clip(canny, 0, 1).astype(np.float32)


def _batch_edges(label_tensor):
    """label: (B, 1, H, W) tensor on CPU/GPU -> edges (B, 1, H, W) tensor."""
    label_np = label_tensor.detach().cpu().numpy()
    B = label_np.shape[0]
    edges = np.zeros_like(label_np, dtype=np.float32)
    for b in range(B):
        edges[b, 0] = _extract_contours_np(label_np[b, 0])
    return torch.from_numpy(edges).to(label_tensor.device)


# ---------------------------------------------------------------------------
#  Per-model native loss functions
# ---------------------------------------------------------------------------
def _dilate_edge(edges_tensor):
    """3x3 dilation on binary edge map."""
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=edges_tensor.device)
    dilated = F.conv2d(edges_tensor.float(), kernel, padding=1, stride=1)
    return (dilated > 0).float()


class FocalFrequencyLoss(nn.Module):
    """Focal Frequency Loss (ICCV 2021) — used by DCNet."""
    def __init__(self, loss_weight=1.0, alpha=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha

    def forward(self, pred, target):
        freq_p = torch.fft.fft2(pred, norm='ortho')
        freq_p = torch.stack([freq_p.real, freq_p.imag], -1)
        freq_t = torch.fft.fft2(target, norm='ortho')
        freq_t = torch.stack([freq_t.real, freq_t.imag], -1)
        diff = (freq_p - freq_t) ** 2
        dist = diff[..., 0] + diff[..., 1]
        weight = torch.sqrt(dist).detach() ** self.alpha
        w_max = weight.amax(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
        weight = (weight / w_max).clamp(0, 1)
        return torch.mean(weight * dist) * self.loss_weight


def _get_criterion(benchmark_key, device):
    """Return (criterion_fn, needs_dual_output: bool) for each model."""

    if benchmark_key == "ABA_FWI":
        l1 = nn.L1Loss()
        l2 = nn.MSELoss()
        def aba_loss(pred, label, edges=None):
            loss = l1(pred, label) + l2(pred, label)
            if edges is not None:
                px = pred[:, :, 1:, :] - pred[:, :, :-1, :]
                py = pred[:, :, :, 1:] - pred[:, :, :, :-1]
                lx = label[:, :, 1:, :] - label[:, :, :-1, :]
                ly = label[:, :, :, 1:] - label[:, :, :, :-1]
                tv_x = torch.zeros_like(pred)
                tv_y = torch.zeros_like(pred)
                tv_x[:, :, 1:, :] = torch.abs(px - lx)
                tv_y[:, :, :, 1:] = torch.abs(py - ly)
                tv = tv_x + tv_y
                ew = _dilate_edge(edges)
                denom = pred.size(0) * ew.sum().clamp(min=1.0)
                loss = loss + torch.sum(tv * ew) / denom
            return loss
        return aba_loss, False

    if benchmark_key == "DCNet":
        mse = nn.MSELoss()
        ffl = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0)
        def dcnet_loss(pred, label, edges=None):
            return mse(pred, label) + ffl(pred, label)
        return dcnet_loss, False

    if benchmark_key == "DDNet70":
        mse = nn.MSELoss()
        ce = nn.CrossEntropyLoss()
        def ddnet_loss(pred_list, label, edges=None):
            vel_pred = pred_list[0]
            edge_pred = pred_list[1]
            loss_mse = mse(vel_pred, label)
            if edges is not None:
                edge_target = torch.squeeze(edges, 1).long()
                loss_ce = ce(edge_pred, edge_target)
                return loss_mse + 1e6 * loss_ce
            return loss_mse
        return ddnet_loss, True

    if benchmark_key == "TU_Net":
        l1 = nn.L1Loss()
        l2 = nn.MSELoss()
        def tu_loss(pred, label, edges=None):
            loss_pixel = 0.5 * l1(pred, label) + 0.5 * l2(pred, label)
            if edges is not None:
                ew = _dilate_edge(edges)
                denom = label.size(0) * ew.sum().clamp(min=1.0)
                loss_edge = torch.sum(ew * torch.abs(pred - label)) / denom
                return 0.7 * loss_pixel + 0.3 * loss_edge
            return loss_pixel
        return tu_loss, False

    if benchmark_key == "FCNVMB":
        mse = nn.MSELoss()
        def fcnvmb_loss(pred, label, edges=None):
            return mse(pred, label)
        return fcnvmb_loss, False

    if benchmark_key == "ConvNeXtKaggle":
        mse = nn.MSELoss()
        def convnext_kaggle_loss(pred, label, edges=None):
            return mse(pred, label)
        return convnext_kaggle_loss, False

    if benchmark_key == "VIFNet":
        mse = nn.MSELoss()
        def vifnet_loss(pred_list, label, edges=None):
            refined = pred_list[0]
            stov = pred_list[1]
            stoc = pred_list[2]
            loss = mse(refined, label) + mse(stov, label)
            if edges is not None:
                with torch.cuda.amp.autocast(enabled=False):
                    stoc_f = stoc.float().clamp(1e-6, 1 - 1e-6)
                    e_flat = edges.float().view(edges.size(0), -1)
                    s_flat = stoc_f.view(stoc_f.size(0), -1)
                    count_neg = (1.0 - e_flat).sum(dim=1)
                    count_pos = e_flat.sum(dim=1)
                    beta = count_neg / (count_neg + count_pos).clamp(min=1.0)
                    pos_weight = (beta / (1 - beta).clamp(min=1e-6)).view(-1, 1)
                    w = torch.where(e_flat > 0.5, pos_weight, torch.ones_like(e_flat))
                    bce = F.binary_cross_entropy(s_flat, e_flat, weight=w, reduction='none')
                    loss_stoc = (bce.mean(dim=1) * (1 - beta)).mean()
                loss = loss + loss_stoc
            return loss
        return vifnet_loss, True

    mse = nn.MSELoss()
    def default_loss(pred, label, edges=None):
        return mse(pred, label)
    return default_loss, False


def _get_model(repo, model_name, output_size=(256, 256), benchmark_key=None):
    """复用 profile 的 get_model 逻辑。"""
    if repo == "openfwi":
        p = os.path.join(ROOT, "OpenFWI")
        if p not in sys.path:
            sys.path.insert(0, p)
        import network
        if model_name not in network.model_dict:
            raise ValueError("Unknown openfwi model: {}".format(model_name))
        return network.model_dict[model_name](output_size=output_size)
    if repo == "futefwi":
        p = os.path.join(ROOT, "FuTE-FWI")
        if p not in sys.path:
            sys.path.insert(0, p)
        from models import FuteFWI, InversionNet, Generator
        if model_name == "FuteFWI":
            return FuteFWI(output_size=output_size)
        if model_name == "InversionNet":
            return InversionNet(output_size=output_size)
        if model_name == "VelocityGAN":
            return Generator(output_size=output_size)
        raise ValueError("Unknown futefwi model: {}".format(model_name))
    if repo == "dcnet":
        p = os.path.join(ROOT, "DCNet")
        if p not in sys.path:
            sys.path.insert(0, p)
        from DCNet import DCModel
        from func.ddnet import DDNet70Model
        from func.comparison_net import InversionNet
        if model_name == "DCNet":
            return DCModel(output_size=output_size)
        if model_name == "DDNet70":
            return DDNet70Model(n_classes=1, in_channels=5, is_deconv=True, is_batchnorm=True, output_size=output_size)
        if model_name == "InversionNet":
            return InversionNet(output_size=output_size)
        raise ValueError("Unknown dcnet model: {}".format(model_name))
    if repo == "ddnet":
        p = os.path.join(ROOT, "ddnet")
        if p not in sys.path:
            sys.path.insert(0, p)
        from net.DDNet70 import DDNet70Model
        from net.InversionNet import InversionNet
        from net.FCNVMB import FCNVMB
        if model_name == "DDNet70":
            return DDNet70Model(n_classes=1, in_channels=5, is_deconv=True, is_batchnorm=True, output_size=output_size)
        if model_name == "InversionNet":
            return InversionNet(output_size=output_size)
        if model_name == "FCNVMB":
            return FCNVMB(n_classes=1, in_channels=5, is_deconv=True, is_batchnorm=True)
        raise ValueError("Unknown ddnet model: {}".format(model_name))
    if repo == "tu-net":
        p = os.path.join(ROOT, "TU-Net")
        if p not in sys.path:
            sys.path.insert(0, p)
        from model.TU_Net import TU_Net
        from model.DDNet70 import DDNet70Model
        from model.InversionNet import InversionNet
        from model.FCNVMB import FCNVMB
        if model_name == "TU_Net":
            return TU_Net(n_classes=1, in_channels=5, is_deconv=True, is_batchnorm=True, output_size=output_size)
        if model_name == "DDNet70":
            return DDNet70Model(n_classes=1, in_channels=5, is_deconv=True, is_batchnorm=True, output_size=output_size)
        if model_name == "InversionNet":
            return InversionNet(output_size=output_size)
        if model_name == "FCNVMB":
            return FCNVMB(n_classes=1, in_channels=5, is_deconv=True, is_batchnorm=True)
        raise ValueError("Unknown tu-net model: {}".format(model_name))
    if repo == "aba-fwi":
        aba_path = os.path.join(ROOT, "ABA-FWI", "ABA-FWI_2.0")
        if aba_path not in sys.path:
            sys.path.insert(0, aba_path)
        from net.ABA_FWI import ABA_FWI
        from net.FCNVMB import FCNVMB_FWI
        if model_name == "ABA_FWI":
            return ABA_FWI(output_size=output_size)
        if model_name == "FCNVMB_FWI":
            return FCNVMB_FWI(model_dim=output_size, in_channels=5)
        raise ValueError("Unknown aba-fwi model: {}".format(model_name))
    if repo == "convnext-kaggle":
        ck_path = os.path.join(ROOT, "ConvNeXt-Kaggle")
        if ck_path not in sys.path:
            sys.path.insert(0, ck_path)
        from kaggle_model import KaggleConvNeXtBaseline
        if model_name == "ConvNeXtKaggle":
            use_pretrained = os.environ.get("KAGGLE_CONVNEXT_PRETRAINED", "1") == "1"
            backbone = os.environ.get("KAGGLE_CONVNEXT_BACKBONE", "convnext_small.fb_in22k_ft_in1k")
            return KaggleConvNeXtBaseline(output_size=output_size, pretrained=use_pretrained, backbone=backbone)
        raise ValueError("Unknown convnext-kaggle model: {}".format(model_name))
    if repo == "vif-net":
        vif_path = os.path.join(ROOT, "VIF-Net")
        if vif_path not in sys.path:
            sys.path.insert(0, vif_path)
        from VIFNet import VIFNet
        if model_name == "VIFNet":
            return VIFNet(output_size=output_size, in_channels=5)
        raise ValueError("Unknown vif-net model: {}".format(model_name))
    raise ValueError("Unsupported repo: {}".format(repo))


class NonFairVIFAdapter(nn.Module):
    """非 Fair 模式下 VIFNet 输入适配：NestedUNet 无法处理 2976×256，先下采样到 256×256。"""
    def __init__(self, vifnet, target_size=(256, 256)):
        super().__init__()
        self.model = vifnet
        self.target_size = target_size if isinstance(target_size, (tuple, list)) else (target_size, target_size)

    def forward(self, x, *args, **kwargs):
        if x.shape[2] != self.target_size[0] or x.shape[3] != self.target_size[1]:
            x = F.interpolate(x, size=self.target_size, mode="area")
        return self.model(x, *args, **kwargs)


def get_model_for_benchmark(benchmark_key, output_size=(256, 256)):
    """获取模型，若 BENCHMARK_FAIR_MODE=1 则对配置内模型使用 FairWrapper（方案一）。"""
    repo, model_name = MODEL_MAP[benchmark_key]
    use_fair = os.environ.get("BENCHMARK_FAIR_MODE", "0") == "1"
    if use_fair and benchmark_key in FAIR_MODELS_CONFIG:
        internal = FAIR_MODELS_CONFIG[benchmark_key]
        model = _get_model(repo, model_name, (internal, internal))
        return FairWrapper(model, target_size=output_size, internal_size=(internal, internal))
    if not use_fair and benchmark_key == "VIFNet":
        model = _get_model(repo, model_name, output_size)
        return NonFairVIFAdapter(model, target_size=output_size)
    return _get_model(repo, model_name, output_size)


# 10 个唯一架构 -> (repo, model_name)
MODEL_MAP = {
    "InversionNet": ("openfwi", "InversionNet"),
    "VelocityGAN": ("futefwi", "VelocityGAN"),
    "FuteFWI": ("futefwi", "FuteFWI"),
    "DCNet": ("dcnet", "DCNet"),
    "DDNet70": ("ddnet", "DDNet70"),
    "TU_Net": ("tu-net", "TU_Net"),
    "ABA_FWI": ("aba-fwi", "ABA_FWI"),
    "ConvNeXtKaggle": ("convnext-kaggle", "ConvNeXtKaggle"),
    "FCNVMB": ("aba-fwi", "FCNVMB_FWI"),  # 使用 ABA-FWI 的 FCNVMB_FWI 包装
    "VIFNet": ("vif-net", "VIFNet"),
}


def _forward(model, x, repo, model_name, output_size=(256, 256)):
    label_dsp = [output_size[0], output_size[1]]
    if repo == "dcnet" and model_name == "DCNet":
        return model(x, label_dsp)
    if repo in ("ddnet", "tu-net") and model_name == "FCNVMB":
        return model(x, label_dsp)
    return model(x)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()))
    p.add_argument("--data-root", required=True)
    p.add_argument("--global-map-csv", required=True)
    p.add_argument("--stats-json", required=True)
    p.add_argument("--align-multiple", type=int, default=32)
    p.add_argument("--align-mode", default="crop")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--save-interval", type=int, default=5, help="每 N 轮保存 checkpoint")
    p.add_argument("--vis-interval", type=int, default=5, help="每 N 轮导出 pred vs GT 对比图")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output-path", required=True)
    p.add_argument("--save-name", default="checkpoint")
    p.add_argument("--suffix", default="")
    p.add_argument("--sync-bn", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="混合精度训练，降低显存（ConvNeXtKaggle 等大模型用）")
    p.add_argument("--init-ckpt", default="", help="可选：训练前加载初始化权重（适用于 ConvNeXtKaggle finetune）")
    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args()


def _load_init_checkpoint(model, ckpt_path, use_ddp=False):
    if not ckpt_path:
        return
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError("init checkpoint not found: {}".format(ckpt_path))
    obj = torch.load(ckpt_path, map_location="cpu")
    state = obj
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                state = obj[key]
                break
    if not isinstance(state, dict):
        raise ValueError("Unsupported checkpoint format: {}".format(ckpt_path))
    # Common wrapper prefixes
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if (not use_ddp) or (torch.distributed.get_rank() == 0):
        print("[InitCkpt] loaded: {}".format(ckpt_path))
        print("[InitCkpt] missing keys: {}, unexpected keys: {}".format(len(missing), len(unexpected)))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    use_ddp = local_rank >= 0
    if use_ddp:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    repo, model_name = MODEL_MAP[args.model]
    output_size = (256, 256)
    model = get_model_for_benchmark(args.model, output_size)
    if args.init_ckpt:
        _load_init_checkpoint(model, args.init_ckpt, use_ddp)
    if args.sync_bn and use_ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    train_set = UnifiedFWIDataset(
        data_root=args.data_root,
        split="train",
        global_map_csv=args.global_map_csv,
        stats_json=args.stats_json,
        align_multiple=args.align_multiple,
        align_mode=args.align_mode,
        output_channel_dim=True,
        need_edge=False,
    )
    val_set = UnifiedFWIDataset(
        data_root=args.data_root,
        split="val",
        global_map_csv=args.global_map_csv,
        stats_json=args.stats_json,
        align_multiple=args.align_multiple,
        align_mode=args.align_mode,
        output_channel_dim=True,
        need_edge=False,
    )
    sampler = DistributedSampler(train_set, shuffle=True) if use_ddp else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    criterion_fn, needs_dual = _get_criterion(args.model, device)
    needs_edges = args.model in ("ABA_FWI", "DDNet70", "TU_Net", "VIFNet")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    out_dir = args.output_path
    os.makedirs(out_dir, exist_ok=True)
    if local_rank <= 0:
        print("[Benchmark] Unified train model={} repo={} seed={} amp={} loss=native".format(
            args.model, repo, args.seed, args.amp))

    # 按验证集 loss 选最佳轮次，最终 checkpoint.pth = 最佳（供 eval 使用）
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(args.epochs):
        if use_ddp:
            sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        cnt = 0
        for batch in train_loader:
            if len(batch) == 3:
                data, label, _ = batch
            else:
                data, label = batch
            if isinstance(label, list):
                label = torch.stack(label)
            data = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            edges = _batch_edges(label) if needs_edges else None
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = _forward(model.module if use_ddp else model, data, repo, model_name, output_size)
                if needs_dual:
                    loss = criterion_fn(pred, label, edges)
                else:
                    if isinstance(pred, list):
                        pred = pred[0]
                    loss = criterion_fn(pred, label, edges)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            cnt += 1

        if local_rank <= 0:
            avg = total_loss / max(cnt, 1)
            print("Epoch {} loss={:.6f}".format(epoch + 1, avg))

            # 验证集上算 loss，用于选最佳轮次
            model_to_eval = model.module if use_ddp else model
            model_to_eval.eval()
            val_total = 0.0
            val_cnt = 0
            with torch.no_grad():
                for vbatch in val_loader:
                    if len(vbatch) == 3:
                        vdata, vlabel, _ = vbatch
                    else:
                        vdata, vlabel = vbatch
                    if isinstance(vlabel, list):
                        vlabel = torch.stack(vlabel)
                    vdata = vdata.to(device, non_blocking=True)
                    vlabel = vlabel.to(device, non_blocking=True)
                    vedges = _batch_edges(vlabel) if needs_edges else None
                    vpred = _forward(model_to_eval, vdata, repo, model_name, output_size)
                    if needs_dual:
                        vloss = criterion_fn(vpred, vlabel, vedges)
                    else:
                        vp = vpred[0] if isinstance(vpred, list) else vpred
                        vloss = criterion_fn(vp, vlabel, vedges)
                    val_total += vloss.item()
                    val_cnt += 1
            model_to_eval.train()
            val_loss = val_total / max(val_cnt, 1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                ckpt_best = {
                    "model_state_dict": model.module.state_dict() if use_ddp else model.state_dict(),
                    "epoch": epoch + 1,
                    "model": args.model,
                    "best_val_loss": best_val_loss,
                }
                best_path = os.path.join(out_dir, "{}_best.pth".format(args.save_name))
                torch.save(ckpt_best, best_path)
                print("Best val loss: {:.6f} at epoch {} -> saved {}".format(best_val_loss, best_epoch, best_path))

            # 每 save_interval 轮保存 checkpoint_epoch.pth（不覆盖 checkpoint.pth，最后统一用最佳）
            if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
                ckpt = {
                    "model_state_dict": model.module.state_dict() if use_ddp else model.state_dict(),
                    "epoch": epoch + 1,
                    "model": args.model,
                }
                save_path = os.path.join(out_dir, "{}_{}.pth".format(args.save_name, epoch + 1))
                torch.save(ckpt, save_path)
                print("Saved: {}".format(save_path))

            # 每 vis_interval 轮导出 pred vs GT 对比图
            if (epoch + 1) % args.vis_interval == 0 or (epoch + 1) == args.epochs:
                vis_dir = os.path.join(out_dir, "visualizations")
                os.makedirs(vis_dir, exist_ok=True)
                model_to_eval = model.module if use_ddp else model
                model_to_eval.eval()
                with torch.no_grad():
                    vbatch = next(iter(val_loader))
                    if len(vbatch) == 3:
                        vdata, vlabel, _ = vbatch
                    else:
                        vdata, vlabel = vbatch
                    if isinstance(vlabel, list):
                        vlabel = torch.stack(vlabel)
                    vdata = vdata.to(device, non_blocking=True)
                    vpred = _forward(model_to_eval, vdata, repo, model_name, output_size)
                    if isinstance(vpred, list):
                        vpred = vpred[0]
                    vpred = vpred.cpu().numpy()
                    vlabel_np = vlabel.numpy()
                p = np.squeeze(vpred[0, 0])
                g = np.squeeze(vlabel_np[0, 0])
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
                vis_path = os.path.join(vis_dir, "epoch_{:04d}.png".format(epoch + 1))
                plt.savefig(vis_path, dpi=150)
                plt.close()
                print("Saved vis: {}".format(vis_path))
                model_to_eval.train()

    # 训练结束后，用验证集最佳 checkpoint 作为 checkpoint.pth，供后续 eval 使用
    if local_rank <= 0 and best_epoch > 0:
        best_path = os.path.join(out_dir, "{}_best.pth".format(args.save_name))
        final_path = os.path.join(out_dir, "{}.pth".format(args.save_name))
        if os.path.isfile(best_path):
            shutil.copy2(best_path, final_path)
            print("[Benchmark] Eval will use best epoch {} (val_loss={:.6f}) -> {}".format(best_epoch, best_val_loss, final_path))
            with open(os.path.join(out_dir, "best_epoch.txt"), "w") as f:
                f.write("best_epoch={}\nbest_val_loss={:.6f}\n".format(best_epoch, best_val_loss))

    if use_ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
