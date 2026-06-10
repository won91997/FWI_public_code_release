#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 nonfair 十模型 metrics.json，并绘制同一 split 内样本的 Reference 与预测图（两张 PNG）。

默认 split=test（eval/pred.npy，1147 行），样本号 NONFAIR_VIZ_SAMPLE 为 test 内索引（与 717 一致）。

训练集：设 NONFAIR_VIZ_SPLIT=train 且 NONFAIR_VIZ_SAMPLE=<在 pred.npy 中的行下标>。
全量导出时行下标等于 train 内 split_idx（如 1361）。
若使用 unified_benchmark_test.py --subset-indices 1361 仅导出一条，则 pred 只有 1 行，请设
  NONFAIR_VIZ_PRED_ROW=0（取 pred 第 0 行，对应 subset 里的那条 train 样本）。
此时各模型需存在：
  <evdir>/train/pred.npy 与 <evdir>/train/gt.npy
（evdir 为 MODELS 里与 test 相同的 eval 目录）。

输出：logs_nonfair/nonfair_10models_reference.png、nonfair_10models_predictions.png；
路径可用 NONFAIR_VIZ_OUT_REFERENCE / NONFAIR_VIZ_OUT_PREDICTIONS 覆盖。
预测图布局可微调：NONFAIR_VIZ_FIG_H、NONFAIR_VIZ_PRED_HSPACE / _WSPACE、
NONFAIR_VIZ_PRED_LEFT / _RIGHT / _TOP / _BOTTOM。

默认横纵轴为物理坐标 (m)，extent 由 NONFAIR_VIZ_EXTENT_X_M / _Z_M（默认 3200×10000）；
子图强制正方形（几何上为拉伸显示）。NONFAIR_VIZ_USE_GRID_INDEX=1 时改用语义为网格下标。

速度色标：NONFAIR_VIZ_VRANGE=panel（默认，本图 GT+预测 min/max 加少量边距）、stats（train_stats.json）、
pctl（分位数，见 NONFAIR_VIZ_PCTL_LO/HI）；或直接设 NONFAIR_VIZ_VMIN / NONFAIR_VIZ_VMAX。

兼容宿主机 Python 3.6（无 PEP 563 annotations）。

效率列：Params(M)、FLOPs(G)、Infer(ms) 来自各模型 profile 日志中的 METRICS 行；
Train(h) 来自 seed 日志里 [INFO] Stage=train 到首个 Stage=eval 的间隔（与 run_benchmark_suite 一致）。
ConvNeXtKaggle / VIFNet 的 seed 日志无 [INFO] 行时，用 checkpoint_5→checkpoint_100 的 mtime 按比例估训练时长。

导出裁白边：NONFAIR_VIZ_SAVE_TIGHT（默认 1）、NONFAIR_VIZ_SAVE_PAD（inch，默认 0.02）。
色条与真值图一致：NONFAIR_VIZ_CBAR_GAP / NONFAIR_VIZ_CBAR_WIDTH（相对整图宽度）；
NONFAIR_VIZ_CBAR_LABEL_FS / NONFAIR_VIZ_CBAR_TICK_FS（色条字号，默认 12/11）；
NONFAIR_VIZ_REF_TITLE_FS / _REF_LABEL_FS / _REF_TICK_FS（Reference 图，默认 14/12/11）；
NONFAIR_VIZ_PRED_TITLE_FS / _PRED_LABEL_FS / _PRED_TICK_FS（预测拼图，默认 12/11/10）。
"""
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

_FONT_MAIN = "Times New Roman"


def _setup_times_new_roman():
    """全图统一 Times New Roman（标题、轴标签、刻度、色条文字）。"""
    plt.rcParams["font.family"] = _FONT_MAIN
    plt.rcParams["axes.unicode_minus"] = False

# 脚本在 benchmark/；仓库根为 Resolution-Constrained
BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
LOGS = ROOT / "logs_nonfair"
MAIN = LOGS / "main_10models_nonfair"
STATS_JSON = BENCH / "generated_split" / "train_stats.json"
GLOBAL_MAP_CSV = BENCH / "generated_split" / "global_map.csv"

_TS_BRACKET = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
_METRICS_LINE = re.compile(
    r"METRICS\s+params_m=([\d.]+)\s+flops_g=([\d.]+)\s+infer_ms=([\d.]+)"
)

# (展示名, metrics.json, pred 目录, profile 日志, seed 日志或 None, 训练输出目录用于 ckpt 时间估长)
MODELS = [
    ("ConvNeXtKaggle",
     LOGS / "convnext_kaggle_nonfair_seed42" / "eval" / "metrics.json",
     LOGS / "convnext_kaggle_nonfair_seed42" / "eval",
     LOGS / "convnext_kaggle_nonfair_seed42" / "ConvNeXtKaggle_profile.log",
     None,
     LOGS / "convnext_kaggle_nonfair_seed42"),
    ("FuteFWI",
     MAIN / "futefwi_seed42" / "eval_results" / "metrics.json",
     MAIN / "futefwi_seed42" / "eval_results",
     MAIN / "FuteFWI.log",
     MAIN / "FuteFWI_seed42.log",
     None),
    ("InversionNet",
     MAIN / "inversionnet_seed42" / "eval" / "metrics.json",
     MAIN / "inversionnet_seed42" / "eval",
     MAIN / "InversionNet.log",
     MAIN / "InversionNet_seed42.log",
     None),
    ("VelocityGAN",
     MAIN / "velocitygan_seed42" / "eval_results" / "metrics.json",
     MAIN / "velocitygan_seed42" / "eval_results",
     MAIN / "VelocityGAN.log",
     MAIN / "VelocityGAN_seed42.log",
     None),
    ("DCNet",
     MAIN / "dcnet_seed42" / "eval" / "metrics.json",
     MAIN / "dcnet_seed42" / "eval",
     MAIN / "DCNet.log",
     MAIN / "DCNet_seed42.log",
     None),
    ("DDNet70",
     MAIN / "ddnet70_seed42" / "eval" / "metrics.json",
     MAIN / "ddnet70_seed42" / "eval",
     MAIN / "DDNet70.log",
     MAIN / "DDNet70_seed42.log",
     None),
    ("TU-Net",
     MAIN / "tu_net_seed42" / "eval" / "metrics.json",
     MAIN / "tu_net_seed42" / "eval",
     MAIN / "TU_Net.log",
     MAIN / "TU_Net_seed42.log",
     None),
    ("ABA-FWI",
     MAIN / "aba_fwi_seed42" / "eval" / "metrics.json",
     MAIN / "aba_fwi_seed42" / "eval",
     MAIN / "ABA_FWI.log",
     MAIN / "ABA_FWI_seed42.log",
     None),
    ("FCNVMB",
     MAIN / "fcnvmb_seed42" / "eval" / "metrics.json",
     MAIN / "fcnvmb_seed42" / "eval",
     MAIN / "FCNVMB.log",
     MAIN / "FCNVMB_seed42.log",
     None),
    ("VIFNet",
     LOGS / "vifnet_nonfair_seed42" / "eval" / "metrics.json",
     LOGS / "vifnet_nonfair_seed42" / "eval",
     LOGS / "vifnet_nonfair_seed42" / "VIFNet_profile.log",
     None,
     LOGS / "vifnet_nonfair_seed42"),
]


def _union_axes_norm_bbox(axes):
    """所有数据子图 position 的并集，用于预测拼图色条与整块图等高。"""
    prs = [ax.get_position() for ax in axes]
    if not prs:
        return None
    x0 = min(p.x0 for p in prs)
    x1 = max(p.x0 + p.width for p in prs)
    y0 = min(p.y0 for p in prs)
    y1 = max(p.y0 + p.height for p in prs)
    return (x0, y0, x1 - x0, y1 - y0)


def _colorbar_right_consistent(fig, mappable, pos_like, gap_frac, width_frac, label_fs, tick_fs):
    """fig.add_axes 色条，与内容区右缘间距 gap_frac、宽度 width_frac（均相对图宽）；高度与 pos 等高。"""
    fig.canvas.draw()
    if hasattr(pos_like, "x0"):
        x0b, y0b, wb, hb = pos_like.x0, pos_like.y0, pos_like.width, pos_like.height
    else:
        x0b, y0b, wb, hb = pos_like
    gap = float(gap_frac)
    w = max(float(width_frac), 1e-4)
    x0 = x0b + wb + gap
    if x0 + w > 0.998:
        x0 = max(x0b + wb + 0.004, 0.998 - w)
    cax = fig.add_axes([x0, y0b, w, hb])
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(
        "Velocity (m/s)",
        fontsize=label_fs,
        fontfamily=_FONT_MAIN,
    )
    cb.ax.tick_params(axis="y", labelsize=tick_fs)
    for _t in cb.ax.get_yticklabels():
        _t.set_fontfamily(_FONT_MAIN)
    return cb


def _savefig_trim(fig, path, dpi, pad_inches, use_tight):
    path = str(path)
    fc = fig.get_facecolor()
    if use_tight and pad_inches is not None and float(pad_inches) >= 0:
        fig.savefig(
            path,
            dpi=dpi,
            facecolor=fc,
            bbox_inches="tight",
            pad_inches=float(pad_inches),
        )
    else:
        fig.savefig(path, dpi=dpi, facecolor=fc)


def _count_split_rows(csv_path, split_name):
    smap = {"train": 0, "val": 1, "test": 2}
    sv = smap[split_name]
    n = 0
    with open(str(csv_path), "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row.get("split", -1)) == sv:
                n += 1
    return n


def _resolve_pred_gt_paths(evdir, split_key):
    """test: evdir/pred.npy；train/val: evdir/<split>/pred.npy（与 unified_benchmark_test 导出一致）。"""
    evdir = Path(evdir)
    split_key = split_key.strip().lower()
    if split_key == "test":
        pp, gp = evdir / "pred.npy", evdir / "gt.npy"
        if pp.is_file() and gp.is_file():
            return pp, gp
        return None, None
    for base in (evdir / split_key, evdir.parent / split_key):
        pp, gp = base / "pred.npy", base / "gt.npy"
        if pp.is_file() and gp.is_file():
            return pp, gp
    return None, None


def _parse_profile_triplet(profile_path):
    """返回 (params_m, flops_g, infer_ms) 或 None。"""
    if not profile_path.is_file():
        return None
    with open(str(profile_path), "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _METRICS_LINE.search(line)
            if m:
                return float(m.group(1)), float(m.group(2)), float(m.group(3))
    return None


def _train_hours_from_seed_log(seed_path):
    """[INFO] Stage=train 到其后首个 Stage=eval（秒时间戳）。"""
    if not seed_path or not seed_path.is_file():
        return None
    last_train = None
    pair_train = None
    pair_eval = None
    with open(str(seed_path), "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            br = _TS_BRACKET.match(line)
            if not br:
                continue
            t = datetime.strptime(br.group(1), "%Y-%m-%d %H:%M:%S")
            if "Stage=train" in line:
                last_train = t
            if "Stage=eval" in line and last_train is not None:
                pair_train = last_train
                pair_eval = t
    if pair_train and pair_eval:
        dt = (pair_eval - pair_train).total_seconds() / 3600.0
        return max(dt, 0.0)
    return None


def _train_hours_from_checkpoints(out_dir):
    """用 ckpt_5 与 ckpt_100 的 mtime 差，按 epoch 比例外推到 100 epoch（无 [INFO] 时的近似）。"""
    if not out_dir or not out_dir.is_dir():
        return None
    p5 = out_dir / "checkpoint_5.pth"
    p100 = out_dir / "checkpoint_100.pth"
    if not p5.is_file() or not p100.is_file():
        return None
    dt = os.path.getmtime(str(p100)) - os.path.getmtime(str(p5))
    if dt <= 0:
        return None
    return (dt / 3600.0) * (100.0 / 95.0)


def _efficiency_row(display_name, profile_path, seed_path, ckpt_out_dir):
    pm, fg, inf = None, None, None
    triplet = _parse_profile_triplet(profile_path)
    if triplet:
        pm, fg, inf = triplet
    th = _train_hours_from_seed_log(seed_path)
    train_h_note = ""
    if th is None:
        th = _train_hours_from_checkpoints(ckpt_out_dir)
        if th is not None:
            train_h_note = " (est. from ckpt mtime 5→100)"
    return {
        "Params_M": pm,
        "FLOPs_G": fg,
        "Train_h": th,
        "Infer_ms": inf,
        "_train_note": train_h_note,
    }


def main():
    _setup_times_new_roman()
    sample_idx = int(os.environ.get("NONFAIR_VIZ_SAMPLE", "0"))
    pred_row_env = os.environ.get("NONFAIR_VIZ_PRED_ROW")
    row_idx = int(pred_row_env) if pred_row_env is not None and str(pred_row_env).strip() != "" else sample_idx
    viz_split = os.environ.get("NONFAIR_VIZ_SPLIT", "test").strip().lower()
    if viz_split not in ("train", "val", "test"):
        viz_split = "test"
    out_png_ref = Path(os.environ.get("NONFAIR_VIZ_OUT_REFERENCE", str(LOGS / "nonfair_10models_reference.png")))
    out_png_pred = Path(os.environ.get("NONFAIR_VIZ_OUT_PREDICTIONS", str(LOGS / "nonfair_10models_predictions.png")))
    explicit_csv = os.environ.get("NONFAIR_VIZ_OUT_CSV")
    want_csv_sidecar = os.environ.get("NONFAIR_VIZ_WRITE_CSV", "").lower() in ("1", "true", "yes")
    if viz_split == "test":
        out_csv = Path(explicit_csv) if explicit_csv else LOGS / "nonfair_10models_metrics.csv"
    else:
        if explicit_csv:
            out_csv = Path(explicit_csv)
        elif want_csv_sidecar:
            out_csv = LOGS / f"nonfair_10models_metrics_{viz_split}_viz.csv"
        else:
            out_csv = None

    if not STATS_JSON.is_file():
        print("missing", STATS_JSON, file=sys.stderr)
        return 1

    with open(STATS_JSON, "r", encoding="utf-8") as f:
        stats = json.load(f)
    vm = stats.get("vmodel", {})
    stats_vmin = float(vm.get("vmin", 2000))
    stats_vmax = float(vm.get("vmax", 6000))

    def _r4(x):
        if x is None:
            return None
        return round(float(x), 4)

    n_expect = None
    if GLOBAL_MAP_CSV.is_file() and viz_split in ("train", "val", "test"):
        n_expect = _count_split_rows(GLOBAL_MAP_CSV, viz_split)

    rows = []
    gt0 = None
    preds = []

    for name, mpath, evdir, prof_path, seed_log, ck_dir in MODELS:
        if not mpath.is_file():
            print("missing metrics:", mpath, file=sys.stderr)
            return 1
        pred_p, gt_p = _resolve_pred_gt_paths(evdir, viz_split)
        if pred_p is None or gt_p is None:
            print(
                "missing pred/gt for split={} under {} (期望 {}/train/pred.npy 与 gt.npy)".format(
                    viz_split,
                    evdir,
                    evdir,
                ),
                file=sys.stderr,
            )
            if viz_split in ("train", "val"):
                print(
                    "请先对每个模型运行 benchmark/unified_benchmark_test.py，例如：\n"
                    "  --split {} --export-dir <该模型的 eval 目录>  （将写入 eval/{}/pred.npy）\n"
                    "其余参数与跑 test 时相同（--model、--checkpoint、--data-root、--global-map-csv、--stats-json）。".format(
                        viz_split,
                        viz_split,
                    ),
                    file=sys.stderr,
                )
            return 1
        with open(mpath, "r", encoding="utf-8") as f:
            m = json.load(f)
        eff = _efficiency_row(name, prof_path, seed_log, ck_dir)
        th_src = ""
        if eff["Train_h"] is not None:
            if eff["_train_note"]:
                th_src = "ckpt_mtime_est"
            else:
                th_src = "seed_log"
        row = {
            "model": name,
            "Params_M": _r4(eff["Params_M"]),
            "FLOPs_G": _r4(eff["FLOPs_G"]),
            "Train_h": _r4(eff["Train_h"]),
            "Train_h_source": th_src,
            "Infer_ms": _r4(eff["Infer_ms"]),
            "MSE": m.get("MSE"),
            "MAE": m.get("MAE"),
            "PSNR": m.get("PSNR"),
            "SSIM": m.get("SSIM"),
            "L1-Grad": m.get("L1-Grad"),
            "LPIPS": m.get("LPIPS"),
        }
        if eff["Params_M"] is None:
            print("warn: no METRICS in profile log for", name, prof_path, file=sys.stderr)
        if eff["Train_h"] is None:
            print("warn: could not infer Train_h for", name, file=sys.stderr)
        rows.append(row)
        pred = np.load(str(pred_p))
        gt = np.load(str(gt_p))
        if n_expect is not None and pred.shape[0] != n_expect:
            msg = "{} pred rows {} != global_map {} count {}".format(
                name, pred.shape[0], viz_split, n_expect
            )
            if viz_split in ("train", "val") and pred.shape[0] < n_expect:
                print("info (subset ok):", msg, file=sys.stderr)
            else:
                print("warn:", msg, file=sys.stderr)
        if row_idx >= pred.shape[0] or row_idx < 0:
            print(
                "row index out of range: NONFAIR_VIZ_PRED_ROW/SAMPLE ->",
                row_idx,
                pred.shape,
                "split",
                viz_split,
                file=sys.stderr,
            )
            return 1
        if gt0 is None:
            gt0 = np.asarray(gt[row_idx], dtype=np.float32)
        else:
            if not np.allclose(gt0, gt[row_idx]):
                print("warning: GT mismatch at row", row_idx, "for", name, file=sys.stderr)
        preds.append((name, np.asarray(pred[row_idx], dtype=np.float32)))

    csv_fields = [
        "model", "Params_M", "FLOPs_G", "Train_h", "Train_h_source", "Infer_ms",
        "MSE", "MAE", "PSNR", "SSIM", "L1-Grad", "LPIPS",
    ]
    if out_csv is not None:
        with open(str(out_csv), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    ev_lo = os.environ.get("NONFAIR_VIZ_VMIN")
    ev_hi = os.environ.get("NONFAIR_VIZ_VMAX")
    if ev_lo is not None and ev_hi is not None:
        vmin = float(ev_lo)
        vmax = float(ev_hi)
        vrange_note = "manual"
    else:
        vrange = os.environ.get("NONFAIR_VIZ_VRANGE", "panel").strip().lower()
        pieces = [gt0.ravel()] + [pr.ravel() for _, pr in preds]
        stack = np.concatenate(pieces)
        if vrange in ("stats", "train", "dataset"):
            vmin, vmax = stats_vmin, stats_vmax
            vrange_note = "train_stats.json vmodel"
        elif vrange in ("pctl", "percentile"):
            pclo = float(os.environ.get("NONFAIR_VIZ_PCTL_LO", "1"))
            pchi = float(os.environ.get("NONFAIR_VIZ_PCTL_HI", "99"))
            vmin = float(np.percentile(stack, pclo))
            vmax = float(np.percentile(stack, pchi))
            vrange_note = "percentile {:.1f}–{:.1f}".format(pclo, pchi)
        else:
            vmin = float(np.min(stack))
            vmax = float(np.max(stack))
            vrange_note = "panel min/max"
        if vrange not in ("stats", "train", "dataset"):
            span = vmax - vmin
            pad = max(span * 0.03, 1.0)
            vmin -= pad
            vmax += pad

    print(
        "Colormap vmin/vmax: {:.2f} — {:.2f} m/s ({})".format(vmin, vmax, vrange_note),
        file=sys.stderr,
    )

    # 默认物理坐标 (m)；正方形子图靠 aspect=auto + set_box_aspect(1)（与真实纵横比不同，刻度仍为米）
    nz, nx = int(gt0.shape[0]), int(gt0.shape[1])
    use_grid_index = os.environ.get("NONFAIR_VIZ_USE_GRID_INDEX", "").lower() in ("1", "true", "yes")
    if use_grid_index:
        extent = (0.0, float(nx), float(nz), 0.0)
        xlim = (0.0, float(nx))
        ylim = (float(nz), 0.0)
        xlab = "Crossline index (0–{})".format(nx)
        ylab = "Depth index (0–{})".format(nz)
        xticks = [round(i * nx / 4.0) for i in range(5)]
        yticks = [round(i * nz / 4.0) for i in range(5)]
    else:
        ex_x1 = float(os.environ.get("NONFAIR_VIZ_EXTENT_X_M", "3200"))
        ex_z1 = float(os.environ.get("NONFAIR_VIZ_EXTENT_Z_M", "10000"))
        extent = (0.0, ex_x1, ex_z1, 0.0)
        xlim = (0.0, ex_x1)
        ylim = (ex_z1, 0.0)
        xlab = "Position (m)"
        ylab = "Depth (m)"
        xticks = [0, ex_x1 * 0.25, ex_x1 * 0.5, ex_x1 * 0.75, ex_x1]
        yticks = [0, ex_z1 * 0.25, ex_z1 * 0.5, ex_z1 * 0.75, ex_z1]

    n = len(preds)
    ncol = int(os.environ.get("NONFAIR_VIZ_NCOL", "5"))
    ncol = max(1, min(ncol, n))
    nrow_pred = int(np.ceil(n / float(ncol)))
    fig_w = float(os.environ.get("NONFAIR_VIZ_FIG_W", "14"))
    # 预测图：高度与「列数/行数」匹配，使 Grid 单格接近正方形，减少 set_box_aspect(1) 带来的大块留白
    pred_left = float(os.environ.get("NONFAIR_VIZ_PRED_LEFT", "0.11"))
    pred_right = float(os.environ.get("NONFAIR_VIZ_PRED_RIGHT", "0.885"))
    pred_bottom = float(os.environ.get("NONFAIR_VIZ_PRED_BOTTOM", "0.07"))
    pred_top = float(os.environ.get("NONFAIR_VIZ_PRED_TOP", "0.97"))
    pred_hspace = float(os.environ.get("NONFAIR_VIZ_PRED_HSPACE", "0.22"))
    pred_wspace = float(os.environ.get("NONFAIR_VIZ_PRED_WSPACE", "0.26"))
    _pw = pred_right - pred_left
    _ph = pred_top - pred_bottom
    _default_h = fig_w * (float(nrow_pred) / float(ncol)) * (_pw / _ph) + 0.62
    fig_h_pred = float(os.environ.get("NONFAIR_VIZ_FIG_H", str(round(_default_h, 2))))

    cbar_gap = float(os.environ.get("NONFAIR_VIZ_CBAR_GAP", "0.028"))
    cbar_width = float(os.environ.get("NONFAIR_VIZ_CBAR_WIDTH", "0.022"))
    cbar_label_fs = int(os.environ.get("NONFAIR_VIZ_CBAR_LABEL_FS", "12"))
    cbar_tick_fs = int(os.environ.get("NONFAIR_VIZ_CBAR_TICK_FS", "11"))
    ref_title_fs = int(os.environ.get("NONFAIR_VIZ_REF_TITLE_FS", "14"))
    ref_label_fs = int(os.environ.get("NONFAIR_VIZ_REF_LABEL_FS", "12"))
    ref_tick_fs = int(os.environ.get("NONFAIR_VIZ_REF_TICK_FS", "11"))
    pred_title_fs = int(os.environ.get("NONFAIR_VIZ_PRED_TITLE_FS", "12"))
    pred_label_fs = int(os.environ.get("NONFAIR_VIZ_PRED_LABEL_FS", "11"))
    pred_tick_fs = int(os.environ.get("NONFAIR_VIZ_PRED_TICK_FS", "10"))
    save_tight = os.environ.get("NONFAIR_VIZ_SAVE_TIGHT", "1").lower() not in (
        "0",
        "false",
        "no",
    )
    save_pad = float(os.environ.get("NONFAIR_VIZ_SAVE_PAD", "0.02"))

    def _style_square_panel(ax):
        """子图区域为正方形；imshow 拉伸填满（刻度仍为 extent 所定义的物理或网格坐标）。"""
        ax.set_aspect("auto")
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)

    dpi = int(os.environ.get("NONFAIR_VIZ_DPI", "170"))

    # —— 图 1：仅 Reference ——
    fig_ref_w = float(os.environ.get("NONFAIR_VIZ_FIG_REF_W", "5.8"))
    fig_ref_h = float(os.environ.get("NONFAIR_VIZ_FIG_REF_H", "5.2"))
    fig_ref = plt.figure(figsize=(fig_ref_w, fig_ref_h), facecolor="white")
    ax_gt = fig_ref.add_subplot(1, 1, 1)
    im_ref = ax_gt.imshow(
        gt0, cmap="jet", vmin=vmin, vmax=vmax, aspect="auto",
        interpolation="nearest", extent=extent,
    )
    ax_gt.set_title("Reference", fontsize=ref_title_fs, fontfamily=_FONT_MAIN)
    ax_gt.set_xlim(xlim[0], xlim[1])
    ax_gt.set_ylim(ylim[0], ylim[1])
    ax_gt.set_xticks(xticks)
    ax_gt.set_yticks(yticks)
    ax_gt.set_xlabel(xlab, fontsize=ref_label_fs, fontfamily=_FONT_MAIN)
    ax_gt.set_ylabel(ylab, fontsize=ref_label_fs, fontfamily=_FONT_MAIN)
    ax_gt.tick_params(axis="both", labelsize=ref_tick_fs)
    for _t in ax_gt.get_xticklabels() + ax_gt.get_yticklabels():
        _t.set_fontfamily(_FONT_MAIN)
    _style_square_panel(ax_gt)
    fig_ref.subplots_adjust(left=0.14, right=0.82, top=0.90, bottom=0.12)
    _colorbar_right_consistent(
        fig_ref,
        im_ref,
        ax_gt.get_position(),
        cbar_gap,
        cbar_width,
        cbar_label_fs,
        cbar_tick_fs,
    )
    fig_ref.canvas.draw()
    _savefig_trim(fig_ref, out_png_ref, dpi, save_pad, save_tight)
    plt.close(fig_ref)

    # —— 图 2：仅各模型预测 ——
    fig_pred = plt.figure(figsize=(fig_w, fig_h_pred), facecolor="white")
    gs = GridSpec(
        nrow_pred,
        ncol,
        figure=fig_pred,
        hspace=pred_hspace,
        wspace=pred_wspace,
        left=pred_left,
        right=pred_right,
        top=pred_top,
        bottom=pred_bottom,
    )

    im_last = None
    pred_axes = []
    for idx in range(n):
        name, pr = preds[idx]
        r = idx // ncol
        c = idx % ncol
        ax = fig_pred.add_subplot(gs[r, c])
        pred_axes.append(ax)
        im_last = ax.imshow(
            pr, cmap="jet", vmin=vmin, vmax=vmax, aspect="auto",
            interpolation="nearest", extent=extent,
        )
        ax.set_title(name, fontsize=pred_title_fs, fontfamily=_FONT_MAIN)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        _style_square_panel(ax)
        ax.tick_params(axis="both", labelsize=pred_tick_fs)
        ax.set_xlabel(xlab, fontsize=pred_label_fs, fontfamily=_FONT_MAIN)
        if c == 0:
            ax.set_ylabel(ylab, fontsize=pred_label_fs, labelpad=2, fontfamily=_FONT_MAIN)
        else:
            ax.set_ylabel("")
        for _t in ax.get_xticklabels() + ax.get_yticklabels():
            _t.set_fontfamily(_FONT_MAIN)

    if im_last is not None and pred_axes:
        ubox = _union_axes_norm_bbox(pred_axes)
        if ubox is not None:
            _colorbar_right_consistent(
                fig_pred,
                im_last,
                ubox,
                cbar_gap,
                cbar_width,
                cbar_label_fs,
                cbar_tick_fs,
            )
    fig_pred.canvas.draw()
    _savefig_trim(fig_pred, out_png_pred, dpi, save_pad, save_tight)
    plt.close(fig_pred)

    if out_csv is not None:
        print("Wrote", out_csv)
    print("Wrote", out_png_ref)
    print("Wrote", out_png_pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
