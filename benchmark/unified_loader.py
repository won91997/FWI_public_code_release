"""
Unified FWI Dataset for benchmark - 统一数据加载器

根据 global_map.csv 和 train_stats.json 加载 OpenFWI 格式数据，
供 InversionNet、VelocityGAN、FuteFWI 等模型使用。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_stats(stats_json: Optional[str]) -> dict:
    """加载 train_stats.json，若不存在则返回默认值"""
    if not stats_json or not os.path.isfile(stats_json):
        return {}
    with open(stats_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_channel_mode(seismic: np.ndarray, channel_mode: str) -> np.ndarray:
    """channel_mode: 'all' 使用全部通道, 'middle' 使用中间通道"""
    if channel_mode == "middle":
        c = seismic.shape[0]
        mid = c // 2
        return seismic[mid : mid + 1]  # (1, T, W)
    return seismic  # (C, T, W)


def _align_shape(
    arr: np.ndarray,
    align_multiple: int,
    align_mode: str,
    target_time: int,
    target_width: int,
    is_seismic: bool,
) -> np.ndarray:
    """
    对齐时间/宽度维度到 align_multiple 的倍数。
    arr: seismic (C,T,W) 或 vmodel (H,W)
    """
    if align_multiple <= 1 and target_time <= 0 and target_width <= 0:
        return arr

    if is_seismic:
        c, t, w = arr.shape
        if target_time > 0:
            t = target_time
        elif align_multiple > 1:
            t = (t // align_multiple) * align_multiple
        if target_width > 0:
            w = target_width
        elif align_multiple > 1:
            w = (w // align_multiple) * align_multiple

        if align_mode == "crop":
            return arr[:, :t, :w].copy()
        if align_mode == "pad":
            out = np.zeros((c, t, w), dtype=arr.dtype)
            out[:, : min(t, arr.shape[1]), : min(w, arr.shape[2])] = arr[
                :, : min(t, arr.shape[1]), : min(w, arr.shape[2])
            ]
            return out
        # none: 仅裁剪到目标尺寸
        return arr[:, :t, :w].copy()
    else:
        h, w = arr.shape
        if target_width > 0:
            w = target_width
        elif align_multiple > 1:
            w = (w // align_multiple) * align_multiple
        if align_multiple > 1:
            h = (h // align_multiple) * align_multiple

        if align_mode == "crop":
            return arr[:h, :w].copy()
        if align_mode == "pad":
            out = np.zeros((h, w), dtype=arr.dtype)
            out[: min(h, arr.shape[0]), : min(w, arr.shape[1])] = arr[
                : min(h, arr.shape[0]), : min(w, arr.shape[1])
            ]
            return out
        return arr[:h, :w].copy()


def _normalize_seismic(seismic: np.ndarray, stats: dict) -> np.ndarray:
    """使用 train_stats 中的 seismic 统计量归一化。文档：先 clip(x,-robust_max,robust_max)，再 / robust_max → [-1,1]"""
    out = seismic.astype(np.float32)
    robust_max = stats.get("seismic", {}).get("robust_max")
    if robust_max is not None and robust_max > 0:
        out = np.clip(out, -float(robust_max), float(robust_max))  # 先 clip
        out = out / float(robust_max)  # 再除 → [-1, 1]
    return out


def _normalize_vmodel(vmodel: np.ndarray, stats: dict) -> np.ndarray:
    """使用 train_stats 中的 vmodel 统计量归一化到 [-1, 1]。文档：min-max 到 [0,1]，clip 后 (x-0.5)*2"""
    out = vmodel.astype(np.float32)
    vmin = stats.get("vmodel", {}).get("vmin")
    vmax = stats.get("vmodel", {}).get("vmax")
    if vmin is not None and vmax is not None and vmax > vmin:
        out = (out - vmin) / (vmax - vmin)
        out = np.clip(out, 0.0, 1.0)  # 文档要求 clip 后再 *2-1
        out = out * 2 - 1  # [-1, 1]
    return out


class UnifiedFWIDataset(Dataset):
    """
    统一 FWI 数据集，基于 global_map.csv 和 train_stats.json。

    Args:
        data_root: 数据根目录（用于解析相对路径，若 csv 中已是绝对路径则可选）
        split: "train" | "val" | "test"，对应 split 列 0/1/2
        global_map_csv: global_map.csv 路径
        stats_json: train_stats.json 路径（train 集统计量，用于归一化）
        time_downsample: 时间维度下采样率，1 表示不采样
        channel_mode: "all" | "middle"
        align_multiple: 对齐到该数的倍数
        align_mode: "none" | "crop" | "pad"
        target_time: 目标时间长度，0 表示自动
        target_width: 目标宽度，0 表示自动
        output_channel_dim: 若 True，vmodel 输出 (1,H,W)；seismic 保持 (C,T,W)
    """

    SPLIT_MAP = {"train": 0, "val": 1, "test": 2}

    def __init__(
        self,
        data_root: str,
        split: str,
        global_map_csv: str,
        stats_json: str = "",
        time_downsample: int = 1,
        channel_mode: str = "all",
        align_multiple: int = 32,
        align_mode: str = "crop",
        target_time: int = 0,
        target_width: int = 0,
        output_channel_dim: bool = True,
        need_edge: bool = False,  # ABA-FWI 扩展用，当前忽略
    ):
        super().__init__()
        self.data_root = Path(data_root) if data_root else None
        self.split_name = split
        self.split_val = self.SPLIT_MAP.get(split, 0)
        self.time_downsample = max(1, time_downsample)
        self.channel_mode = channel_mode
        self.align_multiple = align_multiple
        self.align_mode = align_mode
        self.target_time = target_time
        self.target_width = target_width
        self.output_channel_dim = output_channel_dim

        self.stats = _load_stats(stats_json)
        self.rows = self._load_global_map(global_map_csv)

    def _load_global_map(self, csv_path: str) -> list[dict]:
        import csv

        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row.get("split", -1)) == self.split_val:
                    rows.append(row)
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        seismic_path = row["seismic_path"]
        vmodel_path = row["vmodel_path"]
        in_file_idx = int(row["in_file_idx"])

        seismic = np.load(seismic_path, mmap_mode="r")[in_file_idx]  # (C, T, W)
        vmodel = np.load(vmodel_path, mmap_mode="r")[in_file_idx]  # (H, W)

        # 时间下采样
        if self.time_downsample > 1:
            seismic = seismic[:, :: self.time_downsample, :]

        # 通道选择
        seismic = _apply_channel_mode(seismic, self.channel_mode)

        # 形状对齐
        seismic = _align_shape(
            seismic,
            self.align_multiple,
            self.align_mode,
            self.target_time,
            self.target_width,
            is_seismic=True,
        )
        vmodel = _align_shape(
            vmodel,
            self.align_multiple,
            self.align_mode,
            0,
            self.target_width,
            is_seismic=False,
        )

        # 归一化
        seismic = _normalize_seismic(seismic, self.stats)
        vmodel = _normalize_vmodel(vmodel, self.stats)

        # 转 tensor
        seismic_t = torch.from_numpy(seismic).float()
        vmodel_t = torch.from_numpy(vmodel).float()

        if self.output_channel_dim and vmodel_t.dim() == 2:
            vmodel_t = vmodel_t.unsqueeze(0)  # (1, H, W)

        return seismic_t, vmodel_t
