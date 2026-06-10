# -*- coding: utf-8 -*-
"""
FairWrapper: 内部降维适配法（方案一）
将 256x256 输入下采样到模型"舒适"分辨率（如 70x70），处理后再上采样回 256。
解决感受野/处理密度与 256 分辨率不匹配，实现公平对比。
"""
from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class FairWrapper(nn.Module):
    """
    入口下采样 -> 模型核心处理 -> 出口上采样。
    适用于原设计为 70x70 的 InversionNet、ABA_FWI、FuteFWI、VelocityGAN 等。
    """

    def __init__(self, original_model, target_size=(256, 256), internal_size=(70, 70)):
        super().__init__()
        self.model = original_model
        self.target_size = target_size if isinstance(target_size, (tuple, list)) else (target_size, target_size)
        self.internal_size = internal_size if isinstance(internal_size, (tuple, list)) else (internal_size, internal_size)

    def forward(self, x, *args, **kwargs):
        # 1. 入口下采样：area 区域插值，更好保留波场能量平均值，减少混叠
        x_small = F.interpolate(x, size=self.internal_size, mode="area")
        # 2. 模型核心处理（在低分辨率下进行）
        out = self.model(x_small, *args, **kwargs)
        # 3. 出口上采样：bicubic 双三次插值，输出更平滑，减少锯齿
        if isinstance(out, list):
            out = [F.interpolate(o, size=self.target_size, mode="bicubic", align_corners=False) for o in out]
        else:
            out = F.interpolate(out, size=self.target_size, mode="bicubic", align_corners=False)
        return out


# 启用公平模式的模型及内部分辨率
# U-Net 类（DDNet70、TU_Net、VIFNet 的 NestedUNet）需 2^n 尺寸（64）避免 encoder/decoder 跳跃连接尺寸不匹配
# 其他模型 70 与原始设计一致
FAIR_MODELS_CONFIG = {
    # === 统一脚本（自动生效）===
    "ABA_FWI": 70,       # Transformer，原 70x70，对尺寸不敏感
    "FCNVMB": 70,       # 全卷积，若不涉及深层跳跃连接通常 70 可跑
    "TU_Net": 64,       # U-Net：70→64 适配 /32 下采样，避免 17 vs 16 拼接错位
    "DDNet70": 64,      # U-Net：70→64 适配 /16 下采样，避免 9 vs 8 拼接错位
    "DCNet": 70,        # DenseNet 通常鲁棒
    "ConvNeXtKaggle": 70,
    "VIFNet": 64,       # NestedUNet：需 2^n 尺寸，与 profile_model_metrics 的配置保持一致
    # === 独立脚本（train 中手动集成，eval/profile 由此配置生效）===
    "InversionNet": 70,
    "FuteFWI": 70,
}


def wrap_if_fair(model, model_name, target_size=(256, 256)):
    """若启用公平模式且模型在配置中，则用 FairWrapper 包装。"""
    use_fair = os.environ.get("BENCHMARK_FAIR_MODE", "0") == "1"
    if not use_fair:
        return model
    internal = FAIR_MODELS_CONFIG.get(model_name)
    if internal is None:
        return model
    return FairWrapper(model, target_size=target_size, internal_size=(internal, internal))
