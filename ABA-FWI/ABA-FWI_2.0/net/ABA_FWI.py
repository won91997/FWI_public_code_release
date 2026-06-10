# -*- coding: utf-8 -*-
"""
ABA_FWI wrapper for benchmark - exports ABA_FWI(output_size=...) with forward(x).
"""
import torch.nn as nn
import torch.nn.functional as F
from .ABA_FWI_SEG import ABA_FWI_SEG


class ABA_FWI(nn.Module):
    """ABA-FWI model with output_size interface for benchmark."""

    def __init__(self, output_size=(256, 256)):
        super().__init__()
        self.output_size = output_size
        self.core = ABA_FWI_SEG(
            n_classes=1, in_channels=5, is_deconv=True, is_batchnorm=True
        )

    def forward(self, x):
        label_dsp = [self.output_size[0], self.output_size[1]]
        out = self.core(x, label_dsp)
        # ABA_FWI_SEG 的 UNet 解码器可能因 ceil_mode/stride 输出 255 等非精确尺寸，
        # 与 benchmark 统一 label (256,256) 不匹配。按 IMPLEMENTATION_GUIDE 规范，
        # 末尾用 F.interpolate 保证输出与 output_size 一致，与其他模型（FuteFWI/InversionNet）一致。
        if out.shape[-2:] != tuple(self.output_size):
            out = F.interpolate(out, size=self.output_size, mode="bilinear", align_corners=False)
        return out
