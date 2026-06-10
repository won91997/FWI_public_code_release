# -*- coding: utf-8 -*-
"""
FCNVMB_FWI wrapper for benchmark - FCNVMB with output_size interface.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1),
                nn.ReLU(inplace=True),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1),
                nn.ReLU(inplace=True),
            )

    def forward(self, inputs):
        return self.conv2(self.conv1(inputs))


class unetDown(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetDown, self).__init__()
        self.conv = unetConv2(in_size, out_size, is_batchnorm)
        self.down = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, inputs):
        return self.down(self.conv(inputs))


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, True)
        self.up = (
            nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
            if is_deconv
            else nn.UpsamplingBilinear2d(scale_factor=2)
        )

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset1 = outputs2.size()[2] - inputs1.size()[2]
        offset2 = outputs2.size()[3] - inputs1.size()[3]
        padding = [offset2 // 2, (offset2 + 1) // 2, offset1 // 2, (offset1 + 1) // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class _FCNVMB(nn.Module):
    def __init__(self, n_classes, in_channels, is_deconv, is_batchnorm):
        super(_FCNVMB, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        self.down1 = unetDown(in_channels, filters[0], is_batchnorm)
        self.down2 = unetDown(filters[0], filters[1], is_batchnorm)
        self.down3 = unetDown(filters[1], filters[2], is_batchnorm)
        self.down4 = unetDown(filters[2], filters[3], is_batchnorm)
        self.center = unetConv2(filters[3], filters[4], is_batchnorm)
        self.up4 = unetUp(filters[4], filters[3], is_deconv)
        self.up3 = unetUp(filters[3], filters[2], is_deconv)
        self.up2 = unetUp(filters[2], filters[1], is_deconv)
        self.up1 = unetUp(filters[1], filters[0], is_deconv)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs, label_dsp_dim):
        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        center = self.center(down4)
        up4 = self.up4(down4, center)
        up3 = self.up3(down3, up4)
        up2 = self.up2(down2, up3)
        up1 = self.up1(down1, up2)
        up1 = up1[:, :, 1 : 1 + label_dsp_dim[0], 1 : 1 + label_dsp_dim[1]].contiguous()
        return self.final(up1)


class FCNVMB_FWI(nn.Module):
    """FCNVMB with model_dim (output_size) interface for benchmark."""

    def __init__(self, model_dim=(256, 256), in_channels=5):
        super().__init__()
        self.model_dim = model_dim if isinstance(model_dim, (list, tuple)) else (model_dim, model_dim)
        self.core = _FCNVMB(
            n_classes=1, in_channels=in_channels, is_deconv=True, is_batchnorm=True
        )

    def forward(self, x):
        label_dsp = [self.model_dim[0], self.model_dim[1]]
        out = self.core(x, label_dsp)
        # benchmark 统一 label 为 (256,256)，_FCNVMB 可能因 ceil_mode/stride 输出 255 等非精确尺寸，
        # 与 ABA_FWI 一致，末尾用 F.interpolate 保证输出与 model_dim 一致。
        if out.shape[-2:] != tuple(self.model_dim):
            out = F.interpolate(out, size=self.model_dim, mode="bilinear", align_corners=False)
        return out
