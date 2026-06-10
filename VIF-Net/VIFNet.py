# -*- coding: utf-8 -*-
"""
VIF-Net: Interface Completion in Full Waveform Inversion using Fusion Networks.
Two-stage architecture:
  Stage 1: NestedUNet (SToV) + NestedUNet (SToC) - predict velocity & contour
  Stage 2: EMD (VCToV) - refine velocity by fusing velocity + contour
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=dilation, bias=True),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, bias=True),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class _ConvBlockNested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.activation = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=True)
        self.bn1 = nn.InstanceNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=True)
        self.bn2 = nn.InstanceNorm2d(out_ch)

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        return self.activation(self.bn2(self.conv2(x)))


class NestedUNet(nn.Module):
    def __init__(self, in_channels, in_dsp_dim, out_dsp_dim):
        super().__init__()
        h0, w0 = in_dsp_dim
        h, w = h0, w0
        for _ in range(4):
            h = math.ceil(h / 2)
            w = math.ceil(w / 2)
        for _ in range(4):
            h *= 2
            if h % 2:
                h += 1
            w *= 2
            if w % 2:
                w += 1
        off_h = h - h0
        off_w = w - w0
        self.padding = [off_w // 2, (off_w + 1) // 2, off_h // 2, (off_h + 1) // 2]
        self.out_dsp_dim = out_dsp_dim

        n1 = 64
        f = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(2, 2)
        self.up0_1 = nn.ConvTranspose2d(f[1], f[1], 2, stride=2)
        self.up1_1 = nn.ConvTranspose2d(f[2], f[2], 2, stride=2)
        self.up0_2 = nn.ConvTranspose2d(f[1], f[1], 2, stride=2)
        self.up2_1 = nn.ConvTranspose2d(f[3], f[3], 2, stride=2)
        self.up1_2 = nn.ConvTranspose2d(f[2], f[2], 2, stride=2)
        self.up0_3 = nn.ConvTranspose2d(f[1], f[1], 2, stride=2)
        self.up3_1 = nn.ConvTranspose2d(f[4], f[4], 2, stride=2)
        self.up2_2 = nn.ConvTranspose2d(f[3], f[3], 2, stride=2)
        self.up1_3 = nn.ConvTranspose2d(f[2], f[2], 2, stride=2)
        self.up0_4 = nn.ConvTranspose2d(f[1], f[1], 2, stride=2)

        self.conv0_0 = _ConvBlockNested(in_channels, f[0], f[0])
        self.conv1_0 = _ConvBlockNested(f[0], f[1], f[1])
        self.conv2_0 = _ConvBlockNested(f[1], f[2], f[2])
        self.conv3_0 = _ConvBlockNested(f[2], f[3], f[3])
        self.conv4_0 = _ConvBlockNested(f[3], f[4], f[4])

        self.conv0_1 = _ConvBlockNested(f[0] + f[1], f[0], f[0])
        self.conv1_1 = _ConvBlockNested(f[1] + f[2], f[1], f[1])
        self.conv2_1 = _ConvBlockNested(f[2] + f[3], f[2], f[2])
        self.conv3_1 = _ConvBlockNested(f[3] + f[4], f[3], f[3])

        self.conv0_2 = _ConvBlockNested(f[0] * 2 + f[1], f[0], f[0])
        self.conv1_2 = _ConvBlockNested(f[1] * 2 + f[2], f[1], f[1])
        self.conv2_2 = _ConvBlockNested(f[2] * 2 + f[3], f[2], f[2])

        self.conv0_3 = _ConvBlockNested(f[0] * 3 + f[1], f[0], f[0])
        self.conv1_3 = _ConvBlockNested(f[1] * 3 + f[2], f[1], f[1])

        self.conv0_4 = _ConvBlockNested(f[0] * 4 + f[1], f[0], f[0])
        self.final = nn.Conv2d(f[0], 1, 1)

    def forward(self, x):
        x = F.pad(x, self.padding)
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up0_1(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up1_1(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up0_2(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up2_1(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up1_2(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up0_3(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up3_1(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up2_2(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up1_3(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up0_4(x1_3)], 1))

        ch = self.padding[2]
        cw = self.padding[0]
        x0_4 = x0_4[:, :, ch:ch + self.out_dsp_dim[0], cw:cw + self.out_dsp_dim[1]].contiguous()
        output = self.final(x0_4)
        return (torch.tanh(output) + 1) / 2


class EMD(nn.Module):
    def __init__(self, in_channels, out_dsp_dim, residual_blocks=6):
        super().__init__()
        self.out_dsp_dim = out_dsp_dim
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),
        )
        self.middle = nn.Sequential(*[ResnetBlock(256, dilation=2) for _ in range(residual_blocks)])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 1, 7, padding=0),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2
        return x[:, :, :self.out_dsp_dim[0], :self.out_dsp_dim[1]].contiguous()


class VIFNet(nn.Module):
    """
    Stage 1: SToV + SToC.
    Stage 2: VCToV refinement.
    Returns [refined_velocity, stage1_velocity, stage1_contour].
    """

    def __init__(self, output_size=(256, 256), in_channels=5, **kwargs):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size
        self.stov_net = NestedUNet(in_channels=in_channels, in_dsp_dim=output_size, out_dsp_dim=output_size)
        self.stoc_net = NestedUNet(in_channels=in_channels, in_dsp_dim=output_size, out_dsp_dim=output_size)
        self.vctov_net = EMD(in_channels=2, out_dsp_dim=output_size, residual_blocks=6)

    def forward(self, x):
        stov = self.stov_net(x)
        stoc = self.stoc_net(x)
        vc = torch.cat([stov.detach(), stoc.detach()], dim=1)
        refined = self.vctov_net(vc)
        refined_bm = refined * 2 - 1
        stov_bm = stov * 2 - 1
        return [refined_bm, stov_bm, stoc]
