# -*- coding: utf-8 -*-
"""
--------------

@Time：2024/4/17 15:57

@Author: TU-Net authors

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ConvBlock(nn.Module):
    def __init__(self, para_in_size, para_out_size, para_kernel_size=(3, 3), para_stride=(1, 1),
                 para_padding=(1, 1), para_is_bn=True, para_active_function=nn.ReLU(inplace=True)):
        super(ConvBlock, self).__init__()
        if para_is_bn:
            self.conv = nn.Sequential(
                nn.Conv2d(para_in_size, para_out_size, para_kernel_size, para_stride, para_padding),
                nn.BatchNorm2d(para_out_size),
                para_active_function)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(para_in_size, para_out_size, para_kernel_size, para_stride, para_padding),
                para_active_function)

    def forward(self, x):
        return self.conv(x)


class SeismicRecordDownSampling(nn.Module):
    def __init__(self, para_shot_num):
        super().__init__()
        self.dim_reduce1 = ConvBlock(para_shot_num, 8, para_kernel_size=(3, 1), para_stride=(2, 1), para_padding=(1, 0))
        self.dim_reduce2 = ConvBlock(8, 8, para_kernel_size=(3, 3), para_stride=(1, 1), para_padding=(1, 1))
        self.dim_reduce3 = ConvBlock(8, 16, para_kernel_size=(3, 1), para_stride=(2, 1), para_padding=(1, 0))
        self.dim_reduce4 = ConvBlock(16, 16, para_kernel_size=(3, 3), para_stride=(1, 1), para_padding=(1, 1))
        self.dim_reduce5 = ConvBlock(16, 32, para_kernel_size=(3, 1), para_stride=(2, 1), para_padding=(1, 0))
        self.dim_reduce6 = ConvBlock(32, 32, para_kernel_size=(3, 3), para_stride=(1, 1), para_padding=(1, 1))

    def forward(self, x):
        width = x.shape[3]
        dim_reduce0 = F.interpolate(x, size=[width * 8, width], mode='bilinear', align_corners=False)
        dim_reduce1 = self.dim_reduce1(dim_reduce0)
        dim_reduce2 = self.dim_reduce2(dim_reduce1)
        dim_reduce3 = self.dim_reduce3(dim_reduce2)
        dim_reduce4 = self.dim_reduce4(dim_reduce3)
        dim_reduce5 = self.dim_reduce5(dim_reduce4)
        dim_reduce6 = self.dim_reduce6(dim_reduce5)

        return dim_reduce6


class SeismicRecordDownSampling2(nn.Module):
    def __init__(self, para_shot_num):
        super().__init__()
        self.dim_reduce1 = ConvBlock(para_shot_num, 5, para_kernel_size=(7, 1), para_stride=(2, 1), para_padding=(3, 0))
        self.dim_reduce2 = ConvBlock(5, 8, para_kernel_size=(3, 1), para_stride=(2, 1), para_padding=(1, 0))
        self.dim_reduce3 = ConvBlock(8, 8, para_kernel_size=(3, 1), para_stride=(1, 1), para_padding=(1, 0))
        self.dim_reduce4 = ConvBlock(8, 16, para_kernel_size=(3, 1), para_stride=(2, 1), para_padding=(1, 0))
        self.dim_reduce5 = ConvBlock(16, 16, para_kernel_size=(3, 1), para_stride=(1, 1), para_padding=(1, 0))
        self.dim_reduce6 = ConvBlock(16, 32, para_kernel_size=(3, 1), para_stride=(2, 1), para_padding=(1, 0))
        self.dim_reduce7 = ConvBlock(32, 32, para_kernel_size=(3, 1), para_stride=(1, 1), para_padding=(1, 0))

    def forward(self, x):
        dim_reduce1 = self.dim_reduce1(x)  # (None, 5, 500, 70)
        dim_reduce2 = self.dim_reduce2(dim_reduce1)  # (None, 8, 250, 70)
        dim_reduce3 = self.dim_reduce3(dim_reduce2)  # (None, 8, 250, 70)
        dim_reduce4 = self.dim_reduce4(dim_reduce3)  # (None, 16, 125, 70)
        dim_reduce5 = self.dim_reduce5(dim_reduce4)  # (None, 16, 125, 70)
        dim_reduce6 = self.dim_reduce6(dim_reduce5)  # (None, 32, 63, 70)
        dim_reduce7 = self.dim_reduce7(dim_reduce6)
        w = x.shape[3]
        dim_reduce8 = F.interpolate(dim_reduce7, size=(w, w), mode='bilinear', align_corners=False)

        return dim_reduce8


class UNetConv2(nn.Module):
    def __init__(self, para_in_size, para_out_size, para_is_bn, para_active_func=nn.ReLU(inplace=True)):
        super(UNetConv2, self).__init__()
        if para_is_bn:
            self.conv1 = nn.Sequential(nn.Conv2d(para_in_size, para_out_size, (3, 3), (1, 1), 1),
                                       nn.BatchNorm2d(para_out_size),
                                       para_active_func)
            self.conv2 = nn.Sequential(nn.Conv2d(para_out_size, para_out_size, (3, 3), (1, 1), 1),
                                       nn.BatchNorm2d(para_out_size),
                                       para_active_func)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(para_in_size, para_out_size, (3, 3), (1, 1), 1),
                                       para_active_func)
            self.conv2 = nn.Sequential(nn.Conv2d(para_out_size, para_out_size, (3, 3), (1, 1), 1),
                                       para_active_func)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UNetUp1(nn.Module):
    def __init__(self, in_size, out_size, output_lim, is_deconv=True):
        super(UNetUp1, self).__init__()
        self.output_lim = output_lim
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, input):
        input = self.up(input)
        output = F.interpolate(input, size=self.output_lim, mode='bilinear', align_corners=False)
        return output


class UNetUp2(nn.Module):
    def __init__(self, in_size, out_size, output_lim, is_deconv, active_function=nn.ReLU(inplace=True)):
        super(UNetUp2, self).__init__()
        self.output_lim = output_lim
        self.conv = UNetConv2(in_size, out_size, True, active_function)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = PixelShuffleBlock(in_size, out_size, para_upscale_factor=2)

    def forward(self, input1, input2):
        input2 = self.up(input2)
        input2 = F.interpolate(input2, size=self.output_lim, mode='bilinear', align_corners=False)
        return self.conv(torch.cat([input1, input2], 1))


class PixelShuffle(nn.Module):
    def __init__(self, para_upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = para_upscale_factor

    def forward(self, inputs):
        channels, height, width = inputs.size()
        new_channels = channels // (self.upscale_factor ** 2)
        new_height, new_width = height * self.upscale_factor, width * self.upscale_factor
        inputs = inputs.view(new_channels, self.upscale_factor, self.upscale_factor, height, width)
        inputs = inputs.permute(0, 1, 4, 2, 5, 3).contiguous()
        outputs = inputs.view(new_channels, new_height, new_width)
        return outputs


class PixelShuffleBlock(nn.Module):
    def __init__(self, para_in_size, para_out_size, para_upscale_factor, para_kernel_size=(3, 3),
                 para_stride=(1, 1), para_padding=(1, 1)):
        super(PixelShuffleBlock, self).__init__()
        self.conv = nn.Conv2d(para_in_size, para_out_size * para_upscale_factor ** 2,
                              para_kernel_size, para_stride, para_padding)
        self.ps = nn.PixelShuffle(para_upscale_factor)

    def forward(self, inputs):
        outputs = self.ps(self.conv(inputs))
        return outputs


class ResidualBlock(nn.Module):
    def __init__(self, para_in_size, para_out_size, use_1x1conv=False, para_stride=(1, 1)):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(para_in_size, para_out_size, kernel_size=(3, 3), stride=para_stride, padding=1)
        self.conv2 = nn.Conv2d(para_out_size, para_out_size, kernel_size=(3, 3), padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                para_in_size, para_out_size, kernel_size=(1, 1), stride=para_stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(para_out_size)
        self.bn2 = nn.BatchNorm2d(para_out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return F.relu(y)


class ConvBlockTanh(nn.Module):
    def __init__(self, para_in_size, para_out_size, para_kernel_size=(3, 3), para_stride=(1, 1), para_padding=1):
        super(ConvBlockTanh, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(para_in_size, para_out_size, para_kernel_size, para_stride, para_padding),
                                  nn.BatchNorm2d(para_out_size),
                                  nn.Tanh())

    def forward(self, x):
        return self.conv(x)
