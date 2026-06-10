# -*- coding: utf-8 -*-
"""
--------------

@Time：2024/5/1 11:47

@Author: TU-Net authors

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from distutils.version import LooseVersion
from model.dcn import ModulatedDeformConvPack, modulated_deform_conv


class DCNv2Pack(ModulatedDeformConvPack):
    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        # offset_absmean = torch.mean(torch.abs(offset))
        # if offset_absmean > 50:
        #     logger = get_root_logger()
        #     logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')

        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                                 self.dilation, mask)
        else:
            return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, self.groups, self.deformable_groups)


class TextureWarpingModule(nn.Module):
    def __init__(self, para_in_size, para_cond_size, para_cond_downscale_rate, para_groups, para_pre_offset_size=0):
        super(TextureWarpingModule, self).__init__()
        self.cond_downscale_rate = para_cond_downscale_rate
        self.offset_conv1 = nn.Sequential(
            nn.Conv2d(para_in_size + para_cond_size, para_in_size, kernel_size=(1, 1)),
            nn.GroupNorm(num_groups=32, num_channels=para_in_size, eps=1e-6, affine=True), nn.SiLU(inplace=True),
            nn.Conv2d(para_in_size, para_in_size, groups=para_in_size, kernel_size=(7, 7), padding=3),
            nn.GroupNorm(num_groups=32, num_channels=para_in_size, eps=1e-6, affine=True), nn.SiLU(inplace=True),
            nn.Conv2d(para_in_size, para_in_size, kernel_size=(1, 1)))

        self.offset_conv2 = nn.Sequential(
            nn.Conv2d(para_in_size + para_pre_offset_size, para_in_size, (3, 3), (1, 1), 1),
            nn.GroupNorm(num_groups=32, num_channels=para_in_size, eps=1e-6, affine=True), nn.SiLU(inplace=True))
        self.dcn = DCNv2Pack(para_in_size, para_in_size, 3, padding=1, deformable_groups=para_groups)

    def forward(self, main_feature, condition_feature, previous_offset=None):
        _, _, h, w = condition_feature.shape
        _, _, H, W = main_feature.shape
        condition_feature = F.interpolate(
            condition_feature,
            size=(h+2 // self.cond_downscale_rate, w+2 // self.cond_downscale_rate),
            mode='bilinear',
            align_corners=False)
        condition_feature = F.interpolate(condition_feature, size=(H, W), mode='bilinear', align_corners=False)
        offset = self.offset_conv1(torch.cat([condition_feature, main_feature], dim=1))
        if previous_offset is None:
            offset = self.offset_conv2(offset)
        else:
            offset = self.offset_conv2(torch.cat([offset, previous_offset], dim=1))
        warp_feat = self.dcn(main_feature, offset)
        return warp_feat, offset
