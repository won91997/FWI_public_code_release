# -*- coding: utf-8 -*-
"""
--------------

@Time：2024/7/1 21:11

@Author: TU-Net authors

"""
import torch.nn.functional as F
from model.details import *
from model.dcn.TWM import *


class TU_Net(nn.Module):
    def __init__(self, n_classes, in_channels, is_deconv, is_batchnorm, output_size=(256, 256), **kwargs):
        super(TU_Net, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes
        self.output_size = output_size

        oh, ow = output_size

        # Encoder
        self.pre_seis_conv = SeismicRecordDownSampling(self.in_channels)
        self.down3 = UNetConv2(32, 64, self.is_batchnorm)
        self.max3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.down4 = UNetConv2(64, 128, self.is_batchnorm)
        self.max4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.down5 = UNetConv2(128, 256, self.is_batchnorm)
        self.max5 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.center = UNetConv2(256, 512, self.is_batchnorm)

        # TWM
        self.TWM2 = TextureWarpingModule(256, 32, 4, 4, 0)
        self.TWM3 = TextureWarpingModule(128, 32, 2, 4, 0)
        self.TWM4 = TextureWarpingModule(64, 32, 1, 4, 0)

        # Decoder
        self.Up5 = UNetUp2(512, 256, output_lim=[oh // 4, ow // 4], is_deconv=self.is_deconv)
        self.Up4 = UNetUp2(256, 128, output_lim=[oh // 2, ow // 2], is_deconv=self.is_deconv)
        self.Up3 = UNetUp2(128, 64, output_lim=[oh, ow], is_deconv=self.is_deconv)
        self.pixelShuffleBlock = PixelShuffleBlock(64, 64, para_upscale_factor=1)
        self.residual = ResidualBlock(64, 64, use_1x1conv=False, para_stride=1)
        self.residua2 = ResidualBlock(64, 64, use_1x1conv=False, para_stride=1)
        self.residua3 = ResidualBlock(64, 64, use_1x1conv=False, para_stride=1)
        self.dc2_final = ConvBlockTanh(64, 1)

    def forward(self, inputs, _ = None):
        '''
        :param inputs:      Input Image
        '''
        compress_seis = self.pre_seis_conv(inputs)  # (2, 32, 70, 70)

        down3 = self.down3(compress_seis)  # (2, 64, 70, 70)
        max3 = self.max3(down3)  # (2, 64, 35, 35)
        down4 = self.down4(max3)  # (2, 128, 35, 35)
        max4 = self.max4(down4)  # (2, 128, 18, 18)
        down5 = self.down5(max4)  # (2, 256, 18, 18)
        max5 = self.max5(down5)  # (2, 256, 9, 9)
        center = self.center(max5)  # (2, 512, 9, 9)

        # TWM
        TWM2 = self.TWM2(down5, compress_seis)  # (2, 256, 18, 18)
        TWM3 = self.TWM3(down4, compress_seis)  # (2, 128, 35, 35)
        TWM4 = self.TWM4(down3, compress_seis)  # (2, 64, 70, 70)

        # Decoder
        dc2_up5 = self.Up5(TWM2[0], center)  # (2, 256, 18, 18)
        dc2_up4 = self.Up4(TWM3[0], dc2_up5)  # (2, 128, 35, 35)
        dc2_up3 = self.Up3(TWM4[0], dc2_up4)  # (2, 64, 70, 70)
        pixelShuffleBlock = self.pixelShuffleBlock(dc2_up3)  # (2, 64, 70, 70)
        dc_residual1 = self.residual(pixelShuffleBlock)  # (2, 64, 70, 70)
        dc_residual2 = self.residua2(dc_residual1)  # (2, 64, 70, 70)
        dc_residual3 = self.residua3(dc_residual2)
        dc2_final = self.dc2_final(dc_residual3)
        if dc2_final.shape[-2:] != tuple(self.output_size):
            dc2_final = F.interpolate(dc2_final, size=self.output_size, mode='bilinear', align_corners=False)
        return dc2_final


if __name__ == '__main__':
    model = TU_Net(n_classes=1, in_channels=5, is_deconv=True, is_batchnorm=True)
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    model.to(device)
    from torchsummary import summary
    summary(model, input_size=[(5, 1000, 70)])
