# -*- coding: utf-8 -*-
"""
--------------

@Time：2024/7/8 21:02

@Author: TU-Net authors

"""

from model.details import *
from model.dcn.TWM import *

class TU_Net_SEG(nn.Module):
    def __init__(self, n_classes, in_channels, is_deconv, is_batchnorm):
        super(TU_Net_SEG, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes

        # Encoder
        self.down2 = UNetConv2(29, 32, self.is_batchnorm)
        self.max2 = nn.MaxPool2d(2, 2, ceil_mode=True)
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
        self.Up5 = UNetUp2(512, 256, output_lim=[50, 38], is_deconv=self.is_deconv)
        self.Up4 = UNetUp2(256, 128, output_lim=[100, 76], is_deconv=self.is_deconv)
        self.Up3 = UNetUp2(128, 64, output_lim=[200, 151], is_deconv=self.is_deconv)
        self.pixelShuffleBlock = PixelShuffleBlock(64, 64, para_upscale_factor=2)
        self.residual = ResidualBlock(64, 64, use_1x1conv=False, para_stride=1)
        self.residua2 = ResidualBlock(64, 64, use_1x1conv=False, para_stride=1)
        self.residua3 = ResidualBlock(64, 64, use_1x1conv=False, para_stride=1)
        self.dc2_final = nn.Conv2d(64, 1, 1)

    def forward(self, inputs, _ = None):
        '''
        :param inputs:      Input Image
        '''
        down2 = self.down2(inputs)  # (2, 32, 400, 301)
        max2 = self.max2(down2)  # (2, 32, 200, 151)
        down3 = self.down3(max2)  # (2, 64, 200, 151)
        max3 = self.max3(down3)  # (2, 64, 100, 76)
        down4 = self.down4(max3)  # (2, 128, 100, 76)
        max4 = self.max4(down4)  # (2, 128, 50, 38)
        down5 = self.down5(max4)  # (2, 256, 50, 38)
        max5 = self.max5(down5)  # (2, 256, 25, 19)
        center = self.center(max5)  # (2, 512, 25, 19)

        # TWM
        TWM2 = self.TWM2(down5, down2)  # (2, 256, 50, 38)
        TWM3 = self.TWM3(down4, down2)  # (2, 128, 100, 76)
        TWM4 = self.TWM4(down3, down2)  # (2, 64, 200, 151)
        #
        # # Decoder
        dc2_up5 = self.Up5(TWM2[0], center)  # (2, 256, 50, 38)
        dc2_up4 = self.Up4(TWM3[0], dc2_up5)  # (2, 128, 100, 76)
        dc2_up3 = self.Up3(TWM4[0], dc2_up4)  # (2, 64, 200, 151)
        pixelShuffleBlock = self.pixelShuffleBlock(dc2_up3)  # (2, 64, 400, 302)
        dc_residual1 = self.residual(pixelShuffleBlock)  # (2, 64, 400, 302)
        dc_residual2 = self.residua2(dc_residual1)  # (2, 64, 400, 302)
        dc_residual3 = self.residua3(dc_residual2)  # (2, 64, 400, 302)
        dc_capture = dc_residual3[:, :, 1:1 + 201, 1:1 + 301].contiguous()
        dc2_final = self.dc2_final(dc_capture)  # (2, 1,  201, 301)

        return dc2_final

if __name__ == '__main__':
    model = TU_Net_SEG(n_classes=1, in_channels=29, is_deconv=True, is_batchnorm=True)
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    model.to(device)
    from torchsummary import summary
    summary(model, input_size=[(29, 400, 301)])
    from torchstat import stat
    stat(model, (29, 400, 301))   # Total params: 9,530,727  Total Flops: 58.05GFlops
