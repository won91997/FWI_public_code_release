from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

NORM_LAYERS = {"bn": nn.BatchNorm2d, "in": nn.InstanceNorm2d, "ln": nn.LayerNorm, "gn": lambda c: nn.GroupNorm(min(32, c), c)}


class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm="bn", relu_slop=0.2, dropout=None):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=False))  # inplace=False for DDP compatibility
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm="bn"):
        super(ConvBlock_Tanh, self).__init__()
        layers = [
            nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm="bn"):
        super(DeconvBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(
                in_channels=in_fea,
                out_channels=out_fea,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            )
        ]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=False))  # inplace=False for DDP compatibility
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512,
                 sample_spatial=1.0, output_size=(256, 256), **kwargs):
        super(Generator, self).__init__()
        self.output_size = output_size
        self.convblock1_1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock1_2 = ConvBlock(dim1, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 使用 GroupNorm：bottleneck 输出 1x1，InstanceNorm 要求 spatial>1；GroupNorm 支持 1x1 且 DDP 下单样本安全
        self.bottleneck = ConvBlock(dim4, dim5, kernel_size=1, padding=0, norm="gn")

        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6_1 = DeconvBlock(dim1, 16, kernel_size=4, stride=(2, 1), padding=1)
        self.deconv6_2 = ConvBlock(16, 16)
        self.deconv7 = ConvBlock_Tanh(16, 1)

    def _ckpt(self, fn, x):
        """训练时用 gradient checkpoint 省显存（T4 16GB batch=2 易 OOM），推理时不启用"""
        if self.training:
            return checkpoint(fn, x, use_reentrant=False)
        return fn(x)

    def forward(self, x):
        # Encoder：大空间尺寸耗显存，checkpoint 可省 50%+ 激活内存
        x = self._ckpt(self.convblock1_1, x)
        x = self._ckpt(self.convblock1_2, x)
        x = self._ckpt(self.convblock2_1, x)
        x = self._ckpt(self.convblock2_2, x)
        x = self._ckpt(self.convblock3_1, x)
        x = self._ckpt(self.convblock3_2, x)
        x = self._ckpt(self.convblock4_1, x)
        x = self._ckpt(self.convblock4_2, x)
        x = self._ckpt(self.convblock5_1, x)
        x = self._ckpt(self.convblock5_2, x)
        x = self._ckpt(self.convblock6_1, x)
        x = self._ckpt(self.convblock6_2, x)
        x = self._ckpt(self.convblock7_1, x)
        x = self._ckpt(self.convblock7_2, x)
        x = self.global_pool(x)
        x = self.bottleneck(x)

        # Decoder
        x = self._ckpt(self.deconv1_1, x)
        x = self._ckpt(self.deconv1_2, x)
        x = self._ckpt(self.deconv2_1, x)
        x = self._ckpt(self.deconv2_2, x)
        x = self._ckpt(self.deconv3_1, x)
        x = self._ckpt(self.deconv3_2, x)
        x = self._ckpt(self.deconv4_1, x)
        x = self._ckpt(self.deconv4_2, x)
        x = self._ckpt(self.deconv5_1, x)
        x = self._ckpt(self.deconv5_2, x)
        x = self._ckpt(self.deconv6_1, x)
        x = self._ckpt(self.deconv6_2, x)
        x = self.deconv7(x)
        if x.shape[-2:] != tuple(self.output_size):
            x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        return x


class Discriminator(nn.Module):
    """判别器。norm='in' 时使用 InstanceNorm，与 WGAN-GP 的 create_graph=True 兼容；norm='bn' 为默认。"""
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, norm="in", **kwargs):
        super(Discriminator, self).__init__()
        self.convblock1_1 = ConvBlock(1, dim1, stride=2, norm=norm)
        self.convblock1_2 = ConvBlock(dim1, dim1, norm=norm)
        self.convblock2_1 = ConvBlock(dim1, dim2, stride=2, norm=norm)
        self.convblock2_2 = ConvBlock(dim2, dim2, norm=norm)
        self.convblock3_1 = ConvBlock(dim2, dim3, stride=2, norm=norm)
        self.convblock3_2 = ConvBlock(dim3, dim3, norm=norm)
        self.convblock4_1 = ConvBlock(dim3, dim4, stride=2, norm=norm)
        self.convblock4_2 = ConvBlock(dim4, dim4, norm=norm)
        self.convblock5 = ConvBlock(dim4, 1, kernel_size=5, padding=0, norm=norm)

    def _ckpt(self, fn, x):
        if self.training:
            return checkpoint(fn, x, use_reentrant=False)
        return fn(x)

    def forward(self, x):
        x = self._ckpt(self.convblock1_1, x)
        x = self._ckpt(self.convblock1_2, x)
        x = self._ckpt(self.convblock2_1, x)
        x = self._ckpt(self.convblock2_2, x)
        x = self._ckpt(self.convblock3_1, x)
        x = self._ckpt(self.convblock3_2, x)
        x = self._ckpt(self.convblock4_1, x)
        x = self._ckpt(self.convblock4_2, x)
        x = self.convblock5(x)
        x = x.view(x.shape[0], -1)
        return x
