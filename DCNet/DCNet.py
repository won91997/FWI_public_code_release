"""
Direct method for reading datasets

@author: Sha-li Fu

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import FocalFrequencyLoss

from DCNet_config import Conv2d, config_model

NORM_LAYERS = { 'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm }

# dcmodule
class DCModule(nn.Module):
    def __init__(self, dc, inchannel, outchannel):
        super(DCModule, self).__init__()
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.conv = ConvBlock(inchannel, outchannel)
        # Depthwise Convolution
        self.conv1 = Conv2d(dc, inchannel, inchannel, kernel_size=3, padding=1, groups=inchannel, bias=False)
        self.norm = nn.BatchNorm2d(inchannel)
        self.relu2 = nn.ReLU()
        # Pointwise Convolution
        self.conv2 = nn.Conv2d(inchannel, outchannel, kernel_size=1, padding=0, bias=False)

    def forward(self, x):

        y = self.conv1(x)
        y = self.norm(y)
        y = self.relu2(y)
        y = self.conv2(y)

        if self.inchannel != self.outchannel:
            x =  self.conv(x)
        # skip connection
        y = y + x
        return y

class DCNet(nn.Module):
    def __init__(self, inchannel, dcs, sample_spatial=1.0, output_size=(256, 256), **kwargs):
        super(DCNet, self).__init__()

        filters = [32, 64, 128, 256, 512]
        self.inchannel = inchannel
        self.output_size = output_size
        block_class1 = DCModule
        block_class2 = ConvBlock

        self.block1_1 = block_class2(self.inchannel, filters[1], kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.block1_2 = block_class2(filters[1], filters[1], kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.block1_3 = block_class2(filters[1], filters[1], kernel_size=(3, 1), padding=(1, 0))
        self.block1_4 = block_class2(filters[1], filters[1], kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.block1_5 = block_class2(filters[1], filters[1], kernel_size=(3, 1), padding=(1, 0))

        self.block2_1 = block_class2(filters[1], filters[2], kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.block2_2 = block_class2(filters[2], filters[2], kernel_size=(3, 1), padding=(1, 0))
        self.block2_3 = block_class2(filters[2], filters[2], stride=2)
        self.block2_4 = block_class2(filters[2], filters[1])
        self.block2_right_1 = block_class1(dcs[0], filters[2], filters[2])
        self.block2_right_2 = block_class1(dcs[1], filters[2], filters[2])
        self.block2_right_3 = block_class1(dcs[2], filters[2], filters[2])

        self.block3_1 = block_class2(filters[2], filters[3], stride=2)
        self.block3_2 = block_class2(filters[3], filters[2])
        self.block3_right_1 = block_class1(dcs[0], filters[3], filters[3])
        self.block3_right_2 = block_class1(dcs[1], filters[3], filters[3])
        self.block3_right_3 = block_class1(dcs[2], filters[3], filters[3])

        self.block4_1 = block_class2(filters[3], filters[3], stride=2)
        self.block4_2 = block_class2(filters[3], filters[2])
        self.block4_right_1 = block_class1(dcs[0], filters[3], filters[3])
        self.block4_right_2 = block_class1(dcs[1], filters[3], filters[3])
        self.block4_right_3 = block_class1(dcs[2], filters[3], filters[3])

        self.diBlock0 = block_class2(filters[4], filters[3])
        self.diBlock1 = block_class2(filters[3], filters[2])
        self.diBlock2 = block_class2(filters[2], filters[1])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck = block_class2(filters[3], filters[4], kernel_size=1, padding=0)

        # decoder
        self.deconv1_1 = DeconvBlock(filters[4], filters[4], kernel_size=5)
        self.deconv1_2 = ConvBlock_norm(filters[4], filters[4])
        self.deconv2_1 = DeconvBlock(filters[4], filters[3], kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock_norm(filters[3], filters[3])
        self.deconv3_1 = DeconvBlock(filters[3], filters[2], kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock_norm(filters[2], filters[2])
        self.deconv4_1 = DeconvBlock(filters[2], filters[1], kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock_norm(filters[1], filters[1])
        self.deconv5_1 = DeconvBlock(filters[1], filters[0], kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock_norm(filters[0], filters[0])
        self.deconv6 = ConvBlock_Tanh(filters[0], 1)
        print('initialization done')

    def forward(self, x):
        x1 = self.block1_1(x)
        x1 = self.block1_2(x1)
        x1 = self.block1_3(x1)
        x1 = self.block1_4(x1)
        x1 = self.block1_5(x1)

        x2 = self.block2_1(x1)
        x2 = self.block2_2(x2)
        x2 = self.block2_3(x2)
        x2_0 = self.block2_4(x2)
        x2 = self.block2_right_1(x2)
        x2 = self.block2_right_2(x2)
        x2 = self.block2_right_3(x2)

        x3 = self.block3_1(x2)
        x3_0 = self.block3_2(x3)
        x3 = self.block3_right_1(x3)
        x3 = self.block3_right_2(x3)
        x3 = self.block3_right_3(x3)

        x4 = self.block4_1(x3)
        x4 = self.block4_right_1(x4)
        x4 = self.block4_right_2(x4)
        x4 = self.block4_right_3(x4)

        x = self.global_pool(x4)
        x = self.bottleneck(x)

        # Decoder Part
        x = self.deconv1_1(x)
        x = self.deconv1_2(x)

        x = self.deconv2_1(x)
        offset1 = (x.size()[2] - x4.size()[2])
        offset2 = (x.size()[3] - x4.size()[3])
        padding = [offset2 // 2, (offset2 + 1) // 2, offset1 // 2, (offset1 + 1) // 2]
        outputs1 = F.pad(x4, padding)
        x = torch.cat([outputs1, x], 1)
        x = self.diBlock0(x)
        x = self.deconv2_2(x)

        x = self.deconv3_1(x)
        offset1 = (x.size()[2] - x3_0.size()[2])
        offset2 = (x.size()[3] - x3_0.size()[3])
        padding = [offset2 // 2, (offset2 + 1) // 2, offset1 // 2, (offset1 + 1) // 2]
        outputs1 = F.pad(x3_0, padding)
        x = torch.cat([outputs1, x], 1)
        x = self.diBlock1(x)
        x = self.deconv3_2(x)

        x = self.deconv4_1(x)
        offset1 = (x.size()[2] - x2_0.size()[2])
        offset2 = (x.size()[3] - x2_0.size()[3])
        padding = [offset2 // 2, (offset2 + 1) // 2, offset1 // 2, (offset1 + 1) // 2]
        outputs1 = F.pad(x2_0, padding)
        x = torch.cat([outputs1, x], 1)
        x = self.diBlock2(x)
        x = self.deconv4_2(x)

        x = self.deconv5_1(x)
        x = self.deconv5_2(x)
        x = self.deconv6(x)
        if x.shape[-2:] != tuple(self.output_size):
            x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ConvBlock_norm(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock_norm,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn'):
        super(DeconvBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn'):
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DCModel(nn.Module):
    """U-based network for self-reconstruction task"""
    def __init__(self, output_size=(256, 256), **kwargs):
        super(DCModel, self).__init__()
        self.dcs = config_model("cv-3")
        self.net = DCNet(5, self.dcs, output_size=output_size)

    def forward(self, x, label_dsp_dim):
        x = self.net(x)
        return  x

class LossDCNet:
    def     __init__(self, weights = [1, 1]):
        '''
        Define the loss function of DCNet
        :param weights:         The weights of the two decoders in the calculation of the loss value.
        '''
        # mse
        self.criterion1 = nn.MSELoss()
        # focal loss
        self.focalLoss = FocalFrequencyLoss()

        self.weights = weights


    def __call__(self, outputs1, targets1):
        '''

        :param outputs1: Output of the real image
        :param targets1: Output of the predict image
        :return:
        '''
        mse = self.criterion1(outputs1, targets1)
        print('MSELoss:{:.12f}'.format(mse.item()), end='\t')

        loss = self.focalLoss(outputs1, targets1)
        print('FocalLoss:{:.12f}'.format(loss.item()))

        criterion = (self.weights[0] * mse + self.weights[1] * loss)

        return criterion


model_dict = {
    'DCNet': DCModel
}

