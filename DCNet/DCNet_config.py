"""
Direct method for reading datasets

@author: Sha-li Fu

"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):
    def __init__(self, dc, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.dc = dc

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):

        return self.dc(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def createConvFunc(op_type):
    assert op_type in ['cv', 'od', 'cd'], 'unknown op type: %s' % str(op_type)
    if op_type == 'cv':
        return F.conv2d

    if op_type == 'cd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            assert padding == dilation, 'padding for cd_conv set wrong'

            weights_c = weights.sum(dim=[2, 3], keepdim=True)
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y - yc
        return func
    elif op_type == 'od':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for od_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for od_conv should be 3x3'
            assert padding == dilation, 'padding for od_conv set wrong'
            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            new_weights = torch.zeros_like(weights)
            new_weights[:, :, 0] = weights[:, :, 0]
            new_weights[:, :, 1] = weights[:, :, 1]
            new_weights[:, :, 2] = weights[:, :, 2]
            new_weights[:, :, 3] = weights[:, :, 3]
            new_weights[:, :, 4] = weights[:, :, 4] - weights[:, :, 0]
            new_weights[:, :, 5] = weights[:, :, 5] - weights[:, :, 1] - weights[:, :, 2]
            new_weights[:, :, 6] = weights[:, :, 6]
            new_weights[:, :, 7] = weights[:, :, 7] - weights[:, :, 6]
            new_weights[:, :, 8] = -weights[:, :, 7] - weights[:, :, 6] - weights[:, :, 5]
            weights_conv = new_weights.view(shape)

            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func
    else:
        print('impossible to be here unless you force that')
        return None


nets = {
    'baseline': {
        'layer0':  'cv',
        'layer1':  'od',
        'layer2':  'cd',
        },
    'cv-3': {
        'layer0':  'cv',
        'layer1':  'cv',
        'layer2':  'cv',
        },
    'od-3': {
        'layer0':  'od',
        'layer1':  'od',
        'layer2':  'od',
        },
    'cd-3': {
        'layer0':  'cd',
        'layer1':  'cd',
        'layer2':  'cd',
        }
    }

def config_model(model):
    model_options = list(nets.keys())
    assert model in model_options, \
        'unrecognized model, please choose from %s' % str(model_options)

    print(str(nets[model]))

    dcs = []
    for i in range(3):
        layer_name = 'layer%d' % i
        op = nets[model][layer_name]
        dcs.append(createConvFunc(op))

    return dcs

def config_model_converted(model):
    model_options = list(nets.keys())
    assert model in model_options, \
        'unrecognized model, please choose from %s' % str(model_options)

    print(str(nets[model]))

    dcs = []
    for i in range(3):
        layer_name = 'layer%d' % i
        op = nets[model][layer_name]
        dcs.append(op)

    return dcs

