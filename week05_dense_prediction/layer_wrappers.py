import torch
import torch.nn as nn
from torch.autograd import Variable

def init_layer(layer, weight_init=None, bias_init=None):
    if weight_init is not None:
        layer.weight.data = torch.FloatTensor(weight_init)
    if bias_init is not None:
        layer.bias.data = torch.FloatTensor(bias_init)
    return layer

def Linear(in_features, out_features, bias=True, weight_init=None, bias_init=None):
    layer = nn.Linear(in_features, out_features, bias)
    return init_layer(layer, weight_init, bias_init)

def Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
           bias=True, weight_init=None, bias_init=None):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation,
                      groups, bias)
    return init_layer(layer, weight_init, bias_init)

def BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, weight_init=None,
                bias_init=None):
    layer = nn.BatchNorm1d(num_features, eps, momentum, affine)
    return init_layer(layer, weight_init, bias_init)

def BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, weight_init=None,
                bias_init=None):
    layer = nn.BatchNorm2d(num_features, eps, momentum, affine)
    return init_layer(layer, weight_init, bias_init)

