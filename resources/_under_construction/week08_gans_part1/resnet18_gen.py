from collections import namedtuple
from torch import nn
import numpy as np


ResNetGenConfig = namedtuple('ResNetGenConfig', ['channels', 'seed_dim'])
RES_GEN_CONFIGS = {
    32: ResNetGenConfig([256, 256, 256, 256], 4),
    48: ResNetGenConfig([512, 256, 128, 64], 6),
    64: ResNetGenConfig([16 * 64, 8 * 64, 4 * 64, 2 * 64, 64], 4),
    128: ResNetGenConfig([16 * 64, 16 * 64, 8 * 64, 4 * 64, 2 * 64, 64], 4),
    256: ResNetGenConfig([16 * 64, 16 * 64, 8 * 64, 8 * 64, 4 * 64, 2 * 64, 64], 4)
}


class Reshape(nn.Module):
    def __init__(self, target_shape):
        super(Reshape, self).__init__()
        self.target_shape = target_shape

    def forward(self, input):
        return input.view(self.target_shape)


class ResBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)

        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            self.conv2
            )

        if in_channels == out_channels:
            self.bypass = nn.Upsample(scale_factor=2)
        else:
            self.bypass = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
            )
            nn.init.xavier_uniform_(self.bypass[1].weight.data, 1.0)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class GenWrapper(nn.Module):
    def __init__(self, model, out_img_shape, distribution):
        super(GenWrapper, self).__init__()

        self.model = model
        self.out_img_shape = out_img_shape
        self.distribution = distribution

    def cuda(self, device=None):
        super(GenWrapper, self).cuda(device)
        self.distribution.cuda()

    def forward(self, batch_size):
        img = self.model(self.distribution(batch_size))
        img = img.view(img.shape[0], *self.out_img_shape)
        return img


def make_resnet_generator(resnet_gen_config, channels=3, dim_z=128):
    def make_dense():
        dense = nn.Linear(dim_z, resnet_gen_config.seed_dim**2 * resnet_gen_config.channels[0])
        nn.init.xavier_uniform_(dense.weight.data, 1.)
        return dense

    def make_final():
        final = nn.Conv2d(resnet_gen_config.channels[-1], channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(final.weight.data, 1.)
        return final

    model_channels = resnet_gen_config.channels

    input_layers = [
        make_dense(),
        Reshape([-1, model_channels[0], 4, 4])
    ]
    res_blocks = [
        ResBlockGenerator(model_channels[i], model_channels[i + 1])
        for i in range(len(model_channels) - 1)
    ]
    out_layers = [
        nn.BatchNorm2d(model_channels[-1]),
        nn.ReLU(inplace=True),
        make_final(),
        nn.Tanh()
    ]

    model = nn.Sequential(*(input_layers + res_blocks + out_layers))
    model.dim_z = dim_z

    return model
