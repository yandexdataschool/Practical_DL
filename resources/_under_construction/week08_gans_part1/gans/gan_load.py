import numpy as np
import torch
from torch import nn

from gans.models.BigGAN import BigGAN
from gans.models.StyleGAN2.model import Generator as StyleGAN2Generator


class ConditionedBigGAN(nn.Module):
    def __init__(self, big_gan, target_classes=(239)):
        super(ConditionedBigGAN, self).__init__()
        self.big_gan = big_gan

        self.set_classes(target_classes)
        self.dim_z = self.big_gan.dim_z

    def set_classes(self, target_classes):
        self.target_classes = nn.Parameter(torch.tensor(target_classes, dtype=torch.int64),
                                           requires_grad=False)

    def mixed_classes(self, batch_size):
        if len(self.target_classes.data.shape) == 0:
            return self.target_classes.repeat(batch_size).cuda()
        else:
            return torch.from_numpy(
                np.random.choice(self.target_classes.cpu(), [batch_size])).cuda()

    def forward(self, z, classes=None):
        if classes is None:
            classes = self.mixed_classes(z.shape[0]).to(z.device)
        return self.big_gan(z, self.big_gan.shared(classes))


class StyleGAN2Wrapper(nn.Module):
    def __init__(self, g, shift_in_w):
        super(StyleGAN2Wrapper, self).__init__()
        self.style_gan2 = g
        self.dim_z = 512
        self.dim_shift = self.style_gan2.style_dim if shift_in_w else self.dim_z
        self.shift_in_w = shift_in_w

    def forward(self, input, w_space=False, noise=None):
        if not isinstance(input, list):
            input = [input]
        return self.style_gan2(input, input_is_latent=w_space, noise=noise)[0]


def make_biggan_config(resolution):
    attn_dict = {128: '64', 256: '128', 512: '64'}
    dim_z_dict = {128: 120, 256: 140, 512: 128}
    config = {
        'G_param': 'SN', 'D_param': 'SN',
        'G_ch': 96, 'D_ch': 96,
        'D_wide': True, 'G_shared': True,
        'shared_dim': 128, 'dim_z': dim_z_dict[resolution],
        'hier': True, 'cross_replica': False,
        'mybn': False, 'G_activation': nn.ReLU(inplace=True),
        'G_attn': attn_dict[resolution],
        'norm_style': 'bn',
        'G_init': 'ortho', 'skip_init': True, 'no_optim': True,
        'G_fp16': False, 'G_mixed_precision': False,
        'accumulate_stats': False, 'num_standing_accumulations': 16,
        'G_eval_mode': True,
        'BN_eps': 1e-04, 'SN_eps': 1e-04,
        'num_G_SVs': 1, 'num_G_SV_itrs': 1, 'resolution': resolution,
        'n_classes': 1000}
    return config


def make_big_gan(weights, target_classes=None, resolution=128, n_classes=1000):
    config = make_biggan_config(resolution)
    config['n_classes'] = n_classes
    G = BigGAN.Generator(**config)
    G.load_state_dict(torch.load(weights, map_location=torch.device('cpu')), strict=False)

    if target_classes is None:
        target_classes = np.arange(0, n_classes, 1)
    return ConditionedBigGAN(G, target_classes).cuda().eval()


def make_stylegan2(resolution, weights, shift_in_w=True, target_key='g_ema', g_kwargs={}):
    G = StyleGAN2Generator(resolution, 512, 8, **g_kwargs)
    G.load_state_dict(torch.load(weights)[target_key] if target_key is not None else \
                      torch.load(weights))
    G.cuda().eval()

    return StyleGAN2Wrapper(G, shift_in_w=shift_in_w)
