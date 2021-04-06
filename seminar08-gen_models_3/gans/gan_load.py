import numpy as np
import torch
from torch import nn

from gans.models.StyleGAN2.model import Generator as StyleGAN2Generator


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


def make_stylegan2(resolution, weights, shift_in_w=True, target_key='g_ema', g_kwargs={}):
    G = StyleGAN2Generator(resolution, 512, 8, **g_kwargs)
    G.load_state_dict(torch.load(weights)[target_key] if target_key is not None else \
                      torch.load(weights))
    G.cuda().eval()

    return StyleGAN2Wrapper(G, shift_in_w=shift_in_w)
