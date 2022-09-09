import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_tools.modules import HiddenModule


def expand_to_bach(value, batch_size, target_type):
    try:
        assert value.shape[0] == batch_size, 'batch size is not equal to the tensor size'
    except Exception:
        value = value * torch.ones(batch_size, dtype=target_type)
    return value.to(target_type)


def apply_modulated_conv_2d(input, style, mc2d, weight):
    batch, in_channel, height, width = input.shape
    style = mc2d.modulation(style).view(batch, 1, in_channel, 1, 1)
    weight = mc2d.scale * weight * style

    if mc2d.demodulate:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(batch, mc2d.out_channel, 1, 1, 1)

    weight = weight.view(batch * mc2d.out_channel, in_channel, mc2d.kernel_size, mc2d.kernel_size)

    if mc2d.upsample:
        input = input.view(1, batch * in_channel, height, width)
        weight = weight.view(
            batch, mc2d.out_channel, in_channel, mc2d.kernel_size, mc2d.kernel_size
        )
        weight = weight.transpose(1, 2).reshape(
            batch * in_channel, mc2d.out_channel, mc2d.kernel_size, mc2d.kernel_size
        )
        out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, mc2d.out_channel, height, width)
        out = mc2d.blur(out)

    elif mc2d.downsample:
        input = mc2d.blur(input)
        _, _, height, width = input.shape
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, mc2d.out_channel, height, width)

    else:
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=mc2d.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, mc2d.out_channel, height, width)

    return out


def make_deformable(model, layer, deformable_class, **kwargs):
    model.style_gan2.convs[layer].conv = deformable_class(
        model.style_gan2.convs[layer].conv, **kwargs)

    deformable_conv = model.style_gan2.convs[layer].conv
    model.style_gan2.convs[layer].add_module('deformable_conv', deformable_conv)
    try:
        model.deformable_convs.append(deformable_conv)
    except AttributeError:
        model.deformable_convs = [deformable_conv]
    return model


class DeformableModulatedConv2d(nn.Module):
    def __init__(self, conv_to_deform):
        super(DeformableModulatedConv2d, self).__init__()

        for key, val in conv_to_deform.__dict__.items():
            setattr(self, key, val)
        self.shifts = torch.zeros_like(self.weight)

    def forward(self, x, style):
        return apply_modulated_conv_2d(x, style, self, self.weight + self.shifts)


class DeformableSubspaceModulatedConv2d(nn.Module):
    def __init__(self, conv_to_deform, basis_vectors, directions_count):
        super(DeformableSubspaceModulatedConv2d, self).__init__()

        for key, val in conv_to_deform.__dict__.items():
            setattr(self, key, val)
        self.basis_vectors = basis_vectors
        self.is_deformated = False

        assert self.basis_vectors[0].shape == self.weight.shape, \
            f'unconsisted basis and weight {self.basis_vectors.shape[1:]} != {self.weight.shape}'

        basis_dim = len(self.basis_vectors)
        self.shifts_coords = nn.Parameter(torch.randn((directions_count, basis_dim)))

    def weight_shifts(self, batch):
        # expand scalar shift params if required
        basis_size = len(self.basis_vectors)
        batch_directions = expand_to_bach(self.batch_directions, batch, torch.long).cuda()
        batch_shifts = expand_to_bach(self.batch_shifts, batch, torch.float32).cuda()

        # deformation
        batch_weight_delta = torch.stack(batch * [self.basis_vectors], dim=0)
        # (batch_size, basis_size, *weight.shape)
        batch_basis_coefs = self.shifts_coords[batch_directions]
        # (batch_size, basis_size)

        batch_weight_delta = batch_weight_delta.view(batch, basis_size, -1)
        # (batch_size, basis_size, -1)
        batch_basis_coefs = batch_basis_coefs.unsqueeze(-1)
        # (batch_size, basis_size, -1)

        batch_weight_delta = (batch_weight_delta * batch_basis_coefs).sum(dim=1)
        # (batch_size, -1)
        batch_weight_delta = F.normalize(batch_weight_delta, p=2, dim=1)
        batch_weight_delta *= batch_shifts[:, None]
        # (batch_size, -1)

        batch_weight_delta = batch_weight_delta.view(batch, *self.weight.shape[-4:])
        # (batch_size, c_out, c_in, k_x, k_y)
        return batch_weight_delta

    def forward(self, x, style):
        weight = self.weight
        if self.is_deformated:
            weight = weight + self.weight_shifts(x.shape[0])
        return apply_modulated_conv_2d(x, style, self, weight)


class DeformableSVDModulatedConv2d(nn.Module):
    def __init__(self, conv_to_deform, directions_count):
        super(DeformableSVDModulatedConv2d, self).__init__()
        for key, val in conv_to_deform.__dict__.items():
            setattr(self, key, val)
        self.is_deformated = False

        weight_matrix = \
            conv_to_deform.weight.cpu().detach().numpy().reshape((conv_to_deform.weight.shape[-4:]))
        c_out, c_in, k_x, k_y = weight_matrix.shape
        weight_matrix = np.transpose(weight_matrix, (2, 3, 1, 0))
        weight_matrix = np.reshape(weight_matrix, (k_x * k_y * c_in, c_out))

        u, s, vh = np.linalg.svd(weight_matrix, full_matrices=False)
        u = torch.FloatTensor(u).cuda()
        vh = torch.FloatTensor(vh).cuda()

        self.u = nn.Parameter(u, requires_grad=False)
        self.vh = nn.Parameter(vh, requires_grad=False)

        self.direction_to_eigenvalues_delta = nn.Parameter(
            torch.randn(directions_count, len(s)), requires_grad=True)

    def weight_shifts(self, batch):
        # expand scalar shift params if required
        batch_directions = expand_to_bach(self.batch_directions, batch, torch.long).cuda()
        batch_shifts = expand_to_bach(self.batch_shifts, batch, torch.float32).cuda()

        batch_eigenvalues_delta = self.direction_to_eigenvalues_delta[batch_directions]
        # (batch, len(s))
        batch_weight_delta = self.u @ torch.diag_embed(batch_eigenvalues_delta) @ self.vh

        c_out, c_in, k_x, k_y = self.weight.shape[-4:]
        batch_weight_delta = F.normalize(batch_weight_delta.view(batch, -1), dim=1, p=2)
        batch_weight_delta = batch_weight_delta.view(batch, k_x, k_y, c_in, c_out)
        batch_weight_delta = batch_weight_delta.permute(0, 4, 3, 1, 2)
        assert batch_weight_delta.shape == (batch, c_out, c_in, k_x, k_y)

        batch_weight_delta *= batch_shifts[:, None, None, None, None]
        return batch_weight_delta

    def forward(self, x, style):
        weight = self.weight
        if self.is_deformated:
            weight = weight + self.weight_shifts(x.shape[0])
        return apply_modulated_conv_2d(x, style, self, weight)
