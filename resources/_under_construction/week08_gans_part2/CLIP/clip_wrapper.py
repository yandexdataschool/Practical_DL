from tqdm.auto import tqdm

import torch
from torch import nn
from torch.nn import functional as F

from CLIP import clip


def project(y, X):
    y = y.to(torch.float)
    X = (X.T).to(torch.float)  # make sure basis is columns

    Q, R = torch.linalg.qr(X)
    beta = torch.triangular_solve(Q.T @ y.unsqueeze(-1), R).solution
    proj = (X @ beta)
    return proj.squeeze(-1)


def project_points_to_planes(y, X):
    y = y.to(torch.float).unsqueeze(1)
    X = (X.transpose(-1, -2)).to(torch.float)  # make sure basis is the columns

    Q, R = torch.linalg.qr(X)
    Q = Q.unsqueeze(1)

    # solve A * beta = b
    b = Q.transpose(-1, -2) @ y.unsqueeze(-1)
    b.squeeze_(1)

    beta = torch.triangular_solve(b, R).solution
    proj = (X @ beta)
    return proj.squeeze(-1)


def orthogonalize(vs):  # vs ~ [batch, n, N]
    q = torch.linalg.qr(vs.float().transpose(-1, -2), 'reduced')[0]
    return q.transpose(-1, -2)


def subspaces_principal_angle(vs, us):  # vs, us ~ [batch, n, N]
    vs, us = orthogonalize(vs).transpose(-1, -2), orthogonalize(us).transpose(-1, -2)
    M = vs.transpose(-1, -2) @ us
    svd = torch.svd(M)[1]
    return svd[:, 0] if len(svd.shape) == 2 else svd[0]


def form_subspaces(embeds, factors):
    embeds = embeds.unsqueeze(1)
    if factors is None:
        return embeds

    factors = factors[None].repeat([len(embeds), 1, 1])
    return torch.cat([embeds, factors], dim=1)


def factorized_pairwise_similarity(xs, factors_x, ys, factors_y):
    return subspaces_principal_angle(form_subspaces(xs, factors_x),
                                     form_subspaces(ys, factors_y))


class CLIPWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.perceptor, self.clip_preprocess = clip.load('RN50x16', jit=False)

    def preprocess_clip(self, imgs):
        size = self.clip_preprocess.transforms[0].size
        imgs = self.clip_preprocess.transforms[-1](0.5 * (imgs + 1.))
        return F.interpolate(imgs, (size, size), mode='nearest')

    def clip_embed(self, imgs):
        return self.perceptor.encode_image(self.preprocess_clip(imgs))

    def forward(self, x):
        return self.clip_embed(x)

    def text_embed(self, *texts):
        tokens = torch.cat([clip.tokenize(t) for t in texts])
        return self.perceptor.encode_text(tokens.cuda()).cuda()

    @torch.no_grad()
    def calculate_embeddings(self, dataloader, verbose=True):
        embeds = []
        for sample in tqdm(dataloader) if verbose else dataloader:
            embeds.append(self.clip_embed(sample.cuda()).cpu())
        return torch.cat(embeds)
