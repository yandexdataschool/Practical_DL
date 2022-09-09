import torch
import torch.nn.functional as F

mean_imagenet = torch.FloatTensor([0.485, 0.456, 0.406])[:, None, None]
std_imagenet = torch.FloatTensor([0.229, 0.224, 0.225])[:, None, None]


def prepare_generator_output_for_celeba_regressor(imgs):
    imgs = torch.clamp(imgs, -1, 1)
    imgs = F.interpolate(imgs, 224)
    imgs = (imgs + 1) / 2
    imgs = (imgs - mean_imagenet.to(imgs.device)) / std_imagenet.to(imgs.device)
    return imgs
