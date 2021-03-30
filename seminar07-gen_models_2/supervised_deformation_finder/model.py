import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class CelebaRegressor(nn.Module):
    def __init__(self, discrete_cardinality, cont_cardinality, f_size=512, pretrained=False):
        super().__init__()
        encoder = resnet18(pretrained=pretrained)
        encoder.fc = nn.Sequential(nn.ReLU(), nn.Linear(512, f_size), nn.ReLU())
        self.backbone = encoder

        self.classification_heads = nn.ModuleList(
            [nn.Linear(f_size, d) for d in discrete_cardinality]
        )
        self.continuous_head = nn.Linear(f_size, cont_cardinality)  # 10

    def features(self, x):
        return self.backbone(x)

    def forward(self, x):
        features = self.backbone(x)
        d_features = [head(features) for head in self.classification_heads]
        return d_features


class CelebaAttributeClassifier(nn.Module):
    def __init__(self, attribute_name, ckpt_path, device='cuda'):
        super().__init__()

        with open(f"{os.path.dirname(os.path.abspath(__file__))}/celeba_attributes.txt", "r") as f:
            attributes_list = f.readline().strip().split()

        self.output_ix = attributes_list.index(attribute_name)

        self.regressor = CelebaRegressor(
            discrete_cardinality=[2] * 40, cont_cardinality=10, f_size=512
        ).to(device)
        self.regressor.load_state_dict(
            torch.load(ckpt_path, map_location='cpu')['model_state_dict']
        )

    def forward(self, x):
        return self.regressor(x)[self.output_ix]

    def get_probs(self, x):
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        return probs
