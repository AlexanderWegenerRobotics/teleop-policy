import math

import torch
import torch.nn as nn
import torchvision
from torchvision.ops import FrozenBatchNorm2d


class ResNet18Backbone(nn.Module):
    """ImageNet-pretrained ResNet-18 with frozen BatchNorm, without avgpool and fc."""

    def __init__(self):
        """Load pretrained ResNet-18, freeze BatchNorm and drop the classification head."""
        super().__init__()
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
                                          norm_layer=FrozenBatchNorm2d)
        self.body = nn.Sequential(*list(net.children())[:-2])
        self.num_channels = 512
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        """Images [B, 3, H, W] in [0, 1] to features [B, 512, H/32, W/32]."""
        return self.body((x - self.mean) / self.std)


def sine_pos_embed_2d(h, w, dim, device, temperature=10000):
    """Fixed 2D sinusoidal position embedding [dim, h, w]."""
    pos_dim = dim // 2
    y_embed = torch.arange(h, dtype=torch.float32, device=device).unsqueeze(1).expand(h, w)
    x_embed = torch.arange(w, dtype=torch.float32, device=device).unsqueeze(0).expand(h, w)
    dim_t = torch.arange(pos_dim, dtype=torch.float32, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / pos_dim)
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
    pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)
    return torch.cat([pos_y, pos_x], dim=-1).permute(2, 0, 1)


def sine_pos_embed_1d(n, dim, device, temperature=10000.0):
    """Fixed 1D sinusoidal position embedding [n, dim]."""
    position = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(temperature) / dim))
    pe = torch.zeros(n, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
