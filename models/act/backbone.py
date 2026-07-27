# Per-camera ResNet-18 feature extractor + sinusoidal position embeddings (DETR-style).

import math

import torch
import torch.nn as nn
import torchvision


class ResNet18Backbone(nn.Module):
    """ImageNet-pretrained ResNet-18, truncated before avgpool/fc -> [B,512,H/32,W/32]."""

    def __init__(self):
        super().__init__()
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.body = nn.Sequential(*list(net.children())[:-2])
        self.num_channels = 512

    def forward(self, x):
        return self.body(x)


def sine_pos_embed_2d(h, w, dim, device, temperature=10000):
    """Fixed 2D sinusoidal position embedding for a feature map -> [dim, h, w]."""
    pos_dim = dim // 2
    y_embed = torch.arange(h, dtype=torch.float32, device=device).unsqueeze(1).expand(h, w)
    x_embed = torch.arange(w, dtype=torch.float32, device=device).unsqueeze(0).expand(h, w)
    dim_t = torch.arange(pos_dim, dtype=torch.float32, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / pos_dim)
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
    pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)
    return torch.cat([pos_y, pos_x], dim=-1).permute(2, 0, 1)  # [dim,h,w]


def sine_pos_embed_1d(n, dim, device, temperature=10000.0):
    """Fixed 1D sinusoidal position embedding for a token sequence -> [n, dim]."""
    position = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(temperature) / dim))
    pe = torch.zeros(n, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
