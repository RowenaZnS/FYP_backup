"""
Age-conditioned generation building blocks:
- Target age encoding
- Attention fusion (FiLM + cross-attention + spatial mask)
- Convolutional decoder (generator)
- Patch discriminator
- VGG perceptual loss (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class TargetAgeEncoder(nn.Module):
    """Maps scalar target age (normalized, e.g. age/100) to z_age."""

    def __init__(self, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, age_norm: torch.Tensor) -> torch.Tensor:
        if age_norm.dim() == 1:
            age_norm = age_norm.unsqueeze(-1)
        return self.net(age_norm)


class AttentionFusion(nn.Module):
    """
    Fuses identity feature z_id with target-age code z_age.
    - Channel-wise FiLM modulation on z_id
    - Cross-attention: query = z_id, key/value from z_age
    - Spatial latent (7x7) with learnable spatial mask A
    """

    def __init__(self, dim_id: int = 512, dim_age: int = 128, latent_hw: int = 7, num_heads: int = 8):
        super().__init__()
        self.dim_id = dim_id
        self.latent_hw = latent_hw
        self.latent_spatial = latent_hw * latent_hw

        self.film = nn.Sequential(
            nn.Linear(dim_age, dim_id * 2),
        )
        self.age_kv = nn.Linear(dim_age, dim_id * 2)
        self.cross_attn = nn.MultiheadAttention(dim_id, num_heads, batch_first=True, dropout=0.1)

        self.mask_head = nn.Sequential(
            nn.Linear(dim_id + dim_age, dim_id),
            nn.ReLU(inplace=True),
            nn.Linear(dim_id, self.latent_spatial),
        )

        self.to_spatial = nn.Linear(dim_id, dim_id * self.latent_spatial)
        self.out_norm = nn.LayerNorm(dim_id)

    def forward(self, z_id: torch.Tensor, z_age: torch.Tensor):
        b = z_id.size(0)
        gamma, beta = self.film(z_age).chunk(2, dim=-1)
        z_mod = z_id * (1 + torch.tanh(gamma)) + beta

        kv = self.age_kv(z_age)
        k, v = kv.chunk(2, dim=-1)
        z_q = z_mod.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        attn_out, _ = self.cross_attn(z_q, k, v)
        z_f = self.out_norm(z_mod + attn_out.squeeze(1))

        spatial_flat = self.to_spatial(z_f)
        h = spatial_flat.view(b, self.dim_id, self.latent_spatial)
        h = h.view(b, self.dim_id, self.latent_hw, self.latent_hw)

        mask_logits = self.mask_head(torch.cat([z_id, z_age], dim=-1))
        a = torch.sigmoid(mask_logits.view(b, 1, self.latent_hw, self.latent_hw))
        h = h * a
        return h, a


class FaceGenerator(nn.Module):
    """Decode fused spatial latent (B, C, 7, 7) to RGB image (B, 3, 112, 112)."""

    def __init__(self, in_ch: int = 512, out_size: int = 112):
        super().__init__()
        c = in_ch
        layers = []
        # 7 -> 14 -> 28 -> 56 -> 112
        # Use 4 upsample steps
        chs = [c, 256, 128, 64, 32]
        for i in range(len(chs) - 1):
            layers.append(nn.ConvTranspose2d(chs[i], chs[i + 1], 4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(chs[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.dec = nn.Sequential(*layers)
        self.out_conv = nn.Conv2d(chs[-1], 3, kernel_size=3, padding=1)
        self.out_size = out_size

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.dec(h)
        x = self.out_conv(x)
        x = torch.tanh(x)
        if x.shape[-1] != self.out_size:
            x = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        return x


class PatchDiscriminator(nn.Module):
    """Lightweight patch-style discriminator on 112x112 RGB images."""

    def __init__(self, in_ch: int = 3, base: int = 64):
        super().__init__()
        def block(cin, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 4, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.model = nn.Sequential(
            block(in_ch, base, 2),
            block(base, base * 2, 2),
            block(base * 2, base * 4, 2),
            block(base * 4, base * 8, 2),
            nn.Conv2d(base * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class VGGPerceptualLoss(nn.Module):
    """Expects inputs in [-1, 1]; internally maps to ImageNet mean/std for VGG."""

    def __init__(self, layer_indices=(8, 15, 22), weights: tuple = None):
        super().__init__()
        try:
            try:
                w = models.VGG16_BN_Weights.IMAGENET1K_V1
                vgg = models.vgg16_bn(weights=w).features
            except Exception:
                vgg = models.vgg16_bn(pretrained=True).features
        except Exception:
            vgg = models.vgg16_bn(weights=None).features
        self.vgg = vgg
        self.layer_indices = list(layer_indices)
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if weights is None:
            weights = tuple(1.0 / len(layer_indices) for _ in layer_indices)
        self.weights = weights

    def _to_vgg_input(self, x: torch.Tensor) -> torch.Tensor:
        x01 = (x + 1.0) * 0.5
        return (x01 - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self._to_vgg_input(x)
        y = self._to_vgg_input(y)
        loss = 0.0
        hx, hy = x, y
        for i, layer in enumerate(self.vgg):
            hx = layer(hx)
            hy = layer(hy)
            if i in self.layer_indices:
                idx = self.layer_indices.index(i)
                loss = loss + self.weights[idx] * F.l1_loss(hx, hy)
        return loss
