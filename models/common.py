"""Shared building blocks for TAPOR-like and physics hybrids."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(cin, cout, k, s, p, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class ResidualConv(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(ConvBNReLU(ch, ch), ConvBNReLU(ch, ch))

    def forward(self, x):
        return x + self.block(x)


class UNetEncoder(nn.Module):
    def __init__(self, in_ch=1, base=48):
        super().__init__()
        self.enc1 = nn.Sequential(ConvBNReLU(in_ch, base), ResidualConv(base))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), ConvBNReLU(base, base * 2), ResidualConv(base * 2))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), ConvBNReLU(base * 2, base * 4), ResidualConv(base * 4))
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), ConvBNReLU(base * 4, base * 8), ResidualConv(base * 8))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        return e1, e2, e3, b


class InverseUNet(nn.Module):
    """Predict multi-primitive maps at input resolution."""

    def __init__(self, in_ch=1, base=48, out_ch=5):
        super().__init__()
        self.encoder = UNetEncoder(in_ch, base)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec3 = nn.Sequential(ConvBNReLU(base * 8, base * 4), ResidualConv(base * 4))
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = nn.Sequential(ConvBNReLU(base * 4, base * 2), ResidualConv(base * 2))
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = nn.Sequential(ConvBNReLU(base * 2, base), ResidualConv(base))
        self.head = nn.Conv2d(base, out_ch, 1)
        # learnable depth scale/bias toward RealSense meters
        self.depth_scale = nn.Parameter(torch.tensor(0.6))
        self.depth_bias = nn.Parameter(torch.tensor(0.25))

    def _match(self, x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x):
        e1, e2, e3, b = self.encoder(x)
        d3 = self.up3(b)
        d3 = self._match(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self._match(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self._match(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        out = self.head(d1)
        out = self._match(out, x)
        # channels: depth, eps, Tt(C), reflection, mask_logits
        depth = self.depth_bias.abs() + self.depth_scale.abs() * torch.sigmoid(out[:, 0:1])
        depth = depth.clamp(0.05, 1.5)
        eps = torch.sigmoid(out[:, 1:2]) * 0.9 + 0.05
        tt = out[:, 2:3] * 8.0 + 32.0  # hand Celsius near ambient+delta
        refl = F.softplus(out[:, 3:4])
        mask_logits = out[:, 4:5]
        return {
            "depth": depth,
            "emissivity": eps,
            "surface_temp": tt,
            "reflection": refl,
            "mask_logits": mask_logits,
            "feat": d1,
        }


class ThermoSpaceEncoder(nn.Module):
    """Thermo-space encoder producing spatial feature map."""

    def __init__(self, in_ch=1, out_ch=96):
        super().__init__()
        self.stem = nn.Sequential(ConvBNReLU(in_ch, 48), ResidualConv(48))
        self.down = nn.Sequential(nn.MaxPool2d(2), ConvBNReLU(48, 96), ResidualConv(96))
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(96, out_ch),
            ResidualConv(out_ch),
        )
        self.out_ch = out_ch

    def forward(self, x):
        s = self.stem(x)
        d = self.down(s)
        return self.up(d)


class KPFeatureExtractor(nn.Module):
    """Produce 21 joint tokens from a spatial feature or thermal map."""

    def __init__(self, in_ch=1, feat_dim=64):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBNReLU(in_ch, 48),
            ResidualConv(48),
            ConvBNReLU(48, 96),
            nn.AdaptiveAvgPool2d((6, 8)),
        )
        self.proj = nn.Linear(96 * 6 * 8, 21 * feat_dim)
        self.feat_dim = feat_dim
        self.joint_mix = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(feat_dim, feat_dim, 1),
        )
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x):
        b = x.shape[0]
        h = self.backbone(x).flatten(1)
        tokens = self.proj(h).view(b, 21, self.feat_dim)
        tokens = self.joint_mix(tokens.transpose(1, 2)).transpose(1, 2)
        return self.norm(tokens)


class PositionRetrievalDecoder(nn.Module):
    """Cross-attend KP tokens over spatial map → 21x3 joints."""

    def __init__(self, spatial_ch=96, kp_dim=64, heads=4):
        super().__init__()
        self.q = nn.Linear(kp_dim, kp_dim)
        self.k = nn.Conv2d(spatial_ch, kp_dim, 1)
        self.v = nn.Conv2d(spatial_ch, kp_dim, 1)
        self.attn = nn.MultiheadAttention(kp_dim, heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(kp_dim, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(kp_dim, kp_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(kp_dim * 2, kp_dim),
        )
        self.n1 = nn.LayerNorm(kp_dim)
        self.n2 = nn.LayerNorm(kp_dim)
        self.head = nn.Sequential(
            nn.Linear(kp_dim, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, 3),
        )

    def forward(self, spatial: torch.Tensor, kp: torch.Tensor) -> torch.Tensor:
        k = self.k(spatial).flatten(2).transpose(1, 2)  # B,HW,D
        v = self.v(spatial).flatten(2).transpose(1, 2)
        q = self.q(kp)
        loc, _ = self.attn(q, k, v)
        loc = self.n1(loc + kp)
        pose, _ = self.self_attn(loc, loc, loc)
        pose = self.n2(pose + self.ff(pose))
        return self.head(pose)


class FiLM(nn.Module):
    """Feature-wise linear modulation from pooled physics cues."""

    def __init__(self, cue_dim: int, ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cue_dim, ch * 2), nn.ReLU(inplace=True), nn.Linear(ch * 2, ch * 2))

    def forward(self, feat: torch.Tensor, cue: torch.Tensor) -> torch.Tensor:
        gb = self.net(cue)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return feat * (1.0 + gamma) + beta
