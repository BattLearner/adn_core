"""Shared ADN: temporal convolutions, attention, reference alignment and two heads.
Each battery dataset is fitted independently. The 32 signed coordinates are
expanded into 64 CReLU channels.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


class TemporalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.layers = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation),
            nn.SiLU(),
            nn.Conv1d(width, width, 1),
        )

    def forward(self, x):
        return x + 0.5 * self.layers(self.norm(x.transpose(1, 2)).transpose(1, 2))


class UnifiedADN(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        cfg = {} if cfg is None else cfg
        self.align = cfg.get("initial_alignment", True)
        self.width = 24
        self.length = 128
        self.latent_dim = 32
        self.feature_dim = 64
        self.input = nn.Conv1d(4, self.width, 1)
        self.tcn = nn.Sequential(*(TemporalBlock(self.width, d) for d in [1, 2, 4]))
        self.attention_norm = nn.LayerNorm(self.width)
        self.cls = nn.Parameter(torch.zeros(1, 1, self.width))
        self.attention = nn.MultiheadAttention(
            self.width, 2, batch_first=True, dropout=0
        )
        self.projection = nn.Sequential(
            nn.Linear(self.width * 9, 48), nn.SiLU(), nn.Linear(48, self.latent_dim)
        )
        self.local = nn.Linear(32, self.latent_dim, bias=False)
        nn.init.eye_(self.local.weight)
        self.mix = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        nn.init.eye_(self.mix.weight)
        self.deep_gate = nn.Parameter(torch.tensor(-1.0))
        self.g_B = nn.Linear(self.feature_dim, 1, bias=False)
        nn.init.constant_(self.g_B.weight, math.log(2) / math.sqrt(self.feature_dim))
        self.g_C = nn.Linear(self.feature_dim, 1, bias=False)
        self.register_buffer("normal_scale", torch.tensor(0.02))
        p = torch.arange(self.length + 1).float()[:, None]
        freq = torch.exp(
            torch.arange(0, self.width, 2).float() * (-math.log(10000) / self.width)
        )
        pos = torch.zeros(self.length + 1, self.width)
        pos[:, 0::2] = torch.sin(p * freq)
        pos[:, 1::2] = torch.cos(p * freq)
        self.register_buffer("positional", pos[None] * 0.03)

    def encode(self, x):
        h = self.tcn(self.input(x)).transpose(1, 2)
        c = self.cls.expand(len(x), -1, -1)
        seq = torch.cat([c, h], 1)
        q = self.attention_norm(seq) + self.positional
        attended, _ = self.attention(q, q, q, need_weights=False)
        seq = seq + 0.5 * attended
        local = F.adaptive_avg_pool1d(seq[:, 1:].transpose(1, 2), 8).flatten(1)
        return self.projection(torch.cat([seq[:, 0], local], 1))

    @torch.no_grad()
    def update_normal_scale(self, z):
        self.normal_scale.mul_(0.95).add_(
            z.square().mean().sqrt().clamp_min(0.001) * 0.05
        )

    def probe_features(self, z):
        return z / self.normal_scale.detach().clamp_min(0.001)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != 4:
            raise ValueError("Expected [batch, positions, current V/I + reference V/I]")
        # One current scale from the fixed reference; no future records are used.
        scale = x[:, :, 3].median(1).values[:, None].clamp_min(0.01)
        x = torch.stack(
            [x[:, :, 0], x[:, :, 1] / scale, x[:, :, 2], x[:, :, 3] / scale], -1
        )
        cur = F.adaptive_avg_pool1d(x[:, :, :2].transpose(1, 2), self.length)
        ref = F.adaptive_avg_pool1d(x[:, :, 2:].transpose(1, 2), self.length)
        delta = 20 * (cur - ref)
        zeros = torch.zeros_like(delta)
        context = 2 * ref - 1
        current = torch.cat([delta, context], 1)
        reference = torch.cat([zeros, context], 1)
        allz = self.encode(torch.cat([current, reference], 0))
        a, b = allz.chunk(2)
        local = F.adaptive_avg_pool1d(delta if self.align else 20 * cur, 16).flatten(1)
        raw = self.local(local) + torch.sigmoid(self.deep_gate) * (
            a - b if self.align else a
        )
        h = self.mix(raw)
        z = torch.cat([F.relu(h), F.relu(-h)], 1)
        return self.g_B(z), z
