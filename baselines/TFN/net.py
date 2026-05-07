# net.py
# -*- coding: utf-8 -*-
"""
Hybrid Tensor Fusion Network (TFN) + optional Cross-Attention Fusion
for cognitive workload modeling on HP Omnicept / HPO-CLD-style datasets.

Expected input to model:
- Eye features XE:  shape (B, 16)
- Heart/PPG features XH: shape (B, 16)

Fusion:
- "tfn": outer([XE,1],[XH,1]) -> conv/pool -> inter-modality embedding
- "xattn": attention(FE, FH, FH) over unimodal CNN embeddings

Author: generated for Renty
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    eye_feat_dim: int = 16
    heart_feat_dim: int = 16
    num_classes: int = 3

    # unimodal conv embedding
    uni_conv_channels: int = 32
    uni_conv_kernel: int = 3
    uni_pool: int = 2

    # TFN inter-modality conv embedding
    tfn_conv_channels: int = 32
    tfn_conv_kernel: int = 5
    tfn_pool: int = 2

    # classifier head
    fc_hidden: int = 128
    dropout: float = 0.2

    # cross-attention settings
    attn_dim: int = 64
    attn_heads: int = 4
    attn_dropout: float = 0.1

    fusion: str = "tfn"  # "tfn" or "xattn"


class ConvEmbedding1D(nn.Module):
    """
    Feature-vector -> Conv1D embedding.
    x: (B, F) -> (B,1,F) -> Conv1d -> ReLU -> MaxPool -> Flatten
    """
    def __init__(self, feat_dim: int, out_channels: int, kernel_size: int, pool: int):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(1, out_channels, kernel_size=kernel_size, padding=padding)
        self.pool = nn.MaxPool1d(kernel_size=pool, stride=pool)
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B,1,F)
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = x.flatten(1)
        return x


class TFNInterModality(nn.Module):
    """
    TFN inter-modality: [XE,1] ⊗ [XH,1] -> flatten -> Conv1d -> pool -> flatten
    """
    def __init__(self, eye_dim: int, heart_dim: int, out_channels: int, kernel_size: int, pool: int):
        super().__init__()
        self.eye_dim = eye_dim
        self.heart_dim = heart_dim
        self.tensor_dim = (eye_dim + 1) * (heart_dim + 1)

        padding = kernel_size // 2
        self.conv = nn.Conv1d(1, out_channels, kernel_size=kernel_size, padding=padding)
        self.pool = nn.MaxPool1d(kernel_size=pool, stride=pool)

    def forward(self, xe: torch.Tensor, xh: torch.Tensor) -> torch.Tensor:
        b = xe.shape[0]
        ones_e = torch.ones((b, 1), device=xe.device, dtype=xe.dtype)
        ones_h = torch.ones((b, 1), device=xh.device, dtype=xh.dtype)

        xe1 = torch.cat([xe, ones_e], dim=1)  # (B, Fe+1)
        xh1 = torch.cat([xh, ones_h], dim=1)  # (B, Fh+1)

        t = torch.bmm(xe1.unsqueeze(2), xh1.unsqueeze(1))  # (B, Fe+1, Fh+1)
        t = t.reshape(b, -1)  # (B, (Fe+1)*(Fh+1))
        t = t.unsqueeze(1)    # (B,1,T)

        t = F.relu(self.conv(t))
        t = self.pool(t)
        t = t.flatten(1)
        return t


class CrossAttentionFusion(nn.Module):
    """
    Decoder-like cross attention: Attn(Q=FE, K=FH, V=FH).
    Since FE/FH are vectors, we chunk into a small token sequence.
    """
    def __init__(self, in_dim_e: int, in_dim_h: int, attn_dim: int, heads: int, dropout: float):
        super().__init__()
        self.proj_e = nn.Linear(in_dim_e, attn_dim)
        self.proj_h = nn.Linear(in_dim_h, attn_dim)
        self.attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(attn_dim)
        self.ff = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim * 2, attn_dim),
        )

    @staticmethod
    def _vec_to_tokens(x: torch.Tensor, token_len: int = 8) -> torch.Tensor:
        b, d = x.shape
        if d % token_len == 0:
            d2 = d // token_len
            return x.view(b, token_len, d2)
        pad = (token_len - (d % token_len)) % token_len
        x2 = F.pad(x, (0, pad), mode="constant", value=0.0)
        d_new = x2.shape[1]
        d2 = d_new // token_len
        return x2.view(b, token_len, d2)

    def forward(self, fe: torch.Tensor, fh: torch.Tensor) -> torch.Tensor:
        qe = self.proj_e(fe)  # (B, attn_dim)
        kh = self.proj_h(fh)  # (B, attn_dim)

        q = self._vec_to_tokens(qe, token_len=8)
        k = self._vec_to_tokens(kh, token_len=8)
        v = k

        attn_out, _ = self.attn(q, k, v, need_weights=False)
        x = self.ln(attn_out + q)
        x2 = self.ff(x)
        x = self.ln(x + x2)

        fused = x.mean(dim=1)
        return fused.flatten(1)


class HybridFusionNet(nn.Module):
    """
    Unimodal: FE = CNN(XE), FH = CNN(XH)
    Fusion:
      - tfn:  Fint = TFN(XE, XH)
      - xattn:Fint = CrossAttention(FE, FH)
    Head: concat -> FC -> logits
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.fusion_type = cfg.fusion.lower().strip()

        self.eye_embed = ConvEmbedding1D(
            feat_dim=cfg.eye_feat_dim,
            out_channels=cfg.uni_conv_channels,
            kernel_size=cfg.uni_conv_kernel,
            pool=cfg.uni_pool,
        )
        self.heart_embed = ConvEmbedding1D(
            feat_dim=cfg.heart_feat_dim,
            out_channels=cfg.uni_conv_channels,
            kernel_size=cfg.uni_conv_kernel,
            pool=cfg.uni_pool,
        )

        def out_len(F_: int, pool: int) -> int:
            return (F_ // pool) if F_ % pool == 0 else (F_ // pool)

        eye_out = cfg.uni_conv_channels * out_len(cfg.eye_feat_dim, cfg.uni_pool)
        heart_out = cfg.uni_conv_channels * out_len(cfg.heart_feat_dim, cfg.uni_pool)

        if self.fusion_type == "tfn":
            self.inter = TFNInterModality(
                eye_dim=cfg.eye_feat_dim,
                heart_dim=cfg.heart_feat_dim,
                out_channels=cfg.tfn_conv_channels,
                kernel_size=cfg.tfn_conv_kernel,
                pool=cfg.tfn_pool,
            )
            t_dim = self.inter.tensor_dim
            t_out = cfg.tfn_conv_channels * out_len(t_dim, cfg.tfn_pool)
            fusion_out = eye_out + heart_out + t_out

        elif self.fusion_type == "xattn":
            self.inter = CrossAttentionFusion(
                in_dim_e=eye_out,
                in_dim_h=heart_out,
                attn_dim=cfg.attn_dim,
                heads=cfg.attn_heads,
                dropout=cfg.attn_dropout,
            )
            with torch.no_grad():
                dummy_e = torch.zeros(2, cfg.eye_feat_dim)
                dummy_h = torch.zeros(2, cfg.heart_feat_dim)
                fe = self.eye_embed(dummy_e)
                fh = self.heart_embed(dummy_h)
                fint = self.inter(fe, fh)
                inter_dim = fint.shape[1]
            fusion_out = eye_out + heart_out + inter_dim

        else:
            raise ValueError(f"Unsupported fusion: {cfg.fusion} (use 'tfn' or 'xattn')")

        self.classifier = nn.Sequential(
            nn.Linear(fusion_out, cfg.fc_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fc_hidden, cfg.num_classes),
        )

    def forward(self, xe: torch.Tensor, xh: torch.Tensor) -> torch.Tensor:
        fe = self.eye_embed(xe)
        fh = self.heart_embed(xh)

        if self.fusion_type == "tfn":
            fint = self.inter(xe, xh)
        else:
            fint = self.inter(fe, fh)

        feat = torch.cat([fe, fh, fint], dim=1)
        logits = self.classifier(feat)
        return logits

    def get_config(self) -> Dict[str, Any]:
        return {k: getattr(self.cfg, k) for k in self.cfg.__dict__.keys()}
