# -*- coding: utf-8 -*-
"""
net.py

PhysioFormer (v3, journal-grade)

This module defines a practical multimodal neural architecture for cognitive workload modeling
from *synchronized* eye tracking (Tobii-like) and photoplethysmography (PPG; BITalino-like).

v3 upgrades (over v2)
---------------------
1) **Explainability-ready attention export**:
   - Cross-modal attention can optionally return full attention weights for each head
     (useful for attention heatmaps and quantitative consistency analysis).
2) **Ablation-friendly modality switches**:
   - Enable/disable {eye sequence, PPG sequence, static engineered features} via config flags.
   - Cross-attention/bilinear pooling automatically deactivates when a required modality is missing.
3) **Strict interface contract**:
   - forward(...) returns a fixed 5-tuple:
     (logits, tlx_pred, emb_eye, emb_ppg, attn_dict)
     where missing outputs are returned as None.

Expected model inputs per batch
-------------------------------
- eye_seq:   (B, L, C_eye)  float32
- ppg_seq:   (B, L, C_ppg)  float32
- x_static:  (B, D_static)  float32

Default channels correspond to `data.py` v2/v3:
- eye_seq channels: [pupil, gaze_x, gaze_y, gaze_z, pos_x, pos_y, valid_mask] -> C_eye=7
- ppg_seq channels: [ppg_filtered, ppg_derivative] -> C_ppg=2
- x_static: 16 eye engineered + 16 PPG/HRV engineered -> D_static=32

Author: updated by assistant (v3)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    # input shapes
    seq_len: int = 256
    ppg_seq_len: int = 0  # 0 = same as seq_len; >0 = independent PPG sequence length
    eye_in_ch: int = 7
    ppg_in_ch: int = 2
    static_dim: int = 32

    # patching (token reduction)
    patch_size: int = 8  # 256 -> ~32 tokens
    patch_kernel: int = 8

    # model width/depth
    d_model: int = 96
    n_heads: int = 4
    n_layers_eye: int = 2
    n_layers_ppg: int = 2
    dropout: float = 0.1
    ff_mult: int = 4  # transformer FFN width multiplier
    use_layernorm: bool = True

    # fusion blocks
    use_cross_attn: bool = True
    use_bilinear: bool = True

    # modality switches for ablations
    use_eye: bool = True
    use_ppg: bool = True
    use_static: bool = True
    use_static_gate: bool = False
    static_gate_hidden: int = 64
    static_gate_floor: float = 0.25
    split_static_modalities: bool = False
    use_residual_static_fusion: bool = False
    residual_static_hidden: int = 128
    residual_static_init_gate_bias: float = -2.0

    # heads
    num_classes: int = 3
    head_hidden: int = 256

    # regression (auxiliary)
    use_regression: bool = True


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (buffer-based)."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        t = x.size(1)
        return x + self.pe[:t, :].unsqueeze(0).to(x.dtype)


class PatchEmbed1D(nn.Module):
    """
    Temporal patch embedding (Conv1d stride=patch_size) to reduce token length.

    Input:  (B, T, C)
    Output: (B, T', D)
    """

    def __init__(self, in_ch: int, d_model: int, kernel: int, stride: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, d_model, kernel_size=kernel, stride=stride, padding=kernel // 2)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B,C,T)
        y = self.conv(x)       # (B,D,T')
        y = y.transpose(1, 2)  # (B,T',D)
        y = self.ln(y)
        return y


class SequenceEncoder(nn.Module):
    """
    Per-modality encoder:
      PatchEmbed -> PosEnc -> TransformerEncoder -> mean pool
    """

    def __init__(self, in_ch: int, cfg: ModelConfig, n_layers: int, input_seq_len: int = 0):
        super().__init__()
        self.cfg = cfg
        self.patch = PatchEmbed1D(in_ch, cfg.d_model, kernel=cfg.patch_kernel, stride=cfg.patch_size)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.ff_mult,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers), enable_nested_tensor=False)
        _seq_len = input_seq_len if input_seq_len > 0 else cfg.seq_len
        max_len = max(64, int(_seq_len / max(cfg.patch_size, 1)) + 8)
        self.pos = SinusoidalPositionalEncoding(cfg.d_model, max_len=max_len)
        self.ln = nn.LayerNorm(cfg.d_model) if cfg.use_layernorm else nn.Identity()
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, C)
        returns:
          tokens: (B, T', D)
          pooled: (B, D)
        """
        h = self.patch(x)
        h = self.pos(h)
        h = self.drop(h)
        h = self.encoder(h)
        h = self.ln(h)
        pooled = h.mean(dim=1)
        return h, pooled


class CrossModalAttention(nn.Module):
    """
    Cross attention between two token sequences.

    Returned attention weights follow PyTorch MultiheadAttention conventions:
      attn_w: (B, H, T_q, T_k) if average_attn_weights=False
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_w = self.attn(
            q, kv, kv,
            need_weights=bool(need_weights),
            average_attn_weights=False if need_weights else True,
        )
        x = self.ln1(q + self.drop(attn_out))
        x2 = self.ff(x)
        x = self.ln2(x + self.drop(x2))
        if not need_weights:
            return x, None
        return x, attn_w


class StaticEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SplitStaticEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.eye_net = nn.Sequential(
            nn.Linear(16, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.cardio_net = nn.Sequential(
            nn.Linear(16, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.mix = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
        )

    def encode_parts(self, x_static: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_eye = x_static[:, :16]
        x_cardio = x_static[:, 16:]
        return self.eye_net(x_eye), self.cardio_net(x_cardio)

    def fuse(self, emb_eye_static: torch.Tensor, emb_cardio_static: torch.Tensor) -> torch.Tensor:
        return self.mix(torch.cat([emb_eye_static, emb_cardio_static], dim=1))


class StaticQualityGate(nn.Module):
    """
    Reliability-aware gate for the static physiological branch.

    The gate conditions the static contribution on the current ocular embedding
    and a few standardized quality descriptors:
      - blink rate
      - eye validity ratio
      - PPG quality
      - PPG amplitude MAD
    """

    def __init__(self, d_model: int, hidden: int, floor: float):
        super().__init__()
        self.floor = float(max(0.0, min(1.0, floor)))
        self.net = nn.Sequential(
            nn.Linear(d_model + 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, emb_eye: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        quality = extract_static_quality(x_static)
        quality = torch.nan_to_num(quality, nan=0.0, posinf=0.0, neginf=0.0)
        gate = torch.sigmoid(self.net(torch.cat([emb_eye, quality], dim=1)))
        if self.floor > 0.0:
            gate = self.floor + (1.0 - self.floor) * gate
        return gate


def extract_static_quality(x_static: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            x_static[:, 3],   # blink_rate_hz
            x_static[:, 13],  # eye_valid_ratio
            x_static[:, 29],  # ppg_quality_0_1
            x_static[:, 31],  # ppg_win_amplitude_mad
        ],
        dim=1,
    )


class ResidualStaticFusion(nn.Module):
    """
    Eye-first residual adapter for static physiology.

    The eye pathway provides the stable backbone representation. Static
    physiology is injected only as a gated residual correction, which reduces
    the risk of harming the stronger eye-only baseline when static cues are
    weak or noisy.
    """

    def __init__(self, d_model: int, hidden: int, out_dim: int, dropout: float, init_gate_bias: float):
        super().__init__()
        adapter_in = d_model * 4 + 4
        gate_in = d_model * 2 + 4

        self.eye_head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.static_adapter = nn.Sequential(
            nn.Linear(adapter_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.adapter_gate = nn.Sequential(
            nn.Linear(gate_in, max(32, hidden // 2)),
            nn.GELU(),
            nn.Linear(max(32, hidden // 2), 1),
        )
        self.out_norm = nn.LayerNorm(out_dim)

        last_adapter = self.static_adapter[-1]
        if isinstance(last_adapter, nn.Linear):
            nn.init.zeros_(last_adapter.weight)
            nn.init.zeros_(last_adapter.bias)
        last_gate = self.adapter_gate[-1]
        if isinstance(last_gate, nn.Linear):
            nn.init.zeros_(last_gate.weight)
            nn.init.constant_(last_gate.bias, float(init_gate_bias))

    def forward(self, emb_eye: torch.Tensor, emb_static: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        quality = torch.nan_to_num(extract_static_quality(x_static), nan=0.0, posinf=0.0, neginf=0.0)
        eye_hidden = self.eye_head(emb_eye)

        adapter_feat = torch.cat(
            [emb_eye, emb_static, emb_eye * emb_static, torch.abs(emb_eye - emb_static), quality],
            dim=1,
        )
        delta_hidden = self.static_adapter(adapter_feat)

        gate_feat = torch.cat([emb_eye, emb_static, quality], dim=1)
        alpha = torch.sigmoid(self.adapter_gate(gate_feat))
        fused_hidden = eye_hidden + alpha * delta_hidden
        return self.out_norm(fused_hidden)


class PhysioFormerNet(nn.Module):
    """
    Forward returns a fixed 5-tuple:
      logits:   (B, num_classes)
      tlx_pred: (B,1) or None
      emb_eye:  (B,D) or None   (pooled eye embedding; used for contrastive loss)
      emb_ppg:  (B,D) or None   (pooled ppg embedding; used for contrastive loss)
      attn:     dict or None    {'e2p': attn_w, 'p2e': attn_w}

    Notes
    -----
    - If a modality is disabled (cfg.use_eye/use_ppg/use_static=False),
      the corresponding embeddings are replaced by zeros and the block is skipped.
    - Cross-attention and bilinear fusion require both eye and ppg to be enabled.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.enc_eye = SequenceEncoder(cfg.eye_in_ch, cfg, n_layers=cfg.n_layers_eye) if cfg.use_eye else None
        self.enc_ppg = SequenceEncoder(cfg.ppg_in_ch, cfg, n_layers=cfg.n_layers_ppg,
                                       input_seq_len=cfg.ppg_seq_len) if cfg.use_ppg else None
        self.split_static = bool(cfg.use_static and cfg.split_static_modalities and int(cfg.static_dim) >= 32)
        self.static_enc = StaticEncoder(cfg.static_dim, cfg.d_model, cfg.dropout) if cfg.use_static and not self.split_static else None
        self.static_split_enc = SplitStaticEncoder(cfg.d_model, cfg.dropout) if self.split_static else None
        self.static_gate = None
        if cfg.use_static and cfg.use_static_gate:
            self.static_gate = StaticQualityGate(
                d_model=cfg.d_model,
                hidden=cfg.static_gate_hidden,
                floor=cfg.static_gate_floor,
            )

        self.cross_e2p = None
        self.cross_p2e = None
        if cfg.use_cross_attn and cfg.use_eye and cfg.use_ppg:
            self.cross_e2p = CrossModalAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
            self.cross_p2e = CrossModalAttention(cfg.d_model, cfg.n_heads, cfg.dropout)

        self.use_bilinear = bool(cfg.use_bilinear and cfg.use_eye and cfg.use_ppg)
        if self.use_bilinear:
            self.proj_e = nn.Linear(cfg.d_model, cfg.d_model)
            self.proj_p = nn.Linear(cfg.d_model, cfg.d_model)

        self.use_residual_static_fusion = bool(
            cfg.use_residual_static_fusion and cfg.use_eye and cfg.use_static and (not cfg.use_ppg)
        )
        self.residual_static = None
        if self.use_residual_static_fusion:
            self.residual_static = ResidualStaticFusion(
                d_model=cfg.d_model,
                hidden=cfg.residual_static_hidden,
                out_dim=cfg.head_hidden // 2,
                dropout=cfg.dropout,
                init_gate_bias=cfg.residual_static_init_gate_bias,
            )

        # Compute fusion dimension based on enabled components
        fusion_dim = 0
        if cfg.use_eye:
            fusion_dim += cfg.d_model
        if cfg.use_ppg:
            fusion_dim += cfg.d_model
        if cfg.use_static:
            fusion_dim += cfg.d_model

        if self.cross_e2p is not None and self.cross_p2e is not None:
            fusion_dim += cfg.d_model * 2  # pooled(e2p) + pooled(p2e)

        if self.use_bilinear:
            fusion_dim += cfg.d_model  # low-rank bilinear

        if cfg.use_eye and cfg.use_ppg:
            fusion_dim += cfg.d_model * 2  # emb_e ⊙ emb_p and |emb_e-emb_p|

        self.head = None
        if not self.use_residual_static_fusion:
            self.head = nn.Sequential(
                nn.Linear(fusion_dim, cfg.head_hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.head_hidden, cfg.head_hidden // 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
        self.cls = nn.Linear(cfg.head_hidden // 2, cfg.num_classes)

        self.use_regression = bool(cfg.use_regression)
        self.reg = nn.Linear(cfg.head_hidden // 2, 1) if self.use_regression else None

    def forward(
        self,
        eye_seq: torch.Tensor,
        ppg_seq: torch.Tensor,
        x_static: torch.Tensor,
        return_embeddings: bool = False,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        # Encode enabled modalities; otherwise produce zero embeddings (keeps batch shape consistent)
        B = x_static.shape[0] if x_static is not None else eye_seq.shape[0]
        D = self.cfg.d_model
        device = x_static.device if x_static is not None else eye_seq.device
        dtype = eye_seq.dtype

        tok_e = None
        tok_p = None
        emb_e = None
        emb_p = None
        z = torch.zeros((B, D), device=device, dtype=dtype)

        if self.enc_eye is not None:
            tok_e, emb_e = self.enc_eye(eye_seq)
        if self.enc_ppg is not None:
            tok_p, emb_p = self.enc_ppg(ppg_seq)

        emb_s = None
        if self.static_split_enc is not None:
            emb_s_eye, emb_s_cardio = self.static_split_enc.encode_parts(x_static)
            if self.static_gate is not None:
                emb_s_cardio = emb_s_cardio * self.static_gate(emb_e if emb_e is not None else z, x_static)
            emb_s = self.static_split_enc.fuse(emb_s_eye, emb_s_cardio)
        elif self.static_enc is not None:
            emb_s = self.static_enc(x_static)
            if emb_s is not None and self.static_gate is not None:
                emb_s = emb_s * self.static_gate(emb_e if emb_e is not None else z, x_static)

        # zero-fill for disabled modalities (for fusion stability)
        if emb_e is None:
            emb_e = z
        if emb_p is None:
            emb_p = z
        if emb_s is None:
            emb_s = z

        pooled = []
        if self.cfg.use_eye:
            pooled.append(emb_e)
        if self.cfg.use_ppg:
            pooled.append(emb_p)
        if self.cfg.use_static:
            pooled.append(emb_s)

        attn_dict: Optional[Dict[str, torch.Tensor]] = None

        # Cross-attention
        if self.cross_e2p is not None and self.cross_p2e is not None and tok_e is not None and tok_p is not None:
            e2p, w_e2p = self.cross_e2p(tok_e, tok_p, need_weights=bool(return_attn))
            p2e, w_p2e = self.cross_p2e(tok_p, tok_e, need_weights=bool(return_attn))
            pooled += [e2p.mean(dim=1), p2e.mean(dim=1)]

            if return_attn:
                attn_dict = {"e2p": w_e2p, "p2e": w_p2e}

        # Bilinear (TFN-like) interaction
        if self.use_bilinear:
            bil = self.proj_e(emb_e) * self.proj_p(emb_p)
            pooled += [bil]

        # Simple pairwise interactions (only meaningful if both modalities present)
        if self.cfg.use_eye and self.cfg.use_ppg:
            pooled += [emb_e * emb_p, torch.abs(emb_e - emb_p)]

        if self.use_residual_static_fusion and self.residual_static is not None:
            h = self.residual_static(emb_e, emb_s, x_static)
        else:
            zcat = torch.cat(pooled, dim=1)
            h = self.head(zcat)
        logits = self.cls(h)
        tlx_pred = self.reg(h) if self.use_regression and self.reg is not None else None

        # Return embeddings only if requested (keeps memory lower during training)
        out_emb_e = emb_e if return_embeddings else None
        out_emb_p = emb_p if return_embeddings else None

        return logits, tlx_pred, out_emb_e, out_emb_p, attn_dict

    def get_config(self) -> Dict[str, Any]:
        return asdict(self.cfg)


# -----------------------------
# Contrastive loss (optional)
# -----------------------------
def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Symmetric InfoNCE between two views.
    z1, z2: (B, D)
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = (z1 @ z2.t()) / float(temperature)  # (B,B)
    labels = torch.arange(z1.size(0), device=z1.device)
    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_12 + loss_21)
