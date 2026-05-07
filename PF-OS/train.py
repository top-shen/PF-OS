# -*- coding: utf-8 -*-
"""
train.py

PhysioFormer training entry (v3, journal-grade)

This script trains PhysioFormerNet on HPO-CLD-style multimodal recordings.

Key properties
--------------
- Subject-wise splitting by default (recommended; avoids identity leakage)
- Optional explicit subject lists for CV / ablation reproducibility
- Multi-task objective:
    L = CE(y) + reg_weight * Huber(TLX) + contrastive_weight * InfoNCE(emb_eye, emb_ppg)
- Deterministic window slicing based on Unix microsecond timestamps

Companion scripts (v3)
----------------------
- run_cv.py: subject-wise K-fold CV, bootstrap CI, statistical tests
- run_ablation.py: ablations + significance tests + LaTeX table generation
- explain.py: attention maps + HRV consistency analysis + figures for the paper

Author: updated by assistant (v3)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from data import MultiModalWindowDataset, StreamConfig, get_static_feature_names, validate_dataset
from net import ModelConfig, PhysioFormerNet, info_nce_loss


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Standardizer for static features
# -----------------------------
class Standardizer:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> None:
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        sd = np.where(np.isfinite(sd) & (sd > self.eps), sd, 1.0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        self.mu = mu.astype(np.float32)
        self.sd = sd.astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mu is not None and self.sd is not None
        return (X - self.mu) / (self.sd + self.eps)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mu": self.mu.tolist() if self.mu is not None else None,
            "sd": self.sd.tolist() if self.sd is not None else None,
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.mu = np.array(d["mu"], dtype=np.float32) if d.get("mu") is not None else None
        self.sd = np.array(d["sd"], dtype=np.float32) if d.get("sd") is not None else None


class StaticNormWrapper(Dataset):
    """
    Apply train-fit standardization to x_static only.
    Sequences are already per-window normalized in data.py.
    """

    def __init__(self, base: Dataset, std: Standardizer):
        self.base = base
        self.std = std

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        eye_seq, ppg_seq, x_static, y, y_reg = self.base[idx]
        x_static = self.std.transform(np.asarray(x_static, dtype=np.float32)[None, :])[0]
        return eye_seq, ppg_seq, x_static.astype(np.float32), np.int64(y), np.float32(y_reg)


# -----------------------------
# Metrics
# -----------------------------
def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    out["acc"] = float(np.mean(y_true == y_pred)) if y_true.size else 0.0

    num_classes = 3
    recalls, f1s = [], []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        fp = np.sum((y_true != c) & (y_pred == c))
        rec = tp / (tp + fn + 1e-12)
        prec = tp / (tp + fp + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        recalls.append(rec)
        f1s.append(f1)

    out["recall_macro"] = float(np.mean(recalls))
    out["f1_macro"] = float(np.mean(f1s))

    if y_proba is not None and y_true.size:
        try:
            from sklearn.metrics import roc_auc_score
            out["auc_ovr_macro"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            out["auc_ovr_macro"] = float("nan")
    else:
        out["auc_ovr_macro"] = float("nan")

    return out


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if y_true.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "pearson": float("nan")}
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    try:
        if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
            r = float("nan")
        else:
            r = float(np.corrcoef(y_true, y_pred)[0, 1])
    except Exception:
        r = float("nan")
    out["mae"] = mae
    out["rmse"] = rmse
    out["pearson"] = r
    return out


def attention_entropy(attn_w: np.ndarray, eps: float = 1e-12) -> float:
    """
    attn_w: (H, Tq, Tk) or (Tq, Tk). Returns mean normalized entropy across (H,Tq).
    """
    if attn_w.ndim == 2:
        w = attn_w[None, :, :]
    else:
        w = attn_w
    # Normalize along Tk
    w = np.clip(w, 0.0, 1.0)
    w = w / (np.sum(w, axis=-1, keepdims=True) + eps)
    ent = -np.sum(w * np.log(w + eps), axis=-1)  # (H,Tq)
    # Normalize entropy by log(Tk)
    Tk = w.shape[-1]
    ent = ent / (math.log(Tk + eps) + eps)
    return float(np.mean(ent))


def attention_max(attn_w: np.ndarray) -> float:
    if attn_w.ndim == 2:
        return float(np.mean(np.max(attn_w, axis=-1)))
    return float(np.mean(np.max(attn_w, axis=-1)))


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    return_attn_summary: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Returns:
      y_true, y_pred, y_proba
      extra dict: may include tlx_true/tlx_pred and attention summaries
    """
    model.eval()
    ys, yp, prob = [], [], []
    tlx_true, tlx_pred = [], []
    ent_e2p, ent_p2e, max_e2p, max_p2e = [], [], [], []

    for eye_seq, ppg_seq, x_static, y, y_reg in loader:
        eye_seq = eye_seq.to(device)
        ppg_seq = ppg_seq.to(device)
        x_static = x_static.to(device)
        y = y.to(device)
        y_reg = y_reg.to(device)

        need_embed = False
        need_attn = bool(return_attn_summary)
        logits, tlx_hat, _, _, attn = model(eye_seq, ppg_seq, x_static, return_embeddings=need_embed, return_attn=need_attn)

        p = torch.softmax(logits, dim=1)
        pred = torch.argmax(p, dim=1)

        ys.append(y.cpu().numpy())
        yp.append(pred.cpu().numpy())
        prob.append(p.cpu().numpy())

        if tlx_hat is not None:
            mask = (y_reg >= 0.0)
            if mask.any():
                tlx_true.append(y_reg[mask].cpu().numpy())
                tlx_pred.append(tlx_hat.squeeze(1)[mask].cpu().numpy())

        if need_attn and attn is not None and "e2p" in attn and "p2e" in attn:
            # attn weights: (B, H, Tq, Tk) on CPU
            w_e2p = attn["e2p"].detach().cpu().numpy()
            w_p2e = attn["p2e"].detach().cpu().numpy()
            for b in range(w_e2p.shape[0]):
                ent_e2p.append(attention_entropy(w_e2p[b]))
                ent_p2e.append(attention_entropy(w_p2e[b]))
                max_e2p.append(attention_max(w_e2p[b]))
                max_p2e.append(attention_max(w_p2e[b]))

    y_true = np.concatenate(ys, axis=0) if ys else np.array([], dtype=int)
    y_pred = np.concatenate(yp, axis=0) if yp else np.array([], dtype=int)
    y_proba = np.concatenate(prob, axis=0) if prob else np.zeros((0, 3), dtype=np.float32)

    extra: Dict[str, np.ndarray] = {}
    if tlx_true and tlx_pred:
        extra["tlx_true"] = np.concatenate(tlx_true, axis=0).astype(np.float32)
        extra["tlx_pred"] = np.concatenate(tlx_pred, axis=0).astype(np.float32)
    if return_attn_summary and len(ent_e2p) == len(y_true):
        extra["attn_ent_e2p"] = np.asarray(ent_e2p, dtype=np.float32)
        extra["attn_ent_p2e"] = np.asarray(ent_p2e, dtype=np.float32)
        extra["attn_max_e2p"] = np.asarray(max_e2p, dtype=np.float32)
        extra["attn_max_p2e"] = np.asarray(max_p2e, dtype=np.float32)

    return y_true, y_pred, y_proba, extra


@torch.no_grad()
def compute_val_contrastive(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float = 0.07,
) -> float:
    """Compute mean InfoNCE contrastive loss over a validation loader. Returns NaN if not applicable."""
    model.eval()
    losses, counts = [], []
    for eye_seq, ppg_seq, x_static, y, y_reg in loader:
        eye_seq = eye_seq.to(device)
        ppg_seq = ppg_seq.to(device)
        x_static = x_static.to(device)
        _, _, emb_e, emb_p, _ = model(eye_seq, ppg_seq, x_static, return_embeddings=True, return_attn=False)
        if emb_e is not None and emb_p is not None and emb_e.shape[0] > 1:
            cl = info_nce_loss(emb_e, emb_p, temperature=temperature)
            losses.append(float(cl.item()) * emb_e.shape[0])
            counts.append(emb_e.shape[0])
    if not counts:
        return float("nan")
    return sum(losses) / sum(counts)


def compute_per_subject_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    pred_df must contain columns: participant, y_true, y_pred, p_low, p_med, p_high
    Returns per-subject metrics table.
    """
    rows = []
    for pid, g in pred_df.groupby("participant"):
        y_true = g["y_true"].to_numpy(dtype=int)
        y_pred = g["y_pred"].to_numpy(dtype=int)
        y_proba = g[["p_low", "p_med", "p_high"]].to_numpy(dtype=np.float32)
        m = compute_classification_metrics(y_true, y_pred, y_proba)
        rows.append({"participant": pid, **m, "n_windows": int(len(g))})
    return pd.DataFrame(rows)


# -----------------------------
# Scheduler
# -----------------------------
def make_warmup_cosine_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    """LambdaLR: linear warmup then cosine decay to 0."""
    warmup_steps = max(1, int(warmup_steps))
    total_steps = max(warmup_steps + 1, int(total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# -----------------------------
# Subject handling utilities
# -----------------------------
def parse_subject_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def list_participants(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("hpo-cld")])


def default_subject_split(subject_dirs: List[Path], seed: int, train_ratio: float, val_ratio: float) -> Tuple[List[Path], List[Path], List[Path]]:
    rng = np.random.RandomState(seed)
    idxs = np.arange(len(subject_dirs))
    rng.shuffle(idxs)
    if len(subject_dirs) < 3:
        return subject_dirs, subject_dirs, subject_dirs

    n_train = max(1, int(len(subject_dirs) * train_ratio))
    n_val = max(1, int(len(subject_dirs) * val_ratio))
    n_train = min(n_train, len(subject_dirs) - 2)
    n_val = min(n_val, len(subject_dirs) - n_train - 1)

    train_dirs = [subject_dirs[i] for i in idxs[:n_train]]
    val_dirs = [subject_dirs[i] for i in idxs[n_train:n_train + n_val]]
    test_dirs = [subject_dirs[i] for i in idxs[n_train + n_val:]]
    return train_dirs, val_dirs, test_dirs


def resolve_dirs(root: Path, subjects: List[str]) -> List[Path]:
    out = []
    for s in subjects:
        p = root / s
        if not p.exists():
            raise FileNotFoundError(f"Subject folder not found: {p}")
        out.append(p)
    return out


def apply_model_profile(args: argparse.Namespace, profile: str) -> argparse.Namespace:
    """
    Apply paper-facing profile defaults without changing the original namespace.

    `ocular_static` is the main PhysioFormer-OS operating point:
      - eye sequence + static physiological summaries
      - no raw PPG temporal branch
      - no bilinear pooling
      - no auxiliary regression / contrastive objectives
      - macro-F1-based checkpoint selection
      - quality-aware static gating enabled
    """
    prof = str(profile).lower().strip()
    out = argparse.Namespace(**vars(args))

    if prof in {"ocular_static", "classification"}:
        out.no_ppg = True
        out.no_bilinear = True
        out.no_regression_head = True
        out.reg_weight = 0.0
        out.contrastive_weight = 0.0
        out.select_metric = "f1"
        out.use_static_gate = True

    return out


def dataset_meta_list(ds: Union[MultiModalWindowDataset, Subset]) -> List[Any]:
    """
    Returns list of WindowSampleMeta objects aligned with dataset index order.
    """
    if hasattr(ds, "sample_meta"):
        return list(getattr(ds, "sample_meta"))
    if isinstance(ds, Subset):
        base = ds.dataset
        idxs = list(ds.indices)
        if hasattr(base, "sample_meta"):
            return [base.sample_meta[i] for i in idxs]
    raise AttributeError("Dataset does not expose sample_meta; cannot export predictions with meta.")


# -----------------------------
# Training core
# -----------------------------
def train_one_split(
    root: Path,
    outdir: Path,
    seed: int,
    train_dirs: List[Path],
    val_dirs: List[Path],
    test_dirs: List[Path],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Train one split and evaluate. Returns a run summary dict.
    """
    set_seed(seed)
    device = torch.device(args.device)

    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)


    # -----------------------------
    # Pre-flight dataset integrity check (optional but recommended)
    # -----------------------------
    if not getattr(args, "skip_data_check", False):
        log_path = Path(getattr(args, "data_check_log", "")).expanduser() if getattr(args, "data_check_log", "") else (outdir / "data_check.log")

        # unique subject dirs (by name) across splits
        all_dirs: List[Path] = []
        _seen = set()
        for d in list(train_dirs) + list(val_dirs) + list(test_dirs):
            if d is None:
                continue
            if d.name in _seen:
                continue
            _seen.add(d.name)
            all_dirs.append(d)

        summary = validate_dataset(
            root=root,
            participant_dirs=all_dirs,
            log_path=log_path,
            window_sec=float(args.window_sec),
            stride_sec=float(args.stride_sec),
        )

        if not getattr(args, "no_drop_bad_subjects", False):
            ok = set(summary.get("ok_subjects", []))

            def _filt(dirs: List[Path]) -> List[Path]:
                return [p for p in dirs if p.name in ok]

            train_dirs = _filt(train_dirs)
            val_dirs = _filt(val_dirs)
            test_dirs = _filt(test_dirs)
            dropped = [p.name for p in all_dirs if p.name not in ok]
            if dropped:
                print(f"[data-check] Dropped {len(dropped)} subjects with critical issues. See: {log_path.resolve()}")
        else:
            print(f"[data-check] Completed. See: {log_path.resolve()} (keeping bad subjects per --no_drop_bad_subjects)")

        if len(train_dirs) == 0 or len(val_dirs) == 0 or len(test_dirs) == 0:
            raise RuntimeError("After data-check filtering, at least one split is empty. Inspect data_check.log or pass --no_drop_bad_subjects.")
    # datasets (train has augmentation)
    _stream_cfg = StreamConfig(seq_len=args.seq_len, ppg_seq_len=args.ppg_seq_len)
    ds_train_raw = MultiModalWindowDataset(train_dirs, window_sec=args.window_sec, stride_sec=args.stride_sec,
                                           seq_len=args.seq_len, stream_cfg=_stream_cfg, cache_dir=cache_dir, augment=False,
                                           seed=seed, verbose=args.verbose)
    ds_train_aug = MultiModalWindowDataset(train_dirs, window_sec=args.window_sec, stride_sec=args.stride_sec,
                                           seq_len=args.seq_len, stream_cfg=_stream_cfg, cache_dir=cache_dir, augment=(not args.no_augment),
                                           seed=seed, verbose=False)

    ds_val_raw = MultiModalWindowDataset(val_dirs, window_sec=args.window_sec, stride_sec=args.stride_sec,
                                         seq_len=args.seq_len, stream_cfg=_stream_cfg, cache_dir=cache_dir, augment=False,
                                         seed=seed, verbose=False)
    ds_test_raw = MultiModalWindowDataset(test_dirs, window_sec=args.window_sec, stride_sec=args.stride_sec,
                                          seq_len=args.seq_len, stream_cfg=_stream_cfg, cache_dir=cache_dir, augment=False,
                                          seed=seed, verbose=False)

    # Fallback: window-level split when too few subjects
    # Group overlapping windows by task interval to prevent temporal leakage
    window_split = False
    if len(list_participants(root)) < 3:
        window_split = True
        n = len(ds_train_raw)
        if n < 5:
            raise RuntimeError("Too few windows to train. Consider smaller --window_sec or --stride_sec.")
        rng = np.random.RandomState(seed)

        # Group windows by task interval: consecutive windows with same (pidx, y, tlx) belong to one task
        groups: List[List[int]] = []
        cur_group: List[int] = [0]
        samples = ds_train_raw.samples  # (pidx, s, e, y, tlx, diff)
        for i in range(1, n):
            prev, cur = samples[i - 1], samples[i]
            if prev[0] == cur[0] and prev[3] == cur[3] and prev[5] == cur[5]:
                cur_group.append(i)
            else:
                groups.append(cur_group)
                cur_group = [i]
        groups.append(cur_group)

        # Split at task-group level, then flatten to window indices
        g_perm = rng.permutation(len(groups))
        n_g_train = int(0.8 * len(groups))
        n_g_val = int(0.1 * len(groups))
        train_idx = np.concatenate([groups[g] for g in g_perm[:n_g_train]]) if n_g_train > 0 else np.array([], dtype=int)
        val_idx = np.concatenate([groups[g] for g in g_perm[n_g_train:n_g_train + n_g_val]]) if n_g_val > 0 else np.array([], dtype=int)
        test_idx = np.concatenate([groups[g] for g in g_perm[n_g_train + n_g_val:]]) if len(g_perm) > n_g_train + n_g_val else np.array([], dtype=int)

        ds_train_raw = Subset(ds_train_raw, train_idx)
        ds_train_aug = Subset(ds_train_aug, train_idx)
        ds_val_raw = Subset(ds_val_raw, val_idx)
        ds_test_raw = Subset(ds_test_raw, test_idx)

    # Fit static-feature normalization on *non-augmented train*
    static_list, y_list = [], []
    for i in range(len(ds_train_raw)):
        _eye_seq, _ppg_seq, x_static, y, _y_reg = ds_train_raw[i]
        static_list.append(np.asarray(x_static, dtype=np.float32))
        y_list.append(int(y))
    Xs = np.stack(static_list, axis=0).astype(np.float32) if static_list else np.zeros((0, len(get_static_feature_names())), np.float32)
    ys = np.array(y_list, dtype=int)

    std = Standardizer()
    std.fit(Xs)

    ds_train = StaticNormWrapper(ds_train_aug, std)
    ds_val = StaticNormWrapper(ds_val_raw, std)
    ds_test = StaticNormWrapper(ds_test_raw, std)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    # class weights from train windows
    counts = np.bincount(ys, minlength=3) if ys.size else np.ones(3, dtype=int)
    w = (counts.sum() / (counts + 1e-6)).astype(np.float32)
    w = w / np.mean(w)
    class_weights = torch.tensor(w, dtype=torch.float32, device=device)

    # model config (includes ablation switches)
    cfg = ModelConfig(
        seq_len=args.seq_len,
        ppg_seq_len=args.ppg_seq_len,
        eye_in_ch=7,
        ppg_in_ch=2,
        static_dim=len(get_static_feature_names()),
        patch_size=args.patch_size,
        patch_kernel=args.patch_kernel,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers_eye=args.n_layers_eye,
        n_layers_ppg=args.n_layers_ppg,
        dropout=args.dropout,
        num_classes=3,
        use_regression=(not args.no_regression_head),
        use_cross_attn=(not args.no_cross_attn),
        use_bilinear=(not args.no_bilinear),
        use_eye=(not args.no_eye),
        use_ppg=(not args.no_ppg),
        use_static=(not args.no_static),
        use_static_gate=bool(args.use_static_gate and (not args.no_static)),
        static_gate_hidden=int(args.static_gate_hidden),
        static_gate_floor=float(args.static_gate_floor),
        split_static_modalities=bool(args.split_static_modalities and (not args.no_static)),
        use_residual_static_fusion=bool(args.use_residual_static_fusion and (not args.no_static)),
        residual_static_hidden=int(args.residual_static_hidden),
        residual_static_init_gate_bias=float(args.residual_static_init_gate_bias),
    )
    model = PhysioFormerNet(cfg).to(device)

    # losses & optim
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(args.label_smoothing))
    huber = nn.SmoothL1Loss(beta=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = max(1, int(args.epochs) * max(1, len(dl_train)))
    warmup_steps = max(10, int(0.05 * total_steps))
    scheduler = make_warmup_cosine_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    scaler = torch.amp.GradScaler('cuda', enabled=bool(args.amp and device.type == "cuda"))

    run_cfg = {
        "args": vars(args),
        "model_config": model.get_config(),
        "subjects": {
            "train": [p.name for p in train_dirs],
            "val": [p.name for p in val_dirs],
            "test": [p.name for p in test_dirs],
        },
        "window_split_fallback": bool(window_split),
        "class_counts_train": counts.tolist(),
        "class_weights": w.tolist(),
        "static_standardizer": std.state_dict(),
        "static_feature_names": get_static_feature_names(),
    }
    (outdir / "run_config.json").write_text(json.dumps(run_cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    best_select_score = -1.0
    best_val_f1 = -1.0
    best_path = outdir / "best.pt"
    last_path = outdir / "last.pt"
    patience_counter = 0  # 鏃╁仠璁℃暟鍣?
    if args.verbose:
        print(f"[info] root={root.resolve()}")
        print(f"[info] subjects: train={len(train_dirs)} val={len(val_dirs)} test={len(test_dirs)}")
        print(f"[info] windows: train={len(ds_train)} val={len(ds_val)} test={len(ds_test)}")
        print(f"[info] device={device} seq_len={args.seq_len} window={args.window_sec}s stride={args.stride_sec}s")
        print(f"[info] class_counts={counts.tolist()} weights={w.tolist()}")

    train_loss_history = []  # 璁板綍瀹屾暣鐨勮缁冩崯澶辨洸绾?
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        t0 = time.time()
        loss_meter = []

        pbar = tqdm(
            dl_train,
            desc=f"Epoch {epoch:03d}/{int(args.epochs):03d}",
            leave=False,
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        for eye_seq, ppg_seq, x_static, y, y_reg in pbar:
            eye_seq = eye_seq.to(device)
            ppg_seq = ppg_seq.to(device)
            x_static = x_static.to(device)
            y = y.to(device)
            y_reg = y_reg.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=bool(args.amp and device.type == "cuda")):
                need_embed = bool(args.contrastive_weight and float(args.contrastive_weight) > 0)
                logits, tlx_hat, emb_e, emb_p, _attn = model(
                    eye_seq, ppg_seq, x_static,
                    return_embeddings=need_embed,
                    return_attn=False,
                )
                ce = criterion(logits, y)

                reg = torch.tensor(0.0, device=device)
                if tlx_hat is not None and args.reg_weight and float(args.reg_weight) > 0:
                    mask = (y_reg >= 0.0)
                    if mask.any():
                        reg = huber(tlx_hat.squeeze(1)[mask], y_reg[mask])

                cl = torch.tensor(0.0, device=device)
                if need_embed and emb_e is not None and emb_p is not None:
                    cl = info_nce_loss(emb_e, emb_p, temperature=float(args.contrastive_temp))

                loss = ce + float(args.reg_weight) * reg + float(args.contrastive_weight) * cl

            scaler.scale(loss).backward()

            if args.grad_clip is not None and float(args.grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_meter.append(float(loss.detach().cpu().item()))
            pbar.set_postfix({'loss': f'{loss_meter[-1]:.4f}'})

        train_loss = float(np.mean(loss_meter)) if loss_meter else 0.0
        train_loss_history.append(train_loss)

        # validation
        y_true, y_pred, y_proba, extra = predict(model, dl_val, device, return_attn_summary=False)
        val_metrics = compute_classification_metrics(y_true, y_pred, y_proba)

        if "tlx_true" in extra and "tlx_pred" in extra:
            rt = extra["tlx_true"].astype(np.float64)
            rp = extra["tlx_pred"].astype(np.float64)
            val_metrics.update({f"tlx_{k}": v for k, v in compute_regression_metrics(rt, rp).items()})
        else:
            val_metrics.update({"tlx_mae": float("nan"), "tlx_rmse": float("nan"), "tlx_pearson": float("nan")})

        dt = time.time() - t0
        log = {"epoch": epoch, "train_loss": train_loss, "sec": dt, **val_metrics}
        print(json.dumps(log, ensure_ascii=False))

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "model_config": asdict(cfg),
            "args": vars(args),
            "static_standardizer": std.state_dict(),
        }
        torch.save(ckpt, last_path)

        # best checkpoint selection:
        # default=f1 (classification-first); optional=composite includes auxiliary objectives.
        select_score = float(val_metrics["f1_macro"])
        if str(args.select_metric).lower() == "composite":
            if np.isfinite(val_metrics.get("tlx_pearson", float("nan"))):
                # Small bonus for regression quality (max ~0.05 contribution)
                select_score += 0.05 * max(0.0, val_metrics["tlx_pearson"])
            # Contrastive loss bonus: lower loss -> better cross-modal alignment
            if args.contrastive_weight and float(args.contrastive_weight) > 0:
                val_cl_loss = compute_val_contrastive(model, dl_val, device, temperature=float(args.contrastive_temp))
                if np.isfinite(val_cl_loss):
                    cl_baseline = float(np.log(max(2, args.batch_size)))
                    cl_improvement = max(0.0, (cl_baseline - val_cl_loss) / cl_baseline)
                    select_score += 0.02 * cl_improvement

        best_val_f1 = max(best_val_f1, float(val_metrics["f1_macro"]))
        if select_score > best_select_score + 1e-6:
            best_select_score = float(select_score)
            patience_counter = 0
            torch.save(ckpt, best_path)
            print(
                f"[best] epoch={epoch} f1_macro={val_metrics['f1_macro']:.4f} "
                f"select={best_select_score:.4f} ({args.select_metric}) -> {best_path}"
            )
        else:
            patience_counter += 1
            print(f"[early-stop] F1 not improved for {patience_counter}/{args.early_stop_patience} epochs")

        if patience_counter >= args.early_stop_patience:
            print(f"[early-stop] Training stopped at epoch {epoch}")
            break

    # 淇濆瓨璁粌鎹熷け鏇茬嚎
    loss_history_path = outdir / "train_loss_history.json"
    loss_history_path.write_text(json.dumps(train_loss_history, ensure_ascii=False), encoding="utf-8")

    # final test eval with best checkpoint
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False) if best_path.exists() else torch.load(last_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    y_true, y_pred, y_proba, extra = predict(model, dl_test, device, return_attn_summary=bool(args.test_attn_summary))
    test_metrics = compute_classification_metrics(y_true, y_pred, y_proba)

    if "tlx_true" in extra and "tlx_pred" in extra:
        rt = extra["tlx_true"].astype(np.float64)
        rp = extra["tlx_pred"].astype(np.float64)
        test_metrics.update({f"tlx_{k}": v for k, v in compute_regression_metrics(rt, rp).items()})
    else:
        test_metrics.update({"tlx_mae": float("nan"), "tlx_rmse": float("nan"), "tlx_pearson": float("nan")})

    (outdir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # Export predictions (journal reporting + downstream stats)
    pred_path = None
    subj_path = None
    if args.save_test_predictions:
        metas = dataset_meta_list(ds_test_raw)
        rows = []
        # raw static features (not standardized) for HRV analysis
        static_raw = []
        for i in range(len(ds_test_raw)):
            _eye_seq, _ppg_seq, x_static_raw, _y, _y_reg = ds_test_raw[i]
            static_raw.append(np.asarray(x_static_raw, dtype=np.float32))
        static_raw = np.stack(static_raw, axis=0) if static_raw else np.zeros((0, len(get_static_feature_names())), dtype=np.float32)

        for i, meta in enumerate(metas):
            r = {
                "participant": str(getattr(meta, "participant")),
                "t_start_us": int(getattr(meta, "t_start_us")),
                "t_end_us": int(getattr(meta, "t_end_us")),
                "y_true": int(y_true[i]) if i < len(y_true) else int(getattr(meta, "y")),
                "y_pred": int(y_pred[i]) if i < len(y_pred) else -1,
                "p_low": float(y_proba[i, 0]) if i < len(y_proba) else float("nan"),
                "p_med": float(y_proba[i, 1]) if i < len(y_proba) else float("nan"),
                "p_high": float(y_proba[i, 2]) if i < len(y_proba) else float("nan"),
                "tlx": float(getattr(meta, "tlx", float("nan"))),
                "difficulty": str(getattr(meta, "difficulty", "")),
            }
            # attach raw static features
            for j, name in enumerate(get_static_feature_names()):
                r[f"stat_{name}"] = float(static_raw[i, j]) if i < static_raw.shape[0] else float("nan")

            rows.append(r)

        # attention summary
        if args.test_attn_summary and len(y_true) == len(rows) and "attn_ent_e2p" in extra:
            for i in range(len(rows)):
                rows[i]["attn_ent_e2p"] = float(extra["attn_ent_e2p"][i])
                rows[i]["attn_ent_p2e"] = float(extra["attn_ent_p2e"][i])
                rows[i]["attn_max_e2p"] = float(extra["attn_max_e2p"][i])
                rows[i]["attn_max_p2e"] = float(extra["attn_max_p2e"][i])

        pred_df = pd.DataFrame(rows)
        pred_path = outdir / str(args.save_test_predictions)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(pred_path, index=False, encoding="utf-8")

        # per-subject metrics
        per_subj = compute_per_subject_metrics(pred_df)
        subj_path = outdir / "per_subject_metrics.csv"
        per_subj.to_csv(subj_path, index=False, encoding="utf-8")

    summary = {
        "best_f1_macro": float(best_val_f1),
        "best_select_score": float(best_select_score),
        "select_metric": str(args.select_metric),
        "best_ckpt": str(best_path),
        "last_ckpt": str(last_path),
        "test_metrics": test_metrics,
        "predictions_csv": str(pred_path) if pred_path is not None else "",
        "per_subject_csv": str(subj_path) if subj_path is not None else "",
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


# -----------------------------
# CLI
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="HPO-CLD", help="Dataset root (default: ./HPO-CLD)")
    ap.add_argument("--outdir", type=str, default="runs/physioformer_v3", help="Output directory")
    ap.add_argument("--seed", type=int, default=42)

    # optimization
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--verbose", action="store_true")

    # data integrity check (recommended; runs once at startup and writes a log)
    ap.add_argument("--skip_data_check", action="store_true", help="Skip pre-flight dataset integrity scan")
    ap.add_argument("--data_check_log", type=str, default="", help="Data-check log file path (default: <outdir>/data_check.log)")
    ap.add_argument("--no_drop_bad_subjects", action="store_true", help="Do not drop subjects with critical data issues (may crash later)")

    # windowing
    ap.add_argument("--window_sec", type=float, default=10.0)
    ap.add_argument("--stride_sec", type=float, default=5.0)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--ppg_seq_len", type=int, default=0, help="Independent PPG seq length (0=same as seq_len)")
    ap.add_argument("--cache_dir", type=str, default="cache_mm", help="Cache dir for per-window sequences/features")
    ap.add_argument("--no_augment", action="store_true", help="Disable lightweight train-time augmentation.")

    # model
    ap.add_argument("--patch_size", type=int, default=8)
    ap.add_argument("--patch_kernel", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers_eye", type=int, default=2)
    ap.add_argument("--n_layers_ppg", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_static_gate", action="store_true", help="Enable ocular-conditioned reliability gating for static physiology.")
    ap.add_argument("--static_gate_hidden", type=int, default=64)
    ap.add_argument("--static_gate_floor", type=float, default=0.25)
    ap.add_argument("--split_static_modalities", action="store_true", help="Encode ocular-static and cardio-static features with separate sub-branches.")
    ap.add_argument("--use_residual_static_fusion", action="store_true", help="Use eye-first residual adapter for static physiology.")
    ap.add_argument("--residual_static_hidden", type=int, default=128)
    ap.add_argument("--residual_static_init_gate_bias", type=float, default=-2.0)

    # ablation toggles
    ap.add_argument("--no_cross_attn", action="store_true")
    ap.add_argument("--no_bilinear", action="store_true")
    ap.add_argument("--no_eye", action="store_true")
    ap.add_argument("--no_ppg", action="store_true")
    ap.add_argument("--no_static", action="store_true")
    ap.add_argument("--no_regression_head", action="store_true")

    # split control
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--train_subjects", type=str, default="", help="Comma-separated subject folder names (optional)")
    ap.add_argument("--val_subjects", type=str, default="", help="Comma-separated subject folder names (optional)")
    ap.add_argument("--test_subjects", type=str, default="", help="Comma-separated subject folder names (optional)")

    # losses
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--reg_weight", type=float, default=0.5)
    ap.add_argument("--contrastive_weight", type=float, default=0.1)
    ap.add_argument("--contrastive_temp", type=float, default=0.07)
    ap.add_argument(
        "--select_metric",
        type=str,
        default="f1",
        choices=["f1", "composite"],
        help="Validation metric used to select best checkpoint.",
    )
    ap.add_argument("--early_stop_patience", type=int, default=15)

    # outputs
    ap.add_argument("--save_test_predictions", type=str, default="predictions_test.csv", help="Write test predictions CSV into outdir")
    ap.add_argument("--test_attn_summary", action="store_true", help="Also export attention entropy/max summaries (slower)")
    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)

    subject_dirs = list_participants(root)
    if not subject_dirs:
        raise RuntimeError(f"No participant folders found under: {root.resolve()}")

    # explicit subject lists override default split
    train_subjects = parse_subject_list(args.train_subjects)
    val_subjects = parse_subject_list(args.val_subjects)
    test_subjects = parse_subject_list(args.test_subjects)

    if train_subjects or val_subjects or test_subjects:
        if not (train_subjects and val_subjects and test_subjects):
            raise ValueError("If using explicit subject lists, you must provide --train_subjects, --val_subjects, --test_subjects all together.")
        train_dirs = resolve_dirs(root, train_subjects)
        val_dirs = resolve_dirs(root, val_subjects)
        test_dirs = resolve_dirs(root, test_subjects)
    else:
        train_dirs, val_dirs, test_dirs = default_subject_split(subject_dirs, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    train_one_split(root=root, outdir=outdir, seed=args.seed, train_dirs=train_dirs, val_dirs=val_dirs, test_dirs=test_dirs, args=args)


if __name__ == "__main__":
    main()
