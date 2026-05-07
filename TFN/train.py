# train.py
# -*- coding: utf-8 -*-
"""
Train Hybrid TFN / XAttn fusion network on HPO-CLD-style folders.

Preprocessing is imported from PF-OS/data.py to ensure consistency:
  - 16-dim eye static features (eye_static_features)
  - 16-dim PPG static features (ppg_static_features + amplitude stats)
  - SQI-based PPG channel selection
  - MAD artifact detection + Butterworth bandpass filtering
  - 10s window / 5s stride

Author: generated for Renty (v2 — unified preprocessing)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import preprocessing from PF-OS/data.py
_OURS_DIR = Path(__file__).resolve().parent.parent / "PF-OS"
if str(_OURS_DIR) not in sys.path:
    sys.path.insert(0, str(_OURS_DIR))

from data import (
    ParticipantStreams,
    eye_static_features,
    ppg_static_features,
    ppg_artifact_mask,
    bandpass_ppg,
    safe_float,
    map_task_to_class,
    find_best_file,
    _linear_fill_nan,
)

from net import ModelConfig, HybridFusionNet


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
# Feature extraction helpers (using PF-OS pipeline)
# -----------------------------
def _extract_eye_features(stream: ParticipantStreams, s_us: int, e_us: int) -> np.ndarray:
    """Extract 16-dim eye static features using PF-OS/data.py pipeline."""
    _t_e, pupil, gaze, pos, valid = stream.slice_eye(s_us, e_us)
    return eye_static_features(pupil=pupil, gaze_dir=gaze, pupil_pos=pos,
                               valid=valid, fs=stream.fs_eye)


def _extract_ppg_features(stream: ParticipantStreams, s_us: int, e_us: int) -> np.ndarray:
    """Extract 16-dim PPG static features using PF-OS/data.py pipeline."""
    _t_p, ppg_raw = stream.slice_ppg(s_us, e_us)
    # Artifact detection + interpolation
    ppg_clean = ppg_raw.astype(np.float32).copy()
    art_mask = ppg_artifact_mask(ppg_clean, threshold_mad=5.0)
    if not np.all(art_mask) and np.sum(art_mask) >= 2:
        ppg_clean[~art_mask] = np.nan
        ppg_clean = _linear_fill_nan(ppg_clean).astype(np.float32)
    # Bandpass filter
    ppg_f = bandpass_ppg(ppg_clean, fs=stream.fs_ppg)
    # 14-dim HRV/spectral features
    ppg_stat = ppg_static_features(ppg_f, fs=stream.fs_ppg)
    # 2-dim amplitude stats (matching PF-OS/_build_one)
    ppg_win_med = float(np.nanmedian(ppg_f))
    ppg_win_mad = float(np.nanmedian(np.abs(ppg_f - ppg_win_med)))
    ppg_amp = np.array([safe_float(ppg_win_med), safe_float(ppg_win_mad)], dtype=np.float32)
    return np.concatenate([ppg_stat, ppg_amp])  # 16-dim


# -----------------------------
# Dataset
# -----------------------------
class HPOCLDWindowDataset(Dataset):
    """
    Each item:
      xe: (16,) float32  — eye static features
      xh: (16,) float32  — PPG static features
      y:  int64 in {0,1,2}
    """
    def __init__(
        self,
        participant_dirs: List[Path],
        window_sec: float = 10.0,
        stride_sec: float = 5.0,
        cache_dir: Optional[Path] = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.participant_dirs = participant_dirs
        self.window_us = int(window_sec * 1e6)
        self.stride_us = int(stride_sec * 1e6)
        self.cache_dir = cache_dir
        self.verbose = verbose

        self.meta: List[Dict[str, Path]] = []
        self.samples: List[Tuple[int, int, int, int]] = []  # (pidx, start_us, end_us, y)

        for pdir in participant_dirs:
            tobii = find_best_file(pdir, "*tobii*.csv")
            bitalino = find_best_file(pdir, "*bitalino*.csv")
            labels = find_best_file(pdir, "*labels*.csv")
            self.meta.append({"dir": pdir, "tobii": tobii, "bitalino": bitalino, "labels": labels})

        for pidx, m in enumerate(self.meta):
            lab = pd.read_csv(m["labels"])
            required = {"time_start", "time_end", "task_difficulty"}
            if not required.issubset(set(lab.columns)):
                raise RuntimeError(f"labels.csv missing required columns {required}. "
                                   f"got={lab.columns.tolist()} file={m['labels']}")
            for _, r in lab.iterrows():
                t0 = int(r["time_start"])
                t1 = int(r["time_end"])
                y = map_task_to_class(r.get("task_difficulty", ""))
                start = t0
                while start + self.window_us <= t1:
                    self.samples.append((pidx, start, start + self.window_us, y))
                    start += self.stride_us

        self._stream_cache: Dict[int, ParticipantStreams] = {}
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_streams(self, pidx: int) -> ParticipantStreams:
        if pidx not in self._stream_cache:
            m = self.meta[pidx]
            self._stream_cache[pidx] = ParticipantStreams(m["tobii"], m["bitalino"])
        return self._stream_cache[pidx]

    def _cache_file(self, pidx: int) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        pid = self.meta[pidx]["dir"].name
        return self.cache_dir / f"{pid}_winfeat_v2_w{self.window_us}_s{self.stride_us}.npz"

    def _maybe_build_feature_cache(self, pidx: int) -> Optional[Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]]:
        cf = self._cache_file(pidx)
        if cf is None:
            return None
        if cf.exists():
            data = np.load(cf, allow_pickle=True)
            keys, xe, xh = data["keys"], data["xe"], data["xh"]
            return {(int(k[0]), int(k[1])): (a.astype(np.float32), b.astype(np.float32))
                    for k, a, b in zip(keys, xe, xh)}
        if self.verbose:
            print(f"[cache] building features for {self.meta[pidx]['dir'].name} -> {cf}")
        stream = self._load_streams(pidx)
        wins = [(s, e) for (pi, s, e, _) in self.samples if pi == pidx]
        keys_list, xe_list, xh_list = [], [], []
        for (s, e) in wins:
            fe = _extract_eye_features(stream, s, e)
            fh = _extract_ppg_features(stream, s, e)
            keys_list.append((s, e))
            xe_list.append(fe)
            xh_list.append(fh)
        keys_arr = np.array(keys_list, dtype=np.int64)
        xe_arr = np.stack(xe_list, axis=0).astype(np.float32) if xe_list else np.zeros((0, 16), np.float32)
        xh_arr = np.stack(xh_list, axis=0).astype(np.float32) if xh_list else np.zeros((0, 16), np.float32)
        np.savez_compressed(cf, keys=keys_arr, xe=xe_arr, xh=xh_arr)
        return {(int(k[0]), int(k[1])): (a, b) for k, a, b in zip(keys_arr, xe_arr, xh_arr)}

    def __getitem__(self, idx: int):
        pidx, s, e, y = self.samples[idx]
        feat_cache = None
        if self.cache_dir is not None:
            cache_key = f"_featcache_{pidx}"
            if not hasattr(self, cache_key):
                setattr(self, cache_key, self._maybe_build_feature_cache(pidx))
            feat_cache = getattr(self, cache_key)
        if feat_cache is not None and (s, e) in feat_cache:
            xe, xh = feat_cache[(s, e)]
        else:
            stream = self._load_streams(pidx)
            xe = _extract_eye_features(stream, s, e)
            xh = _extract_ppg_features(stream, s, e)
        return (
            torch.from_numpy(xe.astype(np.float32)),
            torch.from_numpy(xh.astype(np.float32)),
            torch.tensor(int(y), dtype=torch.long),
        )


# -----------------------------
# Normalization
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


class NormalizeWrapper(Dataset):
    """Apply per-modality standardization (fit on train, apply to all)."""
    def __init__(self, base: HPOCLDWindowDataset, eye_std: Standardizer, heart_std: Standardizer):
        self.base = base
        self.eye_std = eye_std
        self.heart_std = heart_std

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        xe, xh, y = self.base[idx]
        xe_np = self.eye_std.transform(xe.numpy().reshape(1, -1)).flatten()
        xh_np = self.heart_std.transform(xh.numpy().reshape(1, -1)).flatten()
        return (
            torch.from_numpy(xe_np.astype(np.float32)),
            torch.from_numpy(xh_np.astype(np.float32)),
            y,
        )


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
    out: Dict[str, float] = {}
    if y_true.size == 0:
        return {"acc": 0.0, "recall_macro": 0.0, "f1_macro": 0.0, "auc_ovr_macro": 0.0}
    out["acc"] = float(accuracy_score(y_true, y_pred))
    out["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    auc = 0.0
    if y_proba is not None and y_proba.ndim == 2 and len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            pass
    out["auc_ovr_macro"] = auc
    return out


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    ys, yp, prob = [], [], []
    for xe, xh, y in loader:
        xe, xh = xe.to(device), xh.to(device)
        logits = model(xe, xh)
        p = torch.softmax(logits, dim=1)
        pred = torch.argmax(p, dim=1)
        ys.append(y.cpu().numpy())
        yp.append(pred.cpu().numpy())
        prob.append(p.cpu().numpy())
    y_true = np.concatenate(ys) if ys else np.array([], dtype=int)
    y_pred = np.concatenate(yp) if yp else np.array([], dtype=int)
    y_proba = np.concatenate(prob) if prob else None
    return compute_metrics(y_true, y_pred, y_proba)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="HPO-CLD")
    ap.add_argument("--outdir", type=str, default="runs/hpo_cld")
    ap.add_argument("--fusion", type=str, default="tfn", choices=["tfn", "xattn"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--window_sec", type=float, default=10.0)
    ap.add_argument("--stride_sec", type=float, default=5.0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--cache_dir", type=str, default="cache_feat_v2")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    pdirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("hpo-cld")])
    if len(pdirs) == 0:
        raise RuntimeError(f"No participant folders found under: {root.resolve()}")

    # Subject-wise split
    rng = np.random.RandomState(args.seed)
    idxs = np.arange(len(pdirs))
    rng.shuffle(idxs)
    n_train = max(1, int(len(pdirs) * args.train_ratio))
    train_dirs = [pdirs[i] for i in idxs[:n_train]]
    test_dirs = [pdirs[i] for i in idxs[n_train:]] or train_dirs.copy()

    ds_train_raw = HPOCLDWindowDataset(train_dirs, window_sec=args.window_sec,
                                       stride_sec=args.stride_sec, cache_dir=cache_dir, verbose=True)
    ds_test_raw = HPOCLDWindowDataset(test_dirs, window_sec=args.window_sec,
                                      stride_sec=args.stride_sec, cache_dir=cache_dir, verbose=False)

    # Fit normalization on train only
    eye_feats, heart_feats, ys = [], [], []
    for i in range(len(ds_train_raw)):
        xe, xh, y = ds_train_raw[i]
        eye_feats.append(xe.numpy())
        heart_feats.append(xh.numpy())
        ys.append(int(y))

    Xeye = np.stack(eye_feats) if eye_feats else np.zeros((0, 16), np.float32)
    Xheart = np.stack(heart_feats) if heart_feats else np.zeros((0, 16), np.float32)
    ys = np.array(ys, dtype=int)

    eye_std = Standardizer()
    heart_std = Standardizer()
    eye_std.fit(Xeye)
    heart_std.fit(Xheart)

    ds_train = NormalizeWrapper(ds_train_raw, eye_std, heart_std)
    ds_test = NormalizeWrapper(ds_test_raw, eye_std, heart_std)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=False)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, drop_last=False)

    # Class weights
    counts = np.bincount(ys, minlength=3) if ys.size else np.ones(3, dtype=int)
    w = (counts.sum() / (counts + 1e-6)).astype(np.float32)
    w = w / np.mean(w)
    class_weights = torch.tensor(w, dtype=torch.float32, device=device)

    cfg = ModelConfig(eye_feat_dim=16, heart_feat_dim=16, num_classes=3, fusion=args.fusion)
    model = HybridFusionNet(cfg).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    run_cfg = {
        "args": vars(args),
        "model_config": model.get_config(),
        "train_subjects": [p.name for p in train_dirs],
        "test_subjects": [p.name for p in test_dirs],
        "class_counts": counts.tolist(),
        "class_weights": w.tolist(),
        "eye_standardizer": eye_std.state_dict(),
        "heart_standardizer": heart_std.state_dict(),
    }
    (outdir / "run_config.json").write_text(json.dumps(run_cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    best_acc = -1.0
    best_path = outdir / "best.pt"
    last_path = outdir / "last.pt"

    print(f"[info] root={root.resolve()}")
    print(f"[info] train subjects={len(train_dirs)} test subjects={len(test_dirs)}")
    print(f"[info] train windows={len(ds_train)} test windows={len(ds_test)}")
    print(f"[info] device={device} fusion={args.fusion}")
    print(f"[info] features: eye=16 ppg=16 (ours-compatible)")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_meter = []

        for xe, xh, y in dl_train:
            xe, xh, y = xe.to(device), xh.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
                logits = model(xe, xh)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            loss_meter.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(loss_meter)) if loss_meter else 0.0
        metrics = evaluate(model, dl_test, device)
        dt = time.time() - t0

        log = {
            "epoch": epoch, "train_loss": train_loss,
            "acc": metrics["acc"], "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"], "auc_ovr_macro": metrics["auc_ovr_macro"],
            "sec": dt,
        }
        print(json.dumps(log, ensure_ascii=False))

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": asdict(cfg),
            "args": vars(args),
            "eye_standardizer": eye_std.state_dict(),
            "heart_standardizer": heart_std.state_dict(),
        }
        torch.save(ckpt, last_path)

        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            torch.save(ckpt, best_path)
            print(f"[best] epoch={epoch} acc={best_acc:.4f} -> {best_path}")

    print(f"[done] best_acc={best_acc:.4f} saved at {best_path}")


if __name__ == "__main__":
    main()
