# -*- coding: utf-8 -*-
"""
test.py

Evaluation / inference for PhysioFormerNet (v3).

This script loads a checkpoint from train.py and evaluates on a set of participants,
exporting window-level predictions.

Example
-------
python test.py --root HPO-CLD --ckpt runs/physioformer_v3/best.pt --participants HPO-CLD010,HPO-CLD011 --save_pred preds.csv

Author: updated by assistant (v3)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from data import MultiModalWindowDataset, StreamConfig, get_static_feature_names, validate_dataset
from net import ModelConfig, PhysioFormerNet


# -----------------------------
# Standardizer (must match train.py)
# -----------------------------
class Standardizer:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mu is not None and self.sd is not None
        return (X - self.mu) / (self.sd + self.eps)

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.mu = np.array(d["mu"], dtype=np.float32) if d.get("mu") is not None else None
        self.sd = np.array(d["sd"], dtype=np.float32) if d.get("sd") is not None else None


class StaticNormWrapper(Dataset):
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
def compute_metrics(
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


def dataset_meta_list(ds: Union[MultiModalWindowDataset, Subset]) -> List[Any]:
    if hasattr(ds, "sample_meta"):
        return list(getattr(ds, "sample_meta"))
    if isinstance(ds, Subset):
        base = ds.dataset
        idxs = list(ds.indices)
        if hasattr(base, "sample_meta"):
            return [base.sample_meta[i] for i in idxs]
    raise AttributeError("Dataset does not expose sample_meta.")


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    ys, yp, prob = [], [], []
    for eye_seq, ppg_seq, x_static, y, y_reg in loader:
        eye_seq = eye_seq.to(device)
        ppg_seq = ppg_seq.to(device)
        x_static = x_static.to(device)

        logits, tlx_hat, _, _, _attn = model(eye_seq, ppg_seq, x_static, return_embeddings=False, return_attn=False)
        p = torch.softmax(logits, dim=1)
        pred = torch.argmax(p, dim=1)

        ys.append(y.numpy())
        yp.append(pred.cpu().numpy())
        prob.append(p.cpu().numpy())

    y_true = np.concatenate(ys, axis=0) if ys else np.array([], dtype=int)
    y_pred = np.concatenate(yp, axis=0) if yp else np.array([], dtype=int)
    y_proba = np.concatenate(prob, axis=0) if prob else np.zeros((0, 3), dtype=np.float32)
    return y_true, y_pred, y_proba


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="HPO-CLD")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--participants", type=str, default="", help="Comma-separated participant folder names (optional)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--window_sec", type=float, default=10.0)
    ap.add_argument("--stride_sec", type=float, default=5.0)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--cache_dir", type=str, default="cache_mm")
    ap.add_argument("--skip_data_check", action="store_true", help="Skip pre-flight dataset integrity scan")
    ap.add_argument("--data_check_log", type=str, default="", help="Data-check log file (default: <ckpt_dir>/data_check_test.log)")
    ap.add_argument("--no_drop_bad_subjects", action="store_true", help="Do not drop subjects with critical data issues (may crash later)")
    ap.add_argument("--save_pred", type=str, default="")
    args = ap.parse_args()

    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = PhysioFormerNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    std = Standardizer()
    std.load_state_dict(ckpt["static_standardizer"])

    root = Path(args.root)
    pdirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("hpo-cld")])
    if not pdirs:
        raise RuntimeError(f"No participant folders found: {root.resolve()}")

    if args.participants.strip():
        wanted = set([s.strip() for s in args.participants.split(",") if s.strip()])
        pdirs = [p for p in pdirs if p.name in wanted]
        if not pdirs:
            raise RuntimeError("No matching participant folders found for --participants")


    # -----------------------------
    # Pre-flight dataset integrity check (optional)
    # -----------------------------
    if not args.skip_data_check:
        log_path = Path(args.data_check_log).expanduser() if args.data_check_log else (Path(args.ckpt).parent / "data_check_test.log")
        summary = validate_dataset(
            root=root,
            participant_dirs=pdirs,
            log_path=log_path,
            window_sec=float(args.window_sec),
            stride_sec=float(args.stride_sec),
        )
        if not args.no_drop_bad_subjects:
            ok = set(summary.get("ok_subjects", []))
            dropped = [p.name for p in pdirs if p.name not in ok]
            if dropped:
                print(f"[data-check] Dropped {len(dropped)} subjects with critical issues before testing. See: {log_path.resolve()}")
            pdirs = [p for p in pdirs if p.name in ok]
        else:
            print(f"[data-check] Completed. See: {log_path.resolve()} (keeping bad subjects per --no_drop_bad_subjects)")
        if not pdirs:
            raise RuntimeError("No valid participants left after data check. Inspect data_check_test.log or pass --no_drop_bad_subjects.")
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    ds_raw = MultiModalWindowDataset(
        pdirs, window_sec=args.window_sec, stride_sec=args.stride_sec,
        seq_len=args.seq_len,
        stream_cfg=StreamConfig(seq_len=args.seq_len, ppg_seq_len=cfg.ppg_seq_len),
        cache_dir=cache_dir, augment=False, seed=42, verbose=False,
    )
    ds = StaticNormWrapper(ds_raw, std)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    y_true, y_pred, y_proba = predict(model, dl, device)
    metrics = compute_metrics(y_true, y_pred, y_proba)

    print(json.dumps({"metrics": metrics, "n_windows": int(len(ds_raw))}, ensure_ascii=False, indent=2))

    if args.save_pred.strip():
        metas = dataset_meta_list(ds_raw)
        rows = []
        for i, meta in enumerate(metas):
            rows.append(
                {
                    "participant": str(getattr(meta, "participant")),
                    "t_start_us": int(getattr(meta, "t_start_us")),
                    "t_end_us": int(getattr(meta, "t_end_us")),
                    "y_true": int(y_true[i]) if i < len(y_true) else int(getattr(meta, "y")),
                    "y_pred": int(y_pred[i]) if i < len(y_pred) else -1,
                    "p_low": float(y_proba[i, 0]) if i < len(y_proba) else float("nan"),
                    "p_med": float(y_proba[i, 1]) if i < len(y_proba) else float("nan"),
                    "p_high": float(y_proba[i, 2]) if i < len(y_proba) else float("nan"),
                }
            )
        out = pd.DataFrame(rows)
        out_path = Path(args.save_pred)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[saved] predictions -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
