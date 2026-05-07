# test.py
# -*- coding: utf-8 -*-
"""
Evaluate / inference for HPO-CLD-style dataset using TFN model.

Preprocessing imported from PF-OS/data.py for consistency.

Author: generated for Renty (v2 — unified preprocessing)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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

# Import shared classes from train.py
from train import (
    HPOCLDWindowDataset,
    Standardizer,
    NormalizeWrapper,
    compute_metrics,
    _extract_eye_features,
    _extract_ppg_features,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
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
    y_proba = np.concatenate(prob) if prob else np.array([], dtype=np.float32)
    return compute_metrics(y_true, y_pred, y_proba), y_true, y_pred, y_proba


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="HPO-CLD")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--window_sec", type=float, default=10.0)
    ap.add_argument("--stride_sec", type=float, default=5.0)
    ap.add_argument("--participants", type=str, default="")
    ap.add_argument("--save_pred", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ModelConfig(**ckpt["model_config"])
    model = HybridFusionNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    eye_std = Standardizer()
    heart_std = Standardizer()
    eye_std.load_state_dict(ckpt["eye_standardizer"])
    heart_std.load_state_dict(ckpt["heart_standardizer"])

    root = Path(args.root)
    pdirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("hpo-cld")])
    if not pdirs:
        raise RuntimeError(f"No participant folders found: {root.resolve()}")

    if args.participants.strip():
        wanted = {s.strip() for s in args.participants.split(",") if s.strip()}
        pdirs = [p for p in pdirs if p.name in wanted]
        if not pdirs:
            raise RuntimeError("No matching participant folders found for --participants")

    ds_raw = HPOCLDWindowDataset(pdirs, window_sec=args.window_sec, stride_sec=args.stride_sec)
    ds = NormalizeWrapper(ds_raw, eye_std, heart_std)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    metrics, y_true, y_pred, y_proba = evaluate(model, dl, device)
    print(json.dumps({"metrics": metrics, "n_windows": int(len(ds))}, ensure_ascii=False, indent=2))

    if args.save_pred.strip():
        rows = []
        for i, (pidx, s, e, y) in enumerate(ds_raw.samples):
            pid = ds_raw.meta[pidx]["dir"].name
            rows.append({
                "participant": pid,
                "t_start_us": int(s),
                "t_end_us": int(e),
                "y_true": int(y_true[i]) if i < len(y_true) else int(y),
                "y_pred": int(y_pred[i]) if i < len(y_pred) else -1,
                "p_low": float(y_proba[i, 0]) if i < len(y_proba) else float("nan"),
                "p_med": float(y_proba[i, 1]) if i < len(y_proba) else float("nan"),
                "p_high": float(y_proba[i, 2]) if i < len(y_proba) else float("nan"),
            })
        out = pd.DataFrame(rows)
        out_path = Path(args.save_pred)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[saved] predictions -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
