# -*- coding: utf-8 -*-
"""
Fair comparison suite for HPO-CLD.

Protocol:
- identical 98-subject filtered cohort
- identical subject-wise 5-fold outer CV
- identical train/val/test partitions inside each fold
- subject-level metrics / confidence intervals
- real outputs written from model runs
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OURS_DIR = _PROJECT_ROOT / "PF-OS"
for _p in [str(_PROJECT_ROOT), str(_OURS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import train as ours_train
from data import MultiModalWindowDataset, StreamConfig, validate_dataset
from net import ModelConfig as OursModelConfig
from net import PhysioFormerNet
from paper_utils import write_latex_table
from stats_utils import bootstrap_ci


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_tfn_net = _load_module("tfn_net_runtime", _PROJECT_ROOT / "TFN" / "net.py")
TFNModelConfig = _tfn_net.ModelConfig
HybridFusionNet = _tfn_net.HybridFusionNet

CLASSICAL_ML = {"KNN", "Gaussian NB", "LDA", "SVM", "Logistic Regression", "Random Forest"}
STATIC_DL = {"TFN"}
SEQUENCE_DL = {"MLP", "CNN", "LSTM"}
OURS_MODEL = "PhysioFormer-OS"
ORDERED_MODELS = [
    "KNN",
    "Gaussian NB",
    "LDA",
    "SVM",
    "Logistic Regression",
    "Random Forest",
    "TFN",
    "MLP",
    "CNN",
    "LSTM",
    OURS_MODEL,
]


def make_folds(subjects: List[str], k: int, seed: int, val_ratio_in_train: float) -> List[Dict[str, Any]]:
    rng = np.random.RandomState(seed)
    subs = list(subjects)
    rng.shuffle(subs)
    test_folds = [list(x) for x in np.array_split(subs, k)]
    test_folds = [f for f in test_folds if len(f) > 0]

    folds: List[Dict[str, Any]] = []
    for i, test_subjects in enumerate(test_folds):
        remaining = [s for s in subjects if s not in set(test_subjects)]
        rng_in = np.random.RandomState(seed + 1000 + i)
        rng_in.shuffle(remaining)
        n_val = max(1, int(round(len(remaining) * float(val_ratio_in_train))))
        n_val = min(n_val, max(1, len(remaining) - 1))
        folds.append(
            {
                "fold": i,
                "train_subjects": remaining[n_val:],
                "val_subjects": remaining[:n_val],
                "test_subjects": test_subjects,
            }
        )
    return folds


def ensure_protocol(
    root: Path,
    outdir: Path,
    k_folds: int,
    seed: int,
    val_ratio_in_train: float,
    window_sec: float,
    stride_sec: float,
) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    protocol_path = outdir / "protocol.json"
    if protocol_path.exists():
        return json.loads(protocol_path.read_text(encoding="utf-8"))

    subject_dirs = ours_train.list_participants(root)
    summary = validate_dataset(
        root=root,
        participant_dirs=subject_dirs,
        log_path=outdir / "data_check.log",
        window_sec=window_sec,
        stride_sec=stride_sec,
    )
    ok_subjects = sorted(summary.get("ok_subjects", []))
    protocol = {
        "root": str(root.resolve()),
        "window_sec": window_sec,
        "stride_sec": stride_sec,
        "seed": seed,
        "k_folds": k_folds,
        "val_ratio_in_train": val_ratio_in_train,
        "ok_subjects": ok_subjects,
        "dropped_subjects": sorted([p.name for p in subject_dirs if p.name not in set(ok_subjects)]),
        "folds": make_folds(ok_subjects, k_folds, seed, val_ratio_in_train),
    }
    protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    return protocol


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

    if y_true.size == 0:
        return {"acc": 0.0, "recall_macro": 0.0, "f1_macro": 0.0, "auc_ovr_macro": 0.0}

    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    auc = float("nan")
    if y_proba is not None and y_proba.ndim == 2 and len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            pass
    out["auc_ovr_macro"] = auc
    return out


def compute_per_subject_metrics(
    participants: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> pd.DataFrame:
    parts = np.asarray(list(participants), dtype=object)
    rows: List[Dict[str, Any]] = []
    for pid in np.unique(parts):
        mask = parts == pid
        m = compute_metrics(y_true[mask], y_pred[mask], y_proba[mask] if y_proba is not None else None)
        rows.append({"participant": str(pid), **m, "n_windows": int(mask.sum())})
    return pd.DataFrame(rows)


def mean_metric(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def format_metric(x: float, digits: int = 3) -> str:
    return "NA" if not np.isfinite(x) else f"{x:.{digits}f}"


def model_type(model_name: str) -> str:
    return "Classical ML" if model_name in CLASSICAL_ML else "Deep Learning"


def load_static_feature_cache(data_dir: Path, ok_subjects: Iterable[str]) -> Dict[str, np.ndarray]:
    npz = np.load(data_dir / "features.npz", allow_pickle=True)
    X = npz["X"].astype(np.float32)
    y = npz["y"].astype(np.int64)
    participants = np.asarray(npz["participants"], dtype=object)
    ok = set(ok_subjects)
    mask = np.array([str(p) in ok for p in participants], dtype=bool)
    return {"X": X[mask], "y": y[mask], "participants": participants[mask]}


def subset_static_fold(cache: Dict[str, np.ndarray], fold_cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    participants = cache["participants"]

    def _mask(names: Sequence[str]) -> np.ndarray:
        s = set(map(str, names))
        return np.array([str(p) in s for p in participants], dtype=bool)

    out = {}
    for split in ["train", "val", "test"]:
        mask = _mask(fold_cfg[f"{split}_subjects"])
        out[f"X_{split}"] = cache["X"][mask]
        out[f"y_{split}"] = cache["y"][mask]
        out[f"p_{split}"] = participants[mask]
    return out


def fit_tfn_model(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    X_val_s: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    device: torch.device,
):
    grid = [
        {"lr": 5e-5, "weight_decay": 0.0, "dropout": 0.2, "epochs": 45},
        {"lr": 1e-4, "weight_decay": 1e-4, "dropout": 0.2, "epochs": 45},
        {"lr": 3e-4, "weight_decay": 1e-4, "dropout": 0.3, "epochs": 35},
    ]

    ds_train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train_s[:, :16].astype(np.float32)),
        torch.from_numpy(X_train_s[:, 16:].astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    Xev = torch.from_numpy(X_val_s[:, :16].astype(np.float32))
    Xhv = torch.from_numpy(X_val_s[:, 16:].astype(np.float32))

    counts = np.bincount(y_train, minlength=3).astype(np.float32)
    w = counts.sum() / np.maximum(counts, 1.0)
    w = w / np.mean(w)
    class_weights = torch.tensor(w, dtype=torch.float32, device=device)

    best_model = None
    best_cfg: Dict[str, Any] = {}
    best_score = -np.inf

    for cfg in grid:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = HybridFusionNet(TFNModelConfig(fusion="tfn", dropout=cfg["dropout"])).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        loader = DataLoader(ds_train, batch_size=64, shuffle=True)
        best_local_state = None
        best_local_score = -np.inf
        patience = 0

        for _ in range(int(cfg["epochs"])):
            model.train()
            for xe, xh, y in loader:
                xe = xe.to(device)
                xh = xh.to(device)
                y = y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xe, xh), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                logits = model(Xev.to(device), Xhv.to(device))
                proba = torch.softmax(logits, dim=1).cpu().numpy()
            score = compute_metrics(y_val, np.argmax(proba, axis=1), proba)["f1_macro"]
            if score > best_local_score + 1e-6:
                best_local_state = copy.deepcopy(model.state_dict())
                best_local_score = score
                patience = 0
            else:
                patience += 1
                if patience >= 6:
                    break

        if best_local_state is not None and best_local_score > best_score:
            model.load_state_dict(best_local_state)
            best_model = copy.deepcopy(model).to(device)
            best_cfg = cfg
            best_score = float(best_local_score)

    if best_model is None:
        raise RuntimeError("TFN training failed.")
    return best_model, best_cfg, best_score


def fit_static_model(model_name: str, fold_data: Dict[str, np.ndarray], seed: int, device: torch.device):
    X_train = fold_data["X_train"]
    y_train = fold_data["y_train"]
    X_val = fold_data["X_val"]
    y_val = fold_data["y_val"]
    X_test = fold_data["X_test"]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    best_model = None
    best_cfg: Dict[str, Any] = {}
    best_score = -np.inf

    def _update(model, cfg):
        nonlocal best_model, best_cfg, best_score
        proba = model.predict_proba(X_val_s)
        score = compute_metrics(y_val, model.predict(X_val_s), proba)["f1_macro"]
        if score > best_score:
            best_model = model
            best_cfg = cfg
            best_score = float(score)

    if model_name == "KNN":
        for k in [3, 5, 7, 11]:
            for weights in ["uniform", "distance"]:
                clf = KNeighborsClassifier(n_neighbors=k, weights=weights)
                clf.fit(X_train_s, y_train)
                _update(clf, {"n_neighbors": k, "weights": weights})
    elif model_name == "Gaussian NB":
        for var_smoothing in [1e-9, 1e-8, 1e-7, 1e-6]:
            clf = GaussianNB(var_smoothing=var_smoothing)
            clf.fit(X_train_s, y_train)
            _update(clf, {"var_smoothing": var_smoothing})
    elif model_name == "LDA":
        for cfg in [{"solver": "svd"}, {"solver": "lsqr", "shrinkage": "auto"}, {"solver": "eigen", "shrinkage": "auto"}]:
            clf = LinearDiscriminantAnalysis(**cfg)
            clf.fit(X_train_s, y_train)
            _update(clf, cfg)
    elif model_name == "SVM":
        for c in [0.5, 1.0, 2.0, 4.0]:
            for gamma in ["scale", 0.05, 0.1]:
                for class_weight in [None, "balanced"]:
                    cfg = {"C": c, "gamma": gamma, "kernel": "rbf", "probability": True, "random_state": seed, "class_weight": class_weight}
                    clf = SVC(**cfg)
                    clf.fit(X_train_s, y_train)
                    _update(clf, cfg)
    elif model_name == "Logistic Regression":
        for c in [0.25, 0.5, 1.0, 2.0, 4.0]:
            for class_weight in [None, "balanced"]:
                cfg = {"max_iter": 2000, "C": c, "class_weight": class_weight, "random_state": seed}
                clf = LogisticRegression(**cfg)
                clf.fit(X_train_s, y_train)
                _update(clf, cfg)
    elif model_name == "Random Forest":
        for depth in [None, 8, 12]:
            for leaf in [1, 2, 4]:
                for class_weight in [None, "balanced"]:
                    cfg = {"n_estimators": 300, "max_depth": depth, "min_samples_leaf": leaf, "class_weight": class_weight, "random_state": seed, "n_jobs": -1}
                    clf = RandomForestClassifier(**cfg)
                    clf.fit(X_train_s, y_train)
                    _update(clf, cfg)
    elif model_name == "TFN":
        model, cfg, score = fit_tfn_model(X_train_s, y_train, X_val_s, y_val, seed, device)
        model.eval()
        with torch.no_grad():
            xe = torch.from_numpy(X_test_s[:, :16].astype(np.float32)).to(device)
            xh = torch.from_numpy(X_test_s[:, 16:].astype(np.float32)).to(device)
            proba = torch.softmax(model(xe, xh), dim=1).cpu().numpy()
        return np.argmax(proba, axis=1), proba, {"selected_config": cfg, "best_val_f1": score}
    else:
        raise ValueError(f"Unsupported static model: {model_name}")

    proba = best_model.predict_proba(X_test_s)
    return best_model.predict(X_test_s), proba, {"selected_config": best_cfg, "best_val_f1": best_score}


class EyeStaticWindowDataset(Dataset):
    def __init__(
        self,
        participant_dirs: List[Path],
        window_sec: float,
        stride_sec: float,
        seq_len: int,
        cache_dir: Path,
        eye_mean: Optional[np.ndarray] = None,
        eye_std: Optional[np.ndarray] = None,
        static_mean: Optional[np.ndarray] = None,
        static_std: Optional[np.ndarray] = None,
    ):
        self.base = MultiModalWindowDataset(
            participant_dirs=participant_dirs,
            window_sec=window_sec,
            stride_sec=stride_sec,
            seq_len=seq_len,
            stream_cfg=StreamConfig(seq_len=seq_len),
            cache_dir=cache_dir,
            augment=False,
            seed=42,
            verbose=False,
        )
        self.eye_mean = eye_mean
        self.eye_std = eye_std
        self.static_mean = static_mean
        self.static_std = static_std

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        eye_seq, _ppg_seq, x_static, y, _ = self.base[idx]
        eye_seq = np.asarray(eye_seq, dtype=np.float32)
        x_static = np.asarray(x_static, dtype=np.float32)
        if self.eye_mean is not None:
            eye_seq = (eye_seq - self.eye_mean) / (self.eye_std + 1e-6)
        if self.static_mean is not None:
            x_static = (x_static - self.static_mean) / (self.static_std + 1e-6)
        return eye_seq.astype(np.float32), x_static.astype(np.float32), np.int64(y), str(self.base.sample_meta[idx].participant)


def fit_eye_static_normalizers(dataset: EyeStaticWindowDataset):
    eye_arr = []
    stat_arr = []
    for i in range(len(dataset.base)):
        eye_seq, _ppg_seq, x_static, _y, _ = dataset.base[i]
        eye_arr.append(np.asarray(eye_seq, dtype=np.float32))
        stat_arr.append(np.asarray(x_static, dtype=np.float32))
    eye = np.stack(eye_arr, axis=0)
    stat = np.stack(stat_arr, axis=0)
    eye_mean = np.nanmean(eye, axis=(0, 1))
    eye_std = np.nanstd(eye, axis=(0, 1))
    eye_std = np.where(np.isfinite(eye_std) & (eye_std > 1e-6), eye_std, 1.0)
    stat_mean = np.nanmean(stat, axis=0)
    stat_std = np.nanstd(stat, axis=0)
    stat_std = np.where(np.isfinite(stat_std) & (stat_std > 1e-6), stat_std, 1.0)
    return eye_mean.astype(np.float32), eye_std.astype(np.float32), stat_mean.astype(np.float32), stat_std.astype(np.float32)


class MatchedMLP(nn.Module):
    def __init__(self, seq_len: int, in_ch: int, static_dim: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.eye = nn.Sequential(nn.Flatten(), nn.Linear(seq_len * in_ch, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.GELU())
        self.static = nn.Sequential(nn.Linear(static_dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 64), nn.GELU())
        self.head = nn.Sequential(nn.Linear(hidden // 2 + 64, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 3))

    def forward(self, eye_seq: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.eye(eye_seq), self.static(x_static)], dim=1))


class MatchedCNN(nn.Module):
    def __init__(self, in_ch: int, static_dim: int, channels: int = 64, dropout: float = 0.3):
        super().__init__()
        self.eye = nn.Sequential(
            nn.Conv1d(in_ch, channels, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(channels, channels * 2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(channels * 2, channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.static = nn.Sequential(nn.Linear(static_dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 64), nn.GELU())
        self.head = nn.Sequential(nn.Linear(channels * 2 + 64, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 3))

    def forward(self, eye_seq: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.eye(eye_seq.transpose(1, 2)).squeeze(-1), self.static(x_static)], dim=1))


class MatchedLSTM(nn.Module):
    def __init__(self, in_ch: int, static_dim: int, hidden: int = 96, dropout: float = 0.3):
        super().__init__()
        self.rnn = nn.LSTM(input_size=in_ch, hidden_size=hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.static = nn.Sequential(nn.Linear(static_dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 64), nn.GELU())
        self.head = nn.Sequential(nn.Linear(hidden * 2 + 64, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 3))

    def forward(self, eye_seq: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        seq_out, _ = self.rnn(eye_seq)
        return self.head(torch.cat([seq_out.mean(dim=1), self.static(x_static)], dim=1))


def build_sequence_model(arch: str, seq_len: int, cfg: Dict[str, Any]) -> nn.Module:
    if arch == "MLP":
        return MatchedMLP(seq_len=seq_len, in_ch=7, static_dim=32, hidden=cfg["hidden"], dropout=cfg["dropout"])
    if arch == "CNN":
        return MatchedCNN(in_ch=7, static_dim=32, channels=cfg["channels"], dropout=cfg["dropout"])
    if arch == "LSTM":
        return MatchedLSTM(in_ch=7, static_dim=32, hidden=cfg["hidden"], dropout=cfg["dropout"])
    raise ValueError(f"Unsupported sequence model: {arch}")


def sequence_grid(arch: str) -> List[Dict[str, Any]]:
    if arch == "MLP":
        return [{"hidden": 512, "dropout": 0.3, "lr": 3e-4, "weight_decay": 1e-4, "epochs": 40}, {"hidden": 768, "dropout": 0.4, "lr": 1e-3, "weight_decay": 1e-4, "epochs": 35}]
    if arch == "CNN":
        return [{"channels": 64, "dropout": 0.3, "lr": 3e-4, "weight_decay": 1e-4, "epochs": 35}, {"channels": 96, "dropout": 0.35, "lr": 1e-3, "weight_decay": 1e-4, "epochs": 30}]
    if arch == "LSTM":
        return [{"hidden": 96, "dropout": 0.3, "lr": 3e-4, "weight_decay": 1e-4, "epochs": 40}, {"hidden": 128, "dropout": 0.35, "lr": 1e-3, "weight_decay": 1e-4, "epochs": 35}]
    raise ValueError(f"Unsupported sequence model: {arch}")


@torch.no_grad()
def predict_sequence_model(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    ys, yp, prob, participants = [], [], [], []
    for eye_seq, x_static, y, participant in loader:
        eye_seq = eye_seq.to(device)
        x_static = x_static.to(device)
        p = torch.softmax(model(eye_seq, x_static), dim=1)
        ys.append(y.numpy())
        yp.append(torch.argmax(p, dim=1).cpu().numpy())
        prob.append(p.cpu().numpy())
        participants.extend(list(participant))
    return np.concatenate(ys), np.concatenate(yp), np.concatenate(prob), participants


def train_sequence_baseline(
    arch: str,
    root: Path,
    fold_cfg: Dict[str, Any],
    cache_dir: Path,
    device: torch.device,
    window_sec: float,
    stride_sec: float,
    seq_len: int,
):
    train_dirs = ours_train.resolve_dirs(root, fold_cfg["train_subjects"])
    val_dirs = ours_train.resolve_dirs(root, fold_cfg["val_subjects"])
    test_dirs = ours_train.resolve_dirs(root, fold_cfg["test_subjects"])

    train_raw = EyeStaticWindowDataset(train_dirs, window_sec, stride_sec, seq_len, cache_dir)
    eye_mean, eye_std, stat_mean, stat_std = fit_eye_static_normalizers(train_raw)
    train_ds = EyeStaticWindowDataset(train_dirs, window_sec, stride_sec, seq_len, cache_dir, eye_mean, eye_std, stat_mean, stat_std)
    val_ds = EyeStaticWindowDataset(val_dirs, window_sec, stride_sec, seq_len, cache_dir, eye_mean, eye_std, stat_mean, stat_std)
    test_ds = EyeStaticWindowDataset(test_dirs, window_sec, stride_sec, seq_len, cache_dir, eye_mean, eye_std, stat_mean, stat_std)

    y_train = np.array([train_ds[i][2] for i in range(len(train_ds))], dtype=np.int64)
    counts = np.bincount(y_train, minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / np.mean(weights)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    dl_train = DataLoader(train_ds, batch_size=64, shuffle=True)
    dl_val = DataLoader(val_ds, batch_size=128, shuffle=False)
    dl_test = DataLoader(test_ds, batch_size=128, shuffle=False)

    best_model = None
    best_cfg: Dict[str, Any] = {}
    best_score = -np.inf

    for cfg in sequence_grid(arch):
        torch.manual_seed(int(fold_cfg["fold"]) + 42)
        np.random.seed(int(fold_cfg["fold"]) + 42)
        model = build_sequence_model(arch, seq_len, cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        best_local_state = None
        best_local_score = -np.inf
        patience = 0

        for _ in range(int(cfg["epochs"])):
            model.train()
            for eye_seq, x_static, y, _participant in dl_train:
                eye_seq = eye_seq.to(device)
                x_static = x_static.to(device)
                y = y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(eye_seq, x_static), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            val_true, val_pred, val_proba, _ = predict_sequence_model(model, dl_val, device)
            score = compute_metrics(val_true, val_pred, val_proba)["f1_macro"]
            if score > best_local_score + 1e-6:
                best_local_state = copy.deepcopy(model.state_dict())
                best_local_score = score
                patience = 0
            else:
                patience += 1
                if patience >= 6:
                    break

        if best_local_state is not None and best_local_score > best_score:
            model.load_state_dict(best_local_state)
            best_model = copy.deepcopy(model).to(device)
            best_cfg = cfg
            best_score = float(best_local_score)

    if best_model is None:
        raise RuntimeError(f"{arch} failed to train.")
    y_true, y_pred, y_proba, participants = predict_sequence_model(best_model, dl_test, device)
    return compute_per_subject_metrics(participants, y_true, y_pred, y_proba), {"selected_config": best_cfg, "best_val_f1": best_score}


def run_physioformer_os(
    root: Path,
    outdir: Path,
    seed: int,
    k_folds: int,
    val_ratio_in_train: float,
    n_boot: int,
    window_sec: float,
    stride_sec: float,
    seq_len: int,
    device: str,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "cv_summary.json").exists():
        return outdir
    cmd = [
        sys.executable,
        str(_OURS_DIR / "run_cv.py"),
        "--root",
        str(root),
        "--seed",
        str(seed),
        "--k_folds",
        str(k_folds),
        "--val_ratio_in_train",
        str(val_ratio_in_train),
        "--n_boot",
        str(n_boot),
        "--window_sec",
        str(window_sec),
        "--stride_sec",
        str(stride_sec),
        "--seq_len",
        str(seq_len),
        "--device",
        device,
        "--cv_outdir",
        str(outdir),
        "--report_dir",
        "report",
        "--cv_profile",
        "consensus",
    ]
    result = subprocess.run(cmd, cwd=_PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError("PhysioFormer-OS CV run failed.")
    return outdir


def summary_from_per_subject(df: pd.DataFrame) -> Dict[str, float]:
    return {"accuracy": mean_metric(df["acc"]), "macro_f1": mean_metric(df["f1_macro"]), "auc_ovr": mean_metric(df["auc_ovr_macro"])}


def save_model_outputs(model_name: str, per_subject_all: pd.DataFrame, meta: Dict[str, Any], outdir: Path) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    per_subject_all.to_csv(outdir / "per_subject_all.csv", index=False, encoding="utf-8")
    summary = {"model": model_name, **summary_from_per_subject(per_subject_all), **meta}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def render_main_results_table(summaries: List[Dict[str, Any]], ours_per_subject: pd.DataFrame, outpath: Path, n_boot: int, seed: int) -> None:
    by_name = {x["model"]: x for x in summaries}
    rows = []
    for name in ORDERED_MODELS:
        s = by_name[name]
        label = r"\textbf{PhysioFormer-OS}" if name == OURS_MODEL else name
        rows.append([label, format_metric(s["accuracy"]), format_metric(s["macro_f1"]), format_metric(s["auc_ovr"]), model_type(name)])
    ci_acc = bootstrap_ci(ours_per_subject["acc"], n_boot=n_boot, seed=seed)
    ci_f1 = bootstrap_ci(ours_per_subject["f1_macro"], n_boot=n_boot, seed=seed + 1)
    ci_auc = bootstrap_ci(ours_per_subject["auc_ovr_macro"], n_boot=n_boot, seed=seed + 2)
    rows.append(["(Ours)", f"[{ci_acc.lo:.3f}, {ci_acc.hi:.3f}]", f"[{ci_f1.lo:.3f}, {ci_f1.hi:.3f}]", f"[{ci_auc.lo:.3f}, {ci_auc.hi:.3f}]", ""])
    write_latex_table(
        outpath,
        caption="Comparison of PhysioFormer-OS against the uniformly re-evaluated baseline suite on HPO-CLD. All methods use the same 98-subject filtered cohort and subject-wise 5-fold CV. PhysioFormer-OS denotes the fixed 0.5/0.5 consensus of the ocular-static expert and the matched eye-only expert.",
        label="tab:main_results",
        col_names=["Method", "Accuracy", "Macro-F1", "AUC (OvR)", "Type"],
        rows=rows,
    )


def params_and_latency() -> pd.DataFrame:
    def _count_params(model: nn.Module) -> int:
        return int(sum(p.numel() for p in model.parameters()))

    def _latency_ms(model: nn.Module, inputs: Tuple[torch.Tensor, ...], repeats: int = 40, warmup: int = 10) -> float:
        model = model.to("cpu").eval()
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(*inputs)
            t0 = time.perf_counter()
            for _ in range(repeats):
                _ = model(*inputs)
            return float((time.perf_counter() - t0) * 1000.0 / repeats)

    b = 32
    eye = torch.randn(b, 256, 7)
    ppg = torch.randn(b, 256, 2)
    stat = torch.randn(b, 32)
    rows = [
        {"model": "TFN", "params": _count_params(HybridFusionNet(TFNModelConfig(fusion="tfn"))), "latency_ms": _latency_ms(HybridFusionNet(TFNModelConfig(fusion="tfn")), (torch.randn(b, 16), torch.randn(b, 16)))},
        {
            "model": "PhysioFormer-OS",
            "params": _count_params(PhysioFormerNet(OursModelConfig(use_eye=True, use_ppg=False, use_static=True, use_bilinear=False, use_regression=False, use_static_gate=True)))
            + _count_params(PhysioFormerNet(OursModelConfig(use_eye=True, use_ppg=False, use_static=False, use_bilinear=False, use_regression=False))),
            "latency_ms": _latency_ms(PhysioFormerNet(OursModelConfig(use_eye=True, use_ppg=False, use_static=True, use_bilinear=False, use_regression=False, use_static_gate=True)), (eye, ppg, stat))
            + _latency_ms(PhysioFormerNet(OursModelConfig(use_eye=True, use_ppg=False, use_static=False, use_bilinear=False, use_regression=False)), (eye, ppg, stat)),
        },
        {"model": "PhysioFormer-S", "params": _count_params(PhysioFormerNet(OursModelConfig(use_eye=True, use_ppg=True, use_static=False, use_bilinear=False, use_regression=False))), "latency_ms": _latency_ms(PhysioFormerNet(OursModelConfig(use_eye=True, use_ppg=True, use_static=False, use_bilinear=False, use_regression=False)), (eye, ppg, stat))},
        {"model": "Eye only", "params": _count_params(PhysioFormerNet(OursModelConfig(use_eye=True, use_ppg=False, use_static=False, use_bilinear=False, use_regression=False))), "latency_ms": _latency_ms(PhysioFormerNet(OursModelConfig(use_eye=True, use_ppg=False, use_static=False, use_bilinear=False, use_regression=False)), (eye, ppg, stat))},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the fair HPO-CLD comparison suite.")
    ap.add_argument("--root", type=str, default="HPO-CLD")
    ap.add_argument("--data_dir", type=str, default="prepared_data")
    ap.add_argument("--outdir", type=str, default="comparison/results/fair_suite")
    ap.add_argument("--models", nargs="*", default=ORDERED_MODELS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k_folds", type=int, default=5)
    ap.add_argument("--val_ratio_in_train", type=float, default=0.15)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--window_sec", type=float, default=10.0)
    ap.add_argument("--stride_sec", type=float, default=5.0)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache_dir", type=str, default="cache_mm")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    device = torch.device(args.device)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    protocol = ensure_protocol(root, outdir, int(args.k_folds), int(args.seed), float(args.val_ratio_in_train), float(args.window_sec), float(args.stride_sec))
    static_cache = load_static_feature_cache(Path(args.data_dir), protocol["ok_subjects"])
    summaries: List[Dict[str, Any]] = []

    for model_name in args.models:
        model_dir = outdir / model_name.replace(" ", "_")
        summary_path = model_dir / "summary.json"
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        if model_name == OURS_MODEL:
            ours_dir = outdir / "PhysioFormer-OS"
            run_physioformer_os(root, ours_dir, int(args.seed), int(args.k_folds), float(args.val_ratio_in_train), int(args.n_boot), float(args.window_sec), float(args.stride_sec), int(args.seq_len), args.device)
            per_subject = pd.read_csv(ours_dir / "per_subject_all.csv")
            summaries.append(save_model_outputs(model_name, per_subject, {"source": str(ours_dir)}, model_dir))
            continue

        fold_tables = []
        fold_meta = []
        for fold_cfg in protocol["folds"]:
            if model_name in CLASSICAL_ML or model_name in STATIC_DL:
                fold_data = subset_static_fold(static_cache, fold_cfg)
                y_pred, y_proba, meta = fit_static_model(model_name, fold_data, int(args.seed) + int(fold_cfg["fold"]), device)
                per_subject = compute_per_subject_metrics(fold_data["p_test"], fold_data["y_test"], y_pred, y_proba)
            elif model_name in SEQUENCE_DL:
                per_subject, meta = train_sequence_baseline(model_name, root, fold_cfg, cache_dir, device, float(args.window_sec), float(args.stride_sec), int(args.seq_len))
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            per_subject["fold"] = int(fold_cfg["fold"])
            fold_tables.append(per_subject)
            fold_meta.append({"fold": int(fold_cfg["fold"]), **meta})

        per_subject_all = pd.concat(fold_tables, axis=0, ignore_index=True)
        summaries.append(save_model_outputs(model_name, per_subject_all, {"fold_meta": fold_meta}, model_dir))

    summary_df = pd.DataFrame(
        [{"method": s["model"], "accuracy": round(float(s["accuracy"]), 3), "macro_f1": round(float(s["macro_f1"]), 3), "auc_ovr": round(float(s["auc_ovr"]), 3), "type": model_type(s["model"])} for s in summaries]
    )
    summary_df["method"] = pd.Categorical(summary_df["method"], ORDERED_MODELS, ordered=True)
    summary_df = summary_df.sort_values("method").reset_index(drop=True)
    (_PROJECT_ROOT / "comparison" / "results").mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(_PROJECT_ROOT / "comparison" / "results" / "comparison_suite.csv", index=False, encoding="utf-8")

    ours_dir = outdir / "PhysioFormer-OS"
    if ours_dir.exists() and (ours_dir / "per_subject_all.csv").exists() and set(ORDERED_MODELS).issubset({s["model"] for s in summaries}):
        ours_subject = pd.read_csv(ours_dir / "per_subject_all.csv")
        render_main_results_table(summaries, ours_subject, _PROJECT_ROOT / "report" / "tables" / "main_results.tex", int(args.n_boot), int(args.seed))
    params_and_latency().to_csv(outdir / "complexity_latency.csv", index=False, encoding="utf-8")

    manifest = {
        "protocol": protocol,
        "models": summaries,
        "comparison_csv": str((_PROJECT_ROOT / "comparison" / "results" / "comparison_suite.csv").resolve()),
        "complexity_csv": str((outdir / "complexity_latency.csv").resolve()),
    }
    (outdir / "suite_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] comparison suite -> {(outdir / 'suite_manifest.json').resolve()}")


if __name__ == "__main__":
    main()
