# baseline_common.py
# -*- coding: utf-8 -*-
"""
Shared data loading and evaluation utilities for traditional ML baselines
(KNN, Logistic Regression, Naive Bayes, Random Forest).

Uses PF-OS/data.py preprocessing pipeline for feature extraction to ensure
fair comparison with the main PhysioFormer model.

Features: 16 eye + 16 PPG = 32 static features per window.
Window: 10s / 5s stride (matching PF-OS).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Add PF-OS/ to path for imports
_OURS_DIR = Path(__file__).resolve().parent / "PF-OS"
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
    get_static_feature_names,
)


def list_participants(root: Path) -> List[Path]:
    """List participant directories under root."""
    root = Path(root)
    pdirs = sorted([p for p in root.iterdir()
                    if p.is_dir() and p.name.lower().startswith("hpo-cld")])
    if not pdirs:
        raise RuntimeError(f"No participant folders found under: {root.resolve()}")
    return pdirs


def subject_split(dirs: List[Path], seed: int = 42,
                  train_ratio: float = 0.8) -> Tuple[List[Path], List[Path]]:
    """Subject-wise train/test split."""
    rng = np.random.RandomState(seed)
    idxs = np.arange(len(dirs))
    rng.shuffle(idxs)
    n_train = max(1, int(len(dirs) * train_ratio))
    train_dirs = [dirs[i] for i in idxs[:n_train]]
    test_dirs = [dirs[i] for i in idxs[n_train:]] or train_dirs.copy()
    return train_dirs, test_dirs


def _extract_features_one_participant(
    pdir: Path,
    window_sec: float = 10.0,
    stride_sec: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 32-dim static features for all windows of one participant.

    Returns (X, y) where X is (N, 32) and y is (N,).
    """
    tobii = find_best_file(pdir, "*tobii*.csv")
    bitalino = find_best_file(pdir, "*bitalino*.csv")
    labels = find_best_file(pdir, "*labels*.csv")

    stream = ParticipantStreams(tobii, bitalino)
    lab = pd.read_csv(labels)

    window_us = int(window_sec * 1e6)
    stride_us = int(stride_sec * 1e6)

    X_list, y_list = [], []
    for _, r in lab.iterrows():
        t0 = int(r["time_start"])
        t1 = int(r["time_end"])
        y = map_task_to_class(r.get("task_difficulty", ""))
        start = t0
        while start + window_us <= t1:
            # Eye features (16-dim)
            _t_e, pupil, gaze, pos, valid = stream.slice_eye(start, start + window_us)
            xe = eye_static_features(pupil=pupil, gaze_dir=gaze,
                                     pupil_pos=pos, valid=valid, fs=stream.fs_eye)

            # PPG features (16-dim)
            _t_p, ppg_raw = stream.slice_ppg(start, start + window_us)
            ppg_clean = ppg_raw.astype(np.float32).copy()
            art_mask = ppg_artifact_mask(ppg_clean, threshold_mad=5.0)
            if not np.all(art_mask) and np.sum(art_mask) >= 2:
                ppg_clean[~art_mask] = np.nan
                ppg_clean = _linear_fill_nan(ppg_clean).astype(np.float32)
            ppg_f = bandpass_ppg(ppg_clean, fs=stream.fs_ppg)
            ppg_stat = ppg_static_features(ppg_f, fs=stream.fs_ppg)  # 14-dim
            ppg_win_med = float(np.nanmedian(ppg_f))
            ppg_win_mad = float(np.nanmedian(np.abs(ppg_f - ppg_win_med)))
            xh = np.concatenate([ppg_stat,
                                 np.array([safe_float(ppg_win_med),
                                           safe_float(ppg_win_mad)], dtype=np.float32)])

            x = np.concatenate([xe, xh])  # 32-dim
            X_list.append(x)
            y_list.append(y)
            start += stride_us

    if not X_list:
        return np.zeros((0, 32), dtype=np.float32), np.array([], dtype=int)
    return np.stack(X_list).astype(np.float32), np.array(y_list, dtype=int)


def extract_features_for_dirs(
    dirs: List[Path],
    window_sec: float = 10.0,
    stride_sec: float = 5.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features for a list of participant directories.
    Returns (X, y) where X is (N_total, 32) and y is (N_total,).
    """
    Xs, ys = [], []
    for i, pdir in enumerate(dirs):
        if verbose:
            print(f"[{i+1}/{len(dirs)}] extracting features: {pdir.name}")
        X_p, y_p = _extract_features_one_participant(pdir, window_sec, stride_sec)
        if X_p.shape[0] > 0:
            Xs.append(X_p)
            ys.append(y_p)
    if not Xs:
        return np.zeros((0, 32), dtype=np.float32), np.array([], dtype=int)
    return np.concatenate(Xs), np.concatenate(ys)


def evaluate_and_report(y_true: np.ndarray, y_pred: np.ndarray,
                        y_proba: np.ndarray = None) -> Dict[str, float]:
    """Compute standard classification metrics."""
    from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

    if y_true.size == 0:
        return {"acc": 0.0, "recall_macro": 0.0, "f1_macro": 0.0, "auc_ovr_macro": 0.0}

    out: Dict[str, float] = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    auc = 0.0
    if y_proba is not None and y_proba.ndim == 2 and len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            pass
    out["auc_ovr_macro"] = auc
    return out


def load_prepared_data(data_dir: Path):
    """
    Load preprocessed features from prepared_data/features.npz.
    Returns (X_train, y_train, X_test, y_test).
    """
    data_dir = Path(data_dir)
    npz = np.load(data_dir / "features.npz", allow_pickle=True)
    X = npz["X"].astype(np.float32)
    y = npz["y"].astype(np.int64)
    is_train = npz["is_train"].astype(bool)
    is_test = npz["is_test"].astype(bool)
    return X[is_train], y[is_train], X[is_test], y[is_test]
