# prepare_data.py
# -*- coding: utf-8 -*-
"""
Preprocess HPO-CLD dataset and save 32-dim static features to a unified folder.

Output structure:
  prepared_data/
    features.npz          — X (N, 32), y (N,), participants (N,), window_info (N, 2)
    feature_names.json    — 32 feature names
    split.json            — train/test subject lists

Usage:
  python comparison/prepare_data.py --root HPO-CLD --outdir prepared_data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add project root and PF-OS/ to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OURS_DIR = _PROJECT_ROOT / "PF-OS"
for p in [str(_PROJECT_ROOT), str(_OURS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

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


def extract_one_participant(
    pdir: Path, window_sec: float, stride_sec: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X, y, win_info) for one participant.
    X: (N, 32), y: (N,), win_info: (N, 2) with [start_us, end_us].
    """
    tobii = find_best_file(pdir, "*tobii*.csv")
    bitalino = find_best_file(pdir, "*bitalino*.csv")
    labels = find_best_file(pdir, "*labels*.csv")

    stream = ParticipantStreams(tobii, bitalino)
    lab = pd.read_csv(labels)

    window_us = int(window_sec * 1e6)
    stride_us = int(stride_sec * 1e6)

    X_list, y_list, win_list = [], [], []
    for _, r in lab.iterrows():
        t0, t1 = int(r["time_start"]), int(r["time_end"])
        y = map_task_to_class(r.get("task_difficulty", ""))
        start = t0
        while start + window_us <= t1:
            end = start + window_us
            # Eye features (16-dim)
            _t_e, pupil, gaze, pos, valid = stream.slice_eye(start, end)
            xe = eye_static_features(pupil=pupil, gaze_dir=gaze,
                                     pupil_pos=pos, valid=valid, fs=stream.fs_eye)
            # PPG features (16-dim)
            _t_p, ppg_raw = stream.slice_ppg(start, end)
            ppg_clean = ppg_raw.astype(np.float32).copy()
            art_mask = ppg_artifact_mask(ppg_clean, threshold_mad=5.0)
            if not np.all(art_mask) and np.sum(art_mask) >= 2:
                ppg_clean[~art_mask] = np.nan
                ppg_clean = _linear_fill_nan(ppg_clean).astype(np.float32)
            ppg_f = bandpass_ppg(ppg_clean, fs=stream.fs_ppg)
            ppg_stat = ppg_static_features(ppg_f, fs=stream.fs_ppg)
            ppg_win_med = float(np.nanmedian(ppg_f))
            ppg_win_mad = float(np.nanmedian(np.abs(ppg_f - ppg_win_med)))
            xh = np.concatenate([ppg_stat,
                                 np.array([safe_float(ppg_win_med),
                                           safe_float(ppg_win_mad)], dtype=np.float32)])

            X_list.append(np.concatenate([xe, xh]))
            y_list.append(y)
            win_list.append([start, end])
            start += stride_us

    if not X_list:
        return (np.zeros((0, 32), dtype=np.float32),
                np.array([], dtype=np.int64),
                np.zeros((0, 2), dtype=np.int64))
    return (np.stack(X_list).astype(np.float32),
            np.array(y_list, dtype=np.int64),
            np.array(win_list, dtype=np.int64))


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess HPO-CLD and save features")
    ap.add_argument("--root", type=str, default="HPO-CLD")
    ap.add_argument("--outdir", type=str, default="prepared_data")
    ap.add_argument("--window_sec", type=float, default=10.0)
    ap.add_argument("--stride_sec", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pdirs = sorted([p for p in root.iterdir()
                    if p.is_dir() and p.name.lower().startswith("hpo-cld")])
    if not pdirs:
        raise RuntimeError(f"No participant folders found under: {root.resolve()}")

    # Subject-wise split
    rng = np.random.RandomState(args.seed)
    idxs = np.arange(len(pdirs))
    rng.shuffle(idxs)
    n_train = max(1, int(len(pdirs) * args.train_ratio))
    train_names = [pdirs[i].name for i in idxs[:n_train]]
    test_names = [pdirs[i].name for i in idxs[n_train:]] or train_names.copy()

    train_set = set(train_names)
    test_set = set(test_names)

    # Extract features for all participants
    all_X, all_y, all_pid, all_win = [], [], [], []
    for i, pdir in enumerate(pdirs):
        print(f"[{i+1}/{len(pdirs)}] {pdir.name} ...", end=" ", flush=True)
        X_p, y_p, win_p = extract_one_participant(pdir, args.window_sec, args.stride_sec)
        print(f"{X_p.shape[0]} windows")
        if X_p.shape[0] > 0:
            all_X.append(X_p)
            all_y.append(y_p)
            all_pid.extend([pdir.name] * X_p.shape[0])
            all_win.append(win_p)

    X = np.concatenate(all_X).astype(np.float32)
    y = np.concatenate(all_y).astype(np.int64)
    participants = np.array(all_pid, dtype=object)
    win_info = np.concatenate(all_win).astype(np.int64)

    # Build train/test masks
    is_train = np.array([p in train_set for p in participants], dtype=bool)
    is_test = np.array([p in test_set for p in participants], dtype=bool)

    # Save
    np.savez_compressed(
        outdir / "features.npz",
        X=X, y=y, participants=participants, win_info=win_info,
        is_train=is_train, is_test=is_test,
    )

    split_info = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "train_subjects": train_names,
        "test_subjects": test_names,
        "n_train_windows": int(is_train.sum()),
        "n_test_windows": int(is_test.sum()),
        "n_total_windows": int(len(y)),
        "class_distribution": {
            "train": {str(c): int((y[is_train] == c).sum()) for c in range(3)},
            "test": {str(c): int((y[is_test] == c).sum()) for c in range(3)},
        },
    }
    (outdir / "split.json").write_text(
        json.dumps(split_info, indent=2, ensure_ascii=False), encoding="utf-8")

    feat_names = get_static_feature_names()
    (outdir / "feature_names.json").write_text(
        json.dumps(feat_names, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[done] saved to {outdir.resolve()}")
    print(f"  features.npz: X={X.shape}, y={y.shape}")
    print(f"  train: {is_train.sum()} windows ({len(train_names)} subjects)")
    print(f"  test:  {is_test.sum()} windows ({len(test_names)} subjects)")
    print(f"  class dist (train): {split_info['class_distribution']['train']}")
    print(f"  class dist (test):  {split_info['class_distribution']['test']}")


if __name__ == "__main__":
    main()
