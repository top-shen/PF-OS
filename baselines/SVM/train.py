# train.py
# -*- coding: utf-8 -*-
"""
SVM baseline for cognitive workload classification on HPO-CLD.

Preprocessing: 32-dim static features (16 eye + 16 PPG) from PF-OS/data.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from baseline_common import list_participants, subject_split, extract_features_for_dirs, evaluate_and_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="HPO-CLD")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--window_sec", type=float, default=10.0)
    ap.add_argument("--stride_sec", type=float, default=5.0)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--gamma", type=str, default="scale")
    ap.add_argument("--outdir", type=str, default="")
    args = ap.parse_args()

    root = Path(args.root)
    dirs = list_participants(root)
    train_dirs, test_dirs = subject_split(dirs, args.seed, args.train_ratio)

    X_train, y_train = extract_features_for_dirs(train_dirs, args.window_sec, args.stride_sec)
    X_test, y_test = extract_features_for_dirs(test_dirs, args.window_sec, args.stride_sec)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = SVC(C=args.C, gamma=args.gamma, kernel="rbf", probability=True, random_state=args.seed)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    metrics = evaluate_and_report(y_test, y_pred, y_proba)

    outdir = Path(args.outdir) if args.outdir else Path(__file__).parent
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "results.json"
    out_path.write_text(
        json.dumps(
            {
                "model": "SVM",
                "metrics": metrics,
                "train_subjects": [p.name for p in train_dirs],
                "test_subjects": [p.name for p in test_dirs],
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "args": vars(args),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"model": "SVM", **metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
