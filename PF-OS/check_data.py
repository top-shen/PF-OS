# -*- coding: utf-8 -*-
"""
check_data.py

Standalone dataset integrity checker for the HPO-CLD multimodal dataset.

This is a lightweight *pre-flight* scan that catches common problems early:
- missing tobii / bitalino / labels files
- empty files
- missing required columns (e.g., timestamp 't')
- label sanity (time_start < time_end)
- coarse overlap between labels and stream timestamps (first/last)

It writes:
- a human-readable log (.log)
- a machine-readable JSON summary with the same name (.json)

Example
-------
python check_data.py --root HPO-CLD --log runs/cv_v3/data_check.log --window_sec 10 --stride_sec 5

Author: assistant (v3)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data import validate_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="HPO-CLD")
    ap.add_argument("--log", type=str, default="data_check.log")
    ap.add_argument("--window_sec", type=float, default=10.0)
    ap.add_argument("--stride_sec", type=float, default=5.0)
    ap.add_argument("--participants", type=str, default="", help="Comma-separated participant folder names (optional)")
    args = ap.parse_args()

    root = Path(args.root)
    pdirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if args.participants.strip():
        wanted = set([s.strip() for s in args.participants.split(",") if s.strip()])
        pdirs = [p for p in pdirs if p.name in wanted]

    if not pdirs:
        raise RuntimeError(f"No participant folders found under: {root.resolve()}")

    validate_dataset(
        root=root,
        participant_dirs=pdirs,
        log_path=Path(args.log),
        window_sec=float(args.window_sec),
        stride_sec=float(args.stride_sec),
    )


if __name__ == "__main__":
    main()
