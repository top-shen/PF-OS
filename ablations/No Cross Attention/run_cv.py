# -*- coding: utf-8 -*-
"""
Recovered wrapper for the no-cross-attention transformer baseline.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        str(root / "PF-OS" / "run_cv.py"),
        "--root",
        "HPO-CLD",
        "--cv_outdir",
        "runs/revision_cv_no_cross",
        "--cv_profile",
        "full",
        "--no_static",
        "--no_cross_attn",
    ]
    raise SystemExit(subprocess.call(cmd, cwd=root))


if __name__ == "__main__":
    main()
