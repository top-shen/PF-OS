# -*- coding: utf-8 -*-
"""Run the fair matched CNN baseline under the shared subject-wise protocol."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, str(root / "comparison" / "run_fair_suite.py"), "--models", "CNN"]
    raise SystemExit(subprocess.call(cmd, cwd=root))


if __name__ == "__main__":
    main()
