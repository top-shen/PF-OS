# paper_utils.py
# -*- coding: utf-8 -*-
"""Utilities for generating LaTeX tables for the paper."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional


def latex_escape(s: str) -> str:
    """Escape special LaTeX characters (minimal)."""
    for ch in ["&", "%", "$", "#", "_", "{", "}"]:
        s = s.replace(ch, "\\" + ch)
    return s


def write_latex_table(
    path: Path,
    caption: str,
    label: str,
    col_names: List[str],
    rows: List[List[str]],
    notes: str = "",
) -> None:
    """Write a simple LaTeX table to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ncols = len(col_names)
    col_spec = "l" + "c" * (ncols - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(col_names) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if notes:
        lines.append(f"\\\\\\footnotesize{{{notes}}}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines), encoding="utf-8")


_PLACEHOLDER_TABLES = ["main_results.tex", "ablation.tex", "stats_tests.tex", "hrv_consistency.tex"]


def write_placeholder_tables(report_dir: Path) -> None:
    """Create empty placeholder .tex files so LaTeX compiles before experiments run."""
    tdir = Path(report_dir) / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    for name in _PLACEHOLDER_TABLES:
        p = tdir / name
        if not p.exists():
            p.write_text(f"% placeholder — will be overwritten by experiment scripts\n", encoding="utf-8")
