# -*- coding: utf-8 -*-
"""
stats_utils.py

Journal-grade statistical utilities for subject-wise evaluation.

Designed for the HPO-CLD workload setting where windows within a subject are NOT independent.
Therefore, all inferential statistics default to **subject-level** resampling and testing.

Author: assistant (v3)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple, Optional

import numpy as np

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


@dataclass
class BootstrapCI:
    mean: float
    lo: float
    hi: float
    n: int
    n_boot: int
    ci: float


def bootstrap_ci(
    x: Sequence[float],
    statistic: Callable[[np.ndarray], float] = lambda a: float(np.mean(a)),
    ci: float = 0.95,
    n_boot: int = 10000,
    seed: int = 42,
) -> BootstrapCI:
    """
    Percentile bootstrap CI for a scalar statistic.

    Parameters
    ----------
    x : per-subject values (independent units)
    statistic : function applied to a 1D numpy array
    ci : confidence level, e.g. 0.95
    n_boot : bootstrap replicates
    """
    arr = np.asarray(list(x), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return BootstrapCI(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0, n_boot=n_boot, ci=ci)

    rng = np.random.RandomState(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        samp = arr[rng.randint(0, n, size=n)]
        boots[i] = float(statistic(samp))

    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1.0 - alpha))
    m = float(statistic(arr))
    return BootstrapCI(mean=m, lo=lo, hi=hi, n=n, n_boot=n_boot, ci=ci)


def cohen_d_paired(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """
    Paired Cohen's d based on difference scores.
    """
    d = (x - y).astype(np.float64)
    if d.size == 0:
        return float("nan")
    sd = float(np.std(d, ddof=1)) if d.size > 1 else float(np.std(d))
    if not np.isfinite(sd) or sd < eps:
        return float("nan")
    return float(np.mean(d) / sd)


def paired_significance_tests(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """
    Paired tests between two conditions over subjects.

    Returns p-values (and effect size) for:
      - paired t-test
      - Wilcoxon signed-rank (if scipy available)
      - Cohen's d (paired)
    """
    xx = np.asarray(list(x), dtype=np.float64)
    yy = np.asarray(list(y), dtype=np.float64)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]

    out: Dict[str, float] = {"n": float(xx.size)}
    if xx.size < 2:
        out.update({"p_ttest": float("nan"), "p_wilcoxon": float("nan"), "d_cohen": float("nan")})
        return out

    # paired t-test
    if stats is not None:
        try:
            t = stats.ttest_rel(xx, yy, nan_policy="omit")
            out["p_ttest"] = float(t.pvalue)
        except Exception:
            out["p_ttest"] = float("nan")
    else:
        out["p_ttest"] = float("nan")

    # Wilcoxon
    if stats is not None:
        try:
            w = stats.wilcoxon(xx, yy, zero_method="wilcox", alternative="two-sided", mode="auto")
            out["p_wilcoxon"] = float(w.pvalue)
        except Exception:
            out["p_wilcoxon"] = float("nan")
    else:
        out["p_wilcoxon"] = float("nan")

    out["d_cohen"] = cohen_d_paired(xx, yy)
    return out


def holm_bonferroni(pvals: Sequence[float]) -> List[float]:
    """
    Holm-Bonferroni adjusted p-values.
    """
    p = np.asarray(list(pvals), dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=np.float64)
    for i, idx in enumerate(order):
        adj[idx] = min(1.0, (m - i) * p[idx])
    # enforce monotonicity
    for i in range(1, m):
        idx_prev = order[i - 1]
        idx_cur = order[i]
        adj[idx_cur] = max(adj[idx_cur], adj[idx_prev])
    return [float(x) for x in adj]


def format_mean_ci(ci: BootstrapCI, digits: int = 3) -> str:
    if not np.isfinite(ci.mean):
        return "NA"
    fmt = f"{{:.{digits}f}}"
    return f"{fmt.format(ci.mean)} [{fmt.format(ci.lo)}, {fmt.format(ci.hi)}]"
