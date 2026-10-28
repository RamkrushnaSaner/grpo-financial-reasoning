"""Statistics used by telemetry, evaluation and reports. Pure NumPy, no model.

Before/after comparisons are PAIRED: both models answer the same questions,
so the right test looks at the questions whose outcome changed (McNemar),
not at two independent proportions.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al., 2021): 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def standard_error_pp(n: int) -> float:
    """Conservative (p = 0.5) standard error of ONE proportion, in percentage points.
    A rough precision guide for a single number; not a test of a difference."""
    return 100 * math.sqrt(0.25 / n)


def window_mean(series: Sequence[Optional[float]], first: bool = True, w: int = 15) -> float:
    """Mean of the first or last ``w`` values (None ignored). Window means, not
    single endpoints: each step averages only G samples."""
    s = [float(x) for x in series if x is not None]
    part = s[:w] if first else s[-w:]
    return float(np.mean(part)) if part else float("nan")


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. ``b`` = pairs that went right -> wrong,
    ``c`` = wrong -> right. Unchanged pairs carry no information."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_binary(before: Sequence[bool], after: Sequence[bool]) -> Dict[str, float]:
    """Paired comparison of a yes/no outcome on the same questions."""
    if len(before) != len(after):
        raise ValueError("paired outcomes must have equal length")
    b = sum(1 for x, y in zip(before, after, strict=True) if x and not y)
    c = sum(1 for x, y in zip(before, after, strict=True) if y and not x)
    n = len(before)
    return dict(n=n, before=sum(before) / n if n else float("nan"),
                after=sum(after) / n if n else float("nan"),
                worse=b, better=c, p_value=mcnemar_exact(b, c))


def paired_bootstrap_ci(before: Sequence[float], after: Sequence[float], n_boot: int = 10_000,
                        seed: int = 0, level: float = 0.95) -> Dict[str, float]:
    """Mean paired difference (after - before) with a percentile bootstrap CI.
    For continuous per-question metrics such as pass@1 and answer length."""
    d = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    if d.size == 0:
        return dict(n=0, mean_diff=float("nan"), ci_low=float("nan"), ci_high=float("nan"))
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return dict(n=int(d.size), mean_diff=float(d.mean()), ci_low=float(lo), ci_high=float(hi))
