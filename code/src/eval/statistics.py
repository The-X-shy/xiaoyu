"""Statistical utilities for experiment evaluation."""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Tuple


def confidence_interval_95(values: np.ndarray) -> Tuple[float, float]:
    """Compute 95% confidence interval for the mean."""
    n = len(values)
    mean = np.mean(values)
    if n <= 1:
        return float(mean), float(mean)
    std = np.std(values, ddof=1)
    from scipy import stats

    t_val = stats.t.ppf(0.975, df=n - 1)
    margin = t_val * std / np.sqrt(n)
    return float(mean - margin), float(mean + margin)


def compute_summary(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a list of values."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ci_low, ci_high = confidence_interval_95(arr)
    return {
        "mean": mean,
        "std": std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": n,
    }
