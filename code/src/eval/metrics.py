"""Evaluation metrics for wavefront reconstruction."""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional


def rmse_coeffs(
    c_true: np.ndarray, c_recon: np.ndarray, exclude_piston: bool = False
) -> float:
    """Compute RMSE between true and reconstructed Zernike coefficients."""
    if exclude_piston:
        c_true = c_true[1:]
        c_recon = c_recon[1:]
    return float(np.sqrt(np.mean((c_true - c_recon) ** 2)))


def rmse_wavefront(W_true: np.ndarray, W_recon: np.ndarray, grid_size: int) -> float:
    """Compute RMSE between true and reconstructed wavefronts on unit disk."""
    x = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, x)
    mask = np.sqrt(X**2 + Y**2) <= 1.0
    diff = W_true[mask] - W_recon[mask]
    if len(diff) == 0:
        return 0.0
    return float(np.sqrt(np.mean(diff**2)))


def success_rate(results: List[Dict[str, Any]]) -> float:
    """Compute success rate from a list of result dicts."""
    if not results:
        return 0.0
    n_success = sum(1 for r in results if r.get("success", False))
    return n_success / len(results)


def compute_dynamic_range(
    records: List[Dict[str, Any]], rmse_max: float = 0.15, sr_min: float = 0.95
) -> float:
    """Find maximum PV where success_rate >= sr_min AND mean_rmse <= rmse_max."""
    if not records:
        return 0.0
    pv_groups: Dict[float, List[Dict]] = {}
    for r in records:
        pv = r["pv_level"]
        if pv not in pv_groups:
            pv_groups[pv] = []
        pv_groups[pv].append(r)
    max_dr = 0.0
    for pv in sorted(pv_groups.keys()):
        group = pv_groups[pv]
        sr = success_rate(group)
        successful = [r for r in group if r.get("success", False)]
        mean_rmse = (
            np.mean([r["rmse"] for r in successful]) if successful else float("inf")
        )
        if sr >= sr_min and mean_rmse <= rmse_max:
            max_dr = pv
        else:
            break
    return max_dr
