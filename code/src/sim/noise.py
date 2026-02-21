"""Noise injection for SHWS simulation."""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


def apply_noise(
    positions: np.ndarray,
    read_sigma: float,
    background: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Apply read noise and background offset to spot positions."""
    result = positions.copy()
    if read_sigma > 0:
        rng = np.random.RandomState(seed)
        result = result + rng.normal(0, read_sigma, size=positions.shape)
    result = result + background
    return result


def apply_missing_spots(
    positions: np.ndarray,
    ratio: float,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly remove spots to simulate missing detections.

    Returns (remaining_positions, keep_mask).
    """
    n = len(positions)
    n_keep = max(0, int(round(n * (1 - ratio))))

    if n_keep == 0:
        return np.empty((0, 2), dtype=float), np.zeros(n, dtype=bool)
    if n_keep >= n:
        return positions.copy(), np.ones(n, dtype=bool)

    rng = np.random.RandomState(seed)
    keep_indices = rng.choice(n, size=n_keep, replace=False)
    keep_indices.sort()

    mask = np.zeros(n, dtype=bool)
    mask[keep_indices] = True
    return positions[mask].copy(), mask
