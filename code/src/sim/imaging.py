"""Imaging simulation: spot projection and centroid extraction.

Simplified model: spots are point positions (no PSF convolution).
Centroid noise models localization uncertainty.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


def simulate_spots(
    ref_positions: np.ndarray,
    displacements: np.ndarray,
) -> np.ndarray:
    """Compute displaced spot positions. Returns ref + disp."""
    return ref_positions + displacements


def extract_centroids(
    spot_positions: np.ndarray,
    noise_px: float,
    pixel_um: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Extract centroids with localization noise."""
    if noise_px == 0.0:
        return spot_positions.copy()
    rng = np.random.RandomState(seed)
    noise_um = noise_px * pixel_um
    noise = rng.normal(0, noise_um, size=spot_positions.shape)
    return spot_positions + noise
