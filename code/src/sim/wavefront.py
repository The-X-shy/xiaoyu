"""Wavefront generation module for SHWS simulation."""

from __future__ import annotations
import numpy as np
from src.recon.zernike import zernike_wavefront, num_zernike_terms
from typing import Optional


def random_zernike_coeffs(
    max_order: int,
    coeff_bound: float,
    seed: int,
    exclude_piston: bool = True,
) -> np.ndarray:
    """Generate random Zernike coefficients within [-bound, bound].

    Piston (j=1, index 0) is set to zero by default.

    Args:
        max_order: Maximum radial order.
        coeff_bound: Coefficients are drawn from [-coeff_bound, coeff_bound].
        seed: Random seed for reproducibility.
        exclude_piston: If True, set piston coefficient to zero.

    Returns:
        1D array of Zernike coefficients.
    """
    rng = np.random.RandomState(seed)
    n_terms = num_zernike_terms(max_order)
    coeffs = rng.uniform(-coeff_bound, coeff_bound, size=n_terms)
    if exclude_piston:
        coeffs[0] = 0.0
    return coeffs


def scale_coeffs_to_pv(
    coeffs: np.ndarray,
    target_pv: float,
    grid_size: int = 256,
    max_order: Optional[int] = None,
) -> np.ndarray:
    """Scale coefficients so wavefront peak-to-valley equals target_pv.

    Args:
        coeffs: Input Zernike coefficients.
        target_pv: Target peak-to-valley value (in waves).
        grid_size: Grid size for PV evaluation.
        max_order: Max radial order (unused, kept for API compat).

    Returns:
        Scaled copy of coefficients.
    """
    W = zernike_wavefront(coeffs, grid_size, max_order)
    x = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, x)
    mask = np.sqrt(X**2 + Y**2) <= 1.0
    W_valid = W[mask]
    if len(W_valid) == 0:
        return coeffs.copy()
    current_pv = np.max(W_valid) - np.min(W_valid)
    if current_pv < 1e-15:
        return coeffs.copy()
    return coeffs * (target_pv / current_pv)


def generate_wavefront(
    coeffs: np.ndarray,
    grid_size: int,
    max_order: Optional[int] = None,
) -> np.ndarray:
    """Generate wavefront from Zernike coefficients.

    Thin wrapper around zernike_wavefront.

    Args:
        coeffs: Zernike coefficients (Noll ordering).
        grid_size: Output grid size.
        max_order: Max radial order (unused, kept for API compat).

    Returns:
        2D wavefront array.
    """
    return zernike_wavefront(coeffs, grid_size, max_order)
