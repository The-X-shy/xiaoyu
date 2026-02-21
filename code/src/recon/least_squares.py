"""Least-squares wavefront reconstruction from slopes."""

from __future__ import annotations
import numpy as np
from typing import Optional

from src.recon.zernike import zernike_wavefront, num_zernike_terms
from src.sim.lenslet import LensletArray


def build_zernike_slope_matrix(
    la: LensletArray,
    max_order: int,
    grid_size: int = 128,
) -> np.ndarray:
    """Build the Zernike-to-slope influence matrix.

    For each Zernike mode j, compute slopes at all subaperture centers.
    Returns matrix G of shape (2*N, n_terms) where N = n_subapertures.
    Rows [0:N] are x-slopes, rows [N:2N] are y-slopes.
    """
    n_terms = num_zernike_terms(max_order)
    n_sub = la.n_subapertures

    G = np.zeros((2 * n_sub, n_terms), dtype=float)
    for j_idx in range(n_terms):
        coeffs = np.zeros(n_terms)
        coeffs[j_idx] = 1.0

        W = zernike_wavefront(coeffs, grid_size)
        slopes = la.compute_slopes(W, grid_size)

        G[:n_sub, j_idx] = slopes[:, 0]
        G[n_sub:, j_idx] = slopes[:, 1]

    return G


def reconstruct_least_squares(
    la: LensletArray,
    slopes: np.ndarray,
    max_order: int,
    grid_size: int = 128,
) -> np.ndarray:
    """Reconstruct Zernike coefficients from measured slopes via least squares.

    Args:
        la: Lenslet array instance.
        slopes: (N, 2) array of measured slopes.
        max_order: Maximum Zernike radial order.
        grid_size: Grid size for matrix construction.

    Returns:
        1D array of reconstructed Zernike coefficients.
    """
    G = build_zernike_slope_matrix(la, max_order, grid_size)
    n_sub = la.n_subapertures

    b = np.concatenate([slopes[:, 0], slopes[:, 1]])

    c, _, _, _ = np.linalg.lstsq(G, b, rcond=None)
    return c
