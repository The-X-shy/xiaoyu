"""ASM objective function for PSO optimization.

J(c) = d_H(E(c), G) + lambda_dup * P_dup(c) + lambda_out * P_out(c) + lambda_reg * ||c||^2
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any

from scipy.spatial import cKDTree
from src.sim.lenslet import LensletArray
from src.sim.wavefront import generate_wavefront
from src.recon.zernike import num_zernike_terms


_la_cache: Dict[str, LensletArray] = {}


def hausdorff_mean(A: np.ndarray, B: np.ndarray) -> float:
    """Compute mean bidirectional Hausdorff distance.
    d_H = 0.5 * (mean_a min_b ||a-b|| + mean_b min_a ||b-a||)
    """
    if len(A) == 0 or len(B) == 0:
        return 1e6
    tree_B = cKDTree(B)
    tree_A = cKDTree(A)
    d_AB, _ = tree_B.query(A)
    d_BA, _ = tree_A.query(B)
    return 0.5 * (np.mean(d_AB) + np.mean(d_BA))


def duplicate_penalty(E: np.ndarray, G: np.ndarray) -> float:
    """Penalize many-to-one matching."""
    if len(E) == 0 or len(G) == 0:
        return 0.0
    tree_G = cKDTree(G)
    _, nn_idx = tree_G.query(E)
    counts = np.bincount(nn_idx, minlength=len(G))
    duplicates = np.sum(np.maximum(counts - 1, 0))
    return float(duplicates) / max(len(E), 1)


def out_of_bounds_penalty(
    E: np.ndarray, sensor_width_um: float, sensor_height_um: float
) -> float:
    """Penalize spots outside sensor bounds."""
    if len(E) == 0:
        return 0.0
    oob_x = np.maximum(-E[:, 0], 0) + np.maximum(E[:, 0] - sensor_width_um, 0)
    oob_y = np.maximum(-E[:, 1], 0) + np.maximum(E[:, 1] - sensor_height_um, 0)
    return float(np.mean(oob_x + oob_y))


def asm_objective(
    coeffs: np.ndarray, observed: np.ndarray, lenslet: LensletArray, cfg: Dict[str, Any]
) -> float:
    """Compute ASM objective function value."""
    asm_cfg = cfg.get("asm", {})
    lambda_dup = asm_cfg.get("lambda_dup", 1.0)
    lambda_out = asm_cfg.get("lambda_out", 1.0)
    lambda_reg = asm_cfg.get("lambda_reg", 1e-3)

    grid_size = 128

    W = generate_wavefront(coeffs, grid_size)
    slopes = lenslet.compute_slopes(W, grid_size)
    E = lenslet.displaced_positions(slopes)

    # Clip expected spots to sensor bounds
    in_bounds = lenslet.check_bounds(E)
    E_in = E[in_bounds]

    if len(E_in) == 0:
        # No expected spots on sensor → maximum penalty
        return 1e6

    d_h = hausdorff_mean(E_in, observed)
    p_dup = duplicate_penalty(E_in, observed)

    # Coverage penalty: penalize having too few in-bounds spots
    n_total = len(E)
    n_in = len(E_in)
    coverage = (n_total - n_in) / max(n_total, 1) * 100.0

    reg = np.sum(coeffs**2)

    return d_h + lambda_dup * p_dup + lambda_out * coverage + lambda_reg * reg
