"""Baseline reconstructor: extrapolation + nearest-neighbor matching.

Algorithm:
1. Start from center subapertures (closest to sensor center).
2. Match center spots to nearest observed spots within tau_nn.
3. For unmatched subapertures, extrapolate expected position from
   matched neighbors, then match via nearest-neighbor.
4. Iterate outward until all subapertures are processed or k_fail
   consecutive failures occur.
5. Reconstruct Zernike coefficients from matched displacements
   via least-squares.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple
from scipy.spatial import cKDTree

from src.sim.lenslet import LensletArray
from src.recon.least_squares import build_zernike_slope_matrix
from src.recon.zernike import num_zernike_terms


_G_CACHE: Dict[Tuple[float, ...], np.ndarray] = {}


def _matrix_cache_key(la: LensletArray, max_order: int, grid_size: int) -> Tuple[float, ...]:
    """Cache key for Zernike slope matrix."""
    return (
        float(la.pitch_um),
        float(la.focal_mm),
        float(la.fill_factor),
        float(la.sensor_width_px),
        float(la.sensor_height_px),
        float(la.pixel_um),
        float(max_order),
        float(grid_size),
    )


def _get_cached_zernike_matrix(
    la: LensletArray, max_order: int, grid_size: int = 128
) -> np.ndarray:
    """Get or build cached slope influence matrix G."""
    key = _matrix_cache_key(la, max_order, grid_size)
    G = _G_CACHE.get(key)
    if G is None:
        G = build_zernike_slope_matrix(la, max_order, grid_size)
        _G_CACHE[key] = G
    return G


def baseline_reconstruct(
    observed: np.ndarray,
    lenslet: LensletArray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Run baseline extrapolation + nearest-neighbor reconstruction.

    Args:
        observed: (M, 2) observed spot positions in um.
        lenslet: LensletArray instance.
        cfg: Configuration dict with baseline/zernike sections.

    Returns:
        Dict with keys: coeffs, success, rmse, n_matched.
    """
    bl_cfg = cfg.get("baseline", {})
    tau_nn_px = bl_cfg.get("tau_nn_px", 5.0)
    k_fail = bl_cfg.get("k_fail", 3)
    rmse_max = bl_cfg.get("rmse_max", 0.15)

    zer_cfg = cfg.get("zernike", {})
    max_order = zer_cfg.get("order", 3)
    n_terms = num_zernike_terms(max_order)

    tau_nn_um = tau_nn_px * lenslet.pixel_um

    ref = lenslet.reference_positions()
    n_sub = len(ref)

    if len(observed) == 0 or n_sub == 0:
        return {
            "coeffs": np.zeros(n_terms),
            "success": False,
            "rmse": float("inf"),
            "n_matched": 0,
            "conflict_ratio": 1.0,
            "solver": "baseline_extrap_nn_cpu",
        }

    # Build KD-tree of observed spots
    obs_tree = cKDTree(observed)

    # Sort subapertures by distance from sensor center (inside-out)
    order = _compute_inside_out_order(ref, lenslet)

    # Matching arrays
    matched = np.full(n_sub, False)
    match_obs_idx = np.full(n_sub, -1, dtype=int)
    matched_disp = np.zeros((n_sub, 2), dtype=float)

    _seed_center_matches(
        observed=observed,
        ref=ref,
        order=order,
        obs_tree=obs_tree,
        matched=matched,
        match_obs_idx=match_obs_idx,
        matched_disp=matched_disp,
    )

    # Phase 2: Propagation — extrapolate from matched neighbors, apply tau_nn
    _propagate_matches(
        observed=observed,
        ref=ref,
        order=order,
        obs_tree=obs_tree,
        tau_nn_um=tau_nn_um,
        k_fail=k_fail,
        matched=matched,
        match_obs_idx=match_obs_idx,
        matched_disp=matched_disp,
    )

    n_matched = int(np.sum(matched))

    if n_matched < 4:
        return {
            "coeffs": np.zeros(n_terms),
            "success": False,
            "rmse": float("inf"),
            "n_matched": n_matched,
            "conflict_ratio": 1.0,
            "solver": "baseline_extrap_nn_cpu",
        }

    # Convert displacements to slopes for matched subapertures
    matched_mask = matched
    disp_matched = matched_disp[matched_mask]
    slopes_matched = disp_matched / lenslet.focal_um

    # Reconstruct from the matched subset
    coeffs_recon = _reconstruct_from_subset(
        lenslet, matched_mask, slopes_matched, max_order
    )

    conflict_ratio = _compute_conflict_ratio(match_obs_idx, matched_mask)

    conflict_max = bl_cfg.get("conflict_ratio_max", 0.2)

    success = conflict_ratio <= conflict_max
    rmse = float(np.sqrt(np.mean(coeffs_recon[1:] ** 2)))  # rough estimate

    return {
        "coeffs": coeffs_recon,
        "success": success,
        "rmse": rmse,
        "n_matched": n_matched,
        "conflict_ratio": conflict_ratio,
        "solver": "baseline_extrap_nn_cpu",
    }


def _compute_inside_out_order(ref: np.ndarray, lenslet: LensletArray) -> np.ndarray:
    """Sort subapertures by distance from sensor center (inside-out)."""
    cx = lenslet.sensor_width_um / 2
    cy = lenslet.sensor_height_um / 2
    center = np.array([cx, cy])
    dist_from_center = np.linalg.norm(ref - center, axis=1)
    return np.argsort(dist_from_center)


def _seed_center_matches(
    observed: np.ndarray,
    ref: np.ndarray,
    order: np.ndarray,
    obs_tree: cKDTree,
    matched: np.ndarray,
    match_obs_idx: np.ndarray,
    matched_disp: np.ndarray,
) -> None:
    """Unconditionally seed center-most subapertures with nearest observed spots."""
    n_seed_target = min(4, len(ref))
    for sub_idx in order[:n_seed_target]:
        ref_pos = ref[sub_idx]
        _, idx = obs_tree.query(ref_pos, k=1)
        matched[sub_idx] = True
        match_obs_idx[sub_idx] = idx
        matched_disp[sub_idx] = observed[idx] - ref_pos


def _propagate_matches(
    observed: np.ndarray,
    ref: np.ndarray,
    order: np.ndarray,
    obs_tree: cKDTree,
    tau_nn_um: float,
    k_fail: int,
    matched: np.ndarray,
    match_obs_idx: np.ndarray,
    matched_disp: np.ndarray,
) -> None:
    """Extrapolate and propagate matching from center to edge."""
    consecutive_fails = 0
    for sub_idx in order:
        if matched[sub_idx]:
            continue

        expected = _extrapolate(sub_idx, ref, matched, matched_disp)
        dist, idx = obs_tree.query(expected, k=1)

        if dist <= tau_nn_um:
            matched[sub_idx] = True
            match_obs_idx[sub_idx] = idx
            matched_disp[sub_idx] = observed[idx] - ref[sub_idx]
            consecutive_fails = 0
        else:
            consecutive_fails += 1
            if consecutive_fails >= k_fail:
                break


def _compute_conflict_ratio(match_obs_idx: np.ndarray, matched_mask: np.ndarray) -> float:
    """Compute one-to-many conflict ratio in matched assignments."""
    n_matched = int(np.sum(matched_mask))
    if n_matched <= 0:
        return 1.0
    used_obs = match_obs_idx[matched_mask]
    n_unique = len(np.unique(used_obs))
    return 1.0 - n_unique / n_matched


def _extrapolate(
    sub_idx: int,
    ref: np.ndarray,
    matched: np.ndarray,
    matched_disp: np.ndarray,
) -> np.ndarray:
    """Extrapolate expected position from matched neighbors.

    If no neighbors are matched, return the reference position.
    """
    ref_pos = ref[sub_idx]

    if not np.any(matched):
        return ref_pos.copy()

    # Find closest matched subapertures
    matched_indices = np.where(matched)[0]
    dists = np.linalg.norm(ref[matched_indices] - ref_pos, axis=1)
    k = min(4, len(matched_indices))
    nearest = np.argsort(dists)[:k]

    # Weighted average displacement of neighbors
    neighbor_idx = matched_indices[nearest]
    neighbor_dists = dists[nearest]

    # Inverse distance weighting
    weights = np.where(neighbor_dists > 1e-10, 1.0 / neighbor_dists, 1e10)
    weights /= np.sum(weights)

    avg_disp = np.sum(matched_disp[neighbor_idx] * weights[:, np.newaxis], axis=0)

    return ref_pos + avg_disp


def _reconstruct_from_subset(
    lenslet: LensletArray,
    matched_mask: np.ndarray,
    slopes_matched: np.ndarray,
    max_order: int,
    grid_size: int = 128,
) -> np.ndarray:
    """Reconstruct Zernike coefficients from a subset of subapertures (CPU LS)."""
    A, b, _ = _build_subset_linear_system(
        lenslet=lenslet,
        matched_mask=matched_mask,
        slopes_matched=slopes_matched,
        max_order=max_order,
        grid_size=grid_size,
    )
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return c


def _build_subset_linear_system(
    lenslet: LensletArray,
    matched_mask: np.ndarray,
    slopes_matched: np.ndarray,
    max_order: int,
    grid_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Build A,b for subset least-squares reconstruction."""
    n_terms = num_zernike_terms(max_order)
    n_sub = lenslet.n_subapertures
    sub_idx = np.where(matched_mask)[0]
    G = _get_cached_zernike_matrix(lenslet, max_order, grid_size)
    A = np.vstack([G[sub_idx, :], G[n_sub + sub_idx, :]])
    b = np.concatenate([slopes_matched[:, 0], slopes_matched[:, 1]])
    return A, b, n_terms
