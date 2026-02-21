"""ASM reconstructor: ICP-based adaptive spot matching.

Uses multi-start Iterative Closest Point (ICP) to find Zernike coefficients
that best explain the observed spot pattern:

1. Precompute Zernike slope basis matrix.
2. Run baseline reconstruction as a warm start.
3. For each random start (+ baseline + zero):
   a. Compute expected spots from current coefficients.
   b. Clip to sensor bounds.
   c. Reciprocal nearest-neighbor matching (mutual matches only).
   d. Solve regularized least-squares for coefficients from matched slopes.
   e. Repeat until convergence.
4. Return the solution with lowest residual.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional

from scipy.spatial import cKDTree

from src.sim.lenslet import LensletArray
from src.recon.zernike import num_zernike_terms
from src.recon.least_squares import build_zernike_slope_matrix


def _trimmed_mean(values: np.ndarray, trim_ratio: float) -> float:
    """Mean of the smallest ceil(trim_ratio * N) values."""
    if values.size == 0:
        return float("inf")
    n_keep = max(1, int(np.ceil(trim_ratio * values.size)))
    part = np.partition(values, n_keep - 1)[:n_keep]
    return float(np.mean(part))


def _build_expected_spots(
    coeffs: np.ndarray,
    ref: np.ndarray,
    G: np.ndarray,
    focal_um: float,
    n_sub: int,
) -> np.ndarray:
    """Compute expected spot positions from Zernike coefficients.

    Args:
        coeffs: (n_terms,) Zernike coefficient vector.
        ref: (N_sub, 2) reference spot positions.
        G: (2*N_sub, n_terms) slope basis matrix.
        focal_um: Focal length in um.
        n_sub: Number of subapertures.

    Returns:
        (N_sub, 2) expected spot positions in um.
    """
    slopes_vec = G @ coeffs  # (2*N_sub,)
    slopes = np.column_stack([slopes_vec[:n_sub], slopes_vec[n_sub:]])  # (N_sub, 2)
    return ref + focal_um * slopes


def _icp_single(
    c0: np.ndarray,
    observed: np.ndarray,
    ref: np.ndarray,
    G: np.ndarray,
    focal_um: float,
    n_sub: int,
    sensor_w: float,
    sensor_h: float,
    lambda_reg: float,
    n_icp_iter: int,
    convergence_tol: float,
    max_match_dist_um: float,
    min_match_ratio: float,
    trim_ratio: float,
    allow_forward_fallback: bool,
) -> Dict[str, Any]:
    """Run one ICP chain from initial coefficients c0.

    Uses reciprocal (mutual) nearest-neighbor matching to avoid
    incorrect assignments that would poison the least-squares fit.

    Returns dict with coeffs, residual, n_matched, converged.
    """
    n_terms = len(c0)
    c = c0.copy()
    obs_tree = cKDTree(observed)

    prev_residual = np.inf
    residual_raw = np.inf
    residual_trimmed = np.inf
    n_matched = 0
    best_coeffs = c.copy()
    best_residual_raw = np.inf
    best_residual_trimmed = np.inf
    best_n_matched = 0

    for it in range(n_icp_iter):
        # 1. Compute expected spots
        E = _build_expected_spots(c, ref, G, focal_um, n_sub)

        # 2. Clip to sensor bounds
        in_bounds = (
            (E[:, 0] >= 0)
            & (E[:, 0] <= sensor_w)
            & (E[:, 1] >= 0)
            & (E[:, 1] <= sensor_h)
        )
        ib_idx = np.where(in_bounds)[0]
        n_in = len(ib_idx)

        if n_in < n_terms:
            break

        # Reject degenerate solutions with too few in-bounds spots
        min_required = max(n_terms * 3, int(len(observed) * min_match_ratio))
        if n_in < min_required:
            break

        # 3. Reciprocal nearest-neighbor matching
        E_in = E[ib_idx]  # (n_in, 2)

        # Forward: each expected spot -> nearest observed
        dists_fwd, nn_fwd = obs_tree.query(E_in)

        # Backward: each observed -> nearest expected (among in-bounds)
        exp_tree = cKDTree(E_in)
        _, nn_bwd = exp_tree.query(observed)

        # Reciprocal matches: E_in[i] -> obs[j] AND obs[j] -> E_in[i]
        reciprocal_mask = np.zeros(n_in, dtype=bool)
        within_dist = dists_fwd <= max_match_dist_um
        for i in range(n_in):
            j = nn_fwd[i]
            if nn_bwd[j] == i and within_dist[i]:
                reciprocal_mask[i] = True

        matched_idx = np.where(reciprocal_mask)[0]
        n_reciprocal = len(matched_idx)

        if n_reciprocal < n_terms and allow_forward_fallback:
            forward_mask = np.where(within_dist)[0]
            matched_idx = forward_mask
            n_matched = len(matched_idx)
        else:
            n_matched = n_reciprocal

        if n_matched < n_terms:
            break

        # Map back to subaperture indices
        sub_idx = ib_idx[matched_idx]
        obs_idx = nn_fwd[matched_idx]

        # 4. Compute target slopes from matched displacements
        matched_obs = observed[obs_idx]  # (n_matched, 2)
        target_disp = matched_obs - ref[sub_idx]  # (n_matched, 2)
        target_slopes = target_disp / focal_um  # (n_matched, 2)

        # 5. Build sub-system for matched subapertures
        G_sub_x = G[sub_idx, :]  # (n_matched, n_terms)
        G_sub_y = G[n_sub + sub_idx, :]  # (n_matched, n_terms)
        A = np.vstack([G_sub_x, G_sub_y])  # (2*n_matched, n_terms)
        b = np.concatenate([target_slopes[:, 0], target_slopes[:, 1]])

        # 6. Regularized least-squares: (A^T A + lambda * I) c = A^T b
        if lambda_reg > 0:
            ATA = A.T @ A + lambda_reg * np.eye(n_terms)
            ATb = A.T @ b
            try:
                c_new = np.linalg.solve(ATA, ATb)
            except np.linalg.LinAlgError:
                c_new, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        else:
            c_new, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # 7. Compute residual with robust trimmed mean
        matched_dists = dists_fwd[matched_idx]
        residual_raw = float(np.mean(matched_dists))
        residual_trimmed = _trimmed_mean(matched_dists, trim_ratio)
        if residual_trimmed < best_residual_trimmed:
            best_residual_raw = residual_raw
            best_residual_trimmed = residual_trimmed
            best_n_matched = n_matched
            best_coeffs = c_new.copy()

        # 8. Check convergence
        coeff_change = np.max(np.abs(c_new - c))
        c = c_new

        if (
            abs(prev_residual - residual_trimmed) < convergence_tol
            and coeff_change < convergence_tol
        ):
            return {
                "coeffs": best_coeffs,
                "residual_raw": best_residual_raw,
                "residual_trimmed": best_residual_trimmed,
                "n_matched": best_n_matched,
                "converged": True,
            }
        prev_residual = residual_trimmed

    if np.isfinite(best_residual_trimmed):
        return {
            "coeffs": best_coeffs,
            "residual_raw": best_residual_raw,
            "residual_trimmed": best_residual_trimmed,
            "n_matched": best_n_matched,
            "converged": False,
        }

    return {
        "coeffs": c,
        "residual_raw": np.inf,
        "residual_trimmed": np.inf,
        "n_matched": n_matched,
        "converged": False,
    }


def asm_reconstruct(
    observed: np.ndarray,
    lenslet: LensletArray,
    cfg: Dict[str, Any],
    seed: int = 42,
    observed_sub_idx: Optional[np.ndarray] = None,
    pv_hint: Optional[float] = None,
    missing_ratio_hint: Optional[float] = None,
) -> Dict[str, Any]:
    """Run ASM reconstruction via multi-start ICP.

    Uses baseline reconstruction as a warm start, plus random starts.

    Args:
        observed: (M, 2) observed spot positions in um.
        lenslet: LensletArray instance.
        cfg: Configuration dict with asm/zernike sections.
        seed: Random seed for random starts.

    Returns:
        Dict with keys: coeffs, success, objective_value, n_iterations.
    """
    zer_cfg = cfg.get("zernike", {})
    asm_cfg = cfg.get("asm", {})

    max_order = zer_cfg.get("order", 3)
    coeff_bound = zer_cfg.get("coeff_bound", 1.0)
    n_terms = num_zernike_terms(max_order)

    n_starts = asm_cfg.get("n_starts", 50)
    n_icp_iter = asm_cfg.get("n_icp_iter", 30)
    lambda_reg = asm_cfg.get("lambda_reg", 1e-3)
    convergence_tol = asm_cfg.get("convergence_tol", 1e-6)
    grid_size = asm_cfg.get("grid_size", 128)
    max_match_dist_factor = asm_cfg.get("max_match_dist_factor", 0.35)
    min_match_ratio = asm_cfg.get("min_match_ratio", 0.2)
    trim_ratio = asm_cfg.get("trim_ratio", 0.9)
    allow_forward_fallback = asm_cfg.get("allow_forward_fallback", False)
    max_match_dist_um = max_match_dist_factor * lenslet.pitch_um

    # Precompute slope basis matrix (expensive, done once)
    G = build_zernike_slope_matrix(lenslet, max_order, grid_size)

    ref = lenslet.reference_positions()
    n_sub = lenslet.n_subapertures
    focal_um = lenslet.focal_um
    sensor_w = lenslet.sensor_width_um
    sensor_h = lenslet.sensor_height_um

    # Optional oracle index hint (simulation debug mode):
    # directly use known subaperture indices for observed spots.
    if observed_sub_idx is not None and len(observed_sub_idx) >= n_terms:
        sub_idx = observed_sub_idx.astype(int)
        target_disp = observed - ref[sub_idx]
        target_slopes = target_disp / focal_um
        G_sub_x = G[sub_idx, :]
        G_sub_y = G[n_sub + sub_idx, :]
        A = np.vstack([G_sub_x, G_sub_y])
        b = np.concatenate([target_slopes[:, 0], target_slopes[:, 1]])
        if lambda_reg > 0:
            ATA = A.T @ A + lambda_reg * np.eye(n_terms)
            ATb = A.T @ b
            try:
                coeffs = np.linalg.solve(ATA, ATb)
            except np.linalg.LinAlgError:
                coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        else:
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        E = _build_expected_spots(coeffs, ref, G, focal_um, n_sub)
        matched = E[sub_idx]
        d = np.linalg.norm(matched - observed, axis=1)
        residual_raw = float(np.mean(d)) if len(d) else np.inf
        residual_trimmed = _trimmed_mean(d, trim_ratio)
        success_threshold = lenslet.pitch_um * 0.5
        success = residual_trimmed < success_threshold
        return {
            "coeffs": coeffs,
            "success": success,
            "objective_value": residual_trimmed,
            "residual_raw": residual_raw,
            "residual_trimmed": residual_trimmed,
            "n_matched": int(len(sub_idx)),
            "n_iterations": 1,
            "solver": "asm_oracle_ls",
        }

    if len(observed) < n_terms:
        return {
            "coeffs": np.zeros(n_terms),
            "success": False,
            "objective_value": np.inf,
            "residual_raw": np.inf,
            "residual_trimmed": np.inf,
            "n_matched": 0,
            "n_iterations": 0,
            "solver": "asm_icp_cpu",
        }

    # Common ICP arguments
    icp_kwargs = dict(
        observed=observed,
        ref=ref,
        G=G,
        focal_um=focal_um,
        n_sub=n_sub,
        sensor_w=sensor_w,
        sensor_h=sensor_h,
        lambda_reg=lambda_reg,
        n_icp_iter=n_icp_iter,
        convergence_tol=convergence_tol,
        max_match_dist_um=max_match_dist_um,
        min_match_ratio=min_match_ratio,
        trim_ratio=trim_ratio,
        allow_forward_fallback=allow_forward_fallback,
    )

    best_result = None
    best_residual = np.inf

    def _try_start(c0):
        nonlocal best_result, best_residual
        result = _icp_single(c0=c0, **icp_kwargs)
        if result["residual_trimmed"] < best_residual:
            best_residual = result["residual_trimmed"]
            best_result = result

    # Start 0: Sorting Match warm start (global matching, no initial guess)
    enable_sorting = asm_cfg.get("enable_sorting", True)
    sorting_coeffs = None
    if enable_sorting:
        from src.recon.sorting_matcher import sorting_match

        sorting_result = sorting_match(observed, lenslet, cfg)
        if sorting_result["success"]:
            sorting_coeffs = sorting_result["coeffs"]
            _try_start(sorting_coeffs)

    # Start 1: baseline warm start (fallback structured matching)
    from src.recon.baseline_extrap_nn import baseline_reconstruct

    bl = baseline_reconstruct(observed, lenslet, cfg)
    if bl["success"]:
        _try_start(bl["coeffs"])

    # Start 2: zero coefficients
    _try_start(np.zeros(n_terms))

    # Remaining starts: random (fewer if sorting succeeded)
    rng = np.random.RandomState(seed)
    search_bound = max(coeff_bound * 2, 2.0)

    n_random = max(n_starts - 3, 0)
    if sorting_coeffs is not None:
        n_random = max(n_random // 5, 2)

    for i in range(n_random):
        c0 = rng.uniform(-search_bound, search_bound, size=n_terms)
        _try_start(c0)

    # Determine success: residual should be small (in um, sub-pitch)
    success_threshold = lenslet.pitch_um * 0.5  # 75 um for 150um pitch
    success = best_residual < success_threshold

    solver_tag = "asm_sorting_icp_cpu" if sorting_coeffs is not None else "asm_icp_cpu"

    return {
        "coeffs": best_result["coeffs"] if best_result else np.zeros(n_terms),
        "success": success,
        "objective_value": best_residual,
        "residual_raw": best_result["residual_raw"] if best_result else np.inf,
        "residual_trimmed": best_result["residual_trimmed"] if best_result else np.inf,
        "n_matched": best_result["n_matched"] if best_result else 0,
        "n_iterations": n_icp_iter,
        "solver": solver_tag,
    }
