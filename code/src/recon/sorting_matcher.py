"""Center-out spot matching for SHWS dynamic range extension.

Replaces broken sorting-based matching with a physics-aware approach:

1. Match spots near the optical center first (where displacements are smallest).
2. Solve for initial Zernike coefficients from these center matches.
3. Predict expected positions for ALL subapertures.
4. Extend matching outward using reciprocal NN between predicted and observed.
5. Iterate until convergence.

This works for any sensor size because it doesn't rely on row structure
(which is destroyed at high PV for large apertures).

Physical basis: For Zernike modes 1-10, displacement at the optical center
is zero for all modes except tilt (Z2, Z3). Even tilt only shifts everything
uniformly, so NN matching near the center always works for the non-tilt
component. We estimate tilt separately from the global center-of-mass shift.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, Tuple

from src.sim.lenslet import LensletArray
from src.recon.zernike import num_zernike_terms
from src.recon.baseline_extrap_nn import _get_cached_zernike_matrix


# ---------------------------------------------------------------------------
# Helper: least-squares solve
# ---------------------------------------------------------------------------


def _solve_coeffs(
    sub_idx: np.ndarray,
    obs_idx: np.ndarray,
    observed: np.ndarray,
    ref: np.ndarray,
    G: np.ndarray,
    n_sub: int,
    n_terms: int,
    focal_um: float,
    lambda_reg: float,
) -> np.ndarray:
    """Solve least-squares for Zernike coefficients from matched pairs."""
    displacements = observed[obs_idx] - ref[sub_idx]
    slopes = displacements / focal_um

    A_x = G[sub_idx, :]
    A_y = G[n_sub + sub_idx, :]
    A = np.vstack([A_x, A_y])
    b = np.concatenate([slopes[:, 0], slopes[:, 1]])

    if lambda_reg > 0:
        ATA = A.T @ A + lambda_reg * np.eye(n_terms)
        ATb = A.T @ b
        try:
            return np.linalg.solve(ATA, ATb)
        except np.linalg.LinAlgError:
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return coeffs
    else:
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return coeffs


# ---------------------------------------------------------------------------
# Helper: expected positions from coefficients
# ---------------------------------------------------------------------------


def _compute_expected_positions(
    coeffs: np.ndarray,
    ref: np.ndarray,
    G: np.ndarray,
    n_sub: int,
    focal_um: float,
) -> np.ndarray:
    """Compute expected spot positions from Zernike coefficients."""
    slopes_vec = G @ coeffs
    expected_pos = ref.copy()
    expected_pos[:, 0] += focal_um * slopes_vec[:n_sub]
    expected_pos[:, 1] += focal_um * slopes_vec[n_sub:]
    return expected_pos


# ---------------------------------------------------------------------------
# Helper: reciprocal NN matching
# ---------------------------------------------------------------------------


def _reciprocal_nn_match(
    expected_pos: np.ndarray,
    observed: np.ndarray,
    sensor_w: float,
    sensor_h: float,
    max_dist: float,
    n_terms: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Reciprocal nearest-neighbor matching between expected and observed.

    Only keeps mutual matches within max_dist.
    """
    from scipy.spatial import cKDTree

    in_bounds = (
        (expected_pos[:, 0] >= 0)
        & (expected_pos[:, 0] <= sensor_w)
        & (expected_pos[:, 1] >= 0)
        & (expected_pos[:, 1] <= sensor_h)
    )
    ib_idx = np.where(in_bounds)[0]
    if len(ib_idx) < n_terms:
        return np.array([], dtype=int), np.array([], dtype=int), 0

    E_in = expected_pos[ib_idx]
    obs_tree = cKDTree(observed)
    exp_tree = cKDTree(E_in)

    dists_fwd, nn_fwd = obs_tree.query(E_in)
    _, nn_bwd = exp_tree.query(observed)

    matched_sub_list = []
    matched_obs_list = []
    used_obs = set()

    for i in range(len(ib_idx)):
        j = nn_fwd[i]
        if dists_fwd[i] <= max_dist and nn_bwd[j] == i and j not in used_obs:
            matched_sub_list.append(ib_idx[i])
            matched_obs_list.append(j)
            used_obs.add(j)

    n_matched = len(matched_sub_list)
    if n_matched == 0:
        return np.array([], dtype=int), np.array([], dtype=int), 0

    return (
        np.array(matched_sub_list, dtype=int),
        np.array(matched_obs_list, dtype=int),
        n_matched,
    )


def _forward_nn_match(
    expected_pos: np.ndarray,
    observed: np.ndarray,
    sensor_w: float,
    sensor_h: float,
    max_dist: float,
    n_terms: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Forward-only NN matching (less strict fallback)."""
    from scipy.spatial import cKDTree

    in_bounds = (
        (expected_pos[:, 0] >= 0)
        & (expected_pos[:, 0] <= sensor_w)
        & (expected_pos[:, 1] >= 0)
        & (expected_pos[:, 1] <= sensor_h)
    )
    ib_idx = np.where(in_bounds)[0]
    if len(ib_idx) < n_terms:
        return np.array([], dtype=int), np.array([], dtype=int), 0

    E_in = expected_pos[ib_idx]
    obs_tree = cKDTree(observed)
    dists_fwd, nn_fwd = obs_tree.query(E_in)

    matched_sub_list = []
    matched_obs_list = []
    used_obs = set()

    order = np.argsort(dists_fwd)
    for i in order:
        j = nn_fwd[i]
        if dists_fwd[i] <= max_dist and j not in used_obs:
            matched_sub_list.append(ib_idx[i])
            matched_obs_list.append(j)
            used_obs.add(j)

    n_matched = len(matched_sub_list)
    if n_matched == 0:
        return np.array([], dtype=int), np.array([], dtype=int), 0

    return (
        np.array(matched_sub_list, dtype=int),
        np.array(matched_obs_list, dtype=int),
        n_matched,
    )


# ---------------------------------------------------------------------------
# Center-out initialization
# ---------------------------------------------------------------------------


def _center_ring_match(
    observed: np.ndarray,
    ref: np.ndarray,
    sensor_w: float,
    sensor_h: float,
    pitch_um: float,
    max_radius_frac: float = 0.35,
    tilt_correct: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Match observed spots to reference near the optical center.

    Near the center, Zernike displacements are smallest (zero for defocus,
    astigmatism, etc.). Only global tilt produces displacement at center.
    We compensate for tilt by shifting observed spots by the difference
    in center-of-mass.

    Strategy:
    1. Estimate global tilt from center-of-mass offset between obs and ref.
    2. Shift observed positions to remove estimated tilt.
    3. For reference spots within radius < max_radius_frac * aperture_radius,
       find nearest shifted-observed spot.
    4. Keep only reciprocal matches within pitch distance.

    Args:
        observed: (M, 2) observed spot positions.
        ref: (N, 2) reference spot positions.
        sensor_w, sensor_h: sensor dimensions in um.
        pitch_um: lenslet pitch in um.
        max_radius_frac: fraction of aperture radius to consider "center".
        tilt_correct: whether to apply tilt correction.

    Returns:
        sub_idx, obs_idx, n_matched
    """
    from scipy.spatial import cKDTree

    # Optical center
    cx = sensor_w / 2.0
    cy = sensor_h / 2.0

    # Aperture radius (from reference grid extent)
    ref_r = np.sqrt((ref[:, 0] - cx) ** 2 + (ref[:, 1] - cy) ** 2)
    aperture_r = ref_r.max()
    center_r = max_radius_frac * aperture_r

    # Select reference spots near center
    center_mask = ref_r <= center_r
    center_ref_idx = np.where(center_mask)[0]
    if len(center_ref_idx) < 5:
        return np.array([], dtype=int), np.array([], dtype=int), 0

    # Estimate tilt: center-of-mass offset
    if tilt_correct and len(observed) > 0:
        obs_com = np.mean(observed, axis=0)
        # Use only center reference spots for COM comparison
        # But we need to estimate which observed spots correspond to center...
        # Use ALL reference COM vs ALL observed COM as a tilt proxy
        ref_com = np.mean(ref, axis=0)
        tilt_offset = obs_com - ref_com
    else:
        tilt_offset = np.zeros(2)

    # Shift observed to compensate tilt
    obs_shifted = observed - tilt_offset

    # NN match: center ref -> shifted observed
    obs_tree = cKDTree(obs_shifted)
    center_ref = ref[center_ref_idx]

    dists, nn_idx = obs_tree.query(center_ref)

    # Also do backward match for reciprocity
    center_tree = cKDTree(center_ref)
    _, nn_bwd = center_tree.query(obs_shifted)

    # Keep reciprocal matches within generous distance
    max_dist = pitch_um * 1.0  # 1 pitch tolerance for initial matching
    matched_sub = []
    matched_obs = []
    used_obs = set()

    for i in range(len(center_ref_idx)):
        j = nn_idx[i]
        if dists[i] <= max_dist and nn_bwd[j] == i and j not in used_obs:
            matched_sub.append(center_ref_idx[i])
            matched_obs.append(j)
            used_obs.add(j)

    n_matched = len(matched_sub)
    if n_matched == 0:
        return np.array([], dtype=int), np.array([], dtype=int), 0

    return (
        np.array(matched_sub, dtype=int),
        np.array(matched_obs, dtype=int),
        n_matched,
    )


def _multi_radius_init(
    observed: np.ndarray,
    ref: np.ndarray,
    G: np.ndarray,
    n_sub: int,
    n_terms: int,
    focal_um: float,
    lambda_reg: float,
    sensor_w: float,
    sensor_h: float,
    pitch_um: float,
) -> Tuple[Optional[np.ndarray], int]:
    """Try center matching at multiple radii to find initial coefficients.

    Starts with a small center radius and expands if too few matches.
    Once we have enough matches for a least-squares solve, we return
    the initial coefficients.

    Returns:
        coeffs or None, n_initial_matches
    """
    # Minimum matches: need at least n_terms+2 for a solvable system
    # (n_terms for x slopes + n_terms for y slopes, so n_terms+2 pairs
    # gives 2*(n_terms+2) equations for n_terms unknowns)
    min_matches = n_terms + 2

    radii = [0.15, 0.25, 0.35, 0.5, 0.7, 1.0]

    for r_frac in radii:
        sub_idx, obs_idx, n_matched = _center_ring_match(
            observed,
            ref,
            sensor_w,
            sensor_h,
            pitch_um,
            max_radius_frac=r_frac,
            tilt_correct=True,
        )

        if n_matched >= min_matches:
            try:
                coeffs = _solve_coeffs(
                    sub_idx,
                    obs_idx,
                    observed,
                    ref,
                    G,
                    n_sub,
                    n_terms,
                    focal_um,
                    lambda_reg,
                )
                return coeffs, n_matched
            except (np.linalg.LinAlgError, ValueError):
                continue

    # Last resort: try without tilt correction
    for r_frac in [0.35, 0.5, 0.7, 1.0]:
        sub_idx, obs_idx, n_matched = _center_ring_match(
            observed,
            ref,
            sensor_w,
            sensor_h,
            pitch_um,
            max_radius_frac=r_frac,
            tilt_correct=False,
        )

        if n_matched >= min_matches:
            try:
                coeffs = _solve_coeffs(
                    sub_idx,
                    obs_idx,
                    observed,
                    ref,
                    G,
                    n_sub,
                    n_terms,
                    focal_um,
                    lambda_reg,
                )
                return coeffs, n_matched
            except (np.linalg.LinAlgError, ValueError):
                continue

    return None, 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def sorting_match(
    observed: np.ndarray,
    lenslet: LensletArray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Match observed spots to subapertures using center-out approach.

    Algorithm:
    1. Match spots near optical center (small displacement) to get initial
       Zernike coefficients.
    2. Use coefficients to predict expected positions for all subapertures.
    3. Reciprocal NN match between predicted and observed.
    4. Re-solve for better coefficients.
    5. Iterate until convergence (progressively tightening match radius).
    6. Track best solution by trimmed residual.

    This replaces the broken sorting-based approach that failed on large
    sensors (>512px) where row structure is destroyed at high PV.

    Args:
        observed: (M, 2) observed spot positions in um.
        lenslet: LensletArray instance.
        cfg: Configuration dict.

    Returns:
        Dict with coeffs, success, n_matched, residual_raw, residual_trimmed.
    """
    zer_cfg = cfg.get("zernike", {})
    asm_cfg = cfg.get("asm", {})
    max_order = zer_cfg.get("order", 3)
    n_terms = num_zernike_terms(max_order)
    lambda_reg = asm_cfg.get("lambda_reg", 1e-3)
    grid_size = asm_cfg.get("grid_size", 128)

    ref = lenslet.reference_positions()
    n_sub = len(ref)
    n_obs = len(observed)
    focal_um = float(lenslet.focal_um)
    pitch_um = float(lenslet.pitch_um)
    sensor_w = float(lenslet.sensor_width_um)
    sensor_h = float(lenslet.sensor_height_um)

    if n_obs < n_terms or n_sub < n_terms:
        return _fail_result(n_terms)

    # Precompute Zernike slope matrix
    G = _get_cached_zernike_matrix(lenslet, max_order, grid_size)

    # Step 1: Center-out initialization
    coeffs, n_init = _multi_radius_init(
        observed,
        ref,
        G,
        n_sub,
        n_terms,
        focal_um,
        lambda_reg,
        sensor_w,
        sensor_h,
        pitch_um,
    )

    if coeffs is None:
        return _fail_result(n_terms)

    # Step 2: Iterative refinement (center-out initialized ICP)
    max_refine_iter = 20
    best_coeffs = coeffs.copy()
    best_sub_idx = np.array([], dtype=int)
    best_obs_idx = np.array([], dtype=int)
    best_n_matched = 0
    best_residual = float("inf")

    for it in range(max_refine_iter):
        # Predict expected positions
        expected_pos = _compute_expected_positions(coeffs, ref, G, n_sub, focal_um)

        # Progressively tighten match distance
        if it < 3:
            max_dist = pitch_um * 1.5
        elif it < 8:
            max_dist = pitch_um * 1.0
        elif it < 14:
            max_dist = pitch_um * 0.7
        else:
            max_dist = pitch_um * 0.5

        # Reciprocal NN match
        new_sub, new_obs, new_n = _reciprocal_nn_match(
            expected_pos,
            observed,
            sensor_w,
            sensor_h,
            max_dist,
            n_terms,
        )

        if new_n < n_terms:
            # Fallback to forward-only with larger radius
            new_sub, new_obs, new_n = _forward_nn_match(
                expected_pos,
                observed,
                sensor_w,
                sensor_h,
                pitch_um * 2.0,
                n_terms,
            )
            if new_n < n_terms:
                break

        # Compute trimmed residual
        dists = np.linalg.norm(expected_pos[new_sub] - observed[new_obs], axis=1)
        n_keep = max(1, int(np.ceil(0.9 * len(dists))))
        dists_sorted = np.sort(dists)
        residual_trimmed = float(np.mean(dists_sorted[:n_keep]))

        if residual_trimmed < best_residual:
            best_residual = residual_trimmed
            best_coeffs = coeffs.copy()
            best_sub_idx = new_sub.copy()
            best_obs_idx = new_obs.copy()
            best_n_matched = new_n

        # Re-solve
        new_coeffs = _solve_coeffs(
            new_sub,
            new_obs,
            observed,
            ref,
            G,
            n_sub,
            n_terms,
            focal_um,
            lambda_reg,
        )

        # Check convergence
        coeff_change = np.max(np.abs(new_coeffs - coeffs))
        coeffs = new_coeffs

        if coeff_change < 1e-6 and it > 0:
            break

    # Final evaluation
    if best_n_matched == 0:
        return _fail_result(n_terms)

    expected_pos = _compute_expected_positions(best_coeffs, ref, G, n_sub, focal_um)
    matched_expected = expected_pos[best_sub_idx]
    matched_observed = observed[best_obs_idx]
    dists = np.linalg.norm(matched_expected - matched_observed, axis=1)
    residual_raw = float(np.mean(dists))

    n_keep = max(1, int(np.ceil(0.9 * len(dists))))
    dists_sorted = np.sort(dists)
    residual_trimmed = float(np.mean(dists_sorted[:n_keep]))

    success_threshold = pitch_um * 0.5
    success = residual_trimmed < success_threshold

    return {
        "coeffs": best_coeffs,
        "success": success,
        "n_matched": best_n_matched,
        "residual_raw": residual_raw,
        "residual_trimmed": residual_trimmed,
        "matched_sub_idx": best_sub_idx,
        "matched_obs_idx": best_obs_idx,
        "solver": "sorting_match",
    }


def _fail_result(n_terms: int) -> Dict[str, Any]:
    """Return a failure result dict."""
    return {
        "coeffs": np.zeros(n_terms),
        "success": False,
        "n_matched": 0,
        "residual_raw": float("inf"),
        "residual_trimmed": float("inf"),
        "matched_sub_idx": np.array([], dtype=int),
        "matched_obs_idx": np.array([], dtype=int),
        "solver": "sorting_match",
    }


# ---------------------------------------------------------------------------
# GPU wrapper
# ---------------------------------------------------------------------------


def sorting_match_gpu(
    observed: np.ndarray,
    lenslet: LensletArray,
    cfg: Dict[str, Any],
    device=None,
) -> Dict[str, Any]:
    """GPU-accelerated version: CPU matching + GPU least-squares."""
    cpu_result = sorting_match(observed, lenslet, cfg)

    if not cpu_result["success"] or cpu_result["n_matched"] == 0:
        cpu_result["solver"] = "sorting_match_gpu"
        return cpu_result

    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    zer_cfg = cfg.get("zernike", {})
    asm_cfg = cfg.get("asm", {})
    max_order = zer_cfg.get("order", 3)
    n_terms = num_zernike_terms(max_order)
    lambda_reg = asm_cfg.get("lambda_reg", 1e-3)
    grid_size = asm_cfg.get("grid_size", 128)

    ref = lenslet.reference_positions()
    n_sub = len(ref)
    focal_um = float(lenslet.focal_um)
    pitch_um = float(lenslet.pitch_um)

    sub_idx = cpu_result["matched_sub_idx"]
    obs_idx = cpu_result["matched_obs_idx"]
    n_matched = cpu_result["n_matched"]

    from src.recon.asm_gpu import _get_cached_g_tensor

    G_t = _get_cached_g_tensor(lenslet, max_order, grid_size, device)
    ref_t = torch.tensor(ref, dtype=torch.float32, device=device)
    obs_t = torch.tensor(observed, dtype=torch.float32, device=device)
    sub_idx_t = torch.tensor(sub_idx[:n_matched], dtype=torch.long, device=device)
    obs_idx_t = torch.tensor(obs_idx[:n_matched], dtype=torch.long, device=device)

    target_disp = obs_t[obs_idx_t] - ref_t[sub_idx_t]
    target_slopes = target_disp / focal_um

    G_sub_x = G_t[sub_idx_t, :]
    G_sub_y = G_t[n_sub + sub_idx_t, :]
    A = torch.cat([G_sub_x, G_sub_y], dim=0)
    b = torch.cat([target_slopes[:, 0], target_slopes[:, 1]], dim=0)

    if lambda_reg > 0:
        ATA = A.T @ A + lambda_reg * torch.eye(
            n_terms, device=device, dtype=torch.float32
        )
        ATb = A.T @ b
        try:
            coeffs_t = torch.linalg.solve(ATA, ATb)
        except RuntimeError:
            coeffs_t = torch.linalg.lstsq(A, b.unsqueeze(1)).solution.squeeze(1)
    else:
        coeffs_t = torch.linalg.lstsq(A, b.unsqueeze(1)).solution.squeeze(1)

    slopes_vec = G_t @ coeffs_t
    slopes_x = slopes_vec[:n_sub]
    slopes_y = slopes_vec[n_sub:]
    E_x = ref_t[:, 0] + focal_um * slopes_x
    E_y = ref_t[:, 1] + focal_um * slopes_y
    E = torch.stack([E_x, E_y], dim=1)

    matched_expected = E[sub_idx_t]
    matched_observed = obs_t[obs_idx_t]
    dists = torch.norm(matched_expected - matched_observed, dim=1)

    residual_raw = float(dists.mean().item()) if dists.numel() else float("inf")
    if dists.numel():
        n_keep = max(1, int(np.ceil(0.9 * dists.numel())))
        d_sorted, _ = torch.sort(dists)
        residual_trimmed = float(d_sorted[:n_keep].mean().item())
    else:
        residual_trimmed = float("inf")

    success_threshold = pitch_um * 0.5
    coeffs = coeffs_t.detach().cpu().numpy()
    success = residual_trimmed < success_threshold

    return {
        "coeffs": coeffs,
        "success": success,
        "n_matched": n_matched,
        "residual_raw": residual_raw,
        "residual_trimmed": residual_trimmed,
        "matched_sub_idx": sub_idx[:n_matched],
        "matched_obs_idx": obs_idx[:n_matched],
        "solver": "sorting_match_gpu",
    }
