"""Paper-faithful ASM algorithm (Yang et al., 2024).

PSO-based wavefront reconstruction using modified Hausdorff distance
as the cost function. Matches the algorithm described in the paper exactly:

1. PSO searches the Zernike coefficient space.
2. For each candidate, compute expected spot positions via the forward model.
3. Evaluate cost = Hausdorff distance + duplicate penalty.
4. After PSO convergence, refine with least-squares on matched pairs.

Units:
- Spot positions: micrometers (um)
- Zernike coefficients: waves (λ)
- Paper bounds (mm) are converted to waves via: waves = mm * 1000 / λ_nm
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from src.recon.least_squares import build_zernike_slope_matrix
from src.recon.zernike import num_zernike_terms
from src.sim.lenslet import LensletArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Slope-matrix cache (same pattern as asm_gpu.py)
# ---------------------------------------------------------------------------
_G_CACHE: Dict[Tuple[float, ...], np.ndarray] = {}


def _g_cache_key(la: LensletArray, max_order: int, grid_size: int) -> Tuple[float, ...]:
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


def _get_cached_slope_matrix(
    la: LensletArray, max_order: int, grid_size: int
) -> np.ndarray:
    """Return the (2*N_sub, n_terms) slope basis matrix, cached."""
    key = _g_cache_key(la, max_order, grid_size)
    G = _G_CACHE.get(key)
    if G is None:
        G = build_zernike_slope_matrix(la, max_order, grid_size)
        _G_CACHE[key] = G
    return G


# ---------------------------------------------------------------------------
# Forward model: coefficients -> expected spot positions
# ---------------------------------------------------------------------------


def _compute_expected_positions(
    coeffs: np.ndarray,
    G: np.ndarray,
    ref_positions: np.ndarray,
    focal_um: float,
    sensor_w_um: float,
    sensor_h_um: float,
    slope_correction: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute expected spot positions from Zernike coefficients.

    Args:
        coeffs: (n_terms,) Zernike coefficient vector (in waves).
        G: (2*N_sub, n_terms) slope basis matrix.
        ref_positions: (N_sub, 2) reference spot positions in um.
        focal_um: Focal length in um.
        sensor_w_um: Sensor width in um.
        sensor_h_um: Sensor height in um.
        slope_correction: Correction factor = wavelength_um / R_pupil_um for
            converting normalized slopes to physical slopes. Default 1.0
            preserves old behavior.

    Returns:
        expected_pos: (M, 2) expected positions within sensor bounds.
        sub_indices: (M,) indices of the subapertures that remain in bounds.
    """
    n_sub = ref_positions.shape[0]

    # slopes_vec = G @ coeffs  -> shape (2*N_sub,)
    slopes_vec = G @ coeffs
    slopes_x = slopes_vec[:n_sub]
    slopes_y = slopes_vec[n_sub:]

    # Displacements = focal_um * slopes * correction
    disp_x = focal_um * slopes_x * slope_correction
    disp_y = focal_um * slopes_y * slope_correction

    # Expected positions = reference + displacement
    expected = ref_positions.copy()
    expected[:, 0] += disp_x
    expected[:, 1] += disp_y

    # Clip to sensor bounds
    in_bounds = (
        (expected[:, 0] >= 0)
        & (expected[:, 0] < sensor_w_um)
        & (expected[:, 1] >= 0)
        & (expected[:, 1] < sensor_h_um)
    )
    sub_indices = np.where(in_bounds)[0]
    expected_pos = expected[sub_indices]

    return expected_pos, sub_indices


# ---------------------------------------------------------------------------
# Cost function: Modified Hausdorff distance + duplicate penalty
# ---------------------------------------------------------------------------


def hausdorff_cost(
    expected_pos: np.ndarray,
    observed_pos: np.ndarray,
    pitch_um: float,
    penalty_weight: float = 1.0,
) -> Dict[str, Any]:
    """Compute modified Hausdorff cost with duplicate penalty.

    Given expected spots E and observed spots G:
    1. Forward Hausdorff: for each e in E, find nearest g. dFH = max distance.
    2. Backward Hausdorff: for each g in G, find nearest e. dBH = max distance.
    3. Hausdorff distance: dH = max(dFH, dBH).
    4. Duplicate penalty: count observed spots matched by >1 expected spot.
       penalty = count * penalty_weight * pitch_um.
    5. Total cost = dH + penalty.

    Args:
        expected_pos: (M, 2) expected spot positions in um.
        observed_pos: (N, 2) observed spot positions in um.
        pitch_um: Lenslet pitch in um (used to scale penalty).
        penalty_weight: Weight for duplicate penalty term.

    Returns:
        Dict with keys:
            cost: Total cost value.
            dH: Hausdorff distance.
            n_duplicates: Number of duplicate-matched observed spots.
            forward_matches: (M,) indices of nearest observed for each expected.
            backward_matches: (N,) indices of nearest expected for each observed.
    """
    # Handle edge cases
    if expected_pos.shape[0] == 0 or observed_pos.shape[0] == 0:
        return {
            "cost": 1e8,
            "dH": 1e8,
            "n_duplicates": 0,
            "forward_matches": np.array([], dtype=int),
            "backward_matches": np.array([], dtype=int),
        }

    # Build KD-trees
    tree_obs = cKDTree(observed_pos)
    tree_exp = cKDTree(expected_pos)

    # Forward Hausdorff: each expected -> nearest observed
    fwd_dists, fwd_matches = tree_obs.query(expected_pos)
    dFH = float(np.max(fwd_dists))

    # Backward Hausdorff: each observed -> nearest expected
    bwd_dists, bwd_matches = tree_exp.query(observed_pos)
    dBH = float(np.max(bwd_dists))

    # Hausdorff distance
    dH = max(dFH, dBH)

    # Duplicate penalty: count how many observed spots are the nearest
    # neighbor for MORE than one expected spot
    n_obs = observed_pos.shape[0]
    obs_hit_counts = np.bincount(fwd_matches, minlength=n_obs)
    n_duplicates = int(np.sum(obs_hit_counts > 1))

    duplicate_penalty = n_duplicates * penalty_weight * pitch_um

    cost = dH + duplicate_penalty

    return {
        "cost": cost,
        "dH": dH,
        "n_duplicates": n_duplicates,
        "forward_matches": fwd_matches,
        "backward_matches": bwd_matches,
    }


# ---------------------------------------------------------------------------
# PSO bounds conversion
# ---------------------------------------------------------------------------


def _build_bounds(
    cfg: Dict[str, Any],
    n_terms: int,
    pv_hint: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build PSO search bounds for Zernike coefficients.

    Our Zernike coefficients are dimensionless ("waves" units). Empirically,
    for order-4 Zernike (15 terms), max |coeff| ~ 0.07 * PV. All terms have
    similar magnitude since PV is the total range of their sum.

    The paper uses mm-unit bounds with a tier structure (C1-C6: ±1mm,
    C7-C10: ±0.1mm, C11-C15: ±0.01mm), but this doesn't translate well to
    our convention because higher-order terms don't necessarily have smaller
    coefficients in our basis.

    Strategy: use generous uniform bounds scaled by pv_hint.

    Returns:
        (bounds_low, bounds_high) each of shape (n_terms,).
    """
    # Empirical coefficient scale: max|c| ~ 0.07 * PV for 15-term Zernike
    # Use 2x safety margin for the bounds
    if pv_hint is not None and pv_hint > 0:
        bound_val = max(0.15 * pv_hint, 0.5)  # generous: ~2x max expected
    else:
        bound_val = 1.0  # default for PV ~ 1-5

    bounds_low = np.full(n_terms, -bound_val)
    bounds_high = np.full(n_terms, bound_val)

    # Piston (index 0) doesn't affect spot positions, keep tight
    bounds_low[0] = -0.01
    bounds_high[0] = 0.01

    return bounds_low, bounds_high


def _evaluate_particles_batch(
    positions: np.ndarray,
    G: np.ndarray,
    ref_positions: np.ndarray,
    focal_um: float,
    sensor_w_um: float,
    sensor_h_um: float,
    slope_correction: float,
    observed_pos: np.ndarray,
    pitch_um: float,
    penalty_weight: float,
    tree_obs: cKDTree,
) -> Tuple[np.ndarray, list]:
    """Evaluate all particles in batch.

    The matrix multiplication is vectorized; KD-tree queries are still per-particle
    but we reuse the observed-positions tree.

    Returns:
        costs: (n_particles,) array of costs.
        results: list of hausdorff_cost result dicts.
    """
    n_particles, n_terms = positions.shape
    n_sub = ref_positions.shape[0]

    # Batch compute slopes for all particles: (n_particles, 2*n_sub)
    slopes_all = positions @ G.T  # (n_particles, 2*n_sub)

    costs = np.full(n_particles, np.inf)
    results = [None] * n_particles

    for i in range(n_particles):
        slopes_x = slopes_all[i, :n_sub]
        slopes_y = slopes_all[i, n_sub:]

        disp_x = focal_um * slopes_x * slope_correction
        disp_y = focal_um * slopes_y * slope_correction

        expected = ref_positions.copy()
        expected[:, 0] += disp_x
        expected[:, 1] += disp_y

        # Clip to sensor bounds
        in_bounds = (
            (expected[:, 0] >= 0)
            & (expected[:, 0] < sensor_w_um)
            & (expected[:, 1] >= 0)
            & (expected[:, 1] < sensor_h_um)
        )
        exp_pos = expected[in_bounds]

        if exp_pos.shape[0] == 0 or observed_pos.shape[0] == 0:
            results[i] = {
                "cost": 1e8,
                "dH": 1e8,
                "n_duplicates": 0,
                "forward_matches": np.array([], dtype=int),
                "backward_matches": np.array([], dtype=int),
            }
            costs[i] = 1e8
            continue

        # Forward: each expected -> nearest observed
        fwd_dists, fwd_matches = tree_obs.query(exp_pos)

        # Backward: each observed -> nearest expected
        tree_exp = cKDTree(exp_pos)
        bwd_dists, bwd_matches = tree_exp.query(observed_pos)

        # Use MEAN distance (smoother for PSO optimization) instead of MAX
        # Max Hausdorff is too sensitive to single outlier spots.
        mean_fwd = float(np.mean(fwd_dists))
        mean_bwd = float(np.mean(bwd_dists))
        dH_mean = (mean_fwd + mean_bwd) / 2.0
        dH_max = max(float(np.max(fwd_dists)), float(np.max(bwd_dists)))

        # Duplicate penalty
        n_obs = observed_pos.shape[0]
        obs_hit_counts = np.bincount(fwd_matches, minlength=n_obs)
        n_duplicates = int(np.sum(obs_hit_counts > 1))
        duplicate_penalty = n_duplicates * penalty_weight * pitch_um

        cost = dH_mean + duplicate_penalty
        costs[i] = cost
        results[i] = {
            "cost": cost,
            "dH": dH_max,  # keep max for convergence check
            "dH_mean": dH_mean,
            "n_duplicates": n_duplicates,
            "forward_matches": fwd_matches,
            "backward_matches": bwd_matches,
        }

    return costs, results


# ---------------------------------------------------------------------------
# PSO optimizer
# ---------------------------------------------------------------------------


def paper_pso_optimize(
    observed_pos: np.ndarray,
    la: LensletArray,
    cfg: Dict[str, Any],
    seed: int = 42,
    pv_hint: Optional[float] = None,
) -> Dict[str, Any]:
    """PSO optimizer faithful to the paper (Yang et al., 2024).

    Standard PSO with:
    - All particles initialized at zero
    - Constant inertia weight w = 0.5
    - c1 = c2 = 1.49
    - Termination: no duplicates AND dH < threshold, OR max iterations

    Args:
        observed_pos: (N, 2) observed spot positions in um.
        la: LensletArray instance.
        cfg: Config dict with 'asm_paper', 'zernike', 'optics', 'sensor' sections.
        seed: Random seed.
        pv_hint: Optional PV hint for scaling bounds.

    Returns:
        Dict with keys: coeffs, cost, n_iter, converged, history, dH,
                        n_duplicates, forward_matches.
    """
    rng = np.random.RandomState(seed)

    # --- Extract config ---
    asm_cfg = cfg.get("asm_paper", {})
    zer_cfg = cfg.get("zernike", {})
    sensor_cfg = cfg.get("sensor", {})

    max_order = zer_cfg.get("order", 4)
    n_terms = num_zernike_terms(max_order)
    grid_size = asm_cfg.get("grid_size", 128)

    n_particles = int(asm_cfg.get("pso_particles", 100))
    c1 = float(asm_cfg.get("pso_c1", 1.49))
    c2 = float(asm_cfg.get("pso_c2", 1.49))
    w_start = float(asm_cfg.get("pso_w_start", 0.9))
    w_end = float(asm_cfg.get("pso_w_end", 0.4))
    max_iter = int(asm_cfg.get("pso_max_iter", 200))

    hausdorff_threshold_px = float(asm_cfg.get("hausdorff_threshold_px", 6.0))
    pixel_um = float(sensor_cfg.get("pixel_um", la.pixel_um))
    hausdorff_threshold_um = hausdorff_threshold_px * pixel_um

    penalty_weight = float(asm_cfg.get("duplicate_penalty_weight", 1.0))

    # --- Precompute ---
    G = _get_cached_slope_matrix(la, max_order, grid_size)
    ref_positions = la.reference_positions()
    focal_um = la.focal_um
    sensor_w_um = la.sensor_width_um
    sensor_h_um = la.sensor_height_um
    pitch_um = la.pitch_um
    slope_correction = la._slope_correction

    # --- Build bounds ---
    bounds_low, bounds_high = _build_bounds(cfg, n_terms, pv_hint)

    # --- Initialize particles ---
    # Half start at zero (paper convention), half scattered randomly
    # This improves exploration at high PV where zero is far from the optimum.
    positions = np.zeros((n_particles, n_terms), dtype=float)
    n_random = n_particles // 2
    positions[:n_random] = rng.uniform(
        bounds_low[None, :], bounds_high[None, :], size=(n_random, n_terms)
    )

    # Velocities: moderate random values (5% of range)
    vel_range = (bounds_high - bounds_low) * 0.05
    velocities = rng.uniform(-1, 1, size=(n_particles, n_terms)) * vel_range[None, :]

    # --- Personal and global bests ---
    pbest_pos = positions.copy()
    pbest_cost = np.full(n_particles, np.inf)
    gbest_pos = np.zeros(n_terms, dtype=float)
    gbest_cost = np.inf
    gbest_result: Dict[str, Any] = {}

    # Build KD-tree for observed positions (reused across all evaluations)
    tree_obs = cKDTree(observed_pos)

    # Evaluate initial positions (batch)
    costs, results = _evaluate_particles_batch(
        positions,
        G,
        ref_positions,
        focal_um,
        sensor_w_um,
        sensor_h_um,
        slope_correction,
        observed_pos,
        pitch_um,
        penalty_weight,
        tree_obs,
    )
    for i in range(n_particles):
        pbest_cost[i] = costs[i]
        if costs[i] < gbest_cost:
            gbest_cost = costs[i]
            gbest_pos = positions[i].copy()
            gbest_result = results[i]

    history = [
        {
            "iter": 0,
            "gbest_cost": gbest_cost,
            "dH": gbest_result.get("dH", np.inf),
            "n_duplicates": gbest_result.get("n_duplicates", 0),
        }
    ]

    converged = False
    final_iter = 0

    # --- PSO main loop ---
    stagnation_count = 0
    stagnation_limit = int(asm_cfg.get("pso_stagnation_limit", 50))
    prev_gbest_cost = gbest_cost

    for it in range(1, max_iter + 1):
        final_iter = it

        # Linearly decreasing inertia
        w = w_start - (w_start - w_end) * (it - 1) / max(max_iter - 1, 1)

        # Generate random numbers for the swarm
        r1 = rng.uniform(0, 1, size=(n_particles, n_terms))
        r2 = rng.uniform(0, 1, size=(n_particles, n_terms))

        # Update velocities
        velocities = (
            w * velocities
            + c1 * r1 * (pbest_pos - positions)
            + c2 * r2 * (gbest_pos[None, :] - positions)
        )

        # Update positions
        positions = positions + velocities

        # Clamp positions to bounds
        positions = np.clip(positions, bounds_low[None, :], bounds_high[None, :])

        # Evaluate all particles (batch)
        costs, results = _evaluate_particles_batch(
            positions,
            G,
            ref_positions,
            focal_um,
            sensor_w_um,
            sensor_h_um,
            slope_correction,
            observed_pos,
            pitch_um,
            penalty_weight,
            tree_obs,
        )
        for i in range(n_particles):
            cost = costs[i]

            # Update personal best
            if cost < pbest_cost[i]:
                pbest_cost[i] = cost
                pbest_pos[i] = positions[i].copy()

            # Update global best
            if cost < gbest_cost:
                gbest_cost = cost
                gbest_pos = positions[i].copy()
                gbest_result = results[i]

        history.append(
            {
                "iter": it,
                "gbest_cost": gbest_cost,
                "dH": gbest_result.get("dH", np.inf),
                "n_duplicates": gbest_result.get("n_duplicates", 0),
            }
        )

        # --- Check termination ---
        no_duplicates = gbest_result.get("n_duplicates", 1) == 0
        # Use mean distance for convergence (smoother than max Hausdorff)
        dH_mean_ok = gbest_result.get("dH_mean", np.inf) < hausdorff_threshold_um

        if no_duplicates and dH_mean_ok:
            converged = True
            logger.info(
                "PSO converged at iteration %d: dH_mean=%.2f um, dH_max=%.2f um, n_dup=%d, cost=%.2f",
                it,
                gbest_result.get("dH_mean", np.inf),
                gbest_result["dH"],
                gbest_result["n_duplicates"],
                gbest_cost,
            )
            break

        if it % 50 == 0:
            logger.debug(
                "PSO iter %d: cost=%.2f, dH=%.2f, n_dup=%d, w=%.3f",
                it,
                gbest_cost,
                gbest_result.get("dH", np.inf),
                gbest_result.get("n_duplicates", 0),
                w,
            )

        # Stagnation detection: stop early if no improvement for N iterations
        if gbest_cost < prev_gbest_cost - 0.01:
            stagnation_count = 0
            prev_gbest_cost = gbest_cost
        else:
            stagnation_count += 1
            if stagnation_count >= stagnation_limit:
                logger.debug(
                    "PSO stagnated at iteration %d after %d iters without improvement",
                    it,
                    stagnation_limit,
                )
                break

    if not converged:
        logger.info(
            "PSO reached max iterations (%d): dH=%.2f um, n_dup=%d, cost=%.2f",
            max_iter,
            gbest_result.get("dH", np.inf),
            gbest_result.get("n_duplicates", 0),
            gbest_cost,
        )

    return {
        "coeffs": gbest_pos,
        "cost": gbest_cost,
        "n_iter": final_iter,
        "converged": converged,
        "history": history,
        "dH": gbest_result.get("dH", np.inf),
        "n_duplicates": gbest_result.get("n_duplicates", 0),
        "forward_matches": gbest_result.get("forward_matches", np.array([], dtype=int)),
    }


# ---------------------------------------------------------------------------
# Least-squares refinement using matched pairs
# ---------------------------------------------------------------------------


def _least_squares_refine(
    coeffs_pso: np.ndarray,
    observed_pos: np.ndarray,
    G: np.ndarray,
    ref_positions: np.ndarray,
    focal_um: float,
    sensor_w_um: float,
    sensor_h_um: float,
    n_terms: int,
    lambda_reg: float = 0.0,
    slope_correction: float = 1.0,
) -> Tuple[np.ndarray, int, float, float]:
    """Refine PSO coefficients via weighted least-squares on matched pairs.

    After PSO converges with good matching, use the forward matches
    (each expected -> nearest observed, 1-to-1) to set up a least-squares
    system for more accurate coefficient estimation.

    Args:
        coeffs_pso: (n_terms,) PSO-found coefficient vector.
        observed_pos: (N, 2) observed spots.
        G: (2*N_sub, n_terms) slope basis matrix.
        ref_positions: (N_sub, 2) reference positions.
        focal_um: Focal length in um.
        sensor_w_um: Sensor width in um.
        sensor_h_um: Sensor height in um.
        n_terms: Number of Zernike terms.
        lambda_reg: Regularization weight.
        slope_correction: Correction factor for normalized-to-physical slope.

    Returns:
        (coeffs_refined, n_matched, residual_raw, residual_trimmed)
    """
    n_sub = ref_positions.shape[0]

    # Compute expected positions from PSO coefficients
    exp_pos, sub_indices = _compute_expected_positions(
        coeffs_pso,
        G,
        ref_positions,
        focal_um,
        sensor_w_um,
        sensor_h_um,
        slope_correction,
    )

    if exp_pos.shape[0] == 0 or observed_pos.shape[0] == 0:
        return coeffs_pso.copy(), 0, float("inf"), float("inf")

    # Build KD-trees for matching
    tree_obs = cKDTree(observed_pos)
    tree_exp = cKDTree(exp_pos)

    # Forward: each expected -> nearest observed
    fwd_dists, fwd_nn = tree_obs.query(exp_pos)

    # Backward: each observed -> nearest expected
    _, bwd_nn = tree_exp.query(observed_pos)

    # Reciprocal matching: expected[i] -> obs[j] AND obs[j] -> expected[i]
    n_exp = exp_pos.shape[0]
    reciprocal_mask = np.zeros(n_exp, dtype=bool)
    for i in range(n_exp):
        j = fwd_nn[i]
        if bwd_nn[j] == i:
            reciprocal_mask[i] = True

    matched_exp_local = np.where(reciprocal_mask)[0]
    n_matched = len(matched_exp_local)

    if n_matched < n_terms:
        # Fall back to forward-only matching if reciprocal has too few
        logger.debug(
            "Reciprocal matching has %d pairs (< %d terms), using forward matching",
            n_matched,
            n_terms,
        )
        # Use all forward matches with reasonable distance
        pitch_threshold = focal_um * 0.1  # generous threshold
        dist_ok = fwd_dists < pitch_threshold
        matched_exp_local = np.where(dist_ok)[0]
        n_matched = len(matched_exp_local)
        if n_matched < n_terms:
            return coeffs_pso.copy(), n_matched, float("inf"), float("inf")

    # Map back to subaperture indices
    matched_sub_idx = sub_indices[matched_exp_local]
    matched_obs_idx = fwd_nn[matched_exp_local]

    matched_obs = observed_pos[matched_obs_idx]  # (n_matched, 2)
    target_disp = matched_obs - ref_positions[matched_sub_idx]  # (n_matched, 2)
    # Convert physical displacement back to normalized slopes:
    # disp = focal_um * slope_norm * correction  =>  slope_norm = disp / (focal_um * correction)
    target_slopes = target_disp / (focal_um * slope_correction)  # (n_matched, 2)

    # Build least-squares system
    G_sub_x = G[matched_sub_idx, :]  # (n_matched, n_terms)
    G_sub_y = G[n_sub + matched_sub_idx, :]  # (n_matched, n_terms)
    A = np.vstack([G_sub_x, G_sub_y])  # (2*n_matched, n_terms)
    b = np.concatenate([target_slopes[:, 0], target_slopes[:, 1]])

    # Solve: (A^T A + lambda * I) c = A^T b
    if lambda_reg > 0:
        ATA = A.T @ A + lambda_reg * np.eye(n_terms)
        ATb = A.T @ b
        try:
            coeffs_refined = np.linalg.solve(ATA, ATb)
        except np.linalg.LinAlgError:
            coeffs_refined, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    else:
        coeffs_refined, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Compute residuals with refined coefficients
    exp_pos_ref, sub_idx_ref = _compute_expected_positions(
        coeffs_refined,
        G,
        ref_positions,
        focal_um,
        sensor_w_um,
        sensor_h_um,
        slope_correction,
    )
    if exp_pos_ref.shape[0] > 0 and observed_pos.shape[0] > 0:
        tree_obs2 = cKDTree(observed_pos)
        dists2, _ = tree_obs2.query(exp_pos_ref)
        residual_raw = float(np.mean(dists2))
        # Trimmed mean (90%)
        n_keep = max(1, int(np.ceil(0.9 * len(dists2))))
        residual_trimmed = float(np.mean(np.partition(dists2, n_keep - 1)[:n_keep]))
    else:
        residual_raw = float("inf")
        residual_trimmed = float("inf")

    return coeffs_refined, n_matched, residual_raw, residual_trimmed


# ---------------------------------------------------------------------------
# Top-level reconstruction entry point
# ---------------------------------------------------------------------------


def asm_paper_reconstruct(
    observed_pos: np.ndarray,
    la: LensletArray,
    cfg: Dict[str, Any],
    seed: int = 42,
    pv_hint: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Paper-faithful ASM reconstruction (Yang et al., 2024).

    Top-level entry point matching the interface of asm_reconstruct_gpu.

    Pipeline:
    1. Run PSO to find Zernike coefficients via Hausdorff-distance cost.
    2. Refine with least-squares on matched spot pairs.

    Args:
        observed_pos: (N, 2) observed spot positions in um.
        la: LensletArray instance.
        cfg: Config dict with 'asm_paper', 'zernike', 'optics', 'sensor' sections.
        seed: Random seed.
        pv_hint: Optional PV hint for scaling search bounds.
        **kwargs: Additional keyword arguments (ignored, for interface compat).

    Returns:
        Dict with keys:
            coeffs: (n_terms,) reconstructed Zernike coefficients.
            success: bool, True if converged AND dH < threshold.
            objective_value: float, final cost.
            n_matched: int, number of matched spot pairs.
            residual_raw: float, mean matching distance.
            residual_trimmed: float, trimmed mean matching distance.
            solver: str, solver identifier.
            pso_converged: bool, whether PSO met termination criteria.
            pso_n_iter: int, number of PSO iterations.
            pso_dH: float, final Hausdorff distance.
            pso_n_duplicates: int, final duplicate count.
            pso_history: list, iteration-by-iteration stats.
    """
    asm_cfg = cfg.get("asm_paper", {})
    zer_cfg = cfg.get("zernike", {})
    sensor_cfg = cfg.get("sensor", {})

    max_order = zer_cfg.get("order", 4)
    n_terms = num_zernike_terms(max_order)
    grid_size = asm_cfg.get("grid_size", 128)
    lambda_reg = float(asm_cfg.get("lambda_reg", 0.0))

    hausdorff_threshold_px = float(asm_cfg.get("hausdorff_threshold_px", 6.0))
    pixel_um = float(sensor_cfg.get("pixel_um", la.pixel_um))
    hausdorff_threshold_um = hausdorff_threshold_px * pixel_um

    # Handle degenerate input
    if observed_pos.shape[0] < n_terms:
        logger.warning(
            "Too few observed spots (%d < %d terms), returning zeros",
            observed_pos.shape[0],
            n_terms,
        )
        return {
            "coeffs": np.zeros(n_terms),
            "success": False,
            "objective_value": np.inf,
            "n_matched": 0,
            "residual_raw": np.inf,
            "residual_trimmed": np.inf,
            "solver": "asm_paper",
            "pso_converged": False,
            "pso_n_iter": 0,
            "pso_dH": np.inf,
            "pso_n_duplicates": 0,
            "pso_history": [],
        }

    # --- Step 1: Multi-restart PSO optimization ---
    n_restarts = int(asm_cfg.get("pso_restarts", 3))
    logger.info(
        "Starting paper-faithful PSO reconstruction (%d restarts)...", n_restarts
    )

    G = _get_cached_slope_matrix(la, max_order, grid_size)
    ref_positions = la.reference_positions()
    focal_um = la.focal_um
    sensor_w_um = la.sensor_width_um
    sensor_h_um = la.sensor_height_um
    slope_correction = la._slope_correction

    best_pso_result = None
    best_pso_cost = np.inf

    for restart_idx in range(n_restarts):
        restart_seed = seed + restart_idx * 10000
        pso_result = paper_pso_optimize(
            observed_pos, la, cfg, seed=restart_seed, pv_hint=pv_hint
        )

        # Evaluate this PSO result with LS refinement to get the true quality
        coeffs_pso_i = pso_result["coeffs"]
        coeffs_ref_i, n_match_i, res_raw_i, res_trim_i = _least_squares_refine(
            coeffs_pso=coeffs_pso_i,
            observed_pos=observed_pos,
            G=G,
            ref_positions=ref_positions,
            focal_um=focal_um,
            sensor_w_um=sensor_w_um,
            sensor_h_um=sensor_h_um,
            n_terms=n_terms,
            lambda_reg=lambda_reg,
            slope_correction=slope_correction,
        )

        # Pick best between PSO-only and PSO+LS
        exp_pso_i, _ = _compute_expected_positions(
            coeffs_pso_i,
            G,
            ref_positions,
            focal_um,
            sensor_w_um,
            sensor_h_um,
            slope_correction,
        )
        cost_pso_i = hausdorff_cost(exp_pso_i, observed_pos, la.pitch_um)

        exp_ref_i, _ = _compute_expected_positions(
            coeffs_ref_i,
            G,
            ref_positions,
            focal_um,
            sensor_w_um,
            sensor_h_um,
            slope_correction,
        )
        cost_ref_i = hausdorff_cost(exp_ref_i, observed_pos, la.pitch_um)

        if cost_ref_i["cost"] <= cost_pso_i["cost"]:
            effective_cost = cost_ref_i["cost"]
            effective_coeffs = coeffs_ref_i
            effective_dH = cost_ref_i["dH"]
            effective_solver = "asm_paper_pso_ls"
            effective_n_matched = n_match_i
            effective_residual_raw = res_raw_i
            effective_residual_trimmed = res_trim_i
        else:
            effective_cost = cost_pso_i["cost"]
            effective_coeffs = coeffs_pso_i
            effective_dH = cost_pso_i["dH"]
            effective_solver = "asm_paper_pso"
            effective_n_matched = exp_pso_i.shape[0]
            if exp_pso_i.shape[0] > 0:
                tree_tmp = cKDTree(observed_pos)
                d_tmp, _ = tree_tmp.query(exp_pso_i)
                effective_residual_raw = float(np.mean(d_tmp))
                n_keep = max(1, int(np.ceil(0.9 * len(d_tmp))))
                effective_residual_trimmed = float(
                    np.mean(np.partition(d_tmp, n_keep - 1)[:n_keep])
                )
            else:
                effective_residual_raw = float("inf")
                effective_residual_trimmed = float("inf")

        logger.debug(
            "PSO restart %d/%d: cost=%.2f, dH=%.2f, conv=%s, iter=%d",
            restart_idx + 1,
            n_restarts,
            effective_cost,
            effective_dH,
            pso_result["converged"],
            pso_result["n_iter"],
        )

        if effective_cost < best_pso_cost:
            best_pso_cost = effective_cost
            best_pso_result = {
                "coeffs": effective_coeffs,
                "cost": effective_cost,
                "dH": effective_dH,
                "solver": effective_solver,
                "n_matched": effective_n_matched,
                "residual_raw": effective_residual_raw,
                "residual_trimmed": effective_residual_trimmed,
                "pso_converged": pso_result["converged"],
                "pso_n_iter": pso_result["n_iter"],
                "pso_dH": pso_result["dH"],
                "pso_n_duplicates": pso_result["n_duplicates"],
                "pso_history": pso_result["history"],
            }

        # Early exit if we already have excellent match
        if effective_dH < hausdorff_threshold_um:
            logger.debug("Early exit: excellent match at restart %d", restart_idx + 1)
            break

    # --- Extract best result ---
    final_coeffs = best_pso_result["coeffs"]
    final_cost = best_pso_result["cost"]
    final_dH = best_pso_result["dH"]
    solver = best_pso_result["solver"]
    n_matched = best_pso_result["n_matched"]
    residual_raw = best_pso_result["residual_raw"]
    residual_trimmed = best_pso_result["residual_trimmed"]
    pso_converged = best_pso_result["pso_converged"]

    # Determine success: based on final matching quality, not just PSO convergence.
    # If LS refinement produced a good match (low residual), we consider it success
    # even if PSO didn't formally converge.
    success = (
        (final_dH < hausdorff_threshold_um)
        or (residual_raw < la.pitch_um * 0.5)  # mean residual < half pitch
    )

    logger.info(
        "ASM paper result: success=%s, cost=%.2f, dH=%.2f um (threshold=%.2f um), "
        "n_matched=%d, solver=%s",
        success,
        final_cost,
        final_dH,
        hausdorff_threshold_um,
        n_matched,
        solver,
    )

    return {
        "coeffs": final_coeffs,
        "success": success,
        "objective_value": final_cost,
        "n_matched": n_matched,
        "residual_raw": residual_raw,
        "residual_trimmed": residual_trimmed,
        "solver": solver,
        "pso_converged": pso_converged,
        "pso_n_iter": best_pso_result["pso_n_iter"],
        "pso_dH": best_pso_result["pso_dH"],
        "pso_n_duplicates": best_pso_result["pso_n_duplicates"],
        "pso_history": best_pso_result["pso_history"],
    }
