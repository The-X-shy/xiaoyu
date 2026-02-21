"""GPU-accelerated baseline reconstructor (behavior-equivalent mode).

This module keeps the same matching/extrapolation logic as CPU baseline while
offloading nearest-neighbor distance computation and subset least-squares to
CUDA when available.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.sim.lenslet import LensletArray
from src.recon.zernike import num_zernike_terms
from src.recon.baseline_extrap_nn import (
    baseline_reconstruct,
    _get_cached_zernike_matrix,
    _compute_conflict_ratio,
    _compute_inside_out_order,
)

try:
    import torch
except ImportError:  # pragma: no cover - exercised on non-torch environments
    torch = None


def _cuda_available() -> bool:
    return torch is not None and torch.cuda.is_available()

_G_TENSOR_CACHE: dict[Tuple[float, ...], "torch.Tensor"] = {}


def _matrix_cache_key(la: LensletArray, max_order: int, grid_size: int) -> Tuple[float, ...]:
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


def _get_cached_g_tensor(
    lenslet: LensletArray,
    max_order: int,
    grid_size: int,
    device: "torch.device",
) -> "torch.Tensor":
    key = _matrix_cache_key(lenslet, max_order, grid_size)
    G_t = _G_TENSOR_CACHE.get(key)
    if G_t is None or G_t.device != device:
        G_np = _get_cached_zernike_matrix(lenslet, max_order, grid_size)
        G_t = torch.tensor(G_np, dtype=torch.float32, device=device)
        _G_TENSOR_CACHE[key] = G_t
    return G_t


def _nearest_obs_gpu(query_xy: "torch.Tensor", observed_t: "torch.Tensor") -> tuple[float, int]:
    diffs = observed_t - query_xy.unsqueeze(0)
    d2 = torch.sum(diffs * diffs, dim=1)
    min_d2, idx = torch.min(d2, dim=0)
    return float(torch.sqrt(min_d2).item()), int(idx.item())


def _extrapolate_gpu(
    sub_idx: int,
    ref_t: "torch.Tensor",
    matched: np.ndarray,
    matched_disp_t: "torch.Tensor",
) -> "torch.Tensor":
    if not np.any(matched):
        return ref_t[sub_idx]

    matched_indices = np.where(matched)[0]
    ref_pos = ref_t[sub_idx]
    ref_neighbors = ref_t[matched_indices]
    dists = torch.norm(ref_neighbors - ref_pos.unsqueeze(0), dim=1)
    k = min(4, len(matched_indices))
    _, nearest = torch.topk(dists, k=k, largest=False)
    neighbor_idx = torch.tensor(
        matched_indices[nearest.detach().cpu().numpy()],
        dtype=torch.long,
        device=ref_t.device,
    )
    neighbor_dists = dists[nearest]
    weights = torch.where(
        neighbor_dists > 1e-10,
        1.0 / neighbor_dists,
        torch.tensor(1e10, device=ref_t.device),
    )
    weights = weights / torch.sum(weights)
    avg_disp = torch.sum(matched_disp_t[neighbor_idx] * weights.unsqueeze(1), dim=0)
    return ref_pos + avg_disp


def _reconstruct_from_subset_gpu(
    lenslet: LensletArray,
    matched_mask: np.ndarray,
    slopes_matched_t: "torch.Tensor",
    max_order: int,
    lambda_reg: float = 0.0,
    grid_size: int = 128,
    device: Optional["torch.device"] = None,
) -> np.ndarray:
    """Subset least-squares entirely on GPU with CPU fallback only on failure."""
    n_terms = num_zernike_terms(max_order)
    if slopes_matched_t.numel() == 0:
        return np.zeros(n_terms, dtype=float)

    if torch is None:
        return np.zeros(n_terms, dtype=float)

    if device is None:
        device = torch.device("cuda" if _cuda_available() else "cpu")

    try:
        sub_idx = np.where(matched_mask)[0]
        n_sub = lenslet.n_subapertures
        G_t = _get_cached_g_tensor(lenslet, max_order, grid_size, device)
        sub_idx_t = torch.tensor(sub_idx, dtype=torch.long, device=device)
        A_t = torch.cat([G_t[sub_idx_t, :], G_t[n_sub + sub_idx_t, :]], dim=0)
        b_t = torch.cat([slopes_matched_t[:, 0], slopes_matched_t[:, 1]], dim=0)
        if lambda_reg > 0:
            ATA = A_t.T @ A_t + lambda_reg * torch.eye(n_terms, device=device)
            ATb = A_t.T @ b_t
            c_t = torch.linalg.solve(ATA, ATb)
        else:
            c_t = torch.linalg.lstsq(A_t, b_t.unsqueeze(1)).solution.squeeze(1)
        return c_t.detach().cpu().numpy()
    except Exception:
        # Fall back to CPU solver to keep behavior robust.
        slopes_matched = slopes_matched_t.detach().cpu().numpy()
        G = _get_cached_zernike_matrix(lenslet, max_order, grid_size)
        sub_idx = np.where(matched_mask)[0]
        n_sub = lenslet.n_subapertures
        A = np.vstack([G[sub_idx, :], G[n_sub + sub_idx, :]])
        b = np.concatenate([slopes_matched[:, 0], slopes_matched[:, 1]])
        c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return c


def baseline_reconstruct_gpu(
    observed: np.ndarray,
    lenslet: LensletArray,
    cfg: Dict[str, Any],
    device: Optional["torch.device"] = None,
) -> Dict[str, Any]:
    """Run baseline extrapolation + NN using CUDA for NN and LS."""
    if not _cuda_available():
        require_gpu = bool(cfg.get("baseline", {}).get("require_gpu", False))
        if require_gpu:
            raise RuntimeError("baseline.require_gpu=true but CUDA is unavailable")
        out = baseline_reconstruct(observed, lenslet, cfg)
        out["solver"] = "baseline_extrap_nn_cpu"
        return out

    if device is None:
        device = torch.device("cuda")

    bl_cfg = cfg.get("baseline", {})
    zer_cfg = cfg.get("zernike", {})

    tau_nn_px = bl_cfg.get("tau_nn_px", 5.0)
    gpu_max_dist_um = bl_cfg.get("gpu_max_dist_um", None)
    tau_nn_um = (
        float(gpu_max_dist_um)
        if gpu_max_dist_um is not None
        else float(tau_nn_px * lenslet.pixel_um)
    )
    k_fail = int(bl_cfg.get("k_fail", 3))
    conflict_max = float(bl_cfg.get("conflict_ratio_max", 0.2))
    lambda_reg = float(cfg.get("asm", {}).get("lambda_reg", 0.0))
    grid_size = int(cfg.get("asm", {}).get("grid_size", 128))
    max_order = int(zer_cfg.get("order", 3))
    n_terms = num_zernike_terms(max_order)

    ref = lenslet.reference_positions()
    n_sub = len(ref)
    if len(observed) == 0 or n_sub == 0:
        return {
            "coeffs": np.zeros(n_terms),
            "success": False,
            "rmse": float("inf"),
            "n_matched": 0,
            "conflict_ratio": 1.0,
            "solver": "baseline_extrap_nn_gpu",
        }

    obs_t = torch.tensor(observed, dtype=torch.float32, device=device)
    ref_t = torch.tensor(ref, dtype=torch.float32, device=device)
    order = _compute_inside_out_order(ref, lenslet)

    matched = np.full(n_sub, False)
    match_obs_idx = np.full(n_sub, -1, dtype=int)
    matched_disp_t = torch.zeros((n_sub, 2), dtype=torch.float32, device=device)

    # Seed matches (same policy as CPU baseline).
    for sub_idx in order[: min(4, n_sub)]:
        ref_pos_t = ref_t[sub_idx]
        _, idx = _nearest_obs_gpu(ref_pos_t, obs_t)
        matched[sub_idx] = True
        match_obs_idx[sub_idx] = idx
        matched_disp_t[sub_idx] = obs_t[idx] - ref_pos_t

    # Propagation matches (inside-out).
    consecutive_fails = 0
    for sub_idx in order:
        if matched[sub_idx]:
            continue
        expected_t = _extrapolate_gpu(sub_idx, ref_t, matched, matched_disp_t)
        dist, idx = _nearest_obs_gpu(expected_t, obs_t)
        if dist <= tau_nn_um:
            matched[sub_idx] = True
            match_obs_idx[sub_idx] = idx
            matched_disp_t[sub_idx] = obs_t[idx] - ref_t[sub_idx]
            consecutive_fails = 0
        else:
            consecutive_fails += 1
            if consecutive_fails >= k_fail:
                break

    n_matched = int(np.sum(matched))
    if n_matched < 4:
        return {
            "coeffs": np.zeros(n_terms),
            "success": False,
            "rmse": float("inf"),
            "n_matched": n_matched,
            "conflict_ratio": 1.0,
            "solver": "baseline_extrap_nn_gpu",
        }

    matched_idx_t = torch.tensor(np.where(matched)[0], dtype=torch.long, device=device)
    slopes_matched_t = matched_disp_t[matched_idx_t] / float(lenslet.focal_um)
    coeffs_recon = _reconstruct_from_subset_gpu(
        lenslet=lenslet,
        matched_mask=matched,
        slopes_matched_t=slopes_matched_t,
        max_order=max_order,
        lambda_reg=lambda_reg,
        grid_size=grid_size,
        device=device,
    )

    conflict_ratio = _compute_conflict_ratio(match_obs_idx, matched)
    success = conflict_ratio <= conflict_max
    rmse = float(np.sqrt(np.mean(coeffs_recon[1:] ** 2)))
    return {
        "coeffs": coeffs_recon,
        "success": success,
        "rmse": rmse,
        "n_matched": n_matched,
        "conflict_ratio": conflict_ratio,
        "solver": "baseline_extrap_nn_gpu",
    }


def baseline_reconstruct_auto(
    observed: np.ndarray,
    lenslet: LensletArray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Auto-select baseline solver: GPU first, then CPU fallback."""
    bl_cfg = cfg.get("baseline", {})
    use_gpu = bool(bl_cfg.get("use_gpu", False))
    require_gpu = bool(bl_cfg.get("require_gpu", False))
    fallback_cpu = bool(bl_cfg.get("gpu_fallback_to_cpu", True))

    if use_gpu:
        if _cuda_available():
            try:
                return baseline_reconstruct_gpu(observed, lenslet, cfg)
            except Exception:
                if not fallback_cpu:
                    raise
        elif require_gpu:
            raise RuntimeError("baseline.require_gpu=true but CUDA is unavailable")

    out = baseline_reconstruct(observed, lenslet, cfg)
    out["solver"] = "baseline_extrap_nn_cpu"
    return out
