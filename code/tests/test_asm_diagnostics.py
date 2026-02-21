"""Diagnostics and regression tests for ASM robustness settings."""

from __future__ import annotations

import numpy as np

from src.eval.metrics import rmse_coeffs
from src.recon.asm_reconstructor import _icp_single
from src.recon.least_squares import build_zernike_slope_matrix, reconstruct_least_squares
from src.recon.zernike import num_zernike_terms
from src.sim.lenslet import LensletArray
from src.sim.pipeline import forward_pipeline


def _base_cfg():
    return {
        "optics": {
            "wavelength_nm": 633.0,
            "pitch_um": 150.0,
            "focal_mm": 6.0,
            "fill_factor": 0.95,
        },
        "sensor": {"pixel_um": 10.0, "width_px": 512, "height_px": 512},
        "noise": {"read_sigma": 0.0, "background": 0.0, "centroid_noise_px": 0.0},
        "zernike": {"order": 3, "coeff_bound": 1.0},
        "baseline": {
            "tau_nn_px": 10.0,
            "k_fail": 5,
            "conflict_ratio_max": 1.0,
            "rmse_max": 0.15,
        },
        "asm": {
            "lambda_reg": 1e-3,
            "grid_size": 128,
            "n_starts": 5,
            "n_icp_iter": 10,
            "convergence_tol": 1e-6,
            "max_match_dist_factor": 0.35,
            "min_match_ratio": 0.2,
            "trim_ratio": 0.9,
            "allow_forward_fallback": False,
        },
        "evaluation": {"rmse_max_lambda": 0.15},
    }


def test_perfect_matching_least_squares_lower_bound():
    """Noiseless perfect correspondence should yield low reconstruction error."""
    cfg = _base_cfg()
    sim = forward_pipeline(cfg, pv=0.5, seed=42, missing_ratio=0.0)
    coeffs_true = sim["coeffs"]
    coeffs_recon = reconstruct_least_squares(
        sim["lenslet"],
        sim["slopes"],
        max_order=cfg["zernike"]["order"],
        grid_size=cfg["asm"]["grid_size"],
    )
    rmse = rmse_coeffs(coeffs_true, coeffs_recon, exclude_piston=True)
    assert rmse < 0.05


def test_match_distance_gate_blocks_far_matches():
    """Distance gate should reject unrealistically far nearest-neighbor matches."""
    cfg = _base_cfg()
    lenslet = LensletArray(
        pitch_um=150.0,
        focal_mm=6.0,
        fill_factor=0.95,
        sensor_width_px=256,
        sensor_height_px=256,
        pixel_um=10.0,
    )
    ref = lenslet.reference_positions()
    observed = ref + np.array([5000.0, 5000.0])  # force large matching distances

    n_terms = num_zernike_terms(cfg["zernike"]["order"])
    G = build_zernike_slope_matrix(lenslet, cfg["zernike"]["order"], grid_size=64)

    out = _icp_single(
        c0=np.zeros(n_terms),
        observed=observed,
        ref=ref,
        G=G,
        focal_um=lenslet.focal_um,
        n_sub=lenslet.n_subapertures,
        sensor_w=lenslet.sensor_width_um,
        sensor_h=lenslet.sensor_height_um,
        lambda_reg=cfg["asm"]["lambda_reg"],
        n_icp_iter=5,
        convergence_tol=1e-6,
        max_match_dist_um=0.1 * lenslet.pitch_um,
        min_match_ratio=0.2,
        trim_ratio=0.9,
        allow_forward_fallback=False,
    )

    assert out["n_matched"] < n_terms
    assert not np.isfinite(out["residual_trimmed"])
