"""Tests for baseline GPU auto-routing and CPU fallback behavior."""

from __future__ import annotations

import pytest

from src.eval.protocol import evaluate_single_sample


def _cfg_small() -> dict:
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
            "tau_nn_px": 5.0,
            "k_fail": 3,
            "conflict_ratio_max": 0.2,
            "rmse_max": 0.15,
            "use_gpu": True,
            "gpu_batch_size": 128,
            "gpu_fallback_to_cpu": True,
        },
        "asm": {
            "lambda_reg": 1e-3,
            "n_starts": 4,
            "n_icp_iter": 8,
            "convergence_tol": 1e-6,
            "use_gpu": True,
            "use_oracle_index_hint": True,
        },
        "evaluation": {
            "rmse_max_lambda": 0.5,
            "success_rate_min": 0.95,
            "required_range_gain": 14.0,
            "n_repeats": 3,
        },
    }


def test_baseline_solver_field_exists():
    out = evaluate_single_sample(cfg=_cfg_small(), method="baseline", pv=0.5, seed=42)
    assert "solver" in out
    assert out["solver"] in {"baseline_extrap_nn_cpu", "baseline_extrap_nn_gpu"}


def test_baseline_gpu_flag_fallback_or_gpu():
    out = evaluate_single_sample(cfg=_cfg_small(), method="baseline", pv=1.0, seed=43)
    # On machines without CUDA this should be CPU fallback.
    # On CUDA machines this can be GPU.
    assert out["solver"] in {"baseline_extrap_nn_cpu", "baseline_extrap_nn_gpu"}
    assert isinstance(out["success"], bool)


def test_asm_oracle_solver_field_exists():
    out = evaluate_single_sample(cfg=_cfg_small(), method="asm", pv=1.0, seed=44)
    assert "solver" in out
    assert out["solver"] in {"asm_oracle_ls", "asm_oracle_ls_gpu", "asm_icp_gpu", "asm_icp_cpu"}
