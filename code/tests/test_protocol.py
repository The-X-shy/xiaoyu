"""Tests for evaluation protocol."""

import numpy as np
import pytest
from src.eval.protocol import evaluate_single_sample, evaluate_method_at_pv


class TestEvaluateSingleSample:
    def setup_method(self):
        self.cfg = {
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
            },
            "asm": {
                "lambda_reg": 1e-3,
                "n_starts": 5,
                "n_icp_iter": 10,
                "convergence_tol": 1e-6,
            },
            "evaluation": {
                "rmse_max_lambda": 0.15,
                "success_rate_min": 0.95,
                "required_range_gain": 14.0,
                "n_repeats": 20,
            },
        }

    def test_baseline_sample(self):
        out = evaluate_single_sample(
            cfg=self.cfg,
            method="baseline",
            pv=0.5,
            seed=42,
            missing_ratio=0.0,
        )
        assert isinstance(out, dict)
        for key in ["method", "pv_level", "seed", "success", "rmse", "missing_ratio"]:
            assert key in out
        assert out["method"] == "baseline"
        assert out["pv_level"] == 0.5

    def test_asm_sample(self):
        out = evaluate_single_sample(
            cfg=self.cfg,
            method="asm",
            pv=0.5,
            seed=42,
            missing_ratio=0.0,
        )
        assert out["method"] == "asm"
        assert "rmse" in out

    def test_evaluate_method_at_pv(self):
        results = evaluate_method_at_pv(
            cfg=self.cfg,
            method="baseline",
            pv=0.5,
            base_seed=42,
            n_repeats=3,
            missing_ratio=0.0,
        )
        assert len(results) == 3
        assert all(r["pv_level"] == 0.5 for r in results)
