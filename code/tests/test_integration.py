"""Integration tests: end-to-end baseline and ASM evaluation."""

import numpy as np
import pytest
from src.eval.protocol import evaluate_single_sample


class TestIntegration:
    def setup_method(self):
        self.cfg = {
            "optics": {
                "wavelength_nm": 633.0,
                "pitch_um": 150.0,
                "focal_mm": 6.0,
                "fill_factor": 0.95,
            },
            "sensor": {"pixel_um": 10.0, "width_px": 512, "height_px": 512},
            "noise": {"read_sigma": 0.01, "background": 0.0, "centroid_noise_px": 0.05},
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
                "rmse_max_lambda": 0.5,
                "success_rate_min": 0.95,
                "required_range_gain": 14.0,
                "n_repeats": 20,
            },
        }

    def test_baseline_low_pv_succeeds(self):
        out = evaluate_single_sample(cfg=self.cfg, method="baseline", pv=0.5, seed=42)
        assert out["success"] is True
        assert out["rmse"] < 0.5

    def test_asm_low_pv_succeeds(self):
        out = evaluate_single_sample(cfg=self.cfg, method="asm", pv=0.5, seed=42)
        assert out["success"] is True

    def test_both_methods_run_at_high_pv(self):
        """Both methods should run without error at moderate PV."""
        bl = evaluate_single_sample(cfg=self.cfg, method="baseline", pv=2.0, seed=42)
        asm = evaluate_single_sample(cfg=self.cfg, method="asm", pv=2.0, seed=42)
        # Verify both produce valid output structure
        assert isinstance(bl["rmse"], float) and bl["rmse"] >= 0
        assert isinstance(asm["rmse"], float) and asm["rmse"] >= 0
