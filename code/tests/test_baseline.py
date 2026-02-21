"""Tests for baseline extrapolation + nearest-neighbor reconstructor."""

import numpy as np
import pytest
from src.recon.baseline_extrap_nn import baseline_reconstruct


class TestBaselineReconstruct:
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
        }

    def test_returns_dict(self):
        from src.sim.pipeline import forward_pipeline

        result = forward_pipeline(self.cfg, pv=0.5, seed=42)
        out = baseline_reconstruct(
            result["observed_positions"],
            result["lenslet"],
            self.cfg,
        )
        assert isinstance(out, dict)
        for key in ["coeffs", "success", "rmse", "n_matched"]:
            assert key in out, f"Missing key: {key}"

    def test_low_pv_succeeds(self):
        from src.sim.pipeline import forward_pipeline

        result = forward_pipeline(self.cfg, pv=0.5, seed=42)
        out = baseline_reconstruct(
            result["observed_positions"],
            result["lenslet"],
            self.cfg,
        )
        assert out["success"] is True
        assert out["rmse"] < 0.5

    def test_high_pv_fails(self):
        from src.sim.pipeline import forward_pipeline

        result = forward_pipeline(self.cfg, pv=15.0, seed=42)
        out = baseline_reconstruct(
            result["observed_positions"],
            result["lenslet"],
            self.cfg,
        )
        assert out["success"] is False

    def test_reconstructed_coeffs_shape(self):
        from src.sim.pipeline import forward_pipeline
        from src.recon.zernike import num_zernike_terms

        result = forward_pipeline(self.cfg, pv=0.5, seed=42)
        out = baseline_reconstruct(
            result["observed_positions"],
            result["lenslet"],
            self.cfg,
        )
        expected_len = num_zernike_terms(self.cfg["zernike"]["order"])
        assert len(out["coeffs"]) == expected_len

    def test_no_noise_accurate(self):
        from src.sim.pipeline import forward_pipeline
        from src.eval.metrics import rmse_coeffs

        result = forward_pipeline(self.cfg, pv=0.5, seed=100)
        out = baseline_reconstruct(
            result["observed_positions"],
            result["lenslet"],
            self.cfg,
        )
        if out["success"]:
            err = rmse_coeffs(result["coeffs"], out["coeffs"], exclude_piston=True)
            assert err < 0.5
