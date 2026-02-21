"""Tests for ASM reconstructor."""

import numpy as np
import pytest
from src.recon.asm_reconstructor import asm_reconstruct


class TestASMReconstruct:
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
            "asm": {
                "lambda_reg": 1e-3,
                "n_starts": 5,  # Small for test speed
                "n_icp_iter": 10,  # Small for test speed
                "convergence_tol": 1e-6,
            },
        }

    def test_returns_dict(self):
        from src.sim.pipeline import forward_pipeline

        result = forward_pipeline(self.cfg, pv=0.5, seed=42)
        out = asm_reconstruct(
            result["observed_positions"],
            result["lenslet"],
            self.cfg,
            seed=42,
        )
        assert isinstance(out, dict)
        for key in ["coeffs", "success", "objective_value", "n_iterations"]:
            assert key in out

    def test_low_pv_succeeds(self):
        from src.sim.pipeline import forward_pipeline

        result = forward_pipeline(self.cfg, pv=0.5, seed=42)
        out = asm_reconstruct(
            result["observed_positions"],
            result["lenslet"],
            self.cfg,
            seed=42,
        )
        assert out["success"] is True

    def test_coeffs_shape(self):
        from src.sim.pipeline import forward_pipeline
        from src.recon.zernike import num_zernike_terms

        result = forward_pipeline(self.cfg, pv=0.5, seed=42)
        out = asm_reconstruct(
            result["observed_positions"],
            result["lenslet"],
            self.cfg,
            seed=42,
        )
        expected = num_zernike_terms(self.cfg["zernike"]["order"])
        assert len(out["coeffs"]) == expected

    def test_reproducible(self):
        from src.sim.pipeline import forward_pipeline

        result = forward_pipeline(self.cfg, pv=0.5, seed=42)
        out1 = asm_reconstruct(
            result["observed_positions"],
            result["lenslet"],
            self.cfg,
            seed=42,
        )
        out2 = asm_reconstruct(
            result["observed_positions"],
            result["lenslet"],
            self.cfg,
            seed=42,
        )
        np.testing.assert_array_equal(out1["coeffs"], out2["coeffs"])
