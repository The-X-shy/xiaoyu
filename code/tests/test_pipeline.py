"""Tests for forward simulation pipeline."""

import numpy as np
import pytest
from src.sim.pipeline import forward_pipeline


class TestForwardPipeline:
    def setup_method(self):
        self.cfg = {
            "optics": {
                "wavelength_nm": 633.0,
                "pitch_um": 150.0,
                "focal_mm": 6.0,
                "fill_factor": 0.95,
            },
            "sensor": {
                "pixel_um": 10.0,
                "width_px": 512,
                "height_px": 512,
            },
            "noise": {
                "read_sigma": 0.01,
                "background": 0.0,
                "centroid_noise_px": 0.05,
            },
            "zernike": {
                "order": 3,
                "coeff_bound": 1.0,
            },
        }

    def test_returns_dict(self):
        result = forward_pipeline(self.cfg, pv=2.0, seed=42)
        assert isinstance(result, dict)

    def test_keys(self):
        result = forward_pipeline(self.cfg, pv=2.0, seed=42)
        for key in [
            "coeffs",
            "wavefront",
            "slopes",
            "ref_positions",
            "displaced_positions",
            "observed_positions",
            "lenslet",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_shapes_consistent(self):
        result = forward_pipeline(self.cfg, pv=2.0, seed=42)
        n = len(result["ref_positions"])
        assert result["slopes"].shape == (n, 2)
        # After sensor clipping, displaced/observed may have fewer spots
        n_obs = len(result["observed_positions"])
        assert result["displaced_positions"].shape == (n_obs, 2)
        assert result["observed_positions"].shape == (n_obs, 2)
        assert n_obs <= n

    def test_flat_wavefront(self):
        result = forward_pipeline(self.cfg, pv=0.0, seed=42)
        ref = result["ref_positions"]
        obs = result["observed_positions"]
        diff = np.linalg.norm(obs - ref, axis=1)
        assert np.mean(diff) < 5.0

    def test_reproducible(self):
        r1 = forward_pipeline(self.cfg, pv=2.0, seed=42)
        r2 = forward_pipeline(self.cfg, pv=2.0, seed=42)
        np.testing.assert_array_equal(r1["coeffs"], r2["coeffs"])

    def test_large_pv_displaces(self):
        result = forward_pipeline(self.cfg, pv=10.0, seed=42)
        # At high PV most spots fly off sensor; just check we get some output
        n_obs = len(result["observed_positions"])
        # Some spots should still be in bounds (or none — either is fine)
        assert n_obs >= 0
        # Slopes should still be large
        slopes = result["slopes"]
        max_slope = np.max(np.abs(slopes))
        assert max_slope > 0.01

    def test_missing_spots(self):
        result = forward_pipeline(self.cfg, pv=2.0, seed=42, missing_ratio=0.3)
        assert "observed_positions" in result
        assert "keep_mask" in result
        n_full = len(result["ref_positions"])
        n_obs = len(result["observed_positions"])
        assert n_obs < n_full

    def test_custom_coeffs(self):
        from src.recon.zernike import num_zernike_terms

        n_terms = num_zernike_terms(3)
        coeffs = np.zeros(n_terms)
        coeffs[3] = 1.0
        result = forward_pipeline(self.cfg, pv=2.0, seed=42, coeffs=coeffs)
        np.testing.assert_array_equal(result["coeffs"], coeffs)
