"""Tests for least-squares wavefront reconstruction."""

import numpy as np
import pytest
from src.recon.least_squares import (
    build_zernike_slope_matrix,
    reconstruct_least_squares,
)


class TestBuildMatrix:
    def test_shape(self):
        from src.sim.lenslet import LensletArray

        la = LensletArray(150.0, 6.0, 0.95, 512, 512, 10.0)
        n = la.n_subapertures
        G = build_zernike_slope_matrix(la, max_order=3, grid_size=128)
        from src.recon.zernike import num_zernike_terms

        n_terms = num_zernike_terms(3)
        assert G.shape == (2 * n, n_terms)


class TestReconstruct:
    def test_identity_recovery(self):
        from src.sim.lenslet import LensletArray
        from src.sim.wavefront import (
            random_zernike_coeffs,
            scale_coeffs_to_pv,
            generate_wavefront,
        )

        cfg = {
            "optics": {
                "wavelength_nm": 633.0,
                "pitch_um": 150.0,
                "focal_mm": 6.0,
                "fill_factor": 0.95,
            },
            "sensor": {"pixel_um": 10.0, "width_px": 512, "height_px": 512},
            "noise": {"read_sigma": 0.0, "background": 0.0, "centroid_noise_px": 0.0},
            "zernike": {"order": 3, "coeff_bound": 1.0},
        }
        # Use low PV so all spots stay in bounds
        la = LensletArray(150.0, 6.0, 0.95, 512, 512, 10.0)
        coeffs = random_zernike_coeffs(max_order=3, coeff_bound=1.0, seed=42)
        coeffs = scale_coeffs_to_pv(coeffs, target_pv=0.1, grid_size=128)

        W = generate_wavefront(coeffs, 128)
        slopes = la.compute_slopes(W, 128)

        c_recon = reconstruct_least_squares(la, slopes, max_order=3, grid_size=128)

        np.testing.assert_allclose(c_recon[1:], coeffs[1:], atol=0.3)

    def test_zero_slopes(self):
        from src.sim.lenslet import LensletArray

        la = LensletArray(150.0, 6.0, 0.95, 512, 512, 10.0)
        slopes = np.zeros((la.n_subapertures, 2))
        c = reconstruct_least_squares(la, slopes, max_order=3, grid_size=128)
        np.testing.assert_allclose(c, 0.0, atol=1e-10)
