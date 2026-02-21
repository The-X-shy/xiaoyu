"""Tests for wavefront generation module."""

import numpy as np
import pytest
from src.sim.wavefront import (
    random_zernike_coeffs,
    scale_coeffs_to_pv,
    generate_wavefront,
)
from src.recon.zernike import num_zernike_terms


class TestRandomCoeffs:
    def test_shape(self):
        c = random_zernike_coeffs(5, 1.0, seed=42)
        assert len(c) == num_zernike_terms(5)

    def test_piston_zero(self):
        c = random_zernike_coeffs(5, 1.0, seed=42)
        assert c[0] == 0.0

    def test_bound(self):
        c = random_zernike_coeffs(10, 0.5, seed=42)
        assert np.all(np.abs(c) <= 0.5)

    def test_reproducible(self):
        c1 = random_zernike_coeffs(5, 1.0, seed=123)
        c2 = random_zernike_coeffs(5, 1.0, seed=123)
        np.testing.assert_array_equal(c1, c2)

    def test_different_seeds(self):
        c1 = random_zernike_coeffs(5, 1.0, seed=1)
        c2 = random_zernike_coeffs(5, 1.0, seed=2)
        assert not np.array_equal(c1, c2)


class TestScalePV:
    def test_scaling(self):
        c = random_zernike_coeffs(3, 1.0, seed=42)
        scaled = scale_coeffs_to_pv(c, target_pv=5.0, grid_size=128)
        from src.recon.zernike import zernike_wavefront

        W = zernike_wavefront(scaled, 128)
        x = np.linspace(-1, 1, 128)
        X, Y = np.meshgrid(x, x)
        mask = np.sqrt(X**2 + Y**2) <= 1.0
        pv = np.max(W[mask]) - np.min(W[mask])
        np.testing.assert_allclose(pv, 5.0, atol=0.1)


class TestGenerate:
    def test_basic(self):
        coeffs = np.zeros(6)
        coeffs[3] = 1.0
        W = generate_wavefront(coeffs, 64, max_order=2)
        assert W.shape == (64, 64)
        assert np.any(W != 0.0)
