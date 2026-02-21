"""Tests for Zernike polynomial module."""

import numpy as np
import pytest
from src.recon.zernike import (
    noll_to_nm,
    zernike_radial,
    zernike_polynomial,
    zernike_wavefront,
    num_zernike_terms,
)


class TestNollIndex:
    def test_first_terms(self):
        assert noll_to_nm(1) == (0, 0)  # piston
        assert noll_to_nm(2) == (1, 1)  # tilt
        assert noll_to_nm(3) == (1, -1)  # tip
        assert noll_to_nm(4) == (2, 0)  # defocus
        assert noll_to_nm(5) == (2, -2)  # astigmatism
        assert noll_to_nm(6) == (2, 2)  # astigmatism

    def test_order_3(self):
        assert noll_to_nm(7) == (3, -1)
        assert noll_to_nm(8) == (3, 1)
        assert noll_to_nm(9) == (3, -3)
        assert noll_to_nm(10) == (3, 3)

    def test_order_4(self):
        assert noll_to_nm(11) == (4, 0)

    def test_invalid(self):
        with pytest.raises(ValueError):
            noll_to_nm(0)


class TestNumTerms:
    def test_values(self):
        assert num_zernike_terms(0) == 1
        assert num_zernike_terms(1) == 3
        assert num_zernike_terms(2) == 6
        assert num_zernike_terms(3) == 10
        assert num_zernike_terms(5) == 21
        assert num_zernike_terms(15) == 136


class TestRadial:
    def test_R00(self):
        r = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(zernike_radial(0, 0, r), [1, 1, 1])

    def test_R11(self):
        r = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(zernike_radial(1, 1, r), r)

    def test_R20(self):
        r = np.array([0.0, 0.5, 1.0])
        expected = 2 * r**2 - 1
        np.testing.assert_allclose(zernike_radial(2, 0, r), expected)


class TestWavefront:
    def test_piston(self):
        coeffs = np.array([1.0])
        W = zernike_wavefront(coeffs, 64, max_order=0)
        assert W.shape == (64, 64)
        # Center should be ~1.0 (with normalization sqrt(1) = 1)
        cx = 32
        assert abs(W[cx, cx] - 1.0) < 0.1

    def test_outside_is_zero(self):
        coeffs = np.array([1.0])
        W = zernike_wavefront(coeffs, 64)
        assert W[0, 0] == 0.0

    def test_defocus_symmetry(self):
        coeffs = np.zeros(6)
        coeffs[3] = 1.0  # defocus (j=4)
        # Use odd grid so center pixel is exactly at origin
        W = zernike_wavefront(coeffs, 129, max_order=2)
        cx = 64
        d = 10
        vals = [W[cx, cx + d], W[cx, cx - d], W[cx + d, cx], W[cx - d, cx]]
        np.testing.assert_allclose(vals, vals[0], atol=1e-6)
