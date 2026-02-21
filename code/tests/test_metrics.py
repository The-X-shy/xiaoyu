"""Tests for evaluation metrics."""

import numpy as np
import pytest
from src.eval.metrics import (
    rmse_coeffs,
    rmse_wavefront,
    success_rate,
    compute_dynamic_range,
)


class TestRMSECoeffs:
    def test_identical(self):
        c = np.array([0.0, 1.0, -0.5])
        assert rmse_coeffs(c, c) == 0.0

    def test_known_value(self):
        c_true = np.array([0.0, 1.0, 0.0])
        c_recon = np.array([0.0, 0.0, 0.0])
        expected = np.sqrt(1.0 / 3)
        np.testing.assert_allclose(rmse_coeffs(c_true, c_recon), expected)

    def test_exclude_piston(self):
        c_true = np.array([5.0, 1.0, 0.0])
        c_recon = np.array([0.0, 1.0, 0.0])
        assert rmse_coeffs(c_true, c_recon, exclude_piston=True) == 0.0


class TestRMSEWavefront:
    def test_identical(self):
        W = np.ones((64, 64))
        assert rmse_wavefront(W, W, grid_size=64) == 0.0

    def test_positive(self):
        W1 = np.zeros((64, 64))
        W2 = np.ones((64, 64))
        r = rmse_wavefront(W1, W2, grid_size=64)
        assert r > 0.0


class TestSuccessRate:
    def test_all_success(self):
        results = [{"success": True}] * 10
        assert success_rate(results) == 1.0

    def test_half_success(self):
        results = [{"success": True}] * 5 + [{"success": False}] * 5
        assert success_rate(results) == 0.5


class TestDynamicRange:
    def test_basic(self):
        records = []
        for pv in np.arange(0.5, 10.5, 0.5):
            for trial in range(5):
                success = pv <= 5.0
                rmse = 0.05 if success else 0.5
                records.append({"pv_level": pv, "success": success, "rmse": rmse})
        dr = compute_dynamic_range(records, rmse_max=0.15, sr_min=0.95)
        assert abs(dr - 5.0) < 0.6
