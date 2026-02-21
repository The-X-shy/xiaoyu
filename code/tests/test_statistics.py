"""Tests for evaluation statistics utilities."""

import numpy as np
import pytest
from src.eval.statistics import compute_summary, confidence_interval_95


class TestComputeSummary:
    def test_basic(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        s = compute_summary(values)
        assert "mean" in s
        assert "std" in s
        assert "ci_low" in s
        assert "ci_high" in s
        np.testing.assert_allclose(s["mean"], 3.0)

    def test_single_value(self):
        s = compute_summary([5.0])
        assert s["mean"] == 5.0


class TestCI95:
    def test_known(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lo, hi = confidence_interval_95(values)
        assert lo < 3.0 < hi

    def test_constant(self):
        values = np.array([5.0, 5.0, 5.0])
        lo, hi = confidence_interval_95(values)
        np.testing.assert_allclose(lo, 5.0, atol=1e-10)
        np.testing.assert_allclose(hi, 5.0, atol=1e-10)
