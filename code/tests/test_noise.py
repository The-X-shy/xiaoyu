"""Tests for noise module."""

import numpy as np
from src.sim.noise import apply_noise, apply_missing_spots


class TestApplyNoise:
    def test_no_noise(self):
        pos = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = apply_noise(pos, read_sigma=0.0, background=0.0, seed=42)
        np.testing.assert_allclose(result, pos)

    def test_noise_adds_spread(self):
        pos = np.ones((50, 2))
        result = apply_noise(pos, read_sigma=0.1, background=0.0, seed=42)
        assert np.std(result) > 0.01

    def test_reproducible(self):
        pos = np.ones((10, 2))
        r1 = apply_noise(pos, 0.1, 0.01, seed=42)
        r2 = apply_noise(pos, 0.1, 0.01, seed=42)
        np.testing.assert_array_equal(r1, r2)


class TestMissingSpots:
    def test_no_missing(self):
        pos = np.ones((10, 2))
        result, mask = apply_missing_spots(pos, ratio=0.0, seed=42)
        assert len(result) == 10
        assert np.all(mask)

    def test_half_missing(self):
        pos = np.ones((100, 2))
        result, mask = apply_missing_spots(pos, ratio=0.5, seed=42)
        assert len(result) == 50

    def test_all_missing(self):
        pos = np.ones((10, 2))
        result, mask = apply_missing_spots(pos, ratio=1.0, seed=42)
        assert len(result) == 0
