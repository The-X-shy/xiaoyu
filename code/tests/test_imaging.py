"""Tests for imaging simulation module."""

import numpy as np
from src.sim.imaging import simulate_spots, extract_centroids


class TestSimulateSpots:
    def test_zero_displacement(self):
        ref = np.array([[100.0, 100.0]])
        disp = np.array([[0.0, 0.0]])
        np.testing.assert_allclose(simulate_spots(ref, disp), ref)

    def test_displacement(self):
        ref = np.array([[100.0, 100.0]])
        disp = np.array([[5.0, -3.0]])
        expected = np.array([[105.0, 97.0]])
        np.testing.assert_allclose(simulate_spots(ref, disp), expected)


class TestExtractCentroids:
    def test_no_noise(self):
        pos = np.array([[100.0, 200.0], [300.0, 400.0]])
        result = extract_centroids(pos, noise_px=0.0, pixel_um=10.0, seed=42)
        np.testing.assert_allclose(result, pos)

    def test_noise_adds_variation(self):
        pos = np.array([[100.0, 200.0]] * 100)
        result = extract_centroids(pos, noise_px=1.0, pixel_um=10.0, seed=42)
        assert np.std(result - pos) > 0.1
