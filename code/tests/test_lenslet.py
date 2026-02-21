"""Tests for lenslet array module."""

import numpy as np
import pytest
from src.sim.lenslet import LensletArray


class TestLensletArray:
    def setup_method(self):
        self.la = LensletArray(
            pitch_um=150.0,
            focal_mm=6.0,
            fill_factor=0.95,
            sensor_width_px=512,
            sensor_height_px=512,
            pixel_um=10.0,
        )

    def test_grid_creation(self):
        ref = self.la.reference_positions()
        assert ref.ndim == 2
        assert ref.shape[1] == 2
        assert len(ref) > 0

    def test_reference_centered(self):
        ref = self.la.reference_positions()
        center_x = self.la.sensor_width_um / 2
        center_y = self.la.sensor_height_um / 2
        np.testing.assert_allclose(np.mean(ref[:, 0]), center_x, atol=self.la.pitch_um)
        np.testing.assert_allclose(np.mean(ref[:, 1]), center_y, atol=self.la.pitch_um)

    def test_subaperture_count(self):
        n = self.la.n_subapertures
        # 512*10=5120um, pitch=150, ~34 per side, circular ~900
        assert 10 < n < 1200

    def test_flat_wavefront_zero_slopes(self):
        W = np.zeros((64, 64))
        slopes = self.la.compute_slopes(W, 64)
        assert slopes.shape[1] == 2
        np.testing.assert_allclose(slopes, 0.0, atol=1e-10)

    def test_displacement_formula(self):
        slopes = np.array([[0.001, -0.002], [0.0, 0.003]])
        disp = self.la.slopes_to_displacements(slopes)
        expected = slopes * self.la.focal_um
        np.testing.assert_allclose(disp, expected)

    def test_check_bounds(self):
        pos = np.array([[100, 100], [-10, 100], [100, 6000]])
        bounds = self.la.check_bounds(pos)
        assert bounds[0] == True
        assert bounds[1] == False
        assert bounds[2] == False
