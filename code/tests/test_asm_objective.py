"""Tests for ASM objective function."""

import numpy as np
import pytest
from src.recon.asm_objective import (
    asm_objective,
    hausdorff_mean,
    duplicate_penalty,
    out_of_bounds_penalty,
)


class TestHausdorffMean:
    def test_identical_sets(self):
        A = np.array([[0.0, 0.0], [1.0, 1.0]])
        d = hausdorff_mean(A, A)
        assert d == 0.0

    def test_positive_distance(self):
        A = np.array([[0.0, 0.0]])
        B = np.array([[3.0, 4.0]])
        d = hausdorff_mean(A, B)
        assert abs(d - 5.0) < 1e-10

    def test_symmetric(self):
        A = np.array([[0.0, 0.0], [1.0, 0.0]])
        B = np.array([[0.5, 0.5], [1.5, 0.5]])
        np.testing.assert_allclose(
            hausdorff_mean(A, B), hausdorff_mean(B, A), atol=1e-10
        )


class TestDuplicatePenalty:
    def test_no_duplicates(self):
        E = np.array([[0.0, 0.0], [100.0, 100.0]])
        G = np.array([[0.0, 0.0], [100.0, 100.0]])
        p = duplicate_penalty(E, G)
        assert p == 0.0

    def test_with_duplicates(self):
        E = np.array([[0.0, 0.0], [0.1, 0.1]])
        G = np.array([[0.0, 0.0], [100.0, 100.0]])
        p = duplicate_penalty(E, G)
        assert p > 0.0


class TestOutOfBoundsPenalty:
    def test_all_in_bounds(self):
        E = np.array([[100.0, 100.0]])
        p = out_of_bounds_penalty(E, 512.0, 512.0)
        assert p == 0.0

    def test_some_out(self):
        E = np.array([[100.0, 100.0], [-10.0, 100.0], [100.0, 600.0]])
        p = out_of_bounds_penalty(E, 512.0, 512.0)
        assert p > 0.0


class TestAsmObjective:
    def test_perfect_match_is_small(self):
        from src.sim.pipeline import forward_pipeline

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
        result = forward_pipeline(cfg, pv=0.01, seed=42)
        c_true = result["coeffs"]
        observed = result["observed_positions"]
        la = result["lenslet"]
        J = asm_objective(c_true, observed, la, cfg)
        assert J < 1.0
