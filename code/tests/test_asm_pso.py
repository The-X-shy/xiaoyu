"""Tests for PSO optimizer."""

import numpy as np
import pytest
from src.recon.asm_pso import pso_optimize


class TestPSOOptimize:
    def test_returns_dict(self):
        def sphere(x):
            return np.sum(x**2)

        result = pso_optimize(
            objective=sphere,
            dim=3,
            bounds=(-5.0, 5.0),
            n_particles=10,
            n_iter=20,
            seed=42,
        )
        assert isinstance(result, dict)
        assert "best_position" in result
        assert "best_value" in result
        assert "history" in result

    def test_finds_minimum_of_sphere(self):
        def sphere(x):
            return np.sum(x**2)

        result = pso_optimize(
            objective=sphere,
            dim=3,
            bounds=(-5.0, 5.0),
            n_particles=30,
            n_iter=100,
            seed=42,
        )
        np.testing.assert_allclose(result["best_position"], 0.0, atol=0.5)
        assert result["best_value"] < 1.0

    def test_respects_bounds(self):
        def sphere(x):
            return np.sum(x**2)

        result = pso_optimize(
            objective=sphere,
            dim=5,
            bounds=(-2.0, 2.0),
            n_particles=20,
            n_iter=50,
            seed=42,
        )
        assert np.all(result["best_position"] >= -2.0)
        assert np.all(result["best_position"] <= 2.0)

    def test_reproducible(self):
        def sphere(x):
            return np.sum(x**2)

        r1 = pso_optimize(sphere, 3, (-5, 5), 20, 50, seed=42)
        r2 = pso_optimize(sphere, 3, (-5, 5), 20, 50, seed=42)
        np.testing.assert_array_equal(r1["best_position"], r2["best_position"])

    def test_inertia_decay(self):
        def sphere(x):
            return np.sum(x**2)

        result = pso_optimize(
            objective=sphere,
            dim=3,
            bounds=(-5.0, 5.0),
            n_particles=20,
            n_iter=50,
            seed=42,
            w_start=0.9,
            w_end=0.4,
        )
        assert result["best_value"] < 1.0

    def test_early_stop(self):
        def sphere(x):
            return np.sum(x**2)

        result = pso_optimize(
            objective=sphere,
            dim=3,
            bounds=(-5.0, 5.0),
            n_particles=30,
            n_iter=500,
            seed=42,
            patience=10,
            eps_obj=1e-8,
        )
        assert len(result["history"]) < 500
