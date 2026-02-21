"""Particle Swarm Optimization for ASM reconstruction."""

from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Any, Optional, Tuple, Union


def pso_optimize(
    objective: Callable[[np.ndarray], float],
    dim: int,
    bounds: Union[Tuple[float, float], np.ndarray],
    n_particles: int = 100,
    n_iter: int = 200,
    seed: int = 42,
    w_start: float = 0.9,
    w_end: float = 0.4,
    c1: float = 1.8,
    c2: float = 1.8,
    patience: int = 20,
    eps_obj: float = 1e-6,
) -> Dict[str, Any]:
    """Run PSO optimization.

    Args:
        objective: Function f(x) -> scalar to minimize.
        dim: Dimensionality of search space.
        bounds: (lower, upper) bounds for all dimensions, or (dim, 2) array.
        n_particles: Number of particles.
        n_iter: Maximum iterations.
        seed: Random seed.
        w_start: Initial inertia weight.
        w_end: Final inertia weight.
        c1: Cognitive learning factor.
        c2: Social learning factor.
        patience: Early-stop patience.
        eps_obj: Minimum improvement threshold.

    Returns:
        Dict with: best_position, best_value, history, n_iterations.
    """
    rng = np.random.RandomState(seed)

    if isinstance(bounds, tuple) and len(bounds) == 2:
        lb = np.full(dim, bounds[0])
        ub = np.full(dim, bounds[1])
    else:
        bounds = np.asarray(bounds)
        lb = bounds[:, 0]
        ub = bounds[:, 1]

    positions = rng.uniform(lb, ub, size=(n_particles, dim))
    velocities = rng.uniform(-(ub - lb) * 0.1, (ub - lb) * 0.1, size=(n_particles, dim))

    fitness = np.array([objective(p) for p in positions])

    pbest_pos = positions.copy()
    pbest_val = fitness.copy()

    gbest_idx = np.argmin(fitness)
    gbest_pos = positions[gbest_idx].copy()
    gbest_val = fitness[gbest_idx]

    history = [gbest_val]
    no_improve_count = 0

    for t in range(1, n_iter):
        w = w_start - (w_start - w_end) * t / n_iter

        r1 = rng.uniform(0, 1, size=(n_particles, dim))
        r2 = rng.uniform(0, 1, size=(n_particles, dim))

        velocities = (
            w * velocities
            + c1 * r1 * (pbest_pos - positions)
            + c2 * r2 * (gbest_pos - positions)
        )

        positions = positions + velocities
        positions = np.clip(positions, lb, ub)

        fitness = np.array([objective(p) for p in positions])

        improved = fitness < pbest_val
        pbest_pos[improved] = positions[improved]
        pbest_val[improved] = fitness[improved]

        min_idx = np.argmin(pbest_val)
        if pbest_val[min_idx] < gbest_val - eps_obj:
            gbest_val = pbest_val[min_idx]
            gbest_pos = pbest_pos[min_idx].copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        history.append(gbest_val)

        if no_improve_count >= patience:
            break

    return {
        "best_position": gbest_pos,
        "best_value": float(gbest_val),
        "history": history,
        "n_iterations": len(history),
    }
