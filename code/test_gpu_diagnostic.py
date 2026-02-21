#!/usr/bin/env python
"""Manual diagnostic for CPU/GPU ASM consistency.

This file is a standalone script, not a pytest test module.
"""

from __future__ import annotations

__test__ = False

import time
import numpy as np

from src.config import load_experiment_config
from src.sim.pipeline import forward_pipeline
from src.recon.asm_reconstructor import asm_reconstruct
from src.eval.metrics import rmse_coeffs


def main() -> None:
    try:
        import torch
    except ImportError:
        print("torch is not installed. Skip GPU diagnostic.")
        return

    if not torch.cuda.is_available():
        print("CUDA is not available. Skip GPU diagnostic.")
        return

    from src.recon.asm_gpu import asm_reconstruct_gpu

    cfg = load_experiment_config("configs/exp_dynamic_range_quick.yaml")
    sim = forward_pipeline(cfg, pv=1.0, seed=20260210)
    observed = sim["observed_positions"]
    lenslet = sim["lenslet"]
    true_coeffs = sim["coeffs"]

    t0 = time.time()
    cpu_result = asm_reconstruct(observed, lenslet, cfg, seed=21260210)
    cpu_time = time.time() - t0
    cpu_rmse = rmse_coeffs(true_coeffs, cpu_result["coeffs"], exclude_piston=True)

    t0 = time.time()
    gpu_result = asm_reconstruct_gpu(observed, lenslet, cfg, seed=21260210)
    gpu_time = time.time() - t0
    gpu_rmse = rmse_coeffs(true_coeffs, gpu_result["coeffs"], exclude_piston=True)

    print("=== CPU vs GPU ASM Diagnostic ===")
    print(f"N observed: {len(observed)}")
    print(
        "CPU: success={}, obj={:.4f}, rmse={:.4f}, n_matched={}, time={:.2f}s".format(
            cpu_result["success"],
            cpu_result["objective_value"],
            cpu_rmse,
            cpu_result.get("n_matched", -1),
            cpu_time,
        )
    )
    print(
        "GPU: success={}, obj={:.4f}, rmse={:.4f}, n_matched={}, time={:.2f}s".format(
            gpu_result["success"],
            gpu_result["objective_value"],
            gpu_rmse,
            gpu_result.get("n_matched", -1),
            gpu_time,
        )
    )
    print(f"Speedup: {cpu_time / max(gpu_time, 1e-9):.2f}x")


if __name__ == "__main__":
    main()
