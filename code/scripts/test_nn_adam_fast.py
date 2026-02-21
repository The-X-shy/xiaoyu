#!/usr/bin/env python3
"""Fast test: NN prediction -> Adam refinement on Chamfer loss.

Lean version: fewer samples, smaller refinement budget.
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
import time

from src.sim.pipeline import forward_pipeline
from src.recon.nn_warmstart import NNWarmStarter
from src.recon.chamfer_optimizer import ChamferOptimizer


def test_nn_adam_refinement():
    cfg = yaml.safe_load(open("configs/base_no_oracle.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sensor_w = cfg["sensor"]["width_px"] * cfg["sensor"]["pixel_um"]
    sensor_h = cfg["sensor"]["height_px"] * cfg["sensor"]["pixel_um"]

    nn_starter = NNWarmStarter(
        model_path="models/nn_warmstart.pt",
        sensor_w_um=sensor_w,
        sensor_h_um=sensor_h,
        n_terms=10,
        device=device,
    )
    print("NN model loaded\n")

    pv_levels = [1.0, 3.0, 5.0, 8.0, 10.0, 15.0]
    n_per_pv = 10

    print(
        f"{'PV':>5s} | {'NN_RMSE':>8s} | {'Adam_RMSE':>9s} | {'<0.15':>6s} | {'<0.20':>6s} | {'time':>6s}"
    )
    print("-" * 60)

    for pv in pv_levels:
        nn_rmses = []
        adam_rmses = []
        times = []

        for i in range(n_per_pv):
            seed = 800000 + int(pv * 1000) + i
            try:
                result = forward_pipeline(cfg, pv=pv, seed=seed)
            except Exception:
                continue

            observed = result["observed_positions"]
            true_coeffs = result["coeffs"]
            la = result["lenslet"]
            if len(observed) < 10:
                continue

            t0 = time.time()

            # NN prediction
            nn_coeffs = nn_starter.predict(observed)
            nn_rmse = float(np.sqrt(np.mean((nn_coeffs - true_coeffs) ** 2)))

            # Fast Adam refinement using ChamferOptimizer internals
            # We skip random search and just refine from NN prediction
            chamfer_opt = ChamferOptimizer(observed, la, cfg, device=device)

            # Override config for fast refinement only
            chamfer_opt.n_sample = 0
            chamfer_opt.refine_iter = 300
            chamfer_opt.refine_lr = 0.002
            chamfer_opt.sample_topk = 1
            chamfer_opt.n_refine = 1
            chamfer_opt.refine_obs_k = min(512, len(observed))
            chamfer_opt.pred_chunk = 1024  # smaller chunks = faster

            chamfer_result = chamfer_opt.run(
                seed=seed + 7000,
                init_coeffs=[nn_coeffs],
            )
            adam_coeffs = chamfer_result["coeffs"]
            adam_rmse = float(np.sqrt(np.mean((adam_coeffs - true_coeffs) ** 2)))

            dt = time.time() - t0
            nn_rmses.append(nn_rmse)
            adam_rmses.append(adam_rmse)
            times.append(dt)

        if not nn_rmses:
            print(f"{pv:5.1f} | {'N/A':>8s}")
            continue

        nn_arr = np.array(nn_rmses)
        adam_arr = np.array(adam_rmses)
        print(
            f"{pv:5.1f} | {nn_arr.mean():8.4f} | {adam_arr.mean():9.4f} | "
            f"{(adam_arr < 0.15).mean() * 100:5.1f}% | {(adam_arr < 0.20).mean() * 100:5.1f}% | "
            f"{np.mean(times):5.1f}s"
        )
        # Print individual results for debugging
        for j, (nr, ar) in enumerate(zip(nn_rmses, adam_rmses)):
            tag = "OK" if ar < 0.15 else ("CLOSE" if ar < 0.20 else "FAIL")
            print(f"       sample {j}: nn={nr:.4f} -> adam={ar:.4f} [{tag}]")

    print("\nDone!")


if __name__ == "__main__":
    test_nn_adam_refinement()
