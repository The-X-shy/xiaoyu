#!/usr/bin/env python3
"""Test NN warm-start -> Chamfer Adam refinement end-to-end.

Tests whether NN predictions can be refined by Adam to achieve RMSE < 0.15.
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
import time

from src.sim.pipeline import forward_pipeline
from src.sim.lenslet import LensletArray
from src.recon.nn_warmstart import NNWarmStarter, extract_features
from src.recon.chamfer_optimizer import ChamferOptimizer


def main():
    cfg = yaml.safe_load(open("configs/base_no_oracle.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sensor_w = cfg["sensor"]["width_px"] * cfg["sensor"]["pixel_um"]
    sensor_h = cfg["sensor"]["height_px"] * cfg["sensor"]["pixel_um"]

    # Load NN model
    nn_starter = NNWarmStarter(
        model_path="models/nn_warmstart.pt",
        sensor_w_um=sensor_w,
        sensor_h_um=sensor_h,
        n_terms=10,
        device=device,
    )
    print("NN model loaded")

    # Override chamfer config for faster Adam-only refinement
    # We'll skip the random search phase and just use NN predictions
    cfg_test = dict(cfg)

    pv_levels = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0]
    n_per_pv = 20

    print(
        f"\n{'PV':>5s} | {'NN_RMSE':>8s} | {'Adam_RMSE':>9s} | {'Improved':>8s} | {'<0.15':>6s} | {'<0.20':>6s} | {'time':>6s}"
    )
    print("-" * 70)

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

            # Step 1: NN prediction
            nn_coeffs = nn_starter.predict(observed)
            nn_rmse = np.sqrt(np.mean((nn_coeffs - true_coeffs) ** 2))

            # Step 2: Chamfer Adam refinement from NN prediction
            # Use NN prediction + ensemble as init_coeffs
            nn_ensemble = nn_starter.predict_ensemble(observed, n_augment=8)

            chamfer_opt = ChamferOptimizer(observed, la, cfg, device=device)

            # Override to skip random sampling, just refine from NN predictions
            chamfer_opt.n_sample = 0  # no random search
            chamfer_opt.n_refine = len(nn_ensemble)
            chamfer_opt.refine_iter = 500
            chamfer_opt.refine_lr = 0.002  # careful lr within basin

            chamfer_result = chamfer_opt.run(
                seed=seed + 7000,
                init_coeffs=nn_ensemble,
            )

            adam_coeffs = chamfer_result["coeffs"]
            adam_rmse = np.sqrt(np.mean((adam_coeffs - true_coeffs) ** 2))

            dt = time.time() - t0

            nn_rmses.append(nn_rmse)
            adam_rmses.append(adam_rmse)
            times.append(dt)

        if not nn_rmses:
            print(f"{pv:5.1f} | {'N/A':>8s} |")
            continue

        nn_rmses = np.array(nn_rmses)
        adam_rmses = np.array(adam_rmses)

        improved = np.mean(adam_rmses < nn_rmses) * 100
        pct_015 = np.mean(adam_rmses < 0.15) * 100
        pct_020 = np.mean(adam_rmses < 0.20) * 100

        print(
            f"{pv:5.1f} | {np.mean(nn_rmses):8.4f} | {np.mean(adam_rmses):9.4f} | "
            f"{improved:7.1f}% | {pct_015:5.1f}% | {pct_020:5.1f}% | "
            f"{np.mean(times):5.1f}s"
        )

    # Also test: what happens if we combine NN + random search (full Chamfer)?
    print("\n\n=== Full Chamfer (NN init + 30k random) ===")
    print(
        f"{'PV':>5s} | {'Full_RMSE':>9s} | {'<0.15':>6s} | {'<0.20':>6s} | {'time':>6s}"
    )
    print("-" * 50)

    for pv in [5.0, 8.0, 10.0, 15.0]:
        rmses = []
        times = []
        for i in range(10):
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

            # NN predictions as init
            nn_ensemble = nn_starter.predict_ensemble(observed, n_augment=16)

            chamfer_opt = ChamferOptimizer(observed, la, cfg, device=device)
            chamfer_opt.refine_lr = 0.002  # safer lr
            chamfer_opt.refine_iter = 500

            chamfer_result = chamfer_opt.run(
                seed=seed + 7000,
                init_coeffs=nn_ensemble,
            )
            full_coeffs = chamfer_result["coeffs"]
            rmse = np.sqrt(np.mean((full_coeffs - true_coeffs) ** 2))
            dt = time.time() - t0

            rmses.append(rmse)
            times.append(dt)

        if rmses:
            rmses = np.array(rmses)
            print(
                f"{pv:5.1f} | {np.mean(rmses):9.4f} | "
                f"{np.mean(rmses < 0.15) * 100:5.1f}% | "
                f"{np.mean(rmses < 0.20) * 100:5.1f}% | "
                f"{np.mean(times):5.1f}s"
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
