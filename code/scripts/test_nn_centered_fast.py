#!/usr/bin/env python3
"""Fast test: NN-centered random search + Adam refinement.

Optimized version:
- 2000 samples (vs 10k) with batch=256 for cdist
- 150 Adam iterations (vs 300)
- Only top-8 chains for Adam (vs 32)
- Prints per-sample results for visibility
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


def main():
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
    print("NN model loaded", flush=True)

    pv_levels = [1.0, 3.0, 5.0, 8.0, 10.0, 15.0]
    n_per_pv = 5

    for sigma in [0.20, 0.35]:
        print(f"\n{'=' * 70}", flush=True)
        print(f"Perturbation sigma = {sigma}", flush=True)
        print(f"{'=' * 70}", flush=True)
        print(
            f"{'PV':>5s} | {'seed':>8s} | {'nSpots':>6s} | {'NN':>8s} | {'Search':>8s} | {'Adam':>8s} | {'time':>6s}",
            flush=True,
        )
        print("-" * 65, flush=True)

        for pv in pv_levels:
            nn_rmses, adam_rmses = [], []

            for i in range(n_per_pv):
                seed = 900000 + int(pv * 1000) + i
                try:
                    result = forward_pipeline(cfg, pv=pv, seed=seed)
                except Exception:
                    continue

                observed = result["observed_positions"]
                true_coeffs = result["coeffs"]
                la = result["lenslet"]
                nspots = len(observed)
                if nspots < 10:
                    continue

                t0 = time.time()

                # NN prediction
                nn_coeffs = nn_starter.predict(observed)
                nn_rmse = float(np.sqrt(np.mean((nn_coeffs - true_coeffs) ** 2)))

                # Create Chamfer optimizer
                chamfer_opt = ChamferOptimizer(observed, la, cfg, device=device)
                D = chamfer_opt.n_terms
                rng = np.random.RandomState(seed + 5000)

                # Generate perturbations around NN prediction
                n_samples = 2000
                center = nn_coeffs.copy()
                perturbations = rng.randn(n_samples, D).astype(np.float32) * sigma
                samples = np.vstack(
                    [
                        center[np.newaxis, :],
                        center[np.newaxis, :] + perturbations,
                        np.zeros((1, D), dtype=np.float32),
                    ]
                )
                samples_t = torch.tensor(samples, dtype=torch.float32, device=device)

                # Evaluate all samples using obs subsample
                obs_sub = chamfer_opt._subsample_obs(min(256, nspots), rng)
                all_obj = []
                # Use small batch to avoid OOM: B=256 * 13224 * 256 = manageable
                for start in range(0, len(samples_t), 256):
                    end = min(start + 256, len(samples_t))
                    with torch.no_grad():
                        obj = chamfer_opt._backward_chamfer_full(
                            samples_t[start:end], obs_sub
                        )
                    all_obj.append(obj)
                all_obj = torch.cat(all_obj)

                # Top-K for Adam
                topk = 8
                _, topk_idx = torch.topk(all_obj, topk, largest=False)
                top_coeffs = samples_t[topk_idx]

                # Best from search
                best_search_idx = int(all_obj.argmin().item())
                best_search = samples[best_search_idx]
                search_rmse = float(np.sqrt(np.mean((best_search - true_coeffs) ** 2)))

                # Adam refinement of top-K
                obs_r = chamfer_opt._subsample_obs(min(512, nspots), rng)
                coeffs_opt = top_coeffs.clone().detach().requires_grad_(True)
                optimizer = torch.optim.Adam([coeffs_opt], lr=0.002)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=150, eta_min=0.0002
                )

                for it in range(150):
                    optimizer.zero_grad()
                    loss = chamfer_opt._backward_chamfer_full(
                        coeffs_opt, obs_r, differentiable=True
                    )
                    loss.sum().backward()
                    torch.nn.utils.clip_grad_norm_([coeffs_opt], max_norm=5.0)
                    optimizer.step()
                    scheduler.step()

                # Select best
                with torch.no_grad():
                    obs_f = chamfer_opt._subsample_obs(min(1024, nspots), rng)
                    final_loss = chamfer_opt._backward_chamfer_full(coeffs_opt, obs_f)
                    best_idx = int(final_loss.argmin().item())
                    adam_c = coeffs_opt[best_idx].detach().cpu().numpy()

                adam_rmse = float(np.sqrt(np.mean((adam_c - true_coeffs) ** 2)))
                dt = time.time() - t0

                nn_rmses.append(nn_rmse)
                adam_rmses.append(adam_rmse)

                print(
                    f"{pv:5.1f} | {seed:8d} | {nspots:6d} | {nn_rmse:8.4f} | "
                    f"{search_rmse:8.4f} | {adam_rmse:8.4f} | {dt:5.1f}s",
                    flush=True,
                )

            if nn_rmses:
                nn_arr = np.array(nn_rmses)
                a_arr = np.array(adam_rmses)
                print(
                    f"  AVG | {'':>8s} | {'':>6s} | {nn_arr.mean():8.4f} | "
                    f"{'':>8s} | {a_arr.mean():8.4f} | "
                    f"<0.15={100 * (a_arr < 0.15).mean():.0f}% <0.20={100 * (a_arr < 0.20).mean():.0f}%",
                    flush=True,
                )

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
