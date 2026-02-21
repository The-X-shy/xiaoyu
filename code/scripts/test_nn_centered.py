#!/usr/bin/env python3
"""Test: NN-centered random search + Adam refinement.

Strategy: Use NN prediction as center, sample perturbations around it
(sigma=0.3), then Adam-refine the best.
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
    print("NN model loaded\n")

    pv_levels = [1.0, 3.0, 5.0, 8.0, 10.0, 15.0]
    n_per_pv = 10

    # Test different perturbation strategies
    for sigma in [0.15, 0.25, 0.40]:
        print(f"\n{'=' * 70}")
        print(f"Perturbation sigma = {sigma}")
        print(f"{'=' * 70}")
        print(
            f"{'PV':>5s} | {'NN_RMSE':>8s} | {'Search':>8s} | {'Adam':>8s} | {'<0.15':>6s} | {'<0.20':>6s} | {'time':>6s}"
        )
        print("-" * 65)

        for pv in pv_levels:
            nn_rmses = []
            search_rmses = []
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

                # Create Chamfer optimizer with custom search around NN center
                chamfer_opt = ChamferOptimizer(observed, la, cfg, device=device)

                # Override sampling: generate perturbations around NN prediction
                D = chamfer_opt.n_terms
                rng = np.random.RandomState(seed + 9000)
                n_samples = 10000

                # Gaussian perturbations around NN prediction
                center = nn_coeffs.copy()
                perturbations = rng.randn(n_samples, D).astype(np.float32) * sigma
                samples = center[np.newaxis, :] + perturbations

                # Also include NN prediction itself and zero
                samples = np.vstack(
                    [center[np.newaxis, :], samples, np.zeros((1, D), dtype=np.float32)]
                )

                samples_t = torch.tensor(samples, dtype=torch.float32, device=device)

                # Evaluate all samples
                obs_sub = chamfer_opt._subsample_obs(256, rng)
                all_obj = []
                for start in range(0, len(samples_t), 2048):
                    end = min(start + 2048, len(samples_t))
                    with torch.no_grad():
                        obj = chamfer_opt._backward_chamfer_full(
                            samples_t[start:end], obs_sub
                        )
                    all_obj.append(obj)
                all_obj = torch.cat(all_obj)

                # Top-K
                topk = 32
                _, topk_idx = torch.topk(all_obj, topk, largest=False)
                top_coeffs = samples_t[topk_idx]

                # Best from search
                best_search_idx = int(all_obj.argmin().item())
                best_search = samples[best_search_idx]
                search_rmse = float(np.sqrt(np.mean((best_search - true_coeffs) ** 2)))

                # Adam refinement of top-K
                obs_r = chamfer_opt._subsample_obs(min(512, len(observed)), rng)
                coeffs_opt = top_coeffs.clone().detach().requires_grad_(True)
                optimizer = torch.optim.Adam([coeffs_opt], lr=0.002)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=300, eta_min=0.0002
                )

                for it in range(300):
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
                    obs_f = chamfer_opt._subsample_obs(min(1024, len(observed)), rng)
                    final_loss = chamfer_opt._backward_chamfer_full(coeffs_opt, obs_f)
                    best_idx = int(final_loss.argmin().item())
                    adam_c = coeffs_opt[best_idx].detach().cpu().numpy()

                adam_rmse = float(np.sqrt(np.mean((adam_c - true_coeffs) ** 2)))
                dt = time.time() - t0

                nn_rmses.append(nn_rmse)
                search_rmses.append(search_rmse)
                adam_rmses.append(adam_rmse)
                times.append(dt)

            if not nn_rmses:
                continue
            nn_arr = np.array(nn_rmses)
            s_arr = np.array(search_rmses)
            a_arr = np.array(adam_rmses)
            print(
                f"{pv:5.1f} | {nn_arr.mean():8.4f} | {s_arr.mean():8.4f} | "
                f"{a_arr.mean():8.4f} | {(a_arr < 0.15).mean() * 100:5.1f}% | "
                f"{(a_arr < 0.20).mean() * 100:5.1f}% | {np.mean(times):5.1f}s"
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
