#!/usr/bin/env python3
"""Test NN v3 (ResNet) prediction -> Adam Chamfer refinement.

Uses the v3 ResNet model (128x128, 3ch) for initial prediction,
then refines with Adam on Chamfer loss.
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
import time

from src.sim.pipeline import forward_pipeline
from src.recon.chamfer_optimizer import ChamferOptimizer

# Import v3 model and image rendering
from scripts.train_nn_v3 import (
    ZernikeResNet,
    spots_to_image_128,
    N_COEFFS,
    RES,
    N_CHANNELS,
)


def load_v3_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device)
    model = ZernikeResNet(n_coeffs=ckpt.get("output_dim", N_COEFFS))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded v3 ResNet model from {model_path}")
    return model


def predict_v3(model, observed, sensor_w, sensor_h, device):
    """Predict Zernike coefficients using v3 model."""
    img = spots_to_image_128(observed, sensor_w, sensor_h)
    inp = torch.from_numpy(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(inp).squeeze(0).cpu().numpy()
    # Full coefficients: [0, c1, c2, ..., c9]
    coeffs = np.zeros(pred.shape[0] + 1, dtype=np.float64)
    coeffs[1:] = pred
    return coeffs


def main():
    cfg = yaml.safe_load(open("configs/base_no_oracle.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sensor_w = cfg["sensor"]["width_px"] * cfg["sensor"]["pixel_um"]
    sensor_h = cfg["sensor"]["height_px"] * cfg["sensor"]["pixel_um"]

    model = load_v3_model("models/nn_v3_resnet.pt", device)

    pv_levels = [1.0, 2.0, 3.0, 5.0, 7.0, 8.0, 10.0, 15.0]
    n_per_pv = 20

    print(
        f"\n{'PV':>5s} | {'NN_RMSE':>8s} | {'Adam_RMSE':>9s} | "
        f"{'NN<0.15':>7s} | {'Adam<0.15':>9s} | {'Adam<0.20':>9s} | {'time':>6s}"
    )
    print("-" * 80)

    for pv in pv_levels:
        nn_rmses = []
        adam_rmses = []
        times = []

        for i in range(n_per_pv):
            seed = 850000 + int(pv * 1000) + i
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
            nn_coeffs = predict_v3(model, observed, sensor_w, sensor_h, device)
            nn_rmse = float(np.sqrt(np.mean((nn_coeffs - true_coeffs) ** 2)))

            # Adam refinement from NN prediction
            chamfer_opt = ChamferOptimizer(observed, la, cfg, device=device)
            chamfer_opt.n_sample = 0  # skip random search
            chamfer_opt.refine_iter = 300
            chamfer_opt.refine_lr = 0.002
            chamfer_opt.sample_topk = 1
            chamfer_opt.n_refine = 1
            chamfer_opt.refine_obs_k = min(512, len(observed))
            chamfer_opt.pred_chunk = 1024

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
        improved = (adam_arr < nn_arr).mean() * 100
        print(
            f"{pv:5.1f} | {nn_arr.mean():8.4f} | {adam_arr.mean():9.4f} | "
            f"{(nn_arr < 0.15).mean() * 100:6.1f}% | {(adam_arr < 0.15).mean() * 100:8.1f}% | "
            f"{(adam_arr < 0.20).mean() * 100:8.1f}% | "
            f"{np.mean(times):5.1f}s  [{improved:.0f}% improved]"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
