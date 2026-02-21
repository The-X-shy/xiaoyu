#!/usr/bin/env python3
"""Test ensemble of v2 CNN + v3 ResNet models.

If their errors are uncorrelated, averaging may improve accuracy at PV=3.
Also test: weighted average, median, best-of-two (oracle pick).
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml

from src.sim.pipeline import forward_pipeline
from src.recon.nn_warmstart import spots_to_image, ZernikeCNN
from scripts.train_nn_v3 import ZernikeResNet, spots_to_image_128, N_COEFFS


def load_v2_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = ZernikeCNN(output_dim=ckpt.get("output_dim", 9))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_v3_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = ZernikeResNet(n_coeffs=ckpt.get("output_dim", N_COEFFS))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_v2(model, observed, sensor_w, sensor_h, device):
    img = spots_to_image(observed, sensor_w, sensor_h)
    inp = torch.from_numpy(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(inp).squeeze(0).cpu().numpy()
    coeffs = np.zeros(pred.shape[0] + 1, dtype=np.float64)
    coeffs[1:] = pred
    return coeffs


def predict_v3(model, observed, sensor_w, sensor_h, device):
    img = spots_to_image_128(observed, sensor_w, sensor_h)
    inp = torch.from_numpy(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(inp).squeeze(0).cpu().numpy()
    coeffs = np.zeros(pred.shape[0] + 1, dtype=np.float64)
    coeffs[1:] = pred
    return coeffs


def main():
    cfg = yaml.safe_load(open("configs/base_no_oracle.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sensor_w = cfg["sensor"]["width_px"] * cfg["sensor"]["pixel_um"]
    sensor_h = cfg["sensor"]["height_px"] * cfg["sensor"]["pixel_um"]

    v2 = load_v2_model("models/nn_warmstart.pt", device)
    v3 = load_v3_model("models/nn_v3_resnet.pt", device)
    print(f"Device: {device}")
    print(f"Loaded v2 CNN and v3 ResNet\n")

    for pv in [2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
        n_samples = 100

        v2_rmses = []
        v3_rmses = []
        avg_rmses = []  # simple average
        median_rmses = []  # median of 2 (= average for 2 models, but different for 3+)
        oracle_rmses = []  # best of two (oracle)
        corr_data = []  # for correlation analysis

        for i in range(n_samples):
            seed = 990000 + int(pv * 1000) + i
            try:
                result = forward_pipeline(cfg, pv=pv, seed=seed)
            except Exception:
                continue

            observed = result["observed_positions"]
            true_coeffs = result["coeffs"]
            if len(observed) < 10:
                continue

            c2 = predict_v2(v2, observed, sensor_w, sensor_h, device)
            c3 = predict_v3(v3, observed, sensor_w, sensor_h, device)

            # Ensemble strategies
            c_avg = (c2 + c3) / 2.0

            rmse_v2 = float(np.sqrt(np.mean((c2 - true_coeffs) ** 2)))
            rmse_v3 = float(np.sqrt(np.mean((c3 - true_coeffs) ** 2)))
            rmse_avg = float(np.sqrt(np.mean((c_avg - true_coeffs) ** 2)))
            rmse_oracle = min(rmse_v2, rmse_v3)

            v2_rmses.append(rmse_v2)
            v3_rmses.append(rmse_v3)
            avg_rmses.append(rmse_avg)
            oracle_rmses.append(rmse_oracle)

            # Per-coefficient error for correlation
            err2 = c2 - true_coeffs
            err3 = c3 - true_coeffs
            corr_data.append((err2, err3))

        v2_a = np.array(v2_rmses)
        v3_a = np.array(v3_rmses)
        avg_a = np.array(avg_rmses)
        oracle_a = np.array(oracle_rmses)

        print(f"=== PV = {pv} ({len(v2_a)} samples) ===")
        print(
            f"  v2 CNN    : mean={v2_a.mean():.4f} | <0.15: {(v2_a < 0.15).mean() * 100:.0f}% | <0.20: {(v2_a < 0.20).mean() * 100:.0f}%"
        )
        print(
            f"  v3 ResNet : mean={v3_a.mean():.4f} | <0.15: {(v3_a < 0.15).mean() * 100:.0f}% | <0.20: {(v3_a < 0.20).mean() * 100:.0f}%"
        )
        print(
            f"  Avg ens.  : mean={avg_a.mean():.4f} | <0.15: {(avg_a < 0.15).mean() * 100:.0f}% | <0.20: {(avg_a < 0.20).mean() * 100:.0f}%"
        )
        print(
            f"  Oracle    : mean={oracle_a.mean():.4f} | <0.15: {(oracle_a < 0.15).mean() * 100:.0f}% | <0.20: {(oracle_a < 0.20).mean() * 100:.0f}%"
        )

        # Error correlation analysis
        if corr_data:
            errs2 = np.array([d[0] for d in corr_data])  # (N, 10)
            errs3 = np.array([d[1] for d in corr_data])
            # Per-coefficient correlation
            corrs = []
            for j in range(errs2.shape[1]):
                if np.std(errs2[:, j]) > 1e-10 and np.std(errs3[:, j]) > 1e-10:
                    c = np.corrcoef(errs2[:, j], errs3[:, j])[0, 1]
                    corrs.append(c)
            if corrs:
                print(
                    f"  Error corr: mean={np.mean(corrs):.3f} min={np.min(corrs):.3f} max={np.max(corrs):.3f}"
                )
        print()

    print("Done!")


if __name__ == "__main__":
    main()
