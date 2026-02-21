#!/usr/bin/env python3
"""Analyze PV=3 failures: are they bias or variance?

Test the same PV=3 samples multiple times with slightly different
model initializations to see if failures are consistent (bias)
or random (variance).

Also: test simple TTA (test-time augmentation) by adding small noise to input.
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
import time

from src.sim.pipeline import forward_pipeline
from scripts.train_nn_v3 import ZernikeResNet, spots_to_image_128, N_COEFFS


def load_v3_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device)
    model = ZernikeResNet(n_coeffs=ckpt.get("output_dim", N_COEFFS))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_v3(model, observed, sensor_w, sensor_h, device):
    img = spots_to_image_128(observed, sensor_w, sensor_h)
    inp = torch.from_numpy(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(inp).squeeze(0).cpu().numpy()
    coeffs = np.zeros(pred.shape[0] + 1, dtype=np.float64)
    coeffs[1:] = pred
    return coeffs


def predict_v3_tta(model, observed, sensor_w, sensor_h, device, n_aug=10, noise_um=5.0):
    """TTA: add small positional noise to observed spots, average predictions."""
    rng = np.random.RandomState(42)
    preds = []

    # Original prediction
    preds.append(predict_v3(model, observed, sensor_w, sensor_h, device))

    # Augmented predictions
    for _ in range(n_aug - 1):
        noise = rng.randn(*observed.shape).astype(np.float32) * noise_um
        obs_aug = observed + noise
        # Clip to sensor bounds
        obs_aug[:, 0] = np.clip(obs_aug[:, 0], 0, sensor_w)
        obs_aug[:, 1] = np.clip(obs_aug[:, 1], 0, sensor_h)
        preds.append(predict_v3(model, obs_aug, sensor_w, sensor_h, device))

    return np.mean(preds, axis=0)


def main():
    cfg = yaml.safe_load(open("configs/base_no_oracle.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sensor_w = cfg["sensor"]["width_px"] * cfg["sensor"]["pixel_um"]
    sensor_h = cfg["sensor"]["height_px"] * cfg["sensor"]["pixel_um"]

    model = load_v3_model("models/nn_v3_resnet.pt", device)
    print(f"Device: {device}")

    # Test at PV=3 and PV=5 with detailed analysis
    for pv in [2.0, 3.0, 4.0, 5.0]:
        n_samples = 100

        print(f"\n=== PV = {pv} ({n_samples} samples) ===")

        nn_rmses = []
        tta_rmses = []
        nspots = []

        for i in range(n_samples):
            seed = 890000 + int(pv * 1000) + i
            try:
                result = forward_pipeline(cfg, pv=pv, seed=seed)
            except Exception:
                continue

            observed = result["observed_positions"]
            true_coeffs = result["coeffs"]
            if len(observed) < 10:
                continue

            # Standard prediction
            nn_coeffs = predict_v3(model, observed, sensor_w, sensor_h, device)
            nn_rmse = float(np.sqrt(np.mean((nn_coeffs - true_coeffs) ** 2)))

            # TTA prediction (10 augmentations, 5um noise)
            tta_coeffs = predict_v3_tta(
                model, observed, sensor_w, sensor_h, device, n_aug=10, noise_um=5.0
            )
            tta_rmse = float(np.sqrt(np.mean((tta_coeffs - true_coeffs) ** 2)))

            nn_rmses.append(nn_rmse)
            tta_rmses.append(tta_rmse)
            nspots.append(len(observed))

        nn_a = np.array(nn_rmses)
        tta_a = np.array(tta_rmses)

        print(
            f"  NN  mean RMSE: {nn_a.mean():.4f} | <0.10: {(nn_a < 0.10).mean() * 100:.0f}% | <0.15: {(nn_a < 0.15).mean() * 100:.0f}% | <0.20: {(nn_a < 0.20).mean() * 100:.0f}%"
        )
        print(
            f"  TTA mean RMSE: {tta_a.mean():.4f} | <0.10: {(tta_a < 0.10).mean() * 100:.0f}% | <0.15: {(tta_a < 0.15).mean() * 100:.0f}% | <0.20: {(tta_a < 0.20).mean() * 100:.0f}%"
        )

        # Analyze failures
        nn_fails = nn_a >= 0.15
        tta_fails = tta_a >= 0.15
        if nn_fails.sum() > 0:
            print(
                f"  NN failures: {nn_fails.sum()} samples, mean RMSE of failures: {nn_a[nn_fails].mean():.4f}"
            )
            # Correlation between NN failure and number of spots
            fail_spots = np.array(nspots)[nn_fails]
            ok_spots = np.array(nspots)[~nn_fails]
            print(
                f"  Failed samples avg spots: {fail_spots.mean():.0f} | OK samples avg spots: {ok_spots.mean():.0f}"
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
