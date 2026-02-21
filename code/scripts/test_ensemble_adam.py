#!/usr/bin/env python3
"""Test ensemble average + Adam Chamfer refinement.

The ensemble average gives RMSE~0.118 at PV=3 (vs 0.119-0.131 for individual models).
This is well within the Chamfer basin of attraction.
Test if Adam from the cleaner ensemble init can push past 95% success.
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
import time

from src.sim.pipeline import forward_pipeline
from src.sim.lenslet import LensletArray
from src.recon.least_squares import build_zernike_slope_matrix
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


def chamfer_adam_refine(
    init_coeffs,
    obs_positions,
    G,
    ref_positions,
    focal_um,
    device,
    lr=0.002,
    n_iter=200,
    pred_chunk=2048,
):
    """Adam refinement using backward Chamfer loss.

    Uses chunked cdist to handle large number of subapertures.
    """
    N_sub = G.shape[0] // 2
    obs_t = torch.tensor(obs_positions, dtype=torch.float32, device=device)
    G_t = torch.tensor(G, dtype=torch.float32, device=device)
    ref_t = torch.tensor(ref_positions, dtype=torch.float32, device=device)

    # Pass all 10 coefficients (piston column in G is zero, so it's harmless)
    coeffs_t = torch.tensor(
        init_coeffs.copy(), dtype=torch.float32, device=device, requires_grad=True
    )

    optimizer = torch.optim.Adam([coeffs_t], lr=lr)

    best_loss = float("inf")
    best_coeffs = coeffs_t.detach().clone()

    for step in range(n_iter):
        optimizer.zero_grad()

        # Forward model: compute predicted positions
        slopes = G_t @ coeffs_t  # (2*N_sub,)
        sx = slopes[:N_sub]
        sy = slopes[N_sub:]
        pred = ref_t + focal_um * torch.stack([sx, sy], dim=-1)  # (N_sub, 2)

        # Backward Chamfer: for each observed spot, find nearest predicted
        # Use chunked cdist to avoid OOM
        M = obs_t.shape[0]
        min_dists = torch.zeros(M, device=device)

        for i in range(0, M, pred_chunk):
            chunk_obs = obs_t[i : i + pred_chunk]  # (chunk, 2)
            # cdist: (chunk, N_sub)
            dists = torch.cdist(chunk_obs.unsqueeze(0), pred.unsqueeze(0)).squeeze(0)
            min_dists[i : i + pred_chunk] = dists.min(dim=1).values

        loss = min_dists.mean()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_coeffs = coeffs_t.detach().clone()

        loss.backward()
        optimizer.step()

    result = best_coeffs.cpu().numpy().astype(np.float64)
    result[0] = 0.0  # piston always zero
    return result


def main():
    cfg = yaml.safe_load(open("configs/base_no_oracle.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sensor_w = cfg["sensor"]["width_px"] * cfg["sensor"]["pixel_um"]
    sensor_h = cfg["sensor"]["height_px"] * cfg["sensor"]["pixel_um"]

    v2 = load_v2_model("models/nn_warmstart.pt", device)
    v3 = load_v3_model("models/nn_v3_resnet.pt", device)

    # Build G matrix and reference positions
    la = LensletArray(
        pitch_um=cfg["optics"]["pitch_um"],
        focal_mm=cfg["optics"]["focal_mm"],
        fill_factor=cfg["optics"]["fill_factor"],
        sensor_width_px=cfg["sensor"]["width_px"],
        sensor_height_px=cfg["sensor"]["height_px"],
        pixel_um=cfg["sensor"]["pixel_um"],
    )
    G = build_zernike_slope_matrix(la, max_order=3)
    ref_positions = la.reference_positions()  # (N_sub, 2)
    focal_um = cfg["optics"]["focal_mm"] * 1000.0  # mm -> um

    print(f"Device: {device}")
    print(f"G shape: {G.shape}, ref positions: {ref_positions.shape}")
    print(f"Focal: {focal_um} um\n")

    # Test at PV levels around the critical boundary
    for pv in [2.0, 3.0, 4.0, 5.0]:
        n_samples = 50  # fewer samples since Adam is slow

        v3_rmses = []
        ens_rmses = []
        ens_adam_rmses = []
        v3_adam_rmses = []

        t0 = time.time()

        for i in range(n_samples):
            seed = 880000 + int(pv * 1000) + i
            try:
                result = forward_pipeline(cfg, pv=pv, seed=seed)
            except Exception:
                continue

            observed = result["observed_positions"]
            true_coeffs = result["coeffs"]
            if len(observed) < 10:
                continue

            # Individual predictions
            c2 = predict_v2(v2, observed, sensor_w, sensor_h, device)
            c3 = predict_v3(v3, observed, sensor_w, sensor_h, device)
            c_ens = (c2 + c3) / 2.0

            # Adam refinement from ensemble
            c_ens_adam = chamfer_adam_refine(
                c_ens,
                observed,
                G,
                ref_positions,
                focal_um,
                device,
                lr=0.002,
                n_iter=200,
                pred_chunk=2048,
            )

            # Adam refinement from v3 only (for comparison)
            c_v3_adam = chamfer_adam_refine(
                c3,
                observed,
                G,
                ref_positions,
                focal_um,
                device,
                lr=0.002,
                n_iter=200,
                pred_chunk=2048,
            )

            rmse_v3 = float(np.sqrt(np.mean((c3 - true_coeffs) ** 2)))
            rmse_ens = float(np.sqrt(np.mean((c_ens - true_coeffs) ** 2)))
            rmse_ens_adam = float(np.sqrt(np.mean((c_ens_adam - true_coeffs) ** 2)))
            rmse_v3_adam = float(np.sqrt(np.mean((c_v3_adam - true_coeffs) ** 2)))

            v3_rmses.append(rmse_v3)
            ens_rmses.append(rmse_ens)
            ens_adam_rmses.append(rmse_ens_adam)
            v3_adam_rmses.append(rmse_v3_adam)

        dt = time.time() - t0

        v3_a = np.array(v3_rmses)
        ens_a = np.array(ens_rmses)
        ea_a = np.array(ens_adam_rmses)
        va_a = np.array(v3_adam_rmses)

        print(f"=== PV = {pv} ({len(v3_a)} samples, {dt:.1f}s) ===")
        print(
            f"  v3 raw      : mean={v3_a.mean():.4f} | <0.15: {(v3_a < 0.15).mean() * 100:.0f}% | <0.20: {(v3_a < 0.20).mean() * 100:.0f}%"
        )
        print(
            f"  Ensemble    : mean={ens_a.mean():.4f} | <0.15: {(ens_a < 0.15).mean() * 100:.0f}% | <0.20: {(ens_a < 0.20).mean() * 100:.0f}%"
        )
        print(
            f"  v3+Adam     : mean={va_a.mean():.4f} | <0.15: {(va_a < 0.15).mean() * 100:.0f}% | <0.20: {(va_a < 0.20).mean() * 100:.0f}%"
        )
        print(
            f"  Ens+Adam    : mean={ea_a.mean():.4f} | <0.15: {(ea_a < 0.15).mean() * 100:.0f}% | <0.20: {(ea_a < 0.20).mean() * 100:.0f}%"
        )
        print()

    print("Done!")


if __name__ == "__main__":
    main()
