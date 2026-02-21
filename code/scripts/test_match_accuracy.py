#!/usr/bin/env python3
"""Diagnose matching accuracy: what fraction of NN-predicted matches are correct?

Compare NN-based matching to oracle (ground truth) matching.
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


def compute_matches(
    observed_t, coeffs, ref_t, G_t, n_sub, focal_um, device, max_dist=200.0
):
    """Compute nearest-neighbor matches from NN-predicted positions."""
    coeffs_t = torch.tensor(coeffs, dtype=torch.float32, device=device)
    slopes = G_t @ coeffs_t
    sx = slopes[:n_sub]
    sy = slopes[n_sub:]
    pred_pos = ref_t.clone()
    pred_pos[:, 0] += focal_um * sx
    pred_pos[:, 1] += focal_um * sy

    N_obs = len(observed_t)

    # For each observed spot, find nearest predicted (all 13224 subs)
    nearest_idx = torch.empty(N_obs, dtype=torch.long, device=device)
    nearest_dist = torch.empty(N_obs, dtype=torch.float32, device=device)

    chunk = 2048
    for i in range(0, N_obs, chunk):
        end = min(i + chunk, N_obs)
        obs_chunk = observed_t[i:end]

        best_d = torch.full((end - i,), float("inf"), device=device)
        best_j = torch.zeros(end - i, dtype=torch.long, device=device)

        pchunk = 4096
        for j in range(0, n_sub, pchunk):
            jend = min(j + pchunk, n_sub)
            d = torch.cdist(
                obs_chunk.unsqueeze(0), pred_pos[j:jend].unsqueeze(0)
            ).squeeze(0)
            min_d, min_j = d.min(dim=1)
            better = min_d < best_d
            best_d[better] = min_d[better]
            best_j[better] = min_j[better] + j

        nearest_idx[i:end] = best_j
        nearest_dist[i:end] = best_d

    return nearest_idx.cpu().numpy(), nearest_dist.cpu().numpy()


def main():
    cfg = yaml.safe_load(open("configs/base_no_oracle.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    opt = cfg["optics"]
    sen = cfg["sensor"]
    zer = cfg["zernike"]

    la = LensletArray(
        pitch_um=opt["pitch_um"],
        focal_mm=opt["focal_mm"],
        fill_factor=opt["fill_factor"],
        sensor_width_px=sen["width_px"],
        sensor_height_px=sen["height_px"],
        pixel_um=sen["pixel_um"],
    )
    ref = la.reference_positions()
    n_sub = len(ref)
    focal_um = la.focal_um
    sensor_w = sen["width_px"] * sen["pixel_um"]
    sensor_h = sen["height_px"] * sen["pixel_um"]
    G = build_zernike_slope_matrix(la, max_order=zer["order"], grid_size=128)
    n_terms = G.shape[1]

    ref_t = torch.tensor(ref, dtype=torch.float32, device=device)
    G_t = torch.tensor(G, dtype=torch.float32, device=device)

    model = load_v3_model("models/nn_v3_resnet.pt", device)

    # Also compute matches from TRUE coefficients for comparison
    pv_levels = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 8.0, 10.0, 15.0]
    n_per_pv = 20

    print(
        f"\n{'PV':>5s} | {'nn_match%':>9s} | {'true_match%':>11s} | {'nn_medDist':>10s} | {'true_medDist':>12s} | {'nObs':>6s}"
    )
    print("-" * 75)

    for pv in pv_levels:
        nn_match_fracs = []
        true_match_fracs = []
        nn_med_dists = []
        true_med_dists = []
        n_obs_list = []

        for i in range(n_per_pv):
            seed = 870000 + int(pv * 1000) + i
            try:
                result = forward_pipeline(cfg, pv=pv, seed=seed)
            except Exception:
                continue

            observed = result["observed_positions"]
            true_coeffs = result["coeffs"]
            oracle_sub_idx = result.get("observed_sub_idx")
            if oracle_sub_idx is None or len(observed) < 10:
                continue

            observed_t = torch.tensor(observed, dtype=torch.float32, device=device)

            # NN prediction
            nn_coeffs = predict_v3(model, observed, sensor_w, sensor_h, device)

            # Matches from NN prediction
            nn_idx, nn_dist = compute_matches(
                observed_t, nn_coeffs, ref_t, G_t, n_sub, focal_um, device
            )

            # Matches from TRUE coefficients
            true_idx, true_dist = compute_matches(
                observed_t, true_coeffs, ref_t, G_t, n_sub, focal_um, device
            )

            # Compare to oracle
            nn_correct = (nn_idx == oracle_sub_idx).mean()
            true_correct = (true_idx == oracle_sub_idx).mean()

            nn_match_fracs.append(nn_correct)
            true_match_fracs.append(true_correct)
            nn_med_dists.append(np.median(nn_dist))
            true_med_dists.append(np.median(true_dist))
            n_obs_list.append(len(observed))

        if nn_match_fracs:
            print(
                f"{pv:5.1f} | {np.mean(nn_match_fracs) * 100:8.1f}% | "
                f"{np.mean(true_match_fracs) * 100:10.1f}% | "
                f"{np.mean(nn_med_dists):10.1f} | {np.mean(true_med_dists):12.1f} | "
                f"{np.mean(n_obs_list):6.0f}"
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
