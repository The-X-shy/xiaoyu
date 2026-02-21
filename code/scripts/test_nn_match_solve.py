#!/usr/bin/env python3
"""Test NN v3 -> position matching -> least-squares (GPU-accelerated).

Key idea: NN predicts approximate coefficients, compute predicted positions,
match observed to predicted via nearest-neighbor, solve LS like oracle.
Then iterate: new coeffs -> new positions -> re-match -> solve.
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


def nn_match_solve_gpu(
    observed_t,
    coeffs,
    ref_t,
    G_t,
    n_sub,
    focal_um,
    sensor_w,
    sensor_h,
    n_terms,
    device,
    lambda_reg=1e-3,
    max_dist_um=150.0,
):
    """GPU-accelerated: predict positions, match, solve LS.

    Returns: new_coeffs (numpy), n_matched, match_frac
    """
    coeffs_t = torch.tensor(coeffs, dtype=torch.float32, device=device)

    # Compute predicted positions
    slopes = G_t @ coeffs_t  # (2*n_sub,)
    sx = slopes[:n_sub]
    sy = slopes[n_sub:]
    pred_pos = ref_t.clone()
    pred_pos[:, 0] += focal_um * sx
    pred_pos[:, 1] += focal_um * sy

    # Filter to in-bounds predictions
    ib = (
        (pred_pos[:, 0] >= 0)
        & (pred_pos[:, 0] <= sensor_w)
        & (pred_pos[:, 1] >= 0)
        & (pred_pos[:, 1] <= sensor_h)
    )
    ib_idx = torch.where(ib)[0]
    pred_ib = pred_pos[ib_idx]  # (N_ib, 2)

    if len(pred_ib) == 0:
        return coeffs, 0, 0.0

    N_obs = len(observed_t)
    N_pred = len(pred_ib)

    # Chunked nearest-neighbor matching (observed -> predicted)
    nearest_pred_idx = torch.empty(N_obs, dtype=torch.long, device=device)
    nearest_dist = torch.empty(N_obs, dtype=torch.float32, device=device)

    chunk = 2048
    for i in range(0, N_obs, chunk):
        end = min(i + chunk, N_obs)
        obs_chunk = observed_t[i:end]  # (c, 2)

        # Chunked cdist over predictions too
        best_d = torch.full((end - i,), float("inf"), device=device)
        best_j = torch.zeros(end - i, dtype=torch.long, device=device)

        pchunk = 4096
        for j in range(0, N_pred, pchunk):
            jend = min(j + pchunk, N_pred)
            d = torch.cdist(
                obs_chunk.unsqueeze(0), pred_ib[j:jend].unsqueeze(0)
            ).squeeze(0)  # (c, pc)
            min_d, min_j = d.min(dim=1)
            better = min_d < best_d
            best_d[better] = min_d[better]
            best_j[better] = min_j[better] + j

        nearest_pred_idx[i:end] = best_j
        nearest_dist[i:end] = best_d

    # Filter by distance threshold
    good = nearest_dist < max_dist_um
    n_good = good.sum().item()

    if n_good < n_terms:
        return coeffs, n_good, 0.0

    # Get subaperture indices
    obs_good = observed_t[good]
    sub_idx = ib_idx[nearest_pred_idx[good]]

    # Solve least-squares
    target_disp = obs_good - ref_t[sub_idx]
    target_slopes = target_disp / focal_um

    G_sub_x = G_t[sub_idx, :]
    G_sub_y = G_t[n_sub + sub_idx, :]
    A = torch.cat([G_sub_x, G_sub_y], dim=0)
    b = torch.cat([target_slopes[:, 0], target_slopes[:, 1]], dim=0)

    ATA = A.T @ A + lambda_reg * torch.eye(n_terms, device=device)
    ATb = A.T @ b
    try:
        coeffs_new = torch.linalg.solve(ATA, ATb)
    except RuntimeError:
        coeffs_new = torch.linalg.lstsq(A, b).solution

    return coeffs_new.cpu().numpy().astype(np.float64), n_good, n_good / max(N_obs, 1)


def iterative_match_solve(
    observed_t,
    nn_coeffs,
    ref_t,
    G_t,
    n_sub,
    focal_um,
    sensor_w,
    sensor_h,
    n_terms,
    device,
    n_iters=5,
    lambda_reg=1e-3,
    init_dist=200.0,
    final_dist=75.0,
):
    """Iterative: match -> solve -> rematch -> solve, shrinking tolerance."""
    coeffs = nn_coeffs.copy()
    n_matched = 0
    mfrac = 0.0

    for it in range(n_iters):
        # Linearly decrease matching distance
        frac = it / max(n_iters - 1, 1)
        dist = init_dist * (1 - frac) + final_dist * frac

        new_coeffs, n_matched, mfrac = nn_match_solve_gpu(
            observed_t,
            coeffs,
            ref_t,
            G_t,
            n_sub,
            focal_um,
            sensor_w,
            sensor_h,
            n_terms,
            device,
            lambda_reg=lambda_reg,
            max_dist_um=dist,
        )

        if n_matched < n_terms:
            break

        coeffs = new_coeffs

    return coeffs, n_matched, mfrac


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
    print(f"n_sub={n_sub}, n_terms={n_terms}, focal_um={focal_um}")

    pv_levels = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 8.0, 10.0, 15.0, 20.0]
    n_per_pv = 50

    print(
        f"\n{'PV':>5s} | {'NN':>8s} | {'1-step':>8s} | {'5-iter':>8s} | "
        f"{'NN<.15':>6s} | {'1s<.15':>6s} | {'5i<.15':>6s} | {'nMatch':>6s} | {'time':>5s}"
    )
    print("-" * 85)

    for pv in pv_levels:
        nn_rmses = []
        one_rmses = []
        iter_rmses = []
        n_matches = []
        times_list = []

        for i in range(n_per_pv):
            seed = 860000 + int(pv * 1000) + i
            try:
                result = forward_pipeline(cfg, pv=pv, seed=seed)
            except Exception:
                continue

            observed = result["observed_positions"]
            true_coeffs = result["coeffs"]
            if len(observed) < 10:
                continue

            observed_t = torch.tensor(observed, dtype=torch.float32, device=device)
            t0 = time.time()

            # NN prediction
            nn_coeffs = predict_v3(model, observed, sensor_w, sensor_h, device)
            nn_rmse = float(np.sqrt(np.mean((nn_coeffs - true_coeffs) ** 2)))

            # Single-step match + solve (200um tolerance)
            one_coeffs, nm1, mf1 = nn_match_solve_gpu(
                observed_t,
                nn_coeffs,
                ref_t,
                G_t,
                n_sub,
                focal_um,
                sensor_w,
                sensor_h,
                n_terms,
                device,
                max_dist_um=200.0,
            )
            one_rmse = float(np.sqrt(np.mean((one_coeffs - true_coeffs) ** 2)))

            # 5-iteration refinement (200 -> 75 um)
            iter_coeffs, nm5, mf5 = iterative_match_solve(
                observed_t,
                nn_coeffs,
                ref_t,
                G_t,
                n_sub,
                focal_um,
                sensor_w,
                sensor_h,
                n_terms,
                device,
                n_iters=5,
                init_dist=200.0,
                final_dist=75.0,
            )
            iter_rmse = float(np.sqrt(np.mean((iter_coeffs - true_coeffs) ** 2)))

            dt = time.time() - t0
            nn_rmses.append(nn_rmse)
            one_rmses.append(one_rmse)
            iter_rmses.append(iter_rmse)
            n_matches.append(nm5)
            times_list.append(dt)

        if nn_rmses:
            nn_a = np.array(nn_rmses)
            one_a = np.array(one_rmses)
            iter_a = np.array(iter_rmses)
            print(
                f"{pv:5.1f} | {nn_a.mean():8.4f} | {one_a.mean():8.4f} | {iter_a.mean():8.4f} | "
                f"{(nn_a < 0.15).mean() * 100:5.1f}% | {(one_a < 0.15).mean() * 100:5.1f}% | "
                f"{(iter_a < 0.15).mean() * 100:5.1f}% | {np.mean(n_matches):6.0f} | "
                f"{np.mean(times_list):4.1f}s"
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
