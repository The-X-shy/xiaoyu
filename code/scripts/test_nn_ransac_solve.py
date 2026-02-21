#!/usr/bin/env python3
"""Test NN v3 -> RANSAC-style robust matching -> LS solve.

Strategy:
1. NN predicts approximate coefficients
2. Compute predicted positions
3. Match ALL observed to nearest predicted (large tolerance)
4. Use RANSAC: randomly select minimal subsets, solve LS, count inliers
5. Final solve uses all inliers from best model
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


def nn_robust_match_solve(
    observed,
    nn_coeffs,
    ref,
    G,
    n_sub,
    focal_um,
    sensor_w,
    sensor_h,
    n_terms,
    inlier_threshold_um=50.0,
    n_ransac=200,
    min_inliers=20,
    lambda_reg=1e-3,
):
    """
    1. Compute predicted positions from NN coeffs
    2. Match each observed spot to nearest predicted
    3. RANSAC: sample minimal subsets, solve LS, count inliers
    4. Final LS from best inlier set
    """
    # Predicted positions from NN
    slopes_flat = G @ nn_coeffs
    sx = slopes_flat[:n_sub]
    sy = slopes_flat[n_sub:]
    pred_pos = ref.copy()
    pred_pos[:, 0] += focal_um * sx
    pred_pos[:, 1] += focal_um * sy

    # Match: for each observed, find nearest predicted
    N_obs = len(observed)
    match_sub_idx = np.empty(N_obs, dtype=int)
    match_dist = np.empty(N_obs)

    for i in range(0, N_obs, 2000):
        end = min(i + 2000, N_obs)
        dx = observed[i:end, 0:1] - pred_pos[:, 0:1].T
        dy = observed[i:end, 1:2] - pred_pos[:, 1:2].T
        d = np.sqrt(dx**2 + dy**2)
        nearest = np.argmin(d, axis=1)
        match_sub_idx[i:end] = nearest
        match_dist[i:end] = d[np.arange(end - i), nearest]

    # Compute matched slopes (using matched subaperture refs)
    matched_ref = ref[match_sub_idx]
    disp = observed - matched_ref
    slopes_obs = disp / focal_um  # (N_obs, 2)

    # Build full system (for inlier evaluation)
    G_x = G[match_sub_idx, :]  # (N_obs, n_terms)
    G_y = G[n_sub + match_sub_idx, :]  # (N_obs, n_terms)

    # RANSAC
    rng = np.random.RandomState(42)
    best_n_inliers = 0
    best_inlier_mask = None
    best_coeffs = nn_coeffs.copy()

    # Minimum sample: n_terms matches (need 2*n_terms equations for n_terms unknowns)
    min_sample = max(n_terms, 10)  # need at least n_terms observations

    for trial in range(n_ransac):
        # Random sample
        idx = rng.choice(N_obs, size=min(min_sample, N_obs), replace=False)

        A_x = G_x[idx]
        A_y = G_y[idx]
        A = np.vstack([A_x, A_y])
        b = np.concatenate([slopes_obs[idx, 0], slopes_obs[idx, 1]])

        # Solve
        ATA = A.T @ A + lambda_reg * np.eye(n_terms)
        ATb = A.T @ b
        try:
            coeffs_trial = np.linalg.solve(ATA, ATb)
        except np.linalg.LinAlgError:
            continue

        # Predict positions from trial coefficients
        slopes_trial = G @ coeffs_trial
        sx_t = slopes_trial[:n_sub]
        sy_t = slopes_trial[n_sub:]
        pred_trial = ref.copy()
        pred_trial[:, 0] += focal_um * sx_t
        pred_trial[:, 1] += focal_um * sy_t

        # For each observed, distance to its matched predicted position
        matched_pred = pred_trial[match_sub_idx]
        residuals = np.sqrt(np.sum((observed - matched_pred) ** 2, axis=1))

        inliers = residuals < inlier_threshold_um
        n_inliers = inliers.sum()

        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_inlier_mask = inliers
            best_coeffs = coeffs_trial

    # Final solve with all inliers
    if best_inlier_mask is not None and best_n_inliers >= n_terms:
        idx = best_inlier_mask
        A = np.vstack([G_x[idx], G_y[idx]])
        b = np.concatenate([slopes_obs[idx, 0], slopes_obs[idx, 1]])
        ATA = A.T @ A + lambda_reg * np.eye(n_terms)
        ATb = A.T @ b
        try:
            best_coeffs = np.linalg.solve(ATA, ATb)
        except np.linalg.LinAlgError:
            pass

    return best_coeffs, best_n_inliers


def iterative_robust_match(
    observed,
    nn_coeffs,
    ref,
    G,
    n_sub,
    focal_um,
    sensor_w,
    sensor_h,
    n_terms,
    n_outer=3,
    n_ransac=100,
    inlier_start=100.0,
    inlier_end=30.0,
    lambda_reg=1e-3,
):
    """Iterative: NN -> robust match+solve -> re-match with updated coeffs."""
    coeffs = nn_coeffs.copy()

    for it in range(n_outer):
        frac = it / max(n_outer - 1, 1)
        inlier_thr = inlier_start * (1 - frac) + inlier_end * frac

        new_coeffs, n_inliers = nn_robust_match_solve(
            observed,
            coeffs,
            ref,
            G,
            n_sub,
            focal_um,
            sensor_w,
            sensor_h,
            n_terms,
            inlier_threshold_um=inlier_thr,
            n_ransac=n_ransac,
            lambda_reg=lambda_reg,
        )

        if n_inliers >= n_terms:
            coeffs = new_coeffs

    return coeffs


def main():
    cfg = yaml.safe_load(open("configs/base_no_oracle.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model = load_v3_model("models/nn_v3_resnet.pt", device)
    print(f"Device: {device}, n_sub={n_sub}, n_terms={n_terms}")

    pv_levels = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]
    n_per_pv = 30

    print(
        f"\n{'PV':>5s} | {'NN':>8s} | {'RANSAC':>8s} | {'Iter':>8s} | "
        f"{'NN<.15':>6s} | {'R<.15':>6s} | {'It<.15':>6s} | {'nInl':>5s} | {'time':>5s}"
    )
    print("-" * 85)

    for pv in pv_levels:
        nn_rmses = []
        ransac_rmses = []
        iter_rmses = []
        n_inliers_list = []
        times_list = []

        for i in range(n_per_pv):
            seed = 880000 + int(pv * 1000) + i
            try:
                result = forward_pipeline(cfg, pv=pv, seed=seed)
            except Exception:
                continue

            observed = result["observed_positions"]
            true_coeffs = result["coeffs"]
            if len(observed) < 10:
                continue

            t0 = time.time()

            nn_coeffs = predict_v3(model, observed, sensor_w, sensor_h, device)
            nn_rmse = float(np.sqrt(np.mean((nn_coeffs - true_coeffs) ** 2)))

            # RANSAC (single pass, 200 trials, 50um inlier threshold)
            r_coeffs, n_inl = nn_robust_match_solve(
                observed,
                nn_coeffs,
                ref,
                G,
                n_sub,
                focal_um,
                sensor_w,
                sensor_h,
                n_terms,
                inlier_threshold_um=50.0,
                n_ransac=200,
            )
            r_rmse = float(np.sqrt(np.mean((r_coeffs - true_coeffs) ** 2)))

            # Iterative (3 outer iters)
            it_coeffs = iterative_robust_match(
                observed,
                nn_coeffs,
                ref,
                G,
                n_sub,
                focal_um,
                sensor_w,
                sensor_h,
                n_terms,
                n_outer=3,
                n_ransac=100,
            )
            it_rmse = float(np.sqrt(np.mean((it_coeffs - true_coeffs) ** 2)))

            dt = time.time() - t0
            nn_rmses.append(nn_rmse)
            ransac_rmses.append(r_rmse)
            iter_rmses.append(it_rmse)
            n_inliers_list.append(n_inl)
            times_list.append(dt)

        if nn_rmses:
            nn_a = np.array(nn_rmses)
            r_a = np.array(ransac_rmses)
            it_a = np.array(iter_rmses)
            print(
                f"{pv:5.1f} | {nn_a.mean():8.4f} | {r_a.mean():8.4f} | {it_a.mean():8.4f} | "
                f"{(nn_a < 0.15).mean() * 100:5.1f}% | {(r_a < 0.15).mean() * 100:5.1f}% | "
                f"{(it_a < 0.15).mean() * 100:5.1f}% | {np.mean(n_inliers_list):5.0f} | "
                f"{np.mean(times_list):4.1f}s"
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
