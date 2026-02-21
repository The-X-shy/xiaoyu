"""Diagnose sorting matcher performance at specific PV levels.

Shows: match accuracy, coefficient comparison, RMSE breakdown.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_experiment_config
from src.sim.pipeline import forward_pipeline
from src.recon.sorting_matcher import sorting_match
from src.recon.baseline_extrap_nn import baseline_reconstruct

cfg = load_experiment_config("configs/base.yaml")

for pv in [2.0, 5.0, 10.0, 30.0]:
    print(f"\n{'=' * 60}")
    print(f"PV = {pv}")
    print(f"{'=' * 60}")

    result = forward_pipeline(cfg, pv=pv, seed=20260210)
    obs = result["observed_positions"]
    la = result["lenslet"]
    true_c = result["coeffs"]
    true_sub_idx = result["observed_sub_idx"]

    # Run sorting match
    sm = sorting_match(obs, la, cfg)

    print(f"  Sorting match success: {sm['success']}")
    print(
        f"  n_matched: {sm['n_matched']} / {len(obs)} observed / {la.n_subapertures} ref"
    )
    print(f"  residual_trimmed: {sm['residual_trimmed']:.2f} um")
    print(f"  solver: {sm.get('solver', '?')}")

    if sm["coeffs"] is not None:
        rmse = np.sqrt(np.mean((sm["coeffs"] - true_c) ** 2))
        rmse_no_piston = np.sqrt(np.mean((sm["coeffs"][1:] - true_c[1:]) ** 2))
        print(f"  RMSE (all): {rmse:.4f}")
        print(f"  RMSE (no piston): {rmse_no_piston:.4f}")
        print(
            f"  True coeffs:    {np.array2string(true_c, precision=4, separator=', ')}"
        )
        print(
            f"  Sorting coeffs: {np.array2string(sm['coeffs'], precision=4, separator=', ')}"
        )

        # Check ratio: coeffs should be close to true_c
        if np.max(np.abs(true_c[1:])) > 0.01:
            ratio = sm["coeffs"][1:] / (true_c[1:] + 1e-10)
            print(
                f"  Ratio (sorting/true): {np.array2string(ratio, precision=3, separator=', ')}"
            )

    # Check match accuracy if we have true_sub_idx
    if "matched_indices" in sm and true_sub_idx is not None:
        mi = sm["matched_indices"]
        if mi is not None:
            # mi maps obs_idx -> ref_sub_idx
            # true_sub_idx maps obs_idx -> true ref_sub_idx
            n_correct = 0
            n_total = 0
            for obs_i, ref_i in enumerate(mi):
                if ref_i >= 0:
                    n_total += 1
                    if obs_i < len(true_sub_idx) and ref_i == true_sub_idx[obs_i]:
                        n_correct += 1
            print(
                f"  Match accuracy: {n_correct}/{n_total} = {n_correct / max(n_total, 1):.2%}"
            )

    # Also test with zero noise
    cfg_no_noise = cfg.copy()
    cfg_no_noise["noise"] = {
        "read_sigma": 0.0,
        "background": 0.0,
        "centroid_noise_px": 0.0,
    }
    result_nn = forward_pipeline(cfg_no_noise, pv=pv, seed=20260210)
    sm_nn = sorting_match(
        result_nn["observed_positions"], result_nn["lenslet"], cfg_no_noise
    )
    if sm_nn["coeffs"] is not None:
        rmse_nn = np.sqrt(np.mean((sm_nn["coeffs"] - result_nn["coeffs"]) ** 2))
        print(f"  [No-noise] success={sm_nn['success']}, RMSE={rmse_nn:.4f}")
