"""Deep diagnostic of sorting matcher at specific PV levels.

Calls sorting_match with verbose mode and checks accuracy against oracle.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_experiment_config
from src.sim.pipeline import forward_pipeline
from src.recon.sorting_matcher import sorting_match
from src.recon.zernike import num_zernike_terms
from src.recon.least_squares import build_zernike_slope_matrix
from src.recon.sorting_matcher import (
    _assign_rows,
    _sort_positions,
    _sort_positions_adaptive,
    _cdf_rank_match,
    _solve_coeffs,
)

cfg = load_experiment_config("configs/base.yaml")
# Test with zero noise first
cfg["noise"] = {"read_sigma": 0.0, "background": 0.0, "centroid_noise_px": 0.0}

for pv in [2.0, 5.0, 10.0, 30.0, 60.0]:
    print(f"\n{'=' * 70}")
    print(f"PV = {pv}, NO NOISE")
    print(f"{'=' * 70}")

    result = forward_pipeline(cfg, pv=pv, seed=42)
    obs = result["observed_positions"]
    la = result["lenslet"]
    true_c = result["coeffs"]
    true_sub_idx = result["observed_sub_idx"]
    ref = la.reference_positions()

    print(f"  n_obs={len(obs)}, n_ref={len(ref)}, ratio={len(obs) / len(ref):.3f}")

    # Step 1: row assignment from ref
    pitch = la.pitch_um
    rows_info, row_labels = _assign_rows(ref, pitch)
    row_counts = [
        r[1] for r in rows_info
    ]  # extract counts from (center_y, count) tuples
    print(f"  n_rows={len(rows_info)}, row_counts (first 5)={row_counts[:5]}")

    # Step 2: sort reference
    ref_order = _sort_positions(ref, row_counts)
    print(f"  ref_order[:10]={ref_order[:10]}")

    # Step 3: sort observed adaptively
    obs_order = _sort_positions_adaptive(obs, row_counts)
    print(f"  obs_order[:10]={obs_order[:10]}")

    # Step 4: CDF rank match
    matched_sub, matched_obs, n_matched = _cdf_rank_match(
        ref_order, obs_order, len(ref), len(obs)
    )
    print(f"  CDF matches: {n_matched}")

    # Check CDF match accuracy
    n_correct = 0
    for i in range(n_matched):
        ri = matched_sub[i]
        oi = matched_obs[i]
        if oi < len(true_sub_idx) and ri == true_sub_idx[oi]:
            n_correct += 1
    print(
        f"  CDF match accuracy: {n_correct}/{n_matched} = {n_correct / max(n_matched, 1):.2%}"
    )

    # Step 5: Full sorting_match
    sm = sorting_match(obs, la, cfg)
    rmse_sm = np.sqrt(np.mean((sm["coeffs"] - true_c) ** 2))
    print(
        f"  sorting_match: success={sm['success']}, n_matched={sm['n_matched']}, RMSE={rmse_sm:.4f}"
    )
    print(f"  solver: {sm.get('solver', '?')}")

    # Step 6: Oracle (perfect matching)
    zer_cfg = cfg.get("zernike", {})
    asm_cfg = cfg.get("asm", {})
    max_order = zer_cfg.get("order", 3)
    n_terms = num_zernike_terms(max_order)
    grid_size = asm_cfg.get("grid_size", 128)
    lambda_reg = asm_cfg.get("lambda_reg", 1e-3)
    G = build_zernike_slope_matrix(la, max_order, grid_size)

    oracle_coeffs = _solve_coeffs(
        obs, ref, G, true_sub_idx, la.n_subapertures, la.focal_um, n_terms, lambda_reg
    )
    if oracle_coeffs is not None:
        rmse_oracle = np.sqrt(np.mean((oracle_coeffs - true_c) ** 2))
        print(f"  Oracle RMSE: {rmse_oracle:.6f}")

    # Show coefficient comparison
    print(f"  True coeffs:    {np.array2string(true_c, precision=4, separator=', ')}")
    print(
        f"  Sorting coeffs: {np.array2string(sm['coeffs'], precision=4, separator=', ')}"
    )
    if oracle_coeffs is not None:
        print(
            f"  Oracle coeffs:  {np.array2string(oracle_coeffs, precision=4, separator=', ')}"
        )
