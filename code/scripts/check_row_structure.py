"""Check if observed spots maintain row structure at high PV.

Key question: Is the gap between rows always larger than the spread within rows?
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_experiment_config
from src.sim.pipeline import forward_pipeline

cfg = load_experiment_config("configs/base.yaml")
cfg["noise"] = {"read_sigma": 0.0, "background": 0.0, "centroid_noise_px": 0.0}

for pv in [2.0, 5.0, 10.0, 30.0, 60.0]:
    print(f"\nPV = {pv}")
    result = forward_pipeline(cfg, pv=pv, seed=42)
    obs = result["observed_positions"]
    la = result["lenslet"]
    true_sub_idx = result["observed_sub_idx"]
    ref = la.reference_positions()
    pitch = la.pitch_um

    # Check true row structure in observed spots
    # Each ref spot has a row based on its y-coordinate
    ref_y = ref[:, 1]
    # Assign each ref spot to a row
    tol = pitch * 0.5
    sorted_idx = np.argsort(ref_y)
    row_ids = np.zeros(len(ref), dtype=int)
    current_row = 0
    current_y = ref_y[sorted_idx[0]]
    for i, idx in enumerate(sorted_idx):
        if ref_y[idx] - current_y > tol:
            current_row += 1
            current_y = ref_y[idx]
        row_ids[idx] = current_row

    # Now check observed spot y-coordinates grouped by their TRUE row
    obs_row_ids = row_ids[true_sub_idx]
    unique_rows = np.unique(obs_row_ids)

    intra_row_spreads = []
    row_centers = []
    for r in unique_rows:
        mask = obs_row_ids == r
        if mask.sum() > 1:
            obs_y = obs[mask, 1]
            spread = obs_y.max() - obs_y.min()
            intra_row_spreads.append(spread)
            row_centers.append(np.mean(obs_y))

    inter_row_gaps = np.diff(sorted(row_centers))

    max_spread = max(intra_row_spreads) if intra_row_spreads else 0
    min_gap = min(inter_row_gaps) if len(inter_row_gaps) > 0 else float("inf")
    median_gap = np.median(inter_row_gaps) if len(inter_row_gaps) > 0 else float("inf")

    print(f"  n_obs={len(obs)}, n_ref={len(ref)}, n_rows={len(unique_rows)}")
    print(f"  Max intra-row y-spread: {max_spread:.2f} um")
    print(f"  Min inter-row y-gap:    {min_gap:.2f} um")
    print(f"  Median inter-row y-gap: {median_gap:.2f} um")
    print(f"  Pitch: {pitch:.2f} um")
    print(
        f"  Row separation > spread: {min_gap > max_spread} (margin={min_gap - max_spread:.2f} um)"
    )

    # Also: what would a threshold of pitch/3 give?
    # Count how many spots would be mis-grouped
    # Sort observed by y and use threshold-based grouping
    obs_y_sorted_idx = np.argsort(obs[:, 1])
    obs_y_sorted = obs[obs_y_sorted_idx, 1]
    gaps_obs = np.diff(obs_y_sorted)
    threshold = pitch * 0.4
    n_gaps_above_threshold = np.sum(gaps_obs > threshold)
    true_n_rows = len(unique_rows)
    print(
        f"  Threshold={threshold:.0f}um: found {n_gaps_above_threshold + 1} groups vs {true_n_rows} true rows"
    )
