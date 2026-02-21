"""Analyze spot displacement distribution as function of distance from center.
At high PV, central spots should have small displacement (dominated by
higher-order terms which are smaller) while edge spots have large displacement."""

import torch, numpy as np, yaml
from src.sim.pipeline import forward_pipeline
from src.sim.lenslet import LensletArray
from src.recon.least_squares import build_zernike_slope_matrix

with open("configs/base_no_oracle.yaml") as f:
    cfg = yaml.safe_load(f)

oc = cfg["optics"]
sc = cfg["sensor"]
la = LensletArray(
    oc["pitch_um"],
    oc["focal_mm"],
    oc.get("fill_factor", 1.0),
    sc["width_px"],
    sc["height_px"],
    sc["pixel_um"],
)
ref = la.reference_positions()
n_sub = len(ref)
pitch = oc["pitch_um"]
focal = oc["focal_mm"] * 1000.0
center = np.array(
    [sc["width_px"] * sc["pixel_um"] / 2, sc["height_px"] * sc["pixel_um"] / 2]
)

G = build_zernike_slope_matrix(la, cfg["zernike"]["order"])

for pv in [5.0, 10.0, 15.0]:
    sim = forward_pipeline(cfg, pv=pv, seed=42)
    obs = sim["observed_positions"]
    c_true = sim["coeffs"]
    keep_mask = sim["keep_mask"]
    displaced = sim["displaced_positions"]  # all positions before clipping

    # Displacement = displaced - ref for all subapertures
    disp = displaced - ref
    disp_mag = np.sqrt(disp[:, 0] ** 2 + disp[:, 1] ** 2)

    # Distance from center for each subaperture
    dist_from_center = np.sqrt(
        (ref[:, 0] - center[0]) ** 2 + (ref[:, 1] - center[1]) ** 2
    )

    # Bin by distance from center
    bins = np.linspace(0, dist_from_center.max(), 10)
    print(
        f"\nPV={pv:.1f} | n_obs={len(obs)} | c_true_norm={np.linalg.norm(c_true):.3f}"
    )
    print(
        f"  {'dist_range':>20s} | {'n_sub':>6s} | {'n_kept':>6s} | {'mean_disp':>10s} | {'max_disp':>10s} | {'disp/pitch':>10s}"
    )
    for i in range(len(bins) - 1):
        mask = (dist_from_center >= bins[i]) & (dist_from_center < bins[i + 1])
        if mask.sum() == 0:
            continue
        n = mask.sum()
        n_kept = keep_mask[mask].sum()
        md = disp_mag[mask].mean()
        mx = disp_mag[mask].max()
        print(
            f"  {bins[i]:8.0f}-{bins[i + 1]:8.0f} um | {n:6d} | {n_kept:6d} | {md:10.1f} um | {mx:10.1f} um | {md / pitch:10.2f}"
        )

    # For the kept (observed) spots: what's the nearest-ref displacement?
    obs_sub_idx = sim.get("observed_sub_idx", None)
    if obs_sub_idx is not None:
        obs_disp = disp[obs_sub_idx]
        obs_disp_mag = np.sqrt(obs_disp[:, 0] ** 2 + obs_disp[:, 1] ** 2)
        print(
            f"  Observed spots: displacement min={obs_disp_mag.min():.1f} "
            f"median={np.median(obs_disp_mag):.1f} "
            f"max={obs_disp_mag.max():.1f} um "
            f"(pitch={pitch:.1f})"
        )

    # How many observed spots have displacement < 1 pitch?
    if obs_sub_idx is not None:
        n_small = (obs_disp_mag < pitch).sum()
        n_medium = (obs_disp_mag < 2 * pitch).sum()
        print(
            f"  Displacements < 1 pitch: {n_small}/{len(obs)} ({100 * n_small / len(obs):.1f}%)"
        )
        print(
            f"  Displacements < 2 pitch: {n_medium}/{len(obs)} ({100 * n_medium / len(obs):.1f}%)"
        )
