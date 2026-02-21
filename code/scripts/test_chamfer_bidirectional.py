"""Test a MODIFIED Chamfer objective that penalizes mismatch between
number of in-bounds predictions and number of observations.

Hypothesis: at PV=10, true coefficients produce exactly 905 in-bounds
predictions matching 905 observations. Zero produces 13224 in-bounds but
only 905 observations. The count mismatch is a strong signal.

New objective:
  loss = backward_chamfer + alpha * |n_inbounds - n_obs| / n_obs
"""

import torch, numpy as np, yaml
from src.sim.pipeline import forward_pipeline
from src.recon.chamfer_optimizer import ChamferOptimizer
from src.sim.lenslet import LensletArray

torch.cuda.empty_cache()

with open("configs/base_no_oracle.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["asm"]["chamfer_pred_chunk"] = 512

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
n_terms = 10

for pv in [1.0, 5.0, 10.0, 15.0]:
    sim = forward_pipeline(cfg, pv=pv, seed=42)
    obs = sim["observed_positions"]
    c_true = sim["coeffs"]
    n_obs = len(obs)

    ch = ChamferOptimizer(obs, la, cfg, device=torch.device("cuda"))
    obs_all = ch.observed_full

    # Build test candidates
    candidates = {
        "true_all": c_true.copy(),
        "true_tt": np.zeros(n_terms, dtype=np.float32),
        "zero": np.zeros(n_terms, dtype=np.float32),
    }
    candidates["true_tt"][1] = c_true[1]
    candidates["true_tt"][2] = c_true[2]

    # Also add: true + noise at various levels
    rng = np.random.RandomState(42)
    for sigma in [0.05, 0.10, 0.20, 0.50]:
        c_noisy = c_true + rng.randn(n_terms).astype(np.float32) * sigma
        c_noisy[0] = 0.0
        candidates[f"true+{sigma}"] = c_noisy

    # Also add random vectors at various scales
    for scale in [0.5, 1.0, 2.0]:
        c_rand = rng.randn(n_terms).astype(np.float32) * scale
        c_rand[0] = 0.0
        candidates[f"rand_{scale}"] = c_rand

    cands_np = np.stack(list(candidates.values()))
    cands_t = torch.tensor(cands_np, dtype=torch.float32, device="cuda")

    # Compute backward Chamfer AND in-bounds counts
    with torch.no_grad():
        objs = ch._backward_chamfer_full(cands_t, obs_all)

        # Compute in-bounds counts
        E = ch._compute_expected(cands_t, ch.ref, ch.G, ch.n_sub)
        ib = (
            (E[:, :, 0] >= 0)
            & (E[:, :, 0] <= ch.sensor_w)
            & (E[:, :, 1] >= 0)
            & (E[:, :, 1] <= ch.sensor_h)
        )
        n_ib = ib.sum(dim=1).float()

        # Count penalty: |n_inbounds - n_obs| / n_obs
        count_penalty = torch.abs(n_ib - n_obs) / n_obs

        # Also try: forward Chamfer (for each prediction, nearest obs)
        # We can compute it chunked
        fwd_chamfer = torch.zeros(len(cands_np), device="cuda")
        for ci in range(len(cands_np)):
            E_single = E[ci : ci + 1]  # (1, n_sub, 2)
            ib_single = ib[ci]  # (n_sub,)
            ib_idx = ib_single.nonzero(as_tuple=True)[0]
            if len(ib_idx) == 0:
                fwd_chamfer[ci] = 100.0
                continue
            E_ib = E_single[0, ib_idx]  # (n_ib, 2)
            # chunk
            fwd_sum = 0.0
            chunk = 2048
            for s in range(0, len(ib_idx), chunk):
                e_end = min(s + chunk, len(ib_idx))
                d = torch.cdist(E_ib[s:e_end].unsqueeze(0), obs_all.unsqueeze(0))[0]
                fwd_sum += (d.min(dim=1).values / ch.pitch_um).sum().item()
            fwd_chamfer[ci] = fwd_sum / max(len(ib_idx), 1)

    print(f"\nPV={pv:.1f} | n_obs={n_obs}")
    print(
        f"  {'name':15s} | {'bwd':>8s} | {'fwd':>8s} | {'count_p':>8s} | {'n_ib':>6s} | {'rmse':>6s} | {'bwd+cp':>8s} | {'bwd+fwd':>8s}"
    )
    print(
        f"  {'-' * 15}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 8}-+-{'-' * 8}"
    )
    for i, (name, c) in enumerate(candidates.items()):
        rmse = np.sqrt(np.mean((c - c_true) ** 2))
        bwd = objs[i].item()
        fwd = fwd_chamfer[i].item()
        cp = count_penalty[i].item()
        nib = int(n_ib[i].item())
        # Combined objectives with various alphas
        combined1 = bwd + 1.0 * cp  # bwd + count penalty
        combined2 = bwd + 0.5 * fwd  # bwd + fwd Chamfer
        print(
            f"  {name:15s} | {bwd:8.4f} | {fwd:8.4f} | {cp:8.4f} | {nib:6d} | {rmse:6.4f} | {combined1:8.4f} | {combined2:8.4f}"
        )

    torch.cuda.empty_cache()
