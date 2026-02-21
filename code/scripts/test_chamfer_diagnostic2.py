"""Diagnostic: compare Chamfer objective at true coefficients vs
tip/tilt-only vs zero. Understand WHY grid search fails."""

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

for pv in [5.0, 10.0, 15.0]:
    sim = forward_pipeline(cfg, pv=pv, seed=42)
    obs = sim["observed_positions"]
    c_true = sim["coeffs"]
    n_obs = len(obs)

    ch = ChamferOptimizer(obs, la, cfg, device=torch.device("cuda"))
    obs_all = ch.observed_full  # all observations

    # Build test candidates
    candidates = {}

    # 1. True coefficients
    candidates["true_all"] = c_true.copy()

    # 2. True tip/tilt only
    c_tt = np.zeros(n_terms, dtype=np.float32)
    c_tt[1] = c_true[1]
    c_tt[2] = c_true[2]
    candidates["true_tt"] = c_tt

    # 3. Zero
    candidates["zero"] = np.zeros(n_terms, dtype=np.float32)

    # 4. Half true tip/tilt
    c_half_tt = np.zeros(n_terms, dtype=np.float32)
    c_half_tt[1] = c_true[1] * 0.5
    c_half_tt[2] = c_true[2] * 0.5
    candidates["half_tt"] = c_half_tt

    # 5. True first 5 terms (order 2)
    c_o2 = np.zeros(n_terms, dtype=np.float32)
    c_o2[:5] = c_true[:5]
    candidates["true_o2"] = c_o2

    # Evaluate all
    cands_np = np.stack(list(candidates.values()))
    cands_t = torch.tensor(cands_np, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        objs = ch._backward_chamfer_full(cands_t, obs_all)

    print(f"\nPV={pv:.1f} | n_obs={n_obs}")
    print(f"  c_true = {np.round(c_true, 4)}")
    for (name, _), obj in zip(candidates.items(), objs):
        c = candidates[name]
        rmse = np.sqrt(np.mean((c - c_true) ** 2))
        print(
            f"  {name:12s}: obj={obj.item():.6f}  rmse={rmse:.4f}  c12=({c[1]:.4f}, {c[2]:.4f})"
        )

    # Also check: how many in-bounds predictions for true_tt vs zero?
    for name in ["true_tt", "zero", "true_all"]:
        c = torch.tensor(
            candidates[name], dtype=torch.float32, device="cuda"
        ).unsqueeze(0)
        slopes = c @ ch.G.T
        sx = slopes[:, : ch.n_sub]
        sy = slopes[:, ch.n_sub :]
        Ex = ch.ref[:, 0].unsqueeze(0) + ch.focal_um * sx
        Ey = ch.ref[:, 1].unsqueeze(0) + ch.focal_um * sy
        ib = (Ex >= 0) & (Ex <= ch.sensor_w) & (Ey >= 0) & (Ey <= ch.sensor_h)
        n_ib = ib.sum().item()
        print(
            f"  {name:12s}: in_bounds={n_ib}/{ch.n_sub} ({100 * n_ib / ch.n_sub:.1f}%)"
        )

    torch.cuda.empty_cache()
