"""Coarse-to-fine grid search over tip/tilt (Z1, Z2) using Chamfer objective,
followed by Adam refinement over all 10 Zernike terms.

Strategy:
  Stage 1: Coarse 2D grid over (Z1, Z2) in [-3, 3]^2, step=0.15
           → 40x40 = 1600 candidates, evaluate via Chamfer
  Stage 2: Fine grid around top-16 coarse hits, step=0.03
           → ~16 * 11*11 = ~1936 candidates
  Stage 3: Adam refinement from top-64 fine hits over all 10 terms
"""

import torch, numpy as np, yaml, time
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

for pv in [1.0, 3.0, 5.0, 8.0, 10.0, 15.0]:
    sim = forward_pipeline(cfg, pv=pv, seed=42)
    obs = sim["observed_positions"]
    c_true = sim["coeffs"]

    ch = ChamferOptimizer(obs, la, cfg, device=torch.device("cuda"))
    # Use a decent subset of observations for scoring
    obs_eval = ch._subsample_obs(min(1024, ch.n_obs), np.random.RandomState(42))

    t0 = time.time()

    # === Stage 1: Coarse grid over (Z1, Z2) ===
    z1_vals = np.arange(-3.0, 3.01, 0.15)
    z2_vals = np.arange(-3.0, 3.01, 0.15)
    z1_grid, z2_grid = np.meshgrid(z1_vals, z2_vals)
    z1_flat = z1_grid.ravel()
    z2_flat = z2_grid.ravel()
    n_coarse = len(z1_flat)

    coarse_coeffs = np.zeros((n_coarse, n_terms), dtype=np.float32)
    coarse_coeffs[:, 1] = z1_flat
    coarse_coeffs[:, 2] = z2_flat

    coarse_t = torch.tensor(coarse_coeffs, dtype=torch.float32, device="cuda")
    coarse_objs = []
    batch_sz = 512
    for s in range(0, n_coarse, batch_sz):
        e = min(s + batch_sz, n_coarse)
        with torch.no_grad():
            obj = ch._backward_chamfer_full(coarse_t[s:e], obs_eval)
        coarse_objs.append(obj)
    coarse_objs = torch.cat(coarse_objs)

    # Best coarse
    top16_k = min(16, n_coarse)
    _, top16_idx = coarse_objs.topk(top16_k, largest=False)
    coarse_best = coarse_t[top16_idx[0]].cpu().numpy()
    coarse_rmse = np.sqrt(np.mean((coarse_best - c_true) ** 2))

    t_coarse = time.time() - t0

    # === Stage 2: Fine grid around top-16 ===
    fine_candidates = []
    fine_step = 0.03
    fine_range = np.arange(-0.15, 0.151, fine_step)  # 11 values

    for ti in range(top16_k):
        base_z1 = coarse_coeffs[top16_idx[ti].item(), 1]
        base_z2 = coarse_coeffs[top16_idx[ti].item(), 2]
        for dz1 in fine_range:
            for dz2 in fine_range:
                c = np.zeros(n_terms, dtype=np.float32)
                c[1] = base_z1 + dz1
                c[2] = base_z2 + dz2
                fine_candidates.append(c)

    fine_np = np.stack(fine_candidates)
    fine_t_tensor = torch.tensor(fine_np, dtype=torch.float32, device="cuda")
    fine_objs = []
    for s in range(0, len(fine_candidates), 512):
        e = min(s + 512, len(fine_candidates))
        with torch.no_grad():
            obj = ch._backward_chamfer_full(fine_t_tensor[s:e], obs_eval)
        fine_objs.append(obj)
    fine_objs = torch.cat(fine_objs)

    fine_best_idx = fine_objs.argmin().item()
    fine_best = fine_candidates[fine_best_idx]
    fine_rmse = np.sqrt(np.mean((fine_best - c_true) ** 2))

    t_fine = time.time() - t0

    # === Stage 3: Adam refinement from top-64 fine candidates ===
    topk_adam = min(64, len(fine_objs))
    _, top_adam_idx = fine_objs.topk(topk_adam, largest=False)
    init_coeffs = fine_t_tensor[top_adam_idx].clone().detach().requires_grad_(True)

    # Use more observations for refinement
    obs_refine = ch._subsample_obs(min(2048, ch.n_obs), np.random.RandomState(123))

    opt = torch.optim.Adam([init_coeffs], lr=0.001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500, eta_min=0.0001)

    for it in range(500):
        opt.zero_grad()
        loss = ch._backward_chamfer_full(init_coeffs, obs_refine, differentiable=True)
        loss.sum().backward()
        torch.nn.utils.clip_grad_norm_([init_coeffs], max_norm=2.0)
        opt.step()
        sched.step()

    with torch.no_grad():
        final_loss = ch._backward_chamfer_full(init_coeffs, obs_refine)
        fbest = final_loss.argmin().item()
        c_final = init_coeffs[fbest].cpu().numpy()

    dt = time.time() - t0
    rmse_final = np.sqrt(np.mean((c_final - c_true) ** 2))

    print(
        f"PV={pv:5.1f} | coarse RMSE={coarse_rmse:.4f} ({t_coarse:.1f}s) | "
        f"fine RMSE={fine_rmse:.4f} ({t_fine:.1f}s) | "
        f"final RMSE={rmse_final:.4f} | n_fine={len(fine_candidates)} | time={dt:.1f}s"
    )
    print(f"  c_true  = {np.round(c_true, 4)}")
    print(f"  coarse  = {np.round(coarse_best, 4)}")
    print(f"  fine    = {np.round(fine_best, 4)}")
    print(f"  final   = {np.round(c_final, 4)}")
    print()

    torch.cuda.empty_cache()
