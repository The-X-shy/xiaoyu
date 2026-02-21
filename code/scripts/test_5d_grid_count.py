"""5D coarse grid search over order-1+2 Zernike terms (Z1..Z4, skip Z0=piston),
using COUNT PENALTY as primary objective (cheap: no cdist needed).

Then refine top candidates with full Chamfer + count + Adam.

Strategy:
  Stage 1: 5D grid search for (Z1,Z2,Z3,Z4,Z5) using count match only
            Step=0.3, range=[-3,3] → 20^5 = 3.2M candidates (but we can reduce)
            Actually: use coarser grid first, then refine
  Stage 1a: 5D grid step=0.5, range=[-3,3] → 12^5 = 248K candidates
  Stage 1b: Fine grid around top-100 with step=0.1
  Stage 2: Evaluate top-1000 with full bwd+count Chamfer
  Stage 3: Adam refinement from top-64
"""

import torch, numpy as np, yaml, time, sys
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
    obs_np = sim["observed_positions"]
    c_true = sim["coeffs"]
    n_obs = len(obs_np)

    ch = ChamferOptimizer(obs_np, la, cfg, device=torch.device("cuda"))

    t0 = time.time()

    # === Stage 1a: Coarse 5D grid, count penalty only ===
    # Only count penalty - very cheap, no cdist needed
    vals = np.arange(-3.0, 3.01, 0.5)  # 13 values
    # 5D grid: 13^5 = 371,293 candidates
    grids = np.meshgrid(*([vals] * 5), indexing="ij")
    grid_5d = np.stack([g.ravel() for g in grids], axis=1).astype(np.float32)
    n_coarse = len(grid_5d)
    print(f"\nPV={pv:.1f} | n_obs={n_obs} | n_coarse_candidates={n_coarse}")
    sys.stdout.flush()

    # Build full 10D candidates (zeros for Z5..Z9)
    coarse_10d = np.zeros((n_coarse, n_terms), dtype=np.float32)
    coarse_10d[:, 1:6] = grid_5d  # Z1,Z2,Z3,Z4,Z5

    # Evaluate COUNT penalty in batches (cheap: just matmul + bounds check)
    coarse_t = torch.tensor(coarse_10d, dtype=torch.float32, device="cuda")
    count_errors = []
    batch_sz = 4096

    for s in range(0, n_coarse, batch_sz):
        e = min(s + batch_sz, n_coarse)
        batch = coarse_t[s:e]
        with torch.no_grad():
            E = ch._compute_expected(batch, ch.ref, ch.G, ch.n_sub)
            ib = (
                (E[:, :, 0] >= 0)
                & (E[:, :, 0] <= ch.sensor_w)
                & (E[:, :, 1] >= 0)
                & (E[:, :, 1] <= ch.sensor_h)
            )
            n_ib = ib.sum(dim=1).float()
            ce = ((n_ib - n_obs) / n_obs) ** 2
        count_errors.append(ce.cpu())

    count_errors = torch.cat(count_errors).numpy()

    t_coarse = time.time() - t0
    print(
        f"  Stage 1a: {t_coarse:.1f}s | count_err min={count_errors.min():.4f} @ idx={count_errors.argmin()}"
    )
    sys.stdout.flush()

    # Find the count-error for the truth
    c_true_5d = c_true[1:6]
    true_idx = np.argmin(np.sum((grid_5d - c_true_5d) ** 2, axis=1))
    print(
        f"  Truth nearest grid: count_err={count_errors[true_idx]:.4f} at {grid_5d[true_idx]}"
    )

    # Top-100 by count error
    top100_idx = np.argsort(count_errors)[:200]
    top100 = coarse_10d[top100_idx]

    # === Stage 1b: Fine grid around top-50 ===
    fine_range = np.arange(-0.3, 0.31, 0.15).astype(np.float32)  # 5 values per dim
    fine_grids = np.meshgrid(*([fine_range] * 5), indexing="ij")
    fine_offsets = np.stack([g.ravel() for g in fine_grids], axis=1)  # (3125, 5)

    n_centers = min(50, len(top100))
    fine_candidates = []
    for ci in range(n_centers):
        center = top100[ci : ci + 1].copy()  # (1, 10)
        expanded = np.tile(center, (len(fine_offsets), 1))  # (3125, 10)
        expanded[:, 1:6] += fine_offsets
        fine_candidates.append(expanded)

    fine_np = np.concatenate(fine_candidates, axis=0)
    n_fine = len(fine_np)
    print(f"  Stage 1b: n_fine={n_fine}")
    sys.stdout.flush()

    fine_t = torch.tensor(fine_np, dtype=torch.float32, device="cuda")
    fine_count_err = []
    for s in range(0, n_fine, batch_sz):
        e = min(s + batch_sz, n_fine)
        batch = fine_t[s:e]
        with torch.no_grad():
            E = ch._compute_expected(batch, ch.ref, ch.G, ch.n_sub)
            ib = (
                (E[:, :, 0] >= 0)
                & (E[:, :, 0] <= ch.sensor_w)
                & (E[:, :, 1] >= 0)
                & (E[:, :, 1] <= ch.sensor_h)
            )
            n_ib = ib.sum(dim=1).float()
            ce = ((n_ib - n_obs) / n_obs) ** 2
        fine_count_err.append(ce.cpu())
    fine_count_err = torch.cat(fine_count_err).numpy()

    t_fine = time.time() - t0
    print(f"  Stage 1b: {t_fine:.1f}s | fine count_err min={fine_count_err.min():.4f}")
    sys.stdout.flush()

    # === Stage 2: Evaluate top-1000 with full bwd+count Chamfer ===
    top1k_idx = np.argsort(fine_count_err)[:1000]
    top1k = fine_t[top1k_idx]

    obs_eval = ch._subsample_obs(min(512, ch.n_obs), np.random.RandomState(42))
    full_objs = []
    for s in range(0, len(top1k), 256):
        e = min(s + 256, len(top1k))
        with torch.no_grad():
            bwd = ch._backward_chamfer_full(top1k[s:e], obs_eval)
            E = ch._compute_expected(top1k[s:e], ch.ref, ch.G, ch.n_sub)
            ib = (
                (E[:, :, 0] >= 0)
                & (E[:, :, 0] <= ch.sensor_w)
                & (E[:, :, 1] >= 0)
                & (E[:, :, 1] <= ch.sensor_h)
            )
            n_ib = ib.sum(dim=1).float()
            cp = ((n_ib - n_obs) / n_obs) ** 2
            total = bwd + 2.0 * cp
        full_objs.append(total.cpu())
    full_objs = torch.cat(full_objs).numpy()

    best_stage2_idx = full_objs.argmin()
    c_stage2 = top1k[best_stage2_idx].cpu().numpy()
    rmse_stage2 = np.sqrt(np.mean((c_stage2 - c_true) ** 2))
    t_stage2 = time.time() - t0
    print(
        f"  Stage 2: {t_stage2:.1f}s | obj={full_objs[best_stage2_idx]:.4f} | rmse={rmse_stage2:.4f}"
    )
    print(f"    c_stage2 = {np.round(c_stage2, 4)}")
    sys.stdout.flush()

    # === Stage 3: Adam refinement from top-64 ===
    top64_idx = np.argsort(full_objs)[:64]
    init = top1k[top64_idx].clone().detach().requires_grad_(True)

    obs_refine = ch._subsample_obs(min(2048, ch.n_obs), np.random.RandomState(123))
    opt = torch.optim.Adam([init], lr=0.001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500, eta_min=0.0001)

    for it in range(500):
        opt.zero_grad()
        loss = ch._backward_chamfer_full(init, obs_refine, differentiable=True)
        loss.sum().backward()
        torch.nn.utils.clip_grad_norm_([init], max_norm=2.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            init[:, 0] = 0.0

    with torch.no_grad():
        final_loss = ch._backward_chamfer_full(init, obs_refine)
        fbest = final_loss.argmin().item()
        c_final = init[fbest].cpu().numpy()

    dt = time.time() - t0
    rmse_final = np.sqrt(np.mean((c_final - c_true) ** 2))
    print(f"  Stage 3: Adam → rmse={rmse_final:.4f} | total time={dt:.1f}s")
    print(f"    c_final = {np.round(c_final, 4)}")
    print(f"    c_true  = {np.round(c_true, 4)}")
    sys.stdout.flush()

    torch.cuda.empty_cache()
