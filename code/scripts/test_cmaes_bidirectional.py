"""CMA-ES with bidirectional Chamfer + count penalty objective.

Uses the count-aware objective:
  loss = backward_chamfer + alpha * |n_inbounds_pred - n_obs| / n_obs

This penalizes the zero solution much more heavily at high PV,
potentially making the landscape searchable.

Strategy: multiple CMA-ES restarts with different initial sigma and centers.
"""

import torch, numpy as np, yaml, time, cma
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


def bidirectional_chamfer_batch(ch, coeffs_batch_t, obs, alpha=1.0):
    """Compute backward Chamfer + count penalty for a batch."""
    B = coeffs_batch_t.shape[0]
    k_obs = obs.shape[0]
    n_obs_int = k_obs

    # Backward Chamfer (existing)
    bwd = ch._backward_chamfer_full(coeffs_batch_t, obs)  # (B,)

    # Count in-bounds predictions
    E = ch._compute_expected(coeffs_batch_t, ch.ref, ch.G, ch.n_sub)
    ib = (
        (E[:, :, 0] >= 0)
        & (E[:, :, 0] <= ch.sensor_w)
        & (E[:, :, 1] >= 0)
        & (E[:, :, 1] <= ch.sensor_h)
    )
    n_ib = ib.sum(dim=1).float()  # (B,)

    count_pen = torch.abs(n_ib - n_obs_int) / n_obs_int

    return bwd + alpha * count_pen


def evaluate_cma_population(ch, pop, obs, alpha=1.0):
    """Evaluate a CMA-ES population on GPU."""
    pop_np = np.array(pop, dtype=np.float32)
    pop_np[:, 0] = 0.0  # piston always 0
    pop_t = torch.tensor(pop_np, dtype=torch.float32, device="cuda")

    batch_sz = 256
    all_objs = []
    for s in range(0, len(pop), batch_sz):
        e = min(s + batch_sz, len(pop))
        with torch.no_grad():
            obj = bidirectional_chamfer_batch(ch, pop_t[s:e], obs, alpha=alpha)
        all_objs.append(obj.cpu().numpy())
    return np.concatenate(all_objs)


for pv in [5.0, 10.0, 15.0]:
    sim = forward_pipeline(cfg, pv=pv, seed=42)
    obs = sim["observed_positions"]
    c_true = sim["coeffs"]
    n_obs = len(obs)

    ch = ChamferOptimizer(obs, la, cfg, device=torch.device("cuda"))
    obs_eval = ch._subsample_obs(min(1024, ch.n_obs), np.random.RandomState(42))

    t0 = time.time()

    # Alpha scales with PV (higher PV = more count mismatch to exploit)
    alpha = 0.5

    best_obj = float("inf")
    best_c = np.zeros(n_terms, dtype=np.float32)

    # Multiple CMA-ES restarts
    n_restarts = 5
    for restart in range(n_restarts):
        # Different starting points: center=0, varied sigma
        sigma = 1.0 + 0.5 * restart
        x0 = np.random.RandomState(42 + restart).randn(n_terms) * 0.1
        x0[0] = 0.0

        es = cma.CMAEvolutionStrategy(
            x0.tolist(),
            sigma,
            {
                "maxiter": 200,
                "popsize": 64,
                "bounds": [[-5.0] * n_terms, [5.0] * n_terms],
                "seed": 42 + restart,
                "verbose": -9,
                "CMA_active": True,
            },
        )

        for gen in range(200):
            if es.stop():
                break
            pop = es.ask()
            vals = evaluate_cma_population(ch, pop, obs_eval, alpha=alpha)
            es.tell(pop, vals.tolist())

        res = es.result
        c_res = np.array(res.xbest, dtype=np.float32)
        c_res[0] = 0.0
        obj_res = res.fbest

        if obj_res < best_obj:
            best_obj = obj_res
            best_c = c_res.copy()

    # Stage 2: Adam refinement from best CMA result
    init = torch.tensor(best_c, dtype=torch.float32, device="cuda").unsqueeze(0)
    # Also try a small perturbation set
    perturbs = torch.randn(31, n_terms, device="cuda") * 0.1
    perturbs[:, 0] = 0.0
    init_batch = torch.cat([init, init + perturbs], dim=0).requires_grad_(True)

    obs_refine = ch._subsample_obs(min(2048, ch.n_obs), np.random.RandomState(123))
    opt = torch.optim.Adam([init_batch], lr=0.001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500, eta_min=0.0001)

    for it in range(500):
        opt.zero_grad()
        loss = ch._backward_chamfer_full(init_batch, obs_refine, differentiable=True)
        loss.sum().backward()
        torch.nn.utils.clip_grad_norm_([init_batch], max_norm=2.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            init_batch[:, 0] = 0.0

    with torch.no_grad():
        final_loss = ch._backward_chamfer_full(init_batch, obs_refine)
        fbest = final_loss.argmin().item()
        c_final = init_batch[fbest].cpu().numpy()

    dt = time.time() - t0
    rmse_cma = np.sqrt(np.mean((best_c - c_true) ** 2))
    rmse_final = np.sqrt(np.mean((c_final - c_true) ** 2))

    print(
        f"PV={pv:5.1f} | CMA RMSE={rmse_cma:.4f} | final RMSE={rmse_final:.4f} | "
        f"n_obs={n_obs} | time={dt:.1f}s"
    )
    print(f"  c_true = {np.round(c_true, 4)}")
    print(f"  c_cma  = {np.round(best_c, 4)}")
    print(f"  c_final= {np.round(c_final, 4)}")
    print()

    torch.cuda.empty_cache()
