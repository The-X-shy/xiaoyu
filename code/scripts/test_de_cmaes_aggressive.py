"""Aggressive CMA-ES + Differential Evolution with strong count penalty.

Key improvements over previous attempt:
1. Much higher alpha (2.0) for count penalty
2. Larger population (128) and more iterations (500)
3. Try Differential Evolution (scipy) as alternative to CMA-ES
4. Bounds [-5, 5] per coefficient
5. Use BOTH CMA-ES restarts AND DE
6. The count penalty is: (n_inbounds - n_obs)^2 / n_obs^2 (quadratic)
"""

import torch, numpy as np, yaml, time
from src.sim.pipeline import forward_pipeline
from src.recon.chamfer_optimizer import ChamferOptimizer
from src.sim.lenslet import LensletArray
from scipy.optimize import differential_evolution

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


def make_objective(ch, obs, n_obs, alpha=2.0):
    """Create a GPU-accelerated objective function for scipy."""

    # Collect evaluations in batches for efficiency
    eval_cache = {}

    def objective_batch(coeffs_batch_np):
        """Evaluate batch of coefficient vectors."""
        coeffs_batch_np = np.array(coeffs_batch_np, dtype=np.float32)
        if coeffs_batch_np.ndim == 1:
            coeffs_batch_np = coeffs_batch_np.reshape(1, -1)
        coeffs_batch_np[:, 0] = 0.0  # piston = 0

        ct = torch.tensor(coeffs_batch_np, dtype=torch.float32, device="cuda")

        with torch.no_grad():
            # Backward Chamfer
            bwd = ch._backward_chamfer_full(ct, obs)  # (B,)

            # Count in-bounds
            E = ch._compute_expected(ct, ch.ref, ch.G, ch.n_sub)
            ib = (
                (E[:, :, 0] >= 0)
                & (E[:, :, 0] <= ch.sensor_w)
                & (E[:, :, 1] >= 0)
                & (E[:, :, 1] <= ch.sensor_h)
            )
            n_ib = ib.sum(dim=1).float()

            # Quadratic count penalty (stronger than linear)
            count_pen = ((n_ib - n_obs) / n_obs) ** 2

            total = bwd + alpha * count_pen

        return total.cpu().numpy()

    def objective_single(x):
        """Single evaluation for scipy."""
        return float(objective_batch(x.reshape(1, -1))[0])

    return objective_single, objective_batch


for pv in [5.0, 10.0, 15.0]:
    sim = forward_pipeline(cfg, pv=pv, seed=42)
    obs_np = sim["observed_positions"]
    c_true = sim["coeffs"]
    n_obs = len(obs_np)

    ch = ChamferOptimizer(obs_np, la, cfg, device=torch.device("cuda"))
    obs_eval = ch._subsample_obs(min(1024, ch.n_obs), np.random.RandomState(42))

    obj_single, obj_batch = make_objective(ch, obs_eval, n_obs, alpha=2.0)

    t0 = time.time()

    # Verify objective at truth vs zero
    obj_true = obj_single(c_true)
    obj_zero = obj_single(np.zeros(n_terms))
    print(
        f"\nPV={pv:.1f} | n_obs={n_obs} | obj@true={obj_true:.4f} | obj@zero={obj_zero:.4f}"
    )

    # Method 1: scipy Differential Evolution
    bounds = [(-5.0, 5.0)] * n_terms
    bounds[0] = (0.0, 0.0)  # piston fixed

    result_de = differential_evolution(
        obj_single,
        bounds,
        seed=42,
        maxiter=300,
        popsize=20,  # actual pop = 20 * 10 = 200
        mutation=(0.5, 1.5),
        recombination=0.9,
        tol=1e-8,
        disp=False,
        polish=False,
    )
    c_de = result_de.x.astype(np.float32)
    c_de[0] = 0.0
    rmse_de = np.sqrt(np.mean((c_de - c_true) ** 2))
    t_de = time.time() - t0

    print(
        f"  DE: rmse={rmse_de:.4f} | obj={result_de.fun:.4f} | nfev={result_de.nfev} | time={t_de:.1f}s"
    )
    print(f"    c_de = {np.round(c_de, 4)}")

    # Method 2: CMA-ES with aggressive settings
    import cma

    best_cma_obj = float("inf")
    best_cma_c = np.zeros(n_terms, dtype=np.float32)

    t_cma_start = time.time()
    for restart in range(8):
        sigma = 1.0 + 0.3 * restart
        x0 = np.random.RandomState(42 + restart * 7).randn(n_terms) * 0.5
        x0[0] = 0.0

        es = cma.CMAEvolutionStrategy(
            x0.tolist(),
            sigma,
            {
                "maxiter": 300,
                "popsize": 128,
                "bounds": [[-5.0] * n_terms, [5.0] * n_terms],
                "seed": 42 + restart,
                "verbose": -9,
                "CMA_active": True,
            },
        )

        for gen in range(300):
            if es.stop():
                break
            pop = es.ask()
            # Batch evaluate
            pop_np = np.array(pop, dtype=np.float32)
            pop_np[:, 0] = 0.0
            vals = obj_batch(pop_np)
            es.tell(pop, vals.tolist())

        res = es.result
        if res.fbest < best_cma_obj:
            best_cma_obj = res.fbest
            best_cma_c = np.array(res.xbest, dtype=np.float32)
            best_cma_c[0] = 0.0

    t_cma = time.time() - t_cma_start
    rmse_cma = np.sqrt(np.mean((best_cma_c - c_true) ** 2))
    print(f"  CMA: rmse={rmse_cma:.4f} | obj={best_cma_obj:.4f} | time={t_cma:.1f}s")
    print(f"    c_cma = {np.round(best_cma_c, 4)}")

    # Take the best overall and do Adam refinement
    if obj_single(best_cma_c) < obj_single(c_de):
        c_best = best_cma_c
        method = "CMA"
    else:
        c_best = c_de
        method = "DE"

    # Adam refinement
    obs_refine = ch._subsample_obs(min(2048, ch.n_obs), np.random.RandomState(123))
    init = torch.tensor(c_best, dtype=torch.float32, device="cuda").unsqueeze(0)
    perturbs = torch.randn(63, n_terms, device="cuda") * 0.05
    perturbs[:, 0] = 0.0
    init_batch = torch.cat([init, init + perturbs], dim=0).requires_grad_(True)

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
    rmse_final = np.sqrt(np.mean((c_final - c_true) ** 2))

    print(f"  Best={method} → Adam: rmse={rmse_final:.4f} | total time={dt:.1f}s")
    print(f"    c_final = {np.round(c_final, 4)}")
    print(f"    c_true  = {np.round(c_true, 4)}")

    torch.cuda.empty_cache()
