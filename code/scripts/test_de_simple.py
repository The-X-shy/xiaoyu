"""Test Differential Evolution with count-penalty Chamfer objective.
Uses scipy DE with vectorized GPU evaluation."""

import torch, numpy as np, yaml, time, sys
from src.sim.pipeline import forward_pipeline
from src.recon.chamfer_optimizer import ChamferOptimizer
from src.sim.lenslet import LensletArray

torch.cuda.empty_cache()

with open("configs/base_no_oracle.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["asm"]["chamfer_pred_chunk"] = 512
cfg["asm"]["chamfer_lambda_reg"] = 0.0  # disable reg, let count penalty do the work

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
n_terms = 10  # D=10, but D=0 is piston=0, so we optimize 9 dims


def make_objective(ch, obs, n_obs, alpha=2.0):
    """Objective over 9 dims (skip piston)."""
    call_count = [0]

    def objective_9d(x):
        """x is 9-dimensional (no piston)."""
        call_count[0] += 1
        c = np.zeros(n_terms, dtype=np.float32)
        c[1:] = np.array(x, dtype=np.float32)
        ct = torch.tensor(c.reshape(1, -1), dtype=torch.float32, device="cuda")

        with torch.no_grad():
            bwd = ch._backward_chamfer_full(ct, obs)
            E = ch._compute_expected(ct, ch.ref, ch.G, ch.n_sub)
            ib = (
                (E[:, :, 0] >= 0)
                & (E[:, :, 0] <= ch.sensor_w)
                & (E[:, :, 1] >= 0)
                & (E[:, :, 1] <= ch.sensor_h)
            )
            n_ib = ib.sum(dim=1).float()
            count_pen = ((n_ib - n_obs) / n_obs) ** 2
            total = bwd + alpha * count_pen

        return float(total.item())

    return objective_9d, call_count


for pv in [5.0, 10.0, 15.0]:
    sim = forward_pipeline(cfg, pv=pv, seed=42)
    obs_np = sim["observed_positions"]
    c_true = sim["coeffs"]
    n_obs = len(obs_np)

    ch = ChamferOptimizer(obs_np, la, cfg, device=torch.device("cuda"))
    obs_eval = ch._subsample_obs(min(512, ch.n_obs), np.random.RandomState(42))

    obj_fn, call_count = make_objective(ch, obs_eval, n_obs, alpha=2.0)

    t0 = time.time()

    # Verify objective at truth vs zero
    c_true_9 = c_true[1:]
    c_zero_9 = np.zeros(9)
    obj_true = obj_fn(c_true_9)
    obj_zero = obj_fn(c_zero_9)
    print(
        f"\nPV={pv:.1f} | n_obs={n_obs} | obj@true={obj_true:.4f} | obj@zero={obj_zero:.4f}"
    )
    sys.stdout.flush()

    # Differential Evolution: 9 dims, bounds [-5, 5]
    from scipy.optimize import differential_evolution

    bounds_9d = [(-5.0, 5.0)] * 9

    result_de = differential_evolution(
        obj_fn,
        bounds_9d,
        seed=42,
        maxiter=200,
        popsize=15,  # actual pop = 15 * 9 = 135
        mutation=(0.5, 1.5),
        recombination=0.9,
        tol=1e-8,
        disp=False,
        polish=False,
    )
    c_de = np.zeros(n_terms, dtype=np.float32)
    c_de[1:] = result_de.x.astype(np.float32)
    rmse_de = np.sqrt(np.mean((c_de - c_true) ** 2))
    t_de = time.time() - t0

    print(
        f"  DE: rmse={rmse_de:.4f} | obj={result_de.fun:.4f} | nfev={call_count[0]} | time={t_de:.1f}s"
    )
    print(f"    c_de   = {np.round(c_de, 4)}")
    print(f"    c_true = {np.round(c_true, 4)}")
    sys.stdout.flush()

    # Adam refinement from DE result
    obs_refine = ch._subsample_obs(min(2048, ch.n_obs), np.random.RandomState(123))
    init = torch.tensor(c_de, dtype=torch.float32, device="cuda").unsqueeze(0)
    perturbs = torch.randn(63, n_terms, device="cuda") * 0.05
    perturbs[:, 0] = 0.0
    init_batch = torch.cat([init, init + perturbs], dim=0).requires_grad_(True)

    opt = torch.optim.Adam([init_batch], lr=0.001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300, eta_min=0.0001)

    for it in range(300):
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
    print(f"  DE+Adam: rmse={rmse_final:.4f} | total time={dt:.1f}s")
    print(f"    c_final = {np.round(c_final, 4)}")
    sys.stdout.flush()

    torch.cuda.empty_cache()
