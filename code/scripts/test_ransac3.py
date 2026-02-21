import torch, numpy as np, yaml, time
from src.sim.pipeline import forward_pipeline
from src.recon.chamfer_optimizer import ChamferOptimizer
from src.sim.lenslet import LensletArray
from src.recon.least_squares import build_zernike_slope_matrix

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
G = build_zernike_slope_matrix(la, cfg["zernike"]["order"])
focal = oc["focal_mm"] * 1000.0
ref = la.reference_positions()
n_sub = len(ref)
n_terms = 10

Gx = G[:n_sub, :]
Gy = G[n_sub:, :]

for pv in [1.0, 5.0, 10.0, 15.0]:
    sim = forward_pipeline(cfg, pv=pv, seed=42)
    obs = sim["observed_positions"]
    c_true = sim["coeffs"]

    ch = ChamferOptimizer(obs, la, cfg, device=torch.device("cuda"))
    obs_sub = ch._subsample_obs(min(512, ch.n_obs), np.random.RandomState(42))

    rng = np.random.RandomState(42)
    n_obs = len(obs)

    t0 = time.time()

    k_nn = 300
    obs_t = torch.tensor(obs, dtype=torch.float32, device="cuda")
    ref_t = torch.tensor(ref, dtype=torch.float32, device="cuda")

    nn_indices = []
    batch_sz = 500
    for s in range(0, n_obs, batch_sz):
        e = min(s + batch_sz, n_obs)
        d = torch.cdist(obs_t[s:e].unsqueeze(0), ref_t.unsqueeze(0))[0]
        _, idx = d.topk(k_nn, largest=False, dim=1)
        nn_indices.append(idx.cpu().numpy())
    nn_indices = np.concatenate(nn_indices, axis=0)

    # Stage 1: single-match RANSAC for tip/tilt
    candidates = []
    n_obs_try = min(20, n_obs)
    obs_try = rng.choice(n_obs, size=n_obs_try, replace=False)

    for oi in obs_try:
        for ki in range(k_nn):
            ri = nn_indices[oi, ki]
            dx = obs[oi, 0] - ref[ri, 0]
            dy = obs[oi, 1] - ref[ri, 1]

            A = np.array(
                [
                    [focal * Gx[ri, 1], focal * Gx[ri, 2]],
                    [focal * Gy[ri, 1], focal * Gy[ri, 2]],
                ]
            )
            b_vec = np.array([dx, dy])

            try:
                c12 = np.linalg.solve(A, b_vec)
            except:
                continue

            c_hyp = np.zeros(n_terms, dtype=np.float32)
            c_hyp[1] = c12[0]
            c_hyp[2] = c12[1]

            if np.linalg.norm(c_hyp) > 15:
                continue
            candidates.append(c_hyp)

    cands_np = np.stack(candidates)
    cands_t = torch.tensor(cands_np, dtype=torch.float32, device="cuda")

    all_objs = []
    for s in range(0, len(candidates), 512):
        e = min(s + 512, len(candidates))
        with torch.no_grad():
            obj = ch._backward_chamfer_full(cands_t[s:e], obs_sub)
        all_objs.append(obj)
    all_objs = torch.cat(all_objs)

    topk_n = min(64, len(all_objs))
    _, top_idx = all_objs.topk(topk_n, largest=False)
    top_c = cands_t[top_idx]

    best_idx = top_idx[0].item()
    c_stage1 = candidates[best_idx]
    rmse_s1 = np.sqrt(np.mean((c_stage1 - c_true) ** 2))

    # Stage 2: Adam refinement of top-64 over ALL terms
    coeffs = top_c.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([coeffs], lr=0.002)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500, eta_min=0.0002)

    for it in range(500):
        opt.zero_grad()
        loss = ch._backward_chamfer_full(coeffs, obs_sub, differentiable=True)
        loss.sum().backward()
        torch.nn.utils.clip_grad_norm_([coeffs], max_norm=2.0)
        opt.step()
        sched.step()

    with torch.no_grad():
        final_loss = ch._backward_chamfer_full(coeffs, obs_sub)
        fbest = final_loss.argmin().item()
        c_final = coeffs[fbest].cpu().numpy()

    dt = time.time() - t0
    rmse_final = np.sqrt(np.mean((c_final - c_true) ** 2))

    print(
        f"PV={pv:5.1f} | stage1(TT) RMSE={rmse_s1:.4f} | final RMSE={rmse_final:.4f} | n_cands={len(candidates)} | time={dt:.1f}s"
    )
    print(f"  c_true = {np.round(c_true, 4)}")
    print(f"  stage1 = {np.round(c_stage1, 4)}")
    print(f"  final  = {np.round(c_final, 4)}")
    print()

    torch.cuda.empty_cache()
