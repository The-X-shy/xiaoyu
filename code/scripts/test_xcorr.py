"""Cross-correlation based displacement field estimation.

Strategy:
1. Render observed spots and reference spots as 2D images (Gaussian blobs)
2. Cross-correlate local windows to estimate local displacement
3. Fit Zernike coefficients to displacement field
4. Use as warm-start for Adam/Chamfer refinement

Key insight: Even though individual spot matches are impossible,
the STATISTICAL displacement pattern is detectable via cross-correlation.
"""

import torch, numpy as np, yaml, time, sys
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
ref = la.reference_positions()
n_sub = len(ref)
n_terms = 10
pitch = oc["pitch_um"]
focal = oc["focal_mm"] * 1000.0
sensor_w = sc["width_px"] * sc["pixel_um"]
sensor_h = sc["height_px"] * sc["pixel_um"]

Gx = G[:n_sub, :]  # (n_sub, n_terms)
Gy = G[n_sub:, :]


def render_spots(positions, sensor_w, sensor_h, bin_size=50.0):
    """Render spot positions as a 2D histogram."""
    nx = int(sensor_w / bin_size)
    ny = int(sensor_h / bin_size)
    img = np.zeros((ny, nx), dtype=np.float32)
    for x, y in positions:
        ix = int(x / bin_size)
        iy = int(y / bin_size)
        if 0 <= ix < nx and 0 <= iy < ny:
            img[iy, ix] += 1.0
    return img


def cross_correlate_2d(img1, img2, max_shift):
    """Cross-correlation of two 2D images, return (dy, dx) of peak."""
    from scipy.signal import fftconvolve

    # Cross-correlation via FFT
    cc = fftconvolve(img1, img2[::-1, ::-1], mode="full")
    cy, cx = np.array(cc.shape) // 2
    # Search within max_shift
    ms = max_shift
    region = cc[cy - ms : cy + ms + 1, cx - ms : cx + ms + 1]
    if region.size == 0:
        return 0.0, 0.0
    peak = np.unravel_index(np.argmax(region), region.shape)
    dy = peak[0] - ms
    dx = peak[1] - ms
    return dy, dx


for pv in [1.0, 5.0, 10.0, 15.0]:
    sim = forward_pipeline(cfg, pv=pv, seed=42)
    obs = sim["observed_positions"]
    c_true = sim["coeffs"]
    n_obs = len(obs)

    t0 = time.time()

    # Render reference and observed spots
    bin_size = pitch  # each bin is one pitch = 150 um
    ref_img = render_spots(ref, sensor_w, sensor_h, bin_size)
    obs_img = render_spots(obs, sensor_w, sensor_h, bin_size)

    # Global cross-correlation to estimate overall shift
    max_shift_bins = 100  # up to 100 pitches of shift
    dy_bins, dx_bins = cross_correlate_2d(obs_img, ref_img, max_shift_bins)
    global_dx = dx_bins * bin_size
    global_dy = dy_bins * bin_size
    print(f"\nPV={pv:.1f} | n_obs={n_obs}")
    print(
        f"  Global shift: dx={global_dx:.1f} um ({dx_bins} bins), dy={global_dy:.1f} um ({dy_bins} bins)"
    )

    # Windowed cross-correlation for local displacements
    # Divide sensor into nw x nw windows
    nw = 6  # 6x6 grid of windows
    win_w = sensor_w / nw
    win_h = sensor_h / nw

    window_centers = []
    window_displacements = []

    for wi in range(nw):
        for wj in range(nw):
            x0 = wj * win_w
            y0 = wi * win_h
            x1 = x0 + win_w
            y1 = y0 + win_h

            # Spots in this window
            ref_mask = (
                (ref[:, 0] >= x0)
                & (ref[:, 0] < x1)
                & (ref[:, 1] >= y0)
                & (ref[:, 1] < y1)
            )
            obs_mask = (
                (obs[:, 0] >= x0)
                & (obs[:, 0] < x1)
                & (obs[:, 1] >= y0)
                & (obs[:, 1] < y1)
            )

            n_ref_w = ref_mask.sum()
            n_obs_w = obs_mask.sum()

            if n_obs_w < 3 or n_ref_w < 3:
                continue

            # Render local images (positions relative to window)
            ref_local = ref[ref_mask] - np.array([x0, y0])
            obs_local = obs[obs_mask] - np.array([x0, y0])

            local_ref_img = render_spots(ref_local, win_w, win_h, bin_size)
            local_obs_img = render_spots(obs_local, win_w, win_h, bin_size)

            if local_ref_img.sum() < 2 or local_obs_img.sum() < 2:
                continue

            ms = min(50, min(local_ref_img.shape) - 1)
            if ms < 1:
                continue
            dy_l, dx_l = cross_correlate_2d(local_obs_img, local_ref_img, ms)

            window_centers.append([(x0 + x1) / 2, (y0 + y1) / 2])
            window_displacements.append([dx_l * bin_size, dy_l * bin_size])

    if len(window_centers) < 3:
        print(f"  Too few valid windows ({len(window_centers)}), skipping")
        continue

    window_centers = np.array(window_centers)
    window_displacements = np.array(window_displacements)
    n_win = len(window_centers)

    print(f"  Valid windows: {n_win}/36")
    print(
        f"  Displacement range: x=[{window_displacements[:, 0].min():.0f}, {window_displacements[:, 0].max():.0f}] "
        f"y=[{window_displacements[:, 1].min():.0f}, {window_displacements[:, 1].max():.0f}] um"
    )

    # Now fit Zernike coefficients to the displacement field
    # For each window center, find the nearest reference subaperture
    from scipy.spatial import cKDTree

    tree = cKDTree(ref)
    _, nearest_ref = tree.query(window_centers)

    # Build the linear system:
    # displacement_x[i] = focal * Gx[nearest_ref[i], :] @ c
    # displacement_y[i] = focal * Gy[nearest_ref[i], :] @ c
    A = np.zeros((2 * n_win, n_terms), dtype=np.float64)
    b = np.zeros(2 * n_win, dtype=np.float64)

    for i in range(n_win):
        ri = nearest_ref[i]
        A[i, :] = focal * Gx[ri, :]
        A[n_win + i, :] = focal * Gy[ri, :]
        b[i] = window_displacements[i, 0]
        b[n_win + i] = window_displacements[i, 1]

    # Solve via least squares
    c_est, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    c_est = c_est.astype(np.float32)
    c_est[0] = 0.0  # piston = 0

    rmse_xcorr = np.sqrt(np.mean((c_est - c_true) ** 2))
    t_xcorr = time.time() - t0
    print(f"  Cross-corr estimate: rmse={rmse_xcorr:.4f} | time={t_xcorr:.1f}s")
    print(f"    c_est  = {np.round(c_est, 4)}")
    print(f"    c_true = {np.round(c_true, 4)}")

    # Adam refinement
    ch = ChamferOptimizer(obs, la, cfg, device=torch.device("cuda"))
    obs_refine = ch._subsample_obs(min(2048, ch.n_obs), np.random.RandomState(123))
    init = torch.tensor(c_est, dtype=torch.float32, device="cuda").unsqueeze(0)
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
    print(f"  Xcorr+Adam: rmse={rmse_final:.4f} | total time={dt:.1f}s")
    print(f"    c_final = {np.round(c_final, 4)}")
    sys.stdout.flush()

    torch.cuda.empty_cache()
