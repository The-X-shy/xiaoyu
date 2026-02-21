#!/usr/bin/env python3
"""Iterative refinement NN: learns to correct coefficient estimates.

Architecture:
  1. Current estimate → forward model → predicted spot positions
  2. Render predicted + observed spots into 2-channel 128x128 image
  3. CNN extracts residual features
  4. Combine with current estimate → predict correction Δcoeffs
  5. new_estimate = old_estimate + Δcoeffs
  6. Repeat K times (K=10 during training, more at inference)

The correction task is much easier than direct prediction:
each step only needs to make a small improvement.

Usage (on GPU server):
    PYTHONPATH=. python3 scripts/train_iterative_nn.py
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim.lenslet import LensletArray
from src.sim.wavefront import random_zernike_coeffs, scale_coeffs_to_pv
from src.recon.least_squares import build_zernike_slope_matrix
from src.recon.nn_warmstart import spots_to_image, ZernikeCNN, IMG_SIZE

# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------

# Image resolution (higher than v2's 64x64)
RES = 128

# Model
N_ITERS_TRAIN = 8  # iterations during training
N_ITERS_EVAL = 16  # iterations during evaluation
N_COEFFS = 9  # Zernike terms (skip piston)

# Training
N_TRAIN = 100000
N_VAL = 2000
BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 1e-4
N_EPOCHS = 40
PV_MIN = 0.5
PV_MAX = 20.0

# Output
MODEL_PATH = "models/nn_iterative.pt"


# ---------------------------------------------------------------------------
#  Spot rendering (fast, numpy-based)
# ---------------------------------------------------------------------------


def render_spots_128(
    positions: np.ndarray,
    sensor_w: float,
    sensor_h: float,
    res: int = RES,
) -> np.ndarray:
    """Render spots into a 1-channel density image at 128x128."""
    img = np.zeros((res, res), dtype=np.float32)
    if len(positions) == 0:
        return img
    xn = np.clip(positions[:, 0] / sensor_w, 0, 1 - 1e-9)
    yn = np.clip(positions[:, 1] / sensor_h, 0, 1 - 1e-9)
    ix = (xn * res).astype(int)
    iy = (yn * res).astype(int)
    np.clip(ix, 0, res - 1, out=ix)
    np.clip(iy, 0, res - 1, out=iy)
    for i in range(len(positions)):
        img[iy[i], ix[i]] += 1.0
    # Normalize
    expected = 13224.0 / (res * res)
    img /= max(expected, 1.0)
    return img


def render_spots_torch(
    positions: torch.Tensor,
    sensor_w: float,
    sensor_h: float,
    res: int = RES,
) -> torch.Tensor:
    """Differentiable-ish spot rendering (actually we just use hard binning)."""
    # For training we pre-render, so this is just for reference
    img = torch.zeros(res, res, dtype=torch.float32, device=positions.device)
    if positions.shape[0] == 0:
        return img
    xn = (positions[:, 0] / sensor_w).clamp(0, 1 - 1e-6)
    yn = (positions[:, 1] / sensor_h).clamp(0, 1 - 1e-6)
    ix = (xn * res).long().clamp(0, res - 1)
    iy = (yn * res).long().clamp(0, res - 1)
    # scatter_add for parallel
    flat_idx = iy * res + ix
    img_flat = img.view(-1)
    img_flat.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
    expected = 13224.0 / (res * res)
    return img_flat.view(res, res) / max(expected, 1.0)


# ---------------------------------------------------------------------------
#  Correction Network
# ---------------------------------------------------------------------------


class CorrectionCNN(nn.Module):
    """CNN that takes a 2-channel residual image + current coefficients
    and outputs a correction vector.

    Input: (B, 2, 128, 128) — channel 0: observed, channel 1: predicted
    + (B, 9) current coefficients

    Output: (B, 9) correction vector (delta coefficients)
    """

    def __init__(self, n_coeffs: int = N_COEFFS, res: int = RES):
        super().__init__()
        self.n_coeffs = n_coeffs

        # CNN for residual image processing
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, 5, stride=2, padding=2),  # 128->64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64->32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 32->16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),  # 16->8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, stride=2, padding=1),  # 8->4
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # 4->1
            nn.Flatten(),
        )

        # Combine image features with current coefficients
        self.head = nn.Sequential(
            nn.Linear(64 + n_coeffs, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_coeffs),
        )

        # Small init for correction output (start with small corrections)
        with torch.no_grad():
            self.head[-1].weight.mul_(0.01)
            self.head[-1].bias.zero_()

    def forward(self, img: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (B, 2, 128, 128) — observed and predicted density images
            coeffs: (B, 9) current coefficient estimate

        Returns:
            (B, 9) correction vector
        """
        feat = self.features(img)  # (B, 64)
        combined = torch.cat([feat, coeffs], dim=1)  # (B, 64+9)
        return self.head(combined)  # (B, 9)


# ---------------------------------------------------------------------------
#  Forward model (compute predicted positions from coefficients)
# ---------------------------------------------------------------------------


class ForwardModel:
    """Compute predicted spot positions from Zernike coefficients."""

    def __init__(self, lenslet: LensletArray, max_order: int):
        G_np = build_zernike_slope_matrix(lenslet, max_order=max_order, grid_size=128)
        ref_np = lenslet.reference_positions()
        self.G = G_np  # (2*N_sub, N_terms)
        self.ref = ref_np  # (N_sub, 2)
        self.n_sub = len(ref_np)
        self.focal_um = float(lenslet.focal_um)
        self.sensor_w = float(lenslet.sensor_width_um)
        self.sensor_h = float(lenslet.sensor_height_um)

    def predict_positions(self, coeffs: np.ndarray) -> np.ndarray:
        """Compute predicted spot positions for given coefficients.

        Args:
            coeffs: (N_terms,) full coefficient vector (with piston).

        Returns:
            (N_visible, 2) in-bounds predicted positions.
        """
        slopes = self.G @ coeffs  # (2*N_sub,)
        sx = slopes[: self.n_sub]
        sy = slopes[self.n_sub :]
        pred = self.ref.copy()
        pred[:, 0] += self.focal_um * sx
        pred[:, 1] += self.focal_um * sy

        # Filter in-bounds
        ib = (
            (pred[:, 0] >= 0)
            & (pred[:, 0] <= self.sensor_w)
            & (pred[:, 1] >= 0)
            & (pred[:, 1] <= self.sensor_h)
        )
        return pred[ib].astype(np.float32)


# ---------------------------------------------------------------------------
#  Data generation
# ---------------------------------------------------------------------------


def build_generator(cfg: dict):
    opt = cfg["optics"]
    sen = cfg["sensor"]
    noi = cfg["noise"]
    zer = cfg["zernike"]

    la = LensletArray(
        pitch_um=opt["pitch_um"],
        focal_mm=opt["focal_mm"],
        fill_factor=opt["fill_factor"],
        sensor_width_px=sen["width_px"],
        sensor_height_px=sen["height_px"],
        pixel_um=sen["pixel_um"],
    )
    ref = la.reference_positions()
    n_sub = len(ref)
    focal_um = la.focal_um
    sensor_w = sen["width_px"] * sen["pixel_um"]
    sensor_h = sen["height_px"] * sen["pixel_um"]
    G = build_zernike_slope_matrix(la, max_order=zer["order"], grid_size=128)
    centroid_noise_um = noi["centroid_noise_px"] * sen["pixel_um"]
    fwd = ForwardModel(la, max_order=zer["order"])
    return la, ref, G, n_sub, focal_um, sensor_w, sensor_h, centroid_noise_um, zer, fwd


def generate_sample(
    ref, G, n_sub, focal_um, sensor_w, sensor_h, centroid_noise_um, zer_cfg, rng
):
    """Generate one training sample.

    Returns: (observed_img_128, observed_positions, true_coeffs) or None
    """
    max_order = zer_cfg["order"]
    coeff_bound = zer_cfg["coeff_bound"]
    pv = np.exp(rng.uniform(np.log(PV_MIN), np.log(PV_MAX)))

    coeffs = random_zernike_coeffs(
        max_order=max_order, coeff_bound=coeff_bound, seed=int(rng.randint(0, 2**31))
    )
    coeffs = scale_coeffs_to_pv(coeffs, target_pv=pv, grid_size=128)

    slopes_flat = G @ coeffs
    sx = slopes_flat[:n_sub]
    sy = slopes_flat[n_sub:]
    displaced = ref.copy()
    displaced[:, 0] += focal_um * sx
    displaced[:, 1] += focal_um * sy

    noise = rng.randn(n_sub, 2).astype(np.float64) * centroid_noise_um
    observed = displaced + noise
    ib = (
        (observed[:, 0] >= 0)
        & (observed[:, 0] <= sensor_w)
        & (observed[:, 1] >= 0)
        & (observed[:, 1] <= sensor_h)
    )
    observed = observed[ib].astype(np.float32)
    if len(observed) < 20:
        return None

    obs_img = render_spots_128(observed, sensor_w, sensor_h)
    return obs_img, observed, coeffs


def generate_dataset(cfg, n_samples, seed, desc="Gen"):
    gen = build_generator(cfg)
    la, ref, G, n_sub, focal_um, sensor_w, sensor_h, centroid_noise_um, zer_cfg, fwd = (
        gen
    )
    rng = np.random.RandomState(seed)

    obs_imgs = []
    all_coeffs = []
    t0 = time.time()
    attempts = 0
    while len(obs_imgs) < n_samples:
        attempts += 1
        result = generate_sample(
            ref, G, n_sub, focal_um, sensor_w, sensor_h, centroid_noise_um, zer_cfg, rng
        )
        if result is not None:
            obs_img, _, coeffs = result
            obs_imgs.append(obs_img)
            all_coeffs.append(coeffs)
            if len(obs_imgs) % 10000 == 0:
                elapsed = time.time() - t0
                print(
                    f"  {desc}: {len(obs_imgs)}/{n_samples} ({len(obs_imgs) / elapsed:.0f}/s)",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(
        f"  {desc}: {n_samples} in {elapsed:.0f}s ({n_samples / elapsed:.0f}/s)",
        flush=True,
    )

    return np.stack(obs_imgs), np.stack(all_coeffs), fwd


# ---------------------------------------------------------------------------
#  Training with iterative refinement
# ---------------------------------------------------------------------------


def train_step(
    model: CorrectionCNN,
    fwd: ForwardModel,
    obs_imgs_batch: torch.Tensor,  # (B, 128, 128) observed density
    true_coeffs_batch: torch.Tensor,  # (B, 10) full coefficients
    device: torch.device,
    n_iters: int = N_ITERS_TRAIN,
):
    """One training step with iterative refinement.

    The loss is the sum of MSE at each iteration step, weighted so later
    iterations count more (curriculum: reward final accuracy most).
    """
    B = obs_imgs_batch.shape[0]
    sensor_w = fwd.sensor_w
    sensor_h = fwd.sensor_h

    # Target: coefficients 1-9 (skip piston)
    target = true_coeffs_batch[:, 1:].to(device)  # (B, 9)

    # Initialize estimate at zero
    current_est = torch.zeros(B, N_COEFFS, dtype=torch.float32, device=device)

    total_loss = torch.tensor(0.0, device=device)
    obs_imgs_dev = obs_imgs_batch.to(device)  # (B, 128, 128)

    for it in range(n_iters):
        # Render predicted spots for current estimate
        pred_imgs = torch.zeros(B, RES, RES, dtype=torch.float32, device=device)
        with torch.no_grad():
            for b in range(B):
                full_c = np.zeros(true_coeffs_batch.shape[1], dtype=np.float32)
                full_c[1:] = current_est[b].detach().cpu().numpy()
                pred_pos = fwd.predict_positions(full_c)
                if len(pred_pos) > 0:
                    pred_imgs[b] = render_spots_torch(
                        torch.tensor(pred_pos, device=device), sensor_w, sensor_h, RES
                    )

        # Stack into 2-channel image: (B, 2, 128, 128)
        img_input = torch.stack([obs_imgs_dev, pred_imgs], dim=1)

        # Predict correction
        delta = model(img_input, current_est)  # (B, 9)

        # Update estimate (with gradient)
        current_est = current_est + delta

        # Loss: MSE weighted by iteration (later = more important)
        weight = (it + 1) / n_iters  # linearly increasing weight
        step_loss = ((current_est - target) ** 2).mean() * weight
        total_loss = total_loss + step_loss

    return total_loss / n_iters, current_est


def evaluate(model, fwd, val_obs_imgs, val_coeffs, device, n_iters=N_ITERS_EVAL):
    """Evaluate model on validation set."""
    model.eval()
    B = len(val_obs_imgs)
    sensor_w = fwd.sensor_w
    sensor_h = fwd.sensor_h

    all_rmses = []
    batch_size = 128

    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        obs_batch = torch.tensor(val_obs_imgs[start:end], dtype=torch.float32)
        coeff_batch = val_coeffs[start:end]
        target = coeff_batch[:, 1:]  # (b, 9)

        b = end - start
        current_est = torch.zeros(b, N_COEFFS, dtype=torch.float32, device=device)

        with torch.no_grad():
            obs_dev = obs_batch.to(device)
            for it in range(n_iters):
                pred_imgs = torch.zeros(b, RES, RES, dtype=torch.float32, device=device)
                for i in range(b):
                    full_c = np.zeros(coeff_batch.shape[1], dtype=np.float32)
                    full_c[1:] = current_est[i].cpu().numpy()
                    pred_pos = fwd.predict_positions(full_c)
                    if len(pred_pos) > 0:
                        pred_imgs[i] = render_spots_torch(
                            torch.tensor(pred_pos, device=device),
                            sensor_w,
                            sensor_h,
                            RES,
                        )
                img_input = torch.stack([obs_dev, pred_imgs], dim=1)
                delta = model(img_input, current_est)
                current_est = current_est + delta

        pred_np = current_est.cpu().numpy()  # (b, 9)
        for i in range(b):
            rmse = np.sqrt(np.mean((pred_np[i] - target[i]) ** 2))
            all_rmses.append(rmse)

    rmses = np.array(all_rmses)
    return rmses.mean(), (rmses < 0.15).mean(), (rmses < 0.20).mean()


def evaluate_per_pv(model, fwd, cfg, device, n_per_pv=25):
    """Per-PV evaluation using forward_pipeline."""
    from src.sim.pipeline import forward_pipeline

    sensor_w = cfg["sensor"]["width_px"] * cfg["sensor"]["pixel_um"]
    sensor_h = cfg["sensor"]["height_px"] * cfg["sensor"]["pixel_um"]
    pv_levels = [1.0, 3.0, 5.0, 8.0, 10.0, 15.0]

    model.eval()
    print(
        f"\n{'PV':>5s} | {'RMSE':>8s} | {'<0.15':>6s} | {'<0.20':>6s} | {'nSpots':>6s}"
    )
    print("-" * 50)

    for pv in pv_levels:
        rmses = []
        nspots = []
        for i in range(n_per_pv):
            seed = 960000 + int(pv * 1000) + i
            try:
                result = forward_pipeline(cfg, pv=pv, seed=seed)
            except Exception:
                continue
            observed = result["observed_positions"]
            true_coeffs = result["coeffs"]
            if len(observed) < 10:
                continue

            # Render observed image
            obs_img = render_spots_128(observed, sensor_w, sensor_h)
            obs_t = torch.tensor(obs_img, dtype=torch.float32, device=device).unsqueeze(
                0
            )

            # Iterative refinement
            current_est = torch.zeros(1, N_COEFFS, dtype=torch.float32, device=device)
            with torch.no_grad():
                for it in range(N_ITERS_EVAL):
                    full_c = np.zeros_like(true_coeffs)
                    full_c[1:] = current_est[0].cpu().numpy()
                    pred_pos = fwd.predict_positions(full_c)
                    pred_img = torch.zeros(RES, RES, dtype=torch.float32, device=device)
                    if len(pred_pos) > 0:
                        pred_img = render_spots_torch(
                            torch.tensor(pred_pos, device=device),
                            sensor_w,
                            sensor_h,
                            RES,
                        )
                    img_input = torch.stack([obs_t.squeeze(0), pred_img]).unsqueeze(
                        0
                    )  # (1, 2, 128, 128)
                    delta = model(img_input, current_est)
                    current_est = current_est + delta

            pred_full = np.zeros_like(true_coeffs)
            pred_full[1:] = current_est[0].cpu().numpy()
            rmse = np.sqrt(np.mean((pred_full - true_coeffs) ** 2))
            rmses.append(rmse)
            nspots.append(len(observed))

        if rmses:
            rmses = np.array(rmses)
            print(
                f"{pv:5.1f} | {rmses.mean():8.4f} | "
                f"{100 * (rmses < 0.15).mean():5.1f}% | "
                f"{100 * (rmses < 0.20).mean():5.1f}% | "
                f"{np.mean(nspots):6.0f}"
            )


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main():
    import yaml

    print("=" * 70)
    print("Iterative Refinement NN Training")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = yaml.safe_load(open("configs/base_no_oracle.yaml"))

    # Generate data
    print(f"\nGenerating {N_TRAIN} training samples...")
    train_imgs, train_coeffs, fwd = generate_dataset(cfg, N_TRAIN, seed=0, desc="Train")
    print(f"Train: imgs={train_imgs.shape}, coeffs={train_coeffs.shape}")

    print(f"\nGenerating {N_VAL} validation samples...")
    val_imgs, val_coeffs, _ = generate_dataset(cfg, N_VAL, seed=500000, desc="Val")
    print(f"Val: imgs={val_imgs.shape}, coeffs={val_coeffs.shape}")

    # Create model
    model = CorrectionCNN(n_coeffs=N_COEFFS, res=RES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: CorrectionCNN, {total_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=LR * 0.01
    )

    # Training
    best_val_rmse = float("inf")
    best_state = None
    n_train = len(train_imgs)

    print(
        f"\n--- Training ({N_EPOCHS} epochs, {n_train} samples, batch={BATCH_SIZE}) ---"
    )

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        model.train()

        # Shuffle
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_train)
            idx = perm[start:end]
            if len(idx) < 4:
                continue

            obs_batch = torch.tensor(train_imgs[idx], dtype=torch.float32)
            coeff_batch = torch.tensor(train_coeffs[idx], dtype=torch.float32)

            loss, _ = train_step(model, fwd, obs_batch, coeff_batch, device)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_loss /= max(n_batches, 1)

        # Validate
        val_rmse, val_f015, val_f020 = evaluate(
            model, fwd, val_imgs, val_coeffs, device
        )
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{N_EPOCHS} | loss={epoch_loss:.6f} | "
            f"val_rmse={val_rmse:.4f} <0.15={val_f015 * 100:.1f}% <0.20={val_f020 * 100:.1f}% | "
            f"lr={lr:.2e} | {elapsed:.0f}s",
            end="",
            flush=True,
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  *BEST*", flush=True)
        else:
            print("", flush=True)

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    # Save
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "iterative_cnn",
            "n_coeffs": N_COEFFS,
            "res": RES,
            "n_iters_train": N_ITERS_TRAIN,
            "n_train": N_TRAIN,
        },
        MODEL_PATH,
    )
    print(f"\nModel saved to {MODEL_PATH}")

    # Final evaluation
    print("\n--- Final Per-PV Evaluation ---")
    evaluate_per_pv(model, fwd, cfg, device, n_per_pv=25)

    print("\nDone!")


if __name__ == "__main__":
    main()
