#!/usr/bin/env python3
"""Fine-tune NN warm-start model with differentiable Chamfer loss.

Strategy: Start from MSE-pretrained CNN, fine-tune with end-to-end
differentiable Chamfer loss through the physical forward model:
  spots → CNN → coefficients → G-matrix → predicted_positions → Chamfer(pred, obs)

This teaches the network to predict coefficients that produce the right
spot pattern, rather than matching coefficient values directly.

Usage (on GPU server):
    PYTHONPATH=. python3 scripts/train_nn_chamfer.py
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim.lenslet import LensletArray
from src.sim.wavefront import random_zernike_coeffs, scale_coeffs_to_pv
from src.recon.least_squares import build_zernike_slope_matrix
from src.recon.nn_warmstart import (
    spots_to_image,
    ZernikeCNN,
    IMG_SIZE,
)

# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------

# Fine-tuning from pretrained model
PRETRAINED_PATH = "models/nn_warmstart.pt"
OUTPUT_PATH = "models/nn_warmstart_chamfer.pt"

# Data generation
N_TRAIN_BATCHES = 4000  # number of training iterations
N_VAL = 500  # validation samples (MSE evaluated)
BATCH_SIZE = 32  # smaller batch for Chamfer (memory-heavy)
PV_MIN = 2.0  # focus on high-PV where MSE fails
PV_MAX = 20.0

# Chamfer forward model config
PRED_CHUNK = 4096  # chunk predictions for cdist
OBS_SUBSAMPLE = 256  # subsample observations per sample
PITCH_NORMALIZE = True  # normalize distances by pitch

# Training
LR = 2e-4  # small LR for fine-tuning
WEIGHT_DECAY = 1e-4
CHAMFER_WEIGHT = 1.0  # weight for Chamfer loss
MSE_WEIGHT = 0.1  # regularizer: keep predictions not insane
EVAL_EVERY = 200  # evaluate every N batches
SAVE_EVERY = 500
GRAD_CLIP = 2.0

# ---------------------------------------------------------------------------
#  Physics-based differentiable forward model
# ---------------------------------------------------------------------------


class DifferentiableChamferLoss(nn.Module):
    """Differentiable Chamfer loss through the physical forward model.

    Given predicted coefficients, computes expected spot positions via
    the G-matrix, then backward Chamfer distance to observed spots.
    """

    def __init__(self, lenslet: LensletArray, max_order: int, device: torch.device):
        super().__init__()
        self.device = device
        self.focal_um = float(lenslet.focal_um)
        self.pitch_um = float(lenslet.pitch_um)
        self.sensor_w = float(lenslet.sensor_width_um)
        self.sensor_h = float(lenslet.sensor_height_um)

        # G matrix: (2*N_sub, N_terms)
        G_np = build_zernike_slope_matrix(lenslet, max_order=max_order, grid_size=128)
        self.register_buffer("G", torch.tensor(G_np, dtype=torch.float32))

        # Reference positions: (N_sub, 2)
        ref_np = lenslet.reference_positions()
        self.n_sub = len(ref_np)
        self.register_buffer("ref", torch.tensor(ref_np, dtype=torch.float32))

    def forward(
        self,
        pred_coeffs: torch.Tensor,
        observed_list: list,
        obs_subsample: int = 256,
    ) -> torch.Tensor:
        """Compute mean backward Chamfer loss for a batch.

        Args:
            pred_coeffs: (B, 9) predicted coefficients (skip piston).
            observed_list: list of B numpy arrays, each (M_i, 2) observed spots.
            obs_subsample: max observations per sample.

        Returns:
            scalar loss (mean over batch).
        """
        B = pred_coeffs.shape[0]
        D = pred_coeffs.shape[1]

        # Build full coefficient vectors (with piston=0)
        full_coeffs = torch.zeros(B, D + 1, dtype=torch.float32, device=self.device)
        full_coeffs[:, 1:] = pred_coeffs

        # Compute expected positions: slopes = G @ coeffs
        # G: (2*N_sub, N_terms), full_coeffs: (B, N_terms)
        slopes = full_coeffs @ self.G.T  # (B, 2*N_sub)
        sx = slopes[:, : self.n_sub]  # (B, N_sub)
        sy = slopes[:, self.n_sub :]  # (B, N_sub)

        # Expected positions: ref + focal * slopes
        E_x = self.ref[:, 0].unsqueeze(0) + self.focal_um * sx  # (B, N_sub)
        E_y = self.ref[:, 1].unsqueeze(0) + self.focal_um * sy  # (B, N_sub)
        E = torch.stack([E_x, E_y], dim=-1)  # (B, N_sub, 2)

        # In-bounds mask (soft: use sigmoid for differentiability)
        margin = self.pitch_um * 0.5
        # Hard mask for selection (detach)
        in_bounds = (
            (E[:, :, 0] >= -margin)
            & (E[:, :, 0] <= self.sensor_w + margin)
            & (E[:, :, 1] >= -margin)
            & (E[:, :, 1] <= self.sensor_h + margin)
        )  # (B, N_sub)

        losses = []
        for b in range(B):
            obs_np = observed_list[b]
            M = len(obs_np)
            if M < 5:
                continue

            # Subsample observations
            if M > obs_subsample:
                idx = np.random.choice(M, obs_subsample, replace=False)
                obs_np = obs_np[idx]
                M = obs_subsample

            obs_t = torch.tensor(
                obs_np, dtype=torch.float32, device=self.device
            )  # (M, 2)

            # Get in-bounds predictions for this sample
            ib_mask = in_bounds[b]  # (N_sub,)
            n_ib = ib_mask.sum().item()

            if n_ib < 5:
                # Penalize: distance of predictions from sensor center
                center = torch.tensor(
                    [self.sensor_w / 2, self.sensor_h / 2],
                    dtype=torch.float32,
                    device=self.device,
                )
                dist = ((E[b] - center) ** 2).sum(dim=-1).sqrt().mean() / self.pitch_um
                losses.append(dist + 10.0)
                continue

            # Select in-bounds predictions
            E_ib = E[b][ib_mask]  # (n_ib, 2)

            # Backward Chamfer: for each observation, nearest in-bounds prediction
            # Use chunked computation for memory
            # cdist: (n_ib, M) — distances normalized by pitch
            if n_ib <= PRED_CHUNK:
                dists = torch.cdist(E_ib.unsqueeze(0), obs_t.unsqueeze(0)).squeeze(
                    0
                )  # (n_ib, M)
                dists = dists / self.pitch_um
                # For each observation (col), find min over predictions (rows)
                # Use straight-through for differentiability
                min_dists_detached, min_idx = dists.detach().min(dim=0)  # (M,)
                # Gather differentiable distances
                min_dists = dists[min_idx, torch.arange(M, device=self.device)]  # (M,)
            else:
                # Chunk over predictions
                min_dists = torch.full(
                    (M,), 1e6, dtype=torch.float32, device=self.device
                )
                min_idx = torch.zeros(M, dtype=torch.long, device=self.device)
                for p_start in range(0, n_ib, PRED_CHUNK):
                    p_end = min(p_start + PRED_CHUNK, n_ib)
                    E_chunk = E_ib[p_start:p_end]  # (chunk, 2)
                    d_chunk = (
                        torch.cdist(E_chunk.unsqueeze(0), obs_t.unsqueeze(0)).squeeze(0)
                        / self.pitch_um
                    )  # (chunk, M)
                    chunk_min_d, chunk_min_i = d_chunk.detach().min(dim=0)  # (M,)
                    better = chunk_min_d < min_dists.detach()
                    # Update with differentiable values
                    new_dists = d_chunk[
                        chunk_min_i, torch.arange(M, device=self.device)
                    ]
                    min_dists = torch.where(better, new_dists, min_dists)
                    min_idx = torch.where(better, chunk_min_i + p_start, min_idx)

            # Cap distances to avoid outlier domination
            min_dists_capped = min_dists.clamp(max=5.0)
            loss_b = min_dists_capped.mean()
            losses.append(loss_b)

        if not losses:
            return torch.tensor(
                0.0, dtype=torch.float32, device=self.device, requires_grad=True
            )

        return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
#  Data generation (online, per-batch)
# ---------------------------------------------------------------------------


def build_generator(cfg: dict):
    """Build reusable components for fast data generation."""
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
    return la, ref, G, n_sub, focal_um, sensor_w, sensor_h, centroid_noise_um, zer


def generate_batch(
    ref,
    G,
    n_sub,
    focal_um,
    sensor_w,
    sensor_h,
    centroid_noise_um,
    zer_cfg,
    batch_size,
    rng,
):
    """Generate a batch of (images, observed_positions, true_coefficients)."""
    images = []
    observed_list = []
    true_coeffs_list = []
    max_order = zer_cfg["order"]
    coeff_bound = zer_cfg["coeff_bound"]

    attempts = 0
    while len(images) < batch_size and attempts < batch_size * 5:
        attempts += 1
        pv = np.exp(rng.uniform(np.log(PV_MIN), np.log(PV_MAX)))

        coeffs = random_zernike_coeffs(
            max_order=max_order,
            coeff_bound=coeff_bound,
            seed=int(rng.randint(0, 2**31)),
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

        in_bounds = (
            (observed[:, 0] >= 0)
            & (observed[:, 0] <= sensor_w)
            & (observed[:, 1] >= 0)
            & (observed[:, 1] <= sensor_h)
        )
        observed = observed[in_bounds].astype(np.float32)

        if len(observed) < 20:
            continue

        img = spots_to_image(observed, sensor_w, sensor_h)
        images.append(img)
        observed_list.append(observed)
        true_coeffs_list.append(coeffs[1:].astype(np.float32))

    if not images:
        return None

    images_t = torch.from_numpy(np.stack(images))
    true_coeffs_t = torch.from_numpy(np.stack(true_coeffs_list))
    return images_t, observed_list, true_coeffs_t


# ---------------------------------------------------------------------------
#  Validation (MSE-based, for tracking coefficient accuracy)
# ---------------------------------------------------------------------------


def validate_mse(model, val_images, val_targets, device):
    """Quick MSE validation on pre-generated data."""
    model.eval()
    with torch.no_grad():
        inp = val_images.to(device)
        tgt = val_targets.to(device)
        pred = model(inp)
        mse = ((pred - tgt) ** 2).mean().item()
        rmse_per = ((pred - tgt) ** 2).mean(dim=1).sqrt()
        mean_rmse = rmse_per.mean().item()
        frac_015 = (rmse_per < 0.15).float().mean().item()
        frac_020 = (rmse_per < 0.20).float().mean().item()
    return mse, mean_rmse, frac_015, frac_020


def generate_val_data(
    ref, G, n_sub, focal_um, sensor_w, sensor_h, centroid_noise_um, zer_cfg, n_val, seed
):
    """Pre-generate validation data."""
    rng = np.random.RandomState(seed)
    max_order = zer_cfg["order"]
    coeff_bound = zer_cfg["coeff_bound"]

    images = []
    targets = []
    observed_list = []

    attempts = 0
    while len(images) < n_val and attempts < n_val * 5:
        attempts += 1
        pv = np.exp(rng.uniform(np.log(PV_MIN), np.log(PV_MAX)))
        coeffs = random_zernike_coeffs(
            max_order=max_order,
            coeff_bound=coeff_bound,
            seed=int(rng.randint(0, 2**31)),
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
        in_bounds = (
            (observed[:, 0] >= 0)
            & (observed[:, 0] <= sensor_w)
            & (observed[:, 1] >= 0)
            & (observed[:, 1] <= sensor_h)
        )
        observed = observed[in_bounds].astype(np.float32)
        if len(observed) < 20:
            continue
        img = spots_to_image(observed, sensor_w, sensor_h)
        images.append(img)
        targets.append(coeffs[1:].astype(np.float32))
        observed_list.append(observed)

    return (
        torch.from_numpy(np.stack(images)),
        torch.from_numpy(np.stack(targets)),
        observed_list,
    )


# ---------------------------------------------------------------------------
#  Per-PV evaluation
# ---------------------------------------------------------------------------


def evaluate_per_pv(model, cfg, device, n_per_pv=25):
    """Evaluate model at specific PV levels using forward_pipeline."""
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
            seed = 950000 + int(pv * 1000) + i
            try:
                result = forward_pipeline(cfg, pv=pv, seed=seed)
            except Exception:
                continue
            observed = result["observed_positions"]
            true_coeffs = result["coeffs"]
            if len(observed) < 10:
                continue
            img = spots_to_image(observed, sensor_w, sensor_h)
            inp = torch.from_numpy(img).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(inp).squeeze(0).cpu().numpy()
            pred_full = np.zeros_like(true_coeffs)
            pred_full[1:] = pred
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
#  Main training loop
# ---------------------------------------------------------------------------


def main():
    import yaml

    print("=" * 70)
    print("Chamfer-Loss Fine-Tuning of NN Warm-Start Model")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg_path = "configs/base_no_oracle.yaml"
    cfg = yaml.safe_load(open(cfg_path))

    # Build physics components
    gen = build_generator(cfg)
    la, ref, G, n_sub, focal_um, sensor_w, sensor_h, centroid_noise_um, zer_cfg = gen

    # Load pretrained model
    print(f"\nLoading pretrained model from {PRETRAINED_PATH}...")
    checkpoint = torch.load(PRETRAINED_PATH, map_location=device, weights_only=True)
    model = ZernikeCNN(
        output_dim=checkpoint.get("output_dim", 9),
        dropout=checkpoint.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} parameters")

    # Build differentiable Chamfer loss
    chamfer_loss_fn = DifferentiableChamferLoss(
        la,
        max_order=zer_cfg["order"],
        device=device,
    ).to(device)

    # MSE loss for regularization
    mse_loss_fn = nn.MSELoss()

    # Generate validation data
    print(f"\nGenerating {N_VAL} validation samples...")
    val_images, val_targets, val_observed = generate_val_data(
        ref,
        G,
        n_sub,
        focal_um,
        sensor_w,
        sensor_h,
        centroid_noise_um,
        zer_cfg,
        N_VAL,
        seed=777777,
    )
    print(f"Val data: {val_images.shape}")

    # Pre-training evaluation
    print("\n--- Pre-fine-tuning evaluation ---")
    mse, mean_rmse, f015, f020 = validate_mse(model, val_images, val_targets, device)
    print(
        f"Val MSE={mse:.6f} RMSE={mean_rmse:.4f} <0.15={f015 * 100:.1f}% <0.20={f020 * 100:.1f}%"
    )
    evaluate_per_pv(model, cfg, device, n_per_pv=25)

    # Optimizer: small LR, only fine-tune
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=N_TRAIN_BATCHES,
        eta_min=LR * 0.01,
    )

    # Training loop
    print(f"\n--- Fine-tuning with Chamfer loss ({N_TRAIN_BATCHES} iterations) ---")
    rng = np.random.RandomState(42)
    best_val_rmse = mean_rmse
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)

    for step in range(1, N_TRAIN_BATCHES + 1):
        model.train()

        # Generate batch online
        batch_data = generate_batch(
            ref,
            G,
            n_sub,
            focal_um,
            sensor_w,
            sensor_h,
            centroid_noise_um,
            zer_cfg,
            BATCH_SIZE,
            rng,
        )
        if batch_data is None:
            continue

        images_t, observed_list, true_coeffs_t = batch_data
        images_t = images_t.to(device)
        true_coeffs_t = true_coeffs_t.to(device)

        # Forward
        pred_coeffs = model(images_t)  # (B, 9)

        # Chamfer loss (through physics)
        loss_chamfer = chamfer_loss_fn(pred_coeffs, observed_list, OBS_SUBSAMPLE)

        # MSE regularizer
        loss_mse = mse_loss_fn(pred_coeffs, true_coeffs_t)

        # Total loss
        loss = CHAMFER_WEIGHT * loss_chamfer + MSE_WEIGHT * loss_mse

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Step {step:5d}/{N_TRAIN_BATCHES} | "
                f"chamfer={loss_chamfer.item():.4f} | "
                f"mse={loss_mse.item():.6f} | "
                f"total={loss.item():.4f} | lr={lr:.2e}",
                flush=True,
            )

        if step % EVAL_EVERY == 0:
            mse, mean_rmse, f015, f020 = validate_mse(
                model, val_images, val_targets, device
            )
            print(
                f"  [EVAL] Val RMSE={mean_rmse:.4f} <0.15={f015 * 100:.1f}% <0.20={f020 * 100:.1f}%",
                flush=True,
            )
            if mean_rmse < best_val_rmse:
                best_val_rmse = mean_rmse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  *NEW BEST* RMSE={best_val_rmse:.4f}", flush=True)

        if step % SAVE_EVERY == 0:
            # Save current best
            model.load_state_dict(best_state)
            save_checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_type": "cnn",
                "output_dim": model.output_dim,
                "dropout": 0.1,
                "img_size": IMG_SIZE,
                "fine_tuned": True,
                "chamfer_steps": step,
            }
            torch.save(save_checkpoint, OUTPUT_PATH)
            print(f"  Checkpoint saved to {OUTPUT_PATH}", flush=True)

    # Restore best
    model.load_state_dict(best_state)

    # Save final
    save_checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_type": "cnn",
        "output_dim": model.output_dim,
        "dropout": 0.1,
        "img_size": IMG_SIZE,
        "fine_tuned": True,
        "chamfer_steps": N_TRAIN_BATCHES,
    }
    torch.save(save_checkpoint, OUTPUT_PATH)
    print(f"\nFinal model saved to {OUTPUT_PATH}")

    # Final evaluation
    print("\n--- Post-fine-tuning evaluation ---")
    mse, mean_rmse, f015, f020 = validate_mse(model, val_images, val_targets, device)
    print(
        f"Val MSE={mse:.6f} RMSE={mean_rmse:.4f} <0.15={f015 * 100:.1f}% <0.20={f020 * 100:.1f}%"
    )
    evaluate_per_pv(model, cfg, device, n_per_pv=25)

    print("\nDone!")


if __name__ == "__main__":
    main()
