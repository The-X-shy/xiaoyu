#!/usr/bin/env python3
"""Train neural network warm-start model (v2: CNN) for Chamfer optimizer.

Uses 2-channel 64x64 spot density images as input instead of hand-crafted features.
Generates synthetic training data using G-matrix fast path.

Usage (on GPU server):
    PYTHONPATH=. python3 scripts/train_nn_warmstart.py
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim.pipeline import forward_pipeline
from src.sim.lenslet import LensletArray
from src.sim.wavefront import random_zernike_coeffs, scale_coeffs_to_pv
from src.recon.least_squares import build_zernike_slope_matrix
from src.recon.nn_warmstart import (
    spots_to_image,
    extract_features,
    ZernikeCNN,
    ZernikeMLP,
    FEATURE_DIM,
    IMG_SIZE,
)


# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------

# Model type: "cnn" (v2) or "mlp" (v1)
MODEL_TYPE = "cnn"

# Training data
N_TRAIN = 50000
N_VAL = 3000
PV_MIN = 0.5
PV_MAX = 20.0
SEED_BASE = 100000

# CNN Model
CNN_OUTPUT_DIM = 9
CNN_DROPOUT = 0.1

# MLP Model (v1 fallback)
MLP_HIDDEN_DIMS = (128, 256, 256, 128)
MLP_DROPOUT = 0.1

# Training
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
N_EPOCHS = 100
LR_SCHEDULE_PATIENCE = 5
EARLY_STOP_PATIENCE = 15

# Output
MODEL_DIR = "models"
MODEL_NAME = "nn_warmstart.pt"


# ---------------------------------------------------------------------------
#  Config loader
# ---------------------------------------------------------------------------


def load_config() -> dict:
    import yaml

    cfg_path = Path(__file__).parent.parent / "configs" / "base_no_oracle.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
#  Fast data generation (G-matrix path)
# ---------------------------------------------------------------------------


def build_fast_generator(cfg: dict):
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
    return ref, G, n_sub, focal_um, sensor_w, sensor_h, centroid_noise_um, zer


def fast_generate_sample(
    ref, G, n_sub, focal_um, sensor_w, sensor_h, centroid_noise_um, zer_cfg, pv, rng
):
    """Generate one (image_or_features, target) pair."""
    max_order = zer_cfg["order"]
    coeff_bound = zer_cfg["coeff_bound"]

    coeffs = random_zernike_coeffs(
        max_order=max_order, coeff_bound=coeff_bound, seed=int(rng.randint(0, 2**31))
    )
    if pv > 0:
        coeffs = scale_coeffs_to_pv(coeffs, target_pv=pv, grid_size=128)
    else:
        coeffs[:] = 0.0

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
    observed = observed[in_bounds]

    if len(observed) < 10:
        return None

    observed_f32 = observed.astype(np.float32)

    if MODEL_TYPE == "cnn":
        img = spots_to_image(observed_f32, sensor_w, sensor_h)
        input_data = img  # (2, 64, 64)
    else:
        input_data = extract_features(observed_f32, sensor_w, sensor_h)  # (86,)

    target = coeffs[1:].astype(np.float32)
    return input_data, target


def generate_dataset(cfg, n_samples, seed_offset=0, desc="Generating"):
    """Generate dataset."""
    gen = build_fast_generator(cfg)
    ref, G, n_sub, focal_um, sensor_w, sensor_h, centroid_noise_um, zer_cfg = gen

    inputs_list = []
    targets_list = []
    rng = np.random.RandomState(seed_offset)

    t0 = time.time()
    attempts = 0
    while len(inputs_list) < n_samples:
        pv = np.exp(rng.uniform(np.log(PV_MIN), np.log(PV_MAX)))
        result = fast_generate_sample(
            ref,
            G,
            n_sub,
            focal_um,
            sensor_w,
            sensor_h,
            centroid_noise_um,
            zer_cfg,
            pv,
            rng,
        )
        attempts += 1
        if result is not None:
            inputs_list.append(result[0])
            targets_list.append(result[1])
            if len(inputs_list) % 5000 == 0:
                elapsed = time.time() - t0
                rate = len(inputs_list) / elapsed
                eta = (n_samples - len(inputs_list)) / max(rate, 0.1)
                print(
                    f"  {desc}: {len(inputs_list)}/{n_samples} "
                    f"({rate:.1f}/s, ETA {eta:.0f}s)"
                )

    elapsed = time.time() - t0
    print(
        f"  {desc}: done {n_samples} samples in {elapsed:.0f}s "
        f"({n_samples / elapsed:.1f}/s, {attempts} attempts)"
    )

    return np.stack(inputs_list), np.stack(targets_list)


# ---------------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------------


class SpotDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------


def train_model(train_inputs, train_targets, val_inputs, val_targets, device):
    train_ds = SpotDataset(train_inputs, train_targets)
    val_ds = SpotDataset(val_inputs, val_targets)
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    if MODEL_TYPE == "cnn":
        model = ZernikeCNN(output_dim=CNN_OUTPUT_DIM, dropout=CNN_DROPOUT).to(device)
    else:
        model = ZernikeMLP(
            input_dim=FEATURE_DIM,
            output_dim=CNN_OUTPUT_DIM,
            hidden_dims=MLP_HIDDEN_DIMS,
            dropout=MLP_DROPOUT,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model type: {MODEL_TYPE}, parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_SCHEDULE_PATIENCE, factor=0.5, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0
        for inp, target in train_dl:
            inp = inp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            pred = model(inp)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        val_rmse_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for inp, target in val_dl:
                inp = inp.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                pred = model(inp)
                loss = criterion(pred, target)
                val_loss += loss.item() * inp.size(0)
                rmse_per = torch.sqrt((pred - target).pow(2).mean(dim=1))
                val_rmse_sum += rmse_per.sum().item()
                n_val += inp.size(0)
        val_loss /= max(n_val, 1)
        val_rmse = val_rmse_sum / max(n_val, 1)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{N_EPOCHS} | "
            f"train_mse={train_loss:.6f} | val_mse={val_loss:.6f} | "
            f"val_rmse={val_rmse:.4f} | lr={lr:.2e} | {elapsed:.1f}s",
            end="",
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  *BEST*")
        else:
            patience_counter += 1
            print()
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(model, cfg, device, n_per_pv=50):
    sensor_w = cfg["sensor"]["width_px"] * cfg["sensor"]["pixel_um"]
    sensor_h = cfg["sensor"]["height_px"] * cfg["sensor"]["pixel_um"]
    pv_levels = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0]

    model.eval()
    print("\n" + "=" * 80)
    print("Per-PV Evaluation")
    print("=" * 80)
    print(
        f"{'PV':>5s} | {'RMSE':>8s} | {'MaxErr':>8s} | "
        f"{'<0.10':>6s} | {'<0.15':>6s} | {'<0.20':>6s} | {'nSpots':>6s}"
    )
    print("-" * 70)

    for pv in pv_levels:
        rmses = []
        max_errs = []
        n_spots_list = []

        for i in range(n_per_pv):
            seed = 900000 + int(pv * 1000) + i
            try:
                result = forward_pipeline(cfg, pv=pv, seed=seed)
            except Exception:
                continue
            observed = result["observed_positions"]
            true_coeffs = result["coeffs"]
            if len(observed) < 10:
                continue

            if MODEL_TYPE == "cnn":
                img = spots_to_image(observed, sensor_w, sensor_h)
                inp = torch.from_numpy(img).unsqueeze(0).to(device)
            else:
                feat = extract_features(observed, sensor_w, sensor_h)
                inp = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(
                    0
                )

            with torch.no_grad():
                pred = model(inp).squeeze(0).cpu().numpy()

            pred_full = np.zeros_like(true_coeffs)
            pred_full[1:] = pred
            diff = pred_full - true_coeffs
            rmse = np.sqrt(np.mean(diff**2))
            max_err = np.max(np.abs(diff))
            rmses.append(rmse)
            max_errs.append(max_err)
            n_spots_list.append(len(observed))

        if not rmses:
            print(f"{pv:5.1f} | {'N/A':>8s}")
            continue
        rmses = np.array(rmses)
        max_errs = np.array(max_errs)
        print(
            f"{pv:5.1f} | {rmses.mean():8.4f} | {max_errs.mean():8.4f} | "
            f"{(rmses < 0.10).mean() * 100:5.1f}% | {(rmses < 0.15).mean() * 100:5.1f}% | "
            f"{(rmses < 0.20).mean() * 100:5.1f}% | {np.mean(n_spots_list):6.0f}"
        )


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 80)
    print(f"Neural Network Warm-Start Training (v2 — {MODEL_TYPE.upper()})")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = load_config()
    print(f"Config: {cfg['experiment']['name']}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Generate data
    print(f"\n--- Generating training data ({N_TRAIN} samples) ---")
    train_inp, train_tgt = generate_dataset(cfg, N_TRAIN, seed_offset=0, desc="Train")

    print(f"\n--- Generating validation data ({N_VAL} samples) ---")
    val_inp, val_tgt = generate_dataset(cfg, N_VAL, seed_offset=500000, desc="Val")

    print(f"\nTrain: {train_inp.shape}, Val: {val_inp.shape}")
    print(f"Target stats — mean: {train_tgt.mean(axis=0).round(4)}")
    print(f"Target stats — std:  {train_tgt.std(axis=0).round(4)}")

    # Train
    print(f"\n--- Training ---")
    model = train_model(train_inp, train_tgt, val_inp, val_tgt, device)

    # Save
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_type": MODEL_TYPE,
        "output_dim": CNN_OUTPUT_DIM,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "pv_range": [PV_MIN, PV_MAX],
    }
    if MODEL_TYPE == "cnn":
        checkpoint["dropout"] = CNN_DROPOUT
        checkpoint["img_size"] = IMG_SIZE
    else:
        checkpoint["input_dim"] = FEATURE_DIM
        checkpoint["hidden_dims"] = list(MLP_HIDDEN_DIMS)
    torch.save(checkpoint, model_path)
    print(f"\nModel saved to {model_path}")

    # Evaluate
    evaluate_model(model, cfg, device, n_per_pv=50)
    print("\nDone!")


if __name__ == "__main__":
    main()
