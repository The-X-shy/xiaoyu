#!/usr/bin/env python3
"""Train a larger CNN (ResNet-style) for direct Zernike prediction.

Key improvements over v2:
  - 128x128 input resolution (vs 64x64) — sub-pitch spatial info
  - ResNet-18-inspired architecture with ~1.8M params (vs 251k)
  - 200k training samples (vs 50k)
  - PV-weighted sampling: 60% from PV=[3,20], 40% from PV=[0.5,3]
  - 3-channel image: density, binary presence, local density variance

Usage (on GPU server):
    PYTHONPATH=. python3 scripts/train_nn_v3.py
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim.lenslet import LensletArray
from src.sim.wavefront import random_zernike_coeffs, scale_coeffs_to_pv
from src.recon.least_squares import build_zernike_slope_matrix

# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------

RES = 128
N_CHANNELS = 3  # density, binary, local_variance
N_COEFFS = 9  # output (skip piston)

N_TRAIN = 200000  # virtual epoch size (lazy generation)
N_VAL = 3000  # pre-generated for consistent eval
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
N_EPOCHS = 30  # each epoch = N_TRAIN fresh samples
PV_MIN = 0.5
PV_MAX = 20.0

MODEL_PATH = "models/nn_v3_resnet.pt"


# ---------------------------------------------------------------------------
#  3-Channel 128x128 Image Rendering
# ---------------------------------------------------------------------------


def spots_to_image_128(
    observed: np.ndarray,
    sensor_w: float,
    sensor_h: float,
) -> np.ndarray:
    """Render spots to 3-channel 128x128 image.

    Channel 0: spot count density (normalized)
    Channel 1: binary presence
    Channel 2: local density variance (3x3 neighborhood)
    """
    res = RES
    img = np.zeros((N_CHANNELS, res, res), dtype=np.float32)
    M = len(observed)
    if M == 0:
        return img

    xn = np.clip(observed[:, 0] / sensor_w, 0, 1 - 1e-9)
    yn = np.clip(observed[:, 1] / sensor_h, 0, 1 - 1e-9)
    ix = np.clip((xn * res).astype(int), 0, res - 1)
    iy = np.clip((yn * res).astype(int), 0, res - 1)

    # Channel 0: density
    for i in range(M):
        img[0, iy[i], ix[i]] += 1.0
    expected = 13224.0 / (res * res)
    img[0] /= max(expected, 1.0)

    # Channel 1: binary
    img[1] = (img[0] > 0).astype(np.float32)

    # Channel 2: local variance (3x3 neighborhood of density)
    from scipy.ndimage import uniform_filter

    mean_local = uniform_filter(img[0], size=3, mode="constant")
    mean_sq_local = uniform_filter(img[0] ** 2, size=3, mode="constant")
    var_local = mean_sq_local - mean_local**2
    img[2] = np.clip(var_local, 0, None)

    return img


# ---------------------------------------------------------------------------
#  ResNet-style CNN
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    """Basic residual block."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class ZernikeResNet(nn.Module):
    """ResNet-inspired CNN: 3ch 128x128 → 9 Zernike coefficients.

    Architecture:
      Conv(3→64, 7, stride=2) + BN + ReLU + MaxPool → 32x32
      ResBlock(64) × 2
      Conv(64→128, 3, stride=2) + BN + ReLU → 16x16
      ResBlock(128) × 2
      Conv(128→256, 3, stride=2) + BN + ReLU → 8x8
      ResBlock(256) × 2
      Conv(256→256, 3, stride=2) + BN + ReLU → 4x4
      AdaptiveAvgPool(1) → 256
      FC(256→128) + ReLU + Dropout
      FC(128→9)

    ~1.8M parameters
    """

    def __init__(self, n_coeffs: int = N_COEFFS, dropout: float = 0.2):
        super().__init__()
        self.n_coeffs = n_coeffs
        self.output_dim = n_coeffs

        self.stem = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 64, 7, stride=2, padding=3),  # 128→64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64→32
        )

        self.layer1 = nn.Sequential(ResBlock(64), ResBlock(64))

        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 32→16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(ResBlock(128), ResBlock(128))

        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 16→8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(ResBlock(256), ResBlock(256))

        self.down4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1),  # 8→4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_coeffs),
        )

        # Small init for output layer
        with torch.no_grad():
            self.head[-1].weight.mul_(0.1)
            self.head[-1].bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.down2(x)
        x = self.layer2(x)
        x = self.down3(x)
        x = self.layer3(x)
        x = self.down4(x)
        x = self.pool(x)
        return self.head(x)


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
    return ref, G, n_sub, focal_um, sensor_w, sensor_h, centroid_noise_um, zer


def generate_sample(
    ref, G, n_sub, focal_um, sensor_w, sensor_h, noise_um, zer_cfg, rng
):
    """Generate one sample with PV-weighted distribution."""
    max_order = zer_cfg["order"]
    coeff_bound = zer_cfg["coeff_bound"]

    # PV-weighted: 60% high-PV [3, 20], 40% low-PV [0.5, 3]
    if rng.random() < 0.6:
        pv = np.exp(rng.uniform(np.log(3.0), np.log(PV_MAX)))
    else:
        pv = np.exp(rng.uniform(np.log(PV_MIN), np.log(3.0)))

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

    noise = rng.randn(n_sub, 2).astype(np.float64) * noise_um
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

    img = spots_to_image_128(observed, sensor_w, sensor_h)
    target = coeffs[1:].astype(np.float32)
    return img, target


def generate_dataset(cfg, n_samples, seed, desc="Gen"):
    """Pre-generate a small dataset (for validation)."""
    gen = build_generator(cfg)
    ref, G, n_sub, focal_um, sensor_w, sensor_h, noise_um, zer_cfg = gen
    rng = np.random.RandomState(seed)

    imgs = []
    targets = []
    t0 = time.time()
    attempts = 0
    while len(imgs) < n_samples:
        attempts += 1
        r = generate_sample(
            ref, G, n_sub, focal_um, sensor_w, sensor_h, noise_um, zer_cfg, rng
        )
        if r is not None:
            imgs.append(r[0])
            targets.append(r[1])
            if len(imgs) % 5000 == 0:
                el = time.time() - t0
                print(
                    f"  {desc}: {len(imgs)}/{n_samples} ({len(imgs) / el:.0f}/s)",
                    flush=True,
                )

    el = time.time() - t0
    print(f"  {desc}: {n_samples} in {el:.0f}s ({n_samples / el:.0f}/s)", flush=True)
    return np.stack(imgs), np.stack(targets)


class SpotDataset(Dataset):
    def __init__(self, imgs, targets):
        self.imgs = torch.from_numpy(imgs)
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.targets[idx]


class LazySpotDataset(Dataset):
    """On-the-fly data generation to avoid OOM.

    Each epoch generates fresh random samples.
    """

    def __init__(self, cfg, epoch_size, base_seed=0):
        self.epoch_size = epoch_size
        self.base_seed = base_seed
        self.epoch = 0  # incremented externally for fresh data each epoch

        gen = build_generator(cfg)
        self.ref, self.G, self.n_sub, self.focal_um = gen[0], gen[1], gen[2], gen[3]
        self.sensor_w, self.sensor_h, self.noise_um, self.zer_cfg = (
            gen[4],
            gen[5],
            gen[6],
            gen[7],
        )

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        # Deterministic seed per (epoch, idx) for reproducibility
        seed = self.base_seed + self.epoch * self.epoch_size + idx
        rng = np.random.RandomState(seed)

        # Try up to 10 times to get a valid sample
        for attempt in range(10):
            r = generate_sample(
                self.ref,
                self.G,
                self.n_sub,
                self.focal_um,
                self.sensor_w,
                self.sensor_h,
                self.noise_um,
                self.zer_cfg,
                rng,
            )
            if r is not None:
                img, target = r
                return torch.from_numpy(img), torch.from_numpy(target)

        # Fallback: return zero sample (very rare)
        return (
            torch.zeros(N_CHANNELS, RES, RES, dtype=torch.float32),
            torch.zeros(N_COEFFS, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------


def train_model(cfg, val_imgs, val_tgt, device):
    # Lazy training dataset (generates fresh data each epoch)
    train_ds = LazySpotDataset(cfg, epoch_size=N_TRAIN, base_seed=0)
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    val_ds = SpotDataset(val_imgs, val_tgt)
    val_dl = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    model = ZernikeResNet(n_coeffs=N_COEFFS, dropout=0.2).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ZernikeResNet, {total_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=LR * 0.01
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience = 0
    max_patience = 20

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        train_ds.set_epoch(epoch)

        # Train
        model.train()
        train_loss = 0.0
        n_b = 0
        for inp, tgt in train_dl:
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            pred = model(inp)
            loss = criterion(pred, tgt)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()
            n_b += 1
        train_loss /= max(n_b, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        val_rmse_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for inp, tgt in val_dl:
                inp = inp.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)
                pred = model(inp)
                loss = criterion(pred, tgt)
                val_loss += loss.item() * inp.size(0)
                rmse_per = ((pred - tgt) ** 2).mean(dim=1).sqrt()
                val_rmse_sum += rmse_per.sum().item()
                n_val += inp.size(0)
        val_loss /= max(n_val, 1)
        val_rmse = val_rmse_sum / max(n_val, 1)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        el = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{N_EPOCHS} | train={train_loss:.6f} | "
            f"val_mse={val_loss:.6f} val_rmse={val_rmse:.4f} | "
            f"lr={lr:.2e} | {el:.0f}s",
            end="",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
            print("  *BEST*", flush=True)
        else:
            patience += 1
            print("", flush=True)
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------


def evaluate_per_pv(model, cfg, device, n_per_pv=50):
    from src.sim.pipeline import forward_pipeline

    sensor_w = cfg["sensor"]["width_px"] * cfg["sensor"]["pixel_um"]
    sensor_h = cfg["sensor"]["height_px"] * cfg["sensor"]["pixel_um"]
    pv_levels = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0]

    model.eval()
    print(
        f"\n{'PV':>5s} | {'RMSE':>8s} | {'Max':>8s} | {'<0.10':>6s} | {'<0.15':>6s} | {'<0.20':>6s} | {'nSpots':>6s}"
    )
    print("-" * 70)

    for pv in pv_levels:
        rmses = []
        max_errs = []
        ns = []
        for i in range(n_per_pv):
            seed = 970000 + int(pv * 1000) + i
            try:
                result = forward_pipeline(cfg, pv=pv, seed=seed)
            except Exception:
                continue
            observed = result["observed_positions"]
            true_c = result["coeffs"]
            if len(observed) < 10:
                continue

            img = spots_to_image_128(observed, sensor_w, sensor_h)
            inp = torch.from_numpy(img).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(inp).squeeze(0).cpu().numpy()

            full = np.zeros_like(true_c)
            full[1:] = pred
            diff = full - true_c
            rmse = np.sqrt(np.mean(diff**2))
            rmses.append(rmse)
            max_errs.append(np.max(np.abs(diff)))
            ns.append(len(observed))

        if rmses:
            r = np.array(rmses)
            m = np.array(max_errs)
            print(
                f"{pv:5.1f} | {r.mean():8.4f} | {m.mean():8.4f} | "
                f"{100 * (r < 0.10).mean():5.1f}% | {100 * (r < 0.15).mean():5.1f}% | "
                f"{100 * (r < 0.20).mean():5.1f}% | {np.mean(ns):6.0f}"
            )


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main():
    import yaml

    print("=" * 70)
    print("Neural Network v3 — ResNet Direct Prediction (128x128, 3ch)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = yaml.safe_load(open("configs/base_no_oracle.yaml"))

    # Check if scipy is available (needed for local variance)
    try:
        from scipy.ndimage import uniform_filter

        print("scipy available for local variance channel")
    except ImportError:
        print("WARNING: scipy not available, channel 2 will be zeros")

    # Generate validation data only (training uses lazy on-the-fly generation)
    print(f"\n--- Generating {N_VAL} validation samples ---")
    val_imgs, val_tgt = generate_dataset(cfg, N_VAL, seed=500000, desc="Val")
    print(f"Val: {val_imgs.shape}")

    print(
        f"\n--- Training (lazy dataset: {N_TRAIN} samples/epoch × {N_EPOCHS} epochs) ---"
    )
    model = train_model(cfg, val_imgs, val_tgt, device)

    # Save
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "resnet",
            "output_dim": N_COEFFS,
            "res": RES,
            "n_channels": N_CHANNELS,
            "n_train": N_TRAIN,
            "dropout": 0.2,
        },
        MODEL_PATH,
    )
    print(f"\nModel saved to {MODEL_PATH}")

    # Evaluate
    print("\n--- Per-PV Evaluation ---")
    evaluate_per_pv(model, cfg, device, n_per_pv=50)

    print("\nDone!")


if __name__ == "__main__":
    main()
