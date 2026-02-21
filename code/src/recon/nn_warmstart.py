"""Neural network warm-start for wavefront reconstruction.

Supports three model architectures:
  - v1 MLP: 86 hand-crafted features → 9 coefficients (legacy)
  - v2 CNN: 2-channel 64×64 spot image → 9 coefficients
  - v3 ResNet: 3-channel 128×128 spot image → 9 coefficients

The prediction just needs to land within the Adam/ICP convergence basin
(~0.2 RMSE per coefficient).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

# ---------------------------------------------------------------------------
#  Spot image rasterisation
# ---------------------------------------------------------------------------

IMG_SIZE = 64  # 64x64 density image


def spots_to_image(
    observed: np.ndarray,
    sensor_w_um: float,
    sensor_h_um: float,
    img_size: int = IMG_SIZE,
) -> np.ndarray:
    """Rasterise spot positions into a 2-channel density image.

    Channel 0: smoothed density (count per cell, Gaussian-blurred)
    Channel 1: binary presence indicator

    Args:
        observed: (M, 2) spot positions in micrometers.
        sensor_w_um: sensor width in um.
        sensor_h_um: sensor height in um.
        img_size: output image size (default 64).

    Returns:
        (2, img_size, img_size) float32 array.
    """
    img = np.zeros((2, img_size, img_size), dtype=np.float32)
    M = len(observed)
    if M == 0:
        return img

    # Normalise positions to [0, 1)
    xn = np.clip(observed[:, 0] / sensor_w_um, 0, 1 - 1e-9)
    yn = np.clip(observed[:, 1] / sensor_h_um, 0, 1 - 1e-9)

    # Bin into image pixels
    ix = np.clip((xn * img_size).astype(int), 0, img_size - 1)
    iy = np.clip((yn * img_size).astype(int), 0, img_size - 1)

    # Channel 0: count density
    for i in range(M):
        img[0, iy[i], ix[i]] += 1.0

    # Normalise density by expected count per cell at PV=0
    # (~13224 spots / 64^2 = ~3.2 spots/cell)
    expected_per_cell = 13224.0 / (img_size * img_size)
    img[0] /= max(expected_per_cell, 1.0)

    # Channel 1: binary presence
    img[1] = (img[0] > 0).astype(np.float32)

    return img


def spots_to_image_torch(
    observed: np.ndarray,
    sensor_w_um: float,
    sensor_h_um: float,
    img_size: int = IMG_SIZE,
) -> torch.Tensor:
    """Same as spots_to_image but returns torch Tensor."""
    img = spots_to_image(observed, sensor_w_um, sensor_h_um, img_size)
    return torch.from_numpy(img)


# ---------------------------------------------------------------------------
#  v1 features (kept for backward compatibility)
# ---------------------------------------------------------------------------

N_HIST_BINS = 8
N_RADIAL_BINS = 8
FEATURE_DIM = N_HIST_BINS * N_HIST_BINS + 10 + N_RADIAL_BINS + 4  # = 86


def extract_features(
    observed: np.ndarray,
    sensor_w_um: float,
    sensor_h_um: float,
) -> np.ndarray:
    """Extract fixed-size feature vector (v1, kept for backward compatibility)."""
    M = len(observed)
    if M == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    x = observed[:, 0]
    y = observed[:, 1]
    cx = sensor_w_um / 2.0
    cy = sensor_h_um / 2.0

    xn = np.clip(x / sensor_w_um, 0, 1 - 1e-9)
    yn = np.clip(y / sensor_h_um, 0, 1 - 1e-9)
    ix = np.clip((xn * N_HIST_BINS).astype(int), 0, N_HIST_BINS - 1)
    iy = np.clip((yn * N_HIST_BINS).astype(int), 0, N_HIST_BINS - 1)
    hist2d = np.zeros((N_HIST_BINS, N_HIST_BINS), dtype=np.float32)
    for i in range(M):
        hist2d[iy[i], ix[i]] += 1.0
    total = hist2d.sum()
    if total > 0:
        hist2d /= total
    hist_flat = hist2d.ravel()

    mean_x = np.mean(x) / sensor_w_um
    mean_y = np.mean(y) / sensor_h_um
    std_x = np.std(x) / sensor_w_um
    std_y = np.std(y) / sensor_h_um
    cov_xy = np.cov(x, y)[0, 1] / (sensor_w_um * sensor_h_um) if M > 1 else 0.0
    n_spots_norm = M / 13000.0
    if M > 2 and np.std(x) > 1e-8:
        skew_x = float(np.mean(((x - np.mean(x)) / np.std(x)) ** 3))
        kurt_x = float(np.mean(((x - np.mean(x)) / np.std(x)) ** 4)) - 3.0
    else:
        skew_x, kurt_x = 0.0, 0.0
    if M > 2 and np.std(y) > 1e-8:
        skew_y = float(np.mean(((y - np.mean(y)) / np.std(y)) ** 3))
        kurt_y = float(np.mean(((y - np.mean(y)) / np.std(y)) ** 4)) - 3.0
    else:
        skew_y, kurt_y = 0.0, 0.0
    global_stats = np.array(
        [
            mean_x,
            mean_y,
            std_x,
            std_y,
            cov_xy,
            n_spots_norm,
            skew_x,
            skew_y,
            kurt_x,
            kurt_y,
        ],
        dtype=np.float32,
    )

    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_max = np.sqrt(cx**2 + cy**2) + 1e-8
    r_norm = r / r_max
    radial_hist = np.zeros(N_RADIAL_BINS, dtype=np.float32)
    bin_edges = np.linspace(0, 1.0, N_RADIAL_BINS + 1)
    for b in range(N_RADIAL_BINS):
        mask = (r_norm >= bin_edges[b]) & (r_norm < bin_edges[b + 1])
        radial_hist[b] = mask.sum()
    if radial_hist.sum() > 0:
        radial_hist /= radial_hist.sum()

    q1 = ((x >= cx) & (y >= cy)).sum()
    q2 = ((x < cx) & (y >= cy)).sum()
    q3 = ((x < cx) & (y < cy)).sum()
    q4 = ((x >= cx) & (y < cy)).sum()
    qtotal = max(q1 + q2 + q3 + q4, 1)
    quadrants = np.array(
        [q1 / qtotal, q2 / qtotal, q3 / qtotal, q4 / qtotal], dtype=np.float32
    )

    return np.concatenate([hist_flat, global_stats, radial_hist, quadrants])


# ---------------------------------------------------------------------------
#  v1 MLP Model (kept for backward compatibility)
# ---------------------------------------------------------------------------


class ZernikeMLP(nn.Module):
    """Small MLP: features -> Zernike coefficients (skip piston)."""

    def __init__(
        self,
        input_dim=FEATURE_DIM,
        output_dim=9,
        hidden_dims=(128, 256, 256, 128),
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend(
                [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True)]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
#  v2 CNN Model
# ---------------------------------------------------------------------------


class ZernikeCNN(nn.Module):
    """ConvNet: 2-channel 64x64 spot image -> 9 Zernike coefficients.

    Architecture:
      Conv2d(2, 32, 5, stride=2, pad=2) -> BN -> ReLU   # 64->32
      Conv2d(32, 64, 3, stride=2, pad=1) -> BN -> ReLU  # 32->16
      Conv2d(64, 128, 3, stride=2, pad=1) -> BN -> ReLU # 16->8
      Conv2d(128, 128, 3, stride=2, pad=1) -> BN -> ReLU # 8->4
      AdaptiveAvgPool2d(1) -> Flatten -> 128
      Linear(128, 64) -> ReLU -> Dropout
      Linear(64, 9)
    """

    def __init__(self, output_dim: int = 9, dropout: float = 0.1):
        super().__init__()
        self.output_dim = output_dim

        self.features = nn.Sequential(
            nn.Conv2d(2, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )
        # Small init for final layer
        with torch.no_grad():
            self.head[-1].weight.mul_(0.1)
            self.head[-1].bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, 64, 64) spot density images.
        Returns:
            (B, output_dim) predicted Zernike coefficients.
        """
        h = self.features(x)
        return self.head(h)


# ---------------------------------------------------------------------------
#  v3 ResNet: 3-channel 128x128 image
# ---------------------------------------------------------------------------

RES_128 = 128
N_CHANNELS_V3 = 3


def spots_to_image_128(
    observed: np.ndarray,
    sensor_w_um: float,
    sensor_h_um: float,
) -> np.ndarray:
    """Render spots to 3-channel 128x128 image.

    Channel 0: spot count density (normalized)
    Channel 1: binary presence
    Channel 2: local density variance (3x3 neighborhood)
    """
    res = RES_128
    img = np.zeros((N_CHANNELS_V3, res, res), dtype=np.float32)
    M = len(observed)
    if M == 0:
        return img

    xn = np.clip(observed[:, 0] / sensor_w_um, 0, 1 - 1e-9)
    yn = np.clip(observed[:, 1] / sensor_h_um, 0, 1 - 1e-9)
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


class _ResBlock(nn.Module):
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
    """ResNet-inspired CNN: 3ch 128x128 -> 9 Zernike coefficients.

    ~4.1M parameters.
    """

    def __init__(self, n_coeffs: int = 9, dropout: float = 0.2):
        super().__init__()
        self.n_coeffs = n_coeffs
        self.output_dim = n_coeffs

        self.stem = nn.Sequential(
            nn.Conv2d(N_CHANNELS_V3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer1 = nn.Sequential(_ResBlock(64), _ResBlock(64))
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(_ResBlock(128), _ResBlock(128))
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(_ResBlock(256), _ResBlock(256))
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
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
#  Inference wrapper (supports CNN, MLP, and ResNet)
# ---------------------------------------------------------------------------


class NNWarmStarter:
    """Load a trained model and predict Zernike coefficients from spot patterns.

    Supports model_type: "mlp", "cnn", "resnet".
    """

    def __init__(
        self,
        model_path: str,
        sensor_w_um: float,
        sensor_h_um: float,
        n_terms: int = 10,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.sensor_w_um = sensor_w_um
        self.sensor_h_um = sensor_h_um
        self.n_terms = n_terms

        checkpoint = torch.load(model_path, map_location=device, weights_only=True)

        model_type = checkpoint.get("model_type", "mlp")
        output_dim = checkpoint.get("output_dim", n_terms - 1)
        self.model_type = model_type

        if model_type == "resnet":
            self.model = ZernikeResNet(
                n_coeffs=output_dim,
                dropout=checkpoint.get("dropout", 0.2),
            )
        elif model_type == "cnn":
            self.model = ZernikeCNN(
                output_dim=output_dim, dropout=checkpoint.get("dropout", 0.1)
            )
        else:
            hidden_dims = checkpoint.get("hidden_dims", (128, 256, 256, 128))
            input_dim = checkpoint.get("input_dim", FEATURE_DIM)
            self.model = ZernikeMLP(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=tuple(hidden_dims),
            )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

    def _make_input(self, observed: np.ndarray) -> torch.Tensor:
        """Create model input tensor from observed spot positions."""
        if self.model_type == "resnet":
            img = spots_to_image_128(observed, self.sensor_w_um, self.sensor_h_um)
            return torch.from_numpy(img).unsqueeze(0).to(self.device)
        elif self.model_type == "cnn":
            img = spots_to_image(observed, self.sensor_w_um, self.sensor_h_um)
            return torch.from_numpy(img).unsqueeze(0).to(self.device)
        else:
            feat = extract_features(observed, self.sensor_w_um, self.sensor_h_um)
            return torch.tensor(
                feat, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

    @torch.no_grad()
    def predict(self, observed: np.ndarray) -> np.ndarray:
        """Predict Zernike coefficients from observed spots."""
        inp = self._make_input(observed)
        pred = self.model(inp).squeeze(0).cpu().numpy()
        coeffs = np.zeros(self.n_terms, dtype=np.float32)
        coeffs[1:] = pred
        return coeffs

    @torch.no_grad()
    def predict_ensemble(
        self,
        observed: np.ndarray,
        n_augment: int = 16,
        noise_scale: float = 0.02,
    ) -> list:
        """Predict with small input perturbations for diversity."""
        if self.model_type == "cnn":
            img = spots_to_image(observed, self.sensor_w_um, self.sensor_h_um)
            imgs = [img]
            rng = np.random.RandomState(42)
            for _ in range(n_augment):
                noise = rng.randn(*img.shape).astype(np.float32) * noise_scale
                imgs.append(img + noise)
            batch = torch.from_numpy(np.stack(imgs)).to(self.device)
        else:
            feat = extract_features(observed, self.sensor_w_um, self.sensor_h_um)
            feats = [feat]
            rng = np.random.RandomState(42)
            for _ in range(n_augment):
                noise = rng.randn(FEATURE_DIM).astype(np.float32) * noise_scale
                feats.append(feat + noise)
            batch = torch.tensor(
                np.stack(feats), dtype=torch.float32, device=self.device
            )

        preds = self.model(batch).cpu().numpy()
        results = []
        for pred in preds:
            coeffs = np.zeros(self.n_terms, dtype=np.float32)
            coeffs[1:] = pred
            results.append(coeffs)
        return results


# ---------------------------------------------------------------------------
#  Multi-model ensemble wrapper
# ---------------------------------------------------------------------------


class NNEnsembleWarmStarter:
    """Load multiple NN models and average their predictions.

    Models can be of different architectures (CNN, ResNet, MLP).
    """

    def __init__(
        self,
        model_paths: list,
        sensor_w_um: float,
        sensor_h_um: float,
        n_terms: int = 10,
        device: Optional[torch.device] = None,
    ):
        self.starters = []
        for path in model_paths:
            try:
                s = NNWarmStarter(
                    model_path=path,
                    sensor_w_um=sensor_w_um,
                    sensor_h_um=sensor_h_um,
                    n_terms=n_terms,
                    device=device,
                )
                self.starters.append(s)
            except Exception:
                pass  # skip models that fail to load
        self.n_terms = n_terms

    def predict(self, observed: np.ndarray) -> np.ndarray:
        """Average prediction from all loaded models."""
        if not self.starters:
            return np.zeros(self.n_terms, dtype=np.float32)
        preds = [s.predict(observed) for s in self.starters]
        return np.mean(preds, axis=0).astype(np.float32)

    def predict_all(self, observed: np.ndarray) -> list:
        """Return individual predictions from all models."""
        return [s.predict(observed) for s in self.starters]
