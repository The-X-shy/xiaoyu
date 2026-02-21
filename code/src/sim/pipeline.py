"""Complete forward simulation pipeline for SHWS."""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional

from src.sim.wavefront import (
    random_zernike_coeffs,
    scale_coeffs_to_pv,
    generate_wavefront,
)
from src.sim.lenslet import LensletArray
from src.sim.imaging import simulate_spots, extract_centroids
from src.sim.noise import apply_noise, apply_missing_spots


def forward_pipeline(
    cfg: Dict[str, Any],
    pv: float,
    seed: int,
    missing_ratio: float = 0.0,
    coeffs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run complete forward simulation pipeline.

    Args:
        cfg: Configuration dict with optics/sensor/noise/zernike sections.
        pv: Target peak-to-valley wavefront amplitude (waves).
        seed: Random seed.
        missing_ratio: Fraction of spots to randomly remove.
        coeffs: Optional pre-specified Zernike coefficients.

    Returns:
        Dict with keys: coeffs, wavefront, slopes, ref_positions,
        displaced_positions, observed_positions, lenslet, keep_mask.
    """
    opt = cfg["optics"]
    sen = cfg["sensor"]
    noi = cfg["noise"]
    zer = cfg["zernike"]

    grid_size = 128  # Development grid size

    # 1. Generate or use provided coefficients
    if coeffs is None:
        coeffs = random_zernike_coeffs(
            max_order=zer["order"],
            coeff_bound=zer["coeff_bound"],
            seed=seed,
        )
        if pv > 0:
            coeffs = scale_coeffs_to_pv(coeffs, target_pv=pv, grid_size=grid_size)
        else:
            coeffs[:] = 0.0

    # 2. Generate wavefront
    W = generate_wavefront(coeffs, grid_size)

    # 3. Create lenslet array and compute slopes
    mla_grid_size = opt.get("grid_size", 0)  # 0 = auto (circular aperture)
    wavelength_um = opt.get("wavelength_nm", 0.0) / 1000.0  # nm -> um
    la = LensletArray(
        pitch_um=opt["pitch_um"],
        focal_mm=opt["focal_mm"],
        fill_factor=opt["fill_factor"],
        sensor_width_px=sen["width_px"],
        sensor_height_px=sen["height_px"],
        pixel_um=sen["pixel_um"],
        mla_grid_size=mla_grid_size,
        wavelength_um=wavelength_um,
    )
    slopes = la.compute_slopes(W, grid_size)

    # 4. Compute displaced positions
    ref_positions = la.reference_positions()
    disp = la.slopes_to_displacements(slopes)
    displaced_positions = ref_positions + disp
    sub_idx_all = np.arange(len(displaced_positions), dtype=int)

    # 5. Add centroid noise
    observed = extract_centroids(
        displaced_positions,
        noise_px=noi["centroid_noise_px"],
        pixel_um=sen["pixel_um"],
        seed=seed + 1,
    )

    # 6. Add read noise
    observed = apply_noise(
        observed,
        read_sigma=noi["read_sigma"],
        background=noi["background"],
        seed=seed + 2,
    )

    # 7. Clip spots to sensor bounds (spots outside sensor are physically invisible)
    sensor_w = sen["width_px"] * sen["pixel_um"]
    sensor_h = sen["height_px"] * sen["pixel_um"]
    in_sensor = (
        (observed[:, 0] >= 0)
        & (observed[:, 0] <= sensor_w)
        & (observed[:, 1] >= 0)
        & (observed[:, 1] <= sensor_h)
    )
    observed = observed[in_sensor]
    displaced_positions = displaced_positions[in_sensor]
    observed_sub_idx = sub_idx_all[in_sensor]

    # 8. Apply missing spots
    keep_mask = np.ones(len(observed), dtype=bool)
    if missing_ratio > 0:
        observed, keep_mask = apply_missing_spots(
            observed,
            ratio=missing_ratio,
            seed=seed + 3,
        )
        observed_sub_idx = observed_sub_idx[keep_mask]

    return {
        "coeffs": coeffs,
        "wavefront": W,
        "slopes": slopes,
        "ref_positions": ref_positions,
        "displaced_positions": displaced_positions,
        "observed_positions": observed,
        "observed_sub_idx": observed_sub_idx,
        "lenslet": la,
        "keep_mask": keep_mask,
    }
