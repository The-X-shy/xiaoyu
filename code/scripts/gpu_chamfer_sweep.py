#!/usr/bin/env python3
"""Chamfer optimizer hyperparameter sweep on GPU.

Tests different configurations to find what converges at high PV.
"""

import torch
import numpy as np
import yaml
import time
from src.sim.pipeline import forward_pipeline
from src.recon.chamfer_optimizer import ChamferOptimizer
from src.recon.zernike import num_zernike_terms
from src.sim.lenslet import LensletArray


def make_la(cfg):
    opt = cfg["optics"]
    sen = cfg["sensor"]
    return LensletArray(
        pitch_um=opt["pitch_um"],
        focal_mm=opt["focal_mm"],
        fill_factor=opt["fill_factor"],
        sensor_width_px=sen["width_px"],
        sensor_height_px=sen["height_px"],
        pixel_um=sen["pixel_um"],
    )


def test_config(cfg, la, pv, description, overrides):
    """Test a specific Chamfer configuration."""
    sim = forward_pipeline(cfg, pv=pv, seed=42)
    obs = sim["observed_positions"]
    c_true = sim["coeffs"]

    # Apply overrides
    test_cfg = yaml.safe_load(yaml.dump(cfg))  # deep copy
    for k, v in overrides.items():
        test_cfg["asm"][k] = v

    t0 = time.time()
    ch = ChamferOptimizer(obs, la, test_cfg, device=torch.device("cuda"))
    ch_result = ch.run(seed=42)
    dt = time.time() - t0

    c_pred = ch_result["coeffs"]
    rmse = np.sqrt(np.mean((c_pred - c_true) ** 2))
    obj = ch_result["objective_value"]
    print(
        f"  PV={pv:5.1f} | RMSE={rmse:.4f} | obj={obj:.4f} | "
        f"time={dt:.1f}s | {description}"
    )
    return rmse


def main():
    print(f"CUDA: {torch.cuda.is_available()}, {torch.cuda.get_device_name(0)}")

    with open("configs/base_no_oracle.yaml") as f:
        cfg = yaml.safe_load(f)

    la = make_la(cfg)

    configs = [
        (
            "baseline (32 starts, 400 iter)",
            {},
        ),
        (
            "128 starts, 500 iter",
            {"chamfer_n_starts": 128, "chamfer_n_iter": 500},
        ),
        (
            "128 starts, 500 iter, lower temp (2.0->0.1)",
            {
                "chamfer_n_starts": 128,
                "chamfer_n_iter": 500,
                "chamfer_temp_start": 2.0,
                "chamfer_temp_end": 0.1,
            },
        ),
        (
            "128 starts, 500 iter, higher lr=0.1",
            {
                "chamfer_n_starts": 128,
                "chamfer_n_iter": 500,
                "chamfer_lr": 0.1,
            },
        ),
        (
            "256 starts, 300 iter, search_bound=4.0",
            {
                "chamfer_n_starts": 256,
                "chamfer_n_iter": 300,
                "chamfer_search_bound": 4.0,
            },
        ),
        (
            "128 starts, 500 iter, no reg",
            {
                "chamfer_n_starts": 128,
                "chamfer_n_iter": 500,
                "chamfer_lambda_reg": 0.0,
            },
        ),
        (
            "128 starts, 500 iter, low OOB penalty",
            {
                "chamfer_n_starts": 128,
                "chamfer_n_iter": 500,
                "chamfer_lambda_ib": 1.0,
            },
        ),
    ]

    test_pvs = [1.0, 5.0, 10.0]

    for desc, overrides in configs:
        print(f"\n=== {desc} ===")
        for pv in test_pvs:
            test_config(cfg, la, pv, desc, overrides)

    print("\n" + "=" * 75)
    print("Done.")


if __name__ == "__main__":
    main()
