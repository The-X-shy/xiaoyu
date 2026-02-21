#!/usr/bin/env python3
"""Diagnostic: evaluate backward Chamfer objective at true coefficients.

This tells us whether the objective has its global minimum at the truth.
If not, the objective itself is broken and no search strategy will help.
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


def main():
    print(f"CUDA: {torch.cuda.is_available()}, {torch.cuda.get_device_name(0)}")

    with open("configs/base_no_oracle.yaml") as f:
        cfg = yaml.safe_load(f)

    la = make_la(cfg)
    n = num_zernike_terms(3)

    for pv in [1.0, 3.0, 5.0, 8.0, 10.0, 15.0]:
        sim = forward_pipeline(cfg, pv=pv, seed=42)
        obs = sim["observed_positions"]
        c_true = sim["coeffs"]

        ch = ChamferOptimizer(obs, la, cfg, device=torch.device("cuda"))

        # Evaluate objective at TRUE coefficients
        c_true_t = torch.tensor(
            c_true.reshape(1, -1).astype(np.float32),
            dtype=torch.float32,
            device=torch.device("cuda"),
        )

        # Use the full observation/prediction sets for fair evaluation
        rng = np.random.RandomState(999)
        obs_f, ref_f, G_f, k_obs_f, k_pred_f = ch._make_subsets(
            min(2048, ch.n_obs), min(4096, ch.n_sub), rng
        )

        with torch.no_grad():
            obj_true = ch._backward_chamfer_objective(
                c_true_t, obs_f, ref_f, G_f, k_pred_f
            ).item()

        # Evaluate at zero
        c_zero_t = torch.zeros((1, n), dtype=torch.float32, device=torch.device("cuda"))
        with torch.no_grad():
            obj_zero = ch._backward_chamfer_objective(
                c_zero_t, obs_f, ref_f, G_f, k_pred_f
            ).item()

        # Evaluate at Chamfer's best result
        ch_result = ch.run(seed=42)
        c_best = ch_result["coeffs"]
        rmse = np.sqrt(np.mean((c_best - c_true) ** 2))
        obj_best = ch_result["objective_value"]

        # Also evaluate sampling phase objective at truth (coarse subsets)
        rng2 = np.random.RandomState(42)  # same seed as run()
        obs_s, ref_s, G_s, k_obs_s, k_pred_s = ch._make_subsets(
            ch.sample_obs_k, ch.sample_pred_k, rng2
        )
        with torch.no_grad():
            obj_true_coarse = ch._backward_chamfer_objective(
                c_true_t, obs_s, ref_s, G_s, k_pred_s
            ).item()

        print(
            f"PV={pv:5.1f} | N_obs={len(obs):5d} | "
            f"obj@true={obj_true:.4f} | obj@true_coarse={obj_true_coarse:.4f} | "
            f"obj@zero={obj_zero:.4f} | "
            f"obj@best={obj_best:.4f} | RMSE={rmse:.4f} | "
            f"c_true_norm={np.linalg.norm(c_true):.2f}"
        )

    print("\nIf obj@true < obj@best, then the search is failing.")
    print("If obj@true > obj@best, then the objective is wrong.")


if __name__ == "__main__":
    main()
