#!/usr/bin/env python3
"""Quick smoke test: Chamfer optimizer on GPU at various PV levels."""

import torch
import numpy as np
import yaml
import time
from src.sim.pipeline import forward_pipeline
from src.recon.chamfer_optimizer import ChamferOptimizer
from src.recon.zernike import num_zernike_terms
from src.sim.lenslet import LensletArray


def main():
    print(f"CUDA: {torch.cuda.is_available()}, {torch.cuda.get_device_name(0)}")

    with open("configs/base_no_oracle.yaml") as f:
        cfg = yaml.safe_load(f)

    opt = cfg["optics"]
    sen = cfg["sensor"]
    la = LensletArray(
        pitch_um=opt["pitch_um"],
        focal_mm=opt["focal_mm"],
        fill_factor=opt["fill_factor"],
        sensor_width_px=sen["width_px"],
        sensor_height_px=sen["height_px"],
        pixel_um=sen["pixel_um"],
    )
    n = num_zernike_terms(3)
    print(f"n_sub={la.n_subapertures}, n_terms={n}")
    print("-" * 75)

    for pv in [1.0, 3.0, 5.0, 8.0, 10.0, 15.0]:
        sim = forward_pipeline(cfg, pv=pv, seed=42)
        obs = sim["observed_positions"]
        c_true = sim["coeffs"]

        t0 = time.time()
        ch = ChamferOptimizer(obs, la, cfg, device=torch.device("cuda"))
        ch_result = ch.run(seed=42)
        dt = time.time() - t0

        c_pred = ch_result["coeffs"]
        rmse = np.sqrt(np.mean((c_pred - c_true) ** 2))
        obj = ch_result["objective_value"]
        print(
            f"PV={pv:5.1f} | N_obs={len(obs):5d} | "
            f"RMSE={rmse:.4f} | obj={obj:.4f} | time={dt:.2f}s"
        )

    print("-" * 75)
    print("Done. RMSE <= 0.15 is success threshold.")


if __name__ == "__main__":
    main()
