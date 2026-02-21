"""Smoke test: GPU ICP on full 2048x2048 sensor.
Tests multiple PV levels and checks for OOM.
"""

import time
import numpy as np
import torch
import yaml

from src.sim.pipeline import forward_pipeline
from src.recon.asm_gpu import asm_reconstruct_gpu
from src.recon.asm_reconstructor import asm_reconstruct
from src.eval.metrics import rmse_coeffs

# Load base config
with open("configs/base.yaml") as f:
    cfg = yaml.safe_load(f)

print(f"Sensor: {cfg['sensor']['width_px']}x{cfg['sensor']['height_px']}")
print(f"ASM n_starts={cfg['asm']['n_starts']}, n_icp_iter={cfg['asm']['n_icp_iter']}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

pv_levels = [0.5, 1.5, 2.5, 5.5, 10.5]

for pv in pv_levels:
    torch.cuda.empty_cache()
    print(f"--- PV={pv} ---")
    sim = forward_pipeline(cfg, pv=pv, seed=42, missing_ratio=0.0)
    obs = sim["observed_positions"]
    print(f"  Observed spots: {obs.shape[0]}")

    # GPU
    try:
        t0 = time.perf_counter()
        gpu_r = asm_reconstruct_gpu(obs, sim["lenslet"], cfg, seed=1042)
        t_gpu = time.perf_counter() - t0
        gpu_rmse = rmse_coeffs(sim["coeffs"], gpu_r["coeffs"], exclude_piston=True)
        print(
            f"  GPU: success={gpu_r['success']}, obj={gpu_r['objective_value']:.3f}, "
            f"rmse={gpu_rmse:.4f}, time={t_gpu:.1f}s"
        )
        print(f"  GPU VRAM peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    except torch.cuda.OutOfMemoryError as e:
        print(f"  GPU OOM: {e}")
    except Exception as e:
        print(f"  GPU error: {e}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    print()

print("Done!")
