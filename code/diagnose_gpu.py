"""Diagnostic: compare CPU vs GPU ICP on small sensor."""

import numpy as np
from src.sim.pipeline import forward_pipeline
from src.recon.asm_reconstructor import asm_reconstruct
from src.recon.asm_gpu import asm_reconstruct_gpu
from src.eval.metrics import rmse_coeffs

cfg = {
    "optics": {
        "wavelength_nm": 633.0,
        "pitch_um": 150.0,
        "focal_mm": 6.0,
        "fill_factor": 0.95,
    },
    "sensor": {"pixel_um": 10.0, "width_px": 512, "height_px": 512},
    "noise": {"read_sigma": 0.01, "background": 0.0, "centroid_noise_px": 0.05},
    "zernike": {"order": 3, "coeff_bound": 1.0},
    "baseline": {
        "tau_nn_px": 5.0,
        "k_fail": 3,
        "conflict_ratio_max": 0.2,
        "rmse_max": 0.15,
    },
    "asm": {
        "lambda_reg": 1e-3,
        "n_starts": 5,
        "n_icp_iter": 10,
        "convergence_tol": 1e-6,
    },
    "evaluation": {
        "rmse_max_lambda": 0.5,
        "success_rate_min": 0.95,
        "required_range_gain": 14.0,
        "n_repeats": 20,
    },
}

sim = forward_pipeline(cfg, pv=0.5, seed=42, missing_ratio=0.0)
print(f"Observed: {sim['observed_positions'].shape}")
print(f"True coeffs: {sim['coeffs']}")

# CPU
cpu_r = asm_reconstruct(sim["observed_positions"], sim["lenslet"], cfg, seed=1042)
cpu_rmse = rmse_coeffs(sim["coeffs"], cpu_r["coeffs"], exclude_piston=True)
print(
    f"CPU: success={cpu_r['success']}, obj={cpu_r['objective_value']:.3f}, rmse={cpu_rmse:.4f}"
)
print(f"CPU coeffs: {cpu_r['coeffs']}")

# GPU
gpu_r = asm_reconstruct_gpu(sim["observed_positions"], sim["lenslet"], cfg, seed=1042)
gpu_rmse = rmse_coeffs(sim["coeffs"], gpu_r["coeffs"], exclude_piston=True)
print(
    f"GPU: success={gpu_r['success']}, obj={gpu_r['objective_value']:.3f}, rmse={gpu_rmse:.4f}"
)
print(f"GPU coeffs: {gpu_r['coeffs']}")

# Also check what evaluate_single_sample returns
from src.eval.protocol import evaluate_single_sample, _USE_GPU_ASM

print(f"\n_USE_GPU_ASM = {_USE_GPU_ASM}")
out = evaluate_single_sample(cfg=cfg, method="asm", pv=0.5, seed=42)
print(f"evaluate_single_sample: success={out['success']}, rmse={out['rmse']:.4f}")
