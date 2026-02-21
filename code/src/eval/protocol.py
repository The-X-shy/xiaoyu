"""Evaluation protocol: standardized sample evaluation."""

from __future__ import annotations
import time
import numpy as np
from typing import Dict, Any, List

from src.sim.pipeline import forward_pipeline
from src.recon.baseline_gpu import baseline_reconstruct_auto
from src.recon.asm_reconstructor import asm_reconstruct
from src.eval.metrics import rmse_coeffs

# Try to import GPU-accelerated ASM
_USE_GPU_ASM = False
try:
    import torch

    if torch.cuda.is_available():
        from src.recon.asm_gpu import asm_reconstruct_gpu

        _USE_GPU_ASM = True
except ImportError:
    pass


def evaluate_single_sample(
    cfg: Dict[str, Any],
    method: str,
    pv: float,
    seed: int,
    missing_ratio: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate a single sample with the specified method.

    Args:
        cfg: Full configuration dict.
        method: "baseline" or "asm".
        pv: Peak-to-valley amplitude (waves).
        seed: Random seed for this sample.
        missing_ratio: Fraction of spots to remove.

    Returns:
        Dict with method, pv_level, seed, success, rmse, missing_ratio,
        runtime_ms, and method-specific fields.
    """
    # Run forward simulation
    sim = forward_pipeline(cfg, pv=pv, seed=seed, missing_ratio=missing_ratio)

    t0 = time.perf_counter()

    if method == "baseline":
        recon = baseline_reconstruct_auto(
            sim["observed_positions"],
            sim["lenslet"],
            cfg,
        )
        coeffs_recon = recon["coeffs"]
        success = recon["success"]
        solver = recon.get("solver", "baseline_extrap_nn_cpu")
    elif method == "asm":
        asm_cfg = cfg.get("asm", {})
        use_oracle_index_hint = asm_cfg.get("use_oracle_index_hint", False)
        want_gpu = bool(asm_cfg.get("use_gpu", True))
        require_gpu = bool(asm_cfg.get("require_gpu", False))
        use_gpu = _USE_GPU_ASM and want_gpu
        if want_gpu and require_gpu and not _USE_GPU_ASM:
            raise RuntimeError("asm.require_gpu=true but CUDA ASM backend is unavailable")
        observed_sub_idx = sim.get("observed_sub_idx") if use_oracle_index_hint else None

        if use_gpu:
            recon = asm_reconstruct_gpu(
                sim["observed_positions"],
                sim["lenslet"],
                cfg,
                seed=seed + 1000,
                observed_sub_idx=observed_sub_idx,
                pv_hint=pv,
                missing_ratio_hint=missing_ratio,
            )
        else:
            recon = asm_reconstruct(
                sim["observed_positions"],
                sim["lenslet"],
                cfg,
                seed=seed + 1000,
                observed_sub_idx=observed_sub_idx,
                pv_hint=pv,
                missing_ratio_hint=missing_ratio,
            )
        coeffs_recon = recon["coeffs"]
        success = recon["success"]
        solver = recon.get("solver", "asm_icp")
    else:
        raise ValueError(f"Unknown method: {method}")

    runtime_ms = (time.perf_counter() - t0) * 1000

    if coeffs_recon is None:
        coeffs_recon = np.zeros_like(sim["coeffs"])
        success = False

    # Compute RMSE against true coefficients
    rmse = rmse_coeffs(sim["coeffs"], coeffs_recon, exclude_piston=True)

    # Check RMSE threshold for success determination
    eval_cfg = cfg.get("evaluation", {})
    rmse_max = eval_cfg.get("rmse_max_lambda", 0.15)
    if rmse > rmse_max:
        success = False

    return {
        "method": method,
        "pv_level": pv,
        "seed": seed,
        "success": success,
        "rmse": float(rmse),
        "missing_ratio": missing_ratio,
        "runtime_ms": float(runtime_ms),
        "objective_value": float(recon.get("objective_value", np.nan)),
        "n_matched": int(recon.get("n_matched", -1)),
        "residual_raw": float(recon.get("residual_raw", np.nan)),
        "residual_trimmed": float(recon.get("residual_trimmed", np.nan)),
        "solver": solver,
    }


def evaluate_method_at_pv(
    cfg: Dict[str, Any],
    method: str,
    pv: float,
    base_seed: int,
    n_repeats: int,
    missing_ratio: float = 0.0,
) -> List[Dict[str, Any]]:
    """Evaluate a method at a specific PV level with multiple repeats.

    Args:
        cfg: Full configuration dict.
        method: "baseline" or "asm".
        pv: Peak-to-valley amplitude.
        base_seed: Base random seed (each repeat uses base_seed + i).
        n_repeats: Number of independent repeats.
        missing_ratio: Fraction of missing spots.

    Returns:
        List of result dicts, one per repeat.
    """
    results = []
    for i in range(n_repeats):
        seed = base_seed + i
        result = evaluate_single_sample(
            cfg=cfg,
            method=method,
            pv=pv,
            seed=seed,
            missing_ratio=missing_ratio,
        )
        results.append(result)
    return results
