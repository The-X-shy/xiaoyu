"""CLI: Generate simulation samples and save to disk."""

from __future__ import annotations
import argparse
import os
import numpy as np

from src.config import load_config, load_experiment_config
from src.sim.pipeline import forward_pipeline


def main():
    parser = argparse.ArgumentParser(description="Generate SHWS simulation samples")
    parser.add_argument("--config", required=True, help="Experiment config YAML path")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    sweep = cfg["sweep"]
    output = cfg["output"]
    base_seed = cfg["experiment"]["seed"]

    raw_dir = output["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)

    sweep_type = sweep["type"]

    if sweep_type == "pv_amplitude":
        pv_min = sweep["pv_min"]
        pv_max = sweep["pv_max"]
        pv_step = sweep["pv_step"]
        n_repeats = sweep["n_repeats"]
        pv_levels = np.arange(pv_min, pv_max + pv_step / 2, pv_step)

        for pv in pv_levels:
            for rep in range(n_repeats):
                seed = base_seed + int(pv * 1000) + rep
                result = forward_pipeline(cfg, pv=pv, seed=seed)
                fname = f"sample_pv{pv:.1f}_rep{rep:03d}.npz"
                np.savez(
                    os.path.join(raw_dir, fname),
                    coeffs=result["coeffs"],
                    observed_positions=result["observed_positions"],
                    ref_positions=result["ref_positions"],
                    slopes=result["slopes"],
                    pv=pv,
                    seed=seed,
                    keep_mask=result["keep_mask"],
                )
        print(f"Generated {len(pv_levels) * n_repeats} samples in {raw_dir}")

    elif sweep_type == "missing_ratio":
        pv_levels = sweep["pv_levels"]
        missing_ratios = sweep["missing_ratios"]
        n_repeats = sweep["n_repeats"]

        count = 0
        for pv in pv_levels:
            for mr in missing_ratios:
                for rep in range(n_repeats):
                    seed = base_seed + int(pv * 1000) + int(mr * 100) + rep
                    result = forward_pipeline(
                        cfg, pv=pv, seed=seed, missing_ratio=mr
                    )
                    fname = f"sample_pv{pv:.1f}_mr{mr:.1f}_rep{rep:03d}.npz"
                    np.savez(
                        os.path.join(raw_dir, fname),
                        coeffs=result["coeffs"],
                        observed_positions=result["observed_positions"],
                        ref_positions=result["ref_positions"],
                        slopes=result["slopes"],
                        pv=pv,
                        seed=seed,
                        missing_ratio=mr,
                        keep_mask=result["keep_mask"],
                    )
                    count += 1
        print(f"Generated {count} samples in {raw_dir}")

    elif sweep_type == "param_scan":
        print("Param scan uses run_baseline/run_asm directly, skipping simulate.")
    else:
        raise ValueError(f"Unknown sweep type: {sweep_type}")


if __name__ == "__main__":
    main()
