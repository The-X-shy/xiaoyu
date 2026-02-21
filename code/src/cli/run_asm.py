"""CLI: Run ASM reconstruction on experiment samples."""

from __future__ import annotations
import argparse
import os
import sys
import time
import copy
import numpy as np
import pandas as pd

from src.config import load_experiment_config
from src.eval.protocol import evaluate_single_sample
from src.eval.metrics import compute_dynamic_range


def _param_scan_pv_levels(sweep: dict) -> list[float]:
    """Resolve PV levels for param-scan sweep with backward compatibility."""
    if "pv_levels" in sweep:
        return [float(v) for v in sweep["pv_levels"]]
    if all(k in sweep for k in ("pv_min", "pv_max", "pv_step")):
        pv_min = float(sweep["pv_min"])
        pv_max = float(sweep["pv_max"])
        pv_step = float(sweep["pv_step"])
        return [float(v) for v in np.arange(pv_min, pv_max + pv_step / 2, pv_step)]
    if "pv_test" in sweep:
        return [float(sweep["pv_test"])]
    return [5.0]


def main():
    parser = argparse.ArgumentParser(description="Run ASM reconstruction")
    parser.add_argument("--config", required=True, help="Experiment config YAML path")
    args = parser.parse_args()
    print(f"STAGE_START run_asm config={args.config}", flush=True)

    cfg = load_experiment_config(args.config)
    sweep = cfg["sweep"]
    output = cfg["output"]
    base_seed = cfg["experiment"]["seed"]
    n_repeats = sweep.get("n_repeats", 20)

    os.makedirs(os.path.dirname(output["table"]), exist_ok=True)

    all_results = []

    # Build list of (pv, missing_ratio) jobs
    jobs = []
    if sweep["type"] == "pv_amplitude":
        pv_levels = np.arange(
            sweep["pv_min"], sweep["pv_max"] + sweep["pv_step"] / 2, sweep["pv_step"]
        )
        for pv in pv_levels:
            for rep in range(n_repeats):
                jobs.append((float(pv), 0.0, base_seed + rep))
    elif sweep["type"] == "missing_ratio":
        for pv in sweep["pv_levels"]:
            for mr in sweep["missing_ratios"]:
                for rep in range(n_repeats):
                    jobs.append((float(pv), float(mr), base_seed + rep))
    elif sweep["type"] == "param_scan":
        pitch_list = sweep["pitch_um_list"]
        focal_list = sweep["focal_mm_list"]
        pv_levels = _param_scan_pv_levels(sweep)
        eval_cfg = cfg.get("evaluation", {})
        rmse_max = float(eval_cfg.get("rmse_max_lambda", 0.15))
        sr_min = float(eval_cfg.get("success_rate_min", 0.95))
        early_stop = bool(sweep.get("early_stop_on_failure", True))

        for pitch in pitch_list:
            for focal in focal_list:
                for pv in pv_levels:
                    pv_results = []
                    for rep in range(n_repeats):
                        seed = base_seed + rep
                        cfg_i = copy.deepcopy(cfg)
                        cfg_i["optics"]["pitch_um"] = float(pitch)
                        cfg_i["optics"]["focal_mm"] = float(focal)
                        t0 = time.time()
                        result = evaluate_single_sample(
                            cfg=cfg_i,
                            method="asm",
                            pv=float(pv),
                            seed=seed,
                            missing_ratio=0.0,
                        )
                        result["pitch_um"] = float(pitch)
                        result["focal_mm"] = float(focal)
                        pv_results.append(result)
                        all_results.append(result)

                        dt = time.time() - t0
                        status = "OK" if result["success"] else "FAIL"
                        print(
                            f"[param_scan] pitch={pitch:.0f} focal={focal:.1f} PV={pv:.1f} "
                            f"seed={seed} {status} rmse={result['rmse']:.4f} {dt:.1f}s",
                            flush=True,
                        )

                    pv_sr = sum(1 for r in pv_results if r["success"]) / len(pv_results)
                    pv_succ_rmse = [r["rmse"] for r in pv_results if r["success"]]
                    pv_mean_rmse = (
                        float(np.mean(pv_succ_rmse)) if pv_succ_rmse else float("inf")
                    )
                    if early_stop and (pv_sr < sr_min or pv_mean_rmse > rmse_max):
                        break

    if sweep["type"] != "param_scan":
        total = len(jobs)
        t_start = time.time()

        for i, job in enumerate(jobs):
            pv, mr, seed = job
            t0 = time.time()
            result = evaluate_single_sample(
                cfg=cfg,
                method="asm",
                pv=pv,
                seed=seed,
                missing_ratio=mr,
            )
            dt = time.time() - t0
            all_results.append(result)

            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (total - i - 1)
            status = "OK" if result["success"] else "FAIL"
            print(
                f"[{i + 1}/{total}] PV={pv:.1f} seed={seed} "
                f"{status} rmse={result['rmse']:.4f} "
                f"{dt:.1f}s (ETA {eta / 60:.1f}min)",
                flush=True,
            )

    # Print per-PV summary
    df = pd.DataFrame(all_results)
    for col in ["objective_value", "n_matched", "solver", "residual_raw", "residual_trimmed"]:
        if col not in df.columns:
            df[col] = np.nan
    if sweep["type"] == "param_scan":
        print("\n--- Param-Scan Summary (ASM) ---")
        eval_cfg = cfg.get("evaluation", {})
        rmse_max = float(eval_cfg.get("rmse_max_lambda", 0.15))
        sr_min = float(eval_cfg.get("success_rate_min", 0.95))
        grouped = df.groupby(["pitch_um", "focal_mm"])
        for (p, f), g in grouped:
            sr = g["success"].mean()
            succ_rmse = g[g["success"] == True]["rmse"]
            mean_rmse = succ_rmse.mean() if len(succ_rmse) else float("nan")
            dr = compute_dynamic_range(g.to_dict("records"), rmse_max=rmse_max, sr_min=sr_min)
            print(
                f"ASM pitch={p:.0f}um focal={f:.1f}mm: "
                f"dr={dr:.1f}, sr={sr:.2f}, rmse={mean_rmse:.4f}"
            )
    else:
        print("\n--- Per-PV Summary ---")
        for pv in sorted(df["pv_level"].unique()):
            sub = df[df["pv_level"] == pv]
            sr = sub["success"].mean()
            mean_rmse = (
                sub[sub["success"] == True]["rmse"].mean() if sr > 0 else float("nan")
            )
            print(f"ASM PV={pv:.1f}: sr={sr:.2f}, mean_rmse={mean_rmse:.4f}")

    out_path = output["table"].replace("_results", "_asm_results")
    df.to_csv(out_path, index=False)
    print(f"\nSOLVER=asm rows={len(df)} out={out_path}", flush=True)
    print(f"STAGE_DONE run_asm config={args.config}", flush=True)


if __name__ == "__main__":
    main()
