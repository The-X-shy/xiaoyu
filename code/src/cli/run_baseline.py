"""CLI: Run baseline reconstruction on experiment samples."""

from __future__ import annotations
import argparse
import os
import copy
import numpy as np
import pandas as pd

from src.config import load_experiment_config
from src.eval.protocol import evaluate_method_at_pv, evaluate_single_sample
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
    parser = argparse.ArgumentParser(description="Run baseline reconstruction")
    parser.add_argument("--config", required=True, help="Experiment config YAML path")
    args = parser.parse_args()
    print(f"STAGE_START run_baseline config={args.config}", flush=True)

    cfg = load_experiment_config(args.config)
    sweep = cfg["sweep"]
    output = cfg["output"]
    base_seed = cfg["experiment"]["seed"]
    n_repeats = sweep.get("n_repeats", 20)

    os.makedirs(os.path.dirname(output["table"]), exist_ok=True)

    all_results = []

    if sweep["type"] == "pv_amplitude":
        pv_levels = np.arange(
            sweep["pv_min"], sweep["pv_max"] + sweep["pv_step"] / 2, sweep["pv_step"]
        )
        for pv in pv_levels:
            results = evaluate_method_at_pv(
                cfg=cfg,
                method="baseline",
                pv=float(pv),
                base_seed=base_seed,
                n_repeats=n_repeats,
            )
            all_results.extend(results)
            sr = sum(1 for r in results if r["success"]) / len(results)
            print(f"Baseline PV={pv:.1f}: success_rate={sr:.2f}")

    elif sweep["type"] == "missing_ratio":
        for pv in sweep["pv_levels"]:
            for mr in sweep["missing_ratios"]:
                results = evaluate_method_at_pv(
                    cfg=cfg,
                    method="baseline",
                    pv=float(pv),
                    base_seed=base_seed,
                    n_repeats=n_repeats,
                    missing_ratio=float(mr),
                )
                all_results.extend(results)
                sr = sum(1 for r in results if r["success"]) / len(results)
                print(f"Baseline PV={pv:.1f} MR={mr:.1f}: sr={sr:.2f}")
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
                pair_results = []
                for pv in pv_levels:
                    pv_results = []
                    for rep in range(n_repeats):
                        cfg_i = copy.deepcopy(cfg)
                        cfg_i["optics"]["pitch_um"] = float(pitch)
                        cfg_i["optics"]["focal_mm"] = float(focal)
                        seed = base_seed + rep
                        r = evaluate_single_sample(
                            cfg=cfg_i,
                            method="baseline",
                            pv=float(pv),
                            seed=seed,
                            missing_ratio=0.0,
                        )
                        r["pitch_um"] = float(pitch)
                        r["focal_mm"] = float(focal)
                        pv_results.append(r)
                        pair_results.append(r)
                        all_results.append(r)
                    pv_sr = sum(1 for r in pv_results if r["success"]) / len(pv_results)
                    pv_succ_rmse = [r["rmse"] for r in pv_results if r["success"]]
                    pv_mean_rmse = (
                        float(np.mean(pv_succ_rmse)) if pv_succ_rmse else float("inf")
                    )
                    if early_stop and (pv_sr < sr_min or pv_mean_rmse > rmse_max):
                        break
                sr = sum(1 for r in pair_results if r["success"]) / len(pair_results)
                succ_rmse = [r["rmse"] for r in pair_results if r["success"]]
                mean_rmse = float(np.mean(succ_rmse)) if succ_rmse else float("nan")
                dr = compute_dynamic_range(pair_results, rmse_max=rmse_max, sr_min=sr_min)
                print(
                    f"Baseline pitch={pitch:.0f}um focal={focal:.1f}mm: "
                    f"dr={dr:.1f}, sr={sr:.2f}, rmse={mean_rmse:.4f}"
                )

    df = pd.DataFrame(all_results)
    out_path = output["table"].replace("_results", "_baseline_results")
    df.to_csv(out_path, index=False)
    print(f"SOLVER=baseline rows={len(df)} out={out_path}", flush=True)
    print(f"STAGE_DONE run_baseline config={args.config}", flush=True)


if __name__ == "__main__":
    main()
