#!/usr/bin/env python3
"""Run official evaluation: ASM with NN warm-start vs baseline.

Measures dynamic range for both methods and computes gain.
Saves per-sample CSV results for reproducibility.
"""

import sys, os, csv, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import time

from src.eval.protocol import evaluate_single_sample
from src.eval.metrics import compute_dynamic_range, success_rate


def main():
    cfg = yaml.safe_load(open("configs/base_no_oracle.yaml"))

    # Ensure no oracle hints
    cfg.setdefault("asm", {})
    cfg["asm"]["use_oracle_index_hint"] = False
    cfg["asm"]["use_gpu"] = True
    cfg["asm"]["enable_nn_warmstart"] = True
    cfg["asm"]["enable_chamfer"] = True
    cfg["asm"]["enable_sorting"] = True

    # Evaluation parameters
    n_repeats = 100
    base_seed = 800000
    pv_levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    log_path = f"outputs/logs/nn_ensemble_eval_{ts}.log"
    bl_csv_path = f"outputs/tables/nn_ensemble_eval_baseline_{ts}.csv"
    asm_csv_path = f"outputs/tables/nn_ensemble_eval_asm_{ts}.csv"
    summary_csv_path = f"outputs/tables/nn_ensemble_eval_summary_{ts}.csv"

    # Helper to write one record to CSV
    csv_fields = [
        "method",
        "pv_level",
        "seed",
        "success",
        "rmse",
        "missing_ratio",
        "runtime_ms",
        "solver",
    ]

    def write_csv(path, records):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            w.writeheader()
            for r in records:
                w.writerow(r)

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    # Verify NN models exist
    nn_models = ["models/nn_warmstart.pt", "models/nn_v3_resnet.pt"]
    for mp in nn_models:
        exists = os.path.exists(mp)
        log(f"NN model {mp}: {'FOUND' if exists else 'MISSING'}")
        if not exists:
            log(f"ERROR: Required NN model {mp} not found. Aborting.")
            return

    log(f"\nEvaluation: {n_repeats} samples per PV level")
    log(f"PV levels: {pv_levels}")
    log(f"Base seed: {base_seed}")
    log(f"Timestamp: {ts}")
    log("")

    # ===== Run baseline evaluation =====
    log("=" * 60)
    log("BASELINE METHOD")
    log("=" * 60)

    baseline_records = []
    for pv in pv_levels:
        t0 = time.time()
        pv_results = []
        for i in range(n_repeats):
            seed = base_seed + i
            result = evaluate_single_sample(cfg, "baseline", pv, seed)
            pv_results.append(result)
        dt = time.time() - t0

        sr = success_rate(pv_results)
        rmses = [r["rmse"] for r in pv_results if r.get("success", False)]
        mean_rmse = np.mean(rmses) if rmses else float("inf")
        status = "PASS" if sr >= 0.95 and mean_rmse <= 0.15 else "FAIL"

        baseline_records.extend(pv_results)
        log(
            f"  PV={pv:4.1f} | SR={sr * 100:5.1f}% | mean_RMSE={mean_rmse:.4f} | {status} | {dt:.1f}s"
        )

    write_csv(bl_csv_path, baseline_records)
    baseline_dr = compute_dynamic_range(baseline_records)
    log(f"\n  Baseline dynamic range: {baseline_dr}")
    log(f"  Saved: {bl_csv_path}")

    # ===== Run ASM evaluation (with NN ensemble) =====
    log("")
    log("=" * 60)
    log("ASM METHOD (with NN ensemble warm-start)")
    log("=" * 60)

    asm_records = []
    for pv in pv_levels:
        t0 = time.time()
        pv_results = []
        for i in range(n_repeats):
            seed = base_seed + i
            result = evaluate_single_sample(cfg, "asm", pv, seed)
            pv_results.append(result)

            # Progress every 10 samples
            if (i + 1) % 10 == 0:
                partial_sr = success_rate(pv_results)
                log(
                    f"    PV={pv:.1f} [{i + 1}/{n_repeats}] partial_sr={partial_sr:.2f} solver={result.get('solver', '?')}"
                )

        dt = time.time() - t0
        sr = success_rate(pv_results)
        rmses = [r["rmse"] for r in pv_results if r.get("success", False)]
        mean_rmse = np.mean(rmses) if rmses else float("inf")
        status = "PASS" if sr >= 0.95 and mean_rmse <= 0.15 else "FAIL"

        # Count solver types
        solvers = {}
        for r in pv_results:
            s = r.get("solver", "unknown")
            solvers[s] = solvers.get(s, 0) + 1

        asm_records.extend(pv_results)
        log(
            f"  PV={pv:4.1f} | SR={sr * 100:5.1f}% | mean_RMSE={mean_rmse:.4f} | {status} | {dt:.1f}s | solvers={solvers}"
        )

    write_csv(asm_csv_path, asm_records)
    asm_dr = compute_dynamic_range(asm_records)
    log(f"\n  ASM dynamic range: {asm_dr}")
    log(f"  Saved: {asm_csv_path}")

    # ===== Summary =====
    log("")
    log("=" * 60)
    log("FINAL SUMMARY")
    log("=" * 60)

    if baseline_dr > 0:
        gain = asm_dr / baseline_dr
        log(f"  Baseline DR: {baseline_dr}")
        log(f"  ASM DR:      {asm_dr}")
        log(f"  Gain:        {gain:.2f}x")
        log(f"  Target:      >= 14.0x")
        log(f"  Result:      {'PASS' if gain >= 14.0 else 'FAIL'}")
    else:
        gain = 0.0
        log(f"  Baseline DR=0, ASM DR={asm_dr}")

    # Per-PV breakdown
    log("\n  Per-PV breakdown:")
    log(
        f"  {'PV':>5s}  {'BL_SR':>6s}  {'ASM_SR':>6s}  {'BL_RMSE':>8s}  {'ASM_RMSE':>8s}  {'BL':>4s}  {'ASM':>4s}"
    )
    log("  " + "-" * 55)

    summary_rows = []
    bl_by_pv = {}
    asm_by_pv = {}
    for r in baseline_records:
        bl_by_pv.setdefault(r["pv_level"], []).append(r)
    for r in asm_records:
        asm_by_pv.setdefault(r["pv_level"], []).append(r)

    for pv in pv_levels:
        bl_group = bl_by_pv.get(pv, [])
        asm_group = asm_by_pv.get(pv, [])
        bl_sr = success_rate(bl_group)
        asm_sr = success_rate(asm_group)
        bl_rmses = [r["rmse"] for r in bl_group if r.get("success", False)]
        asm_rmses = [r["rmse"] for r in asm_group if r.get("success", False)]
        bl_mean = np.mean(bl_rmses) if bl_rmses else float("nan")
        asm_mean = np.mean(asm_rmses) if asm_rmses else float("nan")
        bl_status = "PASS" if bl_sr >= 0.95 and bl_mean <= 0.15 else "FAIL"
        asm_status = "PASS" if asm_sr >= 0.95 and asm_mean <= 0.15 else "FAIL"
        log(
            f"  {pv:5.1f}  {bl_sr:6.2f}  {asm_sr:6.2f}  {bl_mean:8.4f}  {asm_mean:8.4f}  {bl_status:>4s}  {asm_status:>4s}"
        )
        summary_rows.append(
            {
                "pv": pv,
                "bl_sr": bl_sr,
                "asm_sr": asm_sr,
                "bl_mean_rmse": bl_mean,
                "asm_mean_rmse": asm_mean,
                "bl_status": bl_status,
                "asm_status": asm_status,
            }
        )

    # Save summary CSV
    with open(summary_csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "pv",
                "bl_sr",
                "asm_sr",
                "bl_mean_rmse",
                "asm_mean_rmse",
                "bl_status",
                "asm_status",
            ],
        )
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    log(f"\n  Summary saved: {summary_csv_path}")
    log(f"  Log saved: {log_path}")
    log(f"\n  baseline_dr={baseline_dr}, asm_dr={asm_dr}, gain={gain:.2f}x")
    log("\nDone!")


if __name__ == "__main__":
    main()
