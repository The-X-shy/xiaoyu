"""DR scan: Baseline vs ASM (with sorting matcher) on GPU.

Usage:
    python scripts/dr_scan_sorting.py [--n-repeats 5] [--pv-max 80]

Runs both baseline and ASM at each PV level, prints results table,
and determines dynamic range and gain.
"""

import sys
import os
import time
import argparse
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_experiment_config
from src.eval.protocol import evaluate_single_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--pv-max", type=float, default=80.0)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    n_repeats = args.n_repeats
    base_seed = cfg["experiment"]["seed"]
    rmse_max = cfg.get("evaluation", {}).get("rmse_max_lambda", 0.15)
    sr_min = 0.6  # 3/5 = 0.6 is the minimum for DR determination

    # PV levels: fine at low end, coarser at high end
    pv_levels = sorted(
        set(
            list(np.arange(0.5, 5.5, 0.5))
            + list(np.arange(5.0, 20.0, 5.0))
            + list(np.arange(20.0, args.pv_max + 1, 10.0))
        )
    )

    print(f"DR Scan: n_repeats={n_repeats}, pv_max={args.pv_max}, rmse_max={rmse_max}")
    print(f"Sensor: {cfg['sensor']['width_px']}x{cfg['sensor']['height_px']}px")
    print(f"Sorting: {cfg.get('asm', {}).get('enable_sorting', False)}")
    print(f"GPU: {cfg.get('asm', {}).get('use_gpu', False)}")
    print()

    bl_results = {}  # pv -> list of success bools
    asm_results = {}  # pv -> list of (success, rmse)
    bl_dr = None
    bl_stopped = False

    t_total = time.time()

    for pv in pv_levels:
        bl_results[pv] = []
        asm_results[pv] = []

        # --- Baseline ---
        if not bl_stopped:
            for rep in range(n_repeats):
                seed = base_seed + rep
                t0 = time.time()
                r = evaluate_single_sample(cfg, "baseline", pv, seed)
                dt = time.time() - t0
                bl_results[pv].append(r["success"])
                status = "OK" if r["success"] else "FAIL"
                print(
                    f"  BL  PV={pv:5.1f} seed={seed} {status} rmse={r['rmse']:.4f} {dt:.1f}s"
                )

            bl_sr = sum(bl_results[pv]) / n_repeats
            if bl_sr < sr_min and bl_dr is None:
                # Previous PV was the DR boundary
                for prev_pv in sorted(bl_results.keys()):
                    if prev_pv < pv:
                        prev_sr = sum(bl_results[prev_pv]) / n_repeats
                        if prev_sr >= sr_min:
                            bl_dr = prev_pv
                if bl_dr is None:
                    bl_dr = 0.0
                bl_stopped = True
                print(f"  ** Baseline DR boundary: {bl_dr:.1f} lambda **")

        # --- ASM ---
        for rep in range(n_repeats):
            seed = base_seed + rep
            t0 = time.time()
            r = evaluate_single_sample(cfg, "asm", pv, seed)
            dt = time.time() - t0
            asm_results[pv].append((r["success"], r["rmse"], r.get("solver", "?")))
            status = "OK" if r["success"] else "FAIL"
            solver = r.get("solver", "?")
            print(
                f"  ASM PV={pv:5.1f} seed={seed} {status} rmse={r['rmse']:.4f} solver={solver} {dt:.1f}s"
            )

        asm_sr = sum(1 for s, _, _ in asm_results[pv] if s) / n_repeats
        print(
            f"  => PV={pv:.1f}: BL_sr={'N/A' if bl_stopped and pv not in bl_results else f'{sum(bl_results.get(pv, [])) / max(len(bl_results.get(pv, [])), 1):.2f}'}, ASM_sr={asm_sr:.2f}"
        )

        # Stop if ASM fails consistently at high PV
        if pv >= 30.0 and asm_sr < sr_min:
            # Check if the previous level also failed
            prev_pvs = [p for p in pv_levels if p < pv and p in asm_results]
            if prev_pvs:
                prev_pv = max(prev_pvs)
                prev_asm_sr = (
                    sum(1 for s, _, _ in asm_results[prev_pv] if s) / n_repeats
                )
                if prev_asm_sr < sr_min:
                    print(
                        f"  ** ASM failed at PV={prev_pv:.1f} and PV={pv:.1f}, stopping **"
                    )
                    break

        print()

    # Determine ASM DR
    asm_dr = 0.0
    for pv in sorted(asm_results.keys()):
        sr = sum(1 for s, _, _ in asm_results[pv] if s) / n_repeats
        if sr >= sr_min:
            asm_dr = pv

    if bl_dr is None:
        # Baseline never failed
        bl_dr = max(bl_results.keys())

    gain = asm_dr / bl_dr if bl_dr > 0 else float("inf")

    elapsed = time.time() - t_total

    # Print summary table
    print("\n" + "=" * 70)
    print("DYNAMIC RANGE SCAN RESULTS")
    print("=" * 70)
    print(f"{'PV':>6} {'BL_pass':>8} {'ASM_pass':>9} {'ASM_rmse':>10}")
    print("-" * 40)
    for pv in sorted(set(list(bl_results.keys()) + list(asm_results.keys()))):
        bl_pass = f"{sum(bl_results[pv])}/{n_repeats}" if pv in bl_results else "---"
        if pv in asm_results:
            asm_pass = f"{sum(1 for s, _, _ in asm_results[pv] if s)}/{n_repeats}"
            succ_rmse = [rm for s, rm, _ in asm_results[pv] if s]
            asm_rmse = f"{np.mean(succ_rmse):.4f}" if succ_rmse else "---"
        else:
            asm_pass = "---"
            asm_rmse = "---"
        print(f"{pv:6.1f} {bl_pass:>8} {asm_pass:>9} {asm_rmse:>10}")

    print()
    print(f"Baseline DR: {bl_dr:.1f} lambda")
    print(f"ASM DR:      {asm_dr:.1f} lambda")
    print(
        f"Gain:        {gain:.1f}x {'PASS' if gain >= 14.0 else 'FAIL'} (target >= 14x)"
    )
    print(f"Total time:  {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
