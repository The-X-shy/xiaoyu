"""CLI: Evaluate and compare baseline vs ASM results."""

from __future__ import annotations
import argparse
import os
import pandas as pd
import numpy as np

from src.eval.metrics import compute_dynamic_range, success_rate
from src.eval.statistics import compute_summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate experiment results")
    parser.add_argument("--baseline", required=True, help="Baseline results CSV")
    parser.add_argument("--asm", required=True, help="ASM results CSV")
    parser.add_argument("--rmse-max", type=float, default=0.15)
    parser.add_argument("--sr-min", type=float, default=0.95)
    args = parser.parse_args()

    df_bl = pd.read_csv(args.baseline)
    df_asm = pd.read_csv(args.asm)

    # Convert to record dicts
    bl_records = df_bl.to_dict("records")
    asm_records = df_asm.to_dict("records")

    dr_baseline = compute_dynamic_range(bl_records, rmse_max=args.rmse_max, sr_min=args.sr_min)
    dr_asm = compute_dynamic_range(asm_records, rmse_max=args.rmse_max, sr_min=args.sr_min)

    if dr_baseline > 0:
        range_gain = dr_asm / dr_baseline
    else:
        range_gain = float("inf") if dr_asm > 0 else 0.0

    print("=" * 60)
    print("DYNAMIC RANGE EVALUATION")
    print("=" * 60)
    print(f"Baseline DR:  {dr_baseline:.1f} waves (PV)")
    print(f"ASM DR:       {dr_asm:.1f} waves (PV)")
    print(f"Range Gain:   {range_gain:.1f}x")
    print(f"Target:       >= 14.0x")
    print(f"Result:       {'PASS' if range_gain >= 14.0 else 'FAIL'}")
    print("=" * 60)

    # Per-PV summary
    print("\nPer-PV breakdown:")
    print(f"{'PV':>6} {'BL_SR':>8} {'ASM_SR':>8} {'BL_RMSE':>10} {'ASM_RMSE':>10}")
    print("-" * 50)

    all_pvs = sorted(set(df_bl["pv_level"].unique()) | set(df_asm["pv_level"].unique()))
    for pv in all_pvs:
        bl_pv = df_bl[df_bl["pv_level"] == pv]
        asm_pv = df_asm[df_asm["pv_level"] == pv]

        bl_sr = bl_pv["success"].mean() if len(bl_pv) > 0 else 0.0
        asm_sr = asm_pv["success"].mean() if len(asm_pv) > 0 else 0.0

        bl_rmse_vals = bl_pv[bl_pv["success"] == True]["rmse"]
        asm_rmse_vals = asm_pv[asm_pv["success"] == True]["rmse"]

        bl_rmse = bl_rmse_vals.mean() if len(bl_rmse_vals) > 0 else float("nan")
        asm_rmse = asm_rmse_vals.mean() if len(asm_rmse_vals) > 0 else float("nan")

        print(f"{pv:6.1f} {bl_sr:8.2f} {asm_sr:8.2f} {bl_rmse:10.4f} {asm_rmse:10.4f}")

    # Save summary
    out_dir = os.path.dirname(args.baseline)
    summary = {
        "dr_baseline": dr_baseline,
        "dr_asm": dr_asm,
        "range_gain": range_gain,
        "target_gain": 14.0,
        "pass": range_gain >= 14.0,
    }
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(out_dir, "summary_metrics.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
