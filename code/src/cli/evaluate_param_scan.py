"""CLI: Evaluate parameter-scan results for baseline vs ASM."""

from __future__ import annotations
import argparse
import os
import pandas as pd
import numpy as np

from src.eval.metrics import compute_dynamic_range


def _summarize(
    df: pd.DataFrame, method_name: str, rmse_max: float, sr_min: float
) -> pd.DataFrame:
    rows = []
    for (pitch, focal), g in df.groupby(["pitch_um", "focal_mm"]):
        sr = float(g["success"].mean())
        succ = g[g["success"] == True]["rmse"]
        mean_rmse = float(succ.mean()) if len(succ) else np.nan
        dr = compute_dynamic_range(
            g.to_dict("records"),
            rmse_max=rmse_max,
            sr_min=sr_min,
        )
        rows.append(
            {
                "pitch_um": float(pitch),
                "focal_mm": float(focal),
                f"{method_name}_sr": float(sr),
                f"{method_name}_rmse": mean_rmse,
                f"{method_name}_dr": float(dr),
                f"{method_name}_n": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate param-scan comparison")
    parser.add_argument("--baseline", required=True, help="Baseline CSV")
    parser.add_argument("--asm", required=True, help="ASM CSV")
    parser.add_argument(
        "--out",
        default="outputs/tables/param_scan_summary.csv",
        help="Output summary CSV path",
    )
    parser.add_argument("--rmse-max", type=float, default=0.15)
    parser.add_argument("--sr-min", type=float, default=0.95)
    args = parser.parse_args()

    df_bl = pd.read_csv(args.baseline)
    df_asm = pd.read_csv(args.asm)

    req_cols = {"pitch_um", "focal_mm", "success", "rmse"}
    if not req_cols.issubset(df_bl.columns) or not req_cols.issubset(df_asm.columns):
        raise ValueError("Input CSV missing required columns for param scan summary")

    s_bl = _summarize(df_bl, "baseline", rmse_max=args.rmse_max, sr_min=args.sr_min)
    s_asm = _summarize(df_asm, "asm", rmse_max=args.rmse_max, sr_min=args.sr_min)

    merged = s_bl.merge(s_asm, on=["pitch_um", "focal_mm"], how="outer")
    merged["delta_sr"] = merged["asm_sr"] - merged["baseline_sr"]
    merged["rmse_gain"] = merged["baseline_rmse"] - merged["asm_rmse"]
    merged["range_gain"] = np.where(
        merged["baseline_dr"] > 0.0,
        merged["asm_dr"] / merged["baseline_dr"],
        np.where(merged["asm_dr"] > 0.0, np.inf, 0.0),
    )
    merged["pass_14x"] = merged["range_gain"] >= 14.0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    merged.sort_values(["pitch_um", "focal_mm"]).to_csv(args.out, index=False)

    print("=" * 60)
    print("PARAM SCAN SUMMARY")
    print("=" * 60)
    print(f"Saved: {args.out}")
    top = merged.sort_values(["range_gain", "asm_dr"], ascending=[False, False]).head(10)
    print("\nTop-10 parameter pairs by range_gain then asm_dr:")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
