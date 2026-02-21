"""CLI: Generate param-scan recommendation tables and figures."""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _format_value_list(values: Iterable[float]) -> str:
    vals = sorted(float(v) for v in set(values))
    if not vals:
        return "-"
    return ", ".join(f"{v:g}" for v in vals)


def _plot_heatmap(
    pivot: pd.DataFrame,
    title: str,
    out_path: str,
    cmap: str = "viridis",
    fmt: str = "{:.1f}",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    data = pivot.values
    im = ax.imshow(data, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("focal_mm")
    ax.set_ylabel("pitch_um")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{c:g}" for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{i:g}" for i in pivot.index])

    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            v = data[r, c]
            txt = "nan" if np.isnan(v) else fmt.format(v)
            ax.text(c, r, txt, ha="center", va="center", color="white", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(title, rotation=90)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate param-scan report assets")
    parser.add_argument(
        "--summary",
        default="outputs/tables/param_scan_summary.csv",
        help="Input param-scan summary CSV",
    )
    parser.add_argument(
        "--out-fig-dir",
        default="outputs/figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--out-table-dir",
        default="outputs/tables",
        help="Output directory for tables",
    )
    parser.add_argument(
        "--out-report-dir",
        default="outputs/reports",
        help="Output directory for markdown report",
    )
    args = parser.parse_args()

    os.makedirs(args.out_fig_dir, exist_ok=True)
    os.makedirs(args.out_table_dir, exist_ok=True)
    os.makedirs(args.out_report_dir, exist_ok=True)

    df = pd.read_csv(args.summary)
    req_cols = {
        "pitch_um",
        "focal_mm",
        "range_gain",
        "asm_dr",
        "asm_rmse",
        "baseline_dr",
        "pass_14x",
    }
    if not req_cols.issubset(df.columns):
        missing = sorted(req_cols - set(df.columns))
        raise ValueError(f"Missing required columns in summary CSV: {missing}")

    # 1) Heatmaps
    pivot_gain = (
        df.pivot(index="pitch_um", columns="focal_mm", values="range_gain")
        .sort_index()
        .sort_index(axis=1)
    )
    pivot_asm_dr = (
        df.pivot(index="pitch_um", columns="focal_mm", values="asm_dr")
        .sort_index()
        .sort_index(axis=1)
    )
    _plot_heatmap(
        pivot_gain,
        title="Range Gain (ASM / Baseline)",
        out_path=os.path.join(args.out_fig_dir, "param_scan_range_gain_heatmap.png"),
        cmap="magma",
        fmt="{:.1f}",
    )
    _plot_heatmap(
        pivot_asm_dr,
        title="ASM Dynamic Range (waves)",
        out_path=os.path.join(args.out_fig_dir, "param_scan_asm_dr_heatmap.png"),
        cmap="viridis",
        fmt="{:.1f}",
    )

    # 2) Tables
    top10 = (
        df.sort_values(["range_gain", "asm_dr", "asm_rmse"], ascending=[False, False, True])
        .head(10)
        .copy()
    )
    top10_path = os.path.join(args.out_table_dir, "param_scan_recommendation_top10.csv")
    top10.to_csv(top10_path, index=False)

    pass14 = df[df["pass_14x"] == True].copy()
    pass14_path = os.path.join(args.out_table_dir, "param_scan_pass14x.csv")
    pass14.sort_values(["range_gain", "asm_dr"], ascending=[False, False]).to_csv(
        pass14_path, index=False
    )

    # Recommended robust region: pass_14x and keep asm_rmse low + asm_dr high.
    robust = df[(df["pass_14x"] == True) & (df["asm_dr"] >= 20.0) & (df["asm_rmse"] <= 0.01)].copy()
    robust_path = os.path.join(args.out_table_dir, "param_scan_recommendation_robust.csv")
    robust.sort_values(["range_gain", "asm_dr"], ascending=[False, False]).to_csv(
        robust_path, index=False
    )

    interval_rows = []
    for pitch, g in pass14.groupby("pitch_um"):
        best = g.sort_values(["range_gain", "asm_dr"], ascending=[False, False]).iloc[0]
        interval_rows.append(
            {
                "pitch_um": float(pitch),
                "pass14x_focal_mm_list": _format_value_list(g["focal_mm"].tolist()),
                "best_focal_mm_by_gain": float(best["focal_mm"]),
                "best_range_gain": float(best["range_gain"]),
                "best_asm_dr": float(best["asm_dr"]),
            }
        )
    interval_df = pd.DataFrame(interval_rows).sort_values("pitch_um")
    interval_path = os.path.join(args.out_table_dir, "param_scan_recommendation_intervals.csv")
    interval_df.to_csv(interval_path, index=False)

    # 3) Markdown report
    report_path = os.path.join(args.out_report_dir, "param_scan_recommendation.md")
    best_row = top10.iloc[0]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Param Scan Recommendation\n\n")
        f.write("## Core Summary\n")
        f.write(f"- Total parameter pairs: {len(df)}\n")
        f.write(f"- Pairs with `pass_14x=True`: {int(df['pass_14x'].sum())}\n")
        f.write(f"- Max `range_gain`: {float(df['range_gain'].max()):.2f}\n")
        f.write(f"- Median `range_gain`: {float(df['range_gain'].median()):.2f}\n\n")
        f.write("## Best Pair\n")
        f.write(
            f"- pitch_um={best_row['pitch_um']:.0f}, focal_mm={best_row['focal_mm']:.1f}, "
            f"range_gain={best_row['range_gain']:.2f}, asm_dr={best_row['asm_dr']:.1f}, "
            f"baseline_dr={best_row['baseline_dr']:.1f}, asm_rmse={best_row['asm_rmse']:.4f}\n\n"
        )
        f.write("## Recommended Robust Region Rule\n")
        f.write("- `pass_14x == True`\n")
        f.write("- `asm_dr >= 20.0`\n")
        f.write("- `asm_rmse <= 0.01`\n\n")
        f.write("## Generated Files\n")
        f.write("- `outputs/figures/param_scan_range_gain_heatmap.png`\n")
        f.write("- `outputs/figures/param_scan_asm_dr_heatmap.png`\n")
        f.write("- `outputs/tables/param_scan_recommendation_top10.csv`\n")
        f.write("- `outputs/tables/param_scan_pass14x.csv`\n")
        f.write("- `outputs/tables/param_scan_recommendation_robust.csv`\n")
        f.write("- `outputs/tables/param_scan_recommendation_intervals.csv`\n")

    print(f"Saved heatmaps to: {args.out_fig_dir}")
    print(f"Saved recommendation tables to: {args.out_table_dir}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
