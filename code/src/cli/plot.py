"""CLI: Generate experiment figures."""

from __future__ import annotations
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Generate experiment plots")
    parser.add_argument("--baseline", required=True, help="Baseline results CSV")
    parser.add_argument("--asm", required=True, help="ASM results CSV")
    parser.add_argument("--outdir", default="outputs/figures", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df_bl = pd.read_csv(args.baseline)
    df_asm = pd.read_csv(args.asm)

    # Check if this is a missing_ratio experiment
    has_mr = "missing_ratio" in df_bl.columns and df_bl["missing_ratio"].nunique() > 1

    if has_mr:
        _plot_missing_spot(df_bl, df_asm, args.outdir)
    else:
        _plot_dynamic_range(df_bl, df_asm, args.outdir)

    print(f"Plots saved to {args.outdir}")


def _plot_dynamic_range(df_bl, df_asm, outdir):
    """Plot success rate and RMSE vs PV for dynamic range experiment."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for df, label, color in [(df_bl, "Baseline", "blue"), (df_asm, "ASM", "red")]:
        pvs = sorted(df["pv_level"].unique())
        srs = []
        rmses = []
        for pv in pvs:
            sub = df[df["pv_level"] == pv]
            srs.append(sub["success"].mean())
            succ = sub[sub["success"] == True]
            rmses.append(succ["rmse"].mean() if len(succ) > 0 else float("nan"))

        ax1.plot(pvs, srs, "o-", label=label, color=color)
        ax2.plot(pvs, rmses, "o-", label=label, color=color)

    ax1.set_xlabel("PV (waves)")
    ax1.set_ylabel("Success Rate")
    ax1.set_title("Success Rate vs PV Amplitude")
    ax1.axhline(0.95, ls="--", color="gray", alpha=0.5, label="Threshold (0.95)")
    ax1.legend()
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("PV (waves)")
    ax2.set_ylabel("RMSE (waves)")
    ax2.set_title("Reconstruction RMSE vs PV Amplitude")
    ax2.axhline(0.15, ls="--", color="gray", alpha=0.5, label="Threshold (0.15)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dynamic_range.png"), dpi=150)
    plt.close()


def _plot_missing_spot(df_bl, df_asm, outdir):
    """Plot success rate vs missing ratio for different PV levels."""
    fig, ax = plt.subplots(figsize=(8, 6))

    pvs = sorted(df_bl["pv_level"].unique())
    for pv in pvs:
        for df, label, ls in [(df_bl, "Baseline", "--"), (df_asm, "ASM", "-")]:
            sub_pv = df[df["pv_level"] == pv]
            if len(sub_pv) == 0:
                continue
            mrs = sorted(sub_pv["missing_ratio"].unique())
            srs = []
            for mr in mrs:
                sub = sub_pv[sub_pv["missing_ratio"] == mr]
                srs.append(sub["success"].mean())
            ax.plot(mrs, srs, f"o{ls}", label=f"{label} PV={pv:.0f}")

    ax.set_xlabel("Missing Spot Ratio")
    ax.set_ylabel("Success Rate")
    ax.set_title("Robustness to Missing Spots")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "missing_spots.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
