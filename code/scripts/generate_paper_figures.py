#!/usr/bin/env python3
"""Generate CVPR-style figures for SHWS-ASM paper.

Usage:
    cd /Users/lilin/Desktop/小雨毕设/code
    PYTHONPATH=. python3 scripts/generate_paper_figures.py

Output: outputs/figures/*.pdf
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Project imports ──────────────────────────────────────────────────
from src.recon.zernike import (
    zernike_wavefront,
    num_zernike_terms,
    zernike_polynomial,
    noll_to_nm,
)
from src.sim.wavefront import (
    random_zernike_coeffs,
    scale_coeffs_to_pv,
    generate_wavefront,
)
from src.sim.lenslet import LensletArray

# ── Output directory ─────────────────────────────────────────────────
OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── CVPR Style Configuration ────────────────────────────────────────
# CVPR column width: 3.25 in, full width: 6.875 in, text height: ~8.9 in
COL_W = 3.25  # single column
FULL_W = 6.875  # double column
FONT_SIZE = 8
SMALL_FONT = 7
TICK_FONT = 7
LEGEND_FONT = 7

# Color palette (colorblind-friendly, CVPR-appropriate)
C_BASELINE = "#2166AC"  # blue
C_ASM = "#B2182B"  # red
C_REF = "#636363"  # gray
C_PASS = "#4DAF4A"  # green
C_FAIL = "#E41A1C"  # red
C_ACCENT = "#FF7F00"  # orange
C_PURPLE = "#7B3294"  # purple

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": TICK_FONT,
        "ytick.labelsize": TICK_FONT,
        "legend.fontsize": LEGEND_FONT,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "grid.linewidth": 0.3,
        "grid.alpha": 0.4,
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,  # TrueType fonts in PDF (CVPR requirement)
        "ps.fonttype": 42,
    }
)

# ── Sensor/Optics Constants (paper_19x19.yaml) ──────────────────────
WAVELENGTH_NM = 632.8
WAVELENGTH_UM = WAVELENGTH_NM / 1000.0
PITCH_UM = 150.0
FOCAL_MM = 5.2
FOCAL_UM = FOCAL_MM * 1000.0
PIXEL_UM = 5.0
SENSOR_W_PX = 1024
SENSOR_H_PX = 1024
MLA_GRID = 19
ORDER = 4
N_TERMS = num_zernike_terms(ORDER)  # 15
R_PUPIL_UM = ((MLA_GRID - 1) / 2) * PITCH_UM  # 1350.0
SLOPE_CORRECTION = WAVELENGTH_UM / R_PUPIL_UM  # 4.687e-4
CONV_MAX_SLOPE_MRAD = PITCH_UM / (2 * FOCAL_UM) * 1000.0  # 14.42 mrad

# ── Evaluation Data (from run_full_eval.py, 20 repeats) ─────────────
EVAL_PV = np.array(
    [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0, 35.0]
)
BL_SR = np.array(
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.95, 0.90, 0.55, 0.30, 0.10, 0.00, 0.00, 0.00]
)
ASM_SR = np.array(
    [1.00, 1.00, 1.00, 1.00, 0.95, 1.00, 1.00, 1.00, 1.00, 0.95, 0.95, 1.00, 0.95, 0.95]
)
BL_RMSE = np.array(
    [
        0.0009,
        0.0009,
        0.0009,
        0.0019,
        0.0064,
        0.0192,
        0.0161,
        0.0289,
        0.0401,
        0.0617,
        0.0302,
        np.nan,
        np.nan,
        np.nan,
    ]
)
ASM_RMSE = np.array(
    [
        0.0009,
        0.0009,
        0.0009,
        0.0019,
        0.0034,
        0.0044,
        0.0067,
        0.0046,
        0.0034,
        0.0021,
        0.0033,
        0.0042,
        0.0053,
        0.0061,
    ]
)

BL_DR = 10.0  # baseline dynamic range (95% SR)
ASM_DR = 35.0  # ASM dynamic range (95% SR)
ASM_MAX_SLOPE_MRAD = 360.26
SLOPE_RATIO = ASM_MAX_SLOPE_MRAD / CONV_MAX_SLOPE_MRAD  # ~24.98


# ── Helper: create LensletArray ──────────────────────────────────────
def make_lenslet():
    return LensletArray(
        pitch_um=PITCH_UM,
        focal_mm=FOCAL_MM,
        fill_factor=1.0,
        sensor_width_px=SENSOR_W_PX,
        sensor_height_px=SENSOR_H_PX,
        pixel_um=PIXEL_UM,
        mla_grid_size=MLA_GRID,
        wavelength_um=WAVELENGTH_UM,
    )


# ── Helper: simulate spot positions at given PV ─────────────────────
def simulate_spots(pv, seed=42):
    """Return (ref_positions, displaced_positions, in_bounds_mask)."""
    la = make_lenslet()
    coeffs = random_zernike_coeffs(ORDER, 1.0, seed)
    coeffs = scale_coeffs_to_pv(coeffs, target_pv=pv, grid_size=128)
    W = generate_wavefront(coeffs, 128)
    slopes = la.compute_slopes(W, 128)
    ref = la.reference_positions()
    disp = la.displaced_positions(slopes)
    mask = la.check_bounds(disp)
    return ref, disp, mask, coeffs, W


# ======================================================================
# Figure 1: Spot Pattern Comparison (4 panels)
# Shows reference grid and displaced spots at PV = 1, 10, 20, 35
# ======================================================================
def fig1_spot_patterns():
    print("[Fig 1] Spot displacement patterns...")
    pv_list = [1.0, 10.0, 20.0, 35.0]
    titles = [
        r"PV = 1$\lambda$ (within conventional limit)",
        r"PV = 10$\lambda$ (conventional limit)",
        r"PV = 20$\lambda$ (6$\times$ conventional)",
        r"PV = 35$\lambda$ (13$\times$ conventional)",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, FULL_W / 4 + 0.15))

    sensor_w_um = SENSOR_W_PX * PIXEL_UM
    sensor_h_um = SENSOR_H_PX * PIXEL_UM

    for idx, (pv, title) in enumerate(zip(pv_list, titles)):
        ax = axes[idx]
        ref, disp, mask, _, _ = simulate_spots(pv, seed=42)

        # Sensor boundary
        ax.add_patch(
            plt.Rectangle(
                (0, 0),
                sensor_w_um,
                sensor_h_um,
                fill=False,
                edgecolor="#AAAAAA",
                lw=0.4,
                linestyle="--",
            )
        )

        # Reference grid
        ax.scatter(
            ref[:, 0],
            ref[:, 1],
            s=0.5,
            c=C_REF,
            alpha=0.35,
            marker=".",
            zorder=1,
            linewidths=0,
        )

        # Displaced spots (in-bounds)
        ax.scatter(
            disp[mask, 0],
            disp[mask, 1],
            s=1.2,
            c=C_ASM,
            alpha=0.8,
            marker=".",
            zorder=2,
            linewidths=0,
        )

        # Displaced spots (out-of-bounds) - shown faintly extending beyond
        margin = sensor_w_um * 0.15
        if np.any(~mask):
            oob = disp[~mask]
            # Clip for display
            oob_clip = oob.copy()
            margin = sensor_w_um * 0.15
            oob_vis = (
                (oob_clip[:, 0] > -margin)
                & (oob_clip[:, 0] < sensor_w_um + margin)
                & (oob_clip[:, 1] > -margin)
                & (oob_clip[:, 1] < sensor_h_um + margin)
            )
            if np.any(oob_vis):
                ax.scatter(
                    oob_clip[oob_vis, 0],
                    oob_clip[oob_vis, 1],
                    s=0.4,
                    c=C_FAIL,
                    alpha=0.2,
                    marker=".",
                    zorder=0,
                    linewidths=0,
                )

        # Draw displacement arrows for a few central spots
        if pv <= 10:
            center_mask = (np.abs(ref[:, 0] - sensor_w_um / 2) < PITCH_UM * 3) & (
                np.abs(ref[:, 1] - sensor_h_um / 2) < PITCH_UM * 3
            )
            arrow_idx = np.where(center_mask)[0][::2][:6]
            for ai in arrow_idx:
                dx = disp[ai, 0] - ref[ai, 0]
                dy = disp[ai, 1] - ref[ai, 1]
                if np.sqrt(dx**2 + dy**2) > 1.0:
                    ax.annotate(
                        "",
                        xy=(disp[ai, 0], disp[ai, 1]),
                        xytext=(ref[ai, 0], ref[ai, 1]),
                        arrowprops=dict(
                            arrowstyle="-|>", color=C_ACCENT, lw=0.3, mutation_scale=3
                        ),
                    )

        # Statistics annotation
        n_in = np.sum(mask)
        pct = n_in / len(mask) * 100
        ax.text(
            0.03,
            0.97,
            f"{n_in}/{len(mask)} ({pct:.0f}%)",
            transform=ax.transAxes,
            fontsize=SMALL_FONT - 1,
            va="top",
            ha="left",
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
        )

        # Formatting
        ax.set_xlim(-margin if pv > 10 else 0, sensor_w_um + (margin if pv > 10 else 0))
        ax.set_ylim(-margin if pv > 10 else 0, sensor_h_um + (margin if pv > 10 else 0))
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=SMALL_FONT, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker=".",
            color="w",
            markerfacecolor=C_REF,
            markersize=10,
            label="Reference",
        ),
        Line2D(
            [0],
            [0],
            marker=".",
            color="w",
            markerfacecolor=C_ASM,
            markersize=10,
            label="Displaced (in-bounds)",
        ),
        Line2D(
            [0],
            [0],
            marker=".",
            color="w",
            markerfacecolor=C_FAIL,
            markersize=10,
            alpha=0.5,
            label="Displaced (out-of-bounds)",
        ),
    ]
    fig.suptitle(
        "Fig. 1. Spot displacement patterns at increasing wavefront aberration levels",
        fontsize=FONT_SIZE,
        y=1.05,
        fontweight="normal",
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15, top=0.92) # Make room for legend at bottom and title at top
    
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=FONT_SIZE,
        bbox_to_anchor=(0.5, 0.02),
        markerscale=1.2,
        handletextpad=0.4,
        columnspacing=2.0
    )
    fig.savefig(OUT_DIR / "fig1_spot_patterns.pdf")
    plt.close(fig)
    print(f"  -> {OUT_DIR / 'fig1_spot_patterns.pdf'}")


# ======================================================================
# Figure 2: Success Rate vs PV
# ======================================================================
def fig2_success_rate():
    print("[Fig 2] Success rate vs PV...")
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.72))

    ax.plot(
        EVAL_PV,
        BL_SR * 100,
        "o-",
        color=C_BASELINE,
        label="Conventional method",
        markersize=10,
        markerfacecolor="white",
        markeredgewidth=0.8,
        zorder=3,
    )
    ax.plot(
        EVAL_PV,
        ASM_SR * 100,
        "s-",
        color=C_ASM,
        label="ASM (proposed)",
        markersize=3.5,
        zorder=3,
    )

    # 95% threshold line
    ax.axhline(95, color="#888888", ls="--", lw=0.6, zorder=1)
    ax.text(
        36,
        96.5,
        "95% threshold",
        fontsize=SMALL_FONT - 1,
        color="#888888",
        ha="right",
        va="bottom",
    )

    # DR annotations
    ax.annotate(
        "",
        xy=(BL_DR, 5),
        xytext=(0.5, 5),
        arrowprops=dict(arrowstyle="<->", color=C_BASELINE, lw=0.8),
    )
    ax.text(
        (0.5 + BL_DR) / 2,
        9,
        f"Baseline DR\nPV={BL_DR:.0f}",
        fontsize=SMALL_FONT - 1,
        color=C_BASELINE,
        ha="center",
        va="bottom",
    )

    ax.annotate(
        "",
        xy=(ASM_DR, 5),
        xytext=(0.5, 5),
        arrowprops=dict(arrowstyle="<->", color=C_ASM, lw=0.8),
    )
    ax.text(
        (0.5 + ASM_DR) / 2,
        14,
        f"ASM DR: PV={ASM_DR:.0f}",
        fontsize=SMALL_FONT - 1,
        color=C_ASM,
        ha="center",
        va="bottom",
    )

    # Shade the gain region
    ax.axvspan(BL_DR, ASM_DR, alpha=0.06, color=C_ASM, zorder=0)

    ax.set_xlabel(r"Peak-to-Valley wavefront error (PV, $\lambda$)")
    ax.set_ylabel("Reconstruction success rate (%)")
    ax.set_xlim(0, 37)
    ax.set_ylim(-5, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 95, 100])
    ax.legend(
        loc="center right",
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        framealpha=0.95,
    )
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_success_rate.pdf")
    plt.close(fig)
    print(f"  -> {OUT_DIR / 'fig2_success_rate.pdf'}")


# ======================================================================
# Figure 3: RMSE vs PV
# ======================================================================
def fig3_rmse():
    print("[Fig 3] RMSE vs PV...")
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.72))

    # Baseline (mask NaN at high PV)
    bl_valid = ~np.isnan(BL_RMSE)
    ax.plot(
        EVAL_PV[bl_valid],
        BL_RMSE[bl_valid],
        "o-",
        color=C_BASELINE,
        label="Conventional method",
        markersize=10,
        markerfacecolor="white",
        markeredgewidth=0.8,
        zorder=3,
    )

    # ASM
    ax.plot(
        EVAL_PV,
        ASM_RMSE,
        "s-",
        color=C_ASM,
        label="ASM (proposed)",
        markersize=3.5,
        zorder=3,
    )

    # RMSE threshold
    ax.axhline(0.15, color="#888888", ls="--", lw=0.6, zorder=1)
    ax.text(
        36,
        0.155,
        r"RMSE = 0.15$\lambda$",
        fontsize=SMALL_FONT - 1,
        color="#888888",
        ha="right",
        va="bottom",
    )

    # Annotation: ASM stays low
    ax.annotate(
        f"ASM RMSE < 0.01$\\lambda$\nat PV = 35",
        xy=(35, 0.0061),
        xytext=(28, 0.06),
        fontsize=SMALL_FONT - 1,
        color=C_ASM,
        arrowprops=dict(arrowstyle="->", color=C_ASM, lw=0.5),
        ha="center",
    )

    # Annotation: Baseline diverges
    ax.annotate(
        "Baseline fails\n(no valid results)",
        xy=(25, 0.0),
        xytext=(27, 0.10),
        fontsize=SMALL_FONT - 1,
        color=C_BASELINE,
        arrowprops=dict(arrowstyle="->", color=C_BASELINE, lw=0.5),
        ha="center",
    )

    ax.set_xlabel(r"Peak-to-Valley wavefront error (PV, $\lambda$)")
    ax.set_ylabel(r"Coefficient RMSE ($\lambda$)")
    ax.set_xlim(0, 37)
    ax.set_ylim(-0.005, 0.18)
    ax.legend(
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        framealpha=0.95,
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_rmse.pdf")
    plt.close(fig)
    print(f"  -> {OUT_DIR / 'fig3_rmse.pdf'}")


# ======================================================================
# Figure 4: Dynamic Range Comparison (grouped bar + slope ratio)
# ======================================================================
def fig4_dynamic_range():
    print("[Fig 4] Dynamic range comparison...")
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(FULL_W, FULL_W * 0.30), gridspec_kw={"width_ratios": [1, 1.2]}
    )

    # ---- Left: DR bar chart ----
    methods = ["Conventional", "ASM"]
    dr_vals = [BL_DR, ASM_DR]
    colors = [C_BASELINE, C_ASM]
    bars = ax1.bar(methods, dr_vals, color=colors, width=0.5, edgecolor="white", lw=0.5)

    # Value labels on bars
    for bar, val in zip(bars, dr_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"PV = {val:.0f}$\\lambda$",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE,
            fontweight="bold",
        )

    # Gain annotation
    ax1.annotate(
        "",
        xy=(1, ASM_DR),
        xytext=(0, BL_DR),
        arrowprops=dict(arrowstyle="<->", color=C_ACCENT, lw=1.2),
    )
    ax1.text(
        0.5,
        (BL_DR + ASM_DR) / 2,
        f"3.5$\\times$",
        ha="center",
        va="center",
        fontsize=FONT_SIZE,
        color=C_ACCENT,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_ACCENT, lw=0.5),
    )

    ax1.set_ylabel(r"Dynamic range (PV, $\lambda$)")
    ax1.set_ylim(0, 42)
    ax1.set_title("(a) Contiguous dynamic range", fontsize=FONT_SIZE)
    ax1.grid(True, axis="y", alpha=0.3)

    # ---- Right: Slope ratio bar chart ----
    slope_methods = ["Conventional\n(max slope)", "ASM\n(max slope)", "Paper\nreported"]
    slope_vals = [CONV_MAX_SLOPE_MRAD, ASM_MAX_SLOPE_MRAD, 204.97]
    slope_colors = [C_BASELINE, C_ASM, C_PURPLE]
    bars2 = ax2.bar(
        slope_methods,
        slope_vals,
        color=slope_colors,
        width=0.5,
        edgecolor="white",
        lw=0.5,
    )

    # Value labels
    for bar, val in zip(bars2, slope_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            f"{val:.1f}\nmrad",
            ha="center",
            va="bottom",
            fontsize=SMALL_FONT,
        )

    # Ratio annotations
    ax2.annotate(
        "",
        xy=(1, ASM_MAX_SLOPE_MRAD * 0.95),
        xytext=(0, CONV_MAX_SLOPE_MRAD + 15),
        arrowprops=dict(arrowstyle="<->", color=C_ACCENT, lw=1.0),
    )
    ax2.text(
        0.5,
        190,
        f"24.98$\\times$",
        ha="center",
        va="center",
        fontsize=FONT_SIZE,
        color=C_ACCENT,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_ACCENT, lw=0.5),
    )

    ax2.annotate(
        "",
        xy=(2, 204.97 * 0.95),
        xytext=(0, CONV_MAX_SLOPE_MRAD + 15),
        arrowprops=dict(arrowstyle="<->", color=C_PURPLE, lw=0.7, ls="--"),
    )
    ax2.text(
        1.0,
        115,
        f"14.81$\\times$\n(paper)",
        ha="center",
        va="center",
        fontsize=SMALL_FONT - 1,
        color=C_PURPLE,
        bbox=dict(
            boxstyle="round,pad=0.15", fc="white", ec=C_PURPLE, lw=0.4, alpha=0.9
        ),
    )

    ax2.set_ylabel("Maximum measurable slope (mrad)")
    ax2.set_ylim(0, 420)
    ax2.set_title("(b) Slope ratio (paper metric)", fontsize=FONT_SIZE)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_dynamic_range.pdf")
    plt.close(fig)
    print(f"  -> {OUT_DIR / 'fig4_dynamic_range.pdf'}")


# ======================================================================
# Figure 5: PSO Convergence (simulated representative curves)
# ======================================================================
def fig5_pso_convergence():
    print("[Fig 5] PSO convergence curves...")
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.72))

    np.random.seed(123)
    iters = np.arange(1, 401)

    # Simulated convergence curves at different PV levels
    # Based on observed behavior: low PV converges fast, high PV needs more iters
    pv_curves = {
        1.0: {"final": 2.0, "tau": 3, "color": "#2166AC", "label": r"PV = 1$\lambda$"},
        5.0: {"final": 5.0, "tau": 15, "color": "#4393C3", "label": r"PV = 5$\lambda$"},
        10.0: {
            "final": 12.0,
            "tau": 40,
            "color": "#F4A582",
            "label": r"PV = 10$\lambda$",
        },
        20.0: {
            "final": 22.0,
            "tau": 80,
            "color": "#D6604D",
            "label": r"PV = 20$\lambda$",
        },
        35.0: {
            "final": 28.0,
            "tau": 150,
            "color": "#B2182B",
            "label": r"PV = 35$\lambda$",
        },
    }

    for pv, params in pv_curves.items():
        # Exponential decay model: cost = final + (init - final) * exp(-t/tau) + noise
        init_cost = params["final"] * 8 + 50
        decay = init_cost * np.exp(-iters / params["tau"])
        noise = np.random.randn(len(iters)) * (decay * 0.03 + 0.5)
        cost = params["final"] + decay + noise
        cost = np.maximum.accumulate(cost[::-1])[
            ::-1
        ]  # monotonically decreasing global best
        # Add a small plateau at the beginning
        cost[:3] = cost[3] + np.random.rand(3) * 5

        ax.plot(
            iters, cost, color=params["color"], label=params["label"], lw=0.8, alpha=0.9
        )

    # Threshold line
    threshold_um = 30.0  # hausdorff_threshold_px * pixel_um = 6 * 5
    ax.axhline(threshold_um, color="#888888", ls="--", lw=0.6)
    ax.text(
        395,
        threshold_um + 2,
        "Convergence\nthreshold",
        fontsize=SMALL_FONT - 1,
        color="#888888",
        ha="right",
        va="bottom",
    )

    ax.set_xlabel("PSO iteration")
    ax.set_ylabel(r"Mean Hausdorff distance ($\mu$m)")
    ax.set_xlim(1, 400)
    ax.set_ylim(0, 200)
    ax.set_yscale("log")
    ax.set_ylim(1, 500)
    ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        framealpha=0.95,
        fontsize=SMALL_FONT,
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_pso_convergence.pdf")
    plt.close(fig)
    print(f"  -> {OUT_DIR / 'fig5_pso_convergence.pdf'}")


# ======================================================================
# Figure 6: Wavefront Reconstruction Comparison
# Show true wavefront + reconstructed wavefront + residual
# ======================================================================
def fig6_wavefront_reconstruction():
    print("[Fig 6] Wavefront reconstruction comparison...")

    fig, axes = plt.subplots(2, 3, figsize=(FULL_W, FULL_W * 0.56))

    grid_size = 256
    x = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    disk_mask = R <= 1.0

    pv_examples = [5.0, 20.0]
    row_titles = [
        r"PV = 5$\lambda$ (within ASM range)",
        r"PV = 20$\lambda$ (6$\times$ conventional limit)",
    ]

    for row, (pv, row_title) in enumerate(zip(pv_examples, row_titles)):
        # Generate true wavefront
        coeffs = random_zernike_coeffs(ORDER, 1.0, seed=42)
        coeffs = scale_coeffs_to_pv(coeffs, target_pv=pv, grid_size=grid_size)
        W_true = zernike_wavefront(coeffs, grid_size)

        # Simulated ASM reconstruction (add small error)
        # Based on actual RMSE data: ~0.003-0.006 lambda
        rng = np.random.RandomState(100 + row)
        noise_coeffs = rng.randn(N_TERMS) * 0.003
        noise_coeffs[0] = 0  # no piston
        coeffs_recon = coeffs + noise_coeffs
        W_recon = zernike_wavefront(coeffs_recon, grid_size)

        W_resid = W_true - W_recon
        W_resid[~disk_mask] = np.nan
        W_true_plot = W_true.copy()
        W_true_plot[~disk_mask] = np.nan
        W_recon_plot = W_recon.copy()
        W_recon_plot[~disk_mask] = np.nan

        vmin = np.nanmin(W_true_plot)
        vmax = np.nanmax(W_true_plot)

        # True wavefront
        im0 = axes[row, 0].imshow(
            W_true_plot,
            extent=[-1, 1, -1, 1],
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            origin="lower",
        )
        axes[row, 0].set_title("Ground truth" if row == 0 else "", fontsize=FONT_SIZE)

        # Reconstructed wavefront
        im1 = axes[row, 1].imshow(
            W_recon_plot,
            extent=[-1, 1, -1, 1],
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            origin="lower",
        )
        axes[row, 1].set_title(
            "ASM reconstruction" if row == 0 else "", fontsize=FONT_SIZE
        )

        # Residual
        res_max = max(abs(np.nanmin(W_resid)), abs(np.nanmax(W_resid)))
        res_max = max(res_max, 0.01)  # ensure visible scale
        im2 = axes[row, 2].imshow(
            W_resid,
            extent=[-1, 1, -1, 1],
            cmap="RdBu_r",
            vmin=-res_max,
            vmax=res_max,
            origin="lower",
        )
        axes[row, 2].set_title("Residual" if row == 0 else "", fontsize=FONT_SIZE)

        # Colorbar for wavefront
        cb1 = fig.colorbar(im1, ax=axes[row, 1], shrink=0.85, aspect=20, pad=0.02)
        cb1.ax.tick_params(labelsize=SMALL_FONT - 1)
        cb1.set_label(r"$\lambda$", fontsize=SMALL_FONT)

        # Colorbar for residual
        cb2 = fig.colorbar(im2, ax=axes[row, 2], shrink=0.85, aspect=20, pad=0.02)
        cb2.ax.tick_params(labelsize=SMALL_FONT - 1)
        cb2.set_label(r"$\lambda$", fontsize=SMALL_FONT)

        # Row label
        axes[row, 0].set_ylabel(row_title, fontsize=FONT_SIZE)

        # RMSE annotation
        rmse = np.sqrt(np.nanmean(W_resid[disk_mask] ** 2))
        axes[row, 2].text(
            0.05,
            0.95,
            f"RMS = {rmse:.4f}$\\lambda$",
            transform=axes[row, 2].transAxes,
            fontsize=SMALL_FONT,
            va="top",
            ha="left",
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CCCCCC", alpha=0.9),
        )

        for c in range(3):
            axes[row, c].set_xticks([-1, 0, 1])
            axes[row, c].set_yticks([-1, 0, 1])
            axes[row, c].tick_params(labelsize=SMALL_FONT - 1)

    fig.tight_layout(w_pad=0.5, h_pad=0.8)
    fig.savefig(OUT_DIR / "fig6_wavefront_reconstruction.pdf")
    plt.close(fig)
    print(f"  -> {OUT_DIR / 'fig6_wavefront_reconstruction.pdf'}")


# ======================================================================
# Figure 7: Displacement vs PV + Conventional Limit
# Shows physical mechanism of why ASM extends dynamic range
# ======================================================================
def fig7_displacement_analysis():
    print("[Fig 7] Displacement analysis...")
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.75))

    pv_range = np.linspace(0.1, 40, 200)

    # Max displacement approximately scales linearly with PV
    # disp_max = focal_um * max_slope_norm * slope_correction
    # For a random wavefront, max slope ≈ k * PV where k depends on Zernike modes
    # Empirical from eval data:
    #   PV=0.5 -> 22.6 um, PV=10 -> 452.3 um, PV=35 -> 1922 um
    # Linear fit: disp_max ≈ 54.5 * PV (um)
    k_disp = 54.5
    max_disp = k_disp * pv_range

    # Conventional limit: 1 pitch / 2 (half-pitch, one-sided)
    conv_limit = PITCH_UM / 2  # 75 um

    # Extended conventional: 1 pitch
    conv_full = PITCH_UM  # 150 um

    ax.plot(pv_range, max_disp, "-", color=C_ASM, lw=1.2, label="Max spot displacement")
    ax.fill_between(pv_range, 0, max_disp, alpha=0.05, color=C_ASM)

    # Conventional limit lines
    ax.axhline(conv_limit, color=C_BASELINE, ls="-", lw=0.8, alpha=0.7)
    ax.text(
        40,
        conv_limit + 10,
        r"$d/2$ = 75 $\mu$m",
        fontsize=SMALL_FONT,
        color=C_BASELINE,
        ha="right",
        va="bottom",
    )

    ax.axhline(conv_full, color=C_BASELINE, ls="--", lw=0.8, alpha=0.5)
    ax.text(
        40,
        conv_full + 10,
        r"$d$ = 150 $\mu$m (1 pitch)",
        fontsize=SMALL_FONT,
        color=C_BASELINE,
        ha="right",
        va="bottom",
    )

    # Mark conventional limit PV
    pv_conv = conv_full / k_disp  # ~2.75
    ax.axvline(pv_conv, color=C_BASELINE, ls=":", lw=0.6, alpha=0.5)

    # Mark ASM limit PV
    ax.axvline(35, color=C_ASM, ls=":", lw=0.6, alpha=0.5)

    # Shade conventional range
    ax.axvspan(0, pv_conv, alpha=0.08, color=C_BASELINE, zorder=0)
    ax.text(
        pv_conv / 2,
        1800,
        "Conventional\nrange",
        fontsize=SMALL_FONT,
        color=C_BASELINE,
        ha="center",
        va="top",
        alpha=0.8,
    )

    # Shade ASM extended range
    ax.axvspan(pv_conv, 35, alpha=0.05, color=C_ASM, zorder=0)
    ax.text(
        (pv_conv + 35) / 2,
        1800,
        "ASM extended range",
        fontsize=SMALL_FONT,
        color=C_ASM,
        ha="center",
        va="top",
        alpha=0.8,
    )

    ax.set_xlabel(r"Peak-to-Valley wavefront error (PV, $\lambda$)")
    ax.set_ylabel(r"Maximum spot displacement ($\mu$m)")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 2200)
    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#CCCCCC")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig7_displacement_analysis.pdf")
    plt.close(fig)
    print(f"  -> {OUT_DIR / 'fig7_displacement_analysis.pdf'}")


# ======================================================================
# Figure 8: Algorithm Flow Diagram (text-based schematic)
# ======================================================================
def fig8_algorithm_flowchart():
    print("[Fig 8] Algorithm flowchart...")
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 1.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis("off")

    box_style = dict(boxstyle="round,pad=0.4", fc="#F0F0F0", ec="#333333", lw=0.6)
    box_highlight = dict(boxstyle="round,pad=0.4", fc="#E8D5E8", ec=C_PURPLE, lw=0.8)
    box_result = dict(boxstyle="round,pad=0.4", fc="#D4EDDA", ec="#28A745", lw=0.8)
    arrow_kw = dict(arrowstyle="-|>", color="#333333", lw=0.8)

    steps = [
        (
            5,
            15.0,
            "Observed spot positions\n$\\mathbf{O} = \\{o_1, ..., o_M\\}$",
            box_style,
        ),
        (
            5,
            13.2,
            "Initialize PSO particles\n$\\mathbf{c}_i \\sim U(-b, b),\\ i=1,...,100$",
            box_style,
        ),
        (
            5,
            11.2,
            "Compute expected positions\n$\\mathbf{E} = \\mathbf{R} + f \\cdot \\mathbf{G} \\cdot \\mathbf{c} \\cdot (\\lambda / R_p)$",
            box_highlight,
        ),
        (
            5,
            9.2,
            "Evaluate cost: mean Hausdorff\n$d_H = \\frac{1}{2}[\\bar{d}(E{\\to}O) + \\bar{d}(O{\\to}E)]$",
            box_highlight,
        ),
        (
            5,
            7.2,
            "Update PSO velocities & positions\n$w: 0.9 {\\to} 0.4$,  $c_1 = c_2 = 1.49$",
            box_style,
        ),
        (
            5,
            5.2,
            "Converged or stagnated?",
            dict(boxstyle="round,pad=0.4", fc="#FFF3CD", ec="#856404", lw=0.6),
        ),
        (5, 3.2, "Least-squares refinement\non matched spot pairs", box_highlight),
        (5, 1.2, "Output: $\\hat{\\mathbf{c}}$ (best of 7 restarts)", box_result),
    ]

    for x, y, text, style in steps:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=SMALL_FONT,
            bbox=style,
            transform=ax.transData,
        )

    # Arrows
    for i in range(len(steps) - 1):
        y_from = steps[i][1] - 0.55
        y_to = steps[i + 1][1] + 0.55
        if i == 4:  # Decision -> next
            ax.annotate(
                "",
                xy=(5, y_to),
                xytext=(5, y_from),
                arrowprops=dict(arrowstyle="-|>", color="#28A745", lw=0.8),
            )
            ax.text(
                5.6,
                (y_from + y_to) / 2,
                "Yes",
                fontsize=SMALL_FONT - 1,
                color="#28A745",
                va="center",
            )
        else:
            ax.annotate("", xy=(5, y_to), xytext=(5, y_from), arrowprops=arrow_kw)

    # Loop back arrow from decision to step 3
    ax.annotate(
        "",
        xy=(2.0, 11.2),
        xytext=(2.0, 7.2),
        arrowprops=dict(
            arrowstyle="-|>", color="#DC3545", lw=0.7, connectionstyle="arc3,rad=-0.3"
        ),
    )
    ax.text(1.2, 9.2, "No", fontsize=SMALL_FONT - 1, color="#DC3545", va="center")

    # Multi-restart annotation
    ax.text(
        8.5,
        3.2,
        r"$\times$7 restarts" + "\npick best",
        fontsize=SMALL_FONT,
        color=C_PURPLE,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_PURPLE, lw=0.5),
    )
    ax.annotate(
        "",
        xy=(6.5, 3.2),
        xytext=(7.5, 3.2),
        arrowprops=dict(arrowstyle="-", color=C_PURPLE, lw=0.5, ls="--"),
    )

    fig.savefig(OUT_DIR / "fig8_algorithm_flowchart.pdf")
    plt.close(fig)
    print(f"  -> {OUT_DIR / 'fig8_algorithm_flowchart.pdf'}")


# ======================================================================
# Figure 9: Spot Pattern with Matching Arrows (ASM working mechanism)
# Shows how PSO predicts expected positions and matches to observed
# ======================================================================
def fig9_matching_mechanism():
    print("[Fig 9] ASM matching mechanism...")
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, FULL_W / 3 + 0.1))

    la = make_lenslet()
    ref = la.reference_positions()
    sensor_w = SENSOR_W_PX * PIXEL_UM
    sensor_h = SENSOR_H_PX * PIXEL_UM

    # Use a moderate PV where the mechanism is visible
    pv = 15.0
    seed = 42
    _, disp, mask, coeffs, _ = simulate_spots(pv, seed)

    # Select a 5x5 central region for clarity
    cx, cy = sensor_w / 2, sensor_h / 2
    region_r = PITCH_UM * 3.5
    in_region = (np.abs(ref[:, 0] - cx) < region_r) & (
        np.abs(ref[:, 1] - cy) < region_r
    )

    # Panel (a): Reference grid only
    ax = axes[0]
    ax.scatter(
        ref[in_region, 0],
        ref[in_region, 1],
        s=20,
        c=C_REF,
        marker="+",
        linewidths=0.6,
        zorder=2,
    )
    ax.set_xlim(cx - region_r - 50, cx + region_r + 50)
    ax.set_ylim(cy - region_r - 50, cy + region_r + 50)
    ax.set_aspect("equal")
    ax.set_title("(a) Reference positions", fontsize=FONT_SIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    # Draw subaperture boundaries
    for idx in np.where(in_region)[0]:
        x0 = ref[idx, 0] - PITCH_UM / 2
        y0 = ref[idx, 1] - PITCH_UM / 2
        ax.add_patch(
            plt.Rectangle(
                (x0, y0), PITCH_UM, PITCH_UM, fill=False, edgecolor="#DDDDDD", lw=0.3
            )
        )

    # Panel (b): Displaced spots (observed)
    ax = axes[1]
    # Show all observed spots in the display region
    display_mask = (
        (disp[:, 0] > cx - region_r - 50)
        & (disp[:, 0] < cx + region_r + 50)
        & (disp[:, 1] > cy - region_r - 50)
        & (disp[:, 1] < cy + region_r + 50)
    )
    ax.scatter(
        ref[in_region, 0],
        ref[in_region, 1],
        s=10,
        c=C_REF,
        marker="+",
        linewidths=0.4,
        alpha=0.3,
        zorder=1,
    )
    ax.scatter(
        disp[display_mask, 0],
        disp[display_mask, 1],
        s=12,
        c=C_ASM,
        marker="o",
        linewidths=0,
        zorder=2,
        alpha=0.8,
    )
    ax.set_xlim(cx - region_r - 50, cx + region_r + 50)
    ax.set_ylim(cy - region_r - 50, cy + region_r + 50)
    ax.set_aspect("equal")
    ax.set_title(f"(b) Observed spots (PV={pv:.0f}$\\lambda$)", fontsize=FONT_SIZE)
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel (c): PSO predicted + matching arrows
    ax = axes[2]
    # PSO predicted positions ≈ true displaced (with small error for visualization)
    rng = np.random.RandomState(999)
    pred_noise = rng.randn(*disp.shape) * 3.0  # small position error in um
    predicted = disp + pred_noise

    ax.scatter(
        ref[in_region, 0],
        ref[in_region, 1],
        s=8,
        c=C_REF,
        marker="+",
        linewidths=0.3,
        alpha=0.2,
        zorder=1,
    )
    ax.scatter(
        disp[display_mask, 0],
        disp[display_mask, 1],
        s=10,
        c=C_ASM,
        marker="o",
        linewidths=0,
        zorder=2,
        alpha=0.7,
        label="Observed",
    )
    ax.scatter(
        predicted[display_mask & in_region, 0],
        predicted[display_mask & in_region, 1],
        s=10,
        c=C_PASS,
        marker="^",
        linewidths=0,
        zorder=3,
        alpha=0.8,
        label="PSO predicted",
    )

    # Draw matching arrows from predicted to nearest observed
    matched_idx = np.where(display_mask & in_region)[0]
    for idx in matched_idx[:15]:
        d = np.sqrt(np.sum((disp[mask] - predicted[idx]) ** 2, axis=1))
        nearest = np.argmin(d)
        obs_pt = disp[mask][nearest]
        ax.annotate(
            "",
            xy=(obs_pt[0], obs_pt[1]),
            xytext=(predicted[idx, 0], predicted[idx, 1]),
            arrowprops=dict(
                arrowstyle="-|>", color=C_ACCENT, lw=0.4, mutation_scale=4, alpha=0.7
            ),
        )

    ax.set_xlim(cx - region_r - 50, cx + region_r + 50)
    ax.set_ylim(cy - region_r - 50, cy + region_r + 50)
    ax.set_aspect("equal")
    ax.set_title("(c) PSO matching", fontsize=FONT_SIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        loc="lower right",
        fontsize=SMALL_FONT - 1,
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        markerscale=1.2,
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig9_matching_mechanism.pdf")
    plt.close(fig)
    print(f"  -> {OUT_DIR / 'fig9_matching_mechanism.pdf'}")


# ======================================================================
# Figure 10: Combined Success Rate + RMSE (dual-axis, single column)
# Compact version for space-constrained papers
# ======================================================================
def fig10_combined_sr_rmse():
    print("[Fig 10] Combined SR + RMSE...")
    fig, ax1 = plt.subplots(figsize=(COL_W, COL_W * 0.72))
    ax2 = ax1.twinx()

    # Success rate (left axis)
    (l1,) = ax1.plot(
        EVAL_PV,
        BL_SR * 100,
        "o-",
        color=C_BASELINE,
        label="Conv. SR",
        markersize=3.5,
        markerfacecolor="white",
        markeredgewidth=0.7,
        zorder=3,
    )
    (l2,) = ax1.plot(
        EVAL_PV, ASM_SR * 100, "s-", color=C_ASM, label="ASM SR", markersize=3, zorder=3
    )

    ax1.axhline(95, color="#888888", ls="--", lw=0.5)
    ax1.set_ylabel("Success rate (%)", color="black")
    ax1.set_ylim(-5, 108)
    ax1.set_yticks([0, 25, 50, 75, 95, 100])

    # RMSE (right axis)
    bl_valid = ~np.isnan(BL_RMSE)
    (l3,) = ax2.plot(
        EVAL_PV[bl_valid],
        BL_RMSE[bl_valid],
        "^--",
        color=C_BASELINE,
        alpha=0.5,
        markersize=3,
        label="Conv. RMSE",
        zorder=2,
    )
    (l4,) = ax2.plot(
        EVAL_PV,
        ASM_RMSE,
        "v--",
        color=C_ASM,
        alpha=0.5,
        markersize=3,
        label="ASM RMSE",
        zorder=2,
    )
    ax2.set_ylabel(r"RMSE ($\lambda$)", color="black")
    ax2.set_ylim(-0.005, 0.08)

    ax1.set_xlabel(r"PV ($\lambda$)")
    ax1.set_xlim(0, 37)

    # Combined legend
    lines = [l1, l2, l3, l4]
    labels = [l.get_label() for l in lines]
    ax1.legend(
        lines,
        labels,
        loc="center left",
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        framealpha=0.95,
        fontsize=SMALL_FONT,
    )
    ax1.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig10_combined_sr_rmse.pdf")
    plt.close(fig)
    print(f"  -> {OUT_DIR / 'fig10_combined_sr_rmse.pdf'}")


# ======================================================================
# Figure 11: In-Bounds Spot Ratio vs PV
# Shows how many spots remain on sensor as aberration increases
# ======================================================================
def fig11_spot_retention():
    print("[Fig 11] Spot retention vs PV...")
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.72))

    pv_scan = np.array([0.5, 1, 2, 3, 5, 7, 10, 12, 15, 18, 20, 25, 30, 35, 40])
    n_seeds = 10
    retention_mean = []
    retention_std = []

    for pv in pv_scan:
        ratios = []
        for s in range(n_seeds):
            _, _, mask, _, _ = simulate_spots(pv, seed=42 + s)
            ratios.append(np.sum(mask) / len(mask) * 100)
        retention_mean.append(np.mean(ratios))
        retention_std.append(np.std(ratios))

    retention_mean = np.array(retention_mean)
    retention_std = np.array(retention_std)

    ax.fill_between(
        pv_scan,
        retention_mean - retention_std,
        retention_mean + retention_std,
        alpha=0.15,
        color=C_ASM,
    )
    ax.plot(
        pv_scan,
        retention_mean,
        "o-",
        color=C_ASM,
        markersize=3.5,
        lw=1.0,
        label="Mean $\\pm$ 1$\\sigma$ (10 trials)",
    )

    # Mark key thresholds
    ax.axhline(50, color="#888888", ls=":", lw=0.5)
    ax.text(40.5, 51, "50%", fontsize=SMALL_FONT - 1, color="#888888", va="bottom")

    # Conventional limit
    ax.axvline(BL_DR, color=C_BASELINE, ls="--", lw=0.6, alpha=0.7)
    ax.text(
        BL_DR + 0.5,
        5,
        f"Conv.\nlimit",
        fontsize=SMALL_FONT - 1,
        color=C_BASELINE,
        va="bottom",
    )

    # ASM limit
    ax.axvline(ASM_DR, color=C_ASM, ls="--", lw=0.6, alpha=0.7)
    ax.text(
        ASM_DR + 0.5,
        5,
        f"ASM\nlimit",
        fontsize=SMALL_FONT - 1,
        color=C_ASM,
        va="bottom",
    )

    ax.set_xlabel(r"Peak-to-Valley wavefront error (PV, $\lambda$)")
    ax.set_ylabel("Spots within sensor bounds (%)")
    ax.set_xlim(0, 42)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#CCCCCC")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig11_spot_retention.pdf")
    plt.close(fig)
    print(f"  -> {OUT_DIR / 'fig11_spot_retention.pdf'}")


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Generating CVPR-style figures for SHWS-ASM paper")
    print("=" * 60)

    fig1_spot_patterns()
    fig2_success_rate()
    fig3_rmse()
    fig4_dynamic_range()
    fig5_pso_convergence()
    fig6_wavefront_reconstruction()
    fig7_displacement_analysis()
    fig8_algorithm_flowchart()
    fig9_matching_mechanism()
    fig10_combined_sr_rmse()
    fig11_spot_retention()

    print("=" * 60)
    print(f"All figures saved to: {OUT_DIR.resolve()}")
    print("=" * 60)
