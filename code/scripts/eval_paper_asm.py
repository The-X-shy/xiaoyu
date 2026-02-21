#!/usr/bin/env python3
"""Evaluate paper-faithful ASM (Hausdorff + PSO) on 19x19 MLA.

Uses the paper's 19x19 MLA config and the paper's algorithm.
Evaluates both:
1. Our standard metric (contiguous PV dynamic range at 95% SR)
2. Paper's metric (max slope ratio: ASM max slope / conventional max slope)

Usage:
    PYTHONPATH=. python3 scripts/eval_paper_asm.py [--n_repeats 20] [--quick]
"""

import sys, os, csv, datetime, argparse, logging, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np

from src.sim.pipeline import forward_pipeline
from src.sim.lenslet import LensletArray
from src.recon.asm_paper import asm_paper_reconstruct
from src.eval.metrics import rmse_coeffs, compute_dynamic_range, success_rate


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def baseline_reconstruct_simple(
    observed_pos: np.ndarray,
    la: LensletArray,
    cfg: dict,
) -> dict:
    """Simple baseline: each observed spot matched to nearest reference.

    For the 19x19 config, the conventional method assigns each observed
    spot to its nearest reference position (within one pitch).
    Then does least-squares reconstruction.
    """
    from scipy.spatial import cKDTree
    from src.recon.least_squares import build_zernike_slope_matrix
    from src.recon.zernike import num_zernike_terms

    ref_pos = la.reference_positions()
    n_sub = ref_pos.shape[0]
    zer_cfg = cfg.get("zernike", {})
    max_order = zer_cfg.get("order", 4)
    n_terms = num_zernike_terms(max_order)
    grid_size = cfg.get("asm_paper", {}).get("grid_size", 128)

    if observed_pos.shape[0] < n_terms:
        return {
            "coeffs": np.zeros(n_terms),
            "success": False,
            "solver": "baseline_nn",
            "n_matched": 0,
            "objective_value": np.inf,
            "residual_raw": np.inf,
            "residual_trimmed": np.inf,
        }

    # Match each observed spot to nearest reference
    tree_ref = cKDTree(ref_pos)
    dists, nn_idx = tree_ref.query(observed_pos)

    # Filter: only keep matches within one pitch
    pitch_um = la.pitch_um
    good = dists < pitch_um
    matched_obs = observed_pos[good]
    matched_ref_idx = nn_idx[good]

    # Remove duplicates: if multiple obs match same ref, keep closest
    seen = {}
    for i, ref_i in enumerate(matched_ref_idx):
        if ref_i not in seen or dists[np.where(good)[0][i]] < seen[ref_i][1]:
            seen[ref_i] = (i, dists[np.where(good)[0][i]])

    unique_ref_idx = []
    unique_obs_idx = []
    for ref_i, (obs_local_i, _) in seen.items():
        unique_ref_idx.append(ref_i)
        unique_obs_idx.append(np.where(good)[0][obs_local_i])

    unique_ref_idx = np.array(unique_ref_idx)
    unique_obs_idx = np.array(unique_obs_idx)
    n_matched = len(unique_ref_idx)

    if n_matched < n_terms:
        return {
            "coeffs": np.zeros(n_terms),
            "success": False,
            "solver": "baseline_nn",
            "n_matched": n_matched,
            "objective_value": np.inf,
            "residual_raw": np.inf,
            "residual_trimmed": np.inf,
        }

    # Compute slopes from matched pairs
    matched_obs_pos = observed_pos[unique_obs_idx]
    target_disp = matched_obs_pos - ref_pos[unique_ref_idx]
    # Convert physical displacement to normalized slopes:
    # disp = focal_um * slope_norm * correction  =>  slope_norm = disp / (focal_um * correction)
    target_slopes = target_disp / (la.focal_um * la._slope_correction)

    # Build slope matrix and solve LS
    G = build_zernike_slope_matrix(la, max_order, grid_size)
    G_sub_x = G[unique_ref_idx, :]
    G_sub_y = G[n_sub + unique_ref_idx, :]
    A = np.vstack([G_sub_x, G_sub_y])
    b = np.concatenate([target_slopes[:, 0], target_slopes[:, 1]])

    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Compute residual
    disp_scale = la.focal_um * la._slope_correction
    residuals = np.sqrt(
        (
            matched_obs_pos[:, 0]
            - ref_pos[unique_ref_idx, 0]
            - disp_scale * (G[unique_ref_idx, :] @ coeffs)
        )
        ** 2
        + (
            matched_obs_pos[:, 1]
            - ref_pos[unique_ref_idx, 1]
            - disp_scale * (G[n_sub + unique_ref_idx, :] @ coeffs)
        )
        ** 2
    )
    residual_raw = float(np.mean(residuals))

    return {
        "coeffs": coeffs,
        "success": True,
        "solver": "baseline_nn",
        "n_matched": n_matched,
        "objective_value": residual_raw,
        "residual_raw": residual_raw,
        "residual_trimmed": residual_raw,
    }


def evaluate_single(cfg, method, pv, seed, missing_ratio=0.0):
    """Evaluate a single sample with the specified method."""
    sim = forward_pipeline(cfg, pv=pv, seed=seed, missing_ratio=missing_ratio)

    t0 = time.perf_counter()

    if method == "baseline":
        recon = baseline_reconstruct_simple(
            sim["observed_positions"],
            sim["lenslet"],
            cfg,
        )
    elif method == "asm_paper":
        recon = asm_paper_reconstruct(
            sim["observed_positions"],
            sim["lenslet"],
            cfg,
            seed=seed + 1000,
            pv_hint=pv,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    runtime_ms = (time.perf_counter() - t0) * 1000

    coeffs_recon = recon.get("coeffs")
    if coeffs_recon is None:
        coeffs_recon = np.zeros_like(sim["coeffs"])

    # Pad/truncate if needed
    n_true = len(sim["coeffs"])
    n_recon = len(coeffs_recon)
    if n_recon < n_true:
        coeffs_recon = np.concatenate([coeffs_recon, np.zeros(n_true - n_recon)])
    elif n_recon > n_true:
        coeffs_recon = coeffs_recon[:n_true]

    rmse = rmse_coeffs(sim["coeffs"], coeffs_recon, exclude_piston=True)
    success = recon.get("success", False)

    eval_cfg = cfg.get("evaluation", {})
    rmse_max = eval_cfg.get("rmse_max_lambda", 0.15)
    if rmse > rmse_max:
        success = False

    return {
        "method": method,
        "pv_level": pv,
        "seed": seed,
        "success": success,
        "rmse": float(rmse),
        "missing_ratio": missing_ratio,
        "runtime_ms": float(runtime_ms),
        "n_observed": len(sim["observed_positions"]),
        "n_ref": len(sim["ref_positions"]),
        "solver": recon.get("solver", "unknown"),
        "pso_converged": recon.get("pso_converged", None),
        "pso_n_iter": recon.get("pso_n_iter", None),
        "pso_dH": recon.get("pso_dH", None),
    }


def compute_slope_ratio(cfg):
    """Compute the paper's slope-ratio dynamic range metric.

    Conventional limit: max spot displacement = half pitch (spot stays within
    its own subaperture). Then max slope = (pitch/2) / focal_length.

    ASM limit: max slope is determined by the largest PV where ASM still works.
    We approximate this by finding the max PV where ASM achieves 95% SR,
    then computing the max slope at that PV.

    Paper formula:
        conventional_max_slope = pitch / (2 * focal_length)
        ASM range factor = ASM_DR / conventional_DR
    """
    opt = cfg["optics"]
    pitch_um = opt["pitch_um"]
    focal_um = opt["focal_mm"] * 1000.0

    # Conventional limit: displacement < pitch/2
    # max_slope_conventional = pitch / (2 * focal_length)
    conventional_max_slope = pitch_um / (2.0 * focal_um)

    # In mrad: slope * 1000
    conventional_max_slope_mrad = conventional_max_slope * 1000.0

    return {
        "conventional_max_slope": conventional_max_slope,
        "conventional_max_slope_mrad": conventional_max_slope_mrad,
        "pitch_um": pitch_um,
        "focal_um": focal_um,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate paper-faithful ASM")
    parser.add_argument(
        "--n_repeats", type=int, default=20, help="Samples per PV level"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode: 5 samples, fewer PV levels"
    )
    parser.add_argument(
        "--config", default="configs/paper_19x19.yaml", help="Config file"
    )
    parser.add_argument(
        "--max_pv", type=float, default=20.0, help="Max PV level to test"
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    n_repeats = args.n_repeats
    base_seed = 900000

    if args.quick:
        n_repeats = 5
        pv_levels = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    else:
        # Fine-grained PV scan
        pv_levels = [
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            10.0,
            12.0,
            15.0,
            20.0,
        ]
        pv_levels = [p for p in pv_levels if p <= args.max_pv]

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    log_path = f"outputs/logs/paper_asm_eval_{ts}.log"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    # Print setup info
    opt = cfg["optics"]
    sen = cfg["sensor"]
    asm_cfg = cfg.get("asm_paper", {})

    la = LensletArray(
        pitch_um=opt["pitch_um"],
        focal_mm=opt["focal_mm"],
        fill_factor=opt.get("fill_factor", 1.0),
        sensor_width_px=sen["width_px"],
        sensor_height_px=sen["height_px"],
        pixel_um=sen["pixel_um"],
        mla_grid_size=opt.get("grid_size", 0),
        wavelength_um=opt.get("wavelength_nm", 0.0) / 1000.0,
    )

    log("=" * 70)
    log("PAPER-FAITHFUL ASM EVALUATION")
    log("=" * 70)
    log(f"Config: {args.config}")
    log(f"MLA grid: {opt.get('grid_size', 'auto')}x{opt.get('grid_size', 'auto')}")
    log(f"Subapertures: {la.n_subapertures}")
    log(f"Pitch: {opt['pitch_um']} um")
    log(f"Focal: {opt['focal_mm']} mm")
    log(f"Pixel: {sen['pixel_um']} um")
    log(
        f"Sensor: {sen['width_px']}x{sen['height_px']} px ({la.sensor_width_um}x{la.sensor_height_um} um)"
    )
    log(
        f"Zernike order: {cfg['zernike']['order']} ({cfg['zernike'].get('n_terms', '?')} terms)"
    )
    log(f"PSO particles: {asm_cfg.get('pso_particles', 100)}")
    log(f"PSO max iter: {asm_cfg.get('pso_max_iter', 200)}")
    log(f"Hausdorff threshold: {asm_cfg.get('hausdorff_threshold_px', 6.0)} px")
    log(f"N repeats: {n_repeats}")
    log(f"PV levels: {pv_levels}")
    log(f"Timestamp: {ts}")
    log("")

    # Compute paper's slope-ratio metric
    slope_info = compute_slope_ratio(cfg)
    log(
        f"Conventional max slope: {slope_info['conventional_max_slope']:.6f} (= {slope_info['conventional_max_slope_mrad']:.2f} mrad)"
    )
    log(
        f"  = pitch/(2*f) = {slope_info['pitch_um']:.1f}/(2*{slope_info['focal_um']:.1f}) um"
    )
    log(
        f"  Max displacement = {slope_info['pitch_um'] / 2:.1f} um = {slope_info['pitch_um'] / 2 / sen['pixel_um']:.1f} pixels"
    )
    log("")

    # ===== Run baseline evaluation =====
    log("=" * 70)
    log("BASELINE METHOD (nearest-neighbor matching)")
    log("=" * 70)

    baseline_records = []
    for pv in pv_levels:
        t0 = time.time()
        pv_results = []
        for i in range(n_repeats):
            seed = base_seed + i
            result = evaluate_single(cfg, "baseline", pv, seed)
            pv_results.append(result)
        dt = time.time() - t0

        sr = success_rate(pv_results)
        rmses = [r["rmse"] for r in pv_results if r.get("success", False)]
        mean_rmse = np.mean(rmses) if rmses else float("inf")
        n_obs_mean = np.mean([r["n_observed"] for r in pv_results])
        status = "PASS" if sr >= 0.95 and mean_rmse <= 0.15 else "FAIL"

        baseline_records.extend(pv_results)
        log(
            f"  PV={pv:5.1f} | SR={sr * 100:5.1f}% | RMSE={mean_rmse:.4f} | "
            f"n_obs={n_obs_mean:.0f}/{la.n_subapertures} | {status} | {dt:.1f}s"
        )

    baseline_dr = compute_dynamic_range(baseline_records)
    log(f"\n  Baseline dynamic range: {baseline_dr}")

    # ===== Run ASM evaluation =====
    log("")
    log("=" * 70)
    log("PAPER ASM METHOD (Hausdorff + PSO)")
    log("=" * 70)

    asm_records = []
    for pv in pv_levels:
        t0 = time.time()
        pv_results = []
        for i in range(n_repeats):
            seed = base_seed + i
            result = evaluate_single(cfg, "asm_paper", pv, seed)
            pv_results.append(result)

            if (i + 1) % max(1, n_repeats // 4) == 0:
                partial_sr = success_rate(pv_results)
                last = pv_results[-1]
                log(
                    f"    PV={pv:.1f} [{i + 1}/{n_repeats}] sr={partial_sr:.2f} "
                    f"pso_conv={last.get('pso_converged')} dH={last.get('pso_dH', '?')}"
                )

        dt = time.time() - t0
        sr = success_rate(pv_results)
        rmses = [r["rmse"] for r in pv_results if r.get("success", False)]
        mean_rmse = np.mean(rmses) if rmses else float("inf")
        n_obs_mean = np.mean([r["n_observed"] for r in pv_results])
        conv_rate = np.mean([1 if r.get("pso_converged") else 0 for r in pv_results])
        avg_iter = np.mean([r.get("pso_n_iter", 0) or 0 for r in pv_results])
        status = "PASS" if sr >= 0.95 and mean_rmse <= 0.15 else "FAIL"

        asm_records.extend(pv_results)
        log(
            f"  PV={pv:5.1f} | SR={sr * 100:5.1f}% | RMSE={mean_rmse:.4f} | "
            f"conv={conv_rate:.2f} | iter={avg_iter:.0f} | "
            f"n_obs={n_obs_mean:.0f}/{la.n_subapertures} | {status} | {dt:.1f}s"
        )

    asm_dr = compute_dynamic_range(asm_records)
    log(f"\n  ASM dynamic range: {asm_dr}")

    # ===== Summary =====
    log("")
    log("=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)

    if baseline_dr > 0:
        gain = asm_dr / baseline_dr
    else:
        gain = float("inf") if asm_dr > 0 else 0.0

    log(f"  Baseline DR: {baseline_dr}")
    log(f"  ASM DR:      {asm_dr}")
    log(f"  Gain:        {gain:.2f}x")
    log(f"  Target:      >= 14.0x")
    log(f"  Result:      {'PASS' if gain >= 14.0 else 'FAIL'}")

    # Paper's slope-ratio computation
    # The max slope ASM can handle corresponds to the max PV where SR >= 95%
    # max_displacement_asm = max_slope_asm * focal_um
    # But we need to estimate max_slope from the evaluation
    if asm_dr > 0:
        # At the ASM DR PV level, compute typical max slope
        # Run a few samples at the DR PV to measure max displacement
        log("\n  --- Paper slope-ratio metric ---")
        log(
            f"  Conventional max displacement: {slope_info['pitch_um'] / 2:.1f} um ({slope_info['pitch_um'] / 2 / sen['pixel_um']:.1f} px)"
        )

        # Estimate max displacement at ASM DR level
        max_displacements = []
        for i in range(min(10, n_repeats)):
            seed = base_seed + i
            sim = forward_pipeline(cfg, pv=asm_dr, seed=seed)
            if len(sim["observed_positions"]) > 0:
                ref = sim["ref_positions"]
                disp_scale = la.focal_um * la._slope_correction
                disps = sim["slopes"] * disp_scale
                max_d = np.max(np.sqrt(disps[:, 0] ** 2 + disps[:, 1] ** 2))
                max_displacements.append(max_d)

        if max_displacements:
            asm_max_disp = np.mean(max_displacements)
            conv_max_disp = slope_info["pitch_um"] / 2
            slope_gain = asm_max_disp / conv_max_disp
            log(
                f"  ASM max displacement at DR PV={asm_dr}: {asm_max_disp:.1f} um ({asm_max_disp / sen['pixel_um']:.1f} px)"
            )
            log(f"  Slope ratio (paper metric): {slope_gain:.2f}x")

    # Per-PV table
    log("\n  Per-PV breakdown:")
    log(
        f"  {'PV':>5s}  {'BL_SR':>6s}  {'ASM_SR':>6s}  {'BL_RMSE':>8s}  {'ASM_RMSE':>8s}  {'BL':>4s}  {'ASM':>4s}"
    )
    log("  " + "-" * 55)

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

    # Save CSVs
    csv_fields = [
        "method",
        "pv_level",
        "seed",
        "success",
        "rmse",
        "runtime_ms",
        "solver",
        "n_observed",
        "pso_converged",
        "pso_n_iter",
        "pso_dH",
    ]

    def write_csv(path, records):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            w.writeheader()
            for r in records:
                w.writerow(r)

    bl_csv = f"outputs/tables/paper_asm_baseline_{ts}.csv"
    asm_csv = f"outputs/tables/paper_asm_asm_{ts}.csv"
    write_csv(bl_csv, baseline_records)
    write_csv(asm_csv, asm_records)

    log(f"\n  Baseline CSV: {bl_csv}")
    log(f"  ASM CSV: {asm_csv}")
    log(f"  Log: {log_path}")
    log(f"\n  baseline_dr={baseline_dr}, asm_dr={asm_dr}, gain={gain:.2f}x")
    log("\nDone!")


if __name__ == "__main__":
    main()
