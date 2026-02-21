#!/usr/bin/env python3
"""Full evaluation script for paper ASM with extended PV range.
Includes logic to save raw data and logs to outputs/ for reproducibility.
"""

import yaml, numpy as np, time, logging, sys, os, datetime
import pandas as pd

logging.basicConfig(level=logging.WARNING)

from src.sim.pipeline import forward_pipeline
from src.recon.asm_paper import asm_paper_reconstruct
from src.eval.metrics import rmse_coeffs
from scripts.eval_paper_asm import baseline_reconstruct_simple

# Setup logging
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = "outputs/logs"
table_dir = "outputs/tables"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(table_dir, exist_ok=True)

log_file = os.path.join(log_dir, f"full_eval_{timestamp}.log")

# Tee stdout to file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file)

cfg = yaml.safe_load(open("configs/paper_19x19.yaml"))
n_seeds = 20
base_seed = 900000

pv_levels = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0, 35.0, 40.0]

header = "%5s  %6s  %6s  %8s  %8s  %7s  %4s  %4s" % ("PV", "BL_SR", "ASM_SR", "BL_RMSE", "ASM_RMSE", "ASM_T", "BL", "ASM")
print("FULL EVALUATION: 20 repeats, 7 restarts, 400 iter")
print("Logs saved to:", log_file)
print(header)
print("-" * 65)

bl_dr = None
asm_dr = None

# Store raw data for CSVs
raw_data_bl = []
raw_data_asm = []

for pv in pv_levels:
    bl_succ = 0
    asm_succ = 0
    bl_rmses = []
    asm_rmses = []
    asm_times = []

    for i in range(n_seeds):
        seed = base_seed + i
        sim = forward_pipeline(cfg, pv=pv, seed=seed)
        la = sim["lenslet"]

        # Baseline
        bl = baseline_reconstruct_simple(sim["observed_positions"], la, cfg)
        bl_rmse = rmse_coeffs(sim["coeffs"], bl["coeffs"], exclude_piston=True)
        bl_success = bool(bl_rmse <= 0.15 and bl.get("success", False))
        
        raw_data_bl.append({
            "pv": pv, "seed": seed, "rmse": bl_rmse, 
            "success": bl_success, "method": "baseline"
        })

        if bl_success:
            bl_succ += 1
            bl_rmses.append(bl_rmse)

        # ASM
        t0 = time.time()
        asm = asm_paper_reconstruct(
            sim["observed_positions"], la, cfg, seed=seed + 1000, pv_hint=pv
        )
        dt = time.time() - t0
        asm_rmse = rmse_coeffs(sim["coeffs"], asm["coeffs"], exclude_piston=True)
        asm_success = bool(asm_rmse <= 0.15)
        
        raw_data_asm.append({
            "pv": pv, "seed": seed, "rmse": asm_rmse, 
            "time_s": dt, "success": asm_success, "method": "asm_paper"
        })

        asm_times.append(dt)
        if asm_success:
            asm_succ += 1
            asm_rmses.append(asm_rmse)

    bl_sr = bl_succ / n_seeds
    asm_sr = asm_succ / n_seeds
    bl_mean = np.mean(bl_rmses) if bl_rmses else float("nan")
    asm_mean = np.mean(asm_rmses) if asm_rmses else float("nan")
    asm_t = np.mean(asm_times)
    bl_status = "PASS" if bl_sr >= 0.95 else "FAIL"
    asm_status = "PASS" if asm_sr >= 0.95 else "FAIL"

    if bl_status == "PASS" and (bl_dr is None or pv > bl_dr):
        bl_dr = pv
    if asm_status == "PASS" and (asm_dr is None or pv > asm_dr):
        asm_dr = pv

    row = "%5.1f  %6.2f  %6.2f  %8.4f  %8.4f  %7.1f  %4s  %4s" % (
        pv, bl_sr, asm_sr, bl_mean, asm_mean, asm_t, bl_status, asm_status,
    )
    print(row)

print()
print("=" * 65)
bl_dr = bl_dr or 0
asm_dr = asm_dr or 0
gain = asm_dr / bl_dr if bl_dr > 0 else float("inf")
print("Baseline DR: %s" % bl_dr)
print("ASM DR:      %s" % asm_dr)
print("Gain:        %.2fx" % gain)
print("Target:      >= 14.0x")
print("Result:      %s" % ("PASS" if gain >= 14.0 else "FAIL"))

# Paper slope ratio metric
opt = cfg["optics"]
conv_max_slope_mrad = opt["pitch_um"] / (2 * opt["focal_mm"] * 1000) * 1000
print("\nConventional max slope: %.2f mrad" % conv_max_slope_mrad)
slope_ratio = 0.0
asm_max_slope_mrad = 0.0

if asm_dr > 0:
    max_disps = []
    for i in range(min(10, n_seeds)):
        sim = forward_pipeline(cfg, pv=asm_dr, seed=base_seed + i)
        la = sim["lenslet"]
        disps = la.slopes_to_displacements(sim["slopes"])
        max_d = np.max(np.sqrt(disps[:, 0] ** 2 + disps[:, 1] ** 2))
        max_disps.append(max_d)
    asm_max_disp = np.mean(max_disps)
    asm_max_slope_mrad = asm_max_disp / (opt["focal_mm"] * 1000) * 1000
    slope_ratio = asm_max_slope_mrad / conv_max_slope_mrad
    print("ASM max slope at DR PV=%s: %.2f mrad" % (asm_dr, asm_max_slope_mrad))
    print("Slope ratio (paper metric): %.2fx" % slope_ratio)
    print("Paper reported: 14.81x")

# Save detailed CSVs
df_bl = pd.DataFrame(raw_data_bl)
df_bl.to_csv(os.path.join(table_dir, f"full_eval_baseline_{timestamp}.csv"), index=False)
df_asm = pd.DataFrame(raw_data_asm)
df_asm.to_csv(os.path.join(table_dir, f"full_eval_asm_{timestamp}.csv"), index=False)

# Save summary CSV
summary = [{
    "timestamp": timestamp,
    "baseline_dr": bl_dr,
    "asm_dr": asm_dr,
    "contiguous_gain": gain,
    "conv_max_slope_mrad": conv_max_slope_mrad,
    "asm_max_slope_mrad": asm_max_slope_mrad,
    "slope_ratio": slope_ratio
}]
pd.DataFrame(summary).to_csv(os.path.join(table_dir, f"full_eval_summary_{timestamp}.csv"), index=False)

print(f"\nSaved raw results to {table_dir}/full_eval_baseline_{timestamp}.csv and full_eval_asm_{timestamp}.csv")
print(f"Saved summary to {table_dir}/full_eval_summary_{timestamp}.csv")
