"""Run baseline and ASM dynamic range experiments sequentially."""

import subprocess
import sys
import time

t0 = time.time()

# 1. Baseline
print("=" * 60)
print("RUNNING BASELINE EXPERIMENT")
print("=" * 60, flush=True)
r1 = subprocess.run(
    [
        sys.executable,
        "-m",
        "src.cli.run_baseline",
        "--config",
        "configs/exp_dynamic_range_quick.yaml",
    ],
    capture_output=False,
)
t1 = time.time()
print(f"\nBaseline completed in {(t1 - t0) / 60:.1f} minutes\n", flush=True)

# 2. ASM
print("=" * 60)
print("RUNNING ASM EXPERIMENT")
print("=" * 60, flush=True)
r2 = subprocess.run(
    [
        sys.executable,
        "-m",
        "src.cli.run_asm",
        "--config",
        "configs/exp_dynamic_range_quick.yaml",
    ],
    capture_output=False,
)
t2 = time.time()
print(f"\nASM completed in {(t2 - t1) / 60:.1f} minutes\n", flush=True)

# 3. Evaluate
print("=" * 60)
print("EVALUATING RESULTS")
print("=" * 60, flush=True)
r3 = subprocess.run(
    [
        sys.executable,
        "-m",
        "src.cli.evaluate",
        "--baseline",
        "outputs/tables/dynamic_range_quick_baseline_results.csv",
        "--asm",
        "outputs/tables/dynamic_range_quick_asm_results.csv",
    ],
    capture_output=False,
)
t3 = time.time()
print(f"\nTotal time: {(t3 - t0) / 60:.1f} minutes", flush=True)
