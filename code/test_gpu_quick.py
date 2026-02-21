#!/usr/bin/env python
"""Quick manual check for ASM runtime on current machine."""

from __future__ import annotations

__test__ = False

import time

from src.config import load_experiment_config
from src.eval.protocol import evaluate_single_sample, _USE_GPU_ASM


def main() -> None:
    print(f"GPU ASM enabled: {_USE_GPU_ASM}")

    cfg = load_experiment_config("configs/exp_dynamic_range_quick.yaml")

    print("\nRunning single ASM sample at PV=5.0 ...")
    t0 = time.time()
    r = evaluate_single_sample(cfg, "asm", 5.0, 20260210)
    dt = time.time() - t0
    print(f"Result: success={r['success']}, rmse={r['rmse']:.4f}, time={dt:.2f}s")

    print("\nRunning single ASM sample at PV=1.0 ...")
    t0 = time.time()
    r = evaluate_single_sample(cfg, "asm", 1.0, 20260210)
    dt = time.time() - t0
    print(f"Result: success={r['success']}, rmse={r['rmse']:.4f}, time={dt:.2f}s")


if __name__ == "__main__":
    main()
