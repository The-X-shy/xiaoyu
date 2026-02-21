#!/usr/bin/env python
"""Manual diagnostic for legacy PSO objective landscape."""

from __future__ import annotations

__test__ = False

import numpy as np

from src.config import load_experiment_config
from src.sim.pipeline import forward_pipeline
from src.recon.asm_objective import asm_objective


def main() -> None:
    cfg = load_experiment_config("configs/exp_dynamic_range_quick.yaml")
    asm_cfg = cfg.get("asm", {})
    print(
        "lambda_dup={}, lambda_out={}, lambda_reg={}".format(
            asm_cfg.get("lambda_dup"),
            asm_cfg.get("lambda_out"),
            asm_cfg.get("lambda_reg"),
        )
    )

    for pv in [0.5, 1.0, 1.5, 2.5, 5.0, 10.0, 15.0, 19.5]:
        sim = forward_pipeline(cfg, pv=pv, seed=20260210)
        obs = sim["observed_positions"]
        la = sim["lenslet"]
        true_c = sim["coeffs"]
        zero_c = np.zeros_like(true_c)

        obj_true = asm_objective(true_c, obs, la, cfg)
        obj_zero = asm_objective(zero_c, obs, la, cfg)
        better = "TRUE<ZERO" if obj_true < obj_zero else "ZERO<TRUE !!!"
        print(
            f"PV={pv:5.1f}  obj(true)={obj_true:10.1f}  "
            f"obj(zero)={obj_zero:10.1f}  {better}"
        )


if __name__ == "__main__":
    main()
