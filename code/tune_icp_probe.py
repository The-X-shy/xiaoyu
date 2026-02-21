"""Quick ICP parameter probe for no-oracle ASM on GPU.

Runs a small grid around default settings and writes a sortable CSV report.
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path

import pandas as pd
import yaml

from src.eval.protocol import evaluate_single_sample


def main() -> None:
    cfg_path = Path("configs/base_no_oracle.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    asm = cfg["asm"]
    asm["use_oracle_index_hint"] = False
    asm["use_gpu"] = True
    asm["require_gpu"] = True

    combos = []
    for max_dist in [0.5, 0.7, 1.0]:
        for min_ratio in [0.05, 0.02]:
            for fallback in [True, False]:
                combos.append(
                    {
                        "max_match_dist_factor": float(max_dist),
                        "min_match_ratio": float(min_ratio),
                        "allow_forward_fallback": bool(fallback),
                        "n_starts": 80,
                        "n_icp_iter": 40,
                        "lambda_reg": 1.0e-3,
                        "trim_ratio": 0.9,
                    }
                )

    pv_levels = [0.5, 1.5, 2.5, 3.5, 4.5]
    seeds = [20260210, 20260211]

    rows = []
    all_runs = []
    total = len(combos) * len(pv_levels) * len(seeds)
    step = 0

    for i, combo in enumerate(combos):
        cfg_i = copy.deepcopy(cfg)
        cfg_i["asm"].update(combo)
        combo_tag = f"c{i:02d}"

        combo_runs = []
        for pv, seed in itertools.product(pv_levels, seeds):
            step += 1
            r = evaluate_single_sample(
                cfg=cfg_i,
                method="asm",
                pv=float(pv),
                seed=int(seed),
                missing_ratio=0.0,
            )
            r["combo"] = combo_tag
            r.update({f"cfg_{k}": v for k, v in combo.items()})
            combo_runs.append(r)
            all_runs.append(r)
            status = "OK" if r["success"] else "FAIL"
            print(
                f"[{step}/{total}] {combo_tag} PV={pv:.1f} seed={seed} "
                f"{status} rmse={r['rmse']:.4f} n={r['n_matched']} obj={r['objective_value']:.3f}",
                flush=True,
            )

        dfc = pd.DataFrame(combo_runs)
        pv_sr = dfc.groupby("pv_level")["success"].mean().to_dict()
        # Dynamic range proxy on the probe grid: max PV with SR >= 0.95.
        dr_probe = 0.0
        for pv in sorted(pv_levels):
            if pv_sr.get(float(pv), 0.0) >= 0.95:
                dr_probe = float(pv)
            else:
                break

        rows.append(
            {
                "combo": combo_tag,
                **combo,
                "sr_overall": float(dfc["success"].mean()),
                "rmse_overall": float(dfc["rmse"].mean()),
                "rmse_success_only": float(
                    dfc.loc[dfc["success"] == True, "rmse"].mean()
                    if (dfc["success"] == True).any()
                    else float("inf")
                ),
                "dr_probe": dr_probe,
                "sr_pv_0_5": float(pv_sr.get(0.5, 0.0)),
                "sr_pv_1_5": float(pv_sr.get(1.5, 0.0)),
                "sr_pv_2_5": float(pv_sr.get(2.5, 0.0)),
                "sr_pv_3_5": float(pv_sr.get(3.5, 0.0)),
                "sr_pv_4_5": float(pv_sr.get(4.5, 0.0)),
            }
        )

    out_dir = Path("outputs/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_summary = pd.DataFrame(rows).sort_values(
        by=["dr_probe", "sr_overall", "rmse_success_only"],
        ascending=[False, False, True],
    )
    df_runs = pd.DataFrame(all_runs)

    summary_path = out_dir / "icp_probe_summary.csv"
    runs_path = out_dir / "icp_probe_runs.csv"
    df_summary.to_csv(summary_path, index=False)
    df_runs.to_csv(runs_path, index=False)

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {runs_path}")
    print("\nTop 5:")
    print(df_summary.head(5).to_string(index=False))


if __name__ == "__main__":
    main()

