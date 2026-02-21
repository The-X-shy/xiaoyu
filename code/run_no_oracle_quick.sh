#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "STAGE_START no_oracle_quick_dynamic"
/root/miniconda3/bin/conda run --no-capture-output -n base \
  python -m src.cli.run_baseline --config configs/exp_dynamic_range_no_oracle_quick.yaml
/root/miniconda3/bin/conda run --no-capture-output -n base \
  python -m src.cli.run_asm --config configs/exp_dynamic_range_no_oracle_quick.yaml
/root/miniconda3/bin/conda run --no-capture-output -n base \
  python -m src.cli.evaluate \
    --baseline outputs/tables/dynamic_range_no_oracle_quick_baseline_results.csv \
    --asm outputs/tables/dynamic_range_no_oracle_quick_asm_results.csv
cp -f outputs/tables/summary_metrics.csv outputs/tables/summary_metrics_no_oracle_quick.csv
echo "STAGE_DONE no_oracle_quick_dynamic"

echo "STAGE_START no_oracle_quick_missing"
/root/miniconda3/bin/conda run --no-capture-output -n base \
  python -m src.cli.run_baseline --config configs/exp_missing_spot_no_oracle_quick.yaml
/root/miniconda3/bin/conda run --no-capture-output -n base \
  python -m src.cli.run_asm --config configs/exp_missing_spot_no_oracle_quick.yaml
/root/miniconda3/bin/conda run --no-capture-output -n base python - <<'PY'
import pandas as pd
base = "outputs/tables"
b = pd.read_csv(f"{base}/missing_spot_no_oracle_quick_baseline_results.csv")
a = pd.read_csv(f"{base}/missing_spot_no_oracle_quick_asm_results.csv")
rows = []
for mr in sorted(set(b["missing_ratio"]) | set(a["missing_ratio"])):
    bb = b[b["missing_ratio"] == mr]
    aa = a[a["missing_ratio"] == mr]
    rows.append(
        dict(
            missing_ratio=float(mr),
            baseline_sr=float(bb["success"].mean()),
            asm_sr=float(aa["success"].mean()),
            delta_sr=float(aa["success"].mean() - bb["success"].mean()),
            baseline_rmse=float(bb["rmse"].mean()),
            asm_rmse=float(aa["rmse"].mean()),
        )
    )
pd.DataFrame(rows).to_csv(f"{base}/missing_spot_no_oracle_quick_summary_by_ratio.csv", index=False)
print("saved", f"{base}/missing_spot_no_oracle_quick_summary_by_ratio.csv")
PY
echo "STAGE_DONE no_oracle_quick_missing"

echo "STAGE_START no_oracle_quick_param"
/root/miniconda3/bin/conda run --no-capture-output -n base \
  python -m src.cli.run_baseline --config configs/exp_param_scan_no_oracle_quick.yaml
/root/miniconda3/bin/conda run --no-capture-output -n base \
  python -m src.cli.run_asm --config configs/exp_param_scan_no_oracle_quick.yaml
/root/miniconda3/bin/conda run --no-capture-output -n base \
  python -m src.cli.evaluate_param_scan \
    --baseline outputs/tables/param_scan_no_oracle_quick_baseline_results.csv \
    --asm outputs/tables/param_scan_no_oracle_quick_asm_results.csv \
    --out outputs/tables/param_scan_no_oracle_quick_summary.csv
echo "STAGE_DONE no_oracle_quick_param"

echo "ALL_DONE no_oracle_quick"
