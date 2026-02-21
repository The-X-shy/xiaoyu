#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/shws_code

/root/miniconda3/bin/conda run --no-capture-output -n base python -m src.cli.run_baseline --config configs/exp_param_scan.yaml
/root/miniconda3/bin/conda run --no-capture-output -n base python -m src.cli.run_asm --config configs/exp_param_scan.yaml
/root/miniconda3/bin/conda run --no-capture-output -n base python -m src.cli.evaluate_param_scan \
  --baseline outputs/tables/param_scan_baseline_results.csv \
  --asm outputs/tables/param_scan_asm_results.csv \
  --out outputs/tables/param_scan_summary.csv
