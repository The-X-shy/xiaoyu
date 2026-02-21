#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "STAGE_START icp_tune_probe_baseline"
/root/miniconda3/bin/conda run --no-capture-output -n base \
  python -m src.cli.run_baseline --config configs/exp_dynamic_range_probe_no_oracle_baseline.yaml
echo "STAGE_DONE icp_tune_probe_baseline"

run_one() {
  local tag="$1"
  local cfg="$2"
  local asm_out="outputs/tables/dynamic_range_probe_no_oracle_${tag}_asm_results.csv"
  local base_out="outputs/tables/dynamic_range_probe_no_oracle_baseline_baseline_results.csv"
  local summary_out="outputs/tables/summary_metrics_probe_${tag}.csv"

  echo "STAGE_START icp_tune_probe_${tag}"
  /root/miniconda3/bin/conda run --no-capture-output -n base \
    python -m src.cli.run_asm --config "$cfg"
  /root/miniconda3/bin/conda run --no-capture-output -n base \
    python -m src.cli.evaluate --baseline "$base_out" --asm "$asm_out"
  cp -f outputs/tables/summary_metrics.csv "$summary_out"
  echo "STAGE_DONE icp_tune_probe_${tag}"
}

run_one "icp_a" "configs/exp_dynamic_range_probe_no_oracle_icp_a.yaml"
run_one "icp_b" "configs/exp_dynamic_range_probe_no_oracle_icp_b.yaml"
run_one "icp_c" "configs/exp_dynamic_range_probe_no_oracle_icp_c.yaml"

echo "ALL_DONE icp_tune_probe"

