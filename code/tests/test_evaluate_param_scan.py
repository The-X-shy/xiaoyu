"""Tests for param-scan dynamic-range summary metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.cli.evaluate_param_scan import _summarize


def test_summarize_includes_dynamic_range_columns():
    rows = []
    for pv in [0.5, 1.5, 2.5]:
        for seed in [1, 2]:
            rows.append(
                {
                    "pitch_um": 150.0,
                    "focal_mm": 6.0,
                    "pv_level": pv,
                    "seed": seed,
                    "success": pv <= 1.5,
                    "rmse": 0.05 if pv <= 1.5 else 0.30,
                }
            )
    df = pd.DataFrame(rows)

    out = _summarize(df, "baseline", rmse_max=0.15, sr_min=0.95)
    assert {"baseline_sr", "baseline_rmse", "baseline_dr", "baseline_n"}.issubset(
        set(out.columns)
    )
    assert len(out) == 1
    assert np.isclose(out.loc[0, "baseline_dr"], 1.5)
