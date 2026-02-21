# Param Scan Recommendation

## Core Summary
- Total parameter pairs: 49
- Pairs with `pass_14x=True`: 22
- Max `range_gain`: 43.00
- Median `range_gain`: 13.00

## Best Pair
- pitch_um=300, focal_mm=5.0, range_gain=43.00, asm_dr=21.5, baseline_dr=0.5, asm_rmse=0.0099

## Recommended Robust Region Rule
- `pass_14x == True`
- `asm_dr >= 20.0`
- `asm_rmse <= 0.01`

## Generated Files
- `outputs/figures/param_scan_range_gain_heatmap.png`
- `outputs/figures/param_scan_asm_dr_heatmap.png`
- `outputs/tables/param_scan_recommendation_top10.csv`
- `outputs/tables/param_scan_pass14x.csv`
- `outputs/tables/param_scan_recommendation_robust.csv`
- `outputs/tables/param_scan_recommendation_intervals.csv`
