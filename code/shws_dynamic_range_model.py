#!/usr/bin/env python3
"""
SHWS dynamic range vs. microlens parameters (pure Python, no third-party deps).

Model assumptions:
1) Spot displacement for local slope: dx = f * theta.
2) Dynamic-range limit is set by spot staying inside a lenslet-assigned region.
3) Available displacement margin is reduced by diffraction spot size + pixel guard.
4) Software method can extend effective range by a configurable gain factor.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class ModelInput:
    wavelength_m: float
    pitch_m: float
    focal_m: float
    pixel_m: float
    fill_factor: float
    guard_px: float
    centroid_noise_px: float
    software_gain: float


@dataclass
class ModelOutput:
    pitch_um: float
    focal_mm: float
    f_number: float
    airy_radius_um: float
    max_disp_um: float
    theta_max_mrad: float
    theta_max_soft_mrad: float
    slope_resolution_urad: float
    local_opd_pv_waves: float
    dynamic_to_resolution: float
    valid: int


def airy_radius_m(wavelength_m: float, focal_m: float, aperture_m: float) -> float:
    # Airy radius to first minimum: 1.22 * lambda * f / D
    return 1.22 * wavelength_m * focal_m / aperture_m


def evaluate(inp: ModelInput) -> ModelOutput:
    aperture_m = inp.fill_factor * inp.pitch_m
    f_number = inp.focal_m / aperture_m
    r_airy = airy_radius_m(inp.wavelength_m, inp.focal_m, aperture_m)
    guard_m = inp.guard_px * inp.pixel_m

    # Effective half-cell displacement budget.
    max_disp_m = inp.pitch_m / 2.0 - r_airy - guard_m
    valid = 1 if max_disp_m > 0 else 0
    if max_disp_m <= 0:
        max_disp_m = 0.0

    theta_max = max_disp_m / inp.focal_m if inp.focal_m > 0 else 0.0
    theta_max_soft = inp.software_gain * theta_max

    centroid_sigma_m = inp.centroid_noise_px * inp.pixel_m
    slope_resolution = centroid_sigma_m / inp.focal_m if inp.focal_m > 0 else 0.0

    # Local OPD peak-to-valley estimate across one lenslet.
    local_opd_pv_m = theta_max_soft * inp.pitch_m
    local_opd_pv_waves = local_opd_pv_m / inp.wavelength_m if inp.wavelength_m > 0 else 0.0

    dyn_res = theta_max_soft / slope_resolution if slope_resolution > 0 else 0.0

    return ModelOutput(
        pitch_um=inp.pitch_m * 1e6,
        focal_mm=inp.focal_m * 1e3,
        f_number=f_number,
        airy_radius_um=r_airy * 1e6,
        max_disp_um=max_disp_m * 1e6,
        theta_max_mrad=theta_max * 1e3,
        theta_max_soft_mrad=theta_max_soft * 1e3,
        slope_resolution_urad=slope_resolution * 1e6,
        local_opd_pv_waves=local_opd_pv_waves,
        dynamic_to_resolution=dyn_res,
        valid=valid,
    )


def frange(values: str, scale: float = 1.0) -> List[float]:
    return [float(v.strip()) * scale for v in values.split(",") if v.strip()]


def write_csv(path: str, rows: Iterable[ModelOutput]) -> None:
    fields = [
        "pitch_um",
        "focal_mm",
        "f_number",
        "airy_radius_um",
        "max_disp_um",
        "theta_max_mrad",
        "theta_max_soft_mrad",
        "slope_resolution_urad",
        "local_opd_pv_waves",
        "dynamic_to_resolution",
        "valid",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.__dict__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SHWS dynamic range model (software extension + microlens parameters)"
    )
    parser.add_argument("--wavelength-nm", type=float, default=633.0)
    parser.add_argument("--pixel-um", type=float, default=10.0)
    parser.add_argument("--fill-factor", type=float, default=0.95)
    parser.add_argument("--guard-px", type=float, default=1.0)
    parser.add_argument("--centroid-noise-px", type=float, default=0.05)
    parser.add_argument("--software-gain", type=float, default=2.0)
    parser.add_argument(
        "--pitch-list-um",
        type=str,
        default="100,120,140,160,180,200,220,250,300",
        help="comma-separated list",
    )
    parser.add_argument(
        "--focal-list-mm",
        type=str,
        default="3,4,5,6,8,10,12",
        help="comma-separated list",
    )
    parser.add_argument("--out-csv", type=str, default="shws_dynamic_range_results.csv")
    args = parser.parse_args()

    wavelength_m = args.wavelength_nm * 1e-9
    pixel_m = args.pixel_um * 1e-6
    pitch_list_m = frange(args.pitch_list_um, 1e-6)
    focal_list_m = frange(args.focal_list_mm, 1e-3)

    rows: List[ModelOutput] = []
    for p in pitch_list_m:
        for f in focal_list_m:
            out = evaluate(
                ModelInput(
                    wavelength_m=wavelength_m,
                    pitch_m=p,
                    focal_m=f,
                    pixel_m=pixel_m,
                    fill_factor=args.fill_factor,
                    guard_px=args.guard_px,
                    centroid_noise_px=args.centroid_noise_px,
                    software_gain=args.software_gain,
                )
            )
            rows.append(out)

    write_csv(args.out_csv, rows)

    valid_rows = [r for r in rows if r.valid == 1]
    if not valid_rows:
        print("No valid configuration found. Try larger pitch, shorter focal length, or smaller guard.")
        return

    best_dyn = max(valid_rows, key=lambda x: x.theta_max_soft_mrad)
    best_tradeoff = max(valid_rows, key=lambda x: x.dynamic_to_resolution)

    print("=== SHWS Quantitative Summary ===")
    print(f"Total configs: {len(rows)}, Valid: {len(valid_rows)}")
    print(f"Output CSV: {args.out_csv}")
    print()
    print("1) Max effective dynamic range (with software gain)")
    print(
        "   pitch={:.0f} um, focal={:.1f} mm, f#={:.2f}, theta_soft={:.3f} mrad".format(
            best_dyn.pitch_um,
            best_dyn.focal_mm,
            best_dyn.f_number,
            best_dyn.theta_max_soft_mrad,
        )
    )
    print(
        "   local_OPD_PV={:.2f} waves, max_disp={:.2f} um".format(
            best_dyn.local_opd_pv_waves,
            best_dyn.max_disp_um,
        )
    )
    print()
    print("2) Best dynamic-range / resolution ratio")
    print(
        "   pitch={:.0f} um, focal={:.1f} mm, f#={:.2f}, DR/Res={:.1f}".format(
            best_tradeoff.pitch_um,
            best_tradeoff.focal_mm,
            best_tradeoff.f_number,
            best_tradeoff.dynamic_to_resolution,
        )
    )
    print(
        "   theta_soft={:.3f} mrad, slope_resolution={:.2f} urad".format(
            best_tradeoff.theta_max_soft_mrad,
            best_tradeoff.slope_resolution_urad,
        )
    )
    print()
    print("Analytic trend (approx, d≈p): theta_max ≈ p/(2f) - 1.22*lambda/p - guard/f")
    print("So: larger pitch and shorter focal length increase dynamic range, but sensitivity typically decreases.")


if __name__ == "__main__":
    main()
