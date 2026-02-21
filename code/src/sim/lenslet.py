"""Lenslet (microlens) array model for SHWS.

Handles:
- Reference position grid generation (square grid, circular aperture)
- Sub-aperture wavefront slope computation
- Slope to displacement mapping
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


class LensletArray:
    """Square-grid microlens array on SHWS sensor. Units: micrometers."""

    def __init__(
        self,
        pitch_um: float,
        focal_mm: float,
        fill_factor: float,
        sensor_width_px: int,
        sensor_height_px: int,
        pixel_um: float,
        mla_grid_size: int = 0,
        wavelength_um: float = 0.0,
    ):
        self.pitch_um = pitch_um
        self.focal_mm = focal_mm
        self.focal_um = focal_mm * 1000.0
        self.fill_factor = fill_factor
        self.sensor_width_px = sensor_width_px
        self.sensor_height_px = sensor_height_px
        self.pixel_um = pixel_um
        self.sensor_width_um = sensor_width_px * pixel_um
        self.sensor_height_um = sensor_height_px * pixel_um
        self.mla_grid_size = mla_grid_size  # If >0, force exact NxN square grid
        self.wavelength_um = (
            wavelength_um  # Wavelength in um (needed for slope correction)
        )
        self._ref_positions = None
        self._sub_centers_norm = None

        # Precompute slope-to-displacement correction factor.
        # Our wavefront W is in "waves" on normalized coords [-1,1].
        # Gradient dW/dx_norm has units (waves / normalized_unit).
        # Physical slope (radians) = dW/dx_norm * wavelength / R_pupil
        # Displacement = focal_um * physical_slope
        #              = focal_um * dW/dx_norm * wavelength_um / R_pupil_um
        #
        # For mla_grid_size > 0: R_pupil = ((N-1)/2) * pitch
        # For mla_grid_size == 0 (old mode): keep old formula (correction=1)
        #   to preserve backward compatibility.
        if self.mla_grid_size > 0 and self.wavelength_um > 0:
            R_pupil_um = ((self.mla_grid_size - 1) / 2) * self.pitch_um
            self._slope_correction = self.wavelength_um / R_pupil_um
        else:
            self._slope_correction = 1.0

    @property
    def n_subapertures(self) -> int:
        return len(self.reference_positions())

    def reference_positions(self) -> np.ndarray:
        """Generate reference spot positions on sensor. Returns (N,2) in um.

        If mla_grid_size > 0, generates an exact NxN square grid (no circular
        aperture cropping). Otherwise uses the original circular-aperture logic.
        """
        if self._ref_positions is not None:
            return self._ref_positions

        cx = self.sensor_width_um / 2
        cy = self.sensor_height_um / 2

        if self.mla_grid_size > 0:
            # Fixed NxN square grid centered on sensor
            n = self.mla_grid_size
            positions = []
            x_start = cx - ((n - 1) / 2) * self.pitch_um
            y_start = cy - ((n - 1) / 2) * self.pitch_um
            for iy in range(n):
                for ix in range(n):
                    x = x_start + ix * self.pitch_um
                    y = y_start + iy * self.pitch_um
                    positions.append([x, y])
            self._ref_positions = np.array(positions, dtype=float)
        else:
            # Original: fill sensor area with circular aperture
            nx = int(self.sensor_width_um / self.pitch_um)
            ny = int(self.sensor_height_um / self.pitch_um)

            x_start = cx - (nx // 2) * self.pitch_um + self.pitch_um / 2
            y_start = cy - (ny // 2) * self.pitch_um + self.pitch_um / 2

            positions = []
            aperture_radius_um = min(cx, cy) * 0.95  # Slightly inside edge

            for iy in range(ny):
                for ix in range(nx):
                    x = x_start + ix * self.pitch_um
                    y = y_start + iy * self.pitch_um
                    if np.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= aperture_radius_um:
                        positions.append([x, y])

            self._ref_positions = (
                np.array(positions, dtype=float) if positions else np.empty((0, 2))
            )
        return self._ref_positions

    def subaperture_centers_normalized(self) -> np.ndarray:
        """Sub-aperture centers in normalized [-1, 1] coordinates.

        For fixed NxN grid: normalized relative to the MLA extent (half-span).
        For circular grid: normalized relative to sensor half-width.
        """
        if self._sub_centers_norm is not None:
            return self._sub_centers_norm

        ref = self.reference_positions()
        cx = self.sensor_width_um / 2
        cy = self.sensor_height_um / 2

        if self.mla_grid_size > 0:
            # Normalize to the MLA half-span so edge subapertures are at ±1
            half_span = ((self.mla_grid_size - 1) / 2) * self.pitch_um
            if half_span < 1e-12:
                half_span = 1.0  # single subaperture edge case
            norm = np.zeros_like(ref)
            norm[:, 0] = (ref[:, 0] - cx) / half_span
            norm[:, 1] = (ref[:, 1] - cy) / half_span
        else:
            radius = min(cx, cy)
            norm = np.zeros_like(ref)
            norm[:, 0] = (ref[:, 0] - cx) / radius
            norm[:, 1] = (ref[:, 1] - cy) / radius

        self._sub_centers_norm = norm
        return self._sub_centers_norm

    def compute_slopes(self, wavefront: np.ndarray, grid_size: int) -> np.ndarray:
        """Compute local slopes at each sub-aperture center.

        wavefront: 2D array on unit disk grid of shape (grid_size, grid_size).
        Returns: (N, 2) slopes in normalized units.
        """
        dx = 2.0 / (grid_size - 1)
        dWdy, dWdx = np.gradient(wavefront, dx, dx)

        centers = self.subaperture_centers_normalized()
        slopes = np.zeros((len(centers), 2), dtype=float)

        for i, (xn, yn) in enumerate(centers):
            ix = int(np.round((xn + 1) / 2 * (grid_size - 1)))
            iy = int(np.round((yn + 1) / 2 * (grid_size - 1)))
            ix = np.clip(ix, 1, grid_size - 2)
            iy = np.clip(iy, 1, grid_size - 2)
            slopes[i, 0] = dWdx[iy, ix]
            slopes[i, 1] = dWdy[iy, ix]

        return slopes

    def slopes_to_displacements(self, slopes: np.ndarray) -> np.ndarray:
        """Convert slopes to focal plane displacements (um).

        disp = f * slope_norm * correction
        where correction = wavelength_um / R_pupil_um for fixed MLA grid,
        or 1.0 for the old circular-aperture mode (backward compat).
        """
        return slopes * self.focal_um * self._slope_correction

    def displaced_positions(self, slopes: np.ndarray) -> np.ndarray:
        """Compute displaced spot positions from slopes."""
        ref = self.reference_positions()
        disp = self.slopes_to_displacements(slopes)
        return ref + disp

    def check_bounds(self, positions: np.ndarray) -> np.ndarray:
        """Check which spots are within sensor bounds. Returns bool array."""
        x_ok = (positions[:, 0] >= 0) & (positions[:, 0] < self.sensor_width_um)
        y_ok = (positions[:, 1] >= 0) & (positions[:, 1] < self.sensor_height_um)
        return x_ok & y_ok
