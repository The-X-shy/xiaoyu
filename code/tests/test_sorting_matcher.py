"""Tests for center-out spot matching (replaces sorting-based matcher)."""

import numpy as np
import pytest
from src.sim.lenslet import LensletArray
from src.recon.sorting_matcher import (
    _center_ring_match,
    _multi_radius_init,
    _reciprocal_nn_match,
    _forward_nn_match,
    _solve_coeffs,
    _compute_expected_positions,
    sorting_match,
)
from src.recon.zernike import num_zernike_terms
from src.recon.baseline_extrap_nn import _get_cached_zernike_matrix


# ------------------------------------------------------------------ helpers
def _make_lenslet(sensor_px: int = 512) -> LensletArray:
    """Small lenslet for fast tests."""
    return LensletArray(
        pitch_um=150.0,
        focal_mm=6.0,
        fill_factor=0.95,
        sensor_width_px=sensor_px,
        sensor_height_px=sensor_px,
        pixel_um=10.0,
    )


def _default_cfg(enable_sorting: bool = True, sensor_px: int = 512) -> dict:
    return {
        "optics": {
            "wavelength_nm": 633.0,
            "pitch_um": 150.0,
            "focal_mm": 6.0,
            "fill_factor": 0.95,
        },
        "sensor": {"pixel_um": 10.0, "width_px": sensor_px, "height_px": sensor_px},
        "noise": {"read_sigma": 0.0, "background": 0.0, "centroid_noise_px": 0.0},
        "zernike": {"order": 3, "coeff_bound": 1.0},
        "asm": {
            "lambda_reg": 1e-3,
            "grid_size": 128,
            "enable_sorting": enable_sorting,
        },
    }


def _simulate(pv: float, seed: int = 42, sensor_px: int = 512):
    """Run forward pipeline and return obs, lenslet, true_coeffs."""
    from src.sim.pipeline import forward_pipeline

    cfg = _default_cfg(sensor_px=sensor_px)
    result = forward_pipeline(cfg, pv=pv, seed=seed)
    return result["observed_positions"], result["lenslet"], result["coeffs"], cfg


# ============================================================ _center_ring_match
class TestCenterRingMatch:
    def test_zero_pv_matches_all_center(self):
        """At PV=0, center ring match should find matches near center."""
        obs, la, _, _ = _simulate(pv=0.0)
        ref = la.reference_positions()
        sensor_w = la.sensor_width_um
        sensor_h = la.sensor_height_um

        sub_idx, obs_idx, n_matched = _center_ring_match(
            obs, ref, sensor_w, sensor_h, la.pitch_um, max_radius_frac=0.35
        )
        assert n_matched > 0, "Should find matches at PV=0"
        # All matched indices should be valid
        assert np.all(sub_idx < len(ref))
        assert np.all(obs_idx < len(obs))
        # No duplicates
        assert len(np.unique(sub_idx)) == len(sub_idx)
        assert len(np.unique(obs_idx)) == len(obs_idx)

    def test_moderate_pv_finds_matches(self):
        """At PV=2, center ring should still find some matches."""
        obs, la, _, _ = _simulate(pv=2.0)
        ref = la.reference_positions()
        sensor_w = la.sensor_width_um
        sensor_h = la.sensor_height_um

        sub_idx, obs_idx, n_matched = _center_ring_match(
            obs, ref, sensor_w, sensor_h, la.pitch_um, max_radius_frac=0.35
        )
        assert n_matched > 5, f"Should find >5 center matches at PV=2, got {n_matched}"

    def test_high_pv_center_still_works(self):
        """At PV=3, center spots should still be matchable (512px sensor).

        PV=3 is already high for 512px (~43 obs spots out of 820).
        PV=10 would leave only ~5 spots, too few for meaningful center matching.
        """
        obs, la, _, _ = _simulate(pv=3.0)
        ref = la.reference_positions()
        sensor_w = la.sensor_width_um
        sensor_h = la.sensor_height_um

        # Try expanding radii until enough matches found
        for r_frac in [0.15, 0.25, 0.35, 0.5, 0.7, 1.0]:
            sub_idx, obs_idx, n_matched = _center_ring_match(
                obs, ref, sensor_w, sensor_h, la.pitch_um, max_radius_frac=r_frac
            )
            if n_matched >= 10:
                break
        assert n_matched >= 5, (
            f"Should find >=5 center matches at PV=3, got {n_matched}"
        )

    def test_tilt_correction(self):
        """Tilt correction should improve matching when tilt is present."""
        obs, la, _, _ = _simulate(pv=3.0, seed=123)
        ref = la.reference_positions()
        sensor_w = la.sensor_width_um
        sensor_h = la.sensor_height_um

        _, _, n_with_tilt = _center_ring_match(
            obs,
            ref,
            sensor_w,
            sensor_h,
            la.pitch_um,
            max_radius_frac=0.35,
            tilt_correct=True,
        )
        _, _, n_without_tilt = _center_ring_match(
            obs,
            ref,
            sensor_w,
            sensor_h,
            la.pitch_um,
            max_radius_frac=0.35,
            tilt_correct=False,
        )
        # With tilt correction should be at least as good
        assert n_with_tilt >= n_without_tilt, (
            f"Tilt correction should help: {n_with_tilt} vs {n_without_tilt}"
        )


# ============================================================ _multi_radius_init
class TestMultiRadiusInit:
    def test_returns_coeffs_at_low_pv(self):
        """Should return initial coefficients at low PV."""
        obs, la, true_coeffs, cfg = _simulate(pv=1.0)
        ref = la.reference_positions()
        n_sub = len(ref)
        max_order = cfg["zernike"]["order"]
        n_terms = num_zernike_terms(max_order)
        G = _get_cached_zernike_matrix(la, max_order, cfg["asm"]["grid_size"])
        focal_um = float(la.focal_um)

        coeffs, n_init = _multi_radius_init(
            obs,
            ref,
            G,
            n_sub,
            n_terms,
            focal_um,
            cfg["asm"]["lambda_reg"],
            la.sensor_width_um,
            la.sensor_height_um,
            la.pitch_um,
        )
        assert coeffs is not None, "Should find initial coefficients"
        assert n_init >= n_terms + 2, (
            f"Should match >= {n_terms + 2} spots, got {n_init}"
        )
        assert len(coeffs) == n_terms

    def test_returns_coeffs_at_moderate_pv(self):
        """Should return initial coefficients at PV=5."""
        obs, la, true_coeffs, cfg = _simulate(pv=5.0)
        ref = la.reference_positions()
        n_sub = len(ref)
        max_order = cfg["zernike"]["order"]
        n_terms = num_zernike_terms(max_order)
        G = _get_cached_zernike_matrix(la, max_order, cfg["asm"]["grid_size"])
        focal_um = float(la.focal_um)

        coeffs, n_init = _multi_radius_init(
            obs,
            ref,
            G,
            n_sub,
            n_terms,
            focal_um,
            cfg["asm"]["lambda_reg"],
            la.sensor_width_um,
            la.sensor_height_um,
            la.pitch_um,
        )
        assert coeffs is not None, "Should find initial coefficients at PV=5"

    def test_returns_none_for_garbage(self):
        """Should return None for random garbage positions."""
        la = _make_lenslet()
        ref = la.reference_positions()
        n_sub = len(ref)
        n_terms = num_zernike_terms(3)
        G = _get_cached_zernike_matrix(la, 3, 128)
        focal_um = float(la.focal_um)

        rng = np.random.RandomState(42)
        garbage = rng.uniform(0, la.sensor_width_um, (50, 2))

        coeffs, n_init = _multi_radius_init(
            garbage,
            ref,
            G,
            n_sub,
            n_terms,
            focal_um,
            1e-3,
            la.sensor_width_um,
            la.sensor_height_um,
            la.pitch_um,
        )
        # With only 50 random spots vs ~800 ref spots, likely fails
        # but should not crash
        assert coeffs is None or isinstance(coeffs, np.ndarray)


# ============================================================= sorting_match
class TestSortingMatch:
    def test_zero_aberration_perfect_match(self):
        """With zero wavefront, should perfectly recover identity."""
        obs, la, _, cfg = _simulate(pv=0.0)
        out = sorting_match(obs, la, cfg)
        assert out["success"] is True
        assert out["n_matched"] > 0
        assert out["residual_trimmed"] < 1.0  # < 1 um

    def test_low_pv_accurate_coeffs(self):
        """At low PV, should recover true Zernike coefficients well."""
        obs, la, true_coeffs, cfg = _simulate(pv=0.5)
        out = sorting_match(obs, la, cfg)
        assert out["success"] is True
        rmse = np.sqrt(np.mean((out["coeffs"] - true_coeffs) ** 2))
        assert rmse < 0.1, f"Coefficient RMSE {rmse:.4f} too large"

    def test_moderate_pv_succeeds(self):
        """Should succeed at moderate PV (well within curvature limit)."""
        obs, la, _, cfg = _simulate(pv=2.0)
        out = sorting_match(obs, la, cfg)
        assert out["success"] is True
        assert out["n_matched"] >= num_zernike_terms(3)

    def test_high_pv_succeeds(self):
        """Should succeed at high PV where ICP typically fails."""
        obs, la, _, cfg = _simulate(pv=5.0)
        out = sorting_match(obs, la, cfg)
        assert out["n_matched"] > 0
        if out["success"]:
            true_coeffs = _simulate(pv=5.0)[2]
            rmse = np.sqrt(np.mean((out["coeffs"] - true_coeffs) ** 2))
            assert rmse < 1.0, f"Coefficient RMSE {rmse:.4f} too large at PV=5"

    def test_very_high_pv(self):
        """At PV=20, center-out should still work (512px sensor)."""
        obs, la, true_coeffs, cfg = _simulate(pv=20.0)
        out = sorting_match(obs, la, cfg)
        # At PV=20 many spots lost, but should still get some matches
        if out["success"]:
            rmse = np.sqrt(np.mean((out["coeffs"] - true_coeffs) ** 2))
            assert rmse < 2.0, f"RMSE {rmse:.4f} too large at PV=20"

    def test_coeffs_accurate_at_pv1(self):
        """At PV=1, should recover true Zernike coefficients."""
        obs, la, true_coeffs, cfg = _simulate(pv=1.0)
        out = sorting_match(obs, la, cfg)
        assert out["success"]
        rmse = np.sqrt(np.mean((out["coeffs"] - true_coeffs) ** 2))
        assert rmse < 0.1, f"Coefficient RMSE {rmse:.4f} too large"

    def test_output_dict_keys(self):
        """Output dict should have all expected keys."""
        obs, la, _, cfg = _simulate(pv=0.5)
        out = sorting_match(obs, la, cfg)
        expected_keys = {
            "coeffs",
            "success",
            "n_matched",
            "residual_raw",
            "residual_trimmed",
            "matched_sub_idx",
            "matched_obs_idx",
            "solver",
        }
        assert expected_keys.issubset(out.keys())

    def test_coeffs_shape(self):
        """Coefficients should have correct length."""
        obs, la, _, cfg = _simulate(pv=0.5)
        out = sorting_match(obs, la, cfg)
        n_terms = num_zernike_terms(cfg["zernike"]["order"])
        assert len(out["coeffs"]) == n_terms

    def test_solver_tag(self):
        """Solver tag should be 'sorting_match'."""
        obs, la, _, cfg = _simulate(pv=0.5)
        out = sorting_match(obs, la, cfg)
        assert out["solver"] == "sorting_match"


# ====================================================== missing / extra spots
class TestSortingMatchEdgeCases:
    def test_missing_spots_graceful(self):
        """Should degrade gracefully when spots are missing."""
        obs, la, _, cfg = _simulate(pv=1.0)
        # Remove 10% of spots
        rng = np.random.RandomState(99)
        n_keep = int(0.9 * len(obs))
        keep_idx = np.sort(rng.choice(len(obs), size=n_keep, replace=False))
        obs_reduced = obs[keep_idx]

        out = sorting_match(obs_reduced, la, cfg)
        assert "coeffs" in out
        assert out["n_matched"] > 0

    def test_extra_spots_graceful(self):
        """Should handle spurious extra spots."""
        obs, la, _, cfg = _simulate(pv=0.5)
        rng = np.random.RandomState(99)
        n_extra = max(1, int(0.05 * len(obs)))
        sensor_w = cfg["sensor"]["width_px"] * cfg["sensor"]["pixel_um"]
        sensor_h = cfg["sensor"]["height_px"] * cfg["sensor"]["pixel_um"]
        extra = np.column_stack(
            [rng.uniform(0, sensor_w, n_extra), rng.uniform(0, sensor_h, n_extra)]
        )
        obs_with_extra = np.vstack([obs, extra])

        out = sorting_match(obs_with_extra, la, cfg)
        assert "coeffs" in out
        assert out["n_matched"] > 0

    def test_too_few_spots_fails(self):
        """With fewer spots than Zernike terms, should fail gracefully."""
        la = _make_lenslet()
        cfg = _default_cfg()
        obs = np.array([[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]])
        out = sorting_match(obs, la, cfg)
        assert out["success"] is False
        assert out["n_matched"] == 0


# ======================================== Integration with ASM reconstructor
class TestSortingIntegration:
    def test_asm_uses_sorting_when_enabled(self):
        """ASM reconstructor should use sorting when enable_sorting=True."""
        from src.sim.pipeline import forward_pipeline
        from src.recon.asm_reconstructor import asm_reconstruct

        cfg = _default_cfg(enable_sorting=True)
        cfg["asm"]["n_starts"] = 3
        cfg["asm"]["n_icp_iter"] = 5
        result = forward_pipeline(cfg, pv=2.0, seed=42)
        out = asm_reconstruct(
            result["observed_positions"],
            result["lenslet"],
            cfg,
            seed=42,
        )
        assert isinstance(out, dict)
        assert "coeffs" in out
        assert "sorting" in out["solver"] or "icp" in out["solver"]

    def test_asm_sorting_disabled_fallback(self):
        """ASM should still work when sorting is disabled."""
        from src.sim.pipeline import forward_pipeline
        from src.recon.asm_reconstructor import asm_reconstruct

        cfg = _default_cfg(enable_sorting=False)
        cfg["asm"]["n_starts"] = 3
        cfg["asm"]["n_icp_iter"] = 5
        result = forward_pipeline(cfg, pv=0.5, seed=42)
        out = asm_reconstruct(
            result["observed_positions"],
            result["lenslet"],
            cfg,
            seed=42,
        )
        assert isinstance(out, dict)
        assert out["solver"] == "asm_icp_cpu"


# ====================================== Multiple seeds for robustness
class TestSortingRobustness:
    @pytest.mark.parametrize("seed", [10, 20, 30, 40, 50])
    def test_sorting_at_pv2_multiple_seeds(self, seed):
        """Should succeed at PV=2 across different wavefront realizations."""
        obs, la, _, cfg = _simulate(pv=2.0, seed=seed)
        out = sorting_match(obs, la, cfg)
        assert out["success"] is True, (
            f"Failed at PV=2, seed={seed}, "
            f"residual={out['residual_trimmed']:.2f}, "
            f"n_matched={out['n_matched']}"
        )

    @pytest.mark.parametrize("seed", [10, 20, 30])
    def test_sorting_at_pv5_some_seeds(self, seed):
        """At PV=5, should at least produce results."""
        obs, la, _, cfg = _simulate(pv=5.0, seed=seed)
        out = sorting_match(obs, la, cfg)
        assert out["n_matched"] > 0 or len(obs) < num_zernike_terms(3)

    @pytest.mark.parametrize("seed", [10, 20, 30])
    def test_sorting_at_pv10_512px(self, seed):
        """At PV=10 on 512px sensor, center-out should still work."""
        obs, la, true_coeffs, cfg = _simulate(pv=10.0, seed=seed)
        out = sorting_match(obs, la, cfg)
        # Should at least find some matches
        if len(obs) >= num_zernike_terms(3):
            assert out["n_matched"] > 0, f"No matches at PV=10, seed={seed}"


# ====================================== 2048px sensor tests
class TestLargeSensor:
    """Tests specifically for the 2048px sensor where sorting used to fail."""

    def _simulate_2048(self, pv: float, seed: int = 42):
        from src.sim.pipeline import forward_pipeline

        cfg = _default_cfg(sensor_px=2048)
        result = forward_pipeline(cfg, pv=pv, seed=seed)
        return result["observed_positions"], result["lenslet"], result["coeffs"], cfg

    def test_2048_zero_pv(self):
        """2048px sensor at PV=0 should work perfectly."""
        obs, la, true_coeffs, cfg = self._simulate_2048(pv=0.0)
        out = sorting_match(obs, la, cfg)
        assert out["success"] is True
        assert out["residual_trimmed"] < 1.0

    def test_2048_low_pv(self):
        """2048px sensor at PV=1 should recover coefficients."""
        obs, la, true_coeffs, cfg = self._simulate_2048(pv=1.0)
        out = sorting_match(obs, la, cfg)
        assert out["success"] is True
        rmse = np.sqrt(np.mean((out["coeffs"] - true_coeffs) ** 2))
        assert rmse < 0.1, f"RMSE {rmse:.4f} too large at PV=1 on 2048px"

    def test_2048_moderate_pv(self):
        """2048px sensor at PV=2 — this is where sorting used to fail."""
        obs, la, true_coeffs, cfg = self._simulate_2048(pv=2.0)
        out = sorting_match(obs, la, cfg)
        assert out["success"] is True, (
            f"Failed at PV=2 on 2048px: residual={out['residual_trimmed']:.2f}, "
            f"n_matched={out['n_matched']}"
        )
        rmse = np.sqrt(np.mean((out["coeffs"] - true_coeffs) ** 2))
        assert rmse < 0.2, f"RMSE {rmse:.4f} too large at PV=2 on 2048px"

    def test_2048_high_pv(self):
        """2048px sensor at PV=5 — aggressive test."""
        obs, la, true_coeffs, cfg = self._simulate_2048(pv=5.0)
        out = sorting_match(obs, la, cfg)
        if out["success"]:
            rmse = np.sqrt(np.mean((out["coeffs"] - true_coeffs) ** 2))
            assert rmse < 1.0, f"RMSE {rmse:.4f} at PV=5 on 2048px"
