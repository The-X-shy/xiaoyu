"""Tests for ChamferOptimizer (differentiable Chamfer distance optimiser)."""

import numpy as np
from src.sim.lenslet import LensletArray
from src.sim.pipeline import forward_pipeline
from src.recon.chamfer_optimizer import ChamferOptimizer
from src.recon.zernike import num_zernike_terms


def _make_lenslet(sensor_px=512):
    return LensletArray(
        pitch_um=150.0,
        focal_mm=6.0,
        fill_factor=0.95,
        sensor_width_px=sensor_px,
        sensor_height_px=sensor_px,
        pixel_um=10.0,
    )


def _base_cfg(order=3):
    return {
        "optics": {
            "wavelength_nm": 633.0,
            "pitch_um": 150.0,
            "focal_mm": 6.0,
            "fill_factor": 0.95,
        },
        "sensor": {
            "pixel_um": 10.0,
            "width_px": 512,
            "height_px": 512,
        },
        "noise": {
            "read_sigma": 0.0,
            "background": 0.0,
            "centroid_noise_px": 0.0,
        },
        "zernike": {
            "order": order,
            "coeff_bound": 1.0,
        },
        "asm": {
            "grid_size": 128,
            "lambda_reg": 1e-3,
            "chamfer_n_starts": 8,
            "chamfer_n_iter": 300,
            "chamfer_lr": 0.05,
            "chamfer_lr_decay": 0.5,
            "chamfer_decay_steps": 100,
            "chamfer_lambda_reg": 1e-3,
            "chamfer_lambda_ib": 10.0,
            "chamfer_max_points_fwd": 1024,
            "chamfer_max_points_bwd": 1024,
            "chamfer_search_bound": 2.0,
        },
    }


class TestChamferConstruction:
    def test_construction(self):
        """Create a ChamferOptimizer with a small LensletArray and verify init."""
        la = _make_lenslet()
        cfg = _base_cfg()
        observed = la.reference_positions().astype(np.float32)

        opt = ChamferOptimizer(observed, la, cfg)

        assert opt.n_sub == la.n_subapertures
        assert opt.n_obs == len(observed)
        assert opt.n_terms == num_zernike_terms(3)
        assert opt.G.shape == (2 * opt.n_sub, opt.n_terms)


class TestChamferZeroAberration:
    def test_zero_aberration(self):
        """With zero aberration, optimizer should converge to near-zero coefficients."""
        la = _make_lenslet()
        cfg = _base_cfg()
        observed = la.reference_positions().astype(np.float32)

        opt = ChamferOptimizer(observed, la, cfg)
        result = opt.run(seed=42)

        coeffs = result["coeffs"]
        assert coeffs.shape == (num_zernike_terms(3),)
        assert np.all(np.abs(coeffs) < 0.01), (
            f"Expected near-zero coefficients, got max |c|={np.max(np.abs(coeffs)):.4f}"
        )


class TestChamferPV1Recovery:
    def test_pv1_recovery(self):
        """Generate spots at PV~1.0 wave, verify RMSE < 0.1 lambda."""
        cfg = _base_cfg()
        cfg["asm"]["chamfer_n_starts"] = 12
        cfg["asm"]["chamfer_n_iter"] = 300

        sim = forward_pipeline(cfg, pv=1.0, seed=100)
        observed = sim["observed_positions"]
        la = sim["lenslet"]
        true_coeffs = sim["coeffs"]

        opt = ChamferOptimizer(observed, la, cfg)
        result = opt.run(seed=42)

        recon_coeffs = result["coeffs"]
        # Skip piston (index 0) -- unobservable from slopes
        rmse = np.sqrt(np.mean((recon_coeffs[1:] - true_coeffs[1:]) ** 2))
        assert rmse < 0.1, f"PV1 RMSE = {rmse:.4f} waves (expected < 0.1)"


class TestChamferPV5Recovery:
    def test_pv5_recovery(self):
        """Generate spots at PV~5.0 waves, verify RMSE < 0.15 lambda.

        At PV=5.0 with 512px sensor, many spots leave the sensor.  The
        Chamfer optimizer should still converge because it does not need
        explicit matching -- it directly minimises the point-set distance.
        Use a larger sensor if the 512px test proves too constrained.
        """
        cfg = _base_cfg()
        # Use 1024px sensor for PV=5 to keep enough spots on sensor
        cfg["sensor"]["width_px"] = 1024
        cfg["sensor"]["height_px"] = 1024
        cfg["asm"]["chamfer_n_starts"] = 16
        cfg["asm"]["chamfer_n_iter"] = 500
        cfg["asm"]["chamfer_search_bound"] = 4.0
        cfg["asm"]["chamfer_lr"] = 0.03

        sim = forward_pipeline(cfg, pv=5.0, seed=200)
        observed = sim["observed_positions"]
        la = sim["lenslet"]
        true_coeffs = sim["coeffs"]

        opt = ChamferOptimizer(observed, la, cfg)
        result = opt.run(seed=42)

        recon_coeffs = result["coeffs"]
        # Skip piston (index 0)
        rmse = np.sqrt(np.mean((recon_coeffs[1:] - true_coeffs[1:]) ** 2))
        assert rmse < 0.15, f"PV5 RMSE = {rmse:.4f} waves (expected < 0.15)"
