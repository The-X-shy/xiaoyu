"""Matching-free wavefront reconstruction via backward Chamfer distance.

v7 — Full-prediction backward Chamfer with chunked GPU evaluation:
  Phase 1: Sample random coefficient vectors using mixed sampling
           (uniform cube + spherical) and evaluate on the full set of
           subaperture predictions (no pred-subsampling).
  Phase 2: Adam refinement from the best candidates.

CRITICAL FIX over v6: Previous versions subsampled predictions, which
made the objective incorrect at high PV.  At PV=15 only ~3.6% of
predictions are in-bounds, so a random 512-of-13224 subsample keeps
only ~18 in-bounds predictions while there are ~476 observations.
The zero solution with all 512 in-bounds beats the true solution.

Now we always use ALL subapertures and chunk the cdist computation to
stay within GPU memory.
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Dict, Any, Optional, List

from src.sim.lenslet import LensletArray
from src.recon.zernike import num_zernike_terms
from src.recon.asm_gpu import _get_cached_g_tensor


class ChamferOptimizer:
    """GPU-batched backward-Chamfer optimizer: smart random search + Adam.

    Phase 1: Evaluate N_sample random coefficient vectors.
    Phase 2: Run Adam from the top-K starting points.

    Uses ALL subaperture predictions (no subsampling) and chunks the
    distance computation to fit in GPU memory.
    """

    def __init__(
        self,
        observed: np.ndarray,
        lenslet: LensletArray,
        cfg: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        asm_cfg = cfg.get("asm", {})
        zer_cfg = cfg.get("zernike", {})

        self.max_order = int(zer_cfg.get("order", 3))
        self.n_terms = num_zernike_terms(self.max_order)
        grid_size = int(asm_cfg.get("grid_size", 128))

        # Phase 1 config: random sampling
        self.n_sample = int(asm_cfg.get("chamfer_n_sample", 30000))
        self.sample_batch = int(asm_cfg.get("chamfer_sample_batch", 2048))
        self.sample_topk = int(asm_cfg.get("chamfer_sample_topk", 64))
        self.sample_obs_k = int(asm_cfg.get("chamfer_sample_obs_k", 256))

        # Phase 2 config: Adam refinement
        self.n_refine = int(asm_cfg.get("chamfer_n_refine", 64))
        self.refine_iter = int(asm_cfg.get("chamfer_refine_iter", 300))
        self.refine_lr = float(asm_cfg.get("chamfer_refine_lr", 0.05))
        self.refine_obs_k = int(asm_cfg.get("chamfer_refine_obs_k", 512))

        # Search range: max radius for spherical sampling
        self.max_radius = float(asm_cfg.get("chamfer_max_radius", 10.0))

        # Common config
        self.lambda_reg = float(
            asm_cfg.get("chamfer_lambda_reg", asm_cfg.get("lambda_reg", 1e-3))
        )

        # Optics
        self.focal_um = float(lenslet.focal_um)
        self.pitch_um = float(lenslet.pitch_um)
        self.sensor_w = float(lenslet.sensor_width_um)
        self.sensor_h = float(lenslet.sensor_height_um)

        # G matrix: (2*N_sub, n_terms)
        self.G = _get_cached_g_tensor(lenslet, self.max_order, grid_size, device)

        # Reference positions: (N_sub, 2)
        ref_np = lenslet.reference_positions()
        self.n_sub = len(ref_np)
        self.ref = torch.tensor(ref_np, dtype=torch.float32, device=device)

        # Observed spots: (M, 2)
        obs_np = np.asarray(observed, dtype=np.float32)
        self.n_obs = len(obs_np)
        self.observed_full = torch.tensor(obs_np, dtype=torch.float32, device=device)

        # Memory-management: max predictions per chunk in cdist
        # With B=2048, k_obs=256, pred_chunk=2048:
        #   tensor size = 2048*2048*256*4 bytes = 4GB (too much)
        # With B=2048, k_obs=256, pred_chunk=512:
        #   tensor size = 2048*512*256*4 = 1GB (ok)
        # We'll compute the chunk size dynamically based on batch size.
        self.pred_chunk = int(asm_cfg.get("chamfer_pred_chunk", 2048))

    def _compute_expected(
        self,
        coeffs_batch: torch.Tensor,
        ref: torch.Tensor,
        G: torch.Tensor,
        k_pred: int,
    ) -> torch.Tensor:
        """Compute expected spots.  (B, k_pred, 2)"""
        slopes_vec = coeffs_batch @ G.T  # (B, 2*k_pred)
        slopes_x = slopes_vec[:, :k_pred]
        slopes_y = slopes_vec[:, k_pred:]
        E_x = ref[:, 0].unsqueeze(0) + self.focal_um * slopes_x
        E_y = ref[:, 1].unsqueeze(0) + self.focal_um * slopes_y
        return torch.stack([E_x, E_y], dim=-1)

    def _backward_chamfer_full(
        self,
        coeffs_batch: torch.Tensor,
        obs: torch.Tensor,
        differentiable: bool = False,
    ) -> torch.Tensor:
        """Backward Chamfer using ALL predictions (no subsampling).

        For each observation, finds the nearest in-bounds predicted spot.
        Chunks the prediction set to manage GPU memory.

        Returns: (B,) objective values
        """
        B = coeffs_batch.shape[0]
        k_obs = obs.shape[0]
        k_pred = self.n_sub

        # Compute ALL expected positions: (B, n_sub, 2)
        E = self._compute_expected(coeffs_batch, self.ref, self.G, k_pred)

        # In-bounds mask: (B, n_sub)
        in_bounds = (
            (E[:, :, 0] >= 0)
            & (E[:, :, 0] <= self.sensor_w)
            & (E[:, :, 1] >= 0)
            & (E[:, :, 1] <= self.sensor_h)
        )

        large_val = torch.tensor(1e6, dtype=torch.float32, device=self.device)

        # Compute min distance per observation by chunking over predictions
        # nn_dist: (B, k_obs) — minimum distance from each obs to nearest
        # in-bounds prediction
        nn_dist = torch.full((B, k_obs), 1e6, dtype=torch.float32, device=self.device)
        nn_idx = torch.zeros((B, k_obs), dtype=torch.long, device=self.device)

        chunk_size = self.pred_chunk
        for p_start in range(0, k_pred, chunk_size):
            p_end = min(p_start + chunk_size, k_pred)
            E_chunk = E[:, p_start:p_end, :]  # (B, chunk, 2)
            ib_chunk = in_bounds[:, p_start:p_end]  # (B, chunk)

            # Pairwise distances: (B, chunk, k_obs)
            O_exp = obs.unsqueeze(0).expand(B, -1, -1)
            dists_chunk = torch.cdist(E_chunk, O_exp) / self.pitch_um

            # Mask OOB predictions
            oob_mask = (~ib_chunk).unsqueeze(2).expand(-1, -1, k_obs)
            if differentiable:
                # For gradient flow: use straight-through argmin
                dists_for_sel = dists_chunk.detach().clone()
                dists_for_sel[oob_mask] = 1e6
                # min over this chunk's predictions for each obs
                chunk_min_vals, chunk_min_idx = dists_for_sel.min(dim=1)  # (B, k_obs)
                # Gather actual (differentiable) distances at those indices
                idx_exp = chunk_min_idx.unsqueeze(1)  # (B, 1, k_obs)
                chunk_nn = dists_chunk.gather(1, idx_exp).squeeze(1)  # (B, k_obs)
                # Update running min
                better = chunk_min_vals < nn_dist
                nn_dist = torch.where(better, chunk_nn, nn_dist)
                nn_idx = torch.where(better, chunk_min_idx + p_start, nn_idx)
            else:
                dists_chunk[oob_mask] = 1e6
                chunk_min_vals = dists_chunk.min(dim=1).values  # (B, k_obs)
                better = chunk_min_vals < nn_dist
                nn_dist = torch.where(better, chunk_min_vals, nn_dist)

        # For samples with NO in-bounds predictions at all, use fallback
        any_inbounds = in_bounds.any(dim=1)  # (B,)
        sensor_cx = self.sensor_w / 2.0
        sensor_cy = self.sensor_h / 2.0
        dist_to_center = (
            (E[:, :, 0] - sensor_cx) ** 2 + (E[:, :, 1] - sensor_cy) ** 2
        ).sqrt().mean(dim=1) / self.pitch_um

        # Cap individual nn distances
        cap = 5.0
        nn_dist_capped = nn_dist.clamp(max=cap)
        bwd_chamfer = nn_dist_capped.mean(dim=1)  # (B,)

        loss = torch.where(any_inbounds, bwd_chamfer, dist_to_center + 100.0)

        # Mild regularisation
        reg = (coeffs_batch * coeffs_batch).mean(dim=1)

        return loss + self.lambda_reg * reg

    def _subsample_obs(self, obs_k: int, rng: np.random.RandomState) -> torch.Tensor:
        """Subsample observations only (predictions always use full set)."""
        obs_k = min(obs_k, self.n_obs)
        if obs_k < self.n_obs:
            idx = rng.choice(self.n_obs, size=obs_k, replace=False)
            return self.observed_full[
                torch.tensor(idx, dtype=torch.long, device=self.device)
            ]
        return self.observed_full

    def _generate_samples(self, n: int, rng: np.random.RandomState) -> torch.Tensor:
        """Generate mixed random samples: 50% uniform cube, 50% spherical."""
        D = self.n_terms
        n_cube = n // 2
        n_sphere = n - n_cube

        cube_bound = 3.0
        cube = rng.uniform(-cube_bound, cube_bound, size=(n_cube, D))

        gauss = rng.randn(n_sphere, D)
        norms = np.linalg.norm(gauss, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        directions = gauss / norms
        radii = rng.uniform(0, self.max_radius, size=(n_sphere, 1))
        sphere = directions * radii

        samples = np.concatenate([cube, sphere], axis=0).astype(np.float32)
        rng.shuffle(samples)
        return torch.tensor(samples, dtype=torch.float32, device=self.device)

    def run(
        self,
        seed: int = 42,
        init_coeffs: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Two-phase optimisation: smart random sampling + Adam refinement.

        Args:
            seed: random seed.
            init_coeffs: optional list of initial guesses.

        Returns:
            dict with ``coeffs`` (np.ndarray) and ``objective_value`` (float).
        """
        rng = np.random.RandomState(seed)
        torch.manual_seed(seed)

        D = self.n_terms

        # === Phase 1: Dense random sampling (coarse obs subset) ===
        obs_s = self._subsample_obs(self.sample_obs_k, rng)

        all_objectives = []
        all_coeffs = []

        # Include provided initial coefficients
        if init_coeffs is not None:
            extras = []
            for c in init_coeffs:
                c_arr = np.asarray(c, dtype=np.float32).reshape(-1)
                if c_arr.size == D:
                    extras.append(c_arr)
            if extras:
                extras_np = np.stack(extras, axis=0)
                extras_t = torch.tensor(
                    extras_np, dtype=torch.float32, device=self.device
                )
                with torch.no_grad():
                    obj = self._backward_chamfer_full(extras_t, obs_s)
                all_objectives.append(obj)
                all_coeffs.append(extras_t)

        # Include zero
        zero_t = torch.zeros((1, D), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            obj_zero = self._backward_chamfer_full(zero_t, obs_s)
        all_objectives.append(obj_zero)
        all_coeffs.append(zero_t)

        # Generate and evaluate samples in batches
        all_samples = self._generate_samples(self.n_sample, rng)
        for start in range(0, self.n_sample, self.sample_batch):
            end = min(start + self.sample_batch, self.n_sample)
            batch = all_samples[start:end]
            with torch.no_grad():
                obj = self._backward_chamfer_full(batch, obs_s)
            all_objectives.append(obj)
            all_coeffs.append(batch)

        # Select top-K
        all_obj = torch.cat(all_objectives)
        all_c = torch.cat(all_coeffs)
        topk_k = min(self.sample_topk, len(all_obj))
        _, topk_idx = torch.topk(all_obj, topk_k, largest=False)
        top_coeffs = all_c[topk_idx]

        # === Phase 2: Adam refinement (larger obs subset) ===
        obs_r = self._subsample_obs(self.refine_obs_k, rng)

        n_refine = min(self.n_refine, topk_k)
        coeffs = top_coeffs[:n_refine].clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([coeffs], lr=self.refine_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.refine_iter, eta_min=self.refine_lr * 0.01
        )

        for it in range(self.refine_iter):
            optimizer.zero_grad()
            loss = self._backward_chamfer_full(coeffs, obs_r, differentiable=True)
            total_loss = loss.sum()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([coeffs], max_norm=5.0)
            optimizer.step()
            scheduler.step()

        # Select best chain using all observations
        with torch.no_grad():
            obs_f = self._subsample_obs(min(2048, self.n_obs), rng)
            final_loss = self._backward_chamfer_full(coeffs, obs_f)
            best_idx = int(final_loss.argmin().item())
            best_coeffs = coeffs[best_idx].detach().cpu().numpy()
            best_obj = float(final_loss[best_idx].item())

        return {
            "coeffs": best_coeffs,
            "objective_value": best_obj,
        }
