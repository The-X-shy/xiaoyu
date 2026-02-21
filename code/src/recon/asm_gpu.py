"""GPU-accelerated ASM ICP reconstructor using PyTorch.

Batches ICP chains in mini-batches on GPU to avoid OOM.
Uses reciprocal nearest-neighbor matching and baseline warm start.
Falls back to CPU PyTorch if no CUDA device is available.
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple

from src.sim.lenslet import LensletArray
from src.recon.zernike import num_zernike_terms
from src.recon.baseline_extrap_nn import _get_cached_zernike_matrix


_G_TENSOR_CACHE: dict[Tuple[float, ...], torch.Tensor] = {}


def _matrix_cache_key(
    la: LensletArray, max_order: int, grid_size: int
) -> Tuple[float, ...]:
    return (
        float(la.pitch_um),
        float(la.focal_mm),
        float(la.fill_factor),
        float(la.sensor_width_px),
        float(la.sensor_height_px),
        float(la.pixel_um),
        float(max_order),
        float(grid_size),
    )


def _get_cached_g_tensor(
    lenslet: LensletArray,
    max_order: int,
    grid_size: int,
    device: torch.device,
) -> torch.Tensor:
    key = _matrix_cache_key(lenslet, max_order, grid_size)
    G_t = _G_TENSOR_CACHE.get(key)
    if G_t is None or G_t.device != device:
        G_np = _get_cached_zernike_matrix(lenslet, max_order, grid_size)
        G_t = torch.tensor(G_np, dtype=torch.float32, device=device)
        _G_TENSOR_CACHE[key] = G_t
    return G_t


def _estimate_batch_size(n_sub: int, n_obs: int, vram_gb: float = 20.0) -> int:
    """Estimate safe mini-batch size to avoid OOM.

    Primary memory cost: cdist produces (B, N_sub, M) float32 tensor.
    Additional: E_masked (B, N_sub, 2), matched_obs (B, N_sub, 2),
                ATA (B, n_terms, n_terms), etc.

    We budget ~60% of VRAM for the distance matrix.
    """
    bytes_per_element = 4  # float32
    budget_bytes = vram_gb * 1e9 * 0.6
    # cdist tensor: B * n_sub * n_obs * 4 bytes
    cost_per_start = n_sub * n_obs * bytes_per_element
    if cost_per_start == 0:
        return 50
    batch_size = max(1, int(budget_bytes / cost_per_start))
    return batch_size


class BatchedICP:
    """GPU-batched ICP for ASM reconstruction.

    Runs ICP chains in mini-batches on GPU.
    Uses reciprocal (mutual) nearest-neighbor matching.
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

        self.max_order = zer_cfg.get("order", 3)
        self.n_terms = num_zernike_terms(self.max_order)
        self.lambda_reg = asm_cfg.get("lambda_reg", 1e-3)
        self.n_icp_iter = asm_cfg.get("n_icp_iter", 30)
        self.convergence_tol = asm_cfg.get("convergence_tol", 1e-6)
        self.max_match_dist_factor = asm_cfg.get("max_match_dist_factor", 0.35)
        self.min_match_ratio = asm_cfg.get("min_match_ratio", 0.2)
        self.trim_ratio = asm_cfg.get("trim_ratio", 0.9)
        self.allow_forward_fallback = asm_cfg.get("allow_forward_fallback", False)
        self.nn_ratio_threshold = float(asm_cfg.get("nn_ratio_threshold", 0.0))
        if self.nn_ratio_threshold <= 0:
            self.nn_ratio_threshold = None
        self.huber_delta_um = float(asm_cfg.get("huber_delta_um", 0.0))
        self.coarse_align_dist_factor = float(
            asm_cfg.get("coarse_align_dist_factor", 2.0)
        )
        self.topo_k_neighbors = int(asm_cfg.get("topo_k_neighbors", 4))
        self.topo_len_tol = float(asm_cfg.get("topo_len_tol", 0.35))
        self.topo_angle_weight = float(asm_cfg.get("topo_angle_weight", 0.5))
        self.topo_cost_thresh = float(asm_cfg.get("topo_cost_thresh", 1.0))
        self.topo_min_neighbors = int(asm_cfg.get("topo_min_neighbors", 2))
        self.icp_batch_cap = int(asm_cfg.get("icp_batch_cap", 8))

        # Precompute slope basis on CPU then transfer
        grid_size = asm_cfg.get("grid_size", 128)
        self.G = _get_cached_g_tensor(lenslet, self.max_order, grid_size, device)

        # Reference positions
        ref_np = lenslet.reference_positions()
        self.ref = torch.tensor(
            ref_np, dtype=torch.float32, device=device
        )  # (N_sub, 2)
        self.n_sub = len(ref_np)
        self.focal_um = float(lenslet.focal_um)

        # Sensor bounds
        self.sensor_w = float(lenslet.sensor_width_um)
        self.sensor_h = float(lenslet.sensor_height_um)
        self.pitch_um = float(lenslet.pitch_um)
        self.max_match_dist_um = self.max_match_dist_factor * self.pitch_um

        # Observed spots
        self.observed = torch.tensor(
            observed.astype(np.float32), device=device
        )  # (M, 2)
        self.n_obs = len(observed)
        self.min_required = max(
            self.n_terms * 3, int(self.n_obs * self.min_match_ratio)
        )

        # Precompute regularization matrix
        self.reg_eye = None
        if self.lambda_reg > 0:
            self.reg_eye = self.lambda_reg * torch.eye(
                self.n_terms, dtype=torch.float32, device=device
            )

        # Split G into x and y components
        self.G_x = self.G[: self.n_sub, :]  # (N_sub, n_terms)
        self.G_y = self.G[self.n_sub :, :]  # (N_sub, n_terms)

        # Static k-NN graph on reference subaperture centers for topology checks.
        if self.topo_k_neighbors > 0 and self.n_sub > 1:
            k_eff = min(self.topo_k_neighbors, self.n_sub - 1)
            d_ref = np.linalg.norm(
                ref_np[:, None, :] - ref_np[None, :, :], axis=2
            ).astype(np.float32)
            np.fill_diagonal(d_ref, np.inf)
            neigh = np.argsort(d_ref, axis=1)[:, :k_eff].astype(np.int64)
        else:
            neigh = np.zeros((self.n_sub, 0), dtype=np.int64)
        self.neigh_idx = torch.tensor(neigh, dtype=torch.long, device=device)

    def _compute_expected(self, coeffs_batch: torch.Tensor) -> torch.Tensor:
        """Compute expected spots for a batch of coefficient vectors.

        Args:
            coeffs_batch: (B, n_terms)

        Returns:
            (B, N_sub, 2) expected spot positions
        """
        # slopes_vec: (B, 2*N_sub) = (B, n_terms) @ (n_terms, 2*N_sub)
        slopes_vec = coeffs_batch @ self.G.T  # (B, 2*N_sub)

        slopes_x = slopes_vec[:, : self.n_sub]  # (B, N_sub)
        slopes_y = slopes_vec[:, self.n_sub :]  # (B, N_sub)

        E_x = self.ref[:, 0].unsqueeze(0) + self.focal_um * slopes_x  # (B, N_sub)
        E_y = self.ref[:, 1].unsqueeze(0) + self.focal_um * slopes_y  # (B, N_sub)

        return torch.stack([E_x, E_y], dim=-1)  # (B, N_sub, 2)

    def _apply_similarity_alignment(
        self,
        expected: torch.Tensor,
        matched_obs: torch.Tensor,
        match_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Coarse global alignment: similarity transform per chain.

        Align expected spots to observed spots using matched pairs, then
        continue assignment on aligned coordinates.
        """
        E_out = expected.clone()
        B = expected.shape[0]
        for b in range(B):
            idx = torch.where(match_mask[b])[0]
            if idx.numel() < 3:
                continue
            X = expected[b, idx]  # (K, 2)
            Y = matched_obs[b, idx]  # (K, 2)
            mx = X.mean(dim=0)
            my = Y.mean(dim=0)
            Xc = X - mx
            Yc = Y - my
            sx = torch.linalg.norm(Xc)
            sy = torch.linalg.norm(Yc)
            if sx.item() < 1e-6 or sy.item() < 1e-6:
                continue
            Xn = Xc / sx
            Yn = Yc / sy
            H = Xn.T @ Yn
            try:
                U, _, Vh = torch.linalg.svd(H)
            except RuntimeError:
                continue
            R = Vh.T @ U.T
            if torch.det(R).item() < 0:
                Vh[-1, :] *= -1.0
                R = Vh.T @ U.T
            scale = sy / sx
            E_out[b] = (expected[b] - mx) @ R * scale + my
        return E_out

    def _topology_filter(
        self,
        expected: torch.Tensor,
        matched_obs: torch.Tensor,
        match_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Filter assignments using local graph consistency costs."""
        if self.neigh_idx.shape[1] == 0:
            return match_mask
        eps = 1e-6
        neigh = self.neigh_idx  # (N, K)

        # Gather neighbor coordinates: (B, N, K, 2)
        exp_neigh = expected[:, neigh, :]
        obs_neigh = matched_obs[:, neigh, :]

        # Edge vectors from node i to its k neighbors.
        ve = exp_neigh - expected.unsqueeze(2)
        vo = obs_neigh - matched_obs.unsqueeze(2)
        ne = torch.linalg.norm(ve, dim=-1).clamp(min=eps)
        no = torch.linalg.norm(vo, dim=-1).clamp(min=eps)

        len_ratio = torch.abs(no / ne - 1.0) / max(self.topo_len_tol, eps)
        cos_sim = torch.sum(ve * vo, dim=-1) / (ne * no)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        angle_cost = 1.0 - cos_sim
        edge_cost = len_ratio + self.topo_angle_weight * angle_cost

        # Valid edges need both source and neighbor matched.
        src_valid = match_mask.unsqueeze(2)  # (B, N, 1)
        neigh_valid = match_mask[:, neigh]  # (B, N, K)
        valid_edge = src_valid & neigh_valid

        valid_count = valid_edge.float().sum(dim=2)  # (B, N)
        cost_sum = (edge_cost * valid_edge.float()).sum(dim=2)
        topo_cost = cost_sum / valid_count.clamp(min=1.0)

        bad = (valid_count >= float(self.topo_min_neighbors)) & (
            topo_cost > self.topo_cost_thresh
        )
        return match_mask & (~bad)

    def _icp_step(self, coeffs: torch.Tensor) -> torch.Tensor:
        """One ICP iteration for a batch of coefficient vectors.

        Args:
            coeffs: (B, n_terms)

        Returns:
            Updated coeffs: (B, n_terms)
        """
        B = coeffs.shape[0]
        n_terms = self.n_terms

        # 1) Predict expected spots from current coefficients.
        E = self._compute_expected(coeffs)

        # 2) Coarse global alignment (translation/rotation/scale) for robust seeding.
        in_bounds_seed = (
            (E[:, :, 0] >= 0)
            & (E[:, :, 0] <= self.sensor_w)
            & (E[:, :, 1] >= 0)
            & (E[:, :, 1] <= self.sensor_h)
        )
        E_seed = E.clone()
        E_seed[~in_bounds_seed] = 1e8
        d_seed = torch.cdist(E_seed, self.observed)
        min_seed, nn_seed = d_seed.min(dim=2)
        seed_mask = in_bounds_seed & (
            min_seed <= self.coarse_align_dist_factor * self.max_match_dist_um
        )
        matched_obs_seed = self.observed[nn_seed]
        E_aligned = self._apply_similarity_alignment(E, matched_obs_seed, seed_mask)
        del d_seed

        # 3) Assignment on aligned expected spots.
        in_bounds = (
            (E_aligned[:, :, 0] >= 0)
            & (E_aligned[:, :, 0] <= self.sensor_w)
            & (E_aligned[:, :, 1] >= 0)
            & (E_aligned[:, :, 1] <= self.sensor_h)
        )
        E_masked = E_aligned.clone()
        E_masked[~in_bounds] = 1e8

        dists_fwd = torch.cdist(E_masked, self.observed)
        ratio_ok = torch.ones((B, self.n_sub), dtype=torch.bool, device=self.device)
        if self.nn_ratio_threshold is not None and self.n_obs >= 2:
            nn_vals, nn_idx = torch.topk(dists_fwd, k=2, dim=2, largest=False)
            min_dists_fwd = nn_vals[:, :, 0]
            nn_fwd = nn_idx[:, :, 0]
            second_dists = nn_vals[:, :, 1]
            ratio_ok = min_dists_fwd <= self.nn_ratio_threshold * (second_dists + 1e-6)
        else:
            min_dists_fwd, nn_fwd = dists_fwd.min(dim=2)

        dists_bwd = dists_fwd.transpose(1, 2)
        _, nn_bwd = dists_bwd.min(dim=2)
        del dists_fwd, dists_bwd

        reciprocal_check = torch.gather(nn_bwd, 1, nn_fwd)
        arange_sub = (
            torch.arange(self.n_sub, device=self.device).unsqueeze(0).expand(B, -1)
        )
        within_dist = (min_dists_fwd <= self.max_match_dist_um) & ratio_ok
        is_reciprocal = (reciprocal_check == arange_sub) & in_bounds & within_dist

        n_recip = is_reciprocal.float().sum(dim=1)
        if self.allow_forward_fallback:
            use_reciprocal = n_recip >= n_terms
            match_mask = torch.where(
                use_reciprocal.unsqueeze(1).expand(-1, self.n_sub),
                is_reciprocal,
                in_bounds & within_dist,
            )
        else:
            match_mask = is_reciprocal

        matched_obs = self.observed[nn_fwd]  # (B, N_sub, 2)

        # 4) Topology-consistency filtering on local graph edges.
        topo_mask = self._topology_filter(E_aligned, matched_obs, match_mask)
        n_topo = topo_mask.float().sum(dim=1)
        if self.allow_forward_fallback:
            use_topo = n_topo >= n_terms
            match_mask = torch.where(
                use_topo.unsqueeze(1).expand(-1, self.n_sub),
                topo_mask,
                match_mask,
            )
        else:
            match_mask = topo_mask

        n_matched_raw = match_mask.float().sum(dim=1)
        n_matched = n_matched_raw.clamp(min=1)
        mask_float = match_mask.float()
        if self.huber_delta_um > 0:
            robust_w = torch.clamp(
                self.huber_delta_um / (min_dists_fwd + 1e-6), min=0.1, max=1.0
            )
            ls_weights = mask_float * robust_w
        else:
            ls_weights = mask_float

        # 5) Robust LS update with fixed assignment (one alternation step).
        target_disp = matched_obs - self.ref.unsqueeze(0)
        target_slopes = target_disp / self.focal_um
        target_slopes[~match_mask] = 0.0

        wG_x = ls_weights.unsqueeze(-1) * self.G_x.unsqueeze(0)
        wG_y = ls_weights.unsqueeze(-1) * self.G_y.unsqueeze(0)

        G_x_exp = self.G_x.unsqueeze(0).expand(B, -1, -1)
        G_y_exp = self.G_y.unsqueeze(0).expand(B, -1, -1)

        ATA = torch.bmm(wG_x.transpose(1, 2), G_x_exp) + torch.bmm(
            wG_y.transpose(1, 2), G_y_exp
        )
        if self.reg_eye is not None:
            ATA = ATA + self.reg_eye.unsqueeze(0)

        target_sx = target_slopes[:, :, 0] * ls_weights
        target_sy = target_slopes[:, :, 1] * ls_weights

        ATb = torch.bmm(G_x_exp.transpose(1, 2), target_sx.unsqueeze(-1)) + torch.bmm(
            G_y_exp.transpose(1, 2), target_sy.unsqueeze(-1)
        )

        if self.reg_eye is not None:
            coeffs_new = torch.linalg.solve(ATA, ATb).squeeze(-1)
        else:
            coeffs_new = torch.bmm(torch.linalg.pinv(ATA), ATb).squeeze(-1)

        # Skip update for chains with too few matched spots
        insufficient = (n_matched_raw < n_terms) | (n_matched_raw < self.min_required)
        coeffs_new[insufficient] = coeffs[insufficient]

        return coeffs_new

    def _eval_residuals(self, coeffs: torch.Tensor):
        """Evaluate residuals for a batch of coefficient vectors.

        Uses reciprocal matching, consistent with CPU version.

        Args:
            coeffs: (B, n_terms)

        Returns:
            (B,) residual per chain, (B,) n_matched per chain
        """
        B = coeffs.shape[0]

        E = self._compute_expected(coeffs)

        in_bounds_seed = (
            (E[:, :, 0] >= 0)
            & (E[:, :, 0] <= self.sensor_w)
            & (E[:, :, 1] >= 0)
            & (E[:, :, 1] <= self.sensor_h)
        )
        E_seed = E.clone()
        E_seed[~in_bounds_seed] = 1e8
        d_seed = torch.cdist(E_seed, self.observed)
        min_seed, nn_seed = d_seed.min(dim=2)
        seed_mask = in_bounds_seed & (
            min_seed <= self.coarse_align_dist_factor * self.max_match_dist_um
        )
        matched_obs_seed = self.observed[nn_seed]
        E_aligned = self._apply_similarity_alignment(E, matched_obs_seed, seed_mask)
        del d_seed

        in_bounds = (
            (E_aligned[:, :, 0] >= 0)
            & (E_aligned[:, :, 0] <= self.sensor_w)
            & (E_aligned[:, :, 1] >= 0)
            & (E_aligned[:, :, 1] <= self.sensor_h)
        )
        E_masked = E_aligned.clone()
        E_masked[~in_bounds] = 1e8

        dists = torch.cdist(E_masked, self.observed)
        ratio_ok = torch.ones((B, self.n_sub), dtype=torch.bool, device=self.device)
        if self.nn_ratio_threshold is not None and self.n_obs >= 2:
            nn_vals, nn_idx = torch.topk(dists, k=2, dim=2, largest=False)
            min_dists_fwd = nn_vals[:, :, 0]
            nn_fwd = nn_idx[:, :, 0]
            second_dists = nn_vals[:, :, 1]
            ratio_ok = min_dists_fwd <= self.nn_ratio_threshold * (second_dists + 1e-6)
        else:
            min_dists_fwd, nn_fwd = dists.min(dim=2)

        dists_bwd = dists.transpose(1, 2)
        _, nn_bwd = dists_bwd.min(dim=2)
        del dists, dists_bwd

        # Reciprocal check
        reciprocal_check = torch.gather(nn_bwd, 1, nn_fwd)
        arange_sub = (
            torch.arange(self.n_sub, device=self.device).unsqueeze(0).expand(B, -1)
        )
        within_dist = (min_dists_fwd <= self.max_match_dist_um) & ratio_ok
        is_reciprocal = (reciprocal_check == arange_sub) & in_bounds & within_dist

        n_recip = is_reciprocal.float().sum(dim=1)
        if self.allow_forward_fallback:
            use_recip = n_recip >= self.n_terms
            match_mask = torch.where(
                use_recip.unsqueeze(1).expand(-1, self.n_sub),
                is_reciprocal,
                in_bounds & within_dist,
            )
        else:
            match_mask = is_reciprocal

        matched_obs = self.observed[nn_fwd]
        topo_mask = self._topology_filter(E_aligned, matched_obs, match_mask)
        n_topo = topo_mask.float().sum(dim=1)
        if self.allow_forward_fallback:
            use_topo = n_topo >= self.n_terms
            match_mask = torch.where(
                use_topo.unsqueeze(1).expand(-1, self.n_sub),
                topo_mask,
                match_mask,
            )
        else:
            match_mask = topo_mask

        matched_dists = min_dists_fwd * match_mask.float()
        n_matched_raw = match_mask.float().sum(dim=1)
        n_matched = n_matched_raw.clamp(min=1)
        residual_raw = matched_dists.sum(dim=1) / n_matched

        residual_trimmed = []
        for b in range(B):
            d = min_dists_fwd[b][match_mask[b]]
            if d.numel() == 0:
                residual_trimmed.append(float("inf"))
                continue
            n_keep = max(1, int(np.ceil(self.trim_ratio * d.numel())))
            d_sorted, _ = torch.sort(d)
            residual_trimmed.append(float(d_sorted[:n_keep].mean().item()))
        residual_trimmed = torch.tensor(residual_trimmed, device=self.device)

        too_few = n_matched_raw < self.min_required
        residual_raw[too_few] = float("inf")
        residual_trimmed[too_few] = float("inf")

        return residual_trimmed, residual_raw, n_matched_raw

    def probe_start(self, coeffs0: np.ndarray, n_steps: int = 3) -> Dict[str, Any]:
        """Run a short ICP probe from one start and score it."""
        c0 = np.asarray(coeffs0, dtype=np.float32).reshape(-1)
        if c0.size != self.n_terms:
            c0 = np.zeros(self.n_terms, dtype=np.float32)
        c = torch.tensor(c0[None, :], dtype=torch.float32, device=self.device)
        for _ in range(max(1, int(n_steps))):
            c = self._icp_step(c)
        residual_trimmed, residual_raw, n_matched = self._eval_residuals(c)
        return {
            "coeffs": c[0].detach().cpu().numpy(),
            "residual_trimmed": float(residual_trimmed[0].item()),
            "residual_raw": float(residual_raw[0].item()),
            "n_matched": int(n_matched[0].item()),
        }

    def run(
        self,
        n_starts: int = 50,
        search_bound: float = 2.0,
        seed: int = 42,
        baseline_coeffs: Optional[np.ndarray] = None,
        fixed_starts: Optional[list[np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Run mini-batched multi-start ICP.

        Splits starts into mini-batches sized to fit in GPU memory.

        Args:
            n_starts: Number of random starting points.
            search_bound: Bound for random initial coefficients.
            seed: Random seed.
            baseline_coeffs: Optional baseline reconstruction result for warm start.

        Returns:
            Dict with coeffs, residual, n_matched.
        """
        torch.manual_seed(seed)

        n_terms = self.n_terms
        n_starts = max(int(n_starts), 1)

        # Initialize ALL starting points on CPU to save GPU memory
        all_coeffs = torch.empty(n_starts, n_terms, dtype=torch.float32)
        idx = 0

        if fixed_starts:
            for c in fixed_starts:
                if idx >= n_starts:
                    break
                c_arr = np.asarray(c, dtype=np.float32).reshape(-1)
                if c_arr.size != n_terms:
                    continue
                all_coeffs[idx] = torch.tensor(c_arr, dtype=torch.float32)
                idx += 1
        elif baseline_coeffs is not None:
            all_coeffs[0] = torch.tensor(baseline_coeffs, dtype=torch.float32)
            idx = 1

        if idx < n_starts:
            all_coeffs[idx] = 0.0
            idx += 1

        if idx < n_starts:
            all_coeffs[idx:] = torch.empty(
                n_starts - idx, n_terms, dtype=torch.float32
            ).uniform_(-search_bound, search_bound)

        # Determine mini-batch size based on available VRAM
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            vram_gb = props.total_memory / 1e9
        else:
            vram_gb = 8.0  # conservative for CPU
        mini_batch = _estimate_batch_size(self.n_sub, self.n_obs, vram_gb)
        if self.icp_batch_cap > 0:
            mini_batch = min(mini_batch, self.icp_batch_cap)
        mini_batch = min(mini_batch, n_starts)

        # Process in mini-batches
        best_coeffs = None
        best_residual_trimmed = float("inf")
        best_residual_raw = float("inf")
        best_n_matched = 0

        for start in range(0, n_starts, mini_batch):
            end = min(start + mini_batch, n_starts)
            batch_coeffs = all_coeffs[start:end].to(self.device)

            # Run ICP iterations
            for _ in range(self.n_icp_iter):
                batch_coeffs = self._icp_step(batch_coeffs)

            # Evaluate final residuals
            residuals_trimmed, residuals_raw, n_matched = self._eval_residuals(
                batch_coeffs
            )

            # Find best in this batch
            batch_best_idx = residuals_trimmed.argmin().item()
            batch_best_residual = residuals_trimmed[batch_best_idx].item()

            if batch_best_residual < best_residual_trimmed:
                best_residual_trimmed = batch_best_residual
                best_residual_raw = residuals_raw[batch_best_idx].item()
                best_coeffs = batch_coeffs[batch_best_idx].cpu().numpy()
                best_n_matched = int(n_matched[batch_best_idx].item())

            # Free GPU memory between batches
            del batch_coeffs, residuals_trimmed, residuals_raw, n_matched
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return {
            "coeffs": best_coeffs
            if best_coeffs is not None
            else np.zeros(self.n_terms, dtype=np.float32),
            "residual_trimmed": best_residual_trimmed,
            "residual_raw": best_residual_raw,
            "n_matched": best_n_matched,
        }


class BatchedPSO:
    """GPU PSO fallback with a sampled bidirectional distance objective.

    This is used only when ICP fails, to provide a global-search route without
    oracle indices. The objective uses fixed random samples of expected and
    observed spots for tractable CUDA memory usage.
    """

    def __init__(
        self,
        observed: np.ndarray,
        lenslet: LensletArray,
        cfg: Dict[str, Any],
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        asm_cfg = cfg.get("asm", {})
        zer_cfg = cfg.get("zernike", {})
        self.max_order = int(zer_cfg.get("order", 3))
        self.n_terms = num_zernike_terms(self.max_order)
        self.grid_size = int(asm_cfg.get("grid_size", 128))

        self.n_particles = int(asm_cfg.get("pso_particles", 24))
        self.n_iter = int(asm_cfg.get("pso_iters", 30))
        self.w_start = float(asm_cfg.get("pso_w_start", 0.9))
        self.w_end = float(asm_cfg.get("pso_w_end", 0.4))
        self.c1 = float(asm_cfg.get("pso_c1", 1.8))
        self.c2 = float(asm_cfg.get("pso_c2", 1.8))
        self.patience = int(asm_cfg.get("pso_patience", 8))
        self.eps_obj = float(asm_cfg.get("pso_eps_obj", 1e-4))
        self.lambda_reg = float(asm_cfg.get("lambda_reg", 1e-3))
        self.lambda_out = float(asm_cfg.get("pso_lambda_out", 50.0))

        self.pitch_um = float(lenslet.pitch_um)
        self.focal_um = float(lenslet.focal_um)
        self.sensor_w = float(lenslet.sensor_width_um)
        self.sensor_h = float(lenslet.sensor_height_um)

        G_t = _get_cached_g_tensor(lenslet, self.max_order, self.grid_size, device)
        ref_t = torch.tensor(
            lenslet.reference_positions(), dtype=torch.float32, device=device
        )
        n_sub = ref_t.shape[0]
        n_obs = int(len(observed))
        obs_t = torch.tensor(
            observed.astype(np.float32), dtype=torch.float32, device=device
        )

        # Fixed random subset for objective stability.
        rng = np.random.RandomState(seed)
        k_ref = min(int(asm_cfg.get("pso_sample_ref", 1024)), n_sub)
        k_obs = min(int(asm_cfg.get("pso_sample_obs", 1024)), n_obs)
        ref_idx = (
            np.arange(n_sub, dtype=int)
            if k_ref >= n_sub
            else rng.choice(n_sub, size=k_ref, replace=False)
        )
        obs_idx = (
            np.arange(n_obs, dtype=int)
            if k_obs >= n_obs
            else rng.choice(n_obs, size=k_obs, replace=False)
        )

        self.ref = ref_t[torch.tensor(ref_idx, dtype=torch.long, device=device)]
        self.observed = obs_t[torch.tensor(obs_idx, dtype=torch.long, device=device)]
        self.k_ref = int(self.ref.shape[0])
        self.k_obs = int(self.observed.shape[0])

        ridx_t = torch.tensor(ref_idx, dtype=torch.long, device=device)
        self.G_x = G_t[ridx_t, :]
        self.G_y = G_t[n_sub + ridx_t, :]

    def _objective_batch(self, coeffs_batch: torch.Tensor) -> torch.Tensor:
        """Sampled symmetric nearest-distance objective + penalties."""
        B = coeffs_batch.shape[0]
        if self.k_ref == 0 or self.k_obs == 0:
            return torch.full((B,), 1e6, dtype=torch.float32, device=self.device)

        slopes_x = coeffs_batch @ self.G_x.T  # (B, k_ref)
        slopes_y = coeffs_batch @ self.G_y.T  # (B, k_ref)

        E_x = self.ref[:, 0].unsqueeze(0) + self.focal_um * slopes_x
        E_y = self.ref[:, 1].unsqueeze(0) + self.focal_um * slopes_y
        E = torch.stack([E_x, E_y], dim=-1)  # (B, k_ref, 2)

        in_bounds = (
            (E[:, :, 0] >= 0)
            & (E[:, :, 0] <= self.sensor_w)
            & (E[:, :, 1] >= 0)
            & (E[:, :, 1] <= self.sensor_h)
        )
        n_in = in_bounds.float().sum(dim=1).clamp(min=1.0)

        E_masked = E.clone()
        E_masked[~in_bounds] = 1e8

        d = torch.cdist(E_masked, self.observed)  # (B, k_ref, k_obs)
        min_fwd = d.min(dim=2).values  # (B, k_ref)
        min_bwd = d.min(dim=1).values  # (B, k_obs)

        fwd_mean = (min_fwd * in_bounds.float()).sum(dim=1) / n_in
        bwd_mean = min_bwd.mean(dim=1)
        sym_dist = 0.5 * (fwd_mean + bwd_mean)

        coverage_penalty = (1.0 - (n_in / float(self.k_ref))).clamp(min=0.0)
        reg = torch.mean(coeffs_batch * coeffs_batch, dim=1)

        return sym_dist + self.lambda_out * coverage_penalty + self.lambda_reg * reg

    def run(
        self,
        search_bound: float,
        seed: int,
        init_coeffs: Optional[list[np.ndarray]] = None,
    ) -> Dict[str, Any]:
        torch.manual_seed(seed)
        np.random.seed(seed)

        P = self.n_particles
        D = self.n_terms

        pos = torch.empty((P, D), dtype=torch.float32, device=self.device).uniform_(
            -search_bound, search_bound
        )
        vel = torch.empty((P, D), dtype=torch.float32, device=self.device).uniform_(
            -0.1 * search_bound, 0.1 * search_bound
        )

        # Inject deterministic starts.
        insert_idx = 0
        if init_coeffs:
            for c in init_coeffs:
                if c is None:
                    continue
                if insert_idx >= P:
                    break
                c_t = torch.tensor(c, dtype=torch.float32, device=self.device)
                if c_t.numel() == D:
                    pos[insert_idx] = torch.clamp(c_t, -search_bound, search_bound)
                    vel[insert_idx] = 0.0
                    insert_idx += 1

        pbest_pos = pos.clone()
        pbest_val = self._objective_batch(pos)
        g_idx = torch.argmin(pbest_val)
        gbest_pos = pbest_pos[g_idx].clone()
        gbest_val = float(pbest_val[g_idx].item())

        no_improve = 0

        for t in range(self.n_iter):
            w = self.w_start - (self.w_start - self.w_end) * float(t) / max(
                self.n_iter, 1
            )
            r1 = torch.rand((P, D), device=self.device)
            r2 = torch.rand((P, D), device=self.device)
            vel = (
                w * vel
                + self.c1 * r1 * (pbest_pos - pos)
                + self.c2 * r2 * (gbest_pos - pos)
            )
            pos = torch.clamp(pos + vel, -search_bound, search_bound)

            vals = self._objective_batch(pos)
            improved = vals < pbest_val
            pbest_pos[improved] = pos[improved]
            pbest_val[improved] = vals[improved]

            cur_idx = torch.argmin(pbest_val)
            cur_best = float(pbest_val[cur_idx].item())
            if cur_best < gbest_val - self.eps_obj:
                gbest_val = cur_best
                gbest_pos = pbest_pos[cur_idx].clone()
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.patience:
                break

        return {
            "coeffs": gbest_pos.detach().cpu().numpy(),
            "objective_value": float(gbest_val),
        }


def asm_reconstruct_gpu(
    observed: np.ndarray,
    lenslet: LensletArray,
    cfg: Dict[str, Any],
    seed: int = 42,
    observed_sub_idx: Optional[np.ndarray] = None,
    pv_hint: Optional[float] = None,
    missing_ratio_hint: Optional[float] = None,
) -> Dict[str, Any]:
    """Run ASM reconstruction using Sorting Method + GPU ICP refinement.

    Pipeline (no-oracle mode):
    1. Sorting Match: global spot-subaperture matching via row/column sorting
       (Lee 2005, Zhang 2011). This gives a good initial estimate even for
       very large wavefront aberrations (limited by curvature, not slope).
    2. ICP Refinement: use sorting result as warm start for a few ICP
       iterations to fine-tune the matching and coefficients.
    3. Fallback: baseline warm start + random starts as additional ICP seeds.

    Uses baseline reconstruction as warm start, then multi-start ICP on GPU.
    Same interface as asm_reconstruct().
    """
    zer_cfg = cfg.get("zernike", {})
    asm_cfg = cfg.get("asm", {})

    coeff_bound = zer_cfg.get("coeff_bound", 1.0)
    search_bound = max(coeff_bound * 2, 2.0)
    n_starts = asm_cfg.get("n_starts", 50)
    lambda_reg = asm_cfg.get("lambda_reg", 1e-3)
    trim_ratio = asm_cfg.get("trim_ratio", 0.9)
    grid_size = asm_cfg.get("grid_size", 128)
    n_terms = num_zernike_terms(zer_cfg.get("order", 3))
    enable_pso_fallback = bool(asm_cfg.get("enable_pso_fallback", False))
    enable_sorting = bool(asm_cfg.get("enable_sorting", True))
    pso_first_pv_threshold = float(asm_cfg.get("pso_first_pv_threshold", 3.0))
    pso_first_always = bool(asm_cfg.get("pso_first_always", False))
    pso_first_max_missing_ratio = float(
        asm_cfg.get("pso_first_max_missing_ratio", 0.05)
    )

    # Optional oracle index hint (simulation debug mode).
    # Run the same direct LS path as CPU, but keep tensors and solve on CUDA.
    if observed_sub_idx is not None and len(observed_sub_idx) >= n_terms:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        observed_t = torch.tensor(observed, dtype=torch.float32, device=device)
        ref_t = torch.tensor(
            lenslet.reference_positions(), dtype=torch.float32, device=device
        )
        sub_idx_t = torch.tensor(
            observed_sub_idx.astype(int), dtype=torch.long, device=device
        )

        G_t = _get_cached_g_tensor(lenslet, zer_cfg.get("order", 3), grid_size, device)
        n_sub = lenslet.n_subapertures
        focal_um = float(lenslet.focal_um)

        target_disp = observed_t - ref_t[sub_idx_t]
        target_slopes = target_disp / focal_um
        G_sub_x = G_t[sub_idx_t, :]
        G_sub_y = G_t[n_sub + sub_idx_t, :]
        A = torch.cat([G_sub_x, G_sub_y], dim=0)
        b = torch.cat([target_slopes[:, 0], target_slopes[:, 1]], dim=0)

        if lambda_reg > 0:
            ATA = A.T @ A + (lambda_reg * torch.eye(n_terms, device=device))
            ATb = A.T @ b
            try:
                coeffs_t = torch.linalg.solve(ATA, ATb)
            except RuntimeError:
                coeffs_t = torch.linalg.lstsq(A, b.unsqueeze(1)).solution.squeeze(1)
        else:
            coeffs_t = torch.linalg.lstsq(A, b.unsqueeze(1)).solution.squeeze(1)

        slopes_vec = G_t @ coeffs_t
        slopes_x = slopes_vec[:n_sub]
        slopes_y = slopes_vec[n_sub:]
        E_x = ref_t[:, 0] + focal_um * slopes_x
        E_y = ref_t[:, 1] + focal_um * slopes_y
        E = torch.stack([E_x, E_y], dim=1)
        d = torch.norm(E[sub_idx_t] - observed_t, dim=1)
        residual_raw = float(d.mean().item()) if d.numel() else float("inf")
        if d.numel():
            n_keep = max(1, int(np.ceil(trim_ratio * d.numel())))
            d_sorted, _ = torch.sort(d)
            residual_trimmed = float(d_sorted[:n_keep].mean().item())
        else:
            residual_trimmed = float("inf")

        success_threshold = lenslet.pitch_um * 0.5
        coeffs = coeffs_t.detach().cpu().numpy()
        success = residual_trimmed < success_threshold
        return {
            "coeffs": coeffs,
            "success": success,
            "objective_value": residual_trimmed,
            "residual_raw": residual_raw,
            "residual_trimmed": residual_trimmed,
            "n_matched": int(len(observed_sub_idx)),
            "n_iterations": 1,
            "solver": "asm_oracle_ls_gpu",
        }

    # --- Non-oracle mode: robust initialization + ICP refinement ---

    enable_chamfer = bool(asm_cfg.get("enable_chamfer", True))

    # Step 1: Sorting Match (global matching, no initial guess needed)
    sorting_coeffs = None
    sorting_result = None
    if enable_sorting:
        from src.recon.sorting_matcher import sorting_match

        sorting_result = sorting_match(observed, lenslet, cfg)
        if sorting_result["success"]:
            sorting_coeffs = sorting_result["coeffs"]

    # Step 2: Baseline warm start (fallback)
    from src.recon.baseline_gpu import baseline_reconstruct_auto

    bl = baseline_reconstruct_auto(observed, lenslet, cfg)
    baseline_coeffs = bl["coeffs"] if bl["success"] else None

    # Step 2a: Neural network warm start (v2 CNN + v3 ResNet ensemble)
    nn_coeffs = None
    enable_nn = bool(asm_cfg.get("enable_nn_warmstart", True))
    if enable_nn:
        import os
        from src.recon.nn_warmstart import NNEnsembleWarmStarter, NNWarmStarter

        sensor_w_um = float(lenslet.sensor_width_um)
        sensor_h_um = float(lenslet.sensor_height_um)
        nn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Try ensemble first (v2 + v3), fall back to single model
        nn_model_paths = asm_cfg.get("nn_model_paths", None)
        if nn_model_paths is None:
            # Default: look for both v2 CNN and v3 ResNet
            nn_model_paths = []
            for p in ["models/nn_warmstart.pt", "models/nn_v3_resnet.pt"]:
                if os.path.exists(p):
                    nn_model_paths.append(p)

        if len(nn_model_paths) >= 2:
            try:
                nn_ens = NNEnsembleWarmStarter(
                    model_paths=nn_model_paths,
                    sensor_w_um=sensor_w_um,
                    sensor_h_um=sensor_h_um,
                    n_terms=n_terms,
                    device=nn_device,
                )
                nn_coeffs = nn_ens.predict(observed)
            except Exception:
                nn_coeffs = None
        elif len(nn_model_paths) == 1:
            try:
                nn_starter = NNWarmStarter(
                    model_path=nn_model_paths[0],
                    sensor_w_um=sensor_w_um,
                    sensor_h_um=sensor_h_um,
                    n_terms=n_terms,
                    device=nn_device,
                )
                nn_coeffs = nn_starter.predict(observed)
            except Exception:
                nn_coeffs = None

    # Step 2b: Chamfer distance optimizer (global, matching-free warm start)
    chamfer_coeffs = None
    if enable_chamfer:
        from src.recon.chamfer_optimizer import ChamferOptimizer

        chamfer_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        chamfer_opt = ChamferOptimizer(observed, lenslet, cfg, device=chamfer_device)

        # Seed Chamfer with any available warm starts
        chamfer_init: list[np.ndarray] = []
        if nn_coeffs is not None:
            chamfer_init.append(np.asarray(nn_coeffs, dtype=np.float32))
        if sorting_coeffs is not None:
            chamfer_init.append(np.asarray(sorting_coeffs, dtype=np.float32))
        if baseline_coeffs is not None:
            chamfer_init.append(np.asarray(baseline_coeffs, dtype=np.float32))

        chamfer_result = chamfer_opt.run(
            seed=seed + 3000,
            init_coeffs=chamfer_init if chamfer_init else None,
        )
        chamfer_coeffs = np.asarray(chamfer_result["coeffs"], dtype=np.float32)

    # Step 3: ICP refinement with initialization arbitration
    icp = BatchedICP(observed, lenslet, cfg)
    init_probe_steps = int(asm_cfg.get("init_probe_steps", 3))
    init_topk = int(asm_cfg.get("init_topk", 2))
    init_match_bonus = float(asm_cfg.get("init_match_bonus", 0.25))

    def _add_unique_start(
        starts: list[tuple[str, np.ndarray]], name: str, coeffs: Optional[np.ndarray]
    ) -> None:
        if coeffs is None:
            return
        c = np.asarray(coeffs, dtype=np.float32).reshape(-1)
        if c.size != n_terms:
            return
        for _, exist in starts:
            if np.allclose(c, exist, rtol=1e-4, atol=1e-6):
                return
        starts.append((name, c))

    candidate_starts: list[tuple[str, np.ndarray]] = []
    # NN goes first (highest priority — best warm start for high PV)
    _add_unique_start(candidate_starts, "nn", nn_coeffs)
    # Chamfer second (matching-free global optimizer)
    _add_unique_start(candidate_starts, "chamfer", chamfer_coeffs)
    _add_unique_start(candidate_starts, "sorting", sorting_coeffs)
    _add_unique_start(candidate_starts, "baseline", baseline_coeffs)
    _add_unique_start(candidate_starts, "zero", np.zeros(n_terms, dtype=np.float32))

    # High-PV route: run global search first, then use PSO output as a strong start.
    pso_ran_first = False
    if enable_pso_fallback:
        missing_ok = (
            missing_ratio_hint is None
            or float(missing_ratio_hint) <= pso_first_max_missing_ratio
        )
        need_pso_first = pso_first_always or (
            missing_ok
            and pv_hint is not None
            and float(pv_hint) >= pso_first_pv_threshold
        )
        if need_pso_first:
            pso_ran_first = True
            pso = BatchedPSO(
                observed, lenslet, cfg, device=icp.device, seed=seed + 4000
            )
            pso_init: list[np.ndarray] = []
            if nn_coeffs is not None:
                pso_init.append(np.asarray(nn_coeffs, dtype=np.float32))
            if chamfer_coeffs is not None:
                pso_init.append(np.asarray(chamfer_coeffs, dtype=np.float32))
            if sorting_coeffs is not None:
                pso_init.append(np.asarray(sorting_coeffs, dtype=np.float32))
            if baseline_coeffs is not None:
                pso_init.append(np.asarray(baseline_coeffs, dtype=np.float32))
            pso_init.append(np.zeros(n_terms, dtype=np.float32))
            pso_result = pso.run(
                search_bound=search_bound,
                seed=seed + 4500,
                init_coeffs=pso_init,
            )
            _add_unique_start(
                candidate_starts,
                "pso",
                np.asarray(pso_result["coeffs"], dtype=np.float32),
            )

    probe_rows: list[tuple[float, str, np.ndarray, Dict[str, Any]]] = []
    obs_count = max(float(len(observed)), 1.0)
    for name, c0 in candidate_starts:
        probe = icp.probe_start(c0, n_steps=init_probe_steps)
        match_ratio = float(probe["n_matched"]) / obs_count
        score = (
            probe["residual_trimmed"] / max(float(lenslet.pitch_um), 1.0)
            - init_match_bonus * match_ratio
        )
        if not np.isfinite(score):
            score = 1e9
        probe_rows.append((score, name, c0, probe))

    probe_rows.sort(key=lambda x: x[0])
    fixed_starts: list[np.ndarray] = []
    fixed_names: list[str] = []
    for _, name, c0, _ in probe_rows[: max(1, init_topk)]:
        fixed_starts.append(c0)
        fixed_names.append(name)

    icp_n_starts = max(n_starts, len(fixed_starts) + 2)
    result = icp.run(
        n_starts=icp_n_starts,
        search_bound=search_bound,
        seed=seed,
        baseline_coeffs=None,
        fixed_starts=fixed_starts,
    )

    if fixed_names and fixed_names[0] == "pso":
        solver_tag = "asm_pso_icp_gpu"
    elif fixed_names and fixed_names[0] == "nn":
        solver_tag = "asm_nn_icp_gpu"
    elif fixed_names and fixed_names[0] == "chamfer":
        solver_tag = "asm_chamfer_icp_gpu"
    elif fixed_names and fixed_names[0] == "sorting":
        solver_tag = "asm_sorting_icp_gpu"
    elif fixed_names and fixed_names[0] == "baseline":
        solver_tag = "asm_baseline_icp_gpu"
    else:
        solver_tag = "asm_icp_gpu"

    # If sorting gave a direct result and ICP didn't improve, use sorting result
    if sorting_result is not None and sorting_result["success"]:
        if result["residual_trimmed"] > sorting_result["residual_trimmed"]:
            result = {
                "coeffs": sorting_result["coeffs"],
                "residual_trimmed": sorting_result["residual_trimmed"],
                "residual_raw": sorting_result["residual_raw"],
                "n_matched": sorting_result["n_matched"],
            }
            solver_tag = "asm_sorting_gpu"

    # Optional global-search fallback for non-oracle mode.
    if enable_pso_fallback and not pso_ran_first:
        pso_always_refine = bool(asm_cfg.get("pso_always_refine", False))
        trigger_match_ratio = float(asm_cfg.get("pso_trigger_match_ratio", 0.65))
        matched_ratio = float(result["n_matched"]) / max(float(len(observed)), 1.0)
        # Trigger fallback on clear ICP failure patterns.
        should_fallback = (
            not np.isfinite(result["residual_trimmed"])
            or result["n_matched"] < max(3 * n_terms, int(0.1 * max(len(observed), 1)))
            or matched_ratio < trigger_match_ratio
        )
        if should_fallback or pso_always_refine:
            pso = BatchedPSO(
                observed, lenslet, cfg, device=icp.device, seed=seed + 5000
            )
            init = []
            for c0 in fixed_starts:
                init.append(np.asarray(c0, dtype=np.float32))
            if nn_coeffs is not None:
                init.append(np.asarray(nn_coeffs, dtype=np.float32))
            if chamfer_coeffs is not None:
                init.append(np.asarray(chamfer_coeffs, dtype=np.float32))
            if sorting_coeffs is not None:
                init.append(np.asarray(sorting_coeffs, dtype=np.float32))
            if baseline_coeffs is not None:
                init.append(np.asarray(baseline_coeffs, dtype=np.float32))
            if result.get("coeffs") is not None:
                init.append(np.asarray(result["coeffs"], dtype=np.float32))

            pso_result = pso.run(
                search_bound=search_bound,
                seed=seed + 6000,
                init_coeffs=init,
            )

            # Refine PSO best with one ICP chain to recover full-set residual metrics.
            refined = icp.run(
                n_starts=1,
                search_bound=search_bound,
                seed=seed + 7000,
                baseline_coeffs=pso_result["coeffs"],
            )
            # Compare by the same sampled objective used by PSO.
            with torch.no_grad():
                icp_obj = float(
                    pso._objective_batch(
                        torch.tensor(
                            np.asarray(result["coeffs"], dtype=np.float32)[None, :],
                            dtype=torch.float32,
                            device=icp.device,
                        )
                    )[0].item()
                )
                refined_obj = float(
                    pso._objective_batch(
                        torch.tensor(
                            np.asarray(refined["coeffs"], dtype=np.float32)[None, :],
                            dtype=torch.float32,
                            device=icp.device,
                        )
                    )[0].item()
                )

            if refined_obj < icp_obj:
                result = refined
                solver_tag = "asm_icp_pso_gpu"

    # Success threshold: mean residual < half pitch
    success_threshold = lenslet.pitch_um * 0.5
    success = result["residual_trimmed"] < success_threshold

    return {
        "coeffs": result["coeffs"],
        "success": success,
        "objective_value": result["residual_trimmed"],
        "residual_raw": result["residual_raw"],
        "residual_trimmed": result["residual_trimmed"],
        "n_matched": result["n_matched"],
        "n_iterations": asm_cfg.get("n_icp_iter", 30),
        "solver": solver_tag,
    }
