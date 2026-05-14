"""
Hierarchical SplineLinear — multi-scale control points for high-dim tasks.

Problem: Single-scale SplineLinear with 16 control points fails on 256-dim
inputs because the interpolation can't capture fine-grained structure.

Solution: Hierarchical decomposition — coarse control points cover global
structure, fine control points cover local patches.

Usage:
    layer = HierarchicalSplineLinear(256, 128, coarse_pts=16, fine_pts=8, patch_size=32)
    # 16 coarse + (256/32)*(128/32)*8 = 16 + 64 fine = 80 total control points
    # vs 256*128 = 32,768 dense params → 409:1 compression
"""

import math
import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from .spline import EisensteinLattice, SplineLinear


class HierarchicalSplineLinear(nn.Module):
    """
    Two-level spline: coarse grid for global structure + fine patches for local detail.
    
    Weight at position (i, j):
        W[i,j] = α * interpolate_coarse(i,j) + (1-α) * interpolate_fine(patch_i, patch_j, local_i, local_j)
    
    α is learnable (starts at 0.5, lets the model decide global vs local balance).
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        coarse_pts: int = 16,
        fine_pts: int = 8,
        patch_size: int = 32,
        alpha_init: float = 0.5,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.patch_size = patch_size
        
        # Coarse level: covers entire weight matrix
        self.coarse_lattice = EisensteinLattice(coarse_pts)
        self.coarse_values = nn.Parameter(torch.randn(coarse_pts) * 0.01)
        
        # Fine level: one set of control points per patch
        n_in_patches = math.ceil(in_features / patch_size)
        n_out_patches = math.ceil(out_features / patch_size)
        self.n_in_patches = n_in_patches
        self.n_out_patches = n_out_patches
        self.total_fine_patches = n_in_patches * n_out_patches
        
        # Fine control points: (n_patches, fine_pts)
        self.fine_lattice = EisensteinLattice(fine_pts)
        self.fine_values = nn.Parameter(
            torch.randn(self.total_fine_patches, fine_pts) * 0.01
        )
        
        # Learnable blend factor
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Precompute positions
        self._register_positions()
    
    def _register_positions(self):
        """Precompute grid positions for materialization."""
        in_coords = torch.linspace(0, self.in_features - 1, self.in_features)
        out_coords = torch.linspace(0, self.out_features - 1, self.out_features)
        self.register_buffer('_in_coords', in_coords)
        self.register_buffer('_out_coords', out_coords)
    
    def _interpolate_idw(self, query_pos: torch.Tensor, control_pos: torch.Tensor,
                         values: torch.Tensor) -> torch.Tensor:
        """IDW interpolation. query_pos: (Q, 2), control_pos: (N, 2), values: (N,) or (B, N)."""
        diffs = query_pos.unsqueeze(1) - control_pos.unsqueeze(0)  # (Q, N, 2)
        dists_sq = (diffs ** 2).sum(dim=2).clamp(min=1e-8)  # (Q, N)
        inv_dist_sq = 1.0 / dists_sq
        weights = inv_dist_sq / inv_dist_sq.sum(dim=1, keepdim=True)  # (Q, N)
        
        if values.dim() == 1:
            return (weights * values.unsqueeze(0)).sum(dim=1)  # (Q,)
        else:
            # Batched: values is (B, N), weights is (Q, N)
            return (weights.unsqueeze(0) * values.unsqueeze(1)).sum(dim=2)  # (B, Q)
    
    def _materialize_weights(self) -> torch.Tensor:
        """Build full weight matrix from hierarchical control points."""
        # Weight positions: (out_features, in_features, 2)
        grid = torch.stack(torch.meshgrid(
            self._out_coords, self._in_coords, indexing='ij'
        ), dim=-1)
        flat_pos = grid.reshape(-1, 2)  # (out*in, 2)
        
        # Coarse interpolation
        coarse_pos = self.coarse_lattice.positions().to(flat_pos.device)
        coarse_w = self._interpolate_idw(flat_pos, coarse_pos, self.coarse_values)
        
        # Fine interpolation — per-patch
        fine_pos_template = self.fine_lattice.positions().to(flat_pos.device)
        ps = self.patch_size
        
        fine_w = torch.zeros(flat_pos.shape[0], device=flat_pos.device)
        
        for pi in range(self.n_in_patches):
            for pj in range(self.n_out_patches):
                patch_idx = pj * self.n_in_patches + pi
                
                # Global coords for this patch
                in_start = pi * ps
                in_end = min(in_start + ps, self.in_features)
                out_start = pj * ps
                out_end = min(out_start + ps, self.out_features)
                
                # Find elements in this patch
                mask = (
                    (flat_pos[:, 0] >= out_start) & (flat_pos[:, 0] < out_end) &
                    (flat_pos[:, 1] >= in_start) & (flat_pos[:, 1] < in_end)
                )
                
                if mask.sum() == 0:
                    continue
                
                # Local positions within patch, normalized to patch_size
                local_pos = flat_pos[mask].clone()
                local_pos[:, 0] = (local_pos[:, 0] - out_start) / max(out_end - out_start - 1, 1) * (ps - 1)
                local_pos[:, 1] = (local_pos[:, 1] - in_start) / max(in_end - in_start - 1, 1) * (ps - 1)
                
                patch_values = self.fine_values[patch_idx]  # (fine_pts,)
                patch_w = self._interpolate_idw(local_pos, fine_pos_template, patch_values)
                fine_w[mask] = patch_w
        
        # Blend
        alpha = torch.sigmoid(self.alpha)  # Constrain to [0, 1]
        W = alpha * coarse_w + (1 - alpha) * fine_w
        
        return W.reshape(self.out_features, self.in_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._materialize_weights()
        out = x @ W.T
        if self.bias is not None:
            out += self.bias
        return out
    
    def num_control_params(self) -> int:
        count = self.coarse_values.numel() + self.fine_values.numel() + 1  # +1 for alpha
        if self.bias is not None:
            count += self.bias.numel()
        return count
    
    def num_equivalent_dense_params(self) -> int:
        count = self.in_features * self.out_features
        if self.bias is not None:
            count += self.out_features
        return count
    
    def compression_ratio(self) -> float:
        return self.num_equivalent_dense_params() / max(self.num_control_params(), 1)


class HierarchicalSplineClassifier(nn.Module):
    """Classifier using hierarchical spline for high-dim tasks."""
    
    def __init__(self, input_dim: int, hidden: int, num_classes: int,
                 coarse_pts: int = 16, fine_pts: int = 8, patch_size: int = 32):
        super().__init__()
        self.W_query = HierarchicalSplineLinear(
            input_dim, hidden, coarse_pts=coarse_pts,
            fine_pts=fine_pts, patch_size=patch_size
        )
        self.W_value = HierarchicalSplineLinear(
            hidden, hidden, coarse_pts=coarse_pts,
            fine_pts=fine_pts, patch_size=min(patch_size, hidden)
        )
        self.out_head = nn.Linear(hidden, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.relu(self.W_query(x))
        x = self.dropout(x)
        x = self.relu(self.W_value(x))
        return self.out_head(x)


def inject_hierarchical_spline(
    model: nn.Module,
    coarse_pts: int = 16,
    fine_pts: int = 8,
    patch_size: int = 32,
    target_modules: Optional[List[str]] = None,
) -> dict:
    """Replace nn.Linear with HierarchicalSplineLinear."""
    injection_map = {}
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if target_modules and not any(t in name for t in target_modules):
                continue
            hsl = HierarchicalSplineLinear(
                module.in_features, module.out_features,
                coarse_pts=coarse_pts, fine_pts=fine_pts,
                patch_size=patch_size, bias=module.bias is not None,
            )
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], hsl)
            injection_map[name] = "hierarchical-spline"
    return injection_map
