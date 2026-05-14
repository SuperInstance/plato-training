"""
Tensor-Spline: weights parameterized by Eisenstein lattice control points.

Standard: W[i][j] = independent float (262K params for 512×512)
Tensor-Spline: W[i][j] = interpolate(control_points, position(i,j)) (4K with 16 pts)

The Eisenstein lattice ω = e^(2πi/3) gives the densest 2D packing.
Control points live on this lattice. Weights are interpolated from them.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple


class EisensteinLattice:
    """
    Hexagonal control point layout using Eisenstein integer coordinates.
    
    Lattice basis: e1 = (1, 0), e2 = (0.5, √3/2)
    Points: a*e1 + b*e2 for integer a,b
    """
    
    def __init__(self, n_points: int, width: float, height: float):
        """
        Place n_points on a hexagonal grid covering [0,width] × [0,height].
        """
        self.n_points = n_points
        self.width = width
        self.height = height
        
        # Generate hexagonal grid positions
        # Spacing chosen so ~n_points fill the region
        area = width * height
        spacing = math.sqrt(area / (n_points * math.sqrt(3) / 2))
        
        positions = []
        # Hex grid: alternate rows offset by spacing/2
        row = 0
        y = 0.0
        while y <= height and len(positions) < n_points:
            x_offset = spacing / 2 if row % 2 else 0.0
            x = x_offset
            while x <= width and len(positions) < n_points:
                positions.append([x, y])
                x += spacing
            y += spacing * math.sqrt(3) / 2
            row += 1
        
        # If we didn't get enough, add more
        while len(positions) < n_points:
            positions.append([width / 2, height / 2])
        
        self._positions = torch.tensor(positions[:n_points], dtype=torch.float32)
    
    def positions(self) -> torch.Tensor:
        """Return (n_points, 2) tensor of control point positions."""
        return self._positions
    
    def nearest_k(self, point: torch.Tensor, k: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find k nearest control points to a query point.
        
        Args:
            point: (2,) query position
            k: number of neighbors
        
        Returns:
            indices: (k,) indices of nearest control points
            distances: (k,) Euclidean distances
        """
        diffs = self._positions - point.unsqueeze(0)  # (n, 2)
        dists = (diffs ** 2).sum(dim=1).sqrt()  # (n,)
        topk = torch.topk(dists, min(k, len(dists)), largest=False)
        return topk.indices, topk.values


class SplineLinear(nn.Module):
    """
    Linear layer where weights are interpolated from Eisenstein lattice control points.
    
    Instead of learning a (out_features × in_features) weight matrix directly,
    we learn N control point values and interpolate the weight matrix from them.
    
    With N=16 control points for a 512×512 layer:
    - Standard: 262,144 parameters
    - Spline: 16 parameters (16,384:1 compression)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_control_points: int = 16,
        basis: str = "eisenstein",
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_control_points = n_control_points
        self.basis = basis
        
        # Create lattice covering the weight matrix dimensions
        self.lattice = EisensteinLattice(n_control_points, in_features, out_features)
        
        # Control point values — these are the ONLY learnable parameters for weights
        self.control_values = nn.Parameter(torch.randn(n_control_points) * 0.01)
        
        # Bias (optional)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Precompute weight matrix positions for interpolation
        # Each weight W[i,j] has a "position" in (input_dim, output_dim) space
        self._register_weight_positions()
    
    def _register_weight_positions(self):
        """Precompute positions for each weight element."""
        # Create grid of (in_idx, out_idx) positions, normalized to [0, lattice_size]
        in_coords = torch.linspace(0, self.in_features - 1, self.in_features)
        out_coords = torch.linspace(0, self.out_features - 1, self.out_features)
        
        # Store as buffers (not parameters, but saved with model)
        self.register_buffer('_in_coords', in_coords)
        self.register_buffer('_out_coords', out_coords)
    
    def _materialize_weights(self) -> torch.Tensor:
        """
        Interpolate control point values to create full weight matrix.
        
        Returns: (out_features, in_features) weight tensor
        """
        lattice_pos = self.lattice.positions().to(self.control_values.device)  # (N, 2)
        
        # For efficiency, batch the interpolation
        # Weight positions: (out_features, in_features, 2)
        weight_positions = torch.stack(torch.meshgrid(
            self._out_coords, self._in_coords, indexing='ij'
        ), dim=-1)  # (out, in, 2)
        
        flat_pos = weight_positions.reshape(-1, 2)  # (out*in, 2)
        
        # Interpolate using basis function
        if self.basis == "eisenstein":
            weights = self._interpolate_idw(flat_pos, lattice_pos)
        elif self.basis == "gaussian":
            weights = self._interpolate_gaussian(flat_pos, lattice_pos)
        elif self.basis == "bspline":
            weights = self._interpolate_bspline(flat_pos, lattice_pos)
        else:
            raise ValueError(f"Unknown basis: {self.basis}")
        
        return weights.reshape(self.out_features, self.in_features)
    
    def _interpolate_idw(self, query_pos, control_pos) -> torch.Tensor:
        """Inverse distance weighting — standard Eisenstein interpolation."""
        # query_pos: (Q, 2), control_pos: (N, 2)
        # distances: (Q, N)
        diffs = query_pos.unsqueeze(1) - control_pos.unsqueeze(0)  # (Q, N, 2)
        dists = (diffs ** 2).sum(dim=2).sqrt().clamp(min=1e-8)  # (Q, N)
        
        # IDW: weight = (1/d^2) / sum(1/d^2)
        inv_dist_sq = 1.0 / (dists ** 2)  # (Q, N)
        weights = inv_dist_sq / inv_dist_sq.sum(dim=1, keepdim=True)  # (Q, N)
        
        # Weighted sum of control values
        return (weights * self.control_values.unsqueeze(0)).sum(dim=1)  # (Q,)
    
    def _interpolate_gaussian(self, query_pos, control_pos) -> torch.Tensor:
        """Gaussian RBF interpolation."""
        diffs = query_pos.unsqueeze(1) - control_pos.unsqueeze(0)
        dists_sq = (diffs ** 2).sum(dim=2)  # (Q, N)
        
        # Bandwidth: average nearest-neighbor distance
        sigma_sq = max(self.lattice.width, self.lattice.height) ** 2 / max(self.n_control_points, 1)
        
        gauss = torch.exp(-dists_sq / (2 * sigma_sq))
        weights = gauss / gauss.sum(dim=1, keepdim=True).clamp(min=1e-8)
        
        return (weights * self.control_values.unsqueeze(0)).sum(dim=1)
    
    def _interpolate_bspline(self, query_pos, control_pos) -> torch.Tensor:
        """B-spline-like interpolation (cubic kernel)."""
        diffs = query_pos.unsqueeze(1) - control_pos.unsqueeze(0)
        dists = diffs.norm(dim=2).clamp(min=1e-8)
        
        # Cubic B-spline kernel: (1 - d)^3 for d < 1, else 0
        # Normalize distances to [0, 1] range
        max_dist = dists.max().clamp(min=1e-8)
        norm_dists = dists / max_dist
        
        kernel = (1.0 - norm_dists).clamp(min=0) ** 3
        weights = kernel / kernel.sum(dim=1, keepdim=True).clamp(min=1e-8)
        
        return (weights * self.control_values.unsqueeze(0)).sum(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: materialize weights, then multiply."""
        W = self._materialize_weights()
        output = x @ W.T
        if self.bias is not None:
            output += self.bias
        return output
    
    def num_control_params(self) -> int:
        """Number of trainable parameters (control points + bias)."""
        count = self.control_values.numel()
        if self.bias is not None:
            count += self.bias.numel()
        return count
    
    def num_equivalent_dense_params(self) -> int:
        """Parameters if this were a standard nn.Linear."""
        count = self.in_features * self.out_features
        if self.bias is not None:
            count += self.out_features
        return count
    
    def compression_ratio(self) -> float:
        """Compression: dense_params / spline_params."""
        return self.num_equivalent_dense_params() / max(self.num_control_params(), 1)


def inject_spline(
    model: nn.Module,
    n_control_points: int = 16,
    basis: str = "eisenstein",
    target_modules: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Replace nn.Linear layers with SplineLinear.
    
    Args:
        model: PyTorch model
        n_control_points: control points per layer
        basis: interpolation basis ("eisenstein", "gaussian", "bspline")
        target_modules: module name filter (None = replace all Linear layers)
    
    Returns:
        injection_map: dict of layer_name → "spline"
    """
    injection_map = {}
    
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if target_modules is not None and not any(t in name for t in target_modules):
                continue
            
            spline = SplineLinear(
                module.in_features,
                module.out_features,
                n_control_points=n_control_points,
                basis=basis,
                bias=module.bias is not None,
            )
            
            # Navigate to parent and set attribute
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], spline)
            injection_map[name] = "spline"
    
    return injection_map


def compression_ratio(model: nn.Module) -> Dict[str, int | float]:
    """
    Compute compression ratio for a model with SplineLinear layers.
    
    Returns dict with original_params, spline_params, ratio, n_spline_layers, n_dense_layers.
    """
    original = 0
    spline = 0
    n_spline = 0
    n_dense = 0
    
    for name, module in model.named_modules():
        if isinstance(module, SplineLinear):
            original += module.num_equivalent_dense_params()
            spline += module.num_control_params()
            n_spline += 1
        elif isinstance(module, nn.Linear):
            params = sum(p.numel() for p in module.parameters())
            original += params
            spline += params  # Unchanged — counts equally
            n_dense += 1
    
    return {
        "original_params": original,
        "spline_params": spline,
        "ratio": original / max(spline, 1),
        "n_spline_layers": n_spline,
        "n_dense_layers": n_dense,
    }
