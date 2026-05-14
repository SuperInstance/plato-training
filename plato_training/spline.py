"""
Tensor-Spline module for PLATO Training Rooms.

Weights are parameterised by control points on an Eisenstein (hexagonal)
lattice instead of being independent floats — enabling dramatic compression.

Standard:       W[i][j] = learned_float          (262K params for 512×512)
Tensor-Spline:  W[i][j] = interpolate(control_points, position(i,j))
                                                  (16 params with 16 control points)

The Eisenstein lattice uses ω = e^(2πi/3). Lattice points are a + bω
where a, b ∈ ℤ. In Cartesian coordinates: (a − b/2, b·√3/2).

Public API
----------
EisensteinLattice     Hexagonal control-point layout.
SplineLinear          nn.Linear replacement; weights materialised from lattice.
inject_spline         Replace all nn.Linear layers in a model in-place.
compression_ratio     Dense-equivalent param count ÷ actual spline param count.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# EisensteinLattice
# ---------------------------------------------------------------------------

class EisensteinLattice:
    """
    Places N control points on a hexagonal (Eisenstein) lattice.

    The Eisenstein lattice is spanned by {1, ω} where ω = e^(2πi/3).
    A lattice point (a, b) maps to the Cartesian coordinate::

        x = a − b/2          (real part of a + bω)
        y = b · √3/2         (imaginary part of a + bω)

    Control points are the N lattice points closest to the origin,
    normalised so the outermost selected point has unit Euclidean norm.
    This choice maximises density near the centre — the region that most
    influences weight interpolation.

    Args:
        n_points: Number of control points to place on the lattice.
        device:   Torch device for position tensors. Defaults to CPU.

    Raises:
        ValueError: If ``n_points < 1``.

    Example::

        >>> lattice = EisensteinLattice(7)
        >>> lattice.positions().shape
        torch.Size([7, 2])
        >>> # First point is always the origin.
        >>> lattice.positions()[0]
        tensor([0., 0.])
    """

    _SQRT3_HALF: float = math.sqrt(3.0) / 2.0   # sin(60°) ≈ 0.866

    def __init__(
        self,
        n_points: int,
        device: Optional[torch.device] = None,
    ) -> None:
        if n_points < 1:
            raise ValueError(f"n_points must be ≥ 1, got {n_points}")
        self.n_points = n_points
        self._device = torch.device(device) if device is not None else torch.device("cpu")
        self._positions: torch.Tensor = self._build_lattice(n_points).to(self._device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_lattice(self, n: int) -> torch.Tensor:
        """
        Select the *n* Eisenstein integers closest to the origin.

        The search radius R is chosen conservatively: ring r of the
        Eisenstein lattice contains 6r points (r > 0), so the cumulative
        count up to ring R is 1 + 3R(R+1). Solving 1 + 3R(R+1) ≥ n
        gives R ≈ √(n/3). We add a small margin to avoid off-by-one.

        Returns:
            Float32 tensor of shape (n, 2). The origin is always included
            (closest point, distance 0). Positions are normalised so the
            outermost selected point has unit Euclidean norm.
        """
        R = max(int(math.ceil(math.sqrt(n / 3.0))) + 2, 3)

        candidates: List[Tuple[float, float, float]] = []
        for a in range(-R, R + 1):
            for b in range(-R, R + 1):
                x = float(a) - float(b) * 0.5         # Re(a + bω)
                y = float(b) * self._SQRT3_HALF        # Im(a + bω)
                dist_sq = x * x + y * y
                candidates.append((dist_sq, x, y))

        # Primary sort: squared distance; secondary: (x, y) for determinism.
        candidates.sort(key=lambda t: (round(t[0], 9), t[1], t[2]))
        selected = candidates[:n]

        positions = torch.tensor(
            [[x, y] for _, x, y in selected],
            dtype=torch.float32,
        )  # (n, 2)

        # Normalise so the most distant selected point sits at unit radius.
        # Guard against the degenerate n=1 case (only the origin).
        max_dist = positions.norm(dim=1).max()
        if max_dist > 0.0:
            positions = positions / max_dist

        return positions

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def positions(self) -> torch.Tensor:
        """
        Return the control-point coordinates.

        Returns:
            Float32 tensor of shape ``(n_points, 2)`` on the lattice's
            device, with positions normalised to the unit disk.
        """
        return self._positions

    def nearest_k(
        self,
        point: torch.Tensor,
        k: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the k nearest control points to a 2-D query point.

        Args:
            point: Query coordinates, shape ``(2,)``, any device.
            k:     Number of nearest neighbours. Clamped to ``n_points``
                   if larger.

        Returns:
            Tuple ``(distances, indices)`` sorted nearest-first, each of
            shape ``(min(k, n_points),)``.
        """
        point = point.to(self._device)
        dists = torch.norm(self._positions - point, dim=1)   # (n_points,)
        k = min(k, self.n_points)
        distances, indices = torch.topk(dists, k, largest=False)
        return distances, indices

    def to(self, device: torch.device) -> "EisensteinLattice":
        """Move position tensor to *device* in-place and return ``self``."""
        self._device = torch.device(device)
        self._positions = self._positions.to(device)
        return self

    def __repr__(self) -> str:
        return (
            f"EisensteinLattice(n_points={self.n_points}, "
            f"device={self._device})"
        )


# ---------------------------------------------------------------------------
# SplineLinear
# ---------------------------------------------------------------------------

class SplineLinear(nn.Module):
    """
    Drop-in replacement for ``nn.Linear`` with lattice-parameterised weights.

    The full weight matrix W (shape ``out_features × in_features``) is
    re-materialised from control-point scalar values on every forward
    pass. Gradients flow back through the interpolation to the control
    points via standard autograd — the optimizer only updates
    ``n_control_points`` scalars instead of ``out × in`` independent
    weights.

    **Basis functions**

    ``"eisenstein"``
        Normalised inverse-distance-squared weighting using control points
        on the Eisenstein hexagonal lattice. Each weight element receives
        a smooth interpolant from *all* control points, weighted by
        proximity::

            W(p) = Σ_k c_k · d_k⁻² / Σ_k d_k⁻²,   d_k = ||p − L_k|| + ε

    ``"bspline"``
        Bicubic B-spline via ``torch.nn.functional.grid_sample``. Control
        points are arranged on a regular K×K grid (K = ⌈√n_control_points⌉)
        and sampled at each weight position. Excellent locality: each
        weight element is only affected by its 4×4 neighbourhood.

    ``"gaussian"``
        Gaussian radial-basis-function interpolation on the Eisenstein
        lattice with a single learnable log-bandwidth parameter::

            W(p) = Σ_k c_k · exp(−||p − L_k||² / 2σ²) / Σ_k (·),
                   σ = exp(log_bandwidth)

        Bandwidth is trained alongside control values, trading off between
        global smoothness (large σ) and local precision (small σ).

    Args:
        in_features:      Input width (same semantics as ``nn.Linear``).
        out_features:     Output width.
        n_control_points: Number of scalar control-point parameters.
        basis:            One of ``"eisenstein"``, ``"bspline"``,
                          ``"gaussian"``.
        bias:             If True (default), add a learnable bias vector.

    Raises:
        ValueError: If *basis* is not one of the three supported strings.

    Example::

        >>> layer = SplineLinear(512, 512, n_control_points=16)
        >>> x = torch.randn(4, 512)
        >>> layer(x).shape
        torch.Size([4, 512])
        >>> layer.num_trainable_params()   # 16 ctrl + 512 bias
        528
    """

    _VALID_BASES: frozenset = frozenset({"eisenstein", "bspline", "gaussian"})

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_control_points: int = 16,
        basis: str = "eisenstein",
        bias: bool = True,
    ) -> None:
        super().__init__()

        if basis not in self._VALID_BASES:
            raise ValueError(
                f"basis must be one of {set(self._VALID_BASES)}, "
                f"got '{basis!r}'"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.n_control_points = n_control_points
        self.basis = basis

        # ---- Control-point values (the only "weight-like" parameters) ----
        # Kaiming-style initialisation: bound ∝ 1/√fan_in where fan_in is
        # n_control_points, keeping activations well-scaled at init.
        bound = 1.0 / math.sqrt(n_control_points)
        self.control_values = nn.Parameter(
            torch.empty(n_control_points).uniform_(-bound, bound)
        )

        # ---- Basis-specific setup ----------------------------------------
        if basis in ("eisenstein", "gaussian"):
            lattice = EisensteinLattice(n_control_points)
            # Register positions as a buffer: moves with the module on
            # .to(device) / .cuda() calls, but has no gradient.
            self.register_buffer("_lattice_pos", lattice.positions())  # (n_cp, 2)

        if basis == "gaussian":
            # log_bandwidth: exp(0) = 1 at init; trained during forward passes.
            self.log_bandwidth = nn.Parameter(torch.zeros(1))

        if basis == "bspline":
            # K is the grid side length; K² ≥ n_control_points.
            self._grid_k: int = math.ceil(math.sqrt(n_control_points))

        # ---- Fixed weight-matrix sampling grid (registered as buffer) ----
        # Maps every weight element W[i, j] to a 2-D coordinate in [-1, 1]².
        # Row index i → y-axis; column index j → x-axis.
        row_coords = torch.linspace(-1.0, 1.0, out_features)
        col_coords = torch.linspace(-1.0, 1.0, in_features)
        grid_y, grid_x = torch.meshgrid(row_coords, col_coords, indexing="ij")
        # Shape: (out_features * in_features, 2)  — columns are [x, y].
        self.register_buffer(
            "_grid_positions",
            torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1),
        )

        # ---- Optional bias -----------------------------------------------
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    # ------------------------------------------------------------------
    # Weight materialisation
    # ------------------------------------------------------------------

    def _materialize(self) -> torch.Tensor:
        """
        Compute the full weight matrix from control-point values.

        Returns:
            Float tensor of shape ``(out_features, in_features)``
            on the same device as ``control_values``.
        """
        if self.basis == "eisenstein":
            w_flat = self._basis_eisenstein()
        elif self.basis == "bspline":
            w_flat = self._basis_bspline()
        else:
            w_flat = self._basis_gaussian()

        return w_flat.reshape(self.out_features, self.in_features)

    def _basis_eisenstein(self) -> torch.Tensor:
        """
        Normalised inverse-distance-squared weighting on the Eisenstein lattice.

        The kernel ``d^{-2}`` gives strong locality (nearby control points
        dominate) while remaining smooth everywhere. A small ε prevents
        division by zero if a weight position coincides with a lattice point.

        Returns:
            Flat tensor of shape ``(out_features * in_features,)``.
        """
        # dists: (out * in, n_cp)
        dists = torch.cdist(self._grid_positions, self._lattice_pos)
        kernel = 1.0 / (dists.pow(2) + 1e-6)
        kernel = kernel / kernel.sum(dim=1, keepdim=True)
        return kernel @ self.control_values   # (out * in,)

    def _basis_bspline(self) -> torch.Tensor:
        """
        Bicubic B-spline interpolation via ``F.grid_sample``.

        ``control_values`` are reshaped to a ``(1, 1, K, K)`` grid and
        sampled at every weight-element position using PyTorch's built-in
        bicubic kernel. ``grid_sample`` is differentiable w.r.t. the
        input image (not the grid), so gradients flow cleanly to
        ``control_values``. If ``n_control_points < K²`` the extra slots
        are zero-padded.

        Returns:
            Flat tensor of shape ``(out_features * in_features,)``.
        """
        k = self._grid_k
        pad = k * k - self.n_control_points
        ctrl = self.control_values
        if pad > 0:
            ctrl = F.pad(ctrl, (0, pad))           # (K²,)

        # grid_sample input shape: (N=1, C=1, H=K, W=K)
        ctrl_grid = ctrl.reshape(1, 1, k, k)

        # grid_sample grid shape: (N=1, H_out=1, W_out=out*in, 2)
        # Values must be in [-1, 1] — our _grid_positions already are.
        n_pts = self._grid_positions.shape[0]
        sample_grid = self._grid_positions.reshape(1, 1, n_pts, 2)

        sampled = F.grid_sample(
            ctrl_grid,
            sample_grid,
            mode="bicubic",
            padding_mode="border",
            align_corners=True,
        )  # (1, 1, 1, out * in)

        return sampled.reshape(-1)               # (out * in,)

    def _basis_gaussian(self) -> torch.Tensor:
        """
        Gaussian RBF interpolation on the Eisenstein lattice.

        The bandwidth σ = exp(``log_bandwidth``) is trained jointly with
        the control values. A large σ gives globally smooth weights; a
        small σ gives locally precise weights. The ``clamp(min=1e-3)``
        prevents σ from collapsing to zero during training.

        Returns:
            Flat tensor of shape ``(out_features * in_features,)``.
        """
        sigma = self.log_bandwidth.exp().clamp(min=1e-3)
        dists_sq = torch.cdist(
            self._grid_positions, self._lattice_pos
        ).pow(2)                                 # (out * in, n_cp)
        kernel = torch.exp(-dists_sq / (2.0 * sigma ** 2))
        kernel = kernel / (kernel.sum(dim=1, keepdim=True) + 1e-8)
        return kernel @ self.control_values      # (out * in,)

    # ------------------------------------------------------------------
    # nn.Module interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Materialise the weight matrix, then apply the linear transform.

        Args:
            x: Input tensor of shape ``(*, in_features)``.

        Returns:
            Output tensor of shape ``(*, out_features)``.
        """
        W = self._materialize()
        return F.linear(x, W, self.bias)

    def num_trainable_params(self) -> int:
        """Total number of parameters that receive gradients."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_equivalent_dense_params(self) -> int:
        """Parameters that an equivalent ``nn.Linear`` would use."""
        n = self.out_features * self.in_features
        if self.bias is not None:
            n += self.out_features
        return n

    def compression_ratio(self) -> float:
        """
        Compression ratio for this layer versus an equivalent ``nn.Linear``.

        Returns:
            ``dense_equivalent_params / actual_trainable_params``. A value
            of 64 means this layer uses 64× fewer parameters than a dense
            layer of the same shape. Always ≥ 1.0.
        """
        actual = self.num_trainable_params()
        dense = self.num_equivalent_dense_params()
        return float(dense) / float(max(actual, 1))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"n_control_points={self.n_control_points}, "
            f"basis='{self.basis}', "
            f"bias={self.bias is not None}"
        )


# ---------------------------------------------------------------------------
# Model-level utilities
# ---------------------------------------------------------------------------

def inject_spline(
    model: nn.Module,
    n_control_points: int = 16,
    basis: str = "eisenstein",
    target_modules: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Replace every ``nn.Linear`` in *model* with a :class:`SplineLinear`.

    Follows the same structural-surgery pattern as ``inject_lora``:

    1. Snapshot ``model.named_modules()`` before modification.
    2. Navigate to the parent of each ``nn.Linear`` via attribute traversal.
    3. Call ``setattr`` to swap the layer in-place.

    The model is mutated. Original ``nn.Linear`` weight tensors are
    discarded; only the new ``SplineLinear`` control points remain as
    trainable parameters for the replaced layers.

    Args:
        model:            The ``nn.Module`` to modify in-place.
        n_control_points: Control points per replaced layer.
        basis:            Basis function for every injected layer.

    Returns:
        ``injection_map``: dict mapping each replaced module path to the
        string ``"spline"``. Compatible with the ``injection_map`` format
        from ``inject_lora``.

    Example::

        >>> model = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 10))
        >>> imap = inject_spline(model, n_control_points=8)
        >>> list(imap.values())
        ['spline', 'spline']
    """
    # Snapshot before mutation to avoid iterator invalidation.
    named = list(model.named_modules())
    injection_map: Dict[str, str] = {}

    for name, module in named:
        if not isinstance(module, nn.Linear):
            continue
        if target_modules is not None and not any(t in name for t in target_modules):
            continue

        spline = SplineLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            n_control_points=n_control_points,
            basis=basis,
            bias=module.bias is not None,
        )

        # Navigate to the parent and replace the child attribute.
        parts = name.split(".")
        parent: nn.Module = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], spline)

        injection_map[name] = "spline"

    return injection_map


def compression_ratio(model: nn.Module) -> Dict[str, object]:
    """
    Compute the compression achieved by :class:`SplineLinear` layers.

    For each ``SplineLinear``, computes:

    * **Dense equivalent** — ``out_features × in_features`` (+ bias).
    * **Actual spline params** — ``control_values`` (+ ``log_bandwidth``
      for Gaussian) (+ bias).

    Non-``SplineLinear`` parameters (batch-norm, embeddings, etc.)
    contribute equally to numerator and denominator (ratio = 1 for those).

    Args:
        model: An ``nn.Module`` that may contain :class:`SplineLinear`
               modules.

    Returns:
        ``equivalent_params / actual_params``. Values > 1 indicate
        compression. Returns ``1.0`` if the model has no parameters.

    Example::

        >>> layer = SplineLinear(512, 512, n_control_points=16, bias=False)
        >>> compression_ratio(layer)   # 262144 / 16 = 16384.0
        16384.0
    """
    equivalent_total = 0
    actual_total = 0
    n_spline_layers = 0
    n_dense_layers = 0

    # Track fully-qualified parameter names handled inside SplineLinear
    # modules to avoid double-counting when sweeping model.named_parameters().
    accounted: Set[str] = set()

    for mod_name, module in model.named_modules():
        if isinstance(module, SplineLinear):
            n_spline_layers += 1

            # Dense equivalent: full matrix + optional bias.
            dense = module.out_features * module.in_features
            if module.bias is not None:
                dense += module.out_features
            equivalent_total += dense

            # Actual: control_values (+ log_bandwidth for gaussian) (+ bias).
            actual = module.control_values.numel()
            if module.basis == "gaussian":
                actual += module.log_bandwidth.numel()
            if module.bias is not None:
                actual += module.bias.numel()
            actual_total += actual

            # Mark all parameters inside this SplineLinear as handled.
            prefix = f"{mod_name}." if mod_name else ""
            for pname, _ in module.named_parameters():
                accounted.add(f"{prefix}{pname}")

        elif isinstance(module, nn.Linear):
            n_dense_layers += 1

    # Non-SplineLinear parameters count at face value (no compression).
    for pname, param in model.named_parameters():
        if pname not in accounted:
            n = param.numel()
            equivalent_total += n
            actual_total += n

    ratio = (
        float(equivalent_total) / float(actual_total)
        if actual_total > 0
        else 1.0
    )

    return {
        "original_params": equivalent_total,
        "spline_params": actual_total,
        "ratio": ratio,
        "n_spline_layers": n_spline_layers,
        "n_dense_layers": n_dense_layers,
    }
