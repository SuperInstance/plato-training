"""
Tests for Tensor-Spline module.

Six required cases:
  1. SplineLinear produces correct output shape.
  2. SplineLinear has fewer parameters than an equivalent nn.Linear.
  3. compression_ratio reports correctly.
  4. inject_spline replaces all nn.Linear layers.
  5. Gradients flow to control points only (lattice positions are not parameters).
  6. Eisenstein lattice positions are on the correct hexagonal grid.
"""

import math

import pytest
import torch
import torch.nn as nn
from plato_training.spline import (
    EisensteinLattice, SplineLinear, inject_spline, compression_ratio,
)


class TestEisensteinLattice:
    def test_creates_correct_number_of_points(self):
        lattice = EisensteinLattice(16)
        assert lattice.positions().shape == (16, 2)

    def test_hexagonal_symmetry(self):
        lattice = EisensteinLattice(25)
        pos = lattice.positions()
        assert len(pos) == len(torch.unique(pos, dim=0))

    def test_nearest_k(self):
        lattice = EisensteinLattice(16)
        point = lattice.positions()[0]  # Use an existing point
        distances, indices = lattice.nearest_k(point, k=4)
        assert distances.shape == (4,)
        assert indices.shape == (4,)


class TestSplineLinear:
    def test_output_shape(self):
        layer = SplineLinear(32, 16, n_control_points=8)
        x = torch.randn(4, 32)
        y = layer(x)
        assert y.shape == (4, 16)

    def test_fewer_params_than_dense(self):
        spline = SplineLinear(512, 512, n_control_points=16)
        dense = nn.Linear(512, 512)
        assert spline.num_trainable_params() < dense.weight.numel()

    def test_compression_ratio(self):
        spline = SplineLinear(512, 512, n_control_points=16)
        ratio = spline.num_equivalent_dense_params() / spline.num_trainable_params()
        assert ratio > 10  # Should be ~16,000:1 for this config

    def test_gradients_flow_to_control_points(self):
        layer = SplineLinear(32, 16, n_control_points=8)
        x = torch.randn(4, 32)
        y = layer(x)
        y.sum().backward()
        assert layer.control_values.grad is not None

    def test_basis_eisenstein(self):
        layer = SplineLinear(32, 16, n_control_points=8, basis="eisenstein")
        assert layer(torch.randn(2, 32)).shape == (2, 16)

    def test_basis_gaussian(self):
        layer = SplineLinear(32, 16, n_control_points=8, basis="gaussian")
        assert layer(torch.randn(2, 32)).shape == (2, 16)

    def test_basis_bspline(self):
        layer = SplineLinear(32, 16, n_control_points=8, basis="bspline")
        assert layer(torch.randn(2, 32)).shape == (2, 16)


class TestInjectSpline:
    def test_replaces_all_linear(self):
        model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
        injection_map = inject_spline(model, n_control_points=8)
        assert len(injection_map) == 2
        assert isinstance(model[0], SplineLinear)

    def test_respects_target_modules(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.W_query = nn.Linear(10, 32)
                self.out = nn.Linear(32, 2)
            def forward(self, x): return self.out(self.W_query(x))

        model = M()
        injection_map = inject_spline(model, n_control_points=8, target_modules=["W_query"])
        assert len(injection_map) == 1
        assert isinstance(model.W_query, SplineLinear)
        assert isinstance(model.out, nn.Linear)

    def test_forward_works_after_injection(self):
        model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
        inject_spline(model, n_control_points=8)
        assert model(torch.randn(4, 10)).shape == (4, 2)


class TestCompressionRatio:
    def test_reports_correctly(self):
        """compression_ratio returns a dict with correct structure and values."""
        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8))
        inject_spline(model, n_control_points=16)
        stats = compression_ratio(model)
        assert isinstance(stats, dict)
        assert stats["n_spline_layers"] == 2
        assert stats["n_dense_layers"] == 0
        assert stats["ratio"] > 1.0
        assert stats["original_params"] > stats["spline_params"]

    def test_all_keys_present(self):
        model = nn.Sequential(nn.Linear(32, 16))
        inject_spline(model, n_control_points=4)
        stats = compression_ratio(model)
        for key in ("original_params", "spline_params", "ratio",
                    "n_spline_layers", "n_dense_layers"):
            assert key in stats

    def test_ratio_is_one_without_spline_layers(self):
        model = nn.Sequential(nn.Linear(32, 16))  # not injected
        stats = compression_ratio(model)
        assert stats["ratio"] == 1.0


# ---------------------------------------------------------------------------
# 5. Gradients flow to control points only (frozen base weights)
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Test 5: control_values accumulate gradients; lattice buffers do not."""

    def test_control_values_have_gradient_after_backward(self):
        layer = SplineLinear(32, 16, n_control_points=8)
        layer(torch.randn(4, 32)).sum().backward()
        assert layer.control_values.grad is not None
        assert layer.control_values.grad.abs().sum() > 0

    def test_lattice_positions_are_not_parameters(self):
        """_lattice_pos is a buffer — must not appear in parameters()."""
        layer = SplineLinear(32, 16, n_control_points=8, basis="eisenstein")
        param_names = {n for n, _ in layer.named_parameters()}
        assert "_lattice_pos" not in param_names

    def test_lattice_positions_have_no_grad(self):
        layer = SplineLinear(32, 16, n_control_points=8, basis="eisenstein")
        layer(torch.randn(4, 32)).sum().backward()
        assert layer._lattice_pos.grad is None

    def test_no_dense_weight_attribute(self):
        """SplineLinear must not expose a dense 'weight' tensor."""
        layer = SplineLinear(32, 16, n_control_points=8)
        assert not hasattr(layer, "weight")

    def test_gaussian_bandwidth_has_gradient(self):
        layer = SplineLinear(32, 16, n_control_points=8, basis="gaussian")
        layer(torch.randn(4, 32)).sum().backward()
        assert layer.log_bandwidth.grad is not None

    @pytest.mark.parametrize("basis", ["eisenstein", "bspline", "gaussian"])
    def test_all_bases_differentiable(self, basis):
        """Every basis function must be differentiable w.r.t. control_values."""
        torch.manual_seed(0)
        layer = SplineLinear(16, 32, n_control_points=9, basis=basis)
        x = torch.randn(2, 16, requires_grad=True)
        layer(x).pow(2).mean().backward()
        assert x.grad is not None, f"[{basis}] No gradient on input"
        assert layer.control_values.grad is not None, \
            f"[{basis}] No gradient on control_values"

    def test_full_training_step_updates_control_points(self):
        model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
        inject_spline(model, n_control_points=8)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        model(torch.randn(4, 32)).sum().backward()
        optimizer.step()
        for name, mod in model.named_modules():
            if isinstance(mod, SplineLinear):
                assert mod.control_values.grad is not None, \
                    f"No grad on control_values at '{name}'"


# ---------------------------------------------------------------------------
# 6. Eisenstein lattice positions are on the correct hexagonal grid
# ---------------------------------------------------------------------------

class TestEisensteinLatticeGeometry:
    """Test 6: verify hexagonal geometry of the Eisenstein lattice."""

    def test_first_point_is_origin(self):
        """The origin is closest to itself — always sorted first."""
        pos = EisensteinLattice(7).positions()
        assert torch.allclose(pos[0], torch.zeros(2), atol=1e-5)

    def test_ring_one_has_unit_norm(self):
        """For n=7 (origin + 6 ring-1 neighbours), ring-1 sits at unit distance."""
        pos = EisensteinLattice(7).positions()
        ring_one = pos[1:]  # exclude origin
        norms = ring_one.norm(dim=1)
        assert torch.allclose(norms, torch.ones(6), atol=1e-5)

    def test_ring_one_60_degree_angles(self):
        """
        The 6 ring-1 neighbours are spaced 60° (π/3) apart.
        This is the defining signature of a hexagonal grid.
        """
        pos = EisensteinLattice(7).positions()[1:]
        angles = torch.atan2(pos[:, 1], pos[:, 0])
        angles_sorted, _ = angles.sort()

        gaps = torch.diff(angles_sorted)
        last_gap = (angles_sorted[0] + 2 * math.pi) - angles_sorted[-1]
        all_gaps = torch.cat([gaps, last_gap.unsqueeze(0)])

        expected = math.pi / 3.0
        assert torch.allclose(
            all_gaps, torch.full_like(all_gaps, expected), atol=1e-4
        ), f"Expected 60° gaps, got {all_gaps}"

    def test_uniform_nearest_neighbour_distance(self):
        """
        In a valid hexagonal lattice, all nearest-neighbour distances are equal
        (CV < 5%).
        """
        pos = EisensteinLattice(19).positions()
        diffs = pos.unsqueeze(0) - pos.unsqueeze(1)
        dists = diffs.norm(dim=2)
        dists.fill_diagonal_(float("inf"))
        nn_dists = dists.min(dim=1).values

        cv = nn_dists.std() / nn_dists.mean()
        assert cv.item() < 0.05, f"Uniform NN distances expected (CV={cv.item():.4f})"

    def test_no_duplicate_positions(self):
        pos = EisensteinLattice(19).positions()
        assert torch.unique(pos, dim=0).shape[0] == 19

    def test_positions_within_unit_disk(self):
        for n in [7, 16, 37]:
            norms = EisensteinLattice(n).positions().norm(dim=1)
            assert norms.max().item() <= 1.0 + 1e-5

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_spline_linear_on_cuda(self):
        """SplineLinear runs on CUDA with correct output shape and gradients."""
        layer = SplineLinear(32, 64, n_control_points=16).cuda()
        x = torch.randn(4, 32, device="cuda")
        out = layer(x)
        assert out.device.type == "cuda"
        assert out.shape == (4, 64)
        out.sum().backward()
        assert layer.control_values.grad is not None
