"""
Tests for Tensor-Spline module.
"""

import pytest
import torch
import torch.nn as nn
from plato_training.spline import (
    EisensteinLattice, SplineLinear, inject_spline, compression_ratio,
)


class TestEisensteinLattice:
    def test_creates_correct_number_of_points(self):
        lattice = EisensteinLattice(16, 100, 100)
        assert lattice.positions().shape == (16, 2)
    
    def test_hexagonal_symmetry(self):
        """Hex grid has consistent spacing."""
        lattice = EisensteinLattice(25, 100, 100)
        pos = lattice.positions()
        # Check no duplicate positions
        assert len(pos) == len(torch.unique(pos, dim=0))
    
    def test_nearest_k(self):
        lattice = EisensteinLattice(16, 100, 100)
        point = torch.tensor([50.0, 50.0])
        indices, distances = lattice.nearest_k(point, k=4)
        assert indices.shape == (4,)
        assert distances.shape == (4,)
        # Distances should be sorted
        assert (distances[1:] >= distances[:-1]).all()


class TestSplineLinear:
    def test_output_shape(self):
        """SplineLinear produces correct output shape."""
        layer = SplineLinear(32, 16, n_control_points=8)
        x = torch.randn(4, 32)
        y = layer(x)
        assert y.shape == (4, 16)
    
    def test_fewer_params(self):
        """SplineLinear has fewer params than nn.Linear."""
        spline = SplineLinear(512, 512, n_control_points=16)
        dense = nn.Linear(512, 512)
        
        spline_params = sum(p.numel() for p in spline.parameters())
        dense_params = sum(p.numel() for p in dense.parameters())
        
        assert spline_params < dense_params
        assert spline.compression_ratio() > 1.0
    
    def test_gradients_flow_to_control_points(self):
        """Gradients reach control values."""
        layer = SplineLinear(32, 16, n_control_points=8)
        x = torch.randn(4, 32)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        
        assert layer.control_values.grad is not None
        assert layer.control_values.grad.abs().sum() > 0
    
    def test_no_gradient_for_lattice_positions(self):
        """Lattice positions are not learnable."""
        layer = SplineLinear(32, 16, n_control_points=8)
        x = torch.randn(4, 32)
        y = layer(x)
        y.sum().backward()
        
        # Lattice positions should not be in parameters
        param_names = [n for n, _ in layer.named_parameters()]
        assert "lattice" not in param_names
    
    def test_basis_eisenstein(self):
        layer = SplineLinear(32, 16, n_control_points=8, basis="eisenstein")
        x = torch.randn(2, 32)
        assert layer(x).shape == (2, 16)
    
    def test_basis_gaussian(self):
        layer = SplineLinear(32, 16, n_control_points=8, basis="gaussian")
        x = torch.randn(2, 32)
        assert layer(x).shape == (2, 16)
    
    def test_basis_bspline(self):
        layer = SplineLinear(32, 16, n_control_points=8, basis="bspline")
        x = torch.randn(2, 32)
        assert layer(x).shape == (2, 16)
    
    def test_no_bias(self):
        layer = SplineLinear(32, 16, n_control_points=8, bias=False)
        assert layer.bias is None
        x = torch.randn(2, 32)
        assert layer(x).shape == (2, 16)


class TestInjectSpline:
    def test_replaces_all_linear(self):
        """With target_modules=None, replaces all Linear layers."""
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        injection_map = inject_spline(model, n_control_points=8)
        assert len(injection_map) == 2
        
        # Check layers are SplineLinear
        assert isinstance(model[0], SplineLinear)
        assert isinstance(model[2], SplineLinear)
    
    def test_respects_target_modules(self):
        """With target_modules, only replaces matching layers."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.W_query = nn.Linear(10, 32)
                self.W_value = nn.Linear(32, 32)
                self.out = nn.Linear(32, 2)
            def forward(self, x):
                return self.out(self.W_value(self.W_query(x)))
        
        model = TestModel()
        injection_map = inject_spline(model, n_control_points=8, target_modules=["W_query"])
        
        assert len(injection_map) == 1
        assert isinstance(model.W_query, SplineLinear)
        assert isinstance(model.out, nn.Linear)  # NOT replaced
    
    def test_forward_works_after_injection(self):
        model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
        inject_spline(model, n_control_points=8)
        x = torch.randn(4, 10)
        y = model(x)
        assert y.shape == (4, 2)


class TestCompressionRatio:
    def test_reports_correctly(self):
        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8))
        inject_spline(model, n_control_points=16)
        
        stats = compression_ratio(model)
        assert stats["n_spline_layers"] == 2
        assert stats["n_dense_layers"] == 0
        assert stats["ratio"] > 1.0
        assert stats["original_params"] > stats["spline_params"]
    
    def test_mixed_model(self):
        """Model with both spline and dense layers."""
        class Mixed(nn.Module):
            def __init__(self):
                super().__init__()
                self.spline_layer = SplineLinear(64, 32, n_control_points=8)
                self.dense_layer = nn.Linear(32, 4)
            def forward(self, x):
                return self.dense_layer(self.spline_layer(x))
        
        model = Mixed()
        stats = compression_ratio(model)
        assert stats["n_spline_layers"] == 1
        assert stats["n_dense_layers"] == 1
