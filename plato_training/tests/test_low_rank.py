"""
Tests for low-rank linear layer.
"""

import pytest
import torch
import torch.nn as nn
from plato_training.low_rank import (
    LowRankLinear, LowRankClassifier, inject_low_rank,
    recommend_variant, VARIANT_GUIDE,
)


class TestLowRankLinear:
    def test_output_shape(self):
        layer = LowRankLinear(128, 64, rank=16)
        x = torch.randn(4, 128)
        assert layer(x).shape == (4, 64)

    def test_fewer_params(self):
        lr = LowRankLinear(512, 512, rank=16)
        dense = nn.Linear(512, 512)
        assert lr.num_low_rank_params() < dense.weight.numel()

    def test_compression_ratio(self):
        lr = LowRankLinear(512, 512, rank=16)
        # 512*16 + 16*512 = 16,384 vs 512*512 = 262,144 → 16:1
        assert lr.compression_ratio() > 10

    def test_gradients_flow(self):
        layer = LowRankLinear(64, 32, rank=8)
        x = torch.randn(2, 64)
        y = layer(x)
        y.sum().backward()
        assert layer.U.grad is not None
        assert layer.V.grad is not None

    def test_no_bias(self):
        layer = LowRankLinear(64, 32, rank=8, bias=False)
        assert layer.bias is None
        assert layer(torch.randn(2, 64)).shape == (2, 32)

    def test_exact_reconstruction_at_full_rank(self):
        """At rank=min(in,out), should be equivalent to dense."""
        layer = LowRankLinear(32, 16, rank=16)  # Full rank for out
        x = torch.randn(4, 32)
        y = layer(x)
        assert y.shape == (4, 16)


class TestLowRankClassifier:
    def test_forward(self):
        model = LowRankClassifier(256, 64, 5, rank=16)
        x = torch.randn(4, 256)
        assert model(x).shape == (4, 5)

    def test_fewer_params_than_dense(self):
        from plato_training.micro_models import MicroClassifier
        dense = MicroClassifier(256, 64, 5)
        lr = LowRankClassifier(256, 64, 5, rank=16)
        assert sum(p.numel() for p in lr.parameters()) < sum(p.numel() for p in dense.parameters())

    def test_learns_topic_classify(self):
        """Low-rank should beat random on topic-classify."""
        from plato_training.micro_models import _generate_synthetic, TASK_REGISTRY
        config = TASK_REGISTRY["topic-classify"]
        X, y = _generate_synthetic("topic-classify", config)
        split = int(len(X) * 0.8)
        
        model = LowRankClassifier(256, 64, 5, rank=16)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        
        model.train()
        for _ in range(30):
            opt.zero_grad()
            loss = loss_fn(model(X[:split]), y[:split])
            loss.backward()
            opt.step()
        
        model.eval()
        with torch.no_grad():
            acc = (model(X[split:]).argmax(1) == y[split:]).float().mean().item()
        
        assert acc > 0.3, f"Low-rank only got {acc:.1%}"


class TestInjectLowRank:
    def test_replaces_linear(self):
        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4))
        m = inject_low_rank(model, rank=8)
        assert len(m) == 2
        assert isinstance(model[0], LowRankLinear)

    def test_forward_after_injection(self):
        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4))
        inject_low_rank(model, rank=8)
        assert model(torch.randn(2, 64)).shape == (2, 4)


class TestRecommendVariant:
    def test_drift_gets_spline(self):
        assert recommend_variant("detect drift in sensor data") == "spline"

    def test_classify_gets_lowrank(self):
        assert recommend_variant("classify documents by topic") == "lowrank"

    def test_unknown_gets_lowrank(self):
        assert recommend_variant("something random") == "lowrank"
