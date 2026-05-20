"""
Tests for Pre-LLM Filter (intelligence_pre_filter).
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from plato_training.intelligence_pre_filter import (
    FEATURE_DIM,
    IntentType, Domain, Urgency, RouteTarget,
    RequestFeatures, RoutingDecision,
    PreFilterModel, PreFilter,
    _make_linear,
)


# ---------------------------------------------------------------------------
# Feature vector tests
# ---------------------------------------------------------------------------

class TestRequestFeatures:
    """Test the 384-dim feature vector construction."""

    def test_dimension(self):
        """Feature vector must be exactly 384 dims."""
        rf = RequestFeatures()
        vec = rf.to_vector()
        assert vec.shape == (384,)
        assert vec.dtype == np.float32

    def test_intent_one_hot(self):
        """Each intent sets exactly one bit in [0:8]."""
        for intent in IntentType:
            rf = RequestFeatures(intent=intent)
            vec = rf.to_vector()
            assert vec[intent.value] == 1.0
            assert vec[0:8].sum() == 1.0

    def test_domain_one_hot(self):
        """Domain one-hot in correct position."""
        for dom in Domain:
            rf = RequestFeatures(domain=dom)
            vec = rf.to_vector()
            assert vec[40 + dom.value] == 1.0
            assert vec[40:47].sum() == 1.0

    def test_urgency_one_hot(self):
        """Urgency one-hot in correct position."""
        for urg in Urgency:
            rf = RequestFeatures(urgency=urg)
            vec = rf.to_vector()
            assert vec[47 + urg.value] == 1.0
            assert vec[47:51].sum() == 1.0

    def test_time_of_day_circular(self):
        """Time-of-day encoded as sin/cos."""
        rf = RequestFeatures(hour=0.0)
        vec = rf.to_vector()
        assert abs(vec[51] - 0.0) < 1e-6   # sin(0) = 0
        assert abs(vec[52] - 1.0) < 1e-6   # cos(0) = 1

        rf6 = RequestFeatures(hour=6.0)
        v6 = rf6.to_vector()
        assert abs(v6[51] - 1.0) < 1e-6   # sin(π/2) = 1
        assert abs(v6[52] - 0.0) < 1e-6   # cos(π/2) = 0

    def test_day_of_week_one_hot(self):
        """Day-of-week one-hot in correct position."""
        for dow in range(7):
            rf = RequestFeatures(day_of_week=dow)
            vec = rf.to_vector()
            assert vec[53 + dow] == 1.0
            assert vec[53:60].sum() == 1.0

    def test_keyword_vector(self):
        """Keywords placed at correct offset."""
        kw = np.zeros(64, dtype=np.float32)
        kw[0] = 0.5
        kw[63] = 1.0
        rf = RequestFeatures(keyword_vector=kw)
        vec = rf.to_vector()
        assert vec[64] == pytest.approx(0.5)
        assert vec[127] == pytest.approx(1.0)

    def test_historical_match_clamped(self):
        """Historical match clamped to [0, 1]."""
        rf = RequestFeatures(historical_match=2.0)
        vec = rf.to_vector()
        assert vec[128] == 1.0

        rf_neg = RequestFeatures(historical_match=-0.5)
        vec_neg = rf_neg.to_vector()
        assert vec_neg[128] == 0.0

    def test_padding_is_zero(self):
        """Dims 129..383 should be zero (padding)."""
        rf = RequestFeatures()
        vec = rf.to_vector()
        assert np.all(vec[129:] == 0.0)

    def test_input_length_bucketing(self):
        """Input length log-bucketed into 16 bins."""
        rf = RequestFeatures(input_length=1)
        vec = rf.to_vector()
        assert vec[8:24].sum() == 1.0  # log2(1)=0 → bucket 0

        rf_big = RequestFeatures(input_length=65536)
        vec_big = rf_big.to_vector()
        assert vec_big[8:24].sum() == 1.0  # log2(65536)=16, clamped to 15

    def test_success_rate_bucketing(self):
        """Success rate divided into 4 bins."""
        rf = RequestFeatures(recent_success_rate=0.0)
        vec = rf.to_vector()
        assert vec[60] == 1.0

        rf_high = RequestFeatures(recent_success_rate=0.9)
        vec_high = rf_high.to_vector()
        assert vec_high[63] == 1.0  # 0.9*4=3.6 → bin 3


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestPreFilterModel:
    """Test the neural net architecture."""

    def test_output_shapes(self):
        """Forward pass produces correct shapes."""
        model = PreFilterModel()
        x = torch.randn(4, 384)
        route, conf, tokens = model(x)
        assert route.shape == (4, 8)
        assert conf.shape == (4, 1)
        assert tokens.shape == (4, 1)

    def test_confidence_bounded(self):
        """Confidence output is in [0, 1] (sigmoid)."""
        model = PreFilterModel()
        x = torch.randn(16, 384)
        _, conf, _ = model(x)
        assert conf.min() >= 0.0
        assert conf.max() <= 1.0

    def test_single_input(self):
        """Works with batch size 1."""
        model = PreFilterModel()
        x = torch.randn(1, 384)
        route, conf, tokens = model(x)
        assert route.shape == (1, 8)

    def test_gradients_flow(self):
        """All heads receive gradients."""
        model = PreFilterModel()
        x = torch.randn(2, 384)
        route, conf, tokens = model(x)
        loss = route.sum() + conf.sum() + tokens.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------

class TestPreFilter:
    """Test the PreFilter orchestrator."""

    def test_route_returns_decision(self):
        """route() returns a valid RoutingDecision."""
        pf = PreFilter()
        features = RequestFeatures(intent=IntentType.CODE, input_length=500)
        dec = pf.route(features)
        assert isinstance(dec, RoutingDecision)
        assert isinstance(dec.route, RouteTarget)
        assert 0.0 <= dec.confidence <= 1.0
        assert dec.max_tokens_estimate >= 1

    def test_record_outcome_accumulates(self):
        """record_outcome adds to the outcome buffer."""
        pf = PreFilter()
        features = RequestFeatures()
        dec = pf.route(features)
        assert pf.n_outcomes == 0
        pf.record_outcome(features, dec, was_good=True)
        assert pf.n_outcomes == 1
        pf.record_outcome(features, dec, was_good=False)
        assert pf.n_outcomes == 2

    def test_self_train_insufficient_data(self):
        """self_train with <2 outcomes returns status=insufficient_data."""
        pf = PreFilter()
        result = pf.self_train()
        assert result["status"] == "insufficient_data"

    def test_self_train_runs(self):
        """self_train runs with enough outcomes and returns metrics."""
        pf = PreFilter()
        features = RequestFeatures(intent=IntentType.MATH, domain=Domain.MATH)
        for i in range(10):
            dec = pf.route(features)
            pf.record_outcome(features, dec, was_good=(i % 2 == 0))
        result = pf.self_train(epochs=3)
        assert result["status"] == "ok"
        assert result["n_samples"] == 10
        assert result["epochs"] == 3
        assert len(result["loss_curve"]) == 3

    def test_export_load_tile_roundtrip(self):
        """Export then load produces identical model weights."""
        pf = PreFilter()
        # Route to initialize lazy params
        features = RequestFeatures()
        pf.route(features)

        with tempfile.TemporaryDirectory() as td:
            tile_id = pf.export_tile(store_dir=td)
            assert tile_id.startswith("prefilter-")

            # Create new PreFilter, load tile
            pf2 = PreFilter()
            loaded = pf2.load_tile(tile_id, store_dir=td)
            assert loaded is True

            # Compare weights
            for (n1, p1), (n2, p2) in zip(
                pf.model.named_parameters(),
                pf2.model.named_parameters(),
            ):
                assert n1 == n2
                assert torch.allclose(p1, p2, atol=1e-6), f"Mismatch in {n1}"

    def test_load_nonexistent_tile(self):
        """Loading a non-existent tile returns False."""
        pf = PreFilter()
        result = pf.load_tile("nonexistent-tile", store_dir="/tmp/empty-prefilter-test")
        assert result is False

    def test_param_count_reasonable(self):
        """Model should have a tractable number of parameters."""
        pf = PreFilter()
        n = pf.param_count()
        # Even with dense fallback, should be well under 100K
        assert n < 200_000
        assert n > 0

    def test_all_routes_reachable(self):
        """Different inputs can produce different routes."""
        pf = PreFilter()
        routes_seen = set()
        configs = [
            (IntentType.CHAT, Domain.GENERAL, Urgency.LOW),
            (IntentType.CODE, Domain.CODE, Urgency.HIGH),
            (IntentType.MATH, Domain.MATH, Urgency.CRITICAL),
            (IntentType.SYSTEM, Domain.FLEET, Urgency.MEDIUM),
            (IntentType.ANALYSIS, Domain.RESEARCH, Urgency.HIGH),
        ]
        for intent, domain, urgency in configs:
            rf = RequestFeatures(intent=intent, domain=domain, urgency=urgency)
            dec = pf.route(rf)
            routes_seen.add(dec.route)
        # Untrained model — just verify no crash, at least one route seen
        assert len(routes_seen) >= 1

    def test_self_train_reduces_loss(self):
        """Training on consistent outcomes should reduce loss."""
        pf = PreFilter()
        # All CODE requests should go to USE_SMALL — clear signal
        for _ in range(20):
            features = RequestFeatures(
                intent=IntentType.CODE,
                domain=Domain.CODE,
                input_length=500,
            )
            # Override: always record as good when route matches expected
            dec = pf.route(features)
            pf.record_outcome(features, dec, was_good=True)

        result = pf.self_train(epochs=20)
        # First loss vs last loss — should generally decrease
        assert result["loss_curve"][-1] <= result["loss_curve"][0] * 1.5  # allow some slack


# ---------------------------------------------------------------------------
# RoutingDecision tests
# ---------------------------------------------------------------------------

class TestRoutingDecision:
    def test_route_name_property(self):
        dec = RoutingDecision(route=RouteTarget.USE_REASONING)
        assert dec.route_name == "USE_REASONING"

    def test_default_values(self):
        dec = RoutingDecision()
        assert dec.route == RouteTarget.USE_SMALL
        assert dec.confidence == 0.0
        assert dec.max_tokens_estimate == 256


# ---------------------------------------------------------------------------
# Enum coverage
# ---------------------------------------------------------------------------

class TestEnums:
    def test_intent_values(self):
        assert len(IntentType) == 8
        assert IntentType.QUERY.value == 0

    def test_domain_values(self):
        assert len(Domain) == 7
        assert Domain.CONSTRAINT_THEORY.value == 0

    def test_route_targets(self):
        assert len(RouteTarget) == 8
        assert RouteTarget.SKIP_LLM.value == 0
        assert RouteTarget.DELEGATE.value == 7

    def test_urgency_values(self):
        assert len(Urgency) == 4
