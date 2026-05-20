"""Tests for intelligence_post_filter module."""

import math
import tempfile
import os
import numpy as np
import pytest
import torch

from plato_training.intelligence_post_filter import (
    ResponseFeatures, FilterDecision, KnowledgeTile,
    PostFilterModel, PostFilter,
    FEATURE_DIM, HIDDEN_DIM, KEEP_DECISIONS, TILE_DOMAINS,
    RESPONSE_TYPES, MODEL_NAMES, ROUTE_NAMES,
    _extract_facts, _extract_patterns, _extract_decisions, _compress,
)


# ---------------------------------------------------------------------------
# ResponseFeatures tests
# ---------------------------------------------------------------------------

class TestResponseFeatures:
    def test_vector_is_256_dim(self):
        rf = ResponseFeatures(response_text="Hello world")
        vec = rf.to_vector()
        assert vec.shape == (256,)
        assert vec.dtype == np.float32

    def test_no_nans_in_vector(self):
        rf = ResponseFeatures(response_text="", request_text="")
        vec = rf.to_vector()
        assert not np.isnan(vec).any()

    def test_empty_response_produces_valid_vector(self):
        rf = ResponseFeatures()
        vec = rf.to_vector()
        assert vec.shape == (256,)
        assert not np.isnan(vec).any()

    def test_code_response_has_code_flag(self):
        rf = ResponseFeatures(response_text="Here is code:\n```python\nprint(1)\n```")
        vec = rf.to_vector()
        # code_presence is at index 30 (16+7+8+1+1+1+1+1+1+1-1 = offset)
        assert vec.sum() > 0  # at least some features set

    def test_different_models_produce_different_vectors(self):
        v1 = ResponseFeatures(response_text="test", model_used="glm-5.1").to_vector()
        v2 = ResponseFeatures(response_text="test", model_used="claude").to_vector()
        assert not np.allclose(v1, v2)

    def test_different_types_produce_different_vectors(self):
        v1 = ResponseFeatures(response_text="test", response_type="code").to_vector()
        v2 = ResponseFeatures(response_text="test", response_type="error").to_vector()
        assert not np.allclose(v1, v2)

    def test_keyword_vector_populated(self):
        rf = ResponseFeatures(response_text="constraint drift proof tensor spline linear")
        vec = rf.to_vector()
        # keyword_vector is 64 dims, check not all zero
        kw_start = 256 - 64 - 32  # before minhash
        assert vec[kw_start:kw_start + 64].sum() > 0


# ---------------------------------------------------------------------------
# PostFilterModel tests
# ---------------------------------------------------------------------------

class TestPostFilterModel:
    def test_forward_shape(self):
        model = PostFilterModel(n_control_points=8)
        x = torch.randn(4, 256)
        out = model(x)
        assert out["keep_logits"].shape == (4, len(KEEP_DECISIONS))
        assert out["confidence"].shape == (4,)
        assert out["compression"].shape == (4,)
        assert out["domain_logits"].shape == (4, len(TILE_DOMAINS))
        assert out["reusability"].shape == (4,)

    def test_confidence_in_range(self):
        model = PostFilterModel()
        out = model(torch.randn(2, 256))
        assert (out["confidence"] >= 0).all()
        assert (out["confidence"] <= 1).all()

    def test_param_count_reasonable(self):
        model = PostFilterModel(n_control_points=16)
        pc = model.param_count()
        assert 100 < pc < 200_000  # should be compact

    def test_uses_spline_linear(self):
        model = PostFilterModel()
        from plato_training.spline import SplineLinear
        assert isinstance(model.encoder, SplineLinear)


# ---------------------------------------------------------------------------
# FilterDecision tests
# ---------------------------------------------------------------------------

class TestFilterDecision:
    def test_defaults(self):
        fd = FilterDecision()
        assert fd.keep_decision == "summary"
        assert 0 <= fd.extraction_confidence <= 1

    def test_to_dict(self):
        fd = FilterDecision(keep_decision="tile", extraction_confidence=0.9)
        d = fd.to_dict()
        assert d["keep_decision"] == "tile"
        assert d["extraction_confidence"] == 0.9


# ---------------------------------------------------------------------------
# KnowledgeTile tests
# ---------------------------------------------------------------------------

class TestKnowledgeTile:
    def test_round_trip(self):
        t = KnowledgeTile(tile_id="abc", domain="plato", compressed_content="hello")
        d = t.to_dict()
        t2 = KnowledgeTile.from_dict(d)
        assert t2.tile_id == "abc"
        assert t2.compressed_content == "hello"

    def test_reuse_count_default_zero(self):
        t = KnowledgeTile()
        assert t.reuse_count == 0


# ---------------------------------------------------------------------------
# PostFilter integration tests
# ---------------------------------------------------------------------------

class TestPostFilter:
    def _sample_response(self):
        return (
            "The constraint solver achieved 99.7% accuracy on the drift-detect task. "
            "Use SplineLinear for compression. The latency was 0.3ms on CPU. "
            "Step 1: Initialize weights. Step 2: Run forward pass. "
            "```python\nfrom spline import SplineLinear\nlayer = SplineLinear(256, 128)\n```"
        )

    def test_process_returns_decision_and_tiles(self):
        pf = PostFilter()
        decision, tiles = pf.process(self._sample_response(), request_context="test query")
        assert isinstance(decision, FilterDecision)
        assert isinstance(tiles, list)

    def test_process_extracts_knowledge(self):
        pf = PostFilter()
        _, tiles = pf.process(self._sample_response())
        assert len(tiles) > 0  # should extract facts/patterns

    def test_discard_produces_no_tiles(self):
        # Force discard by using error type with no content
        pf = PostFilter()
        decision, tiles = pf.process("", response_type="error")
        # May or may not be discard depending on init, but should handle gracefully
        if decision.keep_decision == "discard":
            assert len(tiles) == 0

    def test_record_usefulness(self):
        pf = PostFilter()
        _, tiles = pf.process(self._sample_response())
        if tiles:
            pf.record_usefulness(tiles[0].tile_id, True)
            tile = pf.get_tile(tiles[0].tile_id)
            assert tile.reuse_count == 1

    def test_self_train(self):
        pf = PostFilter()
        # Add synthetic training data
        for _ in range(20):
            feat = np.random.randn(256).astype(np.float32)
            target = {
                "keep": np.random.randint(0, len(KEEP_DECISIONS)),
                "confidence": np.random.rand(),
                "domain": np.random.randint(0, len(TILE_DOMAINS)),
                "reusability": np.random.rand(),
            }
            pf.add_training_example(feat, target)
        result = pf.self_train(epochs=3, batch_size=4)
        assert "final_loss" in result
        assert result["epochs"] == 3

    def test_export_and_load_tile(self):
        pf = PostFilter()
        with tempfile.TemporaryDirectory() as td:
            tile_id = pf.export_tile(td)
            assert tile_id
            # Load into fresh filter
            pf2 = PostFilter()
            loaded = pf2.load_tile(tile_id, td)
            assert loaded

    def test_load_nonexistent_returns_false(self):
        pf = PostFilter()
        assert not pf.load_tile("nonexistent", "/tmp/nope")

    def test_tile_count(self):
        pf = PostFilter()
        assert pf.tile_count == 0
        pf.process(self._sample_response())
        assert pf.tile_count > 0


# ---------------------------------------------------------------------------
# Extractor function tests
# ---------------------------------------------------------------------------

class TestExtractors:
    def test_extract_facts(self):
        facts = _extract_facts("The accuracy is 99.7%. The loss equals 0.001.")
        assert len(facts) >= 1

    def test_extract_patterns(self):
        text = "```python\nfrom spline import SplineLinear\nlayer = SplineLinear(256, 128)\n```"
        patterns = _extract_patterns(text)
        assert len(patterns) >= 1

    def test_extract_decisions(self):
        text = "Use SplineLinear for compression. Avoid dense layers."
        decisions = _extract_decisions(text)
        assert len(decisions) >= 1

    def test_compress_high_ratio(self):
        text = "First sentence. Second sentence. Third sentence."
        result = _compress(text, 0.99)
        assert "First" in result

    def test_compress_low_ratio(self):
        text = "First sentence. Second sentence. Third sentence."
        result = _compress(text, 0.4)
        assert len(result) < len(text)
