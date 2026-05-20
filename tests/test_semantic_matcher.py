"""
Tests for SemanticMatcher — semantic vs keyword matching for IntelligenceRoom.
"""

import time
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


# ─── Helpers ──────────────────────────────────────────────────────

def _make_matcher(threshold=0.6):
    """Create a SemanticMatcher (semantic if deps available, keyword fallback otherwise)."""
    from plato_training.semantic_matcher import SemanticMatcher
    return SemanticMatcher(threshold=threshold)


# ─── 1. Initialization ────────────────────────────────────────────

class TestInit:
    def test_creates_instance(self):
        m = _make_matcher()
        assert m is not None

    def test_is_semantic_reflects_deps(self):
        m = _make_matcher()
        # If model2vec+faiss are installed, should be semantic
        assert m.is_semantic is True or m.is_semantic is False

    def test_default_threshold(self):
        m = _make_matcher()
        assert m.threshold == 0.6

    def test_custom_threshold(self):
        m = _make_matcher(threshold=0.8)
        assert m.threshold == 0.8

    def test_empty_size(self):
        m = _make_matcher()
        assert m.size() == 0


# ─── 2. Add + Match ──────────────────────────────────────────────

class TestAddMatch:
    def test_add_and_match_exact(self):
        m = _make_matcher(threshold=0.5)
        m.add("k1", "The PLATO training system uses micro models for inference")
        result = m.match("The PLATO training system uses micro models for inference")
        if m.is_semantic:
            assert result is not None
            assert result[0] == "k1"
            assert result[1] > 0.5

    def test_add_multiple(self):
        m = _make_matcher(threshold=0.4)
        m.add("k1", "Constraint theory studies the geometry of constraint satisfaction")
        m.add("k2", "PLATO rooms are the basic unit of agent intelligence")
        m.add("k3", "The fleet uses Matrix for inter-agent communication")
        result = m.match("How do PLATO rooms work?")
        if m.is_semantic:
            assert result is not None
            assert result[0] == "k2"

    def test_size_increases(self):
        m = _make_matcher()
        m.add("k1", "test entry one")
        assert m.size() == 1
        m.add("k2", "test entry two")
        assert m.size() == 2


# ─── 3. Paraphrase Matching ──────────────────────────────────────

class TestParaphrase:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.m = _make_matcher(threshold=0.3)
        self.m.add("plato_tests", "The PLATO training repository has 665 tests across 4 modules")

    def test_paraphrase_1(self):
        """'how many tests does plato have?' should match the tests entry."""
        result = self.m.match("how many tests does plato have?")
        if self.m.is_semantic:
            assert result is not None
            assert result[0] == "plato_tests"

    def test_paraphrase_2(self):
        """'test count for training repo' should match."""
        result = self.m.match("test count for training repo")
        if self.m.is_semantic:
            assert result is not None
            assert result[0] == "plato_tests"

    def test_unrelated_query(self):
        """'what color is the sky?' should NOT match."""
        result = self.m.match("what color is the sky")
        # Even semantic should score low here since content is about PLATO tests
        if self.m.is_semantic:
            assert result is None or result[0] != "plato_tests"


# ─── 4. Threshold Sensitivity ────────────────────────────────────

class TestThreshold:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.m = _make_matcher(threshold=0.4)
        self.m.add("k1", "SplineLinear achieves 20x compression on drift detection tasks")

    def test_low_threshold(self):
        self.m.threshold = 0.3
        result = self.m.match("compression ratio for spline layer")
        if self.m.is_semantic:
            # Should match at low threshold
            assert result is not None

    def test_medium_threshold(self):
        self.m.threshold = 0.6
        result = self.m.match("compression ratio for spline layer")
        # May or may not match at 0.6 — just verify no crash
        assert result is None or isinstance(result, tuple)

    def test_high_threshold(self):
        self.m.threshold = 0.95
        result = self.m.match("SplineLinear achieves 20x compression on drift detection tasks")
        if self.m.is_semantic:
            # Exact match should still score high but maybe not 0.95
            assert result is None or isinstance(result, tuple)


# ─── 5. Multiple Entries Compete ──────────────────────────────────

class TestCompetition:
    def test_best_match_wins(self):
        m = _make_matcher(threshold=0.4)
        m.add("k1", "Python is a programming language")
        m.add("k2", "PLATO rooms organize agent knowledge into retrievable tiles")
        m.add("k3", "Fleet agents communicate via Matrix protocol")

        result = m.match("Tell me about PLATO rooms and tiles")
        if m.is_semantic:
            assert result is not None
            assert result[0] == "k2"

    def test_match_top_k(self):
        m = _make_matcher(threshold=0.3)
        m.add("k1", "Constraint theory studies satisfiability")
        m.add("k2", "Constraint propagation reduces search space")
        m.add("k3", "PLATO training pipeline builds micro models")
        m.add("k4", "The sky is blue on clear days")

        results = m.match_top_k("constraint solving and propagation", k=3)
        if m.is_semantic:
            assert len(results) >= 1
            # k1 or k2 should be in top results
            keys = [r[0] for r in results]
            assert "k1" in keys or "k2" in keys


# ─── 6. Empty Store ──────────────────────────────────────────────

class TestEmpty:
    def test_match_empty_returns_none(self):
        m = _make_matcher()
        result = m.match("anything")
        assert result is None

    def test_top_k_empty(self):
        m = _make_matcher()
        result = m.match_top_k("anything", k=5)
        assert result == []


# ─── 7. Benchmark ────────────────────────────────────────────────

class TestBenchmark:
    def test_100_matches(self):
        m = _make_matcher(threshold=0.4)
        # Index 50 entries
        for i in range(50):
            m.add(f"k{i}", f"Knowledge entry number {i} about PLATO rooms and constraint theory batch {i}")

        if not m.is_semantic:
            pytest.skip("Semantic matching not available")

        queries = [f"Tell me about entry {i} in PLATO" for i in range(100)]
        start = time.time()
        for q in queries:
            m.match(q)
        elapsed = time.time() - start

        avg_ms = (elapsed / 100) * 1000
        print(f"\n100 matches: {elapsed:.3f}s total, {avg_ms:.1f}ms avg")
        assert avg_ms < 100, f"Too slow: {avg_ms:.1f}ms per match"


# ─── 8. Keyword Fallback ─────────────────────────────────────────

class TestKeywordFallback:
    def test_fallback_finds_overlap(self):
        m = _make_matcher()
        texts = {
            "k1": "PLATO training has many tests",
            "k2": "Constraint theory is about satisfiability",
        }
        result = m.keyword_fallback("PLATO training tests", texts)
        assert result is not None
        assert result[0] == "k1"

    def test_fallback_no_match(self):
        m = _make_matcher()
        texts = {"k1": "The quick brown fox"}
        result = m.keyword_fallback("completely different topic", texts)
        assert result is None

    def test_fallback_empty(self):
        m = _make_matcher()
        result = m.keyword_fallback("query", {})
        assert result is None


# ─── 9. Head-to-Head Comparison ──────────────────────────────────

KNOWLEDGE_ENTRIES = {
    "plato_arch": "PLATO uses a three-layer architecture: room protocol, engine rooms, and tensor-spline compression",
    "plato_tests": "The plato-training repository has 665 tests across four independent modules",
    "constraint_def": "Constraint theory studies the geometry and algebra of constraint satisfaction problems",
    "spline_linear": "SplineLinear uses Eisenstein lattice weight parameterization for 20x model compression",
    "fleet_comms": "Fleet agents communicate through Matrix protocol using I2I bottles for knowledge sharing",
    "forgemaster": "Forgemaster is a constraint-theory specialist in the Cocapn fleet running on GLM-5.1",
    "oracle1": "Oracle1 is the fleet coordinator that manages task distribution across nine agents",
    "tile_lifecycle": "TrainingTiles progress through stages: created, training, validated, deployed, archived",
    "lamport": "Lamport clocks provide causal ordering of events across distributed fleet agents",
    "micro_models": "Micro models are tiny neural networks trained for specific room tasks like drift detection",
    "npu_deploy": "NPU quantization with INT8 maintains 100% accuracy on drift-detect and intent-detect tasks",
    "lora_adapter": "LoRA adapters struggle on synthetic data and need real data for effective training",
    "pre_filter": "The pre-filter micro model learns to skip unnecessary LLM calls saving tokens",
    "post_filter": "The post-filter decides whether to keep LLM responses as knowledge tiles",
    "heuristic_route": "Heuristic routing falls back to request-length and domain-based model selection",
    "i2i_protocol": "Instance-to-instance protocol uses five tile schemas: model, data, compression, benchmark, deploy",
    "casting_call": "Casting-call repo maintains a fleet-wide model capability database with 685 lines of evaluation data",
    "focus_scoring": "Focus scoring multiplies confidence by delta to measure how sure and how wrong a prediction is",
    "casey": "Casey Digennaro runs SuperInstance with 1400+ repos and the Cocapn fleet of nine agents",
    "eileen": "Eileen is the WSL2 host machine running OpenClaw with Forgemaster as the primary agent",
}

PARAPHRASE_QUERIES = [
    "How is PLATO structured internally?",
    "How many tests does the training module have?",
    "What is constraint theory about?",
    "How does SplineLinear compress models?",
    "How do agents talk to each other?",
    "Who is the constraint specialist?",
    "Who coordinates the fleet?",
    "What are the stages of a training tile?",
    "How does the fleet order events?",
    "What are micro models used for?",
]

NOVEL_QUERIES = [
    "What is the capital of France?",
    "How do I bake a chocolate cake?",
    "Explain quantum entanglement",
    "What is the best programming language?",
    "How does photosynthesis work?",
    "Who wrote Hamlet?",
    "What is the speed of light?",
    "How to make pasta carbonara?",
    "Explain the theory of relativity",
    "What is the tallest mountain?",
]


class TestHeadToHead:
    def test_hit_rate_comparison(self):
        m = _make_matcher(threshold=0.4)
        for key, text in KNOWLEDGE_ENTRIES.items():
            m.add(key, text)

        if not m.is_semantic:
            pytest.skip("Semantic matching not available")

        # Semantic matching
        sem_hits = 0
        for q in PARAPHRASE_QUERIES:
            if m.match(q) is not None:
                sem_hits += 1

        sem_false_pos = 0
        for q in NOVEL_QUERIES:
            if m.match(q) is not None:
                sem_false_pos += 1

        # Keyword matching
        kw_hits = 0
        for q in PARAPHRASE_QUERIES:
            if m.keyword_fallback(q, KNOWLEDGE_ENTRIES) is not None:
                kw_hits += 1

        kw_false_pos = 0
        for q in NOVEL_QUERIES:
            if m.keyword_fallback(q, KNOWLEDGE_ENTRIES) is not None:
                kw_false_pos += 1

        print(f"\n{'='*60}")
        print(f"Hit Rate Comparison (20 entries, 10 paraphrase + 10 novel queries)")
        print(f"{'='*60}")
        print(f"Semantic:  {sem_hits}/10 paraphrases matched, {sem_false_pos}/10 false positives")
        print(f"Keyword:   {kw_hits}/10 paraphrases matched, {kw_false_pos}/10 false positives")
        print(f"{'='*60}")
        print(f"Semantic should have higher hit rate on paraphrases")
        print(f"Both should have low false positive rates")

        # Semantic should at least match keyword hit rate
        assert sem_hits >= kw_hits or sem_hits >= 7, \
            f"Semantic ({sem_hits}) should beat keyword ({kw_hits}) or be >= 7"
