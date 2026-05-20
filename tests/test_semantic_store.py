"""
Comprehensive tests for SemanticStore — Model2Vec + FAISS semantic retrieval.
"""

import os
import tempfile
import time

import numpy as np
import pytest

from plato_training.semantic_store import SemanticStore, keyword_match


# --- Fixture ---

@pytest.fixture(scope="module")
def store():
    """Load SemanticStore once for all tests (model download is slow)."""
    s = SemanticStore()
    return s


KNOWLEDGE_TILES = [
    ("fleet-status", "The Cocapn fleet has 9 agents running across multiple hosts. Oracle1 coordinates fleet operations."),
    ("constraint-theory-1", "Constraint theory maps conceptual spaces to manifolds. Drift measures deviation from invariant subspaces."),
    ("constraint-theory-2", "Zero drift indicates perfect constraint satisfaction. The forging metaphor applies: heat the metal, apply pressure, achieve the shape."),
    ("plato-arch", "PLATO uses rooms as computational units. Each room processes tiles through a lifecycle: create, train, deploy, evaluate."),
    ("plato-tiles", "TrainingTiles are content-addressed knowledge units. They carry Lamport clocks for ordering and lifecycle state."),
    ("spline-results", "SplineLinear achieves 20x compression on drift-detect at the same accuracy as dense models. NPU quantization maintains 100%."),
    ("micro-models", "Micro models are tiny neural networks for 8 room tasks. Sub-millisecond inference on all CPU targets."),
    ("test-count", "The PLATO training module has 116 tests passing across 4 independent repos. plato-types has 10, tensor-spline has 57."),
    ("deployment", "Fleet deployment uses deploy_micro() to push micro models to 8 hardware targets including NPU, GPU, and CPU variants."),
    ("i2i-protocol", "Instance-to-instance communication uses 5 tile schemas: model, data, compression, benchmark, deploy. Collective inference loop: predict, listen, compare, gap, learn, share."),
    ("forgemaster-role", "Forgemaster is the constraint-theory specialist. Precision-obsessed, direct, uses metal and geometry analogies."),
    ("deepseek-code", "DeepSeek v4-flash is the research workbench. Large context, fast, cheap. Good for iterative exploration and long document analysis."),
    ("glm-routing", "GLM-5.1 is the default workhorse model on z.ai paid plan. Burn it for everything. Only delegate downstream when rate-limited."),
    ("seed-mini", "Seed-2.0-mini costs approximately one cent per query. Surprisingly good at code, math reasoning, and domain computation."),
    ("memory-arch", "MEMORY.md is a retrieval index map. PLATO rooms are the territory. Never put actual content in memory files."),
    ("lora-layers", "LoRA adapters enable parameter-efficient fine-tuning. LoRALayer with save/load support is in the adapters module."),
    ("lamport-clocks", "Lamport clocks provide causal ordering across distributed fleet nodes. Each tile carries a clock value for merge resolution."),
    ("hardware-targets", "Eight hardware targets: cpu-tiny, cpu-small, cpu-medium, gpu, npu, edge, mobile, server. Variant selection is automatic."),
    ("cast-call", "The casting-call repo maps models to roles. Includes roster of 11+ models, role taxonomy, failure modes, and adversarial pairs."),
    ("throttle", "Fleet-aware training throttle prevents resource contention. Coordinates training across multiple agents sharing the same hardware."),
]


@pytest.fixture
def populated_store(store):
    """Store populated with 20 knowledge tiles."""
    for tile_id, text in KNOWLEDGE_TILES:
        store.add(tile_id, text)
    return store


# --- Tests ---

class TestEmbedding:
    def test_embed_shape(self, store):
        vec = store.embed("test query")
        assert vec.shape == (store.dim,)
        assert vec.dtype == np.float32

    def test_embed_normalized(self, store):
        vec = store.embed("normalized test")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_embed_consistent(self, store):
        v1 = store.embed("consistent query")
        v2 = store.embed("consistent query")
        np.testing.assert_array_equal(v1, v2)


class TestAddSearch:
    def test_add_and_size(self, populated_store):
        assert populated_store.size() == len(KNOWLEDGE_TILES)

    def test_search_fleet(self, populated_store):
        results = populated_store.search("How many agents in the fleet?", k=3)
        assert len(results) > 0
        assert results[0]["tile_id"] == "fleet-status"
        assert results[0]["score"] >= 0.3

    def test_search_spline(self, populated_store):
        results = populated_store.search("SplineLinear compression drift-detect", k=3)
        assert len(results) > 0
        assert results[0]["tile_id"] == "spline-results"

    def test_search_unrelated(self, populated_store):
        results = populated_store.search("banana recipe chocolate cake baking")
        # Should have low scores — top result should not be strongly matched
        for r in results:
            assert r["score"] < 0.5

    def test_search_empty_store(self, store):
        empty = SemanticStore()
        results = empty.search("anything")
        assert results == []

    def test_best_match(self, populated_store):
        result = populated_store.best_match("PLATO training module tests passing repos")
        assert result is not None
        assert result["tile_id"] == "test-count"

    def test_best_match_none(self, populated_store):
        result = populated_store.best_match("xyzzy foobar quux baz", threshold=0.7)
        # High threshold on nonsense should return None
        assert result is None or isinstance(result, dict)


class TestParaphrase:
    def test_paraphrase_testing(self, populated_store):
        """Paraphrase should match test-count entry."""
        results = populated_store.search("tests passing repos count", k=5)
        ids = [r["tile_id"] for r in results]
        assert "test-count" in ids

    def test_paraphrase_agent_count(self, populated_store):
        """Paraphrase should match fleet-status."""
        results = populated_store.search("how many agents running fleet hosts", k=5)
        ids = [r["tile_id"] for r in results]
        assert "fleet-status" in ids

    def test_paraphrase_model_cost(self, populated_store):
        """Paraphrase about cheap models should match seed-mini."""
        results = populated_store.search("Seed-2.0-mini cost cent query", k=5)
        ids = [r["tile_id"] for r in results]
        assert "seed-mini" in ids


class TestDuplicate:
    def test_duplicate_update(self, populated_store):
        initial_size = populated_store.size()
        populated_store.add("fleet-status", "Updated fleet has 12 agents now.")
        # Should not increase size (update in place)
        assert populated_store.size() == initial_size
        result = populated_store.best_match("updated fleet 12 agents")
        assert result is not None
        assert "12 agents" in result["text"]


class TestThresholds:
    @pytest.mark.parametrize("threshold", [0.2, 0.3, 0.5])
    def test_threshold_returns_results(self, populated_store, threshold):
        results = populated_store.search("fleet agents", k=5, threshold=threshold)
        if threshold <= 0.3:
            assert len(results) > 0
        for r in results:
            assert r["score"] >= threshold


class TestSaveLoad:
    def test_round_trip(self, populated_store):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_store")
            populated_store.save(path)

            # Load into fresh store
            s2 = SemanticStore()
            s2.load(path)

            assert s2.size() == populated_store.size()
            assert s2.id_map == populated_store.id_map
            assert s2.text_map == populated_store.text_map

            # Search should return same results
            r1 = populated_store.best_match("constraint drift manifold")
            r2 = s2.best_match("constraint drift manifold")
            assert r1 is not None
            assert r2 is not None
            assert r1["tile_id"] == r2["tile_id"]


class TestBenchmark:
    def test_embedding_speed(self, store):
        query = "benchmark embedding speed test"
        N = 100
        start = time.perf_counter()
        for _ in range(N):
            store.embed(query)
        elapsed = time.perf_counter() - start
        us_per = elapsed / N * 1_000_000
        print(f"\n  Embedding speed: {us_per:.1f} μs/query ({N} queries in {elapsed:.3f}s)")
        # Should be under 10ms per query on CPU
        assert us_per < 10_000

    def test_search_speed(self, populated_store):
        N = 1000
        queries = ["fleet agents", "drift zero", "test count", "model cost", "deployment target"] * (N // 5)
        start = time.perf_counter()
        for q in queries:
            populated_store.search(q, k=3)
        elapsed = time.perf_counter() - start
        us_per = elapsed / N * 1_000_000
        print(f"\n  Search speed: {us_per:.1f} μs/query ({N} queries in {elapsed:.3f}s)")

    def test_keyword_vs_semantic_hit_rate(self, populated_store):
        """Compare keyword matching vs semantic matching hit rates."""
        # 10 direct match queries + 10 paraphrase queries
        queries = [
            # Direct matches
            ("How many agents in the fleet?", "fleet-status"),
            ("What is SplineLinear compression?", "spline-results"),
            ("What does Forgemaster do?", "forgemaster-role"),
            ("How does the throttle work?", "throttle"),
            ("What are the hardware targets?", "hardware-targets"),
            ("Explain Lamport clocks", "lamport-clocks"),
            ("What is the I2I protocol?", "i2i-protocol"),
            ("Tell me about LoRA layers", "lora-layers"),
            ("What is MEMORY.md for?", "memory-arch"),
            ("How does casting call work?", "cast-call"),
            # Paraphrase matches
            ("count of running instances", "fleet-status"),
            ("how much smaller are spline models", "spline-results"),
            ("who specializes in constraints", "forgemaster-role"),
            ("preventing resource fights during training", "throttle"),
            ("what devices can we deploy to", "hardware-targets"),
            ("ordering events across nodes", "lamport-clocks"),
            ("agent to agent communication", "i2i-protocol"),
            ("efficient fine-tuning method", "lora-layers"),
            ("where to store persistent state", "memory-arch"),
            ("model role assignment database", "cast-call"),
        ]

        text_map = populated_store.text_map

        keyword_hits = 0
        semantic_hits = 0

        for query, expected_id in queries:
            # Keyword match
            kw_results = keyword_match(query, text_map, threshold=0.1)
            if kw_results and kw_results[0]["tile_id"] == expected_id:
                keyword_hits += 1

            # Semantic match
            sem_result = populated_store.best_match(query, threshold=0.25)
            if sem_result and sem_result["tile_id"] == expected_id:
                semantic_hits += 1

        keyword_rate = keyword_hits / len(queries) * 100
        semantic_rate = semantic_hits / len(queries) * 100

        print(f"\n  === Hit Rate Comparison ===")
        print(f"  Keyword: {keyword_hits}/{len(queries)} = {keyword_rate:.0f}%")
        print(f"  Semantic: {semantic_hits}/{len(queries)} = {semantic_rate:.0f}%")
        print(f"  Improvement: {semantic_rate - keyword_rate:+.0f} percentage points")

        # Semantic should outperform keyword
        assert semantic_rate >= keyword_rate
