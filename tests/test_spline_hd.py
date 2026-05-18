"""
Tests for high-dimensional Tensor-Spline module.

Tests cover:
- BlockDecomposition: splitting, overlapping, reconstruction
- SplineLinearHD: forward pass, compression, per-block stats
- AdaptiveCompression: entropy measurement, strategy selection
- EmbeddingBenchmark: synthetic data, full benchmark pipeline
"""

import math
import pytest
import torch
import torch.nn as nn

from plato_training.spline_hd import (
    AdaptiveCompression,
    AdaptiveLinear,
    BlockDecomposition,
    EmbeddingBenchmark,
    SplineLinearHD,
)


# ---------------------------------------------------------------------------
# BlockDecomposition tests
# ---------------------------------------------------------------------------

class TestBlockDecomposition:

    def test_basic_64_into_4_blocks_of_16(self):
        """64-dim input → 4 non-overlapping blocks of 16."""
        bd = BlockDecomposition(input_dim=64, block_size=16)
        assert bd.n_blocks == 4
        assert bd.block_dims() == [16, 16, 16, 16]

    def test_768_into_48_blocks(self):
        """768-dim input → 48 non-overlapping blocks of 16."""
        bd = BlockDecomposition(input_dim=768, block_size=16)
        assert bd.n_blocks == 48
        assert all(d == 16 for d in bd.block_dims())

    def test_uneven_split(self):
        """65-dim with block_size=16 → last block is 1 dim."""
        bd = BlockDecomposition(input_dim=65, block_size=16)
        assert sum(bd.block_dims()) == 65
        assert bd.block_dims()[-1] == 1  # 4*16=64, +1

    def test_overlap_blocks(self):
        """Overlapping blocks share boundary dimensions."""
        bd = BlockDecomposition(input_dim=64, block_size=16, overlap=2)
        # Step = block_size - overlap = 14
        # Blocks: [0,16), [14,30), [28,44), [42,58), [56,64)
        assert bd.n_blocks == 5
        # Verify overlap exists
        shared = bd.overlapping_indices()
        # First block shares dims 14,15 with second block
        assert 14 in shared[0]
        assert 15 in shared[0]

    def test_extract_and_reconstruct_no_overlap(self):
        """Extract then reconstruct round-trips for non-overlapping."""
        bd = BlockDecomposition(input_dim=64, block_size=16)
        x = torch.randn(8, 64)
        blocks = bd.extract(x)
        assert len(blocks) == 4
        assert blocks[0].shape == (8, 16)

        recon = bd.reconstruct(blocks)
        assert torch.allclose(x, recon, atol=1e-6)

    def test_extract_and_reconstruct_with_overlap(self):
        """Reconstruct with overlap averages overlapping regions."""
        bd = BlockDecomposition(input_dim=64, block_size=20, overlap=4)
        x = torch.randn(8, 64)
        blocks = bd.extract(x)
        recon = bd.reconstruct(blocks, method="mean")
        assert recon.shape == (8, 64)

    def test_overlapping_indices_empty_when_no_overlap(self):
        """No overlapping dims when overlap=0."""
        bd = BlockDecomposition(input_dim=64, block_size=16, overlap=0)
        for idx_list in bd.overlapping_indices():
            assert len(idx_list) == 0

    def test_block_size_1(self):
        """Each dim is its own block when block_size=1."""
        bd = BlockDecomposition(input_dim=8, block_size=1)
        assert bd.n_blocks == 8
        assert all(d == 1 for d in bd.block_dims())

    def test_repr(self):
        bd = BlockDecomposition(input_dim=128, block_size=16)
        r = repr(bd)
        assert "input_dim=128" in r
        assert "block_size=16" in r
        assert "n_blocks=8" in r

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            BlockDecomposition(input_dim=64, block_size=16, overlap=16)

    def test_invalid_block_size_raises(self):
        with pytest.raises(ValueError, match="block_size"):
            BlockDecomposition(input_dim=64, block_size=0)

    def test_slices_cover_full_range(self):
        """All slices together cover every dimension exactly once (no overlap)."""
        bd = BlockDecomposition(input_dim=100, block_size=16)
        all_dims = set()
        for start, end in bd.slices:
            for d in range(start, end):
                all_dims.add(d)
        assert all_dims == set(range(100))

    def test_large_dim(self):
        """1024-dim decomposition works."""
        bd = BlockDecomposition(input_dim=1024, block_size=32)
        assert bd.n_blocks == 32
        assert sum(bd.block_dims()) == 1024


# ---------------------------------------------------------------------------
# SplineLinearHD tests
# ---------------------------------------------------------------------------

class TestSplineLinearHD:

    def test_forward_64_dim(self):
        """Forward pass with 64-dim input."""
        layer = SplineLinearHD(
            in_features=64, out_features=10,
            block_size=16, n_control_points=8,
        )
        x = torch.randn(4, 64)
        out = layer(x)
        assert out.shape == (4, 10)

    def test_forward_768_dim(self):
        """Forward pass with 768-dim input (NLP embedding scale)."""
        layer = SplineLinearHD(
            in_features=768, out_features=10,
            block_size=16, n_control_points=16,
        )
        x = torch.randn(2, 768)
        out = layer(x)
        assert out.shape == (2, 10)

    def test_forward_128_dim_with_overlap(self):
        """Forward pass with overlapping blocks."""
        layer = SplineLinearHD(
            in_features=128, out_features=32,
            block_size=32, overlap=4, n_control_points=8,
        )
        x = torch.randn(8, 128)
        out = layer(x)
        assert out.shape == (8, 32)

    def test_compression_ratio(self):
        """Compression ratio > 1 for reasonable configs."""
        layer = SplineLinearHD(
            in_features=256, out_features=64,
            block_size=16, n_control_points=8,
        )
        cr = layer.compression_ratio()
        assert cr > 1.0

    def test_num_params_less_than_dense(self):
        """SplineLinearHD uses fewer params than equivalent dense."""
        layer = SplineLinearHD(
            in_features=512, out_features=64,
            block_size=16, n_control_points=8,
        )
        assert layer.num_trainable_params() < layer.num_equivalent_dense_params()

    def test_per_block_compression(self):
        """Per-block compression stats are returned."""
        layer = SplineLinearHD(
            in_features=64, out_features=10,
            block_size=16, n_control_points=8,
        )
        stats = layer.per_block_compression()
        assert len(stats) == 4  # 64/16 = 4 blocks
        for s in stats:
            assert "compression" in s
            assert "block_dim" in s
            assert s["compression"] >= 1.0

    def test_gradients_flow(self):
        """Gradients flow to control points through the block structure."""
        layer = SplineLinearHD(
            in_features=64, out_features=10,
            block_size=16, n_control_points=8,
        )
        x = torch.randn(4, 64)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        for block_layer in layer.block_layers:
            assert block_layer.control_values.grad is not None
            assert block_layer.control_values.grad.abs().sum() > 0

    def test_no_bias(self):
        """Works without bias."""
        layer = SplineLinearHD(
            in_features=64, out_features=10,
            block_size=16, n_control_points=8, bias=False,
        )
        x = torch.randn(4, 64)
        out = layer(x)
        assert out.shape == (4, 10)
        assert layer.bias is None

    def test_low_rank_mixing(self):
        """Low-rank mixing reduces parameters."""
        full = SplineLinearHD(
            in_features=128, out_features=10,
            block_size=16, n_control_points=8,
        )
        low = SplineLinearHD(
            in_features=128, out_features=10,
            block_size=16, n_control_points=8,
            mixing_rank=2,
        )
        assert low.num_trainable_params() < full.num_trainable_params()

    def test_bspline_basis(self):
        """Works with bspline basis."""
        layer = SplineLinearHD(
            in_features=64, out_features=10,
            block_size=16, n_control_points=16, basis="bspline",
        )
        x = torch.randn(4, 64)
        out = layer(x)
        assert out.shape == (4, 10)

    def test_gaussian_basis(self):
        """Works with gaussian basis."""
        layer = SplineLinearHD(
            in_features=64, out_features=10,
            block_size=16, n_control_points=8, basis="gaussian",
        )
        x = torch.randn(4, 64)
        out = layer(x)
        assert out.shape == (4, 10)

    def test_batch_input(self):
        """Handles various batch dimensions."""
        layer = SplineLinearHD(
            in_features=64, out_features=10,
            block_size=16, n_control_points=8,
        )
        x = torch.randn(3, 5, 64)  # batch + sequence
        out = layer(x)
        assert out.shape == (3, 5, 10)

    def test_different_block_sizes(self):
        """Block sizes 8, 16, 24, 32 all work for 128-dim."""
        for bs in [8, 16, 24, 32]:
            layer = SplineLinearHD(
                in_features=128, out_features=10,
                block_size=bs, n_control_points=8,
            )
            x = torch.randn(4, 128)
            out = layer(x)
            assert out.shape == (4, 10), f"Failed for block_size={bs}"


# ---------------------------------------------------------------------------
# AdaptiveCompression tests
# ---------------------------------------------------------------------------

class TestAdaptiveCompression:

    def _make_low_entropy_data(self, dim: int) -> torch.Tensor:
        """Data with low variance → low entropy."""
        base = torch.randn(1, dim) * 0.1  # tiny variance
        noise = torch.randn(200, dim) * 0.01
        return base + noise

    def _make_high_entropy_data(self, dim: int) -> torch.Tensor:
        """Data with high variance → high entropy."""
        return torch.randn(200, dim) * 5.0

    def test_entropy_measurement(self):
        """Entropy measurement returns one value per block."""
        bd = BlockDecomposition(input_dim=64, block_size=16)
        ac = AdaptiveCompression(bd)
        data = torch.randn(100, 64)
        entropies = ac.measure_entropy(data)
        assert len(entropies) == 4

    def test_low_entropy_detected(self):
        """Low-entropy data is detected correctly."""
        bd = BlockDecomposition(input_dim=64, block_size=16)
        ac = AdaptiveCompression(bd, aggressive_threshold=0.5)
        data = self._make_low_entropy_data(64)
        entropies = ac.measure_entropy(data)
        # Low variance data should have negative log-variance
        assert all(h < 0 for h in entropies)

    def test_high_entropy_detected(self):
        """High-entropy data is detected correctly."""
        bd = BlockDecomposition(input_dim=64, block_size=16)
        ac = AdaptiveCompression(bd, passthrough_threshold=2.0)
        data = self._make_high_entropy_data(64)
        entropies = ac.measure_entropy(data)
        # High variance data should have positive log-variance
        assert all(h > 0 for h in entropies)

    def test_strategy_selection_aggressive(self):
        """Low entropy → aggressive strategy."""
        bd = BlockDecomposition(input_dim=64, block_size=16)
        ac = AdaptiveCompression(bd, aggressive_threshold=10.0)
        strategies = ac.select_strategy([-10, -5, -1, 0.5])
        assert strategies[0] == "aggressive"
        assert strategies[1] == "aggressive"

    def test_strategy_selection_passthrough(self):
        """High entropy → passthrough strategy."""
        bd = BlockDecomposition(input_dim=64, block_size=16)
        ac = AdaptiveCompression(bd, passthrough_threshold=-1.0)
        strategies = ac.select_strategy([5.0, 10.0, 3.0, 1.0])
        assert all(s == "passthrough" for s in strategies)

    def test_strategy_selection_moderate(self):
        """Medium entropy → moderate strategy."""
        bd = BlockDecomposition(input_dim=64, block_size=16)
        ac = AdaptiveCompression(bd, aggressive_threshold=-1.0, passthrough_threshold=5.0)
        strategies = ac.select_strategy([0.0, 1.0, 2.0, 3.0])
        assert all(s == "moderate" for s in strategies)

    def test_control_point_recommendation(self):
        """Recommendations match strategy."""
        bd = BlockDecomposition(input_dim=64, block_size=16)
        ac = AdaptiveCompression(bd)
        strategies = ["aggressive", "moderate", "passthrough", "moderate"]
        recs = ac.recommend_control_points(strategies, base_control_points=16)
        assert recs[0] == 8   # aggressive: 16 // 2
        assert recs[1] == 16  # moderate
        assert recs[2] == 0   # passthrough
        assert recs[3] == 16  # moderate

    def test_build_adaptive_layer(self):
        """AdaptiveCompression.build_layer produces a working module."""
        bd = BlockDecomposition(input_dim=64, block_size=16)
        ac = AdaptiveCompression(
            bd,
            aggressive_threshold=0.5,
            passthrough_threshold=2.0,
        )
        # Mixed data: some blocks low entropy, some high
        data = torch.randn(100, 64)
        layer = ac.build_layer(out_features=10, data=data, base_control_points=16)
        x = torch.randn(4, 64)
        out = layer(x)
        assert out.shape == (4, 10)

    def test_adaptive_layer_mixed_strategies(self):
        """AdaptiveLinear with mixed strategies works."""
        bd = BlockDecomposition(input_dim=64, block_size=16)
        strategies = ["aggressive", "moderate", "passthrough", "moderate"]
        ctrl = [8, 16, 0, 16]
        layer = AdaptiveLinear(bd, out_features=10, control_points_per_block=ctrl, strategies=strategies)
        x = torch.randn(4, 64)
        out = layer(x)
        assert out.shape == (4, 10)


# ---------------------------------------------------------------------------
# EmbeddingBenchmark tests
# ---------------------------------------------------------------------------

class TestEmbeddingBenchmark:

    def test_synthetic_embeddings_shape(self):
        """Synthetic embeddings have correct shape."""
        bench = EmbeddingBenchmark(n_samples=100, n_classes=5)
        X, y = bench.generate_synthetic_embeddings(64)
        assert X.shape == (100, 64)
        assert y.shape == (100,)
        assert y.min() >= 0
        assert y.max() < 5

    def test_synthetic_embeddings_has_clusters(self):
        """Cluster structure exists (not all same point)."""
        bench = EmbeddingBenchmark(n_samples=200, n_classes=5)
        X, y = bench.generate_synthetic_embeddings(64)
        # Check that within-class variance is less than total variance
        total_var = X.var().item()
        within_var = sum(
            X[y == c].var().item() for c in range(5)
        ) / 5
        assert within_var < total_var

    def test_benchmark_single_dim(self):
        """Benchmark runs for a single dimension (small test)."""
        bench = EmbeddingBenchmark(
            dimensions=[64],
            n_samples=200,
            n_epochs=3,
            n_classes=5,
            block_size=16,
            n_control_points=8,
        )
        results = bench.run()
        assert len(results) == 1
        r = results[0]
        assert r["dim"] == 64
        assert r["spline_params"] > 0
        assert r["dense_params"] > 0
        assert r["compression_ratio"] >= 1.0
        assert 0.0 <= r["spline_accuracy"] <= 1.0
        assert 0.0 <= r["dense_accuracy"] <= 1.0

    def test_benchmark_summary(self):
        """Summary table is generated correctly."""
        bench = EmbeddingBenchmark(
            dimensions=[64],
            n_samples=100,
            n_epochs=2,
        )
        results = bench.run()
        summary = bench.summary(results)
        assert "64" in summary
        assert "Compress" in summary

    def test_benchmark_compression_positive(self):
        """Compression ratio is positive for all dimensions."""
        bench = EmbeddingBenchmark(
            dimensions=[64, 256],
            n_samples=200,
            n_epochs=3,
            block_size=16,
            n_control_points=8,
        )
        results = bench.run()
        for r in results:
            assert r["compression_ratio"] >= 1.0

    def test_benchmark_custom_dims(self):
        """Custom dimension list works."""
        bench = EmbeddingBenchmark(dimensions=[32, 48])
        assert bench.dimensions == [32, 48]

    def test_inference_time_measured(self):
        """Inference time is positive and reasonable."""
        bench = EmbeddingBenchmark(
            dimensions=[64],
            n_samples=100,
            n_epochs=1,
        )
        results = bench.run()
        assert results[0]["spline_inference_ms"] > 0
        assert results[0]["dense_inference_ms"] > 0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_end_to_end_train_spline_hd(self):
        """Train SplineLinearHD on a simple task and verify learning."""
        torch.manual_seed(42)
        layer = SplineLinearHD(
            in_features=64, out_features=4,
            block_size=16, n_control_points=8,
        )
        X = torch.randn(100, 64)
        y = (X[:, 0] > 0).long()  # Simple threshold task

        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        initial_loss = None
        final_loss = None
        for epoch in range(20):
            optimizer.zero_grad()
            logits = layer(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            if epoch == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        # Loss should decrease (model is learning)
        assert final_loss < initial_loss

    def test_spline_hd_vs_dense_approximation(self):
        """SplineLinearHD output is in reasonable range vs dense."""
        torch.manual_seed(0)
        dim = 64
        dense = nn.Linear(dim, 10)
        spline = SplineLinearHD(
            in_features=dim, out_features=10,
            block_size=16, n_control_points=16,
        )

        x = torch.randn(4, dim)
        d_out = dense(x)
        s_out = spline(x)

        # Both should produce finite outputs
        assert torch.isfinite(d_out).all()
        assert torch.isfinite(s_out).all()

    def test_adaptive_full_pipeline(self):
        """Full pipeline: generate data → measure entropy → build adaptive layer → train."""
        torch.manual_seed(42)
        dim = 64
        bd = BlockDecomposition(input_dim=dim, block_size=16)
        ac = AdaptiveCompression(bd)

        # Generate calibration data
        data = torch.randn(200, dim)
        layer = ac.build_layer(out_features=4, data=data, base_control_points=16)

        # Train briefly
        X = torch.randn(100, dim)
        y = torch.randint(0, 4, (100,))
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)

        for _ in range(5):
            optimizer.zero_grad()
            logits = layer(X)
            loss = nn.CrossEntropyLoss()(logits, y)
            loss.backward()
            optimizer.step()

        # Should still produce valid output
        with torch.no_grad():
            out = layer(X[:5])
        assert out.shape == (5, 4)
        assert torch.isfinite(out).all()
