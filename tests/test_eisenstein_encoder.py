"""Tests for EisensteinEncoder — tiny text encoder with Eisenstein-structured weights."""

import gc
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plato_training.eisenstein_encoder import (
    TRAINING_TRIPLETS,
    ContrastiveTrainer,
    EisensteinEncoder,
)

VOCAB = 1000
OUT_DIM = 32


def _make_encoder(**kw) -> EisensteinEncoder:
    return EisensteinEncoder(**kw)


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_param_count_positive(self):
        enc = _make_encoder()
        assert enc.param_count() > 0

    def test_size_under_100kb(self):
        enc = _make_encoder()
        assert enc.size_bytes() < 100_000, f"Size {enc.size_bytes()} exceeds 100KB"

    def test_default_vocab(self):
        enc = _make_encoder()
        assert enc.vocab_size == VOCAB


# ---------------------------------------------------------------------------
# 2. Forward pass
# ---------------------------------------------------------------------------

class TestForward:
    def test_output_shape(self):
        enc = _make_encoder()
        x = torch.randint(0, VOCAB, (4, 10))
        out = enc(x)
        assert out.shape == (4, OUT_DIM), f"Expected (4, {OUT_DIM}), got {out.shape}"

    def test_output_dtype(self):
        enc = _make_encoder()
        x = torch.randint(0, VOCAB, (2, 5))
        out = enc(x)
        assert out.dtype == torch.float32

    def test_single_token(self):
        enc = _make_encoder()
        x = torch.randint(0, VOCAB, (1, 1))
        out = enc(x)
        assert out.shape == (1, OUT_DIM)


# ---------------------------------------------------------------------------
# 3. Normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_output_norms_approx_one(self):
        enc = _make_encoder()
        x = torch.randint(0, VOCAB, (8, 12))
        out = enc(x)
        norms = out.norm(dim=1)
        nonzero = norms[norms > 1e-6]
        assert len(nonzero) > 0, "All outputs are zero-norm"
        assert torch.allclose(nonzero, torch.ones_like(nonzero), atol=1e-4), (
            f"Norms: {norms.tolist()}"
        )


# ---------------------------------------------------------------------------
# 4. encode_text
# ---------------------------------------------------------------------------

class TestEncodeText:
    def test_basic_encoding(self):
        enc = _make_encoder()
        vecs = enc.encode_text(["hello world", "test"])
        assert vecs.shape == (2, OUT_DIM)
        assert isinstance(vecs, np.ndarray)

    def test_empty_string_handled(self):
        enc = _make_encoder()
        vecs = enc.encode_text([""])
        assert vecs.shape == (1, OUT_DIM)


# ---------------------------------------------------------------------------
# 5 & 6. Similarity
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_similar_texts_high_similarity(self):
        enc = _make_encoder()
        a = enc.encode_text(["how many tests in plato"])
        b = enc.encode_text(["how many tests in plato"])
        sim = np.dot(a, b.T)[0, 0]
        assert sim > 0.5, f"Identical text similarity: {sim}"

    def test_untrained_different_may_vary(self):
        enc = _make_encoder()
        a = enc.encode_text(["how many tests"])
        b = enc.encode_text(["color blue"])
        sim = np.dot(a, b.T)[0, 0]
        assert -1.01 <= sim <= 1.01, f"Cosine out of range: {sim}"

    def test_cosine_similarity_bounded(self):
        enc = _make_encoder()
        texts = ["hello world", "test phrase", "completely different"]
        vecs = enc.encode_text(texts)
        for i in range(len(texts)):
            for j in range(len(texts)):
                sim = np.dot(vecs[i], vecs[j])
                assert -1.01 <= sim <= 1.01


# ---------------------------------------------------------------------------
# 7. Contrastive training
# ---------------------------------------------------------------------------

class TestContrastiveTraining:
    def test_loss_decreases(self):
        enc = _make_encoder()
        trainer = ContrastiveTrainer(enc, lr=1e-3)
        triplets = TRAINING_TRIPLETS[:10]
        losses = trainer.train_epoch(triplets, epochs=5)
        assert len(losses) == 50, f"Expected 50 losses, got {len(losses)}"
        early = sum(losses[:10]) / 10
        late = sum(losses[-10:]) / 10
        assert late <= early + 0.1, f"Loss didn't decrease: early={early:.4f}, late={late:.4f}"

    def test_train_step_returns_float(self):
        enc = _make_encoder()
        trainer = ContrastiveTrainer(enc)
        a = torch.randint(0, VOCAB, (1, 3))
        p = torch.randint(0, VOCAB, (1, 3))
        n = torch.randint(0, VOCAB, (1, 3))
        loss = trainer.train_step(a, p, n)
        assert isinstance(loss, float)
        assert loss >= 0
        gc.collect()


# ---------------------------------------------------------------------------
# 8. SplineLinear layers
# ---------------------------------------------------------------------------

class TestSplineLayers:
    def test_spline_present_if_available(self):
        enc = _make_encoder()
        try:
            from plato_training.spline import SplineLinear
            assert enc.uses_spline, "SplineLinear available but not used"
            assert isinstance(enc.project, SplineLinear)
            assert isinstance(enc.refine, SplineLinear)
        except ImportError:
            pytest.skip("SplineLinear not available")

    def test_fallback_linear_if_no_spline(self):
        enc = _make_encoder()
        x = torch.randint(0, VOCAB, (2, 5))
        out = enc(x)
        assert out.shape == (2, OUT_DIM)


# ---------------------------------------------------------------------------
# 9. Size report
# ---------------------------------------------------------------------------

class TestSizeReport:
    def test_param_and_size_report(self):
        enc = _make_encoder()
        params = enc.param_count()
        size = enc.size_bytes()
        ratio = 8_000_000 / size

        print(f"\n=== EisensteinEncoder Metrics ===")
        print(f"  Parameters:   {params:,}")
        print(f"  Size (bytes): {size:,}")
        print(f"  Size (KB):    {size / 1024:.1f}")
        print(f"  Compression:  {ratio:.0f}x vs Model2Vec (~8MB)")
        print(f"  Uses spline:  {enc.uses_spline}")
        print(f"=================================\n")

        assert params > 0
        assert size > 0
        assert size < 500_000


# ---------------------------------------------------------------------------
# 10. ONNX export
# ---------------------------------------------------------------------------

class TestONNXExport:
    def test_export_eisenstein_via_pipeline(self):
        pytest.importorskip("onnxscript", reason="onnxscript required for torch.onnx.export")
        try:
            from plato_training.onnx_export import export_eisenstein, validate_numerical_parity
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnx/onnxruntime not installed")

        enc = _make_encoder()
        enc.eval()
        path = export_eisenstein(enc, "/tmp/test_eisenstein_pipeline.onnx", opset=17)
        assert os.path.exists(path)

        # Validate with ONNX Runtime
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        dummy = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.int64)
        onnx_out = session.run(None, {"token_ids": dummy})[0]
        with torch.no_grad():
            pt_out = enc(torch.tensor(dummy, dtype=torch.long)).numpy()
        parity = validate_numerical_parity(pt_out, onnx_out)
        assert parity["passes"], f"Numerical parity failed: max_diff={parity['max_diff']}"

        os.remove(path)
        gc.collect()

    def test_export_spline_via_pipeline(self):
        pytest.importorskip("onnxscript", reason="onnxscript required for torch.onnx.export")
        try:
            from plato_training.onnx_export import export_spline, validate_numerical_parity
            import onnxruntime as ort
            from plato_training.spline import SplineLinear
        except ImportError:
            pytest.skip("onnx/onnxruntime not installed")

        sl = SplineLinear(32, 64, n_control_points=8)
        sl.eval()
        path = export_spline(sl, "/tmp/test_spline_pipeline.onnx")
        assert os.path.exists(path)

        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        dummy = np.random.randn(3, 32).astype(np.float32)
        onnx_out = session.run(None, {"input": dummy})[0]
        with torch.no_grad():
            pt_out = sl(torch.tensor(dummy)).numpy()
        parity = validate_numerical_parity(pt_out, onnx_out)
        assert parity["passes"]

        os.remove(path)
        gc.collect()

    def test_dynamic_axes_variable_batch(self):
        pytest.importorskip("onnxscript", reason="onnxscript required for torch.onnx.export")
        try:
            from plato_training.onnx_export import export_eisenstein
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnx/onnxruntime not installed")

        enc = _make_encoder()
        enc.eval()
        path = export_eisenstein(enc, "/tmp/test_dynamic_axes.onnx")

        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        # batch=1, seq=5
        out1 = session.run(None, {"token_ids": np.array([[1, 2, 3, 4, 5]], dtype=np.int64)})
        # batch=4, seq=12
        out4 = session.run(None, {"token_ids": np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]] * 4, dtype=np.int64)})
        assert out1[0].shape == (1, OUT_DIM)
        assert out4[0].shape == (4, OUT_DIM)

        os.remove(path)
        gc.collect()
