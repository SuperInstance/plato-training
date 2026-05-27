"""Tests for onnx_export module."""

import os

import numpy as np
import pytest
import torch

from plato_training.eisenstein_encoder import EisensteinEncoder
from plato_training.onnx_export import (
    OPSET_VERSION,
    benchmark_onnx_vs_pytorch,
    export_eisenstein,
    export_spline,
    validate_numerical_parity,
)
from plato_training.spline import SplineLinear

try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:
    HAS_ORT = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_opset_version(self):
        assert isinstance(OPSET_VERSION, int)
        assert OPSET_VERSION >= 14


# ---------------------------------------------------------------------------
# validate_numerical_parity
# ---------------------------------------------------------------------------

class TestValidateNumericalParity:
    def test_identical_arrays_pass(self):
        a = np.random.randn(10, 32).astype(np.float32)
        result = validate_numerical_parity(a, a, tolerance=1e-6)
        assert result["passes"] is True
        assert result["max_diff"] == 0.0
        assert result["cosine_similarity"] == pytest.approx(1.0, abs=1e-5)

    def test_small_diff_passes(self):
        a = np.random.randn(10, 32).astype(np.float32)
        b = a + np.random.randn(10, 32).astype(np.float32) * 1e-6
        result = validate_numerical_parity(a, b, tolerance=1e-4)
        assert result["passes"] is True
        assert result["max_diff"] < 1e-3

    def test_large_diff_fails(self):
        a = np.zeros((10, 32), dtype=np.float32)
        b = np.ones((10, 32), dtype=np.float32)
        result = validate_numerical_parity(a, b, tolerance=0.1)
        assert result["passes"] is False
        assert result["max_diff"] == 1.0

    def test_cosine_similarity_negative(self):
        a = np.ones((1, 10), dtype=np.float32)
        b = -np.ones((1, 10), dtype=np.float32)
        result = validate_numerical_parity(a, b, tolerance=1e-4)
        assert result["cosine_similarity"] == pytest.approx(-1.0, abs=1e-5)

    def test_result_structure(self):
        a = np.random.randn(5, 16).astype(np.float32)
        b = a.copy()
        result = validate_numerical_parity(a, b)
        for key in ["max_diff", "mean_diff", "cosine_similarity", "tolerance", "passes"]:
            assert key in result

    def test_tolerance_respected(self):
        a = np.zeros((1, 1), dtype=np.float32)
        b = np.array([[0.001]], dtype=np.float32)
        assert validate_numerical_parity(a, b, tolerance=0.01)["passes"] is True
        assert validate_numerical_parity(a, b, tolerance=0.0001)["passes"] is False


# ---------------------------------------------------------------------------
# export_eisenstein
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_ORT, reason="onnxruntime not installed")
class TestExportEisenstein:
    def test_export_creates_file(self, tmp_path):
        encoder = EisensteinEncoder(vocab_size=100, embed_dim=8, out_dim=16)
        path = str(tmp_path / "test_eisenstein.onnx")
        result = export_eisenstein(encoder, path=path, opset=14)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    def test_export_dynamic_axes(self, tmp_path):
        """Exported model should accept variable batch sizes."""
        encoder = EisensteinEncoder(vocab_size=100, embed_dim=8, out_dim=16)
        path = str(tmp_path / "test_dynamic.onnx")
        export_eisenstein(encoder, path=path, opset=14)

        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        # Single sample
        ids1 = np.array([[1, 2, 3, 4]], dtype=np.int64)
        out1 = session.run(None, {input_name: ids1})[0]
        assert out1.shape == (1, 16)

        # Batch of 3
        ids3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
        out3 = session.run(None, {input_name: ids3})[0]
        assert out3.shape == (3, 16)

    def test_export_numerical_parity(self, tmp_path):
        """ONNX output should match PyTorch output."""
        encoder = EisensteinEncoder(vocab_size=100, embed_dim=8, out_dim=16)
        encoder.eval()
        path = str(tmp_path / "test_parity.onnx")
        export_eisenstein(encoder, path=path, opset=14)

        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)

        # PyTorch
        with torch.no_grad():
            pt_out = encoder(torch.tensor(ids, dtype=torch.long)).numpy()

        # ONNX
        ort_out = session.run(None, {input_name: ids})[0]

        result = validate_numerical_parity(pt_out, ort_out, tolerance=1e-4)
        assert result["passes"], f"Parity failed: max_diff={result['max_diff']}"


# ---------------------------------------------------------------------------
# export_spline
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_ORT, reason="onnxruntime not installed")
class TestExportSpline:
    def test_export_creates_file(self, tmp_path):
        layer = SplineLinear(16, 8, n_control_points=4)
        path = str(tmp_path / "test_spline.onnx")
        result = export_spline(layer, path=path, opset=14)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    def test_spline_numerical_parity(self, tmp_path):
        layer = SplineLinear(16, 8, n_control_points=4)
        layer.eval()
        path = str(tmp_path / "test_spline_parity.onnx")
        export_spline(layer, path=path, opset=14)

        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        x = np.random.randn(1, 16).astype(np.float32)

        with torch.no_grad():
            pt_out = layer(torch.tensor(x)).numpy()

        ort_out = session.run(None, {input_name: x})[0]

        result = validate_numerical_parity(pt_out, ort_out, tolerance=1e-4)
        assert result["passes"], f"Parity failed: max_diff={result['max_diff']}"


# ---------------------------------------------------------------------------
# benchmark_onnx_vs_pytorch
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_ORT, reason="onnxruntime not installed")
class TestBenchmark:
    def test_benchmark_runs(self, tmp_path):
        encoder = EisensteinEncoder(vocab_size=100, embed_dim=8, out_dim=16)
        path = str(tmp_path / "bench_eisenstein.onnx")
        export_eisenstein(encoder, path=path, opset=14)

        result = benchmark_onnx_vs_pytorch(
            encoder, path, n_iterations=5,
        )
        assert "pytorch_time_s" in result
        assert "onnx_time_s" in result
        assert "speedup" in result
        assert "parity" in result
        assert result["n_iterations"] == 5

    def test_benchmark_custom_texts(self, tmp_path):
        encoder = EisensteinEncoder(vocab_size=100, embed_dim=8, out_dim=16)
        path = str(tmp_path / "bench_custom.onnx")
        export_eisenstein(encoder, path=path, opset=14)

        texts = ["hello world", "test sentence"] * 3
        result = benchmark_onnx_vs_pytorch(
            encoder, path, texts=texts, n_iterations=5,
        )
        assert result["n_iterations"] == 5
