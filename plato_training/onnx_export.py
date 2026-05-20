"""
ONNX export pipeline for Eisenstein encoder and SplineLinear layers.

Exports models to ONNX format (opset 17) for NPU deployment with
dynamic batch/sequence-length axes and numerical parity validation.

Public API
----------
export_eisenstein          Export EisensteinEncoder to ONNX.
export_spline              Export a standalone SplineLinear layer to ONNX.
benchmark_onnx_vs_pytorch  Benchmark ONNX Runtime vs PyTorch inference.
validate_numerical_parity  Check outputs match within tolerance.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import onnx
    import onnxruntime as ort

    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False

from .eisenstein_encoder import EisensteinEncoder
from .spline import SplineLinear

# NPU-compatible opset
OPSET_VERSION = 17


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def export_eisenstein(
    encoder: EisensteinEncoder,
    path: str = "models/eisenstein-fleet-v2.onnx",
    opset: int = OPSET_VERSION,
) -> str:
    """
    Export an EisensteinEncoder to ONNX with dynamic batch/seq_len axes.

    Args:
        encoder: Trained (or untrained) EisensteinEncoder.
        path:    Output .onnx file path (directories created if needed).
        opset:   ONNX opset version (default 17 for NPU compatibility).

    Returns:
        Absolute path to the exported ONNX file.
    """
    if not _HAS_ONNX:
        raise RuntimeError("onnx and onnxruntime are required: pip install onnx onnxruntime")

    encoder.eval()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    dummy = torch.randint(0, encoder.vocab_size, (1, 10))

    torch.onnx.export(
        encoder,
        dummy,
        path,
        input_names=["token_ids"],
        output_names=["embedding"],
        dynamic_axes={
            "token_ids": {0: "batch", 1: "seq_len"},
            "embedding": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    # Validate the exported model
    model = onnx.load(path)
    onnx.checker.check_model(model)
    print(f"[onnx_export] EisensteinEncoder → {path}  ({os.path.getsize(path):,} bytes, opset {opset})")
    return os.path.abspath(path)


def export_spline(
    layer: SplineLinear,
    path: str = "models/splinelinear.onnx",
    opset: int = OPSET_VERSION,
) -> str:
    """
    Export a SplineLinear layer to ONNX.

    Args:
        layer: SplineLinear instance.
        path:  Output .onnx file path.
        opset: ONNX opset version.

    Returns:
        Absolute path to the exported ONNX file.
    """
    if not _HAS_ONNX:
        raise RuntimeError("onnx and onnxruntime are required: pip install onnx onnxruntime")

    layer.eval()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    dummy = torch.randn(1, layer.in_features)

    torch.onnx.export(
        layer,
        dummy,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    model = onnx.load(path)
    onnx.checker.check_model(model)
    print(f"[onnx_export] SplineLinear({layer.in_features}→{layer.out_features}) → {path}  ({os.path.getsize(path):,} bytes)")
    return os.path.abspath(path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_numerical_parity(
    pytorch_output: np.ndarray,
    onnx_output: np.ndarray,
    tolerance: float = 1e-4,
) -> dict:
    """
    Compare PyTorch and ONNX outputs for numerical parity.

    Args:
        pytorch_output: Numpy array from PyTorch inference.
        onnx_output:    Numpy array from ONNX Runtime inference.
        tolerance:      Max allowed absolute difference.

    Returns:
        Dict with max_diff, mean_diff, passes (bool), cosine_similarity.
    """
    diff = np.abs(pytorch_output - onnx_output)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    # Cosine similarity between flattened outputs
    flat_pt = pytorch_output.flatten()
    flat_onnx = onnx_output.flatten()
    cos_sim = float(
        np.dot(flat_pt, flat_onnx) / (np.linalg.norm(flat_pt) * np.linalg.norm(flat_onnx) + 1e-12)
    )

    passes = max_diff <= tolerance

    result = {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "cosine_similarity": cos_sim,
        "tolerance": tolerance,
        "passes": passes,
    }
    status = "✅ PASS" if passes else "❌ FAIL"
    print(f"[onnx_export] Parity check: {status}  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  cos_sim={cos_sim:.6f}")
    return result


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------


def benchmark_onnx_vs_pytorch(
    encoder: EisensteinEncoder,
    onnx_path: str,
    texts: Optional[List[str]] = None,
    n_iterations: int = 1000,
) -> dict:
    """
    Benchmark ONNX Runtime vs PyTorch CPU inference on the EisensteinEncoder.

    Args:
        encoder:      EisensteinEncoder instance (used for PyTorch path).
        onnx_path:    Path to exported ONNX file.
        texts:        Texts to encode. Defaults to PLATO domain phrases.
        n_iterations: Number of texts to encode in each benchmark run.

    Returns:
        Dict with pytorch_time, onnx_time, speedup, parity results.
    """
    if not _HAS_ONNX:
        raise RuntimeError("onnxruntime is required for benchmarking")

    if texts is None:
        texts = [
            "deploy drift detection model",
            "fleet status check",
            "spline linear compression ratio",
            "eisenstein encoder embedding",
            "plato training room session",
        ] * (n_iterations // 5 + 1)
    texts = texts[:n_iterations]

    # --- PyTorch benchmark ---
    encoder.eval()
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            ids = torch.randint(0, encoder.vocab_size, (1, 8))
            encoder(ids)

    start = time.perf_counter()
    pytorch_outputs = []
    with torch.no_grad():
        for text in texts:
            ids_list = []
            for t in [text]:
                tokens = t.lower().split()
                ids = [hash(w) % encoder.vocab_size for w in tokens] or [0]
                ids_list.append(ids)
            max_len = max(len(i) for i in ids_list)
            padded = [i + [0] * (max_len - len(i)) for i in ids_list]
            tensor_ids = torch.tensor(padded, dtype=torch.long)
            out = encoder(tensor_ids).numpy()
            pytorch_outputs.append(out)
    pytorch_time = time.perf_counter() - start

    # --- ONNX Runtime benchmark ---
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Warm up
    for _ in range(10):
        session.run(None, {input_name: np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)})

    start = time.perf_counter()
    onnx_outputs = []
    for text in texts:
        ids_list = []
        for t in [text]:
            tokens = t.lower().split()
            ids = [hash(w) % encoder.vocab_size for w in tokens] or [0]
            ids_list.append(ids)
        max_len = max(len(i) for i in ids_list)
        padded = [i + [0] * (max_len - len(i)) for i in ids_list]
        np_ids = np.array(padded, dtype=np.int64)
        out = session.run(None, {input_name: np_ids})[0]
        onnx_outputs.append(out)
    onnx_time = time.perf_counter() - start

    # --- Parity check ---
    pytorch_arr = np.vstack(pytorch_outputs)
    onnx_arr = np.vstack(onnx_outputs)
    parity = validate_numerical_parity(pytorch_arr, onnx_arr, tolerance=1e-4)

    speedup = pytorch_time / onnx_time if onnx_time > 0 else float("inf")

    result = {
        "n_iterations": n_iterations,
        "pytorch_time_s": round(pytorch_time, 4),
        "onnx_time_s": round(onnx_time, 4),
        "speedup": round(speedup, 2),
        "pytorch_per_text_ms": round(pytorch_time / n_iterations * 1000, 3),
        "onnx_per_text_ms": round(onnx_time / n_iterations * 1000, 3),
        "parity": parity,
    }

    print(f"\n{'='*50}")
    print(f"  ONNX vs PyTorch Benchmark ({n_iterations} texts)")
    print(f"{'='*50}")
    print(f"  PyTorch:  {result['pytorch_time_s']:.3f}s  ({result['pytorch_per_text_ms']:.2f} ms/text)")
    print(f"  ONNX RT:  {result['onnx_time_s']:.3f}s  ({result['onnx_per_text_ms']:.2f} ms/text)")
    print(f"  Speedup:  {result['speedup']:.2f}x")
    print(f"  Parity:   {'✅ PASS' if parity['passes'] else '❌ FAIL'} (max_diff={parity['max_diff']:.2e})")
    print(f"{'='*50}\n")

    return result
