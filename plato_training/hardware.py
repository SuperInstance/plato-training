"""
Hardware Profiles — compile micro models for GPU, CPU, NPU, TPU.

Each profile defines:
- target device and dtype
- compilation strategy (torch.compile, ONNX, TFLite, TensorRT)
- deployment format
- size/latency budgets

Usage:
    deploy_micro("drift-detect", target="gpu")
    deploy_micro("anomaly-flag", target="cpu")
    deploy_micro("spam-classify", target="npu")
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

from .micro_models import (
    train_micro, TASK_REGISTRY, MicroClassifier, SplineClassifier,
    _generate_synthetic, _eval_model,
)
from .low_rank import LowRankClassifier, recommend_variant
from .spline import SplineLinear, compression_ratio as spline_compression
from .types import TrainingTile


# ─── Hardware Profiles ─────────────────────────────────────────────

@dataclass
class HardwareProfile:
    """Hardware target for micro model deployment."""
    name: str
    device: str                  # "cpu", "cuda", "xla" (TPU)
    dtype: str                   # "float32", "float16", "bfloat16", "int8"
    compile_mode: str            # "none", "default", "reduce-overhead", "max-autotune"
    export_format: str           # "pytorch", "onnx", "tflite", "torchscript"
    max_params: int              # Parameter budget
    max_latency_ms: float        # Inference latency budget (single sample)
    description: str
    notes: str = ""

PROFILES: Dict[str, HardwareProfile] = {
    "cpu": HardwareProfile(
        name="cpu",
        device="cpu",
        dtype="float32",
        compile_mode="default",
        export_format="pytorch",
        max_params=50_000,
        max_latency_ms=5.0,
        description="General CPU — x86/ARM, any machine",
        notes="Safe default. Works everywhere.",
    ),
    "cpu-tiny": HardwareProfile(
        name="cpu-tiny",
        device="cpu",
        dtype="float32",
        compile_mode="none",
        export_format="pytorch",
        max_params=5_000,
        max_latency_ms=1.0,
        description="Embedded CPU — ESP32, Cortex-M, microcontroller",
        notes="Extreme constraints. SplineLinear required.",
    ),
    "cpu-fast": HardwareProfile(
        name="cpu-fast",
        device="cpu",
        dtype="float32",
        compile_mode="reduce-overhead",
        export_format="torchscript",
        max_params=100_000,
        max_latency_ms=2.0,
        description="Server CPU with torch.compile",
        notes="For fleet services running on Oracle1 etc.",
    ),
    "gpu": HardwareProfile(
        name="gpu",
        device="cuda",
        dtype="float16",
        compile_mode="max-autotune",
        export_format="pytorch",
        max_params=1_000_000,
        max_latency_ms=0.5,
        description="NVIDIA GPU — RTX 4050, A100, H100",
        notes="Half precision, max autotune, batch inference.",
    ),
    "gpu-small": HardwareProfile(
        name="gpu-small",
        device="cuda",
        dtype="float16",
        compile_mode="default",
        export_format="pytorch",
        max_params=100_000,
        max_latency_ms=1.0,
        description="Small GPU — Jetson Orin Nano, RTX 3050",
        notes="Fleet edge nodes. Keep it tight.",
    ),
    "npu": HardwareProfile(
        name="npu",
        device="cpu",             # NPU via ONNX Runtime / TFLite
        dtype="int8",
        compile_mode="none",
        export_format="onnx",
        max_params=50_000,
        max_latency_ms=2.0,
        description="Neural Processing Unit — Qualcomm Hexagon, Apple Neural Engine",
        notes="Quantized INT8 for NPU dispatch.",
    ),
    "tpu": HardwareProfile(
        name="tpu",
        device="xla",
        dtype="bfloat16",
        compile_mode="default",
        export_format="pytorch",
        max_params=500_000,
        max_latency_ms=1.0,
        description="Google TPU — v4/v5, Edge TPU",
        notes="BFloat16 native. Needs torch_xla.",
    ),
    "wasm": HardwareProfile(
        name="wasm",
        device="cpu",
        dtype="float32",
        compile_mode="none",
        export_format="onnx",
        max_params=20_000,
        max_latency_ms=10.0,
        description="WebAssembly — browser, Cloudflare Workers",
        notes="ONNX export → onnxruntime-web. Size matters.",
    ),
}


# ─── Quantization ───────────────────────────────────────────────────

def quantize_dynamic(model: nn.Module) -> nn.Module:
    """Dynamic INT8 quantization for CPU inference."""
    return torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )


def quantize_static_prepare(model: nn.Module, calibration_loader) -> nn.Module:
    """Prepare for static quantization (needs calibration data)."""
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('x86')
    prepared = torch.quantization.prepare(model)
    # Calibrate
    with torch.no_grad():
        for batch in calibration_loader:
            if isinstance(batch, (list, tuple)):
                prepared(batch[0])
            else:
                prepared(batch)
    return torch.quantization.convert(prepared)


# ─── ONNX Export ────────────────────────────────────────────────────

def export_onnx(model: nn.Module, input_shape: Tuple[int, ...], path: str) -> dict:
    """Export model to ONNX format."""
    model.eval()
    dummy = torch.randn(1, *input_shape)
    
    torch.onnx.export(
        model, dummy, path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    
    size_bytes = os.path.getsize(path)
    return {"format": "onnx", "size_bytes": size_bytes, "path": path}


def export_torchscript(model: nn.Module, input_shape: Tuple[int, ...], path: str) -> dict:
    """Export model to TorchScript."""
    model.eval()
    dummy = torch.randn(1, *input_shape)
    scripted = torch.jit.trace(model, dummy)
    scripted.save(path)
    
    size_bytes = os.path.getsize(path)
    return {"format": "torchscript", "size_bytes": size_bytes, "path": path}


# ─── Build Pipeline ─────────────────────────────────────────────────

@dataclass
class DeployedModel:
    """A micro model built and packaged for a specific hardware target."""
    task: str
    variant: str                 # "dense", "lora", "spline"
    target: str                  # hardware profile name
    model: Any                   # the actual model
    tile: TrainingTile           # PLATO tile with training metadata
    profile: HardwareProfile     # hardware profile used
    metrics: dict                # training metrics
    deploy_info: dict = field(default_factory=dict)  # export info, size, etc.
    
    # Benchmark results (filled by bench())
    latency_ms: float = 0.0
    throughput_samples_s: float = 0.0
    model_size_bytes: int = 0
    
    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "variant": self.variant,
            "target": self.target,
            "latency_ms": round(self.latency_ms, 3),
            "throughput": round(self.throughput_samples_s, 1),
            "model_size_bytes": self.model_size_bytes,
            "tile_id": self.tile.tile_id,
            "accuracy": self.metrics.get("accuracy", 0),
            "deploy_info": self.deploy_info,
        }


def deploy_micro(
    task: str,
    target: str = "cpu",
    variant: str = "auto",       # "auto", "dense", "lora", "spline"
    store_dir: Optional[str] = None,
    bench_rounds: int = 100,
    export: bool = True,
) -> DeployedModel:
    """
    Build, optimize, and package a micro model for a hardware target.
    
    This is the ONE FUNCTION. Click the button. Get a deployed model.
    
    Args:
        task: task name (spam-classify, drift-detect, etc.)
        target: hardware profile (cpu, gpu, npu, tpu, wasm, cpu-tiny, gpu-small)
        variant: "auto" picks best for target, or specify
        store_dir: PLATO tile store directory
        bench_rounds: inference benchmark iterations
        export: whether to export to target format
    
    Returns:
        DeployedModel with model, metrics, benchmarks, and export artifacts
    """
    if task not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task}. Use list_tasks() or one of: {list(TASK_REGISTRY.keys())}")
    if target not in PROFILES:
        raise ValueError(f"Unknown target: {target}. Use one of: {list(PROFILES.keys())}")
    
    config = TASK_REGISTRY[task]
    profile = PROFILES[target]
    
    # Auto-select variant based on profile
    if variant == "auto":
        if profile.max_params <= 5000:
            variant = "spline"  # Tiny targets need compression
        elif profile.dtype == "int8":
            variant = "lowrank"  # INT8 + low-rank = double compression
        elif target in ("gpu", "gpu-small"):
            variant = "lora"    # GPU can handle LoRA efficiently
        else:
            # Use task-aware recommendation
            variant = recommend_variant(config["description"])
    
    # Train
    if store_dir is None:
        store_dir = tempfile.mkdtemp()
    
    model, tile, metrics = train_micro(task, variant=variant, store_dir=store_dir)
    
    # Check parameter budget
    total_params = sum(p.numel() for p in model.parameters())
    if total_params > profile.max_params:
        # Force spline if over budget
        if variant != "spline":
            model, tile, metrics = train_micro(task, variant="spline", store_dir=store_dir)
            total_params = sum(p.numel() for p in model.parameters())
    
    # Apply hardware optimizations
    model = _optimize_for_target(model, profile)
    
    # Move to target device — cast BOTH model and input dtype
    device = _resolve_device(profile.device)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int8": torch.float32,
    }
    target_dtype = dtype_map.get(profile.dtype, torch.float32)
    model = model.to(device=device, dtype=target_dtype)
    model.eval()
    
    # Export if requested
    deploy_info = {}
    if export:
        deploy_info = _export_model(model, config, profile, task, variant)
    
    # Benchmark
    input_dim = config["input_dim"]
    latency, throughput = _benchmark(model, input_dim, device, target_dtype, bench_rounds)
    
    # Serialize for size
    model_size = _model_size_bytes(model)
    
    return DeployedModel(
        task=task,
        variant=variant,
        target=target,
        model=model,
        tile=tile,
        profile=profile,
        metrics=metrics,
        deploy_info=deploy_info,
        latency_ms=latency,
        throughput_samples_s=throughput,
        model_size_bytes=model_size,
    )


def _resolve_device(device_str: str) -> torch.device:
    """Resolve device string, falling back gracefully."""
    if device_str == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")  # Graceful fallback
    if device_str == "xla":
        try:
            import torch_xla
            return torch.device("xla")
        except ImportError:
            return torch.device("cpu")  # No TPU available
    return torch.device(device_str)


def _optimize_for_target(model: nn.Module, profile: HardwareProfile) -> nn.Module:
    """Apply hardware-specific optimizations."""
    
    # Cast dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int8": torch.float32,  # Quantize separately
    }
    dtype = dtype_map.get(profile.dtype, torch.float32)
    model = model.to(dtype)
    
    # INT8 quantization for NPU targets
    if profile.dtype == "int8":
        model = quantize_dynamic(model)
    
    # torch.compile only for GPU targets (CPU compile overhead not worth it for micro models)
    if profile.compile_mode != "none" and profile.device == "cuda":
        try:
            model = torch.compile(model, mode=profile.compile_mode)
        except Exception:
            pass  # Graceful fallback
    
    return model


def _export_model(model, config, profile, task, variant) -> dict:
    """Export model to target format."""
    export_info = {"format": profile.export_format}
    
    tmpdir = tempfile.mkdtemp()
    input_shape = (config["input_dim"],)
    
    try:
        if profile.export_format == "onnx":
            path = os.path.join(tmpdir, f"{task}-{variant}-{profile.name}.onnx")
            export_info.update(export_onnx(model, input_shape, path))
        elif profile.export_format == "torchscript":
            path = os.path.join(tmpdir, f"{task}-{variant}-{profile.name}.pt")
            export_info.update(export_torchscript(model, input_shape, path))
        else:
            # PyTorch native — save state_dict
            path = os.path.join(tmpdir, f"{task}-{variant}-{profile.name}.pth")
            torch.save(model.state_dict(), path)
            export_info["size_bytes"] = os.path.getsize(path)
            export_info["path"] = path
    except Exception as e:
        export_info["export_error"] = str(e)
    
    return export_info


def _benchmark(model, input_dim, device, dtype=torch.float32, rounds=100) -> Tuple[float, float]:
    """Benchmark inference latency and throughput."""
    dummy = torch.randn(1, input_dim, device=device, dtype=dtype)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
    
    # Timed run
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(rounds):
            _ = model(dummy)
    elapsed = time.perf_counter() - start
    
    latency_ms = (elapsed / rounds) * 1000
    throughput = rounds / elapsed
    
    return latency_ms, throughput


def _model_size_bytes(model) -> int:
    """Get model size in bytes."""
    buf = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
    try:
        torch.save(model.state_dict(), buf.name)
        return os.path.getsize(buf.name)
    finally:
        os.unlink(buf.name)


# ─── Fleet Deploy (multiple tasks × multiple targets) ───────────────

def deploy_fleet(
    tasks: Optional[List[str]] = None,
    targets: Optional[List[str]] = None,
) -> Dict[str, Dict[str, dict]]:
    """
    Deploy all micro models across all hardware targets.
    
    The BIG RED BUTTON. One call, everything built and benchmarked.
    
    Returns:
        {task: {target: {accuracy, latency_ms, throughput, model_size_bytes, ...}}}
    """
    if tasks is None:
        tasks = list(TASK_REGISTRY.keys())
    if targets is None:
        targets = list(PROFILES.keys())
    
    results = {}
    for task in tasks:
        results[task] = {}
        for target in targets:
            try:
                deployed = deploy_micro(task, target=target, export=False)
                results[task][target] = deployed.to_dict()
            except Exception as e:
                results[task][target] = {"error": str(e)}
    
    return results


# ─── Room Spec Generator ───────────────────────────────────────────

def generate_room_spec(task: str, target: str = "cpu") -> str:
    """
    Generate a PLATO room spec for deploying this micro model.
    
    Ensigns read this spec to know what the room does, what hardware
    it needs, and how to invoke it.
    """
    config = TASK_REGISTRY[task]
    profile = PROFILES[target]
    
    spec = f"""# Room: micro-{task}
# Target: {profile.name} ({profile.description})
# Generated: {time.strftime('%Y-%m-%d %H:%M')}

## Task
{config['description']}

## Model
- Input: ({config['input_dim']},) float tensor
- Output: {config['num_classes']} classes
- Architecture: MicroClassifier ({config['hidden']} hidden)
- Params: ~{config['input_dim'] * config['hidden'] + config['hidden'] ** 2 + config['hidden'] * config['num_classes']:,}

## Hardware
- Device: {profile.device}
- Dtype: {profile.dtype}
- Export: {profile.export_format}
- Max params: {profile.max_params:,}
- Latency budget: {profile.max_latency_ms}ms

## Usage
```python
from plato_training.micro_models import train_micro
from plato_training.hardware import deploy_micro

# Train
model, tile, metrics = train_micro("{task}")

# Deploy for hardware
deployed = deploy_micro("{task}", target="{target}")
print(f"Accuracy: {{deployed.metrics['accuracy']:.1%}}")
print(f"Latency: {{deployed.latency_ms:.2f}}ms")
print(f"Size: {{deployed.model_size_bytes}} bytes")
```

## Classes
"""
    # Add class labels based on task
    class_labels = {
        "spam-classify": ["not-spam", "spam"],
        "intent-detect": ["query", "command", "question", "chitchat"],
        "anomaly-flag": ["normal", "anomaly"],
        "sentiment": ["negative", "neutral", "positive"],
        "topic-classify": ["topic-0", "topic-1", "topic-2", "topic-3", "topic-4"],
        "priority-rank": ["low", "medium", "high", "critical"],
        "drift-detect": ["stable", "drifting"],
        "tile-relevance": ["not-relevant", "relevant"],
    }
    
    for i, label in enumerate(class_labels.get(task, [f"class-{i}" for i in range(config['num_classes'])])):
        spec += f"- {i}: {label}\n"
    
    return spec
