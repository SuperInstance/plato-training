"""
Tests for hardware deployment pipeline.
"""

import importlib
import pytest
import torch
import torch.nn as nn
from plato_training.hardware import (
    deploy_micro, PROFILES, HardwareProfile,
    export_onnx, export_torchscript, quantize_dynamic,
    generate_room_spec, _benchmark, _model_size_bytes,
)


class TestProfiles:
    def test_all_profiles_have_required_fields(self):
        for name, p in PROFILES.items():
            assert p.device, f"{name} missing device"
            assert p.dtype, f"{name} missing dtype"
            assert p.max_params > 0, f"{name} has no param budget"
            assert p.max_latency_ms > 0, f"{name} has no latency budget"

    def test_profile_names_match_keys(self):
        for key, p in PROFILES.items():
            assert p.name == key

    def test_at_least_4_targets(self):
        assert len(PROFILES) >= 4


class TestQuantize:
    def test_dynamic_quantize_reduces_size(self):
        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4))
        original_size = _model_size_bytes(model)
        quantized = quantize_dynamic(model)
        quantized_size = _model_size_bytes(quantized)
        # Quantized should be smaller (INT8 weights)
        assert quantized_size <= original_size


class TestExport:
    def test_torchscript_export(self, tmp_path):
        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4))
        model.eval()
        path = str(tmp_path / "test.pt")
        info = export_torchscript(model, (64,), path)
        assert info["format"] == "torchscript"
        assert info["size_bytes"] > 0

    @pytest.mark.skipif(
        not __import__('importlib').util.find_spec('onnxscript'),
        reason='onnxscript not installed'
    )
    def test_onnx_export(self, tmp_path):
        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4))
        model.eval()
        path = str(tmp_path / "test.onnx")
        info = export_onnx(model, (64,), path)
        assert info["format"] == "onnx"
        assert info["size_bytes"] > 0


class TestDeployMicro:
    def test_cpu_deploy(self, tmp_path):
        deployed = deploy_micro("spam-classify", target="cpu", export=False,
                                store_dir=str(tmp_path / "s"))
        assert deployed.metrics["accuracy"] > 0.5
        assert deployed.latency_ms > 0
        assert deployed.model_size_bytes > 0

    def test_auto_variant_for_cpu(self, tmp_path):
        deployed = deploy_micro("spam-classify", target="cpu", variant="auto",
                                export=False, store_dir=str(tmp_path / "s"))
        # spam-classify description triggers lowrank recommendation
        assert deployed.variant in ("lowrank", "dense")

    def test_auto_variant_for_cpu_tiny(self, tmp_path):
        deployed = deploy_micro("spam-classify", target="cpu-tiny", variant="auto",
                                export=False, store_dir=str(tmp_path / "s"))
        assert deployed.variant == "spline"

    def test_deploy_returns_tile(self, tmp_path):
        deployed = deploy_micro("anomaly-flag", target="cpu", export=False,
                                store_dir=str(tmp_path / "s"))
        assert deployed.tile is not None
        assert deployed.tile.is_active()

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            deploy_micro("nonexistent-task")

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError, match="Unknown target"):
            deploy_micro("spam-classify", target="quantum-computer")


class TestBenchmark:
    def test_benchmark_produces_results(self):
        model = nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 2))
        model.eval()
        latency, throughput = _benchmark(model, 64, torch.device("cpu"), rounds=50)
        assert latency > 0
        assert throughput > 0


class TestRoomSpec:
    def test_generates_spec(self):
        spec = generate_room_spec("spam-classify")
        assert "spam" in spec.lower()
        assert "float" in spec.lower()
        assert "deploy_micro" in spec

    def test_spec_includes_classes(self):
        spec = generate_room_spec("drift-detect")
        assert "stable" in spec
        assert "drifting" in spec
