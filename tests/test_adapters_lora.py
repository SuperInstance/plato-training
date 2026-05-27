"""Tests for plato_training.adapters.lora — LoRA layer implementation."""

import torch
import torch.nn as nn
import tempfile
from pathlib import Path

import pytest

from plato_training.adapters.lora import (
    LoRALayer,
    inject_lora,
    load_lora_weights,
    save_lora_weights,
)


class TestLoRALayer:
    def test_creation(self):
        original = nn.Linear(32, 16)
        lora = LoRALayer(original, rank=4, alpha=8)
        assert lora.rank == 4
        assert lora.alpha == 8
        assert lora.scaling == 8 / 4
        assert lora.d_in == 32
        assert lora.d_out == 16

    def test_forward_shape(self):
        original = nn.Linear(32, 16)
        lora = LoRALayer(original, rank=4)
        x = torch.randn(2, 32)
        out = lora(x)
        assert out.shape == (2, 16)

    def test_forward_adds_lora_after_update(self):
        """After modifying LoRA weights, output should differ from original."""
        original = nn.Linear(8, 4)
        lora = LoRALayer(original, rank=2)
        # B is zero-initialized, so initially LoRA contributes nothing
        # Set B to non-zero to see the effect
        nn.init.ones_(lora.lora_B)
        x = torch.randn(1, 8)
        original_out = original(x)
        lora_out = lora(x)
        assert not torch.allclose(original_out, lora_out)

    def test_merge_shape(self):
        original = nn.Linear(16, 8)
        lora = LoRALayer(original, rank=4)
        merged = lora.merge()
        assert isinstance(merged, nn.Linear)
        assert merged.weight.shape == (8, 16)

    def test_merge_preserves_output(self):
        """Merged linear should produce same output as LoRALayer."""
        original = nn.Linear(16, 8)
        lora = LoRALayer(original, rank=4)
        x = torch.randn(3, 16)
        with torch.no_grad():
            lora_out = lora(x)
            merged = lora.merge()
            merged_out = merged(x)
        assert torch.allclose(lora_out, merged_out, atol=1e-5)

    def test_num_trainable_params(self):
        original = nn.Linear(10, 5)
        lora = LoRALayer(original, rank=4)
        params = lora.num_trainable_params()
        assert params == 10 * 4 + 4 * 5  # A + B

    def test_lora_state_dict(self):
        original = nn.Linear(10, 5)
        lora = LoRALayer(original, rank=3)
        sd = lora.lora_state_dict()
        assert "lora_A" in sd
        assert "lora_B" in sd
        assert sd["lora_A"].shape == (10, 3)
        assert sd["lora_B"].shape == (3, 5)

    def test_repr(self):
        original = nn.Linear(10, 5)
        lora = LoRALayer(original)
        assert "LoRALayer" in repr(lora)

    def test_dropout(self):
        original = nn.Linear(10, 5)
        lora = LoRALayer(original, dropout=0.1)
        assert isinstance(lora.dropout, nn.Dropout)

    def test_no_dropout(self):
        original = nn.Linear(10, 5)
        lora = LoRALayer(original, dropout=0.0)
        assert isinstance(lora.dropout, nn.Identity)

    def test_bias_frozen(self):
        original = nn.Linear(10, 5, bias=True)
        lora = LoRALayer(original)
        assert not original.weight.requires_grad
        assert not original.bias.requires_grad


class TestInjectLora:
    def test_inject_into_simple_model(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        injection_map = inject_lora(model, rank=4, target_modules=["0", "2"])
        # All Linear modules should match
        assert len(injection_map) == 2

    def test_inject_default_targets(self):
        """Default targets W_query, W_value etc — no matches in a plain model."""
        model = nn.Sequential(nn.Linear(10, 5))
        injection_map = inject_lora(model)
        assert len(injection_map) == 0

    def test_inject_custom_targets(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.W_query = nn.Linear(8, 4)
                self.W_value = nn.Linear(8, 4)
                self.head = nn.Linear(4, 2)

        model = TestModel()
        injection_map = inject_lora(model, rank=2)
        assert "W_query" in injection_map or any("W_query" in k for k in injection_map)
        assert len(injection_map) == 2

    def test_injected_are_lora_layers(self):
        model = nn.Sequential(nn.Linear(10, 5))
        inject_lora(model, rank=4, target_modules=["0"])
        assert isinstance(model[0], LoRALayer)


class TestSaveLoadWeights:
    def test_save_load_roundtrip(self):
        model = nn.Sequential(nn.Linear(10, 5))
        injection_map = inject_lora(model, rank=4, target_modules=["0"])

        # Get original A/B
        original_A = model[0].lora_A.data.clone()
        original_B = model[0].lora_B.data.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "weights.pt")
            save_lora_weights(model, injection_map, path)

            # Perturb weights
            model[0].lora_A.data.fill_(0)
            model[0].lora_B.data.fill_(0)

            # Load back
            load_lora_weights(model, injection_map, path)

            assert torch.allclose(model[0].lora_A.data, original_A)
            assert torch.allclose(model[0].lora_B.data, original_B)

    def test_save_returns_bytes(self):
        model = nn.Sequential(nn.Linear(10, 5))
        injection_map = inject_lora(model, rank=4, target_modules=["0"])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "weights.pt")
            data = save_lora_weights(model, injection_map, path)
            assert isinstance(data, bytes)
            assert len(data) > 0
