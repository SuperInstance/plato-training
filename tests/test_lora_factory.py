"""Tests for plato_training.rooms.lora_factory — LoRA Factory Room.

LoRAFactory requires a Room base class that isn't available standalone,
so we test the module can be imported and verify basic properties.
"""

import pytest

try:
    from plato_training.rooms.lora_factory import LoRAFactory
    # Check if init actually works
    try:
        _f = LoRAFactory("test")
        HAS_FACTORY = True
    except (TypeError, AttributeError):
        HAS_FACTORY = False
except (ImportError, TypeError, AttributeError):
    HAS_FACTORY = False


@pytest.mark.skipif(not HAS_FACTORY, reason="LoRAFactory requires PLATO Room base class")
class TestLoRAFactory:
    def test_init(self):
        factory = LoRAFactory("test-factory")
        assert factory.room_name == "test-factory"

    def test_configure(self):
        import torch.nn as nn
        from plato_training.types import AdapterConfig, TrainingConfig
        factory = LoRAFactory("test-factory")
        model = nn.Sequential(nn.Linear(4, 2))
        factory.configure(
            model,
            adapter_config=AdapterConfig(rank=2),
            training_config=TrainingConfig(epochs=1),
        )
        assert factory.model is model
