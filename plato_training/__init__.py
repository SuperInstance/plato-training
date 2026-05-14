"""
PLATO Training Rooms — LoRA adapters with lifecycle, fleet-aware throttle.

Three engines:
  - PyTorchRoom: LoRA training with throttle
  - TensorFlowRoom: Keras training with throttle
  - (future) TensorSplineRoom: lattice-parameterized training
"""

from .types import (
    TrainingTile, TileType, TileLifecycle, LamportClock,
    AdapterConfig, TrainingConfig, TrainingMetrics, content_hash,
)
from .adapters import LoRALayer, inject_lora, save_lora_weights, load_lora_weights
from .rooms import LoRAFactory
from .store import LocalTileStore
from .throttle import TrainingThrottle, ThrottleLevel, ThrottleState
from .pytorch_room import PyTorchRoom
from .tensorflow_room import TensorFlowRoom

__version__ = "0.2.0"
