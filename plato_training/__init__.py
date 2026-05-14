"""
PLATO Training Rooms — LoRA adapters with lifecycle, fleet-aware throttle, tensor-splines.

Engines:
  - PyTorchRoom: LoRA training with throttle
  - TensorFlowRoom: Keras training with throttle
  - SplineLinear: Eisenstein lattice-parameterized weights (novel)

CLI: plato-train train --room my-model --data data.csv
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
from .spline import SplineLinear, inject_spline, compression_ratio, EisensteinLattice
from .micro_models import train_micro, list_tasks, TASK_REGISTRY

__version__ = "0.3.0"
