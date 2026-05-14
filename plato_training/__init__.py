"""
PLATO Training Rooms — Micro models for ensigns, deployed anywhere.

Engines:
  - PyTorchRoom: LoRA training with throttle
  - TensorFlowRoom: Keras training with throttle
  - SplineLinear: Eisenstein lattice weights (smooth tasks)
  - LowRankLinear: Factorized weights (sharp/classification tasks)
  - HierarchicalSplineLinear: Multi-scale for high-dim tasks

Deploy:
  deploy_micro("drift-detect", target="npu")  # One function
  deploy_fleet()                                # All tasks × all targets

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
from .low_rank import LowRankLinear, LowRankClassifier, inject_low_rank, recommend_variant
from .hierarchical_spline import HierarchicalSplineLinear, HierarchicalSplineClassifier
from .micro_models import train_micro, list_tasks, TASK_REGISTRY
from .hardware import deploy_micro, PROFILES, generate_room_spec
from .data_rooms import DataRoom, DataSpec
from .micro_room import MicroRoom, RoomFactory

__version__ = "0.8.0"
