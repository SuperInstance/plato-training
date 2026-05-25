# plato-training

**PLATO Training Rooms — LoRA adapters with lifecycle management, micro models for agents, deployed anywhere.**

A Python library for training, versioning, and deploying LoRA adapters and micro models as PLATO tiles. Includes throttle-aware training, Eisenstein spline layers, hardware-aware deployment, and an Intelligence Room that distills LLM knowledge into self-improving micro models.

## How It Works

Every training run is a PLATO room. Every artifact is a tile with a lifecycle (Active → Superseded → Archived). Training is a background citizen of the fleet — the throttle mechanism adjusts batch sizes and GPU usage based on fleet load.

```
┌─────────────────────────────────────────────────┐
│              PLATO Room Protocol                 │
│  Tiles · Lifecycle · Lamport Clock · Throttle   │
└────────┬────────────────────┬───────────────────┘
         │                    │
   ┌─────▼─────┐      ┌──────▼──────┐
   │ PyTorch   │      │ TensorFlow  │
   │ Room      │      │ Room        │
   └─────┬─────┘      └──────┬──────┘
         │                    │
   ┌─────▼────────────────────▼──────┐
   │   Tensor-Spline Platform        │
   │   (Eisenstein, Low-Rank, HD)    │
   └─────────────────────────────────┘
```

## Installation

```bash
pip install -e .

# With optional extras:
pip install -e ".[gpu]"       # PyTorch
pip install -e ".[semantic]"  # Model2Vec + FAISS for semantic matching
pip install -e ".[onnx]"      # ONNX export
pip install -e ".[all]"       # Everything
```

**Requirements:** Python ≥ 3.9. Core has no required dependencies (graceful fallbacks).

## Quick Start

### LoRA Adapter Training

```python
from plato_training import (
    LoRAFactory, LoRALayer, inject_lora,
    AdapterConfig, TrainingConfig,
)

# Inject LoRA into any nn.Module
model = ...  # your PyTorch model
injection_map = inject_lora(model, rank=8, alpha=16)

# Or use the factory for full lifecycle management
factory = LoRAFactory("spam-detector")
factory.configure(
    base_model=model,
    adapter_config=AdapterConfig(rank=8, alpha=16),
    training_config=TrainingConfig(epochs=10, learning_rate=1e-4),
)
adapter_tile = factory.train(train_loader, val_loader)
```

### Eisenstein Spline Layers

Novel weight parameterization on the A₂ Eisenstein lattice. Compresses smoothly varying weights with provable structure.

```python
from plato_training import SplineLinear, inject_spline, compression_ratio

# Replace linear layers with spline-parameterized variants
inject_spline(model, control_points=8)

# Or use directly
layer = SplineLinear(in_features=256, out_features=128, control_points=8)
output = layer(input_tensor)

# Check compression
ratio = compression_ratio(layer)
print(f"Spline compression: {ratio:.1f}x")
```

### Micro Models

Train tiny models for agent skills and deploy to any hardware target:

```python
from plato_training import train_micro, deploy_micro, list_tasks

# See available tasks
tasks = list_tasks()
# ["drift-detect", "anomaly-flag", "intent-detect", "sentiment", ...]

# Train a micro model
model = train_micro("drift-detect")

# Deploy to hardware
deployed = deploy_micro(model, target="npu")
result = deployed.predict(sensor_data)
```

### Intelligence Room

Distills LLM knowledge into self-improving micro models. Every LLM call is a training opportunity:

```
Request ──► PreFilter (micro) ──► [LLM?] ──► PostFilter (micro) ──► Response
                  │                                   │
                  ▼                                   ▼
           Routing Decision                     Knowledge Tiles
           (skip unnecessary calls)             (facts, patterns)
```

```python
from plato_training import IntelligenceRoom

room = IntelligenceRoom("main-intel")
response = room.route("explain the Eisenstein lattice")
# Pre-filter learns to skip LLM for known topics
# Post-filter extracts knowledge tiles from responses
# Self-trainer retrains micro models during idle time
```

## Core Components

### Training Tiles

Every artifact is a tile with full provenance:

```python
from plato_training import TrainingTile, TileType, TileLifecycle

tile = TrainingTile(
    name="drift-detect-v3",
    tile_type=TileType.ADAPTER,
    lifecycle=TileLifecycle.ACTIVE,
)
```

Tile types: `DATASET`, `CONFIG`, `CHECKPOINT`, `METRICS`, `ADAPTER`, `PREDICTION`
Lifecycle: `ACTIVE` → `SUPERSEDED` → `ARCHIVED`

### Throttle

Training adjusts to fleet load:

```python
from plato_training import TrainingThrottle

throttle = TrainingThrottle()
state = throttle.check()
# ThrottleLevel.FULL    (0-2 rooms active) → full GPU
# ThrottleLevel.REDUCED (3-5 rooms)        → smaller batches
# ThrottleLevel.MINIMAL (6-9 rooms)        → minimal resources
# ThrottleLevel.PAUSED  (10+ rooms)        → stop training
```

### LoRA Adapters

Full LoRA implementation with save/load round-trip:

```python
from plato_training import LoRALayer, save_lora_weights, load_lora_weights

# Merge adapter back into base model
merged = lora_layer.merge()

# Save/load (supports safetensors)
save_lora_weights(model, injection_map, "adapter.safetensors")
load_lora_weights(model, injection_map, "adapter.safetensors")
```

### Novel Layers

| Layer | Use Case | File |
|-------|----------|------|
| `SplineLinear` | Smooth tasks, Eisenstein lattice weights | `spline.py` |
| `LowRankLinear` | Classification, factorized weights | `low_rank.py` |
| `HierarchicalSplineLinear` | High-dimensional multi-scale | `hierarchical_spline.py` |

### Hardware Deployment

```python
from plato_training import deploy_micro, PROFILES

# Hardware profiles: cpu-tiny, cpu-medium, gpu-small, npu
deployed = deploy_micro(model, target="cpu-tiny")
spec = generate_room_spec(model, target="npu")
```

### ONNX Export

```python
from plato_training import export_eisenstein, benchmark_onnx_vs_pytorch

export_eisenstein(model, "model.onnx")
benchmark_onnx_vs_pytorch(model, test_input)
```

### NPU Bridge

```python
from plato_training import npu_bridge

result = npu_bridge.run_inference(model, input_data, target="npu")
```

## CLI

```bash
plato-train train --room my-model --data data.csv
plato-train list
plato-train deploy --model my-model --target npu
```

## Data Pipeline

```python
from plato_training import DataRoom, DataSpec

spec = DataSpec(
    name="training-data",
    format="csv",
    columns=["text", "label"],
)
data_room = DataRoom(spec)
```

## Testing

```bash
pytest plato_training/tests/ tests/ --tb=short
```

## Architecture

```
plato_training/
├── __init__.py              # Public API, all exports
├── types.py                 # TrainingTile, TileLifecycle, LamportClock, configs
├── adapters/
│   └── lora.py              # LoRALayer, inject_lora, save/load
├── rooms/
│   └── lora_factory.py      # LoRAFactory — full training lifecycle
├── store.py                 # LocalTileStore
├── throttle.py              # TrainingThrottle, ThrottleLevel
├── pytorch_room.py          # PyTorchRoom
├── tensorflow_room.py       # TensorFlowRoom
├── spline.py                # SplineLinear, EisensteinLattice
├── low_rank.py              # LowRankLinear, LowRankClassifier
├── hierarchical_spline.py   # HierarchicalSplineLinear
├── micro_models.py          # train_micro, TASK_REGISTRY
├── micro_room.py            # MicroRoom, RoomFactory
├── hardware.py              # deploy_micro, PROFILES
├── data_rooms.py            # DataRoom, DataSpec
├── intelligence_room.py     # IntelligenceRoom — LLM distillation
├── semantic_matcher.py      # SemanticMatcher (Model2Vec + FAISS)
├── collective.py            # Simulation-first collective inference
├── eisenstein_encoder.py    # Eisenstein integer encoding
├── fleet_tokenizer.py       # BPE tokenizer for fleet
├── onnx_export.py           # ONNX export + benchmarking
├── npu_bridge.py            # NPU inference bridge
├── agent_field.py           # Agent field dynamics
├── i2i.py                   # Instance-to-instance communication
└── cli.py                   # Command-line interface
```

## Related Repos

- **[collective-ai](https://github.com/SuperInstance/collective-ai)** — Extracted collective inference library
- **[fleet-router](https://github.com/SuperInstance/fleet-router)** — Route AI queries to the cheapest model that won't break
- **[snapkit-python](https://github.com/SuperInstance/snapkit-python)** — Tolerance-compressed attention allocation
- **[SuperInstance-papers](https://github.com/SuperInstance/SuperInstance-papers)** — 72+ research white papers

## License

MIT
