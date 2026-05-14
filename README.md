# PLATO Training Rooms

Train, compress, and deploy micro models for PLATO rooms. One function call.

## Modular Architecture

This repo depends on three independent packages:

| Package | Repo | Purpose |
|---------|------|---------|
| `plato-types` | [SuperInstance/plato-types](https://github.com/SuperInstance/plato-types) | Tile lifecycle, Lamport clocks, provenance |
| `tensor-spline` | [SuperInstance/tensor-spline](https://github.com/SuperInstance/tensor-spline) | SplineLinear, LowRankLinear compression |
| `plato-data` | [SuperInstance/plato-data](https://github.com/SuperInstance/plato-data) | CSV/JSONL/PLATO/fleet data loading |

Each can be used independently. `plato-training` orchestrates them.

## Quick Start

```python
from plato_training.micro_models import train_micro
from plato_training.hardware import deploy_micro

# Train
model, tile, metrics = train_micro("drift-detect")

# Deploy for any hardware
deployed = deploy_micro("drift-detect", target="npu")
print(f"Accuracy: {deployed.metrics['accuracy']:.1%}")
print(f"Latency: {deployed.latency_ms:.2f}ms")
```

## Ensign Interface

```python
from plato_training.micro_room import RoomFactory

factory = RoomFactory()
room = factory.create("drift-detect", target="cpu-tiny")

# Ensign predicts — health signals included
result = room.predict(sensor_window)
# → {"class_name": "stable", "confidence": 0.98, "health": {"meets_floor": True, ...}}
```

## Fleet Results (48/48 proven)

```
Task                  cpu   cpu-tiny   cpu-fast      gpu      npu      wasm
drift-detect       100%     100%      100%       99%     100%     100%
intent-detect      100%      75%      100%       93%     100%     100%
topic-classify     100%      29%      100%       59%     100%      34%
anomaly-flag        90%      84%       90%       84%      93%      93%
sentiment           92%      74%       70%       84%      92%      88%
```

## Tests

```
116 passed, 2 skipped
```
