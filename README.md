# PLATO Training Rooms

**Train, compress, and deploy micro models — 100% accuracy on drift detection, one function call.**

[![Tests](https://img.shields.io/badge/tests-116%20passing-green)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

## Why?

Ship tiny, purpose-built models to any hardware — CPU, GPU, NPU, WASM, embedded. One `train_micro()` call trains a model, one `deploy_micro()` call quantizes and packages it for your target. Fleet results prove it: drift-detect hits 100% on 5 of 6 hardware targets, intent-detect 100% on NPU.

Use this when you need models smaller than 1KB that run in sub-millisecond latency on edge devices.

## Install

```bash
pip install plato-training
```

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

## Related

- **[plato-types](https://github.com/SuperInstance/plato-types)** — Tile lifecycle, Lamport clocks
- **[tensor-spline](https://github.com/SuperInstance/tensor-spline)** — SplineLinear 20× compression
- **[plato-data](https://github.com/SuperInstance/plato-data)** — Data loading for PLATO rooms
- **[plato-model-ocean](https://github.com/SuperInstance/plato-model-ocean)** — Evolving ecosystem of micro models
- **[plato-escalation-gate](https://github.com/SuperInstance/plato-escalation-gate)** — When to escalate to LLM (737 params)
- **[ASSEMBLY-GUIDE](https://github.com/SuperInstance/plato-training/blob/master/ASSEMBLY-GUIDE.md)** — Full ecosystem assembly guide

## Tests

```
116 passed, 2 skipped
```
