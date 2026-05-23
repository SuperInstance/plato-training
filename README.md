# plato-training

[![Tests](https://img.shields.io/github/actions/workflow/status/SuperInstance/plato-training/tests.yml?branch=master)](https://github.com/SuperInstance/plato-training/actions)
[![Version](https://img.shields.io/pypi/v/plato-training)](https://pypi.org/project/plato-training/)
[![License](https://img.shields.io/github/license/SuperInstance/plato-training)](https://github.com/SuperInstance/plato-training/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

Train micro models from fleet data. Deploy them anywhere. Tiles carry the intelligence — small models + good procedures beat large models working from scratch.

**Status: Early-stage research.** 773 tests, 16K+ lines of Python, real fleet data flowing end-to-end. Everything works. Nothing is production-hardened.

## Install

```bash
pip install plato-training
```

## Quick Start

```python
from plato_training.micro_models import train_micro
from plato_training.hardware import deploy_micro

# Train a micro model on a fleet task
model, tile, metrics = train_micro("drift-detect")

# Deploy to any hardware target (cpu, gpu, npu, wasm)
deployed = deploy_micro("drift-detect", target="npu")
print(f"Accuracy: {deployed.metrics['accuracy']:.1%}")  # 100%
print(f"Latency: {deployed.latency_ms:.2f}ms")           # <1ms
```

See [demos/](demos/) for full examples.

## The Key Idea: Tile Lifecycle

Tiles in PLATO follow a strict lifecycle:

```
Active ──→ Superseded ──→ Retracted
```

- **Active**: The tile is current. Use it.
- **Superseded**: A better tile replaced it. Still valid, just not latest.
- **Retracted**: Withdrawn. Treat as if it never existed.

You never mutate a tile. You supersede it. This gives you a full provenance chain — every training run, every model version, tracked and ordered with Lamport clocks.

## What's Inside

### Training

| Module | Purpose |
|--------|---------|
| `micro_models.py` | 8 room tasks + training pipeline |
| `gpt2_trainer.py` | GPT-2 micro-model (416K params on fleet data) |
| `gpu_fleet_trainer.py` | GPU fleet training orchestrator |
| `spline.py` | SplineLinear compression — 5-20× size reduction |
| `collective_loop.py` | Multi-agent collective inference |
| `gpt2_room.py` | GPT-2 training room (attention, BPE, next-token) |
| `pytorch_room.py` | PyTorch room (LoRA + throttle) |
| `tensorflow_room.py` | TensorFlow room (Keras + throttle) |

### Semantic

| Module | Purpose |
|--------|---------|
| `intelligence_room.py` | 4-tier cascade (bitvector → fuzzy → semantic → LLM) |
| `semantic_matcher.py` | Knowledge Q&A across tile corpora |
| `semantic_store.py` | Model2Vec + FAISS retrieval |
| `eisenstein_encoder.py` | Contrastive encoder (71.2% hit rate, 627KB) |
| `tutor_judge.py` | TUTOR bitvector matching (93.8% accuracy, zero ML) |

### Infrastructure

| Module | Purpose |
|--------|---------|
| `device_router.py` | Dispatch to CPU/GPU/NPU/WASM |
| `hardware.py` | 8 hardware targets, deploy pipeline |
| `onnx_export.py` | ONNX export for Eisenstein & SplineLinear |
| `cli.py` | `plato-train` CLI |
| `fleet_miner.py` | Git history miner for fleet data |
| `data_pipeline.py` | Fleet data ingestion (30+ features/commit) |

## Fleet Results

48 deployment configs (8 tasks × 6 hardware targets):

```
Task                  cpu   cpu-tiny   cpu-fast      gpu      npu      wasm
drift-detect       100%     100%      100%       99%     100%     100%
intent-detect      100%      75%      100%       93%     100%     100%
topic-classify     100%      29%      100%       59%     100%      34%
anomaly-flag        90%      84%       90%       84%      93%      93%
sentiment           92%      74%       70%       84%      92%      88%
```

28 of 48 configs above 70% accuracy. SplineLinear: 20× size reduction at identical accuracy.

### Honest Assessment

- **Strong (>90%):** drift-detect, intent-detect, anomaly-flag on NPU
- **Medium (70-90%):** anomaly-flag (cpu/gpu), sentiment (cpu/npu)
- **Weak (<70%):** topic-classify on cpu-tiny/gpu/wasm, sentiment on cpu-fast
- **Limitations:** Synthetic data inflates numbers. No real edge deployments yet. GPT-2 trainer is research-grade.

## Documentation

| Document | Description |
|----------|-------------|
| [Philosophy & Theory](docs/PHILOSOPHY.md) | Tiles as procedures, capability ladder, accumulation effect |
| [Semantic Tutor Architecture](docs/SEMANTIC-TUTOR-ARCHITECTURE.md) | TUTOR-style matching design |
| [Heterogeneous Compute](docs/HETEROGENEOUS-COMPUTE.md) | Multi-target deployment |
| [Eisenstein Encoder Results](docs/EISENSTEIN-ENCODER-RESULTS.md) | V1/V2/V3 contrastive encoder benchmarks |
| [Heterogeneous Benchmark (Full)](docs/HETEROGENEOUS-BENCHMARK-FULL.md) | Complete fleet deployment results |
| [Architecture V2](docs/ARCHITECTURE-V2.md) | System architecture overview |
| [docs/README.md](docs/README.md) | Full documentation index |

## Ecosystem

| Repo | Purpose |
|------|---------|
| [plato-core](https://github.com/SuperInstance/plato-core) | Foundation types + mesh registry |
| [plato-types](https://github.com/SuperInstance/plato-types) | Tile lifecycle, Lamport clocks |
| [plato-mcp](https://github.com/SuperInstance/plato-mcp) | PLATO rooms as MCP tools |
| [tensor-spline](https://github.com/SuperInstance/tensor-spline) | SplineLinear 20× compression |
| [plato-data](https://github.com/SuperInstance/plato-data) | Data loading for PLATO rooms |
| [plato-model-ocean](https://github.com/SuperInstance/plato-model-ocean) | Evolving ecosystem of micro models |

## License

MIT
