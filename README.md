# plato-training

[![Tests](https://img.shields.io/github/actions/workflow/status/SuperInstance/plato-training/tests.yml?branch=master)](https://github.com/SuperInstance/plato-training/actions)
[![Version](https://img.shields.io/pypi/v/plato-training)](https://pypi.org/project/plato-training/)
[![License](https://img.shields.io/github/license/SuperInstance/plato-training)](https://github.com/SuperInstance/plato-training/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

> **Train micro models from fleet data. Deploy them anywhere.** Tiles carry the intelligence — small models + good procedures beat large models working from scratch.

**Status: Early-stage research.** 773 tests, 16K+ lines of Python, real fleet data flowing end-to-end. Everything works. Nothing is production-hardened.

---

## Quick Start (30 seconds)

```bash
pip install plato-training
```

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

Need more? See [demos/](demos/) for full examples.

---

## What's Inside

### Training

| Module | Purpose |
|--------|---------|
| `spline.py` | SplineLinear compression — 5-20× size reduction at identical accuracy |
| `gpt2_trainer.py` | GPT-2 micro-model trainer (416K params on fleet data) |
| `gpu_fleet_trainer.py` | GPU fleet training orchestrator |
| `collective.py` | Collective inference primitives |
| `throttle.py` | Fleet-aware training throttle |
| `micro_models.py` | 8 room tasks + training pipeline |
| `collective_loop.py` | Multi-agent collective inference (predict → listen → compare → gap → learn → share) |
| `gpt2_room.py` | GPT-2 training room — attention, BPE, next-token prediction |
| `micro_room.py` | Micro training room |
| `pytorch_room.py` | PyTorch room (LoRA + throttle) |
| `tensorflow_room.py` | TensorFlow room (Keras + throttle) |

### Semantic

| Module | Purpose |
|--------|---------|
| `tutor_judge.py` | TUTOR-style bitvector matching — 93.8% accuracy, zero ML |
| `semantic_store.py` | Model2Vec + FAISS semantic retrieval |
| `semantic_matcher.py` | SemanticMatcher for knowledge Q&A across tile corpora |
| `eisenstein_encoder.py` | Tiny contrastive encoder (71.2% hit rate, 627KB) |
| `intelligence_room.py` | 4-tier cascade matching (bitvector → fuzzy → semantic → LLM) |
| `intelligence_pre_filter.py` | Pre-filter for intelligence room |
| `intelligence_post_filter.py` | Post-filter for intelligence room |
| `intelligence_self_trainer.py` | Self-training loop for intelligence room |

### Infrastructure

| Module | Purpose |
|--------|---------|
| `device_router.py` | Heterogeneous compute router — dispatch to CPU/GPU/NPU/WASM |
| `onnx_export.py` | ONNX export pipeline for Eisenstein & SplineLinear models |
| `npu_bridge.py` | NPU bridge for edge deployment |
| `cli.py` | `plato-train` command-line interface |
| `hardware.py` | 8 hardware targets, deploy pipeline |
| `spline_hd.py` | High-dimensional SplineLinear (Eisenstein lattice) |
| `hierarchical_spline.py` | Hierarchical spline layers |
| `low_rank.py` | Low-rank linear layers |
| `fleet_miner.py` | Git history miner for SuperInstance repos |
| `data_pipeline.py` | Fleet data ingestion, feature extraction (30+ features/commit) |
| `fleet_tokenizer.py` | BPE tokenizer trained on fleet corpus |
| `triplet_miner.py` | Triplet dataset miner from git history |
| `data_rooms.py` | Data loading rooms |
| `store.py` | Content-addressed tile store |
| `plato_forge.py` | Tile forge — compile tiles from procedures |
| `i2i.py` | Instance-to-instance protocol |

### Simulation

| Module | Purpose |
|--------|---------|
| `swarm_rooms.py` | GPU-accelerated multi-agent simulation |
| `agent_field.py` | Agent field dynamics |
| `commit_predictor.py` | Predict commit patterns from fleet data |

---

## Fleet Results (48 configs: 8 tasks × 6 targets)

```
Task                  cpu   cpu-tiny   cpu-fast      gpu      npu      wasm
drift-detect       100%     100%      100%       99%     100%     100%
intent-detect      100%      75%      100%       93%     100%     100%
topic-classify     100%      29%      100%       59%     100%      34%
anomaly-flag        90%      84%       90%       84%      93%      93%
sentiment           92%      74%       70%       84%      92%      88%
```

**28 of 48 configs above 70% accuracy.** SplineLinear: 20× size reduction at identical accuracy. NPU INT8: maintains 100% on drift-detect and intent-detect.

**Honest breakdown:**
- **Strong (>90%):** drift-detect, intent-detect (most targets), anomaly-flag on NPU
- **Medium (70-90%):** anomaly-flag (cpu/gpu), sentiment (cpu/npu)
- **Weak (<70%):** topic-classify on cpu-tiny/gpu/wasm, sentiment on cpu-fast/cpu-tiny

---

## What's Real vs. What's Aspirational

**Real and tested:**
- 773 tests · SplineLinear 20× compression · 48 fleet deployment configs
- GPT-2 trainer on real fleet commit data (416K params)
- Collective inference loop · Sub-millisecond CPU inference

**Research in progress:**
- Real-world data pipelines (currently synthetic + fleet commits)
- SplineLinear scaling beyond drift-detect
- Production NPU deployment (tested in simulation)
- LoRA on real data

**Honest limitations:**
- Synthetic training data inflates accuracy numbers
- No real edge hardware deployments yet (all simulation)
- GPT-2 trainer is research-grade, not production
- The accumulation effect is demonstrated but not yet measured over many cycles

---

## Architecture

Four independent packages, each installable and testable standalone:

```
plato-types          Tile lifecycle, Lamport clocks, provenance tracking
tensor-spline        SplineLinear, LowRankLinear, Hierarchical compression
plato-data           CSV/JSONL/PLATO/fleet data loading
plato-training       Orchestrates everything — micro models, collective, deploy
```

```bash
# Optional sibling packages
pip install tensor-spline plato-data plato-types
```

---

## Documentation

- [Philosophy & Theory](docs/PHILOSOPHY.md) — tiles as procedures, capability ladder, accumulation effect, Three-Structure Theorem
- [Semantic Tutor Architecture](docs/SEMANTIC-TUTOR-ARCHITECTURE.md)
- [Heterogeneous Compute](docs/HETEROGENEOUS-COMPUTE.md)
- [Eisenstein Encoder Results](docs/EISENSTEIN-ENCODER-RESULTS.md)

## Ecosystem

| Repo | Purpose |
|------|---------|
| [plato-types](https://github.com/SuperInstance/plato-types) | Tile lifecycle, Lamport clocks |
| [tensor-spline](https://github.com/SuperInstance/tensor-spline) | SplineLinear 20× compression |
| [plato-data](https://github.com/SuperInstance/plato-data) | Data loading for PLATO rooms |
| [plato-model-ocean](https://github.com/SuperInstance/plato-model-ocean) | Evolving ecosystem of micro models |
| [plato-escalation-gate](https://github.com/SuperInstance/plato-escalation-gate) | When to escalate from micro → LLM (737 params) |
| [constraint-theory-ecosystem](https://github.com/SuperInstance/constraint-theory-ecosystem) | Theoretical foundations, TILE-IS-THE-PROCEDURE |
| [eisenstein-embed](https://github.com/SuperInstance/eisenstein-embed) | 5-layer semantic matching cascade |

## License

MIT
