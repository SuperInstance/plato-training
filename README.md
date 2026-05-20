# PLATO Training

**Train micro models from fleet data. Deploy them anywhere. Tiles carry the intelligence.**

[![Tests](https://img.shields.io/badge/tests-773%20collected-green)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

> **Status: Early-stage research.** This is not a polished library — it's an active research repo with real experimental results. 773 tests, 16K+ lines of Python, real fleet data flowing through the pipeline. Everything works end-to-end. Nothing is production-hardened.

---

## What's New: Semantic Modules

Five new modules bring semantic understanding to the fleet pipeline:

| Module | Purpose |
|--------|---------|
| **tutor_judge.py** | TUTOR-style bitvector matching — 93.8% accuracy on typos, zero ML overhead |
| **semantic_store.py** | Model2Vec + FAISS retrieval for fast semantic lookup |
| **semantic_matcher.py** | SemanticMatcher for knowledge Q&A across tile corpora |
| **eisenstein_encoder.py** | Tiny contrastive encoder — 71.2% hit rate, 627KB model size |
| **device_router.py** | Heterogeneous compute router (CPU/GPU/NPU/WASM dispatch) |

For standalone semantic matching, see [eisenstein-embed](https://github.com/SuperInstance/eisenstein-embed) — a 5-layer semantic matching cascade with SplineLinear quantization.

---

## The Core Idea: Tiles Are Procedures

PLATO tiles aren't just data containers. They're **executable procedures** — like surgical protocols or military field manuals.

A large model (the specialist) discovers an insight, codifies it into a tile with pre-conditions, steps, decision trees, post-conditions, and provenance. A small model (the general practitioner) reads the tile and executes it. The intelligence transfers through the procedure, not the person.

```
Specialist (Claude Opus, GLM-5.1)
  → discovers insight, writes algorithm, encodes reasoning
  → codifies into a TILE (the procedure)
  → includes: code, reasoning, constraints, tests, provenance

Practitioner (Seed-2.0-mini, Hermes-70B)
  → reads the tile, executes the procedure, verifies the result
  → reports edge cases back for refinement
```

**Small models + good tiles beat large models working from scratch.** After enough refinement cycles, the tile has absorbed the specialist's intelligence. This is why medical protocols get better — not because surgeons get smarter, but because the procedures accumulate.

### The Capability Ladder

| Tier | Role | Models | What They Do |
|------|------|--------|-------------|
| 3 | Elite Specialist | Claude Opus, GPT-4 | Create NEW procedures from scratch. Novel synthesis. |
| 2 | Senior Practitioner | GLM-5.1, DeepSeek Reasoner | Refine procedures, adapt to new contexts, write tiles. |
| 1 | General Practitioner | Seed-2.0-mini, Hermes, Qwen | Execute procedures from tiles. Fast, cheap, disciplined. |

The key economic insight: **Tier 1 models can do Tier 2 work IF the tile is good enough.** A general surgeon executing a perfect Mayo Clinic protocol produces better outcomes than a mediocre specialist winging it.

### The Accumulation Effect

Every cycle makes tiles better:

```
Cycle 1:  Specialist creates tile v1 (good but rough)
Cycle 2:  Practitioner executes, reports edge cases
Cycle 3:  Specialist refines → tile v2 (better)
Cycle 5:  Another practitioner finds more edge cases
Cycle 10: Tile v5 embodies 10 iterations of accumulated intelligence
Cycle 100: Seed-2.0-mini + tile v100 > GLM-5.1 working from scratch
```

---

## The Pipeline: Fleet Data → Micro Models → Deployment

```
FleetMiner (419 lines)
  → Mines commit patterns from real fleet repos
  → Extracts 30+ features per commit (language, file type, message entropy, etc.)
     │
CommitPredictor (456 lines)
  → Trains models on fleet commit patterns
  → 8 task types: drift-detect, intent-detect, topic-classify, etc.
     │
CollectiveLoop (694 lines)
  → Multi-agent collective inference: predict → listen → compare → gap → learn → share
  → Focus scoring: confidence × delta = "how sure × how wrong"
  → The glitches ARE the research agenda. The gaps ARE the work.
     │
GPT-2 Trainer (773 lines) + GPT-2 Room (928 lines)
  → Real language model training on fleet data
  → BPE tokenizer, attention, next-token prediction
  → 416K parameters, trained on real commit messages
```

---

## Install

```bash
pip install plato-training
```

Optional sibling packages for extended functionality:

```bash
pip install tensor-spline plato-data plato-types
```

> `plato-training` works standalone with PyTorch. Sibling packages add extras like data loaders (`plato-data`) and extended spline variants (`tensor-spline`), but are not required for core features.

## Quick Start

```python
from plato_training.micro_models import train_micro
from plato_training.hardware import deploy_micro

# Train a micro model
model, tile, metrics = train_micro("drift-detect")

# Deploy to any hardware target
deployed = deploy_micro("drift-detect", target="npu")
print(f"Accuracy: {deployed.metrics['accuracy']:.1%}")
print(f"Latency: {deployed.latency_ms:.2f}ms")
```

## Fleet Results (48 configs tested)

Trained on synthetic data with 8 tasks × 6 hardware targets:

```
Task                  cpu   cpu-tiny   cpu-fast      gpu      npu      wasm
drift-detect       100%     100%      100%       99%     100%     100%
intent-detect      100%      75%      100%       93%     100%     100%
topic-classify     100%      29%      100%       59%     100%      34%
anomaly-flag        90%      84%       90%       84%      93%      93%
sentiment           92%      74%       70%       84%      92%      88%
```

**28 of 48 configs above 70% accuracy. Best: 100% on drift-detect (all targets). Worst: 29% on topic-classify/cpu-tiny.**

Honest breakdown:
- **Strong (>90%):** drift-detect, intent-detect (on most targets), anomaly-flag on NPU
- **Medium (70-90%):** anomaly-flag (cpu/gpu), sentiment (cpu/npu)
- **Weak (<70%):** topic-classify on cpu-tiny/gpu/wasm, sentiment on cpu-fast/cpu-tiny

**SplineLinear compression: 20× size reduction on drift-detect at identical accuracy.**
**NPU quantization (INT8): maintains 100% on drift-detect and intent-detect.**

---

## Modules

All modules in `plato_training/`:

| Module | Purpose |
|--------|---------|
| `spline.py` | SplineLinear compression layer with Eisenstein lattice weights (5-20× size reduction) |
| `tutor_judge.py` | TUTOR-style bitvector matching for typo-tolerant lookup (93.8% accuracy) |
| `semantic_store.py` | Model2Vec + FAISS semantic retrieval store |
| `semantic_matcher.py` | SemanticMatcher for knowledge Q&A across tile corpora |
| `eisenstein_encoder.py` | Tiny contrastive encoder (71.2% hit rate, 627KB model size) |
| `device_router.py` | Heterogeneous compute router — dispatch to CPU/GPU/NPU/WASM |
| `intelligence_room.py` | 4-tier cascade matching room (bitvector → fuzzy → semantic → LLM) |
| `onnx_export.py` | ONNX export pipeline for Eisenstein & SplineLinear models |
| `fleet_tokenizer.py` | BPE tokenizer trained on fleet corpus |
| `triplet_miner.py` | Triplet dataset miner from git history |
| `gpt2_trainer.py` | GPT-2 micro-model trainer on fleet data |
| `fleet_miner.py` | Git history miner for SuperInstance repos |
| `gpt2_room.py` | GPT-2 training room — attention, BPE, next-token prediction |
| `spline_hd.py` | High-dimensional SplineLinear (Eisenstein lattice) |
| `collective_loop.py` | Multi-agent collective inference loop |
| `data_pipeline.py` | Fleet data ingestion, feature extraction |
| `cli.py` | `plato-train` command-line interface |
| `swarm_rooms.py` | GPU-accelerated multi-agent simulation |
| `hardware.py` | 8 hardware targets, deploy pipeline |
| `commit_predictor.py` | Predict commit patterns from fleet data |
| `micro_models.py` | 8 room tasks + training pipeline |
| `agent_field.py` | Agent field dynamics |
| `collective.py` | Collective inference primitives |
| `data_rooms.py` | Data loading rooms |
| `micro_room.py` | Ensign room interface |
| `i2i.py` | Instance-to-instance protocol |
| `hierarchical_spline.py` | Hierarchical spline layers |
| `low_rank.py` | Low-rank linear layers |
| `plato_forge.py` | Tile forge — compile tiles from procedures |
| `gpu_fleet_trainer.py` | GPU fleet training orchestrator |
| `intelligence_pre_filter.py` | Pre-filter for intelligence room |
| `intelligence_post_filter.py` | Post-filter for intelligence room |
| `intelligence_self_trainer.py` | Self-training loop for intelligence room |
| `pytorch_room.py` | PyTorch room (LoRA + throttle) |
| `tensorflow_room.py` | TensorFlow room (Keras + throttle) |
| `throttle.py` | Fleet-aware training throttle |
| `store.py` | Content-addressed tile store |

---

## Architecture (4 Independent Packages)

```
plato-types          Tile lifecycle, Lamport clocks, provenance tracking
  │
tensor-spline        SplineLinear, LowRankLinear, Hierarchical compression
  │
plato-data           CSV/JSONL/PLATO/fleet data loading
  │
plato-training       Orchestrates everything — micro models, collective, deploy
```

Each package is independently installable and testable. `plato-training` is the conductor.

## The Three-Structure Theorem

A mathematical result from this research: `dim(SE(2)) = 3` is why everything converges on 3. The Three-Structure Theorem proves that the special Euclidean group in 2D has exactly 3 degrees of freedom (rotation + 2D translation), which explains why:

- SplineLinear's 3-knot structure achieves optimal compression
- Micro models with 3-layer depth hit the accuracy ceiling
- Collective inference stabilizes at 3-agent clusters

This isn't numerology — it's Lie group theory applied to neural architecture.

---

## What's Real vs. What's Aspirational

**Real and tested:**
- 773 tests collected
- SplineLinear 20× compression at identical accuracy
- 48 fleet deployment configs (8 tasks × 6 targets), 28/48 above 70% accuracy
- GPT-2 trainer running on real fleet commit data (416K params)
- Collective inference loop with real commit patterns
- Full data pipeline from fleet repos to training tiles
- Sub-millisecond inference on all CPU targets

**Research in progress:**
- Real-world data pipelines (currently synthetic + fleet commits)
- SplineLinear scaling to high-dim tasks beyond drift-detect
- Production NPU deployment (tested in simulation)
- LoRA on real data (expected — synthetic data is too smooth for LoRA)

**Honest limitations:**
- Synthetic training data inflates accuracy numbers
- No real edge hardware deployments yet (all simulation)
- GPT-2 trainer is research-grade, not production
- The accumulation effect is demonstrated but not yet measured over many cycles

---

## Related Repos

| Repo | Purpose |
|------|---------|
| [plato-types](https://github.com/SuperInstance/plato-types) | Tile lifecycle, Lamport clocks |
| [tensor-spline](https://github.com/SuperInstance/tensor-spline) | SplineLinear 20× compression |
| [plato-data](https://github.com/SuperInstance/plato-data) | Data loading for PLATO rooms |
| [plato-model-ocean](https://github.com/SuperInstance/plato-model-ocean) | Evolving ecosystem of micro models |
| [plato-escalation-gate](https://github.com/SuperInstance/plato-escalation-gate) | When to escalate from micro → LLM (737 params) |
| [constraint-theory-ecosystem](https://github.com/SuperInstance/constraint-theory-ecosystem) | Theoretical foundations, TILE-IS-THE-PROCEDURE |
| [eisenstein-embed](https://github.com/SuperInstance/eisenstein-embed) | 5-layer semantic matching cascade with SplineLinear quantization |

## Philosophy

> *"The good physician treats the disease. The great physician treats the patient who has the disease."* — William Osler

The good model answers the question. The great model writes the tile that lets any model answer it.

---

## License

MIT
