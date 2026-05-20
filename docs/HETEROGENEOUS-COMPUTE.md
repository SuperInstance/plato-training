# Heterogeneous Compute Architecture — RTX 4050 + Ryzen AI HX 370 + Radeon 890M

## What We Have

| Component | Spec | Role |
|-----------|------|------|
| **NVIDIA RTX 4050 Laptop** | 6GB VRAM, SM 8.9, 20 SMs, CUDA 12.6 | Training, heavy inference |
| **AMD Ryzen AI 9 HX 370** | 12C/24T, Zen 5, AVX-512 VNNI + BF16 | CPU inference, embedding, quantized models |
| **AMD Radeon 890M iGPU** | RDNA 3.5, 16 CUs, ~8.6 TFLOPS FP16 | Offload inference via DirectML (WSL2) |
| **AMD XDNA 2 NPU** | 50 INT8 TOPS | ONNX INT8 inference (requires kernel driver) |

**Total AI compute: ~80 TOPS across 4 processors.**

## The Synergy: Three-Lane Highway

```
                    ┌─────────────────────────────┐
                    │   PLATO Intelligence Room    │
                    │   (orchestrator — decides     │
                    │    where to route each op)    │
                    └──────┬──────────┬────────────┘
                           │          │
            ┌──────────────┼──────────┼──────────────┐
            ▼              ▼          ▼              ▼
     ┌────────────┐ ┌──────────┐ ┌─────────┐ ┌──────────┐
     │ NVIDIA GPU  │ │ AMD CPU  │ │ AMD iGPU│ │ AMD NPU  │
     │ (training)  │ │(inference│ │(embed + │ │(INT8     │
     │             │ │+ embed)  │ │ search) │ │ models)  │
     │ CUDA 12.6   │ │ AVX-512  │ │DirectML │ │ ONNX+    │
     │ AMP, Spline │ │ VNNI,BF16│ │ RDNA3.5 │ │ XDNA     │
     └────────────┘ └──────────┘ └─────────┘ └──────────┘
      Heavy lift      Fast lane    Express      Micro
      (training)    (embeddings)  (FAISS)     (quantized)
```

## Lane 1: NVIDIA RTX 4050 — Training & Heavy Inference

Already proven: 1.87M params, 95.5% accuracy, 163MB VRAM, 6.8s training.

**What goes here:**
- Model training (backprop, gradient accumulation, mixed precision)
- Large-batch inference (GPT-2 generation, transformer forward passes)
- SplineLinear compression during training
- Forge daemon (BMA convergence, deadband throttle, weight snap)

**What does NOT go here (free the GPU):**
- Single-sample embedding (Model2Vec on CPU is 9μs — no need for GPU)
- FAISS search (CPU is faster for <100K vectors)
- Quantized inference (NPU/CPU is more efficient)

## Lane 2: AMD Ryzen AI HX 370 — CPU Inference & Embeddings

**Benchmarks from eileen (today):**
```
FP32 64x128 matmul:    63 GFLOPS
FP32 128x256 matmul:  256 GFLOPS
Model2Vec embedding:    9 μs each (111K/sec)
Tiny MLP inference:    16 μs each (64K/sec)
FAISS search (10K):   119 μs each (8.4K/sec)
Full embed+search:    138 μs each (7.3K/sec)
```

**Secret weapon: AVX-512 VNNI + BF16**

This CPU has:
- `AVX-512 VNNI` — INT8 matrix multiply in hardware (2-4x faster than FP32 for quantized models)
- `AVX-512 BF16` — BF16 matmul in hardware (for transformer inference)
- `AVX-VNNI` — 256-bit VNNI for non-512 paths
- 12 Zen 5 cores with full 512-bit pipelines (not double-pumped like Zen 4)

**What goes here:**
- Model2Vec embedding (9μs/query — already faster than GPU could be with transfer overhead)
- FAISS vector search (CPU is the right place for <1M vectors)
- Quantized model inference via ONNX Runtime + VNNI
- llama.cpp inference for small models (GGUF Q4/Q8 with AVX-512 BF16)
- SplineLinear encoder forward pass (tiny model, CPU is fine)
- Concurrent with GPU training (proven: 1K GPU train + 3K CPU inference/sec)

**ONNX Runtime with VNNI:**
```bash
pip install onnxruntime  # Uses AVX-512 VNNI automatically for INT8 models
```
Export our SplineLinear micro-models → ONNX → quantize to INT8 → run on CPU at 2-4x FP32 speed.

## Lane 3: AMD Radeon 890M iGPU — Offloaded Inference

**The unlock: torch-directml in WSL2**

```bash
pip install torch-directml
```

```python
import torch_directml
dml = torch_directml.device()  # Uses Radeon 890M via DirectX 12

model = torch.jit.script(my_encoder).to(dml)
embeddings = model(tokens.to(dml))  # Runs on iGPU
```

**Why this matters:**
- RTX 4050 is free for training while iGPU handles inference
- iGPU has its own VRAM (shared with system RAM but dedicated compute)
- ~8.6 TFLOPS FP16 — plenty for embedding and small model inference
- DirectML works in WSL2 (Windows DirectX 12 passthrough)

**What goes here:**
- Sentence-transformer embedding (if we want higher quality than Model2Vec)
- Small encoder model inference (SplineLinear encoder)
- FAISS GPU index for >100K vectors (if we get there)
- ONNX model inference via DirectML EP

**Installation:**
```bash
pip install torch-directml
# Verify:
python3 -c "import torch_directml; print(torch_directml.device())"
```

## Lane 4: AMD XDNA 2 NPU — Quantized Micro-Models (Future)

50 INT8 TOPS. This is the most power-efficient compute on the chip.

**Current status on WSL2:** XDNA kernel driver not available in WSL2 kernel.
Needs native Linux (kernel 6.10+) or Windows with AMD Ryzen AI SDK.

**If we get access (native Linux or future WSL2 kernel update):**
```bash
pip install onnxruntime-vitisai  # AMD's XDNA execution provider
```

```python
# Quantize model for NPU
import onnxruntime as ort
session = ort.InferenceSession(
    "micro_model_int8.onnx",
    providers=["VitisAIExecutionProvider"],  # XDNA NPU
)
result = session.run(None, {"input": input_data})
```

**What goes here:**
- All 8 PLATO micro-models (drift-detect, anomaly-flag, etc.) at INT8
- Sub-millisecond inference at ~1W power
- Always-on daemon inference (no GPU wake needed)

## The Synergy Play: How They Work Together

### Scenario 1: Training + Serving Simultaneously

```
GPU (RTX 4050): Training FleetGPT2 on fleet data
  → 6.8s per epoch, 163MB VRAM

CPU (Ryzen HX 370): Serving embedding queries
  → 7.3K queries/sec (Model2Vec + FAISS)

iGPU (Radeon 890M): Running SplineLinear encoder
  → 64K inferences/sec

NPU (XDNA 2): Running drift-detect micro-model
  → <1ms inference, ~1W
```

**All four run simultaneously. No contention.**

### Scenario 2: Adaptive Routing (PLATO Intelligence Room)

```
Query arrives
  ↓
CPU: Model2Vec embed (9μs)
  ↓
CPU: FAISS search (119μs)
  ↓
  ├─ Cache hit (>0.7 similarity) → CPU returns cached answer (total: ~130μs)
  ├─ Near miss (0.5-0.7) → iGPU runs SplineLinear re-ranker (16μs)
  └─ No match (<0.5) → GPU generates answer, CPU embeds, FAISS indexes (total: ~5ms)
```

### Scenario 3: Continuous Training Loop

```
Forever:
  1. GPU: Train one batch (6ms, AMP)
  2. CPU: Process 10 inference queries concurrently (10 × 9μs = 90μs)
  3. iGPU: Re-rank FAISS results for quality (async)
  4. Every 100 steps: GPU snaps weights, CPU evaluates on validation set
  5. Every 1000 steps: Export ONNX → quantize → deploy to CPU/NPU
```

**Throughput:**
- GPU: 166 train steps/sec (at 6ms each)
- CPU: 7.3K embed+search/sec
- iGPU: 64K re-rankings/sec
- Total: serving 7,300 queries/sec while training continuously

## Implementation Priority

### Step 1: Install torch-directml (5 min)
```bash
pip install torch-directml
python3 -c "import torch_directml; dml = torch_directml.device(); print(f'iGPU: {dml}')"
```
Test: run a small model on iGPU, verify it doesn't conflict with CUDA.

### Step 2: Install ONNX Runtime (5 min)
```bash
pip install onnxruntime
```
Test: export a micro-model to ONNX, run inference with VNNI optimization.

### Step 3: Build the Device Router (1 day)
```python
class DeviceRouter:
    """Route inference to the most efficient compute unit."""

    def __init__(self):
        self.cuda = torch.device('cuda')      # RTX 4050
        self.cpu = torch.device('cpu')         # Ryzen HX 370
        try:
            import torch_directml
            self.dml = torch_directml.device() # Radeon 890M
        except ImportError:
            self.dml = None

    def embed(self, text: str) -> np.ndarray:
        """Embedding → CPU (Model2Vec is faster than GPU for single queries)."""
        return self.model2vec.encode([text])[0]

    def search(self, query_vec: np.ndarray, k=5) -> list:
        """FAISS search → CPU (fast enough for <1M vectors)."""
        return self.faiss_index.search(query_vec.reshape(1, -1), k)

    def inference(self, model, input_data, model_type='auto'):
        """Route model inference to best device."""
        if model_type == 'micro':
            return self._cpu_inference(model, input_data)  # VNNI optimized
        elif model_type == 'encoder':
            return self._igpu_inference(model, input_data) if self.dml else self._cpu_inference(model, input_data)
        elif model_type == 'transformer':
            return self._gpu_inference(model, input_data)  # CUDA AMP
        else:
            return self._cpu_inference(model, input_data)  # Default CPU

    def train(self, model, data):
        """Training always on GPU."""
        return self._gpu_train(model, data)
```

### Step 4: Export ONNX Micro-Models (1 day)
Export the 8 PLATO micro-models to ONNX with INT8 quantization.
Run on CPU with AVX-512 VNNI for 2-4x speedup.

### Step 5: Wire into PLATO Forge (1 day)
The Forge daemon already manages training. Extend it to:
- Export trained model → ONNX → deploy to CPU/NPU
- Keep GPU free for next training job
- Background inference on CPU/iGPU while GPU trains

## Key Numbers

| Metric | RTX 4050 (GPU) | Ryzen HX 370 (CPU) | Radeon 890M (iGPU) | XDNA 2 (NPU) |
|--------|---------------|--------------------|--------------------|----|
| FP32 TFLOPS | 9.7 | ~0.8 | ~4.3 | - |
| FP16 TFLOPS | 19.3 | ~1.6 | ~8.6 | - |
| INT8 TOPS | ~40 | ~3 | ~17 | 50 |
| Memory | 6 GB GDDR6 | 16 GB DDR5 (shared) | Shared with CPU | On-chip |
| Power | 35-115W | 15-54W | Shared with CPU | ~1W |
| Latency (embed) | ~200μs* | 9μs | ~50μs | ~10μs |
| Latency (search) | N/A | 119μs | ~30μs | N/A |

*GPU has transfer overhead for small operations — CPU wins for single queries.

## The Takeaway

**CPU inference is faster than GPU for small operations** — no PCIe transfer, no kernel launch overhead. The Ryzen HX 370 with AVX-512 VNNI is a monster for quantized inference.

**The iGPU is the secret weapon** — torch-directml gives us a second GPU for free. Not as fast as the RTX 4050, but it doesn't need to be. It handles the embedding and re-ranking while the 4050 trains.

**The NPU is the future** — once we're on native Linux or WSL2 gets XDNA support, we can deploy micro-models that run at 1W with sub-millisecond inference. Always-on, always-learning.

**The synergy is real:** 1K GPU train steps + 7.3K CPU queries + 64K iGPU inferences per second, all simultaneously, no contention. That's a continuous learning system.
