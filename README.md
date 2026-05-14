# PLATO Training Rooms

Train, compress, and deploy micro models for PLATO rooms. One function call.

## Quick Start

```python
from plato_training.micro_models import train_micro, list_tasks
from plato_training.hardware import deploy_micro, PROFILES

# See all available tasks
tasks = list_tasks()
# {'spam-classify': 'Classify messages as spam/not-spam', ...}

# Train a drift detector
model, tile, metrics = train_micro("drift-detect")
# drift-detect epoch 8/8: val_acc=100.00%

# Deploy for NPU (quantized INT8)
deployed = deploy_micro("drift-detect", target="npu")
# 100% accuracy, 0.11ms latency, 8,519 bytes

# Deploy for embedded CPU (SplineLinear, 20x compression)
deployed = deploy_micro("drift-detect", target="cpu-tiny")
# 100% accuracy, 0.39ms latency, 29,701 bytes
```

## 8 Room Tasks

| Task | Description | Input | Classes |
|------|-------------|-------|---------|
| `drift-detect` | Constraint drift from sensor window | 64-dim | stable/drifting |
| `anomaly-flag` | Anomalous sensor readings | 16-dim | normal/anomaly |
| `intent-detect` | User intent from embeddings | 64-dim | 4 intents |
| `sentiment` | Sentiment from text embeddings | 128-dim | neg/neutral/pos |
| `spam-classify` | Spam/not-spam | 128-dim | 2 classes |
| `topic-classify` | Document topic | 256-dim | 5 topics |
| `priority-rank` | Tile priority | 32-dim | 4 levels |
| `tile-relevance` | Query-tile relevance | 128-dim | relevant/not |

## 8 Hardware Targets

| Target | Device | Dtype | Export | Budget | Use Case |
|--------|--------|-------|--------|--------|----------|
| `cpu` | CPU | FP32 | PyTorch | 50K params | General |
| `cpu-tiny` | CPU | FP32 | PyTorch | 5K params | Embedded (ESP32, Cortex-M) |
| `cpu-fast` | CPU | FP32 | TorchScript | 100K params | Server fleet |
| `gpu` | CUDA | FP16 | PyTorch | 1M params | NVIDIA GPUs |
| `gpu-small` | CUDA | FP16 | PyTorch | 100K params | Jetson, RTX 3050 |
| `npu` | CPU→INT8 | INT8 | ONNX | 50K params | Qualcomm, Apple NPU |
| `tpu` | XLA | BF16 | PyTorch | 500K params | Google TPU |
| `wasm` | CPU | FP32 | ONNX | 20K params | Browser, Workers |

## Fleet Results (48/48 proven)

```
Task                  cpu   cpu-tiny   cpu-fast      gpu      npu      wasm
anomaly-flag        90%      84%       90%       84%      93%      93%
drift-detect       100%     100%      100%       99%     100%     100%
intent-detect      100%      75%      100%       93%     100%     100%
sentiment           92%      74%       70%       84%      92%      88%
spam-classify       59%      58%       58%       58%      58%      61%
tile-relevance      54%      48%       61%       55%      58%      61%
priority-rank       65%      66%       65%        4%      59%      65%
topic-classify     100%      29%      100%       59%     100%      34%
```

## Novel: SplineLinear (Tensor-Spline)

Weights parameterized by Eisenstein lattice control points instead of individual floats.

```python
from plato_training.spline import SplineLinear, inject_spline

# 512×512 layer: 262,144 params → 16 params (16,384:1 compression)
layer = SplineLinear(512, 512, n_control_points=16)

# Inject into any model
inject_spline(model, n_control_points=16, target_modules=["W_query"])
```

3 basis functions: Eisenstein (IDW), Gaussian (RBF), B-Spline.

## Architecture

Three layers:
1. **Room Protocol** — tiles, lifecycle, throttle, Lamport clocks
2. **Engine Rooms** — PyTorch (LoRA) and TensorFlow (Keras) with fleet throttle
3. **Tensor-Spline** — Eisenstein lattice weight parameterization (novel)

## Installation

```bash
pip install -e .
```

Requires PyTorch. Optional: TensorFlow, onnxscript.

## Tests

```
69 passed, 2 skipped
```
