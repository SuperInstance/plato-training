"""
Micro Models — tiny trained models for specific PLATO room tasks.

Each micro model is a proof-of-concept: a few hundred parameters trained in
seconds on synthetic data, proving the pipeline works end-to-end.

Ensigns (junior agents) get these as room skills. Click-button deployment.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .types import AdapterConfig, TrainingConfig, TrainingTile, TileType, TileLifecycle
from .pytorch_room import PyTorchRoom
from .throttle import TrainingThrottle
from .spline import SplineLinear, inject_spline, compression_ratio
from .low_rank import LowRankLinear, LowRankClassifier, recommend_variant


# ─── Task Definitions ──────────────────────────────────────────────

TASK_REGISTRY: Dict[str, dict] = {
    "spam-classify": {
        "description": "Classify messages as spam/not-spam",
        "input_dim": 128,
        "num_classes": 2,
        "hidden": 32,
        "synthetic_size": 500,
        "epochs": 5,
    },
    "intent-detect": {
        "description": "Detect user intent from embedding vectors",
        "input_dim": 64,
        "num_classes": 4,  # query, command, question, chitchat
        "hidden": 32,
        "synthetic_size": 500,
        "epochs": 5,
    },
    "anomaly-flag": {
        "description": "Flag anomalous sensor readings",
        "input_dim": 16,
        "num_classes": 2,  # normal, anomaly
        "hidden": 16,
        "synthetic_size": 1000,
        "epochs": 8,
    },
    "sentiment": {
        "description": "Sentiment analysis on short text embeddings",
        "input_dim": 128,
        "num_classes": 3,  # negative, neutral, positive
        "hidden": 32,
        "synthetic_size": 500,
        "epochs": 5,
    },
    "topic-classify": {
        "description": "Classify document into topics",
        "input_dim": 256,
        "num_classes": 5,
        "hidden": 64,
        "synthetic_size": 500,
        "epochs": 5,
    },
    "priority-rank": {
        "description": "Rank tile priority (low/medium/high/critical)",
        "input_dim": 32,
        "num_classes": 4,
        "hidden": 16,
        "synthetic_size": 500,
        "epochs": 5,
    },
    "drift-detect": {
        "description": "Detect constraint drift from sensor window",
        "input_dim": 64,
        "num_classes": 2,  # drifting, stable
        "hidden": 32,
        "synthetic_size": 800,
        "epochs": 8,
    },
    "tile-relevance": {
        "description": "Score tile relevance to a query",
        "input_dim": 128,
        "num_classes": 2,  # relevant, not-relevant
        "hidden": 32,
        "synthetic_size": 600,
        "epochs": 5,
    },
}


# ─── Synthetic Data Generators ─────────────────────────────────────

def _generate_synthetic(task: str, config: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic training data for a task.
    
    Each task has structured patterns so the model can actually learn something.
    """
    n = config["synthetic_size"]
    dim = config["input_dim"]
    nc = config["num_classes"]
    
    np.random.seed(42)
    
    if task == "spam-classify":
        # Spam messages have specific "trigger" features at positions 0-7
        X = np.random.randn(n, dim).astype(np.float32) * 0.3
        labels = np.zeros(n, dtype=np.int64)
        spam_mask = np.random.rand(n) > 0.5
        # Spam: high values in first 8 features
        X[spam_mask, :8] += np.random.randn(spam_mask.sum(), 8).astype(np.float32) * 2.0
        labels[spam_mask] = 1
        
    elif task == "intent-detect":
        X = np.random.randn(n, dim).astype(np.float32) * 0.5
        labels = np.random.randint(0, nc, n)
        # Each intent activates a different quadrant of features
        for c in range(nc):
            mask = labels == c
            start = c * (dim // nc)
            end = start + (dim // nc)
            X[mask, start:end] += 1.5
            
    elif task == "anomaly-flag":
        X = np.random.randn(n, dim).astype(np.float32) * 0.5
        labels = np.zeros(n, dtype=np.int64)
        # Anomalies: extreme values in any feature
        anomaly_idx = np.random.choice(n, n // 5, replace=False)
        feat_idx = np.random.randint(0, dim, len(anomaly_idx))
        X[anomaly_idx, feat_idx] += np.random.choice([-1, 1], len(anomaly_idx)) * 5.0
        labels[anomaly_idx] = 1
        
    elif task == "sentiment":
        X = np.random.randn(n, dim).astype(np.float32) * 0.3
        labels = np.random.randint(0, nc, n)
        # Positive: high first half, negative: high second half, neutral: uniform
        X[labels == 0, :dim//2] -= 1.0  # negative
        X[labels == 2, :dim//2] += 1.0  # positive
        
    elif task == "priority-rank":
        X = np.random.randn(n, dim).astype(np.float32) * 0.5
        labels = np.zeros(n, dtype=np.int64)
        # Priority based on magnitude of first feature
        mags = np.abs(X[:, 0])
        labels[mags > 1.5] = 3  # critical
        labels[(mags > 1.0) & (mags <= 1.5)] = 2  # high
        labels[(mags > 0.5) & (mags <= 1.0)] = 1  # medium
        # rest = low (0)
        
    elif task == "drift-detect":
        # Time series window: 64 features = 8 timesteps × 8 sensors
        X = np.random.randn(n, dim).astype(np.float32) * 0.3
        labels = np.zeros(n, dtype=np.int64)
        # Drift: monotonic increase across timesteps
        drift_mask = np.random.rand(n) > 0.6
        for i in range(8):
            X[drift_mask, i*8:(i+1)*8] += i * 0.4
        labels[drift_mask] = 1
        
    elif task == "tile-relevance":
        X = np.random.randn(n, dim).astype(np.float32) * 0.4
        labels = np.zeros(n, dtype=np.int64)
        # Relevant tiles: query and tile embeddings are similar (cosine-like)
        rel_mask = np.random.rand(n) > 0.5
        # Make first half (query) and second half (tile) similar for relevant
        X[rel_mask, :dim//2] = X[rel_mask, dim//2:] + np.random.randn(rel_mask.sum(), dim//2).astype(np.float32) * 0.1
        labels[rel_mask] = 1
        
    else:
        # Generic: random patterns per class
        X = np.random.randn(n, dim).astype(np.float32)
        labels = np.random.randint(0, nc, n)
        for c in range(nc):
            mask = labels == c
            X[mask] += np.random.randn(dim).astype(np.float32) * 0.5
    
    return torch.tensor(X), torch.tensor(labels)


# ─── Model Architectures ───────────────────────────────────────────

class MicroClassifier(nn.Module):
    """Tiny classifier for room tasks. ~2K params."""
    
    def __init__(self, input_dim: int, hidden: int, num_classes: int):
        super().__init__()
        self.W_query = nn.Linear(input_dim, hidden)
        self.W_value = nn.Linear(hidden, hidden)
        self.out_head = nn.Linear(hidden, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.relu(self.W_query(x))
        x = self.dropout(x)
        x = self.relu(self.W_value(x))
        return self.out_head(x)


class SplineClassifier(nn.Module):
    """Same architecture but with SplineLinear layers. ~50 params."""
    
    def __init__(self, input_dim: int, hidden: int, num_classes: int, n_control_points: int = 16):
        super().__init__()
        self.W_query = SplineLinear(input_dim, hidden, n_control_points=n_control_points)
        self.W_value = SplineLinear(hidden, hidden, n_control_points=n_control_points)
        self.out_head = nn.Linear(hidden, num_classes)  # Keep output head dense
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.relu(self.W_query(x))
        x = self.dropout(x)
        x = self.relu(self.W_value(x))
        return self.out_head(x)


# ─── Training Pipeline ─────────────────────────────────────────────

def train_micro(
    task: str,
    variant: str = "lora",  # "lora" | "spline" | "dense"
    store_dir: str = ".plato-training",
    n_control_points: int = 16,
    throttle: Optional[TrainingThrottle] = None,
) -> Tuple[nn.Module, TrainingTile, dict]:
    """
    Train a micro model for a specific task.
    
    Returns: (trained_model, tile, metrics_dict)
    
    Args:
        task: task name from TASK_REGISTRY
        variant: "lora" (LoRA adapter), "spline" (SplineLinear), "dense" (full training)
        store_dir: where to save tiles
        n_control_points: for spline variant
        throttle: fleet throttle (default: always full)
    """
    if task not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task}. Available: {list(TASK_REGISTRY.keys())}")
    
    config = TASK_REGISTRY[task]
    
    # Generate synthetic data
    X, y = _generate_synthetic(task, config)
    
    # Split 80/20
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    
    # Build model
    if variant == "spline":
        model = SplineClassifier(config["input_dim"], config["hidden"], config["num_classes"], n_control_points)
        # Train all params (spline layers are already compressed)
        tile = _train_direct(model, train_ds, val_ds, task, config, store_dir, throttle)
    elif variant == "lora":
        model = MicroClassifier(config["input_dim"], config["hidden"], config["num_classes"])
        adapter_config = AdapterConfig(rank=4, alpha=8, target_modules=["W_query", "W_value"])
        train_config = TrainingConfig(epochs=config["epochs"], learning_rate=1e-3, batch_size=32)
        room = PyTorchRoom(task, store_dir=store_dir, throttle=throttle or TrainingThrottle(custom_load_fn=lambda: 0.1))
        tile = room.train(model, train_ds, adapter_config=adapter_config, training_config=train_config, num_classes=config["num_classes"])
        return model, tile, _eval_model(model, val_ds, config["num_classes"])
    else:  # dense
        model = MicroClassifier(config["input_dim"], config["hidden"], config["num_classes"])
        tile = _train_direct(model, train_ds, val_ds, task, config, store_dir, throttle)
    
    return model, tile, _eval_model(model, val_ds, config["num_classes"])


def _train_direct(model, train_ds, val_ds, task, config, store_dir, throttle):
    """Train all parameters directly (for spline and dense variants)."""
    from .types import LamportClock, TrainingMetrics, content_hash
    from .store import LocalTileStore
    import time
    
    clock = LamportClock()
    store = LocalTileStore(store_dir)
    if throttle is None:
        throttle = TrainingThrottle(custom_load_fn=lambda: 0.1)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    start = time.time()
    loss_curve = []
    
    for epoch in range(config["epochs"]):
        state = throttle.check()
        model.train()
        loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        
        epoch_loss = 0.0
        for batch in loader:
            x, y = batch
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            loss_curve.append(loss.item())
        
        # Eval
        val_acc = _eval_model(model, val_ds, config["num_classes"])["accuracy"]
        print(f"  {task} epoch {epoch+1}/{config['epochs']}: loss={epoch_loss/len(loader):.4f} val_acc={val_acc:.2%}")
    
    train_time = time.time() - start
    
    # Save model weights as tile
    lamport = clock.tick()
    buf = torch.save(model.state_dict(), "/tmp/_micro_weights.pt")
    import pathlib
    weight_bytes = pathlib.Path("/tmp/_micro_weights.pt").read_bytes()
    c_hash = content_hash(weight_bytes)
    store.save_weights(c_hash, weight_bytes)
    
    tile = TrainingTile(
        tile_id=f"micro-{task}-{lamport:03d}",
        room=f"micro-{task}",
        tile_type=TileType.ADAPTER,
        state=TileLifecycle.ACTIVE,
        lamport=lamport,
        name=f"micro-{task}",
        description=f"Micro model for {config['description']}",
        content_hash=c_hash,
        metrics=TrainingMetrics(
            final_loss=loss_curve[-1] if loss_curve else 0.0,
            epochs_completed=config["epochs"],
            training_time_seconds=train_time,
            loss_curve=loss_curve,
        ),
    )
    store.save(tile)
    return tile


def _eval_model(model, val_ds, num_classes):
    """Evaluate model accuracy and per-class metrics."""
    model.eval()
    loader = DataLoader(val_ds, batch_size=64)
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
            for c in range(num_classes):
                mask = y == c
                class_correct[c] += (pred[mask] == c).sum().item()
                class_total[c] += mask.sum().item()
    
    return {
        "accuracy": correct / max(total, 1),
        "per_class": {
            c: class_correct[c] / max(class_total[c], 1)
            for c in range(num_classes)
        },
        "total_samples": total,
    }


def list_tasks() -> Dict[str, str]:
    """List available micro model tasks."""
    return {k: v["description"] for k, v in TASK_REGISTRY.items()}


def bench_all_tasks(store_dir: str = ".plato-training") -> Dict[str, dict]:
    """
    Train all micro models across all variants and return results.
    
    This is the proof-of-concept: every task, every variant, one function call.
    """
    results = {}
    
    for task in TASK_REGISTRY:
        results[task] = {}
        for variant in ["dense", "lora", "spline"]:
            try:
                model, tile, metrics = train_micro(task, variant=variant, store_dir=store_dir)
                
                # Count params
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                results[task][variant] = {
                    "accuracy": metrics["accuracy"],
                    "params": total_params,
                    "trainable": trainable_params,
                    "train_time": tile.metrics.training_time_seconds if tile.metrics else 0,
                    "loss": tile.metrics.final_loss if tile.metrics else 0,
                    "tile_id": tile.tile_id,
                    "status": "OK",
                }
            except Exception as e:
                results[task][variant] = {"status": f"FAIL: {e}"}
    
    return results
