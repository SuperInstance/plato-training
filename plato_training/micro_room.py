"""
PLATO Micro Rooms — deploy micro models as PLATO room skills.

An ensign (junior agent) gets a room with a micro model:
1. Room is created with task + target hardware
2. Ensign sends input tensor to room
3. Room runs inference and returns prediction
4. All invocations are logged as tiles

Usage:
    # Create a room
    room = MicroRoom("drift-detect", target="cpu-tiny")
    
    # Ensign invokes it
    result = room.predict(sensor_window)
    # → {"class": "stable", "confidence": 0.98, "class_idx": 0}
    
    # Room logs everything
    room.summary()
    # → {"invocations": 42, "accuracy": 0.95, "avg_latency_ms": 0.39}
"""

import torch
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict

from .micro_models import train_micro, TASK_REGISTRY, _eval_model
from .hardware import deploy_micro, PROFILES, DeployedModel
from .types import (
    TrainingTile, TileType, TileLifecycle, LamportClock,
    content_hash, TrainingMetrics,
)

# Per-task minimum accuracy floors (Claude's recommendation)
# Models below these thresholds refuse to ship
TASK_ACCURACY_FLOORS = {
    "drift-detect": 0.90,
    "anomaly-flag": 0.80,
    "intent-detect": 0.80,
    "sentiment": 0.70,
    "spam-classify": 0.55,
    "topic-classify": 0.60,
    "priority-rank": 0.50,
    "tile-relevance": 0.50,
}
from .store import LocalTileStore
from .low_rank import recommend_variant


@dataclass
class Invocation:
    """Single micro model invocation."""
    timestamp: float
    input_hash: str
    prediction: int
    confidence: float
    latency_ms: float
    correct: Optional[bool] = None  # None if ground truth unknown


@dataclass
class RoomState:
    """Serialized room state."""
    room_id: str
    task: str
    target: str
    variant: str
    created_at: float
    invocations: List[Dict] = field(default_factory=list)
    tile_id: str = ""
    model_size_bytes: int = 0
    accuracy_at_deploy: float = 0.0


class MicroRoom:
    """
    A PLATO room that wraps a trained micro model.
    
    Ensigns invoke predict() — everything else is managed.
    """
    
    def __init__(
        self,
        task: str,
        target: str = "cpu",
        variant: str = "auto",
        room_id: Optional[str] = None,
        store_dir: str = ".plato-rooms",
        model: Optional[DeployedModel] = None,
    ):
        if task not in TASK_REGISTRY:
            raise ValueError(f"Unknown task: {task}. Available: {list(TASK_REGISTRY.keys())}")
        
        self.task = task
        self.target = target
        self.config = TASK_REGISTRY[task]
        self.clock = LamportClock()
        
        # Build or load model
        if model is None:
            self._deployed = deploy_micro(task, target=target, variant=variant, export=False,
                                          store_dir=store_dir)
        else:
            self._deployed = model
        
        self.model = self._deployed.model
        self.model.eval()
        self.variant = self._deployed.variant
        self.room_id = room_id or f"micro-{task}-{self.clock.tick()}"
        
        # Invocation log
        self.invocations: List[Invocation] = []
        self._device = next(self.model.parameters()).device
        self.actual_target = str(self._device)  # What we ACTUALLY got (Claude: surface target mismatch)
        
        # Store
        self.store = LocalTileStore(store_dir)
    
    def predict(
        self,
        x: torch.Tensor,
        ground_truth: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run inference on input tensor.
        
        Args:
            x: input tensor of shape (batch, input_dim) or (input_dim,)
            ground_truth: optional correct class for tracking accuracy
        
        Returns:
            dict with prediction, confidence, class_name, latency_ms
        """
        self.clock.tick()
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = x.to(self._device)
        
        start = time.perf_counter()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)
            conf = probs.max(dim=-1).values
        latency = (time.perf_counter() - start) * 1000
        
        # Log invocation
        inv = Invocation(
            timestamp=time.time(),
            input_hash=content_hash(x.cpu().numpy().tobytes()),
            prediction=pred[0].item(),
            confidence=conf[0].item(),
            latency_ms=latency,
            correct=(pred[0].item() == ground_truth) if ground_truth is not None else None,
        )
        self.invocations.append(inv)
        
        # Build result
        class_names = self._class_names()
        return {
            "prediction": pred[0].item(),
            "confidence": round(conf[0].item(), 4),
            "class_name": class_names[pred[0].item()],
            "latency_ms": round(latency, 3),
            "lamport": self.clock.now(),
            "all_probs": {class_names[i]: round(p, 4) for i, p in enumerate(probs[0].tolist())},
            "health": {
                "invocations_since_deploy": len(self.invocations),
                "meets_floor": self._deployed.metrics.get("accuracy", 0) >= TASK_ACCURACY_FLOORS.get(self.task, 0.5),
                "deploy_accuracy": round(self._deployed.metrics.get("accuracy", 0), 4),
                "floor": TASK_ACCURACY_FLOORS.get(self.task, 0.5),
            },
        }
    
    def predict_batch(self, x: torch.Tensor) -> List[Dict]:
        """Batch prediction."""
        results = []
        x = x.to(self._device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            confs = probs.max(dim=-1).values
        
        class_names = self._class_names()
        for i in range(len(x)):
            results.append({
                "prediction": preds[i].item(),
                "confidence": round(confs[i].item(), 4),
                "class_name": class_names[preds[i].item()],
            })
        return results
    
    def _class_names(self) -> List[str]:
        """Get class labels for this task."""
        labels = {
            "spam-classify": ["not-spam", "spam"],
            "intent-detect": ["query", "command", "question", "chitchat"],
            "anomaly-flag": ["normal", "anomaly"],
            "sentiment": ["negative", "neutral", "positive"],
            "topic-classify": ["topic-0", "topic-1", "topic-2", "topic-3", "topic-4"],
            "priority-rank": ["low", "medium", "high", "critical"],
            "drift-detect": ["stable", "drifting"],
            "tile-relevance": ["not-relevant", "relevant"],
        }
        return labels.get(self.task, [f"class-{i}" for i in range(self.config["num_classes"])])
    
    def summary(self) -> Dict:
        """Room summary for ensign dashboard."""
        n_inv = len(self.invocations)
        labeled = [i for i in self.invocations if i.correct is not None]
        acc = sum(1 for i in labeled if i.correct) / max(len(labeled), 1)
        avg_lat = sum(i.latency_ms for i in self.invocations) / max(n_inv, 1)
        
        return {
            "room_id": self.room_id,
            "task": self.task,
            "target": self.target,
            "variant": self.variant,
            "invocations": n_inv,
            "labeled_invocations": len(labeled),
            "accuracy": round(acc, 4),
            "avg_latency_ms": round(avg_lat, 3),
            "model_size_bytes": self._deployed.model_size_bytes,
            "deploy_accuracy": round(self._deployed.metrics.get("accuracy", 0), 4),
            "classes": self._class_names(),
        }
    
    def save(self) -> str:
        """Save room state to store."""
        state = RoomState(
            room_id=self.room_id,
            task=self.task,
            target=self.target,
            variant=self.variant,
            created_at=time.time(),
            invocations=[asdict(i) for i in self.invocations[-1000:]],  # Last 1000
            tile_id=self._deployed.tile.tile_id,
            model_size_bytes=self._deployed.model_size_bytes,
            accuracy_at_deploy=self._deployed.metrics.get("accuracy", 0),
        )
        
        path = self.store.store_dir / f"room-{self.room_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(state), f, indent=2, default=str)
        
        return str(path)
    
    @classmethod
    def load(cls, room_id: str, store_dir: str = ".plato-rooms") -> "MicroRoom":
        """Load a saved room."""
        path = Path(store_dir) / f"room-{room_id}.json"
        with open(path) as f:
            state = json.load(f)
        
        room = cls(
            task=state["task"],
            target=state["target"],
            variant=state["variant"],
            room_id=room_id,
            store_dir=store_dir,
        )
        return room
    
    def spec(self) -> str:
        """Generate human-readable room spec."""
        from .hardware import generate_room_spec
        return generate_room_spec(self.task, self.target)


class RoomFactory:
    """
    Factory for creating micro rooms.
    
    The ensign's one-stop shop:
        factory = RoomFactory()
        room = factory.create("drift-detect", target="cpu-tiny")
        result = room.predict(sensor_data)
    """
    
    def __init__(self, store_dir: str = ".plato-rooms"):
        self.store_dir = store_dir
        self.rooms: Dict[str, MicroRoom] = {}
    
    def create(self, task: str, target: str = "cpu", variant: str = "auto",
               room_id: Optional[str] = None) -> MicroRoom:
        """Create and register a new micro room."""
        room = MicroRoom(task, target=target, variant=variant,
                        room_id=room_id, store_dir=self.store_dir)
        self.rooms[room.room_id] = room
        return room
    
    def get(self, room_id: str) -> Optional[MicroRoom]:
        """Get existing room by ID."""
        return self.rooms.get(room_id)
    
    def list_rooms(self) -> List[Dict]:
        """List all rooms and their summaries."""
        return [room.summary() for room in self.rooms.values()]
    
    def save_all(self):
        """Save all room states."""
        for room in self.rooms.values():
            room.save()
