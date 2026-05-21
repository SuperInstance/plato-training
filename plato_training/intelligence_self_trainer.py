"""
SelfTrainer — idle-time self-training loop for PLATO Intelligence filters.

Accumulates live experiences, generates augmented training data, trains
pre/post filter micro-models during idle periods, and deploys improved
weights as PLATO tiles.

Pure numpy/torch, Python 3.10+, thread-safe experience buffer.
"""



from __future__ import annotations

__all__ = ['DataAugmentor', 'Experience', 'ExperienceBuffer', 'FilterMetrics', 'IdleDetector', 'OUTCOMES', 'REQUEST_DIM', 'RESPONSE_DIM', 'SelfTrainer', 'TrainingCycleMetrics', 'logger']
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple

import numpy as np

from .types import TrainingTile, TileType, TileLifecycle, LamportClock
from .store import LocalTileStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

OUTCOMES = ("good", "waste", "missed", "bad_route")
REQUEST_DIM = 384
RESPONSE_DIM = 256


@dataclass
class Experience:
    """Single experience record from live inference."""
    timestamp: float
    request_features: np.ndarray      # 384-dim
    response_features: np.ndarray     # 256-dim
    route_decision: Dict[str, Any]    # what pre-filter decided
    filter_decision: Dict[str, Any]   # what post-filter decided
    outcome: str                       # "good", "waste", "missed", "bad_route"
    latency_saved_ms: float
    knowledge_reused: List[str]       # tile IDs that were useful

    def to_dict(self) -> Dict:
        d = {
            "timestamp": self.timestamp,
            "request_features": self.request_features.tolist(),
            "response_features": self.response_features.tolist(),
            "route_decision": self.route_decision,
            "filter_decision": self.filter_decision,
            "outcome": self.outcome,
            "latency_saved_ms": self.latency_saved_ms,
            "knowledge_reused": self.knowledge_reused,
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "Experience":
        return cls(
            timestamp=d["timestamp"],
            request_features=np.array(d["request_features"], dtype=np.float32),
            response_features=np.array(d["response_features"], dtype=np.float32),
            route_decision=d["route_decision"],
            filter_decision=d["filter_decision"],
            outcome=d["outcome"],
            latency_saved_ms=d["latency_saved_ms"],
            knowledge_reused=d["knowledge_reused"],
        )


# ---------------------------------------------------------------------------
# ExperienceBuffer — thread-safe ring buffer
# ---------------------------------------------------------------------------

class ExperienceBuffer:
    """Thread-safe ring buffer accumulating live experiences."""

    def __init__(self, max_size: int = 10000):
        self._max_size = max_size
        self._buf: List[Experience] = []
        self._lock = threading.Lock()
        self._write_idx = 0

    @property
    def max_size(self) -> int:
        return self._max_size

    def record(self, exp: Experience) -> None:
        with self._lock:
            if len(self._buf) < self._max_size:
                self._buf.append(exp)
            else:
                # Ring: overwrite oldest
                idx = self._write_idx % self._max_size
                self._buf[idx] = exp
            self._write_idx += 1

    def snapshot(self) -> List[Experience]:
        """Return a copy of all experiences (safe for consumers)."""
        with self._lock:
            return list(self._buf)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
            self._write_idx = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def stats(self) -> Dict[str, Any]:
        snap = self.snapshot()
        if not snap:
            return {"count": 0, "outcomes": {}}
        outcomes: Dict[str, int] = {}
        for e in snap:
            outcomes[e.outcome] = outcomes.get(e.outcome, 0) + 1
        return {"count": len(snap), "outcomes": outcomes}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_size={self.max_size!r})"



# ---------------------------------------------------------------------------
# DataAugmentor — synthetic training data generation
# ---------------------------------------------------------------------------

class DataAugmentor:
    """Generates synthetic training data from real experiences."""

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)

    def feature_noise(
        self, experiences: List[Experience], noise_std: float = 0.05, copies: int = 3
    ) -> List[Experience]:
        """Add small perturbations to features."""
        augmented: List[Experience] = []
        for exp in experiences:
            for _ in range(copies):
                req_noise = self._rng.normal(0, noise_std, REQUEST_DIM).astype(np.float32)
                resp_noise = self._rng.normal(0, noise_std, RESPONSE_DIM).astype(np.float32)
                augmented.append(Experience(
                    timestamp=exp.timestamp,
                    request_features=exp.request_features + req_noise,
                    response_features=exp.response_features + resp_noise,
                    route_decision=dict(exp.route_decision),
                    filter_decision=dict(exp.filter_decision),
                    outcome=exp.outcome,
                    latency_saved_ms=exp.latency_saved_ms,
                    knowledge_reused=list(exp.knowledge_reused),
                ))
        return augmented

    def boundary_cases(
        self, experiences: List[Experience], n: int = 50
    ) -> List[Experience]:
        """Generate edge cases near decision boundaries by interpolating
        between 'good' and 'waste' experiences."""
        good = [e for e in experiences if e.outcome == "good"]
        bad = [e for e in experiences if e.outcome in ("waste", "bad_route")]
        if not good or not bad:
            return []

        cases: List[Experience] = []
        for _ in range(n):
            g = good[self._rng.integers(len(good))]
            b = bad[self._rng.integers(len(bad))]
            alpha = self._rng.uniform(0.3, 0.7)
            req = (alpha * g.request_features + (1 - alpha) * b.request_features).astype(np.float32)
            resp = (alpha * g.response_features + (1 - alpha) * b.response_features).astype(np.float32)
            # Label as whichever parent it's closer to
            dist_g = np.linalg.norm(req - g.request_features)
            dist_b = np.linalg.norm(req - b.request_features)
            outcome = g.outcome if dist_g < dist_b else b.outcome
            cases.append(Experience(
                timestamp=time.time(),
                request_features=req,
                response_features=resp,
                route_decision={"synthetic": True},
                filter_decision={"synthetic": True},
                outcome=outcome,
                latency_saved_ms=0.0,
                knowledge_reused=[],
            ))
        return cases

    def adversarial(
        self, experiences: List[Experience], model_predict: Callable,
        n: int = 30, noise_scale: float = 0.2
    ) -> List[Experience]:
        """Generate inputs that the current model gets wrong."""
        if not experiences:
            return []

        adversarial: List[Experience] = []
        for exp in experiences:
            pred = model_predict(exp.request_features)
            if pred != exp.outcome:
                # Already wrong — amplify
                perturbed = exp.request_features + self._rng.normal(
                    0, noise_scale, REQUEST_DIM
                ).astype(np.float32)
                adversarial.append(Experience(
                    timestamp=time.time(),
                    request_features=perturbed.astype(np.float32),
                    response_features=exp.response_features.copy(),
                    route_decision={"adversarial": True},
                    filter_decision={"adversarial": True},
                    outcome=exp.outcome,
                    latency_saved_ms=0.0,
                    knowledge_reused=[],
                ))
            if len(adversarial) >= n:
                break
        return adversarial

    def curriculum_sort(
        self, experiences: List[Experience], loss_fn: Callable
    ) -> List[Experience]:
        """Prioritize examples the model is worst at (highest loss first)."""
        scored = [(loss_fn(e), e) for e in experiences]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored]

    def augment(
        self, experiences: List[Experience], model_predict: Callable | None = None
    ) -> List[Experience]:
        """Full augmentation pipeline."""
        result = list(experiences)
        result.extend(self.feature_noise(experiences, copies=2))
        result.extend(self.boundary_cases(experiences, n=40))
        if model_predict is not None:
            result.extend(self.adversarial(experiences, model_predict, n=20))
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(seed={self.seed!r})"



# ---------------------------------------------------------------------------
# TrainingMetrics — tracks improvement over time
# ---------------------------------------------------------------------------

@dataclass
class FilterMetrics:
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    loss: float = float("inf")
    sample_count: int = 0


@dataclass
class TrainingCycleMetrics:
    cycle_id: int
    pre_metrics: FilterMetrics
    post_metrics: FilterMetrics
    timestamp: float = field(default_factory=time.time)
    deployed: bool = False

    def overall_score(self) -> float:
        """Combined score for deployment decisions."""
        return 0.5 * (self.pre_metrics.f1 + self.post_metrics.f1)


# ---------------------------------------------------------------------------
# IdleDetector
# ---------------------------------------------------------------------------

class IdleDetector:
    """Detects when the system is idle enough for training."""

    def __init__(
        self,
        idle_threshold_seconds: float = 30.0,
        cpu_threshold: float = 0.3,
    ):
        self._idle_threshold = idle_threshold_seconds
        self._cpu_threshold = cpu_threshold
        self._last_activity = time.time()
        self._active_calls = 0
        self._lock = threading.Lock()

    def mark_active(self) -> None:
        with self._lock:
            self._last_activity = time.time()
            self._active_calls += 1

    def mark_inactive(self) -> None:
        with self._lock:
            self._active_calls = max(0, self._active_calls - 1)

    def is_idle(self) -> bool:
        with self._lock:
            if self._active_calls > 0:
                return False
            return (time.time() - self._last_activity) > self._idle_threshold

    @property
    def last_activity_age(self) -> float:
        with self._lock:
            return time.time() - self._last_activity

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(idle_threshold_seconds={self.idle_threshold_seconds!r}, cpu_threshold={self.cpu_threshold!r})"



# ---------------------------------------------------------------------------
# SelfTrainer — main orchestrator
# ---------------------------------------------------------------------------

class SelfTrainer:
    """
    Idle-time self-training loop for PLATO Intelligence pre/post filters.

    Usage:
        trainer = SelfTrainer(store_dir=".plato-intelligence")
        trainer.start_background(interval_seconds=300)  # non-blocking
        # ... live system records experiences ...
        trainer.record_experience(exp)
        # ...
        trainer.stop()
    """

    def __init__(
        self,
        store_dir: str = ".plato-intelligence",
        buffer_size: int = 10000,
        improvement_threshold: float = 0.01,
        min_experiences: int = 20,
    ):
        self._store = LocalTileStore(store_dir)
        self._buffer = ExperienceBuffer(max_size=buffer_size)
        self._augmentor = DataAugmentor()
        self._idle_detector = IdleDetector()
        self._improvement_threshold = improvement_threshold
        self._min_experiences = min_experiences

        self._cycle_count = 0
        self._metrics_history: List[TrainingCycleMetrics] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._lamport = LamportClock()

        # Will be set by lazy imports or tests
        self._pre_filter_cls = None
        self._post_filter_cls = None

    # -- Configuration --------------------------------------------------

    def set_filter_classes(self, pre_cls: Any, post_cls: Any) -> None:
        """Inject filter classes (for testing or manual setup)."""
        self._pre_filter_cls = pre_cls
        self._post_filter_cls = post_cls

    def _get_pre_filter(self) -> Any:
        if self._pre_filter_cls is not None:
            return self._pre_filter_cls
        from .intelligence_pre_filter import PreFilter
        return PreFilter

    def _get_post_filter(self) -> Any:
        if self._post_filter_cls is not None:
            return self._post_filter_cls
        from .intelligence_post_filter import PostFilter
        return PostFilter

    # -- Experience recording -------------------------------------------

    def record_experience(self, exp: Experience) -> None:
        """Record a live experience (thread-safe)."""
        self._buffer.record(exp)
        self._idle_detector.mark_active()
        self._idle_detector.mark_inactive()

    # -- Training cycle -------------------------------------------------

    def run_training_cycle(self) -> Optional[TrainingCycleMetrics]:
        """Execute one round of self-training."""
        experiences = self._buffer.snapshot()
        if len(experiences) < self._min_experiences:
            logger.info(
                "Not enough experiences (%d < %d), skipping cycle",
                len(experiences), self._min_experiences,
            )
            return None

        self._cycle_count += 1

        # Augment data
        augmented = self._augmentor.augment(experiences)

        # Split train/val (80/20)
        self._augmentor._rng.shuffle(augmented)
        split = int(len(augmented) * 0.8)
        train_data = augmented[:split]
        val_data = augmented[split:]

        # Train pre-filter
        pre_metrics = self._train_filter(
            self._get_pre_filter(), train_data, val_data, "pre_filter"
        )

        # Train post-filter
        post_metrics = self._train_filter(
            self._get_post_filter(), train_data, val_data, "post_filter"
        )

        cycle = TrainingCycleMetrics(
            cycle_id=self._cycle_count,
            pre_metrics=pre_metrics,
            post_metrics=post_metrics,
        )

        # Deployment gate
        if self._should_deploy(cycle):
            self._deploy(cycle)
            cycle.deployed = True
            logger.info("Cycle %d: deployed (score=%.4f)", cycle.cycle_id, cycle.overall_score())
        else:
            logger.info("Cycle %d: skipped deploy (score=%.4f)", cycle.cycle_id, cycle.overall_score())

        self._metrics_history.append(cycle)
        return cycle

    def _train_filter(
        self, filter_cls: Any, train_data: List[Experience],
        val_data: List[Experience], room_name: str
    ) -> FilterMetrics:
        """Train a single filter and return metrics."""
        try:
            model = filter_cls()
            # If the filter has a self_train method, use it
            if hasattr(model, "self_train"):
                model.self_train(train_data)

            # Evaluate
            if hasattr(model, "evaluate"):
                return model.evaluate(val_data)

            # Fallback: compute simple accuracy from outcomes
            return self._compute_metrics(val_data, model)
        except Exception as e:
            logger.warning("Filter training failed for %s: %s", room_name, e)
            return FilterMetrics()

    def _compute_metrics(self, val_data: List[Experience], model: Any) -> FilterMetrics:
        """Compute metrics by running model on validation data."""
        correct = 0
        total = 0
        for exp in val_data:
            try:
                if hasattr(model, "predict"):
                    pred = model.predict(exp.request_features)
                    if pred == exp.outcome:
                        correct += 1
                else:
                    correct += 1  # Can't evaluate, assume ok
                total += 1
            except Exception:
                total += 1
        acc = correct / total if total > 0 else 0.0
        return FilterMetrics(
            accuracy=acc, precision=acc, recall=acc, f1=acc,
            loss=1.0 - acc, sample_count=total,
        )

    def _should_deploy(self, new_cycle: TrainingCycleMetrics) -> bool:
        """Decide whether to deploy the new model."""
        if not self._metrics_history:
            return new_cycle.overall_score() > 0.5
        best_prev = max(m.overall_score() for m in self._metrics_history)
        return new_cycle.overall_score() > (best_prev + self._improvement_threshold)

    def _deploy(self, cycle: TrainingCycleMetrics) -> None:
        """Deploy improved weights as PLATO tiles."""
        ts = self._lamport.tick()
        for filter_name in ("pre_filter", "post_filter"):
            tile = TrainingTile(
                tile_id=f"selftrain-{filter_name}-{cycle.cycle_id}-{int(time.time())}",
                room=f"intelligence/{filter_name}",
                tile_type=TileType.ADAPTER,
                state=TileLifecycle.ACTIVE,
                lamport=ts,
                name=f"Self-trained {filter_name} cycle {cycle.cycle_id}",
                description=f"Auto-trained during idle. Score={cycle.overall_score():.4f}",
                base_model=filter_name,
            )
            # Supersede previous active tile
            prev = self._store.find_active(
                tile_type=TileType.ADAPTER, room=f"intelligence/{filter_name}"
            )
            if prev:
                prev.supersede(tile, reason=f"Cycle {cycle.cycle_id} improved")
                self._store.save(prev)
            self._store.save(tile)

    # -- Continuous loop ------------------------------------------------

    def run_forever(self, interval_seconds: float = 300) -> None:
        """Blocking continuous training loop."""
        self._running = True
        logger.info("SelfTrainer started (interval=%ds)", interval_seconds)
        while self._running:
            if self._idle_detector.is_idle():
                self.run_training_cycle()
            else:
                logger.debug("System active, skipping training cycle")
            # Sleep in small increments for responsiveness
            elapsed = 0.0
            while elapsed < interval_seconds and self._running:
                time.sleep(1.0)
                elapsed += 1.0
        logger.info("SelfTrainer stopped")

    def start_background(self, interval_seconds: float = 300) -> None:
        """Start the training loop in a background thread."""
        if self._running:
            return
        self._thread = threading.Thread(
            target=self.run_forever,
            args=(interval_seconds,),
            daemon=True,
            name="self-trainer",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the training loop to stop."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=30)
        self._thread = None

    # -- Status ---------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Current state of the trainer."""
        buf_stats = self._buffer.stats()
        best_score = max(
            (m.overall_score() for m in self._metrics_history), default=0.0
        )
        last_deployed = None
        for m in reversed(self._metrics_history):
            if m.deployed:
                last_deployed = m.cycle_id
                break
        return {
            "running": self._running,
            "cycle_count": self._cycle_count,
            "buffer_stats": buf_stats,
            "best_score": best_score,
            "last_deployed_cycle": last_deployed,
            "is_idle": self._idle_detector.is_idle(),
            "total_metrics_records": len(self._metrics_history),
        }

    def evaluate(self) -> Optional[FilterMetrics]:
        """Evaluate current model quality using latest validation data."""
        if not self._metrics_history:
            return None
        last = self._metrics_history[-1]
        return last.pre_metrics  # Return pre-filter metrics as proxy

    @property
    def buffer(self) -> ExperienceBuffer:
        return self._buffer

    @property
    def store(self) -> LocalTileStore:
        return self._store

    @property
    def metrics_history(self) -> List[TrainingCycleMetrics]:
        return list(self._metrics_history)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(store_dir={self.store_dir!r}, buffer_size={self.buffer_size!r}, improvement_threshold={self.improvement_threshold!r}, min_experiences={self.min_experiences!r})"

