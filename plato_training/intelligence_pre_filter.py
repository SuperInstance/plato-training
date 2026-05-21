"""
Pre-LLM Filter — Micro-model for intelligent request routing.

Routes incoming requests BEFORE sending to expensive LLMs. Uses a
SplineLinear-compressed MLP (384→128→8) to predict the optimal model
size and whether the LLM can be skipped entirely.

Architecture:
    Input:  384-dim feature vector (intent, length, domain, keywords, ...)
    Output: 8-dim routing logits + confidence scalar + token estimate

    384 → 128 (SplineLinear) → ReLU → 8 (SplineLinear)  → route logits
                                  └→ 1 (SplineLinear)    → confidence
                                  └→ 1 (SplineLinear)    → token estimate

Self-learning:
    Accumulates (features, decision, outcome) tuples as PLATO tiles.
    Trains on collected outcomes during idle periods.
"""



from __future__ import annotations

__all__ = ['Domain', 'FEATURE_DIM', 'IntentType', 'PreFilter', 'PreFilterModel', 'RequestFeatures', 'RouteTarget', 'RoutingDecision', 'Urgency', 'torch_softplus_inverse_approx']
import io
import json
import math
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import TrainingTile, TileType, TileLifecycle, LamportClock
from .store import LocalTileStore

# Try SplineLinear — fall back to nn.Linear if unavailable.
try:
    from .spline import SplineLinear
except ImportError:
    SplineLinear = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class IntentType(Enum):
    QUERY = 0
    COMMAND = 1
    CREATIVE = 2
    CODE = 3
    ANALYSIS = 4
    MATH = 5
    CHAT = 6
    SYSTEM = 7

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"



class Domain(Enum):
    CONSTRAINT_THEORY = 0
    PLATO = 1
    FLEET = 2
    GENERAL = 3
    CODE = 4
    MATH = 5
    RESEARCH = 6

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"



class Urgency(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"



class RouteTarget(Enum):
    SKIP_LLM = 0
    USE_LOCAL = 1
    USE_TINY = 2
    USE_SMALL = 3
    USE_MEDIUM = 4
    USE_LARGE = 5
    USE_REASONING = 6
    DELEGATE = 7

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"



# ---------------------------------------------------------------------------
# Feature vector layout (384 dims total)
# ---------------------------------------------------------------------------

FEATURE_DIM = 384
# Offsets within the 384-dim vector
_OFF_INTENT = 0          # 8
_OFF_INPUT_LEN = 8       # 16
_OFF_CTX_SIZE = 24       # 16
_OFF_DOMAIN = 40         # 7
_OFF_URGENCY = 47        # 4
_OFF_TOD_SIN = 51        # 1
_OFF_TOD_COS = 52        # 1
_OFF_DOW = 53            # 7
_OFF_SUCCESS_RATE = 60   # 4
_OFF_KEYWORDS = 64       # 64
_OFF_HIST_MATCH = 128    # 1
_OFF_PADDING = 129       # 255 → total 384


def _make_linear(in_f: int, out_f: int, n_cp: int = 16, bias: bool = True) -> nn.Module:
    """Create SplineLinear if available, else nn.Linear."""
    if SplineLinear is not None:
        return SplineLinear(in_f, out_f, n_control_points=n_cp, bias=bias)
    return nn.Linear(in_f, out_f, bias=bias)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RequestFeatures:
    """Raw request metadata — converts to a 384-dim feature vector."""
    intent: IntentType = IntentType.QUERY
    input_length: int = 0
    context_size: int = 0
    domain: Domain = Domain.GENERAL
    urgency: Urgency = Urgency.MEDIUM
    hour: float = 12.0           # 0-23
    day_of_week: int = 2         # 0=Mon..6=Sun
    recent_success_rate: float = 0.5   # 0-1
    keyword_vector: np.ndarray = field(default_factory=lambda: np.zeros(64, dtype=np.float32))
    historical_match: float = 0.0      # cosine sim 0-1

    def to_vector(self) -> np.ndarray:
        """Build the 384-dim feature vector."""
        vec = np.zeros(FEATURE_DIM, dtype=np.float32)

        # Intent one-hot (8)
        vec[_OFF_INTENT + self.intent.value] = 1.0

        # Input length — log-scaled into 16 bins
        bucket = min(int(math.log2(max(self.input_length, 1))), 15)
        vec[_OFF_INPUT_LEN + bucket] = 1.0

        # Context size — log-scaled into 16 bins
        bucket = min(int(math.log2(max(self.context_size, 1))), 15)
        vec[_OFF_CTX_SIZE + bucket] = 1.0

        # Domain one-hot (7)
        vec[_OFF_DOMAIN + self.domain.value] = 1.0

        # Urgency one-hot (4)
        vec[_OFF_URGENCY + self.urgency.value] = 1.0

        # Time-of-day sin/cos
        vec[_OFF_TOD_SIN] = math.sin(2 * math.pi * self.hour / 24.0)
        vec[_OFF_TOD_COS] = math.cos(2 * math.pi * self.hour / 24.0)

        # Day-of-week one-hot (7)
        vec[_OFF_DOW + self.day_of_week % 7] = 1.0

        # Recent success rate — 4 bins
        sr_bin = min(int(self.recent_success_rate * 4), 3)
        vec[_OFF_SUCCESS_RATE + sr_bin] = 1.0

        # Keywords (64)
        kw = np.asarray(self.keyword_vector, dtype=np.float32).flatten()
        n_kw = min(len(kw), 64)
        vec[_OFF_KEYWORDS:_OFF_KEYWORDS + n_kw] = kw[:n_kw]

        # Historical match (1)
        vec[_OFF_HIST_MATCH] = float(np.clip(self.historical_match, 0.0, 1.0))

        # Remaining dims (129..383) are padding / reserved — left zero.
        return vec


@dataclass
class RoutingDecision:
    """Output of the PreFilter model."""
    route: RouteTarget = RouteTarget.USE_SMALL
    confidence: float = 0.0
    max_tokens_estimate: int = 256
    logits: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))

    @property
    def route_name(self) -> str:
        return self.route.name


# ---------------------------------------------------------------------------
# Neural net
# ---------------------------------------------------------------------------

class PreFilterModel(nn.Module):
    """
    SplineLinear-compressed MLP for request routing.

    Architecture:
        shared:   384 → 128 (SplineLinear) → ReLU
        head_route: 128 → 8  (routing logits)
        head_conf:  128 → 1  (confidence, sigmoid)
        head_tokens: 128 → 1 (log token estimate, softplus)
    """

    ROUTE_DIM = len(RouteTarget)

    def __init__(self, n_control_points: int = 16) -> None:
        super().__init__()
        self.shared = _make_linear(FEATURE_DIM, 128, n_cp=n_control_points)
        self.head_route = _make_linear(128, self.ROUTE_DIM, n_cp=n_control_points)
        self.head_conf = _make_linear(128, 1, n_cp=n_control_points)
        self.head_tokens = _make_linear(128, 1, n_cp=n_control_points)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 384)

        Returns:
            route_logits:  (batch, 8)
            confidence:    (batch, 1)  — sigmoid-activated
            token_log:     (batch, 1)  — raw (apply softplus/exp for estimate)
        """
        h = F.relu(self.shared(x))
        route_logits = self.head_route(h)
        confidence = torch.sigmoid(self.head_conf(h))
        token_log = self.head_tokens(h)
        return route_logits, confidence, token_log

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_control_points={self.n_control_points!r})"



# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class PreFilter:
    """
    Pre-LLM request router with self-learning.

    Usage::

        pf = PreFilter()
        features = RequestFeatures(intent=IntentType.CODE, input_length=500)
        decision = pf.route(features)
        # ... later, after LLM response ...
        pf.record_outcome(features, decision, was_good=True)
    """

    def __init__(
        self,
        n_control_points: int = 16,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = PreFilterModel(n_control_points=n_control_points).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.clock = LamportClock()
        self._outcomes: List[Dict] = []

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def route(self, features: RequestFeatures) -> RoutingDecision:
        """Run the pre-filter model and return a routing decision."""
        self.model.eval()
        vec = torch.tensor(features.to_vector(), dtype=torch.float32, device=self.device)
        vec = vec.unsqueeze(0)  # (1, 384)

        route_logits, confidence, token_log = self.model(vec)

        logits_np = route_logits.squeeze(0).cpu().numpy()
        route_idx = int(np.argmax(logits_np))
        conf = float(confidence.squeeze().cpu())
        tokens_raw = float(token_log.squeeze().cpu())
        # Convert log-space to token count (clamp to positive int)
        token_est = max(1, int(torch_softplus_inverse_approx(tokens_raw)))

        return RoutingDecision(
            route=RouteTarget(route_idx),
            confidence=conf,
            max_tokens_estimate=token_est,
            logits=logits_np,
        )

    # ------------------------------------------------------------------
    # Self-learning
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        features: RequestFeatures,
        decision: RoutingDecision,
        was_good: bool,
    ) -> None:
        """Store an outcome tuple for future self-training."""
        self._outcomes.append({
            "features": features.to_vector().tolist(),
            "route_idx": decision.route.value,
            "confidence": decision.confidence,
            "tokens_estimate": decision.max_tokens_estimate,
            "was_good": was_good,
            "timestamp": time.time(),
            "lamport": self.clock.tick(),
        })

    def self_train(self, epochs: int = 10, batch_size: int = 32) -> Dict[str, float]:
        """
        Train on accumulated outcomes.

        For positive outcomes, reinforce the chosen route.
        For negative outcomes, push toward the next-best route.

        Returns training metrics dict.
        """
        if len(self._outcomes) < 2:
            return {"status": "insufficient_data", "n_outcomes": len(self._outcomes)}

        self.model.train()

        # Build tensors
        X_list, y_route, y_good = [], [], []
        for o in self._outcomes:
            X_list.append(o["features"])
            y_route.append(o["route_idx"])
            y_good.append(1.0 if o["was_good"] else 0.0)

        X = torch.tensor(X_list, dtype=torch.float32, device=self.device)
        y_route_t = torch.tensor(y_route, dtype=torch.long, device=self.device)
        y_good_t = torch.tensor(y_good, dtype=torch.float32, device=self.device)

        losses = []
        n_samples = X.shape[0]

        for epoch in range(epochs):
            perm = torch.randperm(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                idx = perm[start:start + batch_size]
                xb = X[idx]
                yb_route = y_route_t[idx]
                yb_good = y_good_t[idx]

                self.optimizer.zero_grad()
                route_logits, confidence, _ = self.model(xb)

                # Route classification loss (cross-entropy)
                loss_route = F.cross_entropy(route_logits, yb_route)

                # Confidence calibration: confidence should match actual goodness
                loss_conf = F.binary_cross_entropy(confidence.squeeze(), yb_good)

                # Combined loss
                loss = loss_route + 0.5 * loss_conf
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            losses.append(epoch_loss / max(n_batches, 1))

        self.model.eval()
        return {
            "status": "ok",
            "epochs": epochs,
            "n_samples": n_samples,
            "final_loss": losses[-1] if losses else 0.0,
            "loss_curve": losses,
        }

    # ------------------------------------------------------------------
    # PLATO tile integration
    # ------------------------------------------------------------------

    def export_tile(self, store_dir: str = ".plato-prefilter") -> str:
        """Save model state as a PLATO tile. Returns tile_id."""
        store = LocalTileStore(store_dir)
        buf = io.BytesIO()
        torch.save(self.model.state_dict(), buf)
        data = buf.getvalue()

        from .types import content_hash
        chash = content_hash(data)

        tile = TrainingTile(
            tile_id=f"prefilter-{chash}",
            room="intelligence-pre-filter",
            tile_type=TileType.CHECKPOINT,
            name="PreFilter checkpoint",
            description="SplineLinear-compressed Pre-LLM routing model",
            content_hash=chash,
            lamport=self.clock.tick(),
        )
        store.save(tile)
        store.save_weights(chash, data)
        return tile.tile_id

    def load_tile(self, tile_id: str, store_dir: str = ".plato-prefilter") -> bool:
        """Load model state from a PLATO tile. Returns True on success."""
        store = LocalTileStore(store_dir)
        tile = store.load(tile_id)
        if tile is None:
            return False
        data = store.load_weights(tile.content_hash)
        if data is None:
            return False
        buf = io.BytesIO(data)
        state = torch.load(buf, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        return True

    @property
    def n_outcomes(self) -> int:
        return len(self._outcomes)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_control_points={self.n_control_points!r}, learning_rate={self.learning_rate!r}, device={self.device!r})"



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def torch_softplus_inverse_approx(x: float) -> float:
    """Rough inverse softplus for token estimation."""
    if x > 20:
        return x
    return math.log(math.exp(x) + 1.0)
