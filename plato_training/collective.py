"""
Collective Inference — rooms predict, sensors listen, gaps surface.

Architecture:
  Every instance is a peer. Everything that's not a level is a room.
  Some rooms have levels (nested rooms). Connections are automated.
  
  Simulation-first: each room predicts what SHOULD happen (t-minus-event).
  Sensors listen for glitches — mismatches between prediction and reality.
  Gaps in understanding become the focus queue. Sound out the rocks.
  
  The glitches ARE the research agenda. The gaps ARE the work.

Core loop (per room, per instance):
  1. PREDICT: "At t-minus-event, I expect X to happen"
  2. LISTEN: sensors watch for what actually happens
  3. COMPARE: prediction vs reality
  4. GAP: if mismatch → gap signal → focus queue
  5. LEARN: focus on the gap, update the room's model
  6. SHARE: broadcast updated understanding to peers via I2I

This is how a fleet of instances collectively discovers what they don't know.
"""

import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum


# ─── Room Types ─────────────────────────────────────────────────────

class RoomKind(Enum):
    """Every room has a kind. Some contain other rooms (levels)."""
    SENSOR = "sensor"         # listens to the world
    MODEL = "model"           # holds a trained model
    PREDICTOR = "predictor"   # predicts what should happen
    COMPARATOR = "comparator" # compares prediction vs reality
    GLUE = "glue"             # algorithmic logic, no tensor math
    LEVEL = "level"           # contains other rooms
    BRIDGE = "bridge"         # I2I connection to another instance
    TRAINING = "training"     # trains models
    INference = "inference"   # runs inference


@dataclass
class RoomAddress:
    """
    How to find a room in the fleet.
    
    Format: instance/room/path/to/nested/room
    Example: forgemaster@eileen/drift-detect/predictor
    """
    instance: str    # "forgemaster@eileen"
    path: List[str]  # ["drift-detect", "predictor"]
    
    @property
    def room_id(self) -> str:
        return "/".join(self.path)
    
    @property
    def full_address(self) -> str:
        return f"{self.instance}/{self.room_id}"
    
    def parent(self) -> Optional["RoomAddress"]:
        if len(self.path) <= 1:
            return None
        return RoomAddress(instance=self.instance, path=self.path[:-1])
    
    def child(self, name: str) -> "RoomAddress":
        return RoomAddress(instance=self.instance, path=self.path + [name])
    
    @classmethod
    def parse(cls, address: str) -> "RoomAddress":
        parts = address.split("/")
        return cls(instance=parts[0], path=parts[1:])
    
    def __str__(self) -> str:
        return self.full_address


# ─── T-Minus Event ──────────────────────────────────────────────────

@dataclass
class TMinusEvent:
    """
    A prediction about what should happen at a future time.
    
    "At t-minus-event, room X predicts Y with confidence Z."
    """
    predictor: str          # room address that made the prediction
    event_type: str         # "drift-exceeds-threshold", "anomaly-detected", "intent-shifts"
    predicted_value: Any    # what we expect
    confidence: float       # 0.0-1.0
    predicted_at: float     # when the prediction was made
    event_time: float       # when the event should occur
    context: Dict = field(default_factory=dict)  # why we predict this
    
    @property
    def time_until_event(self) -> float:
        return max(0, self.event_time - time.time())
    
    @property
    def is_expired(self) -> bool:
        return time.time() > self.event_time
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> "TMinusEvent":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─── Gap Signal ─────────────────────────────────────────────────────

class GapSeverity(Enum):
    LOW = "low"           # minor mismatch, within tolerance
    MEDIUM = "medium"     # significant, needs attention
    HIGH = "high"         # major gap, understanding is wrong
    CRITICAL = "critical" # fundamental model failure

@dataclass
class GapSignal:
    """
    A mismatch between prediction and reality.
    
    This IS the research agenda. This IS what to focus on.
    Gaps are how the fleet discovers what it doesn't know.
    """
    gap_id: str
    room: str              # which room detected the gap
    prediction: TMinusEvent  # what we expected
    actual: Any            # what actually happened
    severity: GapSeverity
    detected_at: float
    delta: float           # numeric distance between prediction and actual
    focus_score: float = 0.0  # higher = more important to investigate
    
    @classmethod
    def create(
        cls,
        room: str,
        prediction: TMinusEvent,
        actual: Any,
        delta: float,
    ) -> "GapSignal":
        """Create a gap signal with auto-computed severity and focus score."""
        severity = GapSeverity.LOW
        if delta > 0.5:
            severity = GapSeverity.MEDIUM
        if delta > 0.8:
            severity = GapSeverity.HIGH
        if delta > 0.95:
            severity = GapSeverity.CRITICAL
        
        # Focus score: high confidence prediction + large gap = high priority
        # "We were SURE and we were WRONG" → focus here
        focus_score = prediction.confidence * delta
        
        gap_id = hashlib.sha256(
            f"{room}:{prediction.event_type}:{prediction.predicted_at}:{actual}".encode()
        ).hexdigest()[:12]
        
        return cls(
            gap_id=gap_id,
            room=room,
            prediction=prediction,
            actual=actual,
            severity=severity,
            detected_at=time.time(),
            delta=delta,
            focus_score=focus_score,
        )
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d["prediction"] = self.prediction.to_dict()
        d["severity"] = self.severity.value
        return d


# ─── Focus Queue ────────────────────────────────────────────────────

class FocusQueue:
    """
    Priority queue of gaps — what to investigate next.
    
    Sorted by focus_score: "We were SURE and WRONG" beats "We were UNSURE and WRONG."
    Sounding out the rocks: the highest-focus gaps ARE the frontier.
    """
    
    def __init__(self):
        self.gaps: List[GapSignal] = []
    
    def add(self, gap: GapSignal):
        self.gaps.append(gap)
        self.gaps.sort(key=lambda g: g.focus_score, reverse=True)
    
    def top(self, n: int = 5) -> List[GapSignal]:
        return self.gaps[:n]
    
    def by_room(self, room: str) -> List[GapSignal]:
        return [g for g in self.gaps if room in g.room]
    
    def by_severity(self, min_severity: GapSeverity = GapSeverity.MEDIUM) -> List[GapSignal]:
        order = {GapSeverity.LOW: 0, GapSeverity.MEDIUM: 1, GapSeverity.HIGH: 2, GapSeverity.CRITICAL: 3}
        min_level = order[min_severity]
        return [g for g in self.gaps if order[g.severity] >= min_level]
    
    def clear_resolved(self, gap_ids: Set[str]):
        self.gaps = [g for g in self.gaps if g.gap_id not in gap_ids]
    
    def summary(self) -> Dict:
        return {
            "total_gaps": len(self.gaps),
            "critical": len([g for g in self.gaps if g.severity == GapSeverity.CRITICAL]),
            "high": len([g for g in self.gaps if g.severity == GapSeverity.HIGH]),
            "top_focus": [
                {"room": g.room, "delta": round(g.delta, 3), "focus": round(g.focus_score, 3)}
                for g in self.top(3)
            ],
        }


# ─── Simulation-First Room ─────────────────────────────────────────

class SimulationRoom:
    """
    A room that predicts before it observes.
    
    The core loop:
      1. predict() → TMinusEvent
      2. observe() → actual data
      3. compare() → GapSignal (or None if prediction matches)
      4. learn() → update internal model from gaps
      5. share() → broadcast to peers via I2I
    """
    
    def __init__(
        self,
        address: RoomAddress,
        kind: RoomKind = RoomKind.PREDICTOR,
        tolerance: float = 0.1,
    ):
        self.address = address
        self.kind = kind
        self.tolerance = tolerance
        self.predictions: List[TMinusEvent] = []
        self.gaps: FocusQueue = FocusQueue()
        self.observations: List[Dict] = []
        self.lamport = 0
        self.child_rooms: Dict[str, "SimulationRoom"] = {}
    
    # ─── Core Loop ───────────────────────────────────────────────
    
    def predict(self, event_type: str, predicted_value: Any,
                confidence: float, horizon_seconds: float = 60.0,
                context: Optional[Dict] = None) -> TMinusEvent:
        """
        Predict what will happen.
        
        "At t-minus-event, I expect X with confidence Y."
        Every prediction is a commitment that can be checked later.
        """
        self.lamport += 1
        
        event = TMinusEvent(
            predictor=str(self.address),
            event_type=event_type,
            predicted_value=predicted_value,
            confidence=confidence,
            predicted_at=time.time(),
            event_time=time.time() + horizon_seconds,
            context=context or {},
        )
        self.predictions.append(event)
        return event
    
    def observe(self, event_type: str, actual_value: Any,
                timestamp: Optional[float] = None) -> Optional[GapSignal]:
        """
        Observe what actually happened. Compare against predictions.
        
        Returns a GapSignal if there's a mismatch, None if prediction was correct.
        """
        ts = timestamp or time.time()
        
        # Find matching predictions
        matching = [
            p for p in self.predictions
            if p.event_type == event_type and not p.is_expired
        ]
        
        if not matching:
            # No prediction — this is an unmodeled event
            self.observations.append({
                "event_type": event_type,
                "value": actual_value,
                "timestamp": ts,
                "predicted": False,
            })
            return None
        
        # Compare against closest prediction
        closest = min(matching, key=lambda p: abs(p.event_time - ts))
        delta = self._compute_delta(closest.predicted_value, actual_value)
        
        self.observations.append({
            "event_type": event_type,
            "value": actual_value,
            "timestamp": ts,
            "predicted": True,
            "predicted_value": closest.predicted_value,
            "delta": delta,
        })
        
        if delta > self.tolerance:
            gap = GapSignal.create(
                room=str(self.address),
                prediction=closest,
                actual=actual_value,
                delta=delta,
            )
            self.gaps.add(gap)
            return gap
        
        return None  # Prediction matched within tolerance
    
    def _compute_delta(self, predicted: Any, actual: Any) -> float:
        """Compute normalized distance between prediction and actual."""
        if isinstance(predicted, (int, float)) and isinstance(actual, (int, float)):
            if predicted == 0 and actual == 0:
                return 0.0
            return abs(predicted - actual) / max(abs(predicted), abs(actual), 1e-8)
        
        if isinstance(predicted, str) and isinstance(actual, str):
            return 0.0 if predicted == actual else 1.0
        
        if isinstance(predicted, bool) and isinstance(actual, bool):
            return 0.0 if predicted == actual else 1.0
        
        if isinstance(predicted, (list, tuple)) and isinstance(actual, (list, tuple)):
            if len(predicted) != len(actual):
                return 1.0
            deltas = [self._compute_delta(p, a) for p, a in zip(predicted, actual)]
            return sum(deltas) / len(deltas)
        
        # Unknown types — use equality
        return 0.0 if predicted == actual else 1.0
    
    # ─── Nested Rooms ────────────────────────────────────────────
    
    def add_child(self, name: str, kind: RoomKind = RoomKind.PREDICTOR) -> "SimulationRoom":
        """Add a nested room (room with levels in other rooms)."""
        child_address = self.address.child(name)
        child = SimulationRoom(child_address, kind=kind, tolerance=self.tolerance)
        self.child_rooms[name] = child
        return child
    
    def get_child(self, path: List[str]) -> Optional["SimulationRoom"]:
        """Navigate to a nested room by path."""
        if not path:
            return self
        first, rest = path[0], path[1:]
        child = self.child_rooms.get(first)
        if child is None:
            return None
        return child.get_child(rest)
    
    # ─── Summary ─────────────────────────────────────────────────
    
    def summary(self) -> Dict:
        return {
            "address": str(self.address),
            "kind": self.kind.value,
            "lamport": self.lamport,
            "predictions": len(self.predictions),
            "observations": len(self.observations),
            "gaps": self.gaps.summary(),
            "children": {name: child.summary() for name, child in self.child_rooms.items()},
        }
    
    def focus_report(self) -> str:
        """Human-readable: what should we focus on?"""
        top = self.gaps.top(5)
        if not top:
            return f"[{self.address}] No gaps — understanding is sound."
        
        lines = [f"[{self.address}] Focus Queue (sound out the rocks):"]
        for i, gap in enumerate(top):
            lines.append(
                f"  {i+1}. {gap.severity.value:8s} δ={gap.delta:.2f} "
                f"focus={gap.focus_score:.2f} "
                f"predicted={gap.prediction.predicted_value} actual={gap.actual}"
            )
        return "\n".join(lines)
