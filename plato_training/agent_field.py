"""
AgentField: Shared tensor field for within-agent room coordination.

NOT a collection of RoomMusicians sending messages.
One tensor. Rooms are views. Coupling is a matrix.
The Eisenstein lattice IS the protocol.

The agent IS the score. Rooms are standing waves in the resonance chamber.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from flux_tensor_midi import FluxVector, TZeroClock, EisensteinSnap


# 9-channel semantics within an agent
CHANNEL_NAMES = [
    "confidence",  # 0: how sure this room is
    "entropy",     # 1: how distributed the prediction is
    "drift",       # 2: rate of state change
    "focus",       # 3: attention weight
    "gap",         # 4: prediction vs reality delta
    "salience",    # 5: importance of this room's output
    "coupling",    # 6: how much this room affects others
    "resonance",   # 7: feedback strength from other rooms
    "phase",       # 8: where in predict/observe cycle
]


@dataclass
class RoomMeta:
    """Metadata for a room within the agent field."""
    name: str
    bpm: float = 120.0
    role: str = "sensor"     # sensor, predictor, comparator, bridge
    chamber: int = 0         # Eisenstein chamber (dodecet index)
    chirality: str = "exploring"  # exploring, locking, locked
    ticks: int = 0


class AgentField:
    """An agent's internal state as a shared tensor field.
    
    One numpy-free implementation: pure Python with FluxVector compatibility.
    Rooms are indices into the shared state arrays.
    Coupling is a matrix. Gaps are self-organizing.
    
    Usage:
        field = AgentField()
        sensor = field.add_room("drift-sensor", role="sensor")
        predictor = field.add_room("drift-predict", role="predictor")
        comparator = field.add_room("drift-compare", role="comparator")
        
        # Wire coupling
        field.couple(predictor, sensor, strength=0.9)
        field.couple(comparator, sensor, strength=0.5)
        field.couple(comparator, predictor, strength=0.5)
        
        # Run
        field.sensor_write(sensor, [0.8, 0.2, 0.01, 0.9, 0.0, 1.0, 0.0, 0.0, 0.0])
        field.tick()
        gaps = field.focus_queue()
    """
    
    def __init__(self, bpm: float = 120.0, damping: float = 0.1):
        self._n = 0
        self._state: List[List[float]] = []          # [room][channel]
        self._salience: List[List[float]] = []        # [room][channel]
        self._tolerance: List[List[float]] = []       # [room][channel]
        self._coupling: List[List[float]] = []        # [room_from][room_to]
        self._meta: Dict[int, RoomMeta] = {}
        self._name_to_idx: Dict[str, int] = {}
        self._clock = TZeroClock(bpm=bpm)
        self._snap = EisensteinSnap(base_period_ms=60000.0 / bpm)
        self._damping = damping
        self._tick_count = 0
    
    @property
    def n_rooms(self) -> int:
        return self._n
    
    @property
    def tick_count(self) -> int:
        return self._tick_count
    
    def add_room(
        self,
        name: str,
        role: str = "sensor",
        bpm: Optional[float] = None,
        initial_state: Optional[List[float]] = None,
    ) -> int:
        """Add a room to the field. Returns its index."""
        idx = self._n
        self._n += 1
        
        state = initial_state or [0.0] * 9
        assert len(state) == 9, f"State must be 9 channels, got {len(state)}"
        
        self._state.append(list(state))
        self._salience.append([1.0] * 9)
        self._tolerance.append([0.01] * 9)
        
        # Expand coupling matrix
        for row in self._coupling:
            row.append(0.0)
        self._coupling.append([0.0] * self._n)
        
        self._meta[idx] = RoomMeta(
            name=name,
            bpm=bpm or self._clock.bpm,
            role=role,
        )
        self._name_to_idx[name] = idx
        
        return idx
    
    def idx(self, name_or_idx) -> int:
        """Resolve name or index to index."""
        if isinstance(name_or_idx, int):
            return name_or_idx
        return self._name_to_idx[name_or_idx]
    
    # ─── State Access ─────────────────────────────────────────
    
    def get_state(self, room: int | str) -> FluxVector:
        """Read room state as FluxVector (copy out)."""
        i = self.idx(room)
        return FluxVector(
            self._state[i],
            salience=self._salience[i],
            tolerance=self._tolerance[i],
        )
    
    def set_state(self, room: int | str, fv: FluxVector):
        """Write FluxVector directly to room's tensor row."""
        i = self.idx(room)
        self._state[i] = list(fv.values)
        self._salience[i] = list(fv.salience)
        self._tolerance[i] = list(fv.tolerance)
    
    def set_channel(self, room: int | str, channel: int, value: float):
        """Write a single channel. The atomic within-agent write."""
        i = self.idx(room)
        self._state[i][channel] = value
    
    def get_channel(self, room: int | str, channel: int) -> float:
        """Read a single channel. Zero-copy within the field."""
        i = self.idx(room)
        return self._state[i][channel]
    
    def sensor_write(self, room: int | str, values: List[float]):
        """Sensor room writes raw values. Sets phase=0 (perceiving)."""
        i = self.idx(room)
        self._state[i] = list(values)
        self._state[i][8] = 0.0  # phase = perceiving
    
    def predict_write(self, room: int | str, confidence: float, values: List[float]):
        """Predictor room writes prediction. Sets phase=0.25 (predicted)."""
        i = self.idx(room)
        self._state[i] = list(values)
        self._state[i][0] = confidence
        self._state[i][8] = 0.25  # phase = predicted
    
    # ─── Coupling ─────────────────────────────────────────────
    
    def couple(self, from_room: int | str, to_room: int | str, strength: float = 0.5):
        """Set coupling strength: from_room is influenced by to_room."""
        i = self.idx(from_room)
        j = self.idx(to_room)
        self._coupling[i][j] = strength
    
    def decouple(self, from_room: int | str, to_room: int | str):
        """Remove coupling."""
        i = self.idx(from_room)
        j = self.idx(to_room)
        self._coupling[i][j] = 0.0
    
    def get_coupling(self, from_room: int | str, to_room: int | str) -> float:
        i = self.idx(from_room)
        j = self.idx(to_room)
        return self._coupling[i][j]
    
    # ─── Side-Channels as Coupling Modulation ─────────────────
    
    def nod(self, from_room: int | str, to_room: int | str, intensity: float = 0.1):
        """Increase coupling (trust more). The nod IS the coupling change."""
        i = self.idx(from_room)
        j = self.idx(to_room)
        self._coupling[i][j] = min(1.0, self._coupling[i][j] + intensity)
    
    def smile(self, from_room: int | str, to_room: int | str, intensity: float = 0.1):
        """Increase coupling AND shift toward other room's state."""
        i = self.idx(from_room)
        j = self.idx(to_room)
        self._coupling[i][j] = min(1.0, self._coupling[i][j] + intensity)
        # Shift state toward the other room (alignment)
        for ch in range(9):
            diff = self._state[j][ch] - self._state[i][ch]
            self._state[i][ch] += diff * intensity * 0.5
    
    def frown(self, from_room: int | str, to_room: int | str, intensity: float = 0.1):
        """Decrease coupling AND raise gap channel."""
        i = self.idx(from_room)
        j = self.idx(to_room)
        self._coupling[i][j] = max(0.0, self._coupling[i][j] - intensity)
        self._state[i][4] += intensity  # gap channel up
        # Update chirality
        if self._state[i][4] > 0.5:
            self._meta[i].chirality = "exploring"
    
    # ─── The Fundamental Update ───────────────────────────────
    
    def tick(self) -> float:
        """Advance all rooms by one tick.
        
        1. Compute coupling forces (weighted sum of neighbor states)
        2. Mix into current state (damped)
        3. Update phase channels
        4. Advance clock
        
        Returns the clock timestamp in ms.
        """
        # 1. Compute coupling forces
        new_state = [list(row) for row in self._state]  # copy
        
        for i in range(self._n):
            for j in range(self._n):
                if i == j or self._coupling[i][j] == 0:
                    continue
                
                c = self._coupling[i][j]
                for ch in range(9):
                    # Coupling force: weighted difference
                    diff = self._state[j][ch] - self._state[i][ch]
                    new_state[i][ch] += c * diff * self._damping
        
        # 2. Apply (with salience-weighted damping)
        for i in range(self._n):
            for ch in range(9):
                # Salience gates how much coupling affects each channel
                s = self._salience[i][ch]
                self._state[i][ch] = new_state[i][ch] * s
        
        # 3. Update phase channels
        for i in range(self._n):
            self._state[i][8] = (self._state[i][8] + 0.25) % 1.0  # 4-phase cycle
            self._meta[i].ticks += 1
        
        # 4. Advance clock
        self._tick_count += 1
        return self._clock.tick()
    
    # ─── Coherence & Gap Analysis ─────────────────────────────
    
    def coherence(self) -> float:
        """Overall agent coherence: mean pairwise cosine similarity."""
        if self._n < 2:
            return 1.0
        total = 0.0
        count = 0
        for i in range(self._n):
            for j in range(i + 1, self._n):
                vi = self._state[i]
                vj = self._state[j]
                mag_i = math.sqrt(sum(x * x for x in vi))
                mag_j = math.sqrt(sum(x * x for x in vj))
                if mag_i > 0 and mag_j > 0:
                    dot = sum(vi[k] * vj[k] for k in range(9))
                    total += dot / (mag_i * mag_j)
                    count += 1
        return total / max(count, 1)
    
    def room_coherence(self, room_a: int | str, room_b: int | str) -> float:
        """Pairwise coherence between two rooms."""
        i = self.idx(room_a)
        j = self.idx(room_b)
        vi = self._state[i]
        vj = self._state[j]
        mag_i = math.sqrt(sum(x * x for x in vi))
        mag_j = math.sqrt(sum(x * x for x in vj))
        if mag_i == 0 or mag_j == 0:
            return 0.0
        return sum(vi[k] * vj[k] for k in range(9)) / (mag_i * mag_j)
    
    def gaps(self) -> List[Tuple[int, float]]:
        """Rooms whose gap channel exceeds tolerance."""
        result = []
        for i in range(self._n):
            gap = self._state[i][4]
            tol = self._tolerance[i][4]
            if gap > tol:
                result.append((i, gap))
        return sorted(result, key=lambda x: -x[1])
    
    def focus_queue(self) -> List[Tuple[str, float]]:
        """Rooms ranked by gap × confidence = what to work on next.
        
        This IS the fleet's focus queue, but computed from the shared tensor
        instead of from message passing.
        """
        scores = []
        for i in range(self._n):
            focus_score = self._state[i][4] * self._state[i][0]  # gap × confidence
            if focus_score > 0:
                name = self._meta[i].name
                scores.append((name, focus_score))
        return sorted(scores, key=lambda x: -x[1])
    
    def within_tolerance(self, room_a: int | str, room_b: int | str) -> bool:
        """Check if two rooms are 'in tune' — all channels within tolerance."""
        i = self.idx(room_a)
        j = self.idx(room_b)
        for ch in range(9):
            diff = abs(self._state[i][ch] - self._state[j][ch])
            tol = max(self._tolerance[i][ch], self._tolerance[j][ch])
            if diff > tol:
                return False
        return True
    
    # ─── Chirality State Machine ──────────────────────────────
    
    def chirality(self, room: int | str) -> str:
        """Get room's current chirality state."""
        i = self.idx(room)
        return self._meta[i].chirality
    
    def update_chirality(self, room: int | str):
        """Update room's chirality based on gap history.
        
        exploring  → locking: gap < tolerance for 3+ ticks
        locking    → locked:  gap < tolerance for 10+ ticks  
        locked     → exploring: gap > tolerance (anomaly)
        """
        i = self.idx(room)
        gap = self._state[i][4]
        tol = self._tolerance[i][4]
        meta = self._meta[i]
        
        if meta.chirality == "exploring" and gap < tol and meta.ticks >= 3:
            meta.chirality = "locking"
        elif meta.chirality == "locking" and gap < tol and meta.ticks >= 10:
            meta.chirality = "locked"
        elif meta.chirality in ("locking", "locked") and gap > tol:
            meta.chirality = "exploring"
    
    # ─── Reports ──────────────────────────────────────────────
    
    def room_report(self, room: int | str) -> str:
        """Human-readable state for one room."""
        i = self.idx(room)
        meta = self._meta[i]
        s = self._state[i]
        lines = [
            f"Room: {meta.name} (idx={i}, role={meta.role})",
            f"  Chirality: {meta.chirality}  Ticks: {meta.ticks}",
            f"  Channels:",
        ]
        for ch in range(9):
            lines.append(f"    {CHANNEL_NAMES[ch]:12s} = {s[ch]:+.4f}  (salience={self._salience[i][ch]:.2f}, tol={self._tolerance[i][ch]:.3f})")
        
        # Coupling
        coupled = [(j, self._coupling[i][j]) for j in range(self._n) if self._coupling[i][j] > 0]
        if coupled:
            lines.append("  Coupled to:")
            for j, c in coupled:
                lines.append(f"    {self._meta[j].name}: {c:.2f}")
        
        return "\n".join(lines)
    
    def field_report(self) -> str:
        """Full agent field report."""
        lines = [
            f"=== AGENT FIELD REPORT ===",
            f"Rooms: {self._n}  Ticks: {self._tick_count}  Coherence: {self.coherence():.3f}",
            f"Clock BPM: {self._clock.bpm}  Drift: {self._clock.drift_ms():.3f}ms",
            "",
        ]
        
        # Focus queue
        fq = self.focus_queue()
        if fq:
            lines.append("Focus Queue:")
            for name, score in fq[:5]:
                lines.append(f"  {name}: {score:.4f}")
        else:
            lines.append("Focus Queue: (empty — no gaps)")
        
        lines.append("")
        
        # Per-room summary
        for i in range(self._n):
            meta = self._meta[i]
            s = self._state[i]
            gap = s[4]
            conf = s[0]
            lines.append(
                f"  {meta.name:20s} gap={gap:.3f} conf={conf:.3f} "
                f"chirality={meta.chirality:10s} ticks={meta.ticks}"
            )
        
        return "\n".join(lines)
