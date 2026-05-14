"""
Training Throttle — fleet-aware resource management.
"""

from __future__ import annotations
import os
import time
import subprocess
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class ThrottleLevel(Enum):
    FULL = "full"
    REDUCED = "reduced"
    MINIMAL = "minimal"
    PAUSED = "paused"


@dataclass
class ThrottleState:
    level: ThrottleLevel
    batch_multiplier: float
    num_workers: int
    gpu_fraction: float
    check_interval_sec: float
    reason: str = ""

    @property
    def should_train(self) -> bool:
        return self.level != ThrottleLevel.PAUSED


LEVELS = {
    "idle":      (0.0, 0.3,  ThrottleLevel.FULL,    1.0, 4, 1.0, 30.0),
    "light":     (0.3, 0.6,  ThrottleLevel.REDUCED,  0.5, 2, 0.5, 15.0),
    "busy":      (0.6, 0.85, ThrottleLevel.MINIMAL,  0.25, 1, 0.25, 10.0),
    "saturated": (0.85, 1.0, ThrottleLevel.PAUSED,   0.0, 0, 0.0, 5.0),
}


def _get_cpu_load() -> float:
    try:
        return min(os.getloadavg()[0] / (os.cpu_count() or 1), 1.0)
    except (OSError, IndexError):
        return 0.5


def _get_gpu_load() -> float:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return min(float(result.stdout.strip().split('\n')[0]) / 8192, 1.0)
    except Exception:
        pass
    return 0.0


class TrainingThrottle:
    def __init__(self, min_level=ThrottleLevel.FULL, prefer_gpu=True, custom_load_fn=None):
        self.min_level = min_level
        self.prefer_gpu = prefer_gpu
        self.custom_load_fn = custom_load_fn
        self._last_check = 0.0
        self._last_state = None
        self._history = []

    def fleet_load(self) -> float:
        if self.custom_load_fn:
            return self.custom_load_fn()
        cpu = _get_cpu_load()
        gpu = _get_gpu_load() if self.prefer_gpu else 0.0
        return max(cpu, gpu)

    def check(self) -> ThrottleState:
        load = self.fleet_load()
        for name, (lo, hi, level, bm, w, gf, iv) in LEVELS.items():
            if lo <= load < hi or (name == "saturated" and load >= hi):
                state = ThrottleState(level, bm, w, gf, iv,
                    reason=f"fleet_load={load:.2f} zone={name}")
                break
        else:
            state = ThrottleState(ThrottleLevel.FULL, 1.0, 4, 1.0, 30.0,
                reason=f"fleet_load={load:.2f}")

        level_order = [ThrottleLevel.FULL, ThrottleLevel.REDUCED, ThrottleLevel.MINIMAL, ThrottleLevel.PAUSED]
        min_idx = level_order.index(self.min_level)
        if level_order.index(state.level) < min_idx:
            state.level = self.min_level
            state.batch_multiplier = min(state.batch_multiplier, 0.25)
            state.num_workers = min(state.num_workers, 1)

        self._last_state = state
        self._last_check = time.time()
        self._history.append((time.time(), state))
        return state

    def should_check(self) -> bool:
        if self._last_state is None: return True
        return time.time() - self._last_check >= self._last_state.check_interval_sec

    def wait_for_idle(self, timeout=3600.0, poll=10.0) -> ThrottleState:
        start = time.time()
        while time.time() - start < timeout:
            state = self.check()
            if state.should_train: return state
            time.sleep(poll)
        raise TimeoutError(f"Fleet not idle after {timeout}s")

    def effective_batch_size(self, base_batch: int) -> int:
        if self.should_check(): self._last_state = self.check()
        if self._last_state is None: return base_batch
        return max(1, int(base_batch * self._last_state.batch_multiplier))

    def history(self): return list(self._history)

    def summary(self) -> str:
        s = self._last_state or self.check()
        return f"[{s.level.value}] batch*{s.batch_multiplier:.2f} workers={s.num_workers} gpu={s.gpu_fraction:.0%} | {s.reason}"
