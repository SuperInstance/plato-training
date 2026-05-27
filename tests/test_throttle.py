"""Tests for plato_training.throttle — fleet-aware resource management."""

import time
from unittest.mock import patch

import pytest

from plato_training.throttle import (
    LEVELS,
    ThrottleLevel,
    ThrottleState,
    TrainingThrottle,
    _get_cpu_load,
    _get_gpu_load,
)


class TestThrottleLevel:
    def test_members(self):
        assert ThrottleLevel.FULL.value == "full"
        assert ThrottleLevel.REDUCED.value == "reduced"
        assert ThrottleLevel.MINIMAL.value == "minimal"
        assert ThrottleLevel.PAUSED.value == "paused"

    def test_repr(self):
        assert "ThrottleLevel" in repr(ThrottleLevel.FULL)


class TestThrottleState:
    def test_should_train_true(self):
        state = ThrottleState(
            level=ThrottleLevel.FULL,
            batch_multiplier=1.0,
            num_workers=4,
            gpu_fraction=1.0,
            check_interval_sec=30.0,
        )
        assert state.should_train is True

    def test_should_train_false_when_paused(self):
        state = ThrottleState(
            level=ThrottleLevel.PAUSED,
            batch_multiplier=0.0,
            num_workers=0,
            gpu_fraction=0.0,
            check_interval_sec=5.0,
        )
        assert state.should_train is False

    def test_reason_default(self):
        state = ThrottleState(
            level=ThrottleLevel.REDUCED,
            batch_multiplier=0.5,
            num_workers=2,
            gpu_fraction=0.5,
            check_interval_sec=15.0,
        )
        assert state.reason == ""


class TestLoadHelpers:
    def test_cpu_load_returns_float(self):
        load = _get_cpu_load()
        assert isinstance(load, float)
        assert 0.0 <= load <= 1.0

    def test_gpu_load_returns_float(self):
        load = _get_gpu_load()
        assert isinstance(load, float)
        assert 0.0 <= load <= 1.0


class TestTrainingThrottle:
    def test_custom_load_fn(self):
        throttle = TrainingThrottle(custom_load_fn=lambda: 0.1)
        state = throttle.check()
        assert state.level == ThrottleLevel.FULL

    def test_custom_load_fn_saturated(self):
        throttle = TrainingThrottle(custom_load_fn=lambda: 0.95)
        state = throttle.check()
        assert state.level == ThrottleLevel.PAUSED
        assert state.should_train is False

    def test_custom_load_fn_reduced(self):
        throttle = TrainingThrottle(custom_load_fn=lambda: 0.45)
        state = throttle.check()
        assert state.level == ThrottleLevel.REDUCED

    def test_custom_load_fn_minimal(self):
        throttle = TrainingThrottle(custom_load_fn=lambda: 0.7)
        state = throttle.check()
        assert state.level == ThrottleLevel.MINIMAL

    def test_min_level_override(self):
        """min_level prevents throttle from being less restrictive than configured."""
        throttle = TrainingThrottle(
            min_level=ThrottleLevel.REDUCED,
            custom_load_fn=lambda: 0.05,  # Would be FULL
        )
        state = throttle.check()
        assert state.level == ThrottleLevel.REDUCED

    def test_history_tracks_checks(self):
        throttle = TrainingThrottle(custom_load_fn=lambda: 0.1)
        throttle.check()
        throttle.check()
        h = throttle.history()
        assert len(h) == 2

    def test_summary(self):
        throttle = TrainingThrottle(custom_load_fn=lambda: 0.1)
        s = throttle.summary()
        assert "full" in s.lower() or "FULL" in s

    def test_should_check_initially_true(self):
        throttle = TrainingThrottle()
        assert throttle.should_check() is True

    def test_effective_batch_size(self):
        throttle = TrainingThrottle(custom_load_fn=lambda: 0.1)
        batch = throttle.effective_batch_size(32)
        assert batch == 32  # FULL = 1.0 multiplier

    def test_effective_batch_size_reduced(self):
        throttle = TrainingThrottle(custom_load_fn=lambda: 0.45)  # REDUCED
        batch = throttle.effective_batch_size(32)
        assert batch == 16  # 32 * 0.5

    def test_repr(self):
        t = TrainingThrottle(prefer_gpu=False)
        r = repr(t)
        assert "TrainingThrottle" in r

    def test_wait_for_idle_raises_on_timeout(self):
        throttle = TrainingThrottle(custom_load_fn=lambda: 0.95)
        with pytest.raises(TimeoutError):
            throttle.wait_for_idle(timeout=0.1, poll=0.05)

    def test_wait_for_idle_returns_when_idle(self):
        throttle = TrainingThrottle(custom_load_fn=lambda: 0.1)
        state = throttle.wait_for_idle(timeout=1.0, poll=0.01)
        assert state.should_train

    def test_fleet_load_uses_custom_fn(self):
        calls = []
        def my_load():
            calls.append(1)
            return 0.2
        throttle = TrainingThrottle(custom_load_fn=my_load)
        throttle.fleet_load()
        assert len(calls) == 1


class TestLevels:
    def test_levels_has_four_zones(self):
        assert set(LEVELS.keys()) == {"idle", "light", "busy", "saturated"}
