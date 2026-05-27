"""Tests for plato_training.types — core data types, no external deps."""

import json
import time

import pytest

from plato_training.types import (
    AdapterConfig,
    LamportClock,
    LifecycleEvent,
    TileLifecycle,
    TileType,
    TrainingConfig,
    TrainingMetrics,
    TrainingTile,
    content_hash,
)


# ---------------------------------------------------------------------------
# TileType / TileLifecycle enums
# ---------------------------------------------------------------------------

class TestTileType:
    def test_values(self):
        assert TileType.DATASET.value == "dataset"
        assert TileType.ADAPTER.value == "adapter"
        assert TileType.PREDICTION.value == "prediction"

    def test_all_members(self):
        expected = {"DATASET", "CHECKPOINT", "ADAPTER", "METRICS", "EVALUATION", "PREDICTION"}
        assert set(TileType.__members__.keys()) == expected


class TestTileLifecycle:
    def test_values(self):
        assert TileLifecycle.ACTIVE.value == "active"
        assert TileLifecycle.SUPERSEDED.value == "superseded"
        assert TileLifecycle.RETRACTED.value == "retracted"

    def test_repr(self):
        r = repr(TileLifecycle.ACTIVE)
        assert "TileLifecycle" in r


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

class TestAdapterConfig:
    def test_defaults(self):
        cfg = AdapterConfig()
        assert cfg.rank == 8
        assert cfg.alpha == 16
        assert cfg.dropout == 0.0
        assert isinstance(cfg.target_modules, list)

    def test_custom(self):
        cfg = AdapterConfig(rank=16, alpha=32, dropout=0.1)
        assert cfg.rank == 16
        assert cfg.alpha == 32
        assert cfg.dropout == 0.1


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.learning_rate == 2e-4
        assert cfg.epochs == 3
        assert cfg.batch_size == 8
        assert cfg.scheduler == "cosine"


class TestTrainingMetrics:
    def test_defaults(self):
        m = TrainingMetrics()
        assert m.train_loss == 0.0
        assert m.loss_curve == []

    def test_custom(self):
        m = TrainingMetrics(train_loss=0.5, val_loss=0.6, epochs_completed=5)
        assert m.train_loss == 0.5
        assert m.epochs_completed == 5


class TestLifecycleEvent:
    def test_defaults(self):
        ev = LifecycleEvent()
        assert ev.from_state == TileLifecycle.ACTIVE
        assert ev.reason == ""
        assert isinstance(ev.timestamp, float)

    def test_custom(self):
        ev = LifecycleEvent(
            from_state=TileLifecycle.ACTIVE,
            to_state=TileLifecycle.SUPERSEDED,
            reason="replaced",
        )
        assert ev.to_state == TileLifecycle.SUPERSEDED


# ---------------------------------------------------------------------------
# TrainingTile
# ---------------------------------------------------------------------------

class TestTrainingTile:
    def _make_tile(self, **kw):
        defaults = dict(
            tile_id="test-001",
            room="test-room",
            tile_type=TileType.ADAPTER,
            state=TileLifecycle.ACTIVE,
            lamport=1,
            name="test-adapter",
        )
        defaults.update(kw)
        return TrainingTile(**defaults)

    def test_basic_creation(self):
        tile = self._make_tile()
        assert tile.tile_id == "test-001"
        assert tile.room == "test-room"
        assert tile.is_active()

    def test_supersede(self):
        old = self._make_tile(tile_id="old-001", lamport=1)
        new = self._make_tile(tile_id="new-002", lamport=2)
        old.supersede(new, reason="better loss")
        assert old.state == TileLifecycle.SUPERSEDED
        assert new.parent_tile == "old-001"
        assert len(old.lifecycle_events) == 1

    def test_retract(self):
        tile = self._make_tile()
        tile.retract(reason="bad data")
        assert tile.state == TileLifecycle.RETRACTED
        assert not tile.is_active()
        assert tile.lifecycle_events[0].reason == "bad data"

    def test_transition(self):
        tile = self._make_tile()
        tile.transition(TileLifecycle.RETRACTED, reason="test", lamport=5)
        assert tile.state == TileLifecycle.RETRACTED
        assert tile.lifecycle_events[0].lamport == 5

    def test_history(self):
        tile = self._make_tile()
        tile.transition(TileLifecycle.SUPERSEDED, "r1", lamport=1)
        tile.transition(TileLifecycle.RETRACTED, "r2", lamport=2)
        h = tile.history()
        assert len(h) == 2
        assert h[0]["from"] == "active"
        assert h[0]["to"] == "superseded"

    def test_summary(self):
        tile = self._make_tile()
        s = tile.summary()
        assert "ADAPTER" in s
        assert "test-adapter" in s

    def test_to_dict_roundtrip(self):
        tile = self._make_tile(
            adapter_config=AdapterConfig(rank=4),
            training_config=TrainingConfig(epochs=10),
            metrics=TrainingMetrics(train_loss=0.3),
        )
        d = tile.to_dict()
        assert d["tile_type"] == "adapter"
        assert d["state"] == "active"
        assert d["adapter_config"]["rank"] == 4

        restored = TrainingTile.from_dict(d)
        assert restored.tile_id == tile.tile_id
        assert restored.adapter_config.rank == 4
        assert restored.metrics.train_loss == 0.3

    def test_to_dict_json_serializable(self):
        tile = self._make_tile()
        d = tile.to_dict()
        # Should not raise
        text = json.dumps(d)
        assert isinstance(text, str)

    def test_from_dict_with_lifecycle_events(self):
        d = {
            "tile_id": "t1",
            "room": "r1",
            "tile_type": "adapter",
            "state": "superseded",
            "lifecycle_events": [
                {"from_state": "active", "to_state": "superseded",
                 "reason": "test", "timestamp": 1.0, "lamport": 1}
            ],
        }
        tile = TrainingTile.from_dict(d)
        assert tile.state == TileLifecycle.SUPERSEDED
        assert len(tile.lifecycle_events) == 1
        assert tile.lifecycle_events[0].from_state == TileLifecycle.ACTIVE

    def test_from_dict_missing_optional_keys(self):
        """from_dict works with required fields present."""
        d = {"tile_id": "t1", "room": "r1", "tile_type": "adapter", "state": "active", "lamport": 1, "name": "n"}
        tile = TrainingTile.from_dict(d)
        assert tile.tile_id == "t1"
        assert tile.room == "r1"


# ---------------------------------------------------------------------------
# LamportClock
# ---------------------------------------------------------------------------

class TestLamportClock:
    def test_tick_increments(self):
        clock = LamportClock()
        assert clock.tick() == 1
        assert clock.tick() == 2
        assert clock.now() == 2

    def test_merge(self):
        a = LamportClock(time=5)
        result = a.merge(10)
        assert result == 11
        assert a.now() == 11

    def test_merge_lower(self):
        a = LamportClock(time=15)
        result = a.merge(3)
        assert result == 16

    def test_repr(self):
        r = repr(LamportClock(time=7))
        assert "7" in r


# ---------------------------------------------------------------------------
# content_hash
# ---------------------------------------------------------------------------

class TestContentHash:
    def test_deterministic(self):
        data = b"hello plato"
        h1 = content_hash(data)
        h2 = content_hash(data)
        assert h1 == h2

    def test_length(self):
        h = content_hash(b"test")
        assert len(h) == 16

    def test_different_data(self):
        h1 = content_hash(b"a")
        h2 = content_hash(b"b")
        assert h1 != h2
