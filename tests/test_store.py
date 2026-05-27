"""Tests for plato_training.store — JSON-backed tile storage."""

import json
import tempfile
from pathlib import Path

import pytest

from plato_training.store import LocalTileStore, TileStore
from plato_training.types import (
    AdapterConfig,
    TileLifecycle,
    TileType,
    TrainingTile,
    content_hash,
)


def _make_tile(**kw):
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


class TestTileStore:
    def test_interface_raises(self):
        store = TileStore()
        with pytest.raises(NotImplementedError):
            store.save(TrainingTile())
        with pytest.raises(NotImplementedError):
            store.load("x")
        with pytest.raises(NotImplementedError):
            store.list_tiles()

    def test_repr(self):
        assert "TileStore" in repr(TileStore())


class TestLocalTileStore:
    def test_init_creates_dirs(self, tmp_path):
        store = LocalTileStore(str(tmp_path / "new-store"))
        assert (tmp_path / "new-store" / "tiles").is_dir()
        assert (tmp_path / "new-store" / "weights").is_dir()

    def test_save_and_load(self, tmp_path):
        store = LocalTileStore(str(tmp_path))
        tile = _make_tile()
        store.save(tile)
        loaded = store.load("test-001")
        assert loaded is not None
        assert loaded.tile_id == "test-001"
        assert loaded.room == "test-room"

    def test_load_missing(self, tmp_path):
        store = LocalTileStore(str(tmp_path))
        assert store.load("nonexistent") is None

    def test_list_tiles(self, tmp_path):
        store = LocalTileStore(str(tmp_path))
        store.save(_make_tile(tile_id="a", room="r1", lamport=1))
        store.save(_make_tile(tile_id="b", room="r1", lamport=2))
        store.save(_make_tile(tile_id="c", room="r2", lamport=3))
        all_tiles = store.list_tiles()
        assert len(all_tiles) == 3
        # Should be sorted by lamport
        assert all_tiles[0].lamport == 1

    def test_list_tiles_filter_room(self, tmp_path):
        store = LocalTileStore(str(tmp_path))
        store.save(_make_tile(tile_id="a", room="r1"))
        store.save(_make_tile(tile_id="b", room="r2"))
        tiles = store.list_tiles(room="r1")
        assert len(tiles) == 1
        assert tiles[0].room == "r1"

    def test_list_tiles_filter_state(self, tmp_path):
        store = LocalTileStore(str(tmp_path))
        tile1 = _make_tile(tile_id="a", state=TileLifecycle.ACTIVE)
        tile2 = _make_tile(tile_id="b", state=TileLifecycle.SUPERSEDED)
        store.save(tile1)
        store.save(tile2)
        active = store.list_tiles(state=TileLifecycle.ACTIVE)
        assert len(active) == 1

    def test_list_tiles_filter_type(self, tmp_path):
        store = LocalTileStore(str(tmp_path))
        store.save(_make_tile(tile_id="a", tile_type=TileType.ADAPTER))
        store.save(_make_tile(tile_id="b", tile_type=TileType.CHECKPOINT))
        adapters = store.list_tiles(tile_type=TileType.ADAPTER)
        assert len(adapters) == 1

    def test_find_active(self, tmp_path):
        store = LocalTileStore(str(tmp_path))
        store.save(_make_tile(tile_id="a", lamport=1, state=TileLifecycle.ACTIVE))
        store.save(_make_tile(tile_id="b", lamport=2, state=TileLifecycle.SUPERSEDED))
        active = store.find_active(room="test-room")
        assert active is not None
        assert active.tile_id == "a"

    def test_find_active_none(self, tmp_path):
        store = LocalTileStore(str(tmp_path))
        assert store.find_active(room="empty") is None

    def test_save_and_load_weights(self, tmp_path):
        store = LocalTileStore(str(tmp_path))
        data = b"fake weights data"
        h = content_hash(data)
        path = store.save_weights(h, data)
        assert path.exists()
        loaded = store.load_weights(h)
        assert loaded == data

    def test_load_weights_missing(self, tmp_path):
        store = LocalTileStore(str(tmp_path))
        assert store.load_weights("nonexistent") is None

    def test_delete(self, tmp_path):
        store = LocalTileStore(str(tmp_path))
        store.save(_make_tile())
        assert store.delete("test-001") is True
        assert store.load("test-001") is None

    def test_delete_missing(self, tmp_path):
        store = LocalTileStore(str(tmp_path))
        assert store.delete("nope") is False

    def test_stats(self, tmp_path):
        store = LocalTileStore(str(tmp_path))
        t1 = _make_tile(tile_id="a", state=TileLifecycle.ACTIVE)
        t2 = _make_tile(tile_id="b", state=TileLifecycle.SUPERSEDED)
        t3 = _make_tile(tile_id="c", state=TileLifecycle.RETRACTED)
        store.save(t1)
        store.save(t2)
        store.save(t3)
        stats = store.stats()
        assert stats["total"] == 3
        assert stats["active"] == 1
        assert stats["superseded"] == 1
        assert stats["retracted"] == 1

    def test_repr(self, tmp_path):
        store = LocalTileStore(str(tmp_path))
        assert "LocalTileStore" in repr(store)

    def test_roundtrip_with_config(self, tmp_path):
        """Full tile with nested dataclasses survives JSON round-trip."""
        from plato_training.types import AdapterConfig, TrainingConfig, TrainingMetrics
        store = LocalTileStore(str(tmp_path))
        tile = _make_tile(
            adapter_config=AdapterConfig(rank=16),
            training_config=TrainingConfig(epochs=5),
            metrics=TrainingMetrics(train_loss=0.1, loss_curve=[0.5, 0.3, 0.1]),
        )
        store.save(tile)
        loaded = store.load("test-001")
        assert loaded.adapter_config.rank == 16
        assert loaded.training_config.epochs == 5
        assert loaded.metrics.loss_curve == [0.5, 0.3, 0.1]
