"""Tests for tensorflow_room module."""

import tempfile

import pytest

from plato_training.types import (
    TileLifecycle,
    TileType,
    TrainingConfig,
)

# Check if TensorFlow is available
try:
    import tensorflow as tf

    HAS_TF = True
except ImportError:
    HAS_TF = False

from plato_training.tensorflow_room import TensorFlowRoom


# ---------------------------------------------------------------------------
# TensorFlowRoom init
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TF, reason="TensorFlow not installed")
class TestTensorFlowRoomInit:
    def test_creates_room(self, tmp_path):
        room = TensorFlowRoom("test-room", store_dir=str(tmp_path / "store"))
        assert room.room_name == "test-room"
        assert room.store is not None

    def test_default_throttle(self, tmp_path):
        room = TensorFlowRoom("room", store_dir=str(tmp_path / "store"))
        assert room.throttle is not None

    def test_custom_throttle(self, tmp_path):
        from plato_training.throttle import TrainingThrottle
        t = TrainingThrottle()
        room = TensorFlowRoom("room", store_dir=str(tmp_path / "store"), throttle=t)
        assert room.throttle is t


# ---------------------------------------------------------------------------
# TensorFlowRoom training
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TF, reason="TensorFlow not installed")
class TestTensorFlowRoomTrain:
    def _make_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation="relu", input_shape=(8,)),
            tf.keras.layers.Dense(4, activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        return model

    def _make_data(self, n=64):
        import numpy as np
        X = np.random.randn(n, 8).astype(np.float32)
        y = np.random.randint(0, 4, n).astype(np.int32)
        return X, y

    def test_basic_training(self, tmp_path):
        room = TensorFlowRoom("test-room", store_dir=str(tmp_path / "store"))
        model = self._make_model()
        dataset = self._make_data(n=32)
        tile = room.train(
            model,
            dataset,
            training_config=TrainingConfig(epochs=2, batch_size=16),
        )
        assert tile is not None
        assert tile.tile_type == TileType.ADAPTER
        assert tile.state == TileLifecycle.ACTIVE
        assert tile.metrics is not None
        assert tile.metrics.epochs_completed == 2

    def test_tile_stored(self, tmp_path):
        room = TensorFlowRoom("test-room", store_dir=str(tmp_path / "store"))
        model = self._make_model()
        dataset = self._make_data(n=32)
        tile = room.train(
            model,
            dataset,
            training_config=TrainingConfig(epochs=1, batch_size=16),
        )
        adapters = room.list_adapters()
        assert len(adapters) >= 1

    def test_active_adapter(self, tmp_path):
        room = TensorFlowRoom("test-room", store_dir=str(tmp_path / "store"))
        model = self._make_model()
        dataset = self._make_data(n=32)
        tile = room.train(
            model,
            dataset,
            training_config=TrainingConfig(epochs=1, batch_size=16),
        )
        active = room.active_adapter()
        assert active is not None
        assert active.tile_id == tile.tile_id

    def test_loss_curve_recorded(self, tmp_path):
        room = TensorFlowRoom("test-room", store_dir=str(tmp_path / "store"))
        model = self._make_model()
        dataset = self._make_data(n=32)
        tile = room.train(
            model,
            dataset,
            training_config=TrainingConfig(epochs=3, batch_size=16),
        )
        assert tile.metrics.loss_curve is not None
        assert len(tile.metrics.loss_curve) == 3

    def test_content_hash_stored(self, tmp_path):
        room = TensorFlowRoom("test-room", store_dir=str(tmp_path / "store"))
        model = self._make_model()
        dataset = self._make_data(n=32)
        tile = room.train(
            model,
            dataset,
            training_config=TrainingConfig(epochs=1, batch_size=16),
        )
        assert tile.content_hash is not None
        assert len(tile.content_hash) > 0

    def test_supersedes_previous(self, tmp_path):
        """Second training should supersede previous adapter if loss improves."""
        room = TensorFlowRoom("test-room", store_dir=str(tmp_path / "store"))
        dataset = self._make_data(n=64)

        model1 = self._make_model()
        tile1 = room.train(
            model1, dataset,
            training_config=TrainingConfig(epochs=1, batch_size=16),
        )

        model2 = self._make_model()
        tile2 = room.train(
            model2, dataset,
            training_config=TrainingConfig(epochs=2, batch_size=16),
        )

        assert tile2.tile_id != tile1.tile_id
        adapters = room.list_adapters()
        assert len(adapters) >= 2


# ---------------------------------------------------------------------------
# No TF — ImportError test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(HAS_TF, reason="TensorFlow is installed")
class TestNoTensorFlow:
    def test_raises_without_tf(self, tmp_path):
        with pytest.raises(ImportError, match="TensorFlow not installed"):
            TensorFlowRoom("room", store_dir=str(tmp_path / "store"))
