"""
Tests for PLATO Training Rooms v0.2 — throttle, PyTorch room, TF room.
"""

import pytest
import torch
import torch.nn as nn
from plato_training import (
    PyTorchRoom, TrainingThrottle, ThrottleLevel, ThrottleState,
    AdapterConfig, TrainingConfig,
)


class TestThrottle:
    def test_idle_system(self):
        throttle = TrainingThrottle(custom_load_fn=lambda: 0.1)
        state = throttle.check()
        assert state.level == ThrottleLevel.FULL
        assert state.batch_multiplier == 1.0
        assert state.should_train

    def test_busy_system(self):
        throttle = TrainingThrottle(custom_load_fn=lambda: 0.7)
        state = throttle.check()
        assert state.level in (ThrottleLevel.MINIMAL, ThrottleLevel.REDUCED)
        assert state.batch_multiplier < 1.0

    def test_saturated_system(self):
        throttle = TrainingThrottle(custom_load_fn=lambda: 0.95)
        state = throttle.check()
        assert state.level == ThrottleLevel.PAUSED
        assert not state.should_train

    def test_min_level_enforced(self):
        throttle = TrainingThrottle(min_level=ThrottleLevel.MINIMAL, custom_load_fn=lambda: 0.1)
        state = throttle.check()
        assert state.level == ThrottleLevel.MINIMAL

    def test_effective_batch_size(self):
        throttle = TrainingThrottle(custom_load_fn=lambda: 0.5)
        effective = throttle.effective_batch_size(32)
        assert 1 <= effective <= 32

    def test_history(self):
        throttle = TrainingThrottle(custom_load_fn=lambda: 0.1)
        throttle.check()
        throttle.check()
        assert len(throttle.history()) == 2

    def test_summary(self):
        throttle = TrainingThrottle(custom_load_fn=lambda: 0.1)
        summary = throttle.summary()
        assert "full" in summary.lower()


class TestPyTorchRoom:
    def _model(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.W_query = nn.Linear(10, 32)
                self.W_value = nn.Linear(32, 32)
                self.out = nn.Linear(32, 2)
            def forward(self, x):
                return self.out(torch.relu(self.W_value(torch.relu(self.W_query(x)))))
        return M()

    def _data(self, n=50):
        X = torch.randn(n, 10)
        return torch.utils.data.TensorDataset(X, (X[:, 0] > 0).long())

    def test_train_produces_active_tile(self, tmp_path):
        room = PyTorchRoom("test", store_dir=str(tmp_path / "s"),
            throttle=TrainingThrottle(custom_load_fn=lambda: 0.1))
        tile = room.train(self._model(), self._data(),
            AdapterConfig(rank=4, alpha=8, target_modules=["W_query", "W_value"]),
            TrainingConfig(epochs=2, learning_rate=1e-3))
        assert tile.is_active()
        assert tile.metrics.epochs_completed == 2
        assert tile.content_hash != ""

    def test_supersede_on_retrain(self, tmp_path):
        room = PyTorchRoom("spam", store_dir=str(tmp_path / "s"),
            throttle=TrainingThrottle(custom_load_fn=lambda: 0.1))
        cfg = AdapterConfig(rank=4, alpha=8, target_modules=["W_query", "W_value"])
        tc = TrainingConfig(epochs=1, learning_rate=1e-3)
        v1 = room.train(self._model(), self._data(), cfg, tc)
        v2 = room.train(self._model(), self._data(), cfg, tc)
        assert v2.is_active()

    def test_throttle_integration(self, tmp_path):
        throttle = TrainingThrottle(custom_load_fn=lambda: 0.4)
        room = PyTorchRoom("throttled", store_dir=str(tmp_path / "s"), throttle=throttle)
        tile = room.train(self._model(), self._data(),
            AdapterConfig(rank=4, alpha=8, target_modules=["W_query", "W_value"]),
            TrainingConfig(epochs=1, learning_rate=1e-3))
        assert tile.is_active()
        assert len(throttle.history()) > 0

    def test_list_and_find_adapters(self, tmp_path):
        room = PyTorchRoom("list", store_dir=str(tmp_path / "s"),
            throttle=TrainingThrottle(custom_load_fn=lambda: 0.1))
        tile = room.train(self._model(), self._data(),
            AdapterConfig(rank=4, alpha=8, target_modules=["W_query", "W_value"]),
            TrainingConfig(epochs=1, learning_rate=1e-3))
        assert len(room.list_adapters()) >= 1
        assert room.active_adapter().tile_id == tile.tile_id


class TestTensorFlowRoom:
    @pytest.fixture(autouse=True)
    def check_tf(self):
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_train_produces_tile(self, tmp_path):
        import tensorflow as tf
        import numpy as np
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(2, activation='softmax')])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        X = np.random.randn(100, 10).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int32)
        from plato_training import TensorFlowRoom, TrainingThrottle
        room = TensorFlowRoom("sentiment", store_dir=str(tmp_path / "s"),
            throttle=TrainingThrottle(custom_load_fn=lambda: 0.1))
        tile = room.train(model, (X, y), TrainingConfig(epochs=2, batch_size=16))
        assert tile.is_active()
        assert tile.content_hash != ""
