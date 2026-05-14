"""
TensorFlow Training Room — same PLATO room protocol, TF engine.
"""

from __future__ import annotations
import time
from typing import Optional, List

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

from .types import (TrainingTile, TileType, TileLifecycle, LamportClock,
    TrainingConfig, TrainingMetrics, content_hash)
from .store import LocalTileStore
from .throttle import TrainingThrottle


if HAS_TF:

    class ThrottleCallback(tf.keras.callbacks.Callback):
        def __init__(self, throttle):
            super().__init__()
            self.throttle = throttle
        def on_epoch_begin(self, epoch, logs=None):
            state = self.throttle.check()
            if not state.should_train:
                print(f"[throttle] {state.level.value} — waiting...")
                self.throttle.wait_for_idle()
            self._state = self.throttle.check()
            print(f"[throttle] epoch {epoch+1}: {self._state.level.value}")

    class CheckpointCallback(tf.keras.callbacks.Callback):
        def __init__(self, store, clock, room_name):
            super().__init__()
            self.store = store
            self.clock = clock
            self.room_name = room_name
            self.best_val_loss = float('inf')
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            val_loss = logs.get('val_loss', logs.get('loss', float('inf')))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                lamport = self.clock.tick()
                tile = TrainingTile(
                    tile_id=f"{self.room_name}-ckpt-{lamport:03d}",
                    room=self.room_name, tile_type=TileType.CHECKPOINT,
                    state=TileLifecycle.ACTIVE, lamport=lamport,
                    name=f"checkpoint-epoch{epoch+1}",
                    metrics=TrainingMetrics(train_loss=logs.get('loss', 0.0),
                        val_loss=val_loss, epochs_completed=epoch+1))
                self.store.save(tile)

    class TensorFlowRoom:
        def __init__(self, room_name, store_dir=".plato-training", throttle=None):
            self.room_name = room_name
            self.clock = LamportClock()
            self.store = LocalTileStore(store_dir)
            self.throttle = throttle or TrainingThrottle()

        def train(self, model, dataset, training_config=None, validation_split=0.2):
            if training_config is None: training_config = TrainingConfig()
            X, y = dataset
            start_time = time.time()

            throttle_cb = ThrottleCallback(self.throttle)
            checkpoint_cb = CheckpointCallback(self.store, self.clock, self.room_name)

            state = self.throttle.check()
            effective_batch = max(1, int(training_config.batch_size * state.batch_multiplier))

            history = model.fit(X, y, batch_size=effective_batch,
                epochs=training_config.epochs, validation_split=validation_split,
                callbacks=[throttle_cb, checkpoint_cb], verbose=1)

            training_time = time.time() - start_time
            lamport = self.clock.tick()
            weight_bytes = b"".join(w.tobytes() for w in model.get_weights())
            c_hash = content_hash(weight_bytes)
            self.store.save_weights(c_hash, weight_bytes)

            loss_curve = history.history.get('loss', [])
            val_loss = history.history.get('val_loss', [None])

            tile = TrainingTile(
                tile_id=f"{self.room_name}-{lamport:03d}", room=self.room_name,
                tile_type=TileType.ADAPTER, state=TileLifecycle.ACTIVE, lamport=lamport,
                name=f"model-{self.room_name}",
                description=f"TF/Keras epochs={training_config.epochs}",
                content_hash=c_hash, base_model=model.__class__.__name__,
                training_config=training_config,
                metrics=TrainingMetrics(
                    train_loss=loss_curve[-1] if loss_curve else 0.0,
                    val_loss=val_loss[-1] if val_loss and val_loss[-1] else 0.0,
                    epochs_completed=training_config.epochs,
                    training_time_seconds=training_time,
                    final_loss=loss_curve[-1] if loss_curve else 0.0,
                    loss_curve=loss_curve),
                source_room=self.room_name)

            previous = self.store.find_active(tile_type=TileType.ADAPTER, room=self.room_name)
            if previous and previous.tile_id != tile.tile_id:
                if tile.metrics and previous.metrics and tile.metrics.final_loss < previous.metrics.final_loss:
                    previous.supersede(tile, reason=f"Better: {tile.metrics.final_loss:.4f}")
                    self.store.save(previous)

            self.store.save(tile)
            return tile

        def list_adapters(self):
            return self.store.list_tiles(room=self.room_name, tile_type=TileType.ADAPTER)

        def active_adapter(self):
            return self.store.find_active(tile_type=TileType.ADAPTER, room=self.room_name)

else:
    class TensorFlowRoom:
        def __init__(self, *a, **kw): raise ImportError("TensorFlow not installed. pip install tensorflow")
