"""
PyTorch Training Room — agent walks in with data, walks out with trained adapter.
"""

from __future__ import annotations
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional, List

from .types import (TrainingTile, TileType, TileLifecycle, LamportClock,
    AdapterConfig, TrainingConfig, TrainingMetrics, content_hash)
from .adapters.lora import inject_lora, save_lora_weights, load_lora_weights
from .store import LocalTileStore
from .throttle import TrainingThrottle


class PyTorchRoom:
    def __init__(self, room_name, store_dir=".plato-training", device=None, throttle=None):
        self.room_name = room_name
        self.clock = LamportClock()
        self.store = LocalTileStore(store_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.throttle = throttle or TrainingThrottle()
        self.model = None

    def set_base_model(self, model):
        self.model = model

    def train(self, model, dataset, adapter_config=None, training_config=None, loss_fn=None, num_classes=None):
        if adapter_config is None: adapter_config = AdapterConfig()
        if training_config is None: training_config = TrainingConfig()
        if loss_fn is None: loss_fn = nn.CrossEntropyLoss()

        start_time = time.time()
        injection_map = inject_lora(model, rank=adapter_config.rank, alpha=adapter_config.alpha,
            target_modules=adapter_config.target_modules, dropout=adapter_config.dropout)
        model.to(self.device)

        if num_classes and hasattr(model, 'out_head'):
            model.out_head = nn.Linear(model.out_head.in_features, num_classes).to(self.device)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay)

        loss_curve = []
        peak_memory = 0.0
        scheduler = None

        for epoch in range(training_config.epochs):
            state = self.throttle.check()
            if not state.should_train:
                print(f"[throttle] {state.level.value} — waiting...")
                self.throttle.wait_for_idle()
                state = self.throttle.check()

            effective_batch = max(1, int(training_config.batch_size * state.batch_multiplier))
            loader = DataLoader(dataset, batch_size=effective_batch, shuffle=True,
                num_workers=state.num_workers)

            if epoch == 0:
                total_steps = len(loader) * training_config.epochs
                scheduler = self._build_scheduler(optimizer, training_config, total_steps)

            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in loader:
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                optimizer.zero_grad()
                logits = model(inputs)
                loss = self._compute_loss(logits, targets, loss_fn)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, training_config.max_grad_norm)
                optimizer.step()
                if scheduler: scheduler.step()
                epoch_loss += loss.item()
                n_batches += 1
                loss_curve.append(loss.item())
                if torch.cuda.is_available():
                    peak_memory = max(peak_memory, torch.cuda.max_memory_allocated() / 1e6)

            print(f"Epoch {epoch+1}/{training_config.epochs}: loss={epoch_loss/max(n_batches,1):.4f} [{state.level.value}] batch={effective_batch}")

        training_time = time.time() - start_time
        model.cpu()
        lamport = self.clock.tick()

        weight_path = str(self.store.weights_dir / f"{self.room_name}-L{lamport}.safetensors")
        raw_bytes = save_lora_weights(model, injection_map, weight_path)
        c_hash = content_hash(raw_bytes)
        self.store.save_weights(c_hash, raw_bytes)

        tile = TrainingTile(
            tile_id=f"{self.room_name}-{lamport:03d}", room=self.room_name,
            tile_type=TileType.ADAPTER, state=TileLifecycle.ACTIVE, lamport=lamport,
            name=f"adapter-{self.room_name}",
            description=f"LoRA r={adapter_config.rank} a={adapter_config.alpha} epochs={training_config.epochs}",
            content_hash=c_hash, base_model="custom",
            adapter_config=adapter_config, training_config=training_config,
            metrics=TrainingMetrics(
                train_loss=loss_curve[-1] if loss_curve else 0.0,
                epochs_completed=training_config.epochs,
                training_time_seconds=training_time,
                peak_memory_mb=peak_memory,
                final_loss=loss_curve[-1] if loss_curve else 0.0,
                loss_curve=loss_curve),
            source_room=self.room_name)

        previous = self.store.find_active(tile_type=TileType.ADAPTER, room=self.room_name)
        if previous and previous.tile_id != tile.tile_id:
            if tile.metrics and previous.metrics and tile.metrics.final_loss < previous.metrics.final_loss:
                previous.supersede(tile, reason=f"Better: {tile.metrics.final_loss:.4f} < {previous.metrics.final_loss:.4f}")
                self.store.save(previous)

        self.store.save(tile)
        return tile

    def load_adapter(self, tile):
        if self.model is None: raise ValueError("No base model. Call set_base_model().")
        model = self.model
        injection_map = inject_lora(model, rank=tile.adapter_config.rank,
            alpha=tile.adapter_config.alpha, target_modules=tile.adapter_config.target_modules)
        import os
        for pattern in [f"{tile.content_hash}", f"{tile.tile_id}", f"{self.room_name}-L{tile.lamport}"]:
            path = str(self.store.weights_dir / f"{pattern}.safetensors")
            if os.path.exists(path):
                load_lora_weights(model, injection_map, path)
                model.cpu()
                return model
        raise FileNotFoundError(f"No weights for {tile.tile_id}")

    def list_adapters(self):
        return self.store.list_tiles(room=self.room_name, tile_type=TileType.ADAPTER)

    def active_adapter(self):
        return self.store.find_active(tile_type=TileType.ADAPTER, room=self.room_name)

    def _build_scheduler(self, optimizer, config, total_steps):
        from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
        if config.scheduler == "cosine":
            return CosineAnnealingLR(optimizer, T_max=total_steps)
        elif config.scheduler == "linear":
            def lr_lambda(step):
                if step < config.warmup_steps: return step / max(config.warmup_steps, 1)
                return max(0.0, 1.0 - (step - config.warmup_steps) / max(total_steps - config.warmup_steps, 1))
            return LambdaLR(optimizer, lr_lambda)
        return None

    def _compute_loss(self, logits, targets, loss_fn):
        if logits.dim() == 2 and targets.dim() == 1: return loss_fn(logits, targets)
        if logits.dim() > 2: return loss_fn(logits.flatten(0, 1), targets.flatten())
        return loss_fn(logits, targets)
