"""Tests for GPU fleet trainer."""

import time
import math
import pytest
import torch
import numpy as np

from plato_training.fleet_miner import CommitPoint
from plato_training.gpu_fleet_trainer import (
    GPUFleetConfig,
    FleetMultiTaskDataset,
    FleetGPT2,
    FleetTrainResult,
    train_gpu_fleet,
    compress_with_spline,
    predict_fleet_gpu,
    export_gpu_tile,
)


def _make_commits(n: int = 200, repos: list = None) -> list:
    """Generate synthetic CommitPoints."""
    if repos is None:
        repos = ["plato-training", "forgemaster", "tensor-spline", "plato-data"]
    commits = []
    now = time.time()
    for i in range(n):
        commits.append(CommitPoint(
            sha=f"synth{i:06d}",
            repo=repos[i % len(repos)],
            author=f"agent-{i % 4}",
            timestamp=now - (n - i) * 1800,  # 30 min apart
            message=f"commit {i}: {'feat' if i % 3 == 0 else 'fix'} something",
            files_changed=(i % 5) + 1,
            insertions=(i % 20) * 10,
            deletions=(i % 10) * 5,
            is_merge=False,
            languages=[".py"] if i % 2 == 0 else [".rs", ".md"],
            cross_refs=["tensor-spline"] if i % 5 == 0 else [],
        ))
    return commits


class TestGPUFleetConfig:
    def test_defaults(self):
        cfg = GPUFleetConfig()
        assert cfg.n_layer == 6
        assert cfg.n_head == 6
        assert cfg.n_embd == 384
        assert cfg.use_amp is True

    def test_param_count(self):
        cfg = GPUFleetConfig()
        n = cfg.model_params()
        assert n > 100_000, f"Expected >100K params, got {n}"

    def test_effective_batch(self):
        cfg = GPUFleetConfig(batch_size=16, grad_accum_steps=4)
        assert cfg.effective_batch_size() == 64

    def test_vram_estimate(self):
        cfg = GPUFleetConfig()
        vram = cfg.estimated_vram_mb()
        assert vram > 0
        # Should fit in 6GB
        assert vram < 6000, f"Estimated {vram}MB, too much for RTX 4050"

    def test_small_config(self):
        cfg = GPUFleetConfig(n_layer=2, n_head=4, n_embd=128, block_size=64)
        n = cfg.model_params()
        assert n < 500_000


class TestFleetMultiTaskDataset:
    def test_creation(self):
        commits = _make_commits(100)
        cfg = GPUFleetConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, seq_len=4)
        ds = FleetMultiTaskDataset(commits, cfg)
        assert len(ds) > 0

    def test_sample_shapes(self):
        commits = _make_commits(100)
        cfg = GPUFleetConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, seq_len=4)
        ds = FleetMultiTaskDataset(commits, cfg)
        sample = ds[0]
        assert sample["input_ids"].shape == (32,)
        assert sample["activity_label"].dim() == 0  # scalar
        assert sample["extension_labels"].shape[0] == 10  # LANG_VOCAB size

    def test_minimal_commits(self):
        """Should handle very few commits."""
        commits = _make_commits(30)  # enough for at least a few windows
        cfg = GPUFleetConfig(n_layer=1, n_head=2, n_embd=32, block_size=16, seq_len=2)
        ds = FleetMultiTaskDataset(commits, cfg)
        # May be 0 if too few for sequences — just verify no crash
        assert isinstance(len(ds), int)


class TestFleetGPT2:
    def test_creation(self):
        cfg = GPUFleetConfig(n_layer=2, n_head=4, n_embd=128, block_size=64)
        model = FleetGPT2(cfg)
        n = sum(p.numel() for p in model.parameters())
        assert n > 0

    def test_forward(self):
        cfg = GPUFleetConfig(n_layer=2, n_head=4, n_embd=128, block_size=64)
        model = FleetGPT2(cfg)
        x = torch.randint(0, 80, (4, 32))  # batch=4, seq=32
        result = model(x)
        assert "activity_logits" in result
        assert "ext_logits" in result
        assert "lm_logits" in result
        assert result["activity_logits"].shape == (4, 10)

    def test_forward_with_labels(self):
        cfg = GPUFleetConfig(n_layer=2, n_head=4, n_embd=128, block_size=64)
        model = FleetGPT2(cfg)
        x = torch.randint(0, 80, (4, 32))
        act = torch.randint(0, 10, (4,))
        ext = torch.randn(4, 10)
        ext = ext / ext.sum(dim=-1, keepdim=True)
        lm = torch.randint(0, 80, (4, 32))
        result = model(x, activity_label=act, extension_labels=ext, lm_labels=lm)
        assert "activity_loss" in result
        assert "ext_loss" in result
        assert "lm_loss" in result
        assert result["activity_loss"].requires_grad

    def test_gpu_forward(self):
        if not torch.cuda.is_available():
            pytest.skip("No GPU")
        cfg = GPUFleetConfig(n_layer=2, n_head=4, n_embd=128, block_size=64)
        model = FleetGPT2(cfg).cuda()
        x = torch.randint(0, 80, (4, 32)).cuda()
        result = model(x)
        assert result["activity_logits"].device.type == "cuda"


class TestTrainGPUFleet:
    def test_train_cpu_small(self):
        """Quick CPU training test."""
        commits = _make_commits(100)
        cfg = GPUFleetConfig(
            n_layer=1, n_head=2, n_embd=32, block_size=16,
            epochs=5, batch_size=8, grad_accum_steps=1,
            use_amp=False, device="cpu", seq_len=4,
        )
        result = train_gpu_fleet(commits, config=cfg, verbose=False)
        assert len(result.train_losses) > 0
        assert result.params_count > 0
        assert result.training_seconds > 0

    def test_train_gpu(self):
        """GPU training with mixed precision."""
        if not torch.cuda.is_available():
            pytest.skip("No GPU")
        commits = _make_commits(200)
        cfg = GPUFleetConfig(
            n_layer=2, n_head=4, n_embd=128, block_size=64,
            epochs=10, batch_size=16, grad_accum_steps=2,
            use_amp=True, device="cuda", seq_len=8,
        )
        result = train_gpu_fleet(commits, config=cfg, verbose=True)
        assert result.peak_vram_mb > 0
        assert result.val_metrics["activity_accuracy"] >= 0
        assert result.training_seconds < 120  # Should be fast on GPU

    def test_loss_decreases(self):
        """Training loss should generally decrease."""
        commits = _make_commits(200)
        cfg = GPUFleetConfig(
            n_layer=2, n_head=4, n_embd=64, block_size=32,
            epochs=20, batch_size=8, grad_accum_steps=1,
            use_amp=False, device="cpu", seq_len=4,
        )
        result = train_gpu_fleet(commits, config=cfg, verbose=False)
        # First 10% vs last 10% of losses
        n = len(result.train_losses)
        first = np.mean(result.train_losses[:max(n // 10, 1)])
        last = np.mean(result.train_losses[-max(n // 10, 1):])
        assert last < first, f"Loss didn't decrease: first={first:.4f}, last={last:.4f}"


class TestCompression:
    def test_spline_analysis(self):
        cfg = GPUFleetConfig(n_layer=2, n_head=4, n_embd=128, block_size=64)
        model = FleetGPT2(cfg)
        result = compress_with_spline(model, rank=32)
        assert result["compression_ratio"] > 1.0
        assert result["total_original"] > 0
        assert result["total_compressed"] > 0

    def test_compression_ratio_reasonable(self):
        """Should achieve at least 5x compression."""
        cfg = GPUFleetConfig(n_layer=4, n_head=6, n_embd=256, block_size=128)
        model = FleetGPT2(cfg)
        result = compress_with_spline(model, rank=16)
        assert result["compression_ratio"] >= 5.0, \
            f"Only {result['compression_ratio']:.1f}x compression"


class TestPrediction:
    def test_predict(self):
        commits = _make_commits(100)
        cfg = GPUFleetConfig(
            n_layer=1, n_head=2, n_embd=32, block_size=16,
            epochs=3, device="cpu", seq_len=4, use_amp=False,
        )
        result = train_gpu_fleet(commits, config=cfg, verbose=False)
        preds = predict_fleet_gpu(result.model, commits, cfg, device="cpu")
        assert len(preds) > 0
        assert "predicted_activity_bin" in preds[0]
        assert "activity_confidence" in preds[0]

    def test_predict_on_gpu(self):
        if not torch.cuda.is_available():
            pytest.skip("No GPU")
        commits = _make_commits(100)
        cfg = GPUFleetConfig(
            n_layer=2, n_head=4, n_embd=64, block_size=32,
            epochs=5, device="cuda", seq_len=4,
        )
        result = train_gpu_fleet(commits, config=cfg, verbose=False)
        preds = predict_fleet_gpu(result.model, commits, cfg, device="cuda")
        assert len(preds) > 0


class TestExportTile:
    def test_export(self, tmp_path):
        commits = _make_commits(50)
        cfg = GPUFleetConfig(
            n_layer=1, n_head=2, n_embd=32, block_size=16,
            epochs=3, device="cpu", seq_len=4, use_amp=False,
        )
        result = train_gpu_fleet(commits, config=cfg, verbose=False)
        tile = export_gpu_tile(
            result,
            store_dir=str(tmp_path / "store"),
            room_name="test-room",
        )
        assert tile.tile_id.startswith("gpu-fleet-")
        assert tile.metrics is not None
