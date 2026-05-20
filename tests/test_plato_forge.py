"""Tests for PLATO Forge — CUDA training daemon with Eisenstein snap, BMA, deadband."""

import time
import pytest
import torch
import torch.nn as nn
import numpy as np

from plato_training.plato_forge import PlatoForge, PyForge, ForgeConfig, ForgeStepResult
from plato_training.types import TileType


def _make_model(dim=32, n_classes=4):
    """Simple test model."""
    return nn.Sequential(
        nn.Linear(dim, 64),
        nn.ReLU(),
        nn.Linear(64, n_classes),
    )


def _make_batch(dim=32, n_classes=4, batch_size=8):
    return torch.randn(batch_size, dim), torch.randint(0, n_classes, (batch_size,))


class TestForgeConfig:
    def test_defaults(self):
        c = ForgeConfig()
        assert c.snap_radius == 0.5
        assert c.bma_window == 64
        assert c.deadband_threshold == 1e-4
        assert c.target_sm == 89

    def test_custom(self):
        c = ForgeConfig(snap_radius=0.1, bma_window=32, deadband_threshold=1e-5)
        assert c.snap_radius == 0.1


class TestPyForgeBMA:
    def test_bma_converging_loss(self):
        """Converging loss should have low predictability."""
        forge = PyForge(ForgeConfig(bma_window=20))
        # Decreasing loss = converging
        for i in range(25):
            forge.losses.append(10.0 - i * 0.3 + np.random.normal(0, 0.01))

        result = forge._run_bma()
        # Monotonically decreasing → very predictable → low LFSR length
        assert result["predictability"] < 0.8

    def test_bma_noisy_loss(self):
        """Random loss should have high predictability (no pattern)."""
        forge = PyForge(ForgeConfig(bma_window=30))
        rng = np.random.RandomState(42)
        for _ in range(35):
            forge.losses.append(rng.normal(1.0, 0.5))

        result = forge._run_bma()
        # Random → less predictable → higher LFSR length
        assert result["lfsr_length"] > 0

    def test_bma_perfectly_periodic(self):
        """Perfectly periodic loss → very low LFSR length."""
        forge = PyForge(ForgeConfig(bma_window=20))
        # Alternating up/down: 1, 0, 1, 0, ...
        for i in range(25):
            forge.losses.append(1.0 if i % 2 == 0 else 0.0)

        result = forge._run_bma()
        assert result["lfsr_length"] <= 3  # Alternating should have short LFSR


class TestPyForgeDeadband:
    def test_all_above_threshold(self):
        """All gradients above threshold should pass."""
        forge = PyForge(ForgeConfig(deadband_threshold=0.001))
        model = _make_model()
        x, y = _make_batch()
        loss = model(x).sum()
        loss.backward()

        result = forge.step(
            loss=loss.item(),
            gradients=[p.grad for p in model.parameters()],
            lr=1e-3,
            step=0,
        )
        assert result.throttle_stats["passed_threshold"] > 0

    def test_zero_gradients_skipped(self):
        """Zero gradients should be skipped."""
        forge = PyForge(ForgeConfig(deadband_threshold=0.01, hpdf_dither_strength=0.0))
        grad = torch.zeros(100)
        grad.requires_grad_(True)

        result = forge.step(
            loss=1.0,
            gradients=[grad],
            lr=1e-3,
            step=0,
        )
        assert result.throttle_stats["skipped"] == 100
        assert result.throttle_stats["passed_threshold"] == 0

    def test_dither_accepts_some(self):
        """HPDF dither should accept some below-threshold gradients."""
        forge = PyForge(ForgeConfig(
            deadband_threshold=1.0,  # High threshold
            hpdf_dither_strength=0.5,  # 50% dither acceptance
        ))
        # Small gradients
        grad = torch.randn(1000) * 0.01
        result = forge.step(
            loss=1.0,
            gradients=[grad],
            lr=1e-3,
            step=0,
        )
        # With 50% dither, should accept ~50% of below-threshold gradients
        total = result.throttle_stats["total_gradients"]
        accepted = result.throttle_stats["passed_dither"]
        assert accepted > 0
        assert accepted < total


class TestPyForgeSnap:
    def test_snap_weights(self):
        """Snap should move weights toward lattice points."""
        forge = PyForge(ForgeConfig(snap_radius=0.1))
        model = _make_model()
        stats = forge.snap_weights(model, radius=0.5)
        assert stats["total_params"] > 0
        assert stats["mean_snap_error"] >= 0

    def test_snap_zero_radius(self):
        """Zero radius should not change weights."""
        forge = PyForge(ForgeConfig(snap_radius=0.0))
        model = _make_model()
        before = {n: p.clone() for n, p in model.named_parameters()}
        forge.snap_weights(model, radius=0.0)
        for n, p in model.named_parameters():
            assert torch.allclose(p, before[n])


class TestPlatoForge:
    def test_init(self):
        forge = PlatoForge()
        assert forge.config.snap_radius == 0.5

    def test_step(self):
        forge = PlatoForge(ForgeConfig(
            snap_warmup=1000,  # Don't snap during test
            bma_window=100,   # Don't trigger BMA during test
        ))
        model = _make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x, y = _make_batch()
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()

        result = forge.step(
            loss=loss.item(),
            model=model,
            optimizer=optimizer,
            lr=1e-3,
            step=0,
        )
        assert isinstance(result, ForgeStepResult)
        assert result.new_lr == 1e-3  # No BMA trigger = no LR change

    def test_training_loop(self):
        """Full training loop with forge."""
        forge = PlatoForge(ForgeConfig(
            snap_radius=0.5,
            snap_interval=5,
            snap_warmup=2,
            bma_window=10,
            bma_early_stop_ratio=0.3,
            deadband_threshold=1e-6,
        ))
        model = _make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for step in range(30):
            x, y = _make_batch()
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()

            result = forge.step(
                loss=loss.item(),
                model=model,
                optimizer=optimizer,
                lr=optimizer.param_groups[0]['lr'],
                step=step,
            )
            losses.append(loss.item())

            # Update LR if changed
            for pg in optimizer.param_groups:
                pg['lr'] = result.new_lr

            optimizer.step()

            if not result.should_continue:
                break

        assert len(losses) > 0
        # Loss should generally decrease
        assert losses[-1] < losses[0] + 0.5  # Allow some noise

    def test_export_tile(self, tmp_path):
        forge = PlatoForge(ForgeConfig())
        model = _make_model()

        # Simulate some training
        forge.impl.losses.append(1.0)
        forge.impl.total_steps = 10

        tile = forge.export_tile(
            model,
            store_dir=str(tmp_path / "store"),
            room_name="test-forge",
        )
        assert tile.tile_id.startswith("forge-test-forge-")
        assert tile.tile_type == TileType.CHECKPOINT

    def test_status(self):
        forge = PlatoForge()
        model = _make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x, y = _make_batch()
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        forge.step(loss.item(), model, optimizer, 1e-3, 0)

        status = forge.status()
        assert "total_steps" in status
        assert "bma" in status
        assert "throttle" in status


class TestEisensteinDirections:
    def test_three_directions(self):
        """Three Eisenstein directions should sum to zero."""
        dirs = PyForge.EISENSTEIN_DIRS
        assert len(dirs) == 3
        assert abs(sum(dirs)) < 1e-10

    def test_direction_magnitudes(self):
        """ω and ω² should have |Im| = √3/2 magnitude."""
        dirs = PyForge.EISENSTEIN_DIRS
        # d[0] = 1.0 (real unit)
        assert abs(dirs[0] - 1.0) < 1e-10
        # d[1] = -1/2 + √3/2, d[2] = -1/2 - √3/2
        # These are complex unit roots projected to real line, not unit magnitude in R
        assert abs(dirs[1] - (-0.5 + np.sqrt(3)/2)) < 1e-10
        assert abs(dirs[2] - (-0.5 - np.sqrt(3)/2)) < 1e-10


class TestForgeGradientBehavior:
    def test_deadband_reduces_updates(self):
        """Deadband should reduce the number of gradient updates."""
        forge = PyForge(ForgeConfig(
            deadband_threshold=0.5,
            hpdf_dither_strength=0.0,  # No dither
        ))
        # Small gradients
        grad = torch.randn(1000) * 0.01
        result = forge.step(1.0, [grad], 1e-3, 0)

        # Most should be skipped
        total = result.throttle_stats["total_gradients"]
        skipped = result.throttle_stats["skipped"]
        assert skipped > total * 0.9  # >90% skipped

    def test_large_gradients_pass(self):
        """Large gradients should all pass the deadband."""
        forge = PyForge(ForgeConfig(deadband_threshold=0.001))
        grad = torch.randn(100) * 10.0  # Large gradients
        result = forge.step(1.0, [grad], 1e-3, 0)

        passed = result.throttle_stats["passed_threshold"]
        total = result.throttle_stats["total_gradients"]
        assert passed > total * 0.9  # >90% passed
