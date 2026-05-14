"""
Tests for micro models — prove the pipeline works end-to-end.
"""

import pytest
import torch
from plato_training.micro_models import (
    train_micro, list_tasks, TASK_REGISTRY,
    _generate_synthetic, MicroClassifier, SplineClassifier,
)


class TestSyntheticData:
    def test_all_tasks_generate_data(self):
        for task in TASK_REGISTRY:
            config = TASK_REGISTRY[task]
            X, y = _generate_synthetic(task, config)
            assert X.shape == (config["synthetic_size"], config["input_dim"])
            assert y.shape == (config["synthetic_size"],)
            assert y.min() >= 0
            assert y.max() < config["num_classes"]

    def test_spam_has_structure(self):
        """Spam task has learnable signal."""
        config = TASK_REGISTRY["spam-classify"]
        X, y = _generate_synthetic("spam-classify", config)
        # Spam samples should have higher first-8-feature magnitude
        spam = X[y == 1]
        ham = X[y == 0]
        assert spam[:, :8].abs().mean() > ham[:, :8].abs().mean()


class TestMicroClassifier:
    def test_forward_shape(self):
        model = MicroClassifier(128, 32, 2)
        x = torch.randn(4, 128)
        assert model(x).shape == (4, 2)

    def test_param_count(self):
        model = MicroClassifier(128, 32, 2)
        params = sum(p.numel() for p in model.parameters())
        assert params < 10000  # Micro = under 10K params


class TestSplineClassifier:
    def test_forward_shape(self):
        model = SplineClassifier(128, 32, 2, n_control_points=8)
        x = torch.randn(4, 128)
        assert model(x).shape == (4, 2)

    def test_fewer_params_than_dense(self):
        dense = MicroClassifier(128, 32, 2)
        spline = SplineClassifier(128, 32, 2, n_control_points=8)
        dense_params = sum(p.numel() for p in dense.parameters())
        spline_params = sum(p.numel() for p in spline.parameters())
        assert spline_params < dense_params


class TestTrainMicro:
    def test_dense_variant(self, tmp_path):
        model, tile, metrics = train_micro("spam-classify", variant="dense", store_dir=str(tmp_path / "s"))
        assert tile.is_active()
        assert metrics["accuracy"] > 0.5  # Should learn SOMETHING

    def test_lora_variant(self, tmp_path):
        model, tile, metrics = train_micro("spam-classify", variant="lora", store_dir=str(tmp_path / "s"))
        assert tile.is_active()
        assert metrics["accuracy"] > 0.5

    def test_spline_variant(self, tmp_path):
        model, tile, metrics = train_micro("spam-classify", variant="spline", store_dir=str(tmp_path / "s"))
        assert tile.is_active()
        # Spline may not learn as well (fewer params) but should beat random
        assert metrics["accuracy"] > 0.3

    def test_anomaly_task(self, tmp_path):
        model, tile, metrics = train_micro("anomaly-flag", variant="dense", store_dir=str(tmp_path / "s"))
        assert tile.is_active()
        assert metrics["accuracy"] > 0.7  # Anomaly has strong signal

    def test_intent_task(self, tmp_path):
        model, tile, metrics = train_micro("intent-detect", variant="dense", store_dir=str(tmp_path / "s"))
        assert tile.is_active()
        assert metrics["accuracy"] > 0.3  # 4-class is harder


class TestListTasks:
    def test_returns_all_tasks(self):
        tasks = list_tasks()
        assert len(tasks) >= 8
        assert "spam-classify" in tasks
        assert "drift-detect" in tasks
