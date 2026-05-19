"""
Tests for gpt2_trainer.py — TinyGPT2 fleet activity predictor.
"""

import time
import math
import numpy as np
import pytest
from datetime import datetime, timezone
from pathlib import Path

from plato_training.gpt2_trainer import (
    # Token encoding
    VOCAB_SIZE, SPECIAL_TOKENS,
    REPO_TOKEN_OFFSET, HOUR_TOKEN_OFFSET, DAY_TOKEN_OFFSET,
    LANG_TOKEN_OFFSET, COUNT_TOKEN_OFFSET,
    hour_to_bin, day_to_bin, repo_to_token, lang_to_token, count_to_bin,
    encode_commit, encode_hour_window,
    # Data preparation
    HourWindow, commits_to_hour_windows, windows_to_sequences,
    CommitSequenceDataset,
    # Model
    TinyGPT2, TinyGPT2Config,
    # Training
    train_fleet_gpt2, evaluate_fleet_gpt2, TrainResult,
    # Inference
    predict_next_hour,
    # Export
    export_tile, load_tile_model,
    # Vocabularies
    REPO_VOCAB, LANG_VOCAB,
)
from plato_training.fleet_miner import CommitPoint


# ─── Fixtures ────────────────────────────────────────────────────────

def make_commit(repo="plato-training", hours_ago=0, author="agent",
                sha="abc123", insertions=10, deletions=5,
                cross_refs=None, languages=None):
    return CommitPoint(
        sha=sha, repo=repo, author=author,
        timestamp=time.time() - hours_ago * 3600,
        message="test commit", files_changed=2,
        insertions=insertions, deletions=deletions,
        is_merge=False, cross_refs=cross_refs or [],
        languages=languages or [".py"],
    )


def make_many_commits(n=100, hours_span=72):
    """Generate n commits spread over hours_span hours across multiple repos."""
    commits = []
    repos = ["plato-training", "plato-types", "tensor-spline", "plato-data"]
    for i in range(n):
        hours_ago = hours_span * (1 - i / n)
        repo = repos[i % len(repos)]
        commits.append(make_commit(
            repo=repo,
            hours_ago=hours_ago,
            sha=f"sha{i:04d}",
            insertions=np.random.randint(1, 50),
            deletions=np.random.randint(1, 20),
            languages=[".py"] if i % 3 != 0 else [".rs"],
            cross_refs=["other-repo"] if i % 5 == 0 else [],
        ))
    return commits


# ─── Token Encoding Tests ────────────────────────────────────────────

class TestTokenEncoding:
    """Test discrete token encoding for commit features."""

    def test_vocab_size(self):
        """Vocab size matches expected layout."""
        # 5 special + 24 repos + 24 hours + 7 days + 10 langs + 10 counts = 80
        assert VOCAB_SIZE == 80

    def test_special_tokens(self):
        assert len(SPECIAL_TOKENS) == 5
        assert "<PAD>" in SPECIAL_TOKENS
        assert "<BOS>" in SPECIAL_TOKENS

    def test_hour_to_bin(self):
        assert hour_to_bin(0) == HOUR_TOKEN_OFFSET
        assert hour_to_bin(23) == HOUR_TOKEN_OFFSET + 23
        assert hour_to_bin(24) == HOUR_TOKEN_OFFSET  # wraps

    def test_day_to_bin(self):
        assert day_to_bin(0) == DAY_TOKEN_OFFSET  # Monday
        assert day_to_bin(6) == DAY_TOKEN_OFFSET + 6  # Sunday

    def test_repo_to_token(self):
        assert repo_to_token("plato-training") == REPO_TOKEN_OFFSET
        assert repo_to_token("unknown") == REPO_TOKEN_OFFSET  # fallback to first

    def test_lang_to_token(self):
        assert lang_to_token(".py") == LANG_TOKEN_OFFSET
        assert lang_to_token(".rs") == LANG_TOKEN_OFFSET + 1

    def test_count_to_bin(self):
        assert count_to_bin(0) == COUNT_TOKEN_OFFSET
        assert count_to_bin(9) == COUNT_TOKEN_OFFSET + 9
        assert count_to_bin(15) == COUNT_TOKEN_OFFSET + 9  # capped at 9

    def test_encode_commit(self):
        tokens = encode_commit(
            repo="plato-training",
            hour=14,
            day=2,
            languages=[".py"],
            commit_count=3,
        )
        assert isinstance(tokens, list)
        assert len(tokens) >= 4  # repo + hour + day + count (at least)
        assert all(0 <= t < VOCAB_SIZE for t in tokens)

    def test_encode_commit_multiple_langs(self):
        tokens = encode_commit(
            repo="tensor-spline",
            hour=10,
            day=1,
            languages=[".py", ".rs", ".md"],
            commit_count=5,
        )
        # repo + hour + day + 3 langs + count = 7
        assert len(tokens) == 7

    def test_encode_commit_max_3_langs(self):
        tokens = encode_commit(
            repo="plato-types",
            hour=8,
            day=0,
            languages=[".py", ".rs", ".md", ".toml", ".json"],  # 5 langs
            commit_count=1,
        )
        # Should only keep 3 languages
        lang_tokens = tokens[3:-1]  # between day and count
        assert len(lang_tokens) <= 3

    def test_encode_hour_window(self):
        tokens = encode_hour_window("plato-training", 14, 2, [".py"], 3)
        assert tokens[0] == SPECIAL_TOKENS.index("<BOS>")
        assert tokens[-1] == SPECIAL_TOKENS.index("<EOS>")
        assert len(tokens) >= 6  # BOS + commit tokens + EOS


# ─── Data Preparation Tests ──────────────────────────────────────────

class TestDataPreparation:
    """Test commit aggregation and sequence creation."""

    def test_commits_to_hour_windows(self):
        commits = [
            make_commit(repo="plato-training", hours_ago=1, languages=[".py"]),
            make_commit(repo="plato-training", hours_ago=1.5, languages=[".rs"]),
            make_commit(repo="plato-types", hours_ago=2, languages=[".py"]),
        ]
        windows = commits_to_hour_windows(commits)
        assert len(windows) >= 2  # at least 2 distinct windows
        assert all(isinstance(w, HourWindow) for w in windows)

    def test_commits_aggregate_per_window(self):
        """Multiple commits in same hour for same repo aggregate."""
        base_time = datetime(2026, 5, 19, 14, 0, tzinfo=timezone.utc).timestamp()
        commits = [
            CommitPoint(
                sha=f"sha{i}", repo="plato-training", author="agent",
                timestamp=base_time + i * 60,  # within same hour
                message=f"commit {i}", files_changed=1,
                insertions=5, deletions=1, is_merge=False,
                languages=[".py"],
            )
            for i in range(5)
        ]
        windows = commits_to_hour_windows(commits)
        # All 5 in same hour -> 1 window
        repo_windows = [w for w in windows if w.repo == "plato-training"]
        assert len(repo_windows) >= 1
        # The matching window should have 5 commits
        matching = [w for w in repo_windows if w.hour == 14]
        assert len(matching) >= 1
        assert matching[0].commit_count == 5

    def test_windows_to_sequences(self):
        windows = [
            HourWindow("plato-training", h, 0, [".py"], h % 3, 10 * h, 5 * h)
            for h in range(24)
        ]
        inputs, targets = windows_to_sequences(windows, seq_len=4, predict_ahead=1)
        assert len(inputs) > 0
        assert len(inputs) == len(targets)
        assert all(isinstance(t, int) for t in targets)

    def test_commit_sequence_dataset(self):
        inputs = [[1, 2, 3], [4, 5, 6, 7, 8]]
        targets = [0, 1]
        ds = CommitSequenceDataset(inputs, targets, max_seq_len=64)
        assert len(ds) == 2
        seq, tgt = ds[0]
        assert seq.shape[0] == 64  # padded
        assert tgt.item() == 0


# ─── Model Tests ──────────────────────────────────────────────────────

class TestTinyGPT2:
    """Test the TinyGPT2 model architecture."""

    def test_config_defaults(self):
        config = TinyGPT2Config()
        assert config.vocab_size == VOCAB_SIZE
        assert config.n_layer == 2
        assert config.n_head == 4
        assert config.n_embd == 128
        assert config.num_classes == 10

    def test_model_creation(self):
        config = TinyGPT2Config()
        model = TinyGPT2(config)
        n_params = sum(p.numel() for p in model.parameters())
        # Should be roughly 500K
        assert 100_000 < n_params < 1_500_000, f"Unexpected param count: {n_params}"

    def test_forward_shape(self):
        config = TinyGPT2Config()
        model = TinyGPT2(config)
        x = torch.randint(0, config.vocab_size, (4, 32))  # batch=4, seq=32
        result = model(x)
        assert result["logits"].shape == (4, config.num_classes)

    def test_forward_with_labels(self):
        config = TinyGPT2Config()
        model = TinyGPT2(config)
        x = torch.randint(0, config.vocab_size, (4, 32))
        labels = torch.tensor([0, 1, 2, 3])
        result = model(x, labels=labels)
        assert "loss" in result
        assert result["loss"].item() > 0

    def test_forward_max_seq_len(self):
        config = TinyGPT2Config()
        model = TinyGPT2(config)
        x = torch.randint(0, config.vocab_size, (2, config.block_size))
        result = model(x)
        assert result["logits"].shape == (2, config.num_classes)

    def test_gradient_flow(self):
        config = TinyGPT2Config()
        model = TinyGPT2(config)
        x = torch.randint(0, config.vocab_size, (4, 16))
        labels = torch.tensor([0, 1, 2, 3])
        result = model(x, labels=labels)
        result["loss"].backward()
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# ─── Training Pipeline Tests ─────────────────────────────────────────

class TestTrainingPipeline:
    """Test the full training pipeline."""

    def test_train_synthetic(self):
        """Train on synthetic data (not enough real data)."""
        commits = make_many_commits(n=50, hours_span=24)
        result = train_fleet_gpt2(
            commits,
            epochs=3,
            batch_size=8,
            config=TinyGPT2Config(),
            verbose=False,
        )
        assert isinstance(result, TrainResult)
        assert len(result.train_loss_history) == 3
        assert result.params_count > 0
        assert 0 <= result.val_accuracy <= 1
        assert result.training_seconds > 0

    def test_train_loss_decreases(self):
        """Training loss should generally decrease."""
        commits = make_many_commits(n=80, hours_span=48)
        result = train_fleet_gpt2(
            commits,
            epochs=15,
            batch_size=8,
            config=TinyGPT2Config(),
            verbose=False,
        )
        # Loss should decrease over training
        assert result.train_loss_history[-1] < result.train_loss_history[0]

    def test_train_custom_config(self):
        """Training with custom config."""
        config = TinyGPT2Config(n_layer=1, n_embd=64, n_head=2)
        commits = make_many_commits(n=50, hours_span=24)
        result = train_fleet_gpt2(
            commits,
            epochs=2,
            config=config,
            verbose=False,
        )
        n_params = sum(p.numel() for p in result.model.parameters())
        assert n_params < 200_000  # smaller config = fewer params


# ─── Inference Tests ──────────────────────────────────────────────────

class TestInference:
    """Test prediction and inference."""

    def test_predict_next_hour(self):
        """predict_next_hour returns valid prediction dict."""
        config = TinyGPT2Config()
        model = TinyGPT2(config)
        result = predict_next_hour(
            model=model,
            repo="plato-training",
            current_hour=14,
            current_day=1,
            recent_languages=[".py"],
            recent_commit_counts=[0, 1, 2, 1, 0, 3, 1, 2],
        )
        assert "predicted_count_bin" in result
        assert "probabilities" in result
        assert "confidence" in result
        assert 0 <= result["predicted_count_bin"] <= 9
        assert len(result["probabilities"]) == 10
        assert abs(sum(result["probabilities"]) - 1.0) < 1e-5

    def test_inference_speed(self):
        """Inference should be fast (<100ms for a single prediction)."""
        config = TinyGPT2Config()
        model = TinyGPT2(config)
        model.eval()

        t0 = time.time()
        for _ in range(10):
            predict_next_hour(
                model=model,
                repo="plato-training",
                current_hour=14,
                current_day=1,
                recent_languages=[".py"],
                recent_commit_counts=[1, 2, 3],
            )
        elapsed = time.time() - t0
        avg_ms = elapsed / 10 * 1000
        assert avg_ms < 100, f"Inference too slow: {avg_ms:.1f}ms avg"


# ─── PLATO Tile Export Tests ──────────────────────────────────────────

class TestTileExport:
    """Test PLATO tile export and loading."""

    def test_export_tile(self, tmp_path):
        """Export a trained model as a PLATO tile."""
        commits = make_many_commits(n=50, hours_span=24)
        result = train_fleet_gpt2(
            commits,
            epochs=2,
            verbose=False,
        )
        tile = export_tile(
            result,
            store_dir=str(tmp_path / ".plato-training"),
        )
        assert tile is not None
        assert tile.tile_id.startswith("gpt2-fleet-predictor")
        assert tile.tile_type.value == "checkpoint"
        assert result.tile is tile

    def test_load_tile_model(self, tmp_path):
        """Export then load roundtrip."""
        store_dir = str(tmp_path / ".plato-training")

        commits = make_many_commits(n=50, hours_span=24)
        result = train_fleet_gpt2(
            commits,
            epochs=2,
            verbose=False,
        )
        tile = export_tile(result, store_dir=store_dir)

        # Load back
        model, loaded_tile = load_tile_model(
            tile.tile_id,
            store_dir=store_dir,
        )
        assert model is not None
        assert loaded_tile.tile_id == tile.tile_id

        # Verify model works after loading
        x = torch.randint(0, VOCAB_SIZE, (1, 16))
        output = model(x)
        assert output["logits"].shape == (1, 10)


# ─── Integration Test ─────────────────────────────────────────────────

class TestIntegration:
    """Full pipeline integration test."""

    def test_mine_train_predict(self):
        """Mine from local repos -> train -> predict -> export."""
        # Mine commits from real local repos
        from plato_training.fleet_miner import FleetMiner

        workspace = Path.home() / ".openclaw" / "workspace"
        miner = FleetMiner(clone_dir=str(workspace))

        all_commits = []
        for repo_name in ["plato-training", "plato-types", "tensor-spline", "plato-data"]:
            try:
                commits = miner.mine_repo(repo_name, max_commits=100)
                all_commits.extend(commits)
            except Exception:
                pass  # Skip repos that can't be mined

        # Ensure we have some data (even if repos are shallow)
        if len(all_commits) < 10:
            all_commits.extend(make_many_commits(n=60, hours_span=48))

        # Train
        result = train_fleet_gpt2(
            all_commits,
            epochs=5,
            batch_size=8,
            verbose=False,
        )

        assert result.val_accuracy >= 0  # at least runs without error
        assert result.params_count > 0

        # Predict
        pred = predict_next_hour(
            result.model,
            repo="plato-training",
            current_hour=14,
            current_day=1,
            recent_languages=[".py"],
            recent_commit_counts=[1, 2, 1, 0],
        )
        assert 0 <= pred["predicted_count_bin"] <= 9


import torch
