"""Tests for gpt2-fleet and collective CLI subcommands."""

import sys
import json
from pathlib import Path

import pytest


# ─── CLI Parsing ────────────────────────────────────────────────────


class TestGPT2FleetParse:
    """Test that gpt2-fleet subcommand parses correctly (no mining needed)."""

    def test_help_via_main(self):
        from plato_training.cli import main
        with pytest.raises(SystemExit) as exc:
            main(["gpt2-fleet", "--help"])
        assert exc.value.code == 0

    def test_default_args(self):
        from plato_training.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["gpt2-fleet"])
        assert args.room == "gpt2-fleet-predictor"
        assert args.org == "SuperInstance"
        assert args.epochs == 20
        assert args.layers == 2
        assert args.heads == 4
        assert args.embed == 128
        assert args.demo is False

    def test_custom_args(self):
        from plato_training.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "gpt2-fleet",
            "--room", "my-room",
            "--repos", "plato-training,forgemaster",
            "--epochs", "50",
            "--layers", "4",
            "--heads", "8",
            "--embed", "256",
            "--demo",
        ])
        assert args.room == "my-room"
        assert args.repos == "plato-training,forgemaster"
        assert args.epochs == 50
        assert args.layers == 4
        assert args.heads == 8
        assert args.embed == 256
        assert args.demo is True


class TestCollectiveParse:
    """Test that collective subcommand parses correctly."""

    def test_help_via_main(self):
        from plato_training.cli import main
        with pytest.raises(SystemExit) as exc:
            main(["collective", "--help"])
        assert exc.value.code == 0

    def test_once_args(self):
        from plato_training.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["collective", "once"])
        assert args.mode == "once"
        assert args.org == "SuperInstance"

    def test_run_args(self):
        from plato_training.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "collective", "run",
            "--interval", "10",
            "--cycles", "5",
        ])
        assert args.mode == "run"
        assert args.interval == 10.0
        assert args.cycles == 5

    def test_status_args(self):
        from plato_training.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["collective", "status"])
        assert args.mode == "status"

    def test_custom_repos(self):
        from plato_training.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "collective", "once",
            "--repos", "forgemaster,plato-training",
        ])
        assert args.repos == "forgemaster,plato-training"


# ─── Collective Status (no mining) ─────────────────────────────────


class TestCollectiveStatus:
    """Test collective status reads history file."""

    def test_status_no_history(self, tmp_path, capsys):
        """Status should succeed even with no history file."""
        from plato_training.cli import main
        hist = tmp_path / "no-history.json"
        # main calls sys.exit, so catch it
        try:
            main(["collective", "status", "--history", str(hist)])
        except SystemExit as e:
            assert e.code == 0

    def test_status_with_history(self, tmp_path, capsys):
        """Status should parse existing history."""
        from plato_training.cli import main
        hist = tmp_path / "history.json"
        hist.write_text(json.dumps({
            "cycle_count": 7,
            "cumulative_gap": 0.1234,
            "pending_predictions": [{"repo": "forgemaster"}],
            "last_cycle": 1700000000.0,
            "baseline_velocity": {"plato-training": 0.5, "forgemaster": 0.3},
        }))
        try:
            main(["collective", "status", "--history", str(hist)])
        except SystemExit as e:
            assert e.code == 0


# ─── Integration: GPT-2 training on synthetic data ─────────────────


class TestGPT2FleetIntegration:
    """Test GPT-2 fleet trainer with synthetic commits (no GitHub needed)."""

    def test_train_synthetic_commits(self):
        """Verify the training pipeline works with synthetic CommitPoints."""
        from plato_training.fleet_miner import CommitPoint
        from plato_training.gpt2_trainer import train_fleet_gpt2, TinyGPT2Config
        import time as _time

        # Generate synthetic commits across multiple repos
        commits = []
        repos = ["plato-training", "forgemaster", "tensor-spline"]
        for i in range(200):
            commits.append(CommitPoint(
                sha=f"abc{i:04d}",
                repo=repos[i % len(repos)],
                author=f"agent-{i % 3}",
                timestamp=_time.time() - (200 - i) * 3600,
                message=f"commit {i}",
                files_changed=(i % 5) + 1,
                insertions=(i % 20) * 10,
                deletions=(i % 10) * 5,
                is_merge=False,
                languages=[".py"] if i % 3 == 0 else [".rs", ".md"],
                cross_refs=["tensor-spline"] if i % 7 == 0 else [],
            ))

        config = TinyGPT2Config(n_layer=2, n_head=2, n_embd=64, block_size=32)
        result = train_fleet_gpt2(
            commits=commits,
            epochs=5,
            batch_size=8,
            config=config,
            verbose=False,
        )

        assert result.val_accuracy >= 0.0
        assert len(result.train_loss_history) == 5
        assert result.params_count > 0
        assert result.training_seconds > 0

    def test_predict_after_train(self):
        """Verify prediction works after training."""
        from plato_training.fleet_miner import CommitPoint
        from plato_training.gpt2_trainer import train_fleet_gpt2, predict_next_hour, TinyGPT2Config
        import time as _time

        commits = []
        for i in range(100):
            commits.append(CommitPoint(
                sha=f"pred{i:04d}",
                repo="forgemaster",
                author="fm",
                timestamp=_time.time() - (100 - i) * 3600,
                message=f"commit {i}",
                files_changed=(i % 4) + 1,
                insertions=10,
                deletions=5,
                is_merge=False,
                languages=[".py"],
                cross_refs=[],
            ))

        config = TinyGPT2Config(n_layer=1, n_head=2, n_embd=32, block_size=16)
        result = train_fleet_gpt2(commits=commits, epochs=3, config=config, verbose=False)

        pred = predict_next_hour(
            model=result.model,
            repo="forgemaster",
            current_hour=14,
            current_day=2,
            recent_languages=[".py"],
            recent_commit_counts=[1, 0, 2, 0, 1],
        )

        assert "predicted_count_bin" in pred
        assert "confidence" in pred
        assert 0 <= pred["confidence"] <= 1
        assert len(pred["probabilities"]) > 0


# ─── Collective Loop Unit ──────────────────────────────────────────


class TestCollectiveLoopUnit:
    """Test collective loop without GitHub (mocked)."""

    def test_cycle_result_fields(self):
        """Verify CycleResult has expected fields."""
        from plato_training.collective_loop import CycleResult
        result = CycleResult(
            cycle_id="test",
            timestamp=1700000000.0,
            repos_observed=3,
            commits_observed=50,
            predictions_made=5,
            predictions_correct=3,
            predictions_missed=1,
            gap_score=0.25,
            focus_items=[],
            top_synergies=[],
            velocity_by_repo={"forgemaster": 0.5},
            transfer_entropy=0.0,
            source_entropy=0.0,
        )
        assert result.cycle_id == "test"
        assert result.gap_score == 0.25
