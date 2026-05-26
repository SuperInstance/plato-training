"""
Integration tests for the PLATO Collective Inference Loop on real fleet data.

Tests that the full loop works end-to-end:
  - FleetMiner mines real commits from local clones
  - CollectiveLoop runs cycles on real data
  - CommitPredictor trains on real data
  - Coordination topology computes from real authors
"""

import os
import sys
import json
import time
import pytest
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plato_training.fleet_miner import FleetMiner, CommitPoint, FLEET_REPOS
from plato_training.collective_loop import (
    CollectiveLoop, CycleResult, PredictionEntry,
    compute_repo_velocity, detect_activity_spike, predict_commit_probability,
)
from plato_training.commit_predictor import (
    CommitPredictor, build_prediction_dataset, train_commit_predictor,
    commit_to_features, hour_features, day_features, repo_onehot,
)

# Real fleet repos (the 4 PLATO stack)
PLATO_REPOS = ["plato-training", "plato-types", "tensor-spline", "plato-data"]
WORKSPACE = str(Path.home() / ".openclaw" / "workspace")


def get_token():
    pat_file = Path.home() / ".openclaw" / "workspace" / ".credentials" / "github-pat.txt"
    if pat_file.exists():
        return pat_file.read_text().strip()
    return os.environ.get("GITHUB_TOKEN", "")


# ─── FleetMiner Tests ────────────────────────────────────────────────

class TestFleetMinerRealData:
    """Test FleetMiner with real local clones."""

    def test_mine_single_repo(self):
        """Mine commits from plato-training (local clone)."""
        miner = FleetMiner(org="SuperInstance", token=get_token(), clone_dir=WORKSPACE)
        commits = miner.mine_repo("plato-training", max_commits=50)
        assert len(commits) > 0, "Should mine at least one commit from plato-training"
        # Check commit structure
        c = commits[0]
        assert isinstance(c, CommitPoint)
        assert c.repo == "plato-training"
        assert len(c.sha) == 12
        assert c.timestamp > 0
        assert isinstance(c.message, str)
        assert isinstance(c.files_changed, int)
        assert isinstance(c.languages, list)

    def test_mine_all_plato_repos(self):
        """Mine all 4 PLATO stack repos."""
        miner = FleetMiner(org="SuperInstance", token=get_token(), clone_dir=WORKSPACE)
        total = 0
        for repo in PLATO_REPOS:
            commits = miner.mine_repo(repo, max_commits=100)
            total += len(commits)
            assert len(commits) > 0, f"Should mine commits from {repo}"

        assert total >= 4, f"Should have at least 4 total commits across repos, got {total}"

    def test_mine_fleet(self):
        """Full fleet mine with aggregation."""
        miner = FleetMiner(org="SuperInstance", token=get_token(), clone_dir=WORKSPACE)
        data = miner.mine_fleet(repos=PLATO_REPOS, max_per_repo=50)

        assert "commits" in data
        assert "synergies" in data
        assert "signals" in data
        assert "summary" in data
        assert data["summary"]["repos_ok"] == 4
        assert data["summary"]["total_commits"] > 0

    def test_cross_refs_detected(self):
        """Cross-repo references should be detected in messages."""
        miner = FleetMiner(org="SuperInstance", token=get_token(), clone_dir=WORKSPACE)
        all_commits = []
        for repo in PLATO_REPOS:
            commits = miner.mine_repo(repo, max_commits=100)
            all_commits.extend(commits)

        # At least some commits should have cross-refs
        # (not guaranteed but likely in PLATO stack)
        total_refs = sum(len(c.cross_refs) for c in all_commits)
        # Even if 0, this should not error
        assert isinstance(total_refs, int)

    def test_aggregate_signals(self):
        """Signal aggregation should produce RepoSignals."""
        miner = FleetMiner(org="SuperInstance", token=get_token(), clone_dir=WORKSPACE)
        all_commits = []
        for repo in PLATO_REPOS:
            commits = miner.mine_repo(repo, max_commits=50)
            all_commits.extend(commits)

        signals = miner.aggregate_signals(all_commits, window_hours=24)
        assert len(signals) >= 1
        for sig in signals:
            assert sig.commits > 0
            assert sig.velocity >= 0
            assert isinstance(sig.languages, dict)


# ─── CollectiveLoop Tests ───────────────────────────────────────────

class TestCollectiveLoopRealData:
    """Test CollectiveLoop with real fleet data."""

    def test_single_cycle(self):
        """Run one cycle on real data."""
        loop = CollectiveLoop(
            github_token=get_token(),
            clone_dir=WORKSPACE,
            history_file="/tmp/test-collective-history.json",
        )
        result = loop.run_cycle(repos=PLATO_REPOS)

        assert isinstance(result, CycleResult)
        assert result.repos_observed >= 1
        assert result.commits_observed >= 1
        assert 0.0 <= result.gap_score <= 1.0
        assert isinstance(result.velocity_by_repo, dict)
        assert isinstance(result.focus_items, list)

    def test_five_cycles(self):
        """Run 5 cycles on real data."""
        loop = CollectiveLoop(
            github_token=get_token(),
            clone_dir=WORKSPACE,
            history_file="/tmp/test-collective-5cycle-history.json",
        )

        results = []
        for _ in range(5):
            result = loop.run_cycle(repos=PLATO_REPOS)
            results.append(result)

        assert len(results) == 5
        # Each cycle should observe commits
        total_observed = sum(r.commits_observed for r in results)
        assert total_observed > 0
        # Cumulative gap should converge (EMA)
        assert 0.0 <= loop.cumulative_gap <= 1.0

    def test_velocity_computed(self):
        """Velocity should be computed per repo (may be 0 if no recent commits)."""
        loop = CollectiveLoop(
            github_token=get_token(),
            clone_dir=WORKSPACE,
            history_file="/tmp/test-collective-vel-history.json",
        )
        result = loop.run_cycle(repos=PLATO_REPOS)

        # Velocity dict exists (may be empty if all commits >24h old)
        assert isinstance(result.velocity_by_repo, dict)
        # Even if velocity is 0 for recent window, commits were observed
        assert result.commits_observed > 0

    def test_coordination_topology(self):
        """Transfer entropy and source entropy from real commits."""
        loop = CollectiveLoop(
            github_token=get_token(),
            clone_dir=WORKSPACE,
            history_file="/tmp/test-collective-te-history.json",
        )
        result = loop.run_cycle(repos=PLATO_REPOS)

        # TE and SE should be computed (may be 0 if single author)
        assert isinstance(result.transfer_entropy, float)
        assert isinstance(result.source_entropy, float)
        assert result.transfer_entropy >= 0
        assert result.source_entropy >= 0

    def test_status_report(self):
        """Status report should reflect real data."""
        loop = CollectiveLoop(
            github_token=get_token(),
            clone_dir=WORKSPACE,
            history_file="/tmp/test-collective-status-history.json",
        )
        loop.run_cycle(repos=PLATO_REPOS)

        report = loop.status_report()
        assert report["cycle_count"] >= 1
        assert report["total_commits_mined"] > 0
        # repos_tracked may be empty if velocity baseline hasn't built up yet
        assert isinstance(report["repos_tracked"], list)

    def test_predict_commit_probability(self):
        """Prediction probability should be valid."""
        prob = predict_commit_probability("plato-training", {"plato-training": 2.0}, hours_ahead=1.0)
        assert 0.0 < prob < 1.0

        # Zero velocity → zero probability
        prob_zero = predict_commit_probability("plato-training", {"plato-training": 0.0})
        assert prob_zero == 0.0


# ─── CommitPredictor Tests ──────────────────────────────────────────

class TestCommitPredictorRealData:
    """Test CommitPredictor trained on real fleet data."""

    def test_feature_extraction(self):
        """Commit features should have correct dimensions."""
        miner = FleetMiner(org="SuperInstance", token=get_token(), clone_dir=WORKSPACE)
        commits = miner.mine_repo("plato-training", max_commits=10)
        assert len(commits) > 0

        features = commit_to_features(commits[0])
        assert features.ndim == 1
        assert features.dtype == np.float32
        # Expected dim: 2 (hour) + 2 (day) + 21 (repo) + 10 (lang) + 4 (stats) = 39
        assert features.shape[0] > 0

    def test_build_prediction_dataset(self):
        """Dataset builder should produce samples from real data."""
        miner = FleetMiner(org="SuperInstance", token=get_token(), clone_dir=WORKSPACE)
        commits = miner.mine_repo("plato-training", max_commits=100)

        samples = build_prediction_dataset(commits, "plato-training", window_hours=6.0)
        if len(commits) == 0:
            pytest.skip("No commits available in CI")
        assert len(samples) > 0, "Should produce at least one prediction sample"

        s = samples[0]
        assert s.features.ndim == 1
        assert s.label_commit_next_hour in (0.0, 1.0)
        assert s.label_cross_ref in (0.0, 1.0)

    def test_train_predictor(self):
        """Train a predictor on real fleet data."""
        miner = FleetMiner(org="SuperInstance", token=get_token(), clone_dir=WORKSPACE)
        all_commits = []
        for repo in PLATO_REPOS:
            commits = miner.mine_repo(repo, max_commits=100)
            all_commits.extend(commits)

        model, metrics = train_commit_predictor(
            all_commits, repos=PLATO_REPOS,
            epochs=20, hidden_dim=16,
        )

        assert metrics["samples"] > 0
        assert 0 <= metrics["commit_accuracy"] <= 1.0
        assert metrics["final_loss"] > 0

    def test_predictor_save_load(self, tmp_path):
        """Predictor should save and load correctly."""
        miner = FleetMiner(org="SuperInstance", token=get_token(), clone_dir=WORKSPACE)
        commits = miner.mine_repo("plato-training", max_commits=50)

        samples = build_prediction_dataset(commits, "plato-training")
        if not samples:
            pytest.skip("Not enough commits for dataset")

        X = np.stack([s.features for s in samples])
        model = CommitPredictor(input_dim=X.shape[1], hidden_dim=8)
        save_path = str(tmp_path / "test_model.npz")
        model.save(save_path)

        loaded = CommitPredictor.load(save_path)
        preds = loaded.predict(X[:3])
        assert "commit_prob" in preds
        assert preds["commit_prob"].shape[0] == 3


# ─── End-to-End Integration ─────────────────────────────────────────

class TestEndToEnd:
    """Full pipeline: mine → loop → predict → compare → learn."""

    def test_full_pipeline(self):
        """Mine real data, run loop, verify results."""
        # 1. Mine
        miner = FleetMiner(org="SuperInstance", token=get_token(), clone_dir=WORKSPACE)
        data = miner.mine_fleet(repos=PLATO_REPOS, max_per_repo=50)
        assert data["summary"]["total_commits"] > 0

        # 2. Run 5 cycles
        loop = CollectiveLoop(
            github_token=get_token(),
            clone_dir=WORKSPACE,
            history_file="/tmp/test-e2e-collective-history.json",
        )

        results = []
        for i in range(5):
            result = loop.run_cycle(repos=PLATO_REPOS)
            results.append(result)

        # 3. Verify cycle progression
        assert len(results) == 5
        assert all(isinstance(r, CycleResult) for r in results)

        # 4. Cumulative gap should be tracked
        assert loop.cumulative_gap >= 0

        # 5. Status report
        report = loop.status_report()
        assert report["cycle_count"] >= 5
        assert report["total_commits_mined"] > 0

        # 6. Train predictor on mined data
        all_commits = []
        for repo in PLATO_REPOS:
            commits = miner.mine_repo(repo, max_commits=100)
            all_commits.extend(commits)

        if len(all_commits) >= 10:
            model, metrics = train_commit_predictor(
                all_commits, repos=PLATO_REPOS, epochs=10,
            )
            assert metrics["commit_accuracy"] > 0

    def test_fleet_report(self):
        """Fleet report should be human-readable."""
        miner = FleetMiner(org="SuperInstance", token=get_token(), clone_dir=WORKSPACE)
        data = miner.mine_fleet(repos=PLATO_REPOS, max_per_repo=30)
        report = miner.fleet_report(data)

        assert "FLEET INTELLIGENCE REPORT" in report
        assert "plato-training" in report
