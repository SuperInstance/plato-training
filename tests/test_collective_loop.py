"""
Tests for collective_loop.py — predict → observe → compare → learn cycle.
"""

import json
import time
import math
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from plato_training.collective_loop import (
    CollectiveLoop, CycleResult, PredictionEntry,
    compute_repo_velocity, detect_activity_spike,
    predict_commit_probability,
)
from plato_training.fleet_miner import CommitPoint


# ─── Fixtures ────────────────────────────────────────────────────────

def make_commit(repo="test-repo", hours_ago=0, author="agent", sha="abc123",
                insertions=10, deletions=5, cross_refs=None):
    """Create a CommitPoint for testing."""
    return CommitPoint(
        sha=sha,
        repo=repo,
        author=author,
        timestamp=time.time() - hours_ago * 3600,
        message="test commit",
        files_changed=2,
        insertions=insertions,
        deletions=deletions,
        is_merge=False,
        cross_refs=cross_refs or [],
        languages=[".py"],
    )


@pytest.fixture
def tmp_history(tmp_path):
    """Temporary history file."""
    return str(tmp_path / "collective-history.json")


@pytest.fixture
def loop(tmp_history):
    """CollectiveLoop with no real GitHub access."""
    with patch("plato_training.collective_loop.FleetMiner"):
        lp = CollectiveLoop(
            github_token=None,
            plato_url="http://localhost:8847",
            history_file=tmp_history,
        )
    return lp


# ─── Activity Pattern Tests ─────────────────────────────────────────

class TestRepoVelocity:
    def test_empty_commits(self):
        assert compute_repo_velocity([]) == {}

    def test_recent_commits(self):
        commits = [
            make_commit(repo="a", hours_ago=1),
            make_commit(repo="a", hours_ago=2),
            make_commit(repo="b", hours_ago=3),
        ]
        vel = compute_repo_velocity(commits, hours=24.0)
        assert vel["a"] == pytest.approx(2.0 / 24.0, abs=0.01)
        assert vel["b"] == pytest.approx(1.0 / 24.0, abs=0.01)

    def test_old_commits_excluded(self):
        commits = [make_commit(repo="a", hours_ago=48)]
        vel = compute_repo_velocity(commits, hours=24.0)
        assert vel == {}


class TestActivitySpike:
    def test_no_spike(self):
        spikes = detect_activity_spike(
            {"a": 1.0}, {"a": 1.0}, threshold=2.0,
        )
        assert spikes == []

    def test_spike_detected(self):
        spikes = detect_activity_spike(
            {"a": 5.0, "b": 0.5}, {"a": 1.0, "b": 0.5}, threshold=2.0,
        )
        assert "a" in spikes
        assert "b" not in spikes

    def test_zero_baseline(self):
        # Repo with no baseline — should not spike (base defaults to 0.1)
        spikes = detect_activity_spike(
            {"new-repo": 1.0}, {}, threshold=2.0,
        )
        # 1.0 > 0.1 * 2.0 = 0.2 → spike
        assert "new-repo" in spikes


class TestPredictProbability:
    def test_zero_velocity(self):
        assert predict_commit_probability("repo", {}) == 0.0

    def test_high_velocity(self):
        vel = {"repo": 10.0}
        prob = predict_commit_probability("repo", vel, hours_ahead=1.0)
        assert 0.9 < prob < 1.0  # Very high velocity → near-certain

    def test_low_velocity(self):
        vel = {"repo": 0.1}
        prob = predict_commit_probability("repo", vel, hours_ahead=1.0)
        assert 0.0 < prob < 0.2  # Low velocity → unlikely

    def test_decay_effect(self):
        vel = {"repo": 1.0}
        prob_low = predict_commit_probability("repo", vel, decay=0.1)
        prob_high = predict_commit_probability("repo", vel, decay=1.0)
        assert prob_high > prob_low


# ─── PredictionEntry Tests ──────────────────────────────────────────

class TestPredictionEntry:
    def test_creation(self):
        pred = PredictionEntry(
            prediction_id="abc123",
            repo="test-repo",
            predicted_at=time.time(),
            predicted_activity="commit",
            confidence=0.8,
            time_window_hours=1.0,
            context="test",
        )
        assert not pred.observed
        assert not pred.correct

    def test_to_dict(self):
        pred = PredictionEntry(
            prediction_id="abc",
            repo="r",
            predicted_at=0.0,
            predicted_activity="commit",
            confidence=0.5,
            time_window_hours=1.0,
            context="c",
        )
        d = pred.__dict__
        assert "prediction_id" in d
        assert d["observed"] is False


# ─── CycleResult Tests ──────────────────────────────────────────────

class TestCycleResult:
    def test_to_dict(self):
        result = CycleResult(
            cycle_id="test",
            timestamp=time.time(),
            repos_observed=5,
            commits_observed=100,
            predictions_made=10,
            predictions_correct=8,
            predictions_missed=2,
            gap_score=0.2,
            focus_items=[],
            top_synergies=[],
            velocity_by_repo={"a": 1.0},
        )
        d = result.to_dict()
        assert d["gap_score"] == 0.2
        assert d["repos_observed"] == 5


# ─── CollectiveLoop Tests ───────────────────────────────────────────

class TestCollectiveLoopInit:
    def test_init_no_history(self, loop):
        assert loop.cycle_count == 0
        assert loop.cumulative_gap == 0.0
        assert loop.all_commits == []

    def test_load_history(self, tmp_history):
        # Write fake history
        Path(tmp_history).write_text(json.dumps({
            "baseline_velocity": {"repo-a": 0.5},
            "cycle_count": 5,
            "cumulative_gap": 0.3,
            "pending_predictions": [],
        }))
        with patch("plato_training.collective_loop.FleetMiner"):
            lp = CollectiveLoop(history_file=tmp_history)
        assert lp.cycle_count == 5
        assert lp.baseline_velocity["repo-a"] == 0.5

    def test_save_and_load(self, loop, tmp_history):
        loop.cycle_count = 3
        loop.baseline_velocity = {"test": 1.5}
        loop._save_history()

        with patch("plato_training.collective_loop.FleetMiner"):
            lp2 = CollectiveLoop(history_file=tmp_history)
        assert lp2.cycle_count == 3
        assert lp2.baseline_velocity["test"] == 1.5


class TestCollectiveLoopPredict:
    def test_predict_with_velocity(self, loop):
        loop.baseline_velocity = {"repo-a": 0.5}
        velocity = {"repo-a": 2.0, "repo-b": 0.01}
        preds = loop._predict(velocity)
        
        # repo-a should get a prediction (high velocity)
        repo_a_preds = [p for p in preds if p.repo == "repo-a"]
        assert len(repo_a_preds) >= 1
        assert repo_a_preds[0].confidence > 0.3

    def test_predict_skips_low_velocity(self, loop):
        velocity = {"quiet-repo": 0.001}
        preds = loop._predict(velocity)
        # Probability < 0.3, so no prediction generated
        assert len(preds) == 0

    def test_predict_detects_spikes(self, loop):
        loop.baseline_velocity = {"spike-repo": 0.5}
        velocity = {"spike-repo": 5.0}
        preds = loop._predict(velocity)
        
        spike_preds = [p for p in preds if p.predicted_activity == "spike"]
        assert len(spike_preds) == 1
        assert spike_preds[0].confidence == 0.8


class TestCollectiveLoopCompare:
    def test_compare_correct(self, loop):
        now = time.time()
        preds = [
            PredictionEntry(
                prediction_id="p1",
                repo="repo-a",
                predicted_at=now - 7200,  # 2 hours ago — window expired
                predicted_activity="commit",
                confidence=0.8,
                time_window_hours=1.0,  # 1hr window → expired
                context="test",
            )
        ]
        observed = [make_commit(repo="repo-a", hours_ago=0.5)]
        
        correct, missed, resolved = loop._compare(preds, observed, window_hours=3.0)
        assert correct == 1
        assert missed == 0

    def test_compare_missed(self, loop):
        now = time.time()
        preds = [
            PredictionEntry(
                prediction_id="p1",
                repo="repo-a",
                predicted_at=now - 3600,
                predicted_activity="commit",
                confidence=0.8,
                time_window_hours=0.5,  # Window already expired
                context="test",
            )
        ]
        observed = [make_commit(repo="repo-b", hours_ago=0.5)]  # Different repo
        
        correct, missed, resolved = loop._compare(preds, observed, window_hours=2.0)
        assert missed == 1
        assert correct == 0


class TestCollectiveLoopLearn:
    def test_gap_score_zero(self, loop):
        # Set baseline close to current to avoid velocity shift detection
        loop.baseline_velocity = {"a": 1.0}
        loop.predictions = []
        gap, items = loop._learn(10, 0, {"a": 1.0})
        assert gap == 0.0
        # Only missed_prediction items (none), velocity shifts should be empty
        missed_items = [i for i in items if i["type"] == "missed_prediction"]
        assert missed_items == []

    def test_gap_score_full(self, loop):
        gap, items = loop._learn(0, 10, {"a": 1.0})
        assert gap == 1.0

    def test_baseline_update(self, loop):
        loop.baseline_velocity = {"a": 1.0}
        loop._learn(5, 5, {"a": 2.0})
        # EMA: 0.8 * 1.0 + 0.2 * 2.0 = 1.2
        assert loop.baseline_velocity["a"] == pytest.approx(1.2, abs=0.01)

    def test_focus_items_from_misses(self, loop):
        loop.predictions = [
            PredictionEntry(
                prediction_id="p1", repo="a", predicted_at=time.time() - 7200,
                predicted_activity="commit", confidence=0.8,
                time_window_hours=0.1, context="test",
                observed=True, correct=False, observed_at=time.time(),
            ),
        ]
        gap, items = loop._learn(0, 1, {})
        assert len(items) >= 1
        assert items[0]["type"] == "missed_prediction"

    def test_velocity_shift_detection(self, loop):
        loop.baseline_velocity = {"a": 1.0, "b": 2.0}
        loop.predictions = []
        gap, items = loop._learn(5, 0, {"a": 5.0, "b": 2.1})
        shifts = [i for i in items if i["type"] == "velocity_shift"]
        assert len(shifts) >= 1
        # repo-a shifted 5x → should be flagged
        shift_repos = [s["repo"] for s in shifts]
        assert "a" in shift_repos


class TestCollectiveLoopCycle:
    def test_single_cycle(self, loop):
        # Mock _observe to return fake commits
        loop._observe = lambda repos=None: [
            make_commit(repo="a", hours_ago=1),
            make_commit(repo="b", hours_ago=2),
        ]
        
        result = loop.run_cycle()
        assert result.commits_observed == 2
        assert result.repos_observed == 2
        assert result.cycle_id
        assert result.timestamp > 0
        assert 0.0 <= result.gap_score <= 1.0

    def test_cycle_increments(self, loop):
        loop._observe = lambda repos=None: []
        loop.run_cycle()
        assert loop.cycle_count == 1
        loop.run_cycle()
        assert loop.cycle_count == 2

    def test_cycle_saves_history(self, loop, tmp_history):
        loop._observe = lambda repos=None: [make_commit()]
        loop.run_cycle()
        
        data = json.loads(Path(tmp_history).read_text())
        assert data["cycle_count"] == 1


class TestStatusReport:
    def test_report(self, loop):
        loop.cycle_count = 10
        loop.baseline_velocity = {"a": 1.0, "b": 0.5}
        loop.all_commits = [make_commit()]
        
        report = loop.status_report()
        assert report["agent"] == "forgemaster"
        assert report["cycle_count"] == 10
        assert "a" in report["velocity"]


# ─── Deduplication Test ─────────────────────────────────────────────

class TestDedup:
    def test_dedup_commits(self, loop):
        c1 = make_commit(repo="a", sha="same123")
        c2 = make_commit(repo="a", sha="same123")
        
        loop._observe = lambda repos=None: [c1, c2]
        loop.run_cycle()
        
        # Should deduplicate
        keys = set(f"{c.sha}:{c.repo}" for c in loop.all_commits)
        assert len(keys) == len(loop.all_commits)
