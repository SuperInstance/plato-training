"""
Tests for commit_predictor.py — real micro model for fleet git data.
"""

import time
import math
import numpy as np
import pytest
from datetime import datetime, timezone

from plato_training.commit_predictor import (
    hour_features, day_features, repo_onehot, lang_features,
    commit_to_features, build_prediction_dataset,
    CommitPredictor, train_commit_predictor,
    REPO_VOCAB, LANG_VOCAB, PredictionSample,
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


def make_many_commits(repo="plato-training", n=50, hours_span=48):
    """Generate n commits spread over hours_span hours."""
    commits = []
    for i in range(n):
        hours_ago = hours_span * (1 - i / n)
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


# ─── Feature Engineering Tests ───────────────────────────────────────

class TestHourFeatures:
    def test_shape(self):
        h = hour_features(time.time())
        assert len(h) == 2

    def test_range(self):
        h = hour_features(time.time())
        assert -1.0 <= h[0] <= 1.0
        assert -1.0 <= h[1] <= 1.0

    def test_midnight(self):
        # 2024-01-01 00:00 UTC
        ts = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()
        h = hour_features(ts)
        assert abs(h[0]) < 0.01  # sin(0) ≈ 0
        assert abs(h[1] - 1.0) < 0.01  # cos(0) ≈ 1

    def test_noon(self):
        ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        h = hour_features(ts)
        assert abs(h[0]) < 0.01  # sin(π) ≈ 0
        assert abs(h[1] + 1.0) < 0.01  # cos(π) ≈ -1


class TestDayFeatures:
    def test_shape(self):
        d = day_features(time.time())
        assert len(d) == 2

    def test_range(self):
        d = day_features(time.time())
        assert -1.0 <= d[0] <= 1.0


class TestRepoOnehot:
    def test_known_repo(self):
        vec = repo_onehot("plato-training")
        assert sum(vec) == 1.0
        assert vec[REPO_VOCAB.index("plato-training")] == 1.0

    def test_unknown_repo(self):
        vec = repo_onehot("unknown-repo-xyz")
        assert sum(vec) == 0.0

    def test_length(self):
        assert len(repo_onehot("plato-training")) == len(REPO_VOCAB)


class TestLangFeatures:
    def test_python(self):
        vec = lang_features([".py"])
        assert vec[LANG_VOCAB.index(".py")] == 1.0
        assert sum(vec) == 1.0

    def test_multiple(self):
        vec = lang_features([".py", ".rs"])
        assert sum(vec) == 2.0

    def test_empty(self):
        vec = lang_features([])
        assert sum(vec) == 0.0


class TestCommitToFeatures:
    def test_shape(self):
        c = make_commit()
        features = commit_to_features(c)
        expected_dim = 2 + 2 + len(REPO_VOCAB) + len(LANG_VOCAB) + 4
        assert features.shape == (expected_dim,)

    def test_values_finite(self):
        c = make_commit()
        features = commit_to_features(c)
        assert np.all(np.isfinite(features))


# ─── Dataset Building Tests ──────────────────────────────────────────

class TestBuildDataset:
    def test_empty_commits(self):
        samples = build_prediction_dataset([], "plato-training")
        assert samples == []

    def test_wrong_repo(self):
        commits = [make_commit(repo="other-repo")]
        samples = build_prediction_dataset(commits, "plato-training")
        assert samples == []

    def test_generates_samples(self):
        commits = make_many_commits(repo="plato-training", n=50, hours_span=48)
        samples = build_prediction_dataset(
            commits, "plato-training",
            window_hours=4.0, step_hours=1.0, lookahead_hours=1.0,
        )
        assert len(samples) > 0
        assert all(isinstance(s, PredictionSample) for s in samples)

    def test_feature_dim_consistent(self):
        commits = make_many_commits(repo="plato-training", n=50, hours_span=48)
        samples = build_prediction_dataset(commits, "plato-training")
        dims = [s.features.shape[0] for s in samples]
        assert len(set(dims)) == 1  # All same dimension

    def test_labels_valid(self):
        commits = make_many_commits(repo="plato-training", n=50, hours_span=48)
        samples = build_prediction_dataset(commits, "plato-training")
        for s in samples:
            assert s.label_commit_next_hour in (0.0, 1.0)
            assert s.label_file_count >= 0.0
            assert s.label_cross_ref in (0.0, 1.0)


# ─── CommitPredictor Tests ──────────────────────────────────────────

class TestCommitPredictor:
    def test_init(self):
        model = CommitPredictor(input_dim=50, hidden_dim=16)
        assert model.W1.shape == (50, 16)
        assert model.W_commit.shape == (16, 1)

    def test_forward_shape(self):
        model = CommitPredictor(input_dim=50, hidden_dim=16)
        X = np.random.randn(10, 50).astype(np.float32)
        commit, files, crossref = model.forward(X)
        assert commit.shape == (10, 1)
        assert files.shape == (10, 1)
        assert crossref.shape == (10, 1)

    def test_forward_range(self):
        model = CommitPredictor(input_dim=50, hidden_dim=16)
        X = np.random.randn(10, 50).astype(np.float32)
        commit, files, crossref = model.forward(X)
        assert np.all(commit >= 0) and np.all(commit <= 1)
        assert np.all(files >= 0)
        assert np.all(crossref >= 0) and np.all(crossref <= 1)

    def test_train_step(self):
        model = CommitPredictor(input_dim=50, hidden_dim=16, lr=0.01)
        X = np.random.randn(20, 50).astype(np.float32)
        y_c = np.random.randint(0, 2, 20).astype(np.float32)
        y_f = np.random.rand(20).astype(np.float32) * 10
        y_x = np.random.randint(0, 2, 20).astype(np.float32)
        
        loss = model.train_step(X, y_c, y_f, y_x)
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    def test_fit_reduces_loss(self):
        np.random.seed(42)
        model = CommitPredictor(input_dim=20, hidden_dim=16, lr=0.05)
        X = np.random.randn(100, 20).astype(np.float32)
        # Make labels correlated with features for learnability
        y_c = (X[:, 0] > 0).astype(np.float32)
        y_f = (np.abs(X[:, 1]) * 3).astype(np.float32)  # small values
        y_x = (X[:, 2] > 0).astype(np.float32)
        
        losses = model.fit(X, y_c, y_f, y_x, epochs=100, verbose=False)
        assert len(losses) == 100
        # Loss should trend downward (compare last 10 avg vs first 10 avg)
        early = np.mean(losses[:10])
        late = np.mean(losses[-10:])
        assert late < early  # Loss should decrease

    def test_predict(self):
        model = CommitPredictor(input_dim=50, hidden_dim=16)
        X = np.random.randn(5, 50).astype(np.float32)
        preds = model.predict(X)
        assert "commit_prob" in preds
        assert "file_count" in preds
        assert "crossref_prob" in preds
        assert preds["commit_prob"].shape == (5,)

    def test_save_load(self, tmp_path):
        model = CommitPredictor(input_dim=30, hidden_dim=16)
        model.W1[0, 0] = 42.0  # Mark it
        
        path = str(tmp_path / "model.npz")
        model.save(path)
        
        loaded = CommitPredictor.load(path)
        assert loaded.W1[0, 0] == pytest.approx(42.0)
        assert loaded.input_dim == 30
        assert loaded.hidden_dim == 16


# ─── Integration Test ────────────────────────────────────────────────

class TestTrainPipeline:
    def test_train_on_synthetic_data(self):
        """End-to-end: generate data → train → evaluate."""
        np.random.seed(42)
        commits = make_many_commits(repo="plato-training", n=80, hours_span=72)
        
        model, metrics = train_commit_predictor(
            commits,
            repos=["plato-training"],
            window_hours=4.0,
            epochs=50,
            hidden_dim=16,
            lr=0.05,
        )
        
        assert metrics["samples"] > 0
        assert 0.0 <= metrics["commit_accuracy"] <= 1.0
        assert 0.0 <= metrics["file_activity_accuracy"] <= 1.0
        assert metrics["final_loss"] > 0.0

    def test_train_on_multiple_repos(self):
        np.random.seed(42)
        commits = []
        for repo in ["plato-training", "tensor-spline"]:
            commits.extend(make_many_commits(repo=repo, n=30, hours_span=48))
        
        model, metrics = train_commit_predictor(
            commits,
            repos=["plato-training", "tensor-spline"],
            epochs=30,
            hidden_dim=16,
        )
        
        assert metrics["repos"] == 2
        assert metrics["samples"] > 0
