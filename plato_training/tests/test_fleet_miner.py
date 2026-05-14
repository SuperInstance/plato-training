"""
Tests for Fleet Git Miner — real data from SuperInstance history.
"""

import pytest
import subprocess
from pathlib import Path
from plato_training.fleet_miner import (
    FleetMiner, CommitPoint, SynergyEvent, FLEET_REPOS,
)


class TestCommitPoint:
    def test_size(self):
        c = CommitPoint(sha="abc", repo="test", author="fm", timestamp=1000.0,
                       message="fix", files_changed=3, insertions=100, deletions=30,
                       is_merge=False)
        assert c.size == 130
        assert c.net_lines == 70

    def test_to_dict(self):
        c = CommitPoint(sha="abc", repo="test", author="fm", timestamp=1000.0,
                       message="fix", files_changed=1, insertions=10, deletions=5,
                       is_merge=False, cross_refs=["other-repo"])
        d = c.to_dict()
        assert d["cross_refs"] == ["other-repo"]


class TestFleetMiner:
    def test_find_cross_refs(self):
        miner = FleetMiner()
        msg = "Fix bug in SuperInstance/plato-types and update SuperInstance/tensor-spline"
        refs = miner._find_cross_refs(msg)
        assert "plato-types" in refs
        assert "tensor-spline" in refs

    def test_no_self_ref(self):
        miner = FleetMiner()
        msg = "Update SuperInstance/plato-training docs"
        refs = miner._find_cross_refs(msg)
        assert "plato-training" in refs

    def test_mine_real_repo(self, tmp_path):
        """Mine a real repo (requires network)."""
        miner = FleetMiner(clone_dir=str(tmp_path / "mine"))
        commits = miner.mine_repo("plato-types", max_commits=50)
        
        # Should get at least some commits
        assert len(commits) >= 1
        assert all(isinstance(c, CommitPoint) for c in commits)
        assert all(c.repo == "plato-types" for c in commits)

    def test_find_synergies(self):
        miner = FleetMiner()
        commits = [
            CommitPoint(sha="a1", repo="plato-training", author="fm", timestamp=1000.0,
                       message="Fix types from SuperInstance/plato-types",
                       files_changed=1, insertions=5, deletions=2, is_merge=False,
                       cross_refs=["plato-types"]),
            CommitPoint(sha="a2", repo="plato-training", author="fm", timestamp=1001.0,
                       message="Use SuperInstance/tensor-spline for compression",
                       files_changed=1, insertions=10, deletions=3, is_merge=False,
                       cross_refs=["tensor-spline"]),
        ]
        synergies = miner.find_synergies(commits)
        assert len(synergies) == 2
        assert synergies[0].source_repo == "plato-training"
        assert synergies[0].target_repo == "plato-types"

    def test_aggregate_signals(self):
        miner = FleetMiner()
        commits = [
            CommitPoint(sha="a1", repo="test-repo", author="fm", timestamp=1000.0,
                       message="init", files_changed=5, insertions=200, deletions=0,
                       is_merge=False, languages=[".py"]),
            CommitPoint(sha="a2", repo="test-repo", author="oracle1", timestamp=2000.0,
                       message="update", files_changed=3, insertions=50, deletions=10,
                       is_merge=False, languages=[".py", ".rs"]),
        ]
        signals = miner.aggregate_signals(commits)
        assert len(signals) == 1
        assert signals[0].commits == 2
        assert signals[0].authors == 2
        assert ".py" in signals[0].languages


class TestFleetRepos:
    def test_known_repos_exist(self):
        assert "plato-training" in FLEET_REPOS
        assert "forgemaster" in FLEET_REPOS
        assert len(FLEET_REPOS) >= 20
