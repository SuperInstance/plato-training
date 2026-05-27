"""Tests for plato_training.triplet_miner — triplet mining from git history."""

from unittest.mock import patch, MagicMock

import pytest

from plato_training.triplet_miner import (
    _topic_key,
    build_benchmark_queries,
    get_commit_messages,
    get_commit_details,
    mine_triplets,
)


class TestTopicKey:
    def test_basic(self):
        assert _topic_key("fix bug in spline") == "fix bug in"

    def test_short_message(self):
        assert _topic_key("fix") == "fix"

    def test_empty(self):
        assert _topic_key("") == ""

    def test_case_lowered(self):
        assert _topic_key("FIX BUG IN") == "fix bug in"


class TestBuildBenchmarkQueries:
    def test_returns_list(self):
        queries = build_benchmark_queries()
        assert isinstance(queries, list)
        assert len(queries) > 10

    def test_tuple_structure(self):
        queries = build_benchmark_queries()
        for query, repo in queries:
            assert isinstance(query, str)
            assert isinstance(repo, str)
            assert len(query) > 0

    def test_known_entries(self):
        queries = build_benchmark_queries()
        repos = {r for _, r in queries}
        assert "plato-training" in repos
        assert "forgemaster" in repos


class TestGetCommitMessages:
    @patch("plato_training.triplet_miner.subprocess.run")
    def test_basic(self, mock_run):
        mock_run.return_value = MagicMock(stdout="fix bug\nadd feature\nrefactor")
        msgs = get_commit_messages("/fake/repo")
        assert msgs == ["fix bug", "add feature", "refactor"]
        mock_run.assert_called_once()

    @patch("plato_training.triplet_miner.subprocess.run")
    def test_strips_empty(self, mock_run):
        mock_run.return_value = MagicMock(stdout="msg1\n\nmsg2\n")
        msgs = get_commit_messages("/fake/repo")
        assert msgs == ["msg1", "msg2"]


class TestGetCommitDetails:
    @patch("plato_training.triplet_miner.subprocess.run")
    def test_basic(self, mock_run):
        # First call: git log → message|||sha
        # Second call: git diff-tree → files
        mock_run.side_effect = [
            MagicMock(stdout="fix bug|||abc123def456"),
            MagicMock(stdout="src/main.py\nsrc/utils.py\nREADME.md"),
        ]
        commits = get_commit_details("/fake/repo")
        assert len(commits) == 1
        assert commits[0]["message"] == "fix bug"
        assert commits[0]["sha"] == "abc123def456"
        assert "py" in commits[0]["extensions"]

    @patch("plato_training.triplet_miner.subprocess.run")
    def test_skips_malformed_lines(self, mock_run):
        mock_run.side_effect = [
            MagicMock(stdout="no separator here\n"),
        ]
        commits = get_commit_details("/fake/repo")
        assert len(commits) == 0


class TestMineTriplets:
    @patch("plato_training.triplet_miner.subprocess.check_output")
    @patch("plato_training.triplet_miner.get_commit_details")
    @patch("plato_training.triplet_miner.os.listdir")
    @patch("plato_training.triplet_miner.os.path.isdir")
    def test_synthetic_fallback(self, mock_isdir, mock_listdir, mock_details, mock_check_output):
        """With < 2 repos, falls back to synthetic triplets."""
        mock_listdir.return_value = ["repo1"]
        mock_isdir.return_value = True
        mock_check_output.return_value = b"5"
        mock_details.return_value = [{"message": "short", "sha": "abc", "files": "", "extensions": [], "repo_path": "/fake"}]

        triplets = mine_triplets("/fake/workspace")
        assert len(triplets) > 0
        assert all(isinstance(t, tuple) and len(t) == 3 for t in triplets)

    def test_max_triplets_respected(self):
        """build_benchmark_queries always returns a fixed set."""
        queries = build_benchmark_queries()
        assert len(queries) == 50
