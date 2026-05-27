"""Tests for plato_training.fleet_tokenizer — BPE tokenizer for fleet commits."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from plato_training.fleet_tokenizer import (
    _get_domain_sentences,
    collect_commit_messages,
    load_fleet_bpe,
    train_fleet_bpe,
)


class TestGetDomainSentences:
    def test_returns_list(self):
        sentences = _get_domain_sentences()
        assert isinstance(sentences, list)
        assert len(sentences) > 10

    def test_entries_are_strings(self):
        sentences = _get_domain_sentences()
        for s in sentences:
            assert isinstance(s, str)
            assert len(s) > 0

    def test_covers_key_terms(self):
        sentences = _get_domain_sentences()
        text = " ".join(sentences)
        assert "plato" in text.lower()
        assert "fleet" in text.lower()


class TestCollectCommitMessages:
    @patch("plato_training.fleet_tokenizer.subprocess.run")
    def test_collects_from_repo(self, mock_run):
        mock_run.return_value = type("R", (), {"stdout": "fix bug\nadd feature\n"})()
        # We need to mock os.listdir and os.path.isdir
        with patch("os.listdir", return_value=["repo1"]):
            with patch("os.path.isdir", side_effect=lambda p: True):
                with patch("os.path.join", lambda *a: "/".join(a)):
                    msgs = collect_commit_messages("/ws")
                    assert len(msgs) == 2

    @patch("plato_training.fleet_tokenizer.subprocess.run")
    def test_handles_exception(self, mock_run):
        mock_run.side_effect = Exception("git failed")
        with patch("os.listdir", return_value=["repo1"]):
            with patch("os.path.isdir", return_value=True):
                with patch("os.path.join", lambda *a: "/".join(a)):
                    msgs = collect_commit_messages("/ws")
                    assert msgs == []


class TestTrainFleetBPE:
    def test_train_and_load(self):
        """Train a small BPE tokenizer and verify it loads."""
        from tokenizers import Tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "bpe.json")
            tokenizer = train_fleet_bpe(
                workspace=tmpdir,
                vocab_size=100,
                save_path=save_path,
            )
            assert isinstance(tokenizer, Tokenizer)
            assert tokenizer.get_vocab_size() > 10

            # Load it back
            loaded = load_fleet_bpe(save_path)
            assert loaded.get_vocab_size() == tokenizer.get_vocab_size()

            # Can encode
            encoding = loaded.encode("fleet status check")
            assert len(encoding.ids) > 0


class TestLoadFleetBPE:
    def test_load_missing_raises(self):
        with pytest.raises(Exception):
            load_fleet_bpe("/nonexistent/path.json")
