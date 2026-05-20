"""
BPE tokenizer trained on fleet commit messages.

Trains a Byte-Pair Encoding tokenizer on all commit messages from workspace repos,
then saves it for use in the Eisenstein encoder and other PLATO modules.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def collect_commit_messages(
    workspace: str,
    max_repos: int = 50,
    max_commits_per_repo: int = 500,
) -> List[str]:
    """Collect all commit messages from git repos in workspace."""
    messages = []
    for d in sorted(os.listdir(workspace)):
        repo = os.path.join(workspace, d)
        if not os.path.isdir(os.path.join(repo, ".git")):
            continue
        if len(messages) // max_commits_per_repo >= max_repos:
            break
        try:
            result = subprocess.run(
                ["git", "-C", repo, "log", f"-{max_commits_per_repo}", "--format=%s", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            msgs = [m.strip() for m in result.stdout.strip().split("\n") if m.strip()]
            messages.extend(msgs)
        except Exception:
            pass
    return messages


def train_fleet_bpe(
    workspace: str,
    vocab_size: int = 5000,
    save_path: str = "models/fleet-bpe.json",
    max_repos: int = 50,
    max_commits_per_repo: int = 500,
) -> Tokenizer:
    """Train a BPE tokenizer on all commit messages in workspace repos."""
    messages = collect_commit_messages(workspace, max_repos, max_commits_per_repo)

    # Add synthetic domain sentences for better coverage
    domain_sentences = _get_domain_sentences()
    messages.extend(domain_sentences)

    print(f"Training BPE on {len(messages)} commit messages + domain sentences")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]"],
        min_frequency=2,
    )
    tokenizer.train_from_iterator(messages, trainer)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    tokenizer.save(save_path)
    print(f"Saved tokenizer to {save_path}, vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer


def load_fleet_bpe(path: str = "models/fleet-bpe.json") -> Tokenizer:
    """Load a trained BPE tokenizer."""
    return Tokenizer.from_file(path)


def _get_domain_sentences() -> List[str]:
    """Synthetic domain sentences for fleet repos — ensures BPE covers fleet vocabulary."""
    return [
        "forgemaster constraint theory specialist agent",
        "forging proofs in computational constraint systems",
        "constraint satisfaction verification mathematical proofs",
        "quality gate stream validation pipeline",
        "stream processing quality assurance automated checks",
        "fleet murmur agent communication messaging",
        "murmur broadcast message routing fleet protocol",
        "fleet health monitor status check daemon",
        "health monitoring agent status fleet dashboard",
        "automerge pull request merge automation",
        "auto merge github workflow branch pr",
        "open shell terminal command execution agent",
        "shell bash terminal interface command line",
        "flux research analysis experiment data",
        "flux experimental research computational analysis",
        "constraint theory python proof solver",
        "constraint propagation algorithm satisfaction",
        "eisenstein integer lattice encoder hexagonal",
        "eisenstein lattice weight parameterization",
        "dodecet twelve tone encoding music",
        "dodecet music encoding twelve tone chromatic",
        "cocapn ai web interface dashboard",
        "plato training room tile lifecycle",
        "training micro models deployment hardware",
        "micro model training deploy hardware target",
        "neural plato deep learning room",
        "penrose memory tiling pattern store",
        "holonomy consensus distributed protocol",
        "distributed consensus holonomy protocol agreement",
        "flux lucid dreaming state machine",
        "polyformalism thinking formal logic reasoning",
        "ai writing generation text creative",
        "pbft rust consensus byzantine fault",
        "practical byzantine fault tolerance rust",
        "signal chain audio processing pipeline",
        "flux virtual machine bytecode runtime",
        "lucineer lucid engineer agent tool",
        "superinstance github organization fleet",
        "fleet github superinstance organization repos",
        "tensor spline weight compression lattice",
        "spline tensor weight lattice parameterization",
        "deployment model drift detection anomaly",
        "intent classification micro model npu quantize",
        "triplet margin contrastive loss training",
        "embedding cosine similarity vector retrieval",
        "tokenization vocabulary hash bpe subword",
        "lora adapter low rank fine tuning",
        "batch training throttle fleet aware",
        "hardware target deploy pipeline inference",
        "collective inference predict observe learn gap",
    ]
