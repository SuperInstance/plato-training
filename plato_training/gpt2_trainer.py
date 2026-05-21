"""
GPT-2 Fleet Trainer — train a tiny GPT-2 on fleet git commit data.

Takes CommitPoint data from FleetMiner, converts to token sequences,
trains a minimal GPT-2 (2 layers, 128 embed, 4 heads, ~500K params),
predicts next-hour activity per repo, and exports as a PLATO tile.

Pure PyTorch CPU. Inference <1ms.
"""



from __future__ import annotations
import math
__all__ = ['COUNT_TOKEN_OFFSET', 'CommitSequenceDataset', 'DAY_NAMES', 'DAY_TOKEN_OFFSET', 'HOUR_TOKEN_OFFSET', 'HourWindow', 'LANG_STOI', 'LANG_TOKEN_OFFSET', 'LANG_VOCAB', 'REPO_STOI', 'REPO_TOKEN_OFFSET', 'REPO_VOCAB', 'SPECIAL_TOKENS', 'TinyAttention', 'TinyBlock', 'TinyGPT2', 'TinyGPT2Config', 'TrainResult', 'VOCAB_SIZE', 'commits_to_hour_windows', 'count_to_bin', 'day_to_bin', 'encode_commit', 'encode_hour_window', 'evaluate_fleet_gpt2', 'export_tile', 'hour_to_bin', 'lang_to_token', 'load_tile_model', 'predict_next_hour', 'repo_to_token', 'train_fleet_gpt2', 'windows_to_sequences']
import time
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Local imports — reuse existing PLATO types
from .types import (
    TrainingTile, TileType, TileLifecycle, LamportClock,
    TrainingConfig, TrainingMetrics, content_hash,
)
from .store import LocalTileStore


# ─── Feature Vocabularies ─────────────────────────────────────────

# Repo vocabulary — fleet repos
REPO_VOCAB = [
    "plato-training", "plato-types", "tensor-spline", "plato-data",
    "plato-sdk", "fleet-memory", "folding-order", "holonomy-consensus",
    "constraint-flow-protocol", "constraint-inference", "intent-inference",
    "penrose-memory", "flux-lucid", "dodecet-encoder",
    "constraint-theory-py", "constraint-theory-core", "ct-demo",
    "neural-plato", "cocapn-ai-web", "openarm",
    "forgemaster", "oracle1-workspace", "plato-vessel-core",
    "casting-call",
]
REPO_STOI = {r: i for i, r in enumerate(REPO_VOCAB)}

# Day-of-week vocabulary (Mon=0, Sun=6)
DAY_NAMES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

# Language vocabulary — common file extensions
LANG_VOCAB = [".py", ".rs", ".ts", ".js", ".md", ".toml", ".yaml", ".json", ".txt", ".sh"]
LANG_STOI = {l: i for i, l in enumerate(LANG_VOCAB)}


# ─── Token Sequence Encoding ──────────────────────────────────────

# Token layout for each commit:
#   [REPO_TOKEN, HOUR_SIN, HOUR_COS, DAY_TOKEN, LANG_TOKENS..., COMMIT_COUNT_NORM]
# We discretize continuous features into bins for a discrete vocabulary.

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<SEP>", "<BOS>", "<EOS>"]
REPO_TOKEN_OFFSET = len(SPECIAL_TOKENS)        # 5..28 = repos (24 repos)
HOUR_TOKEN_OFFSET = REPO_TOKEN_OFFSET + len(REPO_VOCAB)  # 29..52 = 24 hour bins
DAY_TOKEN_OFFSET = HOUR_TOKEN_OFFSET + 24        # 53..59 = 7 days
LANG_TOKEN_OFFSET = DAY_TOKEN_OFFSET + 7         # 60..69 = 10 languages
COUNT_TOKEN_OFFSET = LANG_TOKEN_OFFSET + len(LANG_VOCAB)  # 70..79 = 10 count bins (0-9 commits)
VOCAB_SIZE = COUNT_TOKEN_OFFSET + 10             # 80 tokens total


def hour_to_bin(hour: int) -> int:
    """Map hour (0-23) to token."""
    return HOUR_TOKEN_OFFSET + hour % 24


def day_to_bin(day: int) -> int:
    """Map day of week (0=Mon, 6=Sun) to token."""
    return DAY_TOKEN_OFFSET + day % 7


def repo_to_token(repo: str) -> int:
    """Map repo name to token."""
    return REPO_TOKEN_OFFSET + REPO_STOI.get(repo, 0)


def lang_to_token(ext: str) -> int:
    """Map file extension to token."""
    return LANG_TOKEN_OFFSET + LANG_STOI.get(ext, LANG_STOI.get(".txt", 0))


def count_to_bin(count: int) -> int:
    """Map commit count to a binned token (0-9)."""
    return COUNT_TOKEN_OFFSET + min(count, 9)


def encode_commit(
    repo: str,
    hour: int,
    day: int,
    languages: List[str],
    commit_count: int,
) -> List[int]:
    """
    Encode a single hour-window commit summary as a token sequence.

    Layout: [REPO, HOUR, DAY, LANG1, LANG2, ..., COMMIT_COUNT]
    """
    tokens = [
        repo_to_token(repo),
        hour_to_bin(hour),
        day_to_bin(day),
    ]
    for lang in sorted(set(languages))[:3]:  # max 3 languages
        tokens.append(lang_to_token(lang))
    tokens.append(count_to_bin(commit_count))
    return tokens


def encode_hour_window(
    repo: str,
    hour: int,
    day: int,
    languages: List[str],
    commit_count: int,
) -> List[int]:
    """Encode an hour window as [BOS, commit_tokens..., EOS]."""
    return (
        [SPECIAL_TOKENS.index("<BOS>")]
        + encode_commit(repo, hour, day, languages, commit_count)
        + [SPECIAL_TOKENS.index("<EOS>")]
    )


# ─── Data Preparation ─────────────────────────────────────────────

@dataclass
class HourWindow:
    """Aggregated commit data for one repo in one hour."""
    repo: str
    hour: int          # 0-23
    day: int           # 0=Mon, 6=Sun
    languages: List[str]
    commit_count: int
    total_insertions: int
    total_deletions: int


def commits_to_hour_windows(commits: list) -> List[HourWindow]:
    """
    Aggregate a list of CommitPoint objects into hourly windows per repo.

    Each window captures: repo, hour, day, languages touched, commit count.
    """
    from .fleet_miner import CommitPoint

    windows: Dict[Tuple[str, int, int], HourWindow] = {}

    for cp in commits:
        dt = datetime.fromtimestamp(cp.timestamp, tz=timezone.utc)
        key = (cp.repo, dt.hour, dt.weekday())

        if key not in windows:
            windows[key] = HourWindow(
                repo=cp.repo,
                hour=dt.hour,
                day=dt.weekday(),
                languages=list(set(cp.languages)),
                commit_count=0,
                total_insertions=0,
                total_deletions=0,
            )
        w = windows[key]
        w.commit_count += 1
        w.total_insertions += cp.insertions
        w.total_deletions += cp.deletions
        w.languages = list(set(w.languages + cp.languages))

    return sorted(windows.values(), key=lambda w: (w.repo, w.day, w.hour))


def windows_to_sequences(
    windows: List[HourWindow],
    seq_len: int = 8,
    predict_ahead: int = 1,
) -> Tuple[List[List[int]], List[int]]:
    """
    Convert hour windows into (input_sequence, target) pairs.

    Input: seq_len consecutive hour windows for a repo.
    Target: commit_count bin for the next hour (predict_ahead=1).

    Returns (inputs, targets) where each input is a flat token sequence.
    """
    # Group by repo
    by_repo: Dict[str, List[HourWindow]] = {}
    for w in windows:
        by_repo.setdefault(w.repo, []).append(w)

    inputs: List[List[int]] = []
    targets: List[int] = []

    for repo, repo_windows in by_repo.items():
        # Sort by time (day, hour)
        repo_windows.sort(key=lambda w: (w.day, w.hour))

        # Create sliding window sequences
        for i in range(len(repo_windows) - seq_len - predict_ahead + 1):
            seq_windows = repo_windows[i : i + seq_len]
            target_window = repo_windows[i + seq_len + predict_ahead - 1]

            # Encode each window in the sequence
            tokens = []
            for w in seq_windows:
                tokens.extend(encode_hour_window(
                    w.repo, w.hour, w.day, w.languages, w.commit_count,
                ))

            inputs.append(tokens)
            # Target is the raw count bin (0-9), not the full token ID
            targets.append(min(target_window.commit_count, 9))

    return inputs, targets


class CommitSequenceDataset(Dataset):
    """Dataset of padded commit sequences for GPT-2 training."""

    def __init__(
        self,
        inputs: List[List[int]],
        targets: List[int],
        max_seq_len: int = 64,
    ):
        self.max_seq_len = max_seq_len
        pad_id = SPECIAL_TOKENS.index("<PAD>")

        self.sequences = []
        self.targets = []

        for seq, target in zip(inputs, targets):
            # Truncate or pad
            if len(seq) > max_seq_len:
                seq = seq[:max_seq_len]
            else:
                seq = seq + [pad_id] * (max_seq_len - len(seq))
            self.sequences.append(seq)
            self.targets.append(target)

        self.sequences = torch.tensor(self.sequences, dtype=torch.long)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(inputs={self.inputs!r}, targets={self.targets!r}, max_seq_len={self.max_seq_len!r})"



# ─── Tiny GPT-2 for Classification ───────────────────────────────

@dataclass
class TinyGPT2Config:
    """Minimal GPT-2 config for fleet prediction (~500K params)."""
    vocab_size: int = VOCAB_SIZE       # 80 tokens
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 64
    dropout: float = 0.1
    num_classes: int = 10              # 0-9 commit count bins

    def param_count(self) -> int:
        n = self.vocab_size * self.n_embd          # token embeddings
        n += self.block_size * self.n_embd          # position embeddings
        per_block = (
            3 * self.n_embd * self.n_embd           # QKV
            + self.n_embd * self.n_embd              # output proj
            + 2 * self.n_embd                        # LN
            + 4 * self.n_embd * self.n_embd          # MLP up
            + 4 * self.n_embd * self.n_embd          # MLP down (approx)
            + 2 * self.n_embd                        # MLP LN
        )
        n += per_block * self.n_layer
        n += self.n_embd                             # final LN
        n += self.n_embd * self.num_classes          # classifier head
        return n


class TinyAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: TinyGPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config!r})"



class TinyBlock(nn.Module):
    """Transformer block."""

    def __init__(self, config: TinyGPT2Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = TinyAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config!r})"



class TinyGPT2(nn.Module):
    """
    Minimal GPT-2 for sequence classification.

    Takes a sequence of commit tokens, processes through transformer,
    pools the last hidden state, and predicts commit count class.
    """

    def __init__(self, config: TinyGPT2Config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TinyBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.num_classes)

        # Initialize weights
        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"TinyGPT2: {n_params:,} parameters")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len) token indices
            labels: Optional (batch,) class labels

        Returns:
            Dict with 'logits' (batch, num_classes) and optionally 'loss'
        """
        B, T = input_ids.shape
        assert T <= self.config.block_size

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.wte(input_ids) + self.wpe(pos))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # Pool: mean of non-pad positions
        pad_id = SPECIAL_TOKENS.index("<PAD>")
        mask = (input_ids != pad_id).float().unsqueeze(-1)  # (B, T, 1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, n_embd)

        logits = self.head(x)  # (B, num_classes)

        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(logits, labels)
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config!r})"



# ─── Training ─────────────────────────────────────────────────────

@dataclass
class TrainResult:
    """Result of a training run."""
    model: TinyGPT2
    config: TinyGPT2Config
    train_loss_history: List[float]
    val_accuracy: float
    val_loss: float
    params_count: int
    training_seconds: float
    tile: Optional[TrainingTile] = None


def train_fleet_gpt2(
    commits: list,
    seq_len: int = 8,
    predict_ahead: int = 1,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    val_split: float = 0.2,
    config: Optional[TinyGPT2Config] = None,
    device: Optional[str] = None,
    verbose: bool = True,
) -> TrainResult:
    """
    End-to-end training pipeline.

    1. Convert commits to hour windows
    2. Create sequences (input=8 windows, target=next hour activity)
    3. Train TinyGPT2
    4. Evaluate accuracy
    5. Return results

    Args:
        commits: List of CommitPoint objects from FleetMiner
        seq_len: Number of hour windows per input sequence
        predict_ahead: Hours ahead to predict
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        val_split: Validation fraction
        config: Model config (defaults to ~500K params)
        device: torch device string
        verbose: Print progress

    Returns:
        TrainResult with model, metrics, and tile
    """
    if config is None:
        config = TinyGPT2Config()

    dev = torch.device(device or "cpu")

    # Step 1: Aggregate to hour windows
    windows = commits_to_hour_windows(commits)
    if verbose:
        print(f"Aggregated {len(commits)} commits -> {len(windows)} hour windows")

    # Step 2: Create sequences
    inputs, targets = windows_to_sequences(windows, seq_len=seq_len, predict_ahead=predict_ahead)

    if len(inputs) < 5:
        # Not enough data — generate synthetic windows for testing
        if verbose:
            print(f"Only {len(inputs)} sequences from real data, augmenting with synthetic")
        inputs, targets = _augment_synthetic(inputs, targets, windows, seq_len, n_augment=200)

    if verbose:
        print(f"Created {len(inputs)} sequences, {len(set(targets))} classes")

    # Step 3: Create dataset
    max_seq_tokens = seq_len * 10  # each window encodes to ~8-10 tokens
    dataset = CommitSequenceDataset(inputs, targets, max_seq_len=min(max_seq_tokens, config.block_size))

    # Split train/val
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Step 4: Build model
    model = TinyGPT2(config).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Model: {n_params:,} parameters")

    # Step 5: Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses = []
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for seq_batch, target_batch in train_loader:
            seq_batch = seq_batch.to(dev)
            target_batch = target_batch.to(dev)

            result = model(seq_batch, labels=target_batch)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    training_seconds = time.time() - t0

    # Step 6: Evaluate
    val_loss, val_accuracy = evaluate_fleet_gpt2(model, val_loader, dev)

    if verbose:
        print(f"\nTraining complete in {training_seconds:.1f}s")
        print(f"  Val loss: {val_loss:.4f}, Val accuracy: {val_accuracy:.2%}")
        print(f"  Params: {n_params:,}")

    return TrainResult(
        model=model,
        config=config,
        train_loss_history=train_losses,
        val_accuracy=val_accuracy,
        val_loss=val_loss,
        params_count=n_params,
        training_seconds=training_seconds,
    )


def evaluate_fleet_gpt2(
    model: TinyGPT2,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on validation set. Returns (val_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for seq_batch, target_batch in val_loader:
            seq_batch = seq_batch.to(device)
            target_batch = target_batch.to(device)

            result = model(seq_batch, labels=target_batch)
            total_loss += result["loss"].item() * len(target_batch)

            preds = result["logits"].argmax(dim=-1)
            correct += (preds == target_batch).sum().item()
            total += len(target_batch)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def _augment_synthetic(
    inputs: List[List[int]],
    targets: List[int],
    real_windows: List[HourWindow],
    seq_len: int,
    n_augment: int = 200,
) -> Tuple[List[List[int]], List[int]]:
    """Augment small datasets with synthetic but realistic sequences."""
    aug_inputs = list(inputs)
    aug_targets = list(targets)

    repos = list(set(w.repo for w in real_windows)) or REPO_VOCAB[:4]

    for _ in range(n_augment):
        repo = np.random.choice(repos)
        tokens = [SPECIAL_TOKENS.index("<BOS>")]
        # Generate seq_len windows
        for _ in range(seq_len):
            hour = np.random.randint(0, 24)
            day = np.random.randint(0, 7)
            langs = [np.random.choice(LANG_VOCAB)]
            count = np.random.choice([0, 0, 0, 1, 1, 2, 3])  # weighted toward low counts
            tokens.extend(encode_commit(repo, hour, day, langs, count))
        tokens.append(SPECIAL_TOKENS.index("<EOS>"))
        aug_inputs.append(tokens)
        # Target: raw count bin (0-9)
        target_count = np.random.choice([0, 0, 1, 1, 2, 3])
        aug_targets.append(target_count)

    return aug_inputs, aug_targets


# ─── Inference ────────────────────────────────────────────────────

@torch.no_grad()
def predict_next_hour(
    model: TinyGPT2,
    repo: str,
    current_hour: int,
    current_day: int,
    recent_languages: List[str],
    recent_commit_counts: List[int],
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Predict commit activity for the next hour.

    Args:
        model: Trained TinyGPT2
        repo: Repo name
        current_hour: Current hour (0-23)
        current_day: Current day of week (0=Mon)
        recent_languages: Recent file extensions touched
        recent_commit_counts: Last few hours' commit counts (most recent last)
        device: torch device

    Returns:
        Dict with predicted_count_bin, probabilities, confidence
    """
    dev = torch.device(device or "cpu")
    model.eval()
    model.to(dev)

    # Build input sequence from recent history
    tokens = [SPECIAL_TOKENS.index("<BOS>")]
    for count in recent_commit_counts[-8:]:  # last 8 hours
        tokens.extend(encode_commit(
            repo, (current_hour - len(recent_commit_counts[-8:]) + recent_commit_counts[-8:].index(count)) % 24,
            current_day,
            recent_languages,
            count,
        ))
    tokens.append(SPECIAL_TOKENS.index("<EOS>"))

    # Pad to model's block_size
    max_len = model.config.block_size
    if len(tokens) > max_len:
        tokens = tokens[-max_len:]
    pad_id = SPECIAL_TOKENS.index("<PAD>")
    tokens = tokens + [pad_id] * (max_len - len(tokens))

    input_ids = torch.tensor([tokens], dtype=torch.long, device=dev)

    result = model(input_ids)
    probs = F.softmax(result["logits"][0], dim=-1)
    predicted_bin = probs.argmax().item()

    return {
        "predicted_count_bin": predicted_bin,
        "predicted_count_range": f"{predicted_bin}-{predicted_bin}+",
        "probabilities": probs.tolist(),
        "confidence": probs[predicted_bin].item(),
    }


# ─── PLATO Tile Export ────────────────────────────────────────────

def export_tile(
    result: TrainResult,
    store_dir: str = ".plato-training",
    room_name: str = "gpt2-fleet-predictor",
) -> TrainingTile:
    """
    Export a trained model as a PLATO TrainingTile.

    Serializes the model state dict and creates a tile in the store.
    """
    store = LocalTileStore(store_dir)
    clock = LamportClock()
    lamport = clock.tick()

    # Serialize model
    state_dict = result.model.cpu().state_dict()
    weight_path = str(store.weights_dir / f"gpt2-fleet-L{lamport}.pt")
    torch.save(state_dict, weight_path)

    raw_bytes = Path(weight_path).read_bytes()
    c_hash = content_hash(raw_bytes)

    tile = TrainingTile(
        tile_id=f"gpt2-fleet-predictor-L{lamport:03d}",
        room=room_name,
        tile_type=TileType.CHECKPOINT,
        state=TileLifecycle.ACTIVE,
        lamport=lamport,
        name="gpt2-fleet-predictor",
        description=(
            f"Tiny GPT-2 fleet activity predictor: "
            f"{result.config.n_layer}L {result.config.n_head}H "
            f"{result.config.n_embd}D, {result.params_count:,} params, "
            f"val_acc={result.val_accuracy:.2%}"
        ),
        content_hash=c_hash,
        training_config=TrainingConfig(
            learning_rate=1e-3,
            epochs=len(result.train_loss_history),
            batch_size=16,
        ),
        metrics=TrainingMetrics(
            val_accuracy=result.val_accuracy,
            val_loss=result.val_loss,
            train_loss=result.train_loss_history[-1] if result.train_loss_history else 0,
            final_loss=result.train_loss_history[-1] if result.train_loss_history else 0,
            epochs_completed=len(result.train_loss_history),
            training_time_seconds=result.training_seconds,
            loss_curve=result.train_loss_history,
        ),
        source_room=room_name,
    )
    store.save(tile)
    result.tile = tile

    return tile


def load_tile_model(
    tile_id: str,
    store_dir: str = ".plato-training",
    config: Optional[TinyGPT2Config] = None,
    device: Optional[str] = None,
) -> Tuple[TinyGPT2, TrainingTile]:
    """Load a model from a PLATO tile."""
    store = LocalTileStore(store_dir)
    tile = store.load(tile_id)
    if tile is None:
        raise FileNotFoundError(f"Tile not found: {tile_id}")

    if config is None:
        config = TinyGPT2Config()

    model = TinyGPT2(config)
    weight_path = str(store.weights_dir / f"gpt2-fleet-L{tile.lamport}.pt")
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(torch.device(device or "cpu"))
    return model, tile
