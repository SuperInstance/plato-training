"""
Commit Pattern Predictor — real micro model trained on fleet git data.

Uses commit features (hour-of-day, day-of-week, repo, author, file extensions)
to predict:
  1. Whether a repo will have a commit in the next hour
  2. How many files will be changed
  3. Whether the commit will cross-reference another repo

Architecture: small dense network (deploy_micro-compatible).
Training data: fleet miner output.
"""


__all__ = ['AUTHOR_VOCAB_SIZE', 'CommitPredictor', 'LANG_VOCAB', 'PredictionSample', 'REPO_VOCAB', 'build_prediction_dataset', 'commit_to_features', 'day_features', 'hour_features', 'lang_features', 'repo_onehot', 'train_commit_predictor']

import json
import time
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

from .fleet_miner import FleetMiner, CommitPoint


# ─── Feature Engineering ─────────────────────────────────────────────

REPO_VOCAB = [
    "plato-training", "plato-types", "tensor-spline", "plato-data",
    "constraint-theory-core", "constraint-theory-py", "cocapn-ai-web",
    "forgemaster", "plato-vessel-core", "casting-call",
    "constraint-inference", "intent-inference", "holonomy-consensus",
    "flux-lucid", "dodecet-encoder", "neural-plato",
    "openarm", "plato-model-ocean", "plato-escalation-gate",
    "plato-room-intelligence", "spectral-conservation",
]

LANG_VOCAB = [".py", ".rs", ".js", ".ts", ".md", ".toml", ".json", ".c", ".cpp", ".h"]

AUTHOR_VOCAB_SIZE = 20  # top 20 authors by frequency


def hour_features(timestamp: float) -> List[float]:
    """Cyclical hour-of-day encoding (sin/cos)."""
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    h = dt.hour
    return [math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24)]


def day_features(timestamp: float) -> List[float]:
    """Cyclical day-of-week encoding."""
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    d = dt.weekday()
    return [math.sin(2 * math.pi * d / 7), math.cos(2 * math.pi * d / 7)]


def repo_onehot(repo: str) -> List[float]:
    """One-hot encoding for repo name."""
    vec = [0.0] * len(REPO_VOCAB)
    if repo in REPO_VOCAB:
        vec[REPO_VOCAB.index(repo)] = 1.0
    return vec


def lang_features(languages: List[str]) -> List[float]:
    """Multi-hot encoding for file languages changed."""
    vec = [0.0] * len(LANG_VOCAB)
    for lang in languages:
        ext = lang if lang.startswith(".") else f".{lang}"
        if ext in LANG_VOCAB:
            vec[LANG_VOCAB.index(ext)] = 1.0
    return vec


def commit_to_features(commit: CommitPoint) -> np.ndarray:
    """Convert a commit to a feature vector."""
    parts = []
    parts.extend(hour_features(commit.timestamp))
    parts.extend(day_features(commit.timestamp))
    parts.extend(repo_onehot(commit.repo))
    parts.extend(lang_features(commit.languages))
    parts.append(float(commit.files_changed))
    parts.append(float(commit.insertions))
    parts.append(float(commit.deletions))
    parts.append(float(len(commit.cross_refs) > 0))
    return np.array(parts, dtype=np.float32)


# ─── Windowed Dataset ────────────────────────────────────────────────

@dataclass
class PredictionSample:
    """One training sample: predict next-hour activity from recent window."""
    features: np.ndarray           # aggregated features from past N hours
    label_commit_next_hour: float  # 1.0 if commit in next hour, 0.0 if not
    label_file_count: float        # number of files changed in next hour
    label_cross_ref: float         # 1.0 if cross-ref in next hour
    repo: str
    window_start: float
    window_end: float


def build_prediction_dataset(
    commits: List[CommitPoint],
    repo: str,
    window_hours: float = 6.0,
    step_hours: float = 1.0,
    lookahead_hours: float = 1.0,
) -> List[PredictionSample]:
    """
    Build a dataset of (past window → next hour prediction) samples.
    
    For each time step:
      - Features: aggregated commit activity in [t-window, t]
      - Labels: what happens in [t, t+lookahead]
    """
    if not commits:
        return []
    
    repo_commits = sorted(
        [c for c in commits if c.repo == repo],
        key=lambda c: c.timestamp,
    )
    
    if not repo_commits:
        return []
    
    min_ts = repo_commits[0].timestamp
    max_ts = repo_commits[-1].timestamp
    window_s = window_hours * 3600
    step_s = step_hours * 3600
    lookahead_s = lookahead_hours * 3600
    
    samples = []
    t = min_ts + window_s  # Start after first full window
    
    while t + lookahead_s <= max_ts:
        # Window commits
        window_commits = [
            c for c in repo_commits
            if t - window_s <= c.timestamp < t
        ]
        
        # Lookahead commits
        lookahead_commits = [
            c for c in repo_commits
            if t <= c.timestamp < t + lookahead_s
        ]
        
        # Aggregate window features
        if window_commits:
            agg = np.zeros(len(REPO_VOCAB) + len(LANG_VOCAB) + 7, dtype=np.float32)
            # Count
            agg[0] = len(window_commits)
            # Files
            agg[1] = sum(c.files_changed for c in window_commits)
            # Insertions
            agg[2] = sum(c.insertions for c in window_commits)
            # Deletions
            agg[3] = sum(c.deletions for c in window_commits)
            # Cross refs
            agg[4] = sum(len(c.cross_refs) for c in window_commits)
            # Avg hour (cyclical)
            hours = [datetime.fromtimestamp(c.timestamp, tz=timezone.utc).hour for c in window_commits]
            agg[5] = math.sin(2 * math.pi * np.mean(hours) / 24)
            agg[6] = math.cos(2 * math.pi * np.mean(hours) / 24)
            
            # Time features
            tf = np.array(hour_features(t) + day_features(t), dtype=np.float32)
            features = np.concatenate([tf, agg])
        else:
            tf = np.array(hour_features(t) + day_features(t), dtype=np.float32)
            features = np.concatenate([tf, np.zeros(len(REPO_VOCAB) + len(LANG_VOCAB) + 7, dtype=np.float32)])
        
        # Labels
        has_commit = 1.0 if lookahead_commits else 0.0
        file_count = float(sum(c.files_changed for c in lookahead_commits))
        has_crossref = 1.0 if any(c.cross_refs for c in lookahead_commits) else 0.0
        
        samples.append(PredictionSample(
            features=features,
            label_commit_next_hour=has_commit,
            label_file_count=file_count,
            label_cross_ref=has_crossref,
            repo=repo,
            window_start=t - window_s,
            window_end=t,
        ))
        
        t += step_s
    
    return samples


# ─── Simple Dense Predictor ──────────────────────────────────────────

class CommitPredictor:
    """
    Small dense network for commit prediction.
    
    No PyTorch dependency — pure numpy for maximum portability.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 32, lr: float = 0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        
        # Xavier initialization
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        
        # Three output heads
        self.W_commit = np.random.randn(hidden_dim, 1).astype(np.float32) * scale2
        self.b_commit = np.zeros(1, dtype=np.float32)
        
        self.W_files = np.random.randn(hidden_dim, 1).astype(np.float32) * scale2
        self.b_files = np.zeros(1, dtype=np.float32)
        
        self.W_crossref = np.random.randn(hidden_dim, 1).astype(np.float32) * scale2
        self.b_crossref = np.zeros(1, dtype=np.float32)
        
        self.losses = []
    
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass → (commit_prob, file_count, crossref_prob)."""
        self.X = X
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)
        
        commit = self.sigmoid(self.A1 @ self.W_commit + self.b_commit)
        files = self.sigmoid(self.A1 @ self.W_files + self.b_files)  # bounded 0-1
        crossref = self.sigmoid(self.A1 @ self.W_crossref + self.b_crossref)
        
        return commit, files, crossref
    
    def train_step(
        self,
        X: np.ndarray,
        y_commit: np.ndarray,
        y_files: np.ndarray,
        y_crossref: np.ndarray,
    ) -> float:
        """One training step. Returns loss."""
        batch = X.shape[0]
        
        # Forward
        pred_commit, pred_files, pred_crossref = self.forward(X)
        
        # Binary cross-entropy for commit prediction
        eps = 1e-7
        loss_commit = -np.mean(
            y_commit * np.log(pred_commit + eps) + 
            (1 - y_commit) * np.log(1 - pred_commit + eps)
        )
        # Binary cross-entropy for file activity (has files changed)
        loss_files = -np.mean(
            y_files * np.log(pred_files + eps) + 
            (1 - y_files) * np.log(1 - pred_files + eps)
        )
        # Binary cross-entropy for crossref
        loss_crossref = -np.mean(
            y_crossref * np.log(pred_crossref + eps) + 
            (1 - y_crossref) * np.log(1 - pred_crossref + eps)
        )
        loss = loss_commit + loss_files + loss_crossref
        
        # Backward (simplified gradient)
        d_commit = (pred_commit - y_commit.reshape(-1, 1)) / batch
        d_files = (pred_files - y_files.reshape(-1, 1)) / batch
        d_crossref = (pred_crossref - y_crossref.reshape(-1, 1)) / batch
        
        # Output gradients
        dW_commit = self.A1.T @ d_commit
        dW_files = self.A1.T @ d_files.reshape(-1, 1)
        dW_crossref = self.A1.T @ d_crossref
        
        # Hidden gradient (ReLU)
        dA1 = (d_commit @ self.W_commit.T + 
               d_files.reshape(-1, 1) @ self.W_files.T +
               d_crossref @ self.W_crossref.T)
        dZ1 = dA1 * (self.Z1 > 0).astype(np.float32)
        dW1 = self.X.T @ dZ1
        
        # Update
        clip = max(1.0, self.lr * 10)
        self.W1 -= self.lr * np.clip(dW1, -clip, clip)
        self.b1 -= self.lr * np.clip(dZ1.sum(axis=0), -clip, clip)
        self.W_commit -= self.lr * np.clip(dW_commit, -clip, clip)
        self.b_commit -= self.lr * np.clip(d_commit.sum(axis=0), -clip, clip)
        self.W_files -= self.lr * np.clip(dW_files, -clip, clip)
        self.b_files -= self.lr * np.clip(d_files.sum(), -clip, clip)
        self.W_crossref -= self.lr * np.clip(dW_crossref, -clip, clip)
        self.b_crossref -= self.lr * np.clip(d_crossref.sum(axis=0), -clip, clip)
        
        return float(loss)
    
    def fit(
        self,
        X: np.ndarray,
        y_commit: np.ndarray,
        y_files: np.ndarray,
        y_crossref: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> List[float]:
        """Train the model."""
        n = X.shape[0]
        self.losses = []
        
        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(n)
            epoch_loss = 0.0
            batches = 0
            
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                loss = self.train_step(
                    X[idx], y_commit[idx], y_files[idx], y_crossref[idx],
                )
                epoch_loss += loss
                batches += 1
            
            avg_loss = epoch_loss / max(batches, 1)
            self.losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")
        
        return self.losses
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict commit probability, file count, crossref probability."""
        commit, files, crossref = self.forward(X)
        return {
            "commit_prob": commit.flatten(),
            "file_count": files.flatten(),
            "crossref_prob": crossref.flatten(),
        }
    
    def save(self, path: str):
        """Save weights to numpy file."""
        np.savez(
            path,
            W1=self.W1, b1=self.b1,
            W_commit=self.W_commit, b_commit=self.b_commit,
            W_files=self.W_files, b_files=self.b_files,
            W_crossref=self.W_crossref, b_crossref=self.b_crossref,
            input_dim=np.array(self.input_dim),
            hidden_dim=np.array(self.hidden_dim),
        )
    
    @classmethod
    def load(cls, path: str) -> "CommitPredictor":
        """Load weights from numpy file."""
        data = np.load(path)
        model = cls(
            input_dim=int(data["input_dim"]),
            hidden_dim=int(data["hidden_dim"]),
        )
        model.W1 = data["W1"]
        model.b1 = data["b1"]
        model.W_commit = data["W_commit"]
        model.b_commit = data["b_commit"]
        model.W_files = data["W_files"]
        model.b_files = data["b_files"]
        model.W_crossref = data["W_crossref"]
        model.b_crossref = data["b_crossref"]
        return model

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(input_dim={self.input_dim!r}, hidden_dim={self.hidden_dim!r}, lr={self.lr!r})"



# ─── Training Pipeline ───────────────────────────────────────────────

def train_commit_predictor(
    commits: List[CommitPoint],
    repos: Optional[List[str]] = None,
    window_hours: float = 6.0,
    epochs: int = 100,
    hidden_dim: int = 32,
    lr: float = 0.01,
) -> Tuple[CommitPredictor, Dict]:
    """
    End-to-end: mine commits → build dataset → train predictor.
    
    Returns (model, metrics).
    """
    target_repos = repos or list(set(c.repo for c in commits))
    
    # Build datasets
    all_samples = []
    for repo in target_repos:
        samples = build_prediction_dataset(commits, repo, window_hours=window_hours)
        all_samples.extend(samples)
    
    if not all_samples:
        raise ValueError("No training samples generated")
    
    # Feature dim = 4 (time) + len(REPO_VOCAB) + len(LANG_VOCAB) + 7 (agg)
    feature_dim = all_samples[0].features.shape[0]
    
    X = np.stack([s.features for s in all_samples])
    y_commit = np.array([s.label_commit_next_hour for s in all_samples], dtype=np.float32)
    y_files = np.array([s.label_file_count for s in all_samples], dtype=np.float32)
    y_crossref = np.array([s.label_cross_ref for s in all_samples], dtype=np.float32)
    
    # Normalize features
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X = (X - mean) / std
    
    # Binarize file count: any files changed = 1.0
    y_files = (y_files > 0).astype(np.float32)
    
    # Train
    model = CommitPredictor(input_dim=feature_dim, hidden_dim=hidden_dim, lr=lr)
    model.fit(X, y_commit, y_files, y_crossref, epochs=epochs, verbose=True)
    
    # Evaluate
    preds = model.predict(X)
    commit_acc = float(np.mean(
        (preds["commit_prob"] > 0.5) == (y_commit > 0.5)
    ))
    file_acc = float(np.mean(
        (preds["file_count"] > 0.5) == (y_files > 0.5)
    ))
    crossref_acc = float(np.mean(
        (preds["crossref_prob"] > 0.5) == (y_crossref > 0.5)
    ))
    
    metrics = {
        "samples": len(all_samples),
        "repos": len(target_repos),
        "feature_dim": feature_dim,
        "commit_accuracy": round(commit_acc, 4),
        "file_activity_accuracy": round(file_acc, 4),
        "crossref_accuracy": round(crossref_acc, 4),
        "final_loss": round(model.losses[-1], 4),
        "commit_rate": round(float(y_commit.mean()), 4),
        "crossref_rate": round(float(y_crossref.mean()), 4),
    }
    
    return model, metrics
