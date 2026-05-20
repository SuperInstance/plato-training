"""
Eisenstein Encoder — tiny text encoder with Eisenstein-structured weights.

Architecture:
  1. Token embedding (vocab_size × embed_dim)
  2. Bag-of-words aggregation (mean pooling)
  3. SplineLinear projection (embed_dim → out_dim) — Eisenstein lattice weights
  4. SplineLinear refinement (out_dim → out_dim)
  5. Output: L2-normalized embedding

Target: < 50KB total, competitive with Model2Vec at 100x smaller.
"""

from __future__ import annotations

import gc
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from .spline import SplineLinear

    _HAS_SPLINE = True
except ImportError:
    _HAS_SPLINE = False


class EisensteinEncoder(nn.Module):
    """
    Tiny text encoder with Eisenstein-structured weights.

    Uses SplineLinear layers (weights materialised from control points on
    an Eisenstein hexagonal lattice) for dramatic compression while
    maintaining representational capacity.

    Args:
        vocab_size:      Number of hash buckets for tokenisation.
        embed_dim:       Embedding dimensionality.
        out_dim:         Output embedding dimensionality.
        n_control_points: Control points for SplineLinear layers.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        embed_dim: int = 16,
        out_dim: int = 32,
        n_control_points: int = 8,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        if _HAS_SPLINE:
            self.project = SplineLinear(
                embed_dim, out_dim, n_control_points=n_control_points
            )
            self.refine = SplineLinear(
                out_dim, out_dim, n_control_points=n_control_points
            )
            self._uses_spline = True
        else:
            self.project = nn.Linear(embed_dim, out_dim)
            self.refine = nn.Linear(out_dim, out_dim)
            self._uses_spline = False

        self.act = nn.GELU()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len) integer tensor
        Returns:
            (batch, out_dim) L2-normalized embeddings
        """
        x = self.embedding(token_ids).mean(dim=1)  # (B, embed_dim)
        h = self.act(self.project(x))  # (B, out_dim)
        out = self.norm(self.refine(h))  # (B, out_dim)
        return nn.functional.normalize(out, p=2, dim=1)

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode raw text → numpy vectors. Simple: lowercase, split, hash to vocab."""
        ids_list = []
        for t in texts:
            tokens = t.lower().split()
            ids = [hash(w) % self.vocab_size for w in tokens] or [0]
            ids_list.append(ids)
        max_len = max(len(i) for i in ids_list)
        padded = [i + [0] * (max_len - len(i)) for i in ids_list]
        with torch.no_grad():
            return self(torch.tensor(padded, dtype=torch.long)).numpy()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def size_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.parameters())

    @property
    def uses_spline(self) -> bool:
        return self._uses_spline


# ---------------------------------------------------------------------------
# Contrastive training
# ---------------------------------------------------------------------------


class ContrastiveTrainer:
    """Triplet-margin contrastive trainer for EisensteinEncoder."""

    def __init__(self, encoder: EisensteinEncoder, lr: float = 1e-3) -> None:
        self.encoder = encoder
        self.optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

    def _tokenise(self, texts: List[str]) -> torch.Tensor:
        ids_list = []
        for t in texts:
            tokens = t.lower().split()
            ids = [hash(w) % self.encoder.vocab_size for w in tokens] or [0]
            ids_list.append(ids)
        max_len = max(len(i) for i in ids_list)
        padded = [i + [0] * (max_len - len(i)) for i in ids_list]
        return torch.tensor(padded, dtype=torch.long)

    def train_step(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        margin: float = 0.3,
    ) -> float:
        a = self.encoder(anchor)
        p = self.encoder(positive)
        n = self.encoder(negative)
        loss = torch.clamp(
            (a - p).norm(dim=1) - (n - a).norm(dim=1) + margin, min=0
        ).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_epoch(
        self,
        triplets: List[Tuple[str, str, str]],
        epochs: int = 1,
    ) -> List[float]:
        """Train on text triplets. Returns list of losses per step."""
        losses: List[float] = []
        for _ in range(epochs):
            for anchor_text, pos_text, neg_text in triplets:
                anchor = self._tokenise([anchor_text])
                positive = self._tokenise([pos_text])
                negative = self._tokenise([neg_text])
                loss = self.train_step(anchor, positive, negative)
                losses.append(loss)
        return losses


# ---------------------------------------------------------------------------
# Training data — PLATO domain triplets
# ---------------------------------------------------------------------------

TRAINING_TRIPLETS: List[Tuple[str, str, str]] = [
    ("how many tests in plato", "test count for plato training", "what color is the sky"),
    ("what is splinelinear", "explain spline linear compression", "deploy to production"),
    ("fleet status", "current fleet state", "how to bake bread"),
    ("eisenstein integers", "eisenstein integer lattice", "weather forecast"),
    ("deploy drift detect", "how to deploy drift detection", "recipe for soup"),
    ("constraint theory rust", "constraint theory core crate", "music theory"),
    ("lamport clock plato", "plato lamport timestamp ordering", "cooking recipes"),
    ("collective inference loop", "predict observe learn cycle", "car maintenance"),
    ("tile lifecycle states", "training tile state machine", "gardening tips"),
    ("lora adapter layers", "lora low rank adaptation", "painting techniques"),
    ("fleet deploy command", "deploy to fleet nodes", "grocery shopping"),
    ("micro model npu", "tiny model neural processing unit", "swimming lessons"),
    ("plato training room", "plato training session room", "dance class"),
    ("tensor spline weights", "spline weight parameterization", "knitting patterns"),
    ("drift detection model", "model drift monitoring", "bird watching"),
    ("anomaly flag score", "anomaly detection scoring", "chess opening"),
    ("intent detect classifier", "intent classification model", "yoga poses"),
    ("pytorch room trainer", "pytorch training room", "grocery list"),
    ("tensorflow room keras", "keras training room tensorflow", "movie review"),
    ("hardware deploy pipeline", "hardware target deployment", "book club"),
    ("constraint satisfaction", "constraint solver algorithm", "travel plans"),
    ("forgemaster fleet agent", "forgemaster constraint specialist", "recipe ideas"),
    ("plato data loader", "data loading pipeline plato", "music playlist"),
    ("local tile store", "content addressed tile storage", "vacation plans"),
    ("training throttle fleet", "fleet aware training throttle", "workout routine"),
    ("spline compression ratio", "spline parameter compression", "dinner menu"),
    ("npu quantize int8", "int8 quantization for npu", "morning routine"),
    ("cpu tiny inference", "small model cpu inference", "evening plans"),
    ("gpu lora training", "lora fine tuning on gpu", "shopping list"),
    ("batch training tiles", "tile batch training loop", "coffee order"),
    ("model2vec comparison", "compare with model2vec", "traffic report"),
    ("embedding cosine similarity", "vector similarity metric", "car wash"),
    ("contrastive triplet loss", "triplet margin loss training", "pet grooming"),
    ("hash tokenization vocab", "vocabulary hash tokenization", "laundry"),
    ("relu activation function", "rectified linear unit", "sunset time"),
    ("l2 normalize vectors", "vector normalization layer", "water plants"),
    ("adam optimizer learning", "adam gradient descent", "feed the cat"),
    ("bag of words encode", "bow text encoding", "check mail"),
    ("control points lattice", "lattice control points", "cut the grass"),
    ("eisenstein weight matrix", "lattice weight materialization", "paint fence"),
    ("hexagonal lattice points", "hex lattice coordinate system", "fix the sink"),
    ("inference submillisecond", "fast inference latency", "slow traffic"),
    ("onnx export tiny model", "export model onnx format", "watch television"),
    ("fleet coordination protocol", "fleet agent coordination", "solo hiking"),
    ("collective inference gap", "inference gap learning", "ignore errors"),
    ("i2i bottle delivery", "inter instance bottle protocol", "postal mail"),
    ("git knowledge base", "fleet git repository", "social media"),
    ("micro deploy command", "deploy micro model command", "video games"),
    ("plato ensign training", "ensign model training", "professional sports"),
    ("constraint proof repo", "constraint theory proofs", "fashion trends"),
]
