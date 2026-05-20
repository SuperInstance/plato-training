"""
Post-LLM Filter — micro-model that processes LLM responses and decides
what knowledge to extract and persist as PLATO tiles.

Architecture: 2-layer MLP with SplineLinear (256→128→output heads)
~40K params dense, ~4K after spline compression.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spline import SplineLinear
from .types import TrainingTile, TileType, TileLifecycle, LamportClock


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_DIM = 256
HIDDEN_DIM = 128

RESPONSE_TYPES = ["code", "explanation", "analysis", "creative", "factual", "instruction", "error"]
MODEL_NAMES = ["glm-5.1", "glm-5-turbo", "seed-mini", "hermes", "qwen", "deepseek", "claude", "other"]
ROUTE_NAMES = ["direct", "pipeline", "fallback", "cached", "ensemble", "retry", "research", "escalation"]
KEEP_DECISIONS = ["discard", "summary", "full", "tile", "high_priority_tile"]
TILE_DOMAINS = ["constraint-theory", "fleet", "plato", "code", "ops", "research", "math", "architecture", "debugging", "general"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ResponseFeatures:
    """Converts raw LLM response + context into a 256-dim feature vector."""

    response_text: str = ""
    request_text: str = ""
    response_type: str = "explanation"
    model_used: str = "other"
    request_route: str = "direct"
    latency_ms: float = 1000.0
    response_tokens: int = 100
    existing_tile_count: int = 0

    def _one_hot(self, value: str, options: List[str]) -> List[float]:
        vec = [0.0] * len(options)
        if value in options:
            vec[options.index(value)] = 1.0
        return vec

    def _log_scale(self, val: float, max_val: float = 15.0) -> float:
        return min(max_val, math.log1p(val) / math.log1p(10**max_val) * max_val)

    def _keyword_vector(self, text: str, top_k: int = 64) -> List[float]:
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        stop = {"the", "and", "for", "that", "this", "with", "are", "but", "not", "you",
                "all", "can", "had", "her", "was", "one", "our", "out", "has", "have",
                "from", "been", "will", "they", "what", "which", "their", "would", "there"}
        filtered = [w for w in words if w not in stop]
        counts = Counter(filtered)
        top = [c[0] for c in counts.most_common(top_k)]
        # Deterministic hash to fixed positions
        vec = [0.0] * top_k
        for w in top:
            idx = int(hashlib.md5(w.encode()).hexdigest()[:8], 16) % top_k
            vec[idx] += 1.0
        mx = max(vec) if max(vec) > 0 else 1.0
        return [v / mx for v in vec]

    def _minhash(self, text: str, num_hashes: int = 32) -> List[float]:
        shingles = set()
        words = text.lower().split()
        for i in range(len(words) - 2):
            shingles.add(" ".join(words[i:i + 3]))
        if not shingles:
            return [0.0] * num_hashes
        vec = []
        for i in range(num_hashes):
            h = min(hashlib.sha256(f"{s}|{i}".encode()).hexdigest() for s in shingles)
            vec.append(int(h[:8], 16) / 0xFFFFFFFF)
        return vec

    def to_vector(self) -> np.ndarray:
        """Build the 256-dim feature vector."""
        parts: List[float] = []

        # response_length: log-scaled [0..15] = 16 dims
        resp_len = self._log_scale(len(self.response_text))
        parts += [resp_len] * 16

        # response_type: one-hot = 7 dims
        parts += self._one_hot(self.response_type, RESPONSE_TYPES)

        # model_used: one-hot = 8 dims
        parts += self._one_hot(self.model_used, MODEL_NAMES)

        # Scalar features (1 each)
        text_lower = self.response_text.lower()
        request_lower = self.request_text.lower()

        novelty = min(1.0, len(set(text_lower.split()) - set(request_lower.split()))
                      / max(1, len(set(text_lower.split()))))
        parts.append(novelty)  # novelty_score

        structure = sum([
            bool(re.search(r'```', self.response_text)),
            bool(re.search(r'^\d+\.', self.response_text, re.M)),
            bool(re.search(r'^#{1,4}\s', self.response_text, re.M)),
            bool(re.search(r'[-*]\s', self.response_text)),
            bool(re.search(r'\|.+\|', self.response_text)),
        ]) / 5.0
        parts.append(structure)  # structure_score

        parts.append(min(1.0, novelty * structure))  # domain_match proxy
        parts.append(0.5)  # usefulness_prediction (neutral prior)

        # factual_density
        facts = len(re.findall(r'\d+\.?\d*', self.response_text))
        parts.append(min(1.0, facts / max(1, self.response_tokens) * 10))

        parts.append(1.0 if '```' in self.response_text else 0.0)  # code_presence
        parts.append(1.0 if re.search(r'(therefore|thus|so|because|step\s+\d|first|then|next)',
                                       text_lower) else 0.0)  # has_reasoning
        parts.append(1.0 if re.search(r'^\d+\.\s', self.response_text, re.M) else 0.0)  # has_actionable_steps
        parts.append(min(1.0, len(re.findall(r'(see|ref|according|cite|source)', text_lower)) / 5.0))

        # confidence_score
        conf_words = len(re.findall(r'\b(certain|confident|sure|definitely|clearly|obviously)\b', text_lower))
        hedge_words = len(re.findall(r'\b(maybe|perhaps|might|could|possibly|uncertain|approx)\b', text_lower))
        parts.append(min(1.0, max(0.0, (conf_words - hedge_words) / 5.0 + 0.5)))

        # request_route: one-hot = 8 dims
        parts += self._one_hot(self.request_route, ROUTE_NAMES)

        # keyword_overlap
        resp_words = set(re.findall(r'\b[a-z]{3,}\b', text_lower))
        req_words = set(re.findall(r'\b[a-z]{3,}\b', request_lower))
        overlap = len(resp_words & req_words) / max(1, len(req_words))
        parts.append(overlap)

        # self_correction
        parts.append(1.0 if re.search(r'(however|actually|correction|wait|I was wrong|mistake|let me reconsider)',
                                       text_lower) else 0.0)

        # embedding_similarity_to_corpus (approximate with tile count)
        parts.append(min(1.0, self.existing_tile_count / 100.0))

        # response_tokens: log-scaled = 16 dims
        tok_scaled = self._log_scale(self.response_tokens)
        parts += [tok_scaled] * 16

        # latency_ms: log-scaled = 16 dims
        lat_scaled = self._log_scale(self.latency_ms)
        parts += [lat_scaled] * 16

        # keyword_vector = 64 dims
        parts += self._keyword_vector(self.response_text, 64)

        # embedding_hash (minhash) = 32 dims
        parts += self._minhash(self.response_text, 32)

        # Pad/truncate to exactly 256
        if len(parts) < FEATURE_DIM:
            parts += [0.0] * (FEATURE_DIM - len(parts))
        parts = parts[:FEATURE_DIM]

        return np.array(parts, dtype=np.float32)


@dataclass
class FilterDecision:
    """Output decisions from the PostFilter."""
    keep_decision: str = "summary"        # discard, summary, full, tile, high_priority_tile
    extraction_confidence: float = 0.5    # [0, 1]
    compression_ratio: float = 0.5        # 1.0 = keep all, 0.1 = essence only
    tile_domain: str = "general"          # which PLATO room
    reusability_score: float = 0.5        # [0, 1]

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class KnowledgeTile:
    """Extracted knowledge from an LLM response."""
    tile_id: str = ""
    domain: str = "general"
    content_type: str = "fact"  # fact, pattern, decision, template
    compressed_content: str = ""
    source_model: str = "unknown"
    confidence: float = 0.5
    reuse_count: int = 0
    last_used: float = 0.0
    novelty: float = 0.5
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "KnowledgeTile":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class PostFilterModel(nn.Module):
    """2-layer MLP with SplineLinear: 256→128→output heads."""

    def __init__(self, n_control_points: int = 16):
        super().__init__()
        self.encoder = SplineLinear(FEATURE_DIM, HIDDEN_DIM, n_control_points=n_control_points)
        self.hidden_norm = nn.LayerNorm(HIDDEN_DIM)

        # Output heads
        self.keep_head = SplineLinear(HIDDEN_DIM, len(KEEP_DECISIONS), n_control_points=max(4, n_control_points // 2))
        self.confidence_head = nn.Linear(HIDDEN_DIM, 1)
        self.compression_head = nn.Linear(HIDDEN_DIM, 1)
        self.domain_head = SplineLinear(HIDDEN_DIM, len(TILE_DOMAINS), n_control_points=max(4, n_control_points // 2))
        self.reusability_head = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = F.relu(self.hidden_norm(self.encoder(x)))

        keep_logits = self.keep_head(h)
        confidence = torch.sigmoid(self.confidence_head(h)).squeeze(-1)
        compression = torch.sigmoid(self.compression_head(h)).squeeze(-1)
        domain_logits = self.domain_head(h)
        reusability = torch.sigmoid(self.reusability_head(h)).squeeze(-1)

        return {
            "keep_logits": keep_logits,
            "confidence": confidence,
            "compression": compression,
            "domain_logits": domain_logits,
            "reusability": reusability,
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def _extract_facts(text: str) -> List[str]:
    """Pull key claims containing numbers or strong assertions."""
    facts = []
    for line in text.split('\n'):
        line = line.strip().strip('-*•')
        if len(line) < 10 or len(line) > 300:
            continue
        if re.search(r'\d+\.?\d*%?', line) or re.search(r'\b(is|are|equals|means|produces|results? in)\b', line.lower()):
            facts.append(line)
    return facts[:20]


def _extract_patterns(text: str) -> List[str]:
    """Identify reusable code patterns or formulas."""
    patterns = []
    # Code blocks
    for block in re.findall(r'```[\w]*\n(.*?)```', text, re.S):
        if len(block.strip()) > 20:
            patterns.append(block.strip()[:500])
    # Formulas
    for m in re.finditer(r'([A-Z]\w*\s*[=+]\s*[^.\n]{10,80})', text):
        patterns.append(m.group(0).strip())
    return patterns[:10]


def _extract_decisions(text: str) -> List[str]:
    """Record what worked or didn't."""
    decisions = []
    for pattern in [r'(?:use|used|chose|choosing)\s+(.+?)(?:\.|,|$)',
                    r'(?:avoid|don\'t|never)\s+(.+?)(?:\.|,|$)',
                    r'(?:worked|succeeded|best)\s+(.+?)(?:\.|,|$)']:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            d = m.group(0).strip()
            if len(d) > 10:
                decisions.append(d[:200])
    return decisions[:10]


def _compress(text: str, ratio: float) -> str:
    """Compress text to roughly the given ratio."""
    if ratio >= 0.95:
        return text
    sentences = re.split(r'(?<=[.!?])\s+', text)
    n_keep = max(1, int(len(sentences) * ratio))
    return ' '.join(sentences[:n_keep])


# ---------------------------------------------------------------------------
# PostFilter orchestrator
# ---------------------------------------------------------------------------

class PostFilter:
    """Orchestrates response analysis, knowledge extraction, and self-training."""

    def __init__(self, n_control_points: int = 16, lr: float = 1e-3):
        self.model = PostFilterModel(n_control_points=n_control_points)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.clock = LamportClock()
        self._training_data: List[Tuple[np.ndarray, Dict]] = []
        self._tiles: Dict[str, KnowledgeTile] = {}

    # ---- Main API ----

    def process(self, response_text: str, request_context: str = "",
                model_used: str = "other", response_type: str = "explanation",
                request_route: str = "direct", latency_ms: float = 1000.0,
                response_tokens: int = 100) -> Tuple[FilterDecision, List[KnowledgeTile]]:
        """Analyze an LLM response and return filter decision + extracted tiles."""

        features = ResponseFeatures(
            response_text=response_text,
            request_text=request_context,
            response_type=response_type,
            model_used=model_used,
            request_route=request_route,
            latency_ms=latency_ms,
            response_tokens=response_tokens,
        )
        x = torch.from_numpy(features.to_vector()).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            out = self.model(x)

        decision = FilterDecision(
            keep_decision=KEEP_DECISIONS[out["keep_logits"].argmax(-1).item()],
            extraction_confidence=out["confidence"].item(),
            compression_ratio=out["compression"].item(),
            tile_domain=TILE_DOMAINS[out["domain_logits"].argmax(-1).item()],
            reusability_score=out["reusability"].item(),
        )

        tiles: List[KnowledgeTile] = []
        if decision.keep_decision != "discard":
            compressed = _compress(response_text, decision.compression_ratio)

            # Extract facts
            for fact in _extract_facts(response_text):
                tid = hashlib.sha256(fact.encode()).hexdigest()[:12]
                tiles.append(KnowledgeTile(
                    tile_id=tid, domain=decision.tile_domain,
                    content_type="fact", compressed_content=fact,
                    source_model=model_used, confidence=decision.extraction_confidence,
                    novelty=features.to_vector()[3],  # novelty_score dimension
                ))

            # Extract patterns
            for pat in _extract_patterns(response_text):
                tid = hashlib.sha256(pat.encode()).hexdigest()[:12]
                tiles.append(KnowledgeTile(
                    tile_id=tid, domain=decision.tile_domain,
                    content_type="pattern", compressed_content=pat,
                    source_model=model_used, confidence=decision.extraction_confidence * 0.9,
                    novelty=features.to_vector()[3],
                ))

            # Extract decisions
            for dec in _extract_decisions(response_text):
                tid = hashlib.sha256(dec.encode()).hexdigest()[:12]
                tiles.append(KnowledgeTile(
                    tile_id=tid, domain=decision.tile_domain,
                    content_type="decision", compressed_content=dec,
                    source_model=model_used, confidence=decision.extraction_confidence * 0.8,
                    novelty=features.to_vector()[3],
                ))

            # If no structured extractions, create a summary tile
            if not tiles and decision.keep_decision in ("tile", "high_priority_tile"):
                tid = hashlib.sha256(compressed.encode()).hexdigest()[:12]
                tiles.append(KnowledgeTile(
                    tile_id=tid, domain=decision.tile_domain,
                    content_type="fact", compressed_content=compressed[:500],
                    source_model=model_used, confidence=decision.extraction_confidence,
                ))

        # Store tiles for reuse tracking
        for t in tiles:
            self._tiles[t.tile_id] = t

        return decision, tiles

    # ---- Self-training ----

    def record_usefulness(self, tile_id: str, was_useful: bool):
        """Record training signal: was this tile actually useful?"""
        if tile_id in self._tiles:
            tile = self._tiles[tile_id]
            if was_useful:
                tile.reuse_count += 1
                tile.last_used = time.time()

    def add_training_example(self, features: np.ndarray, target: Dict):
        """Store a (features, target) pair for self-training."""
        self._training_data.append((features, target))

    def self_train(self, epochs: int = 10, batch_size: int = 8):
        """Train on accumulated usefulness data."""
        if not self._training_data:
            return {}

        self.model.train()
        dataset = self._training_data
        n = len(dataset)
        losses = []

        for epoch in range(epochs):
            perm = np.random.permutation(n)
            epoch_loss = 0.0
            count = 0

            for i in range(0, n, batch_size):
                batch_idx = perm[i:i + batch_size]
                x_batch = torch.from_numpy(
                    np.stack([dataset[j][0] for j in batch_idx])
                )

                keep_target = torch.tensor(
                    [t["keep"] for t in [dataset[j][1] for j in batch_idx]]
                )
                conf_target = torch.tensor(
                    [t["confidence"] for t in [dataset[j][1] for j in batch_idx]],
                    dtype=torch.float32
                )
                domain_target = torch.tensor(
                    [t["domain"] for t in [dataset[j][1] for j in batch_idx]]
                )
                reuse_target = torch.tensor(
                    [t["reusability"] for t in [dataset[j][1] for j in batch_idx]],
                    dtype=torch.float32
                )

                out = self.model(x_batch)

                loss = (
                    F.cross_entropy(out["keep_logits"], keep_target) +
                    F.mse_loss(out["confidence"], conf_target) +
                    F.cross_entropy(out["domain_logits"], domain_target) +
                    F.mse_loss(out["reusability"], reuse_target)
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                count += 1

            losses.append(epoch_loss / max(1, count))

        return {"epochs": epochs, "final_loss": losses[-1] if losses else 0.0, "losses": losses}

    # ---- Tile persistence ----

    def export_tile(self, store_dir: str) -> str:
        """Save model state as a PLATO TrainingTile."""
        import os, json
        os.makedirs(store_dir, exist_ok=True)

        data = {
            "model_state": {k: v.tolist() for k, v in self.model.state_dict().items()},
            "param_count": self.model.param_count(),
            "timestamp": time.time(),
            "tile_count": len(self._tiles),
        }
        tile_id = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]
        path = os.path.join(store_dir, f"post_filter_{tile_id}.json")
        with open(path, 'w') as f:
            json.dump(data, f)
        return tile_id

    def load_tile(self, tile_id: str, store_dir: str) -> bool:
        """Load model state from a saved PLATO tile."""
        import os, json
        path = os.path.join(store_dir, f"post_filter_{tile_id}.json")
        if not os.path.exists(path):
            return False
        with open(path) as f:
            data = json.load(f)
        state = {k: torch.tensor(v) for k, v in data["model_state"].items()}
        self.model.load_state_dict(state)
        return True

    def get_tile(self, tile_id: str) -> Optional[KnowledgeTile]:
        return self._tiles.get(tile_id)

    @property
    def tile_count(self) -> int:
        return len(self._tiles)
