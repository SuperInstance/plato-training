"""
Intelligence Room — PLATO room that distills LLM knowledge into self-improving micro-models.

Architecture:
  Request ──► PreFilter (micro) ──► [LLM?] ──► PostFilter (micro) ──► Response
                  │                                   │
                  ▼                                   ▼
           Routing Decision                     Knowledge Tiles
           (model, tokens, skip)                (facts, patterns, decisions)
                  │                                   │
                  ▼                                   ▼
           ┌──────────────────────────────────────────┐
           │          SelfTrainer (idle loop)          │
           │  Accumulates experience → retrains models │
           │  Deploys improved weights as PLATO tiles  │
           └──────────────────────────────────────────┘

The key insight: every LLM call is a training opportunity.
The pre-filter learns to skip unnecessary calls.
The post-filter learns to keep only valuable knowledge.
Both improve continuously from live usage.

Token savings compound: as models improve, fewer LLM calls needed,
and each call is better-targeted (fewer wasted tokens).
"""

from __future__ import annotations
import time
import json
import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
from collections import defaultdict

import torch
import torch.nn as nn

from .types import (
    TrainingTile, TileType, TileLifecycle, LamportClock,
    TrainingConfig, TrainingMetrics, content_hash,
)
from .store import LocalTileStore

# Optional new modules — graceful fallback
try:
    from .tutor_judge import word_similarity as _bitvector_word_similarity
    _HAS_TUTOR_JUDGE = True
except ImportError:
    _HAS_TUTOR_JUDGE = False

try:
    from .semantic_matcher import SemanticMatcher as _SemanticMatcherCls
    _HAS_SEMANTIC_MATCHER = True
except ImportError:
    _HAS_SEMANTIC_MATCHER = False
    _SemanticMatcherCls = None

try:
    from .eisenstein_encoder import EisensteinEncoder as _EisensteinEncoderCls
    _HAS_EISENSTEIN = True
except ImportError:
    _HAS_EISENSTEIN = False
    _EisensteinEncoderCls = None

try:
    from .device_router import DeviceRouter as _DeviceRouterCls
    _HAS_DEVICE_ROUTER = True
except ImportError:
    _HAS_DEVICE_ROUTER = False
    _DeviceRouterCls = None


# ─── Intelligence Tile Types ──────────────────────────────────────

class IntelligenceTileType(Enum):
    """Extended tile types for the intelligence system."""
    KNOWLEDGE = "knowledge"           # Distilled LLM knowledge
    EXPERIENCE = "experience"         # (request, response, outcome) tuple
    ROUTING_DECISION = "routing"      # Pre-filter decision record
    FILTER_DECISION = "filter"        # Post-filter decision record
    TRAINING_SIGNAL = "training"      # Self-trainer signal
    MODEL_CHECKPOINT = "checkpoint"   # Micro-model weights
    VALIDATION_RESULT = "validation"  # Model quality assessment


class Route(Enum):
    """Possible routing decisions for a request."""
    SKIP_LLM = 0          # Already known, use cached knowledge
    USE_LOCAL = 1         # Local micro-model can handle it
    USE_TINY = 2          # Seed-2.0-mini ($0.01)
    USE_SMALL = 3          # Qwen3.6-35B ($0.01)
    USE_MEDIUM = 4         # GLM-5-turbo or Hermes-70B ($0.03)
    USE_LARGE = 5          # GLM-5.1 ($0.05)
    USE_REASONING = 6      # DeepSeek v4-pro ($0.10)
    DELEGATE = 7           # Hand off to subagent


class KeepDecision(Enum):
    """Post-filter decision on what to keep from response."""
    DISCARD = 0             # Not useful
    SUMMARY = 1             # Keep compressed summary
    FULL = 2                # Keep full response
    KNOWLEDGE_TILE = 3      # Extract as structured knowledge
    HIGH_PRIORITY = 4       # Critical knowledge, prioritize


# ─── Knowledge Tile ────────────────────────────────────────────────

@dataclass
class KnowledgeTile:
    """A piece of distilled knowledge extracted from an LLM response."""
    tile_id: str
    domain: str              # constraint-theory, fleet, plato, code, math, etc.
    content_type: str        # fact, pattern, decision, template, code_snippet
    compressed_content: str  # The distilled essence
    source_model: str        # Which LLM produced this
    confidence: float
    reuse_count: int = 0     # How many times this was useful
    last_used: float = 0.0
    novelty: float = 1.0     # How unique (1.0 = novel, 0.0 = duplicate)
    created_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)

    def touch(self):
        """Mark this tile as used."""
        self.reuse_count += 1
        self.last_used = time.time()

    def usefulness_score(self) -> float:
        """How useful this tile has been, decaying over time."""
        age_hours = (time.time() - self.created_at) / 3600
        if age_hours < 1:
            return float(self.reuse_count)
        return self.reuse_count / (1 + 0.1 * age_hours)


# ─── Intelligence Experience ───────────────────────────────────────

@dataclass
class IntelligenceExperience:
    """A complete (request → routing → LLM → filtering → outcome) record."""
    exp_id: str
    timestamp: float = field(default_factory=time.time)

    # Request side
    request_hash: str = ""        # Hash of request content
    request_length: int = 0
    intent: str = ""              # query, command, creative, code, analysis
    domain: str = ""              # constraint-theory, fleet, plato, general
    urgency: str = "medium"       # low, medium, high, critical

    # Routing
    route_taken: str = ""         # Which Route enum was used
    route_confidence: float = 0.0
    route_correct: Optional[bool] = None  # Filled in after outcome

    # Response side
    response_length: int = 0
    response_model: str = ""
    response_latency_ms: float = 0.0

    # Filtering
    keep_decision: str = ""       # Which KeepDecision was used
    knowledge_extracted: int = 0  # Number of knowledge tiles extracted

    # Outcome (filled in later)
    outcome: str = ""             # good, waste, missed, bad_route
    tokens_saved: int = 0
    knowledge_reused: List[str] = field(default_factory=list)
    user_satisfied: Optional[bool] = None

    # Feature vectors (for model training)
    request_features: Optional[np.ndarray] = None  # 384-dim
    response_features: Optional[np.ndarray] = None  # 256-dim


# ─── Intelligence Room ─────────────────────────────────────────────

class IntelligenceRoom:
    """
    The PLATO Intelligence Room — self-improving LLM knowledge distillation.

    Usage:
        room = IntelligenceRoom(store_dir=".plato-intelligence")

        # Before LLM call
        route = room.pre_route(request_text, domain="fleet")
        if route.decision == Route.SKIP_LLM:
            return room.retrieve_cached(request_text)

        # After LLM call
        result = room.post_process(response_text, request_context)
        for tile in result.knowledge_tiles:
            print(f"Extracted: {tile.compressed_content[:80]}")

        # Record outcome (async)
        room.record_outcome(exp_id, outcome="good", tokens_saved=1500)

        # Self-train when idle
        room.self_train_cycle()
    """

    def __init__(
        self,
        store_dir: str = ".plato-intelligence",
        max_knowledge_tiles: int = 10000,
        max_experience: int = 50000,
        use_semantic: bool = True,
        use_eisenstein: bool = True,
        use_device_router: bool = True,
    ):
        self.store = LocalTileStore(store_dir)
        self.max_knowledge = max_knowledge_tiles
        self.max_experience = max_experience

        # Knowledge base
        self.knowledge: Dict[str, KnowledgeTile] = {}
        self.experiences: List[IntelligenceExperience] = []

        # Statistics
        self.stats = {
            "total_requests": 0,
            "llm_calls_saved": 0,
            "tokens_saved": 0,
            "knowledge_tiles_created": 0,
            "knowledge_tiles_reused": 0,
            "self_train_cycles": 0,
            "last_self_train": 0.0,
            "semantic_hits": 0,
            "bitvector_hits": 0,
        }

        # Load state
        self._load_state()

        # Models (lazy-loaded)
        self._pre_filter = None
        self._post_filter = None

        # --- New modules (optional, graceful fallback) ---

        # Semantic matcher for knowledge retrieval
        self._semantic_matcher = None
        if use_semantic and _HAS_SEMANTIC_MATCHER:
            try:
                self._semantic_matcher = _SemanticMatcherCls(threshold=0.6)
                # Re-index existing knowledge
                for key, tile in self.knowledge.items():
                    text = f"{tile.domain} {tile.compressed_content}"
                    self._semantic_matcher.add(key, text)
            except Exception:
                self._semantic_matcher = None

        # Eisenstein encoder for lightweight embeddings
        self._eisenstein_encoder = None
        if use_eisenstein and _HAS_EISENSTEIN:
            try:
                self._eisenstein_encoder = _EisensteinEncoderCls()
                self._eisenstein_encoder.eval()
            except Exception:
                self._eisenstein_encoder = None

        # Device router for inference routing
        self._device_router = None
        if use_device_router and _HAS_DEVICE_ROUTER:
            try:
                self._device_router = _DeviceRouterCls()
            except Exception:
                self._device_router = None

    # ─── Pre-Filter (Routing) ─────────────────────────────────────

    def pre_route(
        self,
        request_text: str,
        domain: str = "general",
        urgency: str = "medium",
        intent: str = "query",
        context_size: int = 0,
    ) -> Dict[str, Any]:
        """
        Route a request before sending to LLM.

        Returns dict with:
          - decision: Route enum
          - confidence: float
          - model_hint: suggested model name
          - max_tokens: suggested token limit
          - cached_answer: if SKIP_LLM, the cached knowledge
        """
        self.stats["total_requests"] += 1

        # Check knowledge base first (zero-cost)
        cached = self._check_knowledge(request_text, domain)
        if cached is not None:
            self.stats["llm_calls_saved"] += 1
            self.stats["tokens_saved"] += self._estimate_tokens(request_text) * 2
            return {
                "decision": Route.SKIP_LLM,
                "confidence": cached.confidence,
                "model_hint": "cached",
                "max_tokens": 0,
                "cached_answer": cached.compressed_content,
            }

        # Use micro-model for routing if available
        if self._pre_filter is not None:
            return self._pre_filter_route(
                request_text, domain, urgency, intent, context_size,
            )

        # Fallback: heuristic routing
        return self._heuristic_route(
            request_text, domain, urgency, intent, context_size,
        )

    def _check_knowledge(
        self, request_text: str, domain: str,
    ) -> Optional[KnowledgeTile]:
        """Check if we already have knowledge that answers this request.

        Cascade: exact → bitvector → semantic → keyword fallback.
        """
        # Tier 1: Exact string match (fastest)
        for tile in self.knowledge.values():
            if tile.compressed_content == request_text:
                tile.touch()
                self.stats["knowledge_tiles_reused"] += 1
                return tile

        # Tier 2: Bitvector matching via TutorJudge (50x faster than embeddings)
        if _HAS_TUTOR_JUDGE:
            tile = self._check_knowledge_bitvector(request_text, domain)
            if tile is not None:
                self.stats["bitvector_hits"] += 1
                return tile

        # Tier 3: Semantic matching via SemanticMatcher (catches paraphrases)
        if self._semantic_matcher is not None and self._semantic_matcher.size() > 0:
            result = self._semantic_matcher.match(request_text)
            if result is not None:
                key, score, _ = result
                tile = self.knowledge.get(key)
                if tile is not None:
                    tile.touch()
                    self.stats["semantic_hits"] += 1
                    self.stats["knowledge_tiles_reused"] += 1
                    return tile

        # Tier 4: Keyword overlap fallback (original logic)
        return self._check_knowledge_keyword(request_text, domain)

    def _check_knowledge_bitvector(
        self, request_text: str, domain: str, threshold: float = 0.5,
    ) -> Optional[KnowledgeTile]:
        """Bitvector matching using TutorJudge word_similarity."""
        query_words = request_text.lower().split()
        best_tile = None
        best_score = 0.0

        for tile in self.knowledge.values():
            if tile.domain != domain and domain != "general":
                continue
            tile_words = tile.compressed_content.lower().split()
            # Average max-similarity per query word
            total_sim = 0.0
            for qw in query_words:
                max_sim = max(
                    (_bitvector_word_similarity(qw, tw) for tw in tile_words),
                    default=0.0,
                )
                total_sim += max_sim
            avg_sim = total_sim / max(len(query_words), 1)
            score = avg_sim * tile.confidence * (1 + 0.1 * min(tile.reuse_count, 10))

            if score > best_score and avg_sim > threshold:
                best_score = score
                best_tile = tile

        if best_tile is not None:
            best_tile.touch()
            self.stats["knowledge_tiles_reused"] += 1
            return best_tile
        return None

    def _check_knowledge_keyword(
        self, request_text: str, domain: str,
    ) -> Optional[KnowledgeTile]:
        """Original keyword overlap matching (fallback)."""
        query_words = set(request_text.lower().split())

        best_tile = None
        best_score = 0.0

        for tile in self.knowledge.values():
            if tile.domain != domain and domain != "general":
                continue

            # Simple keyword overlap scoring
            tile_words = set(tile.compressed_content.lower().split())
            overlap = len(query_words & tile_words) / max(len(query_words), 1)

            # Weight by confidence and usefulness
            score = overlap * tile.confidence * (1 + 0.1 * min(tile.reuse_count, 10))

            if score > best_score and score > 0.3:
                best_score = score
                best_tile = tile

        if best_tile is not None:
            best_tile.touch()
            self.stats["knowledge_tiles_reused"] += 1

        return best_tile

    def _heuristic_route(
        self,
        request_text: str,
        domain: str,
        urgency: str,
        intent: str,
        context_size: int,
    ) -> Dict[str, Any]:
        """Fallback heuristic routing when no model is loaded."""
        length = len(request_text)

        # Code tasks → medium model
        if intent == "code" or "```" in request_text:
            return self._make_route(Route.USE_MEDIUM, 0.7, "glm-5.1", 2000)

        # Math/reasoning → reasoning model
        if domain == "constraint-theory" or intent == "math":
            return self._make_route(Route.USE_REASONING, 0.6, "deepseek-v4-pro", 3000)

        # Short queries → small model
        if length < 100:
            return self._make_route(Route.USE_SMALL, 0.8, "seed-2.0-mini", 500)

        # Urgent → large model (don't waste time on retries)
        if urgency == "critical":
            return self._make_route(Route.USE_LARGE, 0.9, "glm-5.1", 4000)

        # Default: medium
        return self._make_route(Route.USE_MEDIUM, 0.5, "glm-5-turbo", 1500)

    def _pre_filter_route(
        self,
        request_text: str,
        domain: str,
        urgency: str,
        intent: str,
        context_size: int,
    ) -> Dict[str, Any]:
        """Route using the pre-filter micro-model."""
        from .intelligence_pre_filter import PreFilter, RequestFeatures

        if self._pre_filter is None:
            self._pre_filter = PreFilter(store_dir=str(self.store.store_dir))

        features = RequestFeatures(
            text=request_text,
            domain=domain,
            urgency=urgency,
            intent=intent,
            context_size=context_size,
        )
        decision = self._pre_filter.route(features)
        return {
            "decision": Route(decision.route),
            "confidence": decision.confidence,
            "model_hint": decision.model_hint,
            "max_tokens": decision.max_tokens,
            "cached_answer": None,
        }

    def _make_route(
        self, route: Route, confidence: float, model: str, max_tokens: int,
    ) -> Dict[str, Any]:
        return {
            "decision": route,
            "confidence": confidence,
            "model_hint": model,
            "max_tokens": max_tokens,
            "cached_answer": None,
        }

    # ─── Post-Filter (Knowledge Extraction) ───────────────────────

    def post_process(
        self,
        response_text: str,
        request_text: str = "",
        domain: str = "general",
        model_used: str = "unknown",
        latency_ms: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Process an LLM response after receiving it.

        Returns dict with:
          - keep_decision: KeepDecision enum
          - knowledge_tiles: list of extracted KnowledgeTile objects
          - compressed: compressed version of response
          - compression_ratio: float
        """
        # Extract knowledge
        tiles = self._extract_knowledge(
            response_text, request_text, domain, model_used,
        )

        # Decide what to keep
        keep = self._decide_keep(response_text, tiles, domain)

        # Store knowledge tiles
        for tile in tiles:
            self._store_knowledge(tile)

        # Compress if needed
        compressed = response_text
        ratio = 1.0
        if keep == KeepDecision.SUMMARY:
            compressed = self._compress_response(response_text, tiles)
            ratio = len(compressed) / max(len(response_text), 1)

        return {
            "keep_decision": keep,
            "knowledge_tiles": tiles,
            "compressed": compressed,
            "compression_ratio": ratio,
        }

    def _extract_knowledge(
        self,
        response: str,
        request: str,
        domain: str,
        model: str,
    ) -> List[KnowledgeTile]:
        """Extract structured knowledge from an LLM response."""
        tiles = []
        lines = response.split('\n')

        # Extract code blocks
        current_code = []
        in_code = False
        lang = ""
        for line in lines:
            if line.startswith('```'):
                if in_code and current_code:
                    code = '\n'.join(current_code)
                    tile = self._make_knowledge_tile(
                        content=code,
                        domain=domain,
                        content_type="code_snippet",
                        model=model,
                        tags=[lang, "code"],
                    )
                    tiles.append(tile)
                    current_code = []
                in_code = not in_code
                if in_code:
                    lang = line[3:].strip().split()[0] if len(line) > 3 else ""
            elif in_code:
                current_code.append(line)

        # Extract key-value facts (lines with : or =)
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            if ':' in line and len(line) < 200:
                parts = line.split(':', 1)
                if len(parts) == 2 and len(parts[1].strip()) > 5:
                    tile = self._make_knowledge_tile(
                        content=line,
                        domain=domain,
                        content_type="fact",
                        model=model,
                        tags=["fact"],
                    )
                    tiles.append(tile)

        # Extract numbered lists (actionable steps)
        numbered = []
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) > 2 and stripped[0].isdigit() and '.' in stripped[:4]:
                numbered.append(stripped)
        if len(numbered) >= 2:
            tile = self._make_knowledge_tile(
                content='\n'.join(numbered),
                domain=domain,
                content_type="instruction",
                model=model,
                tags=["steps", "instruction"],
            )
            tiles.append(tile)

        # If nothing specific extracted, keep a compressed summary
        if not tiles and len(response) > 100:
            # Take first 200 chars as summary
            summary = response[:200].strip()
            tile = self._make_knowledge_tile(
                content=summary,
                domain=domain,
                content_type="summary",
                model=model,
                tags=["summary"],
            )
            tiles.append(tile)

        return tiles

    def _decide_keep(
        self,
        response: str,
        tiles: List[KnowledgeTile],
        domain: str,
    ) -> KeepDecision:
        """Decide how much of the response to keep."""
        if not tiles:
            return KeepDecision.DISCARD

        # Code-heavy responses → full keep
        code_tiles = [t for t in tiles if t.content_type == "code_snippet"]
        if len(code_tiles) >= 2:
            return KeepDecision.FULL

        # High-value domain → knowledge tile
        high_value = ["constraint-theory", "fleet", "plato", "math"]
        if domain in high_value and len(tiles) >= 2:
            return KeepDecision.KNOWLEDGE_TILE

        # Short response with facts → summary
        if len(response) < 500 and any(t.content_type == "fact" for t in tiles):
            return KeepDecision.SUMMARY

        return KeepDecision.SUMMARY

    def _make_knowledge_tile(
        self,
        content: str,
        domain: str,
        content_type: str,
        model: str,
        tags: List[str] = None,
    ) -> KnowledgeTile:
        """Create a knowledge tile with deduplication."""
        content_hash_val = hashlib.md5(content.encode()).hexdigest()[:12]
        tile_id = f"kn-{domain[:4]}-{content_type[:3]}-{content_hash_val}"

        # Check for duplicates
        for existing in self.knowledge.values():
            if existing.compressed_content == content:
                existing.touch()
                return existing

        # Compute novelty
        novelty = self._compute_novelty(content, domain)

        self.stats["knowledge_tiles_created"] += 1

        return KnowledgeTile(
            tile_id=tile_id,
            domain=domain,
            content_type=content_type,
            compressed_content=content,
            source_model=model,
            confidence=0.8,  # Default, updated by post-filter
            novelty=novelty,
            tags=tags or [],
        )

    def _compute_novelty(self, content: str, domain: str) -> float:
        """Compute how novel this content is vs existing knowledge.

        Uses semantic matcher if available, falls back to keyword overlap.
        """
        if not self.knowledge:
            return 1.0

        # Semantic novelty: 1 - max similarity from semantic matcher
        if self._semantic_matcher is not None and self._semantic_matcher.size() > 0:
            try:
                result = self._semantic_matcher.match(content)
                if result is not None:
                    _, score, _ = result
                    return max(0.0, 1.0 - score)
            except Exception:
                pass

        # Keyword overlap fallback
        words = set(content.lower().split())
        max_overlap = 0.0

        for tile in self.knowledge.values():
            if tile.domain != domain:
                continue
            tile_words = set(tile.compressed_content.lower().split())
            if not tile_words:
                continue
            overlap = len(words & tile_words) / max(len(words), 1)
            max_overlap = max(max_overlap, overlap)

        return 1.0 - max_overlap

    def _compress_response(
        self, response: str, tiles: List[KnowledgeTile],
    ) -> str:
        """Compress response to key knowledge."""
        if not tiles:
            return response[:200]

        parts = []
        for tile in tiles[:5]:  # Top 5 tiles
            parts.append(f"[{tile.content_type}] {tile.compressed_content[:100]}")

        return '\n'.join(parts)

    def _store_knowledge(self, tile: KnowledgeTile):
        """Store a knowledge tile in the room."""
        if tile.tile_id in self.knowledge:
            return  # Already stored (dedup)

        self.knowledge[tile.tile_id] = tile

        # Keep semantic index in sync
        if self._semantic_matcher is not None:
            try:
                text = f"{tile.domain} {tile.compressed_content}"
                self._semantic_matcher.add(tile.tile_id, text)
            except Exception:
                pass

        # Evict oldest/least-useful if over limit
        if len(self.knowledge) > self.max_knowledge:
            self._evict_knowledge()

    def _evict_knowledge(self):
        """Evict least useful knowledge tiles."""
        scored = [
            (tile.usefulness_score(), tile_id, tile)
            for tile_id, tile in self.knowledge.items()
        ]
        scored.sort(key=lambda x: x[0])

        # Remove bottom 10%
        n_evict = max(1, len(scored) // 10)
        for _, tile_id, _ in scored[:n_evict]:
            del self.knowledge[tile_id]

    # ─── Outcome Recording ────────────────────────────────────────

    def record_outcome(
        self,
        exp_id: str,
        outcome: str = "good",
        tokens_saved: int = 0,
        knowledge_reused: List[str] = None,
        user_satisfied: bool = True,
    ):
        """Record the outcome of a completed request."""
        exp = IntelligenceExperience(
            exp_id=exp_id,
            outcome=outcome,
            tokens_saved=tokens_saved,
            knowledge_reused=knowledge_reused or [],
            user_satisfied=user_satisfied,
        )
        self.experiences.append(exp)
        self.stats["tokens_saved"] += tokens_saved

        # Mark reused knowledge
        for tile_id in (knowledge_reused or []):
            if tile_id in self.knowledge:
                self.knowledge[tile_id].touch()

        # Trim experience buffer
        if len(self.experiences) > self.max_experience:
            self.experiences = self.experiences[-self.max_experience:]

    def create_experience(
        self,
        request_text: str,
        route_decision: Dict,
        response_text: str = "",
        model_used: str = "",
        latency_ms: float = 0.0,
    ) -> IntelligenceExperience:
        """Create an experience record for tracking."""
        exp_id = hashlib.md5(
            f"{request_text}:{time.time()}".encode()
        ).hexdigest()[:12]

        exp = IntelligenceExperience(
            exp_id=exp_id,
            request_hash=hashlib.md5(request_text.encode()).hexdigest()[:12],
            request_length=len(request_text),
            domain=route_decision.get("domain", "general"),
            route_taken=route_decision.get("decision", Route.USE_MEDIUM).name
                if isinstance(route_decision.get("decision"), Route)
                else str(route_decision.get("decision", "")),
            route_confidence=route_decision.get("confidence", 0.0),
            response_length=len(response_text),
            response_model=model_used,
            response_latency_ms=latency_ms,
        )
        self.experiences.append(exp)
        return exp

    # ─── Self-Training ────────────────────────────────────────────

    def self_train_cycle(self) -> Dict[str, Any]:
        """
        Run one self-training cycle.

        Returns metrics about the training, or empty dict if insufficient data.
        """
        try:
            from .intelligence_self_trainer import SelfTrainer
        except ImportError:
            return {"error": "self_trainer module not available"}

        trainer = SelfTrainer(store_dir=str(self.store.store_dir))

        # Feed accumulated experiences
        for exp in self.experiences:
            trainer.record_experience(exp)

        # Run training
        metrics = trainer.run_training_cycle()
        if metrics is None:
            return {"status": "skipped", "reason": "insufficient data"}

        self.stats["self_train_cycles"] += 1
        self.stats["last_self_train"] = time.time()

        return metrics

    # ─── Persistence ──────────────────────────────────────────────

    def _load_state(self):
        """Load knowledge and stats from disk."""
        state_path = self.store.store_dir / "intelligence_state.json"
        if state_path.exists():
            try:
                data = json.loads(state_path.read_text())
                self.stats = data.get("stats", self.stats)

                for kd in data.get("knowledge", []):
                    tile = KnowledgeTile(**kd)
                    self.knowledge[tile.tile_id] = tile

                for ed in data.get("experiences", []):
                    exp = IntelligenceExperience(**ed)
                    self.experiences.append(exp)
            except (json.JSONDecodeError, TypeError):
                pass

    def save_state(self):
        """Persist knowledge and stats to disk."""
        self.store.store_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.store.store_dir / "intelligence_state.json"

        data = {
            "stats": self.stats,
            "knowledge": [asdict(t) for t in self.knowledge.values()],
            "experiences": [
                asdict(e) for e in self.experiences[-1000:]  # Keep last 1000
            ],
        }
        state_path.write_text(json.dumps(data, indent=2, default=str))

    # ─── Status ───────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Current state of the intelligence room."""
        # Knowledge stats
        domain_counts = defaultdict(int)
        type_counts = defaultdict(int)
        for tile in self.knowledge.values():
            domain_counts[tile.domain] += 1
            type_counts[tile.content_type] += 1

        # Experience stats
        outcome_counts = defaultdict(int)
        for exp in self.experiences:
            outcome_counts[exp.outcome] += 1

        total_tokens = self.stats["tokens_saved"]
        llm_cost_per_1k = 0.005  # rough average
        dollars_saved = total_tokens * llm_cost_per_1k / 1000

        return {
            "total_requests": self.stats["total_requests"],
            "knowledge_tiles": len(self.knowledge),
            "knowledge_by_domain": dict(domain_counts),
            "knowledge_by_type": dict(type_counts),
            "experiences": len(self.experiences),
            "outcomes": dict(outcome_counts),
            "tokens_saved": total_tokens,
            "dollars_saved": f"${dollars_saved:.2f}",
            "llm_calls_saved": self.stats["llm_calls_saved"],
            "self_train_cycles": self.stats["self_train_cycles"],
            "avg_reuse": (
                sum(t.reuse_count for t in self.knowledge.values())
                / max(len(self.knowledge), 1)
            ),
        }

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars per token)."""
        return len(text) // 4
