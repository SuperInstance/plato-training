"""
Semantic Matcher — Drop-in replacement for keyword-overlap matching in IntelligenceRoom.

Uses Model2Vec static embeddings + FAISS inner-product search for semantic similarity.
Falls back to keyword matching if dependencies are unavailable.

Integration into IntelligenceRoom.pre_route()
=============================================

Replace _check_knowledge() with this pattern:

    class IntelligenceRoom:
        def __init__(self, ...):
            ...
            # After self.knowledge dict is set up:
            self._semantic = SemanticMatcher(threshold=0.6)
            # Re-index existing knowledge
            for key, tile in self.knowledge.items():
                text = f"{tile.domain} {tile.compressed_content}"
                self._semantic.add(key, text)

        def _check_knowledge(self, request_text, domain):
            # Try semantic match first
            sem = self._semantic.match(request_text)
            if sem is not None:
                key, score, _ = sem
                tile = self.knowledge.get(key)
                if tile is not None:
                    tile.touch()
                    self.stats["knowledge_tiles_reused"] += 1
                    return tile

            # Fallback to keyword overlap (existing code)
            return self._check_knowledge_keyword(request_text, domain)

        def _add_knowledge(self, tile):
            self.knowledge[tile.tile_id] = tile
            # Keep semantic index in sync
            self._semantic.add(tile.tile_id, f"{tile.domain} {tile.compressed_content}")

Key integration points:
  1. __init__: Create SemanticMatcher, index existing knowledge
  2. _check_knowledge: Try semantic → fallback keyword
  3. _add_knowledge (wherever tiles are added): Keep semantic index in sync
  4. novelty scoring (_compute_novelty): Use semantic.match() instead of overlap
"""



from __future__ import annotations
import gc
import threading
__all__ = ['SemanticMatcher']
from typing import Dict, List, Optional, Tuple


class SemanticMatcher:
    """
    Drop-in replacement for keyword-overlap matching.
    Uses Model2Vec embeddings + FAISS for semantic similarity.

    Falls back to keyword matching if Model2Vec/FAISS unavailable.
    """

    def __init__(self, dim: int = 256, threshold: float = 0.6):
        self.threshold = threshold
        self._lock = threading.Lock()
        self._embedder = None
        self._index = None
        self._id_map: List[str] = []
        self._text_map: Dict[str, str] = {}
        self._dim = dim
        self._use_semantic = False

        try:
            from model2vec import StaticModel
            self._embedder = StaticModel.from_pretrained("minishlab/M2V_base_output")
            test = self._embedder.encode(["test"])
            self._dim = test.shape[1]
            import faiss
            self._index = faiss.IndexFlatIP(self._dim)
            self._use_semantic = True
        except Exception:
            self._use_semantic = False

    @property
    def is_semantic(self) -> bool:
        return self._use_semantic

    def add(self, key: str, text: str) -> None:
        """Index a knowledge entry."""
        with self._lock:
            if self._use_semantic:
                vec = self._embedder.encode([text]).astype("float32")
                import faiss
                faiss.normalize_L2(vec)
                self._index.add(vec)
                self._id_map.append(key)
                self._text_map[key] = text

    def match(self, query: str) -> Optional[Tuple[str, float, str]]:
        """
        Find best matching knowledge entry.
        Returns (key, score, text) or None if no match above threshold.
        """
        with self._lock:
            if not self._use_semantic or self._index.ntotal == 0:
                return None

            vec = self._embedder.encode([query]).astype("float32")
            import faiss
            faiss.normalize_L2(vec)
            scores, indices = self._index.search(vec, 1)

            if indices[0][0] >= 0 and scores[0][0] >= self.threshold:
                key = self._id_map[indices[0][0]]
                return (key, float(scores[0][0]), self._text_map[key])
            return None

    def match_top_k(self, query: str, k: int = 5) -> List[Tuple[str, float, str]]:
        """Find top-k matches above threshold."""
        with self._lock:
            if not self._use_semantic or self._index.ntotal == 0:
                return []

            vec = self._embedder.encode([query]).astype("float32")
            import faiss
            faiss.normalize_L2(vec)
            scores, indices = self._index.search(vec, min(k, self._index.ntotal))

            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if idx >= 0 and scores[0][i] >= self.threshold:
                    key = self._id_map[idx]
                    results.append((key, float(scores[0][i]), self._text_map[key]))
            return results

    def size(self) -> int:
        with self._lock:
            return self._index.ntotal if self._use_semantic else 0

    def keyword_fallback(self, query: str, texts: Dict[str, str]) -> Optional[Tuple[str, float, str]]:
        """
        Simple keyword overlap fallback (mirrors IntelligenceRoom._check_knowledge logic).
        """
        if not texts:
            return None
        query_words = set(query.lower().split())
        best = None
        best_score = 0.0
        for key, text in texts.items():
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words) / max(len(query_words), 1)
            if overlap > best_score and overlap > 0.3:
                best_score = overlap
                best = (key, overlap, text)
        return best

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self._dim}, threshold={self.threshold!r}, indexed={self.size()})"

