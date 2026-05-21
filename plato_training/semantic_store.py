"""
Semantic retrieval layer for PLATO's Intelligence Room.
Replaces keyword-overlap matching with Model2Vec embeddings + FAISS.
Target: 85%+ cache hit rate (up from 50% with keywords).
"""


__all__ = ['SemanticStore', 'keyword_match']

import gc
import json
import time
from typing import Dict, List, Optional

try:
    import faiss
except ImportError:  # optional dep
    faiss = None
import numpy as np


class SemanticStore:
    """
    Vector store for PLATO knowledge using Model2Vec embeddings + FAISS.

    Replaces keyword-overlap matching with semantic similarity.
    Target: 85%+ cache hit rate (up from 50% with keywords).
    """

    def __init__(self, dim: int = 256, model_name: str = "minishlab/M2V_base_output"):
        from model2vec import StaticModel

        self.embedder = StaticModel.from_pretrained(model_name)
        test = self.embedder.encode(["test"])
        self.dim = test.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.id_map: List[str] = []
        self.text_map: Dict[str, str] = {}

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        vec = self.embedder.encode([text]).astype("float32")
        faiss.normalize_L2(vec)
        return vec[0]

    def add(self, tile_id: str, text: str):
        """Add a knowledge tile to the index. Updates if tile_id exists."""
        if tile_id in self.text_map:
            # Find and remove old entry
            idx = self.id_map.index(tile_id)
            # Rebuild index without the old entry
            old_texts = []
            old_ids = []
            for i, tid in enumerate(self.id_map):
                if i != idx:
                    old_ids.append(tid)
                    old_texts.append(self.text_map[tid])
            self.index.reset()
            self.id_map = []
            self.text_map = {}
            # Re-add everything except the old one, then add updated
            for tid, txt in zip(old_ids, old_texts):
                vec = self.embed(txt).reshape(1, -1)
                self.index.add(vec)
                self.id_map.append(tid)
                self.text_map[tid] = txt

        vec = self.embed(text).reshape(1, -1)
        self.index.add(vec)
        self.id_map.append(tile_id)
        self.text_map[tile_id] = text

    def search(self, query: str, k: int = 5, threshold: float = 0.3) -> List[Dict]:
        """Search for semantically similar knowledge."""
        if self.index.ntotal == 0:
            return []
        vec = self.embed(query).reshape(1, -1)
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= threshold:
                tile_id = self.id_map[idx]
                results.append(
                    {
                        "tile_id": tile_id,
                        "score": float(score),
                        "text": self.text_map[tile_id],
                    }
                )
        return results

    def best_match(self, query: str, threshold: float = 0.3) -> Optional[Dict]:
        """Find the single best match above threshold."""
        results = self.search(query, k=1, threshold=threshold)
        return results[0] if results else None

    def size(self) -> int:
        return self.index.ntotal

    def save(self, path: str):
        """Save index and metadata to disk."""
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.meta", "w") as f:
            json.dump(
                {"id_map": self.id_map, "text_map": self.text_map, "dim": self.dim}, f
            )

    def load(self, path: str):
        """Load index and metadata from disk."""
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.meta") as f:
            meta = json.load(f)
            self.id_map = meta["id_map"]
            self.text_map = meta["text_map"]
            self.dim = meta["dim"]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim!r}, model_name={self.model_name!r})"



def keyword_match(query: str, texts: Dict[str, str], threshold: float = 0.3) -> List[Dict]:
    """Baseline keyword-overlap matching (current system)."""
    q_words = set(query.lower().split())
    results = []
    for tile_id, text in texts.items():
        t_words = set(text.lower().split())
        overlap = len(q_words & t_words) / max(len(q_words), 1)
        if overlap >= threshold:
            results.append({"tile_id": tile_id, "score": overlap, "text": text})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
