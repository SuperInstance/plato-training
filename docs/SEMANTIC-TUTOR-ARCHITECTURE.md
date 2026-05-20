# PLATO Semantic Tutor Architecture

## The Problem

Current PLATO knowledge retrieval uses **keyword overlap** at 0.3 threshold.
The Knowledge Builder beta-test showed **50% cache hit rate** — half the time,
previously answered questions aren't found because the words don't match.

This is the "hard-coded if-then" problem. The system knows things but can't
*recognize* when a new question is asking the same thing differently.

## The Solution: Streamlined Vector Embeddings

Replace keyword matching with **semantic similarity**. Instead of checking
whether two strings share words, project them into a vector space where
*meaning* determines proximity.

### Architecture: 3-Layer Stack

```
Layer 3: Application — Intelligence Room, PreFilter routing, Tutor responses
                ↓ query
Layer 2: Retrieval — FAISS index + cosine similarity → top-k matches
                ↓ embeddings
Layer 1: Embedding — Model2Vec (static, 8MB, CPU, 500x faster than transformers)
```

### Why This Works for PLATO

| Constraint | Solution |
|-----------|----------|
| RTX 4050 (6GB) | All embedding on CPU, GPU free for training |
| No external database | FAISS in-memory, NumPy arrays, save/load to disk |
| Must be fast | Model2Vec: ~8MB, 500x faster than sentence-transformers |
| Must be small | Static embeddings: 256-dim, no transformer needed |
| Must work offline | Everything local, no API calls |
| Must improve over time | New knowledge auto-indexed, self-training adjusts thresholds |

## Layer 1: Embedding Models

### Option A: Model2Vec (RECOMMENDED — fastest, smallest)

```python
from model2vec import StaticModel

model = StaticModel.from_pretrained("minishlab/M2V_base_output")
# 8MB, CPU-only, ~0.01ms per embedding

query_emb = model.encode(["How many tests in plato-training?"])
# Returns 256-dim float32 vectors
```

- **Size:** 8-30MB
- **Speed:** Millions of embeddings per second on CPU
- **Quality:** 85-90% of sentence-transformer quality on semantic similarity
- **How it works:** Distills a sentence-transformer into static token embeddings.
  Each word gets a fixed vector. Sentence embedding = weighted average of token vectors.
  No transformer forward pass needed.
- **Perfect for:** Real-time retrieval, fleet-scale indexing, embedded deployment

### Option B: sentence-transformers/all-MiniLM-L6-v2 (HIGHEST QUALITY)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
# 80MB, CPU, ~5ms per embedding

query_emb = model.encode(["How many tests in plato-training?"])
# Returns 384-dim float32 vectors
```

- **Size:** 80MB
- **Speed:** ~5ms per embedding on CPU (200/s)
- **Quality:** Best general-purpose small model
- **Trade-off:** 500x slower than Model2Vec, but much better semantic understanding
- **Good for:** Offline batch indexing, or when quality matters more than speed

### Option C: Our Own SplineLinear Encoder (NOVEL — research target)

Train a tiny encoder using SplineLinear layers that maps text → 128-dim vectors.
Uses the same Eisenstein lattice compression as the rest of PLATO.

```python
class SplineEncoder(nn.Module):
    """128-dim encoder using SplineLinear for compression."""
    def __init__(self, vocab_size=5000, n_cp=8):
        self.embedding = nn.Embedding(vocab_size, 64)
        self.encoder = SplineLinear(64, 128, n_control_points=n_cp)
        self.projection = SplineLinear(128, 128, n_control_points=n_cp)

    def forward(self, token_ids):
        x = self.embedding(token_ids).mean(dim=1)  # bag-of-words
        h = torch.relu(self.encoder(x))
        return self.projection(h)  # 128-dim output
```

- **Size:** ~50KB (SplineLinear compression)
- **Speed:** Sub-millisecond
- **Quality:** Unknown — needs training on PLATO corpus
- **Novel factor:** Highest. This IS the research contribution.

### Recommendation

**Start with Option A (Model2Vec) for production, prototype Option C (SplineLinear) as research.**

Model2Vec gives us semantic retrieval today. SplineLinear encoder is the
paper-worthy contribution — if it works at even 80% of Model2Vec quality
at 1000x smaller footprint, that's a result.

## Layer 2: Retrieval (FAISS)

```python
import faiss
import numpy as np

class SemanticStore:
    """Vector store for PLATO knowledge tiles using FAISS."""

    def __init__(self, dim=256, embedder=None):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine if normalized
        self.embedder = embedder  # Model2Vec or sentence-transformer
        self.id_map = []  # Maps FAISS index → tile_id

    def add(self, tile_id: str, text: str):
        """Embed text and add to index."""
        vec = self.embedder.encode([text]).astype('float32')
        faiss.normalize_L2(vec)  # Normalize for cosine similarity
        self.index.add(vec)
        self.id_map.append(tile_id)

    def search(self, query: str, k=5, threshold=0.7) -> list:
        """Find semantically similar knowledge."""
        vec = self.embedder.encode([query]).astype('float32')
        faiss.normalize_L2(vec)
        scores, indices = self.index.search(vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= threshold:
                results.append({
                    "tile_id": self.id_map[idx],
                    "score": float(score),
                })
        return results

    def save(self, path: str):
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.ids", 'w') as f:
            for tid in self.id_map:
                f.write(tid + '\n')

    def load(self, path: str):
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.ids") as f:
            self.id_map = [l.strip() for l in f]
```

### Why FAISS over alternatives

| Need | FAISS | ChromaDB | Pinecone | NumPy brute-force |
|------|-------|----------|----------|-------------------|
| No external service | ✅ | ❌ | ❌ | ✅ |
| In-memory | ✅ | ✅ | ❌ | ✅ |
| Fast at 10K+ vectors | ✅ | ✅ | ✅ | ❌ (O(n) scan) |
| Save/load to disk | ✅ | ✅ | N/A | ✅ |
| Already installed | ✅ | ❌ | ❌ | ✅ |
| Python API | ✅ | ✅ | ✅ | ✅ |

FAISS is already in our environment. It's the right tool.

## Layer 3: Application — How This Transforms the Tutor

### Before (keyword matching, 50% hit rate):

```python
# User asks: "What's the test count for the training repo?"
# KB has: "plato-training has 665 tests"
# Keyword overlap("test count training repo", "plato-training 665 tests") = 0.15
# MISS — below 0.3 threshold
```

### After (semantic matching, ~90% hit rate):

```python
# User asks: "What's the test count for the training repo?"
# Embed → query vector
# FAISS search → cosine similarity with KB entries
# "plato-training has 665 tests" → cosine_sim = 0.87
# HIT — well above 0.7 threshold
```

The embedding captures that "test count" ≈ "how many tests" ≈ "test suite size"
≈ "testing numbers". The vector space generalizes across paraphrases.

### The Tutor Loop

```
Student Question
      ↓
  [Embed query] ──→ Model2Vec ──→ 256-dim vector
      ↓
  [FAISS search] ──→ top-k matches with similarity scores
      ↓
  ┌─ No match (score < 0.5) → Route to LLM, embed response, add to KB
  ├─ Weak match (0.5-0.7) → Route to LLM with context, compare, maybe add
  └─ Strong match (> 0.7) → Return cached answer, increment reuse count
      ↓
  [Post-filter] ──→ Was the answer useful? Record outcome.
      ↓
  [Self-train] ──→ Periodically retrain routing on outcomes
```

### What This Enables Beyond Caching

1. **Knowledge Graph Navigation**: "Tell me more about that" → FAISS search
   for semantically related tiles near the current one in embedding space.

2. **Misconception Detection**: Student says something wrong → embed it →
   find nearest correct knowledge tile → compute distance. If distance is
   small, they're close but confused. If large, they're off track.

3. **Progressive Difficulty**: Questions cluster in embedding space by topic.
   Track which clusters the student has mastered → serve questions from
   unexplored regions.

4. **Cross-Domain Transfer**: "How is SplineLinear like deadband?" →
   Both live in constraint-theory embedding cluster → FAISS finds the
   connection automatically.

5. **Fleet Knowledge Sharing**: Each agent embeds its knowledge → export
   FAISS index → other agents can query it. No shared database needed.

## Implementation Plan

### Phase 1: Drop-in Replacement (1 day)
Replace keyword matching in IntelligenceRoom with Model2Vec + FAISS.
Same API, better retrieval. Should jump from 50% → 85%+ hit rate.

### Phase 2: SplineLinear Encoder Research (2-3 days)
Train a tiny SplineLinear-based encoder on the PLATO corpus.
Compare against Model2Vec on our specific retrieval task.
If competitive → paper-worthy result on Eisenstein-structured embeddings.

### Phase 3: Tutor Loop (1 day)
Build the full student-facing loop: query → retrieve → respond → learn.
Wire into PLATO rooms as a "tutor" room type.

### Phase 4: Knowledge Graph (2 days)
Build a proximity graph in embedding space. Each tile knows its neighbors.
Enables "tell me more" and "what's related" navigation.

### Phase 5: Fleet Export (1 day)
Export FAISS indices as I2I tiles. Agents share knowledge via git.
No central vector database needed.

## Key Insight: Why This Is More Than If-Then

A traditional tutor has:
```
IF question == "What is X?" THEN answer = "X is..."
IF question == "How does Y work?" THEN answer = "Y works by..."
```
Fragile. Can't handle paraphrases, typos, or novel questions.

A vector-embedded tutor has:
```
question → embed → find_nearest(knowledge) → respond
```
- Handles paraphrases (same vector neighborhood)
- Handles novel questions (low similarity → route to LLM → learn)
- Handles typos (embedding is robust to noise)
- Handles cross-topics (vector space captures relationships)
- **Improves over time** (new knowledge → new vectors → better coverage)

The embedding space IS the tutor's understanding. Not hardcoded rules —
a learned map of meaning.

## Dependencies

Already installed:
- `faiss-cpu` 1.13.2
- `sentence-transformers` 5.5.0
- `numpy` 2.2.6
- `torch` 2.12.0+cu126
- `sklearn` 1.7.2

Need to install:
- `model2vec` (pip install model2vec) — 8MB static embeddings

## Estimated Impact

| Metric | Before (keywords) | After (embeddings) |
|--------|-------------------|---------------------|
| Cache hit rate | 50% | 85-90% |
| Retrieval latency | ~1ms (keyword scan) | ~1ms (FAISS search) |
| Memory per 1000 tiles | ~50KB (text) | ~1MB (256-dim vectors) |
| Handles paraphrases | No | Yes |
| Handles typos | No | Yes (embedding robustness) |
| Cross-domain transfer | No | Yes (vector proximity) |
| Improves with data | Only linearly | Exponentially (more vectors = better coverage) |

The biggest win: **the system gets smarter the more it's used**.
Every question that routes to an LLM and gets answered adds a new vector
to the space. The next similar question hits cache. Over time, the tutor
builds a dense map of its domain — and the LLM is needed less and less.

This is the bridge from "caching system" to "learning system".
