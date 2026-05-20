# Eisenstein Encoder V2 — Fleet-Scale Training Results

**Date:** 2026-05-20 12:22

## Architecture

| Property | Value |
|----------|-------|
| Vocab size | 5000 (word + bigram hashing) |
| Embed dim | 32 |
| Output dim | 128 |
| Control points | 12 |
| Parameters | 160,536 |
| Size | 627.1 KB |
| Uses SplineLinear | True |

## Training

| Metric | Value |
|--------|-------|
| Triplets | 1,605 |
| Epochs | 150 |
| Loss | 0.6353 → 0.4996 |
| Time | 34.1s |
| LR schedule | Cosine 0.001 → 1e-05 |

### Training Curve
```
Epoch   Loss
   1    ████████████         0.6353
   10   █████████            0.4972
   20   ██████████           0.5012
   40   ██████████           0.5013
   60   █████████            0.4982
  150   █████████            0.4996
```

## Benchmark

**Task:** Given a text query, retrieve the correct repo from 1319 candidates.
**Queries:** 80 entries derived from actual corpus.

### Hit Rate (top-1)

| Method | Hits | Rate |
|--------|------|------|
| **Eisenstein V2** | 47/80 | **58.8%** |
| **Model2Vec** | 69/80 | **86.2%** |
| **Bitvector** | 69/80 | **86.2%** |

### Speed

| Method | Avg query |
|--------|-----------|
| Eisenstein V2 | 1.14 ms |
| Model2Vec | 0.08 ms |
| Bitvector | 0.14 ms |

### Size

| Method | Size |
|--------|------|
| Eisenstein V2 | 627.1 KB |
| Model2Vec | ~400 MB |
| Bitvector | ~0 KB |

## Key Findings

- Eisenstein V2 is **653x smaller** than Model2Vec
- Training on 1,605 triplets from fleet git history
- Used character bigram hashing for better tokenization
- Mixed real commit messages + synthetic domain descriptions

## Next Steps for V3

1. **BPE/WordPiece tokenization** — replace hash buckets with learned subword tokens
2. **Hard negative mining** — sample negatives that are currently confused by the model
3. **Larger training set** — include README content, code comments, file paths
4. **Multi-label** — some queries legitimately match multiple repos
5. **Evaluation** — test on actual retrieval: "find the commit that implements X"
