# Eisenstein Encoder — Real Fleet Data Training Results

**Date:** 2026-05-20 12:07
**Model:** eisenstein-fleet-v1.pt
**Training data:** 49 commits from 5 repos

## Model Architecture

| Property | Value |
|----------|-------|
| Vocab size | 2,000 (hash buckets) |
| Embed dim | 32 |
| Output dim | 64 |
| Parameters | 64,272 |
| Model size | 257,088 bytes (251.1 KB) |
| Uses SplineLinear | True |

## Training

| Metric | Value |
|--------|-------|
| Training triplets | 69 |
| Epochs | 50 |
| Initial loss | 0.6938 |
| Final loss | 0.0160 |
| Loss reduction | 97.7% |
| Training time | 15.0s |

### Repos Used for Training

- **plato-training**: 37 commits
- **tensor-spline**: 4 commits
- **constraint-theory-py**: 2 commits
- **deadband-rs**: 1 commits
- **spectral-conservation**: 5 commits

## Benchmark Results

**Task:** Given a technical query, find the correct repo from 67 candidates.
**Queries:** 50 domain-specific queries across 5 repos.

### Hit Rate (top-1 accuracy)

| Method | Hits | Rate |
|--------|------|------|
| **Eisenstein Encoder** | 16/50 | **32.0%** |
| **Model2Vec** (minishlab/M2V_base_output) | 46/50 | **92.0%** |
| **Bitvector** (hash-based) | 38/50 | **76.0%** |

### Inference Speed

| Method | Avg query time | Encode all candidates |
|--------|---------------|----------------------|
| Eisenstein Encoder | 1.08 ms | 0.00s |
| Model2Vec | 0.04 ms | 0.02s |
| Bitvector | 0.02 ms | ~instant |

### Model Size Comparison

| Method | Size |
|--------|------|
| **Eisenstein Encoder** | **251.1 KB** (64,272 params) |
| Model2Vec | ~211 MB (full model on disk) |
| Bitvector | ~0 KB (no learned parameters) |

## Analysis

### Size vs Quality Trade-off

The Eisenstein encoder achieves this at **862x smaller** than Model2Vec.

### Key Findings

1. **Compression:** Eisenstein encoder is 251KB vs Model2Vec's ~211MB — a **862x** size reduction.
2. **Quality:** Hit rate of 32.0% vs Model2Vec's 92.0% — needs improvement.
3. **Speed:** Eisenstein at 1.1ms/query vs Model2Vec at 0.0ms/query.

### Next Steps

- Scale training data: more repos, more epochs
- Add subword tokenization instead of simple hash buckets
- Fine-tune margin and learning rate
- Test on cross-repo retrieval (find relevant code across fleet)
- Quantize to INT8 for even smaller footprint

## Files

- Trained model: `models/eisenstein-fleet-v1.pt`
- Training script: `train_eisenstein_benchmark.py`
- Encoder source: `plato_training/eisenstein_encoder.py`
