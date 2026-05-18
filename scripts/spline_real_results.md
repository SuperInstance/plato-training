# SplineLinearHD on REAL Text Embeddings

## Benchmark Setup

Does SplineLinearHD maintain its compression advantage on **real noisy text embeddings**
from the PLATO ecosystem?

- **Device:** cuda (NVIDIA GeForce RTX 4050 Laptop GPU)
- **Dataset:** 25 spam + 25 ham + 15 ambiguous email-like texts from the spreader-tool
  pipeline fixtures, plus 46 paragraph/docstring chunks from PLATO repo READMEs and
  source code (plato-types, tensor-spline, plato-data, plato-training).
- **Augmentation:** Each text duplicated and shuffled 5× for variety.
- **Embedding:** Simple TF-IDF (bag-of-words + IDF weighting), no sklearn dependency.
  Features padded with small noise to reach target dimension.
- **Classes:** 3 — spam (0), ham (1), ambiguous (2)
- **Train/Test:** 80/20 split
- **Epochs:** 30
- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-4)
- **Inference:** 200 runs batch=32, warmup 10, CUDA sync

## Results Table

| Dim | Dense Params | Dense Acc | Dense Time (ms) | SHD-16 Params | SHD-16 Acc | SHD-16 Ratio | SHD-16 Time (ms) | SHD-32 Params | SHD-32 Acc | SHD-32 Ratio | SHD-32 Time (ms) |
|----:|:-----------:|:--------:|:--------------:|:-----------:|:--------:|:------------:|:---------------:|:-----------:|:--------:|:------------:|:---------------:|
| 128 | 387 | 0.8800 | 0.019 | 195 | 0.5467 | 2.0x | 1.543 | 83 | 0.4667 | 4.7x | 1.221 |
| 256 | 771 | 0.9467 | 0.019 | 515 | 0.7067 | 1.5x | 6.739 | 195 | 0.6267 | 4.0x | 3.407 |
| 512 | 1539 | 1.0000 | 0.088 | 1539 | 0.7333 | 1.0x | 11.140 | 515 | 0.6000 | 3.0x | 3.742 |

## Per-Class Accuracy (512-dim)

| Model | Spam | Ham | Ambiguous | Avg |
|:-----|:---:|:---:|:--------:|:---:|
| Dense | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| SHD-16 | 0.9130 | 1.0000 | 0.0000 | 0.7333 |
| SHD-32 | 0.4783 | 1.0000 | 0.0000 | 0.6000 |

## Summary

| Metric | Dense | SHD-16 | SHD-32 |
|:------|:-----:|:------:|:------:|
| Avg Accuracy | 0.9422 | 0.6622 | 0.5644 |
| Accuracy Retention | — | 70.3% | 59.9% |
| Avg Compression | — | 1.5× | 3.9× |
| Inference (512d, ms) | 0.088 | 11.140 | 3.742 |

## Key Findings

### 1. SplineLinearHD Cannot Compress Sparse TF-IDF Embeddings

The per-class breakdown tells the story: **SHD-16 gets 0% on the "ambiguous" class** at
512-dim. The ambiguous class (emails with neutral or unclear intent) occupies a diffuse
region of the embedding space that requires nearly all 512 dimensions to discriminate.
SHD-16's block decomposition (32 blocks of 16 dims each) obscures the global patterns
that distinguish ambiguous from ham.

At 128-dim, the model is just barely above random (0.5467 on 3 classes = 0.333 baseline).
At 512-dim with SHD-32, the "ambiguous" class is completely invisible.

### 2. Root Cause: Sparsity × Block Decomposition

TF-IDF embeddings are ~8.5% dense (≈43.5 non-zero entries per 512-dim vector). The
Eisenstein lattice parameterization in each block assumes **local correlations** among
adjacent features — a reasonable assumption for image features or dense embeddings, but
**invalid for bag-of-words TF-IDF** where adjacent TF-IDF dimensions (sorted by count)
are unrelated.

When a block has 16 dims of which only 1–2 carry signal, the lattice cannot learn useful
local structure. The remaining dims become noise that dilutes the Eisenstein phase.

### 3. Comparison to Synthetic Cluster Results

In the synthetic benchmark (`benchmark_spline_hd.py`), SplineLinearHD achieved:
- SHD-16: ~93–97% accuracy retention with 2–7× compression
- SHD-32: ~88–96% accuracy retention with 4–15× compression

On real text embeddings:
- SHD-16: 70% accuracy retention with 1.5× compression
- SHD-32: 60% accuracy retention with 3.9× compression

The gap is **not** about noisy vs clean data — it's about **embedding structure**.
Synthetic cluster embeddings (Gaussian blobs around centroids) have dense local
correlation that aligns with the Eisenstein block decomposition. Real TF-IDF
embeddings are sparse and unaligned with block boundaries.

### 4. When Does SplineLinearHD Work?

SplineLinearHD benefits from embeddings where:
- **Adjacent features are correlated** (dense local structure)
- **Signal is distributed** across most dimensions (not sparse)
- **Block boundaries align** with natural feature groups (e.g., perceptual bands)

This includes: pixel patches, spectrogram windows, sensor time-series, and dense
word embeddings like word2vec or GloVe at the token level.

### 5. Path Forward for Text

For sparse NLP embeddings, two options:

1. **Pre-project to dense space** — Apply a learned linear projection (e.g., 512→128
   dense) before the SplineLinearHD block decomposition. This collapses sparse TF-IDF
   into a dense representation that the lattice can exploit.

2. **Random permutation** — Shuffle embedding dimensions so non-zero features are
   distributed evenly across blocks. With 512 TF-IDF dims and 32 blocks of 16, this
   ensures each block sees ~1.4 non-zero entries on average (vs 0–2 without shuffle).

### Data Characteristics

- 371 texts after augmentation
- TF-IDF matrix: 8.5% density (43.5 non-zero entries/row in 512-dim)
- Words/vocab: 512 features from 371 docs (fit to match target dim)
- Small dataset — larger corpora would give more signal per dimension
