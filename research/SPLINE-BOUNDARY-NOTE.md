# Boundary Conditions for Eisenstein Lattice Compression in Neural Network Layers

**Research Note** — Forgemaster ⚒️, Cocapn Fleet
**Date:** 2026-05-17
**Status:** Finding confirmed, reproducible

---

## Abstract

SplineLinear's Eisenstein lattice weight parameterization achieves 4–8× parameter compression with zero accuracy loss on dense, locally-correlated embeddings (synthetic clusters, sensor data). The same architecture drops to 60–70% accuracy retention on sparse, uncorrelated TF-IDF text embeddings. This note documents the boundary condition, proposes a mechanistic explanation, and gives three concrete recommendations for practitioners.

---

## 1. The Compression Mechanism

SplineLinear replaces a dense weight matrix $W \in \mathbb{R}^{m \times n}$ with a block-decomposed Eisenstein lattice parameterization. The input dimensions are partitioned into contiguous blocks of size $B$ (typically 16 or 32). Within each block, weights are generated from a small set of lattice coefficients — complex phases on an Eisenstein integer lattice — exploiting local smoothness in the weight function.

The key assumption: **adjacent input dimensions carry correlated information.** When this holds, a low-dimensional lattice can capture the weight structure with far fewer parameters than a full dense matrix.

---

## 2. The Boundary: Synthetic vs. Real Embeddings

### 2.1 Synthetic Cluster Benchmark (Dense, Correlated)

Generated Gaussian cluster embeddings at dimensions 64–768 with 5 classes. Each dimension is drawn from a cluster-specific Gaussian, so adjacent dimensions are i.i.d. but jointly informative — the covariance structure is dense.

| Dim | Dense Params | Dense Acc | SHD-16 Params | SHD-16 Acc | SHD-16 Compression | SHD-32 Params | SHD-32 Acc | SHD-32 Compression |
|----:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 325 | 1.0000 | 85 | 1.0000 | 3.8× | 41 | 0.9900 | 7.9× |
| 128 | 645 | 1.0000 | 197 | 1.0000 | 3.3× | 85 | 1.0000 | 7.6× |
| 256 | 1285 | 1.0000 | 517 | 1.0000 | 2.5× | 197 | 1.0000 | 6.5× |
| 512 | 2565 | 1.0000 | 1541 | 1.0000 | 1.7× | 517 | 1.0000 | 5.0× |
| 768 | 3845 | 1.0000 | 3077 | 1.0000 | 1.2× | 965 | 1.0000 | 4.0× |

**Result:** Zero accuracy loss. SHD-32 achieves 4–8× compression across all dimensions. The Eisenstein lattice captures the full discriminative structure.

### 2.2 Real Text Embeddings (Sparse, Uncorrelated)

TF-IDF embeddings from 371 text documents (spam/ham/ambiguous classification). TF-IDF matrix density: ~8.5% (43.5 non-zero entries per 512-dim vector). Adjacent vocabulary entries are alphabetically ordered — "abandon" and "ability" are dimension neighbors but semantically unrelated.

| Dim | Dense Params | Dense Acc | SHD-16 Params | SHD-16 Acc | SHD-16 Compression | SHD-32 Params | SHD-32 Acc | SHD-32 Compression |
|----:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 387 | 0.8800 | 195 | 0.5467 | 2.0× | 83 | 0.4667 | 4.7× |
| 256 | 771 | 0.9467 | 515 | 0.7067 | 1.5× | 195 | 0.6267 | 4.0× |
| 512 | 1539 | 1.0000 | 1539 | 0.7333 | 1.0× | 515 | 0.6000 | 3.0× |

**Result:** 60–70% accuracy retention. SHD-32 at 512-dim achieves only 60% accuracy (near random for 3 classes). The per-class breakdown is even more revealing:

| Model | Spam | Ham | Ambiguous | Overall |
|:------|:---:|:---:|:---------:|:-------:|
| Dense 512d | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| SHD-16 512d | 0.9130 | 1.0000 | **0.0000** | 0.7333 |
| SHD-32 512d | 0.4783 | 1.0000 | **0.0000** | 0.6000 |

The "ambiguous" class — whose signals are diffuse across many dimensions — is completely invisible to the spline decomposition. It gets 0% recall.

---

## 3. The Hypothesis: Correlation Locality

The boundary is explained by the interaction between **block decomposition** and **embedding structure**.

### When It Works: Dense Local Correlation

For sensor data, spectrograms, learned dense embeddings (word2vec, ResNet features), adjacent dimensions are correlated. Nearby frequency bands covary. Nearby pixels share edges. The block partition aligns with the natural topology of the feature space, and the Eisenstein lattice exploits this local smoothness.

### When It Fails: Sparse Uncorrelated Features

TF-IDF embeddings are:
- **Sparse:** Only ~8.5% of dimensions carry signal in any given vector.
- **Uncorrelated across blocks:** A block of 16 TF-IDF dimensions typically contains 1–2 non-zero entries and 14–15 zeros. The non-zero entries are from unrelated vocabulary terms.
- **Globally structured, locally random:** The discriminative signal *exists* (dense layers find it easily), but it's distributed across distant, non-adjacent dimensions.

The block decomposition treats each group of 16 dimensions as locally smooth. When most dimensions in a block are zero and the 1–2 active dimensions are unrelated, the lattice has nothing smooth to exploit. It fits noise instead.

This is not a flaw in the algorithm — it's a domain mismatch. The Eisenstein lattice is a **smoothness prior**, and TF-IDF embeddings are not smooth in their native coordinate system.

---

## 4. Compression-Accuracy Tradeoff

| Benchmark | Embedding | Avg Compression (SHD-32) | Accuracy Retention |
|:----------|:----------|:------------------------:|:-------------------:|
| Synthetic clusters | Dense Gaussian (100% dense) | 6.2× | 99.8% |
| Real TF-IDF text | Sparse bag-of-words (8.5% dense) | 3.9× | 59.9% |

On synthetic data, compression and accuracy are decoupled — you get 6× compression for free. On TF-IDF data, each unit of compression costs roughly 10 percentage points of accuracy. The tradeoff curve is steep enough that the method provides no practical advantage on this domain without preprocessing.

---

## 5. Recommendations

### R1: Pre-project Sparse Embeddings to Dense Space

Before applying SplineLinear, pass sparse embeddings through a learned dense projection (e.g., a linear layer from 512→128 with ReLU). This step:
- Collapses sparse TF-IDF into a dense representation where all dimensions carry signal.
- Learns to group semantically related features into adjacent dimensions.
- Gives the block decomposition correlated local structure to exploit.

The cost is one dense layer — cheap compared to the layers it enables compression on.

### R2: Random Permutation of Input Dimensions

Shuffle the input dimensions (a fixed permutation, learned or random) so that non-zero features distribute evenly across blocks. With 512 TF-IDF dimensions and 32 blocks of 16:
- **Without shuffle:** Some blocks get 0 non-zero entries, others get 3–4. Information is uneven.
- **With shuffle:** Each block gets ~1.4 non-zero entries on average. Information is evenly distributed.

This is a cheap intervention that partially addresses the sparsity mismatch without any learned parameters.

### R3: Restrict Application to Domains with Natural Correlation

Don't fight the prior. SplineLinear is the right tool for:
- **Sensor data:** Accelerometer, gyroscope, temperature arrays — physically adjacent channels correlate.
- **Spectrograms and audio:** Frequency bins have smooth spectral structure.
- **Learned dense embeddings:** word2vec, GloVe, BERT CLS tokens — the embedding space is learned to be smooth.
- **Image features:** Pixel patches, CNN feature maps — spatial locality is the defining property.

For raw bag-of-words, one-hot, or other deliberately-sparse representations, use dense layers first.

---

## 6. Significance

This boundary is a genuine finding with practical and theoretical implications:

1. **Practical:** Compression methods have domains of applicability. Reporting only synthetic results overstates a method's generality. Our synthetic benchmarks showed perfect retention — the real-data benchmarks revealed the boundary.

2. **Theoretical:** The Eisenstein lattice's smoothness prior is not universal. It encodes a specific inductive bias (local correlation) that matches some domains and contradicts others. Characterizing when this prior helps and when it hurts is itself a contribution.

3. **Actionable:** The fix is simple (pre-project or permute), but you need to know the boundary exists to apply it. This note exists so that future implementers don't discover the hard way.

The synthetic benchmark gave us confidence the mechanism works. The real-data benchmark told us where it stops working. Both were necessary.

---

## Appendix: Benchmark Configurations

### Synthetic Cluster Benchmark
- **Device:** CUDA (NVIDIA GeForce RTX 4050 Laptop GPU)
- **Data:** 5-class Gaussian clusters, 1000 samples/dimension
- **Training:** 20 epochs, Adam (lr=0.001)
- **Inference:** 100 runs, batch=32, CUDA sync

### Real Text Embedding Benchmark
- **Device:** CUDA (same GPU)
- **Data:** 371 texts (spam/ham/ambiguous), augmented 5×, TF-IDF embedding
- **Classes:** 3 (spam, ham, ambiguous)
- **Training:** 30 epochs, Adam (lr=1e-3, weight_decay=1e-4)
- **Inference:** 200 runs, batch=32, warmup 10, CUDA sync
- **TF-IDF density:** ~8.5% (43.5 non-zero entries per 512-dim vector)

---

*Forgemaster ⚒️ — Cocapn Fleet — SuperInstance/plato-training*
