# Eisenstein Encoder V3 — Morphology-Aware Tokenization Results

**Date:** 2026-05-20
**Key change:** Morphology-aware hash tokenization — stems shared across word variants.

## Problem (V2 Bottleneck)

V2 used `hash(word) % vocab_size` for tokenization. This means:
- "deploy" → hash("deploy") % 5000 = 3247
- "deployment" → hash("deployment") % 5000 = 1891
- "deploying" → hash("deploying") % 5000 = 4102

These are **completely unrelated tokens** despite sharing the same root.

## Solution (V3)

V3 adds morphological stemming to the hash-based tokenization:
- "deploy" → [hash("deploy"), hash("dep"), hash("ep"), hash("pl"), hash("lo"), hash("oy")]
- "deployment" → [hash("deployment"), hash("deploy"), hash("dep"), hash("ep"), ...]
- "deploying" → [hash("deploying"), hash("deploy"), hash("dep"), hash("ep"), ...]

Now "deploy", "deployment", "deploying" all share `hash("deploy")` — the stemmed token.

### Tokenization Pipeline
1. **Word hash** — same as V2
2. **Stem hash** (NEW) — strip suffix, hash the stem
3. **Character bigrams** — same as V2

Suffixes stripped: -ing, -ment, -tion, -ation, -ness, -ity, -able, -ible, -ful, -less, -ance, -ence, -ize, -ise, -ate, -ify, -er, -ed, -es, -ly, -al, -ty, etc.

### BPE Investigation
BPE was explored extensively but proved ineffective:
- **BPE-5000**: Most words remain whole tokens — "deploy" and "deployment" are separate single tokens. No improvement.
- **BPE-800**: Forces subword splitting ("deploy" → ["dep", "lo", "y"]), but common subwords appear in ALL texts, destroying discriminative signal. Hit rate dropped to 2.5%.
- **BPE + bigrams hybrid**: 28.7% — better but still far behind V2's 60%.
- **Root cause**: BPE provides no implicit similarity signal. The hash+bigram approach works because shared bigrams between related texts create implicit similarity BEFORE training. BPE subword IDs are arbitrary and don't provide this.

The BPE tokenizer is retained in `models/fleet-bpe.json` for future PLATO module use.

## Architecture

| Property | V2 | V3 |
|----------|-----|-----|
| Tokenization | hash + bigram | hash + **stem** + bigram |
| Vocab size | 5,000 | 5,000 |
| Embed dim | 32 | 32 |
| Output dim | 128 | 128 |
| Control points | 12 | 12 |
| Parameters | 160,536 | 160,536 |
| Size | 627 KB | 627.1 KB |
| Uses SplineLinear | ✓ | ✓ |
| Avg token seq len | ~20 | ~35 |

## Training

| Metric | V2 | V3 |
|--------|-----|-----|
| Triplets | ~1,605 | 2,229 |
| Epochs | 150 | 150 |
| Loss start | 0.6353 | 0.6063 |
| Loss end | 0.4996 | 0.4961 |
| Time | 34s | 68.1s |
| LR schedule | Cosine | Cosine |

### Training Curve (V3)
```
Epoch   Loss
    1   ████████████         0.6063
   10   ██████████           0.4995
   20   █████████            0.4946
   30   ██████████           0.5012
   40   ██████████           0.5021
   50   ██████████           0.5007
   60   ██████████           0.5011
   70   █████████            0.4998
   80   ██████████           0.5023
   90   █████████            0.4945
  100   █████████            0.4952
  110   ██████████           0.5024
  120   ██████████           0.5011
  130   █████████            0.4988
  140   █████████            0.4981
  150   █████████            0.4961
```

## Benchmark

**Task:** Given a text query, retrieve the correct repo from 1,755 candidates.
**Queries:** 80 entries derived from actual corpus.

### Hit Rate (top-1)

| Method | Hits | Rate | vs V2 |
|--------|------|------|-------|
| **Eisenstein V3** | 57/80 | **71.2%** | +18.8% |
| **Eisenstein V2** | 48/80 | **60.0%** | baseline |
| **Model2Vec** | 71/80 | **88.8%** | — |
| **Bitvector** | 71/80 | **88.8%** | — |

### Speed

| Method | Avg query |
|--------|-----------|
| Eisenstein V3 | 3.42 ms |
| Eisenstein V2 | 1.57 ms |
| Model2Vec | 0.09 ms |
| Bitvector | 0.18 ms |

### Size

| Method | Size |
|--------|------|
| Eisenstein V3 | **627.1 KB** |
| Eisenstein V2 | 627.1 KB |
| Model2Vec | ~400 MB (**653x larger** than V3) |

## Key Findings

1. **V3 beats V2 by 18.8%** — from 60.0% to 71.2%, exceeding the 70% target
2. **Stem hashing fixes the morphological bottleneck** — related words now share tokens
3. **653x smaller than Model2Vec** at 71.2% vs 88.8% retrieval accuracy
4. **BPE is not the answer** for tiny models — the implicit similarity from hash-based tokenization is essential
5. **The training loss barely moves** in both V2 and V3 — performance comes from tokenization providing implicit similarity, not from the contrastive training
6. **3.42ms inference** — sub-5ms on CPU with 1,755 candidates

## Artifacts

| File | Description |
|------|-------------|
| `models/eisenstein-fleet-v3.pt` | Trained V3 encoder |
| `models/fleet-bpe.json` | BPE tokenizer (for future PLATO modules) |
| `plato_training/fleet_tokenizer.py` | Reusable BPE tokenizer module |
| `train_eisenstein_v3.py` | Training + benchmark script |

## Next Steps for V4

1. **Hard negative mining** — re-mine negatives the model currently confuses
2. **Larger corpus** — include README content, code comments, file paths
3. **Attention pooling** — replace mean pooling with learned attention weights
4. **Multi-scale features** — combine word, stem, bigram, and character-level signals
5. **Proper BPE with pre-trained embeddings** — use frozen Model2Vec embeddings to initialize BPE token embeddings
