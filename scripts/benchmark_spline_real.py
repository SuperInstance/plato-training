#!/usr/bin/env python3
"""
Benchmark SplineLinearHD on REAL text embeddings from PLATO ecosystem data.

Generates TF-IDF embeddings from real email-like text (spam/ham fixtures)
and actual README/documentation content from the PLATO repos.

Key question: Does SplineLinearHD maintain its compression advantage on
REAL noisy text embeddings, or only on clean synthetic clusters?
"""
import sys
import os
import re
import time
import math
import json
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plato_training.spline_hd import SplineLinearHD

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

CONFIG = {
    'n_epochs': 30,
    'lr': 1e-3,
    'batch_size': 64,
    'n_classes': 3,  # spam, ham, ambiguous
    'inference_runs': 200,
}

N_EPOCHS = CONFIG['n_epochs']
LR = CONFIG['lr']
BATCH_SIZE = CONFIG['batch_size']
N_CLASSES = CONFIG['n_classes']
INFERENCE_RUNS = CONFIG['inference_runs']


# ──────────────────────────────────────────────────────────────────────
# 1. REAL TEXT DATA — from spreader-tool email fixtures + READMEs
# ──────────────────────────────────────────────────────────────────────

SPAM_TEXTS = [
    "FREE MONEY NOW CLICK HERE Act now limited time offer $500 guarantee winner prize",
    "You have won a FREE cruise! Click here to claim your prize now! Limited time offer.",
    "Nigerian inheritance fund transfer available. Wire transfer needed for processing.",
    "Buy cheap Viagra and Cialis online. No prescription needed. Pharmacy direct.",
    "Congratulations! You are our grand prize winner. Claim your $10,000 now!",
    "Unsubscribe from our mailing list or click here for exclusive deals.",
    "URGENT: Your account has been compromised. Verify immediately with your password.",
    "Make $5000 a week working from home! No experience necessary. Start today!",
    "Dear customer, your PayPal account needs verification. Click link below.",
    "Limited time offer! Get your FREE iPhone 15 Pro! Just pay shipping and handling.",
    "ACT NOW! This offer expires in 24 hours. Don't miss out on this amazing opportunity!",
    "Your tax refund is ready. Click here to claim your $2,873 refund immediately.",
    "Single women in your area want to meet you! Click here to browse profiles for free.",
    "You have been selected for a FREE vacation to the Bahamas! All expenses paid.",
    "Get rich quick with this proven system. Our clients earn $10k+ monthly guaranteed.",
    "We've detected unusual activity on your Netflix account. Verify login credentials.",
    "Secret investment opportunity - insiders only. Double your money in 30 days.",
    "Your computer has been infected with a virus. Download our scanner immediately.",
    "Lose weight fast with our miracle supplement! Doctors hate this one weird trick.",
    "Claim your free $500 Amazon gift card. Only 100 winners selected worldwide!",
    "Promotional offer - buy one get one free on all electronics! This week only.",
    "Your Microsoft account has been locked due to suspicious activity. Click to unlock.",
    "Dear email user, your mailbox is full. Click here to prevent deletion of messages.",
    "Exclusive crypto investment opportunity! Get 1000x returns in one week!",
    "Your order has shipped. Track your package here: [malicious link]",
]

HAM_TEXTS = [
    "Re: Meeting tomorrow Let's sync at 3pm to review the quarterly report.",
    "Thanks again for the attached report. Please review the Q3 financials before Friday.",
    "Invoice #4589 is attached for your reference. Payment due in 30 days.",
    "Could we push our lunch to 1pm? I have a conflict with the standup meeting.",
    "[Forgemaster] Updated the constraint theory module. PR ready for review.",
    "Following up on the architecture decision for the PLATO training pipeline.",
    "Would it be possible to reschedule our sync to Thursday instead of Wednesday?",
    "Let's grab coffee next week to discuss the Eisenstein lattice implementation.",
    "The CI pipeline is failing on the spline_hd tests. Can someone take a look?",
    "Meeting notes from today's sprint planning. Action items attached.",
    "I've updated the MEMORY.md with the new PLATO room mapping. Please review.",
    "Can you review the PR for tensor-spline? The block decomposition needs attention.",
    "The daily standup is at 10am in the usual channel. Agenda attached.",
    "Just pushed the new agent skill for weather forecasting. Integration tests passing.",
    "We need to discuss the hardware targets for the next epoch milestone.",
    "Azure deployment of drift-detect micro-model completed successfully.",
    "Patch release v0.3.1 - fixes the Lamport clock drift in plato-types.",
    "Sprint retrospective notes from last week. Action items for next sprint.",
    "The benchmark results show 20x compression on CPU-tiny. Worth a deeper look.",
    "Please update the documentation for the SplineLinearHD module before the release.",
    "Your expense report #1042 has been approved. Reimbursement will be processed next week.",
    "Reminder: Code freeze starts Friday at 5pm. All PRs must be merged by then.",
    "Onboarding docs for the new fleet agents are available in the wiki.",
    "The production pipeline is running at 98% accuracy on the test set. Shipping soon.",
    "Updated the roadmap for Q4. Key milestones: spline scaling and real data pipelines.",
]

AMBIGUOUS_TEXTS = [
    "Hello friend, check this out sometime. Could be interesting for your project.",
    "I came across this article and thought of you. Let me know what you think.",
    "Forwarding this message from someone who reached out. Not sure if relevant.",
    "Just a heads up about the system maintenance window this weekend.",
    "Can you take a look at this when you get a chance? No rush.",
    "We should discuss this further. Let me know your availability next week.",
    "FYI - the document you asked about is in the shared drive. Let me know if you need help.",
    "There's been some discussion about changing the process. Thoughts?",
    "Received this notification but I'm not sure what to make of it. Opinions?",
    "Quick question about the deployment schedule. Are we still on track?",
    "Saw your post on the forum and wanted to connect. I work in a similar field.",
    "I'm forwarding this to the right person. Someone will get back to you.",
    "Not sure if this is relevant but thought I'd share it anyway.",
    "Let me check with the team and get back to you on this.",
    "Can we set up a time to discuss the proposal? I have some availability next week.",
]


# ──────────────────────────────────────────────────────────────────────
# 2. ADDITIONAL REAL ECOSYSTEM TEXT — from PLATO READMEs & docs
# ──────────────────────────────────────────────────────────────────────

def read_repo_texts():
    """Read actual content from PLATO repo files for real embeddings."""
    texts = []
    sources = []

    repo_paths = [
        "/home/phoenix/.openclaw/workspace/plato-types/README.md",
        "/home/phoenix/.openclaw/workspace/tensor-spline/README.md",
        "/home/phoenix/.openclaw/workspace/plato-data/README.md",
        "/home/phoenix/.openclaw/workspace/plato-training/README.md",
    ]

    for path in repo_paths:
        if os.path.exists(path):
            with open(path) as f:
                content = f.read()
            # Split into paragraphs
            paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 50]
            texts.extend(paragraphs)
            sources.extend([os.path.basename(os.path.dirname(path))] * len(paragraphs))

    # Also grab some actual PLATO types source for code-level text
    code_files = [
        "/home/phoenix/.openclaw/workspace/plato-training/plato_training/types.py",
        "/home/phoenix/.openclaw/workspace/plato-training/plato_training/spline_hd.py",
    ]
    for path in code_files:
        if os.path.exists(path):
            with open(path) as f:
                content = f.read()
            # Extract docstrings
            docstrings = re.findall(r'"""(.*?)"""', content, re.DOTALL)
            for ds in docstrings[:3]:  # max 3 docstrings per file
                if len(ds.strip()) > 50:
                    texts.append(ds.strip())
                    sources.append(f"{os.path.basename(os.path.dirname(path))}/docstring")

    print(f"  Read {len(texts)} text chunks from PLATO repos")
    return texts, sources


# ──────────────────────────────────────────────────────────────────────
# 3. TF-IDF EMBEDDER (simple, no sklearn dependency)
# ──────────────────────────────────────────────────────────────────────

class SimpleTfIdf:
    """Bag-of-words + TF-IDF weighting. Produces dim-dimensional embeddings."""

    def __init__(self, dim: int = 256, max_features: int = 2000):
        self.dim = dim
        self.max_features = max_features
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.doc_count = 0

    def _tokenize(self, text: str) -> list[str]:
        words = re.findall(r'[a-zA-Z_]+', text.lower())
        return [w for w in words if len(w) > 1 and not w.isdigit()]

    def fit(self, texts: list[str]):
        """Build vocabulary and IDF from corpus."""
        doc_freq: dict[str, int] = Counter()
        all_tokens: list[str] = []

        for text in texts:
            tokens = self._tokenize(text)
            unique = set(tokens)
            for t in unique:
                doc_freq[t] += 1
            all_tokens.extend(tokens)

        # Sort by frequency, take top max_features
        word_counts = Counter(all_tokens)
        top_words = [w for w, _ in word_counts.most_common(self.max_features)]

        self.vocab = {w: i for i, w in enumerate(top_words)}
        num_words = len(top_words)
        self.doc_count = len(texts)

        # Compute IDF
        for w in top_words:
            df = doc_freq.get(w, 1)
            self.idf[w] = math.log((self.doc_count + 1) / (df + 1)) + 1

        print(f"  TF-IDF vocab: {num_words} words from {self.doc_count} docs")

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform texts to (N, dim) TF-IDF vectors. Padded with noise."""
        n = len(texts)
        result = np.zeros((n, self.dim), dtype=np.float32)

        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            if not tokens:
                continue
            tf = Counter(tokens)
            max_tf = max(tf.values())

            # Fill first min(len(vocab), dim) dimensions with TF-IDF
            for word, count in tf.items():
                if word in self.vocab and self.vocab[word] < self.dim:
                    tfidf_value = (count / max_tf) * self.idf.get(word, 1.0)
                    result[i, self.vocab[word]] = tfidf_value

            # Fill remaining dims with small noise
            noise_dim = min(self.dim, 32)
            result[i, -noise_dim:] += np.random.randn(noise_dim) * 0.01

        return result


# ──────────────────────────────────────────────────────────────────────
# 4. Datasets
# ──────────────────────────────────────────────────────────────────────

def build_real_dataset(
    dim: int = 256,
    include_docs: bool = True,
    n_augment: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a real text classification dataset with TF-IDF embeddings.

    Args:
        dim: Embedding dimension (TF-IDF features + padding).
        include_docs: Include PLATO repo readme/docstring chunks.
        n_augment: Generate this many variations per text (for more data).

    Returns:
        X: (N, dim) float32 embeddings
        y: (N,) long labels (0=spam, 1=ham, 2=ambiguous)
    """
    texts = []
    labels = []

    def _augment(text: str, n: int) -> list[str]:
        results = [text]
        for _ in range(n - 1):
            words = text.split()
            if len(words) > 5:
                np.random.shuffle(words)
                results.append(" ".join(words))
            else:
                results.append(text)
        return results

    # Spam: 25 texts × n_augment
    for t in SPAM_TEXTS:
        for t_aug in _augment(t, n_augment):
            texts.append(t_aug)
            labels.append(0)

    # Ham: 25 texts × n_augment
    for t in HAM_TEXTS:
        for t_aug in _augment(t, n_augment):
            texts.append(t_aug)
            labels.append(1)

    # Ambiguous: 15 texts × n_augment
    for t in AMBIGUOUS_TEXTS:
        for t_aug in _augment(t, n_augment):
            texts.append(t_aug)
            labels.append(2)

    if include_docs:
        repo_texts, sources = read_repo_texts()
        # Label repo texts as "ham" — they're technical content
        for t in repo_texts:
            texts.append(t)
            labels.append(1)  # Technical content is "ham"

    print(f"  Dataset: {len(texts)} texts ({len(SPAM_TEXTS)*n_augment} spam, "
          f"{len(HAM_TEXTS)*n_augment + (len(repo_texts) if include_docs else 0)} ham, "
          f"{len(AMBIGUOUS_TEXTS)*n_augment} ambiguous)")

    # Build TF-IDF embeddings
    tfidf = SimpleTfIdf(dim=dim, max_features=min(dim, 2000))
    tfidf.fit(texts)
    X = tfidf.transform(texts)
    y = np.array(labels, dtype=np.int64)

    # Normalize embeddings
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    # Shuffle
    perm = np.random.RandomState(42).permutation(len(y))
    X, y = X[perm], y[perm]

    return X, y


# ──────────────────────────────────────────────────────────────────────
# 5. Training and Evaluation
# ──────────────────────────────────────────────────────────────────────

def train_and_eval(model, X_train, y_train, X_test, y_test):
    """Train and return test accuracy."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(N_EPOCHS):
        perm = torch.randperm(len(y_train))
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(y_train), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb, yb = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(X_test).argmax(dim=1)
                acc = (preds == y_test).float().mean().item()
            print(f"  Epoch {epoch+1}/{N_EPOCHS} | Loss: {total_loss/n_batches:.4f} | Val Acc: {acc:.4f}")
            model.train()

    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        acc = (preds == y_test).float().mean().item()
        # Per-class accuracy
        per_class = {}
        for c in range(N_CLASSES):
            mask = y_test == c
            if mask.sum() > 0:
                per_class[c] = (preds[mask] == c).float().mean().item()

    return acc, per_class


def measure_inference(model, X, n_runs=INFERENCE_RUNS):
    """Measure avg inference time in ms over n_runs of batch=32."""
    model.eval()
    batch = X[:min(32, len(X))]
    with torch.no_grad():
        # warmup
        for _ in range(10):
            _ = model(batch)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = model(batch)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_runs
    return elapsed * 1000  # ms


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ──────────────────────────────────────────────────────────────────────
# 6. Main Benchmark Runner
# ──────────────────────────────────────────────────────────────────────

def run_benchmark(dims=(128, 256, 512), block_sizes=(16, 32), n_augment=5):
    """Run SplineLinearHD vs Dense benchmark on real text embeddings."""

    print(f"\n{'='*70}")
    print(f" SplineLinearHD → REAL TEXT DATA BENCHMARK")
    print(f"{'='*70}")
    print(f" GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f" Embedding dims: {dims}")
    print(f" Block sizes: {block_sizes}")
    print(f" Epochs: {N_EPOCHS} × {len(dims)} datasets")
    print(f" Classes: {N_CLASSES} (spam, ham, ambiguous)")
    print(f" Augmentation: {n_augment}x per text")
    print(f"{'='*70}")

    all_results = []

    for dim in dims:
        print(f"\n{'─'*70}")
        print(f"  DIM={dim}")
        print(f"{'─'*70}")

        # Build dataset for this dim
        print(f"\n  Building dataset (dim={dim})...")
        X, y = build_real_dataset(dim=dim, include_docs=True, n_augment=n_augment)

        X_t = torch.from_numpy(X).float().to(DEVICE)
        y_t = torch.from_numpy(y).long().to(DEVICE)

        n_train = int(0.8 * len(y_t))
        X_train, X_test = X_t[:n_train], X_t[n_train:]
        y_train, y_test = y_t[:n_train], y_t[n_train:]

        print(f"  Train: {len(y_train)}, Test: {len(y_test)}")
        print(f"  Class distribution (train): "
              f"{Counter(y_train.cpu().tolist())}")
        print(f"  Class distribution (test): "
              f"{Counter(y_test.cpu().tolist())}")

        # ── Dense baseline ──
        print(f"\n  Training Dense...")
        dense = nn.Linear(dim, N_CLASSES).to(DEVICE)
        dense_params = count_params(dense)
        dense_acc, dense_per_class = train_and_eval(dense, X_train, y_train, X_test, y_test)
        dense_time = measure_inference(dense, X_test)
        print(f"  ✓ Dense: {dense_params} params, {dense_acc:.4f} acc, {dense_time:.3f} ms")

        row = {"dim": dim, "dense_params": dense_params, "dense_acc": dense_acc,
               "dense_time": dense_time, "dense_per_class": dense_per_class}

        # ── SplineLinearHD variants ──
        for bs in block_sizes:
            label = f"SplineHD-{bs}"
            print(f"\n  Training {label}...")
            spline = SplineLinearHD(
                in_features=dim,
                out_features=N_CLASSES,
                block_size=bs,
                n_control_points=16,
                basis="eisenstein",
                bias=True,
            ).to(DEVICE)

            sp = count_params(spline)
            sp_acc, sp_per_class = train_and_eval(spline, X_train, y_train, X_test, y_test)
            sp_time = measure_inference(spline, X_test)
            ratio = dense_params / max(sp, 1)

            print(f"  ✓ {label}: {sp} params, {sp_acc:.4f} acc, {sp_time:.3f} ms, {ratio:.1f}x compression")

            row[f"spline{bs}_params"] = sp
            row[f"spline{bs}_acc"] = sp_acc
            row[f"spline{bs}_time"] = sp_time
            row[f"spline{bs}_ratio"] = ratio
            row[f"spline{bs}_per_class"] = sp_per_class

        all_results.append(row)

    return all_results


def print_summary_table(results):
    """Print a clean summary table."""
    print(f"\n{'='*120}")
    h = (f"{'Dim':>5} | {'Dense P':>8} | {'Dense A':>7} | {'Dense ms':>8} | "
         f"{'SHD16 P':>9} | {'SHD16 A':>7} | {'SHD16 R':>9} | {'SHD16 ms':>8} | "
         f"{'SHD32 P':>9} | {'SHD32 A':>7} | {'SHD32 R':>9} | {'SHD32 ms':>8}")
    print(h)
    print("-" * len(h))

    for r in results:
        print(
            f"{r['dim']:>5} | {r['dense_params']:>8} | {r['dense_acc']:>7.4f} | {r['dense_time']:>8.3f} | "
            f"{r['spline16_params']:>9} | {r['spline16_acc']:>7.4f} | {r['spline16_ratio']:>8.1f}x | {r['spline16_time']:>8.3f} | "
            f"{r['spline32_params']:>9} | {r['spline32_acc']:>7.4f} | {r['spline32_ratio']:>8.1f}x | {r['spline32_time']:>8.3f}"
        )
    print("-" * len(h))

    # Averages
    avg_d = sum(r['dense_acc'] for r in results) / len(results)
    avg_16 = sum(r['spline16_acc'] for r in results) / len(results)
    avg_32 = sum(r['spline32_acc'] for r in results) / len(results)
    avg_r16 = sum(r['spline16_ratio'] for r in results) / len(results)
    avg_r32 = sum(r['spline32_ratio'] for r in results) / len(results)
    print(f"\n  Averages across all dims:")
    print(f"    Dense acc:    {avg_d:.4f}")
    print(f"    SHD-16 acc:   {avg_16:.4f}  (retention: {avg_16/avg_d:.1%})  compression: {avg_r16:.1f}x")
    print(f"    SHD-32 acc:   {avg_32:.4f}  (retention: {avg_32/avg_d:.1%})  compression: {avg_r32:.1f}x")
    print(f"{'='*120}")


def save_results(results, path):
    """Save results as a detailed markdown report."""
    lines = [
        "# SplineLinearHD on REAL Text Embeddings",
        "",
        "## Benchmark Setup",
        "",
        "Does SplineLinearHD maintain its compression advantage on **real noisy text embeddings** "
        "from the PLATO ecosystem?",
        "",
        "- **Device:** " + str(DEVICE),
        f"- **Dataset:** {len(SPAM_TEXTS)} spam + {len(HAM_TEXTS)} ham + {len(AMBIGUOUS_TEXTS)} ambiguous "
        f"email-like texts, augmented {results[0].get('n_augment', 5)}x each",
        "- **Plus:** paragraphs pulled from PLATO repo READMEs and source docstrings",
        "- **Embedding:** Simple TF-IDF (no sklearn dependency) with normalization",
        "- **Classes:** 3 (spam=0, ham=1, ambiguous=2)",
        "- **Train/Test:** 80/20 split",
        "- **Epochs:** 30",
        "- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-4)",
        "",
        "## Results Table",
        "",
        "| Dim | Dense Params | Dense Acc | Dense Time (ms) | "
        "SHD-16 Params | SHD-16 Acc | SHD-16 Ratio | SHD-16 Time (ms) | "
        "SHD-32 Params | SHD-32 Acc | SHD-32 Ratio | SHD-32 Time (ms) |",
        "|----:|:-----------:|:--------:|:--------------:|"
        ":-----------:|:--------:|:------------:|:---------------:|"
        ":-----------:|:--------:|:------------:|:---------------:|",
    ]

    for r in results:
        lines.append(
            f"| {r['dim']} | {r['dense_params']} | {r['dense_acc']:.4f} | {r['dense_time']:.3f} | "
            f"{r['spline16_params']} | {r['spline16_acc']:.4f} | {r['spline16_ratio']:.1f}x | {r['spline16_time']:.3f} | "
            f"{r['spline32_params']} | {r['spline32_acc']:.4f} | {r['spline32_ratio']:.1f}x | {r['spline32_time']:.3f} |"
        )

    lines.extend([
        "",
        "## Per-Class Accuracy (from 512-dim run, if available)",
        "",
    ])

    # Try to include per-class breakdown
    for r in results:
        if r['dim'] == 512:
            lines.append("### 512-dim Per-Class Accuracy")
            lines.append("")
            lines.append("| Model | Spam | Ham | Ambiguous | Avg |")
            lines.append("|:-----|:---:|:---:|:--------:|:---:|")
            lines.append(
                f"| Dense | {r['dense_per_class'].get(0, 'N/A'):.4f} | "
                f"{r['dense_per_class'].get(1, 'N/A'):.4f} | "
                f"{r['dense_per_class'].get(2, 'N/A'):.4f} | "
                f"{r['dense_acc']:.4f} |"
            )
            for bs in [16, 32]:
                per = r.get(f'spline{bs}_per_class', {})
                lines.append(
                    f"| SHD-{bs} | {per.get(0, 'N/A'):.4f} | "
                    f"{per.get(1, 'N/A'):.4f} | "
                    f"{per.get(2, 'N/A'):.4f} | "
                    f"{r[f'spline{bs}_acc']:.4f} |"
                )
            lines.append("")

    lines.extend([
        "## Summary",
        "",
    ])

    avg_d = sum(r['dense_acc'] for r in results) / len(results)
    avg_16 = sum(r['spline16_acc'] for r in results) / len(results)
    avg_32 = sum(r['spline32_acc'] for r in results) / len(results)
    avg_r16 = sum(r['spline16_ratio'] for r in results) / len(results)
    avg_r32 = sum(r['spline32_ratio'] for r in results) / len(results)

    lines.append(f"- **Avg Dense accuracy:** {avg_d:.4f}")
    lines.append(f"- **Avg SHD-16 accuracy:** {avg_16:.4f} (retention: {avg_16/avg_d:.1%})")
    lines.append(f"- **Avg SHD-32 accuracy:** {avg_32:.4f} (retention: {avg_32/avg_d:.1%})")
    lines.append(f"- **Avg SHD-16 compression:** {avg_r16:.1f}x")
    lines.append(f"- **Avg SHD-32 compression:** {avg_r32:.1f}x")
    lines.append("")

    # Key finding
    lines.append("## Key Finding")
    lines.append("")
    d_gap = avg_r16 * (avg_16 / avg_d)  # compression × accuracy retention
    if d_gap > 10:
        lines.append(
            f"**SplineLinearHD maintains strong compression on real text embeddings.** "
            f"At block_size=16, it achieves {avg_16/avg_d:.1%} of Dense accuracy "
            f"with {avg_r16:.1f}x fewer parameters. The compression-accuracy product "
            f"is {d_gap:.1f}, confirming the approach works on real noisy data."
        )
    else:
        lines.append(
            f"SplineLinearHD on real text shows {'promising' if avg_16/avg_d > 0.85 else 'mixed'} results. "
            f"Accuracy retention: {avg_16/avg_d:.1%} at {avg_r16:.1f}x compression."
        )
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--augment", type=int, default=5, help="Text augmentation factor")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output", type=str, default=None,
                        help="Output markdown path")
    args = parser.parse_args()

    if args.epochs != 30:
        CONFIG['n_epochs'] = args.epochs
        # Re-bind module-level constants
        import sys as _sys
        _mod = _sys.modules[__name__]
        _mod.N_EPOCHS = args.epochs
        _mod.CONFIG['n_epochs'] = args.epochs
        print(f"  Epochs overridden: {args.epochs}")

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    results = run_benchmark(
        dims=tuple(args.dims),
        block_sizes=tuple(args.block_sizes),
        n_augment=args.augment,
    )

    print_summary_table(results)

    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "spline_real_results.md"
    )
    save_results(results, output_path)
