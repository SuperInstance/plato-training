#!/usr/bin/env python3
"""
Benchmark SplineLinearHD on synthetic cluster embeddings.

Produces real numbers for compression ratio, accuracy, and inference time
across dimensions 64, 128, 256, 512, 768 with block_size 16 and 32.
"""
import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plato_training.spline import SplineLinear
from plato_training.spline_hd import SplineLinearHD

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

DIMS = [64, 128, 256, 512, 768]
N_SAMPLES = 1000
N_CLUSTERS = 5
N_EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 64
INFERENCE_RUNS = 100


def generate_data(dim, n_samples, n_clusters):
    """Generate synthetic cluster embeddings."""
    centres = torch.randn(n_clusters, dim) * 2.0
    n_per = n_samples // n_clusters
    X, y = [], []
    for c in range(n_clusters):
        noise = torch.randn(n_per, dim) * 0.5
        X.append(centres[c].unsqueeze(0) + noise)
        y.append(torch.full((n_per,), c, dtype=torch.long))
    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)
    perm = torch.randperm(len(y))
    return X[perm], y[perm]


def train_and_eval(model, X_train, y_train, X_test, y_test):
    """Train and return test accuracy."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(len(y_train))
        for i in range(0, len(y_train), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb, yb = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        acc = (preds == y_test).float().mean().item()
    return acc


def measure_inference(model, X, n_runs=INFERENCE_RUNS):
    """Measure avg inference time in ms over n_runs of batch=32."""
    model.eval()
    batch = X[:32]
    with torch.no_grad():
        # warmup
        for _ in range(5):
            _ = model(batch)
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = model(batch)
        elapsed = (time.perf_counter() - start) / n_runs
    return elapsed * 1000  # ms


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def run_benchmark():
    results = []

    for dim in DIMS:
        print(f"\n--- dim={dim} ---")
        X, y = generate_data(dim, N_SAMPLES, N_CLUSTERS)
        X, y = X.to(DEVICE), y.to(DEVICE)
        n_train = int(0.8 * len(y))
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        # Dense baseline
        dense = nn.Linear(dim, N_CLUSTERS).to(DEVICE)
        dense_params = count_params(dense)
        dense_acc = train_and_eval(dense, X_train, y_train, X_test, y_test)
        dense_time = measure_inference(dense, X_test)
        print(f"  Dense: {dense_params} params, {dense_acc:.4f} acc, {dense_time:.3f} ms")

        row = {
            "dim": dim,
            "dense_params": dense_params,
            "dense_acc": dense_acc,
            "dense_time": dense_time,
        }

        for bs in [16, 32]:
            label = f"SplineHD-{bs}"
            spline = SplineLinearHD(
                in_features=dim,
                out_features=N_CLUSTERS,
                block_size=bs,
                n_control_points=16,
                basis="eisenstein",
                bias=True,
            ).to(DEVICE)

            sp = count_params(spline)
            sp_acc = train_and_eval(spline, X_train, y_train, X_test, y_test)
            sp_time = measure_inference(spline, X_test)
            ratio = dense_params / max(sp, 1)

            print(f"  {label}: {sp} params, {sp_acc:.4f} acc, {sp_time:.3f} ms, {ratio:.1f}x compression")

            row[f"spline{bs}_params"] = sp
            row[f"spline{bs}_acc"] = sp_acc
            row[f"spline{bs}_time"] = sp_time
            row[f"spline{bs}_ratio"] = ratio

        results.append(row)

    return results


def print_table(results):
    header = (
        f"{'Dim':>5} | {'Dense Params':>11} | {'Dense Acc':>8} | {'Dense ms':>8} | "
        f"{'SHD16 Params':>12} | {'SHD16 Acc':>8} | {'SHD16 Ratio':>10} | {'SHD16 ms':>8} | "
        f"{'SHD32 Params':>12} | {'SHD32 Acc':>8} | {'SHD32 Ratio':>10} | {'SHD32 ms':>8}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['dim']:>5} | {r['dense_params']:>11} | {r['dense_acc']:>8.4f} | {r['dense_time']:>8.3f} | "
            f"{r['spline16_params']:>12} | {r['spline16_acc']:>8.4f} | {r['spline16_ratio']:>9.1f}x | {r['spline16_time']:>8.3f} | "
            f"{r['spline32_params']:>12} | {r['spline32_acc']:>8.4f} | {r['spline32_ratio']:>9.1f}x | {r['spline32_time']:>8.3f}"
        )
    print(sep)


def save_markdown(results, path):
    lines = [
        "# SplineLinearHD Benchmark Results",
        "",
        f"- **Device:** {DEVICE}",
        f"- **Samples per dim:** {N_SAMPLES}",
        f"- **Clusters:** {N_CLUSTERS}",
        f"- **Epochs:** {N_EPOCHS}",
        f"- **Learning rate:** {LR}",
        f"- **Inference runs:** {INFERENCE_RUNS} (batch=32)",
        "",
        "## Results Table",
        "",
        "| Dim | Dense Params | Dense Acc | Dense Time (ms) | "
        "SHD-16 Params | SHD-16 Acc | SHD-16 Ratio | SHD-16 Time (ms) | "
        "SHD-32 Params | SHD-32 Acc | SHD-32 Ratio | SHD-32 Time (ms) |",
        "|----:|------------:|----------:|----------------:|"
        "-------------:|-----------:|-------------:|-----------------:|"
        "-------------:|-----------:|-------------:|-----------------:|",
    ]
    for r in results:
        lines.append(
            f"| {r['dim']} | {r['dense_params']} | {r['dense_acc']:.4f} | {r['dense_time']:.3f} | "
            f"{r['spline16_params']} | {r['spline16_acc']:.4f} | {r['spline16_ratio']:.1f}x | {r['spline16_time']:.3f} | "
            f"{r['spline32_params']} | {r['spline32_acc']:.4f} | {r['spline32_ratio']:.1f}x | {r['spline32_time']:.3f} |"
        )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    # Compute averages
    avg_d_acc = sum(r['dense_acc'] for r in results) / len(results)
    avg_16_acc = sum(r['spline16_acc'] for r in results) / len(results)
    avg_32_acc = sum(r['spline32_acc'] for r in results) / len(results)
    avg_16_ratio = sum(r['spline16_ratio'] for r in results) / len(results)
    avg_32_ratio = sum(r['spline32_ratio'] for r in results) / len(results)
    lines.append(f"- **Avg Dense accuracy:** {avg_d_acc:.4f}")
    lines.append(f"- **Avg SHD-16 accuracy:** {avg_16_acc:.4f} (retention: {avg_16_acc/avg_d_acc:.1%})")
    lines.append(f"- **Avg SHD-32 accuracy:** {avg_32_acc:.4f} (retention: {avg_32_acc/avg_d_acc:.1%})")
    lines.append(f"- **Avg SHD-16 compression:** {avg_16_ratio:.1f}x")
    lines.append(f"- **Avg SHD-32 compression:** {avg_32_ratio:.1f}x")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    results = run_benchmark()
    print_table(results)
    save_markdown(
        results,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "spline_hd_benchmark_results.md")
    )
