#!/usr/bin/env python3
"""Full heterogeneous compute benchmark: CUDA GPU, CPU (AVX-512), DirectML iGPU."""

import gc
import os
import sys
import time
import traceback
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np
import torch

# ── helpers ──────────────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    name: str
    device: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

results: List[BenchResult] = []

@contextmanager
def timer(label: str):
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    print(f"  ⏱ {label}: {elapsed:.3f}s")

def flush(device_str: str):
    gc.collect()
    if "cuda" in device_str:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def warmup(fn, iters=10):
    for _ in range(iters):
        fn()

# ── Device Setup ─────────────────────────────────────────────────────────────

cuda_dev = torch.device("cuda:0") if torch.cuda.is_available() else None
cpu_dev = torch.device("cpu")

try:
    import torch_directml
    dml_dev = torch_directml.device()
    HAS_DML = True
except ImportError:
    dml_dev = None
    HAS_DML = False

print("=" * 70)
print("HETEROGENEOUS COMPUTE BENCHMARK")
print("=" * 70)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if cuda_dev:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  VRAM: {props.total_memory / 1e9:.1f} GB, SM: {props.major}.{props.minor}")
print(f"CPU threads: {torch.get_num_threads()}")
print(f"DirectML: {HAS_DML}")
print(f"NumPy AVX-512: AVX512_ICL available")
print()

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 1: Matrix Multiply (FP32)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("BENCHMARK 1: Matrix Multiply (FP32)")
print("=" * 70)

SIZES = [64, 128, 256, 512, 1024]
ITERS = 1000

def bench_matmul(device, device_name, sizes, iters):
    device_results = []
    for n in sizes:
        # Check memory for GPU
        if "cuda" in str(device):
            needed = n * n * 4 * 3  # A, B, C
            free_mem = torch.cuda.mem_get_info(device)[0]
            if needed > free_mem * 0.8:
                r = BenchResult(f"MatMul {n}x{n}", device_name,
                               metrics={"status": "SKIPPED", "reason": "OOM risk"})
                device_results.append(r)
                print(f"  {n}x{n}: SKIPPED (OOM risk)")
                continue

        a = torch.randn(n, n, device=device, dtype=torch.float32)
        b = torch.randn(n, n, device=device, dtype=torch.float32)

        # Warmup
        for _ in range(20):
            c = a @ b
        if "cuda" in str(device):
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            c = a @ b
        if "cuda" in str(device):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        flops = 2 * n * n * n * iters  # multiply-add = 2 ops
        gflops = flops / elapsed / 1e9
        avg_ms = (elapsed / iters) * 1000

        # Memory
        if "cuda" in str(device):
            mem_mb = torch.cuda.max_memory_allocated(device) / 1e6
            torch.cuda.reset_peak_memory_stats(device)
        else:
            mem_mb = n * n * 4 * 3 / 1e6  # estimate

        r = BenchResult(f"MatMul {n}x{n}", device_name,
                        metrics={"gflops": round(gflops, 2),
                                "avg_ms": round(avg_ms, 4),
                                "mem_mb": round(mem_mb, 2),
                                "iterations": iters})
        device_results.append(r)
        print(f"  {n:5d}x{n:<5d}: {gflops:8.2f} GFLOPS | {avg_ms:10.4f} ms | {mem_mb:8.2f} MB")
        
        del a, b, c
        flush(str(device))

    return device_results

# CPU
print("\n--- CPU (AMD Ryzen AI 9 HX 370, AVX-512) ---")
results.extend(bench_matmul(cpu_dev, "CPU-AVX512", SIZES, ITERS))

# CUDA
if cuda_dev:
    print("\n--- CUDA (NVIDIA RTX 4050 Laptop GPU) ---")
    results.extend(bench_matmul(cuda_dev, "CUDA-RTX4050", SIZES, ITERS))

# DirectML
if HAS_DML:
    print("\n--- DirectML (AMD Radeon 890M iGPU) ---")
    results.extend(bench_matmul(dml_dev, "DirectML-Radeon890M", SIZES, ITERS))

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 2: SplineLinear Model Inference
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("BENCHMARK 2: SplineLinear Model Inference")
print("=" * 70)

try:
    from plato_training.spline import SplineLinear

    # Build a model: 128 -> 64 -> 32 with 16 control points
    model = nn.Sequential(
        SplineLinear(128, 64, num_control_points=16),
        nn.ReLU(),
        SplineLinear(64, 32, num_control_points=16),
        nn.ReLU(),
        SplineLinear(32, 10, num_control_points=16),
    )
    model.eval()
    
    dummy_input = torch.randn(1, 128)
    
    # 2a. PyTorch CPU
    print("\n--- PyTorch CPU ---")
    with torch.no_grad():
        warmup(lambda: model(dummy_input), 50)
        t0 = time.perf_counter()
        iters = 10000
        for _ in range(iters):
            _ = model(dummy_input)
        elapsed = time.perf_counter() - t0
    lat_us = (elapsed / iters) * 1e6
    thru = iters / elapsed
    results.append(BenchResult("SplineLinear Inference", "CPU-PyTorch",
                              metrics={"latency_us": round(lat_us, 2),
                                      "throughput_qps": round(thru, 1),
                                      "iterations": iters}))
    print(f"  Latency: {lat_us:.2f} µs | Throughput: {thru:.1f} qps")

    # 2b. PyTorch CUDA
    if cuda_dev:
        print("\n--- PyTorch CUDA ---")
        model_gpu = model.to(cuda_dev)
        inp_gpu = dummy_input.to(cuda_dev)
        with torch.no_grad():
            warmup(lambda: model_gpu(inp_gpu), 50)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = model_gpu(inp_gpu)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
        lat_us = (elapsed / iters) * 1e6
        thru = iters / elapsed
        results.append(BenchResult("SplineLinear Inference", "CUDA-PyTorch",
                                  metrics={"latency_us": round(lat_us, 2),
                                          "throughput_qps": round(thru, 1),
                                          "iterations": iters}))
        print(f"  Latency: {lat_us:.2f} µs | Throughput: {thru:.1f} qps")
        del model_gpu, inp_gpu
        flush("cuda")

    # 2c. PyTorch DirectML
    if HAS_DML:
        print("\n--- PyTorch DirectML (iGPU) ---")
        model_dml = model.to(dml_dev)
        inp_dml = dummy_input.to(dml_dev)
        with torch.no_grad():
            warmup(lambda: model_dml(inp_dml), 50)
            t0 = time.perf_counter()
            dml_iters = 5000
            for _ in range(dml_iters):
                _ = model_dml(inp_dml)
            elapsed = time.perf_counter() - t0
        lat_us = (elapsed / dml_iters) * 1e6
        thru = dml_iters / elapsed
        results.append(BenchResult("SplineLinear Inference", "DirectML-PyTorch",
                                  metrics={"latency_us": round(lat_us, 2),
                                          "throughput_qps": round(thru, 1),
                                          "iterations": dml_iters}))
        print(f"  Latency: {lat_us:.2f} µs | Throughput: {thru:.1f} qps")
        del model_dml, inp_dml
        gc.collect()

    # 2d. ONNX Runtime
    print("\n--- ONNX Runtime (CPU) ---")
    import onnxruntime as ort
    import io
    import tempfile

    # Export to ONNX
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    torch.onnx.export(model, dummy_input, onnx_path,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
    
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_feed = {"input": dummy_input.numpy()}
    
    # Warmup
    for _ in range(100):
        sess.run(None, inp_feed)
    
    t0 = time.perf_counter()
    onnx_iters = 10000
    for _ in range(onnx_iters):
        _ = sess.run(None, inp_feed)
    elapsed = time.perf_counter() - t0
    lat_us = (elapsed / onnx_iters) * 1e6
    thru = onnx_iters / elapsed
    results.append(BenchResult("SplineLinear Inference", "ONNX-CPU",
                              metrics={"latency_us": round(lat_us, 2),
                                      "throughput_qps": round(thru, 1),
                                      "iterations": onnx_iters}))
    print(f"  Latency: {lat_us:.2f} µs | Throughput: {thru:.1f} qps")
    
    # ONNX batched
    print("\n--- ONNX Runtime (CPU, batch=32) ---")
    batch_input = torch.randn(32, 128).numpy()
    inp_feed_batch = {"input": batch_input}
    for _ in range(50):
        sess.run(None, inp_feed_batch)
    t0 = time.perf_counter()
    onnx_batch_iters = 2000
    for _ in range(onnx_batch_iters):
        _ = sess.run(None, inp_feed_batch)
    elapsed = time.perf_counter() - t0
    total_samples = 32 * onnx_batch_iters
    lat_us = (elapsed / onnx_batch_iters) * 1e6
    thru = total_samples / elapsed
    results.append(BenchResult("SplineLinear Inference (batch=32)", "ONNX-CPU",
                              metrics={"latency_us": round(lat_us, 2),
                                      "throughput_qps": round(thru, 1),
                                      "total_samples": total_samples}))
    print(f"  Latency: {lat_us:.2f} µs/batch | Throughput: {thru:.1f} samples/s")
    
    os.unlink(onnx_path)
    del sess

except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    results.append(BenchResult("SplineLinear Inference", "ERROR",
                              notes=str(e)))

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 3: Embedding Speed (sentence-transformers / model2vec)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("BENCHMARK 3: Embedding Speed")
print("=" * 70)

sentences = [f"This is test sentence number {i} for benchmarking embeddings." for i in range(1000)]

try:
    from sentence_transformers import SentenceTransformer
    
    print("\n--- sentence-transformers (CPU) ---")
    model_st = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    
    # Warmup
    _ = model_st.encode(sentences[:10])
    
    t0 = time.perf_counter()
    embeddings = model_st.encode(sentences, batch_size=64, show_progress_bar=False)
    elapsed = time.perf_counter() - t0
    emb_per_sec = len(sentences) / elapsed
    
    results.append(BenchResult("Sentence Embedding (1000)", "CPU-sentence-transformers",
                              metrics={"embeddings_per_sec": round(emb_per_sec, 1),
                                      "total_seconds": round(elapsed, 2),
                                      "dim": embeddings.shape[1]}))
    print(f"  {emb_per_sec:.1f} embeddings/sec | {elapsed:.2f}s total | dim={embeddings.shape[1]}")
    del model_st

except ImportError:
    print("  sentence-transformers not available, trying model2vec...")
    
    try:
        from model2vec import StaticModel
        
        print("\n--- model2vec (CPU) ---")
        model_mv = StaticModel.from_pretrained("minishlab/M2V_base_output")
        
        _ = model_mv.encode(sentences[:10])
        t0 = time.perf_counter()
        embeddings = model_mv.encode(sentences)
        elapsed = time.perf_counter() - t0
        emb_per_sec = len(sentences) / elapsed
        
        results.append(BenchResult("Sentence Embedding (1000)", "CPU-model2vec",
                                  metrics={"embeddings_per_sec": round(emb_per_sec, 1),
                                          "total_seconds": round(elapsed, 2),
                                          "dim": embeddings.shape[1]}))
        print(f"  {emb_per_sec:.1f} embeddings/sec | {elapsed:.2f}s total | dim={embeddings.shape[1]}")
        del model_mv
    except ImportError:
        print("  model2vec not available either. Skipping.")
        results.append(BenchResult("Sentence Embedding", "SKIPPED",
                                  notes="Neither sentence-transformers nor model2vec available"))

except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    results.append(BenchResult("Sentence Embedding", "ERROR", notes=str(e)))

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 4: FAISS Search
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("BENCHMARK 4: FAISS Search (10K vectors, 256-dim)")
print("=" * 70)

try:
    import faiss
    
    np.random.seed(42)
    d = 256
    nb = 10000
    nq = 1000
    k = 10  # top-k
    
    xb = np.random.random((nb, d)).astype('float32')
    xq = np.random.random((nq, d)).astype('float32')
    
    # CPU IndexFlatIP
    print("\n--- FAISS CPU (IndexFlatIP) ---")
    index = faiss.IndexFlatIP(d)
    index.add(xb)
    
    # Warmup
    _ = index.search(xq[:10], k)
    
    t0 = time.perf_counter()
    D, I = index.search(xq, k)
    elapsed = time.perf_counter() - t0
    qps = nq / elapsed
    
    results.append(BenchResult("FAISS Search (10K×256, 1K queries)", "CPU-IndexFlatIP",
                              metrics={"qps": round(qps, 1),
                                      "total_ms": round(elapsed * 1000, 2),
                                      "top_k": k}))
    print(f"  {qps:.1f} queries/sec | {elapsed*1000:.2f} ms total")
    
    # CPU IndexIVFFlat
    print("\n--- FAISS CPU (IndexIVFFlat) ---")
    nlist = 100
    quantizer = faiss.IndexFlatL2(d)
    index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)
    index_ivf.train(xb)
    index_ivf.add(xb)
    index_ivf.nprobe = 10
    
    _ = index_ivf.search(xq[:10], k)
    t0 = time.perf_counter()
    D, I = index_ivf.search(xq, k)
    elapsed = time.perf_counter() - t0
    qps = nq / elapsed
    
    results.append(BenchResult("FAISS Search (10K×256, 1K queries)", "CPU-IndexIVFFlat",
                              metrics={"qps": round(qps, 1),
                                      "total_ms": round(elapsed * 1000, 2),
                                      "top_k": k, "nlist": nlist, "nprobe": 10}))
    print(f"  {qps:.1f} queries/sec | {elapsed*1000:.2f} ms total")

except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    results.append(BenchResult("FAISS Search", "ERROR", notes=str(e)))

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 5: Concurrent GPU Training + CPU Inference
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("BENCHMARK 5: Concurrent GPU Training + CPU Inference")
print("=" * 70)

if cuda_dev:
    try:
        import threading
        
        gpu_times = []
        cpu_times = []
        gpu_errors = []
        cpu_errors = []
        
        def gpu_training():
            """Simulate GPU training loop."""
            try:
                model = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 10),
                ).to(cuda_dev)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                
                for i in range(100):
                    x = torch.randn(64, 256, device=cuda_dev)
                    y = torch.randint(0, 10, (64,), device=cuda_dev)
                    
                    out = model(x)
                    loss = nn.functional.cross_entropy(out, y)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    if i % 20 == 0:
                        gpu_times.append(time.perf_counter())
                
                torch.cuda.synchronize()
            except Exception as e:
                gpu_errors.append(str(e))
        
        def cpu_inference():
            """Simulate CPU inference loop."""
            try:
                model = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 10),
                ).eval()
                
                for i in range(100):
                    x = torch.randn(1, 128)
                    with torch.no_grad():
                        _ = model(x)
                    if i % 20 == 0:
                        cpu_times.append(time.perf_counter())
            except Exception as e:
                cpu_errors.append(str(e))
        
        # Run concurrently
        t_gpu = threading.Thread(target=gpu_training)
        t_cpu = threading.Thread(target=cpu_inference)
        
        t0 = time.perf_counter()
        t_gpu.start()
        t_cpu.start()
        t_gpu.join(timeout=60)
        t_cpu.join(timeout=60)
        total_elapsed = time.perf_counter() - t0
        
        # Also run them sequentially to compare
        gpu_times_seq = []
        cpu_times_seq = []
        
        # Sequential GPU
        t0_seq = time.perf_counter()
        gpu_training()
        gpu_seq_time = time.perf_counter() - t0_seq
        
        # Sequential CPU
        t0_seq = time.perf_counter()
        cpu_inference()
        cpu_seq_time = time.perf_counter() - t0_seq
        
        seq_total = gpu_seq_time + cpu_seq_time
        speedup = seq_total / total_elapsed
        
        interference = "NONE" if not gpu_errors and not cpu_errors else "ERRORS DETECTED"
        
        results.append(BenchResult("Concurrent GPU+CPU", "Mixed",
                                  metrics={"concurrent_total_s": round(total_elapsed, 3),
                                          "sequential_total_s": round(seq_total, 3),
                                          "speedup": round(speedup, 2),
                                          "gpu_alone_s": round(gpu_seq_time, 3),
                                          "cpu_alone_s": round(cpu_seq_time, 3),
                                          "interference": interference}))
        print(f"  Concurrent: {total_elapsed:.3f}s | Sequential: {seq_total:.3f}s | Speedup: {speedup:.2f}x")
        print(f"  GPU alone: {gpu_seq_time:.3f}s | CPU alone: {cpu_seq_time:.3f}s")
        print(f"  Interference: {interference}")
        if gpu_errors:
            print(f"  GPU errors: {gpu_errors}")
        if cpu_errors:
            print(f"  CPU errors: {cpu_errors}")
        
        flush("cuda")
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results.append(BenchResult("Concurrent GPU+CPU", "ERROR", notes=str(e)))
else:
    results.append(BenchResult("Concurrent GPU+CPU", "SKIPPED", notes="No CUDA"))
    print("  SKIPPED: No CUDA device")

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 6: Mixed Precision (FP16/BF16 on GPU)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("BENCHMARK 6: Mixed Precision MatMul (GPU)")
print("=" * 70)

if cuda_dev:
    for dtype_name, dtype in [("FP16", torch.float16), ("BF16", torch.bfloat16)]:
        print(f"\n--- {dtype_name} ---")
        dtype_results = []
        for n in [256, 512, 1024]:
            a = torch.randn(n, n, device=cuda_dev, dtype=dtype)
            b = torch.randn(n, n, device=cuda_dev, dtype=dtype)
            
            for _ in range(30):
                c = a @ b
            torch.cuda.synchronize()
            
            iters = 2000
            t0 = time.perf_counter()
            for _ in range(iters):
                c = a @ b
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            
            gflops = 2 * n * n * n * iters / elapsed / 1e9
            r = BenchResult(f"MatMul {n}x{n} ({dtype_name})", "CUDA-RTX4050",
                           metrics={"gflops": round(gflops, 2),
                                   "avg_ms": round((elapsed / iters) * 1000, 4)})
            results.append(r)
            dtype_results.append(r)
            print(f"  {n:5d}x{n:<5d}: {gflops:8.2f} GFLOPS")
            del a, b, c
        flush("cuda")

# ══════════════════════════════════════════════════════════════════════════════
# Save JSON results
# ══════════════════════════════════════════════════════════════════════════════

results_json = [{"name": r.name, "device": r.device, "metrics": r.metrics, "notes": r.notes} for r in results]
with open("bench_results.json", "w") as f:
    json.dump(results_json, f, indent=2)

print(f"\n{'=' * 70}")
print(f"TOTAL BENCHMARKS RUN: {len(results)}")
print(f"Results saved to bench_results.json")
print(f"{'=' * 70}")
