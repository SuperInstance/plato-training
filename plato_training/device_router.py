"""
DeviceRouter — routes inference to the most efficient compute unit.

Decision logic:
- Single-sample embedding → CPU (Model2Vec, 9μs, no transfer overhead)
- Vector search → CPU FAISS (<1M vectors, fast enough)
- Small model inference → CPU (VNNI-optimized) or iGPU (if available)
- Large model inference → CUDA GPU (AMP, batched)
- Training → ALWAYS CUDA GPU
"""

import gc
import time
import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DeviceRouter:
    """Routes inference to the most efficient compute unit."""

    def __init__(self):
        import torch
        self.torch = torch
        self.cuda_available = torch.cuda.is_available()

        # Try DirectML (AMD iGPU)
        self.dml_available = False
        self.dml = None
        try:
            import torch_directml  # type: ignore
            self.dml = torch_directml.device()
            self.dml_available = True
        except ImportError:
            pass

        # Optional: Model2Vec for embeddings
        self._m2v_model = None
        try:
            from model2vec import StaticModel  # type: ignore
            self._m2v_cls = StaticModel
        except ImportError:
            self._m2v_cls = None

        # Optional: FAISS for vector search
        self._faiss = None
        try:
            import faiss  # type: ignore
            self._faiss = faiss
        except ImportError:
            pass

        # Device assignments
        self.train_device = torch.device('cuda') if self.cuda_available else torch.device('cpu')
        self.infer_device = torch.device('cpu')  # Default: CPU for single-sample

    # ------------------------------------------------------------------
    # Device info
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Report available devices and their status."""
        info: dict[str, Any] = {
            'cuda': self.cuda_available,
            'directml': self.dml_available,
            'model2vec': self._m2v_cls is not None,
            'faiss': self._faiss is not None,
            'train_device': str(self.train_device),
            'infer_device': str(self.infer_device),
        }
        if self.cuda_available:
            info['cuda_device_name'] = self.torch.cuda.get_device_name(0)
            info['cuda_memory_total_mb'] = round(
                self.torch.cuda.get_device_properties(0).total_memory / 1e6
            )
        return info

    # ------------------------------------------------------------------
    # Model size helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_params(model) -> int:
        """Count total parameters of a torch model."""
        return sum(p.numel() for p in model.parameters())

    def _classify_size(self, model) -> str:
        """Return 'micro', 'small', or 'large' based on param count."""
        n = self._count_params(model)
        if n < 1_000:
            return 'micro'
        elif n < 100_000:
            return 'small'
        else:
            return 'large'

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, text: str) -> 'np.ndarray':
        """Embed text on CPU. Fastest for single queries."""
        if self._m2v_cls is not None:
            if self._m2v_model is None:
                self._m2v_model = self._m2v_cls.from_pretrained("minishlab/potion-base-8M")
            return self._m2v_model.encode([text])[0]
        # Fallback: simple keyword hash vector
        return self._keyword_vector(text)

    @staticmethod
    def _keyword_vector(text: str, dim: int = 128) -> 'np.ndarray':
        """Fallback embedding via deterministic keyword hashing."""
        vec = np.zeros(dim, dtype=np.float32)
        for word in text.lower().split():
            idx = hash(word) % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def search(self, query_vec: 'np.ndarray', index: Any = None,
               vectors: 'np.ndarray' = None, k: int = 5,
               threshold: float = 0.6) -> list:
        """Search nearest neighbours. Uses FAISS if available, else numpy."""
        query_vec = np.asarray(query_vec, dtype=np.float32).ravel()

        if self._faiss is not None and index is not None:
            query = query_vec.reshape(1, -1)
            dists, idxs = index.search(query, k)
            results = []
            for d, i in zip(dists[0], idxs[0]):
                if i >= 0 and d >= threshold:
                    results.append({'index': int(i), 'score': float(d)})
            return results

        # Fallback: brute-force cosine similarity
        if vectors is None:
            return []
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = vectors / norms
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        scores = normed @ q_norm
        top_k = np.argsort(scores)[::-1][:k]
        return [
            {'index': int(i), 'score': float(scores[i])}
            for i in top_k if scores[i] >= threshold
        ]

    def build_index(self, vectors: 'np.ndarray') -> Any:
        """Build a FAISS index (or return None if FAISS unavailable)."""
        if self._faiss is None:
            return None
        vectors = np.asarray(vectors, dtype=np.float32)
        dim = vectors.shape[1]
        index = self._faiss.IndexFlatIP(dim)
        index.add(vectors)
        return index

    # ------------------------------------------------------------------
    # Inference routing
    # ------------------------------------------------------------------

    def infer(self, model, x: 'torch.Tensor',
              model_size: str = 'auto') -> 'torch.Tensor':
        """
        Route inference based on model size.

        model_size='micro'  → CPU
        model_size='small'  → iGPU (DirectML) if available, else CPU
        model_size='large'  → CUDA GPU (AMP if available)
        model_size='auto'   → detect from parameter count
        """
        torch = self.torch

        if model_size == 'auto':
            model_size = self._classify_size(model)

        # Choose device
        if model_size == 'large' and self.cuda_available:
            device = torch.device('cuda')
        elif model_size == 'small' and self.dml_available:
            device = self.dml  # DirectML device object
        else:
            device = torch.device('cpu')

        was_training = model.training
        model.eval()
        model_device = next(model.parameters()).device

        # Move if needed
        if str(model_device) != str(device):
            model = model.to(device)

        x = x.to(device)
        input_device = device

        use_amp = (model_size == 'large' and self.cuda_available)

        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast('cuda'):
                    out = model(x)
            else:
                out = model(x)

        # Move output back to CPU to free GPU memory
        out = out.to('cpu')

        if was_training:
            model.train()

        return out

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, model, optimizer, loss_fn, x: 'torch.Tensor',
                   y: 'torch.Tensor') -> float:
        """One training step. Always on GPU if available, else CPU."""
        torch = self.torch
        device = self.train_device

        model = model.to(device)
        model.train()
        x = x.to(device)
        y = y.to(device)

        use_amp = self.cuda_available
        scaler = torch.amp.GradScaler('cuda') if use_amp else None

        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast('cuda'):
                pred = model(x)
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def benchmark(self, model, x: 'torch.Tensor', n_runs: int = 100) -> dict:
        """Compare inference time on CPU vs CUDA for same model."""
        torch = self.torch
        model.eval()
        results = {}

        # CPU benchmark
        model_cpu = model.to('cpu')
        x_cpu = x.to('cpu')
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(n_runs):
                model_cpu(x_cpu)
            results['cpu_ms'] = (time.perf_counter() - t0) / n_runs * 1000

        # CUDA benchmark
        if self.cuda_available:
            model_cuda = model.to('cuda')
            x_cuda = x.to('cuda')
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    model_cuda(x_cuda)
                torch.cuda.synchronize()
            with torch.no_grad():
                t0 = time.perf_counter()
                for _ in range(n_runs):
                    model_cuda(x_cuda)
                torch.cuda.synchronize()
            results['cuda_ms'] = (time.perf_counter() - t0) / n_runs * 1000
            results['speedup'] = results['cpu_ms'] / results['cuda_ms']
            # Cleanup
            del model_cuda, x_cuda
            torch.cuda.empty_cache()

        # Restore model to CPU
        model.to('cpu')
        return results
