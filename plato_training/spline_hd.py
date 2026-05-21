"""
High-dimensional Tensor-Spline module for PLATO Training Rooms.

Extends SplineLinear from low-dimensional tasks (drift-detect) to
NLP-scale embeddings (64–768 dimensions) via hierarchical Eisenstein
lattice decomposition.

Architecture
------------
BlockDecomposition   Splits high-dim space into overlapping blocks.
SplineLinearHD       High-dim SplineLinear using block decomposition.
AdaptiveCompression  Per-block compression measurement and selection.
EmbeddingBenchmark   Benchmark runner for synthetic embeddings.
"""



from __future__ import annotations

__all__ = ['AdaptiveCompression', 'AdaptiveLinear', 'BlockDecomposition', 'EmbeddingBenchmark', 'SplineLinearHD']
import math
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spline import SplineLinear


# ---------------------------------------------------------------------------
# BlockDecomposition
# ---------------------------------------------------------------------------

class BlockDecomposition:
    """
    Decomposes a high-dimensional vector into (optionally overlapping) blocks.

    Given an input dimension ``d`` and a ``block_size``, partitions the
    ``d`` axes into ceil(d / block_size) blocks. When ``overlap > 0``,
    adjacent blocks share ``overlap`` dimensions at each boundary for
    continuity.

    Example (d=64, block_size=16, overlap=2)::

        Block 0: dims [0, 16)   — dims 0..15
        Block 1: dims [14, 30)  — dims 14..29  (2 overlap with block 0)
        Block 2: dims [28, 44)  — dims 28..43  (2 overlap with block 1)
        Block 3: dims [42, 58)  — dims 42..57
        Block 4: dims [56, 64)  — dims 56..63  (partial)

    Args:
        input_dim:    Total input dimensionality.
        block_size:   Dimensions per block (before overlap).
        overlap:      Number of shared dimensions between adjacent blocks.
                      Must be < block_size. Default 0 (no overlap).
        allow_partial: If True, the last block may be smaller than
                      ``block_size``. If False, the last block is padded
                      with zeros conceptually (input_dim must be divisible).

    Raises:
        ValueError: If ``overlap >= block_size`` or ``block_size < 1``.
    """

    def __init__(
        self,
        input_dim: int,
        block_size: int = 16,
        overlap: int = 0,
        allow_partial: bool = True,
    ) -> None:
        if block_size < 1:
            raise ValueError(f"block_size must be ≥ 1, got {block_size}")
        if overlap >= block_size:
            raise ValueError(
                f"overlap ({overlap}) must be < block_size ({block_size})"
            )

        self.input_dim = input_dim
        self.block_size = block_size
        self.overlap = overlap
        self.allow_partial = allow_partial
        self._slices: List[Tuple[int, int]] = self._build_slices()

    def _build_slices(self) -> List[Tuple[int, int]]:
        """Compute (start, end) slices for each block."""
        step = self.block_size - self.overlap
        if step < 1:
            step = 1

        slices: List[Tuple[int, int]] = []
        start = 0
        while start < self.input_dim:
            end = min(start + self.block_size, self.input_dim)
            slices.append((start, end))
            start += step
            # If last block is complete, stop
            if end >= self.input_dim:
                break

        return slices

    @property
    def n_blocks(self) -> int:
        """Number of blocks in the decomposition."""
        return len(self._slices)

    @property
    def slices(self) -> List[Tuple[int, int]]:
        """List of (start, end) index pairs for each block."""
        return list(self._slices)

    def block_dims(self) -> List[int]:
        """Dimensionality of each block (end - start)."""
        return [end - start for start, end in self._slices]

    def extract(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract blocks from an input tensor along the last dimension.

        Args:
            x: Tensor of shape (*, input_dim).

        Returns:
            List of tensors, each of shape (*, block_dim_i).
        """
        return [x[..., start:end] for start, end in self._slices]

    def reconstruct(
        self,
        blocks: List[torch.Tensor],
        method: str = "mean",
    ) -> torch.Tensor:
        """
        Reconstruct a full-dim tensor from blocks.

        For non-overlapping decompositions this is trivial concatenation.
        For overlapping decompositions, overlapping regions are averaged
        (``method="mean"``) or summed (``method="sum"``).

        Args:
            blocks: List of block tensors, each (*, block_dim_i).
            method: "mean" or "sum" for overlapping regions.

        Returns:
            Tensor of shape (*, input_dim).
        """
        shape = blocks[0].shape[:-1]
        device = blocks[0].device
        dtype = blocks[0].dtype

        output = torch.zeros(*shape, self.input_dim, device=device, dtype=dtype)
        counts = torch.zeros(*shape, self.input_dim, device=device, dtype=dtype)

        for block, (start, end) in zip(blocks, self._slices):
            output[..., start:end] += block[..., : end - start]
            counts[..., start:end] += 1

        if method == "mean":
            output = output / counts.clamp(min=1)
        # "sum" leaves output as-is

        return output

    def overlapping_indices(self) -> List[List[int]]:
        """
        For each block, return the list of input dims that are shared
        with at least one other block. Empty if no overlap.
        """
        if self.overlap == 0:
            return [[] for _ in self._slices]

        dim_to_blocks: Dict[int, List[int]] = {}
        for bidx, (start, end) in enumerate(self._slices):
            for d in range(start, end):
                dim_to_blocks.setdefault(d, []).append(bidx)

        result: List[List[int]] = []
        for bidx, (start, end) in enumerate(self._slices):
            shared = [
                d for d in range(start, end)
                if len(dim_to_blocks.get(d, [])) > 1
            ]
            result.append(shared)
        return result

    def __repr__(self) -> str:
        return (
            f"BlockDecomposition(input_dim={self.input_dim}, "
            f"block_size={self.block_size}, overlap={self.overlap}, "
            f"n_blocks={self.n_blocks})"
        )


# ---------------------------------------------------------------------------
# SplineLinearHD
# ---------------------------------------------------------------------------

class SplineLinearHD(nn.Module):
    """
    High-dimensional SplineLinear using block-wise Eisenstein lattices.

    Decomposes a high-dim linear transform into blocks, each parameterised
    by its own SplineLinear. A learnable mixing layer combines block outputs.

    For input dim 768 with block_size=16:
      - 48 blocks, each with its own SplineLinear(in=16, out=out_features)
      - Mixing weights combine the 48 block contributions

    The mixing layer uses a low-rank parameterisation:
      mix = softmax(M @ x_block_reduced) where M is (n_blocks, n_blocks)

    Args:
        in_features:       Input dimensionality (64–768+).
        out_features:      Output dimensionality.
        block_size:        Dimensions per block. Default 16.
        overlap:           Overlap between adjacent blocks. Default 0.
        n_control_points:  Control points per block's SplineLinear.
        basis:             Basis function for each block's SplineLinear.
        mixing_rank:       Rank of the mixing weight matrix. If None,
                           uses full n_blocks × n_blocks.
        bias:              If True, add learnable bias to output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int = 16,
        overlap: int = 0,
        n_control_points: int = 16,
        basis: str = "eisenstein",
        mixing_rank: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.overlap = overlap
        self.n_control_points = n_control_points
        self.basis = basis
        self.mixing_rank = mixing_rank

        # Build decomposition
        self.decomp = BlockDecomposition(
            input_dim=in_features,
            block_size=block_size,
            overlap=overlap,
        )

        # Per-block SplineLinear layers
        self.block_layers = nn.ModuleList()
        for bd in self.decomp.block_dims():
            self.block_layers.append(
                SplineLinear(
                    in_features=bd,
                    out_features=out_features,
                    n_control_points=n_control_points,
                    basis=basis,
                    bias=False,  # bias applied at the end, not per-block
                )
            )

        # Mixing weights: (n_blocks, n_blocks) or low-rank approximation
        n_blocks = self.decomp.n_blocks
        if mixing_rank is not None and mixing_rank < n_blocks:
            # Low-rank: U (n_blocks, r) @ V (r, n_blocks)
            self._mix_u = nn.Parameter(
                torch.randn(n_blocks, mixing_rank) / math.sqrt(mixing_rank)
            )
            self._mix_v = nn.Parameter(
                torch.randn(mixing_rank, n_blocks) / math.sqrt(n_blocks)
            )
            self._mix_bias_param = nn.Parameter(torch.zeros(n_blocks))
        else:
            # Full mixing matrix
            self._mix_weight = nn.Parameter(
                torch.eye(n_blocks) / n_blocks
            )

        # Output bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def _mixing_weights(self) -> torch.Tensor:
        """Compute the mixing weight matrix, shape (n_blocks, n_blocks)."""
        if self.mixing_rank is not None:
            w = self._mix_u @ self._mix_v + self._mix_bias_param.unsqueeze(0)
        else:
            w = self._mix_weight
        # Row-wise softmax for normalized mixing
        return F.softmax(w, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract blocks, apply per-block SplineLinear,
        mix results.

        Args:
            x: Input tensor of shape (*, in_features).

        Returns:
            Output tensor of shape (*, out_features).
        """
        blocks = self.decomp.extract(x)  # list of (*, block_dim_i)

        # Per-block outputs: (n_blocks, *, out_features)
        block_outputs = []
        for i, (blk, layer) in enumerate(zip(blocks, self.block_layers)):
            block_outputs.append(layer(blk))

        # Stack: (n_blocks, batch, out_features) -> (batch, n_blocks, out_features)
        stacked = torch.stack(block_outputs, dim=-2)  # (*, n_blocks, out_features)

        # Mixing: (n_blocks, n_blocks) @ (*, n_blocks, out_features)
        mix_w = self._mixing_weights()  # (n_blocks, n_blocks)

        # Einstein summation: for each sample, weighted sum over blocks
        # mix_w: (n_blocks_out, n_blocks_in)
        # stacked: (*, n_blocks_in, out_features)
        # result: (*, n_blocks_out, out_features) -> sum over n_blocks_out -> (*, out_features)
        output = torch.einsum("ij,...jf->...if", mix_w, stacked)  # (*, n_blocks, out_features)
        output = output.sum(dim=-2)  # (*, out_features)

        if self.bias is not None:
            output = output + self.bias

        return output

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_equivalent_dense_params(self) -> int:
        n = self.in_features * self.out_features
        if self.bias is not None:
            n += self.out_features
        return n

    def compression_ratio(self) -> float:
        dense = self.num_equivalent_dense_params()
        actual = self.num_trainable_params()
        return float(dense) / float(max(actual, 1))

    def per_block_compression(self) -> List[Dict[str, float]]:
        """Compression stats for each block."""
        stats = []
        for i, layer in enumerate(self.block_layers):
            stats.append({
                "block": i,
                "block_dim": layer.in_features,
                "dense_params": layer.num_equivalent_dense_params(),
                "spline_params": layer.num_trainable_params(),
                "compression": layer.compression_ratio(),
            })
        return stats

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"block_size={self.block_size}, "
            f"overlap={self.overlap}, "
            f"n_blocks={self.decomp.n_blocks}, "
            f"n_control_points={self.n_control_points}, "
            f"basis='{self.basis}', "
            f"mixing_rank={self.mixing_rank}, "
            f"bias={self.bias is not None}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_features={self.in_features!r}, out_features={self.out_features!r}, block_size={self.block_size!r}, overlap={self.overlap!r}, n_control_points={self.n_control_points!r})"



# ---------------------------------------------------------------------------
# AdaptiveCompression
# ---------------------------------------------------------------------------

class AdaptiveCompression:
    """
    Measures per-block entropy and selects compression strategy.

    For each block in a decomposition:
    1. Compute the entropy of the block's input distribution.
    2. Low-entropy blocks → aggressive spline compression.
    3. High-entropy blocks → less compression or dense passthrough.

    Entropy is estimated as the mean log-variance of the block's features
    over a calibration dataset. Blocks with low variance have redundant
    information and compress well.

    Args:
        decomp:       BlockDecomposition to measure.
        aggressive_threshold:  Entropy below this → aggressive compression.
        passthrough_threshold: Entropy above this → no compression (dense).
    """

    def __init__(
        self,
        decomp: BlockDecomposition,
        aggressive_threshold: float = 0.5,
        passthrough_threshold: float = 2.0,
    ) -> None:
        self.decomp = decomp
        self.aggressive_threshold = aggressive_threshold
        self.passthrough_threshold = passthrough_threshold

    def measure_entropy(
        self,
        data: torch.Tensor,
    ) -> List[float]:
        """
        Estimate per-block entropy from a calibration dataset.

        Uses log-variance as a proxy for entropy:
        H(block) ≈ 0.5 * log(Var(block) + ε)

        Args:
            data: Calibration tensor of shape (N, input_dim).

        Returns:
            List of entropy estimates, one per block.
        """
        blocks = self.decomp.extract(data)
        entropies = []
        for blk in blocks:
            # Per-dimension variance, then mean log-variance
            var = blk.var(dim=0)  # (block_dim,)
            log_var = (var + 1e-8).log()
            entropy = log_var.mean().item()
            entropies.append(entropy)
        return entropies

    def select_strategy(
        self,
        entropies: List[float],
    ) -> List[str]:
        """
        Select compression strategy per block.

        Returns:
            List of strategies: "aggressive", "moderate", or "passthrough".
        """
        strategies = []
        for h in entropies:
            if h < self.aggressive_threshold:
                strategies.append("aggressive")
            elif h > self.passthrough_threshold:
                strategies.append("passthrough")
            else:
                strategies.append("moderate")
        return strategies

    def recommend_control_points(
        self,
        strategies: List[str],
        base_control_points: int = 16,
    ) -> List[int]:
        """
        Recommend n_control_points per block based on strategy.

        aggressive  → base // 2  (fewer control points, more compression)
        moderate    → base
        passthrough → 0 (use dense nn.Linear instead)
        """
        recommendations = []
        for s in strategies:
            if s == "aggressive":
                recommendations.append(max(base_control_points // 2, 4))
            elif s == "moderate":
                recommendations.append(base_control_points)
            else:  # passthrough
                recommendations.append(0)
        return recommendations

    def build_layer(
        self,
        out_features: int,
        data: torch.Tensor,
        base_control_points: int = 16,
        basis: str = "eisenstein",
    ) -> nn.Module:
        """
        Build a mixed SplineLinear/dense layer based on measured entropy.

        Blocks recommended as passthrough get nn.Linear, all others get
        SplineLinear with the recommended number of control points.

        Returns an AdaptiveLinear module that combines them.
        """
        entropies = self.measure_entropy(data)
        strategies = self.select_strategy(entropies)
        ctrl_recs = self.recommend_control_points(strategies, base_control_points)

        return AdaptiveLinear(
            decomp=self.decomp,
            out_features=out_features,
            control_points_per_block=ctrl_recs,
            strategies=strategies,
            basis=basis,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(decomp={self.decomp!r}, aggressive_threshold={self.aggressive_threshold!r}, passthrough_threshold={self.passthrough_threshold!r})"



class AdaptiveLinear(nn.Module):
    """
    Mixed SplineLinear/dense layer for adaptive compression.

    Each block uses either SplineLinear or nn.Linear depending on the
    compression strategy determined by AdaptiveCompression.
    """

    def __init__(
        self,
        decomp: BlockDecomposition,
        out_features: int,
        control_points_per_block: List[int],
        strategies: List[str],
        basis: str = "eisenstein",
    ) -> None:
        super().__init__()
        self.decomp = decomp
        self.out_features = out_features
        self.strategies = strategies

        self.block_layers = nn.ModuleList()
        for i, ((start, end), n_cp, strategy) in enumerate(
            zip(decomp.slices, control_points_per_block, strategies)
        ):
            bd = end - start
            if strategy == "passthrough" or n_cp == 0:
                self.block_layers.append(nn.Linear(bd, out_features, bias=False))
            else:
                self.block_layers.append(
                    SplineLinear(
                        in_features=bd,
                        out_features=out_features,
                        n_control_points=n_cp,
                        basis=basis,
                        bias=False,
                    )
                )

        # Mixing weights
        n_blocks = decomp.n_blocks
        self._mix_weight = nn.Parameter(
            torch.eye(n_blocks) / n_blocks
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks = self.decomp.extract(x)
        block_outputs = [layer(blk) for blk, layer in zip(blocks, self.block_layers)]
        stacked = torch.stack(block_outputs, dim=-2)
        mix_w = F.softmax(self._mix_weight, dim=1)
        output = torch.einsum("ij,...jf->...if", mix_w, stacked)
        output = output.sum(dim=-2)
        if self.bias is not None:
            output = output + self.bias
        return output

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_equivalent_dense_params(self) -> int:
        return self.decomp.input_dim * self.out_features + self.out_features

    def compression_ratio(self) -> float:
        return float(self.num_equivalent_dense_params()) / max(self.num_trainable_params(), 1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(decomp={self.decomp!r}, out_features={self.out_features!r}, control_points_per_block={self.control_points_per_block!r}, strategies={self.strategies!r}, basis={self.basis!r})"



# ---------------------------------------------------------------------------
# EmbeddingBenchmark
# ---------------------------------------------------------------------------

class EmbeddingBenchmark:
    """
    Benchmark SplineLinearHD on synthetic embeddings of increasing dim.

    Tests dimensions: 64, 128, 256, 512, 768 (configurable).
    For each dimension:
    1. Generate synthetic embeddings with cluster structure.
    2. Train SplineLinearHD for a few epochs.
    3. Measure compression ratio, accuracy retention, inference time,
       parameter count vs dense baseline.

    Args:
        dimensions:  List of embedding dimensions to test.
        block_size:  Block size for decomposition.
        n_control_points: Control points per block.
        n_classes:   Number of output classes.
        n_samples:   Synthetic samples per dimension.
        n_epochs:    Training epochs per benchmark.
        lr:          Learning rate.
        device:      Torch device.
    """

    DEFAULT_DIMS = [64, 128, 256, 512, 768]

    def __init__(
        self,
        dimensions: Optional[List[int]] = None,
        block_size: int = 16,
        n_control_points: int = 16,
        n_classes: int = 10,
        n_samples: int = 1000,
        n_epochs: int = 10,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
    ) -> None:
        self.dimensions = dimensions or self.DEFAULT_DIMS
        self.block_size = block_size
        self.n_control_points = n_control_points
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def generate_synthetic_embeddings(
        self,
        dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic embeddings with cluster structure.

        Creates n_classes clusters with random centres in R^dim.
        Each sample is drawn from a Gaussian around its cluster centre.

        Returns:
            (embeddings, labels) of shape (n_samples, dim) and (n_samples,).
        """
        n_per_class = self.n_samples // self.n_classes
        remainder = self.n_samples % self.n_classes

        centres = torch.randn(self.n_classes, dim) * 2.0
        embeddings = []
        labels = []

        for c in range(self.n_classes):
            n_c = n_per_class + (1 if c < remainder else 0)
            noise = torch.randn(n_c, dim) * 0.5
            embeddings.append(centres[c].unsqueeze(0) + noise)
            labels.append(torch.full((n_c,), c, dtype=torch.long))

        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)

        # Shuffle
        perm = torch.randperm(len(labels))
        return embeddings[perm], labels[perm]

    def benchmark_dimension(
        self,
        dim: int,
    ) -> Dict[str, object]:
        """
        Run benchmark for a single embedding dimension.

        Returns dict with keys:
            dim, spline_params, dense_params, compression_ratio,
            spline_accuracy, dense_accuracy, accuracy_retention,
            spline_inference_ms, dense_inference_ms, inference_speedup
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking dim={dim}")
        print(f"{'='*60}")

        # Generate data
        X, y = self.generate_synthetic_embeddings(dim)
        X, y = X.to(self.device), y.to(self.device)

        # Split 80/20
        n_train = int(0.8 * len(y))
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        # ---- Dense baseline ----
        dense_model = nn.Sequential(
            nn.Linear(dim, self.n_classes),
        ).to(self.device)

        dense_params = sum(p.numel() for p in dense_model.parameters())
        dense_acc = self._train_and_eval(dense_model, X_train, y_train, X_test, y_test)
        dense_time = self._measure_inference(dense_model, X_test)

        # ---- SplineLinearHD ----
        spline_model = SplineLinearHD(
            in_features=dim,
            out_features=self.n_classes,
            block_size=self.block_size,
            n_control_points=self.n_control_points,
            basis="eisenstein",
            bias=True,
        ).to(self.device)

        spline_params = spline_model.num_trainable_params()
        spline_acc = self._train_and_eval(spline_model, X_train, y_train, X_test, y_test)
        spline_time = self._measure_inference(spline_model, X_test, n_runs=5)

        compression = float(dense_params) / max(spline_params, 1)
        retention = spline_acc / max(dense_acc, 1e-8)

        result = {
            "dim": dim,
            "spline_params": spline_params,
            "dense_params": dense_params,
            "compression_ratio": compression,
            "spline_accuracy": spline_acc,
            "dense_accuracy": dense_acc,
            "accuracy_retention": retention,
            "spline_inference_ms": spline_time,
            "dense_inference_ms": dense_time,
        }

        print(f"  Dense:  {dense_params:>8d} params, {dense_acc:.4f} acc, {dense_time:.3f} ms")
        print(f"  Spline: {spline_params:>8d} params, {spline_acc:.4f} acc, {spline_time:.3f} ms")
        print(f"  Compression: {compression:.1f}x, Accuracy retention: {retention:.2%}")

        return result

    def _train_and_eval(
        self,
        model: nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
    ) -> float:
        """Train model and return test accuracy."""
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(self.n_epochs):
            # Mini-batch training
            perm = torch.randperm(len(y_train))
            batch_size = min(64, len(y_train))

            for i in range(0, len(y_train), batch_size):
                idx = perm[i:i + batch_size]
                xb, yb = X_train[idx], y_train[idx]

                optimizer.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            logits = model(X_test)
            preds = logits.argmax(dim=1)
            acc = (preds == y_test).float().mean().item()

        return acc

    def _measure_inference(
        self,
        model: nn.Module,
        X: torch.Tensor,
        n_runs: int = 10,
    ) -> float:
        """Measure average inference time in milliseconds."""
        model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                _ = model(X[:32])

            start = time.perf_counter()
            for _ in range(n_runs):
                _ = model(X[:32])
            elapsed = (time.perf_counter() - start) / n_runs

        return elapsed * 1000  # ms

    def run(self) -> List[Dict[str, object]]:
        """
        Run the full benchmark across all dimensions.

        Returns list of result dicts, one per dimension.
        """
        results = []
        for dim in self.dimensions:
            results.append(self.benchmark_dimension(dim))
        return results

    def summary(self, results: List[Dict[str, object]]) -> str:
        """Format benchmark results as a summary table."""
        lines = [
            f"\n{'Embedding Benchmark Results':^70s}",
            f"{'='*70}",
            f"{'Dim':>6s} | {'Dense Params':>12s} | {'Spline Params':>13s} | "
            f"{'Compress':>8s} | {'Dense Acc':>9s} | {'Spline Acc':>10s} | "
            f"{'Retention':>9s}",
            f"{'-'*6}-+-{'-'*12}-+-{'-'*13}-+-{'-'*8}-+-{'-'*9}-+-{'-'*10}-+-{'-'*9}",
        ]

        for r in results:
            lines.append(
                f"{r['dim']:>6d} | {r['dense_params']:>12d} | {r['spline_params']:>13d} | "
                f"{r['compression_ratio']:>7.1f}x | {r['dense_accuracy']:>8.4f} | "
                f"{r['spline_accuracy']:>9.4f} | {r['accuracy_retention']:>8.1%}"
            )

        lines.append(f"{'='*70}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dimensions={self.dimensions!r}, block_size={self.block_size!r}, n_control_points={self.n_control_points!r}, n_classes={self.n_classes!r}, n_samples={self.n_samples!r})"

