"""
Low-Rank Linear — learned decomposition for compressed weights.

Unlike SplineLinear (which interpolates from control points), LowRankLinear
learns a direct factorization W ≈ U @ V where U is (in, rank) and V is (rank, out).

Honest comparison on topic-classify (256-dim):
  Dense:      100%    20,933 params
  LowRank-16:  80%     7,493 params  (2.8x compression, 80% of accuracy)
  LowRank-32:  80%    14,661 params
  Spline-16:   31%       485 params  (43x compression, but too smooth)
  Spline-64:   41%       581 params

The spline is great for smooth tasks (drift-detect: 100% at 20x compression).
Low-rank is better for tasks needing sharp decision boundaries.
Use the right tool for the right task.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict
from dataclasses import dataclass


class LowRankLinear(nn.Module):
    """
    Low-rank factorized linear layer: W ≈ U @ V + bias.
    
    Params: in_features * rank + rank * out_features (+ out_features bias)
    Compression: (in * out) / (in * rank + rank * out)
    For 512×512 at rank 16: 262,144 → 16,384 (16:1)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        bias: bool = True,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        self.U = nn.Parameter(torch.randn(in_features, rank) * init_scale)
        self.V = nn.Parameter(torch.randn(rank, out_features) * init_scale)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.U @ self.V + (self.bias if self.bias is not None else 0)
    
    def num_low_rank_params(self) -> int:
        count = self.U.numel() + self.V.numel()
        if self.bias is not None:
            count += self.bias.numel()
        return count
    
    def num_equivalent_dense_params(self) -> int:
        count = self.in_features * self.out_features
        if self.bias is not None:
            count += self.out_features
        return count
    
    def compression_ratio(self) -> float:
        return self.num_equivalent_dense_params() / max(self.num_low_rank_params(), 1)


class LowRankClassifier(nn.Module):
    """Classifier using low-rank layers."""
    
    def __init__(self, input_dim: int, hidden: int, num_classes: int, rank: int = 16):
        super().__init__()
        self.W_query = LowRankLinear(input_dim, hidden, rank=rank)
        self.W_value = LowRankLinear(hidden, hidden, rank=rank)
        self.out_head = nn.Linear(hidden, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.relu(self.W_query(x))
        x = self.dropout(x)
        x = self.relu(self.W_value(x))
        return self.out_head(x)


def inject_low_rank(
    model: nn.Module,
    rank: int = 16,
    target_modules: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Replace nn.Linear with LowRankLinear."""
    injection_map = {}
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if target_modules and not any(t in name for t in target_modules):
                continue
            lr = LowRankLinear(
                module.in_features, module.out_features,
                rank=rank, bias=module.bias is not None,
            )
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], lr)
            injection_map[name] = "low-rank"
    return injection_map


# ─── Variant Auto-Selection ────────────────────────────────────────

@dataclass
class CompressionMethod:
    """Describes a compression strategy."""
    name: str
    method: str        # "spline", "lowrank", "dense", "lora"
    best_for: str
    compression: float
    accuracy_retention: float  # estimated
    notes: str = ""


VARIANT_GUIDE = {
    "smooth": CompressionMethod(
        name="spline",
        method="spline",
        best_for="Continuous signals, drift detection, regression",
        compression=20.0,
        accuracy_retention=0.95,
        notes="IDW interpolation excels at smooth functions",
    ),
    "sharp": CompressionMethod(
        name="low-rank",
        method="lowrank",
        best_for="Classification, sharp boundaries, categorical",
        compression=10.0,
        accuracy_retention=0.80,
        notes="Learned factorization handles discontinuities",
    ),
    "adaptive": CompressionMethod(
        name="lora",
        method="lora",
        best_for="Fine-tuning pretrained models, multi-task",
        compression=5.0,
        accuracy_retention=0.85,
        notes="Best when you have a base model to adapt",
    ),
}


def recommend_variant(task_description: str) -> str:
    """Recommend the best variant based on task characteristics."""
    desc = task_description.lower()
    
    # Smooth/continuous tasks → spline
    smooth_keywords = ["drift", "sensor", "continuous", "regression", "smooth", "anomaly", "signal"]
    if any(kw in desc for kw in smooth_keywords):
        return "spline"
    
    # Categorical/classification → low-rank
    sharp_keywords = ["classify", "detect", "intent", "topic", "priority", "category"]
    if any(kw in desc for kw in sharp_keywords):
        return "lowrank"
    
    # Default to low-rank for unknown tasks
    return "lowrank"
