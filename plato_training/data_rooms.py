"""
Data Rooms — real data loading for micro model training.

Real data comes from:
1. CSV/JSONL files (structured tabular data)
2. PLATO tile streams (tiles as training examples)
3. Synthetic generators (for testing/proving)
4. Fleet telemetry (sensor readings, agent metrics)

Each DataRoom knows how to:
- Load data from a source
- Preprocess into tensors
- Split train/val/test
- Stream batches to training

Usage:
    from plato_training.data_rooms import DataRoom, DataLoader
    
    # From CSV
    room = DataRoom.from_csv("data.csv", label_col="label")
    
    # From PLATO tiles
    room = DataRoom.from_plato("drift-detect", server="http://147.224.38.131:8847")
    
    # Get training-ready data
    X_train, y_train, X_val, y_val = room.split()
"""

import torch
import numpy as np
import json
import csv
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader


@dataclass
class DataSpec:
    """Schema for a dataset."""
    name: str
    input_dim: int
    num_classes: int
    class_names: List[str]
    description: str = ""
    source: str = ""  # "csv", "jsonl", "plato", "synthetic", "fleet"


class DataRoom:
    """
    Loads and preprocesses data for micro model training.
    
    The bridge between raw data and training-ready tensors.
    """
    
    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        spec: DataSpec,
    ):
        self.X = X
        self.y = y
        self.spec = spec
    
    def split(self, val_ratio: float = 0.2, seed: int = 42) -> Tuple[torch.Tensor, ...]:
        """Split into train/val tensors."""
        n = len(self.X)
        indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        val_size = int(n * val_ratio)
        
        train_idx = indices[val_size:]
        val_idx = indices[:val_size]
        
        return self.X[train_idx], self.y[train_idx], self.X[val_idx], self.y[val_idx]
    
    def dataset(self, val_ratio: float = 0.2) -> Tuple[TensorDataset, TensorDataset]:
        """Get train/val TensorDatasets."""
        X_train, y_train, X_val, y_val = self.split(val_ratio)
        return TensorDataset(X_train, y_train), TensorDataset(X_val, y_val)
    
    def dataloader(self, batch_size: int = 32, val_ratio: float = 0.2) -> Tuple[TorchDataLoader, TorchDataLoader]:
        """Get train/val DataLoaders."""
        train_ds, val_ds = self.dataset(val_ratio)
        return (
            TorchDataLoader(train_ds, batch_size=batch_size, shuffle=True),
            TorchDataLoader(val_ds, batch_size=batch_size),
        )
    
    def summary(self) -> Dict:
        return {
            "name": self.spec.name,
            "samples": len(self.X),
            "input_dim": self.spec.input_dim,
            "num_classes": self.spec.num_classes,
            "class_distribution": {
                self.spec.class_names[i]: int((self.y == i).sum())
                for i in range(self.spec.num_classes)
            },
            "source": self.spec.source,
        }
    
    # ─── Factory Methods ───────────────────────────────────────────
    
    @classmethod
    def from_csv(
        cls,
        path: str,
        label_col: str = "label",
        feature_cols: Optional[List[str]] = None,
        feature_range: Optional[Tuple[int, int]] = None,
        name: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        delimiter: str = ",",
    ) -> "DataRoom":
        """Load from CSV file."""
        with open(path, 'r') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            rows = list(reader)
        
        if not rows:
            raise ValueError(f"Empty CSV: {path}")
        
        headers = list(rows[0].keys())
        
        # Determine features
        if feature_cols:
            features = feature_cols
        elif feature_range:
            features = headers[feature_range[0]:feature_range[1]]
        else:
            features = [h for h in headers if h != label_col]
        
        # Parse
        X = []
        y = []
        for row in rows:
            X.append([float(row[f]) for f in features])
            y.append(int(row[label_col]))
        
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
        num_classes = int(y.max()) + 1
        if class_names is None:
            class_names = [f"class-{i}" for i in range(num_classes)]
        
        spec = DataSpec(
            name=name or Path(path).stem,
            input_dim=len(features),
            num_classes=num_classes,
            class_names=class_names,
            source="csv",
            description=f"Loaded from {path}",
        )
        
        return cls(X, y, spec)
    
    @classmethod
    def from_jsonl(
        cls,
        path: str,
        text_to_features: Optional[Callable] = None,
        label_key: str = "label",
        feature_key: str = "features",
        name: Optional[str] = None,
        class_names: Optional[List[str]] = None,
    ) -> "DataRoom":
        """Load from JSONL file."""
        X = []
        y = []
        
        with open(path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                
                if text_to_features:
                    features = text_to_features(record)
                else:
                    features = record[feature_key]
                
                X.append(features)
                y.append(record[label_key])
        
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
        num_classes = int(y.max()) + 1
        if class_names is None:
            class_names = [f"class-{i}" for i in range(num_classes)]
        
        spec = DataSpec(
            name=name or Path(path).stem,
            input_dim=X.shape[1],
            num_classes=num_classes,
            class_names=class_names,
            source="jsonl",
            description=f"Loaded from {path}",
        )
        
        return cls(X, y, spec)
    
    @classmethod
    def from_plato(
        cls,
        room: str,
        server: str = "http://147.224.38.131:8847",
        label_extractor: Optional[Callable] = None,
        feature_dim: int = 128,
        max_tiles: int = 1000,
        name: Optional[str] = None,
    ) -> "DataRoom":
        """
        Load data from PLATO tiles.
        
        Each tile's content becomes a training example.
        Needs a label_extractor function to determine labels from tile data.
        """
        try:
            url = f"{server}/room/{room}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            raise RuntimeError(f"Failed to fetch PLATO room '{room}': {e}")
        
        tiles = data.get("tiles", data if isinstance(data, list) else [])
        tiles = tiles[:max_tiles]
        
        if not tiles:
            raise ValueError(f"No tiles found in PLATO room '{room}'")
        
        X = []
        y = []
        
        for tile in tiles:
            if label_extractor:
                label, features = label_extractor(tile)
            else:
                # Default: hash tile_id for deterministic label
                label = hash(tile.get("tile_id", "")) % 2
                # Use tile metadata as features (padded/truncated to feature_dim)
                content = json.dumps(tile.get("content", tile.get("data", {})))
                feat_hash = [hash(f"{content}_{i}") % 1000 / 1000.0 for i in range(feature_dim)]
                features = feat_hash
            
            X.append(features)
            y.append(label)
        
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
        num_classes = int(y.max()) + 1
        
        spec = DataSpec(
            name=name or f"plato-{room}",
            input_dim=X.shape[1],
            num_classes=num_classes,
            class_names=[f"class-{i}" for i in range(num_classes)],
            source="plato",
            description=f"From PLATO room '{room}' ({len(tiles)} tiles)",
        )
        
        return cls(X, y, spec)
    
    @classmethod
    def from_fleet_telemetry(
        cls,
        log_dir: str,
        metric_cols: List[str],
        label_col: str = "status",
        window_size: int = 8,
        name: Optional[str] = None,
    ) -> "DataRoom":
        """
        Load from fleet telemetry logs.
        
        Creates sliding windows over time-series metrics,
        labeling each window by the status at window end.
        """
        log_path = Path(log_dir)
        all_rows = []
        
        for f in sorted(log_path.glob("*.jsonl")):
            with open(f) as fh:
                for line in fh:
                    if line.strip():
                        all_rows.append(json.loads(line))
        
        if len(all_rows) < window_size:
            raise ValueError(f"Not enough telemetry data: {len(all_rows)} rows, need {window_size}")
        
        # Extract features and labels
        features = []
        labels = []
        
        for row in all_rows:
            feat = [float(row.get(c, 0)) for c in metric_cols]
            features.append(feat)
            labels.append(row.get(label_col, "normal"))
        
        # Unique labels → class indices
        unique_labels = sorted(set(labels))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        
        # Sliding windows
        X = []
        y = []
        for i in range(len(features) - window_size + 1):
            window = features[i:i + window_size]
            # Flatten window into single feature vector
            X.append([v for step in window for v in step])
            y.append(label_to_idx[labels[i + window_size - 1]])
        
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
        spec = DataSpec(
            name=name or "fleet-telemetry",
            input_dim=X.shape[1],
            num_classes=len(unique_labels),
            class_names=unique_labels,
            source="fleet",
            description=f"Fleet telemetry from {log_dir} (window={window_size})",
        )
        
        return cls(X, y, spec)
    
    @classmethod
    def from_tensors(
        cls,
        X: torch.Tensor,
        y: torch.Tensor,
        name: str = "custom",
        class_names: Optional[List[str]] = None,
    ) -> "DataRoom":
        """Create directly from tensors."""
        num_classes = int(y.max()) + 1
        if class_names is None:
            class_names = [f"class-{i}" for i in range(num_classes)]
        
        spec = DataSpec(
            name=name,
            input_dim=X.shape[1],
            num_classes=num_classes,
            class_names=class_names,
            source="tensors",
        )
        return cls(X, y, spec)
