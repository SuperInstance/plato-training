"""
Data Pipeline — unified real data loading for micro model training.

Chains: DataLoader → Transform → FeatureStore → DataSplitter → Pipeline

Usage:
    pipeline = Pipeline(
        source=GitCommitLoader(repos=["plato-training"]),
        transform=CommitFeatureExtractor(),
        split=TemporalSplitter(train=0.7, val=0.15, test=0.15),
    )
    train, val, test = pipeline.run()
"""


__all__ = ['CSVLoader', 'ColumnSchema', 'CommitFeatureExtractor', 'DataContainer', 'DataLoader', 'DataSchema', 'DataSplitter', 'DataVersion', 'FeatureLineage', 'FeatureStore', 'FunctionTransform', 'GitCommitLoader', 'IdentityTransform', 'JSONLLoader', 'Pipeline', 'PlatoLoader', 'RandomSplitter', 'StratifiedSplitter', 'TemporalSplitter', 'Transform']

import hashlib
import json
import csv
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass, field

import numpy as np

from .types import LamportClock, content_hash
from .fleet_miner import FleetMiner, CommitPoint, FLEET_REPOS


# ─── Core Data Structures ──────────────────────────────────────────


@dataclass
class ColumnSchema:
    """Schema for a single column."""
    name: str
    dtype: type = float
    required: bool = True
    default: Any = None


@dataclass
class DataSchema:
    """Schema defining expected columns and types for a DataRoom."""
    columns: List[ColumnSchema]
    name: str = "unnamed"
    description: str = ""

    @property
    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    def validate_row(self, row: Dict[str, Any]) -> List[str]:
        """Validate a row against the schema. Returns list of errors."""
        errors = []
        for col in self.columns:
            if col.required and col.name not in row:
                errors.append(f"Missing required column: {col.name}")
            elif col.name in row and row[col.name] is not None:
                try:
                    col.dtype(row[col.name])
                except (ValueError, TypeError):
                    errors.append(
                        f"Column '{col.name}' expected {col.dtype.__name__}, "
                        f"got {type(row[col.name]).__name__}: {row[col.name]!r}"
                    )
        return errors


@dataclass
class DataVersion:
    """Version stamp for a DataRoom's data."""
    clock: LamportClock
    content_hash: str
    row_count: int
    sealed: bool = False
    timestamp: float = field(default_factory=time.time)


class DataContainer:
    """
    Named, versioned, schema-validated data container.

    Data is stored as a list of dicts (rows).
    Once sealed, data is immutable (content-addressed).
    """

    def __init__(
        self,
        name: str,
        schema: DataSchema,
        clock: Optional[LamportClock] = None,
    ):
        self.name = name
        self.schema = schema
        self.clock = clock or LamportClock()
        self._rows: List[Dict[str, Any]] = []
        self._version: Optional[DataVersion] = None
        self._sealed = False

    @property
    def rows(self) -> List[Dict[str, Any]]:
        return list(self._rows)

    @property
    def row_count(self) -> int:
        return len(self._rows)

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def version(self) -> Optional[DataVersion]:
        return self._version

    def add_row(self, row: Dict[str, Any]) -> List[str]:
        """
        Add a row. Returns validation errors (empty if ok).
        Raises if sealed.
        """
        if self._sealed:
            raise RuntimeError(f"DataContainer '{self.name}' is sealed — immutable")
        errors = self.schema.validate_row(row)
        if errors:
            return errors
        # Coerce types and fill defaults
        coerced = {}
        for col in self.schema.columns:
            if col.name in row:
                coerced[col.name] = col.dtype(row[col.name])
            elif col.default is not None:
                coerced[col.name] = col.dtype(col.default)
            elif not col.required:
                coerced[col.name] = None
            else:
                coerced[col.name] = row[col.name]  # will fail
        self._rows.append(coerced)
        self._bump_version()
        return []

    def add_rows(self, rows: List[Dict[str, Any]]) -> List[str]:
        """Add multiple rows. Returns all errors."""
        all_errors = []
        for row in rows:
            errors = self.add_row(row)
            if errors:
                all_errors.extend(errors)
        return all_errors

    def seal(self) -> DataVersion:
        """Seal the container — no more modifications allowed."""
        chash = self._compute_hash()
        self._version = DataVersion(
            clock=self.clock,
            content_hash=chash,
            row_count=len(self._rows),
            sealed=True,
        )
        self._sealed = True
        return self._version

    def _bump_version(self):
        self.clock.tick()
        chash = self._compute_hash()
        self._version = DataVersion(
            clock=self.clock,
            content_hash=chash,
            row_count=len(self._rows),
            sealed=False,
        )

    def _compute_hash(self) -> str:
        blob = json.dumps(self._rows, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def to_numpy(self, columns: Optional[List[str]] = None) -> np.ndarray:
        """Export as numpy array."""
        cols = columns or self.schema.column_names
        return np.array([[row.get(c) for c in cols] for row in self._rows], dtype=np.float64)

    def get_column(self, name: str) -> List[Any]:
        return [row.get(name) for row in self._rows]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, schema={self.schema!r}, clock={self.clock!r})"



# ─── DataLoader Sources ────────────────────────────────────────────


class DataLoader:
    """Base class for data loaders."""
    def load(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"



class CSVLoader(DataLoader):
    """Load data from CSV files."""

    def __init__(
        self,
        path: str,
        delimiter: str = ",",
        encoding: str = "utf-8",
        max_rows: Optional[int] = None,
    ):
        self.path = path
        self.delimiter = delimiter
        self.encoding = encoding
        self.max_rows = max_rows

    def load(self) -> List[Dict[str, Any]]:
        rows = []
        with open(self.path, "r", encoding=self.encoding) as f:
            reader = csv.DictReader(f, delimiter=self.delimiter)
            for i, row in enumerate(reader):
                if self.max_rows and i >= self.max_rows:
                    break
                # Convert numeric strings
                parsed = {}
                for k, v in row.items():
                    if v is None:
                        parsed[k] = None
                        continue
                    try:
                        parsed[k] = int(v)
                    except ValueError:
                        try:
                            parsed[k] = float(v)
                        except ValueError:
                            parsed[k] = v
                rows.append(parsed)
        return rows

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self.path!r}, delimiter={self.delimiter!r}, encoding={self.encoding!r}, max_rows={self.max_rows!r})"



class JSONLLoader(DataLoader):
    """Load data from JSONL (one JSON object per line) files."""

    def __init__(self, path: str, max_rows: Optional[int] = None):
        self.path = path
        self.max_rows = max_rows

    def load(self) -> List[Dict[str, Any]]:
        rows = []
        with open(self.path, "r") as f:
            for i, line in enumerate(f):
                if self.max_rows and i >= self.max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self.path!r}, max_rows={self.max_rows!r})"



class PlatoLoader(DataLoader):
    """Load data from PLATO rooms via HTTP API."""

    def __init__(
        self,
        room: str,
        server: str = "http://147.224.38.131:8847",
        max_tiles: int = 1000,
        timeout: int = 10,
    ):
        self.room = room
        self.server = server
        self.max_tiles = max_tiles
        self.timeout = timeout

    def load(self) -> List[Dict[str, Any]]:
        url = f"{self.server}/room/{self.room}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())

        tiles = data.get("tiles", data if isinstance(data, list) else [])
        return tiles[:self.max_tiles]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(room={self.room!r}, server={self.server!r}, max_tiles={self.max_tiles!r}, timeout={self.timeout!r})"



class GitCommitLoader(DataLoader):
    """Load data from git commit history via FleetMiner."""

    def __init__(
        self,
        repos: Optional[List[str]] = None,
        max_per_repo: int = 200,
        org: str = "SuperInstance",
        token: Optional[str] = None,
        clone_dir: str = "/tmp/fleet-mine",
    ):
        self.repos = repos or FLEET_REPOS
        self.max_per_repo = max_per_repo
        self.org = org
        self.token = token
        self.clone_dir = clone_dir

    def load(self) -> List[Dict[str, Any]]:
        miner = FleetMiner(org=self.org, token=self.token, clone_dir=self.clone_dir)
        rows = []
        for repo in self.repos:
            try:
                commits = miner.mine_repo(repo, max_commits=self.max_per_repo)
                for c in commits:
                    rows.append(c.to_dict())
            except Exception:
                continue  # Skip repos that fail
        return rows

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(repos={self.repos!r}, max_per_repo={self.max_per_repo!r}, org={self.org!r}, token={self.token!r}, clone_dir={self.clone_dir!r})"



# ─── Transforms ────────────────────────────────────────────────────


class Transform:
    """Base class for data transforms."""
    def apply(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"



class CommitFeatureExtractor(Transform):
    """Extract features from git commit dicts."""

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
    ):
        self.feature_columns = feature_columns or [
            "files_changed", "insertions", "deletions",
            "is_merge", "size", "net_lines",
        ]

    def apply(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []
        for row in rows:
            if "sha" not in row:
                continue  # Not a commit row
            features = {}
            for col in self.feature_columns:
                if col == "size":
                    features[col] = row.get("insertions", 0) + row.get("deletions", 0)
                elif col == "net_lines":
                    features[col] = row.get("insertions", 0) - row.get("deletions", 0)
                else:
                    features[col] = row.get(col, 0)
            features["timestamp"] = row.get("timestamp", 0)
            features["repo"] = row.get("repo", "")
            features["author"] = row.get("author", "")
            features["label"] = 1 if row.get("files_changed", 0) > 5 else 0
            result.append(features)
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(feature_columns={self.feature_columns!r})"



class IdentityTransform(Transform):
    """Pass-through transform."""
    def apply(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return list(rows)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"



class FunctionTransform(Transform):
    """Apply an arbitrary function to rows."""
    def __init__(self, fn: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]):
        self.fn = fn

    def apply(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.fn(rows)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fn={self.fn!r})"



# ─── FeatureStore ──────────────────────────────────────────────────


@dataclass
class FeatureLineage:
    """Provenance for a computed feature."""
    feature_name: str
    source_columns: List[str]
    transform: str  # name of the transform that produced it
    description: str = ""


class FeatureStore:
    """
    Computed features with lineage tracking.

    Each feature knows its source columns and what transform produced it.
    Features can be invalidated if source data changes.
    """

    def __init__(self, name: str = "features"):
        self.name = name
        self._features: Dict[str, List[Any]] = {}
        self._lineage: Dict[str, FeatureLineage] = {}
        self._source_hash: Optional[str] = None
        self._n_rows: int = 0

    @property
    def feature_names(self) -> List[str]:
        return list(self._features.keys())

    @property
    def n_rows(self) -> int:
        return self._n_rows

    @property
    def lineage(self) -> Dict[str, FeatureLineage]:
        return dict(self._lineage)

    def compute(
        self,
        rows: List[Dict[str, Any]],
        feature_spec: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> "FeatureStore":
        """
        Compute features from rows.

        feature_spec: {feature_name: {"columns": [...], "transform": "sum"|"mean"|"identity"|...}}
        If None, all columns are used as-is.
        """
        if not rows:
            return self

        self._n_rows = len(rows)
        self._source_hash = hashlib.sha256(
            json.dumps(rows, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        if feature_spec:
            for feat_name, spec in feature_spec.items():
                cols = spec.get("columns", [])
                transform = spec.get("transform", "identity")
                values = self._apply_transform(rows, cols, transform)
                self._features[feat_name] = values
                self._lineage[feat_name] = FeatureLineage(
                    feature_name=feat_name,
                    source_columns=cols,
                    transform=transform,
                )
        else:
            # Auto-detect: use all keys as features
            all_keys = list(rows[0].keys())
            for key in all_keys:
                values = [row.get(key) for row in rows]
                # Only store if values are numeric-ish
                numeric_values = []
                for v in values:
                    if isinstance(v, (int, float, bool)):
                        numeric_values.append(float(v))
                    elif isinstance(v, str):
                        numeric_values.append(float(len(v)))
                    elif isinstance(v, list):
                        numeric_values.append(float(len(v)))
                    else:
                        numeric_values.append(0.0)
                self._features[key] = numeric_values
                self._lineage[key] = FeatureLineage(
                    feature_name=key,
                    source_columns=[key],
                    transform="identity",
                )

        return self

    def _apply_transform(
        self,
        rows: List[Dict[str, Any]],
        columns: List[str],
        transform: str,
    ) -> List[Any]:
        if transform == "identity":
            return [rows[i].get(columns[0]) if columns else None for i in range(len(rows))]
        elif transform == "sum":
            return [
                sum(float(rows[i].get(c, 0) or 0) for c in columns)
                for i in range(len(rows))
            ]
        elif transform == "mean":
            n = max(len(columns), 1)
            return [
                sum(float(rows[i].get(c, 0) or 0) for c in columns) / n
                for i in range(len(rows))
            ]
        elif transform == "ratio":
            if len(columns) >= 2:
                return [
                    (float(rows[i].get(columns[0], 0) or 0)) /
                    max(float(rows[i].get(columns[1], 0) or 0), 1e-9)
                    for i in range(len(rows))
                ]
            return [0.0] * len(rows)
        else:
            # Default: identity on first column
            return [rows[i].get(columns[0]) if columns else None for i in range(len(rows))]

    def get(self, feature_name: str) -> List[Any]:
        return self._features.get(feature_name, [])

    def to_numpy(self, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Export features as numpy array."""
        names = feature_names or self.feature_names
        if not names:
            return np.array([])
        arrays = [self._features.get(n, [0.0] * self._n_rows) for n in names]
        # Coerce to float
        float_arrays = []
        for arr in arrays:
            float_arr = []
            for v in arr:
                try:
                    float_arr.append(float(v))
                except (ValueError, TypeError):
                    float_arr.append(0.0)
            float_arrays.append(float_arr)
        return np.column_stack(float_arrays)

    def invalidate(self, source_hash: Optional[str] = None) -> List[str]:
        """
        Invalidate features whose source has changed.
        Returns list of invalidated feature names.
        """
        if source_hash and source_hash != self._source_hash:
            invalidated = list(self._features.keys())
            self._features.clear()
            self._lineage.clear()
            self._n_rows = 0
            return invalidated
        return []

    def is_valid(self) -> bool:
        return bool(self._features) and self._n_rows > 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"



# ─── DataSplitter ──────────────────────────────────────────────────


class DataSplitter:
    """Base class for data splitting."""
    def split(self, rows: List[Dict[str, Any]]) -> Tuple[List, List, List]:
        """Returns (train, val, test)."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"



class RandomSplitter(DataSplitter):
    """Random train/val/test split."""

    def __init__(self, train: float = 0.7, val: float = 0.15, test: float = 0.15, seed: int = 42):
        assert abs(train + val + test - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {train+val+test}"
        self.train_ratio = train
        self.val_ratio = val
        self.test_ratio = test
        self.seed = seed

    def split(self, rows: List[Dict[str, Any]]) -> Tuple[List, List, List]:
        n = len(rows)
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(n).tolist()

        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        train = [rows[i] for i in indices[:train_end]]
        val = [rows[i] for i in indices[train_end:val_end]]
        test = [rows[i] for i in indices[val_end:]]

        return train, val, test

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(train={self.train!r}, val={self.val!r}, test={self.test!r}, seed={self.seed!r})"



class TemporalSplitter(DataSplitter):
    """
    Temporal-aware train/val/test split.
    
    Orders by timestamp, then splits sequentially so that
    training data is always earlier than test data.
    No future leakage.
    """

    def __init__(
        self,
        train: float = 0.7,
        val: float = 0.15,
        test: float = 0.15,
        time_key: str = "timestamp",
    ):
        assert abs(train + val + test - 1.0) < 1e-6, f"Ratios must sum to 1.0"
        self.train_ratio = train
        self.val_ratio = val
        self.test_ratio = test
        self.time_key = time_key

    def split(self, rows: List[Dict[str, Any]]) -> Tuple[List, List, List]:
        # Sort by time key
        sorted_rows = sorted(rows, key=lambda r: float(r.get(self.time_key, 0)))
        n = len(sorted_rows)

        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        train = sorted_rows[:train_end]
        val = sorted_rows[train_end:val_end]
        test = sorted_rows[val_end:]

        return train, val, test

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(train={self.train!r}, val={self.val!r}, test={self.test!r}, time_key={self.time_key!r})"



class StratifiedSplitter(DataSplitter):
    """Stratified split by label column."""

    def __init__(
        self,
        label_key: str = "label",
        train: float = 0.7,
        val: float = 0.15,
        test: float = 0.15,
        seed: int = 42,
    ):
        assert abs(train + val + test - 1.0) < 1e-6
        self.label_key = label_key
        self.train_ratio = train
        self.val_ratio = val
        self.test_ratio = test
        self.seed = seed

    def split(self, rows: List[Dict[str, Any]]) -> Tuple[List, List, List]:
        rng = np.random.RandomState(self.seed)

        # Group by label
        by_label: Dict[Any, List[Dict]] = {}
        for row in rows:
            label = row.get(self.label_key, 0)
            by_label.setdefault(label, []).append(row)

        train, val, test = [], [], []
        for label, group in by_label.items():
            indices = rng.permutation(len(group)).tolist()
            n = len(indices)
            train_end = int(n * self.train_ratio)
            val_end = train_end + int(n * self.val_ratio)

            train.extend(group[i] for i in indices[:train_end])
            val.extend(group[i] for i in indices[train_end:val_end])
            test.extend(group[i] for i in indices[val_end:])

        return train, val, test

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(label_key={self.label_key!r}, train={self.train!r}, val={self.val!r}, test={self.test!r}, seed={self.seed!r})"



# ─── Pipeline ──────────────────────────────────────────────────────


class Pipeline:
    """
    Chains DataLoader → Transform → FeatureStore → DataSplitter.

    Usage:
        pipeline = Pipeline(
            source=GitCommitLoader(repos=["plato-training"]),
            transform=CommitFeatureExtractor(),
            split=TemporalSplitter(train=0.7, val=0.15, test=0.15),
        )
        train, val, test = pipeline.run()
    """

    def __init__(
        self,
        source: DataLoader,
        transform: Optional[Transform] = None,
        split: Optional[DataSplitter] = None,
        feature_store: Optional[FeatureStore] = None,
        schema: Optional[DataSchema] = None,
        name: str = "pipeline",
    ):
        self.source = source
        self.transform = transform or IdentityTransform()
        self.splitter = split or RandomSplitter()
        self.feature_store = feature_store or FeatureStore(name=name)
        self.schema = schema
        self.name = name

        # Pipeline state
        self._raw_data: List[Dict[str, Any]] = []
        self._transformed_data: List[Dict[str, Any]] = []
        self._train: List[Dict[str, Any]] = []
        self._val: List[Dict[str, Any]] = []
        self._test: List[Dict[str, Any]] = []

    def run(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run the full pipeline. Returns (train, val, test)."""
        # 1. Load
        self._raw_data = self.source.load()

        # 2. Validate (if schema provided)
        if self.schema:
            valid = []
            for row in self._raw_data:
                errors = self.schema.validate_row(row)
                if not errors:
                    valid.append(row)
            self._raw_data = valid

        # 3. Transform
        self._transformed_data = self.transform.apply(self._raw_data)

        # 4. Compute features
        self.feature_store.compute(self._transformed_data)

        # 5. Split
        self._train, self._val, self._test = self.splitter.split(self._transformed_data)

        return self._train, self._val, self._test

    @property
    def raw_data(self) -> List[Dict[str, Any]]:
        return self._raw_data

    @property
    def transformed_data(self) -> List[Dict[str, Any]]:
        return self._transformed_data

    @property
    def features(self) -> FeatureStore:
        return self.feature_store

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "raw_rows": len(self._raw_data),
            "transformed_rows": len(self._transformed_data),
            "train_size": len(self._train),
            "val_size": len(self._val),
            "test_size": len(self._test),
            "features": self.feature_store.feature_names,
            "source_type": type(self.source).__name__,
            "transform_type": type(self.transform).__name__,
            "splitter_type": type(self.splitter).__name__,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(source={self.source!r}, transform={self.transform!r}, split={self.split!r}, feature_store={self.feature_store!r}, schema={self.schema!r})"

