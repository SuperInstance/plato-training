"""
plato-train CLI — PLATO Training Rooms command-line interface.

Commands:
  train     Load model + data, run LoRA fine-tuning, emit tile info
  list      List tiles in a room (with optional type/state filters)
  info      Show full details of a single tile
  throttle  Report current fleet load and recommended throttle state
  serve     (future) Start HTTP API server
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .types import (
    AdapterConfig,
    TrainingConfig,
    TileLifecycle,
    TileType,
)
from .store import LocalTileStore
from .throttle import TrainingThrottle, ThrottleLevel
from .pytorch_room import PyTorchRoom


# ---------------------------------------------------------------------------
# Built-in model registry
# ---------------------------------------------------------------------------

class _SimpleClassifier(nn.Module):
    """Lightweight linear classifier used as the default built-in model."""

    def __init__(self, in_features: int = 128, hidden: int = 256, num_classes: int = 2):
        super().__init__()
        self.W_query = nn.Linear(in_features, hidden)
        self.W_value = nn.Linear(hidden, hidden)
        self.out_head = nn.Linear(hidden, num_classes)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.W_query(x))
        x = self.act(self.W_value(x))
        return self.out_head(x)


_BUILTIN_MODELS = {
    "simple-classifier": _SimpleClassifier,
}


def _load_model(model_name: str) -> nn.Module:
    """Return an nn.Module for the given name or file path."""
    if model_name in _BUILTIN_MODELS:
        return _BUILTIN_MODELS[model_name]()

    path = Path(model_name)
    if path.exists():
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(obj, nn.Module):
            return obj
        raise ValueError(
            f"File '{path}' did not contain an nn.Module (got {type(obj).__name__})"
        )

    raise ValueError(
        f"Unknown model '{model_name}'. "
        f"Provide a path to a saved nn.Module or one of: {list(_BUILTIN_MODELS)}"
    )


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

class _TensorDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _load_csv(path: str) -> _TensorDataset:
    """Load a CSV file. All columns except the last are features; last is label."""
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)  # skip header if present
        # If every value in the header row is numeric, treat it as data
        try:
            first_row = [float(v) for v in header]
            rows.append(first_row)
        except (TypeError, ValueError):
            pass  # it was a real header
        for row in reader:
            try:
                rows.append([float(v) for v in row])
            except ValueError:
                continue  # skip malformed rows

    if not rows:
        raise ValueError(f"No numeric data found in '{path}'")

    data = torch.tensor(rows, dtype=torch.float32)
    X, y = data[:, :-1], data[:, -1].long()
    return _TensorDataset(X, y)


def _load_jsonl(path: str) -> _TensorDataset:
    """
    Load a JSONL file.  Each line must be a JSON object with keys:
      "features": list[float]  and  "label": int
    """
    X_list, y_list = [], []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno} of '{path}': {exc}") from exc
            if "features" not in obj or "label" not in obj:
                raise ValueError(
                    f"Line {lineno} in '{path}' missing 'features' or 'label' key"
                )
            X_list.append(obj["features"])
            y_list.append(obj["label"])

    if not X_list:
        raise ValueError(f"No records found in '{path}'")

    X = torch.tensor(X_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.long)
    return _TensorDataset(X, y)


def _load_dataset(data_path: str) -> _TensorDataset:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: '{data_path}'")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(data_path)
    if suffix in (".jsonl", ".ndjson"):
        return _load_jsonl(data_path)
    # Fallback: try CSV then JSONL
    try:
        return _load_csv(data_path)
    except Exception:
        return _load_jsonl(data_path)


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> int:
    print(f"[plato-train] room={args.room} model={args.model} epochs={args.epochs} rank={args.rank}")

    # Load data first so we can derive in_features for simple-classifier
    dataset: Optional[_TensorDataset] = None
    num_classes: Optional[int] = None
    if args.data:
        print(f"[plato-train] loading data from '{args.data}' ...")
        dataset = _load_dataset(args.data)
        num_classes = int(dataset.y.max().item()) + 1
        print(f"[plato-train] {len(dataset)} samples, {dataset.X.shape[1]} features, {num_classes} classes")

    # Build model — patch simple-classifier input size to match data
    model = _load_model(args.model)
    if isinstance(model, _SimpleClassifier) and dataset is not None:
        in_features = dataset.X.shape[1]
        model = _SimpleClassifier(
            in_features=in_features,
            num_classes=num_classes or 2,
        )

    if dataset is None:
        print("[plato-train] WARNING: no --data provided; creating dummy single-sample dataset", file=sys.stderr)
        in_features = (
            model.W_query.in_features
            if hasattr(model, "W_query")
            else 128
        )
        X = torch.zeros(1, in_features)
        y = torch.zeros(1, dtype=torch.long)
        dataset = _TensorDataset(X, y)
        num_classes = 1

    adapter_cfg = AdapterConfig(rank=args.rank, alpha=args.alpha)
    training_cfg = TrainingConfig(
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    throttle = None if args.no_throttle else TrainingThrottle()

    room = PyTorchRoom(
        room_name=args.room,
        store_dir=args.store_dir,
        throttle=throttle,
    )

    print(f"[plato-train] starting training ...")
    tile = room.train(
        model=model,
        dataset=dataset,
        adapter_config=adapter_cfg,
        training_config=training_cfg,
        num_classes=num_classes,
    )

    print()
    print(f"  tile_id  : {tile.tile_id}")
    print(f"  state    : {tile.state.value}")
    print(f"  hash     : {tile.content_hash}")
    if tile.metrics:
        print(f"  loss     : {tile.metrics.final_loss:.6f}")
        print(f"  time     : {tile.metrics.training_time_seconds:.1f}s")
        if tile.metrics.peak_memory_mb:
            print(f"  peak_mem : {tile.metrics.peak_memory_mb:.1f} MB")
    print(f"  store    : {args.store_dir}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    store = LocalTileStore(args.store_dir)

    tile_type: Optional[TileType] = None
    if args.type:
        try:
            tile_type = TileType(args.type.lower())
        except ValueError:
            valid = [t.value for t in TileType]
            print(f"error: unknown tile type '{args.type}'. Valid: {valid}", file=sys.stderr)
            return 1

    state_filter: Optional[TileLifecycle] = None
    if args.state:
        try:
            state_filter = TileLifecycle(args.state.lower())
        except ValueError:
            valid = [s.value for s in TileLifecycle]
            print(f"error: unknown state '{args.state}'. Valid: {valid}", file=sys.stderr)
            return 1

    tiles = store.list_tiles(room=args.room, tile_type=tile_type, state=state_filter)

    if not tiles:
        print(f"No tiles found in room '{args.room}'.")
        return 0

    col_id    = max(len(t.tile_id) for t in tiles)
    col_type  = max(len(t.tile_type.value) for t in tiles)
    col_state = max(len(t.state.value) for t in tiles)

    header = f"{'TILE ID':<{col_id}}  {'TYPE':<{col_type}}  {'STATE':<{col_state}}  {'LOSS':>10}  DESCRIPTION"
    print(header)
    print("-" * len(header))
    for tile in tiles:
        loss_str = f"{tile.metrics.final_loss:.4f}" if tile.metrics else "     -"
        print(
            f"{tile.tile_id:<{col_id}}  "
            f"{tile.tile_type.value:<{col_type}}  "
            f"{tile.state.value:<{col_state}}  "
            f"{loss_str:>10}  "
            f"{tile.description}"
        )
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    store = LocalTileStore(args.store_dir)
    tile = store.load(args.tile)
    if tile is None:
        print(f"error: tile '{args.tile}' not found in store '{args.store_dir}'", file=sys.stderr)
        return 1

    def _fmt(label: str, value) -> None:
        print(f"  {label:<24} {value}")

    print(f"\nTile: {tile.tile_id}")
    print("=" * 60)
    _fmt("room", tile.room)
    _fmt("type", tile.tile_type.value)
    _fmt("state", tile.state.value)
    _fmt("lamport", tile.lamport)
    _fmt("name", tile.name)
    _fmt("description", tile.description)
    _fmt("content_hash", tile.content_hash)
    _fmt("base_model", tile.base_model or "-")
    _fmt("parent_tile", tile.parent_tile or "-")
    _fmt("timestamp", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(tile.timestamp)))

    if tile.adapter_config:
        ac = tile.adapter_config
        print("\n  Adapter Config:")
        _fmt("  rank", ac.rank)
        _fmt("  alpha", ac.alpha)
        _fmt("  target_modules", ", ".join(ac.target_modules))
        _fmt("  dropout", ac.dropout)

    if tile.training_config:
        tc = tile.training_config
        print("\n  Training Config:")
        _fmt("  epochs", tc.epochs)
        _fmt("  batch_size", tc.batch_size)
        _fmt("  learning_rate", tc.learning_rate)
        _fmt("  scheduler", tc.scheduler)

    if tile.metrics:
        m = tile.metrics
        print("\n  Metrics:")
        _fmt("  final_loss", f"{m.final_loss:.6f}")
        _fmt("  train_loss", f"{m.train_loss:.6f}")
        _fmt("  epochs_completed", m.epochs_completed)
        _fmt("  training_time", f"{m.training_time_seconds:.1f}s")
        if m.peak_memory_mb:
            _fmt("  peak_memory_mb", f"{m.peak_memory_mb:.1f}")
        if m.loss_curve:
            curve_preview = [f"{v:.4f}" for v in m.loss_curve[:5]]
            if len(m.loss_curve) > 5:
                curve_preview.append("...")
            _fmt("  loss_curve", "[" + ", ".join(curve_preview) + "]")

    if tile.lifecycle_events:
        print("\n  Lifecycle History:")
        for ev in tile.lifecycle_events:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ev.timestamp))
            print(f"    [{ts}] L{ev.lamport} {ev.from_state.value} -> {ev.to_state.value}: {ev.reason}")

    print()
    return 0


def cmd_throttle(_args: argparse.Namespace) -> int:
    throttle = TrainingThrottle()
    state = throttle.check()
    load = throttle.fleet_load()

    _BAR_WIDTH = 40
    filled = int(load * _BAR_WIDTH)
    bar = "#" * filled + "-" * (_BAR_WIDTH - filled)

    print(f"\n  Fleet Load   [{bar}] {load:.1%}")
    print(f"  Level        {state.level.value.upper()}")
    print(f"  Reason       {state.reason}")
    print(f"  Batch mult   {state.batch_multiplier:.2f}x")
    print(f"  Workers      {state.num_workers}")
    print(f"  GPU fraction {state.gpu_fraction:.0%}")
    print(f"  Check every  {state.check_interval_sec:.0f}s")

    if state.level == ThrottleLevel.PAUSED:
        print("\n  [!] Fleet saturated — training would be paused until load drops.")
    elif state.level == ThrottleLevel.MINIMAL:
        print("\n  [~] Fleet busy — training at minimal resource usage.")
    elif state.level == ThrottleLevel.REDUCED:
        print("\n  [~] Fleet light — training at reduced resource usage.")
    else:
        print("\n  [ok] Fleet idle — full training resources available.")

    print()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    print(f"[plato-train] HTTP API server not yet implemented (port={args.port}).", file=sys.stderr)
    print("  Use 'plato-train train' for now.", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plato-train",
        description="PLATO Training Rooms — LoRA fine-tuning CLI",
    )
    parser.add_argument("--version", action="version", version="plato-training 0.2.0")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── train ──────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Fine-tune a model with LoRA and save a tile")
    p_train.add_argument("--room",       required=True, help="Room / experiment name")
    p_train.add_argument("--model",      default="simple-classifier",
                         help="Model name (simple-classifier) or path to saved nn.Module")
    p_train.add_argument("--data",       default=None,
                         help="Path to CSV or JSONL training data file")
    p_train.add_argument("--epochs",     type=int,   default=3)
    p_train.add_argument("--rank",       type=int,   default=8,    help="LoRA rank")
    p_train.add_argument("--alpha",      type=int,   default=16,   help="LoRA alpha")
    p_train.add_argument("--batch-size", type=int,   default=8,    dest="batch_size")
    p_train.add_argument("--lr",         type=float, default=2e-4, help="Learning rate")
    p_train.add_argument("--store-dir",  default=".plato-training", dest="store_dir",
                         help="Directory for tile/weight storage")
    p_train.add_argument("--no-throttle", action="store_true", dest="no_throttle",
                         help="Disable fleet-aware throttle")
    p_train.set_defaults(func=cmd_train)

    # ── list ───────────────────────────────────────────────────────────────
    p_list = sub.add_parser("list", help="List tiles in a room")
    p_list.add_argument("--room",      required=True, help="Room name to list tiles for")
    p_list.add_argument("--type",      default=None,
                        help="Filter by tile type (adapter, checkpoint, dataset, …)")
    p_list.add_argument("--state",     default=None,
                        help="Filter by lifecycle state (active, superseded, retracted)")
    p_list.add_argument("--store-dir", default=".plato-training", dest="store_dir")
    p_list.set_defaults(func=cmd_list)

    # ── info ───────────────────────────────────────────────────────────────
    p_info = sub.add_parser("info", help="Show full details of a tile")
    p_info.add_argument("--tile",      required=True, help="Tile ID (e.g. spam-detector-001)")
    p_info.add_argument("--store-dir", default=".plato-training", dest="store_dir")
    p_info.set_defaults(func=cmd_info)

    # ── throttle ───────────────────────────────────────────────────────────
    p_throttle = sub.add_parser("throttle", help="Show current fleet load and throttle state")
    p_throttle.set_defaults(func=cmd_throttle)

    # ── serve ──────────────────────────────────────────────────────────────
    p_serve = sub.add_parser("serve", help="(future) Start HTTP API server")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.set_defaults(func=cmd_serve)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        rc = args.func(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        rc = 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
