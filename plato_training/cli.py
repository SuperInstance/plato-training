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
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .types import AdapterConfig, TrainingConfig, TileLifecycle, TileType
from .store import LocalTileStore
from .throttle import TrainingThrottle, ThrottleLevel
from .pytorch_room import PyTorchRoom


# ---------------------------------------------------------------------------
# Built-in models
# ---------------------------------------------------------------------------


class _SimpleClassifier(nn.Module):
    """
    Lightweight MLP classifier.

    W_query / W_value / out_head naming matches the default LoRA target_modules
    in AdapterConfig so LoRA layers inject without extra configuration.
    """

    def __init__(self, in_features: int = 128, hidden: int = 256, num_classes: int = 2):
        super().__init__()
        self.W_query = nn.Linear(in_features, hidden)
        self.W_value = nn.Linear(hidden, hidden)
        self.out_head = nn.Linear(hidden, num_classes)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.W_query(x))
        x = self.drop(self.act(self.W_value(x)))
        return self.out_head(x)


class _HFWrapper(nn.Module):
    """
    Wraps a HuggingFace SequenceClassification model for PyTorchRoom.

    PyTorchRoom passes batch[0] directly to model() and expects a logit tensor.
    HF models return a dataclass; this wrapper extracts .logits.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids).logits


_BUILTIN_MODELS = {"simple-classifier"}


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class _FloatDataset(Dataset):
    """Numeric or TF-IDF-vectorised dataset."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class _TokenDataset(Dataset):
    """Integer token-id dataset for HuggingFace text models."""

    def __init__(self, texts: List[str], y: List[int], tokenizer, max_length: int = 128):
        enc = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids: torch.Tensor = enc["input_ids"]
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_rows(data_path: str) -> Tuple[List[Dict], List[str]]:
    """Load CSV or JSONL file; return (rows, fieldnames)."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: '{data_path}'")

    if path.suffix.lower() in (".jsonl", ".ndjson"):
        rows = []
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                stripped = line.strip()
                if stripped:
                    try:
                        rows.append(json.loads(stripped))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON on line {lineno}: {exc}") from exc
        headers = list(rows[0].keys()) if rows else []
    else:  # CSV (default)
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            headers = list(reader.fieldnames or [])
            rows = [dict(r) for r in reader]

    if not rows:
        raise ValueError(f"No data found in '{data_path}'")
    return rows, headers


def _detect_label_col(headers: List[str]) -> str:
    """Choose the label column by common names; falls back to the last column."""
    synonyms = {"label", "target", "y", "class", "spam", "category", "sentiment", "output"}
    for h in reversed(headers):
        if h.lower() in synonyms:
            return h
    return headers[-1]


def _detect_text_col(headers: List[str], label_col: str) -> Optional[str]:
    """Choose the text/content column if present; returns None for numeric-only data."""
    synonyms = {
        "text", "message", "content", "sentence", "comment",
        "body", "tweet", "review", "description", "input", "sms",
    }
    for h in headers:
        if h != label_col and h.lower() in synonyms:
            return h
    return None


def _encode_labels(raw: List[str]) -> Tuple[List[int], Dict[str, int]]:
    """
    Map label strings to contiguous integer indices.

    Preserves numeric ordering for integer-valued labels;
    uses sorted string order otherwise.
    """
    unique = sorted(set(raw))
    try:
        int_vals = [int(v) for v in unique]
        remap = {old: new for new, old in enumerate(sorted(set(int_vals)))}
        mapping: Dict[str, int] = {v: remap[int(v)] for v in unique}
    except ValueError:
        mapping = {v: i for i, v in enumerate(unique)}
    return [mapping[v] for v in raw], mapping


def _tfidf_vectorize(texts: List[str], max_features: int = 512) -> torch.Tensor:
    """
    Produce L2-normalised TF-IDF vectors.

    Uses sklearn when available; falls back to a stdlib bag-of-words.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        mat = TfidfVectorizer(max_features=max_features, sublinear_tf=True).fit_transform(texts)
        return torch.tensor(mat.toarray(), dtype=torch.float32)
    except ImportError:
        pass

    # Stdlib fallback — top-K unigrams, L2-normalised counts
    tokenize = lambda s: re.findall(r"[a-z]+", s.lower())
    tokenized = [tokenize(t) for t in texts]
    freq: Counter = Counter()
    for toks in tokenized:
        freq.update(set(toks))
    vocab = {w: i for i, (w, _) in enumerate(freq.most_common(max_features))}
    dim = len(vocab)
    rows = []
    for toks in tokenized:
        v = [0.0] * dim
        for w in toks:
            if w in vocab:
                v[vocab[w]] += 1.0
        norm = (sum(x * x for x in v) ** 0.5) or 1.0
        rows.append([x / norm for x in v])
    return torch.tensor(rows, dtype=torch.float32)


def _detect_lora_targets(model: nn.Module) -> List[str]:
    """
    Infer LoRA injection targets from the model's Linear-layer leaf names.

    Checks common attention naming conventions; falls back to the
    _SimpleClassifier convention (W_query, W_value).
    """
    leaves = {name.split(".")[-1] for name, _ in model.named_modules()}
    if "c_attn" in leaves:          # GPT-2
        return ["c_attn", "c_proj"]
    if "query" in leaves:           # BERT / RoBERTa
        return ["query", "value"]
    if "q_proj" in leaves:          # LLaMA / Mistral / Falcon
        return ["q_proj", "v_proj"]
    return ["W_query", "W_value"]   # _SimpleClassifier default


# ---------------------------------------------------------------------------
# Model + dataset construction (coupled to share dimensionality)
# ---------------------------------------------------------------------------


def _build_model_and_dataset(
    rows: List[Dict],
    headers: List[str],
    model_name: str,
) -> Tuple[nn.Module, Dataset, int, Dict[str, int]]:
    """
    Build (model, dataset, num_classes, label_mapping) together.

    The three data paths are:
      1. HF model + text column  → _TokenDataset with tokenizer
      2. built-in/file + text col → TF-IDF _FloatDataset + _SimpleClassifier resized
      3. built-in/file + numeric  → numeric _FloatDataset + _SimpleClassifier resized
    """
    label_col = _detect_label_col(headers)
    text_col = _detect_text_col(headers, label_col)
    raw_labels = [str(r[label_col]) for r in rows]
    labels, label_map = _encode_labels(raw_labels)
    num_classes = len(label_map)

    is_hf = model_name not in _BUILTIN_MODELS and not Path(model_name).exists()

    tokenizer = None
    model: nn.Module

    if is_hf:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            raise ImportError(
                f"Cannot load '{model_name}': transformers is not installed.\n"
                "  pip install transformers"
            )
        try:
            print(f"[plato-train] loading HuggingFace model '{model_name}' ...")
            hf = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_classes, ignore_mismatched_sizes=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as exc:
            raise ValueError(f"Failed to load '{model_name}': {exc}") from exc
        model = _HFWrapper(hf)
    elif Path(model_name).exists():
        obj = torch.load(str(model_name), map_location="cpu", weights_only=False)
        if not isinstance(obj, nn.Module):
            raise ValueError(
                f"'{model_name}' does not contain an nn.Module (got {type(obj).__name__})"
            )
        model = obj
    else:
        model = None  # type: ignore[assignment]  # rebuilt after input_dim is known

    # ── Build dataset ──────────────────────────────────────────────────────
    dataset: Dataset
    if tokenizer is not None and text_col:
        texts = [str(r[text_col]) for r in rows]
        dataset = _TokenDataset(texts, labels, tokenizer)
    elif text_col:
        texts = [str(r[text_col]) for r in rows]
        X = _tfidf_vectorize(texts)
        y = torch.tensor(labels, dtype=torch.long)
        dataset = _FloatDataset(X, y)
    else:
        feature_cols = [h for h in headers if h not in {label_col, text_col}]
        if not feature_cols:
            raise ValueError("No usable feature columns found in data.")
        try:
            mat = [[float(r[c]) for c in feature_cols] for r in rows]
        except (ValueError, KeyError) as exc:
            raise ValueError(f"Non-numeric feature value: {exc}") from exc
        X = torch.tensor(mat, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        dataset = _FloatDataset(X, y)

    # ── Build / resize _SimpleClassifier ──────────────────────────────────
    if model is None or (model_name == "simple-classifier" and not is_hf):
        if isinstance(dataset, _FloatDataset):
            in_features = dataset.X.shape[1]
        else:
            # _TokenDataset — shouldn't reach here for simple-classifier
            in_features = 512
        model = _SimpleClassifier(in_features=in_features, num_classes=num_classes)

    return model, dataset, num_classes, label_map


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    print(f"[plato-train] room={args.room}  model={args.model}  epochs={args.epochs}  rank={args.rank}")

    print(f"[plato-train] loading data from '{args.data}' ...")
    rows, headers = _load_rows(args.data)

    try:
        model, dataset, num_classes, label_map = _build_model_and_dataset(
            rows, headers, args.model
        )
    except (ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    n_samples = len(dataset)
    print(f"[plato-train] {n_samples:,} samples  |  {num_classes} classes: {label_map}")

    lora_targets = _detect_lora_targets(model)
    adapter_cfg = AdapterConfig(rank=args.rank, alpha=args.alpha, target_modules=lora_targets)
    training_cfg = TrainingConfig(
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Passing custom_load_fn=lambda: 0.0 keeps the throttle machinery intact
    # but reports zero fleet load → always FULL level → no batch reduction.
    throttle = TrainingThrottle(custom_load_fn=lambda: 0.0) if args.no_throttle else None

    room = PyTorchRoom(room_name=args.room, store_dir=args.store_dir, throttle=throttle)

    print(
        f"[plato-train] LoRA targets={lora_targets}  "
        f"batch={args.batch_size}  lr={args.lr}  store={args.store_dir}"
    )
    print("[plato-train] starting training ...")

    try:
        tile = room.train(
            model=model,
            dataset=dataset,
            adapter_config=adapter_cfg,
            training_config=training_cfg,
            num_classes=num_classes,
        )
    except Exception as exc:
        print(f"error during training: {exc}", file=sys.stderr)
        raise

    w = 52
    print(f"\n{'─' * w}")
    print(f"  tile_id    {tile.tile_id}")
    print(f"  state      {tile.state.value}")
    print(f"  type       {tile.tile_type.value}")
    print(f"  hash       {tile.content_hash}")
    if tile.metrics:
        m = tile.metrics
        print(f"  loss       {m.final_loss:.6f}  (after {m.epochs_completed} epoch(s))")
        print(f"  time       {m.training_time_seconds:.1f}s")
        if m.peak_memory_mb:
            print(f"  peak_vram  {m.peak_memory_mb:.1f} MB")
    print(f"  store      {args.store_dir}")
    print(f"{'─' * w}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    store = LocalTileStore(args.store_dir)

    tile_type: Optional[TileType] = None
    if args.type:
        try:
            tile_type = TileType(args.type.lower())
        except ValueError:
            valid = [t.value for t in TileType]
            print(f"error: unknown tile type '{args.type}'. Valid: {', '.join(valid)}", file=sys.stderr)
            return 1

    state_filter: Optional[TileLifecycle] = None
    if args.state:
        try:
            state_filter = TileLifecycle(args.state.lower())
        except ValueError:
            valid = [s.value for s in TileLifecycle]
            print(f"error: unknown state '{args.state}'. Valid: {', '.join(valid)}", file=sys.stderr)
            return 1

    tiles = store.list_tiles(room=args.room, tile_type=tile_type, state=state_filter)

    if not tiles:
        print(f"No tiles found in room '{args.room}'.")
        return 0

    col_id    = max(len(t.tile_id) for t in tiles)
    col_type  = max(len(t.tile_type.value) for t in tiles)
    col_state = max(len(t.state.value) for t in tiles)

    header = (
        f"{'TILE ID':<{col_id}}  {'TYPE':<{col_type}}  "
        f"{'STATE':<{col_state}}  {'LOSS':>10}  DESCRIPTION"
    )
    print(header)
    print("─" * len(header))
    for tile in tiles:
        loss_str = f"{tile.metrics.final_loss:.4f}" if tile.metrics else "         -"
        desc = tile.description[:60] + ("…" if len(tile.description) > 60 else "")
        print(
            f"{tile.tile_id:<{col_id}}  "
            f"{tile.tile_type.value:<{col_type}}  "
            f"{tile.state.value:<{col_state}}  "
            f"{loss_str:>10}  "
            f"{desc}"
        )
    print(f"\n{len(tiles)} tile(s)")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    store = LocalTileStore(args.store_dir)
    tile = store.load(args.tile)
    if tile is None:
        print(f"error: tile '{args.tile}' not found in store '{args.store_dir}'", file=sys.stderr)
        return 1

    def _row(label: str, value) -> None:
        print(f"  {label:<24} {value}")

    print(f"\nTile: {tile.tile_id}")
    print("=" * 60)
    _row("room", tile.room)
    _row("type", tile.tile_type.value)
    _row("state", tile.state.value)
    _row("lamport", tile.lamport)
    _row("name", tile.name)
    _row("description", tile.description)
    _row("content_hash", tile.content_hash or "—")
    _row("base_model", tile.base_model or "—")
    _row("parent_tile", tile.parent_tile or "—")
    _row("timestamp", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(tile.timestamp)))

    if tile.adapter_config:
        ac = tile.adapter_config
        print("\n  Adapter Config:")
        _row("  rank", ac.rank)
        _row("  alpha", ac.alpha)
        _row("  target_modules", ", ".join(ac.target_modules))
        _row("  dropout", ac.dropout)

    if tile.training_config:
        tc = tile.training_config
        print("\n  Training Config:")
        _row("  epochs", tc.epochs)
        _row("  batch_size", tc.batch_size)
        _row("  learning_rate", tc.learning_rate)
        _row("  scheduler", tc.scheduler)
        _row("  warmup_steps", tc.warmup_steps)

    if tile.metrics:
        m = tile.metrics
        print("\n  Metrics:")
        _row("  final_loss", f"{m.final_loss:.6f}")
        _row("  train_loss", f"{m.train_loss:.6f}")
        if m.val_loss:
            _row("  val_loss", f"{m.val_loss:.6f}")
        _row("  epochs_completed", m.epochs_completed)
        _row("  training_time", f"{m.training_time_seconds:.1f}s")
        if m.peak_memory_mb:
            _row("  peak_memory_mb", f"{m.peak_memory_mb:.1f}")
        if m.loss_curve:
            preview = [f"{v:.4f}" for v in m.loss_curve[:8]]
            tail = "…" if len(m.loss_curve) > 8 else ""
            _row("  loss_curve", f"[{', '.join(preview)}{tail}]")

    if tile.lifecycle_events:
        print("\n  Lifecycle History:")
        for ev in tile.lifecycle_events:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ev.timestamp))
            print(f"    [{ts}]  L{ev.lamport}  {ev.from_state.value} → {ev.to_state.value}  {ev.reason}")

    print()
    return 0


def cmd_throttle(_args: argparse.Namespace) -> int:
    throttle = TrainingThrottle()
    load = throttle.fleet_load()
    state = throttle.check()

    bar_width = 40
    filled = int(load * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)

    level_note = {
        ThrottleLevel.FULL:    "  [ok] Fleet idle — full resources available.",
        ThrottleLevel.REDUCED: "  [~]  Fleet light — training at reduced usage.",
        ThrottleLevel.MINIMAL: "  [~]  Fleet busy — training at minimal usage.",
        ThrottleLevel.PAUSED:  "  [!]  Fleet saturated — training would pause until load drops.",
    }

    print(f"\n  Fleet Load   [{bar}] {load:.1%}")
    print(f"  Level        {state.level.value.upper()}")
    print(f"  Reason       {state.reason}")
    print(f"  Batch mult   {state.batch_multiplier:.2f}x")
    print(f"  Workers      {state.num_workers}")
    print(f"  GPU fraction {state.gpu_fraction:.0%}")
    print(f"  Check every  {state.check_interval_sec:.0f}s")
    print()
    print(level_note.get(state.level, ""))
    print()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    print(
        f"[plato-train] HTTP API server not yet implemented (port={args.port}).\n"
        f"  Use 'plato-train train' to train and 'plato-train list' to query tiles.",
        file=sys.stderr,
    )
    return 1


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plato-train",
        description="PLATO Training Rooms — LoRA fine-tuning CLI",
        epilog=(
            "examples:\n"
            "  plato-train train --room spam-detector --model gpt2 --data spam.csv --epochs 3 --rank 8\n"
            "  plato-train train --room clf --data features.csv --no-throttle\n"
            "  plato-train list  --room spam-detector --state active\n"
            "  plato-train info  --tile spam-detector-001\n"
            "  plato-train throttle\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version="plato-training 0.2.0")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── train ──────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Fine-tune a model with LoRA and save a tile")
    p_train.add_argument("--room", required=True, help="Room / experiment name")
    p_train.add_argument(
        "--model", default="simple-classifier",
        help="'simple-classifier', a HuggingFace model name, or a local .pt path",
    )
    p_train.add_argument("--data", required=True, metavar="FILE",
                         help="CSV or JSONL training data")
    p_train.add_argument("--epochs",     type=int,   default=3)
    p_train.add_argument("--rank",       type=int,   default=8,    help="LoRA rank")
    p_train.add_argument("--alpha",      type=int,   default=16,   help="LoRA alpha")
    p_train.add_argument("--batch-size", type=int,   default=8,    dest="batch_size")
    p_train.add_argument("--lr",         type=float, default=2e-4, help="Learning rate")
    p_train.add_argument("--store-dir",  default=".plato-training", dest="store_dir",
                         metavar="DIR",  help="Tile and weight storage directory")
    p_train.add_argument(
        "--no-throttle", action="store_true", dest="no_throttle",
        help="Disable fleet-aware throttle (always train at full batch size)",
    )
    p_train.set_defaults(func=cmd_train)

    # ── list ───────────────────────────────────────────────────────────────
    p_list = sub.add_parser("list", help="List tiles in a room")
    p_list.add_argument("--room",      required=True)
    p_list.add_argument("--type",      default=None,
                        help="Filter by tile type (adapter | checkpoint | dataset | …)")
    p_list.add_argument("--state",     default=None,
                        help="Filter by state (active | superseded | retracted)")
    p_list.add_argument("--store-dir", default=".plato-training", dest="store_dir", metavar="DIR")
    p_list.set_defaults(func=cmd_list)

    # ── info ───────────────────────────────────────────────────────────────
    p_info = sub.add_parser("info", help="Show full details of a tile")
    p_info.add_argument("--tile",      required=True, help="Tile ID (e.g. spam-detector-001)")
    p_info.add_argument("--store-dir", default=".plato-training", dest="store_dir", metavar="DIR")
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
