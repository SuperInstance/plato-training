"""
GPT-2 Room — A PLATO room that trains a minimal GPT-2 on tile text data.

Runs on RTX 4050 (6.4GB VRAM) comfortably with a ~2M param model.
Trains on text extracted from PLATO tiles. Saves checkpoints as tiles.
"""

from __future__ import annotations
import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, List, Tuple, Dict, Iterator
from pathlib import Path

from .types import (TrainingTile, TileType, TileLifecycle, LamportClock,
    TrainingConfig, AdapterConfig, TrainingMetrics, content_hash)
from .store import LocalTileStore
from .throttle import TrainingThrottle


# ─── Configuration ─────────────────────────────────────────────────

@dataclass
class GPT2Config:
    """Minimal GPT-2 configuration that fits in 6.4GB VRAM."""
    vocab_size: int = 256            # Character-level tokenizer
    n_layer: int = 4                 # Transformer blocks
    n_head: int = 4                  # Attention heads
    n_embd: int = 256                # Embedding dimension
    block_size: int = 128            # Max sequence length
    dropout: float = 0.1             # Dropout rate
    bias: bool = True                # Use bias in linear layers
    tie_weights: bool = False        # Tie input/output embeddings (not needed at this scale)
    pad_token_id: int = 0

    def param_count(self) -> int:
        """Estimate parameter count (approximate)."""
        n = self.vocab_size * self.n_embd  # token embeddings
        n += self.block_size * self.n_embd  # position embeddings
        # Per transformer block
        block_params = (
            4 * self.n_embd * self.n_embd  # QKV projections (3 * n_embd * n_embd)
            + self.n_embd * self.n_embd    # Output projection
            + 2 * self.n_embd               # LayerNorms (2 * n_embd)
            + 8 * self.n_embd * self.n_embd  # MLP: 4*n_embd -> n_embd (factor 4)
            + 2 * self.n_embd               # MLP LayerNorms
        )
        n += block_params * self.n_layer
        n += 2 * self.n_embd  # Final LayerNorm
        n += self.n_embd * self.vocab_size  # LM head
        return n


# ─── Casual Self-Attention ─────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with fused QKV."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Fused QKV projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask (constant, not a buffer)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, seq_len, n_embd

        # Fused QKV -> split
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, hs)

        # Re-assemble all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


# ─── MLP (GELU) ───────────────────────────────────────────────────

class NewGELU(nn.Module):
    """GELU activation used in GPT-2 (approximation)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
        ))


class MLP(nn.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.gelu = NewGELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# ─── Transformer Block ────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """GPT-2 transformer block: layer norm -> attn -> resid -> layer norm -> MLP -> resid."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ─── GPT-2 Model ──────────────────────────────────────────────────

class GPT2Model(nn.Module):
    """Minimal GPT-2 implementation with causal language modeling head."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        # Final layer norm and LM head
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie weights (not default for this config size)
        if config.tie_weights:
            self.lm_head.weight = self.wte.weight

        # Init weights
        self.apply(self._init_weights)

        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT2Model: {n_params:,} parameters")

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following GPT-2 pattern."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len) token indices
            labels: Optional (batch, seq_len) for computing language modeling loss

        Returns:
            Dict with 'logits', and optionally 'loss'
        """
        device = input_ids.device
        b, t = input_ids.shape
        assert t <= self.config.block_size, (
            f"Cannot forward of length {t}, block size is {self.config.block_size}"
        )

        # Token + position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)
        tok_emb = self.wte(input_ids)  # (b, t, n_embd)
        pos_emb = self.wpe(pos)        # (1, t, n_embd)
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Language modeling head
        logits = self.lm_head(x)  # (b, t, vocab_size)

        loss = None
        if labels is not None:
            # Flatten for cross-entropy
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-1)

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: (1, seq_len) prompt token indices
            max_new_tokens: How many tokens to generate
            temperature: Sampling temperature (>0)
            top_k: If set, only sample from top-k logits
            do_sample: If False, greedy argmax

        Returns:
            (1, seq_len + max_new_tokens) completed sequence
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop if long enough — we only need block_size tokens for a full context
            if input_ids.size(1) >= self.config.block_size:
                input_ids = input_ids[:, -self.config.block_size:]

            logits = self.forward(input_ids)["logits"]
            logits = logits[:, -1, :]  # (1, vocab_size)

            # Greedy mode (do_sample=False or temperature=0)
            if not do_sample or temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                continue

            # Temperature scaling for sampling
            logits = logits / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        self.train()
        return input_ids


# ─── Text Tokenizer ────────────────────────────────────────────────

class CharTokenizer:
    """Simple character-level tokenizer for GPT-2 training."""

    def __init__(self, chars: Optional[str] = None):
        if chars is None:
            # Default printable ASCII + common tokens
            chars = (
                "\n !\"#$%&'()*+,-./0123456789:;<=>?@"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
                "abcdefghijklmnopqrstuvwxyz{|}~\t"
            )
        self.chars = sorted(set(chars))
        self.vocab_size = len(self.chars) + 1  # +1 for unknown/pad
        self.stoi = {ch: i + 1 for i, ch in enumerate(self.chars)}  # 0 = pad/unk
        self.itos = {i + 1: ch for i, ch in enumerate(self.chars)}
        self.itos[0] = '<PAD>'
        self.pad_token_id = 0
        self.unk_token_id = 0

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(ch, self.unk_token_id) for ch in text]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.itos.get(i, '?') for i in ids if i != self.pad_token_id)

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump({"chars": ''.join(self.chars)}, f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path) as f:
            data = json.load(f)
        return cls(chars=data["chars"])


class BpeTokenizer:
    """Simple BPE tokenizer wrapper (uses tiktoken if available)."""

    def __init__(self, model_name: str = "gpt2"):
        self._available = False
        self.encoder = None
        try:
            import tiktoken
            self.encoder = tiktoken.get_encoding(model_name)
            self._available = True
        except ImportError:
            pass

    @property
    def vocab_size(self) -> int:
        if self._available:
            return self.encoder.n_vocab
        return 0

    @property
    def is_available(self) -> bool:
        return self._available

    def encode(self, text: str) -> List[int]:
        if self._available:
            return self.encoder.encode(text)
        raise RuntimeError("tiktoken not installed")

    def decode(self, ids: List[int]) -> str:
        if self._available:
            return self.encoder.decode(ids)
        raise RuntimeError("tiktoken not installed")


def build_tokenizer(chars: Optional[str] = None) -> CharTokenizer:
    """Build the tokenizer, preferring tiktoken if available."""
    try:
        import tiktoken
        return BpeTokenizer("gpt2")
    except ImportError:
        return CharTokenizer(chars)


# ─── Text Dataset from Tiles ──────────────────────────────────────

class TileTextDataset(Dataset):
    """
    Dataset that reads text from PLATO tiles and creates
    (input, target) pairs for causal language modeling.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: CharTokenizer,
        block_size: int = 128,
        stride: Optional[int] = None,
    ):
        """
        Args:
            texts: List of text strings (from tile contents)
            tokenizer: Tokenizer instance
            block_size: Maximum sequence length
            stride: Overlap stride (default: block_size // 2)
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride or (block_size // 2)

        # Tokenize all texts
        self.tokens = []
        for text in texts:
            self.tokens.extend(tokenizer.encode(text))

        # Build indices for sliding window
        self.indices = []
        if len(self.tokens) < block_size + 1:
            # Pad short text
            self.tokens = self.tokens + [tokenizer.pad_token_id] * (
                block_size + 1 - len(self.tokens)
            )

        for i in range(0, len(self.tokens) - block_size, self.stride):
            self.indices.append(i)

    def __len__(self) -> int:
        return max(1, len(self.indices))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        i = self.indices[idx]
        x = torch.tensor(self.tokens[i:i + self.block_size], dtype=torch.long)
        y = torch.tensor(self.tokens[i + 1:i + self.block_size + 1], dtype=torch.long)
        return x, y


# ─── Training Tracker ──────────────────────────────────────────────

class TrainingTracker:
    """Tracks loss, perplexity, and VRAM usage during GPT-2 training."""

    def __init__(self, store: Optional[LocalTileStore] = None, room_name: str = "gpt2"):
        self.store = store
        self.room_name = room_name
        self.clock = LamportClock()
        self.epoch_losses: List[float] = []
        self.epoch_perplexities: List[float] = []
        self.epoch_ppl: List[float] = []
        self.peak_vram_mb: float = 0.0
        self.start_time: Optional[float] = None
        self.epochs_completed: int = 0
        self.loss_curve: List[float] = []

    def start(self) -> None:
        self.start_time = time.time()

    def log_epoch(self, epoch: int, loss: float, lr: float, grad_norm: float) -> None:
        """Log one epoch of training."""
        ppl = math.exp(min(loss, 20.0))  # Cap to avoid overflow
        self.epoch_losses.append(loss)
        self.epoch_perplexities.append(ppl)
        self.epoch_ppl.append(ppl)
        self.epochs_completed = epoch + 1

        if torch.cuda.is_available():
            self.peak_vram_mb = max(
                self.peak_vram_mb,
                torch.cuda.max_memory_allocated() / 1e6
            )

        print(
            f"  Epoch {epoch+1}: loss={loss:.4f} ppl={ppl:.2f} "
            f"lr={lr:.2e} grad_norm={grad_norm:.4f} "
            f"vram={torch.cuda.memory_allocated()/1e6:.0f}MB" if torch.cuda.is_available()
            else f"  Epoch {epoch+1}: loss={loss:.4f} ppl={ppl:.2f} lr={lr:.2e}"
        )

    def log_step(self, loss_val: float) -> None:
        """Log a single training step."""
        self.loss_curve.append(loss_val)

    def get_metrics(self) -> TrainingMetrics:
        """Build a TrainingMetrics from tracked data."""
        final_loss = self.epoch_losses[-1] if self.epoch_losses else 0.0
        elapsed = time.time() - (self.start_time or time.time())
        return TrainingMetrics(
            train_loss=final_loss,
            epochs_completed=self.epochs_completed,
            training_time_seconds=elapsed,
            peak_memory_mb=self.peak_vram_mb,
            final_loss=final_loss,
            loss_curve=self.loss_curve,
        )

    def create_tile(
        self,
        tile_id: str,
        content_hash: str,
        config: GPT2Config,
        training_config: TrainingConfig,
    ) -> TrainingTile:
        """Create a training tile with tracked metrics."""
        return TrainingTile(
            tile_id=tile_id,
            room=self.room_name,
            tile_type=TileType.CHECKPOINT,
            state=TileLifecycle.ACTIVE,
            lamport=self.clock.tick(),
            name=f"gpt2-epoch-{self.epochs_completed}",
            description=f"GPT-2 checkpoint: {config.n_layer}L {config.n_head}H {config.n_embd}D",
            content_hash=content_hash,
            training_config=training_config,
            metrics=self.get_metrics(),
            source_room=self.room_name,
        )


# ─── GPT-2 Room ────────────────────────────────────────────────────

class GPT2Room:
    """
    A PLATO room that trains GPT-2 on tile text data.

    α dial: at low α, serves pre-trained responses; at high α, fine-tunes on recent tiles.
    """

    def __init__(
        self,
        room_name: str = "gpt2",
        store_dir: str = ".plato-training",
        device: Optional[str] = None,
        throttle: Optional[TrainingThrottle] = None,
        alpha: float = 0.5,
        config: Optional[GPT2Config] = None,
    ):
        self.room_name = room_name
        self.clock = LamportClock()
        self.store = LocalTileStore(store_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.throttle = throttle or TrainingThrottle()
        self.alpha = alpha
        self.config = config or GPT2Config()
        self.model: Optional[GPT2Model] = None
        self.tokenizer: Optional[CharTokenizer] = self._build_tokenizer()
        # Ensure tokenizer fits the model's vocabulary
        self.tokenizer = self._fit_tokenizer_to_config()
        self.tracker = TrainingTracker(store=self.store, room_name=room_name)

    def _build_tokenizer(self) -> CharTokenizer:
        """Build tokenizer, trying BPE first then falling back to character-level."""
        try:
            import tiktoken
            return BpeTokenizer("gpt2")
        except ImportError:
            return CharTokenizer()

    def _fit_tokenizer_to_config(self) -> CharTokenizer:
        """Ensure tokenizer vocab fits in model's vocab_size."""
        if hasattr(self.tokenizer, 'vocab_size') and self.tokenizer.vocab_size <= self.config.vocab_size:
            return self.tokenizer
        # Fall back to char tokenizer with vocab_size-limited chars
        # Reserve 1 slot for pad/unk
        max_chars = self.config.vocab_size - 1
        # Use a subset of common ASCII chars
        common_chars = (
            " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789.,!?;:'\"()-[]{}@#$%^&*+=/\\|~`<>\n\t"
        )
        chars = common_chars[:max_chars]
        return CharTokenizer(chars=chars)

    def load_model(self) -> GPT2Model:
        """Load or create the GPT-2 model."""
        if self.model is None:
            self.model = GPT2Model(self.config)
            self.model.to(self.device)
        return self.model

    def _load_tile_texts(self) -> List[str]:
        """Load text content from PLATO tiles in this room."""
        texts = []
        tiles = self.store.list_tiles(room=self.room_name)
        for tile in tiles:
            try:
                # Try to locate tile content
                path = self.store.tiles_dir / f"{tile.tile_id}.json"
                if path.exists():
                    data = json.loads(path.read_text())
                    # Extract text from tile (handle multiple formats)
                    if isinstance(data, dict):
                        for key in ["text", "content", "description", "prompt", "response"]:
                            if key in data and isinstance(data[key], str):
                                texts.append(data[key])
                                break
            except Exception:
                continue
        return texts

    def _texts_to_dataset(self, texts: List[str]) -> TileTextDataset:
        """Convert text list to a TileTextDataset."""
        return TileTextDataset(
            texts=texts or [" "],
            tokenizer=self.tokenizer,
            block_size=self.config.block_size,
        )

    def train(
        self,
        texts: Optional[List[str]] = None,
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 2,
        warmup_steps: int = 50,
        max_grad_norm: float = 1.0,
        eval_split: float = 0.1,
        log_interval: int = 10,
    ) -> TrainingTile:
        """
        Train GPT-2 on text data.

        Args:
            texts: List of text strings (if None, loads from room tiles)
            epochs: Number of training epochs
            batch_size: Per-GPU batch size
            learning_rate: Peak learning rate
            weight_decay: AdamW weight decay
            gradient_accumulation_steps: Grad accumulation for effective larger batch
            warmup_steps: Learning rate warmup steps
            max_grad_norm: Gradient clipping
            eval_split: Fraction of data for validation
            log_interval: Log every N steps

        Returns:
            TrainingTile with checkpoint info
        """
        # Load model
        model = self.load_model()
        self.tracker.start()

        # Load text data
        if texts is None:
            texts = self._load_tile_texts()
            if not texts:
                texts = ["The quick brown fox jumps over the lazy dog. " * 20]

        print(f"GPT2Room.train: {len(texts)} texts loaded ({sum(len(t) for t in texts)} chars)")

        # Create dataset
        dataset = self._texts_to_dataset(texts)

        # Split train/val
        val_size = max(1, int(len(dataset) * eval_split))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        print(f"  Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

        # Training loop config
        train_config = TrainingConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            warmup_steps=warmup_steps,
            gradient_accumulation=gradient_accumulation_steps,
            scheduler="cosine",
        )

        # DataLoaders
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=(self.device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=(self.device.type == "cuda"),
        )

        # Optimizer with weight decay groups
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'ln_' in name or 'norm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0},
            ],
            lr=learning_rate,
            betas=(0.9, 0.95),
        )

        # Scheduler: linear warmup + cosine decay
        total_steps = len(train_loader) * epochs
        warmup_it = min(warmup_steps, total_steps // 10)

        def get_lr(it: int) -> float:
            if it < warmup_it:
                return learning_rate * (it + 1) / (warmup_it + 1)
            progress = (it - warmup_it) / max(total_steps - warmup_it, 1)
            return learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

        # Training loop
        global_step = 0
        best_val_loss = float('inf')
        model.train()

        for epoch in range(epochs):
            state = self.throttle.check()
            if not state.should_train:
                print(f"[throttle] {state.level.value} — waiting...")
                self.throttle.wait_for_idle()
                state = self.throttle.check()

            epoch_loss = 0.0
            optimizer.zero_grad()

            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)

                # Forward
                logits = model(x)["logits"]

                # Compute loss (shift: predict next token)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=-1,
                )

                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                loss.backward()

                epoch_loss += loss.item() * gradient_accumulation_steps

                # Step optimizer every gradient_accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    self.tracker.log_step(loss.item() * gradient_accumulation_steps)

                    if global_step % log_interval == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        print(
                            f"  Step {global_step}/{total_steps}: "
                            f"loss={loss.item() * gradient_accumulation_steps:.4f} "
                            f"lr={current_lr:.2e}"
                        )

            # End of epoch: handle any remaining gradient
            if (batch_idx + 1) % gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # Validation
            val_loss = self._validate(model, val_loader)
            avg_train_loss = epoch_loss / len(train_loader)

            # Log epoch
            current_lr = scheduler.get_last_lr()[0]
            grad_norm = sum(
                p.grad.norm().item() for p in model.parameters()
                if p.grad is not None
            ) / max(sum(1 for p in model.parameters() if p.grad is not None), 1)

            self.tracker.log_epoch(epoch, avg_train_loss, current_lr, grad_norm)
            print(f"  Validation: loss={val_loss:.4f} ppl={math.exp(min(val_loss, 20.0)):.2f}")

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  Best model so far (val_loss={val_loss:.4f})")

        # Save final checkpoint
        model.cpu()
        lamport = self.clock.tick()

        # Serialize model state
        state_dict = model.state_dict()
        weight_path = str(self.store.weights_dir / f"gpt2-{self.room_name}-L{lamport}.pt")
        torch.save(state_dict, weight_path)
        raw_bytes = Path(weight_path).read_bytes()
        c_hash = content_hash(raw_bytes)

        tile = TrainingTile(
            tile_id=f"gpt2-{self.room_name}-L{lamport:03d}",
            room=self.room_name,
            tile_type=TileType.CHECKPOINT,
            state=TileLifecycle.ACTIVE,
            lamport=lamport,
            name=f"gpt2-{self.room_name}",
            description=(
                f"GPT-2 checkpoint: {self.config.n_layer}L {self.config.n_head}H "
                f"{self.config.n_embd}D, {sum(p.numel() for p in model.parameters()):,} params"
            ),
            content_hash=c_hash,
            training_config=train_config,
            metrics=self.tracker.get_metrics(),
            source_room=self.room_name,
        )
        self.store.save(tile)

        # Save tokenizer
        tok_path = self.store.store_dir / "tokenizer.json"
        if hasattr(self.tokenizer, 'save'):
            self.tokenizer.save(str(tok_path))

        # Move model back to device for continued use
        model.to(self.device)

        print(f"\nTraining complete. Checkpoint: {tile.tile_id}")
        print(f"  Final train loss: {self.tracker.epoch_losses[-1]:.4f}")
        print(f"  Final val loss: {best_val_loss:.4f}")
        if torch.cuda.is_available():
            print(f"  Peak VRAM: {self.tracker.peak_vram_mb:.0f}MB / "
                  f"{torch.cuda.get_device_properties(0).total_memory/1e6:.0f}MB")

        return tile

    def _validate(self, model: GPT2Model, val_loader: DataLoader) -> float:
        """Run validation and return average loss."""
        model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = model(x)["logits"]
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=-1,
                )
                total_loss += loss.item()
                n_batches += 1

        model.train()
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def generate(
        self,
        prompt: str = "",
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Text prompt
            max_new_tokens: Tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling filter
            do_sample: If False, greedy argmax

        Returns:
            Generated text string
        """
        model = self.load_model()
        model.eval()

        # Tokenize prompt
        prompt_ids = self.tokenizer.encode(prompt)
        if len(prompt_ids) > self.config.block_size - 1:
            prompt_ids = prompt_ids[-(self.config.block_size - 1):]

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        # Generate
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
        )

        result = self.tokenizer.decode(output_ids[0].tolist())
        model.train()
        return result

    def load_checkpoint(self, tile_id: str) -> GPT2Model:
        """Load a checkpoint from a tile."""
        tile = self.store.load(tile_id)
        if tile is None:
            raise FileNotFoundError(f"Tile not found: {tile_id}")

        # Load weights
        weight_path = str(self.store.weights_dir / f"gpt2-{self.room_name}-L{tile.lamport}.pt")
        if not Path(weight_path).exists():
            weight_path = str(self.store.weights_dir / f"{tile.content_hash}.safetensors")

        if not Path(weight_path).exists():
            raise FileNotFoundError(f"Weights not found for {tile_id}: {weight_path}")

        model = self.load_model()
        state_dict = torch.load(weight_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        print(f"Loaded checkpoint {tile.tile_id} ({tile.description})")
        return model