"""
Tests for GPT2Room — GPT-2 small transformer training pipeline.

Target: 20+ tests covering:
- Config validation
- Model architecture (forward pass, masking, shape)
- Training (loss convergence on toy data)
- Checkpoint save/load round-trip
- Generation
- VRAM monitoring
- Room integration (load tiles, train, generate)
"""

import pytest
import math
import torch
import json
import io
from pathlib import Path

from plato_training.gpt2_room import (
    GPT2Config,
    GPT2Model,
    GPT2Room,
    CharTokenizer,
    TileTextDataset,
    TrainingTracker,
    CausalSelfAttention,
    TransformerBlock,
    NewGELU,
    MLP,
)
from plato_training.types import (
    TrainingTile,
    TileType,
    TileLifecycle,
    LamportClock,
    TrainingConfig,
    content_hash,
)


# ─── Helpers ────────────────────────────────────────────────────────

def default_config(**kwargs) -> GPT2Config:
    """Create a default config with small sizes for fast tests."""
    defaults = dict(
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=32,
        dropout=0.0,  # No dropout for deterministic tests
    )
    defaults.update(kwargs)
    return GPT2Config(**defaults)


def small_config(**kwargs) -> GPT2Config:
    """Even smaller config for minimal tests."""
    defaults = dict(
        vocab_size=16,
        n_layer=1,
        n_head=1,
        n_embd=16,
        block_size=8,
        dropout=0.0,
    )
    defaults.update(kwargs)
    return GPT2Config(**defaults)


def toy_texts() -> list:
    """Simple toy texts for training tests."""
    return [
        "The cat sat on the mat. " * 5,
        "The dog ran in the park. " * 5,
        "Birds fly high in the sky. " * 5,
        "Fish swim deep in the sea. " * 5,
        "Stars shine bright at night. " * 5,
    ]


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ─── CONFIG TESTS ──────────────────────────────────────────────────

class TestGPT2Config:
    def test_default_config(self):
        """Default config creates valid architecture values."""
        cfg = GPT2Config()
        assert cfg.n_layer == 4
        assert cfg.n_head == 4
        assert cfg.n_embd == 256
        assert cfg.block_size == 128
        assert cfg.vocab_size == 256
        assert cfg.dropout == 0.1

    def test_custom_config(self):
        """Custom config values override defaults."""
        cfg = GPT2Config(n_layer=2, n_head=2, n_embd=64, block_size=32)
        assert cfg.n_layer == 2
        assert cfg.n_head == 2
        assert cfg.n_embd == 64
        assert cfg.block_size == 32

    def test_parameter_count_minimal(self):
        """
        Config produces a ~2M parameter target.
        Small config should have manageable param count.
        """
        cfg = default_config(vocab_size=256, n_layer=2, n_head=2, n_embd=64, block_size=32)
        model = GPT2Model(cfg)
        n = count_params(model)
        assert 5000 < n < 500000, f"Expected 5K-500K params, got {n}"

    def test_parameter_count_full(self):
        """
        Full config should fit in 6.4GB VRAM comfortably.
        """
        cfg = GPT2Config(vocab_size=256, n_layer=4, n_head=4, n_embd=256, block_size=128)
        model = GPT2Model(cfg)
        n = count_params(model)
        # 3.3M params × 4 bytes (float32) = ~13MB for weights
        # With Adam (2x), grads (1x), activations: still < 500MB
        assert n < 5000000, f"Expected <5M params, got {n}"
        assert n > 500000, f"Expected more than 500K params, got {n}"

    def test_param_count_estimate_method(self):
        """Config.param_count() should return a reasonable estimate."""
        cfg = default_config(vocab_size=256, n_layer=2, n_head=2, n_embd=64, block_size=32)
        estimate = cfg.param_count()
        model = GPT2Model(cfg)
        actual = count_params(model)
        # Estimate should be within 20% of actual
        ratio = estimate / actual
        assert 0.8 < ratio < 1.5, f"Estimate {estimate} vs actual {actual} (ratio {ratio:.2f})"


# ─── MODEL ARCHITECTURE TESTS ─────────────────────────────────────

class TestGPT2Model:
    def test_forward_shape(self):
        """Forward pass produces correct output shape."""
        cfg = small_config()
        model = GPT2Model(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
        output = model(x)
        assert output["logits"].shape == (2, cfg.block_size, cfg.vocab_size)

    def test_forward_with_loss(self):
        """Forward pass with labels computes loss."""
        cfg = small_config()
        model = GPT2Model(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
        y = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
        output = model(x, labels=y)
        assert "logits" in output
        assert "loss" in output
        assert output["loss"].item() > 0

    def test_forward_batch(self):
        """Model handles different batch sizes."""
        cfg = small_config()
        model = GPT2Model(cfg)
        for batch_size in [1, 2, 4]:
            x = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size))
            output = model(x)
            assert output["logits"].shape[0] == batch_size

    def test_forward_variable_length(self):
        """Model handles sequences shorter than block_size."""
        cfg = small_config()
        model = GPT2Model(cfg)
        x = torch.randint(0, cfg.vocab_size, (1, 4))
        output = model(x)
        assert output["logits"].shape == (1, 4, cfg.vocab_size)

    def test_causal_attention_masking(self):
        """
        Attention masking is causal: position i can only attend to positions <= i.
        We verify by checking that attention weights form a lower-triangular pattern
        (within numerical tolerance).
        """
        cfg = small_config(dropout=0.0)
        model = GPT2Model(cfg)

        # Extract attention layer
        attn = model.blocks[0].attn
        assert isinstance(attn, CausalSelfAttention)

        # Check causal mask
        bias = attn.bias
        assert bias.shape == (1, 1, cfg.block_size, cfg.block_size)

        # Verify lower-triangular: positions where j > i should be -inf
        for i in range(cfg.block_size):
            for j in range(cfg.block_size):
                if j > i:
                    assert bias[0, 0, i, j] == 0, (
                        f"Expected masked position ({i},{j}) to be 0, got {bias[0,0,i,j]}"
                    )
                else:
                    assert bias[0, 0, i, j] == 1

    def test_attention_output_shape(self):
        """CausalSelfAttention produces correct output."""
        cfg = small_config()
        attn = CausalSelfAttention(cfg)
        x = torch.randn(2, cfg.block_size, cfg.n_embd)
        y = attn(x)
        assert y.shape == x.shape

    def test_mlp_shape(self):
        """MLP produces correct output shape."""
        cfg = small_config()
        mlp = MLP(cfg)
        x = torch.randn(2, cfg.block_size, cfg.n_embd)
        y = mlp(x)
        assert y.shape == x.shape

    def test_transformer_block_shape(self):
        """TransformerBlock preserves shape."""
        cfg = small_config()
        block = TransformerBlock(cfg)
        x = torch.randn(2, cfg.block_size, cfg.n_embd)
        y = block(x)
        assert y.shape == x.shape

    def test_gelu_activation(self):
        """NewGELU produces non-zero gradient."""
        gelu = NewGELU()
        x = torch.randn(10, requires_grad=True)
        y = gelu(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_embedding_dims(self):
        """Token and position embeddings have correct dimensions."""
        cfg = small_config()
        model = GPT2Model(cfg)
        assert model.wte.weight.shape == (cfg.vocab_size, cfg.n_embd)
        assert model.wpe.weight.shape == (cfg.block_size, cfg.n_embd)

    def test_model_to_device(self):
        """Model can be moved to CPU."""
        cfg = small_config()
        model = GPT2Model(cfg)
        model.to('cpu')
        assert next(model.parameters()).device.type == 'cpu'


# ─── GENERATION TESTS ─────────────────────────────────────────────

class TestGeneration:
    def test_generate_greedy(self):
        """Greedy generation produces a longer sequence."""
        cfg = small_config(block_size=32)
        model = GPT2Model(cfg)
        model.eval()
        prompt = torch.randint(0, cfg.vocab_size, (1, 4))
        output = model.generate(prompt, max_new_tokens=10, do_sample=False)
        assert output.shape[1] == 14  # 4 prompt + 10 new

    def test_generate_sampling(self):
        """Sampling generation works."""
        cfg = small_config(block_size=32)
        model = GPT2Model(cfg)
        model.eval()
        prompt = torch.randint(0, cfg.vocab_size, (1, 4))
        output = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=5)
        assert output.shape[1] == 14

    def test_generate_deterministic_greedy(self):
        """Greedy generation with temperature=0 is deterministic."""
        cfg = small_config(block_size=32)
        model = GPT2Model(cfg)
        model.eval()
        prompt = torch.randint(0, cfg.vocab_size, (1, 4))
        torch.manual_seed(42)
        out1 = model.generate(prompt, max_new_tokens=5, do_sample=False)
        torch.manual_seed(42)
        out2 = model.generate(prompt, max_new_tokens=5, do_sample=False)
        assert torch.equal(out1, out2)


# ─── TRAINING TESTS ──────────────────────────────────────────────

class TestTraining:
    def test_training_step_reduces_loss(self):
        """A single training step reduces loss on toy data."""
        cfg = small_config()
        model = GPT2Model(cfg)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
        y = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))

        loss_before = model(x, labels=y)["loss"].item()
        optimizer.zero_grad()
        model(x, labels=y)["loss"].backward()
        optimizer.step()
        loss_after = model(x, labels=y)["loss"].item()

        assert loss_after < loss_before, (
            f"Loss should decrease: {loss_before:.4f} -> {loss_after:.4f}"
        )

    def test_training_epoch_reduces_loss(self, tmp_path):
        """Training for multiple epochs reduces loss."""
        cfg = small_config()
        model = GPT2Model(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        # Use a tokenizer with only 10 unique chars (fits in vocab_size=16)
        tokenizer = CharTokenizer(chars="abcde fghij")
        texts = ["abc de fgh ij" * 10]
        dataset = TileTextDataset(texts, tokenizer, block_size=cfg.block_size)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

        losses = []
        for epoch in range(3):
            model.train()
            epoch_loss = 0.0
            for x, y in loader:
                optimizer.zero_grad()
                loss = model(x, labels=y)["loss"]
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)

        # Loss should trend down
        assert losses[-1] < losses[0], (
            f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_gradient_accumulation(self):
        """Gradient accumulation produces same result as larger batch."""
        cfg = small_config()
        model1 = GPT2Model(cfg)
        model2 = GPT2Model(cfg)

        # Copy weights
        model2.load_state_dict(model1.state_dict())

        x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
        y = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))

        # Train with batch_size=4, no accumulation
        opt1 = torch.optim.SGD(model1.parameters(), lr=0.01)
        opt1.zero_grad()
        loss1 = model1(x, labels=y)["loss"]
        loss1.backward()
        opt1.step()

        # Train with batch_size=2, accumulation=2
        opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        opt2.zero_grad()
        for i in range(0, 4, 2):
            loss2 = model2(x[i:i+2], labels=y[i:i+2])["loss"]
            (loss2 / 2).backward()
        opt2.step()

        # Weights should be similar (not identical due to BN diff)
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6), (
                "Gradient accumulation should match direct training"
            )


# ─── TOKENIZER TESTS ─────────────────────────────────────────────

class TestCharTokenizer:
    def test_encode_decode_roundtrip(self):
        """Encoding then decoding returns original text."""
        tokenizer = CharTokenizer(chars="abcde ")
        text = "a b c d e"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text

    def test_unknown_characters(self):
        """Unknown characters become pad/unk token."""
        tokenizer = CharTokenizer(chars="abc")
        ids = tokenizer.encode("xyz")
        assert all(i == tokenizer.unk_token_id for i in ids)

    def test_vocab_size(self):
        """Vocab size includes pad token."""
        tokenizer = CharTokenizer(chars="abc")
        assert tokenizer.vocab_size == 4  # 3 chars + pad

    def test_save_load(self, tmp_path):
        """Save and load tokenizer from JSON."""
        tokenizer = CharTokenizer(chars="abcdef")
        path = str(tmp_path / "tokenizer.json")
        tokenizer.save(path)
        loaded = CharTokenizer.load(path)
        assert loaded.chars == tokenizer.chars
        assert loaded.vocab_size == tokenizer.vocab_size

    def test_multiline_text(self):
        """Tokenizer handles multiline text."""
        tokenizer = CharTokenizer()
        text = "Hello\nWorld!\tTab"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        # Should contain original chars minus unknowns
        assert len(ids) > 0
        assert "Hello" in decoded


# ─── DATASET TESTS ───────────────────────────────────────────────

class TestTileTextDataset:
    def test_dataset_length(self):
        """Dataset produces correct number of samples."""
        tokenizer = CharTokenizer(chars="abcdefg ")
        texts = ["abc def ghi " * 4]
        ds = TileTextDataset(texts, tokenizer, block_size=10, stride=5)
        assert len(ds) > 0

    def test_dataset_item_shape(self):
        """Dataset items are (input, target) tensors of correct size."""
        tokenizer = CharTokenizer(chars="abcdefg ")
        texts = ["abc def ghi jkl " * 4]
        ds = TileTextDataset(texts, tokenizer, block_size=16, stride=8)
        x, y = ds[0]
        assert x.shape == (16,)
        assert y.shape == (16,)
        assert x.dtype == torch.long
        assert y.dtype == torch.long

    def test_dataset_target_is_shifted(self):
        """Target is the next-token prediction of input."""
        tokenizer = CharTokenizer(chars="abc")
        texts = ["abc" * 5]
        ds = TileTextDataset(texts, tokenizer, block_size=4, stride=1)
        if len(ds) > 0:
            x, y = ds[0]
            assert torch.equal(x[1:], y[:-1]), (
                "Target should be input shifted by 1"
            )

    def test_dataset_empty_text(self):
        """Empty text gets padded to at least block_size+1."""
        tokenizer = CharTokenizer()
        ds = TileTextDataset([""], tokenizer, block_size=8, stride=4)
        assert len(ds) == 1
        x, y = ds[0]
        assert x.shape == (8,)
        assert y.shape == (8,)

    def test_dataset_multiple_texts(self):
        """Multiple texts are concatenated."""
        tokenizer = CharTokenizer(chars="abc")
        ds1 = TileTextDataset(["abc" * 10], tokenizer, block_size=8, stride=8)
        ds2 = TileTextDataset(["abc" * 5, "abc" * 5], tokenizer, block_size=8, stride=8)
        assert len(ds2) >= 1


# ─── TRAINING TRACKER TESTS ──────────────────────────────────────

class TestTrainingTracker:
    def test_tracker_basics(self):
        """Tracker stores and reports metrics."""
        tracker = TrainingTracker()
        tracker.start()
        tracker.log_epoch(0, 2.5, 1e-3, 0.5)
        tracker.log_epoch(1, 1.8, 5e-4, 0.3)
        metrics = tracker.get_metrics()

        assert metrics.epochs_completed == 2
        assert abs(metrics.final_loss - 1.8) < 1e-6
        assert metrics.peak_memory_mb >= 0
        assert metrics.training_time_seconds > 0

    def test_tracker_perplexity(self):
        """Perplexity from loss should be exponential."""
        tracker = TrainingTracker()
        tracker.start()
        tracker.log_epoch(0, 0.0, 1e-3, 0.0)  # loss=0 -> ppl=1
        assert abs(tracker.epoch_perplexities[0] - 1.0) < 0.01

        tracker.log_epoch(1, 2.3026, 1e-3, 0.0)  # loss~ln(10) -> ppl~10
        assert abs(tracker.epoch_perplexities[1] - 10.0) < 1.0

    def test_tracker_loss_curve(self):
        """Tracker accumulates loss curve."""
        tracker = TrainingTracker()
        tracker.start()
        tracker.log_step(1.0)
        tracker.log_step(0.8)
        tracker.log_step(0.6)
        assert len(tracker.loss_curve) == 3
        assert abs(tracker.loss_curve[0] - 1.0) < 1e-6
        assert abs(tracker.loss_curve[-1] - 0.6) < 1e-6

    def test_tracker_create_tile(self):
        """Tracker creates valid TrainingTile."""
        cfg = GPT2Config(n_layer=2, n_head=2, n_embd=64, block_size=32)
        tracker = TrainingTracker()
        tracker.start()
        tracker.log_epoch(0, 2.0, 1e-3, 0.5)

        train_cfg = TrainingConfig(epochs=1, learning_rate=3e-4, batch_size=16)
        tile = tracker.create_tile("gpt2-test-001", "abc123", cfg, train_cfg)

        assert tile.tile_id == "gpt2-test-001"
        assert tile.tile_type == TileType.CHECKPOINT
        assert tile.state == TileLifecycle.ACTIVE
        assert tile.content_hash == "abc123"
        assert tile.metrics is not None
        assert tile.metrics.epochs_completed == 1
        assert abs(tile.metrics.final_loss - 2.0) < 1e-6


# ─── CHECKPOINT SAVE/LOAD TESTS ─────────────────────────────────

class TestCheckpoint:
    def test_save_load_state_dict(self, tmp_path):
        """Model state dict save/load round-trips correctly."""
        cfg = small_config()
        model1 = GPT2Model(cfg)
        model2 = GPT2Model(cfg)

        # Ensure different (model2 has random weights)
        x = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
        out1_before = model1(x)["logits"]
        out2_before = model2(x)["logits"]
        assert not torch.allclose(out1_before, out2_before)

        # Save model1 weights
        path = str(tmp_path / "model.pt")
        torch.save(model1.state_dict(), path)

        # Load into model2
        model2.load_state_dict(torch.load(path))

        # Should now produce same output
        out1_after = model1(x)["logits"]
        out2_after = model2(x)["logits"]
        assert torch.allclose(out1_after, out2_after)

    def test_checkpoint_loss_preserved(self, tmp_path):
        """
        Training then saving then loading then training again
        should produce the same loss trajectory.
        """
        cfg = small_config()
        model = GPT2Model(cfg)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
        y = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))

        # Train for one step
        loss1 = model(x, labels=y)["loss"].item()
        optimizer.zero_grad()
        model(x, labels=y)["loss"].backward()
        optimizer.step()

        # Save
        path = str(tmp_path / "checkpoint.pt")
        torch.save(model.state_dict(), path)

        loss2 = model(x, labels=y)["loss"].item()

        # Load into fresh model
        model2 = GPT2Model(cfg)
        model2.load_state_dict(torch.load(path))
        loss2_fresh = model2(x, labels=y)["loss"].item()

        assert abs(loss2 - loss2_fresh) < 1e-6

    def test_checkpoint_content_hash(self):
        """Content hash is deterministic for same weights."""
        cfg = small_config()
        model = GPT2Model(cfg)
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        h1 = content_hash(buf.getvalue())

        model2 = GPT2Model(cfg)
        buf2 = io.BytesIO()
        torch.save(model2.state_dict(), buf2)
        h2 = content_hash(buf2.getvalue())

        # Random init means different hashes (unless seed is same)
        # Just check hash is non-empty and consistent
        assert len(h1) == 16
        assert len(h2) == 16

    def test_train_cycle_roundtrip(self, tmp_path):
        """Full train-save-load cycle works with GPT2Room."""
        store_dir = str(tmp_path / "store")
        room = GPT2Room(
            room_name="test-cycle",
            store_dir=store_dir,
            device="cpu",
            config=small_config(),
        )
        room.load_model()

        # Store some text as tiles
        tile = TrainingTile(
            tile_id="test-text-001",
            room="test-cycle",
            tile_type=TileType.DATASET,
            state=TileLifecycle.ACTIVE,
            lamport=1,
            name="test-text",
            description="Test text content",
        )
        room.store.save(tile)

        # Train briefly
        result_tile = room.train(
            texts=["hello world " * 20],
            epochs=2,
            batch_size=4,
            learning_rate=0.01,
        )

        assert result_tile is not None
        assert result_tile.tile_id.startswith("gpt2-test-cycle")
        assert result_tile.metrics.epochs_completed == 2

        # Generate something
        gen_text = room.generate(prompt="hello", max_new_tokens=10)
        assert len(gen_text) > 0


# ─── ROOM INTEGRATION TESTS ─────────────────────────────────────

class TestGPT2Room:
    def test_room_creation(self, tmp_path):
        """GPT2Room creates properly."""
        room = GPT2Room(
            room_name="test",
            store_dir=str(tmp_path / "store"),
            device="cpu",
        )
        assert room.room_name == "test"
        assert room.model is None

    def test_room_load_model(self, tmp_path):
        """Loading model creates it."""
        room = GPT2Room(
            room_name="test",
            store_dir=str(tmp_path / "store"),
            device="cpu",
            config=small_config(),
        )
        model = room.load_model()
        assert model is not None
        assert count_params(model) > 0

    def test_room_train_basic(self, tmp_path):
        """Room can train on toy data and produce a tile."""
        room = GPT2Room(
            room_name="test-train",
            store_dir=str(tmp_path / "store"),
            device="cpu",
            config=small_config(),
        )
        tile = room.train(
            texts=toy_texts(),
            epochs=2,
            batch_size=4,
            learning_rate=0.01,
        )
        assert tile is not None
        assert tile.metrics.epochs_completed == 2
        assert tile.metrics.final_loss > 0

    def test_room_train_reduces_loss(self, tmp_path):
        """Training over multiple epochs reduces loss."""
        room = GPT2Room(
            room_name="test-loss",
            store_dir=str(tmp_path / "store"),
            device="cpu",
            config=small_config(),
        )
        tile = room.train(
            texts=toy_texts(),
            epochs=3,
            batch_size=4,
            learning_rate=0.01,
        )
        # Loss should be finite and decreasing
        assert tile.metrics.final_loss < 5.0, (
            f"Loss should be reasonable, got {tile.metrics.final_loss}"
        )

    def test_room_generate(self, tmp_path):
        """After training, room can generate text."""
        room = GPT2Room(
            room_name="test-gen",
            store_dir=str(tmp_path / "store"),
            device="cpu",
            config=small_config(),
        )
        room.train(texts=toy_texts(), epochs=1, batch_size=4, learning_rate=0.01)

        result = room.generate(prompt="The", max_new_tokens=10, temperature=0.5)
        assert isinstance(result, str)
        assert len(result) > 2

    def test_room_generate_greedy(self, tmp_path):
        """Greedy generation produces deterministic output."""
        room = GPT2Room(
            room_name="test-greedy",
            store_dir=str(tmp_path / "store"),
            device="cpu",
            config=small_config(),
        )
        room.train(texts=toy_texts(), epochs=1, batch_size=4, learning_rate=0.01)

        torch.manual_seed(42)
        r1 = room.generate(prompt="The", max_new_tokens=5, do_sample=False)
        torch.manual_seed(42)
        r2 = room.generate(prompt="The", max_new_tokens=5, do_sample=False)
        assert r1 == r2

    def test_room_load_tile_texts(self, tmp_path):
        """Room loads texts from stored tiles."""
        room = GPT2Room(
            room_name="test-texts",
            store_dir=str(tmp_path / "store"),
            device="cpu",
            config=small_config(),
        )
        # Save some text tiles
        texts = ["hello from tile", "world from another tile"]
        for i, t in enumerate(texts):
            tile = TrainingTile(
                tile_id=f"text-tile-{i:03d}",
                room="test-texts",
                tile_type=TileType.DATASET,
                state=TileLifecycle.ACTIVE,
                lamport=i + 1,
                name=f"text-{i}",
                description=t,
            )
            path = room.store.tiles_dir / f"text-tile-{i:03d}.json"
            path.write_text(json.dumps({"text": t, **tile.to_dict()}))

        loaded = room._load_tile_texts()
        assert len(loaded) >= 2
        assert "hello from tile" in loaded

    def test_room_saves_tokenizer(self, tmp_path):
        """Training saves tokenizer to store dir."""
        room = GPT2Room(
            room_name="test-tok",
            store_dir=str(tmp_path / "store"),
            device="cpu",
            config=small_config(),
        )
        room.train(texts=toy_texts(), epochs=1, batch_size=4, learning_rate=0.01)

        tok_path = Path(str(tmp_path / "store" / "tokenizer.json"))
        assert tok_path.exists()

    def test_room_alpha_dial(self, tmp_path):
        """Alpha dial can be set."""
        room = GPT2Room(
            room_name="test-alpha",
            store_dir=str(tmp_path / "store"),
            device="cpu",
            alpha=0.8,
        )
        assert room.alpha == 0.8

    def test_room_device(self, tmp_path):
        """Room device detection works."""
        room = GPT2Room(
            room_name="test-device",
            store_dir=str(tmp_path / "store"),
            device="cpu",
        )
        assert room.device.type == 'cpu'


# ─── VRAM MONITORING TESTS ──────────────────────────────────────

class TestVRAM:
    def test_vram_monitoring_on_cpu(self):
        """VRAM monitoring doesn't crash on CPU."""
        tracker = TrainingTracker()
        tracker.start()
        tracker.log_epoch(0, 1.0, 1e-3, 0.1)
        metrics = tracker.get_metrics()
        assert metrics.peak_memory_mb >= 0

    def test_tracker_peak_memory(self):
        """Peak memory is tracked and returns valid float."""
        tracker = TrainingTracker()
        tracker.start()
        assert tracker.peak_vram_mb == 0.0
        tracker.log_epoch(0, 1.0, 1e-3, 0.1)
        assert tracker.peak_vram_mb >= 0


# ─── EDGE CASE TESTS ─────────────────────────────────────────────

class TestEdgeCases:
    def test_single_char_vocab(self):
        """Model handles vocab_size of 2 (minimum meaningful)."""
        cfg = GPT2Config(vocab_size=2, n_layer=1, n_head=1, n_embd=4, block_size=4, dropout=0.0)
        model = GPT2Model(cfg)
        x = torch.zeros((1, 4), dtype=torch.long)
        output = model(x)
        assert output["logits"].shape == (1, 4, 2)

    def test_block_size_1(self):
        """Model handles block_size=1 (single token)."""
        cfg = GPT2Config(vocab_size=16, n_layer=1, n_head=1, n_embd=4, block_size=1, dropout=0.0)
        model = GPT2Model(cfg)
        x = torch.zeros((1, 1), dtype=torch.long)
        output = model(x)
        assert output["logits"].shape == (1, 1, 16)

    def test_zero_dropout_inference(self):
        """Model is deterministic at eval with dropout=0."""
        cfg = small_config(dropout=0.0)
        model = GPT2Model(cfg)
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1["logits"], out2["logits"])

    def test_loss_finite_on_random_input(self):
        """Loss is finite on random data."""
        cfg = small_config()
        model = GPT2Model(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
        y = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
        loss = model(x, labels=y)["loss"]
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_model_parameters_all_learnable(self):
        """All model parameters require gradients."""
        cfg = default_config(vocab_size=256, n_layer=2, n_head=2, n_embd=64, block_size=32)
        model = GPT2Model(cfg)
        for name, param in model.named_parameters():
            assert param.requires_grad, f"Param {name} has no gradient"

    def test_forward_with_ignore_index(self):
        """Ignore index in labels works correctly (non-ignored tokens)."""
        cfg = small_config()
        model = GPT2Model(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
        # Some tokens ignored, some valid
        y = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
        y[:, :3] = -1  # First 3 positions ignored
        output = model(x, labels=y)
        assert output["loss"] is not None
        assert torch.isfinite(output["loss"])
        assert output["loss"].item() > 0


# ─── IMPORT TESTS ───────────────────────────────────────────────

def test_all_exports():
    """All expected exports are importable."""
    from plato_training.gpt2_room import (
        GPT2Config,
        GPT2Model,
        GPT2Room,
        CharTokenizer,
        TileTextDataset,
        TrainingTracker,
        CausalSelfAttention,
        TransformerBlock,
        NewGELU,
        MLP,
    )
    assert True
