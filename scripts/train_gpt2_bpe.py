#!/usr/bin/env python3
"""
GPT-2 BPE Training — Real PLATO Corpus

Upgraded from character-level (224K chars, 162 vocab) to BPE tokenization
with 1MB+ of real corpus text and a bigger model (6L/6H/384D, ~10M params).

Collects EVERY available text file from the workspace:
  - All READMEs from every cloned repo (~120 repos)
  - All Python source files from spreader-tool, plato-training, openhuman/cocapn, openarm/cocapn
  - All markdown docs from SuperInstance/docs/
  - All essays from ai-writings/
  - All signal chain papers
  - AGENTS.md, TOOLS.md, MEMORY.md, HEARTBEAT.md, IDENTITY.md, SOUL.md
  - All flux-research docs
Target: 1MB+ of real corpus text

Uses tiktoken GPT-2 BPE tokenizer (50257 vocab).
Model: n_layer=6, n_head=6, n_embd=384, block_size=256 (~10M params).
Train: 2000 steps, cos LR schedule, gradient accumulation.
Generates samples every 500 steps.
"""

import sys
import os
import time
import math
import json
import torch
import textwrap
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from plato_training.gpt2_room import (
    GPT2Config, GPT2Model, CausalSelfAttention, MLP, TransformerBlock,
    NewGELU,
)
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

WORKSPACE = Path("/home/phoenix/.openclaw/workspace")


# ═══════════════════════════════════════════════════════════════════════
# BPE Tokenizer (uses tiktoken)
# ═══════════════════════════════════════════════════════════════════════

class Gpt2BpeTokenizer:
    """GPT-2 BPE tokenizer using tiktoken."""

    def __init__(self):
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab  # 50,257
        self.bos_token_id = None  # GPT-2 has no BOS
        self.pad_token_id = 50256  # EOS = pad
        self.eos_token_id = 50256

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text, disallowed_special=())

    def encode_chunked(self, text: str, chunk_size: int = 500_000) -> list[int]:
        """Encode large texts in chunks to avoid memory spikes."""
        tokens = []
        for i in range(0, len(text), chunk_size):
            tokens.extend(self.enc.encode(text[i:i+chunk_size], disallowed_special=()))
        return tokens

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)

    def __len__(self):
        return self.vocab_size


# ═══════════════════════════════════════════════════════════════════════
# Data Collection — EVERYTHING
# ═══════════════════════════════════════════════════════════════════════

def collect_corpus() -> str:
    """Collect EVERY available text file from workspace. Target: 1MB+."""
    sources = []

    def add_if_exists(path, label=None):
        p = Path(path)
        if p.exists() and p.is_file():
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
                sources.append(text)
                print(f"  + {(label or p.name):50s} {len(text):>8,} chars")
            except Exception as e:
                print(f"  ! {p.name}: {e}")

    def add_tree(root, glob_pattern, label_prefix=""):
        root = Path(root)
        if not root.exists():
            return
        for f in sorted(root.glob(glob_pattern)):
            if f.is_file() and f.stat().st_size > 0:
                try:
                    text = f.read_text(encoding="utf-8", errors="replace")
                    # Skip tiny files and binary-seeming files
                    if len(text) < 10 or text.count("\0") > 10:
                        continue
                    rel = f.relative_to(WORKSPACE)
                    sources.append(text)
                    print(f"  + {label_prefix}/{rel.name:45s} {len(text):>8,} chars")
                except:
                    pass

    # ── Core config / memory files ──
    print("\n  ── Core Files ──")
    for fname in ["MEMORY.md", "HEARTBEAT.md", "AGENTS.md", "TOOLS.md",
                   "IDENTITY.md", "SOUL.md", "USER.md", "COMMS.md"]:
        add_if_exists(WORKSPACE / fname)

    # ── All READMEs from every cloned repo ──
    print("\n  ── All READMEs ──")
    for p in sorted(WORKSPACE.glob("*/README.md")):
        add_if_exists(p)

    # Also root README
    add_if_exists(WORKSPACE / "README.md")

    # ── Python source files from key repos ──
    print("\n  ── Python Sources (spreader-tool) ──")
    add_tree(WORKSPACE / "spreader-tool", "**/*.py", "spreader-tool")

    print("\n  ── Python Sources (plato-training) ──")
    add_tree(WORKSPACE / "plato-training", "**/*.py", "plato-training")

    print("\n  ── Python Sources (openhuman/cocapn) ──")
    add_tree(WORKSPACE / "openhuman", "**/*.py", "openhuman")

    print("\n  ── Python Sources (openarm/cocapn) ──")
    add_tree(WORKSPACE / "openarm/cocapn", "**/*.py", "openarm")

    # ── SuperInstance docs ──
    print("\n  ── SuperInstance Docs ──")
    add_tree(WORKSPACE / "SuperInstance", "*.md", "SuperInstance")
    add_tree(WORKSPACE / "SuperInstance/docs", "*.md", "SuperInstance/docs")
    add_tree(WORKSPACE / "SuperInstance/architecture", "*.md", "SuperInstance/arch")
    add_tree(WORKSPACE / "SuperInstance/protocols", "*.md", "SuperInstance/protocols")
    add_tree(WORKSPACE / "SuperInstance/agents", "*.md", "SuperInstance/agents")
    add_tree(WORKSPACE / "SuperInstance/research", "*.md", "SuperInstance/research")
    add_tree(WORKSPACE / "SuperInstance/INDEXES", "*.md", "SuperInstance/indexes")
    add_tree(WORKSPACE / "SuperInstance/writing", "*.md", "SuperInstance/writing")
    add_tree(WORKSPACE / "SuperInstance/workshop", "*.md", "SuperInstance/workshop")
    add_tree(WORKSPACE / "SuperInstance/cultural-perspectives", "*.md", "SuperInstance/cultural")
    add_tree(WORKSPACE / "SuperInstance/message-in-a-bottle", "*.md", "SuperInstance/miab")

    # ── ai-writings ──
    print("\n  ── AI Writings ──")
    add_tree(WORKSPACE / "ai-writings", "*.md", "ai-writings")

    # ── Signal chain papers ──
    print("\n  ── Signal Chain Papers ──")
    add_tree(WORKSPACE / "signal-chain", "*.md", "signal-chain")
    add_tree(WORKSPACE / "signal-chain/papers", "*.md", "signal-chain/papers")

    # ── Flux research docs ──
    print("\n  ── Flux Research ──")
    add_tree(WORKSPACE / "flux-research", "*.md", "flux-research")
    add_tree(WORKSPACE / "flux-research/ten-forward", "*.md", "flux-research/tf")
    add_tree(WORKSPACE / "flux-research/autonomous", "*.md", "flux-research/auto")
    add_tree(WORKSPACE / "flux-research/dsml-sessions", "*.md", "flux-research/dsml")
    add_tree(WORKSPACE / "flux-research/kimi-critique", "*.md", "flux-research/kimi-critique")

    # ── Papers from all repos with papers dir ──
    print("\n  ── Additional Markdown Sources ──")
    for dirpath in WORKSPACE.glob("*/papers"):
        add_tree(dirpath, "*.md", f"{dirpath.name}/papers")

    # ── Additional Python from various repos ──
    print("\n  ── Additional Python Sources ──")
    for repo in ["tensor-spline", "plato-types", "plato-data",
                  "eisenstein", "neural-plato", "fleet-math-py",
                  "fleet-router", "penrose", "ct-demo",
                  "constraint-theory-py", "constraint-inference",
                  "architecture", "plato-mcp", "plato-room-intelligence",
                  "papers", "dissertation"]:
        add_tree(WORKSPACE / repo, "**/*.py", repo)

    corpus = "\n\n".join(sources)
    print(f"\n  ═══════════════════════════════════════════════════════")
    print(f"  TOTAL CORPUS: {len(corpus):,} chars")
    print(f"              ~{len(corpus)//4:,} tokens (approx)")
    return corpus


# ═══════════════════════════════════════════════════════════════════════
# BPE Dataset
# ═══════════════════════════════════════════════════════════════════════

class BpeTextDataset(Dataset):
    """Sliding-window dataset over BPE-encoded text."""

    def __init__(self, texts: list[str], tokenizer, block_size: int = 256,
                 stride: int = 128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride

        # Tokenize all texts (chunked for large corpora)
        self.tokens = []
        for text in texts:
            try:
                if len(text) > 500_000:
                    self.tokens.extend(tokenizer.encode_chunked(text))
                else:
                    self.tokens.extend(tokenizer.encode(text))
            except Exception as e:
                print(f"  ! Skipping text block: {e}")
                continue

        print(f"  Tokenized corpus: {len(self.tokens):,} BPE tokens")

        # Build sliding window indices
        self.indices = []
        for i in range(0, len(self.tokens) - block_size, stride):
            self.indices.append(i)

        print(f"  Dataset samples: {len(self.indices):,}")

    def __len__(self) -> int:
        return max(1, len(self.indices))

    def __getitem__(self, idx: int):
        i = self.indices[idx]
        x = torch.tensor(self.tokens[i:i + self.block_size], dtype=torch.long)
        y = torch.tensor(self.tokens[i + 1:i + self.block_size + 1], dtype=torch.long)
        return x, y


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════

def generate_sample(model, tokenizer, prompt, device, config, max_new=200):
    """Generate text from prompt with temperature sampling."""
    model.eval()
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) > config.block_size - 1:
        prompt_ids = prompt_ids[-(config.block_size - 1):]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output_ids = input_ids
        for _ in range(max_new):
            if output_ids.size(1) >= config.block_size:
                output_ids = output_ids[:, -config.block_size:]
            logits = model(output_ids)["logits"]
            logits = logits[:, -1, :]
            logits = logits / 0.8
            # top-k=50
            v, _ = torch.topk(logits, min(50, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            output_ids = torch.cat([output_ids, next_token], dim=1)

    generated = tokenizer.decode(output_ids[0].tolist())
    model.train()
    return generated


def train_gpt2_bpe():
    print("=" * 70)
    print("  GPT-2 BPE TRAINING — REAL PLATO CORPUS")
    print("  Model: 6L 6H 384D, block=256, ~10M params")
    print("  Tokenizer: GPT-2 BPE (tiktoken, 50,257 vocab)")
    print("  Steps: 2000, LR: 3e-4, Warmup: 200")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e6:.0f}MB")

    # ── Collect corpus ──
    print("\n[1/6] Collecting training corpus...")
    corpus = collect_corpus()

    # ── Tokenizer ──
    print("\n[2/6] Initializing BPE tokenizer...")
    tokenizer = Gpt2BpeTokenizer()
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # ── Config ──
    print("\n[3/6] Creating GPT2Config (6L 6H 384D)...")
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_layer=6,
        n_head=6,
        n_embd=384,
        block_size=256,
        dropout=0.1,
        bias=True,
    )
    param_count = config.param_count()
    print(f"  Config: {config.n_layer}L {config.n_head}H {config.n_embd}D")
    print(f"  Block size: {config.block_size}")
    print(f"  Estimated params: {param_count:,}")

    # ── Model ──
    print("\n[4/6] Creating model...")
    model = GPT2Model(config)
    model.to(device)
    actual_params = sum(p.numel() for p in model.parameters())
    print(f"  Actual params: {actual_params:,}")

    # ── Dataset ──
    print("\n[5/6] Creating dataset...")
    dataset = BpeTextDataset(
        texts=[corpus],
        tokenizer=tokenizer,
        block_size=config.block_size,
        stride=config.block_size // 2,
    )
    print(f"  Total samples: {len(dataset)}")

    # Split train/val
    val_size = max(1, int(len(dataset) * 0.05))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # ── Training setup ──
    batch_size = 8
    grad_accum_steps = 4
    num_steps = 2000
    warmup_steps = 200
    learning_rate = 3e-4
    max_grad_norm = 1.0

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device == "cuda"),
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
            {'params': decay_params, 'weight_decay': 0.01},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=learning_rate, betas=(0.9, 0.95),
    )

    def get_lr(it: int) -> float:
        if it < warmup_steps:
            return learning_rate * (it + 1) / (warmup_steps + 1)
        progress = (it - warmup_steps) / max(num_steps - warmup_steps, 1)
        return learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    # ── Training loop ──
    print(f"\n[6/6] Training for {num_steps} steps...")
    print(f"  Batch: {batch_size}, Grad accum: {grad_accum_steps}")
    print(f"  Effective batch: {batch_size * grad_accum_steps}")
    print(f"  LR: {learning_rate}, Warmup: {warmup_steps}")
    print(f"  {'─' * 60}")

    loss_curve = []
    val_losses = []
    global_step = 0
    epoch = 0
    start_time = time.time()
    best_val_loss = float('inf')
    all_generations = []

    # Generation prompts
    gen_prompts = [
        "The PLATO system defines rooms as",
        "The signal chain architecture is based on",
        "In the fleet architecture, each agent",
        "A constraint satisfaction problem consists of",
        "The I2I protocol enables",
    ]

    model.train()
    optimizer.zero_grad()

    # For tracking time
    step_times = []

    while global_step < num_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if global_step >= num_steps:
                break

            t0 = time.time()
            x, y = x.to(device), y.to(device)

            logits = model(x)["logits"]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=-1,
            )
            loss = loss / grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = get_lr(global_step)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                dt = time.time() - t0
                step_times.append(dt)

                loss_val = loss.item() * grad_accum_steps
                loss_curve.append(loss_val)
                lr_val = optimizer.param_groups[0]['lr']

                # Log every 10 steps
                if global_step % 10 == 0 or global_step <= 5:
                    est_remain = (num_steps - global_step) * (sum(step_times[-10:]) / max(len(step_times[-10:]), 1))
                    vram_str = ""
                    if device == "cuda":
                        vram = torch.cuda.memory_allocated() / 1e6
                        vram_str = f" vram={vram:.0f}MB"
                    print(f"  Step {global_step:4d}/{num_steps}: loss={loss_val:.4f} lr={lr_val:.2e}{vram_str} eta={est_remain:.0f}s")

                # Validation + generation every 500 steps
                do_gen = (global_step % 500 == 0) or (global_step == num_steps)

                if global_step % 100 == 0 or do_gen:
                    model.eval()
                    val_loss = 0.0
                    n_val = 0
                    with torch.no_grad():
                        for vx, vy in val_loader:
                            vx, vy = vx.to(device), vy.to(device)
                            vlogits = model(vx)["logits"]
                            vl = F.cross_entropy(
                                vlogits.view(-1, vlogits.size(-1)),
                                vy.view(-1),
                                ignore_index=-1,
                            )
                            val_loss += vl.item()
                            n_val += 1
                    avg_val_loss = val_loss / max(n_val, 1)
                    val_losses.append((global_step, avg_val_loss))
                    val_ppl = math.exp(min(avg_val_loss, 20.0))
                    print(f"  ══ VALIDATION (step {global_step}): loss={avg_val_loss:.4f} ppl={val_ppl:.2f} ══")

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        print(f"  ══ New best model! ══")

                    model.train()

                # Generate samples every 500 steps (not first 100 steps of first gen)
                if do_gen:
                    print(f"\n  ── Generated Text (step {global_step}) ──")
                    for prompt in gen_prompts:
                        generated = generate_sample(model, tokenizer, prompt, device, config, max_new=150)
                        all_generations.append((global_step, prompt, generated))
                        print(f"\n  Prompt: \"{prompt}\"")
                        # Print first few lines
                        lines = generated.split('\n')[:5]
                        for line in lines:
                            if line.strip():
                                print(f"  > {line.strip()[:120]}")
                    print(f"\n  {'─' * 60}")

        epoch += 1

    training_time = time.time() - start_time

    # ── Final validation ──
    model.eval()
    final_val_loss = 0.0
    n_val = 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx, vy = vx.to(device), vy.to(device)
            vlogits = model(vx)["logits"]
            vl = F.cross_entropy(
                vlogits.view(-1, vlogits.size(-1)),
                vy.view(-1),
                ignore_index=-1,
            )
            final_val_loss += vl.item()
            n_val += 1
    final_val_loss = final_val_loss / max(n_val, 1)
    final_perplexity = math.exp(min(final_val_loss, 20.0))

    peak_vram_mb = 0
    if device == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6

    avg_step_time = sum(step_times) / max(len(step_times), 1)

    print(f"\n{'=' * 70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Training time: {training_time:.1f}s ({training_time/60:.1f}min)")
    print(f"  Avg step time: {avg_step_time:.3f}s")
    print(f"  Steps/s: {1.0/avg_step_time:.1f}")
    print(f"  Final train loss: {loss_curve[-1]:.4f}")
    print(f"  Final val loss: {final_val_loss:.4f}")
    print(f"  Final perplexity: {final_perplexity:.2f}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Peak VRAM: {peak_vram_mb:.0f}MB" if device == "cuda" else "  Peak VRAM: N/A")
    print(f"  Parameters: {actual_params:,}")

    # ── Final generation (long) ──
    print(f"\n{'=' * 70}")
    print(f"  FINAL GENERATIONS")
    print(f"{'=' * 70}")

    final_gen_prompts = [
        "PLATO rooms are",
        "The for-fleet repository contains",
        "The tensor spline compresses",
        "Constraint theory states that",
    ]

    for prompt in final_gen_prompts:
        generated = generate_sample(model, tokenizer, prompt, device, config, max_new=300)
        all_generations.append(("FINAL", prompt, generated))
        print(f"\n  Prompt: \"{prompt}\"")
        print(f"  {'─' * 60}")
        lines = textwrap.wrap(generated[:500], width=70)
        for line in lines[:15]:
            print(f"  {line}")

    # ── Save results ──
    results_path = Path(__file__).parent / "training_results_bpe.md"
    with open(results_path, 'w') as f:
        f.write(f"# GPT-2 BPE Training Results\n\n")
        f.write(f"**Date:** 2026-05-17\n")
        f.write(f"**Model:** GPT-2 ({config.n_layer}L {config.n_head}H {config.n_embd}D)\n")
        f.write(f"**Tokenizer:** GPT-2 BPE (tiktoken, {tokenizer.vocab_size} vocab)\n")
        f.write(f"**Parameters:** {actual_params:,}\n")
        f.write(f"**Device:** {device}\n")
        f.write(f"**Training steps:** {num_steps}\n\n")

        f.write("## Corpus\n\n")
        f.write(f"- Total corpus size: {len(corpus):,} characters\n")
        f.write(f"- Tokenized length: {len(dataset.tokens):,} BPE tokens\n")
        f.write(f"- Dataset samples: {len(dataset)}\n\n")

        f.write("## Training Metrics\n\n")
        f.write(f"- **Training time:** {training_time:.1f}s ({training_time/60:.1f}min)\n")
        f.write(f"- **Avg step time:** {avg_step_time:.3f}s\n")
        f.write(f"- **Steps/s:** {1.0/avg_step_time:.1f}\n")
        f.write(f"- **Final train loss:** {loss_curve[-1]:.4f}\n")
        f.write(f"- **Final val loss:** {final_val_loss:.4f}\n")
        f.write(f"- **Final perplexity:** {final_perplexity:.2f}\n")
        f.write(f"- **Best val loss:** {best_val_loss:.4f}\n")
        f.write(f"- **Peak VRAM:** {peak_vram_mb:.0f}MB\n")
        f.write(f"- **Parameters:** {actual_params:,}\n\n")

        f.write("## Loss Curve (sampled)\n\n")
        f.write("| Step | Train Loss |\n")
        f.write("|------|-----------|\n")
        for i in range(0, len(loss_curve), max(1, len(loss_curve) // 20)):
            val_info = ""
            for vs, vl in val_losses:
                if abs(vs - (i + 1)) <= 5:
                    val_info = f" (val: {vl:.4f})"
            f.write(f"| {i} | {loss_curve[i]:.4f}{val_info} |\n")
        f.write(f"| {len(loss_curve)-1} | {loss_curve[-1]:.4f} |\n")

        f.write("\n## Validation Checkpoints\n\n")
        f.write("| Step | Val Loss | Perplexity |\n")
        f.write("|------|----------|------------|\n")
        for vs, vl in val_losses:
            vppl = math.exp(min(vl, 20.0))
            f.write(f"| {vs} | {vl:.4f} | {vppl:.2f} |\n")

        f.write("\n## Generated Text Samples\n\n")
        for item in all_generations:
            step, prompt, text = item
            f.write(f"### Step {step}: \"{prompt}\"\n\n")
            f.write("```\n")
            f.write(text[:800])
            if len(text) > 800:
                f.write("\n[...]")
            f.write("\n```\n\n")

        f.write("## Training Configuration\n\n")
        f.write("```python\n")
        f.write(f"n_layer={config.n_layer}\n")
        f.write(f"n_head={config.n_head}\n")
        f.write(f"n_embd={config.n_embd}\n")
        f.write(f"block_size={config.block_size}\n")
        f.write(f"vocab_size={config.vocab_size}\n")
        f.write(f"batch_size={batch_size}\n")
        f.write(f"gradient_accumulation_steps={grad_accum_steps}\n")
        f.write(f"effective_batch_size={batch_size * grad_accum_steps}\n")
        f.write(f"learning_rate={learning_rate}\n")
        f.write(f"warmup_steps={warmup_steps}\n")
        f.write(f"num_steps={num_steps}\n")
        f.write(f"max_grad_norm={max_grad_norm}\n")
        f.write(f"optimizer=AdamW\n")
        f.write(f"weight_decay=0.01\n")
        f.write(f"scheduler=cosine_with_warmup\n")
        f.write("```\n")

    print(f"\n  Results saved to: {results_path}")
    print(f"  Done!\n")

    return {
        "config": config,
        "model": model,
        "tokenizer": tokenizer,
        "loss_curve": loss_curve,
        "val_losses": val_losses,
        "final_val_loss": final_val_loss,
        "final_perplexity": final_perplexity,
        "param_count": actual_params,
        "training_time": training_time,
        "peak_vram_mb": peak_vram_mb,
        "generations": all_generations,
    }


if __name__ == "__main__":
    train_gpt2_bpe()
