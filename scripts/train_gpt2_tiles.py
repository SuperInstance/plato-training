"""
Train a GPT-2 model on REAL PLATO workspace text data.

Collects all READMEs, memory files, papers, and code docs from the workspace,
builds a character-level corpus, and trains a minimal GPT-2 for 100+ steps.
"""

import sys
import os
import time
import math
import json
import torch
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from plato_training.gpt2_room import (
    GPT2Config, GPT2Model, GPT2Room, CharTokenizer, TileTextDataset,
    CausalSelfAttention, MLP, TransformerBlock, TrainingTracker, NewGELU,
)
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

WORKSPACE = Path("/home/phoenix/.openclaw/workspace")

# ─── Collect ALL text from the workspace ───────────────────────────────

def collect_corpus() -> str:
    """Collect text from all specified sources, concatenate into one corpus."""
    sources = []

    # 1. README files from key repos
    readme_paths = [
        WORKSPACE / "spreader-tool" / "README.md",
        WORKSPACE / "plato-training" / "README.md",
        WORKSPACE / "tensor-spline" / "README.md",
        WORKSPACE / "spectral-conservation" / "README.md",
        WORKSPACE / "signal-chain" / "README.md",
    ]
    for p in readme_paths:
        if p.exists():
            sources.append(p.read_text())
            print(f"  + {p.name}: {len(sources[-1])} chars")

    # 2. Memory files
    memory_paths = [
        WORKSPACE / "MEMORY.md",
        WORKSPACE / "HEARTBEAT.md",
        WORKSPACE / "AGENTS.md",
        WORKSPACE / "memory" / "2026-05-17.md",
    ]
    for p in memory_paths:
        if p.exists():
            sources.append(p.read_text())
            print(f"  + {p.name}: {len(sources[-1])} chars")

    # 3. Signal chain paper
    paper = WORKSPACE / "signal-chain" / "papers" / "SIGNAL-CHAIN.md"
    if paper.exists():
        sources.append(paper.read_text())
        print(f"  + SIGNAL-CHAIN.md: {len(sources[-1])} chars")

    # 4. GPT-2 Room source code (the model itself is meta-training data)
    gpt2_room = WORKSPACE / "plato-training" / "plato_training" / "gpt2_room.py"
    if gpt2_room.exists():
        sources.append(gpt2_room.read_text())
        print(f"  + gpt2_room.py: {len(sources[-1])} chars")

    corpus = "\n\n".join(sources)
    print(f"\n  TOTAL CORPUS: {len(corpus):,} chars")
    return corpus


# ─── Build a more complete tokenizer from the corpus ───────────────────

def build_corpus_tokenizer(corpus: str) -> CharTokenizer:
    """Build a CharTokenizer that covers all chars in the corpus."""
    chars = sorted(set(corpus))
    print(f"  Unique chars in corpus: {len(chars)}")
    return CharTokenizer(chars=''.join(chars))


# ─── Main Training Function ────────────────────────────────────────────

def train_gpt2_tiles():
    print("=" * 70)
    print("  GPT-2 TRAINING ON REAL PLATO TILE DATA")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e6:.0f}MB")

    # Collect corpus
    print("\n[1/6] Collecting training corpus...")
    corpus = collect_corpus()

    # Build tokenizer from corpus
    print("\n[2/6] Building tokenizer...")
    tokenizer = build_corpus_tokenizer(corpus)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Create config
    print("\n[3/6] Creating GPT2Config...")
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_layer=4,
        n_head=4,
        n_embd=256,
        block_size=128,
        dropout=0.1,
        bias=True,
    )
    param_count = config.param_count()
    print(f"  Config: {config.n_layer}L {config.n_head}H {config.n_embd}D")
    print(f"  Block size: {config.block_size}")
    print(f"  Estimated params: {param_count:,}")

    # Create model
    print("\n[4/6] Creating model...")
    model = GPT2Model(config)
    model.to(device)
    actual_params = sum(p.numel() for p in model.parameters())
    print(f"  Actual params: {actual_params:,}")

    # Create dataset
    print("\n[5/6] Creating dataset and training...")
    dataset = TileTextDataset(
        texts=[corpus],
        tokenizer=tokenizer,
        block_size=config.block_size,
        stride=config.block_size // 2,
    )
    print(f"  Dataset samples: {len(dataset)}")

    # Split train/val
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    batch_size = 16
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device == "cuda"),
    )

    # Optimizer
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bias' in name or 'ln_' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    learning_rate = 3e-4
    optimizer = torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': 0.01},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=learning_rate, betas=(0.9, 0.95),
    )

    # Training for 100 steps
    num_steps = 100
    warmup_steps = 10
    grad_accum_steps = 2
    max_grad_norm = 1.0

    def get_lr(it):
        if it < warmup_steps:
            return learning_rate * (it + 1) / (warmup_steps + 1)
        progress = (it - warmup_steps) / max(num_steps - warmup_steps, 1)
        return learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    # Track loss curve
    loss_curve = []
    val_losses = []
    global_step = 0
    epoch = 0
    start_time = time.time()

    model.train()
    optimizer.zero_grad()

    while global_step < num_steps:
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            if global_step >= num_steps:
                break

            x, y = x.to(device), y.to(device)

            logits = model(x)["logits"]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=-1,
            )
            loss = loss / grad_accum_steps
            loss.backward()

            epoch_loss += loss.item() * grad_accum_steps

            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                # Set LR
                for param_group in optimizer.param_groups:
                    param_group['lr'] = get_lr(global_step)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                loss_val = loss.item() * grad_accum_steps
                loss_curve.append(loss_val)
                epoch_steps += 1

                lr_val = optimizer.param_groups[0]['lr']
                print(f"  Step {global_step}/{num_steps}: loss={loss_val:.4f} lr={lr_val:.2e}", end="")
                if device == "cuda":
                    vram = torch.cuda.memory_allocated() / 1e6
                    print(f" vram={vram:.0f}MB", end="")
                print()

                # Validation every 25 steps
                if global_step % 25 == 0:
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
                    print(f"  >>> VALIDATION (step {global_step}): loss={avg_val_loss:.4f} ppl={val_ppl:.2f}")
                    model.train()

        epoch += 1

    training_time = time.time() - start_time

    # Compute final validation
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

    # Track peak VRAM
    peak_vram_mb = 0
    if device == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6

    print(f"\n[6/6] Training complete!")
    print(f"  Training time: {training_time:.1f}s")
    print(f"  Final train loss: {loss_curve[-1]:.4f}" if loss_curve else "")
    print(f"  Final val loss: {final_val_loss:.4f}")
    print(f"  Final perplexity: {final_perplexity:.2f}")
    print(f"  Peak VRAM: {peak_vram_mb:.0f}MB" if device == "cuda" else "  Peak VRAM: N/A (CPU)")

    # ─── Generate Text ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  GENERATING TEXT")
    print("=" * 70)

    prompts = [
        "The PLATO room system",
        "The signal chain architecture",
        "SplineLinear compresses weights",
    ]

    generations = []
    for prompt in prompts:
        print(f"\n  Prompt: \"{prompt}\"")
        prompt_ids = tokenizer.encode(prompt)
        if len(prompt_ids) > config.block_size - 1:
            prompt_ids = prompt_ids[-(config.block_size - 1):]

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            output_ids = input_ids
            for _ in range(500):
                if output_ids.size(1) >= config.block_size:
                    output_ids = output_ids[:, -config.block_size:]
                logits = model(output_ids)["logits"]
                logits = logits[:, -1, :]

                # Temperature sampling
                temperature = 0.8
                top_k = 40
                logits = logits / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                output_ids = torch.cat([output_ids, next_token], dim=1)

        generated_text = tokenizer.decode(output_ids[0].tolist())
        generations.append((prompt, generated_text))
        print(f"  Generated ({len(generated_text)} chars):")
        print(f"  {generated_text[:200]}...")
        print()

    # ─── Save Results ─────────────────────────────────────────────────
    results_path = Path(__file__).parent / "training_results.md"
    with open(results_path, 'w') as f:
        f.write("# GPT-2 Training Results — Real PLATO Tile Data\n\n")
        f.write(f"**Date:** 2026-05-17\n")
        f.write(f"**Model:** GPT-2 ({config.n_layer}L {config.n_head}H {config.n_embd}D)\n")
        f.write(f"**Parameters:** {actual_params:,}\n")
        f.write(f"**Device:** {device}\n")
        f.write(f"**Training steps:** {num_steps}\n\n")

        f.write("## Corpus\n\n")
        f.write(f"- Total corpus size: {len(corpus):,} characters\n")
        f.write(f"- Vocab size (unique chars): {tokenizer.vocab_size}\n")
        f.write(f"- Dataset samples (sliding window): {len(dataset)}\n\n")

        f.write("## Training Loss Curve\n\n")
        f.write("| Step | Train Loss |\n")
        f.write("|------|-----------|\n")
        # Sample at regular intervals
        for i in range(0, len(loss_curve), max(1, len(loss_curve)//10)):
            val_info = ""
            for vs, vl in val_losses:
                if vs == i + 1 or abs(vs - (i + 1)) <= 2:
                    val_info = f" (val: {vl:.4f})"
            f.write(f"| {i} | {loss_curve[i]:.4f}{val_info} |\n")

        f.write(f"\n### Key points\n\n")
        # Show step 0, 25, 50, 75, 100
        for target in [0, 25, 50, 75, 100]:
            idx = min(target, len(loss_curve) - 1)
            if idx >= 0:
                f.write(f"- **Step {target}:** loss = {loss_curve[idx]:.4f}\n")

        f.write(f"\n## Validation Results\n\n")
        for vs, vl in val_losses:
            vppl = math.exp(min(vl, 20.0))
            f.write(f"- Step {vs}: val_loss = {vl:.4f}, perplexity = {vppl:.2f}\n")

        f.write(f"\n## Final Metrics\n\n")
        f.write(f"- **Final training loss:** {loss_curve[-1]:.4f}\n")
        f.write(f"- **Final validation loss:** {final_val_loss:.4f}\n")
        f.write(f"- **Final perplexity:** {final_perplexity:.2f}\n")
        f.write(f"- **Parameter count:** {actual_params:,}\n")
        f.write(f"- **Training time:** {training_time:.1f}s ({training_time/num_steps:.2f}s/step)\n")
        if device == "cuda":
            f.write(f"- **Peak VRAM:** {peak_vram_mb:.0f}MB\n")
        else:
            f.write(f"- **Peak VRAM:** N/A (CPU training)\n")
        f.write(f"- **Device:** {device}\n\n")

        f.write("## Generated Text Samples\n\n")
        for prompt, text in generations:
            f.write(f"### Prompt: \"{prompt}\"\n\n")
            f.write("```\n")
            f.write(text)
            f.write("\n```\n\n")

        # Save training config
        f.write("## Training Configuration\n\n")
        f.write(f"```python\n")
        f.write(f"n_layer={config.n_layer}\n")
        f.write(f"n_head={config.n_head}\n")
        f.write(f"n_embd={config.n_embd}\n")
        f.write(f"block_size={config.block_size}\n")
        f.write(f"batch_size={batch_size}\n")
        f.write(f"learning_rate={learning_rate}\n")
        f.write(f"gradient_accumulation_steps={grad_accum_steps}\n")
        f.write(f"max_grad_norm={max_grad_norm}\n")
        f.write(f"warmup_steps={warmup_steps}\n")
        f.write(f"num_steps={num_steps}\n")
        f.write(f"```\n")

    print(f"\n  Results saved to: {results_path}")
    print(f"  Done!")

    return {
        "config": config,
        "model": model,
        "loss_curve": loss_curve,
        "final_val_loss": final_val_loss,
        "final_perplexity": final_perplexity,
        "param_count": actual_params,
        "training_time": training_time,
        "peak_vram_mb": peak_vram_mb,
        "generations": generations,
    }


if __name__ == "__main__":
    train_gpt2_tiles()
