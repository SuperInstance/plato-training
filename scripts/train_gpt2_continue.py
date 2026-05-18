"""
Continue GPT-2 training — 1100 total steps with text generation every 200 steps.
"""

import sys, os, time, math, json, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from plato_training.gpt2_room import GPT2Config, GPT2Model, CharTokenizer, TileTextDataset
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

WORKSPACE = Path("/home/phoenix/.openclaw/workspace")
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


def collect_corpus() -> str:
    sources = []
    for p in [
        WORKSPACE / "spreader-tool" / "README.md",
        WORKSPACE / "plato-training" / "README.md",
        WORKSPACE / "tensor-spline" / "README.md",
        WORKSPACE / "spectral-conservation" / "README.md",
        WORKSPACE / "signal-chain" / "README.md",
    ]:
        if p.exists():
            sources.append(p.read_text())
            print(f"  + {p.name}: {len(sources[-1])} chars")
    for p in [
        WORKSPACE / "MEMORY.md",
        WORKSPACE / "HEARTBEAT.md",
        WORKSPACE / "AGENTS.md",
        WORKSPACE / "memory" / "2026-05-17.md",
    ]:
        if p.exists():
            sources.append(p.read_text())
            print(f"  + {p.name}: {len(sources[-1])} chars")
    paper = WORKSPACE / "signal-chain" / "papers" / "SIGNAL-CHAIN.md"
    if paper.exists():
        sources.append(paper.read_text())
        print(f"  + SIGNAL-CHAIN.md: {len(sources[-1])} chars")
    gpt2_room = WORKSPACE / "plato-training" / "plato_training" / "gpt2_room.py"
    if gpt2_room.exists():
        sources.append(gpt2_room.read_text())
        print(f"  + gpt2_room.py: {len(sources[-1])} chars")
    corpus = "\n\n".join(sources)
    print(f"\n  TOTAL CORPUS: {len(corpus):,} chars")
    return corpus


def generate_text(model, tokenizer, config, prompt, device, max_new=300, temperature=0.8, top_k=40):
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) > config.block_size - 1:
        prompt_ids = prompt_ids[-(config.block_size - 1):]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        output_ids = input_ids
        for _ in range(max_new):
            if output_ids.size(1) >= config.block_size:
                output_ids = output_ids[:, -config.block_size:]
            logits = model(output_ids)["logits"][:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            output_ids = torch.cat([output_ids, next_token], dim=1)
    return tokenizer.decode(output_ids[0].tolist())


def main():
    print("=" * 70)
    print("  GPT-2 CONTINUATION TRAINING — 1100 STEPS")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("\n[1] Corpus...")
    corpus = collect_corpus()
    chars = sorted(set(corpus))
    tokenizer = CharTokenizer(chars=''.join(chars))
    print(f"  Vocab: {tokenizer.vocab_size}")

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_layer=4, n_head=4, n_embd=256,
        block_size=128, dropout=0.1, bias=True,
    )
    model = GPT2Model(config)
    model.to(device)
    actual_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {actual_params:,}")

    dataset = TileTextDataset(texts=[corpus], tokenizer=tokenizer, block_size=config.block_size, stride=config.block_size // 2)
    val_size = max(1, int(len(dataset) * 0.1))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size], generator=torch.Generator().manual_seed(42))
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device=="cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device=="cuda"))

    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and not any(x in n for x in ['bias', 'ln_', 'norm'])]
    no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and any(x in n for x in ['bias', 'ln_', 'norm'])]
    lr = 3e-4
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': 0.01},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=lr, betas=(0.9, 0.95))

    num_steps = 1100
    warmup = 20
    accum = 2
    max_grad = 1.0

    def get_lr(it):
        if it < warmup:
            return lr * (it + 1) / (warmup + 1)
        return lr * 0.5 * (1.0 + math.cos(math.pi * (it - warmup) / max(num_steps - warmup, 1)))

    loss_curve = []
    val_losses = []
    all_gens = []
    start = time.time()

    model.train()
    optimizer.zero_grad()

    data_iter = iter(train_loader)
    accum_count = 0

    for step in range(num_steps):
        step_loss = 0.0
        for _ in range(accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)
            x, y = x.to(device), y.to(device)
            logits = model(x)["logits"]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
            loss = loss / accum
            loss.backward()
            step_loss += loss.item() * accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
        for pg in optimizer.param_groups:
            pg['lr'] = get_lr(step)
        optimizer.step()
        optimizer.zero_grad()

        loss_curve.append(step_loss)

        if step % 100 == 0 or step == num_steps - 1:
            print(f"  Step {step}/{num_steps}: loss={step_loss:.4f} lr={get_lr(step):.2e}")

        # Validate + generate every 200 steps
        if (step + 1) % 200 == 0:
            model.eval()
            vl = 0.0
            nv = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    vlogits = model(vx)["logits"]
                    vl += F.cross_entropy(vlogits.view(-1, vlogits.size(-1)), vy.view(-1), ignore_index=-1).item()
                    nv += 1
            avg_vl = vl / max(nv, 1)
            ppl = math.exp(min(avg_vl, 20.0))
            val_losses.append((step + 1, avg_vl))
            print(f"  >>> VAL step {step+1}: loss={avg_vl:.4f} ppl={ppl:.2f}")

            prompts = ["The PLATO room system", "SplineLinear compresses", "The fleet of agents", "Tiles are the fundamental", "The signal chain"]
            print(f"\n  === GENERATIONS AT STEP {step+1} ===")
            for p in prompts:
                text = generate_text(model, tokenizer, config, p, device, max_new=300)
                all_gens.append((step + 1, p, text))
                print(f"  [{p}]: {text[:250]}")
            print()

            ckpt = CHECKPOINT_DIR / f"gpt2_step_{step+1}.pt"
            torch.save({'step': step+1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'config': {'n_layer': config.n_layer, 'n_head': config.n_head, 'n_embd': config.n_embd,
                                   'block_size': config.block_size, 'vocab_size': config.vocab_size, 'dropout': config.dropout, 'bias': config.bias},
                        'loss': step_loss, 'val_loss': avg_vl}, ckpt)
            print(f"  Checkpoint: {ckpt}")
            model.train()

    elapsed = time.time() - start

    # Final validation
    model.eval()
    fvl = 0.0
    nv = 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx, vy = vx.to(device), vy.to(device)
            fvl += F.cross_entropy(model(vx)["logits"].view(-1, model(vx)["logits"].size(-1)), vy.view(-1), ignore_index=-1).item()
            nv += 1
    fvl /= max(nv, 1)
    fppl = math.exp(min(fvl, 20.0))

    # Final generations
    final_prompts = ["The PLATO room system", "SplineLinear compresses weights", "The signal chain architecture",
                     "Fleet agents communicate through", "Tiles are the fundamental unit", "Training micro models for"]
    final_gens = []
    print(f"\n  === FINAL GENERATIONS (step {num_steps}) ===")
    for p in final_prompts:
        text = generate_text(model, tokenizer, config, p, device, max_new=400)
        final_gens.append((p, text))
        print(f"  [{p}]: {text[:300]}\n")

    peak_vram = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0

    # Save final checkpoint
    torch.save({'step': num_steps, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'config': {'n_layer': config.n_layer, 'n_head': config.n_head, 'n_embd': config.n_embd,
                           'block_size': config.block_size, 'vocab_size': config.vocab_size, 'dropout': config.dropout, 'bias': config.bias},
                'loss': loss_curve[-1], 'val_loss': fvl}, CHECKPOINT_DIR / "gpt2_final.pt")

    # Write results
    results_path = Path(__file__).parent / "training_results.md"
    with open(results_path, 'w') as f:
        f.write("# GPT-2 Training Results — Real PLATO Tile Data\n\n")
        f.write(f"**Date:** 2026-05-17\n")
        f.write(f"**Model:** GPT-2 ({config.n_layer}L {config.n_head}H {config.n_embd}D)\n")
        f.write(f"**Parameters:** {actual_params:,}\n")
        f.write(f"**Device:** {device}")
        if device == "cuda":
            f.write(f" ({torch.cuda.get_device_name(0)})")
        f.write(f"\n**Total training steps:** {num_steps}\n\n")

        f.write("## Corpus\n\n")
        f.write(f"- Total corpus size: {len(corpus):,} characters\n")
        f.write(f"- Vocab size: {tokenizer.vocab_size}\n")
        f.write(f"- Dataset samples: {len(dataset)}\n\n")

        f.write("## Training Loss Curve\n\n")
        f.write("| Step | Train Loss |\n")
        f.write("|------|-----------|\n")
        for s in [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1099]:
            if s < len(loss_curve):
                f.write(f"| {s} | {loss_curve[s]:.4f} |\n")

        f.write(f"\n## Validation Results\n\n")
        for vs, vl in val_losses:
            f.write(f"- **Step {vs}:** val_loss = {vl:.4f}, perplexity = {math.exp(min(vl, 20.0)):.2f}\n")

        f.write(f"\n## Final Metrics\n\n")
        f.write(f"- **Final training loss:** {loss_curve[-1]:.4f}\n")
        f.write(f"- **Final validation loss:** {fvl:.4f}\n")
        f.write(f"- **Final perplexity:** {fppl:.2f}\n")
        f.write(f"- **Training time:** {elapsed:.1f}s ({elapsed/num_steps:.3f}s/step)\n")
        if device == "cuda":
            f.write(f"- **Peak VRAM:** {peak_vram:.0f}MB\n")
        f.write(f"- **Device:** {device}\n\n")

        f.write("## Text Generation Progress\n\n")
        for step, prompt, text in all_gens:
            f.write(f"### Step {step} — \"{prompt}\"\n\n```\n{text}\n```\n\n")

        f.write("## Final Generated Text\n\n")
        for prompt, text in final_gens:
            f.write(f"### \"{prompt}\"\n\n```\n{text}\n```\n\n")

        f.write("## Training Configuration\n\n```python\n")
        f.write(f"n_layer={config.n_layer}\nn_head={config.n_head}\nn_embd={config.n_embd}\n")
        f.write(f"block_size={config.block_size}\nbatch_size={batch_size}\nlearning_rate={lr}\n")
        f.write(f"gradient_accumulation_steps={accum}\nwarmup_steps={warmup}\nnum_steps={num_steps}\n```\n")

    print(f"\n  Results saved to: {results_path}")
    print(f"  Time: {elapsed:.1f}s | Final loss: {loss_curve[-1]:.4f} | Val: {fvl:.4f} | PPL: {fppl:.2f}")
    print("  DONE!")


if __name__ == "__main__":
    main()
