# PLATO Training Rooms

**LoRA adapters with lifecycle. Predict before you train.**

Every adapter has a deployment story: Active, Superseded, or Retracted. No more orphaned model files.

```bash
pip install plato-training
plato-train init spam-detector --task classification --model gpt2
plato-train train spam-detector --data spam.csv --epochs 3
# → adapter tile created: spam-detector-001 (Active, val_loss=0.21)

plato-train train spam-detector --data spam_v2.csv --epochs 5
# → spam-detector-001 → Superseded. spam-detator-002 → Active.

plato-train list --state active
# → spam-detector-002 (Active, val_loss=0.15)
```

## Why?

Every ML team rediscovers the same problem: adapter file chaos. `model_v2_final_REAL_final.pt`. PLATO gives adapters the same lifecycle as services — they earn their place or get out.

## Architecture

Four rooms, not eight:

| Room | Purpose |
|------|---------|
| `DataRoom` | Tokenize, split, publish dataset tiles |
| `ModelRoom` | Load base models, manage checkpoints |
| `LoRAFactory` | Train LoRA adapters → tiles with lifecycle |
| `EvalRoom` | Evaluate adapters, auto-retract failures |

Every artifact is a tile with lifecycle (Active/Superseded/Retracted) and causal ordering (Lamport clocks).

## Installation

```bash
pip install plato-training
```

## Quick Start

```python
from plato_training import LoRAFactory, AdapterConfig, TrainingConfig

# Create a factory
factory = LoRAFactory("spam-detector")

# Configure LoRA
factory.configure(
    base_model="gpt2",
    adapter_config=AdapterConfig(rank=8, alpha=16),
    training_config=TrainingConfig(epochs=3, learning_rate=2e-4),
)

# Train — produces a tile with lifecycle
tile = factory.train(train_loader, val_loader)
print(tile.summary())
# → [ADAPTER] spam-detector (active, L1) base=gpt2

# Retrain — old tile gets Superseded
tile_v2 = factory.train(train_loader_v2, val_loader)
# → spam-detector-001: Superseded
# → spam-detator-002: Active
```

## What's Different From PEFT/Axolotl/Unsloth?

| Feature | PEFT | Axolotl | Unsloth | **PLATO** |
|---------|------|---------|---------|-----------|
| LoRA training | ✅ | ✅ | ✅ | ✅ |
| Adapter lifecycle | ❌ | ❌ | ❌ | **✅** |
| Simulation-first | ❌ | ❌ | ❌ | **✅** |
| Agent-native discovery | ❌ | ❌ | ❌ | **✅** |
| Fleet sharing | ❌ | ❌ | ❌ | **✅** |

We don't compete on training speed. We compete on **adapter management**.

## Credits

- LoRA implementation based on [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) (Apache 2.0)
- Tile lifecycle from [PLATO Room Server v3](https://github.com/SuperInstance/plato-vessel-core)

## License

MIT
