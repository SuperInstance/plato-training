# Philosophy & Theory

## The Core Idea: Tiles Are Procedures

PLATO tiles aren't just data containers. They're **executable procedures** — like surgical protocols or military field manuals.

A large model (the specialist) discovers an insight, codifies it into a tile with pre-conditions, steps, decision trees, post-conditions, and provenance. A small model (the general practitioner) reads the tile and executes it. The intelligence transfers through the procedure, not the person.

```
Specialist (Claude Opus, GLM-5.1)
  → discovers insight, writes algorithm, encodes reasoning
  → codifies into a TILE (the procedure)
  → includes: code, reasoning, constraints, tests, provenance

Practitioner (Seed-2.0-mini, Hermes-70B)
  → reads the tile, executes the procedure, verifies the result
  → reports edge cases back for refinement
```

**Small models + good tiles beat large models working from scratch.** After enough refinement cycles, the tile has absorbed the specialist's intelligence. This is why medical protocols get better — not because surgeons get smarter, but because the procedures accumulate.

### The Capability Ladder

| Tier | Role | Models | What They Do |
|------|------|--------|-------------|
| 3 | Elite Specialist | Claude Opus, GPT-4 | Create NEW procedures from scratch. Novel synthesis. |
| 2 | Senior Practitioner | GLM-5.1, DeepSeek Reasoner | Refine procedures, adapt to new contexts, write tiles. |
| 1 | General Practitioner | Seed-2.0-mini, Hermes, Qwen | Execute procedures from tiles. Fast, cheap, disciplined. |

The key economic insight: **Tier 1 models can do Tier 2 work IF the tile is good enough.** A general surgeon executing a perfect Mayo Clinic protocol produces better outcomes than a mediocre specialist winging it.

### The Accumulation Effect

Every cycle makes tiles better:

```
Cycle 1:  Specialist creates tile v1 (good but rough)
Cycle 2:  Practitioner executes, reports edge cases
Cycle 3:  Specialist refines → tile v2 (better)
Cycle 5:  Another practitioner finds more edge cases
Cycle 10: Tile v5 embodies 10 iterations of accumulated intelligence
Cycle 100: Seed-2.0-mini + tile v100 > GLM-5.1 working from scratch
```

## The Three-Structure Theorem

A mathematical result from this research: `dim(SE(2)) = 3` is why everything converges on 3. The Three-Structure Theorem proves that the special Euclidean group in 2D has exactly 3 degrees of freedom (rotation + 2D translation), which explains why:

- SplineLinear's 3-knot structure achieves optimal compression
- Micro models with 3-layer depth hit the accuracy ceiling
- Collective inference stabilizes at 3-agent clusters

This isn't numerology — it's Lie group theory applied to neural architecture.

---

> *"The good physician treats the disease. The great physician treats the patient who has the disease."* — William Osler

The good model answers the question. The great model writes the tile that lets any model answer it.
