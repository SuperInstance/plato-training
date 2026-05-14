# PLATO Training Rooms — Architecture v2

## Three Layers

```
Layer 3: Tensor-Spline Platform (novel — our paradigm)
         ↓ builds on
Layer 2: TF Training Rooms    PyTorch Training Rooms
         ↓ both use                ↓ both use
Layer 1: PLATO Room Protocol (tiles, lifecycle, throttle, simulation-first)
```

Build Layer 2 first. Layer 3 iterates last.

---

## Layer 1: The Room Protocol (already exists)

Every training run is a PLATO room. Every artifact is a tile.

```
TrainingRoom {
  room_type: "pytorch" | "tensorflow" | "tensor-spline"
  base_model: "gpt2-small" | "bert-base" | "custom"
  status: "idle" | "training" | "evaluating" | "complete"
  throttle_state: "full" | "reduced" | "minimal" | "paused"
  
  tiles: [
    DatasetTile (Active)
    ConfigTile (Active)
    CheckpointTile (Active → Superseded by better one)
    MetricsTile (Active)
    AdapterTile (Active → Superseded by retrain)
  ]
}
```

## Layer 2: Engine Rooms

### The Throttle Mechanism

The key insight: **training is a background citizen of the fleet, not the main event.**

```
                    ┌─────────────────┐
                    │  PLATO Server    │
                    │  /rooms/active   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Throttle Daemon │
                    │                  │
                    │  Query: how many │
                    │  rooms are active│
                    │  right now?      │
                    │                  │
                    │  0-2: FULL GPU   │
                    │  3-5: REDUCED    │
                    │  6+:  MINIMAL    │
                    │  10+: PAUSED     │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐ ┌─────▼─────┐ ┌──────▼──────┐
        │ PyTorch    │ │ TF Room   │ │ Tensor-     │
        │ Room       │ │           │ │ Spline Room │
        │            │ │           │ │             │
        │ batch_size │ │ num_steps │ │ control_pts │
        │ workers    │ │           │ │             │
        │ GPU_mem    │ │           │ │             │
        └────────────┘ └───────────┘ └─────────────┘
```

**How it works:**
1. Training loop calls `throttle.check()` every N batches
2. `throttle.check()` queries PLATO `/stats` or reads a local fleet-load metric
3. Returns a `ThrottleState` with recommended batch_size, num_workers, GPU_fraction
4. Training loop adjusts dynamically (PyTorch: DataLoader workers + batch size; TF: tf.data prefetch + steps)

**Fleet load metric (simple, no PLATO dependency):**
```python
def fleet_load():
    """0.0 = idle, 1.0 = saturated"""
    load = os.getloadavg()[0] / os.cpu_count()
    gpu_mem = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    return max(load, gpu_mem)
```

### PyTorch Training Room

```python
class PyTorchRoom:
    """An agent walks in with data, walks out with a trained adapter."""
    
    def __init__(self, room_name, store_dir=".plato-training"):
        self.name = room_name
        self.store = LocalTileStore(store_dir)
        self.clock = LamportClock()
        self.throttle = TrainingThrottle()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def train(self, model, dataset, config):
        """
        The agent-facing API. Three lines:
        
        room = PyTorchRoom("spam-detector")
        tile = room.train(model, dataset, config)
        # tile has lifecycle, weights, metrics
        """
        # 1. Register dataset as tile
        data_tile = self._register_dataset(dataset)
        
        # 2. Inject LoRA
        injection_map = inject_lora(model, config.adapter)
        
        # 3. Training loop with throttle
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)
        
        for epoch in range(config.epochs):
            # CHECK THROTTLE every epoch
            state = self.throttle.check()
            if state == "paused":
                self._wait_for_idle()
                continue
            
            # Adjust batch size dynamically
            effective_batch = config.batch_size * state.batch_multiplier
            loader = DataLoader(dataset, batch_size=effective_batch, 
                              num_workers=state.num_workers)
            
            for batch in loader:
                loss = train_step(model, batch, optimizer)
                
                # Save checkpoint periodically
                if step % config.checkpoint_interval == 0:
                    self._save_checkpoint(model, step, loss)
            
            scheduler.step()
        
        # 4. Save final adapter as tile
        adapter_tile = self._save_adapter(model, config, injection_map)
        
        # 5. Evaluate and lifecycle
        if self._is_better(adapter_tile):
            self._supersede_previous(adapter_tile)
        
        return adapter_tile
```

### TF Training Room

```python
class TensorFlowRoom:
    """Same pattern, TF engine."""
    
    def train(self, model, dataset, config):
        """
        Uses tf.keras.Model.fit under the hood.
        Throttle adjusts: steps_per_epoch, prefetch_buffer_size, num_parallel_calls
        """
        # 1. Register dataset
        data_tile = self._register_dataset(dataset)
        
        # 2. Build tf.data pipeline with throttle
        tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)
        
        # 3. Callback-based throttle (checks every epoch)
        throttle_cb = ThrottleCallback(self.throttle)
        checkpoint_cb = PlatoCheckpointCallback(self.store, self.clock)
        
        # 4. Train
        model.fit(
            tf_dataset.batch(config.batch_size),
            epochs=config.epochs,
            callbacks=[throttle_cb, checkpoint_cb],
        )
        
        # 5. Save as tile
        adapter_tile = self._save_weights(model, config)
        
        return adapter_tile
```

### The Agent's Perspective

From an agent's point of view, training is just another room:

```
Agent: "I need a spam classifier"
  → Creates PyTorchRoom("spam-detector")
  → room.train(model, data, config)
  → Gets back a TrainingTile
  → tile.state = Active
  → tile.metrics.final_loss = 0.023
  → tile.content_hash = "a3f8..."

Agent: "That wasn't good enough, retrain with more data"
  → room.train(model, more_data, config)
  → New tile created
  → Old tile automatically superseded
  → Old tile.history() shows: "Superseded by spam-detector-003"

Agent: "Actually the first one was better"
  → room.retract(new_tile, reason="worse val loss")
  → Reactivate old tile
  → Lifecycle events record the flip
```

---

## Layer 3: Tensor-Spline Platform (Design)

### What Is It?

Standard neural networks: W is a dense matrix of independent floats.
Tensor-Spline: W is **parameterized by control points on an Eisenstein lattice**.

```
Standard:     W[i][j] = learned_float
Tensor-Spline: W[i][j] = interpolate(control_points, i, j)
```

### Why This Changes Everything

1. **Compression**: 512×512 matrix = 262K params. With 64 Eisenstein control points = 128K params (2:1). More aggressive: 16 control points = 4K params (64:1).

2. **Built-in regularization**: The spline IS smooth. No weight decay needed. The lattice structure prevents pathological weights.

3. **Constraint-native**: Control points live in Eisenstein space. Snap to lattice → quantize. Drift → detect. Supersede → replace control points.

4. **Simulation-first**: Before training, predict which control points will move. After training, confirm. Only unexpected movements generate new tiles.

5. **Fleet-native**: Two agents can contribute control points independently. Merge = spline composition, not weight averaging.

### The Math

```
Eisenstein lattice: ω = e^(2πi/3), lattice points = a + bω

Control points: C_k at lattice positions L_k
Weight W[i][j] = Σ_k C_k · B(||pos(i,j) - L_k||) / Σ_k B(||pos(i,j) - L_k||)

where B is a basis function (B-spline, Gaussian, or our Eisenstein snap kernel)
```

This means:
- Forward pass: standard matrix multiply (weights are materialized from control points)
- Backward pass: gradients flow to control points, not individual weights
- Update: move control points, re-materialize weights
- The spline is the compression; the lattice is the constraint

### The Spline Room

```python
class TensorSplineRoom(PyTorchRoom):
    """
    Training room where weights live on an Eisenstein lattice.
    
    Inherits throttle, lifecycle, tiles from PyTorchRoom.
    Overrides: how weights are parameterized, saved, and loaded.
    """
    
    def inject_spline(self, model, lattice_config):
        """
        Replace nn.Linear layers with SplineLinear layers.
        
        SplineLinear stores control points, materializes weights on forward.
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                spline = SplineLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    n_control_points=lattice_config.n_points,
                    basis=lattice_config.basis,  # "eisenstein" | "bspline" | "gaussian"
                    rank=lattice_config.rank,
                )
                # Initialize from pretrained weights
                spline.fit_control_points(module.weight.data)
                replace_module(model, name, spline)
    
    def train(self, model, dataset, config):
        """
        Same throttle-aware training loop.
        Different: only control point gradients flow.
        """
        self.inject_spline(model, config.lattice)
        # ... standard training loop with throttle ...
        
        # Save: only control points, not full weights
        # A 512×512 layer with 16 control points = 16 × 2 (real+imag) = 32 floats
        # vs 262K floats for the dense layer
        # That's 8000:1 compression
```

### How It Connects to Everything

| Fleet Concept | Tensor-Spline Expression |
|---|---|
| Eisenstein integers | Control points ON the lattice |
| Constraint theory | Weights constrained by lattice geometry |
| Penrose memory | Non-repeating control point layout |
| Simulation-first | Predict which control points move |
| Tile lifecycle | Control point snapshots as tiles |
| Lamport clocks | Causal ordering of control point updates |
| Bloom filters | Quick check: "did this control point change?" |
| PLATO rooms | Each spline layer IS a room of control points |
| Zero drift | Snap gradients to lattice before applying |
| Holonomy | Check: does a cycle of control point updates return to start? |

---

## Implementation Priority

### Phase 0: PyTorch Room with Throttle (do first)
- `PyTorchRoom` class
- `TrainingThrottle` (fleet-load-aware)
- `LocalTileStore` (already built)
- `LoRALayer` with save/load (already built)
- CLI: `plato-train pytorch --model gpt2 --data spam.csv --throttle`

### Phase 1: TF Room
- `TensorFlowRoom` class
- `ThrottleCallback` for keras
- Same tile lifecycle, same store

### Phase 2: Tensor-Spline (novel)
- `SplineLinear` layer
- `EisensteinLattice` control point layout
- `TensorSplineRoom`
- Paper: "Lattice-Parameterized Neural Networks"
- HN angle: "Show HN: We replaced weight matrices with Eisenstein lattice splines"

### Phase 3: Agentic Training
- Any agent can say "train me a model on X"
- The room picks the engine (PyTorch/TF/Spline) based on problem type
- Training throttles based on fleet load
- Results are tiles with lifecycle

---

## The Throttle is the Real Innovation Here

Not the training. Not even the tensor-spline. The **throttle**.

Every ML framework assumes it owns the GPU. In our fleet, the GPU is shared. Training is a background task that yields to foreground agent work. This is genuinely different from anything in PyTorch/TF ecosystems.

The throttle enables:
- **Agent-driven training**: Agent submits a training job, it runs when fleet is idle
- **Priority inversion**: Urgent model gets full GPU, background training pauses
- **Fleet-aware scheduling**: If Oracle1 is running CUDA benchmarks on the same GPU, my training backs off
- **Cost optimization**: Don't pay for full GPU time when you only need 30% of it

This is the fleet paradigm applied to ML training. The room is the abstraction. The throttle is the mechanism. The lifecycle is the memory.
