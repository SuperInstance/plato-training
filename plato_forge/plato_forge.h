/*
 * plato_forge.h — PLATO Forge: C/CUDA bridge for continuous training daemon
 *
 * The Forge sits between PyTorch and the PLATO tile store, providing:
 *
 *   1. Eisenstein Weight Snap — quantize float weights to Eisenstein lattice
 *      points for natural compression. 6 hexagonal sectors → 6-way symmetry
 *      in weight space. Snap error = compression loss. Adjustable radius.
 *
 *   2. BMA Training Signal — Berlekamp-Massey on the loss sequence detects
 *      when training has plateaued (LFSR fits the loss pattern → predictable
 *      → nothing new being learned). Triggers early stop or learning rate decay.
 *
 *   3. Deadband Gradient Throttle — skip gradient updates when weight changes
 *      fall within the deadband. Saves GPU cycles, prevents oscillation.
 *      Uses HPDF dither for stochastic resonance (occasionally accept small
 *      updates to escape local minima).
 *
 *   4. Weight Pipeline — double-buffered weight memory so training and
 *      inference never block each other. Training writes to buffer A while
 *      inference reads from buffer B, then swap.
 *
 *   5. PLATO Tile Export — serialize trained weights + metrics as a PLATO
 *      tile that can be loaded by Python (PyTorch) or C (llama.cpp).
 *
 * Target: RTX 4050 Laptop GPU (sm_89, 6GB VRAM, 20 SMs)
 *
 * Novel technique: "Old language, new chip"
 *   - BMA (1968 algorithm from coding theory) → training convergence detector
 *   - Eisenstein lattice (1837 number theory) → weight quantization scheme
 *   - Deadband (control theory, 1940s) → gradient update throttle
 *   - HPDF dither (signal processing, 1940s) → stochastic gradient noise
 */

#ifndef PLATO_FORGE_H
#define PLATO_FORGE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Configuration ──────────────────────────────────────────────── */

typedef struct {
    /* Eisenstein snap parameters */
    float snap_radius;          /* Lattice snap radius (0.0 = no snap, 1.0 = full) */
    int snap_interval;          /* Apply snap every N training steps */
    int snap_warmup;            /* Don't snap for first N steps */

    /* BMA convergence detection */
    int bma_window;             /* Loss sequence length for BMA */
    float bma_early_stop_ratio; /* Stop if LFSR length < ratio * window */
    float bma_lr_decay;         /* Multiply LR by this when BMA triggers */

    /* Deadband gradient throttle */
    float deadband_threshold;   /* Skip update if |delta| < threshold */
    float hpdf_dither_strength; /* Probability of accepting below-threshold update */
    int throttle_min_updates;   /* Minimum updates per step (don't over-throttle) */

    /* Weight pipeline */
    int double_buffer;          /* Enable double-buffered weight swap */
    int pipeline_swap_interval; /* Swap buffers every N steps */

    /* Memory */
    int max_weight_bytes;       /* Maximum weight tensor size in bytes */
    int max_tiles;              /* Maximum PLATO tiles in ring buffer */

    /* GPU */
    int target_sm;              /* Target SM architecture (89 for RTX 4050) */
    int preferred_block_size;   /* CUDA block size (0 = auto) */
} ForgeConfig;

/* Default config for RTX 4050 */
static inline ForgeConfig forge_default_config(void) {
    ForgeConfig c = {0};
    c.snap_radius = 0.5f;
    c.snap_interval = 100;
    c.snap_warmup = 500;
    c.bma_window = 64;
    c.bma_early_stop_ratio = 0.5f;
    c.bma_lr_decay = 0.5f;
    c.deadband_threshold = 1e-4f;
    c.hpdf_dither_strength = 0.05f;
    c.throttle_min_updates = 32;
    c.double_buffer = 1;
    c.pipeline_swap_interval = 10;
    c.max_weight_bytes = 64 * 1024 * 1024;  /* 64 MB */
    c.max_tiles = 1000;
    c.target_sm = 89;
    c.preferred_block_size = 256;
    return c;
}

/* ── Data Types ──────────────────────────────────────────────────── */

/* Snap result for a single weight tensor */
typedef struct {
    int n_params;               /* Number of parameters snapped */
    int n_snapped;              /* Number actually changed */
    float mean_snap_error;      /* Average distance moved */
    float max_snap_error;       /* Maximum distance moved */
    float compression_ratio;    /* Effective compression from lattice quantization */
    int chambers[6];            /* Count of weights in each Weyl chamber */
} ForgeSnapStats;

/* BMA convergence result */
typedef struct {
    int lfsr_length;            /* Shortest LFSR that fits the loss sequence */
    float predictability;       /* lfsr_length / window_length (1.0 = perfect fit) */
    int should_stop;            /* 1 if training should stop */
    float lr_multiplier;        /* Suggested LR multiplier */
} ForgeBMAResult;

/* Deadband throttle result */
typedef struct {
    int total_gradients;        /* Total gradients evaluated */
    int passed_threshold;       /* Gradients above deadband */
    int passed_dither;          /* Gradients below threshold but accepted by dither */
    int skipped;                /* Gradients skipped entirely */
    float throttle_ratio;       /* Fraction of updates actually applied */
    float dither_accepted_ratio;/* Fraction of dither trials accepted */
} ForgeThrottleStats;

/* A PLATO training tile in C */
typedef struct {
    char tile_id[64];           /* Unique tile identifier */
    char room[64];              /* Room name */
    int lamport;                /* Lamport clock value */
    int n_layers;               /* Number of model layers */
    int n_params;               /* Total parameters */
    int compressed_bytes;       /* Size of compressed weights */

    /* Metrics */
    float train_loss;
    float val_loss;
    float train_accuracy;
    float val_accuracy;
    int epochs;
    float training_time_seconds;

    /* Forge-specific metrics */
    float snap_compression_ratio;
    float throttle_ratio;
    float bma_predictability;
    int total_steps;
} ForgeTile;

/* Full forge state */
typedef struct ForgeState ForgeState;  /* Opaque */

/* ── Core API ───────────────────────────────────────────────────── */

/**
 * forge_init — Initialize the PLATO Forge daemon.
 *
 * Allocates GPU memory, creates CUDA streams, initializes state.
 * Returns opaque state pointer, or NULL on failure.
 */
ForgeState* forge_init(ForgeConfig config);

/**
 * forge_destroy — Clean up forge state and free GPU memory.
 */
void forge_destroy(ForgeState* state);

/**
 * forge_register_weights — Register a weight tensor with the forge.
 *
 * The forge tracks this tensor for snap/throttle/pipeline operations.
 * weights: device pointer (GPU memory)
 * n: number of float values
 * name: human-readable name (e.g., "transformer.h.0.attn.qkv.weight")
 * Returns: handle ID for this weight tensor, or -1 on error.
 */
int forge_register_weights(ForgeState* state, float* weights, int n, const char* name);

/**
 * forge_step — Process one training step through the forge.
 *
 * Applies deadband throttle to gradients, optionally snaps weights,
 * checks BMA convergence on loss sequence, and manages pipeline swap.
 *
 * gradients: device pointer to gradient tensor (n floats)
 * n: number of gradient values
 * loss: current training loss
 * step: global step number
 * lr: current learning rate (may be modified by BMA)
 *
 * Returns: updated learning rate (may be decayed by BMA)
 */
float forge_step(
    ForgeState* state,
    float* gradients,
    int n,
    float loss,
    int step,
    float lr
);

/**
 * forge_swap_buffers — Swap double-buffered weights.
 *
 * Training buffer becomes inference buffer and vice versa.
 * Call this at a safe point (between batches).
 */
void forge_swap_buffers(ForgeState* state);

/**
 * forge_export_tile — Export current weights as a PLATO tile.
 *
 * Compresses weights via Eisenstein snap + run-length encoding,
 * serializes to tile format, and copies to host memory.
 *
 * out_bytes: pre-allocated host buffer for tile data
 * out_capacity: size of out_bytes buffer
 * Returns: actual bytes written, or -1 on error.
 */
int forge_export_tile(
    ForgeState* state,
    void* out_bytes,
    int out_capacity,
    ForgeTile* out_tile
);

/**
 * forge_load_tile — Load weights from a PLATO tile.
 *
 * Decompresses tile data and loads weights into GPU memory.
 * tile_bytes: host pointer to tile data
 * tile_bytes_len: length of tile data
 * tile: tile metadata
 * Returns: 0 on success, -1 on error.
 */
int forge_load_tile(
    ForgeState* state,
    const void* tile_bytes,
    int tile_bytes_len,
    const ForgeTile* tile
);

/**
 * forge_get_stats — Get current forge statistics.
 */
void forge_get_stats(
    ForgeState* state,
    ForgeSnapStats* snap,
    ForgeBMAResult* bma,
    ForgeThrottleStats* throttle
);

/**
 * forge_should_train — Check if forge recommends continuing training.
 *
 * Returns 1 if training should continue, 0 if converged/idle.
 */
int forge_should_train(ForgeState* state);

/**
 * forge_idle_train — Run idle-time self-training on GPU.
 *
 * Called when the main training loop is paused. Uses accumulated
 * experiences to refine the micro-models (pre/post filter).
 *
 * Returns: number of steps actually trained (0 if nothing to do).
 */
int forge_idle_train(ForgeState* state, int max_steps);

#ifdef __cplusplus
}
#endif

#endif /* PLATO_FORGE_H */
