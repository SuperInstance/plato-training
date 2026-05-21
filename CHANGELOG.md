# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2025-05-20

### Added
- R5 release: full Eisenstein encoder with contrastive training
- SplineLinearHD for high-dimensional adaptive compression
- Swarm rooms with CRDT convergence and Eisenstein lattice snapping
- Collective loop with transfer-entropy coordination topology
- GPT-2 fleet trainer with multi-task learning and tile export
- Commit predictor with temporal features
- ONNX export pipeline for Eisenstein encoder and spline models
- Intelligence room with pre/post filters, self-trainer, and knowledge tiles
- Device router with CUDA/DirectML/CPU fallback
- Data pipeline with CSV/JSONL loaders, feature store, and splitters
- Plato forge with BMA optimization, deadband, and Eisenstein snap
- Semantic store and matcher (keyword + model2vec)
- Agent field with chirality, coherence, and coupling dynamics
- LoRA adapters and factory
- Tutor judge with bit-vector similarity and pattern matching
- CLI entry point (`plato-train`) for GPT-2 fleet and collective operations
- 487+ tests covering all modules
- `py.typed` marker for PEP 561 compliance
- `__all__` exports on all public modules
- `__repr__` on all public classes
- Full type annotations on public API
- CONTRIBUTING.md guide

### Changed
- Improved thread safety across shared mutable state
- Input validation hardened across all public APIs

### Note
- 2 ONNX export tests for Eisenstein encoder may fail due to upstream
  PyTorch/ONNX SSA compatibility (tracked separately)
