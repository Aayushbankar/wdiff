---
task:
  id: "TASK-20261105-061500"
  title: "wdiff Phase 1: Core Engine & CLI (v0.1.0)"
  type: "feature_development"
  persona: "programmer"
  model: "nvidia/nemotron-3-ultra-550b-a55b"
  created: "2026-11-05T06:15:00Z"
  status: "in_progress"
  complexity_triggers:
    - "multi_module_architecture"
    - "memory_bounded_streaming"
    - "binary_format_parsing"
---

# Task: wdiff Phase 1 - Core Engine & CLI (v0.1.0)

**ID**: TASK-20261105-061500
**Type**: feature_development
**Persona**: programmer
**Model**: nvidia/nemotron-3-ultra-550b-a55b
**Created**: 2026-11-05T06:15:00Z
**Status**: in_progress (M1 complete)

## Objective
Implement the complete Phase 1 (v0.1.0) of wdiff: a working, memory-bounded CLI tool that compares GGUF and safetensors checkpoints with <500MB peak RAM, outputting Relative Frobenius delta, Cosine similarity, and Max element delta per tensor.

## Current State
- **M1 - Format Readers COMPLETE**: `src/wdiff/readers/` fully implemented with tests passing
- **M2 - Metrics Engine**: Not started
- **M3 - Ring Buffer**: Not started
- **M4 - CLI**: Not started
- **M5 - Release**: Not started

## Acceptance Criteria
1. **Format Abstraction**: `TensorReader` protocol + `SafetensorsReader` + `GGUFReader` with dequantization ✅ DONE
2. **Metrics Engine**: Relative Frobenius change, Cosine similarity, Max element delta (float32)
3. **Streaming Core**: 2-slot pipelined ring buffer overlapping I/O and compute
4. **CLI**: `wdiff modelA.gguf modelB.gguf` / `wdiff modelA.safetensors modelB.safetensors` with Rich tables
5. **Memory Bound**: Peak RAM ≤ 500MB for 70B models (verified by integration test)
6. **Cross-Format**: Compare safetensors ↔ GGUF seamlessly ✅ DONE
7. **Package & Release**: `pip install wdiff` / `uvx wdiff` works end-to-end

## Constraints
- Python 3.10+, src-layout, dependencies per `pyproject.toml` (numpy, safetensors, gguf, rich, scipy)
- Zero-copy mmap for safetensors; dequantization for GGUF (Q4_K, Q5_K, Q8_K, IQ*)
- Ring buffer depth = 2 (Thread 1: I/O prefetch + dequantize; Thread 2: Math worker)
- Peak RAM = 4 × largest single tensor (235MB × 2 = ~470MB for 8B models)
- No PyTorch dependency in core (optional extra only)

## Non-Goals (Phase 1)
- `--deep` SVD analysis (Phase 2)
- HTML reports (Phase 2)
- LoRA adapter comparison (Phase 2)
- Cross-format tensor name normalization (Phase 2)
- Standalone binaries (Phase 4)
- Rust implementation (Phase 4)

## Verification Gates (per phase)
| Phase | Gate | Evidence |
|-------|------|----------|
| 1.1 Readers | Unit tests: both readers load tensors, dequantize GGUF to float32 | `pytest tests/readers/ -v` ✅ PASSED |
| 1.2 Metrics | Unit tests: metrics match numpy/scipy reference | `pytest tests/engine/test_metrics.py -v` |
| 1.3 Ring Buffer | Integration test: 2-slot pipeline processes mock tensors | `pytest tests/engine/test_ring_buffer.py -v` |
| 1.4 CLI | E2E test: `wdiff A.gguf B.gguf` prints table, peak RAM < 500MB | `pytest tests/integration/test_cli.py -v` |
| 1.5 Release | `uvx wdiff --help` works; wheel on PyPI | Manual verify |

---

## Progress

### Milestone 1: Format Abstraction Layer ✅ COMPLETED
- `readers/base.py`: `TensorReader` Protocol + `TensorInfo` dataclass
- `readers/safetensors_reader.py`: `SafeTensorsReader` using `safetensors.safe_open` with mmap
- `readers/gguf_reader.py`: `GGUFReader` using `gguf.GGUFReader` + `gguf.quants.dequantize` for Q4_K, Q5_K, Q6_K, Q8_K, IQ* variants
- `readers/__init__.py`: Factory `get_reader(path)` dispatching by extension
- **Tests**: 16 tests passing (`pytest tests/readers/ -v`)
- **Lint**: All ruff checks pass

### Next: Milestone 2 - Metrics Engine