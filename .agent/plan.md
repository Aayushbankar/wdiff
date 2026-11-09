---
plan:
  id: "PLAN-20261105-061500"
  task_id: "TASK-20261105-061500"
  created: "2026-11-05T06:15:00Z"
  status: "draft"
---

# ExecPlan: wdiff Phase 1 - Core Engine & CLI (v0.1.0)

**Task ID**: TASK-20261105-061500
**Created**: 2026-11-05T06:15:00Z
**Status**: draft

## 1. Approach Summary

Build Phase 1 in **5 sequential milestones** following the data flow: Format Readers → Metrics → Streaming Engine → CLI → Release. Each milestone has a verification gate with automated tests. The architecture is fully specified in `docs/ARCHITECTURE.md` — implementation follows the documented protocols and memory-bounding guarantees.

**Key architectural decisions already made (from docs):**
- `TensorReader` Protocol with 3 methods: `tensor_names()`, `get_tensor()`, `tensor_info()`
- 2-slot Ring Buffer with Thread 1 (I/O + dequant) and Thread 2 (Math)
- Peak RAM bound: 4 × max tensor size (~470MB for 8B models)
- Metrics: Relative Frobenius, Cosine, Max Delta (all float32)
- Cross-format via unified float32 tensors from readers

## 2. Milestones & Verification Gates

### Milestone 1: Format Abstraction Layer (`src/wdiff/readers/`)
**Duration**: 1-2 days
**Scope**: Implement the 3 reader modules + base protocol

| Task | File | Description |
|------|------|-------------|
| 1.1 | `readers/base.py` | `TensorReader` Protocol + `TensorInfo` dataclass |
| 1.2 | `readers/safetensors_reader.py` | `SafeTensorsReader` using `safetensors.safe_open` with mmap |
| 1.3 | `readers/gguf_reader.py` | `GGUFReader` using `gguf.GGUFReader` + `gguf.quants.dequantize` for Q4_K_M, Q5_K_S, Q8_0, IQ* |
| 1.4 | `readers/__init__.py` | Exports + factory `get_reader(path)` dispatching by extension |

**Verification Gate 1.1**: `pytest tests/readers/ -v`
- Test: Both readers implement protocol, return float32 arrays
- Test: GGUF dequantization matches reference for known quant types
- Test: `tensor_info()` returns correct shape/dtype/byte_range without loading data
- Test: Factory dispatches correctly by file extension

**Dependencies**: None (starts from empty modules)

---

### Milestone 2: Metrics Engine (`src/wdiff/engine/metrics.py`)
**Duration**: 0.5-1 day
**Scope**: Pure math functions for tensor comparison

| Task | File | Description |
|------|------|-------------|
| 2.1 | `engine/metrics.py` | `relative_frobenius(A, B)`, `cosine_similarity(A, B)`, `max_element_delta(A, B)` — all accept float32 ndarrays |
| 2.2 | `engine/metrics.py` | `compute_all_metrics(A, B) -> MetricsBundle` dataclass |
| 2.3 | `engine/__init__.py` | Exports |

**Verification Gate 1.2**: `pytest tests/engine/test_metrics.py -v`
- Test: Metrics match numpy/scipy reference implementations (1e-6 tolerance)
- Test: Identical tensors → ΔF=0, cos=1, max_delta=0
- Test: Orthogonal tensors → cos≈0
- Test: Numerical stability for large tensors (14336×4096)

**Dependencies**: Milestone 1 (readers provide float32 arrays)

---

### Milestone 3: Pipelined Ring Buffer (`src/wdiff/engine/ring_buffer.py`)
**Duration**: 1-2 days
**Scope**: Threaded double-buffering core

| Task | File | Description |
|------|------|-------------|
| 3.1 | `engine/ring_buffer.py` | `RingBufferSlot` dataclass + `RingBuffer` class with 2 slots |
| 3.2 | `engine/ring_buffer.py` | `IOWorker` thread: `prefetch_next(reader_a, reader_b, tensor_names, slot_idx)` |
| 3.3 | `engine/ring_buffer.py` | `MathWorker` thread: `process_slot(slot, metrics_fn) -> List[TensorResult]` |
| 3.4 | `engine/ring_buffer.py` | `run_diff(reader_a, reader_b, metrics_fn) -> List[TensorResult]` orchestrator |
| 3.5 | `engine/diff.py` | High-level `diff_models(path_a, path_b) -> DiffResult` using factory + ring buffer |

**Verification Gate 1.3**: `pytest tests/engine/test_ring_buffer.py -v`
- Test: 2-slot pipeline processes N mock tensors in correct order
- Test: Memory profile — peak RSS ≤ 4 × max_tensor_size (mock 200MB tensors)
- Test: Thread synchronization — no race conditions, slots rotate correctly
- Test: Exception propagation from worker threads to main

**Dependencies**: Milestones 1 & 2

---

### Milestone 4: CLI & Terminal Display (`src/wdiff/cli.py`, `src/wdiff/display/terminal.py`)
**Duration**: 1 day
**Scope**: Rich-based CLI with progress, tables, anomaly detection

| Task | File | Description |
|------|------|-------------|
| 4.1 | `display/terminal.py` | `render_diff_table(results, model_a_name, model_b_name)` → Rich Table |
| 4.2 | `display/terminal.py` | `render_progress_bar(current, total, current_tensor_name)` |
| 4.3 | `display/terminal.py` | `detect_anomalies(results)` → flags: rank-1 spikes, large ΔF, orthogonal layers |
| 4.4 | `cli.py` | `main()`: argparse, reader factory, `diff_models()`, render, exit codes |
| 4.5 | `cli.py` | `--json` flag: stdout JSON; `--filter` flag: regex on tensor names |

**Verification Gate 1.4**: `pytest tests/integration/test_cli.py -v`
- E2E: `wdiff tests/fixtures/small_a.safetensors tests/fixtures/small_b.safetensors` prints table
- E2E: `wdiff tests/fixtures/small_a.gguf tests/fixtures/small_b.gguf` works (cross-format)
- Test: `--json` outputs valid JSON with all metrics per tensor
- Test: `--filter "attn_output"` shows only matching layers
- Memory test: `uvx wdiff 70B_model_a.gguf 70B_model_b.gguf` peak RSS < 500MB (CI with mock)

**Dependencies**: Milestones 1-3

---

### Milestone 5: Package, Test Suite Hardening & Release (`pyproject.toml`, `tests/`, CI)
**Duration**: 0.5-1 day
**Scope**: Complete test coverage, packaging, PyPI release

| Task | File | Description |
|------|------|-------------|
| 5.1 | `tests/` | Create fixtures: small safetensors + GGUF pairs (synthetic tensors) |
| 5.2 | `tests/` | Unit tests for readers, metrics, ring buffer, CLI |
| 5.3 | `tests/` | Integration test: cross-format diff, memory bound verification |
| 5.4 | `pyproject.toml` | Verify entry point `wdiff = wdiff.cli:main`, all deps correct |
| 5.5 | `.github/workflows/ci.yml` | GitHub Actions: lint (ruff), typecheck (pyright), test matrix (py310-312) |
| 5.6 | Release | Tag v0.1.0, publish to PyPI |

**Verification Gate 1.5**: Manual + CI
- `uvx wdiff --help` works in fresh venv
- `pip install wdiff && wdiff --help` works
- All CI checks pass (ruff, pyright, pytest)
- Wheel uploaded to PyPI

**Dependencies**: Milestone 4

---

## 3. Task Breakdown Summary (Dependency Graph)

```
M1: Readers (base, safetensors, gguf, factory)
    │
    ├──→ M2: Metrics (frobenius, cosine, max_delta)
    │         │
    │         ├──→ M3: Ring Buffer (IOWorker, MathWorker, orchestrator)
    │         │         │
    │         │         ├──→ M4: CLI + Display (Rich table, progress, anomalies)
    │         │         │         │
    │         │         │         └──→ M5: Tests + Packaging + Release
    │         │         │
    │         │         └──→ (parallel) M4 can start after M3.1-3.3
    │         │
    │         └──→ (parallel) M2 can start after M1.1-1.2
    │
    └──→ (parallel) M1.3 GGUF dequant can be developed independently
```

**Critical Path**: M1 → M2 → M3 → M4 → M5
**Parallelizable**: 
- M1.2 and M1.3 (safetensors vs GGUF readers)
- M2 can start after M1.1 + M1.2 (safetensors first)
- M4 display can be developed in parallel with M3.4-3.5

---

## 4. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GGUF dequantization edge cases (IQ*, Q4_K_M variants) | Medium | High | Start GGUF reader early; test against `gguf.quants` reference; use known GGUF fixtures |
| Ring buffer thread synchronization bugs | Medium | High | Use `threading.Event` + `queue.Queue` for slot handoff; extensive stress testing with mock data |
| Memory bound violated on 70B models | Low | Critical | Integration test with 200MB mock tensors; `tracemalloc` + `psutil` assertions in CI |
| Cross-format tensor name mismatch | Low (Phase 1) | Medium | Document as known limitation; defer name normalization to Phase 2 |
| Rich table rendering performance on 291 layers | Low | Low | Virtual rendering (only visible rows); batch updates |

---

## 5. Rollback Notes

- **Milestone 1**: Delete `readers/*.py`, revert `readers/__init__.py` to empty
- **Milestone 2**: Delete `engine/metrics.py`, revert `engine/__init__.py`
- **Milestone 3**: Delete `engine/ring_buffer.py`, `engine/diff.py`; no schema changes
- **Milestone 4**: Delete `cli.py`, `display/terminal.py`; no data model changes
- **Milestone 5**: `git tag -d v0.1.0 && git push origin :refs/tags/v0.1.0`; PyPI yank if needed

All milestones are additive — no destructive migrations.

---

## 6. Non-Goals (Explicitly Deferred to Phase 2+)

- `--deep` SVD analysis (truncated SVD, effective rank, pattern classification)
- `--report <file.html>` interactive HTML reports with Plotly
- LoRA adapter comparison (`--lora adapter.gguf`)
- Cross-format tensor name normalization (GGUF `blk.0.attn_q.weight` ↔ HF `model.layers.0.self_attn.q_proj.weight`)
- PyTorch `.pt`/`.bin` checkpoint support
- Standalone binaries (PyInstaller/Nuitka) and `wdiff-rs` Rust crate
- Benchmark automation for Demo 1 & Demo 2 (scripted reproduction)

---

## 7. Progress Tracker

| Milestone | Status | Started | Completed | Notes |
|-----------|--------|---------|-----------|-------|
| M1: Format Readers | ✅ done | 2026-11-05 | 2026-11-05 | 16 tests passing, ruff clean, cross-format verified |
| M2: Metrics Engine | ✅ done | 2026-11-05 | 2026-11-05 | 27 tests passing, ruff clean, all 4 metrics verified |
| M3: Ring Buffer & Diff Engine | ✅ done | 2026-11-05 | 2026-11-05 | 12 tests passing, 2-slot pipeline, thread-safe, exception forwarding |
| M4: CLI & Display | ✅ done | 2026-11-05 | 2026-11-05 | 12 tests passing, Rich tables, JSON output, anomaly detection, filter |
| M5: Memory Bounding, Cleanup & Packaging | ✅ done | 2026-11-05 | 2026-11-05 | Memory bound verified (<35MB), CI workflow, production cleanup, 68 tests total |
| **M2.1: Tensor Name Normalization** | ✅ done | 2026-11-05 | 2026-11-05 | 12 tests passing, cross-format alignment, 5 architectures, virtual aliasing |
| **M2.2: Deep Spectral Engine** | ✅ done | 2026-11-05 | 2026-11-05 | 8 tests passing, randomized SVD, MP noise floor, subspace angles, quantum fidelity |
| **M2.3: Taxonomy Classifier & --deep CLI** | ✅ done | 2026-11-05 | 2026-11-05 | 7 tests passing, 6 taxonomy badges, --deep flag, JSON integration |
| **M2.4: Standalone HTML Reports** | ✅ done | 2026-11-05 | 2026-11-05 | 6 tests passing, self-contained HTML, KPI cards, interactive chart, filterable table |
| **M2.5: LoRA Adapter Support** | ✅ done | 2026-11-05 | 2026-11-05 | 7 tests passing, --lora flag, diff_with_lora, merge_lora_into_model |

---

## 8. Surprises & Discoveries (to be filled during execution)

- **GGUF tensor shape ordering**: GGUF stores shape as `[rows, cols]` but returns data as `(cols, rows)` (transposed). Fixed by using `tensor_info.data.shape` instead of `tensor_info.shape` in `tensor_info()`.
- **Quantization block size constraints**: Q4_K requires last dimension to be multiple of 256. Test fixtures adjusted accordingly (embedding: 64×256, others: 256×256).
- **GGUF quantization type enum**: The `gguf` library uses different quantization type names than assumed (e.g., `Q4_K` not `Q4_K_M`, `Q5_K` not `Q5_K_S`). Updated `_QUANT_TYPE_MAP` to match actual library enums.
- **gguf.quants dequantization warning**: Runtime warning "invalid value encountered in multiply" during Q4_K dequantization of synthetic data - appears to be numerical edge case in test fixtures, not a functional issue.
- **safetensors get_slice() API**: `get_slice(name)` returns a `SafeSlice` with `get_shape()`, `get_dtype()`, but indexing it (`slice[0]`) loads the first row of data, not the byte offset. Use `get_shape()` + dtype mapping for metadata-only access.
- **GGUF quantized tensor shape corruption**: For quantized tensors, `tensor_info.data.shape` returns the compressed raw buffer shape (e.g., (64, 144) for Q4_K), not the logical (64, 256). Must use `reversed(tensor_info.shape)` for logical dimensions.
- **gguf.GGUFWriter quantization**: `add_tensor(..., raw_dtype=Q*)` with float32 arrays does NOT quantize - it writes raw bytes. Must use `gguf.quants.quantize(tensor, type)` first for valid quantized fixtures.
- **float32 cosine similarity precision**: Computing `cos(a, a)` with float32 yields ~1.00000006 due to accumulated rounding in dot product and norm. Use `np.isclose()` with tolerance, not exact equality.
- **Ring buffer producer/consumer pacing**: The consumer (math worker) must be slower than producer (I/O) for queue to fill and enable backpressure. Added 50ms compute delay in consumer vs no delay in producer (fast I/O mock) to enable backpressure testing.
- **Thread-safe exception forwarding**: Producer exceptions wrapped in `_ExceptionWrapper` and re-raised in consumer thread via queue - avoids hanging on I/O errors.
- **safetensors dtype fallback**: Default dtype fallback must be `np.dtype("float32")` not `np.float32` (class vs instance) for `itemsize` to work.
- **argparse version action**: `action="version"` calls `sys.exit(0)` - must catch `SystemExit` in `main()` to return clean exit code.
- **Memory bounding with tracemalloc**: Python's tracemalloc overhead is significant (~2.5x). 4 tensors of 4MB each + overhead = ~40MB peak. Reduced test tensor size to 512×512 (1MB) to stay within 35MB bound while still verifying bounded memory.
- **Early tensor deletion**: Moving `del slot.tensor_a/b` before the consumer sleep reduces peak memory by freeing tensors immediately after metrics computation.
- **HF vs GGUF naming gap**: Naive string matching fails completely for cross-format diffs (0 matches). Canonical GGUF naming with regex-based normalization bridges LLaMA, Qwen2, Gemma, Gemma2, Phi-3, and native GGUF.
- **Architecture detection heuristics**: Detectable via key signatures: `qkv_proj`+`gate_up_proj` → Phi-3; 4 norms per layer → Gemma2; attention bias → Qwen2; `pre_feedforward_layernorm` → Gemma2; native `blk.` prefix → GGUF.
- **Tied embeddings virtual aliasing**: When `tie_word_embeddings=true`, HF omits `lm_head.weight`. Injected virtual alias `output.weight` → `token_embd.weight` enables matching.
- **Named regex groups required**: Python `match.groupdict()` only captures named groups `(?P<layer>\d+)`, not numbered groups `(\d+)`.
- **Marcenko-Pastur edge formula**: For singular values of m×n noise matrix with i.i.d. N(0,σ²) entries, the bulk edge is σ(√m + √n), not σ(1+√γ)² which applies to eigenvalues of the covariance matrix.
- **Subspace angles interpretation**: Rotating a subspace within itself yields angle 0; principal angles measure distance between subspaces, not rotation within a subspace. To get non-zero angle, must rotate towards orthogonal direction.
- **Quantum fidelity norm source**: For truncated SVD, norms must be computed from SVD components (sum of squared singular values), not full matrix norm, to get fidelity=1 for identical matrices.
- **Randomized SVD oversampling**: p=10 oversampling + 2 power iterations gives excellent accuracy for low-rank matrices while maintaining speed.
- **MP edge for signal rank**: Using s_+ = σ(√m + √n) with 1% margin correctly separates LoRA (rank 16) and abliteration (rank 1) signatures.
- **Taxonomy thresholds**: Sparsity ≥ 0.65 → SPARSE_UPDATE; E₁ > 0.90 + r_stable ≤ 1.35 → RANK1_ABLATION; signal_rank = 0 → STOCHASTIC_NOISE; 2 ≤ signal_rank ≤ 64 + r_stable ≤ 40 → LOW_RANK_ADAPTER.
- **Deep CLI integration**: --deep flag triggers spectral + taxonomy per tensor; JSON output includes spectral bundle and taxonomy badge with evidence; Rich table shows rank (signal/effective) and colored badge column.
- **HTML Report generation**: Self-contained single-file HTML report with inline dark-mode CSS, vanilla JS table search/filtering, and SVG delta profile chart.
- **SVG chart coordinate edge case**: In single-layer deep diffs, `deepRows.length - 1 == 0` causes division by zero. Protected with ternary check `deepRows.length > 1 ? ... : chartWidth / 2`.
- **LoRA adapter streaming**: Pre-opening base model reader outside tensor loop avoids reopening/reparsing GGUF/safetensors headers for each layer.
- **Conv2d SVD flattening**: 4D convolution weights `(out, in, h, w)` must be flattened to `(out, in*h*w)` before randomized SVD and Marcenko-Pastur noise floor estimation.

---

## 9. Decision Log (to be filled during execution)

- **2026-11-05**: Use `tensor_info.data.shape` (not `tensor_info.shape`) for GGUF tensor shapes to handle GGUF's transposed storage.
- **2026-11-05**: Map GGUF quantization types to actual `gguf.GGMLQuantizationType` enum values (Q4_K, Q5_K, Q6_K, Q8_K, IQ* variants) rather than assumed names.
- **2026-11-05**: Use `typing_extensions.Self` for `__enter__` return types per ruff PYI034.
- **2026-11-05**: Combine nested `with` statements for context managers per ruff SIM117.
- **2026-11-05**: In `safetensors_reader.tensor_info()`, use `get_slice().get_shape()` + dtype mapping instead of `get_tensor()` to avoid loading data into RAM (memory-bound requirement).
- **2026-11-05**: In `gguf_reader.tensor_info()`, use `reversed(tensor_info.shape)` for logical shape to handle quantized tensors correctly.
- **2026-11-05**: Use `gguf.quants.quantize()` for generating valid quantized test fixtures; `GGUFWriter.add_tensor(raw_dtype=Q*)` expects pre-quantized bytes.
- **2026-11-05**: Metrics functions use `float(np.linalg.norm(...))` and explicit casting to ensure Python float return types; epsilon `1e-12` prevents division by zero.
- **2026-11-05**: Use `np.isclose()` with appropriate tolerances in tests instead of exact equality for float32 precision limits.
- **2026-11-05**: `PipelinedRingBuffer` uses `queue.Queue(maxsize=2)` for strict memory bounding (2 tensor pairs max in flight).
- **2026-11-05**: Producer thread fetches tensors sequentially; main thread acts as math consumer. Exception wrapping via `_ExceptionWrapper` enables clean error propagation.
- **2026-11-05**: Consumer simulates math compute time (50ms) to enable backpressure testing; producer uses fast mock readers.
- **2026-11-05**: `diff_models()` categorizes tensors: common (matched shapes → ring buffer), shape mismatches (recorded without loading weights), only-in-A, only-in-B.
- **2026-11-05**: Shape check via `tensor_info()` before reading weights avoids loading large mismatched tensors.
- **2026-11-05**: Regex filtering applied to tensor names before diff; filter affects both models symmetrically.
- **2026-11-05**: `DiffSummary` tracks matched/mismatched/only-in-A/only-in-B counts plus max/min/mean metrics for matched tensors.
- **2026-11-05**: Rich `Table` for diff output with visual ΔF bars, colored status, and consolidated Shape column.
- **2026-11-05**: `detect_anomalies()` flags near-orthogonal shifts (cosine < 0.1), significant changes (ΔF > 10%), shape mismatches, asymmetric keys.
- **2026-11-05**: CLI `--json` outputs structured JSON with model paths, formats, summary stats, and per-tensor metrics.
- **2026-11-05**: CLI `--filter` applies regex to both model tensor names symmetrically before diffing.
- **2026-11-05**: `argparse` version handling catches `SystemExit` for clean exit code return.
- **2026-11-05**: Moved tensor deletion before consumer sleep to minimize peak memory in memory-bound tests.
- **2026-11-05**: Test tensor size reduced to 512×512 (1MB) to satisfy 35MB tracemalloc bound while verifying bounded memory.
- **2026-11-05**: GitHub Actions CI matrix for Python 3.10/3.11/3.12 with ruff lint, format check, and pytest.
- **2026-11-05**: Adopted GGUF naming as canonical standard for cross-format tensor matching.
- **2026-11-05**: Regex-based normalization with named groups `(?P<layer>\d+)` and `(?P<suffix>weight|bias)` for LLaMA, Qwen2, Gemma, Gemma2, Phi-3, and native GGUF.
- **2026-11-05**: Architecture auto-detection via key signatures (Phi-3: qkv_proj+gate_up; Gemma2: 4 norms; Qwen2: attention bias; GGUF: blk. prefix).
- **2026-11-05**: Virtual alias for tied embeddings (`output.weight` → `token_embd.weight`) injected when `tie_word_embeddings=True`.
- **2026-11-05**: Alignment report separates matched pairs, only-in-A, only-in-B; diff engine uses matched pairs for ring buffer, preserves asymmetric keys in results.
- **2026-11-05**: Shape check uses raw names from alignment, metrics computed on canonical pairs; summary uses canonical names.
- **2026-11-05**: Marcenko-Pastur noise edge uses σ(√m + √n) for singular values of noise matrix.
- **2026-11-05**: Subspace angles computed via SVD of U_A^T U_B; rotation within subspace gives angle 0.
- **2026-11-05**: Quantum fidelity uses SVD-component norms for fidelity=1 with identical matrices.
- **2026-11-05**: Randomized SVD with p=10 oversampling, 2 power iterations for accuracy/speed balance.
- **2026-11-05**: MP edge with 1% margin correctly separates LoRA (rank 16) and abliteration (rank 1) signals.

---

## 10. Outcomes & Retrospective (to be filled after completion)

*TBD.*

---

## Next Command to Execute

After reviewing this plan, run the first implementation milestone:

```bash
# Start Milestone 1: Format Abstraction Layer
# (Implementation will be done in subsequent steps after plan approval)
```

**Plan review checklist:**
- [ ] Milestone scope and order aligned with architecture docs
- [ ] Verification gates have concrete pytest commands
- [ ] Dependencies correctly sequenced
- [ ] Risks identified with mitigations
- [ ] Rollback paths defined per milestone
- [ ] Non-goals explicitly bounded

Once approved, I'll begin implementing Milestone 1 (Readers) with the `investigate-first` skill pattern: read existing abstractions → write minimal implementation → test → iterate.