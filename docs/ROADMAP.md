# `wdiff` Development Roadmap

This document outlines the version milestones, feature targets, and release criteria for `wdiff`.

---

## 🎯 Release Schedule & Milestones

### Phase 1: Core Engine & CLI (`v0.1.0`) — COMPLETE ✅
*Goal: Ship a working, memory-bounded CLI tool that compares GGUF and safetensors checkpoints.*

- [x] Technical architecture specification & mathematical foundations
- [x] Project scaffolding & packaging (`pyproject.toml`)
- [x] Implement `src/wdiff/readers/base.py`: Abstract `TensorReader` interface
- [x] Implement `src/wdiff/readers/safetensors_reader.py`: Memory-mapped safetensors parser
- [x] Implement `src/wdiff/readers/gguf_reader.py`: Memory-mapped GGUF parser + NumPy dequantization
- [x] Implement `src/wdiff/engine/metrics.py`: Relative Frobenius delta, Cosine similarity, Max element delta
- [x] Implement `src/wdiff/engine/ring_buffer.py`: 2-slot prefetch queue overlapping disk I/O and CPU math
- [x] Implement `src/wdiff/display/terminal.py`: `rich` CLI tables, progress bars, and anomaly alerts
- [x] Unit & integration test suite verifying streaming no-accumulation (tracemalloc peak < 35MB through the ring buffer; see `tests/integration/test_memory.py`)
- [x] Packaging verification & CI matrix (Python 3.10, 3.11, 3.12)

---

### Phase 2: Deep Spectral Engine, HTML Visualizer & LoRA (`v0.2.0`) — COMPLETE ✅
*Goal: Enable deep rank analysis, automated taxonomy classification, standalone HTML reports, and on-the-fly LoRA diffing.*

- [x] Implement Milestone 2.1: Canonical architecture normalizer (`naming.py`) bridging LLaMA, Qwen2, Gemma/Gemma2 4-norm, Phi-3 fused QKV, GGUF, and virtual tied embeddings.
- [x] Implement Milestone 2.2: Deep Spectral Engine (`spectral.py`) featuring Fast Randomized SVD (9.2× faster than ARPACK on CPU), Marčenko-Pastur noise floor via MAD, Grassmannian principal angles, and Quantum Uhlmann fidelity shortcut.
- [x] Implement Milestone 2.3: Taxonomy Classifier (`taxonomy.py`) with 6 automated badges (`UNMODIFIED`, `RANK1_ABLATION`, `LOW_RANK_ADAPTER`, `SPARSE_UPDATE`, `FULL_RANK_CPT`, `STOCHASTIC_NOISE`) & `--deep` CLI integration.
- [x] Implement Milestone 2.4: Standalone offline HTML visualizer (`--report`) with KPI summary cards, interactive SVG delta chart, and client-side instant search.
- [x] Implement Milestone 2.5: LoRA Adapter Engine (`--lora`, `--lora-scale`) streaming base weights against adapter deltas on the fly without disk merging.
- [x] 112 / 112 automated tests passing, 91% code coverage, 100% ruff clean.

---

### Phase 3: Production Systems & Semantic Intelligence (`v0.3.0`) — COMPLETE ✅
*Goal: Eliminate all production barriers, enable single-model audits, and decode human-readable concepts from weight deltas.*
*(See `docs/PHASE_3_ARCHITECTURE_AND_FOUNDATIONS.md` for full technical specification and `docs/ERRORS_AND_FIXES.md` for engineering bug logs)*

- [x] **Milestone 3.1: Single-Model Deep Inspection (`wdiff inspect <model>`)** ✅ COMPLETE
  - Instant structural & spectral health audits without a reference model (`wdiff inspect <model>`)
  - Auto-routing single positional models (`wdiff <model>`) with `--deep`, `--filter`, `--report`, `--json`
  - Dead / saturated neuron detection ($\|w\| \le 10^{-7}$)
  - Dettmers outlier ratio with strided median estimation for $<5$s runtime on $>300\text{M}$ models
  - Rich inspection tables with automatic terminal proportional scaling
  - 123 / 123 automated tests passing, 100% ruff clean.
- [x] **Milestone 3.2: Multi-File Sharded SafeTensors Reader (`ShardedSafeTensorsReader`)** ✅ COMPLETE
  - Support for multi-shard checkpoints via `model.safetensors.index.json`, directories, and shard files
  - Dynamic weight map extraction from directories with unindexed shards
  - LRU handle cache (default max 2 open readers) maintaining $<35$MB memory bound across 80+ shards (supporting 70B and 405B models)
  - Seamless cross-format diffing and inspection between sharded safetensors, single-file safetensors, and GGUF
  - 143 / 143 automated tests passing, 100% ruff clean.
- [x] **Milestone 3.3: RoPE Coordinate De-Interleaving & Parity** ✅ COMPLETE
  - Inverted GGML SIMD interleaved coordinate pairs ($[-x_1, x_0, -x_3, x_2, \dots]$) back to HuggingFace split-half format ($[-x_{d/2:}, x_{:d/2}]$)
  - Adaptive parity checking via inner-product alignment ($\langle W_{\text{deint}}, W_{\text{hf}} \rangle > \langle W_{\text{raw}}, W_{\text{hf}} \rangle$) preventing false over-permutation
  - GGUF metadata reader extracting `attention.head_count` and `attention.head_count_kv`
  - Grouped-Query Attention (GQA) support for unequal Q and KV head counts
  - CLI control with `--no-rope-parity` flag
  - 159 / 159 automated tests passing, 100% ruff clean.
- [x] **Milestone 3.4: Partial Shape Matching & Vocabulary Expansion Slicing** ✅ COMPLETE
  - Single-axis variance common-prefix slicing for expanded vocabularies (e.g. 128,000 $\to$ 129,280 tokens)
  - Isolates extra rows as `PARTIAL_MATCH` with complete expansion telemetry (`extra_count`, `extra_norm`, `extra_mean`, `extra_std`)
  - Multi-axis dimensional divergence and ndim differences safely fall back to `SHAPE_MISMATCH`
  - Unified inline dimension notation: `128,000→129,280 × 4,096`
  - Preserves natural architectural layer sequence in terminal displays, HTML reports, and JSON exports
  - 169 / 169 automated tests passing, 100% ruff clean.
- [x] **Milestone 3.5: Semantic Concept Translation via Logit Lens** ✅ COMPLETE
  - Project FFN down-projection delta singular vectors ($u_1 \in \mathbb{R}^{d_{\text{model}}}$) through the unembedding head $W_U \in \mathbb{R}^{V \times d_{\text{model}}}$ (*Geva et al.*, *nostalgebraist*)
  - Bipolar concept spectrum decoding: top-$K$ positive tokens (+ pole) and top-$K$ negative tokens (- pole)
  - Memory-bounded streaming chunk projection ($\le 4096$ token rows, $<70$MB RAM ceiling across 128k+ vocabularies)
  - Universal vocabulary auto-discovery across GGUF embedded metadata, HuggingFace `tokenizer.json`, and `vocab.json`
  - Integrated into single-pass streaming ring buffer with `--lens`, `--tokenizer`, and `--lens-top-k` CLI controls
  - Rich terminal concept panels, interactive HTML report KPI cards/expandable concept rows, and full JSON export
  - 186 / 186 automated tests passing, 100% ruff clean.

---

### Phase 4: Pre-Release Engineering Audit & Hardening (`v0.3.0`) — COMPLETE ✅
*Goal: Comprehensive reality check, vulnerability mitigation, packaging hygiene, and specification reconciliation prior to public launch.*

- [x] **Milestone 4.1: CI Dependency Resolution & Packaging Security (P0)** ✅ COMPLETE
  - Fixed GitHub Actions CI workflow in `.github/workflows/ci.yml` to use `uv sync --extra dev --python ${{ matrix.python-version }}` ensuring pytest and ruff are installed.
  - Hardened `pyproject.toml` sdist configuration (`[tool.hatch.build.targets.sdist]`) with strict exclusions for `/.agent`, `/AGENTS.md`, `/benchmarks`, `/tests/fixtures`, and backup files, preventing internal artifacts from leaking into release tarballs.
- [x] **Milestone 4.2: Security & Defensive Input Validation (P1)** ✅ COMPLETE
  - Neutralized Stored Cross-Site Scripting (XSS) in HTML reports (`report.py`) by escaping layer names, adversarial vocabulary tokens (e.g., `<|im_start|>`, `<script>`), status tags, and report titles via client-side `escapeHtml()` and Python `html.escape()`. Added unit tests verifying adversarial payloads render harmlessly.
  - Guarded `ShardedSafeTensorsReader` against path traversal attacks (`shard_path.resolve().is_relative_to(index_dir.resolve())`) preventing malicious index files from reading outside the checkpoint directory. Added regression tests in `tests/readers/test_sharded.py`.
- [x] **Milestone 4.3: Architecture & Mathematical Documentation Reconciliation (P1)** ✅ COMPLETE
  - Reconciled `docs/ARCHITECTURE.md` to accurately indicate that legacy PyTorch checkpoint support is planned for Phase 5 rather than currently implemented.
  - Clarified that `diff_models()` executes paired layer comparison synchronously via `_run_paired_buffer()` to keep paired-slot memory bounded to two tensor pairs while eliminating GIL lock contention.
  - Aligned Effective Rank formulas in `docs/ARCHITECTURE.md` and `docs/learning/02_SPECTRAL_ANALYSIS_AND_SVD.md` with `spectral.py`: Shannon spectral entropy of normalized squared singular value energy $p_i = \sigma_i^2 / \sum \sigma_j^2$, and primary direction energy $E_1 = \sigma_1^2 / \sum \sigma_i^2$.
- [x] **Milestone 4.4: Code Hygiene & Performance Optimization (P2)** ✅ COMPLETE
  - Removed duplicate `_format_shape` function definition in `terminal.py`.
  - Removed triplicated serialization methods and dead `LoRAResultWrapper` fallback in `cli.py`.
  - Hoisted per-tensor imports outside inner loop in `lora.py`.
  - Removed unnecessary dummy delta taxonomy computation on identical tensors ($w - w$) in `inspect.py`.
  - Cleaned untracked editor backup files (`*.bak*`).
- [x] Full automated test suite passing (207/207 tests, 0 failures), 100% ruff clean.

---

### Phase 5: Public Launch, Demos & Standalone Binaries (`v0.4.0`) — NEXT 🚀
*Goal: Execute public launch with high-visibility benchmarks and zero-dependency binaries.*

- [ ] Benchmark 1: `Meta-Llama-3-8B-Instruct` vs `Llama-3-8B-Instruct-abliterated` (Demonstrate Rank-1 safety direction removal)
- [ ] Benchmark 2: Full-model forensic autopsy of 4-bit quantization damage across layers
- [ ] Publish technical launch articles on Medium, Dev.to, and Hacker News ("Show HN: wdiff")
- [ ] GitHub Actions CI workflow to compile standalone PyInstaller single-file binaries for Linux (x86_64, aarch64) and macOS (ARM64)
