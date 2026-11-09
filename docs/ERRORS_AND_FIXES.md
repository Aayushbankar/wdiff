# `wdiff` Engineering Ledger: Errors, Root Causes & Fixes Log

This document serves as the permanent forensic ledger of all engineering bugs, architectural bottlenecks, mathematical edge cases, and unexpected behaviors encountered during the development of `wdiff`, alongside their root causes and permanent fixes.

---

## Phase 1 & Phase 2 Historical Ledger (Errors 1 – 11)

### Error 1: SafeTensors Memory Leak in `tensor_info()`
- **Symptom**: Calling `tensor_info(name)` caused gradual memory accumulation during header inspection on large checkpoints.
- **Root Cause**: The initial implementation called `self._file.get_tensor(name).shape`, which forced the entire multi-gigabyte weight matrix into RAM rather than reading disk metadata.
- **Fix**: Switched to `self._file.get_slice(name).get_shape()`, reading logical dimensions directly from the SafeTensors header without allocating tensor buffers.

### Error 2: GGUF Quantized Logical Dimension Inversion
- **Symptom**: GGUF quantized tensors returned transposed or corrupted 2D shapes compared to HuggingFace SafeTensors equivalents.
- **Root Cause**: `gguf-py` stores quantized tensor shapes in raw uint8 storage byte counts and reverses the dimensional order relative to PyTorch / NumPy row-major conventions.
- **Fix**: Inverted dimensions via `tuple(reversed(tensor_info.shape))` and mapped quantized block sizes to logical parameter counts.

### Error 3: Canonical Aliasing Crash in `--deep`
- **Symptom**: Running `diff_models(..., deep=True)` crashed with `KeyError` when diffing models with aliased naming conventions (e.g., LLaMA vs Qwen2).
- **Root Cause**: `compute_spectral_bundle()` attempted to load weights from disk using the canonicalized name (e.g., `model.layers.0.attn_q.weight`) rather than the raw file keys (`raw_name_a` / `raw_name_b`).
- **Fix**: Decoupled canonicalized keys (used strictly for layer matching and alignment) from raw disk keys (used for reader queries).

### Error 4: Double Model I/O & Memory Spike During Deep Diff
- **Symptom**: Memory consumption spiked to $>2\times$ expected bounds when `--deep` was passed.
- **Root Cause**: The engine ran a second full streaming pass across both checkpoints to compute spectral metrics after completing the Frobenius/Cosine pass.
- **Fix**: Hoisted SVD and taxonomy classification into the single-pass streaming ring buffer (`_run_paired_buffer`). Each layer pair is read from disk exactly once.

### Error 5: 4D Conv2D Matrix Unpack Crash in SVD
- **Symptom**: Diffing Vision Transformers (ViT) or Stable Diffusion models crashed with `ValueError: expected 2D matrix` inside Randomized SVD.
- **Root Cause**: 4D convolution weights ($C_{\text{out}} \times C_{\text{in}} \times H \times W$) were passed directly to SVD solvers without spatial flattening.
- **Fix**: Added automatic spatial reshaping: `A = w.reshape(w.shape[0], -1)` before spectral decomposition, preserving input/output channel subspace geometry.

### Error 6: SVG Profile Chart Division by Zero on Single-Row Diffs
- **Symptom**: HTML report generation crashed with `ZeroDivisionError` when a diff contained exactly one modified layer.
- **Root Cause**: SVG X-coordinate normalization calculated `(x - x_min) / (x_max - x_min)` without checking if `x_max == x_min`.
- **Fix**: Added a guard `deepRows.length > 1 ? ... : chartWidth / 2` in `report.py`.

### Error 7: Duplicate `_serialize_lora_diff_result` in `cli.py`
- **Symptom**: Code smell and potential shadowing detected during static analysis.
- **Root Cause**: Merge conflict artifact during Milestone 2.5 LoRA integration.
- **Fix**: Removed the duplicate serialization function and consolidated result schema.

### Error 8: LoRA File Handle Reopening in Layer Streaming Loop
- **Symptom**: Noticeable I/O latency when computing `diff_with_lora`.
- **Root Cause**: The LoRA reader context manager was instantiated inside the layer iteration loop, opening and closing the adapter file on every tensor.
- **Fix**: Hoisted `with get_reader(lora_path) as lora_reader:` outside the streaming loop.

### Error 9: Hardcoded Zero Summary Metrics in `LoRAResultWrapper`
- **Symptom**: Running `wdiff base.safetensors --lora adapter.safetensors` printed 0 for all summary metrics in the Rich terminal panel.
- **Root Cause**: The adapter wrapper mock object initialized summary counters to 0 rather than forwarding computed aggregations from `LoRADiffResult`.
- **Fix**: Created `_LoRAResultWrapper` mapping real computed `mean_frobenius`, `max_frobenius`, and `min_cosine` values.

### Error 10: Flaky Producer Race Condition in Ring Buffer Tests
- **Symptom**: `test_producer_blocks_when_buffer_full` occasionally failed in high-load CI environments.
- **Root Cause**: Test relied on an artificial `time.sleep(0.005)` to sync threads rather than synchronization primitives.
- **Fix**: Replaced timing delays with thread event synchronization.

### Error 11: Upstream `gguf-py` Crash on SD 1.5 GGUF 4D Convolutions
- **Symptom**: Unhandled Python traceback (`ValueError: cannot reshape array...`) when attempting to dequantize 4D convolution tensors in certain non-LLM GGUF checkpoints.
- **Root Cause**: Upstream `gguf-py` dequantization routines only supported 1D and 2D tensor layouts.
- **Fix**: Trapped the dequantization error gracefully in `GGUFReader`, printing a clean human-readable error to stderr with exit code 1.

---

## Phase 3 Milestone 3.1 Forensic Ledger (Errors 12 – 15)

### Error 12: CLI Flag Parser Trap (`wdiff <model> --filter ...`)
- **Symptom**:
  Running `wdiff /path/to/model.gguf --filter "blk\.0\."` crashed with:
  ```text
  usage: wdiff [-h] [-f TENSOR_FILTER] [--deep] [--lora PATH] ...
  wdiff: error: Either model_b or --lora must be provided
  ```
- **Root Cause**:
  To auto-route single models to `inspect` mode, the developer agent wrote:
  ```python
  positional_args = [a for a in (argv or []) if not a.startswith("-")]
  should_inspect = len(positional_args) == 1 and ...
  ```
  In command `wdiff model.gguf --filter "blk\.0\."`, the argument `"blk\.0\."` does not start with `-`. Thus, `positional_args` was evaluated as `["model.gguf", "blk\\.0\\."]`. With length 2, `should_inspect` became `False`, falling through to the diff parser where `model_b` was missing.
- **Fix**:
  Unified argument parsing through `parser.parse_args(argv)`. When `args.model_b is None` and `not args.lora`, the command unambiguously auto-routes to `inspect_model` with full support for `--filter`, `--deep`, `--json`, and `--report`.
- **Regression Test**:
  `tests/engine/test_inspect.py::TestCLIInspect::test_cli_single_model_with_filter_auto_inspect`.

---

### Error 13: Rich Table Column Collapse (`┏┳┳`) & Duplicate Headers
- **Symptom**:
  In non-TTY environments or standard 80-column terminal outputs, running `wdiff inspect <model>` printed:
  ```text
  ┏┳┳━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━┳┓
  ┃┃┃   Dtype   ┃      Params ┃     L2 Norm ┃   Outlier ┃ Dead  ┃    Status     ┃┃
  ```
  The first two columns (`Layer` and `Shape`) were collapsed to 0 width (`┏┳┳`), completely invisible to the user.
- **Root Causes**:
  1. **Column Width Budget Exceeded**: Hard fixed widths (`width=10`, `width=12`, etc.) were assigned to 6 numeric columns, consuming 71 of 80 terminal characters. Rich allocated the remaining 0 characters to flexible columns (`Layer` and `Shape`).
  2. **Row Cell Count Mismatch**: `row_data.append(Text(t.status))` was invoked twice in `render_inspection_table` (lines 270 and 280), injecting a 9th cell into an 8-column table and forcing Rich to generate an empty column header `''`.
  3. **Duplicate Status in Diff Table**: `render_diff_table` in `terminal.py` called `table.add_column("Status")` twice (lines 78 and 84).
- **Fix**:
  1. Converted `Layer` and `Shape` to flexible proportional columns (`min_width=20, ratio=2` and `min_width=8, ratio=1`).
  2. Abbreviated dtypes (`float32` $\to$ `f32`, `float16` $\to$ `f16`, `bfloat16` $\to$ `bf16`), reclaiming 4 columns.
  3. Removed the duplicated `row_data.append(Text(t.status))` and duplicate `Status` column additions.
- **Regression Test**:
  `tests/display/test_terminal.py::TestTerminalDisplay::test_render_inspection_table` and `assert columns.count("Status") == 1`.

---

### Error 14: 1D Bias Vectors Erroneously Flagged as Rank-1 Dominant
- **Symptom**:
  Inspecting diffusion checkpoints (`diffusion_pytorch_model.safetensors`) with `--deep` produced false-alarm anomaly spam:
  ```text
  ⚠ Rank-1 dominant: encoder.layers.0.bias (E1=100.0%)
  ⚠ Rank-1 dominant: encoder.layers.1.conv.0.bias (E1=100.0%)
  ...
  ```
- **Root Cause**:
  In linear algebra, any 1D vector $v \in \mathbb{R}^d$ has matrix rank 1, meaning the first singular value trivially accounts for 100.0% of the spectral energy ($\sigma_1^2 / \sum \sigma_i^2 = 1.0$). Flagging 1D biases and scalars as low-rank matrix interventions is mathematically vacuous and clutters real forensic results.
- **Fix**:
  Added a guard `len(t.shape) >= 2` in `detect_inspection_anomalies()`, restricting rank-1 dominance checks strictly to 2D weight matrices and 4D convolution kernels.

---

### Error 15: 360M Outlier Score Quickselect Bottleneck (38.4s Latency)
- **Symptom**:
  Running `wdiff inspect` on `smollm2-360m-instruct-q8_0.gguf` took **38.43 seconds** despite streaming memory remaining bounded.
- **Root Cause**:
  `_compute_outlier_score` evaluated `np.median(flat)` and `np.median(np.abs(flat - median_val))` on the raw un-subsampled array. For large embedding and projection layers ($49,152 \times 576 = 28.3\text{M}$ float32 elements), NumPy ran two full $O(N)$ quickselect passes and allocated redundant 113MB temporary arrays per layer across 290 tensors.
- **Fix**:
  Decoupled outlier detection from distribution median estimation:
  - `max_val = float(np.max(flat))` remains **100% exact** across all $N$ elements so zero coordinate spikes are ever missed.
  - For tensors with $>100,000$ parameters, `median` and `MAD` (the bulk distribution denominator) are calculated using a deterministic strided subsample of 100,000 elements ($O(1)$ memory, sub-millisecond quickselect).
  - Runtime dropped from **38.43 seconds to 4.82 seconds** (an 8× speedup) with identical quantization risk classification.

---

### Error 16: Sharded Format Blindness & Sibling Standalone Hijacking
- **Symptom**:
  1. Running `diff_models()` on sharded checkpoints reported `format_a: unknown` and `format_b: unknown` in summary panels and HTML reports.
  2. Passing a standalone `.safetensors` file located in the same directory as an unrelated `model.safetensors.index.json` caused the standalone file to be hijacked by `ShardedSafeTensorsReader`.
- **Root Cause**:
  1. `src/wdiff/engine/diff.py` had a legacy private `_detect_format` function checking only `.endswith(".safetensors")` and `.endswith(".gguf")`, unaware of index files or directories.
  2. `_is_sharded_safetensors` in `src/wdiff/readers/__init__.py` returned `True` for any `.safetensors` file if any `*.safetensors.index.json` existed in `path.parent`, even if the file was an independent model not listed in `weight_map`.
- **Fix**:
  1. Updated `diff.py`'s `_detect_format` to delegate to `wdiff.readers.detect_format`, properly identifying `"sharded_safetensors"`.
  2. Updated `_is_sharded_safetensors` to parse `weight_map` and verify `path.name in set(weight_map.values())` before treating a single `.safetensors` file as a shard.
- **Regression Test**:
  `tests/readers/test_sharded.py::TestShardedDetection::test_standalone_safetensors_not_hijacked_by_sibling_index`.

---

### Error 17: RoPE Blind Over-Deinterleaving & Missing NumPy Import in Diff Engine
- **Symptom**:
  1. Comparing `model_a.safetensors` against `model_a.gguf` with default `rope_parity=True` produced a false divergence on `attn_q` and `attn_k` ($\text{Cosine} \approx 0.019$, $\Delta F \approx 140\%$).
  2. The developer agent bypassed this in `tests/engine/test_diff.py` by forcing `rope_parity=False`.
  3. When running via the CLI, `diff_models` crashed with `NameError: name 'np' is not defined`.
- **Root Cause**:
  1. `deinterleave_rope()` was applied unconditionally to any cross-format GGUF attention projection. Synthetic GGUF models or GGUFs not converted through llama.cpp's RoPE permutation logic were erroneously permuted *away* from HuggingFace coordinates.
  2. `src/wdiff/engine/diff.py` lacked `import numpy as np`.
- **Fix**:
  1. Added `import numpy as np` to `src/wdiff/engine/diff.py`.
  2. Made RoPE parity adaptive: before replacing a GGUF tensor with its deinterleaved version, it compares the inner product:
     $$\langle W_{\text{deinterleaved}}, W_{\text{hf}} \rangle > \langle W_{\text{raw}}, W_{\text{hf}} \rangle$$
     If de-interleaving improves alignment (as in real llama.cpp converted GGUFs), it applies the de-interleaving. If the GGUF was already un-permuted (as in synthetic fixtures or non-RoPE models), it keeps the raw tensor.
- **Regression Test**:
  `tests/engine/test_diff.py::TestDiffModels::test_cross_format_identical_safetensors_and_gguf` (passes with default `rope_parity=True`).

---

### Error 18: Partial Shape Dimension Dropping & Result Sequence Scrambling
- **Symptom**:
  1. Inspecting vocabulary-expanded models displayed truncated shape strings in the terminal: `'128,000 × 4,096 → 129,280'`, dropping the trailing dimension on the right side.
  2. Tensor results in terminal tables and HTML reports were reordered: vocabulary-expanded embeddings (`model.embed_tokens.weight`) were pushed to the bottom of the table below layer 31 rather than appearing at layer 0 in natural architectural sequence.
- **Root Cause**:
  1. In `src/wdiff/display/terminal.py`, `_format_partial_shape()` appended matching dimensions solely to `parts_a` and omitted them from `parts_b`. When joined with ` → `, `parts_b` only contained the differing dimension.
  2. In `src/wdiff/engine/diff.py`, partial match tensor pairs were evaluated in an early block and accumulated into a separate `partial_results` list. At line 351, `tensor_results.extend(partial_results)` appended them after all regular matched tensors, destroying the layer-by-layer ordering established by `CanonicalAligner`.
- **Fix**:
  1. Rewrote `_format_partial_shape` to construct unified inline dimension notations: `f"{dim_a:,}→{dim_b:,}"` on differing axes and `f"{dim_a:,}"` on identical axes, producing clean notation: `128,000→129,280 × 4,096`.
  2. Refactored `diff_models()` to store partial match outcomes in a lookup dictionary and assemble `tensor_results` strictly in the natural sequence of `alignment.matched_pairs`.
- **Regression Test**:
  `tests/engine/test_partial_shape.py::test_partial_match_vocabulary_expansion_2d` and full test suite verification.

---

### Error 19: Logit Lens SVD Dimension Orientation Bug & Redundant Secondary Streaming Pass
- **Symptom**:
  1. `decode_layer_concept()` in `src/wdiff/engine/lens.py` failed or crashed on standard HuggingFace PyTorch checkpoints where `down_proj.weight` has shape `(d_model, d_ffn)`.
  2. Running `diff_models(..., lens=True)` caused a double disk I/O pass, re-reading all down-projection layers from disk after the primary streaming pass had already finished.
  3. `src/wdiff/engine/diff.py` raised static and runtime `NameError: name 'Path' is not defined` when `Path` was used without import.
  4. 31 static analysis errors were present in the codebase (unused local variables, unhandled blind exceptions, and unformatted imports).
- **Root Cause**:
  1. In `src/wdiff/engine/lens.py`, the developer assumed `d_model = diff_matrix.shape[1]`. In PyTorch, `shape[0]` is the output dimension entering the residual stream ($d_{\text{model}}$). Furthermore, checking `elif U.shape[1] == d_model` was algebraically flawed because `U.shape[1]` is $k \le 4$, never 4,096.
  2. In `src/wdiff/engine/diff.py`, lens decoding was placed in a separate post-hoc loop, re-reading multi-gigabyte tensors from readers rather than computing projections while the weights were already resident in the ring buffer.
  3. Missing imports (`from pathlib import Path`, `import contextlib`).
- **Fix**:
  1. Extracted target dimension $d_{\text{target}}$ from $W_U$ (`vocab_size, d_model`). Evaluated both left and right singular vectors against $d_{\text{target}}$: `if U.shape[0] == d_target: u1 = U[:, 0] elif Vt.shape[1] == d_target: u1 = Vt[0, :]`. This guarantees concept decoding works regardless of whether weights are stored in PyTorch or transposed order.
  2. Hoisted lens decoding directly into `_run_paired_buffer`. Tensors `tensor_a` and `tensor_b` are reused without additional disk reads, strictly preserving the $<35$MB / $<70$MB RAM ceiling.
  3. Added missing imports, eliminated blind exceptions, cleaned all 31 ruff errors, and formatted the codebase with `ruff format .`.
- **Regression Test**:
  `tests/engine/test_logit_lens.py` and full suite verification (186/186 passing, 0 ruff errors).

---

### Error 20: Memory Ballooning in `wu_matrix` Logit Lens Pre-loading & float32 Dot Product Overflow
- **Symptom**:
  1. Running full-model differential analysis with `--lens` on 128k vocabulary models (e.g. `LFM3-1.7B-Q4_K_M`) triggered severe memory allocation spikes ($>3\text{GB}$ heap) and heavy swap thrashing, risking `SIGKILL 137` on memory-constrained systems.
  2. Evaluating Frobenius norm and cosine similarity across large embedding tensors ($128,000 \times 2,048$) produced `RuntimeWarning: overflow encountered in dot`.
- **Root Cause**:
  1. In `diff_models()`, `wu_matrix` was eagerly dequantized into a monolithic $128,000 \times 2,048$ float32 array (1.05 GB). In GGUF quantized models, this forced 1GB of memory to be allocated up front in addition to ring buffer tensors.
  2. In `src/wdiff/engine/metrics.py`, computing `sqnorm = x.dot(x)` on 262M float32 elements overflowed float32 dynamic range ($>3.4 \times 10^{38}$) before being cast to float64.
- **Fix**:
  1. Replaced monolithic array loading with lazy chunked getters: `_make_gguf_getter` (using direct 2D GGUF slice indexing `t_info.data[start:end]` and `dequantize()`) and `_make_safe_getter` (using SafeTensors `get_slice()[start:end]`). Only 4,096 vocabulary rows (32 MB) are materialized in RAM at any given moment during concept decoding.
  2. Implemented chunked 4,096-row streaming in `metrics.py` for tensors exceeding 2M elements, accumulating dot products and norms strictly in `float64`.
- **Regression Test**:
  `tests/engine/test_logit_lens.py` and full Benchmark 1 execution.

---

### Error 21: Column Width Collapse in Rich Diff Table on 80-Column Terminals
- **Symptom**:
  Running `wdiff` with `--deep` in standard 80-column terminal environments caused the `Layer` and `Shape` columns to collapse into zero-width vertical borders (`┏┳┳`), rendering layer names invisible.
- **Root Cause**:
  Fixed character widths across all table columns summed to over 82 characters, exceeding the terminal viewport width and causing Rich's layout solver to squash non-fixed columns to zero width.
- **Fix**:
  Set `expand=True` on `Table` and assigned explicit proportional column ratios (`ratio=3, min_width=16` for Layer, `ratio=2, min_width=10` for Shape, `width=10` for Cosine Sim).
- **Regression Test**:
  Verified terminal rendering with 80-column terminal emulation.

---

### Error 22: GGUFReader `close()` No-Op Causing Memory Leak & `SIGKILL 137` OOM on Multi-Model Batches
- **Symptom**:
  Running sequential deep inspections across multiple local checkpoints (or diffing followed by batch inspection within the same Python process) resulted in monotonic memory growth ($>9\text{GB}$ VmSize / active mapped pages), culminating in `Command exited with code 137` (`SIGKILL` by Linux OOM killer).
- **Root Cause**:
  `GGUFReader.close()` was implemented as a no-op (`pass`). Each GGUF file opened by `_GGUFReader(path)` established an internal `mmap` (`_reader.data._mmap`) spanning the full size of the file (e.g. 1GB–4GB). Because `close()` failed to close this mmap object and retained references in `_tensor_map`, the operating system retained memory mappings across all sequentially inspected models, eventually exhausting system RAM and swap.
- **Fix**:
  1. Updated `GGUFReader.close()` to explicitly close the underlying memory map:
     ```python
     if hasattr(self, "_reader") and self._reader is not None:
         if hasattr(self._reader, "data") and hasattr(self._reader.data, "_mmap"):
             with contextlib.suppress(Exception):
                 self._reader.data._mmap.close()
         self._reader = None
         self._tensor_map.clear()
         import gc
         gc.collect()
     ```
  2. Verified that `inspect_model()` in a `with` context immediately unmaps the checkpoint file upon completion.
- **Regression Test**:
  `tests/readers/test_readers.py` suite (all 16 tests passing) and multi-model benchmark suite execution.

---

### Error 23: Massive Tensor Vocabulary Embedding Memory Spikes in Single-Model Inspection
- **Symptom**:
  Running `inspect_model()` on models with large vocabularies (e.g. `Qwen3.5-4B-3D-Coder` with 248,320 vocabulary entries $\times$ 2,560 hidden size = 635,699,200 parameters) caused sudden memory allocation spikes, swap exhaustion, and `SIGKILL 137` by the OS OOM killer.
- **Root Cause**:
  1. Materializing the single `token_embd.weight` tensor in float32 required an immediate 2.425 GB allocation.
  2. In `_compute_outlier_score()`, calling `flat = np.abs(w.ravel())` duplicated the full 635M float32 array into a second 2.425 GB buffer, creating an immediate $>4.85\text{GB}$ memory spike.
  3. In `_count_dead_neurons()`, computing full 2D vector norms across 635M elements generated large intermediate reduction arrays.
- **Fix**:
  1. Introduced a massive-tensor chunked streaming inspection pathway in `src/wdiff/engine/inspect.py` for tensors exceeding 20M parameters. For large GGUF embeddings, rows are sliced in 4,096-token chunks (32–40 MB) directly from the underlying data, dequantized locally, and accumulated in `float64` for exact Frobenius norm, mean, variance, dead-neuron, and outlier metrics without materializing the 2.4 GB tensor.
  2. Rewrote `_compute_outlier_score()` to compute `max_val` scalar-wise (`max(abs(min), abs(max))`) and subsample row/column strides *before* applying `np.abs()`, reducing sample memory from 2.4 GB to 400 KB.
  3. Chunked `_count_dead_neurons()` in 4,096-row blocks for matrices exceeding 20M elements.
- **Regression Test**:
  Successfully inspected `Qwen3.5-4B-3D-Coder` (4.2B parameters) in 86.10s under strict memory boundaries; all 10 tests in `tests/engine/test_inspect.py` passing.



