# `wdiff` Technical Architecture

This document describes the architectural design, data pipelines, memory bounding strategies, and mathematical foundations of `wdiff`.

---

## 1. High-Level Architecture

`wdiff` is organized into three decoupled subsystems:

```
┌─────────────────────────────────────────────────────────────────┐
│                          1. CLI & UI                            │
│  ┌───────────────────────┐  ┌────────────────────────────────┐  │
│  │ Rich Terminal Output  │  │ Interactive HTML Report Engine │  │
│  └───────────┬───────────┘  └────────────────┬───────────────┘  │
└──────────────┼───────────────────────────────┼──────────────────┘
               │                               │
┌──────────────▼───────────────────────────────▼──────────────────┐
│                   2. Streaming Engine Core                      │
│                                                                 │
│   ┌────────────────────────────────────────────────────────┐    │
│   │               Ring Buffer (Depth = 2, sync default)    │    │
│   │                                                        │    │
│   │   [ Slot 0: Processing ]    [ Slot 1: Prefetch (opt) ] │    │
│   │   • Thread 2: Math Worker      • Thread 1: I/O Worker  │    │
│   │     - Truncated SVD              - mmap offset seek    │    │
│   │     - Frobenius Delta            - gguf dequantize     │    │
│   │     - Cosine Similarity          - float32 cast        │    │
│   └───────────────────────────▲────────────────────────────┘    │
│                               │                                 │
└───────────────────────────────┼─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                 3. Format Abstraction Layer                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │   GGUF Reader    │  │Safetensors Reader│  │PyTorch (Plan) │  │
│  │  (gguf-py, mmap) │  │  (zero-copy mmap)│  │ (Phase 5)     │  │
│  └──────────────────┘  └──────────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Format Abstraction Layer

Different LLM formats package weights with varying layouts and quantization schemes. `wdiff` unifies them under an abstract `TensorReader` protocol:

```python
class TensorReader(Protocol):
    def tensor_names(self) -> list[str]:
        """Returns ordered list of all tensor keys in the file header."""
        ...
        
    def get_tensor(self, name: str) -> np.ndarray:
        """Fetches a single tensor, dequantizes if necessary, and returns float32."""
        ...
        
    def tensor_info(self, name: str) -> TensorInfo:
        """Returns shape, dtype, and on-disk byte footprint without loading weights."""
        ...
```

### Format Implementations

1. **Safetensors (`safetensors.safe_open`):**
   - The file header is a lightweight JSON string at the beginning of the file specifying byte ranges `[begin, end]`.
   - `safe_open` uses OS memory mapping (`mmap`). Accessing a single tensor reads *only* the specific byte range into RAM without touching the rest of the model file.
2. **GGUF (`gguf.GGUFReader`):**
   - The GGUF header provides a table of contents containing name, shape, quantization type (`GGMLQuantizationType`), and 64-bit byte offsets.
   - For unquantized tensors (`F32`, `F16`, `BF16`), zero-copy views are constructed via `np.frombuffer`.
   - For block-quantized tensors (`Q4_K_M`, `Q5_K_S`, `Q8_0`, `IQ*`), the reader dynamically applies `gguf.quants.dequantize()` on the individual tensor chunk, converting it to `np.float32`.
3. **PyTorch Checkpoints (`torch.load(..., mmap=True)`):**
   - *Planned for a future milestone (Phase 5).* Currently, checkpoints should be converted to SafeTensors (e.g. via Hugging Face `save_pretrained` or conversion scripts) or GGUF before analysis. Native support for legacy `.pt` and `.bin` weights will be introduced in a future update.

---

## 3. Paired Ring Buffer (Double-Buffering, Synchronous Default)

### The Bottleneck Problem
In a naive sequential implementation:
$$\text{Total Time} = \sum_{i=1}^{N} \left( T_{\text{I/O\_read}}(i) + T_{\text{dequant}}(i) + T_{\text{math}}(i) \right)$$
The CPU sits completely idle while waiting for disk reads, and the disk sits idle while the CPU performs matrix operations.

### The Ring Buffer Architecture
`wdiff` bounds memory usage using a synchronous paired streaming buffer pattern implemented via `_run_paired_buffer()` in `src/wdiff/engine/diff.py` and `ring_buffer.py`. Rather than holding entire model state in RAM:

1. **Paired Ingestion:** Iterates layer-by-layer matching keys across models via canonical name resolution.
2. **Synchronous Execution:** In each slot step, the pair of corresponding weights is loaded via `mmap` (and dequantized on demand if GGUF), evaluated immediately for elementwise, Frobenius, and spectral metrics, and promptly dereferenced to return memory back to the OS allocator.
3. **Pipelined Asynchrony (Optional):** While `ring_buffer.py` provides threaded consumer/producer slots for decoupled I/O prefetching, the engine runs synchronously by default in `_run_paired_buffer()` to avoid Python GIL overhead, thread context-switching latency, and lock contention during memory-mapped reads.

### Memory Bounding Guarantee
- The paired streaming buffer is a 2-slot queue: at most two tensor pairs are resident in ring slots at once ($2 \times \max(\text{TensorSize})$; e.g. $2 \times 235\text{MB} \approx 470\text{MB}$ for a FP32 `14336 × 4096` projection).
- Metric computation streams in $\sim$16.7M-element chunks with FP64 accumulation (`compute_all_metrics`), adding a transient working set of $\approx 0.7\text{GB}$ independent of tensor count; the deep path additionally materializes one full difference tensor ($+1 \times T$) per layer, which overlaps memory already freed by the chunked metrics pass (observed peak delta only $+25\text{MB}$ at $T = 235\text{MB}$).
- **Measured** (2026-11-09, Linux x86-64, paired `14336 × 4096` FP32 tensors, $T = 235\text{MB}$): peak anonymous heap = **1,143 MB** streaming, **1,168 MB** `--deep`. Process RSS additionally includes mmap'd checkpoint pages (both files fully resident adds $\approx 2 \times \text{file size}$).
- Peak anonymous heap is invariant to total model parameter count (whether 7B or 70B); it is bounded by the **largest single tensor** — empirically $\text{peak} \approx 2T + 0.7\text{GB}$ (streaming; `--deep` measured $+25\text{MB}$ at this shape). A largest tensor of $0.5\text{GB}$ FP32 alone implies $\approx 1.7\text{GB}$ heap before any mmap pages — e.g. a 1.7B GGUF pair with a 0.5 GB `lm_head` OOMs on a host with only 1.7 GB free.

---

## 4. Mathematical Metrics & Formulas

For two tensors $W_A, W_B \in \mathbb{R}^{m \times n}$ representing the same layer across two models:

### 1. Relative Frobenius Change ($\Delta_F$)
Measures the overall magnitude of the parameter shift normalized by the base model's norm:
$$\Delta_F = \frac{\|W_A - W_B\|_F}{\|W_A\|_F} = \frac{\sqrt{\sum_{i,j} (A_{ij} - B_{ij})^2}}{\sqrt{\sum_{i,j} A_{ij}^2}}$$
- $\Delta_F = 0.0$: Weights are identical.
- $\Delta_F \ll 0.05$: Surgical fine-tuning or low-rank adaptation.
- $\Delta_F > 0.50$: Substantial architectural retraining or full-model replacement.

### 2. Cosine Similarity ($\cos \theta$)
Flattens matrices into 1D vectors $v_A, v_B \in \mathbb{R}^{mn}$ to measure directional alignment:
$$\cos \theta = \frac{v_A \cdot v_B}{\|v_A\|_2 \|v_B\|_2}$$
- $\cos \theta = 1.0$: Exact parallel directional orientation.
- $\cos \theta \approx 0.0$: Orthogonal feature spaces (catastrophic forgetting or re-initialization).

### 3. Truncated SVD, Energy Ratios & Effective Rank (`--deep`)
For 2D weight matrices, the difference matrix is:
$$\Delta W = W_A - W_B \in \mathbb{R}^{m \times n}$$
Using fast randomized SVD or `scipy.sparse.linalg.svds`, we compute the top-$k$ singular values ($k = \min(64, \min(m, n) - 1)$):
$$\Delta W \approx \sum_{i=1}^{k} \sigma_i u_i v_i^T, \quad \sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_k \ge 0$$

From the singular value spectrum, we compute:
1. **Primary Direction Energy Ratio ($E_1$):**
   $$E_1 = \frac{\sigma_1^2}{\sum_{i=1}^k \sigma_i^2}$$
   Measures the fraction of total variance / perturbation energy concentrated along the dominant singular vector. If $E_1 > 0.90$, the intervention is practically **Rank-1** (the hallmark of refusal abliteration).
2. **Stable Rank ($r_{\text{stable}}$):**
   $$r_{\text{stable}} = \frac{\|\Delta W\|_F^2}{\|\Delta W\|_2^2} = \frac{\sum_{i=1}^k \sigma_i^2}{\sigma_1^2}$$
   Continuous, noise-resistant proxy for algebraic rank.
3. **Effective Rank ($r_{\text{eff}}$):**
   Treating normalized squared singular values as an energy probability distribution:
   $$p_i = \frac{\sigma_i^2}{\sum_{j=1}^k \sigma_j^2}, \quad p_i > 0$$
   Effective rank is the exponential of the Shannon spectral entropy:
   $$r_{\text{eff}} = \exp\left( -\sum_{i=1}^k p_i \ln p_i \right)$$
   Measures the true information-theoretic degrees of freedom in the weight update.

---

## 5. Standalone Binary Distribution Strategy

For experimenters and CI/CD pipelines that forbid Python runtimes:

1. **`uvx` Instant Execution (Primary):**
   `uvx wdiff modelA.gguf modelB.gguf` downloads the wheel into an isolated, ephemeral cache and executes immediately in under 1 second.
2. **PyInstaller / Nuitka Single-File Bundles (Phase 2):**
   Self-contained native executables (`wdiff-linux-x86_64`, `wdiff-darwin-arm64`) generated via GitHub Actions CI containing the embedded Python runtime + NumPy + C-dequantizers.
3. **Rust Native Crate `wdiff-rs` (Phase 3):**
   Standalone binary utilizing `clap-rs` for CLI arguments, `safetensors-rs` for zero-copy safetensors parsing, and FFI to `ggml.c` for GGUF parsing and SIMD dequantization.
