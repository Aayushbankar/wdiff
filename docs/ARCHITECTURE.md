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
│   │               Pipelined Ring Buffer (Depth = 2)        │    │
│   │                                                        │    │
│   │   [ Slot 0: Processing ]       [ Slot 1: Prefetching ] │    │
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
│  │   GGUF Reader    │  │Safetensors Reader│  │PyTorch Reader │  │
│  │  (gguf-py, mmap) │  │  (zero-copy mmap)│  │ (mmap=True)   │  │
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
   - Supported for legacy `.pt` and `.bin` checkpoints using PyTorch’s built-in lazy mmap reader.

---

## 3. Pipelined Ring Buffer (Double-Buffering)

### The Bottleneck Problem
In a naive sequential implementation:
$$\text{Total Time} = \sum_{i=1}^{N} \left( T_{\text{I/O\_read}}(i) + T_{\text{dequant}}(i) + T_{\text{math}}(i) \right)$$
The CPU sits completely idle while waiting for disk reads, and the disk sits idle while the CPU performs matrix operations.

### The Ring Buffer Solution
`wdiff` overlaps I/O and compute using a 2-slot circular buffer managed across two worker threads:

```python
class RingBufferSlot:
    name: str
    tensor_a: np.ndarray  # float32
    tensor_b: np.ndarray  # float32
    state: Literal["EMPTY", "LOADING", "READY", "PROCESSING"]
```

1. **Thread 1 (I/O Prefetch Worker):**
   - Reads tensor $N+1$ from Model A and Model B via `mmap`.
   - Performs dequantization (if GGUF).
   - Writes the pair into `Slot[1]`.
2. **Thread 2 (Math Worker):**
   - Simultaneously consumes `Slot[0]` (Tensor $N$).
   - Computes Frobenius norm, Cosine similarity, and Truncated SVD.
   - Releases `Slot[0]` memory back to the OS.
3. **Buffer Advance:**
   - Once Math finishes Slot 0 and I/O finishes Slot 1, the slots rotate in $O(1)$ time.

### Memory Bounding Guarantee
- For an 8B parameter model, the largest single tensor (e.g., intermediate projection `14336 × 4096` in FP16) is $\approx 117\text{MB}$.
- In FP32 dequantized form, this is $\approx 235\text{MB}$.
- With a ring buffer of depth 2:
  $$\text{Peak RAM} \le 2 \times (2 \times \max(\text{TensorSize})) \approx 4 \times 117\text{MB} \approx 470\text{MB}$$
- Peak RAM is strictly invariant to total model parameter count (whether 7B or 70B); it is bounded solely by the largest individual layer matrix.

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

### 3. Truncated SVD & Effective Rank (`--deep`)
For 2D weight matrices, the difference matrix is:
$$\Delta W = W_A - W_B \in \mathbb{R}^{m \times n}$$
Using `scipy.sparse.linalg.svds`, we compute the top-$k$ singular values ($k = \min(64, \min(m, n) - 1)$):
$$\Delta W \approx \sum_{i=1}^{k} \sigma_i u_i v_i^T, \quad \sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_k \ge 0$$

From the singular value spectrum, we compute:
1. **Primary Direction Ratio ($R_1$):**
   $$R_1 = \frac{\sigma_1}{\sum_{i=1}^k \sigma_i}$$
   If $R_1 > 0.95$, the change is practically **Rank-1** (characteristic of refusal abliteration).
2. **Effective Rank ($r_{\text{eff}}$):**
   $$r_{\text{eff}} = \frac{\left( \sum_{i=1}^k \sigma_i \right)^2}{\sum_{i=1}^k \sigma_i^2}$$
   Measures the effective dimensionality of the fine-tuning intervention.

---

## 5. Standalone Binary Distribution Strategy

For experimenters and CI/CD pipelines that forbid Python runtimes:

1. **`uvx` Instant Execution (Primary):**
   `uvx wdiff modelA.gguf modelB.gguf` downloads the wheel into an isolated, ephemeral cache and executes immediately in under 1 second.
2. **PyInstaller / Nuitka Single-File Bundles (Phase 2):**
   Self-contained native executables (`wdiff-linux-x86_64`, `wdiff-darwin-arm64`) generated via GitHub Actions CI containing the embedded Python runtime + NumPy + C-dequantizers.
3. **Rust Native Crate `wdiff-rs` (Phase 3):**
   Standalone binary utilizing `clap-rs` for CLI arguments, `safetensors-rs` for zero-copy safetensors parsing, and FFI to `ggml.c` for GGUF parsing and SIMD dequantization.
