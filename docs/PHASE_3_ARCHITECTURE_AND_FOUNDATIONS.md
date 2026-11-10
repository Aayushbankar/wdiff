# Phase 3 Architecture: Production Systems & Semantic Intelligence

This document specifies the technical design, research foundations, and implementation milestones for **Phase 3** of `wdiff`.

---

## 1. Technical Audit of v0.2.0 Baseline

Prior to Phase 3 development, all foundational subsystems from Phase 1 and Phase 2 were audited for performance, memory bounding, and edge-case resilience across 112 automated tests:

| Subsystem | Location | Audit Score | Capabilities Validated | Production Limitations Identified |
| :--- | :--- | :---: | :--- | :--- |
| **Safetensors Reader** | `src/wdiff/readers/safetensors_reader.py` | 9.5 / 10 | Zero-copy mmap, `get_slice()` metadata inspection with 0MB RAM footprint. | Single-file only; raises error on directory or `model.safetensors.index.json`. |
| **GGUF Reader** | `src/wdiff/readers/gguf_reader.py` | 8.0 / 10 | Pure NumPy dequantization (`Q8_0`, `Q4_K`), reverse C-order logical shape calculation. | Pure-Python NumPy dequantization throughput (~50s for 10 layers on 1.7B); upstream `gguf` bug on 4D conv layers. |
| **Canonical Normalizer** | `src/wdiff/readers/naming.py` | 9.0 / 10 | Regex heuristic auto-detection for LLaMA, Qwen2, Gemma (2-norm), Gemma2 (4-norm), Phi-3 (fused), GGUF, tied embeddings aliasing. | Does not invert RoPE SIMD coordinate permutations; does not slice fused `qkv_proj`. |
| **Streaming Engine** | `src/wdiff/engine/ring_buffer.py`, `diff.py` | 9.5 / 10 | 2-slot prefetch queue, strictly bounded memory (<35MB streaming math, ~300MB on real models). | Requires exact shape match; skips metric computation on expanded vocabularies. |
| **Deep Spectral Engine** | `src/wdiff/engine/spectral.py` | 9.5 / 10 | Halko-Martinsson-Tropp Randomized SVD (9.2× faster than ARPACK on CPU), Marčenko-Pastur noise floor via MAD, Grassmannian angles, Quantum Uhlmann fidelity (0.05ms). Conv2d flattening. | Truncated rank fixed at $k=16$; does not adapt dynamically based on matrix aspect ratio. |
| **Taxonomy Classifier** | `src/wdiff/engine/taxonomy.py` | 9.0 / 10 | 6-badge automated classification (`UNMODIFIED`, `RANK1_ABLATION`, `LOW_RANK_ADAPTER`, `SPARSE_UPDATE`, `FULL_RANK_CPT`, `STOCHASTIC_NOISE`). | Fixed heuristic thresholds ($E_1 > 0.90$, sparsity $\ge 0.65$). |
| **HTML Report Visualizer** | `src/wdiff/display/report.py` | 9.0 / 10 | 100% self-contained single-file HTML, zero external CDN dependencies, SVG delta chart, vanilla JS table search/filtering. | Single-row division-by-zero fixed; tooltips pending enhancement. |
| **LoRA Engine** | `src/wdiff/engine/lora.py` | 8.5 / 10 | Streams base weights against LoRA adapter $B \times A$ deltas on the fly without materializing merged model to disk; rank detection. | Supports standard LoRA; does not support DoRA magnitude vectors or QLoRA NF4 base weights. |

---

## 2. Technical Debt & Prerequisite Refactoring

Phase 3 implementation resolves four items of technical debt:

1. **CLI Subcommand & Positional Routing**:
   [`src/wdiff/cli.py`](file:///mnt/work/projects/wdiff/src/wdiff/cli.py) currently requires `model_a [model_b]`. The parser will be refactored to support single-model inspection (`wdiff inspect <model>`) without requiring dummy secondary arguments.
2. **Directory & Index Reader Dispatching**:
   [`src/wdiff/readers/__init__.py`](file:///mnt/work/projects/wdiff/src/wdiff/readers/__init__.py) factory `get_reader()` will be extended to detect directories containing `*.index.json` or direct paths to `model.safetensors.index.json`.
3. **Rigid Shape Matching**:
   [`src/wdiff/engine/diff.py`](file:///mnt/work/projects/wdiff/src/wdiff/engine/diff.py) lines 107–118 abort on any `shape_a != shape_b`. This will be updated to detect single-axis vocabulary mismatches and compute metrics on the overlapping common prefix.
4. **RoPE SIMD Coordinate Swaps**:
   In cross-format comparisons (HuggingFace Safetensors ↔ llama.cpp GGUF), `q_proj` and `k_proj` will undergo automatic coordinate de-interleaving prior to metric calculation.

---

## 3. Phase 3 Architecture: The 5 Engineering Pillars

```
                                    PHASE 3 ARCHITECTURE
                                             │
         ┌───────────────────────────────────┼───────────────────────────────────┐
         ▼                                   ▼                                   ▼
[Pillar 1: Single-Model Inspect]   [Pillar 2: Sharded SafeTensors]    [Pillar 3: RoPE Normalizer]
`wdiff inspect <model>`            `model.safetensors.index.json`     Invert SIMD permutations on
Dead neurons, outlier channels,    LRU handle cache for 70B & 405B    q/k_proj for bit-for-bit parity
and spectral condition numbers.    maintaining <35MB memory bound.    between PyTorch and GGUF.
         │                                   │                                   │
         └───────────────────────────────────┼───────────────────────────────────┘
                                             │
         ┌───────────────────────────────────┴───────────────────────────────────┐
         ▼                                                                       ▼
[Pillar 4: Vocabulary Expansion Slicing]                            [Pillar 5: Semantic Logit Lens]
Common-prefix slicing for vocab mismatches                          Project FFN down_proj delta singular
(128k -> 129k) + isolating new token rows.                          vectors through lm_head to extract concepts.
```

---

## 4. Deep-Dive: Specifications, Research & Mathematical Mechanisms

### Pillar 1: Single-Model Deep Inspection (`wdiff inspect <model>`)
* **Core Problem**: Users frequently evaluate a single model file downloaded from the web without a reference checkpoint. They need to verify structural integrity, parameter utilization, and quantization vulnerability without loading PyTorch.
* **Research Applied**:
  * **Dettmers et al. (NeurIPS 2022)** (*LLM.int8()*):
    $$\text{Outlier Ratio} = \frac{\max |W_{ij}| - \text{median}(|W|)}{\text{MAD}(W)}$$
    Emergent outlier features ($>6\sigma$) concentrate in attention projections. Layers with Outlier Ratio $>15$ are flagged as high risk for post-training quantization failure.
  * **Spectral Regularity Indicators** (*implemented*): Effective rank (Shannon entropy of normalized squared singular-value energy), stable rank, primary-direction energy, and the Marčenko–Pastur noise-floor signal rank — all derived from the randomized SVD ($k=16$) — serve as correlation/structure indicators per the same self-regularization literature. A full-spectrum power-law fit $\alpha$ (Martin & Mahoney, JMLR 2021, $p(\lambda) \sim \lambda^{-\alpha}$) is **not implemented** in v0.3.0: it would require an exact full SVD per layer, which the randomized engine deliberately avoids. *(Struck during the 2026-11-09 documentation re-audit; the previous text described an unimplemented feature.)*
  * **Dead Neuron Detection**:
    Identifies rows/columns where $\|W_{i, :}\|_2 \le 10^{-7}$, reporting wasted model capacity.
* **Implementation Location**:
  `src/wdiff/engine/inspect.py` and CLI command `wdiff inspect <model>`.

---

### Pillar 2: Multi-File Sharded Checkpoint Support (`ShardedSafeTensorsReader`)
* **Core Problem**: All production models $\ge 7\text{B}$ are split into `model-00001-of-00004.safetensors` coordinated by `model.safetensors.index.json`.
* **Implementation Design**:
  * Create `src/wdiff/readers/sharded_reader.py` conforming to `TensorReader`.
  * `weight_map: dict[str, str]` maps canonical and raw tensor keys to shard filenames.
  * **LRU Shard Handle Pool**: Maintains a maximum of 2 active `safe_open` file handles simultaneously. When streaming through layer 0 to layer 31, handles are automatically rotated and closed.
  * **Memory Guarantee**: Streaming math memory remains strictly bounded $<35$MB even on a 405B parameter checkpoint across 80 shards.

---

### Pillar 3: RoPE Coordinate Normalization (HuggingFace ↔ GGUF Parity)
* **Core Problem**: PyTorch HuggingFace stores Rotary Position Embeddings using split-half pairs:
  $$x_{\text{rot}} = [-x_{d/2:], x_{:d/2}]$$
  GGML / llama.cpp permutes them into interleaved pairs for SIMD vector efficiency:
  $$x_{\text{rot}} = [-x_1, x_0, -x_3, x_2, \dots]$$
  Cross-format diffing of `q_proj` and `k_proj` yields false divergence ($\text{Cosine} \approx 0.05, \Delta F \approx 1.30$).
* **Research Applied**:
  * **Su et al. (2021)** (*RoFormer: Enhanced Transformer with Rotary Position Embedding*).
* **Implementation Design**:
  * In `src/wdiff/readers/naming.py` and `src/wdiff/engine/diff.py`:
  * Detect when comparing Safetensors vs GGUF on `attn_q` or `attn_k`.
  * For head dimension $d_{\text{head}}$ (e.g. 128), apply inverse permutation $P_{\text{de-interleave}}$ mapping interleaved coordinates back to split-half format prior to metric calculation.
  * Restores bit-for-bit parity ($\text{Cosine} = 1.0000$).

---

### Pillar 4: Partial Shape Matching & Vocabulary Expansion Slicing
* **Core Problem**: Fine-tuned multilingual and domain models append tokenizer tokens (e.g. 128,000 to 129,280 tokens). Currently flagged as `SHAPE_MISMATCH` and skipped.
* **Implementation Design**:
  * In `src/wdiff/engine/diff.py`:
  * Detect single-axis dimensional variance where $A \in \mathbb{R}^{V_A \times d}$ and $B \in \mathbb{R}^{V_B \times d}$ with $V_A \ne V_B$.
  * Let $V_{\text{common}} = \min(V_A, V_B)$.
  * Slice common prefix $A_{[:V_{\text{common}}, :]}$ and $B_{[:V_{\text{common}}, :]}$ through the ring buffer to evaluate metrics on shared tokens.
  * Isolate the extra rows $B_{[V_{\text{common}}:, :]}$ as `EXPANDED_VOCAB (N new tokens)` with dedicated norm reporting.

---

### Pillar 5: Semantic Concept Translation (The Logit Lens)
* **Core Problem**: Mathematical deltas like `Rel. ΔF = 0.0142` do not reveal what knowledge or capabilities were modified.
* **Research Applied**:
  * **Geva et al. (EMNLP 2021)** (*Transformer Feed-Forward Layers Are Key-Value Memories*):
    MLP down-projections emit semantic value vectors $v_i$ directly into the residual stream:
    $$h_{\text{out}} = \sum_{i=1}^{d_{\text{ffn}}} \sigma(k_i^T x) \cdot v_i$$
  * **nostalgebraist (2020)** (*interpreting GPT: the logit lens*):
    Residual stream vectors are mapped into vocabulary space via the unembedding head $W_U \in \mathbb{R}^{V \times d}$:
    $$\text{logits} = W_U \cdot v$$
* **Implementation Design**:
  * For modified FFN down-projection layers (`ffn_down`, `down_proj`):
  * Extract primary left singular vector $u_1 \in \mathbb{R}^{d_{\text{model}}}$ from Randomized SVD of $\Delta W_{\text{down}}$.
  * Project $u_1$ through unembedding matrix $W_U$:
    $$z = W_U \cdot u_1 \in \mathbb{R}^V$$
  * Extract top-5 token IDs and decode them using embedded tokenizer vocabularies into human strings:
    ```
    layer.24.ffn_down.weight: [Low-Rank Knowledge Edit]
      Decoded Concept Directions: ["medical", "diagnosis", "dosage", "symptoms"]
    ```

---

## 5. Phase 3 Milestone Execution Roadmap

| Milestone | Deliverable | Target Module | Verification Gate |
| :--- | :--- | :--- | :--- |
| **M3.1** | **Single-Model Deep Inspection** | `src/wdiff/engine/inspect.py`, `src/wdiff/cli.py` | `tests/engine/test_inspect.py` |
| **M3.2** | **Sharded SafeTensors Reader** | `src/wdiff/readers/sharded_reader.py` | `tests/readers/test_sharded.py` |
| **M3.3** | **RoPE De-Interleaving & Parity** | `src/wdiff/readers/naming.py`, `diff.py` | `tests/engine/test_rope_parity.py` |
| **M3.4** | **Partial Vocab Slicing** | `src/wdiff/engine/diff.py` | `tests/engine/test_partial_shape.py` |
| **M3.5** | **Semantic Logit Lens** | `src/wdiff/engine/lens.py`, `cli.py`, `report.py`| `tests/engine/test_logit_lens.py` |
