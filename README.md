# `wdiff` 🔬
### *git diff, but for neural network weights.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streaming heap](https://img.shields.io/badge/streaming%20heap-bounded%20by%20largest%20tensor-brightgreen.svg)](#-real-world-benchmark-results)
[![Tests: 207 Passed](https://img.shields.io/badge/tests-207%20passed-success.svg)](tests/)

**`wdiff`** is an ultra-fast, memory-bounded CLI and visual differential analyzer for Large Language Model checkpoints (`.safetensors`, `.gguf`, and sharded indices).

It lets you inspect, measure, translate, and visualize model weights — detecting fine-tuning shifts, LoRA adapters, vocabulary expansions, safety abliterations, dead neurons, and quantization risks — **on consumer hardware with bounded streaming memory (peak anonymous heap scales with the largest tensor — measured ≈1.1 GB for a 235 MB tensor pair; see [benchmark results](#-real-world-benchmark-results) for process RSS on multi-GB checkpoints)**.

> ⚠️ **Name collision notice**: the package name `wdiff` on PyPI belongs to an unrelated
> Spanish spelling analyzer — **this project is not published there yet**. Until it is
> released under a distinct name, install directly from GitHub (below). Do not use
> `pip install wdiff` / `uvx wdiff`; they will install the wrong tool.

```bash
# Instant execution straight from GitHub with uv (no install or venv required)
uvx --from git+https://github.com/Aayushbankar/wdiff wdiff base-model.gguf finetuned-model.gguf

# Or standard pip install from GitHub
pip install "git+https://github.com/Aayushbankar/wdiff.git"

# Compare two models with SVD rank analysis and semantic concept decoding
wdiff modelA.safetensors modelB.gguf --deep --lens --report diff_report.html

# Inspect a single model for dead neurons and quantization risks
wdiff inspect model.gguf
```

---

## ⚡ Key Capabilities

### 1. Dual-Model Differential Autopsy
Compare any two checkpoints across HuggingFace SafeTensors and GGUF (`Q4_K_M`, `Q5_K_S`, `Q8_0`, etc.):
- **Frobenius Delta ($\Delta F$) & Cosine Similarity**: Quantifies exact weight magnitude shift and directional alignment.
- **Random Matrix Theory & Truncated SVD (`--deep`)**: Distinguishes between **Stochastic SGD noise**, **Low-Rank Adapters (LoRA/DPO)**, and **Surgical Rank-1 Ablations** (direction removal).
- **Semantic Logit Lens (`--lens`)**: Multiplies the primary singular vector ($\sigma_1$) through the model's unembedding matrix ($W_U$) to decode which plain-English concepts the layer promotes or suppresses.
- **RoPE Coordinate Parity**: Automatically de-interleaves GGML rotational coordinates to match HuggingFace split-half layout, eliminating false coordinate divergence.
- **Partial Vocabulary Slicing**: Automatically detects single-axis vocabulary expansions (e.g., 128k → 129k tokens) and computes metrics on the shared prefix while recording expansion telemetry.
- **Virtual LoRA Folding (`--lora`)**: Compares a base model against a virtual LoRA adapter ($W_0 + \alpha \Delta W$) on the fly without merging files to disk.

### 2. Single-Model Deep Inspection
Run forensic health checks on any checkpoint with zero GPU requirement:
- **Dettmers Outlier Scoring**: Measures feature outlier spikes ($(\max|w| - \text{median}) / \text{MAD}$) to assess 4-bit quantization collapse risk.
- **Dead Neuron Detection**: Identifies rows and columns with L2 norm $< 10^{-7}$.
- **Architecture Auto-Detection**: Recognizes LLaMA, Qwen2, Gemma, Gemma2, Phi-3, and GGUF topologies.
- **Interactive Visual Reports (`--report`)**: Generates self-contained HTML files containing interactive SVG profile charts, KPI summary cards, and filterable search tables.

---

## 🖥️ Terminal UI Preview

```
╭────────────────────────────────── Summary ───────────────────────────────────╮
│ Models: LFM3-1.7B-Q4_K_M.gguf (gguf) vs LFM3-1.7B-Uncensored-AD-Q4_K_M.gguf  │
│ Duration: 33.35s | Matched: 287/287 tensors (100%)                           │
│ Mean ΔF: 0.042027 | Max ΔF: 0.115392 | Min Cosine: 0.993328                  │
╰──────────────────────────────────────────────────────────────────────────────╯

┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳
┃ Layer            ┃      Shape ┃   Rel. ΔF ┃ Cosine Sim ┃  Max |Δ| ┃   Rank   ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇
│ blk.11.ffn_down  │     2048 × │  ░░░░░░░░ │     0.9954 │   0.0629 │   s=3    │
│ weight           │       7680 │    0.0962 │            │          │  e=1.3   │
│ blk.12.ffn_down  │     2048 × │  ░░░░░░░░ │     0.9944 │   0.0505 │   s=16   │
│ weight           │       7680 │    0.1062 │            │          │  e=1.4   │
└──────────────────┴────────────┴───────────┴────────────┴──────────┴──────────┴

╭──────────────────────── Logit Lens Concept Decoding ─────────────────────────╮
│ Layer: blk.11.ffn_down.weight | σ₁: 4.0908 | Energy: 98.62%                   │
│   Promoted (+): dining (+0.10), granite (+0.09), clic (+0.09)                │
│   Suppressed (-): unfortunately (-0.10), Mot (-0.10), unc (-0.09)            │
╰──────────────────────────────────────────────────────────────────────────────╯

Anomalies Detected:
  ⚠ Rank-1 Ablation Detected: blk.11.ffn_down.weight (98% confidence)
  ⚠ Rank-1 Ablation Detected: blk.12.ffn_down.weight (96% confidence)
```

---

## 📊 Real-World Benchmark Results

Tested across **19.6 Billion parameters** on a standard consumer CPU (Linux x86_64, memory bounded):

| Model Architecture | Size | Parameter Count | Tensors | Time | Throughput | Dead Neurons | Outlier Layers |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SmolLM2-360M-Instruct (Q8_0)** | 368.5 MB | 361,821,120 | 290 | 7.02s | **51.5 M param/s** | 0 | 219 |
| **SmolLM2-1.7B-Instruct (Q4_K_M)** | 1,006.7 MB | 1,711,376,384 | 218 | 28.05s | **61.0 M param/s** | 0 | 192 |
| **Qwen2.5-1.5B-Instruct (Q4_K_M)** | 1,065.6 MB | 1,777,088,000 | 339 | 36.10s | **49.2 M param/s** | 1 | 282 |
| **Qwen2.5-Coder-3B-Instruct (Q4_K_M)** | 1,840.5 MB | 3,085,938,688 | 434 | 60.07s | **51.4 M param/s** | 0 | 337 |
| **Qwen3.5-4B-3D-Coder (Q4_K_M)** | 2,583.3 MB | 4,205,751,296 | 426 | 81.59s | **51.5 M param/s** | 0 | 296 |
| **K2-Horizon-4B (Q4_K_M)** | 3,010.4 MB | 5,058,255,360 | 327 | 92.35s | **54.8 M param/s** | 741 | 240 |
| **LFM3-1.7B Base (Q4_K_M)** | 1,048.7 MB | 1,728,801,600 | 287 | 34.92s | **49.5 M param/s** | 60 | 228 |
| **LFM3-1.7B Uncensored (Q4_K_M)** | 1,079.8 MB | 1,728,801,600 | 287 | 36.06s | **47.9 M param/s** | 60 | 228 |
| **SD-v1.5 VAE Decoder (SafeTensors)** | 9.3 MB | 2,445,063 | 134 | 0.11s | **21.3 M param/s** | 0 | 50 |

*(See full methodology and findings in [`benchmarks/BENCHMARK_REPORT.md`](benchmarks/BENCHMARK_REPORT.md))*

---

## 🚀 Quickstart

### Installation

> ⚠️ PyPI's `wdiff` name is taken by an unrelated tool (see notice above). Use GitHub installs until this project publishes under a distinct name.

```bash
# Using uv (recommended, instant execution from GitHub)
uvx --from git+https://github.com/Aayushbankar/wdiff wdiff --help

# Using pip from GitHub
pip install "git+https://github.com/Aayushbankar/wdiff.git"

# Clone for local development
git clone https://github.com/Aayushbankar/wdiff.git
cd wdiff
uv sync --extra dev   # or: pip install -e ".[dev]"
```

### CLI Command Guide

```bash
# 1. Compare two models (auto-routes to diff)
wdiff modelA.safetensors modelB.gguf

# 2. Deep forensic diff with SVD and semantic logit lens
wdiff modelA.gguf modelB.gguf --deep --lens

# 3. Inspect a single checkpoint for health and quantization readiness
wdiff inspect model.gguf

# 4. Filter specific layers using regex
wdiff modelA.gguf modelB.gguf -f "blk\.1[0-4]\.ffn_down"

# 5. Generate a standalone interactive HTML report
wdiff modelA.gguf modelB.gguf --report diff_report.html

# 6. Apply a virtual LoRA adapter to modelA before comparing
wdiff base.safetensors target.safetensors --lora adapter.safetensors --lora-scale 1.0

# 7. Output machine-readable JSON for CI/CD pipelines
wdiff modelA.gguf modelB.gguf --json > diff.json
```

---

## 🔬 Mathematical Metrics Reference

| Metric | Formula | Interpretation |
| :--- | :--- | :--- |
| **Relative Frobenius Delta** | $\frac{\|W_A - W_B\|_F}{\|W_A\|_F}$ | Total relative magnitude of change (0.00% = identical). |
| **Cosine Similarity** | $\frac{\text{vec}(W_A) \cdot \text{vec}(W_B)}{\|W_A\|_F \|W_B\|_F}$ | Directional alignment of weights in high-dimensional space. |
| **Dettmers Outlier Score** | $\frac{\max|W| - \text{median}(|W|)}{\text{MAD} / 0.6745 + 10^{-12}}$ | Feature outlier magnitude; scores $>15.0$ predict quantization collapse. |
| **Effective Rank (`--deep`)** | $\exp\left(-\sum p_i \ln p_i\right)$ | Entropy-based dimension count of the weight perturbation. |
| **Signal Rank (`--deep`)** | $\sum \mathbb{I}(\sigma_i > s_+)$ | Number of singular values exceeding the Marchenko-Pastur bulk noise edge. |
| **Logit Lens Projection** | $W_U \cdot u_1$ | Projects the primary singular vector into vocabulary logit space. |

---

## 🛠️ Architecture & Documentation

- [**Technical Architecture & Streaming Buffers**](docs/ARCHITECTURE.md): Zero-copy `mmap`, bounded synchronous streaming buffer (optional threaded prefetch), and format abstraction.
- [**Engineering Errors & Fixes Ledger**](docs/ERRORS_AND_FIXES.md): Forensic record of all 23 bugs, edge cases, and architectural fixes.
- [**Research & Forensic Demos**](docs/RESEARCH_DEMOS.md): Mathematical proofs of safety abliteration and successor model delta geometry.
- [**Development Roadmap**](docs/ROADMAP.md): Complete milestone ledger and future native compilation plans.

---

## ⚖️ License

MIT License. See [LICENSE](LICENSE) for details.
