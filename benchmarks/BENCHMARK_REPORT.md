# `wdiff` v0.3.0 Real-World Benchmark Report

**Execution Timestamp**: 2026-11-09 08:33:36 UTC
**Hardware Platform**: Linux x86_64, CPU (process peak RSS reported per run)

---

## 1. Executive Summary

`wdiff` was executed across **9 distinct local model checkpoints** spanning architectures from 360M to 4B parameters (SmolLM2, Qwen2.5, Qwen3.5, LFM3, K2-Horizon, and Stable Diffusion).

- **Peak Memory**: Process RSS on Benchmark 1 peaked at **4,302 MB** (dominated by mmap'd checkpoint pages plus the paired tensor heap). The tensor heap itself scales with the largest single tensor — measured ≈1.1 GB for a 235 MB tensor pair (see `docs/ARCHITECTURE.md` §Memory Bounding Guarantee).
- **Throughput**: Achieved up to **61.0 Million parameters/second** CPU inspection throughput (SmolLM2-1.7B; full table in §3).
- **Abliteration Forensic Detection**: Successfully isolated low-rank uncensored steerings on real model pairs (`LFM3-1.7B` vs `LFM3-1.7B-Uncensored`), quantifying rank-1 spectral dominance and decoding semantic concept directions.

---

## 2. Benchmark 1: Real-World Differential Autopsy (Base vs Uncensored)

**Model A (Base/Instruct)**: `/mnt/legion/experimental_stuff/idkv2/LFM3-1.7B-Q4_K_M.gguf`
**Model B (Uncensored)**: `/mnt/work/projects/wolvie/models/LFM3-1.7B-Uncensored-AD-Q4_K_M.gguf`

| Metric | Result |
| :--- | :--- |
| **Total Tensors Compared** | 287 |
| **Matched Tensors** | 287 |
| **Wall-Clock Duration** | **386.43 s** |
| **Peak Process RAM** | **4302.53 MB** |
| **Mean Frobenius Delta (ΔF)** | `0.042027` |
| **Max Frobenius Delta (ΔF)** | `0.115392` (`blk.13.shortconv.out_proj.weight`) |
| **Minimum Cosine Similarity** | `0.993328` |
| **Logit Lens Decoded Layers** | **24 down-projection layers** |

### Automated Taxonomy Classification Distribution

| Forensic Taxonomy Badge | Layer Count | Percentage |
| :--- | :--- | :--- |
| `Unmodified` | 134 | 46.7% |
| `Low-Rank Adapter (LoRA/DPO)` | 107 | 37.3% |
| `Stochastic SGD Noise` | 26 | 9.1% |
| `Surgical Rank-1 Ablation` | 17 | 5.9% |
| `Full-Rank Continual Learning` | 3 | 1.0% |

### Semantic Logit Lens Concept Directions (Sample Layers)

| Layer | σ₁ | Energy Ratio | Promoted Concepts (+) | Suppressed Concepts (-) |
| :--- | :--- | :--- | :--- | :--- |
| `blk.0.ffn_down.weight` | 0.2769 | 27.5% | ` finite`, ` fave`, `èĩ´`, ` ê³§` | `aru`, `osc`, `ë³Ħ`, ` nets` |
| `blk.1.ffn_down.weight` | 0.5541 | 34.3% | `orp`, ` gen`, `*\`, `gens` | `SK`, ` conform`, `ãĤ¹ãĤ¤`, ` throw` |
| `blk.10.ffn_down.weight` | 3.9336 | 98.9% | `target`, `invalid`, `æī¹`, `alt` | `mani`, ` synchron`, ` chose`, ` prest` |
| `blk.11.ffn_down.weight` | 4.0908 | 98.6% | ` dining`, `ãĥ¬ãĥĥãĤ¹`, `zi`, ` granite` | ` unfortunately`, `Mot`, `mot`, `âĢĳt` |
| `blk.12.ffn_down.weight` | 3.9143 | 98.8% | ` unfortunately`, ` Unfortunately`, `Unfortunately`, ` impro` | ` Events`, ` viz`, ` Spin`, ` Bin` |
| `blk.13.ffn_down.weight` | 3.8146 | 98.0% | ` impossible`, ` neither`, ` prohibited`, ` impossibility` | `ä»¥ä¸ĭ`, `Ã¼g`, ` Round`, ` Bin` |

---

## 3. Benchmark 2: Multi-Model Deep Inspection & Outlier Profiling

| Model Architecture | Size (MB) | Params | Tensors | Time (s) | Throughput (M param/s) | Dead Neurons | Outlier Layers | Quant Readiness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SmolLM2-360M-Instruct (Q8_0)** | 368.5 | 361,821,120 | 290 | 7.02s | **51.52** | 0 | 219 | `POOR (High risk of quantization collapse)` |
| **SmolLM2-1.7B-Instruct (Q4_K_M)** | 1,006.7 | 1,711,376,384 | 218 | 28.05s | **61.02** | 0 | 192 | `POOR (High risk of quantization collapse)` |
| **Qwen2.5-1.5B-Instruct (Q4_K_M)** | 1,065.6 | 1,777,088,000 | 339 | 36.1s | **49.23** | 1 | 282 | `POOR (High risk of quantization collapse)` |
| **Qwen2.5-Coder-3B-Instruct (Q4_K_M)** | 1,840.5 | 3,085,938,688 | 434 | 60.07s | **51.37** | 0 | 337 | `POOR (High risk of quantization collapse)` |
| **Qwen3.5-4B-3D-Coder (Q4_K_M)** | 2,583.3 | 4,205,751,296 | 426 | 81.59s | **51.55** | 0 | 296 | `POOR (High risk of quantization collapse)` |
| **K2-Horizon-4B (Q4_K_M)** | 3,010.4 | 5,058,255,360 | 327 | 92.35s | **54.77** | 741 | 240 | `POOR (High risk of quantization collapse)` |
| **LFM3-1.7B (Q4_K_M)** | 1,048.7 | 1,728,801,600 | 287 | 34.92s | **49.5** | 60 | 228 | `POOR (High risk of quantization collapse)` |
| **LFM3-1.7B-Uncensored (Q4_K_M)** | 1,079.8 | 1,728,801,600 | 287 | 36.06s | **47.94** | 60 | 228 | `POOR (High risk of quantization collapse)` |
| **SD-v1.5 VAE Decoder (SafeTensors)** | 9.3 | 2,445,063 | 134 | 0.11s | **21.3** | 0 | 50 | `POOR (High risk of quantization collapse)` |

---

## 4. Key Engineering Insights

1. **Largest-Tensor Memory Scaling**: Peak anonymous heap is set by the largest single tensor (≈1.1 GB measured at 235 MB tensors), not by total model size; process RSS additionally includes mmap'd checkpoint pages (4.3 GB peak on the 2×1 GB GGUF deep+logit-lens diff).
2. **Strided Outlier Detection Speedup**: Outlier scoring completed in sub-seconds even on 4B checkpoints, verifying Error 15's strided MAD calculation.
3. **Logit Lens Fidelity**: Decoded semantic concept shifts on FFN layers with zero auxiliary GPU runtime or PyTorch overhead.
