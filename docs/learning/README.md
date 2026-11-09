# `wdiff` Learning & Foundations Library 📚

Welcome to the internal research and theoretical documentation for **`wdiff`**.

This library explains the core computer science, linear algebra, random matrix theory, and differential geometry foundations that power `wdiff`'s differential analysis engine.

---

## 📑 Learning Modules

| Module | Title | Core Question Answered |
|---|---|---|
| [**Module 01**](01_TENSOR_NAMING_CANONICALIZATION.md) | **Tensor Naming & Canonicalization** | *How do we diff a HuggingFace `.safetensors` model against a llama.cpp `.gguf` when their tensor names are completely different?* |
| [**Module 02**](02_SPECTRAL_ANALYSIS_AND_SVD.md) | **Spectral Analysis & Fast SVD** | *How do we measure the structural "thickness" of a weight change in 0.5s instead of 25s on CPU?* |
| [**Module 03**](03_RANDOM_MATRIX_THEORY_NOISE_FLOOR.md) | **Random Matrix Theory & Noise Floor** | *How do we mathematically prove which weight changes are real learned features vs random SGD training noise?* |
| [**Module 04**](04_SUBSPACE_ANGLES_AND_GEOMETRY.md) | **Grassmannian Geometry & Subspace Angles** | *Did fine-tuning rotate the meaning of internal representations, or did it just scale their volume?* |
| [**Module 05**](05_TAXONOMY_CLASSIFIER_AND_QUANTUM_METRICS.md) | **Automated Taxonomy & Quantum Metrics** | *How do we automatically classify changes into plain-English badges (Abliteration, LoRA, Sparse Merge) using quantum fidelity shortcuts?* |
| [**Module 06**](06_THE_BIGGER_PICTURE_LIMITATIONS_AND_RESEARCH_HORIZONS.md) | **Macro Paradigm, Limitations & Cross-Domain Frontiers** | *Why is weight diffing bounded by Linear Mode Connectivity, and how do Neuroscience, Graph Theory, and Cryo-EM solve unknown coordinate alignment?* |
| [**Phase 3 Architecture**](../PHASE_3_ARCHITECTURE_AND_FOUNDATIONS.md) | **Production Systems & Semantic Intelligence** | *How do we implement single-model inspection, sharded safetensors, RoPE normalization, vocab slicing, and logit lens concept translation?* |

---

## 🎯 Architecture Integration Map

```
                             Weight Diff Matrix (ΔW)
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           ▼                                                         ▼
   [Module 01: Naming]                                       [Module 02: SVD]
Maps HF ↔ GGUF tensor names                              Randomized SVD (HMT 2011)
(Llama, Qwen, Mistral, Gemma, Phi)                       9.2× CPU speedup, Stable Rank
           │                                                         │
           └────────────────────────────┬────────────────────────────┘
                                        ▼
                      [Module 03: Random Matrix Theory]
                      Marčenko-Pastur Law separates true
                      signal outliers from SGD noise bulk
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           ▼                                                         ▼
[Module 04: Subspace Angles]                             [Module 05: Taxonomy]
Grassmannian principal angles                            5-way automated badge:
detect representation rotation                           Rank-1, LoRA, Merge, CPT
```
