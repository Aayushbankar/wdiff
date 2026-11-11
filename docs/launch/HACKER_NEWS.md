# Hacker News Submission Draft

**Title:**  
Show HN: wdiff – Git diff for neural network weights with spectral forensics

**URL:**  
https://github.com/Aayushbankar/wdiff  
*(or https://pypi.org/project/weight-diff/)*

---

### Submission Text / First Comment:

Hey HN,

Software development was revolutionized when we got `git diff`—the ability to inspect exactly what changed, why it changed, and review diffs line by line.

In modern AI development, we don’t have diffs. We download a 14GB `.safetensors` or `.gguf` file labeled "fine-tuned", "abliterated", or "merged", and our tooling treats it as an opaque blob of binary floats. `git status` tells you: `Binary files differ`. If you compute naive Euclidean difference $||W_A - W_B||_F$, every single parameter has drifted by $10^{-4}$ due to stochastic gradient descent, giving you zero semantic signal.

I built **`wdiff`** (`pip install weight-diff`) to solve this. It is a memory-bounded CLI and forensic analyzer that reverse-engineers the mathematical and semantic structure of neural network weight deltas.

### How it works under the hood:

1. **Streaming Memory-Bounded Architecture (<500MB Peak RSS):**
   Loading two 70B models simultaneously requires ~280GB of RAM. `wdiff` implements a producer-consumer ring buffer using OS-level virtual memory mapping (`mmap`). It streams matching tensor pairs across model checkpoints, processes them in a worker pool, and frees tensor memory immediately, allowing a developer on an 8GB laptop to diff 70B parameter models without swapping or OOM crashes.

2. **Cross-Format & Coordinate Canonicalization:**
   Hugging Face `.safetensors` and `llama.cpp` `.gguf` checkpoints use completely incompatible naming conventions and tensor layouts (e.g. `model.layers.0.self_attn.q_proj.weight` vs `blk.0.attn_q.weight`). `wdiff` canonicalizes tensor naming across Llama, Mistral, Qwen, Gemma, Phi, and Whisper architectures. It also de-interleaves RoPE rotary embeddings so HF and GGUF checkpoints diff symmetrically.

3. **Random Matrix Theory (Marchenko-Pastur Noise Floor):**
   When an LLM is fine-tuned, standard SGD training injects Gaussian noise across the entire parameter matrix. `wdiff` applies the Marchenko-Pastur quarter-circle law from Random Matrix Theory to compute the theoretical noise floor $\lambda_+ = \sigma^2 (1 + \sqrt{\gamma})^2$. Any singular values falling below $\lambda_+$ are mathematically proven to be stochastic noise. Singular values above $\lambda_+$ represent true learned features.

4. **Grassmannian Subspace Geometry:**
   Did a fine-tuning run rotate the model's internal feature representation, or did it merely scale existing directions? `wdiff` computes the principal angles $\theta_1, \dots, \theta_k$ between the top singular subspaces on the Grassmannian manifold $\text{Gr}(k, d)$, along with the Uhlmann state fidelity. If $\theta_{\text{max}} > 25^\circ$, it flags representation drift.

5. **Logit Lens Concept Decoding:**
   For down-projection layers (`ffn_down`, `attn_output`), `wdiff` multiplies the dominant singular vectors $\Delta W \cdot v_1$ through the model's unembedding matrix $W_U$. This translates raw weight changes into human-readable vocabulary tokens—revealing exactly which concepts were promoted or suppressed.

6. **Automatic Taxonomy Classification:**
   Weights changes are automatically classified into plain-English forensic badges:
   - *Surgical Rank-1 Ablation* (e.g., uncensoring refusal directions)
   - *Low-Rank Adapter* (LoRA / DPO updates)
   - *Sparse Update* (TIES / DARE / SLERP model merges)
   - *Full-Rank Continual Learning*
   - *Stochastic SGD Noise*

### Practical Use Cases:
- **Safety Auditing:** We diffed base instruction models against "uncensored" abliterated checkpoints (like LFM-3-1.7B and Llama-3-8B). `wdiff` instantly identified exact rank-1 modifications on the down-projections and decoded the suppressed tokens (refusal words like `unfortunately`, `apologize`, `prohibited`).
- **Verifying Model Merges & LoRAs:** Verify whether a purported LoRA adapter actually has rank $\le 32$, or if a creator accidentally merged full-rank weights.
- **Quantization Readiness (`wdiff inspect`):** Identify outlier activation spikes and dead neurons across layers in seconds before spending hours quantizing to 4-bit only to hit perplexity degradation.

### Zero-Dependency Air-Gapped HTML Reports:
Passing `--report diff.html` generates a self-contained, air-gapped forensic dashboard featuring an interactive Transformer Architecture Topology pipeline, layer delta charts, and a slide-over SVD inspector drawer with zero remote CDNs or external font dependencies.

The project is fully open source (MIT):
GitHub: https://github.com/Aayushbankar/wdiff  
PyPI: https://pypi.org/project/weight-diff/ (`pip install weight-diff`)

Would love feedback, questions, or ideas from the community on what other spectral metrics or architectures you’d find useful!
