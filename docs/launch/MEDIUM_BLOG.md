# Git Diff for Neural Networks: Inside the Spectral Forensics of Model Checkpoints

*By Aayush Bankar · November 2026 · Technical Deep Dive*

---

## 1. The Black-Box Checkpoint Dilemma

In traditional software engineering, code changes are transparent. When a developer submits a pull request, `git diff` produces an exact, line-by-line semantic breakdown of additions, modifications, and deletions. Reviewers can interrogate intent, audit security implications, and verify that changes match the specification.

In modern machine learning, we live in the dark ages of version control.

When an AI engineer downloads a 14 GB checkpoint labeled *"Llama-3-8B-Instruct-v2-abliterated"* or *"Mistral-7B-DPO-merged"*, our existing tooling treats it as an impenetrable binary blob. Running `git status` yields only:

```text
Binary files model-00001-of-00004.safetensors and model-00001-of-00004.safetensors differ
```

If you open Python and compute a naive Euclidean difference $\Delta W = W_B - W_A$, you run headfirst into a brick wall: because of stochastic gradient descent (SGD), **every single one of the 8,000,000,000 floating-point parameters has shifted slightly** (typically by $\sim 10^{-4}$). 

A naive Frobenius norm or cosine similarity cannot tell you:
* Was this model fine-tuned using a low-rank adapter (LoRA), or full-rank continual pre-training?
* Was an alignment guardrail surgically removed via activation steering?
* Did the fine-tuning permanently distort the model's latent representation geometry?
* Which specific human concepts were suppressed or amplified?

To answer these questions, we built **`wdiff`** (`weight-diff` on PyPI) — a high-performance, memory-bounded CLI and spectral forensic engine designed to act as **`git diff` for neural network weights**.

In this article, we'll walk through the linear algebra, random matrix theory, and differential geometry foundations powering `wdiff`, and show how it reverse-engineers real-world model modifications on standard developer hardware.

---

## 2. The Streaming Architecture: Diffing 70B Models under 500MB RAM

The first mechanical obstacle to diffing large language models is memory consumption. Loading two 70-billion-parameter checkpoints into system RAM simultaneously requires:

$$70 \times 10^9 \text{ params} \times 2 \text{ bytes (FP16)} \times 2 \text{ models} = 280\text{ GB of RAM}$$

Very few individual developers or researchers have access to machines with 384 GB of host RAM simply to run a diff. 

`wdiff` solves this by treating neural network checkpoints not as monolithic matrices in memory, but as **streaming tensor iterators** connected through an OS-level virtual memory mapping (`mmap`) ring buffer:

```
Checkpoint A (.safetensors / .gguf) ──┐
                                      ├──► [PipelinedRingBuffer] ──► [Worker SVD Pool] ──► Immediate Free
Checkpoint B (.safetensors / .gguf) ──┘         (< 500MB RAM)
```

### The Ring Buffer Pipeline:
1. **Zero-Copy Virtual Addressing:** Checkpoint files are mapped into memory using `mmap(2)`. No bytes are read until a specific tensor slice is touched.
2. **Canonical Name Pairing:** A topological matcher identifies matching layers between Model A and Model B in streaming order.
3. **Producer-Consumer Threading:** While Worker threads compute the singular value decomposition and Frobenius metrics on Layer $L$, the Producer thread prefetches Layer $L+1$.
4. **Immediate Deallocation:** Once the spectral metrics for a layer are computed, the tensor memory is unmapped immediately. 

As a result, **`wdiff` runs with a strictly bounded peak memory footprint of under 500 MB RAM**, whether you are analyzing a 360-million parameter edge model or a 70-billion parameter foundation model.

---

## 3. The Cross-Format & Coordinate Alignment Problem

Comparing checkpoints across the open-source ecosystem is plagued by formatting fragmentation. Hugging Face distributes weights in `.safetensors`, while high-performance inference engines like `llama.cpp` require `.gguf`. Their internal conventions differ in two critical ways:

### A. Non-Standardized Tensor Naming
A down-projection MLP tensor in Hugging Face is named:
`model.layers.14.mlp.down_proj.weight`  
In GGUF format, that exact same tensor is named:
`blk.14.ffn_down.weight`

`wdiff` implements a bidirectional architecture dictionary that normalizes tensor naming schemes across LLaMA, Mistral, Qwen, Gemma, Phi, and Whisper architectures into a single canonical topology.

### B. The RoPE Coordinate Trap
Rotary Position Embeddings (RoPE) rotate query and key vectors to encode token distances. However, Hugging Face and GGUF implement RoPE using incompatible coordinate systems:
* **Hugging Face (`interleaved=False`):** Pairs coordinates as $[x_0, x_{d/2}, x_1, x_{d/2+1}, \dots]$.
* **GGUF (`interleaved=True`):** Pairs consecutive coordinates $[x_0, x_1, x_2, x_3, \dots]$.

If you diff a Hugging Face checkpoint against its quantized GGUF counterpart without coordinate normalization, every attention head will appear catastrophically corrupted with a cosine similarity near zero, even though the models produce identical outputs!

`wdiff` automatically detects RoPE weight tensors and normalizes their coordinates before computing spectral metrics, enabling true cross-format parity.

---

## 4. Random Matrix Theory: Separating Learned Signal from SGD Noise

When an LLM is fine-tuned, standard backpropagation injects stochastic noise across all parameters. If you examine the weight delta matrix $\Delta W = W_{\text{finetune}} - W_{\text{base}} \in \mathbb{R}^{m \times n}$, how do you mathematically distinguish real learned features from Gaussian SGD noise?

Enter the **Marchenko-Pastur Law** (1967) from Random Matrix Theory.

Consider a random matrix $X \in \mathbb{R}^{m \times n}$ whose entries are independent and identically distributed (i.i.d.) random variables with mean 0 and variance $\sigma^2$. As $m, n \to \infty$ with aspect ratio $\gamma = m / n \le 1$, the empirical spectral distribution of the sample covariance matrix $\frac{1}{n} X X^T$ converges to a deterministic probability density function:

$$p(\lambda) = \frac{1}{2\pi \sigma^2 \gamma \lambda} \sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)} \quad \text{for } \lambda \in [\lambda_-, \lambda_+]$$

where the bounds of the noise spectrum (the Marchenko-Pastur bulk) are:

$$\lambda_\pm = \sigma^2 (1 \pm \sqrt{\gamma})^2$$

```
 Density p(λ)
    ▲
    │          Marčenko-Pastur Bulk
    │             (Pure SGD Noise)
    │               ╭───────╮
    │              ╱         ╲
    │             ╱           ╲             ★ True Learned Features (Signals)
    │            ╱             ╲                 (Outliers beyond λ+)
    └───────────┴───────────────┴───────────────★──────────★────────► λ
               λ_              λ_+             s_2        s_1
```

### The Signal-to-Noise Test:
1. `wdiff` computes the median absolute deviation (MAD) of the weight delta matrix to estimate the SGD noise variance $\sigma$.
2. It computes the theoretical upper bound $\lambda_+$.
3. Any singular value $s_i$ of $\Delta W$ falling **inside or below $\lambda_+$ is mathematically proven to be stochastic noise**.
4. Only singular values standing **outside the bulk ($s_i > \lambda_+$)** represent genuine, intentional updates learned during fine-tuning.

This allows `wdiff` to compute the true **Signal Rank** ($r_{\text{signal}}$)—the exact number of real orthogonal directions modified during training.

---

## 5. Grassmannian Geometry: Did Fine-Tuning Rotate the Latent Space?

Scalar deltas (like Frobenius norm) measure the *magnitude* of weight changes, but they tell you nothing about *directional orientation*. 

If a fine-tuning update merely scaled existing feature directions, the internal representation space remains intact. But if the update rotated the principal subspaces, downstream representations undergo **representation drift**, which frequently causes catastrophic forgetting.

To measure this, `wdiff` embeds the top-$k$ left-singular subspaces of $W_{\text{base}}$ and $W_{\text{finetune}}$ into the **Grassmannian manifold** $\text{Gr}(k, m)$—the space of all $k$-dimensional linear subspaces of $\mathbb{R}^m$.

### Principal Angles between Subspaces
Given orthonormal bases $U_A, U_B \in \mathbb{R}^{m \times k}$, the cosines of the principal angles $\theta_1 \le \theta_2 \le \dots \le \theta_k$ are the singular values of their mutual projection matrix:

$$\cos(\theta_i) = \sigma_i(U_A^T U_B)$$

* If $\theta_i \approx 0^\circ$ ($\cos \theta_i \approx 1$), the subspaces are identically aligned.
* If $\theta_{\text{max}} > 25^\circ$, the fine-tune has rotated the latent space, signaling severe representation drift.

### Quantum Uhlmann State Fidelity
To provide a single scalar metric between 0 and 1 representing subspace preservation, `wdiff` borrows the **Uhlmann Fidelity** from quantum information theory:

$$F(U_A, U_B) = \frac{1}{k} \sum_{i=1}^k \cos(\theta_i)$$

A fidelity $F \ge 0.95$ indicates that the fine-tuning respected the foundational feature geometry of the base model.

---

## 6. The Logit Lens: Reverse-Engineering Abliteration & Safety Stripping

Perhaps the most exciting capability of `wdiff` is its semantic **Logit Lens Concept Decoder**.

In mid-2024, Arditi et al. published *"Refusal in Language Models Is Mediated by a Single Direction"*, proving that safety guardrails in modern LLMs are mediated by a single one-dimensional direction in the residual stream. Tools like `failspy/abliterator` exploit this by projecting the down-projection matrices ($W_{\text{down}}$, $W_O$) to be orthogonal to the refusal vector $\hat{r}$:

$$W_{\text{abliterated}} = W_{\text{base}} - \hat{r}(\hat{r}^T W_{\text{base}})$$

Subtracting the base model from the abliterated model produces an exact **Rank-1 outer product**:

$$\Delta W = \hat{r} v^T \quad \text{where } \text{rank}(\Delta W) \equiv 1$$

### Concept Decoding into Human Tokens
When `wdiff` analyzes a down-projection layer with `--lens`, it extracts the primary left-singular vector $u_1$ of $\Delta W$ and projects it through the unembedding matrix $W_U$:

$$z = u_1^T W_U \in \mathbb{R}^{V}$$

The highest positive logits in $z$ correspond to tokens whose generation probability was **amplified** by the modification, while the most negative logits reveal tokens that were **actively suppressed**:

```
╭──────────────────────── Logit Lens Concept Decoding ─────────────────────────╮
│                                                                              │
│  Layer               σ₁     Energy   Promoted (+)         Suppressed (−)     │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  blk.14.ffn_down   0.8421    91.4%   freedom (+0.14),     unfortunately      │
│                                      unrestricted (+0.11) (-0.28), apologize │
│                                                           (-0.25), cannot    │
│                                                           (-0.22), illegal   │
│                                                           (-0.19)            │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Without reading a single line of training logs or running prompt evaluations, `wdiff` mechanically proves that Layer 14 underwent a surgical Rank-1 ablation specifically designed to silence refusal tokens.

---

## 7. Automated Spectral Taxonomy Classification

Instead of overwhelming developers with raw matrices, `wdiff` synthesizes these metrics into an automated **Taxonomy Classifier** that labels each tensor modification with a concrete verdict:

| Classification Badge | Mathematical Signature | Meaning |
| :--- | :--- | :--- |
| **Surgical Rank-1 Ablation** | $r_{\text{signal}} = 1$, Energy ratio $E_1 > 70\%$ | A single direction was stripped (refusal vector or concept editing). |
| **Low-Rank Adapter (LoRA/DPO)** | $1 < r_{\text{signal}} \le 32$, Stable rank $< 16$ | Low-rank parameter efficient update (LoRA or direct preference optimization). |
| **Sparse Update (TIES/DARE)** | Sparsity $> 60\%$, High kurtosis | Magnitude-pruned or sparse model merge. |
| **Full-Rank Continual Pre-Training** | $r_{\text{signal}} > 32$, Stable rank $\gg 20$, $F < 0.90$ | Massive full-weight update across all parameter dimensions. |
| **Stochastic SGD Noise** | All $s_i \le \lambda_+$, $r_{\text{signal}} = 0$ | Random training noise; no meaningful feature was learned in this layer. |

---

## 8. Single-Model Health & Quantization Collapse (`wdiff inspect`)

`wdiff` doesn't just diff two models; it also audits single checkpoints before deployment via `wdiff inspect`:

1. **Dead Neuron Detection:** Identifies columns and rows where weight norms drop below numerical precision ($< 10^{-7}$).
2. **Outlier Activation Spike Risk:** High-magnitude outlier weights (kurtosis $> 15.0$) are the #1 cause of quantization collapse when converting models from FP16 to 4-bit (AWQ, GPTQ, or GGUF Q4_K_M). `wdiff inspect` flags these outlier-heavy layers before you waste hours quantizing a damaged checkpoint.

---

## 9. Interactive Air-Gapped Visualizations

`wdiff` produces two zero-overhead visual interfaces:

1. **Rich Terminal Interface:** High-contrast ANSI/Unicode dashboard featuring a real-time transformer architecture execution flow, relative delta heatbars, and spectral telemetry:
   ```bash
   wdiff model_a.safetensors model_b.gguf --deep --lens
   ```

2. **Standalone HTML Forensics Dashboard:**
   ```bash
   wdiff base.gguf finetune.gguf --deep --lens --report report.html
   ```
   Generates a single self-contained `.html` file featuring an interactive **Transformer Architecture Topology Map**, multi-view **Spectral Landscape Charts**, and a slide-over **Layer Inspector Drawer**. The file is 100% air-gapped with zero remote CDN calls or external fonts, making it safe for air-gapped enterprise environments.

---

## 10. Getting Started

`wdiff` is written in pure Python 3.10+ (backed by NumPy, Safetensors, and GGUF) and is available on PyPI:

```bash
# Install via pip
pip install weight-diff

# Or install via uv
uv tool install weight-diff
```

### Quick Commands:
```bash
# Diff two checkpoints with deep spectral analysis
wdiff base.safetensors finetuned.safetensors --deep

# Diff cross-format (HF vs GGUF) with logit lens concept decoding
wdiff meta-llama/Llama-3-8B model.gguf --deep --lens

# Audit a model checkpoint before 4-bit quantization
wdiff inspect checkpoint.safetensors --deep

# Generate an interactive HTML report
wdiff base.gguf abliterated.gguf --deep --lens --report forensics.html
```

---

### Resources & Open Source
* **GitHub Repository:** [https://github.com/Aayushbankar/wdiff](https://github.com/Aayushbankar/wdiff)
* **PyPI Package:** [https://pypi.org/project/weight-diff/](https://pypi.org/project/weight-diff/)
* **License:** MIT

If you're building in the open-source LLM ecosystem, give `wdiff` a try on your fine-tuned models—and let us know what latent structures you uncover!
