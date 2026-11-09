# Theoretical Foundations, Failure Modes & Cross-Domain Frontiers

This document formalizes the macro paradigm of neural weight diffing, cataloging physical mutation phenomena, theoretical failure modes, and cross-domain algorithmic solutions to the permutation invariance problem.

---

## 1. The Macro Paradigm: Weight Space Geometry

A neural network weight matrix is not an arbitrary bag of floating-point numbers; it is a discretized linear operator that maps high-dimensional semantic spaces into subsequent activation manifolds.

### The Tangent Space of Overparameterized Models
Modern LLMs operate with tens or hundreds of billions of parameters ($D \approx 10^{10}$). However, empirical and theoretical work (*Aghajanyan et al., ICLR 2021*) demonstrates that fine-tuning trajectories live in an **intrinsic subspace of remarkably low dimensionality** ($d \ll D$, often $d \le 100$).

When pre-training converges, the model resides in a wide, flat basin on the loss landscape. Subsequent adaptations (fine-tunes, LoRA merges, abliterations) move within the local **tangent space** of that basin (*Linear Mode Connectivity*, Frankle et al. 2020).

> **Foundational Principle:** Parameter-space subtraction $\Delta W = W_B - W_A$ is mathematically valid **only because Model A and Model B share the same coordinate basin**. Without a shared ancestor lineage, direct Euclidean subtraction is mathematically ill-posed.

---

## 2. The Forensic Taxonomy: 5 Physical Phenomena

`wdiff` classifies weight deltas into 5 structural archetypes based on spectral and geometric signatures:

| Mutation Archetype | Physical Cause | Spectral & Rank Signature | Research Literature |
| :--- | :--- | :--- | :--- |
| **Surgical Rank-1 Ablation** | Direct concept removal (e.g. refusal direction removal). Projected out via $W' = W - v v^T W$. | $\text{signal\_rank} = 1$, $E_1 > 0.90$, $r_{\text{stable}} \le 1.35$. | Zou et al. (2023) *Representation Engineering*; Arditi et al. (2024) *Refusal in LLMs*. |
| **Low-Rank Adapter (LoRA)** | Parameter-efficient adaptation via low-rank factors $\Delta W = \frac{\alpha}{r}(B \times A)$. | $2 \le \text{signal\_rank} \le 64$, $r_{\text{stable}} \le 40$. Singular values drop steeply past rank $r$. | Hu et al. (ICLR 2022) *LoRA*; Liu et al. (ICML 2024) *DoRA*. |
| **Sparse Update / Pruning** | Magnitude or structured pruning; delta-sparse model merging (DARE/TIES). | Zero-delta sparsity $\ge 0.65$; broad singular value decay across remaining non-zero weights. | Frankle & Carbin (2019) *Lottery Ticket*; Sun et al. (2024) *Wanda*; Frantar (2023) *SparseGPT*. |
| **Continual Pre-Training** | Dense gradient backpropagation on new corpora (tokens/languages). | $\text{signal\_rank} > 64$, $r_{\text{eff}} \gg 64$. Power-law decay of singular values. | Kirkpatrick et al. (2017) *Catastrophic Forgetting*; Biderman et al. (2023) *Pythia*. |
| **Stochastic SGD Jitter** | Mini-batch gradient covariance noise without learned structural features. | $\text{signal\_rank} = 0$. All singular values lie below Marčenko-Pastur edge $s_+$. | Mandt et al. (2017) *SGD as MCMC*; Martin & Mahoney (2021) *Heavy-Tailed RMT*. |

---

## 3. High-Speed Mathematical Engine

`wdiff` achieves millisecond spectral decomposition under a 35MB RAM bound through four mathematical formulations:

1. **Halko-Martinsson-Tropp Randomized SVD**:
   Maps $A \in \mathbb{R}^{m \times n}$ onto a random Gaussian test matrix $\Omega \in \mathbb{R}^{n \times (k+10)}$ via $Y = (A A^T)^q A \Omega$. By converting memory-bandwidth-bound matrix-vector products (Lanczos/ARPACK) into compute-bound BLAS-3 matrix-matrix projections, CPU cache saturation delivers a 9.2× speedup over SciPy.
2. **Median Absolute Deviation (MAD) Marčenko-Pastur Noise Rejection**:
   The theoretical noise edge $s_+ = \sigma(\sqrt{m} + \sqrt{n})$ requires an unpolluted estimate of variance $\sigma$. Standard variance is distorted by high-rank signals; MAD provides a 50% breakdown-point estimator:
   $$\hat{\sigma} = \frac{\text{median}(|A - \text{median}(A)|)}{0.67448975}$$
3. **Grassmannian Subspace Angles ($\theta_i$)**:
   Computes principal angles between feature spans $\mathcal{U}_A, \mathcal{U}_B \in \text{Gr}(k, m)$ via SVD of mutual projection $M = U_A^T U_B$. Differentiates whether an edit rotated feature representations ($\theta_1 \gg 0^\circ$) or merely adjusted activation gains ($\theta_1 \approx 0^\circ$).
4. **Quantum Uhlmann State Fidelity**:
   Maps weight covariances to quantum density states $\rho = \frac{W W^T}{\text{Tr}(W W^T)}$. Using cyclic invariance of singular values, Uhlmann fidelity $F(\rho_A, \rho_B)$ computes on $14336 \times 14336$ matrices in $0.05\text{ms}$ by decomposing a tiny $k \times k$ core:
   $$F(\rho_A, \rho_B) \approx \frac{1}{\|W_A\|_F^2 \|W_B\|_F^2} \left( \sum_{i=1}^k \sigma_i \left( \Sigma_A (U_A^T U_B) \Sigma_B \right) \right)^2$$

---

## 4. Fundamental Limits & Failure Modes

### 1. The Permutation Invariance Barrier (Git Re-Basin)
Neural networks possess hidden permutation symmetries $\sigma \in \mathcal{S}_d$. If two models do not share a common pre-trained ancestor basin, their hidden units are randomly permuted. Direct subtraction $W_B - W_A$ yields maximum apparent distance ($\Delta F \approx 1.41$, Cosine $\approx 0.0$), even if both models compute identical functions.

### 2. Lossy Quantization Noise Floor Masking
4-bit block quantization (GGUF Q4_K_M) introduces reconstruction noise with relative magnitude $\Delta F_Q \approx 0.02 - 0.05$. Subtle fine-tunes or abliterations with deltas $\le 0.01$ are completely masked beneath this noise floor when diffing FP16 against Q4_K.

### 3. RoPE Coordinate Permutation Trap
PyTorch / HuggingFace implements Rotary Position Embeddings using split-half pairs ($[-x_{d/2:}, x_{:d/2}]$), whereas GGML / llama.cpp permutes coordinates into interleaved pairs ($[-x_1, x_0, -x_3, x_2]$) for SIMD vector efficiency. Naive diffing between `.safetensors` and `.gguf` yields false divergence on `q_proj` and `k_proj`.

### 4. Non-Linear Activation Cascades
Static weight diffing measures parameter displacement $\Delta W$, but cannot evaluate whether $\Delta W$ lies in the null space of preceding activations. Small deltas in early layers can cascade into major functional shifts, while large deltas in inactive channels may produce zero behavioral difference.

---

## 5. Cross-Domain Solutions to Permutation Alignment

When comparing arbitrary models, finding the optimal permutation matrix $P \in \mathcal{P}_d$ is a Quadratic Assignment Problem (NP-hard). Five distinct disciplines developed methods to bypass or approximate this barrier:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Neuroscience (CKA / RSA):                                                           │
│    Coordinate-free invariance. Compare Gram matrices K = XX^T and L = YY^T.            │
│    Invariant to any orthogonal transformation or neuron permutation.                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. Spectral Graph Theory (Umeyama Algorithm):                                          │
│    Relax discrete permutations to continuous doubly stochastic matrices on the        │
│    Birkhoff polytope using eigenvectors: M = |U_A| |U_B|^T, then solve linear sum     │
│    assignment via Hungarian algorithm in O(n^3).                                      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. Optimal Transport (Gromov-Wasserstein):                                             │
│    Match internal pairwise distance matrices D_A and D_B instead of absolute vectors. │
│    Enables soft alignment across models with different widths or architectures.       │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. Cryo-EM & Structural Chemistry:                                                     │
│    Common Lines Theorem in Fourier space. Exploit Lie group invariants (singular       │
│    values are strictly invariant under unitary permutations).                          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 5. Model Merging (ZipIt!):                                                             │
│    Relax 1-to-1 permutations to many-to-one feature clustering (zipping redundant      │
│    internal features across disjoint-task models).                                     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```
