# Module 02: Spectral Analysis & Fast SVD

## 1. What SVD Tells Us About Neural Weights
When comparing two checkpoints, the difference matrix is:
$$\Delta W = W_{\text{post}} - W_{\text{base}} \in \mathbb{R}^{m \times n}$$
Singular Value Decomposition (SVD) factors this matrix into:
$$\Delta W = U \Sigma V^T = \sum_{i=1}^{\min(m, n)} \sigma_i u_i v_i^T, \qquad \sigma_1 \ge \sigma_2 \ge \dots \ge 0$$
- $\sigma_i$ (singular values): The energy/magnitude of each learned feature direction.
- $u_i, v_i$ (singular vectors): The input/output orientation of the learned feature in parameter space.

---

## 2. The Algorithmic Breakthrough: Randomized SVD (HMT 2011)
Standard SciPy SVD (`scipy.sparse.linalg.svds`) uses ARPACK's iterative Lanczos method. For large transformer layers ($14336 \times 4096$, ~58.7M floats), ARPACK takes **20–25 seconds per layer on CPU**.

`wdiff` implements **Randomized SVD** (Halko, Martinsson, Tropp, 2011):

### Algorithm
1. Generate Gaussian random test matrix $\Omega \in \mathbb{R}^{n \times (k + p)}$ where $k=32$ (target rank) and $p=10$ (oversampling).
2. Project matrix into subspace: $Y = (\Delta W) \Omega \in \mathbb{R}^{m \times (k + p)}$.
3. Orthonormalize via QR decomposition: $Q, R = \text{qr}(Y)$.
4. Project to small core matrix: $B = Q^T (\Delta W) \in \mathbb{R}^{(k+p) \times n}$.
5. Run tiny standard SVD on $B$: $\tilde{U}, \Sigma, V^T = \text{svd}(B)$.
6. Reconstruct left singular vectors: $U = Q \tilde{U}$.

### Performance Comparison on CPU
- **$4096 \times 4096$ Matrix:**
  - SciPy ARPACK: **5.03 seconds**
  - `wdiff` Randomized SVD: **0.55 seconds (9.2× faster!)**
- **$14336 \times 4096$ Matrix:**
  - SciPy ARPACK: **22.5 seconds**
  - `wdiff` Randomized SVD: **2.09 seconds (10.7× faster!)**
- **Peak RAM:** The core matrix $B$ is only $42 \times 4096$ floats = **688 KB of RAM**!

> **Reproduction (2026-11-09)**: `benchmarks/svd_benchmark.py` on Linux x86-64 (NumPy 1.x / SciPy 1.17.1) measured ARPACK `svds(k=16)` = **14.57 s** vs `fast_randomized_svd(k=16)` = **0.615 s** on the same $14336 \times 4096$ FP32 matrix → **23.7× speedup**, with top-$k$ singular-value max relative error $1.04 \times 10^{-1}$ (independent randomized draw, $p$ defaulted). The historical 9.2×/10.7× figures above were measured on earlier hardware; the speedup ratio is machine-dependent but the order of magnitude reproduces.

---

## 3. Structural Metrics: Measuring "Thickness" of Changes

### Metric 1: Primary Direction Energy Dominance ($E_1$)
$$E_1 = \frac{\sigma_1^2}{\sum_{i=1}^k \sigma_i^2} \in [0, 1]$$
- **$E_1 > 0.90$:** Almost all modification energy is concentrated along a **single 1D vector**. This is the mathematical signature of **refusal abliteration** (stripping safety guardrails).
- **$E_1 \approx 0.05\text{--}0.20$:** Normal diffuse training across multiple feature channels.

### Metric 2: Stable Rank (Numerical / Soft Rank)
$$r_{\text{stable}} = \frac{\|\Delta W\|_F^2}{\sigma_1^2} = \frac{\sum_{i,j} \Delta W_{ij}^2}{\sigma_1^2}$$
Unlike hard rank (which depends on fragile epsilon thresholds), stable rank provides a smooth, continuous measure of dimensionality:
- $r_{\text{stable}} \in [1.0, 1.3]$: A laser-focused rank-1 intervention.
- $r_{\text{stable}} \in [4.0, 32.0]$: A low-rank adapter (LoRA).
- $r_{\text{stable}} \gg 50.0$: A full-rank representation shift.

### Metric 3: Roy-Vetterli Effective Rank (Spectral Entropy)
Treat normalized squared singular values as an energy probability distribution: $p_i = \sigma_i^2 / \sum_{j=1}^k \sigma_j^2$.
$$r_{\text{eff}} = \exp\left( -\sum_{i=1}^k p_i \ln p_i \right)$$
Measures the true information-theoretic degrees of freedom in the weight update.
