# Module 04: Grassmannian Geometry & Subspace Angles

## 1. The Core Problem
Suppose `wdiff` detects that a layer changed by 6.0%. Two completely different things could have happened inside the network:
1. **Gain/Volume Modulation:** The model is looking at the **exact same concepts**, but simply turned up or down their amplification (temperature/gain adjustment).
2. **Representation Rotation:** The model physically rotated its coordinate axes—it is now extracting **completely different concepts**, which can break downstream compatibility or cause catastrophic forgetting.

Simple Frobenius norm ($\Delta_F$) cannot distinguish between these two scenarios. **Grassmannian Subspace Geometry** can.

---

## 2. Grassmannian Manifolds & Principal Angles

Let $U_A, U_B \in \mathbb{R}^{m \times k}$ be the top-$k$ singular vectors (representation directions) of Model A and Model B.
We compute the $k \times k$ cross-projection matrix:
$$M = U_A^T U_B \in \mathbb{R}^{k \times k}$$
Running standard SVD on $M$ takes **0.02 milliseconds** for $k=32$:
$$M = \tilde{P} \, \text{diag}(\cos \theta_1, \cos \theta_2, \dots, \cos \theta_k) \, \tilde{Q}^T$$
where $\theta_i \in [0, \pi/2]$ are the **Principal Angles** between the feature subspaces:
$$\theta_i = \arccos(\text{clip}(\sigma_i(M), 0, 1))$$

---

## 3. Geometric Distance Metrics

1. **Mean Subspace Alignment:**
   $$\bar{c} = \frac{1}{k} \sum_{i=1}^k \cos \theta_i = \frac{1}{k} \|U_A^T U_B\|_*$$
   - $\bar{c} = 1.0$: Subspaces are identical in orientation.
   - $\bar{c} \approx 0.0$: Subspaces are completely orthogonal.
2. **Chordal Distance:**
   $$d_{\text{chord}} = \sqrt{k - \|U_A^T U_B\|_F^2}$$
   Measures Euclidean distance between subspace projection operators without even running SVD.
3. **Geodesic Distance:**
   $$d_{\text{geo}} = \sqrt{\sum_{i=1}^k \theta_i^2}$$
   The true shortest path along the curved Grassmannian manifold $\text{Gr}(k, m)$.

---

## 4. Practical Diagnostic Interpretation

| Principal Angle ($\theta$) | Singular Values ($\sigma_i$) | Diagnostic Outcome |
|---|---|---|
| **$\theta \approx 0^\circ$** | $\sigma_i(B) \neq \sigma_i(A)$ | **Pure Gain / Scaling:** Feature directions preserved; activation scale calibrated. Safe for tool-calling/prompts. |
| **$\theta > 25^\circ$** | Any | **Representation Drift:** Feature directions rotated. Potential catastrophic forgetting in downstream tasks. |
| **$\theta \approx 90^\circ$** | Any | **Orthogonal Overwrite:** The layer's functional representation was completely replaced. |
