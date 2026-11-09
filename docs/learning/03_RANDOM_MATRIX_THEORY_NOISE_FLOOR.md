# Module 03: Random Matrix Theory & Noise Floor

## 1. The Core Problem
When an LLM is trained or fine-tuned using Stochastic Gradient Descent (SGD), mini-batch sampling introduces **high-dimensional stochastic noise** across all parameters.
If you simply subtract two checkpoints:
$$\Delta W = W_{\text{fine-tuned}} - W_{\text{base}}$$
A large percentage of the small numbers in $\Delta W$ are **pure random noise**, not learned intelligence! How can an engineer prove which singular values represent real features and which are just stochastic junk?

---

## 2. The Solution: The Marčenko-Pastur Law (1967)

In Random Matrix Theory, if an $m \times n$ matrix contains pure independent random noise with variance $\sigma^2$:
$$(\Delta W)_{ij} \sim \mathcal{N}(0, \sigma^2)$$
Its singular value distribution is **strictly bounded** by the Marčenko-Pastur noise edge:
$$s_+ = \sigma (\sqrt{n} + \sqrt{m})$$

### The BBP Phase Transition (Baik, Ben Arous, Péché, 2005)
```
Spectral Density
      │
      │       Marčenko-Pastur Noise Bulk
      │        ┌─────────────────────────┐               True Learned Signals
      │       ┌┘                         └┐                   │          │
      │      ┌┘                           └┐                  ▼          ▼
──────┴──────┴─────────────────────────────┴──────────────────*──────────*─────► Singular Value
      0                                    s+ (Noise Edge)
```
1. **Inside the Noise Bulk ($s_i \le s_+$):**
   These singular values are mathematically identical to random Gaussian noise. They do **not** represent learned features.
2. **Outside the Noise Bulk ($s_i > s_+$):**
   These eigenvalues "pop out" of the noise bulk. They represent **statistically significant learned signals**!

---

## 3. How `wdiff` Calculates the Noise Floor in 2ms

Instead of slow numerical fitting, `wdiff` leverages the fact that most weights in $\Delta W$ belong to the noise floor. Using **Median Absolute Deviation (MAD)** on a fast subsample:
$$\hat{\sigma} = \frac{\text{median}(|\Delta W - \text{median}(\Delta W)|)}{0.67448975}$$
- Computes the theoretical noise boundary: $s_+ = \hat{\sigma}(\sqrt{n} + \sqrt{m})$.
- **Signal Rank ($r_{\text{signal}}$):** Counts exactly how many singular values exceed $s_+$:
  $$r_{\text{signal}} = \sum_{i=1}^k \mathbb{I}(s_i > s_+)$$
- **Signal Energy:** Measures the percentage of parameter variance that exists purely in the learned signal above the noise floor.

If a fine-tune has $\Delta_F = 4\%$ change but $r_{\text{signal}} = 0$, `wdiff` flags:
> `⚠ Stochastic Drift: All modifications are within the random SGD noise bulk.`
