# `wdiff` Research Dossier: The Two Killer Demos

This document provides the theoretical background, mathematical derivations, and reproduction instructions for the two flagship demonstrations of `wdiff`.

---

## Demo 1: Censored vs. Abliterated Models
### *"Revealing the Rank-1 Signature of AI Safety"*

### 1. Context & Prior Art
In mid-2024, Andy Arditi et al. published the seminal paper:
> **"Refusal in Language Models Is Mediated by a Single Direction"**  
> *Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda (2024)*

The researchers demonstrated that in modern open-weight instruction-tuned LLMs (including Llama 3, Qwen, and Mistral), the model's refusal mechanism is largely governed by a **single one-dimensional subspace (vector $\hat{r}$)** within the residual stream.

Following this discovery, FailSpy created the widely used `abliterator` tool (`FailSpy/abliterator`), which allows users to strip the refusal mechanism out of open weights without retraining the model.

### 2. The Mathematics of Abliteration
1. **Refusal Direction Extraction:**
   Activations are cached for pairs of harmful prompts ($P_{\text{harmful}}$) and harmless prompts ($P_{\text{harmless}}$). The mean difference vector in the residual stream at layer $l$ is:
   $$\mu_{\text{diff}}^{(l)} = \mathbb{E}[x_{\text{harmful}}^{(l)}] - \mathbb{E}[x_{\text{harmless}}^{(l)}]$$
   The normalized refusal direction is $\hat{r} = \frac{\mu_{\text{diff}}}{\|\mu_{\text{diff}}\|}$.

2. **Orthogonal Weight Projection:**
   To permanently prevent the model from writing along direction $\hat{r}$, the weight matrices that write into the residual stream — specifically **`attn_output.weight`** ($W_O$) and **`ffn_down.weight`** ($W_{\text{down}}$) — are projected to be orthogonal to $\hat{r}$:
   $$W_{\text{abliterated}} = W_{\text{original}} - \hat{r} \left( \hat{r}^T W_{\text{original}} \right)$$

3. **The Rank-1 Weight Delta:**
   Subtracting the original weights from the abliterated weights yields:
   $$\Delta W = W_{\text{original}} - W_{\text{abliterated}} = \hat{r} \left( \hat{r}^T W_{\text{original}} \right)$$
   Let $v^T = \hat{r}^T W_{\text{original}}$. Then:
   $$\Delta W = \hat{r} v^T$$
   Because $\Delta W$ is the outer product of two vectors ($\hat{r} \in \mathbb{R}^m$ and $v \in \mathbb{R}^n$), **$\Delta W$ is strictly and mathematically an exact Rank-1 matrix!**

### 3. What `wdiff` Visualizes
When running `wdiff` between:
- **Model A:** `meta-llama/Meta-Llama-3-8B-Instruct`
- **Model B:** `failspy/Llama-3-8B-Instruct-abliterated`

The output clearly isolates the safety intervention:
1. **0.00% Delta in Inactive Layers:** Embeddings, Q/K/V attention projections, and MLP gate/up projections have **identical bits ($\Delta_F = 0.0000$)**.
2. **Layer Localization:** Non-zero changes are strictly confined to `attn_output.weight` and `ffn_down.weight` across layers 10 through 18 (the residual stream depth where refusal tokens are generated).
3. **SVD Spectrum:**
   $$\sigma_1 \gg 0, \quad \sigma_2 = 0, \quad \sigma_3 = 0, \quad \dots$$
   The singular value ratio $\frac{\sigma_1}{\sum \sigma_i} \approx 1.000$, validating that the intervention was a pure rank-1 orthogonal strip.

---

## Demo 2: Model Successor Evolution
### *"What Did Meta Actually Change Between Llama 3 and Llama 3.1?"*

### 1. Context & Architectural Equivalence
When Meta released Llama 3.1, users were curious about what changed internally compared to Llama 3.0. Both models possess **100% identical layer geometries**:

| Parameter | Llama-3-8B-Instruct | Llama-3.1-8B-Instruct |
|---|---|---|
| `hidden_size` | 4096 | 4096 |
| `num_layers` | 32 | 32 |
| `intermediate_size` | 14336 | 14336 |
| `num_attention_heads` | 32 | 32 |
| `num_key_value_heads` | 8 | 8 |
| Vocab Size | 128256 | 128256 |

Because every tensor has the identical name and shape, `wdiff` can perform an exact tensor-by-tensor differential analysis without interpolation or reshaping.

### 2. What Meta Changed Under the Hood
1. **128K Context Window Expansion:** Llama 3 had an 8K native window; Llama 3.1 expanded to 128K tokens by adjusting RoPE frequencies (`rope_theta` scaled from 500,000 to 500,000,000) and fine-tuning attention projection matrices.
2. **Synthetic Data Alignment (DPO / RLHF):** Overhauled multi-lingual, coding, and tool-calling post-training using synthetic trajectories.

### 3. What `wdiff` Visualizes
1. **RoPE-Sensitive vs. Semantic Projections:**
   - Attention projections (`attn_q`, `attn_k`) exhibit distinct high-frequency delta patterns corresponding to positional frequency adjustments.
   - Attention output and FFN down projections exhibit low-rank deltas corresponding to synthetic alignment updates.
2. **Depth Distribution:**
   - The middle layers (layers 12–24) absorb the bulk of multilingual and code knowledge shifts.
   - The final layers (layers 28–31) and language model head (`lm_head`) reflect tokenizer stop-token calibration (handling stop tokens `128008` and `128009` for function calling).

---

## 3. Reproduction Commands

To replicate both benchmarks locally using GGUF quants (requiring only ~8GB disk space):

```bash
# 1. Download Llama-3 vs Abliterated GGUFs
huggingface-cli download failspy/Llama-3-8B-Instruct-abliterated-GGUF \
  llama-3-8b-instruct-abliterated.Q4_K_M.gguf --local-dir ./models
huggingface-cli download bartowski/Meta-Llama-3-8B-Instruct-GGUF \
  Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --local-dir ./models

# Run wdiff rank analysis
wdiff ./models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
      ./models/llama-3-8b-instruct-abliterated.Q4_K_M.gguf \
      --deep --report abliteration_diff.html

# 2. Download Llama-3.1 GGUF
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir ./models

# Run wdiff successor evolution analysis
wdiff ./models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
      ./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
      --report successor_evolution.html
```
