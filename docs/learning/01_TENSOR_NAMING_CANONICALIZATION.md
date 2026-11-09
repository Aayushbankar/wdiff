# Module 01: Tensor Naming & Canonicalization

## 1. The Problem
When a developer attempts to compare a model from Hugging Face against its quantized `.gguf` counterpart from `llama.cpp`:
- **Hugging Face (`.safetensors`):** `model.layers.0.self_attn.q_proj.weight`
- **llama.cpp (`.gguf`):** `blk.0.attn_q.weight`

If a diffing tool matches keys by naive string equality (`names_a & names_b`), **0 tensors match**. The tool reports 100% deletion of Model A and 100% addition of Model B.

---

## 2. The Solution: Canonical GGUF Naming Standard
`wdiff` adopts GGUF-compatible naming as its internal universal representation. All incoming Hugging Face keys are automatically converted to their canonical counterparts via compiled regular expressions.

### Translation Table Across Top 5 Model Families

| Role | Hugging Face (PyTorch) Pattern | Canonical `wdiff` Pattern |
|---|---|---|
| **Token Embeddings** | `model.embed_tokens.weight` | `token_embd.weight` |
| **Attention Query** | `model.layers.{i}.self_attn.q_proj.weight` | `blk.{i}.attn_q.weight` |
| **Attention Key** | `model.layers.{i}.self_attn.k_proj.weight` | `blk.{i}.attn_k.weight` |
| **Attention Value** | `model.layers.{i}.self_attn.v_proj.weight` | `blk.{i}.attn_v.weight` |
| **Attention Output** | `model.layers.{i}.self_attn.o_proj.weight` | `blk.{i}.attn_output.weight` |
| **Input LayerNorm** | `model.layers.{i}.input_layernorm.weight` | `blk.{i}.attn_norm.weight` |
| **Post-Attn Norm** | `model.layers.{i}.post_attention_layernorm.weight` | `blk.{i}.ffn_norm.weight` *(LLaMA standard)* |
| **SwiGLU Gate** | `model.layers.{i}.mlp.gate_proj.weight` | `blk.{i}.ffn_gate.weight` |
| **SwiGLU Up** | `model.layers.{i}.mlp.up_proj.weight` | `blk.{i}.ffn_up.weight` |
| **SwiGLU Down** | `model.layers.{i}.mlp.down_proj.weight` | `blk.{i}.ffn_down.weight` |
| **Final Norm** | `model.norm.weight` | `output_norm.weight` |
| **LM Head** | `lm_head.weight` | `output.weight` |

---

## 3. Four Hidden Traps & How `wdiff` Resolves Them

### Trap 1: The Gemma-2 "Four-Norm" Architecture
In standard LLaMA models, there are 2 LayerNorms per block. Gemma-2 introduced **4 LayerNorms per block**:
1. `input_layernorm` $\rightarrow$ `blk.{i}.attn_norm`
2. `post_attention_layernorm` $\rightarrow$ `blk.{i}.post_attention_norm` *(NOT ffn_norm!)*
3. `pre_feedforward_layernorm` $\rightarrow$ `blk.{i}.ffn_norm`
4. `post_feedforward_layernorm` $\rightarrow$ `blk.{i}.post_ffw_norm`

*Resolution:* `wdiff` inspects whether `pre_feedforward_layernorm` exists in the key list. If present, it automatically activates the Gemma-2 4-norm disambiguation routing.

### Trap 2: Tied Word Embeddings
In smaller models (e.g. Gemma 2B, LLaMA-3.2 1B), `tie_word_embeddings: true` is enabled. The `lm_head.weight` is never written to disk because it reuses `embed_tokens.weight`.
*Resolution:* `wdiff` injects a **virtual alias** mapping `output.weight` to `token_embd.weight`, annotated as `[Tied: token_embd.weight]`.

### Trap 3: Fused Projections (Phi-3 / ChatGLM)
Phi-3 fuses $Q, K, V$ into `qkv_proj` ($[D_Q + D_K + D_V \times H]$) and gate/up into `gate_up_proj` ($[2I \times H]$).
*Resolution:* When comparing a fused model against an unfused model, `wdiff` uses a zero-copy strided slicing adapter along axis 0.

### Trap 4: Gemma RMSNorm Unit Offset
PyTorch Gemma evaluates norms as $x \cdot (w + 1)$. GGUF stores $w + 1$ directly.
*Resolution:* `wdiff` applies a $+1.0$ offset during HF Gemma normalization, preventing artificial ~100% delta errors.
