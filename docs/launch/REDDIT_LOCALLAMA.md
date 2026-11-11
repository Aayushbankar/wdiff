# Reddit r/LocalLLaMA & r/MachineLearning Post Draft

**Title:**  
[P] wdiff: A git diff for LLM weights (SVD spectral analysis, refusal decoding, cross-format HF vs GGUF, runs in <500MB RAM)

**Subreddits:**  
`r/LocalLLaMA`, `r/MachineLearning`, `r/OpenSourceAI`

---

### Post Content:

Hey r/LocalLLaMA!

Like many of you, I spend a lot of time testing fine-tunes, merged checkpoints (SLERP/DARE/TIES), and abliterated models. But whenever I downloaded two 8B or 70B checkpoints, I was always flying completely blind:
- Did this fine-tune actually learn new rank structure, or is it mostly random SGD noise?
- Is this claimed "LoRA merge" really low rank ($\le 32$), or did the author full-rank drift the weights?
- Which layers actually had their refusal directions stripped in that "uncensored" model?
- Why is my Q4_K_M quant collapsing in perplexity? (Which layers have outlier spikes?)

To solve this, I built **`wdiff`** (`weight-diff` on PyPI). It’s an open-source CLI and spectral forensic tool that works like **`git diff` for neural network weights**.

### What makes it special for local LLM work:

1. **Runs on standard hardware (<500MB peak RAM):**
   You don't need 128GB of RAM to diff two 70B models. `wdiff` streams tensor pairs directly using an `mmap` ring buffer, computes SVD on CPU, and deallocates immediately. Peak RAM stays under 500MB regardless of model size.

2. **Diff Hugging Face Safetensors against llama.cpp GGUF directly:**
   It maps HF names (`model.layers.0.mlp.down_proj.weight`) to GGUF names (`blk.0.ffn_down.weight`) automatically, and normalizes RoPE coordinates so HF and GGUF checkpoints diff with mathematical symmetry.

3. **Random Matrix Theory (Marchenko-Pastur Law):**
   Instead of just showing Euclidean distance (where every float shifted by 0.0001), it calculates the theoretical noise floor of the weight delta matrix. Anything below the noise floor is proven to be SGD noise; anything above is true learned signal.

4. **Logit Lens Concept Decoding (`--lens`):**
   For down-projection layers, it takes the top singular vector of the weight delta and multiplies it by the unembedding matrix. It literally prints the English words that were **promoted** or **suppressed** by the fine-tune!
   *When we diffed base models against abliterated/uncensored checkpoints, it immediately showed rank-1 updates and dumped the suppressed refusal words (`unfortunately`, `apologize`, `cannot`).*

5. **Single Checkpoint Health & Quant Readiness (`wdiff inspect`):**
   Scans a model for dead neurons and outlier activation spikes (kurtosis $> 15$). If a model has hundreds of outlier spikes, you know in 30 seconds that a 4-bit quant will suffer degradation before you spend hours running `llama-quantize`.

6. **Air-Gapped HTML Report (`--report report.html`):**
   Generates a standalone single-file dashboard with an interactive Transformer Architecture Topology pipeline, layer delta charts, and a slide-over SVD inspector drawer with Marchenko-Pastur plots (zero remote CDN dependencies).

### Installation & Quick Start:
```bash
pip install weight-diff

# Diff two models with deep SVD & logit lens
wdiff base.safetensors finetune.gguf --deep --lens

# Audit a model for quantization collapse risk
wdiff inspect model.gguf --deep

# Generate an interactive HTML report
wdiff base.gguf finetune.gguf --deep --lens --report diff.html
```

Code is 100% open source under MIT:  
- **GitHub:** https://github.com/Aayushbankar/wdiff  
- **PyPI:** https://pypi.org/project/weight-diff/  

I'd love to hear what models you guys test this on and what kind of weird weight deltas or merge artifacts you find!
