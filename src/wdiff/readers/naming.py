from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Architecture(Enum):
    LLAMA = "llama"
    QWEN2 = "qwen2"
    MISTRAL = "mistral"
    GEMMA = "gemma"
    GEMMA2 = "gemma2"
    PHI3 = "phi3"
    GENERIC_GGUF = "generic_gguf"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CanonicalTensor:
    canonical_name: str
    raw_name: str
    architecture: Architecture
    is_tied: bool = False


@dataclass(frozen=True, slots=True)
class AlignmentReport:
    matched_pairs: list[tuple[str, str, str]]
    only_in_a: list[str]
    only_in_b: list[str]


# Canonical GGUF naming patterns
_CANONICAL_PATTERNS = {
    "token_embd.weight": "token_embd.weight",
    "output_norm.weight": "output_norm.weight",
    "output.weight": "output.weight",
}


_LAYER_PATTERN = r"(?P<layer>\d+)"
_SUFFIX_PATTERN = r"(?P<suffix>weight|bias)"


_LLAMA_PATTERNS = [
    (re.compile(rf"model\.embed_tokens\.{_SUFFIX_PATTERN}"), "token_embd.{suffix}"),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.self_attn\.q_proj\.{_SUFFIX_PATTERN}"),
        "blk.{layer}.attn_q.{suffix}",
    ),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.self_attn\.k_proj\.{_SUFFIX_PATTERN}"),
        "blk.{layer}.attn_k.{suffix}",
    ),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.self_attn\.v_proj\.{_SUFFIX_PATTERN}"),
        "blk.{layer}.attn_v.{suffix}",
    ),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.self_attn\.o_proj\.{_SUFFIX_PATTERN}"),
        "blk.{layer}.attn_output.{suffix}",
    ),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.input_layernorm\.{_SUFFIX_PATTERN}"),
        "blk.{layer}.attn_norm.{suffix}",
    ),
    (
        re.compile(
            rf"model\.layers\.{_LAYER_PATTERN}\.post_attention_layernorm\.{_SUFFIX_PATTERN}"
        ),
        "blk.{layer}.ffn_norm.{suffix}",
    ),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.mlp\.gate_proj\.{_SUFFIX_PATTERN}"),
        "blk.{layer}.ffn_gate.{suffix}",
    ),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.mlp\.up_proj\.{_SUFFIX_PATTERN}"),
        "blk.{layer}.ffn_up.{suffix}",
    ),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.mlp\.down_proj\.{_SUFFIX_PATTERN}"),
        "blk.{layer}.ffn_down.{suffix}",
    ),
    (re.compile(rf"model\.norm\.{_SUFFIX_PATTERN}"), "output_norm.{suffix}"),
    (re.compile(rf"lm_head\.{_SUFFIX_PATTERN}"), "output.{suffix}"),
]

_QWEN2_PATTERNS = [
    *[(re.compile(p.pattern), repl) for p, repl in _LLAMA_PATTERNS],
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.self_attn\.q_proj\.(?P<suffix>bias)"),
        "blk.{layer}.attn_q.{suffix}",
    ),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.self_attn\.k_proj\.(?P<suffix>bias)"),
        "blk.{layer}.attn_k.{suffix}",
    ),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.self_attn\.v_proj\.(?P<suffix>bias)"),
        "blk.{layer}.attn_v.{suffix}",
    ),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.self_attn\.o_proj\.(?P<suffix>bias)"),
        "blk.{layer}.attn_output.{suffix}",
    ),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.mlp\.gate_proj\.(?P<suffix>bias)"),
        "blk.{layer}.ffn_gate.{suffix}",
    ),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.mlp\.up_proj\.(?P<suffix>bias)"),
        "blk.{layer}.ffn_up.{suffix}",
    ),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.mlp\.down_proj\.(?P<suffix>bias)"),
        "blk.{layer}.ffn_down.{suffix}",
    ),
]

_GEMMA_PATTERNS = _LLAMA_PATTERNS

_GEMMA2_PATTERNS = [
    (re.compile(rf"model\.embed_tokens\.{_SUFFIX_PATTERN}"), "token_embd.{suffix}"),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.self_attn\.q_proj\.{_SUFFIX_PATTERN}"),
        "blk.{layer}.attn_q.{suffix}",
    ),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.self_attn\.k_proj\.{_SUFFIX_PATTERN}"),
        "blk.{layer}.attn_k.{suffix}",
    ),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.self_attn\.v_proj\.{_SUFFIX_PATTERN}"),
        "blk.{layer}.attn_v.{suffix}",
    ),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.self_attn\.o_proj\.{_SUFFIX_PATTERN}"),
        "blk.{layer}.attn_output.{suffix}",
    ),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.input_layernorm\.{_SUFFIX_PATTERN}"),
        "blk.{layer}.attn_norm.{suffix}",
    ),
    (
        re.compile(
            rf"model\.layers\.{_LAYER_PATTERN}\.post_attention_layernorm\.{_SUFFIX_PATTERN}"
        ),
        "blk.{layer}.post_attention_norm.{suffix}",
    ),
    (
        re.compile(
            rf"model\.layers\.{_LAYER_PATTERN}\.pre_feedforward_layernorm\.{_SUFFIX_PATTERN}"
        ),
        "blk.{layer}.ffn_norm.{suffix}",
    ),
    (
        re.compile(
            rf"model\.layers\.{_LAYER_PATTERN}\.post_feedforward_layernorm\.{_SUFFIX_PATTERN}"
        ),
        "blk.{layer}.post_ffw_norm.{suffix}",
    ),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.mlp\.gate_proj\.{_SUFFIX_PATTERN}"),
        "blk.{layer}.ffn_gate.{suffix}",
    ),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.mlp\.up_proj\.{_SUFFIX_PATTERN}"),
        "blk.{layer}.ffn_up.{suffix}",
    ),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.mlp\.down_proj\.{_SUFFIX_PATTERN}"),
        "blk.{layer}.ffn_down.{suffix}",
    ),
    (re.compile(rf"model\.norm\.{_SUFFIX_PATTERN}"), "output_norm.{suffix}"),
    (re.compile(rf"lm_head\.{_SUFFIX_PATTERN}"), "output.{suffix}"),
]

_PHI3_PATTERNS = [
    (re.compile(rf"model\.embed_tokens\.{_SUFFIX_PATTERN}"), "token_embd.{suffix}"),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.self_attn\.qkv_proj\.{_SUFFIX_PATTERN}"),
        "blk.{layer}.attn_qkv.{suffix}",
    ),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.self_attn\.o_proj\.{_SUFFIX_PATTERN}"),
        "blk.{layer}.attn_output.{suffix}",
    ),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.input_layernorm\.{_SUFFIX_PATTERN}"),
        "blk.{layer}.attn_norm.{suffix}",
    ),
    (
        re.compile(
            rf"model\.layers\.{_LAYER_PATTERN}\.post_attention_layernorm\.{_SUFFIX_PATTERN}"
        ),
        "blk.{layer}.ffn_norm.{suffix}",
    ),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.mlp\.gate_up_proj\.{_SUFFIX_PATTERN}"),
        "blk.{layer}.ffn_up.{suffix}",
    ),
    (
        re.compile(rf"model\.layers\.{_LAYER_PATTERN}\.mlp\.down_proj\.{_SUFFIX_PATTERN}"),
        "blk.{layer}.ffn_down.{suffix}",
    ),
    (re.compile(rf"model\.norm\.{_SUFFIX_PATTERN}"), "output_norm.{suffix}"),
    (re.compile(rf"lm_head\.{_SUFFIX_PATTERN}"), "output.{suffix}"),
]

_GGUF_PASSTHROUGH = [
    (
        re.compile(
            rf"^(token_embd|output_norm|output|blk\.\d+\.(attn_q|attn_k|attn_v|attn_output|attn_norm|ffn_norm|ffn_gate|ffn_up|ffn_down|post_attention_norm|post_ffw_norm|attn_qkv)\.{_SUFFIX_PATTERN})$"
        ),
        lambda m: m.group(0),
    ),
]


_ARCH_PATTERNS = {
    Architecture.LLAMA: _LLAMA_PATTERNS,
    Architecture.QWEN2: _QWEN2_PATTERNS,
    Architecture.MISTRAL: _LLAMA_PATTERNS,
    Architecture.GEMMA: _GEMMA_PATTERNS,
    Architecture.GEMMA2: _GEMMA2_PATTERNS,
    Architecture.PHI3: _PHI3_PATTERNS,
    Architecture.GENERIC_GGUF: _GGUF_PASSTHROUGH,
    Architecture.UNKNOWN: _LLAMA_PATTERNS,
}


class TensorNameNormalizer:
    def __init__(self, architecture: Architecture) -> None:
        self.architecture = architecture
        self._patterns = _ARCH_PATTERNS.get(architecture, _LLAMA_PATTERNS)

    @staticmethod
    def detect_architecture(keys: list[str]) -> Architecture:
        # Check for GGUF native
        if any(
            k.startswith("blk.")
            or k in ("token_embd.weight", "output_norm.weight", "output.weight")
            for k in keys
        ) and not any(k.startswith("model.") for k in keys):
            return Architecture.GENERIC_GGUF

        # Check for Phi-3 (fused qkv and gate_up)
        if any("qkv_proj" in k for k in keys) and any("gate_up_proj" in k for k in keys):
            return Architecture.PHI3

        # Check for Gemma-2 (4 norms per layer)
        if any("pre_feedforward_layernorm" in k for k in keys) and any(
            "post_feedforward_layernorm" in k for k in keys
        ):
            return Architecture.GEMMA2

        # Check for Qwen2 (has bias in attention projections)
        if any("q_proj.bias" in k or "k_proj.bias" in k or "v_proj.bias" in k for k in keys):
            return Architecture.QWEN2

        # Check for Gemma (has input_layernorm but not 4 norms)
        if any("input_layernorm" in k for k in keys) and not any(
            "pre_feedforward_layernorm" in k for k in keys
        ):
            return Architecture.GEMMA

        # Default to LLaMA (HuggingFace format)
        if any("model.layers" in k and "self_attn" in k for k in keys):
            return Architecture.LLAMA

        # Check for llama.cpp / local format (layer.X.attn_Y.weight)
        if any("layer." in k and "attn_" in k for k in keys):
            return Architecture.LLAMA

        return Architecture.UNKNOWN

    def normalize(self, raw_name: str) -> CanonicalTensor:
        for pattern, replacement in self._patterns:
            match = pattern.match(raw_name)
            if match:
                if callable(replacement):
                    canonical = replacement(match)
                else:
                    groups = match.groupdict()
                    if "suffix" not in groups:
                        groups["suffix"] = "weight"
                    canonical = replacement.format(**groups)
                return CanonicalTensor(
                    canonical_name=canonical,
                    raw_name=raw_name,
                    architecture=self.architecture,
                )
        # No match - return as-is with UNKNOWN architecture
        return CanonicalTensor(
            canonical_name=raw_name,
            raw_name=raw_name,
            architecture=Architecture.UNKNOWN,
        )


def align_tensor_names(
    keys_a: list[str],
    keys_b: list[str],
    tie_word_embeddings: bool = False,
) -> AlignmentReport:
    arch_a = TensorNameNormalizer.detect_architecture(keys_a)
    arch_b = TensorNameNormalizer.detect_architecture(keys_b)

    normalizer_a = TensorNameNormalizer(arch_a)
    normalizer_b = TensorNameNormalizer(arch_b)

    # Normalize all keys
    canon_a = {normalizer_a.normalize(k).canonical_name: k for k in keys_a}
    canon_b = {normalizer_b.normalize(k).canonical_name: k for k in keys_b}

    # Handle tied embeddings
    if (
        tie_word_embeddings
        and "token_embd.weight" in canon_a
        and "output.weight" not in canon_a
        and "output.weight" in canon_b
    ):
        # Virtual alias: output.weight maps to token_embd.weight
        canon_a["output.weight"] = canon_a["token_embd.weight"]

    all_canonical = set(canon_a.keys()) | set(canon_b.keys())

    matched_pairs = []
    only_in_a = []
    only_in_b = []

    for canon in sorted(all_canonical):
        raw_a = canon_a.get(canon)
        raw_b = canon_b.get(canon)

        if raw_a is not None and raw_b is not None:
            matched_pairs.append((canon, raw_a, raw_b))
        elif raw_a is not None:
            only_in_a.append(raw_a)
        else:
            only_in_b.append(raw_b)

    return AlignmentReport(
        matched_pairs=matched_pairs,
        only_in_a=only_in_a,
        only_in_b=only_in_b,
    )
