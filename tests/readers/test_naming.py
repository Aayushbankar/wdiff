from __future__ import annotations

from wdiff.readers.naming import (
    Architecture,
    TensorNameNormalizer,
    align_tensor_names,
)


class TestArchitectureDetection:
    def test_detect_llama(self):
        keys = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]
        assert TensorNameNormalizer.detect_architecture(keys) == Architecture.LLAMA

    def test_detect_qwen2_via_biases(self):
        keys = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.mlp.gate_proj.weight",
        ]
        assert TensorNameNormalizer.detect_architecture(keys) == Architecture.QWEN2

    def test_detect_gemma2_via_4_norms(self):
        keys = [
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.pre_feedforward_layernorm.weight",
            "model.layers.0.post_feedforward_layernorm.weight",
        ]
        assert TensorNameNormalizer.detect_architecture(keys) == Architecture.GEMMA2

    def test_detect_phi3_via_fused_qkv(self):
        keys = [
            "model.layers.0.self_attn.qkv_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
        ]
        assert TensorNameNormalizer.detect_architecture(keys) == Architecture.PHI3

    def test_detect_native_gguf(self):
        keys = [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.0.ffn_gate.weight",
            "output_norm.weight",
        ]
        assert TensorNameNormalizer.detect_architecture(keys) == Architecture.GENERIC_GGUF


class TestTensorNormalization:
    def test_llama_normalization(self):
        normalizer = TensorNameNormalizer(Architecture.LLAMA)
        assert (
            normalizer.normalize("model.embed_tokens.weight").canonical_name == "token_embd.weight"
        )
        assert (
            normalizer.normalize("model.layers.5.self_attn.q_proj.weight").canonical_name
            == "blk.5.attn_q.weight"
        )
        assert (
            normalizer.normalize("model.layers.5.self_attn.o_proj.weight").canonical_name
            == "blk.5.attn_output.weight"
        )
        assert (
            normalizer.normalize("model.layers.5.input_layernorm.weight").canonical_name
            == "blk.5.attn_norm.weight"
        )
        assert (
            normalizer.normalize("model.layers.5.post_attention_layernorm.weight").canonical_name
            == "blk.5.ffn_norm.weight"
        )
        assert (
            normalizer.normalize("model.layers.5.mlp.gate_proj.weight").canonical_name
            == "blk.5.ffn_gate.weight"
        )
        assert (
            normalizer.normalize("model.layers.5.mlp.up_proj.weight").canonical_name
            == "blk.5.ffn_up.weight"
        )
        assert (
            normalizer.normalize("model.layers.5.mlp.down_proj.weight").canonical_name
            == "blk.5.ffn_down.weight"
        )
        assert normalizer.normalize("model.norm.weight").canonical_name == "output_norm.weight"
        assert normalizer.normalize("lm_head.weight").canonical_name == "output.weight"

    def test_qwen2_bias_normalization(self):
        normalizer = TensorNameNormalizer(Architecture.QWEN2)
        res = normalizer.normalize("model.layers.0.self_attn.q_proj.bias")
        assert res.canonical_name == "blk.0.attn_q.bias"

    def test_gemma2_four_norm_normalization(self):
        normalizer = TensorNameNormalizer(Architecture.GEMMA2)
        assert (
            normalizer.normalize("model.layers.0.input_layernorm.weight").canonical_name
            == "blk.0.attn_norm.weight"
        )
        assert (
            normalizer.normalize("model.layers.0.post_attention_layernorm.weight").canonical_name
            == "blk.0.post_attention_norm.weight"
        )
        assert (
            normalizer.normalize("model.layers.0.pre_feedforward_layernorm.weight").canonical_name
            == "blk.0.ffn_norm.weight"
        )
        assert (
            normalizer.normalize("model.layers.0.post_feedforward_layernorm.weight").canonical_name
            == "blk.0.post_ffw_norm.weight"
        )

    def test_phi3_fused_normalization(self):
        normalizer = TensorNameNormalizer(Architecture.PHI3)
        assert (
            normalizer.normalize("model.layers.0.self_attn.qkv_proj.weight").canonical_name
            == "blk.0.attn_qkv.weight"
        )
        assert (
            normalizer.normalize("model.layers.0.mlp.gate_up_proj.weight").canonical_name
            == "blk.0.ffn_up.weight"
        )

    def test_passthrough_native_gguf(self):
        normalizer = TensorNameNormalizer(Architecture.GENERIC_GGUF)
        assert normalizer.normalize("blk.0.attn_q.weight").canonical_name == "blk.0.attn_q.weight"
        assert normalizer.normalize("token_embd.weight").canonical_name == "token_embd.weight"


class TestModelAlignment:
    def test_align_hf_and_gguf_keys(self):
        hf_keys = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "lm_head.weight",
            "model.layers.0.only_hf.weight",
        ]
        gguf_keys = [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.0.ffn_gate.weight",
            "output.weight",
            "blk.0.only_gguf.weight",
        ]

        alignment = align_tensor_names(hf_keys, gguf_keys)
        assert len(alignment.matched_pairs) == 4

        # Verify matched pairs map correctly (canonical_name, raw_name_a, raw_name_b)
        matched_dict = {
            canonical: (raw_a, raw_b) for canonical, raw_a, raw_b in alignment.matched_pairs
        }
        assert "token_embd.weight" in matched_dict
        assert matched_dict["token_embd.weight"] == (
            "model.embed_tokens.weight",
            "token_embd.weight",
        )
        assert matched_dict["blk.0.attn_q.weight"] == (
            "model.layers.0.self_attn.q_proj.weight",
            "blk.0.attn_q.weight",
        )

        assert alignment.only_in_a == ["model.layers.0.only_hf.weight"]
        assert alignment.only_in_b == ["blk.0.only_gguf.weight"]

    def test_tied_embeddings_virtual_aliasing(self):
        # When model has tie_word_embeddings: true, lm_head/output.weight is omitted
        hf_keys = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ]
        gguf_keys = [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "output.weight",  # GGUF explicitly had output.weight
        ]

        alignment = align_tensor_names(hf_keys, gguf_keys, tie_word_embeddings=True)
        matched_dict = {
            canonical: (raw_a, raw_b) for canonical, raw_a, raw_b in alignment.matched_pairs
        }

        # output.weight should be matched via virtual alias pointing to embed_tokens
        assert "output.weight" in matched_dict
        assert matched_dict["output.weight"] == ("model.embed_tokens.weight", "output.weight")
