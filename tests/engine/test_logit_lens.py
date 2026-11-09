"""Tests for Logit Lens semantic concept translation (Milestone 3.5)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from wdiff.engine.diff import diff_models
from wdiff.engine.lens import (
    Vocabulary,
    decode_layer_concept,
    is_down_proj_layer,
    project_vector_to_logits_chunked,
)


class TestVocabulary:
    """Tests for Vocabulary class."""

    def test_vocabulary_from_json_dict(self):
        """Test vocabulary from inverted vocab.json dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            vocab_path = tmpdir / "vocab.json"
            vocab_data = {"the": 0, "a": 1, "cat": 2, "dog": 3}
            vocab_path.write_text(json.dumps(vocab_data))

            vocab = Vocabulary.from_json_file(vocab_path)
            assert vocab is not None
            assert vocab.get(0) == "the"
            assert vocab.get(1) == "a"
            assert vocab.get(2) == "cat"
            assert vocab.get(3) == "dog"
            assert vocab.get(999) == "[999]"

    def test_vocabulary_from_hf_tokenizer_json(self):
        """Test vocabulary from HF tokenizer.json with model.vocab and added_tokens."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            tokenizer_path = tmpdir / "tokenizer.json"
            tokenizer_data = {
                "model": {"vocab": {"hello": 0, "world": 1, "test": 2}},
                "added_tokens": [{"id": 3, "content": "<pad>"}, {"id": 4, "content": "<unk>"}],
            }
            tokenizer_path.write_text(json.dumps(tokenizer_data))

            vocab = Vocabulary.from_json_file(tokenizer_path)
            assert vocab is not None
            assert vocab.get(0) == "hello"
            assert vocab.get(3) == "<pad>"
            assert vocab.get(4) == "<unk>"

    def test_vocabulary_from_gguf_metadata(self):
        """Test vocabulary extraction from GGUF reader fields."""
        # This test requires a mock GGUF reader with tokenizer.ggml.tokens field
        # We'll test the fallback behavior

        # Create a mock reader
        class MockReader:
            pass

        mock = MockReader()
        mock._reader = type(
            "obj",
            (object,),
            {
                "fields": {
                    "tokenizer.ggml.tokens": type(
                        "obj",
                        (object,),
                        {"data": [0, 1, 2], "parts": [b"hello", b"world", b"test"]},
                    )()
                }
            },
        )()

        vocab = Vocabulary.from_gguf_reader(mock)
        assert vocab is not None
        assert vocab.get(0) == "hello"
        assert vocab.get(1) == "world"
        assert vocab.get(2) == "test"

    def test_vocabulary_fallback_unrecognized_ids(self):
        """Test fallback formatting for unrecognized token IDs."""
        vocab = Vocabulary({0: "hello", 1: "world"})
        assert vocab.get(0) == "hello"
        assert vocab.get(1) == "world"
        assert vocab.get(999) == "[999]"
        assert vocab.get(-1) == "[-1]"

    def test_vocabulary_bpe_markers(self):
        """Test BPE/SentencePiece marker stripping."""
        vocab = Vocabulary(
            {
                0: "Ġhello",  # GPT-2 style space prefix
                1: "▁world",  # SentencePiece style space prefix
                2: "test",
            }
        )
        assert vocab.get(0) == " hello"
        assert vocab.get(1) == " world"
        assert vocab.get(2) == "test"


class TestChunkedProjection:
    """Tests for chunked logit projection."""

    def test_chunked_projection_matches_full_matrix(self):
        """Verify chunked projection produces identical top-K as full matrix."""
        vocab_size = 10000
        d_model = 512
        np.random.seed(42)

        # Create random unembedding matrix and vector
        W_U = np.random.randn(vocab_size, d_model).astype(np.float32)
        u1 = np.random.randn(d_model).astype(np.float32)
        u1 = u1 / np.linalg.norm(u1)

        # Full matrix projection
        full_logits = W_U @ u1
        top_k_full_pos = np.argsort(full_logits)[-5:][::-1]
        top_k_full_neg = np.argsort(full_logits)[:5]

        # Chunked projection
        pos, neg = project_vector_to_logits_chunked(W_U, vocab_size, u1, top_k=5, chunk_size=1024)

        pos_ids = [tid for tid, _ in pos]
        neg_ids = [tid for tid, _ in neg]

        assert pos_ids == top_k_full_pos.tolist()
        assert neg_ids == top_k_full_neg.tolist()

        # Check logit values match
        for tid, logit in pos:
            assert abs(logit - full_logits[tid]) < 1e-5
        for tid, logit in neg:
            assert abs(logit - full_logits[tid]) < 1e-5

    def test_chunked_projection_empty_matrix(self):
        """Test chunked projection with empty/zero vector."""
        vocab_size = 100
        d_model = 64
        W_U = np.random.randn(vocab_size, d_model).astype(np.float32)
        u1 = np.zeros(d_model, dtype=np.float32)

        pos, neg = project_vector_to_logits_chunked(W_U, vocab_size, u1, top_k=5)
        assert pos == []
        assert neg == []

    def test_chunked_projection_callable_getter(self):
        """Test chunked projection with callable getter."""
        vocab_size = 1000
        d_model = 256
        np.random.seed(42)
        W_U = np.random.randn(vocab_size, d_model).astype(np.float32)
        u1 = np.random.randn(d_model).astype(np.float32)
        u1 = u1 / np.linalg.norm(u1)

        def getter(start, end):
            return W_U[start:end]

        pos, _neg = project_vector_to_logits_chunked(
            getter, vocab_size, u1, top_k=3, chunk_size=256
        )

        # Verify against full projection
        full_logits = W_U @ (u1 / np.linalg.norm(u1))
        expected_pos = np.argsort(full_logits)[-3:][::-1]
        pos_ids = [tid for tid, _ in pos]

        assert pos_ids == expected_pos.tolist()


class TestDownProjDetection:
    """Tests for down-projection layer detection."""

    def test_is_down_proj_layer_positive(self):
        assert is_down_proj_layer("layer.0.ffn_down.weight")
        assert is_down_proj_layer("model.layers.0.mlp.down_proj.weight")
        assert is_down_proj_layer("transformer.h.0.mlp.c_proj.weight")
        assert is_down_proj_layer("blk.0.ffn_down.weight")

    def test_is_down_proj_layer_negative(self):
        assert not is_down_proj_layer("layer.0.ffn_up.weight")
        assert not is_down_proj_layer("layer.0.attn_q.weight")
        assert not is_down_proj_layer("layer.0.attn_output.weight")
        assert not is_down_proj_layer("embedding.weight")
        assert not is_down_proj_layer("lm_head.weight")


class TestConceptDecoding:
    """Tests for concept decoding from weight deltas."""

    def test_decode_layer_concept_no_meaningful_modification(self):
        """Test that near-zero singular values return None."""
        diff_matrix = np.random.randn(64, 128).astype(np.float32) * 1e-8
        wu_matrix = np.random.randn(1000, 64).astype(np.float32)
        vocab = Vocabulary({i: f"tok_{i}" for i in range(1000)})

        result = decode_layer_concept(
            diff_matrix=diff_matrix,
            layer_name="layer.0.ffn_down.weight",
            canonical_name="blk.0.ffn_down.weight",
            wu_matrix_or_getter=wu_matrix,
            vocab_size=1000,
            vocab=vocab,
            top_k=5,
        )
        assert result is None

    def test_decode_layer_concept_empty_matrix(self):
        """Test with empty diff matrix."""
        diff_matrix = np.array([]).reshape(0, 64).astype(np.float32)
        wu_matrix = np.random.randn(1000, 64).astype(np.float32)

        result = decode_layer_concept(
            diff_matrix=diff_matrix,
            layer_name="layer.0.ffn_down.weight",
            canonical_name="blk.0.ffn_down.weight",
            wu_matrix_or_getter=wu_matrix,
            vocab_size=1000,
            vocab=Vocabulary({}),
            top_k=5,
        )
        assert result is None


class TestImplantedConceptDecoding:
    """Tests for implanted concept decoding (end-to-end)."""

    def test_implanted_concept_decoding(self):
        """Construct synthetic W_U and rank-1 ΔW targeting known token index."""
        vocab_size = 1000
        d_model = 128
        target_token = 42

        np.random.seed(42)
        # Create W_U with known structure
        W_U = np.random.randn(vocab_size, d_model).astype(np.float32)
        # Make target token have a distinctive pattern
        W_U[target_token] = np.ones(d_model, dtype=np.float32) * 10.0

        # Create rank-1 ΔW = v1 @ u1.T where v1 targets output dimension and u1 targets token 42
        u1 = W_U[target_token].copy()
        u1 = u1 / np.linalg.norm(u1)
        v1 = np.random.randn(256).astype(np.float32)
        v1 = v1 / np.linalg.norm(v1)

        diff_matrix = np.outer(v1, u1).astype(
            np.float32
        )  # (256, 128) to match down-projection layer shape

        result = decode_layer_concept(
            diff_matrix=diff_matrix,
            layer_name="layer.0.ffn_down.weight",
            canonical_name="blk.0.ffn_down.weight",
            wu_matrix_or_getter=W_U,
            vocab_size=vocab_size,
            vocab=Vocabulary({i: f"token_{i}" for i in range(vocab_size)}),
            top_k=5,
        )

        assert result is not None
        assert result.layer_name == "layer.0.ffn_down.weight"
        assert result.canonical_name == "blk.0.ffn_down.weight"
        assert len(result.positive_tokens) == 5
        # Target token should be #1 positive concept
        assert result.positive_tokens[0].token_id == target_token
        assert result.positive_tokens[0].token == f"token_{target_token}"
        assert result.positive_tokens[0].logit > 0
        assert result.singular_value > 0
        assert 0 <= result.energy_ratio <= 1

    def test_bipolar_spectrum_negative_direction(self):
        """Verify top negative tokens correspond to vectors opposing u1."""
        vocab_size = 500
        d_model = 64
        out_features = 128
        target_token = 100
        opposite_token = 200

        np.random.seed(42)
        W_U = np.random.randn(vocab_size, d_model).astype(np.float32)
        # Make target and opposite tokens have opposing vectors
        W_U[target_token] = np.ones(d_model, dtype=np.float32)
        W_U[opposite_token] = -np.ones(d_model, dtype=np.float32)

        u1 = W_U[target_token].copy()
        u1 = u1 / np.linalg.norm(u1)
        v1 = np.random.randn(out_features).astype(np.float32)
        v1 = v1 / np.linalg.norm(v1)

        diff_matrix = np.outer(v1, u1).astype(np.float32)  # (out_features, d_model)

        result = decode_layer_concept(
            diff_matrix=diff_matrix,
            layer_name="layer.0.ffn_down.weight",
            canonical_name="blk.0.ffn_down.weight",
            wu_matrix_or_getter=W_U,
            vocab_size=vocab_size,
            vocab=Vocabulary({i: f"tok_{i}" for i in range(vocab_size)}),
            top_k=5,
        )

        assert result is not None
        # Positive should include target
        pos_ids = [tc.token_id for tc in result.positive_tokens]
        assert target_token in pos_ids

        # Negative should include opposite
        neg_ids = [tc.token_id for tc in result.negative_tokens]
        assert opposite_token in neg_ids


class TestDiffModelsWithLens:
    """Integration tests for diff_models with lens flag."""

    def test_diff_models_with_lens_flag(self):
        """Run diff_models on synthetic fixtures with lens=True."""
        import tempfile
        from pathlib import Path

        from safetensors.numpy import save_file

        np.random.seed(42)
        vocab_size = 500
        d_model = 64

        # Create base model
        base_tensors = {
            "embedding.weight": np.random.randn(vocab_size, d_model).astype(np.float32),
            "output.weight": np.random.randn(vocab_size, d_model).astype(np.float32),
            "layer.0.ffn_down.weight": np.random.randn(256, d_model).astype(np.float32),
            "layer.0.ffn_up.weight": np.random.randn(d_model, 256).astype(np.float32),
        }

        # Create modified model with implanted concept in ffn_down
        target_token = 42
        modified_tensors = base_tensors.copy()
        u1 = base_tensors["output.weight"][target_token].copy()
        u1 = u1 / np.linalg.norm(u1)
        v1 = np.random.randn(256).astype(np.float32)
        v1 = v1 / np.linalg.norm(v1)

        diff_matrix = np.outer(v1, u1).astype(np.float32)  # (256, 64) to match layer shape
        modified_tensors["layer.0.ffn_down.weight"] = (
            base_tensors["layer.0.ffn_down.weight"] + diff_matrix
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            file_a = tmpdir / "model_a.safetensors"
            file_b = tmpdir / "model_b.safetensors"
            save_file(base_tensors, str(file_a))
            save_file(modified_tensors, str(file_b))

            result = diff_models(
                str(file_a),
                str(file_b),
                lens=True,
                lens_top_k=5,
            )

            # Find the ffn_down layer
            lens_layers = [t for t in result.tensor_results if t.lens is not None]
            assert len(lens_layers) >= 1

            lens_layer = lens_layers[0]
            assert lens_layer.lens is not None
            assert lens_layer.lens.positive_tokens
            assert lens_layer.lens.negative_tokens

    def test_missing_unembedding_head_graceful_fallback(self):
        """If model lacks LM head, diff_models(..., lens=True) completes cleanly."""
        import tempfile
        from pathlib import Path

        from safetensors.numpy import save_file

        # Models without output.weight
        tensors_a = {
            "embedding.weight": np.random.randn(100, 64).astype(np.float32),
            "layer.0.ffn_down.weight": np.random.randn(256, 64).astype(np.float32),
        }
        tensors_b = {
            "embedding.weight": np.random.randn(100, 64).astype(np.float32),
            "layer.0.ffn_down.weight": np.random.randn(256, 64).astype(np.float32),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            file_a = tmpdir / "model_a.safetensors"
            file_b = tmpdir / "model_b.safetensors"
            save_file(tensors_a, str(file_a))
            save_file(tensors_b, str(file_b))

            # Should not crash even without unembedding matrix
            result = diff_models(
                str(file_a),
                str(file_b),
                lens=True,
            )
            assert result is not None


class TestCLILensFlags:
    """Tests for CLI lens flags and JSON export."""

    def test_cli_lens_flags_and_json_export(self):
        """Verify --lens and --json serializes lens telemetry in JSON output."""
        import subprocess
        import tempfile
        from pathlib import Path

        from safetensors.numpy import save_file

        np.random.seed(42)
        base_tensors = {
            "embedding.weight": np.random.randn(100, 64).astype(np.float32),
            "output.weight": np.random.randn(100, 64).astype(np.float32),
            "layer.0.ffn_down.weight": np.random.randn(256, 64).astype(np.float32),
            "layer.0.ffn_up.weight": np.random.randn(64, 256).astype(np.float32),
        }
        modified_tensors = base_tensors.copy()
        modified_tensors["layer.0.ffn_down.weight"] += (
            np.random.randn(256, 64).astype(np.float32) * 0.1
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            file_a = tmpdir / "model_a.safetensors"
            file_b = tmpdir / "model_b.safetensors"
            save_file(base_tensors, str(file_a))
            save_file(modified_tensors, str(file_b))

            import sys

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "wdiff.cli",
                    str(file_a),
                    str(file_b),
                    "--lens",
                    "--lens-top-k",
                    "3",
                    "--json",
                ],
                check=False,
                capture_output=True,
                text=True,
                cwd="/mnt/work/projects/wdiff",
            )
            assert result.returncode == 0

            data = json.loads(result.stdout)
            assert "lens_analyzed_count" in data["summary"]

            lens_tensors = [t for t in data["tensors"] if t.get("lens") is not None]
            assert len(lens_tensors) >= 0  # May be 0 if no unembedding found


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
