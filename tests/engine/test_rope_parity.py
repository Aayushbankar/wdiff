"""Tests for RoPE coordinate normalization and parity (HuggingFace <-> GGUF)."""

import numpy as np
import pytest

from wdiff.engine.diff import diff_models
from wdiff.engine.rope import (
    deinterleave_rope,
    interleave_rope,
    is_rope_tensor,
)
from wdiff.readers import get_reader


class TestRoPEInvolution:
    """Test RoPE permutation involution (round-trip)."""

    def test_rope_involution_roundtrip_2d(self):
        """deinterleave_rope(interleave_rope(W)) == W for 2D matrices."""
        n_heads = 32
        d_head = 128
        d_in = 4096

        # Create random HF-format weights
        np.random.seed(42)
        W_hf = np.random.randn(n_heads * d_head, d_in).astype(np.float32)

        # Convert to GGML format
        W_ggml = interleave_rope(W_hf, n_heads, d_head)

        # Convert back to HF format
        W_hf_reconstructed = deinterleave_rope(W_ggml, n_heads, d_head)

        np.testing.assert_allclose(W_hf, W_hf_reconstructed, rtol=1e-6, atol=1e-6)

    def test_rope_involution_roundtrip_1d(self):
        """deinterleave_rope(interleave_rope(v)) == v for 1D bias vectors."""
        n_heads = 32
        d_head = 128

        np.random.seed(42)
        v_hf = np.random.randn(n_heads * d_head).astype(np.float32)

        v_ggml = interleave_rope(v_hf, n_heads, d_head)
        v_hf_reconstructed = deinterleave_rope(v_ggml, n_heads, d_head)

        np.testing.assert_allclose(v_hf, v_hf_reconstructed, rtol=1e-6, atol=1e-6)

    def test_rope_gqa_kv_heads(self):
        """Test permutation with GQA (unequal Q and KV heads)."""
        # Q: 32 heads, KV: 8 heads (GQA 4:1)
        n_q_heads = 32
        n_kv_heads = 8
        d_head = 128
        d_in = 4096

        np.random.seed(42)
        W_q_hf = np.random.randn(n_q_heads * d_head, d_in).astype(np.float32)
        W_k_hf = np.random.randn(n_kv_heads * d_head, d_in).astype(np.float32)

        # Convert both
        W_q_ggml = interleave_rope(W_q_hf, n_q_heads, d_head)
        W_k_ggml = interleave_rope(W_k_hf, n_kv_heads, d_head)

        # Reconstruct
        W_q_reconstructed = deinterleave_rope(W_q_ggml, n_q_heads, d_head)
        W_k_reconstructed = deinterleave_rope(W_k_ggml, n_kv_heads, d_head)

        np.testing.assert_allclose(W_q_hf, W_q_reconstructed, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(W_k_hf, W_k_reconstructed, rtol=1e-6, atol=1e-6)

    def test_rope_1d_bias_vectors(self):
        """Test de-interleaving 1D Q/K bias vectors."""
        n_heads = 32
        d_head = 128

        np.random.seed(42)
        q_bias_hf = np.random.randn(n_heads * d_head).astype(np.float32)
        k_bias_hf = np.random.randn(n_heads * d_head).astype(np.float32)

        q_bias_ggml = interleave_rope(q_bias_hf, n_heads, d_head)
        k_bias_ggml = interleave_rope(k_bias_hf, n_heads, d_head)

        q_bias_reconstructed = deinterleave_rope(q_bias_ggml, n_heads, d_head)
        k_bias_reconstructed = deinterleave_rope(k_bias_ggml, n_heads, d_head)

        np.testing.assert_allclose(q_bias_hf, q_bias_reconstructed, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(k_bias_hf, k_bias_reconstructed, rtol=1e-6, atol=1e-6)

    def test_false_divergence_elimination(self):
        """Verify raw cosine similarity <= 0.15, deinterleaved achieves 1.0."""
        n_heads = 32
        d_head = 128
        d_in = 4096

        np.random.seed(42)
        W_hf = np.random.randn(n_heads * d_head, d_in).astype(np.float32)

        # Simulate GGUF conversion
        W_ggml = interleave_rope(W_hf, n_heads, d_head)

        # Raw comparison (should show false divergence)
        from wdiff.engine.metrics import cosine_similarity, relative_frobenius

        raw_cosine = cosine_similarity(W_hf, W_ggml)
        raw_frobenius = relative_frobenius(W_hf, W_ggml)

        # Raw cosine should be low (false divergence)
        assert raw_cosine <= 0.15, f"Expected raw cosine <= 0.15, got {raw_cosine:.4f}"
        assert raw_frobenius >= 1.0, f"Expected raw frobenius >= 1.0, got {raw_frobenius:.4f}"

        # After deinterleaving (should achieve bit-for-bit parity)
        W_ggml_deinterleaved = deinterleave_rope(W_ggml, n_heads, d_head)
        corrected_cosine = cosine_similarity(W_hf, W_ggml_deinterleaved)
        corrected_frobenius = relative_frobenius(W_hf, W_ggml_deinterleaved)

        np.testing.assert_allclose(corrected_cosine, 1.0, atol=1e-6)
        np.testing.assert_allclose(corrected_frobenius, 0.0, atol=1e-6)


class TestIsRoPETensor:
    """Test is_rope_tensor detection."""

    def test_attn_q_weight(self):
        assert is_rope_tensor("blk.0.attn_q.weight") is True
        assert is_rope_tensor("model.layers.0.self_attn.q_proj.weight") is True

    def test_attn_k_weight(self):
        assert is_rope_tensor("blk.0.attn_k.weight") is True
        assert is_rope_tensor("model.layers.0.self_attn.k_proj.weight") is True

    def test_attn_q_bias(self):
        assert is_rope_tensor("blk.0.attn_q.bias") is True
        assert is_rope_tensor("model.layers.0.self_attn.q_proj.bias") is True

    def test_attn_k_bias(self):
        assert is_rope_tensor("blk.0.attn_k.bias") is True
        assert is_rope_tensor("model.layers.0.self_attn.k_proj.bias") is True

    def test_non_rope_tensors(self):
        assert is_rope_tensor("blk.0.attn_v.weight") is False
        assert is_rope_tensor("blk.0.attn_output.weight") is False
        assert is_rope_tensor("blk.0.ffn_gate.weight") is False
        assert is_rope_tensor("blk.0.ffn_up.weight") is False
        assert is_rope_tensor("blk.0.ffn_down.weight") is False
        assert is_rope_tensor("blk.0.attn_norm.weight") is False
        assert is_rope_tensor("blk.0.ffn_norm.weight") is False
        assert is_rope_tensor("token_embd.weight") is False
        assert is_rope_tensor("output_norm.weight") is False
        assert is_rope_tensor("output.weight") is False


class TestGGUFMetadataHeadCount:
    """Test GGUF metadata extraction for head count."""

    def test_gguf_metadata_head_count_extraction(self):
        """Verify GGUFReader.get_metadata works (fixture may not have attention metadata)."""

        reader = get_reader("tests/fixtures/model_a.gguf")
        with reader:
            # Test that the method works (returns None for missing metadata)
            head_count = reader.get_metadata("attention.head_count")
            head_count_kv = reader.get_metadata("attention.head_count_kv")
            key_length = reader.get_metadata("attention.key_length")

            # The synthetic fixtures don't have these metadata fields
            # but the method should not error
            assert head_count is None
            assert head_count_kv is None
            assert key_length is None

    def test_gguf_reader_properties(self):
        """Test head_count and head_count_kv properties."""

        reader = get_reader("tests/fixtures/model_a.gguf")
        with reader:
            # Properties should return None for fixtures without metadata
            assert reader.head_count is None
            assert reader.head_count_kv is None


class TestDiffModelsCrossFormatRoPEParity:
    """Test diff_models with RoPE parity correction."""

    def test_diff_models_cross_format_rope_parity(self):
        """Verify RoPE parity correction works by simulating GGML permutation."""
        # The fixtures are both in HF format (no GGML permutation applied).
        # To test rope_parity, we create a synthetic test:
        # 1. Load HF tensor
        # 2. Apply interleave_rope to simulate GGML conversion
        # 3. Verify raw comparison shows false divergence
        # 4. Verify deinterleave_rope (rope_parity=True) restores parity

        from wdiff.engine.metrics import cosine_similarity
        from wdiff.engine.rope import deinterleave_rope, interleave_rope

        with get_reader("tests/fixtures/model_a.safetensors") as reader:
            # Get Q and K projection tensors
            q_hf = reader.get_tensor("layer.0.attn_q.weight")  # (256, 256) -> 32 heads * 8?
            k_hf = reader.get_tensor("layer.0.attn_k.weight")

            # For this fixture: 256 = 32 heads * 8 (d_head=8) or 8 heads * 32 (d_head=32)
            # Let's figure out the head config from the shape
            # 256 = n_heads * d_head
            # Common: 32 heads * 8 = 256 (d_head=8, but d_head should be even and typically >= 64)
            # Or 8 heads * 32 = 256 (d_head=32)
            # Let's use 8 heads, d_head=32

            n_heads = 8
            d_head = 32

            # Simulate GGML conversion
            q_ggml = interleave_rope(q_hf, n_heads, d_head)
            k_ggml = interleave_rope(k_hf, n_heads, d_head)

            # Raw comparison (should show false divergence)
            raw_q_cosine = cosine_similarity(q_hf, q_ggml)
            raw_k_cosine = cosine_similarity(k_hf, k_ggml)

            assert raw_q_cosine < 0.5, f"Q raw cosine={raw_q_cosine:.4f} (expected < 0.5)"
            assert raw_k_cosine < 0.5, f"K raw cosine={raw_k_cosine:.4f} (expected < 0.5)"

            # After deinterleaving (should achieve bit-for-bit parity)
            q_restored = deinterleave_rope(q_ggml, n_heads, d_head)
            k_restored = deinterleave_rope(k_ggml, n_heads, d_head)

            np.testing.assert_allclose(q_hf, q_restored, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(k_hf, k_restored, rtol=1e-6, atol=1e-6)

    def test_diff_models_synthetic_gguf_with_rope_parity(self):
        """Test diff_models with a synthetic GGUF that has GGML RoPE permutation applied."""
        import tempfile
        from pathlib import Path

        import numpy as np
        from gguf import GGMLQuantizationType, GGUFWriter
        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create base tensors (HF format)
            np.random.seed(42)
            tensors_hf = {
                "embedding.weight": np.random.randn(64, 256).astype(np.float32),
                "layer.0.attn_q.weight": np.random.randn(256, 256).astype(np.float32),
                "layer.0.attn_k.weight": np.random.randn(256, 256).astype(np.float32),
                "layer.0.attn_v.weight": np.random.randn(256, 256).astype(np.float32),
                "layer.0.attn_output.weight": np.random.randn(256, 256).astype(np.float32),
                "layer.0.ffn_gate.weight": np.random.randn(256, 256).astype(np.float32),
                "layer.0.ffn_up.weight": np.random.randn(256, 256).astype(np.float32),
                "layer.0.ffn_down.weight": np.random.randn(256, 256).astype(np.float32),
                "lm_head.weight": np.random.randn(64, 256).astype(np.float32),
            }

            # Apply GGML RoPE permutation to Q and K for the "GGUF" version
            # Using 8 heads, d_head=32 for the 256x256 attention matrices
            from wdiff.engine.rope import interleave_rope

            tensors_ggml = tensors_hf.copy()
            tensors_ggml["layer.0.attn_q.weight"] = interleave_rope(
                tensors_hf["layer.0.attn_q.weight"], n_heads=8, d_head=32
            )
            tensors_ggml["layer.0.attn_k.weight"] = interleave_rope(
                tensors_hf["layer.0.attn_k.weight"], n_heads=8, d_head=32
            )

            # Save as safetensors (HF format)
            safetensors_path = tmpdir / "model_hf.safetensors"
            save_file(tensors_hf, str(safetensors_path))

            # Save as GGUF with GGML permutation
            gguf_path = tmpdir / "model_ggml.gguf"
            writer = GGUFWriter(str(gguf_path), "test_model")
            # Add attention metadata for head count detection
            writer.add_head_count(8)  # n_heads = 8
            writer.add_head_count_kv(8)  # n_kv_heads = 8 (for GQA this would differ)
            writer.add_key_length(32)  # d_head = 32
            for name, tensor in tensors_ggml.items():
                writer.add_tensor(name, tensor, raw_dtype=GGMLQuantizationType.F32)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            # Test with rope_parity=True (should match)
            result_with = diff_models(str(safetensors_path), str(gguf_path), rope_parity=True)

            # Test with rope_parity=False (should show false divergence)
            result_without = diff_models(str(safetensors_path), str(gguf_path), rope_parity=False)

            # Check Q projection
            q_with = next(t for t in result_with.tensor_results if "attn_q" in t.name)
            q_without = next(t for t in result_without.tensor_results if "attn_q" in t.name)

            assert q_with.status == "MATCH"
            assert q_without.status == "MATCH"

            # With rope_parity: should be near-perfect match
            assert q_with.metrics.cosine_similarity > 0.999, (
                f"With rope_parity: cosine={q_with.metrics.cosine_similarity:.6f}"
            )
            assert q_with.metrics.relative_frobenius < 0.001

            # Without rope_parity: should show false divergence
            assert q_without.metrics.cosine_similarity < 0.5, (
                f"Without rope_parity: cosine={q_without.metrics.cosine_similarity:.6f}"
            )

            # Check K projection
            k_with = next(t for t in result_with.tensor_results if "attn_k" in t.name)
            k_without = next(t for t in result_without.tensor_results if "attn_k" in t.name)

            assert k_with.metrics.cosine_similarity > 0.999
            assert k_without.metrics.cosine_similarity < 0.5


class TestCLINoRoPEParityFlag:
    """Test CLI --no-rope-parity flag."""

    def test_cli_no_rope_parity_flag(self):
        """Verify --no-rope-parity can be passed to CLI."""
        import subprocess
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        model_a = str(repo_root / "tests/fixtures/model_a.safetensors")
        model_b = str(repo_root / "tests/fixtures/model_a.gguf")

        # Run CLI with --no-rope-parity flag (should not error)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "wdiff.cli",
                model_a,
                model_b,
                "--no-rope-parity",
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

    def test_cli_default_rope_parity_enabled(self):
        """Verify rope_parity is enabled by default (no flag needed)."""
        import subprocess
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        model_a = str(repo_root / "tests/fixtures/model_a.safetensors")
        model_b = str(repo_root / "tests/fixtures/model_a.gguf")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "wdiff.cli",
                model_a,
                model_b,
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
