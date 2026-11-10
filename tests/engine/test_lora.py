from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wdiff.engine.lora import (
    LoRAAdapter,
    LoRADiffResult,
    apply_lora,
    diff_with_lora,
    load_lora_adapter,
)


class TestLoRAAdapter:
    def test_load_lora_adapter_safetensors(self, tmp_path):
        """Test loading LoRA adapter from safetensors format."""
        # Create a mock LoRA adapter
        rank = 8
        lora_data = {
            "base_model.layers.0.self_attn.q_proj.lora_A.weight": np.random.randn(rank, 64).astype(
                np.float32
            ),
            "base_model.layers.0.self_attn.q_proj.lora_B.weight": np.random.randn(256, rank).astype(
                np.float32
            ),
            "base_model.layers.0.self_attn.k_proj.lora_A.weight": np.random.randn(rank, 64).astype(
                np.float32
            ),
            "base_model.layers.0.self_attn.k_proj.lora_B.weight": np.random.randn(256, rank).astype(
                np.float32
            ),
            "base_model.layers.0.self_attn.v_proj.lora_A.weight": np.random.randn(rank, 64).astype(
                np.float32
            ),
            "base_model.layers.0.self_attn.v_proj.lora_B.weight": np.random.randn(256, rank).astype(
                np.float32
            ),
            "base_model.layers.0.self_attn.o_proj.lora_A.weight": np.random.randn(rank, 256).astype(
                np.float32
            ),
            "base_model.layers.0.self_attn.o_proj.lora_B.weight": np.random.randn(256, rank).astype(
                np.float32
            ),
        }

        adapter_path = tmp_path / "adapter.safetensors"
        from safetensors.numpy import save_file

        save_file(lora_data, str(adapter_path))

        adapter = load_lora_adapter(str(adapter_path))

        assert isinstance(adapter, LoRAAdapter)
        assert adapter.rank == 8
        assert "base_model.layers.0.self_attn.q_proj" in adapter.layers
        assert len(adapter.layers) == 4  # q, k, v, o projections

    def test_load_lora_adapter_reads_alpha_from_config(self, tmp_path):
        """Sibling adapter_config.json (PEFT) supplies lora_alpha for alpha/r scaling."""
        from safetensors.numpy import save_file

        rank = 8
        lora_data = {
            "base_model.layers.0.self_attn.q_proj.lora_A.weight": np.random.randn(rank, 64).astype(
                np.float32
            ),
            "base_model.layers.0.self_attn.q_proj.lora_B.weight": np.random.randn(64, rank).astype(
                np.float32
            ),
        }
        adapter_path = tmp_path / "adapter_model.safetensors"
        save_file(lora_data, str(adapter_path))
        (tmp_path / "adapter_config.json").write_text(
            '{"lora_alpha": 16, "r": 8}', encoding="utf-8"
        )

        adapter = load_lora_adapter(str(adapter_path))
        layer = adapter.layers["base_model.layers.0.self_attn.q_proj"]
        assert layer.alpha == 16.0
        assert layer.scale() == pytest.approx(16.0 / rank)

        # No sibling config -> alpha stays None, scale falls back to 1.0
        no_cfg_dir = tmp_path / "no_config"
        no_cfg_dir.mkdir()
        bare_path = no_cfg_dir / "adapter.safetensors"
        save_file(lora_data, str(bare_path))
        bare = load_lora_adapter(str(bare_path))
        assert bare.layers["base_model.layers.0.self_attn.q_proj"].alpha is None
        assert bare.layers["base_model.layers.0.self_attn.q_proj"].scale() == 1.0

    def test_load_lora_adapter_ignores_malformed_config(self, tmp_path):
        """Malformed or alpha-less adapter_config.json falls back to scale 1.0."""
        from safetensors.numpy import save_file

        lora_data = {
            "base_model.layers.0.self_attn.q_proj.lora_A.weight": np.random.randn(8, 64).astype(
                np.float32
            ),
            "base_model.layers.0.self_attn.q_proj.lora_B.weight": np.random.randn(64, 8).astype(
                np.float32
            ),
        }
        adapter_path = tmp_path / "adapter.safetensors"
        save_file(lora_data, str(adapter_path))

        (tmp_path / "adapter_config.json").write_text("{not json", encoding="utf-8")
        adapter = load_lora_adapter(str(adapter_path))
        assert adapter.layers["base_model.layers.0.self_attn.q_proj"].alpha is None

        (tmp_path / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
        adapter = load_lora_adapter(str(adapter_path))
        assert adapter.layers["base_model.layers.0.self_attn.q_proj"].alpha is None

    def test_apply_lora_to_base_model(self):
        """Test applying LoRA adapter to base model weights."""
        # Base weight matrix
        base_W = np.random.randn(256, 64).astype(np.float32)

        # LoRA matrices (rank 8)
        lora_A = np.random.randn(8, 64).astype(np.float32) * 0.01
        lora_B = np.random.randn(256, 8).astype(np.float32) * 0.01

        merged_W = apply_lora(base_W, lora_A, lora_B, scale=1.0)

        # Merged weight should have same shape
        assert merged_W.shape == base_W.shape
        # Merged weight should differ from base
        assert not np.allclose(merged_W, base_W)
        # Difference should be approximately B @ A (within float32 precision)
        expected_delta = lora_B @ lora_A
        # Use relaxed tolerance for float32 precision
        assert np.allclose(merged_W - base_W, expected_delta, rtol=1e-4, atol=1e-6)

    def test_apply_lora_with_scale(self):
        """Test applying LoRA with scaling factor."""
        base_W = np.zeros((10, 10), dtype=np.float32)
        lora_A = np.ones((4, 10), dtype=np.float32)
        lora_B = np.ones((10, 4), dtype=np.float32)

        # Scale = 2.0
        merged_W = apply_lora(base_W, lora_A, lora_B, scale=2.0)

        # B @ A = 4 * ones(10, 10), scaled by 2 = 8 * ones
        expected = 2.0 * (lora_B @ lora_A)
        assert np.allclose(merged_W, expected)

    def test_lora_adapter_with_alpha_scaling(self):
        """Test LoRA with alpha/r scaling (common in PEFT)."""
        base_W = np.zeros((10, 10), dtype=np.float32)
        lora_A = np.ones((4, 10), dtype=np.float32)
        lora_B = np.ones((10, 4), dtype=np.float32)

        alpha = 16
        rank = 4
        scale = alpha / rank  # = 4.0

        merged_W = apply_lora(base_W, lora_A, lora_B, scale=scale)
        expected = scale * (lora_B @ lora_A)
        assert np.allclose(merged_W, expected)


class TestLoRADiff:
    def test_diff_with_lora_adapter(self, tmp_path):
        """Test diffing base model vs base+LoRA merged model."""
        fixtures = Path(__file__).parent.parent / "fixtures"
        base_model = str(fixtures / "model_a.safetensors")

        # Create a LoRA adapter
        # Note: tensor names in fixtures are like "layer.0.attn_q.weight" with shape (256, 256)
        # LoRA adapter layer names should match tensor name without ".weight" suffix
        # lora_A: (rank, in_features) = (8, 256), lora_B: (out_features, rank) = (256, 8)
        rank = 8
        in_features = 256
        out_features = 256
        lora_data = {
            "layer.0.attn_q.lora_A.weight": np.random.randn(rank, in_features).astype(np.float32)
            * 0.01,
            "layer.0.attn_q.lora_B.weight": np.random.randn(out_features, rank).astype(np.float32)
            * 0.01,
            "layer.0.attn_k.lora_A.weight": np.random.randn(rank, in_features).astype(np.float32)
            * 0.01,
            "layer.0.attn_k.lora_B.weight": np.random.randn(out_features, rank).astype(np.float32)
            * 0.01,
            "layer.0.attn_v.lora_A.weight": np.random.randn(rank, in_features).astype(np.float32)
            * 0.01,
            "layer.0.attn_v.lora_B.weight": np.random.randn(out_features, rank).astype(np.float32)
            * 0.01,
        }
        adapter_path = tmp_path / "test_adapter.safetensors"
        from safetensors.numpy import save_file

        save_file(lora_data, str(adapter_path))

        # Diff with LoRA
        result = diff_with_lora(
            base_model_path=base_model,
            lora_path=str(adapter_path),
            lora_scale=1.0,
        )

        assert isinstance(result, LoRADiffResult)
        assert len(result.tensor_results) > 0
        # Should have matched layers with non-zero deltas
        assert any(t.metrics and t.metrics.relative_frobenius > 0 for t in result.tensor_results)

    def test_lora_scale_flag_scales_delta(self, tmp_path):
        """--lora-scale must multiply the adapter delta (previously ignored)."""
        from safetensors.numpy import save_file

        fixtures = Path(__file__).parent.parent / "fixtures"
        base_model = str(fixtures / "model_a.safetensors")

        rank = 8
        lora_data = {
            "layer.0.attn_q.lora_A.weight": np.random.randn(rank, 256).astype(np.float32) * 0.01,
            "layer.0.attn_q.lora_B.weight": np.random.randn(256, rank).astype(np.float32) * 0.01,
        }
        adapter_path = tmp_path / "adapter.safetensors"
        save_file(lora_data, str(adapter_path))

        result_1x = diff_with_lora(
            base_model_path=base_model,
            lora_path=str(adapter_path),
            lora_scale=1.0,
            tensor_filter="layer\\.0\\.attn_q",
        )
        result_2x = diff_with_lora(
            base_model_path=base_model,
            lora_path=str(adapter_path),
            lora_scale=2.0,
            tensor_filter="layer\\.0\\.attn_q",
        )

        t1 = next(t for t in result_1x.tensor_results if t.metrics.relative_frobenius > 0)
        t2 = next(t for t in result_2x.tensor_results if t.metrics.relative_frobenius > 0)
        assert t1.name == t2.name
        # Frobenius norm of the delta is exactly linear in scale; ||W|| is unchanged
        assert np.isclose(
            t2.metrics.relative_frobenius, 2.0 * t1.metrics.relative_frobenius, rtol=1e-4
        )

    def test_diff_lora_vs_base_directly(self, tmp_path):
        """Test comparing base model directly with LoRA-merged model."""
        fixtures = Path(__file__).parent.parent / "fixtures"
        base_model = str(fixtures / "model_a.safetensors")

        # Create LoRA and merged model
        lora_data = {
            "layer.0.attn_q.lora_A.weight": np.random.randn(8, 256).astype(np.float32) * 0.01,
            "layer.0.attn_q.lora_B.weight": np.random.randn(256, 8).astype(np.float32) * 0.01,
        }
        adapter_path = tmp_path / "adapter.safetensors"
        from safetensors.numpy import save_file

        save_file(lora_data, str(adapter_path))

        # This should work when fully implemented
        result = diff_with_lora(
            base_model_path=base_model,
            lora_path=str(adapter_path),
            lora_scale=1.0,
        )

        assert result is not None


class TestLoRACLI:
    def test_cli_lora_flag(self, tmp_path, capsys):
        """Test CLI --lora flag."""
        fixtures = Path(__file__).parent.parent / "fixtures"
        str(fixtures / "model_a.safetensors")

        # Create minimal LoRA adapter
        # Use fixture-compatible names: layer.0.attn_q with correct dimensions (256x256)
        lora_data = {
            "layer.0.attn_q.lora_A.weight": np.random.randn(8, 256).astype(np.float32),
            "layer.0.attn_q.lora_B.weight": np.random.randn(256, 8).astype(np.float32),
        }
        adapter_path = tmp_path / "adapter.safetensors"
        from safetensors.numpy import save_file

        save_file(lora_data, str(adapter_path))

        from wdiff.cli import main

        code = main(
            [
                str(fixtures / "model_a.safetensors"),
                "--lora",
                str(adapter_path),
                "--lora-scale",
                "1.0",
                "--filter",
                "attn_q",
            ]
        )
        assert code == 0
        out, _ = capsys.readouterr()
        assert "layer.0.attn_q" in out
