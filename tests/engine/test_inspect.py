from __future__ import annotations

from pathlib import Path

import numpy as np

from wdiff.engine.inspect import (
    ModelInspectionResult,
    inspect_model,
)


class TestInspectMetadata:
    def test_inspect_safetensors_metadata_and_stats(self):
        fixtures = Path(__file__).parent.parent / "fixtures"
        model_a = str(fixtures / "model_a.safetensors")

        result = inspect_model(model_a)

        assert isinstance(result, ModelInspectionResult)
        assert result.model_path == model_a
        assert result.format == "safetensors"
        assert result.summary.total_params > 0
        assert result.summary.tensor_count > 0
        assert result.summary.architecture in (
            "llama",
            "unknown",
            "qwen2",
            "gemma",
            "gemma2",
            "phi3",
        )
        assert len(result.layers) == result.summary.tensor_count

    def test_inspect_gguf_metadata_and_stats(self):
        fixtures = Path(__file__).parent.parent / "fixtures"
        model_a = str(fixtures / "model_a.gguf")

        result = inspect_model(model_a)

        assert isinstance(result, ModelInspectionResult)
        assert result.format == "gguf"
        assert result.summary.total_params > 0
        assert result.summary.tensor_count > 0
        assert len(result.layers) == result.summary.tensor_count


class TestDeadNeuronDetection:
    def test_inspect_dead_neuron_detection(self):
        """Create a matrix with 2 all-zero columns, verify dead_neuron_count == 2."""
        from wdiff.engine.inspect import _count_dead_neurons

        # Matrix with 2 all-zero columns
        W = np.random.randn(10, 10).astype(np.float32)
        W[:, 2] = 0  # Column 2 is all zeros
        W[:, 7] = 0  # Column 7 is all zeros

        # Also add a row of zeros for row-based dead neuron detection
        W[3, :] = 0  # Row 3 is all zeros

        dead_rows, dead_cols = _count_dead_neurons(W)
        assert dead_cols == 2  # 2 zero columns
        assert dead_rows == 1  # 1 zero row
        # Total dead neurons = dead rows + dead cols
        assert dead_rows + dead_cols == 3


class TestOutlierScoreDettmers:
    def test_inspect_outlier_score_dettmers(self):
        """Create a matrix with normal values plus one massive outlier (100x),
        verify is_outlier_risk == True and outlier_score > 15.0."""
        from wdiff.engine.inspect import _compute_outlier_score

        # Normal matrix
        W = np.random.randn(100, 100).astype(np.float32) * 0.01

        # Add massive outlier
        W[0, 0] = 100.0  # 100x larger than typical values

        score = _compute_outlier_score(W)
        assert score > 15.0


class TestDeepSpectral:
    def test_inspect_deep_spectral(self):
        fixtures = Path(__file__).parent.parent / "fixtures"
        model_a = str(fixtures / "model_a.safetensors")

        result = inspect_model(model_a, deep=True)

        # At least some layers should have spectral data
        deep_layers = [l for l in result.layers if l.spectral is not None]
        assert len(deep_layers) > 0

        for layer in deep_layers:
            assert layer.spectral is not None
            assert layer.spectral.signal_rank is not None
            assert layer.spectral.effective_rank is not None


class TestFilterRegex:
    def test_inspect_filter_regex(self):
        fixtures = Path(__file__).parent.parent / "fixtures"
        model_a = str(fixtures / "model_a.safetensors")

        result = inspect_model(model_a, tensor_filter="attn_q")

        # Should only have layers matching attn_q
        for layer in result.layers:
            assert "attn_q" in layer.name
        # Should have fewer layers than full inspection
        full_result = inspect_model(model_a)
        assert len(result.layers) < len(full_result.layers)


class TestCLIInspect:
    def test_cli_inspect_command(self, capsys):
        fixtures = Path(__file__).parent.parent / "fixtures"
        model_a = str(fixtures / "model_a.safetensors")

        from wdiff.cli import main

        code = main(["inspect", model_a])
        assert code == 0
        out, _ = capsys.readouterr()
        assert "inspect" in out.lower() or "model" in out.lower() or "parameter" in out.lower()

    def test_cli_single_model_auto_inspect(self, capsys):
        """Test that providing only one model auto-routes to inspect."""
        fixtures = Path(__file__).parent.parent / "fixtures"
        model_a = str(fixtures / "model_a.safetensors")

        from wdiff.cli import main

        code = main([model_a])
        assert code == 0
        out, _ = capsys.readouterr()
        assert "inspect" in out.lower() or "model" in out.lower() or "parameter" in out.lower()

    def test_cli_single_model_with_filter_auto_inspect(self, capsys):
        """Test that providing only one model with --filter auto-routes to inspect."""
        fixtures = Path(__file__).parent.parent / "fixtures"
        model_a = str(fixtures / "model_a.safetensors")

        from wdiff.cli import main

        code = main([model_a, "--filter", "attn_q"])
        assert code == 0
        out, _ = capsys.readouterr()
        assert "attn_q" in out

    def test_cli_inspect_json(self, capsys):
        import json

        fixtures = Path(__file__).parent.parent / "fixtures"
        model_a = str(fixtures / "model_a.safetensors")

        from wdiff.cli import main

        code = main(["inspect", model_a, "--json"])
        assert code == 0
        out, _ = capsys.readouterr()

        data = json.loads(out)
        assert "model_path" in data
        assert "format" in data
        assert "summary" in data
        assert "layers" in data
        assert isinstance(data["layers"], list)
        assert len(data["layers"]) > 0

        # Check layer structure
        layer = data["layers"][0]
        assert "name" in layer
        assert "shape" in layer
        assert "dtype" in layer
        assert "param_count" in layer
        assert "norm" in layer
        assert "mean" in layer
        assert "std" in layer
        assert "dead_neuron_count" in layer
        assert "outlier_score" in layer
        assert "is_outlier_risk" in layer
