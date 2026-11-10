"""Tests for Partial Shape Matching & Vocabulary Expansion Slicing (Milestone 3.4)."""

import numpy as np
import pytest

from wdiff.engine.diff import diff_models


class TestPartialShapeMatching:
    """Tests for single-axis common-prefix slicing (vocabulary expansion)."""

    def test_partial_match_vocabulary_expansion_2d(self):
        """Base (128, 64) vs Expanded (160, 64) - vocabulary expansion on axis 0."""
        # Create identical base tensors
        np.random.seed(42)
        base = np.random.randn(128, 64).astype(np.float32)

        # Expanded model has 32 extra tokens
        extra = np.random.randn(32, 64).astype(np.float32)
        expanded = np.vstack([base, extra])

        # Save as temporary files
        import tempfile
        from pathlib import Path

        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            file_a = tmpdir / "base.safetensors"
            file_b = tmpdir / "expanded.safetensors"

            save_file({"embedding.weight": base}, str(file_a))
            save_file({"embedding.weight": expanded}, str(file_b))

            result = diff_models(str(file_a), str(file_b))

            # Find the embedding tensor
            t = next(t for t in result.tensor_results if t.name == "embedding.weight")

            assert t.status == "PARTIAL_MATCH"
            assert t.metrics is not None
            # Shared 128 rows should be identical
            assert t.metrics.cosine_similarity == pytest.approx(1.0, abs=1e-6)
            assert t.metrics.relative_frobenius == pytest.approx(0.0, abs=1e-6)

            # Check partial_info
            assert t.partial_info is not None
            assert t.partial_info.common_shape == (128, 64)
            assert t.partial_info.differing_axis == 0
            assert t.partial_info.extra_count == 32
            assert t.partial_info.extra_source == "model_b"
            assert t.partial_info.extra_norm > 0

    def test_partial_match_truncated_vocabulary(self):
        """Base (160, 64) vs Pruned (128, 64) - extra tokens in model_a."""
        np.random.seed(42)
        base = np.random.randn(160, 64).astype(np.float32)
        pruned = base[:128, :]

        import tempfile
        from pathlib import Path

        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            file_a = tmpdir / "base.safetensors"
            file_b = tmpdir / "pruned.safetensors"

            save_file({"embedding.weight": base}, str(file_a))
            save_file({"embedding.weight": pruned}, str(file_b))

            result = diff_models(str(file_a), str(file_b))

            t = next(t for t in result.tensor_results if t.name == "embedding.weight")

            assert t.status == "PARTIAL_MATCH"
            assert t.metrics.cosine_similarity == pytest.approx(1.0, abs=1e-6)

            assert t.partial_info.extra_count == 32
            assert t.partial_info.extra_source == "model_a"

    def test_partial_match_1d_vectors(self):
        """Embedding biases (128,) vs (160,) - 1D bias vectors."""
        np.random.seed(42)
        base = np.random.randn(128).astype(np.float32)
        extra = np.random.randn(32).astype(np.float32)
        expanded = np.concatenate([base, extra])

        import tempfile
        from pathlib import Path

        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            file_a = tmpdir / "base.safetensors"
            file_b = tmpdir / "expanded.safetensors"

            save_file({"bias": base}, str(file_a))
            save_file({"bias": expanded}, str(file_b))

            result = diff_models(str(file_a), str(file_b))

            t = next(t for t in result.tensor_results if t.name == "bias")

            assert t.status == "PARTIAL_MATCH"
            assert t.metrics.cosine_similarity == pytest.approx(1.0, abs=1e-6)
            assert t.partial_info.common_shape == (128,)
            assert t.partial_info.differing_axis == 0
            assert t.partial_info.extra_count == 32
            assert t.partial_info.extra_source == "model_b"

    def test_multi_axis_mismatch_falls_back_to_shape_mismatch(self):
        """(128, 64) vs (160, 128) - variance on both axes should be SHAPE_MISMATCH."""
        np.random.seed(42)
        base = np.random.randn(128, 64).astype(np.float32)
        expanded = np.random.randn(160, 128).astype(np.float32)

        import tempfile
        from pathlib import Path

        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            file_a = tmpdir / "a.safetensors"
            file_b = tmpdir / "b.safetensors"

            save_file({"weight": base}, str(file_a))
            save_file({"weight": expanded}, str(file_b))

            result = diff_models(str(file_a), str(file_b))

            t = next(t for t in result.tensor_results if t.name == "weight")

            assert t.status == "SHAPE_MISMATCH"
            assert t.partial_info is None
            assert t.metrics is None

    def test_different_ndim_falls_back_to_shape_mismatch(self):
        """1D (128,) vs 2D (128, 64) - different ndim should be SHAPE_MISMATCH."""
        np.random.seed(42)
        base_1d = np.random.randn(128).astype(np.float32)
        base_2d = np.random.randn(128, 64).astype(np.float32)

        import tempfile
        from pathlib import Path

        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            file_a = tmpdir / "a.safetensors"
            file_b = tmpdir / "b.safetensors"

            save_file({"weight": base_1d}, str(file_a))
            save_file({"weight": base_2d}, str(file_b))

            result = diff_models(str(file_a), str(file_b))

            t = next(t for t in result.tensor_results if t.name == "weight")

            assert t.status == "SHAPE_MISMATCH"
            assert t.partial_info is None

    def test_partial_match_with_drift_on_shared_tokens(self):
        """Base (128, 64) vs Expanded (160, 64) with drift on shared tokens."""
        np.random.seed(42)
        base = np.random.randn(128, 64).astype(np.float32)
        # Add small drift to shared portion
        drift = np.random.randn(128, 64).astype(np.float32) * 0.01
        shared_with_drift = base + drift
        extra = np.random.randn(32, 64).astype(np.float32)
        expanded = np.vstack([shared_with_drift, extra])

        import tempfile
        from pathlib import Path

        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            file_a = tmpdir / "base.safetensors"
            file_b = tmpdir / "expanded.safetensors"

            save_file({"embedding.weight": base}, str(file_a))
            save_file({"embedding.weight": expanded}, str(file_b))

            result = diff_models(str(file_a), str(file_b))

            t = next(t for t in result.tensor_results if t.name == "embedding.weight")

            assert t.status == "PARTIAL_MATCH"
            # Should detect drift on shared tokens
            assert t.metrics.relative_frobenius > 0.0
            assert t.metrics.cosine_similarity < 1.0
            assert t.metrics.cosine_similarity > 0.9  # Small drift

            assert t.partial_info.extra_count == 32
            assert t.partial_info.extra_source == "model_b"

    def test_partial_match_with_deep_analysis(self):
        """Partial match with --deep should include spectral and taxonomy."""
        np.random.seed(42)
        base = np.random.randn(128, 64).astype(np.float32)
        extra = np.random.randn(32, 64).astype(np.float32)
        expanded = np.vstack([base, extra])

        import tempfile
        from pathlib import Path

        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            file_a = tmpdir / "base.safetensors"
            file_b = tmpdir / "expanded.safetensors"

            save_file({"embedding.weight": base}, str(file_a))
            save_file({"embedding.weight": expanded}, str(file_b))

            result = diff_models(str(file_a), str(file_b), deep=True)

            t = next(t for t in result.tensor_results if t.name == "embedding.weight")

            assert t.status == "PARTIAL_MATCH"
            assert t.spectral is not None
            assert t.taxonomy is not None
            assert t.partial_info is not None

    def test_partial_match_summary_counts(self):
        """DiffSummary should include partial_match_count."""
        np.random.seed(42)
        base = np.random.randn(128, 64).astype(np.float32)
        expanded = np.vstack([base, np.random.randn(32, 64).astype(np.float32)])

        import tempfile
        from pathlib import Path

        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            file_a = tmpdir / "base.safetensors"
            file_b = tmpdir / "expanded.safetensors"

            save_file({"embedding.weight": base}, str(file_a))
            save_file({"embedding.weight": expanded}, str(file_b))

            result = diff_models(str(file_a), str(file_b))

            assert result.summary.partial_match_count == 1
            assert result.summary.matched_count == 0
            assert result.summary.mismatched_shape_count == 0

    def test_cli_json_serializes_partial_info(self):
        """Verify --json includes partial_info dictionary."""
        import json
        import subprocess

        np.random.seed(42)
        base = np.random.randn(128, 64).astype(np.float32)
        expanded = np.vstack([base, np.random.randn(32, 64).astype(np.float32)])

        import tempfile
        from pathlib import Path

        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            file_a = tmpdir / "base.safetensors"
            file_b = tmpdir / "expanded.safetensors"

            save_file({"embedding.weight": base}, str(file_a))
            save_file({"embedding.weight": expanded}, str(file_b))

            import sys

            result = subprocess.run(
                [sys.executable, "-m", "wdiff.cli", str(file_a), str(file_b), "--json"],
                check=False,
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).resolve().parents[2]),
            )
            assert result.returncode == 0

            data = json.loads(result.stdout)

            assert data["summary"]["partial_match_count"] == 1

            tensor = next(t for t in data["tensors"] if t["name"] == "embedding.weight")
            assert tensor["status"] == "PARTIAL_MATCH"
            assert tensor["partial_info"] is not None
            assert tensor["partial_info"]["extra_count"] == 32
            assert tensor["partial_info"]["extra_source"] == "model_b"
            assert "common_shape" in tensor["partial_info"]
            assert "differing_axis" in tensor["partial_info"]

    def test_html_report_renders_partial_match(self):
        """Verify generate_html_report includes partial match status and counts."""
        from wdiff.display.report import generate_html_report

        np.random.seed(42)
        base = np.random.randn(128, 64).astype(np.float32)
        expanded = np.vstack([base, np.random.randn(32, 64).astype(np.float32)])

        import tempfile
        from pathlib import Path

        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            file_a = tmpdir / "base.safetensors"
            file_b = tmpdir / "expanded.safetensors"
            report_path = tmpdir / "report.html"

            save_file({"embedding.weight": base}, str(file_a))
            save_file({"embedding.weight": expanded}, str(file_b))

            from wdiff.engine.diff import diff_models

            result = diff_models(str(file_a), str(file_b))

            generate_html_report(result, str(report_path))

            html = report_path.read_text()

            # Check for partial match count in KPI
            assert "Partial Matches" in html
            assert "PARTIAL_MATCH" in html
            assert "status-partial_match" in html

            # Check that the partial match appears in the table data
            assert "embedding.weight" in html
            assert "128" in html  # common size
            assert "160" in html  # expanded size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
