from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from wdiff.engine.diff import (
    DiffResult,
    diff_models,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestDiffModels:
    def test_identical_models_safetensors(self):
        res = diff_models(
            str(FIXTURES_DIR / "model_a.safetensors"),
            str(FIXTURES_DIR / "model_a.safetensors"),
        )
        assert isinstance(res, DiffResult)
        assert res.summary.matched_count == 9
        assert res.summary.mismatched_shape_count == 0
        assert res.summary.only_in_a_count == 0
        assert res.summary.only_in_b_count == 0
        assert res.summary.mean_frobenius == pytest.approx(0.0, abs=1e-6)
        assert res.summary.min_cosine_value == pytest.approx(1.0, abs=1e-6)
        assert res.duration_seconds > 0.0
        # Peak process RSS telemetry (ru_maxrss): 0.0 only where resource is unavailable
        if sys.platform == "win32":
            assert res.peak_ram_mb == 0.0
        else:
            assert res.peak_ram_mb > 0.0

    def test_cross_format_identical_safetensors_and_gguf(self):
        # Adaptive RoPE parity automatically detects un-permuted vs permuted weights
        res = diff_models(
            str(FIXTURES_DIR / "model_a.safetensors"),
            str(FIXTURES_DIR / "model_a.gguf"),
        )
        assert res.format_a == "safetensors"
        assert res.format_b == "gguf"
        assert res.summary.matched_count == 9
        assert res.summary.mean_frobenius == pytest.approx(0.0, abs=1e-5)
        assert res.summary.min_cosine_value == pytest.approx(1.0, abs=1e-5)

    def test_perturbed_models(self):
        res = diff_models(
            str(FIXTURES_DIR / "model_a.safetensors"),
            str(FIXTURES_DIR / "model_b.safetensors"),
        )
        assert res.summary.matched_count == 9
        assert res.summary.mean_frobenius > 0.0
        assert res.summary.max_frobenius_tensor is not None
        assert res.summary.max_frobenius_value > 0.0

    def test_filter_regex(self):
        res = diff_models(
            str(FIXTURES_DIR / "model_a.safetensors"),
            str(FIXTURES_DIR / "model_b.safetensors"),
            tensor_filter="attn_output",
        )
        assert len(res.tensor_results) == 1
        assert res.tensor_results[0].name == "layer.0.attn_output.weight"
        assert res.summary.matched_count == 1

    def test_missing_tensors_in_b(self, tmp_path):
        from safetensors.numpy import save_file

        file_a = str(tmp_path / "a.safetensors")
        file_b = str(tmp_path / "b.safetensors")

        save_file({"shared": np.zeros((10, 10)), "only_a": np.zeros((10, 10))}, file_a)
        save_file({"shared": np.zeros((10, 10)), "only_b": np.zeros((10, 10))}, file_b)

        res = diff_models(file_a, file_b)
        assert res.summary.matched_count == 1
        assert res.summary.only_in_a_count == 1
        assert res.summary.only_in_b_count == 1

        statuses = {t.name: t.status for t in res.tensor_results}
        assert statuses["shared"] == "MATCH"
        assert statuses["only_a"] == "ONLY_IN_A"
        assert statuses["only_b"] == "ONLY_IN_B"

    def test_shape_mismatch_handling(self, tmp_path):
        from safetensors.numpy import save_file

        file_a = str(tmp_path / "a.safetensors")
        file_b = str(tmp_path / "b.safetensors")

        save_file({"layer": np.zeros((10, 10))}, file_a)
        save_file({"layer": np.zeros((20, 20))}, file_b)

        res = diff_models(file_a, file_b)
        assert res.summary.mismatched_shape_count == 1
        assert res.summary.matched_count == 0
        assert res.tensor_results[0].status == "SHAPE_MISMATCH"
        assert res.tensor_results[0].metrics is None

    def test_deep_diff_all_tensors_including_aliased_names(self):
        """Deep diff must succeed across all tensors even when canonical names differ from raw keys."""
        from wdiff.engine.taxonomy import TaxonomyBadge

        res = diff_models(
            str(FIXTURES_DIR / "model_a.safetensors"),
            str(FIXTURES_DIR / "model_b.safetensors"),
            deep=True,
        )
        assert res.summary.matched_count == 9
        for t in res.tensor_results:
            if t.status == "MATCH":
                assert t.spectral is not None
                assert t.taxonomy is not None
                assert t.taxonomy.badge == TaxonomyBadge.STOCHASTIC_NOISE

    def test_deep_diff_cross_format_unmodified(self):
        """Cross-format deep diff of identical models should classify all tensors as UNMODIFIED."""
        from wdiff.engine.taxonomy import TaxonomyBadge

        # Fixtures are both in HF format. Disable rope_parity for this test.
        res = diff_models(
            str(FIXTURES_DIR / "model_a.safetensors"),
            str(FIXTURES_DIR / "model_a.gguf"),
            deep=True,
            rope_parity=False,
        )
        assert res.summary.matched_count == 9
        for t in res.tensor_results:
            if t.status == "MATCH":
                assert t.spectral is not None
                assert t.taxonomy is not None
                assert t.taxonomy.badge == TaxonomyBadge.UNMODIFIED
