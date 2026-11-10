from __future__ import annotations

import numpy as np

from wdiff.engine.metrics import compute_all_metrics
from wdiff.engine.spectral import compute_spectral_bundle
from wdiff.engine.taxonomy import (
    TaxonomyBadge,
    TaxonomyResult,
    classify_weight_delta,
)


class TestTaxonomyClassification:
    def test_classify_unmodified_zero_delta(self):
        delta = np.zeros((128, 128), dtype=np.float32)
        metrics = compute_all_metrics(delta, delta)
        spectral = compute_spectral_bundle(delta)

        result = classify_weight_delta(delta, metrics, spectral)
        assert isinstance(result, TaxonomyResult)
        assert result.badge == TaxonomyBadge.UNMODIFIED

    def test_classify_rank1_safety_ablation(self):
        """Simulate Rank-1 abliteration: delta is pure rank-1 outer product."""
        np.random.seed(42)
        d = 256
        W = np.random.randn(d, d).astype(np.float32)
        r = np.random.randn(d, 1).astype(np.float32)
        r = r / np.linalg.norm(r)
        delta_W = r @ (r.T @ W)

        metrics = compute_all_metrics(W, W + delta_W)
        spectral = compute_spectral_bundle(delta_W)

        result = classify_weight_delta(delta_W, metrics, spectral)
        assert result.badge == TaxonomyBadge.RANK1_ABLATION
        assert "rank-1" in result.explanation.lower()
        assert result.confidence >= 0.90

    def test_classify_lora_adapter(self):
        """Simulate rank-16 LoRA adapter with background noise."""
        np.random.seed(42)
        m, n, rank = 256, 256, 16
        B = np.random.randn(m, rank).astype(np.float32) * 0.1
        A = np.random.randn(rank, n).astype(np.float32) * 0.1
        delta_W = B @ A + np.random.randn(m, n).astype(np.float32) * 0.002
        W = np.random.randn(m, n).astype(np.float32)

        metrics = compute_all_metrics(W, W + delta_W)
        spectral = compute_spectral_bundle(delta_W)

        result = classify_weight_delta(delta_W, metrics, spectral)
        assert result.badge == TaxonomyBadge.LOW_RANK_ADAPTER
        assert "lora" in result.explanation.lower() or "low-rank" in result.explanation.lower()

    def test_classify_sparse_pruning_merge(self):
        """Simulate a TIES / DARE merge with 80% sparsity."""
        np.random.seed(42)
        delta_W = np.random.randn(128, 128).astype(np.float32) * 0.1
        mask = np.random.rand(128, 128) > 0.80  # 80% zeros
        delta_W = delta_W * mask
        W = np.random.randn(128, 128).astype(np.float32)

        metrics = compute_all_metrics(W, W + delta_W)
        spectral = compute_spectral_bundle(delta_W)

        result = classify_weight_delta(delta_W, metrics, spectral)
        assert result.badge == TaxonomyBadge.SPARSE_UPDATE
        assert "sparse" in result.explanation.lower()

    def test_classify_stochastic_sgd_noise(self):
        """Simulate pure SGD noise where signal_rank == 0."""
        np.random.seed(42)
        delta_W = np.random.randn(128, 128).astype(np.float32) * 0.005
        W = np.random.randn(128, 128).astype(np.float32)

        metrics = compute_all_metrics(W, W + delta_W)
        spectral = compute_spectral_bundle(delta_W)

        result = classify_weight_delta(delta_W, metrics, spectral)
        assert result.badge == TaxonomyBadge.STOCHASTIC_NOISE


class TestDeepCLIIntegration:
    def test_cli_deep_flag_runs_without_error(self, capsys):
        from pathlib import Path

        from wdiff.cli import main

        fixtures = Path(__file__).parent.parent / "fixtures"
        model_a = str(fixtures / "model_a.safetensors")
        model_b = str(fixtures / "model_b.safetensors")

        code = main([model_a, model_b, "--deep", "--filter", "attn_output"])
        assert code == 0
        out, _ = capsys.readouterr()
        assert "Rank" in out or "Badge" in out or "Classification" in out or "Dominance" in out

    def test_cli_deep_json_includes_spectral_data(self, capsys):
        import json
        from pathlib import Path

        from wdiff.cli import main

        fixtures = Path(__file__).parent.parent / "fixtures"
        model_a = str(fixtures / "model_a.safetensors")
        model_b = str(fixtures / "model_b.safetensors")

        code = main([model_a, model_b, "--deep", "--json", "--filter", "attn_output"])
        assert code == 0
        out, _ = capsys.readouterr()

        data = json.loads(out)
        tensor = data["tensors"][0]
        assert "spectral" in tensor
        spectral = tensor["spectral"]
        assert "stable_rank" in spectral
        assert "primary_direction_energy" in spectral
        assert "taxonomy" in tensor
        assert "badge" in tensor["taxonomy"]
        assert "subspace" in tensor
        subspace = tensor["subspace"]
        assert subspace is not None
        assert "principal_angles_deg" in subspace
        assert "max_principal_angle" in subspace
        assert "uhlmann_fidelity" in subspace
        assert 0.0 <= subspace["uhlmann_fidelity"] <= 1.0
        assert len(subspace["principal_angles_deg"]) > 0
