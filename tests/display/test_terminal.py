from __future__ import annotations

from pathlib import Path

import pytest
from rich.panel import Panel
from rich.table import Table

from wdiff.display.terminal import (
    detect_anomalies,
    render_diff_table,
    render_summary_panel,
)
from wdiff.engine.diff import DiffResult, DiffSummary, TensorDiffResult
from wdiff.engine.metrics import MetricsBundle


@pytest.fixture
def sample_diff_result() -> DiffResult:
    summary = DiffSummary()
    mb1 = MetricsBundle(
        relative_frobenius=0.0001,
        cosine_similarity=0.9999,
        max_element_delta=0.001,
        mean_element_delta=0.0001,
    )
    mb2 = MetricsBundle(
        relative_frobenius=0.1542,
        cosine_similarity=0.9123,
        max_element_delta=0.4521,
        mean_element_delta=0.0891,
    )
    mb_ortho = MetricsBundle(
        relative_frobenius=0.8921,
        cosine_similarity=0.0123,
        max_element_delta=1.2341,
        mean_element_delta=0.4123,
    )

    t1 = TensorDiffResult(
        name="layer.0.attn_q.weight",
        shape_a=(64, 64),
        shape_b=(64, 64),
        metrics=mb1,
        status="MATCH",
    )
    t2 = TensorDiffResult(
        name="layer.10.attn_output.weight",
        shape_a=(64, 64),
        shape_b=(64, 64),
        metrics=mb2,
        status="MATCH",
    )
    t3 = TensorDiffResult(
        name="layer.14.ffn_down.weight",
        shape_a=(64, 64),
        shape_b=(64, 64),
        metrics=mb_ortho,
        status="MATCH",
    )
    t4 = TensorDiffResult(
        name="layer.mismatched.weight",
        shape_a=(64, 64),
        shape_b=(128, 128),
        metrics=None,
        status="SHAPE_MISMATCH",
    )
    t5 = TensorDiffResult(
        name="layer.only_a.weight",
        shape_a=(64, 64),
        shape_b=(),
        metrics=None,
        status="ONLY_IN_A",
    )

    summary.add_match(t1.name, mb1)
    summary.add_match(t2.name, mb2)
    summary.add_match(t3.name, mb_ortho)
    summary.mismatched_shape_count = 1
    summary.only_in_a_count = 1

    return DiffResult(
        model_a_path="/models/llama3-base.safetensors",
        model_b_path="/models/llama3-instruct.gguf",
        format_a="safetensors",
        format_b="gguf",
        tensor_results=[t1, t2, t3, t4, t5],
        summary=summary,
        duration_seconds=1.45,
    )


class TestTerminalDisplay:
    def test_render_diff_table_returns_rich_table(self, sample_diff_result):
        table = render_diff_table(sample_diff_result)
        assert isinstance(table, Table)
        columns = [col.header for col in table.columns]
        assert "Layer" in columns
        assert "Status" in columns
        assert columns.count("Status") == 1
        assert "Rel. ΔF" in columns
        assert "Cosine Sim" in columns
        assert len(table.rows) == 5

    def test_render_inspection_table(self):
        from wdiff.display.terminal import render_inspection_table
        from wdiff.engine.inspect import inspect_model

        fixtures = Path(__file__).parent.parent / "fixtures"
        model_a = str(fixtures / "model_a.safetensors")
        result = inspect_model(model_a)

        table = render_inspection_table(result)
        assert isinstance(table, Table)
        columns = [col.header for col in table.columns]
        assert "Layer" in columns
        assert "Shape" in columns
        assert "Status" in columns
        assert columns.count("Status") == 1
        assert len(table.rows) == result.summary.tensor_count

    def test_render_summary_panel(self, sample_diff_result):
        panel = render_summary_panel(sample_diff_result)
        assert isinstance(panel, Panel)
        # Panel should contain key summary metrics
        panel_str = str(panel.renderable)
        assert "Matched: 3" in panel_str or "3" in panel_str
        assert "safetensors" in panel_str
        assert "gguf" in panel_str

    def test_detect_anomalies_flags_high_deltas_and_orthogonality(self, sample_diff_result):
        anomalies = detect_anomalies(sample_diff_result)
        assert isinstance(anomalies, list)
        assert len(anomalies) >= 2

        anomaly_text = " ".join(anomalies)
        # Should flag near-orthogonal shift
        assert "orthogonal" in anomaly_text.lower() or "cosine" in anomaly_text.lower()
        # Should flag shape mismatch
        assert "shape" in anomaly_text.lower()
