from __future__ import annotations

import re
from pathlib import Path

from wdiff.cli import main
from wdiff.engine.diff import diff_models


class TestHTMLReportGeneration:
    def test_generate_html_report_creates_file(self, tmp_path):
        from wdiff.display.report import generate_html_report

        fixtures = Path(__file__).parent.parent / "fixtures"
        model_a = str(fixtures / "model_a.safetensors")
        model_b = str(fixtures / "model_b.safetensors")

        result = diff_models(model_a, model_b)
        report_path = tmp_path / "test_report.html"

        output_path = generate_html_report(result, report_path)

        assert output_path == report_path
        assert report_path.exists()
        assert report_path.stat().st_size > 1000

    def test_html_report_is_completely_self_contained(self, tmp_path):
        from wdiff.display.report import generate_html_report

        fixtures = Path(__file__).parent.parent / "fixtures"
        model_a = str(fixtures / "model_a.safetensors")
        model_b = str(fixtures / "model_b.safetensors")

        result = diff_models(model_a, model_b)
        report_path = tmp_path / "test_report.html"
        generate_html_report(result, report_path)

        html_content = report_path.read_text()

        # Check for external script src URLs
        script_srcs = re.findall(r'<script\s+src=["\']([^"\']+)["\']', html_content)
        for src in script_srcs:
            assert not src.startswith("http://"), f"External script found: {src}"
            assert not src.startswith("https://"), f"External script found: {src}"

        # Check for external link href URLs
        link_hrefs = re.findall(r'<link\s+href=["\']([^"\']+)["\']', html_content)
        for href in link_hrefs:
            assert not href.startswith("http://"), f"External stylesheet found: {href}"
            assert not href.startswith("https://"), f"External stylesheet found: {href}"

        # Check for external font imports
        assert "fonts.googleapis.com" not in html_content
        assert "fonts.gstatic.com" not in html_content

    def test_html_report_contains_metadata_and_summary(self, tmp_path):
        from wdiff.display.report import generate_html_report

        fixtures = Path(__file__).parent.parent / "fixtures"
        model_a = str(fixtures / "model_a.safetensors")
        model_b = str(fixtures / "model_b.safetensors")

        result = diff_models(model_a, model_b)
        report_path = tmp_path / "test_report.html"
        generate_html_report(result, report_path)

        html_content = report_path.read_text()

        # Check for model paths
        assert "model_a.safetensors" in html_content
        assert "model_b.safetensors" in html_content
        # Check for summary metrics
        assert "Mean Rel. ΔF" in html_content
        assert "Min Cosine" in html_content
        assert "Duration" in html_content

    def test_html_report_contains_all_layer_names(self, tmp_path):
        from wdiff.display.report import generate_html_report

        fixtures = Path(__file__).parent.parent / "fixtures"
        model_a = str(fixtures / "model_a.safetensors")
        model_b = str(fixtures / "model_b.safetensors")

        result = diff_models(model_a, model_b)
        report_path = tmp_path / "test_report.html"
        generate_html_report(result, report_path)

        html_content = report_path.read_text()

        # Check for some layer names in the table
        assert "embedding.weight" in html_content
        assert "layer.0.attn_q.weight" in html_content
        assert "layer.0.attn_output.weight" in html_content

    def test_html_report_includes_taxonomy_and_spectral_when_deep(self, tmp_path):
        from wdiff.display.report import generate_html_report

        fixtures = Path(__file__).parent.parent / "fixtures"
        model_a = str(fixtures / "model_a.safetensors")
        model_b = str(fixtures / "model_b.safetensors")

        result = diff_models(model_a, model_b, deep=True)
        report_path = tmp_path / "test_report_deep.html"
        generate_html_report(result, report_path)

        html_content = report_path.read_text()

        # Check for taxonomy badge names
        assert "Stochastic SGD Noise" in html_content or "Unmodified" in html_content
        # Check for spectral data
        assert "signal_rank" in html_content.lower() or "Signal Rank" in html_content
        assert "effective_rank" in html_content.lower() or "Effective Rank" in html_content
        # Check for subspace geometry columns (Grassmannian theta-max + Uhlmann fidelity)
        assert "θmax°" in html_content
        assert "uhlmann_fidelity" in html_content
        assert "Fidelity" in html_content


class TestCLIReportFlag:
    def test_cli_report_flag_generates_file(self, tmp_path, capsys):
        from pathlib import Path

        fixtures = Path(__file__).parent.parent / "fixtures"
        model_a = str(fixtures / "model_a.safetensors")
        model_b = str(fixtures / "model_b.safetensors")
        report_path = tmp_path / "cli_report.html"

        code = main([model_a, model_b, "--report", str(report_path)])
        assert code == 0
        assert report_path.exists()
        assert report_path.stat().st_size > 1000

        out, _ = capsys.readouterr()
        assert "Report saved" in out or "report" in out.lower()


class TestHTMLReportSecurity:
    def test_report_title_xss_sanitization(self, tmp_path):
        from wdiff.display.report import generate_html_report

        fixtures = Path(__file__).parent.parent / "fixtures"
        model_a = str(fixtures / "model_a.safetensors")
        model_b = str(fixtures / "model_b.safetensors")

        result = diff_models(model_a, model_b)
        report_path = tmp_path / "xss_report.html"

        malicious_title = '<script>alert("xss")</script>'
        generate_html_report(result, report_path, title=malicious_title)

        content = report_path.read_text()
        assert "<title>&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;</title>" in content
        assert "<h1>&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;</h1>" in content
        assert "<h1><script>" not in content
        assert "function escapeHtml(str)" in content

    def test_adversarial_tokens_and_layer_names_escaped(self, tmp_path):
        from wdiff.display.report import generate_html_report
        from wdiff.engine.diff import DiffResult, DiffSummary, TensorDiffResult
        from wdiff.engine.lens import LayerLensResult, TokenConcept

        adversarial_result = DiffResult(
            model_a_path="<script>alert(1)</script>",
            model_b_path="normal_b",
            format_a="safetensors",
            format_b="safetensors",
            tensor_results=[
                TensorDiffResult(
                    name="<img src=x onerror=alert(1)>",
                    shape_a=(10, 10),
                    shape_b=(10, 10),
                    status="MATCH",
                    metrics=None,
                    spectral=None,
                    taxonomy=None,
                    partial_info=None,
                    lens=LayerLensResult(
                        layer_name="layer.0",
                        canonical_name="layer.0",
                        positive_tokens=[
                            TokenConcept(token_id=1, token="<|im_start|>", logit=5.0),
                            TokenConcept(
                                token_id=2, token="<script>alert('token')</script>", logit=4.0
                            ),
                        ],
                        negative_tokens=[
                            TokenConcept(token_id=3, token="<unk>", logit=-3.0),
                        ],
                        singular_value=1.0,
                        energy_ratio=0.9,
                    ),
                )
            ],
            summary=DiffSummary(
                matched_count=1,
                partial_match_count=0,
                mismatched_shape_count=0,
                only_in_a_count=0,
                only_in_b_count=0,
                lens_analyzed_count=1,
                mean_frobenius=0.0,
                max_frobenius_tensor=None,
                max_frobenius_value=0.0,
                min_cosine_value=1.0,
            ),
            duration_seconds=0.1,
        )

        report_path = tmp_path / "adversarial_tokens_report.html"
        generate_html_report(adversarial_result, report_path, title="<script>alert(1)</script>")
        content = report_path.read_text()

        assert "escapeHtml(row.name)" in content
        assert "escapeHtml(tc.token)" in content
        assert "function escapeHtml(str)" in content
        assert "<h1><script>alert(1)</script>" not in content
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in content
        assert "<|im_start|>" in content  # present in JSON data table

    def test_inspection_report_single_escaping(self, tmp_path):
        from wdiff.display.report import generate_html_report
        from wdiff.engine.inspect import InspectionSummary, ModelInspectionResult

        summary = InspectionSummary(
            total_params=100,
            trainable_params_mb=0.1,
            architecture="test",
            tensor_count=1,
            dead_neurons_total=0,
            outlier_risk_layer_count=0,
            quantization_readiness="READY",
            duration_seconds=1.0,
            mean_frobenius=0.0,
            min_cosine_value=1.0,
        )
        res = ModelInspectionResult(
            model_path="test.safetensors",
            format="safetensors",
            summary=summary,
            layers=[],
        )

        report_path = tmp_path / "inspection_xss.html"
        generate_html_report(res, report_path, title="<script>alert('inspect')</script>")
        content = report_path.read_text()

        assert "<title>&lt;script&gt;alert(&#x27;inspect&#x27;)&lt;/script&gt;</title>" in content
        assert "&amp;lt;script" not in content
        assert "<script>alert" not in content

    def test_script_breakout_in_table_data_is_escaped(self, tmp_path):
        """Inline JSON in <script> must not allow </script> breakout from data."""
        from wdiff.display.report import generate_html_report
        from wdiff.engine.diff import DiffResult, DiffSummary, TensorDiffResult
        from wdiff.engine.metrics import MetricsBundle

        results = DiffResult(
            model_a_path="model_a.safetensors",
            model_b_path="model_b.safetensors",
            format_a="safetensors",
            format_b="safetensors",
            tensor_results=[
                TensorDiffResult(
                    name='layer.0.</script><script>alert("xss")</script>.weight',
                    shape_a=(1, 1),
                    shape_b=(1, 1),
                    metrics=MetricsBundle(
                        relative_frobenius=0.0,
                        cosine_similarity=1.0,
                        max_element_delta=0.0,
                        mean_element_delta=0.0,
                    ),
                    status="MATCH",
                )
            ],
            summary=DiffSummary(
                matched_count=1,
                max_frobenius_value=0.0,
                mean_frobenius=0.0,
                min_cosine_value=1.0,
            ),
            duration_seconds=0.1,
            peak_ram_mb=42.0,
        )

        out = tmp_path / "breakout.html"
        generate_html_report(results, str(out))
        content = out.read_text(encoding="utf-8")

        # Raw closing script tag from data must not appear inside inline script payload
        assert '</script><script>alert("xss")</script>.weight' not in content
        assert "<\\/script>" in content
        assert "<script>alert" not in content
