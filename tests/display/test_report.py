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
