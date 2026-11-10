from __future__ import annotations

import json
from pathlib import Path

from wdiff.cli import main

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
MODEL_A_ST = str(FIXTURES_DIR / "model_a.safetensors")
MODEL_B_ST = str(FIXTURES_DIR / "model_b.safetensors")
MODEL_A_GGUF = str(FIXTURES_DIR / "model_a.gguf")


class TestCLI:
    def test_cli_basic_run(self, capsys):
        code = main([MODEL_A_ST, MODEL_B_ST])
        assert code == 0
        out, _ = capsys.readouterr()
        assert "model_a.safetensors" in out
        assert "model_b.safetensors" in out
        assert "embedding.weight" in out

    def test_cli_cross_format(self, capsys):
        code = main([MODEL_A_ST, MODEL_A_GGUF])
        assert code == 0
        out, _ = capsys.readouterr()
        assert "safetensors" in out
        assert "gguf" in out

    def test_cli_json_output(self, capsys):
        code = main([MODEL_A_ST, MODEL_B_ST, "--json"])
        assert code == 0
        out, _ = capsys.readouterr()

        data = json.loads(out)
        assert "model_a" in data
        assert "model_b" in data
        assert "summary" in data
        assert "tensors" in data
        assert "peak_ram_mb" in data
        assert data["peak_ram_mb"] >= 0.0

        summary = data["summary"]
        assert summary["matched_count"] == 9
        assert "mean_frobenius" in summary
        assert "min_cosine" in summary

        tensors = data["tensors"]
        assert len(tensors) == 9
        assert tensors[0]["name"] == "embedding.weight"
        assert "relative_frobenius" in tensors[0]["metrics"]
        assert "cosine_similarity" in tensors[0]["metrics"]

    def test_cli_filter(self, capsys):
        code = main([MODEL_A_ST, MODEL_B_ST, "--filter", "attn_output"])
        assert code == 0
        out, _ = capsys.readouterr()
        assert "layer.0.attn_output.weight" in out
        assert "embedding.weight" not in out

    def test_cli_missing_file_exits_with_error(self, capsys):
        code = main(["non_existent_a.safetensors", MODEL_B_ST])
        assert code != 0
        _, err = capsys.readouterr()
        assert "not found" in err.lower() or "error" in err.lower()

    def test_cli_version(self, capsys):
        code = main(["--version"])
        assert code == 0
        out, _ = capsys.readouterr()
        assert "wdiff" in out
        assert "0.3.0" in out

    def test_cli_lora_json(self, capsys):
        lora_st = str(FIXTURES_DIR / "demo_lora.safetensors")
        code = main([MODEL_A_ST, "--lora", lora_st, "--json"])
        assert code == 0
        out, _ = capsys.readouterr()

        data = json.loads(out)
        assert "summary" in data
        assert "tensors" in data
        assert data["summary"]["matched_count"] == 9
        assert any(t["name"] == "layer.0.attn_q.weight" and t["metrics"] for t in data["tensors"])

    def test_cli_lora_report(self, capsys, tmp_path):
        lora_st = str(FIXTURES_DIR / "demo_lora.safetensors")
        report_file = tmp_path / "lora_report.html"
        code = main([MODEL_A_ST, "--lora", lora_st, "--report", str(report_file)])
        assert code == 0
        assert report_file.exists()
        content = report_file.read_text()
        assert "layer.0.attn_q.weight" in content
        assert "wdiff Report" in content
