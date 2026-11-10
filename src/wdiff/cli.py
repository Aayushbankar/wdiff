from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wdiff import __version__
from wdiff.engine.diff import DiffResult, diff_models
from wdiff.engine.inspect import ModelInspectionResult, inspect_model
from wdiff.engine.lora import LoRADiffResult, diff_with_lora
from wdiff.engine.spectral import SpectralBundle, SubspaceGeometry
from wdiff.engine.taxonomy import TaxonomyResult


def _serialize_spectral_bundle(spectral: SpectralBundle | None) -> dict | None:
    if spectral is None:
        return None
    return {
        "singular_values": list(spectral.singular_values),
        "primary_direction_energy": spectral.primary_direction_energy,
        "stable_rank": spectral.stable_rank,
        "effective_rank": spectral.effective_rank,
        "signal_rank": spectral.signal_rank,
        "marcenko_pastur_edge": spectral.marcenko_pastur_edge,
        "signal_energy_ratio": spectral.signal_energy_ratio,
    }


def _serialize_subspace_geometry(subspace: SubspaceGeometry | None) -> dict | None:
    if subspace is None:
        return None
    return {
        "principal_angles_deg": list(subspace.principal_angles_deg),
        "mean_cosine": subspace.mean_cosine,
        "chordal_distance": subspace.chordal_distance,
        "geodesic_distance": subspace.geodesic_distance,
        "max_principal_angle": subspace.max_principal_angle,
        "uhlmann_fidelity": subspace.uhlmann_fidelity,
    }


def _serialize_taxonomy_result(taxonomy: TaxonomyResult | None) -> dict | None:
    if taxonomy is None:
        return None
    return {
        "badge": taxonomy.badge.value,
        "confidence": taxonomy.confidence,
        "explanation": taxonomy.explanation,
        "evidence": taxonomy.evidence,
    }


def _serialize_lora_diff_result(result: LoRADiffResult) -> dict:
    return {
        "summary": result.summary,
        "tensors": [
            {
                "name": t.name,
                "shape_a": list(t.shape_a),
                "shape_b": list(t.shape_b),
                "status": t.status,
                "metrics": (
                    {
                        "relative_frobenius": t.metrics.relative_frobenius,
                        "cosine_similarity": t.metrics.cosine_similarity,
                        "max_element_delta": t.metrics.max_element_delta,
                        "mean_element_delta": t.metrics.mean_element_delta,
                    }
                    if t.metrics
                    else None
                ),
                "spectral": _serialize_spectral_bundle(t.spectral)
                if hasattr(t, "spectral") and t.spectral
                else None,
                "taxonomy": _serialize_taxonomy_result(t.taxonomy)
                if hasattr(t, "taxonomy") and t.taxonomy
                else None,
            }
            for t in result.tensor_results
        ],
    }


def _serialize_inspection_result(result: ModelInspectionResult) -> dict:
    return {
        "model_path": result.model_path,
        "format": result.format,
        "summary": {
            "total_params": result.summary.total_params,
            "trainable_params_mb": result.summary.trainable_params_mb,
            "architecture": result.summary.architecture,
            "tensor_count": result.summary.tensor_count,
            "dead_neurons_total": result.summary.dead_neurons_total,
            "outlier_risk_layer_count": result.summary.outlier_risk_layer_count,
            "quantization_readiness": result.summary.quantization_readiness,
            "duration_seconds": result.summary.duration_seconds,
        },
        "layers": [
            {
                "name": l.name,
                "shape": list(l.shape),
                "dtype": l.dtype,
                "param_count": l.param_count,
                "norm": l.norm,
                "mean": l.mean,
                "std": l.std,
                "dead_neuron_count": l.dead_neuron_count,
                "outlier_score": l.outlier_score,
                "is_outlier_risk": l.is_outlier_risk,
                "spectral": l.spectral is not None,
            }
            for l in result.layers
        ],
    }


class _LoRAResultWrapper:
    """Adapts a LoRADiffResult to conform to DiffResult protocol for rendering."""

    def __init__(self, lora_result: LoRADiffResult, lora_path: str = "") -> None:
        self.model_a_path = "base model"
        self.model_b_path = f"base + LoRA ({Path(lora_path).name})" if lora_path else "base + LoRA"
        self.format_a = "safetensors"
        self.format_b = "safetensors"
        self.tensor_results = lora_result.tensor_results
        self.duration_seconds = lora_result.summary.get("duration_seconds", 0.0)

        class Summary:
            def __init__(self, s: dict) -> None:
                self.matched_count = s.get("matched_count", 0)
                self.partial_match_count = s.get("partial_match_count", 0)
                self.mismatched_shape_count = s.get("mismatched_shape_count", 0)
                self.only_in_a_count = s.get("only_in_a_count", 0)
                self.only_in_b_count = s.get("only_in_b_count", 0)
                self.lens_analyzed_count = s.get("lens_analyzed_count", 0)
                self.max_frobenius_tensor = s.get("max_frobenius_tensor")
                self.max_frobenius_value = s.get("max_frobenius_value", 0.0)
                self.min_cosine_value = s.get("min_cosine_value", 1.0)
                self.mean_frobenius = s.get("mean_frobenius", 0.0)

        self.summary = Summary(lora_result.summary)
        self.duration_seconds = lora_result.summary.get("duration_seconds", 0.0)


def _serialize_diff_result(result: DiffResult) -> dict:
    return {
        "model_a": result.model_a_path,
        "model_b": result.model_b_path,
        "format_a": result.format_a,
        "format_b": result.format_b,
        "duration_seconds": result.duration_seconds,
        "peak_ram_mb": result.peak_ram_mb,
        "summary": {
            "matched_count": result.summary.matched_count,
            "partial_match_count": result.summary.partial_match_count,
            "mismatched_shape_count": result.summary.mismatched_shape_count,
            "only_in_a_count": result.summary.only_in_a_count,
            "only_in_b_count": result.summary.only_in_b_count,
            "lens_analyzed_count": result.summary.lens_analyzed_count,
            "mean_frobenius": result.summary.mean_frobenius,
            "max_frobenius": result.summary.max_frobenius_value,
            "max_frobenius_tensor": result.summary.max_frobenius_tensor,
            "min_cosine": result.summary.min_cosine_value,
        },
        "tensors": [
            {
                "name": t.name,
                "shape_a": list(t.shape_a),
                "shape_b": list(t.shape_b),
                "status": t.status,
                "metrics": (
                    {
                        "relative_frobenius": t.metrics.relative_frobenius,
                        "cosine_similarity": t.metrics.cosine_similarity,
                        "max_element_delta": t.metrics.max_element_delta,
                        "mean_element_delta": t.metrics.mean_element_delta,
                    }
                    if t.metrics
                    else None
                ),
                "spectral": _serialize_spectral_bundle(t.spectral),
                "subspace": _serialize_subspace_geometry(
                    t.subspace if hasattr(t, "subspace") else None
                ),
                "taxonomy": _serialize_taxonomy_result(t.taxonomy),
                "partial_info": (
                    {
                        "common_shape": list(t.partial_info.common_shape),
                        "differing_axis": t.partial_info.differing_axis,
                        "extra_count": t.partial_info.extra_count,
                        "extra_source": t.partial_info.extra_source,
                        "extra_norm": t.partial_info.extra_norm,
                        "extra_mean": t.partial_info.extra_mean,
                        "extra_std": t.partial_info.extra_std,
                    }
                    if t.partial_info
                    else None
                ),
                "lens": (
                    {
                        "layer_name": t.lens.layer_name,
                        "canonical_name": t.lens.canonical_name,
                        "positive_tokens": [
                            {"token_id": tc.token_id, "token": tc.token, "logit": tc.logit}
                            for tc in t.lens.positive_tokens
                        ],
                        "negative_tokens": [
                            {"token_id": tc.token_id, "token": tc.token, "logit": tc.logit}
                            for tc in t.lens.negative_tokens
                        ],
                        "singular_value": t.lens.singular_value,
                        "energy_ratio": t.lens.energy_ratio,
                    }
                    if t.lens
                    else None
                ),
            }
            for t in result.tensor_results
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wdiff",
        description="git diff, but for neural network weights. Ultra-fast, memory-bounded LLM differential analyzer.",
    )
    parser.add_argument(
        "model_a", nargs="?", help="Path to first model checkpoint (.safetensors or .gguf)"
    )
    parser.add_argument(
        "model_b", nargs="?", help="Path to second model checkpoint (.safetensors or .gguf)"
    )
    parser.add_argument(
        "-f", "--filter", dest="tensor_filter", help="Regex filter for tensor names"
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Enable deep spectral analysis and taxonomy classification",
    )
    parser.add_argument(
        "--lora", metavar="PATH", help="Path to LoRA adapter (.safetensors) to apply to model_a"
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="Scaling factor for LoRA adapter (default: 1.0)",
    )
    parser.add_argument(
        "--report", metavar="PATH", help="Generate standalone interactive HTML report at PATH"
    )
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout")
    parser.add_argument(
        "--no-rope-parity",
        action="store_true",
        help="Disable RoPE coordinate normalization for cross-format diffs (HF safetensors vs GGUF)",
    )
    parser.add_argument(
        "--lens",
        action="store_true",
        help="Enable semantic concept translation via logit lens on down-projection layers",
    )
    parser.add_argument(
        "--tokenizer",
        metavar="PATH",
        help="Path to tokenizer.json, vocab.json, or model directory for vocabulary lookup",
    )
    parser.add_argument(
        "--lens-top-k",
        type=int,
        default=5,
        help="Number of positive/negative concept tokens per layer (default: 5)",
    )
    parser.add_argument("-v", "--version", action="version", version=f"wdiff {__version__}")

    if argv is None:
        argv = sys.argv[1:]

    # Support 'inspect' as an optional explicit command prefix: 'wdiff inspect <model> [options]'
    is_explicit_inspect = bool(argv and argv[0] == "inspect")
    if is_explicit_inspect:
        argv = argv[1:]

    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0

    if not args.model_a:
        parser.print_help()
        return 0

    # Auto-route to inspect if:
    # 1. Explicit 'inspect' prefix was used, OR
    # 2. Only model_a was provided and --lora was not provided
    should_inspect = is_explicit_inspect or (args.model_b is None and not args.lora)

    if should_inspect:
        model_path = Path(args.model_a)
        if not model_path.exists():
            print(f"Error: File not found: {model_path}", file=sys.stderr)
            return 1

        try:
            result = inspect_model(
                str(model_path), deep=args.deep, tensor_filter=args.tensor_filter
            )
        except Exception as e:  # noqa: BLE001
            print(f"Error during inspection: {e}", file=sys.stderr)
            return 1

        if args.report:
            from wdiff.display.report import generate_html_report

            report_path = Path(args.report)
            generate_html_report(result, report_path)
            size_mb = report_path.stat().st_size / (1024 * 1024)
            print(f"Report saved to {report_path} ({size_mb:.2f} MB)")

        if args.json:
            json_output = _serialize_inspection_result(result)
            print(json.dumps(json_output, indent=2))
        else:
            from wdiff.display.terminal import render_inspection

            render_inspection(result)

        return 0

    # Not inspect command - validate diff/lora arguments
    if args.lora and args.model_b:
        print(
            "Error: Cannot use --lora with model_b. Use either --lora or model_b.", file=sys.stderr
        )
        return 1
    if not args.lora and not args.model_b:
        parser.error("Either model_b or --lora must be provided")

    path_a = Path(args.model_a)

    if not path_a.exists():
        print(f"Error: File not found: {path_a}", file=sys.stderr)
        return 1

    try:
        if args.lora:
            lora_path = Path(args.lora)
            if not lora_path.exists():
                print(f"Error: LoRA adapter not found: {lora_path}", file=sys.stderr)
                return 1

            result = diff_with_lora(
                str(path_a),
                str(lora_path),
                lora_scale=args.lora_scale,
            )
        else:
            path_b = Path(args.model_b)
            if not path_b.exists():
                print(f"Error: File not found: {path_b}", file=sys.stderr)
                return 1

            result = diff_models(
                str(path_a),
                str(path_b),
                tensor_filter=args.tensor_filter,
                deep=args.deep,
                rope_parity=not args.no_rope_parity,
                lens=args.lens,
                tokenizer=args.tokenizer,
                lens_top_k=args.lens_top_k,
            )
    except Exception as e:  # noqa: BLE001
        print(f"Error during diff: {e}", file=sys.stderr)
        return 1

    # Adapt LoRA result if needed
    rendered_result = (
        _LoRAResultWrapper(result, lora_path=getattr(args, "lora", "") or "")
        if isinstance(result, LoRADiffResult)
        else result
    )

    if hasattr(args, "report") and args.report:
        from wdiff.display.report import generate_html_report

        report_path = Path(args.report)
        generate_html_report(rendered_result, report_path)
        size_mb = report_path.stat().st_size / (1024 * 1024)
        print(f"Report saved to {report_path} ({size_mb:.2f} MB)")

    if args.json:
        if isinstance(result, LoRADiffResult):
            json_output = _serialize_lora_diff_result(result)
        elif hasattr(result, "tensor_results") and hasattr(result, "summary"):
            json_output = _serialize_diff_result(result)
        else:
            json_output = {}
        print(json.dumps(json_output, indent=2))
    else:
        from wdiff.display.terminal import render_diff

        render_diff(rendered_result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
