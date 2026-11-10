from __future__ import annotations

import contextlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


@dataclass(slots=True)
class LoRALayer:
    """Single LoRA adapter layer with A and B matrices."""

    name: str
    lora_A: np.ndarray  # shape (rank, in_features)
    lora_B: np.ndarray  # shape (out_features, rank)
    alpha: float | None = None

    @property
    def rank(self) -> int:
        return self.lora_A.shape[0]

    @property
    def in_features(self) -> int:
        return self.lora_A.shape[1]

    @property
    def out_features(self) -> int:
        return self.lora_B.shape[0]

    def scale(self) -> float:
        """Compute LoRA scaling factor (alpha / rank if alpha provided, else 1.0)."""
        if self.alpha is not None:
            return self.alpha / self.rank
        return 1.0


@dataclass(slots=True)
class LoRAAdapter:
    """LoRA adapter containing multiple layers."""

    layers: dict[str, LoRALayer]
    base_model_prefix: str = "base_model.model."

    @property
    def rank(self) -> int:
        """Get the rank (assumes all layers have same rank)."""
        for layer in self.layers.values():
            return layer.rank
        return 0

    def get_layer_for_tensor(self, tensor_name: str) -> LoRALayer | None:
        """Find the LoRA layer that corresponds to a tensor name."""
        # Try exact match first
        if tensor_name in self.layers:
            return self.layers[tensor_name]
        # Try with .weight suffix removed
        if tensor_name.endswith(".weight"):
            base_name = tensor_name[:-7]  # remove '.weight'
            if base_name in self.layers:
                return self.layers[base_name]
        return None


def _parse_lora_key(key: str) -> tuple[str, str] | None:
    """
    Parse a LoRA key to extract layer name and matrix type.

    Expected formats:
    - base_model.model.layers.0.self_attn.q_proj.lora_A.weight
    - base_model.model.layers.0.self_attn.q_proj.lora_B.weight
    - layer.0.self_attn.q_proj.lora_A.weight (without base_model prefix)

    Returns:
        (layer_name, matrix_type) where matrix_type is 'lora_A' or 'lora_B'
    """
    # Remove base_model.model. prefix if present
    key = key.removeprefix("base_model.model.")

    # Pattern: layer_name.lora_A.weight or layer_name.lora_B.weight
    match = re.match(r"^(.+)\.lora_([AB])\.weight$", key)
    if not match:
        return None

    layer_name = match.group(1)
    matrix_type = match.group(2)  # 'A' or 'B'
    return layer_name, f"lora_{matrix_type}"


def _load_adapter_alpha(path: str) -> float | None:
    """
    Read `lora_alpha` from adapter_config.json beside the adapter file (PEFT layout).

    Returns None when the config is absent or does not carry a usable alpha,
    preserving the previous scale-1.0 fallback.
    """
    config_path = Path(path).parent / "adapter_config.json"
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(config, dict):
        return None
    alpha = config.get("lora_alpha")
    if alpha is None:
        return None
    try:
        return float(alpha)
    except (TypeError, ValueError):
        return None


def load_lora_adapter(path: str) -> LoRAAdapter:
    """
    Load LoRA adapter from safetensors file.

    Expected format: keys like "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
    A sibling adapter_config.json (PEFT) supplies lora_alpha for alpha/r scaling.
    """
    adapter = LoRAAdapter(layers={})
    alpha = _load_adapter_alpha(path)

    with safe_open(path, framework="numpy", device="cpu") as f:
        # Group tensors by layer
        layer_data: dict[str, dict] = {}

        for key in f.keys():  # noqa: SIM118
            parsed = _parse_lora_key(key)
            if parsed is None:
                continue
            layer_name, matrix_type = parsed

            if layer_name not in layer_data:
                layer_data[layer_name] = {}

            layer_data[layer_name][matrix_type] = f.get_tensor(key)

        # Build LoRALayer objects
        for layer_name, matrices in layer_data.items():
            if "lora_A" not in matrices or "lora_B" not in matrices:
                continue  # Skip incomplete layers

            lora_A = matrices["lora_A"]
            lora_B = matrices["lora_B"]

            # Validate shapes
            if lora_A.shape[0] != lora_B.shape[1]:
                raise ValueError(
                    f"LoRA rank mismatch for {layer_name}: "
                    f"lora_A rank {lora_A.shape[0]} != lora_B rank {lora_B.shape[1]}"
                )

            adapter.layers[layer_name] = LoRALayer(
                name=layer_name,
                lora_A=lora_A.astype(np.float32),
                lora_B=lora_B.astype(np.float32),
                alpha=alpha,
            )

    return adapter


def apply_lora(
    base_weight: np.ndarray,
    lora_A: np.ndarray,
    lora_B: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Apply LoRA to a base weight matrix.

    Merged weight = base_weight + scale * (lora_B @ lora_A)

    Parameters:
        base_weight: Base weight matrix of shape (out_features, in_features)
        lora_A: LoRA A matrix of shape (rank, in_features)
        lora_B: LoRA B matrix of shape (out_features, rank)
        scale: Scaling factor (typically alpha / rank)

    Returns:
        Merged weight matrix of same shape as base_weight
    """
    # Validate shapes
    rank = lora_A.shape[0]
    lora_A.shape[1]
    lora_B.shape[0]

    assert lora_A.shape == (rank, base_weight.shape[1]), (
        f"lora_A shape mismatch: {lora_A.shape} vs expected ({rank}, {base_weight.shape[1]})"
    )
    assert lora_B.shape == (base_weight.shape[0], rank), (
        f"lora_B shape mismatch: {lora_B.shape} vs expected ({base_weight.shape[0]}, {rank})"
    )

    # Compute LoRA delta: B @ A
    delta = lora_B @ lora_A

    # Apply scaling
    merged = base_weight + scale * delta

    return merged


def apply_lora_adapter(
    base_weights: dict[str, np.ndarray],
    adapter: LoRAAdapter,
    default_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Apply LoRA adapter to a full set of base model weights.

    Parameters:
        base_weights: Dict mapping layer names to weight matrices
        adapter: LoRAAdapter containing lora_A and lora_B for each layer
        default_scale: Default scale if layer doesn't have alpha

    Returns:
        Dict of merged weight matrices (same keys as base_weights)
    """
    merged = {}

    for layer_name, base_weight in base_weights.items():
        layer = adapter.get_layer_for_tensor(layer_name)
        if layer:
            merged[layer_name] = apply_lora(
                base_weight, layer.lora_A, layer.lora_B, scale=layer.scale()
            )
        else:
            merged[layer_name] = base_weight.copy()

    return merged


def merge_lora_into_model(
    base_model_path: str,
    lora_path: str,
    output_path: str,
    scale: float = 1.0,
) -> None:
    """
    Merge LoRA adapter into base model and save merged model.

    Parameters:
        base_model_path: Path to base model (safetensors)
        lora_path: Path to LoRA adapter (safetensors)
        output_path: Path to save merged model (safetensors)
        scale: Global scale factor
    """
    # Load base model
    base_weights = {}
    with safe_open(base_model_path, framework="numpy", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118
            base_weights[key] = f.get_tensor(key)

    # Load LoRA adapter
    adapter = load_lora_adapter(lora_path)

    # Apply adapter
    merged_weights = {}
    for key, weight in base_weights.items():
        layer = adapter.get_layer_for_tensor(key)
        if layer:
            merged_weights[key] = apply_lora(
                weight,
                layer.lora_A,
                layer.lora_B,
                scale=layer.scale() * scale,
            )
        else:
            merged_weights[key] = weight.copy()

    # Save merged model
    save_file(merged_weights, output_path)


@dataclass(slots=True)
class LoRADiffResult:
    """Result of diffing base model with LoRA adapter."""

    tensor_results: list
    summary: dict


def diff_with_lora(
    base_model_path: str,
    lora_path: str,
    lora_scale: float = 1.0,
    tensor_filter: str | None = None,
    progress_callback: callable | None = None,
    tie_word_embeddings: bool = False,
    deep: bool = False,
    lens: bool = False,
    tokenizer: str | None = None,
    lens_top_k: int = 8,
) -> LoRADiffResult:
    """
    Diff a base model against the same model with LoRA adapter applied.

    This avoids materializing the full merged model by computing deltas on-the-fly.
    """
    import re
    import time

    from wdiff.engine.diff import TensorDiffResult
    from wdiff.engine.lens import (
        Vocabulary,
        decode_layer_concept,
        is_down_proj_layer,
    )
    from wdiff.engine.metrics import MetricsBundle, compute_all_metrics
    from wdiff.engine.spectral import compute_spectral_bundle
    from wdiff.engine.taxonomy import classify_weight_delta

    @dataclass(slots=True)
    class DiffSummary:
        matched_count: int = 0
        partial_match_count: int = 0
        mismatched_shape_count: int = 0
        only_in_a_count: int = 0
        only_in_b_count: int = 0
        lens_analyzed_count: int = 0
        max_frobenius_tensor: str | None = None
        max_frobenius_value: float = 0.0
        min_cosine_value: float = 1.0
        mean_frobenius: float = 0.0
        _frobenius_sum: float = 0.0
        _frobenius_count: int = 0

        def add_match(self, name: str, metrics: MetricsBundle) -> None:
            self.matched_count += 1
            self._frobenius_sum += metrics.relative_frobenius
            self._frobenius_count += 1
            self.mean_frobenius = self._frobenius_sum / self._frobenius_count
            if metrics.relative_frobenius > self.max_frobenius_value:
                self.max_frobenius_value = metrics.relative_frobenius
                self.max_frobenius_tensor = name
            self.min_cosine_value = min(self.min_cosine_value, metrics.cosine_similarity)

    start_time = time.perf_counter()

    # Load LoRA adapter
    adapter = load_lora_adapter(lora_path)

    # Get tensor names from base model
    from wdiff.readers import get_reader

    wu_matrix = None
    wu_vocab_size = 0
    lens_vocab = None
    if lens:
        wu_candidates = [
            "output.weight",
            "lm_head.weight",
            "token_embd.weight",
            "model.embed_tokens.weight",
        ]
        with get_reader(base_model_path) as r:
            for cand in wu_candidates:
                if cand in r.tensor_names():
                    with contextlib.suppress(Exception):
                        info = r.tensor_info(cand)
                        wu_vocab_size = info.shape[0]
                        from wdiff.readers.gguf_reader import GGUFReader

                        if isinstance(r, GGUFReader) and cand in r._tensor_map:
                            from gguf.quants import dequantize

                            t_info = r._tensor_map[cand]
                            qtype = int(t_info.tensor_type)

                            def _make_gguf_getter(ti, qt):
                                def _getter(start: int, end: int) -> np.ndarray:
                                    chunk = ti.data[start:end]
                                    return dequantize(chunk, qt).astype(np.float32, copy=False)

                                return _getter

                            wu_matrix = _make_gguf_getter(t_info, qtype)
                        elif hasattr(r, "_file") and hasattr(r._file, "get_slice"):

                            def _make_safe_getter(reader_obj, tensor_name):
                                def _getter(start: int, end: int) -> np.ndarray:
                                    return reader_obj._file.get_slice(tensor_name)[
                                        start:end
                                    ].astype(np.float32, copy=False)

                                return _getter

                            wu_matrix = _make_safe_getter(r, cand)
                        else:
                            wu_matrix = r.get_tensor(cand)
                        break
        if wu_matrix is not None:
            if tokenizer:
                lens_vocab = Vocabulary.from_json_file(tokenizer) or Vocabulary.auto_discover(
                    tokenizer
                )
            if not lens_vocab or not lens_vocab._id_to_token:
                lens_vocab = Vocabulary.auto_discover(base_model_path)
            if not lens_vocab:
                lens_vocab = Vocabulary({})

    with get_reader(base_model_path) as reader_a:
        names_a = list(reader_a.tensor_names())

    # Filter if needed
    if tensor_filter:
        pattern = re.compile(tensor_filter)
        names_a = [n for n in names_a if pattern.search(n)]

    # For LoRA diff, we compare base model vs base+LoRA
    tensor_results = []
    summary = DiffSummary()

    with get_reader(base_model_path) as reader_a:
        for name in names_a:
            info_a = reader_a.tensor_info(name)

            # Check if this layer has a LoRA adapter
            layer = adapter.get_layer_for_tensor(name)
            if layer:
                base_weight = reader_a.get_tensor(name)
                merged_weight = apply_lora(
                    base_weight,
                    layer.lora_A,
                    layer.lora_B,
                    scale=layer.scale() * lora_scale,
                )
                metrics = compute_all_metrics(base_weight, merged_weight)
                diff_matrix = merged_weight - base_weight
                diff_frob = float(np.linalg.norm(diff_matrix.astype(np.float64, copy=False)))

                spectral = None
                taxonomy = None
                lens_result = None
                run_lens = lens and wu_matrix is not None and is_down_proj_layer(name)
                if deep or run_lens:
                    if deep:
                        spectral = compute_spectral_bundle(
                            diff_matrix, k=16, frobenius_norm=diff_frob
                        )
                        taxonomy = classify_weight_delta(diff_matrix, metrics, spectral)
                    if run_lens:
                        v_size = (
                            wu_vocab_size
                            if wu_vocab_size > 0
                            else (wu_matrix.shape[0] if hasattr(wu_matrix, "shape") else 0)
                        )
                        vocab = lens_vocab if lens_vocab is not None else Vocabulary({})
                        try:
                            lens_result = decode_layer_concept(
                                diff_matrix=diff_matrix,
                                layer_name=name,
                                canonical_name=name,
                                wu_matrix_or_getter=wu_matrix,
                                vocab_size=v_size,
                                vocab=vocab,
                                top_k=lens_top_k,
                            )
                            if lens_result is not None:
                                summary.lens_analyzed_count += 1
                        except Exception:  # noqa: BLE001
                            lens_result = None

                tensor_results.append(
                    TensorDiffResult(
                        name=name,
                        shape_a=info_a.shape,
                        shape_b=info_a.shape,
                        metrics=metrics,
                        status="MATCH",
                        spectral=spectral,
                        taxonomy=taxonomy,
                        lens=lens_result,
                    )
                )
                summary.add_match(name, metrics)
                del diff_matrix, base_weight, merged_weight
            else:
                tensor_results.append(
                    TensorDiffResult(
                        name=name,
                        shape_a=info_a.shape,
                        shape_b=info_a.shape,
                        metrics=MetricsBundle(0.0, 1.0, 0.0, 0.0),
                        status="MATCH",
                        spectral=None,
                        taxonomy=None,
                        lens=None,
                    )
                )
                summary.add_match(name, MetricsBundle(0.0, 1.0, 0.0, 0.0))

    duration = time.perf_counter() - start_time

    return LoRADiffResult(
        tensor_results=tensor_results,
        summary={
            "matched_count": summary.matched_count,
            "partial_match_count": summary.partial_match_count,
            "mismatched_shape_count": summary.mismatched_shape_count,
            "only_in_a_count": summary.only_in_a_count,
            "only_in_b_count": summary.only_in_b_count,
            "lens_analyzed_count": summary.lens_analyzed_count,
            "max_frobenius_tensor": summary.max_frobenius_tensor,
            "max_frobenius_value": summary.max_frobenius_value,
            "min_cosine_value": summary.min_cosine_value,
            "mean_frobenius": summary.mean_frobenius,
            "duration_seconds": duration,
        },
    )


# Re-export for convenience
__all__ = [
    "LoRAAdapter",
    "LoRADiffResult",
    "LoRALayer",
    "apply_lora",
    "apply_lora_adapter",
    "diff_with_lora",
    "load_lora_adapter",
    "merge_lora_into_model",
]
