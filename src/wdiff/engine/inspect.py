from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wdiff.engine.spectral import SpectralBundle, compute_spectral_bundle
from wdiff.readers import get_reader


@dataclass(slots=True)
class LayerInspectionResult:
    name: str
    shape: tuple[int, ...]
    dtype: str
    param_count: int
    norm: float
    mean: float
    std: float
    dead_neuron_count: int  # rows/columns where norm < 1e-7
    outlier_score: float  # (max|w| - median) / (MAD + 1e-12)
    is_outlier_risk: bool  # outlier_score > 15.0
    spectral: SpectralBundle | None = None
    taxonomy: object | None = None

    @property
    def status(self) -> str:
        if self.dead_neuron_count > 0:
            return "DEAD_NEURONS"
        elif self.is_outlier_risk:
            return "OUTLIER_SPIKE"
        return "HEALTHY"


@dataclass(slots=True)
class InspectionSummary:
    total_params: int
    trainable_params_mb: float
    architecture: str
    tensor_count: int
    dead_neurons_total: int
    outlier_risk_layer_count: int
    quantization_readiness: str  # "EXCELLENT", "MODERATE", "POOR"
    duration_seconds: float
    mean_frobenius: float = 0.0
    min_cosine_value: float = 1.0


@dataclass(slots=True)
class ModelInspectionResult:
    model_path: str
    format: str
    summary: InspectionSummary
    layers: list[LayerInspectionResult]


def _count_dead_neurons(w: np.ndarray) -> tuple[int, int]:
    """Count dead neurons (rows/columns with L2 norm < 1e-7).

    Returns:
        (dead_row_count, dead_col_count)
    """
    if w.ndim != 2:
        return 0, 0

    # For huge matrices (>20M elements), compute in chunks to avoid memory ballooning
    if w.size > 20_000_000:
        chunk_size = 4096
        dead_rows = 0
        col_sq_sums = np.zeros(w.shape[1], dtype=np.float64)
        for start in range(0, w.shape[0], chunk_size):
            end = min(start + chunk_size, w.shape[0])
            chunk = w[start:end]
            row_norms = np.linalg.norm(chunk, axis=1)
            dead_rows += int(np.sum(row_norms < 1e-7))
            col_sq_sums += np.sum(chunk.astype(np.float64, copy=False) ** 2, axis=0)
        dead_cols = int(np.sum(np.sqrt(col_sq_sums) < 1e-7))
        return dead_rows, dead_cols

    # Check rows (neurons in output dimension)
    row_norms = np.linalg.norm(w, axis=1)
    dead_rows = int(np.sum(row_norms < 1e-7))

    # Check columns (neurons in input dimension)
    col_norms = np.linalg.norm(w, axis=0)
    dead_cols = int(np.sum(col_norms < 1e-7))

    return dead_rows, dead_cols


def _compute_outlier_score(w: np.ndarray) -> float:
    """Compute Dettmers et al. (LLM.int8()) outlier score.

    Score = (max|w| - median(|w|)) / (MAD / 0.67448975 + 1e-12)

    Where MAD = median(|w - median(w)|)

    Returns:
        Outlier score. If > 15.0, indicates quantization risk.
    """
    n_elements = w.size
    if n_elements == 0:
        return 0.0

    # Compute max without allocating full np.abs(w) array
    max_val = float(max(abs(float(np.min(w))), abs(float(np.max(w)))))

    # For massive tensors (>100k params), subsample for median and MAD
    # to maintain strict streaming bounds while keeping max_val exact
    if n_elements > 100_000:
        stride = n_elements // 100_000
        sample = np.abs(w.ravel()[::stride])
    else:
        sample = np.abs(w.ravel())

    median_val = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median_val)))

    # Avoid division by zero
    mad_scaled = mad / 0.67448975 + 1e-12

    score = (max_val - median_val) / mad_scaled
    return score


def _compute_quantization_readiness(outlier_risk_layer_count: int) -> str:
    """Assess quantization readiness based on outlier risk layer count."""
    if outlier_risk_layer_count == 0:
        return "EXCELLENT (Safe for Q4_K_M)"
    elif outlier_risk_layer_count <= 3:
        return "MODERATE (Consider protecting outlier layers)"
    else:
        return "POOR (High risk of quantization collapse)"


def _get_health_status(layer: LayerInspectionResult) -> str:
    """Get health status for a layer."""
    if layer.dead_neuron_count > 0:
        return "DEAD_NEURONS"
    elif layer.is_outlier_risk:
        return "OUTLIER_SPIKE"
    return "HEALTHY"


def inspect_model(
    model_path: str | Path,
    deep: bool = False,
    tensor_filter: str | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> ModelInspectionResult:
    """
    Inspect a single model checkpoint.

    Parameters:
        model_path: Path to model checkpoint (.safetensors or .gguf)
        deep: Whether to compute spectral analysis (SVD)
        tensor_filter: Regex filter for tensor names
        progress_callback: Optional callback(completed, total, layer_name)

    Returns:
        ModelInspectionResult with summary and per-layer inspections.
    """
    start_time = time.perf_counter()
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    reader = get_reader(str(model_path))

    with reader:
        format_name = reader.__class__.__name__.replace("Reader", "").lower()
        if "safetensors" in format_name:
            format_name = "safetensors"
        elif "gguf" in format_name:
            format_name = "gguf"

        # Get all tensor names
        all_names = list(reader.tensor_names())

        # Filter if needed
        if tensor_filter:
            pattern = re.compile(tensor_filter)
            names = [n for n in all_names if pattern.search(n)]
        else:
            names = all_names

        # Detect architecture
        from wdiff.readers import TensorNameNormalizer

        architecture = TensorNameNormalizer.detect_architecture(names)

        # Process each tensor
        layer_results = []
        total_params = 0
        dead_neurons_total = 0
        outlier_risk_count = 0

        total = len(names)

        for i, name in enumerate(names):
            if progress_callback:
                progress_callback(i, total, name)

            # Get tensor info
            t_info = reader.tensor_info(name)

            # Chunked streaming path (4096-row chunks) bounding the per-chunk working set
            from wdiff.readers.gguf_reader import GGUFReader

            if (
                isinstance(reader, GGUFReader)
                and hasattr(reader, "_tensor_map")
                and name in reader._tensor_map
                and int(np.prod(t_info.shape)) > 20_000_000
                and reader._tensor_map[name].data.ndim == 2
            ):
                t_raw = reader._tensor_map[name]
                from gguf.quants import dequantize

                qtype = int(t_raw.tensor_type)
                n_rows = t_raw.data.shape[0]
                shape = t_info.shape
                param_count = int(np.prod(shape))
                total_params += param_count

                chunk_size = 4096
                total_sum = 0.0
                sum_sq = 0.0
                max_val = 0.0
                sample_vals = []
                dead_rows = 0
                col_sq_sums = None

                for start in range(0, n_rows, chunk_size):
                    end = min(start + chunk_size, n_rows)
                    raw_slice = t_raw.data[start:end]
                    chunk = dequantize(raw_slice, qtype)
                    if col_sq_sums is None:
                        col_sq_sums = np.zeros(chunk.shape[1], dtype=np.float64)
                    chunk_f64 = chunk.astype(np.float64, copy=False)
                    total_sum += float(np.sum(chunk_f64))
                    sum_sq += float(np.sum(chunk_f64**2))
                    max_val = max(max_val, float(np.max(np.abs(chunk))))
                    row_norms = np.linalg.norm(chunk, axis=1)
                    dead_rows += int(np.sum(row_norms < 1e-7))
                    col_sq_sums += np.sum(chunk_f64**2, axis=0)
                    step = max(1, chunk.shape[0] // 50)
                    sample_vals.append(chunk[::step, ::100].ravel())

                all_samples = np.abs(np.concatenate(sample_vals))
                med = float(np.median(all_samples))
                mad = float(np.median(np.abs(all_samples - med))) / 0.67448975 + 1e-12
                outlier_score = (max_val - med) / mad
                norm = float(np.sqrt(sum_sq))
                mean = float(total_sum / param_count)
                variance = max(0.0, float(sum_sq / param_count - mean**2))
                std = float(np.sqrt(variance))
                dead_cols = int(np.sum(np.sqrt(col_sq_sums) < 1e-7))
                dead_neuron_count = dead_rows + dead_cols
                dead_neurons_total += dead_neuron_count
                is_outlier_risk = outlier_score > 15.0
                if is_outlier_risk:
                    outlier_risk_count += 1

                layer_results.append(
                    LayerInspectionResult(
                        name=name,
                        shape=shape,
                        dtype="float32",
                        param_count=param_count,
                        norm=norm,
                        mean=mean,
                        std=std,
                        dead_neuron_count=dead_neuron_count,
                        outlier_score=outlier_score,
                        is_outlier_risk=is_outlier_risk,
                        spectral=None,
                        taxonomy=None,
                    )
                )
                continue

            w = reader.get_tensor(name)

            # Compute basic stats
            param_count = int(np.prod(w.shape))
            total_params += param_count

            # Compute norm, mean, std
            flat = w.ravel()
            norm = float(np.linalg.norm(flat))
            mean = float(np.mean(flat))
            std = float(np.std(flat))

            # Dead neuron detection
            dead_rows, dead_cols = _count_dead_neurons(w)
            dead_neuron_count = dead_rows + dead_cols
            dead_neurons_total += dead_neuron_count

            # Outlier score
            outlier_score = _compute_outlier_score(w)
            is_outlier_risk = outlier_score > 15.0
            if is_outlier_risk:
                outlier_risk_count += 1

            # Deep spectral analysis
            spectral = None
            taxonomy = None
            if deep:
                spectral = compute_spectral_bundle(w, k=16)

            # Get dtype string
            dtype_str = str(w.dtype)

            layer_results.append(
                LayerInspectionResult(
                    name=name,
                    shape=w.shape,
                    dtype=dtype_str,
                    param_count=param_count,
                    norm=norm,
                    mean=mean,
                    std=std,
                    dead_neuron_count=dead_neuron_count,
                    outlier_score=outlier_score,
                    is_outlier_risk=is_outlier_risk,
                    spectral=spectral,
                    taxonomy=taxonomy,
                )
            )

            # Free memory immediately
            del w

        time.perf_counter() - start_time

        # Determine quantization readiness
        _compute_quantization_readiness(sum(1 for l in layer_results if l.is_outlier_risk))

        summary = InspectionSummary(
            total_params=total_params,
            trainable_params_mb=total_params * 4 / (1024 * 1024),  # assuming float32
            architecture=architecture.value
            if hasattr(architecture, "value")
            else str(architecture),
            tensor_count=len(layer_results),
            dead_neurons_total=dead_neurons_total,
            outlier_risk_layer_count=outlier_risk_count,
            quantization_readiness=_compute_quantization_readiness(outlier_risk_count),
            duration_seconds=time.perf_counter() - start_time,
            mean_frobenius=0.0,
            min_cosine_value=1.0,
        )

        return ModelInspectionResult(
            model_path=str(model_path),
            format=format_name,
            summary=summary,
            layers=layer_results,
        )


__all__ = [
    "InspectionSummary",
    "LayerInspectionResult",
    "ModelInspectionResult",
    "_compute_outlier_score",
    "_compute_quantization_readiness",
    "_count_dead_neurons",
    "inspect_model",
]
