from __future__ import annotations

import contextlib
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wdiff.engine.lens import (
    LayerLensResult,
    Vocabulary,
    decode_layer_concept,
    is_down_proj_layer,
)
from wdiff.engine.metrics import MetricsBundle, compute_all_metrics
from wdiff.engine.ring_buffer import PipelinedRingBuffer
from wdiff.engine.rope import deinterleave_rope, is_rope_tensor
from wdiff.engine.spectral import (
    SpectralBundle,
    SubspaceGeometry,
    compute_spectral_bundle,
    compute_subspace_geometry,
)
from wdiff.engine.taxonomy import TaxonomyResult, classify_weight_delta
from wdiff.readers import align_tensor_names, get_reader
from wdiff.readers.gguf_reader import GGUFReader


@dataclass(slots=True)
class PartialMatchInfo:
    """Information about partial shape match (single-axis variance slicing)."""

    common_shape: tuple[int, ...]
    differing_axis: int
    extra_count: int
    extra_source: str  # "model_a" or "model_b"
    extra_norm: float
    extra_mean: float
    extra_std: float


@dataclass(slots=True)
class TensorDiffResult:
    name: str
    shape_a: tuple[int, ...]
    shape_b: tuple[int, ...]
    metrics: MetricsBundle | None
    status: str
    error_message: str | None = None
    spectral: SpectralBundle | None = None
    taxonomy: TaxonomyResult | None = None
    partial_info: PartialMatchInfo | None = None
    lens: LayerLensResult | None = None
    subspace: SubspaceGeometry | None = None


@dataclass(slots=True)
class DiffSummary:
    matched_count: int = 0
    mismatched_shape_count: int = 0
    partial_match_count: int = 0
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

    def add_partial_match(self, name: str, metrics: MetricsBundle) -> None:
        self.partial_match_count += 1
        self._frobenius_sum += metrics.relative_frobenius
        self._frobenius_count += 1
        self.mean_frobenius = self._frobenius_sum / self._frobenius_count
        if metrics.relative_frobenius > self.max_frobenius_value:
            self.max_frobenius_value = metrics.relative_frobenius
            self.max_frobenius_tensor = name
        self.min_cosine_value = min(self.min_cosine_value, metrics.cosine_similarity)


@dataclass(slots=True)
class DiffResult:
    model_a_path: str
    model_b_path: str
    format_a: str
    format_b: str
    tensor_results: list[TensorDiffResult]
    summary: DiffSummary
    duration_seconds: float
    peak_ram_mb: float = 0.0


def _detect_format(path: str) -> str:
    try:
        from wdiff.readers import detect_format

        return detect_format(path)
    except (ValueError, FileNotFoundError, OSError):
        if path.endswith(".safetensors"):
            return "safetensors"
        if path.endswith(".gguf"):
            return "gguf"
        return "unknown"


def _try_partial_match(
    reader_a,
    reader_b,
    raw_name_a: str,
    raw_name_b: str,
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
    canonical_name: str,
    deep: bool,
    is_cross_format: bool,
    gguf_reader,
    rope_parity: bool,
) -> TensorDiffResult | None:
    """
    Attempt single-axis common-prefix slicing for shape-mismatched tensors.

    Returns a TensorDiffResult with status PARTIAL_MATCH if successful,
    or None if multi-axis variance or different ndim (falls back to SHAPE_MISMATCH).
    """
    if len(shape_a) != len(shape_b):
        return None

    # Find differing axes
    diff_axes = [axis for axis in range(len(shape_a)) if shape_a[axis] != shape_b[axis]]
    if len(diff_axes) != 1:
        return None

    diff_axis = diff_axes[0]
    common_size = min(shape_a[diff_axis], shape_b[diff_axis])

    # Determine which model has extra tokens
    if shape_b[diff_axis] > shape_a[diff_axis]:
        extra_source = "model_b"
        extra_count = shape_b[diff_axis] - shape_a[diff_axis]
    else:
        extra_source = "model_a"
        extra_count = shape_a[diff_axis] - shape_b[diff_axis]

    # Build slice for common prefix
    slice_a = [slice(None)] * len(shape_a)
    slice_b = [slice(None)] * len(shape_b)
    slice_a[diff_axis] = slice(0, common_size)
    slice_b[diff_axis] = slice(0, common_size)

    # Load tensors and slice common prefix
    tensor_a = reader_a.get_tensor(raw_name_a)
    tensor_b = reader_b.get_tensor(raw_name_b)

    common_a = tensor_a[tuple(slice_a)]
    common_b = tensor_b[tuple(slice_b)]

    # Extract expanded portion for telemetry
    if extra_source == "model_b":
        extra_slice = [slice(None)] * len(shape_b)
        extra_slice[diff_axis] = slice(common_size, None)
        extra_tensor = tensor_b[tuple(extra_slice)]
    else:
        extra_slice = [slice(None)] * len(shape_a)
        extra_slice[diff_axis] = slice(common_size, None)
        extra_tensor = tensor_a[tuple(extra_slice)]

    # Apply RoPE parity if needed (for cross-format)
    if is_cross_format and rope_parity and is_rope_tensor(canonical_name):
        n_heads = _get_head_count_for_tensor(
            canonical_name,
            gguf_reader.head_count if gguf_reader else None,
            gguf_reader.head_count_kv if gguf_reader else None,
        )
        if n_heads is not None and n_heads > 0:
            if isinstance(reader_a, GGUFReader):
                candidate = deinterleave_rope(common_a, n_heads)
                if np.vdot(candidate, common_b) > np.vdot(common_a, common_b):
                    common_a = candidate
            elif isinstance(reader_b, GGUFReader):
                candidate = deinterleave_rope(common_b, n_heads)
                if np.vdot(common_a, candidate) > np.vdot(common_a, common_b):
                    common_b = candidate

    # Compute metrics on common prefix
    metrics = compute_all_metrics(common_a, common_b)

    # Compute expansion telemetry
    extra_norm = float(np.linalg.norm(extra_tensor))
    extra_mean = float(np.mean(extra_tensor))
    extra_std = float(np.std(extra_tensor))

    # Build common shape (with common_size on diff_axis)
    common_shape = list(shape_a)
    common_shape[diff_axis] = common_size
    common_shape = tuple(common_shape)

    # Compute spectral/taxonomy if deep
    spectral = None
    taxonomy = None
    subspace = None
    if deep:
        diff_matrix = common_a - common_b
        diff_frob = float(np.linalg.norm(diff_matrix.astype(np.float64, copy=False)))
        spectral = compute_spectral_bundle(diff_matrix, k=16, frobenius_norm=diff_frob)
        taxonomy = classify_weight_delta(diff_matrix, metrics, spectral)
        subspace = compute_subspace_geometry(common_a, common_b, k=16)
        del diff_matrix

    partial_info = PartialMatchInfo(
        common_shape=common_shape,
        differing_axis=diff_axis,
        extra_count=extra_count,
        extra_source=extra_source,
        extra_norm=extra_norm,
        extra_mean=extra_mean,
        extra_std=extra_std,
    )

    return TensorDiffResult(
        name=canonical_name,
        shape_a=shape_a,
        shape_b=shape_b,
        metrics=metrics,
        status="PARTIAL_MATCH",
        spectral=spectral,
        taxonomy=taxonomy,
        partial_info=partial_info,
        subspace=subspace,
    )


def _process_peak_ram_mb() -> float:
    """Process peak RSS in MB (ru_maxrss: KiB on Linux, bytes on macOS; 0.0 if unavailable)."""
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return 0.0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak / (1024 * 1024)
    return peak / 1024


def diff_models(
    path_a: str,
    path_b: str,
    tensor_filter: str | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    tie_word_embeddings: bool = False,
    deep: bool = False,
    rope_parity: bool = True,
    lens: bool = False,
    tokenizer: str | Path | None = None,
    lens_top_k: int = 5,
) -> DiffResult:
    start_time = time.perf_counter()

    reader_a = get_reader(path_a)
    reader_b = get_reader(path_b)

    # Check if this is a cross-format diff (safetensors vs gguf)
    format_a = _detect_format(path_a)
    format_b = _detect_format(path_b)
    is_cross_format = (
        format_a in ("safetensors", "sharded_safetensors") and format_b == "gguf"
    ) or (format_a == "gguf" and format_b in ("safetensors", "sharded_safetensors"))

    # Identify which reader is GGUF for RoPE parity
    gguf_reader = None
    if is_cross_format and rope_parity:
        if format_a == "gguf":
            gguf_reader = reader_a
        elif format_b == "gguf":
            gguf_reader = reader_b

    with reader_a, reader_b:
        # Lens analysis setup: find unembedding matrix (W_U) in either model
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
            for cand in wu_candidates:
                for r in (reader_b, reader_a):
                    if cand in r.tensor_names():
                        with contextlib.suppress(Exception):
                            info = r.tensor_info(cand)
                            wu_vocab_size = info.shape[0]
                            # Use memory-efficient chunked getters to avoid 1GB+ allocations
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
                    break

            if wu_matrix is not None:
                if tokenizer:
                    lens_vocab = Vocabulary.from_json_file(tokenizer) or Vocabulary.auto_discover(
                        tokenizer
                    )
                if not lens_vocab or not lens_vocab._id_to_token:
                    lens_vocab = Vocabulary.auto_discover(path_b, reader_b)
                if not lens_vocab or not lens_vocab._id_to_token:
                    lens_vocab = Vocabulary.auto_discover(path_a, reader_a)
                if not lens_vocab:
                    lens_vocab = Vocabulary({})

        names_a = list(reader_a.tensor_names())
        names_b = list(reader_b.tensor_names())

        if tensor_filter:
            pattern = re.compile(tensor_filter)
            names_a = [n for n in names_a if pattern.search(n)]
            names_b = [n for n in names_b if pattern.search(n)]

        # Use alignment system for cross-format tensor matching
        alignment = align_tensor_names(names_a, names_b, tie_word_embeddings=tie_word_embeddings)

        tensor_results: list[TensorDiffResult] = []
        summary = DiffSummary()

        # Check shapes for matched pairs and collect raw names
        matched_results: dict[str, TensorDiffResult] = {}
        shape_matched: list[
            tuple[str, str, str, tuple[int, ...], tuple[int, ...]]
        ] = []  # (canonical, raw_a, raw_b, shape_a, shape_b)
        for canonical_name, raw_name_a, raw_name_b in alignment.matched_pairs:
            info_a = reader_a.tensor_info(raw_name_a)
            info_b = reader_b.tensor_info(raw_name_b)

            if info_a.shape != info_b.shape:
                # Try partial match (single-axis variance slicing)
                partial_result = _try_partial_match(
                    reader_a,
                    reader_b,
                    raw_name_a,
                    raw_name_b,
                    info_a.shape,
                    info_b.shape,
                    canonical_name,
                    deep,
                    is_cross_format,
                    gguf_reader,
                    rope_parity,
                )
                if partial_result is not None:
                    matched_results[canonical_name] = partial_result
                    summary.add_partial_match(canonical_name, partial_result.metrics)
                else:
                    matched_results[canonical_name] = TensorDiffResult(
                        name=canonical_name,
                        shape_a=info_a.shape,
                        shape_b=info_b.shape,
                        metrics=None,
                        status="SHAPE_MISMATCH",
                    )
                    summary.mismatched_shape_count += 1
            else:
                shape_matched.append(
                    (canonical_name, raw_name_a, raw_name_b, info_a.shape, info_b.shape)
                )

        # Process shape-matched tensors through ring buffer
        if shape_matched:
            names_a_for_buffer = [raw_a for _, raw_a, _, _, _ in shape_matched]
            names_b_for_buffer = [raw_b for _, _, raw_b, _, _ in shape_matched]

            buffer = PipelinedRingBuffer(max_depth=2)
            results = _run_paired_buffer(
                buffer,
                reader_a,
                reader_b,
                names_a_for_buffer,
                names_b_for_buffer,
                progress_callback,
                deep=deep,
                rope_parity=rope_parity,
                is_cross_format=is_cross_format,
                gguf_reader=gguf_reader,
                canonical_names=[canon for canon, _, _, _, _ in shape_matched],
                lens=lens,
                wu_matrix=wu_matrix,
                wu_vocab_size=wu_vocab_size,
                lens_vocab=lens_vocab,
                lens_top_k=lens_top_k,
            )

            for (canonical_name, _, _, shape_a, shape_b), (
                metrics,
                spectral,
                taxonomy,
                lens_result,
                subspace,
            ) in zip(shape_matched, results):
                matched_results[canonical_name] = TensorDiffResult(
                    name=canonical_name,
                    shape_a=shape_a,
                    shape_b=shape_b,
                    metrics=metrics,
                    status="MATCH",
                    spectral=spectral,
                    taxonomy=taxonomy,
                    lens=lens_result,
                    subspace=subspace,
                )
                summary.add_match(canonical_name, metrics)
                if lens_result is not None:
                    summary.lens_analyzed_count += 1

        # Build tensor_results preserving original aligned pair order
        for canonical_name, _, _ in alignment.matched_pairs:
            if canonical_name in matched_results:
                tensor_results.append(matched_results[canonical_name])

        # Handle tensors only in A
        for raw_name in alignment.only_in_a:
            info_a = reader_a.tensor_info(raw_name)
            tensor_results.append(
                TensorDiffResult(
                    name=raw_name,
                    shape_a=info_a.shape,
                    shape_b=(),
                    metrics=None,
                    status="ONLY_IN_A",
                )
            )
            summary.only_in_a_count += 1

        # Handle tensors only in B
        for raw_name in alignment.only_in_b:
            info_b = reader_b.tensor_info(raw_name)
            tensor_results.append(
                TensorDiffResult(
                    name=raw_name,
                    shape_a=(),
                    shape_b=info_b.shape,
                    metrics=None,
                    status="ONLY_IN_B",
                )
            )
            summary.only_in_b_count += 1

        duration = time.perf_counter() - start_time

        return DiffResult(
            model_a_path=path_a,
            model_b_path=path_b,
            format_a=_detect_format(path_a),
            format_b=_detect_format(path_b),
            tensor_results=tensor_results,
            summary=summary,
            duration_seconds=duration,
            peak_ram_mb=round(_process_peak_ram_mb(), 2),
        )


def _get_head_count_for_tensor(
    name: str, head_count: int | None, head_count_kv: int | None
) -> int | None:
    """Get the appropriate head count for a RoPE tensor (Q uses head_count, K uses head_count_kv)."""
    if "attn_q" in name or "q_proj" in name:
        return head_count
    if "attn_k" in name or "k_proj" in name:
        return head_count_kv
    return None


def _run_paired_buffer(
    buffer: PipelinedRingBuffer,
    reader_a,
    reader_b,
    names_a: list[str],
    names_b: list[str],
    progress_callback: Callable[[int, int, str], None] | None = None,
    deep: bool = False,
    rope_parity: bool = True,
    is_cross_format: bool = False,
    gguf_reader=None,
    canonical_names: list[str] | None = None,
    lens: bool = False,
    wu_matrix: np.ndarray | Callable[[int, int], np.ndarray] | None = None,
    wu_vocab_size: int = 0,
    lens_vocab: Vocabulary | None = None,
    lens_top_k: int = 5,
) -> list[
    tuple[
        MetricsBundle,
        SpectralBundle | None,
        TaxonomyResult | None,
        LayerLensResult | None,
        SubspaceGeometry | None,
    ]
]:
    """Run ring buffer with paired tensor names from different models."""
    results: list[
        tuple[
            MetricsBundle,
            SpectralBundle | None,
            TaxonomyResult | None,
            LayerLensResult | None,
            SubspaceGeometry | None,
        ]
    ] = []
    total = len(names_a)

    # Extract head counts from GGUF reader if available
    head_count = None
    head_count_kv = None
    if is_cross_format and rope_parity and gguf_reader is not None:
        head_count = gguf_reader.head_count
        head_count_kv = gguf_reader.head_count_kv

    for i, (name_a, name_b) in enumerate(zip(names_a, names_b)):
        tensor_a = reader_a.get_tensor(name_a)
        tensor_b = reader_b.get_tensor(name_b)

        canonical_name = canonical_names[i] if canonical_names else name_a

        # Apply RoPE parity correction for cross-format diffs
        if is_cross_format and rope_parity and is_rope_tensor(canonical_name):
            # Determine which tensor is from GGUF (needs deinterleaving)
            # The GGUF tensor is in interleaved format, HF is in split-half
            # We need to deinterleave the GGUF tensor to match HF
            n_heads = _get_head_count_for_tensor(canonical_name, head_count, head_count_kv)

            # If we have head count, use it; otherwise try common values
            if n_heads is not None and n_heads > 0:
                # Determine which reader is GGUF
                if isinstance(reader_a, GGUFReader):
                    candidate_a = deinterleave_rope(tensor_a, n_heads)
                    if np.vdot(candidate_a, tensor_b) > np.vdot(tensor_a, tensor_b):
                        tensor_a = candidate_a
                elif isinstance(reader_b, GGUFReader):
                    candidate_b = deinterleave_rope(tensor_b, n_heads)
                    if np.vdot(tensor_a, candidate_b) > np.vdot(tensor_a, tensor_b):
                        tensor_b = candidate_b
            else:
                # Fallback: try common head dimensions
                for test_d_head in (128, 64, 256, 96, 32):
                    test_n_heads = tensor_a.shape[0] // test_d_head
                    if tensor_a.shape[0] % test_d_head == 0 and test_n_heads > 0:
                        if isinstance(reader_a, GGUFReader):
                            candidate_a = deinterleave_rope(tensor_a, test_n_heads, test_d_head)
                            if np.vdot(candidate_a, tensor_b) > np.vdot(tensor_a, tensor_b):
                                tensor_a = candidate_a
                                break
                        elif isinstance(reader_b, GGUFReader):
                            candidate_b = deinterleave_rope(tensor_b, test_n_heads, test_d_head)
                            if np.vdot(tensor_a, candidate_b) > np.vdot(tensor_a, tensor_b):
                                tensor_b = candidate_b
                                break

        metrics = buffer.compute_metrics(tensor_a, tensor_b)
        spectral = None
        taxonomy = None
        lens_result = None
        subspace = None

        run_lens = lens and wu_matrix is not None and is_down_proj_layer(canonical_name)
        if deep or run_lens:
            diff_matrix = tensor_a - tensor_b
            if deep:
                diff_frob = float(np.linalg.norm(diff_matrix.astype(np.float64, copy=False)))
                spectral = compute_spectral_bundle(diff_matrix, k=16, frobenius_norm=diff_frob)
                taxonomy = classify_weight_delta(diff_matrix, metrics, spectral)
                subspace = compute_subspace_geometry(tensor_a, tensor_b, k=16)
            if run_lens:
                v_size = (
                    wu_vocab_size
                    if wu_vocab_size > 0
                    else (wu_matrix.shape[0] if hasattr(wu_matrix, "shape") else 0)
                )
                with contextlib.suppress(Exception):
                    lens_result = decode_layer_concept(
                        diff_matrix=diff_matrix,
                        layer_name=canonical_name,
                        canonical_name=canonical_name,
                        wu_matrix_or_getter=wu_matrix,
                        vocab_size=v_size,
                        vocab=lens_vocab if lens_vocab is not None else Vocabulary({}),
                        top_k=lens_top_k,
                    )
            del diff_matrix

        results.append((metrics, spectral, taxonomy, lens_result, subspace))

        if progress_callback:
            progress_callback(i + 1, total, name_a)

        del tensor_a
        del tensor_b

    return results
