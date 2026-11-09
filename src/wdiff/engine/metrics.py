from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MetricsBundle:
    relative_frobenius: float
    cosine_similarity: float
    max_element_delta: float
    mean_element_delta: float


_EPS = 1e-12


def _validate_shapes(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")


def relative_frobenius(a: np.ndarray, b: np.ndarray) -> float:
    _validate_shapes(a, b)
    if a.size > 2_000_000 and a.ndim >= 2:
        sq_diff = 0.0
        sq_a = 0.0
        chunk_rows = max(1, 4096 * 4096 // a.shape[-1])
        for i in range(0, a.shape[0], chunk_rows):
            d_c = (a[i : i + chunk_rows] - b[i : i + chunk_rows]).astype(np.float64, copy=False)
            a_c = a[i : i + chunk_rows].astype(np.float64, copy=False)
            sq_diff += float(np.sum(d_c**2))
            sq_a += float(np.sum(a_c**2))
        return float(np.sqrt(sq_diff) / (np.sqrt(sq_a) + _EPS))
    diff = (a - b).astype(np.float64, copy=False)
    norm_diff = float(np.linalg.norm(diff))
    norm_a = float(np.linalg.norm(a.astype(np.float64, copy=False)))
    return norm_diff / (norm_a + _EPS)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    _validate_shapes(a, b)
    if a.size > 2_000_000 and a.ndim >= 2:
        dot = 0.0
        sq_a = 0.0
        sq_b = 0.0
        chunk_rows = max(1, 4096 * 4096 // a.shape[-1])
        for i in range(0, a.shape[0], chunk_rows):
            a_c = a[i : i + chunk_rows].astype(np.float64, copy=False)
            b_c = b[i : i + chunk_rows].astype(np.float64, copy=False)
            dot += float(np.sum(a_c * b_c))
            sq_a += float(np.sum(a_c**2))
            sq_b += float(np.sum(b_c**2))
        return float(dot / (np.sqrt(sq_a) * np.sqrt(sq_b) + _EPS))
    a_flat = a.ravel().astype(np.float64, copy=False)
    b_flat = b.ravel().astype(np.float64, copy=False)
    dot = float(np.dot(a_flat, b_flat))
    norm_a = float(np.linalg.norm(a_flat))
    norm_b = float(np.linalg.norm(b_flat))
    return dot / (norm_a * norm_b + _EPS)


def max_element_delta(a: np.ndarray, b: np.ndarray) -> float:
    _validate_shapes(a, b)
    if a.size > 2_000_000 and a.ndim >= 2:
        max_d = 0.0
        chunk_rows = max(1, 4096 * 4096 // a.shape[-1])
        for i in range(0, a.shape[0], chunk_rows):
            d_c = np.abs(a[i : i + chunk_rows] - b[i : i + chunk_rows])
            max_d = max(max_d, float(np.max(d_c)))
        return max_d
    diff = np.abs(a - b)
    return float(np.max(diff))


def mean_element_delta(a: np.ndarray, b: np.ndarray) -> float:
    _validate_shapes(a, b)
    if a.size > 2_000_000 and a.ndim >= 2:
        sum_d = 0.0
        chunk_rows = max(1, 4096 * 4096 // a.shape[-1])
        for i in range(0, a.shape[0], chunk_rows):
            d_c = np.abs(a[i : i + chunk_rows] - b[i : i + chunk_rows])
            sum_d += float(np.sum(d_c.astype(np.float64, copy=False)))
        return float(sum_d / a.size)
    diff = np.abs(a - b)
    return float(np.mean(diff))


def compute_all_metrics(a: np.ndarray, b: np.ndarray) -> MetricsBundle:
    _validate_shapes(a, b)
    if a.size > 2_000_000 and a.ndim >= 2:
        sq_diff = 0.0
        sq_a = 0.0
        sq_b = 0.0
        dot = 0.0
        max_d = 0.0
        sum_d = 0.0
        chunk_rows = max(1, 4096 * 4096 // a.shape[-1])
        for i in range(0, a.shape[0], chunk_rows):
            a_c = a[i : i + chunk_rows]
            b_c = b[i : i + chunk_rows]
            d_c = (a_c - b_c).astype(np.float64, copy=False)
            a_c64 = a_c.astype(np.float64, copy=False)
            b_c64 = b_c.astype(np.float64, copy=False)
            sq_diff += float(np.sum(d_c**2))
            sq_a += float(np.sum(a_c64**2))
            sq_b += float(np.sum(b_c64**2))
            dot += float(np.sum(a_c64 * b_c64))
            abs_d = np.abs(d_c)
            max_d = max(max_d, float(np.max(abs_d)))
            sum_d += float(np.sum(abs_d))
        rel_frob = float(np.sqrt(sq_diff) / (np.sqrt(sq_a) + _EPS))
        cos_sim = float(dot / (np.sqrt(sq_a) * np.sqrt(sq_b) + _EPS))
        mean_d = float(sum_d / a.size)
        return MetricsBundle(
            relative_frobenius=rel_frob,
            cosine_similarity=cos_sim,
            max_element_delta=max_d,
            mean_element_delta=mean_d,
        )
    return MetricsBundle(
        relative_frobenius=relative_frobenius(a, b),
        cosine_similarity=cosine_similarity(a, b),
        max_element_delta=max_element_delta(a, b),
        mean_element_delta=mean_element_delta(a, b),
    )
