from __future__ import annotations

import re

import numpy as np

_ROPE_Q_PATTERNS = [
    re.compile(r".*attn_q\.(weight|bias)$"),
    re.compile(r".*q_proj\.(weight|bias)$"),
]
_ROPE_K_PATTERNS = [
    re.compile(r".*attn_k\.(weight|bias)$"),
    re.compile(r".*k_proj\.(weight|bias)$"),
]


def is_rope_tensor(name: str) -> bool:
    """Check if tensor name represents a query or key projection (RoPE-affected)."""
    for pattern in _ROPE_Q_PATTERNS:
        if pattern.match(name):
            return True
    for pattern in _ROPE_K_PATTERNS:
        if pattern.match(name):
            return True
    return False


def _is_q_proj(name: str) -> bool:
    for pattern in _ROPE_Q_PATTERNS:
        if pattern.match(name):
            return True
    return False


def _is_k_proj(name: str) -> bool:
    for pattern in _ROPE_K_PATTERNS:
        if pattern.match(name):
            return True
    return False


def _get_head_count_for_tensor(
    name: str,
    head_count: int | None,
    head_count_kv: int | None,
) -> int | None:
    """Get the appropriate head count for a RoPE tensor (Q uses head_count, K uses head_count_kv)."""
    if _is_q_proj(name):
        return head_count
    if _is_k_proj(name):
        return head_count_kv
    return None


def _validate_dims(weights: np.ndarray, n_heads: int, d_head: int | None) -> tuple[int, int]:
    """Validate and compute dimensions."""
    if weights.ndim not in (1, 2):
        raise ValueError(f"Expected 1D or 2D array, got {weights.ndim}D")

    if d_head is None:
        if weights.shape[0] % n_heads != 0:
            raise ValueError(
                f"Cannot infer d_head: shape[0]={weights.shape[0]} not divisible by n_heads={n_heads}"
            )
        d_head = weights.shape[0] // n_heads

    if d_head % 2 != 0:
        raise ValueError(f"d_head must be even, got {d_head}")

    if weights.shape[0] != n_heads * d_head:
        raise ValueError(
            f"Shape mismatch: weights.shape[0]={weights.shape[0]} != n_heads * d_head = {n_heads * d_head}"
        )

    return n_heads, d_head


def deinterleave_rope(weights: np.ndarray, n_heads: int, d_head: int | None = None) -> np.ndarray:
    """
    Invert GGML interleaved coordinate pairs back to HuggingFace split-half format.

    GGML (interleaved): [-x1, x0, -x3, x2, ...] per head
    HF (split-half):    [-x_{d/2:}, x_{:d/2}] per head

    Args:
        weights: Array of shape (n_heads * d_head, d_in) or (n_heads * d_head,)
        n_heads: Number of attention heads
        d_head: Head dimension (optional, inferred if not provided)

    Returns:
        De-interleaved array in HF format, float32
    """
    n_heads, d_head = _validate_dims(weights, n_heads, d_head)
    half_d = d_head // 2

    if weights.ndim == 1:
        # 1D bias vector: (n_heads * d_head,)
        reshaped = weights.reshape(n_heads, d_head)
        # GGML: [h, 2*half] where pairs are [-x1, x0, -x3, x2, ...]
        # Reshape to [h, half, 2] -> swap axes 1,2 -> [h, 2, half] -> reshape to [h, 2*half]
        reshaped = reshaped.reshape(n_heads, half_d, 2)
        reshaped = reshaped.swapaxes(1, 2)
        result = reshaped.reshape(n_heads * d_head)
    else:
        # 2D matrix: (n_heads * d_head, d_in)
        reshaped = weights.reshape(n_heads, d_head, weights.shape[1])
        reshaped = reshaped.reshape(n_heads, half_d, 2, weights.shape[1])
        reshaped = reshaped.swapaxes(1, 2)
        result = reshaped.reshape(n_heads * d_head, weights.shape[1])

    return result.astype(np.float32, copy=False)


def interleave_rope(weights: np.ndarray, n_heads: int, d_head: int | None = None) -> np.ndarray:
    """
    Forward transformation from HuggingFace split-half format to GGML interleaved format.

    HF (split-half):    [-x_{d/2:}, x_{:d/2}] per head
    GGML (interleaved): [-x1, x0, -x3, x2, ...] per head

    Args:
        weights: Array of shape (n_heads * d_head, d_in) or (n_heads * d_head,)
        n_heads: Number of attention heads
        d_head: Head dimension (optional, inferred if not provided)

    Returns:
        Interleaved array in GGML format, float32
    """
    n_heads, d_head = _validate_dims(weights, n_heads, d_head)
    half_d = d_head // 2

    if weights.ndim == 1:
        # 1D bias vector: (n_heads * d_head,)
        reshaped = weights.reshape(n_heads, d_head)
        # HF: [h, 2*half] -> reshape to [h, 2, half] -> swap axes 1,2 -> [h, half, 2] -> reshape
        reshaped = reshaped.reshape(n_heads, 2, half_d)
        reshaped = reshaped.swapaxes(1, 2)
        result = reshaped.reshape(n_heads * d_head)
    else:
        # 2D matrix: (n_heads * d_head, d_in)
        reshaped = weights.reshape(n_heads, d_head, weights.shape[1])
        reshaped = reshaped.reshape(n_heads, 2, half_d, weights.shape[1])
        reshaped = reshaped.swapaxes(1, 2)
        result = reshaped.reshape(n_heads * d_head, weights.shape[1])

    return result.astype(np.float32, copy=False)
