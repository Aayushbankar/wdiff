from __future__ import annotations

import heapq
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wdiff.engine.spectral import fast_randomized_svd


@dataclass(frozen=True, slots=True)
class TokenConcept:
    token_id: int
    token: str
    logit: float


@dataclass(frozen=True, slots=True)
class LayerLensResult:
    layer_name: str
    canonical_name: str
    positive_tokens: list[TokenConcept]
    negative_tokens: list[TokenConcept]
    singular_value: float
    energy_ratio: float


class Vocabulary:
    """Token ID to string decoder with multi-format support."""

    def __init__(self, id_to_token: dict[int, str]):
        self._id_to_token = id_to_token

    @classmethod
    def from_gguf_reader(cls, reader) -> Vocabulary | None:
        """Extract vocabulary from GGUF reader fields."""
        if not hasattr(reader, "_reader") or not hasattr(reader._reader, "fields"):
            return None

        fields = reader._reader.fields
        # Look for tokenizer.ggml.tokens field
        for name, field in fields.items():
            if (
                "tokenizer.ggml.tokens" in name.lower()
                and hasattr(field, "data")
                and hasattr(field, "parts")
            ):
                tokens = []
                try:
                    # data contains indices, parts contains the actual token bytes
                    for idx in field.data:
                        if idx < len(field.parts):
                            token_bytes = field.parts[idx]
                            if hasattr(token_bytes, "tobytes"):
                                token_bytes = token_bytes.tobytes()
                            token = token_bytes.decode("utf-8", errors="replace")
                            tokens.append(token)
                    if tokens:
                        return cls({i: t for i, t in enumerate(tokens)})
                except (UnicodeDecodeError, IndexError, AttributeError):
                    pass
        return None

    @classmethod
    def from_json_file(cls, path: str | Path) -> Vocabulary | None:
        """Load vocabulary from tokenizer.json, vocab.json, or token list JSON."""
        path = Path(path)
        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        id_to_token = {}

        # Format 1: tokenizer.json with model.vocab
        if isinstance(data, dict) and "model" in data and "vocab" in data["model"]:
            vocab = data["model"]["vocab"]
            if isinstance(vocab, dict):
                # Invert {token: id} -> {id: token}
                for token, tid in vocab.items():
                    if isinstance(tid, int) and tid >= 0:
                        id_to_token[tid] = token

        # Format 2: tokenizer.json with added_tokens
        if isinstance(data, dict) and "added_tokens" in data:
            for item in data["added_tokens"]:
                if isinstance(item, dict) and "id" in item and "content" in item:
                    tid = item["id"]
                    if isinstance(tid, int) and tid >= 0:
                        id_to_token[tid] = item["content"]

        # Format 3: vocab.json (direct token->id mapping)
        if not id_to_token and isinstance(data, dict):
            # Try to invert assuming {token: id}
            inverted = {}
            for token, tid in data.items():
                if isinstance(tid, int) and tid >= 0:
                    inverted[tid] = token
            if inverted:
                id_to_token = inverted

        # Format 4: JSON list of tokens ["tok0", "tok1", ...]
        if not id_to_token and isinstance(data, list):
            for i, token in enumerate(data):
                if isinstance(token, str):
                    id_to_token[i] = token

        if id_to_token:
            return cls(id_to_token)

        return None

    @classmethod
    def auto_discover(cls, model_path: str | Path, reader=None) -> Vocabulary:
        """Auto-discover vocabulary from model directory or GGUF reader."""
        model_path = Path(model_path)

        # 1. Try GGUF reader fields first
        if reader is not None:
            vocab = cls.from_gguf_reader(reader)
            if vocab:
                return vocab

        # 2. Look for tokenizer files in model directory
        search_dirs = [model_path.parent, model_path]
        if model_path.is_file():
            search_dirs = [model_path.parent]

        for search_dir in search_dirs:
            for filename in ["tokenizer.json", "vocab.json"]:
                vocab_path = search_dir / filename
                vocab = cls.from_json_file(vocab_path)
                if vocab:
                    return vocab

        # 3. Fallback: synthetic vocabulary
        return cls({})

    def get(self, token_id: int) -> str:
        """Get token string, fallback to [token_id] format."""
        if token_id in self._id_to_token:
            token = self._id_to_token[token_id]
            # Strip common BPE/SentencePiece markers
            if token.startswith("Ġ"):
                return " " + token[1:]
            if token.startswith("▁"):
                return " " + token[1:]
            return token
        return f"[{token_id}]"


def project_vector_to_logits_chunked(
    wu_matrix_or_getter: np.ndarray | Callable[[int, int], np.ndarray],
    vocab_size: int,
    u1: np.ndarray,
    top_k: int = 5,
    chunk_size: int = 4096,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """
    Project u1 through W_U in chunks to guarantee <70MB peak RAM.

    Args:
        wu_matrix_or_getter: Either full W_U matrix (vocab_size, d_model) or callable(row_start, row_end) -> chunk
        vocab_size: Total vocabulary size
        u1: Singular vector of shape (d_model,)
        top_k: Number of top tokens to return for each pole
        chunk_size: Rows per chunk (default 4096 keeps memory <70MB)

    Returns:
        (top_positive, top_negative) as sorted lists of (token_id, logit)
    """
    # Normalize u1
    u1_norm = float(np.linalg.norm(u1))
    if u1_norm == 0:
        return [], []
    u1 = u1 / u1_norm

    # Min-heaps for top-K (positive: max logits, negative: min logits = max -logits)
    pos_heap: list[tuple[float, int]] = []  # (logit, token_id) - min-heap for max logits
    neg_heap: list[tuple[float, int]] = []  # (-logit, token_id) - min-heap for min logits

    def push_heaps(token_id: int, logit: float):
        if len(pos_heap) < top_k:
            heapq.heappush(pos_heap, (logit, token_id))
        elif logit > pos_heap[0][0]:
            heapq.heapreplace(pos_heap, (logit, token_id))

        if len(neg_heap) < top_k:
            heapq.heappush(neg_heap, (-logit, token_id))
        elif -logit > neg_heap[0][0]:  # logit more negative
            heapq.heapreplace(neg_heap, (-logit, token_id))

    # Process in chunks
    for chunk_start in range(0, vocab_size, chunk_size):
        chunk_end = min(chunk_start + chunk_size, vocab_size)

        if callable(wu_matrix_or_getter):
            chunk = wu_matrix_or_getter(chunk_start, chunk_end)
        else:
            chunk = wu_matrix_or_getter[chunk_start:chunk_end]

        # Compute logits: chunk @ u1
        logits = chunk @ u1

        # Update heaps
        for i, logit in enumerate(logits):
            token_id = chunk_start + i
            push_heaps(token_id, float(logit))

    # Extract sorted results
    positive = sorted([(tid, logit) for logit, tid in pos_heap], key=lambda x: -x[1])
    negative = sorted([(tid, -logit) for logit, tid in neg_heap], key=lambda x: x[1])

    return positive, negative


def is_down_proj_layer(name: str) -> bool:
    """Check if layer name represents a down-projection (MLP output)."""
    name_lower = name.lower()
    return "ffn_down" in name_lower or "down_proj" in name_lower or "c_proj" in name_lower


def decode_layer_concept(
    diff_matrix: np.ndarray,
    layer_name: str,
    canonical_name: str,
    wu_matrix_or_getter: np.ndarray | Callable[[int, int], np.ndarray],
    vocab_size: int,
    vocab: Vocabulary,
    top_k: int = 5,
) -> LayerLensResult | None:
    """
    Decode semantic concept from down-projection weight delta.

    Args:
        diff_matrix: ΔW = W_B - W_A for the layer
        layer_name: Raw layer name
        canonical_name: Canonical name from alignment
        wu_matrix_or_getter: Unembedding matrix or chunk getter
        vocab_size: Vocabulary size (number of tokens)
        vocab: Vocabulary decoder
        top_k: Number of concept tokens per pole

    Returns:
        LayerLensResult or None if no meaningful concept found
    """
    if diff_matrix.size == 0:
        return None

    # Compute truncated SVD (k=4 for primary singular vectors)
    U, s, Vt = fast_randomized_svd(diff_matrix, k=min(4, min(diff_matrix.shape)))

    if len(s) == 0 or s[0] < 1e-6:
        return None

    # Determine the target hidden dimension from W_U (vocab_size, d_model)
    if hasattr(wu_matrix_or_getter, "shape") and len(wu_matrix_or_getter.shape) >= 2:
        d_target = wu_matrix_or_getter.shape[1]
    elif callable(wu_matrix_or_getter):
        sample = wu_matrix_or_getter(0, 1)
        d_target = sample.shape[1] if sample.ndim >= 2 else sample.shape[0]
    else:
        d_target = diff_matrix.shape[0]

    # Find which singular vector matches d_target
    # In PyTorch down_proj (d_model, d_ffn), the output residual direction is U[:, 0]
    u1 = None
    if U.shape[0] == d_target:
        u1 = U[:, 0]
    elif Vt.shape[1] == d_target:
        u1 = Vt[0, :]
    elif U.shape[0] == diff_matrix.shape[0]:
        # Fallback to output dimension
        u1 = U[:, 0]
    elif Vt.shape[1] == diff_matrix.shape[1]:
        u1 = Vt[0, :]

    if u1 is None or u1.size == 0:
        return None

    # Project through unembedding matrix
    pos_tokens, neg_tokens = project_vector_to_logits_chunked(
        wu_matrix_or_getter, vocab_size, u1, top_k=top_k
    )

    # Resolve token strings
    positive_concepts = [
        TokenConcept(token_id=tid, token=vocab.get(tid), logit=logit) for tid, logit in pos_tokens
    ]
    negative_concepts = [
        TokenConcept(token_id=tid, token=vocab.get(tid), logit=logit) for tid, logit in neg_tokens
    ]

    # Energy ratio: s_0^2 / sum(s_i^2)
    energy_ratio = float(s[0] ** 2 / np.sum(s**2)) if np.sum(s**2) > 0 else 0.0

    return LayerLensResult(
        layer_name=layer_name,
        canonical_name=canonical_name,
        positive_tokens=positive_concepts,
        negative_tokens=negative_concepts,
        singular_value=float(s[0]),
        energy_ratio=energy_ratio,
    )
