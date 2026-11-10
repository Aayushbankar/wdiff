from __future__ import annotations

import contextlib

import gguf
import numpy as np
from gguf import GGUFReader as _GGUFReader
from gguf.quants import GGMLQuantizationType, dequantize
from typing_extensions import Self

from .base import TensorInfo

_QUANT_TYPE_MAP: dict[int, str] = {
    int(GGMLQuantizationType.F32): "F32",
    int(GGMLQuantizationType.F16): "F16",
    int(GGMLQuantizationType.BF16): "BF16",
    int(GGMLQuantizationType.F64): "F64",
    int(GGMLQuantizationType.Q4_0): "Q4_0",
    int(GGMLQuantizationType.Q4_1): "Q4_1",
    int(GGMLQuantizationType.Q5_0): "Q5_0",
    int(GGMLQuantizationType.Q5_1): "Q5_1",
    int(GGMLQuantizationType.Q8_0): "Q8_0",
    int(GGMLQuantizationType.Q8_1): "Q8_1",
    int(GGMLQuantizationType.Q8_K): "Q8_K",
    int(GGMLQuantizationType.Q2_K): "Q2_K",
    int(GGMLQuantizationType.Q3_K): "Q3_K",
    int(GGMLQuantizationType.Q4_K): "Q4_K",
    int(GGMLQuantizationType.Q5_K): "Q5_K",
    int(GGMLQuantizationType.Q6_K): "Q6_K",
    int(GGMLQuantizationType.IQ1_S): "IQ1_S",
    int(GGMLQuantizationType.IQ1_M): "IQ1_M",
    int(GGMLQuantizationType.IQ2_S): "IQ2_S",
    int(GGMLQuantizationType.IQ2_XS): "IQ2_XS",
    int(GGMLQuantizationType.IQ2_XXS): "IQ2_XXS",
    int(GGMLQuantizationType.IQ3_S): "IQ3_S",
    int(GGMLQuantizationType.IQ3_XXS): "IQ3_XXS",
    int(GGMLQuantizationType.IQ4_NL): "IQ4_NL",
    int(GGMLQuantizationType.IQ4_XS): "IQ4_XS",
    int(GGMLQuantizationType.Q1_0): "Q1_0",
    int(GGMLQuantizationType.TQ1_0): "TQ1_0",
    int(GGMLQuantizationType.TQ2_0): "TQ2_0",
    int(GGMLQuantizationType.MXFP4): "MXFP4",
    int(GGMLQuantizationType.NVFP4): "NVFP4",
}


def _logical_shape(tensor_info: gguf.TensorInfo) -> tuple[int, ...]:
    """GGUF stores shape as [rows, cols] (Fortran order), but NumPy uses C-order (rows, cols).
    The logical shape for a 2D tensor is (cols, rows) = reversed(tensor_info.shape)."""
    return tuple(int(x) for x in reversed(tensor_info.shape))


class GGUFReader:
    def __init__(self, path: str) -> None:
        self._path = path
        self._reader = _GGUFReader(path)
        self._tensor_map: dict[str, gguf.TensorInfo] = {
            tensor.name: tensor for tensor in self._reader.tensors
        }
        self._head_count: int | None = None
        self._head_count_kv: int | None = None
        self._extract_attention_metadata()

    def _extract_attention_metadata(self) -> None:
        """Extract attention metadata from GGUF fields (head_count, head_count_kv, key_length)."""
        if not hasattr(self._reader, "fields") or not self._reader.fields:
            return

        # fields is an OrderedDict of name -> ReaderField
        for name, field in self._reader.fields.items():
            name_lower = name.lower()
            # Look for attention.head_count, attention.head_count_kv, attention.key_length
            # The value is in field.parts[-1] (the last part)
            if name_lower.endswith("attention.head_count") and not name_lower.endswith("kv"):
                try:
                    if hasattr(field, "parts") and len(field.parts) > 0:
                        self._head_count = int(field.parts[-1][0])
                except (ValueError, TypeError, IndexError):
                    pass
            elif name_lower.endswith("attention.head_count_kv"):
                try:
                    if hasattr(field, "parts") and len(field.parts) > 0:
                        self._head_count_kv = int(field.parts[-1][0])
                except (ValueError, TypeError, IndexError):
                    pass

    def get_metadata(self, key_suffix: str):
        """
        Get metadata value by key suffix.

        Args:
            key_suffix: Suffix to match (e.g., "attention.head_count", "attention.key_length")

        Returns:
            The scalar value if found, None otherwise
        """
        if not hasattr(self._reader, "fields") or not self._reader.fields:
            return None

        key_suffix_lower = key_suffix.lower()
        for name, field in self._reader.fields.items():
            if name.lower().endswith(key_suffix_lower):
                try:
                    if hasattr(field, "parts") and len(field.parts) > 0:
                        return int(field.parts[-1][0])
                except (ValueError, TypeError, IndexError):
                    pass
        return None

    @property
    def head_count(self) -> int | None:
        """Number of attention query heads (n_heads)."""
        return self._head_count

    @property
    def head_count_kv(self) -> int | None:
        """Number of attention key/value heads (n_kv_heads)."""
        return self._head_count_kv

    def tensor_names(self) -> list[str]:
        return list(self._tensor_map.keys())

    def get_tensor(self, name: str) -> np.ndarray:
        if name not in self._tensor_map:
            raise KeyError(f"Tensor '{name}' not found in GGUF file")

        tensor_info = self._tensor_map[name]
        data = tensor_info.data

        quant_type = int(tensor_info.tensor_type)

        if quant_type in (
            int(GGMLQuantizationType.F32),
            int(GGMLQuantizationType.F16),
            int(GGMLQuantizationType.BF16),
        ):
            if data.dtype != np.float32:
                data = data.astype(np.float32, copy=False)
            return data

        dequantized = dequantize(data, quant_type)
        if dequantized.dtype != np.float32:
            dequantized = dequantized.astype(np.float32, copy=False)
        return dequantized

    def tensor_info(self, name: str) -> TensorInfo:
        if name not in self._tensor_map:
            raise KeyError(f"Tensor '{name}' not found in GGUF file")

        tensor_info = self._tensor_map[name]
        quant_type = int(tensor_info.tensor_type)
        quant_name = _QUANT_TYPE_MAP.get(quant_type, f"UNKNOWN_{quant_type}")

        return TensorInfo(
            name=name,
            shape=_logical_shape(tensor_info),
            dtype=tensor_info.data.dtype,
            byte_offset=tensor_info.data_offset,
            byte_size=tensor_info.data.nbytes,
            quantization=quant_name,
        )

    def close(self) -> None:
        if hasattr(self, "_reader") and self._reader is not None:
            if hasattr(self._reader, "data") and hasattr(self._reader.data, "_mmap"):
                with contextlib.suppress(Exception):
                    self._reader.data._mmap.close()
            self._reader = None
            self._tensor_map.clear()
            import gc

            gc.collect()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
