from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from typing_extensions import Self


@dataclass(frozen=True, slots=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    byte_offset: int
    byte_size: int
    quantization: str | None = None


@runtime_checkable
class TensorReader(Protocol):
    def tensor_names(self) -> list[str]:
        """Returns ordered list of all tensor keys in the file header."""
        ...

    def get_tensor(self, name: str) -> np.ndarray:
        """Fetches a single tensor, dequantizes if necessary, and returns float32."""
        ...

    def tensor_info(self, name: str) -> TensorInfo:
        """Returns shape, dtype, and on-disk byte footprint without loading weights."""
        ...

    def close(self) -> None:
        """Release any resources (mmap handles, file descriptors)."""
        ...

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
