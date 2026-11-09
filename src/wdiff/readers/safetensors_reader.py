from __future__ import annotations

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file as load_safetensors
from typing_extensions import Self

from .base import TensorInfo

_SAFE_DTYPE_MAP = {
    "F32": np.dtype("float32"),
    "F16": np.dtype("float16"),
    "BF16": np.dtype("float16"),  # bfloat16 not in numpy, use float16 as proxy
    "I64": np.dtype("int64"),
    "I32": np.dtype("int32"),
    "I16": np.dtype("int16"),
    "I8": np.dtype("int8"),
    "U8": np.dtype("uint8"),
    "BOOL": np.dtype("bool"),
}


class SafeTensorsReader:
    def __init__(self, path: str) -> None:
        self._path = path
        self._file = safe_open(path, framework="numpy", device="cpu")
        self._metadata = self._file.metadata()

    def tensor_names(self) -> list[str]:
        return list(self._file.keys())

    def get_tensor(self, name: str) -> np.ndarray:
        tensor = self._file.get_tensor(name)
        if tensor.dtype != np.float32:
            tensor = tensor.astype(np.float32, copy=False)
        return tensor

    def tensor_info(self, name: str) -> TensorInfo:
        slice_info = self._file.get_slice(name)
        shape = tuple(slice_info.get_shape())
        dtype_str = slice_info.get_dtype()
        dtype = _SAFE_DTYPE_MAP.get(dtype_str, np.dtype("float32"))
        itemsize = dtype.itemsize
        byte_size = int(np.prod(shape) * itemsize)
        byte_offset = 0  # slice_info[0] returns first row data, not offset

        return TensorInfo(
            name=name,
            shape=shape,
            dtype=dtype,
            byte_offset=byte_offset,
            byte_size=byte_size,
            quantization=None,
        )

    def close(self) -> None:
        if hasattr(self, "_file") and self._file is not None:
            self._file.__exit__(None, None, None)
            self._file = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def load_safetensors_dict(path: str) -> dict[str, np.ndarray]:
    data = load_safetensors(path)
    return {
        k: v.astype(np.float32, copy=False) if v.dtype != np.float32 else v for k, v in data.items()
    }
