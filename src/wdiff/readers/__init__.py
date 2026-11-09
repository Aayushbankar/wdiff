from __future__ import annotations

from pathlib import Path
from typing import Literal

from .base import TensorInfo, TensorReader
from .gguf_reader import GGUFReader
from .naming import (
    AlignmentReport,
    Architecture,
    CanonicalTensor,
    TensorNameNormalizer,
    align_tensor_names,
)
from .safetensors_reader import SafeTensorsReader, load_safetensors_dict
from .sharded_reader import ShardedSafeTensorsReader

__all__ = [
    "AlignmentReport",
    "Architecture",
    "CanonicalTensor",
    "GGUFReader",
    "SafeTensorsReader",
    "ShardedSafeTensorsReader",
    "TensorInfo",
    "TensorNameNormalizer",
    "TensorReader",
    "align_tensor_names",
    "get_reader",
    "load_safetensors_dict",
]


FormatType = Literal["safetensors", "gguf", "sharded_safetensors"]


def _is_sharded_safetensors(path: Path) -> bool:
    if path.is_file():
        # Check for index file (any name ending with .safetensors.index.json)
        if path.name.endswith(".safetensors.index.json"):
            return True
        # Check for shard file with index alongside
        if path.suffix == ".safetensors":
            # Look for any .safetensors.index.json in the same directory
            for idx in path.parent.glob("*.safetensors.index.json"):
                try:
                    import json

                    with open(idx) as f:
                        data = json.load(f)
                    weight_map = data.get("weight_map", {})
                    if path.name in set(weight_map.values()):
                        return True
                except (OSError, json.JSONDecodeError, KeyError):
                    return True
    elif path.is_dir():
        # Check for any index file in directory
        for _idx in path.glob("*.safetensors.index.json"):
            return True
        # Check for shard files without index
        shard_files = list(path.glob("*.safetensors"))
        if shard_files:
            return True
    return False


def detect_format(path: str) -> FormatType:
    p = Path(path)
    if _is_sharded_safetensors(p):
        return "sharded_safetensors"
    suffix = p.suffix.lower()
    if suffix in (".safetensors",):
        return "safetensors"
    if suffix in (".gguf",):
        return "gguf"
    raise ValueError(
        f"Unsupported file format: {suffix}. Supported: .safetensors, .gguf, .safetensors.index.json"
    )


def get_reader(path: str) -> TensorReader:
    fmt = detect_format(path)
    if fmt == "safetensors":
        return SafeTensorsReader(path)
    if fmt == "gguf":
        return GGUFReader(path)
    if fmt == "sharded_safetensors":
        return ShardedSafeTensorsReader(path)
    raise ValueError(f"Unsupported format: {fmt}")
