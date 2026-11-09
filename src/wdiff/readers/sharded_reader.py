from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from safetensors import safe_open
from typing_extensions import Self

from .base import TensorInfo
from .safetensors_reader import SafeTensorsReader


class ShardedSafeTensorsReader:
    def __init__(self, path: str | Path, max_open_handles: int = 2) -> None:
        self._path = Path(path)
        self._max_open_handles = max_open_handles
        self._open_shards: OrderedDict[str, SafeTensorsReader] = OrderedDict()
        self._weight_map: dict[str, str] = {}
        self._shard_files: dict[str, Path] = {}
        self._tensor_names_cache: list[str] | None = None
        self._index_path: Path | None = None

        self._resolve_index_and_shards()

    def _resolve_index_and_shards(self) -> None:
        path = self._path

        if path.is_file():
            if path.name.endswith(".safetensors.index.json"):
                self._index_path = path
            elif path.suffix == ".safetensors":
                # Find any .safetensors.index.json in the same directory
                index_files = list(path.parent.glob("*.safetensors.index.json"))
                if index_files:
                    self._index_path = index_files[0]
                else:
                    raise ValueError(
                        f"No .safetensors.index.json found alongside {path}. "
                        "Sharded reader requires an index file."
                    )
            else:
                raise ValueError(f"Unsupported file: {path}")
        elif path.is_dir():
            index_files = list(path.glob("*.safetensors.index.json"))
            if index_files:
                self._index_path = index_files[0]
            else:
                shard_files = sorted(path.glob("*.safetensors"))
                if not shard_files:
                    raise ValueError(f"No .safetensors files found in directory: {path}")
                self._index_path = None
                self._build_dynamic_weight_map(shard_files)
                return
        else:
            raise ValueError(f"Path does not exist: {path}")

        if self._index_path:
            self._parse_index()

    def _build_dynamic_weight_map(self, shard_files: list[Path]) -> None:
        for shard_file in shard_files:
            with safe_open(shard_file, framework="numpy", device="cpu") as f:
                for name in f.keys():  # noqa: SIM118 - SafeOpen is not iterable directly
                    self._weight_map[name] = shard_file.name
                    self._shard_files[shard_file.name] = shard_file

    def _parse_index(self) -> None:
        if not self._index_path:
            return

        with open(self._index_path) as f:
            index_data = json.load(f)

        self._weight_map = index_data.get("weight_map", {})

        index_dir = self._index_path.parent
        for shard_name in set(self._weight_map.values()):
            shard_path = index_dir / shard_name
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard file not found: {shard_path}")
            self._shard_files[shard_name] = shard_path

    def _get_shard_reader(self, shard_name: str) -> SafeTensorsReader:
        if shard_name in self._open_shards:
            self._open_shards.move_to_end(shard_name)
            return self._open_shards[shard_name]

        if len(self._open_shards) >= self._max_open_handles:
            _oldest_shard_name, oldest_reader = self._open_shards.popitem(last=False)
            oldest_reader.close()

        shard_path = self._shard_files[shard_name]
        reader = SafeTensorsReader(str(shard_path))
        self._open_shards[shard_name] = reader
        return reader

    def tensor_names(self) -> list[str]:
        if self._tensor_names_cache is None:
            self._tensor_names_cache = list(self._weight_map.keys())
        return self._tensor_names_cache

    def get_tensor(self, name: str) -> np.ndarray:
        if name not in self._weight_map:
            raise KeyError(f"Tensor '{name}' not found in sharded model")

        shard_name = self._weight_map[name]
        reader = self._get_shard_reader(shard_name)
        tensor = reader.get_tensor(name)
        return tensor

    def tensor_info(self, name: str) -> TensorInfo:
        if name not in self._weight_map:
            raise KeyError(f"Tensor '{name}' not found in sharded model")

        shard_name = self._weight_map[name]
        reader = self._get_shard_reader(shard_name)
        return reader.tensor_info(name)

    def close(self) -> None:
        for reader in self._open_shards.values():
            reader.close()
        self._open_shards.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
