"""Tests for ShardedSafeTensorsReader."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from wdiff.readers import ShardedSafeTensorsReader, detect_format, get_reader
from wdiff.readers.base import TensorReader


class TestShardedReaderProtocol:
    def test_sharded_reader_implements_protocol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            tensors = {
                "embedding.weight": np.random.randn(64, 256).astype(np.float32),
                "layer.0.attn_q.weight": np.random.randn(256, 256).astype(np.float32),
            }
            shard_path = tmpdir / "model-00001-of-00002.safetensors"
            save_file(tensors, shard_path)

            index_data = {
                "metadata": {"total_size": 1000},
                "weight_map": {
                    "embedding.weight": "model-00001-of-00002.safetensors",
                    "layer.0.attn_q.weight": "model-00001-of-00002.safetensors",
                },
            }
            index_path = tmpdir / "model.safetensors.index.json"
            index_path.write_text(json.dumps(index_data))

            reader = ShardedSafeTensorsReader(str(index_path))
            assert isinstance(reader, TensorReader)
            reader.close()

    def test_sharded_reader_from_index_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            tensors_shard1 = {
                "embedding.weight": np.random.randn(64, 256).astype(np.float32),
                "layer.0.attn_q.weight": np.random.randn(256, 256).astype(np.float32),
            }
            tensors_shard2 = {
                "layer.0.attn_k.weight": np.random.randn(256, 256).astype(np.float32),
                "layer.0.attn_v.weight": np.random.randn(256, 256).astype(np.float32),
            }

            shard1_path = tmpdir / "model-00001-of-00002.safetensors"
            shard2_path = tmpdir / "model-00002-of-00002.safetensors"
            save_file(tensors_shard1, shard1_path)
            save_file(tensors_shard2, shard2_path)

            index_data = {
                "metadata": {"total_size": 1000},
                "weight_map": {
                    "embedding.weight": "model-00001-of-00002.safetensors",
                    "layer.0.attn_q.weight": "model-00001-of-00002.safetensors",
                    "layer.0.attn_k.weight": "model-00002-of-00002.safetensors",
                    "layer.0.attn_v.weight": "model-00002-of-00002.safetensors",
                },
            }
            index_path = tmpdir / "model.safetensors.index.json"
            index_path.write_text(json.dumps(index_data))

            reader = ShardedSafeTensorsReader(str(index_path))
            names = reader.tensor_names()
            assert len(names) == 4
            assert "embedding.weight" in names
            assert "layer.0.attn_q.weight" in names
            assert "layer.0.attn_k.weight" in names
            assert "layer.0.attn_v.weight" in names
            reader.close()

    def test_sharded_reader_from_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            tensors_shard1 = {
                "embedding.weight": np.random.randn(64, 256).astype(np.float32),
                "layer.0.attn_q.weight": np.random.randn(256, 256).astype(np.float32),
            }
            tensors_shard2 = {
                "layer.0.attn_k.weight": np.random.randn(256, 256).astype(np.float32),
                "layer.0.attn_v.weight": np.random.randn(256, 256).astype(np.float32),
            }

            shard1_path = tmpdir / "model-00001-of-00002.safetensors"
            shard2_path = tmpdir / "model-00002-of-00002.safetensors"
            save_file(tensors_shard1, shard1_path)
            save_file(tensors_shard2, shard2_path)

            index_data = {
                "metadata": {"total_size": 1000},
                "weight_map": {
                    "embedding.weight": "model-00001-of-00002.safetensors",
                    "layer.0.attn_q.weight": "model-00001-of-00002.safetensors",
                    "layer.0.attn_k.weight": "model-00002-of-00002.safetensors",
                    "layer.0.attn_v.weight": "model-00002-of-00002.safetensors",
                },
            }
            index_path = tmpdir / "model.safetensors.index.json"
            index_path.write_text(json.dumps(index_data))

            reader = ShardedSafeTensorsReader(str(tmpdir))
            names = reader.tensor_names()
            assert len(names) == 4
            assert "embedding.weight" in names
            assert "layer.0.attn_q.weight" in names
            assert "layer.0.attn_k.weight" in names
            assert "layer.0.attn_v.weight" in names
            reader.close()

    def test_sharded_reader_from_shard_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            tensors_shard1 = {
                "embedding.weight": np.random.randn(64, 256).astype(np.float32),
                "layer.0.attn_q.weight": np.random.randn(256, 256).astype(np.float32),
            }
            tensors_shard2 = {
                "layer.0.attn_k.weight": np.random.randn(256, 256).astype(np.float32),
                "layer.0.attn_v.weight": np.random.randn(256, 256).astype(np.float32),
            }

            shard1_path = tmpdir / "model-00001-of-00002.safetensors"
            shard2_path = tmpdir / "model-00002-of-00002.safetensors"
            save_file(tensors_shard1, shard1_path)
            save_file(tensors_shard2, shard2_path)

            index_data = {
                "metadata": {"total_size": 1000},
                "weight_map": {
                    "embedding.weight": "model-00001-of-00002.safetensors",
                    "layer.0.attn_q.weight": "model-00001-of-00002.safetensors",
                    "layer.0.attn_k.weight": "model-00002-of-00002.safetensors",
                    "layer.0.attn_v.weight": "model-00002-of-00002.safetensors",
                },
            }
            index_path = tmpdir / "model.safetensors.index.json"
            index_path.write_text(json.dumps(index_data))

            reader = ShardedSafeTensorsReader(str(shard1_path))
            names = reader.tensor_names()
            assert len(names) == 4
            reader.close()

    def test_sharded_reader_auto_discover_no_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            tensors_shard1 = {
                "embedding.weight": np.random.randn(64, 256).astype(np.float32),
                "layer.0.attn_q.weight": np.random.randn(256, 256).astype(np.float32),
            }
            tensors_shard2 = {
                "layer.0.attn_k.weight": np.random.randn(256, 256).astype(np.float32),
                "layer.0.attn_v.weight": np.random.randn(256, 256).astype(np.float32),
            }

            shard1_path = tmpdir / "model-00001-of-00002.safetensors"
            shard2_path = tmpdir / "model-00002-of-00002.safetensors"
            save_file(tensors_shard1, shard1_path)
            save_file(tensors_shard2, shard2_path)

            reader = ShardedSafeTensorsReader(str(tmpdir))
            names = reader.tensor_names()
            assert len(names) == 4
            assert "embedding.weight" in names
            assert "layer.0.attn_q.weight" in names
            assert "layer.0.attn_k.weight" in names
            assert "layer.0.attn_v.weight" in names
            reader.close()

    def test_sharded_reader_get_tensor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            embedding = np.random.randn(64, 256).astype(np.float32)
            attn_q = np.random.randn(256, 256).astype(np.float32)

            tensors_shard1 = {
                "embedding.weight": embedding,
                "layer.0.attn_q.weight": attn_q,
            }
            tensors_shard2 = {
                "layer.0.attn_k.weight": np.random.randn(256, 256).astype(np.float32),
                "layer.0.attn_v.weight": np.random.randn(256, 256).astype(np.float32),
            }

            shard1_path = tmpdir / "model-00001-of-00002.safetensors"
            shard2_path = tmpdir / "model-00002-of-00002.safetensors"
            save_file(tensors_shard1, shard1_path)
            save_file(tensors_shard2, shard2_path)

            index_data = {
                "metadata": {"total_size": 1000},
                "weight_map": {
                    "embedding.weight": "model-00001-of-00002.safetensors",
                    "layer.0.attn_q.weight": "model-00001-of-00002.safetensors",
                    "layer.0.attn_k.weight": "model-00002-of-00002.safetensors",
                    "layer.0.attn_v.weight": "model-00002-of-00002.safetensors",
                },
            }
            index_path = tmpdir / "model.safetensors.index.json"
            index_path.write_text(json.dumps(index_data))

            reader = ShardedSafeTensorsReader(str(index_path))

            # Test get_tensor from shard 1
            loaded_embedding = reader.get_tensor("embedding.weight")
            assert loaded_embedding.dtype == np.float32
            assert loaded_embedding.shape == (64, 256)
            np.testing.assert_allclose(loaded_embedding, embedding, rtol=1e-5, atol=1e-6)

            loaded_attn_q = reader.get_tensor("layer.0.attn_q.weight")
            assert loaded_attn_q.dtype == np.float32
            assert loaded_attn_q.shape == (256, 256)
            np.testing.assert_allclose(loaded_attn_q, attn_q, rtol=1e-5, atol=1e-6)

            # Test get_tensor from shard 2
            loaded_attn_k = reader.get_tensor("layer.0.attn_k.weight")
            assert loaded_attn_k.dtype == np.float32
            assert loaded_attn_k.shape == (256, 256)

            loaded_attn_v = reader.get_tensor("layer.0.attn_v.weight")
            assert loaded_attn_v.dtype == np.float32
            assert loaded_attn_v.shape == (256, 256)

            reader.close()

    def test_sharded_reader_tensor_info(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            tensors_shard1 = {
                "embedding.weight": np.random.randn(64, 256).astype(np.float32),
                "layer.0.attn_q.weight": np.random.randn(256, 256).astype(np.float32),
            }

            shard1_path = tmpdir / "model-00001-of-00002.safetensors"
            save_file(tensors_shard1, shard1_path)

            index_data = {
                "metadata": {"total_size": 1000},
                "weight_map": {
                    "embedding.weight": "model-00001-of-00002.safetensors",
                    "layer.0.attn_q.weight": "model-00001-of-00002.safetensors",
                },
            }
            index_path = tmpdir / "model.safetensors.index.json"
            index_path.write_text(json.dumps(index_data))

            reader = ShardedSafeTensorsReader(str(index_path))

            info = reader.tensor_info("embedding.weight")
            assert info.name == "embedding.weight"
            assert info.shape == (64, 256)
            assert info.dtype == np.float32
            assert info.quantization is None
            assert info.byte_size > 0
            assert isinstance(info.byte_offset, int)

            info2 = reader.tensor_info("layer.0.attn_q.weight")
            assert info2.shape == (256, 256)
            assert info2.dtype == np.float32

            reader.close()

    def test_lru_handle_pool_rotation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            tensors_shard1 = {"embedding.weight": np.random.randn(64, 256).astype(np.float32)}
            tensors_shard2 = {"layer.0.attn_q.weight": np.random.randn(256, 256).astype(np.float32)}
            tensors_shard3 = {"layer.0.attn_k.weight": np.random.randn(256, 256).astype(np.float32)}

            shard1_path = tmpdir / "model-00001-of-00003.safetensors"
            shard2_path = tmpdir / "model-00002-of-00003.safetensors"
            shard3_path = tmpdir / "model-00003-of-00003.safetensors"
            save_file(tensors_shard1, shard1_path)
            save_file(tensors_shard2, shard2_path)
            save_file(tensors_shard3, shard3_path)

            index_data = {
                "metadata": {"total_size": 1000},
                "weight_map": {
                    "embedding.weight": "model-00001-of-00003.safetensors",
                    "layer.0.attn_q.weight": "model-00002-of-00003.safetensors",
                    "layer.0.attn_k.weight": "model-00003-of-00003.safetensors",
                },
            }
            index_path = tmpdir / "model.safetensors.index.json"
            index_path.write_text(json.dumps(index_data))

            reader = ShardedSafeTensorsReader(str(index_path), max_open_handles=2)

            assert len(reader._open_shards) == 0

            # Access shard 1
            _ = reader.get_tensor("embedding.weight")
            assert len(reader._open_shards) == 1
            assert "model-00001-of-00003.safetensors" in reader._open_shards

            # Access shard 2
            _ = reader.get_tensor("layer.0.attn_q.weight")
            assert len(reader._open_shards) == 2
            assert "model-00002-of-00003.safetensors" in reader._open_shards

            # Access shard 3 - should evict shard 1
            _ = reader.get_tensor("layer.0.attn_k.weight")
            assert len(reader._open_shards) == 2
            assert "model-00003-of-00003.safetensors" in reader._open_shards
            assert "model-00002-of-00003.safetensors" in reader._open_shards
            assert "model-00001-of-00003.safetensors" not in reader._open_shards

            # Access shard 2 again - should be MRU
            _ = reader.get_tensor("layer.0.attn_q.weight")
            assert len(reader._open_shards) == 2

            # Access shard 1 again - should evict shard 3
            _ = reader.get_tensor("embedding.weight")
            assert len(reader._open_shards) == 2
            assert "model-00001-of-00003.safetensors" in reader._open_shards
            assert "model-00002-of-00003.safetensors" in reader._open_shards
            assert "model-00003-of-00003.safetensors" not in reader._open_shards

            reader.close()

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            tensors = {
                "embedding.weight": np.random.randn(64, 256).astype(np.float32),
                "layer.0.attn_q.weight": np.random.randn(256, 256).astype(np.float32),
            }
            shard_path = tmpdir / "model-00001-of-00002.safetensors"
            save_file(tensors, shard_path)

            index_data = {
                "metadata": {"total_size": 1000},
                "weight_map": {
                    "embedding.weight": "model-00001-of-00002.safetensors",
                    "layer.0.attn_q.weight": "model-00001-of-00002.safetensors",
                },
            }
            index_path = tmpdir / "model.safetensors.index.json"
            index_path.write_text(json.dumps(index_data))

            with ShardedSafeTensorsReader(str(index_path)) as reader:
                names = reader.tensor_names()
                assert len(names) == 2

    def test_close_clears_handles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            tensors = {
                "embedding.weight": np.random.randn(64, 256).astype(np.float32),
            }
            shard_path = tmpdir / "model-00001-of-00001.safetensors"
            save_file(tensors, shard_path)

            index_data = {
                "metadata": {"total_size": 1000},
                "weight_map": {"embedding.weight": "model-00001-of-00001.safetensors"},
            }
            index_path = tmpdir / "model.safetensors.index.json"
            index_path.write_text(json.dumps(index_data))

            reader = ShardedSafeTensorsReader(str(index_path))
            _ = reader.get_tensor("embedding.weight")
            assert len(reader._open_shards) == 1

            reader.close()
            assert len(reader._open_shards) == 0

    def test_key_error_for_missing_tensor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            tensors = {"embedding.weight": np.random.randn(64, 256).astype(np.float32)}
            shard_path = tmpdir / "model-00001-of-00001.safetensors"
            save_file(tensors, shard_path)

            index_data = {
                "metadata": {"total_size": 1000},
                "weight_map": {"embedding.weight": "model-00001-of-00001.safetensors"},
            }
            index_path = tmpdir / "model.safetensors.index.json"
            index_path.write_text(json.dumps(index_data))

            reader = ShardedSafeTensorsReader(str(index_path))
            with pytest.raises(KeyError):
                reader.get_tensor("nonexistent.weight")
            with pytest.raises(KeyError):
                reader.tensor_info("nonexistent.weight")
            reader.close()


class TestShardedDetection:
    def test_auto_detect_sharded_from_index_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            index_path = tmpdir / "model.safetensors.index.json"
            index_path.write_text('{"weight_map": {}}')
            fmt = detect_format(str(index_path))
            assert fmt == "sharded_safetensors"

    def test_auto_detect_sharded_from_directory_with_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            index_path = tmpdir / "model.safetensors.index.json"
            index_path.write_text('{"weight_map": {}}')
            fmt = detect_format(str(tmpdir))
            assert fmt == "sharded_safetensors"

    def test_auto_detect_sharded_from_shard_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            shard_path = tmpdir / "model-00001-of-00002.safetensors"
            shard_path.write_text("dummy")
            index_path = tmpdir / "model.safetensors.index.json"
            index_path.write_text('{"weight_map": {"a": "model-00001-of-00002.safetensors"}}')
            fmt = detect_format(str(shard_path))
            assert fmt == "sharded_safetensors"

    def test_standalone_safetensors_not_hijacked_by_sibling_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            shard_path = tmpdir / "model-00001-of-00002.safetensors"
            shard_path.write_text("dummy")
            standalone_path = tmpdir / "adapter.safetensors"
            standalone_path.write_text("dummy")
            index_path = tmpdir / "model.safetensors.index.json"
            index_path.write_text('{"weight_map": {"a": "model-00001-of-00002.safetensors"}}')
            # Standalone file should be regular safetensors, not sharded!
            fmt = detect_format(str(standalone_path))
            assert fmt == "safetensors"

    def test_auto_detect_sharded_from_directory_no_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            shard_path = tmpdir / "model-00001-of-00002.safetensors"
            shard_path.write_text("dummy")
            fmt = detect_format(str(tmpdir))
            assert fmt == "sharded_safetensors"

    def test_get_reader_routes_to_sharded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            tensors = {"embedding.weight": np.random.randn(64, 256).astype(np.float32)}
            shard_path = tmpdir / "model-00001-of-00001.safetensors"
            save_file(tensors, shard_path)

            index_data = {
                "metadata": {"total_size": 1000},
                "weight_map": {"embedding.weight": "model-00001-of-00001.safetensors"},
            }
            index_path = tmpdir / "model.safetensors.index.json"
            index_path.write_text(json.dumps(index_data))

            reader = get_reader(str(index_path))
            assert isinstance(reader, ShardedSafeTensorsReader)
            reader.close()

            reader2 = get_reader(str(tmpdir))
            assert isinstance(reader2, ShardedSafeTensorsReader)
            reader2.close()

            reader3 = get_reader(str(shard_path))
            assert isinstance(reader3, ShardedSafeTensorsReader)
            reader3.close()


class TestShardedIntegration:
    def test_sharded_integration_with_inspect(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            tensors = {
                "embedding.weight": np.random.randn(64, 256).astype(np.float32),
                "layer.0.attn_q.weight": np.random.randn(256, 256).astype(np.float32),
            }
            shard_path = tmpdir / "model-00001-of-00001.safetensors"
            save_file(tensors, shard_path)

            index_data = {
                "metadata": {"total_size": 1000},
                "weight_map": {
                    "embedding.weight": "model-00001-of-00001.safetensors",
                    "layer.0.attn_q.weight": "model-00001-of-00001.safetensors",
                },
            }
            index_path = tmpdir / "model.safetensors.index.json"
            index_path.write_text(json.dumps(index_data))

            from wdiff.engine.inspect import inspect_model

            result = inspect_model(str(index_path))
            assert result.summary.tensor_count == 2
            assert result.summary.total_params > 0

    def test_sharded_integration_with_diff(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            base_tensors = {
                "embedding.weight": np.random.randn(64, 256).astype(np.float32),
                "layer.0.attn_q.weight": np.random.randn(256, 256).astype(np.float32),
            }
            modified_tensors = {
                k: v + np.random.randn(*v.shape).astype(np.float32) * 0.01
                for k, v in base_tensors.items()
            }

            shard_path_a = tmpdir / "model_a-00001-of-00001.safetensors"
            shard_path_b = tmpdir / "model_b-00001-of-00001.safetensors"
            save_file(base_tensors, shard_path_a)
            save_file(modified_tensors, shard_path_b)

            index_data_a = {
                "metadata": {"total_size": 1000},
                "weight_map": {
                    "embedding.weight": "model_a-00001-of-00001.safetensors",
                    "layer.0.attn_q.weight": "model_a-00001-of-00001.safetensors",
                },
            }
            index_data_b = {
                "metadata": {"total_size": 1000},
                "weight_map": {
                    "embedding.weight": "model_b-00001-of-00001.safetensors",
                    "layer.0.attn_q.weight": "model_b-00001-of-00001.safetensors",
                },
            }
            index_path_a = tmpdir / "model_a.safetensors.index.json"
            index_path_b = tmpdir / "model_b.safetensors.index.json"
            index_path_a.write_text(json.dumps(index_data_a))
            index_path_b.write_text(json.dumps(index_data_b))

            from wdiff.engine.diff import diff_models

            result = diff_models(str(index_path_a), str(index_path_b))
            assert result.summary.matched_count == 2
            assert result.summary.mismatched_shape_count == 0
            assert result.summary.only_in_a_count == 0
            assert result.summary.only_in_b_count == 0

    def test_sharded_cross_format_with_single_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            base_tensors = {
                "embedding.weight": np.random.randn(64, 256).astype(np.float32),
                "layer.0.attn_q.weight": np.random.randn(256, 256).astype(np.float32),
            }

            shard_path = tmpdir / "model-00001-of-00001.safetensors"
            save_file(base_tensors, shard_path)

            index_data = {
                "metadata": {"total_size": 1000},
                "weight_map": {
                    "embedding.weight": "model-00001-of-00001.safetensors",
                    "layer.0.attn_q.weight": "model-00001-of-00001.safetensors",
                },
            }
            index_path = tmpdir / "model.safetensors.index.json"
            index_path.write_text(json.dumps(index_data))

            from wdiff.engine.diff import diff_models

            result = diff_models(str(index_path), str(shard_path))
            assert result.summary.matched_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
