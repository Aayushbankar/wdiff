from pathlib import Path

import numpy as np
import pytest

from wdiff.readers import detect_format, get_reader
from wdiff.readers.base import TensorInfo, TensorReader

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


class TestDetectFormat:
    def test_safetensors(self):
        assert detect_format("model.safetensors") == "safetensors"
        assert detect_format("path/to/model.safetensors") == "safetensors"

    def test_gguf(self):
        assert detect_format("model.gguf") == "gguf"
        assert detect_format("path/to/model.gguf") == "gguf"

    def test_invalid(self):
        with pytest.raises(ValueError):
            detect_format("model.pt")
        with pytest.raises(ValueError):
            detect_format("model.bin")


class TestSafeTensorsReader:
    @pytest.fixture
    def reader(self):
        with get_reader(f"{FIXTURE_DIR}/model_a.safetensors") as r:
            yield r

    def test_implements_protocol(self, reader):
        assert isinstance(reader, TensorReader)

    def test_tensor_names(self, reader):
        names = reader.tensor_names()
        assert len(names) == 9
        assert "embedding.weight" in names
        assert "layer.0.attn_q.weight" in names

    def test_get_tensor_returns_float32(self, reader):
        tensor = reader.get_tensor("embedding.weight")
        assert tensor.dtype == np.float32
        assert tensor.shape == (64, 256)

    def test_tensor_info(self, reader):
        info = reader.tensor_info("embedding.weight")
        assert isinstance(info, TensorInfo)
        assert info.name == "embedding.weight"
        assert info.shape == (64, 256)
        assert info.dtype == np.float32
        assert info.quantization is None
        assert info.byte_size > 0
        assert isinstance(info.byte_offset, int)

    def test_context_manager(self):
        with get_reader(f"{FIXTURE_DIR}/model_a.safetensors") as reader:
            names = reader.tensor_names()
            assert len(names) == 9


class TestGGUFReader:
    @pytest.fixture
    def reader_f32(self):
        with get_reader(f"{FIXTURE_DIR}/model_a.gguf") as r:
            yield r

    @pytest.fixture
    def reader_q8(self):
        with get_reader(f"{FIXTURE_DIR}/model_a_q8_0.gguf") as r:
            yield r

    def test_implements_protocol(self, reader_f32):
        assert isinstance(reader_f32, TensorReader)

    def test_tensor_names(self, reader_f32):
        names = reader_f32.tensor_names()
        assert len(names) == 9
        assert "embedding.weight" in names

    def test_get_tensor_f32_returns_float32(self, reader_f32):
        tensor = reader_f32.get_tensor("embedding.weight")
        assert tensor.dtype == np.float32
        assert tensor.shape == (64, 256)

    def test_get_tensor_q8_dequantizes_to_float32(self, reader_q8):
        tensor = reader_q8.get_tensor("embedding.weight")
        assert tensor.dtype == np.float32
        assert tensor.shape == (64, 256)

    def test_tensor_info_f32(self, reader_f32):
        info = reader_f32.tensor_info("embedding.weight")
        assert isinstance(info, TensorInfo)
        assert info.name == "embedding.weight"
        assert info.shape == (64, 256)
        assert info.quantization == "F32"

    def test_tensor_info_q8(self, reader_q8):
        info = reader_q8.tensor_info("embedding.weight")
        assert info.quantization == "Q8_0"
        assert info.shape == (64, 256)
        assert isinstance(info.byte_offset, int)

    def test_context_manager(self):
        with get_reader(f"{FIXTURE_DIR}/model_a.gguf") as reader:
            names = reader.tensor_names()
            assert len(names) == 9


class TestCrossFormatConsistency:
    def test_same_tensors_different_formats(self):
        with (
            get_reader(f"{FIXTURE_DIR}/model_a.safetensors") as r1,
            get_reader(f"{FIXTURE_DIR}/model_a.gguf") as r2,
        ):
            names1 = set(r1.tensor_names())
            names2 = set(r2.tensor_names())
            assert names1 == names2

            for name in names1:
                t1 = r1.get_tensor(name)
                t2 = r2.get_tensor(name)
                assert t1.shape == t2.shape
                assert t1.dtype == t2.dtype == np.float32
                np.testing.assert_allclose(t1, t2, rtol=1e-5, atol=1e-6)
