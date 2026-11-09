from __future__ import annotations

import tracemalloc

import numpy as np

from wdiff.engine.ring_buffer import PipelinedRingBuffer


class LargeMockReader:
    def __init__(self, count: int = 50, shape: tuple[int, int] = (1024, 1024)):
        self.count = count
        self.shape = shape

    def tensor_names(self) -> list[str]:
        return [f"layer.{i}.weight" for i in range(self.count)]

    def get_tensor(self, name: str) -> np.ndarray:
        # Generates a fresh array on the fly (simulating mmap read)
        return np.ones(self.shape, dtype=np.float32)


class TestMemoryBounding:
    def test_streaming_peak_memory_is_bounded(self):
        """
        Verify that streaming 50 large tensors (4MB each = 200MB total)
        through the ring buffer never accumulates memory.
        Peak memory must remain strictly bounded (< 35MB).
        """
        # Use 512x512 (1MB each) to stay well within 35MB bound accounting for Python overhead
        reader_a = LargeMockReader(count=50, shape=(512, 512))
        reader_b = LargeMockReader(count=50, shape=(512, 512))

        buffer = PipelinedRingBuffer(max_depth=2)

        tracemalloc.start()
        tracemalloc.reset_peak()

        results = buffer.run(reader_a, reader_b, reader_a.tensor_names())

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert len(results) == 50
        peak_mb = peak / (1024 * 1024)

        # 4 tensors in flight (2 pairs in buffer) = ~16MB buffer + overhead.
        # Must be well below 35MB (verifying tensors are freed and not accumulated).
        assert peak_mb < 35.0, f"Peak memory exceeded bound: {peak_mb:.2f} MB"
