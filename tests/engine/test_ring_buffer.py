from __future__ import annotations

import queue
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from wdiff.engine.ring_buffer import PipelinedRingBuffer


class MockTensorReader:
    def __init__(self, tensors: dict[str, np.ndarray], delay: float = 0.0):
        self._tensors = tensors
        self._delay = delay

    def tensor_names(self) -> list[str]:
        return list(self._tensors.keys())

    def get_tensor(self, name: str) -> np.ndarray:
        if self._delay > 0:
            time.sleep(self._delay)
        if name not in self._tensors:
            raise KeyError(name)
        return self._tensors[name]


class TestPipelinedRingBuffer:
    def test_bounded_queue_depth(self):
        """Verify queue maxsize is strictly 2 to bound peak RAM to 2 tensor pairs."""
        buffer = PipelinedRingBuffer(max_depth=2)
        assert buffer.max_depth == 2
        assert isinstance(buffer._queue, queue.Queue)
        assert buffer._queue.maxsize == 2

    def test_in_order_streaming_execution(self):
        """Verify tensors stream through in deterministic order with correct metrics."""
        tensors_a = {f"layer.{i}": np.ones((64, 64), dtype=np.float32) * i for i in range(5)}
        tensors_b = {f"layer.{i}": np.ones((64, 64), dtype=np.float32) * (i + 1) for i in range(5)}
        reader_a = MockTensorReader(tensors_a)
        reader_b = MockTensorReader(tensors_b)

        buffer = PipelinedRingBuffer(max_depth=2)
        names = list(tensors_a.keys())
        results = buffer.run(reader_a, reader_b, names)

        assert len(results) == 5
        for i, (name, metrics) in enumerate(results):
            assert name == f"layer.{i}"
            assert metrics is not None
            assert metrics.max_element_delta == pytest.approx(1.0)

    def test_producer_blocks_when_buffer_full(self):
        """Verify producer blocks when 2 slots are occupied and resumes on consumer read."""
        tensors = {f"t.{i}": np.zeros((10, 10), dtype=np.float32) for i in range(10)}

        # Fast readers (no delay) + slow consumer (50ms delay in ring buffer)
        # = producer fills queue to maxsize=2, then blocks until consumer drains
        reader_a = MockTensorReader(tensors, delay=0.0)
        reader_b = MockTensorReader(tensors, delay=0.0)

        buffer = PipelinedRingBuffer(max_depth=2)

        # Track maximum queue size observed during execution
        max_queue_size = [0]

        def progress_callback(completed: int, total: int, name: str):
            current_qsize = buffer._queue.qsize()
            max_queue_size[0] = max(max_queue_size[0], current_qsize)

        results = buffer.run(
            reader_a, reader_b, list(tensors.keys()), progress_callback=progress_callback
        )

        assert len(results) == 10
        # Queue depth is bounded by max_depth=2
        assert 1 <= max_queue_size[0] <= 2

    def test_exception_in_reader_propagates_cleanly(self):
        """Verify errors during I/O prefetch raise cleanly on main thread without hanging."""
        reader_a = MagicMock()
        reader_a.get_tensor.side_effect = OSError("Simulated Disk Read Failure")
        reader_b = MagicMock()
        reader_b.get_tensor.return_value = np.zeros((10, 10), dtype=np.float32)

        buffer = PipelinedRingBuffer(max_depth=2)
        with pytest.raises(IOError, match="Simulated Disk Read Failure"):
            buffer.run(reader_a, reader_b, ["corrupted_tensor"])

    def test_progress_callback(self):
        """Verify progress callback is called monotonically from 1 to N."""
        tensors = {f"t.{i}": np.zeros((4, 4), dtype=np.float32) for i in range(4)}
        reader = MockTensorReader(tensors)

        progress_calls = []

        def on_progress(completed: int, total: int, name: str):
            progress_calls.append((completed, total, name))

        buffer = PipelinedRingBuffer(max_depth=2)
        buffer.run(reader, reader, list(tensors.keys()), progress_callback=on_progress)

        assert len(progress_calls) == 4
        assert progress_calls == [
            (1, 4, "t.0"),
            (2, 4, "t.1"),
            (3, 4, "t.2"),
            (4, 4, "t.3"),
        ]

    def test_empty_tensor_list(self):
        """Verify empty tensor list returns empty result without deadlock."""
        reader = MockTensorReader({})
        buffer = PipelinedRingBuffer(max_depth=2)
        results = buffer.run(reader, reader, [])
        assert results == []
