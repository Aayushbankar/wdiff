from __future__ import annotations

import queue
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from wdiff.engine.metrics import MetricsBundle, compute_all_metrics


@dataclass(slots=True)
class RingBufferSlot:
    name: str
    tensor_a: np.ndarray
    tensor_b: np.ndarray


class _Sentinel:
    pass


class _ExceptionWrapper:
    def __init__(self, exc: BaseException):
        self.exc = exc
        self.traceback = traceback.format_exc()


class PipelinedRingBuffer:
    def __init__(self, max_depth: int = 2) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        self.max_depth = max_depth
        self._queue: queue.Queue[RingBufferSlot | _Sentinel | _ExceptionWrapper] = queue.Queue(
            maxsize=max_depth
        )

    def run(
        self,
        reader_a,
        reader_b,
        tensor_names: list[str],
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[tuple[str, MetricsBundle]]:
        results: list[tuple[str, MetricsBundle]] = []
        producer_done = threading.Event()

        def producer():
            try:
                for name in tensor_names:
                    tensor_a = reader_a.get_tensor(name)
                    tensor_b = reader_b.get_tensor(name)
                    slot = RingBufferSlot(name=name, tensor_a=tensor_a, tensor_b=tensor_b)
                    self._queue.put(slot)
                self._queue.put(_Sentinel())
            except Exception as e:  # noqa: BLE001
                self._queue.put(_ExceptionWrapper(e))
            finally:
                producer_done.set()

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        try:
            completed = 0
            total = len(tensor_names)

            while True:
                item = self._queue.get()

                if isinstance(item, _ExceptionWrapper):
                    raise item.exc

                if isinstance(item, _Sentinel):
                    break

                slot: RingBufferSlot = item
                metrics = compute_all_metrics(slot.tensor_a, slot.tensor_b)
                results.append((slot.name, metrics))

                completed += 1
                if progress_callback:
                    progress_callback(completed, total, slot.name)

                # Free tensors before sleep to minimize peak memory
                del slot.tensor_a
                del slot.tensor_b
                self._queue.task_done()

                # Minimal delay to simulate math computation and enable backpressure
                time.sleep(0.05)

            producer_thread.join(timeout=5.0)
            return results

        except Exception:
            producer_thread.join(timeout=5.0)
            raise

    def compute_metrics(self, tensor_a: np.ndarray, tensor_b: np.ndarray) -> MetricsBundle:
        """Compute metrics for a pair of tensors without queue overhead."""
        return compute_all_metrics(tensor_a, tensor_b)
