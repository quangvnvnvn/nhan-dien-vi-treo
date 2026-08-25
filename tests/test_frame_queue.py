from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from camera.frame_queue import LatestFrameQueue


class LatestFrameQueueTests(unittest.TestCase):
    def test_newest_frame_replaces_unconsumed_frame(self) -> None:
        queue = LatestFrameQueue()
        first = np.full((2, 2, 3), 1, dtype=np.uint8)
        newest = np.full((2, 2, 3), 2, dtype=np.uint8)

        queue.put(first, captured_at=1.0)
        queue.put(newest, captured_at=2.0)
        packet = queue.get_latest(timeout=0)

        self.assertIsNotNone(packet)
        assert packet is not None
        self.assertEqual(packet.sequence, 2)
        self.assertEqual(packet.captured_at, 2.0)
        self.assertTrue(np.array_equal(packet.frame, newest))
        self.assertEqual(queue.metrics.dropped, 1)
        self.assertEqual(queue.metrics.consumed, 1)

    def test_close_unblocks_waiting_consumer(self) -> None:
        queue = LatestFrameQueue()
        received: list[object] = []
        ready = threading.Event()

        def consumer() -> None:
            ready.set()
            received.append(queue.get_latest(timeout=None))

        thread = threading.Thread(target=consumer)
        thread.start()
        self.assertTrue(ready.wait(0.5))
        time.sleep(0.02)
        queue.close()
        thread.join(0.5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(received, [None])

    def test_closed_queue_rejects_new_frames_and_can_reopen(self) -> None:
        queue = LatestFrameQueue()
        queue.close()
        with self.assertRaises(RuntimeError):
            queue.put(np.zeros((1, 1), dtype=np.uint8))

        queue.reopen(reset_metrics=True)
        queue.put(np.zeros((1, 1), dtype=np.uint8))
        self.assertEqual(queue.metrics.produced, 1)
        self.assertFalse(queue.metrics.closed)
