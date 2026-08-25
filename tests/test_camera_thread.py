from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from camera.camera_thread import CameraThread


class FakeCapture:
    """Capture deterministic, không cần webcam/RTSP trong test."""

    def __init__(self, frames: list[np.ndarray], *, opened: bool = True) -> None:
        self._frames = list(frames)
        self._opened = opened
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - khớp OpenCV
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self) -> None:
        self.released = True


class CameraThreadTests(unittest.TestCase):
    def test_capture_worker_keeps_latest_frame_and_releases_handle(self) -> None:
        fake = FakeCapture([
            np.full((2, 2, 3), 10, dtype=np.uint8),
            np.full((2, 2, 3), 20, dtype=np.uint8),
        ])
        worker = CameraThread(
            source=0,
            target_fps=None,
            max_consecutive_failures=1,
            failure_backoff_seconds=0,
            capture_factory=lambda _source: fake,
        )

        # Chạy QThread thật với capture giả, không cần QApplication/camera.
        worker.start()
        self.assertTrue(worker.wait(1_000))

        metrics = worker.metrics
        self.assertTrue(fake.released)
        self.assertFalse(metrics.is_open)
        self.assertEqual(metrics.frames_captured, 2)
        self.assertEqual(metrics.frames_emitted, 2)
        self.assertEqual(metrics.queue_dropped_frames, 1)
        self.assertEqual(metrics.read_failures, 1)
        packet = worker.latest_frames.get_latest(timeout=0)
        self.assertIsNone(packet, "queue được đóng khi worker kết thúc")

    def test_unopenable_source_reports_error_and_releases_handle(self) -> None:
        fake = FakeCapture([], opened=False)
        errors: list[str] = []
        worker = CameraThread(
            source="rtsp://example.invalid/stream",
            capture_factory=lambda _source: fake,
        )
        worker.camera_error.connect(errors.append)

        worker.run()

        self.assertTrue(fake.released)
        self.assertFalse(worker.metrics.is_open)
        self.assertTrue(errors)
        self.assertIn("Không mở được camera", errors[0])

    def test_source_validation_and_safe_stop_when_idle(self) -> None:
        with self.assertRaises(ValueError):
            CameraThread(source=-1)
        with self.assertRaises(ValueError):
            CameraThread(source="  ")
        with self.assertRaises(ValueError):
            CameraThread(target_fps=0)

        worker = CameraThread()
        self.assertTrue(worker.stop(timeout_ms=0))

    def test_video_file_stops_cleanly_at_end_instead_of_retrying(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "sample.mp4"
            source.touch()
            fake = FakeCapture([np.full((2, 2, 3), 10, dtype=np.uint8)])
            errors: list[str] = []
            worker = CameraThread(
                source=str(source),
                target_fps=None,
                capture_factory=lambda _source: fake,
            )
            worker.camera_error.connect(errors.append)

            worker.run()

        self.assertTrue(fake.released)
        self.assertEqual(worker.metrics.frames_captured, 1)
        self.assertEqual(worker.metrics.read_failures, 0)
        self.assertEqual(worker.metrics.last_error, "Đã phát hết video kiểm tra")
        self.assertEqual(errors, ["Đã phát hết video kiểm tra"])
