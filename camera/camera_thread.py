"""Worker QThread đọc webcam hoặc RTSP qua OpenCV.

Lớp này chỉ capture frame. Nó không chạy detector trên QThread để inference có
thể được tách riêng và luôn lấy frame mới nhất từ :class:`LatestFrameQueue`.
"""
from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path
import threading
import time
from collections.abc import Callable

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from camera.camera_interface import (
    CameraError,
    CameraMetrics,
    CameraSource,
    VideoCaptureLike,
)
from camera.frame_queue import LatestFrameQueue


LOGGER = logging.getLogger(__name__)
CaptureFactory = Callable[[CameraSource], VideoCaptureLike]


class CameraThread(QThread):
    """Đọc webcam/RTSP trên worker thread và phát frame BGR mới nhất.

    Signals:
        frame_ready: phát ``numpy.ndarray`` BGR sau mỗi lần đọc thành công.
        camera_opened: phát mô tả nguồn sau khi OpenCV mở thành công.
        camera_error: chuỗi lỗi ngắn, phù hợp để hiện lên giao diện.
        error_details: :class:`CameraError` có ngữ cảnh cho log/audit.
        metrics_updated: :class:`CameraMetrics` immutable.
        camera_stopped: phát sau khi handle camera đã được release.

    ``stop`` không dùng ``terminate``. Nó yêu cầu worker dừng, đóng frame queue
    để đánh thức consumer và chờ một khoảng hữu hạn để tránh khóa GUI.
    """

    frame_ready = Signal(np.ndarray)
    camera_opened = Signal(str)
    camera_error = Signal(str)
    error_details = Signal(object)
    metrics_updated = Signal(object)
    camera_stopped = Signal()

    def __init__(
        self,
        source: CameraSource = 0,
        *,
        target_fps: float | None = 30.0,
        max_consecutive_failures: int = 15,
        failure_backoff_seconds: float = 0.05,
        metrics_interval_seconds: float = 1.0,
        capture_factory: CaptureFactory | None = None,
        latest_frames: LatestFrameQueue | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._validate_source(source)
        if target_fps is not None and target_fps <= 0:
            raise ValueError("target_fps phải dương hoặc None")
        if max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures phải từ 1 trở lên")
        if failure_backoff_seconds < 0:
            raise ValueError("failure_backoff_seconds không được âm")
        if metrics_interval_seconds <= 0:
            raise ValueError("metrics_interval_seconds phải dương")

        self._source = source
        self._is_video_file = isinstance(source, str) and Path(source).is_file()
        self._target_fps = target_fps
        self._max_consecutive_failures = max_consecutive_failures
        self._failure_backoff_seconds = failure_backoff_seconds
        self._metrics_interval_seconds = metrics_interval_seconds
        self._capture_factory = capture_factory or self._default_capture_factory
        self._latest_frames = latest_frames or LatestFrameQueue()
        self._stop_requested = threading.Event()
        self._capture_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._capture: VideoCaptureLike | None = None
        self._metrics = CameraMetrics(source=source)
        self._last_metrics_emit_at = 0.0

    @staticmethod
    def _default_capture_factory(source: CameraSource) -> VideoCaptureLike:
        return cv2.VideoCapture(source)

    @staticmethod
    def _validate_source(source: CameraSource) -> None:
        if isinstance(source, bool) or not isinstance(source, (int, str)):
            raise TypeError("source phải là số webcam hoặc chuỗi URL/đường dẫn")
        if isinstance(source, int) and source < 0:
            raise ValueError("chỉ số webcam không được âm")
        if isinstance(source, str) and not source.strip():
            raise ValueError("URL/đường dẫn camera không được để trống")

    @property
    def source(self) -> CameraSource:
        return self._source

    @property
    def latest_frames(self) -> LatestFrameQueue:
        """Buffer dung lượng một để worker inference lấy frame mới nhất."""
        return self._latest_frames

    @property
    def metrics(self) -> CameraMetrics:
        """Snapshot metric hiện tại, an toàn khi gọi từ GUI thread."""
        with self._metrics_lock:
            return self._metrics

    def set_source(self, source: CameraSource) -> None:
        """Đổi nguồn trước khi start; không đổi giữa một phiên capture."""
        self._validate_source(source)
        if self.isRunning():
            raise RuntimeError("Không thể đổi source khi camera đang chạy")
        self._source = source
        self._is_video_file = isinstance(source, str) and Path(source).is_file()
        self._replace_metrics(source=source)

    def start(self, priority: QThread.Priority = QThread.Priority.InheritPriority) -> None:
        """Khởi động một phiên mới và mở lại latest-frame queue."""
        if self.isRunning():
            raise RuntimeError("Camera đang chạy")
        self._stop_requested.clear()
        self._latest_frames.reopen(reset_metrics=True)
        self._last_metrics_emit_at = 0.0
        super().start(priority)

    def stop(self, timeout_ms: int = 3_000) -> bool:
        """Yêu cầu dừng an toàn; trả về ``True`` khi worker đã kết thúc.

        Khi URL RTSP hoặc driver treo ngay trong ``read()``, OpenCV có thể không
        trả quyền điều khiển ngay. Trường hợp đó hàm trả ``False`` sau timeout;
        caller không nên dùng ``terminate`` vì dễ làm rò handle camera.
        """
        if timeout_ms < 0:
            raise ValueError("timeout_ms không được âm")
        self._stop_requested.set()
        self.requestInterruption()
        self._latest_frames.close()
        if not self.isRunning():
            return True
        if self.isCurrentThread():
            return False
        return self.wait(timeout_ms)

    def run(self) -> None:
        """Vòng lặp QThread; chỉ được QThread gọi, không gọi trực tiếp từ UI."""
        capture: VideoCaptureLike | None = None
        started_at = time.monotonic()
        self._replace_metrics(
            source=self._source,
            is_open=False,
            started_at=started_at,
            stopped_at=None,
            frames_captured=0,
            frames_emitted=0,
            queue_dropped_frames=0,
            read_failures=0,
            consecutive_read_failures=0,
            last_frame_at=None,
            last_error=None,
        )
        try:
            capture = self._open_capture()
            if capture is None:
                return
            self._replace_metrics(is_open=True)
            self.camera_opened.emit(self._source_label())
            self._emit_metrics(force=True)
            self._capture_loop(capture)
        finally:
            if capture is not None:
                self._release_capture(capture)
            with self._capture_lock:
                self._capture = None
            self._latest_frames.close()
            self._replace_metrics(is_open=False, stopped_at=time.monotonic())
            self._emit_metrics(force=True)
            self.camera_stopped.emit()

    def _open_capture(self) -> VideoCaptureLike | None:
        capture: VideoCaptureLike | None = None
        try:
            capture = self._capture_factory(self._source)
            with self._capture_lock:
                self._capture = capture
            if not capture.isOpened():
                self._emit_error("Không mở được camera: " + self._source_label(), recoverable=False)
                self._release_capture(capture)
                with self._capture_lock:
                    self._capture = None
                return None
            return capture
        except Exception as error:
            self._emit_error(
                "Lỗi khi mở camera: " + self._source_label(),
                recoverable=False,
                exception=error,
            )
            if capture is not None:
                self._release_capture(capture)
                with self._capture_lock:
                    self._capture = None
            return None

    def _release_capture(self, capture: VideoCaptureLike) -> None:
        try:
            capture.release()
        except Exception:  # pragma: no cover - phụ thuộc driver OpenCV
            LOGGER.exception("Không thể release nguồn camera %s", self._source_label())

    def _capture_loop(self, capture: VideoCaptureLike) -> None:
        frame_interval = None if self._target_fps is None else 1.0 / self._target_fps
        while not self._should_stop():
            cycle_started = time.monotonic()
            try:
                ok, frame = capture.read()
            except Exception as error:
                self._handle_read_failure(error)
                if self.metrics.consecutive_read_failures >= self._max_consecutive_failures:
                    break
                self._wait_after_failure()
                continue

            if not ok or not isinstance(frame, np.ndarray) or frame.size == 0:
                if self._is_video_file:
                    self._emit_error("Đã phát hết video kiểm tra", recoverable=False)
                    break
                self._handle_read_failure(None)
                if self.metrics.consecutive_read_failures >= self._max_consecutive_failures:
                    break
                self._wait_after_failure()
                continue

            if self._should_stop():
                break
            captured_at = time.monotonic()
            try:
                self._latest_frames.put(frame, captured_at=captured_at)
            except RuntimeError:
                # Queue được close bởi stop() giữa read và put().
                break

            queue_metrics = self._latest_frames.metrics
            current = self.metrics
            self._replace_metrics(
                frames_captured=current.frames_captured + 1,
                frames_emitted=current.frames_emitted + 1,
                queue_dropped_frames=queue_metrics.dropped,
                consecutive_read_failures=0,
                last_frame_at=captured_at,
            )
            self.frame_ready.emit(frame)
            self._emit_metrics()
            if frame_interval is not None:
                remaining = frame_interval - (time.monotonic() - cycle_started)
                if remaining > 0:
                    self._stop_requested.wait(remaining)

    def _handle_read_failure(self, exception: Exception | None) -> None:
        current = self.metrics
        failures = current.consecutive_read_failures + 1
        self._replace_metrics(
            read_failures=current.read_failures + 1,
            consecutive_read_failures=failures,
        )
        # Một lỗi đầu tiên thông báo rằng stream đã gián đoạn; lỗi cuối nêu rõ
        # worker sẽ tự dừng. Không phát mỗi retry để không spam GUI/log.
        if failures == 1:
            self._emit_error("Không đọc được frame từ camera, đang thử lại", recoverable=True, exception=exception)
        elif failures >= self._max_consecutive_failures:
            self._emit_error(
                f"Dừng camera sau {failures} lần không đọc được frame liên tiếp",
                recoverable=False,
                exception=exception,
            )

    def _wait_after_failure(self) -> None:
        if self._failure_backoff_seconds:
            self._stop_requested.wait(self._failure_backoff_seconds)

    def _should_stop(self) -> bool:
        return self._stop_requested.is_set() or self.isInterruptionRequested()

    def _replace_metrics(self, **changes: object) -> CameraMetrics:
        with self._metrics_lock:
            self._metrics = replace(self._metrics, **changes)
            return self._metrics

    def _emit_metrics(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self._last_metrics_emit_at >= self._metrics_interval_seconds:
            self._last_metrics_emit_at = now
            self.metrics_updated.emit(self.metrics)

    def _emit_error(
        self,
        message: str,
        *,
        recoverable: bool,
        exception: Exception | None = None,
    ) -> None:
        if exception is None:
            LOGGER.warning("%s", message)
        else:
            LOGGER.warning("%s: %s", message, exception, exc_info=True)
        metrics = self._replace_metrics(last_error=message)
        details = CameraError(
            message=message,
            source=self._source,
            recoverable=recoverable,
            consecutive_failures=metrics.consecutive_read_failures,
            exception_type=type(exception).__name__ if exception is not None else None,
            occurred_at=time.monotonic(),
        )
        self.camera_error.emit(message)
        self.error_details.emit(details)
        self._emit_metrics(force=True)

    def _source_label(self) -> str:
        return f"webcam {self._source}" if isinstance(self._source, int) else self._source
