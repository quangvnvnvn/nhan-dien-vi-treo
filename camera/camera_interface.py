"""Kiểu dữ liệu và protocol dùng chung cho nguồn camera.

Module này cố ý không phụ thuộc vào giao diện.  Điều đó giúp thay thế webcam
bằng RTSP, video file hoặc camera giả trong unit test mà không cần thiết bị
thật.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np


CameraSource: TypeAlias = int | str
"""Chỉ số webcam (ví dụ ``0``) hoặc URL RTSP/đường dẫn video."""


@runtime_checkable
class VideoCaptureLike(Protocol):
    """Phần API OpenCV mà :class:`CameraThread` thực sự cần."""

    def isOpened(self) -> bool:  # noqa: N802 - khớp API OpenCV
        """Trả về camera có mở được hay không."""

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Đọc một frame BGR của OpenCV."""

    def release(self) -> None:
        """Giải phóng handle của nguồn video."""


@dataclass(frozen=True, slots=True)
class FramePacket:
    """Một frame cùng số thứ tự và thời điểm capture đơn điệu."""

    sequence: int
    frame: np.ndarray
    captured_at: float


@dataclass(frozen=True, slots=True)
class FrameQueueMetrics:
    """Metric của buffer latest-frame (dung lượng luôn tối đa một frame)."""

    produced: int = 0
    consumed: int = 0
    dropped: int = 0
    has_frame: bool = False
    closed: bool = False


@dataclass(frozen=True, slots=True)
class CameraMetrics:
    """Snapshot immutable của trạng thái capture, an toàn để gửi ra GUI."""

    source: CameraSource
    is_open: bool = False
    started_at: float | None = None
    stopped_at: float | None = None
    frames_captured: int = 0
    frames_emitted: int = 0
    queue_dropped_frames: int = 0
    read_failures: int = 0
    consecutive_read_failures: int = 0
    last_frame_at: float | None = None
    last_error: str | None = None

    @property
    def average_capture_fps(self) -> float:
        """FPS trung bình của vòng đời hiện tại, không phải FPS tức thời."""
        if self.started_at is None:
            return 0.0
        end = self.stopped_at if self.stopped_at is not None else self.last_frame_at
        if end is None or end <= self.started_at:
            return 0.0
        return self.frames_captured / (end - self.started_at)


@dataclass(frozen=True, slots=True)
class CameraError:
    """Thông tin lỗi có cấu trúc bên cạnh tín hiệu chuỗi cho UI."""

    message: str
    source: CameraSource
    recoverable: bool
    consecutive_failures: int = 0
    exception_type: str | None = None
    occurred_at: float = field(default_factory=time.monotonic)
