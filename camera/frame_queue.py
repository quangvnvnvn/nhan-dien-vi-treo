"""Buffer thread-safe giữ duy nhất frame mới nhất.

Không dùng ``queue.Queue(maxsize=1)`` vì khi đầy nó thường chặn producer hoặc
giữ frame cũ. Với camera realtime, frame cũ không còn giá trị: inference luôn
nên nhận frame mới nhất để giảm độ trễ.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from camera.camera_interface import FramePacket, FrameQueueMetrics


class LatestFrameQueue:
    """Queue dung lượng một frame, tự ghi đè frame chưa được lấy.

    ``put`` không sao chép mảng NumPy để tránh copy tốn kém ở tốc độ camera.
    Caller không được sửa frame sau khi đã đưa vào queue. ``get_latest`` lấy
    và xóa frame hiện có, nên consumer chậm chỉ nhận được frame mới nhất.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: FramePacket | None = None
        self._next_sequence = 1
        self._produced = 0
        self._consumed = 0
        self._dropped = 0
        self._closed = False

    def put(self, frame: np.ndarray, *, captured_at: float | None = None) -> FramePacket:
        """Đưa frame mới nhất vào buffer và ghi đè frame cũ nếu cần.

        Raises:
            TypeError: nếu ``frame`` không phải ``numpy.ndarray``.
            RuntimeError: nếu queue đã được đóng.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame phải là numpy.ndarray")
        packet = FramePacket(
            sequence=0,
            frame=frame,
            captured_at=time.monotonic() if captured_at is None else captured_at,
        )
        with self._condition:
            if self._closed:
                raise RuntimeError("LatestFrameQueue đã đóng")
            if self._latest is not None:
                self._dropped += 1
            packet = FramePacket(self._next_sequence, packet.frame, packet.captured_at)
            self._next_sequence += 1
            self._produced += 1
            self._latest = packet
            self._condition.notify_all()
        return packet

    def get_latest(self, timeout: float | None = None) -> FramePacket | None:
        """Lấy frame mới nhất, hoặc ``None`` khi hết thời gian chờ/đã đóng.

        ``timeout=None`` nghĩa là chờ vô thời hạn. Giá trị âm không hợp lệ vì
        thường che giấu lỗi cấu hình ở consumer.
        """
        if timeout is not None and timeout < 0:
            raise ValueError("timeout phải lớn hơn hoặc bằng 0")
        with self._condition:
            if self._latest is None and not self._closed:
                if timeout is None:
                    self._condition.wait_for(lambda: self._latest is not None or self._closed)
                else:
                    self._condition.wait_for(
                        lambda: self._latest is not None or self._closed,
                        timeout=timeout,
                    )
            if self._latest is None:
                return None
            packet = self._latest
            self._latest = None
            self._consumed += 1
            return packet

    def peek_latest(self) -> FramePacket | None:
        """Trả frame mới nhất mà không lấy nó khỏi buffer."""
        with self._condition:
            return self._latest

    def clear(self) -> None:
        """Bỏ frame đang chờ, dùng khi consumer đổi chế độ/pipeline."""
        with self._condition:
            if self._latest is not None:
                self._latest = None
                self._dropped += 1

    def close(self) -> None:
        """Đánh thức consumer đang chờ và từ chối frame mới."""
        with self._condition:
            self._closed = True
            self._latest = None
            self._condition.notify_all()

    def reopen(self, *, reset_metrics: bool = False) -> None:
        """Mở lại queue trước một phiên capture mới."""
        with self._condition:
            self._closed = False
            self._latest = None
            if reset_metrics:
                self._next_sequence = 1
                self._produced = 0
                self._consumed = 0
                self._dropped = 0
            self._condition.notify_all()

    @property
    def metrics(self) -> FrameQueueMetrics:
        """Lấy snapshot metric không thay đổi được."""
        with self._condition:
            return FrameQueueMetrics(
                produced=self._produced,
                consumed=self._consumed,
                dropped=self._dropped,
                has_frame=self._latest is not None,
                closed=self._closed,
            )
