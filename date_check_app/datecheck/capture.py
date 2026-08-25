"""Capture OpenCV không chặn giao diện kiểm tra date."""
from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal


class CaptureThread(QThread):
    frame_ready = Signal(object)
    state_changed = Signal(str)
    error = Signal(str)

    def __init__(self, source: int | str, *, target_fps: float = 25.0) -> None:
        super().__init__()
        self.source = source
        self.target_fps = max(1.0, float(target_fps))
        self._running = True

    @staticmethod
    def source_from(kind: str, value: str) -> int | str:
        value = value.strip()
        if kind == "usb":
            if not value.isdecimal():
                raise ValueError("Camera USB phải là số, ví dụ 0")
            return int(value)
        if kind == "video":
            path = Path(value).expanduser()
            if not path.is_file():
                raise ValueError("Không tìm thấy tệp video")
            return str(path)
        if not value:
            raise ValueError("Địa chỉ RTSP không được để trống")
        return value

    def stop(self) -> None:
        self._running = False
        self.wait(1500)

    def run(self) -> None:
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            self.error.emit("Không mở được nguồn camera/video")
            return
        self.state_changed.emit("Đang chạy")
        delay = 1.0 / self.target_fps
        is_video = isinstance(self.source, str) and Path(self.source).is_file()
        try:
            while self._running:
                started = time.monotonic()
                ok, frame = capture.read()
                if not ok:
                    if is_video:
                        self.state_changed.emit("Đã hết video")
                        break
                    self.error.emit("Không đọc được frame từ camera")
                    self.msleep(100)
                    continue
                if isinstance(frame, np.ndarray) and frame.size:
                    self.frame_ready.emit(frame)
                remaining = delay - (time.monotonic() - started)
                if remaining > 0:
                    self.msleep(max(1, int(remaining * 1000)))
        finally:
            capture.release()
            self.state_changed.emit("Đã dừng")
