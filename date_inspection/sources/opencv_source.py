from __future__ import annotations

from pathlib import Path
import time

import cv2

from .base_source import FramePacket, FrameSource


class OpenCVSource(FrameSource):
    def __init__(self, source: int | str) -> None:
        self.source = source
        self.capture: cv2.VideoCapture | None = None
        self._fps = 30.0
        self._frame_count = 0
        self._number = 0

    def open(self) -> None:
        self.capture = cv2.VideoCapture(self.source, cv2.CAP_DSHOW if isinstance(self.source, int) else cv2.CAP_ANY)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError("Không thể mở nguồn video/camera.")
        reported_fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self._fps = reported_fps if reported_fps > 1 else 30.0
        self._frame_count = max(0, int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self._number = max(0, int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)))

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def read(self) -> FramePacket | None:
        if self.capture is None:
            raise RuntimeError("Nguồn chưa được mở.")
        ok, image = self.capture.read()
        if not ok or image is None:
            return None
        self._number = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        timestamp = self._number / self._fps if self._frame_count else time.monotonic()
        return FramePacket(image=image, frame_number=max(0, self._number), timestamp=timestamp)

    def seek(self, frame_number: int) -> None:
        if self.capture is None or not self._frame_count:
            raise RuntimeError("Nguồn này không hỗ trợ tua video")
        target = max(0, min(self._frame_count - 1, int(frame_number)))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        self._number = target

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class OpenCVCameraSource(OpenCVSource):
    def __init__(self, index: int) -> None:
        super().__init__(index)


class VideoFileSource(OpenCVSource):
    def __init__(self, path: str) -> None:
        if not Path(path).is_file():
            raise ValueError("Không tìm thấy tệp video đã chọn.")
        super().__init__(path)
