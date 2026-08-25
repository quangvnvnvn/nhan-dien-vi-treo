from __future__ import annotations

import threading
import time

from PySide6.QtCore import QThread, Signal

from sources import FramePacket, FrameSource


class FrameReader(QThread):
    packet_ready = Signal(object)
    ended = Signal()
    failed = Signal(str)

    def __init__(self, source: FrameSource, is_video: bool) -> None:
        super().__init__()
        self.source, self.is_video = source, is_video
        self._running, self._paused, self._speed = True, False, 1.0
        self._lock = threading.Lock()
        self._seek_target: int | None = None

    def set_paused(self, value: bool) -> None:
        with self._lock: self._paused = value

    def set_speed(self, value: float) -> None:
        with self._lock: self._speed = max(.1, value)

    def seek(self, frame: int) -> None:
        with self._lock: self._seek_target = frame

    def stop(self) -> None:
        self._running = False
        self.wait(2000)

    def run(self) -> None:
        try:
            self.source.open()
            while self._running:
                with self._lock:
                    paused, speed, seek_target = self._paused, self._speed, self._seek_target
                    self._seek_target = None
                if seek_target is not None:
                    self.source.seek(seek_target)
                    paused = False  # seek luôn hiển thị ngay frame được chọn
                    with self._lock: self._paused = True
                if paused:
                    self.msleep(15); continue
                started = time.monotonic(); packet = self.source.read()
                if packet is None:
                    self.ended.emit(); break
                self.packet_ready.emit(packet)
                if self.is_video:
                    remaining = 1 / self.source.fps / speed - (time.monotonic() - started)
                    if remaining > 0: self.msleep(max(1, int(remaining * 1000)))
        except Exception as error:
            self.failed.emit(str(error))
        finally:
            self.source.release()
