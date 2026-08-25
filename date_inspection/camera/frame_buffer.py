from __future__ import annotations

from collections import deque
from threading import Lock

from sources import FramePacket


class FrameBuffer:
    def __init__(self, capacity: int = 30) -> None:
        self._frames: deque[FramePacket] = deque(maxlen=max(1, capacity))
        self._lock = Lock()

    def append(self, packet: FramePacket) -> None:
        with self._lock:
            self._frames.append(packet)

    def snapshot(self) -> list[FramePacket]:
        with self._lock:
            return list(self._frames)
