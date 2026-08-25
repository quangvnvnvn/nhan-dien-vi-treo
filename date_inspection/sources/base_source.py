from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FramePacket:
    image: np.ndarray
    frame_number: int
    timestamp: float


class FrameSource(ABC):
    """Nguồn frame có thể thay thế mà pipeline không cần biết loại camera."""
    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read(self) -> FramePacket | None: ...

    @abstractmethod
    def release(self) -> None: ...

    @property
    @abstractmethod
    def fps(self) -> float: ...

    @property
    def frame_count(self) -> int:
        return 0

    def seek(self, frame_number: int) -> None:
        raise RuntimeError("Nguồn này không hỗ trợ tua video")
