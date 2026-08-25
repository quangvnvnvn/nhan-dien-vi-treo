from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionMetrics:
    present: bool
    occupancy_ratio: float
    motion_score: float
    contour_area: float
    contour: np.ndarray | None


class ProductDetector(ABC):
    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def detect(self, frame: np.ndarray, zone: tuple[float, float, float, float]) -> DetectionMetrics: ...
