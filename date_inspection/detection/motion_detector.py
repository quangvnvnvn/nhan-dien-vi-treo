from __future__ import annotations

import cv2
import numpy as np

from .product_detector import DetectionMetrics, ProductDetector


class MotionContourDetector(ProductDetector):
    """Detector nhẹ: background running-average + threshold + contour trong vùng đã vẽ."""
    def __init__(self, threshold: int = 30, min_contour_area: int = 800, max_contour_area: int = 10_000_000,
                 occupancy_threshold: float = 0.025, sensitivity: float = 1.0) -> None:
        self.threshold = threshold
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area
        self.occupancy_threshold = occupancy_threshold
        self.sensitivity = sensitivity
        self._background: np.ndarray | None = None

    def reset(self) -> None:
        self._background = None

    def detect(self, frame: np.ndarray, zone: tuple[float, float, float, float]) -> DetectionMetrics:
        h, w = frame.shape[:2]
        x, y, zw, zh = zone
        left, top = int(x * w), int(y * h)
        right, bottom = min(w, int((x + zw) * w)), min(h, int((y + zh) * h))
        if right <= left or bottom <= top:
            return DetectionMetrics(False, 0.0, 0.0, 0.0, None)
        gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self._background is None:
            self._background = gray.astype(np.float32)
            return DetectionMetrics(False, 0.0, 0.0, 0.0, None)
        delta = cv2.absdiff(gray, cv2.convertScaleAbs(self._background))
        threshold = max(5, int(self.threshold / max(0.2, self.sensitivity)))
        _, mask = cv2.threshold(delta, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        selected = max(contours, key=cv2.contourArea, default=None)
        area = float(cv2.contourArea(selected)) if selected is not None else 0.0
        occupancy = float(cv2.countNonZero(mask)) / float(mask.size)
        present = self.min_contour_area <= area <= self.max_contour_area and occupancy >= self.occupancy_threshold
        # Chỉ cập nhật nền khi vùng trống để tránh "nuốt" sản phẩm đang chạy.
        if not present:
            cv2.accumulateWeighted(gray, self._background, 0.04)
        contour_full = selected + np.array([[[left, top]]]) if selected is not None else None
        return DetectionMetrics(present, occupancy, float(np.mean(delta)), area, contour_full)
