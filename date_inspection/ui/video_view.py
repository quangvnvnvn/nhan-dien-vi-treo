from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel


class VideoView(QLabel):
    zone_drawn = Signal(tuple)

    def __init__(self) -> None:
        super().__init__("Chọn video hoặc camera để bắt đầu")
        self.setMinimumSize(720, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#101827; color:#cbd5e1; border:1px solid #334155;")
        self.frame: np.ndarray | None = None
        self.zone: tuple[float, float, float, float] | None = None
        self.contour: np.ndarray | None = None
        self.state = "CHƯA CẤU HÌNH VÙNG PHÁT HIỆN"
        self.metrics = ""
        self._drawing = False
        self._start: tuple[float, float] | None = None
        self._draft: tuple[float, float, float, float] | None = None

    def set_frame(self, frame: np.ndarray) -> None:
        self.frame = frame.copy()
        self._refresh()

    def set_overlay(self, zone: tuple[float, float, float, float] | None, state: str, metrics: str,
                    contour: np.ndarray | None = None) -> None:
        self.zone, self.state, self.metrics, self.contour = zone, state, metrics, contour
        self._refresh()

    def begin_zone_draw(self) -> None:
        if self.frame is not None:
            self._drawing = True
            self.setCursor(Qt.CrossCursor)

    def _display_rect(self) -> QRect:
        pixmap = self.pixmap()
        if pixmap is None:
            return QRect()
        return QRect((self.width() - pixmap.width()) // 2, (self.height() - pixmap.height()) // 2, pixmap.width(), pixmap.height())

    def _frame_point(self, point: Any) -> tuple[float, float] | None:
        if self.frame is None:
            return None
        rect = self._display_rect()
        if rect.isEmpty() or not rect.contains(point):
            return None
        h, w = self.frame.shape[:2]
        return ((point.x() - rect.x()) * w / rect.width(), (point.y() - rect.y()) * h / rect.height())

    def mousePressEvent(self, event: Any) -> None:
        if self._drawing and event.button() == Qt.LeftButton:
            self._start = self._frame_point(event.position().toPoint())
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:
        end = self._frame_point(event.position().toPoint()) if self._drawing else None
        if self._drawing and self._start and end and self.frame is not None:
            self._draft = self._normalise(self._start, end, self.frame.shape[1], self.frame.shape[0])
            self._refresh()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        if self._drawing and event.button() == Qt.LeftButton:
            end = self._frame_point(event.position().toPoint())
            self._drawing = False
            self.unsetCursor()
            if self._start and end and self.frame is not None:
                roi = self._normalise(self._start, end, self.frame.shape[1], self.frame.shape[0])
                if roi:
                    self.zone_drawn.emit(roi)
            self._start, self._draft = None, None
            self._refresh()
            return
        super().mouseReleaseEvent(event)

    @staticmethod
    def _normalise(start: tuple[float, float], end: tuple[float, float], width: int, height: int) -> tuple[float, float, float, float] | None:
        x0, x1 = sorted((start[0] / width, end[0] / width)); y0, y1 = sorted((start[1] / height, end[1] / height))
        return (x0, y0, x1 - x0, y1 - y0) if x1 - x0 >= .02 and y1 - y0 >= .02 else None

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event); self._refresh()

    def _refresh(self) -> None:
        if self.frame is None:
            return
        rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
        painter = QPainter(image); painter.setRenderHint(QPainter.Antialiasing)
        roi = self._draft or self.zone
        if roi:
            x, y, w, h = roi; rectangle = QRectF(x * image.width(), y * image.height(), w * image.width(), h * image.height())
            painter.setPen(QPen(QColor("#22d3ee"), max(2, image.width() // 360))); painter.drawRect(rectangle)
            painter.drawText(rectangle.adjusted(5, 5, -5, -5), Qt.AlignTop | Qt.AlignLeft, "PRODUCT DETECTION ZONE")
        if self.contour is not None:
            painter.setPen(QPen(QColor("#fbbf24"), max(2, image.width() // 450)))
            for point in self.contour.reshape(-1, 2): painter.drawPoint(int(point[0]), int(point[1]))
        painter.setPen(QPen(QColor("#ffffff"), max(1, image.width() // 600)))
        painter.drawText(12, 26, self.state); painter.drawText(12, 50, self.metrics)
        painter.end()
        self.setPixmap(QPixmap.fromImage(image).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
