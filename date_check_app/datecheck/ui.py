"""Giao diện realtime cho ứng dụng kiểm tra date độc lập."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRect, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from datecheck.capture import CaptureThread
from datecheck.date_parser import DateExpectation, DateStatus, DateValidation
from datecheck.ocr_engine import DateOcrEngine, OcrResult


APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "data"
SETTINGS_PATH = DATA_DIR / "settings.json"


class SettingsStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {"rois": {}}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.data.update(loaded)
        except (OSError, ValueError, TypeError):
            pass

    def roi(self, key: str) -> tuple[float, float, float, float] | None:
        value = self.data.get("rois", {}).get(key)
        if not isinstance(value, list) or len(value) != 4:
            return None
        try:
            values = tuple(float(item) for item in value)
        except (TypeError, ValueError):
            return None
        if values[2] <= 0 or values[3] <= 0:
            return None
        return values

    def save_roi(self, key: str, roi: tuple[float, float, float, float]) -> None:
        rois = self.data.setdefault("rois", {})
        rois[key] = list(roi)
        self._save()

    def remove_roi(self, key: str) -> None:
        self.data.setdefault("rois", {}).pop(key, None)
        self._save()

    def expectation(self) -> dict[str, str]:
        value = self.data.get("expectation", {})
        if not isinstance(value, dict):
            return {}
        return {key: str(value.get(key, "")) for key in ("date", "dm", "batch")}

    def save_expectation(self, expectation: DateExpectation) -> None:
        self.data["expectation"] = {
            "date": expectation.manufacture_date,
            "dm": expectation.dm_code,
            "batch": expectation.batch_code,
        }
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")


class VideoView(QLabel):
    roi_drawn = Signal(tuple)

    def __init__(self) -> None:
        super().__init__("Chọn camera hoặc video để bắt đầu")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(700, 520)
        self.setStyleSheet("background:#0d1628; color:#9ab0cd; border:1px solid #334155;")
        self._frame: np.ndarray | None = None
        self._roi: tuple[float, float, float, float] | None = None
        self._drawing = False
        self._start: tuple[float, float] | None = None
        self._draft: tuple[float, float, float, float] | None = None
        self._result: DateValidation | None = None

    def set_frame(self, frame: np.ndarray) -> None:
        self._frame = frame.copy()
        self._update_pixmap()

    def set_roi(self, roi: tuple[float, float, float, float] | None) -> None:
        self._roi = roi
        self._update_pixmap()

    def set_result(self, result: DateValidation | None) -> None:
        self._result = result
        self._update_pixmap()

    def begin_draw(self) -> None:
        if self._frame is None:
            return
        self._drawing = True
        self._draft = None
        self.setCursor(Qt.CrossCursor)

    def _display_rect(self) -> QRect:
        if self._frame is None or self.pixmap() is None:
            return QRect()
        pixmap = self.pixmap()
        x = (self.width() - pixmap.width()) // 2
        y = (self.height() - pixmap.height()) // 2
        return QRect(x, y, pixmap.width(), pixmap.height())

    def _frame_point(self, point: Any) -> tuple[float, float] | None:
        if self._frame is None:
            return None
        rect = self._display_rect()
        if rect.isEmpty() or not rect.contains(point):
            return None
        height, width = self._frame.shape[:2]
        return (point.x() - rect.x()) / rect.width() * width, (point.y() - rect.y()) / rect.height() * height

    @staticmethod
    def _normalise_rect(start: tuple[float, float], end: tuple[float, float], width: int, height: int) -> tuple[float, float, float, float] | None:
        x0, x1 = sorted((start[0] / width, end[0] / width))
        y0, y1 = sorted((start[1] / height, end[1] / height))
        if (x1 - x0) < 0.02 or (y1 - y0) < 0.015:
            return None
        return max(0.0, x0), max(0.0, y0), min(1.0, x1 - x0), min(1.0, y1 - y0)

    def mousePressEvent(self, event: Any) -> None:
        if self._drawing and event.button() == Qt.LeftButton:
            self._start = self._frame_point(event.position().toPoint())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:
        if self._drawing and self._start is not None:
            end = self._frame_point(event.position().toPoint())
            if end is not None and self._frame is not None:
                self._draft = self._normalise_rect(self._start, end, self._frame.shape[1], self._frame.shape[0])
                self._update_pixmap()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        if self._drawing and event.button() == Qt.LeftButton:
            end = self._frame_point(event.position().toPoint())
            roi = None
            if end is not None and self._start is not None and self._frame is not None:
                roi = self._normalise_rect(self._start, end, self._frame.shape[1], self._frame.shape[0])
            self._drawing = False
            self._start = None
            self._draft = None
            self.unsetCursor()
            if roi is not None:
                self.roi_drawn.emit(roi)
            self._update_pixmap()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._frame is None:
            return
        rgb = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        rect_to_draw = self._draft or self._roi
        if rect_to_draw is not None:
            x, y, width, height = rect_to_draw
            rectangle = QRectF(x * image.width(), y * image.height(), width * image.width(), height * image.height())
            painter.setPen(QPen(QColor("#22d3ee"), max(2, image.width() // 360)))
            painter.drawRect(rectangle)
            painter.setPen(QPen(QColor("#dbeafe"), max(1, image.width() // 560)))
            painter.drawText(rectangle.adjusted(5, 5, -5, -5), Qt.AlignTop | Qt.AlignLeft, "ROI DATE")
        if self._result is not None:
            palette = {DateStatus.PASS: "#22c55e", DateStatus.FAIL: "#ef4444", DateStatus.REVIEW: "#facc15"}
            painter.setPen(QPen(QColor(palette[self._result.status]), max(2, image.width() // 300)))
            painter.drawText(14, 34, f"{self._result.status} - {self._result.detail[:56]}")
        painter.end()
        pixmap = QPixmap.fromImage(image).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pixmap)


class OcrWorker(QThread):
    result_ready = Signal(object)
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._best_crop: np.ndarray | None = None
        self._best_quality = -1.0
        self._expectation = DateExpectation()
        self._running = True

    def submit(self, frame: np.ndarray, roi: tuple[float, float, float, float] | None, expectation: DateExpectation) -> None:
        if roi is None:
            return
        crop = self._crop(frame, roi)
        if crop.size == 0:
            return
        quality = self._quality(crop)
        with self._lock:
            # OCR mất lâu hơn tốc độ camera.  Thay vì luôn thay bởi frame mới
            # (thường nhòe lúc vỉ đang chạy), giữ frame sắc nét nhất cho tới
            # lượt OCR kế tiếp.
            if self._best_crop is None or quality >= self._best_quality:
                self._best_crop = crop.copy()
                self._best_quality = quality
            self._expectation = expectation

    def stop(self) -> None:
        self._running = False
        self.wait(2500)

    def run(self) -> None:
        try:
            engine = DateOcrEngine()
            while self._running:
                with self._lock:
                    crop, expectation = self._best_crop, self._expectation
                    self._best_crop = None
                    self._best_quality = -1.0
                if crop is None:
                    self.msleep(40)
                    continue
                self.result_ready.emit(engine.inspect(crop, expectation))
        except Exception as error:
            self.error.emit(f"OCR lỗi: {error}")

    @staticmethod
    def _crop(frame: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
        height, width = frame.shape[:2]
        x, y, roi_width, roi_height = roi
        # Người vận hành vẽ ROI bao quanh hai dòng date.  Chừa viền để không
        # cắt mất NSX/DM khi vị trí vỉ dao động nhẹ trên băng tải.
        # Date của video mẫu nằm lệch khá xa về trái so với vùng người dùng
        # đã khoanh.  Chừa biên ngang rộng để vẫn lấy trọn nhãn khi vỉ rung
        # hoặc trôi ngang qua vị trí kiểm tra.
        pad_x = roi_width * 0.55
        pad_y = roi_height * 0.28
        left = max(0, min(width - 1, round((x - pad_x) * width)))
        top = max(0, min(height - 1, round((y - pad_y) * height)))
        right = max(left + 1, min(width, round((x + roi_width + pad_x) * width)))
        bottom = max(top + 1, min(height, round((y + roi_height + pad_y) * height)))
        return frame[top:bottom, left:right]

    @staticmethod
    def _quality(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Nét chữ và độ tương phản đều cần thiết; chỉ số dùng để chọn frame,
        # không phải độ tin cậy OCR.
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast = float(np.std(gray))
        return sharpness + contrast * 0.35


class DateCheckWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Kiểm tra Date — NSX / DM / giờ")
        self.resize(1450, 900)
        self.settings = SettingsStore(SETTINGS_PATH)
        self.capture: CaptureThread | None = None
        self.ocr: OcrWorker | None = None
        self.roi: tuple[float, float, float, float] | None = None
        self._last_key: tuple[object, ...] | None = None
        self._stable_hits = 0
        self._last_logged_key: tuple[object, ...] | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)

        source = QFrame()
        source.setFrameShape(QFrame.StyledPanel)
        source_layout = QGridLayout(source)
        self.source_type = QComboBox()
        self.source_type.addItem("Camera USB", "usb")
        self.source_type.addItem("RTSP", "rtsp")
        self.source_type.addItem("Tệp video", "video")
        self.source_value = QLineEdit("0")
        self.choose_video = QPushButton("Chọn video…")
        self.start_button = QPushButton("▶ Bắt đầu")
        self.stop_button = QPushButton("■ Dừng")
        self.stop_button.setEnabled(False)
        self.runtime_status = QLabel("Sẵn sàng")
        source_layout.addWidget(QLabel("Nguồn:"), 0, 0)
        source_layout.addWidget(self.source_type, 0, 1)
        source_layout.addWidget(QLabel("Địa chỉ / số camera:"), 0, 2)
        source_layout.addWidget(self.source_value, 0, 3)
        source_layout.addWidget(self.choose_video, 0, 4)
        source_layout.addWidget(self.start_button, 0, 5)
        source_layout.addWidget(self.stop_button, 0, 6)
        source_layout.addWidget(self.runtime_status, 0, 7)
        source_layout.setColumnStretch(3, 1)
        layout.addWidget(source)

        splitter = QSplitter(Qt.Horizontal)
        controls = QWidget()
        controls.setMinimumWidth(340)
        control_layout = QVBoxLayout(controls)
        criterion = QFrame(); criterion.setFrameShape(QFrame.StyledPanel)
        criteria_form = QFormLayout(criterion)
        self.expected_date = QLineEdit(); self.expected_date.setPlaceholderText("Ví dụ: 220826 — để trống: chỉ kiểm tra mẫu")
        self.expected_dm = QLineEdit("DM1")
        self.expected_batch = QLineEdit(); self.expected_batch.setPlaceholderText("Tùy chọn, ví dụ AB")
        saved_expectation = self.settings.expectation()
        self.expected_date.setText(saved_expectation.get("date", ""))
        self.expected_dm.setText(saved_expectation.get("dm") or "DM1")
        self.expected_batch.setText(saved_expectation.get("batch", ""))
        criteria_form.addRow("NSX mong đợi (DDMMYY):", self.expected_date)
        criteria_form.addRow("Mã DM:", self.expected_dm)
        criteria_form.addRow("Mã XX:", self.expected_batch)
        control_layout.addWidget(QLabel("TIÊU CHÍ KIỂM TRA"))
        control_layout.addWidget(criterion)

        roi_box = QFrame(); roi_box.setFrameShape(QFrame.StyledPanel)
        roi_layout = QVBoxLayout(roi_box)
        self.roi_status = QLabel("ROI: chưa cấu hình")
        self.draw_roi = QPushButton("Vẽ ROI DATE trên khung hình")
        self.clear_roi = QPushButton("Xóa ROI của nguồn này")
        roi_layout.addWidget(self.roi_status)
        roi_layout.addWidget(self.draw_roi)
        roi_layout.addWidget(self.clear_roi)
        control_layout.addWidget(QLabel("ROI CỐ ĐỊNH THEO CAMERA"))
        control_layout.addWidget(roi_box)
        control_layout.addStretch(1)
        splitter.addWidget(controls)

        right = QWidget(); right_layout = QVBoxLayout(right)
        self.video = VideoView()
        right_layout.addWidget(self.video, 1)
        self.result_label = QLabel("Chưa có kết quả OCR")
        self.result_label.setWordWrap(True)
        self.result_label.setMinimumHeight(56)
        self.result_label.setStyleSheet("padding:10px; background:#fef3c7; color:#92400e; border:1px solid #f59e0b;")
        right_layout.addWidget(self.result_label)
        self.raw_ocr = QLabel("OCR: --")
        self.raw_ocr.setWordWrap(True)
        right_layout.addWidget(self.raw_ocr)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        metrics = QFrame(); metrics.setFrameShape(QFrame.StyledPanel)
        metric_layout = QHBoxLayout(metrics)
        self.pass_metric = self._metric("Đạt", "0")
        self.fail_metric = self._metric("NG", "0")
        self.review_metric = self._metric("Review", "0")
        for metric in (self.pass_metric, self.fail_metric, self.review_metric):
            metric_layout.addWidget(metric)
        layout.addWidget(metrics)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Thời gian", "Kết quả", "NSX", "DM", "XX", "Giờ", "Ghi chú"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(180)
        layout.addWidget(self.table)
        self.setCentralWidget(root)

        self.source_type.currentIndexChanged.connect(self._on_source_type)
        self.choose_video.clicked.connect(self._choose_video)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.draw_roi.clicked.connect(self.video.begin_draw)
        self.clear_roi.clicked.connect(self._clear_roi)
        self.video.roi_drawn.connect(self._set_roi)
        self.expected_date.editingFinished.connect(self._save_expectation)
        self.expected_dm.editingFinished.connect(self._save_expectation)
        self.expected_batch.editingFinished.connect(self._save_expectation)

    @staticmethod
    def _metric(title: str, value: str) -> QLabel:
        return QLabel(DateCheckWindow._metric_html(title, value))

    @staticmethod
    def _metric_html(title: str, value: str) -> str:
        return f"<div style='text-align:center;color:#94a3b8'>{title}</div><div style='text-align:center;font-size:24px;font-weight:700'>{value}</div>"

    def _on_source_type(self) -> None:
        kind = str(self.source_type.currentData())
        self.choose_video.setVisible(kind == "video")
        if kind == "usb" and not self.source_value.text().strip():
            self.source_value.setText("0")
        self._load_source_roi()

    def _choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Chọn video date", str(Path.home() / "Downloads"), "Video (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self.source_value.setText(path)
            self._load_source_roi()

    def _source_key(self) -> str:
        return f"{self.source_type.currentData()}::{self.source_value.text().strip().casefold()}"

    def _load_source_roi(self) -> None:
        self.roi = self.settings.roi(self._source_key())
        self.video.set_roi(self.roi)
        self._update_roi_label()

    def _set_roi(self, roi: tuple[float, float, float, float]) -> None:
        self.roi = roi
        self.settings.save_roi(self._source_key(), roi)
        self.video.set_roi(roi)
        self._update_roi_label()

    def _clear_roi(self) -> None:
        self.settings.remove_roi(self._source_key())
        self.roi = None
        self.video.set_roi(None)
        self._update_roi_label()

    def _update_roi_label(self) -> None:
        if self.roi is None:
            self.roi_status.setText("ROI: chưa cấu hình — hãy vẽ một lần cho nguồn này")
        else:
            x, y, width, height = self.roi
        self.roi_status.setText(f"ROI đã lưu: x={x:.1%}, y={y:.1%}, rộng={width:.1%}, cao={height:.1%}\nOCR tự lấy thêm viền để không cắt mất date khi vỉ lệch.")

    def _expectation(self) -> DateExpectation:
        return DateExpectation(self.expected_date.text(), self.expected_dm.text(), self.expected_batch.text())

    def _save_expectation(self) -> None:
        self.settings.save_expectation(self._expectation())

    def start(self) -> None:
        self.stop()
        try:
            source = CaptureThread.source_from(str(self.source_type.currentData()), self.source_value.text())
        except ValueError as error:
            QMessageBox.warning(self, "Nguồn chưa hợp lệ", str(error))
            return
        self._load_source_roi()
        self.capture = CaptureThread(source)
        self.ocr = OcrWorker()
        self.capture.frame_ready.connect(self._on_frame)
        self.capture.state_changed.connect(self.runtime_status.setText)
        self.capture.error.connect(self._on_error)
        self.ocr.result_ready.connect(self._on_ocr_result)
        self.ocr.error.connect(self._on_error)
        self.capture.start(); self.ocr.start()
        self.start_button.setEnabled(False); self.stop_button.setEnabled(True)
        self.runtime_status.setText("Đang khởi động OCR…")

    def stop(self) -> None:
        if self.capture is not None:
            self.capture.stop(); self.capture = None
        if self.ocr is not None:
            self.ocr.stop(); self.ocr = None
        self.start_button.setEnabled(True); self.stop_button.setEnabled(False)

    def closeEvent(self, event: Any) -> None:
        self.stop()
        super().closeEvent(event)

    def _on_frame(self, frame: np.ndarray) -> None:
        self.video.set_frame(frame)
        if self.ocr is not None and self.roi is not None:
            self.ocr.submit(frame, self.roi, self._expectation())

    def _on_ocr_result(self, result: OcrResult) -> None:
        self.video.set_result(result.validation)
        self.raw_ocr.setText(f"OCR ({result.confidence:.0%}): {result.text or '--'}")
        validation = result.validation
        palette = {DateStatus.PASS: ("#dcfce7", "#166534"), DateStatus.FAIL: ("#fee2e2", "#991b1b"), DateStatus.REVIEW: ("#fef3c7", "#92400e")}
        background, text = palette[validation.status]
        self.result_label.setStyleSheet(f"padding:10px; background:{background}; color:{text}; border:1px solid {text};")
        read = validation.read
        self.result_label.setText(f"{validation.status}: {validation.detail}\nNSX={read.manufacture_date or '--'} | DM={read.dm_code or '--'} | XX={read.batch_code or '--'} | Giờ={read.time_value or '--'}")
        key = (validation.status, *read.key)
        if key == self._last_key:
            self._stable_hits += 1
        else:
            self._last_key, self._stable_hits = key, 1
        if self._stable_hits >= 2 and key != self._last_logged_key:
            self._last_logged_key = key
            self._append_event(validation)

    def _append_event(self, validation: DateValidation) -> None:
        read = validation.read
        row = self.table.rowCount(); self.table.insertRow(row)
        values = [datetime.now().strftime("%H:%M:%S"), validation.status, read.manufacture_date or "--", read.dm_code or "--", read.batch_code or "--", read.time_value or "--", validation.detail]
        colors = {DateStatus.PASS: QColor("#166534"), DateStatus.FAIL: QColor("#b91c1c"), DateStatus.REVIEW: QColor("#a16207")}
        for column, value in enumerate(values):
            item = QTableWidgetItem(value); item.setForeground(colors[validation.status]); self.table.setItem(row, column, item)
        self._refresh_metrics()

    def _refresh_metrics(self) -> None:
        statuses = [self.table.item(row, 1).text() for row in range(self.table.rowCount())]
        self.pass_metric.setText(self._metric_html("Đạt", str(statuses.count(DateStatus.PASS))))
        self.fail_metric.setText(self._metric_html("NG", str(statuses.count(DateStatus.FAIL))))
        self.review_metric.setText(self._metric_html("Review", str(statuses.count(DateStatus.REVIEW))))

    def _on_error(self, message: str) -> None:
        self.runtime_status.setText(message)
        self.result_label.setText(message)


def run() -> int:
    app = QApplication.instance() or QApplication([])
    window = DateCheckWindow()
    window.show()
    return app.exec()
