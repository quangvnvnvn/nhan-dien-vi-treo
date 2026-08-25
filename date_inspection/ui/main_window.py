from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QFormLayout, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QSlider, QSpinBox, QDoubleSpinBox,
    QTabWidget, QVBoxLayout, QWidget)

from camera.frame_buffer import FrameBuffer
from config.config_manager import ConfigManager
from detection import MotionContourDetector
from processing.frame_reader import FrameReader
from processing.inspection_pipeline import InspectionPipeline
from processing.product_tracker import ProductTracker
from sources import OpenCVCameraSource, VideoFileSource
from .video_view import VideoView


class InspectionWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Kiểm tra DATE trên băng tải — Giai đoạn 1–2")
        self.resize(1480, 900)
        self.config = ConfigManager(); self.reader: FrameReader | None = None
        self.buffer = FrameBuffer(30); self.pipeline: InspectionPipeline | None = None
        self.zone: tuple[float, float, float, float] | None = None
        self.source_key = ""; self.is_video = False; self.last_frame = 0; self.total_frames = 0; self.fps = 30.0
        self.detected = self.duplicates = 0
        self._build_ui(); self._load_detector_values()

    def _build_ui(self) -> None:
        tabs = QTabWidget(); tabs.addTab(self._build_run_page(video=False), "GIÁM SÁT")
        tabs.addTab(self._build_run_page(video=True), "TEST VIDEO")
        tabs.addTab(self._placeholder("Giai đoạn 3: DATE ROI, ring buffer và chọn frame nét nhất."), "TEST ẢNH")
        tabs.addTab(self._build_detection_page(), "PHÁT HIỆN SẢN PHẨM")
        tabs.addTab(self._placeholder("Giai đoạn 4: preprocessing và OCR DATE."), "XỬ LÝ DATE")
        tabs.addTab(self._placeholder("Giai đoạn 5: bảng lịch sử và ảnh kết quả."), "LỊCH SỬ")
        tabs.addTab(self._placeholder("Debug đang hiển thị trực tiếp tại tab TEST VIDEO."), "DEBUG")
        tabs.addTab(self._placeholder("Cấu hình được tự động lưu trong config/settings.json."), "CÀI ĐẶT")
        self.setCentralWidget(tabs)

    def _placeholder(self, text: str) -> QWidget:
        widget = QWidget(); layout = QVBoxLayout(widget); label = QLabel(text); label.setAlignment(Qt.AlignCenter); layout.addWidget(label); return widget

    def _build_run_page(self, video: bool) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page)
        controls = QFrame(); grid = QGridLayout(controls)
        if video:
            self.video_path = QLabel("Chưa chọn file video")
            choose = QPushButton("Chọn video…"); choose.clicked.connect(self.choose_video)
            self.play = QPushButton("▶ Play"); self.pause = QPushButton("❚❚ Pause"); self.stop = QPushButton("■ Stop"); self.restart = QPushButton("↺ Restart")
            self.play.clicked.connect(lambda: self._pause(False)); self.pause.clicked.connect(lambda: self._pause(True)); self.stop.clicked.connect(self.stop_source); self.restart.clicked.connect(lambda: self.seek(0))
            grid.addWidget(choose, 0, 0); grid.addWidget(self.video_path, 0, 1, 1, 3); grid.addWidget(self.play, 0, 4); grid.addWidget(self.pause, 0, 5); grid.addWidget(self.stop, 0, 6); grid.addWidget(self.restart, 0, 7)
            self.seek_slider = QSlider(Qt.Horizontal); self.seek_slider.sliderReleased.connect(lambda: self.seek(self.seek_slider.value()))
            self.frame_label = QLabel("Frame: -- / -- | 00:00 / 00:00")
            self.speed = QComboBox(); self.speed.addItems(["0.25x", "0.5x", "1x", "2x"]); self.speed.setCurrentText("1x"); self.speed.currentTextChanged.connect(self.set_speed)
            for amount in (1, 5, 10):
                button = QPushButton(f"+{amount} frame"); button.clicked.connect(lambda _=False, n=amount: self.step(n)); grid.addWidget(button, 2, 4 + (amount // 5))
            previous = QPushButton("← Frame trước"); previous.clicked.connect(lambda: self.step(-1))
            grid.addWidget(self.seek_slider, 1, 0, 1, 7); grid.addWidget(self.frame_label, 2, 0, 1, 3); grid.addWidget(self.speed, 1, 7); grid.addWidget(previous, 2, 3)
            self.video_view = VideoView(); self.video_view.zone_drawn.connect(self.set_zone)
            layout.addWidget(controls); layout.addWidget(self.video_view, 1)
        else:
            self.camera_combo = QComboBox(); self.camera_combo.addItems([str(i) for i in range(8)])
            start = QPushButton("▶ Bắt đầu camera"); stop = QPushButton("■ Dừng"); start.clicked.connect(self.start_camera); stop.clicked.connect(self.stop_source)
            self.monitor_status = QLabel("Sẵn sàng")
            grid.addWidget(QLabel("Camera Windows:"), 0, 0); grid.addWidget(self.camera_combo, 0, 1); grid.addWidget(start, 0, 2); grid.addWidget(stop, 0, 3); grid.addWidget(self.monitor_status, 0, 4)
            self.monitor_view = VideoView(); self.monitor_view.zone_drawn.connect(self.set_zone)
            layout.addWidget(controls); layout.addWidget(self.monitor_view, 1)
        return page

    def _build_detection_page(self) -> QWidget:
        page = QWidget(); layout = QHBoxLayout(page); panel = QFrame(); form = QFormLayout(panel)
        self.threshold = QSpinBox(); self.threshold.setRange(5, 255)
        self.min_area = QSpinBox(); self.min_area.setRange(1, 10_000_000)
        self.max_area = QSpinBox(); self.max_area.setRange(1, 100_000_000)
        self.occupancy = QDoubleSpinBox(); self.occupancy.setRange(.001, 1); self.occupancy.setDecimals(3); self.occupancy.setSingleStep(.005)
        self.present_frames = QSpinBox(); self.present_frames.setRange(1, 100)
        self.absent_frames = QSpinBox(); self.absent_frames.setRange(1, 100)
        self.debounce = QSpinBox(); self.debounce.setRange(0, 10_000); self.debounce.setSuffix(" ms")
        self.gap = QSpinBox(); self.gap.setRange(0, 1000)
        self.direction = QComboBox(); self.direction.addItems(["trái → phải", "phải → trái", "trên → dưới", "dưới → trên"])
        fields = [("Threshold:", self.threshold), ("Diện tích contour tối thiểu:", self.min_area), ("Diện tích contour tối đa:", self.max_area), ("Occupancy tối thiểu:", self.occupancy), ("Frame có sản phẩm:", self.present_frames), ("Frame sản phẩm đã ra:", self.absent_frames), ("Debounce:", self.debounce), ("Khoảng cách tối thiểu (frame):", self.gap), ("Hướng chạy:", self.direction)]
        for title, control in fields: form.addRow(title, control)
        save = QPushButton("Lưu và áp dụng detector"); save.clicked.connect(self.apply_detector); form.addRow(save)
        draw = QPushButton("Vẽ PRODUCT DETECTION ZONE"); draw.clicked.connect(self.draw_zone); form.addRow(draw)
        self.detector_info = QLabel("Chưa nhận frame."); self.detector_info.setWordWrap(True)
        layout.addWidget(panel); layout.addWidget(self.detector_info, 1)
        return page

    def _load_detector_values(self) -> None:
        values = self.config.detector()
        self.threshold.setValue(int(values.get("threshold", 30))); self.min_area.setValue(int(values.get("min_contour_area", 800))); self.max_area.setValue(int(values.get("max_contour_area", 10_000_000)))
        self.occupancy.setValue(float(values.get("occupancy_threshold", .025))); self.present_frames.setValue(int(values.get("minimum_frames_present", 3))); self.absent_frames.setValue(int(values.get("minimum_frames_absent", 4)))
        self.debounce.setValue(int(values.get("debounce_ms", 300))); self.gap.setValue(int(values.get("minimum_gap_frames", 8))); self.direction.setCurrentText(str(values.get("direction", "trái → phải")))

    def apply_detector(self) -> None:
        values = {"threshold": self.threshold.value(), "min_contour_area": self.min_area.value(), "max_contour_area": self.max_area.value(), "occupancy_threshold": self.occupancy.value(), "minimum_frames_present": self.present_frames.value(), "minimum_frames_absent": self.absent_frames.value(), "debounce_ms": self.debounce.value(), "minimum_gap_frames": self.gap.value(), "sensitivity": 1.0, "direction": self.direction.currentText()}
        self.config.set_detector(values); self._make_pipeline(); self.detector_info.setText("Đã áp dụng. Detector sẽ khởi tạo nền lại ở frame kế tiếp.")

    def _make_pipeline(self) -> None:
        values = self.config.detector()
        detector = MotionContourDetector(**{key: values[key] for key in ("threshold", "min_contour_area", "max_contour_area", "occupancy_threshold", "sensitivity")})
        tracker = ProductTracker(values["minimum_frames_present"], values["minimum_frames_absent"], values["debounce_ms"], values["minimum_gap_frames"])
        self.pipeline = InspectionPipeline(detector, tracker); self.detected = self.duplicates = 0

    def choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Chọn video băng tải", str(Path.home() / "Videos"), "Video (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self.video_path.setText(path); self.start_source(VideoFileSource(path), True, path)

    def start_camera(self) -> None:
        index = int(self.camera_combo.currentText()); self.start_source(OpenCVCameraSource(index), False, f"camera::{index}")

    def start_source(self, source: Any, is_video: bool, source_key: str) -> None:
        self.stop_source(); self.is_video, self.source_key = is_video, source_key
        self.zone = self.config.roi(source_key, "product_detection_zone"); self._make_pipeline()
        self.reader = FrameReader(source, is_video); self.reader.packet_ready.connect(self.on_packet); self.reader.failed.connect(self.on_error); self.reader.ended.connect(self.on_end); self.reader.start()

    def stop_source(self) -> None:
        if self.reader: self.reader.stop(); self.reader = None
        if hasattr(self, "monitor_status"): self.monitor_status.setText("Đã dừng")

    def on_packet(self, packet: Any) -> None:
        self.buffer.append(packet); self.last_frame = packet.frame_number
        view = self.video_view if self.is_video else self.monitor_view; view.set_frame(packet.image)
        if self.pipeline:
            result = self.pipeline.process(packet, self.zone)
            if result:
                if result.event.triggered: self.detected += 1
                if result.event.duplicate_blocked: self.duplicates += 1
                text = f"occupancy: {result.metrics.occupancy_ratio:.2%} | motion: {result.metrics.motion_score:.1f} | contour: {result.metrics.contour_area:.0f} | tổng: {self.detected} | chặn trùng: {self.duplicates}"
                view.set_overlay(self.zone, result.event.state.value, text, result.metrics.contour)
                self.detector_info.setText(text + f"\nTrạng thái: {result.event.state.value}" + (f"\nPRODUCT {result.event.product_id} DETECTED" if result.event.triggered else ""))
        if self.is_video:
            self.total_frames = self.reader.source.frame_count if self.reader else 0; self.fps = self.reader.source.fps if self.reader else 30
            self.seek_slider.setMaximum(max(0, self.total_frames - 1)); self.seek_slider.setValue(packet.frame_number)
            self.frame_label.setText(f"Frame: {packet.frame_number + 1} / {self.total_frames} | {self._time(packet.timestamp)} / {self._time(self.total_frames / self.fps if self.fps else 0)}")
        elif hasattr(self, "monitor_status"):
            self.monitor_status.setText(f"Đang chạy | Frame {packet.frame_number}")

    @staticmethod
    def _time(seconds: float) -> str:
        return f"{int(seconds // 60):02}:{int(seconds % 60):02}"

    def on_end(self) -> None:
        if self.is_video: self.frame_label.setText(self.frame_label.text() + " — Đã hết video")

    def on_error(self, message: str) -> None:
        QMessageBox.warning(self, "Lỗi nguồn", message)

    def _pause(self, value: bool) -> None:
        if self.reader: self.reader.set_paused(value)

    def set_speed(self, label: str) -> None:
        if self.reader: self.reader.set_speed(float(label.rstrip("x")))

    def seek(self, frame: int) -> None:
        if self.reader and self.is_video: self.reader.seek(frame)

    def step(self, amount: int) -> None:
        if self.reader and self.is_video:
            self.reader.set_paused(True); self.reader.seek(max(0, self.last_frame + amount))

    def draw_zone(self) -> None:
        view = self.video_view if self.is_video else self.monitor_view
        view.begin_zone_draw()

    def set_zone(self, zone: tuple[float, float, float, float]) -> None:
        self.zone = zone
        if self.source_key: self.config.set_roi(self.source_key, "product_detection_zone", zone)
        if self.pipeline: self.pipeline.reset()
        view = self.video_view if self.is_video else self.monitor_view; view.set_overlay(zone, "ĐÃ LƯU PRODUCT DETECTION ZONE", "Đợi detector tạo background…")

    def closeEvent(self, event: Any) -> None:
        self.stop_source(); super().closeEvent(event)


def run() -> int:
    app = QApplication.instance() or QApplication([])
    window = InspectionWindow(); window.show(); return app.exec()
