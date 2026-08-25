"""Trang camera thời gian thực, độc lập với driver camera cụ thể.

Module này chỉ chứa giao diện và các hợp đồng tích hợp.  Nó không tự mở
``cv2.VideoCapture`` hay import một CameraThread cụ thể, vì phần cứng/camera
production cần được quản lý trong một worker riêng.  Adapter camera tương lai
nên phát ``frame_ready = Signal(object)`` với một trong các dạng sau:

* ``QImage`` hoặc ``QPixmap`` để chỉ hiển thị;
* :class:`FramePacket` khi cần giữ cả khung hình gốc và preview;
* một mapping ``{"frame": raw_frame, "preview": QImage, ...}``.

Sau đó gọi :meth:`RealtimePage.attach_camera_thread` và
:meth:`RealtimePage.set_pipeline`.  Mọi lỗi từ camera/pipeline đều được bắt
và hiển thị trên giao diện, không làm rơi ứng dụng desktop.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, TYPE_CHECKING
import time

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from core.models import ProductProfile, SlotSpec

if TYPE_CHECKING:
    from core.colors import ColorCatalog
    from training.product_manager import ProductManager


NormalizedRect = tuple[float, float, float, float]
NormalizedLine = tuple[tuple[float, float], tuple[float, float]]
ManualSlotPoint = tuple[float, float, str]
# ``sample_radius`` is normalized to the full frame when set.  ``None`` keeps
# the compact marker used for legacy automatic sampling.
ManualSlotMarker = tuple[int, int, tuple[float, float], str, float | None]


@dataclass(frozen=True, slots=True)
class CameraStartRequest:
    """Cấu hình UI gửi cho controller/camera worker khi người dùng bấm bắt đầu."""

    source: str
    source_type: str
    expected_product_id: str | None
    roi: NormalizedRect | None
    counting_line: NormalizedLine | None
    counting_direction: str = "top_to_bottom"
    # Băng tải chạy/dừng dùng capture ổn định để tránh kết luận trên ảnh mờ.
    inspection_mode: str = "on_stop"
    # Camera/góc vỉ cố định: đối chiếu màu tại từng vị trí slot của profile.
    position_locked_color: bool = True


@dataclass(frozen=True, slots=True)
class RealtimeContext:
    """Ngữ cảnh bất biến đi kèm mỗi frame khi gọi pipeline suy luận."""

    frame_index: int
    captured_at: datetime
    source: str
    source_type: str
    expected_product_id: str | None
    roi: NormalizedRect | None
    counting_line: NormalizedLine | None


@dataclass(slots=True)
class FramePacket:
    """Gói frame tùy chọn cho adapter camera.

    ``frame`` có thể là ndarray hoặc đối tượng native của pipeline.  Giao diện
    chỉ render ``preview`` khi đó là QImage/QPixmap; nhờ vậy nó không phụ thuộc
    OpenCV hoặc NumPy.
    """

    frame: object
    preview: QImage | QPixmap | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RealtimeInference:
    """Kết quả tối thiểu pipeline có thể trả về cho dashboard."""

    status: str = "UNKNOWN"
    product_id: str | None = None
    detected_colors: Sequence[str] = field(default_factory=tuple)
    confidence: float = 0.0
    count_delta: int = 0
    detected_delta: int = 0
    pass_delta: int = 0
    fail_delta: int = 0
    unknown_delta: int = 0
    detail: str = "Chưa có kết quả suy luận"
    annotated_frame: QImage | QPixmap | None = None
    fps: float | None = None
    visible_products: int = 0
    fault_products: int = 0
    alert_active: bool = False
    alert_detail: str = ""


class CameraThreadProtocol(Protocol):
    """Hợp đồng tối thiểu của CameraThread tương lai, không phụ thuộc driver."""

    frame_ready: Any

    def start(self) -> None: ...

    def stop(self) -> None: ...


FramePipeline = Callable[[object, RealtimeContext], RealtimeInference | Mapping[str, Any] | None]


class VideoSurface(QLabel):
    """QLabel khung video có thể vẽ ROI và đường đếm bằng tọa độ chuẩn hóa 0..1."""

    roi_drawn = Signal(object)
    line_drawn = Signal(object)
    manual_slot_chosen = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(640, 420)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self._pixmap: QPixmap | None = None
        self._roi: NormalizedRect | None = None
        self._line: NormalizedLine | None = None
        self._draw_mode: str | None = None
        self._drag_start: QPointF | None = None
        self._drag_current: QPointF | None = None
        self._manual_slot_markers: list[ManualSlotMarker] = []

    @property
    def draw_mode(self) -> str | None:
        return self._draw_mode

    def set_frame(self, pixmap: QPixmap | None) -> None:
        self._pixmap = pixmap
        self.update()

    def set_roi(self, roi: NormalizedRect | None) -> None:
        self._roi = roi
        self.update()

    def set_counting_line(self, line: NormalizedLine | None) -> None:
        self._line = line
        self.update()

    def set_manual_slot_markers(self, markers: Sequence[ManualSlotMarker]) -> None:
        """Vẽ các tâm slot do người vận hành đã chốt thủ công."""
        self._manual_slot_markers = list(markers)
        self.update()

    def set_draw_mode(self, mode: str | None) -> None:
        if mode not in {None, "roi", "line", "slot"}:
            raise ValueError("Chế độ vẽ chỉ có thể là roi, line, slot hoặc None")
        self._draw_mode = mode
        self._drag_start = None
        self._drag_current = None
        cursor = Qt.CursorShape.CrossCursor if mode else Qt.CursorShape.ArrowCursor
        self.setCursor(cursor)
        self.update()

    def _image_rect(self) -> QRectF:
        if self._pixmap is None or self._pixmap.isNull():
            return QRectF()
        scaled = self._pixmap.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        return QRectF(
            (self.width() - scaled.width()) / 2,
            (self.height() - scaled.height()) / 2,
            scaled.width(),
            scaled.height(),
        )

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _to_normalized(self, point: QPointF) -> QPointF | None:
        rect = self._image_rect()
        if rect.isEmpty() or not rect.contains(point):
            return None
        return QPointF(
            self._clamp((point.x() - rect.left()) / rect.width()),
            self._clamp((point.y() - rect.top()) / rect.height()),
        )

    def _from_normalized(self, point: tuple[float, float]) -> QPointF | None:
        rect = self._image_rect()
        if rect.isEmpty():
            return None
        return QPointF(rect.left() + point[0] * rect.width(), rect.top() + point[1] * rect.height())

    def _normalized_rect_to_widget(self, roi: NormalizedRect) -> QRectF | None:
        first = self._from_normalized((roi[0], roi[1]))
        second = self._from_normalized((roi[0] + roi[2], roi[1] + roi[3]))
        if first is None or second is None:
            return None
        return QRectF(first, second).normalized()

    def paintEvent(self, _event: object) -> None:  # noqa: N802 - Qt API name
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111827"))
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        image_rect = self._image_rect()
        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QColor("#cbd5e1"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "CHƯA CÓ HÌNH ẢNH CAMERA\nNhập nguồn camera rồi bấm BẮT ĐẦU",
            )
        else:
            painter.drawPixmap(image_rect.toRect(), self._pixmap)

        if self._roi is not None:
            widget_roi = self._normalized_rect_to_widget(self._roi)
            if widget_roi is not None:
                painter.setPen(QPen(QColor("#22c55e"), 2, Qt.PenStyle.DashLine))
                painter.drawRect(widget_roi)
                painter.setPen(QColor("#dcfce7"))
                painter.drawText(widget_roi.topLeft() + QPointF(6, 18), "ROI")

        if self._line is not None:
            start = self._from_normalized(self._line[0])
            end = self._from_normalized(self._line[1])
            if start is not None and end is not None:
                painter.setPen(QPen(QColor("#f97316"), 3))
                painter.drawLine(start, end)
                painter.setPen(QColor("#ffedd5"))
                painter.drawText(start + QPointF(6, -6), "ĐƯỜNG ĐẾM")

        color_map = {
            "purple": "#c084fc", "blue": "#38bdf8", "green": "#4ade80",
        }
        image_rect = self._image_rect()
        for strip_index, slot_index, normalized, color, sample_radius in self._manual_slot_markers:
            point = self._from_normalized(normalized)
            if point is None:
                continue
            marker_color = QColor(color_map.get(color, "#facc15"))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(marker_color, 3))
            radius = 10.0
            if sample_radius is not None and image_rect is not None:
                radius = max(6.0, sample_radius * min(image_rect.width(), image_rect.height()))
            painter.drawEllipse(point, radius, radius)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(point + QPointF(radius + 3, -radius), f"V{strip_index}-S{slot_index}")

        if self._drag_start is not None and self._drag_current is not None:
            if self._draw_mode == "roi":
                painter.setPen(QPen(QColor("#38bdf8"), 2, Qt.PenStyle.DashLine))
                painter.drawRect(QRectF(self._drag_start, self._drag_current).normalized())
            elif self._draw_mode == "line":
                painter.setPen(QPen(QColor("#facc15"), 3, Qt.PenStyle.DashLine))
                painter.drawLine(self._drag_start, self._drag_current)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API name
        if event.button() == Qt.MouseButton.LeftButton and self._draw_mode:
            normalized = self._to_normalized(event.position())
            if normalized is not None:
                if self._draw_mode == "slot":
                    self.manual_slot_chosen.emit((normalized.x(), normalized.y()))
                    event.accept()
                    return
                self._drag_start = event.position()
                self._drag_current = event.position()
                event.accept()
                self.update()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API name
        if self._drag_start is not None and self._draw_mode:
            normalized = self._to_normalized(event.position())
            if normalized is not None:
                self._drag_current = event.position()
                event.accept()
                self.update()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API name
        if event.button() != Qt.MouseButton.LeftButton or self._drag_start is None or not self._draw_mode:
            super().mouseReleaseEvent(event)
            return
        start = self._to_normalized(self._drag_start)
        end = self._to_normalized(event.position())
        mode = self._draw_mode
        self._drag_start = None
        self._drag_current = None
        self.set_draw_mode(None)
        if start is None or end is None:
            return
        if mode == "roi":
            left, right = sorted((start.x(), end.x()))
            top, bottom = sorted((start.y(), end.y()))
            if (right - left) >= 0.02 and (bottom - top) >= 0.02:
                self.roi_drawn.emit((left, top, right - left, bottom - top))
        elif mode == "line":
            distance = abs(start.x() - end.x()) + abs(start.y() - end.y())
            if distance >= 0.04:
                self.line_drawn.emit(((start.x(), start.y()), (end.x(), end.y())))


class RealtimePage(QWidget):
    """Dashboard camera thời gian thực với các extension hook rõ ràng.

    Production controller chạy camera/AI ở worker riêng và chỉ phát ``QImage``
    đã copy + :class:`RealtimeInference` qua signal về lớp này.
    """

    start_requested = Signal(object)  # CameraStartRequest
    stop_requested = Signal()
    roi_changed = Signal(object)  # NormalizedRect | None
    counting_line_changed = Signal(object)  # NormalizedLine | None
    counting_direction_changed = Signal(str)
    product_profile_changed = Signal(object)  # str | None
    profiles_changed = Signal()
    inference_received = Signal(object)  # RealtimeInference

    def __init__(
        self,
        manager: ProductManager | None = None,
        colors_catalog: ColorCatalog | None = None,
    ) -> None:
        super().__init__()
        self._manager = manager
        self._colors_catalog = colors_catalog
        self._camera_thread: object | None = None
        self._camera_connections: list[tuple[object, object]] = []
        self._pipeline: FramePipeline | None = None
        self._roi: NormalizedRect | None = None
        self._counting_line: NormalizedLine | None = None
        self._frame_index = 0
        self._frame_times: deque[float] = deque(maxlen=40)
        self._last_alert_signature = ""
        self._alert_active = False
        self._alert_phase = False
        self._alert_hold_until = 0.0
        self._alert_timer = QTimer(self)
        self._alert_timer.setInterval(280)
        self._alert_timer.timeout.connect(self._toggle_alert_flash)
        self._export_directory: Path | None = None
        self._manual_working_strips: list[list[ManualSlotPoint]] = []
        self._active_manual_strip: list[ManualSlotPoint] | None = None
        self._manual_reference_roi: NormalizedRect | None = None
        self._build()
        self.profile_selector.currentIndexChanged.connect(self._emit_product_profile_changed)
        self.profile_selector.currentIndexChanged.connect(self._load_manual_color_locks)
        self.refresh_profiles()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        heading = QHBoxLayout()
        title = QLabel("CAMERA THỜI GIAN THỰC")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        self.camera_state = QLabel("● CHƯA KẾT NỐI")
        self.camera_state.setStyleSheet("color: #94a3b8; font-weight: 700;")
        heading.addWidget(title)
        heading.addStretch()
        heading.addWidget(self.camera_state)
        root.addLayout(heading)

        source_box = QGroupBox("Nguồn camera")
        source_layout = QHBoxLayout(source_box)
        self.source_type = QComboBox()
        self.source_type.addItem("Camera USB", "usb")
        self.source_type.addItem("Luồng RTSP", "stream")
        self.source_type.addItem("Tệp video", "video")
        self.source_type.currentIndexChanged.connect(self._update_source_hint)
        self.source_input = QLineEdit("0")
        self.source_input.setMinimumWidth(280)
        self.choose_video_button = QPushButton("CHỌN VIDEO…")
        self.choose_video_button.clicked.connect(self._choose_video_file)
        self.start_button = QPushButton("▶ BẮT ĐẦU")
        self.stop_button = QPushButton("■ DỪNG")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        source_layout.addWidget(QLabel("Loại nguồn:"))
        source_layout.addWidget(self.source_type)
        source_layout.addWidget(QLabel("Nguồn:"))
        source_layout.addWidget(self.source_input, 1)
        source_layout.addWidget(self.choose_video_button)
        source_layout.addWidget(self.start_button)
        source_layout.addWidget(self.stop_button)
        root.addWidget(source_box)
        self._update_source_hint()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_control_panel())
        splitter.addWidget(self._build_video_panel())
        splitter.setSizes([350, 930])
        root.addWidget(splitter, 1)

        root.addWidget(self._build_metrics())
        root.addWidget(self._build_event_table(), 1)
        self._set_status("Sẵn sàng. Gắn CameraThread hoặc dùng start_requested để kết nối camera.")

    def _build_control_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        product_box = QGroupBox("Sản phẩm kiểm tra")
        product_form = QFormLayout(product_box)
        self.profile_selector = QComboBox()
        refresh_button = QPushButton("↻ TẢI PROFILE")
        refresh_button.clicked.connect(self.refresh_profiles)
        product_form.addRow("Product Profile:", self.profile_selector)
        product_form.addRow("", refresh_button)
        layout.addWidget(product_box)

        manual_lock_box = QGroupBox("Khóa màu thủ công theo từng slot")
        manual_lock_layout = QVBoxLayout(manual_lock_box)
        self.manual_lock_hint = QLabel(
            "Chọn Product Profile, sau đó tự chốt màu cho từng slot của vỉ mẫu. "
            "Cấu hình lưu vào profile và được áp dụng cho mọi vỉ cùng loại trong ROI."
        )
        self.manual_lock_hint.setWordWrap(True)
        self.manual_lock_hint.setStyleSheet("color: #bfdbfe;")
        self.manual_slot_form = QFormLayout()
        self.manual_slot_selectors: list[QComboBox] = []
        self.save_manual_lock_button = QPushButton("LƯU KHÓA MÀU TỪNG SLOT")
        self.save_manual_lock_button.clicked.connect(self._save_manual_color_locks)
        manual_lock_layout.addWidget(self.manual_lock_hint)
        manual_lock_layout.addLayout(self.manual_slot_form)
        manual_lock_layout.addWidget(self.save_manual_lock_button)
        layout.addWidget(manual_lock_box)

        manual_position_box = QGroupBox("Khóa vị trí quét thủ công theo từng vỉ")
        manual_position_layout = QVBoxLayout(manual_position_box)
        self.manual_position_hint = QLabel(
            "Dùng khi camera và ROI cố định: bấm CHỐT VỈ MỚI, rồi click lần lượt tâm "
            "Slot 1 đến Slot cuối trên khung hình. Có thể chốt nhiều vỉ trong cùng ROI."
        )
        self.manual_position_hint.setWordWrap(True)
        self.manual_position_hint.setStyleSheet("color: #fde68a;")
        self.manual_position_progress = QLabel("Chưa có vị trí vỉ được chốt.")
        self.manual_position_progress.setWordWrap(True)
        self.manual_position_color = QComboBox()
        self.manual_sample_radius = QDoubleSpinBox()
        self.manual_sample_radius.setRange(0.0, 15.0)
        self.manual_sample_radius.setDecimals(1)
        self.manual_sample_radius.setSingleStep(0.5)
        self.manual_sample_radius.setSuffix(" % ROI")
        self.manual_sample_radius.setSpecialValueText("Tự động")
        self.manual_sample_radius.setToolTip(
            "Bán kính vòng tròn lấy mẫu màu theo cạnh ngắn ROI. "
            "Đặt 0 để giữ chế độ tự động cũ; vòng quá lớn có thể lẫn màu slot bên cạnh."
        )
        self.manual_sample_radius.valueChanged.connect(lambda _value: self._refresh_manual_slot_markers())
        position_form = QFormLayout()
        position_form.addRow("Màu slot đang chọn:", self.manual_position_color)
        position_form.addRow("Bán kính vòng lấy màu:", self.manual_sample_radius)
        manual_position_layout.addWidget(self.manual_position_hint)
        manual_position_layout.addLayout(position_form)
        sampling_note = QLabel("0 = tự động. Thử 3–8% ROI và quan sát vòng tròn trên ảnh trước khi lưu.")
        sampling_note.setWordWrap(True)
        sampling_note.setStyleSheet("color: #94a3b8;")
        manual_position_layout.addWidget(sampling_note)
        manual_position_layout.addWidget(self.manual_position_progress)
        position_buttons = QHBoxLayout()
        self.begin_manual_strip_button = QPushButton("CHỐT VỈ MỚI")
        self.begin_manual_strip_button.clicked.connect(self._begin_manual_strip)
        self.undo_manual_slot_button = QPushButton("LÙI 1 SLOT")
        self.undo_manual_slot_button.clicked.connect(self._undo_manual_slot)
        position_buttons.addWidget(self.begin_manual_strip_button)
        position_buttons.addWidget(self.undo_manual_slot_button)
        manual_position_layout.addLayout(position_buttons)
        layout_buttons = QHBoxLayout()
        self.remove_manual_strip_button = QPushButton("XÓA VỈ CUỐI")
        self.remove_manual_strip_button.clicked.connect(self._remove_last_manual_strip)
        self.clear_manual_layout_button = QPushButton("XÓA TẤT CẢ VỊ TRÍ")
        self.clear_manual_layout_button.clicked.connect(self._clear_manual_layout)
        layout_buttons.addWidget(self.remove_manual_strip_button)
        layout_buttons.addWidget(self.clear_manual_layout_button)
        manual_position_layout.addLayout(layout_buttons)
        self.save_manual_layout_button = QPushButton("LƯU VỊ TRÍ VỈ THEO PROFILE")
        self.save_manual_layout_button.clicked.connect(self._save_manual_scan_layout)
        manual_position_layout.addWidget(self.save_manual_layout_button)
        layout.addWidget(manual_position_box)

        mode_box = QGroupBox("Chế độ quét")
        mode_layout = QVBoxLayout(mode_box)
        self.inspection_mode = QComboBox()
        self.inspection_mode.addItem("Chụp khi băng tải dừng (khuyến nghị)", "on_stop")
        self.inspection_mode.addItem("Quét liên tục realtime", "realtime")
        mode_note = QLabel(
            "Chế độ chụp khi dừng sẽ chờ ảnh ổn định, chọn khung nét nhất và "
            "quét một lần cho mỗi nhịp dừng."
        )
        mode_note.setWordWrap(True)
        mode_note.setStyleSheet("color: #94a3b8;")
        mode_layout.addWidget(self.inspection_mode)
        mode_layout.addWidget(mode_note)
        self.position_locked_color = QCheckBox("Khóa màu theo từng vị trí slot (camera/góc vỉ cố định)")
        self.position_locked_color.setChecked(True)
        self.position_locked_color.setToolTip(
            "Mỗi slot phải có màu được đo trực tiếp tại tâm của nó. "
            "Tắt mục này nếu thay camera hoặc thay đổi góc lắp đặt."
        )
        mode_layout.addWidget(self.position_locked_color)
        layout.addWidget(mode_box)

        roi_box = QGroupBox("Vùng quan sát (ROI)")
        roi_layout = QVBoxLayout(roi_box)
        self.roi_summary = QLabel("Chưa chọn ROI — pipeline sẽ nhận toàn bộ khung hình.")
        self.roi_summary.setWordWrap(True)
        self.draw_roi_button = QPushButton("VẼ ROI TRÊN KHUNG HÌNH")
        self.draw_roi_button.setCheckable(True)
        self.draw_roi_button.toggled.connect(self._toggle_roi_draw)
        clear_roi_button = QPushButton("XÓA ROI")
        clear_roi_button.clicked.connect(self.clear_roi)
        roi_layout.addWidget(self.roi_summary)
        roi_layout.addWidget(self.draw_roi_button)
        roi_layout.addWidget(clear_roi_button)
        layout.addWidget(roi_box)

        note = QLabel(
            "Chế độ giám sát ROI: có thể đặt nhiều vỉ trong cùng vùng. Hệ thống "
            "không đếm vỉ; khi thấy vỉ sai màu hoặc thiếu slot, banner đỏ sẽ nhấp nháy ngay."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #94a3b8; padding: 6px;")
        layout.addWidget(note)

        export_box = QGroupBox("Tra cứu kết quả Excel")
        export_layout = QVBoxLayout(export_box)
        self.excel_summary = QLabel("Dữ liệu Excel lịch sử vẫn có thể tra cứu theo ngày. Chế độ này chỉ giám sát cảnh báo, không cộng số lượng vỉ.")
        self.excel_summary.setWordWrap(True)
        self.open_excel_folder_button = QPushButton("MỞ THƯ MỤC EXCEL")
        self.open_excel_folder_button.setEnabled(False)
        self.open_excel_folder_button.clicked.connect(self.open_export_directory)
        export_layout.addWidget(self.excel_summary)
        export_layout.addWidget(self.open_excel_folder_button)
        layout.addWidget(export_box)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(container)
        return scroll

    def _build_video_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.alert_banner = QLabel("✓ GIÁM SÁT BÌNH THƯỜNG — CHƯA PHÁT HIỆN VỈ LỖI")
        self.alert_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.alert_banner.setMinimumHeight(54)
        self.alert_banner.setWordWrap(True)
        self._apply_alert_banner_style()
        layout.addWidget(self.alert_banner)
        self.video_surface = VideoSurface()
        self.video_surface.roi_drawn.connect(self._set_roi_from_surface)
        self.video_surface.manual_slot_chosen.connect(self._capture_manual_slot_position)
        self.video_surface.setStyleSheet("border: 1px solid #334155; border-radius: 6px;")
        layout.addWidget(self.video_surface, 1)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("padding: 8px; border: 1px solid #334155; border-radius: 4px;")
        layout.addWidget(self.status_label)
        return panel

    def _build_metrics(self) -> QWidget:
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        grid = QGridLayout(panel)
        self.metric_values: dict[str, QLabel] = {}
        for column, (key, label, value) in enumerate((
            ("fps", "Tốc độ", "0.0 FPS"),
            ("visible", "Vỉ trong ROI", "0"),
            ("faults", "Vỉ lỗi hiện tại", "0"),
            ("alert", "Cảnh báo", "BÌNH THƯỜNG"),
        )):
            title = QLabel(label)
            title.setStyleSheet("color: #94a3b8;")
            number = QLabel(value)
            number.setStyleSheet("font-size: 21px; font-weight: 700;")
            grid.addWidget(title, 0, column, Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(number, 1, column, Qt.AlignmentFlag.AlignCenter)
            self.metric_values[key] = number
        return panel

    def _build_event_table(self) -> QTableWidget:
        self.event_table = QTableWidget(0, 7)
        self.event_table.setHorizontalHeaderLabels((
            "Thời gian", "Sản phẩm", "Kết quả", "Màu phát hiện", "Tin cậy", "Vỉ lỗi", "Ghi chú",
        ))
        self.event_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.event_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.event_table.setAlternatingRowColors(True)
        self.event_table.horizontalHeader().setStretchLastSection(True)
        return self.event_table

    def _update_source_hint(self) -> None:
        source_type = self.source_type.currentData()
        self.choose_video_button.setVisible(source_type == "video")
        if source_type == "usb":
            self.source_input.setPlaceholderText("Ví dụ: 0 hoặc 1")
            if not self.source_input.text().strip():
                self.source_input.setText("0")
        elif source_type == "video":
            self.source_input.setPlaceholderText("Chọn tệp video để kiểm tra")
            if self.source_input.text().strip() == "0":
                self.source_input.clear()
        else:
            self.source_input.setPlaceholderText("rtsp://... hoặc http://...")

    def _choose_video_file(self) -> None:
        """Chọn video cục bộ để chạy cùng pipeline như luồng camera."""
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Chọn video kiểm tra",
            self.source_input.text().strip() or "",
            "Tệp video (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v);;Tất cả tệp (*.*)",
        )
        if not path:
            return
        self.source_input.setText(path)
        self._set_status(
            f"Đã chọn video: {Path(path).name}. Chọn Product Profile rồi bấm BẮT ĐẦU.",
            level="info",
        )

    def refresh_profiles(self) -> None:
        """Nạp profile mới mà không buộc UI phụ thuộc ProductManager cụ thể."""
        previous = self.profile_selector.currentData()
        self.profile_selector.clear()
        self.profile_selector.addItem("-- Không chọn profile --", None)
        if self._manager is not None:
            try:
                profiles = self._manager.list_profiles()
            except Exception as error:  # dữ liệu cấu hình lỗi không được làm sập camera UI
                self._set_status(f"Không tải được Product Profile: {error}", level="warning")
                profiles = []
            display = getattr(self._colors_catalog, "display", lambda color: color)
            for profile in profiles:
                colors = " - ".join(display(color) for color in profile.expected_colors)
                self.profile_selector.addItem(
                    f"{profile.product_id} — {profile.name} ({colors})", profile.product_id
                )
        index = self.profile_selector.findData(previous)
        self.profile_selector.setCurrentIndex(index if index >= 0 else 0)
        self._load_manual_color_locks()

    def _load_manual_position_layout(self) -> None:
        """Tải tọa độ đã lưu từ profile để người dùng xem/chỉnh tiếp."""
        self._manual_working_strips.clear()
        self._active_manual_strip = None
        product_id = self.profile_selector.currentData()
        profile = self._manager.get(product_id) if isinstance(product_id, str) and self._manager else None
        if profile is None:
            self._manual_reference_roi = None
            self.manual_sample_radius.blockSignals(True)
            self.manual_sample_radius.setValue(0.0)
            self.manual_sample_radius.blockSignals(False)
            self._refresh_manual_position_controls()
            self._refresh_manual_slot_markers()
            return
        self._manual_reference_roi = profile.manual_scan_roi
        self.manual_sample_radius.blockSignals(True)
        self.manual_sample_radius.setValue((profile.manual_scan_sample_radius or 0.0) * 100.0)
        self.manual_sample_radius.blockSignals(False)
        for strip in sorted(profile.manual_scan_strips, key=lambda item: item.index):
            points: list[ManualSlotPoint] = []
            for slot in sorted(strip.slots, key=lambda item: item.index):
                x, y = self._manual_position_to_surface(slot.x, slot.y, profile.manual_scan_roi)
                points.append((x, y, slot.expected_color))
            if points:
                self._manual_working_strips.append(points)
        self._refresh_manual_position_controls()
        self._refresh_manual_slot_markers()

    def _refresh_manual_position_controls(self) -> None:
        product_id = self.profile_selector.currentData()
        profile = self._manager.get(product_id) if isinstance(product_id, str) and self._manager else None
        catalog_profiles = getattr(self._colors_catalog, "profiles", {})
        self.manual_position_color.blockSignals(True)
        self.manual_position_color.clear()
        display = getattr(self._colors_catalog, "display", lambda color: color)
        for color in catalog_profiles:
            self.manual_position_color.addItem(str(display(color)), color)
        self.manual_position_color.blockSignals(False)
        enabled = profile is not None and bool(catalog_profiles)
        for widget in (
            self.manual_position_color, self.manual_sample_radius, self.begin_manual_strip_button,
            self.undo_manual_slot_button, self.remove_manual_strip_button,
            self.clear_manual_layout_button, self.save_manual_layout_button,
        ):
            widget.setEnabled(enabled)
        if profile is None:
            self.manual_position_progress.setText("Chọn Product Profile trước khi chốt vị trí vỉ.")
            return
        if self._active_manual_strip is not None:
            next_slot = len(self._active_manual_strip) + 1
            self._select_default_manual_slot_color(profile, next_slot)
            self.manual_position_progress.setText(
                f"Đang chốt Vỉ {len(self._manual_working_strips) + 1}: click tâm Slot "
                f"{next_slot}/{len(profile.slots)} trên khung hình."
            )
            return
        source_note = "đã lưu" if self._manual_working_strips else "chưa lưu"
        self.manual_position_progress.setText(
            f"Đã chốt {len(self._manual_working_strips)} vỉ ({source_note}). "
            "Bấm CHỐT VỈ MỚI để chọn vỉ tiếp theo."
        )

    def _select_default_manual_slot_color(self, profile: ProductProfile, slot_number: int) -> None:
        """Ưu tiên màu profile cho slot tiếp theo, nhưng vẫn cho phép đổi từng vỉ."""
        if not 1 <= slot_number <= len(profile.slots):
            return
        expected = sorted(profile.slots, key=lambda item: item.index)[slot_number - 1].expected_color
        index = self.manual_position_color.findData(expected)
        if index >= 0:
            self.manual_position_color.setCurrentIndex(index)

    @staticmethod
    def _manual_position_to_surface(
        x: float,
        y: float,
        roi: NormalizedRect | None,
    ) -> tuple[float, float]:
        if roi is None:
            return x, y
        return roi[0] + x * roi[2], roi[1] + y * roi[3]

    @staticmethod
    def _surface_position_to_manual(
        x: float,
        y: float,
        roi: NormalizedRect,
    ) -> tuple[float, float] | None:
        if roi[2] <= 0.0 or roi[3] <= 0.0:
            return None
        local_x = (x - roi[0]) / roi[2]
        local_y = (y - roi[1]) / roi[3]
        if not (-0.002 <= local_x <= 1.002 and -0.002 <= local_y <= 1.002):
            return None
        return max(0.0, min(1.0, local_x)), max(0.0, min(1.0, local_y))

    def _begin_manual_strip(self) -> None:
        product_id = self.profile_selector.currentData()
        profile = self._manager.get(product_id) if isinstance(product_id, str) and self._manager else None
        if profile is None:
            QMessageBox.warning(self, "Chưa chọn profile", "Chọn Product Profile trước khi chốt vị trí vỉ.")
            return
        if self._roi is None:
            QMessageBox.warning(
                self,
                "Cần ROI cố định",
                "Vẽ ROI trước. Tọa độ slot sẽ được lưu theo ROI này để không quét nhầm khi camera cố định.",
            )
            return
        if self._active_manual_strip is not None:
            self._set_status("Đang chốt một vỉ; click đủ các slot hoặc dùng LÙI 1 SLOT.", level="warning")
            return
        if (
            self._manual_working_strips
            and self._manual_reference_roi is not None
            and not self._same_roi(self._roi, self._manual_reference_roi)
        ):
            QMessageBox.warning(
                self,
                "ROI đã thay đổi",
                "Các vỉ đang chốt thuộc ROI cũ. Bấm XÓA TẤT CẢ VỊ TRÍ rồi chốt lại theo ROI mới.",
            )
            return
        self._active_manual_strip = []
        self._manual_reference_roi = self._roi
        self.video_surface.set_draw_mode("slot")
        self._refresh_manual_position_controls()
        self._set_status(
            f"Chọn lần lượt tâm Slot 1 đến Slot {len(profile.slots)} cho Vỉ {len(self._manual_working_strips) + 1}.",
            level="info",
        )

    def _capture_manual_slot_position(self, point: object) -> None:
        if self._active_manual_strip is None or not isinstance(point, tuple) or len(point) != 2:
            return
        product_id = self.profile_selector.currentData()
        profile = self._manager.get(product_id) if isinstance(product_id, str) and self._manager else None
        color = self.manual_position_color.currentData()
        if profile is None or not isinstance(color, str):
            return
        x, y = float(point[0]), float(point[1])
        if self._roi is None or self._surface_position_to_manual(x, y, self._roi) is None:
            self._set_status("Điểm slot phải nằm trong ROI đã chọn.", level="warning")
            return
        self._active_manual_strip.append((x, y, color))
        if len(self._active_manual_strip) >= len(profile.slots):
            self._manual_working_strips.append(self._active_manual_strip)
            self._active_manual_strip = None
            self.video_surface.set_draw_mode(None)
            self._set_status(
                f"Đã chốt Vỉ {len(self._manual_working_strips)}. Bấm CHỐT VỈ MỚI để thêm vỉ khác, rồi LƯU.",
                level="success",
            )
        self._refresh_manual_position_controls()
        self._refresh_manual_slot_markers()

    def _undo_manual_slot(self) -> None:
        if self._active_manual_strip:
            self._active_manual_strip.pop()
            self._refresh_manual_position_controls()
            self._refresh_manual_slot_markers()
            return
        self._set_status("Chưa có slot đang chốt để lùi.", level="warning")

    def _remove_last_manual_strip(self) -> None:
        if self._active_manual_strip is not None:
            self._set_status("Hãy hoàn tất hoặc lùi các slot của vỉ đang chốt trước.", level="warning")
            return
        if not self._manual_working_strips:
            self._set_status("Chưa có vỉ nào để xóa.", level="warning")
            return
        self._manual_working_strips.pop()
        self._refresh_manual_position_controls()
        self._refresh_manual_slot_markers()

    def _clear_manual_layout(self) -> None:
        self._manual_working_strips.clear()
        self._active_manual_strip = None
        self._manual_reference_roi = self._roi
        self.video_surface.set_draw_mode(None)
        self._refresh_manual_position_controls()
        self._refresh_manual_slot_markers()
        self._set_status("Đã xóa các vị trí thủ công trong màn hình. Bấm LƯU để xóa khỏi Product Profile.", level="info")

    def _refresh_manual_slot_markers(self) -> None:
        sample_radius = self._manual_preview_sample_radius()
        markers: list[ManualSlotMarker] = []
        for strip_index, slots in enumerate(self._manual_working_strips, start=1):
            markers.extend(
                (strip_index, slot_index, (x, y), color, sample_radius)
                for slot_index, (x, y, color) in enumerate(slots, start=1)
            )
        if self._active_manual_strip is not None:
            active_index = len(self._manual_working_strips) + 1
            markers.extend(
                (active_index, slot_index, (x, y), color, sample_radius)
                for slot_index, (x, y, color) in enumerate(self._active_manual_strip, start=1)
            )
        self.video_surface.set_manual_slot_markers(markers)

    def _manual_sample_radius_fraction(self) -> float | None:
        """Trả về bán kính đặt tay theo cạnh ngắn ROI, hoặc None là tự động."""
        value = self.manual_sample_radius.value() / 100.0
        return value if value >= 0.005 else None

    def _manual_preview_sample_radius(self) -> float | None:
        """Đổi bán kính theo ROI sang tỷ lệ toàn khung để VideoSurface vẽ preview."""
        radius = self._manual_sample_radius_fraction()
        roi = self._roi or self._manual_reference_roi
        if radius is None or roi is None:
            return None
        return radius * min(roi[2], roi[3])

    @staticmethod
    def _manual_slot_radii(points: list[tuple[float, float]]) -> list[float]:
        """Chọn vùng lấy mẫu nhỏ hơn khoảng cách lân cận để slot không lẫn nhau."""
        radii: list[float] = []
        for index, point in enumerate(points):
            distances = [
                ((point[0] - other[0]) ** 2 + (point[1] - other[1]) ** 2) ** 0.5
                for other_index, other in enumerate(points) if other_index != index
            ]
            nearest = min(distances) if distances else 0.12
            radii.append(max(0.012, min(0.10, nearest * 0.28)))
        return radii

    @staticmethod
    def _same_roi(first: NormalizedRect | None, second: NormalizedRect | None) -> bool:
        if first is None or second is None:
            return first is second
        return max(abs(left - right) for left, right in zip(first, second, strict=True)) <= 0.005

    def _save_manual_scan_layout(self) -> None:
        product_id = self.profile_selector.currentData()
        profile = self._manager.get(product_id) if isinstance(product_id, str) and self._manager else None
        if profile is None:
            QMessageBox.warning(self, "Chưa chọn profile", "Chọn Product Profile trước khi lưu vị trí.")
            return
        if self._active_manual_strip is not None:
            QMessageBox.warning(self, "Chưa chốt xong vỉ", "Hoàn tất đủ slot cho vỉ đang chốt trước khi lưu.")
            return
        if self._roi is None:
            QMessageBox.warning(self, "Cần ROI", "Vẽ ROI cố định trước khi lưu vị trí slot.")
            return
        if (
            self._manual_reference_roi is not None
            and not self._same_roi(self._roi, self._manual_reference_roi)
        ):
            QMessageBox.warning(
                self,
                "ROI đã thay đổi",
                "Các điểm vị trí hiện có thuộc ROI cũ. Xóa chúng và chốt lại theo ROI mới trước khi lưu.",
            )
            return
        expected_count = len(profile.slots)
        if not self._manual_working_strips:
            QMessageBox.warning(self, "Chưa có vỉ", "Bấm CHỐT VỈ MỚI và click các slot trước khi lưu.")
            return
        from core.models import ManualStripLayout

        layouts: list[ManualStripLayout] = []
        for strip_index, points in enumerate(self._manual_working_strips, start=1):
            if len(points) != expected_count:
                QMessageBox.warning(
                    self, "Thiếu slot",
                    f"Vỉ {strip_index} có {len(points)} slot, cần đúng {expected_count} slot.",
                )
                return
            local_points = [self._surface_position_to_manual(x, y, self._roi) for x, y, _color in points]
            if any(point is None for point in local_points):
                QMessageBox.warning(self, "Slot ngoài ROI", f"Vỉ {strip_index} có slot nằm ngoài ROI.")
                return
            local_xy = [point for point in local_points if point is not None]
            radii = self._manual_slot_radii(local_xy)
            layouts.append(
                ManualStripLayout(
                    index=strip_index,
                    slots=[
                        SlotSpec(slot_index, point[0], point[1], color, radius)
                        for slot_index, ((point, radius), (_x, _y, color)) in enumerate(
                            zip(zip(local_xy, radii, strict=True), points, strict=True), start=1
                        )
                    ],
                )
            )
        locked_profile = ProductProfile(
            product_id=profile.product_id,
            name=profile.name,
            slots=profile.slots,
            minimum_confidence=profile.minimum_confidence,
            enabled=profile.enabled,
            manual_scan_strips=layouts,
            manual_scan_roi=self._roi,
            manual_scan_sample_radius=self._manual_sample_radius_fraction(),
        )
        try:
            self._manager.update(profile.product_id, locked_profile)
        except (KeyError, ValueError, OSError) as error:
            QMessageBox.warning(self, "Không thể lưu vị trí", str(error))
            return
        self._manual_reference_roi = self._roi
        self.refresh_profiles()
        selected_index = self.profile_selector.findData(profile.product_id)
        self.profile_selector.setCurrentIndex(selected_index if selected_index >= 0 else 0)
        self.product_profile_changed.emit(profile.product_id)
        self.profiles_changed.emit()
        sampling = self._manual_sample_radius_fraction()
        sampling_note = "tự động" if sampling is None else f"{sampling:.1%} cạnh ngắn ROI"
        self._set_status(
            f"Đã lưu {len(layouts)} vỉ và {len(layouts) * expected_count} vị trí slot theo ROI của {profile.product_id}; "
            f"vòng lấy màu: {sampling_note}. Realtime sẽ quét trực tiếp các điểm này, không dò hình học tự động.",
            level="success",
        )

    def _load_manual_color_locks(self, *_args: object) -> None:
        """Hiện các ô chốt màu theo slot của profile đang được chọn.

        Danh sách này là ý định sản phẩm do người vận hành xác nhận, không
        lấy từ nhãn màu mà AI suy luận trên ảnh.  Vì vậy khi camera/góc vỉ cố
        định, thay đổi ở đây sẽ là chuẩn để đánh PASS/NG cho từng vỉ trong ROI.
        """
        while self.manual_slot_form.count():
            item = self.manual_slot_form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.manual_slot_selectors.clear()

        product_id = self.profile_selector.currentData()
        profile = self._manager.get(product_id) if isinstance(product_id, str) and self._manager else None
        catalog_profiles = getattr(self._colors_catalog, "profiles", {})
        if profile is None or not catalog_profiles:
            self.manual_lock_hint.setText(
                "Chọn một Product Profile để tự chốt màu cho từng slot của vỉ."
            )
            self.save_manual_lock_button.setEnabled(False)
            self._load_manual_position_layout()
            return

        display = getattr(self._colors_catalog, "display", lambda color: color)
        for slot in sorted(profile.slots, key=lambda item: item.index):
            selector = QComboBox()
            for color_key in catalog_profiles:
                selector.addItem(str(display(color_key)), color_key)
            selected_index = selector.findData(slot.expected_color)
            selector.setCurrentIndex(selected_index if selected_index >= 0 else 0)
            selector.setToolTip(
                f"Slot {slot.index}: màu chuẩn do người vận hành chốt thủ công."
            )
            self.manual_slot_form.addRow(f"Slot {slot.index}:", selector)
            self.manual_slot_selectors.append(selector)
        self.manual_lock_hint.setText(
            f"Đang chốt vỉ '{profile.product_id}'. Chọn màu chuẩn cho {len(profile.slots)} slot, "
            "rồi bấm LƯU. Mọi vỉ của profile này trong ROI sẽ kiểm theo đúng từng vị trí."
        )
        self.save_manual_lock_button.setEnabled(True)
        self._load_manual_position_layout()

    def _save_manual_color_locks(self) -> None:
        """Lưu chuỗi màu mà người vận hành chốt, giữ nguyên hình học profile."""
        product_id = self.profile_selector.currentData()
        profile = self._manager.get(product_id) if isinstance(product_id, str) and self._manager else None
        if profile is None:
            QMessageBox.warning(self, "Chưa chọn profile", "Chọn Product Profile trước khi khóa màu thủ công.")
            return
        colors = [selector.currentData() for selector in self.manual_slot_selectors]
        if len(colors) != len(profile.slots) or any(not isinstance(color, str) or not color for color in colors):
            QMessageBox.warning(self, "Thiếu màu", "Hãy chọn một màu cho tất cả slot trước khi lưu.")
            return

        ordered_slots = sorted(profile.slots, key=lambda item: item.index)
        locked_slots = [
            SlotSpec(
                index=slot.index,
                x=slot.x,
                y=slot.y,
                expected_color=color,
                radius=slot.radius,
            )
            for slot, color in zip(ordered_slots, colors, strict=True)
        ]
        locked_profile = ProductProfile(
            product_id=profile.product_id,
            name=profile.name,
            slots=locked_slots,
            minimum_confidence=profile.minimum_confidence,
            enabled=profile.enabled,
            manual_scan_strips=profile.manual_scan_strips,
            manual_scan_roi=profile.manual_scan_roi,
            manual_scan_sample_radius=profile.manual_scan_sample_radius,
        )
        try:
            self._manager.update(profile.product_id, locked_profile)
        except (KeyError, ValueError, OSError) as error:
            QMessageBox.warning(self, "Không thể lưu khóa màu", str(error))
            return

        # Pipeline đọc profile từ ProductManager trong mỗi lần quét. Phát
        # signal để worker đang chạy và các trang khác nhận profile mới ngay.
        self.refresh_profiles()
        selected_index = self.profile_selector.findData(profile.product_id)
        self.profile_selector.setCurrentIndex(selected_index if selected_index >= 0 else 0)
        self.product_profile_changed.emit(profile.product_id)
        self.profiles_changed.emit()
        labels = " - ".join(
            str(getattr(self._colors_catalog, "display", lambda color: color)(color)) for color in colors
        )
        self._set_status(
            f"Đã khóa màu thủ công cho {profile.product_id}: {labels}. Áp dụng ngay cho từng slot.",
            level="success",
        )

    def set_pipeline(self, pipeline: FramePipeline | None) -> None:
        """Đặt callback suy luận. Callback lỗi sẽ được cô lập trong submit_frame."""
        self._pipeline = pipeline
        self._set_status(
            "Đã gắn pipeline suy luận." if pipeline else "Đã bỏ pipeline suy luận; chỉ hiển thị video.",
            level="info",
        )

    def set_export_directory(self, directory: str | Path) -> None:
        """Cho người vận hành mở nhanh dữ liệu Excel lịch sử theo ngày."""
        self._export_directory = Path(directory)
        self.excel_summary.setText(
            "Dữ liệu Excel lịch sử theo ngày: " + str(self._export_directory / "YYYY-MM-DD.xlsx")
            + ". Chế độ giám sát hiện tại không cộng số lượng vỉ."
        )
        self.open_excel_folder_button.setEnabled(True)

    def open_export_directory(self) -> None:
        if self._export_directory is None:
            return
        self._export_directory.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._export_directory)))

    def attach_camera_thread(self, camera_thread: CameraThreadProtocol | object, *, display_raw_frames: bool = True) -> None:
        """Gắn một CameraThread có ``frame_ready`` signal, không import implementation."""
        self.detach_camera_thread()
        signal = getattr(camera_thread, "frame_ready", None)
        if signal is None or not callable(getattr(signal, "connect", None)):
            raise TypeError("CameraThread cần có frame_ready signal với phương thức connect().")
        if display_raw_frames:
            signal.connect(self.submit_frame)
            self._camera_connections.append((signal, self.submit_frame))
        for error_name in ("camera_error", "error_occurred"):
            error_signal = getattr(camera_thread, error_name, None)
            if callable(getattr(error_signal, "connect", None)):
                error_signal.connect(self._on_camera_error)
                self._camera_connections.append((error_signal, self._on_camera_error))
        self._camera_thread = camera_thread
        self._set_camera_state("ĐÃ GẮN CAMERA", "#38bdf8")
        self._set_status(
            "Đã gắn CameraThread. Chọn nguồn và bấm BẮT ĐẦU."
            if display_raw_frames else "Đã gắn CameraThread; khung hình sẽ hiển thị qua worker inference."
        )

    def detach_camera_thread(self) -> None:
        """Tháo signal an toàn trước khi thay thế camera worker."""
        for signal, slot in self._camera_connections:
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        self._camera_connections.clear()
        self._camera_thread = None

    def start_camera(self) -> None:
        source = self.source_input.text().strip()
        if not source:
            QMessageBox.warning(self, "Thiếu nguồn", "Nhập chỉ số camera, địa chỉ RTSP hoặc chọn tệp video.")
            return
        if self.source_type.currentData() == "video" and not Path(source).is_file():
            QMessageBox.warning(self, "Không tìm thấy video", "Hãy chọn một tệp video tồn tại trên máy.")
            return
        request = CameraStartRequest(
            source=source,
            source_type=str(self.source_type.currentData()),
            expected_product_id=self.profile_selector.currentData(),
            roi=self._roi,
            counting_line=None,
            counting_direction="top_to_bottom",
            inspection_mode=str(self.inspection_mode.currentData()),
            position_locked_color=self.position_locked_color.isChecked(),
        )
        self.start_requested.emit(request)
        if self._camera_thread is None:
            self._set_camera_state("CHỜ CAMERA", "#facc15")
            self._set_status("Đã gửi yêu cầu khởi động. Controller cần gắn CameraThread để cấp khung hình.", level="warning")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            return
        try:
            configure = getattr(self._camera_thread, "configure", None)
            if callable(configure):
                configure(request)
            is_running = getattr(self._camera_thread, "isRunning", None)
            already_running = bool(is_running()) if callable(is_running) else False
            start = getattr(self._camera_thread, "start", None)
            if callable(start) and not already_running:
                start()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self._set_camera_state("ĐANG KHỞI ĐỘNG", "#facc15")
            self._set_status("Đang chờ khung hình đầu tiên từ camera...")
        except Exception as error:
            self._set_camera_state("LỖI CAMERA", "#ef4444")
            self._set_status(f"Không thể khởi động camera: {error}", level="error")

    def stop_camera(self) -> None:
        self.stop_requested.emit()
        if self._camera_thread is not None:
            try:
                stop = getattr(self._camera_thread, "stop", None)
                if callable(stop):
                    stop()
                else:
                    interruption = getattr(self._camera_thread, "requestInterruption", None)
                    if callable(interruption):
                        interruption()
            except Exception as error:
                self._set_status(f"Không thể dừng camera sạch sẽ: {error}", level="warning")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._set_camera_state("ĐÃ DỪNG", "#94a3b8")
        self._set_status("Đã gửi yêu cầu dừng camera.")

    @Slot(object)
    def submit_frame(self, packet_or_frame: object) -> None:
        """Nhận frame qua Qt signal; pipeline và hiển thị đều được bảo vệ lỗi."""
        raw_frame, preview, metadata = self._unpack_frame(packet_or_frame)
        pixmap = self._as_pixmap(preview) or self._as_pixmap(raw_frame)
        if pixmap is not None:
            self.video_surface.set_frame(pixmap)
        elif self._frame_index == 0:
            self._set_status("Frame không có QImage/QPixmap preview; adapter camera cần cung cấp preview để hiển thị.", level="warning")

        self._frame_index += 1
        self._update_fps()
        self._set_camera_state("ĐANG CHẠY", "#22c55e")
        if self._pipeline is None:
            self._set_status("Đang nhận video. Chưa gắn pipeline nhận diện.")
            return
        context = RealtimeContext(
            frame_index=self._frame_index,
            captured_at=datetime.now(),
            source=self.source_input.text().strip(),
            source_type=str(self.source_type.currentData()),
            expected_product_id=self.profile_selector.currentData(),
            roi=self._roi,
            counting_line=None,
        )
        try:
            inference = self._coerce_inference(self._pipeline(raw_frame, context), metadata)
        except Exception as error:
            self._set_status(f"Pipeline suy luận lỗi tại frame {self._frame_index}: {error}", level="error")
            return
        if inference is not None:
            self.apply_inference(inference)

    @Slot(object)
    def apply_inference(self, inference: RealtimeInference | Mapping[str, Any] | object) -> None:
        """Extension hook nếu pipeline worker trả kết quả qua signal riêng."""
        normalized = self._coerce_inference(inference, {})
        if normalized is None:
            return
        overlay = self._as_pixmap(normalized.annotated_frame)
        if overlay is not None:
            self.video_surface.set_frame(overlay)
        status = self._normalized_status(normalized.status)
        visible_products = max(0, int(normalized.visible_products))
        fault_products = max(0, int(normalized.fault_products))
        self.metric_values["visible"].setText(str(visible_products))
        self.metric_values["faults"].setText(str(fault_products))
        self.metric_values["alert"].setText("NG — CẢNH BÁO" if normalized.alert_active else "BÌNH THƯỜNG")
        if normalized.fps is not None:
            self.metric_values["fps"].setText(f"{normalized.fps:.1f} FPS")
        self._set_fault_alert(normalized.alert_active, normalized.alert_detail or normalized.detail)
        # Chi tiết detector (geometry score, confidence) dao động theo từng
        # frame. Chỉ ghi một dòng khi đợt cảnh báo bắt đầu, không làm đầy bảng
        # bằng hàng trăm bản sao của cùng một vỉ lỗi.
        signature = f"{normalized.product_id}:{status}:{fault_products}"
        if normalized.alert_active and signature != self._last_alert_signature:
            self._append_event(normalized, status)
        self._last_alert_signature = signature if normalized.alert_active else ""
        self._set_status(normalized.detail or f"Kết quả {status}", level=self._status_level(status))
        self.inference_received.emit(normalized)

    def _unpack_frame(self, packet_or_frame: object) -> tuple[object, object | None, Mapping[str, Any]]:
        if isinstance(packet_or_frame, FramePacket):
            return packet_or_frame.frame, packet_or_frame.preview, packet_or_frame.metadata
        if isinstance(packet_or_frame, Mapping) and "frame" in packet_or_frame:
            preview = packet_or_frame.get("preview")
            metadata = packet_or_frame.get("metadata", {})
            return packet_or_frame["frame"], preview, metadata if isinstance(metadata, Mapping) else {}
        return packet_or_frame, None, {}

    @staticmethod
    def _as_pixmap(value: object | None) -> QPixmap | None:
        if isinstance(value, QPixmap):
            return value if not value.isNull() else None
        if isinstance(value, QImage):
            return QPixmap.fromImage(value) if not value.isNull() else None
        return None

    @staticmethod
    def _normalized_status(value: object) -> str:
        status = str(value).upper().strip()
        return status if status in {"PASS", "FAIL", "UNKNOWN"} else "UNKNOWN"

    def _coerce_inference(
        self, result: RealtimeInference | Mapping[str, Any] | object | None, metadata: Mapping[str, Any]
    ) -> RealtimeInference | None:
        if result is None:
            return None
        if isinstance(result, RealtimeInference):
            return result
        # Kết quả từ worker có thể là dataclass độc lập UI, miễn có các field
        # tương ứng. Chuyển sang model UI tại boundary này để không kéo PySide
        # vào tầng inference.
        if result is not None and hasattr(result, "status") and hasattr(result, "annotated_frame"):
            return RealtimeInference(
                status=self._normalized_status(getattr(result, "status", "UNKNOWN")),
                product_id=getattr(result, "product_id", None),
                detected_colors=tuple(getattr(result, "detected_colors", ())),
                confidence=float(getattr(result, "confidence", 0.0)),
                count_delta=max(0, int(getattr(result, "count_delta", 0))),
                detected_delta=max(0, int(getattr(result, "detected_delta", 0))),
                pass_delta=max(0, int(getattr(result, "pass_delta", 0))),
                fail_delta=max(0, int(getattr(result, "fail_delta", 0))),
                unknown_delta=max(0, int(getattr(result, "unknown_delta", 0))),
                detail=str(getattr(result, "detail", "Worker không cung cấp ghi chú")),
                annotated_frame=getattr(result, "annotated_frame", None),
                fps=getattr(result, "fps", None),
                visible_products=max(0, int(getattr(result, "visible_products", 0))),
                fault_products=max(0, int(getattr(result, "fault_products", 0))),
                alert_active=bool(getattr(result, "alert_active", False)),
                alert_detail=str(getattr(result, "alert_detail", "")),
            )
        if not isinstance(result, Mapping):
            raise TypeError("Pipeline phải trả RealtimeInference, mapping, hoặc None.")
        colors = result.get("detected_colors", result.get("colors", ()))
        if isinstance(colors, str):
            colors = (colors,)
        try:
            confidence = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        try:
            count_delta = max(0, int(result.get("count_delta", 0)))
        except (TypeError, ValueError):
            count_delta = 0
        fps = result.get("fps", metadata.get("fps"))
        try:
            fps_value = float(fps) if fps is not None else None
        except (TypeError, ValueError):
            fps_value = None
        return RealtimeInference(
            status=self._normalized_status(result.get("status", "UNKNOWN")),
            product_id=str(result["product_id"]) if result.get("product_id") is not None else None,
            detected_colors=tuple(str(color) for color in colors) if isinstance(colors, Sequence) else (),
            confidence=confidence,
            count_delta=count_delta,
            detected_delta=max(0, int(result.get("detected_delta", 0))),
            pass_delta=max(0, int(result.get("pass_delta", count_delta))),
            fail_delta=max(0, int(result.get("fail_delta", 0))),
            unknown_delta=max(0, int(result.get("unknown_delta", 0))),
            detail=str(result.get("detail", "Pipeline không cung cấp ghi chú")),
            annotated_frame=result.get("annotated_frame", result.get("overlay")),
            fps=fps_value,
            visible_products=max(0, int(result.get("visible_products", 0))),
            fault_products=max(0, int(result.get("fault_products", 0))),
            alert_active=bool(result.get("alert_active", False)),
            alert_detail=str(result.get("alert_detail", "")),
        )

    def _update_fps(self) -> None:
        now = time.monotonic()
        self._frame_times.append(now)
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                self.metric_values["fps"].setText(f"{(len(self._frame_times) - 1) / elapsed:.1f} FPS")

    def _set_fault_alert(self, active: bool, detail: str) -> None:
        """Bật/tắt banner báo lỗi. Giữ tối thiểu 1.2 giây để không bị lọt mắt."""
        if active:
            self._alert_active = True
            self._alert_phase = True
            self._alert_hold_until = time.monotonic() + 1.2
            self.alert_banner.setText("⚠ " + (detail or "CẢNH BÁO NG: PHÁT HIỆN VỈ LỖI TRONG ROI"))
            if not self._alert_timer.isActive():
                self._alert_timer.start()
            self._apply_alert_banner_style()
            return
        if self._alert_active and time.monotonic() < self._alert_hold_until:
            return
        self._alert_active = False
        self._alert_phase = False
        self._alert_timer.stop()
        self.alert_banner.setText("✓ GIÁM SÁT BÌNH THƯỜNG — CHƯA PHÁT HIỆN VỈ LỖI")
        self._apply_alert_banner_style()

    def _toggle_alert_flash(self) -> None:
        if not self._alert_active:
            self._alert_timer.stop()
            return
        self._alert_phase = not self._alert_phase
        self._apply_alert_banner_style()

    def _apply_alert_banner_style(self) -> None:
        if self._alert_active:
            background, foreground, border = (
                ("#dc2626", "#ffffff", "#fecaca")
                if self._alert_phase else ("#fef2f2", "#b91c1c", "#ef4444")
            )
        else:
            background, foreground, border = "#14532d", "#dcfce7", "#22c55e"
        self.alert_banner.setStyleSheet(
            f"padding: 10px; border: 2px solid {border}; border-radius: 6px; "
            f"background: {background}; color: {foreground}; font-size: 16px; font-weight: 800;"
        )

    def _append_event(self, inference: RealtimeInference, status: str) -> None:
        row = self.event_table.rowCount()
        self.event_table.insertRow(row)
        values = (
            datetime.now().strftime("%H:%M:%S"),
            inference.product_id or self.profile_selector.currentData() or "--",
            status,
            " - ".join(inference.detected_colors) or "--",
            f"{inference.confidence:.1%}",
            str(max(1, inference.fault_products)) if inference.alert_active else "--",
            inference.detail,
        )
        for column, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            if column == 2:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                colors = {"PASS": "#166534", "FAIL": "#991b1b", "UNKNOWN": "#854d0e"}
                item.setBackground(QColor(colors[status]))
                item.setForeground(QColor("#ffffff"))
            self.event_table.setItem(row, column, item)
        # Bảng chỉ là dashboard, audit đầy đủ nên nằm ở tầng service/database.
        while self.event_table.rowCount() > 200:
            self.event_table.removeRow(0)
        self.event_table.scrollToBottom()

    def _toggle_roi_draw(self, enabled: bool) -> None:
        if enabled:
            self.video_surface.set_draw_mode("roi")
            self._set_status("Kéo chuột trên khung hình để chọn vùng ROI.")
        elif self.video_surface.draw_mode == "roi":
            self.video_surface.set_draw_mode(None)

    @staticmethod
    def _uncheck_button(button: QPushButton) -> None:
        button.blockSignals(True)
        button.setChecked(False)
        button.blockSignals(False)

    def _set_roi_from_surface(self, roi: object) -> None:
        if not isinstance(roi, tuple) or len(roi) != 4:
            return
        self._roi = roi  # type: ignore[assignment]
        self.video_surface.set_roi(self._roi)
        self.roi_summary.setText(
            f"ROI: x={roi[0]:.0%}, y={roi[1]:.0%}, rộng={roi[2]:.0%}, cao={roi[3]:.0%}."
        )
        self._uncheck_button(self.draw_roi_button)
        self._refresh_manual_slot_markers()
        self.roi_changed.emit(self._roi)
        if self._manual_working_strips and not self._same_roi(self._roi, self._manual_reference_roi):
            self._set_status(
                "ROI đã thay đổi; các vị trí vỉ thủ công đang thuộc ROI cũ. Xóa và chốt lại trước khi lưu/quét.",
                level="warning",
            )
        else:
            self._set_status("Đã cập nhật ROI cho pipeline.")

    def clear_roi(self) -> None:
        self._roi = None
        self.video_surface.set_roi(None)
        self._refresh_manual_slot_markers()
        self.roi_summary.setText("Chưa chọn ROI — pipeline sẽ nhận toàn bộ khung hình.")
        self.roi_changed.emit(None)
        self._set_status("Đã xóa ROI. Vị trí quét thủ công chỉ dùng được khi chọn lại ROI đã lưu.", level="warning")

    def _emit_product_profile_changed(self, *_args: object) -> None:
        self.product_profile_changed.emit(self.profile_selector.currentData())

    @Slot(str)
    def show_runtime_error(self, message: str) -> None:
        self._set_camera_state("LỖI INFERENCE", "#ef4444")
        self._set_status(message, level="error")

    @Slot(object)
    def _on_camera_error(self, error: object) -> None:
        message = str(error)
        if message.startswith("Đã phát hết video"):
            self._set_camera_state("ĐÃ HẾT VIDEO", "#38bdf8")
            self._set_status(message + " Bấm BẮT ĐẦU để phát lại.", level="info")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            return
        self._set_camera_state("LỖI CAMERA", "#ef4444")
        self._set_status(f"Camera báo lỗi: {message}", level="error")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def _set_camera_state(self, text: str, color: str) -> None:
        self.camera_state.setText(f"● {text}")
        self.camera_state.setStyleSheet(f"color: {color}; font-weight: 700;")

    @staticmethod
    def _status_level(status: str) -> str:
        return {"PASS": "success", "FAIL": "error", "UNKNOWN": "warning"}[status]

    def _set_status(self, message: str, level: str = "info") -> None:
        colors = {
            "info": ("#dbeafe", "#1d4ed8"),
            "success": ("#dcfce7", "#166534"),
            "warning": ("#fef3c7", "#92400e"),
            "error": ("#fee2e2", "#991b1b"),
        }
        background, foreground = colors.get(level, colors["info"])
        self.status_label.setText(message)
        self.status_label.setStyleSheet(
            f"padding: 8px; border: 1px solid {foreground}; border-radius: 4px; "
            f"background: {background}; color: {foreground};"
        )
