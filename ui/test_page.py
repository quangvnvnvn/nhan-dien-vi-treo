"""Giao diện TEST MODE cho ảnh tĩnh.

Trang này chỉ trình bày và điều phối việc kiểm tra ảnh. Nó không thay đổi
Product Profile hoặc dữ liệu huấn luyện; mọi quyết định PASS/FAIL/UNKNOWN vẫn
nằm ở ``TestService`` và validator.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.colors import ColorCatalog
from core.models import ProductProfile, ProductStatus, SlotSpec, ValidationResult
from services.test_service import TestService
from training.product_manager import ProductManager

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


_STATUS_STYLE = {
    ProductStatus.PASS: ("#166534", "#f0fdf4", "PASS  •  ĐẠT"),
    ProductStatus.FAIL: ("#b91c1c", "#fff1f2", "FAIL  •  NG"),
    ProductStatus.UNKNOWN: ("#a16207", "#fffbeb", "UNKNOWN  •  REVIEW"),
}

_REASON_LABEL = {
    "MISSING_ITEM": "Thiếu viên",
    "EXTRA_ITEM": "Dư viên",
    "WRONG_COLOR": "Sai màu",
    "WRONG_PRODUCT": "Sai sản phẩm",
    "LOW_CONFIDENCE": "Độ tin cậy thấp",
    "INVALID_GEOMETRY": "Hình học chưa đạt",
    "UNKNOWN": "Chưa xác định",
}


class ImagePreview(QWidget):
    """Preview tự co giãn mà không làm méo ảnh nguồn."""

    def __init__(self) -> None:
        super().__init__()
        self._path: Path | None = None
        self._source = QPixmap()
        self.setObjectName("testPreview")
        self.setMinimumWidth(420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        self.caption = QLabel("XEM ẢNH")
        self.caption.setObjectName("previewCaption")
        self.caption.setStyleSheet("font-weight: 700;")
        self.path_label = QLabel("Chọn một ảnh ở danh sách bên trái để xem chi tiết.")
        self.path_label.setObjectName("previewPath")
        self.path_label.setWordWrap(True)
        self.canvas = QLabel("CHƯA CHỌN ẢNH")
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas.setMinimumSize(360, 240)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setStyleSheet(
            "background: #171717; border: 1px dashed #64748b; border-radius: 8px; "
            "color: #cbd5e1; font-weight: 700; padding: 12px;"
        )
        layout.addWidget(self.caption)
        layout.addWidget(self.path_label)
        layout.addWidget(self.canvas, 1)

    def set_path(self, path: Path | None) -> None:
        self._path = path
        self._source = QPixmap(str(path)) if path is not None else QPixmap()
        if path is None:
            self.caption.setText("XEM ẢNH")
            self.path_label.setText("Chọn một ảnh ở danh sách bên trái để xem chi tiết.")
            self.path_label.setToolTip("")
        elif self._source.isNull():
            self.caption.setText("KHÔNG THỂ HIỂN THỊ ẢNH")
            self.path_label.setText(path.name)
            self.path_label.setToolTip(str(path))
        else:
            self.caption.setText(f"{path.name}  •  {self._source.width()} × {self._source.height()} px")
            self.path_label.setText(str(path))
            self.path_label.setToolTip(str(path))
        self._render()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt API name
        super().resizeEvent(event)
        self._render()

    def _render(self) -> None:
        if self._source.isNull():
            self.canvas.setPixmap(QPixmap())
            self.canvas.setText("CHƯA CHỌN ẢNH" if self._path is None else "KHÔNG THỂ ĐỌC ẢNH")
            return
        size = self.canvas.size()
        if size.width() <= 1 or size.height() <= 1:
            return
        self.canvas.setText("")
        self.canvas.setPixmap(
            self._source.scaled(
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class MetricCard(QFrame):
    """Ô chỉ số nhỏ, dùng chung cho tổng hợp batch test."""

    def __init__(self, title: str, accent: str) -> None:
        super().__init__()
        self.setObjectName("metricCard")
        self.setMinimumHeight(72)
        self.setStyleSheet(
            "QFrame#metricCard {"
            f"border-left: 4px solid {accent}; border-top: 1px solid #4b5563; "
            "border-right: 1px solid #4b5563; border-bottom: 1px solid #4b5563; "
            "border-radius: 8px; background: #303030; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(1)
        title_label = QLabel(title.upper())
        title_label.setStyleSheet("color: #cbd5e1; font-size: 11px; font-weight: 700;")
        self.value = QLabel("0")
        self.value.setStyleSheet("color: #f8fafc; font-size: 22px; font-weight: 800;")
        layout.addWidget(title_label)
        layout.addWidget(self.value)

    def set_value(self, value: str | int) -> None:
        self.value.setText(str(value))


class TestPage(QWidget):
    """Batch test ảnh với profile đã được người vận hành xác nhận."""

    def __init__(self, manager: ProductManager, service: TestService, colors_catalog: ColorCatalog) -> None:
        super().__init__()
        self.manager = manager
        self.service = service
        self.colors_catalog = colors_catalog
        self.images: list[Path] = []
        self._result_counts = {status: 0 for status in ProductStatus}
        self._build()
        self.refresh_profiles()

    def _build(self) -> None:
        self.setObjectName("testModePage")
        self.setStyleSheet(
            "QFrame#testCard { border: 1px solid #4b5563; border-radius: 8px; background: #303030; }"
            "QFrame#resultDetail { border: 1px solid #475569; border-radius: 7px; background: #262626; }"
            "QLabel#pageTitle { font-size: 20px; font-weight: 800; color: #f8fafc; }"
            "QLabel#pageSubtitle { color: #cbd5e1; }"
            "QLabel#profileName { font-size: 15px; font-weight: 800; color: #f8fafc; }"
            "QLabel#profileMeta { color: #cbd5e1; }"
            "QLabel#profileColors { color: #a5f3fc; font-weight: 700; }"
            "QLabel#queueMeta { color: #cbd5e1; }"
            "QLabel#detailTitle { color: #cbd5e1; font-weight: 800; }"
            "QPushButton#primaryTestButton { background: #15803d; color: #ffffff; "
            "font-weight: 800; padding: 7px 14px; border: 1px solid #22c55e; border-radius: 6px; }"
            "QPushButton#primaryTestButton:hover { background: #16a34a; }"
            "QPushButton#dangerButton { color: #fecaca; }"
            "QTableWidget { gridline-color: #4b5563; selection-background-color: #1d4ed8; }"
            "QHeaderView::section { background: #3f3f46; color: #f8fafc; padding: 7px; "
            "border: 0; border-right: 1px solid #52525b; font-weight: 700; }"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        heading = QHBoxLayout()
        heading_text = QVBoxLayout()
        title = QLabel("KIỂM TRA ẢNH")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Chọn Product Profile, thêm ảnh, sau đó chạy kiểm tra theo lô.")
        subtitle.setObjectName("pageSubtitle")
        heading_text.addWidget(title)
        heading_text.addWidget(subtitle)
        heading.addLayout(heading_text)
        heading.addStretch()
        self.batch_state = QLabel("SẴN SÀNG")
        self.batch_state.setStyleSheet(
            "color: #bae6fd; border: 1px solid #0369a1; border-radius: 10px; "
            "padding: 4px 10px; font-weight: 800;"
        )
        heading.addWidget(self.batch_state)
        root.addLayout(heading)

        notice = QFrame()
        notice.setObjectName("testCard")
        notice_layout = QHBoxLayout(notice)
        notice_layout.setContentsMargins(12, 8, 12, 8)
        notice_icon = QLabel("i")
        notice_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        notice_icon.setFixedSize(18, 18)
        notice_icon.setStyleSheet(
            "background: #1d4ed8; color: white; border-radius: 9px; font-weight: 800;"
        )
        notice_text = QLabel(
            "Kết quả chỉ PASS khi nhận đủ slot, đúng màu và đủ độ tin cậy. "
            "Ảnh bị che, quá nghiêng hoặc phản sáng mạnh sẽ được đưa vào REVIEW thay vì kết luận sai."
        )
        notice_text.setWordWrap(True)
        notice_layout.addWidget(notice_icon, 0, Qt.AlignmentFlag.AlignTop)
        notice_layout.addWidget(notice_text, 1)
        root.addWidget(notice)

        root.addWidget(self._build_profile_card())

        metric_layout = QHBoxLayout()
        metric_layout.setSpacing(8)
        self.metrics = {
            "images": MetricCard("Ảnh đã chọn", "#38bdf8"),
            "pass": MetricCard("Đạt", "#22c55e"),
            "fail": MetricCard("Không đạt", "#ef4444"),
            "unknown": MetricCard("Cần review", "#facc15"),
        }
        for card in self.metrics.values():
            metric_layout.addWidget(card)
        root.addLayout(metric_layout)

        work_splitter = QSplitter(Qt.Orientation.Horizontal)
        work_splitter.setChildrenCollapsible(False)
        work_splitter.addWidget(self._build_image_queue())
        self.preview = ImagePreview()
        work_splitter.addWidget(self.preview)
        work_splitter.setSizes([420, 780])
        root.addWidget(work_splitter, 5)

        root.addWidget(self._build_results_panel(), 4)
        self._update_queue_summary()
        self._update_profile_summary()

    def _build_profile_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("testCard")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(12)

        selector_layout = QVBoxLayout()
        selector_label = QLabel("PRODUCT PROFILE DÙNG ĐỂ SO SÁNH")
        selector_label.setStyleSheet("color: #cbd5e1; font-size: 11px; font-weight: 800;")
        self.profile_selector = QComboBox()
        self.profile_selector.setMinimumWidth(360)
        self.profile_selector.setToolTip("Chỉ profile được chọn mới dùng để kiểm tra lô ảnh hiện tại.")
        self.profile_selector.currentIndexChanged.connect(self._update_profile_summary)
        selector_layout.addWidget(selector_label)
        selector_layout.addWidget(self.profile_selector)
        layout.addLayout(selector_layout, 2)

        profile_separator = QFrame()
        profile_separator.setFrameShape(QFrame.Shape.VLine)
        profile_separator.setStyleSheet("color: #64748b;")
        layout.addWidget(profile_separator)

        self.profile_name = QLabel("Chưa chọn Product Profile")
        self.profile_name.setObjectName("profileName")
        self.profile_meta = QLabel("Chọn một profile để xem cấu hình slot và ngưỡng kiểm tra.")
        self.profile_meta.setObjectName("profileMeta")
        self.profile_meta.setWordWrap(True)
        self.profile_colors = QLabel("")
        self.profile_colors.setObjectName("profileColors")
        self.profile_colors.setWordWrap(True)
        summary_layout = QVBoxLayout()
        summary_layout.addWidget(self.profile_name)
        summary_layout.addWidget(self.profile_meta)
        summary_layout.addWidget(self.profile_colors)
        layout.addLayout(summary_layout, 3)

        self.refresh_button = QPushButton("↻ TẢI PROFILE")
        self.refresh_button.setToolTip("Tải lại danh sách profile đã lưu.")
        self.refresh_button.clicked.connect(self.refresh_profiles)
        layout.addWidget(self.refresh_button, 0, Qt.AlignmentFlag.AlignVCenter)
        return card

    def _build_image_queue(self) -> QFrame:
        card = QFrame()
        card.setObjectName("testCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header_label = QLabel("DANH SÁCH ẢNH")
        header_label.setStyleSheet("font-weight: 800;")
        self.queue_summary = QLabel()
        self.queue_summary.setObjectName("queueMeta")
        header.addWidget(header_label)
        header.addStretch()
        header.addWidget(self.queue_summary)
        layout.addLayout(header)

        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.image_list.setAlternatingRowColors(True)
        self.image_list.setToolTip("Chọn ảnh để xem preview. Kết quả mới nhất sẽ hiển thị bằng màu ở danh sách.")
        self.image_list.currentRowChanged.connect(self.show_preview)
        self.image_list.currentRowChanged.connect(self._update_remove_state)
        layout.addWidget(self.image_list, 1)

        action_row = QHBoxLayout()
        self.add_images_button = QPushButton("+ THÊM ẢNH")
        self.add_folder_button = QPushButton("+ THÊM THƯ MỤC")
        self.remove_button = QPushButton("BỎ ẢNH")
        self.remove_button.setObjectName("dangerButton")
        self.add_images_button.clicked.connect(self.add_images)
        self.add_folder_button.clicked.connect(self.add_folder)
        self.remove_button.clicked.connect(self.remove_selected)
        action_row.addWidget(self.add_images_button)
        action_row.addWidget(self.add_folder_button)
        action_row.addWidget(self.remove_button)
        layout.addLayout(action_row)

        self.run_button = QPushButton("▶ CHẠY KIỂM TRA")
        self.run_button.setObjectName("primaryTestButton")
        self.run_button.setMinimumHeight(36)
        self.run_button.setToolTip("Chạy nhận diện cho toàn bộ ảnh đang có trong danh sách.")
        self.run_button.clicked.connect(self.run_test)
        layout.addWidget(self.run_button)
        return card

    def _build_results_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        header = QHBoxLayout()
        title = QLabel("KẾT QUẢ KIỂM TRA")
        title.setStyleSheet("font-size: 15px; font-weight: 800;")
        self.result_summary = QLabel("Chưa có kết quả.")
        self.result_summary.setObjectName("queueMeta")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.result_summary)
        layout.addLayout(header)

        self.results = QTableWidget(0, 7)
        self.results.setHorizontalHeaderLabels([
            "Ảnh", "Kết quả", "Độ tin cậy", "Màu kỳ vọng", "Màu phát hiện", "Lý do", "Ghi chú ngắn",
        ])
        self.results.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.results.setAlternatingRowColors(True)
        self.results.setWordWrap(False)
        self.results.setMinimumHeight(170)
        self.results.verticalHeader().setVisible(False)
        self.results.verticalHeader().setDefaultSectionSize(30)
        table_header = self.results.horizontalHeader()
        table_header.setStretchLastSection(True)
        for column, width in enumerate((180, 140, 104, 190, 190, 145)):
            self.results.setColumnWidth(column, width)
            table_header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
        table_header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.results.itemSelectionChanged.connect(self._show_selected_result_detail)
        layout.addWidget(self.results, 1)

        detail_card = QFrame()
        detail_card.setObjectName("resultDetail")
        detail_layout = QVBoxLayout(detail_card)
        detail_layout.setContentsMargins(10, 7, 10, 8)
        detail_layout.setSpacing(3)
        detail_title = QLabel("CHI TIẾT DÒNG ĐƯỢC CHỌN")
        detail_title.setObjectName("detailTitle")
        self.result_detail = QLabel("Chọn một kết quả để xem đầy đủ sản phẩm, màu và ghi chú.")
        self.result_detail.setWordWrap(True)
        self.result_detail.setTextFormat(Qt.TextFormat.RichText)
        self.result_detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        detail_layout.addWidget(detail_title)
        detail_layout.addWidget(self.result_detail)
        layout.addWidget(detail_card)
        return container

    def refresh_profiles(self) -> None:
        previous = self.profile_selector.currentData()
        self.profile_selector.blockSignals(True)
        self.profile_selector.clear()
        self.profile_selector.addItem("-- Chọn Product Profile --", None)
        for profile in self.manager.list_profiles():
            color_text = "  →  ".join(self.colors_catalog.display(color) for color in profile.expected_colors)
            self.profile_selector.addItem(
                f"{profile.product_id} — {profile.name}  •  {len(profile.slots)} slot  •  {color_text}",
                profile.product_id,
            )
        index = self.profile_selector.findData(previous)
        self.profile_selector.setCurrentIndex(index if index >= 0 else 0)
        self.profile_selector.blockSignals(False)
        self._update_profile_summary()

    def _update_profile_summary(self, *_args: object) -> None:
        product_id = self.profile_selector.currentData()
        profile = self.manager.get(product_id) if product_id else None
        if profile is None:
            self.profile_name.setText("Chưa chọn Product Profile")
            self.profile_meta.setText("Chọn một profile để xem cấu hình slot và ngưỡng kiểm tra.")
            self.profile_colors.setText("")
            self.profile_name.setToolTip("")
            return
        colors = [self.colors_catalog.display(color) for color in profile.expected_colors]
        self.profile_name.setText(f"{profile.product_id} — {profile.name}")
        self.profile_meta.setText(f"{len(profile.slots)} slot  •  Ngưỡng tin cậy: {profile.minimum_confidence:.0%}")
        self.profile_colors.setText("Màu theo thứ tự: " + "  →  ".join(colors))
        self.profile_name.setToolTip(f"Mã sản phẩm: {profile.product_id}\nTên: {profile.name}")
        self.profile_colors.setToolTip("  →  ".join(colors))

    def add_images(self) -> None:
        names, _ = QFileDialog.getOpenFileNames(
            self,
            "Chọn ảnh test",
            "",
            "Ảnh (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        self._append_paths(Path(name) for name in names)

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục ảnh test")
        if not folder:
            return
        try:
            paths = sorted(
                (path for path in Path(folder).iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS),
                key=lambda path: path.name.lower(),
            )
        except OSError as error:
            QMessageBox.warning(self, "Không thể đọc thư mục", str(error))
            return
        self._append_paths(paths)

    def _append_paths(self, paths: Iterable[Path]) -> None:
        existing = {path.resolve() for path in self.images}
        added = 0
        for path in paths:
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in existing or not resolved.is_file():
                continue
            self.images.append(resolved)
            item = QListWidgetItem(resolved.name)
            item.setData(Qt.ItemDataRole.UserRole, str(resolved))
            item.setToolTip(str(resolved))
            self.image_list.addItem(item)
            existing.add(resolved)
            added += 1
        if added:
            self._clear_results()
            self._update_queue_summary()
            self.batch_state.setText(f"ĐÃ THÊM {added} ẢNH")
            self.batch_state.setStyleSheet(
                "color: #bbf7d0; border: 1px solid #15803d; border-radius: 10px; "
                "padding: 4px 10px; font-weight: 800;"
            )
            if self.image_list.currentRow() < 0:
                self.image_list.setCurrentRow(0)

    def remove_selected(self) -> None:
        row = self.image_list.currentRow()
        if row < 0:
            return
        self.images.pop(row)
        self.image_list.takeItem(row)
        self._clear_results()
        self._update_queue_summary()
        if self.images:
            self.image_list.setCurrentRow(min(row, len(self.images) - 1))
        else:
            self.preview.set_path(None)
        self.batch_state.setText("ĐÃ CẬP NHẬT DANH SÁCH")
        self.batch_state.setStyleSheet(
            "color: #bae6fd; border: 1px solid #0369a1; border-radius: 10px; "
            "padding: 4px 10px; font-weight: 800;"
        )

    def _update_remove_state(self, *_args: object) -> None:
        self.remove_button.setEnabled(self.image_list.currentRow() >= 0)

    def _update_queue_summary(self) -> None:
        count = len(self.images)
        self.queue_summary.setText("Chưa có ảnh" if count == 0 else f"{count} ảnh đã chọn")
        self.metrics["images"].set_value(count)
        self._update_remove_state()

    def show_preview(self, row: int) -> None:
        self.preview.set_path(self.images[row] if 0 <= row < len(self.images) else None)

    def run_test(self) -> None:
        product_id = self.profile_selector.currentData()
        profile = self.manager.get(product_id) if product_id else None
        if profile is None:
            QMessageBox.warning(self, "Thiếu profile", "Tạo hoặc chọn Product Profile trước khi chạy kiểm tra.")
            return
        if not self.images:
            QMessageBox.warning(self, "Chưa có ảnh", "Thêm ít nhất một ảnh hoặc thư mục ảnh trước khi chạy kiểm tra.")
            return
        profile = self._normalized_profile(profile)
        if profile is None:
            return

        self._clear_results()
        self.run_button.setEnabled(False)
        self.run_button.setText("ĐANG KIỂM TRA...")
        self.batch_state.setText("ĐANG CHẠY")
        self.batch_state.setStyleSheet(
            "color: #fde68a; border: 1px solid #a16207; border-radius: 10px; "
            "padding: 4px 10px; font-weight: 800;"
        )
        try:
            for image in self.images:
                try:
                    result = self.service.inspect_image(image, profile)
                except (OSError, ValueError) as error:
                    self._add_read_error(image, profile.product_id, str(error))
                    continue
                self._add_result(image, profile.product_id, result)
        finally:
            self.run_button.setEnabled(True)
            self.run_button.setText("▶ CHẠY KIỂM TRA")

        total = self.results.rowCount()
        self.batch_state.setText("ĐÃ HOÀN THÀNH")
        self.batch_state.setStyleSheet(
            "color: #bbf7d0; border: 1px solid #15803d; border-radius: 10px; "
            "padding: 4px 10px; font-weight: 800;"
        )
        self.result_summary.setText(
            f"{total} ảnh  •  PASS {self._result_counts[ProductStatus.PASS]}  •  "
            f"FAIL {self._result_counts[ProductStatus.FAIL]}  •  "
            f"REVIEW {self._result_counts[ProductStatus.UNKNOWN]}"
        )
        if total:
            self.results.selectRow(0)

    def _normalized_profile(self, profile: ProductProfile) -> ProductProfile | None:
        """Hỗ trợ profile cũ có tên màu tiếng Việt mà không tự ghi đè dữ liệu."""
        slots: list[SlotSpec] = []
        invalid: list[str] = []
        for spec in profile.slots:
            color = self.colors_catalog.normalize(spec.expected_color)
            if color is None:
                invalid.append(spec.expected_color)
                continue
            slots.append(SlotSpec(spec.index, spec.x, spec.y, color, spec.radius))
        if invalid:
            QMessageBox.warning(
                self,
                "Product Profile chưa hợp lệ",
                f"Màu chưa được cấu hình: {', '.join(invalid)}. Hãy sửa Product Profile trước khi kiểm tra.",
            )
            return None
        return ProductProfile(profile.product_id, profile.name, slots, profile.minimum_confidence, profile.enabled)

    def _clear_results(self) -> None:
        self.results.setRowCount(0)
        self._result_counts = {status: 0 for status in ProductStatus}
        self.metrics["pass"].set_value(0)
        self.metrics["fail"].set_value(0)
        self.metrics["unknown"].set_value(0)
        self.result_summary.setText("Chưa có kết quả.")
        self.result_detail.setText("Chọn một kết quả để xem đầy đủ sản phẩm, màu và ghi chú.")
        for row in range(self.image_list.count()):
            item = self.image_list.item(row)
            item.setBackground(QColor())
            item.setForeground(QColor())
            path = item.data(Qt.ItemDataRole.UserRole) or item.text()
            item.setToolTip(str(path))

    def _add_read_error(self, image: Path, expected: str, detail: str) -> None:
        """Lỗi đọc file được hiển thị rõ như một dòng REVIEW, không làm dừng lô test."""
        result = ValidationResult(
            product_id=None,
            status=ProductStatus.UNKNOWN,
            reason=None,
            confidence=0.0,
            detail=f"Không thể đọc ảnh: {detail}",
        )
        self._add_result(image, expected, result)

    def _add_result(self, image: Path, expected: str, result: ValidationResult) -> None:
        row = self.results.rowCount()
        self.results.insertRow(row)
        profile = self.manager.get(expected)
        expected_colors = "  →  ".join(
            self.colors_catalog.display(color) for color in profile.expected_colors
        ) if profile else "--"
        observed_colors = "  →  ".join(
            self.colors_catalog.display(ob.color) for ob in result.observations
        ) or "Chưa xác định"
        reason_code = result.reason.value if result.reason else "--"
        reason_text = _REASON_LABEL.get(reason_code, reason_code)
        background, foreground, badge = _STATUS_STYLE[result.status]
        values = (
            image.name,
            badge,
            f"{result.confidence:.1%}",
            expected_colors,
            observed_colors,
            reason_text,
            self._short_detail(result.detail),
        )
        full_details = {
            "path": str(image),
            "expected_product": expected,
            "detected_product": result.product_id or "--",
            "expected_colors": expected_colors,
            "detected_colors": observed_colors,
            "status": result.status.value,
            "reason": reason_text,
            "reason_code": reason_code,
            "confidence": f"{result.confidence:.1%}",
            "detail": result.detail or "--",
        }
        row_fill = QColor({
            ProductStatus.PASS: "#173a2b",
            ProductStatus.FAIL: "#421f25",
            ProductStatus.UNKNOWN: "#4a3a16",
        }[result.status])
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setToolTip(value if column != 6 else full_details["detail"])
            item.setData(Qt.ItemDataRole.UserRole, full_details if column == 0 else None)
            if column == 1:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setBackground(QColor(background))
                item.setForeground(QColor(foreground))
                item.setToolTip(f"{result.status.value}: {reason_text}\n{full_details['detail']}")
            elif column == 2:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setBackground(row_fill)
            else:
                item.setBackground(row_fill)
            self.results.setItem(row, column, item)

        self._result_counts[result.status] += 1
        self.metrics["pass"].set_value(self._result_counts[ProductStatus.PASS])
        self.metrics["fail"].set_value(self._result_counts[ProductStatus.FAIL])
        self.metrics["unknown"].set_value(self._result_counts[ProductStatus.UNKNOWN])
        self._apply_result_to_image_list(image, result.status, reason_text)

    def _apply_result_to_image_list(self, image: Path, status: ProductStatus, reason: str) -> None:
        background, foreground, badge = _STATUS_STYLE[status]
        for row in range(self.image_list.count()):
            item = self.image_list.item(row)
            item_path = item.data(Qt.ItemDataRole.UserRole)
            if item_path != str(image):
                continue
            item.setBackground(QColor(background))
            item.setForeground(QColor(foreground))
            item.setToolTip(f"{image}\n{badge}\nLý do: {reason}")
            break

    def _show_selected_result_detail(self) -> None:
        row = self.results.currentRow()
        if row < 0:
            return
        item = self.results.item(row, 0)
        details = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if not isinstance(details, dict):
            return
        self.result_detail.setText(
            "<b>{status}</b> &nbsp; | &nbsp; <b>Ảnh:</b> {image}<br>"
            "<b>Sản phẩm:</b> kỳ vọng {expected_product} &rarr; phát hiện {detected_product} &nbsp; "
            "<b>Độ tin cậy:</b> {confidence}<br>"
            "<b>Màu:</b> {expected_colors} &rarr; {detected_colors}<br>"
            "<b>Lý do:</b> {reason} ({reason_code})<br>"
            "<b>Ghi chú:</b> {detail}".format(
                status=escape(str(details["status"])),
                image=escape(Path(str(details["path"])).name),
                expected_product=escape(str(details["expected_product"])),
                detected_product=escape(str(details["detected_product"])),
                confidence=escape(str(details["confidence"])),
                expected_colors=escape(str(details["expected_colors"])),
                detected_colors=escape(str(details["detected_colors"])),
                reason=escape(str(details["reason"])),
                reason_code=escape(str(details["reason_code"])),
                detail=escape(str(details["detail"])),
            )
        )

    @staticmethod
    def _short_detail(detail: str, limit: int = 95) -> str:
        compact = " ".join((detail or "--").split())
        return compact if len(compact) <= limit else f"{compact[:limit - 1]}…"
