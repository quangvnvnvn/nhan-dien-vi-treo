"""Trang tạo/sửa Product Profile có xác nhận thủ công của người vận hành."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from core.colors import ColorCatalog
from core.models import ProductProfile, SlotSpec
from training.product_manager import ProductManager


class ProductPage(QWidget):
    profiles_changed = Signal()

    def __init__(self, manager: ProductManager, colors_catalog: ColorCatalog) -> None:
        super().__init__()
        self.manager = manager
        self.colors_catalog = colors_catalog
        self.product_id = QLineEdit()
        self.name = QLineEdit()
        self.colors = QLineEdit()
        self.colors.setPlaceholderText("Ví dụ: Tím, Tím, Xanh dương, Tím, Tím")
        self.minimum_confidence = QDoubleSpinBox()
        self.minimum_confidence.setRange(0.50, 1.00)
        self.minimum_confidence.setSingleStep(0.01)
        self.minimum_confidence.setValue(0.85)
        self.list_widget = QListWidget()
        self._selected_product_id: str | None = None
        self._build()
        self.refresh()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("PRODUCT MANAGER — chỉ lưu sau khi người dùng xác nhận."))
        available_colors = QLabel(
            f"Màu đã cấu hình: {self.colors_catalog.accepted_names()}. "
            "Nhập theo thứ tự slot, ngăn cách bằng dấu phẩy."
        )
        available_colors.setWordWrap(True)
        available_colors.setStyleSheet("padding: 8px; background: #e0f2fe; color: #0c4a6e; border-radius: 4px;")
        layout.addWidget(available_colors)
        form = QFormLayout()
        form.addRow("Mã sản phẩm *", self.product_id)
        form.addRow("Tên sản phẩm *", self.name)
        form.addRow("Màu theo thứ tự slot *", self.colors)
        form.addRow("Ngưỡng confidence", self.minimum_confidence)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.save_button = QPushButton("LƯU PRODUCT PROFILE")
        self.save_button.clicked.connect(self.save_profile)
        clear = QPushButton("XÓA FORM")
        clear.clicked.connect(self.clear_form)
        self.delete_button = QPushButton("XÓA PROFILE ĐANG CHỌN")
        self.delete_button.setEnabled(False)
        self.delete_button.setStyleSheet("color: #fecaca; border-color: #b91c1c;")
        self.delete_button.clicked.connect(self.delete_selected_profile)
        buttons.addWidget(self.save_button); buttons.addWidget(clear); buttons.addWidget(self.delete_button); buttons.addStretch()
        layout.addLayout(buttons)
        layout.addWidget(QLabel("Profile đã lưu"))
        self.selection_hint = QLabel("Chọn một profile trong danh sách để sửa hoặc xóa.")
        self.selection_hint.setStyleSheet("color: #cbd5e1; padding: 3px 0;")
        layout.addWidget(self.selection_hint)
        self.list_widget.itemSelectionChanged.connect(self.load_selected_profile)
        layout.addWidget(self.list_widget)

    def refresh(self, select_product_id: str | None = None) -> None:
        selected = select_product_id if select_product_id is not None else self._selected_product_id
        self.list_widget.clear()
        for profile in self.manager.list_profiles():
            item = QListWidgetItem(
                f"{profile.product_id} — {profile.name} | {len(profile.slots)} slot | "
                f"{' - '.join(self.colors_catalog.display(color) for color in profile.expected_colors)}"
            )
            item.setData(Qt.ItemDataRole.UserRole, profile.product_id)
            self.list_widget.addItem(item)
            if profile.product_id == selected:
                self.list_widget.setCurrentItem(item)

    def clear_form(self) -> None:
        self._selected_product_id = None
        self.list_widget.blockSignals(True)
        self.list_widget.setCurrentItem(None)
        self.list_widget.clearSelection()
        self.list_widget.blockSignals(False)
        self.product_id.clear(); self.name.clear(); self.colors.clear()
        self.minimum_confidence.setValue(0.85)
        self.save_button.setText("LƯU PRODUCT PROFILE")
        self.delete_button.setEnabled(False)
        self.selection_hint.setText("Chọn một profile trong danh sách để sửa hoặc xóa.")

    def load_selected_profile(self) -> None:
        item = self.list_widget.currentItem()
        product_id = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if not isinstance(product_id, str):
            return
        profile = self.manager.get(product_id)
        if profile is None:
            return
        self._selected_product_id = profile.product_id
        self.product_id.setText(profile.product_id)
        self.name.setText(profile.name)
        self.colors.setText(", ".join(self.colors_catalog.display(color) for color in profile.expected_colors))
        self.minimum_confidence.setValue(profile.minimum_confidence)
        self.save_button.setText("CẬP NHẬT PRODUCT PROFILE")
        self.delete_button.setEnabled(True)
        self.selection_hint.setText(f"Đang chọn: {profile.product_id} — {profile.name}. Có thể cập nhật hoặc xóa profile này.")
        self.selection_hint.setStyleSheet("color: #cbd5e1; padding: 3px 0;")

    def save_profile(self) -> None:
        product_id, name = self.product_id.text().strip(), self.name.text().strip()
        entered_colors = [value.strip() for value in self.colors.text().split(",") if value.strip()]
        colors = [self.colors_catalog.normalize(value) for value in entered_colors]
        if not product_id or not name or not colors:
            QMessageBox.warning(self, "Thiếu thông tin", "Nhập mã, tên và ít nhất một màu cho Product Profile.")
            return
        unknown = [value for value, normalized in zip(entered_colors, colors, strict=True) if normalized is None]
        if unknown:
            QMessageBox.warning(
                self, "Màu chưa được cấu hình",
                f"Chưa nhận diện được: {', '.join(unknown)}. Màu hiện có: {self.colors_catalog.accepted_names()}.",
            )
            return
        slot_count = len(colors)
        slots = [SlotSpec(index=index + 1, x=(index + 0.5) / slot_count, y=0.5,
                          expected_color=color) for index, color in enumerate(colors) if color is not None]
        # Giữ lại bố cục vị trí manual khi người dùng chỉ sửa tên/màu ở trang
        # quản lý. Nếu nhập mã profile đã có thay vì chọn từ danh sách, vẫn
        # phải lấy bản cũ để không làm mất các điểm slot đã chốt trên camera.
        previous = self.manager.get(self._selected_product_id) if self._selected_product_id else self.manager.get(product_id)
        profile = ProductProfile(
            product_id, name, slots, self.minimum_confidence.value(),
            enabled=previous.enabled if previous is not None else True,
            manual_scan_strips=previous.manual_scan_strips if previous is not None else [],
            manual_scan_roi=previous.manual_scan_roi if previous is not None else None,
            manual_scan_sample_radius=previous.manual_scan_sample_radius if previous is not None else None,
        )
        try:
            if self._selected_product_id:
                self.manager.update(self._selected_product_id, profile)
                message = f"Đã cập nhật {product_id}."
            elif self.manager.get(product_id) is not None:
                answer = QMessageBox.question(
                    self, "Mã sản phẩm đã tồn tại",
                    f"Mã '{product_id}' đã tồn tại. Bạn có muốn cập nhật profile này không?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer is not QMessageBox.StandardButton.Yes:
                    return
                self.manager.update(product_id, profile)
                message = f"Đã cập nhật {product_id}."
            else:
                self.manager.save(profile)
                message = f"Đã lưu {product_id}."
        except (KeyError, ValueError) as error:
            QMessageBox.warning(self, "Không thể lưu", str(error))
            return
        self._selected_product_id = product_id
        self.refresh(product_id); self.profiles_changed.emit()
        QMessageBox.information(self, "Đã lưu", message + " Hãy xác nhận slot trên ảnh chuẩn trước khi dùng production.")

    def delete_selected_profile(self) -> None:
        product_id = self._selected_product_id
        if not product_id:
            current_item = self.list_widget.currentItem()
            current_id = current_item.data(Qt.ItemDataRole.UserRole) if current_item is not None else None
            product_id = current_id if isinstance(current_id, str) else None
        if not product_id:
            QMessageBox.information(self, "Chưa chọn profile", "Chọn một Product Profile trong danh sách trước khi xóa.")
            return
        answer = QMessageBox.question(
            self, "Xác nhận xóa Product Profile",
            f"Bạn có chắc muốn xóa profile '{product_id}'? Thao tác này không thể hoàn tác.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if not self.manager.delete(product_id):
            QMessageBox.warning(self, "Không tìm thấy", f"Profile '{product_id}' không còn tồn tại.")
            self.clear_form(); self.refresh()
            return
        self.clear_form()
        self.refresh()
        self.profiles_changed.emit()
        self.selection_hint.setText(f"ĐÃ XÓA Product Profile '{product_id}'.")
        self.selection_hint.setStyleSheet("color: #bbf7d0; font-weight: 700; padding: 3px 0;")
