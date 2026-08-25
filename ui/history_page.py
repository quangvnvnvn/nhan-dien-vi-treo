"""Tra cứu các kết quả realtime đã lưu theo ngày trong ứng dụng."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from PySide6.QtCore import QDate, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from services.daily_excel_exporter import DailyExcelExporter


class HistoryPage(QWidget):
    """Hiển thị nhật ký JSONL vốn là nguồn của file Excel hằng ngày."""

    def __init__(self, export_directory: Path) -> None:
        super().__init__()
        self.exporter = DailyExcelExporter(export_directory)
        self._records: list[dict[str, object]] = []
        self._build()
        self.refresh()

    def _build(self) -> None:
        self.setObjectName("historyPage")
        self.setStyleSheet(
            "QFrame#historyCard { border: 1px solid #475569; border-radius: 8px; background: #303030; }"
            "QLabel#historyTitle { font-size: 20px; font-weight: 800; color: #f8fafc; }"
            "QLabel#historySubtitle { color: #cbd5e1; }"
            "QPushButton#historyPrimary { background: #0369a1; color: #ffffff; font-weight: 800; "
            "padding: 7px 14px; border: 1px solid #38bdf8; border-radius: 6px; }"
            "QPushButton#historyPrimary:hover { background: #0284c7; }"
            "QTableWidget { gridline-color: #4b5563; selection-background-color: #1d4ed8; }"
            "QHeaderView::section { background: #3f3f46; color: #f8fafc; padding: 7px; "
            "border: 0; border-right: 1px solid #52525b; font-weight: 700; }"
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        heading = QHBoxLayout()
        heading_text = QVBoxLayout()
        title = QLabel("TRA CỨU KẾT QUẢ")
        title.setObjectName("historyTitle")
        subtitle = QLabel("Chọn ngày để xem lại các vỉ đã đi qua đường đếm và mở file Excel tương ứng.")
        subtitle.setObjectName("historySubtitle")
        heading_text.addWidget(title)
        heading_text.addWidget(subtitle)
        heading.addLayout(heading_text)
        heading.addStretch()
        self.result_state = QLabel("SẴN SÀNG")
        self.result_state.setStyleSheet(
            "color: #bae6fd; border: 1px solid #0369a1; border-radius: 10px; padding: 4px 10px; font-weight: 800;"
        )
        heading.addWidget(self.result_state)
        root.addLayout(heading)

        controls = QFrame()
        controls.setObjectName("historyCard")
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(12, 10, 12, 10)
        self.date_selector = QDateEdit(QDate.currentDate())
        self.date_selector.setCalendarPopup(True)
        self.date_selector.setDisplayFormat("dd/MM/yyyy")
        self.date_selector.dateChanged.connect(self._load_selected_date)
        self.available_dates = QComboBox()
        self.available_dates.currentIndexChanged.connect(self._select_available_date)
        refresh_button = QPushButton("↻ TẢI KẾT QUẢ")
        refresh_button.setObjectName("historyPrimary")
        refresh_button.clicked.connect(self.refresh)
        self.open_excel_button = QPushButton("MỞ FILE EXCEL")
        self.open_excel_button.clicked.connect(self.open_selected_excel)
        open_folder_button = QPushButton("MỞ THƯ MỤC EXCEL")
        open_folder_button.clicked.connect(self.open_export_directory)
        controls_layout.addWidget(QLabel("Ngày:"))
        controls_layout.addWidget(self.date_selector)
        controls_layout.addWidget(QLabel("Ngày có dữ liệu:"))
        controls_layout.addWidget(self.available_dates, 1)
        controls_layout.addWidget(refresh_button)
        controls_layout.addWidget(self.open_excel_button)
        controls_layout.addWidget(open_folder_button)
        root.addWidget(controls)

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Tìm nhanh:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Nhập mã sản phẩm, PASS / FAIL / REVIEW, màu hoặc ghi chú…")
        self.search_input.textChanged.connect(self._render_records)
        search_row.addWidget(self.search_input, 1)
        root.addLayout(search_row)

        summary = QFrame()
        summary.setObjectName("historyCard")
        summary_layout = QGridLayout(summary)
        self.metrics: dict[str, QLabel] = {}
        for index, (key, label) in enumerate((
            ("total", "TỔNG LƯỢT"), ("pass", "ĐẠT"), ("fail", "KHÔNG ĐẠT"),
            ("unknown", "CẦN REVIEW"), ("counted", "ĐÃ ĐẾM"),
        )):
            caption = QLabel(label)
            caption.setStyleSheet("color: #cbd5e1; font-weight: 700;")
            value = QLabel("0")
            value.setStyleSheet("font-size: 22px; font-weight: 800; color: #f8fafc;")
            summary_layout.addWidget(caption, 0, index, Qt.AlignmentFlag.AlignCenter)
            summary_layout.addWidget(value, 1, index, Qt.AlignmentFlag.AlignCenter)
            self.metrics[key] = value
        root.addWidget(summary)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels((
            "Thời gian", "Sản phẩm", "Kết quả", "Màu phát hiện", "Độ tin cậy",
            "Tăng đếm", "Mã track", "Lý do", "Ghi chú",
        ))
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.table, 1)

        self.notice = QLabel()
        self.notice.setWordWrap(True)
        self.notice.setStyleSheet("padding: 8px; border: 1px solid #475569; border-radius: 6px; color: #cbd5e1;")
        root.addWidget(self.notice)

    @property
    def selected_day(self) -> date:
        value = self.date_selector.date()
        return date(value.year(), value.month(), value.day())

    def refresh(self) -> None:
        """Nạp lại ngày có data; được gọi cả khi người dùng chuyển sang tab."""
        available_days = self.exporter.available_days()
        selected = self.selected_day
        self.available_dates.blockSignals(True)
        self.available_dates.clear()
        self.available_dates.addItem("-- Chọn ngày đã có dữ liệu --", None)
        for value in available_days:
            self.available_dates.addItem(value.strftime("%d/%m/%Y"), value.isoformat())
        self.available_dates.blockSignals(False)
        if selected not in available_days and available_days:
            selected = available_days[0]
            self.date_selector.blockSignals(True)
            self.date_selector.setDate(QDate(selected.year, selected.month, selected.day))
            self.date_selector.blockSignals(False)
        index = self.available_dates.findData(selected.isoformat())
        self.available_dates.setCurrentIndex(index if index >= 0 else 0)
        self._load_selected_date()

    def _select_available_date(self, _index: int) -> None:
        value = self.available_dates.currentData()
        if not isinstance(value, str):
            return
        try:
            chosen = date.fromisoformat(value)
        except ValueError:
            return
        self.date_selector.setDate(QDate(chosen.year, chosen.month, chosen.day))

    def _load_selected_date(self, *_args: object) -> None:
        self._records = self.exporter.records_for_day(self.selected_day)
        self._set_summary()
        self._render_records()
        excel_path = self.exporter.workbook_path_for(datetime.combine(self.selected_day, datetime.min.time()))
        self.open_excel_button.setEnabled(excel_path.is_file())
        if self._records:
            self.result_state.setText(f"{len(self._records)} KẾT QUẢ")
            self.result_state.setStyleSheet(
                "color: #bbf7d0; border: 1px solid #16a34a; border-radius: 10px; padding: 4px 10px; font-weight: 800;"
            )
            self.notice.setText(f"Đang xem ngày {self.selected_day:%d/%m/%Y}. Có thể tìm nhanh hoặc mở file Excel.")
        else:
            self.result_state.setText("CHƯA CÓ DỮ LIỆU")
            self.result_state.setStyleSheet(
                "color: #fde68a; border: 1px solid #ca8a04; border-radius: 10px; padding: 4px 10px; font-weight: 800;"
            )
            self.notice.setText(
                f"Chưa có event nào qua đường đếm trong ngày {self.selected_day:%d/%m/%Y}. "
                "Kết quả sẽ xuất hiện tự động sau lần kiểm tra tiếp theo."
            )

    def _set_summary(self) -> None:
        statuses = [str(item.get("status", "UNKNOWN")).upper() for item in self._records]
        counted = sum(int(item.get("count_increment", 0) or 0) for item in self._records)
        values = {
            "total": len(self._records), "pass": statuses.count("PASS"), "fail": statuses.count("FAIL"),
            "unknown": statuses.count("UNKNOWN"), "counted": counted,
        }
        for key, value in values.items():
            self.metrics[key].setText(str(value))

    def _render_records(self, *_args: object) -> None:
        query = self.search_input.text().strip().casefold()
        self.table.setRowCount(0)
        for record in self._records:
            values = self._record_values(record)
            if query and query not in " ".join(values).casefold():
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 2:
                    status = value.upper()
                    fills = {"PASS": "#166534", "FAIL": "#991b1b", "UNKNOWN": "#854d0e"}
                    item.setBackground(QColor(fills.get(status, "#475569")))
                    item.setForeground(QColor("#ffffff"))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                elif column in {0, 4, 5, 6}:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, column, item)

    @staticmethod
    def _record_values(record: dict[str, object]) -> tuple[str, ...]:
        confidence = float(record.get("confidence", 0.0) or 0.0)
        return (
            str(record.get("time", "--")), str(record.get("product_id", "--")),
            str(record.get("status", "UNKNOWN")), str(record.get("colors", "--")),
            f"{max(0.0, min(1.0, confidence)):.1%}", str(record.get("count_increment", 0)),
            str(record.get("track_id", "--")), str(record.get("reason", "--")),
            str(record.get("detail", "--")),
        )

    def open_selected_excel(self) -> None:
        path = self.exporter.workbook_path_for(datetime.combine(self.selected_day, datetime.min.time()))
        if not path.is_file():
            QMessageBox.information(self, "Chưa có file Excel", "Ngày đã chọn chưa có file Excel để mở.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def open_export_directory(self) -> None:
        self.exporter.output_directory.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.exporter.output_directory)))
