"""Cửa sổ chính, kết nối Product Manager và TEST MODE."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow, QTabWidget

from ai.color_classifier import ColorClassifier
from ai.validator import ProductValidator
from core.colors import ColorCatalog
from core.config import load_yaml
from services.realtime_controller import RealtimeController
from services.test_service import TestService
from training.product_manager import ProductManager
from ui.product_page import ProductPage
from ui.realtime_page import RealtimePage
from ui.test_page import TestPage
from ui.history_page import HistoryPage


class MainWindow(QMainWindow):
    def __init__(self, manager: ProductManager, color_profiles_path: Path) -> None:
        super().__init__()
        self.setWindowTitle("Hệ thống nhận diện và đếm vỉ")
        self.resize(1280, 820)
        root = color_profiles_path.parent.parent
        colors = load_yaml(color_profiles_path).get("colors", {})
        config = load_yaml(root / "config" / "config.yaml")
        catalog = ColorCatalog(colors)
        service = TestService(ColorClassifier(colors), ProductValidator())
        self.product_page = ProductPage(manager, catalog)
        self.test_page = TestPage(manager, service, catalog)
        self.realtime_page = RealtimePage(manager, catalog)
        self.realtime_controller = RealtimeController(self.realtime_page, manager, catalog, colors, root, config)
        self.history_page = HistoryPage(root / "data" / "exports")
        self.product_page.profiles_changed.connect(self.test_page.refresh_profiles)
        self.product_page.profiles_changed.connect(self.realtime_page.refresh_profiles)
        self.realtime_page.profiles_changed.connect(self.test_page.refresh_profiles)
        self.realtime_page.profiles_changed.connect(self.product_page.refresh)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.realtime_page, "CAMERA REALTIME")
        self.tabs.addTab(self.test_page, "TEST MODE")
        self.tabs.addTab(self.history_page, "TRA CỨU EXCEL")
        self.tabs.addTab(self.product_page, "PRODUCT MANAGER")
        self.tabs.currentChanged.connect(self._refresh_history_when_opened)
        self.setCentralWidget(self.tabs)

    def _refresh_history_when_opened(self, index: int) -> None:
        if self.tabs.widget(index) is self.history_page:
            self.history_page.refresh()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API name
        self.realtime_controller.shutdown()
        event.accept()
