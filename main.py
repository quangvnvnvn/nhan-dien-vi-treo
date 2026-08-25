"""Điểm vào desktop. PySide6 chỉ được tải khi người dùng khởi động GUI."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from training.product_manager import ProductManager

ROOT = Path(__file__).resolve().parent


def configure_logging() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(filename=ROOT / "logs" / "app.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> int:
    configure_logging()
    try:
        from PySide6.QtWidgets import QApplication
        from ui.main_window import MainWindow
    except ImportError:
        print("Thiếu PySide6. Hãy chạy: python -m pip install -r requirements.txt")
        return 2
    manager = ProductManager(ROOT / "config" / "products.yaml")
    app = QApplication(sys.argv)
    window = MainWindow(manager, ROOT / "config" / "colors.yaml")
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
