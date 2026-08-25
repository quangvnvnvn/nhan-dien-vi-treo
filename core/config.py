"""Đọc cấu hình thay đổi được mà không nhúng ngưỡng vào mã nguồn."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Cấu hình phải là mapping: {path}")
    return value
