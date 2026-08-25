"""Đọc/ghi cấu hình cục bộ, không phụ thuộc nguồn camera hay video."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigManager:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path(__file__).with_name("settings.json")
        self.data: dict[str, Any] = {"detector": {}, "rois": {}}
        self.load()

    def load(self) -> None:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                self.data.update(value)
        except (OSError, ValueError):
            pass

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def detector(self) -> dict[str, Any]:
        return dict(self.data.get("detector", {}))

    def set_detector(self, values: dict[str, Any]) -> None:
        self.data["detector"] = values
        self.save()

    def roi(self, source_key: str, kind: str) -> tuple[float, float, float, float] | None:
        value = self.data.get("rois", {}).get(source_key, {}).get(kind)
        if not isinstance(value, list) or len(value) != 4:
            return None
        try:
            roi = tuple(float(item) for item in value)
        except (TypeError, ValueError):
            return None
        return roi if roi[2] > 0 and roi[3] > 0 else None

    def set_roi(self, source_key: str, kind: str, roi: tuple[float, float, float, float] | None) -> None:
        rois = self.data.setdefault("rois", {}).setdefault(source_key, {})
        if roi is None:
            rois.pop(kind, None)
        else:
            rois[kind] = list(roi)
        self.save()
