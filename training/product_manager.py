"""Lưu các Product Profile đa loại vào YAML, không hard-code sản phẩm."""
from __future__ import annotations

from pathlib import Path

import yaml

from core.models import ManualStripLayout, ProductProfile, SlotSpec


class ProductManager:
    def __init__(self, path: Path) -> None:
        self.path = path

    def list_profiles(self) -> list[ProductProfile]:
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {"products": []}
        return [self._deserialize(item) for item in raw.get("products", [])]

    def get(self, product_id: str) -> ProductProfile | None:
        return next((p for p in self.list_profiles() if p.product_id == product_id), None)

    def save(self, profile: ProductProfile) -> None:
        profiles = [p for p in self.list_profiles() if p.product_id != profile.product_id]
        profiles.append(profile)
        self._write_profiles(profiles)

    def update(self, original_product_id: str, profile: ProductProfile) -> None:
        """Cập nhật profile, có thể đổi mã mà không ghi đè profile khác."""
        profiles = self.list_profiles()
        if original_product_id != profile.product_id and any(
            item.product_id == profile.product_id for item in profiles
        ):
            raise ValueError(f"Mã sản phẩm '{profile.product_id}' đã tồn tại.")
        updated = False
        replacement: list[ProductProfile] = []
        for item in profiles:
            if item.product_id == original_product_id:
                replacement.append(profile)
                updated = True
            else:
                replacement.append(item)
        if not updated:
            raise KeyError(f"Không tìm thấy Product Profile '{original_product_id}'.")
        self._write_profiles(replacement)

    def delete(self, product_id: str) -> bool:
        """Xóa profile theo mã; False khi mã không tồn tại."""
        profiles = self.list_profiles()
        replacement = [item for item in profiles if item.product_id != product_id]
        if len(replacement) == len(profiles):
            return False
        self._write_profiles(replacement)
        return True

    def _write_profiles(self, profiles: list[ProductProfile]) -> None:
        payload = {"products": [self._serialize(p) for p in profiles]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")

    @staticmethod
    def _serialize(profile: ProductProfile) -> dict:
        payload = {"product_id": profile.product_id, "name": profile.name,
                   "minimum_confidence": profile.minimum_confidence, "enabled": profile.enabled,
                   "slots": [{"index": s.index, "x": s.x, "y": s.y,
                              "expected_color": s.expected_color, "radius": s.radius} for s in profile.slots]}
        if profile.manual_scan_strips:
            payload["manual_scan"] = {
                "roi": list(profile.manual_scan_roi) if profile.manual_scan_roi is not None else None,
                "sample_radius": profile.manual_scan_sample_radius,
                "strips": [
                    {"index": strip.index, "slots": [
                        {"index": slot.index, "x": slot.x, "y": slot.y,
                         "expected_color": slot.expected_color, "radius": slot.radius}
                        for slot in strip.slots
                    ]}
                    for strip in profile.manual_scan_strips
                ],
            }
        return payload

    @staticmethod
    def _deserialize(value: dict) -> ProductProfile:
        slots = [SlotSpec(**slot) for slot in value["slots"]]
        manual_raw = value.get("manual_scan", {})
        manual_raw = manual_raw if isinstance(manual_raw, dict) else {}
        manual_strips = [
            ManualStripLayout(
                index=int(strip["index"]),
                slots=[SlotSpec(**slot) for slot in strip.get("slots", [])],
            )
            for strip in manual_raw.get("strips", [])
            if isinstance(strip, dict)
        ]
        raw_roi = manual_raw.get("roi")
        manual_roi = (
            tuple(float(component) for component in raw_roi)
            if isinstance(raw_roi, (list, tuple)) and len(raw_roi) == 4
            else None
        )
        raw_sample_radius = manual_raw.get("sample_radius")
        manual_sample_radius = (
            float(raw_sample_radius)
            if isinstance(raw_sample_radius, (int, float)) and 0.005 <= float(raw_sample_radius) <= 0.25
            else None
        )
        return ProductProfile(product_id=value["product_id"], name=value["name"], slots=slots,
                              minimum_confidence=float(value.get("minimum_confidence", 0.85)),
                              enabled=bool(value.get("enabled", True)),
                              manual_scan_strips=manual_strips,
                              manual_scan_roi=manual_roi,
                              manual_scan_sample_radius=manual_sample_radius)
