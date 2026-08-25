"""Danh mục màu cấu hình được: chuẩn hóa đầu vào và hiển thị tiếng Việt."""
from __future__ import annotations

from typing import Any


class ColorCatalog:
    def __init__(self, profiles: dict[str, Any]) -> None:
        self.profiles = profiles
        self._lookup: dict[str, str] = {}
        for key, profile in profiles.items():
            terms = [key, str(profile.get("display_name", key)), *profile.get("aliases", [])]
            for term in terms:
                self._lookup[self._key(str(term))] = key

    def normalize(self, name: str) -> str | None:
        """Chuyển tên người dùng nhập thành khóa màu trong cấu hình."""
        return self._lookup.get(self._key(name))

    def display(self, name: str | None) -> str:
        if not name:
            return "--"
        profile = self.profiles.get(name)
        return str(profile.get("display_name", name)) if profile else name

    def accepted_names(self) -> str:
        return ", ".join(self.display(key) for key in self.profiles)

    @staticmethod
    def _key(value: str) -> str:
        return " ".join(value.strip().lower().replace("_", " ").split())
