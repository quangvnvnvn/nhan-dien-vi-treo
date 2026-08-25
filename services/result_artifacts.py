"""Lưu bằng chứng ảnh/JSON cho vỉ lỗi hoặc không chắc chắn."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from core.models import FailureReason, ProductStatus, ValidationResult

LOGGER = logging.getLogger(__name__)


class ResultArtifactStore:
    """Ghi ảnh evidence khi object vượt đường đếm nhưng không đạt PASS."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def save_non_pass(self, frame: np.ndarray, track_id: int, result: ValidationResult) -> Path | None:
        if result.status is ProductStatus.PASS:
            return None
        category = self._category(result.reason)
        destination = self.root / "FAIL" / category
        destination.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        product = self._safe_part(result.product_id or "unknown")
        stem = f"{timestamp}_{product}_ID{track_id}_{category}"
        image_path = destination / f"{stem}.jpg"
        metadata_path = destination / f"{stem}.json"
        try:
            # cv2.imwrite không xử lý nhất quán đường dẫn Unicode trên Windows.
            # Encode trước rồi ghi bytes để evidence luôn được lưu đúng nơi.
            encoded_ok, encoded = cv2.imencode(image_path.suffix, frame)
            if not encoded_ok:
                raise OSError("OpenCV không ghi được ảnh")
            encoded.tofile(image_path)
            metadata_path.write_text(
                json.dumps(result.metadata(), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            return image_path
        except (OSError, cv2.error):
            LOGGER.exception("Không lưu được evidence cho track %s", track_id)
            return None

    @staticmethod
    def _category(reason: FailureReason | None) -> str:
        mapping = {
            FailureReason.MISSING_ITEM: "missing",
            FailureReason.WRONG_COLOR: "wrong_color",
            FailureReason.WRONG_PRODUCT: "wrong_product",
            FailureReason.LOW_CONFIDENCE: "low_confidence",
            FailureReason.INVALID_GEOMETRY: "invalid_geometry",
            FailureReason.UNKNOWN: "unknown",
        }
        return mapping.get(reason, "unknown")

    @staticmethod
    def _safe_part(value: str) -> str:
        return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
