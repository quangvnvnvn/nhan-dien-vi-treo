"""Các model dữ liệu dùng chung; không chứa quyết định theo từng sản phẩm."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class ProductStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class FailureReason(StrEnum):
    MISSING_ITEM = "MISSING_ITEM"
    EXTRA_ITEM = "EXTRA_ITEM"
    WRONG_COLOR = "WRONG_COLOR"
    WRONG_PRODUCT = "WRONG_PRODUCT"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    UNKNOWN = "UNKNOWN"
    INVALID_GEOMETRY = "INVALID_GEOMETRY"


@dataclass(slots=True)
class SlotSpec:
    index: int
    x: float
    y: float
    expected_color: str
    radius: float = 0.08


@dataclass(slots=True)
class ManualStripLayout:
    """Tọa độ đã được người vận hành chốt cho một vỉ trong ROI cố định.

    ``slots`` dùng tọa độ chuẩn hóa theo ROI (0..1), không theo bounding box
    mà detector tự tìm.  Nhờ vậy camera cố định có thể lấy đúng màu ở từng
    vị trí người dùng đã chọn và không cần đoán lại hình học của vỉ.
    """

    index: int
    slots: list[SlotSpec]


@dataclass(slots=True)
class ProductProfile:
    product_id: str
    name: str
    slots: list[SlotSpec]
    minimum_confidence: float = 0.85
    enabled: bool = True
    # Bố cục tùy chọn cho camera/ROI cố định. Danh sách rỗng giữ nguyên chế
    # độ detector tự tìm vỉ, tránh thay đổi hành vi của những profile cũ.
    manual_scan_strips: list[ManualStripLayout] = field(default_factory=list)
    # ROI gốc dùng lúc người vận hành chốt tọa độ. Thay ROI sẽ bị từ chối để
    # không quét nhầm vị trí cũ trên một khung hình mới.
    manual_scan_roi: tuple[float, float, float, float] | None = None
    # Bán kính *vùng lấy mẫu màu* thủ công, theo tỷ lệ cạnh ngắn của ROI.
    # ``None`` giữ hành vi cũ: tự tính theo khoảng cách các slot đã chốt.
    manual_scan_sample_radius: float | None = None

    @property
    def expected_colors(self) -> list[str]:
        return [slot.expected_color for slot in self.slots]


@dataclass(slots=True)
class SlotObservation:
    index: int
    detected: bool
    color: str | None
    slot_confidence: float
    color_confidence: float


@dataclass(slots=True)
class ValidationResult:
    product_id: str | None
    status: ProductStatus
    reason: FailureReason | None
    observations: list[SlotObservation] = field(default_factory=list)
    confidence: float = 0.0
    detail: str = ""

    def metadata(self) -> dict[str, Any]:
        return asdict(self)
