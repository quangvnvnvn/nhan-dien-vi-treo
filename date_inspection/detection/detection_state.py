from __future__ import annotations

from enum import Enum


class DetectionState(str, Enum):
    IDLE = "KHÔNG CÓ SẢN PHẨM"
    ENTERING = "SẢN PHẨM ĐANG VÀO"
    PRESENT = "SẢN PHẨM ĐANG TRONG VÙNG"
    CAPTURED = "ĐÃ TRIGGER - ĐỢI SẢN PHẨM RA"
    EXITING = "SẢN PHẨM ĐÃ RA"
