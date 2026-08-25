"""Quy tắc PASS/FAIL/UNKNOWN: khi thiếu bằng chứng luôn chọn UNKNOWN."""
from __future__ import annotations

from core.models import FailureReason, ProductProfile, ProductStatus, SlotObservation, ValidationResult


class ProductValidator:
    def validate(self, profile: ProductProfile | None, observations: list[SlotObservation],
                 product_confidence: float) -> ValidationResult:
        if profile is None:
            return ValidationResult(None, ProductStatus.UNKNOWN, FailureReason.UNKNOWN,
                                    observations, product_confidence, "Chưa xác định được Product Profile")
        if product_confidence < profile.minimum_confidence:
            return ValidationResult(profile.product_id, ProductStatus.UNKNOWN, FailureReason.LOW_CONFIDENCE,
                                    observations, product_confidence, "Độ tin cậy model chưa đạt ngưỡng")
        if len(observations) != len(profile.slots):
            return ValidationResult(profile.product_id, ProductStatus.UNKNOWN, FailureReason.UNKNOWN,
                                    observations, product_confidence, "Số slot quan sát không khớp profile")
        # Thiếu viên là lỗi vật lý dứt khoát. Kiểm tra trước màu để một slot
        # khác bị chói/nhòe không che mất kết luận NG "thiếu viên".
        for spec, observed in zip(profile.slots, observations, strict=True):
            if not observed.detected:
                return ValidationResult(profile.product_id, ProductStatus.FAIL, FailureReason.MISSING_ITEM,
                                        observations, product_confidence, f"Thiếu viên tại slot {spec.index}")
        for spec, observed in zip(profile.slots, observations, strict=True):
            if observed.color is None or observed.color_confidence < profile.minimum_confidence:
                # Hình học đã xác nhận một slot có viên màu, nhưng màu đó
                # không thuộc bất kỳ màu hợp lệ nào của hệ thống: đây là NG
                # (sai màu), không phải REVIEW. REVIEW chỉ dành cho slot mờ
                # hoặc frame không đủ chất lượng để nhìn thấy viên.
                if observed.detected and observed.slot_confidence >= profile.minimum_confidence:
                    return ValidationResult(
                        profile.product_id,
                        ProductStatus.FAIL,
                        FailureReason.WRONG_COLOR,
                        observations,
                        product_confidence,
                        f"Màu không hợp lệ hoặc sai vị trí tại slot {spec.index}",
                    )
                return ValidationResult(profile.product_id, ProductStatus.UNKNOWN, FailureReason.LOW_CONFIDENCE,
                                        observations, product_confidence, f"Màu slot {spec.index} chưa chắc chắn")
            if observed.color != spec.expected_color:
                return ValidationResult(profile.product_id, ProductStatus.FAIL, FailureReason.WRONG_COLOR,
                                        observations, product_confidence, f"Sai màu tại slot {spec.index}")
        confidence = min([product_confidence, *(o.slot_confidence for o in observations),
                          *(o.color_confidence for o in observations)])
        return ValidationResult(profile.product_id, ProductStatus.PASS, None, observations, confidence, "Đạt tất cả quy tắc")
