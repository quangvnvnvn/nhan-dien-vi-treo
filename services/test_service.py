"""TEST MODE: chỉ kiểm tra ảnh tĩnh, không tự huấn luyện hay tự ghi profile."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ai.color_classifier import ColorClassifier
from ai.profile_color_detector import ProfileColorSequenceDetector
from ai.product_detector import GeometricProductDetector
from ai.validator import ProductValidator
from core.models import FailureReason, ProductProfile, ProductStatus, SlotObservation, ValidationResult


class TestService:
    def __init__(self, color_classifier: ColorClassifier, validator: ProductValidator,
                 detector: GeometricProductDetector | None = None) -> None:
        self.color_classifier = color_classifier
        self.validator = validator
        self.detector = detector or GeometricProductDetector()
        self.profile_color_detector = ProfileColorSequenceDetector(color_classifier.profiles)

    def inspect_image(self, image_path: Path, profile: ProductProfile | None) -> ValidationResult:
        # cv2.imread có thể thất bại trên Windows khi đường dẫn chứa tiếng Việt.
        # Đọc bytes qua NumPy rồi decode giúp TEST MODE làm việc ổn định với
        # tên thư mục/tệp do người vận hành đặt.
        try:
            encoded = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
        except (OSError, ValueError):
            image = None
        if image is None:
            raise ValueError(f"Không đọc được ảnh: {image_path}")
        if profile is None:
            return self.validator.validate(None, [], 0.0)
        detection = self.detector.detect(image, len(profile.slots))
        if not detection.slots:
            # Khi các ô cùng màu dính nhau qua vách nhựa, Hough circle không
            # thể tách từng vòng riêng. Fallback này chỉ dùng chuỗi màu đã
            # khai báo trong profile và kết quả vẫn đi qua validator.
            detection = self.profile_color_detector.detect(image, profile)
        if not detection.slots:
            partial = self._detect_partial(image, len(profile.slots))
            if partial is not None and len(partial.slots) == len(profile.slots) - 1:
                observations = [
                    SlotObservation(
                        spec.index,
                        index < len(partial.slots),
                        spec.expected_color if index < len(partial.slots) else None,
                        0.9 if index < len(partial.slots) else 0.0,
                        0.9 if index < len(partial.slots) else 0.0,
                    )
                    for index, spec in enumerate(profile.slots)
                ]
                result = self.validator.validate(profile, observations, partial.confidence)
                result.detail = f"{result.detail}. {partial.detail}"
                return result
            return ValidationResult(
                profile.product_id, ProductStatus.UNKNOWN, FailureReason.UNKNOWN, [], 0.0,
                f"Không xác thực được vỉ: {detection.detail}",
            )
        return self._validate_slots(image, profile, detection.slots, detection.confidence, detection.detail)

    def inspect_normalized_product(self, image: np.ndarray, profile: ProductProfile | None) -> ValidationResult:
        """Kiểm tra crop vỉ đã được detector/perspective module xác nhận."""
        if profile is None:
            return self.validator.validate(None, [], 0.0)
        slots = [
            (round(spec.x * image.shape[1]), round(spec.y * image.shape[0]),
             max(4, round(spec.radius * min(image.shape[:2]))))
            for spec in profile.slots
        ]
        return self._validate_slots(image, profile, slots, 0.9, "Crop vỉ đã được chuẩn hóa")

    def _validate_slots(self, image: np.ndarray, profile: ProductProfile,
                        slots: list[object], product_confidence: float, detail: str) -> ValidationResult:
        forward_observations = self._observe_slots(image, profile, slots)
        forward = self.validator.validate(profile, forward_observations, product_confidence=product_confidence)
        reverse_observations = self._observe_slots(image, profile, list(reversed(slots)))
        reverse = self.validator.validate(profile, reverse_observations, product_confidence=product_confidence)
        if forward.status is ProductStatus.PASS:
            result = forward
        elif reverse.status is ProductStatus.PASS:
            result = reverse
            result.detail = f"{result.detail} (đã tự căn chiều vỉ đảo ngược)"
        elif forward.status is ProductStatus.FAIL:
            result = forward
        elif reverse.status is ProductStatus.FAIL:
            result = reverse
            result.detail = f"{result.detail} (đã tự căn chiều vỉ đảo ngược)"
        elif forward.status is ProductStatus.UNKNOWN or reverse.status is ProductStatus.UNKNOWN:
            result = ValidationResult(
                profile.product_id, ProductStatus.UNKNOWN, FailureReason.UNKNOWN,
                forward_observations, min(forward.confidence, reverse.confidence),
                "Không đủ bằng chứng màu để xác định chiều vỉ",
            )
        else:
            result = forward
        result.detail = f"{result.detail}. {detail}"
        return result

    def _observe_slots(self, image: np.ndarray, profile: ProductProfile,
                       slots: list[object]) -> list[SlotObservation]:
        h, w = image.shape[:2]
        samples: list[np.ndarray] = []
        slot_metadata: list[tuple[object, bool]] = []
        for detected_slot in slots:
            if hasattr(detected_slot, "x"):
                x, y, radius = detected_slot.x, detected_slot.y, detected_slot.radius
            else:
                x, y, radius = detected_slot
            mask = np.zeros((h, w), dtype=np.uint8)
            sample_radius = getattr(detected_slot, "sample_radius", None)
            cv2.circle(mask, (x, y), sample_radius or max(4, round(radius * 0.45)), 255, -1)
            pixels = image[mask > 0]
            samples.append(pixels)
            slot_metadata.append((detected_slot, self._has_colored_item(pixels)))

        classify_sequence = getattr(self.color_classifier, "classify_sequence", None)
        if callable(classify_sequence):
            colors = list(classify_sequence(samples, profile.expected_colors))
        else:
            colors = [self.color_classifier.classify(pixels) for pixels in samples]

        observations: list[SlotObservation] = []
        for spec, (detected_slot, has_colored_item), color in zip(profile.slots, slot_metadata, colors, strict=True):
            matched_color = getattr(detected_slot, "matched_color", None)
            visible = matched_color is not None or has_colored_item
            # Component profile chỉ giúp định vị, không được ghi đè một màu
            # độc lập đang mâu thuẫn với nó.
            observed_color = color.name if visible and color else None
            color_confidence = color.confidence if visible and color else 0.0
            observations.append(SlotObservation(spec.index, visible, observed_color,
                                                0.9 if visible else 0.0, color_confidence))
        return observations

    def _detect_partial(self, image: np.ndarray, expected_slots: int) -> object | None:
        detect_partial = getattr(self.detector, "detect_partial", None)
        if not callable(detect_partial):
            return None
        try:
            return detect_partial(image, expected_slots)
        except Exception:
            return None

    @staticmethod
    def _has_colored_item(pixels: np.ndarray) -> bool:
        if len(pixels) == 0:
            return False
        try:
            hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        except cv2.error:
            return False
        colored = (hsv[:, 1] >= 35) & (hsv[:, 2] >= 25)
        return int(np.count_nonzero(colored)) >= max(12, int(np.ceil(len(hsv) * 0.05)))
