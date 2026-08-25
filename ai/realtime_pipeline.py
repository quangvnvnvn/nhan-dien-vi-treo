"""Pipeline kiểm tra một khung hình camera theo hướng an toàn.

Module này không đếm sản phẩm và không giữ trạng thái giữa các khung hình.  Nó
trả về crop, bounding box và khung hình đã chú thích để lớp camera/tracker có
thể dùng lại.  Với detector hình học hiện có, mọi trường hợp không đủ bằng
chứng (góc nhìn khó, nhiều vỉ, màu mơ hồ, profile không hợp lệ) đều là
``UNKNOWN``; pipeline không suy đoán ``PASS``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import combinations
import logging
from math import ceil, floor
from typing import Callable, TypeAlias

import cv2
import numpy as np

from ai.color_classifier import ColorClassifier
from ai.profile_color_detector import ProfileColorSequenceDetector
from ai.product_detector import DetectedSlot, GeometricProductDetector, ProductDetection
from ai.validator import ProductValidator
from core.models import (
    FailureReason,
    ManualStripLayout,
    ProductProfile,
    ProductStatus,
    SlotObservation,
    SlotSpec,
    ValidationResult,
)
from training.product_manager import ProductManager


BoundingBox: TypeAlias = tuple[int, int, int, int]
"""Bounding box theo thứ tự ``(x, y, width, height)`` trong khung BGR gốc."""

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SlotOverlay:
    """Thông tin một slot để UI vẽ hoặc tracker liên kết với khung hình."""

    index: int
    center: tuple[int, int]
    radius: int
    detected: bool
    color: str | None
    color_confidence: float
    slot_confidence: float


@dataclass(slots=True)
class ProductInspection:
    """Một candidate vỉ trong khung hình.

    ``validation`` có thể là UNKNOWN dù detector tìm được candidate, ví dụ khi
    cùng một frame chứa nhiều vỉ.  Caller không nên dùng một candidate UNKNOWN
    làm tín hiệu đếm/loại hàng tự động.
    """

    product_id: str
    bounding_box: BoundingBox
    validation: ValidationResult
    product_crop: np.ndarray
    slots: list[SlotOverlay]
    detector_confidence: float
    detector_detail: str
    slot_order: str = "forward"


@dataclass(slots=True)
class FrameInspectionResult:
    """Kết quả một lần xử lý frame, sẵn sàng cho UI camera hoặc tracker."""

    selected_product_id: str | None
    validation: ValidationResult
    bounding_box: BoundingBox | None
    product_crop: np.ndarray | None
    detections: list[ProductInspection]
    annotated_frame: np.ndarray
    detail: str


class FrameInspectionPipeline:
    """Ghép profile, detector, phân loại màu và validator cho một BGR frame.

    Product ID nên được chọn tại UI.  Khi không có Product ID, pipeline không
    tự đoán loại vỉ từ danh sách profile vì hành vi đó có thể tạo PASS nhầm.
    Detector hiện tại là detector hình học; do đó pipeline cố ý trả UNKNOWN
    với các góc che khuất/phối cảnh mà detector chưa chứng minh được.
    """

    def __init__(
        self,
        product_manager: ProductManager,
        color_classifier: ColorClassifier,
        detector: GeometricProductDetector | None = None,
        validator: ProductValidator | None = None,
        *,
        minimum_detector_confidence: float = 0.85,
        check_for_multiple_candidates: bool = True,
        allow_multiple_candidates: bool = False,
        maximum_monitor_candidates: int = 12,
        enable_color_guided_fallback: bool = False,
        color_guided_max_dimension: int = 480,
        profile_component_limit_per_color: int = 6,
        strict_position_color_validation: bool = False,
        color_normalizer: Callable[[str], str | None] | None = None,
    ) -> None:
        if not 0.0 <= minimum_detector_confidence <= 1.0:
            raise ValueError("minimum_detector_confidence phải nằm trong [0, 1]")
        self.product_manager = product_manager
        self.color_classifier = color_classifier
        self.detector = detector or GeometricProductDetector()
        self.validator = validator or ProductValidator()
        self.minimum_detector_confidence = minimum_detector_confidence
        self.check_for_multiple_candidates = check_for_multiple_candidates
        # Test Mode giữ hành vi một candidate an toàn. Realtime conveyor có
        # thể có hai vỉ cùng lúc trong khung, nên controller bật cờ này để
        # tracker nhận từng vỉ đã PASS một cách độc lập.
        self.allow_multiple_candidates = allow_multiple_candidates
        # Chế độ giám sát ROI có thể chứa nhiều vỉ cùng lúc.  Giới hạn hữu
        # hạn bảo vệ tốc độ realtime, còn detector được chạy lặp trên phần
        # frame chưa bị che bởi candidate đã xác nhận.
        self.maximum_monitor_candidates = max(1, int(maximum_monitor_candidates))
        # Fallback màu chỉ cần định vị tâm vùng màu, sau đó validator vẫn lấy
        # mẫu ở ảnh gốc. Giới hạn ảnh phân tích và số component tránh việc ROI
        # có 10+ vỉ tạo hàng nghìn tổ hợp PCA cho mỗi lượt quét.
        self.color_guided_max_dimension = max(160, int(color_guided_max_dimension))
        self.profile_component_limit_per_color = max(4, int(profile_component_limit_per_color))
        # Với camera/băng tải cố định, mỗi tâm slot sau khi tìm được phải có
        # bằng chứng màu độc lập ngay tại vị trí đó. Không cho nhãn component
        # màu (dùng để tìm hình học) tự biến thành bằng chứng PASS.
        self.strict_position_color_validation = strict_position_color_validation
        self.profile_color_detector = ProfileColorSequenceDetector(color_classifier.profiles)
        # Chỉ bật ở realtime production.  Nhánh này dùng thứ tự màu của
        # Product Profile để cứu trường hợp vỉ nhỏ trong ROI rộng, nhưng vẫn
        # chỉ thay thế candidate thường khi toàn bộ validation đạt PASS.
        self.enable_color_guided_fallback = enable_color_guided_fallback
        self._color_normalizer = color_normalizer or self._default_color_normalizer

    def inspect_frame(
        self,
        frame_bgr: np.ndarray,
        selected_product_id: str | None = None,
        *,
        allow_profile_color_fallback: bool = False,
        roi: tuple[float, float, float, float] | None = None,
    ) -> FrameInspectionResult:
        """Kiểm tra duy nhất một BGR frame và không thay đổi profile/database.

        Invalid input và lỗi detector được chuyển thành UNKNOWN để vòng lặp
        camera không bị dừng.  ``annotated_frame`` luôn là một ndarray BGR độc
        lập với frame đầu vào.
        """
        if not self._is_valid_bgr_frame(frame_bgr):
            LOGGER.warning("Bỏ qua frame không hợp lệ: type=%s", type(frame_bgr).__name__)
            canvas = np.zeros((1, 1, 3), dtype=np.uint8)
            validation = self._unknown(None, "Frame camera không đúng định dạng BGR uint8")
            return FrameInspectionResult(
                selected_product_id,
                validation,
                None,
                None,
                [],
                canvas,
                validation.detail,
            )

        canvas = np.ascontiguousarray(frame_bgr.copy())
        profile, profile_error = self._get_safe_profile(selected_product_id)
        if profile is None:
            validation = self._unknown(None, profile_error)
            self._draw_banner(canvas, validation)
            return FrameInspectionResult(
                selected_product_id,
                validation,
                None,
                None,
                [],
                canvas,
                validation.detail,
            )

        normalized_profile, normalization_error = self._normalize_profile_colors(profile)
        if normalized_profile is None:
            validation = self._unknown(profile, normalization_error)
            self._draw_banner(canvas, validation)
            return FrameInspectionResult(
                selected_product_id,
                validation,
                None,
                None,
                [],
                canvas,
                validation.detail,
            )

        LOGGER.debug(
            "Bắt đầu kiểm tra frame: profile=%s, shape=%s",
            normalized_profile.product_id,
            tuple(frame_bgr.shape),
        )
        # Camera/ROI cố định có thể được người vận hành chốt sẵn từng tâm
        # slot, kể cả khi có nhiều vỉ đồng thời. Khi đó không gọi Hough hay
        # gom component màu nữa: lấy mẫu đúng tại các tọa độ đã chốt nhanh hơn
        # và loại hẳn lỗi chọn nhầm slot từ vỉ lân cận.
        if normalized_profile.manual_scan_strips:
            if not self._manual_layout_matches_roi(normalized_profile, roi):
                detail = (
                    "ROI hiện tại khác ROI đã dùng khi khóa vị trí thủ công. "
                    "Chọn lại ROI cũ hoặc thiết lập/lưu lại vị trí slot."
                )
                validation = self._unknown(normalized_profile, detail)
                self._draw_banner(canvas, validation)
                return FrameInspectionResult(
                    selected_product_id,
                    validation,
                    None,
                    None,
                    [],
                    canvas,
                    detail,
                )
            return self._inspect_manual_scan_layout(frame_bgr, normalized_profile, canvas)

        primary = self._detect(frame_bgr, len(normalized_profile.slots))
        primary_inspection = (
            self._inspect_candidate(frame_bgr, normalized_profile, primary)
            if primary is not None and primary.slots
            else None
        )
        used_global_profile_color_fallback = False

        # Trước hết căn fallback vào đúng candidate hình học. Cách này sửa
        # hue của slot 3 nhưng không lấy nhầm cụm màu của vỉ khác khi hai vỉ
        # xuất hiện đồng thời trên băng tải.
        if (
            self.enable_color_guided_fallback
            and allow_profile_color_fallback
            and primary_inspection is not None
            and primary_inspection.validation.status is not ProductStatus.PASS
        ):
            local_primary = self._guided_inspection_in_box(
                frame_bgr,
                normalized_profile,
                primary_inspection.bounding_box,
            )
            if local_primary is not None:
                primary, primary_inspection = local_primary

        # Hough có thể chọn nhầm tâm slot hoặc lấy cả phản xạ nhựa, làm màu
        # xanh dương tối bị đọc thành Tím (FAIL) dù chuỗi profile thấy đúng.
        # Fallback chỉ được phép thay thế khi tự xác minh đủ toàn bộ profile
        # thành PASS; vì vậy nó có thể sửa cả FAIL sai màu nhưng không thể biến
        # hình nền mơ hồ hoặc vỉ thiếu slot thành PASS.
        if (
            self.enable_color_guided_fallback
            and allow_profile_color_fallback
            and (primary_inspection is None or primary_inspection.validation.status is not ProductStatus.PASS)
        ):
            guided = self.profile_color_detector.detect(frame_bgr, normalized_profile)
            if guided.slots:
                guided_inspection = self._inspect_candidate(frame_bgr, normalized_profile, guided)
                if guided_inspection.validation.status is ProductStatus.PASS:
                    primary = guided
                    primary_inspection = guided_inspection
                    used_global_profile_color_fallback = True
                    LOGGER.info("Dùng fallback màu+hình học cho profile=%s", normalized_profile.product_id)

        # Ở băng tải thực tế vỉ có thể nhỏ trong ROI cao và nền sáng làm những
        # vùng màu bị chia nhỏ. ``ProfileColorSequenceDetector`` ưu tiên run
        # màu lớn để chống nhiễu; nhánh này dùng từng component màu nhỏ rồi
        # kiểm chứng lại hàng 5 slot, nên cứu được vỉ rõ nhưng xa camera mà
        # không hạ tiêu chuẩn PASS.
        if (
            self.enable_color_guided_fallback
            and allow_profile_color_fallback
            and (primary_inspection is None or primary_inspection.validation.status is not ProductStatus.PASS)
        ):
            small_colored = self._detect_profile_colored_row(frame_bgr, normalized_profile)
            if small_colored is not None:
                small_inspection = self._inspect_candidate(frame_bgr, normalized_profile, small_colored)
                if small_inspection.validation.status is ProductStatus.PASS:
                    primary = small_colored
                    primary_inspection = small_inspection
                    used_global_profile_color_fallback = True
                    LOGGER.info("Dùng fallback component màu nhỏ cho profile=%s", normalized_profile.product_id)

        if primary_inspection is None:
            # Khi thiếu đúng một viên, detector đủ-slot cố ý trả rỗng để không
            # thể PASS nhầm. Tuy vậy 4/5 slot có hình học rõ ràng là bằng chứng
            # đủ để loại NG; đưa candidate này vào tracker thay vì thả trôi
            # thành UNKNOWN. Nhánh partial chỉ chạy sau mọi đường PASS.
            partial = self._detect_partial(frame_bgr, len(normalized_profile.slots))
            partial_inspection = (
                self._inspect_missing_slot_candidate(frame_bgr, normalized_profile, partial)
                if partial is not None and partial.slots
                else None
            )
            if partial_inspection is not None:
                self._draw_inspection(canvas, partial_inspection)
                return FrameInspectionResult(
                    selected_product_id,
                    partial_inspection.validation,
                    partial_inspection.bounding_box,
                    partial_inspection.product_crop,
                    [partial_inspection],
                    canvas,
                    partial_inspection.validation.detail,
                )
            detail = primary.detail if primary is not None else "Detector gặp lỗi khi xử lý frame"
            validation = self._unknown(normalized_profile, f"Không xác thực được vỉ: {detail}")
            self._draw_banner(canvas, validation)
            return FrameInspectionResult(
                selected_product_id,
                validation,
                None,
                None,
                [],
                canvas,
                validation.detail,
            )

        # Một ROI vận hành có thể có nhiều vỉ.  Detector chỉ trả candidate tốt
        # nhất, vì vậy lần lượt mask từng candidate đã xác nhận rồi kiểm tra
        # phần còn lại.  Quan trọng: vỉ thứ hai/ng thứ n bị sai màu hoặc thiếu
        # slot cũng phải được trả về để tầng giám sát bật cảnh báo, thay vì
        # chỉ giữ những candidate PASS như cơ chế đếm cũ.
        inspections = [primary_inspection]
        if self.check_for_multiple_candidates:
            self._collect_additional_inspections(
                frame_bgr,
                normalized_profile,
                inspections,
                allow_guided=allow_profile_color_fallback,
            )

        if len(inspections) > 1 and not self.allow_multiple_candidates:
            detail = "Phát hiện nhiều candidate vỉ trong cùng frame; cần vùng ROI tách riêng"
            forced = self._unknown(normalized_profile, detail, observations=primary_inspection.validation.observations)
            for inspection in inspections:
                inspection.validation = self._unknown(
                    normalized_profile,
                    detail,
                    observations=inspection.validation.observations,
                )
            for inspection in inspections:
                self._draw_inspection(canvas, inspection)
            LOGGER.warning("Frame có nhiều candidate; chuyển trạng thái sang UNKNOWN")
            return FrameInspectionResult(
                selected_product_id,
                forced,
                primary_inspection.bounding_box,
                primary_inspection.product_crop,
                inspections,
                canvas,
                detail,
            )

        # Bất kỳ vỉ FAIL nào trong ROI phải chi phối trạng thái của frame. Nó
        # giúp realtime worker có thể bật báo động ngay cả khi vỉ đầu đang PASS.
        failed = next((item for item in inspections if item.validation.status is ProductStatus.FAIL), None)
        frame_validation = failed.validation if failed is not None else primary_inspection.validation
        for inspection in inspections:
            self._draw_inspection(canvas, inspection)
        if len(inspections) > 1:
            failures = sum(item.validation.status is ProductStatus.FAIL for item in inspections)
            detail = (
                f"Phát hiện đồng thời {len(inspections)} vỉ trong ROI; "
                f"{failures} vỉ NG."
                if failures
                else f"Phát hiện đồng thời {len(inspections)} vỉ trong ROI; chưa thấy lỗi xác thực."
            )
        else:
            detail = primary_inspection.validation.detail
        LOGGER.debug(
            "Hoàn thành kiểm tra frame: profile=%s status=%s confidence=%.3f",
            normalized_profile.product_id,
            frame_validation.status,
            frame_validation.confidence,
        )
        return FrameInspectionResult(
            selected_product_id,
            frame_validation,
            primary_inspection.bounding_box,
            primary_inspection.product_crop,
            inspections,
            canvas,
            detail,
        )

    def _get_safe_profile(self, selected_product_id: str | None) -> tuple[ProductProfile | None, str]:
        if not selected_product_id:
            return None, "Chưa chọn Product Profile; không tự đoán để tránh PASS nhầm"
        try:
            profile = self.product_manager.get(selected_product_id)
        except Exception:  # profile I/O không được làm dừng camera loop
            LOGGER.exception("Không đọc được Product Profile %s", selected_product_id)
            return None, "Không đọc được Product Profile"
        if profile is None:
            return None, f"Không tìm thấy Product Profile: {selected_product_id}"
        if not profile.enabled:
            return None, f"Product Profile đang bị tắt: {selected_product_id}"
        if len(profile.slots) < 2:
            return None, "Product Profile phải có ít nhất 2 slot"
        return profile, ""

    def _normalize_profile_colors(self, profile: ProductProfile) -> tuple[ProductProfile | None, str]:
        """Chuẩn hoá tên màu tiếng Việt về key mà ColorClassifier trả về."""
        normalized_slots: list[SlotSpec] = []
        for slot in profile.slots:
            expected_color = self._color_normalizer(slot.expected_color)
            if expected_color is None:
                return None, f"Màu trong Product Profile chưa được cấu hình: {slot.expected_color}"
            normalized_slots.append(
                SlotSpec(slot.index, slot.x, slot.y, expected_color, slot.radius)
            )
        normalized_manual_strips: list[ManualStripLayout] = []
        for strip in profile.manual_scan_strips:
            normalized_manual_slots: list[SlotSpec] = []
            for slot in strip.slots:
                expected_color = self._color_normalizer(slot.expected_color)
                if expected_color is None:
                    return None, (
                        "Màu trong bố cục vị trí thủ công chưa được cấu hình: "
                        f"{slot.expected_color}"
                    )
                normalized_manual_slots.append(
                    SlotSpec(slot.index, slot.x, slot.y, expected_color, slot.radius)
                )
            normalized_manual_strips.append(ManualStripLayout(strip.index, normalized_manual_slots))
        return (
            ProductProfile(
                product_id=profile.product_id,
                name=profile.name,
                slots=normalized_slots,
                minimum_confidence=profile.minimum_confidence,
                enabled=profile.enabled,
                manual_scan_strips=normalized_manual_strips,
                manual_scan_roi=profile.manual_scan_roi,
                manual_scan_sample_radius=profile.manual_scan_sample_radius,
            ),
            "",
        )

    @staticmethod
    def _manual_layout_matches_roi(
        profile: ProductProfile,
        roi: tuple[float, float, float, float] | None,
    ) -> bool:
        """Không cho tọa độ manual của ROI cũ bị dùng cho ROI mới."""
        expected = profile.manual_scan_roi
        if expected is None:
            return roi is None
        if roi is None or len(expected) != 4:
            return False
        return max(abs(float(actual) - float(saved)) for actual, saved in zip(roi, expected, strict=True)) <= 0.005

    def _inspect_manual_scan_layout(
        self,
        frame_bgr: np.ndarray,
        profile: ProductProfile,
        canvas: np.ndarray,
    ) -> FrameInspectionResult:
        """Kiểm từng vỉ/từng slot theo tọa độ người dùng đã chốt trong ROI."""
        frame_height, frame_width = frame_bgr.shape[:2]
        scale = min(frame_height, frame_width)
        roi = profile.manual_scan_roi
        roi_scale = (
            min(frame_width * roi[2], frame_height * roi[3])
            if roi is not None and roi[2] > 0.0 and roi[3] > 0.0
            else scale
        )
        manual_sample_radius = profile.manual_scan_sample_radius
        inspections: list[ProductInspection] = []
        for strip in sorted(profile.manual_scan_strips, key=lambda item: item.index):
            ordered_slots = sorted(strip.slots, key=lambda item: item.index)
            if len(ordered_slots) != len(profile.slots):
                validation = self._unknown(
                    profile,
                    f"Vỉ {strip.index} có {len(ordered_slots)} slot đã khóa, "
                    f"cần đúng {len(profile.slots)} slot.",
                )
                inspections.append(
                    ProductInspection(
                        profile.product_id,
                        (0, 0, 0, 0),
                        validation,
                        np.empty((0, 0, 3), dtype=np.uint8),
                        [],
                        0.0,
                        "Bố cục vị trí thủ công không hợp lệ",
                    )
                )
                continue
            detected_slots = [
                DetectedSlot(
                    x=round(slot.x * frame_width),
                    y=round(slot.y * frame_height),
                    radius=max(5, round(slot.radius * scale)),
                    sample_radius=(
                        max(4, round(manual_sample_radius * roi_scale))
                        if manual_sample_radius is not None
                        else max(4, round(slot.radius * scale * 0.62))
                    ),
                )
                for slot in ordered_slots
            ]
            # Mỗi strip có thể được người vận hành chốt chuỗi màu riêng.
            # Profile tạm chỉ thay mô tả slot; mã, ngưỡng và trạng thái vẫn
            # chính là Product Profile đã chọn.
            strip_profile = ProductProfile(
                product_id=profile.product_id,
                name=profile.name,
                slots=ordered_slots,
                minimum_confidence=profile.minimum_confidence,
                enabled=profile.enabled,
            )
            detection = ProductDetection(
                detected_slots,
                0.99,
                f"Vỉ {strip.index}: vị trí {len(detected_slots)} slot được khóa thủ công",
            )
            inspections.append(self._inspect_candidate(frame_bgr, strip_profile, detection))

        if not inspections:
            validation = self._unknown(profile, "Chưa có vỉ nào được khóa vị trí thủ công")
            self._draw_banner(canvas, validation)
            return FrameInspectionResult(
                profile.product_id,
                validation,
                None,
                None,
                [],
                canvas,
                validation.detail,
            )
        for inspection in inspections:
            self._draw_inspection(canvas, inspection)
        failed = next((item for item in inspections if item.validation.status is ProductStatus.FAIL), None)
        unknown = next((item for item in inspections if item.validation.status is ProductStatus.UNKNOWN), None)
        leading = failed or unknown or inspections[0]
        failures = sum(item.validation.status is ProductStatus.FAIL for item in inspections)
        detail = (
            f"Quét {len(inspections)} vỉ theo vị trí khóa thủ công; {failures} vỉ NG."
            if failures else
            f"Quét {len(inspections)} vỉ theo vị trí khóa thủ công; tất cả vỉ đạt."
        )
        return FrameInspectionResult(
            profile.product_id,
            leading.validation,
            leading.bounding_box,
            leading.product_crop,
            inspections,
            canvas,
            detail,
        )

    def _default_color_normalizer(self, value: str) -> str | None:
        needle = self._normalization_key(value)
        for color_key, config in self.color_classifier.profiles.items():
            accepted = [color_key, str(config.get("display_name", color_key)), *config.get("aliases", [])]
            if any(needle == self._normalization_key(str(name)) for name in accepted):
                return color_key
        return None

    @staticmethod
    def _normalization_key(value: str) -> str:
        return " ".join(value.strip().casefold().replace("_", " ").split())

    def _detect(self, frame_bgr: np.ndarray, expected_slots: int) -> ProductDetection | None:
        try:
            return self.detector.detect(frame_bgr, expected_slots)
        except Exception:
            LOGGER.exception("Detector hình học gặp lỗi")
            return None

    def _detect_partial(self, frame_bgr: np.ndarray, expected_slots: int) -> ProductDetection | None:
        """Gọi capability partial nếu detector hỗ trợ, giữ tương thích plugin cũ."""
        detect_partial = getattr(self.detector, "detect_partial", None)
        if not callable(detect_partial):
            return None
        try:
            return detect_partial(frame_bgr, expected_slots)
        except Exception:
            LOGGER.exception("Detector thiếu-slot gặp lỗi")
            return None

    def _inspect_missing_slot_candidate(
        self,
        frame_bgr: np.ndarray,
        profile: ProductProfile,
        detection: ProductDetection,
    ) -> ProductInspection | None:
        """Tạo kết luận NG khi thấy rõ ``n-1`` slot của profile ``n`` slot.

        Không suy diễn màu để tạo PASS. Candidate chỉ được đưa vào đây khi
        detector đủ-slot đã thất bại và detector partial qua cùng cổng hình
        học; validation luôn có một SlotObservation ``detected=False`` nên
        kết quả là ``FAIL / MISSING_ITEM``.
        """
        expected_count = len(profile.slots)
        if len(detection.slots) != expected_count - 1:
            return None
        box = self._candidate_box(detection.slots, frame_bgr.shape)
        minimum = max(profile.minimum_confidence, self.minimum_detector_confidence)
        if (
            box is None
            or detection.confidence < minimum
            or not self._slots_are_valid(detection.slots, frame_bgr.shape, len(detection.slots))
        ):
            return None
        # Khi candidate đã chạm biên frame/ROI thì không thể phân biệt một
        # viên bị thiếu với một viên còn nằm ngoài vùng nhìn thấy. Không bật
        # cảnh báo NG trên bằng chứng bị cắt mép; chờ vỉ đi trọn vào ROI rồi
        # đánh giá lại. Đây là điều kiện đặc biệt quan trọng với băng tải.
        if self._box_touches_frame_edge(box, frame_bgr.shape):
            return None

        ordered_slots, geometry_missing_position, missing_center, nominal_radius = self._partial_slot_layout(
            detection.slots,
            expected_count,
        )
        missing_position = geometry_missing_position
        if detection.missing_slot_index is not None and 0 <= detection.missing_slot_index < expected_count:
            # ``ProfileColorSequenceDetector.detect_partial`` đã giữ thứ tự
            # theo Product Profile (kể cả khi camera nhìn ngược), vì vậy biết
            # chính xác slot nào bị bỏ. Không dùng khoảng hở hình học để đoán
            # sai một slot cùng màu nằm sát nhau.
            ordered_slots = list(detection.slots)
            missing_position = detection.missing_slot_index
            missing_center, nominal_radius = self._missing_center_for_profile_order(
                ordered_slots,
                missing_position,
            )
        # Bốn tâm hình học chưa đủ để kết luận thiếu viên: một vỉ đạt có thể
        # bị detector bỏ sót đúng một ô vì phản xạ nhựa, hoặc chỉ mới đi một
        # phần vào ROI.  Trước khi báo NG, phải xác nhận vị trí được nội suy
        # thực sự trống màu.  Nếu vị trí vẫn có viên màu hoặc nằm sát biên
        # ảnh, đó là bằng chứng thiếu không đủ mạnh và candidate bị bỏ qua.
        # Nhờ vậy một vỉ PASS không bị sinh thêm "vỉ thiếu" giả ở lượt quét
        # phần frame còn lại sau khi đã mask candidate chính.
        if not self._is_missing_slot_proven_empty(frame_bgr, missing_center, nominal_radius):
            return None
        if self._has_colored_partial_continuation(frame_bgr, ordered_slots, nominal_radius):
            return None
        visible_specs = [
            spec for position, spec in enumerate(profile.slots)
            if position != missing_position
        ]
        partial_profile = ProductProfile(
            profile.product_id,
            profile.name,
            visible_specs,
            profile.minimum_confidence,
            profile.enabled,
        )
        visible_observations, visible_overlays = self._make_slot_overlays(
            frame_bgr,
            partial_profile,
            ordered_slots,
        )
        observations_by_index = {item.index: item for item in visible_observations}
        overlays_by_index = {item.index: item for item in visible_overlays}
        missing_spec = profile.slots[missing_position]
        observations: list[SlotObservation] = []
        overlays: list[SlotOverlay] = []
        for spec in profile.slots:
            if spec.index == missing_spec.index:
                observations.append(SlotObservation(spec.index, False, None, 0.0, 0.0))
                overlays.append(
                    SlotOverlay(spec.index, missing_center, nominal_radius, False, None, 0.0, 0.0)
                )
            else:
                observations.append(observations_by_index[spec.index])
                overlays.append(overlays_by_index[spec.index])

        validation = self.validator.validate(profile, observations, detection.confidence)
        validation.detail = (
            f"{validation.detail}. Phát hiện rõ {len(detection.slots)}/{expected_count} slot; "
            f"vị trí thiếu là ước lượng theo hình học. {detection.detail}"
        )
        return ProductInspection(
            profile.product_id,
            box,
            validation,
            self._crop(frame_bgr, box),
            overlays,
            detection.confidence,
            detection.detail,
            "partial",
        )

    @staticmethod
    def _missing_center_for_profile_order(
        slots: list[DetectedSlot],
        missing_position: int,
    ) -> tuple[tuple[int, int], int]:
        """Nội suy tâm slot thiếu khi thứ tự profile đã được detector giữ."""
        points = np.asarray([(slot.x, slot.y) for slot in slots], dtype=float)
        nominal_radius = max(4, int(round(np.median([slot.radius for slot in slots]))))
        if len(points) < 2:
            return (0, 0), nominal_radius
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        spacing = max(8.0, float(np.median(distances)))
        if missing_position <= 0:
            direction = points[1] - points[0]
            length = max(float(np.linalg.norm(direction)), 1.0)
            missing = points[0] - direction / length * spacing
        elif missing_position >= len(points):
            direction = points[-1] - points[-2]
            length = max(float(np.linalg.norm(direction)), 1.0)
            missing = points[-1] + direction / length * spacing
        else:
            missing = (points[missing_position - 1] + points[missing_position]) / 2.0
        return (int(round(missing[0])), int(round(missing[1]))), nominal_radius

    @staticmethod
    def _partial_slot_layout(
        slots: list[DetectedSlot],
        expected_count: int,
    ) -> tuple[list[DetectedSlot], int, tuple[int, int], int]:
        """Sắp slot theo trục vỉ và chèn vị trí thiếu ước lượng.

        Khoảng trống lớn hơn hẳn các khoảng còn lại cho biết slot thiếu ở
        giữa. Nếu thiếu ở hai đầu không thể phân biệt chỉ từ geometry, ta chỉ
        cần đánh NG (vị trí hiển thị là ước lượng ở cuối chuỗi).
        """
        centers = np.asarray([(slot.x, slot.y) for slot in slots], dtype=float)
        center = centers.mean(axis=0)
        _, _, vectors = np.linalg.svd(centers - center, full_matrices=False)
        axis = vectors[0]
        dominant_axis = 0 if abs(axis[0]) >= abs(axis[1]) else 1
        if axis[dominant_axis] < 0:
            axis = -axis
        projection = (centers - center) @ axis
        order = np.argsort(projection)
        ordered = [slots[int(index)] for index in order]
        ordered_projection = projection[order]
        spacing = np.diff(ordered_projection)
        nominal_spacing = float(np.median(spacing)) if len(spacing) else max(8.0, float(np.mean([slot.radius for slot in slots]) * 2.2))
        nominal_spacing = max(8.0, nominal_spacing)
        missing_position = expected_count - 1
        if len(spacing) >= 2:
            largest_index = int(np.argmax(spacing))
            typical = float(np.median(np.delete(spacing, largest_index))) if len(spacing) > 2 else float(np.min(spacing))
            if typical > 0 and spacing[largest_index] >= typical * 1.55:
                missing_position = largest_index + 1

        if missing_position == 0:
            missing = np.asarray((ordered[0].x, ordered[0].y), dtype=float) - axis * nominal_spacing
        elif missing_position >= len(ordered):
            missing = np.asarray((ordered[-1].x, ordered[-1].y), dtype=float) + axis * nominal_spacing
        else:
            before = np.asarray((ordered[missing_position - 1].x, ordered[missing_position - 1].y), dtype=float)
            after = np.asarray((ordered[missing_position].x, ordered[missing_position].y), dtype=float)
            missing = (before + after) / 2.0
        nominal_radius = max(4, int(round(np.median([slot.radius for slot in slots]))))
        return (
            ordered,
            missing_position,
            (int(round(missing[0])), int(round(missing[1]))),
            nominal_radius,
        )

    def _is_missing_slot_proven_empty(
        self,
        frame_bgr: np.ndarray,
        center: tuple[int, int],
        nominal_radius: int,
    ) -> bool:
        """Kiểm tra ô được nội suy có thật sự trống trước khi kết luận NG.

        ``detect_partial`` chỉ quan sát được ``n - 1`` tâm. Không có cổng này
        nó sẽ biến một slot bị bỏ sót bởi Hough/contour thành ``MISSING_ITEM``.
        Dùng vùng lấy mẫu nhỏ ở tâm ô (giống vùng phân loại màu) để không bị
        vách nhựa trong suốt quanh ô đánh lừa. Ô sát mép frame không thể chứng
        minh là trống vì viên có thể nằm ngoài ảnh, nên luôn để UNKNOWN.
        """
        colored = self._sampled_slot_has_colored_item(frame_bgr, center, nominal_radius)
        return colored is False

    def _has_colored_partial_continuation(
        self,
        frame_bgr: np.ndarray,
        ordered_slots: list[DetectedSlot],
        nominal_radius: int,
    ) -> bool:
        """Phát hiện ô màu bị bỏ sót ở hai đầu chuỗi partial.

        Một detector có thể lấy nhầm 4/5 slot liên tiếp của vỉ đạt và suy ra
        slot thứ năm ở *đầu còn lại*.  Vị trí suy ra trống nhưng đầu đối diện
        thực tế vẫn có viên màu, nên đó không phải vỉ thiếu.  Dò hai điểm tiếp
        tuyến theo trục dãy sẽ chặn chính xác false-NG này mà vẫn giữ được lỗi
        thiếu thật ở đầu/cuối (khi cả hai vùng ngoài đều không có viên).
        """
        if len(ordered_slots) < 2:
            return False
        centers = np.asarray([(slot.x, slot.y) for slot in ordered_slots], dtype=float)
        center = centers.mean(axis=0)
        try:
            _, _, vectors = np.linalg.svd(centers - center, full_matrices=False)
        except np.linalg.LinAlgError:
            return False
        axis = vectors[0]
        dominant_axis = 0 if abs(axis[0]) >= abs(axis[1]) else 1
        if axis[dominant_axis] < 0:
            axis = -axis
        projection = (centers - center) @ axis
        order = np.argsort(projection)
        sorted_centers = centers[order]
        spacing = np.diff(projection[order])
        nominal_spacing = float(np.median(spacing)) if len(spacing) else 0.0
        if nominal_spacing < 4.0:
            return False
        endpoints = (
            sorted_centers[0] - axis * nominal_spacing,
            sorted_centers[-1] + axis * nominal_spacing,
        )
        for point in endpoints:
            coordinate = (int(round(point[0])), int(round(point[1])))
            if self._sampled_slot_has_colored_item(frame_bgr, coordinate, nominal_radius) is True:
                return True
        return False

    @staticmethod
    def _sampled_slot_has_colored_item(
        frame_bgr: np.ndarray,
        center: tuple[int, int],
        nominal_radius: int,
    ) -> bool | None:
        """Trả ``None`` khi tâm sample sát biên frame, nếu không là có màu."""
        height, width = frame_bgr.shape[:2]
        x, y = center
        radius = max(4, int(round(nominal_radius * 0.45)))
        if x - radius < 0 or y - radius < 0 or x + radius >= width or y + radius >= height:
            return None
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        pixels = frame_bgr[mask > 0]
        return FrameInspectionPipeline._has_colored_item(pixels)

    def _guided_inspection_in_box(
        self,
        frame_bgr: np.ndarray,
        profile: ProductProfile,
        box: BoundingBox | None,
    ) -> tuple[ProductDetection, ProductInspection] | None:
        """Xác nhận candidate bằng đúng chuỗi màu trong vùng của candidate.

        Không chạy trên toàn frame: trên băng tải, vỉ kế tiếp có thể đã đi vào
        khung và làm fallback toàn cục ghép màu của hai vỉ khác nhau.
        """
        if box is None:
            return None
        x, y, width, height = box
        crop = frame_bgr[y:y + height, x:x + width]
        if crop.size == 0:
            return None
        guided = self.profile_color_detector.detect(crop, profile)
        if not guided.slots:
            return None
        translated = ProductDetection(
            [replace(slot, x=slot.x + x, y=slot.y + y) for slot in guided.slots],
            guided.confidence,
            guided.detail,
        )
        inspection = self._inspect_candidate(frame_bgr, profile, translated)
        if inspection.validation.status is not ProductStatus.PASS:
            return None
        return translated, inspection

    def _detect_profile_colored_row(
        self,
        frame_bgr: np.ndarray,
        profile: ProductProfile,
    ) -> ProductDetection | None:
        """Tìm một hàng slot nhỏ bằng bằng chứng màu của profile đã chọn.

        Đây là fallback có chủ đích cho camera: lúc ROI bao gồm cả bàn/máy
        tính, viền nhựa trong suốt đôi khi làm Hough bỏ sót một slot.  Ta không
        dùng fallback này để *đoán* loại sản phẩm: các component phải có đúng
        thứ tự màu (hoặc thứ tự đảo) trong profile, sau đó vẫn được
        :class:`ProductValidator` kiểm tra lại ở caller.
        """
        analysis_frame, inverse_scale = self._color_guided_analysis_frame(frame_bgr)
        candidates = self._profile_color_components(analysis_frame, profile)
        count = len(profile.slots)
        if len(candidates) < count:
            return None

        expected = tuple(slot.expected_color for slot in profile.slots)
        reversed_expected = tuple(reversed(expected))
        best: tuple[float, np.ndarray] | None = None
        # Mỗi màu chỉ giữ tối đa vài component lớn nhất, nên tổ hợp này nhỏ
        # trong realtime nhưng vẫn chịu được nền màu/đèn phản chiếu.
        for indexes in combinations(range(len(candidates)), count):
            selected = np.asarray(
                [[candidates[index][0], candidates[index][1], candidates[index][2], index] for index in indexes],
                dtype=float,
            )
            try:
                fit = self.detector._best_perspective_row(selected, count)
            except (AttributeError, ValueError, np.linalg.LinAlgError):
                # Detector custom của khách hàng không nhất thiết expose helper
                # nội bộ; fallback này đơn giản bị bỏ qua, camera vẫn chạy.
                return None
            if fit is None:
                continue
            labels = tuple(candidates[int(item[3])][3] for item in fit.selected)
            if labels not in {expected, reversed_expected}:
                continue
            if (
                fit.score > 0.72
                or fit.projective_error > 0.13
                or fit.line_error > 1.25
                or fit.minimum_spacing_ratio < 0.55
            ):
                continue
            radii = fit.selected[:, 2]
            radius_cv = float(np.std(radii) / max(float(np.mean(radii)), 1.0))
            # Một "dãy" gồm bảng màu/nhãn nền thường có component to nhỏ lẫn
            # lộn. Các viên cùng loại có thể méo phối cảnh, nhưng phần màu
            # bên trong vẫn không được chênh quá mạnh như vậy.
            if radius_cv > 0.55:
                continue
            # Ưu tiên dòng thẳng/đều và kích thước component tương đồng;
            # diện tích chỉ là tie-break để không chọn bụi màu rất nhỏ.
            area_bonus = sum(candidates[int(item[3])][4] for item in fit.selected) / 1_000_000.0
            score = fit.score + radius_cv * 0.18 - area_bonus
            if best is None or score < best[0]:
                best = (score, fit.selected)

        if best is None:
            return None

        selected = best[1]
        labels = tuple(candidates[int(item[3])][3] for item in selected)
        if labels == reversed_expected and labels != expected:
            # Camera có thể nhìn vỉ từ đầu ngược lại. Chuẩn hoá slot trả về
            # theo thứ tự Product Profile để validator không báo sai màu giả.
            selected = selected[::-1]

        slots: list[DetectedSlot] = []
        for item in selected:
            source = candidates[int(item[3])]
            color_radius = max(4, int(round(source[2] * inverse_scale)))
            sample_radius = max(4, int(round(color_radius * 0.82)))
            slots.append(
                DetectedSlot(
                    int(round(source[0] * inverse_scale)),
                    int(round(source[1] * inverse_scale)),
                    max(sample_radius + 4, int(round(color_radius * 1.75))),
                    sample_radius=sample_radius,
                    side_view=True,
                    matched_color=source[3],
                )
            )
        confidence = max(0.85, min(0.91, 0.93 - best[0] * 0.08))
        return ProductDetection(
            slots,
            confidence,
            "Đã nhận diện dãy slot nhỏ bằng màu + hình học của Product Profile",
        )

    def _color_guided_analysis_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """Thu gọn riêng bước tìm component màu, giữ tọa độ ảnh gốc ở output."""
        largest = max(frame_bgr.shape[:2])
        if largest <= self.color_guided_max_dimension:
            return frame_bgr, 1.0
        scale = self.color_guided_max_dimension / largest
        resized = cv2.resize(
            frame_bgr,
            (max(1, round(frame_bgr.shape[1] * scale)), max(1, round(frame_bgr.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, 1.0 / scale

    def _profile_color_components(
        self,
        frame_bgr: np.ndarray,
        profile: ProductProfile,
    ) -> list[tuple[float, float, float, str, int]]:
        """Trả component màu ``(x, y, bán_kính, màu, diện_tích)`` có thể là slot."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        height, width = hsv.shape[:2]
        frame_area = height * width
        min_area = max(12, int(round(frame_area * 0.00003)))
        max_area = max(min_area + 1, int(round(frame_area * 0.020)))
        required_by_color: dict[str, int] = {}
        for slot in profile.slots:
            required_by_color[slot.expected_color] = required_by_color.get(slot.expected_color, 0) + 1

        components: list[tuple[float, float, float, str, int]] = []
        for color, required_count in required_by_color.items():
            raw = self.color_classifier.profiles.get(color)
            mask = self._color_mask(hsv, raw)
            if mask is None:
                continue
            count, _, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
            color_components: list[tuple[float, float, float, str, int]] = []
            for index in range(1, count):
                x, y, component_width, component_height, area = (int(value) for value in stats[index])
                if not min_area <= area <= max_area:
                    continue
                if component_width < 3 or component_height < 3:
                    continue
                center_x, center_y = (float(value) for value in centers[index])
                radius = float(np.sqrt(area / np.pi))
                color_components.append((center_x, center_y, radius, color, area))
            # Same-colour fragments from an icon/specular edge can appear near
            # one another.  Keep the largest one per local centre.
            color_components.sort(key=lambda item: item[4], reverse=True)
            kept: list[tuple[float, float, float, str, int]] = []
            for component in color_components:
                if any(
                    np.hypot(component[0] - present[0], component[1] - present[1])
                    < max(5.0, min(component[2], present[2]) * 0.90)
                    for present in kept
                ):
                    continue
                kept.append(component)
                # Sáu candidate/màu vẫn dư cho 5 slot, nhưng giảm mạnh tổ
                # hợp sai giữa nhiều vỉ trong ROI.  Sau mỗi vỉ tìm thấy vùng
                # đó bị mask nên các vỉ nhỏ hơn sẽ được lên lượt kế tiếp.
                if len(kept) >= max(required_count, self.profile_component_limit_per_color):
                    break
            components.extend(kept)
        return components

    @staticmethod
    def _color_mask(hsv: np.ndarray, raw_profile: object) -> np.ndarray | None:
        if not isinstance(raw_profile, Mapping):
            return None
        try:
            lower = np.asarray(raw_profile["hsv_min"], dtype=np.uint8)
            upper = np.asarray(raw_profile["hsv_max"], dtype=np.uint8)
        except (KeyError, TypeError, ValueError):
            return None
        if lower.shape != (3,) or upper.shape != (3,) or np.any(lower > upper):
            return None
        mask = cv2.inRange(hsv, lower, upper)
        has_extension = "hsv_extension_min" in raw_profile or "hsv_extension_max" in raw_profile
        if has_extension:
            try:
                extension_lower = np.asarray(raw_profile["hsv_extension_min"], dtype=np.uint8)
                extension_upper = np.asarray(raw_profile["hsv_extension_max"], dtype=np.uint8)
            except (KeyError, TypeError, ValueError):
                return None
            if (
                extension_lower.shape != (3,)
                or extension_upper.shape != (3,)
                or np.any(extension_lower > extension_upper)
            ):
                return None
            mask |= cv2.inRange(hsv, extension_lower, extension_upper)
        # Các pixel nhựa gần trắng có thể vô tình nằm sát dải hue; chỉ giữ
        # phần đủ bão hoà để component phản ánh viên màu bên trong.
        mask[(hsv[:, :, 1] < 50) | (hsv[:, :, 2] < 35)] = 0
        return mask

    def _find_secondary_candidate(
        self,
        frame_bgr: np.ndarray,
        primary_box: BoundingBox | list[BoundingBox],
        expected_slots: int,
    ) -> ProductDetection | None:
        if not self.check_for_multiple_candidates:
            return None
        boxes = primary_box if isinstance(primary_box, list) else [primary_box]
        candidate = self._detect(self._mask_candidate_boxes(frame_bgr, boxes), expected_slots)
        if candidate is None or not candidate.slots:
            return None
        if candidate.confidence < self.minimum_detector_confidence:
            return None
        if self._candidate_box(candidate.slots, frame_bgr.shape) is None:
            return None
        return candidate

    def _find_secondary_partial_candidate(
        self,
        frame_bgr: np.ndarray,
        boxes: list[BoundingBox],
        expected_slots: int,
    ) -> ProductDetection | None:
        """Tìm một vỉ thiếu slot ở phần ROI không thuộc vỉ đã nhận ra."""
        candidate = self._detect_partial(self._mask_candidate_boxes(frame_bgr, boxes), expected_slots)
        if candidate is None or not candidate.slots:
            return None
        if self._candidate_box(candidate.slots, frame_bgr.shape) is None:
            return None
        return candidate

    def _find_secondary_profile_color_candidate(
        self,
        frame_bgr: np.ndarray,
        boxes: list[BoundingBox],
        profile: ProductProfile,
    ) -> ProductInspection | None:
        """Tìm vỉ PASS tiếp theo bằng chuỗi màu trong phần ROI chưa dùng.

        Vỉ nhựa trong suốt đặt nghiêng thường không đủ vòng tròn Hough để
        ``_find_secondary_candidate`` trả về một candidate hình học. Trước đây
        fallback màu chỉ dùng cho vỉ đầu tiên, nên ROI có bốn vỉ có thể chỉ
        hiện một vỉ PASS. Mask các vỉ đã chốt trước khi fallback giúp mỗi lượt
        chỉ lấy một dãy màu còn lại, sau đó vẫn kiểm tra màu độc lập ở ảnh gốc.
        """
        masked = self._mask_candidate_boxes(frame_bgr, boxes)
        for detection in (
            self.profile_color_detector.detect(masked, profile),
            self._detect_profile_colored_row(masked, profile),
        ):
            if detection is None or not detection.slots:
                continue
            inspection = self._inspect_candidate(frame_bgr, profile, detection)
            if inspection.validation.status is ProductStatus.PASS:
                return inspection
        return None

    def _find_secondary_profile_color_partial_candidate(
        self,
        frame_bgr: np.ndarray,
        boxes: list[BoundingBox],
        profile: ProductProfile,
    ) -> ProductInspection | None:
        """Tìm vỉ thiếu một viên bằng chuỗi màu trong phần ROI còn lại."""
        masked = self._mask_candidate_boxes(frame_bgr, boxes)
        detect_partial = getattr(self.profile_color_detector, "detect_partial", None)
        if not callable(detect_partial):
            # Giữ tương thích với plugin/test-double detector màu cũ.
            return None
        detection = detect_partial(masked, profile)
        if not detection.slots:
            return None
        return self._inspect_missing_slot_candidate(frame_bgr, profile, detection)

    @staticmethod
    def _mask_candidate_boxes(frame_bgr: np.ndarray, boxes: list[BoundingBox]) -> np.ndarray:
        """Che các vỉ đã dùng bằng màu nền trung tính trước lượt detector kế tiếp."""
        masked = frame_bgr.copy()
        fill = tuple(int(value) for value in np.median(frame_bgr.reshape(-1, 3), axis=0))
        for x, y, width, height in boxes:
            padding = max(8, round(min(width, height) * 0.10))
            x0 = max(0, x - padding)
            y0 = max(0, y - padding)
            x1 = min(masked.shape[1], x + width + padding)
            y1 = min(masked.shape[0], y + height + padding)
            cv2.rectangle(masked, (x0, y0), (x1, y1), fill, -1)
        return masked

    def _collect_additional_inspections(
        self,
        frame_bgr: np.ndarray,
        profile: ProductProfile,
        inspections: list[ProductInspection],
        *,
        allow_guided: bool,
    ) -> None:
        """Bổ sung các vỉ còn lại trong ROI, kể cả vỉ NG thiếu/sai slot.

        Không đưa candidate UNKNOWN đơn thuần vào danh sách vì các phản xạ
        nhựa/nền có thể sinh hình học yếu.  FAIL là bằng chứng sản phẩm lỗi
        hữu ích cho báo động; PASS là vỉ hợp lệ đang hiện diện.
        """
        expected_slots = len(profile.slots)
        while len(inspections) < self.maximum_monitor_candidates:
            boxes = [item.bounding_box for item in inspections]
            detection = self._find_secondary_candidate(frame_bgr, boxes, expected_slots)
            candidate: ProductInspection | None = None
            if detection is not None and detection.slots:
                candidate = self._inspect_candidate(frame_bgr, profile, detection)
                if allow_guided and candidate.validation.status is not ProductStatus.PASS:
                    local = self._guided_inspection_in_box(frame_bgr, profile, candidate.bounding_box)
                    if local is not None:
                        _detection, candidate = local

            # Khi detector hình học không thấy vỉ tiếp theo (hoặc chỉ sinh
            # candidate UNKNOWN), thử fallback màu đầy đủ *trước* detector
            # n-1 slot. Trong ROI có nhiều vỉ đạt, nhánh n-1 trước đây lại
            # chạy Hough thêm một lượt cho từng vỉ và làm FPS tụt mạnh dù
            # không đóng góp kết luận nào. Không thay candidate FAIL bằng
            # PASS khác: một NG có bằng chứng rõ phải được giữ nguyên.
            if allow_guided and (candidate is None or candidate.validation.status is ProductStatus.UNKNOWN):
                color_candidate = self._find_secondary_profile_color_candidate(frame_bgr, boxes, profile)
                if color_candidate is not None:
                    candidate = color_candidate

            # Chỉ khi cả Hough đầy đủ lẫn chuỗi màu 5 slot đều không có kết
            # quả, mới chạy detector 4/5 slot để tìm vỉ thiếu. Đây là nhánh
            # lỗi hiếm, nên tách nó khỏi đường PASS nhanh ở trên.
            if candidate is None:
                partial = self._find_secondary_partial_candidate(frame_bgr, boxes, expected_slots)
                if partial is not None and partial.slots:
                    candidate = self._inspect_missing_slot_candidate(frame_bgr, profile, partial)

            # Chuỗi màu đủ n slot mới được phép PASS. Nếu fallback đầy đủ đã
            # từ chối vì slot bị tách giả, thử riêng n-1 slot để có thể báo
            # NG thiếu viên thay vì bỏ qua vỉ lỗi hoặc PASS nhầm.
            if allow_guided and candidate is None:
                partial_color_candidate = self._find_secondary_profile_color_partial_candidate(
                    frame_bgr,
                    boxes,
                    profile,
                )
                if partial_color_candidate is not None:
                    candidate = partial_color_candidate

            if candidate is None:
                break
            if not self._boxes_are_distinct_from_all(candidate.bounding_box, boxes):
                # Tương thích chế độ cũ: chỉ cần detector báo thêm candidate
                # là frame phải REVIEW, kể cả khi hai bounding box chồng lên.
                if not self.allow_multiple_candidates:
                    inspections.append(candidate)
                break
            if candidate.validation.status is ProductStatus.UNKNOWN:
                # Không đủ bằng chứng để phát cảnh báo hay che thêm vùng;
                # dừng lượt quét để tránh lặp trên cùng phản xạ trong ROI.
                # Ở chế độ legacy một candidate thứ hai vẫn phải khiến frame
                # thành UNKNOWN (fail-safe), nên giữ nó để caller áp chính
                # sách đó phía trên.
                if not self.allow_multiple_candidates:
                    inspections.append(candidate)
                break
            inspections.append(candidate)

    @classmethod
    def _boxes_are_distinct_from_all(cls, candidate: BoundingBox, boxes: list[BoundingBox]) -> bool:
        return all(cls._boxes_are_distinct(box, candidate) for box in boxes)

    def _inspect_candidate(
        self,
        frame_bgr: np.ndarray,
        profile: ProductProfile,
        detection: ProductDetection,
    ) -> ProductInspection:
        box = self._candidate_box(detection.slots, frame_bgr.shape)
        if box is None or not self._slots_are_valid(detection.slots, frame_bgr.shape, len(profile.slots)):
            validation = self._unknown(profile, "Hình học slot không hợp lệ hoặc vượt ra ngoài frame")
            return ProductInspection(
                profile.product_id,
                box or (0, 0, 0, 0),
                validation,
                np.empty((0, 0, 3), dtype=np.uint8),
                [],
                detection.confidence,
                detection.detail,
            )

        crop = self._crop(frame_bgr, box)
        minimum = max(profile.minimum_confidence, self.minimum_detector_confidence)
        if detection.confidence < minimum:
            validation = self._unknown(
                profile,
                f"Độ tin cậy detector ({detection.confidence:.0%}) chưa đạt ngưỡng ({minimum:.0%})",
                confidence=detection.confidence,
            )
            return ProductInspection(
                profile.product_id,
                box,
                validation,
                crop,
                self._make_slot_overlays(frame_bgr, profile, detection.slots)[1],
                detection.confidence,
                detection.detail,
            )

        if self.strict_position_color_validation and not self._matches_fixed_profile_layout(
            detection.slots,
            profile,
        ):
            validation = self._unknown(
                profile,
                "Hình học candidate không khớp mẫu vị trí slot đã khóa theo góc camera",
                confidence=detection.confidence,
            )
            return ProductInspection(
                profile.product_id,
                box,
                validation,
                crop,
                self._make_slot_overlays(frame_bgr, profile, detection.slots)[1],
                detection.confidence,
                detection.detail,
            )

        forward_observations, forward_slots = self._make_slot_overlays(frame_bgr, profile, detection.slots)
        forward = self.validator.validate(profile, forward_observations, detection.confidence)

        # A product may be photographed after a 180 degree rotation or with the
        # camera on the opposite side of the conveyor.  Check both unambiguous
        # directions; never choose PASS if either direction lacks evidence.
        # Với profile đối xứng như Tím–Tím–Xanh dương–Tím–Tím, thứ tự đảo
        # hoàn toàn giống thứ tự xuôi. Không chạy phân loại màu lần hai giúp
        # camera xử lý nhanh hơn đáng kể mà không làm thay đổi kết luận.
        if profile.expected_colors == list(reversed(profile.expected_colors)):
            validation, overlays, order = forward, forward_slots, "symmetric"
        else:
            reversed_slots = list(reversed(detection.slots))
            reverse_observations, reverse_overlays = self._make_slot_overlays(frame_bgr, profile, reversed_slots)
            reverse = self.validator.validate(profile, reverse_observations, detection.confidence)
            validation, overlays, order = self._choose_orientation(forward, forward_slots, reverse, reverse_overlays)
        validation.detail = f"{validation.detail}. {detection.detail}"
        return ProductInspection(
            profile.product_id,
            box,
            validation,
            crop,
            overlays,
            detection.confidence,
            detection.detail,
            order,
        )

    @staticmethod
    def _matches_fixed_profile_layout(slots: list[DetectedSlot], profile: ProductProfile) -> bool:
        """Đối chiếu bố cục tương đối với mẫu Product Profile đã khóa.

        Không khóa tọa độ tuyệt đối trên ảnh vì một ROI có thể chứa nhiều vỉ.
        Hệ thống khóa trục và các khoảng cách chuẩn hoá của mỗi vỉ, nhờ vậy
        một dãy màu lẫn từ hai vỉ không thể vượt qua chỉ vì trùng màu.
        """
        if len(slots) != len(profile.slots) or len(slots) < 3:
            return False
        observed = np.asarray([(slot.x, slot.y) for slot in slots], dtype=float)
        expected = np.asarray([(spec.x, spec.y) for spec in profile.slots], dtype=float)
        try:
            observed_axis = np.linalg.svd(observed - observed.mean(axis=0), full_matrices=False)[2][0]
            expected_axis = np.linalg.svd(expected - expected.mean(axis=0), full_matrices=False)[2][0]
        except np.linalg.LinAlgError:
            return False
        observed_projection = np.sort((observed - observed.mean(axis=0)) @ observed_axis)
        expected_projection = np.sort((expected - expected.mean(axis=0)) @ expected_axis)
        observed_spacing = np.diff(observed_projection)
        expected_spacing = np.diff(expected_projection)
        if np.any(observed_spacing < 2.0) or np.any(expected_spacing <= 0.0):
            return False
        # ``expected`` dùng tọa độ chuẩn hóa 0..1 nên tổng khoảng cách có thể
        # nhỏ hơn 1. Không được chặn mẫu số tại 1.0, nếu không một bố cục
        # manual đúng hoàn toàn cũng bị co sai tỷ lệ và rơi vào UNKNOWN.
        observed_relative = observed_spacing / max(float(observed_spacing.sum()), 1e-6)
        expected_relative = expected_spacing / max(float(expected_spacing.sum()), 1e-6)
        # Cùng một camera cố định vẫn có sai số perspective nhẹ giữa các làn;
        # ngưỡng này giữ được vỉ thật nghiêng nhưng chặn slot bị tách giả.
        return float(np.max(np.abs(observed_relative - expected_relative))) <= 0.16

    def _make_slot_overlays(
        self,
        frame_bgr: np.ndarray,
        profile: ProductProfile,
        detected_slots: list[DetectedSlot],
    ) -> tuple[list[SlotObservation], list[SlotOverlay]]:
        samples: list[np.ndarray] = []
        radii: list[int] = []
        height, width = frame_bgr.shape[:2]
        for slot in detected_slots:
            mask = np.zeros((height, width), dtype=np.uint8)
            if slot.sample_radius is not None:
                radius = max(4, slot.sample_radius)
                cv2.circle(mask, (slot.x, slot.y), radius, 255, -1)
            elif slot.radius_x is not None and slot.radius_y is not None:
                radius_x = max(4, round(slot.radius_x * 0.45))
                radius_y = max(4, round(slot.radius_y * 0.45))
                radius = max(radius_x, radius_y)
                cv2.ellipse(mask, (slot.x, slot.y), (radius_x, radius_y), slot.angle_deg, 0, 360, 255, -1)
            else:
                radius = max(4, round(slot.radius * 0.45))
                cv2.circle(mask, (slot.x, slot.y), radius, 255, -1)
            samples.append(frame_bgr[mask > 0])
            radii.append(radius)

        classifier_failed = False
        try:
            classify_sequence = getattr(self.color_classifier, "classify_sequence", None)
            if callable(classify_sequence):
                color_results = list(classify_sequence(samples, profile.expected_colors))
            else:
                color_results = [self.color_classifier.classify(sample) for sample in samples]
            if len(color_results) != len(samples):
                raise ValueError("Bộ phân loại màu trả về sai số lượng slot")
        except Exception as error:
            LOGGER.warning("Không phân loại được chuỗi màu: %s", error)
            color_results = [None] * len(samples)
            classifier_failed = True

        observations: list[SlotObservation] = []
        overlays: list[SlotOverlay] = []
        for index, (spec, slot, pixels, radius) in enumerate(zip(profile.slots, detected_slots, samples, radii, strict=True)):
            # Vách nhựa/cell rỗng vẫn có texture nên không thể dùng std để
            # coi là "có viên". Chỉ slot có đủ pixel màu bão hoà mới là
            # detected; cell rõ nhưng không có màu sẽ đi vào MISSING_ITEM.
            # Chế độ cố định vị trí: màu được chứng minh bằng pixel lấy tại
            # đúng tâm slot, không chỉ vì detector màu đã gán nhãn component.
            visible = self._has_colored_item(pixels)
            color_result = color_results[index] if visible and not classifier_failed else None
            color = color_result.name if color_result is not None else None
            color_confidence = color_result.confidence if color_result is not None else 0.0
            # ``matched_color`` chỉ là nhãn của component đã hỗ trợ *tìm vị
            # trí* slot, không phải bằng chứng độc lập về màu. Trước đây nhãn
            # này ghi đè cả khi ColorClassifier nhìn thấy một màu khác. Điều
            # đó có thể biến một viên Tím ở slot Xanh dương thành PASS. Khi
            # hai bộ đọc mâu thuẫn, giữ kết quả phân loại độc lập để validator
            # trả NG; chỉ dùng dải hiệu chỉnh khi classifier chưa xác định
            # được màu nào.
            if color_result is None and visible:
                # Ở chế độ khóa vị trí chỉ dùng dải HSV hiệu chỉnh của *màu
                # kỳ vọng tại chính slot này*. Dải đó vẫn đo từ pixel thật,
                # nên không thể dùng nhãn component để biến màu sai thành PASS.
                contextual_color = (
                    spec.expected_color
                    if self.strict_position_color_validation
                    else slot.matched_color or spec.expected_color
                )
                # Một số webcam nén viên xanh dương đậm về hue giáp Tím.  Dải
                # ``profile_hsv_extension`` là bằng chứng *riêng cho profile*
                # (không dùng để nới ColorClassifier toàn cục): chỉ khi chưa
                # có màu cạnh tranh và phần lớn pixel ở slot khớp dải này mới
                # được hiệu chỉnh. Nhờ đó vỉ tím thật không thể qua profile
                # xanh dương chỉ vì nằm ở vị trí slot 3.
                extension_confidence = self._profile_extension_confidence(pixels, contextual_color)
                if extension_confidence > 0.0:
                    color = contextual_color
                    color_confidence = extension_confidence
            # Lỗi runtime classifier là lỗi hạ tầng, không phải lỗi sản phẩm:
            # hạ confidence để validator giữ REVIEW thay vì báo NG giả.
            slot_confidence = min(0.95, 0.90 if visible and not classifier_failed else 0.0)
            observations.append(
                SlotObservation(spec.index, visible, color, slot_confidence, color_confidence)
            )
            overlays.append(
                SlotOverlay(
                    spec.index,
                    (slot.x, slot.y),
                    radius,
                    visible,
                    color,
                    color_confidence,
                    slot_confidence,
                )
            )
        return observations, overlays

    @staticmethod
    def _has_colored_item(pixels: np.ndarray) -> bool:
        """Phân biệt viên màu với cell nhựa rỗng trong một slot đã xác định."""
        if len(pixels) == 0:
            return False
        try:
            hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        except cv2.error:
            return False
        colored = (hsv[:, 1] >= 35) & (hsv[:, 2] >= 25)
        count = int(np.count_nonzero(colored))
        return count >= max(12, int(np.ceil(len(hsv) * 0.05)))

    def _profile_extension_confidence(self, pixels: np.ndarray, expected_color: str) -> float:
        """Bằng chứng HSV mở rộng, chỉ hợp lệ tại slot đã được profile chỉ định.

        Các dải ``profile_hsv_extension_*`` giải quyết sai lệch hue đã đo trên
        camera thực tế.  Chúng không được nhập vào phân loại màu toàn cục vì
        có thể giao với màu lân cận (ví dụ blue/tím).  Một slot chỉ được sửa
        khi có ít nhất 24 pixel đủ bão hoà và 60% phần màu của slot thuộc dải.
        """
        raw = self.color_classifier.profiles.get(expected_color)
        if not isinstance(raw, Mapping) or not {
            "profile_hsv_extension_min", "profile_hsv_extension_max",
        }.issubset(raw):
            return 0.0
        try:
            lower = np.asarray(raw["profile_hsv_extension_min"], dtype=np.uint8)
            upper = np.asarray(raw["profile_hsv_extension_max"], dtype=np.uint8)
            hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        except (TypeError, ValueError, cv2.error):
            return 0.0
        if lower.shape != (3,) or upper.shape != (3,) or np.any(lower > upper):
            return 0.0
        eligible = (hsv[:, 1] >= 50) & (hsv[:, 2] >= 35)
        eligible_count = int(np.count_nonzero(eligible))
        if eligible_count < 24:
            return 0.0
        matched = cv2.inRange(hsv.reshape(-1, 1, 3), lower, upper).reshape(-1) > 0
        coverage = float(np.count_nonzero(matched & eligible) / eligible_count)
        if coverage < 0.60:
            return 0.0
        return min(0.96, 0.86 + coverage * 0.10)

    def _choose_orientation(
        self,
        forward: ValidationResult,
        forward_overlays: list[SlotOverlay],
        reverse: ValidationResult,
        reverse_overlays: list[SlotOverlay],
    ) -> tuple[ValidationResult, list[SlotOverlay], str]:
        if forward.status is ProductStatus.PASS:
            return forward, forward_overlays, "forward"
        if reverse.status is ProductStatus.PASS:
            reverse.detail = f"{reverse.detail} (chuỗi slot đảo chiều theo góc camera)"
            return reverse, reverse_overlays, "reversed"
        # FAIL do validator tạo ra chỉ khi có bằng chứng dứt khoát: đủ slot
        # nhưng sai màu, hoặc một observation xác nhận thiếu viên. Không được
        # hạ lỗi này xuống UNKNOWN chỉ vì hướng còn lại bị chói/nhòe; nếu làm
        # vậy tracker không thể ghi NG cho vỉ lỗi.
        if forward.status is ProductStatus.FAIL:
            return forward, forward_overlays, "forward"
        if reverse.status is ProductStatus.FAIL:
            reverse.detail = f"{reverse.detail} (chuỗi slot đảo chiều theo góc camera)"
            return reverse, reverse_overlays, "reversed"
        if forward.status is ProductStatus.UNKNOWN or reverse.status is ProductStatus.UNKNOWN:
            unknown = self._unknown(
                None,
                "Không đủ bằng chứng màu để xác định chiều vỉ",
                observations=forward.observations,
                confidence=min(forward.confidence, reverse.confidence),
            )
            unknown.product_id = forward.product_id
            return unknown, forward_overlays, "ambiguous"
        return forward, forward_overlays, "forward"

    @staticmethod
    def _candidate_box(slots: list[DetectedSlot], shape: tuple[int, ...]) -> BoundingBox | None:
        if not slots:
            return None
        height, width = shape[:2]
        radii = [slot.radius for slot in slots]
        if any(radius <= 0 for radius in radii):
            return None
        padding = max(8, round(max(radii) * 1.2))
        left = max(0, floor(min(slot.x - slot.radius for slot in slots) - padding))
        top = max(0, floor(min(slot.y - slot.radius for slot in slots) - padding))
        right = min(width, ceil(max(slot.x + slot.radius for slot in slots) + padding))
        bottom = min(height, ceil(max(slot.y + slot.radius for slot in slots) + padding))
        if right <= left or bottom <= top:
            return None
        return left, top, right - left, bottom - top

    @staticmethod
    def _boxes_are_distinct(first: BoundingBox, second: BoundingBox) -> bool:
        """Loại candidate lặp lại của cùng một vỉ sau khi mask một lần."""
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[0] + first[2], second[0] + second[2])
        bottom = min(first[1] + first[3], second[1] + second[3])
        overlap = max(0, right - left) * max(0, bottom - top)
        smaller_area = max(1, min(first[2] * first[3], second[2] * second[3]))
        return overlap / smaller_area < 0.35

    @staticmethod
    def _box_touches_frame_edge(box: BoundingBox, shape: tuple[int, ...]) -> bool:
        """True nếu candidate bị crop bởi rìa của frame/ROI."""
        x, y, width, height = box
        frame_height, frame_width = shape[:2]
        return x <= 0 or y <= 0 or x + width >= frame_width or y + height >= frame_height

    @staticmethod
    def _slots_are_valid(slots: list[DetectedSlot], shape: tuple[int, ...], expected_count: int) -> bool:
        if len(slots) != expected_count:
            return False
        height, width = shape[:2]
        for slot in slots:
            if slot.radius < 4 or not (0 <= slot.x < width and 0 <= slot.y < height):
                return False
        for index, first in enumerate(slots):
            for second in slots[index + 1:]:
                distance = float(np.hypot(first.x - second.x, first.y - second.y))
                if distance < max(4.0, min(first.radius, second.radius) * 0.4):
                    return False
        return True

    @staticmethod
    def _crop(frame_bgr: np.ndarray, box: BoundingBox) -> np.ndarray:
        x, y, width, height = box
        return frame_bgr[y:y + height, x:x + width].copy()

    @staticmethod
    def _is_valid_bgr_frame(frame_bgr: object) -> bool:
        return (
            isinstance(frame_bgr, np.ndarray)
            and frame_bgr.ndim == 3
            and frame_bgr.shape[2] == 3
            and frame_bgr.shape[0] > 0
            and frame_bgr.shape[1] > 0
            and frame_bgr.dtype == np.uint8
        )

    @staticmethod
    def _unknown(
        profile: ProductProfile | None,
        detail: str,
        *,
        observations: list[SlotObservation] | None = None,
        confidence: float = 0.0,
    ) -> ValidationResult:
        return ValidationResult(
            profile.product_id if profile is not None else None,
            ProductStatus.UNKNOWN,
            FailureReason.UNKNOWN,
            observations or [],
            confidence,
            detail,
        )

    @staticmethod
    def _draw_banner(canvas: np.ndarray, validation: ValidationResult) -> None:
        # cv2.putText dùng font Hershey không có glyph tiếng Việt.  Nội dung
        # giải thích đầy đủ vẫn hiển thị Unicode ở panel UI bên dưới; trên ảnh
        # chỉ ghi nhãn ASCII ngắn để không tạo chữ lỗi như "Kh?ng x?c...".
        label = f"{validation.status.value} - REVIEW"
        cv2.putText(canvas, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2, cv2.LINE_AA)

    @staticmethod
    def _draw_inspection(canvas: np.ndarray, inspection: ProductInspection) -> None:
        palette = {
            ProductStatus.PASS: (0, 190, 0),
            ProductStatus.FAIL: (0, 0, 230),
            ProductStatus.UNKNOWN: (0, 215, 255),
        }
        color = palette[inspection.validation.status]
        x, y, width, height = inspection.bounding_box
        cv2.rectangle(canvas, (x, y), (x + width, y + height), color, 2)
        label = (
            f"{inspection.product_id} {inspection.validation.status} "
            f"{inspection.validation.confidence:.0%}"
        )
        cv2.putText(canvas, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
        for slot in inspection.slots:
            slot_color = color if slot.color is not None else (80, 80, 80)
            cv2.circle(canvas, slot.center, slot.radius, slot_color, 2)
            cv2.putText(
                canvas,
                str(slot.index),
                (slot.center[0] - 7, slot.center[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                slot_color,
                1,
                cv2.LINE_AA,
            )
