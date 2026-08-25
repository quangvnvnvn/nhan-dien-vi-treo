"""Kiểm thử hành vi fail-safe của FrameInspectionPipeline."""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from ai.color_classifier import ColorClassifier, ColorResult
from ai.product_detector import DetectedSlot, GeometricProductDetector, ProductDetection
from ai.realtime_pipeline import FrameInspectionPipeline
from ai.validator import ProductValidator
from core.models import ManualStripLayout, ProductProfile, ProductStatus, SlotSpec


class _ProductManager:
    def __init__(self, profile: ProductProfile) -> None:
        self.profile = profile

    def get(self, product_id: str) -> ProductProfile | None:
        return self.profile if product_id == self.profile.product_id else None


class _SequencedDetector:
    """Detector giả có thể phát ra candidate thứ hai khi pipeline mask candidate đầu."""

    def __init__(self, *detections: ProductDetection) -> None:
        self.detections = list(detections)
        self.calls = 0
        self._geometry = GeometricProductDetector()

    def detect(self, image: np.ndarray, expected_slots: int) -> ProductDetection:
        del image, expected_slots
        self.calls += 1
        if self.detections:
            return self.detections.pop(0)
        return ProductDetection([], 0.0, "Không còn candidate")

    def _best_perspective_row(self, candidates: np.ndarray, count: int) -> object:
        return self._geometry._best_perspective_row(candidates, count)


class _BrokenClassifier:
    def __init__(self, profiles: dict[str, object]) -> None:
        self.profiles = profiles

    def classify(self, pixels: np.ndarray) -> object:
        del pixels
        raise RuntimeError("cấu hình màu lỗi")


class _PartialOnlyDetector:
    """Giả lập detector thấy chắc n-1 slot sau khi detector đủ-slot thất bại."""

    def __init__(self, slots: list[DetectedSlot]) -> None:
        self.slots = slots

    def detect(self, image: np.ndarray, expected_slots: int) -> ProductDetection:
        del image, expected_slots
        return ProductDetection([], 0.0, "Không đủ slot")

    def detect_partial(self, image: np.ndarray, expected_slots: int) -> ProductDetection:
        del image
        return ProductDetection(self.slots, 0.95, f"Rõ {len(self.slots)}/{expected_slots} slot")


class _BlueReadAsPurpleClassifier:
    """Giả lập webcam làm classifier tổng quát đọc blue thành purple."""

    def __init__(self, delegate: ColorClassifier) -> None:
        self._delegate = delegate
        self.profiles = delegate.profiles

    def classify(self, pixels: np.ndarray) -> ColorResult:
        result = self._delegate.classify(pixels)
        if result.name == "blue":
            return ColorResult("purple", result.confidence)
        return result


class _NoProfileSequenceDetector:
    """Buộc test đi qua fallback component nhỏ thay vì fallback run lớn."""

    def detect(self, image: np.ndarray, profile: ProductProfile) -> ProductDetection:
        del image, profile
        return ProductDetection([], 0.0, "Không có run màu lớn")


class RealtimePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = ProductProfile(
            "VT001",
            "Vỉ kiểm thử",
            [
                SlotSpec(1, 0.2, 0.5, "Tím"),
                SlotSpec(2, 0.5, 0.5, "Xanh dương"),
                SlotSpec(3, 0.8, 0.5, "Tím"),
            ],
            minimum_confidence=0.85,
        )
        # LAB cố ý mở rộng trong test; HSV là điều kiện phân biệt màu.
        classifier = ColorClassifier({
            "purple": {
                "display_name": "Tím",
                "aliases": ["tím"],
                "hsv_min": [140, 50, 20], "hsv_max": [170, 255, 255],
                "lab_min": [0, 0, 0], "lab_max": [255, 255, 255],
            },
            "blue": {
                "display_name": "Xanh dương",
                "aliases": ["xanh dương"],
                "hsv_min": [105, 50, 20], "hsv_max": [130, 255, 255],
                "lab_min": [0, 0, 0], "lab_max": [255, 255, 255],
            },
        })
        self.manager = _ProductManager(self.profile)
        self.classifier = classifier
        self.slots = [DetectedSlot(80, 100, 32), DetectedSlot(160, 100, 32), DetectedSlot(240, 100, 32)]

    def _frame_with_expected_colours(self) -> np.ndarray:
        frame = np.zeros((220, 320, 3), dtype=np.uint8)
        for slot, bgr in zip(self.slots, [(180, 0, 180), (255, 0, 0), (180, 0, 180)], strict=True):
            cv2.circle(frame, (slot.x, slot.y), 15, bgr, -1)
        return frame

    def _pipeline(
        self,
        *detections: ProductDetection,
        classifier: ColorClassifier | None = None,
    ) -> FrameInspectionPipeline:
        return FrameInspectionPipeline(
            self.manager,  # type: ignore[arg-type] - test double has ProductManager protocol.
            classifier or self.classifier,
            detector=_SequencedDetector(*detections),  # type: ignore[arg-type]
            validator=ProductValidator(),
        )

    def test_pass_returns_crop_bbox_and_annotated_frame(self) -> None:
        pipeline = self._pipeline(
            ProductDetection(self.slots, 0.95, "hình học tốt"),
            ProductDetection([], 0.0, "không có candidate thứ hai"),
        )

        result = pipeline.inspect_frame(self._frame_with_expected_colours(), "VT001")

        self.assertEqual(result.validation.status, ProductStatus.PASS)
        self.assertIsNotNone(result.bounding_box)
        self.assertIsNotNone(result.product_crop)
        self.assertEqual(len(result.detections), 1)
        self.assertEqual(result.detections[0].bounding_box, result.bounding_box)
        self.assertEqual(result.annotated_frame.shape, (220, 320, 3))
        self.assertFalse(np.array_equal(result.annotated_frame, self._frame_with_expected_colours()))

    def test_manual_scan_layout_reads_each_user_locked_slot_without_detector(self) -> None:
        """Camera cố định dùng đúng các tâm do người vận hành chốt, không dò Hough."""
        self.profile.manual_scan_strips = [
            ManualStripLayout(
                1,
                [
                    SlotSpec(1, 80 / 320, 100 / 220, "purple", .10),
                    SlotSpec(2, 160 / 320, 100 / 220, "blue", .10),
                    SlotSpec(3, 240 / 320, 100 / 220, "purple", .10),
                ],
            )
        ]
        self.profile.manual_scan_roi = (.1, .2, .7, .6)
        detector = _SequencedDetector(ProductDetection([], 0.0, "không được gọi"))
        pipeline = FrameInspectionPipeline(
            self.manager, self.classifier, detector=detector, validator=ProductValidator(),
            strict_position_color_validation=True,
        )

        result = pipeline.inspect_frame(
            self._frame_with_expected_colours(), "VT001", roi=(.1, .2, .7, .6),
        )

        self.assertEqual(result.validation.status, ProductStatus.PASS)
        self.assertEqual(len(result.detections), 1)
        self.assertEqual(detector.calls, 0)

    def test_manual_scan_layout_uses_profile_sample_radius(self) -> None:
        """Bán kính người vận hành chốt dùng theo cạnh ngắn ROI, không theo slot cũ."""
        self.profile.manual_scan_strips = [
            ManualStripLayout(
                1,
                [
                    SlotSpec(1, 80 / 320, 100 / 220, "purple", .10),
                    SlotSpec(2, 160 / 320, 100 / 220, "blue", .10),
                    SlotSpec(3, 240 / 320, 100 / 220, "purple", .10),
                ],
            )
        ]
        self.profile.manual_scan_roi = (.1, .2, .7, .6)
        self.profile.manual_scan_sample_radius = .05

        result = self._pipeline().inspect_frame(
            self._frame_with_expected_colours(), "VT001", roi=(.1, .2, .7, .6),
        )

        self.assertEqual(result.validation.status, ProductStatus.PASS)
        self.assertEqual([slot.radius for slot in result.detections[0].slots], [7, 7, 7])

    def test_manual_scan_layout_rejects_when_active_roi_changes(self) -> None:
        self.profile.manual_scan_strips = [
            ManualStripLayout(1, [
                SlotSpec(1, .25, .45, "purple", .10),
                SlotSpec(2, .50, .45, "blue", .10),
                SlotSpec(3, .75, .45, "purple", .10),
            ])
        ]
        self.profile.manual_scan_roi = (.1, .2, .7, .6)

        result = self._pipeline().inspect_frame(
            self._frame_with_expected_colours(), "VT001", roi=(.2, .2, .7, .6),
        )

        self.assertEqual(result.validation.status, ProductStatus.UNKNOWN)
        self.assertIn("ROI hiện tại khác", result.detail)

    def test_side_view_uses_inner_sampling_radius(self) -> None:
        """Side-view không được lấy cả vách nhựa khi đọc màu viên."""
        side_slots = [
            DetectedSlot(80, 100, 70, sample_radius=15, side_view=True),
            DetectedSlot(160, 100, 70, sample_radius=15, side_view=True),
            DetectedSlot(240, 100, 70, sample_radius=15, side_view=True),
        ]
        pipeline = self._pipeline(
            ProductDetection(side_slots, 0.90, "Hough side-view"),
            ProductDetection([], 0.0, "không có candidate thứ hai"),
        )

        result = pipeline.inspect_frame(self._frame_with_expected_colours(), "VT001")

        self.assertEqual(result.validation.status, ProductStatus.PASS)
        self.assertEqual([slot.color for slot in result.validation.observations], ["purple", "blue", "purple"])

    def test_profile_blue_extension_resolves_webcam_hue_inside_a_confirmed_slot(self) -> None:
        """Blue hue=121 chỉ được sửa trong slot blue của profile, không nới tím chung."""
        profile = ProductProfile(
            "VT-BLUE-EXT",
            "Blue extension",
            [
                SlotSpec(1, 0.2, 0.5, "purple"),
                SlotSpec(2, 0.5, 0.5, "blue"),
                SlotSpec(3, 0.8, 0.5, "purple"),
            ],
        )
        colors = {
            "purple": {
                "hsv_min": [115, 50, 20], "hsv_max": [170, 255, 255],
                "lab_min": [0, 0, 0], "lab_max": [255, 255, 255],
            },
            "blue": {
                "hsv_min": [90, 50, 20], "hsv_max": [114, 255, 255],
                "profile_hsv_extension_min": [115, 40, 25],
                "profile_hsv_extension_max": [125, 255, 255],
                "lab_min": [0, 0, 0], "lab_max": [255, 255, 255],
            },
        }
        frame = np.zeros((220, 320, 3), dtype=np.uint8)
        purple = tuple(int(value) for value in cv2.cvtColor(
            np.uint8([[[132, 130, 110]]]), cv2.COLOR_HSV2BGR
        )[0, 0])
        blue_near_purple = tuple(int(value) for value in cv2.cvtColor(
            np.uint8([[[121, 145, 95]]]), cv2.COLOR_HSV2BGR
        )[0, 0])
        for slot, bgr in zip(self.slots, [purple, blue_near_purple, purple], strict=True):
            cv2.circle(frame, (slot.x, slot.y), 15, bgr, -1)
        pipeline = FrameInspectionPipeline(
            _ProductManager(profile),  # type: ignore[arg-type] - test double has ProductManager protocol.
            ColorClassifier(colors),
            detector=_SequencedDetector(
                ProductDetection(self.slots, 0.95, "Hough webcam"),
                ProductDetection([], 0.0, "không có candidate thứ hai"),
            ),  # type: ignore[arg-type]
            validator=ProductValidator(),
        )

        result = pipeline.inspect_frame(frame, "VT-BLUE-EXT")

        self.assertEqual(result.validation.status, ProductStatus.PASS)
        self.assertEqual([slot.color for slot in result.validation.observations], ["purple", "blue", "purple"])

    def test_profile_colored_fallback_recovers_small_slots_when_hough_has_none(self) -> None:
        """ROI rộng vẫn có thể cứu vỉ nhỏ khi chuỗi màu profile khớp hoàn toàn."""
        pipeline = FrameInspectionPipeline(
            self.manager,  # type: ignore[arg-type] - test double has ProductManager protocol.
            self.classifier,
            detector=_SequencedDetector(
                ProductDetection([], 0.0, "Hough không thấy slot"),
                ProductDetection([], 0.0, "Không có candidate thứ hai"),
            ),  # type: ignore[arg-type]
            validator=ProductValidator(),
            enable_color_guided_fallback=True,
        )

        result = pipeline.inspect_frame(
            self._frame_with_expected_colours(),
            "VT001",
            allow_profile_color_fallback=True,
        )

        self.assertEqual(result.validation.status, ProductStatus.PASS)
        self.assertEqual([slot.color for slot in result.validation.observations], ["purple", "blue", "purple"])
        self.assertIn("màu + hình học", result.detail)

    def test_small_component_fallback_recovers_when_large_runs_are_split(self) -> None:
        """ROI cao/nền sáng không được làm bỏ sót các viên màu nhỏ, tách rời."""
        pipeline = FrameInspectionPipeline(
            self.manager,  # type: ignore[arg-type] - test double has ProductManager protocol.
            self.classifier,
            detector=_SequencedDetector(
                ProductDetection([], 0.0, "Hough không thấy slot"),
                ProductDetection([], 0.0, "Không có candidate thứ hai"),
            ),  # type: ignore[arg-type]
            validator=ProductValidator(),
            enable_color_guided_fallback=True,
        )
        pipeline.profile_color_detector = _NoProfileSequenceDetector()  # type: ignore[assignment]

        result = pipeline.inspect_frame(
            self._frame_with_expected_colours(),
            "VT001",
            allow_profile_color_fallback=True,
        )

        self.assertEqual(result.validation.status, ProductStatus.PASS)
        self.assertEqual([slot.color for slot in result.validation.observations], ["purple", "blue", "purple"])
        self.assertIn("slot nhỏ", result.detail)

    def test_profile_fallback_never_overrides_a_conflicting_colour_reader(self) -> None:
        """Fallback định vị theo profile không được ép một màu mâu thuẫn thành PASS."""
        pipeline = FrameInspectionPipeline(
            self.manager,  # type: ignore[arg-type] - test double has ProductManager protocol.
            _BlueReadAsPurpleClassifier(self.classifier),
            detector=_SequencedDetector(
                ProductDetection(self.slots, 0.95, "hình học tốt nhưng blue bị lệch hue"),
                ProductDetection([], 0.0, "không có candidate thứ hai"),
            ),  # type: ignore[arg-type]
            validator=ProductValidator(),
            enable_color_guided_fallback=True,
        )

        result = pipeline.inspect_frame(
            self._frame_with_expected_colours(),
            "VT001",
            allow_profile_color_fallback=True,
        )

        self.assertEqual(result.validation.status, ProductStatus.FAIL)
        self.assertEqual(result.validation.reason.value, "WRONG_COLOR")

    def test_profile_component_label_cannot_override_conflicting_colour(self) -> None:
        """Nhãn component theo profile không được biến màu sai thành PASS."""
        wrong_colour_frame = self._frame_with_expected_colours()
        cv2.circle(wrong_colour_frame, (self.slots[1].x, self.slots[1].y), 20, (180, 0, 180), -1)
        labelled_slots = [
            DetectedSlot(slot.x, slot.y, slot.radius, matched_color=("blue" if index == 1 else "purple"))
            for index, slot in enumerate(self.slots)
        ]
        pipeline = self._pipeline(
            ProductDetection(labelled_slots, 0.95, "fallback có nhãn component"),
            ProductDetection([], 0.0, "không có candidate thứ hai"),
        )

        result = pipeline.inspect_frame(wrong_colour_frame, "VT001")

        self.assertEqual(result.validation.status, ProductStatus.FAIL)
        self.assertEqual(result.validation.reason.value, "WRONG_COLOR")
        self.assertEqual(result.validation.observations[1].color, "purple")

    def test_missing_selected_profile_is_unknown(self) -> None:
        pipeline = self._pipeline(ProductDetection(self.slots, 0.95, "không dùng"))

        result = pipeline.inspect_frame(self._frame_with_expected_colours())

        self.assertEqual(result.validation.status, ProductStatus.UNKNOWN)
        self.assertEqual(result.detections, [])
        self.assertIn("Chưa chọn", result.detail)

    def test_low_detector_confidence_cannot_pass(self) -> None:
        pipeline = self._pipeline(
            ProductDetection(self.slots, 0.60, "độ tin cậy thấp"),
            ProductDetection([], 0.0, "không dùng"),
        )

        result = pipeline.inspect_frame(self._frame_with_expected_colours(), "VT001")

        self.assertEqual(result.validation.status, ProductStatus.UNKNOWN)
        self.assertIn("detector", result.detail)

    def test_partial_clear_pack_is_ng_missing_item_not_unknown(self) -> None:
        """Khi chỉ còn n-1 slot rõ ràng, realtime phải đưa NG vào tracker."""
        frame = self._frame_with_expected_colours()
        # Ô thứ ba thật sự trống; detector chỉ thấy hai ô còn lại.
        cv2.circle(frame, (self.slots[2].x, self.slots[2].y), 18, (0, 0, 0), -1)
        pipeline = FrameInspectionPipeline(
            self.manager,  # type: ignore[arg-type] - test double has ProductManager protocol.
            self.classifier,
            detector=_PartialOnlyDetector(self.slots[:2]),  # type: ignore[arg-type]
            validator=ProductValidator(),
        )

        result = pipeline.inspect_frame(frame, "VT001")

        self.assertEqual(result.validation.status, ProductStatus.FAIL)
        self.assertEqual(result.validation.reason.value, "MISSING_ITEM")
        self.assertEqual(len(result.detections), 1)
        self.assertEqual(len(result.validation.observations), 3)
        self.assertTrue(any(not item.detected for item in result.validation.observations))

    def test_partial_candidate_is_not_ng_when_interpolated_slot_still_has_item(self) -> None:
        """Bỏ sót một tâm detector không được biến vỉ đạt thành thiếu viên."""
        pipeline = FrameInspectionPipeline(
            self.manager,  # type: ignore[arg-type] - test double has ProductManager protocol.
            self.classifier,
            detector=_PartialOnlyDetector(self.slots[:2]),  # type: ignore[arg-type]
            validator=ProductValidator(),
        )

        result = pipeline.inspect_frame(self._frame_with_expected_colours(), "VT001")

        self.assertEqual(result.validation.status, ProductStatus.UNKNOWN)
        self.assertEqual(result.detections, [])
        self.assertIn("Không xác thực", result.detail)

    def test_partial_candidate_is_not_ng_when_first_slot_was_overlooked(self) -> None:
        """4/5 slot cuối của vỉ đạt không được suy đoán là thiếu slot cuối."""
        pipeline = FrameInspectionPipeline(
            self.manager,  # type: ignore[arg-type] - test double has ProductManager protocol.
            self.classifier,
            detector=_PartialOnlyDetector(self.slots[1:]),  # type: ignore[arg-type]
            validator=ProductValidator(),
        )

        result = pipeline.inspect_frame(self._frame_with_expected_colours(), "VT001")

        self.assertEqual(result.validation.status, ProductStatus.UNKNOWN)
        self.assertEqual(result.detections, [])

    def test_partial_candidate_touching_frame_edge_is_not_ng(self) -> None:
        """Vỉ mới vào ROI chưa trọn không được báo thiếu viên giả."""
        edge_slots = [DetectedSlot(22, 100, 24), DetectedSlot(100, 100, 24)]
        frame = np.zeros((220, 320, 3), dtype=np.uint8)
        cv2.circle(frame, (22, 100), 14, (180, 0, 180), -1)
        cv2.circle(frame, (100, 100), 14, (255, 0, 0), -1)
        pipeline = FrameInspectionPipeline(
            self.manager,  # type: ignore[arg-type] - test double has ProductManager protocol.
            self.classifier,
            detector=_PartialOnlyDetector(edge_slots),  # type: ignore[arg-type]
            validator=ProductValidator(),
        )

        result = pipeline.inspect_frame(frame, "VT001")

        self.assertEqual(result.validation.status, ProductStatus.UNKNOWN)
        self.assertEqual(result.detections, [])

    def test_clear_empty_cell_is_missing_item_ng(self) -> None:
        frame = self._frame_with_expected_colours()
        cv2.circle(frame, (self.slots[1].x, self.slots[1].y), 20, (0, 0, 0), -1)
        pipeline = self._pipeline(
            ProductDetection(self.slots, 0.95, "5 cell hình học rõ"),
            ProductDetection([], 0.0, "không có candidate thứ hai"),
        )

        result = pipeline.inspect_frame(frame, "VT001")

        self.assertEqual(result.validation.status, ProductStatus.FAIL)
        self.assertEqual(result.validation.reason.value, "MISSING_ITEM")

    def test_colour_in_wrong_position_is_ng(self) -> None:
        frame = self._frame_with_expected_colours()
        # Đổi chỗ màu đầu và màu giữa, còn đủ cả ba slot.
        cv2.circle(frame, (self.slots[0].x, self.slots[0].y), 20, (255, 0, 0), -1)
        cv2.circle(frame, (self.slots[1].x, self.slots[1].y), 20, (180, 0, 180), -1)
        pipeline = self._pipeline(
            ProductDetection(self.slots, 0.95, "5 slot hình học rõ"),
            ProductDetection([], 0.0, "không có candidate thứ hai"),
        )

        result = pipeline.inspect_frame(frame, "VT001")

        self.assertEqual(result.validation.status, ProductStatus.FAIL)
        self.assertEqual(result.validation.reason.value, "WRONG_COLOR")

    def test_multiple_candidates_are_unknown_and_exposed_for_tracker(self) -> None:
        second_slots = [DetectedSlot(70, 170, 24), DetectedSlot(150, 170, 24), DetectedSlot(230, 170, 24)]
        pipeline = self._pipeline(
            ProductDetection(self.slots, 0.95, "candidate một"),
            ProductDetection(second_slots, 0.95, "candidate hai"),
        )

        result = pipeline.inspect_frame(self._frame_with_expected_colours(), "VT001")

        self.assertEqual(result.validation.status, ProductStatus.UNKNOWN)
        self.assertEqual(len(result.detections), 2)
        self.assertTrue(all(item.validation.status is ProductStatus.UNKNOWN for item in result.detections))
        self.assertIn("nhiều candidate", result.detail)

    def test_realtime_can_track_two_separate_valid_packs_in_one_frame(self) -> None:
        """Băng tải có vỉ kế tiếp vào khung không được làm mất vỉ trước."""
        first = [DetectedSlot(80, 70, 28), DetectedSlot(160, 70, 28), DetectedSlot(240, 70, 28)]
        second = [DetectedSlot(80, 190, 28), DetectedSlot(160, 190, 28), DetectedSlot(240, 190, 28)]
        frame = np.zeros((280, 320, 3), dtype=np.uint8)
        for row in (first, second):
            for slot, bgr in zip(row, [(180, 0, 180), (255, 0, 0), (180, 0, 180)], strict=True):
                cv2.circle(frame, (slot.x, slot.y), 15, bgr, -1)
        pipeline = FrameInspectionPipeline(
            self.manager,  # type: ignore[arg-type] - test double has ProductManager protocol.
            self.classifier,
            detector=_SequencedDetector(
                ProductDetection(first, 0.95, "candidate một"),
                ProductDetection(second, 0.95, "candidate hai"),
            ),  # type: ignore[arg-type]
            validator=ProductValidator(),
            allow_multiple_candidates=True,
        )

        result = pipeline.inspect_frame(frame, "VT001")

        self.assertEqual(result.validation.status, ProductStatus.PASS)
        self.assertEqual(len(result.detections), 2)
        self.assertTrue(all(item.validation.status is ProductStatus.PASS for item in result.detections))
        self.assertIn("đồng thời 2 vỉ", result.detail)

    def test_realtime_color_fallback_collects_four_separate_packs(self) -> None:
        """ROI rộng có bốn vỉ nghiêng vẫn phải giữ đủ bốn, không chỉ vỉ đầu."""
        frame = np.zeros((300, 440, 3), dtype=np.uint8)
        for y in (70, 220):
            for left in (70, 270):
                row = [
                    DetectedSlot(left, y, 30),
                    DetectedSlot(left + 50, y, 30),
                    DetectedSlot(left + 100, y, 30),
                ]
                for slot, bgr in zip(row, [(180, 0, 180), (255, 0, 0), (180, 0, 180)], strict=True):
                    cv2.circle(frame, (slot.x, slot.y), 15, bgr, -1)
        pipeline = FrameInspectionPipeline(
            self.manager,  # type: ignore[arg-type] - kiểm thử fallback ảnh rộng.
            self.classifier,
            detector=_SequencedDetector(
                ProductDetection([], 0.0, "Hough không thấy vỉ nghiêng"),
            ),  # type: ignore[arg-type]
            validator=ProductValidator(),
            allow_multiple_candidates=True,
            enable_color_guided_fallback=True,
        )

        result = pipeline.inspect_frame(frame, "VT001", allow_profile_color_fallback=True)

        self.assertEqual(result.validation.status, ProductStatus.PASS)
        self.assertEqual(len(result.detections), 4)
        self.assertTrue(all(item.validation.status is ProductStatus.PASS for item in result.detections))
        self.assertIn("đồng thời 4 vỉ", result.detail)

    def test_realtime_exposes_secondary_ng_for_roi_alerting(self) -> None:
        """Vỉ NG ở cùng ROI không được bị bỏ qua chỉ vì vỉ đầu đang PASS."""
        first = [DetectedSlot(80, 70, 28), DetectedSlot(160, 70, 28), DetectedSlot(240, 70, 28)]
        second = [DetectedSlot(80, 190, 28), DetectedSlot(160, 190, 28), DetectedSlot(240, 190, 28)]
        frame = np.zeros((280, 320, 3), dtype=np.uint8)
        for slot, bgr in zip(first, [(180, 0, 180), (255, 0, 0), (180, 0, 180)], strict=True):
            cv2.circle(frame, (slot.x, slot.y), 15, bgr, -1)
        # Slot giữa của vỉ thứ hai phải là Xanh dương nhưng cố ý để Tím.
        for slot, bgr in zip(second, [(180, 0, 180), (180, 0, 180), (180, 0, 180)], strict=True):
            cv2.circle(frame, (slot.x, slot.y), 15, bgr, -1)
        pipeline = FrameInspectionPipeline(
            self.manager,  # type: ignore[arg-type] - test double has ProductManager protocol.
            self.classifier,
            detector=_SequencedDetector(
                ProductDetection(first, 0.95, "vỉ đạt"),
                ProductDetection(second, 0.95, "vỉ sai màu"),
            ),  # type: ignore[arg-type]
            validator=ProductValidator(),
            allow_multiple_candidates=True,
        )

        result = pipeline.inspect_frame(frame, "VT001")

        self.assertEqual(len(result.detections), 2)
        self.assertEqual(result.detections[0].validation.status, ProductStatus.PASS)
        self.assertEqual(result.detections[1].validation.status, ProductStatus.FAIL)
        self.assertEqual(result.validation.status, ProductStatus.FAIL)
        self.assertIn("1 vỉ NG", result.detail)

    def test_invalid_bgr_input_is_unknown_instead_of_raising(self) -> None:
        pipeline = self._pipeline(ProductDetection(self.slots, 0.95, "không dùng"))

        result = pipeline.inspect_frame(np.zeros((10, 10), dtype=np.uint8), "VT001")

        self.assertEqual(result.validation.status, ProductStatus.UNKNOWN)
        self.assertEqual(result.annotated_frame.shape, (1, 1, 3))

    def test_color_classifier_error_becomes_unknown(self) -> None:
        pipeline = self._pipeline(
            ProductDetection(self.slots, 0.95, "hình học tốt"),
            ProductDetection([], 0.0, "không dùng"),
            classifier=_BrokenClassifier(self.classifier.profiles),  # type: ignore[arg-type]
        )

        result = pipeline.inspect_frame(self._frame_with_expected_colours(), "VT001")

        self.assertEqual(result.validation.status, ProductStatus.UNKNOWN)
        self.assertIn("màu", result.detail.casefold())


if __name__ == "__main__":
    unittest.main()
