import unittest

import cv2
import numpy as np

from ai.profile_color_detector import ProfileColorSequenceDetector
from core.models import ProductProfile, SlotSpec


class ProfileColorSequenceDetectorTests(unittest.TestCase):
    @staticmethod
    def _colors() -> dict[str, dict[str, list[int]]]:
        return {
            "green": {"hsv_min": [36, 35, 25], "hsv_max": [89, 255, 255]},
            "blue": {"hsv_min": [90, 35, 25], "hsv_max": [114, 255, 255]},
        }

    @staticmethod
    def _profile() -> ProductProfile:
        return ProductProfile(
            "demo",
            "Dãy màu chéo",
            [
                SlotSpec(index, 0.1, 0.5, color)
                for index, color in enumerate(["green", "green", "blue", "green", "green"], start=1)
            ],
        )

    def test_splits_touching_same_colour_runs_along_diagonal_pack(self) -> None:
        """Hai viên cùng màu dính nhau vẫn phải tạo thành hai slot riêng."""
        frame = np.full((480, 520, 3), 35, dtype=np.uint8)
        centers = [(100, 100), (160, 140), (240, 195), (320, 250), (380, 290)]
        bgr_colors = [(0, 200, 0), (0, 200, 0), (200, 40, 0), (0, 200, 0), (0, 200, 0)]
        for center, color in zip(centers, bgr_colors, strict=True):
            cv2.circle(frame, center, 42, color, -1)
        detection = ProfileColorSequenceDetector(self._colors()).detect(frame, self._profile())

        self.assertEqual(len(detection.slots), 5)
        self.assertGreaterEqual(detection.confidence, 0.85)
        self.assertTrue(all(slot.sample_radius is not None for slot in detection.slots))
        self.assertEqual([slot.matched_color for slot in detection.slots], ["green", "green", "blue", "green", "green"])
        self.assertEqual(
            [(slot.x, slot.y) for slot in detection.slots],
            [(99, 99), (161, 141), (240, 195), (319, 249), (381, 291)],
        )

    def test_accepts_a_flexible_pack_where_centre_slot_is_visibly_higher(self) -> None:
        """Góc camera làm slot giữa tạo chữ V không được biến thành UNKNOWN."""
        frame = np.full((440, 520, 3), 35, dtype=np.uint8)
        circles = [
            ((110, 260), (0, 200, 0), 26),
            ((150, 260), (0, 200, 0), 26),
            ((230, 195), (200, 40, 0), 18),
            ((310, 260), (0, 200, 0), 26),
            ((350, 260), (0, 200, 0), 26),
        ]
        for center, color, radius in circles:
            cv2.circle(frame, center, radius, color, -1)

        detection = ProfileColorSequenceDetector(self._colors()).detect(frame, self._profile())

        self.assertEqual(len(detection.slots), 5)
        self.assertGreaterEqual(detection.confidence, 0.85)
        self.assertEqual((detection.slots[2].x, detection.slots[2].y), (230, 195))

    def test_profile_only_blue_extension_handles_dark_webcam_blue(self) -> None:
        """Blue tối qua webcam chỉ được nới trong chuỗi profile đã khớp."""
        colors = self._colors()
        colors["blue"].update({
            "profile_hsv_extension_min": [115, 40, 25],
            "profile_hsv_extension_max": [125, 255, 255],
        })
        frame = np.full((300, 520, 3), 35, dtype=np.uint8)
        blue_bgr = tuple(int(value) for value in cv2.cvtColor(
            np.uint8([[[118, 75, 145]]]), cv2.COLOR_HSV2BGR
        )[0, 0])
        for center, color in zip(
            # Hai cặp xanh lá phải chạm nhau để đây là đúng tình huống
            # detector run-màu được thiết kế để tách, không phải lấy một
            # hình tròn riêng rồi bịa thêm slot từ chính nó.
            [(80, 150), (135, 150), (220, 150), (305, 150), (360, 150)],
            [(0, 190, 0), (0, 190, 0), blue_bgr, (0, 190, 0), (0, 190, 0)],
            strict=True,
        ):
            cv2.circle(frame, center, 30, color, -1)

        detection = ProfileColorSequenceDetector(colors).detect(frame, self._profile())

        self.assertEqual(len(detection.slots), 5)
        self.assertEqual(
            [slot.matched_color for slot in detection.slots],
            ["green", "green", "blue", "green", "green"],
        )

    def test_rejects_a_run_when_one_component_is_split_into_a_tiny_fake_slot(self) -> None:
        """Một slot xanh không được bị tách đôi để che vỉ thiếu viên."""
        frame = np.full((360, 520, 3), 35, dtype=np.uint8)
        circles = [
            ((120, 180), (0, 200, 0), 27),
            # Không có viên xanh thứ hai ở đầu dãy: khoảng cách slot giả
            # sau khi split component sẽ rất nhỏ so với các khoảng còn lại.
            ((250, 180), (200, 40, 0), 27),
            ((340, 180), (0, 200, 0), 31),
            ((405, 180), (0, 200, 0), 31),
        ]
        for center, color, radius in circles:
            cv2.circle(frame, center, radius, color, -1)

        detection = ProfileColorSequenceDetector(self._colors()).detect(frame, self._profile())

        self.assertEqual(detection.slots, [])
        self.assertIn("quá sát", detection.detail)

    def test_detects_four_profile_coloured_slots_and_records_the_missing_position(self) -> None:
        """Fallback n-1 slot phải giữ metadata để pipeline báo NG đúng ô."""
        frame = np.full((400, 560, 3), 35, dtype=np.uint8)
        circles = [
            ((105, 210), (0, 200, 0), 29),
            ((225, 210), (200, 40, 0), 28),
            ((310, 210), (0, 200, 0), 31),
            ((370, 210), (0, 200, 0), 31),
        ]
        for center, color, radius in circles:
            cv2.circle(frame, center, radius, color, -1)

        detection = ProfileColorSequenceDetector(self._colors()).detect_partial(frame, self._profile())

        self.assertEqual(len(detection.slots), 4)
        self.assertEqual(detection.missing_slot_index, 1)
        self.assertEqual([slot.matched_color for slot in detection.slots], ["green", "blue", "green", "green"])


if __name__ == "__main__":
    unittest.main()
