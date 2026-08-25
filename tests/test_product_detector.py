import unittest
from pathlib import Path

import cv2
import numpy as np

from ai.product_detector import GeometricProductDetector


class ProductDetectorTests(unittest.TestCase):
    def test_regular_row_is_selected_and_ordered(self) -> None:
        circles = np.array([
            [500, 100, 60], [100, 101, 60], [300, 99, 59], [700, 100, 61], [900, 102, 60],
            [300, 450, 90],
        ], dtype=float)
        score, selected = GeometricProductDetector._best_regular_row(circles, 5)
        self.assertLess(score, 0.05)
        self.assertIsNotNone(selected)
        self.assertEqual(selected[:, 0].astype(int).tolist(), [100, 300, 500, 700, 900])

    def test_chroma_contours_detect_a_moderately_perspective_row(self) -> None:
        """Fallback ellipse phải nhận được mặt vỉ nghiêng, không cần Hough."""
        image = np.full((190, 340, 3), 28, dtype=np.uint8)
        centers = [(38, 58), (80, 69), (128, 82), (184, 97), (251, 115)]
        axes = [(19, 16), (21, 17), (24, 19), (27, 21), (30, 23)]
        for center, axis in zip(centers, axes, strict=True):
            cv2.ellipse(image, center, axis, 12, 0, 360, (185, 0, 185), -1)
        detector = GeometricProductDetector(
            hough_param2=1000,
            enable_clahe_fallback=False,
            maximum_slot_radius_fraction=0.18,
        )

        result = detector.detect(image, 5)

        self.assertEqual(len(result.slots), 5)
        self.assertIn("contour vùng màu", result.detail)
        self.assertGreaterEqual(result.confidence, 0.85)
        self.assertTrue(any(slot.radius_x != slot.radius_y for slot in result.slots))

    def test_edge_on_ellipses_remain_unknown(self) -> None:
        """Không được coi các ô dẹt nhìn cạnh là đủ bằng chứng để PASS."""
        image = np.full((150, 340, 3), 28, dtype=np.uint8)
        for x in (42, 101, 160, 219, 278):
            cv2.ellipse(image, (x, 75), (25, 5), 0, 0, 360, (185, 0, 185), -1)
        detector = GeometricProductDetector(
            hough_param2=1000,
            enable_clahe_fallback=False,
            minimum_ellipse_axis_ratio=0.50,
        )

        result = detector.detect(image, 5)

        self.assertEqual(result.slots, [])
        self.assertEqual(result.confidence, 0.0)

    def test_reference_photos_support_front_back_and_side_outcome(self) -> None:
        root = Path(__file__).resolve().parents[1]
        detector = GeometricProductDetector()
        expected_counts = {
            "sample_001_unconfirmed.png": 5,
            "sample_002_front_unconfirmed.jpg": 5,
            # Góc cạnh vẫn có thể xác nhận khi 5 tâm Hough tạo thành dãy rõ
            # và mỗi tâm được căn lại về vùng màu bên trong vỉ.
            "sample_003_side_unconfirmed.jpg": 5,
            "sample_004_back_unconfirmed.jpg": 5,
        }
        for name, expected_count in expected_counts.items():
            with self.subTest(name=name):
                # cv2.imread không xử lý ổn định absolute Unicode path trên một
                # số bản OpenCV Windows; imdecode từ bytes thì ổn định.
                data = np.fromfile(root / "data" / "samples" / "inbox" / name, dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                self.assertIsNotNone(image)
                result = detector.detect(image, 5)
                self.assertEqual(len(result.slots), expected_count)
                if name == "sample_003_side_unconfirmed.jpg":
                    self.assertIn("side-view", result.detail)
                    self.assertTrue(all(slot.side_view for slot in result.slots))
                    self.assertTrue(all(slot.sample_radius is not None for slot in result.slots))
