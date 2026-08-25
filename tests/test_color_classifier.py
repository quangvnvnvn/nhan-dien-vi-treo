from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from ai.color_classifier import ColorClassifier
from core.colors import ColorCatalog
from core.config import load_yaml


ROOT = Path(__file__).resolve().parents[1]


class ColorClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.colors = load_yaml(ROOT / "config" / "colors.yaml")["colors"]
        self.classifier = ColorClassifier(self.colors)
        self.catalog = ColorCatalog(self.colors)

    @staticmethod
    def _pixels(*groups: tuple[tuple[int, int, int], int]) -> np.ndarray:
        return np.vstack([
            np.tile(np.array(bgr, dtype=np.uint8), (count, 1))
            for bgr, count in groups
        ])

    def test_green_vietnamese_aliases_are_available(self) -> None:
        self.assertEqual(self.catalog.normalize("Xanh lá"), "green")
        self.assertEqual(self.catalog.normalize("xanh la"), "green")
        self.assertEqual(self.catalog.normalize("xanh lục"), "green")
        self.assertEqual(self.catalog.display("green"), "Xanh lá")

    def test_colored_pixels_win_over_low_saturation_reflections(self) -> None:
        result = self.classifier.classify(self._pixels(
            ((0, 190, 0), 70),
            ((245, 245, 245), 180),
            ((235, 238, 240), 50),
        ))

        self.assertEqual(result.name, "green")
        self.assertGreaterEqual(result.confidence, 0.85)

    def test_close_blue_purple_mix_stays_unknown(self) -> None:
        result = self.classifier.classify(self._pixels(
            ((180, 0, 180), 80),
            ((255, 0, 0), 80),
        ))

        self.assertIsNone(result.name)
        self.assertEqual(result.confidence, 0.0)

    def test_high_saturation_blue_at_hue_boundary_stays_blue(self) -> None:
        """Blue đậm OpenCV hue=120 không bị xếp vào Tím."""
        result = self.classifier.classify(self._pixels(((255, 0, 0), 120)))

        self.assertEqual(result.name, "blue")
        self.assertGreaterEqual(result.confidence, 0.85)

    def test_purple_starts_after_blue_band(self) -> None:
        """Dải Tím camera (hue=130) không chồng với Xanh dương."""
        bgr = tuple(int(value) for value in cv2.cvtColor(
            np.uint8([[[130, 130, 130]]]), cv2.COLOR_HSV2BGR
        )[0, 0])

        result = self.classifier.classify(self._pixels((bgr, 120)))

        self.assertEqual(result.name, "purple")
        self.assertGreaterEqual(result.confidence, 0.85)

    def test_sequence_requires_blue_slot_to_differ_from_purple_slots(self) -> None:
        """Một viên Tím ở vị trí slot 3 không được hiệu chỉnh thành Blue."""
        purple = tuple(int(value) for value in cv2.cvtColor(
            np.uint8([[[130, 130, 140]]]), cv2.COLOR_HSV2BGR
        )[0, 0])
        samples = [self._pixels((purple, 120)) for _ in range(5)]

        results = self.classifier.classify_sequence(
            samples,
            ["purple", "purple", "blue", "purple", "purple"],
        )

        self.assertEqual([item.name for item in results], ["purple"] * 5)

    def test_sequence_recovers_real_blue_when_it_is_separate_from_purple(self) -> None:
        """Blue hue=121 giữa các viên Tím hue=130 được xác nhận độc lập."""
        purple = tuple(int(value) for value in cv2.cvtColor(
            np.uint8([[[130, 130, 140]]]), cv2.COLOR_HSV2BGR
        )[0, 0])
        blue = tuple(int(value) for value in cv2.cvtColor(
            np.uint8([[[121, 145, 100]]]), cv2.COLOR_HSV2BGR
        )[0, 0])
        samples = [self._pixels((purple, 120)), self._pixels((purple, 120)), self._pixels((blue, 120)),
                   self._pixels((purple, 120)), self._pixels((purple, 120))]

        results = self.classifier.classify_sequence(
            samples,
            ["purple", "purple", "blue", "purple", "purple"],
        )

        self.assertEqual([item.name for item in results], ["purple", "purple", "blue", "purple", "purple"])

    def test_too_few_colored_pixels_stays_unknown(self) -> None:
        result = self.classifier.classify(self._pixels(
            ((0, 190, 0), 12),
            ((245, 245, 245), 300),
        ))

        self.assertIsNone(result.name)
        self.assertEqual(result.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
