import unittest
from pathlib import Path

from ai.color_classifier import ColorClassifier
from ai.product_detector import GeometricProductDetector
from ai.validator import ProductValidator
from core.config import load_yaml
from core.models import ProductProfile, ProductStatus, SlotSpec
from services.test_service import TestService


class TestServiceTests(unittest.TestCase):
    @staticmethod
    def _lavender_profile() -> ProductProfile:
        """Fixture độc lập để test không phụ thuộc profile người dùng đang sửa/xóa."""
        colors = ["purple", "purple", "blue", "purple", "purple"]
        slots = [
            SlotSpec(index=index + 1, x=(index + 0.5) / len(colors), y=0.5, expected_color=color)
            for index, color in enumerate(colors)
        ]
        return ProductProfile("fixture-lavender", "Lavender fixture", slots, minimum_confidence=0.85)

    def test_reads_image_under_unicode_workspace_path(self) -> None:
        """OpenCV phải đọc được ảnh dù thư mục làm việc có dấu tiếng Việt."""
        root = Path(__file__).resolve().parents[1]
        image_path = root / "data" / "samples" / "inbox" / "sample_001_unconfirmed.png"
        service = TestService(ColorClassifier({}), ProductValidator())

        result = service.inspect_image(image_path, None)

        self.assertEqual(result.status, ProductStatus.UNKNOWN)

    def test_reference_back_photo_passes_after_color_boundary_resolution(self) -> None:
        """Tím áp sát ngưỡng xanh dương vẫn phải thắng khi độ phủ áp đảo."""
        root = Path(__file__).resolve().parents[1]
        colors = load_yaml(root / "config" / "colors.yaml")["colors"]
        profile = self._lavender_profile()
        service = TestService(ColorClassifier(colors), ProductValidator(), GeometricProductDetector())

        result = service.inspect_image(
            root / "data" / "samples" / "inbox" / "sample_004_back_unconfirmed.jpg",
            profile,
        )

        self.assertEqual(result.status, ProductStatus.PASS)
        self.assertEqual([slot.color for slot in result.observations], [
            "purple", "purple", "blue", "purple", "purple",
        ])

    def test_reference_side_photo_passes_after_side_view_refinement(self) -> None:
        """Ảnh cạnh được căn vào phần viên màu thay vì vách nhựa trong suốt."""
        root = Path(__file__).resolve().parents[1]
        colors = load_yaml(root / "config" / "colors.yaml")["colors"]
        profile = self._lavender_profile()
        service = TestService(ColorClassifier(colors), ProductValidator(), GeometricProductDetector())

        result = service.inspect_image(
            root / "data" / "samples" / "inbox" / "sample_003_side_unconfirmed.jpg",
            profile,
        )

        self.assertEqual(result.status, ProductStatus.PASS)
        self.assertGreaterEqual(result.confidence, 0.85)
        self.assertEqual([slot.color for slot in result.observations], [
            "purple", "purple", "blue", "purple", "purple",
        ])


if __name__ == "__main__":
    unittest.main()
