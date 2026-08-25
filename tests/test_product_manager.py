import tempfile
import unittest
from pathlib import Path

from core.models import ManualStripLayout, ProductProfile, SlotSpec
from training.product_manager import ProductManager


class ProductManagerTests(unittest.TestCase):
    def test_save_replaces_same_product_id(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "products.yaml"
            path.write_text("products: []\n", encoding="utf-8")
            manager = ProductManager(path)
            manager.save(ProductProfile("VT001", "Bản 1", [SlotSpec(1, .5, .5, "purple")]))
            manager.save(ProductProfile("VT001", "Bản 2", [SlotSpec(1, .5, .5, "blue")]))
            profiles = manager.list_profiles()
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0].name, "Bản 2")
        self.assertEqual(profiles[0].expected_colors, ["blue"])

    def test_update_can_change_product_id_without_losing_other_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "products.yaml"
            path.write_text("products: []\n", encoding="utf-8")
            manager = ProductManager(path)
            manager.save(ProductProfile("A", "First", [SlotSpec(1, 0.5, 0.5, "green")]))
            manager.save(ProductProfile("B", "Other", [SlotSpec(1, 0.5, 0.5, "blue")]))
            manager.update("A", ProductProfile("A2", "Updated", [SlotSpec(1, 0.5, 0.5, "purple")]))
            self.assertIsNone(manager.get("A"))
            self.assertEqual(manager.get("A2").name, "Updated")
            self.assertEqual(manager.get("B").name, "Other")

    def test_delete_removes_only_requested_profile(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "products.yaml"
            path.write_text("products: []\n", encoding="utf-8")
            manager = ProductManager(path)
            manager.save(ProductProfile("A", "First", [SlotSpec(1, 0.5, 0.5, "green")]))
            manager.save(ProductProfile("B", "Other", [SlotSpec(1, 0.5, 0.5, "blue")]))
            self.assertTrue(manager.delete("A"))
            self.assertFalse(manager.delete("missing"))
            self.assertIsNone(manager.get("A"))
            self.assertIsNotNone(manager.get("B"))

    def test_manual_scan_layout_round_trips_with_profile(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "products.yaml"
            path.write_text("products: []\n", encoding="utf-8")
            manager = ProductManager(path)
            profile = ProductProfile(
                "FIXED", "Camera cố định", [SlotSpec(1, .5, .5, "green")],
                manual_scan_strips=[
                    ManualStripLayout(1, [SlotSpec(1, .25, .75, "green", .04)])
                ],
                manual_scan_roi=(.1, .2, .7, .6),
                manual_scan_sample_radius=.042,
            )
            manager.save(profile)
            reloaded = manager.get("FIXED")
        self.assertIsNotNone(reloaded)
        assert reloaded is not None
        self.assertEqual(reloaded.manual_scan_roi, (.1, .2, .7, .6))
        self.assertEqual(reloaded.manual_scan_sample_radius, .042)
        self.assertEqual(len(reloaded.manual_scan_strips), 1)
        self.assertEqual(reloaded.manual_scan_strips[0].slots[0].expected_color, "green")
