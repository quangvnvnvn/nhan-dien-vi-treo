import unittest

from core.colors import ColorCatalog


class ColorCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = ColorCatalog({
            "purple": {"display_name": "Tím", "aliases": ["tím", "lavender"]},
            "blue": {"display_name": "Xanh dương", "aliases": ["xanh duong"]},
        })

    def test_vietnamese_input_is_normalized(self) -> None:
        self.assertEqual(self.catalog.normalize("Tím"), "purple")
        self.assertEqual(self.catalog.normalize("xanh dương"), "blue")

    def test_display_name_is_vietnamese(self) -> None:
        self.assertEqual(self.catalog.display("purple"), "Tím")
