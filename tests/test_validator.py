import unittest

from ai.validator import ProductValidator
from core.models import FailureReason, ProductProfile, ProductStatus, SlotObservation, SlotSpec


class ValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = ProductProfile("VT001", "Vỉ A", [
            SlotSpec(1, .2, .5, "purple"), SlotSpec(2, .5, .5, "blue")])
        self.validator = ProductValidator()

    def test_missing_item_is_fail(self) -> None:
        result = self.validator.validate(self.profile, [
            SlotObservation(1, True, "purple", .99, .99), SlotObservation(2, False, None, 0, 0)], .99)
        self.assertEqual(result.status, ProductStatus.FAIL)
        self.assertEqual(result.reason, FailureReason.MISSING_ITEM)

    def test_ambiguous_color_is_unknown(self) -> None:
        result = self.validator.validate(self.profile, [
            SlotObservation(1, True, "purple", .99, .99), SlotObservation(2, True, None, .40, .4)], .99)
        self.assertEqual(result.status, ProductStatus.UNKNOWN)

    def test_clear_unknown_colour_is_wrong_colour_ng(self) -> None:
        result = self.validator.validate(self.profile, [
            SlotObservation(1, True, "purple", .99, .99), SlotObservation(2, True, None, .99, .0)], .99)
        self.assertEqual(result.status, ProductStatus.FAIL)
        self.assertEqual(result.reason, FailureReason.WRONG_COLOR)

    def test_wrong_color_is_fail(self) -> None:
        result = self.validator.validate(self.profile, [
            SlotObservation(1, True, "purple", .99, .99), SlotObservation(2, True, "purple", .99, .99)], .99)
        self.assertEqual(result.reason, FailureReason.WRONG_COLOR)


if __name__ == "__main__":
    unittest.main()
