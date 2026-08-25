from __future__ import annotations

import unittest

from datecheck.date_parser import DateExpectation, DateParser, DateStatus


class DateParserTests(unittest.TestCase):
    def test_valid_two_line_date_passes(self) -> None:
        result = DateParser.validate("NSX 220826 DM1\nAB 08:35", DateExpectation("220826", "DM1", "AB"))

        self.assertEqual(result.status, DateStatus.PASS)
        self.assertEqual(result.read.time_value, "08:35")

    def test_wrong_date_is_ng(self) -> None:
        result = DateParser.validate("NSX 210826 DM1\nAB 08:35", DateExpectation("220826", "DM1", "AB"))

        self.assertEqual(result.status, DateStatus.FAIL)

    def test_incomplete_ocr_is_review(self) -> None:
        result = DateParser.validate("NSX 220826", DateExpectation())

        self.assertEqual(result.status, DateStatus.REVIEW)

    def test_invalid_calendar_date_is_review(self) -> None:
        result = DateParser.validate("NSX 390226 DM1\nAB 08:35", DateExpectation())

        self.assertEqual(result.status, DateStatus.REVIEW)

    def test_common_ocr_misreads_are_normalised_only_in_date_fields(self) -> None:
        result = DateParser.validate("N5X 13O826 DMl\n07 O8;3S", DateExpectation("130826", "DM1", "07"))

        self.assertEqual(result.status, DateStatus.PASS)
        self.assertEqual(result.read.time_value, "08:35")


if __name__ == "__main__":
    unittest.main()
