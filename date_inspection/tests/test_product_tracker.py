from __future__ import annotations

import unittest

from detection.product_detector import DetectionMetrics
from processing.product_tracker import ProductTracker


PRESENT = DetectionMetrics(True, .1, 12, 1000, None)
ABSENT = DetectionMetrics(False, 0, 0, 0, None)


class ProductTrackerTests(unittest.TestCase):
    def test_one_product_only_triggers_once_until_it_exits(self) -> None:
        tracker = ProductTracker(minimum_present=3, minimum_absent=2, debounce_ms=0, minimum_gap_frames=0)
        events = [tracker.update(PRESENT, number) for number in range(3)]
        self.assertTrue(events[-1].triggered)
        self.assertEqual(events[-1].product_id, "P000001")
        self.assertFalse(any(tracker.update(PRESENT, number).triggered for number in range(3, 10)))
        tracker.update(ABSENT, 10); tracker.update(ABSENT, 11)
        next_events = [tracker.update(PRESENT, number) for number in range(12, 15)]
        self.assertTrue(next_events[-1].triggered)
        self.assertEqual(next_events[-1].product_id, "P000002")

    def test_short_noise_does_not_trigger(self) -> None:
        tracker = ProductTracker(minimum_present=3, minimum_absent=2)
        tracker.update(PRESENT, 0); tracker.update(PRESENT, 1); event = tracker.update(ABSENT, 2)
        self.assertFalse(event.triggered)

    def test_debounce_blocks_a_new_trigger_that_arrives_too_soon(self) -> None:
        tracker = ProductTracker(minimum_present=1, minimum_absent=1, debounce_ms=300, minimum_gap_frames=0)
        self.assertTrue(tracker.update(PRESENT, 0, now=1.0).triggered)
        tracker.update(ABSENT, 1, now=1.01)
        tracker.update(ABSENT, 2, now=1.02)
        event = tracker.update(PRESENT, 3, now=1.1)
        self.assertTrue(event.duplicate_blocked)


if __name__ == "__main__":
    unittest.main()
