import unittest

from ai.tracker import CentroidTracker, CountingDirection, TrackingInput
from core.models import ProductStatus, ValidationResult


def passing() -> ValidationResult:
    return ValidationResult("VT001", ProductStatus.PASS, None, confidence=0.95)


def failing() -> ValidationResult:
    return ValidationResult("VT001", ProductStatus.FAIL, None, confidence=0.95)


class TrackerTests(unittest.TestCase):
    def test_counts_once_after_validated_track_crosses_line(self) -> None:
        tracker = CentroidTracker(min_validation_hits=2, distance_threshold=100)
        first = tracker.update([TrackingInput((10, 10, 20, 20), passing())], 100, CountingDirection.TOP_TO_BOTTOM)
        second = tracker.update([TrackingInput((10, 60, 20, 20), passing())], 100, CountingDirection.TOP_TO_BOTTOM)
        third = tracker.update([TrackingInput((10, 110, 20, 20), passing())], 100, CountingDirection.TOP_TO_BOTTOM)
        fourth = tracker.update([TrackingInput((10, 150, 20, 20), passing())], 100, CountingDirection.TOP_TO_BOTTOM)
        self.assertEqual(len(first.newly_counted), 0)
        self.assertEqual(len(second.newly_counted), 0)
        self.assertEqual(len(third.newly_counted), 1)
        self.assertEqual(len(fourth.newly_counted), 0)

    def test_unresolved_track_reports_review_only_after_leaving_frame(self) -> None:
        tracker = CentroidTracker(max_age=0, min_validation_hits=2, distance_threshold=100)
        unknown = ValidationResult("VT001", ProductStatus.UNKNOWN, None)
        tracker.update([TrackingInput((10, 10, 20, 20), unknown)], 100, CountingDirection.TOP_TO_BOTTOM)
        update = tracker.update([TrackingInput((10, 110, 20, 20), unknown)], 100, CountingDirection.TOP_TO_BOTTOM)
        self.assertEqual(update.newly_counted, [])
        self.assertEqual(update.crossed_not_counted, [])
        expired = tracker.update([], 100, CountingDirection.TOP_TO_BOTTOM)
        self.assertEqual(len(expired.crossed_not_counted), 1)

    def test_failure_after_crossing_upgrades_pending_review_to_ng(self) -> None:
        """Sai màu/thiếu slot thấy rõ muộn vẫn phải ghi NG, không mất lượt."""
        tracker = CentroidTracker(min_validation_hits=2, distance_threshold=100)
        unknown = ValidationResult("VT001", ProductStatus.UNKNOWN, None)
        tracker.update([TrackingInput((10, 10, 20, 20), unknown)], 100, CountingDirection.TOP_TO_BOTTOM)
        pending = tracker.update([TrackingInput((10, 110, 20, 20), unknown)], 100, CountingDirection.TOP_TO_BOTTOM)
        rejected = tracker.update([TrackingInput((10, 135, 20, 20), failing())], 100, CountingDirection.TOP_TO_BOTTOM)

        self.assertEqual(pending.crossed_not_counted, [])
        self.assertEqual(len(rejected.crossed_not_counted), 1)
        self.assertIs(rejected.crossed_not_counted[0].validation.status, ProductStatus.FAIL)

    def test_fast_pack_counts_when_its_leading_edge_reaches_line(self) -> None:
        """Không bỏ sót vỉ cao: đáy qua vạch trước khi tâm kịp qua."""
        tracker = CentroidTracker(min_validation_hits=2, distance_threshold=180)
        tracker.update([TrackingInput((10, 100, 100, 180), passing())], 400, CountingDirection.TOP_TO_BOTTOM)
        update = tracker.update([TrackingInput((10, 250, 100, 180), passing())], 400, CountingDirection.TOP_TO_BOTTOM)

        self.assertEqual(len(update.newly_counted), 1)

    def test_fast_valid_pack_seen_first_at_line_is_counted(self) -> None:
        """Bỏ frame trước vạch không được làm mất một vỉ PASS rõ ràng."""
        tracker = CentroidTracker(min_validation_hits=1, distance_threshold=320)
        update = tracker.update(
            [TrackingInput((10, 85, 100, 40), passing())],
            100,
            CountingDirection.TOP_TO_BOTTOM,
        )

        self.assertEqual(len(update.newly_counted), 1)

    def test_fast_invalid_pack_seen_first_at_line_is_ng(self) -> None:
        """Vỉ thiếu viên/sai màu cũng phải có event khi xuất hiện muộn."""
        tracker = CentroidTracker(min_validation_hits=1, distance_threshold=320)
        update = tracker.update(
            [TrackingInput((10, 85, 100, 40), failing())],
            100,
            CountingDirection.TOP_TO_BOTTOM,
        )

        self.assertEqual(len(update.crossed_not_counted), 1)
        self.assertIs(update.crossed_not_counted[0].validation.status, ProductStatus.FAIL)
