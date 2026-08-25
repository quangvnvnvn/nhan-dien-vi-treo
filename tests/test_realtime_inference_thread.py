import tempfile
import unittest
from pathlib import Path

import numpy as np

from ai.realtime_pipeline import FrameInspectionResult, ProductInspection
from ai.tracker import CentroidTracker, CountingDirection
from camera.frame_queue import LatestFrameQueue
from core.models import FailureReason, ProductStatus, ValidationResult
from services.realtime_inference_thread import RealtimeInferenceThread
from services.result_artifacts import ResultArtifactStore


class _MonitoringPipeline:
    """Pipeline giả có một vỉ PASS và một vỉ FAIL trong cùng ROI."""

    def inspect_frame(self, frame: np.ndarray, product_id: str | None, **_kwargs: object) -> FrameInspectionResult:
        passed = ProductInspection(
            product_id or "VT001", (10, 10, 50, 30),
            ValidationResult(product_id or "VT001", ProductStatus.PASS, None, confidence=0.95),
            frame[10:40, 10:60].copy(), [], 0.95, "vỉ đạt",
        )
        failed = ProductInspection(
            product_id or "VT001", (90, 10, 50, 30),
            ValidationResult(
                product_id or "VT001", ProductStatus.FAIL, FailureReason.WRONG_COLOR,
                confidence=0.95, detail="Sai màu tại slot 3",
            ),
            frame[10:40, 90:140].copy(), [], 0.95, "vỉ sai màu",
        )
        return FrameInspectionResult(
            product_id, failed.validation, failed.bounding_box, failed.product_crop,
            [passed, failed], frame.copy(), "Phát hiện đồng thời 2 vỉ; 1 vỉ NG.",
        )


class _CountingPipeline:
    """Pipeline giả để kiểm tra worker chỉ gọi inference lúc vỉ đã đứng yên."""

    def __init__(self) -> None:
        self.calls = 0

    def inspect_frame(self, frame: np.ndarray, product_id: str | None, **_kwargs: object) -> FrameInspectionResult:
        self.calls += 1
        validation = ValidationResult(product_id or "VT001", ProductStatus.PASS, None, confidence=0.95, detail="Vỉ đạt")
        return FrameInspectionResult(
            product_id,
            validation,
            None,
            None,
            [],
            frame.copy(),
            "Vỉ đạt",
        )


class RealtimeInferenceThreadTests(unittest.TestCase):
    def test_diagonal_counting_line_is_rejected_for_axis_tracker(self) -> None:
        self.assertIsNone(
            RealtimeInferenceThread._line_position(
                ((0.1, 0.1), (0.9, 0.9)), (480, 640, 3), CountingDirection.TOP_TO_BOTTOM,
            )
        )
        self.assertEqual(
            RealtimeInferenceThread._line_position(
                ((0.0, 0.65), (1.0, 0.65)), (480, 640, 3), CountingDirection.TOP_TO_BOTTOM,
            ),
            312.0,
        )

    def test_non_pass_artifact_has_image_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            # Cần bảo đảm evidence vẫn ghi được khi một phần đường dẫn có dấu.
            store = ResultArtifactStore(Path(folder) / "kết quả kiểm tra")
            result = ValidationResult("VT001", ProductStatus.UNKNOWN, FailureReason.UNKNOWN, confidence=0.2)
            saved = store.save_non_pass(np.zeros((20, 20, 3), dtype=np.uint8), 17, result)
            self.assertIsNotNone(saved)
            assert saved is not None
            self.assertTrue(saved.exists())
            self.assertTrue(saved.with_suffix(".json").exists())

    def test_roi_monitoring_alerts_immediately_without_counting(self) -> None:
        worker = RealtimeInferenceThread(
            LatestFrameQueue(),
            _MonitoringPipeline(),  # type: ignore[arg-type] - kiểm thử boundary pipeline.
            CentroidTracker(),
        )
        worker.configure(
            product_id="VT001", roi=None, counting_line=None,
            direction=CountingDirection.TOP_TO_BOTTOM,
            inspection_mode="realtime",
        )

        output = worker._process(np.zeros((80, 180, 3), dtype=np.uint8))

        self.assertEqual(output.status, ProductStatus.FAIL.value)
        self.assertEqual(output.visible_products, 2)
        self.assertEqual(output.fault_products, 1)
        self.assertTrue(output.alert_active)
        self.assertIn("CẢNH BÁO NG", output.alert_detail)
        self.assertEqual(output.count_delta, 0)
        self.assertEqual(output.detected_delta, 0)

    def test_stop_and_scan_inspects_once_per_stop_period(self) -> None:
        pipeline = _CountingPipeline()
        worker = RealtimeInferenceThread(LatestFrameQueue(), pipeline, CentroidTracker())  # type: ignore[arg-type]
        worker.configure(
            product_id="VT001",
            roi=None,
            counting_line=None,
            direction=CountingDirection.TOP_TO_BOTTOM,
            inspection_mode="on_stop",
            stable_frames_required=3,
            motion_threshold=2.0,
        )
        stopped = np.zeros((80, 120, 3), dtype=np.uint8)

        for _ in range(3):
            output = worker._process(stopped)
        self.assertEqual(output.status, ProductStatus.PASS.value)
        self.assertEqual(pipeline.calls, 1)

        for _ in range(5):
            worker._process(stopped)
        self.assertEqual(pipeline.calls, 1, "Một nhịp dừng chỉ được kết luận một lần")

        moving = np.full((80, 120, 3), 80, dtype=np.uint8)
        output = worker._process(moving)
        self.assertEqual(output.status, ProductStatus.UNKNOWN.value)
        self.assertIn("đang chạy", output.detail.lower())
        for _ in range(3):
            output = worker._process(moving)
        self.assertEqual(output.status, ProductStatus.PASS.value)
        self.assertEqual(pipeline.calls, 2)

    def test_realtime_mode_inspects_every_frame(self) -> None:
        pipeline = _CountingPipeline()
        worker = RealtimeInferenceThread(LatestFrameQueue(), pipeline, CentroidTracker())  # type: ignore[arg-type]
        worker.configure(
            product_id="VT001",
            roi=None,
            counting_line=None,
            direction=CountingDirection.TOP_TO_BOTTOM,
            inspection_mode="realtime",
        )
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        worker._process(frame)
        worker._process(frame)
        self.assertEqual(pipeline.calls, 2)
