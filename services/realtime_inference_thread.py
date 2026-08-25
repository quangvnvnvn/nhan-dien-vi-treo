"""Worker inference realtime: lấy frame mới nhất, kiểm tra, tracking và audit."""
from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import threading
import time
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from ai.realtime_pipeline import FrameInspectionPipeline, FrameInspectionResult
from ai.tracker import CentroidTracker, CountingDirection, TrackingInput
from camera.frame_queue import LatestFrameQueue
from core.models import ProductStatus
from database.database import ResultRepository
from services.daily_excel_exporter import DailyExcelExporter
from services.result_artifacts import ResultArtifactStore

LOGGER = logging.getLogger(__name__)
NormalizedRect = tuple[float, float, float, float]
NormalizedLine = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True, slots=True)
class InferenceSettings:
    product_id: str | None = None
    roi: NormalizedRect | None = None
    counting_line: NormalizedLine | None = None
    direction: CountingDirection = CountingDirection.TOP_TO_BOTTOM
    # ``on_stop`` lấy một ảnh nét khi băng tải đã đứng yên; ``realtime`` giữ
    # cách kiểm tra mỗi frame để dùng khi sản phẩm chạy liên tục.
    inspection_mode: str = "on_stop"
    stable_frames_required: int = 6
    motion_threshold: float = 2.0


@dataclass(frozen=True, slots=True)
class InferenceMetrics:
    frames_processed: int = 0
    latest_latency_ms: float = 0.0
    average_fps: float = 0.0
    dropped_frames: int = 0
    errors: int = 0


@dataclass(slots=True)
class RealtimeInferenceOutput:
    status: str
    product_id: str | None
    detected_colors: tuple[str, ...]
    confidence: float
    count_delta: int
    detected_delta: int
    pass_delta: int
    fail_delta: int
    unknown_delta: int
    detail: str
    annotated_frame: QImage
    fps: float
    visible_products: int = 0
    fault_products: int = 0
    alert_active: bool = False
    alert_detail: str = ""


class RealtimeInferenceThread(QThread):
    """Không chạy inference trong UI thread; buffer camera luôn bỏ frame cũ."""

    inference_ready = Signal(object)
    metrics_updated = Signal(object)
    inference_error = Signal(str)

    def __init__(
        self,
        frames: LatestFrameQueue,
        pipeline: FrameInspectionPipeline,
        tracker: CentroidTracker,
        *,
        repository: ResultRepository | None = None,
        artifact_store: ResultArtifactStore | None = None,
        daily_excel_exporter: DailyExcelExporter | None = None,
        color_display: Callable[[str | None], str] | None = None,
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        self.frames = frames
        self.pipeline = pipeline
        self.tracker = tracker
        self.repository = repository
        self.artifact_store = artifact_store
        self.daily_excel_exporter = daily_excel_exporter
        self.color_display = color_display or (lambda color: color or "--")
        self._stop_requested = threading.Event()
        self._settings_lock = threading.Lock()
        self._settings = InferenceSettings()
        self._metrics_lock = threading.Lock()
        self._metrics = InferenceMetrics()
        self._started_at = 0.0
        self._stop_scan_lock = threading.Lock()
        self._motion_reference: np.ndarray | None = None
        self._stable_frame_count = 0
        self._scan_epoch = 0
        self._captured_epoch = -1
        self._best_stable_frame: np.ndarray | None = None
        self._best_sharpness = float("-inf")
        self._captured_output: RealtimeInferenceOutput | None = None

    @property
    def metrics(self) -> InferenceMetrics:
        with self._metrics_lock:
            return self._metrics

    def configure(
        self,
        *,
        product_id: str | None,
        roi: NormalizedRect | None,
        counting_line: NormalizedLine | None,
        direction: CountingDirection,
        inspection_mode: str = "on_stop",
        stable_frames_required: int = 6,
        motion_threshold: float = 2.0,
    ) -> None:
        mode = inspection_mode if inspection_mode in {"on_stop", "realtime"} else "on_stop"
        with self._settings_lock:
            self._settings = InferenceSettings(
                product_id,
                roi,
                counting_line,
                direction,
                mode,
                max(1, int(stable_frames_required)),
                max(0.0, float(motion_threshold)),
            )
        self._reset_stop_and_scan_state()

    def update_roi(self, roi: NormalizedRect | None) -> None:
        with self._settings_lock:
            self._settings = replace(self._settings, roi=roi)
        self._reset_stop_and_scan_state()

    def update_product_id(self, product_id: str | None) -> None:
        with self._settings_lock:
            self._settings = replace(self._settings, product_id=product_id)
        self._reset_stop_and_scan_state()

    def update_counting_line(self, line: NormalizedLine | None) -> None:
        with self._settings_lock:
            self._settings = replace(self._settings, counting_line=line)

    def update_direction(self, direction: CountingDirection) -> None:
        with self._settings_lock:
            self._settings = replace(self._settings, direction=direction)

    def start(self, priority: QThread.Priority = QThread.Priority.InheritPriority) -> None:
        if self.isRunning():
            return
        self._stop_requested.clear()
        self._started_at = time.monotonic()
        with self._metrics_lock:
            self._metrics = InferenceMetrics()
        self._reset_stop_and_scan_state()
        super().start(priority)

    def stop(self, timeout_ms: int = 3_000) -> bool:
        self._stop_requested.set(); self.requestInterruption()
        if not self.isRunning():
            return True
        if self.isCurrentThread():
            return False
        return self.wait(timeout_ms)

    def run(self) -> None:
        while not self._stop_requested.is_set() and not self.isInterruptionRequested():
            packet = self.frames.get_latest(timeout=0.25)
            if packet is None:
                if self.frames.metrics.closed:
                    break
                continue
            started = time.monotonic()
            try:
                output = self._process(packet.frame)
                self.inference_ready.emit(output)
                self._record_metrics(time.monotonic() - started)
            except Exception as error:  # a bad frame must not stop production capture
                LOGGER.exception("Lỗi inference realtime")
                self._record_error()
                self.inference_error.emit(f"Lỗi inference: {error}")

    def _process(self, frame: np.ndarray) -> RealtimeInferenceOutput:
        with self._settings_lock:
            settings = self._settings
        if settings.inspection_mode == "on_stop":
            return self._process_stop_and_scan(frame, settings)
        return self._inspect_frame(frame, settings)

    def _inspect_frame(self, frame: np.ndarray, settings: InferenceSettings) -> RealtimeInferenceOutput:
        roi_frame, offset = self._apply_roi(frame, settings.roi)
        # Fallback mới chỉ chấp nhận toàn bộ chuỗi run màu trong Product
        # Profile, nằm trên cùng một trục và phải PASS validator. Vì vậy nó
        # vẫn an toàn khi ROI đang rộng (trường hợp thao tác camera phổ biến),
        # không còn bỏ lỡ vỉ rõ nét chỉ vì người vận hành chưa vẽ ROI sát.
        allow_color_guided = True
        result = self.pipeline.inspect_frame(
            roi_frame,
            settings.product_id,
            allow_profile_color_fallback=allow_color_guided,
            roi=settings.roi,
        )
        annotated = frame.copy()
        x_offset, y_offset = offset
        annotated[y_offset:y_offset + result.annotated_frame.shape[0],
                  x_offset:x_offset + result.annotated_frame.shape[1]] = result.annotated_frame
        # Giám sát ROI không dùng tracker/đường đếm nữa: cùng một lúc có thể
        # có nhiều vỉ, và chỉ cần một vỉ NG là báo động phải bật ngay ở frame
        # hiện tại. Vì vậy không cộng số lượt hay xuất Excel theo lần cắt line.
        failures = [item for item in result.detections if item.validation.status is ProductStatus.FAIL]
        visible_products = len(result.detections)
        fault_products = len(failures)
        if failures:
            reasons = "; ".join(item.validation.detail for item in failures[:3] if item.validation.detail)
            alert_detail = (
                f"CẢNH BÁO NG: phát hiện {fault_products} vỉ lỗi trong ROI"
                + (f" — {reasons}" if reasons else "")
            )
        elif visible_products:
            alert_detail = f"ROI đang có {visible_products} vỉ; chưa phát hiện vỉ lỗi xác thực."
        else:
            alert_detail = result.detail
        colors = tuple(self.color_display(slot.color) for item in result.detections for slot in item.slots)
        return RealtimeInferenceOutput(
            status=result.validation.status.value,
            product_id=result.validation.product_id,
            detected_colors=colors,
            confidence=result.validation.confidence,
            count_delta=0,
            detected_delta=0,
            pass_delta=0,
            fail_delta=0,
            unknown_delta=0,
            detail=alert_detail,
            annotated_frame=self._to_qimage(annotated),
            fps=self.metrics.average_fps,
            visible_products=visible_products,
            fault_products=fault_products,
            alert_active=bool(failures),
            alert_detail=alert_detail,
        )

    def _process_stop_and_scan(
        self,
        frame: np.ndarray,
        settings: InferenceSettings,
    ) -> RealtimeInferenceOutput:
        """Chỉ kiểm tra sau khi ROI đứng yên, một lần cho mỗi nhịp dừng."""
        roi_frame, _ = self._apply_roi(frame, settings.roi)
        motion_frame = self._motion_image(roi_frame)
        sharpness = self._sharpness_score(roi_frame)

        with self._stop_scan_lock:
            if self._motion_reference is None:
                self._motion_reference = motion_frame
                self._stable_frame_count = 1
                self._best_stable_frame = frame.copy()
                self._best_sharpness = sharpness
                return self._passive_output(
                    frame,
                    "Đang chờ băng tải dừng ổn định 1/"
                    f"{settings.stable_frames_required} khung…",
                )

            motion_score = float(cv2.absdiff(motion_frame, self._motion_reference).mean())
            self._motion_reference = motion_frame
            if motion_score > settings.motion_threshold:
                # Một chuyển động mới bắt đầu một chu kỳ scan mới. Không gọi
                # pipeline ở đây vì frame có thể mờ hoặc chỉ hiện một phần vỉ.
                self._scan_epoch += 1
                self._stable_frame_count = 0
                self._best_stable_frame = None
                self._best_sharpness = float("-inf")
                self._captured_output = None
                return self._passive_output(
                    frame,
                    "Băng tải đang chạy — chờ dừng để chụp ảnh kiểm tra.",
                )

            self._stable_frame_count += 1
            if sharpness >= self._best_sharpness:
                self._best_stable_frame = frame.copy()
                self._best_sharpness = sharpness

            if self._stable_frame_count < settings.stable_frames_required:
                return self._passive_output(
                    frame,
                    "Đang chờ ảnh ổn định "
                    f"{self._stable_frame_count}/{settings.stable_frames_required} khung…",
                )

            if self._captured_epoch != self._scan_epoch:
                captured_frame = self._best_stable_frame if self._best_stable_frame is not None else frame
                output = self._inspect_frame(captured_frame, settings)
                prefix = (
                    "Đã chụp khung ổn định "
                    f"({self._stable_frame_count} khung, nét nhất) — "
                )
                detail = prefix + output.detail
                self._captured_output = replace(output, detail=detail, alert_detail=detail)
                self._captured_epoch = self._scan_epoch
                return self._captured_output

            if self._captured_output is not None:
                detail = "Đã quét khung ổn định; chờ băng tải chạy sang nhịp kế tiếp."
                return replace(
                    self._captured_output,
                    fps=self.metrics.average_fps,
                    detail=detail,
                    alert_detail=detail if not self._captured_output.alert_active else self._captured_output.alert_detail,
                )

        return self._passive_output(frame, "Đang chờ băng tải dừng để chụp ảnh kiểm tra.")

    def _reset_stop_and_scan_state(self) -> None:
        with self._stop_scan_lock:
            self._motion_reference = None
            self._stable_frame_count = 0
            self._scan_epoch = 0
            self._captured_epoch = -1
            self._best_stable_frame = None
            self._best_sharpness = float("-inf")
            self._captured_output = None

    def _passive_output(self, frame: np.ndarray, detail: str) -> RealtimeInferenceOutput:
        return RealtimeInferenceOutput(
            status=ProductStatus.UNKNOWN.value,
            product_id=None,
            detected_colors=(),
            confidence=0.0,
            count_delta=0,
            detected_delta=0,
            pass_delta=0,
            fail_delta=0,
            unknown_delta=0,
            detail=detail,
            annotated_frame=self._to_qimage(frame.copy()),
            fps=self.metrics.average_fps,
            visible_products=0,
            fault_products=0,
            alert_active=False,
            alert_detail=detail,
        )

    @staticmethod
    def _motion_image(frame: np.ndarray) -> np.ndarray:
        """Ảnh xám nhỏ, làm mờ nhẹ để đánh giá chuyển động ổn định."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        max_side = max(height, width)
        if max_side > 240:
            scale = 240.0 / max_side
            gray = cv2.resize(
                gray,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return cv2.GaussianBlur(gray, (5, 5), 0)

    @staticmethod
    def _sharpness_score(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _audit_event(self, frame: np.ndarray, track_id: int, result: object) -> None:
        try:
            if self.repository is not None:
                self.repository.save(str(track_id), result)  # type: ignore[arg-type]
            if self.artifact_store is not None:
                self.artifact_store.save_non_pass(frame, track_id, result)  # type: ignore[arg-type]
        except Exception:
            LOGGER.exception("Không lưu được audit event track=%s", track_id)

    def _export_event(self, track_id: int, result: object, counted: bool) -> None:
        """Excel là bản tra cứu vận hành, lỗi xuất không được dừng camera."""
        if self.daily_excel_exporter is None:
            return
        try:
            self.daily_excel_exporter.record_event(
                track_id=track_id,
                result=result,  # type: ignore[arg-type]
                counted=counted,
                color_display=self.color_display,
            )
        except Exception:
            LOGGER.exception("Không xuất được Excel cho event track=%s", track_id)

    def _record_metrics(self, elapsed_seconds: float) -> None:
        with self._metrics_lock:
            count = self._metrics.frames_processed + 1
            elapsed_total = max(time.monotonic() - self._started_at, 1e-6)
            self._metrics = InferenceMetrics(
                frames_processed=count,
                latest_latency_ms=elapsed_seconds * 1_000,
                average_fps=count / elapsed_total,
                dropped_frames=self.frames.metrics.dropped,
                errors=self._metrics.errors,
            )
        self.metrics_updated.emit(self.metrics)

    def _record_error(self) -> None:
        with self._metrics_lock:
            self._metrics = replace(self._metrics, errors=self._metrics.errors + 1)

    @staticmethod
    def _apply_roi(frame: np.ndarray, roi: NormalizedRect | None) -> tuple[np.ndarray, tuple[int, int]]:
        if roi is None:
            return frame, (0, 0)
        height, width = frame.shape[:2]
        x = max(0, min(width - 1, round(roi[0] * width)))
        y = max(0, min(height - 1, round(roi[1] * height)))
        right = max(x + 1, min(width, round((roi[0] + roi[2]) * width)))
        bottom = max(y + 1, min(height, round((roi[1] + roi[3]) * height)))
        return frame[y:bottom, x:right], (x, y)

    @staticmethod
    def _offset_box(box: tuple[int, int, int, int], offset: tuple[int, int]) -> tuple[int, int, int, int]:
        return box[0] + offset[0], box[1] + offset[1], box[2], box[3]

    @staticmethod
    def _line_position(line: NormalizedLine, frame_shape: tuple[int, ...], direction: CountingDirection) -> float | None:
        start, end = line
        if direction in {CountingDirection.LEFT_TO_RIGHT, CountingDirection.RIGHT_TO_LEFT}:
            if abs(start[0] - end[0]) > 0.05:
                return None
            return ((start[0] + end[0]) / 2) * frame_shape[1]
        if abs(start[1] - end[1]) > 0.05:
            return None
        return ((start[1] + end[1]) / 2) * frame_shape[0]

    @staticmethod
    def _to_qimage(frame_bgr: np.ndarray) -> QImage:
        height, width = frame_bgr.shape[:2]
        image = QImage(frame_bgr.data, width, height, frame_bgr.strides[0], QImage.Format.Format_BGR888)
        return image.copy()
