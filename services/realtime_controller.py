"""Kết nối UI realtime với camera, worker inference, tracker và audit local."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject

from ai.color_classifier import ColorClassifier
from ai.product_detector import GeometricProductDetector
from ai.realtime_pipeline import FrameInspectionPipeline
from ai.tracker import CentroidTracker, CountingDirection
from camera.camera_thread import CameraThread
from core.colors import ColorCatalog
from database.database import ResultRepository
from services.daily_excel_exporter import DailyExcelExporter
from services.realtime_inference_thread import RealtimeInferenceThread
from services.result_artifacts import ResultArtifactStore
from training.product_manager import ProductManager

LOGGER = logging.getLogger(__name__)


class RealtimeController(QObject):
    """Tầng điều phối không để camera/inference chặn event loop của PySide6."""

    def __init__(self, page: Any, manager: ProductManager, colors: ColorCatalog,
                 color_profiles: dict[str, Any], root: Path, config: dict[str, Any]) -> None:
        super().__init__(page)
        self.page = page
        self.manager = manager
        self.colors = colors
        self.root = root
        self.config = config
        detection = config.get("detection", {})
        detector = GeometricProductDetector(
            minimum_slot_radius_px=int(detection.get("min_slot_radius_px", 12)),
            minimum_slot_radius_fraction=float(detection.get("min_slot_radius_fraction", 0.032)),
            maximum_slot_radius_fraction=float(detection.get("max_slot_radius_fraction", 0.10)),
            minimum_slot_distance_fraction=float(detection.get("min_slot_distance_fraction", 0.07)),
            hough_param1=float(detection.get("hough_param1", 100)),
            hough_param2=float(detection.get("hough_param2", 36)),
            maximum_geometry_score=float(detection.get("max_geometry_score", 0.42)),
            candidate_limit=int(detection.get("candidate_limit", 9)),
            processing_max_dimension=(
                int(detection["processing_max_dimension"])
                if int(detection.get("processing_max_dimension", 0)) > 0
                else None
            ),
            enable_chroma_contour_fallback=bool(detection.get("enable_chroma_contour_fallback", True)),
            chroma_saturation_min=int(detection.get("chroma_saturation_min", 80)),
            chroma_value_min=int(detection.get("chroma_value_min", 45)),
            minimum_ellipse_axis_ratio=float(detection.get("minimum_ellipse_axis_ratio", 0.50)),
            contour_min_radius_factor=float(detection.get("contour_min_radius_factor", 0.65)),
            enable_clahe_fallback=bool(detection.get("enable_clahe_fallback", True)),
            clahe_clip_limit=float(detection.get("clahe_clip_limit", 2.0)),
            clahe_tile_grid_size=int(detection.get("clahe_tile_grid_size", 8)),
            maximum_projective_spacing_error=float(detection.get("max_projective_spacing_error", 0.065)),
            maximum_line_deviation_ratio=float(detection.get("max_line_deviation_ratio", 0.70)),
            minimum_spacing_to_radius=float(detection.get("minimum_spacing_to_radius", 1.20)),
            enable_side_view_fallback=bool(detection.get("enable_side_view_fallback", True)),
            side_view_maximum_geometry_score=float(detection.get("side_view_max_geometry_score", 0.30)),
            side_view_minimum_spacing_to_radius=float(detection.get("side_view_min_spacing_to_radius", 0.80)),
            side_view_sample_radius_fraction=float(detection.get("side_view_sample_radius_fraction", 0.26)),
            side_view_refine_cross_axis_fraction=float(detection.get("side_view_refine_cross_axis_fraction", 0.70)),
            side_view_refine_along_axis_fraction=float(detection.get("side_view_refine_along_axis_fraction", 0.22)),
            side_view_chroma_saturation_min=int(detection.get("side_view_chroma_saturation_min", 50)),
            side_view_chroma_value_min=int(detection.get("side_view_chroma_value_min", 45)),
            side_view_minimum_chroma_fraction=float(detection.get("side_view_min_chroma_fraction", 0.20)),
        )
        self.pipeline = FrameInspectionPipeline(
            manager,
            ColorClassifier(color_profiles),
            color_normalizer=colors.normalize,
            detector=detector,
            minimum_detector_confidence=float(detection.get("confidence", 0.85)),
            allow_multiple_candidates=bool(detection.get("allow_multiple_candidates", True)),
            maximum_monitor_candidates=int(detection.get("maximum_monitor_candidates", 12)),
            enable_color_guided_fallback=bool(detection.get("enable_color_guided_fallback", True)),
            color_guided_max_dimension=int(detection.get("color_guided_max_dimension", 480)),
            profile_component_limit_per_color=int(detection.get("profile_component_limit_per_color", 6)),
            strict_position_color_validation=bool(detection.get("strict_position_color_validation", True)),
        )
        self._default_candidate_limit = detector.candidate_limit
        self._fixed_pose_candidate_limit = max(
            len(manager.list_profiles()[0].slots) if manager.list_profiles() else 5,
            int(detection.get("fixed_pose_candidate_limit", 6)),
        )
        tracking = config.get("tracking", {})
        self._stop_and_scan = config.get("stop_and_scan", {})
        self.tracker = CentroidTracker(
            max_age=int(tracking.get("max_age", 40)),
            min_validation_hits=int(tracking.get("min_hits", 1)),
            distance_threshold=float(tracking.get("distance_threshold", 320)),
            first_seen_crossing_margin_fraction=float(
                tracking.get("first_seen_crossing_margin_fraction", 0.75)
            ),
        )
        self.repository = ResultRepository(root / "data" / "database" / "production.db")
        self.repository.initialize()
        self.artifacts = ResultArtifactStore(root / "data" / "results")
        # Xuất Excel ở thread nền: camera không chờ Node và không hiện cửa sổ CMD.
        self.daily_excel_exporter = DailyExcelExporter(root / "data" / "exports", asynchronous=True)
        set_export_directory = getattr(page, "set_export_directory", None)
        if callable(set_export_directory):
            set_export_directory(self.daily_excel_exporter.output_directory)
        self.camera: CameraThread | None = None
        self.inference: RealtimeInferenceThread | None = None
        page.start_requested.connect(self.start)
        page.stop_requested.connect(self.stop)

    def start(self, request: object) -> None:
        """Tạo một session mới; camera/AI chạy ở hai QThread khác nhau."""
        self.stop()
        try:
            source = self._source_from_request(request)
            self.pipeline.strict_position_color_validation = bool(
                getattr(
                    request,
                    "position_locked_color",
                    self.config.get("detection", {}).get("strict_position_color_validation", True),
                )
            )
            # Góc camera cố định không cần thử hàng trăm tổ hợp phản xạ
            # Hough. Giữ 6 candidate (5 slot thật + một phản xạ) để giảm độ
            # trễ mỗi nhịp dừng, còn fallback màu theo Profile vẫn là lớp cứu
            # cho vỉ nhỏ/nhựa trong.
            self.pipeline.detector.candidate_limit = (
                self._fixed_pose_candidate_limit
                if self.pipeline.strict_position_color_validation
                else self._default_candidate_limit
            )
            fps = float(self.config.get("camera", {}).get("fps", 30))
            self.tracker.reset()
            self.camera = CameraThread(source, target_fps=fps, parent=self)
            self.inference = RealtimeInferenceThread(
                self.camera.latest_frames,
                self.pipeline,
                self.tracker,
                repository=self.repository,
                artifact_store=self.artifacts,
                daily_excel_exporter=self.daily_excel_exporter,
                color_display=self.colors.display,
                parent=self,
            )
            direction = self._direction_from_request(request)
            self.inference.configure(
                product_id=getattr(request, "expected_product_id", None),
                roi=getattr(request, "roi", None),
                # Realtime hiện là chế độ giám sát lỗi trong ROI, không còn
                # phụ thuộc vào đường đếm hoặc hướng chạy của vỉ.
                counting_line=None,
                direction=direction,
                inspection_mode=str(getattr(request, "inspection_mode", "on_stop")),
                stable_frames_required=int(self._stop_and_scan.get("stable_frames_required", 6)),
                motion_threshold=float(self._stop_and_scan.get("motion_threshold", 2.0)),
            )
            self.inference.inference_ready.connect(self.page.apply_inference)
            self.inference.inference_error.connect(self.page.show_runtime_error)
            self.page.roi_changed.connect(self.inference.update_roi)
            self.page.product_profile_changed.connect(self.inference.update_product_id)
            # Chỉ worker inference phát QImage annotate; không render ndarray raw
            # trong UI thread và không chạy model trên UI thread.
            self.page.attach_camera_thread(self.camera, display_raw_frames=False)
            # ``start_requested`` có thể được Qt xếp hàng thay vì chạy đồng bộ
            # với click trên giao diện. Vì vậy controller phải tự chạy capture
            # sau khi đã gắn worker, không chờ RealtimePage gọi ``start`` lần
            # hai. Nếu callback chạy ngay, RealtimePage chỉ thấy isRunning() và
            # không tạo phiên camera trùng lặp.
            self.camera.start()
            self.inference.start()
        except Exception as error:
            LOGGER.exception("Không tạo được realtime session")
            # Không để lại worker dở dang nếu một bước gắn signal/camera lỗi.
            self.stop()
            self.page.show_runtime_error(f"Không thể khởi động camera: {error}")

    def stop(self) -> None:
        """Dừng worker trước rồi dừng capture, không dùng terminate()."""
        if self.inference is not None:
            self.inference.stop()
            self._disconnect_worker_signals(self.inference)
            self.inference = None
        if self.camera is not None:
            self.camera.stop()
            self.camera = None

    def shutdown(self) -> None:
        self.stop()

    def _update_direction(self, value: str) -> None:
        if self.inference is not None:
            try:
                self.inference.update_direction(CountingDirection(value))
            except ValueError:
                self.page.show_runtime_error(f"Chiều đếm không hợp lệ: {value}")

    def _disconnect_worker_signals(self, worker: RealtimeInferenceThread) -> None:
        for signal, target in (
            (worker.inference_ready, self.page.apply_inference),
            (worker.inference_error, self.page.show_runtime_error),
            (self.page.roi_changed, worker.update_roi),
            (self.page.product_profile_changed, worker.update_product_id),
        ):
            try:
                signal.disconnect(target)
            except (RuntimeError, TypeError):
                pass

    @staticmethod
    def _source_from_request(request: object) -> int | str:
        source = str(getattr(request, "source", "")).strip()
        source_type = str(getattr(request, "source_type", "usb"))
        if source_type == "usb":
            if not source.isdecimal():
                raise ValueError("Camera USB phải là số, ví dụ 0")
            return int(source)
        if source_type == "video":
            video_path = Path(source).expanduser()
            if not video_path.is_file():
                raise ValueError("Không tìm thấy tệp video đã chọn")
            return str(video_path)
        if not source:
            raise ValueError("Nguồn RTSP không được để trống")
        return source

    @staticmethod
    def _direction_from_request(request: object) -> CountingDirection:
        try:
            return CountingDirection(str(getattr(request, "counting_direction", "top_to_bottom")))
        except ValueError as error:
            raise ValueError("Chiều đếm không hợp lệ") from error
