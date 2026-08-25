"""Kiểm tra chuyển nguồn video từ giao diện sang CameraThread."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PySide6.QtCore import QObject, Signal

from core.colors import ColorCatalog
from services.realtime_controller import RealtimeController
from training.product_manager import ProductManager


class _FakePage(QObject):
    start_requested = Signal(object)
    stop_requested = Signal()
    roi_changed = Signal(object)
    product_profile_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.attached_camera: object | None = None
        self.runtime_errors: list[str] = []

    def attach_camera_thread(self, camera: object, *, display_raw_frames: bool) -> None:
        self.attached_camera = camera
        self.display_raw_frames = display_raw_frames

    def apply_inference(self, _output: object) -> None:
        pass

    def show_runtime_error(self, message: str) -> None:
        self.runtime_errors.append(message)

    def set_export_directory(self, _directory: Path) -> None:
        pass


class _FakeCamera(QObject):
    frame_ready = Signal(object)
    camera_error = Signal(str)

    def __init__(self, source: object, **_kwargs: object) -> None:
        super().__init__()
        self.source = source
        self.latest_frames = object()
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> bool:
        self.stopped = True
        return True


class _FakeInference(QObject):
    inference_ready = Signal(object)
    inference_error = Signal(str)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        super().__init__()
        self.configure_args: dict[str, object] = {}
        self.started = False
        self.stopped = False

    def configure(self, **kwargs: object) -> None:
        self.configure_args = kwargs

    def update_roi(self, _roi: object) -> None:
        pass

    def update_product_id(self, _product_id: object) -> None:
        pass

    def start(self) -> None:
        self.started = True

    def stop(self) -> bool:
        self.stopped = True
        return True


class _FakeRepository:
    def __init__(self, _path: Path) -> None:
        pass

    def initialize(self) -> None:
        pass


class _FakeArtifacts:
    def __init__(self, _path: Path) -> None:
        pass


class _FakeExcelExporter:
    def __init__(self, path: Path, **_kwargs: object) -> None:
        self.output_directory = path


class RealtimeVideoSourceTests(unittest.TestCase):
    def test_existing_video_path_is_preserved(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "inspection.mp4"
            source.touch()

            result = RealtimeController._source_from_request(
                SimpleNamespace(source=str(source), source_type="video")
            )

        self.assertEqual(result, str(source))

    def test_missing_video_path_is_rejected_before_camera_open(self) -> None:
        with self.assertRaisesRegex(ValueError, "Không tìm thấy tệp video"):
            RealtimeController._source_from_request(
                SimpleNamespace(source="C:/not-found/inspection.mp4", source_type="video")
            )

    def test_controller_starts_capture_after_attaching_it_to_page(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory, \
             patch("services.realtime_controller.CameraThread", _FakeCamera), \
             patch("services.realtime_controller.RealtimeInferenceThread", _FakeInference), \
             patch("services.realtime_controller.ResultRepository", _FakeRepository), \
             patch("services.realtime_controller.ResultArtifactStore", _FakeArtifacts), \
             patch("services.realtime_controller.DailyExcelExporter", _FakeExcelExporter):
            page = _FakePage()
            controller = RealtimeController(
                page,
                ProductManager(root / "config" / "products.yaml"),
                ColorCatalog({}),
                {},
                Path(directory),
                {
                    "camera": {"fps": 30},
                    "stop_and_scan": {"stable_frames_required": 6, "motion_threshold": 2.0},
                },
            )
            controller.start(
                SimpleNamespace(
                    source="0",
                    source_type="usb",
                    expected_product_id="Vim-GT",
                    roi=None,
                    counting_direction="top_to_bottom",
                    inspection_mode="on_stop",
                )
            )

            assert controller.camera is not None
            assert controller.inference is not None
            self.assertIs(page.attached_camera, controller.camera)
            self.assertTrue(controller.camera.started)
            self.assertTrue(controller.inference.started)
            self.assertEqual(controller.inference.configure_args["inspection_mode"], "on_stop")
            self.assertEqual(controller.inference.configure_args["stable_frames_required"], 6)
            self.assertEqual(controller.inference.configure_args["motion_threshold"], 2.0)
            self.assertEqual(page.runtime_errors, [])
            controller.stop()
            self.assertTrue(controller.camera is None)
            self.assertTrue(controller.inference is None)


if __name__ == "__main__":
    unittest.main()
