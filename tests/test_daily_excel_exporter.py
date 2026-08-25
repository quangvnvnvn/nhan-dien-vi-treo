import json
import os
import subprocess
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from core.models import FailureReason, ProductStatus, SlotObservation, ValidationResult
from services.daily_excel_exporter import DailyExcelExporter


class DailyExcelExporterTests(unittest.TestCase):
    def test_events_share_one_workbook_per_local_day(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_runner(_args: list[str], **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        moment = datetime(2026, 8, 20, 9, 30, 15)
        result = ValidationResult(
            "1", ProductStatus.PASS, None,
            observations=[SlotObservation(1, True, "green", 0.9, 0.9)],
            confidence=0.91,
            detail="Đạt tất cả quy tắc.",
        )
        with tempfile.TemporaryDirectory() as folder:
            exporter = DailyExcelExporter(
                Path(folder), now_provider=lambda: moment, node_executable="node",
                script_path=Path(folder) / "writer.mjs", process_runner=fake_runner,
            )
            first_path = exporter.record_event(
                track_id=7, result=result, counted=True, color_display=lambda color: "Xanh lá" if color == "green" else "--",
            )
            second_path = exporter.record_event(
                track_id=8, result=result, counted=False, color_display=lambda color: color or "--",
            )

            self.assertEqual(first_path, Path(folder) / "2026-08-20.xlsx")
            self.assertEqual(second_path, first_path)
            records = [json.loads(line) for line in first_path.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([record["track_id"] for record in records], ["7", "8"])
            self.assertEqual(records[0]["colors"], "Xanh lá")
            second_payload = json.loads(str(calls[-1]["input"]))
            self.assertEqual(len(second_payload["records"]), 2)
            self.assertEqual(second_payload["records"][0]["count_increment"], 1)
            self.assertEqual(second_payload["records"][1]["count_increment"], 0)

    def test_excel_process_is_hidden_on_windows(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_runner(_args: list[str], **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        result = ValidationResult("1", ProductStatus.PASS, None)
        with tempfile.TemporaryDirectory() as folder:
            exporter = DailyExcelExporter(
                Path(folder), node_executable="node", script_path=Path(folder) / "writer.mjs", process_runner=fake_runner,
            )
            exporter.record_event(track_id=1, result=result, counted=True, color_display=lambda color: color or "--")

        if os.name == "nt":
            self.assertEqual(calls[0]["creationflags"], subprocess.CREATE_NO_WINDOW)
            self.assertEqual(calls[0]["startupinfo"].wShowWindow, subprocess.SW_HIDE)  # type: ignore[union-attr]
        else:
            self.assertNotIn("creationflags", calls[0])

    def test_result_reason_and_confidence_are_exported(self) -> None:
        result = ValidationResult(
            "1234", ProductStatus.FAIL, FailureReason.WRONG_COLOR, confidence=1.2,
        )
        record = DailyExcelExporter._make_record(
            4, result, False, lambda color: color or "--", datetime(2026, 8, 20, 12, 0, 0),
        )
        self.assertEqual(record["reason"], "WRONG_COLOR")
        self.assertEqual(record["confidence"], 1.0)
        self.assertEqual(record["count_increment"], 0)
