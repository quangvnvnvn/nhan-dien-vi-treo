"""Ghi sự kiện kiểm tra realtime thành một tệp Excel cho từng ngày.

Sự kiện được lưu bền vững vào JSONL trước, sau đó tệp Excel của ngày đó được
tạo lại từ toàn bộ sự kiện trong ngày. Cách này giúp file Excel không bị hỏng
khi ứng dụng bị đóng giữa chừng và không phụ thuộc vào một tiến trình Excel
đang mở.
"""
from __future__ import annotations

from datetime import date, datetime
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Callable, Protocol

from core.models import ValidationResult

LOGGER = logging.getLogger(__name__)


class ProcessRunner(Protocol):
    def __call__(self, args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]: ...


class DailyExcelExporter:
    """Xuất ``data/exports/YYYY-MM-DD.xlsx`` cho các sự kiện đã qua đường đếm."""

    def __init__(
        self,
        output_directory: Path,
        *,
        now_provider: Callable[[], datetime] = datetime.now,
        node_executable: str | Path | None = None,
        script_path: Path | None = None,
        process_runner: ProcessRunner = subprocess.run,
        asynchronous: bool = False,
    ) -> None:
        self.output_directory = Path(output_directory)
        self._now_provider = now_provider
        self._node_executable = str(node_executable) if node_executable is not None else None
        self._script_path = script_path or Path(__file__).with_name("daily_excel_exporter.mjs")
        self._process_runner = process_runner
        self._asynchronous = asynchronous
        self._lock = threading.Lock()
        self._active_days: set[date] = set()
        self._pending_days: set[date] = set()

    def workbook_path_for(self, local_time: datetime) -> Path:
        return self.output_directory / f"{local_time:%Y-%m-%d}.xlsx"

    def records_for_day(self, value: date) -> list[dict[str, object]]:
        """Trả về các event đã ghi trong ngày để giao diện tra cứu hiển thị."""
        journal_path = self.output_directory / f"{value:%Y-%m-%d}.jsonl"
        return self._read_journal(journal_path) if journal_path.is_file() else []

    def available_days(self) -> list[date]:
        """Các ngày có dữ liệu, mới nhất đứng đầu."""
        if not self.output_directory.is_dir():
            return []
        values: list[date] = []
        for journal_path in self.output_directory.glob("????-??-??.jsonl"):
            try:
                values.append(datetime.strptime(journal_path.stem, "%Y-%m-%d").date())
            except ValueError:
                continue
        return sorted(set(values), reverse=True)

    def record_event(
        self,
        *,
        track_id: int | str,
        result: ValidationResult,
        counted: bool,
        color_display: Callable[[str | None], str],
    ) -> Path:
        """Lưu một event và cập nhật workbook của đúng ngày local.

        Chỉ các event đã cắt đường đếm được gọi vào đây, vì vậy số liệu trong
        Excel khớp với bảng realtime và không bị lặp theo từng frame.
        """
        recorded_at = self._now_provider()
        record = self._make_record(track_id, result, counted, color_display, recorded_at)
        workbook_path = self.workbook_path_for(recorded_at)
        journal_path = workbook_path.with_suffix(".jsonl")
        with self._lock:
            self.output_directory.mkdir(parents=True, exist_ok=True)
            with journal_path.open("a", encoding="utf-8", newline="\n") as journal:
                journal.write(json.dumps(record, ensure_ascii=False) + "\n")
            if self._asynchronous:
                self._schedule_background_build(workbook_path, recorded_at)
            else:
                records = self._read_journal(journal_path)
                self._build_workbook(workbook_path, recorded_at, records)
        return workbook_path

    def _schedule_background_build(self, workbook_path: Path, recorded_at: datetime) -> None:
        """Không chặn tracker/camera khi Node đang tạo workbook.

        Mỗi ngày chỉ có một worker xuất Excel. Nếu có event mới trong lúc worker
        đang chạy, nó sẽ tạo lại file thêm một lần với toàn bộ nhật ký mới nhất.
        """
        day = recorded_at.date()
        if day in self._active_days:
            self._pending_days.add(day)
            return
        self._active_days.add(day)
        worker = threading.Thread(
            target=self._build_in_background,
            args=(workbook_path, recorded_at),
            name=f"excel-export-{day.isoformat()}",
            daemon=True,
        )
        worker.start()

    def _build_in_background(self, workbook_path: Path, recorded_at: datetime) -> None:
        day = recorded_at.date()
        try:
            while True:
                with self._lock:
                    records = self._read_journal(workbook_path.with_suffix(".jsonl"))
                    self._pending_days.discard(day)
                self._build_workbook(workbook_path, recorded_at, records)
                with self._lock:
                    if day not in self._pending_days:
                        self._active_days.discard(day)
                        return
        except Exception:
            with self._lock:
                self._active_days.discard(day)
            LOGGER.exception("Không tạo được Excel nền cho ngày %s", day.isoformat())

    @staticmethod
    def _make_record(
        track_id: int | str,
        result: ValidationResult,
        counted: bool,
        color_display: Callable[[str | None], str],
        recorded_at: datetime,
    ) -> dict[str, object]:
        colors = " - ".join(color_display(slot.color) for slot in result.observations) or "--"
        return {
            "timestamp": recorded_at.isoformat(timespec="seconds"),
            "time": recorded_at.strftime("%H:%M:%S"),
            "product_id": result.product_id or "--",
            "status": result.status.value,
            "colors": colors,
            "confidence": round(max(0.0, min(1.0, float(result.confidence))), 4),
            "count_increment": 1 if counted else 0,
            "track_id": str(track_id),
            "reason": result.reason.value if result.reason is not None else "--",
            "detail": result.detail or "--",
        }

    @staticmethod
    def _read_journal(journal_path: Path) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        with journal_path.open("r", encoding="utf-8") as journal:
            for line in journal:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    LOGGER.warning("Bỏ qua dòng nhật ký Excel lỗi: %s", journal_path)
                    continue
                if isinstance(value, dict):
                    records.append(value)
        return records

    def _build_workbook(
        self, workbook_path: Path, recorded_at: datetime, records: list[dict[str, object]]
    ) -> None:
        node = self._node_executable or self._resolve_node_executable()
        payload = {
            "date": recorded_at.strftime("%Y-%m-%d"),
            "generated_at": self._now_provider().isoformat(timespec="seconds"),
            "output_path": str(workbook_path),
            "records": records,
        }
        completed = self._process_runner(
            [node, str(self._script_path)],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            cwd=str(self._script_path.parent),
            timeout=45,
            check=False,
            **self._hidden_process_options(),
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "Không rõ lỗi").strip()
            raise RuntimeError(f"Không tạo được Excel ngày {recorded_at:%Y-%m-%d}: {detail}")
        if not workbook_path.is_file() and self._process_runner is subprocess.run:
            raise RuntimeError("Trình xuất Excel không tạo ra tệp kết quả.")

    @staticmethod
    def _hidden_process_options() -> dict[str, object]:
        """Node là console app; tuyệt đối không hiện cửa sổ CMD trên Windows."""
        if os.name != "nt":
            return {}
        startup_info = subprocess.STARTUPINFO()
        startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup_info.wShowWindow = subprocess.SW_HIDE
        return {
            "creationflags": subprocess.CREATE_NO_WINDOW,
            "startupinfo": startup_info,
        }

    @staticmethod
    def _resolve_node_executable() -> str:
        override = os.environ.get("VI_EXCEL_NODE")
        if override:
            return override
        codex_node = (
            Path.home()
            / ".cache"
            / "codex-runtimes"
            / "codex-primary-runtime"
            / "dependencies"
            / "node"
            / "bin"
            / "node.exe"
        )
        if codex_node.is_file():
            return str(codex_node)
        system_node = shutil.which("node")
        if system_node:
            return system_node
        raise RuntimeError("Không tìm thấy Node.js để tạo tệp Excel.")
