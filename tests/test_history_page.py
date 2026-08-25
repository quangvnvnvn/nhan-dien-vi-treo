import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from core.models import ProductStatus, SlotObservation, ValidationResult
from services.daily_excel_exporter import DailyExcelExporter
from ui.history_page import HistoryPage


class HistoryPageTests(unittest.TestCase):
    def test_history_reads_same_daily_journal_as_excel_exporter(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            exporter = DailyExcelExporter(Path(folder))
            journal = Path(folder) / "2026-08-20.jsonl"
            journal.write_text(
                '{"time":"10:30:00","product_id":"1","status":"PASS","colors":"Xanh lá","confidence":0.9,"count_increment":1,"track_id":"7","reason":"--","detail":"Đạt"}\n',
                encoding="utf-8",
            )
            self.assertEqual(exporter.available_days()[0].isoformat(), "2026-08-20")
            records = exporter.records_for_day(datetime(2026, 8, 20).date())
            self.assertEqual(HistoryPage._record_values(records[0])[2], "PASS")
            self.assertEqual(HistoryPage._record_values(records[0])[4], "90.0%")

    def test_history_formats_unknown_when_record_has_missing_fields(self) -> None:
        values = HistoryPage._record_values({"status": ProductStatus.UNKNOWN.value})
        self.assertEqual(values[0], "--")
        self.assertEqual(values[2], "UNKNOWN")
        self.assertEqual(values[4], "0.0%")
