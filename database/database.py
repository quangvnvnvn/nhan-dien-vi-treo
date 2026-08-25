"""Lưu kết quả một cách truy vết được vào SQLite local."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from core.models import ValidationResult


class ResultRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, product_id TEXT,
                track_id TEXT, status TEXT NOT NULL, reason TEXT, confidence REAL NOT NULL,
                metadata_json TEXT NOT NULL)""")

    def save(self, track_id: str | None, result: ValidationResult) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute("INSERT INTO inspections VALUES (NULL, ?, ?, ?, ?, ?, ?, ?)",
                         (datetime.now(timezone.utc).isoformat(), result.product_id, track_id,
                          result.status.value, result.reason.value if result.reason else None,
                          result.confidence, json.dumps(result.metadata(), ensure_ascii=False)))
