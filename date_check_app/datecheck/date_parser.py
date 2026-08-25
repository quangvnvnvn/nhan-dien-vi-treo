"""Phân tích và kiểm tra date in hai dòng từ kết quả OCR."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import re


class DateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "NG"
    REVIEW = "REVIEW"


@dataclass(frozen=True, slots=True)
class DateExpectation:
    manufacture_date: str = ""
    dm_code: str = "DM1"
    batch_code: str = ""


@dataclass(frozen=True, slots=True)
class DateRead:
    manufacture_date: str | None
    dm_code: str | None
    batch_code: str | None
    time_value: str | None
    normalized_text: str

    @property
    def key(self) -> tuple[str | None, str | None, str | None, str | None]:
        return self.manufacture_date, self.dm_code, self.batch_code, self.time_value


@dataclass(frozen=True, slots=True)
class DateValidation:
    status: DateStatus
    read: DateRead
    detail: str


class DateParser:
    """Đọc đúng mẫu ``NSX ddmmyy DM1`` và ``XX hh:mm``.

    ``XX`` được xem là mã lô gồm hai ký tự chữ/số. Nếu không cần kiểm tra mã
    lô, người vận hành để trống trường kỳ vọng; phần giờ vẫn được kiểm tra hợp
    lệ 00:00–23:59.
    """

    # Máy in nhiệt / laser thường làm EasyOCR nhầm S ↔ 5, X ↔ K và 1 ↔ I.
    # Chỉ nới ở *nhãn* NSX/DM, không sửa toàn bộ chuỗi để không biến lỗi đọc
    # thành một kết quả PASS giả.
    _LINE_ONE = re.compile(r"N[S5][XK]\s*([0-9O]{6})\s*D[MNN][1IL]", re.IGNORECASE)
    _LINE_TWO = re.compile(r"\b([A-Z0-9]{2})\s*([0-2O][0-9OIL])\s*[:.;]\s*([0-5S][0-9OILS])\b")

    @classmethod
    def parse(cls, text: str) -> DateRead:
        normalized = cls._normalize(text)
        compact = re.sub(r"\s+", "", normalized)
        first = cls._LINE_ONE.search(normalized) or re.search(r"N[S5][XK]([0-9O]{6})D[MNN][1IL]", compact)
        second = cls._LINE_TWO.search(normalized)
        manufacture_date = cls._numeric(first.group(1)) if first else None
        dm_code = "DM1" if first else None
        batch_code = second.group(1) if second else None
        time_value = f"{cls._numeric(second.group(2))}:{cls._numeric(second.group(3))}" if second else None
        if manufacture_date and not cls._is_calendar_date(manufacture_date):
            manufacture_date = None
        return DateRead(manufacture_date, dm_code, batch_code, time_value, normalized)

    @classmethod
    def validate(cls, text: str, expectation: DateExpectation) -> DateValidation:
        read = cls.parse(text)
        if not read.manufacture_date or not read.dm_code or not read.batch_code or not read.time_value:
            return DateValidation(DateStatus.REVIEW, read, "Chưa đọc đủ mẫu: NSX ddmmyy DM1 / XX hh:mm")

        wanted_date = cls._digits(expectation.manufacture_date)
        wanted_dm = expectation.dm_code.strip().upper().replace("I", "1")
        wanted_batch = expectation.batch_code.strip().upper()
        if wanted_date and read.manufacture_date != wanted_date:
            return DateValidation(DateStatus.FAIL, read, f"Sai NSX: đọc {read.manufacture_date}, yêu cầu {wanted_date}")
        if wanted_dm and read.dm_code != wanted_dm:
            return DateValidation(DateStatus.FAIL, read, f"Sai mã DM: đọc {read.dm_code}, yêu cầu {wanted_dm}")
        if wanted_batch and read.batch_code != wanted_batch:
            return DateValidation(DateStatus.FAIL, read, f"Sai mã XX: đọc {read.batch_code}, yêu cầu {wanted_batch}")
        return DateValidation(DateStatus.PASS, read, "Đúng mẫu date và đạt tiêu chí đã cấu hình")

    @staticmethod
    def _normalize(text: str) -> str:
        value = str(text or "").upper()
        value = value.replace("Đ", "D")
        value = value.replace("|", "1")
        value = re.sub(r"[^A-Z0-9\s:.;]", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _digits(value: str) -> str:
        return DateParser._numeric(value)

    @staticmethod
    def _numeric(value: str) -> str:
        """Quy đổi các nhầm lẫn OCR phổ biến, chỉ cho trường số."""
        replacements = str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5"})
        return str(value).upper().translate(replacements)

    @staticmethod
    def _is_calendar_date(value: str) -> bool:
        try:
            datetime.strptime(value, "%d%m%y")
        except ValueError:
            return False
        return True
