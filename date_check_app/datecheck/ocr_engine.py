"""OCR cục bộ cho vùng date cố định của camera."""
from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Iterable

import cv2
import numpy as np

from datecheck.date_parser import DateExpectation, DateParser, DateStatus, DateValidation


@dataclass(frozen=True, slots=True)
class OcrResult:
    validation: DateValidation
    text: str
    confidence: float
    words: tuple[str, ...]
    roi_image: np.ndarray


class DateOcrEngine:
    """EasyOCR lazy-load cho chữ date nhỏ, in mờ trong fixed ROI."""

    _reader_lock = threading.Lock()
    _reader: object | None = None

    def __init__(self) -> None:
        self._read_lock = threading.Lock()

    @classmethod
    def _get_reader(cls) -> object:
        with cls._reader_lock:
            if cls._reader is None:
                try:
                    import easyocr
                    import torch
                except ImportError as error:
                    raise RuntimeError("Thiếu EasyOCR. Chạy pip install -r requirements-date-check.txt") from error
                # GPU dùng được sẽ tự tăng tốc; CPU vẫn đủ khi OCR theo ROI
                # và không chạy mọi frame.
                cls._reader = easyocr.Reader(["en"], gpu=bool(torch.cuda.is_available()), verbose=False)
            return cls._reader

    @staticmethod
    def preprocess_variants(roi_bgr: np.ndarray) -> list[np.ndarray]:
        if roi_bgr.size == 0:
            return []
        # Date trong luồng camera thường chỉ cao vài pixel.  Phóng to trước,
        # rồi dùng các phiên bản tương phản khác nhau để chống lóa vỏ nhựa.
        scale = 4.0 if min(roi_bgr.shape[:2]) < 420 else 2.5
        enlarged = cv2.resize(roi_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 5, 35, 35)
        clahe = cv2.createCLAHE(clipLimit=3.2, tileGridSize=(8, 8)).apply(denoised)
        blurred = cv2.GaussianBlur(clahe, (0, 0), 1.1)
        sharpened = cv2.addWeighted(clahe, 1.7, blurred, -0.7, 0)
        adaptive = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 4,
        )
        otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return [enlarged, sharpened, adaptive, otsu]

    def inspect(self, roi_bgr: np.ndarray, expectation: DateExpectation) -> OcrResult:
        if roi_bgr.size == 0:
            validation = DateParser.validate("", expectation)
            return OcrResult(validation, "", 0.0, (), roi_bgr)

        reader = self._get_reader()
        candidates: list[tuple[DateValidation, str, float, tuple[str, ...]]] = []
        with self._read_lock:
            for image in self.preprocess_variants(roi_bgr):
                # detail=1 giúp sắp theo vị trí dòng, tránh trộn text nhãn nền.
                items = reader.readtext(  # type: ignore[attr-defined]
                    image,
                    detail=1,
                    paragraph=False,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:.; ",
                    min_size=4,
                    text_threshold=0.28,
                    low_text=0.16,
                    link_threshold=0.12,
                    contrast_ths=0.03,
                    adjust_contrast=0.85,
                    canvas_size=3200,
                    mag_ratio=1.6,
                )
                words, confidence = self._ordered_words(items)
                text = " ".join(words)
                validation = DateParser.validate(text, expectation)
                candidates.append((validation, text, confidence, tuple(words)))
                # Nếu ảnh đầu đã đọc đủ mẫu, không làm chậm video bằng các
                # phiên bản dự phòng.  Trường hợp NG cũng cần giữ lại vì date
                # đã được đọc đủ để cảnh báo chính xác.
                if validation.status in (DateStatus.PASS, DateStatus.FAIL):
                    break

        # Ưu tiên PASS rồi NG (đọc đủ nhưng sai) rồi REVIEW; trong cùng loại
        # chọn confidence OCR cao nhất. Nhờ đó không bỏ mất lỗi date rõ ràng.
        rank = {DateStatus.PASS: 2, DateStatus.FAIL: 1, DateStatus.REVIEW: 0}
        validation, text, confidence, words = max(candidates, key=lambda item: (rank[item[0].status], item[2]))
        return OcrResult(validation, text, confidence, words, roi_bgr.copy())

    @staticmethod
    def _ordered_words(items: Iterable[object]) -> tuple[list[str], float]:
        detected: list[tuple[float, float, str, float]] = []
        for item in items:
            if not isinstance(item, (tuple, list)) or len(item) < 3:
                continue
            box, text, confidence = item[0], str(item[1]).strip(), item[2]
            if not text or not isinstance(box, (tuple, list)) or not box:
                continue
            try:
                x = float(box[0][0])
                y = float(box[0][1])
                score = float(confidence)
            except (IndexError, TypeError, ValueError):
                continue
            detected.append((y, x, text, score))
        if not detected:
            return [], 0.0
        detected.sort(key=lambda value: (round(value[0] / 24.0), value[1]))
        return [value[2] for value in detected], float(np.mean([value[3] for value in detected]))
