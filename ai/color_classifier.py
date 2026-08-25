"""Phân loại màu bảo thủ bằng độ phủ pixel HSV + LAB.

Vỉ trong suốt thường tạo nhiều pixel chói, gần trắng.  Vì vậy không dùng một
pixel median cho toàn slot: bộ phân loại loại pixel ít bão hòa theo cấu hình,
đo độ phủ của từng màu trên phần pixel còn lại, rồi chỉ trả về kết quả khi màu
thắng đủ rõ so với các màu cạnh tranh.  Cấu hình ngưỡng nằm trong
``config/colors.yaml`` để hiệu chỉnh theo camera/đèn thực tế.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True)
class ColorResult:
    name: str | None
    confidence: float


@dataclass(frozen=True, slots=True)
class _SamplingConfig:
    minimum_saturation: float
    minimum_value: float
    maximum_value: float
    minimum_eligible_pixels: int
    minimum_eligible_fraction: float


@dataclass(frozen=True, slots=True)
class _ScoringConfig:
    minimum_hsv_coverage: float
    minimum_lab_coverage: float
    minimum_combined_coverage: float
    target_hsv_coverage: float
    target_lab_coverage: float
    target_combined_coverage: float
    minimum_matching_pixels: int
    target_matching_pixels: int
    minimum_agreement: float
    minimum_confidence: float
    dominance_margin: float
    maximum_competitor_ratio: float
    hsv_coverage_weight: float
    lab_coverage_weight: float
    combined_coverage_weight: float
    support_weight: float
    agreement_weight: float


@dataclass(frozen=True, slots=True)
class _ColorProfile:
    name: str
    hsv_min: np.ndarray
    hsv_max: np.ndarray
    lab_min: np.ndarray
    lab_max: np.ndarray
    sampling: _SamplingConfig | None
    scoring: _ScoringConfig | None
    hsv_extension_min: np.ndarray | None = None
    hsv_extension_max: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class _Candidate:
    profile: _ColorProfile
    hsv_coverage: float
    lab_coverage: float
    combined_coverage: float
    agreement: float
    matching_pixels: int
    confidence: float


class ColorClassifier:
    """Phân loại từ tập pixel BGR, không suy đoán khi màu chưa đủ bằng chứng.

    Mỗi profile sản xuất phải có ``classification.sampling`` và
    ``classification.scoring``.  Profile cũ thiếu hai phần này chỉ có thể PASS
    khi *toàn bộ* pixel đồng thời khớp HSV và LAB; đây là nhánh tương thích rất
    nghiêm ngặt, không có ngưỡng mặc định nới lỏng trong mã nguồn.
    """

    def __init__(self, profiles: dict[str, Any]) -> None:
        # Giữ nguyên mapping công khai vì pipeline dùng nó để chuẩn hóa alias.
        self.profiles = profiles
        self._configured_profiles = tuple(
            parsed
            for name, profile in profiles.items()
            if (parsed := self._parse_profile(str(name), profile)) is not None
        )

    def classify(self, bgr_pixels: np.ndarray) -> ColorResult:
        """Trả một màu duy nhất hoặc ``None`` khi ảnh thiếu/mơ hồ bằng chứng."""
        pixels = self._flatten_bgr_pixels(bgr_pixels)
        if pixels is None or not self._configured_profiles:
            return ColorResult(None, 0.0)

        try:
            hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV).reshape(-1, 3)
            lab = cv2.cvtColor(pixels, cv2.COLOR_BGR2LAB).reshape(-1, 3)
        except cv2.error:
            return ColorResult(None, 0.0)

        candidates = [
            candidate
            for profile in self._configured_profiles
            if (candidate := self._score_profile(profile, hsv, lab)) is not None
        ]
        if not candidates:
            return ColorResult(None, 0.0)

        candidates.sort(
            key=lambda item: (item.confidence, item.combined_coverage, item.matching_pixels),
            reverse=True,
        )
        winner = candidates[0]
        if self._is_ambiguous(winner, candidates[1:]):
            return ColorResult(None, 0.0)
        return ColorResult(winner.profile.name, winner.confidence)

    def classify_sequence(
        self,
        bgr_samples: list[np.ndarray],
        expected_colors: list[str],
    ) -> list[ColorResult]:
        """Phân loại từng slot và chỉ hiệu chỉnh khi tương quan màu rất rõ.

        Vỉ nhựa trong làm hue ``Tím`` và ``Xanh dương`` dịch theo ánh sáng,
        do đó một slot riêng lẻ đôi khi nằm ở vùng biên. Tuy nhiên trên một vỉ
        đúng, viên Xanh dương phải tách rõ khỏi *tất cả* viên Tím còn lại.
        Hàm này chỉ dùng quan hệ đó khi mọi slot xanh dương có hue thấp hơn
        mọi slot tím tối thiểu 3 đơn vị OpenCV. Điều này có thể cứu ảnh webcam
        lệch màu, nhưng không thể biến vỉ có viên tím đặt nhầm tại slot xanh
        dương thành PASS.
        """
        results = [self.classify(sample) for sample in bgr_samples]
        if len(bgr_samples) != len(expected_colors) or not bgr_samples:
            return results

        palette = set(expected_colors)
        if not palette.issubset({"purple", "blue"}) or not {"purple", "blue"}.issubset(palette):
            return results

        hues = [self._median_colored_hue(sample) for sample in bgr_samples]
        blue_indexes = [index for index, color in enumerate(expected_colors) if color == "blue"]
        purple_indexes = [index for index, color in enumerate(expected_colors) if color == "purple"]
        blue_hues = [hues[index] for index in blue_indexes if hues[index] is not None]
        purple_hues = [hues[index] for index in purple_indexes if hues[index] is not None]
        if len(blue_hues) != len(blue_indexes) or len(purple_hues) != len(purple_indexes):
            return results

        # Không dùng trung bình: chỉ một viên tím thực sự nằm sai vị trí cũng
        # phải chặn PASS thay vì bị che bởi các slot đúng còn lại.
        if max(blue_hues) + 3.0 > min(purple_hues):
            return results

        corrected: list[ColorResult] = []
        for expected, result in zip(expected_colors, results, strict=True):
            # Nếu classifier độc lập thấy màu thứ ba (ví dụ Xanh lá) thì đó
            # là NG rõ ràng và tuyệt đối không được sequence ghi đè.
            if result.name is not None and result.name not in {"purple", "blue"}:
                corrected.append(result)
            else:
                corrected.append(ColorResult(expected, max(result.confidence, 0.90)))
        return corrected

    @staticmethod
    def _median_colored_hue(bgr_pixels: np.ndarray) -> float | None:
        pixels = ColorClassifier._flatten_bgr_pixels(bgr_pixels)
        if pixels is None:
            return None
        try:
            hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        except cv2.error:
            return None
        eligible = (hsv[:, 1] >= 35) & (hsv[:, 2] >= 25)
        if int(np.count_nonzero(eligible)) < 24:
            return None
        return float(np.median(hsv[eligible, 0]))

    @staticmethod
    def _flatten_bgr_pixels(bgr_pixels: object) -> np.ndarray | None:
        """Chuẩn hóa mọi shape ``(..., 3)`` thành ``(N, 1, 3)`` cho OpenCV."""
        if not isinstance(bgr_pixels, np.ndarray) or bgr_pixels.size == 0:
            return None
        if bgr_pixels.ndim < 2 or bgr_pixels.shape[-1] != 3 or bgr_pixels.dtype != np.uint8:
            return None
        return np.ascontiguousarray(bgr_pixels.reshape(-1, 1, 3))

    @classmethod
    def _parse_profile(cls, name: str, raw_profile: object) -> _ColorProfile | None:
        if not isinstance(raw_profile, Mapping):
            return None
        try:
            hsv_min = cls._triplet(raw_profile["hsv_min"])
            hsv_max = cls._triplet(raw_profile["hsv_max"])
            lab_min = cls._triplet(raw_profile["lab_min"])
            lab_max = cls._triplet(raw_profile["lab_max"])
        except (KeyError, TypeError, ValueError):
            return None
        if np.any(hsv_min > hsv_max) or np.any(lab_min > lab_max):
            return None
        extension_keys = {"hsv_extension_min", "hsv_extension_max"}
        if extension_keys & raw_profile.keys() and not extension_keys <= raw_profile.keys():
            return None
        try:
            hsv_extension_min = cls._triplet(raw_profile["hsv_extension_min"]) if extension_keys <= raw_profile.keys() else None
            hsv_extension_max = cls._triplet(raw_profile["hsv_extension_max"]) if extension_keys <= raw_profile.keys() else None
        except (TypeError, ValueError):
            return None
        if hsv_extension_min is not None and np.any(hsv_extension_min > hsv_extension_max):
            return None
        if "classification" not in raw_profile:
            return _ColorProfile(
                name, hsv_min, hsv_max, lab_min, lab_max, None, None,
                hsv_extension_min, hsv_extension_max,
            )
        classification = raw_profile["classification"]
        if not isinstance(classification, Mapping):
            return None
        sampling_raw = classification.get("sampling")
        scoring_raw = classification.get("scoring")
        if not isinstance(sampling_raw, Mapping) or not isinstance(scoring_raw, Mapping):
            return None
        try:
            sampling = _SamplingConfig(
                minimum_saturation=cls._number(sampling_raw, "minimum_saturation"),
                minimum_value=cls._number(sampling_raw, "minimum_value"),
                maximum_value=cls._number(sampling_raw, "maximum_value"),
                minimum_eligible_pixels=cls._positive_integer(sampling_raw, "minimum_eligible_pixels"),
                minimum_eligible_fraction=cls._fraction(sampling_raw, "minimum_eligible_fraction"),
            )
            scoring = _ScoringConfig(
                minimum_hsv_coverage=cls._fraction(scoring_raw, "minimum_hsv_coverage"),
                minimum_lab_coverage=cls._fraction(scoring_raw, "minimum_lab_coverage"),
                minimum_combined_coverage=cls._fraction(scoring_raw, "minimum_combined_coverage"),
                target_hsv_coverage=cls._positive_fraction(scoring_raw, "target_hsv_coverage"),
                target_lab_coverage=cls._positive_fraction(scoring_raw, "target_lab_coverage"),
                target_combined_coverage=cls._positive_fraction(scoring_raw, "target_combined_coverage"),
                minimum_matching_pixels=cls._positive_integer(scoring_raw, "minimum_matching_pixels"),
                target_matching_pixels=cls._positive_integer(scoring_raw, "target_matching_pixels"),
                minimum_agreement=cls._fraction(scoring_raw, "minimum_agreement"),
                minimum_confidence=cls._fraction(scoring_raw, "minimum_confidence"),
                dominance_margin=cls._fraction(scoring_raw, "dominance_margin"),
                maximum_competitor_ratio=cls._fraction(scoring_raw, "maximum_competitor_ratio"),
                hsv_coverage_weight=cls._nonnegative_number(scoring_raw, "hsv_coverage_weight"),
                lab_coverage_weight=cls._nonnegative_number(scoring_raw, "lab_coverage_weight"),
                combined_coverage_weight=cls._nonnegative_number(scoring_raw, "combined_coverage_weight"),
                support_weight=cls._nonnegative_number(scoring_raw, "support_weight"),
                agreement_weight=cls._nonnegative_number(scoring_raw, "agreement_weight"),
            )
        except (KeyError, TypeError, ValueError):
            return None

        if (
            sampling.minimum_value > sampling.maximum_value
            or scoring.target_matching_pixels < scoring.minimum_matching_pixels
            or scoring.target_hsv_coverage < scoring.minimum_hsv_coverage
            or scoring.target_lab_coverage < scoring.minimum_lab_coverage
            or scoring.target_combined_coverage < scoring.minimum_combined_coverage
            or cls._weight_sum(scoring) <= 0.0
        ):
            return None
        return _ColorProfile(
            name, hsv_min, hsv_max, lab_min, lab_max, sampling, scoring,
            hsv_extension_min, hsv_extension_max,
        )

    @staticmethod
    def _triplet(value: object) -> np.ndarray:
        values = np.asarray(value, dtype=np.int16)
        if values.shape != (3,):
            raise ValueError("Ngưỡng HSV/LAB phải gồm đúng 3 kênh")
        return values

    @staticmethod
    def _number(mapping: Mapping[str, object], key: str) -> float:
        value = mapping[key]
        if isinstance(value, bool):
            raise ValueError(f"{key} không được là boolean")
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{key} phải là số hữu hạn")
        return number

    @classmethod
    def _nonnegative_number(cls, mapping: Mapping[str, object], key: str) -> float:
        number = cls._number(mapping, key)
        if number < 0.0:
            raise ValueError(f"{key} không được âm")
        return number

    @classmethod
    def _fraction(cls, mapping: Mapping[str, object], key: str) -> float:
        number = cls._nonnegative_number(mapping, key)
        if number > 1.0:
            raise ValueError(f"{key} phải thuộc [0, 1]")
        return number

    @classmethod
    def _positive_fraction(cls, mapping: Mapping[str, object], key: str) -> float:
        number = cls._fraction(mapping, key)
        if number == 0.0:
            raise ValueError(f"{key} phải lớn hơn 0")
        return number

    @classmethod
    def _positive_integer(cls, mapping: Mapping[str, object], key: str) -> int:
        number = cls._number(mapping, key)
        if not number.is_integer() or number <= 0.0:
            raise ValueError(f"{key} phải là số nguyên dương")
        return int(number)

    @classmethod
    def _score_profile(
        cls,
        profile: _ColorProfile,
        hsv: np.ndarray,
        lab: np.ndarray,
    ) -> _Candidate | None:
        if profile.sampling is None or profile.scoring is None:
            return cls._score_legacy_profile(profile, hsv, lab)
        sampling = profile.sampling
        eligible = (
            (hsv[:, 1] >= sampling.minimum_saturation)
            & (hsv[:, 2] >= sampling.minimum_value)
            & (hsv[:, 2] <= sampling.maximum_value)
        )
        eligible_pixels = int(np.count_nonzero(eligible))
        if (
            eligible_pixels < sampling.minimum_eligible_pixels
            or eligible_pixels / len(hsv) < sampling.minimum_eligible_fraction
        ):
            return None

        hsv_match = cls._within(hsv, profile.hsv_min, profile.hsv_max)
        if profile.hsv_extension_min is not None and profile.hsv_extension_max is not None:
            hsv_match |= cls._within(hsv, profile.hsv_extension_min, profile.hsv_extension_max)
        hsv_match &= eligible
        lab_match = eligible & cls._within(lab, profile.lab_min, profile.lab_max)
        combined_match = hsv_match & lab_match
        hsv_pixels = int(np.count_nonzero(hsv_match))
        lab_pixels = int(np.count_nonzero(lab_match))
        matching_pixels = int(np.count_nonzero(combined_match))
        if hsv_pixels == 0 or lab_pixels == 0 or matching_pixels == 0:
            return None

        hsv_coverage = hsv_pixels / eligible_pixels
        lab_coverage = lab_pixels / eligible_pixels
        combined_coverage = matching_pixels / eligible_pixels
        agreement = matching_pixels / min(hsv_pixels, lab_pixels)
        scoring = profile.scoring
        if (
            matching_pixels < scoring.minimum_matching_pixels
            or hsv_coverage < scoring.minimum_hsv_coverage
            or lab_coverage < scoring.minimum_lab_coverage
            or combined_coverage < scoring.minimum_combined_coverage
            or agreement < scoring.minimum_agreement
        ):
            return None

        confidence = cls._confidence(
            scoring,
            hsv_coverage,
            lab_coverage,
            combined_coverage,
            matching_pixels,
            agreement,
        )
        if confidence < scoring.minimum_confidence:
            return None
        return _Candidate(
            profile,
            hsv_coverage,
            lab_coverage,
            combined_coverage,
            agreement,
            matching_pixels,
            confidence,
        )

    @classmethod
    def _score_legacy_profile(
        cls,
        profile: _ColorProfile,
        hsv: np.ndarray,
        lab: np.ndarray,
    ) -> _Candidate | None:
        """Tương thích profile cũ theo điều kiện nghiêm ngặt tuyệt đối.

        Nhánh này chỉ tồn tại để các profile/test cũ không bị crash khi nâng
        cấp.  Một hỗn hợp phản quang hoặc hai màu luôn cho UNKNOWN, vì bất kỳ
        pixel lệch nào cũng làm profile bị loại.
        """
        hsv_match = cls._within(hsv, profile.hsv_min, profile.hsv_max)
        lab_match = cls._within(lab, profile.lab_min, profile.lab_max)
        combined_match = hsv_match & lab_match
        matching_pixels = int(np.count_nonzero(combined_match))
        if matching_pixels != len(hsv):
            return None
        return _Candidate(profile, 1.0, 1.0, 1.0, 1.0, matching_pixels, 1.0)

    @staticmethod
    def _within(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        return np.all((values >= lower) & (values <= upper), axis=1)

    @classmethod
    def _confidence(
        cls,
        scoring: _ScoringConfig,
        hsv_coverage: float,
        lab_coverage: float,
        combined_coverage: float,
        matching_pixels: int,
        agreement: float,
    ) -> float:
        weighted = (
            scoring.hsv_coverage_weight * min(hsv_coverage / scoring.target_hsv_coverage, 1.0)
            + scoring.lab_coverage_weight * min(lab_coverage / scoring.target_lab_coverage, 1.0)
            + scoring.combined_coverage_weight * min(
                combined_coverage / scoring.target_combined_coverage,
                1.0,
            )
            + scoring.support_weight * min(matching_pixels / scoring.target_matching_pixels, 1.0)
            + scoring.agreement_weight * agreement
        )
        return weighted / cls._weight_sum(scoring)

    @staticmethod
    def _weight_sum(scoring: _ScoringConfig) -> float:
        return (
            scoring.hsv_coverage_weight
            + scoring.lab_coverage_weight
            + scoring.combined_coverage_weight
            + scoring.support_weight
            + scoring.agreement_weight
        )

    @staticmethod
    def _is_ambiguous(winner: _Candidate, competitors: list[_Candidate]) -> bool:
        if not competitors:
            return False
        runner_up = competitors[0]
        scoring = winner.profile.scoring
        if scoring is None:
            return True
        confidence_ratio = runner_up.confidence / winner.confidence
        coverage_gap = winner.combined_coverage - runner_up.combined_coverage
        # Confidence is deliberately capped once a color has enough pixels.
        # Therefore a weak runner-up can have a score close to 1.0 merely from
        # a small overlap at an HSV boundary (blue/purple in particular).  It
        # is ambiguous only when both scores are close *and* their pixel
        # coverage is close; a clearly dominant color remains identifiable.
        return (
            confidence_ratio >= scoring.maximum_competitor_ratio
            and coverage_gap <= scoring.dominance_margin
        )
