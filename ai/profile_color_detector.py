"""Fallback phát hiện vỉ theo cụm màu đã khai báo trong Product Profile.

Hough circle là lựa chọn đầu tiên.  Tuy nhiên ở ảnh chụp xiên, vách nhựa làm
hai viên kề nhau có thể hợp thành một vùng màu liên tục.  Module này chỉ dùng
khi Hough không tìm được vỉ: nó kiểm tra các *run* màu theo profile (ví dụ
Xanh lá x2 - Xanh dương - Xanh lá x2), tách các cụm màu lớn theo trục vỉ và
vẫn giao phần quyết định PASS/FAIL cho ProductValidator.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from math import pi
from typing import Any

import cv2
import numpy as np

from ai.product_detector import DetectedSlot, ProductDetection
from core.models import ProductProfile


@dataclass(frozen=True, slots=True)
class _ColorComponent:
    color: str
    label: int
    labels: np.ndarray
    x: int
    y: int
    width: int
    height: int
    area: int
    center_x: float
    center_y: float

    @property
    def radius(self) -> float:
        return float(np.sqrt(self.area / pi))


@dataclass(frozen=True, slots=True)
class _Run:
    color: str
    count: int


class ProfileColorSequenceDetector:
    """Tách dãy viên màu khi các vùng cùng màu dính nhau qua vỉ nhựa."""

    def __init__(
        self,
        color_profiles: Mapping[str, Any],
        *,
        minimum_component_area_fraction: float = 0.002,
        maximum_component_area_fraction: float = 0.15,
        maximum_candidates_per_color: int = 8,
        sample_radius_fraction: float = 0.25,
        minimum_split_spacing_ratio: float = 0.45,
    ) -> None:
        if not 0 < minimum_component_area_fraction < maximum_component_area_fraction <= 1:
            raise ValueError("Ngưỡng diện tích cụm màu không hợp lệ")
        if (
            maximum_candidates_per_color < 1
            or not 0 < sample_radius_fraction < 0.5
            or not 0 < minimum_split_spacing_ratio < 1
        ):
            raise ValueError("Cấu hình tách cụm màu không hợp lệ")
        self.color_profiles = color_profiles
        self.minimum_component_area_fraction = minimum_component_area_fraction
        self.maximum_component_area_fraction = maximum_component_area_fraction
        self.maximum_candidates_per_color = maximum_candidates_per_color
        self.sample_radius_fraction = sample_radius_fraction
        # Khi một ô màu bị vách nhựa/phản xạ làm thành một component lớn,
        # ``_split_run_components`` có thể tách nó theo thống kê. Khoảng cách
        # giữa các tâm sau khi tách vẫn phải hợp lý so với các slot kế cận;
        # nếu không một ô tím bị tách đôi sẽ biến vỉ thiếu thành PASS giả.
        self.minimum_split_spacing_ratio = minimum_split_spacing_ratio

    def detect(self, image_bgr: np.ndarray, profile: ProductProfile) -> ProductDetection:
        if not self._is_bgr_image(image_bgr) or len(profile.slots) < 3:
            return ProductDetection([], 0.0, "Không đủ dữ liệu để tách cụm màu theo Product Profile")
        runs = self._runs(profile)
        # Một màu duy nhất không thể giúp phân biệt vỉ với nền; để Hough xử lý.
        if len(runs) < 3:
            return ProductDetection([], 0.0, "Chuỗi màu chưa đủ đặc trưng để tách cụm")

        components = self._components(image_bgr, runs)
        selected, axis, score = self._best_run_sequence(components, runs)
        if selected is None or axis is None:
            return ProductDetection([], 0.0, "Không tìm được chuỗi cụm màu đúng thứ tự")

        slots = self._split_run_components(selected, runs, axis)
        if len(slots) != len(profile.slots):
            return ProductDetection([], 0.0, "Không tách đủ slot từ cụm màu")
        if not self._has_plausible_split_spacing(slots):
            return ProductDetection(
                [],
                0.0,
                "Từ chối chuỗi màu: một slot bị tách thành hai tâm quá sát nhau",
            )
        confidence = max(0.85, min(0.91, 0.93 - score * 0.10))
        return ProductDetection(
            slots,
            confidence,
            "Đã nhận diện dãy slot bằng màu + hình học Product Profile; tách theo trục vỉ",
        )

    def detect_partial(self, image_bgr: np.ndarray, profile: ProductProfile) -> ProductDetection:
        """Tìm dãy ``n - 1`` slot đúng màu để xác nhận vỉ thiếu viên.

        Đây không phải đường PASS: caller phải kiểm tra vị trí bị thiếu thật
        sự trống trước khi sinh NG. Thao tác tách component vẫn cần thiết cho
        hai viên cùng màu đang chạm nhau, nhưng khác ``detect`` ở chỗ vị trí
        slot bị bỏ được lưu rõ ràng trong kết quả.
        """
        expected = tuple(slot.expected_color for slot in profile.slots)
        if not self._is_bgr_image(image_bgr) or len(expected) < 4:
            return ProductDetection([], 0.0, "Không đủ dữ liệu để xác nhận vỉ thiếu slot")

        # Ưu tiên slot thiếu ở giữa. Với hai slot kề nhau cùng màu, cấu hình
        # có thể trùng nhau sau khi bỏ một vị trí; ưu tiên vị trí nội suy ở
        # giữa giúp pipeline lấy đúng vùng trống để xác nhận NG.
        omissions = sorted(range(len(expected)), key=lambda index: (index in {0, len(expected) - 1}, index))
        best: tuple[float, list[DetectedSlot], int] | None = None
        for missing_index in omissions:
            visible_colors = expected[:missing_index] + expected[missing_index + 1:]
            runs = self._runs_from_colors(visible_colors)
            # Một run đơn/đôi màu không đủ đặc trưng để phân biệt vỉ với nền.
            if len(runs) < 3:
                continue
            components = self._components(image_bgr, runs)
            selected, axis, score = self._best_run_sequence(components, runs)
            if selected is None or axis is None:
                continue
            slots = self._split_run_components(selected, runs, axis)
            if len(slots) != len(visible_colors) or not self._has_plausible_split_spacing(slots):
                continue
            if best is None or score < best[0]:
                best = (score, slots, missing_index)

        if best is None:
            return ProductDetection([], 0.0, "Không tìm được chuỗi n-1 slot đúng màu")
        score, slots, missing_index = best
        confidence = max(0.85, min(0.91, 0.93 - score * 0.10))
        return ProductDetection(
            slots,
            confidence,
            "Đã nhận diện chuỗi màu n-1 slot; cần xác minh vùng slot thiếu",
            missing_slot_index=missing_index,
        )

    def _has_plausible_split_spacing(self, slots: list[DetectedSlot]) -> bool:
        """Chặn slot giả do tách một component màu thành nhiều phần.

        Cổng run-màu cần tách được hai viên cùng màu đang chạm nhau, nhưng
        không được chấp nhận một khoảng cách rất nhỏ bên cạnh các khoảng cách
        bình thường của cùng vỉ. Phép chiếu PCA khiến kiểm tra này không phụ
        thuộc vỉ nằm ngang, dọc hay chéo.
        """
        if len(slots) < 3:
            return True
        points = np.asarray([(slot.x, slot.y) for slot in slots], dtype=float)
        center = points.mean(axis=0)
        try:
            _, _, vectors = np.linalg.svd(points - center, full_matrices=False)
        except np.linalg.LinAlgError:
            return False
        axis = vectors[0]
        projection = np.sort((points - center) @ axis)
        spacing = np.diff(projection)
        if len(spacing) == 0 or np.any(spacing < 2.0):
            return False
        median_spacing = max(float(np.median(spacing)), 1.0)
        return float(np.min(spacing)) / median_spacing >= self.minimum_split_spacing_ratio

    def _components(self, image_bgr: np.ndarray, runs: tuple[_Run, ...]) -> dict[str, list[_ColorComponent]]:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        frame_area = image_bgr.shape[0] * image_bgr.shape[1]
        minimum_area = max(24, int(round(frame_area * self.minimum_component_area_fraction)))
        maximum_area = max(minimum_area + 1, int(round(frame_area * self.maximum_component_area_fraction)))
        components: dict[str, list[_ColorComponent]] = {}
        for color in {run.color for run in runs}:
            mask = self._mask_for_color(hsv, self.color_profiles.get(color))
            if mask is None:
                components[color] = []
                continue
            count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
            items: list[_ColorComponent] = []
            for label in range(1, count):
                x, y, width, height, area = (int(value) for value in stats[label])
                if not minimum_area <= area <= maximum_area or width < 3 or height < 3:
                    continue
                center_x, center_y = (float(value) for value in centers[label])
                items.append(_ColorComponent(color, label, labels, x, y, width, height, area, center_x, center_y))
            items.sort(key=lambda item: item.area, reverse=True)
            components[color] = items[:self.maximum_candidates_per_color]
        return components

    def _best_run_sequence(
        self,
        components: dict[str, list[_ColorComponent]],
        runs: tuple[_Run, ...],
    ) -> tuple[tuple[_ColorComponent, ...] | None, np.ndarray | None, float]:
        choices = [components.get(run.color, []) for run in runs]
        if any(not choices_for_run for choices_for_run in choices):
            return None, None, float("inf")
        expected = tuple(run.color for run in runs)
        reversed_expected = tuple(reversed(expected))
        best: tuple[float, tuple[_ColorComponent, ...], np.ndarray] | None = None
        for picked in product(*choices):
            if len({(item.color, item.label) for item in picked}) != len(picked):
                continue
            points = np.asarray([(item.center_x, item.center_y) for item in picked], dtype=float)
            radii = np.asarray([item.radius for item in picked], dtype=float)
            center = points.mean(axis=0)
            try:
                _, _, vectors = np.linalg.svd(points - center, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            axis, normal = vectors[0], vectors[1]
            dominant = 0 if abs(axis[0]) >= abs(axis[1]) else 1
            if axis[dominant] < 0:
                axis, normal = -axis, -normal
            projection = (points - center) @ axis
            order = np.argsort(projection)
            ordered = tuple(picked[index] for index in order)
            labels = tuple(item.color for item in ordered)
            if labels not in {expected, reversed_expected}:
                continue
            # Chuẩn hoá chiều trả về theo thứ tự profile.  Điều này rất quan
            # trọng với profile có số slot cùng màu không đối xứng, ví dụ
            # Tím - Xanh dương - Tím - Tím - Tím: nếu camera nhìn ngược,
            # component cũng phải được ghép với đúng ``run.count``.
            if labels == reversed_expected and labels != expected:
                ordered = tuple(reversed(ordered))
                axis = -axis
            spacing = np.diff(np.sort(projection))
            if len(spacing) == 0 or np.any(spacing < 2):
                continue
            mean_radius = max(float(radii.mean()), 1.0)
            line_error = float(np.mean(np.abs((points - center) @ normal)) / mean_radius)
            spacing_error = float(np.std(spacing) / max(float(np.mean(spacing)), 1.0))
            radius_error = float(np.std(radii) / mean_radius)
            # Vỉ treo mềm/đặt nghiêng có thể võng ở ô giữa, nên tâm các ô
            # không phải lúc nào cũng thẳng hàng.  Thứ tự màu, khoảng cách và
            # kích thước component vẫn là bằng chứng bắt buộc; chỉ nới lỗi
            # đường thẳng để nhận được góc chụp cong chữ V/chữ U thực tế.
            if line_error > 1.10 or spacing_error > 0.55 or radius_error > 0.45:
                continue
            score = line_error + spacing_error * 0.55 + radius_error * 0.25
            if best is None or score < best[0]:
                best = (score, ordered, axis)
        if best is None:
            return None, None, float("inf")
        return best[1], best[2], best[0]

    def _split_run_components(
        self,
        selected: tuple[_ColorComponent, ...],
        runs: tuple[_Run, ...],
        axis: np.ndarray,
    ) -> list[DetectedSlot]:
        all_points = np.asarray([(item.center_x, item.center_y) for item in selected], dtype=float)
        origin = all_points.mean(axis=0)
        estimated: list[tuple[float, float, int, str]] = []
        for component, run in zip(selected, runs, strict=True):
            ys, xs = np.where(component.labels == component.label)
            points = np.column_stack((xs, ys)).astype(float)
            if len(points) < run.count:
                return []
            projections = (points - origin) @ axis
            groups = self._cluster_1d(projections, run.count)
            for group in groups:
                member_points = points[group]
                if len(member_points) < 12:
                    return []
                center_x, center_y = member_points.mean(axis=0)
                sample_radius = max(4, int(round(np.sqrt(len(member_points) / pi) * self.sample_radius_fraction)))
                estimated.append((center_x, center_y, sample_radius, run.color))
        estimated.sort(key=lambda item: (np.asarray(item[:2]) - origin) @ axis)
        return [
            DetectedSlot(
                int(round(center_x)),
                int(round(center_y)),
                max(sample_radius + 4, int(round(sample_radius * 1.75))),
                sample_radius=sample_radius,
                side_view=True,
                matched_color=matched_color,
            )
            for center_x, center_y, sample_radius, matched_color in estimated
        ]

    @staticmethod
    def _cluster_1d(values: np.ndarray, count: int) -> list[np.ndarray]:
        if count == 1:
            return [np.ones(len(values), dtype=bool)]
        centers = np.quantile(values, (np.arange(count, dtype=float) + 0.5) / count)
        labels = np.zeros(len(values), dtype=int)
        for _ in range(24):
            labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
            updated = np.asarray([
                values[labels == index].mean() if np.any(labels == index) else centers[index]
                for index in range(count)
            ])
            if np.allclose(updated, centers, atol=0.25):
                break
            centers = updated
        order = np.argsort(centers)
        return [labels == index for index in order]

    @staticmethod
    def _runs(profile: ProductProfile) -> tuple[_Run, ...]:
        return ProfileColorSequenceDetector._runs_from_colors(
            tuple(slot.expected_color for slot in profile.slots)
        )

    @staticmethod
    def _runs_from_colors(colors: tuple[str, ...]) -> tuple[_Run, ...]:
        runs: list[_Run] = []
        for color in colors:
            if runs and runs[-1].color == color:
                runs[-1] = _Run(runs[-1].color, runs[-1].count + 1)
            else:
                runs.append(_Run(color, 1))
        return tuple(runs)

    @staticmethod
    def _mask_for_color(hsv: np.ndarray, raw_profile: object) -> np.ndarray | None:
        if not isinstance(raw_profile, Mapping):
            return None
        try:
            lower = np.asarray(raw_profile["hsv_min"], dtype=np.uint8)
            upper = np.asarray(raw_profile["hsv_max"], dtype=np.uint8)
        except (KeyError, TypeError, ValueError):
            return None
        if lower.shape != (3,) or upper.shape != (3,) or np.any(lower > upper):
            return None
        mask = cv2.inRange(hsv, lower, upper)
        for extension in (
            ("hsv_extension_min", "hsv_extension_max"),
            ("profile_hsv_extension_min", "profile_hsv_extension_max"),
        ):
            if not any(key in raw_profile for key in extension):
                continue
            try:
                extension_lower = np.asarray(raw_profile[extension[0]], dtype=np.uint8)
                extension_upper = np.asarray(raw_profile[extension[1]], dtype=np.uint8)
            except (KeyError, TypeError, ValueError):
                return None
            if (
                extension_lower.shape != (3,)
                or extension_upper.shape != (3,)
                or np.any(extension_lower > extension_upper)
            ):
                return None
            mask |= cv2.inRange(hsv, extension_lower, extension_upper)
        mask[(hsv[:, :, 1] < 50) | (hsv[:, :, 2] < 35)] = 0
        return mask

    @staticmethod
    def _is_bgr_image(image: object) -> bool:
        return (
            isinstance(image, np.ndarray)
            and image.ndim == 3
            and image.shape[2] == 3
            and image.shape[0] > 0
            and image.shape[1] > 0
            and image.dtype == np.uint8
        )
