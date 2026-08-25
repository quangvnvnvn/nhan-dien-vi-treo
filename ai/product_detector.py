"""Detector hình học bảo thủ cho vỉ có các ô nằm trên một trục.

Detector này không thay thế model segmentation đã được huấn luyện. Nó kết hợp
Hough circle với contour của vùng có độ bão hoà cao để xử lý ảnh nghiêng/phối
cảnh vừa phải, nhưng vẫn từ chối ảnh chụp cạnh khi mặt ô không đủ rõ.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import pi

import cv2
import numpy as np


@dataclass(slots=True)
class DetectedSlot:
    x: int
    y: int
    radius: int
    # ``radius`` giữ tương thích với tracker cũ; ellipse dùng khi góc camera
    # làm slot không còn tròn theo pixel.
    radius_x: int | None = None
    radius_y: int | None = None
    angle_deg: float = 0.0
    # Góc nhìn cạnh làm Hough trả về bán kính lớn/chồng nhau.  Sampling radius
    # tách riêng để lấy đúng phần viên màu bên trong, không lấy cả vỉ nhựa.
    sample_radius: int | None = None
    side_view: bool = False
    # Chỉ có ở fallback theo Product Profile: pixel tại tâm đã thuộc một cụm
    # HSV của màu này. Nó giúp giữ nhận diện ổn định khi ánh sáng/camera làm
    # bộ phân loại tổng quát dịch xanh dương sang tím ở biên hue.
    matched_color: str | None = None


@dataclass(slots=True)
class ProductDetection:
    slots: list[DetectedSlot]
    confidence: float
    detail: str
    # ``None`` cho detector đầy đủ. Fallback theo màu có thể xác định chính
    # xác vị trí profile bị thiếu khi nó chỉ tìm được n - 1 slot; pipeline
    # dùng metadata này để báo NG đúng ô thay vì suy diễn từ một khoảng trống.
    missing_slot_index: int | None = None


@dataclass(slots=True)
class _RowFit:
    score: float
    selected: np.ndarray
    projective_error: float
    line_error: float
    minimum_spacing_ratio: float


class GeometricProductDetector:
    """Tìm dãy ô gần tròn/ellipse nằm đều trên một trục.

    Cổng geometry dùng sai số projective thay vì buộc khoảng cách pixel phải
    đều tuyệt đối. Nhờ vậy góc nhìn xiên vừa phải vẫn có cơ hội được nhận diện,
    còn circle giả chồng lấn ở ảnh chụp cạnh bị loại an toàn.
    """

    def __init__(
        self,
        *,
        minimum_slot_radius_px: int = 12,
        minimum_slot_radius_fraction: float = 0.032,
        maximum_slot_radius_fraction: float = 0.10,
        minimum_slot_distance_fraction: float = 0.07,
        hough_param1: float = 100.0,
        hough_param2: float = 36.0,
        maximum_geometry_score: float = 0.42,
        # 9 ứng viên vẫn dư cho một vỉ 5 slot và phản xạ lân cận, nhưng giảm
        # tổ hợp phối cảnh từ C(14, 5)=2002 xuống C(9, 5)=126. Đây là phần
        # quyết định FPS trên camera/băng tải độ phân giải dọc.
        candidate_limit: int = 9,
        processing_max_dimension: int | None = None,
        enable_chroma_contour_fallback: bool = True,
        chroma_saturation_min: int = 80,
        chroma_value_min: int = 45,
        minimum_ellipse_axis_ratio: float = 0.50,
        contour_min_radius_factor: float = 0.65,
        enable_clahe_fallback: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: int = 8,
        maximum_projective_spacing_error: float = 0.065,
        maximum_line_deviation_ratio: float = 0.70,
        minimum_spacing_to_radius: float = 1.20,
        enable_side_view_fallback: bool = True,
        side_view_maximum_geometry_score: float = 0.30,
        side_view_minimum_spacing_to_radius: float = 0.80,
        side_view_sample_radius_fraction: float = 0.26,
        side_view_refine_cross_axis_fraction: float = 0.70,
        side_view_refine_along_axis_fraction: float = 0.22,
        side_view_chroma_saturation_min: int = 50,
        side_view_chroma_value_min: int = 45,
        side_view_minimum_chroma_fraction: float = 0.20,
    ) -> None:
        if minimum_slot_radius_px < 2 or not 0 < minimum_slot_radius_fraction < maximum_slot_radius_fraction:
            raise ValueError("Ngưỡng bán kính slot không hợp lệ")
        if not 0 < minimum_slot_distance_fraction < 1 or hough_param1 <= 0 or hough_param2 <= 0:
            raise ValueError("Ngưỡng Hough không hợp lệ")
        if candidate_limit < 2 or not 0 < maximum_geometry_score <= 2:
            raise ValueError("Ngưỡng geometry không hợp lệ")
        if processing_max_dimension is not None and processing_max_dimension < 128:
            raise ValueError("Kích thước xử lý tối đa phải từ 128 px")
        if not 0 <= chroma_saturation_min <= 255 or not 0 <= chroma_value_min <= 255:
            raise ValueError("Ngưỡng vùng màu không hợp lệ")
        if not 0 < minimum_ellipse_axis_ratio <= 1 or contour_min_radius_factor <= 0:
            raise ValueError("Ngưỡng ellipse không hợp lệ")
        if clahe_clip_limit <= 0 or clahe_tile_grid_size < 2:
            raise ValueError("Cấu hình CLAHE không hợp lệ")
        if not 0 < maximum_projective_spacing_error < 1 or not 0 < maximum_line_deviation_ratio < 2:
            raise ValueError("Ngưỡng phối cảnh không hợp lệ")
        if minimum_spacing_to_radius <= 0:
            raise ValueError("Ngưỡng khoảng cách slot không hợp lệ")
        if not 0 < side_view_maximum_geometry_score <= 2 or side_view_minimum_spacing_to_radius <= 0:
            raise ValueError("Ngưỡng side-view không hợp lệ")
        if not 0 < side_view_sample_radius_fraction < 0.5:
            raise ValueError("Bán kính lấy mẫu side-view không hợp lệ")
        if not 0 < side_view_refine_cross_axis_fraction <= 1.5 or not 0 <= side_view_refine_along_axis_fraction <= 0.5:
            raise ValueError("Phạm vi căn chỉnh side-view không hợp lệ")
        if not 0 <= side_view_chroma_saturation_min <= 255 or not 0 <= side_view_chroma_value_min <= 255:
            raise ValueError("Ngưỡng màu side-view không hợp lệ")
        if not 0 < side_view_minimum_chroma_fraction <= 1:
            raise ValueError("Ngưỡng bằng chứng màu side-view không hợp lệ")

        self.minimum_slot_radius_px = minimum_slot_radius_px
        self.minimum_slot_radius_fraction = minimum_slot_radius_fraction
        self.maximum_slot_radius_fraction = maximum_slot_radius_fraction
        self.minimum_slot_distance_fraction = minimum_slot_distance_fraction
        self.hough_param1 = hough_param1
        self.hough_param2 = hough_param2
        self.maximum_geometry_score = maximum_geometry_score
        self.candidate_limit = candidate_limit
        self.processing_max_dimension = processing_max_dimension
        self.enable_chroma_contour_fallback = enable_chroma_contour_fallback
        self.chroma_saturation_min = chroma_saturation_min
        self.chroma_value_min = chroma_value_min
        self.minimum_ellipse_axis_ratio = minimum_ellipse_axis_ratio
        self.contour_min_radius_factor = contour_min_radius_factor
        self.enable_clahe_fallback = enable_clahe_fallback
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.maximum_projective_spacing_error = maximum_projective_spacing_error
        self.maximum_line_deviation_ratio = maximum_line_deviation_ratio
        self.minimum_spacing_to_radius = minimum_spacing_to_radius
        self.enable_side_view_fallback = enable_side_view_fallback
        self.side_view_maximum_geometry_score = side_view_maximum_geometry_score
        self.side_view_minimum_spacing_to_radius = side_view_minimum_spacing_to_radius
        self.side_view_sample_radius_fraction = side_view_sample_radius_fraction
        self.side_view_refine_cross_axis_fraction = side_view_refine_cross_axis_fraction
        self.side_view_refine_along_axis_fraction = side_view_refine_along_axis_fraction
        self.side_view_chroma_saturation_min = side_view_chroma_saturation_min
        self.side_view_chroma_value_min = side_view_chroma_value_min
        self.side_view_minimum_chroma_fraction = side_view_minimum_chroma_fraction

    def detect(self, image: np.ndarray, expected_slots: int) -> ProductDetection:
        """Phát hiện slot, có thể thu gọn nội bộ để giữ FPS camera.

        Chỉ nhánh hình học chạy trên ảnh thu gọn; tọa độ/bán kính kết quả luôn
        được đổi về khung hình gốc. Vì vậy UI, ROI và đường đếm không bị lệch.
        """
        if not self._is_bgr_image(image):
            return self._detect_native(image, expected_slots)
        maximum = self.processing_max_dimension
        largest = max(image.shape[:2])
        if maximum is None or largest <= maximum:
            return self._detect_native(image, expected_slots)
        scale = maximum / largest
        resized = cv2.resize(
            image,
            (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        detected = self._detect_native(resized, expected_slots)
        if not detected.slots:
            return detected
        inverse = 1.0 / scale
        slots = [
            DetectedSlot(
                int(round(slot.x * inverse)),
                int(round(slot.y * inverse)),
                max(4, int(round(slot.radius * inverse))),
                max(4, int(round(slot.radius_x * inverse))) if slot.radius_x is not None else None,
                max(4, int(round(slot.radius_y * inverse))) if slot.radius_y is not None else None,
                slot.angle_deg,
                max(4, int(round(slot.sample_radius * inverse))) if slot.sample_radius is not None else None,
                slot.side_view,
                slot.matched_color,
            )
            for slot in detected.slots
        ]
        return ProductDetection(
            slots,
            detected.confidence,
            f"{detected.detail}; xử lý nhanh ở {resized.shape[1]}x{resized.shape[0]}",
        )

    def detect_partial(self, image: np.ndarray, expected_slots: int) -> ProductDetection:
        """Tìm một vỉ thiếu đúng một slot để caller có thể kết luận NG.

        Hàm này không được dùng thay cho ``detect`` trong đường PASS: nó chỉ
        được gọi sau khi không tìm được đủ slot. Vì vậy 4 ô của một vỉ 5 ô
        không bao giờ làm hạ một vỉ đầy đủ xuống NG. Cổng geometry vẫn chính
        là cổng của detector thông thường, chỉ thay số ô mục tiêu thành ``n-1``.
        """
        partial_slots = expected_slots - 1
        if partial_slots < 2:
            return ProductDetection([], 0.0, "Không đủ slot để xác nhận vỉ thiếu viên")
        detected = self.detect(image, partial_slots)
        if len(detected.slots) != partial_slots:
            return ProductDetection([], 0.0, detected.detail)
        return ProductDetection(
            detected.slots,
            detected.confidence,
            f"{detected.detail}; candidate thiếu 1/{expected_slots} slot",
        )

    def _detect_native(self, image: np.ndarray, expected_slots: int) -> ProductDetection:
        if expected_slots < 2:
            return ProductDetection([], 0.0, "Product Profile phải có từ 2 slot trở lên")
        if not self._is_bgr_image(image):
            return ProductDetection([], 0.0, "Ảnh đầu vào không đúng định dạng BGR uint8")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        shorter_edge = min(image.shape[:2])
        min_radius = max(self.minimum_slot_radius_px, round(shorter_edge * self.minimum_slot_radius_fraction))
        max_radius = max(min_radius + 2, round(shorter_edge * self.maximum_slot_radius_fraction))
        min_distance = max(round(shorter_edge * self.minimum_slot_distance_fraction), min_radius * 2)

        # Hough gốc là nguồn bảo thủ/nhanh nhất cho realtime.
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        hough_candidates = self._hough_candidates(blurred, min_distance, min_radius, max_radius)
        fit = self._best_safe_fit(hough_candidates, expected_slots)
        if fit is not None:
            return self._to_detection(fit, "Hough circle")

        # Ở góc nhìn cạnh, các vách nhựa làm Hough sinh vòng tròn lớn chồng
        # nhau nhưng tâm vẫn xếp thành một chuỗi rất đều.  Căn lại từng tâm về
        # vùng có màu bão hòa rồi để validator kiểm tra từng viên/màu.
        if self.enable_side_view_fallback:
            side_fit = self._best_side_view_fit(hough_candidates, expected_slots)
            if side_fit is not None:
                return self._to_side_view_detection(image, side_fit)

        # Vỉ hơi nghiêng có thể mất vòng tròn nhưng vẫn có contour vật liệu màu.
        # Đây chỉ là cổng hình học theo saturation, không phân loại màu/PASS.
        if self.enable_chroma_contour_fallback:
            fit = self._best_safe_fit(
                self._chroma_ellipse_candidates(image, min_radius, max_radius),
                expected_slots,
            )
            if fit is not None:
                return self._to_detection(fit, "contour vùng màu")

        # CLAHE là fallback cuối để không tăng false candidate ở ảnh đủ sáng.
        if self.enable_clahe_fallback:
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=(self.clahe_tile_grid_size, self.clahe_tile_grid_size),
            ).apply(gray)
            enhanced = cv2.GaussianBlur(clahe, (9, 9), 2)
            fit = self._best_safe_fit(
                self._hough_candidates(enhanced, min_distance, min_radius, max_radius),
                expected_slots,
            )
            if fit is not None:
                return self._to_detection(fit, "Hough + CLAHE")

        return ProductDetection(
            [],
            0.0,
            "Không tìm đủ slot rõ ràng; góc nhìn quá xiên, bị che hoặc hình học chưa đủ tin cậy",
        )

    def _hough_candidates(
        self,
        image_gray: np.ndarray,
        minimum_distance: int,
        minimum_radius: int,
        maximum_radius: int,
    ) -> np.ndarray:
        circles = cv2.HoughCircles(
            image_gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=minimum_distance,
            param1=self.hough_param1,
            param2=self.hough_param2,
            minRadius=minimum_radius,
            maxRadius=maximum_radius,
        )
        if circles is None:
            return np.empty((0, 6), dtype=float)
        raw = np.round(circles[0]).astype(float)
        radii = raw[:, 2:3]
        return np.column_stack((raw, radii, radii, np.zeros((len(raw), 1), dtype=float)))

    def _chroma_ellipse_candidates(
        self,
        image_bgr: np.ndarray,
        minimum_radius: int,
        maximum_radius: int,
    ) -> np.ndarray:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask = np.where(
            (hsv[:, :, 1] >= self.chroma_saturation_min) & (hsv[:, :, 2] >= self.chroma_value_min),
            255,
            0,
        ).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        minimum_area = pi * (minimum_radius * self.contour_min_radius_factor) ** 2
        maximum_area = pi * (maximum_radius * 1.35) ** 2
        candidates: list[tuple[float, list[float]]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if len(contour) < 5 or not minimum_area <= area <= maximum_area:
                continue
            (x, y), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
            major = max(axis_a, axis_b) / 2.0
            minor = min(axis_a, axis_b) / 2.0
            if minor <= 0 or major <= 0:
                continue
            aspect = minor / major
            effective_radius = float(np.sqrt(major * minor))
            if (
                aspect < self.minimum_ellipse_axis_ratio
                or effective_radius < minimum_radius * self.contour_min_radius_factor
                or effective_radius > maximum_radius * 1.35
            ):
                continue
            # Chuẩn hoá ``angle`` về trục bán kính lớn cho cv2.ellipse.
            if axis_b > axis_a:
                angle = (angle + 90.0) % 180.0
            candidates.append((area, [x, y, effective_radius, major, minor, angle]))
        candidates.sort(key=lambda item: item[0], reverse=True)
        if not candidates:
            return np.empty((0, 6), dtype=float)
        return np.asarray([candidate for _, candidate in candidates], dtype=float)

    def _best_safe_fit(self, candidates: np.ndarray, expected_slots: int) -> _RowFit | None:
        if len(candidates) < expected_slots:
            return None
        fit = self._best_perspective_row(candidates[:self.candidate_limit], expected_slots)
        if fit is None:
            return None
        if fit.score > self.maximum_geometry_score:
            return None
        if fit.projective_error > self.maximum_projective_spacing_error:
            return None
        if fit.line_error > self.maximum_line_deviation_ratio:
            return None
        # Circle giả ở ảnh chụp cạnh thường chồng lấn mạnh trên cạnh nhựa.
        if fit.minimum_spacing_ratio < self.minimum_spacing_to_radius:
            return None
        return fit

    def _to_detection(self, fit: _RowFit, source: str) -> ProductDetection:
        slots: list[DetectedSlot] = []
        for candidate in fit.selected:
            x, y, radius = candidate[:3]
            radius_x = candidate[3] if len(candidate) > 3 else radius
            radius_y = candidate[4] if len(candidate) > 4 else radius
            angle = candidate[5] if len(candidate) > 5 else 0.0
            slots.append(
                DetectedSlot(
                    int(round(x)),
                    int(round(y)),
                    max(4, int(round(radius))),
                    max(4, int(round(radius_x))),
                    max(4, int(round(radius_y))),
                    float(angle),
                )
            )
        confidence = max(0.85, min(0.96, 0.96 - fit.score * 0.15))
        return ProductDetection(
            slots,
            confidence,
            f"Đã nhận diện {len(slots)} slot ({source}), geometry score={fit.score:.3f}, "
            f"perspective error={fit.projective_error:.3f}",
        )

    def _best_side_view_fit(self, candidates: np.ndarray, expected_slots: int) -> _RowFit | None:
        """Chấp nhận chuỗi tâm Hough chồng lấn chỉ khi trục/spacing rất rõ."""
        if len(candidates) < expected_slots:
            return None
        fit = self._best_perspective_row(candidates[:self.candidate_limit], expected_slots)
        if fit is None:
            return None
        if fit.score > self.side_view_maximum_geometry_score:
            return None
        if fit.projective_error > self.maximum_projective_spacing_error:
            return None
        if fit.line_error > self.maximum_line_deviation_ratio:
            return None
        if fit.minimum_spacing_ratio < self.side_view_minimum_spacing_to_radius:
            return None
        return fit

    def _to_side_view_detection(self, image_bgr: np.ndarray, fit: _RowFit) -> ProductDetection:
        centers = fit.selected[:, :2]
        center = centers.mean(axis=0)
        _, _, vectors = np.linalg.svd(centers - center, full_matrices=False)
        axis = vectors[0]
        dominant_axis = 0 if abs(axis[0]) >= abs(axis[1]) else 1
        if axis[dominant_axis] < 0:
            axis = -axis
        projection = (centers - center) @ axis
        spacing = np.diff(np.sort(projection))
        typical_spacing = float(np.median(spacing))
        sample_radius = max(6, int(round(np.min(spacing) * self.side_view_sample_radius_fraction)))
        slot_radius = max(sample_radius + 4, int(round(typical_spacing * 0.36)))
        refined_centers = self._refine_side_view_centers(
            image_bgr,
            fit.selected[:, :2],
            axis,
            typical_spacing,
            sample_radius,
        )
        slots = [
            DetectedSlot(
                int(x), int(y), slot_radius, slot_radius, slot_radius, 0.0,
                sample_radius=sample_radius,
                side_view=True,
            )
            for x, y in refined_centers
        ]
        confidence = max(0.85, min(0.93, 0.94 - fit.score * 0.20))
        return ProductDetection(
            slots,
            confidence,
            f"Đã nhận diện {len(slots)} slot (Hough side-view + căn vùng màu), "
            f"geometry score={fit.score:.3f}, perspective error={fit.projective_error:.3f}",
        )

    def _refine_side_view_centers(
        self,
        image_bgr: np.ndarray,
        centers: np.ndarray,
        axis: np.ndarray,
        typical_spacing: float,
        sample_radius: int,
    ) -> list[tuple[int, int]]:
        """Dịch tâm về phần viên có màu, tránh vùng vách nhựa trong suốt."""
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        # Các màu đã hỗ trợ (xanh lá, xanh dương, tím) nằm trong khoảng hue này;
        # loại màu nâu của thùng carton và vùng nhựa nhạt để tâm không trôi.
        chroma = (
            (hsv[:, :, 0] >= 35)
            & (hsv[:, :, 0] <= 170)
            & (hsv[:, :, 1] >= self.side_view_chroma_saturation_min)
            & (hsv[:, :, 2] >= self.side_view_chroma_value_min)
        ).astype(np.uint8)
        integral = cv2.integral(chroma, sdepth=cv2.CV_32S)
        normal = np.array([-axis[1], axis[0]])
        cross_limit = max(0, int(round(typical_spacing * self.side_view_refine_cross_axis_fraction)))
        along_limit = max(0, int(round(typical_spacing * self.side_view_refine_along_axis_fraction)))
        step = max(4, int(round(sample_radius * 0.45)))
        height, width = chroma.shape
        window_area = (sample_radius * 2 + 1) ** 2
        refined: list[tuple[int, int]] = []
        for initial in centers:
            best_score = -1
            best_point = tuple(int(round(value)) for value in initial)
            for cross_offset in range(-cross_limit, cross_limit + 1, step):
                for along_offset in range(-along_limit, along_limit + 1, step):
                    point = initial + normal * cross_offset + axis * along_offset
                    x, y = (int(round(point[0])), int(round(point[1])))
                    left, top = x - sample_radius, y - sample_radius
                    right, bottom = x + sample_radius + 1, y + sample_radius + 1
                    if left < 0 or top < 0 or right > width or bottom > height:
                        continue
                    score = int(
                        integral[bottom, right] - integral[top, right]
                        - integral[bottom, left] + integral[top, left]
                    )
                    if score > best_score:
                        best_score, best_point = score, (x, y)
            if best_score >= window_area * self.side_view_minimum_chroma_fraction:
                refined.append(best_point)
            else:
                refined.append(tuple(int(round(value)) for value in initial))
        return refined

    def _best_perspective_row(self, candidates: np.ndarray, count: int) -> _RowFit | None:
        best: _RowFit | None = None
        for indexes in combinations(range(len(candidates)), count):
            selected = candidates[list(indexes)]
            centers = selected[:, :2]
            center = centers.mean(axis=0)
            _, _, vectors = np.linalg.svd(centers - center, full_matrices=False)
            axis, normal = vectors[0], vectors[1]
            dominant_axis = 0 if abs(axis[0]) >= abs(axis[1]) else 1
            if axis[dominant_axis] < 0:
                axis, normal = -axis, -normal
            projection = (centers - center) @ axis
            order = np.argsort(projection)
            projection = projection[order]
            ordered = selected[order]
            spacing = np.diff(projection)
            if np.any(spacing < 1):
                continue
            mean_radius = float(np.mean(ordered[:, 2]))
            if mean_radius <= 0:
                continue
            perpendicular = np.abs((centers - center) @ normal)
            line_error = float(np.mean(perpendicular) / mean_radius)
            projective_error = self._projective_spacing_error(projection)
            radius_cv = float(np.std(ordered[:, 2]) / mean_radius)
            minimum_spacing_ratio = float(np.min(spacing) / mean_radius)
            score = line_error + (2.0 * projective_error) + (0.35 * radius_cv)
            candidate = _RowFit(score, ordered, projective_error, line_error, minimum_spacing_ratio)
            if best is None or candidate.score < best.score:
                best = candidate
        return best

    @staticmethod
    def _projective_spacing_error(projection: np.ndarray) -> float:
        """Fit homography 1-D cho dãy slot cách đều trong không gian thật."""
        values = np.asarray(projection, dtype=float)
        if len(values) < 3 or np.any(np.diff(values) <= 0):
            return float("inf")
        index = np.arange(len(values), dtype=float)
        matrix = np.column_stack((index, np.ones_like(index), -values * index, -values))
        try:
            _, _, vh = np.linalg.svd(matrix)
        except np.linalg.LinAlgError:
            return float("inf")
        a, b, c, d = vh[-1]
        denominator = c * index + d
        if np.any(np.abs(denominator) < 1e-8):
            return float("inf")
        predicted = (a * index + b) / denominator
        return float(np.sqrt(np.mean((predicted - values) ** 2)) / max(float(np.ptp(values)), 1.0))

    @staticmethod
    def _best_regular_row(circles: np.ndarray, count: int) -> tuple[float, np.ndarray | None]:
        """API cũ giữ lại cho test/khách gọi cần score spacing theo pixel."""
        best_score = float("inf")
        best_selected: np.ndarray | None = None
        for indexes in combinations(range(len(circles)), count):
            selected = circles[list(indexes)]
            centers = selected[:, :2]
            center = centers.mean(axis=0)
            _, _, vectors = np.linalg.svd(centers - center, full_matrices=False)
            axis, normal = vectors[0], vectors[1]
            dominant_axis = 0 if abs(axis[0]) >= abs(axis[1]) else 1
            if axis[dominant_axis] < 0:
                axis, normal = -axis, -normal
            projection = (centers - center) @ axis
            ordered = selected[np.argsort(projection)]
            spacing = np.diff(np.sort(projection))
            if np.any(spacing < 1):
                continue
            perpendicular = np.abs((centers - center) @ normal)
            mean_radius = selected[:, 2].mean()
            score = (
                spacing.std() / spacing.mean()
                + perpendicular.mean() / max(mean_radius, 1.0)
                + selected[:, 2].std() / max(mean_radius, 1.0)
            )
            if score < best_score:
                best_score, best_selected = score, ordered
        return best_score, best_selected

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
