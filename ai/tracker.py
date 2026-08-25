"""Tracking ID và chống đếm trùng; không tăng counter theo frame."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import hypot

from core.models import ProductStatus, ValidationResult


class CountingDirection(StrEnum):
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"
    TOP_TO_BOTTOM = "top_to_bottom"
    BOTTOM_TO_TOP = "bottom_to_top"


class TrackLifecycle(StrEnum):
    DETECTED = "DETECTED"
    TRACKING = "TRACKING"
    VALIDATED = "VALIDATED"
    COUNTED = "COUNTED"
    REJECTED = "REJECTED"


BoundingBox = tuple[int, int, int, int]


@dataclass(slots=True)
class TrackingInput:
    bounding_box: BoundingBox
    validation: ValidationResult

    @property
    def center(self) -> tuple[float, float]:
        x, y, width, height = self.bounding_box
        return x + width / 2, y + height / 2


@dataclass(slots=True)
class Track:
    track_id: int
    bounding_box: BoundingBox
    center: tuple[float, float]
    validation: ValidationResult
    hits: int = 1
    missed_frames: int = 0
    valid_streak: int = 0
    origin_side: bool = False
    crossed_line: bool = False
    counted: bool = False
    resolved: bool = False
    counted_this_update: bool = False
    rejected_this_update: bool = False
    lifecycle: TrackLifecycle = TrackLifecycle.DETECTED


@dataclass(slots=True)
class TrackingUpdate:
    tracks: list[Track] = field(default_factory=list)
    newly_counted: list[Track] = field(default_factory=list)
    crossed_not_counted: list[Track] = field(default_factory=list)


class CentroidTracker:
    """Tracker nhẹ, bảo thủ, phù hợp làm adapter trước ByteTrack/BoT-SORT."""

    def __init__(self, max_age: int = 40, min_validation_hits: int = 1,
                 distance_threshold: float = 320.0,
                 first_seen_crossing_margin_fraction: float = 0.75) -> None:
        self.max_age = max_age
        self.min_validation_hits = max(1, min_validation_hits)
        self.distance_threshold = distance_threshold
        self.first_seen_crossing_margin_fraction = max(0.0, first_seen_crossing_margin_fraction)
        self._next_id = 1
        self._tracks: dict[int, Track] = {}

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()

    def update(self, detections: list[TrackingInput], line_position: float,
               direction: CountingDirection) -> TrackingUpdate:
        """Gắn detection vào track và chỉ phát sự kiện count duy nhất một lần."""
        unmatched_tracks = set(self._tracks)
        unmatched_detections = set(range(len(detections)))
        pairs: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for index, detection in enumerate(detections):
                distance = hypot(track.center[0] - detection.center[0], track.center[1] - detection.center[1])
                if distance <= self.distance_threshold:
                    pairs.append((distance, track_id, index))
        for _, track_id, index in sorted(pairs):
            if track_id not in unmatched_tracks or index not in unmatched_detections:
                continue
            self._update_track(self._tracks[track_id], detections[index], line_position, direction)
            unmatched_tracks.remove(track_id); unmatched_detections.remove(index)
        for track_id in unmatched_tracks:
            self._tracks[track_id].missed_frames += 1
        for index in unmatched_detections:
            detection = detections[index]
            track = Track(self._next_id, detection.bounding_box, detection.center, detection.validation)
            track.valid_streak = int(detection.validation.status is ProductStatus.PASS)
            if track.valid_streak >= self.min_validation_hits:
                # Ở tốc độ băng tải cao chỉ còn hai frame nhìn rõ trước/sau
                # vạch. Frame đầu PASS đã đủ là một xác thực, không phải đợi
                # thêm hai frame rồi bỏ lỡ vỉ.
                track.lifecycle = TrackLifecycle.VALIDATED
            # Đếm theo mép đi đầu, không chỉ tâm hộp. Vỉ chạy nhanh có thể
            # mất nhận diện ngay sau khi tâm qua vạch, trong khi mép trước đã
            # cắt vạch ở frame PASS cuối cùng.
            track.origin_side = self._leading_edge_on_origin_side(
                track.bounding_box, line_position, direction,
            )
            if self._first_seen_at_counting_line(track.bounding_box, line_position, direction):
                # Worker dùng latest-frame buffer nên đôi lúc bỏ qua đúng
                # frame mà mép vỉ vừa qua vạch. Nếu vỉ lần đầu được nhìn thấy
                # ngay tại/sau vạch trong khoảng một chiều dài vỉ, giữ event ở
                # trạng thái pending để frame sau vẫn có thể chốt PASS hoặc NG.
                track.crossed_line = True
                self._resolve_after_crossing(track)
            self._tracks[track.track_id] = track
            self._next_id += 1
        # Không chốt REVIEW ngay khoảnh khắc mép vỉ cắt đường đếm. Với băng
        # tải nhanh, vài frame đầu thường chưa đủ màu/slot, nhưng những frame
        # kế tiếp mới chứng minh được lỗi sai màu hoặc thiếu viên. Chỉ chốt
        # UNKNOWN khi track đã rời khung; FAIL/PASS có thể được nâng cấp sau
        # khi đã qua vạch.
        expired = [
            track for track in self._tracks.values()
            if track.missed_frames > self.max_age
        ]
        for track in expired:
            if track.crossed_line and not track.resolved:
                track.resolved = True
                track.lifecycle = TrackLifecycle.REJECTED
                track.rejected_this_update = True
        self._tracks = {
            track_id: track for track_id, track in self._tracks.items()
            if track.missed_frames <= self.max_age
        }
        newly_counted = [track for track in self._tracks.values() if track.counted_this_update]
        crossed_not_counted = [
            *[track for track in self._tracks.values() if track.rejected_this_update],
            *[track for track in expired if track.rejected_this_update],
        ]
        for track in newly_counted:
            track.counted_this_update = False
        for track in crossed_not_counted:
            track.rejected_this_update = False
        return TrackingUpdate(list(self._tracks.values()), newly_counted, crossed_not_counted)

    def _update_track(self, track: Track, detection: TrackingInput, line_position: float,
                      direction: CountingDirection) -> None:
        was_origin_side = self._leading_edge_on_origin_side(
            track.bounding_box, line_position, direction,
        )
        track.bounding_box, track.center, track.validation = detection.bounding_box, detection.center, detection.validation
        track.hits += 1; track.missed_frames = 0
        if detection.validation.status is ProductStatus.PASS:
            track.valid_streak += 1
            if track.valid_streak >= self.min_validation_hits and not track.resolved:
                track.lifecycle = TrackLifecycle.VALIDATED
        else:
            track.valid_streak = 0
            if not track.counted:
                track.lifecycle = TrackLifecycle.TRACKING
        now_origin_side = self._leading_edge_on_origin_side(
            track.bounding_box, line_position, direction,
        )
        if track.origin_side and was_origin_side and not now_origin_side and not track.crossed_line:
            track.crossed_line = True
        if track.crossed_line and not track.resolved:
            self._resolve_after_crossing(track)

    @staticmethod
    def _resolve_after_crossing(track: Track) -> None:
        """Chốt PASS/NG khi có bằng chứng, không biến lỗi thành REVIEW sớm.

        ``FAIL`` là bằng chứng dứt khoát (sai màu hoặc thiếu slot), nên được
        ghi NG ngay cả khi frame trước lúc qua vạch còn UNKNOWN. ``PASS`` vẫn
        cần đủ chuỗi xác thực để tránh tăng đếm nhầm.
        """
        if track.validation.status is ProductStatus.FAIL:
            track.resolved = True
            track.lifecycle = TrackLifecycle.REJECTED
            track.rejected_this_update = True
            return
        if track.lifecycle is TrackLifecycle.VALIDATED:
            track.counted = True
            track.resolved = True
            track.lifecycle = TrackLifecycle.COUNTED
            track.counted_this_update = True

    def _first_seen_at_counting_line(
        self,
        bounding_box: BoundingBox,
        line_position: float,
        direction: CountingDirection,
    ) -> bool:
        """Bắt lượt đầu xuất hiện ngay cạnh vạch khi một vài frame bị bỏ qua.

        Không nhận các vỉ đã ở xa phía sau vạch: cửa sổ chỉ rộng tối đa một
        phần kích thước vỉ, đủ cho một nhịp inference bị trễ nhưng tránh đếm
        lại vật đứng yên trong vùng quan sát.
        """
        x, y, width, height = bounding_box
        if direction is CountingDirection.LEFT_TO_RIGHT:
            return x + width >= line_position and x <= line_position + width * self.first_seen_crossing_margin_fraction
        if direction is CountingDirection.RIGHT_TO_LEFT:
            return x <= line_position and x + width >= line_position - width * self.first_seen_crossing_margin_fraction
        if direction is CountingDirection.TOP_TO_BOTTOM:
            return y + height >= line_position and y <= line_position + height * self.first_seen_crossing_margin_fraction
        return y <= line_position and y + height >= line_position - height * self.first_seen_crossing_margin_fraction

    @staticmethod
    def _is_origin_side(center: tuple[float, float], line_position: float,
                        direction: CountingDirection) -> bool:
        x, y = center
        return {
            CountingDirection.LEFT_TO_RIGHT: x < line_position,
            CountingDirection.RIGHT_TO_LEFT: x > line_position,
            CountingDirection.TOP_TO_BOTTOM: y < line_position,
            CountingDirection.BOTTOM_TO_TOP: y > line_position,
        }[direction]

    @staticmethod
    def _leading_edge_on_origin_side(
        bounding_box: BoundingBox,
        line_position: float,
        direction: CountingDirection,
    ) -> bool:
        """Kiểm tra mép đi đầu còn ở phía xuất phát của đường đếm hay không."""
        x, y, width, height = bounding_box
        return {
            CountingDirection.LEFT_TO_RIGHT: x + width < line_position,
            CountingDirection.RIGHT_TO_LEFT: x > line_position,
            CountingDirection.TOP_TO_BOTTOM: y + height < line_position,
            CountingDirection.BOTTOM_TO_TOP: y > line_position,
        }[direction]
