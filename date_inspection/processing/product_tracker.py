from __future__ import annotations

from dataclasses import dataclass
import time

from detection import DetectionMetrics, DetectionState


@dataclass(frozen=True)
class TrackerEvent:
    state: DetectionState
    product_id: str | None = None
    triggered: bool = False
    duplicate_blocked: bool = False


class ProductTracker:
    """Máy trạng thái chỉ phát sinh đúng một trigger rồi bắt buộc chờ sản phẩm rời vùng."""
    def __init__(self, minimum_present: int = 3, minimum_absent: int = 4, debounce_ms: int = 300,
                 minimum_gap_frames: int = 8) -> None:
        self.minimum_present = max(1, minimum_present)
        self.minimum_absent = max(1, minimum_absent)
        self.debounce_ms = max(0, debounce_ms)
        self.minimum_gap_frames = max(0, minimum_gap_frames)
        self.reset()

    def reset(self) -> None:
        self.state = DetectionState.IDLE
        self._present_count = 0
        self._absent_count = 0
        self._last_trigger_time = -float("inf")
        self._last_exit_frame = -10**9
        self._sequence = 0

    def update(self, metrics: DetectionMetrics, frame_number: int, now: float | None = None) -> TrackerEvent:
        now = time.monotonic() if now is None else now
        if metrics.present:
            self._present_count += 1
            self._absent_count = 0
        else:
            self._absent_count += 1
            self._present_count = 0

        if self.state == DetectionState.IDLE:
            if metrics.present:
                self.state = DetectionState.ENTERING
                if self._present_count >= self.minimum_present:
                    if (frame_number - self._last_exit_frame < self.minimum_gap_frames or
                            (now - self._last_trigger_time) * 1000 < self.debounce_ms):
                        return TrackerEvent(self.state, duplicate_blocked=True)
                    self._sequence += 1
                    self._last_trigger_time = now
                    self.state = DetectionState.PRESENT
                    return TrackerEvent(self.state, f"P{self._sequence:06d}", triggered=True)
            return TrackerEvent(self.state)

        if self.state == DetectionState.ENTERING:
            if not metrics.present:
                self.state = DetectionState.IDLE
            elif self._present_count >= self.minimum_present:
                if (frame_number - self._last_exit_frame < self.minimum_gap_frames or
                        (now - self._last_trigger_time) * 1000 < self.debounce_ms):
                    return TrackerEvent(self.state, duplicate_blocked=True)
                self._sequence += 1
                self._last_trigger_time = now
                self.state = DetectionState.PRESENT
                return TrackerEvent(self.state, f"P{self._sequence:06d}", triggered=True)
            return TrackerEvent(self.state)

        if self.state in (DetectionState.PRESENT, DetectionState.CAPTURED):
            if not metrics.present:
                self.state = DetectionState.EXITING
            elif self.state == DetectionState.PRESENT:
                self.state = DetectionState.CAPTURED
            return TrackerEvent(self.state)

        if self.state == DetectionState.EXITING:
            if metrics.present:
                # Bóng rung ngay sau lúc ra không thể tạo product mới: trở lại product cũ.
                self.state = DetectionState.CAPTURED
            elif self._absent_count >= self.minimum_absent:
                self.state = DetectionState.IDLE
                self._last_exit_frame = frame_number
            return TrackerEvent(self.state)
        return TrackerEvent(self.state)
