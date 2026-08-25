from __future__ import annotations

from dataclasses import dataclass

from detection import DetectionMetrics, DetectionState, ProductDetector
from sources import FramePacket
from .product_tracker import ProductTracker, TrackerEvent


@dataclass(frozen=True)
class PipelineResult:
    packet: FramePacket
    metrics: DetectionMetrics
    event: TrackerEvent


class InspectionPipeline:
    def __init__(self, detector: ProductDetector, tracker: ProductTracker) -> None:
        self.detector = detector
        self.tracker = tracker

    def reset(self) -> None:
        self.detector.reset()
        self.tracker.reset()

    def process(self, packet: FramePacket, zone: tuple[float, float, float, float] | None) -> PipelineResult | None:
        if zone is None:
            return None
        metrics = self.detector.detect(packet.image, zone)
        event = self.tracker.update(metrics, packet.frame_number)
        return PipelineResult(packet, metrics, event)
