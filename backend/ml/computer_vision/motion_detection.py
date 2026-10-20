
"""
app/ai/cv/motion_detection.py

Industrial Motion Detection & Monitoring
========================================

Python 3.11+

Features:
- Real-time camera motion detection
- MOG2 background subtraction
- Frame-difference detection
- Moving object bounding boxes
- Centroid-based object tracking
- Restricted-zone motion monitoring
- Motion event generation
- Video and webcam processing
- Annotated video frames
- Motion area percentage
- Async frame processing
- Tenant/camera/equipment metadata
- JSON-compatible detection results

Dependencies:
    pip install numpy opencv-python

IMPORTANT:
    Motion detection does not identify people,
    machines, vehicles or object intent.

    Shadows, camera vibration, lighting changes,
    reflections and environmental motion may
    trigger false positives.

    This module must not be the sole basis for
    machine guarding, emergency stops, access
    control or worker safety decisions.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class MotionDetectionError(Exception):
    """Base motion detection exception."""


class MotionValidationError(MotionDetectionError):
    """Invalid input or configuration."""


class MotionProcessingError(MotionDetectionError):
    """Video or frame processing failure."""


# ============================================================
# ENUMS
# ============================================================

class MotionDetectionMode(str, Enum):
    MOG2 = "mog2"
    FRAME_DIFFERENCE = "frame_difference"


class MotionLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class MotionEventType(str, Enum):
    MOTION_DETECTED = "motion_detected"
    RESTRICTED_ZONE_MOTION = (
        "restricted_zone_motion"
    )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(slots=True)
class MotionDetectionConfig:
    mode: MotionDetectionMode = (
        MotionDetectionMode.MOG2
    )

    min_image_width: int = 64
    min_image_height: int = 64
    max_image_pixels: int = 20_000_000

    blur_kernel_size: int = 5
    difference_threshold: int = 25

    morphology_kernel_size: int = 5
    min_motion_area_px: int = 250

    mog2_history: int = 500
    mog2_var_threshold: float = 16.0
    detect_shadows: bool = True

    max_tracking_distance_px: float = 80.0
    max_missing_frames: int = 10

    moderate_motion_percent: float = 2.0
    high_motion_percent: float = 10.0

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            self.mode = MotionDetectionMode(
                self.mode
            )

        positive_fields = (
            "min_image_width",
            "min_image_height",
            "max_image_pixels",
            "blur_kernel_size",
            "morphology_kernel_size",
            "min_motion_area_px",
            "mog2_history",
            "max_missing_frames",
        )

        for name in positive_fields:
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise MotionValidationError(
                    f"{name} must be a positive integer"
                )

        if self.blur_kernel_size % 2 == 0:
            raise MotionValidationError(
                "blur_kernel_size must be odd"
            )

        if not (
            0 <= self.difference_threshold <= 255
        ):
            raise MotionValidationError(
                "Invalid difference_threshold"
            )

        if self.max_tracking_distance_px <= 0:
            raise MotionValidationError(
                "max_tracking_distance_px must be positive"
            )

        if not (
            0
            < self.moderate_motion_percent
            < self.high_motion_percent
            <= 100
        ):
            raise MotionValidationError(
                "Invalid motion thresholds"
            )


# ============================================================
# DATA SCHEMAS
# ============================================================

@dataclass(slots=True)
class RestrictedZone:
    zone_id: str
    name: str

    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "bounding_box": {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
            },
        }


@dataclass(slots=True)
class MotionObject:
    object_id: str

    x: int
    y: int
    width: int
    height: int

    centroid_x: float
    centroid_y: float

    pixel_area: int

    zone_ids: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "bounding_box": {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
            },
            "centroid": {
                "x": self.centroid_x,
                "y": self.centroid_y,
            },
            "pixel_area": self.pixel_area,
            "zone_ids": list(self.zone_ids),
        }


@dataclass(slots=True)
class MotionEvent:
    event_id: str
    event_type: str
    camera_id: str
    company_id: str

    object_id: str | None
    zone_id: str | None

    detected_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "camera_id": self.camera_id,
            "company_id": self.company_id,
            "object_id": self.object_id,
            "zone_id": self.zone_id,
            "detected_at": (
                self.detected_at.isoformat()
            ),
        }


@dataclass(slots=True)
class MotionDetectionResult:
    inspection_id: str

    company_id: str
    camera_id: str
    equipment_id: str | None

    motion_detected: bool
    motion_level: str
    motion_area_percent: float

    image_width: int
    image_height: int

    objects: list[MotionObject]
    events: list[MotionEvent]

    restricted_zone_motion_detected: bool

    inspected_at: datetime

    requires_human_review: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "inspection_id": self.inspection_id,
            "company_id": self.company_id,
            "camera_id": self.camera_id,
            "equipment_id": self.equipment_id,
            "motion_detected": (
                self.motion_detected
            ),
            "motion_level": self.motion_level,
            "motion_area_percent": (
                self.motion_area_percent
            ),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "objects": [
                item.to_dict()
                for item in self.objects
            ],
            "events": [
                event.to_dict()
                for event in self.events
            ],
            "restricted_zone_motion_detected": (
                self.restricted_zone_motion_detected
            ),
            "inspected_at": (
                self.inspected_at.isoformat()
            ),
            "requires_human_review": (
                self.requires_human_review
            ),
            "metadata": dict(self.metadata),
        }


# ============================================================
# OBJECT TRACKER
# ============================================================

@dataclass(slots=True)
class _Track:
    track_id: str
    x: float
    y: float
    missing_frames: int = 0


class CentroidTracker:
    """
    Simple nearest-centroid tracker.

    Not a replacement for DeepSORT/ByteTrack.

    IDs can switch when objects cross or
    become occluded.
    """

    def __init__(
        self,
        *,
        max_distance: float = 80.0,
        max_missing_frames: int = 10,
    ) -> None:
        self.max_distance = max_distance
        self.max_missing_frames = (
            max_missing_frames
        )

        self._tracks: dict[str, _Track] = {}

    def update(
        self,
        objects: list[MotionObject],
    ) -> list[MotionObject]:
        unmatched_tracks = set(
            self._tracks.keys()
        )

        assigned_tracks: set[str] = set()

        for obj in objects:
            best_track_id: str | None = None
            best_distance = float("inf")

            for track_id in unmatched_tracks:
                if track_id in assigned_tracks:
                    continue

                track = self._tracks[track_id]

                distance = math.hypot(
                    obj.centroid_x - track.x,
                    obj.centroid_y - track.y,
                )

                if (
                    distance < best_distance
                    and distance <= self.max_distance
                ):
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is None:
                new_id = str(uuid.uuid4())

                self._tracks[new_id] = _Track(
                    track_id=new_id,
                    x=obj.centroid_x,
                    y=obj.centroid_y,
                )

                obj.object_id = new_id

            else:
                track = self._tracks[
                    best_track_id
                ]

                track.x = obj.centroid_x
                track.y = obj.centroid_y
                track.missing_frames = 0

                obj.object_id = best_track_id

                assigned_tracks.add(
                    best_track_id
                )

                unmatched_tracks.discard(
                    best_track_id
                )

        for track_id in list(
            unmatched_tracks
        ):
            track = self._tracks.get(
                track_id
            )

            if track is None:
                continue

            track.missing_frames += 1

            if (
                track.missing_frames
                > self.max_missing_frames
            ):
                del self._tracks[track_id]

        return objects

    def reset(self) -> None:
        self._tracks.clear()


# ============================================================
# MOTION DETECTOR
# ============================================================

class MotionDetector:
    """
    Stateful motion detector.

    One instance should normally be used per
    camera stream.

    Background models and trackers are
    stream-specific.
    """

    def __init__(
        self,
        config: MotionDetectionConfig | None = None,
        *,
        restricted_zones: Sequence[
            RestrictedZone
        ] | None = None,
    ) -> None:
        self.config = (
            config
            or MotionDetectionConfig()
        )

        self.restricted_zones = list(
            restricted_zones or []
        )

        self._lock = threading.RLock()

        self._previous_gray: (
            np.ndarray | None
        ) = None

        self._background_subtractor = (
            cv2.createBackgroundSubtractorMOG2(
                history=self.config.mog2_history,
                varThreshold=(
                    self.config.mog2_var_threshold
                ),
                detectShadows=(
                    self.config.detect_shadows
                ),
            )
        )

        self._tracker = CentroidTracker(
            max_distance=(
                self.config.max_tracking_distance_px
            ),
            max_missing_frames=(
                self.config.max_missing_frames
            ),
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_image(
        self,
        image: np.ndarray,
    ) -> None:
        if not isinstance(
            image,
            np.ndarray,
        ):
            raise MotionValidationError(
                "Image must be a NumPy array"
            )

        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
        ):
            raise MotionValidationError(
                "Expected uint8 BGR image"
            )

        height, width = image.shape[:2]

        if (
            width < self.config.min_image_width
            or height < self.config.min_image_height
        ):
            raise MotionValidationError(
                "Image dimensions too small"
            )

        if (
            width * height
            > self.config.max_image_pixels
        ):
            raise MotionValidationError(
                "Image exceeds pixel limit"
            )

    @staticmethod
    def _validate_identity(
        company_id: str,
        camera_id: str,
    ) -> None:
        if (
            not isinstance(company_id, str)
            or not company_id.strip()
        ):
            raise MotionValidationError(
                "company_id is required"
            )

        if (
            not isinstance(camera_id, str)
            or not camera_id.strip()
        ):
            raise MotionValidationError(
                "camera_id is required"
            )

    # ========================================================
    # FRAME DIFFERENCE
    # ========================================================

    def _frame_difference_mask(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        grayscale = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        size = self.config.blur_kernel_size

        grayscale = cv2.GaussianBlur(
            grayscale,
            (size, size),
            0,
        )

        if self._previous_gray is None:
            self._previous_gray = grayscale

            return np.zeros_like(
                grayscale
            )

        if (
            self._previous_gray.shape
            != grayscale.shape
        ):
            self._previous_gray = grayscale

            return np.zeros_like(
                grayscale
            )

        difference = cv2.absdiff(
            self._previous_gray,
            grayscale,
        )

        self._previous_gray = grayscale

        _, mask = cv2.threshold(
            difference,
            self.config.difference_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        return mask

    # ========================================================
    # MOG2 BACKGROUND SUBTRACTION
    # ========================================================

    def _mog2_mask(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        mask = (
            self._background_subtractor.apply(
                image
            )
        )

        # MOG2 uses 127 for shadows.
        # Keep only foreground value 255.
        _, binary = cv2.threshold(
            mask,
            200,
            255,
            cv2.THRESH_BINARY,
        )

        return binary

    # ========================================================
    # MASK CLEANING
    # ========================================================

    def _clean_mask(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        size = (
            self.config.morphology_kernel_size
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (size, size),
        )

        opened = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        return cv2.morphologyEx(
            opened,
            cv2.MORPH_CLOSE,
            kernel,
        )

    # ========================================================
    # OBJECT EXTRACTION
    # ========================================================

    def _extract_objects(
        self,
        mask: np.ndarray,
    ) -> list[MotionObject]:
        count, _, stats, centroids = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )
        )

        objects: list[MotionObject] = []

        for label in range(1, count):
            area = int(
                stats[
                    label,
                    cv2.CC_STAT_AREA,
                ]
            )

            if (
                area
                < self.config.min_motion_area_px
            ):
                continue

            x = int(
                stats[label, cv2.CC_STAT_LEFT]
            )

            y = int(
                stats[label, cv2.CC_STAT_TOP]
            )

            width = int(
                stats[label, cv2.CC_STAT_WIDTH]
            )

            height = int(
                stats[label, cv2.CC_STAT_HEIGHT]
            )

            centroid_x = float(
                centroids[label][0]
            )

            centroid_y = float(
                centroids[label][1]
            )

            objects.append(
                MotionObject(
                    object_id="",
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    centroid_x=centroid_x,
                    centroid_y=centroid_y,
                    pixel_area=area,
                )
            )

        return objects

    # ========================================================
    # RESTRICTED ZONES
    # ========================================================

    @staticmethod
    def _overlaps_zone(
        obj: MotionObject,
        zone: RestrictedZone,
    ) -> bool:
        x1 = max(
            obj.x,
            zone.x,
        )

        y1 = max(
            obj.y,
            zone.y,
        )

        x2 = min(
            obj.x + obj.width,
            zone.x + zone.width,
        )

        y2 = min(
            obj.y + obj.height,
            zone.y + zone.height,
        )

        return (
            x2 > x1
            and y2 > y1
        )

    def _assign_zones(
        self,
        objects: list[MotionObject],
    ) -> None:
        for obj in objects:
            obj.zone_ids = [
                zone.zone_id
                for zone in self.restricted_zones
                if self._overlaps_zone(
                    obj,
                    zone,
                )
            ]

    # ========================================================
    # MOTION LEVEL
    # ========================================================

    def _motion_level(
        self,
        area_percent: float,
    ) -> MotionLevel:
        if area_percent <= 0:
            return MotionLevel.NONE

        if (
            area_percent
            < self.config.moderate_motion_percent
        ):
            return MotionLevel.LOW

        if (
            area_percent
            < self.config.high_motion_percent
        ):
            return MotionLevel.MODERATE

        return MotionLevel.HIGH

    # ========================================================
    # FRAME INSPECTION
    # ========================================================

    def detect(
        self,
        image: np.ndarray,
        *,
        company_id: str,
        camera_id: str,
        equipment_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MotionDetectionResult:
        self._validate_identity(
            company_id,
            camera_id,
        )

        self._validate_image(image)

        with self._lock:
            if (
                self.config.mode
                == MotionDetectionMode.MOG2
            ):
                raw_mask = self._mog2_mask(
                    image
                )

            else:
                raw_mask = (
                    self._frame_difference_mask(
                        image
                    )
                )

            mask = self._clean_mask(
                raw_mask
            )

            objects = self._extract_objects(
                mask
            )

            objects = self._tracker.update(
                objects
            )

            self._assign_zones(
                objects
            )

        height, width = image.shape[:2]

        # Count accepted object pixels, rather
        # than all thresholded noise pixels.
        accepted_area = sum(
            obj.pixel_area
            for obj in objects
        )

        area_percent = (
            accepted_area
            / (width * height)
            * 100.0
        )

        level = self._motion_level(
            area_percent
        )

        now = datetime.now(
            timezone.utc
        )

        events: list[MotionEvent] = []

        for obj in objects:
            events.append(
                MotionEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=(
                        MotionEventType
                        .MOTION_DETECTED.value
                    ),
                    camera_id=camera_id,
                    company_id=company_id,
                    object_id=obj.object_id,
                    zone_id=None,
                    detected_at=now,
                )
            )

            for zone_id in obj.zone_ids:
                events.append(
                    MotionEvent(
                        event_id=str(uuid.uuid4()),
                        event_type=(
                            MotionEventType
                            .RESTRICTED_ZONE_MOTION
                            .value
                        ),
                        camera_id=camera_id,
                        company_id=company_id,
                        object_id=obj.object_id,
                        zone_id=zone_id,
                        detected_at=now,
                    )
                )

        restricted_motion = any(
            obj.zone_ids
            for obj in objects
        )

        return MotionDetectionResult(
            inspection_id=str(uuid.uuid4()),
            company_id=company_id,
            camera_id=camera_id,
            equipment_id=equipment_id,
            motion_detected=bool(objects),
            motion_level=level.value,
            motion_area_percent=round(
                area_percent,
                4,
            ),
            image_width=width,
            image_height=height,
            objects=objects,
            events=events,
            restricted_zone_motion_detected=(
                restricted_motion
            ),
            inspected_at=now,
            requires_human_review=(
                restricted_motion
            ),
            metadata=dict(metadata or {}),
        )

    # ========================================================
    # ASYNC
    # ========================================================

    async def detect_async(
        self,
        image: np.ndarray,
        *,
        company_id: str,
        camera_id: str,
        equipment_id: str | None = None,
    ) -> MotionDetectionResult:
        return await asyncio.to_thread(
            self.detect,
            image,
            company_id=company_id,
            camera_id=camera_id,
            equipment_id=equipment_id,
        )

    # ========================================================
    # ANNOTATION
    # ========================================================

    def annotate(
        self,
        image: np.ndarray,
        result: MotionDetectionResult,
    ) -> np.ndarray:
        self._validate_image(image)

        annotated = image.copy()

        for zone in self.restricted_zones:
            cv2.rectangle(
                annotated,
                (zone.x, zone.y),
                (
                    zone.x + zone.width,
                    zone.y + zone.height,
                ),
                (0, 255, 255),
                2,
            )

            cv2.putText(
                annotated,
                zone.name,
                (
                    zone.x,
                    max(20, zone.y - 8),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        for obj in result.objects:
            color = (
                (0, 0, 255)
                if obj.zone_ids
                else (0, 255, 0)
            )

            cv2.rectangle(
                annotated,
                (obj.x, obj.y),
                (
                    obj.x + obj.width,
                    obj.y + obj.height,
                ),
                color,
                2,
            )

            cv2.circle(
                annotated,
                (
                    int(obj.centroid_x),
                    int(obj.centroid_y),
                ),
                4,
                color,
                -1,
            )

            cv2.putText(
                annotated,
                f"Motion {obj.object_id[:8]}",
                (
                    obj.x,
                    max(20, obj.y - 8),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        return annotated

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:
        with self._lock:
            self._previous_gray = None

            self._background_subtractor = (
                cv2.createBackgroundSubtractorMOG2(
                    history=(
                        self.config.mog2_history
                    ),
                    varThreshold=(
                        self.config.mog2_var_threshold
                    ),
                    detectShadows=(
                        self.config.detect_shadows
                    ),
                )
            )

            self._tracker.reset()


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class MotionVideoProcessor:
    """
    Video processing wrapper.

    Supports file paths and integer camera
    device indexes.

    Uses one MotionDetector per video stream.
    """

    def __init__(
        self,
        detector: MotionDetector,
    ) -> None:
        self.detector = detector

    def process_video(
        self,
        source: str | int,
        *,
        company_id: str,
        camera_id: str,
        equipment_id: str | None = None,
        output_path: str | None = None,
        max_frames: int | None = None,
    ) -> list[MotionDetectionResult]:
        capture = cv2.VideoCapture(
            source
        )

        if not capture.isOpened():
            raise MotionProcessingError(
                "Unable to open video source"
            )

        writer: cv2.VideoWriter | None = None

        results: list[
            MotionDetectionResult
        ] = []

        try:
            frame_count = 0

            while True:
                if (
                    max_frames is not None
                    and frame_count >= max_frames
                ):
                    break

                success, frame = capture.read()

                if not success:
                    break

                result = self.detector.detect(
                    frame,
                    company_id=company_id,
                    camera_id=camera_id,
                    equipment_id=equipment_id,
                )

                results.append(result)

                if output_path is not None:
                    annotated = (
                        self.detector.annotate(
                            frame,
                            result,
                        )
                    )

                    if writer is None:
                        height, width = (
                            annotated.shape[:2]
                        )

                        fps = capture.get(
                            cv2.CAP_PROP_FPS
                        )

                        if (
                            not math.isfinite(fps)
                            or fps <= 0
                        ):
                            fps = 25.0

                        writer = cv2.VideoWriter(
                            output_path,
                            cv2.VideoWriter_fourcc(
                                *"mp4v"
                            ),
                            fps,
                            (width, height),
                        )

                        if not writer.isOpened():
                            raise MotionProcessingError(
                                "Unable to create output video"
                            )

                    writer.write(
                        annotated
                    )

                frame_count += 1

        finally:
            capture.release()

            if writer is not None:
                writer.release()

        return results


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    detector = MotionDetector(
        MotionDetectionConfig(
            mode=MotionDetectionMode.MOG2,
            min_motion_area_px=300,
        ),
        restricted_zones=[
            RestrictedZone(
                zone_id="ZONE-01",
                name="Restricted Machinery Area",
                x=100,
                y=100,
                width=350,
                height=250,
            )
        ],
    )

    processor = MotionVideoProcessor(
        detector
    )

    results = processor.process_video(
        "industrial_camera.mp4",
        company_id="company-001",
        camera_id="CAM-01",
        equipment_id="MACHINE-101",
        output_path="motion_annotated.mp4",
        max_frames=300,
    )

    print(
        "Processed frames:",
        len(results),
    )

    print(
        "Motion frames:",
        sum(
            result.motion_detected
            for result in results
        ),
    )

    print(
        "Restricted-zone motion frames:",
        sum(
            result.restricted_zone_motion_detected
            for result in results
        ),
    )


if __name__ == "__main__":
    example()
