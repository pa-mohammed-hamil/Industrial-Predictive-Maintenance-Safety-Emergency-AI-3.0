
"""
app/ai/cv/restricted_zone_detection.py

Industrial Restricted Zone Detection
====================================

Python 3.11+

Dependencies:
    pip install numpy opencv-python

Optional person detection:
    pip install ultralytics

Features:
- Polygon and rectangular restricted zones
- Person detection integration
- Foot-point based zone occupancy
- Entry and exit event generation
- Dwell-time monitoring
- Multi-camera state isolation
- Stable object ID association
- Configurable detection thresholds
- Frame annotation
- Batch/video-frame processing
- JSON-compatible results

SAFETY:
This is an advisory computer-vision system.
It is not certified for personnel protection,
machine guarding, access control or emergency
shutdown.

Detection and tracking can fail due to camera
occlusion, lighting, perspective or model errors.

Do not use it as the sole safety mechanism.
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
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from app.ai.cv.person_detection import (
    PersonDetector,
    PersonDetectionConfig,
    PersonDetectionBackend,
    DetectedPerson,
)


logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class RestrictedZoneDetectionError(Exception):
    """Base restricted-zone exception."""


class RestrictedZoneValidationError(
    RestrictedZoneDetectionError
):
    """Invalid zone or image input."""


# ============================================================
# ENUMS
# ============================================================

class ZoneEventType(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"
    DWELL_EXCEEDED = "dwell_exceeded"


class ZoneRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(slots=True)
class RestrictedZoneConfig:
    minimum_dwell_seconds: float = 3.0

    max_tracking_distance_px: float = 100.0

    max_missing_frames: int = 10

    footpoint_offset_ratio: float = 0.03

    emit_entry_events: bool = True
    emit_exit_events: bool = True
    emit_dwell_events: bool = True

    def __post_init__(self) -> None:
        if self.minimum_dwell_seconds < 0:
            raise RestrictedZoneValidationError(
                "minimum_dwell_seconds cannot "
                "be negative"
            )

        if self.max_tracking_distance_px <= 0:
            raise RestrictedZoneValidationError(
                "max_tracking_distance_px must "
                "be positive"
            )

        if self.max_missing_frames < 0:
            raise RestrictedZoneValidationError(
                "max_missing_frames cannot "
                "be negative"
            )

        if not (
            0 <= self.footpoint_offset_ratio <= 0.5
        ):
            raise RestrictedZoneValidationError(
                "Invalid footpoint_offset_ratio"
            )


# ============================================================
# ZONE SCHEMAS
# ============================================================

@dataclass(slots=True)
class RestrictedZone:
    zone_id: str
    name: str

    polygon: list[tuple[int, int]]

    risk_level: ZoneRiskLevel = (
        ZoneRiskLevel.HIGH
    )

    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.zone_id.strip():
            raise RestrictedZoneValidationError(
                "zone_id is required"
            )

        if len(self.polygon) < 3:
            raise RestrictedZoneValidationError(
                "Polygon requires at least 3 points"
            )

        if isinstance(self.risk_level, str):
            self.risk_level = ZoneRiskLevel(
                self.risk_level
            )

        points = np.array(
            self.polygon,
            dtype=np.int32,
        )

        if cv2.contourArea(points) <= 0:
            raise RestrictedZoneValidationError(
                "Zone polygon must have positive area"
            )

    @classmethod
    def rectangle(
        cls,
        *,
        zone_id: str,
        name: str,
        x: int,
        y: int,
        width: int,
        height: int,
        risk_level: ZoneRiskLevel = (
            ZoneRiskLevel.HIGH
        ),
    ) -> "RestrictedZone":
        if width <= 0 or height <= 0:
            raise RestrictedZoneValidationError(
                "Rectangle dimensions must be positive"
            )

        return cls(
            zone_id=zone_id,
            name=name,
            polygon=[
                (x, y),
                (x + width, y),
                (x + width, y + height),
                (x, y + height),
            ],
            risk_level=risk_level,
        )

    def contains(
        self,
        x: float,
        y: float,
    ) -> bool:
        if not self.enabled:
            return False

        contour = np.array(
            self.polygon,
            dtype=np.float32,
        )

        return (
            cv2.pointPolygonTest(
                contour,
                (float(x), float(y)),
                False,
            )
            >= 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "polygon": [
                {"x": x, "y": y}
                for x, y in self.polygon
            ],
            "risk_level": self.risk_level.value,
            "enabled": self.enabled,
        }


# ============================================================
# DETECTED PERSON
# ============================================================

@dataclass(slots=True)
class ZonePerson:
    tracking_id: str

    x: int
    y: int
    width: int
    height: int

    confidence: float

    foot_x: float
    foot_y: float

    occupied_zone_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracking_id": self.tracking_id,
            "bounding_box": {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
            },
            "confidence": self.confidence,
            "foot_point": {
                "x": self.foot_x,
                "y": self.foot_y,
            },
            "occupied_zone_ids": list(
                self.occupied_zone_ids
            ),
        }


# ============================================================
# EVENT SCHEMA
# ============================================================

@dataclass(slots=True)
class RestrictedZoneEvent:
    event_id: str

    company_id: str
    camera_id: str

    tracking_id: str
    zone_id: str

    event_type: str
    risk_level: str

    occurred_at: datetime

    dwell_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "company_id": self.company_id,
            "camera_id": self.camera_id,
            "tracking_id": self.tracking_id,
            "zone_id": self.zone_id,
            "event_type": self.event_type,
            "risk_level": self.risk_level,
            "occurred_at": (
                self.occurred_at.isoformat()
            ),
            "dwell_seconds": (
                self.dwell_seconds
            ),
        }


# ============================================================
# RESULT SCHEMA
# ============================================================

@dataclass(slots=True)
class RestrictedZoneDetectionResult:
    inspection_id: str

    company_id: str
    camera_id: str
    equipment_id: str | None

    person_count: int

    restricted_zone_occupied: bool

    persons: list[ZonePerson]
    events: list[RestrictedZoneEvent]

    occupied_zone_ids: list[str]

    image_width: int
    image_height: int

    inspected_at: datetime

    requires_human_review: bool
    safety_escalation_recommended: bool

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "inspection_id": self.inspection_id,
            "company_id": self.company_id,
            "camera_id": self.camera_id,
            "equipment_id": self.equipment_id,
            "person_count": self.person_count,
            "restricted_zone_occupied": (
                self.restricted_zone_occupied
            ),
            "persons": [
                person.to_dict()
                for person in self.persons
            ],
            "events": [
                event.to_dict()
                for event in self.events
            ],
            "occupied_zone_ids": list(
                self.occupied_zone_ids
            ),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "inspected_at": (
                self.inspected_at.isoformat()
            ),
            "requires_human_review": (
                self.requires_human_review
            ),
            "safety_escalation_recommended": (
                self.safety_escalation_recommended
            ),
            "metadata": dict(self.metadata),
        }


# ============================================================
# INTERNAL TRACKING STATE
# ============================================================

@dataclass(slots=True)
class _TrackedPerson:
    tracking_id: str

    foot_x: float
    foot_y: float

    missing_frames: int = 0

    entered_zones: dict[
        str,
        datetime,
    ] = field(default_factory=dict)

    dwell_emitted: set[str] = field(
        default_factory=set
    )


# ============================================================
# RESTRICTED ZONE DETECTOR
# ============================================================

class RestrictedZoneDetector:
    """
    Stateful industrial zone occupancy monitor.

    Maintains independent tracking state
    for each (company_id, camera_id).

    PersonDetector supplies bounding boxes.
    Foot-point geometry determines zone
    membership.

    Object tracking is centroid based and
    may experience identity switches.
    """

    def __init__(
        self,
        *,
        zones: Sequence[RestrictedZone],
        person_detector: PersonDetector | None = None,
        config: RestrictedZoneConfig | None = None,
    ) -> None:
        self.config = (
            config or RestrictedZoneConfig()
        )

        self.zones = {
            zone.zone_id: zone
            for zone in zones
        }

        if len(self.zones) != len(zones):
            raise RestrictedZoneValidationError(
                "Duplicate zone_id values"
            )

        self.person_detector = (
            person_detector
            or PersonDetector(
                PersonDetectionConfig(
                    backend=(
                        PersonDetectionBackend.YOLO
                    )
                )
            )
        )

        self._tracks: dict[
            tuple[str, str],
            dict[str, _TrackedPerson],
        ] = {}

        self._lock = threading.RLock()

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_identity(
        company_id: str,
        camera_id: str,
    ) -> None:
        if not company_id or not company_id.strip():
            raise RestrictedZoneValidationError(
                "company_id is required"
            )

        if not camera_id or not camera_id.strip():
            raise RestrictedZoneValidationError(
                "camera_id is required"
            )

    # ========================================================
    # FOOT POINT
    # ========================================================

    def _foot_point(
        self,
        person: DetectedPerson,
    ) -> tuple[float, float]:
        foot_x = (
            person.x + person.width / 2.0
        )

        foot_y = (
            person.y
            + person.height
            - (
                person.height
                * self.config.footpoint_offset_ratio
            )
        )

        return foot_x, foot_y

    # ========================================================
    # MATCH PERSON TO TRACK
    # ========================================================

    def _match_track(
        self,
        tracks: dict[str, _TrackedPerson],
        foot_x: float,
        foot_y: float,
        assigned: set[str],
    ) -> _TrackedPerson:
        best_track: (
            _TrackedPerson | None
        ) = None

        best_distance = float("inf")

        for track_id, track in tracks.items():
            if track_id in assigned:
                continue

            distance = math.hypot(
                foot_x - track.foot_x,
                foot_y - track.foot_y,
            )

            if (
                distance
                < best_distance
                and distance
                <= self.config.max_tracking_distance_px
            ):
                best_track = track
                best_distance = distance

        if best_track is None:
            tracking_id = str(uuid.uuid4())

            best_track = _TrackedPerson(
                tracking_id=tracking_id,
                foot_x=foot_x,
                foot_y=foot_y,
            )

            tracks[tracking_id] = best_track

        best_track.foot_x = foot_x
        best_track.foot_y = foot_y
        best_track.missing_frames = 0

        assigned.add(
            best_track.tracking_id
        )

        return best_track

    # ========================================================
    # EVENT CREATION
    # ========================================================

    @staticmethod
    def _create_event(
        *,
        company_id: str,
        camera_id: str,
        tracking_id: str,
        zone: RestrictedZone,
        event_type: ZoneEventType,
        now: datetime,
        dwell_seconds: float = 0.0,
    ) -> RestrictedZoneEvent:
        return RestrictedZoneEvent(
            event_id=str(uuid.uuid4()),
            company_id=company_id,
            camera_id=camera_id,
            tracking_id=tracking_id,
            zone_id=zone.zone_id,
            event_type=event_type.value,
            risk_level=zone.risk_level.value,
            occurred_at=now,
            dwell_seconds=round(
                dwell_seconds,
                3,
            ),
        )

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
        timestamp: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RestrictedZoneDetectionResult:
        self._validate_identity(
            company_id,
            camera_id,
        )

        if timestamp is None:
            timestamp = datetime.now(
                timezone.utc
            )

        if timestamp.tzinfo is None:
            raise RestrictedZoneValidationError(
                "timestamp must be timezone-aware"
            )

        detection = (
            self.person_detector.detect(
                image,
                company_id=company_id,
                camera_id=camera_id,
                equipment_id=equipment_id,
            )
        )

        key = (
            company_id,
            camera_id,
        )

        events: list[
            RestrictedZoneEvent
        ] = []

        persons: list[ZonePerson] = []

        with self._lock:
            tracks = self._tracks.setdefault(
                key,
                {},
            )

            assigned: set[str] = set()

            for detected_person in detection.persons:
                foot_x, foot_y = (
                    self._foot_point(
                        detected_person
                    )
                )

                track = self._match_track(
                    tracks,
                    foot_x,
                    foot_y,
                    assigned,
                )

                occupied_zones = [
                    zone
                    for zone in self.zones.values()
                    if zone.contains(
                        foot_x,
                        foot_y,
                    )
                ]

                occupied_ids = {
                    zone.zone_id
                    for zone in occupied_zones
                }

                previous_ids = set(
                    track.entered_zones
                )

                # Zone entries
                for zone in occupied_zones:
                    if (
                        zone.zone_id
                        not in previous_ids
                    ):
                        track.entered_zones[
                            zone.zone_id
                        ] = timestamp

                        if (
                            self.config
                            .emit_entry_events
                        ):
                            events.append(
                                self._create_event(
                                    company_id=company_id,
                                    camera_id=camera_id,
                                    tracking_id=(
                                        track.tracking_id
                                    ),
                                    zone=zone,
                                    event_type=(
                                        ZoneEventType.ENTRY
                                    ),
                                    now=timestamp,
                                )
                            )

                    entered_at = (
                        track.entered_zones[
                            zone.zone_id
                        ]
                    )

                    dwell = max(
                        0.0,
                        (
                            timestamp - entered_at
                        ).total_seconds(),
                    )

                    if (
                        self.config
                        .emit_dwell_events
                        and dwell
                        >= self.config
                        .minimum_dwell_seconds
                        and zone.zone_id
                        not in track.dwell_emitted
                    ):
                        events.append(
                            self._create_event(
                                company_id=company_id,
                                camera_id=camera_id,
                                tracking_id=(
                                    track.tracking_id
                                ),
                                zone=zone,
                                event_type=(
                                    ZoneEventType
                                    .DWELL_EXCEEDED
                                ),
                                now=timestamp,
                                dwell_seconds=dwell,
                            )
                        )

                        track.dwell_emitted.add(
                            zone.zone_id
                        )

                # Zone exits
                exited_ids = (
                    previous_ids - occupied_ids
                )

                for zone_id in exited_ids:
                    zone = self.zones[
                        zone_id
                    ]

                    entered_at = (
                        track.entered_zones.pop(
                            zone_id
                        )
                    )

                    track.dwell_emitted.discard(
                        zone_id
                    )

                    if self.config.emit_exit_events:
                        dwell = max(
                            0.0,
                            (
                                timestamp - entered_at
                            ).total_seconds(),
                        )

                        events.append(
                            self._create_event(
                                company_id=company_id,
                                camera_id=camera_id,
                                tracking_id=(
                                    track.tracking_id
                                ),
                                zone=zone,
                                event_type=(
                                    ZoneEventType.EXIT
                                ),
                                now=timestamp,
                                dwell_seconds=dwell,
                            )
                        )

                persons.append(
                    ZonePerson(
                        tracking_id=(
                            track.tracking_id
                        ),
                        x=detected_person.x,
                        y=detected_person.y,
                        width=(
                            detected_person.width
                        ),
                        height=(
                            detected_person.height
                        ),
                        confidence=(
                            detected_person.confidence
                        ),
                        foot_x=foot_x,
                        foot_y=foot_y,
                        occupied_zone_ids=sorted(
                            occupied_ids
                        ),
                    )
                )

            # Remove expired tracks.
            for track_id in list(tracks):
                if track_id in assigned:
                    continue

                track = tracks[track_id]
                track.missing_frames += 1

                if (
                    track.missing_frames
                    > self.config.max_missing_frames
                ):
                    # Tracking loss is not proof
                    # of a physical zone exit.
                    del tracks[track_id]

        occupied_zone_ids = sorted({
            zone_id
            for person in persons
            for zone_id in person.occupied_zone_ids
        })

        occupied = bool(
            occupied_zone_ids
        )

        return RestrictedZoneDetectionResult(
            inspection_id=str(uuid.uuid4()),
            company_id=company_id,
            camera_id=camera_id,
            equipment_id=equipment_id,
            person_count=len(persons),
            restricted_zone_occupied=occupied,
            persons=persons,
            events=events,
            occupied_zone_ids=(
                occupied_zone_ids
            ),
            image_width=(
                detection.image_width
            ),
            image_height=(
                detection.image_height
            ),
            inspected_at=timestamp,
            requires_human_review=occupied,
            safety_escalation_recommended=occupied,
            metadata=dict(metadata or {}),
        )

    # ========================================================
    # ASYNC INSPECTION
    # ========================================================

    async def detect_async(
        self,
        image: np.ndarray,
        *,
        company_id: str,
        camera_id: str,
        equipment_id: str | None = None,
    ) -> RestrictedZoneDetectionResult:
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
        result: RestrictedZoneDetectionResult,
    ) -> np.ndarray:
        annotated = image.copy()

        overlay = annotated.copy()

        for zone in self.zones.values():
            if not zone.enabled:
                continue

            polygon = np.array(
                zone.polygon,
                dtype=np.int32,
            )

            color = (
                (0, 0, 255)
                if zone.zone_id
                in result.occupied_zone_ids
                else (0, 255, 255)
            )

            cv2.fillPoly(
                overlay,
                [polygon],
                color,
            )

            cv2.polylines(
                annotated,
                [polygon],
                True,
                color,
                2,
            )

            first_point = (
                zone.polygon[0]
            )

            cv2.putText(
                annotated,
                zone.name,
                first_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

        annotated = cv2.addWeighted(
            overlay,
            0.15,
            annotated,
            0.85,
            0,
        )

        for person in result.persons:
            color = (
                (0, 0, 255)
                if person.occupied_zone_ids
                else (0, 255, 0)
            )

            cv2.rectangle(
                annotated,
                (person.x, person.y),
                (
                    person.x + person.width,
                    person.y + person.height,
                ),
                color,
                2,
            )

            cv2.circle(
                annotated,
                (
                    int(person.foot_x),
                    int(person.foot_y),
                ),
                5,
                color,
                -1,
            )

            cv2.putText(
                annotated,
                (
                    "Person "
                    f"{person.tracking_id[:8]}"
                ),
                (
                    person.x,
                    max(20, person.y - 8),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        return annotated

    # ========================================================
    # RESET CAMERA
    # ========================================================

    def reset_camera(
        self,
        *,
        company_id: str,
        camera_id: str,
    ) -> None:
        with self._lock:
            self._tracks.pop(
                (company_id, camera_id),
                None,
            )


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    import json

    detector = RestrictedZoneDetector(
        zones=[
            RestrictedZone.rectangle(
                zone_id="ZONE-01",
                name="Robot Work Cell",
                x=100,
                y=100,
                width=350,
                height=250,
                risk_level=ZoneRiskLevel.CRITICAL,
            ),
            RestrictedZone(
                zone_id="ZONE-02",
                name="High Voltage Area",
                polygon=[
                    (500, 100),
                    (750, 130),
                    (700, 400),
                    (480, 350),
                ],
                risk_level=ZoneRiskLevel.HIGH,
            ),
        ],
        config=RestrictedZoneConfig(
            minimum_dwell_seconds=3.0,
            max_tracking_distance_px=100.0,
        ),
    )

    image = cv2.imread(
        "industrial_workers.jpg"
    )

    if image is None:
        raise FileNotFoundError(
            "industrial_workers.jpg"
        )

    result = detector.detect(
        image,
        company_id="company-001",
        camera_id="CAM-01",
        equipment_id="ROBOT-101",
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
        )
    )

    annotated = detector.annotate(
        image,
        result,
    )

    cv2.imwrite(
        "restricted_zone_annotated.jpg",
        annotated,
    )


if __name__ == "__main__":
    example()
