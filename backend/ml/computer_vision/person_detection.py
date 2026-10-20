
"""
app/ai/cv/person_detection.py

Industrial Person Detection & Worker Monitoring
================================================

Python 3.11+

Features:
- Person detection using YOLO
- OpenCV HOG fallback
- Person bounding boxes
- Detection confidence scores
- Restricted-zone monitoring
- Worker-count estimation
- Per-camera inspection results
- Annotated image generation
- Batch and async inference
- Tenant/equipment metadata
- JSON-compatible results

Dependencies:
    pip install numpy opencv-python

Optional YOLO backend:
    pip install ultralytics

IMPORTANT:
    Person detection does not establish worker
    identity, authorization, PPE compliance,
    intent, or physical safety.

    Occlusion, lighting, camera angle and
    model bias can cause missed detections.

    Do not use this module as the sole basis
    for emergency stops, machine guarding,
    personnel protection or access control.
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

class PersonDetectionError(Exception):
    """Base person detection exception."""


class PersonValidationError(PersonDetectionError):
    """Invalid image or configuration."""


class PersonModelError(PersonDetectionError):
    """Model loading or inference error."""


# ============================================================
# ENUMS
# ============================================================

class PersonDetectionBackend(str, Enum):
    YOLO = "yolo"
    OPENCV_HOG = "opencv_hog"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(slots=True)
class PersonDetectionConfig:
    backend: PersonDetectionBackend = (
        PersonDetectionBackend.YOLO
    )

    model_path: str = "yolo11n.pt"

    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45

    min_image_width: int = 64
    min_image_height: int = 64
    max_image_pixels: int = 20_000_000

    max_detections: int = 100

    # COCO person class.
    person_class_id: int = 0

    # YOLO inference device.
    # None = framework default.
    device: str | None = None

    # HOG settings.
    hog_win_stride: tuple[int, int] = (8, 8)
    hog_padding: tuple[int, int] = (8, 8)
    hog_scale: float = 1.05

    def __post_init__(self) -> None:
        if isinstance(self.backend, str):
            self.backend = PersonDetectionBackend(
                self.backend
            )

        if not (
            0 < self.confidence_threshold < 1
        ):
            raise PersonValidationError(
                "Invalid confidence threshold"
            )

        if not (
            0 < self.iou_threshold < 1
        ):
            raise PersonValidationError(
                "Invalid IoU threshold"
            )

        for name in (
            "min_image_width",
            "min_image_height",
            "max_image_pixels",
            "max_detections",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise PersonValidationError(
                    f"{name} must be positive"
                )

        if self.hog_scale <= 1:
            raise PersonValidationError(
                "hog_scale must be greater than 1"
            )


# ============================================================
# RESULT SCHEMAS
# ============================================================

@dataclass(slots=True)
class PersonZone:
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
class DetectedPerson:
    detection_id: str

    x: int
    y: int
    width: int
    height: int

    confidence: float

    zone_ids: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "bounding_box": {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
            },
            "confidence": self.confidence,
            "zone_ids": list(self.zone_ids),
        }


@dataclass(slots=True)
class PersonDetectionResult:
    inspection_id: str

    company_id: str
    camera_id: str | None
    equipment_id: str | None

    person_detected: bool
    person_count: int

    restricted_zone_person_detected: bool

    image_width: int
    image_height: int

    persons: list[DetectedPerson]

    backend: str
    inspected_at: datetime

    requires_human_review: bool = False
    safety_escalation_recommended: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "inspection_id": self.inspection_id,
            "company_id": self.company_id,
            "camera_id": self.camera_id,
            "equipment_id": self.equipment_id,
            "person_detected": self.person_detected,
            "person_count": self.person_count,
            "restricted_zone_person_detected": (
                self.restricted_zone_person_detected
            ),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "persons": [
                person.to_dict()
                for person in self.persons
            ],
            "backend": self.backend,
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
# YOLO PERSON DETECTOR
# ============================================================

class YOLOPersonDetector:
    """
    Ultralytics YOLO person detection.

    Uses the COCO person class by default.
    """

    def __init__(
        self,
        config: PersonDetectionConfig,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise PersonModelError(
                "Install ultralytics"
            ) from exc

        try:
            self.model = YOLO(
                config.model_path
            )
        except Exception as exc:
            raise PersonModelError(
                "Unable to load YOLO model"
            ) from exc

        self.config = config
        self._lock = threading.RLock()

    def predict(
        self,
        image: np.ndarray,
    ) -> list[DetectedPerson]:
        with self._lock:
            try:
                results = self.model.predict(
                    source=image,
                    conf=(
                        self.config
                        .confidence_threshold
                    ),
                    iou=(
                        self.config
                        .iou_threshold
                    ),
                    classes=[
                        self.config.person_class_id
                    ],
                    max_det=(
                        self.config.max_detections
                    ),
                    device=self.config.device,
                    verbose=False,
                )
            except Exception as exc:
                raise PersonModelError(
                    "YOLO inference failed"
                ) from exc

        height, width = image.shape[:2]

        persons: list[DetectedPerson] = []

        for result in results:
            boxes = getattr(
                result,
                "boxes",
                None,
            )

            if boxes is None:
                continue

            for box in boxes:
                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                confidence = float(
                    box.conf[0].item()
                )

                if not np.isfinite(
                    coordinates
                ).all():
                    continue

                x1, y1, x2, y2 = (
                    coordinates.tolist()
                )

                x1 = max(
                    0,
                    min(width, int(x1)),
                )

                y1 = max(
                    0,
                    min(height, int(y1)),
                )

                x2 = max(
                    0,
                    min(width, int(x2)),
                )

                y2 = max(
                    0,
                    min(height, int(y2)),
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                persons.append(
                    DetectedPerson(
                        detection_id=str(
                            uuid.uuid4()
                        ),
                        x=x1,
                        y=y1,
                        width=x2 - x1,
                        height=y2 - y1,
                        confidence=confidence,
                    )
                )

        return persons


# ============================================================
# OPENCV HOG PERSON DETECTOR
# ============================================================

class HOGPersonDetector:
    """
    OpenCV HOG + SVM pedestrian detector.

    Intended as a lightweight fallback.

    It may perform poorly on industrial
    camera angles, partial bodies and PPE.
    """

    def __init__(
        self,
        config: PersonDetectionConfig,
    ) -> None:
        self.config = config

        self.hog = cv2.HOGDescriptor()

        self.hog.setSVMDetector(
            cv2.HOGDescriptor_getDefaultPeopleDetector()
        )

        self._lock = threading.RLock()

    def predict(
        self,
        image: np.ndarray,
    ) -> list[DetectedPerson]:
        with self._lock:
            boxes, weights = (
                self.hog.detectMultiScale(
                    image,
                    winStride=(
                        self.config.hog_win_stride
                    ),
                    padding=(
                        self.config.hog_padding
                    ),
                    scale=(
                        self.config.hog_scale
                    ),
                )
            )

        persons: list[DetectedPerson] = []

        height, width = image.shape[:2]

        for (
            box,
            weight,
        ) in zip(boxes, weights):
            x, y, w, h = (
                int(value)
                for value in box
            )

            # HOG weights are SVM margins,
            # not calibrated probabilities.
            confidence = float(
                1.0 / (
                    1.0
                    + math.exp(
                        -max(
                            -50.0,
                            min(50.0, float(weight)),
                        )
                    )
                )
            )

            if (
                confidence
                < self.config.confidence_threshold
            ):
                continue

            x1 = max(0, x)
            y1 = max(0, y)

            x2 = min(
                width,
                x + w,
            )

            y2 = min(
                height,
                y + h,
            )

            if x2 <= x1 or y2 <= y1:
                continue

            persons.append(
                DetectedPerson(
                    detection_id=str(uuid.uuid4()),
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                    confidence=confidence,
                )
            )

        return sorted(
            persons,
            key=lambda person: (
                -person.confidence
            ),
        )[:self.config.max_detections]


# ============================================================
# MAIN PERSON DETECTOR
# ============================================================

class PersonDetector:
    """
    Industrial camera person detector.

    Restricted-zone detections are based on
    bounding-box overlap.

    A detected overlap does not establish
    actual entry into a physical hazard zone.
    """

    def __init__(
        self,
        config: PersonDetectionConfig | None = None,
        *,
        restricted_zones: Sequence[
            PersonZone
        ] | None = None,
    ) -> None:
        self.config = (
            config
            or PersonDetectionConfig()
        )

        self.restricted_zones = list(
            restricted_zones or []
        )

        if (
            self.config.backend
            == PersonDetectionBackend.YOLO
        ):
            self._detector = (
                YOLOPersonDetector(
                    self.config
                )
            )

        else:
            self._detector = (
                HOGPersonDetector(
                    self.config
                )
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
            raise PersonValidationError(
                "Image must be a NumPy array"
            )

        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
        ):
            raise PersonValidationError(
                "Expected uint8 BGR image"
            )

        height, width = image.shape[:2]

        if (
            width < self.config.min_image_width
            or height < self.config.min_image_height
        ):
            raise PersonValidationError(
                "Image dimensions too small"
            )

        if (
            width * height
            > self.config.max_image_pixels
        ):
            raise PersonValidationError(
                "Image exceeds pixel limit"
            )

    @staticmethod
    def _validate_company(
        company_id: str,
    ) -> None:
        if (
            not isinstance(company_id, str)
            or not company_id.strip()
        ):
            raise PersonValidationError(
                "company_id is required"
            )

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    def load_image(
        self,
        path: str | Path,
    ) -> np.ndarray:
        path = Path(path)

        if not path.is_file():
            raise PersonValidationError(
                "Image file not found"
            )

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise PersonValidationError(
                "Unable to decode image"
            )

        self._validate_image(image)

        return image

    def decode_image(
        self,
        data: bytes,
    ) -> np.ndarray:
        if not data:
            raise PersonValidationError(
                "Image bytes cannot be empty"
            )

        buffer = np.frombuffer(
            data,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            buffer,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise PersonValidationError(
                "Invalid image data"
            )

        self._validate_image(image)

        return image

    # ========================================================
    # RESTRICTED ZONE GEOMETRY
    # ========================================================

    @staticmethod
    def _overlaps_zone(
        person: DetectedPerson,
        zone: PersonZone,
    ) -> bool:
        x1 = max(
            person.x,
            zone.x,
        )

        y1 = max(
            person.y,
            zone.y,
        )

        x2 = min(
            person.x + person.width,
            zone.x + zone.width,
        )

        y2 = min(
            person.y + person.height,
            zone.y + zone.height,
        )

        return (
            x2 > x1
            and y2 > y1
        )

    def _assign_zones(
        self,
        persons: list[DetectedPerson],
    ) -> None:
        for person in persons:
            person.zone_ids = [
                zone.zone_id
                for zone in self.restricted_zones
                if self._overlaps_zone(
                    person,
                    zone,
                )
            ]

    # ========================================================
    # DETECT
    # ========================================================

    def detect(
        self,
        image: np.ndarray,
        *,
        company_id: str,
        camera_id: str | None = None,
        equipment_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PersonDetectionResult:
        self._validate_company(
            company_id
        )

        self._validate_image(
            image
        )

        persons = self._detector.predict(
            image
        )

        self._assign_zones(
            persons
        )

        restricted_detected = any(
            person.zone_ids
            for person in persons
        )

        height, width = image.shape[:2]

        return PersonDetectionResult(
            inspection_id=str(uuid.uuid4()),
            company_id=company_id,
            camera_id=camera_id,
            equipment_id=equipment_id,
            person_detected=bool(persons),
            person_count=len(persons),
            restricted_zone_person_detected=(
                restricted_detected
            ),
            image_width=width,
            image_height=height,
            persons=persons,
            backend=self.config.backend.value,
            inspected_at=datetime.now(
                timezone.utc
            ),
            requires_human_review=(
                restricted_detected
            ),
            safety_escalation_recommended=(
                restricted_detected
            ),
            metadata=dict(metadata or {}),
        )

    # ========================================================
    # FILE / BYTES
    # ========================================================

    def detect_file(
        self,
        path: str | Path,
        *,
        company_id: str,
        camera_id: str | None = None,
        equipment_id: str | None = None,
    ) -> PersonDetectionResult:
        image = self.load_image(
            path
        )

        return self.detect(
            image,
            company_id=company_id,
            camera_id=camera_id,
            equipment_id=equipment_id,
            metadata={
                "filename": Path(path).name
            },
        )

    def detect_bytes(
        self,
        data: bytes,
        *,
        company_id: str,
        camera_id: str | None = None,
        equipment_id: str | None = None,
    ) -> PersonDetectionResult:
        image = self.decode_image(
            data
        )

        return self.detect(
            image,
            company_id=company_id,
            camera_id=camera_id,
            equipment_id=equipment_id,
        )

    # ========================================================
    # BATCH
    # ========================================================

    def detect_batch(
        self,
        images: Sequence[np.ndarray],
        *,
        company_id: str,
        camera_id: str | None = None,
        equipment_id: str | None = None,
    ) -> list[PersonDetectionResult]:
        return [
            self.detect(
                image,
                company_id=company_id,
                camera_id=camera_id,
                equipment_id=equipment_id,
            )
            for image in images
        ]

    # ========================================================
    # ASYNC
    # ========================================================

    async def detect_async(
        self,
        image: np.ndarray,
        *,
        company_id: str,
        camera_id: str | None = None,
        equipment_id: str | None = None,
    ) -> PersonDetectionResult:
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
        result: PersonDetectionResult,
    ) -> np.ndarray:
        self._validate_image(
            image
        )

        if (
            image.shape[1] != result.image_width
            or image.shape[0] != result.image_height
        ):
            raise PersonValidationError(
                "Image dimensions do not match result"
            )

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

        for person in result.persons:
            color = (
                (0, 0, 255)
                if person.zone_ids
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

            label = (
                "Person "
                f"{person.confidence:.2f}"
            )

            cv2.putText(
                annotated,
                label,
                (
                    person.x,
                    max(20, person.y - 8),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            annotated,
            f"Persons: {result.person_count}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return annotated


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    import json

    detector = PersonDetector(
        PersonDetectionConfig(
            backend=PersonDetectionBackend.YOLO,
            model_path="yolo11n.pt",
            confidence_threshold=0.4,
        ),
        restricted_zones=[
            PersonZone(
                zone_id="ZONE-01",
                name="Machine Hazard Area",
                x=100,
                y=100,
                width=400,
                height=300,
            )
        ],
    )

    result = detector.detect_file(
        "industrial_workers.jpg",
        company_id="company-001",
        camera_id="CAM-01",
        equipment_id="MACHINE-101",
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
        )
    )

    image = detector.load_image(
        "industrial_workers.jpg"
    )

    annotated = detector.annotate(
        image,
        result,
    )

    cv2.imwrite(
        "person_detection_annotated.jpg",
        annotated,
    )


if __name__ == "__main__":
    example()
