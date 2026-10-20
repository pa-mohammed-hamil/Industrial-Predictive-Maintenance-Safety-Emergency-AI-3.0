
"""
app/ai/cv/fire_detection.py

Industrial Fire & Flame Detection
=================================

Python 3.11+

Dependencies:
    pip install numpy opencv-python

Optional:
    pip install onnxruntime

Features:
- Fire-colored pixel candidate detection
- HSV + RGB heuristic filtering
- Optional ONNX flame segmentation
- Morphological noise removal
- Bounding box extraction
- Fire candidate area estimation
- Video-frame temporal confirmation
- Annotated images
- Batch image processing
- Async API
- Tenant and equipment metadata
- JSON-compatible inspection results

SAFETY:
This module is NOT a certified fire detection system.

The heuristic can mistake lights, welding,
reflections and orange surfaces for flames.
It can also miss real fires.

Never use this module as the sole source for
fire alarms, evacuation, shutdown, suppression
or emergency response decisions.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class FireDetectionError(Exception):
    """Base fire detection exception."""


class FireValidationError(FireDetectionError):
    """Invalid image or configuration."""


class FireModelError(FireDetectionError):
    """Fire segmentation model failure."""


# ============================================================
# ENUMS
# ============================================================

class FireDetectionMode(str, Enum):
    COLOR_HEURISTIC = "color_heuristic"
    ONNX_SEGMENTATION = "onnx_segmentation"


class FireScreeningLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(slots=True)
class FireDetectionConfig:
    mode: FireDetectionMode = (
        FireDetectionMode.COLOR_HEURISTIC
    )

    min_image_width: int = 64
    min_image_height: int = 64
    max_image_pixels: int = 20_000_000

    min_region_area_px: int = 80
    morphology_kernel_size: int = 3

    # OpenCV HSV hue: 0..179
    hsv_lower: tuple[int, int, int] = (0, 100, 150)
    hsv_upper: tuple[int, int, int] = (35, 255, 255)

    # Require red to dominate blue for
    # warm flame-colored pixels.
    min_red_blue_difference: int = 45

    low_area_percent: float = 0.05
    moderate_area_percent: float = 1.0
    high_area_percent: float = 5.0

    # Temporal confirmation for video streams.
    temporal_window: int = 5
    temporal_required_frames: int = 3

    onnx_model_path: str | None = None
    onnx_input_width: int = 512
    onnx_input_height: int = 512
    onnx_probability_threshold: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            self.mode = FireDetectionMode(self.mode)

        positive_fields = (
            "min_image_width",
            "min_image_height",
            "max_image_pixels",
            "min_region_area_px",
            "morphology_kernel_size",
            "temporal_window",
            "temporal_required_frames",
            "onnx_input_width",
            "onnx_input_height",
        )

        for name in positive_fields:
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise FireValidationError(
                    f"{name} must be a positive integer"
                )

        if (
            self.temporal_required_frames
            > self.temporal_window
        ):
            raise FireValidationError(
                "temporal_required_frames cannot "
                "exceed temporal_window"
            )

        if not (
            0 <= self.min_red_blue_difference <= 255
        ):
            raise FireValidationError(
                "Invalid red/blue difference"
            )

        thresholds = (
            self.low_area_percent,
            self.moderate_area_percent,
            self.high_area_percent,
        )

        if not (
            0 <= thresholds[0]
            < thresholds[1]
            < thresholds[2]
            <= 100
        ):
            raise FireValidationError(
                "Area thresholds must be increasing"
            )

        if not (
            0 < self.onnx_probability_threshold < 1
        ):
            raise FireValidationError(
                "Invalid ONNX probability threshold"
            )


# ============================================================
# RESULT SCHEMAS
# ============================================================

@dataclass(slots=True)
class FireRegion:
    region_id: str
    x: int
    y: int
    width: int
    height: int
    pixel_area: int
    image_area_percent: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "bounding_box": {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
            },
            "pixel_area": self.pixel_area,
            "image_area_percent": (
                self.image_area_percent
            ),
        }


@dataclass(slots=True)
class FireDetectionResult:
    inspection_id: str
    company_id: str
    asset_id: str | None
    equipment_id: str | None
    camera_id: str | None

    fire_candidate_detected: bool
    screening_level: str

    candidate_area_percent: float
    candidate_pixel_count: int

    image_width: int
    image_height: int

    regions: list[FireRegion]
    detection_mode: str
    inspected_at: datetime

    # This is a review/escalation recommendation,
    # not an automatic emergency command.
    requires_human_review: bool = True
    safety_escalation_recommended: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "inspection_id": self.inspection_id,
            "company_id": self.company_id,
            "asset_id": self.asset_id,
            "equipment_id": self.equipment_id,
            "camera_id": self.camera_id,
            "fire_candidate_detected": (
                self.fire_candidate_detected
            ),
            "screening_level": self.screening_level,
            "candidate_area_percent": (
                self.candidate_area_percent
            ),
            "candidate_pixel_count": (
                self.candidate_pixel_count
            ),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "regions": [
                region.to_dict()
                for region in self.regions
            ],
            "detection_mode": self.detection_mode,
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


@dataclass(slots=True)
class FireTemporalResult:
    camera_id: str
    company_id: str

    current_frame_detected: bool
    candidate_frames_in_window: int
    frames_observed: int
    required_frames: int

    temporally_confirmed: bool
    safety_escalation_recommended: bool

    current_detection: FireDetectionResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "company_id": self.company_id,
            "current_frame_detected": (
                self.current_frame_detected
            ),
            "candidate_frames_in_window": (
                self.candidate_frames_in_window
            ),
            "frames_observed": self.frames_observed,
            "required_frames": self.required_frames,
            "temporally_confirmed": (
                self.temporally_confirmed
            ),
            "safety_escalation_recommended": (
                self.safety_escalation_recommended
            ),
            "current_detection": (
                self.current_detection.to_dict()
            ),
        }


# ============================================================
# ONNX FIRE SEGMENTATION
# ============================================================

class ONNXFireSegmenter:
    """
    Expected ONNX contract:

    Input:
        RGB float32 NCHW, values in [0,1]

    Output:
        Single-channel flame probability mask.

    Models requiring mean/std normalization,
    sigmoid activation, or different output
    layouts need an adapted preprocessing
    and postprocessing implementation.
    """

    def __init__(
        self,
        model_path: str,
        *,
        width: int = 512,
        height: int = 512,
        threshold: float = 0.5,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise FireModelError(
                "Install onnxruntime"
            ) from exc

        path = Path(model_path)

        if not path.is_file():
            raise FireModelError(
                "ONNX model file not found"
            )

        try:
            self.session = ort.InferenceSession(
                str(path),
                providers=[
                    "CPUExecutionProvider"
                ],
            )
        except Exception as exc:
            raise FireModelError(
                "Unable to load ONNX fire model"
            ) from exc

        self.width = width
        self.height = height
        self.threshold = threshold

        self.input_name = (
            self.session.get_inputs()[0].name
        )

        self.output_name = (
            self.session.get_outputs()[0].name
        )

    def predict(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        height, width = image.shape[:2]

        resized = cv2.resize(
            image,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA,
        )

        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB,
        )

        tensor = (
            rgb.astype(np.float32) / 255.0
        )

        tensor = np.transpose(
            tensor,
            (2, 0, 1),
        )[None, ...]

        try:
            output = self.session.run(
                [self.output_name],
                {self.input_name: tensor},
            )[0]
        except Exception as exc:
            raise FireModelError(
                "ONNX fire inference failed"
            ) from exc

        probabilities = np.asarray(
            output,
            dtype=np.float32,
        )

        if probabilities.ndim == 4:
            if (
                probabilities.shape[0] != 1
                or probabilities.shape[1] != 1
            ):
                raise FireModelError(
                    "Expected NCHW single-channel output"
                )

            probabilities = probabilities[0, 0]

        elif probabilities.ndim == 3:
            if probabilities.shape[0] != 1:
                raise FireModelError(
                    "Expected one output channel"
                )

            probabilities = probabilities[0]

        if probabilities.ndim != 2:
            raise FireModelError(
                "Unexpected segmentation output shape"
            )

        if not np.isfinite(
            probabilities
        ).all():
            raise FireModelError(
                "Model output contains invalid values"
            )

        probabilities = cv2.resize(
            probabilities,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

        return (
            probabilities >= self.threshold
        ).astype(np.uint8) * 255


# ============================================================
# MAIN FIRE DETECTOR
# ============================================================

class FireDetector:
    """
    Industrial fire-colored region screening.

    Each image is processed independently.

    Temporal confirmation is managed separately
    by FireTemporalMonitor.
    """

    def __init__(
        self,
        config: FireDetectionConfig | None = None,
    ) -> None:
        self.config = (
            config or FireDetectionConfig()
        )

        self._onnx_model: (
            ONNXFireSegmenter | None
        ) = None

        if (
            self.config.mode
            == FireDetectionMode.ONNX_SEGMENTATION
        ):
            if not self.config.onnx_model_path:
                raise FireValidationError(
                    "onnx_model_path is required"
                )

            self._onnx_model = ONNXFireSegmenter(
                self.config.onnx_model_path,
                width=self.config.onnx_input_width,
                height=self.config.onnx_input_height,
                threshold=(
                    self.config
                    .onnx_probability_threshold
                ),
            )

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_image(
        self,
        image: np.ndarray,
    ) -> None:
        if not isinstance(image, np.ndarray):
            raise FireValidationError(
                "Image must be a NumPy array"
            )

        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
        ):
            raise FireValidationError(
                "Expected uint8 BGR image"
            )

        height, width = image.shape[:2]

        if (
            width < self.config.min_image_width
            or height < self.config.min_image_height
        ):
            raise FireValidationError(
                "Image dimensions are too small"
            )

        if (
            width * height
            > self.config.max_image_pixels
        ):
            raise FireValidationError(
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
            raise FireValidationError(
                "company_id is required"
            )

    # ========================================================
    # IMAGE LOADING
    # ========================================================

    def load_image(
        self,
        path: str | Path,
    ) -> np.ndarray:
        path = Path(path)

        if not path.is_file():
            raise FireValidationError(
                "Image file not found"
            )

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise FireValidationError(
                "Unable to decode image"
            )

        self._validate_image(image)

        return image

    def decode_image(
        self,
        data: bytes,
    ) -> np.ndarray:
        if not data:
            raise FireValidationError(
                "Image data cannot be empty"
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
            raise FireValidationError(
                "Invalid image data"
            )

        self._validate_image(image)

        return image

    # ========================================================
    # COLOR HEURISTIC
    # ========================================================

    def _color_mask(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        hsv_mask = cv2.inRange(
            hsv,
            np.array(
                self.config.hsv_lower,
                dtype=np.uint8,
            ),
            np.array(
                self.config.hsv_upper,
                dtype=np.uint8,
            ),
        )

        blue = image[:, :, 0].astype(
            np.int16
        )

        green = image[:, :, 1].astype(
            np.int16
        )

        red = image[:, :, 2].astype(
            np.int16
        )

        warm_pixels = (
            (red > blue + (
                self.config
                .min_red_blue_difference
            ))
            & (red >= green)
            & (green >= blue)
        )

        warm_mask = (
            warm_pixels.astype(np.uint8)
            * 255
        )

        return cv2.bitwise_and(
            hsv_mask,
            warm_mask,
        )

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

        closed = cv2.morphologyEx(
            opened,
            cv2.MORPH_CLOSE,
            kernel,
        )

        count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                closed,
                connectivity=8,
            )
        )

        cleaned = np.zeros_like(
            closed
        )

        for label in range(1, count):
            area = int(
                stats[
                    label,
                    cv2.CC_STAT_AREA,
                ]
            )

            if (
                area
                >= self.config.min_region_area_px
            ):
                cleaned[
                    labels == label
                ] = 255

        return cleaned

    # ========================================================
    # SEGMENTATION
    # ========================================================

    def segment(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        self._validate_image(image)

        if (
            self.config.mode
            == FireDetectionMode.ONNX_SEGMENTATION
        ):
            if self._onnx_model is None:
                raise FireModelError(
                    "ONNX fire model unavailable"
                )

            mask = self._onnx_model.predict(
                image
            )

        else:
            mask = self._color_mask(
                image
            )

        return self._clean_mask(
            mask
        )

    # ========================================================
    # REGIONS
    # ========================================================

    @staticmethod
    def _extract_regions(
        mask: np.ndarray,
    ) -> list[FireRegion]:
        height, width = mask.shape[:2]

        total_area = width * height

        count, _, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )
        )

        regions: list[FireRegion] = []

        for label in range(1, count):
            x = int(
                stats[label, cv2.CC_STAT_LEFT]
            )

            y = int(
                stats[label, cv2.CC_STAT_TOP]
            )

            region_width = int(
                stats[label, cv2.CC_STAT_WIDTH]
            )

            region_height = int(
                stats[label, cv2.CC_STAT_HEIGHT]
            )

            area = int(
                stats[label, cv2.CC_STAT_AREA]
            )

            regions.append(
                FireRegion(
                    region_id=str(uuid.uuid4()),
                    x=x,
                    y=y,
                    width=region_width,
                    height=region_height,
                    pixel_area=area,
                    image_area_percent=(
                        area / total_area * 100.0
                    ),
                )
            )

        return sorted(
            regions,
            key=lambda region: (
                -region.pixel_area
            ),
        )

    # ========================================================
    # SCREENING LEVEL
    # ========================================================

    def _screening_level(
        self,
        area_percent: float,
    ) -> FireScreeningLevel:
        if area_percent <= 0:
            return FireScreeningLevel.NONE

        if (
            area_percent
            < self.config.moderate_area_percent
        ):
            return FireScreeningLevel.LOW

        if (
            area_percent
            < self.config.high_area_percent
        ):
            return FireScreeningLevel.MODERATE

        return FireScreeningLevel.HIGH

    # ========================================================
    # DETECTION
    # ========================================================

    def detect(
        self,
        image: np.ndarray,
        *,
        company_id: str,
        asset_id: str | None = None,
        equipment_id: str | None = None,
        camera_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FireDetectionResult:
        self._validate_company(company_id)
        self._validate_image(image)

        mask = self.segment(image)

        height, width = image.shape[:2]

        candidate_pixels = int(
            cv2.countNonZero(mask)
        )

        area_percent = (
            candidate_pixels
            / (width * height)
            * 100.0
        )

        regions = self._extract_regions(
            mask
        )

        screening_level = (
            self._screening_level(area_percent)
        )

        detected = bool(regions)

        return FireDetectionResult(
            inspection_id=str(uuid.uuid4()),
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
            camera_id=camera_id,
            fire_candidate_detected=detected,
            screening_level=(
                screening_level.value
            ),
            candidate_area_percent=round(
                area_percent,
                4,
            ),
            candidate_pixel_count=(
                candidate_pixels
            ),
            image_width=width,
            image_height=height,
            regions=regions,
            detection_mode=self.config.mode.value,
            inspected_at=datetime.now(
                timezone.utc
            ),
            requires_human_review=True,
            safety_escalation_recommended=detected,
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
        asset_id: str | None = None,
        equipment_id: str | None = None,
        camera_id: str | None = None,
    ) -> FireDetectionResult:
        image = self.load_image(path)

        return self.detect(
            image,
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
            camera_id=camera_id,
            metadata={
                "filename": Path(path).name
            },
        )

    def detect_bytes(
        self,
        data: bytes,
        *,
        company_id: str,
        asset_id: str | None = None,
        equipment_id: str | None = None,
        camera_id: str | None = None,
    ) -> FireDetectionResult:
        image = self.decode_image(data)

        return self.detect(
            image,
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
            camera_id=camera_id,
        )

    # ========================================================
    # ASYNC DETECTION
    # ========================================================

    async def detect_async(
        self,
        image: np.ndarray,
        *,
        company_id: str,
        asset_id: str | None = None,
        equipment_id: str | None = None,
        camera_id: str | None = None,
    ) -> FireDetectionResult:
        return await asyncio.to_thread(
            self.detect,
            image,
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
            camera_id=camera_id,
        )

    # ========================================================
    # BATCH
    # ========================================================

    def detect_batch(
        self,
        images: Sequence[np.ndarray],
        *,
        company_id: str,
        asset_id: str | None = None,
        equipment_id: str | None = None,
        camera_id: str | None = None,
    ) -> list[FireDetectionResult]:
        return [
            self.detect(
                image,
                company_id=company_id,
                asset_id=asset_id,
                equipment_id=equipment_id,
                camera_id=camera_id,
            )
            for image in images
        ]

    # ========================================================
    # ANNOTATION
    # ========================================================

    def annotate(
        self,
        image: np.ndarray,
        result: FireDetectionResult,
    ) -> np.ndarray:
        self._validate_image(image)

        if (
            image.shape[1] != result.image_width
            or image.shape[0] != result.image_height
        ):
            raise FireValidationError(
                "Image dimensions do not match result"
            )

        annotated = image.copy()

        for region in result.regions:
            cv2.rectangle(
                annotated,
                (region.x, region.y),
                (
                    region.x + region.width,
                    region.y + region.height,
                ),
                (0, 0, 255),
                2,
            )

            cv2.putText(
                annotated,
                "Flame candidate",
                (
                    region.x,
                    max(20, region.y - 8),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        status = (
            "FIRE CANDIDATE"
            if result.fire_candidate_detected
            else "NO CANDIDATE DETECTED"
        )

        cv2.putText(
            annotated,
            status,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255)
            if result.fire_candidate_detected
            else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return annotated


# ============================================================
# TEMPORAL MONITOR
# ============================================================

class FireTemporalMonitor:
    """
    Tracks detections over recent frames.

    Each (company_id, camera_id) has independent
    temporal state.

    Temporal confirmation only indicates repeated
    positive model outputs. It does not prove fire.

    A missing or failed frame must not be treated
    as evidence that the environment is safe.
    """

    def __init__(
        self,
        detector: FireDetector,
    ) -> None:
        self.detector = detector
        self.config = detector.config

        self._history: dict[
            tuple[str, str],
            deque[bool],
        ] = {}

        self._lock = threading.RLock()

    def inspect_frame(
        self,
        image: np.ndarray,
        *,
        company_id: str,
        camera_id: str,
        asset_id: str | None = None,
        equipment_id: str | None = None,
    ) -> FireTemporalResult:
        if not camera_id or not camera_id.strip():
            raise FireValidationError(
                "camera_id is required"
            )

        detection = self.detector.detect(
            image,
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
            camera_id=camera_id,
        )

        key = (company_id, camera_id)

        with self._lock:
            history = self._history.setdefault(
                key,
                deque(
                    maxlen=self.config.temporal_window
                ),
            )

            history.append(
                detection.fire_candidate_detected
            )

            observed = len(history)
            positives = sum(history)

            confirmed = (
                observed
                >= self.config.temporal_required_frames
                and positives
                >= self.config.temporal_required_frames
            )

        return FireTemporalResult(
            camera_id=camera_id,
            company_id=company_id,
            current_frame_detected=(
                detection.fire_candidate_detected
            ),
            candidate_frames_in_window=positives,
            frames_observed=observed,
            required_frames=(
                self.config.temporal_required_frames
            ),
            temporally_confirmed=confirmed,
            safety_escalation_recommended=(
                detection.fire_candidate_detected
                or confirmed
            ),
            current_detection=detection,
        )

    def reset(
        self,
        *,
        company_id: str,
        camera_id: str,
    ) -> None:
        with self._lock:
            self._history.pop(
                (company_id, camera_id),
                None,
            )


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    import json

    detector = FireDetector(
        FireDetectionConfig(
            mode=(
                FireDetectionMode.COLOR_HEURISTIC
            ),
            temporal_window=5,
            temporal_required_frames=3,
        )
    )

    image = detector.load_image(
        "industrial_fire_sample.jpg"
    )

    result = detector.detect(
        image,
        company_id="company-001",
        asset_id="ASSET-101",
        equipment_id="FURNACE-101",
        camera_id="CAM-01",
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
        "fire_detection_annotated.jpg",
        annotated,
    )

    monitor = FireTemporalMonitor(detector)

    temporal_result = monitor.inspect_frame(
        image,
        company_id="company-001",
        camera_id="CAM-01",
        equipment_id="FURNACE-101",
    )

    print(
        json.dumps(
            temporal_result.to_dict(),
            indent=2,
        )
    )


if __name__ == "__main__":
    example()
