
"""
app/ai/cv/smoke_detection.py

Industrial Smoke Detection
==========================

Python 3.11+

Dependencies:
    pip install numpy opencv-python

Optional:
    pip install onnxruntime

Features:
- Smoke-colored region candidate detection
- Low-saturation gray/white smoke screening
- Optional ONNX smoke segmentation
- Connected-component region extraction
- Bounding boxes
- Candidate image-area percentage
- Temporal persistence monitoring
- Annotated frames
- File, bytes, batch and async inference
- Tenant and camera metadata
- JSON-compatible inspection results

SAFETY:
This module is not a certified smoke detector.

Color-based screening cannot reliably
distinguish smoke from fog, steam, dust,
clouds, glare, or gray equipment surfaces.

It can miss dark smoke and low-contrast smoke.

Use certified smoke/fire detectors and
independent safety systems for actual
industrial fire protection.
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
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class SmokeDetectionError(Exception):
    """Base smoke detection exception."""


class SmokeValidationError(SmokeDetectionError):
    """Invalid image or configuration."""


class SmokeModelError(SmokeDetectionError):
    """Smoke segmentation model failure."""


# ============================================================
# ENUMS
# ============================================================

class SmokeDetectionMode(str, Enum):
    COLOR_HEURISTIC = "color_heuristic"
    ONNX_SEGMENTATION = "onnx_segmentation"


class SmokeScreeningLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(slots=True)
class SmokeDetectionConfig:
    mode: SmokeDetectionMode = (
        SmokeDetectionMode.COLOR_HEURISTIC
    )

    min_image_width: int = 64
    min_image_height: int = 64
    max_image_pixels: int = 20_000_000

    min_region_area_px: int = 150
    morphology_kernel_size: int = 5

    # Smoke-like low-saturation pixels.
    max_saturation: int = 65
    min_brightness: int = 65
    max_brightness: int = 245

    # Ignore near-black and near-white pixels.
    min_gray_value: int = 65
    max_gray_value: int = 245

    moderate_area_percent: float = 3.0
    high_area_percent: float = 15.0

    temporal_window: int = 5
    temporal_required_frames: int = 3

    onnx_model_path: str | None = None
    onnx_input_width: int = 512
    onnx_input_height: int = 512
    onnx_probability_threshold: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            self.mode = SmokeDetectionMode(
                self.mode
            )

        for name in (
            "min_image_width",
            "min_image_height",
            "max_image_pixels",
            "min_region_area_px",
            "morphology_kernel_size",
            "temporal_window",
            "temporal_required_frames",
            "onnx_input_width",
            "onnx_input_height",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise SmokeValidationError(
                    f"{name} must be a positive integer"
                )

        if (
            self.temporal_required_frames
            > self.temporal_window
        ):
            raise SmokeValidationError(
                "Temporal required frames cannot "
                "exceed temporal window"
            )

        if not (
            0
            < self.moderate_area_percent
            < self.high_area_percent
            <= 100
        ):
            raise SmokeValidationError(
                "Invalid area thresholds"
            )

        if not (
            0 < self.onnx_probability_threshold < 1
        ):
            raise SmokeValidationError(
                "Invalid ONNX probability threshold"
            )


# ============================================================
# RESULT SCHEMAS
# ============================================================

@dataclass(slots=True)
class SmokeRegion:
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
class SmokeDetectionResult:
    inspection_id: str

    company_id: str
    asset_id: str | None
    equipment_id: str | None
    camera_id: str | None

    smoke_candidate_detected: bool
    screening_level: str

    candidate_area_percent: float
    candidate_pixel_count: int

    image_width: int
    image_height: int

    regions: list[SmokeRegion]

    detection_mode: str
    inspected_at: datetime

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
            "smoke_candidate_detected": (
                self.smoke_candidate_detected
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
class SmokeTemporalResult:
    company_id: str
    camera_id: str

    current_frame_detected: bool
    candidate_frames_in_window: int
    frames_observed: int
    required_frames: int

    temporally_confirmed: bool

    safety_escalation_recommended: bool

    current_detection: SmokeDetectionResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "company_id": self.company_id,
            "camera_id": self.camera_id,
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
# ONNX SMOKE SEGMENTER
# ============================================================

class ONNXSmokeSegmenter:
    """
    Expected model contract:

    Input:
        RGB float32 NCHW [0,1]

    Output:
        Single-channel smoke probability mask.

    Adapt normalization and output parsing
    to the actual trained model.
    """

    def __init__(
        self,
        model_path: str,
        *,
        width: int,
        height: int,
        threshold: float,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise SmokeModelError(
                "Install onnxruntime"
            ) from exc

        path = Path(model_path)

        if not path.is_file():
            raise SmokeModelError(
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
            raise SmokeModelError(
                "Unable to load ONNX smoke model"
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
            rgb.astype(np.float32)
            / 255.0
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
            raise SmokeModelError(
                "ONNX inference failed"
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
                raise SmokeModelError(
                    "Expected NCHW single-channel output"
                )

            probabilities = probabilities[0, 0]

        elif probabilities.ndim == 3:
            if probabilities.shape[0] != 1:
                raise SmokeModelError(
                    "Expected one output channel"
                )

            probabilities = probabilities[0]

        if probabilities.ndim != 2:
            raise SmokeModelError(
                "Unexpected model output shape"
            )

        if not np.isfinite(
            probabilities
        ).all():
            raise SmokeModelError(
                "Invalid model probabilities"
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
# SMOKE DETECTOR
# ============================================================

class SmokeDetector:
    """
    Industrial smoke candidate detector.

    Color heuristic mode identifies gray/white,
    low-saturation image regions.

    This is not physical smoke verification.
    """

    def __init__(
        self,
        config: SmokeDetectionConfig | None = None,
    ) -> None:
        self.config = (
            config
            or SmokeDetectionConfig()
        )

        self._onnx_model: (
            ONNXSmokeSegmenter | None
        ) = None

        if (
            self.config.mode
            == SmokeDetectionMode.ONNX_SEGMENTATION
        ):
            if not self.config.onnx_model_path:
                raise SmokeValidationError(
                    "onnx_model_path is required"
                )

            self._onnx_model = (
                ONNXSmokeSegmenter(
                    self.config.onnx_model_path,
                    width=(
                        self.config.onnx_input_width
                    ),
                    height=(
                        self.config.onnx_input_height
                    ),
                    threshold=(
                        self.config
                        .onnx_probability_threshold
                    ),
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
            raise SmokeValidationError(
                "Image must be NumPy array"
            )

        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
        ):
            raise SmokeValidationError(
                "Expected uint8 BGR image"
            )

        height, width = image.shape[:2]

        if (
            width < self.config.min_image_width
            or height < self.config.min_image_height
        ):
            raise SmokeValidationError(
                "Image dimensions too small"
            )

        if (
            width * height
            > self.config.max_image_pixels
        ):
            raise SmokeValidationError(
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
            raise SmokeValidationError(
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
            raise SmokeValidationError(
                "Image file not found"
            )

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise SmokeValidationError(
                "Unable to decode image"
            )

        self._validate_image(image)

        return image

    def decode_image(
        self,
        data: bytes,
    ) -> np.ndarray:
        if not data:
            raise SmokeValidationError(
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
            raise SmokeValidationError(
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

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        saturation = hsv[:, :, 1]
        brightness = hsv[:, :, 2]

        low_saturation = (
            saturation
            <= self.config.max_saturation
        )

        brightness_range = (
            (
                brightness
                >= self.config.min_brightness
            )
            &
            (
                brightness
                <= self.config.max_brightness
            )
        )

        gray_range = (
            (
                gray
                >= self.config.min_gray_value
            )
            &
            (
                gray
                <= self.config.max_gray_value
            )
        )

        candidate = (
            low_saturation
            & brightness_range
            & gray_range
        )

        return (
            candidate.astype(np.uint8)
            * 255
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
            == SmokeDetectionMode.ONNX_SEGMENTATION
        ):
            if self._onnx_model is None:
                raise SmokeModelError(
                    "ONNX model unavailable"
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
    # REGION EXTRACTION
    # ========================================================

    @staticmethod
    def _extract_regions(
        mask: np.ndarray,
    ) -> list[SmokeRegion]:
        height, width = mask.shape[:2]

        total_area = width * height

        count, _, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )
        )

        regions: list[SmokeRegion] = []

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
                SmokeRegion(
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
    ) -> SmokeScreeningLevel:
        if area_percent <= 0:
            return SmokeScreeningLevel.NONE

        if (
            area_percent
            < self.config.moderate_area_percent
        ):
            return SmokeScreeningLevel.LOW

        if (
            area_percent
            < self.config.high_area_percent
        ):
            return SmokeScreeningLevel.MODERATE

        return SmokeScreeningLevel.HIGH

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
        metadata: Mapping[str, Any] | None = None,
    ) -> SmokeDetectionResult:
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

        return SmokeDetectionResult(
            inspection_id=str(uuid.uuid4()),
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
            camera_id=camera_id,
            smoke_candidate_detected=detected,
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
    ) -> SmokeDetectionResult:
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
    ) -> SmokeDetectionResult:
        image = self.decode_image(data)

        return self.detect(
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
    ) -> list[SmokeDetectionResult]:
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
    # ASYNC
    # ========================================================

    async def detect_async(
        self,
        image: np.ndarray,
        *,
        company_id: str,
        asset_id: str | None = None,
        equipment_id: str | None = None,
        camera_id: str | None = None,
    ) -> SmokeDetectionResult:
        return await asyncio.to_thread(
            self.detect,
            image,
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
            camera_id=camera_id,
        )

    # ========================================================
    # ANNOTATION
    # ========================================================

    def annotate(
        self,
        image: np.ndarray,
        result: SmokeDetectionResult,
    ) -> np.ndarray:
        self._validate_image(image)

        if (
            image.shape[1] != result.image_width
            or image.shape[0] != result.image_height
        ):
            raise SmokeValidationError(
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
                (0, 165, 255),
                2,
            )

            cv2.putText(
                annotated,
                "Smoke candidate",
                (
                    region.x,
                    max(20, region.y - 8),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            annotated,
            (
                "Smoke candidate area: "
                f"{result.candidate_area_percent:.2f}%"
            ),
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return annotated


# ============================================================
# TEMPORAL MONITOR
# ============================================================

class SmokeTemporalMonitor:
    """
    Temporal smoke-candidate monitor.

    Tracks detection persistence per camera.

    Repeated positive frames are not proof
    of actual smoke.

    A missing frame or inference failure must
    not be interpreted as a safe condition.
    """

    def __init__(
        self,
        detector: SmokeDetector,
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
    ) -> SmokeTemporalResult:
        if not camera_id or not camera_id.strip():
            raise SmokeValidationError(
                "camera_id is required"
            )

        detection = self.detector.detect(
            image,
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
            camera_id=camera_id,
        )

        key = (
            company_id,
            camera_id,
        )

        with self._lock:
            history = self._history.setdefault(
                key,
                deque(
                    maxlen=self.config.temporal_window
                ),
            )

            history.append(
                detection.smoke_candidate_detected
            )

            positives = sum(history)
            observed = len(history)

            confirmed = (
                observed
                >= self.config.temporal_required_frames
                and positives
                >= self.config.temporal_required_frames
            )

        return SmokeTemporalResult(
            company_id=company_id,
            camera_id=camera_id,
            current_frame_detected=(
                detection.smoke_candidate_detected
            ),
            candidate_frames_in_window=positives,
            frames_observed=observed,
            required_frames=(
                self.config.temporal_required_frames
            ),
            temporally_confirmed=confirmed,
            safety_escalation_recommended=(
                detection.smoke_candidate_detected
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

    detector = SmokeDetector(
        SmokeDetectionConfig(
            mode=(
                SmokeDetectionMode.COLOR_HEURISTIC
            ),
            temporal_window=5,
            temporal_required_frames=3,
        )
    )

    image = detector.load_image(
        "industrial_smoke.jpg"
    )

    result = detector.detect(
        image,
        company_id="company-001",
        asset_id="ASSET-101",
        equipment_id="BOILER-101",
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
        "smoke_detection_annotated.jpg",
        annotated,
    )

    monitor = SmokeTemporalMonitor(
        detector
    )

    temporal = monitor.inspect_frame(
        image,
        company_id="company-001",
        camera_id="CAM-01",
        equipment_id="BOILER-101",
    )

    print(
        json.dumps(
            temporal.to_dict(),
            indent=2,
        )
    )


if __name__ == "__main__":
    example()
