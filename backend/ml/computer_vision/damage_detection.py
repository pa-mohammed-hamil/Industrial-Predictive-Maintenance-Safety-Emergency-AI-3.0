
"""
app/ai/cv/damage_detection.py

Industrial Equipment Damage Detection
=====================================

Python 3.11+

Capabilities
------------
- Industrial surface damage candidate detection
- Surface texture and edge anomaly screening
- Optional ONNX semantic segmentation
- Region bounding boxes
- Damage candidate area estimation
- Damage severity screening
- Batch equipment inspection
- Annotated image generation
- Asset and equipment metadata
- JSON-compatible results
- OpenCV-based processing

Dependencies:
    pip install numpy opencv-python

Optional:
    pip install onnxruntime

IMPORTANT:
    The default OpenCV detector is a heuristic.

    It detects visually unusual regions, not
    verified mechanical damage.

    It cannot reliably distinguish dents,
    cracks, corrosion, scratches, shadows,
    paint patterns, welds or manufacturing
    features without a trained model.

    Do not use this module alone to declare
    industrial equipment safe or unsafe.
"""

from __future__ import annotations

import logging
import uuid

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

class DamageDetectionError(Exception):
    """Base damage detection exception."""


class DamageValidationError(DamageDetectionError):
    """Invalid image or configuration."""


class DamageModelError(DamageDetectionError):
    """Model loading or inference failed."""


# ============================================================
# ENUMS
# ============================================================

class DamageDetectionMode(str, Enum):
    HEURISTIC = "heuristic"
    ONNX_SEGMENTATION = "onnx_segmentation"


class DamageSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTENSIVE = "extensive"


class DamageType(str, Enum):
    SURFACE_ANOMALY = "surface_anomaly"
    MODEL_DETECTED_DAMAGE = "model_detected_damage"
    UNKNOWN = "unknown"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(slots=True)
class DamageDetectionConfig:
    mode: DamageDetectionMode = (
        DamageDetectionMode.HEURISTIC
    )

    min_image_width: int = 64
    min_image_height: int = 64
    max_image_pixels: int = 20_000_000

    blur_kernel_size: int = 5

    canny_threshold_1: int = 50
    canny_threshold_2: int = 150

    local_contrast_kernel_size: int = 15

    anomaly_percentile: float = 90.0

    morphology_kernel_size: int = 5
    min_region_area_px: int = 150

    low_threshold_percent: float = 1.0
    moderate_threshold_percent: float = 5.0
    high_threshold_percent: float = 15.0
    extensive_threshold_percent: float = 35.0

    onnx_model_path: str | None = None
    onnx_input_width: int = 512
    onnx_input_height: int = 512
    onnx_probability_threshold: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            self.mode = DamageDetectionMode(
                self.mode
            )

        for name in (
            "min_image_width",
            "min_image_height",
            "max_image_pixels",
            "blur_kernel_size",
            "local_contrast_kernel_size",
            "morphology_kernel_size",
            "min_region_area_px",
            "onnx_input_width",
            "onnx_input_height",
        ):
            value = getattr(self, name)

            if value <= 0:
                raise DamageValidationError(
                    f"{name} must be positive"
                )

        if self.blur_kernel_size % 2 == 0:
            raise DamageValidationError(
                "blur_kernel_size must be odd"
            )

        if not 0 < self.anomaly_percentile < 100:
            raise DamageValidationError(
                "anomaly_percentile must be in (0,100)"
            )

        if not (
            0 < self.onnx_probability_threshold < 1
        ):
            raise DamageValidationError(
                "Invalid ONNX probability threshold"
            )

        thresholds = (
            self.low_threshold_percent,
            self.moderate_threshold_percent,
            self.high_threshold_percent,
            self.extensive_threshold_percent,
        )

        if thresholds != tuple(sorted(thresholds)):
            raise DamageValidationError(
                "Severity thresholds must be ascending"
            )


# ============================================================
# RESULT SCHEMAS
# ============================================================

@dataclass(slots=True)
class DamageRegion:
    region_id: str

    x: int
    y: int
    width: int
    height: int

    pixel_area: int
    image_area_percent: float

    damage_type: str

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
            "damage_type": self.damage_type,
        }


@dataclass(slots=True)
class DamageDetectionResult:
    inspection_id: str

    company_id: str
    asset_id: str | None
    equipment_id: str | None

    detected: bool
    severity: str

    damage_area_percent: float
    damage_pixel_count: int
    damage_region_count: int

    image_width: int
    image_height: int

    regions: list[DamageRegion]

    detection_mode: str
    inspected_at: datetime

    requires_human_review: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "inspection_id": self.inspection_id,
            "company_id": self.company_id,
            "asset_id": self.asset_id,
            "equipment_id": self.equipment_id,
            "detected": self.detected,
            "severity": self.severity,
            "damage_area_percent": (
                self.damage_area_percent
            ),
            "damage_pixel_count": (
                self.damage_pixel_count
            ),
            "damage_region_count": (
                self.damage_region_count
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
            "metadata": dict(self.metadata),
        }


# ============================================================
# ONNX DAMAGE SEGMENTER
# ============================================================

class ONNXDamageSegmenter:
    """
    Expected ONNX model contract:

    Input:
        NCHW RGB float32 [0,1]

    Output:
        Single-channel damage probability map.

    Adapt preprocessing and output handling
    to the trained model.
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
            raise DamageModelError(
                "Install onnxruntime"
            ) from exc

        path = Path(model_path)

        if not path.is_file():
            raise DamageModelError(
                "ONNX model not found"
            )

        try:
            self.session = ort.InferenceSession(
                str(path),
                providers=[
                    "CPUExecutionProvider"
                ],
            )
        except Exception as exc:
            raise DamageModelError(
                "Unable to load ONNX model"
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
        original_height, original_width = (
            image.shape[:2]
        )

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
            raise DamageModelError(
                "ONNX inference failed"
            ) from exc

        probabilities = np.asarray(
            output,
            dtype=np.float32,
        )

        if probabilities.ndim == 4:
            probabilities = probabilities[0, 0]

        elif probabilities.ndim == 3:
            probabilities = probabilities[0]

        if probabilities.ndim != 2:
            raise DamageModelError(
                "Unexpected model output shape"
            )

        if not np.isfinite(
            probabilities
        ).all():
            raise DamageModelError(
                "Invalid model output values"
            )

        probabilities = cv2.resize(
            probabilities,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR,
        )

        return (
            probabilities >= self.threshold
        ).astype(np.uint8) * 255


# ============================================================
# DAMAGE DETECTOR
# ============================================================

class DamageDetector:
    """
    Equipment surface damage candidate detector.
    """

    def __init__(
        self,
        config: DamageDetectionConfig | None = None,
    ) -> None:
        self.config = (
            config
            or DamageDetectionConfig()
        )

        self._onnx_model: (
            ONNXDamageSegmenter | None
        ) = None

        if (
            self.config.mode
            == DamageDetectionMode.ONNX_SEGMENTATION
        ):
            if not self.config.onnx_model_path:
                raise DamageValidationError(
                    "onnx_model_path is required"
                )

            self._onnx_model = (
                ONNXDamageSegmenter(
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
            raise DamageValidationError(
                "Image must be a NumPy array"
            )

        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
        ):
            raise DamageValidationError(
                "Expected uint8 BGR image"
            )

        height, width = image.shape[:2]

        if (
            width < self.config.min_image_width
            or height < self.config.min_image_height
        ):
            raise DamageValidationError(
                "Image dimensions are too small"
            )

        if (
            width * height
            > self.config.max_image_pixels
        ):
            raise DamageValidationError(
                "Image exceeds pixel limit"
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
            raise DamageValidationError(
                "Image file not found"
            )

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise DamageValidationError(
                "Unable to decode image"
            )

        self._validate_image(image)

        return image

    def decode_image(
        self,
        data: bytes,
    ) -> np.ndarray:
        if not data:
            raise DamageValidationError(
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
            raise DamageValidationError(
                "Invalid image data"
            )

        self._validate_image(image)

        return image

    # ========================================================
    # HEURISTIC DAMAGE MASK
    # ========================================================

    def _heuristic_mask(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        grayscale = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        blurred = cv2.GaussianBlur(
            grayscale,
            (
                self.config.blur_kernel_size,
                self.config.blur_kernel_size,
            ),
            0,
        )

        edges = cv2.Canny(
            blurred,
            self.config.canny_threshold_1,
            self.config.canny_threshold_2,
        )

        size = (
            self.config.local_contrast_kernel_size
        )

        background = cv2.GaussianBlur(
            blurred,
            (size | 1, size | 1),
            0,
        )

        difference = cv2.absdiff(
            blurred,
            background,
        )

        threshold = float(
            np.percentile(
                difference,
                self.config.anomaly_percentile,
            )
        )

        _, contrast_mask = cv2.threshold(
            difference,
            threshold,
            255,
            cv2.THRESH_BINARY,
        )

        combined = cv2.bitwise_or(
            edges,
            contrast_mask,
        )

        return combined

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

        closed = cv2.morphologyEx(
            mask,
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
    # SEGMENT
    # ========================================================

    def segment(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        self._validate_image(image)

        if (
            self.config.mode
            == DamageDetectionMode.ONNX_SEGMENTATION
        ):
            if self._onnx_model is None:
                raise DamageModelError(
                    "ONNX model unavailable"
                )

            mask = self._onnx_model.predict(
                image
            )

        else:
            mask = self._heuristic_mask(
                image
            )

        return self._clean_mask(
            mask
        )

    # ========================================================
    # REGIONS
    # ========================================================

    def _extract_regions(
        self,
        mask: np.ndarray,
    ) -> list[DamageRegion]:
        height, width = mask.shape[:2]

        image_area = width * height

        count, _, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )
        )

        damage_type = (
            DamageType.MODEL_DETECTED_DAMAGE.value
            if self.config.mode
            == DamageDetectionMode.ONNX_SEGMENTATION
            else DamageType.SURFACE_ANOMALY.value
        )

        regions: list[DamageRegion] = []

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
                DamageRegion(
                    region_id=str(uuid.uuid4()),
                    x=x,
                    y=y,
                    width=region_width,
                    height=region_height,
                    pixel_area=area,
                    image_area_percent=(
                        area / image_area * 100.0
                    ),
                    damage_type=damage_type,
                )
            )

        return sorted(
            regions,
            key=lambda region: (
                -region.pixel_area
            ),
        )

    # ========================================================
    # SEVERITY SCREENING
    # ========================================================

    def _classify_severity(
        self,
        percentage: float,
    ) -> DamageSeverity:
        if percentage == 0:
            return DamageSeverity.NONE

        if (
            percentage
            < self.config.moderate_threshold_percent
        ):
            return DamageSeverity.LOW

        if (
            percentage
            < self.config.high_threshold_percent
        ):
            return DamageSeverity.MODERATE

        if (
            percentage
            < self.config.extensive_threshold_percent
        ):
            return DamageSeverity.HIGH

        return DamageSeverity.EXTENSIVE

    # ========================================================
    # DETECT
    # ========================================================

    def detect(
        self,
        image: np.ndarray,
        *,
        company_id: str,
        asset_id: str | None = None,
        equipment_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DamageDetectionResult:
        if (
            not isinstance(company_id, str)
            or not company_id.strip()
        ):
            raise DamageValidationError(
                "company_id is required"
            )

        self._validate_image(image)

        mask = self.segment(
            image
        )

        height, width = image.shape[:2]

        pixel_count = int(
            cv2.countNonZero(mask)
        )

        percentage = (
            pixel_count
            / (width * height)
            * 100.0
        )

        regions = self._extract_regions(
            mask
        )

        severity = self._classify_severity(
            percentage
        )

        return DamageDetectionResult(
            inspection_id=str(uuid.uuid4()),
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
            detected=bool(regions),
            severity=severity.value,
            damage_area_percent=round(
                percentage,
                4,
            ),
            damage_pixel_count=pixel_count,
            damage_region_count=len(regions),
            image_width=width,
            image_height=height,
            regions=regions,
            detection_mode=self.config.mode.value,
            inspected_at=datetime.now(
                timezone.utc
            ),
            requires_human_review=True,
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
    ) -> DamageDetectionResult:
        image = self.load_image(
            path
        )

        return self.detect(
            image,
            company_id=company_id,
            asset_id=asset_id,
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
        asset_id: str | None = None,
        equipment_id: str | None = None,
    ) -> DamageDetectionResult:
        image = self.decode_image(
            data
        )

        return self.detect(
            image,
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
        )

    # ========================================================
    # ANNOTATION
    # ========================================================

    def annotate(
        self,
        image: np.ndarray,
        result: DamageDetectionResult,
    ) -> np.ndarray:
        self._validate_image(image)

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
                "Damage candidate",
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

        cv2.putText(
            annotated,
            (
                "Candidate area: "
                f"{result.damage_area_percent:.2f}%"
            ),
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return annotated

    # ========================================================
    # BATCH
    # ========================================================

    def detect_batch(
        self,
        images: Sequence[np.ndarray],
        *,
        company_id: str,
        asset_id: str | None = None,
    ) -> list[DamageDetectionResult]:
        return [
            self.detect(
                image,
                company_id=company_id,
                asset_id=asset_id,
            )
            for image in images
        ]


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    import json

    detector = DamageDetector(
        DamageDetectionConfig(
            mode=DamageDetectionMode.HEURISTIC
        )
    )

    result = detector.detect_file(
        "equipment_damage.jpg",
        company_id="company-001",
        asset_id="ASSET-101",
        equipment_id="MOTOR-101",
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
        )
    )

    image = detector.load_image(
        "equipment_damage.jpg"
    )

    annotated = detector.annotate(
        image,
        result,
    )

    cv2.imwrite(
        "damage_annotated.jpg",
        annotated,
    )


if __name__ == "__main__":
    example()
