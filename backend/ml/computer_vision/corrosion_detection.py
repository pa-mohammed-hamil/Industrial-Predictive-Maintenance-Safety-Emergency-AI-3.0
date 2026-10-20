
"""
app/ai/cv/corrosion_detection.py

Industrial Corrosion Detection & Segmentation
=============================================

Python 3.11+

Features:
- Industrial equipment image analysis
- Corrosion candidate segmentation
- Rust-colored region detection
- Affected surface area estimation
- Severity screening
- Morphological noise removal
- Bounding box extraction
- Annotated image output
- Batch image inspection
- Optional trained segmentation model (ONNX)
- Tenant/asset metadata
- JSON-compatible results

Dependencies:
    pip install numpy opencv-python

Optional:
    pip install onnxruntime

Important:
    Color-based detection is a heuristic.
    It cannot establish material loss, wall thickness,
    structural integrity, or actual corrosion depth.

    Production safety decisions require validated
    models and qualified inspection procedures.
"""

from __future__ import annotations

import math
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

class CorrosionDetectionError(Exception):
    """Base corrosion detection exception."""


class CorrosionValidationError(CorrosionDetectionError):
    """Invalid image or configuration."""


class CorrosionModelError(CorrosionDetectionError):
    """Model loading or inference failure."""


# ============================================================
# ENUMS
# ============================================================

class DetectionMode(str, Enum):
    COLOR_HEURISTIC = "color_heuristic"
    ONNX_SEGMENTATION = "onnx_segmentation"


class CorrosionSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTENSIVE = "extensive"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(slots=True)
class CorrosionDetectionConfig:
    mode: DetectionMode = DetectionMode.COLOR_HEURISTIC

    min_image_width: int = 64
    min_image_height: int = 64
    max_image_pixels: int = 20_000_000

    min_region_area_px: int = 100

    morphology_kernel_size: int = 5

    # HSV color ranges for rust-like appearance.
    # These values are heuristic and need calibration.
    hsv_lower_1: tuple[int, int, int] = (0, 45, 30)
    hsv_upper_1: tuple[int, int, int] = (25, 255, 230)

    hsv_lower_2: tuple[int, int, int] = (170, 45, 30)
    hsv_upper_2: tuple[int, int, int] = (179, 255, 230)

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
            self.mode = DetectionMode(self.mode)

        if self.min_image_width <= 0:
            raise CorrosionValidationError(
                "min_image_width must be positive"
            )

        if self.min_image_height <= 0:
            raise CorrosionValidationError(
                "min_image_height must be positive"
            )

        if self.max_image_pixels <= 0:
            raise CorrosionValidationError(
                "max_image_pixels must be positive"
            )

        if self.min_region_area_px <= 0:
            raise CorrosionValidationError(
                "min_region_area_px must be positive"
            )

        if self.morphology_kernel_size <= 0:
            raise CorrosionValidationError(
                "morphology_kernel_size must be positive"
            )

        if not (
            0.0
            < self.onnx_probability_threshold
            < 1.0
        ):
            raise CorrosionValidationError(
                "Invalid ONNX probability threshold"
            )

        thresholds = (
            self.low_threshold_percent,
            self.moderate_threshold_percent,
            self.high_threshold_percent,
            self.extensive_threshold_percent,
        )

        if tuple(sorted(thresholds)) != thresholds:
            raise CorrosionValidationError(
                "Severity thresholds must be ascending"
            )


# ============================================================
# RESULT SCHEMAS
# ============================================================

@dataclass(slots=True)
class CorrosionRegion:
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
class CorrosionDetectionResult:
    inspection_id: str

    company_id: str
    asset_id: str | None
    equipment_id: str | None

    detected: bool
    severity: str

    corrosion_area_percent: float
    corrosion_pixel_count: int

    image_width: int
    image_height: int

    regions: list[CorrosionRegion]

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
            "corrosion_area_percent": (
                self.corrosion_area_percent
            ),
            "corrosion_pixel_count": (
                self.corrosion_pixel_count
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
# ONNX MODEL ADAPTER
# ============================================================

class ONNXCorrosionSegmenter:
    """
    ONNX semantic segmentation adapter.

    Expected model:
        Input:
            NCHW float32 RGB tensor, normalized [0, 1]

        Output:
            Single-channel corrosion probability mask.

    Adjust preprocessing/postprocessing to match
    the actual trained model's contract.
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
            raise CorrosionModelError(
                "Install onnxruntime"
            ) from exc

        path = Path(model_path)

        if not path.is_file():
            raise CorrosionModelError(
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
            raise CorrosionModelError(
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
        image_bgr: np.ndarray,
    ) -> np.ndarray:
        original_height, original_width = (
            image_bgr.shape[:2]
        )

        resized = cv2.resize(
            image_bgr,
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
            raise CorrosionModelError(
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
            raise CorrosionModelError(
                "Unexpected segmentation output shape"
            )

        if not np.isfinite(
            probabilities
        ).all():
            raise CorrosionModelError(
                "Model output contains invalid values"
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
# CORROSION DETECTOR
# ============================================================

class CorrosionDetector:
    """
    Industrial corrosion screening service.
    """

    def __init__(
        self,
        config: CorrosionDetectionConfig | None = None,
    ) -> None:
        self.config = (
            config
            or CorrosionDetectionConfig()
        )

        self._onnx_segmenter: (
            ONNXCorrosionSegmenter | None
        ) = None

        if (
            self.config.mode
            == DetectionMode.ONNX_SEGMENTATION
        ):
            if not self.config.onnx_model_path:
                raise CorrosionValidationError(
                    "onnx_model_path is required"
                )

            self._onnx_segmenter = (
                ONNXCorrosionSegmenter(
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
    # IMAGE VALIDATION
    # ========================================================

    def _validate_image(
        self,
        image: np.ndarray,
    ) -> None:
        if not isinstance(
            image,
            np.ndarray,
        ):
            raise CorrosionValidationError(
                "Image must be a NumPy array"
            )

        if (
            image.ndim != 3
            or image.shape[2] != 3
        ):
            raise CorrosionValidationError(
                "Expected BGR image with 3 channels"
            )

        if image.dtype != np.uint8:
            raise CorrosionValidationError(
                "Expected uint8 image"
            )

        height, width = image.shape[:2]

        if (
            width < self.config.min_image_width
            or height < self.config.min_image_height
        ):
            raise CorrosionValidationError(
                "Image dimensions are too small"
            )

        if (
            width * height
            > self.config.max_image_pixels
        ):
            raise CorrosionValidationError(
                "Image exceeds pixel limit"
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
            raise CorrosionValidationError(
                "Image file not found"
            )

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise CorrosionValidationError(
                "Unable to decode image"
            )

        self._validate_image(image)

        return image

    def decode_image(
        self,
        data: bytes,
    ) -> np.ndarray:
        if not data:
            raise CorrosionValidationError(
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
            raise CorrosionValidationError(
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

        mask1 = cv2.inRange(
            hsv,
            np.array(
                self.config.hsv_lower_1,
                dtype=np.uint8,
            ),
            np.array(
                self.config.hsv_upper_1,
                dtype=np.uint8,
            ),
        )

        mask2 = cv2.inRange(
            hsv,
            np.array(
                self.config.hsv_lower_2,
                dtype=np.uint8,
            ),
            np.array(
                self.config.hsv_upper_2,
                dtype=np.uint8,
            ),
        )

        return cv2.bitwise_or(
            mask1,
            mask2,
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
            == DetectionMode.ONNX_SEGMENTATION
        ):
            if self._onnx_segmenter is None:
                raise CorrosionModelError(
                    "ONNX segmenter unavailable"
                )

            mask = self._onnx_segmenter.predict(
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
    ) -> list[CorrosionRegion]:
        height, width = mask.shape[:2]

        image_area = width * height

        count, _, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )
        )

        regions = []

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
                CorrosionRegion(
                    region_id=str(uuid.uuid4()),
                    x=x,
                    y=y,
                    width=region_width,
                    height=region_height,
                    pixel_area=area,
                    image_area_percent=(
                        area / image_area * 100.0
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
    # SEVERITY SCREENING
    # ========================================================

    def _classify_severity(
        self,
        percentage: float,
    ) -> CorrosionSeverity:
        if (
            percentage
            < self.config.low_threshold_percent
        ):
            return (
                CorrosionSeverity.NONE
                if percentage == 0
                else CorrosionSeverity.LOW
            )

        if (
            percentage
            < self.config.moderate_threshold_percent
        ):
            return CorrosionSeverity.LOW

        if (
            percentage
            < self.config.high_threshold_percent
        ):
            return CorrosionSeverity.MODERATE

        if (
            percentage
            < self.config.extensive_threshold_percent
        ):
            return CorrosionSeverity.HIGH

        return CorrosionSeverity.EXTENSIVE

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
    ) -> CorrosionDetectionResult:
        if not company_id or not company_id.strip():
            raise CorrosionValidationError(
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

        total_pixels = width * height

        percentage = (
            pixel_count
            / total_pixels
            * 100.0
        )

        regions = self._extract_regions(
            mask
        )

        severity = self._classify_severity(
            percentage
        )

        return CorrosionDetectionResult(
            inspection_id=str(uuid.uuid4()),
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
            detected=bool(regions),
            severity=severity.value,
            corrosion_area_percent=round(
                percentage,
                4,
            ),
            corrosion_pixel_count=pixel_count,
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
    # FILE DETECTION
    # ========================================================

    def detect_file(
        self,
        path: str | Path,
        *,
        company_id: str,
        asset_id: str | None = None,
        equipment_id: str | None = None,
    ) -> CorrosionDetectionResult:
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

    # ========================================================
    # BYTES DETECTION
    # ========================================================

    def detect_bytes(
        self,
        data: bytes,
        *,
        company_id: str,
        asset_id: str | None = None,
        equipment_id: str | None = None,
    ) -> CorrosionDetectionResult:
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
        result: CorrosionDetectionResult,
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
                "Corrosion candidate",
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
                f"{result.corrosion_area_percent:.2f}%"
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
    ) -> list[CorrosionDetectionResult]:
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

    detector = CorrosionDetector(
        CorrosionDetectionConfig(
            mode=DetectionMode.COLOR_HEURISTIC
        )
    )

    result = detector.detect_file(
        "sample_equipment.jpg",
        company_id="company-001",
        asset_id="ASSET-101",
        equipment_id="PIPE-101",
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
        )
    )

    image = detector.load_image(
        "sample_equipment.jpg"
    )

    annotated = detector.annotate(
        image,
        result,
    )

    cv2.imwrite(
        "corrosion_annotated.jpg",
        annotated,
    )


if __name__ == "__main__":
    example()
