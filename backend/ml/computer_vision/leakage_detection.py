
"""
app/ai/cv/leakage_detection.py

Industrial Leakage Detection
============================

Features:
- Visible liquid leakage candidate detection
- Reference-image change detection
- Wet/dark region screening
- Optional ONNX segmentation
- Leak candidate bounding boxes
- Affected image-area estimation
- Severity screening
- Batch and async inspection
- Annotated images and masks
- Asset, equipment and tenant metadata

Dependencies:
    pip install numpy opencv-python

Optional:
    pip install onnxruntime

IMPORTANT:
    This module is not a certified leak detector.

    Image-based screening cannot reliably detect
    invisible gas leaks, internal leaks, or
    determine fluid identity or leak rate.

    Heuristic mode requires a comparable
    reference image and may mistake lighting,
    shadows, reflections, dirt, or motion for
    leakage.

    For hazardous leaks, use appropriate
    certified sensors and inspection procedures.
"""

from __future__ import annotations

import asyncio
import logging
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

class LeakageDetectionError(Exception):
    """Base leakage detection exception."""


class LeakageValidationError(LeakageDetectionError):
    """Invalid input or configuration."""


class LeakageModelError(LeakageDetectionError):
    """Model loading or inference failure."""


# ============================================================
# ENUMS
# ============================================================

class LeakageDetectionMode(str, Enum):
    REFERENCE_COMPARISON = "reference_comparison"
    ONNX_SEGMENTATION = "onnx_segmentation"


class LeakageSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTENSIVE = "extensive"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(slots=True)
class LeakageDetectionConfig:
    mode: LeakageDetectionMode = (
        LeakageDetectionMode.REFERENCE_COMPARISON
    )

    min_image_width: int = 64
    min_image_height: int = 64
    max_image_pixels: int = 20_000_000

    min_region_area_px: int = 100
    morphology_kernel_size: int = 5

    blur_kernel_size: int = 5

    # Pixel-change threshold in grayscale units.
    difference_threshold: int = 30

    # Detect darkening relative to reference.
    min_darkening: int = 15

    # Reference alignment is not performed.
    # Both images must be captured with the
    # same framing, scale, and camera position.
    require_same_dimensions: bool = True

    moderate_area_percent: float = 2.0
    high_area_percent: float = 10.0
    extensive_area_percent: float = 25.0

    onnx_model_path: str | None = None
    onnx_input_width: int = 512
    onnx_input_height: int = 512
    onnx_probability_threshold: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            self.mode = LeakageDetectionMode(
                self.mode
            )

        for name in (
            "min_image_width",
            "min_image_height",
            "max_image_pixels",
            "min_region_area_px",
            "morphology_kernel_size",
            "blur_kernel_size",
            "onnx_input_width",
            "onnx_input_height",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise LeakageValidationError(
                    f"{name} must be a positive integer"
                )

        if self.blur_kernel_size % 2 == 0:
            raise LeakageValidationError(
                "blur_kernel_size must be odd"
            )

        if not (
            0 <= self.difference_threshold <= 255
        ):
            raise LeakageValidationError(
                "Invalid difference_threshold"
            )

        if not (
            0 <= self.min_darkening <= 255
        ):
            raise LeakageValidationError(
                "Invalid min_darkening"
            )

        if not (
            0 < self.onnx_probability_threshold < 1
        ):
            raise LeakageValidationError(
                "Invalid ONNX probability threshold"
            )

        if not (
            0
            < self.moderate_area_percent
            < self.high_area_percent
            < self.extensive_area_percent
            <= 100
        ):
            raise LeakageValidationError(
                "Area thresholds must be increasing"
            )


# ============================================================
# RESULT SCHEMAS
# ============================================================

@dataclass(slots=True)
class LeakageRegion:
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
class LeakageDetectionResult:
    inspection_id: str

    company_id: str
    asset_id: str | None
    equipment_id: str | None

    leakage_candidate_detected: bool
    severity: str

    affected_area_percent: float
    affected_pixel_count: int

    image_width: int
    image_height: int

    regions: list[LeakageRegion]

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
            "leakage_candidate_detected": (
                self.leakage_candidate_detected
            ),
            "severity": self.severity,
            "affected_area_percent": (
                self.affected_area_percent
            ),
            "affected_pixel_count": (
                self.affected_pixel_count
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


# ============================================================
# ONNX LEAKAGE SEGMENTER
# ============================================================

class ONNXLeakageSegmenter:
    """
    Expected ONNX contract:

    Input:
        RGB float32 NCHW [0,1]

    Output:
        Single-channel probability map.

    Adapt preprocessing and output handling
    to the actual trained model.
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
            raise LeakageModelError(
                "Install onnxruntime"
            ) from exc

        path = Path(model_path)

        if not path.is_file():
            raise LeakageModelError(
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
            raise LeakageModelError(
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
            raise LeakageModelError(
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
                raise LeakageModelError(
                    "Expected single-channel NCHW output"
                )

            probabilities = probabilities[0, 0]

        elif probabilities.ndim == 3:
            if probabilities.shape[0] != 1:
                raise LeakageModelError(
                    "Expected one output channel"
                )

            probabilities = probabilities[0]

        if probabilities.ndim != 2:
            raise LeakageModelError(
                "Unexpected segmentation output shape"
            )

        if not np.isfinite(
            probabilities
        ).all():
            raise LeakageModelError(
                "Invalid model output values"
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
# LEAKAGE DETECTOR
# ============================================================

class LeakageDetector:
    """
    Industrial visible leakage candidate detector.

    Reference-comparison mode detects darkened,
    changed regions between comparable images.

    It does not identify the fluid or confirm
    that the changes represent actual leakage.
    """

    def __init__(
        self,
        config: LeakageDetectionConfig | None = None,
    ) -> None:
        self.config = (
            config
            or LeakageDetectionConfig()
        )

        self._onnx_model: (
            ONNXLeakageSegmenter | None
        ) = None

        if (
            self.config.mode
            == LeakageDetectionMode.ONNX_SEGMENTATION
        ):
            if not self.config.onnx_model_path:
                raise LeakageValidationError(
                    "onnx_model_path is required"
                )

            self._onnx_model = (
                ONNXLeakageSegmenter(
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
        if not isinstance(image, np.ndarray):
            raise LeakageValidationError(
                "Image must be a NumPy array"
            )

        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
        ):
            raise LeakageValidationError(
                "Expected uint8 BGR image"
            )

        height, width = image.shape[:2]

        if (
            width < self.config.min_image_width
            or height < self.config.min_image_height
        ):
            raise LeakageValidationError(
                "Image dimensions too small"
            )

        if (
            width * height
            > self.config.max_image_pixels
        ):
            raise LeakageValidationError(
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
            raise LeakageValidationError(
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
            raise LeakageValidationError(
                "Image file not found"
            )

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise LeakageValidationError(
                "Unable to decode image"
            )

        self._validate_image(image)

        return image

    def decode_image(
        self,
        data: bytes,
    ) -> np.ndarray:
        if not data:
            raise LeakageValidationError(
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
            raise LeakageValidationError(
                "Invalid image data"
            )

        self._validate_image(image)

        return image

    # ========================================================
    # REFERENCE COMPARISON
    # ========================================================

    def _reference_mask(
        self,
        image: np.ndarray,
        reference_image: np.ndarray,
    ) -> np.ndarray:
        self._validate_image(
            reference_image
        )

        if image.shape != reference_image.shape:
            raise LeakageValidationError(
                "Reference and inspection images "
                "must have matching dimensions"
            )

        size = self.config.blur_kernel_size

        current_gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        reference_gray = cv2.cvtColor(
            reference_image,
            cv2.COLOR_BGR2GRAY,
        )

        current_gray = cv2.GaussianBlur(
            current_gray,
            (size, size),
            0,
        )

        reference_gray = cv2.GaussianBlur(
            reference_gray,
            (size, size),
            0,
        )

        difference = cv2.absdiff(
            current_gray,
            reference_gray,
        )

        darkening = cv2.subtract(
            reference_gray,
            current_gray,
        )

        _, change_mask = cv2.threshold(
            difference,
            self.config.difference_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        _, darkening_mask = cv2.threshold(
            darkening,
            self.config.min_darkening,
            255,
            cv2.THRESH_BINARY,
        )

        return cv2.bitwise_and(
            change_mask,
            darkening_mask,
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
    # SEGMENT
    # ========================================================

    def segment(
        self,
        image: np.ndarray,
        *,
        reference_image: np.ndarray | None = None,
    ) -> np.ndarray:
        self._validate_image(image)

        if (
            self.config.mode
            == LeakageDetectionMode.ONNX_SEGMENTATION
        ):
            if self._onnx_model is None:
                raise LeakageModelError(
                    "ONNX model unavailable"
                )

            mask = self._onnx_model.predict(
                image
            )

        else:
            if reference_image is None:
                raise LeakageValidationError(
                    "reference_image is required "
                    "in reference-comparison mode"
                )

            mask = self._reference_mask(
                image,
                reference_image,
            )

        return self._clean_mask(mask)

    # ========================================================
    # REGION EXTRACTION
    # ========================================================

    @staticmethod
    def _extract_regions(
        mask: np.ndarray,
    ) -> list[LeakageRegion]:
        height, width = mask.shape[:2]

        total_area = width * height

        count, _, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )
        )

        regions: list[LeakageRegion] = []

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
                LeakageRegion(
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
    # SEVERITY SCREENING
    # ========================================================

    def _classify_severity(
        self,
        area_percent: float,
    ) -> LeakageSeverity:
        if area_percent <= 0:
            return LeakageSeverity.NONE

        if (
            area_percent
            < self.config.moderate_area_percent
        ):
            return LeakageSeverity.LOW

        if (
            area_percent
            < self.config.high_area_percent
        ):
            return LeakageSeverity.MODERATE

        if (
            area_percent
            < self.config.extensive_area_percent
        ):
            return LeakageSeverity.HIGH

        return LeakageSeverity.EXTENSIVE

    # ========================================================
    # DETECT
    # ========================================================

    def detect(
        self,
        image: np.ndarray,
        *,
        company_id: str,
        reference_image: np.ndarray | None = None,
        asset_id: str | None = None,
        equipment_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LeakageDetectionResult:
        self._validate_company(company_id)
        self._validate_image(image)

        mask = self.segment(
            image,
            reference_image=reference_image,
        )

        height, width = image.shape[:2]

        affected_pixels = int(
            cv2.countNonZero(mask)
        )

        area_percent = (
            affected_pixels
            / (width * height)
            * 100.0
        )

        regions = self._extract_regions(mask)

        severity = self._classify_severity(
            area_percent
        )

        detected = bool(regions)

        return LeakageDetectionResult(
            inspection_id=str(uuid.uuid4()),
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
            leakage_candidate_detected=detected,
            severity=severity.value,
            affected_area_percent=round(
                area_percent,
                4,
            ),
            affected_pixel_count=(
                affected_pixels
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
    # FILE DETECTION
    # ========================================================

    def detect_file(
        self,
        path: str | Path,
        *,
        company_id: str,
        reference_path: str | Path | None = None,
        asset_id: str | None = None,
        equipment_id: str | None = None,
    ) -> LeakageDetectionResult:
        image = self.load_image(path)

        reference = (
            self.load_image(reference_path)
            if reference_path is not None
            else None
        )

        return self.detect(
            image,
            company_id=company_id,
            reference_image=reference,
            asset_id=asset_id,
            equipment_id=equipment_id,
            metadata={
                "filename": Path(path).name,
                "reference_filename": (
                    Path(reference_path).name
                    if reference_path is not None
                    else None
                ),
            },
        )

    # ========================================================
    # BYTE DETECTION
    # ========================================================

    def detect_bytes(
        self,
        data: bytes,
        *,
        company_id: str,
        reference_data: bytes | None = None,
        asset_id: str | None = None,
        equipment_id: str | None = None,
    ) -> LeakageDetectionResult:
        image = self.decode_image(data)

        reference = (
            self.decode_image(reference_data)
            if reference_data is not None
            else None
        )

        return self.detect(
            image,
            company_id=company_id,
            reference_image=reference,
            asset_id=asset_id,
            equipment_id=equipment_id,
        )

    # ========================================================
    # ASYNC DETECTION
    # ========================================================

    async def detect_async(
        self,
        image: np.ndarray,
        *,
        company_id: str,
        reference_image: np.ndarray | None = None,
        asset_id: str | None = None,
        equipment_id: str | None = None,
    ) -> LeakageDetectionResult:
        return await asyncio.to_thread(
            self.detect,
            image,
            company_id=company_id,
            reference_image=reference_image,
            asset_id=asset_id,
            equipment_id=equipment_id,
        )

    # ========================================================
    # BATCH DETECTION
    # ========================================================

    def detect_batch(
        self,
        images: Sequence[np.ndarray],
        *,
        company_id: str,
        reference_images: Sequence[
            np.ndarray
        ] | None = None,
        asset_id: str | None = None,
        equipment_id: str | None = None,
    ) -> list[LeakageDetectionResult]:
        if reference_images is not None:
            if (
                len(reference_images)
                != len(images)
            ):
                raise LeakageValidationError(
                    "Reference image count mismatch"
                )

        results = []

        for index, image in enumerate(images):
            reference = (
                reference_images[index]
                if reference_images is not None
                else None
            )

            results.append(
                self.detect(
                    image,
                    company_id=company_id,
                    reference_image=reference,
                    asset_id=asset_id,
                    equipment_id=equipment_id,
                )
            )

        return results

    # ========================================================
    # ANNOTATION
    # ========================================================

    def annotate(
        self,
        image: np.ndarray,
        result: LeakageDetectionResult,
    ) -> np.ndarray:
        self._validate_image(image)

        if (
            image.shape[1] != result.image_width
            or image.shape[0] != result.image_height
        ):
            raise LeakageValidationError(
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
                "Leak candidate",
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
                f"{result.affected_area_percent:.2f}%"
            ),
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

    detector = LeakageDetector(
        LeakageDetectionConfig(
            mode=(
                LeakageDetectionMode
                .REFERENCE_COMPARISON
            )
        )
    )

    result = detector.detect_file(
        "equipment_current.jpg",
        reference_path="equipment_reference.jpg",
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
        "equipment_current.jpg"
    )

    annotated = detector.annotate(
        image,
        result,
    )

    cv2.imwrite(
        "leakage_annotated.jpg",
        annotated,
    )


if __name__ == "__main__":
    example()
