
"""
app/ai/cv/defect_detection.py

Unified Industrial Defect Detection Engine
==========================================

Python 3.11+

Features
--------
- Crack candidate detection
- Corrosion candidate detection
- Surface damage candidate detection
- Multi-detector orchestration
- Optional ONNX models through detector adapters
- Defect region aggregation
- Duplicate region suppression
- Bounding box coordinates
- Annotated image generation
- Per-category visual severity screening
- Batch inspection
- JSON-compatible inspection reports
- Tenant-aware result metadata
- Async inspection support

Dependencies:
    pip install numpy opencv-python

Optional:
    pip install onnxruntime

Project modules:
    app.ai.cv.crack_detection
    app.ai.cv.corrosion_detection
    app.ai.cv.damage_detection

Safety:
    Detection results are screening indicators.
    They do not establish structural integrity,
    equipment fitness, or operational safety.
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

from app.ai.cv.crack_detection import (
    CrackDetector,
    CrackDetectionConfig,
)

from app.ai.cv.corrosion_detection import (
    CorrosionDetector,
    CorrosionDetectionConfig,
)

from app.ai.cv.damage_detection import (
    DamageDetector,
    DamageDetectionConfig,
)


logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class DefectDetectionError(Exception):
    """Base defect detection exception."""


class DefectValidationError(DefectDetectionError):
    """Invalid inspection input."""


class DefectProcessingError(DefectDetectionError):
    """Inspection processing failed."""


# ============================================================
# ENUMS
# ============================================================

class DefectType(str, Enum):
    CRACK = "crack"
    CORROSION = "corrosion"
    SURFACE_DAMAGE = "surface_damage"


class DefectSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTENSIVE = "extensive"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(slots=True)
class DefectDetectionConfig:
    enable_crack_detection: bool = True
    enable_corrosion_detection: bool = True
    enable_damage_detection: bool = True

    min_image_width: int = 64
    min_image_height: int = 64
    max_image_pixels: int = 20_000_000

    duplicate_iou_threshold: float = 0.85

    max_regions_per_type: int = 100

    def __post_init__(self) -> None:
        if not any((
            self.enable_crack_detection,
            self.enable_corrosion_detection,
            self.enable_damage_detection,
        )):
            raise DefectValidationError(
                "At least one detector must be enabled"
            )

        if not (
            0.0
            <= self.duplicate_iou_threshold
            <= 1.0
        ):
            raise DefectValidationError(
                "duplicate_iou_threshold must be "
                "between 0 and 1"
            )

        for name in (
            "min_image_width",
            "min_image_height",
            "max_image_pixels",
            "max_regions_per_type",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise DefectValidationError(
                    f"{name} must be a positive integer"
                )


# ============================================================
# DATA SCHEMAS
# ============================================================

@dataclass(slots=True)
class DefectRegion:
    region_id: str
    defect_type: str

    x: int
    y: int
    width: int
    height: int

    pixel_area: int

    severity: str

    attributes: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "defect_type": self.defect_type,
            "bounding_box": {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
            },
            "pixel_area": self.pixel_area,
            "severity": self.severity,
            "attributes": dict(self.attributes),
        }


@dataclass(slots=True)
class DefectCategorySummary:
    defect_type: str
    detected: bool
    severity: str
    region_count: int

    metrics: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "defect_type": self.defect_type,
            "detected": self.detected,
            "severity": self.severity,
            "region_count": self.region_count,
            "metrics": dict(self.metrics),
        }


@dataclass(slots=True)
class DefectDetectionResult:
    inspection_id: str

    company_id: str
    asset_id: str | None
    equipment_id: str | None

    image_width: int
    image_height: int

    detected: bool
    overall_severity: str

    total_regions: int

    regions: list[DefectRegion]

    summaries: list[DefectCategorySummary]

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
            "image_width": self.image_width,
            "image_height": self.image_height,
            "detected": self.detected,
            "overall_severity": (
                self.overall_severity
            ),
            "total_regions": self.total_regions,
            "regions": [
                item.to_dict()
                for item in self.regions
            ],
            "summaries": [
                item.to_dict()
                for item in self.summaries
            ],
            "inspected_at": (
                self.inspected_at.isoformat()
            ),
            "requires_human_review": (
                self.requires_human_review
            ),
            "metadata": dict(self.metadata),
        }


# ============================================================
# GEOMETRY
# ============================================================

class DefectGeometry:
    @staticmethod
    def intersection_over_union(
        a: DefectRegion,
        b: DefectRegion,
    ) -> float:
        ax1, ay1 = a.x, a.y
        ax2 = a.x + a.width
        ay2 = a.y + a.height

        bx1, by1 = b.x, b.y
        bx2 = b.x + b.width
        by2 = b.y + b.height

        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)

        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)

        intersection_width = max(
            0,
            x2 - x1,
        )

        intersection_height = max(
            0,
            y2 - y1,
        )

        intersection = (
            intersection_width
            * intersection_height
        )

        area_a = a.width * a.height
        area_b = b.width * b.height

        union = (
            area_a + area_b - intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union


# ============================================================
# MAIN DEFECT DETECTOR
# ============================================================

class IndustrialDefectDetector:
    """
    Unified industrial visual inspection service.

    Detector outputs are combined without
    assuming the individual models have
    calibrated or comparable confidence scores.
    """

    SEVERITY_ORDER = {
        DefectSeverity.NONE.value: 0,
        DefectSeverity.LOW.value: 1,
        DefectSeverity.MODERATE.value: 2,
        DefectSeverity.HIGH.value: 3,
        DefectSeverity.EXTENSIVE.value: 4,
    }

    COLORS = {
        DefectType.CRACK.value: (
            0, 0, 255
        ),
        DefectType.CORROSION.value: (
            0, 165, 255
        ),
        DefectType.SURFACE_DAMAGE.value: (
            255, 0, 255
        ),
    }

    def __init__(
        self,
        config: DefectDetectionConfig | None = None,
        *,
        crack_detector: CrackDetector | None = None,
        corrosion_detector: CorrosionDetector | None = None,
        damage_detector: DamageDetector | None = None,
    ) -> None:
        self.config = (
            config
            or DefectDetectionConfig()
        )

        self.crack_detector = (
            crack_detector
            or CrackDetector(
                CrackDetectionConfig()
            )
        )

        self.corrosion_detector = (
            corrosion_detector
            or CorrosionDetector(
                CorrosionDetectionConfig()
            )
        )

        self.damage_detector = (
            damage_detector
            or DamageDetector(
                DamageDetectionConfig()
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
            raise DefectValidationError(
                "Image must be a NumPy array"
            )

        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
        ):
            raise DefectValidationError(
                "Expected uint8 BGR image"
            )

        height, width = image.shape[:2]

        if (
            width < self.config.min_image_width
            or height < self.config.min_image_height
        ):
            raise DefectValidationError(
                "Image dimensions are too small"
            )

        if (
            width * height
            > self.config.max_image_pixels
        ):
            raise DefectValidationError(
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
            raise DefectValidationError(
                "company_id is required"
            )

    # ========================================================
    # IMAGE INPUT
    # ========================================================

    def load_image(
        self,
        path: str | Path,
    ) -> np.ndarray:
        path = Path(path)

        if not path.is_file():
            raise DefectValidationError(
                "Image file not found"
            )

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise DefectValidationError(
                "Unable to decode image"
            )

        self._validate_image(image)

        return image

    def decode_image(
        self,
        data: bytes,
    ) -> np.ndarray:
        if not data:
            raise DefectValidationError(
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
            raise DefectValidationError(
                "Invalid image data"
            )

        self._validate_image(image)

        return image

    # ========================================================
    # REGION CONVERSION
    # ========================================================

    @staticmethod
    def _convert_regions(
        source_regions: Sequence[Any],
        *,
        defect_type: DefectType,
        severity: str,
    ) -> list[DefectRegion]:
        converted = []

        for source in source_regions:
            attributes: dict[str, Any] = {}

            if hasattr(
                source,
                "estimated_length_px",
            ):
                attributes[
                    "estimated_length_px"
                ] = source.estimated_length_px

            if hasattr(
                source,
                "elongation_ratio",
            ):
                attributes[
                    "elongation_ratio"
                ] = source.elongation_ratio

            if hasattr(
                source,
                "image_area_percent",
            ):
                attributes[
                    "image_area_percent"
                ] = source.image_area_percent

            converted.append(
                DefectRegion(
                    region_id=str(uuid.uuid4()),
                    defect_type=defect_type.value,
                    x=int(source.x),
                    y=int(source.y),
                    width=int(source.width),
                    height=int(source.height),
                    pixel_area=int(
                        source.pixel_area
                    ),
                    severity=severity,
                    attributes=attributes,
                )
            )

        return converted

    # ========================================================
    # DUPLICATE SUPPRESSION
    # ========================================================

    def _suppress_duplicates(
        self,
        regions: Sequence[DefectRegion],
    ) -> list[DefectRegion]:
        """
        Suppress highly overlapping regions
        within the same defect category.

        Different defect types may overlap and
        are deliberately preserved.
        """
        sorted_regions = sorted(
            regions,
            key=lambda item: (
                -item.pixel_area,
                item.defect_type,
            ),
        )

        selected: list[DefectRegion] = []

        for candidate in sorted_regions:
            duplicate = False

            for existing in selected:
                if (
                    candidate.defect_type
                    != existing.defect_type
                ):
                    continue

                iou = (
                    DefectGeometry
                    .intersection_over_union(
                        candidate,
                        existing,
                    )
                )

                if (
                    iou
                    >= self.config.duplicate_iou_threshold
                ):
                    duplicate = True
                    break

            if not duplicate:
                selected.append(candidate)

        return selected

    # ========================================================
    # SEVERITY AGGREGATION
    # ========================================================

    def _overall_severity(
        self,
        summaries: Sequence[
            DefectCategorySummary
        ],
    ) -> str:
        if not summaries:
            return DefectSeverity.NONE.value

        return max(
            (
                item.severity
                for item in summaries
            ),
            key=lambda value: (
                self.SEVERITY_ORDER.get(
                    value,
                    0,
                )
            ),
        )

    # ========================================================
    # INSPECTION
    # ========================================================

    def detect(
        self,
        image: np.ndarray,
        *,
        company_id: str,
        asset_id: str | None = None,
        equipment_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DefectDetectionResult:
        self._validate_company(
            company_id
        )

        self._validate_image(
            image
        )

        regions: list[DefectRegion] = []
        summaries: list[DefectCategorySummary] = []

        # ----------------------------------------------------
        # CRACK DETECTION
        # ----------------------------------------------------

        if self.config.enable_crack_detection:
            crack_result = (
                self.crack_detector.detect(
                    image,
                    company_id=company_id,
                    asset_id=asset_id,
                    equipment_id=equipment_id,
                )
            )

            crack_regions = self._convert_regions(
                crack_result.regions,
                defect_type=DefectType.CRACK,
                severity=crack_result.severity,
            )

            regions.extend(
                crack_regions[
                    :self.config.max_regions_per_type
                ]
            )

            summaries.append(
                DefectCategorySummary(
                    defect_type=(
                        DefectType.CRACK.value
                    ),
                    detected=crack_result.detected,
                    severity=crack_result.severity,
                    region_count=(
                        crack_result.crack_count
                    ),
                    metrics={
                        "total_crack_length_px": (
                            crack_result
                            .total_crack_length_px
                        ),
                        "maximum_crack_length_px": (
                            crack_result
                            .maximum_crack_length_px
                        ),
                    },
                )
            )

        # ----------------------------------------------------
        # CORROSION DETECTION
        # ----------------------------------------------------

        if self.config.enable_corrosion_detection:
            corrosion_result = (
                self.corrosion_detector.detect(
                    image,
                    company_id=company_id,
                    asset_id=asset_id,
                    equipment_id=equipment_id,
                )
            )

            corrosion_regions = (
                self._convert_regions(
                    corrosion_result.regions,
                    defect_type=(
                        DefectType.CORROSION
                    ),
                    severity=(
                        corrosion_result.severity
                    ),
                )
            )

            regions.extend(
                corrosion_regions[
                    :self.config.max_regions_per_type
                ]
            )

            summaries.append(
                DefectCategorySummary(
                    defect_type=(
                        DefectType.CORROSION.value
                    ),
                    detected=(
                        corrosion_result.detected
                    ),
                    severity=(
                        corrosion_result.severity
                    ),
                    region_count=len(
                        corrosion_result.regions
                    ),
                    metrics={
                        "corrosion_area_percent": (
                            corrosion_result
                            .corrosion_area_percent
                        ),
                        "corrosion_pixel_count": (
                            corrosion_result
                            .corrosion_pixel_count
                        ),
                    },
                )
            )

        # ----------------------------------------------------
        # DAMAGE DETECTION
        # ----------------------------------------------------

        if self.config.enable_damage_detection:
            damage_result = (
                self.damage_detector.detect(
                    image,
                    company_id=company_id,
                    asset_id=asset_id,
                    equipment_id=equipment_id,
                )
            )

            damage_regions = self._convert_regions(
                damage_result.regions,
                defect_type=(
                    DefectType.SURFACE_DAMAGE
                ),
                severity=damage_result.severity,
            )

            regions.extend(
                damage_regions[
                    :self.config.max_regions_per_type
                ]
            )

            summaries.append(
                DefectCategorySummary(
                    defect_type=(
                        DefectType.SURFACE_DAMAGE.value
                    ),
                    detected=damage_result.detected,
                    severity=damage_result.severity,
                    region_count=(
                        damage_result
                        .damage_region_count
                    ),
                    metrics={
                        "damage_area_percent": (
                            damage_result
                            .damage_area_percent
                        ),
                        "damage_pixel_count": (
                            damage_result
                            .damage_pixel_count
                        ),
                    },
                )
            )

        # ----------------------------------------------------
        # AGGREGATION
        # ----------------------------------------------------

        regions = self._suppress_duplicates(
            regions
        )

        overall_severity = (
            self._overall_severity(
                summaries
            )
        )

        height, width = image.shape[:2]

        return DefectDetectionResult(
            inspection_id=str(uuid.uuid4()),
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
            image_width=width,
            image_height=height,
            detected=bool(regions),
            overall_severity=overall_severity,
            total_regions=len(regions),
            regions=regions,
            summaries=summaries,
            inspected_at=datetime.now(
                timezone.utc
            ),
            requires_human_review=True,
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
        asset_id: str | None = None,
        equipment_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DefectDetectionResult:
        return await asyncio.to_thread(
            self.detect,
            image,
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
            metadata=metadata,
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
    ) -> DefectDetectionResult:
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
    ) -> DefectDetectionResult:
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
    # BATCH INSPECTION
    # ========================================================

    def detect_batch(
        self,
        images: Sequence[np.ndarray],
        *,
        company_id: str,
        asset_id: str | None = None,
    ) -> list[DefectDetectionResult]:
        return [
            self.detect(
                image,
                company_id=company_id,
                asset_id=asset_id,
            )
            for image in images
        ]

    # ========================================================
    # IMAGE ANNOTATION
    # ========================================================

    def annotate(
        self,
        image: np.ndarray,
        result: DefectDetectionResult,
    ) -> np.ndarray:
        self._validate_image(
            image
        )

        annotated = image.copy()

        for region in result.regions:
            color = self.COLORS.get(
                region.defect_type,
                (255, 255, 255),
            )

            cv2.rectangle(
                annotated,
                (region.x, region.y),
                (
                    region.x + region.width,
                    region.y + region.height,
                ),
                color,
                2,
            )

            label = (
                f"{region.defect_type} "
                f"({region.severity})"
            )

            cv2.putText(
                annotated,
                label,
                (
                    region.x,
                    max(20, region.y - 8),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            annotated,
            (
                "Defect candidates: "
                f"{result.total_regions}"
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

    detector = IndustrialDefectDetector(
        DefectDetectionConfig(
            enable_crack_detection=True,
            enable_corrosion_detection=True,
            enable_damage_detection=True,
        )
    )

    result = detector.detect_file(
        "industrial_equipment.jpg",
        company_id="company-001",
        asset_id="ASSET-101",
        equipment_id="PUMP-201",
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
        )
    )

    image = detector.load_image(
        "industrial_equipment.jpg"
    )

    annotated = detector.annotate(
        image,
        result,
    )

    cv2.imwrite(
        "defect_inspection_annotated.jpg",
        annotated,
    )


if __name__ == "__main__":
    example()
