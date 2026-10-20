
"""
app/ai/cv/overheating_detection.py

Industrial Overheating & Thermal Anomaly Detection
==================================================

Python 3.11+

Features
--------
- Calibrated thermal temperature-map analysis
- Equipment-specific temperature thresholds
- Absolute overheating threshold detection
- Relative thermal hotspot detection
- Connected-component hotspot extraction
- Bounding boxes and hotspot coordinates
- Maximum, minimum and mean temperature
- Hotspot area percentage
- Optional ONNX hotspot segmentation
- Thermal image annotation
- Batch and async processing
- Tenant-aware inspection results
- JSON-compatible outputs

Dependencies:
    pip install numpy opencv-python

Optional:
    pip install onnxruntime

IMPORTANT:
    A normal RGB photograph or false-color
    thermal JPEG does not provide calibrated
    physical temperatures.

    Real temperature measurements require
    calibrated radiometric thermal data or
    equivalent validated sensor measurements.

    Emissivity, reflections, distance, ambient
    conditions and sensor calibration affect
    thermal accuracy.

    This module is supplementary monitoring
    software, not a certified thermal protection
    or emergency shutdown system.
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

class OverheatingDetectionError(Exception):
    """Base thermal detection exception."""


class ThermalValidationError(
    OverheatingDetectionError
):
    """Invalid thermal input or configuration."""


class ThermalModelError(
    OverheatingDetectionError
):
    """ONNX model loading or inference error."""


# ============================================================
# ENUMS
# ============================================================

class ThermalDetectionMode(str, Enum):
    ABSOLUTE_TEMPERATURE = (
        "absolute_temperature"
    )

    RELATIVE_HOTSPOT = (
        "relative_hotspot"
    )

    COMBINED = "combined"

    ONNX_SEGMENTATION = (
        "onnx_segmentation"
    )


class ThermalSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(slots=True)
class OverheatingDetectionConfig:
    mode: ThermalDetectionMode = (
        ThermalDetectionMode.COMBINED
    )

    # Example screening thresholds only.
    # Replace with engineering-approved
    # equipment-specific limits.
    warning_temperature_c: float = 65.0
    high_temperature_c: float = 85.0
    critical_temperature_c: float = 105.0

    relative_hotspot_delta_c: float = 15.0

    min_image_width: int = 16
    min_image_height: int = 16
    max_image_pixels: int = 20_000_000

    min_hotspot_area_px: int = 4
    morphology_kernel_size: int = 3

    min_valid_temperature_c: float = -100.0
    max_valid_temperature_c: float = 2000.0

    onnx_model_path: str | None = None
    onnx_input_width: int = 512
    onnx_input_height: int = 512
    onnx_probability_threshold: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            self.mode = ThermalDetectionMode(
                self.mode
            )

        if not (
            self.warning_temperature_c
            < self.high_temperature_c
            < self.critical_temperature_c
        ):
            raise ThermalValidationError(
                "Temperature thresholds must increase"
            )

        if self.relative_hotspot_delta_c <= 0:
            raise ThermalValidationError(
                "relative_hotspot_delta_c must "
                "be positive"
            )

        for name in (
            "min_image_width",
            "min_image_height",
            "max_image_pixels",
            "min_hotspot_area_px",
            "morphology_kernel_size",
            "onnx_input_width",
            "onnx_input_height",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ThermalValidationError(
                    f"{name} must be a positive integer"
                )

        if not (
            self.min_valid_temperature_c
            < self.max_valid_temperature_c
        ):
            raise ThermalValidationError(
                "Invalid temperature measurement range"
            )

        if not (
            0 < self.onnx_probability_threshold < 1
        ):
            raise ThermalValidationError(
                "Invalid ONNX probability threshold"
            )


# ============================================================
# RESULT SCHEMAS
# ============================================================

@dataclass(slots=True)
class ThermalHotspot:
    hotspot_id: str

    x: int
    y: int
    width: int
    height: int

    pixel_area: int

    maximum_temperature_c: float
    mean_temperature_c: float

    image_area_percent: float

    severity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "hotspot_id": self.hotspot_id,
            "bounding_box": {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
            },
            "pixel_area": self.pixel_area,
            "maximum_temperature_c": (
                self.maximum_temperature_c
            ),
            "mean_temperature_c": (
                self.mean_temperature_c
            ),
            "image_area_percent": (
                self.image_area_percent
            ),
            "severity": self.severity,
        }


@dataclass(slots=True)
class OverheatingDetectionResult:
    inspection_id: str

    company_id: str
    asset_id: str | None
    equipment_id: str | None
    camera_id: str | None

    overheating_detected: bool
    severity: str

    minimum_temperature_c: float
    mean_temperature_c: float
    maximum_temperature_c: float

    hotspot_area_percent: float
    hotspot_pixel_count: int

    image_width: int
    image_height: int

    hotspots: list[ThermalHotspot]

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
            "overheating_detected": (
                self.overheating_detected
            ),
            "severity": self.severity,
            "minimum_temperature_c": (
                self.minimum_temperature_c
            ),
            "mean_temperature_c": (
                self.mean_temperature_c
            ),
            "maximum_temperature_c": (
                self.maximum_temperature_c
            ),
            "hotspot_area_percent": (
                self.hotspot_area_percent
            ),
            "hotspot_pixel_count": (
                self.hotspot_pixel_count
            ),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "hotspots": [
                hotspot.to_dict()
                for hotspot in self.hotspots
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
# ONNX HOTSPOT SEGMENTER
# ============================================================

class ONNXThermalSegmenter:
    """
    Expected model contract:

    Input:
        [1,1,H,W] normalized temperature map.

    Output:
        [1,1,H,W] hotspot probabilities.

    Input normalization must match training.

    ONNX predictions are not calibrated
    temperature measurements.
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
            raise ThermalModelError(
                "Install onnxruntime"
            ) from exc

        path = Path(model_path)

        if not path.is_file():
            raise ThermalModelError(
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
            raise ThermalModelError(
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
        temperature_map: np.ndarray,
        *,
        normalization_min_c: float,
        normalization_max_c: float,
    ) -> np.ndarray:
        height, width = (
            temperature_map.shape
        )

        resized = cv2.resize(
            temperature_map,
            (self.width, self.height),
            interpolation=cv2.INTER_LINEAR,
        )

        denominator = (
            normalization_max_c
            - normalization_min_c
        )

        if denominator <= 0:
            raise ThermalModelError(
                "Invalid normalization range"
            )

        normalized = np.clip(
            (
                resized
                - normalization_min_c
            ) / denominator,
            0.0,
            1.0,
        ).astype(np.float32)

        tensor = normalized[
            None,
            None,
            :,
            :,
        ]

        try:
            output = self.session.run(
                [self.output_name],
                {self.input_name: tensor},
            )[0]
        except Exception as exc:
            raise ThermalModelError(
                "ONNX inference failed"
            ) from exc

        probabilities = np.asarray(
            output,
            dtype=np.float32,
        )

        if probabilities.shape != (
            1,
            1,
            self.height,
            self.width,
        ):
            raise ThermalModelError(
                "Unexpected ONNX output shape"
            )

        probabilities = probabilities[0, 0]

        if not np.isfinite(
            probabilities
        ).all():
            raise ThermalModelError(
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
# THERMAL IMAGE UTILITIES
# ============================================================

class ThermalImageUtils:
    @staticmethod
    def normalize_for_display(
        temperature_map: np.ndarray,
    ) -> np.ndarray:
        minimum = float(
            np.min(temperature_map)
        )

        maximum = float(
            np.max(temperature_map)
        )

        if maximum <= minimum:
            normalized = np.zeros(
                temperature_map.shape,
                dtype=np.uint8,
            )

        else:
            normalized = (
                (
                    temperature_map - minimum
                )
                / (maximum - minimum)
                * 255.0
            ).astype(np.uint8)

        return cv2.applyColorMap(
            normalized,
            cv2.COLORMAP_INFERNO,
        )


# ============================================================
# MAIN OVERHEATING DETECTOR
# ============================================================

class OverheatingDetector:
    """
    Industrial thermal inspection engine.

    Accepts calibrated Celsius temperature
    maps as 2D NumPy arrays.

    The configured thresholds are illustrative
    and must be approved for each equipment
    type and operating condition.
    """

    SEVERITY_ORDER = {
        ThermalSeverity.NONE.value: 0,
        ThermalSeverity.LOW.value: 1,
        ThermalSeverity.MODERATE.value: 2,
        ThermalSeverity.HIGH.value: 3,
        ThermalSeverity.CRITICAL.value: 4,
    }

    def __init__(
        self,
        config: (
            OverheatingDetectionConfig | None
        ) = None,
    ) -> None:
        self.config = (
            config
            or OverheatingDetectionConfig()
        )

        self._onnx_model: (
            ONNXThermalSegmenter | None
        ) = None

        if (
            self.config.mode
            == ThermalDetectionMode.ONNX_SEGMENTATION
        ):
            if not self.config.onnx_model_path:
                raise ThermalValidationError(
                    "onnx_model_path is required"
                )

            self._onnx_model = (
                ONNXThermalSegmenter(
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

    def _validate_temperature_map(
        self,
        temperature_map: np.ndarray,
    ) -> np.ndarray:
        if not isinstance(
            temperature_map,
            np.ndarray,
        ):
            raise ThermalValidationError(
                "Temperature map must be NumPy array"
            )

        if temperature_map.ndim != 2:
            raise ThermalValidationError(
                "Expected 2D temperature map"
            )

        height, width = (
            temperature_map.shape
        )

        if (
            width < self.config.min_image_width
            or height < self.config.min_image_height
        ):
            raise ThermalValidationError(
                "Temperature map dimensions too small"
            )

        if (
            width * height
            > self.config.max_image_pixels
        ):
            raise ThermalValidationError(
                "Temperature map exceeds pixel limit"
            )

        try:
            values = temperature_map.astype(
                np.float32
            )
        except (TypeError, ValueError) as exc:
            raise ThermalValidationError(
                "Temperature map must be numeric"
            ) from exc

        if not np.isfinite(values).all():
            raise ThermalValidationError(
                "Temperature map contains NaN or infinity"
            )

        if (
            float(values.min())
            < self.config.min_valid_temperature_c
        ):
            raise ThermalValidationError(
                "Temperature below configured sensor range"
            )

        if (
            float(values.max())
            > self.config.max_valid_temperature_c
        ):
            raise ThermalValidationError(
                "Temperature above configured sensor range"
            )

        return values

    @staticmethod
    def _validate_company(
        company_id: str,
    ) -> None:
        if (
            not isinstance(company_id, str)
            or not company_id.strip()
        ):
            raise ThermalValidationError(
                "company_id is required"
            )

    # ========================================================
    # LOAD CALIBRATED TEMPERATURE DATA
    # ========================================================

    def load_temperature_map(
        self,
        path: str | Path,
    ) -> np.ndarray:
        path = Path(path)

        if not path.is_file():
            raise ThermalValidationError(
                "Temperature file not found"
            )

        if path.suffix.lower() == ".npy":
            values = np.load(
                path,
                allow_pickle=False,
            )

        elif path.suffix.lower() == ".csv":
            values = np.loadtxt(
                path,
                delimiter=",",
                dtype=np.float32,
            )

        else:
            raise ThermalValidationError(
                "Supported temperature formats: "
                ".npy, .csv"
            )

        return self._validate_temperature_map(
            values
        )

    # ========================================================
    # ABSOLUTE HOTSPOT MASK
    # ========================================================

    def _absolute_mask(
        self,
        temperature_map: np.ndarray,
    ) -> np.ndarray:
        return (
            temperature_map
            >= self.config.warning_temperature_c
        ).astype(np.uint8) * 255

    # ========================================================
    # RELATIVE HOTSPOT MASK
    # ========================================================

    def _relative_mask(
        self,
        temperature_map: np.ndarray,
    ) -> np.ndarray:
        baseline = float(
            np.median(temperature_map)
        )

        threshold = (
            baseline
            + self.config.relative_hotspot_delta_c
        )

        return (
            temperature_map >= threshold
        ).astype(np.uint8) * 255

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

        # Closing fills small holes while
        # retaining compact hotspot regions.
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
                >= self.config.min_hotspot_area_px
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
        temperature_map: np.ndarray,
    ) -> np.ndarray:
        values = self._validate_temperature_map(
            temperature_map
        )

        mode = self.config.mode

        if (
            mode
            == ThermalDetectionMode.ABSOLUTE_TEMPERATURE
        ):
            mask = self._absolute_mask(
                values
            )

        elif (
            mode
            == ThermalDetectionMode.RELATIVE_HOTSPOT
        ):
            mask = self._relative_mask(
                values
            )

        elif (
            mode
            == ThermalDetectionMode.COMBINED
        ):
            absolute = self._absolute_mask(
                values
            )

            relative = self._relative_mask(
                values
            )

            mask = cv2.bitwise_or(
                absolute,
                relative,
            )

        else:
            if self._onnx_model is None:
                raise ThermalModelError(
                    "ONNX model unavailable"
                )

            mask = self._onnx_model.predict(
                values,
                normalization_min_c=(
                    self.config
                    .min_valid_temperature_c
                ),
                normalization_max_c=(
                    self.config
                    .max_valid_temperature_c
                ),
            )

        return self._clean_mask(
            mask
        )

    # ========================================================
    # SEVERITY SCREENING
    # ========================================================

    def _classify_temperature(
        self,
        temperature_c: float,
    ) -> ThermalSeverity:
        if (
            temperature_c
            >= self.config.critical_temperature_c
        ):
            return ThermalSeverity.CRITICAL

        if (
            temperature_c
            >= self.config.high_temperature_c
        ):
            return ThermalSeverity.HIGH

        if (
            temperature_c
            >= self.config.warning_temperature_c
        ):
            return ThermalSeverity.MODERATE

        return ThermalSeverity.LOW

    # ========================================================
    # HOTSPOT EXTRACTION
    # ========================================================

    def _extract_hotspots(
        self,
        mask: np.ndarray,
        temperature_map: np.ndarray,
    ) -> list[ThermalHotspot]:
        height, width = mask.shape[:2]

        total_pixels = width * height

        count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )
        )

        hotspots: list[ThermalHotspot] = []

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

            region_values = temperature_map[
                labels == label
            ]

            maximum = float(
                np.max(region_values)
            )

            mean = float(
                np.mean(region_values)
            )

            severity = (
                self._classify_temperature(
                    maximum
                )
            )

            hotspots.append(
                ThermalHotspot(
                    hotspot_id=str(uuid.uuid4()),
                    x=x,
                    y=y,
                    width=region_width,
                    height=region_height,
                    pixel_area=area,
                    maximum_temperature_c=round(
                        maximum,
                        3,
                    ),
                    mean_temperature_c=round(
                        mean,
                        3,
                    ),
                    image_area_percent=round(
                        area / total_pixels * 100.0,
                        4,
                    ),
                    severity=severity.value,
                )
            )

        return sorted(
            hotspots,
            key=lambda item: (
                -item.maximum_temperature_c
            ),
        )

    # ========================================================
    # DETECT
    # ========================================================

    def detect(
        self,
        temperature_map: np.ndarray,
        *,
        company_id: str,
        asset_id: str | None = None,
        equipment_id: str | None = None,
        camera_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> OverheatingDetectionResult:
        self._validate_company(
            company_id
        )

        values = self._validate_temperature_map(
            temperature_map
        )

        mask = self.segment(
            values
        )

        height, width = values.shape

        minimum = float(
            np.min(values)
        )

        mean = float(
            np.mean(values)
        )

        maximum = float(
            np.max(values)
        )

        hotspot_pixels = int(
            cv2.countNonZero(mask)
        )

        area_percent = (
            hotspot_pixels
            / (width * height)
            * 100.0
        )

        hotspots = self._extract_hotspots(
            mask,
            values,
        )

        detected = bool(hotspots)

        if detected:
            severity = max(
                (
                    item.severity
                    for item in hotspots
                ),
                key=lambda value: (
                    self.SEVERITY_ORDER[value]
                ),
            )
        else:
            severity = (
                ThermalSeverity.NONE.value
            )

        # An absolute limit exceedance is
        # relevant even if morphology removes
        # an isolated hot pixel. Preserve
        # that fact in escalation logic.
        limit_exceeded = (
            maximum
            >= self.config.warning_temperature_c
        )

        escalation = (
            detected or limit_exceeded
        )

        return OverheatingDetectionResult(
            inspection_id=str(uuid.uuid4()),
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
            camera_id=camera_id,
            overheating_detected=(
                detected or limit_exceeded
            ),
            severity=(
                self._classify_temperature(
                    maximum
                ).value
                if limit_exceeded
                else severity
            ),
            minimum_temperature_c=round(
                minimum,
                3,
            ),
            mean_temperature_c=round(
                mean,
                3,
            ),
            maximum_temperature_c=round(
                maximum,
                3,
            ),
            hotspot_area_percent=round(
                area_percent,
                4,
            ),
            hotspot_pixel_count=(
                hotspot_pixels
            ),
            image_width=width,
            image_height=height,
            hotspots=hotspots,
            detection_mode=self.config.mode.value,
            inspected_at=datetime.now(
                timezone.utc
            ),
            requires_human_review=True,
            safety_escalation_recommended=(
                escalation
            ),
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
        camera_id: str | None = None,
    ) -> OverheatingDetectionResult:
        temperature_map = (
            self.load_temperature_map(path)
        )

        return self.detect(
            temperature_map,
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
            camera_id=camera_id,
            metadata={
                "filename": Path(path).name
            },
        )

    # ========================================================
    # BATCH DETECTION
    # ========================================================

    def detect_batch(
        self,
        temperature_maps: Sequence[
            np.ndarray
        ],
        *,
        company_id: str,
        asset_id: str | None = None,
        equipment_id: str | None = None,
    ) -> list[OverheatingDetectionResult]:
        return [
            self.detect(
                temperature_map,
                company_id=company_id,
                asset_id=asset_id,
                equipment_id=equipment_id,
            )
            for temperature_map in temperature_maps
        ]

    # ========================================================
    # ASYNC DETECTION
    # ========================================================

    async def detect_async(
        self,
        temperature_map: np.ndarray,
        *,
        company_id: str,
        asset_id: str | None = None,
        equipment_id: str | None = None,
        camera_id: str | None = None,
    ) -> OverheatingDetectionResult:
        return await asyncio.to_thread(
            self.detect,
            temperature_map,
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
            camera_id=camera_id,
        )

    # ========================================================
    # THERMAL ANNOTATION
    # ========================================================

    def annotate(
        self,
        temperature_map: np.ndarray,
        result: OverheatingDetectionResult,
    ) -> np.ndarray:
        values = self._validate_temperature_map(
            temperature_map
        )

        if (
            values.shape[1] != result.image_width
            or values.shape[0] != result.image_height
        ):
            raise ThermalValidationError(
                "Temperature map dimensions "
                "do not match result"
            )

        annotated = (
            ThermalImageUtils
            .normalize_for_display(
                values
            )
        )

        for hotspot in result.hotspots:
            color = (0, 0, 255)

            cv2.rectangle(
                annotated,
                (hotspot.x, hotspot.y),
                (
                    hotspot.x + hotspot.width,
                    hotspot.y + hotspot.height,
                ),
                color,
                2,
            )

            label = (
                f"{hotspot.maximum_temperature_c:.1f} C"
            )

            cv2.putText(
                annotated,
                label,
                (
                    hotspot.x,
                    max(20, hotspot.y - 8),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            annotated,
            (
                "Max: "
                f"{result.maximum_temperature_c:.1f} C"
            ),
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
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

    detector = OverheatingDetector(
        OverheatingDetectionConfig(
            mode=ThermalDetectionMode.COMBINED,
            warning_temperature_c=65.0,
            high_temperature_c=85.0,
            critical_temperature_c=105.0,
        )
    )

    # Example calibrated temperature data.
    # In production, obtain values from a
    # radiometric thermal sensor.
    temperature_map = np.full(
        (240, 320),
        35.0,
        dtype=np.float32,
    )

    temperature_map[
        80:130,
        120:180,
    ] = 92.0

    result = detector.detect(
        temperature_map,
        company_id="company-001",
        asset_id="ASSET-101",
        equipment_id="MOTOR-101",
        camera_id="THERMAL-CAM-01",
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
        )
    )

    annotated = detector.annotate(
        temperature_map,
        result,
    )

    cv2.imwrite(
        "overheating_annotated.jpg",
        annotated,
    )


if __name__ == "__main__":
    example()
