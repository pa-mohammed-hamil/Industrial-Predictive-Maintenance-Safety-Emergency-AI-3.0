
"""
app/ai/cv/crack_detection.py

Industrial Surface Crack Detection
==================================

Python 3.11+

Features:
- Surface crack candidate detection
- Grayscale and contrast enhancement
- Black-hat morphology for dark linear features
- Adaptive thresholding
- Morphological noise filtering
- Connected-component analysis
- Crack skeletonization
- Pixel-based crack length estimation
- Bounding box extraction
- Visual severity screening
- Optional ONNX segmentation model
- Annotated images
- Batch inspection
- Tenant and asset metadata

Dependencies:
    pip install numpy opencv-python

Optional:
    pip install onnxruntime

IMPORTANT:
    This is a visual screening module.

    The heuristic detector may confuse scratches,
    joints, shadows, seams and surface markings
    with cracks.

    It cannot establish crack depth, material
    integrity, fracture risk or equipment safety.

    Human inspection and validated NDT methods
    are required for safety-critical decisions.
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

class CrackDetectionError(Exception):
    """Base crack detection exception."""


class CrackValidationError(CrackDetectionError):
    """Invalid image or configuration."""


class CrackModelError(CrackDetectionError):
    """Model loading or inference error."""


# ============================================================
# ENUMS
# ============================================================

class CrackDetectionMode(str, Enum):
    IMAGE_PROCESSING = "image_processing"
    ONNX_SEGMENTATION = "onnx_segmentation"


class CrackSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTENSIVE = "extensive"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(slots=True)
class CrackDetectionConfig:
    mode: CrackDetectionMode = (
        CrackDetectionMode.IMAGE_PROCESSING
    )

    min_image_width: int = 64
    min_image_height: int = 64
    max_image_pixels: int = 20_000_000

    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8

    blackhat_kernel_size: int = 17

    adaptive_block_size: int = 21
    adaptive_c: int = -3

    min_component_area: int = 15
    min_component_length_px: float = 15.0

    # A higher elongation requirement reduces
    # some blob-like false positives.
    min_elongation_ratio: float = 2.0

    morphology_kernel_size: int = 3

    low_length_threshold_px: float = 20.0
    moderate_length_threshold_px: float = 80.0
    high_length_threshold_px: float = 200.0
    extensive_length_threshold_px: float = 500.0

    onnx_model_path: str | None = None
    onnx_input_width: int = 512
    onnx_input_height: int = 512
    onnx_probability_threshold: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            self.mode = CrackDetectionMode(
                self.mode
            )

        if self.min_image_width <= 0:
            raise CrackValidationError(
                "min_image_width must be positive"
            )

        if self.min_image_height <= 0:
            raise CrackValidationError(
                "min_image_height must be positive"
            )

        if self.max_image_pixels <= 0:
            raise CrackValidationError(
                "max_image_pixels must be positive"
            )

        if self.blackhat_kernel_size <= 0:
            raise CrackValidationError(
                "blackhat_kernel_size must be positive"
            )

        if self.adaptive_block_size < 3:
            raise CrackValidationError(
                "adaptive_block_size must be >= 3"
            )

        if self.adaptive_block_size % 2 == 0:
            raise CrackValidationError(
                "adaptive_block_size must be odd"
            )

        if self.morphology_kernel_size <= 0:
            raise CrackValidationError(
                "morphology_kernel_size must be positive"
            )

        if self.min_component_area <= 0:
            raise CrackValidationError(
                "min_component_area must be positive"
            )

        if self.min_component_length_px <= 0:
            raise CrackValidationError(
                "min_component_length_px must be positive"
            )

        if self.min_elongation_ratio < 1.0:
            raise CrackValidationError(
                "min_elongation_ratio must be >= 1"
            )

        if not (
            0.0
            < self.onnx_probability_threshold
            < 1.0
        ):
            raise CrackValidationError(
                "Invalid ONNX probability threshold"
            )


# ============================================================
# RESULT SCHEMAS
# ============================================================

@dataclass(slots=True)
class CrackRegion:
    region_id: str

    x: int
    y: int
    width: int
    height: int

    pixel_area: int
    estimated_length_px: float

    elongation_ratio: float

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
            "estimated_length_px": (
                self.estimated_length_px
            ),
            "elongation_ratio": (
                self.elongation_ratio
            ),
        }


@dataclass(slots=True)
class CrackDetectionResult:
    inspection_id: str

    company_id: str
    asset_id: str | None
    equipment_id: str | None

    detected: bool
    severity: str

    crack_count: int
    total_crack_length_px: float
    maximum_crack_length_px: float

    image_width: int
    image_height: int

    regions: list[CrackRegion]

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
            "crack_count": self.crack_count,
            "total_crack_length_px": (
                self.total_crack_length_px
            ),
            "maximum_crack_length_px": (
                self.maximum_crack_length_px
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
# ONNX SEGMENTATION ADAPTER
# ============================================================

class ONNXCrackSegmenter:
    """
    Expected model contract:

    Input:
        NCHW RGB float32 tensor [0, 1]

    Output:
        Single-channel crack probability map.

    Preprocessing must be adapted to the actual
    trained model's input normalization.
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
            raise CrackModelError(
                "Install onnxruntime"
            ) from exc

        path = Path(model_path)

        if not path.is_file():
            raise CrackModelError(
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
            raise CrackModelError(
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
            raise CrackModelError(
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
            raise CrackModelError(
                "Unexpected ONNX output shape"
            )

        if not np.isfinite(
            probabilities
        ).all():
            raise CrackModelError(
                "ONNX output contains invalid values"
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
# SKELETONIZATION
# ============================================================

class CrackSkeletonizer:
    """
    Morphological skeletonization.

    Output:
        Binary skeleton with approximately
        one-pixel-wide lines.

    Skeleton pixel count is an approximate length
    measure, not calibrated physical length.
    """

    @staticmethod
    def skeletonize(
        mask: np.ndarray,
    ) -> np.ndarray:
        binary = (
            mask > 0
        ).astype(np.uint8) * 255

        skeleton = np.zeros_like(
            binary
        )

        element = cv2.getStructuringElement(
            cv2.MORPH_CROSS,
            (3, 3),
        )

        while cv2.countNonZero(binary) > 0:
            eroded = cv2.erode(
                binary,
                element,
            )

            opened = cv2.dilate(
                eroded,
                element,
            )

            residue = cv2.subtract(
                binary,
                opened,
            )

            skeleton = cv2.bitwise_or(
                skeleton,
                residue,
            )

            binary = eroded

        return skeleton


# ============================================================
# MAIN CRACK DETECTOR
# ============================================================

class CrackDetector:
    """
    Industrial surface crack candidate detector.
    """

    def __init__(
        self,
        config: CrackDetectionConfig | None = None,
    ) -> None:
        self.config = (
            config
            or CrackDetectionConfig()
        )

        self.skeletonizer = CrackSkeletonizer()

        self._onnx_model: (
            ONNXCrackSegmenter | None
        ) = None

        if (
            self.config.mode
            == CrackDetectionMode.ONNX_SEGMENTATION
        ):
            if not self.config.onnx_model_path:
                raise CrackValidationError(
                    "onnx_model_path is required"
                )

            self._onnx_model = (
                ONNXCrackSegmenter(
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
            raise CrackValidationError(
                "Image must be a NumPy array"
            )

        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
        ):
            raise CrackValidationError(
                "Expected uint8 BGR image"
            )

        height, width = image.shape[:2]

        if (
            width < self.config.min_image_width
            or height < self.config.min_image_height
        ):
            raise CrackValidationError(
                "Image dimensions too small"
            )

        if (
            width * height
            > self.config.max_image_pixels
        ):
            raise CrackValidationError(
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
            raise CrackValidationError(
                "Image file not found"
            )

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise CrackValidationError(
                "Unable to decode image"
            )

        self._validate_image(image)

        return image

    def decode_image(
        self,
        data: bytes,
    ) -> np.ndarray:
        if not data:
            raise CrackValidationError(
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
            raise CrackValidationError(
                "Invalid image data"
            )

        self._validate_image(image)

        return image

    # ========================================================
    # IMAGE PROCESSING
    # ========================================================

    def _heuristic_mask(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        grayscale = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        clahe = cv2.createCLAHE(
            clipLimit=(
                self.config.clahe_clip_limit
            ),
            tileGridSize=(
                self.config.clahe_tile_grid_size,
                self.config.clahe_tile_grid_size,
            ),
        )

        enhanced = clahe.apply(
            grayscale
        )

        size = (
            self.config.blackhat_kernel_size
        )

        blackhat_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (size, size),
            )
        )

        blackhat = cv2.morphologyEx(
            enhanced,
            cv2.MORPH_BLACKHAT,
            blackhat_kernel,
        )

        mask = cv2.adaptiveThreshold(
            blackhat,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.config.adaptive_block_size,
            self.config.adaptive_c,
        )

        return mask

    # ========================================================
    # CLEANING
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

        count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                opened,
                connectivity=8,
            )
        )

        cleaned = np.zeros_like(
            opened
        )

        for label in range(1, count):
            area = int(
                stats[
                    label,
                    cv2.CC_STAT_AREA,
                ]
            )

            width = int(
                stats[
                    label,
                    cv2.CC_STAT_WIDTH,
                ]
            )

            height = int(
                stats[
                    label,
                    cv2.CC_STAT_HEIGHT,
                ]
            )

            short_side = max(
                1,
                min(width, height),
            )

            long_side = max(
                width,
                height,
            )

            elongation = (
                long_side / short_side
            )

            if (
                area
                >= self.config.min_component_area
                and elongation
                >= self.config.min_elongation_ratio
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
            == CrackDetectionMode.ONNX_SEGMENTATION
        ):
            if self._onnx_model is None:
                raise CrackModelError(
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
    # REGION EXTRACTION
    # ========================================================

    def _extract_regions(
        self,
        mask: np.ndarray,
    ) -> list[CrackRegion]:
        count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )
        )

        regions: list[CrackRegion] = []

        for label in range(1, count):
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

            area = int(
                stats[label, cv2.CC_STAT_AREA]
            )

            component_mask = np.zeros_like(
                mask
            )

            component_mask[
                labels == label
            ] = 255

            skeleton = (
                self.skeletonizer.skeletonize(
                    component_mask
                )
            )

            estimated_length = float(
                cv2.countNonZero(
                    skeleton
                )
            )

            if (
                estimated_length
                < self.config.min_component_length_px
            ):
                continue

            short_side = max(
                1,
                min(width, height),
            )

            elongation = (
                max(width, height)
                / short_side
            )

            regions.append(
                CrackRegion(
                    region_id=str(uuid.uuid4()),
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    pixel_area=area,
                    estimated_length_px=(
                        estimated_length
                    ),
                    elongation_ratio=elongation,
                )
            )

        return sorted(
            regions,
            key=lambda region: (
                -region.estimated_length_px
            ),
        )

    # ========================================================
    # SEVERITY
    # ========================================================

    def _classify_severity(
        self,
        max_length: float,
    ) -> CrackSeverity:
        if max_length <= 0:
            return CrackSeverity.NONE

        if (
            max_length
            < self.config.moderate_length_threshold_px
        ):
            return CrackSeverity.LOW

        if (
            max_length
            < self.config.high_length_threshold_px
        ):
            return CrackSeverity.MODERATE

        if (
            max_length
            < self.config.extensive_length_threshold_px
        ):
            return CrackSeverity.HIGH

        return CrackSeverity.EXTENSIVE

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
        metadata: dict[str, Any] | None = None,
    ) -> CrackDetectionResult:
        if (
            not isinstance(company_id, str)
            or not company_id.strip()
        ):
            raise CrackValidationError(
                "company_id is required"
            )

        self._validate_image(image)

        mask = self.segment(
            image
        )

        regions = self._extract_regions(
            mask
        )

        lengths = [
            region.estimated_length_px
            for region in regions
        ]

        total_length = sum(
            lengths
        )

        maximum_length = max(
            lengths,
            default=0.0,
        )

        severity = self._classify_severity(
            maximum_length
        )

        height, width = image.shape[:2]

        return CrackDetectionResult(
            inspection_id=str(uuid.uuid4()),
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
            detected=bool(regions),
            severity=severity.value,
            crack_count=len(regions),
            total_crack_length_px=round(
                total_length,
                2,
            ),
            maximum_crack_length_px=round(
                maximum_length,
                2,
            ),
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
    # DETECT FILE
    # ========================================================

    def detect_file(
        self,
        path: str | Path,
        *,
        company_id: str,
        asset_id: str | None = None,
        equipment_id: str | None = None,
    ) -> CrackDetectionResult:
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
    # DETECT BYTES
    # ========================================================

    def detect_bytes(
        self,
        data: bytes,
        *,
        company_id: str,
        asset_id: str | None = None,
        equipment_id: str | None = None,
    ) -> CrackDetectionResult:
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
        result: CrackDetectionResult,
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

            label = (
                "Crack candidate "
                f"{region.estimated_length_px:.0f}px"
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
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            annotated,
            (
                f"Crack candidates: "
                f"{result.crack_count}"
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
    # BATCH DETECTION
    # ========================================================

    def detect_batch(
        self,
        images: Sequence[np.ndarray],
        *,
        company_id: str,
        asset_id: str | None = None,
    ) -> list[CrackDetectionResult]:
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

    detector = CrackDetector(
        CrackDetectionConfig(
            mode=(
                CrackDetectionMode.IMAGE_PROCESSING
            )
        )
    )

    result = detector.detect_file(
        "sample_crack.jpg",
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
        "sample_crack.jpg"
    )

    annotated = detector.annotate(
        image,
        result,
    )

    cv2.imwrite(
        "crack_annotated.jpg",
        annotated,
    )


if __name__ == "__main__":
    example()
