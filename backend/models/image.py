"""
app/ai/ml/models/image.py

Industrial Image Intelligence Models
====================================

Python 3.11+
Pydantic v2

Install:
    pip install "pydantic>=2,<3"

This module stores validated image metadata and
computer vision results.

It does not decode image bytes, run inference,
or persist files.

Image storage and processing belong in:
    app/services/inspection_service.py
    app/ai/cv/
    app/repositories/
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================
# ENUMS
# ============================================================

class ImageType(str, Enum):
    RGB = "rgb"
    GRAYSCALE = "grayscale"
    THERMAL = "thermal"
    INFRARED = "infrared"
    MULTISPECTRAL = "multispectral"
    DOCUMENT = "document"
    OTHER = "other"


class ImageFormat(str, Enum):
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    TIFF = "tiff"
    BMP = "bmp"


class ImageSource(str, Enum):
    UPLOAD = "upload"
    CAMERA = "camera"
    CCTV = "cctv"
    DRONE = "drone"
    ROBOT = "robot"
    THERMAL_CAMERA = "thermal_camera"
    MOBILE = "mobile"
    INSPECTION_SYSTEM = "inspection_system"


class ImageProcessingStatus(str, Enum):
    UPLOADED = "uploaded"
    VALIDATING = "validating"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


class ImageInspectionType(str, Enum):
    CRACK = "crack"
    CORROSION = "corrosion"
    DAMAGE = "damage"
    DEFECT = "defect"
    LEAKAGE = "leakage"
    FIRE = "fire"
    SMOKE = "smoke"
    OVERHEATING = "overheating"
    PPE = "ppe"
    PERSON = "person"
    COMPONENT = "component"
    OCR = "ocr"
    GENERAL = "general"


class DetectionSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REINSPECTION = "needs_reinspection"


# ============================================================
# BASE MODEL
# ============================================================

class ImageBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
    )


# ============================================================
# IMAGE CONTEXT
# ============================================================

class ImageAssetContext(ImageBaseModel):
    company_id: str = Field(
        min_length=1,
        max_length=100,
    )

    asset_id: str | None = None
    equipment_id: str | None = None
    component_id: str | None = None

    site_id: str | None = None
    production_line_id: str | None = None

    inspection_id: str | None = None
    work_order_id: str | None = None

    camera_id: str | None = None


# ============================================================
# IMAGE DIMENSIONS
# ============================================================

class ImageDimensions(ImageBaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    channels: int = Field(
        default=3,
        ge=1,
        le=32,
    )

    @property
    def pixel_count(self) -> int:
        return self.width * self.height


# ============================================================
# IMAGE STORAGE
# ============================================================

class ImageStorage(ImageBaseModel):
    storage_provider: str = Field(
        min_length=1,
        max_length=100,
    )

    bucket: str | None = None

    object_key: str = Field(
        min_length=1,
        max_length=2048,
    )

    content_type: str = Field(
        min_length=1,
        max_length=100,
    )

    file_size_bytes: int = Field(
        ge=0,
    )

    sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-fA-F]{64}$",
    )

    encrypted: bool = True

    @field_validator("sha256")
    @classmethod
    def normalize_hash(
        cls,
        value: str,
    ) -> str:
        return value.lower()


# ============================================================
# BOUNDING BOX
# ============================================================

class ImageBoundingBox(ImageBaseModel):
    x_min: float = Field(ge=0, le=1)
    y_min: float = Field(ge=0, le=1)

    x_max: float = Field(ge=0, le=1)
    y_max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_coordinates(
        self,
    ) -> "ImageBoundingBox":
        if self.x_max <= self.x_min:
            raise ValueError(
                "x_max must exceed x_min"
            )

        if self.y_max <= self.y_min:
            raise ValueError(
                "y_max must exceed y_min"
            )

        return self

    @property
    def normalized_area(self) -> float:
        return (
            (self.x_max - self.x_min)
            * (self.y_max - self.y_min)
        )

    def to_pixel_coordinates(
        self,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        if width <= 0 or height <= 0:
            raise ValueError(
                "Image dimensions must be positive"
            )

        return (
            round(self.x_min * width),
            round(self.y_min * height),
            round(self.x_max * width),
            round(self.y_max * height),
        )


# ============================================================
# SEGMENTATION MASK REFERENCE
# ============================================================

class ImageSegmentationMask(ImageBaseModel):
    mask_id: UUID = Field(
        default_factory=uuid4
    )

    object_key: str = Field(
        min_length=1,
        max_length=2048,
    )

    width: int = Field(gt=0)
    height: int = Field(gt=0)

    encoding: str = "png"

    affected_area_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )


# ============================================================
# MODEL REFERENCE
# ============================================================

class ImageModelReference(ImageBaseModel):
    model_id: str = Field(
        min_length=1,
        max_length=200,
    )

    model_name: str = Field(
        min_length=1,
        max_length=200,
    )

    model_version: str = Field(
        min_length=1,
        max_length=100,
    )

    inference_run_id: str | None = None

    preprocessing_version: str | None = None

    confidence_threshold: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )


# ============================================================
# DETECTED OBJECT
# ============================================================

class ImageDetection(ImageBaseModel):
    detection_id: UUID = Field(
        default_factory=uuid4
    )

    inspection_type: ImageInspectionType

    class_name: str = Field(
        min_length=1,
        max_length=150,
    )

    confidence: float = Field(
        ge=0,
        le=1,
    )

    severity: DetectionSeverity = (
        DetectionSeverity.UNKNOWN
    )

    bounding_box: ImageBoundingBox | None = None

    segmentation_mask: (
        ImageSegmentationMask | None
    ) = None

    affected_area_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    component_id: str | None = None

    description: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @field_validator("metadata")
    @classmethod
    def validate_metadata(
        cls,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        # Keep metadata JSON-compatible and
        # reject NaN/Infinity.
        import json

        json.dumps(
            value,
            allow_nan=False,
        )

        return value


# ============================================================
# OCR RESULT
# ============================================================

class ImageOCRResult(ImageBaseModel):
    ocr_id: UUID = Field(
        default_factory=uuid4
    )

    extracted_text: str

    language: str | None = None

    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    detected_serial_numbers: list[str] = Field(
        default_factory=list
    )

    detected_part_numbers: list[str] = Field(
        default_factory=list
    )

    model_reference: (
        ImageModelReference | None
    ) = None


# ============================================================
# THERMAL MEASUREMENT
# ============================================================

class ImageThermalMeasurement(ImageBaseModel):
    measurement_id: UUID = Field(
        default_factory=uuid4
    )

    minimum_temperature_c: float

    maximum_temperature_c: float

    mean_temperature_c: float

    calibrated: bool = False

    sensor_reference: str | None = None

    calibration_reference: str | None = None

    @model_validator(mode="after")
    def validate_temperatures(
        self,
    ) -> "ImageThermalMeasurement":
        values = (
            self.minimum_temperature_c,
            self.maximum_temperature_c,
            self.mean_temperature_c,
        )

        if not all(
            math.isfinite(value)
            for value in values
        ):
            raise ValueError(
                "Thermal values must be finite"
            )

        if not (
            self.minimum_temperature_c
            <= self.mean_temperature_c
            <= self.maximum_temperature_c
        ):
            raise ValueError(
                "Temperature values are inconsistent"
            )

        if (
            self.calibrated
            and not self.calibration_reference
        ):
            raise ValueError(
                "Calibration reference is required "
                "for calibrated measurements"
            )

        return self


# ============================================================
# INSPECTION RESULT
# ============================================================

class ImageInspectionResult(ImageBaseModel):
    result_id: UUID = Field(
        default_factory=uuid4
    )

    inspection_type: ImageInspectionType

    model_reference: ImageModelReference

    detections: list[ImageDetection] = Field(
        default_factory=list
    )

    visual_health_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    thermal_measurement: (
        ImageThermalMeasurement | None
    ) = None

    ocr_result: ImageOCRResult | None = None

    inference_duration_ms: float | None = Field(
        default=None,
        ge=0,
    )

    requires_human_review: bool = True

    reviewed_by: str | None = None

    review_status: ReviewStatus = (
        ReviewStatus.PENDING
    )

    processed_at: datetime = Field(
        default_factory=utc_now
    )

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    @property
    def highest_severity(self) -> str:
        order = {
            "unknown": -1,
            "none": 0,
            "low": 1,
            "moderate": 2,
            "high": 3,
            "critical": 4,
        }

        if not self.detections:
            return "none"

        return max(
            (
                item.severity
                for item in self.detections
            ),
            key=lambda severity: order[
                severity
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")

        data["detection_count"] = (
            self.detection_count
        )

        data["highest_severity"] = (
            self.highest_severity
        )

        return data


# ============================================================
# MAIN IMAGE MODEL
# ============================================================

class IndustrialImage(ImageBaseModel):
    image_id: UUID = Field(
        default_factory=uuid4
    )

    context: ImageAssetContext

    filename: str = Field(
        min_length=1,
        max_length=255,
    )

    image_type: ImageType

    image_format: ImageFormat

    source: ImageSource

    dimensions: ImageDimensions

    storage: ImageStorage

    processing_status: ImageProcessingStatus = (
        ImageProcessingStatus.UPLOADED
    )

    inspection_results: list[
        ImageInspectionResult
    ] = Field(
        default_factory=list
    )

    captured_at: datetime | None = None

    uploaded_by: str | None = None

    uploaded_at: datetime = Field(
        default_factory=utc_now
    )

    processed_at: datetime | None = None

    error_message: str | None = None

    tags: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    created_at: datetime = Field(
        default_factory=utc_now
    )

    updated_at: datetime = Field(
        default_factory=utc_now
    )

    @field_validator(
        "captured_at",
        "uploaded_at",
        "processed_at",
        "created_at",
        "updated_at",
    )
    @classmethod
    def validate_datetime(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        if (
            value is not None
            and value.tzinfo is None
        ):
            raise ValueError(
                "Timestamps must be timezone-aware"
            )

        return value

    @field_validator("tags")
    @classmethod
    def normalize_tags(
        cls,
        values: list[str],
    ) -> list[str]:
        return list(
            dict.fromkeys(
                item.strip()
                for item in values
                if item.strip()
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json"
        )

    def add_inspection_result(
        self,
        result: ImageInspectionResult,
    ) -> None:
        if any(
            existing.result_id == result.result_id
            for existing in self.inspection_results
        ):
            raise ValueError(
                "Duplicate inspection result"
            )

        self.inspection_results.append(result)

        self.processing_status = (
            ImageProcessingStatus.COMPLETED
        )

        self.processed_at = utc_now()
        self.updated_at = utc_now()

    def mark_processing(self) -> None:
        if self.processing_status not in {
            ImageProcessingStatus.UPLOADED.value,
            ImageProcessingStatus.VALIDATING.value,
            ImageProcessingStatus.QUEUED.value,
        }:
            raise ValueError(
                "Image cannot enter processing "
                "from its current status"
            )

        self.processing_status = (
            ImageProcessingStatus.PROCESSING
        )

        self.updated_at = utc_now()

    def mark_failed(
        self,
        error_message: str,
    ) -> None:
        if not error_message.strip():
            raise ValueError(
                "Failure reason is required"
            )

        self.processing_status = (
            ImageProcessingStatus.FAILED
        )

        self.error_message = error_message
        self.updated_at = utc_now()

    def get_all_detections(
        self,
    ) -> list[ImageDetection]:
        return [
            detection
            for result in self.inspection_results
            for detection in result.detections
        ]

    def requires_review(self) -> bool:
        return any(
            result.requires_human_review
            and result.review_status
            == ReviewStatus.PENDING.value
            for result in self.inspection_results
        )


# ============================================================
# CREATE SCHEMA
# ============================================================

class ImageCreate(ImageBaseModel):
    context: ImageAssetContext

    filename: str = Field(
        min_length=1,
        max_length=255,
    )

    image_type: ImageType

    image_format: ImageFormat

    source: ImageSource

    dimensions: ImageDimensions

    storage: ImageStorage

    captured_at: datetime | None = None

    uploaded_by: str | None = None

    tags: list[str] = Field(
        default_factory=list
    )


# ============================================================
# UPDATE SCHEMA
# ============================================================

class ImageUpdate(ImageBaseModel):
    tags: list[str] | None = None

    metadata: dict[str, Any] | None = None


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class ImageResponse(IndustrialImage):
    pass


# ============================================================
# IMAGE SUMMARY
# ============================================================

class ImageSummary(ImageBaseModel):
    image_id: UUID

    company_id: str
    asset_id: str | None

    filename: str

    image_type: ImageType

    processing_status: ImageProcessingStatus

    inspection_count: int
    detection_count: int

    requires_human_review: bool

    uploaded_at: datetime

    @classmethod
    def from_image(
        cls,
        image: IndustrialImage,
    ) -> "ImageSummary":
        return cls(
            image_id=image.image_id,
            company_id=image.context.company_id,
            asset_id=image.context.asset_id,
            filename=image.filename,
            image_type=image.image_type,
            processing_status=image.processing_status,
            inspection_count=len(
                image.inspection_results
            ),
            detection_count=len(
                image.get_all_detections()
            ),
            requires_human_review=(
                image.requires_review()
            ),
            uploaded_at=image.uploaded_at,
        )


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    import json

    image = IndustrialImage(
        context=ImageAssetContext(
            company_id="company-001",
            asset_id="ASSET-101",
            equipment_id="PUMP-101",
            component_id="BEARING-101",
            site_id="SITE-01",
        ),
        filename="bearing_inspection.jpg",
        image_type=ImageType.RGB,
        image_format=ImageFormat.JPEG,
        source=ImageSource.UPLOAD,
        dimensions=ImageDimensions(
            width=1920,
            height=1080,
            channels=3,
        ),
        storage=ImageStorage(
            storage_provider="s3",
            bucket="industrial-inspection-images",
            object_key=(
                "company-001/ASSET-101/"
                "bearing_inspection.jpg"
            ),
            content_type="image/jpeg",
            file_size_bytes=245000,
            sha256="a" * 64,
        ),
    )

    image.mark_processing()

    # Illustrative detection, not actual CV inference.
    result = ImageInspectionResult(
        inspection_type=ImageInspectionType.CRACK,
        model_reference=ImageModelReference(
            model_id="crack-detector",
            model_name="Industrial Crack Detector",
            model_version="1.0.0",
        ),
        detections=[
            ImageDetection(
                inspection_type=(
                    ImageInspectionType.CRACK
                ),
                class_name="surface_crack",
                confidence=0.91,
                severity=DetectionSeverity.HIGH,
                bounding_box=ImageBoundingBox(
                    x_min=0.20,
                    y_min=0.25,
                    x_max=0.45,
                    y_max=0.55,
                ),
                description=(
                    "Candidate surface crack"
                ),
            )
        ],
        requires_human_review=True,
    )

    image.add_inspection_result(result)

    print(
        json.dumps(
            ImageSummary.from_image(
                image
            ).model_dump(mode="json"),
            indent=2,
        )
    )


if __name__ == "__main__":
    example()