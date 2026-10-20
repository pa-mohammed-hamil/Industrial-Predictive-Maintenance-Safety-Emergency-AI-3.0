
"""
app/ai/cv/image_classifier.py

Industrial Image Classification Engine
======================================

Python 3.11+

Features:
- Industrial equipment image classification
- PyTorch / TorchVision models
- ResNet18, ResNet50, EfficientNet-B0
- Custom trained checkpoint loading
- ImageNet pretrained classification
- Top-K predictions
- Confidence probabilities
- Batch image classification
- CPU / CUDA inference
- Async inference
- Image file and byte inputs
- Tenant and equipment metadata
- JSON-compatible results

Dependencies:
    pip install torch torchvision pillow numpy

IMPORTANT:
    ImageNet pretrained models predict ImageNet
    categories, not industrial fault classes.

    For defect, safety, fire, corrosion, or
    equipment-condition classification, use a
    separately trained and validated checkpoint
    with matching class labels.

    Confidence scores are model probabilities,
    not guarantees of correctness.
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import threading
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms


logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class ImageClassificationError(Exception):
    """Base image classification exception."""


class ImageValidationError(ImageClassificationError):
    """Invalid image or request."""


class ImageModelError(ImageClassificationError):
    """Model loading or inference error."""


# ============================================================
# ENUMS
# ============================================================

class ClassificationArchitecture(str, Enum):
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    EFFICIENTNET_B0 = "efficientnet_b0"


class ClassificationMode(str, Enum):
    IMAGENET = "imagenet"
    CUSTOM = "custom"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(slots=True)
class ImageClassifierConfig:
    architecture: ClassificationArchitecture = (
        ClassificationArchitecture.RESNET18
    )

    mode: ClassificationMode = (
        ClassificationMode.IMAGENET
    )

    checkpoint_path: str | None = None

    class_names: list[str] = field(
        default_factory=list
    )

    device: str = "auto"

    image_size: int = 224

    batch_size: int = 16

    max_image_pixels: int = 20_000_000
    max_image_bytes: int = 20_000_000

    top_k: int = 5

    # ImageNet normalization.
    mean: tuple[float, float, float] = (
        0.485,
        0.456,
        0.406,
    )

    std: tuple[float, float, float] = (
        0.229,
        0.224,
        0.225,
    )

    def __post_init__(self) -> None:
        if isinstance(self.architecture, str):
            self.architecture = (
                ClassificationArchitecture(
                    self.architecture
                )
            )

        if isinstance(self.mode, str):
            self.mode = ClassificationMode(
                self.mode
            )

        for name in (
            "image_size",
            "batch_size",
            "max_image_pixels",
            "max_image_bytes",
            "top_k",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ImageValidationError(
                    f"{name} must be positive"
                )

        if self.mode == ClassificationMode.CUSTOM:
            if not self.checkpoint_path:
                raise ImageValidationError(
                    "Custom mode requires checkpoint_path"
                )

            if not self.class_names:
                raise ImageValidationError(
                    "Custom mode requires class_names"
                )

        if len(set(self.class_names)) != len(
            self.class_names
        ):
            raise ImageValidationError(
                "class_names must be unique"
            )


# ============================================================
# RESULT SCHEMAS
# ============================================================

@dataclass(slots=True)
class ClassificationPrediction:
    class_id: int
    label: str
    confidence: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "label": self.label,
            "confidence": self.confidence,
            "rank": self.rank,
        }


@dataclass(slots=True)
class ImageClassificationResult:
    prediction_id: str

    company_id: str
    asset_id: str | None
    equipment_id: str | None

    predicted_class: str
    confidence: float

    predictions: list[
        ClassificationPrediction
    ]

    architecture: str
    classification_mode: str

    image_width: int
    image_height: int

    predicted_at: datetime

    requires_human_review: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "company_id": self.company_id,
            "asset_id": self.asset_id,
            "equipment_id": self.equipment_id,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "predictions": [
                item.to_dict()
                for item in self.predictions
            ],
            "architecture": self.architecture,
            "classification_mode": (
                self.classification_mode
            ),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "predicted_at": (
                self.predicted_at.isoformat()
            ),
            "requires_human_review": (
                self.requires_human_review
            ),
            "metadata": dict(self.metadata),
        }


# ============================================================
# MODEL FACTORY
# ============================================================

class IndustrialModelFactory:
    """
    Constructs supported classification models.
    """

    @staticmethod
    def create(
        architecture: ClassificationArchitecture,
        *,
        num_classes: int,
        pretrained: bool,
    ) -> tuple[nn.Module, list[str]]:
        if architecture == (
            ClassificationArchitecture.RESNET18
        ):
            weights = (
                models.ResNet18_Weights.DEFAULT
                if pretrained
                else None
            )

            model = models.resnet18(
                weights=weights
            )

            if pretrained:
                return (
                    model,
                    list(weights.meta["categories"]),
                )

            model.fc = nn.Linear(
                model.fc.in_features,
                num_classes,
            )

            return model, []

        if architecture == (
            ClassificationArchitecture.RESNET50
        ):
            weights = (
                models.ResNet50_Weights.DEFAULT
                if pretrained
                else None
            )

            model = models.resnet50(
                weights=weights
            )

            if pretrained:
                return (
                    model,
                    list(weights.meta["categories"]),
                )

            model.fc = nn.Linear(
                model.fc.in_features,
                num_classes,
            )

            return model, []

        if architecture == (
            ClassificationArchitecture.EFFICIENTNET_B0
        ):
            weights = (
                models.EfficientNet_B0_Weights.DEFAULT
                if pretrained
                else None
            )

            model = models.efficientnet_b0(
                weights=weights
            )

            if pretrained:
                return (
                    model,
                    list(weights.meta["categories"]),
                )

            in_features = (
                model.classifier[-1].in_features
            )

            model.classifier[-1] = nn.Linear(
                in_features,
                num_classes,
            )

            return model, []

        raise ImageModelError(
            f"Unsupported architecture: {architecture}"
        )


# ============================================================
# IMAGE CLASSIFIER
# ============================================================

class IndustrialImageClassifier:
    """
    Industrial image classification service.

    Model initialization is lazy.

    Inference supports:
        - PIL Image
        - File paths
        - Encoded image bytes
        - Batches
    """

    def __init__(
        self,
        config: ImageClassifierConfig | None = None,
    ) -> None:
        self.config = (
            config
            or ImageClassifierConfig()
        )

        self.device = self._resolve_device(
            self.config.device
        )

        self._model: nn.Module | None = None

        self._class_names: list[str] = []

        self._load_lock = threading.RLock()
        self._inference_lock = threading.RLock()

        self.transform = transforms.Compose([
            transforms.Resize(
                (
                    self.config.image_size,
                    self.config.image_size,
                )
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.mean,
                std=self.config.std,
            ),
        ])

    # ========================================================
    # DEVICE
    # ========================================================

    @staticmethod
    def _resolve_device(
        device: str,
    ) -> torch.device:
        if device == "auto":
            return torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        resolved = torch.device(device)

        if (
            resolved.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise ImageModelError(
                "CUDA requested but unavailable"
            )

        return resolved

    # ========================================================
    # MODEL LOADING
    # ========================================================

    def _load_model(self) -> nn.Module:
        with self._load_lock:
            if self._model is not None:
                return self._model

            is_imagenet = (
                self.config.mode
                == ClassificationMode.IMAGENET
            )

            num_classes = (
                1000
                if is_imagenet
                else len(self.config.class_names)
            )

            try:
                model, labels = (
                    IndustrialModelFactory.create(
                        self.config.architecture,
                        num_classes=num_classes,
                        pretrained=is_imagenet,
                    )
                )

                if is_imagenet:
                    self._class_names = labels

                else:
                    self._class_names = list(
                        self.config.class_names
                    )

                    checkpoint_path = Path(
                        self.config.checkpoint_path
                    )

                    if not checkpoint_path.is_file():
                        raise ImageModelError(
                            "Checkpoint file not found"
                        )

                    # Only load trusted checkpoints.
                    checkpoint = torch.load(
                        str(checkpoint_path),
                        map_location="cpu",
                        weights_only=True,
                    )

                    if (
                        isinstance(checkpoint, dict)
                        and "state_dict" in checkpoint
                    ):
                        checkpoint = checkpoint[
                            "state_dict"
                        ]

                    if not isinstance(
                        checkpoint,
                        dict,
                    ):
                        raise ImageModelError(
                            "Expected a state_dict checkpoint"
                        )

                    model.load_state_dict(
                        checkpoint,
                        strict=True,
                    )

                model.to(self.device)
                model.eval()

            except ImageModelError:
                raise

            except Exception as exc:
                raise ImageModelError(
                    "Unable to initialize classifier"
                ) from exc

            self._model = model

            logger.info(
                "Image classifier loaded: %s",
                self.config.architecture.value,
            )

            return self._model

    # ========================================================
    # IMAGE VALIDATION
    # ========================================================

    def _validate_image(
        self,
        image: Image.Image,
    ) -> Image.Image:
        if not isinstance(
            image,
            Image.Image,
        ):
            raise ImageValidationError(
                "Expected PIL Image"
            )

        width, height = image.size

        if width <= 0 or height <= 0:
            raise ImageValidationError(
                "Invalid image dimensions"
            )

        if (
            width * height
            > self.config.max_image_pixels
        ):
            raise ImageValidationError(
                "Image exceeds pixel limit"
            )

        try:
            image = ImageOps.exif_transpose(
                image
            )

            return image.convert("RGB")

        except Exception as exc:
            raise ImageValidationError(
                "Unable to normalize image"
            ) from exc

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    def load_image(
        self,
        path: str | Path,
    ) -> Image.Image:
        path = Path(path)

        if not path.is_file():
            raise ImageValidationError(
                "Image file not found"
            )

        if (
            path.stat().st_size
            > self.config.max_image_bytes
        ):
            raise ImageValidationError(
                "Image file exceeds byte limit"
            )

        try:
            with Image.open(path) as image:
                return self._validate_image(
                    image
                )

        except (
            UnidentifiedImageError,
            OSError,
        ) as exc:
            raise ImageValidationError(
                "Unable to decode image"
            ) from exc

    def decode_image(
        self,
        data: bytes,
    ) -> Image.Image:
        if not data:
            raise ImageValidationError(
                "Image bytes cannot be empty"
            )

        if (
            len(data)
            > self.config.max_image_bytes
        ):
            raise ImageValidationError(
                "Image bytes exceed limit"
            )

        try:
            with Image.open(
                io.BytesIO(data)
            ) as image:
                return self._validate_image(
                    image
                )

        except (
            UnidentifiedImageError,
            OSError,
        ) as exc:
            raise ImageValidationError(
                "Invalid encoded image"
            ) from exc

    # ========================================================
    # INFERENCE
    # ========================================================

    def _predict_probabilities(
        self,
        images: Sequence[Image.Image],
    ) -> np.ndarray:
        if not images:
            return np.empty(
                (0, 0),
                dtype=np.float32,
            )

        model = self._load_model()

        batches: list[np.ndarray] = []

        with self._inference_lock:
            for start in range(
                0,
                len(images),
                self.config.batch_size,
            ):
                batch_images = images[
                    start:
                    start + self.config.batch_size
                ]

                tensors = [
                    self.transform(image)
                    for image in batch_images
                ]

                batch = torch.stack(
                    tensors
                ).to(self.device)

                with torch.inference_mode():
                    logits = model(batch)

                    if logits.ndim != 2:
                        raise ImageModelError(
                            "Expected [batch, classes] logits"
                        )

                    probabilities = (
                        torch.softmax(
                            logits,
                            dim=1,
                        )
                    )

                batches.append(
                    probabilities.cpu().numpy()
                )

        return np.concatenate(
            batches,
            axis=0,
        )

    # ========================================================
    # RESULT BUILDING
    # ========================================================

    def _build_result(
        self,
        *,
        image: Image.Image,
        probabilities: np.ndarray,
        company_id: str,
        asset_id: str | None,
        equipment_id: str | None,
        metadata: dict[str, Any] | None,
    ) -> ImageClassificationResult:
        if not np.isfinite(
            probabilities
        ).all():
            raise ImageModelError(
                "Nonfinite model probabilities"
            )

        if (
            len(probabilities)
            != len(self._class_names)
        ):
            raise ImageModelError(
                "Class count does not match model output"
            )

        top_k = min(
            self.config.top_k,
            len(self._class_names),
        )

        indices = np.argsort(
            probabilities
        )[::-1][:top_k]

        predictions = []

        for rank, index in enumerate(
            indices,
            start=1,
        ):
            class_id = int(index)

            predictions.append(
                ClassificationPrediction(
                    class_id=class_id,
                    label=(
                        self._class_names[class_id]
                    ),
                    confidence=float(
                        probabilities[class_id]
                    ),
                    rank=rank,
                )
            )

        if not predictions:
            raise ImageModelError(
                "No classification predictions"
            )

        top_prediction = predictions[0]

        width, height = image.size

        return ImageClassificationResult(
            prediction_id=str(uuid.uuid4()),
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
            predicted_class=(
                top_prediction.label
            ),
            confidence=(
                top_prediction.confidence
            ),
            predictions=predictions,
            architecture=(
                self.config.architecture.value
            ),
            classification_mode=(
                self.config.mode.value
            ),
            image_width=width,
            image_height=height,
            predicted_at=datetime.now(
                timezone.utc
            ),
            requires_human_review=True,
            metadata=dict(metadata or {}),
        )

    # ========================================================
    # CLASSIFY IMAGE
    # ========================================================

    def classify(
        self,
        image: Image.Image,
        *,
        company_id: str,
        asset_id: str | None = None,
        equipment_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ImageClassificationResult:
        if (
            not isinstance(company_id, str)
            or not company_id.strip()
        ):
            raise ImageValidationError(
                "company_id is required"
            )

        image = self._validate_image(
            image
        )

        probabilities = (
            self._predict_probabilities(
                [image]
            )
        )

        return self._build_result(
            image=image,
            probabilities=probabilities[0],
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
            metadata=metadata,
        )

    # ========================================================
    # CLASSIFY FILE / BYTES
    # ========================================================

    def classify_file(
        self,
        path: str | Path,
        *,
        company_id: str,
        asset_id: str | None = None,
        equipment_id: str | None = None,
    ) -> ImageClassificationResult:
        image = self.load_image(
            path
        )

        return self.classify(
            image,
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
            metadata={
                "filename": Path(path).name
            },
        )

    def classify_bytes(
        self,
        data: bytes,
        *,
        company_id: str,
        asset_id: str | None = None,
        equipment_id: str | None = None,
    ) -> ImageClassificationResult:
        image = self.decode_image(
            data
        )

        return self.classify(
            image,
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
        )

    # ========================================================
    # BATCH CLASSIFICATION
    # ========================================================

    def classify_batch(
        self,
        images: Sequence[Image.Image],
        *,
        company_id: str,
        asset_id: str | None = None,
        equipment_id: str | None = None,
    ) -> list[ImageClassificationResult]:
        if (
            not isinstance(company_id, str)
            or not company_id.strip()
        ):
            raise ImageValidationError(
                "company_id is required"
            )

        normalized_images = [
            self._validate_image(image)
            for image in images
        ]

        if not normalized_images:
            return []

        probabilities = (
            self._predict_probabilities(
                normalized_images
            )
        )

        return [
            self._build_result(
                image=image,
                probabilities=probabilities[index],
                company_id=company_id,
                asset_id=asset_id,
                equipment_id=equipment_id,
                metadata={
                    "batch_index": index
                },
            )
            for index, image in enumerate(
                normalized_images
            )
        ]

    # ========================================================
    # ASYNC CLASSIFICATION
    # ========================================================

    async def classify_async(
        self,
        image: Image.Image,
        *,
        company_id: str,
        asset_id: str | None = None,
        equipment_id: str | None = None,
    ) -> ImageClassificationResult:
        return await asyncio.to_thread(
            self.classify,
            image,
            company_id=company_id,
            asset_id=asset_id,
            equipment_id=equipment_id,
        )

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    def get_model_info(
        self,
    ) -> dict[str, Any]:
        return {
            "architecture": (
                self.config.architecture.value
            ),
            "mode": self.config.mode.value,
            "device": str(self.device),
            "loaded": self._model is not None,
            "class_count": (
                len(self._class_names)
                if self._model is not None
                else (
                    1000
                    if self.config.mode
                    == ClassificationMode.IMAGENET
                    else len(self.config.class_names)
                )
            ),
            "image_size": (
                self.config.image_size
            ),
            "batch_size": (
                self.config.batch_size
            ),
        }


# ============================================================
# EXAMPLE
# ============================================================

def example() -> None:
    import json

    classifier = IndustrialImageClassifier(
        ImageClassifierConfig(
            architecture=(
                ClassificationArchitecture.RESNET18
            ),
            mode=ClassificationMode.IMAGENET,
            top_k=5,
        )
    )

    result = classifier.classify_file(
        "industrial_equipment.jpg",
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


if __name__ == "__main__":
    example()
