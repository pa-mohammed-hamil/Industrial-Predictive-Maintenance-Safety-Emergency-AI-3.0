
"""
tests/unit/test_smoke_detection.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

Smoke Detection Engine Unit Tests

Run:
    pytest tests/unit/test_smoke_detection.py -v

Coverage:
    - Smoke detection
    - Normal scene detection
    - Smoke confidence
    - Smoke bounding boxes
    - Smoke density
    - Severity classification
    - Safety alert decisions
    - RGB image validation
    - Grayscale image handling
    - Batch processing
    - Invalid frames
    - False-positive scenarios
    - Deterministic inference
    - Input immutability

IMPORTANT:
Connect SmokeDetectionAdapter to the actual
production smoke detection engine.

Synthetic images are used for basic contract
and robustness tests. They are not a validated
smoke/fire detection benchmark.

No physical alarms or emergency actuators
are triggered by these tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pytest


# ==================================================
# Domain models
# ==================================================

@dataclass(frozen=True)
class SmokeBox:
    x: int
    y: int
    width: int
    height: int
    confidence: float


@dataclass(frozen=True)
class SmokeDetectionResult:
    smoke_detected: bool
    confidence: float
    severity: str
    smoke_density: float
    alert_required: bool
    bounding_boxes: tuple[SmokeBox, ...] = ()


# ==================================================
# Production interface
# ==================================================

class SmokeDetectorProtocol(Protocol):

    def detect(
        self,
        frame: np.ndarray,
    ) -> SmokeDetectionResult:
        ...

    def detect_batch(
        self,
        frames: list[np.ndarray],
    ) -> list[SmokeDetectionResult]:
        ...


# ==================================================
# Production adapter
# ==================================================

class SmokeDetectionAdapter:
    """
    Connect to the actual smoke detection model.

    Expected module location:

        app/safety/smoke_engine.py

    Example:

        from app.safety.smoke_engine import (
            SmokeDetectionEngine
        )

        self.detector = SmokeDetectionEngine(...)

    Update constructor and method mappings to
    match the actual production implementation.

    Do not implement a second detection model
    inside this test adapter.
    """

    def __init__(self) -> None:
        self.detector: Any = None

        # TODO:
        # Initialize your production smoke detector.

    def detect(
        self,
        frame: np.ndarray,
    ) -> SmokeDetectionResult:

        if self.detector is None:
            raise RuntimeError(
                "Smoke detector is not configured"
            )

        # Example production mapping:
        #
        # raw = self.detector.detect(frame)
        #
        # boxes = tuple(
        #     SmokeBox(
        #         x=int(box["x"]),
        #         y=int(box["y"]),
        #         width=int(box["width"]),
        #         height=int(box["height"]),
        #         confidence=float(box["confidence"]),
        #     )
        #     for box in raw.get("bounding_boxes", [])
        # )
        #
        # return SmokeDetectionResult(
        #     smoke_detected=bool(
        #         raw["smoke_detected"]
        #     ),
        #     confidence=float(raw["confidence"]),
        #     severity=str(raw["severity"]),
        #     smoke_density=float(
        #         raw["smoke_density"]
        #     ),
        #     alert_required=bool(
        #         raw["alert_required"]
        #     ),
        #     bounding_boxes=boxes,
        # )

        raise NotImplementedError(
            "Map the production smoke detector output"
        )

    def detect_batch(
        self,
        frames: list[np.ndarray],
    ) -> list[SmokeDetectionResult]:

        return [
            self.detect(frame)
            for frame in frames
        ]


# ==================================================
# Fixtures
# ==================================================

@pytest.fixture
def detector() -> SmokeDetectorProtocol:

    adapter = SmokeDetectionAdapter()

    if adapter.detector is None:
        pytest.skip(
            "Connect SmokeDetectionAdapter to "
            "app/safety/smoke_engine.py"
        )

    return adapter


@pytest.fixture
def clean_frame() -> np.ndarray:

    return np.zeros(
        (224, 224, 3),
        dtype=np.uint8,
    )


@pytest.fixture
def gray_scene() -> np.ndarray:

    return np.full(
        (224, 224, 3),
        120,
        dtype=np.uint8,
    )


@pytest.fixture
def smoke_like_frame() -> np.ndarray:
    """
    Synthetic smoke-like visual pattern.

    This is NOT ground-truth smoke imagery.
    """

    frame = np.zeros(
        (224, 224, 3),
        dtype=np.uint8,
    )

    rng = np.random.default_rng(42)

    for y in range(40, 180):
        for x in range(60, 170):

            distance = (
                ((x - 115) / 55) ** 2
                + ((y - 110) / 70) ** 2
            )

            if distance < 1.0:
                intensity = int(
                    rng.integers(90, 190)
                )

                frame[y, x] = [
                    intensity,
                    intensity,
                    intensity,
                ]

    return frame


@pytest.fixture
def dense_smoke_like_frame() -> np.ndarray:

    frame = np.zeros(
        (224, 224, 3),
        dtype=np.uint8,
    )

    frame[20:200, 20:200] = 175

    return frame


@pytest.fixture
def bright_non_smoke_frame() -> np.ndarray:

    frame = np.zeros(
        (224, 224, 3),
        dtype=np.uint8,
    )

    frame[40:180, 40:180] = 255

    return frame


# ==================================================
# Basic detection
# ==================================================

class TestBasicSmokeDetection:

    def test_clean_scene_result(
        self,
        detector: SmokeDetectorProtocol,
        clean_frame: np.ndarray,
    ) -> None:

        result = detector.detect(clean_frame)

        assert isinstance(
            result,
            SmokeDetectionResult,
        )

    def test_clean_scene_no_smoke(
        self,
        detector: SmokeDetectorProtocol,
        clean_frame: np.ndarray,
    ) -> None:

        result = detector.detect(clean_frame)

        assert result.smoke_detected is False

    def test_smoke_like_scene_result(
        self,
        detector: SmokeDetectorProtocol,
        smoke_like_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            smoke_like_frame
        )

        assert isinstance(
            result.smoke_detected,
            bool,
        )

    def test_dense_smoke_like_scene_result(
        self,
        detector: SmokeDetectorProtocol,
        dense_smoke_like_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            dense_smoke_like_frame
        )

        assert isinstance(
            result,
            SmokeDetectionResult,
        )

    def test_result_boolean_fields(
        self,
        detector: SmokeDetectorProtocol,
        smoke_like_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            smoke_like_frame
        )

        assert isinstance(
            result.smoke_detected,
            bool,
        )

        assert isinstance(
            result.alert_required,
            bool,
        )


# ==================================================
# Confidence
# ==================================================

class TestSmokeConfidence:

    @pytest.mark.parametrize(
        "frame_type",
        [
            "clean",
            "gray",
            "smoke",
            "dense",
        ],
    )
    def test_confidence_range(
        self,
        detector: SmokeDetectorProtocol,
        clean_frame: np.ndarray,
        gray_scene: np.ndarray,
        smoke_like_frame: np.ndarray,
        dense_smoke_like_frame: np.ndarray,
        frame_type: str,
    ) -> None:

        frames = {
            "clean": clean_frame,
            "gray": gray_scene,
            "smoke": smoke_like_frame,
            "dense": dense_smoke_like_frame,
        }

        result = detector.detect(
            frames[frame_type]
        )

        assert math.isfinite(result.confidence)
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_not_nan(
        self,
        detector: SmokeDetectorProtocol,
        smoke_like_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            smoke_like_frame
        )

        assert not math.isnan(
            result.confidence
        )


# ==================================================
# Smoke severity
# ==================================================

class TestSmokeSeverity:

    def test_valid_severity(
        self,
        detector: SmokeDetectorProtocol,
        smoke_like_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            smoke_like_frame
        )

        assert result.severity.lower() in {
            "none",
            "normal",
            "low",
            "medium",
            "high",
            "critical",
            "warning",
        }

    def test_clean_scene_no_critical_alert(
        self,
        detector: SmokeDetectorProtocol,
        clean_frame: np.ndarray,
    ) -> None:

        result = detector.detect(clean_frame)

        assert result.severity.lower() != "critical"

    def test_alert_field_is_boolean(
        self,
        detector: SmokeDetectorProtocol,
        dense_smoke_like_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            dense_smoke_like_frame
        )

        assert isinstance(
            result.alert_required,
            bool,
        )


# ==================================================
# Smoke density
# ==================================================

class TestSmokeDensity:

    def test_density_nonnegative(
        self,
        detector: SmokeDetectorProtocol,
        smoke_like_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            smoke_like_frame
        )

        assert result.smoke_density >= 0

    def test_density_finite(
        self,
        detector: SmokeDetectorProtocol,
        dense_smoke_like_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            dense_smoke_like_frame
        )

        assert math.isfinite(
            result.smoke_density
        )

    def test_clean_scene_density(
        self,
        detector: SmokeDetectorProtocol,
        clean_frame: np.ndarray,
    ) -> None:

        result = detector.detect(clean_frame)

        assert result.smoke_density >= 0

    def test_density_result_numeric(
        self,
        detector: SmokeDetectorProtocol,
        smoke_like_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            smoke_like_frame
        )

        assert isinstance(
            result.smoke_density,
            (int, float),
        )


# ==================================================
# Bounding boxes
# ==================================================

class TestSmokeBoundingBoxes:

    def test_box_dimensions_valid(
        self,
        detector: SmokeDetectorProtocol,
        smoke_like_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            smoke_like_frame
        )

        for box in result.bounding_boxes:
            assert box.width > 0
            assert box.height > 0
            assert box.x >= 0
            assert box.y >= 0

    def test_boxes_within_frame(
        self,
        detector: SmokeDetectorProtocol,
        smoke_like_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            smoke_like_frame
        )

        height, width = smoke_like_frame.shape[:2]

        for box in result.bounding_boxes:
            assert box.x + box.width <= width
            assert box.y + box.height <= height

    def test_box_confidence_range(
        self,
        detector: SmokeDetectorProtocol,
        smoke_like_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            smoke_like_frame
        )

        for box in result.bounding_boxes:
            assert 0 <= box.confidence <= 1

    def test_box_type(
        self,
        detector: SmokeDetectorProtocol,
        smoke_like_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            smoke_like_frame
        )

        assert isinstance(
            result.bounding_boxes,
            tuple,
        )


# ==================================================
# False-positive scenarios
# ==================================================

class TestSmokeFalsePositives:

    def test_gray_scene_processing(
        self,
        detector: SmokeDetectorProtocol,
        gray_scene: np.ndarray,
    ) -> None:

        result = detector.detect(gray_scene)

        assert isinstance(
            result,
            SmokeDetectionResult,
        )

    def test_bright_object_processing(
        self,
        detector: SmokeDetectorProtocol,
        bright_non_smoke_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            bright_non_smoke_frame
        )

        # A bright object can resemble smoke
        # in simplistic image heuristics.
        # This is a robustness test, not a
        # validated false-positive benchmark.
        assert isinstance(
            result.smoke_detected,
            bool,
        )

    def test_low_light_processing(
        self,
        detector: SmokeDetectorProtocol,
    ) -> None:

        frame = np.full(
            (224, 224, 3),
            5,
            dtype=np.uint8,
        )

        result = detector.detect(frame)

        assert isinstance(
            result,
            SmokeDetectionResult,
        )

    def test_camera_noise_processing(
        self,
        detector: SmokeDetectorProtocol,
    ) -> None:

        rng = np.random.default_rng(123)

        frame = rng.integers(
            0,
            15,
            size=(224, 224, 3),
            dtype=np.uint8,
        )

        result = detector.detect(frame)

        assert isinstance(
            result,
            SmokeDetectionResult,
        )


# ==================================================
# Image format handling
# ==================================================

class TestSmokeImageFormats:

    def test_rgb_frame_supported(
        self,
        detector: SmokeDetectorProtocol,
        clean_frame: np.ndarray,
    ) -> None:

        result = detector.detect(clean_frame)

        assert isinstance(
            result,
            SmokeDetectionResult,
        )

    def test_grayscale_frame_handling(
        self,
        detector: SmokeDetectorProtocol,
    ) -> None:

        grayscale = np.zeros(
            (224, 224),
            dtype=np.uint8,
        )

        # Some detectors support grayscale
        # directly; others require RGB.
        try:
            result = detector.detect(grayscale)
        except (ValueError, TypeError):
            return

        assert isinstance(
            result,
            SmokeDetectionResult,
        )

    def test_different_frame_size(
        self,
        detector: SmokeDetectorProtocol,
    ) -> None:

        frame = np.zeros(
            (480, 640, 3),
            dtype=np.uint8,
        )

        result = detector.detect(frame)

        assert isinstance(
            result,
            SmokeDetectionResult,
        )

    def test_non_square_frame(
        self,
        detector: SmokeDetectorProtocol,
    ) -> None:

        frame = np.zeros(
            (240, 320, 3),
            dtype=np.uint8,
        )

        result = detector.detect(frame)

        assert isinstance(
            result,
            SmokeDetectionResult,
        )


# ==================================================
# Batch detection
# ==================================================

class TestSmokeBatchDetection:

    def test_batch_result_count(
        self,
        detector: SmokeDetectorProtocol,
        clean_frame: np.ndarray,
        smoke_like_frame: np.ndarray,
        dense_smoke_like_frame: np.ndarray,
    ) -> None:

        frames = [
            clean_frame,
            smoke_like_frame,
            dense_smoke_like_frame,
        ]

        results = detector.detect_batch(frames)

        assert len(results) == 3

    def test_batch_result_types(
        self,
        detector: SmokeDetectorProtocol,
        clean_frame: np.ndarray,
        smoke_like_frame: np.ndarray,
    ) -> None:

        results = detector.detect_batch([
            clean_frame,
            smoke_like_frame,
        ])

        assert all(
            isinstance(result, SmokeDetectionResult)
            for result in results
        )

    def test_empty_batch(
        self,
        detector: SmokeDetectorProtocol,
    ) -> None:

        assert detector.detect_batch([]) == []

    def test_batch_matches_individual(
        self,
        detector: SmokeDetectorProtocol,
        clean_frame: np.ndarray,
        smoke_like_frame: np.ndarray,
    ) -> None:

        frames = [
            clean_frame,
            smoke_like_frame,
        ]

        batch = detector.detect_batch(frames)

        individual = [
            detector.detect(frame)
            for frame in frames
        ]

        for batch_result, single_result in zip(
            batch,
            individual,
        ):
            assert (
                batch_result.smoke_detected
                == single_result.smoke_detected
            )

            assert batch_result.confidence == (
                pytest.approx(
                    single_result.confidence
                )
            )

    def test_large_batch(
        self,
        detector: SmokeDetectorProtocol,
        clean_frame: np.ndarray,
    ) -> None:

        frames = [
            clean_frame.copy()
            for _ in range(50)
        ]

        results = detector.detect_batch(frames)

        assert len(results) == 50


# ==================================================
# Input validation
# ==================================================

class TestSmokeInputValidation:

    def test_none_frame_rejected(
        self,
        detector: SmokeDetectorProtocol,
    ) -> None:

        with pytest.raises(
            (ValueError, TypeError)
        ):
            detector.detect(
                None  # type: ignore[arg-type]
            )

    def test_empty_frame_rejected(
        self,
        detector: SmokeDetectorProtocol,
    ) -> None:

        frame = np.array(
            [],
            dtype=np.uint8,
        )

        with pytest.raises(
            (ValueError, TypeError)
        ):
            detector.detect(frame)

    def test_invalid_frame_dimensions_rejected(
        self,
        detector: SmokeDetectorProtocol,
    ) -> None:

        frame = np.zeros(
            (10,),
            dtype=np.uint8,
        )

        with pytest.raises(
            (ValueError, TypeError)
        ):
            detector.detect(frame)

    def test_nan_frame_rejected(
        self,
        detector: SmokeDetectorProtocol,
    ) -> None:

        frame = np.full(
            (224, 224, 3),
            np.nan,
            dtype=np.float32,
        )

        with pytest.raises(
            (ValueError, TypeError)
        ):
            detector.detect(frame)

    def test_infinite_frame_rejected(
        self,
        detector: SmokeDetectorProtocol,
    ) -> None:

        frame = np.full(
            (224, 224, 3),
            np.inf,
            dtype=np.float32,
        )

        with pytest.raises(
            (ValueError, TypeError)
        ):
            detector.detect(frame)

    def test_invalid_input_type_rejected(
        self,
        detector: SmokeDetectorProtocol,
    ) -> None:

        with pytest.raises(
            (ValueError, TypeError)
        ):
            detector.detect(
                "invalid image"  # type: ignore[arg-type]
            )

    def test_zero_height_frame_rejected(
        self,
        detector: SmokeDetectorProtocol,
    ) -> None:

        frame = np.zeros(
            (0, 224, 3),
            dtype=np.uint8,
        )

        with pytest.raises(
            (ValueError, TypeError)
        ):
            detector.detect(frame)


# ==================================================
# Detection integrity
# ==================================================

class TestSmokeDetectionIntegrity:

    def test_input_frame_not_mutated(
        self,
        detector: SmokeDetectorProtocol,
        smoke_like_frame: np.ndarray,
    ) -> None:

        original = smoke_like_frame.copy()

        detector.detect(smoke_like_frame)

        assert np.array_equal(
            smoke_like_frame,
            original,
        )

    def test_deterministic_detection(
        self,
        detector: SmokeDetectorProtocol,
        smoke_like_frame: np.ndarray,
    ) -> None:

        results = [
            detector.detect(
                smoke_like_frame
            )
            for _ in range(5)
        ]

        first = results[0]

        for result in results[1:]:
            assert (
                result.smoke_detected
                == first.smoke_detected
            )

            assert result.confidence == (
                pytest.approx(
                    first.confidence
                )
            )

    def test_output_confidence_finite(
        self,
        detector: SmokeDetectorProtocol,
        smoke_like_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            smoke_like_frame
        )

        assert math.isfinite(
            result.confidence
        )

    def test_output_density_finite(
        self,
        detector: SmokeDetectorProtocol,
        smoke_like_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            smoke_like_frame
        )

        assert math.isfinite(
            result.smoke_density
        )

    def test_output_type(
        self,
        detector: SmokeDetectorProtocol,
        smoke_like_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            smoke_like_frame
        )

        assert isinstance(
            result,
            SmokeDetectionResult,
        )
