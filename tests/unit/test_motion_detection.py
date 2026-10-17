
"""
tests/unit/test_motion_detection.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

Motion Detection Engine Unit Tests

Run:
    pytest tests/unit/test_motion_detection.py -v

Coverage:
    - Motion detection
    - Stationary scenes
    - Moving objects
    - Motion confidence
    - Bounding boxes
    - Restricted-zone movement
    - Camera noise
    - Lighting changes
    - Batch processing
    - Frame validation
    - Frame immutability
    - Deterministic predictions

IMPORTANT:
Connect MotionDetectionAdapter to the actual
app/safety/motion_engine.py implementation.

These tests do not control industrial equipment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pytest


# ==================================================
# Result models
# ==================================================

@dataclass(frozen=True)
class MotionBox:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class MotionResult:
    motion_detected: bool
    confidence: float
    motion_boxes: tuple[MotionBox, ...] = ()
    motion_area: float = 0.0
    restricted_zone_violation: bool = False


# ==================================================
# Detector protocol
# ==================================================

class MotionDetectorProtocol(Protocol):

    def detect(
        self,
        previous_frame: np.ndarray,
        current_frame: np.ndarray,
        restricted_zone: tuple[int, int, int, int]
        | None = None,
    ) -> MotionResult:
        ...

    def detect_batch(
        self,
        frames: list[np.ndarray],
    ) -> list[MotionResult]:
        ...


# ==================================================
# Production adapter
# ==================================================

class MotionDetectionAdapter:
    """
    Connect to the production motion engine.

    Example:

        from app.safety.motion_engine import (
            MotionEngine
        )

        self.detector = MotionEngine(...)

    Update the import, constructor, and method
    mappings to match the actual implementation.

    Do not implement a second motion algorithm
    here when testing the production detector.
    """

    def __init__(self) -> None:
        self.detector: Any = None

        # TODO:
        # Initialize the actual MotionEngine.

    def detect(
        self,
        previous_frame: np.ndarray,
        current_frame: np.ndarray,
        restricted_zone: tuple[int, int, int, int]
        | None = None,
    ) -> MotionResult:

        if self.detector is None:
            raise RuntimeError(
                "Motion detector is not configured"
            )

        # Example:
        #
        # raw = self.detector.detect(
        #     previous_frame=previous_frame,
        #     current_frame=current_frame,
        #     restricted_zone=restricted_zone,
        # )
        #
        # boxes = tuple(
        #     MotionBox(**box)
        #     for box in raw.get("motion_boxes", [])
        # )
        #
        # return MotionResult(
        #     motion_detected=bool(
        #         raw["motion_detected"]
        #     ),
        #     confidence=float(raw["confidence"]),
        #     motion_boxes=boxes,
        #     motion_area=float(
        #         raw.get("motion_area", 0)
        #     ),
        #     restricted_zone_violation=bool(
        #         raw.get(
        #             "restricted_zone_violation",
        #             False,
        #         )
        #     ),
        # )

        raise NotImplementedError(
            "Map the production motion detection output"
        )

    def detect_batch(
        self,
        frames: list[np.ndarray],
    ) -> list[MotionResult]:

        if len(frames) < 2:
            return []

        return [
            self.detect(
                frames[index - 1],
                frames[index],
            )
            for index in range(1, len(frames))
        ]


# ==================================================
# Fixtures
# ==================================================

@pytest.fixture
def detector() -> MotionDetectorProtocol:

    adapter = MotionDetectionAdapter()

    if adapter.detector is None:
        pytest.skip(
            "Connect MotionDetectionAdapter to "
            "app/safety/motion_engine.py"
        )

    return adapter


@pytest.fixture
def static_frame() -> np.ndarray:
    return np.zeros(
        (224, 224, 3),
        dtype=np.uint8,
    )


@pytest.fixture
def moving_frame() -> np.ndarray:

    frame = np.zeros(
        (224, 224, 3),
        dtype=np.uint8,
    )

    frame[70:130, 70:130] = 255

    return frame


@pytest.fixture
def shifted_frame() -> np.ndarray:

    frame = np.zeros(
        (224, 224, 3),
        dtype=np.uint8,
    )

    frame[90:150, 110:170] = 255

    return frame


@pytest.fixture
def noisy_frame() -> np.ndarray:

    rng = np.random.default_rng(42)

    return rng.integers(
        0,
        5,
        size=(224, 224, 3),
        dtype=np.uint8,
    )


# ==================================================
# Basic motion detection
# ==================================================

class TestBasicMotionDetection:

    def test_stationary_scene(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            static_frame,
            static_frame.copy(),
        )

        assert result.motion_detected is False

    def test_moving_object(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
        moving_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            static_frame,
            moving_frame,
        )

        assert result.motion_detected is True

    def test_shifted_object(
        self,
        detector: MotionDetectorProtocol,
        moving_frame: np.ndarray,
        shifted_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            moving_frame,
            shifted_frame,
        )

        assert result.motion_detected is True

    def test_identical_frames_no_motion(
        self,
        detector: MotionDetectorProtocol,
        moving_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            moving_frame,
            moving_frame.copy(),
        )

        assert result.motion_detected is False

    def test_result_type(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
        moving_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            static_frame,
            moving_frame,
        )

        assert isinstance(
            result,
            MotionResult,
        )


# ==================================================
# Confidence
# ==================================================

class TestMotionConfidence:

    def test_confidence_range(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
        moving_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            static_frame,
            moving_frame,
        )

        assert 0.0 <= result.confidence <= 1.0

    def test_stationary_confidence_finite(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            static_frame,
            static_frame.copy(),
        )

        assert np.isfinite(result.confidence)

    def test_motion_confidence_finite(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
        moving_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            static_frame,
            moving_frame,
        )

        assert np.isfinite(result.confidence)


# ==================================================
# Motion bounding boxes
# ==================================================

class TestMotionBoundingBoxes:

    def test_boxes_have_valid_dimensions(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
        moving_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            static_frame,
            moving_frame,
        )

        for box in result.motion_boxes:
            assert box.width > 0
            assert box.height > 0
            assert box.x >= 0
            assert box.y >= 0

    def test_boxes_within_frame(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
        moving_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            static_frame,
            moving_frame,
        )

        height, width = moving_frame.shape[:2]

        for box in result.motion_boxes:
            assert box.x + box.width <= width
            assert box.y + box.height <= height

    def test_motion_area_nonnegative(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
        moving_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            static_frame,
            moving_frame,
        )

        assert np.isfinite(result.motion_area)
        assert result.motion_area >= 0


# ==================================================
# Restricted-zone monitoring
# ==================================================

class TestRestrictedZoneMotion:

    def test_motion_inside_restricted_zone(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
        moving_frame: np.ndarray,
    ) -> None:

        zone = (50, 50, 150, 150)

        result = detector.detect(
            static_frame,
            moving_frame,
            restricted_zone=zone,
        )

        assert result.restricted_zone_violation is True

    def test_motion_outside_restricted_zone(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
        moving_frame: np.ndarray,
    ) -> None:

        zone = (160, 160, 220, 220)

        result = detector.detect(
            static_frame,
            moving_frame,
            restricted_zone=zone,
        )

        assert result.restricted_zone_violation is False

    def test_stationary_restricted_zone(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
    ) -> None:

        zone = (50, 50, 150, 150)

        result = detector.detect(
            static_frame,
            static_frame.copy(),
            restricted_zone=zone,
        )

        assert result.restricted_zone_violation is False


# ==================================================
# Camera noise and lighting
# ==================================================

class TestMotionNoiseHandling:

    def test_low_level_camera_noise(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
        noisy_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            static_frame,
            noisy_frame,
        )

        assert isinstance(
            result.motion_detected,
            bool,
        )

    def test_brightness_change(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
    ) -> None:

        bright = np.full_like(
            static_frame,
            100,
        )

        result = detector.detect(
            static_frame,
            bright,
        )

        # A global lighting change can resemble
        # motion to a basic frame-difference model.
        # This test validates output, not accuracy.
        assert isinstance(
            result,
            MotionResult,
        )

    def test_small_motion(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
    ) -> None:

        frame = static_frame.copy()

        frame[100:105, 100:105] = 255

        result = detector.detect(
            static_frame,
            frame,
        )

        assert isinstance(
            result.motion_detected,
            bool,
        )


# ==================================================
# Batch processing
# ==================================================

class TestBatchMotionDetection:

    def test_batch_result_count(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
        moving_frame: np.ndarray,
        shifted_frame: np.ndarray,
    ) -> None:

        frames = [
            static_frame,
            moving_frame,
            shifted_frame,
        ]

        results = detector.detect_batch(frames)

        assert len(results) == 2

    def test_batch_result_types(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
        moving_frame: np.ndarray,
    ) -> None:

        results = detector.detect_batch([
            static_frame,
            moving_frame,
        ])

        assert all(
            isinstance(result, MotionResult)
            for result in results
        )

    def test_empty_batch(
        self,
        detector: MotionDetectorProtocol,
    ) -> None:

        assert detector.detect_batch([]) == []

    def test_single_frame_batch(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
    ) -> None:

        assert detector.detect_batch([
            static_frame
        ]) == []

    def test_batch_matches_individual(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
        moving_frame: np.ndarray,
        shifted_frame: np.ndarray,
    ) -> None:

        frames = [
            static_frame,
            moving_frame,
            shifted_frame,
        ]

        batch = detector.detect_batch(frames)

        individual = [
            detector.detect(
                frames[i - 1],
                frames[i],
            )
            for i in range(1, len(frames))
        ]

        for batch_result, single_result in zip(
            batch,
            individual,
        ):
            assert (
                batch_result.motion_detected
                == single_result.motion_detected
            )


# ==================================================
# Invalid inputs
# ==================================================

class TestMotionInputValidation:

    def test_none_previous_frame_rejected(
        self,
        detector: MotionDetectorProtocol,
        moving_frame: np.ndarray,
    ) -> None:

        with pytest.raises(
            (ValueError, TypeError)
        ):
            detector.detect(
                None,  # type: ignore[arg-type]
                moving_frame,
            )

    def test_none_current_frame_rejected(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
    ) -> None:

        with pytest.raises(
            (ValueError, TypeError)
        ):
            detector.detect(
                static_frame,
                None,  # type: ignore[arg-type]
            )

    def test_empty_frame_rejected(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
    ) -> None:

        empty = np.array(
            [],
            dtype=np.uint8,
        )

        with pytest.raises(
            (ValueError, TypeError)
        ):
            detector.detect(
                static_frame,
                empty,
            )

    def test_mismatched_frame_sizes_rejected(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
    ) -> None:

        smaller = np.zeros(
            (100, 100, 3),
            dtype=np.uint8,
        )

        with pytest.raises(
            (ValueError, TypeError)
        ):
            detector.detect(
                static_frame,
                smaller,
            )

    def test_invalid_frame_shape_rejected(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
    ) -> None:

        invalid = np.zeros(
            (10,),
            dtype=np.uint8,
        )

        with pytest.raises(
            (ValueError, TypeError)
        ):
            detector.detect(
                static_frame,
                invalid,
            )

    def test_nan_frame_rejected(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
    ) -> None:

        invalid = np.full(
            (224, 224, 3),
            np.nan,
            dtype=np.float32,
        )

        with pytest.raises(
            (ValueError, TypeError)
        ):
            detector.detect(
                static_frame,
                invalid,
            )

    def test_invalid_restricted_zone_rejected(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
        moving_frame: np.ndarray,
    ) -> None:

        invalid_zone = (150, 150, 50, 50)

        with pytest.raises(
            (ValueError, TypeError)
        ):
            detector.detect(
                static_frame,
                moving_frame,
                restricted_zone=invalid_zone,
            )


# ==================================================
# Data integrity
# ==================================================

class TestMotionDetectionIntegrity:

    def test_frames_not_mutated(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
        moving_frame: np.ndarray,
    ) -> None:

        previous_copy = static_frame.copy()
        current_copy = moving_frame.copy()

        detector.detect(
            static_frame,
            moving_frame,
        )

        assert np.array_equal(
            static_frame,
            previous_copy,
        )

        assert np.array_equal(
            moving_frame,
            current_copy,
        )

    def test_deterministic_motion_detection(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
        moving_frame: np.ndarray,
    ) -> None:

        results = [
            detector.detect(
                static_frame,
                moving_frame,
            )
            for _ in range(5)
        ]

        first = results[0]

        for result in results[1:]:
            assert (
                result.motion_detected
                == first.motion_detected
            )

            assert result.confidence == pytest.approx(
                first.confidence
            )

    def test_motion_confidence_is_finite(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
        moving_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            static_frame,
            moving_frame,
        )

        assert np.isfinite(
            result.confidence
        )

    def test_motion_area_is_finite(
        self,
        detector: MotionDetectorProtocol,
        static_frame: np.ndarray,
        moving_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            static_frame,
            moving_frame,
        )

        assert np.isfinite(
            result.motion_area
        )
