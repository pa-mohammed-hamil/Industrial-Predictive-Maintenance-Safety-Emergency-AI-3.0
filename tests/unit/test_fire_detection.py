
"""
tests/unit/test_fire_detection.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

Fire Detection Engine Unit Tests

Run:
    pytest tests/unit/test_fire_detection.py -v

Coverage:
    - Fire and non-fire detection
    - Flame confidence
    - Smoke confidence
    - Thermal fire indicators
    - Fire severity
    - Multiple camera frames
    - False-positive scenarios
    - Invalid image inputs
    - Corrupt frames
    - Batch processing
    - Deterministic inference
    - Input immutability
    - Alert recommendation
    - Safety-critical behavior

IMPORTANT:
Connect FireDetectorAdapter to the production
fire detection implementation.

The synthetic test images do not establish
real-world model accuracy. Labeled image/video
datasets are required for that purpose.

These tests must never activate real alarms,
sprinklers, relays, PLCs, or emergency dispatch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pytest


# ==================================================
# Result model
# ==================================================

@dataclass(frozen=True)
class FireDetectionResult:
    fire_detected: bool
    fire_confidence: float
    smoke_detected: bool = False
    smoke_confidence: float = 0.0
    severity: str = "none"
    alert_required: bool = False


# ==================================================
# Detector interface
# ==================================================

class FireDetectorProtocol(Protocol):

    def detect(
        self,
        frame: np.ndarray,
        temperature: float | None = None,
    ) -> FireDetectionResult:
        ...

    def detect_batch(
        self,
        frames: list[np.ndarray],
    ) -> list[FireDetectionResult]:
        ...


# ==================================================
# Production adapter
# ==================================================

class FireDetectorAdapter:
    """
    Connect to app/safety/fire_engine.py.

    Example:

        from app.safety.fire_engine import FireEngine

        self.detector = FireEngine(...)

    Replace the example with the actual class,
    constructor, inference method, and result
    schema used by your project.

    Do not substitute a mock detector for
    production inference tests.
    """

    def __init__(self) -> None:
        self.detector: Any = None

        # TODO:
        # self.detector = FireEngine(...)

    def detect(
        self,
        frame: np.ndarray,
        temperature: float | None = None,
    ) -> FireDetectionResult:

        if self.detector is None:
            raise RuntimeError(
                "Production fire detector not configured"
            )

        # Example integration:
        #
        # raw = self.detector.detect(
        #     image=frame,
        #     temperature=temperature,
        # )
        #
        # return FireDetectionResult(
        #     fire_detected=bool(raw["fire_detected"]),
        #     fire_confidence=float(
        #         raw["fire_confidence"]
        #     ),
        #     smoke_detected=bool(
        #         raw.get("smoke_detected", False)
        #     ),
        #     smoke_confidence=float(
        #         raw.get("smoke_confidence", 0.0)
        #     ),
        #     severity=raw.get("severity", "none"),
        #     alert_required=bool(
        #         raw.get("alert_required", False)
        #     ),
        # )

        raise NotImplementedError(
            "Map production fire detection output"
        )

    def detect_batch(
        self,
        frames: list[np.ndarray],
    ) -> list[FireDetectionResult]:

        return [
            self.detect(frame)
            for frame in frames
        ]


# ==================================================
# Fixtures
# ==================================================

@pytest.fixture
def detector() -> FireDetectorProtocol:

    adapter = FireDetectorAdapter()

    if adapter.detector is None:
        pytest.skip(
            "Connect FireDetectorAdapter to "
            "app/safety/fire_engine.py"
        )

    return adapter


@pytest.fixture
def normal_frame() -> np.ndarray:
    """
    Synthetic neutral frame.
    Not a substitute for a labeled non-fire image.
    """

    return np.full(
        (224, 224, 3),
        100,
        dtype=np.uint8,
    )


@pytest.fixture
def flame_like_frame() -> np.ndarray:
    """
    Synthetic orange/yellow pattern.

    This is not guaranteed to represent a real
    fire image to a trained vision model.
    """

    frame = np.zeros(
        (224, 224, 3),
        dtype=np.uint8,
    )

    # OpenCV-style BGR color layout.
    frame[50:180, 70:160] = [0, 140, 255]
    frame[80:160, 90:140] = [0, 255, 255]

    return frame


@pytest.fixture
def smoke_like_frame() -> np.ndarray:

    frame = np.zeros(
        (224, 224, 3),
        dtype=np.uint8,
    )

    frame[40:180, 50:180] = [160, 160, 160]

    return frame


# ==================================================
# Output validation
# ==================================================

class TestFireDetectionOutput:

    def test_result_type(
        self,
        detector: FireDetectorProtocol,
        normal_frame: np.ndarray,
    ) -> None:

        result = detector.detect(normal_frame)

        assert isinstance(
            result,
            FireDetectionResult,
        )

    def test_fire_detection_is_boolean(
        self,
        detector: FireDetectorProtocol,
        normal_frame: np.ndarray,
    ) -> None:

        result = detector.detect(normal_frame)

        assert isinstance(
            result.fire_detected,
            bool,
        )

    def test_confidence_range(
        self,
        detector: FireDetectorProtocol,
        normal_frame: np.ndarray,
    ) -> None:

        result = detector.detect(normal_frame)

        assert math.isfinite(
            result.fire_confidence
        )

        assert (
            0.0
            <= result.fire_confidence
            <= 1.0
        )

    def test_smoke_confidence_range(
        self,
        detector: FireDetectorProtocol,
        smoke_like_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            smoke_like_frame
        )

        assert math.isfinite(
            result.smoke_confidence
        )

        assert (
            0.0
            <= result.smoke_confidence
            <= 1.0
        )

    def test_severity_valid(
        self,
        detector: FireDetectorProtocol,
        normal_frame: np.ndarray,
    ) -> None:

        result = detector.detect(normal_frame)

        assert result.severity in {
            "none",
            "low",
            "medium",
            "high",
            "critical",
        }


# ==================================================
# Fire detection scenarios
# ==================================================

class TestFireScenarios:

    def test_normal_frame_processing(
        self,
        detector: FireDetectorProtocol,
        normal_frame: np.ndarray,
    ) -> None:

        result = detector.detect(normal_frame)

        assert isinstance(
            result.fire_detected,
            bool,
        )

    def test_flame_like_frame_processing(
        self,
        detector: FireDetectorProtocol,
        flame_like_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            flame_like_frame
        )

        assert isinstance(
            result,
            FireDetectionResult,
        )

        assert 0 <= result.fire_confidence <= 1

    def test_smoke_like_frame_processing(
        self,
        detector: FireDetectorProtocol,
        smoke_like_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            smoke_like_frame
        )

        assert isinstance(
            result.smoke_detected,
            bool,
        )

    def test_bright_yellow_false_positive_candidate(
        self,
        detector: FireDetectorProtocol,
    ) -> None:

        frame = np.zeros(
            (224, 224, 3),
            dtype=np.uint8,
        )

        frame[:] = [0, 255, 255]

        result = detector.detect(frame)

        # A yellow image alone does not establish
        # that actual fire is present.
        assert isinstance(
            result.fire_detected,
            bool,
        )

    def test_red_object_false_positive_candidate(
        self,
        detector: FireDetectorProtocol,
    ) -> None:

        frame = np.zeros(
            (224, 224, 3),
            dtype=np.uint8,
        )

        frame[50:170, 50:170] = [0, 0, 255]

        result = detector.detect(frame)

        assert isinstance(
            result,
            FireDetectionResult,
        )


# ==================================================
# Thermal indicators
# ==================================================

class TestThermalFireDetection:

    def test_normal_temperature(
        self,
        detector: FireDetectorProtocol,
        normal_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            normal_frame,
            temperature=35.0,
        )

        assert isinstance(
            result,
            FireDetectionResult,
        )

    def test_high_temperature(
        self,
        detector: FireDetectorProtocol,
        normal_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            normal_frame,
            temperature=150.0,
        )

        assert isinstance(
            result,
            FireDetectionResult,
        )

    def test_extreme_temperature(
        self,
        detector: FireDetectorProtocol,
        normal_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            normal_frame,
            temperature=400.0,
        )

        assert 0 <= result.fire_confidence <= 1

    @pytest.mark.parametrize(
        "invalid_temperature",
        [
            float("nan"),
            float("inf"),
            float("-inf"),
        ],
    )
    def test_invalid_temperature_rejected(
        self,
        detector: FireDetectorProtocol,
        normal_frame: np.ndarray,
        invalid_temperature: float,
    ) -> None:

        with pytest.raises(
            (ValueError, TypeError)
        ):
            detector.detect(
                normal_frame,
                temperature=invalid_temperature,
            )


# ==================================================
# Invalid frame handling
# ==================================================

class TestInvalidFrames:

    def test_none_frame_rejected(
        self,
        detector: FireDetectorProtocol,
    ) -> None:

        with pytest.raises(
            (ValueError, TypeError)
        ):
            detector.detect(None)  # type: ignore

    def test_empty_frame_rejected(
        self,
        detector: FireDetectorProtocol,
    ) -> None:

        frame = np.array(
            [],
            dtype=np.uint8,
        )

        with pytest.raises(
            (ValueError, TypeError)
        ):
            detector.detect(frame)

    def test_invalid_frame_shape_rejected(
        self,
        detector: FireDetectorProtocol,
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
        detector: FireDetectorProtocol,
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
        detector: FireDetectorProtocol,
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


# ==================================================
# Batch detection
# ==================================================

class TestBatchFireDetection:

    def test_batch_result_count(
        self,
        detector: FireDetectorProtocol,
        normal_frame: np.ndarray,
        flame_like_frame: np.ndarray,
        smoke_like_frame: np.ndarray,
    ) -> None:

        frames = [
            normal_frame,
            flame_like_frame,
            smoke_like_frame,
        ]

        results = detector.detect_batch(frames)

        assert len(results) == 3

    def test_batch_result_types(
        self,
        detector: FireDetectorProtocol,
        normal_frame: np.ndarray,
    ) -> None:

        frames = [
            normal_frame.copy()
            for _ in range(5)
        ]

        results = detector.detect_batch(frames)

        assert all(
            isinstance(result, FireDetectionResult)
            for result in results
        )

    def test_empty_batch(
        self,
        detector: FireDetectorProtocol,
    ) -> None:

        results = detector.detect_batch([])

        assert results == []

    def test_batch_matches_individual(
        self,
        detector: FireDetectorProtocol,
        normal_frame: np.ndarray,
        flame_like_frame: np.ndarray,
    ) -> None:

        frames = [
            normal_frame,
            flame_like_frame,
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
                batch_result.fire_detected
                == single_result.fire_detected
            )

            assert batch_result.fire_confidence == (
                pytest.approx(
                    single_result.fire_confidence
                )
            )


# ==================================================
# Data integrity
# ==================================================

class TestFireDetectionIntegrity:

    def test_frame_not_mutated(
        self,
        detector: FireDetectorProtocol,
        normal_frame: np.ndarray,
    ) -> None:

        original = normal_frame.copy()

        detector.detect(normal_frame)

        assert np.array_equal(
            normal_frame,
            original,
        )

    def test_deterministic_inference(
        self,
        detector: FireDetectorProtocol,
        normal_frame: np.ndarray,
    ) -> None:

        results = [
            detector.detect(normal_frame)
            for _ in range(5)
        ]

        first = results[0]

        for result in results[1:]:
            assert (
                result.fire_detected
                == first.fire_detected
            )

            assert result.fire_confidence == (
                pytest.approx(
                    first.fire_confidence
                )
            )


# ==================================================
# Safety-critical output behavior
# ==================================================

class TestFireSafetyBehavior:

    def test_alert_field_is_boolean(
        self,
        detector: FireDetectorProtocol,
        normal_frame: np.ndarray,
    ) -> None:

        result = detector.detect(normal_frame)

        assert isinstance(
            result.alert_required,
            bool,
        )

    def test_severity_consistency(
        self,
        detector: FireDetectorProtocol,
        normal_frame: np.ndarray,
    ) -> None:

        result = detector.detect(normal_frame)

        if result.severity == "critical":
            assert result.alert_required is True

    def test_critical_fire_requires_alert(
        self,
        detector: FireDetectorProtocol,
        flame_like_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            flame_like_frame,
            temperature=400.0,
        )

        if result.severity == "critical":
            assert result.alert_required is True

    def test_fire_confidence_is_finite(
        self,
        detector: FireDetectorProtocol,
        flame_like_frame: np.ndarray,
    ) -> None:

        result = detector.detect(
            flame_like_frame
        )

        assert math.isfinite(
            result.fire_confidence
        )

    def test_no_external_actuation_in_unit_tests(
        self,
        detector: FireDetectorProtocol,
        normal_frame: np.ndarray,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        This test only checks the prediction result.

        Real integrations for alarms, sprinklers,
        PLCs, and dispatch must be separately
        mocked or disabled by the test environment.
        """

        result = detector.detect(normal_frame)

        assert isinstance(
            result,
            FireDetectionResult,
        )
