
"""
tests/unit/test_anomaly.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

Anomaly Detection Unit Tests

Coverage:
    1. Normal sensor readings
    2. High temperature
    3. High vibration
    4. High pressure
    5. High electrical current
    6. Combined anomalies
    7. Anomaly score validity
    8. Input validation
    9. Missing sensor values
   10. NaN and infinity
   11. Batch predictions
   12. Deterministic predictions
   13. Input immutability
   14. Boundary conditions
   15. Model consistency
   16. Safety-critical anomaly behavior

Run:
    pytest tests/unit/test_anomaly.py -v

IMPORTANT:
The actual anomaly detection implementation and
API are not available in this conversation.

Connect the adapter below to your implementation.
The example threshold values are test-policy
assumptions, not industrial safety standards.

Never use these thresholds to control live
industrial equipment.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

import pytest


# ==================================================
# Test data
# ==================================================

@dataclass(frozen=True)
class SensorReading:
    temperature: float
    vibration: float
    pressure: float
    current: float


NORMAL = SensorReading(
    temperature=45.0,
    vibration=2.5,
    pressure=4.2,
    current=12.0,
)

HIGH_TEMPERATURE = SensorReading(
    temperature=150.0,
    vibration=2.5,
    pressure=4.2,
    current=12.0,
)

HIGH_VIBRATION = SensorReading(
    temperature=45.0,
    vibration=25.0,
    pressure=4.2,
    current=12.0,
)

HIGH_PRESSURE = SensorReading(
    temperature=45.0,
    vibration=2.5,
    pressure=25.0,
    current=12.0,
)

HIGH_CURRENT = SensorReading(
    temperature=45.0,
    vibration=2.5,
    pressure=4.2,
    current=50.0,
)

COMBINED_ANOMALY = SensorReading(
    temperature=160.0,
    vibration=30.0,
    pressure=28.0,
    current=55.0,
)


# ==================================================
# Normalized output
# ==================================================

@dataclass(frozen=True)
class AnomalyResult:
    is_anomaly: bool
    anomaly_score: float
    severity: str | None = None


# ==================================================
# Detector interface
# ==================================================

class AnomalyDetectorProtocol(Protocol):

    def predict(
        self,
        reading: SensorReading,
    ) -> AnomalyResult:
        ...

    def predict_batch(
        self,
        readings: list[SensorReading],
    ) -> list[AnomalyResult]:
        ...


# ==================================================
# Adapter configuration
# ==================================================

class AnomalyDetectorAdapter:
    """
    Connect this adapter to the actual application.

    Example:

        from app.ml.anomaly_detection.detector import (
            AnomalyDetector
        )

        self.detector = AnomalyDetector()

    Then implement predict() using the actual
    method name and output schema.

    Do not replace this adapter with a fake detector
    when the goal is to verify production behavior.
    """

    def __init__(self) -> None:
        self.detector: Any = None

        # TODO: Initialize actual detector here.

    def predict(
        self,
        reading: SensorReading,
    ) -> AnomalyResult:

        if self.detector is None:
            raise NotImplementedError(
                "Connect AnomalyDetectorAdapter "
                "to the actual anomaly detector."
            )

        # Example mapping only:
        #
        # result = self.detector.predict({
        #     "temperature": reading.temperature,
        #     "vibration": reading.vibration,
        #     "pressure": reading.pressure,
        #     "current": reading.current,
        # })
        #
        # return AnomalyResult(
        #     is_anomaly=bool(result["is_anomaly"]),
        #     anomaly_score=float(
        #         result["anomaly_score"]
        #     ),
        #     severity=result.get("severity"),
        # )

        raise NotImplementedError(
            "Implement the production prediction mapping."
        )

    def predict_batch(
        self,
        readings: list[SensorReading],
    ) -> list[AnomalyResult]:

        return [
            self.predict(reading)
            for reading in readings
        ]


# ==================================================
# Fixtures
# ==================================================

@pytest.fixture
def detector() -> AnomalyDetectorProtocol:
    """
    Replace this fixture with the configured
    production anomaly detector adapter.
    """

    adapter = AnomalyDetectorAdapter()

    if adapter.detector is None:
        pytest.skip(
            "Production anomaly detector adapter "
            "has not been configured."
        )

    return adapter


@pytest.fixture
def normal_reading() -> SensorReading:
    return NORMAL


@pytest.fixture
def abnormal_readings() -> list[SensorReading]:
    return [
        HIGH_TEMPERATURE,
        HIGH_VIBRATION,
        HIGH_PRESSURE,
        HIGH_CURRENT,
        COMBINED_ANOMALY,
    ]


# ==================================================
# Normal telemetry tests
# ==================================================

class TestNormalTelemetry:

    def test_normal_reading_not_anomaly(
        self,
        detector: AnomalyDetectorProtocol,
        normal_reading: SensorReading,
    ) -> None:

        result = detector.predict(normal_reading)

        assert isinstance(result, AnomalyResult)
        assert result.is_anomaly is False

    def test_normal_anomaly_score_range(
        self,
        detector: AnomalyDetectorProtocol,
        normal_reading: SensorReading,
    ) -> None:

        result = detector.predict(normal_reading)

        assert math.isfinite(result.anomaly_score)
        assert 0.0 <= result.anomaly_score <= 1.0

    def test_normal_reading_deterministic(
        self,
        detector: AnomalyDetectorProtocol,
        normal_reading: SensorReading,
    ) -> None:

        first = detector.predict(normal_reading)
        second = detector.predict(normal_reading)

        assert first.is_anomaly == second.is_anomaly

        assert first.anomaly_score == pytest.approx(
            second.anomaly_score
        )


# ==================================================
# Abnormal sensor tests
# ==================================================

class TestAbnormalTelemetry:

    @pytest.mark.parametrize(
        "reading",
        [
            HIGH_TEMPERATURE,
            HIGH_VIBRATION,
            HIGH_PRESSURE,
            HIGH_CURRENT,
            COMBINED_ANOMALY,
        ],
        ids=[
            "high-temperature",
            "high-vibration",
            "high-pressure",
            "high-current",
            "combined-anomaly",
        ],
    )
    def test_abnormal_sensor_detection(
        self,
        detector: AnomalyDetectorProtocol,
        reading: SensorReading,
    ) -> None:

        result = detector.predict(reading)

        assert result.is_anomaly is True

    def test_combined_anomaly_detected(
        self,
        detector: AnomalyDetectorProtocol,
    ) -> None:

        result = detector.predict(
            COMBINED_ANOMALY
        )

        assert result.is_anomaly is True

    def test_abnormal_score_is_finite(
        self,
        detector: AnomalyDetectorProtocol,
        abnormal_readings: list[SensorReading],
    ) -> None:

        for reading in abnormal_readings:
            result = detector.predict(reading)

            assert math.isfinite(
                result.anomaly_score
            )

    def test_abnormal_score_range(
        self,
        detector: AnomalyDetectorProtocol,
        abnormal_readings: list[SensorReading],
    ) -> None:

        for reading in abnormal_readings:
            result = detector.predict(reading)

            assert (
                0.0
                <= result.anomaly_score
                <= 1.0
            )


# ==================================================
# Anomaly score tests
# ==================================================

class TestAnomalyScores:

    def test_combined_anomaly_score(
        self,
        detector: AnomalyDetectorProtocol,
    ) -> None:

        normal = detector.predict(NORMAL)

        abnormal = detector.predict(
            COMBINED_ANOMALY
        )

        # Assumes larger score means more anomalous.
        assert (
            abnormal.anomaly_score
            >= normal.anomaly_score
        )

    def test_severity_is_valid(
        self,
        detector: AnomalyDetectorProtocol,
    ) -> None:

        result = detector.predict(
            HIGH_TEMPERATURE
        )

        if result.severity is not None:
            assert result.severity in {
                "normal",
                "low",
                "medium",
                "high",
                "critical",
            }


# ==================================================
# Batch prediction tests
# ==================================================

class TestBatchPrediction:

    def test_batch_prediction_count(
        self,
        detector: AnomalyDetectorProtocol,
    ) -> None:

        readings = [
            NORMAL,
            HIGH_TEMPERATURE,
            HIGH_VIBRATION,
            HIGH_PRESSURE,
            HIGH_CURRENT,
        ]

        results = detector.predict_batch(readings)

        assert len(results) == len(readings)

    def test_batch_prediction_types(
        self,
        detector: AnomalyDetectorProtocol,
    ) -> None:

        readings = [
            NORMAL,
            HIGH_TEMPERATURE,
            HIGH_VIBRATION,
        ]

        results = detector.predict_batch(readings)

        assert all(
            isinstance(result, AnomalyResult)
            for result in results
        )

    def test_batch_matches_single_prediction(
        self,
        detector: AnomalyDetectorProtocol,
    ) -> None:

        readings = [
            NORMAL,
            HIGH_TEMPERATURE,
            HIGH_VIBRATION,
        ]

        batch = detector.predict_batch(readings)

        individual = [
            detector.predict(reading)
            for reading in readings
        ]

        for batch_result, single_result in zip(
            batch,
            individual,
        ):
            assert (
                batch_result.is_anomaly
                == single_result.is_anomaly
            )

            assert (
                batch_result.anomaly_score
                == pytest.approx(
                    single_result.anomaly_score
                )
            )

    def test_empty_batch(
        self,
        detector: AnomalyDetectorProtocol,
    ) -> None:

        results = detector.predict_batch([])

        assert results == []

    def test_large_batch(
        self,
        detector: AnomalyDetectorProtocol,
    ) -> None:

        readings = [NORMAL] * 100

        results = detector.predict_batch(readings)

        assert len(results) == 100


# ==================================================
# Input validation tests
# ==================================================

class TestInputValidation:

    @pytest.mark.parametrize(
        "invalid_value",
        [
            float("nan"),
            float("inf"),
            float("-inf"),
        ],
    )
    def test_invalid_temperature_rejected(
        self,
        detector: AnomalyDetectorProtocol,
        invalid_value: float,
    ) -> None:

        reading = SensorReading(
            temperature=invalid_value,
            vibration=2.5,
            pressure=4.2,
            current=12.0,
        )

        with pytest.raises(
            (ValueError, TypeError)
        ):
            detector.predict(reading)

    @pytest.mark.parametrize(
        "invalid_value",
        [
            float("nan"),
            float("inf"),
            float("-inf"),
        ],
    )
    def test_invalid_vibration_rejected(
        self,
        detector: AnomalyDetectorProtocol,
        invalid_value: float,
    ) -> None:

        reading = SensorReading(
            temperature=45.0,
            vibration=invalid_value,
            pressure=4.2,
            current=12.0,
        )

        with pytest.raises(
            (ValueError, TypeError)
        ):
            detector.predict(reading)

    def test_negative_vibration_rejected(
        self,
        detector: AnomalyDetectorProtocol,
    ) -> None:

        reading = SensorReading(
            temperature=45.0,
            vibration=-10.0,
            pressure=4.2,
            current=12.0,
        )

        with pytest.raises(ValueError):
            detector.predict(reading)

    def test_missing_sensor_value_rejected(
        self,
        detector: AnomalyDetectorProtocol,
    ) -> None:

        invalid = {
            "temperature": 45.0,
            "vibration": 2.5,
            "pressure": 4.2,
        }

        with pytest.raises(
            (TypeError, ValueError, KeyError)
        ):
            detector.predict(invalid)  # type: ignore


# ==================================================
# Boundary tests
# ==================================================

class TestBoundaryConditions:

    def test_zero_vibration(
        self,
        detector: AnomalyDetectorProtocol,
    ) -> None:

        reading = SensorReading(
            temperature=45.0,
            vibration=0.0,
            pressure=4.2,
            current=12.0,
        )

        result = detector.predict(reading)

        assert isinstance(result, AnomalyResult)

    def test_zero_current(
        self,
        detector: AnomalyDetectorProtocol,
    ) -> None:

        reading = SensorReading(
            temperature=25.0,
            vibration=0.0,
            pressure=0.0,
            current=0.0,
        )

        result = detector.predict(reading)

        assert isinstance(result, AnomalyResult)

    def test_low_temperature(
        self,
        detector: AnomalyDetectorProtocol,
    ) -> None:

        reading = SensorReading(
            temperature=-10.0,
            vibration=2.0,
            pressure=4.0,
            current=10.0,
        )

        result = detector.predict(reading)

        assert isinstance(result, AnomalyResult)

    def test_extreme_temperature(
        self,
        detector: AnomalyDetectorProtocol,
    ) -> None:

        reading = SensorReading(
            temperature=1000.0,
            vibration=2.5,
            pressure=4.2,
            current=12.0,
        )

        try:
            result = detector.predict(reading)

        except ValueError:
            # Acceptable if the detector rejects
            # out-of-domain sensor readings.
            return

        assert result.is_anomaly is True


# ==================================================
# Data integrity tests
# ==================================================

class TestDataIntegrity:

    def test_input_not_mutated(
        self,
        detector: AnomalyDetectorProtocol,
    ) -> None:

        reading = NORMAL
        original = deepcopy(reading)

        detector.predict(reading)

        assert reading == original

    def test_prediction_repeatability(
        self,
        detector: AnomalyDetectorProtocol,
    ) -> None:

        results = [
            detector.predict(NORMAL)
            for _ in range(5)
        ]

        first = results[0]

        for result in results[1:]:
            assert (
                result.is_anomaly
                == first.is_anomaly
            )

            assert (
                result.anomaly_score
                == pytest.approx(
                    first.anomaly_score
                )
            )


# ==================================================
# Industrial safety behavior
# ==================================================

class TestSafetyCriticalAnomalies:

    def test_extreme_combined_reading(
        self,
        detector: AnomalyDetectorProtocol,
    ) -> None:

        result = detector.predict(
            COMBINED_ANOMALY
        )

        assert result.is_anomaly is True

    def test_critical_anomaly_has_valid_score(
        self,
        detector: AnomalyDetectorProtocol,
    ) -> None:

        result = detector.predict(
            COMBINED_ANOMALY
        )

        assert math.isfinite(
            result.anomaly_score
        )

        assert 0 <= result.anomaly_score <= 1

    def test_no_false_negative_for_test_scenario(
        self,
        detector: AnomalyDetectorProtocol,
    ) -> None:

        critical_readings = [
            HIGH_TEMPERATURE,
            HIGH_VIBRATION,
            HIGH_PRESSURE,
            HIGH_CURRENT,
            COMBINED_ANOMALY,
        ]

        for reading in critical_readings:
            result = detector.predict(reading)

            assert result.is_anomaly, (
                "Expected anomaly for synthetic "
                f"test scenario: {reading}"
            )
