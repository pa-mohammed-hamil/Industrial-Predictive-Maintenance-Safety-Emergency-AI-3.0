
"""
tests/unit/test_failure_prediction.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

Machine Failure Prediction Unit Tests

Run:
    pytest tests/unit/test_failure_prediction.py -v

Coverage:
    - Healthy machine prediction
    - High-risk machine prediction
    - Temperature, vibration, pressure, current
    - Failure probability range
    - Prediction labels
    - Model confidence
    - Batch prediction
    - Feature validation
    - Missing and non-finite values
    - Determinism
    - Input immutability
    - Extreme telemetry
    - Probability calibration checks

IMPORTANT:
This is a proposed test contract. Connect the
adapter to your actual trained failure prediction
model before executing model-dependent tests.

The synthetic readings and expected classifications
are illustrative and must be validated against the
model's training distribution and engineering limits.

No physical equipment or safety actuator is used.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

import pytest


# ==================================================
# Domain models
# ==================================================

@dataclass(frozen=True)
class MachineTelemetry:
    temperature: float
    vibration: float
    pressure: float
    current: float
    operating_hours: float


@dataclass(frozen=True)
class FailurePrediction:
    failure_probability: float
    predicted_failure: bool
    confidence: float | None = None
    risk_level: str | None = None


HEALTHY_MACHINE = MachineTelemetry(
    temperature=45.0,
    vibration=2.0,
    pressure=4.0,
    current=12.0,
    operating_hours=1000.0,
)

HIGH_RISK_MACHINE = MachineTelemetry(
    temperature=140.0,
    vibration=25.0,
    pressure=20.0,
    current=45.0,
    operating_hours=18000.0,
)

HIGH_TEMPERATURE = MachineTelemetry(
    temperature=145.0,
    vibration=2.0,
    pressure=4.0,
    current=12.0,
    operating_hours=1000.0,
)

HIGH_VIBRATION = MachineTelemetry(
    temperature=45.0,
    vibration=30.0,
    pressure=4.0,
    current=12.0,
    operating_hours=1000.0,
)

HIGH_CURRENT = MachineTelemetry(
    temperature=45.0,
    vibration=2.0,
    pressure=4.0,
    current=50.0,
    operating_hours=1000.0,
)


# ==================================================
# Model protocol
# ==================================================

class FailurePredictorProtocol(Protocol):

    def predict(
        self,
        telemetry: MachineTelemetry,
    ) -> FailurePrediction:
        ...

    def predict_batch(
        self,
        readings: list[MachineTelemetry],
    ) -> list[FailurePrediction]:
        ...


# ==================================================
# Production adapter
# ==================================================

class FailurePredictionAdapter:
    """
    Connect this adapter to the actual model.

    Example import path (adjust as necessary):

        from app.ml.predictive_maintenance.\
failure_predictor import FailurePredictor

        self.model = FailurePredictor(...)

    Load a test model artifact and map the real
    prediction response to FailurePrediction.

    Do not reimplement prediction logic here:
    that would test the adapter rather than
    the actual production model.
    """

    def __init__(self) -> None:
        self.model: Any = None

        # TODO:
        # Initialize and load the actual trained
        # failure prediction model here.

    def predict(
        self,
        telemetry: MachineTelemetry,
    ) -> FailurePrediction:

        if self.model is None:
            raise RuntimeError(
                "Failure prediction model not configured"
            )

        # Example integration:
        #
        # raw = self.model.predict({
        #     "temperature": telemetry.temperature,
        #     "vibration": telemetry.vibration,
        #     "pressure": telemetry.pressure,
        #     "current": telemetry.current,
        #     "operating_hours": telemetry.operating_hours,
        # })
        #
        # return FailurePrediction(
        #     failure_probability=float(
        #         raw["failure_probability"]
        #     ),
        #     predicted_failure=bool(
        #         raw["predicted_failure"]
        #     ),
        #     confidence=raw.get("confidence"),
        #     risk_level=raw.get("risk_level"),
        # )

        raise NotImplementedError(
            "Map the production model output"
        )

    def predict_batch(
        self,
        readings: list[MachineTelemetry],
    ) -> list[FailurePrediction]:

        return [
            self.predict(reading)
            for reading in readings
        ]


# ==================================================
# Fixtures
# ==================================================

@pytest.fixture
def predictor() -> FailurePredictorProtocol:

    adapter = FailurePredictionAdapter()

    if adapter.model is None:
        pytest.skip(
            "Connect FailurePredictionAdapter to "
            "the trained failure prediction model."
        )

    return adapter


@pytest.fixture
def healthy_machine() -> MachineTelemetry:
    return HEALTHY_MACHINE


@pytest.fixture
def high_risk_machine() -> MachineTelemetry:
    return HIGH_RISK_MACHINE


# ==================================================
# Healthy machine tests
# ==================================================

class TestHealthyMachine:

    def test_healthy_machine_prediction(
        self,
        predictor: FailurePredictorProtocol,
        healthy_machine: MachineTelemetry,
    ) -> None:

        result = predictor.predict(healthy_machine)

        assert isinstance(result, FailurePrediction)
        assert result.predicted_failure is False

    def test_healthy_machine_probability(
        self,
        predictor: FailurePredictorProtocol,
        healthy_machine: MachineTelemetry,
    ) -> None:

        result = predictor.predict(healthy_machine)

        assert 0.0 <= result.failure_probability <= 1.0

    def test_healthy_machine_prediction_repeatability(
        self,
        predictor: FailurePredictorProtocol,
        healthy_machine: MachineTelemetry,
    ) -> None:

        first = predictor.predict(healthy_machine)
        second = predictor.predict(healthy_machine)

        assert (
            first.predicted_failure
            == second.predicted_failure
        )

        assert first.failure_probability == pytest.approx(
            second.failure_probability
        )


# ==================================================
# High-risk machine tests
# ==================================================

class TestHighRiskMachine:

    def test_high_risk_machine_detected(
        self,
        predictor: FailurePredictorProtocol,
        high_risk_machine: MachineTelemetry,
    ) -> None:

        result = predictor.predict(high_risk_machine)

        assert result.predicted_failure is True

    def test_high_risk_probability_valid(
        self,
        predictor: FailurePredictorProtocol,
        high_risk_machine: MachineTelemetry,
    ) -> None:

        result = predictor.predict(high_risk_machine)

        assert math.isfinite(
            result.failure_probability
        )

        assert 0 <= result.failure_probability <= 1

    def test_high_risk_probability_vs_healthy(
        self,
        predictor: FailurePredictorProtocol,
    ) -> None:

        healthy = predictor.predict(
            HEALTHY_MACHINE
        )

        risky = predictor.predict(
            HIGH_RISK_MACHINE
        )

        # A model-specific expectation that must be
        # validated against the trained model.
        assert (
            risky.failure_probability
            >= healthy.failure_probability
        )


# ==================================================
# Individual sensor tests
# ==================================================

class TestSensorFailurePrediction:

    @pytest.mark.parametrize(
        "reading",
        [
            HIGH_TEMPERATURE,
            HIGH_VIBRATION,
            HIGH_CURRENT,
        ],
        ids=[
            "high-temperature",
            "high-vibration",
            "high-current",
        ],
    )
    def test_sensor_failure_probability_valid(
        self,
        predictor: FailurePredictorProtocol,
        reading: MachineTelemetry,
    ) -> None:

        result = predictor.predict(reading)

        assert 0 <= result.failure_probability <= 1

    @pytest.mark.parametrize(
        "reading",
        [
            HIGH_TEMPERATURE,
            HIGH_VIBRATION,
            HIGH_CURRENT,
        ],
    )
    def test_sensor_prediction_returns_boolean(
        self,
        predictor: FailurePredictorProtocol,
        reading: MachineTelemetry,
    ) -> None:

        result = predictor.predict(reading)

        assert isinstance(
            result.predicted_failure,
            bool,
        )


# ==================================================
# Prediction output validation
# ==================================================

class TestPredictionOutput:

    @pytest.mark.parametrize(
        "reading",
        [
            HEALTHY_MACHINE,
            HIGH_RISK_MACHINE,
            HIGH_TEMPERATURE,
            HIGH_VIBRATION,
            HIGH_CURRENT,
        ],
    )
    def test_probability_is_finite(
        self,
        predictor: FailurePredictorProtocol,
        reading: MachineTelemetry,
    ) -> None:

        result = predictor.predict(reading)

        assert math.isfinite(
            result.failure_probability
        )

    @pytest.mark.parametrize(
        "reading",
        [
            HEALTHY_MACHINE,
            HIGH_RISK_MACHINE,
        ],
    )
    def test_confidence_range(
        self,
        predictor: FailurePredictorProtocol,
        reading: MachineTelemetry,
    ) -> None:

        result = predictor.predict(reading)

        if result.confidence is not None:
            assert math.isfinite(result.confidence)
            assert 0 <= result.confidence <= 1

    def test_risk_level_is_valid(
        self,
        predictor: FailurePredictorProtocol,
    ) -> None:

        result = predictor.predict(
            HIGH_RISK_MACHINE
        )

        if result.risk_level is not None:
            assert result.risk_level in {
                "low",
                "medium",
                "high",
                "critical",
            }

    def test_prediction_matches_configured_threshold(
        self,
        predictor: FailurePredictorProtocol,
    ) -> None:

        threshold = 0.5  # Replace with actual policy.

        result = predictor.predict(
            HIGH_RISK_MACHINE
        )

        assert result.predicted_failure == (
            result.failure_probability >= threshold
        )


# ==================================================
# Batch prediction
# ==================================================

class TestBatchFailurePrediction:

    def test_batch_prediction_count(
        self,
        predictor: FailurePredictorProtocol,
    ) -> None:

        readings = [
            HEALTHY_MACHINE,
            HIGH_RISK_MACHINE,
            HIGH_TEMPERATURE,
            HIGH_VIBRATION,
        ]

        results = predictor.predict_batch(readings)

        assert len(results) == len(readings)

    def test_batch_prediction_types(
        self,
        predictor: FailurePredictorProtocol,
    ) -> None:

        results = predictor.predict_batch([
            HEALTHY_MACHINE,
            HIGH_RISK_MACHINE,
        ])

        assert all(
            isinstance(result, FailurePrediction)
            for result in results
        )

    def test_batch_matches_individual_predictions(
        self,
        predictor: FailurePredictorProtocol,
    ) -> None:

        readings = [
            HEALTHY_MACHINE,
            HIGH_RISK_MACHINE,
            HIGH_VIBRATION,
        ]

        batch = predictor.predict_batch(readings)

        individual = [
            predictor.predict(reading)
            for reading in readings
        ]

        for batch_result, single_result in zip(
            batch,
            individual,
        ):
            assert (
                batch_result.predicted_failure
                == single_result.predicted_failure
            )

            assert batch_result.failure_probability == (
                pytest.approx(
                    single_result.failure_probability
                )
            )

    def test_empty_batch(
        self,
        predictor: FailurePredictorProtocol,
    ) -> None:

        assert predictor.predict_batch([]) == []

    def test_large_batch(
        self,
        predictor: FailurePredictorProtocol,
    ) -> None:

        readings = [HEALTHY_MACHINE] * 100

        results = predictor.predict_batch(readings)

        assert len(results) == 100


# ==================================================
# Invalid telemetry
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
        predictor: FailurePredictorProtocol,
        invalid_value: float,
    ) -> None:

        reading = MachineTelemetry(
            temperature=invalid_value,
            vibration=2.0,
            pressure=4.0,
            current=12.0,
            operating_hours=1000.0,
        )

        with pytest.raises(
            (ValueError, TypeError)
        ):
            predictor.predict(reading)

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
        predictor: FailurePredictorProtocol,
        invalid_value: float,
    ) -> None:

        reading = MachineTelemetry(
            temperature=45.0,
            vibration=invalid_value,
            pressure=4.0,
            current=12.0,
            operating_hours=1000.0,
        )

        with pytest.raises(
            (ValueError, TypeError)
        ):
            predictor.predict(reading)

    def test_negative_operating_hours_rejected(
        self,
        predictor: FailurePredictorProtocol,
    ) -> None:

        reading = MachineTelemetry(
            temperature=45.0,
            vibration=2.0,
            pressure=4.0,
            current=12.0,
            operating_hours=-100.0,
        )

        with pytest.raises(ValueError):
            predictor.predict(reading)

    def test_missing_feature_rejected(
        self,
        predictor: FailurePredictorProtocol,
    ) -> None:

        incomplete = {
            "temperature": 45.0,
            "vibration": 2.0,
            "pressure": 4.0,
            "current": 12.0,
        }

        with pytest.raises(
            (ValueError, TypeError, KeyError)
        ):
            predictor.predict(
                incomplete  # type: ignore[arg-type]
            )


# ==================================================
# Edge cases
# ==================================================

class TestFailurePredictionEdgeCases:

    def test_zero_operating_hours(
        self,
        predictor: FailurePredictorProtocol,
    ) -> None:

        reading = MachineTelemetry(
            temperature=25.0,
            vibration=0.0,
            pressure=0.0,
            current=0.0,
            operating_hours=0.0,
        )

        result = predictor.predict(reading)

        assert isinstance(result, FailurePrediction)

    def test_extreme_temperature(
        self,
        predictor: FailurePredictorProtocol,
    ) -> None:

        reading = MachineTelemetry(
            temperature=1000.0,
            vibration=2.0,
            pressure=4.0,
            current=12.0,
            operating_hours=1000.0,
        )

        try:
            result = predictor.predict(reading)
        except ValueError:
            # Rejecting out-of-domain readings is
            # an acceptable model policy.
            return

        assert 0 <= result.failure_probability <= 1

    def test_high_operating_hours(
        self,
        predictor: FailurePredictorProtocol,
    ) -> None:

        reading = MachineTelemetry(
            temperature=45.0,
            vibration=2.0,
            pressure=4.0,
            current=12.0,
            operating_hours=100000.0,
        )

        result = predictor.predict(reading)

        assert 0 <= result.failure_probability <= 1


# ==================================================
# Data integrity
# ==================================================

class TestPredictionIntegrity:

    def test_input_not_mutated(
        self,
        predictor: FailurePredictorProtocol,
    ) -> None:

        reading = HEALTHY_MACHINE
        original = deepcopy(reading)

        predictor.predict(reading)

        assert reading == original

    def test_deterministic_prediction(
        self,
        predictor: FailurePredictorProtocol,
    ) -> None:

        results = [
            predictor.predict(HEALTHY_MACHINE)
            for _ in range(5)
        ]

        first = results[0]

        for result in results[1:]:
            assert (
                result.predicted_failure
                == first.predicted_failure
            )

            assert result.failure_probability == (
                pytest.approx(
                    first.failure_probability
                )
            )


# ==================================================
# Model calibration helper tests
# ==================================================

class TestFailureProbabilityCalibration:

    def test_probability_range_across_examples(
        self,
        predictor: FailurePredictorProtocol,
    ) -> None:

        readings = [
            HEALTHY_MACHINE,
            HIGH_RISK_MACHINE,
            HIGH_TEMPERATURE,
            HIGH_VIBRATION,
            HIGH_CURRENT,
        ]

        for reading in readings:
            result = predictor.predict(reading)

            assert math.isfinite(
                result.failure_probability
            )

            assert 0 <= result.failure_probability <= 1

    def test_high_risk_vs_healthy_probability(
        self,
        predictor: FailurePredictorProtocol,
    ) -> None:

        normal = predictor.predict(
            HEALTHY_MACHINE
        )

        high_risk = predictor.predict(
            HIGH_RISK_MACHINE
        )

        assert (
            high_risk.failure_probability
            >= normal.failure_probability
        )
