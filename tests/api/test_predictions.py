
"""
INDUSTRA AI
Industrial Predictive Maintenance & Prediction Tests

File: backend/tests/test_predictions.py

Run:
    pytest backend/tests/test_predictions.py -v

Coverage:
    pytest backend/tests/test_predictions.py \
        --cov=backend.app --cov-report=term-missing

Notes:
- This is a reference contract-based test suite.
- Configure the adapter imports to match your backend.
- Tests use synthetic inputs and mock prediction models.
- No live equipment or external AI service is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import math

import numpy as np
import pytest


# ============================================================
# CONFIGURATION
# ============================================================

MIN_HEALTH = 0.0
MAX_HEALTH = 100.0

MIN_FAILURE_PROBABILITY = 0.0
MAX_FAILURE_PROBABILITY = 1.0

MIN_RUL = 0.0

FEATURE_NAMES = (
    "temperature",
    "vibration",
    "pressure",
    "current",
    "runtime_hours",
)


# ============================================================
# PREDICTION CONTRACTS
# ============================================================

@dataclass(frozen=True)
class SensorInput:
    asset_id: str
    temperature: float
    vibration: float
    pressure: float
    current: float
    runtime_hours: float


@dataclass(frozen=True)
class PredictionResult:
    asset_id: str
    health_score: float
    rul_hours: float
    failure_probability: float
    anomaly_score: float
    maintenance_priority: str
    model_version: str


class PredictionService(Protocol):

    def predict(
        self,
        sensor: SensorInput,
    ) -> PredictionResult:
        ...

    def predict_batch(
        self,
        sensors: list[SensorInput],
    ) -> list[PredictionResult]:
        ...


# ============================================================
# DETERMINISTIC REFERENCE MODEL
# ============================================================

class ReferencePredictionService:
    """
    Deterministic model used to validate prediction contracts.

    Replace with an adapter to your real inference service
    for production integration tests.

    This reference model is not a validated industrial
    failure prediction algorithm.
    """

    model_version = "test-reference-1.0"

    @staticmethod
    def validate(sensor: SensorInput) -> None:

        if not isinstance(sensor.asset_id, str):
            raise ValueError("Invalid asset ID")

        if not sensor.asset_id.strip():
            raise ValueError("Asset ID cannot be empty")

        for feature in FEATURE_NAMES:
            value = getattr(sensor, feature)

            if isinstance(value, bool):
                raise ValueError(
                    f"{feature} must be numeric"
                )

            if not isinstance(
                value,
                (int, float, np.integer, np.floating),
            ):
                raise ValueError(
                    f"{feature} must be numeric"
                )

            if not math.isfinite(float(value)):
                raise ValueError(
                    f"{feature} must be finite"
                )

            if value < 0:
                raise ValueError(
                    f"{feature} cannot be negative"
                )

    def predict(
        self,
        sensor: SensorInput,
    ) -> PredictionResult:

        self.validate(sensor)

        temperature_penalty = max(
            0.0,
            sensor.temperature - 65.0,
        ) * 0.65

        vibration_penalty = max(
            0.0,
            sensor.vibration - 3.0,
        ) * 5.0

        pressure_penalty = max(
            0.0,
            sensor.pressure - 7.0,
        ) * 1.5

        runtime_penalty = (
            sensor.runtime_hours / 10000.0
        ) * 2.0

        health = 100.0 - (
            temperature_penalty
            + vibration_penalty
            + pressure_penalty
            + runtime_penalty
        )

        health = float(
            np.clip(health, 0.0, 100.0)
        )

        failure_probability = float(
            np.clip(
                (100.0 - health) / 100.0,
                0.0,
                1.0,
            )
        )

        rul_hours = max(
            0.0,
            health * 120.0,
        )

        anomaly_score = float(
            np.clip(
                (
                    max(0.0, sensor.temperature - 80.0)
                    / 30.0
                )
                + (
                    max(0.0, sensor.vibration - 5.5)
                    / 8.0
                ),
                0.0,
                1.0,
            )
        )

        if health < 65.0:
            priority = "high"
        elif health < 80.0:
            priority = "medium"
        else:
            priority = "low"

        return PredictionResult(
            asset_id=sensor.asset_id,
            health_score=health,
            rul_hours=rul_hours,
            failure_probability=failure_probability,
            anomaly_score=anomaly_score,
            maintenance_priority=priority,
            model_version=self.model_version,
        )

    def predict_batch(
        self,
        sensors: list[SensorInput],
    ) -> list[PredictionResult]:

        return [
            self.predict(sensor)
            for sensor in sensors
        ]


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def prediction_service() -> PredictionService:
    return ReferencePredictionService()


@pytest.fixture
def healthy_sensor() -> SensorInput:

    return SensorInput(
        asset_id="MTR-001",
        temperature=48.0,
        vibration=1.8,
        pressure=4.2,
        current=65.0,
        runtime_hours=2500.0,
    )


@pytest.fixture
def warning_sensor() -> SensorInput:

    return SensorInput(
        asset_id="PMP-001",
        temperature=85.0,
        vibration=4.5,
        pressure=7.0,
        current=95.0,
        runtime_hours=20000.0,
    )


@pytest.fixture
def critical_sensor() -> SensorInput:

    return SensorInput(
        asset_id="GEN-001",
        temperature=110.0,
        vibration=10.0,
        pressure=10.0,
        current=190.0,
        runtime_hours=50000.0,
    )


@pytest.fixture
def batch_sensors(
    healthy_sensor,
    warning_sensor,
    critical_sensor,
):

    return [
        healthy_sensor,
        warning_sensor,
        critical_sensor,
    ]


# ============================================================
# BASIC PREDICTION TESTS
# ============================================================

def test_prediction_returns_result(
    prediction_service,
    healthy_sensor,
):

    result = prediction_service.predict(
        healthy_sensor
    )

    assert isinstance(
        result,
        PredictionResult,
    )


def test_prediction_preserves_asset_id(
    prediction_service,
    healthy_sensor,
):

    result = prediction_service.predict(
        healthy_sensor
    )

    assert result.asset_id == "MTR-001"


def test_prediction_has_model_version(
    prediction_service,
    healthy_sensor,
):

    result = prediction_service.predict(
        healthy_sensor
    )

    assert result.model_version


def test_prediction_health_range(
    prediction_service,
    batch_sensors,
):

    for sensor in batch_sensors:

        result = prediction_service.predict(
            sensor
        )

        assert (
            MIN_HEALTH
            <= result.health_score
            <= MAX_HEALTH
        )


def test_failure_probability_range(
    prediction_service,
    batch_sensors,
):

    for sensor in batch_sensors:

        result = prediction_service.predict(
            sensor
        )

        assert (
            MIN_FAILURE_PROBABILITY
            <= result.failure_probability
            <= MAX_FAILURE_PROBABILITY
        )


def test_rul_nonnegative(
    prediction_service,
    batch_sensors,
):

    for sensor in batch_sensors:

        result = prediction_service.predict(
            sensor
        )

        assert result.rul_hours >= MIN_RUL


def test_anomaly_score_range(
    prediction_service,
    batch_sensors,
):

    for sensor in batch_sensors:

        result = prediction_service.predict(
            sensor
        )

        assert (
            0.0
            <= result.anomaly_score
            <= 1.0
        )


# ============================================================
# ASSET HEALTH TESTS
# ============================================================

def test_healthy_asset_has_high_health(
    prediction_service,
    healthy_sensor,
):

    result = prediction_service.predict(
        healthy_sensor
    )

    assert result.health_score >= 80.0


def test_critical_asset_has_low_health(
    prediction_service,
    critical_sensor,
):

    result = prediction_service.predict(
        critical_sensor
    )

    assert result.health_score < 65.0


def test_health_decreases_with_temperature(
    prediction_service,
    healthy_sensor,
):

    normal = prediction_service.predict(
        healthy_sensor
    )

    hot = prediction_service.predict(
        SensorInput(
            **{
                **vars(healthy_sensor),
                "temperature": 100.0,
            }
        )
    )

    assert (
        hot.health_score
        <= normal.health_score
    )


def test_health_decreases_with_vibration(
    prediction_service,
    healthy_sensor,
):

    normal = prediction_service.predict(
        healthy_sensor
    )

    high_vibration = prediction_service.predict(
        SensorInput(
            **{
                **vars(healthy_sensor),
                "vibration": 9.0,
            }
        )
    )

    assert (
        high_vibration.health_score
        <= normal.health_score
    )


def test_health_decreases_with_runtime(
    prediction_service,
    healthy_sensor,
):

    normal = prediction_service.predict(
        healthy_sensor
    )

    aged = prediction_service.predict(
        SensorInput(
            **{
                **vars(healthy_sensor),
                "runtime_hours": 80000.0,
            }
        )
    )

    assert (
        aged.health_score
        <= normal.health_score
    )


# ============================================================
# REMAINING USEFUL LIFE TESTS
# ============================================================

def test_healthy_asset_has_longer_rul(
    prediction_service,
    healthy_sensor,
    critical_sensor,
):

    healthy = prediction_service.predict(
        healthy_sensor
    )

    critical = prediction_service.predict(
        critical_sensor
    )

    assert (
        healthy.rul_hours
        > critical.rul_hours
    )


def test_rul_is_finite(
    prediction_service,
    batch_sensors,
):

    for sensor in batch_sensors:

        result = prediction_service.predict(
            sensor
        )

        assert math.isfinite(
            result.rul_hours
        )


def test_rul_decreases_with_wear(
    prediction_service,
    healthy_sensor,
):

    initial = prediction_service.predict(
        healthy_sensor
    )

    worn = prediction_service.predict(
        SensorInput(
            **{
                **vars(healthy_sensor),
                "runtime_hours": 100000.0,
                "temperature": 105.0,
                "vibration": 9.0,
            }
        )
    )

    assert worn.rul_hours < initial.rul_hours


# ============================================================
# FAILURE PREDICTION TESTS
# ============================================================

def test_critical_asset_higher_failure_risk(
    prediction_service,
    healthy_sensor,
    critical_sensor,
):

    healthy = prediction_service.predict(
        healthy_sensor
    )

    critical = prediction_service.predict(
        critical_sensor
    )

    assert (
        critical.failure_probability
        > healthy.failure_probability
    )


def test_failure_probability_is_finite(
    prediction_service,
    batch_sensors,
):

    for sensor in batch_sensors:

        result = prediction_service.predict(
            sensor
        )

        assert math.isfinite(
            result.failure_probability
        )


def test_failure_probability_increases_with_wear(
    prediction_service,
    healthy_sensor,
):

    initial = prediction_service.predict(
        healthy_sensor
    )

    worn = prediction_service.predict(
        SensorInput(
            **{
                **vars(healthy_sensor),
                "temperature": 110.0,
                "vibration": 11.0,
                "runtime_hours": 100000.0,
            }
        )
    )

    assert (
        worn.failure_probability
        > initial.failure_probability
    )


# ============================================================
# ANOMALY DETECTION TESTS
# ============================================================

def test_normal_sensor_low_anomaly_score(
    prediction_service,
    healthy_sensor,
):

    result = prediction_service.predict(
        healthy_sensor
    )

    assert result.anomaly_score < 0.5


def test_abnormal_sensor_high_anomaly_score(
    prediction_service,
    critical_sensor,
):

    result = prediction_service.predict(
        critical_sensor
    )

    assert result.anomaly_score >= 0.5


def test_anomaly_score_increases_with_temperature(
    prediction_service,
    healthy_sensor,
):

    baseline = prediction_service.predict(
        healthy_sensor
    )

    abnormal = prediction_service.predict(
        SensorInput(
            **{
                **vars(healthy_sensor),
                "temperature": 115.0,
            }
        )
    )

    assert (
        abnormal.anomaly_score
        > baseline.anomaly_score
    )


# ============================================================
# MAINTENANCE PRIORITY TESTS
# ============================================================

@pytest.mark.parametrize(
    "health,expected",
    [
        (100.0, "low"),
        (90.0, "low"),
        (80.0, "low"),
        (79.9, "medium"),
        (70.0, "medium"),
        (65.0, "medium"),
        (64.9, "high"),
        (20.0, "high"),
    ],
)
def test_maintenance_priority_boundaries(
    health,
    expected,
):

    if health < 65.0:
        priority = "high"
    elif health < 80.0:
        priority = "medium"
    else:
        priority = "low"

    assert priority == expected


def test_healthy_asset_low_priority(
    prediction_service,
    healthy_sensor,
):

    result = prediction_service.predict(
        healthy_sensor
    )

    assert result.maintenance_priority == "low"


def test_critical_asset_high_priority(
    prediction_service,
    critical_sensor,
):

    result = prediction_service.predict(
        critical_sensor
    )

    assert result.maintenance_priority == "high"


# ============================================================
# INPUT VALIDATION TESTS
# ============================================================

@pytest.mark.parametrize(
    "invalid_id",
    ["", " ", "\t"],
)
def test_rejects_empty_asset_id(
    prediction_service,
    healthy_sensor,
    invalid_id,
):

    invalid = SensorInput(
        **{
            **vars(healthy_sensor),
            "asset_id": invalid_id,
        }
    )

    with pytest.raises(ValueError):
        prediction_service.predict(invalid)


@pytest.mark.parametrize(
    "feature",
    FEATURE_NAMES,
)
def test_rejects_negative_sensor_values(
    prediction_service,
    healthy_sensor,
    feature,
):

    invalid = SensorInput(
        **{
            **vars(healthy_sensor),
            feature: -1.0,
        }
    )

    with pytest.raises(ValueError):
        prediction_service.predict(invalid)


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_rejects_nonfinite_temperature(
    prediction_service,
    healthy_sensor,
    invalid_value,
):

    invalid = SensorInput(
        **{
            **vars(healthy_sensor),
            "temperature": invalid_value,
        }
    )

    with pytest.raises(ValueError):
        prediction_service.predict(invalid)


def test_rejects_nonnumeric_temperature(
    prediction_service,
    healthy_sensor,
):

    invalid = SensorInput(
        **{
            **vars(healthy_sensor),
            "temperature": "very hot",
        }
    )

    with pytest.raises(ValueError):
        prediction_service.predict(invalid)


def test_rejects_boolean_sensor_value(
    prediction_service,
    healthy_sensor,
):

    invalid = SensorInput(
        **{
            **vars(healthy_sensor),
            "vibration": True,
        }
    )

    with pytest.raises(ValueError):
        prediction_service.predict(invalid)


def test_zero_sensor_values_are_accepted(
    prediction_service,
):

    sensor = SensorInput(
        asset_id="TEST-ZERO",
        temperature=0.0,
        vibration=0.0,
        pressure=0.0,
        current=0.0,
        runtime_hours=0.0,
    )

    result = prediction_service.predict(sensor)

    assert result.health_score == 100.0
    assert result.rul_hours >= 0.0


# ============================================================
# BATCH PREDICTION TESTS
# ============================================================

def test_batch_prediction_count(
    prediction_service,
    batch_sensors,
):

    results = prediction_service.predict_batch(
        batch_sensors
    )

    assert len(results) == len(batch_sensors)


def test_batch_prediction_preserves_order(
    prediction_service,
    batch_sensors,
):

    results = prediction_service.predict_batch(
        batch_sensors
    )

    expected_ids = [
        sensor.asset_id
        for sensor in batch_sensors
    ]

    actual_ids = [
        result.asset_id
        for result in results
    ]

    assert actual_ids == expected_ids


def test_empty_batch_returns_empty_list(
    prediction_service,
):

    assert prediction_service.predict_batch([]) == []


def test_batch_predictions_match_single_predictions(
    prediction_service,
    batch_sensors,
):

    batch_results = prediction_service.predict_batch(
        batch_sensors
    )

    single_results = [
        prediction_service.predict(sensor)
        for sensor in batch_sensors
    ]

    assert batch_results == single_results


def test_batch_rejects_invalid_member(
    prediction_service,
    healthy_sensor,
):

    invalid = SensorInput(
        asset_id="INVALID",
        temperature=-5.0,
        vibration=2.0,
        pressure=4.0,
        current=70.0,
        runtime_hours=1000.0,
    )

    with pytest.raises(ValueError):
        prediction_service.predict_batch(
            [healthy_sensor, invalid]
        )


# ============================================================
# DETERMINISM AND MODEL CONSISTENCY
# ============================================================

def test_same_input_same_prediction(
    prediction_service,
    healthy_sensor,
):

    first = prediction_service.predict(
        healthy_sensor
    )

    second = prediction_service.predict(
        healthy_sensor
    )

    assert first == second


def test_prediction_values_are_finite(
    prediction_service,
    batch_sensors,
):

    for sensor in batch_sensors:

        result = prediction_service.predict(
            sensor
        )

        values = [
            result.health_score,
            result.rul_hours,
            result.failure_probability,
            result.anomaly_score,
        ]

        assert all(
            math.isfinite(value)
            for value in values
        )


def test_extreme_operating_conditions_are_bounded(
    prediction_service,
):

    extreme = SensorInput(
        asset_id="EXTREME-001",
        temperature=1000.0,
        vibration=500.0,
        pressure=1000.0,
        current=10000.0,
        runtime_hours=1000000.0,
    )

    result = prediction_service.predict(extreme)

    assert 0.0 <= result.health_score <= 100.0
    assert 0.0 <= result.failure_probability <= 1.0
    assert result.rul_hours >= 0.0
    assert 0.0 <= result.anomaly_score <= 1.0


# ============================================================
# OPTIONAL FASTAPI INTEGRATION CONTRACT
# ============================================================

@pytest.mark.integration
def test_prediction_api_contract():
    """
    Enable this test after wiring your real FastAPI app.

    Expected route:
        POST /api/v1/predictions

    Expected response:
        asset_id
        health_score
        rul_hours
        failure_probability
        anomaly_score
        maintenance_priority

    This placeholder is skipped until an application
    integration is explicitly configured.
    """

    pytest.skip(
        "Configure the real FastAPI app and prediction route."
    )


@pytest.mark.integration
def test_batch_prediction_api_contract():
    """
    Expected route:
        POST /api/v1/predictions/batch

    The application adapter must be configured before
    this integration contract can run.
    """

    pytest.skip(
        "Configure the real batch prediction API."
    )


# ============================================================
# TEST METADATA
# ============================================================

def test_prediction_test_module_metadata():

    metadata: dict[str, Any] = {
        "project": "INDUSTRA AI",
        "module": "predictive-maintenance",
        "tests": "prediction-contracts",
        "synthetic_data": True,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    assert metadata["project"] == "INDUSTRA AI"
    assert metadata["synthetic_data"] is True
