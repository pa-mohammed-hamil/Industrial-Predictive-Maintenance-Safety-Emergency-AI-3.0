
"""
tests/unit/test_rul.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

Remaining Useful Life (RUL) Prediction Tests

Run:
    pytest tests/unit/test_rul.py -v

Coverage:
    - RUL prediction
    - Healthy/degraded/critical assets
    - RUL units and boundaries
    - Temperature and vibration sensitivity
    - Failure probability
    - Operating hours
    - Confidence intervals
    - Maintenance recommendations
    - Batch predictions
    - Invalid sensor readings
    - Deterministic inference
    - Input immutability

IMPORTANT:
Connect RULPredictorAdapter to the actual
production RUL prediction implementation.

Synthetic telemetry is used for contract tests.
Real prediction accuracy must be evaluated with
run-to-failure datasets and known failure times.

No industrial equipment is controlled here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Protocol

import pytest


# ==================================================
# Input models
# ==================================================

@dataclass(frozen=True)
class RULTelemetry:
    asset_id: str
    temperature: float
    vibration: float
    pressure: float
    current: float
    operating_hours: float
    failure_probability: float = 0.0
    health_score: float = 100.0


@dataclass(frozen=True)
class RULPrediction:
    asset_id: str
    rul_hours: float
    confidence: float | None = None
    lower_bound_hours: float | None = None
    upper_bound_hours: float | None = None
    risk_level: str | None = None
    maintenance_required: bool | None = None


# ==================================================
# Synthetic test telemetry
# ==================================================

HEALTHY_ASSET = RULTelemetry(
    asset_id="MOTOR-001",
    temperature=45.0,
    vibration=2.0,
    pressure=4.0,
    current=12.0,
    operating_hours=1000.0,
    failure_probability=0.02,
    health_score=95.0,
)

DEGRADED_ASSET = RULTelemetry(
    asset_id="MOTOR-002",
    temperature=85.0,
    vibration=10.0,
    pressure=8.0,
    current=25.0,
    operating_hours=9000.0,
    failure_probability=0.45,
    health_score=55.0,
)

CRITICAL_ASSET = RULTelemetry(
    asset_id="MOTOR-003",
    temperature=140.0,
    vibration=25.0,
    pressure=18.0,
    current=50.0,
    operating_hours=18000.0,
    failure_probability=0.92,
    health_score=10.0,
)


# ==================================================
# Production interface
# ==================================================

class RULPredictorProtocol(Protocol):

    def predict(
        self,
        telemetry: RULTelemetry,
    ) -> RULPrediction:
        ...

    def predict_batch(
        self,
        readings: list[RULTelemetry],
    ) -> list[RULPrediction]:
        ...


# ==================================================
# Production adapter
# ==================================================

class RULPredictorAdapter:
    """
    Connect this adapter to your actual RUL model.

    Possible module:

        app/ml/predictive_maintenance/rul.py

    Example:

        from app.ml.predictive_maintenance.rul import (
            RULPredictor
        )

        self.model = RULPredictor(...)

    Adapt the production input/output schemas
    to RULTelemetry and RULPrediction.

    Do not create a second prediction algorithm
    inside the adapter. The tests should verify
    the actual production model.
    """

    def __init__(self) -> None:
        self.model: Any = None

        # TODO:
        # Initialize production predictor and
        # load its trained model artifacts.

    def predict(
        self,
        telemetry: RULTelemetry,
    ) -> RULPrediction:

        if self.model is None:
            raise RuntimeError(
                "Production RUL predictor not configured"
            )

        # Example integration:
        #
        # raw = self.model.predict(
        #     asset_id=telemetry.asset_id,
        #     temperature=telemetry.temperature,
        #     vibration=telemetry.vibration,
        #     pressure=telemetry.pressure,
        #     current=telemetry.current,
        #     operating_hours=telemetry.operating_hours,
        #     failure_probability=(
        #         telemetry.failure_probability
        #     ),
        #     health_score=telemetry.health_score,
        # )
        #
        # return RULPrediction(
        #     asset_id=str(raw["asset_id"]),
        #     rul_hours=float(raw["rul_hours"]),
        #     confidence=raw.get("confidence"),
        #     lower_bound_hours=raw.get(
        #         "lower_bound_hours"
        #     ),
        #     upper_bound_hours=raw.get(
        #         "upper_bound_hours"
        #     ),
        #     risk_level=raw.get("risk_level"),
        #     maintenance_required=raw.get(
        #         "maintenance_required"
        #     ),
        # )

        raise NotImplementedError(
            "Map the production RUL prediction output"
        )

    def predict_batch(
        self,
        readings: list[RULTelemetry],
    ) -> list[RULPrediction]:

        return [
            self.predict(reading)
            for reading in readings
        ]


# ==================================================
# Fixtures
# ==================================================

@pytest.fixture
def predictor() -> RULPredictorProtocol:

    adapter = RULPredictorAdapter()

    if adapter.model is None:
        pytest.skip(
            "Connect RULPredictorAdapter to "
            "the production RUL model."
        )

    return adapter


@pytest.fixture
def healthy_asset() -> RULTelemetry:
    return HEALTHY_ASSET


@pytest.fixture
def degraded_asset() -> RULTelemetry:
    return DEGRADED_ASSET


@pytest.fixture
def critical_asset() -> RULTelemetry:
    return CRITICAL_ASSET


# ==================================================
# Basic prediction
# ==================================================

class TestRULPrediction:

    def test_healthy_asset_prediction(
        self,
        predictor: RULPredictorProtocol,
        healthy_asset: RULTelemetry,
    ) -> None:

        result = predictor.predict(healthy_asset)

        assert isinstance(result, RULPrediction)
        assert result.rul_hours >= 0

    def test_degraded_asset_prediction(
        self,
        predictor: RULPredictorProtocol,
        degraded_asset: RULTelemetry,
    ) -> None:

        result = predictor.predict(degraded_asset)

        assert math.isfinite(result.rul_hours)
        assert result.rul_hours >= 0

    def test_critical_asset_prediction(
        self,
        predictor: RULPredictorProtocol,
        critical_asset: RULTelemetry,
    ) -> None:

        result = predictor.predict(critical_asset)

        assert math.isfinite(result.rul_hours)
        assert result.rul_hours >= 0

    def test_asset_id_preserved(
        self,
        predictor: RULPredictorProtocol,
        healthy_asset: RULTelemetry,
    ) -> None:

        result = predictor.predict(healthy_asset)

        assert result.asset_id == healthy_asset.asset_id

    def test_rul_is_numeric(
        self,
        predictor: RULPredictorProtocol,
        healthy_asset: RULTelemetry,
    ) -> None:

        result = predictor.predict(healthy_asset)

        assert isinstance(
            result.rul_hours,
            (int, float),
        )

    def test_rul_is_finite(
        self,
        predictor: RULPredictorProtocol,
        healthy_asset: RULTelemetry,
    ) -> None:

        result = predictor.predict(healthy_asset)

        assert math.isfinite(result.rul_hours)


# ==================================================
# Asset health comparisons
# ==================================================

class TestRULHealthComparison:

    def test_healthy_asset_has_higher_rul(
        self,
        predictor: RULPredictorProtocol,
    ) -> None:

        healthy = predictor.predict(
            HEALTHY_ASSET
        )

        degraded = predictor.predict(
            DEGRADED_ASSET
        )

        critical = predictor.predict(
            CRITICAL_ASSET
        )

        # This is an illustrative monotonicity
        # requirement for the synthetic profiles.
        assert (
            healthy.rul_hours
            >= degraded.rul_hours
            >= critical.rul_hours
        )

    def test_high_temperature_effect(
        self,
        predictor: RULPredictorProtocol,
    ) -> None:

        normal = HEALTHY_ASSET

        hot = replace(
            normal,
            temperature=130.0,
        )

        normal_result = predictor.predict(normal)
        hot_result = predictor.predict(hot)

        assert (
            hot_result.rul_hours
            <= normal_result.rul_hours
        )

    def test_high_vibration_effect(
        self,
        predictor: RULPredictorProtocol,
    ) -> None:

        normal = HEALTHY_ASSET

        vibrating = replace(
            normal,
            vibration=25.0,
        )

        normal_result = predictor.predict(normal)
        vibration_result = predictor.predict(
            vibrating
        )

        assert (
            vibration_result.rul_hours
            <= normal_result.rul_hours
        )

    def test_high_failure_probability_effect(
        self,
        predictor: RULPredictorProtocol,
    ) -> None:

        normal = HEALTHY_ASSET

        risky = replace(
            normal,
            failure_probability=0.95,
        )

        normal_result = predictor.predict(normal)
        risky_result = predictor.predict(risky)

        assert (
            risky_result.rul_hours
            <= normal_result.rul_hours
        )

    def test_low_health_score_effect(
        self,
        predictor: RULPredictorProtocol,
    ) -> None:

        normal = HEALTHY_ASSET

        unhealthy = replace(
            normal,
            health_score=10.0,
        )

        normal_result = predictor.predict(normal)
        unhealthy_result = predictor.predict(
            unhealthy
        )

        assert (
            unhealthy_result.rul_hours
            <= normal_result.rul_hours
        )


# ==================================================
# Operating hours
# ==================================================

class TestOperatingHours:

    def test_higher_operating_hours_effect(
        self,
        predictor: RULPredictorProtocol,
    ) -> None:

        low_hours = HEALTHY_ASSET

        high_hours = replace(
            low_hours,
            operating_hours=15000.0,
        )

        low_result = predictor.predict(low_hours)
        high_result = predictor.predict(high_hours)

        # Assumes other operating conditions
        # remain equal and age is a degradation
        # factor in the production model.
        assert (
            high_result.rul_hours
            <= low_result.rul_hours
        )

    def test_zero_operating_hours(
        self,
        predictor: RULPredictorProtocol,
    ) -> None:

        asset = replace(
            HEALTHY_ASSET,
            operating_hours=0.0,
        )

        result = predictor.predict(asset)

        assert result.rul_hours >= 0

    def test_large_operating_hours(
        self,
        predictor: RULPredictorProtocol,
    ) -> None:

        asset = replace(
            HEALTHY_ASSET,
            operating_hours=100000.0,
        )

        result = predictor.predict(asset)

        assert math.isfinite(result.rul_hours)


# ==================================================
# Confidence and uncertainty
# ==================================================

class TestRULConfidence:

    def test_confidence_range(
        self,
        predictor: RULPredictorProtocol,
        healthy_asset: RULTelemetry,
    ) -> None:

        result = predictor.predict(healthy_asset)

        if result.confidence is not None:
            assert math.isfinite(result.confidence)
            assert 0 <= result.confidence <= 1

    def test_prediction_interval_valid(
        self,
        predictor: RULPredictorProtocol,
        healthy_asset: RULTelemetry,
    ) -> None:

        result = predictor.predict(healthy_asset)

        lower = result.lower_bound_hours
        upper = result.upper_bound_hours

        if lower is not None and upper is not None:
            assert math.isfinite(lower)
            assert math.isfinite(upper)

            assert lower >= 0
            assert upper >= lower

    def test_prediction_within_interval(
        self,
        predictor: RULPredictorProtocol,
        healthy_asset: RULTelemetry,
    ) -> None:

        result = predictor.predict(healthy_asset)

        lower = result.lower_bound_hours
        upper = result.upper_bound_hours

        if lower is not None and upper is not None:
            assert (
                lower
                <= result.rul_hours
                <= upper
            )

    def test_valid_risk_level(
        self,
        predictor: RULPredictorProtocol,
        critical_asset: RULTelemetry,
    ) -> None:

        result = predictor.predict(critical_asset)

        if result.risk_level is not None:
            assert result.risk_level.lower() in {
                "low",
                "medium",
                "high",
                "critical",
            }

    def test_maintenance_required_type(
        self,
        predictor: RULPredictorProtocol,
        critical_asset: RULTelemetry,
    ) -> None:

        result = predictor.predict(critical_asset)

        if result.maintenance_required is not None:
            assert isinstance(
                result.maintenance_required,
                bool,
            )


# ==================================================
# Batch predictions
# ==================================================

class TestRULBatchPrediction:

    def test_batch_result_count(
        self,
        predictor: RULPredictorProtocol,
    ) -> None:

        assets = [
            HEALTHY_ASSET,
            DEGRADED_ASSET,
            CRITICAL_ASSET,
        ]

        results = predictor.predict_batch(assets)

        assert len(results) == 3

    def test_batch_preserves_asset_ids(
        self,
        predictor: RULPredictorProtocol,
    ) -> None:

        assets = [
            HEALTHY_ASSET,
            DEGRADED_ASSET,
            CRITICAL_ASSET,
        ]

        results = predictor.predict_batch(assets)

        assert [r.asset_id for r in results] == [
            "MOTOR-001",
            "MOTOR-002",
            "MOTOR-003",
        ]

    def test_batch_result_types(
        self,
        predictor: RULPredictorProtocol,
    ) -> None:

        results = predictor.predict_batch([
            HEALTHY_ASSET,
            DEGRADED_ASSET,
        ])

        assert all(
            isinstance(result, RULPrediction)
            for result in results
        )

    def test_empty_batch(
        self,
        predictor: RULPredictorProtocol,
    ) -> None:

        assert predictor.predict_batch([]) == []

    def test_batch_matches_individual(
        self,
        predictor: RULPredictorProtocol,
    ) -> None:

        assets = [
            HEALTHY_ASSET,
            DEGRADED_ASSET,
            CRITICAL_ASSET,
        ]

        batch = predictor.predict_batch(assets)

        individual = [
            predictor.predict(asset)
            for asset in assets
        ]

        for batch_result, single_result in zip(
            batch,
            individual,
        ):
            assert batch_result.asset_id == (
                single_result.asset_id
            )

            assert batch_result.rul_hours == (
                pytest.approx(
                    single_result.rul_hours
                )
            )

    def test_large_batch(
        self,
        predictor: RULPredictorProtocol,
    ) -> None:

        assets = [
            replace(
                HEALTHY_ASSET,
                asset_id=f"MOTOR-{i:04d}",
            )
            for i in range(100)
        ]

        results = predictor.predict_batch(assets)

        assert len(results) == 100


# ==================================================
# Input validation
# ==================================================

class TestRULValidation:

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
        predictor: RULPredictorProtocol,
        invalid_value: float,
    ) -> None:

        asset = replace(
            HEALTHY_ASSET,
            temperature=invalid_value,
        )

        with pytest.raises(
            (ValueError, TypeError)
        ):
            predictor.predict(asset)

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
        predictor: RULPredictorProtocol,
        invalid_value: float,
    ) -> None:

        asset = replace(
            HEALTHY_ASSET,
            vibration=invalid_value,
        )

        with pytest.raises(
            (ValueError, TypeError)
        ):
            predictor.predict(asset)

    def test_negative_operating_hours_rejected(
        self,
        predictor: RULPredictorProtocol,
    ) -> None:

        asset = replace(
            HEALTHY_ASSET,
            operating_hours=-100.0,
        )

        with pytest.raises(ValueError):
            predictor.predict(asset)

    def test_negative_vibration_rejected(
        self,
        predictor: RULPredictorProtocol,
    ) -> None:

        asset = replace(
            HEALTHY_ASSET,
            vibration=-5.0,
        )

        with pytest.raises(ValueError):
            predictor.predict(asset)

    @pytest.mark.parametrize(
        "probability",
        [
            -0.1,
            1.1,
            2.0,
        ],
    )
    def test_invalid_failure_probability(
        self,
        predictor: RULPredictorProtocol,
        probability: float,
    ) -> None:

        asset = replace(
            HEALTHY_ASSET,
            failure_probability=probability,
        )

        with pytest.raises(ValueError):
            predictor.predict(asset)

    @pytest.mark.parametrize(
        "score",
        [
            -1.0,
            101.0,
            150.0,
        ],
    )
    def test_invalid_health_score(
        self,
        predictor: RULPredictorProtocol,
        score: float,
    ) -> None:

        asset = replace(
            HEALTHY_ASSET,
            health_score=score,
        )

        with pytest.raises(ValueError):
            predictor.predict(asset)

    def test_empty_asset_id_rejected(
        self,
        predictor: RULPredictorProtocol,
    ) -> None:

        asset = replace(
            HEALTHY_ASSET,
            asset_id="",
        )

        with pytest.raises(ValueError):
            predictor.predict(asset)

    def test_missing_telemetry_rejected(
        self,
        predictor: RULPredictorProtocol,
    ) -> None:

        with pytest.raises(
            (ValueError, TypeError)
        ):
            predictor.predict(
                None  # type: ignore[arg-type]
            )


# ==================================================
# Prediction integrity
# ==================================================

class TestRULPredictionIntegrity:

    def test_input_not_mutated(
        self,
        predictor: RULPredictorProtocol,
        healthy_asset: RULTelemetry,
    ) -> None:

        original = replace(healthy_asset)

        predictor.predict(healthy_asset)

        assert healthy_asset == original

    def test_deterministic_prediction(
        self,
        predictor: RULPredictorProtocol,
        healthy_asset: RULTelemetry,
    ) -> None:

        predictions = [
            predictor.predict(
                healthy_asset
            ).rul_hours
            for _ in range(5)
        ]

        for value in predictions:
            assert value == pytest.approx(
                predictions[0]
            )

    def test_prediction_never_negative(
        self,
        predictor: RULPredictorProtocol,
    ) -> None:

        for asset in [
            HEALTHY_ASSET,
            DEGRADED_ASSET,
            CRITICAL_ASSET,
        ]:
            result = predictor.predict(asset)

            assert result.rul_hours >= 0

    def test_prediction_never_nan(
        self,
        predictor: RULPredictorProtocol,
    ) -> None:

        for asset in [
            HEALTHY_ASSET,
            DEGRADED_ASSET,
            CRITICAL_ASSET,
        ]:
            result = predictor.predict(asset)

            assert math.isfinite(result.rul_hours)

    def test_prediction_output_type(
        self,
        predictor: RULPredictorProtocol,
        healthy_asset: RULTelemetry,
    ) -> None:

        result = predictor.predict(healthy_asset)

        assert isinstance(
            result,
            RULPrediction,
        )
