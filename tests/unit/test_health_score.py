
"""
tests/unit/test_health_score.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

Asset Health Score Unit Tests

Run:
    pytest tests/unit/test_health_score.py -v

Tests:
    - Healthy and degraded equipment
    - Critical equipment health
    - Temperature, vibration, pressure, current
    - Anomaly and failure risk
    - Remaining useful life
    - Health score boundaries
    - Health category classification
    - Batch processing
    - Invalid telemetry
    - Missing features
    - Deterministic scoring
    - Input immutability
    - Financial/safety risk indicators

IMPORTANT:
Connect HealthScoreAdapter to the production
health scoring implementation.

Expected thresholds and scoring direction are
illustrative. Calibrate them against the real
asset type, model, and engineering requirements.

No physical equipment is controlled by these tests.
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
class AssetTelemetry:
    temperature: float
    vibration: float
    pressure: float
    current: float
    operating_hours: float
    anomaly_score: float = 0.0
    failure_probability: float = 0.0
    rul_hours: float | None = None


@dataclass(frozen=True)
class HealthScoreResult:
    health_score: float
    health_status: str
    risk_level: str
    maintenance_required: bool
    confidence: float | None = None


# ==================================================
# Synthetic test readings
# ==================================================

HEALTHY_ASSET = AssetTelemetry(
    temperature=45.0,
    vibration=2.0,
    pressure=4.0,
    current=12.0,
    operating_hours=1000.0,
    anomaly_score=0.05,
    failure_probability=0.02,
    rul_hours=12000.0,
)

DEGRADED_ASSET = AssetTelemetry(
    temperature=85.0,
    vibration=10.0,
    pressure=8.0,
    current=25.0,
    operating_hours=9000.0,
    anomaly_score=0.55,
    failure_probability=0.45,
    rul_hours=2000.0,
)

CRITICAL_ASSET = AssetTelemetry(
    temperature=140.0,
    vibration=25.0,
    pressure=18.0,
    current=50.0,
    operating_hours=18000.0,
    anomaly_score=0.95,
    failure_probability=0.92,
    rul_hours=100.0,
)


# ==================================================
# Production interface
# ==================================================

class HealthScoreProtocol(Protocol):

    def calculate(
        self,
        telemetry: AssetTelemetry,
    ) -> HealthScoreResult:
        ...

    def calculate_batch(
        self,
        readings: list[AssetTelemetry],
    ) -> list[HealthScoreResult]:
        ...


# ==================================================
# Production adapter
# ==================================================

class HealthScoreAdapter:
    """
    Connect to your actual health scoring engine.

    Possible module location:

        app/ml/predictive_maintenance/health_score.py

    Example:

        from app.ml.predictive_maintenance.\
health_score import HealthScoreEngine

        self.engine = HealthScoreEngine(...)

    Adapt the production input/output schemas
    to the test models above.

    Do not implement a second scoring algorithm
    here; test the production algorithm itself.
    """

    def __init__(self) -> None:
        self.engine: Any = None

        # TODO:
        # Initialize your actual health scoring
        # engine and load required model artifacts.

    def calculate(
        self,
        telemetry: AssetTelemetry,
    ) -> HealthScoreResult:

        if self.engine is None:
            raise RuntimeError(
                "Health score engine is not configured"
            )

        # Example:
        #
        # raw = self.engine.calculate(
        #     temperature=telemetry.temperature,
        #     vibration=telemetry.vibration,
        #     pressure=telemetry.pressure,
        #     current=telemetry.current,
        #     operating_hours=telemetry.operating_hours,
        #     anomaly_score=telemetry.anomaly_score,
        #     failure_probability=(
        #         telemetry.failure_probability
        #     ),
        #     rul_hours=telemetry.rul_hours,
        # )
        #
        # return HealthScoreResult(
        #     health_score=float(raw["health_score"]),
        #     health_status=str(raw["health_status"]),
        #     risk_level=str(raw["risk_level"]),
        #     maintenance_required=bool(
        #         raw["maintenance_required"]
        #     ),
        #     confidence=raw.get("confidence"),
        # )

        raise NotImplementedError(
            "Map the production health score result"
        )

    def calculate_batch(
        self,
        readings: list[AssetTelemetry],
    ) -> list[HealthScoreResult]:

        return [
            self.calculate(reading)
            for reading in readings
        ]


# ==================================================
# Fixtures
# ==================================================

@pytest.fixture
def engine() -> HealthScoreProtocol:

    adapter = HealthScoreAdapter()

    if adapter.engine is None:
        pytest.skip(
            "Connect HealthScoreAdapter to "
            "your production health scoring engine."
        )

    return adapter


@pytest.fixture
def healthy_asset() -> AssetTelemetry:
    return HEALTHY_ASSET


@pytest.fixture
def degraded_asset() -> AssetTelemetry:
    return DEGRADED_ASSET


@pytest.fixture
def critical_asset() -> AssetTelemetry:
    return CRITICAL_ASSET


# ==================================================
# Basic health scoring
# ==================================================

class TestBasicHealthScore:

    def test_healthy_asset_score(
        self,
        engine: HealthScoreProtocol,
        healthy_asset: AssetTelemetry,
    ) -> None:

        result = engine.calculate(healthy_asset)

        assert isinstance(
            result,
            HealthScoreResult,
        )

        assert 0 <= result.health_score <= 100

    def test_degraded_asset_score(
        self,
        engine: HealthScoreProtocol,
        degraded_asset: AssetTelemetry,
    ) -> None:

        result = engine.calculate(degraded_asset)

        assert 0 <= result.health_score <= 100

    def test_critical_asset_score(
        self,
        engine: HealthScoreProtocol,
        critical_asset: AssetTelemetry,
    ) -> None:

        result = engine.calculate(critical_asset)

        assert 0 <= result.health_score <= 100

    def test_health_score_ordering(
        self,
        engine: HealthScoreProtocol,
    ) -> None:

        healthy = engine.calculate(
            HEALTHY_ASSET
        )

        degraded = engine.calculate(
            DEGRADED_ASSET
        )

        critical = engine.calculate(
            CRITICAL_ASSET
        )

        # Assumes 100 means healthiest.
        assert (
            healthy.health_score
            >= degraded.health_score
            >= critical.health_score
        )


# ==================================================
# Health status classification
# ==================================================

class TestHealthStatus:

    @pytest.mark.parametrize(
        "asset",
        [
            HEALTHY_ASSET,
            DEGRADED_ASSET,
            CRITICAL_ASSET,
        ],
    )
    def test_health_status_valid(
        self,
        engine: HealthScoreProtocol,
        asset: AssetTelemetry,
    ) -> None:

        result = engine.calculate(asset)

        assert result.health_status.lower() in {
            "excellent",
            "good",
            "healthy",
            "fair",
            "degraded",
            "poor",
            "critical",
            "failed",
        }

    @pytest.mark.parametrize(
        "asset",
        [
            HEALTHY_ASSET,
            DEGRADED_ASSET,
            CRITICAL_ASSET,
        ],
    )
    def test_risk_level_valid(
        self,
        engine: HealthScoreProtocol,
        asset: AssetTelemetry,
    ) -> None:

        result = engine.calculate(asset)

        assert result.risk_level.lower() in {
            "low",
            "medium",
            "high",
            "critical",
        }

    def test_maintenance_required_is_boolean(
        self,
        engine: HealthScoreProtocol,
    ) -> None:

        result = engine.calculate(
            DEGRADED_ASSET
        )

        assert isinstance(
            result.maintenance_required,
            bool,
        )

    def test_critical_asset_requires_maintenance(
        self,
        engine: HealthScoreProtocol,
    ) -> None:

        result = engine.calculate(
            CRITICAL_ASSET
        )

        assert result.maintenance_required is True


# ==================================================
# Temperature sensitivity
# ==================================================

class TestTemperatureSensitivity:

    def test_high_temperature_reduces_health(
        self,
        engine: HealthScoreProtocol,
    ) -> None:

        normal = HEALTHY_ASSET

        hot = AssetTelemetry(
            **{
                **normal.__dict__,
                "temperature": 130.0,
            }
        )

        normal_score = engine.calculate(
            normal
        ).health_score

        hot_score = engine.calculate(
            hot
        ).health_score

        assert hot_score <= normal_score


# ==================================================
# Vibration sensitivity
# ==================================================

class TestVibrationSensitivity:

    def test_high_vibration_reduces_health(
        self,
        engine: HealthScoreProtocol,
    ) -> None:

        normal = HEALTHY_ASSET

        vibrating = AssetTelemetry(
            **{
                **normal.__dict__,
                "vibration": 25.0,
            }
        )

        normal_score = engine.calculate(
            normal
        ).health_score

        vibration_score = engine.calculate(
            vibrating
        ).health_score

        assert vibration_score <= normal_score


# ==================================================
# Electrical current
# ==================================================

class TestCurrentSensitivity:

    def test_high_current_reduces_health(
        self,
        engine: HealthScoreProtocol,
    ) -> None:

        normal = HEALTHY_ASSET

        overloaded = AssetTelemetry(
            **{
                **normal.__dict__,
                "current": 50.0,
            }
        )

        normal_score = engine.calculate(
            normal
        ).health_score

        overload_score = engine.calculate(
            overloaded
        ).health_score

        assert overload_score <= normal_score


# ==================================================
# Anomaly and failure risk
# ==================================================

class TestAnomalyAndFailureRisk:

    def test_high_anomaly_score_reduces_health(
        self,
        engine: HealthScoreProtocol,
    ) -> None:

        normal = HEALTHY_ASSET

        anomalous = AssetTelemetry(
            **{
                **normal.__dict__,
                "anomaly_score": 0.95,
            }
        )

        normal_score = engine.calculate(
            normal
        ).health_score

        anomaly_score = engine.calculate(
            anomalous
        ).health_score

        assert anomaly_score <= normal_score

    def test_high_failure_probability_reduces_health(
        self,
        engine: HealthScoreProtocol,
    ) -> None:

        normal = HEALTHY_ASSET

        risky = AssetTelemetry(
            **{
                **normal.__dict__,
                "failure_probability": 0.95,
            }
        )

        normal_score = engine.calculate(
            normal
        ).health_score

        risky_score = engine.calculate(
            risky
        ).health_score

        assert risky_score <= normal_score


# ==================================================
# Remaining useful life
# ==================================================

class TestRemainingUsefulLife:

    def test_low_rul_reduces_health(
        self,
        engine: HealthScoreProtocol,
    ) -> None:

        normal = HEALTHY_ASSET

        low_rul = AssetTelemetry(
            **{
                **normal.__dict__,
                "rul_hours": 50.0,
            }
        )

        normal_score = engine.calculate(
            normal
        ).health_score

        low_rul_score = engine.calculate(
            low_rul
        ).health_score

        assert low_rul_score <= normal_score

    def test_missing_rul_supported(
        self,
        engine: HealthScoreProtocol,
    ) -> None:

        data = AssetTelemetry(
            **{
                **HEALTHY_ASSET.__dict__,
                "rul_hours": None,
            }
        )

        result = engine.calculate(data)

        assert 0 <= result.health_score <= 100


# ==================================================
# Confidence
# ==================================================

class TestHealthScoreConfidence:

    def test_confidence_range(
        self,
        engine: HealthScoreProtocol,
    ) -> None:

        result = engine.calculate(
            HEALTHY_ASSET
        )

        if result.confidence is not None:
            assert math.isfinite(
                result.confidence
            )

            assert 0 <= result.confidence <= 1


# ==================================================
# Batch processing
# ==================================================

class TestBatchHealthScore:

    def test_batch_result_count(
        self,
        engine: HealthScoreProtocol,
    ) -> None:

        assets = [
            HEALTHY_ASSET,
            DEGRADED_ASSET,
            CRITICAL_ASSET,
        ]

        results = engine.calculate_batch(assets)

        assert len(results) == 3

    def test_batch_result_types(
        self,
        engine: HealthScoreProtocol,
    ) -> None:

        results = engine.calculate_batch([
            HEALTHY_ASSET,
            DEGRADED_ASSET,
        ])

        assert all(
            isinstance(result, HealthScoreResult)
            for result in results
        )

    def test_empty_batch(
        self,
        engine: HealthScoreProtocol,
    ) -> None:

        assert engine.calculate_batch([]) == []

    def test_batch_matches_individual(
        self,
        engine: HealthScoreProtocol,
    ) -> None:

        assets = [
            HEALTHY_ASSET,
            DEGRADED_ASSET,
            CRITICAL_ASSET,
        ]

        batch = engine.calculate_batch(assets)

        individual = [
            engine.calculate(asset)
            for asset in assets
        ]

        for batch_result, single_result in zip(
            batch,
            individual,
        ):
            assert batch_result.health_score == (
                pytest.approx(
                    single_result.health_score
                )
            )

    def test_large_batch(
        self,
        engine: HealthScoreProtocol,
    ) -> None:

        assets = [HEALTHY_ASSET] * 100

        results = engine.calculate_batch(assets)

        assert len(results) == 100


# ==================================================
# Input validation
# ==================================================

class TestHealthScoreValidation:

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
        engine: HealthScoreProtocol,
        invalid_value: float,
    ) -> None:

        data = AssetTelemetry(
            **{
                **HEALTHY_ASSET.__dict__,
                "temperature": invalid_value,
            }
        )

        with pytest.raises(
            (ValueError, TypeError)
        ):
            engine.calculate(data)

    def test_negative_operating_hours_rejected(
        self,
        engine: HealthScoreProtocol,
    ) -> None:

        data = AssetTelemetry(
            **{
                **HEALTHY_ASSET.__dict__,
                "operating_hours": -100.0,
            }
        )

        with pytest.raises(ValueError):
            engine.calculate(data)

    @pytest.mark.parametrize(
        "probability",
        [
            -0.1,
            1.1,
            2.0,
        ],
    )
    def test_invalid_failure_probability_rejected(
        self,
        engine: HealthScoreProtocol,
        probability: float,
    ) -> None:

        data = AssetTelemetry(
            **{
                **HEALTHY_ASSET.__dict__,
                "failure_probability": probability,
            }
        )

        with pytest.raises(ValueError):
            engine.calculate(data)

    @pytest.mark.parametrize(
        "anomaly",
        [
            -0.1,
            1.1,
        ],
    )
    def test_invalid_anomaly_score_rejected(
        self,
        engine: HealthScoreProtocol,
        anomaly: float,
    ) -> None:

        data = AssetTelemetry(
            **{
                **HEALTHY_ASSET.__dict__,
                "anomaly_score": anomaly,
            }
        )

        with pytest.raises(ValueError):
            engine.calculate(data)

    def test_missing_telemetry_rejected(
        self,
        engine: HealthScoreProtocol,
    ) -> None:

        incomplete = {
            "temperature": 45.0,
            "vibration": 2.0,
        }

        with pytest.raises(
            (ValueError, TypeError, KeyError)
        ):
            engine.calculate(
                incomplete  # type: ignore[arg-type]
            )


# ==================================================
# Data integrity
# ==================================================

class TestHealthScoreIntegrity:

    def test_input_not_mutated(
        self,
        engine: HealthScoreProtocol,
    ) -> None:

        telemetry = HEALTHY_ASSET
        original = deepcopy(telemetry)

        engine.calculate(telemetry)

        assert telemetry == original

    def test_deterministic_score(
        self,
        engine: HealthScoreProtocol,
    ) -> None:

        scores = [
            engine.calculate(
                HEALTHY_ASSET
            ).health_score
            for _ in range(5)
        ]

        for score in scores:
            assert score == pytest.approx(
                scores[0]
            )

    def test_health_score_is_finite(
        self,
        engine: HealthScoreProtocol,
    ) -> None:

        for asset in [
            HEALTHY_ASSET,
            DEGRADED_ASSET,
            CRITICAL_ASSET,
        ]:
            result = engine.calculate(asset)

            assert math.isfinite(
                result.health_score
            )

    def test_health_score_range(
        self,
        engine: HealthScoreProtocol,
    ) -> None:

        for asset in [
            HEALTHY_ASSET,
            DEGRADED_ASSET,
            CRITICAL_ASSET,
        ]:
            result = engine.calculate(asset)

            assert (
                0.0
                <= result.health_score
                <= 100.0
            )
