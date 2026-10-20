
"""
app/services/health_score_service.py

Industrial Predictive Maintenance, Safety & Emergency AI

Machine / Asset Health Score Service

Features:
- Equipment health score (0-100)
- Telemetry condition scoring
- Anomaly risk integration
- Failure prediction integration
- Remaining Useful Life integration
- Inspection condition integration
- Configurable scoring weights
- Health classification
- Contributing factors
- Maintenance recommendations
- Batch asset health analysis
- Optional asset repository updates

IMPORTANT:
Health scores are decision-support indicators.
They are not certified industrial safety measures.
"""

from __future__ import annotations

import inspect
import logging
import math

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4


logger = logging.getLogger(__name__)


class HealthScoreError(Exception):
    pass


class HealthValidationError(HealthScoreError):
    pass


class HealthStatus(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def finite_number(
    value: Any,
    field_name: str,
) -> float:

    if isinstance(value, bool):
        raise HealthValidationError(
            f"{field_name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HealthValidationError(
            f"Invalid {field_name}"
        ) from exc

    if not math.isfinite(result):
        raise HealthValidationError(
            f"{field_name} must be finite"
        )

    return result


def validate_score(
    value: Any,
    field_name: str,
) -> float:

    score = finite_number(
        value,
        field_name,
    )

    if not 0 <= score <= 100:
        raise HealthValidationError(
            f"{field_name} must be between 0 and 100"
        )

    return score


@dataclass(frozen=True)
class SensorRange:
    minimum: float
    maximum: float
    warning_minimum: Optional[float] = None
    warning_maximum: Optional[float] = None


@dataclass
class HealthScoreResult:
    assessment_id: str
    asset_id: str
    health_score: float
    health_status: str
    risk_score: float
    component_scores: dict[str, float]
    contributing_factors: list[str]
    recommendations: list[str]
    assessed_at: str
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HealthScoreService:
    """
    Weighted equipment health scoring service.

    Optional asset repository contract:

    async update(asset_id, data)

    Component inputs:
        telemetry: raw sensor readings
        anomaly_risk_score: 0-100
        failure_probability: 0-1
        rul_hours: remaining useful life
        expected_life_hours: expected operating life
        inspection_score: 0-100

    All reference thresholds must be calibrated
    for the specific machine and operating mode.
    """

    DEFAULT_WEIGHTS = {
        "telemetry": 0.25,
        "anomaly": 0.20,
        "failure": 0.25,
        "rul": 0.20,
        "inspection": 0.10,
    }

    # Demonstration limits only.
    DEFAULT_SENSOR_RANGES = {
        "temperature": SensorRange(
            minimum=0,
            maximum=100,
            warning_maximum=80,
        ),
        "vibration": SensorRange(
            minimum=0,
            maximum=12,
            warning_maximum=7,
        ),
        "pressure": SensorRange(
            minimum=0,
            maximum=250,
            warning_maximum=200,
        ),
        "current": SensorRange(
            minimum=0,
            maximum=100,
            warning_maximum=80,
        ),
        "voltage": SensorRange(
            minimum=180,
            maximum=260,
            warning_minimum=200,
            warning_maximum=240,
        ),
    }

    def __init__(
        self,
        asset_repository: Any = None,
        weights: Optional[
            Mapping[str, float]
        ] = None,
        sensor_ranges: Optional[
            Mapping[str, SensorRange]
        ] = None,
    ):

        self.repository = asset_repository

        self.weights = dict(
            weights or self.DEFAULT_WEIGHTS
        )

        self.sensor_ranges = dict(
            sensor_ranges
            if sensor_ranges is not None
            else self.DEFAULT_SENSOR_RANGES
        )

        expected = set(self.DEFAULT_WEIGHTS)

        if set(self.weights) != expected:
            raise HealthValidationError(
                "Weights must contain: "
                + ", ".join(sorted(expected))
            )

        for key, value in self.weights.items():
            weight = finite_number(
                value,
                f"weight_{key}",
            )

            if weight < 0:
                raise HealthValidationError(
                    "Weights cannot be negative"
                )

            self.weights[key] = weight

        total = sum(self.weights.values())

        if not math.isclose(
            total, 1.0, abs_tol=1e-6
        ):
            raise HealthValidationError(
                "Weights must sum to 1.0"
            )

    @staticmethod
    def classify_health(
        score: float,
    ) -> HealthStatus:

        score = validate_score(
            score, "health_score"
        )

        if score >= 90:
            return HealthStatus.EXCELLENT

        if score >= 75:
            return HealthStatus.GOOD

        if score >= 55:
            return HealthStatus.FAIR

        if score >= 30:
            return HealthStatus.POOR

        return HealthStatus.CRITICAL

    def calculate_sensor_score(
        self,
        sensor_name: str,
        value: float,
    ) -> float:

        reading = finite_number(
            value,
            sensor_name,
        )

        sensor_range = self.sensor_ranges.get(
            sensor_name
        )

        if sensor_range is None:
            raise HealthValidationError(
                f"No range configured for {sensor_name}"
            )

        minimum = sensor_range.minimum
        maximum = sensor_range.maximum

        if minimum >= maximum:
            raise HealthValidationError(
                "Invalid sensor range"
            )

        if reading < minimum:
            deviation = (
                minimum - reading
            ) / (maximum - minimum)

            return round(
                max(0, 30 - deviation * 100),
                2,
            )

        if reading > maximum:
            deviation = (
                reading - maximum
            ) / (maximum - minimum)

            return round(
                max(0, 30 - deviation * 100),
                2,
            )

        warning_min = (
            sensor_range.warning_minimum
        )
        warning_max = (
            sensor_range.warning_maximum
        )

        if (
            warning_min is not None
            and reading < warning_min
        ):
            span = max(
                warning_min - minimum,
                1e-9,
            )

            return round(
                60 + 40 * (
                    (reading - minimum) / span
                ),
                2,
            )

        if (
            warning_max is not None
            and reading > warning_max
        ):
            span = max(
                maximum - warning_max,
                1e-9,
            )

            return round(
                60 + 40 * (
                    (maximum - reading) / span
                ),
                2,
            )

        return 100.0

    def calculate_telemetry_score(
        self,
        telemetry: Mapping[str, Any],
    ) -> tuple[float, list[str]]:

        if not isinstance(telemetry, Mapping):
            raise HealthValidationError(
                "Telemetry must be a mapping"
            )

        scores = []
        factors = []

        for name, value in telemetry.items():

            if name not in self.sensor_ranges:
                continue

            score = self.calculate_sensor_score(
                name,
                value,
            )

            scores.append(score)

            if score < 75:
                factors.append(
                    f"Abnormal {name} reading: {value}"
                )

        if not scores:
            raise HealthValidationError(
                "No supported telemetry sensors"
            )

        return (
            round(
                sum(scores) / len(scores),
                2,
            ),
            factors,
        )

    @staticmethod
    def calculate_failure_score(
        failure_probability: float,
    ) -> float:

        probability = finite_number(
            failure_probability,
            "failure_probability",
        )

        if not 0 <= probability <= 1:
            raise HealthValidationError(
                "Failure probability must be 0-1"
            )

        return round(
            (1 - probability) * 100,
            2,
        )

    @staticmethod
    def calculate_rul_score(
        rul_hours: float,
        expected_life_hours: float,
    ) -> float:

        remaining = finite_number(
            rul_hours, "rul_hours"
        )

        expected = finite_number(
            expected_life_hours,
            "expected_life_hours",
        )

        if remaining < 0 or expected <= 0:
            raise HealthValidationError(
                "Invalid RUL or expected life"
            )

        ratio = min(
            remaining / expected,
            1.0,
        )

        return round(
            ratio * 100,
            2,
        )

    @staticmethod
    def calculate_anomaly_score(
        anomaly_risk_score: float,
    ) -> float:

        risk = validate_score(
            anomaly_risk_score,
            "anomaly_risk_score",
        )

        return round(
            100 - risk,
            2,
        )

    def calculate_health_score(
        self,
        *,
        asset_id: str,
        telemetry: Optional[
            Mapping[str, Any]
        ] = None,
        anomaly_risk_score: Optional[float] = None,
        failure_probability: Optional[float] = None,
        rul_hours: Optional[float] = None,
        expected_life_hours: Optional[float] = None,
        inspection_score: Optional[float] = None,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> HealthScoreResult:

        if not isinstance(asset_id, str) or not (
            asset_id.strip()
        ):
            raise HealthValidationError(
                "asset_id is required"
            )

        components = {}
        factors = []

        if telemetry is not None:
            score, sensor_factors = (
                self.calculate_telemetry_score(
                    telemetry
                )
            )

            components["telemetry"] = score
            factors.extend(sensor_factors)

        if anomaly_risk_score is not None:
            score = self.calculate_anomaly_score(
                anomaly_risk_score
            )

            components["anomaly"] = score

            if score < 60:
                factors.append(
                    "High anomaly risk detected"
                )

        if failure_probability is not None:
            score = self.calculate_failure_score(
                failure_probability
            )

            components["failure"] = score

            if score < 60:
                factors.append(
                    "Elevated predicted failure risk"
                )

        if rul_hours is not None:
            if expected_life_hours is None:
                raise HealthValidationError(
                    "expected_life_hours is required "
                    "when rul_hours is provided"
                )

            score = self.calculate_rul_score(
                rul_hours,
                expected_life_hours,
            )

            components["rul"] = score

            if score < 40:
                factors.append(
                    "Low remaining useful life"
                )

        if inspection_score is not None:
            score = validate_score(
                inspection_score,
                "inspection_score",
            )

            components["inspection"] = score

            if score < 60:
                factors.append(
                    "Poor inspection condition"
                )

        if not components:
            raise HealthValidationError(
                "At least one health component "
                "must be supplied"
            )

        # Renormalize weights across available signals.
        # Missing components are not treated as healthy.
        available_weight = sum(
            self.weights[name]
            for name in components
        )

        if available_weight <= 0:
            raise HealthValidationError(
                "Available component weights "
                "must be positive"
            )

        health_score = sum(
            components[name] * self.weights[name]
            for name in components
        ) / available_weight

        health_score = round(
            max(0, min(100, health_score)),
            2,
        )

        risk_score = round(
            100 - health_score,
            2,
        )

        status = self.classify_health(
            health_score
        )

        recommendations = (
            self.generate_recommendations(
                status,
                factors,
            )
        )

        return HealthScoreResult(
            assessment_id=str(uuid4()),
            asset_id=asset_id,
            health_score=health_score,
            health_status=status.value,
            risk_score=risk_score,
            component_scores=components,
            contributing_factors=factors,
            recommendations=recommendations,
            assessed_at=utc_now(),
            metadata={
                **dict(metadata or {}),
                "available_components": list(
                    components
                ),
                "missing_components": [
                    name
                    for name in self.DEFAULT_WEIGHTS
                    if name not in components
                ],
                "effective_weight_sum": round(
                    available_weight,
                    4,
                ),
            },
        )

    @staticmethod
    def generate_recommendations(
        status: HealthStatus,
        factors: Sequence[str],
    ) -> list[str]:

        recommendations = []

        if status == HealthStatus.CRITICAL:
            recommendations.append(
                "Request urgent assessment by "
                "authorized maintenance personnel."
            )

        elif status == HealthStatus.POOR:
            recommendations.append(
                "Prioritize maintenance inspection "
                "and investigate abnormal signals."
            )

        elif status == HealthStatus.FAIR:
            recommendations.append(
                "Increase monitoring frequency "
                "and review maintenance planning."
            )

        elif status == HealthStatus.GOOD:
            recommendations.append(
                "Continue scheduled preventive "
                "maintenance and monitoring."
            )

        else:
            recommendations.append(
                "Maintain routine monitoring "
                "and equipment inspections."
            )

        if factors:
            recommendations.append(
                "Review contributing factors "
                "against manufacturer specifications."
            )

        recommendations.append(
            "Do not use this score as the sole "
            "basis for safety-critical decisions."
        )

        return recommendations

    def calculate_batch(
        self,
        assets: Sequence[
            Mapping[str, Any]
        ],
    ) -> list[HealthScoreResult]:

        results = []

        for asset in assets:
            results.append(
                self.calculate_health_score(
                    **asset
                )
            )

        return results

    async def update_asset_health(
        self,
        result: HealthScoreResult,
    ) -> Any:

        if self.repository is None:
            raise HealthScoreError(
                "Asset repository not configured"
            )

        method = getattr(
            self.repository,
            "update",
            None,
        )

        if method is None:
            raise HealthScoreError(
                "Repository must implement update"
            )

        update_data = {
            "health_score": result.health_score,
            "risk_level": (
                "critical"
                if result.health_score < 25
                else "high"
                if result.health_score < 50
                else "medium"
                if result.health_score < 75
                else "low"
            ),
            "updated_at": datetime.now(
                timezone.utc
            ),
        }

        response = method(
            result.asset_id,
            update_data,
        )

        if inspect.isawaitable(response):
            response = await response

        logger.info(
            "Updated asset health asset=%s score=%.2f",
            result.asset_id,
            result.health_score,
        )

        return response


if __name__ == "__main__":

    import json

    service = HealthScoreService()

    result = service.calculate_health_score(
        asset_id="MACHINE-001",
        telemetry={
            "temperature": 92,
            "vibration": 9.5,
            "pressure": 185,
            "current": 75,
            "voltage": 230,
        },
        anomaly_risk_score=70,
        failure_probability=0.65,
        rul_hours=400,
        expected_life_hours=5000,
        inspection_score=65,
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
        )
    )
