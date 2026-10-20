
"""
app/services/failure_prediction_service.py

Industrial Predictive Maintenance, Safety & Emergency AI

Failure Prediction Service

Features:
- Equipment failure probability prediction
- Real-time telemetry analysis
- Historical feature extraction
- ML model integration
- Model registry support
- Failure risk classification
- Prediction horizon tracking
- Batch asset predictions
- Predictive maintenance recommendations
- Optional repository persistence
- Async model execution
- Structured API-ready responses

IMPORTANT:
The built-in heuristic is for demonstrations only.
Production failure probabilities require a trained,
validated and calibrated predictive model.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from statistics import mean, pstdev
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4


logger = logging.getLogger(__name__)


# ============================================================
# ENUMS
# ============================================================

class FailureRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PredictionSource(str, Enum):
    ML_MODEL = "ml_model"
    HEURISTIC = "heuristic_demo"


# ============================================================
# EXCEPTIONS
# ============================================================

class FailurePredictionError(Exception):
    pass


class PredictionValidationError(FailurePredictionError):
    pass


class ModelUnavailableError(FailurePredictionError):
    pass


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class FailurePredictionResult:
    prediction_id: str
    asset_id: str
    failure_probability: float
    risk_level: str
    prediction_horizon_hours: int
    model_source: str
    model_version: str
    predicted_at: str

    features: dict[str, float] = field(
        default_factory=dict
    )

    contributing_factors: list[str] = field(
        default_factory=list
    )

    recommendations: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def finite_number(
    value: Any,
    name: str,
) -> float:

    if isinstance(value, bool):
        raise PredictionValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PredictionValidationError(
            f"Invalid {name}"
        ) from exc

    if not math.isfinite(result):
        raise PredictionValidationError(
            f"{name} must be finite"
        )

    return result


def probability(value: Any) -> float:
    result = finite_number(
        value,
        "failure_probability",
    )

    if not 0 <= result <= 1:
        raise PredictionValidationError(
            "Probability must be between 0 and 1"
        )

    return result


# ============================================================
# FAILURE PREDICTION SERVICE
# ============================================================

class FailurePredictionService:
    """
    Failure prediction application service.

    Supported model interface:

        model.predict_proba(X)

    where X is a two-dimensional feature matrix.

    The positive failure class must be identified
    through model.classes_ when available.

    Optional repository interface:

        async create_prediction(data: dict)
        async get_predictions(asset_id: str)

    A model registry can supply models externally.
    """

    FEATURE_ORDER = (
        "temperature",
        "vibration",
        "pressure",
        "current",
        "voltage",
        "temperature_mean",
        "temperature_std",
        "vibration_mean",
        "vibration_std",
        "pressure_mean",
        "current_mean",
    )

    # Demonstration reference limits.
    # Replace with per-equipment specifications.
    DEMO_LIMITS = {
        "temperature": (20.0, 80.0),
        "vibration": (0.0, 7.0),
        "pressure": (0.0, 200.0),
        "current": (0.0, 80.0),
        "voltage": (200.0, 240.0),
    }

    def __init__(
        self,
        prediction_repository: Any = None,
        model: Any = None,
        model_registry: Any = None,
        model_name: str = "failure_prediction",
        model_version: str = "unversioned",
        prediction_horizon_hours: int = 24,
        allow_demo_heuristic: bool = False,
    ):

        if (
            isinstance(prediction_horizon_hours, bool)
            or not isinstance(
                prediction_horizon_hours, int
            )
            or prediction_horizon_hours < 1
        ):
            raise PredictionValidationError(
                "Prediction horizon must be "
                "a positive integer"
            )

        self.repository = prediction_repository
        self.model = model
        self.model_registry = model_registry
        self.model_name = model_name
        self.model_version = model_version
        self.prediction_horizon_hours = (
            prediction_horizon_hours
        )
        self.allow_demo_heuristic = (
            allow_demo_heuristic
        )

    # --------------------------------------------------------
    # TELEMETRY VALIDATION
    # --------------------------------------------------------

    def validate_telemetry(
        self,
        telemetry: Mapping[str, Any],
    ) -> dict[str, float]:

        if not isinstance(telemetry, Mapping):
            raise PredictionValidationError(
                "Telemetry must be a mapping"
            )

        if not telemetry:
            raise PredictionValidationError(
                "Telemetry cannot be empty"
            )

        cleaned = {}

        for name, value in telemetry.items():
            cleaned[str(name)] = finite_number(
                value,
                str(name),
            )

        return cleaned

    # --------------------------------------------------------
    # HISTORICAL FEATURE EXTRACTION
    # --------------------------------------------------------

    def extract_features(
        self,
        telemetry: Mapping[str, Any],
        history: Optional[
            Sequence[Mapping[str, Any]]
        ] = None,
    ) -> dict[str, float]:

        current = self.validate_telemetry(
            telemetry
        )

        historical = list(history or [])

        features = {}

        for name in (
            "temperature",
            "vibration",
            "pressure",
            "current",
            "voltage",
        ):
            features[name] = current.get(
                name, 0.0
            )

        for name in (
            "temperature",
            "vibration",
            "pressure",
            "current",
        ):

            values = []

            for record in historical:
                if name in record:
                    values.append(
                        finite_number(
                            record[name],
                            name,
                        )
                    )

            if name in current:
                values.append(current[name])

            features[f"{name}_mean"] = (
                mean(values) if values else 0.0
            )

            if name in (
                "temperature",
                "vibration",
            ):
                features[f"{name}_std"] = (
                    pstdev(values)
                    if len(values) >= 2
                    else 0.0
                )

        return {
            name: round(
                features.get(name, 0.0),
                6,
            )
            for name in self.FEATURE_ORDER
        }

    # --------------------------------------------------------
    # MODEL RESOLUTION
    # --------------------------------------------------------

    async def _resolve_model(self) -> Any:

        if self.model is not None:
            return self.model

        if self.model_registry is None:
            return None

        getter = getattr(
            self.model_registry,
            "get_model",
            None,
        )

        if getter is None:
            raise ModelUnavailableError(
                "Model registry must implement get_model"
            )

        if inspect.iscoroutinefunction(getter):
            resolved = await getter(
                self.model_name
            )
        else:
            resolved = await asyncio.to_thread(
                getter,
                self.model_name,
            )

        if resolved is not None:
            self.model = resolved

        return resolved

    # --------------------------------------------------------
    # MACHINE LEARNING PREDICTION
    # --------------------------------------------------------

    @staticmethod
    def _predict_model_sync(
        model: Any,
        feature_vector: list[float],
    ) -> float:

        if not hasattr(model, "predict_proba"):
            raise ModelUnavailableError(
                "Model must expose predict_proba"
            )

        matrix = [feature_vector]

        probabilities = model.predict_proba(
            matrix
        )

        if len(probabilities) != 1:
            raise FailurePredictionError(
                "Expected one prediction row"
            )

        row = probabilities[0]

        classes = getattr(
            model,
            "classes_",
            None,
        )

        if classes is not None:
            class_values = list(classes)

            positive_indices = [
                index
                for index, label in enumerate(
                    class_values
                )
                if label in (1, True, "failure")
            ]

            if len(positive_indices) != 1:
                raise FailurePredictionError(
                    "Cannot identify positive "
                    "failure class"
                )

            index = positive_indices[0]

        else:
            raise FailurePredictionError(
                "Model classes_ is required "
                "to identify failure probability"
            )

        return probability(row[index])

    async def predict_with_model(
        self,
        features: Mapping[str, float],
    ) -> float:

        model = await self._resolve_model()

        if model is None:
            raise ModelUnavailableError(
                "Failure prediction model unavailable"
            )

        feature_vector = [
            features[name]
            for name in self.FEATURE_ORDER
        ]

        return await asyncio.to_thread(
            self._predict_model_sync,
            model,
            feature_vector,
        )

    # --------------------------------------------------------
    # DEMONSTRATION HEURISTIC
    # --------------------------------------------------------

    def demo_heuristic_prediction(
        self,
        telemetry: Mapping[str, float],
    ) -> float:
        """
        Demo risk indicator, NOT a calibrated
        failure probability.

        Used only when allow_demo_heuristic=True.
        """

        score = 0.0
        checked = 0

        for sensor, (minimum, maximum) in (
            self.DEMO_LIMITS.items()
        ):
            if sensor not in telemetry:
                continue

            checked += 1
            value = telemetry[sensor]

            if value < minimum:
                deviation = (
                    minimum - value
                ) / max(abs(minimum), 1.0)

            elif value > maximum:
                deviation = (
                    value - maximum
                ) / max(abs(maximum), 1.0)

            else:
                deviation = 0.0

            score += min(
                deviation * 3.0,
                1.0,
            )

        if checked == 0:
            raise PredictionValidationError(
                "No supported sensors available "
                "for demo heuristic"
            )

        return round(
            min(score / checked, 1.0),
            6,
        )

    # --------------------------------------------------------
    # RISK CLASSIFICATION
    # --------------------------------------------------------

    @staticmethod
    def classify_risk(
        score: float,
    ) -> FailureRisk:

        value = probability(score)

        # Illustrative thresholds.
        if value >= 0.80:
            return FailureRisk.CRITICAL

        if value >= 0.60:
            return FailureRisk.HIGH

        if value >= 0.30:
            return FailureRisk.MEDIUM

        return FailureRisk.LOW

    # --------------------------------------------------------
    # CONTRIBUTING FACTORS
    # --------------------------------------------------------

    def identify_contributing_factors(
        self,
        telemetry: Mapping[str, float],
    ) -> list[str]:

        factors = []

        for sensor, (minimum, maximum) in (
            self.DEMO_LIMITS.items()
        ):

            if sensor not in telemetry:
                continue

            value = telemetry[sensor]

            if value > maximum:
                factors.append(
                    f"{sensor} exceeds demo reference "
                    f"maximum ({value} > {maximum})"
                )

            elif value < minimum:
                factors.append(
                    f"{sensor} is below demo reference "
                    f"minimum ({value} < {minimum})"
                )

        return factors

    # --------------------------------------------------------
    # MAINTENANCE RECOMMENDATIONS
    # --------------------------------------------------------

    @staticmethod
    def generate_recommendations(
        risk: FailureRisk,
        factors: Sequence[str],
    ) -> list[str]:

        recommendations = []

        if risk == FailureRisk.CRITICAL:
            recommendations.append(
                "Escalate for immediate assessment "
                "by authorized maintenance and "
                "safety personnel."
            )

        elif risk == FailureRisk.HIGH:
            recommendations.append(
                "Prioritize an equipment inspection "
                "and review the maintenance schedule."
            )

        elif risk == FailureRisk.MEDIUM:
            recommendations.append(
                "Increase monitoring and investigate "
                "abnormal operating conditions."
            )

        else:
            recommendations.append(
                "Continue routine condition monitoring "
                "and scheduled maintenance."
            )

        if factors:
            recommendations.append(
                "Review sensor readings against "
                "manufacturer-approved operating limits."
            )

        recommendations.append(
            "Do not use AI predictions alone to "
            "authorize equipment shutdown or restart."
        )

        return recommendations

    # --------------------------------------------------------
    # MAIN PREDICTION
    # --------------------------------------------------------

    async def predict_failure(
        self,
        *,
        asset_id: str,
        telemetry: Mapping[str, Any],
        history: Optional[
            Sequence[Mapping[str, Any]]
        ] = None,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> FailurePredictionResult:

        if not isinstance(asset_id, str) or not (
            asset_id.strip()
        ):
            raise PredictionValidationError(
                "asset_id is required"
            )

        current = self.validate_telemetry(
            telemetry
        )

        features = self.extract_features(
            current,
            history,
        )

        model = await self._resolve_model()

        if model is not None:
            score = await self.predict_with_model(
                features
            )
            source = PredictionSource.ML_MODEL
            calibrated = False

        elif self.allow_demo_heuristic:
            score = self.demo_heuristic_prediction(
                current
            )
            source = PredictionSource.HEURISTIC
            calibrated = False

        else:
            raise ModelUnavailableError(
                "No prediction model configured. "
                "Demo fallback is disabled."
            )

        risk = self.classify_risk(score)

        factors = self.identify_contributing_factors(
            current
        )

        recommendations = (
            self.generate_recommendations(
                risk,
                factors,
            )
        )

        result = FailurePredictionResult(
            prediction_id=str(uuid4()),
            asset_id=asset_id,
            failure_probability=round(score, 6),
            risk_level=risk.value,
            prediction_horizon_hours=(
                self.prediction_horizon_hours
            ),
            model_source=source.value,
            model_version=(
                self.model_version
                if source == PredictionSource.ML_MODEL
                else "demo-heuristic-v1"
            ),
            predicted_at=utc_now(),
            features=features,
            contributing_factors=factors,
            recommendations=recommendations,
            metadata={
                **dict(metadata or {}),
                "calibrated_probability": calibrated,
                "score_interpretation": (
                    "uncalibrated_model_score"
                    if source == PredictionSource.ML_MODEL
                    else "demo_risk_indicator"
                ),
                "sensor_count": len(current),
                "historical_records": len(
                    history or []
                ),
            },
        )

        logger.info(
            "Failure prediction asset=%s "
            "source=%s score=%.4f risk=%s",
            asset_id,
            source.value,
            score,
            risk.value,
        )

        return result

    # --------------------------------------------------------
    # BATCH PREDICTION
    # --------------------------------------------------------

    async def predict_batch(
        self,
        assets: Sequence[
            Mapping[str, Any]
        ],
    ) -> list[FailurePredictionResult]:

        results = []

        for asset in assets:
            result = await self.predict_failure(
                asset_id=asset["asset_id"],
                telemetry=asset["telemetry"],
                history=asset.get("history"),
                metadata=asset.get("metadata"),
            )

            results.append(result)

        return results

    # --------------------------------------------------------
    # DATABASE PERSISTENCE
    # --------------------------------------------------------

    async def save_prediction(
        self,
        result: FailurePredictionResult,
    ) -> Any:

        if self.repository is None:
            raise FailurePredictionError(
                "Prediction repository not configured"
            )

        return await (
            self.repository.create_prediction(
                result.to_dict()
            )
        )

    async def get_prediction_history(
        self,
        asset_id: str,
    ) -> Any:

        if self.repository is None:
            raise FailurePredictionError(
                "Prediction repository not configured"
            )

        return await (
            self.repository.get_predictions(
                asset_id
            )
        )

    async def predict_and_save(
        self,
        *,
        asset_id: str,
        telemetry: Mapping[str, Any],
        history: Optional[
            Sequence[Mapping[str, Any]]
        ] = None,
    ) -> FailurePredictionResult:

        result = await self.predict_failure(
            asset_id=asset_id,
            telemetry=telemetry,
            history=history,
        )

        await self.save_prediction(result)

        return result


# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":

    import json

    async def main():

        service = FailurePredictionService(
            allow_demo_heuristic=True
        )

        result = await service.predict_failure(
            asset_id="MACHINE-001",
            telemetry={
                "temperature": 105.0,
                "vibration": 10.5,
                "pressure": 190.0,
                "current": 75.0,
                "voltage": 230.0,
            },
            history=[
                {
                    "temperature": 70.0,
                    "vibration": 4.2,
                    "pressure": 175.0,
                    "current": 65.0,
                },
                {
                    "temperature": 73.0,
                    "vibration": 4.5,
                    "pressure": 180.0,
                    "current": 68.0,
                },
            ],
        )

        print(
            json.dumps(
                result.to_dict(),
                indent=2,
            )
        )

    asyncio.run(main())
