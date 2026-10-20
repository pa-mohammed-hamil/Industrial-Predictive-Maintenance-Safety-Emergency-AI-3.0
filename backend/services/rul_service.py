
"""
app/services/rul_service.py

Industrial Predictive Maintenance, Safety & Emergency AI
Remaining Useful Life (RUL) Prediction Service

Features:
- Remaining useful life prediction
- ML model integration
- Time-series telemetry feature extraction
- Component degradation estimation
- Optional explicit heuristic fallback
- Confidence interval support
- Asset health integration
- Maintenance urgency classification
- RUL trend analysis
- Batch predictions
- Model metadata and traceability
- Optional asynchronous persistence
- Optional audit logging
- In-memory development repository

Expected ML model:
    model.predict(X)

Optional model registry:
    registry.get_model(model_name)

Optional repository:
    repository.create_rul_prediction(data)
    repository.get_rul_prediction(prediction_id)
    repository.list_rul_predictions(filters)

Important:
    RUL predictions are estimates, not guarantees.
    Models must be validated for the asset class,
    operating environment and prediction horizon.
    This service does not control industrial equipment.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
import math
import statistics

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4


logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class RULServiceError(Exception):
    """Base RUL service exception."""


class RULValidationError(RULServiceError):
    """Invalid RUL input."""


class RULModelError(RULServiceError):
    """RUL model inference failed."""


class RULPredictionNotFoundError(RULServiceError):
    """RUL prediction not found."""


class RULRepositoryError(RULServiceError):
    """RUL persistence failed."""


# ============================================================
# ENUMS
# ============================================================

class RULRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class RULPredictionSource(str, Enum):
    ML_MODEL = "ml_model"
    HEURISTIC = "heuristic"


class RULUnit(str, Enum):
    HOURS = "hours"
    DAYS = "days"
    CYCLES = "cycles"


class RULTrend(str, Enum):
    STABLE = "stable"
    IMPROVING = "improving"
    DEGRADING = "degrading"
    INSUFFICIENT_DATA = "insufficient_data"


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_text(
    value: Any,
    name: str,
    max_length: int = 255,
) -> str:

    if not isinstance(value, str):
        raise RULValidationError(
            f"{name} must be a string"
        )

    result = value.strip()

    if not result:
        raise RULValidationError(
            f"{name} cannot be empty"
        )

    if len(result) > max_length:
        raise RULValidationError(
            f"{name} exceeds {max_length} characters"
        )

    return result


def optional_text(
    value: Optional[str],
    name: str,
) -> Optional[str]:

    if value is None:
        return None

    return validate_text(value, name)


def finite_float(
    value: Any,
    name: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:

    if isinstance(value, bool):
        raise RULValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RULValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise RULValidationError(
            f"{name} must be finite"
        )

    if minimum is not None and result < minimum:
        raise RULValidationError(
            f"{name} must be >= {minimum}"
        )

    if maximum is not None and result > maximum:
        raise RULValidationError(
            f"{name} must be <= {maximum}"
        )

    return result


def validate_datetime(
    value: datetime,
    name: str,
) -> datetime:

    if not isinstance(value, datetime):
        raise RULValidationError(
            f"{name} must be datetime"
        )

    if value.tzinfo is None or value.utcoffset() is None:
        raise RULValidationError(
            f"{name} must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


def parse_enum(
    enum_class: type[Enum],
    value: Any,
    name: str,
) -> Enum:

    try:
        return enum_class(value)
    except (TypeError, ValueError) as exc:
        raise RULValidationError(
            f"Invalid {name}: {value}"
        ) from exc


def serialize(value: Any) -> Any:

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return serialize(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): serialize(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [serialize(item) for item in value]

    return value


async def invoke(
    method: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:

    result = method(*args, **kwargs)

    if inspect.isawaitable(result):
        return await result

    return result


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class RULSensorFeatures:
    temperature_mean: Optional[float] = None
    temperature_max: Optional[float] = None
    vibration_mean: Optional[float] = None
    vibration_max: Optional[float] = None
    pressure_mean: Optional[float] = None
    current_mean: Optional[float] = None
    voltage_mean: Optional[float] = None
    operating_hours: Optional[float] = None
    cycle_count: Optional[float] = None
    health_score: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass
class RULPredictionResult:
    prediction_id: str
    company_id: str
    site_id: str
    asset_id: str

    remaining_useful_life: float
    unit: str
    risk_level: str

    prediction_source: str
    model_name: Optional[str]
    model_version: Optional[str]

    lower_bound: Optional[float]
    upper_bound: Optional[float]
    confidence_level: Optional[float]

    estimated_total_life: Optional[float]
    consumed_life_percent: Optional[float]

    health_score: Optional[float]
    features: dict[str, float]

    recommendations: list[str]

    predicted_at: datetime = field(
        default_factory=utc_now
    )

    machine_id: Optional[str] = None
    component_id: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class RULTrendResult:
    asset_id: str
    trend: str
    prediction_count: int
    first_rul: Optional[float]
    latest_rul: Optional[float]
    change: Optional[float]
    slope_per_day: Optional[float]
    unit: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


# ============================================================
# RUL SERVICE
# ============================================================

class RULService:
    """
    Remaining useful life estimation service.

    ML inference requires a compatible trained model.
    Heuristic estimates are disabled by default.

    The default model feature order is illustrative.
    A production model must use its exact training
    feature schema and preprocessing pipeline.
    """

    DEFAULT_FEATURE_ORDER = (
        "temperature_mean",
        "temperature_max",
        "vibration_mean",
        "vibration_max",
        "pressure_mean",
        "current_mean",
        "voltage_mean",
        "operating_hours",
        "cycle_count",
        "health_score",
    )

    def __init__(
        self,
        rul_repository: Any = None,
        model_registry: Any = None,
        model: Any = None,
        model_name: str = "rul_prediction",
        model_version: Optional[str] = None,
        allow_heuristic_fallback: bool = False,
        feature_order: Optional[Sequence[str]] = None,
        critical_threshold_hours: float = 100.0,
        high_threshold_hours: float = 500.0,
        moderate_threshold_hours: float = 1500.0,
        audit_service: Any = None,
    ):

        self.repository = rul_repository
        self.model_registry = model_registry
        self.model = model
        self.model_name = validate_text(
            model_name,
            "model_name",
        )
        self.model_version = model_version

        self.allow_heuristic_fallback = bool(
            allow_heuristic_fallback
        )

        self.feature_order = tuple(
            feature_order or self.DEFAULT_FEATURE_ORDER
        )

        if (
            not self.feature_order
            or len(self.feature_order)
            != len(set(self.feature_order))
        ):
            raise RULValidationError(
                "feature_order must be non-empty and unique"
            )

        self.critical_threshold_hours = finite_float(
            critical_threshold_hours,
            "critical_threshold_hours",
            0,
        )

        self.high_threshold_hours = finite_float(
            high_threshold_hours,
            "high_threshold_hours",
            0,
        )

        self.moderate_threshold_hours = finite_float(
            moderate_threshold_hours,
            "moderate_threshold_hours",
            0,
        )

        if not (
            self.critical_threshold_hours
            < self.high_threshold_hours
            < self.moderate_threshold_hours
        ):
            raise RULValidationError(
                "RUL thresholds must be strictly increasing"
            )

        self.audit = audit_service

    # ========================================================
    # TELEMETRY FEATURE EXTRACTION
    # ========================================================

    @staticmethod
    def extract_features(
        telemetry_records: Sequence[
            Mapping[str, Any]
        ],
        *,
        operating_hours: Optional[float] = None,
        cycle_count: Optional[float] = None,
        health_score: Optional[float] = None,
    ) -> RULSensorFeatures:

        if not telemetry_records:
            raise RULValidationError(
                "At least one telemetry record is required"
            )

        sensor_values: dict[str, list[float]] = {
            "temperature": [],
            "vibration": [],
            "pressure": [],
            "current": [],
            "voltage": [],
        }

        for record in telemetry_records:
            if not isinstance(record, Mapping):
                raise RULValidationError(
                    "Telemetry records must be mappings"
                )

            for sensor in sensor_values:
                value = record.get(sensor)

                if value is not None:
                    sensor_values[sensor].append(
                        finite_float(
                            value,
                            sensor,
                        )
                    )

        def mean_of(
            sensor: str,
        ) -> Optional[float]:

            values = sensor_values[sensor]

            return (
                statistics.fmean(values)
                if values
                else None
            )

        def max_of(
            sensor: str,
        ) -> Optional[float]:

            values = sensor_values[sensor]

            return max(values) if values else None

        return RULSensorFeatures(
            temperature_mean=mean_of("temperature"),
            temperature_max=max_of("temperature"),
            vibration_mean=mean_of("vibration"),
            vibration_max=max_of("vibration"),
            pressure_mean=mean_of("pressure"),
            current_mean=mean_of("current"),
            voltage_mean=mean_of("voltage"),
            operating_hours=(
                finite_float(
                    operating_hours,
                    "operating_hours",
                    0,
                )
                if operating_hours is not None
                else None
            ),
            cycle_count=(
                finite_float(
                    cycle_count,
                    "cycle_count",
                    0,
                )
                if cycle_count is not None
                else None
            ),
            health_score=(
                finite_float(
                    health_score,
                    "health_score",
                    0,
                    100,
                )
                if health_score is not None
                else None
            ),
        )

    # ========================================================
    # MODEL LOADING
    # ========================================================

    async def _get_model(self) -> Any:

        if self.model is not None:
            return self.model

        if self.model_registry is None:
            return None

        method = getattr(
            self.model_registry,
            "get_model",
            None,
        )

        if method is None:
            raise RULModelError(
                "Model registry does not implement get_model"
            )

        try:
            return await invoke(
                method,
                self.model_name,
            )
        except Exception as exc:
            raise RULModelError(
                "Unable to load RUL prediction model"
            ) from exc

    # ========================================================
    # MODEL FEATURE VECTOR
    # ========================================================

    def build_feature_vector(
        self,
        features: RULSensorFeatures,
    ) -> list[float]:

        data = features.to_dict()

        vector = []

        for feature_name in self.feature_order:
            if feature_name not in data:
                raise RULValidationError(
                    f"Unknown model feature: {feature_name}"
                )

            value = data[feature_name]

            if value is None:
                raise RULValidationError(
                    f"Missing required model feature: "
                    f"{feature_name}"
                )

            vector.append(
                finite_float(
                    value,
                    feature_name,
                )
            )

        return vector

    # ========================================================
    # ML PREDICTION
    # ========================================================

    async def _predict_with_model(
        self,
        model: Any,
        features: RULSensorFeatures,
    ) -> float:

        vector = self.build_feature_vector(
            features
        )

        predict_method = getattr(
            model,
            "predict",
            None,
        )

        if predict_method is None:
            raise RULModelError(
                "RUL model must implement predict"
            )

        try:
            predictions = await invoke(
                predict_method,
                [vector],
            )

            prediction = predictions[0]

            # Accept scalar predictions and 1-column
            # prediction matrices.
            if isinstance(
                prediction,
                (list, tuple),
            ):
                if len(prediction) != 1:
                    raise RULModelError(
                        "Model must return one RUL value"
                    )

                prediction = prediction[0]

            return finite_float(
                prediction,
                "predicted_rul",
                0,
            )

        except RULServiceError:
            raise
        except Exception as exc:
            raise RULModelError(
                "RUL model inference failed"
            ) from exc

    # ========================================================
    # DEMONSTRATION HEURISTIC
    # ========================================================

    @staticmethod
    def _heuristic_predict(
        features: RULSensorFeatures,
        estimated_total_life_hours: float,
    ) -> float:
        """
        Illustrative engineering heuristic.

        NOT a trained or calibrated predictive model.
        Should not be used for safety-critical decisions.
        """

        total_life = finite_float(
            estimated_total_life_hours,
            "estimated_total_life_hours",
            0,
        )

        if total_life <= 0:
            raise RULValidationError(
                "Estimated total life must be positive"
            )

        operating_hours = (
            features.operating_hours or 0.0
        )

        baseline_remaining = max(
            0.0,
            total_life - operating_hours,
        )

        degradation_factor = 1.0

        if features.temperature_max is not None:
            if features.temperature_max > 85:
                degradation_factor *= 0.80

            if features.temperature_max > 100:
                degradation_factor *= 0.70

        if features.vibration_max is not None:
            if features.vibration_max > 7:
                degradation_factor *= 0.80

            if features.vibration_max > 12:
                degradation_factor *= 0.65

        if features.health_score is not None:
            degradation_factor *= max(
                0.20,
                features.health_score / 100.0,
            )

        return max(
            0.0,
            baseline_remaining * degradation_factor,
        )

    # ========================================================
    # RISK CLASSIFICATION
    # ========================================================

    def classify_risk(
        self,
        remaining_hours: float,
    ) -> RULRiskLevel:

        remaining = finite_float(
            remaining_hours,
            "remaining_hours",
            0,
        )

        if remaining <= self.critical_threshold_hours:
            return RULRiskLevel.CRITICAL

        if remaining <= self.high_threshold_hours:
            return RULRiskLevel.HIGH

        if remaining <= self.moderate_threshold_hours:
            return RULRiskLevel.MODERATE

        return RULRiskLevel.LOW

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    @staticmethod
    def generate_recommendations(
        risk_level: RULRiskLevel,
    ) -> list[str]:

        if risk_level == RULRiskLevel.CRITICAL:
            return [
                "Request an immediate engineering review.",
                "Verify model predictions against current "
                "telemetry and inspection evidence.",
                "Follow site-approved safety procedures "
                "and operating restrictions.",
                "Prepare a maintenance or replacement plan.",
            ]

        if risk_level == RULRiskLevel.HIGH:
            return [
                "Prioritize condition-based inspection.",
                "Review spare parts and maintenance capacity.",
                "Increase telemetry review frequency "
                "according to site procedures.",
            ]

        if risk_level == RULRiskLevel.MODERATE:
            return [
                "Review maintenance scheduling.",
                "Track degradation trends and model accuracy.",
            ]

        return [
            "Continue approved preventive maintenance.",
            "Monitor RUL trends and data quality.",
        ]

    # ========================================================
    # SINGLE RUL PREDICTION
    # ========================================================

    async def predict_rul(
        self,
        *,
        company_id: str,
        site_id: str,
        asset_id: str,
        telemetry_records: Sequence[
            Mapping[str, Any]
        ],
        operating_hours: Optional[float] = None,
        cycle_count: Optional[float] = None,
        health_score: Optional[float] = None,
        estimated_total_life_hours: Optional[float] = None,
        machine_id: Optional[str] = None,
        component_id: Optional[str] = None,
        confidence_level: Optional[float] = None,
        uncertainty_fraction: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        persist: bool = False,
    ) -> RULPredictionResult:

        company = validate_text(
            company_id,
            "company_id",
        )

        site = validate_text(
            site_id,
            "site_id",
        )

        asset = validate_text(
            asset_id,
            "asset_id",
        )

        features = self.extract_features(
            telemetry_records,
            operating_hours=operating_hours,
            cycle_count=cycle_count,
            health_score=health_score,
        )

        model = await self._get_model()

        source: RULPredictionSource

        if model is not None:
            predicted_rul = await self._predict_with_model(
                model,
                features,
            )

            source = RULPredictionSource.ML_MODEL

        elif self.allow_heuristic_fallback:
            if estimated_total_life_hours is None:
                raise RULValidationError(
                    "estimated_total_life_hours is required "
                    "for heuristic estimation"
                )

            predicted_rul = self._heuristic_predict(
                features,
                estimated_total_life_hours,
            )

            source = RULPredictionSource.HEURISTIC

        else:
            raise RULModelError(
                "No trained RUL model is available. "
                "Configure a model or explicitly enable "
                "the demonstration heuristic."
            )

        # ----------------------------------------------------
        # CONFIDENCE INTERVAL
        # ----------------------------------------------------

        lower_bound = None
        upper_bound = None
        selected_confidence = None

        if (
            confidence_level is not None
            or uncertainty_fraction is not None
        ):
            if (
                confidence_level is None
                or uncertainty_fraction is None
            ):
                raise RULValidationError(
                    "confidence_level and "
                    "uncertainty_fraction must be "
                    "provided together"
                )

            selected_confidence = finite_float(
                confidence_level,
                "confidence_level",
                0,
                1,
            )

            if selected_confidence <= 0:
                raise RULValidationError(
                    "confidence_level must be positive"
                )

            uncertainty = finite_float(
                uncertainty_fraction,
                "uncertainty_fraction",
                0,
                1,
            )

            lower_bound = max(
                0.0,
                predicted_rul * (1.0 - uncertainty),
            )

            upper_bound = (
                predicted_rul * (1.0 + uncertainty)
            )

        # ----------------------------------------------------
        # CONSUMED LIFE
        # ----------------------------------------------------

        total_life = None
        consumed_life_percent = None

        if estimated_total_life_hours is not None:
            total_life = finite_float(
                estimated_total_life_hours,
                "estimated_total_life_hours",
                0,
            )

            if total_life <= 0:
                raise RULValidationError(
                    "estimated_total_life_hours "
                    "must be positive"
                )

            if operating_hours is not None:
                consumed_life_percent = min(
                    100.0,
                    max(
                        0.0,
                        operating_hours / total_life * 100.0,
                    ),
                )

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        risk = self.classify_risk(
            predicted_rul
        )

        result = RULPredictionResult(
            prediction_id=str(uuid4()),
            company_id=company,
            site_id=site,
            asset_id=asset,
            machine_id=optional_text(
                machine_id,
                "machine_id",
            ),
            component_id=optional_text(
                component_id,
                "component_id",
            ),
            remaining_useful_life=round(
                predicted_rul,
                2,
            ),
            unit=RULUnit.HOURS.value,
            risk_level=risk.value,
            prediction_source=source.value,
            model_name=(
                self.model_name
                if source == RULPredictionSource.ML_MODEL
                else None
            ),
            model_version=(
                self.model_version
                if source == RULPredictionSource.ML_MODEL
                else None
            ),
            lower_bound=(
                round(lower_bound, 2)
                if lower_bound is not None
                else None
            ),
            upper_bound=(
                round(upper_bound, 2)
                if upper_bound is not None
                else None
            ),
            confidence_level=selected_confidence,
            estimated_total_life=total_life,
            consumed_life_percent=(
                round(consumed_life_percent, 2)
                if consumed_life_percent is not None
                else None
            ),
            health_score=features.health_score,
            features={
                key: value
                for key, value
                in features.to_dict().items()
                if value is not None
            },
            recommendations=(
                self.generate_recommendations(risk)
            ),
            metadata={
                **dict(metadata or {}),
                "feature_order": list(self.feature_order),
                "prediction_unit": "hours",
                "prediction_limitations": (
                    "Estimated remaining useful life "
                    "requires asset-specific validation."
                ),
                "interval_note": (
                    "User-supplied uncertainty bounds "
                    "are not calibrated confidence "
                    "intervals unless independently "
                    "validated."
                    if lower_bound is not None
                    else None
                ),
            },
        )

        if persist:
            await self.save_prediction(result)

        return result

    # ========================================================
    # BATCH PREDICTION
    # ========================================================

    async def predict_batch(
        self,
        requests: Sequence[
            Mapping[str, Any]
        ],
        *,
        persist: bool = False,
    ) -> list[RULPredictionResult]:

        results = []

        for request in requests:
            if not isinstance(request, Mapping):
                raise RULValidationError(
                    "Each batch request must be a mapping"
                )

            result = await self.predict_rul(
                **{
                    **dict(request),
                    "persist": persist,
                }
            )

            results.append(result)

        return results

    # ========================================================
    # RUL TREND ANALYSIS
    # ========================================================

    @staticmethod
    def analyze_rul_trend(
        *,
        asset_id: str,
        predictions: Sequence[
            Mapping[str, Any]
        ],
        stable_tolerance_hours_per_day: float = 2.0,
    ) -> RULTrendResult:

        identifier = validate_text(
            asset_id,
            "asset_id",
        )

        tolerance = finite_float(
            stable_tolerance_hours_per_day,
            "stable_tolerance_hours_per_day",
            0,
        )

        records: list[
            tuple[datetime, float, str]
        ] = []

        for prediction in predictions:
            if not isinstance(prediction, Mapping):
                raise RULValidationError(
                    "Prediction history must contain mappings"
                )

            record_asset = prediction.get(
                "asset_id"
            )

            if record_asset != identifier:
                continue

            timestamp = prediction.get(
                "predicted_at"
            )

            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(
                        timestamp.replace("Z", "+00:00")
                    )
                except ValueError as exc:
                    raise RULValidationError(
                        "Invalid prediction timestamp"
                    ) from exc

            timestamp = validate_datetime(
                timestamp,
                "predicted_at",
            )

            rul = finite_float(
                prediction.get(
                    "remaining_useful_life"
                ),
                "remaining_useful_life",
                0,
            )

            unit = str(
                prediction.get("unit", "hours")
            )

            if unit != RULUnit.HOURS.value:
                raise RULValidationError(
                    "Trend analysis requires RUL in hours"
                )

            records.append(
                (timestamp, rul, unit)
            )

        records.sort(
            key=lambda record: record[0]
        )

        if len(records) < 2:
            return RULTrendResult(
                asset_id=identifier,
                trend=RULTrend.INSUFFICIENT_DATA.value,
                prediction_count=len(records),
                first_rul=(
                    records[0][1]
                    if records
                    else None
                ),
                latest_rul=(
                    records[-1][1]
                    if records
                    else None
                ),
                change=None,
                slope_per_day=None,
                unit=(
                    RULUnit.HOURS.value
                    if records
                    else None
                ),
            )

        first_time, first_rul, _ = records[0]
        last_time, latest_rul, _ = records[-1]

        elapsed_days = (
            last_time - first_time
        ).total_seconds() / 86400.0

        if elapsed_days <= 0:
            raise RULValidationError(
                "Distinct prediction timestamps "
                "are required for trend analysis"
            )

        change = latest_rul - first_rul

        slope = change / elapsed_days

        # Under normal operation RUL decreases as time
        # passes. Compare predicted RUL with elapsed
        # operating time to estimate unexpected degradation.
        expected_change = -elapsed_days * 24.0

        excess_change = change - expected_change

        excess_slope = excess_change / elapsed_days

        if abs(excess_slope) <= tolerance:
            trend = RULTrend.STABLE

        elif excess_slope < 0:
            trend = RULTrend.DEGRADING

        else:
            trend = RULTrend.IMPROVING

        return RULTrendResult(
            asset_id=identifier,
            trend=trend.value,
            prediction_count=len(records),
            first_rul=round(first_rul, 2),
            latest_rul=round(latest_rul, 2),
            change=round(change, 2),
            slope_per_day=round(slope, 2),
            unit=RULUnit.HOURS.value,
        )

    # ========================================================
    # RUL UNIT CONVERSION
    # ========================================================

    @staticmethod
    def convert_rul(
        *,
        value: float,
        from_unit: str | RULUnit,
        to_unit: str | RULUnit,
        operating_hours_per_day: float = 24.0,
    ) -> float:

        source = parse_enum(
            RULUnit,
            from_unit,
            "from_unit",
        )

        target = parse_enum(
            RULUnit,
            to_unit,
            "to_unit",
        )

        amount = finite_float(
            value,
            "value",
            0,
        )

        hours_per_day = finite_float(
            operating_hours_per_day,
            "operating_hours_per_day",
            0.000001,
            24,
        )

        if source == target:
            return round(amount, 2)

        if RULUnit.CYCLES in {source, target}:
            raise RULValidationError(
                "Cycle conversion requires a validated "
                "asset-specific cycles-per-hour rate"
            )

        if source == RULUnit.HOURS:
            return round(
                amount / hours_per_day,
                2,
            )

        return round(
            amount * hours_per_day,
            2,
        )

    # ========================================================
    # AUDIT
    # ========================================================

    async def _audit(
        self,
        *,
        prediction_id: str,
        action: str,
        details: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> None:

        if self.audit is None:
            return

        method = getattr(
            self.audit,
            "record_event",
            None,
        )

        if method is None:
            return

        try:
            await invoke(
                method,
                {
                    "event_id": str(uuid4()),
                    "entity_type": "rul_prediction",
                    "entity_id": prediction_id,
                    "actor_id": "system",
                    "action": action,
                    "details": dict(details or {}),
                    "timestamp": utc_now().isoformat(),
                },
            )

        except Exception:
            logger.exception(
                "RUL audit failed for prediction %s",
                prediction_id,
            )

    # ========================================================
    # SAVE PREDICTION
    # ========================================================

    async def save_prediction(
        self,
        prediction: RULPredictionResult,
    ) -> Any:

        if self.repository is None:
            raise RULRepositoryError(
                "RUL repository is not configured"
            )

        if not isinstance(
            prediction,
            RULPredictionResult,
        ):
            raise RULValidationError(
                "Expected RULPredictionResult"
            )

        try:
            saved = await invoke(
                self.repository.create_rul_prediction,
                prediction.to_dict(),
            )
        except Exception as exc:
            raise RULRepositoryError(
                "Unable to save RUL prediction"
            ) from exc

        await self._audit(
            prediction_id=prediction.prediction_id,
            action="rul_prediction_created",
            details={
                "asset_id": prediction.asset_id,
                "risk_level": prediction.risk_level,
                "remaining_useful_life": (
                    prediction.remaining_useful_life
                ),
                "unit": prediction.unit,
                "prediction_source": (
                    prediction.prediction_source
                ),
            },
        )

        return saved

    # ========================================================
    # GET PREDICTION
    # ========================================================

    async def get_prediction(
        self,
        prediction_id: str,
    ) -> Any:

        if self.repository is None:
            raise RULRepositoryError(
                "RUL repository is not configured"
            )

        identifier = validate_text(
            prediction_id,
            "prediction_id",
        )

        record = await invoke(
            self.repository.get_rul_prediction,
            identifier,
        )

        if record is None:
            raise RULPredictionNotFoundError(
                f"RUL prediction not found: {identifier}"
            )

        return record

    # ========================================================
    # LIST PREDICTIONS
    # ========================================================

    async def list_predictions(
        self,
        *,
        company_id: Optional[str] = None,
        site_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        risk_level: Optional[str] = None,
    ) -> Any:

        if self.repository is None:
            raise RULRepositoryError(
                "RUL repository is not configured"
            )

        filters: dict[str, Any] = {}

        for name, value in {
            "company_id": company_id,
            "site_id": site_id,
            "asset_id": asset_id,
        }.items():
            if value is not None:
                filters[name] = validate_text(
                    value,
                    name,
                )

        if risk_level is not None:
            filters["risk_level"] = parse_enum(
                RULRiskLevel,
                risk_level,
                "risk_level",
            ).value

        return await invoke(
            self.repository.list_rul_predictions,
            filters,
        )


# ============================================================
# IN-MEMORY REPOSITORY
# ============================================================

class InMemoryRULRepository:
    """
    Development-only repository.

    Not durable or concurrency-safe.
    """

    def __init__(self):
        self.records: dict[
            str,
            dict[str, Any],
        ] = {}

    async def create_rul_prediction(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:

        record = copy.deepcopy(data)

        prediction_id = record["prediction_id"]

        if prediction_id in self.records:
            raise RULRepositoryError(
                "Duplicate RUL prediction ID"
            )

        self.records[prediction_id] = record

        return copy.deepcopy(record)

    async def get_rul_prediction(
        self,
        prediction_id: str,
    ) -> Optional[dict[str, Any]]:

        record = self.records.get(
            prediction_id
        )

        return (
            copy.deepcopy(record)
            if record is not None
            else None
        )

    async def list_rul_predictions(
        self,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:

        return [
            copy.deepcopy(record)
            for record in self.records.values()
            if all(
                record.get(key) == value
                for key, value in filters.items()
            )
        ]


# ============================================================
# DEMONSTRATION
# ============================================================

async def main() -> None:

    logging.basicConfig(
        level=logging.INFO
    )

    repository = InMemoryRULRepository()

    # Demonstration mode only.
    # Production should provide a trained model.
    service = RULService(
        rul_repository=repository,
        allow_heuristic_fallback=True,
        model_name="industrial_rul_model",
        model_version="1.0.0",
    )

    telemetry = [
        {
            "temperature": 80.0,
            "vibration": 5.5,
            "pressure": 180.0,
            "current": 45.0,
            "voltage": 230.0,
        },
        {
            "temperature": 85.0,
            "vibration": 6.2,
            "pressure": 182.0,
            "current": 46.0,
            "voltage": 229.0,
        },
        {
            "temperature": 91.0,
            "vibration": 8.5,
            "pressure": 185.0,
            "current": 48.0,
            "voltage": 228.0,
        },
    ]

    prediction = await service.predict_rul(
        company_id="COMPANY-001",
        site_id="SITE-001",
        asset_id="MACHINE-001",
        machine_id="MOTOR-001",
        telemetry_records=telemetry,
        operating_hours=4200,
        cycle_count=18000,
        health_score=62,
        estimated_total_life_hours=5000,
        metadata={
            "environment": "development",
            "purpose": "demonstration",
        },
        persist=True,
    )

    print("\nRUL PREDICTION")

    print(
        json.dumps(
            prediction.to_dict(),
            indent=2,
        )
    )

    print("\nRUL IN OPERATING DAYS")

    print(
        service.convert_rul(
            value=prediction.remaining_useful_life,
            from_unit="hours",
            to_unit="days",
            operating_hours_per_day=20,
        )
    )

    history = await service.list_predictions(
        asset_id="MACHINE-001"
    )

    print("\nPREDICTION HISTORY")

    print(
        json.dumps(
            {
                "total_predictions": len(history),
                "asset_id": "MACHINE-001",
            },
            indent=2,
        )
    )

    print("\nRUL TREND")

    trend = service.analyze_rul_trend(
        asset_id="MACHINE-001",
        predictions=history,
    )

    print(
        json.dumps(
            trend.to_dict(),
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
    