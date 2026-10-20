# app/ml/anomaly_detection/multivariate_anomaly.py

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite, sqrt
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4


# ============================================================
# Enums
# ============================================================


class MultivariateAnomalyType(str, Enum):
    POINT = "point"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    CORRELATION_BREAK = "correlation_break"
    MULTI_SENSOR_DEVIATION = "multi_sensor_deviation"
    OPERATING_REGIME_SHIFT = "operating_regime_shift"
    SENSOR_DISAGREEMENT = "sensor_disagreement"
    DEGRADATION_PATTERN = "degradation_pattern"
    UNKNOWN = "unknown"


class MultivariateSeverity(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MultivariateState(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    ANOMALOUS = "anomalous"
    CRITICAL = "critical"
    INSUFFICIENT_DATA = "insufficient_data"


class DetectionMethod(str, Enum):
    Z_SCORE = "z_score"
    MAHALANOBIS_LIKE = "mahalanobis_like"
    CORRELATION = "correlation"
    ENSEMBLE = "ensemble"


# ============================================================
# Models
# ============================================================


@dataclass(slots=True)
class MultivariateObservation:
    """
    One synchronized observation containing multiple sensor features.

    Example:
        {
            "temperature_c": 92.5,
            "vibration_rms": 8.1,
            "current_a": 118.0,
            "pressure_bar": 6.3,
            "rpm": 1750
        }
    """

    timestamp: datetime
    features: Mapping[str, float]

    asset_id: str | None = None
    sensor_group_id: str | None = None
    site_id: str | None = None
    zone_id: str | None = None

    confidence: float = 1.0
    operating_mode: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = _clamp(self.confidence)

        cleaned: dict[str, float] = {}

        for name, value in self.features.items():
            if _valid_number(value):
                cleaned[str(name)] = float(value)

        self.features = cleaned


@dataclass(slots=True)
class FeatureStatistics:
    name: str
    mean: float
    std: float
    minimum: float
    maximum: float
    sample_count: int


@dataclass(slots=True)
class FeatureDeviation:
    feature: str
    value: float
    expected_mean: float
    expected_std: float
    z_score: float
    normalized_score: float
    anomalous: bool


@dataclass(slots=True)
class CorrelationBreak:
    feature_a: str
    feature_b: str
    baseline_correlation: float
    recent_correlation: float
    difference: float
    score: float


@dataclass(slots=True)
class MultivariateEvidence:
    name: str
    value: float | str
    score: float
    description: str

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MultivariateAnomaly:
    anomaly_id: str
    timestamp: datetime

    anomaly_type: MultivariateAnomalyType
    severity: MultivariateSeverity

    score: float
    confidence: float

    asset_id: str | None

    contributing_features: list[str]
    deviations: list[FeatureDeviation]
    correlation_breaks: list[CorrelationBreak]
    evidence: list[MultivariateEvidence]

    requires_investigation: bool = False
    requires_maintenance_review: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MultivariateAssessment:
    assessment_id: str
    timestamp: datetime
    asset_id: str | None

    state: MultivariateState
    severity: MultivariateSeverity

    anomaly_score: float
    confidence: float

    method: DetectionMethod

    feature_count: int
    anomalous_feature_count: int

    contributing_features: list[str]

    deviations: list[FeatureDeviation]
    correlation_breaks: list[CorrelationBreak]
    anomalies: list[MultivariateAnomaly]
    evidence: list[MultivariateEvidence]

    requires_investigation: bool = False
    requires_maintenance_review: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_anomalous(self) -> bool:
        return self.severity in {
            MultivariateSeverity.MEDIUM,
            MultivariateSeverity.HIGH,
            MultivariateSeverity.CRITICAL,
        }

    @property
    def is_high_risk(self) -> bool:
        return self.severity in {
            MultivariateSeverity.HIGH,
            MultivariateSeverity.CRITICAL,
        }


# ============================================================
# Configuration
# ============================================================


@dataclass(slots=True)
class MultivariatePolicy:
    min_history: int = 20
    max_history: int = 1000

    minimum_confidence: float = 0.50

    z_warning: float = 2.0
    z_anomaly: float = 3.0
    z_critical: float = 5.0

    medium_score: float = 0.40
    high_score: float = 0.70
    critical_score: float = 0.90

    minimum_anomalous_features: int = 2

    correlation_window: int = 30
    minimum_correlation_samples: int = 10
    strong_correlation: float = 0.70
    correlation_break_threshold: float = 0.40

    point_weight: float = 0.60
    joint_weight: float = 0.25
    correlation_weight: float = 0.15


# ============================================================
# Baseline Model
# ============================================================


class MultivariateBaseline:
    """
    Rolling baseline for industrial features.

    This is deliberately lightweight and dependency-free.

    For production, this can later be replaced or augmented with:
      - Isolation Forest
      - Robust covariance
      - PCA
      - One-Class SVM
      - Autoencoder
      - LSTM/Transformer anomaly detector
    """

    def __init__(self, max_history: int = 1000) -> None:
        self._history: deque[MultivariateObservation] = deque(
            maxlen=max_history
        )

    def add(self, observation: MultivariateObservation) -> None:
        self._history.append(observation)

    def extend(
        self,
        observations: Iterable[MultivariateObservation],
    ) -> None:
        for observation in observations:
            self.add(observation)

    @property
    def size(self) -> int:
        return len(self._history)

    @property
    def observations(self) -> list[MultivariateObservation]:
        return list(self._history)

    def feature_values(self, feature: str) -> list[float]:
        return [
            obs.features[feature]
            for obs in self._history
            if feature in obs.features
        ]

    def statistics(
        self,
        feature: str,
    ) -> FeatureStatistics | None:

        values = self.feature_values(feature)

        if not values:
            return None

        return FeatureStatistics(
            name=feature,
            mean=mean(values),
            std=pstdev(values) if len(values) > 1 else 0.0,
            minimum=min(values),
            maximum=max(values),
            sample_count=len(values),
        )

    def feature_names(self) -> list[str]:
        names: set[str] = set()

        for observation in self._history:
            names.update(observation.features)

        return sorted(names)


# ============================================================
# Detector
# ============================================================


class MultivariateAnomalyDetector:
    """
    Industrial multivariate anomaly detector.

    Detects:
      - simultaneous sensor deviations
      - unusual combinations of otherwise valid sensor values
      - correlated sensor relationship breakdowns
      - operating-pattern shifts
      - multi-sensor degradation patterns

    It produces anomaly intelligence only.

    It MUST NOT directly:
      - stop machinery
      - trigger emergency shutdown
      - isolate electrical systems
      - evacuate workers

    Safety actions belong to the deterministic decision/safety layer.
    """

    _SEVERITY_RANK = {
        MultivariateSeverity.NORMAL: 0,
        MultivariateSeverity.LOW: 1,
        MultivariateSeverity.MEDIUM: 2,
        MultivariateSeverity.HIGH: 3,
        MultivariateSeverity.CRITICAL: 4,
    }

    def __init__(
        self,
        policy: MultivariatePolicy | None = None,
        baseline: MultivariateBaseline | None = None,
    ) -> None:

        self.policy = policy or MultivariatePolicy()

        self.baseline = baseline or MultivariateBaseline(
            max_history=self.policy.max_history
        )

    # ========================================================
    # Training / baseline
    # ========================================================

    def fit(
        self,
        observations: Iterable[MultivariateObservation],
    ) -> None:
        self.baseline.extend(observations)

    def update_baseline(
        self,
        observation: MultivariateObservation,
        *,
        only_if_normal: bool = True,
    ) -> bool:
        """
        Prevent anomalous observations from contaminating the baseline.
        """

        if only_if_normal and self.baseline.size >= self.policy.min_history:
            assessment = self.assess(
                observation,
                update_baseline=False,
            )

            if assessment.is_anomalous:
                return False

        self.baseline.add(observation)
        return True

    # ========================================================
    # Public detection
    # ========================================================

    def assess(
        self,
        observation: MultivariateObservation,
        *,
        update_baseline: bool = False,
    ) -> MultivariateAssessment:

        if observation.confidence < self.policy.minimum_confidence:
            return self._insufficient_assessment(
                observation,
                reason="low_observation_confidence",
            )

        if self.baseline.size < self.policy.min_history:
            if update_baseline:
                self.baseline.add(observation)

            return self._insufficient_assessment(
                observation,
                reason="insufficient_baseline_history",
            )

        deviations = self._feature_deviations(observation)

        correlation_breaks = self._correlation_breaks(
            observation
        )

        anomalous_deviations = [
            deviation
            for deviation in deviations
            if deviation.anomalous
        ]

        point_score = self._point_score(deviations)

        joint_score = self._joint_score(
            anomalous_deviations,
            len(deviations),
        )

        correlation_score = max(
            (
                correlation.score
                for correlation in correlation_breaks
            ),
            default=0.0,
        )

        anomaly_score = self._fuse_scores(
            point_score=point_score,
            joint_score=joint_score,
            correlation_score=correlation_score,
        )

        confidence = self._assessment_confidence(
            observation,
            deviations,
        )

        severity = self._severity(
            anomaly_score,
            anomalous_deviations,
        )

        anomaly_type = self._classify_anomaly(
            anomalous_deviations,
            correlation_breaks,
        )

        contributing_features = self._contributing_features(
            anomalous_deviations,
            correlation_breaks,
        )

        evidence = self._build_evidence(
            anomalous_deviations,
            correlation_breaks,
            point_score,
            joint_score,
            correlation_score,
        )

        anomalies: list[MultivariateAnomaly] = []

        if severity != MultivariateSeverity.NORMAL:
            anomalies.append(
                MultivariateAnomaly(
                    anomaly_id=str(uuid4()),
                    timestamp=observation.timestamp,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    score=anomaly_score,
                    confidence=confidence,
                    asset_id=observation.asset_id,
                    contributing_features=contributing_features,
                    deviations=anomalous_deviations,
                    correlation_breaks=correlation_breaks,
                    evidence=evidence,
                    requires_investigation=(
                        severity
                        in {
                            MultivariateSeverity.MEDIUM,
                            MultivariateSeverity.HIGH,
                            MultivariateSeverity.CRITICAL,
                        }
                    ),
                    requires_maintenance_review=(
                        severity
                        in {
                            MultivariateSeverity.HIGH,
                            MultivariateSeverity.CRITICAL,
                        }
                    ),
                    metadata={
                        "operating_mode": observation.operating_mode,
                        "sensor_group_id": observation.sensor_group_id,
                    },
                )
            )

        assessment = MultivariateAssessment(
            assessment_id=str(uuid4()),
            timestamp=observation.timestamp,
            asset_id=observation.asset_id,
            state=self._state(severity),
            severity=severity,
            anomaly_score=anomaly_score,
            confidence=confidence,
            method=DetectionMethod.ENSEMBLE,
            feature_count=len(deviations),
            anomalous_feature_count=len(anomalous_deviations),
            contributing_features=contributing_features,
            deviations=deviations,
            correlation_breaks=correlation_breaks,
            anomalies=anomalies,
            evidence=evidence,
            requires_investigation=(
                severity
                in {
                    MultivariateSeverity.MEDIUM,
                    MultivariateSeverity.HIGH,
                    MultivariateSeverity.CRITICAL,
                }
            ),
            requires_maintenance_review=(
                severity
                in {
                    MultivariateSeverity.HIGH,
                    MultivariateSeverity.CRITICAL,
                }
            ),
            metadata={
                "baseline_size": self.baseline.size,
                "operating_mode": observation.operating_mode,
                "site_id": observation.site_id,
                "zone_id": observation.zone_id,
            },
        )

        if update_baseline and not assessment.is_anomalous:
            self.baseline.add(observation)

        return assessment

    def detect(
        self,
        observation: MultivariateObservation,
    ) -> list[MultivariateAnomaly]:
        return self.assess(observation).anomalies

    def assess_many(
        self,
        observations: Iterable[MultivariateObservation],
        *,
        update_baseline: bool = False,
    ) -> list[MultivariateAssessment]:

        return [
            self.assess(
                observation,
                update_baseline=update_baseline,
            )
            for observation in observations
        ]

    # ========================================================
    # Feature deviation
    # ========================================================

    def _feature_deviations(
        self,
        observation: MultivariateObservation,
    ) -> list[FeatureDeviation]:

        output: list[FeatureDeviation] = []

        for feature, value in observation.features.items():

            statistics = self.baseline.statistics(feature)

            if (
                statistics is None
                or statistics.sample_count < self.policy.min_history
            ):
                continue

            std = statistics.std

            if std <= 1e-12:
                z_score = (
                    0.0
                    if abs(value - statistics.mean) <= 1e-12
                    else self.policy.z_critical
                )
            else:
                z_score = abs(
                    (value - statistics.mean) / std
                )

            normalized_score = _clamp(
                z_score / self.policy.z_critical
            )

            output.append(
                FeatureDeviation(
                    feature=feature,
                    value=value,
                    expected_mean=statistics.mean,
                    expected_std=std,
                    z_score=z_score,
                    normalized_score=normalized_score,
                    anomalous=(
                        z_score >= self.policy.z_anomaly
                    ),
                )
            )

        return output

    # ========================================================
    # Correlation analysis
    # ========================================================

    def _correlation_breaks(
        self,
        observation: MultivariateObservation,
    ) -> list[CorrelationBreak]:

        features = [
            feature
            for feature in observation.features
            if len(
                self.baseline.feature_values(feature)
            ) >= self.policy.minimum_correlation_samples
        ]

        if len(features) < 2:
            return []

        history = self.baseline.observations

        recent = history[-self.policy.correlation_window :]

        results: list[CorrelationBreak] = []

        for index, feature_a in enumerate(features):

            for feature_b in features[index + 1 :]:

                baseline_a, baseline_b = _paired_values(
                    history,
                    feature_a,
                    feature_b,
                )

                recent_a, recent_b = _paired_values(
                    recent,
                    feature_a,
                    feature_b,
                )

                if (
                    len(baseline_a)
                    < self.policy.minimum_correlation_samples
                    or len(recent_a)
                    < self.policy.minimum_correlation_samples
                ):
                    continue

                baseline_corr = _pearson(
                    baseline_a,
                    baseline_b,
                )

                recent_corr = _pearson(
                    recent_a,
                    recent_b,
                )

                if (
                    abs(baseline_corr)
                    < self.policy.strong_correlation
                ):
                    continue

                difference = abs(
                    baseline_corr - recent_corr
                )

                if (
                    difference
                    < self.policy.correlation_break_threshold
                ):
                    continue

                score = _clamp(
                    difference / 2.0
                )

                results.append(
                    CorrelationBreak(
                        feature_a=feature_a,
                        feature_b=feature_b,
                        baseline_correlation=baseline_corr,
                        recent_correlation=recent_corr,
                        difference=difference,
                        score=score,
                    )
                )

        return sorted(
            results,
            key=lambda item: item.score,
            reverse=True,
        )

    # ========================================================
    # Scoring
    # ========================================================

    @staticmethod
    def _point_score(
        deviations: Sequence[FeatureDeviation],
    ) -> float:

        if not deviations:
            return 0.0

        scores = sorted(
            (
                deviation.normalized_score
                for deviation in deviations
            ),
            reverse=True,
        )

        strongest = scores[: min(3, len(scores))]

        return _clamp(mean(strongest))

    def _joint_score(
        self,
        deviations: Sequence[FeatureDeviation],
        total_features: int,
    ) -> float:

        if not deviations or total_features <= 0:
            return 0.0

        ratio = len(deviations) / total_features

        strength = mean(
            deviation.normalized_score
            for deviation in deviations
        )

        minimum_bonus = (
            0.15
            if len(deviations)
            >= self.policy.minimum_anomalous_features
            else 0.0
        )

        return _clamp(
            0.50 * ratio
            + 0.50 * strength
            + minimum_bonus
        )

    def _fuse_scores(
        self,
        *,
        point_score: float,
        joint_score: float,
        correlation_score: float,
    ) -> float:

        policy = self.policy

        score = (
            point_score * policy.point_weight
            + joint_score * policy.joint_weight
            + correlation_score * policy.correlation_weight
        )

        return _clamp(score)

    def _assessment_confidence(
        self,
        observation: MultivariateObservation,
        deviations: Sequence[FeatureDeviation],
    ) -> float:

        if not deviations:
            return _clamp(
                observation.confidence * 0.50
            )

        coverage = (
            len(deviations)
            / max(len(observation.features), 1)
        )

        history_factor = min(
            1.0,
            self.baseline.size
            / max(self.policy.min_history * 3, 1),
        )

        return _clamp(
            observation.confidence
            * (
                0.60
                + 0.20 * coverage
                + 0.20 * history_factor
            )
        )

    # ========================================================
    # Classification
    # ========================================================

    def _severity(
        self,
        score: float,
        deviations: Sequence[FeatureDeviation],
    ) -> MultivariateSeverity:

        critical_features = sum(
            1
            for deviation in deviations
            if deviation.z_score >= self.policy.z_critical
        )

        if (
            score >= self.policy.critical_score
            or critical_features >= 2
        ):
            return MultivariateSeverity.CRITICAL

        if score >= self.policy.high_score:
            return MultivariateSeverity.HIGH

        if score >= self.policy.medium_score:
            return MultivariateSeverity.MEDIUM

        if deviations:
            return MultivariateSeverity.LOW

        return MultivariateSeverity.NORMAL

    def _classify_anomaly(
        self,
        deviations: Sequence[FeatureDeviation],
        correlations: Sequence[CorrelationBreak],
    ) -> MultivariateAnomalyType:

        if correlations and not deviations:
            return MultivariateAnomalyType.CORRELATION_BREAK

        if (
            len(deviations)
            >= self.policy.minimum_anomalous_features
            and correlations
        ):
            return MultivariateAnomalyType.COLLECTIVE

        if (
            len(deviations)
            >= self.policy.minimum_anomalous_features
        ):
            return (
                MultivariateAnomalyType.MULTI_SENSOR_DEVIATION
            )

        if len(deviations) == 1:
            return MultivariateAnomalyType.POINT

        return MultivariateAnomalyType.UNKNOWN

    # ========================================================
    # Evidence
    # ========================================================

    @staticmethod
    def _contributing_features(
        deviations: Sequence[FeatureDeviation],
        correlations: Sequence[CorrelationBreak],
    ) -> list[str]:

        features = {
            deviation.feature
            for deviation in deviations
        }

        for correlation in correlations:
            features.add(correlation.feature_a)
            features.add(correlation.feature_b)

        return sorted(features)

    @staticmethod
    def _build_evidence(
        deviations: Sequence[FeatureDeviation],
        correlations: Sequence[CorrelationBreak],
        point_score: float,
        joint_score: float,
        correlation_score: float,
    ) -> list[MultivariateEvidence]:

        evidence: list[MultivariateEvidence] = []

        for deviation in deviations:
            evidence.append(
                MultivariateEvidence(
                    name=deviation.feature,
                    value=deviation.value,
                    score=deviation.normalized_score,
                    description=(
                        f"{deviation.feature} deviated "
                        f"{deviation.z_score:.2f} standard "
                        "deviations from baseline"
                    ),
                    metadata={
                        "expected_mean":
                            deviation.expected_mean,
                        "expected_std":
                            deviation.expected_std,
                    },
                )
            )

        for correlation in correlations:
            evidence.append(
                MultivariateEvidence(
                    name=(
                        f"{correlation.feature_a}:"
                        f"{correlation.feature_b}"
                    ),
                    value=correlation.difference,
                    score=correlation.score,
                    description=(
                        "Historical sensor correlation changed "
                        f"from {correlation.baseline_correlation:.3f} "
                        f"to {correlation.recent_correlation:.3f}"
                    ),
                )
            )

        evidence.extend(
            [
                MultivariateEvidence(
                    name="point_score",
                    value=point_score,
                    score=point_score,
                    description="Individual feature anomaly score",
                ),
                MultivariateEvidence(
                    name="joint_score",
                    value=joint_score,
                    score=joint_score,
                    description="Joint multi-feature anomaly score",
                ),
                MultivariateEvidence(
                    name="correlation_score",
                    value=correlation_score,
                    score=correlation_score,
                    description="Sensor correlation-break score",
                ),
            ]
        )

        return evidence

    # ========================================================
    # State
    # ========================================================

    @staticmethod
    def _state(
        severity: MultivariateSeverity,
    ) -> MultivariateState:

        mapping = {
            MultivariateSeverity.NORMAL:
                MultivariateState.NORMAL,
            MultivariateSeverity.LOW:
                MultivariateState.WARNING,
            MultivariateSeverity.MEDIUM:
                MultivariateState.ANOMALOUS,
            MultivariateSeverity.HIGH:
                MultivariateState.ANOMALOUS,
            MultivariateSeverity.CRITICAL:
                MultivariateState.CRITICAL,
        }

        return mapping[severity]

    def _insufficient_assessment(
        self,
        observation: MultivariateObservation,
        *,
        reason: str,
    ) -> MultivariateAssessment:

        return MultivariateAssessment(
            assessment_id=str(uuid4()),
            timestamp=observation.timestamp,
            asset_id=observation.asset_id,
            state=MultivariateState.INSUFFICIENT_DATA,
            severity=MultivariateSeverity.NORMAL,
            anomaly_score=0.0,
            confidence=observation.confidence,
            method=DetectionMethod.ENSEMBLE,
            feature_count=len(observation.features),
            anomalous_feature_count=0,
            contributing_features=[],
            deviations=[],
            correlation_breaks=[],
            anomalies=[],
            evidence=[
                MultivariateEvidence(
                    name="baseline",
                    value=self.baseline.size,
                    score=0.0,
                    description=reason,
                )
            ],
            metadata={
                "reason": reason,
                "required_history":
                    self.policy.min_history,
            },
        )


# ============================================================
# Optional robust distance
# ============================================================


def standardized_multivariate_distance(
    observation: Mapping[str, float],
    baseline: Sequence[Mapping[str, float]],
) -> float:
    """
    Dependency-free diagonal Mahalanobis-like distance.

    This deliberately ignores covariance between features.
    Use sklearn RobustCovariance / MinCovDet or a trained model
    when true covariance-aware distance is required.
    """

    if not baseline:
        return 0.0

    squared_distance = 0.0
    usable_features = 0

    for feature, value in observation.items():

        values = [
            float(row[feature])
            for row in baseline
            if (
                feature in row
                and _valid_number(row[feature])
            )
        ]

        if len(values) < 3:
            continue

        mu = mean(values)
        sigma = pstdev(values)

        if sigma <= 1e-12:
            continue

        z = (float(value) - mu) / sigma

        squared_distance += z * z
        usable_features += 1

    if usable_features == 0:
        return 0.0

    return sqrt(
        squared_distance / usable_features
    )


# ============================================================
# Convenience API
# ============================================================


def create_observation(
    features: Mapping[str, float],
    *,
    asset_id: str | None = None,
    sensor_group_id: str | None = None,
    operating_mode: str | None = None,
    confidence: float = 1.0,
    timestamp: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> MultivariateObservation:

    return MultivariateObservation(
        timestamp=timestamp or datetime.now(timezone.utc),
        features=features,
        asset_id=asset_id,
        sensor_group_id=sensor_group_id,
        operating_mode=operating_mode,
        confidence=confidence,
        metadata=metadata or {},
    )


# ============================================================
# Math utilities
# ============================================================


def _paired_values(
    observations: Sequence[MultivariateObservation],
    feature_a: str,
    feature_b: str,
) -> tuple[list[float], list[float]]:

    values_a: list[float] = []
    values_b: list[float] = []

    for observation in observations:

        if (
            feature_a not in observation.features
            or feature_b not in observation.features
        ):
            continue

        a = observation.features[feature_a]
        b = observation.features[feature_b]

        if not (_valid_number(a) and _valid_number(b)):
            continue

        values_a.append(float(a))
        values_b.append(float(b))

    return values_a, values_b


def _pearson(
    x: Sequence[float],
    y: Sequence[float],
) -> float:

    if len(x) != len(y) or len(x) < 2:
        return 0.0

    mean_x = mean(x)
    mean_y = mean(y)

    numerator = sum(
        (a - mean_x) * (b - mean_y)
        for a, b in zip(x, y)
    )

    denominator_x = sum(
        (a - mean_x) ** 2
        for a in x
    )

    denominator_y = sum(
        (b - mean_y) ** 2
        for b in y
    )

    denominator = sqrt(
        denominator_x * denominator_y
    )

    if denominator <= 1e-12:
        return 0.0

    return max(
        -1.0,
        min(1.0, numerator / denominator),
    )


def _valid_number(value: Any) -> bool:
    try:
        return isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:

    return max(
        minimum,
        min(maximum, float(value)),
    )


# ============================================================
# Default detector
# ============================================================


default_multivariate_anomaly_detector = (
    MultivariateAnomalyDetector()
)


__all__ = [
    "MultivariateAnomalyType",
    "MultivariateSeverity",
    "MultivariateState",
    "DetectionMethod",
    "MultivariateObservation",
    "FeatureStatistics",
    "FeatureDeviation",
    "CorrelationBreak",
    "MultivariateEvidence",
    "MultivariateAnomaly",
    "MultivariateAssessment",
    "MultivariatePolicy",
    "MultivariateBaseline",
    "MultivariateAnomalyDetector",
    "standardized_multivariate_distance",
    "create_observation",
    "default_multivariate_anomaly_detector",
]