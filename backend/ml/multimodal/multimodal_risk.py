# app/ml/multimodal/multimodal_risk.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from math import exp, isfinite
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4


# ============================================================
# Enums
# ============================================================


class RiskLevel(IntEnum):
    NORMAL = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class RiskState(str, Enum):
    NORMAL = "normal"
    WATCH = "watch"
    ELEVATED = "elevated"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"
    UNCERTAIN = "uncertain"


class RiskCategory(str, Enum):
    SAFETY = "safety"
    EQUIPMENT = "equipment"
    FAILURE = "failure"
    THERMAL = "thermal"
    ELECTRICAL = "electrical"
    MECHANICAL = "mechanical"
    FIRE = "fire"
    ENVIRONMENTAL = "environmental"
    PERSONNEL = "personnel"
    PRODUCTION = "production"
    MAINTENANCE = "maintenance"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"


class RiskSource(str, Enum):
    SENSOR = "sensor"
    TELEMETRY = "telemetry"
    TEMPERATURE_ANOMALY = "temperature_anomaly"
    VIBRATION_ANOMALY = "vibration_anomaly"
    ELECTRICAL_ANOMALY = "electrical_anomaly"
    SENSOR_ANOMALY = "sensor_anomaly"
    MULTIVARIATE_ANOMALY = "multivariate_anomaly"
    COMPUTER_VISION = "computer_vision"
    IMAGE_SENSOR_FUSION = "image_sensor_fusion"
    ANOMALY_FUSION = "anomaly_fusion"
    INCIDENT_FUSION = "incident_fusion"
    FAILURE_PREDICTION = "failure_prediction"
    RUL = "rul"
    HEALTH_SCORE = "health_score"
    SAFETY_ENGINE = "safety_engine"
    RULE_ENGINE = "rule_engine"
    OPERATOR = "operator"
    UNKNOWN = "unknown"


class RiskDimension(str, Enum):
    LIKELIHOOD = "likelihood"
    CONSEQUENCE = "consequence"
    URGENCY = "urgency"
    EXPOSURE = "exposure"
    PERSISTENCE = "persistence"
    CORROBORATION = "corroboration"
    ASSET_CRITICALITY = "asset_criticality"


class RecommendedDisposition(str, Enum):
    NONE = "none"
    MONITOR = "monitor"
    INVESTIGATE = "investigate"
    MAINTENANCE_REVIEW = "maintenance_review"
    SAFETY_REVIEW = "safety_review"
    PRIORITY_REVIEW = "priority_review"
    EMERGENCY_POLICY_REVIEW = "emergency_policy_review"


# ============================================================
# Input Risk Signal
# ============================================================


@dataclass(slots=True)
class RiskSignal:
    signal_id: str
    timestamp: datetime

    source: RiskSource
    category: RiskCategory

    # Normalized 0..1 likelihood / anomaly strength.
    score: float

    # Reliability of the source/model.
    confidence: float

    asset_id: str | None = None
    site_id: str | None = None
    zone_id: str | None = None
    component_id: str | None = None

    source_id: str | None = None

    severity: RiskLevel = RiskLevel.LOW

    # Optional consequence estimate independent of probability.
    consequence: float = 0.50

    # Optional urgency estimate.
    urgency: float = 0.50

    # Optional exposure estimate. Useful for personnel,
    # environmental or production exposure.
    exposure: float = 0.50

    # Whether evidence is persistent rather than transient.
    persistence: float = 0.0

    # Whether upstream evidence has already been independently
    # corroborated.
    corroborated: bool = False

    independent_source_key: str | None = None

    description: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = _clamp(self.score)
        self.confidence = _clamp(self.confidence)
        self.consequence = _clamp(self.consequence)
        self.urgency = _clamp(self.urgency)
        self.exposure = _clamp(self.exposure)
        self.persistence = _clamp(self.persistence)

        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(
                tzinfo=timezone.utc
            )

    @property
    def effective_score(self) -> float:
        return _clamp(
            self.score * self.confidence
        )


# ============================================================
# Asset Context
# ============================================================


@dataclass(slots=True)
class AssetRiskContext:
    asset_id: str | None = None

    # 0..1. 1 means loss/failure has major business/safety impact.
    criticality: float = 0.50

    # 0..1. Higher means more people are exposed.
    personnel_exposure: float = 0.0

    # 0..1.
    production_impact: float = 0.50

    # 0..1.
    environmental_impact: float = 0.0

    # 0..1. Poor health should be closer to 1 here.
    degradation: float = 0.0

    # 0..1. Prior incidents/failures.
    historical_risk: float = 0.0

    redundancy_available: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.criticality = _clamp(self.criticality)
        self.personnel_exposure = _clamp(
            self.personnel_exposure
        )
        self.production_impact = _clamp(
            self.production_impact
        )
        self.environmental_impact = _clamp(
            self.environmental_impact
        )
        self.degradation = _clamp(
            self.degradation
        )
        self.historical_risk = _clamp(
            self.historical_risk
        )


# ============================================================
# Risk Dimension Result
# ============================================================


@dataclass(slots=True)
class RiskDimensionScore:
    dimension: RiskDimension
    score: float
    weight: float
    contribution: float
    description: str = ""


@dataclass(slots=True)
class CategoryRisk:
    category: RiskCategory

    score: float
    confidence: float
    level: RiskLevel

    signal_count: int

    strongest_signal_id: str | None = None

    contributing_sources: list[RiskSource] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Correlation
# ============================================================


@dataclass(slots=True)
class MultimodalRiskCorrelation:
    signal_count: int
    source_count: int
    independent_source_count: int
    category_count: int

    temporal_agreement: float
    source_diversity: float
    category_diversity: float
    confidence_agreement: float

    corroborated: bool
    conflicting: bool

    earliest_timestamp: datetime | None = None
    latest_timestamp: datetime | None = None
    duration_seconds: float = 0.0


# ============================================================
# Final Assessment
# ============================================================


@dataclass(slots=True)
class MultimodalRiskAssessment:
    assessment_id: str
    timestamp: datetime

    asset_id: str | None
    site_id: str | None
    zone_id: str | None
    component_id: str | None

    risk_score: float
    confidence: float

    level: RiskLevel
    state: RiskState

    dominant_category: RiskCategory

    category_risks: list[CategoryRisk]
    dimensions: list[RiskDimensionScore]

    correlation: MultimodalRiskCorrelation

    signals: list[RiskSignal]

    disposition: RecommendedDisposition

    requires_investigation: bool
    requires_maintenance_review: bool
    requires_safety_review: bool
    requires_priority_review: bool
    requires_emergency_policy_review: bool

    rationale: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_high_risk(self) -> bool:
        return self.level >= RiskLevel.HIGH

    @property
    def is_critical(self) -> bool:
        return self.level == RiskLevel.CRITICAL


# ============================================================
# Policy
# ============================================================


@dataclass(slots=True)
class MultimodalRiskPolicy:
    correlation_window_seconds: float = 120.0

    minimum_signal_confidence: float = 0.45
    minimum_signal_score: float = 0.15

    minimum_independent_sources: int = 2

    likelihood_weight: float = 0.28
    consequence_weight: float = 0.20
    urgency_weight: float = 0.14
    exposure_weight: float = 0.10
    persistence_weight: float = 0.08
    corroboration_weight: float = 0.10
    asset_criticality_weight: float = 0.10

    low_threshold: float = 0.25
    medium_threshold: float = 0.45
    high_threshold: float = 0.70
    critical_threshold: float = 0.90

    corroboration_bonus: float = 0.08
    diversity_bonus: float = 0.05

    conflict_penalty: float = 0.12
    low_confidence_penalty: float = 0.08

    max_signals: int = 100


# ============================================================
# Risk Engine
# ============================================================


class MultimodalRiskEngine:
    """
    Multimodal operational risk scoring.

    Combines:
        - sensor anomaly evidence
        - temperature anomalies
        - vibration anomalies
        - electrical anomalies
        - multivariate anomalies
        - CV evidence
        - image/sensor fusion
        - anomaly fusion
        - incident fusion
        - failure prediction
        - RUL
        - asset health
        - deterministic safety evidence

    This module estimates risk.

    It does NOT authorize:
        * emergency shutdown
        * breaker trips
        * machine isolation
        * suppression systems
        * worker evacuation

    Those actions must be evaluated by deterministic safety and
    decision policies.
    """

    def __init__(
        self,
        policy: MultimodalRiskPolicy | None = None,
    ) -> None:
        self.policy = (
            policy
            or MultimodalRiskPolicy()
        )

    # ========================================================
    # Public API
    # ========================================================

    def assess(
        self,
        signals: Iterable[RiskSignal],
        *,
        context: AssetRiskContext | None = None,
        asset_id: str | None = None,
        site_id: str | None = None,
        zone_id: str | None = None,
        component_id: str | None = None,
    ) -> MultimodalRiskAssessment:

        context = (
            context
            or AssetRiskContext(
                asset_id=asset_id
            )
        )

        prepared = self._prepare(
            signals,
            asset_id=asset_id,
            site_id=site_id,
            zone_id=zone_id,
            component_id=component_id,
        )

        prepared = self._deduplicate(
            prepared
        )

        prepared = self._limit_window(
            prepared
        )

        if not prepared:
            return self._empty_assessment(
                context=context,
                asset_id=asset_id,
                site_id=site_id,
                zone_id=zone_id,
                component_id=component_id,
            )

        correlation = self._correlate(
            prepared
        )

        category_risks = (
            self._category_risks(
                prepared
            )
        )

        likelihood = self._likelihood(
            prepared
        )

        consequence = self._consequence(
            prepared,
            context,
        )

        urgency = self._urgency(
            prepared
        )

        exposure = self._exposure(
            prepared,
            context,
        )

        persistence = self._persistence(
            prepared
        )

        corroboration = (
            self._corroboration_score(
                prepared,
                correlation,
            )
        )

        dimensions = self._dimensions(
            likelihood=likelihood,
            consequence=consequence,
            urgency=urgency,
            exposure=exposure,
            persistence=persistence,
            corroboration=corroboration,
            criticality=context.criticality,
        )

        risk_score = sum(
            item.contribution
            for item in dimensions
        )

        if correlation.corroborated:
            risk_score += (
                self.policy
                .corroboration_bonus
            )

        if (
            correlation.source_diversity
            >= 0.50
            and correlation.category_diversity
            >= 0.50
        ):
            risk_score += (
                self.policy.diversity_bonus
            )

        if correlation.conflicting:
            risk_score -= (
                self.policy.conflict_penalty
            )

        confidence = self._confidence(
            prepared,
            correlation,
        )

        if confidence < 0.50:
            risk_score -= (
                self.policy
                .low_confidence_penalty
            )

        risk_score = _clamp(
            risk_score
        )

        level = self._level(
            risk_score,
            prepared,
        )

        dominant_category = (
            category_risks[0].category
            if category_risks
            else RiskCategory.UNKNOWN
        )

        state = self._state(
            level,
            confidence,
        )

        disposition = self._disposition(
            level,
            dominant_category,
            correlation,
        )

        resolved_asset = (
            asset_id
            or context.asset_id
            or _most_common(
                [
                    item.asset_id
                    for item in prepared
                    if item.asset_id
                ]
            )
        )

        resolved_site = (
            site_id
            or _most_common(
                [
                    item.site_id
                    for item in prepared
                    if item.site_id
                ]
            )
        )

        resolved_zone = (
            zone_id
            or _most_common(
                [
                    item.zone_id
                    for item in prepared
                    if item.zone_id
                ]
            )
        )

        resolved_component = (
            component_id
            or _most_common(
                [
                    item.component_id
                    for item in prepared
                    if item.component_id
                ]
            )
        )

        safety_review = (
            level >= RiskLevel.HIGH
            and dominant_category
            in {
                RiskCategory.SAFETY,
                RiskCategory.FIRE,
                RiskCategory.THERMAL,
                RiskCategory.ELECTRICAL,
                RiskCategory.ENVIRONMENTAL,
                RiskCategory.PERSONNEL,
            }
        )

        maintenance_review = (
            level >= RiskLevel.MEDIUM
            and dominant_category
            in {
                RiskCategory.EQUIPMENT,
                RiskCategory.FAILURE,
                RiskCategory.THERMAL,
                RiskCategory.ELECTRICAL,
                RiskCategory.MECHANICAL,
                RiskCategory.MAINTENANCE,
            }
        )

        priority_review = (
            level >= RiskLevel.HIGH
        )

        emergency_policy_review = (
            level == RiskLevel.CRITICAL
            and (
                safety_review
                or correlation.corroborated
            )
        )

        rationale = self._rationale(
            level=level,
            dominant_category=dominant_category,
            confidence=confidence,
            correlation=correlation,
            category_risks=category_risks,
            context=context,
        )

        return MultimodalRiskAssessment(
            assessment_id=str(uuid4()),
            timestamp=max(
                item.timestamp
                for item in prepared
            ),
            asset_id=resolved_asset,
            site_id=resolved_site,
            zone_id=resolved_zone,
            component_id=resolved_component,
            risk_score=risk_score,
            confidence=confidence,
            level=level,
            state=state,
            dominant_category=dominant_category,
            category_risks=category_risks,
            dimensions=dimensions,
            correlation=correlation,
            signals=prepared,
            disposition=disposition,
            requires_investigation=(
                level >= RiskLevel.MEDIUM
            ),
            requires_maintenance_review=(
                maintenance_review
            ),
            requires_safety_review=(
                safety_review
            ),
            requires_priority_review=(
                priority_review
            ),
            requires_emergency_policy_review=(
                emergency_policy_review
            ),
            rationale=rationale,
            metadata={
                "signal_count":
                    len(prepared),
                "asset_criticality":
                    context.criticality,
                "production_impact":
                    context.production_impact,
                "degradation":
                    context.degradation,
            },
        )

    def assess_many(
        self,
        groups: Iterable[
            Iterable[RiskSignal]
        ],
        *,
        contexts: Mapping[
            str,
            AssetRiskContext,
        ]
        | None = None,
    ) -> list[MultimodalRiskAssessment]:

        results: list[
            MultimodalRiskAssessment
        ] = []

        for group in groups:
            materialized = list(group)

            asset_id = _most_common(
                [
                    item.asset_id
                    for item in materialized
                    if item.asset_id
                ]
            )

            context = (
                contexts.get(asset_id)
                if (
                    contexts
                    and asset_id
                )
                else None
            )

            results.append(
                self.assess(
                    materialized,
                    context=context,
                    asset_id=asset_id,
                )
            )

        return results

    # ========================================================
    # Preparation
    # ========================================================

    def _prepare(
        self,
        signals: Iterable[RiskSignal],
        *,
        asset_id: str | None,
        site_id: str | None,
        zone_id: str | None,
        component_id: str | None,
    ) -> list[RiskSignal]:

        output: list[
            RiskSignal
        ] = []

        for signal in signals:

            if (
                signal.confidence
                < self.policy
                .minimum_signal_confidence
            ):
                continue

            if (
                signal.score
                < self.policy
                .minimum_signal_score
            ):
                continue

            if (
                asset_id is not None
                and signal.asset_id
                not in {
                    None,
                    asset_id,
                }
            ):
                continue

            if (
                site_id is not None
                and signal.site_id
                not in {
                    None,
                    site_id,
                }
            ):
                continue

            if (
                zone_id is not None
                and signal.zone_id
                not in {
                    None,
                    zone_id,
                }
            ):
                continue

            if (
                component_id is not None
                and signal.component_id
                not in {
                    None,
                    component_id,
                }
            ):
                continue

            output.append(signal)

        output.sort(
            key=lambda item:
                item.timestamp
        )

        return output

    # ========================================================
    # Deduplication
    # ========================================================

    def _deduplicate(
        self,
        signals: Sequence[
            RiskSignal
        ],
    ) -> list[RiskSignal]:

        selected: dict[
            tuple[Any, ...],
            RiskSignal,
        ] = {}

        for signal in signals:

            key = (
                signal.source,
                signal.category,
                signal.source_id,
                signal.asset_id,
                signal.component_id,
            )

            existing = selected.get(
                key
            )

            if (
                existing is None
                or signal.effective_score
                > existing.effective_score
            ):
                selected[key] = signal

        output = list(
            selected.values()
        )

        output.sort(
            key=lambda item:
                item.timestamp
        )

        return output[
            -self.policy.max_signals :
        ]

    # ========================================================
    # Time Window
    # ========================================================

    def _limit_window(
        self,
        signals: Sequence[
            RiskSignal
        ],
    ) -> list[RiskSignal]:

        if not signals:
            return []

        latest = max(
            item.timestamp
            for item in signals
        )

        return [
            item
            for item in signals
            if (
                latest
                - item.timestamp
            ).total_seconds()
            <= self.policy
            .correlation_window_seconds
        ]

    # ========================================================
    # Correlation
    # ========================================================

    def _correlate(
        self,
        signals: Sequence[
            RiskSignal
        ],
    ) -> MultimodalRiskCorrelation:

        if not signals:
            return (
                MultimodalRiskCorrelation(
                    signal_count=0,
                    source_count=0,
                    independent_source_count=0,
                    category_count=0,
                    temporal_agreement=0.0,
                    source_diversity=0.0,
                    category_diversity=0.0,
                    confidence_agreement=0.0,
                    corroborated=False,
                    conflicting=False,
                )
            )

        earliest = min(
            item.timestamp
            for item in signals
        )

        latest = max(
            item.timestamp
            for item in signals
        )

        duration = (
            latest - earliest
        ).total_seconds()

        temporal = exp(
            -duration
            / max(
                self.policy
                .correlation_window_seconds,
                1e-6,
            )
        )

        sources = {
            item.source
            for item in signals
        }

        categories = {
            item.category
            for item in signals
        }

        independent = {
            (
                item.independent_source_key
                or item.source_id
                or item.signal_id
            )
            for item in signals
        }

        source_diversity = min(
            1.0,
            len(sources) / 4.0,
        )

        category_diversity = min(
            1.0,
            len(categories) / 4.0,
        )

        confidences = [
            item.confidence
            for item in signals
        ]

        confidence_agreement = (
            _agreement(confidences)
        )

        corroborated = (
            len(independent)
            >= self.policy
            .minimum_independent_sources
            and (
                any(
                    item.corroborated
                    for item in signals
                )
                or len(sources) >= 2
            )
        )

        # Large disagreement between strong evidence can indicate
        # uncertainty/model conflict.
        effective = [
            item.effective_score
            for item in signals
        ]

        conflicting = (
            len(effective) >= 2
            and (
                max(effective)
                - min(effective)
            ) >= 0.70
            and confidence_agreement < 0.65
        )

        return MultimodalRiskCorrelation(
            signal_count=len(signals),
            source_count=len(sources),
            independent_source_count=(
                len(independent)
            ),
            category_count=len(
                categories
            ),
            temporal_agreement=_clamp(
                temporal
            ),
            source_diversity=(
                source_diversity
            ),
            category_diversity=(
                category_diversity
            ),
            confidence_agreement=(
                confidence_agreement
            ),
            corroborated=corroborated,
            conflicting=conflicting,
            earliest_timestamp=earliest,
            latest_timestamp=latest,
            duration_seconds=max(
                0.0,
                duration,
            ),
        )

    # ========================================================
    # Category Risk
    # ========================================================

    def _category_risks(
        self,
        signals: Sequence[
            RiskSignal
        ],
    ) -> list[CategoryRisk]:

        grouped: dict[
            RiskCategory,
            list[RiskSignal],
        ] = {}

        for signal in signals:
            grouped.setdefault(
                signal.category,
                [],
            ).append(signal)

        output: list[
            CategoryRisk
        ] = []

        for category, items in (
            grouped.items()
        ):

            ordered = sorted(
                items,
                key=lambda item:
                    item.effective_score,
                reverse=True,
            )

            strongest = ordered[0]

            score = (
                strongest.effective_score
            )

            if len(ordered) > 1:
                supporting = mean(
                    item.effective_score
                    for item in ordered[1:]
                )

                score = _clamp(
                    score * 0.75
                    + supporting * 0.25
                )

            confidence = mean(
                item.confidence
                for item in items
            )

            output.append(
                CategoryRisk(
                    category=category,
                    score=score,
                    confidence=(
                        _clamp(
                            confidence
                        )
                    ),
                    level=self._score_level(
                        score
                    ),
                    signal_count=len(
                        items
                    ),
                    strongest_signal_id=(
                        strongest.signal_id
                    ),
                    contributing_sources=(
                        list(
                            {
                                item.source
                                for item in items
                            }
                        )
                    ),
                )
            )

        output.sort(
            key=lambda item:
                item.score,
            reverse=True,
        )

        return output

    # ========================================================
    # Risk Dimensions
    # ========================================================

    @staticmethod
    def _likelihood(
        signals: Sequence[
            RiskSignal
        ],
    ) -> float:

        scores = sorted(
            (
                item.effective_score
                for item in signals
            ),
            reverse=True,
        )

        strongest = scores[0]

        if len(scores) == 1:
            return strongest

        support = mean(
            scores[1:]
        )

        return _clamp(
            strongest * 0.70
            + support * 0.30
        )

    @staticmethod
    def _consequence(
        signals: Sequence[
            RiskSignal
        ],
        context: AssetRiskContext,
    ) -> float:

        signal_consequence = max(
            (
                item.consequence
                for item in signals
            ),
            default=0.0,
        )

        contextual = (
            context.criticality * 0.40
            + context.production_impact
            * 0.25
            + context.personnel_exposure
            * 0.20
            + context.environmental_impact
            * 0.15
        )

        return _clamp(
            signal_consequence * 0.60
            + contextual * 0.40
        )

    @staticmethod
    def _urgency(
        signals: Sequence[
            RiskSignal
        ],
    ) -> float:

        return max(
            (
                item.urgency
                for item in signals
            ),
            default=0.0,
        )

    @staticmethod
    def _exposure(
        signals: Sequence[
            RiskSignal
        ],
        context: AssetRiskContext,
    ) -> float:

        signal_exposure = max(
            (
                item.exposure
                for item in signals
            ),
            default=0.0,
        )

        contextual = max(
            context.personnel_exposure,
            context.environmental_impact,
            context.production_impact,
        )

        return _clamp(
            signal_exposure * 0.60
            + contextual * 0.40
        )

    @staticmethod
    def _persistence(
        signals: Sequence[
            RiskSignal
        ],
    ) -> float:

        return max(
            (
                item.persistence
                for item in signals
            ),
            default=0.0,
        )

    @staticmethod
    def _corroboration_score(
        signals: Sequence[
            RiskSignal
        ],
        correlation: MultimodalRiskCorrelation,
    ) -> float:

        explicit = (
            mean(
                [
                    1.0
                    if item.corroborated
                    else 0.0
                    for item in signals
                ]
            )
            if signals
            else 0.0
        )

        return _clamp(
            explicit * 0.35
            + correlation.temporal_agreement
            * 0.20
            + correlation.source_diversity
            * 0.20
            + correlation.category_diversity
            * 0.10
            + correlation.confidence_agreement
            * 0.15
        )

    def _dimensions(
        self,
        *,
        likelihood: float,
        consequence: float,
        urgency: float,
        exposure: float,
        persistence: float,
        corroboration: float,
        criticality: float,
    ) -> list[RiskDimensionScore]:

        p = self.policy

        data = [
            (
                RiskDimension.LIKELIHOOD,
                likelihood,
                p.likelihood_weight,
                "Probability/strength of abnormal condition.",
            ),
            (
                RiskDimension.CONSEQUENCE,
                consequence,
                p.consequence_weight,
                "Potential consequence if the condition progresses.",
            ),
            (
                RiskDimension.URGENCY,
                urgency,
                p.urgency_weight,
                "How quickly review or intervention may be required.",
            ),
            (
                RiskDimension.EXPOSURE,
                exposure,
                p.exposure_weight,
                "Personnel, environmental or production exposure.",
            ),
            (
                RiskDimension.PERSISTENCE,
                persistence,
                p.persistence_weight,
                "Persistence of the abnormal condition.",
            ),
            (
                RiskDimension.CORROBORATION,
                corroboration,
                p.corroboration_weight,
                "Independent multimodal evidence agreement.",
            ),
            (
                RiskDimension.ASSET_CRITICALITY,
                criticality,
                p.asset_criticality_weight,
                "Operational importance of the affected asset.",
            ),
        ]

        return [
            RiskDimensionScore(
                dimension=dimension,
                score=_clamp(score),
                weight=weight,
                contribution=_clamp(
                    score
                ) * weight,
                description=description,
            )
            for (
                dimension,
                score,
                weight,
                description,
            ) in data
        ]

    # ========================================================
    # Confidence
    # ========================================================

    @staticmethod
    def _confidence(
        signals: Sequence[
            RiskSignal
        ],
        correlation: MultimodalRiskCorrelation,
    ) -> float:

        if not signals:
            return 0.0

        base = mean(
            item.confidence
            for item in signals
        )

        confidence = (
            base * 0.60
            + correlation
            .confidence_agreement
            * 0.15
            + correlation
            .source_diversity
            * 0.10
            + correlation
            .temporal_agreement
            * 0.10
            + (
                1.0
                if correlation.corroborated
                else 0.0
            ) * 0.05
        )

        if correlation.conflicting:
            confidence -= 0.15

        return _clamp(
            confidence
        )

    # ========================================================
    # Risk Level
    # ========================================================

    def _level(
        self,
        score: float,
        signals: Sequence[
            RiskSignal
        ],
    ) -> RiskLevel:

        calculated = (
            self._score_level(score)
        )

        strongest_input = max(
            (
                item.severity
                for item in signals
            ),
            default=RiskLevel.NORMAL,
        )

        # Upstream deterministic CRITICAL evidence must not be
        # downgraded by averaging.
        if (
            strongest_input
            == RiskLevel.CRITICAL
        ):
            return RiskLevel.CRITICAL

        return max(
            calculated,
            strongest_input,
        )

    def _score_level(
        self,
        score: float,
    ) -> RiskLevel:

        p = self.policy

        if score >= p.critical_threshold:
            return RiskLevel.CRITICAL

        if score >= p.high_threshold:
            return RiskLevel.HIGH

        if score >= p.medium_threshold:
            return RiskLevel.MEDIUM

        if score >= p.low_threshold:
            return RiskLevel.LOW

        return RiskLevel.NORMAL

    # ========================================================
    # State
    # ========================================================

    @staticmethod
    def _state(
        level: RiskLevel,
        confidence: float,
    ) -> RiskState:

        if confidence < 0.35:
            return RiskState.UNCERTAIN

        if level == RiskLevel.CRITICAL:
            return RiskState.CRITICAL

        if level == RiskLevel.HIGH:
            return RiskState.HIGH_RISK

        if level == RiskLevel.MEDIUM:
            return RiskState.ELEVATED

        if level == RiskLevel.LOW:
            return RiskState.WATCH

        return RiskState.NORMAL

    # ========================================================
    # Disposition
    # ========================================================

    @staticmethod
    def _disposition(
        level: RiskLevel,
        category: RiskCategory,
        correlation: MultimodalRiskCorrelation,
    ) -> RecommendedDisposition:

        if level == RiskLevel.NORMAL:
            return RecommendedDisposition.NONE

        if level == RiskLevel.LOW:
            return RecommendedDisposition.MONITOR

        if level == RiskLevel.MEDIUM:
            if category in {
                RiskCategory.MECHANICAL,
                RiskCategory.EQUIPMENT,
                RiskCategory.FAILURE,
                RiskCategory.MAINTENANCE,
            }:
                return (
                    RecommendedDisposition
                    .MAINTENANCE_REVIEW
                )

            return (
                RecommendedDisposition
                .INVESTIGATE
            )

        if level == RiskLevel.HIGH:
            if category in {
                RiskCategory.SAFETY,
                RiskCategory.FIRE,
                RiskCategory.THERMAL,
                RiskCategory.ELECTRICAL,
                RiskCategory.PERSONNEL,
                RiskCategory.ENVIRONMENTAL,
            }:
                return (
                    RecommendedDisposition
                    .SAFETY_REVIEW
                )

            return (
                RecommendedDisposition
                .PRIORITY_REVIEW
            )

        if (
            level == RiskLevel.CRITICAL
            and correlation.corroborated
        ):
            return (
                RecommendedDisposition
                .EMERGENCY_POLICY_REVIEW
            )

        return (
            RecommendedDisposition
            .PRIORITY_REVIEW
        )

    # ========================================================
    # Rationale
    # ========================================================

    @staticmethod
    def _rationale(
        *,
        level: RiskLevel,
        dominant_category: RiskCategory,
        confidence: float,
        correlation: MultimodalRiskCorrelation,
        category_risks: Sequence[
            CategoryRisk
        ],
        context: AssetRiskContext,
    ) -> list[str]:

        rationale = [
            (
                f"Overall risk classified as "
                f"{level.name}."
            ),
            (
                f"Dominant risk category: "
                f"{dominant_category.value}."
            ),
            (
                f"Assessment confidence: "
                f"{confidence:.2f}."
            ),
        ]

        if correlation.corroborated:
            rationale.append(
                (
                    "Risk is corroborated by "
                    f"{correlation.independent_source_count} "
                    "independent evidence sources."
                )
            )

        if correlation.conflicting:
            rationale.append(
                "Evidence disagreement detected; human review is recommended."
            )

        if category_risks:
            strongest = category_risks[0]

            rationale.append(
                (
                    f"Strongest category score is "
                    f"{strongest.score:.2f} from "
                    f"{strongest.signal_count} signal(s)."
                )
            )

        if context.criticality >= 0.80:
            rationale.append(
                "Affected asset has high operational criticality."
            )

        if context.personnel_exposure >= 0.70:
            rationale.append(
                "Elevated personnel exposure increases consequence risk."
            )

        if context.degradation >= 0.70:
            rationale.append(
                "Asset degradation context increases failure concern."
            )

        return rationale

    # ========================================================
    # Empty Result
    # ========================================================

    def _empty_assessment(
        self,
        *,
        context: AssetRiskContext,
        asset_id: str | None,
        site_id: str | None,
        zone_id: str | None,
        component_id: str | None,
    ) -> MultimodalRiskAssessment:

        return MultimodalRiskAssessment(
            assessment_id=str(uuid4()),
            timestamp=datetime.now(
                timezone.utc
            ),
            asset_id=(
                asset_id
                or context.asset_id
            ),
            site_id=site_id,
            zone_id=zone_id,
            component_id=component_id,
            risk_score=0.0,
            confidence=0.0,
            level=RiskLevel.NORMAL,
            state=RiskState.UNCERTAIN,
            dominant_category=(
                RiskCategory.UNKNOWN
            ),
            category_risks=[],
            dimensions=[],
            correlation=(
                MultimodalRiskCorrelation(
                    signal_count=0,
                    source_count=0,
                    independent_source_count=0,
                    category_count=0,
                    temporal_agreement=0.0,
                    source_diversity=0.0,
                    category_diversity=0.0,
                    confidence_agreement=0.0,
                    corroborated=False,
                    conflicting=False,
                )
            ),
            signals=[],
            disposition=(
                RecommendedDisposition.NONE
            ),
            requires_investigation=False,
            requires_maintenance_review=False,
            requires_safety_review=False,
            requires_priority_review=False,
            requires_emergency_policy_review=False,
            rationale=[
                "Insufficient multimodal evidence for risk assessment."
            ],
        )


# ============================================================
# Constructors / Adapters
# ============================================================


def create_risk_signal(
    source: RiskSource,
    category: RiskCategory,
    score: float,
    confidence: float,
    *,
    severity: RiskLevel = RiskLevel.LOW,
    consequence: float = 0.50,
    urgency: float = 0.50,
    exposure: float = 0.50,
    persistence: float = 0.0,
    corroborated: bool = False,
    asset_id: str | None = None,
    site_id: str | None = None,
    zone_id: str | None = None,
    component_id: str | None = None,
    source_id: str | None = None,
    independent_source_key: str | None = None,
    description: str = "",
    timestamp: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> RiskSignal:

    return RiskSignal(
        signal_id=str(uuid4()),
        timestamp=(
            timestamp
            or datetime.now(
                timezone.utc
            )
        ),
        source=source,
        category=category,
        score=score,
        confidence=confidence,
        asset_id=asset_id,
        site_id=site_id,
        zone_id=zone_id,
        component_id=component_id,
        source_id=source_id,
        severity=severity,
        consequence=consequence,
        urgency=urgency,
        exposure=exposure,
        persistence=persistence,
        corroborated=corroborated,
        independent_source_key=(
            independent_source_key
        ),
        description=description,
        metadata=metadata or {},
    )


def temperature_risk_signal(
    score: float,
    confidence: float,
    *,
    asset_id: str | None = None,
    sensor_id: str | None = None,
    component_id: str | None = None,
    zone_id: str | None = None,
    severity: RiskLevel = RiskLevel.MEDIUM,
    persistence: float = 0.0,
    timestamp: datetime | None = None,
) -> RiskSignal:

    return create_risk_signal(
        RiskSource.TEMPERATURE_ANOMALY,
        RiskCategory.THERMAL,
        score,
        confidence,
        severity=severity,
        consequence=0.70,
        urgency=0.65,
        persistence=persistence,
        asset_id=asset_id,
        component_id=component_id,
        zone_id=zone_id,
        source_id=sensor_id,
        independent_source_key=(
            f"temperature:{sensor_id}"
            if sensor_id
            else None
        ),
        description=(
            "Temperature anomaly risk"
        ),
        timestamp=timestamp,
    )


def vibration_risk_signal(
    score: float,
    confidence: float,
    *,
    asset_id: str | None = None,
    sensor_id: str | None = None,
    component_id: str | None = None,
    severity: RiskLevel = RiskLevel.MEDIUM,
    persistence: float = 0.0,
    timestamp: datetime | None = None,
) -> RiskSignal:

    return create_risk_signal(
        RiskSource.VIBRATION_ANOMALY,
        RiskCategory.MECHANICAL,
        score,
        confidence,
        severity=severity,
        consequence=0.60,
        urgency=0.50,
        persistence=persistence,
        asset_id=asset_id,
        component_id=component_id,
        source_id=sensor_id,
        independent_source_key=(
            f"vibration:{sensor_id}"
            if sensor_id
            else None
        ),
        description=(
            "Vibration anomaly risk"
        ),
        timestamp=timestamp,
    )


def electrical_risk_signal(
    score: float,
    confidence: float,
    *,
    asset_id: str | None = None,
    sensor_id: str | None = None,
    component_id: str | None = None,
    severity: RiskLevel = RiskLevel.MEDIUM,
    timestamp: datetime | None = None,
) -> RiskSignal:

    return create_risk_signal(
        RiskSource.ELECTRICAL_ANOMALY,
        RiskCategory.ELECTRICAL,
        score,
        confidence,
        severity=severity,
        consequence=0.75,
        urgency=0.70,
        asset_id=asset_id,
        component_id=component_id,
        source_id=sensor_id,
        independent_source_key=(
            f"electrical:{sensor_id}"
            if sensor_id
            else None
        ),
        description=(
            "Electrical anomaly risk"
        ),
        timestamp=timestamp,
    )


def incident_risk_signal(
    score: float,
    confidence: float,
    *,
    category: RiskCategory,
    incident_id: str,
    asset_id: str | None = None,
    site_id: str | None = None,
    zone_id: str | None = None,
    component_id: str | None = None,
    severity: RiskLevel = RiskLevel.HIGH,
    corroborated: bool = True,
    timestamp: datetime | None = None,
) -> RiskSignal:

    return create_risk_signal(
        RiskSource.INCIDENT_FUSION,
        category,
        score,
        confidence,
        severity=severity,
        consequence=0.80,
        urgency=0.80,
        exposure=0.60,
        persistence=0.50,
        corroborated=corroborated,
        asset_id=asset_id,
        site_id=site_id,
        zone_id=zone_id,
        component_id=component_id,
        source_id=incident_id,
        independent_source_key=(
            f"incident:{incident_id}"
        ),
        description=(
            "Fused incident risk evidence"
        ),
        timestamp=timestamp,
    )


# ============================================================
# Helpers
# ============================================================


def _agreement(
    values: Sequence[float],
) -> float:

    if not values:
        return 0.0

    if len(values) == 1:
        return values[0]

    average = mean(values)

    deviation = mean(
        abs(
            value - average
        )
        for value in values
    )

    return _clamp(
        1.0 - deviation
    )


def _most_common(
    values: Sequence[str],
) -> str | None:

    if not values:
        return None

    counts: dict[str, int] = {}

    for value in values:
        counts[value] = (
            counts.get(
                value,
                0,
            )
            + 1
        )

    return max(
        counts,
        key=counts.get,  # type: ignore[arg-type]
    )


def _valid_number(
    value: Any,
) -> bool:

    try:
        return isfinite(
            float(value)
        )
    except (
        TypeError,
        ValueError,
    ):
        return False


def _clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:

    if not _valid_number(
        value
    ):
        return minimum

    return max(
        minimum,
        min(
            maximum,
            float(value),
        ),
    )


# ============================================================
# Default Engine
# ============================================================


default_multimodal_risk_engine = (
    MultimodalRiskEngine()
)


__all__ = [
    "RiskLevel",
    "RiskState",
    "RiskCategory",
    "RiskSource",
    "RiskDimension",
    "RecommendedDisposition",
    "RiskSignal",
    "AssetRiskContext",
    "RiskDimensionScore",
    "CategoryRisk",
    "MultimodalRiskCorrelation",
    "MultimodalRiskAssessment",
    "MultimodalRiskPolicy",
    "MultimodalRiskEngine",
    "create_risk_signal",
    "temperature_risk_signal",
    "vibration_risk_signal",
    "electrical_risk_signal",
    "incident_risk_signal",
    "default_multimodal_risk_engine",
]