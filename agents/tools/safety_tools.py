# app/agents/tools/safety_tools.py

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid4


# =========================================================
# Utilities
# =========================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    return max(low, min(high, float(value)))


def finite(
    value: Any,
    *,
    default: float = 0.0,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(result):
        return default

    return result


def ensure_utc(
    value: datetime | None,
) -> datetime | None:
    if value is None:
        return None

    if value.tzinfo is None:
        return value.replace(
            tzinfo=timezone.utc
        )

    return value.astimezone(timezone.utc)


def json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    if is_dataclass(value):
        return json_safe(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            json_safe(item)
            for item in value
        ]

    if (
        isinstance(value, float)
        and not math.isfinite(value)
    ):
        return None

    return value


# =========================================================
# Enums
# =========================================================

class HazardType(str, Enum):
    FIRE = "fire"
    SMOKE = "smoke"
    OVERHEATING = "overheating"

    ELECTRICAL = "electrical"
    ARC_FLASH = "arc_flash"

    GAS_LEAK = "gas_leak"
    CHEMICAL_LEAK = "chemical_leak"
    FLUID_LEAK = "fluid_leak"

    EXPLOSION_RISK = "explosion_risk"

    HIGH_PRESSURE = "high_pressure"
    PRESSURE_LOSS = "pressure_loss"

    HYDRAULIC_FAILURE = "hydraulic_failure"
    PNEUMATIC_FAILURE = "pneumatic_failure"

    MACHINE_FAILURE = "machine_failure"
    STRUCTURAL_FAILURE = "structural_failure"

    ROTATING_EQUIPMENT = "rotating_equipment"
    MOVING_MACHINERY = "moving_machinery"

    WORKER_INJURY = "worker_injury"
    WORKER_FALL = "worker_fall"

    UNSAFE_PROXIMITY = "unsafe_proximity"
    RESTRICTED_ZONE = "restricted_zone"
    PPE_VIOLATION = "ppe_violation"

    TEMPERATURE = "temperature"
    VIBRATION = "vibration"

    UNKNOWN = "unknown"


class SafetySeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class SafetyRiskLevel(str, Enum):
    LOW = "low"
    GUARDED = "guarded"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class SafetyStatus(str, Enum):
    SAFE = "safe"
    MONITOR = "monitor"
    WARNING = "warning"
    DANGER = "danger"
    EMERGENCY = "emergency"
    UNKNOWN = "unknown"


class SafetyActionType(str, Enum):
    MONITOR = "monitor"
    VERIFY_SENSOR = "verify_sensor"

    INSPECT = "inspect"
    MANUAL_INSPECTION = "manual_inspection"

    NOTIFY_OPERATOR = "notify_operator"
    NOTIFY_SUPERVISOR = "notify_supervisor"
    NOTIFY_SAFETY_TEAM = "notify_safety_team"

    CREATE_INCIDENT = "create_incident"

    RESTRICT_ACCESS = "restrict_access"
    ESTABLISH_EXCLUSION_ZONE = (
        "establish_exclusion_zone"
    )

    PPE_CORRECTION = "ppe_correction"

    EVACUATE_ZONE = "evacuate_zone"
    EVACUATE_SITE = "evacuate_site"

    FIRE_RESPONSE = "fire_response"
    MEDICAL_RESPONSE = "medical_response"

    ISOLATION_REVIEW = "isolation_review"
    LOCKOUT_TAGOUT_REVIEW = (
        "lockout_tagout_review"
    )

    REQUEST_LOAD_REDUCTION = (
        "request_load_reduction"
    )

    REQUEST_MACHINE_STOP = (
        "request_machine_stop"
    )

    REQUEST_POWER_ISOLATION = (
        "request_power_isolation"
    )

    REQUEST_PROCESS_ISOLATION = (
        "request_process_isolation"
    )

    EMERGENCY_RESPONSE = "emergency_response"

    HUMAN_REVIEW = "human_review"


class SafetyActionAuthority(str, Enum):
    ADVISORY = "advisory"
    OPERATOR = "operator"
    SUPERVISOR = "supervisor"
    SAFETY_TEAM = "safety_team"

    DETERMINISTIC_CONTROLLER = (
        "deterministic_controller"
    )

    SIS = "safety_instrumented_system"

    EMERGENCY_TEAM = "emergency_team"


class EvidenceSource(str, Enum):
    TELEMETRY = "telemetry"
    COMPUTER_VISION = "computer_vision"
    THERMAL_CAMERA = "thermal_camera"

    PLC = "plc"
    SCADA = "scada"

    FIRE_DETECTOR = "fire_detector"
    SMOKE_DETECTOR = "smoke_detector"
    GAS_DETECTOR = "gas_detector"

    FAILURE_MODEL = "failure_model"
    ANOMALY_MODEL = "anomaly_model"

    INCIDENT = "incident"

    OPERATOR = "operator"
    TECHNICIAN = "technician"

    SAFETY_SYSTEM = "safety_system"

    OTHER = "other"


class SafetyDataCoverage(str, Enum):
    COMPLETE = "complete"
    GOOD = "good"
    LIMITED = "limited"
    INSUFFICIENT = "insufficient"


# =========================================================
# Evidence
# =========================================================

@dataclass(slots=True)
class SafetyEvidence:
    evidence_id: str

    hazard_type: HazardType

    source: EvidenceSource

    description: str

    severity: float
    confidence: float

    timestamp: datetime

    asset_id: str | None = None
    site_id: str | None = None
    zone_id: str | None = None

    component: str | None = None

    value: float | None = None
    unit: str | None = None

    critical: bool = False

    verified: bool = False

    provenance_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.severity = clamp(self.severity)
        self.confidence = clamp(self.confidence)

        self.timestamp = (
            ensure_utc(self.timestamp)
            or utc_now()
        )


# =========================================================
# Asset/site context
# =========================================================

@dataclass(slots=True)
class SafetyAssetContext:
    asset_id: str

    asset_name: str | None = None
    asset_type: str | None = None

    site_id: str | None = None
    zone_id: str | None = None

    operating: bool | None = None

    safety_critical: bool = False
    production_critical: bool = False

    worker_present: bool | None = None

    hazardous_material_present: bool = False
    combustible_material_present: bool = False

    high_voltage: bool = False
    high_pressure: bool = False

    rotating_equipment: bool = False
    moving_machinery: bool = False

    emergency_shutdown_available: bool | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class SafetySiteContext:
    site_id: str | None = None
    zone_id: str | None = None

    occupied: bool | None = None

    worker_count: int | None = None

    evacuation_required: bool = False

    fire_system_available: bool | None = None
    ventilation_available: bool | None = None

    emergency_team_available: bool | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Safety action
# =========================================================

@dataclass(slots=True)
class SafetyAction:
    action_id: str

    action_type: SafetyActionType

    description: str

    priority: int

    authority: SafetyActionAuthority

    immediate: bool = False

    requires_human_approval: bool = False

    requires_deterministic_controller: bool = False

    asset_id: str | None = None
    zone_id: str | None = None

    reason: str | None = None

    executed: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Risk factor
# =========================================================

@dataclass(slots=True)
class SafetyRiskFactor:
    hazard_type: HazardType

    score: float

    confidence: float

    weighted_score: float

    critical: bool

    evidence_count: int

    description: str


# =========================================================
# Safety assessment
# =========================================================

@dataclass(slots=True)
class SafetyAssessment:
    assessment_id: str

    asset_id: str

    site_id: str | None
    zone_id: str | None

    status: SafetyStatus

    risk_level: SafetyRiskLevel

    physical_risk_score: float

    response_priority_score: float

    confidence: float

    data_coverage: SafetyDataCoverage

    dominant_hazard: HazardType | None

    factors: list[SafetyRiskFactor]

    evidence: list[SafetyEvidence]

    actions: list[SafetyAction]

    incident_required: bool

    emergency_response_required: bool

    machine_stop_review_required: bool

    evacuation_review_required: bool

    human_review_required: bool

    generated_at: datetime = field(
        default_factory=utc_now
    )

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# =========================================================
# Safety event
# =========================================================

@dataclass(slots=True)
class SafetyEventRequest:
    event_id: str

    idempotency_key: str

    asset_id: str

    site_id: str | None
    zone_id: str | None

    hazard_type: HazardType

    risk_level: SafetyRiskLevel

    description: str

    occurred_at: datetime

    evidence_ids: list[str]

    trace_id: str | None = None
    request_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Config
# =========================================================

@dataclass(slots=True)
class SafetyToolsConfig:
    stale_evidence_seconds: float = 120.0

    severely_stale_seconds: float = 600.0

    minimum_confidence: float = 0.50

    elevated_threshold: float = 0.40
    high_threshold: float = 0.60
    critical_threshold: float = 0.80
    emergency_threshold: float = 0.95

    fire_emergency_threshold: float = 0.80
    explosion_emergency_threshold: float = 0.75

    worker_hazard_critical_threshold: float = 0.80

    require_independent_verification: bool = True

    minimum_independent_sources: int = 2


# =========================================================
# Safety tools
# =========================================================

class SafetyTools:

    def __init__(
        self,
        *,
        config: SafetyToolsConfig | None = None,
    ) -> None:
        self.config = (
            config or SafetyToolsConfig()
        )

    # =====================================================
    # Evidence
    # =====================================================

    def validate_evidence(
        self,
        evidence: SafetyEvidence,
        *,
        now: datetime | None = None,
    ) -> tuple[bool, list[str]]:

        now = ensure_utc(now) or utc_now()

        warnings: list[str] = []

        age = max(
            0.0,
            (
                now - evidence.timestamp
            ).total_seconds(),
        )

        valid = True

        if (
            age
            > self.config
            .severely_stale_seconds
        ):
            valid = False

            warnings.append(
                "Safety evidence is severely stale."
            )

        elif (
            age
            > self.config
            .stale_evidence_seconds
        ):
            warnings.append(
                "Safety evidence is stale."
            )

        if (
            evidence.confidence
            < self.config.minimum_confidence
        ):
            warnings.append(
                "Safety evidence has low confidence."
            )

        return valid, warnings

    def deduplicate_evidence(
        self,
        evidence: Sequence[SafetyEvidence],
    ) -> list[SafetyEvidence]:

        result: dict[
            tuple[
                str,
                str,
                str | None,
                str | None,
            ],
            SafetyEvidence,
        ] = {}

        provenance_seen: set[str] = set()

        for item in evidence:
            if item.provenance_id:
                if item.provenance_id in provenance_seen:
                    continue

                provenance_seen.add(
                    item.provenance_id
                )

            key = (
                item.hazard_type.value,
                item.source.value,
                item.asset_id,
                item.component,
            )

            existing = result.get(key)

            if existing is None:
                result[key] = item
                continue

            existing_score = (
                existing.severity
                * existing.confidence
            )

            new_score = (
                item.severity
                * item.confidence
            )

            if new_score > existing_score:
                result[key] = item

        return list(result.values())

    # =====================================================
    # Correlation
    # =====================================================

    @staticmethod
    def filter_context(
        *,
        asset: SafetyAssetContext,
        evidence: Sequence[SafetyEvidence],
    ) -> list[SafetyEvidence]:

        result: list[SafetyEvidence] = []

        for item in evidence:
            if (
                item.asset_id is not None
                and item.asset_id
                != asset.asset_id
            ):
                continue

            if (
                asset.site_id is not None
                and item.site_id is not None
                and item.site_id
                != asset.site_id
            ):
                continue

            if (
                asset.zone_id is not None
                and item.zone_id is not None
                and item.zone_id
                != asset.zone_id
            ):
                continue

            result.append(item)

        return result

    # =====================================================
    # Data coverage
    # =====================================================

    def calculate_coverage(
        self,
        evidence: Sequence[SafetyEvidence],
    ) -> tuple[SafetyDataCoverage, float]:

        if not evidence:
            return (
                SafetyDataCoverage.INSUFFICIENT,
                0.0,
            )

        independent_sources = {
            item.source
            for item in evidence
        }

        verified = sum(
            item.verified
            for item in evidence
        )

        source_score = clamp(
            len(independent_sources) / 4.0
        )

        verified_score = clamp(
            verified / max(len(evidence), 1)
        )

        confidence_score = (
            sum(
                item.confidence
                for item in evidence
            )
            / len(evidence)
        )

        score = clamp(
            0.45 * source_score
            + 0.20 * verified_score
            + 0.35 * confidence_score
        )

        if score >= 0.85:
            level = SafetyDataCoverage.COMPLETE

        elif score >= 0.60:
            level = SafetyDataCoverage.GOOD

        elif score >= 0.30:
            level = SafetyDataCoverage.LIMITED

        else:
            level = (
                SafetyDataCoverage.INSUFFICIENT
            )

        return level, round(score, 4)

    # =====================================================
    # Hazard factors
    # =====================================================

    def calculate_factors(
        self,
        evidence: Sequence[SafetyEvidence],
    ) -> list[SafetyRiskFactor]:

        grouped: dict[
            HazardType,
            list[SafetyEvidence],
        ] = {}

        for item in evidence:
            grouped.setdefault(
                item.hazard_type,
                [],
            ).append(item)

        factors: list[SafetyRiskFactor] = []

        for hazard, items in grouped.items():
            scores = [
                item.severity
                * item.confidence
                for item in items
            ]

            strongest = max(scores)

            # Multiple independent observations
            # strengthen confidence without allowing
            # simple duplication to push risk > 1.
            independent_sources = {
                item.source
                for item in items
            }

            source_bonus = min(
                0.15,
                max(
                    0,
                    len(independent_sources) - 1,
                )
                * 0.05,
            )

            verified_bonus = (
                0.05
                if any(
                    item.verified
                    for item in items
                )
                else 0.0
            )

            score = clamp(
                strongest
                + source_bonus
                + verified_bonus
            )

            confidence = clamp(
                sum(
                    item.confidence
                    for item in items
                )
                / len(items)
            )

            critical = any(
                item.critical
                for item in items
            )

            if critical:
                score = max(score, 0.90)

            factors.append(
                SafetyRiskFactor(
                    hazard_type=hazard,

                    score=round(score, 4),

                    confidence=round(
                        confidence,
                        4,
                    ),

                    weighted_score=round(
                        score * confidence,
                        4,
                    ),

                    critical=critical,

                    evidence_count=len(items),

                    description=(
                        f"{hazard.value} hazard "
                        f"supported by "
                        f"{len(items)} evidence "
                        f"item(s)."
                    ),
                )
            )

        return sorted(
            factors,
            key=lambda item: item.score,
            reverse=True,
        )

    # =====================================================
    # Independent verification
    # =====================================================

    def independent_verification(
        self,
        *,
        hazard: HazardType,
        evidence: Sequence[SafetyEvidence],
    ) -> dict[str, Any]:

        matching = [
            item
            for item in evidence
            if item.hazard_type == hazard
        ]

        sources = {
            item.source
            for item in matching
        }

        verified_items = [
            item
            for item in matching
            if item.verified
        ]

        independently_verified = (
            len(sources)
            >= self.config
            .minimum_independent_sources
            or bool(verified_items)
        )

        return {
            "hazard_type": hazard.value,

            "independently_verified":
                independently_verified,

            "independent_source_count":
                len(sources),

            "sources": sorted(
                item.value
                for item in sources
            ),

            "verified_evidence_count":
                len(verified_items),
        }

    # =====================================================
    # Risk
    # =====================================================

    def calculate_risk(
        self,
        *,
        asset: SafetyAssetContext,
        site: SafetySiteContext | None,
        evidence: Sequence[SafetyEvidence],
    ) -> tuple[
        float,
        float,
        SafetyRiskLevel,
        SafetyStatus,
        HazardType | None,
    ]:

        factors = self.calculate_factors(
            evidence
        )

        if not factors:
            return (
                0.0,
                0.0,
                SafetyRiskLevel.LOW,
                SafetyStatus.UNKNOWN,
                None,
            )

        dominant = factors[0]

        # Physical safety risk should not be reduced
        # because the asset is commercially
        # unimportant.
        physical_risk = dominant.score

        if len(factors) > 1:
            secondary = max(
                item.score
                for item in factors[1:]
            )

            physical_risk = clamp(
                physical_risk
                + 0.10 * secondary
            )

        # Deterministic hazard overrides.
        for factor in factors:
            if factor.critical:
                physical_risk = max(
                    physical_risk,
                    0.90,
                )

            if (
                factor.hazard_type
                == HazardType.FIRE
                and factor.score
                >= self.config
                .fire_emergency_threshold
            ):
                physical_risk = max(
                    physical_risk,
                    0.95,
                )

            if (
                factor.hazard_type
                == HazardType.EXPLOSION_RISK
                and factor.score
                >= self.config
                .explosion_emergency_threshold
            ):
                physical_risk = max(
                    physical_risk,
                    0.98,
                )

            if (
                factor.hazard_type
                in {
                    HazardType.WORKER_INJURY,
                    HazardType.WORKER_FALL,
                    HazardType.UNSAFE_PROXIMITY,
                }
                and factor.score
                >= self.config
                .worker_hazard_critical_threshold
            ):
                physical_risk = max(
                    physical_risk,
                    0.90,
                )

        worker_exposure = (
            asset.worker_present is True
            or (
                site is not None
                and (
                    site.occupied is True
                    or (
                        site.worker_count is not None
                        and site.worker_count > 0
                    )
                )
            )
        )

        # Response priority is separate from physical
        # risk and may increase due to worker exposure,
        # hazardous materials, etc.
        response_priority = physical_risk

        if worker_exposure:
            response_priority += 0.08

        if asset.hazardous_material_present:
            response_priority += 0.08

        if asset.combustible_material_present:
            response_priority += 0.05

        if asset.high_voltage:
            response_priority += 0.03

        if asset.high_pressure:
            response_priority += 0.03

        if asset.safety_critical:
            response_priority += 0.05

        response_priority = clamp(
            response_priority
        )

        if (
            physical_risk
            >= self.config.emergency_threshold
        ):
            level = SafetyRiskLevel.EMERGENCY
            status = SafetyStatus.EMERGENCY

        elif (
            physical_risk
            >= self.config.critical_threshold
        ):
            level = SafetyRiskLevel.CRITICAL
            status = SafetyStatus.DANGER

        elif (
            physical_risk
            >= self.config.high_threshold
        ):
            level = SafetyRiskLevel.HIGH
            status = SafetyStatus.DANGER

        elif (
            physical_risk
            >= self.config.elevated_threshold
        ):
            level = SafetyRiskLevel.ELEVATED
            status = SafetyStatus.WARNING

        elif physical_risk >= 0.20:
            level = SafetyRiskLevel.GUARDED
            status = SafetyStatus.MONITOR

        else:
            level = SafetyRiskLevel.LOW
            status = SafetyStatus.SAFE

        return (
            round(physical_risk, 4),
            round(response_priority, 4),
            level,
            status,
            dominant.hazard_type,
        )

    # =====================================================
    # Recommended actions
    # =====================================================

    def recommend_actions(
        self,
        *,
        asset: SafetyAssetContext,
        site: SafetySiteContext | None,
        risk_level: SafetyRiskLevel,
        dominant_hazard: HazardType | None,
        evidence: Sequence[SafetyEvidence],
    ) -> list[SafetyAction]:

        actions: list[SafetyAction] = []

        def add(
            action_type: SafetyActionType,
            description: str,
            *,
            priority: int,
            authority: SafetyActionAuthority,
            immediate: bool = False,
            human: bool = False,
            controller: bool = False,
            reason: str | None = None,
        ) -> None:
            actions.append(
                SafetyAction(
                    action_id=new_id(),

                    action_type=action_type,

                    description=description,

                    priority=max(
                        1,
                        min(priority, 100),
                    ),

                    authority=authority,

                    immediate=immediate,

                    requires_human_approval=human,

                    requires_deterministic_controller=(
                        controller
                    ),

                    asset_id=asset.asset_id,
                    zone_id=asset.zone_id,

                    reason=reason,
                )
            )

        if not evidence:
            add(
                SafetyActionType.VERIFY_SENSOR,
                (
                    "Collect or verify safety "
                    "telemetry before declaring the "
                    "asset safe."
                ),
                priority=60,
                authority=(
                    SafetyActionAuthority.OPERATOR
                ),
                human=True,
            )

            return actions

        add(
            SafetyActionType.NOTIFY_OPERATOR,
            (
                "Notify the responsible operator "
                "of the detected safety condition."
            ),
            priority=50,
            authority=(
                SafetyActionAuthority.OPERATOR
            ),
        )

        if risk_level in {
            SafetyRiskLevel.HIGH,
            SafetyRiskLevel.CRITICAL,
            SafetyRiskLevel.EMERGENCY,
        }:
            add(
                SafetyActionType.CREATE_INCIDENT,
                (
                    "Create or update the canonical "
                    "safety incident."
                ),
                priority=85,
                authority=(
                    SafetyActionAuthority
                    .SUPERVISOR
                ),
                immediate=True,
            )

            add(
                SafetyActionType.NOTIFY_SAFETY_TEAM,
                (
                    "Notify the site safety team "
                    "for immediate review."
                ),
                priority=90,
                authority=(
                    SafetyActionAuthority
                    .SAFETY_TEAM
                ),
                immediate=True,
            )

        if dominant_hazard in {
            HazardType.FIRE,
            HazardType.SMOKE,
            HazardType.EXPLOSION_RISK,
            HazardType.GAS_LEAK,
            HazardType.CHEMICAL_LEAK,
        }:
            add(
                SafetyActionType
                .ESTABLISH_EXCLUSION_ZONE,
                (
                    "Establish an exclusion zone "
                    "around the affected area."
                ),
                priority=90,
                authority=(
                    SafetyActionAuthority
                    .SAFETY_TEAM
                ),
                immediate=True,
            )

        if dominant_hazard in {
            HazardType.FIRE,
            HazardType.SMOKE,
        }:
            add(
                SafetyActionType.FIRE_RESPONSE,
                (
                    "Activate the approved site fire "
                    "response procedure."
                ),
                priority=100,
                authority=(
                    SafetyActionAuthority
                    .EMERGENCY_TEAM
                ),
                immediate=True,
            )

        if dominant_hazard in {
            HazardType.WORKER_INJURY,
            HazardType.WORKER_FALL,
        }:
            add(
                SafetyActionType.MEDICAL_RESPONSE,
                (
                    "Request the approved site "
                    "medical response procedure."
                ),
                priority=100,
                authority=(
                    SafetyActionAuthority
                    .EMERGENCY_TEAM
                ),
                immediate=True,
            )

        if dominant_hazard == (
            HazardType.PPE_VIOLATION
        ):
            add(
                SafetyActionType.PPE_CORRECTION,
                (
                    "Correct PPE non-compliance "
                    "before hazardous work continues."
                ),
                priority=70,
                authority=(
                    SafetyActionAuthority
                    .SAFETY_TEAM
                ),
                human=True,
            )

        if dominant_hazard == (
            HazardType.RESTRICTED_ZONE
        ):
            add(
                SafetyActionType.RESTRICT_ACCESS,
                (
                    "Restrict unauthorized access "
                    "to the hazardous zone."
                ),
                priority=85,
                authority=(
                    SafetyActionAuthority
                    .SAFETY_TEAM
                ),
                immediate=True,
            )

        if dominant_hazard in {
            HazardType.ELECTRICAL,
            HazardType.ARC_FLASH,
        }:
            add(
                SafetyActionType
                .REQUEST_POWER_ISOLATION,
                (
                    "Request electrical isolation "
                    "through the approved deterministic "
                    "control and lockout procedure."
                ),
                priority=95,
                authority=(
                    SafetyActionAuthority
                    .DETERMINISTIC_CONTROLLER
                ),
                immediate=True,
                human=True,
                controller=True,
            )

            add(
                SafetyActionType
                .LOCKOUT_TAGOUT_REVIEW,
                (
                    "Require qualified lockout/tagout "
                    "review before maintenance."
                ),
                priority=95,
                authority=(
                    SafetyActionAuthority
                    .SAFETY_TEAM
                ),
                human=True,
            )

        if dominant_hazard in {
            HazardType.GAS_LEAK,
            HazardType.CHEMICAL_LEAK,
            HazardType.FLUID_LEAK,
            HazardType.HIGH_PRESSURE,
            HazardType.HYDRAULIC_FAILURE,
            HazardType.PNEUMATIC_FAILURE,
        }:
            add(
                SafetyActionType
                .REQUEST_PROCESS_ISOLATION,
                (
                    "Request process isolation using "
                    "the approved deterministic "
                    "isolation procedure."
                ),
                priority=95,
                authority=(
                    SafetyActionAuthority
                    .DETERMINISTIC_CONTROLLER
                ),
                immediate=True,
                human=True,
                controller=True,
            )

        machine_hazards = {
            HazardType.OVERHEATING,
            HazardType.MACHINE_FAILURE,
            HazardType.STRUCTURAL_FAILURE,
            HazardType.ROTATING_EQUIPMENT,
            HazardType.MOVING_MACHINERY,
            HazardType.UNSAFE_PROXIMITY,
            HazardType.VIBRATION,
        }

        if (
            dominant_hazard in machine_hazards
            and risk_level
            in {
                SafetyRiskLevel.CRITICAL,
                SafetyRiskLevel.EMERGENCY,
            }
        ):
            add(
                SafetyActionType
                .REQUEST_MACHINE_STOP,
                (
                    "Request safe machine stop "
                    "through the authorized "
                    "deterministic safety-control "
                    "path."
                ),
                priority=100,
                authority=(
                    SafetyActionAuthority
                    .DETERMINISTIC_CONTROLLER
                ),
                immediate=True,
                human=True,
                controller=True,
            )

        if (
            risk_level
            == SafetyRiskLevel.EMERGENCY
        ):
            add(
                SafetyActionType
                .EMERGENCY_RESPONSE,
                (
                    "Activate the approved emergency "
                    "response workflow."
                ),
                priority=100,
                authority=(
                    SafetyActionAuthority
                    .EMERGENCY_TEAM
                ),
                immediate=True,
            )

            occupied = (
                asset.worker_present is True
                or (
                    site is not None
                    and (
                        site.occupied is True
                        or (
                            site.worker_count
                            is not None
                            and site.worker_count > 0
                        )
                    )
                )
            )

            if occupied:
                add(
                    SafetyActionType.EVACUATE_ZONE,
                    (
                        "Request evacuation of the "
                        "affected zone according to "
                        "the approved emergency plan."
                    ),
                    priority=100,
                    authority=(
                        SafetyActionAuthority
                        .EMERGENCY_TEAM
                    ),
                    immediate=True,
                    human=True,
                )

        if risk_level in {
            SafetyRiskLevel.GUARDED,
            SafetyRiskLevel.ELEVATED,
        }:
            add(
                SafetyActionType.INSPECT,
                (
                    "Perform a qualified safety "
                    "inspection of the affected "
                    "asset and surrounding area."
                ),
                priority=60,
                authority=(
                    SafetyActionAuthority
                    .SAFETY_TEAM
                ),
                human=True,
            )

        return self._deduplicate_actions(
            actions
        )

    # =====================================================
    # Full assessment
    # =====================================================

    def assess(
        self,
        *,
        asset: SafetyAssetContext,
        evidence: Sequence[SafetyEvidence],
        site: SafetySiteContext | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> SafetyAssessment:

        now = utc_now()

        contextual = self.filter_context(
            asset=asset,
            evidence=evidence,
        )

        contextual = self.deduplicate_evidence(
            contextual
        )

        valid_evidence: list[
            SafetyEvidence
        ] = []

        warnings: list[str] = []

        for item in contextual:
            valid, item_warnings = (
                self.validate_evidence(
                    item,
                    now=now,
                )
            )

            warnings.extend(item_warnings)

            if valid:
                valid_evidence.append(item)

        coverage, coverage_score = (
            self.calculate_coverage(
                valid_evidence
            )
        )

        factors = self.calculate_factors(
            valid_evidence
        )

        (
            physical_risk,
            response_priority,
            risk_level,
            status,
            dominant_hazard,
        ) = self.calculate_risk(
            asset=asset,
            site=site,
            evidence=valid_evidence,
        )

        actions = self.recommend_actions(
            asset=asset,
            site=site,
            risk_level=risk_level,
            dominant_hazard=dominant_hazard,
            evidence=valid_evidence,
        )

        if valid_evidence:
            evidence_confidence = (
                sum(
                    item.confidence
                    for item in valid_evidence
                )
                / len(valid_evidence)
            )

            confidence = clamp(
                0.70 * evidence_confidence
                + 0.30 * coverage_score
            )
        else:
            confidence = 0.0

        if (
            dominant_hazard is not None
            and self.config
            .require_independent_verification
        ):
            verification = (
                self.independent_verification(
                    hazard=dominant_hazard,
                    evidence=valid_evidence,
                )
            )

            if not verification[
                "independently_verified"
            ]:
                warnings.append(
                    "Dominant hazard has not yet "
                    "been independently verified."
                )

                confidence *= 0.85

        if (
            coverage
            == SafetyDataCoverage.INSUFFICIENT
        ):
            warnings.append(
                "Safety data coverage is "
                "insufficient. Absence of evidence "
                "must not be interpreted as proof "
                "of safety."
            )

        incident_required = (
            risk_level
            in {
                SafetyRiskLevel.HIGH,
                SafetyRiskLevel.CRITICAL,
                SafetyRiskLevel.EMERGENCY,
            }
        )

        emergency_required = (
            risk_level
            == SafetyRiskLevel.EMERGENCY
        )

        machine_stop_review = any(
            action.action_type
            == SafetyActionType
            .REQUEST_MACHINE_STOP
            for action in actions
        )

        evacuation_review = any(
            action.action_type
            in {
                SafetyActionType.EVACUATE_ZONE,
                SafetyActionType.EVACUATE_SITE,
            }
            for action in actions
        )

        human_review = (
            risk_level
            in {
                SafetyRiskLevel.HIGH,
                SafetyRiskLevel.CRITICAL,
                SafetyRiskLevel.EMERGENCY,
            }
            or confidence
            < self.config.minimum_confidence
            or coverage
            == SafetyDataCoverage.INSUFFICIENT
            or any(
                action.requires_human_approval
                for action in actions
            )
        )

        if any(
            action.requires_deterministic_controller
            for action in actions
        ):
            warnings.append(
                "Control actions in this assessment "
                "are requests only. This tool does "
                "not directly actuate PLC, relay, "
                "SIS, E-stop, valve, breaker, or "
                "other physical safety equipment."
            )

        return SafetyAssessment(
            assessment_id=new_id(),

            asset_id=asset.asset_id,

            site_id=asset.site_id,
            zone_id=asset.zone_id,

            status=status,

            risk_level=risk_level,

            physical_risk_score=(
                physical_risk
            ),

            response_priority_score=(
                response_priority
            ),

            confidence=round(
                confidence,
                4,
            ),

            data_coverage=coverage,

            dominant_hazard=dominant_hazard,

            factors=factors,

            evidence=valid_evidence,

            actions=actions,

            incident_required=(
                incident_required
            ),

            emergency_response_required=(
                emergency_required
            ),

            machine_stop_review_required=(
                machine_stop_review
            ),

            evacuation_review_required=(
                evacuation_review
            ),

            human_review_required=(
                human_review
            ),

            warnings=list(
                dict.fromkeys(warnings)
            ),

            metadata={
                "request_id": request_id,
                "trace_id": trace_id,
                "coverage_score":
                    coverage_score,
            },
        )

    # =====================================================
    # Incident/event request
    # =====================================================

    def build_safety_event(
        self,
        assessment: SafetyAssessment,
        *,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> SafetyEventRequest | None:

        if (
            not assessment.incident_required
            or assessment.dominant_hazard is None
        ):
            return None

        timestamp_bucket = (
            assessment.generated_at
            .replace(
                minute=(
                    assessment.generated_at.minute
                    // 5
                    * 5
                ),
                second=0,
                microsecond=0,
            )
        )

        key_material = (
            f"{assessment.asset_id}|"
            f"{assessment.site_id}|"
            f"{assessment.zone_id}|"
            f"{assessment.dominant_hazard.value}|"
            f"{timestamp_bucket.isoformat()}"
        )

        idempotency_key = hashlib.sha256(
            key_material.encode("utf-8")
        ).hexdigest()

        return SafetyEventRequest(
            event_id=new_id(),

            idempotency_key=idempotency_key,

            asset_id=assessment.asset_id,

            site_id=assessment.site_id,
            zone_id=assessment.zone_id,

            hazard_type=(
                assessment.dominant_hazard
            ),

            risk_level=assessment.risk_level,

            description=(
                f"{assessment.dominant_hazard.value} "
                f"safety condition detected with "
                f"risk level "
                f"{assessment.risk_level.value}."
            ),

            occurred_at=(
                assessment.generated_at
            ),

            evidence_ids=[
                item.evidence_id
                for item in assessment.evidence
            ],

            trace_id=trace_id,
            request_id=request_id,

            metadata={
                "physical_risk_score":
                    assessment
                    .physical_risk_score,

                "response_priority_score":
                    assessment
                    .response_priority_score,

                "confidence":
                    assessment.confidence,

                "status":
                    assessment.status.value,
            },
        )

    # =====================================================
    # Telemetry adapter
    # =====================================================

    def evidence_from_telemetry(
        self,
        *,
        asset_id: str,
        sensor_type: str,
        value: float,
        severity: float,
        confidence: float,
        timestamp: datetime,
        site_id: str | None = None,
        zone_id: str | None = None,
        sensor_id: str | None = None,
        component: str | None = None,
        unit: str | None = None,
        critical: bool = False,
    ) -> SafetyEvidence:

        sensor = sensor_type.lower()

        mapping = {
            "temperature":
                HazardType.OVERHEATING,

            "vibration":
                HazardType.VIBRATION,

            "pressure":
                HazardType.HIGH_PRESSURE,

            "current":
                HazardType.ELECTRICAL,

            "voltage":
                HazardType.ELECTRICAL,
        }

        hazard = mapping.get(
            sensor,
            HazardType.UNKNOWN,
        )

        return SafetyEvidence(
            evidence_id=new_id(),

            hazard_type=hazard,

            source=EvidenceSource.TELEMETRY,

            description=(
                f"Abnormal {sensor_type} telemetry "
                f"detected."
            ),

            severity=severity,
            confidence=confidence,

            timestamp=timestamp,

            asset_id=asset_id,
            site_id=site_id,
            zone_id=zone_id,

            component=component,

            value=value,
            unit=unit,

            critical=critical,

            provenance_id=(
                f"telemetry:{sensor_id}:"
                f"{timestamp.isoformat()}"
                if sensor_id
                else None
            ),

            metadata={
                "sensor_id": sensor_id,
                "sensor_type": sensor_type,
            },
        )

    # =====================================================
    # CV adapter
    # =====================================================

    def evidence_from_cv(
        self,
        *,
        asset_id: str,
        detection_type: str,
        confidence: float,
        severity: float,
        timestamp: datetime,
        site_id: str | None = None,
        zone_id: str | None = None,
        camera_id: str | None = None,
        component: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SafetyEvidence:

        detection = detection_type.lower()

        mapping = {
            "fire": HazardType.FIRE,
            "smoke": HazardType.SMOKE,

            "ppe":
                HazardType.PPE_VIOLATION,

            "ppe_violation":
                HazardType.PPE_VIOLATION,

            "restricted_zone":
                HazardType.RESTRICTED_ZONE,

            "unsafe_proximity":
                HazardType.UNSAFE_PROXIMITY,

            "worker_fall":
                HazardType.WORKER_FALL,

            "person_fall":
                HazardType.WORKER_FALL,
        }

        hazard = mapping.get(
            detection,
            HazardType.UNKNOWN,
        )

        return SafetyEvidence(
            evidence_id=new_id(),

            hazard_type=hazard,

            source=(
                EvidenceSource.COMPUTER_VISION
            ),

            description=(
                f"Computer-vision detection: "
                f"{detection_type}."
            ),

            severity=severity,
            confidence=confidence,

            timestamp=timestamp,

            asset_id=asset_id,
            site_id=site_id,
            zone_id=zone_id,

            component=component,

            critical=(
                hazard
                in {
                    HazardType.FIRE,
                    HazardType.WORKER_FALL,
                }
                and severity >= 0.80
            ),

            provenance_id=(
                f"cv:{camera_id}:"
                f"{detection_type}:"
                f"{timestamp.isoformat()}"
                if camera_id
                else None
            ),

            metadata=dict(metadata or {}),
        )

    # =====================================================
    # Worker safety adapter
    # =====================================================

    def evidence_from_worker_event(
        self,
        *,
        asset_id: str,
        event_type: str,
        severity: float,
        confidence: float,
        timestamp: datetime,
        site_id: str | None = None,
        zone_id: str | None = None,
        worker_id: str | None = None,
    ) -> SafetyEvidence:

        event = event_type.lower()

        mapping = {
            "injury":
                HazardType.WORKER_INJURY,

            "fall":
                HazardType.WORKER_FALL,

            "unsafe_proximity":
                HazardType.UNSAFE_PROXIMITY,

            "restricted_zone":
                HazardType.RESTRICTED_ZONE,

            "ppe_violation":
                HazardType.PPE_VIOLATION,
        }

        hazard = mapping.get(
            event,
            HazardType.UNKNOWN,
        )

        return SafetyEvidence(
            evidence_id=new_id(),

            hazard_type=hazard,

            source=EvidenceSource.SAFETY_SYSTEM,

            description=(
                f"Worker safety event: "
                f"{event_type}."
            ),

            severity=severity,
            confidence=confidence,

            timestamp=timestamp,

            asset_id=asset_id,
            site_id=site_id,
            zone_id=zone_id,

            critical=(
                severity >= 0.80
                and hazard
                in {
                    HazardType.WORKER_INJURY,
                    HazardType.WORKER_FALL,
                    HazardType.UNSAFE_PROXIMITY,
                }
            ),

            provenance_id=(
                f"worker:{worker_id}:"
                f"{event_type}:"
                f"{timestamp.isoformat()}"
                if worker_id
                else None
            ),

            metadata={
                "worker_id": worker_id,
            },
        )

    # =====================================================
    # Action dedupe
    # =====================================================

    @staticmethod
    def _deduplicate_actions(
        actions: Sequence[SafetyAction],
    ) -> list[SafetyAction]:

        result: list[SafetyAction] = []

        seen: set[
            tuple[
                SafetyActionType,
                str | None,
            ]
        ] = set()

        for item in sorted(
            actions,
            key=lambda action: action.priority,
            reverse=True,
        ):
            key = (
                item.action_type,
                item.zone_id,
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result


# =========================================================
# Default instance
# =========================================================

safety_tools = SafetyTools()


# =========================================================
# Agent-facing functions
# =========================================================

def assess_safety(
    *,
    asset: SafetyAssetContext,
    evidence: Sequence[SafetyEvidence],
    site: SafetySiteContext | None = None,
    request_id: str | None = None,
    trace_id: str | None = None,
) -> SafetyAssessment:

    return safety_tools.assess(
        asset=asset,
        evidence=evidence,
        site=site,
        request_id=request_id,
        trace_id=trace_id,
    )


def calculate_safety_risk(
    *,
    asset: SafetyAssetContext,
    evidence: Sequence[SafetyEvidence],
    site: SafetySiteContext | None = None,
) -> dict[str, Any]:

    assessment = safety_tools.assess(
        asset=asset,
        evidence=evidence,
        site=site,
    )

    return {
        "physical_risk_score":
            assessment.physical_risk_score,

        "response_priority_score":
            assessment.response_priority_score,

        "risk_level":
            assessment.risk_level.value,

        "status":
            assessment.status.value,

        "confidence":
            assessment.confidence,

        "dominant_hazard": (
            assessment.dominant_hazard.value
            if assessment.dominant_hazard
            else None
        ),
    }


def create_safety_event(
    assessment: SafetyAssessment,
    *,
    request_id: str | None = None,
    trace_id: str | None = None,
) -> SafetyEventRequest | None:

    return safety_tools.build_safety_event(
        assessment,
        request_id=request_id,
        trace_id=trace_id,
    )


# =========================================================
# Exports
# =========================================================

__all__ = [
    "HazardType",
    "SafetySeverity",
    "SafetyRiskLevel",
    "SafetyStatus",
    "SafetyActionType",
    "SafetyActionAuthority",
    "EvidenceSource",
    "SafetyDataCoverage",

    "SafetyEvidence",
    "SafetyAssetContext",
    "SafetySiteContext",

    "SafetyAction",
    "SafetyRiskFactor",
    "SafetyAssessment",
    "SafetyEventRequest",

    "SafetyToolsConfig",
    "SafetyTools",

    "safety_tools",

    "assess_safety",
    "calculate_safety_risk",
    "create_safety_event",
]