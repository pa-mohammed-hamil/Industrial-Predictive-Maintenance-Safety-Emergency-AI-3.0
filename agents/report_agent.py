# app/agents/report_agent.py

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence
from uuid import UUID, uuid4


# =========================================================
# Utilities
# =========================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


def _finite(value: float, *, name: str) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    return value


def _non_negative(value: float, *, name: str) -> float:
    value = _finite(value, name=name)

    if value < 0:
        raise ValueError(f"{name} cannot be negative.")

    return value


def _probability(value: float, *, name: str) -> float:
    value = _finite(value, name=name)

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1."
        )

    return value


def _clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    return max(low, min(high, float(value)))


def _json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    if is_dataclass(value):
        return _json_safe(asdict(value))

    if hasattr(value, "model_dump"):
        try:
            return _json_safe(
                value.model_dump(mode="json")
            )
        except Exception:
            return _json_safe(value.model_dump())

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    return value


# =========================================================
# Enums
# =========================================================

class ReportType(str, Enum):
    ASSET_HEALTH = "asset_health"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    FAILURE_ANALYSIS = "failure_analysis"
    RUL = "rul"
    INSPECTION = "inspection"
    SAFETY = "safety"
    INCIDENT = "incident"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"
    REPAIR_REPLACEMENT = "repair_replacement"
    COST = "cost"
    DAILY_OPERATIONS = "daily_operations"
    SHIFT = "shift"
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    COMPREHENSIVE = "comprehensive"


class ReportFormat(str, Enum):
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    PDF = "pdf"


class ReportAudience(str, Enum):
    OPERATOR = "operator"
    TECHNICIAN = "technician"
    MAINTENANCE = "maintenance"
    ENGINEER = "engineer"
    SAFETY_MANAGER = "safety_manager"
    PLANT_MANAGER = "plant_manager"
    EXECUTIVE = "executive"
    AUDITOR = "auditor"
    CUSTOMER = "customer"


class ReportSeverity(str, Enum):
    NORMAL = "normal"
    INFORMATION = "information"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class FindingCategory(str, Enum):
    ASSET_HEALTH = "asset_health"
    TELEMETRY = "telemetry"
    ANOMALY = "anomaly"
    FAILURE = "failure"
    RUL = "rul"
    DEFECT = "defect"
    SAFETY = "safety"
    INCIDENT = "incident"
    MAINTENANCE = "maintenance"
    REPAIR_REPLACEMENT = "repair_replacement"
    COST = "cost"
    MODEL_RELIABILITY = "model_reliability"
    OTHER = "other"


class RecommendationPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    IMMEDIATE = "immediate"


class ReportStatus(str, Enum):
    GENERATED = "generated"
    PARTIAL = "partial"
    REVIEW_REQUIRED = "review_required"
    FAILED = "failed"


# =========================================================
# Source references
# =========================================================

@dataclass(slots=True)
class ReportSource:
    source_id: str
    source_type: str

    title: str | None = None

    document_id: str | None = None
    page: int | None = None
    chunk_id: str | None = None

    uri: str | None = None

    confidence: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.confidence is not None:
            self.confidence = _probability(
                self.confidence,
                name="confidence",
            )


# =========================================================
# Finding
# =========================================================

@dataclass(slots=True)
class ReportFinding:
    finding_id: str

    category: FindingCategory

    title: str
    description: str

    severity: ReportSeverity

    confidence: float = 1.0

    asset_id: str | None = None
    component: str | None = None

    value: float | None = None
    unit: str | None = None

    source_ids: list[str] = field(
        default_factory=list
    )

    requires_action: bool = False
    safety_critical: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.confidence = _probability(
            self.confidence,
            name="confidence",
        )


# =========================================================
# Recommendation
# =========================================================

@dataclass(slots=True)
class ReportRecommendation:
    recommendation_id: str

    title: str
    description: str

    priority: RecommendationPriority

    action_type: str

    reason: str

    component: str | None = None

    due_within_hours: float | None = None

    requires_human_approval: bool = True

    source_finding_ids: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.due_within_hours is not None:
            self.due_within_hours = _non_negative(
                self.due_within_hours,
                name="due_within_hours",
            )


# =========================================================
# Metrics
# =========================================================

@dataclass(slots=True)
class ReportMetric:
    name: str

    value: float | int | str | bool | None

    unit: str | None = None

    description: str | None = None

    status: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Section
# =========================================================

@dataclass(slots=True)
class ReportSection:
    section_id: str

    title: str

    summary: str | None = None

    findings: list[ReportFinding] = field(
        default_factory=list
    )

    recommendations: list[
        ReportRecommendation
    ] = field(default_factory=list)

    metrics: list[ReportMetric] = field(
        default_factory=list
    )

    sources: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Asset context
# =========================================================

@dataclass(slots=True)
class ReportAssetContext:
    asset_id: str

    asset_name: str | None = None
    asset_type: str | None = None

    site_id: str | None = None
    zone_id: str | None = None
    location: str | None = None

    manufacturer: str | None = None
    model_number: str | None = None
    serial_number: str | None = None

    operating: bool | None = None

    safety_critical: bool = False
    production_critical: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Agent input
# =========================================================

@dataclass(slots=True)
class ReportAgentInput:
    report_type: ReportType

    asset: ReportAssetContext | None = None

    audience: ReportAudience = (
        ReportAudience.ENGINEER
    )

    output_format: ReportFormat = (
        ReportFormat.JSON
    )

    title: str | None = None

    telemetry: Mapping[str, Any] | None = None

    asset_health: Mapping[str, Any] | None = None

    anomaly_analysis: Mapping[str, Any] | None = None

    failure_analysis: Mapping[str, Any] | None = None

    rul_analysis: Mapping[str, Any] | None = None

    inspection: Mapping[str, Any] | None = None

    safety_analysis: Mapping[str, Any] | None = None

    incident: Mapping[str, Any] | None = None

    emergency_response: Mapping[
        str,
        Any,
    ] | None = None

    maintenance_plan: Mapping[
        str,
        Any,
    ] | None = None

    repair_replacement: Mapping[
        str,
        Any,
    ] | None = None

    cost_analysis: Mapping[str, Any] | None = None

    production: Mapping[str, Any] | None = None

    model_monitoring: Mapping[
        str,
        Any,
    ] | None = None

    data_drift: Mapping[str, Any] | None = None
    model_drift: Mapping[str, Any] | None = None

    additional_findings: list[
        ReportFinding
    ] = field(default_factory=list)

    additional_recommendations: list[
        ReportRecommendation
    ] = field(default_factory=list)

    sources: list[ReportSource] = field(
        default_factory=list
    )

    start_time: datetime | None = None
    end_time: datetime | None = None

    incident_id: str | None = None
    inspection_id: str | None = None
    maintenance_plan_id: str | None = None
    decision_id: str | None = None

    request_id: str | None = None
    trace_id: str | None = None

    generated_by: str = "report_agent"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# =========================================================
# Result
# =========================================================

@dataclass(slots=True)
class ReportAgentResult:
    report_id: str

    report_type: ReportType
    report_format: ReportFormat
    audience: ReportAudience

    title: str

    status: ReportStatus
    severity: ReportSeverity

    executive_summary: str

    asset: ReportAssetContext | None

    sections: list[ReportSection]

    findings: list[ReportFinding]

    recommendations: list[
        ReportRecommendation
    ]

    sources: list[ReportSource]

    total_findings: int
    critical_findings: int

    total_recommendations: int
    immediate_actions: int

    confidence: float

    human_review_required: bool
    safety_review_required: bool

    generated_at: datetime

    period_start: datetime | None = None
    period_end: datetime | None = None

    incident_id: str | None = None
    inspection_id: str | None = None
    maintenance_plan_id: str | None = None
    decision_id: str | None = None

    request_id: str | None = None
    trace_id: str | None = None

    rendered_content: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)


# =========================================================
# Protocols
# =========================================================

class ReportRepository(Protocol):

    async def save_report(
        self,
        report: ReportAgentResult,
    ) -> Any:
        ...


class ObjectStorage(Protocol):

    async def upload_text(
        self,
        *,
        key: str,
        content: str,
        content_type: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        ...


class NotificationService(Protocol):

    async def notify(
        self,
        *,
        title: str,
        message: str,
        severity: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        ...


# =========================================================
# Configuration
# =========================================================

@dataclass(slots=True)
class ReportAgentConfig:
    minimum_confidence: float = 0.60

    critical_failure_probability: float = 0.85
    high_failure_probability: float = 0.65

    critical_rul_hours: float = 24.0
    warning_rul_hours: float = 168.0

    critical_health_score: float = 30.0
    warning_health_score: float = 60.0

    critical_anomaly_score: float = 0.90
    warning_anomaly_score: float = 0.65

    notify_critical_reports: bool = True

    persist_reports: bool = True

    upload_rendered_report: bool = False


# =========================================================
# Agent
# =========================================================

class ReportAgent:

    def __init__(
        self,
        *,
        config: ReportAgentConfig | None = None,
        repository: ReportRepository | None = None,
        object_storage: ObjectStorage | None = None,
        notification_service: (
            NotificationService | None
        ) = None,
    ) -> None:

        self.config = config or ReportAgentConfig()

        self.repository = repository
        self.object_storage = object_storage

        self.notification_service = (
            notification_service
        )

    # -----------------------------------------------------
    # Main
    # -----------------------------------------------------

    async def run(
        self,
        data: ReportAgentInput,
    ) -> ReportAgentResult:

        report_id = new_id()

        findings = self._collect_findings(data)

        findings.extend(
            data.additional_findings
        )

        recommendations = (
            self._collect_recommendations(
                data,
                findings,
            )
        )

        recommendations.extend(
            data.additional_recommendations
        )

        findings = self._deduplicate_findings(
            findings
        )

        recommendations = (
            self._deduplicate_recommendations(
                recommendations
            )
        )

        severity = self._severity(findings)

        confidence = self._confidence(
            data,
            findings,
        )

        safety_review = any(
            finding.safety_critical
            or finding.category
            == FindingCategory.SAFETY
            and finding.severity
            in {
                ReportSeverity.HIGH,
                ReportSeverity.CRITICAL,
                ReportSeverity.EMERGENCY,
            }
            for finding in findings
        )

        human_review = (
            confidence
            < self.config.minimum_confidence
            or safety_review
            or severity
            in {
                ReportSeverity.CRITICAL,
                ReportSeverity.EMERGENCY,
            }
        )

        sections = self._build_sections(
            data=data,
            findings=findings,
            recommendations=recommendations,
        )

        title = (
            data.title
            or self._default_title(data)
        )

        executive_summary = (
            self._executive_summary(
                data=data,
                findings=findings,
                recommendations=(
                    recommendations
                ),
                severity=severity,
            )
        )

        status = (
            ReportStatus.REVIEW_REQUIRED
            if human_review
            else ReportStatus.GENERATED
        )

        result = ReportAgentResult(
            report_id=report_id,

            report_type=data.report_type,

            report_format=data.output_format,

            audience=data.audience,

            title=title,

            status=status,

            severity=severity,

            executive_summary=(
                executive_summary
            ),

            asset=data.asset,

            sections=sections,

            findings=findings,

            recommendations=(
                recommendations
            ),

            sources=data.sources,

            total_findings=len(findings),

            critical_findings=sum(
                finding.severity
                in {
                    ReportSeverity.CRITICAL,
                    ReportSeverity.EMERGENCY,
                }
                for finding in findings
            ),

            total_recommendations=len(
                recommendations
            ),

            immediate_actions=sum(
                recommendation.priority
                == RecommendationPriority.IMMEDIATE
                for recommendation
                in recommendations
            ),

            confidence=round(
                confidence,
                4,
            ),

            human_review_required=(
                human_review
            ),

            safety_review_required=(
                safety_review
            ),

            generated_at=utc_now(),

            period_start=data.start_time,
            period_end=data.end_time,

            incident_id=data.incident_id,
            inspection_id=data.inspection_id,

            maintenance_plan_id=(
                data.maintenance_plan_id
            ),

            decision_id=data.decision_id,

            request_id=data.request_id,
            trace_id=data.trace_id,

            metadata={
                **data.metadata,
                "generated_by":
                    data.generated_by,
            },
        )

        result.rendered_content = (
            self.render(result)
        )

        await self._post_process(result)

        return result

    # -----------------------------------------------------
    # Findings
    # -----------------------------------------------------

    def _collect_findings(
        self,
        data: ReportAgentInput,
    ) -> list[ReportFinding]:

        findings: list[ReportFinding] = []

        asset_id = (
            data.asset.asset_id
            if data.asset
            else None
        )

        # ---------------------------------------------
        # Asset health
        # ---------------------------------------------

        if data.asset_health:
            health_score = self._number(
                data.asset_health,
                "health_score",
            )

            if health_score is not None:
                severity = ReportSeverity.NORMAL

                if (
                    health_score
                    <= self.config
                    .critical_health_score
                ):
                    severity = (
                        ReportSeverity.CRITICAL
                    )

                elif (
                    health_score
                    <= self.config
                    .warning_health_score
                ):
                    severity = (
                        ReportSeverity.WARNING
                    )

                findings.append(
                    ReportFinding(
                        finding_id=new_id(),

                        category=(
                            FindingCategory
                            .ASSET_HEALTH
                        ),

                        title="Asset health score",

                        description=(
                            "Current asset health "
                            f"score is "
                            f"{health_score:.1f}/100."
                        ),

                        severity=severity,

                        asset_id=asset_id,

                        value=health_score,

                        unit="score",

                        requires_action=(
                            severity
                            != ReportSeverity.NORMAL
                        ),
                    )
                )

        # ---------------------------------------------
        # Failure prediction
        # ---------------------------------------------

        failure_probability = None

        if data.failure_analysis:
            failure_probability = self._number(
                data.failure_analysis,
                "failure_probability",
            )

        if (
            failure_probability is None
            and data.asset_health
        ):
            failure_probability = self._number(
                data.asset_health,
                "failure_probability",
            )

        if failure_probability is not None:
            probability = _clamp(
                failure_probability
            )

            if (
                probability
                >= self.config
                .critical_failure_probability
            ):
                severity = ReportSeverity.CRITICAL

            elif (
                probability
                >= self.config
                .high_failure_probability
            ):
                severity = ReportSeverity.HIGH

            else:
                severity = ReportSeverity.INFORMATION

            findings.append(
                ReportFinding(
                    finding_id=new_id(),

                    category=(
                        FindingCategory.FAILURE
                    ),

                    title=(
                        "Predicted failure risk"
                    ),

                    description=(
                        "Predicted failure "
                        f"probability is "
                        f"{probability:.1%}."
                    ),

                    severity=severity,

                    confidence=(
                        self._mapping_confidence(
                            data.failure_analysis
                        )
                    ),

                    asset_id=asset_id,

                    value=probability,

                    unit="probability",

                    requires_action=(
                        severity
                        in {
                            ReportSeverity.HIGH,
                            ReportSeverity.CRITICAL,
                        }
                    ),

                    safety_critical=(
                        data.asset.safety_critical
                        if data.asset
                        else False
                    ),
                )
            )

        # ---------------------------------------------
        # RUL
        # ---------------------------------------------

        rul = None

        if data.rul_analysis:
            rul = self._number(
                data.rul_analysis,
                "predicted_rul_hours",
            )

            if rul is None:
                rul = self._number(
                    data.rul_analysis,
                    "predicted_rul",
                )

        if rul is None and data.asset_health:
            rul = self._number(
                data.asset_health,
                "predicted_rul_hours",
            )

        if rul is not None:
            if (
                rul
                <= self.config
                .critical_rul_hours
            ):
                severity = ReportSeverity.CRITICAL

            elif (
                rul
                <= self.config
                .warning_rul_hours
            ):
                severity = ReportSeverity.HIGH

            else:
                severity = ReportSeverity.INFORMATION

            findings.append(
                ReportFinding(
                    finding_id=new_id(),

                    category=FindingCategory.RUL,

                    title=(
                        "Remaining useful life"
                    ),

                    description=(
                        "Predicted remaining useful "
                        f"life is {rul:.1f} hours."
                    ),

                    severity=severity,

                    confidence=(
                        self._mapping_confidence(
                            data.rul_analysis
                        )
                    ),

                    asset_id=asset_id,

                    value=rul,

                    unit="hours",

                    requires_action=(
                        severity
                        in {
                            ReportSeverity.HIGH,
                            ReportSeverity.CRITICAL,
                        }
                    ),
                )
            )

        # ---------------------------------------------
        # Anomaly
        # ---------------------------------------------

        if data.anomaly_analysis:
            anomaly = self._number(
                data.anomaly_analysis,
                "anomaly_score",
            )

            detected = bool(
                data.anomaly_analysis.get(
                    "detected",
                    anomaly is not None
                    and anomaly
                    >= self.config
                    .warning_anomaly_score,
                )
            )

            if detected:
                score = anomaly or 0.0

                severity = (
                    ReportSeverity.CRITICAL
                    if score
                    >= self.config
                    .critical_anomaly_score
                    else ReportSeverity.WARNING
                )

                findings.append(
                    ReportFinding(
                        finding_id=new_id(),

                        category=(
                            FindingCategory.ANOMALY
                        ),

                        title=(
                            "Abnormal equipment "
                            "behaviour"
                        ),

                        description=(
                            "Anomaly analysis detected "
                            "abnormal operating "
                            f"behaviour with score "
                            f"{score:.1%}."
                        ),

                        severity=severity,

                        confidence=(
                            self._mapping_confidence(
                                data.anomaly_analysis
                            )
                        ),

                        asset_id=asset_id,

                        value=score,

                        unit="score",

                        requires_action=True,
                    )
                )

        # ---------------------------------------------
        # Inspection
        # ---------------------------------------------

        if data.inspection:
            defect_detected = bool(
                data.inspection.get(
                    "defect_detected",
                    False,
                )
            )

            if defect_detected:
                defect_type = str(
                    data.inspection.get(
                        "defect_type",
                        "unknown",
                    )
                )

                defect_severity = str(
                    data.inspection.get(
                        "severity",
                        "medium",
                    )
                ).lower()

                severity = self._severity_from_text(
                    defect_severity
                )

                findings.append(
                    ReportFinding(
                        finding_id=new_id(),

                        category=(
                            FindingCategory.DEFECT
                        ),

                        title=(
                            "Inspection defect detected"
                        ),

                        description=(
                            f"Inspection detected "
                            f"{defect_type} with "
                            f"{defect_severity} "
                            "severity."
                        ),

                        severity=severity,

                        confidence=(
                            self._mapping_confidence(
                                data.inspection
                            )
                        ),

                        asset_id=asset_id,

                        component=(
                            data.inspection.get(
                                "component"
                            )
                        ),

                        requires_action=True,

                        safety_critical=(
                            severity
                            in {
                                ReportSeverity.CRITICAL,
                                ReportSeverity.EMERGENCY,
                            }
                        ),
                    )
                )

        # ---------------------------------------------
        # Safety
        # ---------------------------------------------

        if data.safety_analysis:
            risk = self._number(
                data.safety_analysis,
                "risk_score",
            )

            emergency = bool(
                data.safety_analysis.get(
                    "emergency",
                    False,
                )
            )

            hazard = str(
                data.safety_analysis.get(
                    "hazard",
                    "industrial safety risk",
                )
            )

            if emergency:
                severity = ReportSeverity.EMERGENCY

            elif risk is not None and risk >= 0.85:
                severity = ReportSeverity.CRITICAL

            elif risk is not None and risk >= 0.60:
                severity = ReportSeverity.HIGH

            else:
                severity = ReportSeverity.WARNING

            findings.append(
                ReportFinding(
                    finding_id=new_id(),

                    category=FindingCategory.SAFETY,

                    title="Safety assessment",

                    description=(
                        f"Safety analysis identified "
                        f"{hazard}."
                    ),

                    severity=severity,

                    confidence=(
                        self._mapping_confidence(
                            data.safety_analysis
                        )
                    ),

                    asset_id=asset_id,

                    value=risk,

                    unit=(
                        "risk_score"
                        if risk is not None
                        else None
                    ),

                    requires_action=(
                        severity
                        != ReportSeverity.NORMAL
                    ),

                    safety_critical=True,
                )
            )

        # ---------------------------------------------
        # Incident
        # ---------------------------------------------

        if data.incident:
            incident_severity = str(
                data.incident.get(
                    "severity",
                    "warning",
                )
            )

            findings.append(
                ReportFinding(
                    finding_id=new_id(),

                    category=(
                        FindingCategory.INCIDENT
                    ),

                    title="Industrial incident",

                    description=str(
                        data.incident.get(
                            "summary",
                            data.incident.get(
                                "description",
                                (
                                    "Industrial incident "
                                    "recorded."
                                ),
                            ),
                        )
                    ),

                    severity=(
                        self._severity_from_text(
                            incident_severity
                        )
                    ),

                    confidence=1.0,

                    asset_id=asset_id,

                    requires_action=True,

                    safety_critical=(
                        incident_severity.lower()
                        in {
                            "critical",
                            "emergency",
                        }
                    ),
                )
            )

        # ---------------------------------------------
        # Maintenance
        # ---------------------------------------------

        if data.maintenance_plan:
            required = bool(
                data.maintenance_plan.get(
                    "maintenance_required",
                    False,
                )
            )

            if required:
                priority = str(
                    data.maintenance_plan.get(
                        "priority",
                        "normal",
                    )
                )

                findings.append(
                    ReportFinding(
                        finding_id=new_id(),

                        category=(
                            FindingCategory
                            .MAINTENANCE
                        ),

                        title=(
                            "Maintenance required"
                        ),

                        description=str(
                            data.maintenance_plan.get(
                                "summary",
                                (
                                    "Maintenance agent "
                                    "recommends action."
                                ),
                            )
                        ),

                        severity=(
                            self._severity_from_priority(
                                priority
                            )
                        ),

                        asset_id=asset_id,

                        requires_action=True,
                    )
                )

        # ---------------------------------------------
        # Repair / replacement
        # ---------------------------------------------

        if data.repair_replacement:
            decision = str(
                data.repair_replacement.get(
                    "decision",
                    "human_review",
                )
            )

            priority = str(
                data.repair_replacement.get(
                    "priority",
                    "normal",
                )
            )

            findings.append(
                ReportFinding(
                    finding_id=new_id(),

                    category=(
                        FindingCategory
                        .REPAIR_REPLACEMENT
                    ),

                    title=(
                        "Repair / replacement decision"
                    ),

                    description=str(
                        data.repair_replacement.get(
                            "summary",
                            (
                                "Lifecycle decision: "
                                f"{decision}."
                            ),
                        )
                    ),

                    severity=(
                        self._severity_from_priority(
                            priority
                        )
                    ),

                    confidence=(
                        self._mapping_confidence(
                            data.repair_replacement
                        )
                    ),

                    asset_id=asset_id,

                    requires_action=(
                        decision
                        in {
                            "repair",
                            "replace",
                            "human_review",
                        }
                    ),
                )
            )

        # ---------------------------------------------
        # Model reliability
        # ---------------------------------------------

        if (
            data.data_drift
            and bool(
                data.data_drift.get(
                    "drift_detected",
                    False,
                )
            )
        ):
            findings.append(
                ReportFinding(
                    finding_id=new_id(),

                    category=(
                        FindingCategory
                        .MODEL_RELIABILITY
                    ),

                    title="Data drift detected",

                    description=(
                        "Input data drift was "
                        "detected. Model-driven "
                        "recommendations should be "
                        "reviewed with additional "
                        "care."
                    ),

                    severity=ReportSeverity.WARNING,

                    asset_id=asset_id,

                    requires_action=True,
                )
            )

        if (
            data.model_drift
            and bool(
                data.model_drift.get(
                    "drift_detected",
                    False,
                )
            )
        ):
            findings.append(
                ReportFinding(
                    finding_id=new_id(),

                    category=(
                        FindingCategory
                        .MODEL_RELIABILITY
                    ),

                    title="Model drift detected",

                    description=(
                        "Production model performance "
                        "drift was detected."
                    ),

                    severity=ReportSeverity.HIGH,

                    asset_id=asset_id,

                    requires_action=True,
                )
            )

        return findings

    # -----------------------------------------------------
    # Recommendations
    # -----------------------------------------------------

    def _collect_recommendations(
        self,
        data: ReportAgentInput,
        findings: Sequence[ReportFinding],
    ) -> list[ReportRecommendation]:

        recommendations: list[
            ReportRecommendation
        ] = []

        for finding in findings:
            if not finding.requires_action:
                continue

            priority = (
                self._recommendation_priority(
                    finding.severity
                )
            )

            if (
                finding.category
                == FindingCategory.FAILURE
            ):
                recommendations.append(
                    self._recommendation(
                        title=(
                            "Perform failure "
                            "diagnostic review"
                        ),
                        description=(
                            "Review predicted failure "
                            "mode, contributing "
                            "features and physical "
                            "condition before failure."
                        ),
                        priority=priority,
                        action_type=(
                            "failure_analysis"
                        ),
                        reason=finding.description,
                        finding=finding,
                    )
                )

            elif (
                finding.category
                == FindingCategory.RUL
            ):
                recommendations.append(
                    self._recommendation(
                        title=(
                            "Schedule maintenance "
                            "before RUL window"
                        ),
                        description=(
                            "Plan inspection, repair "
                            "or replacement before "
                            "the predicted remaining "
                            "useful life is exhausted."
                        ),
                        priority=priority,
                        action_type=(
                            "schedule_maintenance"
                        ),
                        reason=finding.description,
                        finding=finding,
                    )
                )

            elif (
                finding.category
                == FindingCategory.DEFECT
            ):
                recommendations.append(
                    self._recommendation(
                        title=(
                            "Inspect detected defect"
                        ),
                        description=(
                            "Validate defect severity "
                            "and determine repair or "
                            "replacement requirements."
                        ),
                        priority=priority,
                        action_type=(
                            "physical_inspection"
                        ),
                        reason=finding.description,
                        finding=finding,
                    )
                )

            elif (
                finding.category
                == FindingCategory.SAFETY
            ):
                recommendations.append(
                    self._recommendation(
                        title="Safety review",
                        description=(
                            "Escalate the condition "
                            "to qualified safety "
                            "personnel and follow the "
                            "approved site safety "
                            "procedure."
                        ),
                        priority=priority,
                        action_type=(
                            "safety_review"
                        ),
                        reason=finding.description,
                        finding=finding,
                    )
                )

            elif (
                finding.category
                == FindingCategory.MAINTENANCE
            ):
                recommendations.append(
                    self._recommendation(
                        title=(
                            "Execute maintenance plan"
                        ),
                        description=(
                            "Review and execute the "
                            "approved maintenance "
                            "actions and work order."
                        ),
                        priority=priority,
                        action_type=(
                            "maintenance"
                        ),
                        reason=finding.description,
                        finding=finding,
                    )
                )

            elif (
                finding.category
                == FindingCategory
                .REPAIR_REPLACEMENT
            ):
                recommendations.append(
                    self._recommendation(
                        title=(
                            "Review lifecycle decision"
                        ),
                        description=(
                            "Validate repair versus "
                            "replacement economics, "
                            "safety and downtime "
                            "impact before approval."
                        ),
                        priority=priority,
                        action_type=(
                            "repair_replacement_review"
                        ),
                        reason=finding.description,
                        finding=finding,
                    )
                )

            elif (
                finding.category
                == FindingCategory
                .MODEL_RELIABILITY
            ):
                recommendations.append(
                    self._recommendation(
                        title=(
                            "Review model reliability"
                        ),
                        description=(
                            "Verify the recommendation "
                            "against sensor evidence, "
                            "inspection results and "
                            "engineering judgement."
                        ),
                        priority=priority,
                        action_type=(
                            "model_review"
                        ),
                        reason=finding.description,
                        finding=finding,
                    )
                )

        return recommendations

    @staticmethod
    def _recommendation(
        *,
        title: str,
        description: str,
        priority: RecommendationPriority,
        action_type: str,
        reason: str,
        finding: ReportFinding,
    ) -> ReportRecommendation:

        hours = {
            RecommendationPriority.LOW: 720.0,
            RecommendationPriority.MEDIUM: 168.0,
            RecommendationPriority.HIGH: 72.0,
            RecommendationPriority.URGENT: 24.0,
            RecommendationPriority.IMMEDIATE: 0.0,
        }[priority]

        return ReportRecommendation(
            recommendation_id=new_id(),

            title=title,
            description=description,

            priority=priority,

            action_type=action_type,

            reason=reason,

            component=finding.component,

            due_within_hours=hours,

            requires_human_approval=True,

            source_finding_ids=[
                finding.finding_id
            ],
        )

    # -----------------------------------------------------
    # Sections
    # -----------------------------------------------------

    def _build_sections(
        self,
        *,
        data: ReportAgentInput,
        findings: Sequence[ReportFinding],
        recommendations: Sequence[
            ReportRecommendation
        ],
    ) -> list[ReportSection]:

        sections: list[ReportSection] = []

        if data.asset:
            sections.append(
                ReportSection(
                    section_id=new_id(),

                    title="Asset Overview",

                    summary=(
                        self._asset_summary(
                            data.asset
                        )
                    ),

                    metrics=self._asset_metrics(
                        data
                    ),
                )
            )

        category_titles = {
            FindingCategory.ASSET_HEALTH:
                "Asset Health",

            FindingCategory.TELEMETRY:
                "Telemetry",

            FindingCategory.ANOMALY:
                "Anomaly Detection",

            FindingCategory.FAILURE:
                "Failure Prediction",

            FindingCategory.RUL:
                "Remaining Useful Life",

            FindingCategory.DEFECT:
                "Inspection Findings",

            FindingCategory.SAFETY:
                "Safety",

            FindingCategory.INCIDENT:
                "Incident",

            FindingCategory.MAINTENANCE:
                "Maintenance",

            FindingCategory.REPAIR_REPLACEMENT:
                "Repair / Replacement",

            FindingCategory.COST:
                "Cost Intelligence",

            FindingCategory.MODEL_RELIABILITY:
                "AI / Model Reliability",

            FindingCategory.OTHER:
                "Other Findings",
        }

        for category, title in category_titles.items():
            section_findings = [
                item
                for item in findings
                if item.category == category
            ]

            if not section_findings:
                continue

            finding_ids = {
                item.finding_id
                for item in section_findings
            }

            section_recommendations = [
                item
                for item in recommendations
                if any(
                    finding_id
                    in finding_ids
                    for finding_id
                    in item.source_finding_ids
                )
            ]

            sections.append(
                ReportSection(
                    section_id=new_id(),

                    title=title,

                    summary=(
                        f"{len(section_findings)} "
                        f"finding(s) identified."
                    ),

                    findings=section_findings,

                    recommendations=(
                        section_recommendations
                    ),
                )
            )

        if data.cost_analysis:
            sections.append(
                ReportSection(
                    section_id=new_id(),

                    title="Cost Analysis",

                    summary=str(
                        data.cost_analysis.get(
                            "summary",
                            (
                                "Cost intelligence "
                                "available for the "
                                "evaluated asset."
                            ),
                        )
                    ),

                    metrics=(
                        self._mapping_metrics(
                            data.cost_analysis
                        )
                    ),
                )
            )

        if data.production:
            sections.append(
                ReportSection(
                    section_id=new_id(),

                    title="Production Impact",

                    summary=str(
                        data.production.get(
                            "summary",
                            (
                                "Production impact "
                                "information."
                            ),
                        )
                    ),

                    metrics=(
                        self._mapping_metrics(
                            data.production
                        )
                    ),
                )
            )

        sections.append(
            ReportSection(
                section_id=new_id(),

                title="Recommended Actions",

                summary=(
                    f"{len(recommendations)} "
                    "recommended action(s)."
                ),

                recommendations=list(
                    recommendations
                ),
            )
        )

        return sections

    # -----------------------------------------------------
    # Severity
    # -----------------------------------------------------

    @staticmethod
    def _severity(
        findings: Sequence[ReportFinding],
    ) -> ReportSeverity:

        if not findings:
            return ReportSeverity.NORMAL

        rank = {
            ReportSeverity.NORMAL: 0,
            ReportSeverity.INFORMATION: 1,
            ReportSeverity.WARNING: 2,
            ReportSeverity.HIGH: 3,
            ReportSeverity.CRITICAL: 4,
            ReportSeverity.EMERGENCY: 5,
        }

        return max(
            (
                finding.severity
                for finding in findings
            ),
            key=lambda item: rank[item],
        )

    @staticmethod
    def _severity_from_text(
        value: str,
    ) -> ReportSeverity:

        value = value.lower().strip()

        mapping = {
            "normal": ReportSeverity.NORMAL,
            "info": ReportSeverity.INFORMATION,
            "information":
                ReportSeverity.INFORMATION,
            "low": ReportSeverity.INFORMATION,
            "medium": ReportSeverity.WARNING,
            "moderate": ReportSeverity.WARNING,
            "warning": ReportSeverity.WARNING,
            "high": ReportSeverity.HIGH,
            "urgent": ReportSeverity.HIGH,
            "critical": ReportSeverity.CRITICAL,
            "emergency": ReportSeverity.EMERGENCY,
        }

        return mapping.get(
            value,
            ReportSeverity.WARNING,
        )

    @staticmethod
    def _severity_from_priority(
        value: str,
    ) -> ReportSeverity:

        mapping = {
            "low": ReportSeverity.INFORMATION,
            "normal": ReportSeverity.WARNING,
            "medium": ReportSeverity.WARNING,
            "high": ReportSeverity.HIGH,
            "urgent": ReportSeverity.CRITICAL,
            "immediate": ReportSeverity.EMERGENCY,
        }

        return mapping.get(
            value.lower(),
            ReportSeverity.WARNING,
        )

    @staticmethod
    def _recommendation_priority(
        severity: ReportSeverity,
    ) -> RecommendationPriority:

        mapping = {
            ReportSeverity.NORMAL:
                RecommendationPriority.LOW,

            ReportSeverity.INFORMATION:
                RecommendationPriority.LOW,

            ReportSeverity.WARNING:
                RecommendationPriority.MEDIUM,

            ReportSeverity.HIGH:
                RecommendationPriority.HIGH,

            ReportSeverity.CRITICAL:
                RecommendationPriority.URGENT,

            ReportSeverity.EMERGENCY:
                RecommendationPriority.IMMEDIATE,
        }

        return mapping[severity]

    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------

    @staticmethod
    def _confidence(
        data: ReportAgentInput,
        findings: Sequence[ReportFinding],
    ) -> float:

        confidence_values = [
            finding.confidence
            for finding in findings
        ]

        if confidence_values:
            base = (
                sum(confidence_values)
                / len(confidence_values)
            )
        else:
            base = 0.60

        available_sources = sum([
            data.telemetry is not None,
            data.asset_health is not None,
            data.anomaly_analysis is not None,
            data.failure_analysis is not None,
            data.rul_analysis is not None,
            data.inspection is not None,
            data.safety_analysis is not None,
            data.maintenance_plan is not None,
            data.repair_replacement is not None,
            data.cost_analysis is not None,
        ])

        coverage = min(
            available_sources / 6.0,
            1.0,
        )

        confidence = (
            base * 0.80
            + coverage * 0.20
        )

        if (
            data.data_drift
            and data.data_drift.get(
                "drift_detected",
                False,
            )
        ):
            confidence *= 0.90

        if (
            data.model_drift
            and data.model_drift.get(
                "drift_detected",
                False,
            )
        ):
            confidence *= 0.80

        return _clamp(confidence)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    @staticmethod
    def _executive_summary(
        *,
        data: ReportAgentInput,
        findings: Sequence[ReportFinding],
        recommendations: Sequence[
            ReportRecommendation
        ],
        severity: ReportSeverity,
    ) -> str:

        asset = (
            data.asset.asset_name
            if (
                data.asset
                and data.asset.asset_name
            )
            else (
                data.asset.asset_id
                if data.asset
                else "evaluated system"
            )
        )

        critical = sum(
            item.severity
            in {
                ReportSeverity.CRITICAL,
                ReportSeverity.EMERGENCY,
            }
            for item in findings
        )

        immediate = sum(
            item.priority
            == RecommendationPriority.IMMEDIATE
            for item in recommendations
        )

        summary = (
            f"The {data.report_type.value} report "
            f"for {asset} identified "
            f"{len(findings)} finding(s). "
            f"Overall report severity is "
            f"{severity.value}. "
        )

        if critical:
            summary += (
                f"{critical} critical or emergency "
                "finding(s) require elevated "
                "attention. "
            )

        if immediate:
            summary += (
                f"{immediate} immediate action(s) "
                "are recommended. "
            )

        if not findings:
            summary += (
                "No reportable abnormal condition "
                "was identified from the supplied "
                "evidence."
            )

        return summary.strip()

    @staticmethod
    def _default_title(
        data: ReportAgentInput,
    ) -> str:

        type_name = (
            data.report_type.value
            .replace("_", " ")
            .title()
        )

        if data.asset:
            asset = (
                data.asset.asset_name
                or data.asset.asset_id
            )

            return f"{type_name} Report - {asset}"

        return f"{type_name} Report"

    @staticmethod
    def _asset_summary(
        asset: ReportAssetContext,
    ) -> str:

        name = (
            asset.asset_name
            or asset.asset_id
        )

        parts = [
            f"Asset: {name}",
        ]

        if asset.asset_type:
            parts.append(
                f"Type: {asset.asset_type}"
            )

        if asset.site_id:
            parts.append(
                f"Site: {asset.site_id}"
            )

        if asset.zone_id:
            parts.append(
                f"Zone: {asset.zone_id}"
            )

        return ". ".join(parts) + "."

    # -----------------------------------------------------
    # Metrics extraction
    # -----------------------------------------------------

    @staticmethod
    def _number(
        mapping: Mapping[str, Any],
        key: str,
    ) -> float | None:

        value = mapping.get(key)

        if value is None:
            return None

        try:
            value = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(value):
            return None

        return value

    @staticmethod
    def _mapping_confidence(
        mapping: Mapping[str, Any] | None,
    ) -> float:

        if not mapping:
            return 0.75

        value = mapping.get(
            "confidence",
            0.75,
        )

        try:
            return _clamp(float(value))
        except (TypeError, ValueError):
            return 0.75

    def _asset_metrics(
        self,
        data: ReportAgentInput,
    ) -> list[ReportMetric]:

        metrics: list[ReportMetric] = []

        for source in (
            data.asset_health,
            data.failure_analysis,
            data.rul_analysis,
        ):
            if not source:
                continue

            metrics.extend(
                self._mapping_metrics(source)
            )

        return self._deduplicate_metrics(
            metrics
        )

    @staticmethod
    def _mapping_metrics(
        mapping: Mapping[str, Any],
    ) -> list[ReportMetric]:

        preferred = {
            "health_score",
            "risk_score",
            "failure_probability",
            "predicted_rul_hours",
            "anomaly_score",
            "avoided_cost",
            "downtime_savings",
            "maintenance_roi_percent",
            "repair_cost",
            "replacement_cost",
            "expected_failure_cost",
            "oee",
            "availability",
            "performance",
            "quality",
        }

        metrics: list[ReportMetric] = []

        for key, value in mapping.items():
            if key not in preferred:
                continue

            if isinstance(
                value,
                (int, float, str, bool),
            ):
                metrics.append(
                    ReportMetric(
                        name=key,
                        value=value,
                    )
                )

        return metrics

    @staticmethod
    def _deduplicate_metrics(
        metrics: Sequence[ReportMetric],
    ) -> list[ReportMetric]:

        result: dict[str, ReportMetric] = {}

        for metric in metrics:
            result[metric.name] = metric

        return list(result.values())

    # -----------------------------------------------------
    # Deduplication
    # -----------------------------------------------------

    @staticmethod
    def _deduplicate_findings(
        findings: Sequence[ReportFinding],
    ) -> list[ReportFinding]:

        rank = {
            ReportSeverity.NORMAL: 0,
            ReportSeverity.INFORMATION: 1,
            ReportSeverity.WARNING: 2,
            ReportSeverity.HIGH: 3,
            ReportSeverity.CRITICAL: 4,
            ReportSeverity.EMERGENCY: 5,
        }

        unique: dict[
            tuple[str, str, str | None],
            ReportFinding,
        ] = {}

        for finding in findings:
            key = (
                finding.category.value,
                finding.title.lower(),
                finding.component,
            )

            current = unique.get(key)

            if (
                current is None
                or rank[finding.severity]
                > rank[current.severity]
            ):
                unique[key] = finding

        return list(unique.values())

    @staticmethod
    def _deduplicate_recommendations(
        recommendations: Sequence[
            ReportRecommendation
        ],
    ) -> list[ReportRecommendation]:

        rank = {
            RecommendationPriority.LOW: 1,
            RecommendationPriority.MEDIUM: 2,
            RecommendationPriority.HIGH: 3,
            RecommendationPriority.URGENT: 4,
            RecommendationPriority.IMMEDIATE: 5,
        }

        unique: dict[
            tuple[str, str | None],
            ReportRecommendation,
        ] = {}

        for item in recommendations:
            key = (
                item.action_type,
                item.component,
            )

            current = unique.get(key)

            if (
                current is None
                or rank[item.priority]
                > rank[current.priority]
            ):
                unique[key] = item

        return list(unique.values())

    # -----------------------------------------------------
    # Rendering
    # -----------------------------------------------------

    def render(
        self,
        report: ReportAgentResult,
    ) -> str:

        if (
            report.report_format
            == ReportFormat.JSON
        ):
            return json.dumps(
                report.to_dict(),
                indent=2,
                ensure_ascii=False,
            )

        if (
            report.report_format
            == ReportFormat.MARKDOWN
        ):
            return self._render_markdown(report)

        if (
            report.report_format
            == ReportFormat.HTML
        ):
            return self._render_html(report)

        if (
            report.report_format
            == ReportFormat.TEXT
        ):
            return self._render_text(report)

        # PDF generation belongs to the document/report
        # rendering layer. Return markdown as source content.
        if (
            report.report_format
            == ReportFormat.PDF
        ):
            return self._render_markdown(report)

        raise ValueError(
            f"Unsupported report format: "
            f"{report.report_format}"
        )

    @staticmethod
    def _render_markdown(
        report: ReportAgentResult,
    ) -> str:

        lines = [
            f"# {report.title}",
            "",
            f"**Report ID:** `{report.report_id}`",
            f"**Generated:** {report.generated_at.isoformat()}",
            f"**Severity:** {report.severity.value.upper()}",
            f"**Status:** {report.status.value}",
            f"**Confidence:** {report.confidence:.1%}",
            "",
            "## Executive Summary",
            "",
            report.executive_summary,
            "",
        ]

        for section in report.sections:
            lines.extend([
                f"## {section.title}",
                "",
            ])

            if section.summary:
                lines.extend([
                    section.summary,
                    "",
                ])

            if section.metrics:
                lines.extend([
                    "| Metric | Value | Unit |",
                    "|---|---:|---|",
                ])

                for metric in section.metrics:
                    lines.append(
                        f"| {metric.name} | "
                        f"{metric.value} | "
                        f"{metric.unit or ''} |"
                    )

                lines.append("")

            for finding in section.findings:
                lines.extend([
                    (
                        f"### {finding.title} "
                        f"({finding.severity.value})"
                    ),
                    "",
                    finding.description,
                    "",
                ])

            if section.recommendations:
                lines.extend([
                    "### Recommended Actions",
                    "",
                ])

                for recommendation in (
                    section.recommendations
                ):
                    lines.append(
                        "- "
                        f"**{recommendation.priority.value.upper()}** "
                        f"{recommendation.title}: "
                        f"{recommendation.description}"
                    )

                lines.append("")

        if report.sources:
            lines.extend([
                "## Evidence Sources",
                "",
            ])

            for source in report.sources:
                title = (
                    source.title
                    or source.source_id
                )

                location = ""

                if source.page is not None:
                    location = (
                        f", page {source.page}"
                    )

                lines.append(
                    f"- {title} "
                    f"({source.source_type}{location})"
                )

        return "\n".join(lines)

    @staticmethod
    def _render_text(
        report: ReportAgentResult,
    ) -> str:

        lines = [
            report.title,
            "=" * len(report.title),
            "",
            f"Severity: {report.severity.value}",
            f"Status: {report.status.value}",
            f"Confidence: {report.confidence:.1%}",
            "",
            "EXECUTIVE SUMMARY",
            report.executive_summary,
            "",
        ]

        for section in report.sections:
            lines.extend([
                section.title.upper(),
                "-" * len(section.title),
            ])

            if section.summary:
                lines.append(section.summary)

            for finding in section.findings:
                lines.append(
                    f"[{finding.severity.value.upper()}] "
                    f"{finding.title}: "
                    f"{finding.description}"
                )

            for recommendation in (
                section.recommendations
            ):
                lines.append(
                    f"[ACTION/{recommendation.priority.value.upper()}] "
                    f"{recommendation.title}: "
                    f"{recommendation.description}"
                )

            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _render_html(
        report: ReportAgentResult,
    ) -> str:

        import html

        sections: list[str] = []

        for section in report.sections:
            content = ""

            if section.summary:
                content += (
                    "<p>"
                    + html.escape(section.summary)
                    + "</p>"
                )

            for finding in section.findings:
                content += (
                    "<article>"
                    f"<h3>{html.escape(finding.title)}</h3>"
                    f"<p><strong>{html.escape(finding.severity.value.upper())}</strong></p>"
                    f"<p>{html.escape(finding.description)}</p>"
                    "</article>"
                )

            if section.recommendations:
                content += "<ul>"

                for recommendation in (
                    section.recommendations
                ):
                    content += (
                        "<li>"
                        f"<strong>{html.escape(recommendation.priority.value.upper())}</strong> "
                        f"{html.escape(recommendation.title)} — "
                        f"{html.escape(recommendation.description)}"
                        "</li>"
                    )

                content += "</ul>"

            sections.append(
                "<section>"
                f"<h2>{html.escape(section.title)}</h2>"
                f"{content}"
                "</section>"
            )

        return (
            "<!doctype html>"
            "<html>"
            "<head>"
            "<meta charset='utf-8'>"
            f"<title>{html.escape(report.title)}</title>"
            "</head>"
            "<body>"
            f"<h1>{html.escape(report.title)}</h1>"
            f"<p><strong>Severity:</strong> "
            f"{html.escape(report.severity.value)}</p>"
            f"<p><strong>Confidence:</strong> "
            f"{report.confidence:.1%}</p>"
            "<h2>Executive Summary</h2>"
            f"<p>{html.escape(report.executive_summary)}</p>"
            f"{''.join(sections)}"
            "</body>"
            "</html>"
        )

    # -----------------------------------------------------
    # Persistence / notification
    # -----------------------------------------------------

    async def _post_process(
        self,
        result: ReportAgentResult,
    ) -> None:

        self._record_metrics(result)
        self._log_report(result)

        if (
            self.config.persist_reports
            and self.repository
        ):
            try:
                await self.repository.save_report(
                    result
                )
            except Exception:
                pass

        if (
            self.config.upload_rendered_report
            and self.object_storage
            and result.rendered_content
        ):
            await self._upload(result)

        if (
            self.config.notify_critical_reports
            and self.notification_service
            and result.severity
            in {
                ReportSeverity.CRITICAL,
                ReportSeverity.EMERGENCY,
            }
        ):
            await self._notify(result)

    async def _upload(
        self,
        result: ReportAgentResult,
    ) -> None:

        if self.object_storage is None:
            return

        extension = {
            ReportFormat.JSON: "json",
            ReportFormat.MARKDOWN: "md",
            ReportFormat.HTML: "html",
            ReportFormat.TEXT: "txt",
            ReportFormat.PDF: "md",
        }[result.report_format]

        content_type = {
            ReportFormat.JSON:
                "application/json",

            ReportFormat.MARKDOWN:
                "text/markdown",

            ReportFormat.HTML:
                "text/html",

            ReportFormat.TEXT:
                "text/plain",

            ReportFormat.PDF:
                "text/markdown",
        }[result.report_format]

        key = (
            "reports/"
            f"{result.report_type.value}/"
            f"{result.report_id}.{extension}"
        )

        try:
            await self.object_storage.upload_text(
                key=key,

                content=(
                    result.rendered_content
                    or ""
                ),

                content_type=content_type,

                metadata={
                    "report_id":
                        result.report_id,

                    "report_type":
                        result.report_type.value,

                    "severity":
                        result.severity.value,

                    "asset_id": (
                        result.asset.asset_id
                        if result.asset
                        else ""
                    ),
                },
            )
        except Exception:
            pass

    async def _notify(
        self,
        result: ReportAgentResult,
    ) -> None:

        if self.notification_service is None:
            return

        try:
            await self.notification_service.notify(
                title=(
                    f"{result.severity.value.title()} "
                    "Industrial AI Report"
                ),

                message=result.executive_summary,

                severity=result.severity.value,

                metadata={
                    "report_id":
                        result.report_id,

                    "report_type":
                        result.report_type.value,

                    "asset_id": (
                        result.asset.asset_id
                        if result.asset
                        else None
                    ),

                    "trace_id":
                        result.trace_id,
                },
            )

        except Exception:
            pass

    # -----------------------------------------------------
    # Observability
    # -----------------------------------------------------

    @staticmethod
    def _record_metrics(
        result: ReportAgentResult,
    ) -> None:

        try:
            from app.observability.metrics import (
                registry,
            )

            registry.counter(
                "report_agent_runs_total"
            ).inc(
                labels={
                    "report_type":
                        result.report_type.value,

                    "format":
                        result.report_format.value,

                    "status":
                        result.status.value,
                }
            )

            registry.histogram(
                "report_agent_confidence"
            ).observe(
                result.confidence,

                labels={
                    "report_type":
                        result.report_type.value,
                },
            )

            registry.counter(
                "report_findings_total"
            ).inc(
                result.total_findings,

                labels={
                    "severity":
                        result.severity.value,
                },
            )

        except Exception:
            pass

    @staticmethod
    def _log_report(
        result: ReportAgentResult,
    ) -> None:

        try:
            from app.observability.logging import (
                get_logger,
            )

            logger = get_logger(
                "industrial.report_agent"
            )

            payload = {
                "report_id":
                    result.report_id,

                "report_type":
                    result.report_type.value,

                "severity":
                    result.severity.value,

                "status":
                    result.status.value,

                "confidence":
                    result.confidence,

                "total_findings":
                    result.total_findings,

                "critical_findings":
                    result.critical_findings,

                "total_recommendations":
                    result.total_recommendations,

                "immediate_actions":
                    result.immediate_actions,

                "asset_id": (
                    result.asset.asset_id
                    if result.asset
                    else None
                ),
            }

            if (
                result.severity
                == ReportSeverity.EMERGENCY
            ):
                logger.critical(
                    "Emergency industrial report "
                    "generated.",
                    extra={
                        "report": payload
                    },
                )

            elif (
                result.severity
                in {
                    ReportSeverity.CRITICAL,
                    ReportSeverity.HIGH,
                }
            ):
                logger.warning(
                    "Industrial report generated "
                    "with elevated findings.",
                    extra={
                        "report": payload
                    },
                )

            else:
                logger.info(
                    "Industrial report generated.",
                    extra={
                        "report": payload
                    },
                )

        except Exception:
            pass


# =========================================================
# Factory
# =========================================================

def create_report_agent(
    *,
    config: ReportAgentConfig | None = None,
    repository: ReportRepository | None = None,
    object_storage: ObjectStorage | None = None,
    notification_service: (
        NotificationService | None
    ) = None,
) -> ReportAgent:

    return ReportAgent(
        config=config,
        repository=repository,
        object_storage=object_storage,
        notification_service=notification_service,
    )


report_agent = ReportAgent()


__all__ = [
    "ReportType",
    "ReportFormat",
    "ReportAudience",
    "ReportSeverity",
    "FindingCategory",
    "RecommendationPriority",
    "ReportStatus",

    "ReportSource",
    "ReportFinding",
    "ReportRecommendation",
    "ReportMetric",
    "ReportSection",
    "ReportAssetContext",

    "ReportAgentInput",
    "ReportAgentResult",

    "ReportRepository",
    "ObjectStorage",
    "NotificationService",

    "ReportAgentConfig",
    "ReportAgent",

    "create_report_agent",
    "report_agent",
]