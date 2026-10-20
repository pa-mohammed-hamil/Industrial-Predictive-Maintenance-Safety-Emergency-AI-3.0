
"""
app/services/inspection_service.py

Industrial Predictive Maintenance, Safety & Emergency AI

Equipment Inspection Service

Features:
- Create and manage equipment inspections
- Inspection checklist management
- Manual and AI-assisted findings
- Machine condition scoring (0-100)
- Defect severity classification
- Inspection evidence references
- Inspection review and approval
- Corrective maintenance recommendations
- Inspection history and statistics
- Optional async repository integration
- Optional audit logging

SAFETY:
Inspection scores are decision-support indicators.
AI findings must be reviewed by qualified personnel.
This service does not certify equipment safety or
authorize machine restart.
"""

from __future__ import annotations

import copy
import inspect
import logging
import math

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4


logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class InspectionServiceError(Exception):
    pass


class InspectionValidationError(InspectionServiceError):
    pass


class InspectionNotFoundError(InspectionServiceError):
    pass


class InspectionTransitionError(InspectionServiceError):
    pass


# ============================================================
# ENUMS
# ============================================================

class InspectionType(str, Enum):
    ROUTINE = "routine"
    PREVENTIVE = "preventive"
    CONDITION_BASED = "condition_based"
    SAFETY = "safety"
    POST_FAILURE = "post_failure"
    POST_REPAIR = "post_repair"
    VISUAL = "visual"
    THERMAL = "thermal"
    VIBRATION = "vibration"
    ELECTRICAL = "electrical"


class InspectionStatus(str, Enum):
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    REJECTED = "rejected"


class FindingSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FindingSource(str, Enum):
    MANUAL = "manual"
    COMPUTER_VISION = "computer_vision"
    SENSOR = "sensor"
    AI_MODEL = "ai_model"


class ChecklistStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    NOT_CHECKED = "not_checked"


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def required_text(
    value: Any,
    name: str,
    max_length: int = 1000,
) -> str:

    if not isinstance(value, str):
        raise InspectionValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value:
        raise InspectionValidationError(
            f"{name} cannot be empty"
        )

    if len(value) > max_length:
        raise InspectionValidationError(
            f"{name} exceeds {max_length} characters"
        )

    return value


def optional_text(
    value: Optional[str],
    name: str,
    max_length: int = 1000,
) -> Optional[str]:

    if value is None:
        return None

    return required_text(
        value,
        name,
        max_length,
    )


def numeric(
    value: Any,
    name: str,
    minimum: float = 0.0,
    maximum: Optional[float] = None,
) -> float:

    if isinstance(value, bool):
        raise InspectionValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise InspectionValidationError(
            f"Invalid {name}"
        ) from exc

    if not math.isfinite(result):
        raise InspectionValidationError(
            f"{name} must be finite"
        )

    if result < minimum:
        raise InspectionValidationError(
            f"{name} cannot be below {minimum}"
        )

    if maximum is not None and result > maximum:
        raise InspectionValidationError(
            f"{name} cannot exceed {maximum}"
        )

    return result


def parse_enum(
    enum_class: type[Enum],
    value: Any,
    name: str,
) -> Enum:

    try:
        return enum_class(value)
    except (ValueError, TypeError) as exc:
        raise InspectionValidationError(
            f"Invalid {name}: {value}"
        ) from exc


def serialize(value: Any) -> Any:

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return serialize(asdict(value))

    if isinstance(value, dict):
        return {
            str(key): serialize(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            serialize(item)
            for item in value
        ]

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

@dataclass
class InspectionChecklistItem:
    item_id: str
    title: str
    status: str = ChecklistStatus.NOT_CHECKED.value
    notes: Optional[str] = None
    weight: float = 1.0


@dataclass
class InspectionFinding:
    finding_id: str
    defect_type: str
    description: str
    severity: str
    source: str
    confidence: Optional[float] = None
    evidence_ids: list[str] = field(
        default_factory=list
    )
    requires_maintenance: bool = False
    created_at: datetime = field(
        default_factory=utc_now
    )


@dataclass
class InspectionRecord:
    inspection_id: str
    company_id: str
    site_id: str
    asset_id: str
    inspection_type: str
    status: str
    title: str
    inspector_id: str

    zone_id: Optional[str] = None
    description: Optional[str] = None

    checklist: list[InspectionChecklistItem] = field(
        default_factory=list
    )

    findings: list[InspectionFinding] = field(
        default_factory=list
    )

    evidence_ids: list[str] = field(
        default_factory=list
    )

    condition_score: Optional[float] = None
    recommendations: list[str] = field(
        default_factory=list
    )

    reviewed_by: Optional[str] = None
    review_notes: Optional[str] = None

    created_at: datetime = field(
        default_factory=utc_now
    )
    updated_at: datetime = field(
        default_factory=utc_now
    )
    submitted_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


# ============================================================
# WORKFLOW RULES
# ============================================================

ALLOWED_TRANSITIONS = {
    InspectionStatus.DRAFT: {
        InspectionStatus.IN_PROGRESS,
    },
    InspectionStatus.IN_PROGRESS: {
        InspectionStatus.SUBMITTED,
    },
    InspectionStatus.SUBMITTED: {
        InspectionStatus.APPROVED,
        InspectionStatus.REJECTED,
    },
    InspectionStatus.REJECTED: {
        InspectionStatus.IN_PROGRESS,
    },
    InspectionStatus.APPROVED: set(),
}


# ============================================================
# INSPECTION SERVICE
# ============================================================

class InspectionService:
    """
    Equipment inspection management.

    Expected repository interface:

        async create_inspection(data: dict)
        async get_inspection(inspection_id: str)
        async update_inspection(
            inspection_id: str,
            updates: dict
        )
        async list_inspections(filters: dict)

    Optional audit interface:

        async record_event(data: dict)

    Production deployments must enforce authorization,
    tenant isolation, and concurrency-safe updates.
    """

    SEVERITY_PENALTIES = {
        FindingSeverity.INFO.value: 0,
        FindingSeverity.LOW.value: 5,
        FindingSeverity.MEDIUM.value: 15,
        FindingSeverity.HIGH.value: 30,
        FindingSeverity.CRITICAL.value: 50,
    }

    def __init__(
        self,
        inspection_repository: Any,
        audit_service: Any = None,
    ):

        if inspection_repository is None:
            raise InspectionValidationError(
                "inspection_repository is required"
            )

        self.repository = inspection_repository
        self.audit = audit_service

    @staticmethod
    def _field(
        record: Any,
        name: str,
        default: Any = None,
    ) -> Any:

        if isinstance(record, Mapping):
            return record.get(name, default)

        return getattr(record, name, default)

    async def _audit(
        self,
        *,
        inspection_id: str,
        action: str,
        actor_id: str,
        details: Optional[
            dict[str, Any]
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
                    "entity_type": "inspection",
                    "entity_id": inspection_id,
                    "action": action,
                    "actor_id": actor_id,
                    "details": details or {},
                    "timestamp": utc_now().isoformat(),
                },
            )
        except Exception:
            logger.exception(
                "Inspection audit delivery failed: %s",
                inspection_id,
            )

    # --------------------------------------------------------
    # CREATE INSPECTION
    # --------------------------------------------------------

    async def create_inspection(
        self,
        *,
        company_id: str,
        site_id: str,
        asset_id: str,
        inspection_type: str | InspectionType,
        title: str,
        inspector_id: str,
        zone_id: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> Any:

        kind = parse_enum(
            InspectionType,
            inspection_type,
            "inspection_type",
        )

        record = InspectionRecord(
            inspection_id=str(uuid4()),
            company_id=required_text(
                company_id,
                "company_id",
                100,
            ),
            site_id=required_text(
                site_id,
                "site_id",
                100,
            ),
            asset_id=required_text(
                asset_id,
                "asset_id",
                100,
            ),
            inspection_type=kind.value,
            status=InspectionStatus.DRAFT.value,
            title=required_text(
                title,
                "title",
                255,
            ),
            inspector_id=required_text(
                inspector_id,
                "inspector_id",
                100,
            ),
            zone_id=optional_text(
                zone_id,
                "zone_id",
                100,
            ),
            description=optional_text(
                description,
                "description",
                5000,
            ),
            metadata=dict(metadata or {}),
        )

        created = await invoke(
            self.repository.create_inspection,
            record.to_dict(),
        )

        await self._audit(
            inspection_id=record.inspection_id,
            action="inspection_created",
            actor_id=record.inspector_id,
            details={
                "asset_id": record.asset_id,
                "type": kind.value,
            },
        )

        return created

    # --------------------------------------------------------
    # GET INSPECTION
    # --------------------------------------------------------

    async def get_inspection(
        self,
        inspection_id: str,
    ) -> Any:

        inspection_id = required_text(
            inspection_id,
            "inspection_id",
            100,
        )

        record = await invoke(
            self.repository.get_inspection,
            inspection_id,
        )

        if record is None:
            raise InspectionNotFoundError(
                f"Inspection not found: {inspection_id}"
            )

        return record

    # --------------------------------------------------------
    # EDITABILITY
    # --------------------------------------------------------

    def _ensure_editable(
        self,
        record: Any,
    ) -> None:

        status = self._field(
            record,
            "status",
        )

        if status not in {
            InspectionStatus.DRAFT.value,
            InspectionStatus.IN_PROGRESS.value,
            InspectionStatus.REJECTED.value,
        }:
            raise InspectionTransitionError(
                "Inspection cannot be edited "
                "in its current status"
            )

    # --------------------------------------------------------
    # ADD CHECKLIST ITEM
    # --------------------------------------------------------

    async def add_checklist_item(
        self,
        *,
        inspection_id: str,
        title: str,
        actor_id: str,
        weight: float = 1.0,
    ) -> Any:

        record = await self.get_inspection(
            inspection_id
        )

        self._ensure_editable(record)

        item = InspectionChecklistItem(
            item_id=str(uuid4()),
            title=required_text(
                title,
                "title",
                500,
            ),
            weight=numeric(
                weight,
                "weight",
                minimum=0.01,
            ),
        )

        checklist = [
            serialize(existing)
            for existing in self._field(
                record,
                "checklist",
                [],
            )
        ]

        checklist.append(
            serialize(item)
        )

        updated = await invoke(
            self.repository.update_inspection,
            inspection_id,
            {
                "checklist": checklist,
                "condition_score": None,
                "updated_at": utc_now().isoformat(),
            },
        )

        await self._audit(
            inspection_id=inspection_id,
            action="checklist_item_added",
            actor_id=actor_id,
            details={
                "item_id": item.item_id,
            },
        )

        return updated

    # --------------------------------------------------------
    # UPDATE CHECKLIST ITEM
    # --------------------------------------------------------

    async def update_checklist_item(
        self,
        *,
        inspection_id: str,
        item_id: str,
        status: str | ChecklistStatus,
        actor_id: str,
        notes: Optional[str] = None,
    ) -> Any:

        record = await self.get_inspection(
            inspection_id
        )

        self._ensure_editable(record)

        target = parse_enum(
            ChecklistStatus,
            status,
            "checklist_status",
        )

        checklist = [
            serialize(item)
            for item in self._field(
                record,
                "checklist",
                [],
            )
        ]

        item = next(
            (
                item
                for item in checklist
                if item["item_id"] == item_id
            ),
            None,
        )

        if item is None:
            raise InspectionNotFoundError(
                "Checklist item not found"
            )

        item["status"] = target.value
        item["notes"] = notes

        updated = await invoke(
            self.repository.update_inspection,
            inspection_id,
            {
                "checklist": checklist,
                "condition_score": None,
                "updated_at": utc_now().isoformat(),
            },
        )

        await self._audit(
            inspection_id=inspection_id,
            action="checklist_item_updated",
            actor_id=actor_id,
            details={
                "item_id": item_id,
                "status": target.value,
            },
        )

        return updated

    # --------------------------------------------------------
    # ADD FINDING
    # --------------------------------------------------------

    async def add_finding(
        self,
        *,
        inspection_id: str,
        defect_type: str,
        description: str,
        severity: str | FindingSeverity,
        actor_id: str,
        source: str | FindingSource = (
            FindingSource.MANUAL
        ),
        confidence: Optional[float] = None,
        evidence_ids: Optional[
            Sequence[str]
        ] = None,
    ) -> Any:

        record = await self.get_inspection(
            inspection_id
        )

        self._ensure_editable(record)

        severity_enum = parse_enum(
            FindingSeverity,
            severity,
            "severity",
        )

        source_enum = parse_enum(
            FindingSource,
            source,
            "source",
        )

        if confidence is not None:
            confidence = numeric(
                confidence,
                "confidence",
                minimum=0,
                maximum=1,
            )

        finding = InspectionFinding(
            finding_id=str(uuid4()),
            defect_type=required_text(
                defect_type,
                "defect_type",
                100,
            ),
            description=required_text(
                description,
                "description",
                5000,
            ),
            severity=severity_enum.value,
            source=source_enum.value,
            confidence=confidence,
            evidence_ids=list(
                evidence_ids or []
            ),
            requires_maintenance=(
                severity_enum in {
                    FindingSeverity.MEDIUM,
                    FindingSeverity.HIGH,
                    FindingSeverity.CRITICAL,
                }
            ),
        )

        findings = [
            serialize(item)
            for item in self._field(
                record,
                "findings",
                [],
            )
        ]

        findings.append(
            serialize(finding)
        )

        updated = await invoke(
            self.repository.update_inspection,
            inspection_id,
            {
                "findings": findings,
                "condition_score": None,
                "updated_at": utc_now().isoformat(),
            },
        )

        await self._audit(
            inspection_id=inspection_id,
            action="finding_added",
            actor_id=actor_id,
            details={
                "finding_id": finding.finding_id,
                "severity": severity_enum.value,
                "source": source_enum.value,
            },
        )

        return updated

    # --------------------------------------------------------
    # ADD EVIDENCE REFERENCE
    # --------------------------------------------------------

    async def add_evidence(
        self,
        *,
        inspection_id: str,
        evidence_id: str,
        actor_id: str,
    ) -> Any:

        record = await self.get_inspection(
            inspection_id
        )

        self._ensure_editable(record)

        evidence_id = required_text(
            evidence_id,
            "evidence_id",
            255,
        )

        evidence = list(
            self._field(
                record,
                "evidence_ids",
                [],
            )
        )

        if evidence_id not in evidence:
            evidence.append(evidence_id)

        updated = await invoke(
            self.repository.update_inspection,
            inspection_id,
            {
                "evidence_ids": evidence,
                "updated_at": utc_now().isoformat(),
            },
        )

        await self._audit(
            inspection_id=inspection_id,
            action="evidence_added",
            actor_id=actor_id,
            details={
                "evidence_id": evidence_id,
            },
        )

        return updated

    # --------------------------------------------------------
    # CONDITION SCORE
    # --------------------------------------------------------

    def calculate_condition_score(
        self,
        *,
        checklist: Sequence[
            Mapping[str, Any]
        ],
        findings: Sequence[
            Mapping[str, Any]
        ],
    ) -> dict[str, Any]:
        """
        Illustrative condition score.

        Checklist score:
            PASS = full weight
            FAIL = zero
            N/A = excluded

        Finding penalties are applied after
        checklist scoring.

        Unchecked items prevent final scoring.
        """

        total_weight = 0.0
        passed_weight = 0.0

        for item in checklist:

            status = parse_enum(
                ChecklistStatus,
                item["status"],
                "checklist_status",
            )

            weight = numeric(
                item.get("weight", 1),
                "weight",
                minimum=0.01,
            )

            if status == ChecklistStatus.NOT_CHECKED:
                raise InspectionValidationError(
                    "Cannot calculate final score "
                    "with unchecked checklist items"
                )

            if status == ChecklistStatus.NOT_APPLICABLE:
                continue

            total_weight += weight

            if status == ChecklistStatus.PASS:
                passed_weight += weight

        if total_weight > 0:
            checklist_score = (
                passed_weight / total_weight
            ) * 100
        else:
            checklist_score = 100.0

        penalty = 0.0

        for finding in findings:
            severity = parse_enum(
                FindingSeverity,
                finding["severity"],
                "finding_severity",
            )

            penalty += self.SEVERITY_PENALTIES[
                severity.value
            ]

        condition_score = max(
            0.0,
            checklist_score - penalty,
        )

        condition_score = round(
            condition_score,
            2,
        )

        return {
            "condition_score": condition_score,
            "checklist_score": round(
                checklist_score,
                2,
            ),
            "finding_penalty": round(
                penalty,
                2,
            ),
            "finding_count": len(findings),
            "scoring_method": (
                "weighted_checklist_minus_penalties"
            ),
        }

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    @staticmethod
    def generate_recommendations(
        findings: Sequence[
            Mapping[str, Any]
        ],
        condition_score: float,
    ) -> list[str]:

        recommendations = []

        if condition_score < 30:
            recommendations.append(
                "Request urgent review by qualified "
                "maintenance and safety personnel."
            )

        elif condition_score < 60:
            recommendations.append(
                "Prioritize corrective maintenance "
                "and follow-up inspection."
            )

        elif condition_score < 80:
            recommendations.append(
                "Investigate identified defects "
                "during maintenance planning."
            )

        else:
            recommendations.append(
                "Continue scheduled preventive "
                "maintenance and inspections."
            )

        if any(
            finding["severity"] in {
                FindingSeverity.HIGH.value,
                FindingSeverity.CRITICAL.value,
            }
            for finding in findings
        ):
            recommendations.append(
                "Escalate high-severity findings "
                "for authorized technical assessment."
            )

        if any(
            finding["source"] in {
                FindingSource.AI_MODEL.value,
                FindingSource.COMPUTER_VISION.value,
            }
            for finding in findings
        ):
            recommendations.append(
                "Verify AI-generated findings "
                "through qualified human inspection."
            )

        return recommendations

    # --------------------------------------------------------
    # SUBMIT INSPECTION
    # --------------------------------------------------------

    async def submit_inspection(
        self,
        *,
        inspection_id: str,
        actor_id: str,
    ) -> Any:

        record = await self.get_inspection(
            inspection_id
        )

        status = self._field(
            record,
            "status",
        )

        if status != InspectionStatus.IN_PROGRESS.value:
            raise InspectionTransitionError(
                "Only in-progress inspections "
                "can be submitted"
            )

        checklist = [
            serialize(item)
            for item in self._field(
                record,
                "checklist",
                [],
            )
        ]

        findings = [
            serialize(item)
            for item in self._field(
                record,
                "findings",
                [],
            )
        ]

        if not checklist:
            raise InspectionValidationError(
                "Inspection checklist is empty"
            )

        score_data = self.calculate_condition_score(
            checklist=checklist,
            findings=findings,
        )

        recommendations = (
            self.generate_recommendations(
                findings,
                score_data["condition_score"],
            )
        )

        now = utc_now().isoformat()

        updated = await invoke(
            self.repository.update_inspection,
            inspection_id,
            {
                "status": (
                    InspectionStatus.SUBMITTED.value
                ),
                "condition_score": (
                    score_data["condition_score"]
                ),
                "recommendations": recommendations,
                "submitted_at": now,
                "updated_at": now,
            },
        )

        await self._audit(
            inspection_id=inspection_id,
            action="inspection_submitted",
            actor_id=actor_id,
            details=score_data,
        )

        return updated

    # --------------------------------------------------------
    # START / RESTART INSPECTION
    # --------------------------------------------------------

    async def start_inspection(
        self,
        *,
        inspection_id: str,
        actor_id: str,
    ) -> Any:

        record = await self.get_inspection(
            inspection_id
        )

        current = parse_enum(
            InspectionStatus,
            self._field(record, "status"),
            "status",
        )

        if (
            InspectionStatus.IN_PROGRESS
            not in ALLOWED_TRANSITIONS[current]
        ):
            raise InspectionTransitionError(
                "Cannot start inspection "
                "from current status"
            )

        updated = await invoke(
            self.repository.update_inspection,
            inspection_id,
            {
                "status": (
                    InspectionStatus.IN_PROGRESS.value
                ),
                "updated_at": utc_now().isoformat(),
            },
        )

        await self._audit(
            inspection_id=inspection_id,
            action="inspection_started",
            actor_id=actor_id,
        )

        return updated

    # --------------------------------------------------------
    # REVIEW INSPECTION
    # --------------------------------------------------------

    async def review_inspection(
        self,
        *,
        inspection_id: str,
        reviewer_id: str,
        approved: bool,
        review_notes: str,
    ) -> Any:

        record = await self.get_inspection(
            inspection_id
        )

        if (
            self._field(record, "status")
            != InspectionStatus.SUBMITTED.value
        ):
            raise InspectionTransitionError(
                "Only submitted inspections "
                "can be reviewed"
            )

        if not isinstance(approved, bool):
            raise InspectionValidationError(
                "approved must be boolean"
            )

        target = (
            InspectionStatus.APPROVED
            if approved
            else InspectionStatus.REJECTED
        )

        now = utc_now().isoformat()

        updated = await invoke(
            self.repository.update_inspection,
            inspection_id,
            {
                "status": target.value,
                "reviewed_by": required_text(
                    reviewer_id,
                    "reviewer_id",
                    100,
                ),
                "review_notes": required_text(
                    review_notes,
                    "review_notes",
                    5000,
                ),
                "reviewed_at": now,
                "updated_at": now,
            },
        )

        await self._audit(
            inspection_id=inspection_id,
            action="inspection_reviewed",
            actor_id=reviewer_id,
            details={
                "approved": approved,
                "status": target.value,
            },
        )

        return updated

    # --------------------------------------------------------
    # LIST INSPECTIONS
    # --------------------------------------------------------

    async def list_inspections(
        self,
        *,
        company_id: Optional[str] = None,
        site_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Any:

        filters = {}

        if company_id:
            filters["company_id"] = company_id

        if site_id:
            filters["site_id"] = site_id

        if asset_id:
            filters["asset_id"] = asset_id

        if status:
            filters["status"] = parse_enum(
                InspectionStatus,
                status,
                "status",
            ).value

        return await invoke(
            self.repository.list_inspections,
            filters,
        )

    # --------------------------------------------------------
    # INSPECTION STATISTICS
    # --------------------------------------------------------

    async def get_inspection_statistics(
        self,
        *,
        asset_id: str,
    ) -> dict[str, Any]:

        records = await self.list_inspections(
            asset_id=asset_id
        )

        scores = []
        total_findings = 0

        status_counts = {
            status.value: 0
            for status in InspectionStatus
        }

        for record in records:

            status = self._field(
                record,
                "status",
            )

            if status in status_counts:
                status_counts[status] += 1

            score = self._field(
                record,
                "condition_score",
            )

            if score is not None:
                scores.append(
                    numeric(
                        score,
                        "condition_score",
                        maximum=100,
                    )
                )

            total_findings += len(
                self._field(
                    record,
                    "findings",
                    [],
                )
            )

        average_score = (
            round(
                sum(scores) / len(scores),
                2,
            )
            if scores
            else None
        )

        return {
            "asset_id": asset_id,
            "total_inspections": len(records),
            "average_condition_score": average_score,
            "total_findings": total_findings,
            "status_counts": status_counts,
            "generated_at": utc_now().isoformat(),
        }


# ============================================================
# IN-MEMORY DEMONSTRATION REPOSITORY
# ============================================================

class InMemoryInspectionRepository:

    def __init__(self):
        self.records: dict[str, dict] = {}

    async def create_inspection(
        self,
        data: dict,
    ) -> dict:

        self.records[data["inspection_id"]] = (
            copy.deepcopy(data)
        )

        return copy.deepcopy(data)

    async def get_inspection(
        self,
        inspection_id: str,
    ) -> Optional[dict]:

        record = self.records.get(
            inspection_id
        )

        return (
            copy.deepcopy(record)
            if record is not None
            else None
        )

    async def update_inspection(
        self,
        inspection_id: str,
        updates: dict,
    ) -> dict:

        if inspection_id not in self.records:
            raise InspectionNotFoundError(
                "Inspection not found"
            )

        self.records[inspection_id].update(
            copy.deepcopy(updates)
        )

        return copy.deepcopy(
            self.records[inspection_id]
        )

    async def list_inspections(
        self,
        filters: dict,
    ) -> list[dict]:

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

if __name__ == "__main__":

    import asyncio
    import json

    async def main():

        repository = (
            InMemoryInspectionRepository()
        )

        service = InspectionService(
            inspection_repository=repository
        )

        inspection = await service.create_inspection(
            company_id="COMPANY-001",
            site_id="SITE-001",
            asset_id="MACHINE-001",
            inspection_type="preventive",
            title="Motor condition inspection",
            inspector_id="ENGINEER-001",
        )

        inspection_id = inspection["inspection_id"]

        await service.start_inspection(
            inspection_id=inspection_id,
            actor_id="ENGINEER-001",
        )

        updated = await service.add_checklist_item(
            inspection_id=inspection_id,
            title="Check cooling fan",
            actor_id="ENGINEER-001",
        )

        item_id = updated["checklist"][0]["item_id"]

        await service.update_checklist_item(
            inspection_id=inspection_id,
            item_id=item_id,
            status="pass",
            actor_id="ENGINEER-001",
        )

        await service.add_finding(
            inspection_id=inspection_id,
            defect_type="corrosion",
            description=(
                "Minor corrosion detected "
                "on motor housing."
            ),
            severity="low",
            actor_id="ENGINEER-001",
        )

        await service.submit_inspection(
            inspection_id=inspection_id,
            actor_id="ENGINEER-001",
        )

        result = await service.get_inspection(
            inspection_id
        )

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

    asyncio.run(main())
