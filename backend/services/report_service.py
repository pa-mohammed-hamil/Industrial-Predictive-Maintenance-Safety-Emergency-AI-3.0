
"""
app/services/report_service.py

Industrial Predictive Maintenance, Safety & Emergency AI
========================================================

Enterprise Report Generation Service

Features:
- Asset health reports
- Predictive maintenance reports
- Failure prediction reports
- Anomaly detection reports
- Inspection reports
- Industrial safety reports
- Incident and emergency reports
- Maintenance performance reports
- Repair and replacement cost reports
- Production loss reports
- Financial and operational summaries
- Executive dashboard reports
- Custom reports
- JSON, CSV, HTML and Markdown exports
- Time-range filtering
- Report statistics and aggregation
- Optional asynchronous repository integration
- Optional audit integration
- In-memory demonstration repository

Expected repository interface:
    async create_report(data: dict)
    async get_report(report_id: str)
    async list_reports(filters: dict)

Optional audit interface:
    async record_event(data: dict)

Notes:
- Source data must be tenant-authorized before reporting.
- CSV output includes spreadsheet-formula injection protection.
- HTML output escapes untrusted text.
- Generated reports are snapshots, not live monitoring.
- PDF generation can be implemented by a separate exporter.
"""

from __future__ import annotations

import asyncio
import copy
import csv
import html
import inspect
import io
import json
import logging
import math
import re

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4


logger = logging.getLogger(__name__)

MONEY_QUANT = Decimal("0.01")
ZERO = Decimal("0")


# ============================================================
# EXCEPTIONS
# ============================================================

class ReportServiceError(Exception):
    """Base report service exception."""


class ReportValidationError(ReportServiceError):
    """Invalid report request or source data."""


class ReportNotFoundError(ReportServiceError):
    """Requested report does not exist."""


class ReportRepositoryError(ReportServiceError):
    """Report repository operation failed."""


class ReportExportError(ReportServiceError):
    """Report export operation failed."""


# ============================================================
# ENUMS
# ============================================================

class ReportType(str, Enum):
    ASSET_HEALTH = "asset_health"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    FAILURE_PREDICTION = "failure_prediction"
    ANOMALY = "anomaly"
    INSPECTION = "inspection"
    SAFETY = "safety"
    INCIDENT = "incident"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"
    REPAIR_COST = "repair_cost"
    REPLACEMENT_COST = "replacement_cost"
    PRODUCTION_LOSS = "production_loss"
    FINANCIAL = "financial"
    EXECUTIVE = "executive"
    CUSTOM = "custom"


class ReportFormat(str, Enum):
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    MARKDOWN = "markdown"


class ReportStatus(str, Enum):
    GENERATED = "generated"
    ARCHIVED = "archived"


class ReportSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def required_text(
    value: Any,
    name: str,
    max_length: int = 255,
) -> str:

    if not isinstance(value, str):
        raise ReportValidationError(
            f"{name} must be a string"
        )

    result = value.strip()

    if not result:
        raise ReportValidationError(
            f"{name} cannot be empty"
        )

    if len(result) > max_length:
        raise ReportValidationError(
            f"{name} exceeds {max_length} characters"
        )

    return result


def optional_text(
    value: Optional[str],
    name: str,
    max_length: int = 255,
) -> Optional[str]:

    if value is None:
        return None

    return required_text(value, name, max_length)


def parse_enum(
    enum_class: type[Enum],
    value: Any,
    name: str,
) -> Enum:

    try:
        return enum_class(value)
    except (ValueError, TypeError) as exc:
        raise ReportValidationError(
            f"Invalid {name}: {value}"
        ) from exc


def validate_datetime(
    value: datetime,
    name: str,
) -> datetime:

    if not isinstance(value, datetime):
        raise ReportValidationError(
            f"{name} must be datetime"
        )

    if value.tzinfo is None or value.utcoffset() is None:
        raise ReportValidationError(
            f"{name} must be timezone-aware"
        )

    return value


def to_decimal(
    value: Any,
    name: str,
    minimum: Optional[Decimal] = None,
) -> Decimal:

    if isinstance(value, bool):
        raise ReportValidationError(
            f"{name} must be numeric"
        )

    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ReportValidationError(
            f"Invalid numeric value for {name}"
        ) from exc

    if not result.is_finite():
        raise ReportValidationError(
            f"{name} must be finite"
        )

    if minimum is not None and result < minimum:
        raise ReportValidationError(
            f"{name} must be >= {minimum}"
        )

    return result


def money(value: Any) -> Decimal:
    return to_decimal(
        value,
        "money",
    ).quantize(
        MONEY_QUANT,
        rounding=ROUND_HALF_UP,
    )


def serialize(value: Any) -> Any:

    if isinstance(value, Decimal):
        return str(value)

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
        return [
            serialize(item)
            for item in value
        ]

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReportValidationError(
                "Non-finite floating-point value"
            )

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


def safe_csv_cell(value: Any) -> str:
    """
    Protect CSV output against spreadsheet formula injection.

    The prefix is applied after leading whitespace is checked,
    so cells containing spaces before formula characters are
    also protected.
    """

    if value is None:
        return ""

    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(
            serialize(value),
            ensure_ascii=False,
        )
    else:
        text = str(value)

    if re.match(r"^[\s\x00-\x1f]*[=+\-@]", text):
        return "'" + text

    return text


def flatten_mapping(
    data: Mapping[str, Any],
    prefix: str = "",
) -> dict[str, Any]:

    result: dict[str, Any] = {}

    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, Mapping):
            result.update(
                flatten_mapping(value, path)
            )
        else:
            result[path] = value

    return result


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class ReportPeriod:
    start: Optional[datetime] = None
    end: Optional[datetime] = None

    def __post_init__(self) -> None:

        if self.start is not None:
            validate_datetime(
                self.start,
                "period.start",
            )

        if self.end is not None:
            validate_datetime(
                self.end,
                "period.end",
            )

        if (
            self.start is not None
            and self.end is not None
            and self.end <= self.start
        ):
            raise ReportValidationError(
                "Report period end must be after start"
            )


@dataclass
class ReportFinding:
    finding_id: str
    title: str
    description: str
    severity: str
    source: Optional[str] = None
    asset_id: Optional[str] = None
    recommendation: Optional[str] = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass
class ReportSection:
    section_id: str
    title: str
    description: str
    metrics: dict[str, Any] = field(
        default_factory=dict
    )
    records: list[dict[str, Any]] = field(
        default_factory=list
    )
    findings: list[ReportFinding] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass
class GeneratedReport:
    report_id: str
    report_type: str
    title: str
    description: str
    company_id: str
    site_id: Optional[str]
    asset_id: Optional[str]
    status: str
    period: ReportPeriod
    sections: list[ReportSection]
    summary: dict[str, Any]
    recommendations: list[str]
    created_by: str
    generated_at: datetime
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class ReportExport:
    report_id: str
    format: str
    filename: str
    content_type: str
    content: str
    generated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


# ============================================================
# REPORT SERVICE
# ============================================================

class ReportService:
    """
    Enterprise reporting service for industrial AI.

    Source records must be provided by an authorized
    application layer or trusted data integration.

    No database access is required for report generation.
    """

    SEVERITY_ORDER = {
        ReportSeverity.INFO.value: 0,
        ReportSeverity.LOW.value: 1,
        ReportSeverity.MEDIUM.value: 2,
        ReportSeverity.HIGH.value: 3,
        ReportSeverity.CRITICAL.value: 4,
    }

    def __init__(
        self,
        report_repository: Any = None,
        audit_service: Any = None,
        default_company_id: Optional[str] = None,
    ):

        self.repository = report_repository
        self.audit = audit_service
        self.default_company_id = default_company_id

    # ========================================================
    # FINDING CREATION
    # ========================================================

    @staticmethod
    def create_finding(
        *,
        title: str,
        description: str,
        severity: str | ReportSeverity = (
            ReportSeverity.INFO
        ),
        source: Optional[str] = None,
        asset_id: Optional[str] = None,
        recommendation: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ReportFinding:

        selected_severity = parse_enum(
            ReportSeverity,
            severity,
            "severity",
        )

        return ReportFinding(
            finding_id=str(uuid4()),
            title=required_text(
                title,
                "title",
            ),
            description=required_text(
                description,
                "description",
                5000,
            ),
            severity=selected_severity.value,
            source=optional_text(
                source,
                "source",
            ),
            asset_id=optional_text(
                asset_id,
                "asset_id",
            ),
            recommendation=optional_text(
                recommendation,
                "recommendation",
                2000,
            ),
            metadata=dict(metadata or {}),
        )

    # ========================================================
    # SECTION CREATION
    # ========================================================

    @staticmethod
    def create_section(
        *,
        title: str,
        description: str,
        metrics: Optional[Mapping[str, Any]] = None,
        records: Optional[
            Sequence[Mapping[str, Any]]
        ] = None,
        findings: Optional[
            Sequence[ReportFinding]
        ] = None,
    ) -> ReportSection:

        normalized_records = []

        for record in records or []:
            if not isinstance(record, Mapping):
                raise ReportValidationError(
                    "Report records must be mappings"
                )

            normalized_records.append(
                serialize(dict(record))
            )

        normalized_findings = []

        for finding in findings or []:
            if not isinstance(
                finding,
                ReportFinding,
            ):
                raise ReportValidationError(
                    "findings must contain ReportFinding"
                )

            normalized_findings.append(finding)

        return ReportSection(
            section_id=str(uuid4()),
            title=required_text(
                title,
                "title",
            ),
            description=required_text(
                description,
                "description",
                5000,
            ),
            metrics=serialize(
                dict(metrics or {})
            ),
            records=normalized_records,
            findings=normalized_findings,
        )

    # ========================================================
    # METRIC HELPERS
    # ========================================================

    @staticmethod
    def calculate_average(
        values: Sequence[Any],
    ) -> Optional[Decimal]:

        if not values:
            return None

        numeric = [
            to_decimal(value, "metric")
            for value in values
        ]

        return (
            sum(numeric, ZERO)
            / Decimal(len(numeric))
        ).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )

    @staticmethod
    def calculate_total(
        values: Sequence[Any],
    ) -> Decimal:

        return sum(
            (
                to_decimal(value, "metric")
                for value in values
            ),
            ZERO,
        )

    @staticmethod
    def calculate_percentage(
        numerator: Any,
        denominator: Any,
    ) -> Decimal:

        numerator_value = to_decimal(
            numerator,
            "numerator",
        )

        denominator_value = to_decimal(
            denominator,
            "denominator",
        )

        if denominator_value == ZERO:
            return ZERO

        return (
            numerator_value
            / denominator_value
            * Decimal("100")
        ).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )

    # ========================================================
    # GENERIC REPORT CREATION
    # ========================================================

    def generate_report(
        self,
        *,
        report_type: str | ReportType,
        title: str,
        description: str,
        company_id: Optional[str] = None,
        site_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        sections: Optional[
            Sequence[ReportSection]
        ] = None,
        summary: Optional[Mapping[str, Any]] = None,
        recommendations: Optional[
            Sequence[str]
        ] = None,
        created_by: str = "system",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> GeneratedReport:

        selected_type = parse_enum(
            ReportType,
            report_type,
            "report_type",
        )

        selected_company_id = (
            company_id
            or self.default_company_id
        )

        if selected_company_id is None:
            raise ReportValidationError(
                "company_id is required"
            )

        period = ReportPeriod(
            start=period_start,
            end=period_end,
        )

        normalized_sections = []

        for section in sections or []:
            if not isinstance(
                section,
                ReportSection,
            ):
                raise ReportValidationError(
                    "sections must contain ReportSection"
                )

            normalized_sections.append(section)

        normalized_recommendations = [
            required_text(
                recommendation,
                "recommendation",
                2000,
            )
            for recommendation in (
                recommendations or []
            )
        ]

        return GeneratedReport(
            report_id=str(uuid4()),
            report_type=selected_type.value,
            title=required_text(
                title,
                "title",
            ),
            description=required_text(
                description,
                "description",
                5000,
            ),
            company_id=required_text(
                selected_company_id,
                "company_id",
            ),
            site_id=optional_text(
                site_id,
                "site_id",
            ),
            asset_id=optional_text(
                asset_id,
                "asset_id",
            ),
            status=ReportStatus.GENERATED.value,
            period=period,
            sections=normalized_sections,
            summary=serialize(
                dict(summary or {})
            ),
            recommendations=normalized_recommendations,
            created_by=required_text(
                created_by,
                "created_by",
            ),
            generated_at=utc_now(),
            metadata=serialize(
                dict(metadata or {})
            ),
        )

    # ========================================================
    # ASSET HEALTH REPORT
    # ========================================================

    def generate_asset_health_report(
        self,
        *,
        company_id: str,
        asset_records: Sequence[
            Mapping[str, Any]
        ],
        site_id: Optional[str] = None,
        created_by: str = "system",
    ) -> GeneratedReport:

        records = [
            serialize(dict(record))
            for record in asset_records
        ]

        scores: list[Decimal] = []
        critical_assets: list[str] = []
        findings: list[ReportFinding] = []

        for record in records:
            asset_id = str(
                record.get("asset_id", "unknown")
            )

            raw_score = record.get(
                "health_score"
            )

            if raw_score is None:
                continue

            score = to_decimal(
                raw_score,
                "health_score",
                ZERO,
            )

            if score > 100:
                raise ReportValidationError(
                    "health_score cannot exceed 100"
                )

            scores.append(score)

            if score < 40:
                critical_assets.append(asset_id)

                findings.append(
                    self.create_finding(
                        title="Critical Asset Health",
                        description=(
                            f"Asset {asset_id} has "
                            f"health score {score}."
                        ),
                        severity="critical",
                        source="health_score_service",
                        asset_id=asset_id,
                        recommendation=(
                            "Arrange a qualified inspection "
                            "and review operating restrictions."
                        ),
                    )
                )

        average_score = self.calculate_average(
            scores
        )

        section = self.create_section(
            title="Asset Health Overview",
            description=(
                "Health scores and asset-level "
                "condition information."
            ),
            metrics={
                "total_assets": len(records),
                "assets_with_health_score": len(scores),
                "average_health_score": (
                    str(average_score)
                    if average_score is not None
                    else None
                ),
                "critical_assets": len(
                    critical_assets
                ),
            },
            records=records,
            findings=findings,
        )

        recommendations = []

        if critical_assets:
            recommendations.append(
                "Prioritize qualified inspection of "
                "assets with critical health scores."
            )

        return self.generate_report(
            report_type=ReportType.ASSET_HEALTH,
            title="Industrial Asset Health Report",
            description=(
                "Condition and health analysis "
                "of industrial assets."
            ),
            company_id=company_id,
            site_id=site_id,
            sections=[section],
            summary=section.metrics,
            recommendations=recommendations,
            created_by=created_by,
        )

    # ========================================================
    # PREDICTIVE MAINTENANCE REPORT
    # ========================================================

    def generate_predictive_maintenance_report(
        self,
        *,
        company_id: str,
        prediction_records: Sequence[
            Mapping[str, Any]
        ],
        site_id: Optional[str] = None,
        created_by: str = "system",
    ) -> GeneratedReport:

        records = [
            serialize(dict(record))
            for record in prediction_records
        ]

        high_risk = []
        findings = []

        for record in records:
            risk = str(
                record.get("risk_level", "")
            ).lower()

            asset_id = str(
                record.get("asset_id", "unknown")
            )

            if risk in {"high", "critical"}:
                high_risk.append(record)

                findings.append(
                    self.create_finding(
                        title="Elevated Failure Risk",
                        description=(
                            f"Asset {asset_id} has "
                            f"{risk} predicted failure risk."
                        ),
                        severity=risk,
                        source="failure_prediction_service",
                        asset_id=asset_id,
                        recommendation=(
                            "Review model confidence, "
                            "telemetry and maintenance history "
                            "before scheduling intervention."
                        ),
                    )
                )

        section = self.create_section(
            title="Failure Risk Predictions",
            description=(
                "Model-generated asset failure "
                "risk predictions."
            ),
            metrics={
                "total_predictions": len(records),
                "high_or_critical_predictions": len(
                    high_risk
                ),
            },
            records=records,
            findings=findings,
        )

        return self.generate_report(
            report_type=ReportType.PREDICTIVE_MAINTENANCE,
            title="Predictive Maintenance Report",
            description=(
                "Industrial equipment failure risk "
                "and maintenance planning report."
            ),
            company_id=company_id,
            site_id=site_id,
            sections=[section],
            summary=section.metrics,
            recommendations=[
                "Validate high-risk predictions using "
                "recent telemetry and maintenance records.",
                "Prioritize maintenance by risk, "
                "operational impact and asset criticality.",
            ],
            created_by=created_by,
        )

    # ========================================================
    # ANOMALY REPORT
    # ========================================================

    def generate_anomaly_report(
        self,
        *,
        company_id: str,
        anomaly_records: Sequence[
            Mapping[str, Any]
        ],
        site_id: Optional[str] = None,
        created_by: str = "system",
    ) -> GeneratedReport:

        records = [
            serialize(dict(record))
            for record in anomaly_records
        ]

        severity_counts = {
            "low": 0,
            "medium": 0,
            "high": 0,
            "critical": 0,
        }

        findings = []

        for record in records:
            severity = str(
                record.get("severity", "low")
            ).lower()

            if severity in severity_counts:
                severity_counts[severity] += 1

            if severity in {"high", "critical"}:
                findings.append(
                    self.create_finding(
                        title="Significant Sensor Anomaly",
                        description=str(
                            record.get(
                                "description",
                                "Elevated anomaly detected.",
                            )
                        ),
                        severity=severity,
                        source="anomaly_service",
                        asset_id=(
                            str(record["asset_id"])
                            if record.get("asset_id")
                            else None
                        ),
                        recommendation=(
                            "Validate sensor readings and "
                            "investigate the underlying cause."
                        ),
                    )
                )

        section = self.create_section(
            title="Anomaly Detection Overview",
            description=(
                "Sensor and multivariate anomaly events."
            ),
            metrics={
                "total_anomalies": len(records),
                "severity_counts": severity_counts,
            },
            records=records,
            findings=findings,
        )

        return self.generate_report(
            report_type=ReportType.ANOMALY,
            title="Industrial Anomaly Detection Report",
            description=(
                "Detected equipment and sensor anomalies."
            ),
            company_id=company_id,
            site_id=site_id,
            sections=[section],
            summary=section.metrics,
            recommendations=[
                "Review high-severity anomalies "
                "against sensor calibration and history.",
            ],
            created_by=created_by,
        )

    # ========================================================
    # INCIDENT REPORT
    # ========================================================

    def generate_incident_report(
        self,
        *,
        company_id: str,
        incident_records: Sequence[
            Mapping[str, Any]
        ],
        site_id: Optional[str] = None,
        created_by: str = "system",
    ) -> GeneratedReport:

        records = [
            serialize(dict(record))
            for record in incident_records
        ]

        severity_counts: dict[str, int] = {}
        status_counts: dict[str, int] = {}
        findings = []

        for record in records:
            severity = str(
                record.get("severity", "unknown")
            ).lower()

            status = str(
                record.get("status", "unknown")
            ).lower()

            severity_counts[severity] = (
                severity_counts.get(severity, 0) + 1
            )

            status_counts[status] = (
                status_counts.get(status, 0) + 1
            )

            if severity in {"high", "critical"}:
                findings.append(
                    self.create_finding(
                        title="High-Severity Incident",
                        description=str(
                            record.get(
                                "description",
                                "High-severity incident recorded.",
                            )
                        ),
                        severity=severity,
                        source="incident_service",
                        asset_id=(
                            str(record["asset_id"])
                            if record.get("asset_id")
                            else None
                        ),
                        recommendation=(
                            "Follow the approved incident "
                            "response, investigation and "
                            "corrective-action procedures."
                        ),
                    )
                )

        section = self.create_section(
            title="Incident Overview",
            description=(
                "Incident severity, status and "
                "investigation records."
            ),
            metrics={
                "total_incidents": len(records),
                "severity_counts": severity_counts,
                "status_counts": status_counts,
            },
            records=records,
            findings=findings,
        )

        return self.generate_report(
            report_type=ReportType.INCIDENT,
            title="Industrial Incident Report",
            description=(
                "Industrial incident and safety "
                "investigation summary."
            ),
            company_id=company_id,
            site_id=site_id,
            sections=[section],
            summary=section.metrics,
            recommendations=[
                "Review unresolved incidents and "
                "verify corrective-action completion.",
                "Escalate critical incidents according "
                "to site emergency procedures.",
            ],
            created_by=created_by,
        )

    # ========================================================
    # FINANCIAL REPORT
    # ========================================================

    def generate_financial_report(
        self,
        *,
        company_id: str,
        cost_records: Sequence[
            Mapping[str, Any]
        ],
        currency: str = "INR",
        site_id: Optional[str] = None,
        created_by: str = "system",
    ) -> GeneratedReport:

        selected_currency = required_text(
            currency,
            "currency",
            3,
        ).upper()

        if (
            len(selected_currency) != 3
            or not selected_currency.isascii()
            or not selected_currency.isalpha()
        ):
            raise ReportValidationError(
                "Currency must be a three-letter code"
            )

        records = [
            serialize(dict(record))
            for record in cost_records
        ]

        total_cost = ZERO
        category_totals: dict[str, Decimal] = {}

        for record in records:
            record_currency = record.get("currency")

            if (
                record_currency is not None
                and str(record_currency).upper()
                != selected_currency
            ):
                raise ReportValidationError(
                    "Mixed currencies require conversion"
                )

            amount = to_decimal(
                record.get("amount", 0),
                "amount",
                ZERO,
            )

            category = str(
                record.get("category", "other")
            )

            total_cost += amount

            category_totals[category] = (
                category_totals.get(category, ZERO)
                + amount
            )

        metrics = {
            "currency": selected_currency,
            "total_records": len(records),
            "total_cost": str(money(total_cost)),
            "category_totals": {
                category: str(money(amount))
                for category, amount
                in category_totals.items()
            },
        }

        section = self.create_section(
            title="Financial Cost Breakdown",
            description=(
                "Operational, maintenance and "
                "industrial cost analysis."
            ),
            metrics=metrics,
            records=records,
        )

        return self.generate_report(
            report_type=ReportType.FINANCIAL,
            title="Industrial Financial Report",
            description=(
                "Financial analysis of industrial "
                "operating and maintenance costs."
            ),
            company_id=company_id,
            site_id=site_id,
            sections=[section],
            summary=metrics,
            recommendations=[
                "Review major cost categories for "
                "potential efficiency improvements.",
                "Compare planned budgets with "
                "actual operational expenditure.",
            ],
            created_by=created_by,
            metadata={
                "currency": selected_currency,
            },
        )

    # ========================================================
    # EXECUTIVE REPORT
    # ========================================================

    def generate_executive_report(
        self,
        *,
        company_id: str,
        asset_health_records: Sequence[
            Mapping[str, Any]
        ] = (),
        prediction_records: Sequence[
            Mapping[str, Any]
        ] = (),
        incident_records: Sequence[
            Mapping[str, Any]
        ] = (),
        maintenance_records: Sequence[
            Mapping[str, Any]
        ] = (),
        financial_records: Sequence[
            Mapping[str, Any]
        ] = (),
        site_id: Optional[str] = None,
        created_by: str = "system",
    ) -> GeneratedReport:

        assets = [
            serialize(dict(item))
            for item in asset_health_records
        ]

        predictions = [
            serialize(dict(item))
            for item in prediction_records
        ]

        incidents = [
            serialize(dict(item))
            for item in incident_records
        ]

        maintenance = [
            serialize(dict(item))
            for item in maintenance_records
        ]

        financial = [
            serialize(dict(item))
            for item in financial_records
        ]

        scores = [
            to_decimal(
                record["health_score"],
                "health_score",
                ZERO,
            )
            for record in assets
            if record.get("health_score") is not None
        ]

        if any(score > 100 for score in scores):
            raise ReportValidationError(
                "health_score cannot exceed 100"
            )

        high_risk_predictions = sum(
            1
            for record in predictions
            if str(
                record.get("risk_level", "")
            ).lower() in {"high", "critical"}
        )

        critical_incidents = sum(
            1
            for record in incidents
            if str(
                record.get("severity", "")
            ).lower() == "critical"
        )

        completed_maintenance = sum(
            1
            for record in maintenance
            if str(
                record.get("status", "")
            ).lower() == "completed"
        )

        total_cost = sum(
            (
                to_decimal(
                    record.get("amount", 0),
                    "amount",
                    ZERO,
                )
                for record in financial
            ),
            ZERO,
        )

        currencies = {
            str(record["currency"]).upper()
            for record in financial
            if record.get("currency") is not None
        }

        if len(currencies) > 1:
            raise ReportValidationError(
                "Executive report financial records "
                "contain mixed currencies"
            )

        average_health = self.calculate_average(
            scores
        )

        summary = {
            "total_assets": len(assets),
            "average_asset_health": (
                str(average_health)
                if average_health is not None
                else None
            ),
            "total_predictions": len(predictions),
            "high_risk_predictions": (
                high_risk_predictions
            ),
            "total_incidents": len(incidents),
            "critical_incidents": critical_incidents,
            "total_maintenance_records": len(
                maintenance
            ),
            "completed_maintenance": (
                completed_maintenance
            ),
            "total_recorded_cost": str(
                money(total_cost)
            ),
            "currency": (
                next(iter(currencies))
                if currencies
                else None
            ),
        }

        sections = [
            self.create_section(
                title="Asset Health",
                description="Fleet health overview.",
                metrics={
                    "total_assets": len(assets),
                    "average_health_score": (
                        summary["average_asset_health"]
                    ),
                },
                records=assets,
            ),
            self.create_section(
                title="Predictive Maintenance",
                description="Equipment failure predictions.",
                metrics={
                    "total_predictions": len(predictions),
                    "high_risk_predictions": (
                        high_risk_predictions
                    ),
                },
                records=predictions,
            ),
            self.create_section(
                title="Safety and Incidents",
                description="Incident overview.",
                metrics={
                    "total_incidents": len(incidents),
                    "critical_incidents": (
                        critical_incidents
                    ),
                },
                records=incidents,
            ),
            self.create_section(
                title="Maintenance Operations",
                description="Maintenance execution summary.",
                metrics={
                    "total_maintenance_records": len(
                        maintenance
                    ),
                    "completed_maintenance": (
                        completed_maintenance
                    ),
                },
                records=maintenance,
            ),
            self.create_section(
                title="Financial Performance",
                description="Recorded cost overview.",
                metrics={
                    "total_recorded_cost": str(
                        money(total_cost)
                    ),
                    "currency": summary["currency"],
                },
                records=financial,
            ),
        ]

        recommendations = []

        if high_risk_predictions:
            recommendations.append(
                "Prioritize validation and review of "
                "high-risk equipment predictions."
            )

        if critical_incidents:
            recommendations.append(
                "Review critical incident investigations "
                "and verify corrective actions."
            )

        if average_health is not None and average_health < 60:
            recommendations.append(
                "Review fleet-wide asset condition "
                "and maintenance priorities."
            )

        return self.generate_report(
            report_type=ReportType.EXECUTIVE,
            title="Industrial Operations Executive Report",
            description=(
                "Executive overview of asset health, "
                "predictive maintenance, safety, "
                "operations and financial performance."
            ),
            company_id=company_id,
            site_id=site_id,
            sections=sections,
            summary=summary,
            recommendations=recommendations,
            created_by=created_by,
        )

    # ========================================================
    # EXPORT: JSON
    # ========================================================

    @staticmethod
    def export_json(
        report: GeneratedReport,
    ) -> str:

        return json.dumps(
            report.to_dict(),
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # EXPORT: CSV
    # ========================================================

    @staticmethod
    def export_csv(
        report: GeneratedReport,
    ) -> str:

        output = io.StringIO(
            newline=""
        )

        writer = csv.writer(output)

        writer.writerow([
            "section",
            "record_index",
            "field",
            "value",
        ])

        for section in report.sections:
            for index, record in enumerate(
                section.records,
                start=1,
            ):
                flattened = flatten_mapping(
                    record
                )

                for key, value in flattened.items():
                    writer.writerow([
                        safe_csv_cell(section.title),
                        index,
                        safe_csv_cell(key),
                        safe_csv_cell(value),
                    ])

            for key, value in flatten_mapping(
                section.metrics
            ).items():
                writer.writerow([
                    safe_csv_cell(section.title),
                    0,
                    safe_csv_cell(f"metric.{key}"),
                    safe_csv_cell(value),
                ])

        return output.getvalue()

    # ========================================================
    # EXPORT: MARKDOWN
    # ========================================================

    @staticmethod
    def export_markdown(
        report: GeneratedReport,
    ) -> str:

        lines = [
            f"# {report.title}",
            "",
            report.description,
            "",
            f"**Report ID:** {report.report_id}",
            f"**Report Type:** {report.report_type}",
            f"**Company:** {report.company_id}",
            f"**Generated:** {report.generated_at.isoformat()}",
            "",
            "## Executive Summary",
            "",
        ]

        for key, value in report.summary.items():
            lines.append(
                f"- **{key}:** {value}"
            )

        for section in report.sections:
            lines.extend([
                "",
                f"## {section.title}",
                "",
                section.description,
                "",
            ])

            for key, value in section.metrics.items():
                lines.append(
                    f"- **{key}:** {value}"
                )

            if section.findings:
                lines.extend([
                    "",
                    "### Findings",
                    "",
                ])

                for finding in section.findings:
                    lines.append(
                        f"- **[{finding.severity.upper()}] "
                        f"{finding.title}:** "
                        f"{finding.description}"
                    )

        if report.recommendations:
            lines.extend([
                "",
                "## Recommendations",
                "",
            ])

            for recommendation in report.recommendations:
                lines.append(
                    f"- {recommendation}"
                )

        return "\n".join(lines) + "\n"

    # ========================================================
    # EXPORT: HTML
    # ========================================================

    @staticmethod
    def export_html(
        report: GeneratedReport,
    ) -> str:

        def esc(value: Any) -> str:
            return html.escape(
                str(value),
                quote=True,
            )

        sections_html = []

        for section in report.sections:
            metrics_html = "".join(
                (
                    "<tr>"
                    f"<th>{esc(key)}</th>"
                    f"<td>{esc(value)}</td>"
                    "</tr>"
                )
                for key, value in section.metrics.items()
            )

            findings_html = "".join(
                (
                    '<div class="finding">'
                    f"<strong>{esc(finding.severity.upper())}"
                    f" - {esc(finding.title)}</strong>"
                    f"<p>{esc(finding.description)}</p>"
                    "</div>"
                )
                for finding in section.findings
            )

            sections_html.append(
                "<section>"
                f"<h2>{esc(section.title)}</h2>"
                f"<p>{esc(section.description)}</p>"
                f"<table>{metrics_html}</table>"
                f"{findings_html}"
                "</section>"
            )

        recommendations_html = "".join(
            f"<li>{esc(item)}</li>"
            for item in report.recommendations
        )

        summary_html = "".join(
            (
                "<tr>"
                f"<th>{esc(key)}</th>"
                f"<td>{esc(value)}</td>"
                "</tr>"
            )
            for key, value in report.summary.items()
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(report.title)}</title>
<style>
* {{
    box-sizing: border-box;
}}
body {{
    margin: 0;
    padding: 32px;
    background: #f4f7fb;
    color: #172033;
    font-family: Arial, Helvetica, sans-serif;
    line-height: 1.6;
}}
.container {{
    max-width: 1100px;
    margin: auto;
}}
header, section {{
    background: white;
    padding: 28px;
    margin-bottom: 22px;
    border: 1px solid #e3e8f0;
    border-radius: 12px;
}}
h1 {{
    margin-top: 0;
    font-size: 30px;
}}
h2 {{
    margin-top: 0;
    font-size: 22px;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 16px;
}}
th, td {{
    border-bottom: 1px solid #e3e8f0;
    padding: 12px;
    text-align: left;
    vertical-align: top;
    overflow-wrap: anywhere;
}}
th {{
    width: 40%;
    color: #526176;
}}
.finding {{
    padding: 14px;
    margin-top: 12px;
    border-left: 4px solid #e6a23c;
    background: #fffaf0;
}}
.meta {{
    color: #526176;
    font-size: 14px;
}}
@media print {{
    body {{
        background: white;
        padding: 0;
    }}
    header, section {{
        border: none;
        break-inside: avoid;
    }}
}}
</style>
</head>
<body>
<div class="container">
<header>
<h1>{esc(report.title)}</h1>
<p>{esc(report.description)}</p>
<p class="meta">
Report ID: {esc(report.report_id)}
</p>
<p class="meta">
Generated: {esc(report.generated_at.isoformat())}
</p>
</header>
<section>
<h2>Executive Summary</h2>
<table>{summary_html}</table>
</section>
{''.join(sections_html)}
<section>
<h2>Recommendations</h2>
<ul>{recommendations_html}</ul>
</section>
</div>
</body>
</html>"""

    # ========================================================
    # GENERIC EXPORT
    # ========================================================

    def export_report(
        self,
        report: GeneratedReport,
        export_format: str | ReportFormat,
    ) -> ReportExport:

        if not isinstance(report, GeneratedReport):
            raise ReportValidationError(
                "Expected GeneratedReport"
            )

        selected_format = parse_enum(
            ReportFormat,
            export_format,
            "export_format",
        )

        exporters = {
            ReportFormat.JSON: (
                self.export_json,
                "application/json",
                "json",
            ),
            ReportFormat.CSV: (
                self.export_csv,
                "text/csv",
                "csv",
            ),
            ReportFormat.HTML: (
                self.export_html,
                "text/html",
                "html",
            ),
            ReportFormat.MARKDOWN: (
                self.export_markdown,
                "text/markdown",
                "md",
            ),
        }

        exporter, content_type, extension = exporters[
            selected_format
        ]

        try:
            content = exporter(report)
        except ReportServiceError:
            raise
        except Exception as exc:
            raise ReportExportError(
                "Unable to export report"
            ) from exc

        return ReportExport(
            report_id=report.report_id,
            format=selected_format.value,
            filename=(
                f"{report.report_type}_"
                f"{report.report_id}.{extension}"
            ),
            content_type=content_type,
            content=content,
            generated_at=utc_now(),
        )

    # ========================================================
    # AUDIT
    # ========================================================

    async def _audit(
        self,
        *,
        report_id: str,
        actor_id: str,
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
                    "entity_type": "report",
                    "entity_id": report_id,
                    "actor_id": actor_id,
                    "action": action,
                    "details": dict(details or {}),
                    "timestamp": utc_now().isoformat(),
                },
            )

        except Exception:
            logger.exception(
                "Report audit failed: %s",
                report_id,
            )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    async def save_report(
        self,
        report: GeneratedReport,
    ) -> Any:

        if self.repository is None:
            raise ReportRepositoryError(
                "Report repository is not configured"
            )

        if not isinstance(report, GeneratedReport):
            raise ReportValidationError(
                "Expected GeneratedReport"
            )

        saved = await invoke(
            self.repository.create_report,
            report.to_dict(),
        )

        await self._audit(
            report_id=report.report_id,
            actor_id=report.created_by,
            action="report_generated",
            details={
                "report_type": report.report_type,
                "company_id": report.company_id,
                "site_id": report.site_id,
            },
        )

        return saved

    # ========================================================
    # GET REPORT
    # ========================================================

    async def get_report(
        self,
        report_id: str,
    ) -> Any:

        if self.repository is None:
            raise ReportRepositoryError(
                "Report repository is not configured"
            )

        identifier = required_text(
            report_id,
            "report_id",
        )

        record = await invoke(
            self.repository.get_report,
            identifier,
        )

        if record is None:
            raise ReportNotFoundError(
                f"Report not found: {identifier}"
            )

        return record

    # ========================================================
    # LIST REPORTS
    # ========================================================

    async def list_reports(
        self,
        *,
        company_id: Optional[str] = None,
        site_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        report_type: Optional[str] = None,
    ) -> Any:

        if self.repository is None:
            raise ReportRepositoryError(
                "Report repository is not configured"
            )

        filters: dict[str, Any] = {}

        for name, value in {
            "company_id": company_id,
            "site_id": site_id,
            "asset_id": asset_id,
        }.items():
            if value is not None:
                filters[name] = required_text(
                    value,
                    name,
                )

        if report_type is not None:
            filters["report_type"] = parse_enum(
                ReportType,
                report_type,
                "report_type",
            ).value

        return await invoke(
            self.repository.list_reports,
            filters,
        )


# ============================================================
# IN-MEMORY REPOSITORY
# ============================================================

class InMemoryReportRepository:
    """
    Development-only report repository.

    Not durable or concurrency-safe.
    """

    def __init__(self):
        self.records: dict[
            str,
            dict[str, Any],
        ] = {}

    async def create_report(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:

        record = copy.deepcopy(data)
        report_id = record["report_id"]

        if report_id in self.records:
            raise ReportRepositoryError(
                "Duplicate report ID"
            )

        self.records[report_id] = record

        return copy.deepcopy(record)

    async def get_report(
        self,
        report_id: str,
    ) -> Optional[dict[str, Any]]:

        record = self.records.get(report_id)

        return (
            copy.deepcopy(record)
            if record is not None
            else None
        )

    async def list_reports(
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

    repository = InMemoryReportRepository()

    service = ReportService(
        report_repository=repository,
        default_company_id="COMPANY-001",
    )

    # --------------------------------------------------------
    # ASSET HEALTH REPORT
    # --------------------------------------------------------

    health_report = (
        service.generate_asset_health_report(
            company_id="COMPANY-001",
            site_id="SITE-001",
            asset_records=[
                {
                    "asset_id": "MACHINE-001",
                    "name": "Industrial Motor",
                    "health_score": 82,
                    "status": "operational",
                },
                {
                    "asset_id": "MACHINE-002",
                    "name": "Hydraulic Pump",
                    "health_score": 35,
                    "status": "degraded",
                },
                {
                    "asset_id": "MACHINE-003",
                    "name": "Air Compressor",
                    "health_score": 71,
                    "status": "operational",
                },
            ],
            created_by="ENGINEER-001",
        )
    )

    await service.save_report(
        health_report
    )

    print("\nASSET HEALTH REPORT")

    print(
        json.dumps(
            health_report.to_dict(),
            indent=2,
        )
    )

    # --------------------------------------------------------
    # PREDICTIVE MAINTENANCE REPORT
    # --------------------------------------------------------

    predictive_report = (
        service.generate_predictive_maintenance_report(
            company_id="COMPANY-001",
            site_id="SITE-001",
            prediction_records=[
                {
                    "asset_id": "MACHINE-001",
                    "risk_level": "low",
                    "failure_probability": 0.12,
                },
                {
                    "asset_id": "MACHINE-002",
                    "risk_level": "critical",
                    "failure_probability": 0.88,
                },
            ],
        )
    )

    await service.save_report(
        predictive_report
    )

    # --------------------------------------------------------
    # INCIDENT REPORT
    # --------------------------------------------------------

    incident_report = (
        service.generate_incident_report(
            company_id="COMPANY-001",
            site_id="SITE-001",
            incident_records=[
                {
                    "incident_id": "INC-001",
                    "asset_id": "MACHINE-002",
                    "severity": "high",
                    "status": "investigating",
                    "description": (
                        "Hydraulic pump overheating"
                    ),
                },
                {
                    "incident_id": "INC-002",
                    "asset_id": "MACHINE-003",
                    "severity": "medium",
                    "status": "resolved",
                    "description": (
                        "Abnormal compressor vibration"
                    ),
                },
            ],
        )
    )

    await service.save_report(
        incident_report
    )

    # --------------------------------------------------------
    # EXECUTIVE REPORT
    # --------------------------------------------------------

    executive_report = (
        service.generate_executive_report(
            company_id="COMPANY-001",
            site_id="SITE-001",
            asset_health_records=[
                {
                    "asset_id": "MACHINE-001",
                    "health_score": 82,
                },
                {
                    "asset_id": "MACHINE-002",
                    "health_score": 35,
                },
            ],
            prediction_records=[
                {
                    "asset_id": "MACHINE-002",
                    "risk_level": "critical",
                },
            ],
            incident_records=[
                {
                    "incident_id": "INC-001",
                    "severity": "critical",
                },
            ],
            maintenance_records=[
                {
                    "maintenance_id": "MAINT-001",
                    "status": "completed",
                },
                {
                    "maintenance_id": "MAINT-002",
                    "status": "scheduled",
                },
            ],
            financial_records=[
                {
                    "category": "repair",
                    "amount": 15000,
                    "currency": "INR",
                },
                {
                    "category": "maintenance",
                    "amount": 5000,
                    "currency": "INR",
                },
            ],
        )
    )

    await service.save_report(
        executive_report
    )

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    json_export = service.export_report(
        executive_report,
        ReportFormat.JSON,
    )

    csv_export = service.export_report(
        executive_report,
        ReportFormat.CSV,
    )

    html_export = service.export_report(
        executive_report,
        ReportFormat.HTML,
    )

    markdown_export = service.export_report(
        executive_report,
        ReportFormat.MARKDOWN,
    )

    print("\nEXPORT FILES")

    for export in [
        json_export,
        csv_export,
        html_export,
        markdown_export,
    ]:
        print(
            export.filename,
            export.content_type,
            len(export.content),
        )

    # --------------------------------------------------------
    # REPORT LIST
    # --------------------------------------------------------

    reports = await service.list_reports(
        company_id="COMPANY-001"
    )

    print("\nREPORT SUMMARY")

    print(
        json.dumps(
            {
                "total_reports": len(reports),
                "report_types": [
                    report["report_type"]
                    for report in reports
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())

"""
app/services/report_service.py

Industrial Predictive Maintenance, Safety & Emergency AI
========================================================

Enterprise Report Generation Service

Features:
- Asset health reports
- Predictive maintenance reports
- Failure prediction reports
- Anomaly detection reports
- Inspection reports
- Industrial safety reports
- Incident and emergency reports
- Maintenance performance reports
- Repair and replacement cost reports
- Production loss reports
- Financial and operational summaries
- Executive dashboard reports
- Custom reports
- JSON, CSV, HTML and Markdown exports
- Time-range filtering
- Report statistics and aggregation
- Optional asynchronous repository integration
- Optional audit integration
- In-memory demonstration repository

Expected repository interface:
    async create_report(data: dict)
    async get_report(report_id: str)
    async list_reports(filters: dict)

Optional audit interface:
    async record_event(data: dict)

Notes:
- Source data must be tenant-authorized before reporting.
- CSV output includes spreadsheet-formula injection protection.
- HTML output escapes untrusted text.
- Generated reports are snapshots, not live monitoring.
- PDF generation can be implemented by a separate exporter.
"""

from __future__ import annotations

import asyncio
import copy
import csv
import html
import inspect
import io
import json
import logging
import math
import re

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4


logger = logging.getLogger(__name__)

MONEY_QUANT = Decimal("0.01")
ZERO = Decimal("0")


# ============================================================
# EXCEPTIONS
# ============================================================

class ReportServiceError(Exception):
    """Base report service exception."""


class ReportValidationError(ReportServiceError):
    """Invalid report request or source data."""


class ReportNotFoundError(ReportServiceError):
    """Requested report does not exist."""


class ReportRepositoryError(ReportServiceError):
    """Report repository operation failed."""


class ReportExportError(ReportServiceError):
    """Report export operation failed."""


# ============================================================
# ENUMS
# ============================================================

class ReportType(str, Enum):
    ASSET_HEALTH = "asset_health"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    FAILURE_PREDICTION = "failure_prediction"
    ANOMALY = "anomaly"
    INSPECTION = "inspection"
    SAFETY = "safety"
    INCIDENT = "incident"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"
    REPAIR_COST = "repair_cost"
    REPLACEMENT_COST = "replacement_cost"
    PRODUCTION_LOSS = "production_loss"
    FINANCIAL = "financial"
    EXECUTIVE = "executive"
    CUSTOM = "custom"


class ReportFormat(str, Enum):
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    MARKDOWN = "markdown"


class ReportStatus(str, Enum):
    GENERATED = "generated"
    ARCHIVED = "archived"


class ReportSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def required_text(
    value: Any,
    name: str,
    max_length: int = 255,
) -> str:

    if not isinstance(value, str):
        raise ReportValidationError(
            f"{name} must be a string"
        )

    result = value.strip()

    if not result:
        raise ReportValidationError(
            f"{name} cannot be empty"
        )

    if len(result) > max_length:
        raise ReportValidationError(
            f"{name} exceeds {max_length} characters"
        )

    return result


def optional_text(
    value: Optional[str],
    name: str,
    max_length: int = 255,
) -> Optional[str]:

    if value is None:
        return None

    return required_text(value, name, max_length)


def parse_enum(
    enum_class: type[Enum],
    value: Any,
    name: str,
) -> Enum:

    try:
        return enum_class(value)
    except (ValueError, TypeError) as exc:
        raise ReportValidationError(
            f"Invalid {name}: {value}"
        ) from exc


def validate_datetime(
    value: datetime,
    name: str,
) -> datetime:

    if not isinstance(value, datetime):
        raise ReportValidationError(
            f"{name} must be datetime"
        )

    if value.tzinfo is None or value.utcoffset() is None:
        raise ReportValidationError(
            f"{name} must be timezone-aware"
        )

    return value


def to_decimal(
    value: Any,
    name: str,
    minimum: Optional[Decimal] = None,
) -> Decimal:

    if isinstance(value, bool):
        raise ReportValidationError(
            f"{name} must be numeric"
        )

    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ReportValidationError(
            f"Invalid numeric value for {name}"
        ) from exc

    if not result.is_finite():
        raise ReportValidationError(
            f"{name} must be finite"
        )

    if minimum is not None and result < minimum:
        raise ReportValidationError(
            f"{name} must be >= {minimum}"
        )

    return result


def money(value: Any) -> Decimal:
    return to_decimal(
        value,
        "money",
    ).quantize(
        MONEY_QUANT,
        rounding=ROUND_HALF_UP,
    )


def serialize(value: Any) -> Any:

    if isinstance(value, Decimal):
        return str(value)

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
        return [
            serialize(item)
            for item in value
        ]

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReportValidationError(
                "Non-finite floating-point value"
            )

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


def safe_csv_cell(value: Any) -> str:
    """
    Protect CSV output against spreadsheet formula injection.

    The prefix is applied after leading whitespace is checked,
    so cells containing spaces before formula characters are
    also protected.
    """

    if value is None:
        return ""

    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(
            serialize(value),
            ensure_ascii=False,
        )
    else:
        text = str(value)

    if re.match(r"^[\s\x00-\x1f]*[=+\-@]", text):
        return "'" + text

    return text


def flatten_mapping(
    data: Mapping[str, Any],
    prefix: str = "",
) -> dict[str, Any]:

    result: dict[str, Any] = {}

    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, Mapping):
            result.update(
                flatten_mapping(value, path)
            )
        else:
            result[path] = value

    return result


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class ReportPeriod:
    start: Optional[datetime] = None
    end: Optional[datetime] = None

    def __post_init__(self) -> None:

        if self.start is not None:
            validate_datetime(
                self.start,
                "period.start",
            )

        if self.end is not None:
            validate_datetime(
                self.end,
                "period.end",
            )

        if (
            self.start is not None
            and self.end is not None
            and self.end <= self.start
        ):
            raise ReportValidationError(
                "Report period end must be after start"
            )


@dataclass
class ReportFinding:
    finding_id: str
    title: str
    description: str
    severity: str
    source: Optional[str] = None
    asset_id: Optional[str] = None
    recommendation: Optional[str] = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass
class ReportSection:
    section_id: str
    title: str
    description: str
    metrics: dict[str, Any] = field(
        default_factory=dict
    )
    records: list[dict[str, Any]] = field(
        default_factory=list
    )
    findings: list[ReportFinding] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass
class GeneratedReport:
    report_id: str
    report_type: str
    title: str
    description: str
    company_id: str
    site_id: Optional[str]
    asset_id: Optional[str]
    status: str
    period: ReportPeriod
    sections: list[ReportSection]
    summary: dict[str, Any]
    recommendations: list[str]
    created_by: str
    generated_at: datetime
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


@dataclass(frozen=True)
class ReportExport:
    report_id: str
    format: str
    filename: str
    content_type: str
    content: str
    generated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return serialize(self)


# ============================================================
# REPORT SERVICE
# ============================================================

class ReportService:
    """
    Enterprise reporting service for industrial AI.

    Source records must be provided by an authorized
    application layer or trusted data integration.

    No database access is required for report generation.
    """

    SEVERITY_ORDER = {
        ReportSeverity.INFO.value: 0,
        ReportSeverity.LOW.value: 1,
        ReportSeverity.MEDIUM.value: 2,
        ReportSeverity.HIGH.value: 3,
        ReportSeverity.CRITICAL.value: 4,
    }

    def __init__(
        self,
        report_repository: Any = None,
        audit_service: Any = None,
        default_company_id: Optional[str] = None,
    ):

        self.repository = report_repository
        self.audit = audit_service
        self.default_company_id = default_company_id

    # ========================================================
    # FINDING CREATION
    # ========================================================

    @staticmethod
    def create_finding(
        *,
        title: str,
        description: str,
        severity: str | ReportSeverity = (
            ReportSeverity.INFO
        ),
        source: Optional[str] = None,
        asset_id: Optional[str] = None,
        recommendation: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ReportFinding:

        selected_severity = parse_enum(
            ReportSeverity,
            severity,
            "severity",
        )

        return ReportFinding(
            finding_id=str(uuid4()),
            title=required_text(
                title,
                "title",
            ),
            description=required_text(
                description,
                "description",
                5000,
            ),
            severity=selected_severity.value,
            source=optional_text(
                source,
                "source",
            ),
            asset_id=optional_text(
                asset_id,
                "asset_id",
            ),
            recommendation=optional_text(
                recommendation,
                "recommendation",
                2000,
            ),
            metadata=dict(metadata or {}),
        )

    # ========================================================
    # SECTION CREATION
    # ========================================================

    @staticmethod
    def create_section(
        *,
        title: str,
        description: str,
        metrics: Optional[Mapping[str, Any]] = None,
        records: Optional[
            Sequence[Mapping[str, Any]]
        ] = None,
        findings: Optional[
            Sequence[ReportFinding]
        ] = None,
    ) -> ReportSection:

        normalized_records = []

        for record in records or []:
            if not isinstance(record, Mapping):
                raise ReportValidationError(
                    "Report records must be mappings"
                )

            normalized_records.append(
                serialize(dict(record))
            )

        normalized_findings = []

        for finding in findings or []:
            if not isinstance(
                finding,
                ReportFinding,
            ):
                raise ReportValidationError(
                    "findings must contain ReportFinding"
                )

            normalized_findings.append(finding)

        return ReportSection(
            section_id=str(uuid4()),
            title=required_text(
                title,
                "title",
            ),
            description=required_text(
                description,
                "description",
                5000,
            ),
            metrics=serialize(
                dict(metrics or {})
            ),
            records=normalized_records,
            findings=normalized_findings,
        )

    # ========================================================
    # METRIC HELPERS
    # ========================================================

    @staticmethod
    def calculate_average(
        values: Sequence[Any],
    ) -> Optional[Decimal]:

        if not values:
            return None

        numeric = [
            to_decimal(value, "metric")
            for value in values
        ]

        return (
            sum(numeric, ZERO)
            / Decimal(len(numeric))
        ).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )

    @staticmethod
    def calculate_total(
        values: Sequence[Any],
    ) -> Decimal:

        return sum(
            (
                to_decimal(value, "metric")
                for value in values
            ),
            ZERO,
        )

    @staticmethod
    def calculate_percentage(
        numerator: Any,
        denominator: Any,
    ) -> Decimal:

        numerator_value = to_decimal(
            numerator,
            "numerator",
        )

        denominator_value = to_decimal(
            denominator,
            "denominator",
        )

        if denominator_value == ZERO:
            return ZERO

        return (
            numerator_value
            / denominator_value
            * Decimal("100")
        ).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )

    # ========================================================
    # GENERIC REPORT CREATION
    # ========================================================

    def generate_report(
        self,
        *,
        report_type: str | ReportType,
        title: str,
        description: str,
        company_id: Optional[str] = None,
        site_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        sections: Optional[
            Sequence[ReportSection]
        ] = None,
        summary: Optional[Mapping[str, Any]] = None,
        recommendations: Optional[
            Sequence[str]
        ] = None,
        created_by: str = "system",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> GeneratedReport:

        selected_type = parse_enum(
            ReportType,
            report_type,
            "report_type",
        )

        selected_company_id = (
            company_id
            or self.default_company_id
        )

        if selected_company_id is None:
            raise ReportValidationError(
                "company_id is required"
            )

        period = ReportPeriod(
            start=period_start,
            end=period_end,
        )

        normalized_sections = []

        for section in sections or []:
            if not isinstance(
                section,
                ReportSection,
            ):
                raise ReportValidationError(
                    "sections must contain ReportSection"
                )

            normalized_sections.append(section)

        normalized_recommendations = [
            required_text(
                recommendation,
                "recommendation",
                2000,
            )
            for recommendation in (
                recommendations or []
            )
        ]

        return GeneratedReport(
            report_id=str(uuid4()),
            report_type=selected_type.value,
            title=required_text(
                title,
                "title",
            ),
            description=required_text(
                description,
                "description",
                5000,
            ),
            company_id=required_text(
                selected_company_id,
                "company_id",
            ),
            site_id=optional_text(
                site_id,
                "site_id",
            ),
            asset_id=optional_text(
                asset_id,
                "asset_id",
            ),
            status=ReportStatus.GENERATED.value,
            period=period,
            sections=normalized_sections,
            summary=serialize(
                dict(summary or {})
            ),
            recommendations=normalized_recommendations,
            created_by=required_text(
                created_by,
                "created_by",
            ),
            generated_at=utc_now(),
            metadata=serialize(
                dict(metadata or {})
            ),
        )

    # ========================================================
    # ASSET HEALTH REPORT
    # ========================================================

    def generate_asset_health_report(
        self,
        *,
        company_id: str,
        asset_records: Sequence[
            Mapping[str, Any]
        ],
        site_id: Optional[str] = None,
        created_by: str = "system",
    ) -> GeneratedReport:

        records = [
            serialize(dict(record))
            for record in asset_records
        ]

        scores: list[Decimal] = []
        critical_assets: list[str] = []
        findings: list[ReportFinding] = []

        for record in records:
            asset_id = str(
                record.get("asset_id", "unknown")
            )

            raw_score = record.get(
                "health_score"
            )

            if raw_score is None:
                continue

            score = to_decimal(
                raw_score,
                "health_score",
                ZERO,
            )

            if score > 100:
                raise ReportValidationError(
                    "health_score cannot exceed 100"
                )

            scores.append(score)

            if score < 40:
                critical_assets.append(asset_id)

                findings.append(
                    self.create_finding(
                        title="Critical Asset Health",
                        description=(
                            f"Asset {asset_id} has "
                            f"health score {score}."
                        ),
                        severity="critical",
                        source="health_score_service",
                        asset_id=asset_id,
                        recommendation=(
                            "Arrange a qualified inspection "
                            "and review operating restrictions."
                        ),
                    )
                )

        average_score = self.calculate_average(
            scores
        )

        section = self.create_section(
            title="Asset Health Overview",
            description=(
                "Health scores and asset-level "
                "condition information."
            ),
            metrics={
                "total_assets": len(records),
                "assets_with_health_score": len(scores),
                "average_health_score": (
                    str(average_score)
                    if average_score is not None
                    else None
                ),
                "critical_assets": len(
                    critical_assets
                ),
            },
            records=records,
            findings=findings,
        )

        recommendations = []

        if critical_assets:
            recommendations.append(
                "Prioritize qualified inspection of "
                "assets with critical health scores."
            )

        return self.generate_report(
            report_type=ReportType.ASSET_HEALTH,
            title="Industrial Asset Health Report",
            description=(
                "Condition and health analysis "
                "of industrial assets."
            ),
            company_id=company_id,
            site_id=site_id,
            sections=[section],
            summary=section.metrics,
            recommendations=recommendations,
            created_by=created_by,
        )

    # ========================================================
    # PREDICTIVE MAINTENANCE REPORT
    # ========================================================

    def generate_predictive_maintenance_report(
        self,
        *,
        company_id: str,
        prediction_records: Sequence[
            Mapping[str, Any]
        ],
        site_id: Optional[str] = None,
        created_by: str = "system",
    ) -> GeneratedReport:

        records = [
            serialize(dict(record))
            for record in prediction_records
        ]

        high_risk = []
        findings = []

        for record in records:
            risk = str(
                record.get("risk_level", "")
            ).lower()

            asset_id = str(
                record.get("asset_id", "unknown")
            )

            if risk in {"high", "critical"}:
                high_risk.append(record)

                findings.append(
                    self.create_finding(
                        title="Elevated Failure Risk",
                        description=(
                            f"Asset {asset_id} has "
                            f"{risk} predicted failure risk."
                        ),
                        severity=risk,
                        source="failure_prediction_service",
                        asset_id=asset_id,
                        recommendation=(
                            "Review model confidence, "
                            "telemetry and maintenance history "
                            "before scheduling intervention."
                        ),
                    )
                )

        section = self.create_section(
            title="Failure Risk Predictions",
            description=(
                "Model-generated asset failure "
                "risk predictions."
            ),
            metrics={
                "total_predictions": len(records),
                "high_or_critical_predictions": len(
                    high_risk
                ),
            },
            records=records,
            findings=findings,
        )

        return self.generate_report(
            report_type=ReportType.PREDICTIVE_MAINTENANCE,
            title="Predictive Maintenance Report",
            description=(
                "Industrial equipment failure risk "
                "and maintenance planning report."
            ),
            company_id=company_id,
            site_id=site_id,
            sections=[section],
            summary=section.metrics,
            recommendations=[
                "Validate high-risk predictions using "
                "recent telemetry and maintenance records.",
                "Prioritize maintenance by risk, "
                "operational impact and asset criticality.",
            ],
            created_by=created_by,
        )

    # ========================================================
    # ANOMALY REPORT
    # ========================================================

    def generate_anomaly_report(
        self,
        *,
        company_id: str,
        anomaly_records: Sequence[
            Mapping[str, Any]
        ],
        site_id: Optional[str] = None,
        created_by: str = "system",
    ) -> GeneratedReport:

        records = [
            serialize(dict(record))
            for record in anomaly_records
        ]

        severity_counts = {
            "low": 0,
            "medium": 0,
            "high": 0,
            "critical": 0,
        }

        findings = []

        for record in records:
            severity = str(
                record.get("severity", "low")
            ).lower()

            if severity in severity_counts:
                severity_counts[severity] += 1

            if severity in {"high", "critical"}:
                findings.append(
                    self.create_finding(
                        title="Significant Sensor Anomaly",
                        description=str(
                            record.get(
                                "description",
                                "Elevated anomaly detected.",
                            )
                        ),
                        severity=severity,
                        source="anomaly_service",
                        asset_id=(
                            str(record["asset_id"])
                            if record.get("asset_id")
                            else None
                        ),
                        recommendation=(
                            "Validate sensor readings and "
                            "investigate the underlying cause."
                        ),
                    )
                )

        section = self.create_section(
            title="Anomaly Detection Overview",
            description=(
                "Sensor and multivariate anomaly events."
            ),
            metrics={
                "total_anomalies": len(records),
                "severity_counts": severity_counts,
            },
            records=records,
            findings=findings,
        )

        return self.generate_report(
            report_type=ReportType.ANOMALY,
            title="Industrial Anomaly Detection Report",
            description=(
                "Detected equipment and sensor anomalies."
            ),
            company_id=company_id,
            site_id=site_id,
            sections=[section],
            summary=section.metrics,
            recommendations=[
                "Review high-severity anomalies "
                "against sensor calibration and history.",
            ],
            created_by=created_by,
        )

    # ========================================================
    # INCIDENT REPORT
    # ========================================================

    def generate_incident_report(
        self,
        *,
        company_id: str,
        incident_records: Sequence[
            Mapping[str, Any]
        ],
        site_id: Optional[str] = None,
        created_by: str = "system",
    ) -> GeneratedReport:

        records = [
            serialize(dict(record))
            for record in incident_records
        ]

        severity_counts: dict[str, int] = {}
        status_counts: dict[str, int] = {}
        findings = []

        for record in records:
            severity = str(
                record.get("severity", "unknown")
            ).lower()

            status = str(
                record.get("status", "unknown")
            ).lower()

            severity_counts[severity] = (
                severity_counts.get(severity, 0) + 1
            )

            status_counts[status] = (
                status_counts.get(status, 0) + 1
            )

            if severity in {"high", "critical"}:
                findings.append(
                    self.create_finding(
                        title="High-Severity Incident",
                        description=str(
                            record.get(
                                "description",
                                "High-severity incident recorded.",
                            )
                        ),
                        severity=severity,
                        source="incident_service",
                        asset_id=(
                            str(record["asset_id"])
                            if record.get("asset_id")
                            else None
                        ),
                        recommendation=(
                            "Follow the approved incident "
                            "response, investigation and "
                            "corrective-action procedures."
                        ),
                    )
                )

        section = self.create_section(
            title="Incident Overview",
            description=(
                "Incident severity, status and "
                "investigation records."
            ),
            metrics={
                "total_incidents": len(records),
                "severity_counts": severity_counts,
                "status_counts": status_counts,
            },
            records=records,
            findings=findings,
        )

        return self.generate_report(
            report_type=ReportType.INCIDENT,
            title="Industrial Incident Report",
            description=(
                "Industrial incident and safety "
                "investigation summary."
            ),
            company_id=company_id,
            site_id=site_id,
            sections=[section],
            summary=section.metrics,
            recommendations=[
                "Review unresolved incidents and "
                "verify corrective-action completion.",
                "Escalate critical incidents according "
                "to site emergency procedures.",
            ],
            created_by=created_by,
        )

    # ========================================================
    # FINANCIAL REPORT
    # ========================================================

    def generate_financial_report(
        self,
        *,
        company_id: str,
        cost_records: Sequence[
            Mapping[str, Any]
        ],
        currency: str = "INR",
        site_id: Optional[str] = None,
        created_by: str = "system",
    ) -> GeneratedReport:

        selected_currency = required_text(
            currency,
            "currency",
            3,
        ).upper()

        if (
            len(selected_currency) != 3
            or not selected_currency.isascii()
            or not selected_currency.isalpha()
        ):
            raise ReportValidationError(
                "Currency must be a three-letter code"
            )

        records = [
            serialize(dict(record))
            for record in cost_records
        ]

        total_cost = ZERO
        category_totals: dict[str, Decimal] = {}

        for record in records:
            record_currency = record.get("currency")

            if (
                record_currency is not None
                and str(record_currency).upper()
                != selected_currency
            ):
                raise ReportValidationError(
                    "Mixed currencies require conversion"
                )

            amount = to_decimal(
                record.get("amount", 0),
                "amount",
                ZERO,
            )

            category = str(
                record.get("category", "other")
            )

            total_cost += amount

            category_totals[category] = (
                category_totals.get(category, ZERO)
                + amount
            )

        metrics = {
            "currency": selected_currency,
            "total_records": len(records),
            "total_cost": str(money(total_cost)),
            "category_totals": {
                category: str(money(amount))
                for category, amount
                in category_totals.items()
            },
        }

        section = self.create_section(
            title="Financial Cost Breakdown",
            description=(
                "Operational, maintenance and "
                "industrial cost analysis."
            ),
            metrics=metrics,
            records=records,
        )

        return self.generate_report(
            report_type=ReportType.FINANCIAL,
            title="Industrial Financial Report",
            description=(
                "Financial analysis of industrial "
                "operating and maintenance costs."
            ),
            company_id=company_id,
            site_id=site_id,
            sections=[section],
            summary=metrics,
            recommendations=[
                "Review major cost categories for "
                "potential efficiency improvements.",
                "Compare planned budgets with "
                "actual operational expenditure.",
            ],
            created_by=created_by,
            metadata={
                "currency": selected_currency,
            },
        )

    # ========================================================
    # EXECUTIVE REPORT
    # ========================================================

    def generate_executive_report(
        self,
        *,
        company_id: str,
        asset_health_records: Sequence[
            Mapping[str, Any]
        ] = (),
        prediction_records: Sequence[
            Mapping[str, Any]
        ] = (),
        incident_records: Sequence[
            Mapping[str, Any]
        ] = (),
        maintenance_records: Sequence[
            Mapping[str, Any]
        ] = (),
        financial_records: Sequence[
            Mapping[str, Any]
        ] = (),
        site_id: Optional[str] = None,
        created_by: str = "system",
    ) -> GeneratedReport:

        assets = [
            serialize(dict(item))
            for item in asset_health_records
        ]

        predictions = [
            serialize(dict(item))
            for item in prediction_records
        ]

        incidents = [
            serialize(dict(item))
            for item in incident_records
        ]

        maintenance = [
            serialize(dict(item))
            for item in maintenance_records
        ]

        financial = [
            serialize(dict(item))
            for item in financial_records
        ]

        scores = [
            to_decimal(
                record["health_score"],
                "health_score",
                ZERO,
            )
            for record in assets
            if record.get("health_score") is not None
        ]

        if any(score > 100 for score in scores):
            raise ReportValidationError(
                "health_score cannot exceed 100"
            )

        high_risk_predictions = sum(
            1
            for record in predictions
            if str(
                record.get("risk_level", "")
            ).lower() in {"high", "critical"}
        )

        critical_incidents = sum(
            1
            for record in incidents
            if str(
                record.get("severity", "")
            ).lower() == "critical"
        )

        completed_maintenance = sum(
            1
            for record in maintenance
            if str(
                record.get("status", "")
            ).lower() == "completed"
        )

        total_cost = sum(
            (
                to_decimal(
                    record.get("amount", 0),
                    "amount",
                    ZERO,
                )
                for record in financial
            ),
            ZERO,
        )

        currencies = {
            str(record["currency"]).upper()
            for record in financial
            if record.get("currency") is not None
        }

        if len(currencies) > 1:
            raise ReportValidationError(
                "Executive report financial records "
                "contain mixed currencies"
            )

        average_health = self.calculate_average(
            scores
        )

        summary = {
            "total_assets": len(assets),
            "average_asset_health": (
                str(average_health)
                if average_health is not None
                else None
            ),
            "total_predictions": len(predictions),
            "high_risk_predictions": (
                high_risk_predictions
            ),
            "total_incidents": len(incidents),
            "critical_incidents": critical_incidents,
            "total_maintenance_records": len(
                maintenance
            ),
            "completed_maintenance": (
                completed_maintenance
            ),
            "total_recorded_cost": str(
                money(total_cost)
            ),
            "currency": (
                next(iter(currencies))
                if currencies
                else None
            ),
        }

        sections = [
            self.create_section(
                title="Asset Health",
                description="Fleet health overview.",
                metrics={
                    "total_assets": len(assets),
                    "average_health_score": (
                        summary["average_asset_health"]
                    ),
                },
                records=assets,
            ),
            self.create_section(
                title="Predictive Maintenance",
                description="Equipment failure predictions.",
                metrics={
                    "total_predictions": len(predictions),
                    "high_risk_predictions": (
                        high_risk_predictions
                    ),
                },
                records=predictions,
            ),
            self.create_section(
                title="Safety and Incidents",
                description="Incident overview.",
                metrics={
                    "total_incidents": len(incidents),
                    "critical_incidents": (
                        critical_incidents
                    ),
                },
                records=incidents,
            ),
            self.create_section(
                title="Maintenance Operations",
                description="Maintenance execution summary.",
                metrics={
                    "total_maintenance_records": len(
                        maintenance
                    ),
                    "completed_maintenance": (
                        completed_maintenance
                    ),
                },
                records=maintenance,
            ),
            self.create_section(
                title="Financial Performance",
                description="Recorded cost overview.",
                metrics={
                    "total_recorded_cost": str(
                        money(total_cost)
                    ),
                    "currency": summary["currency"],
                },
                records=financial,
            ),
        ]

        recommendations = []

        if high_risk_predictions:
            recommendations.append(
                "Prioritize validation and review of "
                "high-risk equipment predictions."
            )

        if critical_incidents:
            recommendations.append(
                "Review critical incident investigations "
                "and verify corrective actions."
            )

        if average_health is not None and average_health < 60:
            recommendations.append(
                "Review fleet-wide asset condition "
                "and maintenance priorities."
            )

        return self.generate_report(
            report_type=ReportType.EXECUTIVE,
            title="Industrial Operations Executive Report",
            description=(
                "Executive overview of asset health, "
                "predictive maintenance, safety, "
                "operations and financial performance."
            ),
            company_id=company_id,
            site_id=site_id,
            sections=sections,
            summary=summary,
            recommendations=recommendations,
            created_by=created_by,
        )

    # ========================================================
    # EXPORT: JSON
    # ========================================================

    @staticmethod
    def export_json(
        report: GeneratedReport,
    ) -> str:

        return json.dumps(
            report.to_dict(),
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # EXPORT: CSV
    # ========================================================

    @staticmethod
    def export_csv(
        report: GeneratedReport,
    ) -> str:

        output = io.StringIO(
            newline=""
        )

        writer = csv.writer(output)

        writer.writerow([
            "section",
            "record_index",
            "field",
            "value",
        ])

        for section in report.sections:
            for index, record in enumerate(
                section.records,
                start=1,
            ):
                flattened = flatten_mapping(
                    record
                )

                for key, value in flattened.items():
                    writer.writerow([
                        safe_csv_cell(section.title),
                        index,
                        safe_csv_cell(key),
                        safe_csv_cell(value),
                    ])

            for key, value in flatten_mapping(
                section.metrics
            ).items():
                writer.writerow([
                    safe_csv_cell(section.title),
                    0,
                    safe_csv_cell(f"metric.{key}"),
                    safe_csv_cell(value),
                ])

        return output.getvalue()

    # ========================================================
    # EXPORT: MARKDOWN
    # ========================================================

    @staticmethod
    def export_markdown(
        report: GeneratedReport,
    ) -> str:

        lines = [
            f"# {report.title}",
            "",
            report.description,
            "",
            f"**Report ID:** {report.report_id}",
            f"**Report Type:** {report.report_type}",
            f"**Company:** {report.company_id}",
            f"**Generated:** {report.generated_at.isoformat()}",
            "",
            "## Executive Summary",
            "",
        ]

        for key, value in report.summary.items():
            lines.append(
                f"- **{key}:** {value}"
            )

        for section in report.sections:
            lines.extend([
                "",
                f"## {section.title}",
                "",
                section.description,
                "",
            ])

            for key, value in section.metrics.items():
                lines.append(
                    f"- **{key}:** {value}"
                )

            if section.findings:
                lines.extend([
                    "",
                    "### Findings",
                    "",
                ])

                for finding in section.findings:
                    lines.append(
                        f"- **[{finding.severity.upper()}] "
                        f"{finding.title}:** "
                        f"{finding.description}"
                    )

        if report.recommendations:
            lines.extend([
                "",
                "## Recommendations",
                "",
            ])

            for recommendation in report.recommendations:
                lines.append(
                    f"- {recommendation}"
                )

        return "\n".join(lines) + "\n"

    # ========================================================
    # EXPORT: HTML
    # ========================================================

    @staticmethod
    def export_html(
        report: GeneratedReport,
    ) -> str:

        def esc(value: Any) -> str:
            return html.escape(
                str(value),
                quote=True,
            )

        sections_html = []

        for section in report.sections:
            metrics_html = "".join(
                (
                    "<tr>"
                    f"<th>{esc(key)}</th>"
                    f"<td>{esc(value)}</td>"
                    "</tr>"
                )
                for key, value in section.metrics.items()
            )

            findings_html = "".join(
                (
                    '<div class="finding">'
                    f"<strong>{esc(finding.severity.upper())}"
                    f" - {esc(finding.title)}</strong>"
                    f"<p>{esc(finding.description)}</p>"
                    "</div>"
                )
                for finding in section.findings
            )

            sections_html.append(
                "<section>"
                f"<h2>{esc(section.title)}</h2>"
                f"<p>{esc(section.description)}</p>"
                f"<table>{metrics_html}</table>"
                f"{findings_html}"
                "</section>"
            )

        recommendations_html = "".join(
            f"<li>{esc(item)}</li>"
            for item in report.recommendations
        )

        summary_html = "".join(
            (
                "<tr>"
                f"<th>{esc(key)}</th>"
                f"<td>{esc(value)}</td>"
                "</tr>"
            )
            for key, value in report.summary.items()
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(report.title)}</title>
<style>
* {{
    box-sizing: border-box;
}}
body {{
    margin: 0;
    padding: 32px;
    background: #f4f7fb;
    color: #172033;
    font-family: Arial, Helvetica, sans-serif;
    line-height: 1.6;
}}
.container {{
    max-width: 1100px;
    margin: auto;
}}
header, section {{
    background: white;
    padding: 28px;
    margin-bottom: 22px;
    border: 1px solid #e3e8f0;
    border-radius: 12px;
}}
h1 {{
    margin-top: 0;
    font-size: 30px;
}}
h2 {{
    margin-top: 0;
    font-size: 22px;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 16px;
}}
th, td {{
    border-bottom: 1px solid #e3e8f0;
    padding: 12px;
    text-align: left;
    vertical-align: top;
    overflow-wrap: anywhere;
}}
th {{
    width: 40%;
    color: #526176;
}}
.finding {{
    padding: 14px;
    margin-top: 12px;
    border-left: 4px solid #e6a23c;
    background: #fffaf0;
}}
.meta {{
    color: #526176;
    font-size: 14px;
}}
@media print {{
    body {{
        background: white;
        padding: 0;
    }}
    header, section {{
        border: none;
        break-inside: avoid;
    }}
}}
</style>
</head>
<body>
<div class="container">
<header>
<h1>{esc(report.title)}</h1>
<p>{esc(report.description)}</p>
<p class="meta">
Report ID: {esc(report.report_id)}
</p>
<p class="meta">
Generated: {esc(report.generated_at.isoformat())}
</p>
</header>
<section>
<h2>Executive Summary</h2>
<table>{summary_html}</table>
</section>
{''.join(sections_html)}
<section>
<h2>Recommendations</h2>
<ul>{recommendations_html}</ul>
</section>
</div>
</body>
</html>"""

    # ========================================================
    # GENERIC EXPORT
    # ========================================================

    def export_report(
        self,
        report: GeneratedReport,
        export_format: str | ReportFormat,
    ) -> ReportExport:

        if not isinstance(report, GeneratedReport):
            raise ReportValidationError(
                "Expected GeneratedReport"
            )

        selected_format = parse_enum(
            ReportFormat,
            export_format,
            "export_format",
        )

        exporters = {
            ReportFormat.JSON: (
                self.export_json,
                "application/json",
                "json",
            ),
            ReportFormat.CSV: (
                self.export_csv,
                "text/csv",
                "csv",
            ),
            ReportFormat.HTML: (
                self.export_html,
                "text/html",
                "html",
            ),
            ReportFormat.MARKDOWN: (
                self.export_markdown,
                "text/markdown",
                "md",
            ),
        }

        exporter, content_type, extension = exporters[
            selected_format
        ]

        try:
            content = exporter(report)
        except ReportServiceError:
            raise
        except Exception as exc:
            raise ReportExportError(
                "Unable to export report"
            ) from exc

        return ReportExport(
            report_id=report.report_id,
            format=selected_format.value,
            filename=(
                f"{report.report_type}_"
                f"{report.report_id}.{extension}"
            ),
            content_type=content_type,
            content=content,
            generated_at=utc_now(),
        )

    # ========================================================
    # AUDIT
    # ========================================================

    async def _audit(
        self,
        *,
        report_id: str,
        actor_id: str,
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
                    "entity_type": "report",
                    "entity_id": report_id,
                    "actor_id": actor_id,
                    "action": action,
                    "details": dict(details or {}),
                    "timestamp": utc_now().isoformat(),
                },
            )

        except Exception:
            logger.exception(
                "Report audit failed: %s",
                report_id,
            )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    async def save_report(
        self,
        report: GeneratedReport,
    ) -> Any:

        if self.repository is None:
            raise ReportRepositoryError(
                "Report repository is not configured"
            )

        if not isinstance(report, GeneratedReport):
            raise ReportValidationError(
                "Expected GeneratedReport"
            )

        saved = await invoke(
            self.repository.create_report,
            report.to_dict(),
        )

        await self._audit(
            report_id=report.report_id,
            actor_id=report.created_by,
            action="report_generated",
            details={
                "report_type": report.report_type,
                "company_id": report.company_id,
                "site_id": report.site_id,
            },
        )

        return saved

    # ========================================================
    # GET REPORT
    # ========================================================

    async def get_report(
        self,
        report_id: str,
    ) -> Any:

        if self.repository is None:
            raise ReportRepositoryError(
                "Report repository is not configured"
            )

        identifier = required_text(
            report_id,
            "report_id",
        )

        record = await invoke(
            self.repository.get_report,
            identifier,
        )

        if record is None:
            raise ReportNotFoundError(
                f"Report not found: {identifier}"
            )

        return record

    # ========================================================
    # LIST REPORTS
    # ========================================================

    async def list_reports(
        self,
        *,
        company_id: Optional[str] = None,
        site_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        report_type: Optional[str] = None,
    ) -> Any:

        if self.repository is None:
            raise ReportRepositoryError(
                "Report repository is not configured"
            )

        filters: dict[str, Any] = {}

        for name, value in {
            "company_id": company_id,
            "site_id": site_id,
            "asset_id": asset_id,
        }.items():
            if value is not None:
                filters[name] = required_text(
                    value,
                    name,
                )

        if report_type is not None:
            filters["report_type"] = parse_enum(
                ReportType,
                report_type,
                "report_type",
            ).value

        return await invoke(
            self.repository.list_reports,
            filters,
        )


# ============================================================
# IN-MEMORY REPOSITORY
# ============================================================

class InMemoryReportRepository:
    """
    Development-only report repository.

    Not durable or concurrency-safe.
    """

    def __init__(self):
        self.records: dict[
            str,
            dict[str, Any],
        ] = {}

    async def create_report(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:

        record = copy.deepcopy(data)
        report_id = record["report_id"]

        if report_id in self.records:
            raise ReportRepositoryError(
                "Duplicate report ID"
            )

        self.records[report_id] = record

        return copy.deepcopy(record)

    async def get_report(
        self,
        report_id: str,
    ) -> Optional[dict[str, Any]]:

        record = self.records.get(report_id)

        return (
            copy.deepcopy(record)
            if record is not None
            else None
        )

    async def list_reports(
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

    repository = InMemoryReportRepository()

    service = ReportService(
        report_repository=repository,
        default_company_id="COMPANY-001",
    )

    # --------------------------------------------------------
    # ASSET HEALTH REPORT
    # --------------------------------------------------------

    health_report = (
        service.generate_asset_health_report(
            company_id="COMPANY-001",
            site_id="SITE-001",
            asset_records=[
                {
                    "asset_id": "MACHINE-001",
                    "name": "Industrial Motor",
                    "health_score": 82,
                    "status": "operational",
                },
                {
                    "asset_id": "MACHINE-002",
                    "name": "Hydraulic Pump",
                    "health_score": 35,
                    "status": "degraded",
                },
                {
                    "asset_id": "MACHINE-003",
                    "name": "Air Compressor",
                    "health_score": 71,
                    "status": "operational",
                },
            ],
            created_by="ENGINEER-001",
        )
    )

    await service.save_report(
        health_report
    )

    print("\nASSET HEALTH REPORT")

    print(
        json.dumps(
            health_report.to_dict(),
            indent=2,
        )
    )

    # --------------------------------------------------------
    # PREDICTIVE MAINTENANCE REPORT
    # --------------------------------------------------------

    predictive_report = (
        service.generate_predictive_maintenance_report(
            company_id="COMPANY-001",
            site_id="SITE-001",
            prediction_records=[
                {
                    "asset_id": "MACHINE-001",
                    "risk_level": "low",
                    "failure_probability": 0.12,
                },
                {
                    "asset_id": "MACHINE-002",
                    "risk_level": "critical",
                    "failure_probability": 0.88,
                },
            ],
        )
    )

    await service.save_report(
        predictive_report
    )

    # --------------------------------------------------------
    # INCIDENT REPORT
    # --------------------------------------------------------

    incident_report = (
        service.generate_incident_report(
            company_id="COMPANY-001",
            site_id="SITE-001",
            incident_records=[
                {
                    "incident_id": "INC-001",
                    "asset_id": "MACHINE-002",
                    "severity": "high",
                    "status": "investigating",
                    "description": (
                        "Hydraulic pump overheating"
                    ),
                },
                {
                    "incident_id": "INC-002",
                    "asset_id": "MACHINE-003",
                    "severity": "medium",
                    "status": "resolved",
                    "description": (
                        "Abnormal compressor vibration"
                    ),
                },
            ],
        )
    )

    await service.save_report(
        incident_report
    )

    # --------------------------------------------------------
    # EXECUTIVE REPORT
    # --------------------------------------------------------

    executive_report = (
        service.generate_executive_report(
            company_id="COMPANY-001",
            site_id="SITE-001",
            asset_health_records=[
                {
                    "asset_id": "MACHINE-001",
                    "health_score": 82,
                },
                {
                    "asset_id": "MACHINE-002",
                    "health_score": 35,
                },
            ],
            prediction_records=[
                {
                    "asset_id": "MACHINE-002",
                    "risk_level": "critical",
                },
            ],
            incident_records=[
                {
                    "incident_id": "INC-001",
                    "severity": "critical",
                },
            ],
            maintenance_records=[
                {
                    "maintenance_id": "MAINT-001",
                    "status": "completed",
                },
                {
                    "maintenance_id": "MAINT-002",
                    "status": "scheduled",
                },
            ],
            financial_records=[
                {
                    "category": "repair",
                    "amount": 15000,
                    "currency": "INR",
                },
                {
                    "category": "maintenance",
                    "amount": 5000,
                    "currency": "INR",
                },
            ],
        )
    )

    await service.save_report(
        executive_report
    )

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    json_export = service.export_report(
        executive_report,
        ReportFormat.JSON,
    )

    csv_export = service.export_report(
        executive_report,
        ReportFormat.CSV,
    )

    html_export = service.export_report(
        executive_report,
        ReportFormat.HTML,
    )

    markdown_export = service.export_report(
        executive_report,
        ReportFormat.MARKDOWN,
    )

    print("\nEXPORT FILES")

    for export in [
        json_export,
        csv_export,
        html_export,
        markdown_export,
    ]:
        print(
            export.filename,
            export.content_type,
            len(export.content),
        )

    # --------------------------------------------------------
    # REPORT LIST
    # --------------------------------------------------------

    reports = await service.list_reports(
        company_id="COMPANY-001"
    )

    print("\nREPORT SUMMARY")

    print(
        json.dumps(
            {
                "total_reports": len(reports),
                "report_types": [
                    report["report_type"]
                    for report in reports
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
