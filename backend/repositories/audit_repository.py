
"""
Industrial AI Audit Repository
==============================

File: app/repositories/audit_repository.py

Python: 3.11+
SQLAlchemy: 2.x

Features:
- Append-only audit events
- Multi-tenant isolation
- User and system activity logging
- AI agent and model execution audits
- Safety and emergency action auditing
- Maintenance and asset change history
- Approval and review audit records
- Authentication and access audits
- Before/after change tracking
- Correlation and request tracing
- Time-range and entity-based filtering
- Paginated audit history
- Compliance statistics
- Audit export queries

Dependencies:
    pip install "sqlalchemy[asyncio]>=2.0"

Expected model:
    app.models.audit_log.AuditLog

Required model columns:
    id
    company_id
    event_type
    action
    actor_type
    occurred_at

Recommended additional columns:
    site_id
    actor_id
    actor_name
    entity_type
    entity_id
    resource_type
    resource_id
    outcome
    severity
    description
    request_id
    correlation_id
    trace_id
    session_id
    ip_address
    user_agent
    before_data
    after_data
    metadata_json
    created_at

Important:
- No update or delete methods are provided.
- Repository methods flush but do not commit.
- Sensitive data must be redacted before logging.
- Database permissions should enforce append-only access.
- For tamper evidence, use an external immutable log
  destination or cryptographically verified audit chain.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping, Sequence
from uuid import UUID

from sqlalchemy import (
    and_,
    func,
    or_,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.inspection import inspect

from app.models.audit_log import AuditLog


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================
# ENUMS
# ============================================================

class AuditActorType(str, Enum):
    USER = "user"
    ADMIN = "admin"
    SYSTEM = "system"
    SERVICE = "service"
    AI_AGENT = "ai_agent"
    ML_MODEL = "ml_model"
    DEVICE = "device"
    EXTERNAL_SYSTEM = "external_system"


class AuditEventType(str, Enum):
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    USER_MANAGEMENT = "user_management"
    ASSET = "asset"
    MACHINE = "machine"
    TELEMETRY = "telemetry"
    ANOMALY = "anomaly"
    PREDICTION = "prediction"
    MAINTENANCE = "maintenance"
    INSPECTION = "inspection"
    SAFETY = "safety"
    INCIDENT = "incident"
    EMERGENCY = "emergency"
    COST = "cost"
    REPORT = "report"
    AI_AGENT = "ai_agent"
    MODEL = "model"
    CONFIGURATION = "configuration"
    SECURITY = "security"
    DATA_ACCESS = "data_access"
    COMPLIANCE = "compliance"
    OTHER = "other"


class AuditOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    PARTIAL = "partial"
    PENDING = "pending"


class AuditSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# EXCEPTIONS
# ============================================================

class AuditRepositoryError(Exception):
    """Base audit repository exception."""


class AuditValidationError(AuditRepositoryError):
    """Invalid audit repository input."""


class AuditNotFoundError(AuditRepositoryError):
    """Audit record was not found in tenant scope."""


# ============================================================
# AUDIT REPOSITORY
# ============================================================

class AuditRepository:
    """
    Append-only, tenant-scoped audit repository.

    The caller must provide the authenticated company_id.

    Important:
        Never trust company_id or actor_id supplied
        directly by an unverified client.

    Transaction management:
        The repository calls flush().
        The service layer calls commit()/rollback().
    """

    CREATE_FIELDS = frozenset({
        "id",
        "site_id",
        "event_type",
        "action",
        "actor_type",
        "actor_id",
        "actor_name",
        "entity_type",
        "entity_id",
        "resource_type",
        "resource_id",
        "outcome",
        "severity",
        "description",
        "request_id",
        "correlation_id",
        "trace_id",
        "session_id",
        "ip_address",
        "user_agent",
        "before_data",
        "after_data",
        "metadata_json",
        "occurred_at",
    })

    SORT_FIELDS = frozenset({
        "occurred_at",
        "created_at",
        "severity",
        "event_type",
    })

    REQUIRED_COLUMNS = frozenset({
        "id",
        "company_id",
        "event_type",
        "action",
        "actor_type",
        "occurred_at",
    })

    def __init__(self, session: AsyncSession):
        self.session = session

        self.columns = {
            column.key
            for column in inspect(AuditLog).mapper.columns
        }

        missing = self.REQUIRED_COLUMNS - self.columns

        if missing:
            raise AuditRepositoryError(
                "AuditLog model is missing columns: "
                f"{sorted(missing)}"
            )

    # ========================================================
    # INTERNAL HELPERS
    # ========================================================

    @staticmethod
    def _normalize(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value

        return value

    @staticmethod
    def _require_text(
        value: str,
        name: str,
    ) -> str:
        if not isinstance(value, str):
            raise AuditValidationError(
                f"{name} must be a string"
            )

        value = value.strip()

        if not value:
            raise AuditValidationError(
                f"{name} is required"
            )

        return value

    @staticmethod
    def _validate_datetime(
        value: datetime | None,
        name: str,
    ) -> None:
        if value is None:
            return

        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise AuditValidationError(
                f"{name} must be timezone-aware"
            )

    @classmethod
    def _validate_period(
        cls,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> None:
        cls._validate_datetime(
            start_time,
            "start_time",
        )

        cls._validate_datetime(
            end_time,
            "end_time",
        )

        if (
            start_time is not None
            and end_time is not None
            and start_time > end_time
        ):
            raise AuditValidationError(
                "start_time cannot exceed end_time"
            )

    @staticmethod
    def _validate_limit(
        limit: int,
        maximum: int = 1000,
    ) -> None:
        if not isinstance(limit, int) or isinstance(
            limit, bool
        ):
            raise AuditValidationError(
                "limit must be an integer"
            )

        if not 1 <= limit <= maximum:
            raise AuditValidationError(
                f"limit must be between 1 and {maximum}"
            )

    def _column(self, name: str):
        if name not in self.columns:
            raise AuditValidationError(
                f"AuditLog model has no column '{name}'"
            )

        return getattr(AuditLog, name)

    def _validate_payload(
        self,
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(data, Mapping):
            raise AuditValidationError(
                "Audit data must be a mapping"
            )

        payload = {
            key: self._normalize(value)
            for key, value in data.items()
        }

        unsupported = set(payload) - self.CREATE_FIELDS

        if unsupported:
            raise AuditValidationError(
                "Unsupported audit fields: "
                f"{sorted(unsupported)}"
            )

        missing_columns = set(payload) - self.columns

        if missing_columns:
            raise AuditValidationError(
                "Audit fields missing from model: "
                f"{sorted(missing_columns)}"
            )

        for name in (
            "event_type",
            "action",
            "actor_type",
        ):
            payload[name] = self._require_text(
                payload.get(name),
                name,
            )

        return payload

    def _scope(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
    ):
        company_id = self._require_text(
            company_id,
            "company_id",
        )

        conditions = [
            self._column("company_id") == company_id
        ]

        if site_id is not None:
            conditions.append(
                self._column("site_id") == site_id
            )

        return and_(*conditions)

    # ========================================================
    # CREATE AUDIT EVENT
    # ========================================================

    async def create(
        self,
        *,
        company_id: str,
        data: Mapping[str, Any],
    ) -> AuditLog:
        company_id = self._require_text(
            company_id,
            "company_id",
        )

        payload = self._validate_payload(data)

        payload["company_id"] = company_id

        payload.setdefault(
            "occurred_at",
            utc_now(),
        )

        self._validate_datetime(
            payload["occurred_at"],
            "occurred_at",
        )

        if "created_at" in self.columns:
            payload.setdefault(
                "created_at",
                utc_now(),
            )

        audit = AuditLog(**payload)

        self.session.add(audit)

        await self.session.flush()

        return audit

    # ========================================================
    # BULK AUDIT INSERT
    # ========================================================

    async def create_many(
        self,
        *,
        company_id: str,
        events: Sequence[Mapping[str, Any]],
    ) -> list[AuditLog]:
        if len(events) > 1000:
            raise AuditValidationError(
                "Maximum 1000 audit events per batch"
            )

        company_id = self._require_text(
            company_id,
            "company_id",
        )

        objects: list[AuditLog] = []

        for event in events:
            payload = self._validate_payload(event)

            payload["company_id"] = company_id

            payload.setdefault(
                "occurred_at",
                utc_now(),
            )

            self._validate_datetime(
                payload["occurred_at"],
                "occurred_at",
            )

            if "created_at" in self.columns:
                payload.setdefault(
                    "created_at",
                    utc_now(),
                )

            objects.append(AuditLog(**payload))

        self.session.add_all(objects)

        await self.session.flush()

        return objects

    # ========================================================
    # GET BY ID
    # ========================================================

    async def get_by_id(
        self,
        audit_id: UUID | str,
        *,
        company_id: str,
    ) -> AuditLog | None:
        query = (
            select(AuditLog)
            .where(
                self._column("id") == audit_id,
                self._scope(
                    company_id=company_id,
                ),
            )
        )

        result = await self.session.execute(query)

        return result.scalar_one_or_none()

    async def get_required(
        self,
        audit_id: UUID | str,
        *,
        company_id: str,
    ) -> AuditLog:
        audit = await self.get_by_id(
            audit_id,
            company_id=company_id,
        )

        if audit is None:
            raise AuditNotFoundError(
                f"Audit event '{audit_id}' not found"
            )

        return audit

    # ========================================================
    # GENERIC EVENT LOGGER
    # ========================================================

    async def log_event(
        self,
        *,
        company_id: str,
        event_type: AuditEventType | str,
        action: str,
        actor_type: AuditActorType | str,
        actor_id: str | None = None,
        site_id: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        outcome: AuditOutcome | str = AuditOutcome.SUCCESS,
        severity: AuditSeverity | str = AuditSeverity.INFO,
        description: str | None = None,
        request_id: str | None = None,
        correlation_id: str | None = None,
        before_data: dict[str, Any] | None = None,
        after_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditLog:
        payload: dict[str, Any] = {
            "event_type": event_type,
            "action": action,
            "actor_type": actor_type,
            "actor_id": actor_id,
            "site_id": site_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "outcome": outcome,
            "severity": severity,
            "description": description,
            "request_id": request_id,
            "correlation_id": correlation_id,
            "before_data": before_data,
            "after_data": after_data,
            "metadata_json": metadata or {},
        }

        payload = {
            key: value
            for key, value in payload.items()
            if key in self.columns and value is not None
        }

        return await self.create(
            company_id=company_id,
            data=payload,
        )

    # ========================================================
    # AI AGENT AUDIT
    # ========================================================

    async def log_agent_action(
        self,
        *,
        company_id: str,
        agent_id: str,
        action: str,
        entity_type: str | None = None,
        entity_id: str | None = None,
        outcome: AuditOutcome | str = AuditOutcome.SUCCESS,
        description: str | None = None,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditLog:
        return await self.log_event(
            company_id=company_id,
            event_type=AuditEventType.AI_AGENT,
            action=action,
            actor_type=AuditActorType.AI_AGENT,
            actor_id=agent_id,
            entity_type=entity_type,
            entity_id=entity_id,
            outcome=outcome,
            description=description,
            correlation_id=correlation_id,
            metadata=metadata,
        )

    # ========================================================
    # SAFETY AUDIT
    # ========================================================

    async def log_safety_action(
        self,
        *,
        company_id: str,
        action: str,
        actor_id: str,
        safety_event_id: str,
        outcome: AuditOutcome | str = AuditOutcome.SUCCESS,
        severity: AuditSeverity | str = AuditSeverity.HIGH,
        description: str | None = None,
        site_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditLog:
        return await self.log_event(
            company_id=company_id,
            site_id=site_id,
            event_type=AuditEventType.SAFETY,
            action=action,
            actor_type=AuditActorType.USER,
            actor_id=actor_id,
            entity_type="safety_event",
            entity_id=safety_event_id,
            outcome=outcome,
            severity=severity,
            description=description,
            metadata=metadata,
        )

    # ========================================================
    # EMERGENCY AUDIT
    # ========================================================

    async def log_emergency_action(
        self,
        *,
        company_id: str,
        action: str,
        actor_id: str,
        emergency_id: str,
        outcome: AuditOutcome | str = AuditOutcome.SUCCESS,
        description: str | None = None,
        site_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditLog:
        return await self.log_event(
            company_id=company_id,
            site_id=site_id,
            event_type=AuditEventType.EMERGENCY,
            action=action,
            actor_type=AuditActorType.USER,
            actor_id=actor_id,
            entity_type="emergency",
            entity_id=emergency_id,
            outcome=outcome,
            severity=AuditSeverity.CRITICAL,
            description=description,
            metadata=metadata,
        )

    # ========================================================
    # AUTHENTICATION AUDIT
    # ========================================================

    async def log_authentication(
        self,
        *,
        company_id: str,
        user_id: str,
        action: str,
        outcome: AuditOutcome | str,
        ip_address: str | None = None,
        user_agent: str | None = None,
        request_id: str | None = None,
        description: str | None = None,
    ) -> AuditLog:
        return await self.log_event(
            company_id=company_id,
            event_type=AuditEventType.AUTHENTICATION,
            action=action,
            actor_type=AuditActorType.USER,
            actor_id=user_id,
            outcome=outcome,
            description=description,
            request_id=request_id,
            metadata={
                "ip_address": ip_address,
                "user_agent": user_agent,
            },
        )

    # ========================================================
    # LIST AUDIT EVENTS
    # ========================================================

    async def list(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        event_type: str | None = None,
        action: str | None = None,
        actor_type: str | None = None,
        actor_id: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        outcome: str | None = None,
        severity: str | None = None,
        correlation_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[AuditLog], int]:
        self._validate_period(
            start_time,
            end_time,
        )

        if page < 1:
            raise AuditValidationError(
                "page must be at least 1"
            )

        self._validate_limit(
            page_size,
            maximum=100,
        )

        conditions = [
            self._scope(
                company_id=company_id,
                site_id=site_id,
            )
        ]

        filters = {
            "event_type": event_type,
            "action": action,
            "actor_type": actor_type,
            "actor_id": actor_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "outcome": outcome,
            "severity": severity,
            "correlation_id": correlation_id,
        }

        for field, value in filters.items():
            if value is not None:
                conditions.append(
                    self._column(field) == value
                )

        if start_time is not None:
            conditions.append(
                self._column("occurred_at")
                >= start_time
            )

        if end_time is not None:
            conditions.append(
                self._column("occurred_at")
                <= end_time
            )

        predicate = and_(*conditions)

        count_query = (
            select(func.count())
            .select_from(AuditLog)
            .where(predicate)
        )

        total = (
            await self.session.execute(count_query)
        ).scalar_one()

        query = (
            select(AuditLog)
            .where(predicate)
            .order_by(
                self._column("occurred_at").desc(),
                self._column("id").desc(),
            )
            .offset((page - 1) * page_size)
            .limit(page_size)
        )

        result = await self.session.execute(query)

        return (
            list(result.scalars().all()),
            int(total),
        )

    # ========================================================
    # ENTITY AUDIT HISTORY
    # ========================================================

    async def get_entity_history(
        self,
        *,
        company_id: str,
        entity_type: str,
        entity_id: str,
        limit: int = 500,
    ) -> list[AuditLog]:
        self._validate_limit(limit)

        query = (
            select(AuditLog)
            .where(
                self._scope(
                    company_id=company_id,
                ),
                self._column("entity_type")
                == entity_type,
                self._column("entity_id")
                == entity_id,
            )
            .order_by(
                self._column("occurred_at").desc(),
                self._column("id").desc(),
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # ========================================================
    # ACTOR ACTIVITY
    # ========================================================

    async def get_actor_activity(
        self,
        *,
        company_id: str,
        actor_id: str,
        hours: int = 24,
        limit: int = 200,
    ) -> list[AuditLog]:
        if hours <= 0:
            raise AuditValidationError(
                "hours must be positive"
            )

        self._validate_limit(limit)

        cutoff = utc_now() - timedelta(
            hours=hours
        )

        query = (
            select(AuditLog)
            .where(
                self._scope(
                    company_id=company_id,
                ),
                self._column("actor_id") == actor_id,
                self._column("occurred_at") >= cutoff,
            )
            .order_by(
                self._column("occurred_at").desc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # ========================================================
    # CORRELATION TRACE
    # ========================================================

    async def get_correlation_history(
        self,
        *,
        company_id: str,
        correlation_id: str,
        limit: int = 500,
    ) -> list[AuditLog]:
        self._validate_limit(limit)

        query = (
            select(AuditLog)
            .where(
                self._scope(
                    company_id=company_id,
                ),
                self._column("correlation_id")
                == correlation_id,
            )
            .order_by(
                self._column("occurred_at").asc(),
                self._column("id").asc(),
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # ========================================================
    # RECENT SECURITY EVENTS
    # ========================================================

    async def get_security_events(
        self,
        *,
        company_id: str,
        hours: int = 24,
        limit: int = 200,
    ) -> list[AuditLog]:
        if hours <= 0:
            raise AuditValidationError(
                "hours must be positive"
            )

        self._validate_limit(limit)

        cutoff = utc_now() - timedelta(
            hours=hours
        )

        query = (
            select(AuditLog)
            .where(
                self._scope(
                    company_id=company_id,
                ),
                self._column("event_type").in_([
                    "authentication",
                    "authorization",
                    "security",
                    "data_access",
                ]),
                self._column("occurred_at") >= cutoff,
            )
            .order_by(
                self._column("occurred_at").desc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # ========================================================
    # AUDIT STATISTICS
    # ========================================================

    async def get_statistics(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        self._validate_period(
            start_time,
            end_time,
        )

        conditions = [
            self._scope(
                company_id=company_id,
                site_id=site_id,
            )
        ]

        if start_time is not None:
            conditions.append(
                self._column("occurred_at")
                >= start_time
            )

        if end_time is not None:
            conditions.append(
                self._column("occurred_at")
                <= end_time
            )

        predicate = and_(*conditions)

        total_query = (
            select(func.count())
            .select_from(AuditLog)
            .where(predicate)
        )

        total = (
            await self.session.execute(total_query)
        ).scalar_one()

        event_counts: dict[str, int] = {}
        outcome_counts: dict[str, int] = {}
        severity_counts: dict[str, int] = {}

        for field, destination in (
            ("event_type", event_counts),
            ("outcome", outcome_counts),
            ("severity", severity_counts),
        ):
            if field not in self.columns:
                continue

            column = self._column(field)

            query = (
                select(
                    column,
                    func.count(),
                )
                .where(predicate)
                .group_by(column)
            )

            result = await self.session.execute(query)

            for value, count in result.all():
                key = str(self._normalize(value))
                destination[key] = int(count)

        return {
            "company_id": company_id,
            "site_id": site_id,
            "total_events": int(total),
            "failed_events": (
                outcome_counts.get("failure", 0)
            ),
            "denied_events": (
                outcome_counts.get("denied", 0)
            ),
            "critical_events": (
                severity_counts.get("critical", 0)
            ),
            "event_type_counts": event_counts,
            "outcome_counts": outcome_counts,
            "severity_counts": severity_counts,
            "generated_at": utc_now(),
        }


__all__ = [
    "AuditRepository",
    "AuditRepositoryError",
    "AuditValidationError",
    "AuditNotFoundError",
    "AuditActorType",
    "AuditEventType",
    "AuditOutcome",
    "AuditSeverity",
]
