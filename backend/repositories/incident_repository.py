
"""
Industrial Incident Repository
==============================

File: app/repositories/incident_repository.py

Features:
- Create industrial incidents
- Bulk incident ingestion
- Multi-tenant company/site isolation
- Incident retrieval, search and pagination
- Incident status and severity management
- Incident investigation tracking
- Root-cause analysis
- Corrective action tracking
- Incident escalation
- Human review and incident closure
- Incident history and statistics
- Soft deletion

Expected SQLAlchemy model:
    app.models.incident.Incident

Required columns:
    id
    company_id
    site_id
    incident_type
    severity
    status
    occurred_at

Recommended optional columns:
    asset_id
    machine_id
    incident_number
    title
    description
    priority
    risk_level
    source
    location
    reported_by
    reported_at
    investigation_status
    investigated_by
    investigated_at
    root_cause
    corrective_actions
    escalated_at
    escalated_by
    escalation_reason
    reviewed_by
    reviewed_at
    review_notes
    resolved_by
    resolved_at
    closed_by
    closed_at
    closure_notes
    metadata_json
    created_at
    updated_at
    deleted_at

Transactions are controlled by the service layer.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping, Sequence
from uuid import UUID

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.inspection import inspect

from app.models.incident import Incident


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class IncidentRepositoryError(Exception):
    """Base incident repository exception."""


class IncidentNotFoundError(IncidentRepositoryError):
    """Incident was not found in the tenant scope."""


class IncidentValidationError(IncidentRepositoryError):
    """Invalid incident data or query."""


class IncidentConflictError(IncidentRepositoryError):
    """Incident identity conflict."""


class IncidentRepository:
    """
    Async persistence layer for industrial incidents.

    Every read/write requires company_id.

    The repository does not commit or roll back transactions.
    """

    CREATE_FIELDS = frozenset({
        "id",
        "site_id",
        "asset_id",
        "machine_id",
        "incident_number",
        "incident_type",
        "severity",
        "status",
        "title",
        "description",
        "priority",
        "risk_level",
        "source",
        "location",
        "reported_by",
        "reported_at",
        "occurred_at",
        "metadata_json",
    })

    UPDATE_FIELDS = frozenset({
        "title",
        "description",
        "severity",
        "priority",
        "risk_level",
        "location",
        "metadata_json",
    })

    SORT_FIELDS = frozenset({
        "occurred_at",
        "reported_at",
        "created_at",
        "updated_at",
        "severity",
        "priority",
        "status",
    })

    ACTIVE_STATUSES = frozenset({
        "reported",
        "open",
        "acknowledged",
        "investigating",
        "in_progress",
        "escalated",
        "resolved",
    })

    CLOSED_STATUSES = frozenset({
        "closed",
        "cancelled",
    })

    CRITICAL_SEVERITIES = frozenset({
        "critical",
        "catastrophic",
    })

    def __init__(self, session: AsyncSession):
        self.session = session

        self.columns = {
            column.key
            for column in inspect(Incident).mapper.columns
        }

        required = {
            "id",
            "company_id",
            "site_id",
            "incident_type",
            "severity",
            "status",
            "occurred_at",
        }

        missing = required - self.columns

        if missing:
            raise IncidentRepositoryError(
                "Incident model missing columns: "
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
            raise IncidentValidationError(
                f"{name} must be a string"
            )

        value = value.strip()

        if not value:
            raise IncidentValidationError(
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

        if not isinstance(value, datetime):
            raise IncidentValidationError(
                f"{name} must be a datetime"
            )

        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise IncidentValidationError(
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
            raise IncidentValidationError(
                "start_time cannot exceed end_time"
            )

    @staticmethod
    def _validate_limit(
        limit: int,
        maximum: int = 1000,
    ) -> None:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= maximum
        ):
            raise IncidentValidationError(
                f"limit must be between 1 and {maximum}"
            )

    def _column(self, name: str):
        if name not in self.columns:
            raise IncidentValidationError(
                f"Incident model has no column '{name}'"
            )

        return getattr(Incident, name)

    def _validate_payload(
        self,
        data: Mapping[str, Any],
        *,
        allowed: frozenset[str],
    ) -> dict[str, Any]:
        if not isinstance(data, Mapping):
            raise IncidentValidationError(
                "Incident data must be a mapping"
            )

        payload = {
            key: self._normalize(value)
            for key, value in data.items()
        }

        unsupported = set(payload) - allowed

        if unsupported:
            raise IncidentValidationError(
                f"Unsupported fields: {sorted(unsupported)}"
            )

        missing_columns = set(payload) - self.columns

        if missing_columns:
            raise IncidentValidationError(
                "Fields missing from Incident model: "
                f"{sorted(missing_columns)}"
            )

        for name in (
            "occurred_at",
            "reported_at",
        ):
            if name in payload:
                self._validate_datetime(
                    payload[name],
                    name,
                )

        return payload

    def _scope(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        asset_id: str | None = None,
        machine_id: str | None = None,
        include_deleted: bool = False,
    ):
        company_id = self._require_text(
            company_id,
            "company_id",
        )

        conditions = [
            self._column("company_id") == company_id
        ]

        filters = {
            "site_id": site_id,
            "asset_id": asset_id,
            "machine_id": machine_id,
        }

        for name, value in filters.items():
            if value is not None:
                conditions.append(
                    self._column(name) == value
                )

        if (
            not include_deleted
            and "deleted_at" in self.columns
        ):
            conditions.append(
                self._column("deleted_at").is_(None)
            )

        return and_(*conditions)

    def _touch(self, incident: Incident) -> None:
        if "updated_at" in self.columns:
            incident.updated_at = utc_now()

    # ========================================================
    # CREATE INCIDENT
    # ========================================================

    async def create(
        self,
        *,
        company_id: str,
        data: Mapping[str, Any],
    ) -> Incident:
        company_id = self._require_text(
            company_id,
            "company_id",
        )

        payload = self._validate_payload(
            data,
            allowed=self.CREATE_FIELDS,
        )

        for name in (
            "site_id",
            "incident_type",
            "severity",
        ):
            payload[name] = self._require_text(
                payload.get(name),
                name,
            )

        payload["company_id"] = company_id

        payload.setdefault(
            "status",
            "reported",
        )

        payload.setdefault(
            "occurred_at",
            utc_now(),
        )

        if "reported_at" in self.columns:
            payload.setdefault(
                "reported_at",
                utc_now(),
            )

        if "created_at" in self.columns:
            payload.setdefault(
                "created_at",
                utc_now(),
            )

        if "updated_at" in self.columns:
            payload.setdefault(
                "updated_at",
                utc_now(),
            )

        incident = Incident(**payload)

        self.session.add(incident)
        await self.session.flush()

        return incident

    # ========================================================
    # BULK CREATE
    # ========================================================

    async def create_many(
        self,
        *,
        company_id: str,
        records: Sequence[Mapping[str, Any]],
    ) -> list[Incident]:
        if len(records) > 1000:
            raise IncidentValidationError(
                "Maximum 1000 incidents per batch"
            )

        created: list[Incident] = []

        for record in records:
            created.append(
                await self.create(
                    company_id=company_id,
                    data=record,
                )
            )

        return created

    # ========================================================
    # GET BY ID
    # ========================================================

    async def get_by_id(
        self,
        incident_id: UUID | str,
        *,
        company_id: str,
        include_deleted: bool = False,
    ) -> Incident | None:
        query = (
            select(Incident)
            .where(
                self._column("id") == incident_id,
                self._scope(
                    company_id=company_id,
                    include_deleted=include_deleted,
                ),
            )
        )

        result = await self.session.execute(query)

        return result.scalar_one_or_none()

    async def get_required(
        self,
        incident_id: UUID | str,
        *,
        company_id: str,
    ) -> Incident:
        incident = await self.get_by_id(
            incident_id,
            company_id=company_id,
        )

        if incident is None:
            raise IncidentNotFoundError(
                f"Incident '{incident_id}' not found"
            )

        return incident

    # ========================================================
    # GET BY INCIDENT NUMBER
    # ========================================================

    async def get_by_number(
        self,
        incident_number: str,
        *,
        company_id: str,
    ) -> Incident | None:
        incident_number = self._require_text(
            incident_number,
            "incident_number",
        )

        query = (
            select(Incident)
            .where(
                self._scope(
                    company_id=company_id,
                ),
                self._column("incident_number")
                == incident_number,
            )
        )

        result = await self.session.execute(query)

        return result.scalar_one_or_none()

    # ========================================================
    # UPDATE INCIDENT
    # ========================================================

    async def update(
        self,
        incident_id: UUID | str,
        *,
        company_id: str,
        data: Mapping[str, Any],
    ) -> Incident:
        incident = await self.get_required(
            incident_id,
            company_id=company_id,
        )

        payload = self._validate_payload(
            data,
            allowed=self.UPDATE_FIELDS,
        )

        for key, value in payload.items():
            setattr(incident, key, value)

        self._touch(incident)

        await self.session.flush()

        return incident

    # ========================================================
    # UPDATE STATUS
    # ========================================================

    async def update_status(
        self,
        incident_id: UUID | str,
        *,
        company_id: str,
        status: str,
    ) -> Incident:
        incident = await self.get_required(
            incident_id,
            company_id=company_id,
        )

        incident.status = self._require_text(
            status,
            "status",
        )

        self._touch(incident)
        await self.session.flush()

        return incident

    # ========================================================
    # START INVESTIGATION
    # ========================================================

    async def start_investigation(
        self,
        incident_id: UUID | str,
        *,
        company_id: str,
        investigated_by: str,
    ) -> Incident:
        investigated_by = self._require_text(
            investigated_by,
            "investigated_by",
        )

        incident = await self.get_required(
            incident_id,
            company_id=company_id,
        )

        incident.status = "investigating"

        if "investigation_status" in self.columns:
            incident.investigation_status = (
                "in_progress"
            )

        if "investigated_by" in self.columns:
            incident.investigated_by = investigated_by

        if "investigated_at" in self.columns:
            incident.investigated_at = utc_now()

        self._touch(incident)
        await self.session.flush()

        return incident

    # ========================================================
    # ROOT CAUSE ANALYSIS
    # ========================================================

    async def record_root_cause(
        self,
        incident_id: UUID | str,
        *,
        company_id: str,
        root_cause: str,
        investigated_by: str | None = None,
    ) -> Incident:
        root_cause = self._require_text(
            root_cause,
            "root_cause",
        )

        incident = await self.get_required(
            incident_id,
            company_id=company_id,
        )

        self._column("root_cause")
        incident.root_cause = root_cause

        if (
            investigated_by is not None
            and "investigated_by" in self.columns
        ):
            incident.investigated_by = (
                investigated_by
            )

        if "investigation_status" in self.columns:
            incident.investigation_status = (
                "completed"
            )

        self._touch(incident)
        await self.session.flush()

        return incident

    # ========================================================
    # CORRECTIVE ACTIONS
    # ========================================================

    async def update_corrective_actions(
        self,
        incident_id: UUID | str,
        *,
        company_id: str,
        corrective_actions: list[dict[str, Any]],
    ) -> Incident:
        incident = await self.get_required(
            incident_id,
            company_id=company_id,
        )

        self._column("corrective_actions")

        if not isinstance(
            corrective_actions,
            list,
        ):
            raise IncidentValidationError(
                "corrective_actions must be a list"
            )

        incident.corrective_actions = (
            corrective_actions
        )

        self._touch(incident)
        await self.session.flush()

        return incident

    # ========================================================
    # ESCALATE INCIDENT
    # ========================================================

    async def escalate(
        self,
        incident_id: UUID | str,
        *,
        company_id: str,
        escalated_by: str,
        reason: str,
    ) -> Incident:
        escalated_by = self._require_text(
            escalated_by,
            "escalated_by",
        )

        reason = self._require_text(
            reason,
            "reason",
        )

        incident = await self.get_required(
            incident_id,
            company_id=company_id,
        )

        incident.status = "escalated"

        if "escalated_by" in self.columns:
            incident.escalated_by = escalated_by

        if "escalated_at" in self.columns:
            incident.escalated_at = utc_now()

        if "escalation_reason" in self.columns:
            incident.escalation_reason = reason

        self._touch(incident)
        await self.session.flush()

        return incident

    # ========================================================
    # REVIEW INCIDENT
    # ========================================================

    async def review(
        self,
        incident_id: UUID | str,
        *,
        company_id: str,
        reviewed_by: str,
        notes: str | None = None,
    ) -> Incident:
        reviewed_by = self._require_text(
            reviewed_by,
            "reviewed_by",
        )

        incident = await self.get_required(
            incident_id,
            company_id=company_id,
        )

        if "reviewed_by" in self.columns:
            incident.reviewed_by = reviewed_by

        if "reviewed_at" in self.columns:
            incident.reviewed_at = utc_now()

        if (
            notes is not None
            and "review_notes" in self.columns
        ):
            incident.review_notes = notes

        self._touch(incident)
        await self.session.flush()

        return incident

    # ========================================================
    # RESOLVE INCIDENT
    # ========================================================

    async def resolve(
        self,
        incident_id: UUID | str,
        *,
        company_id: str,
        resolved_by: str,
    ) -> Incident:
        resolved_by = self._require_text(
            resolved_by,
            "resolved_by",
        )

        incident = await self.get_required(
            incident_id,
            company_id=company_id,
        )

        incident.status = "resolved"

        if "resolved_by" in self.columns:
            incident.resolved_by = resolved_by

        if "resolved_at" in self.columns:
            incident.resolved_at = utc_now()

        self._touch(incident)
        await self.session.flush()

        return incident

    # ========================================================
    # CLOSE INCIDENT
    # ========================================================

    async def close(
        self,
        incident_id: UUID | str,
        *,
        company_id: str,
        closed_by: str,
        closure_notes: str,
    ) -> Incident:
        closed_by = self._require_text(
            closed_by,
            "closed_by",
        )

        closure_notes = self._require_text(
            closure_notes,
            "closure_notes",
        )

        incident = await self.get_required(
            incident_id,
            company_id=company_id,
        )

        if str(self._normalize(incident.status)) != "resolved":
            raise IncidentValidationError(
                "Incident must be resolved before closure"
            )

        incident.status = "closed"

        if "closed_by" in self.columns:
            incident.closed_by = closed_by

        if "closed_at" in self.columns:
            incident.closed_at = utc_now()

        if "closure_notes" in self.columns:
            incident.closure_notes = closure_notes

        self._touch(incident)
        await self.session.flush()

        return incident

    # ========================================================
    # SOFT DELETE
    # ========================================================

    async def soft_delete(
        self,
        incident_id: UUID | str,
        *,
        company_id: str,
    ) -> bool:
        if "deleted_at" not in self.columns:
            raise IncidentRepositoryError(
                "Soft deletion requires deleted_at"
            )

        incident = await self.get_by_id(
            incident_id,
            company_id=company_id,
        )

        if incident is None:
            return False

        incident.deleted_at = utc_now()

        self._touch(incident)
        await self.session.flush()

        return True

    # ========================================================
    # LIST / SEARCH
    # ========================================================

    async def list(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        asset_id: str | None = None,
        machine_id: str | None = None,
        incident_type: str | None = None,
        severity: str | None = None,
        priority: str | None = None,
        risk_level: str | None = None,
        status: str | None = None,
        search: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "occurred_at",
        descending: bool = True,
    ) -> tuple[list[Incident], int]:
        self._validate_period(
            start_time,
            end_time,
        )

        if page < 1:
            raise IncidentValidationError(
                "page must be >= 1"
            )

        self._validate_limit(
            page_size,
            maximum=100,
        )

        if sort_by not in self.SORT_FIELDS:
            raise IncidentValidationError(
                f"Invalid sort field: {sort_by}"
            )

        conditions = [
            self._scope(
                company_id=company_id,
                site_id=site_id,
                asset_id=asset_id,
                machine_id=machine_id,
            )
        ]

        filters = {
            "incident_type": incident_type,
            "severity": severity,
            "priority": priority,
            "risk_level": risk_level,
            "status": status,
        }

        for name, value in filters.items():
            if value is not None:
                conditions.append(
                    self._column(name) == value
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

        if search:
            pattern = f"%{search.strip()}%"

            search_fields = [
                name
                for name in (
                    "title",
                    "description",
                    "incident_number",
                )
                if name in self.columns
            ]

            if search_fields:
                conditions.append(
                    or_(*[
                        self._column(name).ilike(pattern)
                        for name in search_fields
                    ])
                )

        predicate = and_(*conditions)

        total = (
            await self.session.execute(
                select(func.count())
                .select_from(Incident)
                .where(predicate)
            )
        ).scalar_one()

        order_column = self._column(sort_by)

        ordering = (
            order_column.desc()
            if descending
            else order_column.asc()
        )

        query = (
            select(Incident)
            .where(predicate)
            .order_by(
                ordering,
                self._column("id").asc(),
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
    # ACTIVE INCIDENTS
    # ========================================================

    async def get_active(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        limit: int = 100,
    ) -> list[Incident]:
        self._validate_limit(limit)

        query = (
            select(Incident)
            .where(
                self._scope(
                    company_id=company_id,
                    site_id=site_id,
                ),
                self._column("status").in_(
                    self.ACTIVE_STATUSES
                ),
            )
            .order_by(
                self._column("occurred_at").desc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # ========================================================
    # CRITICAL INCIDENTS
    # ========================================================

    async def get_critical(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        limit: int = 100,
    ) -> list[Incident]:
        self._validate_limit(limit)

        query = (
            select(Incident)
            .where(
                self._scope(
                    company_id=company_id,
                    site_id=site_id,
                ),
                self._column("severity").in_(
                    self.CRITICAL_SEVERITIES
                ),
                self._column("status").notin_(
                    self.CLOSED_STATUSES
                ),
            )
            .order_by(
                self._column("occurred_at").desc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # ========================================================
    # ASSET INCIDENT HISTORY
    # ========================================================

    async def get_asset_history(
        self,
        *,
        company_id: str,
        asset_id: str,
        limit: int = 500,
    ) -> list[Incident]:
        self._validate_limit(limit)

        query = (
            select(Incident)
            .where(
                self._scope(
                    company_id=company_id,
                    asset_id=asset_id,
                )
            )
            .order_by(
                self._column("occurred_at").desc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # ========================================================
    # INCIDENT STATISTICS
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

        total = (
            await self.session.execute(
                select(func.count())
                .select_from(Incident)
                .where(predicate)
            )
        ).scalar_one()

        status_counts: dict[str, int] = {}
        severity_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}

        for field, destination in (
            ("status", status_counts),
            ("severity", severity_counts),
            ("incident_type", type_counts),
        ):
            column = self._column(field)

            result = await self.session.execute(
                select(column, func.count())
                .where(predicate)
                .group_by(column)
            )

            for value, count in result.all():
                destination[
                    str(self._normalize(value))
                ] = int(count)

        return {
            "company_id": company_id,
            "site_id": site_id,
            "total_incidents": int(total),
            "active_incidents": sum(
                count
                for status, count in status_counts.items()
                if status in self.ACTIVE_STATUSES
            ),
            "critical_incidents": sum(
                count
                for severity, count in severity_counts.items()
                if severity in self.CRITICAL_SEVERITIES
            ),
            "closed_incidents": (
                status_counts.get("closed", 0)
            ),
            "status_counts": status_counts,
            "severity_counts": severity_counts,
            "incident_type_counts": type_counts,
            "generated_at": utc_now(),
        }


__all__ = [
    "IncidentRepository",
    "IncidentRepositoryError",
    "IncidentNotFoundError",
    "IncidentValidationError",
    "IncidentConflictError",
]
