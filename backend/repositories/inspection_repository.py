
"""
Industrial Inspection Repository
================================

File: app/repositories/inspection_repository.py

Python: 3.11+
SQLAlchemy: 2.x

Features:
- Create and retrieve asset inspections
- Bulk inspection ingestion
- Multi-tenant company/site isolation
- Computer-vision inspection metadata
- Defect and anomaly tracking
- Inspection status and severity management
- Human review and approval workflows
- Corrective actions and maintenance recommendations
- Asset inspection history
- Critical and failed inspection queries
- Inspection analytics and statistics
- Soft deletion

Expected model:
    app.models.inspection.Inspection

Required columns:
    id
    company_id
    site_id
    asset_id
    inspection_type
    status
    inspected_at

Recommended optional columns:
    machine_id
    inspection_number
    inspection_source
    inspection_method
    inspector_id
    image_url
    image_storage_key
    defect_type
    defect_count
    defect_severity
    defect_confidence
    condition_score
    risk_level
    result
    description
    findings
    recommendations
    corrective_actions
    model_name
    model_version
    reviewed_by
    reviewed_at
    review_notes
    approved_by
    approved_at
    resolved_by
    resolved_at
    metadata_json
    created_at
    updated_at
    deleted_at

Transactions are managed by the service layer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Any, Mapping, Sequence
from uuid import UUID

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.inspection import inspect

from app.models.inspection import Inspection


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class InspectionRepositoryError(Exception):
    """Base inspection repository exception."""


class InspectionNotFoundError(InspectionRepositoryError):
    """Inspection was not found."""


class InspectionValidationError(InspectionRepositoryError):
    """Invalid inspection data."""


class InspectionConflictError(InspectionRepositoryError):
    """Inspection identity conflict."""


class InspectionRepository:
    """
    Async repository for industrial asset inspections.

    Security:
        All operations are scoped by company_id.

    Transaction ownership:
        Repository -> flush()
        Service -> commit() / rollback()

    Computer vision inference belongs to the AI layer.
    """

    CREATE_FIELDS = frozenset({
        "id",
        "site_id",
        "asset_id",
        "machine_id",
        "inspection_number",
        "inspection_type",
        "inspection_source",
        "inspection_method",
        "inspector_id",
        "status",
        "image_url",
        "image_storage_key",
        "defect_type",
        "defect_count",
        "defect_severity",
        "defect_confidence",
        "condition_score",
        "risk_level",
        "result",
        "description",
        "findings",
        "recommendations",
        "corrective_actions",
        "model_name",
        "model_version",
        "inspected_at",
        "metadata_json",
    })

    UPDATE_FIELDS = frozenset({
        "inspection_method",
        "defect_type",
        "defect_count",
        "defect_severity",
        "defect_confidence",
        "condition_score",
        "risk_level",
        "result",
        "description",
        "findings",
        "recommendations",
        "corrective_actions",
        "metadata_json",
    })

    SORT_FIELDS = frozenset({
        "inspected_at",
        "created_at",
        "updated_at",
        "defect_count",
        "defect_confidence",
        "condition_score",
        "status",
    })

    CRITICAL_SEVERITIES = frozenset({
        "critical",
        "severe",
        "extreme",
    })

    FAILED_RESULTS = frozenset({
        "failed",
        "fail",
        "unsafe",
        "rejected",
    })

    ACTIVE_STATUSES = frozenset({
        "pending",
        "scheduled",
        "in_progress",
        "under_review",
    })

    def __init__(self, session: AsyncSession):
        self.session = session

        self.columns = {
            column.key
            for column in inspect(Inspection).mapper.columns
        }

        required = {
            "id",
            "company_id",
            "site_id",
            "asset_id",
            "inspection_type",
            "status",
            "inspected_at",
        }

        missing = required - self.columns

        if missing:
            raise InspectionRepositoryError(
                "Inspection model missing columns: "
                f"{sorted(missing)}"
            )

    # ========================================================
    # HELPERS
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
            raise InspectionValidationError(
                f"{name} must be a string"
            )

        value = value.strip()

        if not value:
            raise InspectionValidationError(
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
            raise InspectionValidationError(
                f"{name} must be datetime"
            )

        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise InspectionValidationError(
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
            raise InspectionValidationError(
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
            raise InspectionValidationError(
                f"limit must be between 1 and {maximum}"
            )

    @staticmethod
    def _validate_number(
        value: Any,
        name: str,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise InspectionValidationError(
                f"{name} must be numeric"
            )

        number = float(value)

        if not isfinite(number):
            raise InspectionValidationError(
                f"{name} must be finite"
            )

        if minimum is not None and number < minimum:
            raise InspectionValidationError(
                f"{name} must be >= {minimum}"
            )

        if maximum is not None and number > maximum:
            raise InspectionValidationError(
                f"{name} must be <= {maximum}"
            )

        return number

    def _column(self, name: str):
        if name not in self.columns:
            raise InspectionValidationError(
                f"Inspection model has no column '{name}'"
            )

        return getattr(Inspection, name)

    def _validate_payload(
        self,
        data: Mapping[str, Any],
        *,
        allowed: frozenset[str],
    ) -> dict[str, Any]:
        if not isinstance(data, Mapping):
            raise InspectionValidationError(
                "Inspection data must be a mapping"
            )

        payload = {
            key: self._normalize(value)
            for key, value in data.items()
        }

        unsupported = set(payload) - allowed

        if unsupported:
            raise InspectionValidationError(
                f"Unsupported fields: {sorted(unsupported)}"
            )

        missing_columns = set(payload) - self.columns

        if missing_columns:
            raise InspectionValidationError(
                "Fields missing from Inspection model: "
                f"{sorted(missing_columns)}"
            )

        if (
            "defect_count" in payload
            and payload["defect_count"] is not None
        ):
            count = payload["defect_count"]

            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise InspectionValidationError(
                    "defect_count must be >= 0"
                )

        if (
            "defect_confidence" in payload
            and payload["defect_confidence"] is not None
        ):
            payload["defect_confidence"] = (
                self._validate_number(
                    payload["defect_confidence"],
                    "defect_confidence",
                    minimum=0,
                    maximum=1,
                )
            )

        if (
            "condition_score" in payload
            and payload["condition_score"] is not None
        ):
            payload["condition_score"] = (
                self._validate_number(
                    payload["condition_score"],
                    "condition_score",
                    minimum=0,
                    maximum=100,
                )
            )

        if "inspected_at" in payload:
            self._validate_datetime(
                payload["inspected_at"],
                "inspected_at",
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

        for name, value in {
            "site_id": site_id,
            "asset_id": asset_id,
            "machine_id": machine_id,
        }.items():
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

    def _touch(self, inspection: Inspection) -> None:
        if "updated_at" in self.columns:
            inspection.updated_at = utc_now()

    # ========================================================
    # CREATE INSPECTION
    # ========================================================

    async def create(
        self,
        *,
        company_id: str,
        data: Mapping[str, Any],
    ) -> Inspection:
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
            "asset_id",
            "inspection_type",
        ):
            payload[name] = self._require_text(
                payload.get(name),
                name,
            )

        payload["company_id"] = company_id

        payload.setdefault(
            "status",
            "pending",
        )

        payload.setdefault(
            "inspected_at",
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

        inspection = Inspection(**payload)

        self.session.add(inspection)

        await self.session.flush()

        return inspection

    # ========================================================
    # BULK CREATE
    # ========================================================

    async def create_many(
        self,
        *,
        company_id: str,
        records: Sequence[Mapping[str, Any]],
    ) -> list[Inspection]:
        if len(records) > 1000:
            raise InspectionValidationError(
                "Maximum 1000 inspections per batch"
            )

        created: list[Inspection] = []

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
        inspection_id: UUID | str,
        *,
        company_id: str,
        include_deleted: bool = False,
    ) -> Inspection | None:
        query = (
            select(Inspection)
            .where(
                self._column("id") == inspection_id,
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
        inspection_id: UUID | str,
        *,
        company_id: str,
    ) -> Inspection:
        inspection = await self.get_by_id(
            inspection_id,
            company_id=company_id,
        )

        if inspection is None:
            raise InspectionNotFoundError(
                f"Inspection '{inspection_id}' not found"
            )

        return inspection

    # ========================================================
    # GET BY INSPECTION NUMBER
    # ========================================================

    async def get_by_number(
        self,
        inspection_number: str,
        *,
        company_id: str,
    ) -> Inspection | None:
        inspection_number = self._require_text(
            inspection_number,
            "inspection_number",
        )

        query = (
            select(Inspection)
            .where(
                self._scope(company_id=company_id),
                self._column("inspection_number")
                == inspection_number,
            )
        )

        result = await self.session.execute(query)

        return result.scalar_one_or_none()

    # ========================================================
    # UPDATE INSPECTION
    # ========================================================

    async def update(
        self,
        inspection_id: UUID | str,
        *,
        company_id: str,
        data: Mapping[str, Any],
    ) -> Inspection:
        inspection = await self.get_required(
            inspection_id,
            company_id=company_id,
        )

        payload = self._validate_payload(
            data,
            allowed=self.UPDATE_FIELDS,
        )

        for key, value in payload.items():
            setattr(inspection, key, value)

        self._touch(inspection)
        await self.session.flush()

        return inspection

    # ========================================================
    # UPDATE STATUS
    # ========================================================

    async def update_status(
        self,
        inspection_id: UUID | str,
        *,
        company_id: str,
        status: str,
    ) -> Inspection:
        inspection = await self.get_required(
            inspection_id,
            company_id=company_id,
        )

        inspection.status = self._require_text(
            status,
            "status",
        )

        self._touch(inspection)
        await self.session.flush()

        return inspection

    # ========================================================
    # RECORD AI INSPECTION RESULTS
    # ========================================================

    async def record_ai_result(
        self,
        inspection_id: UUID | str,
        *,
        company_id: str,
        defect_type: str | None = None,
        defect_count: int | None = None,
        defect_severity: str | None = None,
        defect_confidence: float | None = None,
        condition_score: float | None = None,
        risk_level: str | None = None,
        result: str | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> Inspection:
        inspection = await self.get_required(
            inspection_id,
            company_id=company_id,
        )

        data = {
            "defect_type": defect_type,
            "defect_count": defect_count,
            "defect_severity": defect_severity,
            "defect_confidence": defect_confidence,
            "condition_score": condition_score,
            "risk_level": risk_level,
            "result": result,
            "model_name": model_name,
            "model_version": model_version,
        }

        data = {
            key: value
            for key, value in data.items()
            if value is not None
        }

        payload = self._validate_payload(
            data,
            allowed=frozenset(data),
        )

        for key, value in payload.items():
            setattr(inspection, key, value)

        inspection.status = "under_review"

        self._touch(inspection)
        await self.session.flush()

        return inspection

    # ========================================================
    # CORRECTIVE ACTIONS
    # ========================================================

    async def update_corrective_actions(
        self,
        inspection_id: UUID | str,
        *,
        company_id: str,
        corrective_actions: list[dict[str, Any]],
    ) -> Inspection:
        inspection = await self.get_required(
            inspection_id,
            company_id=company_id,
        )

        self._column("corrective_actions")

        if not isinstance(corrective_actions, list):
            raise InspectionValidationError(
                "corrective_actions must be a list"
            )

        inspection.corrective_actions = (
            corrective_actions
        )

        self._touch(inspection)
        await self.session.flush()

        return inspection

    # ========================================================
    # HUMAN REVIEW
    # ========================================================

    async def review(
        self,
        inspection_id: UUID | str,
        *,
        company_id: str,
        reviewed_by: str,
        notes: str | None = None,
    ) -> Inspection:
        reviewed_by = self._require_text(
            reviewed_by,
            "reviewed_by",
        )

        inspection = await self.get_required(
            inspection_id,
            company_id=company_id,
        )

        if "reviewed_by" in self.columns:
            inspection.reviewed_by = reviewed_by

        if "reviewed_at" in self.columns:
            inspection.reviewed_at = utc_now()

        if (
            notes is not None
            and "review_notes" in self.columns
        ):
            inspection.review_notes = notes

        self._touch(inspection)
        await self.session.flush()

        return inspection

    # ========================================================
    # APPROVE INSPECTION
    # ========================================================

    async def approve(
        self,
        inspection_id: UUID | str,
        *,
        company_id: str,
        approved_by: str,
    ) -> Inspection:
        approved_by = self._require_text(
            approved_by,
            "approved_by",
        )

        inspection = await self.get_required(
            inspection_id,
            company_id=company_id,
        )

        inspection.status = "approved"

        if "approved_by" in self.columns:
            inspection.approved_by = approved_by

        if "approved_at" in self.columns:
            inspection.approved_at = utc_now()

        self._touch(inspection)
        await self.session.flush()

        return inspection

    # ========================================================
    # RESOLVE INSPECTION
    # ========================================================

    async def resolve(
        self,
        inspection_id: UUID | str,
        *,
        company_id: str,
        resolved_by: str,
    ) -> Inspection:
        resolved_by = self._require_text(
            resolved_by,
            "resolved_by",
        )

        inspection = await self.get_required(
            inspection_id,
            company_id=company_id,
        )

        inspection.status = "resolved"

        if "resolved_by" in self.columns:
            inspection.resolved_by = resolved_by

        if "resolved_at" in self.columns:
            inspection.resolved_at = utc_now()

        self._touch(inspection)
        await self.session.flush()

        return inspection

    # ========================================================
    # SOFT DELETE
    # ========================================================

    async def soft_delete(
        self,
        inspection_id: UUID | str,
        *,
        company_id: str,
    ) -> bool:
        if "deleted_at" not in self.columns:
            raise InspectionRepositoryError(
                "Soft deletion requires deleted_at"
            )

        inspection = await self.get_by_id(
            inspection_id,
            company_id=company_id,
        )

        if inspection is None:
            return False

        inspection.deleted_at = utc_now()

        self._touch(inspection)
        await self.session.flush()

        return True

    # ========================================================
    # LIST INSPECTIONS
    # ========================================================

    async def list(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        asset_id: str | None = None,
        machine_id: str | None = None,
        inspection_type: str | None = None,
        status: str | None = None,
        result: str | None = None,
        defect_severity: str | None = None,
        risk_level: str | None = None,
        model_name: str | None = None,
        search: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "inspected_at",
        descending: bool = True,
    ) -> tuple[list[Inspection], int]:
        self._validate_period(
            start_time,
            end_time,
        )

        if page < 1:
            raise InspectionValidationError(
                "page must be >= 1"
            )

        self._validate_limit(
            page_size,
            maximum=100,
        )

        if sort_by not in self.SORT_FIELDS:
            raise InspectionValidationError(
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
            "inspection_type": inspection_type,
            "status": status,
            "result": result,
            "defect_severity": defect_severity,
            "risk_level": risk_level,
            "model_name": model_name,
        }

        for name, value in filters.items():
            if value is not None:
                conditions.append(
                    self._column(name) == value
                )

        if start_time is not None:
            conditions.append(
                self._column("inspected_at")
                >= start_time
            )

        if end_time is not None:
            conditions.append(
                self._column("inspected_at")
                <= end_time
            )

        if search:
            pattern = f"%{search.strip()}%"

            searchable = [
                name
                for name in (
                    "inspection_number",
                    "description",
                    "defect_type",
                )
                if name in self.columns
            ]

            if searchable:
                conditions.append(
                    or_(*[
                        self._column(name).ilike(pattern)
                        for name in searchable
                    ])
                )

        predicate = and_(*conditions)

        total = (
            await self.session.execute(
                select(func.count())
                .select_from(Inspection)
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
            select(Inspection)
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
    # CRITICAL INSPECTIONS
    # ========================================================

    async def get_critical(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        limit: int = 100,
    ) -> list[Inspection]:
        self._validate_limit(limit)

        query = (
            select(Inspection)
            .where(
                self._scope(
                    company_id=company_id,
                    site_id=site_id,
                ),
                self._column("defect_severity").in_(
                    self.CRITICAL_SEVERITIES
                ),
            )
            .order_by(
                self._column("inspected_at").desc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # ========================================================
    # FAILED INSPECTIONS
    # ========================================================

    async def get_failed(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        limit: int = 100,
    ) -> list[Inspection]:
        self._validate_limit(limit)

        query = (
            select(Inspection)
            .where(
                self._scope(
                    company_id=company_id,
                    site_id=site_id,
                ),
                self._column("result").in_(
                    self.FAILED_RESULTS
                ),
            )
            .order_by(
                self._column("inspected_at").desc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # ========================================================
    # ASSET INSPECTION HISTORY
    # ========================================================

    async def get_asset_history(
        self,
        *,
        company_id: str,
        asset_id: str,
        limit: int = 500,
    ) -> list[Inspection]:
        self._validate_limit(limit)

        query = (
            select(Inspection)
            .where(
                self._scope(
                    company_id=company_id,
                    asset_id=asset_id,
                )
            )
            .order_by(
                self._column("inspected_at").desc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # ========================================================
    # LATEST ASSET INSPECTION
    # ========================================================

    async def get_latest_asset_inspection(
        self,
        *,
        company_id: str,
        asset_id: str,
        inspection_type: str | None = None,
    ) -> Inspection | None:
        conditions = [
            self._scope(
                company_id=company_id,
                asset_id=asset_id,
            )
        ]

        if inspection_type is not None:
            conditions.append(
                self._column("inspection_type")
                == inspection_type
            )

        query = (
            select(Inspection)
            .where(and_(*conditions))
            .order_by(
                self._column("inspected_at").desc(),
                self._column("id").desc(),
            )
            .limit(1)
        )

        result = await self.session.execute(query)

        return result.scalar_one_or_none()

    # ========================================================
    # INSPECTION STATISTICS
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
                self._column("inspected_at")
                >= start_time
            )

        if end_time is not None:
            conditions.append(
                self._column("inspected_at")
                <= end_time
            )

        predicate = and_(*conditions)

        total = (
            await self.session.execute(
                select(func.count())
                .select_from(Inspection)
                .where(predicate)
            )
        ).scalar_one()

        status_counts: dict[str, int] = {}
        severity_counts: dict[str, int] = {}
        result_counts: dict[str, int] = {}

        for field, destination in (
            ("status", status_counts),
            ("defect_severity", severity_counts),
            ("result", result_counts),
        ):
            if field not in self.columns:
                continue

            column = self._column(field)

            result = await self.session.execute(
                select(
                    column,
                    func.count(),
                )
                .where(predicate)
                .group_by(column)
            )

            for value, count in result.all():
                destination[
                    str(self._normalize(value))
                ] = int(count)

        average_condition = None

        if "condition_score" in self.columns:
            value = (
                await self.session.execute(
                    select(
                        func.avg(
                            self._column("condition_score")
                        )
                    )
                    .where(predicate)
                )
            ).scalar_one()

            if value is not None:
                average_condition = float(value)

        return {
            "company_id": company_id,
            "site_id": site_id,
            "total_inspections": int(total),
            "critical_inspections": sum(
                count
                for severity, count in severity_counts.items()
                if severity in self.CRITICAL_SEVERITIES
            ),
            "failed_inspections": sum(
                count
                for result, count in result_counts.items()
                if result in self.FAILED_RESULTS
            ),
            "average_condition_score": average_condition,
            "status_counts": status_counts,
            "defect_severity_counts": severity_counts,
            "result_counts": result_counts,
            "generated_at": utc_now(),
        }


__all__ = [
    "InspectionRepository",
    "InspectionRepositoryError",
    "InspectionNotFoundError",
    "InspectionValidationError",
    "InspectionConflictError",
]
