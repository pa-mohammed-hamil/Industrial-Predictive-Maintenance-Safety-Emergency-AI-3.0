
"""
Industrial Failure Repository
=============================

File: app/repositories/failure_repository.py

Python: 3.11+
SQLAlchemy: 2.x

Responsibilities:
- Persist actual and predicted equipment failures
- Maintain tenant-scoped failure records
- Track failure probability and RUL
- Retrieve failure history and high-risk failures
- Record failure investigations and root causes
- Track human reviews and resolution
- Aggregate failure statistics and trends
- Soft-delete records where permitted

Dependencies:
    pip install "sqlalchemy[asyncio]>=2.0"

Expected SQLAlchemy model:
    app.models.failure.Failure

Required model columns:
    id
    company_id
    site_id
    asset_id
    failure_type
    status
    detected_at

Recommended model columns:
    machine_id
    component_id
    sensor_id
    failure_mode
    severity
    risk_level
    failure_probability
    rul_hours
    predicted_failure_at
    actual_failure_at
    description
    root_cause
    model_name
    model_version
    reviewed_by
    reviewed_at
    review_notes
    resolved_by
    resolved_at
    resolution_notes
    metadata_json
    created_at
    updated_at
    deleted_at

The service layer owns database transactions.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from math import isfinite
from typing import Any, Mapping, Sequence
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.inspection import inspect

from app.models.failure import Failure


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class FailureRepositoryError(Exception):
    """Base repository exception."""


class FailureNotFoundError(FailureRepositoryError):
    """Failure record not found."""


class FailureValidationError(FailureRepositoryError):
    """Invalid failure repository input."""


class FailureRepository:
    """
    Async persistence layer for equipment failures.

    Security:
        Every operation is scoped by company_id.

    Transactions:
        Writes call flush(), not commit().

    Important:
        The repository stores predictions and observations.
        It does not make operational safety decisions.
    """

    CREATE_FIELDS = frozenset({
        "id",
        "site_id",
        "asset_id",
        "machine_id",
        "component_id",
        "sensor_id",
        "failure_type",
        "failure_mode",
        "status",
        "severity",
        "risk_level",
        "failure_probability",
        "rul_hours",
        "predicted_failure_at",
        "actual_failure_at",
        "description",
        "root_cause",
        "model_name",
        "model_version",
        "metadata_json",
        "detected_at",
    })

    UPDATE_FIELDS = frozenset({
        "failure_mode",
        "severity",
        "risk_level",
        "description",
        "root_cause",
        "metadata_json",
    })

    SORT_FIELDS = frozenset({
        "detected_at",
        "created_at",
        "updated_at",
        "failure_probability",
        "rul_hours",
        "severity",
    })

    HIGH_RISK_LEVELS = frozenset({
        "high",
        "critical",
        "extreme",
    })

    ACTIVE_STATUSES = frozenset({
        "predicted",
        "detected",
        "open",
        "investigating",
        "confirmed",
        "in_progress",
        "escalated",
    })

    CLOSED_STATUSES = frozenset({
        "resolved",
        "closed",
        "false_positive",
    })

    def __init__(self, session: AsyncSession):
        self.session = session

        self.columns = {
            column.key
            for column in inspect(Failure).mapper.columns
        }

        required = {
            "id",
            "company_id",
            "site_id",
            "asset_id",
            "failure_type",
            "status",
            "detected_at",
        }

        missing = required - self.columns

        if missing:
            raise FailureRepositoryError(
                "Failure model missing required columns: "
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
    def _require_text(value: str, name: str) -> str:
        if not isinstance(value, str):
            raise FailureValidationError(
                f"{name} must be a string"
            )

        value = value.strip()

        if not value:
            raise FailureValidationError(
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

        if value.tzinfo is None or value.utcoffset() is None:
            raise FailureValidationError(
                f"{name} must be timezone-aware"
            )

    @classmethod
    def _validate_period(
        cls,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> None:
        cls._validate_datetime(start_time, "start_time")
        cls._validate_datetime(end_time, "end_time")

        if (
            start_time is not None
            and end_time is not None
            and start_time > end_time
        ):
            raise FailureValidationError(
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
            raise FailureValidationError(
                f"limit must be between 1 and {maximum}"
            )

    def _column(self, name: str):
        if name not in self.columns:
            raise FailureValidationError(
                f"Failure model has no column '{name}'"
            )
        return getattr(Failure, name)

    def _validate_payload(
        self,
        data: Mapping[str, Any],
        *,
        allowed: frozenset[str],
    ) -> dict[str, Any]:
        if not isinstance(data, Mapping):
            raise FailureValidationError(
                "Failure data must be a mapping"
            )

        payload = {
            key: self._normalize(value)
            for key, value in data.items()
        }

        unsupported = set(payload) - allowed

        if unsupported:
            raise FailureValidationError(
                f"Unsupported fields: {sorted(unsupported)}"
            )

        missing_columns = set(payload) - self.columns

        if missing_columns:
            raise FailureValidationError(
                "Fields missing from Failure model: "
                f"{sorted(missing_columns)}"
            )

        if (
            "failure_probability" in payload
            and payload["failure_probability"] is not None
        ):
            probability = payload["failure_probability"]

            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise FailureValidationError(
                    "failure_probability must be in [0, 1]"
                )

        if (
            "rul_hours" in payload
            and payload["rul_hours"] is not None
        ):
            rul = payload["rul_hours"]

            if (
                isinstance(rul, bool)
                or not isinstance(rul, (int, float))
                or not isfinite(rul)
                or rul < 0
            ):
                raise FailureValidationError(
                    "rul_hours must be finite and >= 0"
                )

        for name in (
            "detected_at",
            "predicted_failure_at",
            "actual_failure_at",
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

    # ========================================================
    # CREATE
    # ========================================================

    async def create(
        self,
        *,
        company_id: str,
        data: Mapping[str, Any],
    ) -> Failure:
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
            "failure_type",
        ):
            payload[name] = self._require_text(
                payload.get(name),
                name,
            )

        payload["company_id"] = company_id

        payload.setdefault("status", "detected")
        payload.setdefault("detected_at", utc_now())

        if "created_at" in self.columns:
            payload.setdefault("created_at", utc_now())

        if "updated_at" in self.columns:
            payload.setdefault("updated_at", utc_now())

        failure = Failure(**payload)

        self.session.add(failure)
        await self.session.flush()

        return failure

    async def create_many(
        self,
        *,
        company_id: str,
        records: Sequence[Mapping[str, Any]],
    ) -> list[Failure]:
        if len(records) > 1000:
            raise FailureValidationError(
                "Maximum 1000 records per batch"
            )

        created: list[Failure] = []

        for record in records:
            created.append(
                await self.create(
                    company_id=company_id,
                    data=record,
                )
            )

        return created

    # ========================================================
    # GET
    # ========================================================

    async def get_by_id(
        self,
        failure_id: UUID | str,
        *,
        company_id: str,
        include_deleted: bool = False,
    ) -> Failure | None:
        query = (
            select(Failure)
            .where(
                self._column("id") == failure_id,
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
        failure_id: UUID | str,
        *,
        company_id: str,
    ) -> Failure:
        failure = await self.get_by_id(
            failure_id,
            company_id=company_id,
        )

        if failure is None:
            raise FailureNotFoundError(
                f"Failure '{failure_id}' not found"
            )

        return failure

    # ========================================================
    # UPDATE
    # ========================================================

    async def update(
        self,
        failure_id: UUID | str,
        *,
        company_id: str,
        data: Mapping[str, Any],
    ) -> Failure:
        failure = await self.get_required(
            failure_id,
            company_id=company_id,
        )

        payload = self._validate_payload(
            data,
            allowed=self.UPDATE_FIELDS,
        )

        for key, value in payload.items():
            setattr(failure, key, value)

        if "updated_at" in self.columns:
            failure.updated_at = utc_now()

        await self.session.flush()
        return failure

    # ========================================================
    # UPDATE PREDICTION
    # ========================================================

    async def update_prediction(
        self,
        failure_id: UUID | str,
        *,
        company_id: str,
        failure_probability: float | None = None,
        rul_hours: float | None = None,
        predicted_failure_at: datetime | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> Failure:
        failure = await self.get_required(
            failure_id,
            company_id=company_id,
        )

        data = {
            "failure_probability": failure_probability,
            "rul_hours": rul_hours,
            "predicted_failure_at": predicted_failure_at,
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
            setattr(failure, key, value)

        if "updated_at" in self.columns:
            failure.updated_at = utc_now()

        await self.session.flush()
        return failure

    # ========================================================
    # CONFIRM ACTUAL FAILURE
    # ========================================================

    async def confirm_failure(
        self,
        failure_id: UUID | str,
        *,
        company_id: str,
        actual_failure_at: datetime,
        confirmed_by: str,
        description: str | None = None,
    ) -> Failure:
        self._validate_datetime(
            actual_failure_at,
            "actual_failure_at",
        )

        self._require_text(
            confirmed_by,
            "confirmed_by",
        )

        failure = await self.get_required(
            failure_id,
            company_id=company_id,
        )

        failure.status = "confirmed"

        if "actual_failure_at" in self.columns:
            failure.actual_failure_at = actual_failure_at

        if "reviewed_by" in self.columns:
            failure.reviewed_by = confirmed_by

        if "reviewed_at" in self.columns:
            failure.reviewed_at = utc_now()

        if description is not None:
            self._column("description")
            failure.description = description

        if "updated_at" in self.columns:
            failure.updated_at = utc_now()

        await self.session.flush()
        return failure

    # ========================================================
    # ROOT CAUSE
    # ========================================================

    async def update_root_cause(
        self,
        failure_id: UUID | str,
        *,
        company_id: str,
        root_cause: str,
        investigated_by: str | None = None,
        notes: str | None = None,
    ) -> Failure:
        root_cause = self._require_text(
            root_cause,
            "root_cause",
        )

        failure = await self.get_required(
            failure_id,
            company_id=company_id,
        )

        self._column("root_cause")
        failure.root_cause = root_cause

        if (
            investigated_by is not None
            and "reviewed_by" in self.columns
        ):
            failure.reviewed_by = investigated_by

        if (
            notes is not None
            and "review_notes" in self.columns
        ):
            failure.review_notes = notes

        if "updated_at" in self.columns:
            failure.updated_at = utc_now()

        await self.session.flush()
        return failure

    # ========================================================
    # REVIEW
    # ========================================================

    async def review(
        self,
        failure_id: UUID | str,
        *,
        company_id: str,
        reviewed_by: str,
        notes: str | None = None,
        false_positive: bool = False,
    ) -> Failure:
        reviewed_by = self._require_text(
            reviewed_by,
            "reviewed_by",
        )

        failure = await self.get_required(
            failure_id,
            company_id=company_id,
        )

        if "reviewed_by" in self.columns:
            failure.reviewed_by = reviewed_by

        if "reviewed_at" in self.columns:
            failure.reviewed_at = utc_now()

        if (
            notes is not None
            and "review_notes" in self.columns
        ):
            failure.review_notes = notes

        if false_positive:
            failure.status = "false_positive"

        if "updated_at" in self.columns:
            failure.updated_at = utc_now()

        await self.session.flush()
        return failure

    # ========================================================
    # RESOLVE
    # ========================================================

    async def resolve(
        self,
        failure_id: UUID | str,
        *,
        company_id: str,
        resolved_by: str,
        resolution_notes: str | None = None,
    ) -> Failure:
        resolved_by = self._require_text(
            resolved_by,
            "resolved_by",
        )

        failure = await self.get_required(
            failure_id,
            company_id=company_id,
        )

        failure.status = "resolved"

        if "resolved_by" in self.columns:
            failure.resolved_by = resolved_by

        if "resolved_at" in self.columns:
            failure.resolved_at = utc_now()

        if (
            resolution_notes is not None
            and "resolution_notes" in self.columns
        ):
            failure.resolution_notes = resolution_notes

        if "updated_at" in self.columns:
            failure.updated_at = utc_now()

        await self.session.flush()
        return failure

    # ========================================================
    # SOFT DELETE
    # ========================================================

    async def soft_delete(
        self,
        failure_id: UUID | str,
        *,
        company_id: str,
    ) -> bool:
        if "deleted_at" not in self.columns:
            raise FailureRepositoryError(
                "Soft deletion requires deleted_at"
            )

        failure = await self.get_by_id(
            failure_id,
            company_id=company_id,
        )

        if failure is None:
            return False

        failure.deleted_at = utc_now()

        if "updated_at" in self.columns:
            failure.updated_at = utc_now()

        await self.session.flush()
        return True

    # ========================================================
    # FILTER / PAGINATION
    # ========================================================

    async def list(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        asset_id: str | None = None,
        machine_id: str | None = None,
        failure_type: str | None = None,
        severity: str | None = None,
        risk_level: str | None = None,
        status: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        min_probability: float | None = None,
        max_rul_hours: float | None = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "detected_at",
        descending: bool = True,
    ) -> tuple[list[Failure], int]:
        self._validate_period(start_time, end_time)

        if page < 1:
            raise FailureValidationError(
                "page must be >= 1"
            )

        self._validate_limit(page_size, maximum=100)

        if sort_by not in self.SORT_FIELDS:
            raise FailureValidationError(
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

        for field, value in {
            "failure_type": failure_type,
            "severity": severity,
            "risk_level": risk_level,
            "status": status,
        }.items():
            if value is not None:
                conditions.append(
                    self._column(field) == value
                )

        if start_time is not None:
            conditions.append(
                self._column("detected_at")
                >= start_time
            )

        if end_time is not None:
            conditions.append(
                self._column("detected_at")
                <= end_time
            )

        if min_probability is not None:
            if (
                not isfinite(min_probability)
                or not 0 <= min_probability <= 1
            ):
                raise FailureValidationError(
                    "min_probability must be in [0, 1]"
                )

            conditions.append(
                self._column("failure_probability")
                >= min_probability
            )

        if max_rul_hours is not None:
            if (
                not isfinite(max_rul_hours)
                or max_rul_hours < 0
            ):
                raise FailureValidationError(
                    "max_rul_hours must be >= 0"
                )

            conditions.append(
                self._column("rul_hours")
                <= max_rul_hours
            )

        predicate = and_(*conditions)

        count_query = (
            select(func.count())
            .select_from(Failure)
            .where(predicate)
        )

        total = (
            await self.session.execute(count_query)
        ).scalar_one()

        order_column = self._column(sort_by)

        ordering = (
            order_column.desc()
            if descending
            else order_column.asc()
        )

        query = (
            select(Failure)
            .where(predicate)
            .order_by(
                ordering,
                self._column("id").asc(),
            )
            .offset((page - 1) * page_size)
            .limit(page_size)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all()), int(total)

    # ========================================================
    # HIGH-RISK FAILURES
    # ========================================================

    async def get_high_risk(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        limit: int = 100,
    ) -> list[Failure]:
        self._validate_limit(limit)

        query = (
            select(Failure)
            .where(
                self._scope(
                    company_id=company_id,
                    site_id=site_id,
                ),
                self._column("risk_level").in_(
                    self.HIGH_RISK_LEVELS
                ),
                self._column("status").notin_(
                    self.CLOSED_STATUSES
                ),
            )
            .order_by(
                self._column("detected_at").desc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)
        return list(result.scalars().all())

    # ========================================================
    # LOW RUL FAILURES
    # ========================================================

    async def get_low_rul(
        self,
        *,
        company_id: str,
        threshold_hours: float = 72.0,
        site_id: str | None = None,
        limit: int = 100,
    ) -> list[Failure]:
        self._validate_limit(limit)

        if (
            not isfinite(threshold_hours)
            or threshold_hours < 0
        ):
            raise FailureValidationError(
                "threshold_hours must be >= 0"
            )

        query = (
            select(Failure)
            .where(
                self._scope(
                    company_id=company_id,
                    site_id=site_id,
                ),
                self._column("rul_hours")
                <= threshold_hours,
                self._column("status").notin_(
                    self.CLOSED_STATUSES
                ),
            )
            .order_by(
                self._column("rul_hours").asc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)
        return list(result.scalars().all())

    # ========================================================
    # ASSET FAILURE HISTORY
    # ========================================================

    async def get_asset_history(
        self,
        *,
        company_id: str,
        asset_id: str,
        limit: int = 500,
    ) -> list[Failure]:
        self._validate_limit(limit)

        query = (
            select(Failure)
            .where(
                self._scope(
                    company_id=company_id,
                    asset_id=asset_id,
                )
            )
            .order_by(
                self._column("detected_at").desc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)
        return list(result.scalars().all())

    # ========================================================
    # STATISTICS
    # ========================================================

    async def get_statistics(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        self._validate_period(start_time, end_time)

        conditions = [
            self._scope(
                company_id=company_id,
                site_id=site_id,
            )
        ]

        if start_time is not None:
            conditions.append(
                self._column("detected_at")
                >= start_time
            )

        if end_time is not None:
            conditions.append(
                self._column("detected_at")
                <= end_time
            )

        predicate = and_(*conditions)

        total = (
            await self.session.execute(
                select(func.count())
                .select_from(Failure)
                .where(predicate)
            )
        ).scalar_one()

        status_counts: dict[str, int] = {}
        risk_counts: dict[str, int] = {}

        for field, destination in (
            ("status", status_counts),
            ("risk_level", risk_counts),
        ):
            if field not in self.columns:
                continue

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

        avg_probability = None
        avg_rul_hours = None

        if "failure_probability" in self.columns:
            value = (
                await self.session.execute(
                    select(
                        func.avg(
                            self._column(
                                "failure_probability"
                            )
                        )
                    )
                    .where(predicate)
                )
            ).scalar_one()

            if value is not None:
                avg_probability = float(value)

        if "rul_hours" in self.columns:
            value = (
                await self.session.execute(
                    select(
                        func.avg(
                            self._column("rul_hours")
                        )
                    )
                    .where(predicate)
                )
            ).scalar_one()

            if value is not None:
                avg_rul_hours = float(value)

        return {
            "company_id": company_id,
            "site_id": site_id,
            "total_failures": int(total),
            "active_failures": sum(
                count
                for status, count in status_counts.items()
                if status in self.ACTIVE_STATUSES
            ),
            "high_risk_failures": sum(
                count
                for risk, count in risk_counts.items()
                if risk in self.HIGH_RISK_LEVELS
            ),
            "average_failure_probability": (
                avg_probability
            ),
            "average_rul_hours": avg_rul_hours,
            "status_counts": status_counts,
            "risk_counts": risk_counts,
            "generated_at": utc_now(),
        }


__all__ = [
    "FailureRepository",
    "FailureRepositoryError",
    "FailureNotFoundError",
    "FailureValidationError",
]
