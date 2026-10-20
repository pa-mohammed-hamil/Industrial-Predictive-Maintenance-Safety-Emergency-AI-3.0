
"""
app/repositories/anomaly_repository.py

Async SQLAlchemy 2.x repository for industrial anomalies.

Expected Anomaly model columns:
    id                  UUID (primary key)
    company_id          str
    site_id             str
    asset_id            str
    machine_id          str | None
    sensor_id           str | None
    anomaly_type        str
    severity            str
    status              str
    anomaly_score       float
    description         str | None
    detected_at         timezone-aware datetime
    created_at          timezone-aware datetime
    updated_at          timezone-aware datetime
    deleted_at          timezone-aware datetime | None

Optional columns:
    reviewed_by
    reviewed_at
    review_notes
    resolved_at
    acknowledged_by
    acknowledged_at
    model_name
    model_version
    metadata_json

The repository does not commit transactions.
The service layer owns commit/rollback.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from math import isfinite
from typing import Any, Mapping, Sequence
from uuid import UUID

from sqlalchemy import (
    and_,
    func,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.inspection import inspect

from app.models.anomaly import Anomaly


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AnomalyRepositoryError(Exception):
    pass


class AnomalyNotFoundError(AnomalyRepositoryError):
    pass


class AnomalyValidationError(AnomalyRepositoryError):
    pass


class AnomalyRepository:
    """
    Tenant-scoped anomaly persistence.

    Design:
    - Explicit company_id on all reads/writes
    - No automatic commits
    - Optional soft deletion
    - Allowlisted fields
    - Stable pagination
    - SQL-level filtering and aggregation
    """

    CREATE_FIELDS = frozenset({
        "id",
        "site_id",
        "asset_id",
        "machine_id",
        "sensor_id",
        "anomaly_type",
        "severity",
        "status",
        "anomaly_score",
        "description",
        "detected_at",
        "model_name",
        "model_version",
        "metadata_json",
    })

    UPDATE_FIELDS = frozenset({
        "severity",
        "status",
        "description",
        "reviewed_by",
        "reviewed_at",
        "review_notes",
        "resolved_at",
        "acknowledged_by",
        "acknowledged_at",
        "metadata_json",
    })

    SORT_FIELDS = frozenset({
        "detected_at",
        "created_at",
        "updated_at",
        "anomaly_score",
        "severity",
    })

    ACTIVE_STATUSES = frozenset({
        "detected",
        "open",
        "acknowledged",
        "investigating",
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
            for column in inspect(Anomaly).mapper.columns
        }

        required = {
            "id",
            "company_id",
            "site_id",
            "asset_id",
            "detected_at",
        }

        missing = required - self.columns

        if missing:
            raise AnomalyRepositoryError(
                "Anomaly model missing columns: "
                f"{sorted(missing)}"
            )

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------

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
            raise AnomalyValidationError(
                f"{name} must be a string"
            )

        value = value.strip()

        if not value:
            raise AnomalyValidationError(
                f"{name} is required"
            )

        return value

    def _column(self, name: str):
        if name not in self.columns:
            raise AnomalyValidationError(
                f"Unsupported Anomaly column: {name}"
            )

        return getattr(Anomaly, name)

    def _validate_payload(
        self,
        payload: Mapping[str, Any],
        *,
        allowed: frozenset[str],
    ) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise AnomalyValidationError(
                "Payload must be a mapping"
            )

        data = {
            key: self._normalize(value)
            for key, value in payload.items()
        }

        unsupported = set(data) - allowed

        if unsupported:
            raise AnomalyValidationError(
                "Unsupported fields: "
                f"{sorted(unsupported)}"
            )

        missing_columns = set(data) - self.columns

        if missing_columns:
            raise AnomalyValidationError(
                "Fields missing from Anomaly model: "
                f"{sorted(missing_columns)}"
            )

        return data

    def _scope(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        asset_id: str | None = None,
        machine_id: str | None = None,
        sensor_id: str | None = None,
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
            "sensor_id": sensor_id,
        }

        for field, value in filters.items():
            if value is not None:
                conditions.append(
                    self._column(field) == value
                )

        if (
            not include_deleted
            and "deleted_at" in self.columns
        ):
            conditions.append(
                self._column("deleted_at").is_(None)
            )

        return and_(*conditions)

    @staticmethod
    def _validate_period(
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> None:
        for name, value in (
            ("start_time", start_time),
            ("end_time", end_time),
        ):
            if (
                value is not None
                and (
                    value.tzinfo is None
                    or value.utcoffset() is None
                )
            ):
                raise AnomalyValidationError(
                    f"{name} must be timezone-aware"
                )

        if (
            start_time is not None
            and end_time is not None
            and start_time > end_time
        ):
            raise AnomalyValidationError(
                "start_time cannot exceed end_time"
            )

    @staticmethod
    def _validate_limit(
        limit: int,
        maximum: int = 1000,
    ) -> None:
        if not 1 <= limit <= maximum:
            raise AnomalyValidationError(
                f"limit must be between 1 and {maximum}"
            )

    # --------------------------------------------------------
    # CREATE
    # --------------------------------------------------------

    async def create(
        self,
        *,
        company_id: str,
        data: Mapping[str, Any],
    ) -> Anomaly:
        company_id = self._require_text(
            company_id,
            "company_id",
        )

        payload = self._validate_payload(
            data,
            allowed=self.CREATE_FIELDS,
        )

        for name in ("site_id", "asset_id"):
            self._require_text(
                payload.get(name),
                name,
            )

        payload["company_id"] = company_id

        payload.setdefault(
            "detected_at",
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

        anomaly = Anomaly(**payload)

        self.session.add(anomaly)

        await self.session.flush()

        return anomaly

    async def create_many(
        self,
        *,
        company_id: str,
        records: Sequence[Mapping[str, Any]],
    ) -> list[Anomaly]:
        if not records:
            return []

        if len(records) > 1000:
            raise AnomalyValidationError(
                "Maximum 1000 anomalies per batch"
            )

        created = []

        for data in records:
            created.append(
                await self.create(
                    company_id=company_id,
                    data=data,
                )
            )

        return created

    # --------------------------------------------------------
    # GET
    # --------------------------------------------------------

    async def get_by_id(
        self,
        anomaly_id: UUID | str,
        *,
        company_id: str,
        include_deleted: bool = False,
    ) -> Anomaly | None:
        statement = (
            select(Anomaly)
            .where(
                self._column("id") == anomaly_id,
                self._scope(
                    company_id=company_id,
                    include_deleted=include_deleted,
                ),
            )
        )

        result = await self.session.execute(
            statement
        )

        return result.scalar_one_or_none()

    async def get_required(
        self,
        anomaly_id: UUID | str,
        *,
        company_id: str,
    ) -> Anomaly:
        anomaly = await self.get_by_id(
            anomaly_id,
            company_id=company_id,
        )

        if anomaly is None:
            raise AnomalyNotFoundError(
                f"Anomaly {anomaly_id} not found"
            )

        return anomaly

    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    async def update(
        self,
        anomaly_id: UUID | str,
        *,
        company_id: str,
        data: Mapping[str, Any],
    ) -> Anomaly:
        anomaly = await self.get_required(
            anomaly_id,
            company_id=company_id,
        )

        payload = self._validate_payload(
            data,
            allowed=self.UPDATE_FIELDS,
        )

        for key, value in payload.items():
            setattr(anomaly, key, value)

        if "updated_at" in self.columns:
            anomaly.updated_at = utc_now()

        await self.session.flush()

        return anomaly

    # --------------------------------------------------------
    # ACKNOWLEDGE
    # --------------------------------------------------------

    async def acknowledge(
        self,
        anomaly_id: UUID | str,
        *,
        company_id: str,
        acknowledged_by: str,
    ) -> Anomaly:
        anomaly = await self.get_required(
            anomaly_id,
            company_id=company_id,
        )

        acknowledged_by = self._require_text(
            acknowledged_by,
            "acknowledged_by",
        )

        if "status" in self.columns:
            anomaly.status = "acknowledged"

        if "acknowledged_by" in self.columns:
            anomaly.acknowledged_by = acknowledged_by

        if "acknowledged_at" in self.columns:
            anomaly.acknowledged_at = utc_now()

        if "updated_at" in self.columns:
            anomaly.updated_at = utc_now()

        await self.session.flush()

        return anomaly

    # --------------------------------------------------------
    # RESOLVE
    # --------------------------------------------------------

    async def resolve(
        self,
        anomaly_id: UUID | str,
        *,
        company_id: str,
        resolved_by: str,
        resolution_notes: str | None = None,
    ) -> Anomaly:
        anomaly = await self.get_required(
            anomaly_id,
            company_id=company_id,
        )

        resolved_by = self._require_text(
            resolved_by,
            "resolved_by",
        )

        if "status" in self.columns:
            anomaly.status = "resolved"

        if "resolved_at" in self.columns:
            anomaly.resolved_at = utc_now()

        if "reviewed_by" in self.columns:
            anomaly.reviewed_by = resolved_by

        if (
            resolution_notes is not None
            and "review_notes" in self.columns
        ):
            anomaly.review_notes = resolution_notes

        if "updated_at" in self.columns:
            anomaly.updated_at = utc_now()

        await self.session.flush()

        return anomaly

    # --------------------------------------------------------
    # REVIEW
    # --------------------------------------------------------

    async def review(
        self,
        anomaly_id: UUID | str,
        *,
        company_id: str,
        reviewed_by: str,
        notes: str | None = None,
        is_false_positive: bool = False,
    ) -> Anomaly:
        anomaly = await self.get_required(
            anomaly_id,
            company_id=company_id,
        )

        reviewed_by = self._require_text(
            reviewed_by,
            "reviewed_by",
        )

        if "reviewed_by" in self.columns:
            anomaly.reviewed_by = reviewed_by

        if "reviewed_at" in self.columns:
            anomaly.reviewed_at = utc_now()

        if (
            notes is not None
            and "review_notes" in self.columns
        ):
            anomaly.review_notes = notes

        if (
            is_false_positive
            and "status" in self.columns
        ):
            anomaly.status = "false_positive"

        if "updated_at" in self.columns:
            anomaly.updated_at = utc_now()

        await self.session.flush()

        return anomaly

    # --------------------------------------------------------
    # SOFT DELETE
    # --------------------------------------------------------

    async def soft_delete(
        self,
        anomaly_id: UUID | str,
        *,
        company_id: str,
    ) -> bool:
        if "deleted_at" not in self.columns:
            raise AnomalyRepositoryError(
                "Soft deletion requires deleted_at "
                "on the Anomaly model"
            )

        anomaly = await self.get_by_id(
            anomaly_id,
            company_id=company_id,
        )

        if anomaly is None:
            return False

        anomaly.deleted_at = utc_now()

        if "updated_at" in self.columns:
            anomaly.updated_at = utc_now()

        await self.session.flush()

        return True

    # --------------------------------------------------------
    # LIST / FILTER
    # --------------------------------------------------------

    async def list(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        asset_id: str | None = None,
        machine_id: str | None = None,
        sensor_id: str | None = None,
        anomaly_type: str | None = None,
        severity: str | None = None,
        status: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        min_anomaly_score: float | None = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "detected_at",
        descending: bool = True,
    ) -> tuple[list[Anomaly], int]:
        self._validate_period(
            start_time,
            end_time,
        )

        if page < 1:
            raise AnomalyValidationError(
                "page must be >= 1"
            )

        self._validate_limit(
            page_size,
            maximum=100,
        )

        if sort_by not in self.SORT_FIELDS:
            raise AnomalyValidationError(
                f"Invalid sort field: {sort_by}"
            )

        conditions = [
            self._scope(
                company_id=company_id,
                site_id=site_id,
                asset_id=asset_id,
                machine_id=machine_id,
                sensor_id=sensor_id,
            )
        ]

        filters = {
            "anomaly_type": anomaly_type,
            "severity": severity,
            "status": status,
        }

        for field, value in filters.items():
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

        if min_anomaly_score is not None:
            if (
                not isfinite(min_anomaly_score)
                or not 0 <= min_anomaly_score <= 1
            ):
                raise AnomalyValidationError(
                    "min_anomaly_score must be in [0, 1]"
                )

            conditions.append(
                self._column("anomaly_score")
                >= min_anomaly_score
            )

        predicate = and_(*conditions)

        count_query = (
            select(func.count())
            .select_from(Anomaly)
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
            select(Anomaly)
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

    # --------------------------------------------------------
    # RECENT ANOMALIES
    # --------------------------------------------------------

    async def get_recent(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        hours: int = 24,
        limit: int = 50,
    ) -> list[Anomaly]:
        if hours <= 0:
            raise AnomalyValidationError(
                "hours must be positive"
            )

        self._validate_limit(limit)

        cutoff = utc_now() - timedelta(
            hours=hours
        )

        query = (
            select(Anomaly)
            .where(
                self._scope(
                    company_id=company_id,
                    site_id=site_id,
                ),
                self._column("detected_at") >= cutoff,
            )
            .order_by(
                self._column("detected_at").desc(),
                self._column("id").asc(),
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # --------------------------------------------------------
    # ACTIVE ANOMALIES
    # --------------------------------------------------------

    async def get_active(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        asset_id: str | None = None,
        limit: int = 100,
    ) -> list[Anomaly]:
        self._validate_limit(limit)

        query = (
            select(Anomaly)
            .where(
                self._scope(
                    company_id=company_id,
                    site_id=site_id,
                    asset_id=asset_id,
                ),
                self._column("status").in_(
                    self.ACTIVE_STATUSES
                ),
            )
            .order_by(
                self._column("detected_at").desc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # --------------------------------------------------------
    # CRITICAL ANOMALIES
    # --------------------------------------------------------

    async def get_critical(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        limit: int = 100,
    ) -> list[Anomaly]:
        self._validate_limit(limit)

        query = (
            select(Anomaly)
            .where(
                self._scope(
                    company_id=company_id,
                    site_id=site_id,
                ),
                self._column("severity") == "critical",
            )
            .order_by(
                self._column("detected_at").desc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # --------------------------------------------------------
    # ASSET HISTORY
    # --------------------------------------------------------

    async def get_asset_history(
        self,
        *,
        company_id: str,
        asset_id: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> list[Anomaly]:
        self._validate_period(
            start_time,
            end_time,
        )

        self._validate_limit(
            limit,
            maximum=10000,
        )

        conditions = [
            self._scope(
                company_id=company_id,
                asset_id=asset_id,
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

        query = (
            select(Anomaly)
            .where(and_(*conditions))
            .order_by(
                self._column("detected_at").desc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

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
                self._column("detected_at")
                >= start_time
            )

        if end_time is not None:
            conditions.append(
                self._column("detected_at")
                <= end_time
            )

        predicate = and_(*conditions)

        total_query = (
            select(func.count())
            .select_from(Anomaly)
            .where(predicate)
        )

        total = (
            await self.session.execute(total_query)
        ).scalar_one()

        status_counts: dict[str, int] = {}
        severity_counts: dict[str, int] = {}

        if "status" in self.columns:
            status_column = self._column("status")

            result = await self.session.execute(
                select(
                    status_column,
                    func.count(),
                )
                .where(predicate)
                .group_by(status_column)
            )

            status_counts = {
                str(self._normalize(status)): int(count)
                for status, count in result.all()
            }

        if "severity" in self.columns:
            severity_column = self._column(
                "severity"
            )

            result = await self.session.execute(
                select(
                    severity_column,
                    func.count(),
                )
                .where(predicate)
                .group_by(severity_column)
            )

            severity_counts = {
                str(self._normalize(severity)): int(count)
                for severity, count in result.all()
            }

        average_score = None

        if "anomaly_score" in self.columns:
            result = await self.session.execute(
                select(
                    func.avg(
                        self._column("anomaly_score")
                    )
                )
                .where(predicate)
            )

            score = result.scalar_one()

            if score is not None:
                average_score = float(score)

        active_count = sum(
            count
            for status, count in status_counts.items()
            if status in self.ACTIVE_STATUSES
        )

        return {
            "company_id": company_id,
            "site_id": site_id,
            "total_anomalies": int(total),
            "active_anomalies": active_count,
            "critical_anomalies": (
                severity_counts.get("critical", 0)
            ),
            "average_anomaly_score": average_score,
            "status_counts": status_counts,
            "severity_counts": severity_counts,
            "generated_at": utc_now(),
        }


__all__ = [
    "AnomalyRepository",
    "AnomalyRepositoryError",
    "AnomalyNotFoundError",
    "AnomalyValidationError",
]
