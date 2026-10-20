
"""
Industrial Cost Intelligence Repository
=======================================

File: app/repositories/cost_repository.py

Python: 3.11+
SQLAlchemy: 2.x

Features:
- Cost estimate creation and retrieval
- Multi-tenant company/site isolation
- Repair and replacement cost tracking
- Labor, spare parts, downtime and energy costs
- Production loss and emergency costs
- Asset-level financial history
- Cost estimate status and approval tracking
- Cost breakdown analytics
- Time-range financial queries
- Monthly cost trends
- Cost dashboard statistics
- Soft deletion
- Pagination and filtering

Dependencies:
    pip install "sqlalchemy[asyncio]>=2.0"

Expected model:
    app.models.cost.CostEstimate

Required columns:
    id
    company_id
    site_id
    asset_id
    cost_type
    total_cost
    currency
    status
    created_at

Recommended optional columns:
    machine_id
    maintenance_id
    incident_id
    emergency_id
    estimate_name
    description
    labor_cost
    spare_parts_cost
    downtime_cost
    production_loss_cost
    energy_cost
    emergency_cost
    scrap_cost
    repair_cost
    replacement_cost
    estimated_savings
    roi_percent
    forecast_date
    approved_by
    approved_at
    approval_notes
    metadata_json
    updated_at
    deleted_at

The repository does not commit transactions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Mapping, Sequence
from uuid import UUID

from sqlalchemy import (
    and_,
    extract,
    func,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.inspection import inspect

from app.models.cost import CostEstimate


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================
# EXCEPTIONS
# ============================================================

class CostRepositoryError(Exception):
    """Base repository exception."""


class CostNotFoundError(CostRepositoryError):
    """Cost estimate not found."""


class CostValidationError(CostRepositoryError):
    """Invalid cost repository input."""


# ============================================================
# COST REPOSITORY
# ============================================================

class CostRepository:
    """
    Tenant-aware industrial cost repository.

    All data operations require company_id.

    Transactions:
        - flush() after writes
        - commit() handled by service layer

    Financial calculations should use Decimal.
    """

    CREATE_FIELDS = frozenset({
        "id",
        "site_id",
        "asset_id",
        "machine_id",
        "maintenance_id",
        "incident_id",
        "emergency_id",
        "cost_type",
        "estimate_name",
        "description",
        "total_cost",
        "currency",
        "status",
        "labor_cost",
        "spare_parts_cost",
        "downtime_cost",
        "production_loss_cost",
        "energy_cost",
        "emergency_cost",
        "scrap_cost",
        "repair_cost",
        "replacement_cost",
        "estimated_savings",
        "roi_percent",
        "forecast_date",
        "metadata_json",
    })

    UPDATE_FIELDS = frozenset({
        "estimate_name",
        "description",
        "status",
        "total_cost",
        "labor_cost",
        "spare_parts_cost",
        "downtime_cost",
        "production_loss_cost",
        "energy_cost",
        "emergency_cost",
        "scrap_cost",
        "repair_cost",
        "replacement_cost",
        "estimated_savings",
        "roi_percent",
        "forecast_date",
        "metadata_json",
    })

    COST_COMPONENT_FIELDS = frozenset({
        "labor_cost",
        "spare_parts_cost",
        "downtime_cost",
        "production_loss_cost",
        "energy_cost",
        "emergency_cost",
        "scrap_cost",
        "repair_cost",
        "replacement_cost",
    })

    SORT_FIELDS = frozenset({
        "created_at",
        "updated_at",
        "total_cost",
        "cost_type",
        "status",
        "forecast_date",
    })

    APPROVED_STATUSES = frozenset({
        "approved",
        "completed",
        "finalized",
    })

    PENDING_STATUSES = frozenset({
        "draft",
        "pending",
        "pending_approval",
        "under_review",
    })

    def __init__(self, session: AsyncSession):
        self.session = session

        self.columns = {
            column.key
            for column in inspect(CostEstimate).mapper.columns
        }

        required = {
            "id",
            "company_id",
            "site_id",
            "asset_id",
            "cost_type",
            "total_cost",
            "currency",
            "status",
            "created_at",
        }

        missing = required - self.columns

        if missing:
            raise CostRepositoryError(
                "CostEstimate model missing columns: "
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
            raise CostValidationError(
                f"{name} must be a string"
            )

        value = value.strip()

        if not value:
            raise CostValidationError(
                f"{name} is required"
            )

        return value

    @staticmethod
    def _money(
        value: Any,
        name: str,
        *,
        allow_negative: bool = False,
    ) -> Decimal:
        if isinstance(value, bool):
            raise CostValidationError(
                f"{name} must be numeric"
            )

        try:
            amount = Decimal(str(value))
        except (
            InvalidOperation,
            TypeError,
            ValueError,
        ) as exc:
            raise CostValidationError(
                f"{name} must be numeric"
            ) from exc

        if not amount.is_finite():
            raise CostValidationError(
                f"{name} must be finite"
            )

        if not allow_negative and amount < 0:
            raise CostValidationError(
                f"{name} cannot be negative"
            )

        return amount

    def _column(self, name: str):
        if name not in self.columns:
            raise CostValidationError(
                f"CostEstimate model has no column '{name}'"
            )

        return getattr(CostEstimate, name)

    def _validate_payload(
        self,
        data: Mapping[str, Any],
        *,
        allowed: frozenset[str],
    ) -> dict[str, Any]:
        if not isinstance(data, Mapping):
            raise CostValidationError(
                "Cost data must be a mapping"
            )

        payload = {
            key: self._normalize(value)
            for key, value in data.items()
        }

        unsupported = set(payload) - allowed

        if unsupported:
            raise CostValidationError(
                "Unsupported cost fields: "
                f"{sorted(unsupported)}"
            )

        missing_columns = set(payload) - self.columns

        if missing_columns:
            raise CostValidationError(
                "Fields missing from CostEstimate model: "
                f"{sorted(missing_columns)}"
            )

        monetary_fields = (
            self.COST_COMPONENT_FIELDS
            | {"total_cost", "estimated_savings"}
        )

        for name in monetary_fields:
            if name in payload and payload[name] is not None:
                payload[name] = self._money(
                    payload[name],
                    name,
                )

        if (
            "roi_percent" in payload
            and payload["roi_percent"] is not None
        ):
            payload["roi_percent"] = self._money(
                payload["roi_percent"],
                "roi_percent",
                allow_negative=True,
            )

        return payload

    def _scope(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        asset_id: str | None = None,
        include_deleted: bool = False,
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

        if asset_id is not None:
            conditions.append(
                self._column("asset_id") == asset_id
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
            if value is not None and (
                value.tzinfo is None
                or value.utcoffset() is None
            ):
                raise CostValidationError(
                    f"{name} must be timezone-aware"
                )

        if (
            start_time is not None
            and end_time is not None
            and start_time > end_time
        ):
            raise CostValidationError(
                "start_time cannot exceed end_time"
            )

    # ========================================================
    # CREATE COST ESTIMATE
    # ========================================================

    async def create(
        self,
        *,
        company_id: str,
        data: Mapping[str, Any],
    ) -> CostEstimate:
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
            "cost_type",
            "currency",
        ):
            self._require_text(
                payload.get(name),
                name,
            )

        if "total_cost" not in payload:
            raise CostValidationError(
                "total_cost is required"
            )

        payload["company_id"] = company_id

        payload.setdefault(
            "status",
            "draft",
        )

        payload.setdefault(
            "created_at",
            utc_now(),
        )

        if "updated_at" in self.columns:
            payload.setdefault(
                "updated_at",
                utc_now(),
            )

        estimate = CostEstimate(**payload)

        self.session.add(estimate)

        await self.session.flush()

        return estimate

    # ========================================================
    # BULK CREATE
    # ========================================================

    async def create_many(
        self,
        *,
        company_id: str,
        records: Sequence[Mapping[str, Any]],
    ) -> list[CostEstimate]:
        if len(records) > 1000:
            raise CostValidationError(
                "Maximum 1000 cost estimates per batch"
            )

        created = []

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
        estimate_id: UUID | str,
        *,
        company_id: str,
        include_deleted: bool = False,
    ) -> CostEstimate | None:
        query = (
            select(CostEstimate)
            .where(
                self._column("id") == estimate_id,
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
        estimate_id: UUID | str,
        *,
        company_id: str,
    ) -> CostEstimate:
        estimate = await self.get_by_id(
            estimate_id,
            company_id=company_id,
        )

        if estimate is None:
            raise CostNotFoundError(
                f"Cost estimate '{estimate_id}' not found"
            )

        return estimate

    # ========================================================
    # UPDATE COST ESTIMATE
    # ========================================================

    async def update(
        self,
        estimate_id: UUID | str,
        *,
        company_id: str,
        data: Mapping[str, Any],
    ) -> CostEstimate:
        estimate = await self.get_required(
            estimate_id,
            company_id=company_id,
        )

        payload = self._validate_payload(
            data,
            allowed=self.UPDATE_FIELDS,
        )

        for key, value in payload.items():
            setattr(estimate, key, value)

        if "updated_at" in self.columns:
            estimate.updated_at = utc_now()

        await self.session.flush()

        return estimate

    # ========================================================
    # UPDATE COST BREAKDOWN
    # ========================================================

    async def update_breakdown(
        self,
        estimate_id: UUID | str,
        *,
        company_id: str,
        breakdown: Mapping[str, Any],
        recalculate_total: bool = False,
    ) -> CostEstimate:
        estimate = await self.get_required(
            estimate_id,
            company_id=company_id,
        )

        payload = self._validate_payload(
            breakdown,
            allowed=self.COST_COMPONENT_FIELDS,
        )

        for key, value in payload.items():
            setattr(estimate, key, value)

        if recalculate_total:
            total = Decimal("0")

            for name in self.COST_COMPONENT_FIELDS:
                if name in self.columns:
                    value = getattr(
                        estimate,
                        name,
                        None,
                    )

                    if value is not None:
                        total += self._money(
                            value,
                            name,
                        )

            estimate.total_cost = total

        if "updated_at" in self.columns:
            estimate.updated_at = utc_now()

        await self.session.flush()

        return estimate

    # ========================================================
    # APPROVAL
    # ========================================================

    async def approve(
        self,
        estimate_id: UUID | str,
        *,
        company_id: str,
        approved_by: str,
        notes: str | None = None,
    ) -> CostEstimate:
        estimate = await self.get_required(
            estimate_id,
            company_id=company_id,
        )

        approved_by = self._require_text(
            approved_by,
            "approved_by",
        )

        estimate.status = "approved"

        if "approved_by" in self.columns:
            estimate.approved_by = approved_by

        if "approved_at" in self.columns:
            estimate.approved_at = utc_now()

        if (
            notes is not None
            and "approval_notes" in self.columns
        ):
            estimate.approval_notes = notes

        if "updated_at" in self.columns:
            estimate.updated_at = utc_now()

        await self.session.flush()

        return estimate

    # ========================================================
    # REJECT
    # ========================================================

    async def reject(
        self,
        estimate_id: UUID | str,
        *,
        company_id: str,
        reviewed_by: str,
        reason: str,
    ) -> CostEstimate:
        estimate = await self.get_required(
            estimate_id,
            company_id=company_id,
        )

        self._require_text(
            reviewed_by,
            "reviewed_by",
        )

        reason = self._require_text(
            reason,
            "reason",
        )

        estimate.status = "rejected"

        if "approved_by" in self.columns:
            estimate.approved_by = reviewed_by

        if "approval_notes" in self.columns:
            estimate.approval_notes = reason

        if "updated_at" in self.columns:
            estimate.updated_at = utc_now()

        await self.session.flush()

        return estimate

    # ========================================================
    # SOFT DELETE
    # ========================================================

    async def soft_delete(
        self,
        estimate_id: UUID | str,
        *,
        company_id: str,
    ) -> bool:
        if "deleted_at" not in self.columns:
            raise CostRepositoryError(
                "Soft deletion requires deleted_at"
            )

        estimate = await self.get_by_id(
            estimate_id,
            company_id=company_id,
        )

        if estimate is None:
            return False

        estimate.deleted_at = utc_now()

        if "updated_at" in self.columns:
            estimate.updated_at = utc_now()

        await self.session.flush()

        return True

    # ========================================================
    # LIST COST ESTIMATES
    # ========================================================

    async def list(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        asset_id: str | None = None,
        machine_id: str | None = None,
        cost_type: str | None = None,
        status: str | None = None,
        currency: str | None = None,
        min_cost: Decimal | None = None,
        max_cost: Decimal | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "created_at",
        descending: bool = True,
    ) -> tuple[list[CostEstimate], int]:
        self._validate_period(
            start_time,
            end_time,
        )

        if page < 1:
            raise CostValidationError(
                "page must be at least 1"
            )

        if not 1 <= page_size <= 100:
            raise CostValidationError(
                "page_size must be between 1 and 100"
            )

        if sort_by not in self.SORT_FIELDS:
            raise CostValidationError(
                f"Invalid sort field: {sort_by}"
            )

        conditions = [
            self._scope(
                company_id=company_id,
                site_id=site_id,
                asset_id=asset_id,
            )
        ]

        filters = {
            "machine_id": machine_id,
            "cost_type": cost_type,
            "status": status,
            "currency": currency,
        }

        for field, value in filters.items():
            if value is not None:
                conditions.append(
                    self._column(field) == value
                )

        if min_cost is not None:
            conditions.append(
                self._column("total_cost")
                >= self._money(
                    min_cost,
                    "min_cost",
                )
            )

        if max_cost is not None:
            conditions.append(
                self._column("total_cost")
                <= self._money(
                    max_cost,
                    "max_cost",
                )
            )

        if start_time is not None:
            conditions.append(
                self._column("created_at")
                >= start_time
            )

        if end_time is not None:
            conditions.append(
                self._column("created_at")
                <= end_time
            )

        predicate = and_(*conditions)

        count_query = (
            select(func.count())
            .select_from(CostEstimate)
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
            select(CostEstimate)
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
    # ASSET COST HISTORY
    # ========================================================

    async def get_asset_cost_history(
        self,
        *,
        company_id: str,
        asset_id: str,
        limit: int = 500,
    ) -> list[CostEstimate]:
        if not 1 <= limit <= 1000:
            raise CostValidationError(
                "limit must be between 1 and 1000"
            )

        query = (
            select(CostEstimate)
            .where(
                self._scope(
                    company_id=company_id,
                    asset_id=asset_id,
                )
            )
            .order_by(
                self._column("created_at").desc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # ========================================================
    # PENDING APPROVALS
    # ========================================================

    async def get_pending_approvals(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        limit: int = 100,
    ) -> list[CostEstimate]:
        if not 1 <= limit <= 1000:
            raise CostValidationError(
                "limit must be between 1 and 1000"
            )

        query = (
            select(CostEstimate)
            .where(
                self._scope(
                    company_id=company_id,
                    site_id=site_id,
                ),
                self._column("status").in_(
                    self.PENDING_STATUSES
                ),
            )
            .order_by(
                self._column("created_at").desc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # ========================================================
    # COST TOTAL BY TYPE
    # ========================================================

    async def get_cost_totals_by_type(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        currency: str | None = None,
    ) -> dict[str, Decimal]:
        conditions = [
            self._scope(
                company_id=company_id,
                site_id=site_id,
            )
        ]

        if currency is not None:
            conditions.append(
                self._column("currency") == currency
            )
        else:
            distinct_currencies = (
                await self.session.execute(
                    select(
                        func.count(
                            func.distinct(
                                self._column("currency")
                            )
                        )
                    )
                    .where(and_(*conditions))
                )
            ).scalar_one()

            if distinct_currencies > 1:
                raise CostValidationError(
                    "Specify currency when multiple "
                    "currencies are present"
                )

        cost_type_column = self._column(
            "cost_type"
        )

        query = (
            select(
                cost_type_column,
                func.sum(
                    self._column("total_cost")
                ),
            )
            .where(and_(*conditions))
            .group_by(cost_type_column)
        )

        result = await self.session.execute(query)

        return {
            str(self._normalize(cost_type)): (
                Decimal(str(total or 0))
            )
            for cost_type, total in result.all()
        }

    # ========================================================
    # MONTHLY COST TRENDS
    # ========================================================

    async def get_monthly_cost_trends(
        self,
        *,
        company_id: str,
        year: int,
        site_id: str | None = None,
        currency: str | None = None,
    ) -> list[dict[str, Any]]:
        if not 2000 <= year <= 2200:
            raise CostValidationError(
                "Invalid year"
            )

        conditions = [
            self._scope(
                company_id=company_id,
                site_id=site_id,
            ),
            extract(
                "year",
                self._column("created_at"),
            ) == year,
        ]

        if currency is not None:
            conditions.append(
                self._column("currency") == currency
            )
        else:
            currency_count = (
                await self.session.execute(
                    select(
                        func.count(
                            func.distinct(
                                self._column("currency")
                            )
                        )
                    )
                    .where(and_(*conditions))
                )
            ).scalar_one()

            if currency_count > 1:
                raise CostValidationError(
                    "Specify currency for monthly trends"
                )

        month_expression = extract(
            "month",
            self._column("created_at"),
        )

        query = (
            select(
                month_expression.label("month"),
                func.sum(
                    self._column("total_cost")
                ).label("total_cost"),
                func.count().label("count"),
            )
            .where(and_(*conditions))
            .group_by(month_expression)
            .order_by(month_expression.asc())
        )

        result = await self.session.execute(query)

        return [
            {
                "year": year,
                "month": int(month),
                "total_cost": Decimal(
                    str(total or 0)
                ),
                "estimate_count": int(count),
                "currency": currency,
            }
            for month, total, count in result.all()
        ]

    # ========================================================
    # COST DASHBOARD STATISTICS
    # ========================================================

    async def get_statistics(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        currency: str | None = None,
    ) -> dict[str, Any]:
        conditions = [
            self._scope(
                company_id=company_id,
                site_id=site_id,
            )
        ]

        if currency is not None:
            conditions.append(
                self._column("currency") == currency
            )
        else:
            currency_count = (
                await self.session.execute(
                    select(
                        func.count(
                            func.distinct(
                                self._column("currency")
                            )
                        )
                    )
                    .where(and_(*conditions))
                )
            ).scalar_one()

            if currency_count > 1:
                raise CostValidationError(
                    "Specify currency for cost statistics"
                )

        predicate = and_(*conditions)

        summary_query = (
            select(
                func.count(),
                func.sum(
                    self._column("total_cost")
                ),
                func.avg(
                    self._column("total_cost")
                ),
            )
            .where(predicate)
        )

        count, total, average = (
            await self.session.execute(
                summary_query
            )
        ).one()

        status_column = self._column("status")

        status_query = (
            select(
                status_column,
                func.count(),
            )
            .where(predicate)
            .group_by(status_column)
        )

        result = await self.session.execute(
            status_query
        )

        status_counts = {
            str(self._normalize(status)): int(value)
            for status, value in result.all()
        }

        pending = sum(
            value
            for status, value in status_counts.items()
            if status in self.PENDING_STATUSES
        )

        approved = sum(
            value
            for status, value in status_counts.items()
            if status in self.APPROVED_STATUSES
        )

        return {
            "company_id": company_id,
            "site_id": site_id,
            "currency": currency,
            "total_estimates": int(count),
            "total_estimated_cost": Decimal(
                str(total or 0)
            ),
            "average_estimated_cost": Decimal(
                str(average or 0)
            ),
            "pending_approvals": pending,
            "approved_estimates": approved,
            "status_counts": status_counts,
            "generated_at": utc_now(),
        }


__all__ = [
    "CostRepository",
    "CostRepositoryError",
    "CostNotFoundError",
    "CostValidationError",
]
