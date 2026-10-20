
"""
Industrial Asset Repository
===========================

File: app/repositories/asset_repository.py

Python: 3.11+
SQLAlchemy: 2.x

Features:
- Asset creation and registration
- Asset retrieval by ID and code
- Multi-tenant company/site isolation
- Asset updates and status changes
- Asset health and condition updates
- Lifecycle and criticality management
- Asset financial information updates
- Maintenance-related asset queries
- Risk-based asset filtering
- Pagination and search
- Soft deletion
- Asset statistics and dashboard queries

Dependencies:
    pip install "sqlalchemy[asyncio]>=2.0"

Expected model:
    app.models.asset.Asset

Required columns:
    id
    company_id
    site_id
    asset_code
    asset_name

Common optional columns:
    asset_type
    status
    condition
    criticality
    risk_level
    health_score
    machine_id
    production_line_id
    manufacturer
    model_number
    serial_number
    installation_date
    commissioning_date
    last_maintenance_at
    next_maintenance_at
    purchase_cost
    replacement_cost
    current_value
    metadata_json
    created_at
    updated_at
    deleted_at

The service layer owns transaction commits.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from math import isfinite
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

from app.models.asset import Asset


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================
# EXCEPTIONS
# ============================================================

class AssetRepositoryError(Exception):
    """Base asset repository exception."""


class AssetNotFoundError(AssetRepositoryError):
    """Asset does not exist in the requested tenant."""


class AssetValidationError(AssetRepositoryError):
    """Invalid asset repository input."""


class AssetConflictError(AssetRepositoryError):
    """Asset identity or uniqueness conflict."""


# ============================================================
# ASSET REPOSITORY
# ============================================================

class AssetRepository:
    """
    Async SQLAlchemy asset repository.

    Security:
    - Every query is scoped by company_id.
    - Site filters are supported.
    - Tenant ownership cannot be changed by update().
    - Soft-deleted records are excluded by default.

    Transactions:
    - add(), flush(), refresh()
    - No automatic commit or rollback.
    """

    CREATE_FIELDS = frozenset({
        "id",
        "site_id",
        "asset_code",
        "asset_name",
        "asset_type",
        "status",
        "condition",
        "criticality",
        "risk_level",
        "health_score",
        "machine_id",
        "production_line_id",
        "manufacturer",
        "model_number",
        "serial_number",
        "installation_date",
        "commissioning_date",
        "purchase_cost",
        "replacement_cost",
        "current_value",
        "metadata_json",
    })

    UPDATE_FIELDS = frozenset({
        "asset_name",
        "asset_type",
        "condition",
        "criticality",
        "risk_level",
        "machine_id",
        "production_line_id",
        "manufacturer",
        "model_number",
        "serial_number",
        "installation_date",
        "commissioning_date",
        "purchase_cost",
        "replacement_cost",
        "current_value",
        "metadata_json",
    })

    FINANCIAL_FIELDS = frozenset({
        "purchase_cost",
        "replacement_cost",
        "current_value",
        "book_value",
        "market_value",
        "salvage_value",
        "maintenance_cost",
        "operating_cost",
        "downtime_cost_per_hour",
    })

    SORT_FIELDS = frozenset({
        "asset_name",
        "asset_code",
        "created_at",
        "updated_at",
        "health_score",
        "criticality",
        "status",
    })

    ACTIVE_STATUSES = frozenset({
        "active",
        "operational",
        "running",
        "standby",
    })

    HIGH_RISK_LEVELS = frozenset({
        "high",
        "critical",
        "extreme",
    })

    def __init__(self, session: AsyncSession):
        self.session = session

        self.columns = {
            column.key
            for column in inspect(Asset).mapper.columns
        }

        required = {
            "id",
            "company_id",
            "site_id",
            "asset_code",
            "asset_name",
        }

        missing = required - self.columns

        if missing:
            raise AssetRepositoryError(
                "Asset model is missing required "
                f"columns: {sorted(missing)}"
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
        field_name: str,
    ) -> str:
        if not isinstance(value, str):
            raise AssetValidationError(
                f"{field_name} must be a string"
            )

        value = value.strip()

        if not value:
            raise AssetValidationError(
                f"{field_name} cannot be empty"
            )

        return value

    def _column(self, name: str):
        if name not in self.columns:
            raise AssetValidationError(
                f"Asset model has no column '{name}'"
            )

        return getattr(Asset, name)

    def _validate_payload(
        self,
        data: Mapping[str, Any],
        *,
        allowed: frozenset[str],
    ) -> dict[str, Any]:
        if not isinstance(data, Mapping):
            raise AssetValidationError(
                "Asset data must be a mapping"
            )

        payload = {
            key: self._normalize(value)
            for key, value in data.items()
        }

        unsupported = set(payload) - allowed

        if unsupported:
            raise AssetValidationError(
                f"Unsupported asset fields: "
                f"{sorted(unsupported)}"
            )

        missing_columns = set(payload) - self.columns

        if missing_columns:
            raise AssetValidationError(
                "Fields missing from Asset model: "
                f"{sorted(missing_columns)}"
            )

        return payload

    def _scope(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        asset_id: UUID | str | None = None,
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
                self._column("id") == asset_id
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
    def _validate_pagination(
        page: int,
        page_size: int,
    ) -> None:
        if page < 1:
            raise AssetValidationError(
                "page must be at least 1"
            )

        if not 1 <= page_size <= 100:
            raise AssetValidationError(
                "page_size must be between 1 and 100"
            )

    @staticmethod
    def _validate_datetime(
        value: datetime | None,
        field_name: str,
    ) -> None:
        if value is None:
            return

        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise AssetValidationError(
                f"{field_name} must be timezone-aware"
            )

    # ========================================================
    # CREATE ASSET
    # ========================================================

    async def create(
        self,
        *,
        company_id: str,
        data: Mapping[str, Any],
    ) -> Asset:
        company_id = self._require_text(
            company_id,
            "company_id",
        )

        payload = self._validate_payload(
            data,
            allowed=self.CREATE_FIELDS,
        )

        for field in (
            "site_id",
            "asset_code",
            "asset_name",
        ):
            self._require_text(
                payload.get(field),
                field,
            )

        existing = await self.get_by_code(
            payload["asset_code"],
            company_id=company_id,
            site_id=payload["site_id"],
        )

        if existing is not None:
            raise AssetConflictError(
                "Asset code already exists in this site"
            )

        payload["company_id"] = company_id

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

        asset = Asset(**payload)

        self.session.add(asset)

        await self.session.flush()

        return asset

    # ========================================================
    # BULK CREATE
    # ========================================================

    async def create_many(
        self,
        *,
        company_id: str,
        records: Sequence[Mapping[str, Any]],
    ) -> list[Asset]:
        if len(records) > 1000:
            raise AssetValidationError(
                "Maximum 1000 assets per batch"
            )

        created: list[Asset] = []

        for record in records:
            asset = await self.create(
                company_id=company_id,
                data=record,
            )

            created.append(asset)

        return created

    # ========================================================
    # GET ASSET BY ID
    # ========================================================

    async def get_by_id(
        self,
        asset_id: UUID | str,
        *,
        company_id: str,
        site_id: str | None = None,
        include_deleted: bool = False,
    ) -> Asset | None:
        statement = (
            select(Asset)
            .where(
                self._scope(
                    company_id=company_id,
                    site_id=site_id,
                    asset_id=asset_id,
                    include_deleted=include_deleted,
                )
            )
        )

        result = await self.session.execute(
            statement
        )

        return result.scalar_one_or_none()

    async def get_required(
        self,
        asset_id: UUID | str,
        *,
        company_id: str,
        site_id: str | None = None,
    ) -> Asset:
        asset = await self.get_by_id(
            asset_id,
            company_id=company_id,
            site_id=site_id,
        )

        if asset is None:
            raise AssetNotFoundError(
                f"Asset '{asset_id}' was not found"
            )

        return asset

    # ========================================================
    # GET ASSET BY CODE
    # ========================================================

    async def get_by_code(
        self,
        asset_code: str,
        *,
        company_id: str,
        site_id: str | None = None,
    ) -> Asset | None:
        asset_code = self._require_text(
            asset_code,
            "asset_code",
        )

        statement = (
            select(Asset)
            .where(
                self._scope(
                    company_id=company_id,
                    site_id=site_id,
                ),
                self._column("asset_code")
                == asset_code,
            )
        )

        result = await self.session.execute(
            statement
        )

        return result.scalar_one_or_none()

    # ========================================================
    # UPDATE ASSET
    # ========================================================

    async def update(
        self,
        asset_id: UUID | str,
        *,
        company_id: str,
        data: Mapping[str, Any],
    ) -> Asset:
        asset = await self.get_required(
            asset_id,
            company_id=company_id,
        )

        payload = self._validate_payload(
            data,
            allowed=self.UPDATE_FIELDS,
        )

        for key, value in payload.items():
            setattr(asset, key, value)

        if "updated_at" in self.columns:
            asset.updated_at = utc_now()

        await self.session.flush()

        return asset

    # ========================================================
    # UPDATE ASSET STATUS
    # ========================================================

    async def update_status(
        self,
        asset_id: UUID | str,
        *,
        company_id: str,
        status: str,
    ) -> Asset:
        asset = await self.get_required(
            asset_id,
            company_id=company_id,
        )

        status = self._require_text(
            status,
            "status",
        )

        self._column("status")

        asset.status = status

        if "updated_at" in self.columns:
            asset.updated_at = utc_now()

        await self.session.flush()

        return asset

    # ========================================================
    # UPDATE ASSET HEALTH
    # ========================================================

    async def update_health(
        self,
        asset_id: UUID | str,
        *,
        company_id: str,
        health_score: float,
        condition: str | None = None,
        risk_level: str | None = None,
    ) -> Asset:
        if (
            not isfinite(health_score)
            or not 0 <= health_score <= 100
        ):
            raise AssetValidationError(
                "health_score must be between 0 and 100"
            )

        asset = await self.get_required(
            asset_id,
            company_id=company_id,
        )

        self._column("health_score")

        asset.health_score = health_score

        if condition is not None:
            self._column("condition")
            asset.condition = condition

        if risk_level is not None:
            self._column("risk_level")
            asset.risk_level = risk_level

        if "updated_at" in self.columns:
            asset.updated_at = utc_now()

        await self.session.flush()

        return asset

    # ========================================================
    # UPDATE FINANCIAL DATA
    # ========================================================

    async def update_financial_info(
        self,
        asset_id: UUID | str,
        *,
        company_id: str,
        financial_data: Mapping[str, Any],
    ) -> Asset:
        asset = await self.get_required(
            asset_id,
            company_id=company_id,
        )

        payload = self._validate_payload(
            financial_data,
            allowed=self.FINANCIAL_FIELDS,
        )

        for key, value in payload.items():
            if value is not None:
                amount = Decimal(str(value))

                if not amount.is_finite() or amount < 0:
                    raise AssetValidationError(
                        f"{key} must be non-negative "
                        "and finite"
                    )

                setattr(asset, key, amount)

        if "updated_at" in self.columns:
            asset.updated_at = utc_now()

        await self.session.flush()

        return asset

    # ========================================================
    # RECORD MAINTENANCE TIMESTAMPS
    # ========================================================

    async def update_maintenance_dates(
        self,
        asset_id: UUID | str,
        *,
        company_id: str,
        last_maintenance_at: datetime | None = None,
        next_maintenance_at: datetime | None = None,
    ) -> Asset:
        self._validate_datetime(
            last_maintenance_at,
            "last_maintenance_at",
        )

        self._validate_datetime(
            next_maintenance_at,
            "next_maintenance_at",
        )

        asset = await self.get_required(
            asset_id,
            company_id=company_id,
        )

        if last_maintenance_at is not None:
            self._column("last_maintenance_at")
            asset.last_maintenance_at = (
                last_maintenance_at
            )

        if next_maintenance_at is not None:
            self._column("next_maintenance_at")
            asset.next_maintenance_at = (
                next_maintenance_at
            )

        if "updated_at" in self.columns:
            asset.updated_at = utc_now()

        await self.session.flush()

        return asset

    # ========================================================
    # SOFT DELETE
    # ========================================================

    async def soft_delete(
        self,
        asset_id: UUID | str,
        *,
        company_id: str,
    ) -> bool:
        if "deleted_at" not in self.columns:
            raise AssetRepositoryError(
                "Soft deletion requires deleted_at "
                "on the Asset model"
            )

        asset = await self.get_by_id(
            asset_id,
            company_id=company_id,
        )

        if asset is None:
            return False

        asset.deleted_at = utc_now()

        if "updated_at" in self.columns:
            asset.updated_at = utc_now()

        await self.session.flush()

        return True

    # ========================================================
    # LIST AND FILTER ASSETS
    # ========================================================

    async def list(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        asset_type: str | None = None,
        status: str | None = None,
        condition: str | None = None,
        criticality: str | None = None,
        risk_level: str | None = None,
        machine_id: str | None = None,
        production_line_id: str | None = None,
        search: str | None = None,
        min_health_score: float | None = None,
        max_health_score: float | None = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "asset_name",
        descending: bool = False,
    ) -> tuple[list[Asset], int]:
        self._validate_pagination(
            page,
            page_size,
        )

        if sort_by not in self.SORT_FIELDS:
            raise AssetValidationError(
                f"Invalid sort field: {sort_by}"
            )

        for value, name in (
            (min_health_score, "min_health_score"),
            (max_health_score, "max_health_score"),
        ):
            if value is not None and (
                not isfinite(value)
                or not 0 <= value <= 100
            ):
                raise AssetValidationError(
                    f"{name} must be between 0 and 100"
                )

        if (
            min_health_score is not None
            and max_health_score is not None
            and min_health_score > max_health_score
        ):
            raise AssetValidationError(
                "Minimum health exceeds maximum health"
            )

        conditions = [
            self._scope(
                company_id=company_id,
                site_id=site_id,
            )
        ]

        filters = {
            "asset_type": asset_type,
            "status": status,
            "condition": condition,
            "criticality": criticality,
            "risk_level": risk_level,
            "machine_id": machine_id,
            "production_line_id": production_line_id,
        }

        for field, value in filters.items():
            if value is not None:
                conditions.append(
                    self._column(field) == value
                )

        if min_health_score is not None:
            conditions.append(
                self._column("health_score")
                >= min_health_score
            )

        if max_health_score is not None:
            conditions.append(
                self._column("health_score")
                <= max_health_score
            )

        if search:
            pattern = f"%{search.strip()}%"

            conditions.append(
                or_(
                    self._column("asset_name").ilike(
                        pattern
                    ),
                    self._column("asset_code").ilike(
                        pattern
                    ),
                )
            )

        predicate = and_(*conditions)

        count_query = (
            select(func.count())
            .select_from(Asset)
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
            select(Asset)
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
    # HIGH-RISK ASSETS
    # ========================================================

    async def get_high_risk_assets(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        limit: int = 100,
    ) -> list[Asset]:
        if not 1 <= limit <= 1000:
            raise AssetValidationError(
                "limit must be between 1 and 1000"
            )

        query = (
            select(Asset)
            .where(
                self._scope(
                    company_id=company_id,
                    site_id=site_id,
                ),
                self._column("risk_level").in_(
                    self.HIGH_RISK_LEVELS
                ),
            )
            .order_by(
                self._column("health_score").asc(),
                self._column("id").asc(),
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # ========================================================
    # LOW-HEALTH ASSETS
    # ========================================================

    async def get_low_health_assets(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        threshold: float = 40.0,
        limit: int = 100,
    ) -> list[Asset]:
        if (
            not isfinite(threshold)
            or not 0 <= threshold <= 100
        ):
            raise AssetValidationError(
                "threshold must be between 0 and 100"
            )

        if not 1 <= limit <= 1000:
            raise AssetValidationError(
                "limit must be between 1 and 1000"
            )

        query = (
            select(Asset)
            .where(
                self._scope(
                    company_id=company_id,
                    site_id=site_id,
                ),
                self._column("health_score") < threshold,
            )
            .order_by(
                self._column("health_score").asc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # ========================================================
    # ASSETS DUE FOR MAINTENANCE
    # ========================================================

    async def get_maintenance_due_assets(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        due_before: datetime | None = None,
        limit: int = 100,
    ) -> list[Asset]:
        if due_before is None:
            due_before = utc_now()

        self._validate_datetime(
            due_before,
            "due_before",
        )

        if not 1 <= limit <= 1000:
            raise AssetValidationError(
                "limit must be between 1 and 1000"
            )

        query = (
            select(Asset)
            .where(
                self._scope(
                    company_id=company_id,
                    site_id=site_id,
                ),
                self._column("next_maintenance_at")
                <= due_before,
            )
            .order_by(
                self._column("next_maintenance_at").asc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # ========================================================
    # ASSET STATISTICS
    # ========================================================

    async def get_statistics(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
    ) -> dict[str, Any]:
        predicate = self._scope(
            company_id=company_id,
            site_id=site_id,
        )

        total_query = (
            select(func.count())
            .select_from(Asset)
            .where(predicate)
        )

        total = (
            await self.session.execute(total_query)
        ).scalar_one()

        status_counts: dict[str, int] = {}
        risk_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}

        for field, destination in (
            ("status", status_counts),
            ("risk_level", risk_counts),
            ("asset_type", type_counts),
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

        average_health = None

        if "health_score" in self.columns:
            query = (
                select(
                    func.avg(
                        self._column("health_score")
                    )
                )
                .where(predicate)
            )

            value = (
                await self.session.execute(query)
            ).scalar_one()

            if value is not None:
                average_health = float(value)

        active_assets = sum(
            count
            for status, count in status_counts.items()
            if status in self.ACTIVE_STATUSES
        )

        high_risk_assets = sum(
            count
            for risk, count in risk_counts.items()
            if risk in self.HIGH_RISK_LEVELS
        )

        return {
            "company_id": company_id,
            "site_id": site_id,
            "total_assets": int(total),
            "active_assets": active_assets,
            "high_risk_assets": high_risk_assets,
            "average_health_score": average_health,
            "status_counts": status_counts,
            "risk_counts": risk_counts,
            "asset_type_counts": type_counts,
            "generated_at": utc_now(),
        }


__all__ = [
    "AssetRepository",
    "AssetRepositoryError",
    "AssetNotFoundError",
    "AssetValidationError",
    "AssetConflictError",
]
