
"""
app/services/asset_service.py

Industrial Predictive Maintenance, Safety & Emergency AI

Asset Management Service

Features:
- Asset registration
- Asset retrieval and listing
- Asset information updates
- Soft deletion
- Asset status management
- Machine health score management
- Maintenance scheduling
- Asset statistics
- Asset risk classification
- Repository integration
- Structured logging
- UTC timestamps
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class AssetStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    FAILED = "failed"
    RETIRED = "retired"


class AssetRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AssetServiceError(Exception):
    pass


class AssetNotFoundError(AssetServiceError):
    pass


class AssetValidationError(AssetServiceError):
    pass


class AssetConflictError(AssetServiceError):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_uuid(value: str | UUID) -> UUID:
    try:
        return UUID(str(value))
    except (ValueError, TypeError, AttributeError) as exc:
        raise AssetValidationError(
            "Invalid asset UUID"
        ) from exc


class AssetService:
    """
    Asset application service.

    Expected repository contract:

    async create(data: dict)
    async get_by_id(asset_id: UUID)
    async get_by_code(asset_code: str)
    async list_assets(
        skip: int,
        limit: int,
        filters: dict
    )
    async update(asset_id: UUID, data: dict)
    async count(filters: dict)

    Repository methods should enforce organization
    isolation and use parameterized database queries.
    """

    def __init__(
        self,
        asset_repository: Any,
    ):
        if asset_repository is None:
            raise ValueError(
                "asset_repository is required"
            )

        self.repository = asset_repository

    @staticmethod
    def _validate_text(
        value: Any,
        field_name: str,
        max_length: int = 255,
    ) -> str:
        if not isinstance(value, str):
            raise AssetValidationError(
                f"{field_name} must be a string"
            )

        cleaned = value.strip()

        if not cleaned:
            raise AssetValidationError(
                f"{field_name} cannot be empty"
            )

        if len(cleaned) > max_length:
            raise AssetValidationError(
                f"{field_name} exceeds {max_length} characters"
            )

        return cleaned

    @staticmethod
    def _validate_health_score(
        score: float,
    ) -> float:
        try:
            if isinstance(score, bool):
                raise ValueError

            value = float(score)
        except (TypeError, ValueError) as exc:
            raise AssetValidationError(
                "Health score must be numeric"
            ) from exc

        import math

        if not math.isfinite(value):
            raise AssetValidationError(
                "Health score must be finite"
            )

        if not 0 <= value <= 100:
            raise AssetValidationError(
                "Health score must be between 0 and 100"
            )

        return round(value, 2)

    @staticmethod
    def calculate_risk_level(
        health_score: float,
    ) -> AssetRiskLevel:
        """
        Demonstration classification.
        Calibrate against actual equipment risks.
        """

        if health_score < 25:
            return AssetRiskLevel.CRITICAL

        if health_score < 50:
            return AssetRiskLevel.HIGH

        if health_score < 75:
            return AssetRiskLevel.MEDIUM

        return AssetRiskLevel.LOW

    async def create_asset(
        self,
        *,
        name: str,
        asset_code: str,
        company_id: str | UUID,
        site_id: str | UUID,
        asset_type: str,
        manufacturer: Optional[str] = None,
        model_number: Optional[str] = None,
        serial_number: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Any:

        name = self._validate_text(name, "name")
        asset_code = self._validate_text(
            asset_code, "asset_code", 100
        )
        asset_type = self._validate_text(
            asset_type, "asset_type", 100
        )

        company_uuid = validate_uuid(company_id)
        site_uuid = validate_uuid(site_id)

        existing = await self.repository.get_by_code(
            asset_code
        )

        if existing is not None:
            raise AssetConflictError(
                f"Asset code '{asset_code}' already exists"
            )

        now = utc_now()

        data = {
            "id": uuid4(),
            "name": name,
            "asset_code": asset_code,
            "company_id": company_uuid,
            "site_id": site_uuid,
            "asset_type": asset_type,
            "manufacturer": manufacturer,
            "model_number": model_number,
            "serial_number": serial_number,
            "description": description,
            "asset_metadata": dict(metadata or {}),
            "status": AssetStatus.ACTIVE.value,
            "health_score": 100.0,
            "risk_level": AssetRiskLevel.LOW.value,
            "last_maintenance_at": None,
            "next_maintenance_at": None,
            "is_deleted": False,
            "created_at": now,
            "updated_at": now,
        }

        asset = await self.repository.create(data)

        logger.info(
            "Asset created: id=%s code=%s",
            data["id"],
            asset_code,
        )

        return asset

    async def get_asset(
        self,
        asset_id: str | UUID,
    ) -> Any:

        asset_uuid = validate_uuid(asset_id)

        asset = await self.repository.get_by_id(
            asset_uuid
        )

        if asset is None or getattr(
            asset, "is_deleted", False
        ):
            raise AssetNotFoundError(
                f"Asset {asset_uuid} not found"
            )

        return asset

    async def list_assets(
        self,
        *,
        skip: int = 0,
        limit: int = 50,
        company_id: Optional[str | UUID] = None,
        site_id: Optional[str | UUID] = None,
        status: Optional[str] = None,
        asset_type: Optional[str] = None,
    ) -> Any:

        if skip < 0:
            raise AssetValidationError(
                "skip cannot be negative"
            )

        if not 1 <= limit <= 500:
            raise AssetValidationError(
                "limit must be between 1 and 500"
            )

        filters: dict[str, Any] = {
            "is_deleted": False,
        }

        if company_id is not None:
            filters["company_id"] = validate_uuid(
                company_id
            )

        if site_id is not None:
            filters["site_id"] = validate_uuid(
                site_id
            )

        if status is not None:
            try:
                filters["status"] = AssetStatus(
                    status
                ).value
            except ValueError as exc:
                raise AssetValidationError(
                    "Invalid asset status"
                ) from exc

        if asset_type is not None:
            filters["asset_type"] = asset_type

        return await self.repository.list_assets(
            skip=skip,
            limit=limit,
            filters=filters,
        )

    async def update_asset(
        self,
        asset_id: str | UUID,
        updates: dict[str, Any],
    ) -> Any:

        asset = await self.get_asset(asset_id)

        allowed_fields = {
            "name",
            "asset_type",
            "manufacturer",
            "model_number",
            "serial_number",
            "description",
            "asset_metadata",
        }

        unknown_fields = (
            set(updates) - allowed_fields
        )

        if unknown_fields:
            raise AssetValidationError(
                "Unsupported update fields: "
                + ", ".join(sorted(unknown_fields))
            )

        data = {
            key: value
            for key, value in updates.items()
            if key in allowed_fields
        }

        if "name" in data:
            data["name"] = self._validate_text(
                data["name"], "name"
            )

        if "asset_type" in data:
            data["asset_type"] = self._validate_text(
                data["asset_type"], "asset_type"
            )

        if "asset_metadata" in data:
            if not isinstance(
                data["asset_metadata"], dict
            ):
                raise AssetValidationError(
                    "asset_metadata must be an object"
                )

        if not data:
            return asset

        data["updated_at"] = utc_now()

        updated = await self.repository.update(
            validate_uuid(asset_id),
            data,
        )

        logger.info(
            "Asset updated: %s",
            asset_id,
        )

        return updated

    async def update_status(
        self,
        asset_id: str | UUID,
        status: str | AssetStatus,
    ) -> Any:

        await self.get_asset(asset_id)

        try:
            validated_status = AssetStatus(status)
        except ValueError as exc:
            raise AssetValidationError(
                f"Invalid asset status: {status}"
            ) from exc

        updated = await self.repository.update(
            validate_uuid(asset_id),
            {
                "status": validated_status.value,
                "updated_at": utc_now(),
            },
        )

        logger.info(
            "Asset %s status changed to %s",
            asset_id,
            validated_status.value,
        )

        return updated

    async def update_health_score(
        self,
        asset_id: str | UUID,
        health_score: float,
    ) -> Any:

        await self.get_asset(asset_id)

        score = self._validate_health_score(
            health_score
        )

        risk_level = self.calculate_risk_level(
            score
        )

        updated = await self.repository.update(
            validate_uuid(asset_id),
            {
                "health_score": score,
                "risk_level": risk_level.value,
                "updated_at": utc_now(),
            },
        )

        logger.info(
            "Asset %s health=%.2f risk=%s",
            asset_id,
            score,
            risk_level.value,
        )

        return updated

    async def schedule_maintenance(
        self,
        asset_id: str | UUID,
        scheduled_at: datetime,
    ) -> Any:

        await self.get_asset(asset_id)

        if not isinstance(scheduled_at, datetime):
            raise AssetValidationError(
                "scheduled_at must be datetime"
            )

        if (
            scheduled_at.tzinfo is None
            or scheduled_at.utcoffset() is None
        ):
            raise AssetValidationError(
                "scheduled_at must be timezone-aware"
            )

        scheduled_utc = scheduled_at.astimezone(
            timezone.utc
        )

        if scheduled_utc <= utc_now():
            raise AssetValidationError(
                "Maintenance must be scheduled "
                "in the future"
            )

        updated = await self.repository.update(
            validate_uuid(asset_id),
            {
                "next_maintenance_at": scheduled_utc,
                "updated_at": utc_now(),
            },
        )

        logger.info(
            "Maintenance scheduled for asset %s",
            asset_id,
        )

        return updated

    async def complete_maintenance(
        self,
        asset_id: str | UUID,
    ) -> Any:

        await self.get_asset(asset_id)

        now = utc_now()

        updated = await self.repository.update(
            validate_uuid(asset_id),
            {
                "last_maintenance_at": now,
                "next_maintenance_at": None,
                "updated_at": now,
            },
        )

        logger.info(
            "Maintenance completed: asset=%s",
            asset_id,
        )

        return updated

    async def delete_asset(
        self,
        asset_id: str | UUID,
    ) -> Any:
        """
        Soft-delete an asset to preserve
        historical maintenance and telemetry data.
        """

        await self.get_asset(asset_id)

        updated = await self.repository.update(
            validate_uuid(asset_id),
            {
                "is_deleted": True,
                "status": AssetStatus.RETIRED.value,
                "updated_at": utc_now(),
            },
        )

        logger.info(
            "Asset soft-deleted: %s",
            asset_id,
        )

        return updated

    async def get_asset_statistics(
        self,
        *,
        company_id: Optional[str | UUID] = None,
        site_id: Optional[str | UUID] = None,
    ) -> dict[str, Any]:

        filters: dict[str, Any] = {
            "is_deleted": False,
        }

        if company_id is not None:
            filters["company_id"] = validate_uuid(
                company_id
            )

        if site_id is not None:
            filters["site_id"] = validate_uuid(
                site_id
            )

        total = await self.repository.count(filters)

        counts = {}

        for status in AssetStatus:
            status_filters = {
                **filters,
                "status": status.value,
            }

            counts[status.value] = (
                await self.repository.count(
                    status_filters
                )
            )

        return {
            "total_assets": total,
            "status_counts": counts,
            "active_assets": counts["active"],
            "maintenance_assets": counts[
                "maintenance"
            ],
            "failed_assets": counts["failed"],
            "retired_assets": counts["retired"],
            "generated_at": utc_now().isoformat(),
        }
