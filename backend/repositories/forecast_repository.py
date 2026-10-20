
"""
Industrial Forecast Repository
==============================

File: app/repositories/forecast_repository.py

Python: 3.11+
SQLAlchemy: 2.x

Features:
- Create and retrieve forecasts
- Multi-tenant company/site isolation
- Sensor, asset and machine forecast history
- Multiple forecast categories
- Forecast horizon and frequency tracking
- Model/version metadata
- Prediction intervals and confidence
- Forecast status and review workflows
- Forecast accuracy metrics
- Forecast trend aggregation
- Paginated queries
- Soft deletion
- Dashboard statistics

Expected model:
    app.models.forecast.Forecast

Required columns:
    id
    company_id
    site_id
    asset_id
    forecast_type
    status
    created_at

Optional columns:
    machine_id
    sensor_id
    metric_name
    forecast_frequency
    forecast_horizon
    forecast_start_at
    forecast_end_at
    predicted_value
    lower_bound
    upper_bound
    confidence_level
    model_name
    model_version
    model_type
    actual_value
    absolute_error
    squared_error
    percentage_error
    reviewed_by
    reviewed_at
    review_notes
    metadata_json
    updated_at
    deleted_at

The repository never commits transactions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Any, Mapping, Sequence
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.inspection import inspect

from app.models.forecast import Forecast


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================
# EXCEPTIONS
# ============================================================

class ForecastRepositoryError(Exception):
    """Base forecast repository exception."""


class ForecastNotFoundError(ForecastRepositoryError):
    """Forecast was not found."""


class ForecastValidationError(ForecastRepositoryError):
    """Invalid forecast input."""


# ============================================================
# FORECAST REPOSITORY
# ============================================================

class ForecastRepository:
    """
    Async SQLAlchemy forecast persistence.

    All reads and writes require company_id.

    This repository does not perform forecasting.
    Forecast generation belongs in the ML/service layer.

    The service layer owns commit/rollback.
    """

    CREATE_FIELDS = frozenset({
        "id",
        "site_id",
        "asset_id",
        "machine_id",
        "sensor_id",
        "forecast_type",
        "status",
        "metric_name",
        "forecast_frequency",
        "forecast_horizon",
        "forecast_start_at",
        "forecast_end_at",
        "predicted_value",
        "lower_bound",
        "upper_bound",
        "confidence_level",
        "model_name",
        "model_version",
        "model_type",
        "metadata_json",
    })

    UPDATE_FIELDS = frozenset({
        "metric_name",
        "forecast_frequency",
        "forecast_horizon",
        "forecast_start_at",
        "forecast_end_at",
        "predicted_value",
        "lower_bound",
        "upper_bound",
        "confidence_level",
        "metadata_json",
    })

    SORT_FIELDS = frozenset({
        "created_at",
        "updated_at",
        "forecast_start_at",
        "forecast_end_at",
        "predicted_value",
        "confidence_level",
        "status",
    })

    FORECAST_TYPES = frozenset({
        "sensor",
        "asset_health",
        "failure_risk",
        "rul",
        "maintenance_demand",
        "downtime",
        "energy",
        "production",
        "cost",
        "load",
        "temperature",
        "vibration",
        "pressure",
    })

    ACTIVE_STATUSES = frozenset({
        "pending",
        "running",
        "completed",
    })

    TERMINAL_STATUSES = frozenset({
        "completed",
        "failed",
        "cancelled",
        "expired",
    })

    def __init__(self, session: AsyncSession):
        self.session = session

        self.columns = {
            column.key
            for column in inspect(Forecast).mapper.columns
        }

        required = {
            "id",
            "company_id",
            "site_id",
            "asset_id",
            "forecast_type",
            "status",
            "created_at",
        }

        missing = required - self.columns

        if missing:
            raise ForecastRepositoryError(
                "Forecast model missing columns: "
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
            raise ForecastValidationError(
                f"{name} must be a string"
            )

        value = value.strip()

        if not value:
            raise ForecastValidationError(
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
            raise ForecastValidationError(
                f"{name} must be a datetime"
            )

        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ForecastValidationError(
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
            raise ForecastValidationError(
                "Start time cannot exceed end time"
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
            raise ForecastValidationError(
                f"{name} must be numeric"
            )

        number = float(value)

        if not isfinite(number):
            raise ForecastValidationError(
                f"{name} must be finite"
            )

        if minimum is not None and number < minimum:
            raise ForecastValidationError(
                f"{name} must be >= {minimum}"
            )

        if maximum is not None and number > maximum:
            raise ForecastValidationError(
                f"{name} must be <= {maximum}"
            )

        return number

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
            raise ForecastValidationError(
                f"limit must be between 1 and {maximum}"
            )

    def _column(self, name: str):
        if name not in self.columns:
            raise ForecastValidationError(
                f"Forecast model has no column '{name}'"
            )

        return getattr(Forecast, name)

    def _validate_payload(
        self,
        data: Mapping[str, Any],
        *,
        allowed: frozenset[str],
    ) -> dict[str, Any]:
        if not isinstance(data, Mapping):
            raise ForecastValidationError(
                "Forecast data must be a mapping"
            )

        payload = {
            key: self._normalize(value)
            for key, value in data.items()
        }

        unsupported = set(payload) - allowed

        if unsupported:
            raise ForecastValidationError(
                f"Unsupported fields: {sorted(unsupported)}"
            )

        missing_columns = set(payload) - self.columns

        if missing_columns:
            raise ForecastValidationError(
                "Fields missing from Forecast model: "
                f"{sorted(missing_columns)}"
            )

        for name in (
            "predicted_value",
            "lower_bound",
            "upper_bound",
            "actual_value",
        ):
            if name in payload and payload[name] is not None:
                payload[name] = self._validate_number(
                    payload[name],
                    name,
                )

        if (
            "confidence_level" in payload
            and payload["confidence_level"] is not None
        ):
            payload["confidence_level"] = (
                self._validate_number(
                    payload["confidence_level"],
                    "confidence_level",
                    minimum=0,
                    maximum=1,
                )
            )

        if (
            "forecast_horizon" in payload
            and payload["forecast_horizon"] is not None
        ):
            horizon = payload["forecast_horizon"]

            if (
                isinstance(horizon, bool)
                or not isinstance(horizon, int)
                or horizon < 1
            ):
                raise ForecastValidationError(
                    "forecast_horizon must be a "
                    "positive integer"
                )

        for name in (
            "forecast_start_at",
            "forecast_end_at",
        ):
            if name in payload:
                self._validate_datetime(
                    payload[name],
                    name,
                )

        start = payload.get("forecast_start_at")
        end = payload.get("forecast_end_at")

        self._validate_period(start, end)

        lower = payload.get("lower_bound")
        upper = payload.get("upper_bound")

        if (
            lower is not None
            and upper is not None
            and lower > upper
        ):
            raise ForecastValidationError(
                "lower_bound cannot exceed upper_bound"
            )

        return payload

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

        for name, value in {
            "site_id": site_id,
            "asset_id": asset_id,
            "machine_id": machine_id,
            "sensor_id": sensor_id,
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
    # CREATE FORECAST
    # ========================================================

    async def create(
        self,
        *,
        company_id: str,
        data: Mapping[str, Any],
    ) -> Forecast:
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
            "forecast_type",
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
            "created_at",
            utc_now(),
        )

        if "updated_at" in self.columns:
            payload.setdefault(
                "updated_at",
                utc_now(),
            )

        forecast = Forecast(**payload)

        self.session.add(forecast)

        await self.session.flush()

        return forecast

    # ========================================================
    # BULK CREATE
    # ========================================================

    async def create_many(
        self,
        *,
        company_id: str,
        records: Sequence[Mapping[str, Any]],
    ) -> list[Forecast]:
        if len(records) > 1000:
            raise ForecastValidationError(
                "Maximum 1000 forecasts per batch"
            )

        created: list[Forecast] = []

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
        forecast_id: UUID | str,
        *,
        company_id: str,
        include_deleted: bool = False,
    ) -> Forecast | None:
        query = (
            select(Forecast)
            .where(
                self._column("id") == forecast_id,
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
        forecast_id: UUID | str,
        *,
        company_id: str,
    ) -> Forecast:
        forecast = await self.get_by_id(
            forecast_id,
            company_id=company_id,
        )

        if forecast is None:
            raise ForecastNotFoundError(
                f"Forecast '{forecast_id}' not found"
            )

        return forecast

    # ========================================================
    # UPDATE FORECAST
    # ========================================================

    async def update(
        self,
        forecast_id: UUID | str,
        *,
        company_id: str,
        data: Mapping[str, Any],
    ) -> Forecast:
        forecast = await self.get_required(
            forecast_id,
            company_id=company_id,
        )

        payload = self._validate_payload(
            data,
            allowed=self.UPDATE_FIELDS,
        )

        for key, value in payload.items():
            setattr(forecast, key, value)

        if "updated_at" in self.columns:
            forecast.updated_at = utc_now()

        await self.session.flush()

        return forecast

    # ========================================================
    # UPDATE STATUS
    # ========================================================

    async def update_status(
        self,
        forecast_id: UUID | str,
        *,
        company_id: str,
        status: str,
    ) -> Forecast:
        forecast = await self.get_required(
            forecast_id,
            company_id=company_id,
        )

        status = self._require_text(
            status,
            "status",
        )

        forecast.status = status

        if "updated_at" in self.columns:
            forecast.updated_at = utc_now()

        await self.session.flush()

        return forecast

    # ========================================================
    # RECORD ACTUAL VALUE / EVALUATION
    # ========================================================

    async def record_actual(
        self,
        forecast_id: UUID | str,
        *,
        company_id: str,
        actual_value: float,
    ) -> Forecast:
        actual = self._validate_number(
            actual_value,
            "actual_value",
        )

        forecast = await self.get_required(
            forecast_id,
            company_id=company_id,
        )

        self._column("actual_value")
        self._column("predicted_value")

        predicted = forecast.predicted_value

        if predicted is None:
            raise ForecastValidationError(
                "Cannot evaluate a forecast without "
                "predicted_value"
            )

        predicted = self._validate_number(
            predicted,
            "predicted_value",
        )

        error = actual - predicted

        forecast.actual_value = actual

        if "absolute_error" in self.columns:
            forecast.absolute_error = abs(error)

        if "squared_error" in self.columns:
            forecast.squared_error = error ** 2

        if "percentage_error" in self.columns:
            forecast.percentage_error = (
                abs(error / actual) * 100
                if actual != 0
                else None
            )

        if "updated_at" in self.columns:
            forecast.updated_at = utc_now()

        await self.session.flush()

        return forecast

    # ========================================================
    # REVIEW FORECAST
    # ========================================================

    async def review(
        self,
        forecast_id: UUID | str,
        *,
        company_id: str,
        reviewed_by: str,
        notes: str | None = None,
    ) -> Forecast:
        reviewed_by = self._require_text(
            reviewed_by,
            "reviewed_by",
        )

        forecast = await self.get_required(
            forecast_id,
            company_id=company_id,
        )

        if "reviewed_by" in self.columns:
            forecast.reviewed_by = reviewed_by

        if "reviewed_at" in self.columns:
            forecast.reviewed_at = utc_now()

        if (
            notes is not None
            and "review_notes" in self.columns
        ):
            forecast.review_notes = notes

        if "updated_at" in self.columns:
            forecast.updated_at = utc_now()

        await self.session.flush()

        return forecast

    # ========================================================
    # SOFT DELETE
    # ========================================================

    async def soft_delete(
        self,
        forecast_id: UUID | str,
        *,
        company_id: str,
    ) -> bool:
        if "deleted_at" not in self.columns:
            raise ForecastRepositoryError(
                "Soft deletion requires deleted_at"
            )

        forecast = await self.get_by_id(
            forecast_id,
            company_id=company_id,
        )

        if forecast is None:
            return False

        forecast.deleted_at = utc_now()

        if "updated_at" in self.columns:
            forecast.updated_at = utc_now()

        await self.session.flush()

        return True

    # ========================================================
    # LIST FORECASTS
    # ========================================================

    async def list(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        asset_id: str | None = None,
        machine_id: str | None = None,
        sensor_id: str | None = None,
        forecast_type: str | None = None,
        status: str | None = None,
        metric_name: str | None = None,
        model_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "created_at",
        descending: bool = True,
    ) -> tuple[list[Forecast], int]:
        self._validate_period(
            start_time,
            end_time,
        )

        if page < 1:
            raise ForecastValidationError(
                "page must be at least 1"
            )

        self._validate_limit(
            page_size,
            maximum=100,
        )

        if sort_by not in self.SORT_FIELDS:
            raise ForecastValidationError(
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
            "forecast_type": forecast_type,
            "status": status,
            "metric_name": metric_name,
            "model_name": model_name,
        }

        for field, value in filters.items():
            if value is not None:
                conditions.append(
                    self._column(field) == value
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

        total = (
            await self.session.execute(
                select(func.count())
                .select_from(Forecast)
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
            select(Forecast)
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
    # ASSET FORECAST HISTORY
    # ========================================================

    async def get_asset_history(
        self,
        *,
        company_id: str,
        asset_id: str,
        forecast_type: str | None = None,
        limit: int = 500,
    ) -> list[Forecast]:
        self._validate_limit(limit)

        conditions = [
            self._scope(
                company_id=company_id,
                asset_id=asset_id,
            )
        ]

        if forecast_type is not None:
            conditions.append(
                self._column("forecast_type")
                == forecast_type
            )

        query = (
            select(Forecast)
            .where(and_(*conditions))
            .order_by(
                self._column("created_at").desc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # ========================================================
    # LATEST FORECAST
    # ========================================================

    async def get_latest(
        self,
        *,
        company_id: str,
        asset_id: str,
        forecast_type: str,
        metric_name: str | None = None,
    ) -> Forecast | None:
        conditions = [
            self._scope(
                company_id=company_id,
                asset_id=asset_id,
            ),
            self._column("forecast_type")
            == forecast_type,
        ]

        if metric_name is not None:
            conditions.append(
                self._column("metric_name")
                == metric_name
            )

        query = (
            select(Forecast)
            .where(and_(*conditions))
            .order_by(
                self._column("created_at").desc(),
                self._column("id").desc(),
            )
            .limit(1)
        )

        result = await self.session.execute(query)

        return result.scalar_one_or_none()

    # ========================================================
    # MODEL FORECAST HISTORY
    # ========================================================

    async def get_model_history(
        self,
        *,
        company_id: str,
        model_name: str,
        model_version: str | None = None,
        limit: int = 500,
    ) -> list[Forecast]:
        self._validate_limit(limit)

        conditions = [
            self._scope(company_id=company_id),
            self._column("model_name") == model_name,
        ]

        if model_version is not None:
            conditions.append(
                self._column("model_version")
                == model_version
            )

        query = (
            select(Forecast)
            .where(and_(*conditions))
            .order_by(
                self._column("created_at").desc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # ========================================================
    # FORECAST ACCURACY
    # ========================================================

    async def get_accuracy_metrics(
        self,
        *,
        company_id: str,
        forecast_type: str | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> dict[str, Any]:
        required = {
            "actual_value",
            "predicted_value",
            "absolute_error",
            "squared_error",
        }

        missing = required - self.columns

        if missing:
            raise ForecastRepositoryError(
                "Forecast evaluation columns missing: "
                f"{sorted(missing)}"
            )

        conditions = [
            self._scope(company_id=company_id),
            self._column("actual_value").is_not(None),
            self._column("predicted_value").is_not(None),
        ]

        for field, value in {
            "forecast_type": forecast_type,
            "model_name": model_name,
            "model_version": model_version,
        }.items():
            if value is not None:
                conditions.append(
                    self._column(field) == value
                )

        query = (
            select(
                func.count(),
                func.avg(
                    self._column("absolute_error")
                ),
                func.sqrt(
                    func.avg(
                        self._column("squared_error")
                    )
                ),
            )
            .where(and_(*conditions))
        )

        count, mae, rmse = (
            await self.session.execute(query)
        ).one()

        return {
            "company_id": company_id,
            "forecast_type": forecast_type,
            "model_name": model_name,
            "model_version": model_version,
            "evaluated_forecasts": int(count),
            "mae": (
                float(mae)
                if mae is not None
                else None
            ),
            "rmse": (
                float(rmse)
                if rmse is not None
                else None
            ),
            "generated_at": utc_now(),
        }

    # ========================================================
    # DASHBOARD STATISTICS
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

        total = (
            await self.session.execute(
                select(func.count())
                .select_from(Forecast)
                .where(predicate)
            )
        ).scalar_one()

        status_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}

        for field, destination in (
            ("status", status_counts),
            ("forecast_type", type_counts),
        ):
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

        return {
            "company_id": company_id,
            "site_id": site_id,
            "total_forecasts": int(total),
            "completed_forecasts": (
                status_counts.get("completed", 0)
            ),
            "failed_forecasts": (
                status_counts.get("failed", 0)
            ),
            "pending_forecasts": (
                status_counts.get("pending", 0)
            ),
            "status_counts": status_counts,
            "forecast_type_counts": type_counts,
            "generated_at": utc_now(),
        }


__all__ = [
    "ForecastRepository",
    "ForecastRepositoryError",
    "ForecastNotFoundError",
    "ForecastValidationError",
]
