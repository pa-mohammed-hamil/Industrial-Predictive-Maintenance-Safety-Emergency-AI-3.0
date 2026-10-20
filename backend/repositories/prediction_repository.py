
"""
app/repositories/prediction_repository.py

Industrial AI Prediction Repository
===================================

Python 3.11+
SQLAlchemy 2.x
AsyncSession
PostgreSQL compatible

Features:
- Create predictions
- Batch prediction storage
- Tenant-scoped retrieval
- Asset/machine prediction history
- Failure probability and RUL predictions
- Model metadata and confidence tracking
- Actual outcome recording
- Prediction error calculations
- Human review and verification
- Status management
- Prediction search and pagination
- High-risk prediction retrieval
- Model performance statistics
- Soft deletion

Expected model:
    app.models.prediction.Prediction

Required columns:
    id
    company_id
    site_id
    asset_id
    prediction_type
    status
    predicted_at

Recommended optional columns:
    machine_id
    sensor_id
    prediction_value
    prediction_unit
    failure_probability
    rul_hours
    health_score
    risk_level
    confidence_score
    lower_bound
    upper_bound
    prediction_horizon_hours
    model_name
    model_version
    model_type
    model_run_id
    input_reference
    actual_value
    absolute_error
    squared_error
    percentage_error
    verified_by
    verified_at
    reviewed_by
    reviewed_at
    review_notes
    metadata_json
    created_at
    updated_at
    deleted_at

Repository flushes but never commits.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from math import isfinite, sqrt
from typing import Any, Mapping, Sequence
from uuid import UUID

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.inspection import inspect

from app.models.prediction import Prediction


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PredictionRepositoryError(Exception):
    """Base repository exception."""


class PredictionNotFoundError(PredictionRepositoryError):
    """Prediction does not exist within tenant scope."""


class PredictionValidationError(PredictionRepositoryError):
    """Invalid prediction input."""


class PredictionConflictError(PredictionRepositoryError):
    """Prediction state conflict."""


class PredictionRepository:
    """
    Tenant-scoped asynchronous repository for AI predictions.

    Model inference must occur in the ML/DL service layer.

    This repository persists model outputs and evaluation data.
    """

    CREATE_FIELDS = frozenset({
        "site_id",
        "asset_id",
        "machine_id",
        "sensor_id",
        "prediction_type",
        "status",
        "prediction_value",
        "prediction_unit",
        "failure_probability",
        "rul_hours",
        "health_score",
        "risk_level",
        "confidence_score",
        "lower_bound",
        "upper_bound",
        "prediction_horizon_hours",
        "model_name",
        "model_version",
        "model_type",
        "model_run_id",
        "input_reference",
        "predicted_at",
        "metadata_json",
    })

    UPDATE_FIELDS = frozenset({
        "risk_level",
        "metadata_json",
    })

    SORT_FIELDS = frozenset({
        "predicted_at",
        "created_at",
        "updated_at",
        "failure_probability",
        "rul_hours",
        "health_score",
        "confidence_score",
        "status",
    })

    HIGH_RISK_LEVELS = frozenset({
        "high",
        "critical",
        "extreme",
    })

    ACTIVE_STATUSES = frozenset({
        "pending",
        "generated",
        "review_required",
        "reviewed",
        "verified",
    })

    TERMINAL_STATUSES = frozenset({
        "rejected",
        "expired",
        "invalid",
    })

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

        self.columns = {
            column.key
            for column in inspect(Prediction).mapper.columns
        }

        required = {
            "id",
            "company_id",
            "site_id",
            "asset_id",
            "prediction_type",
            "status",
            "predicted_at",
        }

        missing = required - self.columns

        if missing:
            raise PredictionRepositoryError(
                "Prediction model missing columns: "
                f"{sorted(missing)}"
            )

    # ========================================================
    # VALIDATION HELPERS
    # ========================================================

    @staticmethod
    def _normalize(value: Any) -> Any:
        return value.value if isinstance(value, Enum) else value

    @staticmethod
    def _require_text(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise PredictionValidationError(
                f"{name} must be a nonempty string"
            )

        return value.strip()

    @staticmethod
    def _validate_datetime(
        value: datetime | None,
        name: str,
    ) -> None:
        if value is None:
            return

        if (
            not isinstance(value, datetime)
            or value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise PredictionValidationError(
                f"{name} must be timezone-aware datetime"
            )

    @staticmethod
    def _number(
        value: Any,
        name: str,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        if isinstance(value, bool):
            raise PredictionValidationError(
                f"{name} must be numeric"
            )

        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            raise PredictionValidationError(
                f"{name} must be numeric"
            ) from None

        if not isfinite(number):
            raise PredictionValidationError(
                f"{name} must be finite"
            )

        if minimum is not None and number < minimum:
            raise PredictionValidationError(
                f"{name} must be >= {minimum}"
            )

        if maximum is not None and number > maximum:
            raise PredictionValidationError(
                f"{name} must be <= {maximum}"
            )

        return number

    @staticmethod
    def _validate_limit(
        value: int,
        maximum: int = 1000,
    ) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= maximum
        ):
            raise PredictionValidationError(
                f"limit must be between 1 and {maximum}"
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
            raise PredictionValidationError(
                "start_time cannot exceed end_time"
            )

    def _column(self, name: str):
        if name not in self.columns:
            raise PredictionRepositoryError(
                f"Prediction model has no column '{name}'"
            )

        return getattr(Prediction, name)

    def _validate_payload(
        self,
        data: Mapping[str, Any],
        *,
        allowed: frozenset[str],
    ) -> dict[str, Any]:
        if not isinstance(data, Mapping):
            raise PredictionValidationError(
                "Prediction data must be a mapping"
            )

        payload = {
            key: self._normalize(value)
            for key, value in data.items()
        }

        unsupported = set(payload) - allowed

        if unsupported:
            raise PredictionValidationError(
                f"Unsupported fields: {sorted(unsupported)}"
            )

        unavailable = set(payload) - self.columns

        if unavailable:
            raise PredictionValidationError(
                f"Unavailable model columns: {sorted(unavailable)}"
            )

        for field in (
            "failure_probability",
            "confidence_score",
        ):
            if field in payload and payload[field] is not None:
                payload[field] = self._number(
                    payload[field],
                    field,
                    minimum=0,
                    maximum=1,
                )

        if (
            "health_score" in payload
            and payload["health_score"] is not None
        ):
            payload["health_score"] = self._number(
                payload["health_score"],
                "health_score",
                minimum=0,
                maximum=100,
            )

        for field in (
            "rul_hours",
            "prediction_horizon_hours",
        ):
            if field in payload and payload[field] is not None:
                payload[field] = self._number(
                    payload[field],
                    field,
                    minimum=0,
                )

        for field in (
            "prediction_value",
            "lower_bound",
            "upper_bound",
            "actual_value",
        ):
            if field in payload and payload[field] is not None:
                payload[field] = self._number(
                    payload[field], field
                )

        if (
            payload.get("lower_bound") is not None
            and payload.get("upper_bound") is not None
            and payload["lower_bound"] > payload["upper_bound"]
        ):
            raise PredictionValidationError(
                "lower_bound cannot exceed upper_bound"
            )

        if "predicted_at" in payload:
            self._validate_datetime(
                payload["predicted_at"], "predicted_at"
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
            company_id, "company_id"
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

        if not include_deleted and "deleted_at" in self.columns:
            conditions.append(
                self._column("deleted_at").is_(None)
            )

        return and_(*conditions)

    def _touch(self, prediction: Prediction) -> None:
        if "updated_at" in self.columns:
            prediction.updated_at = utc_now()

    # ========================================================
    # CREATE
    # ========================================================

    async def create(
        self,
        *,
        company_id: str,
        data: Mapping[str, Any],
    ) -> Prediction:
        company_id = self._require_text(
            company_id, "company_id"
        )

        payload = self._validate_payload(
            data,
            allowed=self.CREATE_FIELDS,
        )

        for field in (
            "site_id",
            "asset_id",
            "prediction_type",
        ):
            payload[field] = self._require_text(
                payload.get(field), field
            )

        payload["company_id"] = company_id
        payload.setdefault("status", "generated")
        payload.setdefault("predicted_at", utc_now())

        if "created_at" in self.columns:
            payload.setdefault("created_at", utc_now())

        if "updated_at" in self.columns:
            payload.setdefault("updated_at", utc_now())

        prediction = Prediction(**payload)

        self.session.add(prediction)
        await self.session.flush()

        return prediction

    # ========================================================
    # BATCH CREATE
    # ========================================================

    async def create_many(
        self,
        *,
        company_id: str,
        records: Sequence[Mapping[str, Any]],
    ) -> list[Prediction]:
        if len(records) > 1000:
            raise PredictionValidationError(
                "Maximum 1000 predictions per batch"
            )

        created: list[Prediction] = []

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
        prediction_id: UUID | str,
        *,
        company_id: str,
        include_deleted: bool = False,
    ) -> Prediction | None:
        query = (
            select(Prediction)
            .where(
                self._column("id") == prediction_id,
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
        prediction_id: UUID | str,
        *,
        company_id: str,
    ) -> Prediction:
        prediction = await self.get_by_id(
            prediction_id,
            company_id=company_id,
        )

        if prediction is None:
            raise PredictionNotFoundError(
                f"Prediction '{prediction_id}' not found"
            )

        return prediction

    # ========================================================
    # UPDATE
    # ========================================================

    async def update(
        self,
        prediction_id: UUID | str,
        *,
        company_id: str,
        data: Mapping[str, Any],
    ) -> Prediction:
        prediction = await self.get_required(
            prediction_id,
            company_id=company_id,
        )

        payload = self._validate_payload(
            data,
            allowed=self.UPDATE_FIELDS,
        )

        if str(self._normalize(prediction.status)) in (
            self.TERMINAL_STATUSES
        ):
            raise PredictionConflictError(
                "Cannot update terminal prediction"
            )

        for key, value in payload.items():
            setattr(prediction, key, value)

        self._touch(prediction)
        await self.session.flush()

        return prediction

    # ========================================================
    # UPDATE STATUS
    # ========================================================

    async def update_status(
        self,
        prediction_id: UUID | str,
        *,
        company_id: str,
        status: str,
    ) -> Prediction:
        prediction = await self.get_required(
            prediction_id,
            company_id=company_id,
        )

        prediction.status = self._require_text(
            status, "status"
        )

        self._touch(prediction)
        await self.session.flush()

        return prediction

    # ========================================================
    # RECORD ACTUAL OUTCOME
    # ========================================================

    async def record_actual(
        self,
        prediction_id: UUID | str,
        *,
        company_id: str,
        actual_value: float,
        verified_by: str | None = None,
    ) -> Prediction:
        prediction = await self.get_required(
            prediction_id,
            company_id=company_id,
        )

        actual = self._number(
            actual_value, "actual_value"
        )

        self._column("actual_value")
        prediction.actual_value = actual

        predicted = getattr(
            prediction, "prediction_value", None
        )

        if predicted is not None:
            predicted = self._number(
                predicted, "prediction_value"
            )

            error = actual - predicted

            if "absolute_error" in self.columns:
                prediction.absolute_error = abs(error)

            if "squared_error" in self.columns:
                prediction.squared_error = error ** 2

            if "percentage_error" in self.columns:
                prediction.percentage_error = (
                    abs(error) / abs(actual) * 100
                    if actual != 0
                    else None
                )

        if verified_by is not None:
            verified_by = self._require_text(
                verified_by, "verified_by"
            )

            if "verified_by" in self.columns:
                prediction.verified_by = verified_by

            if "verified_at" in self.columns:
                prediction.verified_at = utc_now()

            prediction.status = "verified"

        self._touch(prediction)
        await self.session.flush()

        return prediction

    # ========================================================
    # HUMAN REVIEW
    # ========================================================

    async def review(
        self,
        prediction_id: UUID | str,
        *,
        company_id: str,
        reviewed_by: str,
        notes: str | None = None,
    ) -> Prediction:
        reviewed_by = self._require_text(
            reviewed_by, "reviewed_by"
        )

        prediction = await self.get_required(
            prediction_id,
            company_id=company_id,
        )

        if "reviewed_by" in self.columns:
            prediction.reviewed_by = reviewed_by

        if "reviewed_at" in self.columns:
            prediction.reviewed_at = utc_now()

        if notes is not None and "review_notes" in self.columns:
            prediction.review_notes = notes

        prediction.status = "reviewed"

        self._touch(prediction)
        await self.session.flush()

        return prediction

    # ========================================================
    # SOFT DELETE
    # ========================================================

    async def soft_delete(
        self,
        prediction_id: UUID | str,
        *,
        company_id: str,
    ) -> bool:
        self._column("deleted_at")

        prediction = await self.get_by_id(
            prediction_id,
            company_id=company_id,
        )

        if prediction is None:
            return False

        prediction.deleted_at = utc_now()

        self._touch(prediction)
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
        prediction_type: str | None = None,
        status: str | None = None,
        risk_level: str | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
        min_confidence: float | None = None,
        min_failure_probability: float | None = None,
        max_rul_hours: float | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "predicted_at",
        descending: bool = True,
    ) -> tuple[list[Prediction], int]:
        self._validate_period(
            start_time, end_time
        )

        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise PredictionValidationError(
                "page must be >= 1"
            )

        self._validate_limit(page_size, 100)

        if sort_by not in self.SORT_FIELDS:
            raise PredictionValidationError(
                f"Invalid sort field '{sort_by}'"
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
            "prediction_type": prediction_type,
            "status": status,
            "risk_level": risk_level,
            "model_name": model_name,
            "model_version": model_version,
        }

        for field, value in filters.items():
            if value is not None:
                conditions.append(
                    self._column(field) == value
                )

        if min_confidence is not None:
            threshold = self._number(
                min_confidence,
                "min_confidence",
                minimum=0,
                maximum=1,
            )
            conditions.append(
                self._column("confidence_score") >= threshold
            )

        if min_failure_probability is not None:
            threshold = self._number(
                min_failure_probability,
                "min_failure_probability",
                minimum=0,
                maximum=1,
            )
            conditions.append(
                self._column("failure_probability") >= threshold
            )

        if max_rul_hours is not None:
            threshold = self._number(
                max_rul_hours,
                "max_rul_hours",
                minimum=0,
            )
            conditions.append(
                self._column("rul_hours") <= threshold
            )

        if start_time is not None:
            conditions.append(
                self._column("predicted_at") >= start_time
            )

        if end_time is not None:
            conditions.append(
                self._column("predicted_at") <= end_time
            )

        predicate = and_(*conditions)

        total = (
            await self.session.execute(
                select(func.count())
                .select_from(Prediction)
                .where(predicate)
            )
        ).scalar_one()

        order_column = self._column(sort_by)

        query = (
            select(Prediction)
            .where(predicate)
            .order_by(
                order_column.desc()
                if descending
                else order_column.asc(),
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
    # ASSET HISTORY
    # ========================================================

    async def get_asset_history(
        self,
        *,
        company_id: str,
        asset_id: str,
        prediction_type: str | None = None,
        limit: int = 500,
    ) -> list[Prediction]:
        self._validate_limit(limit)

        conditions = [
            self._scope(
                company_id=company_id,
                asset_id=asset_id,
            )
        ]

        if prediction_type is not None:
            conditions.append(
                self._column("prediction_type")
                == prediction_type
            )

        query = (
            select(Prediction)
            .where(and_(*conditions))
            .order_by(
                self._column("predicted_at").desc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # ========================================================
    # MACHINE HISTORY
    # ========================================================

    async def get_machine_history(
        self,
        *,
        company_id: str,
        machine_id: str,
        limit: int = 500,
    ) -> list[Prediction]:
        self._validate_limit(limit)

        query = (
            select(Prediction)
            .where(
                self._scope(
                    company_id=company_id,
                    machine_id=machine_id,
                )
            )
            .order_by(
                self._column("predicted_at").desc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # ========================================================
    # LATEST PREDICTION
    # ========================================================

    async def get_latest(
        self,
        *,
        company_id: str,
        asset_id: str,
        prediction_type: str,
    ) -> Prediction | None:
        query = (
            select(Prediction)
            .where(
                self._scope(
                    company_id=company_id,
                    asset_id=asset_id,
                ),
                self._column("prediction_type")
                == prediction_type,
            )
            .order_by(
                self._column("predicted_at").desc(),
                self._column("id").desc(),
            )
            .limit(1)
        )

        result = await self.session.execute(query)

        return result.scalar_one_or_none()

    # ========================================================
    # HIGH-RISK PREDICTIONS
    # ========================================================

    async def get_high_risk(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        min_probability: float = 0.8,
        limit: int = 100,
    ) -> list[Prediction]:
        self._validate_limit(limit)

        threshold = self._number(
            min_probability,
            "min_probability",
            minimum=0,
            maximum=1,
        )

        risk_conditions = []

        if "failure_probability" in self.columns:
            risk_conditions.append(
                self._column("failure_probability")
                >= threshold
            )

        if "risk_level" in self.columns:
            risk_conditions.append(
                self._column("risk_level").in_(
                    self.HIGH_RISK_LEVELS
                )
            )

        if not risk_conditions:
            raise PredictionRepositoryError(
                "High-risk queries require risk fields"
            )

        query = (
            select(Prediction)
            .where(
                self._scope(
                    company_id=company_id,
                    site_id=site_id,
                ),
                or_(*risk_conditions),
            )
            .order_by(
                self._column("predicted_at").desc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # ========================================================
    # LOW-RUL PREDICTIONS
    # ========================================================

    async def get_low_rul(
        self,
        *,
        company_id: str,
        max_rul_hours: float = 72.0,
        site_id: str | None = None,
        limit: int = 100,
    ) -> list[Prediction]:
        self._validate_limit(limit)

        threshold = self._number(
            max_rul_hours,
            "max_rul_hours",
            minimum=0,
        )

        query = (
            select(Prediction)
            .where(
                self._scope(
                    company_id=company_id,
                    site_id=site_id,
                ),
                self._column("rul_hours") <= threshold,
            )
            .order_by(
                self._column("rul_hours").asc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # ========================================================
    # MODEL ACCURACY METRICS
    # ========================================================

    async def get_accuracy_metrics(
        self,
        *,
        company_id: str,
        model_name: str | None = None,
        prediction_type: str | None = None,
    ) -> dict[str, Any]:
        self._column("actual_value")
        self._column("prediction_value")

        conditions = [
            self._scope(company_id=company_id),
            self._column("actual_value").is_not(None),
            self._column("prediction_value").is_not(None),
        ]

        if model_name is not None:
            conditions.append(
                self._column("model_name") == model_name
            )

        if prediction_type is not None:
            conditions.append(
                self._column("prediction_type")
                == prediction_type
            )

        error = (
            self._column("actual_value")
            - self._column("prediction_value")
        )

        result = await self.session.execute(
            select(
                func.count(),
                func.avg(func.abs(error)),
                func.avg(error * error),
            )
            .where(and_(*conditions))
        )

        count, mae, mse = result.one()

        return {
            "company_id": company_id,
            "model_name": model_name,
            "prediction_type": prediction_type,
            "evaluated_predictions": int(count),
            "mae": float(mae) if mae is not None else None,
            "mse": float(mse) if mse is not None else None,
            "rmse": (
                sqrt(float(mse))
                if mse is not None
                else None
            ),
            "generated_at": utc_now(),
        }

    # ========================================================
    # STATISTICS
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
                .select_from(Prediction)
                .where(predicate)
            )
        ).scalar_one()

        distributions: dict[str, dict[str, int]] = {}

        for field in (
            "prediction_type",
            "status",
            "risk_level",
        ):
            if field not in self.columns:
                continue

            column = self._column(field)

            result = await self.session.execute(
                select(column, func.count())
                .where(predicate)
                .group_by(column)
            )

            distributions[field] = {
                str(self._normalize(value)): int(count)
                for value, count in result.all()
            }

        averages: dict[str, float | None] = {}

        for field in (
            "failure_probability",
            "rul_hours",
            "health_score",
            "confidence_score",
        ):
            if field not in self.columns:
                continue

            value = (
                await self.session.execute(
                    select(func.avg(self._column(field)))
                    .where(predicate)
                )
            ).scalar_one()

            averages[field] = (
                float(value) if value is not None else None
            )

        return {
            "company_id": company_id,
            "site_id": site_id,
            "total_predictions": int(total),
            "prediction_type_counts": distributions.get(
                "prediction_type", {}
            ),
            "status_counts": distributions.get(
                "status", {}
            ),
            "risk_counts": distributions.get(
                "risk_level", {}
            ),
            "average_failure_probability": averages.get(
                "failure_probability"
            ),
            "average_rul_hours": averages.get("rul_hours"),
            "average_health_score": averages.get(
                "health_score"
            ),
            "average_confidence_score": averages.get(
                "confidence_score"
            ),
            "generated_at": utc_now(),
        }


__all__ = [
    "PredictionRepository",
    "PredictionRepositoryError",
    "PredictionNotFoundError",
    "PredictionValidationError",
    "PredictionConflictError",
]
