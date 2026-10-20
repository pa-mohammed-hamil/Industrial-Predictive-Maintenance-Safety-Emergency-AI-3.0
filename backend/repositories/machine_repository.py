
"""
Industrial Machine Repository
=============================

File: app/repositories/machine_repository.py

Python: 3.11+
SQLAlchemy: 2.x

Features:
- Machine registration
- Tenant-aware CRUD operations
- Company/site/asset isolation
- Machine code and serial number lookup
- Machine operating status
- Machine health and condition tracking
- Failure risk and RUL updates
- Maintenance scheduling
- Machine metadata management
- Search, filtering and pagination
- Critical and unhealthy machine queries
- Maintenance-due queries
- Dashboard statistics
- Soft deletion

Expected SQLAlchemy model:
    app.models.machine.Machine

Required columns:
    id
    company_id
    site_id
    machine_code
    machine_name

Recommended optional columns:
    asset_id
    production_line_id
    machine_type
    category
    manufacturer
    model_number
    serial_number
    status
    operating_mode
    condition
    criticality
    risk_level
    health_score
    failure_probability
    rul_hours
    installation_date
    commissioning_date
    last_maintenance_at
    next_maintenance_at
    last_telemetry_at
    last_failure_at
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

from app.models.machine import Machine


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MachineRepositoryError(Exception):
    """Base machine repository exception."""


class MachineNotFoundError(MachineRepositoryError):
    """Machine not found."""


class MachineValidationError(MachineRepositoryError):
    """Invalid machine repository input."""


class MachineConflictError(MachineRepositoryError):
    """Machine uniqueness conflict."""


class MachineRepository:
    """
    Async SQLAlchemy repository for industrial machines.

    Security:
        All reads and writes are scoped by company_id.

    Transactions:
        Repository calls flush().
        Service layer handles commit/rollback.

    Important:
        Database state changes do not directly control
        physical industrial machinery.
    """

    CREATE_FIELDS = frozenset({
        "id",
        "site_id",
        "asset_id",
        "production_line_id",
        "machine_code",
        "machine_name",
        "machine_type",
        "category",
        "manufacturer",
        "model_number",
        "serial_number",
        "status",
        "operating_mode",
        "condition",
        "criticality",
        "risk_level",
        "health_score",
        "failure_probability",
        "rul_hours",
        "installation_date",
        "commissioning_date",
        "last_maintenance_at",
        "next_maintenance_at",
        "last_telemetry_at",
        "last_failure_at",
        "metadata_json",
    })

    UPDATE_FIELDS = frozenset({
        "machine_name",
        "machine_type",
        "category",
        "manufacturer",
        "model_number",
        "serial_number",
        "operating_mode",
        "condition",
        "criticality",
        "metadata_json",
    })

    SORT_FIELDS = frozenset({
        "machine_name",
        "machine_code",
        "created_at",
        "updated_at",
        "health_score",
        "failure_probability",
        "rul_hours",
        "next_maintenance_at",
        "status",
    })

    HIGH_RISK_LEVELS = frozenset({
        "high",
        "critical",
        "extreme",
    })

    ACTIVE_STATUSES = frozenset({
        "active",
        "running",
        "operational",
        "idle",
        "standby",
    })

    OFFLINE_STATUSES = frozenset({
        "offline",
        "stopped",
        "out_of_service",
        "decommissioned",
    })

    def __init__(self, session: AsyncSession):
        self.session = session

        self.columns = {
            column.key
            for column in inspect(Machine).mapper.columns
        }

        required = {
            "id",
            "company_id",
            "site_id",
            "machine_code",
            "machine_name",
        }

        missing = required - self.columns

        if missing:
            raise MachineRepositoryError(
                "Machine model missing columns: "
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
            raise MachineValidationError(
                f"{name} must be a string"
            )

        value = value.strip()

        if not value:
            raise MachineValidationError(
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
            raise MachineValidationError(
                f"{name} must be datetime"
            )

        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise MachineValidationError(
                f"{name} must be timezone-aware"
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
            raise MachineValidationError(
                f"{name} must be numeric"
            )

        number = float(value)

        if not isfinite(number):
            raise MachineValidationError(
                f"{name} must be finite"
            )

        if minimum is not None and number < minimum:
            raise MachineValidationError(
                f"{name} must be >= {minimum}"
            )

        if maximum is not None and number > maximum:
            raise MachineValidationError(
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
            raise MachineValidationError(
                f"limit must be between 1 and {maximum}"
            )

    def _column(self, name: str):
        if name not in self.columns:
            raise MachineValidationError(
                f"Machine model has no column '{name}'"
            )

        return getattr(Machine, name)

    def _validate_payload(
        self,
        data: Mapping[str, Any],
        *,
        allowed: frozenset[str],
    ) -> dict[str, Any]:
        if not isinstance(data, Mapping):
            raise MachineValidationError(
                "Machine data must be a mapping"
            )

        payload = {
            key: self._normalize(value)
            for key, value in data.items()
        }

        unsupported = set(payload) - allowed

        if unsupported:
            raise MachineValidationError(
                f"Unsupported fields: {sorted(unsupported)}"
            )

        missing_columns = set(payload) - self.columns

        if missing_columns:
            raise MachineValidationError(
                "Fields missing from Machine model: "
                f"{sorted(missing_columns)}"
            )

        if (
            "health_score" in payload
            and payload["health_score"] is not None
        ):
            payload["health_score"] = (
                self._validate_number(
                    payload["health_score"],
                    "health_score",
                    minimum=0,
                    maximum=100,
                )
            )

        if (
            "failure_probability" in payload
            and payload["failure_probability"] is not None
        ):
            payload["failure_probability"] = (
                self._validate_number(
                    payload["failure_probability"],
                    "failure_probability",
                    minimum=0,
                    maximum=1,
                )
            )

        if (
            "rul_hours" in payload
            and payload["rul_hours"] is not None
        ):
            payload["rul_hours"] = (
                self._validate_number(
                    payload["rul_hours"],
                    "rul_hours",
                    minimum=0,
                )
            )

        for name in (
            "installation_date",
            "commissioning_date",
            "last_maintenance_at",
            "next_maintenance_at",
            "last_telemetry_at",
            "last_failure_at",
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

    def _touch(self, machine: Machine) -> None:
        if "updated_at" in self.columns:
            machine.updated_at = utc_now()

    # ========================================================
    # CREATE MACHINE
    # ========================================================

    async def create(
        self,
        *,
        company_id: str,
        data: Mapping[str, Any],
    ) -> Machine:
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
            "machine_code",
            "machine_name",
        ):
            payload[name] = self._require_text(
                payload.get(name),
                name,
            )

        existing = await self.get_by_code(
            payload["machine_code"],
            company_id=company_id,
            site_id=payload["site_id"],
        )

        if existing is not None:
            raise MachineConflictError(
                "Machine code already exists at this site"
            )

        payload["company_id"] = company_id

        if "status" in self.columns:
            payload.setdefault(
                "status",
                "active",
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

        machine = Machine(**payload)

        self.session.add(machine)
        await self.session.flush()

        return machine

    # ========================================================
    # BULK CREATE
    # ========================================================

    async def create_many(
        self,
        *,
        company_id: str,
        records: Sequence[Mapping[str, Any]],
    ) -> list[Machine]:
        if len(records) > 1000:
            raise MachineValidationError(
                "Maximum 1000 machines per batch"
            )

        created: list[Machine] = []

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
        machine_id: UUID | str,
        *,
        company_id: str,
        include_deleted: bool = False,
    ) -> Machine | None:
        query = (
            select(Machine)
            .where(
                self._column("id") == machine_id,
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
        machine_id: UUID | str,
        *,
        company_id: str,
    ) -> Machine:
        machine = await self.get_by_id(
            machine_id,
            company_id=company_id,
        )

        if machine is None:
            raise MachineNotFoundError(
                f"Machine '{machine_id}' not found"
            )

        return machine

    # ========================================================
    # GET BY MACHINE CODE
    # ========================================================

    async def get_by_code(
        self,
        machine_code: str,
        *,
        company_id: str,
        site_id: str,
    ) -> Machine | None:
        machine_code = self._require_text(
            machine_code,
            "machine_code",
        )

        query = (
            select(Machine)
            .where(
                self._scope(
                    company_id=company_id,
                    site_id=site_id,
                ),
                self._column("machine_code")
                == machine_code,
            )
        )

        result = await self.session.execute(query)

        return result.scalar_one_or_none()

    # ========================================================
    # GET BY SERIAL NUMBER
    # ========================================================

    async def get_by_serial_number(
        self,
        serial_number: str,
        *,
        company_id: str,
    ) -> list[Machine]:
        serial_number = self._require_text(
            serial_number,
            "serial_number",
        )

        query = (
            select(Machine)
            .where(
                self._scope(
                    company_id=company_id,
                ),
                self._column("serial_number")
                == serial_number,
            )
            .order_by(
                self._column("machine_name").asc()
            )
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # ========================================================
    # UPDATE MACHINE
    # ========================================================

    async def update(
        self,
        machine_id: UUID | str,
        *,
        company_id: str,
        data: Mapping[str, Any],
    ) -> Machine:
        machine = await self.get_required(
            machine_id,
            company_id=company_id,
        )

        payload = self._validate_payload(
            data,
            allowed=self.UPDATE_FIELDS,
        )

        for key, value in payload.items():
            setattr(machine, key, value)

        self._touch(machine)
        await self.session.flush()

        return machine

    # ========================================================
    # UPDATE STATUS
    # ========================================================

    async def update_status(
        self,
        machine_id: UUID | str,
        *,
        company_id: str,
        status: str,
        operating_mode: str | None = None,
    ) -> Machine:
        machine = await self.get_required(
            machine_id,
            company_id=company_id,
        )

        self._column("status")

        machine.status = self._require_text(
            status,
            "status",
        )

        if operating_mode is not None:
            self._column("operating_mode")

            machine.operating_mode = (
                self._require_text(
                    operating_mode,
                    "operating_mode",
                )
            )

        self._touch(machine)
        await self.session.flush()

        return machine

    # ========================================================
    # UPDATE HEALTH
    # ========================================================

    async def update_health(
        self,
        machine_id: UUID | str,
        *,
        company_id: str,
        health_score: float,
        condition: str | None = None,
        risk_level: str | None = None,
    ) -> Machine:
        machine = await self.get_required(
            machine_id,
            company_id=company_id,
        )

        self._column("health_score")

        machine.health_score = (
            self._validate_number(
                health_score,
                "health_score",
                minimum=0,
                maximum=100,
            )
        )

        if condition is not None:
            self._column("condition")
            machine.condition = condition

        if risk_level is not None:
            self._column("risk_level")
            machine.risk_level = risk_level

        self._touch(machine)
        await self.session.flush()

        return machine

    # ========================================================
    # UPDATE FAILURE RISK / RUL
    # ========================================================

    async def update_prediction(
        self,
        machine_id: UUID | str,
        *,
        company_id: str,
        failure_probability: float | None = None,
        rul_hours: float | None = None,
        risk_level: str | None = None,
    ) -> Machine:
        machine = await self.get_required(
            machine_id,
            company_id=company_id,
        )

        if failure_probability is not None:
            self._column("failure_probability")

            machine.failure_probability = (
                self._validate_number(
                    failure_probability,
                    "failure_probability",
                    minimum=0,
                    maximum=1,
                )
            )

        if rul_hours is not None:
            self._column("rul_hours")

            machine.rul_hours = (
                self._validate_number(
                    rul_hours,
                    "rul_hours",
                    minimum=0,
                )
            )

        if risk_level is not None:
            self._column("risk_level")
            machine.risk_level = risk_level

        self._touch(machine)
        await self.session.flush()

        return machine

    # ========================================================
    # UPDATE MAINTENANCE DATES
    # ========================================================

    async def update_maintenance_dates(
        self,
        machine_id: UUID | str,
        *,
        company_id: str,
        last_maintenance_at: datetime | None = None,
        next_maintenance_at: datetime | None = None,
    ) -> Machine:
        machine = await self.get_required(
            machine_id,
            company_id=company_id,
        )

        if last_maintenance_at is not None:
            self._validate_datetime(
                last_maintenance_at,
                "last_maintenance_at",
            )

            self._column("last_maintenance_at")

            machine.last_maintenance_at = (
                last_maintenance_at
            )

        if next_maintenance_at is not None:
            self._validate_datetime(
                next_maintenance_at,
                "next_maintenance_at",
            )

            self._column("next_maintenance_at")

            machine.next_maintenance_at = (
                next_maintenance_at
            )

        self._touch(machine)
        await self.session.flush()

        return machine

    # ========================================================
    # UPDATE TELEMETRY TIMESTAMP
    # ========================================================

    async def update_last_telemetry(
        self,
        machine_id: UUID | str,
        *,
        company_id: str,
        telemetry_at: datetime,
    ) -> Machine:
        self._validate_datetime(
            telemetry_at,
            "telemetry_at",
        )

        machine = await self.get_required(
            machine_id,
            company_id=company_id,
        )

        self._column("last_telemetry_at")

        machine.last_telemetry_at = telemetry_at

        self._touch(machine)
        await self.session.flush()

        return machine

    # ========================================================
    # SOFT DELETE
    # ========================================================

    async def soft_delete(
        self,
        machine_id: UUID | str,
        *,
        company_id: str,
    ) -> bool:
        if "deleted_at" not in self.columns:
            raise MachineRepositoryError(
                "Soft deletion requires deleted_at"
            )

        machine = await self.get_by_id(
            machine_id,
            company_id=company_id,
        )

        if machine is None:
            return False

        machine.deleted_at = utc_now()

        self._touch(machine)
        await self.session.flush()

        return True

    # ========================================================
    # LIST MACHINES
    # ========================================================

    async def list(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        asset_id: str | None = None,
        production_line_id: str | None = None,
        machine_type: str | None = None,
        status: str | None = None,
        condition: str | None = None,
        criticality: str | None = None,
        risk_level: str | None = None,
        search: str | None = None,
        min_health: float | None = None,
        max_health: float | None = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "machine_name",
        descending: bool = False,
    ) -> tuple[list[Machine], int]:
        if page < 1:
            raise MachineValidationError(
                "page must be >= 1"
            )

        self._validate_limit(
            page_size,
            maximum=100,
        )

        if sort_by not in self.SORT_FIELDS:
            raise MachineValidationError(
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
            "production_line_id": production_line_id,
            "machine_type": machine_type,
            "status": status,
            "condition": condition,
            "criticality": criticality,
            "risk_level": risk_level,
        }

        for field, value in filters.items():
            if value is not None:
                conditions.append(
                    self._column(field) == value
                )

        if min_health is not None:
            minimum = self._validate_number(
                min_health,
                "min_health",
                minimum=0,
                maximum=100,
            )

            conditions.append(
                self._column("health_score") >= minimum
            )

        if max_health is not None:
            maximum = self._validate_number(
                max_health,
                "max_health",
                minimum=0,
                maximum=100,
            )

            conditions.append(
                self._column("health_score") <= maximum
            )

        if search:
            pattern = f"%{search.strip()}%"

            search_fields = [
                "machine_name",
                "machine_code",
            ]

            for optional in (
                "manufacturer",
                "model_number",
                "serial_number",
            ):
                if optional in self.columns:
                    search_fields.append(optional)

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
                .select_from(Machine)
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
            select(Machine)
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
    # HIGH-RISK MACHINES
    # ========================================================

    async def get_high_risk(
        self,
        *,
        company_id: str,
        site_id: str | None = None,
        limit: int = 100,
    ) -> list[Machine]:
        self._validate_limit(limit)

        query = (
            select(Machine)
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
                self._column("machine_name").asc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # ========================================================
    # LOW-HEALTH MACHINES
    # ========================================================

    async def get_low_health(
        self,
        *,
        company_id: str,
        threshold: float = 40.0,
        site_id: str | None = None,
        limit: int = 100,
    ) -> list[Machine]:
        self._validate_limit(limit)

        threshold = self._validate_number(
            threshold,
            "threshold",
            minimum=0,
            maximum=100,
        )

        query = (
            select(Machine)
            .where(
                self._scope(
                    company_id=company_id,
                    site_id=site_id,
                ),
                self._column("health_score")
                <= threshold,
            )
            .order_by(
                self._column("health_score").asc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # ========================================================
    # MAINTENANCE-DUE MACHINES
    # ========================================================

    async def get_maintenance_due(
        self,
        *,
        company_id: str,
        before: datetime | None = None,
        site_id: str | None = None,
        limit: int = 100,
    ) -> list[Machine]:
        self._validate_limit(limit)

        before = before or utc_now()

        self._validate_datetime(
            before,
            "before",
        )

        query = (
            select(Machine)
            .where(
                self._scope(
                    company_id=company_id,
                    site_id=site_id,
                ),
                self._column("next_maintenance_at")
                <= before,
            )
            .order_by(
                self._column("next_maintenance_at").asc()
            )
            .limit(limit)
        )

        result = await self.session.execute(query)

        return list(result.scalars().all())

    # ========================================================
    # MACHINE STATISTICS
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
                .select_from(Machine)
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

        average_health = None
        average_rul = None

        if "health_score" in self.columns:
            value = (
                await self.session.execute(
                    select(
                        func.avg(
                            self._column("health_score")
                        )
                    )
                    .where(predicate)
                )
            ).scalar_one()

            if value is not None:
                average_health = float(value)

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
                average_rul = float(value)

        return {
            "company_id": company_id,
            "site_id": site_id,
            "total_machines": int(total),
            "active_machines": sum(
                count
                for status, count in status_counts.items()
                if status in self.ACTIVE_STATUSES
            ),
            "offline_machines": sum(
                count
                for status, count in status_counts.items()
                if status in self.OFFLINE_STATUSES
            ),
            "high_risk_machines": sum(
                count
                for risk, count in risk_counts.items()
                if risk in self.HIGH_RISK_LEVELS
            ),
            "average_health_score": average_health,
            "average_rul_hours": average_rul,
            "status_counts": status_counts,
            "risk_counts": risk_counts,
            "generated_at": utc_now(),
        }


__all__ = [
    "MachineRepository",
    "MachineRepositoryError",
    "MachineNotFoundError",
    "MachineValidationError",
    "MachineConflictError",
]
