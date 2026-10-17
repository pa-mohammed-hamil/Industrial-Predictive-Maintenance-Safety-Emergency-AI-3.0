"""
scripts/seed_database.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

PostgreSQL / SQLite Development Database Seeder
================================================

Python 3.11+

Dependencies:
    pip install "sqlalchemy>=2,<3" pandas psycopg[binary]

Usage:
    python scripts/seed_database.py --init-db
    python scripts/seed_database.py --seed-all
    python scripts/seed_database.py --seed-all --batch-size 1000

SQLite development:
    python scripts/seed_database.py \
        --database-url sqlite:///./industrial_dev.db \
        --init-db --seed-all

Environment:
    DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/industrial_ai

Features:
    - PostgreSQL and SQLite
    - SQLAlchemy 2.x
    - Database initialization
    - Companies, sites and assets
    - Sensor telemetry
    - Failure events
    - Maintenance history
    - Safety events and incidents
    - Financial cost records
    - Foreign key relationships
    - Tenant-aware records
    - Deterministic identifiers
    - Idempotent inserts
    - Batch processing
    - Transaction handling
    - Data validation
    - Seed summary
    - Command-line interface

Security:
    - Development and testing only
    - Never resets existing production tables
    - No hardcoded database credentials
    - No automatic destructive operations
    - Source CSV data is validated
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    func,
    select,
)

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)


LOGGER = logging.getLogger("seed_database")


# ============================================================
# DATABASE BASE
# ============================================================

class Base(DeclarativeBase):
    pass


# ============================================================
# COMPANY
# ============================================================

class SeedCompany(Base):
    __tablename__ = "seed_companies"

    company_id: Mapped[str] = mapped_column(
        String(100),
        primary_key=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


# ============================================================
# SITE
# ============================================================

class SeedSite(Base):
    __tablename__ = "seed_sites"

    site_id: Mapped[str] = mapped_column(
        String(150),
        primary_key=True,
    )

    company_id: Mapped[str] = mapped_column(
        String(100),
        ForeignKey("seed_companies.company_id"),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )


# ============================================================
# ASSET
# ============================================================

class SeedAsset(Base):
    __tablename__ = "seed_assets"

    asset_id: Mapped[str] = mapped_column(
        String(150),
        primary_key=True,
    )

    company_id: Mapped[str] = mapped_column(
        String(100),
        ForeignKey("seed_companies.company_id"),
        nullable=False,
        index=True,
    )

    site_id: Mapped[str] = mapped_column(
        String(150),
        ForeignKey("seed_sites.site_id"),
        nullable=False,
        index=True,
    )

    equipment_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    age_years: Mapped[float | None] = mapped_column(
        Float,
    )

    initial_health_score: Mapped[float | None] = (
        mapped_column(Float)
    )

    operating_hours_per_day: Mapped[
        float | None
    ] = mapped_column(Float)

    criticality: Mapped[str | None] = mapped_column(
        String(50)
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="active",
    )

    source_dataset: Mapped[str] = mapped_column(
        String(100),
        default="synthetic",
    )


# ============================================================
# TELEMETRY
# ============================================================

class SeedTelemetry(Base):
    __tablename__ = "seed_telemetry"

    reading_id: Mapped[str] = mapped_column(
        String(200),
        primary_key=True,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    asset_id: Mapped[str] = mapped_column(
        String(150),
        ForeignKey("seed_assets.asset_id"),
        nullable=False,
        index=True,
    )

    company_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    site_id: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    temperature_c: Mapped[float | None] = (
        mapped_column(Float)
    )

    vibration_mm_s: Mapped[float | None] = (
        mapped_column(Float)
    )

    pressure_bar: Mapped[float | None] = (
        mapped_column(Float)
    )

    current_a: Mapped[float | None] = (
        mapped_column(Float)
    )

    voltage_v: Mapped[float | None] = (
        mapped_column(Float)
    )

    rpm: Mapped[float | None] = (
        mapped_column(Float)
    )

    power_kw: Mapped[float | None] = (
        mapped_column(Float)
    )

    energy_kwh: Mapped[float | None] = (
        mapped_column(Float)
    )

    flow_rate_l_min: Mapped[float | None] = (
        mapped_column(Float)
    )

    humidity_percent: Mapped[float | None] = (
        mapped_column(Float)
    )

    acoustic_db: Mapped[float | None] = (
        mapped_column(Float)
    )

    health_score: Mapped[float | None] = (
        mapped_column(Float)
    )

    anomaly_label: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    anomaly_type: Mapped[str | None] = (
        mapped_column(String(100))
    )


# ============================================================
# FAILURE EVENTS
# ============================================================

class SeedFailure(Base):
    __tablename__ = "seed_failures"

    failure_id: Mapped[str] = mapped_column(
        String(150),
        primary_key=True,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    asset_id: Mapped[str] = mapped_column(
        String(150),
        ForeignKey("seed_assets.asset_id"),
        nullable=False,
        index=True,
    )

    company_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    site_id: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    failure_mode: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    severity: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    downtime_hours: Mapped[float] = mapped_column(
        Float,
        default=0,
    )

    repair_cost: Mapped[float] = mapped_column(
        Float,
        default=0,
    )

    currency: Mapped[str] = mapped_column(
        String(10),
        default="INR",
    )


# ============================================================
# MAINTENANCE HISTORY
# ============================================================

class SeedMaintenance(Base):
    __tablename__ = "seed_maintenance"

    maintenance_id: Mapped[str] = mapped_column(
        String(150),
        primary_key=True,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    asset_id: Mapped[str] = mapped_column(
        String(150),
        ForeignKey("seed_assets.asset_id"),
        nullable=False,
        index=True,
    )

    company_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    site_id: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    maintenance_type: Mapped[str] = (
        mapped_column(
            String(100),
            nullable=False,
        )
    )

    failure_id: Mapped[str | None] = (
        mapped_column(
            String(150),
            ForeignKey("seed_failures.failure_id"),
            nullable=True,
        )
    )

    health_before: Mapped[float | None] = (
        mapped_column(Float)
    )

    health_after: Mapped[float | None] = (
        mapped_column(Float)
    )

    estimated_cost: Mapped[float] = (
        mapped_column(
            Float,
            default=0,
        )
    )

    currency: Mapped[str] = mapped_column(
        String(10),
        default="INR",
    )


# ============================================================
# SAFETY EVENTS
# ============================================================

class SeedSafetyEvent(Base):
    __tablename__ = "seed_safety_events"

    event_id: Mapped[str] = mapped_column(
        String(150),
        primary_key=True,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    company_id: Mapped[str] = mapped_column(
        String(100),
        ForeignKey("seed_companies.company_id"),
        nullable=False,
        index=True,
    )

    site_id: Mapped[str] = mapped_column(
        String(150),
        ForeignKey("seed_sites.site_id"),
        nullable=False,
    )

    zone_id: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    event_type: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    severity: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    risk_score: Mapped[float] = mapped_column(
        Float,
        default=0,
    )

    detection_source: Mapped[str | None] = (
        mapped_column(String(100))
    )

    incident_label: Mapped[int] = (
        mapped_column(
            Integer,
            default=0,
        )
    )

    emergency_label: Mapped[int] = (
        mapped_column(
            Integer,
            default=0,
        )
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="detected",
    )


# ============================================================
# SAFETY INCIDENTS
# ============================================================

class SeedSafetyIncident(Base):
    __tablename__ = "seed_safety_incidents"

    incident_id: Mapped[str] = mapped_column(
        String(150),
        primary_key=True,
    )

    event_id: Mapped[str] = mapped_column(
        String(150),
        ForeignKey("seed_safety_events.event_id"),
        nullable=False,
        unique=True,
    )

    company_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    site_id: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    incident_type: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    severity: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    injury_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    downtime_hours: Mapped[float] = (
        mapped_column(
            Float,
            default=0,
        )
    )

    estimated_cost: Mapped[float] = (
        mapped_column(
            Float,
            default=0,
        )
    )

    currency: Mapped[str] = mapped_column(
        String(10),
        default="INR",
    )


# ============================================================
# COST RECORDS
# ============================================================

class SeedCost(Base):
    __tablename__ = "seed_costs"

    event_id: Mapped[str] = mapped_column(
        String(150),
        primary_key=True,
    )

    asset_id: Mapped[str] = mapped_column(
        String(150),
        ForeignKey("seed_assets.asset_id"),
        nullable=False,
        index=True,
    )

    company_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    site_id: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    maintenance_type: Mapped[str | None] = (
        mapped_column(String(100))
    )

    labor_cost: Mapped[float] = mapped_column(
        Float,
        default=0,
    )

    spare_parts_cost: Mapped[float] = (
        mapped_column(
            Float,
            default=0,
        )
    )

    service_cost: Mapped[float] = mapped_column(
        Float,
        default=0,
    )

    inspection_cost: Mapped[float] = (
        mapped_column(
            Float,
            default=0,
        )
    )

    direct_maintenance_cost: Mapped[float] = (
        mapped_column(
            Float,
            default=0,
        )
    )

    total_event_cost: Mapped[float] = (
        mapped_column(
            Float,
            default=0,
        )
    )

    currency: Mapped[str] = mapped_column(
        String(10),
        default="INR",
    )


# ============================================================
# DATABASE CONNECTION
# ============================================================

def create_database_engine(
    database_url: str,
):
    engine = create_engine(
        database_url,
        pool_pre_ping=True,
        future=True,
    )

    if engine.dialect.name == "sqlite":
        @event.listens_for(
            engine,
            "connect",
        )
        def enable_foreign_keys(
            connection,
            connection_record,
        ):
            cursor = connection.cursor()
            cursor.execute(
                "PRAGMA foreign_keys=ON"
            )
            cursor.close()

    return engine


# ============================================================
# CSV UTILITIES
# ============================================================

def clean_value(
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(value, float):
        if pd.isna(value):
            return None

    if isinstance(value, str):
        stripped = value.strip()

        if stripped.lower() in {
            "",
            "nan",
            "none",
            "null",
        }:
            return None

        return stripped

    return value


def parse_datetime(
    value: Any,
) -> datetime:
    timestamp = pd.to_datetime(
        value,
        utc=True,
        errors="raise",
    )

    return timestamp.to_pydatetime()


def optional_float(
    value: Any,
) -> float | None:
    value = clean_value(value)

    if value is None:
        return None

    return float(value)


def optional_int(
    value: Any,
) -> int | None:
    value = clean_value(value)

    if value is None:
        return None

    return int(value)


def read_csv_batches(
    path: Path,
    batch_size: int,
) -> Iterator[list[dict[str, Any]]]:
    if not path.exists():
        LOGGER.warning(
            "Dataset not found: %s",
            path,
        )
        return

    for dataframe in pd.read_csv(
        path,
        chunksize=batch_size,
        dtype=str,
        keep_default_na=False,
    ):
        records = []

        for record in dataframe.to_dict(
            orient="records"
        ):
            records.append({
                key: clean_value(value)
                for key, value in record.items()
            })

        yield records


# ============================================================
# DATABASE SEEDER
# ============================================================

class IndustrialDatabaseSeeder:
    def __init__(
        self,
        database_url: str,
        batch_size: int = 1000,
    ) -> None:
        if batch_size < 1:
            raise ValueError(
                "batch_size must be positive"
            )

        self.engine = create_database_engine(
            database_url
        )

        self.SessionLocal = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
        )

        self.batch_size = batch_size

    # --------------------------------------------------------
    # INITIALIZE
    # --------------------------------------------------------

    def initialize_database(self) -> None:
        Base.metadata.create_all(
            self.engine
        )

        LOGGER.info(
            "Seed database tables initialized"
        )

    # --------------------------------------------------------
    # IDEMPOTENT INSERT
    # --------------------------------------------------------

    def insert_rows(
        self,
        model: type[Base],
        rows: list[dict[str, Any]],
    ) -> int:
        if not rows:
            return 0

        table = model.__table__

        dialect = self.engine.dialect.name

        if dialect == "postgresql":
            statement = pg_insert(
                table
            ).values(rows)

            statement = (
                statement.on_conflict_do_nothing()
            )

        elif dialect == "sqlite":
            statement = sqlite_insert(
                table
            ).values(rows)

            statement = (
                statement.on_conflict_do_nothing()
            )

        else:
            raise RuntimeError(
                "Only PostgreSQL and SQLite "
                "are supported"
            )

        with self.engine.begin() as connection:
            result = connection.execute(
                statement
            )

        return max(
            result.rowcount or 0,
            0,
        )

    # --------------------------------------------------------
    # SEED COMPANIES / SITES
    # --------------------------------------------------------

    def ensure_company_site(
        self,
        company_id: str,
        site_id: str,
    ) -> None:
        self.insert_rows(
            SeedCompany,
            [{
                "company_id": company_id,
                "name": company_id,
            }],
        )

        self.insert_rows(
            SeedSite,
            [{
                "site_id": site_id,
                "company_id": company_id,
                "name": site_id,
            }],
        )

    # --------------------------------------------------------
    # SEED ASSETS
    # --------------------------------------------------------

    def seed_assets(
        self,
        path: Path,
    ) -> int:
        inserted = 0

        for batch in read_csv_batches(
            path,
            self.batch_size,
        ):
            companies = {}
            sites = {}
            rows = []

            for record in batch:
                company_id = record[
                    "company_id"
                ]

                site_id = record[
                    "site_id"
                ]

                companies[company_id] = {
                    "company_id": company_id,
                    "name": company_id,
                }

                sites[site_id] = {
                    "site_id": site_id,
                    "company_id": company_id,
                    "name": site_id,
                }

                rows.append({
                    "asset_id": record["asset_id"],
                    "company_id": company_id,
                    "site_id": site_id,
                    "equipment_type": record[
                        "equipment_type"
                    ],
                    "age_years": optional_float(
                        record.get("age_years")
                    ),
                    "initial_health_score": (
                        optional_float(
                            record.get(
                                "initial_health_score",
                                record.get(
                                    "condition_score"
                                ),
                            )
                        )
                    ),
                    "operating_hours_per_day": (
                        optional_float(
                            record.get(
                                "operating_hours_per_day"
                            )
                        )
                    ),
                    "criticality": record.get(
                        "criticality"
                    ),
                    "status": record.get(
                        "status"
                    ) or "active",
                })

            self.insert_rows(
                SeedCompany,
                list(companies.values()),
            )

            self.insert_rows(
                SeedSite,
                list(sites.values()),
            )

            inserted += self.insert_rows(
                SeedAsset,
                rows,
            )

        LOGGER.info(
            "Assets inserted: %d",
            inserted,
        )

        return inserted

    # --------------------------------------------------------
    # SEED TELEMETRY
    # --------------------------------------------------------

    def seed_telemetry(
        self,
        path: Path,
    ) -> int:
        inserted = 0

        numeric_columns = [
            "temperature_c",
            "vibration_mm_s",
            "pressure_bar",
            "current_a",
            "voltage_v",
            "rpm",
            "power_kw",
            "energy_kwh",
            "flow_rate_l_min",
            "humidity_percent",
            "acoustic_db",
            "health_score",
        ]

        for batch in read_csv_batches(
            path,
            self.batch_size,
        ):
            rows = []

            for record in batch:
                row = {
                    "reading_id": (
                        record["reading_id"]
                    ),
                    "timestamp": parse_datetime(
                        record["timestamp"]
                    ),
                    "asset_id": (
                        record["asset_id"]
                    ),
                    "company_id": (
                        record["company_id"]
                    ),
                    "site_id": (
                        record["site_id"]
                    ),
                    "anomaly_label": (
                        optional_int(
                            record.get(
                                "anomaly_label"
                            )
                        ) or 0
                    ),
                    "anomaly_type": (
                        record.get(
                            "anomaly_type"
                        )
                    ),
                }

                for column in numeric_columns:
                    row[column] = optional_float(
                        record.get(column)
                    )

                rows.append(row)

            inserted += self.insert_rows(
                SeedTelemetry,
                rows,
            )

        LOGGER.info(
            "Telemetry inserted: %d",
            inserted,
        )

        return inserted

    # --------------------------------------------------------
    # SEED FAILURES
    # --------------------------------------------------------

    def seed_failures(
        self,
        path: Path,
    ) -> int:
        inserted = 0

        for batch in read_csv_batches(
            path,
            self.batch_size,
        ):
            rows = []

            for record in batch:
                rows.append({
                    "failure_id": (
                        record["failure_id"]
                    ),
                    "timestamp": parse_datetime(
                        record["timestamp"]
                    ),
                    "asset_id": (
                        record["asset_id"]
                    ),
                    "company_id": (
                        record["company_id"]
                    ),
                    "site_id": (
                        record["site_id"]
                    ),
                    "failure_mode": (
                        record["failure_mode"]
                    ),
                    "severity": (
                        record["severity"]
                    ),
                    "downtime_hours": (
                        optional_float(
                            record.get(
                                "downtime_hours"
                            )
                        ) or 0
                    ),
                    "repair_cost": (
                        optional_float(
                            record.get(
                                "repair_cost"
                            )
                        ) or 0
                    ),
                    "currency": (
                        record.get("currency")
                        or "INR"
                    ),
                })

            inserted += self.insert_rows(
                SeedFailure,
                rows,
            )

        LOGGER.info(
            "Failures inserted: %d",
            inserted,
        )

        return inserted

    # --------------------------------------------------------
    # SEED MAINTENANCE
    # --------------------------------------------------------

    def seed_maintenance(
        self,
        path: Path,
    ) -> int:
        inserted = 0

        for batch in read_csv_batches(
            path,
            self.batch_size,
        ):
            rows = []

            for record in batch:
                rows.append({
                    "maintenance_id": (
                        record["maintenance_id"]
                    ),
                    "timestamp": parse_datetime(
                        record["timestamp"]
                    ),
                    "asset_id": (
                        record["asset_id"]
                    ),
                    "company_id": (
                        record["company_id"]
                    ),
                    "site_id": (
                        record["site_id"]
                    ),
                    "maintenance_type": (
                        record["maintenance_type"]
                    ),
                    "failure_id": (
                        record.get("failure_id")
                    ),
                    "health_before": (
                        optional_float(
                            record.get(
                                "health_before"
                            )
                        )
                    ),
                    "health_after": (
                        optional_float(
                            record.get(
                                "health_after"
                            )
                        )
                    ),
                    "estimated_cost": (
                        optional_float(
                            record.get(
                                "estimated_cost"
                            )
                        ) or 0
                    ),
                    "currency": (
                        record.get("currency")
                        or "INR"
                    ),
                })

            inserted += self.insert_rows(
                SeedMaintenance,
                rows,
            )

        LOGGER.info(
            "Maintenance inserted: %d",
            inserted,
        )

        return inserted

    # --------------------------------------------------------
    # SEED SAFETY EVENTS
    # --------------------------------------------------------

    def seed_safety_events(
        self,
        path: Path,
    ) -> int:
        inserted = 0

        for batch in read_csv_batches(
            path,
            self.batch_size,
        ):
            companies = {}
            sites = {}
            rows = []

            for record in batch:
                company_id = record[
                    "company_id"
                ]

                site_id = record[
                    "site_id"
                ]

                companies[company_id] = {
                    "company_id": company_id,
                    "name": company_id,
                }

                sites[site_id] = {
                    "site_id": site_id,
                    "company_id": company_id,
                    "name": site_id,
                }

                rows.append({
                    "event_id": (
                        record["event_id"]
                    ),
                    "timestamp": parse_datetime(
                        record["timestamp"]
                    ),
                    "company_id": company_id,
                    "site_id": site_id,
                    "zone_id": (
                        record["zone_id"]
                    ),
                    "event_type": (
                        record["event_type"]
                    ),
                    "severity": (
                        record["severity"]
                    ),
                    "risk_score": (
                        optional_float(
                            record.get(
                                "risk_score"
                            )
                        ) or 0
                    ),
                    "detection_source": (
                        record.get(
                            "detection_source"
                        )
                    ),
                    "incident_label": (
                        optional_int(
                            record.get(
                                "incident_label"
                            )
                        ) or 0
                    ),
                    "emergency_label": (
                        optional_int(
                            record.get(
                                "emergency_label"
                            )
                        ) or 0
                    ),
                    "status": (
                        record.get("status")
                        or "detected"
                    ),
                })

            self.insert_rows(
                SeedCompany,
                list(companies.values()),
            )

            self.insert_rows(
                SeedSite,
                list(sites.values()),
            )

            inserted += self.insert_rows(
                SeedSafetyEvent,
                rows,
            )

        LOGGER.info(
            "Safety events inserted: %d",
            inserted,
        )

        return inserted

    # --------------------------------------------------------
    # SEED SAFETY INCIDENTS
    # --------------------------------------------------------

    def seed_safety_incidents(
        self,
        path: Path,
    ) -> int:
        inserted = 0

        for batch in read_csv_batches(
            path,
            self.batch_size,
        ):
            rows = []

            for record in batch:
                rows.append({
                    "incident_id": (
                        record["incident_id"]
                    ),
                    "event_id": (
                        record["event_id"]
                    ),
                    "company_id": (
                        record["company_id"]
                    ),
                    "site_id": (
                        record["site_id"]
                    ),
                    "incident_type": (
                        record["incident_type"]
                    ),
                    "severity": (
                        record["severity"]
                    ),
                    "injury_count": (
                        optional_int(
                            record.get(
                                "injury_count"
                            )
                        ) or 0
                    ),
                    "downtime_hours": (
                        optional_float(
                            record.get(
                                "downtime_hours"
                            )
                        ) or 0
                    ),
                    "estimated_cost": (
                        optional_float(
                            record.get(
                                "estimated_cost"
                            )
                        ) or 0
                    ),
                    "currency": (
                        record.get("currency")
                        or "INR"
                    ),
                })

            inserted += self.insert_rows(
                SeedSafetyIncident,
                rows,
            )

        LOGGER.info(
            "Safety incidents inserted: %d",
            inserted,
        )

        return inserted

    # --------------------------------------------------------
    # SEED COSTS
    # --------------------------------------------------------

    def seed_costs(
        self,
        path: Path,
    ) -> int:
        inserted = 0

        for batch in read_csv_batches(
            path,
            self.batch_size,
        ):
            rows = []

            for record in batch:
                rows.append({
                    "event_id": (
                        record["event_id"]
                    ),
                    "asset_id": (
                        record["asset_id"]
                    ),
                    "company_id": (
                        record["company_id"]
                    ),
                    "site_id": (
                        record["site_id"]
                    ),
                    "maintenance_type": (
                        record.get(
                            "maintenance_type"
                        )
                    ),
                    "labor_cost": (
                        optional_float(
                            record.get(
                                "labor_cost"
                            )
                        ) or 0
                    ),
                    "spare_parts_cost": (
                        optional_float(
                            record.get(
                                "spare_parts_cost"
                            )
                        ) or 0
                    ),
                    "service_cost": (
                        optional_float(
                            record.get(
                                "service_cost"
                            )
                        ) or 0
                    ),
                    "inspection_cost": (
                        optional_float(
                            record.get(
                                "inspection_cost"
                            )
                        ) or 0
                    ),
                    "direct_maintenance_cost": (
                        optional_float(
                            record.get(
                                "direct_maintenance_cost"
                            )
                        ) or 0
                    ),
                    "total_event_cost": (
                        optional_float(
                            record.get(
                                "total_event_cost"
                            )
                        ) or 0
                    ),
                    "currency": (
                        record.get("currency")
                        or "INR"
                    ),
                })

            inserted += self.insert_rows(
                SeedCost,
                rows,
            )

        LOGGER.info(
            "Cost records inserted: %d",
            inserted,
        )

        return inserted

    # --------------------------------------------------------
    # SEED ALL
    # --------------------------------------------------------

    def seed_all(
        self,
        data_dir: Path,
    ) -> dict[str, int]:
        """
        Important:
        Assets must exist before telemetry,
        failures, maintenance and cost records.

        Generated datasets from different scripts
        may use overlapping asset IDs. Use one
        canonical asset registry for a combined
        seed, or separate database instances.
        """

        telemetry_dir = (
            data_dir / "telemetry"
        )

        failures_dir = (
            data_dir / "failures"
        )

        safety_dir = (
            data_dir / "safety"
        )

        costs_dir = (
            data_dir / "costs"
        )

        counts = {}

        # Canonical asset registry:
        # Prefer telemetry assets.
        asset_path = (
            telemetry_dir / "assets.csv"
        )

        if not asset_path.exists():
            asset_path = (
                failures_dir / "assets.csv"
            )

        if not asset_path.exists():
            asset_path = (
                costs_dir / "asset_financials.csv"
            )

        counts["assets"] = self.seed_assets(
            asset_path
        )

        counts["telemetry"] = (
            self.seed_telemetry(
                telemetry_dir / "telemetry.csv"
            )
        )

        counts["failures"] = (
            self.seed_failures(
                failures_dir / "failure_events.csv"
            )
        )

        counts["maintenance"] = (
            self.seed_maintenance(
                failures_dir / "maintenance_history.csv"
            )
        )

        counts["safety_events"] = (
            self.seed_safety_events(
                safety_dir / "safety_events.csv"
            )
        )

        counts["safety_incidents"] = (
            self.seed_safety_incidents(
                safety_dir / "safety_incidents.csv"
            )
        )

        counts["costs"] = (
            self.seed_costs(
                costs_dir / "maintenance_costs.csv"
            )
        )

        return counts

    # --------------------------------------------------------
    # DATABASE COUNTS
    # --------------------------------------------------------

    def database_counts(
        self,
    ) -> dict[str, int]:
        models = {
            "companies": SeedCompany,
            "sites": SeedSite,
            "assets": SeedAsset,
            "telemetry": SeedTelemetry,
            "failures": SeedFailure,
            "maintenance": SeedMaintenance,
            "safety_events": SeedSafetyEvent,
            "safety_incidents": SeedSafetyIncident,
            "costs": SeedCost,
        }

        counts = {}

        with self.SessionLocal() as session:
            for name, model in models.items():
                counts[name] = session.scalar(
                    select(
                        func.count()
                    ).select_from(model)
                ) or 0

        return counts

    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    def close(self) -> None:
        self.engine.dispose()


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Seed the industrial AI "
            "development database"
        )
    )

    parser.add_argument(
        "--database-url",
        default=os.getenv(
            "DATABASE_URL",
            "sqlite:///./industrial_dev.db",
        ),
    )

    parser.add_argument(
        "--data-dir",
        default="data/synthetic",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--init-db",
        action="store_true",
    )

    parser.add_argument(
        "--seed-all",
        action="store_true",
    )

    parser.add_argument(
        "--counts",
        action="store_true",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=(
            logging.DEBUG
            if args.verbose
            else logging.INFO
        ),
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    if not (
        args.init_db
        or args.seed_all
        or args.counts
    ):
        print(
            "Specify --init-db, --seed-all "
            "or --counts"
        )
        return

    seeder = IndustrialDatabaseSeeder(
        database_url=args.database_url,
        batch_size=args.batch_size,
    )

    try:
        if args.init_db or args.seed_all:
            seeder.initialize_database()

        if args.seed_all:
            counts = seeder.seed_all(
                Path(args.data_dir)
            )

            print("\nInserted records:")

            for name, count in counts.items():
                print(
                    f"  {name}: {count}"
                )

        if args.counts or args.seed_all:
            counts = seeder.database_counts()

            print("\nDatabase totals:")

            for name, count in counts.items():
                print(
                    f"  {name}: {count}"
                )

    finally:
        seeder.close()


if __name__ == "__main__":
    main()