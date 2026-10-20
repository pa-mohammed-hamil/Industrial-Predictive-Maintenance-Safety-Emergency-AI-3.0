# app/config/database.py

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)


# =========================================================
# Database configuration
# =========================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/industrial_ai",
)

SQL_ECHO = os.getenv(
    "SQL_ECHO",
    "false",
).lower() == "true"

DB_POOL_SIZE = int(
    os.getenv("DB_POOL_SIZE", "10")
)

DB_MAX_OVERFLOW = int(
    os.getenv("DB_MAX_OVERFLOW", "20")
)

DB_POOL_TIMEOUT = int(
    os.getenv("DB_POOL_TIMEOUT", "30")
)

DB_POOL_RECYCLE = int(
    os.getenv("DB_POOL_RECYCLE", "1800")
)


# =========================================================
# SQLAlchemy base
# =========================================================

class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy ORM models.

    Example:

        from app.config.database import Base

        class Asset(Base):
            __tablename__ = "assets"
    """

    pass


# =========================================================
# Engine creation
# =========================================================

def create_database_engine(
    database_url: str = DATABASE_URL,
) -> AsyncEngine:
    url = make_url(database_url)

    kwargs = {
        "echo": SQL_ECHO,
        "pool_pre_ping": True,
    }

    # SQLite does not support PostgreSQL-style
    # pool configuration.
    if not url.drivername.startswith("sqlite"):
        kwargs.update(
            {
                "pool_size": DB_POOL_SIZE,
                "max_overflow": DB_MAX_OVERFLOW,
                "pool_timeout": DB_POOL_TIMEOUT,
                "pool_recycle": DB_POOL_RECYCLE,
            }
        )

    return create_async_engine(
        database_url,
        **kwargs,
    )


engine = create_database_engine()


# =========================================================
# Session factory
# =========================================================

AsyncSessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# =========================================================
# FastAPI dependency
# =========================================================

async def get_db() -> AsyncGenerator[
    AsyncSession,
    None,
]:
    """
    FastAPI database-session dependency.

    Example:

        @router.get("/assets")
        async def assets(
            db: AsyncSession = Depends(get_db),
        ):
            ...
    """

    async with AsyncSessionFactory() as session:
        try:
            yield session

        except Exception:
            await session.rollback()

            logger.exception(
                "Database transaction failed"
            )

            raise

        finally:
            await session.close()


# =========================================================
# Transaction context
# =========================================================

@asynccontextmanager
async def database_session() -> AsyncGenerator[
    AsyncSession,
    None,
]:
    """
    Transaction-aware session for services,
    workers and scripts.

    Automatically commits on success and
    rolls back on failure.

    Example:

        async with database_session() as db:
            db.add(asset)
    """

    async with AsyncSessionFactory() as session:
        try:
            yield session

            await session.commit()

        except Exception:
            await session.rollback()

            logger.exception(
                "Database transaction rolled back"
            )

            raise

        finally:
            await session.close()


# =========================================================
# Health check
# =========================================================

async def database_health_check() -> dict:
    try:
        async with engine.connect() as connection:
            result = await connection.execute(
                text("SELECT 1")
            )

            value = result.scalar_one()

        return {
            "status": "healthy",
            "connected": value == 1,
        }

    except SQLAlchemyError as exc:
        logger.exception(
            "Database health check failed"
        )

        return {
            "status": "unhealthy",
            "connected": False,
            "error": str(exc),
        }


# =========================================================
# Initialize tables
# =========================================================

async def init_database() -> None:
    """
    Create ORM tables.

    Development/testing helper only.

    Production schema changes should normally
    be managed using Alembic migrations.
    """

    logger.info(
        "Initializing database tables"
    )

    async with engine.begin() as connection:
        await connection.run_sync(
            Base.metadata.create_all
        )

    logger.info(
        "Database initialization complete"
    )


# =========================================================
# Drop tables
# =========================================================

async def drop_database() -> None:
    """
    Drop all ORM tables.

    Intended only for automated tests or
    explicit local-development reset scripts.
    """

    environment = os.getenv(
        "APP_ENV",
        "development",
    ).lower()

    if environment in {
        "production",
        "prod",
    }:
        raise RuntimeError(
            "Refusing to drop database tables "
            "in production."
        )

    logger.warning(
        "Dropping all database tables"
    )

    async with engine.begin() as connection:
        await connection.run_sync(
            Base.metadata.drop_all
        )


# =========================================================
# Connection test
# =========================================================

async def test_database_connection() -> bool:
    try:
        async with engine.connect() as connection:
            await connection.execute(
                text("SELECT 1")
            )

        logger.info(
            "Database connection successful"
        )

        return True

    except Exception:
        logger.exception(
            "Database connection failed"
        )

        return False


# =========================================================
# Shutdown
# =========================================================

async def close_database() -> None:
    logger.info(
        "Closing database connection pool"
    )

    await engine.dispose()

    logger.info(
        "Database pool closed"
    )