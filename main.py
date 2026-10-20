from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("industrial-ai")

BASE_DIR = Path(__file__).resolve().parent
# Frontend is in parent directory
FRONTEND_DIR = BASE_DIR.parent / "frontend"

APP_NAME = "Industrial Predictive Maintenance & Safety AI"
APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s", APP_NAME)

    # Initialize shared resources here:
    #
    # PostgreSQL
    # Redis
    # Kafka
    # MQTT
    # OPC-UA
    # ML model registry
    # RAG/vector DB
    # background workers

    yield

    logger.info("Shutting down %s", APP_NAME)

    # Close shared resources here.


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Industrial predictive maintenance, safety, "
        "emergency intelligence and cost analytics platform."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
):
    logger.exception(
        "Unhandled error: %s %s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "internal_server_error",
            "message": "Unexpected server error.",
        },
    )


# ---------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------

@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "healthy",
        "service": APP_NAME,
        "version": APP_VERSION,
    }


@app.get("/api/v1/system/status", tags=["System"])
async def system_status():
    return {
        "success": True,
        "backend": "online",
        "frontend": FRONTEND_DIR.exists(),
        "version": APP_VERSION,
    }


# ---------------------------------------------------------
# Demo dashboard endpoint
# Frontend can call /api/v1/dashboard
# ---------------------------------------------------------

@app.get("/api/v1/dashboard", tags=["Dashboard"])
async def dashboard():
    return {
        "success": True,
        "data": {
            "assets": 24,
            "healthy_assets": 19,
            "warning_assets": 4,
            "critical_assets": 1,
            "open_incidents": 3,
            "active_alerts": 7,
            "maintenance_due": 5,
            "average_health_score": 87.4,
        },
    }


# ---------------------------------------------------------
# Demo telemetry endpoint
# ---------------------------------------------------------

@app.get("/api/v1/telemetry/{asset_id}", tags=["Telemetry"])
async def telemetry(asset_id: str):
    return {
        "success": True,
        "asset_id": asset_id,
        "telemetry": {
            "temperature_c": 62.4,
            "vibration_rms": 2.8,
            "pressure_bar": 5.3,
            "current_a": 48.7,
            "power_kw": 31.2,
            "energy_kwh": 384.6,
            "health_score": 91.0,
        },
    }


# ---------------------------------------------------------
# Include real project routers here
# ---------------------------------------------------------

def register_routers(application: FastAPI) -> None:
    """
    Import your actual routers here as they are implemented.

    Example:

    from app.api.v1.router import api_router

    application.include_router(
        api_router,
        prefix="/api/v1",
    )
    """

    try:
        from app.api.v1.router import api_router

        application.include_router(
            api_router,
            prefix="/api/v1",
        )

        logger.info("API v1 router registered.")

    except ImportError as exc:
        # Keeps development server runnable while
        # individual project modules are being built.
        logger.warning(
            "API router not loaded: %s",
            exc,
        )


register_routers(app)


# ---------------------------------------------------------
# Static frontend directories
# ---------------------------------------------------------

STATIC_DIR = FRONTEND_DIR / "static"
CSS_DIR = FRONTEND_DIR / "css"
JS_DIR = FRONTEND_DIR / "js"
ASSETS_DIR = FRONTEND_DIR / "assets"

if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=STATIC_DIR),
        name="static",
    )

if CSS_DIR.exists():
    app.mount(
        "/css",
        StaticFiles(directory=CSS_DIR),
        name="css",
    )

if JS_DIR.exists():
    app.mount(
        "/js",
        StaticFiles(directory=JS_DIR),
        name="js",
    )

if ASSETS_DIR.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=ASSETS_DIR),
        name="assets",
    )


# ---------------------------------------------------------
# Frontend index.html
# IMPORTANT:
# Put this AFTER API routes.
# ---------------------------------------------------------

@app.get(
    "/",
    response_class=FileResponse,
    include_in_schema=False,
)
async def frontend_index():
    index_file = FRONTEND_DIR / "index.html"

    if not index_file.exists():
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "message": (
                    "frontend/index.html was not found."
                ),
            },
        )

    return FileResponse(index_file)


# ---------------------------------------------------------
# Frontend page routing
#
# /dashboard.html
# /assets/asset-details.html
# /safety/safety-dashboard.html
# etc.
# ---------------------------------------------------------

@app.get(
    "/{page_name}.html",
    response_class=FileResponse,
    include_in_schema=False,
)
async def frontend_page(page_name: str):
    # Prevent path traversal.
    safe_name = Path(page_name).name

    page = FRONTEND_DIR / f"{safe_name}.html"

    if not page.exists():
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "message": "Frontend page not found.",
            },
        )

    return FileResponse(page)


@app.get(
    "/{folder}/{page_name}.html",
    response_class=FileResponse,
    include_in_schema=False,
)
async def frontend_subfolder_page(folder: str, page_name: str):
    # Prevent path traversal.
    safe_folder = Path(folder).name
    safe_page = Path(page_name).name

    page = FRONTEND_DIR / safe_folder / f"{safe_page}.html"

    if not page.exists():
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "message": f"Frontend page not found: /{safe_folder}/{safe_page}.html",
            },
        )

    return FileResponse(page)


# ---------------------------------------------------------
# Development runner
# ---------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv(
            "HOST",
            "0.0.0.0",
        ),
        port=int(
            os.getenv(
                "PORT",
                "8000",
            )
        ),
        reload=(
            os.getenv(
                "RELOAD",
                "true",
            ).lower()
            == "true"
        ),
        log_level="info",
    )