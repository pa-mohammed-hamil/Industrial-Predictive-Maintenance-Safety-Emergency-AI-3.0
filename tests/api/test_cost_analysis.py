
"""
INDUSTRA AI
backend/tests/test_assets.py

Industrial Asset Management API Tests

Coverage:
- Asset CRUD
- Asset data validation
- Asset health
- Component health
- Asset inspections
- Asset lifecycle
- Asset valuation
- Asset history
- Asset uploads
- Filtering and pagination
- Error handling

Run:
    pip install pytest fastapi httpx
    pytest backend/tests/test_assets.py -v

Integration:
    Replace the demo_app fixture with your production FastAPI app.
"""

from __future__ import annotations

import copy
import io
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field, field_validator


# ============================================================
# MODELS
# ============================================================

class AssetCreate(BaseModel):
    asset_id: str = Field(min_length=3, max_length=50)
    name: str = Field(min_length=2, max_length=150)
    plant: str = Field(min_length=2)
    category: str = Field(min_length=2)
    manufacturer: str = Field(min_length=2)
    acquisition_cost: float = Field(ge=0)
    health_score: float = Field(ge=0, le=100)
    remaining_useful_life: int = Field(ge=0)
    annual_failure_probability: float = Field(ge=0, le=1)
    status: str = "operational"

    @field_validator("asset_id")
    @classmethod
    def validate_asset_id(cls, value: str) -> str:
        if not value.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Invalid asset identifier")
        return value


class AssetUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=2)
    plant: str | None = Field(default=None, min_length=2)
    health_score: float | None = Field(default=None, ge=0, le=100)
    remaining_useful_life: int | None = Field(default=None, ge=0)
    status: str | None = None


class InspectionCreate(BaseModel):
    inspection_type: str = Field(min_length=2)
    inspector: str = Field(min_length=2)
    condition: str
    notes: str = ""
    passed: bool


class ComponentCreate(BaseModel):
    component_id: str = Field(min_length=2)
    name: str = Field(min_length=2)
    health_score: float = Field(ge=0, le=100)
    temperature: float | None = None
    vibration: float | None = Field(default=None, ge=0)


class ValuationRequest(BaseModel):
    useful_life_years: float = Field(gt=0)
    age_years: float = Field(ge=0)
    salvage_value: float = Field(ge=0)


# ============================================================
# TEST DATA
# ============================================================

SAMPLE_ASSETS = [
    {
        "asset_id": "MTR-001",
        "name": "Main Drive Motor",
        "plant": "Plant A",
        "category": "Electric Motor",
        "manufacturer": "Siemens",
        "acquisition_cost": 520000,
        "health_score": 74,
        "remaining_useful_life": 48,
        "annual_failure_probability": 0.32,
        "status": "operational",
    },
    {
        "asset_id": "GEN-001",
        "name": "Power Generator",
        "plant": "Plant B",
        "category": "Generator",
        "manufacturer": "Cummins",
        "acquisition_cost": 1100000,
        "health_score": 52,
        "remaining_useful_life": 24,
        "annual_failure_probability": 0.43,
        "status": "maintenance_required",
    },
    {
        "asset_id": "PMP-001",
        "name": "Cooling Pump",
        "plant": "Plant B",
        "category": "Pump",
        "manufacturer": "KSB",
        "acquisition_cost": 280000,
        "health_score": 84,
        "remaining_useful_life": 69,
        "annual_failure_probability": 0.20,
        "status": "operational",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_health(score: float) -> str:
    if score < 65:
        return "critical"
    if score < 80:
        return "warning"
    return "healthy"


def maintenance_priority(asset: dict[str, Any]) -> str:
    if (
        asset["health_score"] < 65
        or asset["annual_failure_probability"] > 0.35
    ):
        return "high"

    if (
        asset["health_score"] < 80
        or asset["annual_failure_probability"] > 0.25
    ):
        return "medium"

    return "low"


# ============================================================
# ISOLATED DEMONSTRATION API
# ============================================================

@pytest.fixture
def demo_app() -> FastAPI:
    """
    In-memory demonstration API.

    IMPORTANT:
    This fixture is a test double, not the production asset API.
    For real integration testing, import the actual FastAPI app
    and override its database dependencies with a test database.
    """

    app = FastAPI(title="INDUSTRA AI Asset Test API")

    assets: dict[str, dict[str, Any]] = {
        item["asset_id"]: copy.deepcopy(item)
        for item in SAMPLE_ASSETS
    }

    inspections: dict[str, list] = {}
    components: dict[str, list] = {}
    history: dict[str, list] = {}
    uploads: dict[str, list] = {}

    def require_asset(asset_id: str) -> dict[str, Any]:
        asset = assets.get(asset_id)

        if asset is None:
            raise HTTPException(
                status_code=404,
                detail="Asset not found",
            )

        return asset

    def add_history(
        asset_id: str,
        action: str,
        details: dict | None = None,
    ) -> None:
        history.setdefault(asset_id, []).append(
            {
                "timestamp": utc_now(),
                "action": action,
                "details": details or {},
            }
        )

    @app.get("/api/v1/assets")
    def list_assets(
        plant: str | None = None,
        status: str | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
    ):
        results = list(assets.values())

        if plant is not None:
            results = [
                item for item in results
                if item["plant"] == plant
            ]

        if status is not None:
            results = [
                item for item in results
                if item["status"] == status
            ]

        return {
            "total": len(results),
            "items": results[offset:offset + limit],
        }

    @app.post("/api/v1/assets", status_code=201)
    def create_asset(payload: AssetCreate):
        data = payload.model_dump()
        asset_id = data["asset_id"]

        if asset_id in assets:
            raise HTTPException(
                status_code=409,
                detail="Asset already exists",
            )

        assets[asset_id] = data
        add_history(asset_id, "asset_created")

        return data

    @app.get("/api/v1/assets/{asset_id}")
    def get_asset(asset_id: str):
        return require_asset(asset_id)

    @app.patch("/api/v1/assets/{asset_id}")
    def update_asset(
        asset_id: str,
        payload: AssetUpdate,
    ):
        asset = require_asset(asset_id)

        updates = payload.model_dump(exclude_unset=True)

        asset.update(updates)

        add_history(
            asset_id,
            "asset_updated",
            updates,
        )

        return asset

    @app.delete("/api/v1/assets/{asset_id}")
    def delete_asset(asset_id: str):
        require_asset(asset_id)

        del assets[asset_id]

        return {
            "deleted": True,
            "asset_id": asset_id,
        }

    @app.get("/api/v1/assets/{asset_id}/health")
    def get_asset_health(asset_id: str):
        asset = require_asset(asset_id)

        return {
            "asset_id": asset_id,
            "health_score": asset["health_score"],
            "classification": classify_health(
                asset["health_score"]
            ),
            "remaining_useful_life": asset[
                "remaining_useful_life"
            ],
            "annual_failure_probability": asset[
                "annual_failure_probability"
            ],
            "maintenance_priority": maintenance_priority(asset),
        }

    @app.post(
        "/api/v1/assets/{asset_id}/components",
        status_code=201,
    )
    def create_component(
        asset_id: str,
        payload: ComponentCreate,
    ):
        require_asset(asset_id)

        component = payload.model_dump()
        existing = components.setdefault(asset_id, [])

        if any(
            item["component_id"] == component["component_id"]
            for item in existing
        ):
            raise HTTPException(
                status_code=409,
                detail="Component already exists",
            )

        existing.append(component)

        add_history(
            asset_id,
            "component_added",
            {"component_id": component["component_id"]},
        )

        return component

    @app.get("/api/v1/assets/{asset_id}/components")
    def list_components(asset_id: str):
        require_asset(asset_id)

        return {
            "items": components.get(asset_id, []),
        }

    @app.post(
        "/api/v1/assets/{asset_id}/inspections",
        status_code=201,
    )
    def create_inspection(
        asset_id: str,
        payload: InspectionCreate,
    ):
        require_asset(asset_id)

        inspection = {
            "inspection_id": (
                f"INS-{len(inspections.get(asset_id, [])) + 1:04d}"
            ),
            "asset_id": asset_id,
            "created_at": utc_now(),
            **payload.model_dump(),
        }

        inspections.setdefault(asset_id, []).append(
            inspection
        )

        add_history(
            asset_id,
            "inspection_created",
            {
                "inspection_id": inspection["inspection_id"],
                "passed": inspection["passed"],
            },
        )

        return inspection

    @app.get("/api/v1/assets/{asset_id}/inspections")
    def list_inspections(asset_id: str):
        require_asset(asset_id)

        return {
            "items": inspections.get(asset_id, []),
        }

    @app.get("/api/v1/assets/{asset_id}/lifecycle")
    def get_lifecycle(asset_id: str):
        asset = require_asset(asset_id)

        return {
            "asset_id": asset_id,
            "stage": (
                "replacement_planning"
                if asset["remaining_useful_life"] < 36
                else "operation"
            ),
            "remaining_useful_life_months": asset[
                "remaining_useful_life"
            ],
            "acquisition_cost": asset["acquisition_cost"],
        }

    @app.post("/api/v1/assets/{asset_id}/valuation")
    def calculate_valuation(
        asset_id: str,
        payload: ValuationRequest,
    ):
        asset = require_asset(asset_id)

        acquisition = asset["acquisition_cost"]

        if payload.salvage_value > acquisition:
            raise HTTPException(
                status_code=422,
                detail="Salvage value cannot exceed acquisition cost",
            )

        depreciable = acquisition - payload.salvage_value

        annual_depreciation = (
            depreciable / payload.useful_life_years
        )

        accumulated = min(
            payload.age_years,
            payload.useful_life_years,
        ) * annual_depreciation

        book_value = max(
            payload.salvage_value,
            acquisition - accumulated,
        )

        return {
            "asset_id": asset_id,
            "acquisition_cost": acquisition,
            "annual_depreciation": annual_depreciation,
            "accumulated_depreciation": accumulated,
            "book_value": book_value,
            "method": "straight_line",
        }

    @app.get("/api/v1/assets/{asset_id}/history")
    def get_history(asset_id: str):
        require_asset(asset_id)

        return {
            "items": history.get(asset_id, []),
        }

    @app.post(
        "/api/v1/assets/{asset_id}/upload",
        status_code=201,
    )
    async def upload_asset_file(
        asset_id: str,
        file: UploadFile = File(...),
    ):
        require_asset(asset_id)

        allowed = {
            "application/pdf",
            "image/png",
            "image/jpeg",
        }

        if file.content_type not in allowed:
            raise HTTPException(
                status_code=415,
                detail="Unsupported file type",
            )

        content = await file.read()

        if len(content) > 2 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="File exceeds 2 MB",
            )

        record = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content),
        }

        uploads.setdefault(asset_id, []).append(record)

        add_history(
            asset_id,
            "file_uploaded",
            {"filename": file.filename},
        )

        return record

    @app.get("/api/v1/assets/{asset_id}/uploads")
    def list_uploads(asset_id: str):
        require_asset(asset_id)

        return {
            "items": uploads.get(asset_id, []),
        }

    return app


@pytest.fixture
def client(demo_app: FastAPI) -> TestClient:
    with TestClient(demo_app) as test_client:
        yield test_client


# ============================================================
# ASSET CRUD TESTS
# ============================================================

def test_list_assets(client):
    response = client.get("/api/v1/assets")

    assert response.status_code == 200
    assert response.json()["total"] == 3


def test_get_existing_asset(client):
    response = client.get("/api/v1/assets/MTR-001")

    assert response.status_code == 200
    assert response.json()["name"] == "Main Drive Motor"


def test_get_nonexistent_asset(client):
    response = client.get("/api/v1/assets/UNKNOWN")

    assert response.status_code == 404


def test_create_asset(client):
    payload = {
        **SAMPLE_ASSETS[0],
        "asset_id": "NEW-001",
        "name": "New Industrial Motor",
    }

    response = client.post("/api/v1/assets", json=payload)

    assert response.status_code == 201
    assert response.json()["asset_id"] == "NEW-001"

    get_response = client.get("/api/v1/assets/NEW-001")

    assert get_response.status_code == 200


def test_duplicate_asset_rejected(client):
    response = client.post(
        "/api/v1/assets",
        json=SAMPLE_ASSETS[0],
    )

    assert response.status_code == 409


def test_update_asset(client):
    response = client.patch(
        "/api/v1/assets/MTR-001",
        json={"health_score": 82},
    )

    assert response.status_code == 200
    assert response.json()["health_score"] == 82


def test_delete_asset(client):
    response = client.delete("/api/v1/assets/PMP-001")

    assert response.status_code == 200

    response = client.get("/api/v1/assets/PMP-001")

    assert response.status_code == 404


def test_delete_missing_asset(client):
    response = client.delete("/api/v1/assets/UNKNOWN")

    assert response.status_code == 404


# ============================================================
# ASSET VALIDATION TESTS
# ============================================================

@pytest.mark.parametrize(
    "health_score",
    [-1, 101, 150],
)
def test_invalid_health_score(client, health_score):
    payload = {
        **SAMPLE_ASSETS[0],
        "asset_id": "TEST-001",
        "health_score": health_score,
    }

    response = client.post("/api/v1/assets", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "probability",
    [-0.1, 1.1, 2],
)
def test_invalid_failure_probability(client, probability):
    payload = {
        **SAMPLE_ASSETS[0],
        "asset_id": "TEST-002",
        "annual_failure_probability": probability,
    }

    response = client.post("/api/v1/assets", json=payload)

    assert response.status_code == 422


def test_negative_acquisition_cost_rejected(client):
    payload = {
        **SAMPLE_ASSETS[0],
        "asset_id": "TEST-003",
        "acquisition_cost": -1000,
    }

    response = client.post("/api/v1/assets", json=payload)

    assert response.status_code == 422


def test_negative_rul_rejected(client):
    payload = {
        **SAMPLE_ASSETS[0],
        "asset_id": "TEST-004",
        "remaining_useful_life": -12,
    }

    response = client.post("/api/v1/assets", json=payload)

    assert response.status_code == 422


def test_invalid_asset_identifier(client):
    payload = {
        **SAMPLE_ASSETS[0],
        "asset_id": "INVALID ID!",
    }

    response = client.post("/api/v1/assets", json=payload)

    assert response.status_code == 422


def test_missing_required_name(client):
    payload = {
        **SAMPLE_ASSETS[0],
        "asset_id": "TEST-005",
    }

    del payload["name"]

    response = client.post("/api/v1/assets", json=payload)

    assert response.status_code == 422


def test_invalid_update_rejected(client):
    response = client.patch(
        "/api/v1/assets/MTR-001",
        json={"health_score": 200},
    )

    assert response.status_code == 422


# ============================================================
# FILTERING AND PAGINATION
# ============================================================

def test_filter_assets_by_plant(client):
    response = client.get(
        "/api/v1/assets",
        params={"plant": "Plant B"},
    )

    assert response.status_code == 200
    assert response.json()["total"] == 2


def test_filter_assets_by_status(client):
    response = client.get(
        "/api/v1/assets",
        params={"status": "operational"},
    )

    assert response.status_code == 200
    assert response.json()["total"] == 2


def test_asset_pagination(client):
    response = client.get(
        "/api/v1/assets",
        params={"limit": 1, "offset": 1},
    )

    assert response.status_code == 200
    assert len(response.json()["items"]) == 1


def test_invalid_pagination_limit(client):
    response = client.get(
        "/api/v1/assets",
        params={"limit": 0},
    )

    assert response.status_code == 422


# ============================================================
# ASSET HEALTH TESTS
# ============================================================

def test_healthy_asset(client):
    response = client.get(
        "/api/v1/assets/PMP-001/health"
    )

    assert response.status_code == 200
    assert response.json()["classification"] == "healthy"


def test_warning_asset(client):
    response = client.get(
        "/api/v1/assets/MTR-001/health"
    )

    assert response.status_code == 200
    assert response.json()["classification"] == "warning"


def test_critical_asset(client):
    response = client.get(
        "/api/v1/assets/GEN-001/health"
    )

    assert response.status_code == 200
    assert response.json()["classification"] == "critical"


@pytest.mark.parametrize(
    "score,expected",
    [
        (0, "critical"),
        (64.9, "critical"),
        (65, "warning"),
        (79.9, "warning"),
        (80, "healthy"),
        (100, "healthy"),
    ],
)
def test_health_classification_boundaries(score, expected):
    assert classify_health(score) == expected


def test_high_maintenance_priority(client):
    response = client.get(
        "/api/v1/assets/GEN-001/health"
    )

    assert response.json()["maintenance_priority"] == "high"


def test_health_endpoint_missing_asset(client):
    response = client.get(
        "/api/v1/assets/UNKNOWN/health"
    )

    assert response.status_code == 404


# ============================================================
# COMPONENT HEALTH TESTS
# ============================================================

def test_create_component(client):
    payload = {
        "component_id": "BRG-001",
        "name": "Motor Bearing",
        "health_score": 72,
        "temperature": 68,
        "vibration": 3.4,
    }

    response = client.post(
        "/api/v1/assets/MTR-001/components",
        json=payload,
    )

    assert response.status_code == 201
    assert response.json()["component_id"] == "BRG-001"


def test_list_components(client):
    payload = {
        "component_id": "SNS-001",
        "name": "Temperature Sensor",
        "health_score": 95,
    }

    client.post(
        "/api/v1/assets/MTR-001/components",
        json=payload,
    )

    response = client.get(
        "/api/v1/assets/MTR-001/components"
    )

    assert response.status_code == 200
    assert len(response.json()["items"]) == 1


def test_duplicate_component_rejected(client):
    payload = {
        "component_id": "BRG-002",
        "name": "Bearing",
        "health_score": 80,
    }

    client.post(
        "/api/v1/assets/MTR-001/components",
        json=payload,
    )

    response = client.post(
        "/api/v1/assets/MTR-001/components",
        json=payload,
    )

    assert response.status_code == 409


def test_invalid_component_health(client):
    payload = {
        "component_id": "BRG-003",
        "name": "Bearing",
        "health_score": 120,
    }

    response = client.post(
        "/api/v1/assets/MTR-001/components",
        json=payload,
    )

    assert response.status_code == 422


def test_component_for_missing_asset(client):
    response = client.get(
        "/api/v1/assets/UNKNOWN/components"
    )

    assert response.status_code == 404


# ============================================================
# ASSET INSPECTION TESTS
# ============================================================

def test_create_asset_inspection(client):
    payload = {
        "inspection_type": "Visual Inspection",
        "inspector": "Maintenance Engineer",
        "condition": "Acceptable",
        "notes": "No visible damage",
        "passed": True,
    }

    response = client.post(
        "/api/v1/assets/MTR-001/inspections",
        json=payload,
    )

    assert response.status_code == 201
    assert response.json()["passed"] is True


def test_list_asset_inspections(client):
    payload = {
        "inspection_type": "Safety Inspection",
        "inspector": "Safety Officer",
        "condition": "Needs Review",
        "passed": False,
    }

    client.post(
        "/api/v1/assets/GEN-001/inspections",
        json=payload,
    )

    response = client.get(
        "/api/v1/assets/GEN-001/inspections"
    )

    assert response.status_code == 200
    assert len(response.json()["items"]) == 1


def test_inspection_missing_required_field(client):
    response = client.post(
        "/api/v1/assets/MTR-001/inspections",
        json={
            "inspection_type": "Visual",
            "condition": "Good",
            "passed": True,
        },
    )

    assert response.status_code == 422


def test_inspection_for_unknown_asset(client):
    response = client.get(
        "/api/v1/assets/UNKNOWN/inspections"
    )

    assert response.status_code == 404


# ============================================================
# ASSET LIFECYCLE TESTS
# ============================================================

def test_asset_lifecycle_operation(client):
    response = client.get(
        "/api/v1/assets/MTR-001/lifecycle"
    )

    assert response.status_code == 200
    assert response.json()["stage"] == "operation"


def test_asset_lifecycle_replacement_planning(client):
    response = client.get(
        "/api/v1/assets/GEN-001/lifecycle"
    )

    assert response.status_code == 200
    assert response.json()["stage"] == "replacement_planning"


def test_lifecycle_rul_consistency(client):
    response = client.get(
        "/api/v1/assets/PMP-001/lifecycle"
    )

    assert response.json()["remaining_useful_life_months"] == 69


def test_lifecycle_unknown_asset(client):
    response = client.get(
        "/api/v1/assets/UNKNOWN/lifecycle"
    )

    assert response.status_code == 404


# ============================================================
# ASSET VALUATION TESTS
# ============================================================

def test_straight_line_depreciation(client):
    response = client.post(
        "/api/v1/assets/MTR-001/valuation",
        json={
            "useful_life_years": 10,
            "age_years": 4,
            "salvage_value": 20000,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["annual_depreciation"] == pytest.approx(50000)
    assert data["accumulated_depreciation"] == pytest.approx(200000)
    assert data["book_value"] == pytest.approx(320000)


def test_depreciation_does_not_go_below_salvage(client):
    response = client.post(
        "/api/v1/assets/MTR-001/valuation",
        json={
            "useful_life_years": 10,
            "age_years": 20,
            "salvage_value": 20000,
        },
    )

    assert response.status_code == 200
    assert response.json()["book_value"] == pytest.approx(20000)


def test_zero_useful_life_rejected(client):
    response = client.post(
        "/api/v1/assets/MTR-001/valuation",
        json={
            "useful_life_years": 0,
            "age_years": 2,
            "salvage_value": 20000,
        },
    )

    assert response.status_code == 422


def test_salvage_above_purchase_rejected(client):
    response = client.post(
        "/api/v1/assets/MTR-001/valuation",
        json={
            "useful_life_years": 10,
            "age_years": 2,
            "salvage_value": 900000,
        },
    )

    assert response.status_code == 422


# ============================================================
# ASSET HISTORY TESTS
# ============================================================

def test_asset_creation_recorded_in_history(client):
    payload = {
        **SAMPLE_ASSETS[0],
        "asset_id": "HST-001",
    }

    client.post("/api/v1/assets", json=payload)

    response = client.get(
        "/api/v1/assets/HST-001/history"
    )

    assert response.status_code == 200
    assert response.json()["items"][0]["action"] == "asset_created"


def test_asset_update_recorded_in_history(client):
    client.patch(
        "/api/v1/assets/MTR-001",
        json={"health_score": 79},
    )

    response = client.get(
        "/api/v1/assets/MTR-001/history"
    )

    actions = [
        item["action"]
        for item in response.json()["items"]
    ]

    assert "asset_updated" in actions


def test_inspection_recorded_in_history(client):
    client.post(
        "/api/v1/assets/MTR-001/inspections",
        json={
            "inspection_type": "Visual",
            "inspector": "Engineer",
            "condition": "Good",
            "passed": True,
        },
    )

    response = client.get(
        "/api/v1/assets/MTR-001/history"
    )

    actions = [
        item["action"]
        for item in response.json()["items"]
    ]

    assert "inspection_created" in actions


def test_unknown_asset_history(client):
    response = client.get(
        "/api/v1/assets/UNKNOWN/history"
    )

    assert response.status_code == 404


# ============================================================
# ASSET UPLOAD TESTS
# ============================================================

def test_upload_asset_pdf(client):
    response = client.post(
        "/api/v1/assets/MTR-001/upload",
        files={
            "file": (
                "inspection.pdf",
                io.BytesIO(b"%PDF-1.4\nDemo inspection"),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 201
    assert response.json()["filename"] == "inspection.pdf"


def test_upload_asset_image(client):
    response = client.post(
        "/api/v1/assets/MTR-001/upload",
        files={
            "file": (
                "asset.png",
                io.BytesIO(b"\x89PNG\r\n\x1a\nDemo"),
                "image/png",
            )
        },
    )

    assert response.status_code == 201


def test_unsupported_upload_type(client):
    response = client.post(
        "/api/v1/assets/MTR-001/upload",
        files={
            "file": (
                "script.exe",
                io.BytesIO(b"EXE"),
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 415


def test_upload_exceeds_size_limit(client):
    oversized = b"A" * (2 * 1024 * 1024 + 1)

    response = client.post(
        "/api/v1/assets/MTR-001/upload",
        files={
            "file": (
                "large.pdf",
                io.BytesIO(oversized),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 413


def test_list_asset_uploads(client):
    client.post(
        "/api/v1/assets/MTR-001/upload",
        files={
            "file": (
                "manual.pdf",
                io.BytesIO(b"%PDF-1.4"),
                "application/pdf",
            )
        },
    )

    response = client.get(
        "/api/v1/assets/MTR-001/uploads"
    )

    assert response.status_code == 200
    assert len(response.json()["items"]) == 1


def test_upload_unknown_asset(client):
    response = client.post(
        "/api/v1/assets/UNKNOWN/upload",
        files={
            "file": (
                "manual.pdf",
                io.BytesIO(b"%PDF-1.4"),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 404


# ============================================================
# DATA ISOLATION AND CONSISTENCY
# ============================================================

def test_asset_update_does_not_change_other_assets(client):
    client.patch(
        "/api/v1/assets/MTR-001",
        json={"health_score": 45},
    )

    response = client.get("/api/v1/assets/PMP-001")

    assert response.json()["health_score"] == 84


def test_component_isolation_between_assets(client):
    client.post(
        "/api/v1/assets/MTR-001/components",
        json={
            "component_id": "BRG-100",
            "name": "Bearing",
            "health_score": 70,
        },
    )

    response = client.get(
        "/api/v1/assets/GEN-001/components"
    )

    assert response.json()["items"] == []


def test_inspection_isolation_between_assets(client):
    client.post(
        "/api/v1/assets/MTR-001/inspections",
        json={
            "inspection_type": "Visual",
            "inspector": "Engineer",
            "condition": "Good",
            "passed": True,
        },
    )

    response = client.get(
        "/api/v1/assets/PMP-001/inspections"
    )

    assert response.json()["items"] == []


def test_asset_health_reflects_updates(client):
    client.patch(
        "/api/v1/assets/MTR-001",
        json={"health_score": 90},
    )

    response = client.get(
        "/api/v1/assets/MTR-001/health"
    )

    assert response.json()["classification"] == "healthy"


def test_history_timestamp_is_valid_iso8601(client):
    client.patch(
        "/api/v1/assets/MTR-001",
        json={"health_score": 75},
    )

    response = client.get(
        "/api/v1/assets/MTR-001/history"
    )

    timestamp = response.json()["items"][0]["timestamp"]

    parsed = datetime.fromisoformat(timestamp)

    assert parsed.tzinfo is not None


# ============================================================
# ADDITIONAL BOUNDARY TESTS
# ============================================================

@pytest.mark.parametrize(
    "score,expected",
    [
        (64, "high"),
        (65, "medium"),
        (79, "medium"),
        (80, "low"),
        (100, "low"),
    ],
)
def test_maintenance_priority_health_boundaries(score, expected):
    item = {
        "health_score": score,
        "annual_failure_probability": 0.10,
    }

    assert maintenance_priority(item) == expected


def test_zero_cost_asset_is_allowed(client):
    payload = {
        **SAMPLE_ASSETS[0],
        "asset_id": "ZERO-001",
        "acquisition_cost": 0,
    }

    response = client.post("/api/v1/assets", json=payload)

    assert response.status_code == 201


def test_full_health_score_allowed(client):
    response = client.patch(
        "/api/v1/assets/MTR-001",
        json={"health_score": 100},
    )

    assert response.status_code == 200


def test_zero_rul_allowed(client):
    response = client.patch(
        "/api/v1/assets/MTR-001",
        json={"remaining_useful_life": 0},
    )

    assert response.status_code == 200

    lifecycle = client.get(
        "/api/v1/assets/MTR-001/lifecycle"
    )

    assert lifecycle.json()["stage"] == "replacement_planning"


def test_empty_asset_filter_result(client):
    response = client.get(
        "/api/v1/assets",
        params={"plant": "Nonexistent Plant"},
    )

    assert response.status_code == 200
    assert response.json()["items"] == []
    assert response.json()["total"] == 0
