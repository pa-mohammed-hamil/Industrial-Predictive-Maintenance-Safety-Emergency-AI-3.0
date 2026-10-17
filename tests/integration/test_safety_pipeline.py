
"""
tests/integration/test_safety_pipeline.py

Industrial Predictive Maintenance, Safety &
Emergency Intelligence AI

Safety pipeline integration tests.

Run:
    pytest tests/integration/test_safety_pipeline.py -v

NOTE:
These tests define a proposed API contract based on
the project directory structure.

Adjust endpoint paths, request fields and response
schemas to match your implemented backend.

All tests must run in a safe simulation environment.
Never connect test execution to real industrial actuators,
alarms, emergency dispatch, or production PLCs.
"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app


API = "/api/v1"

SAFETY = f"{API}/safety"
FIRE = f"{API}/fire"
SMOKE = f"{API}/smoke"
OVERHEATING = f"{API}/overheating"
MOTION = f"{API}/motion"
INCIDENTS = f"{API}/incidents"
EMERGENCY = f"{API}/emergency"


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sensor_payload() -> dict[str, Any]:
    return {
        "asset_id": "MACHINE-001",
        "site_id": "PLANT-001",
        "zone_id": "ZONE-A",
        "temperature": 45.0,
        "vibration": 2.5,
        "pressure": 4.2,
        "current": 12.0,
        "smoke_confidence": 0.05,
        "fire_confidence": 0.02,
        "motion_detected": False,
        "person_detected": False,
        "restricted_zone": False,
    }


def post_json(
    client: TestClient,
    endpoint: str,
    payload: dict[str, Any],
    expected_status: int = 200,
) -> dict[str, Any]:

    response = client.post(
        endpoint,
        json=payload,
    )

    assert response.status_code == expected_status, (
        f"POST {endpoint}\n"
        f"Expected: {expected_status}\n"
        f"Actual: {response.status_code}\n"
        f"Body: {response.text}"
    )

    result = response.json()
    assert isinstance(result, dict)
    return result


class TestFireDetection:

    def test_fire_detected(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
    ) -> None:

        payload = {
            **sensor_payload,
            "fire_confidence": 0.97,
        }

        result = post_json(client, FIRE, payload)

        assert result["detected"] is True
        assert result["event_type"] == "fire"

    def test_no_fire(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
    ) -> None:

        result = post_json(client, FIRE, sensor_payload)

        assert result["detected"] is False

    @pytest.mark.parametrize(
        "confidence",
        [-0.1, 1.1, 2.0],
    )
    def test_invalid_fire_confidence(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
        confidence: float,
    ) -> None:

        payload = {
            **sensor_payload,
            "fire_confidence": confidence,
        }

        response = client.post(FIRE, json=payload)

        assert response.status_code == 422


class TestSmokeDetection:

    def test_smoke_detected(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
    ) -> None:

        payload = {
            **sensor_payload,
            "smoke_confidence": 0.95,
        }

        result = post_json(client, SMOKE, payload)

        assert result["detected"] is True
        assert result["event_type"] == "smoke"

    def test_no_smoke(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
    ) -> None:

        result = post_json(client, SMOKE, sensor_payload)

        assert result["detected"] is False


class TestOverheatingDetection:

    def test_overheating_detected(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
    ) -> None:

        payload = {
            **sensor_payload,
            "temperature": 115.0,
        }

        result = post_json(client, OVERHEATING, payload)

        assert result["detected"] is True

    def test_normal_temperature(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
    ) -> None:

        result = post_json(
            client,
            OVERHEATING,
            sensor_payload,
        )

        assert result["detected"] is False

    def test_temperature_threshold(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
    ) -> None:

        payload = {
            **sensor_payload,
            "temperature": 90.0,
            "threshold": 85.0,
        }

        result = post_json(
            client,
            OVERHEATING,
            payload,
        )

        assert result["detected"] is True


class TestMotionDetection:

    def test_motion_detected(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
    ) -> None:

        payload = {
            **sensor_payload,
            "motion_detected": True,
        }

        result = post_json(client, MOTION, payload)

        assert result["detected"] is True

    def test_no_motion(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
    ) -> None:

        result = post_json(client, MOTION, sensor_payload)

        assert result["detected"] is False


class TestWorkerSafety:

    def test_restricted_zone_violation(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
    ) -> None:

        payload = {
            **sensor_payload,
            "person_detected": True,
            "restricted_zone": True,
        }

        result = post_json(
            client,
            f"{SAFETY}/restricted-zone",
            payload,
        )

        assert result["violation"] is True

    def test_no_restricted_zone_violation(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
    ) -> None:

        payload = {
            **sensor_payload,
            "person_detected": False,
            "restricted_zone": True,
        }

        result = post_json(
            client,
            f"{SAFETY}/restricted-zone",
            payload,
        )

        assert result["violation"] is False

    def test_unsafe_proximity(
        self,
        client: TestClient,
    ) -> None:

        payload = {
            "asset_id": "MACHINE-001",
            "person_distance_m": 0.5,
            "minimum_safe_distance_m": 2.0,
        }

        result = post_json(
            client,
            f"{SAFETY}/unsafe-proximity",
            payload,
        )

        assert result["unsafe"] is True

    def test_safe_proximity(
        self,
        client: TestClient,
    ) -> None:

        payload = {
            "asset_id": "MACHINE-001",
            "person_distance_m": 5.0,
            "minimum_safe_distance_m": 2.0,
        }

        result = post_json(
            client,
            f"{SAFETY}/unsafe-proximity",
            payload,
        )

        assert result["unsafe"] is False


class TestSafetyRisk:

    def test_normal_risk_score(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
    ) -> None:

        result = post_json(
            client,
            f"{SAFETY}/risk",
            sensor_payload,
        )

        assert 0 <= result["risk_score"] <= 1

    def test_high_risk_conditions(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
    ) -> None:

        payload = {
            **sensor_payload,
            "temperature": 120.0,
            "vibration": 15.0,
            "fire_confidence": 0.98,
            "smoke_confidence": 0.95,
        }

        result = post_json(
            client,
            f"{SAFETY}/risk",
            payload,
        )

        assert result["risk_score"] >= 0.7

    def test_risk_increases_with_hazards(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
    ) -> None:

        normal = post_json(
            client,
            f"{SAFETY}/risk",
            sensor_payload,
        )

        dangerous = {
            **sensor_payload,
            "temperature": 120.0,
            "fire_confidence": 0.99,
            "smoke_confidence": 0.95,
        }

        high = post_json(
            client,
            f"{SAFETY}/risk",
            dangerous,
        )

        assert high["risk_score"] >= normal["risk_score"]


class TestMultiSensorFusion:

    def test_fire_smoke_temperature_fusion(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
    ) -> None:

        payload = {
            **sensor_payload,
            "temperature": 125.0,
            "fire_confidence": 0.97,
            "smoke_confidence": 0.94,
        }

        result = post_json(
            client,
            f"{SAFETY}/correlate",
            payload,
        )

        assert result["hazard_detected"] is True
        assert "risk_score" in result

    def test_normal_sensor_fusion(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
    ) -> None:

        result = post_json(
            client,
            f"{SAFETY}/correlate",
            sensor_payload,
        )

        assert result["hazard_detected"] is False

    def test_multisignal_severity(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
    ) -> None:

        payload = {
            **sensor_payload,
            "temperature": 130.0,
            "fire_confidence": 0.99,
            "smoke_confidence": 0.98,
        }

        result = post_json(
            client,
            f"{SAFETY}/correlate",
            payload,
        )

        assert result["severity"] in {
            "high",
            "critical",
        }


class TestIncidentIntegration:

    def test_safety_event_creates_incident(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
    ) -> None:

        payload = {
            **sensor_payload,
            "temperature": 130.0,
            "fire_confidence": 0.99,
            "smoke_confidence": 0.96,
        }

        result = post_json(
            client,
            f"{SAFETY}/process",
            payload,
        )

        assert result["incident_created"] is True
        assert result["incident_id"]

        response = client.get(
            f"{INCIDENTS}/{result['incident_id']}"
        )

        assert response.status_code == 200

        incident = response.json()

        assert incident["asset_id"] == "MACHINE-001"

    def test_normal_conditions_no_incident(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
    ) -> None:

        result = post_json(
            client,
            f"{SAFETY}/process",
            sensor_payload,
        )

        assert result["incident_created"] is False


class TestEmergencyEscalation:

    def test_critical_event_escalation(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
    ) -> None:

        payload = {
            **sensor_payload,
            "temperature": 140.0,
            "fire_confidence": 0.99,
            "smoke_confidence": 0.99,
        }

        result = post_json(
            client,
            f"{SAFETY}/process",
            payload,
        )

        assert result["severity"] == "critical"
        assert result["incident_id"]

        escalation = post_json(
            client,
            f"{EMERGENCY}/escalate",
            {
                "incident_id": result["incident_id"],
                "escalation_level": "critical",
            },
        )

        assert escalation["escalated"] is True

    def test_normal_event_no_escalation(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
    ) -> None:

        result = post_json(
            client,
            f"{SAFETY}/process",
            sensor_payload,
        )

        assert result.get("escalated", False) is False


class TestSafetyValidation:

    @pytest.mark.parametrize(
        "field",
        [
            "asset_id",
            "site_id",
            "zone_id",
        ],
    )
    def test_missing_required_fields(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
        field: str,
    ) -> None:

        payload = deepcopy(sensor_payload)
        payload.pop(field)

        response = client.post(
            f"{SAFETY}/process",
            json=payload,
        )

        assert response.status_code == 422

    @pytest.mark.parametrize(
        "field",
        [
            "fire_confidence",
            "smoke_confidence",
        ],
    )
    def test_invalid_confidence(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
        field: str,
    ) -> None:

        payload = {
            **sensor_payload,
            field: 1.5,
        }

        response = client.post(
            f"{SAFETY}/process",
            json=payload,
        )

        assert response.status_code == 422

    def test_invalid_sensor_type(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
    ) -> None:

        payload = {
            **sensor_payload,
            "temperature": "not-a-number",
        }

        response = client.post(
            f"{SAFETY}/process",
            json=payload,
        )

        assert response.status_code == 422


class TestSafetyReliability:

    def test_risk_calculation_deterministic(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
    ) -> None:

        first = post_json(
            client,
            f"{SAFETY}/risk",
            sensor_payload,
        )

        second = post_json(
            client,
            f"{SAFETY}/risk",
            sensor_payload,
        )

        assert first["risk_score"] == pytest.approx(
            second["risk_score"]
        )

    def test_asset_identity_preserved(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
    ) -> None:

        result = post_json(
            client,
            f"{SAFETY}/process",
            sensor_payload,
        )

        assert result["asset_id"] == "MACHINE-001"

    def test_safety_response_structure(
        self,
        client: TestClient,
        sensor_payload: dict[str, Any],
    ) -> None:

        result = post_json(
            client,
            f"{SAFETY}/process",
            sensor_payload,
        )

        assert "risk_score" in result
        assert "severity" in result
        assert "incident_created" in result

        assert 0 <= result["risk_score"] <= 1
