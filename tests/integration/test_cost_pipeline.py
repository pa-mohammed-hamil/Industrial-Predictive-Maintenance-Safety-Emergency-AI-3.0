
"""
tests/integration/test_cost_pipeline.py

Integration tests for the Industrial AI cost intelligence pipeline.

Run:
    pytest tests/integration/test_cost_pipeline.py -v

Expected application contract:
    POST /api/v1/cost-analysis
    POST /api/v1/repair-replacement
    POST /api/v1/financial-forecasts

These endpoints and payloads are proposed contracts based on
the uploaded project structure. Adapt them to the actual API
schemas before running against the application.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app


API_PREFIX = "/api/v1"

COST_ENDPOINT = f"{API_PREFIX}/cost-analysis"
DECISION_ENDPOINT = f"{API_PREFIX}/repair-replacement"
FORECAST_ENDPOINT = f"{API_PREFIX}/financial-forecasts"


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def asset_data() -> dict[str, Any]:
    return {
        "asset_id": "PUMP-001",
        "asset_type": "industrial_pump",
        "asset_age_years": 5,
        "purchase_cost": 100000.0,
        "replacement_cost": 120000.0,
        "currency": "INR",
    }


@pytest.fixture
def cost_payload(asset_data: dict[str, Any]) -> dict[str, Any]:
    return {
        **asset_data,
        "repair_cost": 15000.0,
        "spare_parts_cost": 5000.0,
        "labor_cost": 3000.0,
        "downtime_hours": 8.0,
        "downtime_cost_per_hour": 2000.0,
        "production_loss_per_hour": 1500.0,
        "energy_cost": 1000.0,
        "scrap_cost": 500.0,
        "emergency_cost": 2500.0,
        "failure_probability": 0.25,
        "failure_impact_cost": 80000.0,
    }


def post_json(
    client: TestClient,
    endpoint: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    response = client.post(endpoint, json=payload)

    assert response.status_code == 200, (
        f"Endpoint: {endpoint}\n"
        f"Status: {response.status_code}\n"
        f"Response: {response.text}"
    )

    body = response.json()
    assert isinstance(body, dict)
    return body


def assert_valid_amount(value: Any) -> None:
    assert isinstance(value, (int, float))
    assert not isinstance(value, bool)
    assert math.isfinite(value)
    assert value >= 0


class TestCostPipeline:

    def test_cost_pipeline_success(
        self,
        client: TestClient,
        cost_payload: dict[str, Any],
    ) -> None:
        result = post_json(
            client,
            COST_ENDPOINT,
            cost_payload,
        )

        assert result["asset_id"] == "PUMP-001"
        assert "total_cost" in result
        assert_valid_amount(result["total_cost"])

    def test_repair_cost(
        self,
        client: TestClient,
        cost_payload: dict[str, Any],
    ) -> None:
        result = post_json(client, COST_ENDPOINT, cost_payload)

        assert result["repair_cost"] == pytest.approx(15000.0)

    def test_spare_parts_cost(
        self,
        client: TestClient,
        cost_payload: dict[str, Any],
    ) -> None:
        result = post_json(client, COST_ENDPOINT, cost_payload)

        assert result["spare_parts_cost"] == pytest.approx(5000.0)

    def test_labor_cost(
        self,
        client: TestClient,
        cost_payload: dict[str, Any],
    ) -> None:
        result = post_json(client, COST_ENDPOINT, cost_payload)

        assert result["labor_cost"] == pytest.approx(3000.0)

    def test_downtime_cost(
        self,
        client: TestClient,
        cost_payload: dict[str, Any],
    ) -> None:
        result = post_json(client, COST_ENDPOINT, cost_payload)

        expected = 8.0 * 2000.0

        assert result["downtime_cost"] == pytest.approx(expected)

    def test_production_loss(
        self,
        client: TestClient,
        cost_payload: dict[str, Any],
    ) -> None:
        result = post_json(client, COST_ENDPOINT, cost_payload)

        expected = 8.0 * 1500.0

        assert result["production_loss"] == pytest.approx(expected)

    def test_emergency_cost(
        self,
        client: TestClient,
        cost_payload: dict[str, Any],
    ) -> None:
        result = post_json(client, COST_ENDPOINT, cost_payload)

        assert result["emergency_cost"] == pytest.approx(2500.0)

    def test_energy_cost(
        self,
        client: TestClient,
        cost_payload: dict[str, Any],
    ) -> None:
        result = post_json(client, COST_ENDPOINT, cost_payload)

        assert result["energy_cost"] == pytest.approx(1000.0)

    def test_scrap_cost(
        self,
        client: TestClient,
        cost_payload: dict[str, Any],
    ) -> None:
        result = post_json(client, COST_ENDPOINT, cost_payload)

        assert result["scrap_cost"] == pytest.approx(500.0)

    def test_expected_failure_cost(
        self,
        client: TestClient,
        cost_payload: dict[str, Any],
    ) -> None:
        result = post_json(client, COST_ENDPOINT, cost_payload)

        expected = 0.25 * 80000.0

        assert result["expected_failure_cost"] == pytest.approx(
            expected
        )

    def test_total_cost(
        self,
        client: TestClient,
        cost_payload: dict[str, Any],
    ) -> None:
        result = post_json(client, COST_ENDPOINT, cost_payload)

        expected = (
            15000.0
            + 5000.0
            + 3000.0
            + 16000.0
            + 12000.0
            + 1000.0
            + 500.0
            + 2500.0
            + 20000.0
        )

        assert result["total_cost"] == pytest.approx(expected)

    def test_zero_downtime(
        self,
        client: TestClient,
        cost_payload: dict[str, Any],
    ) -> None:
        payload = {
            **cost_payload,
            "downtime_hours": 0.0,
        }

        result = post_json(client, COST_ENDPOINT, payload)

        assert result["downtime_cost"] == 0
        assert result["production_loss"] == 0

    def test_zero_failure_probability(
        self,
        client: TestClient,
        cost_payload: dict[str, Any],
    ) -> None:
        payload = {
            **cost_payload,
            "failure_probability": 0.0,
        }

        result = post_json(client, COST_ENDPOINT, payload)

        assert result["expected_failure_cost"] == 0

    @pytest.mark.parametrize(
        "invalid_probability",
        [-0.1, 1.1, 2.0],
    )
    def test_invalid_failure_probability(
        self,
        client: TestClient,
        cost_payload: dict[str, Any],
        invalid_probability: float,
    ) -> None:
        payload = {
            **cost_payload,
            "failure_probability": invalid_probability,
        }

        response = client.post(
            COST_ENDPOINT,
            json=payload,
        )

        assert response.status_code == 422

    @pytest.mark.parametrize(
        "field",
        [
            "repair_cost",
            "replacement_cost",
            "spare_parts_cost",
            "labor_cost",
            "downtime_hours",
            "energy_cost",
        ],
    )
    def test_negative_cost_rejected(
        self,
        client: TestClient,
        cost_payload: dict[str, Any],
        field: str,
    ) -> None:
        payload = {
            **cost_payload,
            field: -100.0,
        }

        response = client.post(
            COST_ENDPOINT,
            json=payload,
        )

        assert response.status_code == 422


class TestRepairReplacementPipeline:

    @pytest.fixture
    def decision_payload(self) -> dict[str, Any]:
        return {
            "asset_id": "PUMP-001",
            "repair_cost": 20000.0,
            "replacement_cost": 120000.0,
            "repair_downtime_cost": 5000.0,
            "replacement_downtime_cost": 10000.0,
            "repair_expected_future_cost": 10000.0,
            "replacement_expected_future_cost": 3000.0,
        }

    def test_decision_response(
        self,
        client: TestClient,
        decision_payload: dict[str, Any],
    ) -> None:
        result = post_json(
            client,
            DECISION_ENDPOINT,
            decision_payload,
        )

        assert result["asset_id"] == "PUMP-001"
        assert result["recommendation"] in {
            "repair",
            "replace",
        }

    def test_repair_is_cheaper(
        self,
        client: TestClient,
        decision_payload: dict[str, Any],
    ) -> None:
        result = post_json(
            client,
            DECISION_ENDPOINT,
            decision_payload,
        )

        assert result["repair_total_cost"] == pytest.approx(
            35000.0
        )

        assert result["replacement_total_cost"] == pytest.approx(
            133000.0
        )

        assert result["recommendation"] == "repair"

    def test_replacement_is_cheaper(
        self,
        client: TestClient,
        decision_payload: dict[str, Any],
    ) -> None:
        payload = {
            **decision_payload,
            "repair_cost": 150000.0,
            "replacement_cost": 60000.0,
        }

        result = post_json(
            client,
            DECISION_ENDPOINT,
            payload,
        )

        assert result["recommendation"] == "replace"

    def test_equal_cost_decision(
        self,
        client: TestClient,
        decision_payload: dict[str, Any],
    ) -> None:
        payload = {
            **decision_payload,
            "repair_cost": 50000.0,
            "replacement_cost": 50000.0,
            "repair_downtime_cost": 0.0,
            "replacement_downtime_cost": 0.0,
            "repair_expected_future_cost": 0.0,
            "replacement_expected_future_cost": 0.0,
        }

        result = post_json(
            client,
            DECISION_ENDPOINT,
            payload,
        )

        assert result["repair_total_cost"] == pytest.approx(
            result["replacement_total_cost"]
        )

        assert result["recommendation"] in {
            "repair",
            "replace",
        }


class TestFinancialForecastPipeline:

    @pytest.fixture
    def forecast_payload(self) -> dict[str, Any]:
        return {
            "asset_id": "PUMP-001",
            "current_annual_cost": 100000.0,
            "annual_growth_rate": 0.05,
            "forecast_years": 5,
            "currency": "INR",
        }

    def test_forecast_generation(
        self,
        client: TestClient,
        forecast_payload: dict[str, Any],
    ) -> None:
        result = post_json(
            client,
            FORECAST_ENDPOINT,
            forecast_payload,
        )

        assert "forecasts" in result
        assert len(result["forecasts"]) == 5

    def test_forecast_growth(
        self,
        client: TestClient,
        forecast_payload: dict[str, Any],
    ) -> None:
        result = post_json(
            client,
            FORECAST_ENDPOINT,
            forecast_payload,
        )

        forecasts = result["forecasts"]

        for year_index, forecast in enumerate(
            forecasts,
            start=1,
        ):
            expected = 100000.0 * (1.05 ** year_index)

            assert forecast["predicted_cost"] == pytest.approx(
                expected
            )

    def test_zero_growth(
        self,
        client: TestClient,
        forecast_payload: dict[str, Any],
    ) -> None:
        payload = {
            **forecast_payload,
            "annual_growth_rate": 0.0,
        }

        result = post_json(
            client,
            FORECAST_ENDPOINT,
            payload,
        )

        for forecast in result["forecasts"]:
            assert forecast["predicted_cost"] == pytest.approx(
                100000.0
            )

    def test_invalid_forecast_years(
        self,
        client: TestClient,
        forecast_payload: dict[str, Any],
    ) -> None:
        payload = {
            **forecast_payload,
            "forecast_years": -1,
        }

        response = client.post(
            FORECAST_ENDPOINT,
            json=payload,
        )

        assert response.status_code == 422


class TestCostPipelineReliability:

    def test_deterministic_calculation(
        self,
        client: TestClient,
        cost_payload: dict[str, Any],
    ) -> None:
        first = post_json(client, COST_ENDPOINT, cost_payload)
        second = post_json(client, COST_ENDPOINT, cost_payload)

        assert first["total_cost"] == pytest.approx(
            second["total_cost"]
        )

    def test_cost_components_nonnegative(
        self,
        client: TestClient,
        cost_payload: dict[str, Any],
    ) -> None:
        result = post_json(client, COST_ENDPOINT, cost_payload)

        fields = [
            "repair_cost",
            "spare_parts_cost",
            "labor_cost",
            "downtime_cost",
            "production_loss",
            "emergency_cost",
            "expected_failure_cost",
            "total_cost",
        ]

        for field in fields:
            assert_valid_amount(result[field])

    def test_currency_preserved(
        self,
        client: TestClient,
        cost_payload: dict[str, Any],
    ) -> None:
        result = post_json(client, COST_ENDPOINT, cost_payload)

        assert result["currency"] == "INR"

    def test_asset_identity_preserved(
        self,
        client: TestClient,
        cost_payload: dict[str, Any],
    ) -> None:
        result = post_json(client, COST_ENDPOINT, cost_payload)

        assert result["asset_id"] == cost_payload["asset_id"]
